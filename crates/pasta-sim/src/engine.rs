//! Background jobs for the app. The window never waits: it submits work and
//! picks up the newest finished result on its next frame. A newer request
//! cancels older work between Newton steps.

use crate::project::{DieOutput, MeshQuality, Project, run_solid, solve_die};
use arc_swap::ArcSwapOption;
use pasta_solid::{Snapshot, SolidResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct DieJob {
    pub project: Project,
    pub quality: MeshQuality,
}

/// Grow the pasta from a finished die solve.
#[derive(Clone)]
pub struct GrowJob {
    pub project: Project,
    pub die: Arc<DieOutput>,
}

/// Latest progress of the running grow job.
#[derive(Clone)]
pub struct GrowProgress {
    pub generation: u64,
    pub snapshot: Snapshot,
}

/// Results published by the workers.
#[derive(Default)]
pub struct Outputs {
    pub die: ArcSwapOption<(u64, DieOutput)>,
    pub die_error: ArcSwapOption<(u64, String)>,
    pub grow_progress: ArcSwapOption<GrowProgress>,
    pub piece: ArcSwapOption<(u64, SolidResult)>,
    pub grow_error: ArcSwapOption<(u64, String)>,
}

pub struct Engine {
    die_gen: Arc<AtomicU64>,
    grow_gen: Arc<AtomicU64>,
    die_tx: Sender<(u64, DieJob)>,
    grow_tx: Sender<(u64, GrowJob)>,
    pub outputs: Arc<Outputs>,
    /// Called whenever a worker publishes something (e.g. to wake the UI).
    notify: Arc<Mutex<Option<Box<dyn Fn() + Send>>>>,
}

impl Engine {
    pub fn new(grow_threads: usize) -> Self {
        let die_gen = Arc::new(AtomicU64::new(0));
        let grow_gen = Arc::new(AtomicU64::new(0));
        let outputs = Arc::new(Outputs::default());
        let notify: Arc<Mutex<Option<Box<dyn Fn() + Send>>>> = Arc::new(Mutex::new(None));
        let (die_tx, die_rx) = channel::<(u64, DieJob)>();
        let (grow_tx, grow_rx) = channel::<(u64, GrowJob)>();
        {
            let (g, out, note) = (die_gen.clone(), outputs.clone(), notify.clone());
            std::thread::Builder::new()
                .name("die-flow".into())
                .spawn(move || die_worker(die_rx, g, out, note))
                .expect("spawn die worker");
        }
        {
            let (g, out, note) = (grow_gen.clone(), outputs.clone(), notify.clone());
            std::thread::Builder::new()
                .name("grow".into())
                .spawn(move || grow_worker(grow_rx, g, out, note, grow_threads))
                .expect("spawn grow worker");
        }
        Self { die_gen, grow_gen, die_tx, grow_tx, outputs, notify }
    }

    pub fn set_notify(&self, f: impl Fn() + Send + 'static) {
        *self.notify.lock().unwrap() = Some(Box::new(f));
    }

    /// Submits a die solve; returns its generation.
    pub fn submit_die(&self, job: DieJob) -> u64 {
        let g = self.die_gen.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.die_tx.send((g, job));
        g
    }

    pub fn submit_grow(&self, job: GrowJob) -> u64 {
        let g = self.grow_gen.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.grow_tx.send((g, job));
        g
    }

    /// Cancels the running grow job, if any.
    pub fn cancel_grow(&self) {
        self.grow_gen.fetch_add(1, Ordering::SeqCst);
    }

    pub fn grow_generation(&self) -> u64 {
        self.grow_gen.load(Ordering::SeqCst)
    }

    pub fn die_generation(&self) -> u64 {
        self.die_gen.load(Ordering::SeqCst)
    }
}

fn ping(note: &Mutex<Option<Box<dyn Fn() + Send>>>) {
    if let Some(f) = note.lock().unwrap().as_ref() {
        f();
    }
}

fn die_worker(
    rx: Receiver<(u64, DieJob)>,
    generation: Arc<AtomicU64>,
    out: Arc<Outputs>,
    note: Arc<Mutex<Option<Box<dyn Fn() + Send>>>>,
) {
    let mut last: Option<DieOutput> = None;
    while let Ok(mut msg) = rx.recv() {
        // Coalesce: only the newest request matters.
        while let Ok(newer) = rx.try_recv() {
            msg = newer;
        }
        let (g, job) = msg;
        let cancel = || generation.load(Ordering::SeqCst) != g;
        match solve_die(&job.project, job.quality, last.as_ref(), 2, &cancel) {
            Ok(o) => {
                last = Some(o.clone());
                out.die.store(Some(Arc::new((g, o))));
                out.die_error.store(None);
            }
            Err(e) if e == "cancelled" => continue,
            Err(e) => out.die_error.store(Some(Arc::new((g, e)))),
        }
        ping(&note);
    }
}

fn grow_worker(
    rx: Receiver<(u64, GrowJob)>,
    generation: Arc<AtomicU64>,
    out: Arc<Outputs>,
    note: Arc<Mutex<Option<Box<dyn Fn() + Send>>>>,
    threads: usize,
) {
    while let Ok(mut msg) = rx.recv() {
        while let Ok(newer) = rx.try_recv() {
            msg = newer;
        }
        let (g, mut job) = msg;
        if generation.load(Ordering::SeqCst) != g {
            continue;
        }
        job.project.solid.threads = threads;
        let cancel = || generation.load(Ordering::SeqCst) != g;
        let mut progress = |s: &Snapshot| {
            out.grow_progress.store(Some(Arc::new(GrowProgress { generation: g, snapshot: s.clone() })));
            ping(&note);
        };
        match run_solid(&job.project, &job.die, &mut progress, &cancel) {
            Ok(r) => {
                out.piece.store(Some(Arc::new((g, r))));
                out.grow_error.store(None);
            }
            Err(e) if e == "cancelled" => continue,
            Err(e) => out.grow_error.store(Some(Arc::new((g, e)))),
        }
        ping(&note);
    }
}
