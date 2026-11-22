//! Background jobs for the app. The window never waits: it submits work and
//! picks up the newest finished result on its next frame. A newer request
//! cancels older work between Newton steps.

use crate::project::{DieOutput, MeshQuality, Project, run_shell, solve_die};
use arc_swap::ArcSwapOption;
use pasta_shell::{ShellResult, Snapshot};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct DieJob {
    pub project: Project,
    pub quality: MeshQuality,
}

#[derive(Clone)]
pub struct ShellJob {
    pub project: Project,
    pub die: Arc<DieOutput>,
}

/// Latest progress of the running shell job.
#[derive(Clone)]
pub struct ShellProgress {
    pub generation: u64,
    pub snapshot: Snapshot,
}

/// Results published by the workers.
#[derive(Default)]
pub struct Outputs {
    pub die: ArcSwapOption<(u64, DieOutput)>,
    pub die_error: ArcSwapOption<(u64, String)>,
    pub shell_progress: ArcSwapOption<ShellProgress>,
    pub shell: ArcSwapOption<(u64, ShellResult)>,
    pub shell_error: ArcSwapOption<(u64, String)>,
}

pub struct Engine {
    die_gen: Arc<AtomicU64>,
    shell_gen: Arc<AtomicU64>,
    die_tx: Sender<(u64, DieJob)>,
    shell_tx: Sender<(u64, ShellJob)>,
    pub outputs: Arc<Outputs>,
    /// Called whenever a worker publishes something (e.g. to wake the UI).
    notify: Arc<Mutex<Option<Box<dyn Fn() + Send>>>>,
}

impl Engine {
    pub fn new(shell_threads: usize) -> Self {
        let die_gen = Arc::new(AtomicU64::new(0));
        let shell_gen = Arc::new(AtomicU64::new(0));
        let outputs = Arc::new(Outputs::default());
        let notify: Arc<Mutex<Option<Box<dyn Fn() + Send>>>> = Arc::new(Mutex::new(None));
        let (die_tx, die_rx) = channel::<(u64, DieJob)>();
        let (shell_tx, shell_rx) = channel::<(u64, ShellJob)>();
        {
            let (g, out, note) = (die_gen.clone(), outputs.clone(), notify.clone());
            std::thread::Builder::new()
                .name("die-flow".into())
                .spawn(move || die_worker(die_rx, g, out, note))
                .expect("spawn die worker");
        }
        {
            let (g, out, note) = (shell_gen.clone(), outputs.clone(), notify.clone());
            std::thread::Builder::new()
                .name("shell".into())
                .spawn(move || shell_worker(shell_rx, g, out, note, shell_threads))
                .expect("spawn shell worker");
        }
        Self { die_gen, shell_gen, die_tx, shell_tx, outputs, notify }
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

    pub fn submit_shell(&self, job: ShellJob) -> u64 {
        let g = self.shell_gen.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.shell_tx.send((g, job));
        g
    }

    /// Cancels the running shell job, if any.
    pub fn cancel_shell(&self) {
        self.shell_gen.fetch_add(1, Ordering::SeqCst);
    }

    pub fn shell_generation(&self) -> u64 {
        self.shell_gen.load(Ordering::SeqCst)
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

fn shell_worker(
    rx: Receiver<(u64, ShellJob)>,
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
        job.project.shell.threads = threads;
        let cancel = || generation.load(Ordering::SeqCst) != g;
        let mut progress = |s: &Snapshot| {
            out.shell_progress.store(Some(Arc::new(ShellProgress { generation: g, snapshot: s.clone() })));
            ping(&note);
        };
        match run_shell(&job.project, &job.die, &mut progress, &cancel) {
            Ok(r) => {
                out.shell.store(Some(Arc::new((g, r))));
                out.shell_error.store(None);
            }
            Err(e) if e == "cancelled" => continue,
            Err(e) => out.shell_error.store(Some(Arc::new((g, e)))),
        }
        ping(&note);
    }
}
