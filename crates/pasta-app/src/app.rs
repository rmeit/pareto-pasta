//! Application state, layout and the link to the background engine.

use crate::die_canvas::DieCanvas;
use crate::viewport::{ColorMap, Overlay, SceneMesh, ViewPreset, ViewSettings, Viewport, save_png};
use pasta_sim::shell::{Orientation, ShellResult, measure};
use pasta_sim::{DieJob, DieOutput, Engine, MeshQuality, Project, RunReport, ShellJob, ShellProgress, write_obj, write_stl};
use std::path::PathBuf;
use std::sync::Arc;

/// What colours the pasta.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorMode {
    Plain,
    ExitSpeed,
    Age,
    Thickness,
}

impl ColorMode {
    pub const ALL: [ColorMode; 4] = [ColorMode::Plain, ColorMode::ExitSpeed, ColorMode::Age, ColorMode::Thickness];

    pub fn label(self) -> &'static str {
        match self {
            ColorMode::Plain => "Dough",
            ColorMode::ExitSpeed => "Exit speed",
            ColorMode::Age => "Time since exit",
            ColorMode::Thickness => "Thickness",
        }
    }
}

/// Which shape the 3D view shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShapeView {
    /// The keyframe chosen on the timeline.
    Timeline,
    Hanging,
    Relaxed,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Command {
    Grow,
    Stop,
    Undo,
    Redo,
    Open,
    Save,
    SaveAs,
    ExportObj,
    ExportStl,
    ExportSvg,
    ExportDxf,
    ExportReport,
    Screenshot,
    FitView,
    ViewFront,
    ViewSide,
    ViewTop,
    ViewIso,
    PinGhost,
    ClearGhost,
    ToggleWireframe,
}

impl Command {
    pub const ALL: [Command; 21] = [
        Command::Grow,
        Command::Stop,
        Command::Undo,
        Command::Redo,
        Command::Open,
        Command::Save,
        Command::SaveAs,
        Command::ExportObj,
        Command::ExportStl,
        Command::ExportSvg,
        Command::ExportDxf,
        Command::ExportReport,
        Command::Screenshot,
        Command::FitView,
        Command::ViewFront,
        Command::ViewSide,
        Command::ViewTop,
        Command::ViewIso,
        Command::PinGhost,
        Command::ClearGhost,
        Command::ToggleWireframe,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Command::Grow => "Grow pasta",
            Command::Stop => "Stop growing",
            Command::Undo => "Undo",
            Command::Redo => "Redo",
            Command::Open => "Open project…",
            Command::Save => "Save project",
            Command::SaveAs => "Save project as…",
            Command::ExportObj => "Export pasta as OBJ…",
            Command::ExportStl => "Export pasta as STL…",
            Command::ExportSvg => "Export die drawing as SVG…",
            Command::ExportDxf => "Export die drawing as DXF…",
            Command::ExportReport => "Export report as JSON…",
            Command::Screenshot => "Save 3D view as PNG…",
            Command::FitView => "Fit view",
            Command::ViewFront => "View from front",
            Command::ViewSide => "View from side",
            Command::ViewTop => "View from top",
            Command::ViewIso => "View at an angle",
            Command::PinGhost => "Pin result for comparison",
            Command::ClearGhost => "Clear comparison",
            Command::ToggleWireframe => "Toggle wireframe",
        }
    }

    pub fn shortcut(self) -> &'static str {
        match self {
            Command::Grow => "G",
            Command::Undo => "Ctrl+Z",
            Command::Redo => "Ctrl+Shift+Z",
            Command::Open => "Ctrl+O",
            Command::Save => "Ctrl+S",
            Command::FitView => "F",
            Command::ToggleWireframe => "W",
            _ => "",
        }
    }
}

/// A pinned result to compare against.
pub struct Ghost {
    pub mesh: SceneMesh,
    pub profile: Option<Vec<[f64; 2]>>,
}

pub struct App {
    pub engine: Engine,
    pub project: Project,
    pub path: Option<PathBuf>,
    history: Vec<Project>,
    future: Vec<Project>,
    edit_start: Option<Project>,
    /// Last project sent to the die worker, and with which quality.
    submitted: Option<(Project, MeshQuality)>,
    pub die: Option<Arc<(u64, DieOutput)>>,
    pub die_error: Option<String>,
    fine_gen: u64,
    pub shell_gen: u64,
    pub shell_progress: Option<Arc<ShellProgress>>,
    pub shell: Option<Arc<(u64, ShellResult)>>,
    pub shell_error: Option<String>,
    pub auto_grow: bool,
    pub viewport: Option<Viewport>,
    pub view: ViewSettings,
    pub color_mode: ColorMode,
    pub shape_view: ShapeView,
    /// Keyframe shown when on the timeline; None = the latest.
    pub timeline: Option<usize>,
    pub playing: bool,
    last_play: f64,
    shown: Option<(u64, usize, ShapeView, ColorMode, usize)>,
    overlay_for: Option<(String, Orientation)>,
    pub ghost: Option<Ghost>,
    pub canvas: DieCanvas,
    pub palette_open: bool,
    pub palette_query: String,
    pub status: String,
    pub pick_info: Option<String>,
    fit_pending: bool,
    /// Rows at the last automatic refit during growth.
    fit_rows: usize,
    /// PASTA_SCREENSHOT=<path>: save a window screenshot once the pasta has
    /// grown, then quit (used to check the UI without a person watching).
    screenshot: Option<(PathBuf, u32, Arc<std::sync::atomic::AtomicBool>)>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>) -> Self {
        let engine = Engine::new(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8));
        let ctx = cc.egui_ctx.clone();
        engine.set_notify(move || ctx.request_repaint());
        style(&cc.egui_ctx);
        let viewport = cc.wgpu_render_state.as_ref().map(Viewport::new);
        let (project, path, status) = match path {
            Some(p) => match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|s| Project::from_ron(&s)) {
                Ok(proj) => (proj, Some(p), String::new()),
                Err(e) => (Project::from_preset("riccia").unwrap(), None, format!("Could not open {}: {e}", p.display())),
            },
            None => (Project::from_preset("riccia").unwrap(), None, String::new()),
        };
        let mut app = Self {
            engine,
            project,
            path,
            history: Vec::new(),
            future: Vec::new(),
            edit_start: None,
            submitted: None,
            die: None,
            die_error: None,
            fine_gen: 0,
            shell_gen: 0,
            shell_progress: None,
            shell: None,
            shell_error: None,
            auto_grow: true,
            viewport,
            view: ViewSettings::default(),
            color_mode: ColorMode::Plain,
            shape_view: ShapeView::Timeline,
            timeline: None,
            playing: false,
            last_play: 0.0,
            shown: None,
            overlay_for: None,
            ghost: None,
            canvas: DieCanvas::default(),
            palette_open: false,
            palette_query: String::new(),
            status,
            pick_info: None,
            fit_pending: true,
            fit_rows: 0,
            screenshot: std::env::var_os("PASTA_SCREENSHOT").map(|p| (PathBuf::from(p), 0, Arc::new(std::sync::atomic::AtomicBool::new(false)))),
        };
        app.submit_die(MeshQuality::Fine);
        app
    }

    fn submit_die(&mut self, quality: MeshQuality) {
        let g = self.engine.submit_die(DieJob { project: self.project.clone(), quality });
        if quality == MeshQuality::Fine {
            self.fine_gen = g;
        } else {
            // A draft solve means the die is changing: stop growing.
            self.engine.cancel_shell();
        }
        self.submitted = Some((self.project.clone(), quality));
    }

    pub fn grow(&mut self) {
        let Some(d) = &self.die else { return };
        if d.0 != self.fine_gen {
            return;
        }
        let die = Arc::new(d.1.clone());
        self.shell_gen = self.engine.submit_shell(ShellJob { project: self.project.clone(), die });
        self.shell_error = None;
        self.timeline = None;
        self.shape_view = ShapeView::Timeline;
        self.fit_pending = true;
    }

    pub fn stop(&mut self) {
        self.engine.cancel_shell();
        self.shell_gen = 0;
    }

    pub fn shell_running(&self) -> bool {
        self.shell_gen != 0 && self.shell.as_ref().map(|s| s.0 != self.shell_gen).unwrap_or(true) && self.shell_error.is_none()
    }

    pub fn undo(&mut self) {
        if let Some(p) = self.history.pop() {
            self.future.push(std::mem::replace(&mut self.project, p));
            self.submit_die(MeshQuality::Fine);
        }
    }

    pub fn redo(&mut self) {
        if let Some(p) = self.future.pop() {
            self.history.push(std::mem::replace(&mut self.project, p));
            self.submit_die(MeshQuality::Fine);
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.history.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }

    /// Replaces the whole project (preset or file), as one undoable edit.
    pub fn load_project(&mut self, p: Project, path: Option<PathBuf>) {
        self.history.push(std::mem::replace(&mut self.project, p));
        self.future.clear();
        self.path = path;
        self.canvas = DieCanvas::default();
        self.stop();
        self.shell = None;
        self.shell_progress = None;
        self.submit_die(MeshQuality::Fine);
        self.fit_pending = true;
    }

    /// Picks up finished work from the engine.
    fn poll_engine(&mut self) {
        let out = self.engine.outputs.clone();
        if let Some(d) = out.die.load_full()
            && self.die.as_ref().map(|x| x.0) != Some(d.0) {
                let fine_done = d.0 == self.fine_gen && d.1.quality == MeshQuality::Fine;
                self.die = Some(d);
                self.die_error = None;
                if fine_done && self.auto_grow {
                    self.grow();
                }
            }
        if let Some(e) = out.die_error.load_full()
            && e.0 >= self.die.as_ref().map(|d| d.0).unwrap_or(0) {
                self.die_error = Some(e.1.clone());
            }
        if let Some(p) = out.shell_progress.load_full()
            && p.generation == self.shell_gen {
                self.shell_progress = Some(p);
            }
        if let Some(s) = out.shell.load_full()
            && s.0 == self.shell_gen && self.shell.as_ref().map(|x| x.0) != Some(s.0) {
                self.shell = Some(s);
                self.shape_view = ShapeView::Hanging;
                self.fit_pending = true;
            }
        if let Some(e) = out.shell_error.load_full()
            && e.0 == self.shell_gen {
                self.shell_error = Some(e.1.clone());
            }
    }

    /// Tracks edits: drafts while the pointer is held, a fine solve and an
    /// undo step when it is released.
    fn track_edits(&mut self, ctx: &egui::Context, before: &Project) {
        let held = ctx.input(|i| i.pointer.any_down());
        if self.project != *before {
            if self.edit_start.is_none() {
                self.edit_start = Some(before.clone());
            }
            if held {
                self.submit_die(MeshQuality::Draft);
            }
        }
        if !held {
            if let Some(start) = self.edit_start.take()
                && start != self.project {
                    self.history.push(start);
                    self.future.clear();
                }
            let up_to_date = matches!(&self.submitted, Some((p, q)) if *p == self.project && *q == MeshQuality::Fine);
            if !up_to_date {
                self.submit_die(MeshQuality::Fine);
            }
        }
    }

    pub fn run_command(&mut self, cmd: Command, rs: Option<&eframe::egui_wgpu::RenderState>) {
        let result: Result<(), String> = (|| {
            match cmd {
                Command::Grow => self.grow(),
                Command::Stop => self.stop(),
                Command::Undo => self.undo(),
                Command::Redo => self.redo(),
                Command::Open => {
                    if let Some(p) = rfd::FileDialog::new().add_filter("pasta project", &["ron"]).pick_file() {
                        let s = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
                        let proj = Project::from_ron(&s)?;
                        self.load_project(proj, Some(p));
                    }
                }
                Command::Save | Command::SaveAs => {
                    let path = match (&self.path, cmd) {
                        (Some(p), Command::Save) => Some(p.clone()),
                        _ => rfd::FileDialog::new().add_filter("pasta project", &["ron"]).set_file_name(format!("{}.ron", self.project.die.name)).save_file(),
                    };
                    if let Some(p) = path {
                        std::fs::write(&p, self.project.to_ron()).map_err(|e| e.to_string())?;
                        self.status = format!("Saved {}", p.display());
                        self.path = Some(p);
                    }
                }
                Command::ExportObj | Command::ExportStl => {
                    let shell = self.shell.as_ref().ok_or("Grow the pasta first")?;
                    let r = &shell.1;
                    let pos = match self.shape_view {
                        ShapeView::Relaxed => r.relaxed.as_ref().unwrap_or(&r.hanging),
                        _ => &r.hanging,
                    };
                    let ext = if cmd == Command::ExportObj { "obj" } else { "stl" };
                    if let Some(p) = rfd::FileDialog::new().add_filter(ext, &[ext]).set_file_name(format!("{}.{ext}", self.project.die.name)).save_file() {
                        let tris = r.outside_triangles();
                        if cmd == Command::ExportObj { write_obj(&p, pos, &tris) } else { write_stl(&p, pos, &tris) }.map_err(|e| e.to_string())?;
                        self.status = format!("Wrote {}", p.display());
                    }
                }
                Command::ExportSvg | Command::ExportDxf => {
                    let ext = if cmd == Command::ExportSvg { "svg" } else { "dxf" };
                    if let Some(p) = rfd::FileDialog::new().add_filter(ext, &[ext]).set_file_name(format!("{}.{ext}", self.project.die.name)).save_file() {
                        let text = if ext == "svg" { pasta_sim::die::to_svg(&self.project.die) } else { pasta_sim::die::to_dxf(&self.project.die) };
                        std::fs::write(&p, text).map_err(|e| e.to_string())?;
                        self.status = format!("Wrote {}", p.display());
                    }
                }
                Command::ExportReport => {
                    let die = self.die.as_ref().ok_or("No die result yet")?;
                    let report = RunReport::new(&self.project, &die.1, self.shell.as_ref().map(|s| &s.1));
                    if let Some(p) = rfd::FileDialog::new().add_filter("json", &["json"]).set_file_name("report.json").save_file() {
                        std::fs::write(&p, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
                        self.status = format!("Wrote {}", p.display());
                    }
                }
                Command::Screenshot => {
                    let (Some(vp), Some(rs)) = (self.viewport.as_mut(), rs) else { return Err("No 3D view".into()) };
                    if let Some(p) = rfd::FileDialog::new().add_filter("png", &["png"]).set_file_name("pasta.png").save_file() {
                        let (w, h) = (1920, 1200);
                        let px = vp.render_image(rs, &self.view, w, h);
                        save_png(&p, w, h, &px).map_err(|e| e.to_string())?;
                        self.status = format!("Wrote {}", p.display());
                    }
                }
                Command::FitView => {
                    if let Some(vp) = self.viewport.as_mut() {
                        vp.fit_to_view();
                    }
                }
                Command::ViewFront | Command::ViewSide | Command::ViewTop | Command::ViewIso => {
                    if let Some(vp) = self.viewport.as_mut() {
                        vp.set_view(match cmd {
                            Command::ViewFront => ViewPreset::Front,
                            Command::ViewSide => ViewPreset::Side,
                            Command::ViewTop => ViewPreset::Top,
                            _ => ViewPreset::Iso,
                        });
                    }
                }
                Command::PinGhost => {
                    let mesh = self.current_mesh().ok_or("Nothing to pin yet")?;
                    let profile = self.die.as_ref().and_then(|d| d.1.profile(self.project.opening)).map(|p| p.s.iter().zip(&p.v).map(|(s, v)| [*s, *v]).collect());
                    self.ghost = Some(Ghost { mesh: SceneMesh { scalars: None, ..mesh }, profile });
                    self.overlay_for = None;
                }
                Command::ClearGhost => {
                    self.ghost = None;
                    self.overlay_for = None;
                }
                Command::ToggleWireframe => self.view.wireframe = !self.view.wireframe,
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.status = e;
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) -> Option<Command> {
        use egui::{Key, Modifiers};
        let typing = ctx.egui_wants_keyboard_input();
        ctx.input_mut(|i| {
            if i.consume_key(Modifiers::COMMAND, Key::K) {
                self.palette_open = !self.palette_open;
                self.palette_query.clear();
                return None;
            }
            if i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z) || i.consume_key(Modifiers::COMMAND, Key::Y) {
                return Some(Command::Redo);
            }
            if i.consume_key(Modifiers::COMMAND, Key::Z) {
                return Some(Command::Undo);
            }
            if i.consume_key(Modifiers::COMMAND, Key::S) {
                return Some(Command::Save);
            }
            if i.consume_key(Modifiers::COMMAND, Key::O) {
                return Some(Command::Open);
            }
            if typing {
                return None;
            }
            if i.consume_key(Modifiers::NONE, Key::G) {
                return Some(Command::Grow);
            }
            if i.consume_key(Modifiers::NONE, Key::F) {
                return Some(Command::FitView);
            }
            if i.consume_key(Modifiers::NONE, Key::W) {
                return Some(Command::ToggleWireframe);
            }
            if i.consume_key(Modifiers::NONE, Key::Space) {
                self.playing = !self.playing;
            }
            None
        })
    }

    /// The pasta mesh currently selected for display.
    fn current_mesh(&self) -> Option<SceneMesh> {
        let r = &self.shell.as_ref()?.1;
        let pos = match self.shape_view {
            ShapeView::Hanging => r.hanging.clone(),
            ShapeView::Relaxed => r.relaxed.clone().unwrap_or_else(|| r.hanging.clone()),
            ShapeView::Timeline => {
                let k = self.timeline.unwrap_or(r.keyframes.len().saturating_sub(1)).min(r.keyframes.len().saturating_sub(1));
                r.keyframes.get(k)?.positions.as_ref().clone()
            }
        };
        Some(SceneMesh { positions: pos, indices: Vec::new(), scalars: None })
    }

    /// Builds the 3D mesh from the newest progress or result.
    fn update_viewport_mesh(&mut self) {
        // Source: the finished result, else the latest progress keyframe.
        let (key, positions, tri_count, stations): (u64, Arc<Vec<[f32; 3]>>, usize, usize);
        let triangles: Option<&Vec<[u32; 3]>>;
        let progress_tris;
        let result = self.shell.as_ref().filter(|s| s.0 == self.shell_gen || !self.shell_running());
        let kf_index;
        if let Some(s) = result {
            let r = &s.1;
            key = s.0;
            stations = r.stations;
            kf_index = self.timeline.unwrap_or(r.keyframes.len().saturating_sub(1)).min(r.keyframes.len().saturating_sub(1));
            let (pos, count) = match self.shape_view {
                ShapeView::Hanging => (Arc::new(r.hanging.clone()), r.triangles.len()),
                ShapeView::Relaxed => (Arc::new(r.relaxed.clone().unwrap_or_else(|| r.hanging.clone())), r.triangles.len()),
                ShapeView::Timeline => {
                    let kf = &r.keyframes[kf_index];
                    (kf.positions.clone(), kf.triangles)
                }
            };
            positions = pos;
            tri_count = count;
            triangles = Some(&r.triangles);
        } else if let Some(p) = &self.shell_progress {
            key = p.generation;
            kf_index = p.snapshot.stats.rows;
            positions = p.snapshot.keyframe.positions.clone();
            tri_count = p.snapshot.keyframe.triangles;
            stations = self.project.shell.stations.max(4);
            // Rebuild the structured triangulation for the progress frame.
            let rows = positions.len() / stations;
            let closed = self.die.as_ref().and_then(|d| d.1.profile(self.project.opening)).map(|p| p.closed).unwrap_or(false);
            progress_tris = strip_triangles(rows, stations, closed);
            triangles = Some(&progress_tris);
        } else {
            return;
        }
        let token = (key, kf_index, self.shape_view, self.color_mode, positions.len());
        if self.shown == Some(token) {
            return;
        }
        self.shown = Some(token);
        let Some(all) = triangles else { return };
        // Only what has left the die (z <= 0) is pasta.
        let outside = |i: u32| positions.get(i as usize).map(|p| p[2] <= 1e-4).unwrap_or(false);
        let mut indices = Vec::with_capacity(tri_count * 3);
        for t in all.iter().take(tri_count) {
            if t.iter().all(|&i| outside(i)) {
                indices.extend_from_slice(t);
            }
        }
        let scalars = match self.color_mode {
            ColorMode::Plain => None,
            ColorMode::ExitSpeed | ColorMode::Thickness => self.die.as_ref().and_then(|d| d.1.profile(self.project.opening)).map(|p| {
                // Map each node's station back onto the profile by index
                // fraction (the shell resamples the stations).
                let src = if self.color_mode == ColorMode::ExitSpeed { &p.v } else { &p.thickness };
                station_values(src, stations, positions.len())
            }),
            ColorMode::Age => {
                let rows = positions.len() / stations.max(1);
                Some((0..positions.len()).map(|i| (rows - i / stations.max(1)) as f32).collect())
            }
        };
        let mesh = SceneMesh { positions: positions.as_ref().clone(), indices, scalars };
        if let Some(vp) = self.viewport.as_mut() {
            vp.set_mesh(&mesh);
            // Refit when the finished piece arrives, and during growth each
            // time the piece doubles, so it stays in frame without fighting
            // someone orbiting the camera.
            let rows = mesh.positions.len() / stations.max(1);
            let finished = self.shell.as_ref().is_some_and(|s| s.0 == self.shell_gen);
            if self.fit_pending && !mesh.indices.is_empty() && (finished || rows >= 2 * self.fit_rows.max(4)) {
                vp.set_view(ViewPreset::Iso);
                self.fit_rows = rows;
                if finished {
                    self.fit_pending = false;
                    self.fit_rows = 0;
                }
            }
        }
    }

    fn update_overlay(&mut self) {
        let key = (format!("{:?}{}", self.project.die, self.ghost.is_some()), self.project.shell.orientation);
        if self.overlay_for.as_ref() == Some(&key) {
            return;
        }
        self.overlay_for = Some(key);
        let die = &self.project.die;
        let outlines = die.outlines(256).into_iter().map(|(_, poly)| poly.iter().map(|p| [p[0] as f32, p[1] as f32, 0.0]).collect()).collect();
        let overlay = Overlay {
            ghost: self.ghost.as_ref().map(|g| g.mesh.clone()),
            die_outline: outlines,
            die_plate: Some(([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], (die.plate_diameter * 0.5) as f32)),
        };
        if let Some(vp) = self.viewport.as_mut() {
            vp.set_overlay(&overlay);
        }
    }
}

/// The shell's structured triangulation for `rows` rows (same order as the
/// simulator creates them).
fn strip_triangles(rows: usize, n_s: usize, closed: bool) -> Vec<[u32; 3]> {
    let segs = if closed { n_s } else { n_s.saturating_sub(1) };
    let mut out = Vec::new();
    for r in 0..rows.saturating_sub(1) {
        for i in 0..segs {
            let i1 = (i + 1) % n_s;
            let a = (r * n_s + i) as u32;
            let b = (r * n_s + i1) as u32;
            let c = ((r + 1) * n_s + i1) as u32;
            let d = ((r + 1) * n_s + i) as u32;
            out.push([a, b, c]);
            out.push([a, c, d]);
        }
    }
    out
}

/// Per-node values from a per-station profile, by station fraction.
fn station_values(src: &[f64], stations: usize, nodes: usize) -> Vec<f32> {
    let m = src.len().max(1);
    (0..nodes)
        .map(|i| {
            let s = i % stations;
            let f = s as f64 / (stations - 1).max(1) as f64;
            src[((f * (m - 1) as f64).round() as usize).min(m - 1)] as f32
        })
        .collect()
}

fn style(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::System);
    let accent = egui::Color32::from_rgb(0xE0, 0x8A, 0x1E);
    for theme in [egui::Theme::Dark, egui::Theme::Light] {
        ctx.style_mut_of(theme, |s| {
            s.spacing.item_spacing = egui::vec2(8.0, 6.0);
            s.spacing.button_padding = egui::vec2(8.0, 4.0);
            s.visuals.selection.bg_fill = accent.linear_multiply(0.6);
            s.visuals.selection.stroke.color = accent;
            s.visuals.hyperlink_color = accent;
            for w in [&mut s.visuals.widgets.inactive, &mut s.visuals.widgets.hovered, &mut s.visuals.widgets.active] {
                w.corner_radius = egui::CornerRadius::same(6);
            }
            s.visuals.window_corner_radius = egui::CornerRadius::same(8);
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let rs = frame.wgpu_render_state().cloned();
        self.poll_engine();
        if let Some(cmd) = self.shortcuts(&ctx) {
            self.run_command(cmd, rs.as_ref());
        }
        let before = self.project.clone();
        self.view.dark = ctx.theme() == egui::Theme::Dark;
        self.view.up = match self.project.shell.orientation {
            Orientation::Down => [0.0, 0.0, 1.0],
            Orientation::Horizontal => [0.0, 1.0, 0.0],
        };
        if self.color_mode != ColorMode::Plain && self.view.color_map == ColorMap::Coolwarm {
            self.view.color_map = ColorMap::Viridis;
        }

        let mut cmd = crate::panels::top_bar(self, ui);
        if let Some(c) = crate::panels::bottom_bar(self, ui) {
            cmd = Some(c);
        }
        egui::Panel::left("die").default_size(440.0).min_size(320.0).resizable(true).show(ui, |ui| {
            crate::panels::die_panel(self, ui);
        });
        egui::Panel::right("params").default_size(330.0).min_size(280.0).resizable(true).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| crate::panels::params_panel(self, ui));
        });
        self.update_viewport_mesh();
        self.update_overlay();
        egui::CentralPanel::default().show(ui, |ui| {
            match (self.viewport.as_mut(), rs.as_ref()) {
                (Some(vp), Some(rs)) => {
                    if let Some(p) = vp.ui(ui, rs, &self.view) {
                        self.pick_info = Some(pick_text(self, p.vertex as usize, p.position));
                    }
                }
                _ => {
                    ui.label("The 3D view needs the wgpu renderer.");
                }
            }
            crate::panels::view_overlay(self, ui);
        });
        if self.palette_open
            && let Some(c) = crate::panels::palette(self, &ctx) {
                cmd = Some(c);
            }
        if let Some(c) = cmd {
            self.run_command(c, rs.as_ref());
        }
        self.track_edits(&ctx, &before);
        self.auto_screenshot(&ctx);
        if self.playing {
            let now = ctx.input(|i| i.time);
            let n = self.shell.as_ref().map(|s| s.1.keyframes.len()).unwrap_or(0);
            if n > 0 && now - self.last_play > 1.0 / 30.0 {
                self.last_play = now;
                self.shape_view = ShapeView::Timeline;
                let k = self.timeline.map(|k| k + 1).unwrap_or(0);
                self.timeline = Some(if k >= n { 0 } else { k });
            }
            ctx.request_repaint();
        }
    }
}

impl App {
    fn auto_screenshot(&mut self, ctx: &egui::Context) {
        let Some((path, frames, done)) = &mut self.screenshot else { return };
        ctx.request_repaint();
        let ready = self.shell.is_some() || self.shell_error.is_some() || ctx.input(|i| i.time) > 600.0;
        if !ready {
            return;
        }
        *frames += 1;
        if *frames == 10 {
            let (path, done) = (path.clone(), done.clone());
            ctx.request_screenshot(move |img| {
                let [w, h] = img.size;
                match save_png(&path, w as u32, h as u32, img.as_raw()) {
                    Ok(()) => println!("wrote {}", path.display()),
                    Err(e) => println!("screenshot failed: {e}"),
                }
                done.store(true, std::sync::atomic::Ordering::SeqCst);
            });
        }
        if done.load(std::sync::atomic::Ordering::SeqCst) || *frames > 600 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

fn pick_text(app: &App, vertex: usize, pos: [f32; 3]) -> String {
    let Some(s) = &app.shell else { return String::new() };
    let r = &s.1;
    let n_s = r.stations.max(1);
    let row = vertex / n_s;
    let station = vertex % n_s;
    let speed = r.speed_at_node.get(vertex).copied().unwrap_or(0.0);
    let h = r.thickness_at_node.get(vertex).copied().unwrap_or(0.0);
    let rows = r.hanging.len() / n_s;
    let m = measure(r, &r.hanging);
    let _ = m;
    format!(
        "Station {} of {}, row {} of {}\nExit speed {:.2} mm/s, thickness {:.2} mm\nPosition ({:.1}, {:.1}, {:.1}) mm",
        station + 1,
        n_s,
        rows - row,
        rows,
        speed,
        h,
        pos[0],
        pos[1],
        pos[2]
    )
}
