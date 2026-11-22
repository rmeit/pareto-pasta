//! Interactive demo of the Polyscope-style viewport on a procedurally ruffled ribbon.
//!
//! Run: `cargo run -p pasta-app --example viewport_demo`
//! Headless check: `VIEWPORT_DEMO_SCREENSHOT=1 cargo run -p pasta-app --example viewport_demo`
//! renders a set of PNGs into the scratchpad directory, runs a short update/render benchmark and exits.

#[path = "../src/viewport/mod.rs"]
mod viewport;

use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use viewport::{
    CameraState, ColorMap, Material, Overlay, Pick, SceneMesh, ViewPreset, ViewSettings, Viewport, save_png,
};

const OUT_DIR: &str =
    "/tmp/claude-1000/-home-ryan-Desktop-pareto-pasta/117da548-21ea-4a03-b5e2-15ac87e50cdb/scratchpad";

const LENGTH: f32 = 100.0;
const WIDTH: f32 = 25.0;

/// Fill `mesh` with a ruffled ribbon: `nu` vertices along the 100 mm length (X), `nv` across the
/// 25 mm width (Y). Edges oscillate out of plane (Z) with amplitude growing toward the edges.
/// Scalars = out-of-plane displacement. Reuses the mesh's allocations.
fn ribbon(mesh: &mut SceneMesh, nu: usize, nv: usize, phase: f32, amp: f32) {
    let rebuild_topology = mesh.positions.len() != nu * nv;
    mesh.positions.clear();
    let scalars = mesh.scalars.get_or_insert_with(Vec::new);
    scalars.clear();
    let wavelength = 14.0;
    for i in 0..nu {
        let x = -LENGTH * 0.5 + LENGTH * i as f32 / (nu - 1) as f32;
        let xn = x / (LENGTH * 0.5);
        let base_z = 6.0 + 2.5 * (1.0 - xn * xn); // gentle arch
        for j in 0..nv {
            let s = -1.0 + 2.0 * j as f32 / (nv - 1) as f32;
            let y = WIDTH * 0.5 * s;
            let edge = s.abs().powi(3);
            // the two edges ruffle out of phase, like mafaldine
            let edge_phase = if s > 0.0 { 0.0 } else { std::f32::consts::FRAC_PI_2 };
            let k = std::f32::consts::TAU / wavelength;
            let wave = (k * x + phase + edge_phase).sin();
            let dz = amp * edge * wave;
            // slight in-plane gathering of the edges, so the ruffle reads in silhouette
            let dy = -0.6 * amp * edge * s.signum() * (0.5 + 0.5 * (2.0 * (k * x + phase + edge_phase)).cos()) * 0.3;
            mesh.positions.push([x, y + dy, base_z + dz]);
            scalars.push(dz);
        }
    }
    if rebuild_topology {
        mesh.indices.clear();
        for i in 0..nu - 1 {
            for j in 0..nv - 1 {
                let v00 = (i * nv + j) as u32;
                let v10 = ((i + 1) * nv + j) as u32;
                let v01 = (i * nv + j + 1) as u32;
                let v11 = ((i + 1) * nv + j + 1) as u32;
                mesh.indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
            }
        }
    }
}

fn demo_overlay(nu: usize, nv: usize) -> Overlay {
    let mut ghost = SceneMesh::default();
    ribbon(&mut ghost, nu / 2, nv / 2, 1.4, 2.5);
    ghost.scalars = None;
    for p in &mut ghost.positions {
        p[2] += 3.0;
    }
    // die plate just upstream of the ribbon start, normal along the extrusion axis (X)
    let px = -LENGTH * 0.5 - 4.0;
    let pz = 6.0;
    let mut slot = Vec::new();
    let n = 40;
    for i in 0..=n {
        let y = -WIDTH * 0.5 + WIDTH * i as f32 / n as f32;
        slot.push([px, y, pz + 0.6]);
    }
    for i in 0..=n {
        let y = WIDTH * 0.5 - WIDTH * i as f32 / n as f32;
        slot.push([px, y, pz - 0.6]);
    }
    Overlay {
        ghost: Some(ghost),
        die_outline: vec![slot],
        die_plate: Some(([px, 0.0, pz], [1.0, 0.0, 0.0], 22.0)),
    }
}

struct DemoApp {
    vp: Viewport,
    settings: ViewSettings,
    mesh: SceneMesh,
    resolution: (usize, usize),
    with_scalars: bool,
    auto_range: bool,
    range: (f32, f32),
    show_ghost: bool,
    show_die: bool,
    slice_on: bool,
    slice_x: f32,
    up_choice: usize,
    animate: bool,
    phase: f32,
    amp: f32,
    base_srgb: [u8; 3],
    back_srgb: [u8; 3],
    last_pick: Option<Pick>,
    status: String,
    screenshot_mode: bool,
    /// Headless mode: frames since the offscreen screenshots, and whether the window grab is saved.
    window_shot: Option<(u32, Arc<AtomicBool>)>,
    last_frame: Instant,
    frame_ms: f32,
}

const UPS: [([f32; 3], &str); 4] = [
    ([0.0, 0.0, 1.0], "+Z"),
    ([0.0, 1.0, 0.0], "+Y"),
    ([1.0, 0.0, 0.0], "+X"),
    ([0.0, 0.0, -1.0], "-Z"),
];

fn to_u8(c: [f32; 3]) -> [u8; 3] {
    c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}
fn to_f32(c: [u8; 3]) -> [f32; 3] {
    c.map(|v| v as f32 / 255.0)
}

impl DemoApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("viewport_demo needs the wgpu renderer");
        let vp = Viewport::new(rs);
        let settings = ViewSettings::default();
        let mut app = Self {
            vp,
            settings,
            mesh: SceneMesh::default(),
            resolution: (200, 64),
            with_scalars: false,
            auto_range: true,
            range: (-4.0, 4.0),
            show_ghost: false,
            show_die: false,
            slice_on: false,
            slice_x: 20.0,
            up_choice: 0,
            animate: false,
            phase: 0.0,
            amp: 4.0,
            base_srgb: to_u8(settings.base_color),
            back_srgb: to_u8(settings.back_color),
            last_pick: None,
            status: String::new(),
            window_shot: None,
            screenshot_mode: std::env::var("VIEWPORT_DEMO_SCREENSHOT").is_ok_and(|v| v == "1"),
            last_frame: Instant::now(),
            frame_ms: 0.0,
        };
        app.rebuild_mesh();
        app.rebuild_overlay();
        app.vp.set_view(ViewPreset::Iso);
        app
    }

    fn rebuild_mesh(&mut self) {
        let (nu, nv) = self.resolution;
        ribbon(&mut self.mesh, nu, nv, self.phase, self.amp);
        if self.with_scalars {
            self.vp.set_mesh(&self.mesh);
        } else {
            // same geometry, no scalars: temporarily take them out to avoid a copy
            let s = self.mesh.scalars.take();
            self.vp.set_mesh(&self.mesh);
            self.mesh.scalars = s;
        }
    }

    fn rebuild_overlay(&mut self) {
        let (nu, nv) = self.resolution;
        let mut o = demo_overlay(nu, nv);
        if !self.show_ghost {
            o.ghost = None;
        }
        if !self.show_die {
            o.die_outline.clear();
            o.die_plate = None;
        }
        self.vp.set_overlay(&o);
    }

    fn sync_settings(&mut self) {
        let s = &mut self.settings;
        s.base_color = to_f32(self.base_srgb);
        s.back_color = to_f32(self.back_srgb);
        s.scalar_range = if self.auto_range { None } else { Some(self.range) };
        s.up = UPS[self.up_choice].0;
        s.slice = self
            .slice_on
            .then_some(([self.slice_x, 0.0, 0.0], [1.0, 0.0, 0.0]));
    }

    fn controls(&mut self, ui: &mut egui::Ui, rs: &eframe::egui_wgpu::RenderState) {
        ui.heading("Viewport");
        let mut mesh_changed = false;
        let mut overlay_changed = false;

        egui::ComboBox::from_label("Material")
            .selected_text(format!("{:?}", self.settings.material))
            .show_ui(ui, |ui| {
                for m in [Material::Clay, Material::Wax, Material::Flat] {
                    ui.selectable_value(&mut self.settings.material, m, format!("{m:?}"));
                }
            });
        ui.horizontal(|ui| {
            ui.color_edit_button_srgb(&mut self.base_srgb);
            ui.label("front");
            ui.color_edit_button_srgb(&mut self.back_srgb);
            ui.label("back");
        });
        ui.separator();
        mesh_changed |= ui.checkbox(&mut self.with_scalars, "Scalars (displacement)").changed();
        egui::ComboBox::from_label("Colour map")
            .selected_text(format!("{:?}", self.settings.color_map))
            .show_ui(ui, |ui| {
                for m in [ColorMap::Viridis, ColorMap::Coolwarm, ColorMap::Turbo] {
                    ui.selectable_value(&mut self.settings.color_map, m, format!("{m:?}"));
                }
            });
        ui.checkbox(&mut self.auto_range, "Auto range");
        ui.add_enabled_ui(!self.auto_range, |ui| {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.range.0).speed(0.05).prefix("min "));
                ui.add(egui::DragValue::new(&mut self.range.1).speed(0.05).prefix("max "));
            });
        });
        ui.checkbox(&mut self.settings.show_colorbar, "Colour bar");
        ui.separator();
        ui.checkbox(&mut self.settings.wireframe, "Wireframe");
        ui.checkbox(&mut self.settings.ground, "Ground + shadow");
        ui.checkbox(&mut self.settings.dark, "Dark");
        egui::ComboBox::from_label("Up")
            .selected_text(UPS[self.up_choice].1)
            .show_ui(ui, |ui| {
                for (i, (_, name)) in UPS.iter().enumerate() {
                    ui.selectable_value(&mut self.up_choice, i, *name);
                }
            });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.slice_on, "Slice x <");
            ui.add_enabled(
                self.slice_on,
                egui::Slider::new(&mut self.slice_x, -55.0..=55.0),
            );
        });
        ui.separator();
        overlay_changed |= ui.checkbox(&mut self.show_ghost, "Ghost mesh").changed();
        overlay_changed |= ui.checkbox(&mut self.show_die, "Die plate + outline").changed();
        ui.separator();
        ui.horizontal(|ui| {
            for (p, name) in [
                (ViewPreset::Front, "Front"),
                (ViewPreset::Side, "Side"),
                (ViewPreset::Top, "Top"),
                (ViewPreset::Iso, "Iso"),
            ] {
                if ui.button(name).clicked() {
                    self.vp.set_view(p);
                }
            }
        });
        if ui.button("Fit").clicked() {
            self.vp.fit_to_view();
        }
        ui.separator();
        ui.checkbox(&mut self.animate, "Animate positions");
        mesh_changed |= ui
            .add(egui::Slider::new(&mut self.amp, 0.0..=8.0).text("ruffle mm"))
            .changed();
        let res = [(200, 64), (400, 128), (800, 256)];
        egui::ComboBox::from_label("Grid")
            .selected_text(format!("{} x {}", self.resolution.1, self.resolution.0))
            .show_ui(ui, |ui| {
                for r in res {
                    let tris = 2 * (r.0 - 1) * (r.1 - 1);
                    if ui
                        .selectable_value(&mut self.resolution, r, format!("{} x {} ({tris} tris)", r.1, r.0))
                        .changed()
                    {
                        mesh_changed = true;
                        overlay_changed = true;
                    }
                }
            });
        if ui.button("Screenshot").clicked() {
            let path = Path::new(OUT_DIR).join("viewport_demo.png");
            let t = Instant::now();
            let px = self.vp.render_image(rs, &self.settings, 1280, 800);
            self.status = match save_png(&path, 1280, 800, &px) {
                Ok(()) => format!("saved {} ({:.0} ms)", path.display(), t.elapsed().as_secs_f32() * 1e3),
                Err(e) => format!("screenshot failed: {e}"),
            };
        }
        ui.separator();
        ui.label(format!(
            "{} triangles\nframe {:.1} ms, last render (CPU) {:.2} ms",
            self.mesh.indices.len() / 3,
            self.frame_ms,
            self.vp.last_render_cpu_ms()
        ));
        if let Some(p) = self.last_pick {
            ui.label(format!(
                "pick: vertex {} tri {} at ({:.2}, {:.2}, {:.2})",
                p.vertex, p.triangle, p.position[0], p.position[1], p.position[2]
            ));
        }
        if !self.status.is_empty() {
            ui.label(&self.status);
        }
        ui.small("left drag: orbit · right/middle or shift+left: pan · scroll: zoom · double-click: fit · click: pick");

        if mesh_changed {
            self.rebuild_mesh();
        }
        if overlay_changed {
            self.rebuild_overlay();
        }
    }

    /// Headless verification: a set of screenshots, then a benchmark of the update path.
    fn run_screenshots(&mut self, rs: &eframe::egui_wgpu::RenderState) {
        let out = PathBuf::from(OUT_DIR);
        let _ = std::fs::create_dir_all(&out);
        let shot = |app: &mut Self, name: &str| {
            app.sync_settings();
            let t = Instant::now();
            let px = app.vp.render_image(rs, &app.settings, 1280, 800);
            let ms = t.elapsed().as_secs_f32() * 1e3;
            let path = out.join(name);
            match save_png(&path, 1280, 800, &px) {
                Ok(()) => println!("wrote {} (render+readback {ms:.1} ms)", path.display()),
                Err(e) => println!("failed to write {}: {e}", path.display()),
            }
        };
        shot(self, "viewport_demo.png");
        self.settings.wireframe = true;
        shot(self, "viewport_demo_wire.png");
        // close-up so individual triangles are visible
        let saved = self.vp.camera_state();
        self.vp.set_camera_state(CameraState {
            target: [30.0, 8.0, 8.0],
            distance: 28.0,
            ..saved
        });
        shot(self, "viewport_demo_wire_closeup.png");
        self.vp.set_camera_state(saved);
        self.settings.wireframe = false;

        self.with_scalars = true;
        self.settings.color_map = ColorMap::Coolwarm;
        self.rebuild_mesh();
        shot(self, "viewport_demo_scalars.png");

        self.settings.material = Material::Wax;
        self.show_ghost = true;
        self.show_die = true;
        self.slice_on = true;
        self.slice_x = 25.0;
        self.rebuild_overlay();
        shot(self, "viewport_demo_overlay.png");

        self.settings.dark = true;
        self.settings.color_map = ColorMap::Turbo;
        self.show_ghost = false;
        self.slice_on = false;
        self.rebuild_overlay();
        self.vp.set_view(ViewPreset::Front);
        shot(self, "viewport_demo_dark_front.png");

        self.settings.dark = false;
        self.settings.material = Material::Clay;
        self.with_scalars = false;
        self.show_die = false;
        self.rebuild_mesh();
        self.rebuild_overlay();
        self.vp.set_view(ViewPreset::Top);
        shot(self, "viewport_demo_top.png");
        self.up_choice = 1;
        self.vp.set_view(ViewPreset::Iso);
        shot(self, "viewport_demo_up_y.png");
        self.up_choice = 0;
        self.vp.set_view(ViewPreset::Iso);

        let pick = self.vp.pick_ndc(0.0, 0.0, &self.settings);
        println!("pick through view centre: {pick:?}");
        let pick = self.vp.pick_ndc(0.98, 0.98, &self.settings);
        println!("pick through top-right corner (expect None): {pick:?}");

        // benchmark: ~100k triangles, positions + scalars change every frame
        self.with_scalars = true;
        self.resolution = (400, 128);
        self.rebuild_mesh();
        self.sync_settings();
        let tris = self.mesh.indices.len() / 3;
        let n = 60;
        let (mut gen_ms, mut set_ms, mut render_ms) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            self.phase += 0.05;
            let t = Instant::now();
            let (nu, nv) = self.resolution;
            ribbon(&mut self.mesh, nu, nv, self.phase, self.amp);
            gen_ms.push(t.elapsed().as_secs_f32() * 1e3);
            let t = Instant::now();
            self.vp.set_mesh(&self.mesh);
            set_ms.push(t.elapsed().as_secs_f32() * 1e3);
            let t = Instant::now();
            let _ = self.vp.render_image(rs, &self.settings, 1280, 800);
            render_ms.push(t.elapsed().as_secs_f32() * 1e3);
        }
        let mut static_ms = Vec::new();
        for _ in 0..n {
            let t = Instant::now();
            let _ = self.vp.render_image(rs, &self.settings, 1280, 800);
            static_ms.push(t.elapsed().as_secs_f32() * 1e3);
        }
        println!(
            "bench {tris} tris, 1280x800 4xMSAA: median ribbon gen {:.2} ms | set_mesh (normals+expand+upload) {:.2} ms | \
             render_image incl. shadow rebuild + readback {:.2} ms | render_image, mesh unchanged {:.2} ms",
            median(&mut gen_ms),
            median(&mut set_ms),
            median(&mut render_ms),
            median(&mut static_ms),
        );
        self.animate = false;
    }
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

impl eframe::App for DemoApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let now = Instant::now();
        self.frame_ms = (now - self.last_frame).as_secs_f32() * 1e3;
        self.last_frame = now;
        let Some(rs) = frame.wgpu_render_state().cloned() else {
            ui.label("wgpu renderer required");
            return;
        };

        if self.screenshot_mode {
            self.screenshot_mode = false;
            self.run_screenshots(&rs);
            // then grab the real window (egui path: Viewport::ui + colour bar)
            self.settings.color_map = ColorMap::Viridis;
            self.settings.show_colorbar = true;
            self.vp.set_view(ViewPreset::Iso);
            self.window_shot = Some((0, Arc::new(AtomicBool::new(false))));
        }
        if let Some((frames, done)) = &mut self.window_shot {
            *frames += 1;
            ui.ctx().request_repaint();
            if *frames == 5 {
                let done = done.clone();
                ui.ctx().request_screenshot(move |img| {
                    let path = Path::new(OUT_DIR).join("viewport_demo_window.png");
                    let [w, h] = img.size;
                    match save_png(&path, w as u32, h as u32, img.as_raw()) {
                        Ok(()) => println!("wrote {} (window grab)", path.display()),
                        Err(e) => println!("window grab failed: {e}"),
                    }
                    done.store(true, Ordering::SeqCst);
                });
            }
            if done.load(Ordering::SeqCst) || *frames > 300 {
                if *frames > 300 {
                    println!("window grab timed out (window not being redrawn?)");
                }
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        if self.animate {
            self.phase += 0.05;
            self.rebuild_mesh();
            ui.ctx().request_repaint();
        }

        egui::Panel::left("controls")
            .default_size(280.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, &rs));
            });
        self.sync_settings();
        egui::CentralPanel::default().show(ui, |ui| {
            if let Some(p) = self.vp.ui(ui, &rs, &self.settings) {
                self.last_pick = Some(p);
            }
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 860.0])
            .with_title("pasta viewport demo"),
        ..Default::default()
    };
    eframe::run_native(
        "viewport_demo",
        options,
        Box::new(|cc| Ok(Box::new(DemoApp::new(cc)))),
    )
}
