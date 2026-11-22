//! Polyscope-style interactive 3D mesh viewport.
//!
//! The scene is rendered with wgpu into an offscreen texture (4x MSAA, resolved into an
//! `Rgba8Unorm` texture through an `Rgba8UnormSrgb` view) that egui displays as an image.
//! Rendering only happens when something changed (camera, mesh, overlay, settings or size).
//!
//! Usage from an eframe app (wgpu renderer):
//! ```ignore
//! let rs = frame.wgpu_render_state().unwrap();
//! let mut vp = Viewport::new(rs);
//! vp.set_mesh(&mesh);
//! // each frame, inside some panel:
//! if let Some(pick) = vp.ui(ui, rs, &settings) { /* ... */ }
//! ```

mod camera;
mod geom;
#[allow(unused_imports)] // used by the app, not by the demo example
pub(crate) use geom::colormap_srgb;
mod gpu;

use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu;
use glam::Vec3;

use camera::Camera;
use geom::{GpuVertex, LineVertex};
use gpu::{Gpu, Target, Uniforms};

// ------------------------------------------------------------------------------------ public API

/// A triangle mesh in world space (mm).
#[derive(Clone, Debug, Default)]
pub struct SceneMesh {
    pub positions: Vec<[f32; 3]>,
    /// Triangle list (3 indices per triangle).
    pub indices: Vec<u32>,
    /// Per-vertex value for colour mapping; `None` = plain material colour.
    pub scalars: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Material {
    /// Soft matte with a gentle rim light.
    #[default]
    Clay,
    /// Clay plus a soft specular highlight.
    Wax,
    /// Unlit.
    Flat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorMap {
    #[default]
    Viridis,
    Coolwarm,
    Turbo,
}

/// Display settings, passed every frame. Colours are sRGB components in 0..1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewSettings {
    pub material: Material,
    /// Front-face colour (sRGB 0..1).
    pub base_color: [f32; 3],
    /// Back-face colour (sRGB 0..1), used when there are no scalars.
    pub back_color: [f32; 3],
    pub color_map: ColorMap,
    /// `None` = auto from data.
    pub scalar_range: Option<(f32, f32)>,
    /// Barycentric edge overlay on the main mesh.
    pub wireframe: bool,
    /// Ground plane below the mesh with a soft shadow.
    pub ground: bool,
    /// World up direction (gravity is -up); the ground is perpendicular to it.
    pub up: [f32; 3],
    /// (point, normal): fragments on the positive side of the plane are discarded.
    pub slice: Option<([f32; 3], [f32; 3])>,
    pub show_colorbar: bool,
    /// Dark background / ground.
    pub dark: bool,
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self {
            material: Material::Clay,
            base_color: [0.96, 0.86, 0.58],
            back_color: [0.58, 0.68, 0.80],
            color_map: ColorMap::Viridis,
            scalar_range: None,
            wireframe: false,
            ground: true,
            up: [0.0, 0.0, 1.0],
            slice: None,
            show_colorbar: true,
            dark: false,
        }
    }
}

/// Extra geometry drawn with the main mesh.
#[derive(Clone, Debug, Default)]
pub struct Overlay {
    /// Translucent comparison mesh (alpha ~0.3, scalars ignored).
    pub ghost: Option<SceneMesh>,
    /// Closed polylines (die openings) drawn as thick lines.
    pub die_outline: Vec<Vec<[f32; 3]>>,
    /// (centre, normal, radius): translucent disc for the die plate.
    pub die_plate: Option<([f32; 3], [f32; 3], f32)>,
}

/// Result of clicking on the mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    /// Mesh vertex closest to the hit point (within the hit triangle).
    pub vertex: u32,
    /// Exact hit point on the surface (world space).
    pub position: [f32; 3],
    /// Index of the hit triangle (`indices[3*triangle..3*triangle+3]`).
    pub triangle: u32,
}

/// Turntable camera state (angles in radians, relative to the `up` frame), for saving/restoring
/// views.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraState {
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewPreset {
    Front,
    Side,
    Top,
    Iso,
}

/// Encode RGBA8 pixels to a PNG file.
pub fn save_png(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let w = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(w, width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = enc.write_header().map_err(std::io::Error::other)?;
    writer.write_image_data(rgba).map_err(std::io::Error::other)?;
    writer.finish().map_err(std::io::Error::other)
}

// ------------------------------------------------------------------------------------ internals

/// Ground plane placement (with hysteresis so animated meshes don't make it jitter).
#[derive(Clone, Copy, Debug)]
struct Ground {
    /// Height along `up` of the plane.
    level: f32,
    /// Horizontal centre (coordinates along camera.a / camera.b).
    center: [f32; 2],
    /// Half extent of the shadow map square.
    extent: f32,
}

type RendererLock = Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>;

/// Interactive 3D viewport. Create once per view; GPU resources are owned here.
pub struct Viewport {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: RendererLock,
    max_dim: u32,
    gpu: Gpu,

    target: Option<Target>,
    texture_id: Option<egui::TextureId>,

    camera: Camera,
    camera_fitted: bool,
    /// A fit was requested; redo it once the real viewport aspect is known.
    fit_pending: bool,
    aspect: f32,
    pixels_per_point: f32,

    // CPU copy of the main mesh (picking, bounds), plus reusable scratch buffers.
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    normals: Vec<[f32; 3]>,
    scratch: Vec<GpuVertex>,
    line_scratch: Vec<LineVertex>,
    has_scalars: bool,
    data_range: Option<(f32, f32)>,
    mesh_bounds: Option<(Vec3, Vec3)>,
    overlay_bounds: Option<(Vec3, Vec3)>,
    /// Representative overlay points (plate rim, outlines, ghost box corners).
    overlay_points: Vec<Vec3>,

    ground: Option<Ground>,
    last_settings: Option<ViewSettings>,
    dirty: bool,
    shadow_dirty: bool,
    last_render_cpu_ms: f32,
}

impl Viewport {
    pub fn new(render_state: &egui_wgpu::RenderState) -> Self {
        let device = render_state.device.clone();
        let gpu = Gpu::new(&device);
        Self {
            max_dim: device.limits().max_texture_dimension_2d,
            queue: render_state.queue.clone(),
            renderer: render_state.renderer.clone(),
            device,
            gpu,
            target: None,
            texture_id: None,
            camera: Camera::new(Vec3::Z),
            camera_fitted: false,
            fit_pending: false,
            aspect: 1.6,
            pixels_per_point: 1.0,
            positions: Vec::new(),
            indices: Vec::new(),
            normals: Vec::new(),
            scratch: Vec::new(),
            line_scratch: Vec::new(),
            has_scalars: false,
            data_range: None,
            mesh_bounds: None,
            overlay_bounds: None,
            overlay_points: Vec::new(),
            ground: None,
            last_settings: None,
            dirty: true,
            shadow_dirty: true,
            last_render_cpu_ms: 0.0,
        }
    }

    /// Replace the main mesh. Only uploads into existing GPU buffers unless the mesh grew.
    pub fn set_mesh(&mut self, mesh: &SceneMesh) {
        self.positions.clear();
        self.positions.extend_from_slice(&mesh.positions);
        self.indices.clear();
        self.indices.extend_from_slice(&mesh.indices[..mesh.indices.len() / 3 * 3]);
        geom::vertex_normals(&self.positions, &self.indices, &mut self.normals);
        let scalars = mesh
            .scalars
            .as_deref()
            .filter(|s| s.len() == self.positions.len());
        self.has_scalars = scalars.is_some();
        self.data_range = scalars.and_then(geom::scalar_range);
        geom::expand(&self.positions, &self.normals, scalars, &self.indices, &mut self.scratch);
        self.gpu.mesh.upload(&self.device, &self.queue, &self.scratch);

        self.mesh_bounds = geom::bounds(&self.positions);
        self.update_ground(false);
        self.shadow_dirty = true;
        self.dirty = true;
        if !self.camera_fitted && self.mesh_bounds.is_some() {
            self.fit_to_view();
        }
    }

    /// Replace the overlay geometry (ghost mesh, die outline, die plate).
    pub fn set_overlay(&mut self, overlay: &Overlay) {
        self.overlay_points.clear();

        // ghost
        self.scratch.clear();
        if let Some(g) = &overlay.ghost {
            let n = g.indices.len() / 3 * 3;
            let mut normals = Vec::new();
            geom::vertex_normals(&g.positions, &g.indices[..n], &mut normals);
            geom::expand(&g.positions, &normals, None, &g.indices[..n], &mut self.scratch);
            if let Some((lo, hi)) = geom::bounds(&g.positions) {
                self.overlay_points.extend(box_corners(lo, hi));
            }
        }
        self.gpu.ghost.upload(&self.device, &self.queue, &self.scratch);

        // plate disc + rim, outline polylines
        self.scratch.clear();
        self.line_scratch.clear();
        let push_polyline = |pts: &[Vec3], out: &mut Vec<LineVertex>| {
            if pts.len() < 2 {
                return;
            }
            for i in 0..pts.len() {
                let a = pts[i];
                let b = pts[(i + 1) % pts.len()];
                if pts.len() == 2 && i == 1 {
                    break;
                }
                let (a, b) = (a.to_array(), b.to_array());
                for (t, s) in [(0.0, -1.0), (1.0, -1.0), (1.0, 1.0), (0.0, -1.0), (1.0, 1.0), (0.0, 1.0)] {
                    out.push(LineVertex { a, b, ts: [t, s] });
                }
            }
        };
        if let Some((c, n, r)) = overlay.die_plate {
            let c = Vec3::from(c);
            let n = Vec3::from(n).try_normalize().unwrap_or(Vec3::Z);
            let (t1, t2) = n.any_orthonormal_pair();
            const SEG: usize = 96;
            let rim: Vec<Vec3> = (0..SEG)
                .map(|i| {
                    let th = i as f32 / SEG as f32 * std::f32::consts::TAU;
                    c + (t1 * th.cos() + t2 * th.sin()) * r
                })
                .collect();
            for i in 0..SEG {
                for p in [c, rim[i], rim[(i + 1) % SEG]] {
                    self.scratch.push(GpuVertex {
                        pos: p.to_array(),
                        normal: n.to_array(),
                        scalar: 0.0,
                    });
                }
            }
            push_polyline(&rim, &mut self.line_scratch);
            self.overlay_points.extend_from_slice(&rim);
        }
        self.gpu.plate.upload(&self.device, &self.queue, &self.scratch);

        for poly in &overlay.die_outline {
            let pts: Vec<Vec3> = poly.iter().map(|p| Vec3::from(*p)).collect();
            push_polyline(&pts, &mut self.line_scratch);
            self.overlay_points.extend_from_slice(&pts);
        }
        self.gpu.lines.upload(&self.device, &self.queue, &self.line_scratch);

        self.overlay_bounds = self
            .overlay_points
            .iter()
            .filter(|p| p.is_finite())
            .fold(None, |acc: Option<(Vec3, Vec3)>, &p| {
                Some(acc.map_or((p, p), |(lo, hi)| (lo.min(p), hi.max(p))))
            });
        self.update_ground(true);
        self.dirty = true;
    }

    /// Frame the current mesh bounds (or the overlay if there is no mesh).
    pub fn fit_to_view(&mut self) {
        let aspect = self.aspect;
        if let Some((lo, hi)) = self.mesh_bounds {
            let pts = self.positions.iter().map(|p| Vec3::from(*p)).filter(|p| p.is_finite());
            self.camera.fit_points((lo + hi) * 0.5, pts, aspect);
        } else if let Some((lo, hi)) = self.overlay_bounds {
            self.camera
                .fit_points((lo + hi) * 0.5, self.overlay_points.iter().copied(), aspect);
        } else {
            return;
        }
        self.camera_fitted = true;
        self.fit_pending = true;
        self.dirty = true;
    }

    /// Standard views relative to the current `up` direction. Also re-frames the mesh.
    pub fn set_view(&mut self, preset: ViewPreset) {
        let (yaw, pitch) = match preset {
            ViewPreset::Front => (-90.0f32, 0.0f32),
            ViewPreset::Side => (0.0, 0.0),
            ViewPreset::Top => (-90.0, 90.0),
            ViewPreset::Iso => (-60.0, 28.0),
        };
        self.camera.yaw = yaw.to_radians();
        self.camera.pitch = pitch.to_radians();
        self.fit_to_view();
        self.dirty = true;
    }

    pub fn camera_state(&self) -> CameraState {
        CameraState {
            target: self.camera.target.to_array(),
            yaw: self.camera.yaw,
            pitch: self.camera.pitch,
            distance: self.camera.distance,
        }
    }

    pub fn set_camera_state(&mut self, c: CameraState) {
        self.camera.target = Vec3::from(c.target);
        self.camera.yaw = c.yaw;
        let lim = std::f32::consts::FRAC_PI_2;
        self.camera.pitch = c.pitch.clamp(-lim, lim);
        self.camera.distance = c.distance.max(1e-4);
        self.camera_fitted = true;
        self.fit_pending = false;
        self.dirty = true;
    }

    /// CPU time spent encoding + submitting the last render, in milliseconds.
    pub fn last_render_cpu_ms(&self) -> f32 {
        self.last_render_cpu_ms
    }

    /// Effective scalar range shown by the colour map (settings override or data range).
    pub fn scalar_range(&self, settings: &ViewSettings) -> Option<(f32, f32)> {
        if !self.has_scalars {
            return None;
        }
        settings.scalar_range.or(self.data_range)
    }

    /// Draw into `ui` filling the available rect and handle mouse input:
    /// orbit (left drag), pan (right/middle drag or shift+left), zoom to cursor (scroll / pinch),
    /// double-click = fit. Returns a pick when the user clicks (without dragging) on the mesh.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        render_state: &egui_wgpu::RenderState,
        settings: &ViewSettings,
    ) -> Option<Pick> {
        let avail = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
        if rect.width() < 1.0 || rect.height() < 1.0 {
            return None;
        }
        let ppp = ui.pixels_per_point();
        if ppp != self.pixels_per_point {
            self.pixels_per_point = ppp;
            self.dirty = true;
        }
        let w = ((rect.width() * ppp).round() as u32).clamp(1, self.max_dim);
        let h = ((rect.height() * ppp).round() as u32).clamp(1, self.max_dim);
        self.aspect = w as f32 / h as f32;
        self.apply_settings(settings);
        if self.fit_pending {
            self.fit_to_view();
            self.fit_pending = false;
        }

        let pick = self.handle_input(ui, &response, rect, settings);

        // (re)create the render target on resize
        if self.target.as_ref().is_none_or(|t| t.width != w || t.height != h) {
            let target = Target::new(&render_state.device, w, h);
            let mut renderer = render_state.renderer.write();
            match self.texture_id {
                Some(id) => renderer.update_egui_texture_from_wgpu_texture(
                    &render_state.device,
                    &target.egui_view,
                    wgpu::FilterMode::Linear,
                    id,
                ),
                None => {
                    self.texture_id = Some(renderer.register_native_texture(
                        &render_state.device,
                        &target.egui_view,
                        wgpu::FilterMode::Linear,
                    ));
                }
            }
            self.target = Some(target);
            self.dirty = true;
        }

        if self.dirty {
            let start = std::time::Instant::now();
            let line_px = 2.0 * ppp;
            let uniforms = self.uniforms(settings, w, h, line_px);
            let rebuild_shadow = self.shadow_dirty && settings.ground;
            if let Some(target) = &self.target {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("viewport frame"),
                });
                self.gpu.encode_frame(
                    &self.queue,
                    &mut encoder,
                    target,
                    &uniforms,
                    rebuild_shadow,
                    settings.ground,
                );
                self.queue.submit(Some(encoder.finish()));
            }
            if rebuild_shadow {
                self.shadow_dirty = false;
            }
            self.dirty = false;
            self.last_render_cpu_ms = start.elapsed().as_secs_f32() * 1000.0;
        }

        let painter = ui.painter_at(rect);
        if let Some(id) = self.texture_id {
            painter.image(
                id,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        if settings.show_colorbar
            && let Some(range) = self.scalar_range(settings)
        {
            draw_colorbar(&painter, rect, settings, range);
        }
        pick
    }

    /// Render the current view at the given size and return tightly packed RGBA8 pixels
    /// (sRGB, row-major, top row first).
    pub fn render_image(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        settings: &ViewSettings,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let device = &render_state.device;
        let queue = &render_state.queue;
        let width = width.clamp(1, self.max_dim);
        let height = height.clamp(1, self.max_dim);
        self.apply_settings(settings);

        let target = Target::new(device, width, height);
        let line_px = (height as f32 / 400.0).max(2.0);
        let saved_aspect = self.aspect;
        self.aspect = width as f32 / height as f32;
        if self.fit_pending {
            // frame for this image's aspect; the interactive view refits on its next `ui`
            self.fit_to_view();
        }
        let uniforms = self.uniforms(settings, width, height, line_px);
        self.aspect = saved_aspect;
        let rebuild_shadow = self.shadow_dirty && settings.ground;

        let bytes_per_row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport readback"),
            size: bytes_per_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viewport screenshot"),
        });
        self.gpu
            .encode_frame(queue, &mut encoder, &target, &uniforms, rebuild_shadow, settings.ground);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.resolve,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        if rebuild_shadow {
            self.shadow_dirty = false;
        }
        // the shared uniform buffer now holds the screenshot camera: re-render the view next time
        self.dirty = true;

        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        let mut out = vec![0u8; (width * height * 4) as usize];
        if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            return out;
        }
        if !matches!(rx.recv(), Ok(Ok(()))) {
            return out;
        }
        if let Ok(view) = readback.get_mapped_range(..) {
            let row = (width * 4) as usize;
            for (y, dst) in out.chunks_exact_mut(row).enumerate() {
                let src = y * bytes_per_row as usize;
                dst.copy_from_slice(&view[src..src + row]);
            }
        }
        readback.unmap();
        out
    }

    // -------------------------------------------------------------------------------- private

    fn apply_settings(&mut self, s: &ViewSettings) {
        if self.last_settings.as_ref() == Some(s) {
            return;
        }
        let prev = self.last_settings;
        let up_changed = prev.is_none_or(|p| p.up != s.up);
        if up_changed {
            self.camera.set_up(Vec3::from(s.up));
            self.update_ground(true);
        }
        if prev.is_none_or(|p| p.slice != s.slice || p.ground != s.ground) {
            self.shadow_dirty = true;
        }
        self.last_settings = Some(*s);
        self.dirty = true;
    }

    /// Bounding sphere of everything (mesh + overlay), for near/far planes.
    fn scene_sphere(&self) -> (Vec3, f32) {
        let b = match (self.mesh_bounds, self.overlay_bounds) {
            (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1))),
            (a, b) => a.or(b),
        };
        match b {
            Some((lo, hi)) => ((lo + hi) * 0.5, ((hi - lo) * 0.5).length().max(1e-3)),
            None => (Vec3::ZERO, 50.0),
        }
    }

    /// Recompute the ground plane placement; with `force = false` small changes are ignored.
    fn update_ground(&mut self, force: bool) {
        let (lo, hi) = match (self.mesh_bounds, self.overlay_bounds) {
            (Some(a), Some(b)) => (a.0.min(b.0), a.1.max(b.1)),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => return,
        };
        let (a, b, up) = (self.camera.a, self.camera.b, self.camera.up);
        let mut hmin = f32::INFINITY;
        let (mut amin, mut amax) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut bmin, mut bmax) = (f32::INFINITY, f32::NEG_INFINITY);
        for c in box_corners(lo, hi) {
            hmin = hmin.min(c.dot(up));
            amin = amin.min(c.dot(a));
            amax = amax.max(c.dot(a));
            bmin = bmin.min(c.dot(b));
            bmax = bmax.max(c.dot(b));
        }
        // the AABB is conservative for tilted up vectors: use the actual points
        if !self.positions.is_empty() || !self.overlay_points.is_empty() {
            hmin = self
                .positions
                .iter()
                .map(|p| Vec3::from(*p))
                .chain(self.overlay_points.iter().copied())
                .filter(|p| p.is_finite())
                .map(|p| p.dot(up))
                .fold(f32::INFINITY, f32::min);
        }
        let diag = (hi - lo).length().max(1e-3);
        let level = hmin - 0.02 * diag;
        let center = [(amin + amax) * 0.5, (bmin + bmax) * 0.5];
        let extent = ((amax - amin).max(bmax - bmin) * 0.5 + 0.15 * diag) * 1.15;
        let new = Ground {
            level,
            center,
            extent,
        };
        let keep = !force
            && self.ground.is_some_and(|g| {
                level >= g.level
                    && level <= g.level + 0.1 * diag
                    && (extent / g.extent - 1.0).abs() < 0.25
                    && (center[0] - g.center[0]).hypot(center[1] - g.center[1]) < 0.1 * diag
            });
        if !keep {
            self.ground = Some(new);
            self.shadow_dirty = true;
            self.dirty = true;
        }
    }

    fn uniforms(&self, s: &ViewSettings, width: u32, height: u32, line_px: f32) -> Uniforms {
        let lin = |c: [f32; 3], a: f32| {
            [
                geom::srgb_to_linear(c[0].clamp(0.0, 1.0)),
                geom::srgb_to_linear(c[1].clamp(0.0, 1.0)),
                geom::srgb_to_linear(c[2].clamp(0.0, 1.0)),
                a,
            ]
        };
        let v4 = |v: Vec3, w: f32| [v.x, v.y, v.z, w];

        let cam = &self.camera;
        let frame = cam.frame();
        let ground = self.ground.unwrap_or(Ground {
            level: 0.0,
            center: [0.0, 0.0],
            extent: 50.0,
        });
        let ground_origin = cam.a * ground.center[0] + cam.b * ground.center[1] + cam.up * ground.level;
        let ground_half = ground.extent * 3.0;
        let spacing = nice_step(ground.extent / 4.0);

        let (sc, sr) = self.scene_sphere();
        let mut radius = sr.max((ground_origin - sc).length() + ground_half * std::f32::consts::SQRT_2);
        if !s.ground {
            radius = sr;
        }
        let aspect = width as f32 / height as f32;
        let view = cam.view();
        let proj = cam.proj(aspect, sc, radius);
        let above = (frame.eye - ground_origin).dot(cam.up) > 0.0;

        let (smin, smax) = self.scalar_range(s).unwrap_or((0.0, 1.0));
        let (slice_p, slice_n, slice_on) = match s.slice {
            Some((p, n)) => (Vec3::from(p), Vec3::from(n).normalize_or_zero(), 1.0),
            None => (Vec3::ZERO, Vec3::Z, 0.0),
        };
        let dark = s.dark;
        Uniforms {
            view_proj: (proj * view).to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            camera_pos: v4(frame.eye, 1.0),
            base_color: lin(s.base_color, 1.0),
            back_color: lin(s.back_color, 1.0),
            ghost_color: lin([0.55, 0.66, 0.86], 0.3),
            plate_color: lin([0.70, 0.72, 0.78], 0.28),
            line_color: if dark {
                lin([0.85, 0.88, 0.95], 1.0)
            } else {
                lin([0.13, 0.16, 0.24], 1.0)
            },
            wire_color: lin([0.10, 0.10, 0.12], 0.7),
            slice_point: v4(slice_p, slice_on),
            slice_normal: v4(slice_n, 0.0),
            scalar: [
                smin,
                smax,
                if self.has_scalars { 1.0 } else { 0.0 },
                match s.color_map {
                    ColorMap::Viridis => 0.0,
                    ColorMap::Coolwarm => 1.0,
                    ColorMap::Turbo => 2.0,
                },
            ],
            flags: [
                match s.material {
                    Material::Clay => 0.0,
                    Material::Wax => 1.0,
                    Material::Flat => 2.0,
                },
                if s.wireframe { 1.0 } else { 0.0 },
                if s.ground { 1.0 } else { 0.0 },
                if dark { 1.0 } else { 0.0 },
            ],
            ground_origin: v4(ground_origin, ground.extent),
            ground_u: v4(cam.a, ground_half),
            ground_v: v4(cam.b, spacing),
            up: v4(cam.up, if above { 1.0 } else { 0.0 }),
            viewport: [width as f32, height as f32, line_px, 0.0],
            bg_top: if dark {
                lin([0.20, 0.21, 0.24], 1.0)
            } else {
                lin([0.99, 0.99, 1.0], 1.0)
            },
            bg_bottom: if dark {
                lin([0.07, 0.07, 0.08], 1.0)
            } else {
                lin([0.84, 0.85, 0.88], 1.0)
            },
            ground_color: if dark {
                lin([0.15, 0.15, 0.17], 0.6)
            } else {
                lin([0.93, 0.93, 0.94], 0.5)
            },
        }
    }

    fn handle_input(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        rect: egui::Rect,
        settings: &ViewSettings,
    ) -> Option<Pick> {
        let shift = ui.input(|i| i.modifiers.shift);
        let delta = response.drag_delta();
        if delta != egui::Vec2::ZERO {
            let pan = response.dragged_by(egui::PointerButton::Secondary)
                || response.dragged_by(egui::PointerButton::Middle)
                || (shift && response.dragged_by(egui::PointerButton::Primary));
            if pan {
                self.camera.pan(delta.x, delta.y, rect.height());
                self.dirty = true;
            } else if response.dragged_by(egui::PointerButton::Primary) {
                self.camera.orbit(delta.x * 0.008, delta.y * 0.008);
                self.dirty = true;
            }
        }

        let to_ndc = |p: egui::Pos2| {
            (
                (p.x - rect.left()) / rect.width() * 2.0 - 1.0,
                1.0 - (p.y - rect.top()) / rect.height() * 2.0,
            )
        };

        if response.hovered() {
            let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta(), i.zoom_delta()));
            let mut factor = (-scroll.y * 0.0025).exp();
            if zoom > 0.0 {
                factor /= zoom;
            }
            if (factor - 1.0).abs() > 1e-6 {
                let (nx, ny) = response.hover_pos().map(to_ndc).unwrap_or((0.0, 0.0));
                self.camera.zoom_at(factor, nx, ny, self.aspect);
                self.dirty = true;
            }
        }

        if response.double_clicked() {
            self.fit_to_view();
            return None;
        }
        if response.clicked_by(egui::PointerButton::Primary) {
            let pos = response.interact_pointer_pos()?;
            let (nx, ny) = to_ndc(pos);
            return self.pick_ndc(nx, ny, settings);
        }
        None
    }


    /// Cast a ray through normalised viewport coordinates (x, y in -1..1, y up) of the last
    /// shown view and return the closest hit on the main mesh (honouring the slice plane).
    pub fn pick_ndc(&self, nx: f32, ny: f32, settings: &ViewSettings) -> Option<Pick> {
        let origin = self.camera.frame().eye;
        let dir = self.camera.ray_dir(nx, ny, self.aspect);
        let slice = settings
            .slice
            .map(|(p, n)| (Vec3::from(p), Vec3::from(n)));
        let keep = |x: Vec3| slice.is_none_or(|(p, n)| (x - p).dot(n) <= 0.0);
        let (tri, t, u, v) = geom::raycast(&self.positions, &self.indices, origin, dir, keep)?;
        let ids = &self.indices[tri * 3..tri * 3 + 3];
        let w = [1.0 - u - v, u, v];
        let k = (0..3).max_by(|&i, &j| w[i].total_cmp(&w[j])).unwrap_or(0);
        Some(Pick {
            vertex: ids[k],
            position: (origin + dir * t).to_array(),
            triangle: tri as u32,
        })
    }
}

impl Drop for Viewport {
    fn drop(&mut self) {
        if let Some(id) = self.texture_id.take() {
            self.renderer.write().free_texture(&id);
        }
    }
}

fn box_corners(lo: Vec3, hi: Vec3) -> [Vec3; 8] {
    std::array::from_fn(|i| {
        Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        )
    })
}

/// 1, 2 or 5 times a power of ten, close to `x`.
fn nice_step(x: f32) -> f32 {
    let x = x.max(1e-6);
    let p = 10f32.powf(x.log10().floor());
    let m = x / p;
    let s = if m < 1.5 {
        1.0
    } else if m < 3.5 {
        2.0
    } else if m < 7.5 {
        5.0
    } else {
        10.0
    };
    s * p
}

/// Format a colour-bar label with precision suited to the range `span`.
fn format_value(v: f32, span: f32) -> String {
    let span = span.abs().max(f32::MIN_POSITIVE);
    let v = if v.abs() < span * 1e-4 { 0.0 } else { v };
    let a = v.abs().max(span);
    if !(1e-3..1e5).contains(&a) {
        format!("{v:.2e}")
    } else {
        let decimals = (2.0 - span.log10().floor()).clamp(0.0, 5.0) as usize;
        format!("{v:.decimals$}")
    }
}

fn draw_colorbar(painter: &egui::Painter, rect: egui::Rect, s: &ViewSettings, (lo, hi): (f32, f32)) {
    let bar_h = (rect.height() * 0.45).clamp(60.0, 260.0);
    let bar_w = 14.0;
    let margin = 16.0;
    let bar = egui::Rect::from_min_size(
        egui::pos2(rect.right() - margin - bar_w, rect.center().y - bar_h * 0.5),
        egui::vec2(bar_w, bar_h),
    );
    let text_color = if s.dark {
        egui::Color32::from_gray(225)
    } else {
        egui::Color32::from_gray(30)
    };
    let bg = if s.dark {
        egui::Color32::from_black_alpha(140)
    } else {
        egui::Color32::from_white_alpha(170)
    };
    let font = egui::FontId::proportional(12.0);
    let labels = [(hi, bar.top()), ((lo + hi) * 0.5, bar.center().y), (lo, bar.bottom())];
    let label_w = labels
        .iter()
        .map(|(v, _)| {
            painter
                .layout_no_wrap(format_value(*v, hi - lo), font.clone(), text_color)
                .size()
                .x
        })
        .fold(0.0f32, f32::max);
    let panel = egui::Rect::from_min_max(
        egui::pos2(bar.left() - label_w - 14.0, bar.top() - 10.0),
        egui::pos2(bar.right() + 6.0, bar.bottom() + 10.0),
    );
    painter.rect_filled(panel, 4.0, bg);

    let mut mesh = egui::Mesh::default();
    const N: usize = 48;
    for i in 0..=N {
        let t = i as f32 / N as f32;
        let c = geom::colormap_srgb(s.color_map, t);
        let col = egui::Color32::from_rgb(
            (c[0] * 255.0).round() as u8,
            (c[1] * 255.0).round() as u8,
            (c[2] * 255.0).round() as u8,
        );
        let y = bar.bottom() - t * bar.height();
        mesh.colored_vertex(egui::pos2(bar.left(), y), col);
        mesh.colored_vertex(egui::pos2(bar.right(), y), col);
        if i > 0 {
            let k = (2 * i) as u32;
            mesh.add_triangle(k - 2, k - 1, k);
            mesh.add_triangle(k - 1, k + 1, k);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
    painter.rect_stroke(
        bar,
        0.0,
        egui::Stroke::new(1.0, text_color.gamma_multiply(0.6)),
        egui::StrokeKind::Outside,
    );
    for (v, y) in labels {
        painter.line_segment(
            [egui::pos2(bar.left() - 4.0, y), egui::pos2(bar.left(), y)],
            egui::Stroke::new(1.0, text_color),
        );
        painter.text(
            egui::pos2(bar.left() - 7.0, y),
            egui::Align2::RIGHT_CENTER,
            format_value(v, hi - lo),
            font.clone(),
            text_color,
        );
    }
}
