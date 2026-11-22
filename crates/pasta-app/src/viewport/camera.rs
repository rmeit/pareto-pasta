//! Turntable orbit camera: yaw/pitch/distance around a target point, relative to a world up vector.

use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Camera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Orthonormal frame: `a`, `b` span the horizontal plane, `up` is world up.
    pub a: Vec3,
    pub b: Vec3,
    pub up: Vec3,
}

/// Camera-space basis vectors and eye position for the current state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraFrame {
    pub eye: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

impl Camera {
    pub fn new(up: Vec3) -> Self {
        let mut c = Self {
            target: Vec3::ZERO,
            yaw: -60f32.to_radians(),
            pitch: 28f32.to_radians(),
            distance: 100.0,
            fov_y: 40f32.to_radians(),
            a: Vec3::X,
            b: Vec3::Y,
            up: Vec3::Z,
        };
        c.set_up(up);
        c
    }

    /// Rebuild the horizontal frame from a new up vector. `a` is the world axis least aligned
    /// with `up` (Gram–Schmidt), `b = up × a`.
    pub fn set_up(&mut self, up: Vec3) {
        let up = up.try_normalize().unwrap_or(Vec3::Z);
        let axes = [Vec3::X, Vec3::Y, Vec3::Z];
        let mut best = axes[0];
        let mut best_dot = f32::INFINITY;
        for ax in axes {
            let d = ax.dot(up).abs();
            if d < best_dot - 1e-6 {
                best_dot = d;
                best = ax;
            }
        }
        let a = (best - up * best.dot(up)).normalize();
        let b = up.cross(a);
        self.a = a;
        self.b = b;
        self.up = up;
    }

    pub fn frame(&self) -> CameraFrame {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let horiz = self.a * cy + self.b * sy;
        let dir = horiz * cp + self.up * sp; // from target toward eye
        let eye = self.target + dir * self.distance;
        let forward = -dir;
        let right = self.b * cy - self.a * sy;
        let cam_up = right.cross(forward).normalize();
        CameraFrame {
            eye,
            forward,
            right,
            up: cam_up,
        }
    }

    pub fn view(&self) -> Mat4 {
        let f = self.frame();
        glam::camera::rh::view::look_to_mat4(f.eye, f.forward, f.up)
    }

    /// Perspective projection (wgpu depth range 0..1) with near/far fitted to a bounding sphere.
    pub fn proj(&self, aspect: f32, scene_center: Vec3, scene_radius: f32) -> Mat4 {
        let f = self.frame();
        let r = scene_radius.max(1e-3);
        let d = (scene_center - f.eye).dot(f.forward);
        let far = (d + r).max(r * 0.01) * 1.05;
        let near = (d - r).max(far * 1e-4).max(1e-4) * 0.9;
        glam::camera::rh::proj::directx::perspective(self.fov_y, aspect.max(1e-3), near, far)
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx;
        let lim = std::f32::consts::FRAC_PI_2;
        self.pitch = (self.pitch + dy).clamp(-lim, lim);
    }

    /// Pan by a screen delta in points, given the viewport height in points.
    pub fn pan(&mut self, dx: f32, dy: f32, height: f32) {
        let f = self.frame();
        let scale = 2.0 * self.distance * (self.fov_y * 0.5).tan() / height.max(1.0);
        self.target += (-f.right * dx + f.up * dy) * scale;
    }

    /// World-space ray direction through normalised device coordinates (x, y in -1..1, y up).
    pub fn ray_dir(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> Vec3 {
        let f = self.frame();
        let t = (self.fov_y * 0.5).tan();
        (f.forward + f.right * (ndc_x * t * aspect) + f.up * (ndc_y * t)).normalize()
    }

    /// Zoom by `factor` (<1 = closer) keeping the point under the cursor fixed on screen.
    pub fn zoom_at(&mut self, factor: f32, ndc_x: f32, ndc_y: f32, aspect: f32) {
        let f = self.frame();
        let dir = self.ray_dir(ndc_x, ndc_y, aspect);
        let denom = dir.dot(f.forward).max(1e-3);
        let p = f.eye + dir * (self.distance / denom);
        self.target = p + (self.target - p) * factor;
        self.distance *= factor;
    }

    /// Frame a point set from the current viewing direction: the eye is moved to the tightest
    /// position where every point projects inside the view (with a small margin), centred on
    /// the projected extents. `center` is a reference point (e.g. the bounds centre).
    pub fn fit_points(&mut self, center: Vec3, points: impl Iterator<Item = Vec3>, aspect: f32) {
        let f = self.frame();
        let margin = 0.92;
        let tan_v = (self.fov_y * 0.5).tan() * margin;
        let tan_h = tan_v * aspect.max(1e-3);
        // In camera coordinates (x right, y up, z forward) relative to `center`, a point is
        // inside the horizontal frustum of an eye at (xe, ze) iff
        //   x - tan_h z <= xe - tan_h ze   and   x + tan_h z >= xe + tan_h ze.
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::INFINITY);
        let (mut c, mut d) = (f32::NEG_INFINITY, f32::INFINITY);
        let mut extent = 0.0f32;
        let mut any = false;
        for p in points {
            let r = p - center;
            let (x, y, z) = (r.dot(f.right), r.dot(f.up), r.dot(f.forward));
            a = a.max(x - tan_h * z);
            b = b.min(x + tan_h * z);
            c = c.max(y - tan_v * z);
            d = d.min(y + tan_v * z);
            extent = extent.max(r.length());
            any = true;
        }
        if !any {
            return;
        }
        let xe = 0.5 * (a + b);
        let ye = 0.5 * (c + d);
        let ze = ((b - a) / (2.0 * tan_h)).min((d - c) / (2.0 * tan_v));
        // orbit around the reference depth; keep the eye outside the object
        let dist = (-ze).max(extent * 0.25).max(1e-3);
        self.target = center + f.right * xe + f.up * ye;
        self.distance = dist;
    }
}
