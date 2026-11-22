//! The die canvas: the opening drawn to scale, coloured by exit speed, with
//! draggable control points and a probe under the cursor.

use crate::viewport::{ColorMap, colormap_srgb};
use egui::{Color32, Pos2, Rect, Sense, Shape, Stroke, Vec2, epaint::Mesh};
use pasta_sim::die::{Opening, TriMesh2};
use pasta_sim::flow::FlowSolution;
use pasta_sim::{DieOutput, Project};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Handle {
    /// Control point k of slot opening o.
    Slot(usize, usize),
    /// Outline vertex k of polygon opening o.
    Poly(usize, usize),
}

pub struct DieCanvas {
    /// Display-only widening of slots across their gap.
    pub exaggerate: f64,
    drag: Option<Handle>,
    hover: Option<Handle>,
    /// Extra zoom on top of fit-to-bounds, and pan in mm.
    zoom: f32,
    pan: Vec2,
    /// Bounds used for the view, frozen while dragging so the view does
    /// not shift under the cursor.
    frozen_bounds: Option<([f64; 2], [f64; 2])>,
}

impl Default for DieCanvas {
    fn default() -> Self {
        Self { exaggerate: 4.0, drag: None, hover: None, zoom: 1.0, pan: Vec2::ZERO, frozen_bounds: None }
    }
}

/// Node positions as drawn: slot nodes pushed away from the centre-line by
/// the exaggeration factor.
fn display_nodes(mesh: &TriMesh2, ex: f64) -> Vec<[f64; 2]> {
    let mut out = mesh.nodes.clone();
    if ex == 1.0 {
        return out;
    }
    for om in &mesh.openings {
        if let Some(g) = &om.slot {
            for i in 0..g.n_stations {
                let c = g.centre[i];
                for j in 0..g.n_layers {
                    let k = g.node(i, j);
                    let p = mesh.nodes[k];
                    out[k] = [c[0] + ex * (p[0] - c[0]), c[1] + ex * (p[1] - c[1])];
                }
            }
        }
    }
    out
}

/// The die with its slot gaps widened for display.
fn display_die(die: &pasta_sim::die::Die, ex: f64) -> pasta_sim::die::Die {
    let mut d = die.clone();
    if ex != 1.0 {
        for o in &mut d.openings {
            if let Opening::Slot(s) = o {
                for p in &mut s.gap.points {
                    p[1] *= ex;
                }
            }
        }
    }
    d
}

/// Maps die millimetres (y up) to screen points.
#[derive(Clone, Copy)]
struct Xform {
    centre: [f64; 2],
    scale: f32,
    origin: Pos2,
}

impl Xform {
    fn to_screen(&self, p: [f64; 2]) -> Pos2 {
        Pos2::new(
            self.origin.x + ((p[0] - self.centre[0]) as f32) * self.scale,
            self.origin.y - ((p[1] - self.centre[1]) as f32) * self.scale,
        )
    }

    fn to_die(&self, p: Pos2) -> [f64; 2] {
        [
            self.centre[0] + ((p.x - self.origin.x) / self.scale) as f64,
            self.centre[1] - ((p.y - self.origin.y) / self.scale) as f64,
        ]
    }
}

pub fn speed_color(t: f32) -> Color32 {
    let c = colormap_srgb(ColorMap::Viridis, t.clamp(0.0, 1.0));
    Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
}

impl DieCanvas {
    /// Draws the canvas. Returns (probe text, clicked opening).
    pub fn ui(&mut self, ui: &mut egui::Ui, project: &mut Project, die: Option<&DieOutput>, height: f32) -> (Option<String>, Option<usize>) {
        let (resp, painter) = ui.allocate_painter(Vec2::new(ui.available_width(), height), Sense::click_and_drag());
        let rect = resp.rect;
        let visuals = ui.visuals().clone();
        painter.rect_filled(rect, 6.0, visuals.extreme_bg_color);
        // View transform.
        let bounds = || display_die(&project.die, self.exaggerate.max(1.0)).bounds();
        let (lo, hi) = if self.drag.is_some() { self.frozen_bounds.unwrap_or_else(bounds) } else { bounds() };
        if self.drag.is_none() {
            self.frozen_bounds = Some((lo, hi));
        }
        let span = [(hi[0] - lo[0]).max(1.0), (hi[1] - lo[1]).max(1.0)];
        let fit = ((rect.width() - 48.0) / span[0] as f32).min((rect.height() - 48.0) / span[1] as f32).max(1e-3);
        let xf = Xform {
            centre: [0.5 * (lo[0] + hi[0]) - self.pan.x as f64, 0.5 * (lo[1] + hi[1]) - self.pan.y as f64],
            scale: fit * self.zoom,
            origin: rect.center(),
        };
        // Zoom with the scroll wheel, pan with the right button.
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta().y);
            if scroll != 0.0 {
                self.zoom = (self.zoom * (1.0 + scroll * 0.002)).clamp(0.2, 40.0);
            }
        }
        if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
            let d = resp.drag_delta();
            self.pan += Vec2::new(d.x / xf.scale, -d.y / xf.scale);
        }
        if resp.double_clicked() {
            self.zoom = 1.0;
            self.pan = Vec2::ZERO;
        }

        // Heat map.
        let mut probe = None;
        let mut clicked_opening = None;
        let ex = self.exaggerate.max(1.0);
        let shown_die = display_die(&project.die, ex);
        if let Some(d) = die {
            let vmax = d.flow.w.iter().cloned().fold(0.0, f64::max).max(1e-12);
            let nodes = display_nodes(&d.mesh, ex);
            let mut mesh = Mesh::default();
            for (k, p) in nodes.iter().enumerate() {
                mesh.colored_vertex(xf.to_screen(*p), speed_color((d.flow.w[k] / vmax) as f32));
            }
            for t in &d.mesh.tris {
                mesh.add_triangle(t[0], t[1], t[2]);
            }
            painter.with_clip_rect(rect).add(Shape::mesh(mesh));
            // Probe.
            if let Some(pos) = resp.hover_pos() {
                let p = xf.to_die(pos);
                if let Some((t, bary)) = find_triangle(&d.mesh, &nodes, p) {
                    // Same barycentric weights in the true geometry.
                    let tri = d.mesh.tris[t].map(|i| d.mesh.nodes[i as usize]);
                    let q = [
                        bary[0] * tri[0][0] + bary[1] * tri[1][0] + bary[2] * tri[2][0],
                        bary[0] * tri[0][1] + bary[1] * tri[1][1] + bary[2] * tri[2][1],
                    ];
                    probe = Some(probe_text(&d.mesh, &d.flow, t, q, project));
                    if resp.clicked() {
                        clicked_opening = Some(d.mesh.tri_opening[t] as usize);
                    }
                }
            }
            // Colour key.
            let key = Rect::from_min_size(rect.left_bottom() + Vec2::new(12.0, -22.0), Vec2::new(120.0, 8.0));
            let mut bar = Mesh::default();
            for k in 0..=16 {
                let t = k as f32 / 16.0;
                let x = key.left() + t * key.width();
                let c = speed_color(t);
                bar.colored_vertex(Pos2::new(x, key.top()), c);
                bar.colored_vertex(Pos2::new(x, key.bottom()), c);
                if k > 0 {
                    let i = (2 * k) as u32;
                    bar.add_triangle(i - 2, i - 1, i);
                    bar.add_triangle(i - 1, i + 1, i);
                }
            }
            painter.add(Shape::mesh(bar));
            let font = egui::FontId::proportional(11.0);
            painter.text(key.left_top() + Vec2::new(0.0, -2.0), egui::Align2::LEFT_BOTTOM, "0", font.clone(), visuals.weak_text_color());
            painter.text(key.right_top() + Vec2::new(0.0, -2.0), egui::Align2::RIGHT_BOTTOM, format!("{vmax:.1} mm/s"), font, visuals.weak_text_color());
        }

        // Outlines, highlighting the opening that grows.
        for (k, poly) in shown_die.outlines(256) {
            let pts: Vec<Pos2> = poly.iter().map(|p| xf.to_screen(*p)).collect();
            let col = if k == project.opening { visuals.strong_text_color() } else { visuals.weak_text_color() };
            painter.with_clip_rect(rect).add(Shape::closed_line(pts, Stroke::new(if k == project.opening { 1.5 } else { 1.0 }, col)));
        }

        // Handles.
        let accent = visuals.selection.stroke.color;
        let mut handles = Vec::new();
        for (o, op) in project.die.openings.iter().enumerate() {
            match op {
                Opening::Slot(s) => {
                    for (k, p) in s.centerline.iter().enumerate() {
                        handles.push((Handle::Slot(o, k), *p));
                    }
                }
                Opening::Polygon(p) => {
                    if p.outer.len() <= 16 {
                        for (k, q) in p.outer.iter().enumerate() {
                            handles.push((Handle::Poly(o, k), *q));
                        }
                    }
                }
            }
        }
        let pointer = resp.hover_pos();
        self.hover = None;
        if let Some(pos) = pointer {
            let mut best = 9.0f32;
            for (h, p) in &handles {
                let d = xf.to_screen(*p).distance(pos);
                if d < best {
                    best = d;
                    self.hover = Some(*h);
                }
            }
        }
        if resp.drag_started_by(egui::PointerButton::Primary) {
            self.drag = self.hover;
        }
        if resp.drag_stopped() {
            self.drag = None;
        }
        if let (Some(h), Some(pos)) = (self.drag, resp.interact_pointer_pos())
            && resp.dragged_by(egui::PointerButton::Primary) {
                let mut p = xf.to_die(pos);
                // Snap to 0.05 mm unless Alt is held.
                if !ui.input(|i| i.modifiers.alt) {
                    p = p.map(|v| (v / 0.05).round() * 0.05);
                }
                match h {
                    Handle::Slot(o, k) => {
                        if let Some(Opening::Slot(s)) = project.die.openings.get_mut(o) {
                            s.centerline[k] = p;
                        }
                    }
                    Handle::Poly(o, k) => {
                        if let Some(Opening::Polygon(q)) = project.die.openings.get_mut(o) {
                            q.outer[k] = p;
                        }
                    }
                }
            }
        for (h, p) in &handles {
            let c = xf.to_screen(*p);
            let active = Some(*h) == self.drag || Some(*h) == self.hover;
            painter.circle(c, if active { 5.5 } else { 4.0 }, if active { accent } else { visuals.panel_fill }, Stroke::new(1.5, accent));
        }
        if self.hover.is_some() || self.drag.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
        // Scale bar.
        let mm = nice_length(60.0 / xf.scale);
        let a = rect.right_bottom() + Vec2::new(-16.0 - mm as f32 * xf.scale, -18.0);
        let b = rect.right_bottom() + Vec2::new(-16.0, -18.0);
        painter.line_segment([a, b], Stroke::new(1.5, visuals.text_color()));
        painter.text(a + Vec2::new(0.0, -4.0), egui::Align2::LEFT_BOTTOM, format!("{mm} mm"), egui::FontId::proportional(11.0), visuals.text_color());
        (probe, clicked_opening)
    }
}

fn nice_length(target: f32) -> f64 {
    let t = target.max(1e-6) as f64;
    let p = 10f64.powf(t.log10().floor());
    for m in [1.0, 2.0, 5.0, 10.0] {
        if m * p >= t {
            return m * p;
        }
    }
    10.0 * p
}

/// Triangle containing p in the given node positions, with barycentric
/// weights.
fn find_triangle(mesh: &TriMesh2, nodes: &[[f64; 2]], p: [f64; 2]) -> Option<(usize, [f64; 3])> {
    mesh.tris.iter().enumerate().find_map(|(k, t)| {
        let [a, b, c] = t.map(|i| nodes[i as usize]);
        let det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if det == 0.0 {
            return None;
        }
        let l1 = ((p[0] - a[0]) * (c[1] - a[1]) - (p[1] - a[1]) * (c[0] - a[0])) / det;
        let l2 = ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / det;
        let l0 = 1.0 - l1 - l2;
        (l0 >= -1e-9 && l1 >= -1e-9 && l2 >= -1e-9).then_some((k, [l0, l1, l2]))
    })
}

fn probe_text(mesh: &TriMesh2, flow: &FlowSolution, t: usize, p: [f64; 2], project: &Project) -> String {
    let w = flow.speed_in_triangle(mesh, t, p);
    let k = mesh.tri_opening[t] as usize;
    let mut s = format!(
        "Opening {}\nSpeed {:.2} mm/s ({:.2}× the opening's mean)\nShear rate {:.1} /s\nLand depth {:.2} mm",
        k + 1,
        w,
        w / flow.mean_speed[k].max(1e-12),
        flow.tri_shear[t],
        mesh.tri_land[t]
    );
    if let Some(g) = &mesh.openings[k].slot {
        // Nearest station for the local gap.
        let i = (0..g.n_stations)
            .min_by(|&a, &b| {
                let da = (g.centre[a][0] - p[0]).powi(2) + (g.centre[a][1] - p[1]).powi(2);
                let db = (g.centre[b][0] - p[0]).powi(2) + (g.centre[b][1] - p[1]).powi(2);
                da.total_cmp(&db)
            })
            .unwrap_or(0);
        s.push_str(&format!("\nGap {:.3} mm at {:.2} mm along the slot", g.gap[i], g.s[i]));
    }
    let _ = project;
    s
}
