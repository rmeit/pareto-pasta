use crate::sim::SolidResult;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Shape summary of a grown piece.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SolidMeasurements {
    /// Extent along the piece's longest principal axis (mm).
    pub length: f64,
    /// Extents along the other two principal axes (mm).
    pub width: f64,
    pub depth: f64,
    /// Total rotation of the cross-section about the piece's axis, in
    /// turns, positive counter-clockwise looking from the die (along the
    /// extrusion). Measured with the section point farthest from the
    /// section's centre.
    pub twist_turns: f64,
    /// Axial length per full turn (mm), if it turns at all.
    pub pitch: Option<f64>,
    /// Radius of the circle through the start, middle and end of the
    /// piece's centre-line (mm); None if it is straight.
    pub bend_radius: Option<f64>,
}

/// Measures the cut piece in the given positions (`hanging` or `relaxed`).
pub fn measure(r: &SolidResult, positions: &[[f32; 3]]) -> SolidMeasurements {
    let np = r.section.points.len();
    let nc = r.section.n_corners;
    let pos = |i: usize| DVec3::new(positions[i][0] as f64, positions[i][1] as f64, positions[i][2] as f64);
    let pts: Vec<DVec3> = (0..positions.len()).filter(|&i| r.outside[i]).map(pos).collect();
    if pts.len() < 4 {
        return SolidMeasurements::default();
    }
    let (axes, ext) = principal_extents(&pts);
    // Rows wholly outside the die, oldest (the tip) first.
    let levels = positions.len() / np;
    let rows: Vec<usize> =
        (0..levels).step_by(2).filter(|&l| (0..nc).all(|k| r.outside.get(l * np + k).copied().unwrap_or(false))).collect();
    let centre = |l: usize| (0..nc).map(|k| pos(l * np + k)).sum::<DVec3>() / nc as f64;
    // Reference point: farthest from the section centroid.
    let sc = r.section.points[..nc].iter().fold([0.0, 0.0], |a, p| [a[0] + p[0] / nc as f64, a[1] + p[1] / nc as f64]);
    let kref = (0..nc)
        .max_by(|&a, &b| {
            let d = |k: usize| (r.section.points[k][0] - sc[0]).powi(2) + (r.section.points[k][1] - sc[1]).powi(2);
            d(a).total_cmp(&d(b))
        })
        .unwrap();
    let mut twist = 0.0;
    let mut prev: Option<DVec3> = None;
    let cs: Vec<DVec3> = rows.iter().map(|&l| centre(l)).collect();
    for (q, &l) in rows.iter().enumerate() {
        let a = cs[q.saturating_sub(1)];
        let b = cs[(q + 1).min(cs.len() - 1)];
        // Axis pointing back toward the die (the direction of growth).
        let t = (a - b).try_normalize().unwrap_or(DVec3::Z);
        let d = pos(l * np + kref) - cs[q];
        let d = (d - t * d.dot(t)).try_normalize();
        if let (Some(p), Some(d)) = (prev, d) {
            let s = p.cross(d).dot(t);
            twist += s.atan2(p.dot(d));
        }
        prev = d.or(prev);
    }
    let turns = twist / std::f64::consts::TAU;
    let bend_radius = if cs.len() >= 3 {
        let (a, b, c) = (cs[0], cs[cs.len() / 2], cs[cs.len() - 1]);
        let ab = (b - a).length();
        let bc = (c - b).length();
        let ca = (a - c).length();
        let area2 = (b - a).cross(c - a).length();
        (area2 > 1e-9 * ab * bc).then(|| ab * bc * ca / (2.0 * area2)).filter(|r| *r < 1e4 * ext[0].max(1e-9))
    } else {
        None
    };
    let _ = axes;
    SolidMeasurements {
        length: ext[0],
        width: ext[1],
        depth: ext[2],
        twist_turns: turns,
        pitch: (turns.abs() > 0.05).then(|| ext[0] / turns.abs()),
        bend_radius,
    }
}

/// Principal axes (largest variance first) and the extent along each.
fn principal_extents(pts: &[DVec3]) -> ([DVec3; 3], [f64; 3]) {
    let n = pts.len() as f64;
    let c = pts.iter().copied().sum::<DVec3>() / n;
    let mut m = [[0.0f64; 3]; 3];
    for p in pts {
        let d = *p - c;
        let d = [d.x, d.y, d.z];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += d[i] * d[j] / n;
            }
        }
    }
    // Power iteration with deflation is enough for a 3 x 3 covariance.
    let mut axes = [DVec3::X, DVec3::Y, DVec3::Z];
    let mut mm = m;
    for k in 0..3 {
        let mut v = [DVec3::X, DVec3::Y, DVec3::Z][k] + DVec3::new(0.3, 0.2, 0.1);
        for _ in 0..200 {
            let w = DVec3::new(
                mm[0][0] * v.x + mm[0][1] * v.y + mm[0][2] * v.z,
                mm[1][0] * v.x + mm[1][1] * v.y + mm[1][2] * v.z,
                mm[2][0] * v.x + mm[2][1] * v.y + mm[2][2] * v.z,
            );
            let mut w = w;
            for a in axes.iter().take(k) {
                w -= *a * w.dot(*a);
            }
            match w.try_normalize() {
                Some(u) => v = u,
                None => break,
            }
        }
        for a in axes.iter().take(k) {
            v -= *a * v.dot(*a);
        }
        axes[k] = v.try_normalize().unwrap_or(axes[k]);
        let lam = {
            let a = axes[k];
            let w = DVec3::new(
                mm[0][0] * a.x + mm[0][1] * a.y + mm[0][2] * a.z,
                mm[1][0] * a.x + mm[1][1] * a.y + mm[1][2] * a.z,
                mm[2][0] * a.x + mm[2][1] * a.y + mm[2][2] * a.z,
            );
            w.dot(a)
        };
        let a = [axes[k].x, axes[k].y, axes[k].z];
        for i in 0..3 {
            for j in 0..3 {
                mm[i][j] -= lam * a[i] * a[j];
            }
        }
    }
    let ext = axes.map(|a| {
        let (lo, hi) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
            let s = p.dot(a);
            (lo.min(s), hi.max(s))
        });
        hi - lo
    });
    (axes, ext)
}
