//! Statistics of a simulated piece, as the review recommends comparing
//! (wavelength, amplitude, curl) rather than point-by-point shapes.

use crate::ShellResult;
use glam::DVec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EdgeRuffle {
    /// Mean distance between crests (mm). None when the edge is flat.
    pub wavelength: Option<f64>,
    /// Out-of-plane amplitude (mm): √2 × RMS of the detrended signal.
    pub amplitude: f64,
    /// Edge length over centre-line length.
    pub length_ratio: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Measurements {
    /// Length of the piece along its centre-line (mm).
    pub length: f64,
    /// Radius of the centre-line's mean curvature (mm); None when straight.
    pub curl_radius: Option<f64>,
    pub left_edge: Option<EdgeRuffle>,
    pub right_edge: Option<EdgeRuffle>,
}

/// Measures the given shape (hanging or relaxed positions).
pub fn measure(r: &ShellResult, positions: &[[f32; 3]]) -> Measurements {
    let n_s = r.stations;
    let p = |i: usize| -> DVec3 { DVec3::from_array(positions[i].map(|v| v as f64)) };
    // Rows whose nodes are all outside the die.
    let rows: Vec<usize> = (0..positions.len() / n_s)
        .filter(|&row| (0..n_s).all(|i| r.outside.get(row * n_s + i).copied().unwrap_or(false)))
        .collect();
    if rows.len() < 4 {
        return Measurements::default();
    }
    // Centre-line: for an open strip the middle station; for a tube the
    // mean of each ring.
    let centre_of = |row: usize| -> DVec3 {
        if r.closed {
            (0..n_s).map(|i| p(row * n_s + i)).sum::<DVec3>() / n_s as f64
        } else {
            p(row * n_s + n_s / 2)
        }
    };
    let centres: Vec<DVec3> = rows.iter().map(|&row| centre_of(row)).collect();
    let mut length = 0.0;
    let mut s = vec![0.0];
    for w in centres.windows(2) {
        length += (w[1] - w[0]).length();
        s.push(length);
    }
    // Mean curvature from the total turning of the tangent, with tangents
    // taken over a window so the jitter of single nodes does not add up.
    let nc = centres.len();
    let w = (nc / 20).max(2).min(nc / 4).max(1);
    let tangent = |k: usize| (centres[(k + w).min(nc - 1)] - centres[k.saturating_sub(w)]).normalize_or_zero();
    let mut turning = 0.0;
    for k in w..nc - w - 1 {
        turning += tangent(k).dot(tangent(k + 1)).clamp(-1.0, 1.0).acos();
    }
    let span = s[nc - w - 1] - s[w];
    let kappa = turning / span.max(1e-300);
    let curl_radius = if kappa * span > 0.05 { Some(1.0 / kappa) } else { None };

    let mut out = Measurements { length, curl_radius, left_edge: None, right_edge: None };
    if r.closed {
        return out;
    }
    let edge = |station: usize| -> EdgeRuffle {
        let mut sig = Vec::with_capacity(rows.len());
        let mut edge_len = 0.0;
        let mut prev: Option<DVec3> = None;
        for (k, &row) in rows.iter().enumerate() {
            let c = centres[k];
            let e = p(row * n_s + station);
            if let Some(q) = prev {
                edge_len += (e - q).length();
            }
            prev = Some(e);
            let k0 = k.saturating_sub(1);
            let k1 = (k + 1).min(rows.len() - 1);
            let tangent = (centres[k1] - centres[k0]).normalize_or_zero();
            let width = (p(row * n_s + n_s - 1) - p(row * n_s)).normalize_or_zero();
            let normal = tangent.cross(width).normalize_or_zero();
            sig.push((e - c).dot(normal));
        }
        // Remove the slow trend (a quadratic fit in arc length).
        let detrended = detrend_quadratic(&s, &sig);
        let rms = (detrended.iter().map(|v| v * v).sum::<f64>() / detrended.len() as f64).sqrt();
        let amplitude = std::f64::consts::SQRT_2 * rms;
        // Zero crossings with a little hysteresis.
        let band = 0.3 * rms;
        let mut crossings = Vec::new();
        let mut sign = 0i32;
        for (k, v) in detrended.iter().enumerate() {
            let sgn = if *v > band { 1 } else if *v < -band { -1 } else { 0 };
            if sgn != 0 && sign != 0 && sgn != sign {
                crossings.push(s[k]);
            }
            if sgn != 0 {
                sign = sgn;
            }
        }
        let thickness = r.input.thickness[station];
        let wavelength = if crossings.len() >= 3 && amplitude > 0.25 * thickness {
            Some(2.0 * (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64)
        } else {
            None
        };
        EdgeRuffle { wavelength, amplitude, length_ratio: edge_len / length.max(1e-300) }
    };
    out.left_edge = Some(edge(0));
    out.right_edge = Some(edge(n_s - 1));
    out
}

/// Subtracts the least-squares quadratic in x from y.
fn detrend_quadratic(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len().min(y.len());
    let xm = x[..n].iter().sum::<f64>() / n as f64;
    let xs: Vec<f64> = x[..n].iter().map(|v| v - xm).collect();
    // Normal equations for [1, x, x²].
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    for k in 0..n {
        let basis = [1.0, xs[k], xs[k] * xs[k]];
        for i in 0..3 {
            b[i] += basis[i] * y[k];
            for j in 0..3 {
                a[i][j] += basis[i] * basis[j];
            }
        }
    }
    // Gaussian elimination (3x3, well conditioned after centring).
    for c in 0..3 {
        let piv = a[c][c];
        if piv.abs() < 1e-300 {
            return y[..n].to_vec();
        }
        for r in c + 1..3 {
            let f = a[r][c] / piv;
            for k in c..3 {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut coef = [0.0; 3];
    for r in (0..3).rev() {
        let mut v = b[r];
        for k in r + 1..3 {
            v -= a[r][k] * coef[k];
        }
        coef[r] = v / a[r][r];
    }
    (0..n).map(|k| y[k] - (coef[0] + coef[1] * xs[k] + coef[2] * xs[k] * xs[k])).collect()
}
