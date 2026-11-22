//! Reduction of the speed map to a profile along each slot's centre-line,
//! and the readouts that follow from it.

use crate::FlowSolution;
use pasta_die::TriMesh2;
use serde::{Deserialize, Serialize};

/// Speed and geometry along one slot, after die swell.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlotProfile {
    pub opening: usize,
    pub closed: bool,
    pub length: f64,
    /// Arc length of each station (mm).
    pub s: Vec<f64>,
    pub centre: Vec<[f64; 2]>,
    pub normal: Vec<[f64; 2]>,
    /// Signed centre-line curvature (1/mm).
    pub curvature: Vec<f64>,
    /// Die gap (mm), before swell.
    pub gap: Vec<f64>,
    /// Sheet thickness after swell (mm).
    pub thickness: Vec<f64>,
    /// Mean speed across the gap after swell (mm/s).
    pub v: Vec<f64>,
    /// Speed bias across the gap: w ≈ v (1 + b t), t in [-1, 1] from the
    /// -normal wall to the +normal wall.
    pub bias: Vec<f64>,
    /// Natural curvature along the extrusion direction from the bias
    /// (1/mm), positive when the +normal face is faster.
    pub bend_curvature: Vec<f64>,
}

/// Profiles of every slot opening. Polygon openings are skipped.
pub fn profiles(mesh: &TriMesh2, sol: &FlowSolution, swell: f64) -> Vec<SlotProfile> {
    let swell = swell.max(1e-6);
    let mut out = Vec::new();
    for (k, om) in mesh.openings.iter().enumerate() {
        let Some(g) = &om.slot else { continue };
        let nl = g.n_layers;
        let mut v = Vec::with_capacity(g.n_stations);
        let mut bias = Vec::with_capacity(g.n_stations);
        for i in 0..g.n_stations {
            // Trapezoid rule across the gap in t.
            let mut int_w = 0.0;
            let mut int_wt = 0.0;
            for j in 0..nl - 1 {
                let (t0, t1) = (g.layer_t(j), g.layer_t(j + 1));
                let (w0, w1) = (sol.w[g.node(i, j)], sol.w[g.node(i, j + 1)]);
                let dt = t1 - t0;
                int_w += 0.5 * dt * (w0 + w1);
                // Exact for w linear on the interval.
                int_wt += dt * (w0 * (2.0 * t0 + t1) + w1 * (t0 + 2.0 * t1)) / 6.0;
            }
            let mean = 0.5 * int_w;
            v.push(mean / swell);
            // w = v (1 + b t) gives ∫ w t dt = 2 v b / 3.
            bias.push(if mean.abs() > 1e-300 { 1.5 * int_wt / mean } else { 0.0 });
        }
        let thickness: Vec<f64> = g.gap.iter().map(|h| h * swell).collect();
        let bend_curvature = bias.iter().zip(&thickness).map(|(b, h)| 2.0 * b / h).collect();
        out.push(SlotProfile {
            opening: k,
            closed: g.closed,
            length: g.length,
            s: g.s.clone(),
            centre: g.centre.clone(),
            normal: g.normal.clone(),
            curvature: g.curvature.clone(),
            gap: g.gap.clone(),
            thickness,
            v,
            bias,
            bend_curvature,
        });
    }
    out
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TubeBend {
    /// Relative speed variation around the tube (first Fourier mode).
    pub amplitude: f64,
    /// Bend curvature (1/mm): amplitude over tube radius.
    pub curvature: f64,
    /// Unit vector in the die plane from the tube's axis toward its fast
    /// side. The tube bends the other way.
    pub fast_side: [f64; 2],
}

/// Readouts of one slot profile. See the Tier 0 plan for their meaning.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Readouts {
    pub opening: usize,
    /// Edge speed (peak in the outer quarter of each side) over centre
    /// speed, for open slots.
    pub edge_centre_ratio: Option<f64>,
    /// Fastest over slowest station speed.
    pub max_min_ratio: f64,
    /// In-plane curl radius v / |v'| from a straight-line fit (mm), with the
    /// fit's R^2.
    pub curl_radius: Option<(f64, f64)>,
    /// Bending radius from the speed bias across the gap (mm).
    pub bend_radius: Option<f64>,
    pub tube: Option<TubeBend>,
    /// −v''/v along the slot (1/mm^2): negative wants to ruffle.
    pub ruffle_map: Vec<f64>,
    /// Edge excess over the buckling threshold 2.51 (H/W)^2.
    pub ruffle_margin: Option<f64>,
    /// Width of the fast band at the edges (mm).
    pub fast_band: Option<f64>,
    /// Wavelength estimate C h^0.4 l^0.6 (mm).
    pub wavelength: Option<f64>,
    /// Amplitude estimate (λ/π) √(ratio − 1) (mm).
    pub amplitude: Option<f64>,
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

/// `ruffle_c` is the wavelength prefactor C, to be fitted to real pasta.
pub fn readouts(p: &SlotProfile, ruffle_c: f64) -> Readouts {
    let n = p.v.len();
    let vmax = p.v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let vmin = p.v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_min_ratio = vmax / vmin.max(1e-300);

    // Smoothed second derivative for the ruffle map.
    let ds = p.length / if p.closed { n as f64 } else { (n - 1).max(1) as f64 };
    let idx = |i: isize| -> usize {
        if p.closed { i.rem_euclid(n as isize) as usize } else { i.clamp(0, n as isize - 1) as usize }
    };
    let win = 2isize;
    let smooth: Vec<f64> = (0..n as isize)
        .map(|i| (-win..=win).map(|o| p.v[idx(i + o)]).sum::<f64>() / (2 * win + 1) as f64)
        .collect();
    let ruffle_map: Vec<f64> = (0..n as isize)
        .map(|i| {
            let d2 = (smooth[idx(i + 1)] - 2.0 * smooth[idx(i)] + smooth[idx(i - 1)]) / (ds * ds);
            -d2 / smooth[idx(i)].max(1e-300)
        })
        .collect();

    let kb = mean(&p.bend_curvature);
    let bend_radius = if kb.abs() > 1e-9 { Some(1.0 / kb.abs()) } else { None };

    let mut r = Readouts {
        opening: p.opening,
        edge_centre_ratio: None,
        max_min_ratio,
        curl_radius: None,
        bend_radius,
        tube: None,
        ruffle_map,
        ruffle_margin: None,
        fast_band: None,
        wavelength: None,
        amplitude: None,
    };

    if p.closed {
        let (mut c, mut s) = (0.0, 0.0);
        for i in 0..n {
            let th = std::f64::consts::TAU * p.s[i] / p.length;
            c += p.v[i] * th.cos();
            s += p.v[i] * th.sin();
        }
        let vbar = mean(&p.v);
        let a = 2.0 * (c * c + s * s).sqrt() / (n as f64 * vbar);
        let radius = p.length / std::f64::consts::TAU;
        // Station nearest the fast phase.
        let phase = s.atan2(c);
        let i_fast = (0..n)
            .max_by(|&x, &y| {
                let cx = (std::f64::consts::TAU * p.s[x] / p.length - phase).cos();
                let cy = (std::f64::consts::TAU * p.s[y] / p.length - phase).cos();
                cx.total_cmp(&cy)
            })
            .unwrap_or(0);
        let cx = mean(&p.centre.iter().map(|q| q[0]).collect::<Vec<_>>());
        let cy = mean(&p.centre.iter().map(|q| q[1]).collect::<Vec<_>>());
        let d = [p.centre[i_fast][0] - cx, p.centre[i_fast][1] - cy];
        let dl = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-300);
        r.tube = Some(TubeBend { amplitude: a, curvature: a / radius, fast_side: [d[0] / dl, d[1] / dl] });
        return r;
    }

    // Open slot. The edge speed is the peak in the outer quarter of each
    // side: the very last station can be slowed by the end wall.
    let quarter = (n / 4).max(1);
    let centre_n = ((n as f64 * 0.10).ceil() as usize).max(1);
    let c0 = n / 2 - centre_n.min(n / 2) / 2;
    let v_centre = mean(&p.v[c0..(c0 + centre_n).min(n)]);
    let peak = |range: std::ops::Range<usize>| -> (usize, f64) {
        range.map(|i| (i, p.v[i])).fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a })
    };
    let (il, v_left) = peak(0..quarter);
    let (ir, v_right) = peak(n - quarter..n);
    let ratio = 0.5 * (v_left + v_right) / v_centre.max(1e-300);
    r.edge_centre_ratio = Some(ratio);

    // Straight-line fit v = a + b s.
    let sm = mean(&p.s);
    let vm = mean(&p.v);
    let sxx: f64 = p.s.iter().map(|s| (s - sm).powi(2)).sum();
    let sxy: f64 = p.s.iter().zip(&p.v).map(|(s, v)| (s - sm) * (v - vm)).sum();
    let syy: f64 = p.v.iter().map(|v| (v - vm).powi(2)).sum();
    if sxx > 0.0 && syy > 0.0 {
        let b = sxy / sxx;
        let r2 = sxy * sxy / (sxx * syy);
        if b.abs() > 1e-12 {
            r.curl_radius = Some((vm / b.abs(), r2));
        }
    }

    let excess = ratio - 1.0;
    if excess > 0.0 {
        let h_c = p.thickness[n / 2];
        let half_w = 0.5 * p.length;
        let threshold = 2.51 * (h_c / half_w).powi(2);
        r.ruffle_margin = Some(excess / threshold);
        // Fast band: the run of stations around each edge peak that is
        // faster than the midpoint between centre and peak speed.
        let band_width = |ip: usize, vp: f64| -> usize {
            let mid = 0.5 * (v_centre + vp);
            let mut lo = ip;
            while lo > 0 && p.v[lo - 1] > mid {
                lo -= 1;
            }
            let mut hi = ip;
            while hi + 1 < n && p.v[hi + 1] > mid {
                hi += 1;
            }
            hi - lo + 1
        };
        let band = 0.5 * (band_width(il, v_left) + band_width(ir, v_right)) as f64 * ds;
        if band > 0.0 {
            let h_e = 0.5 * (p.thickness[il] + p.thickness[ir]);
            let lambda = ruffle_c * h_e.powf(0.4) * band.powf(0.6);
            r.fast_band = Some(band);
            r.wavelength = Some(lambda);
            r.amplitude = Some(lambda / std::f64::consts::PI * excess.sqrt());
        }
    }
    r
}
