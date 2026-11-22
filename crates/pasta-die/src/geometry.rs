use serde::{Deserialize, Serialize};

/// A piecewise-linear function of fractional arc length u in [0, 1].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    /// (u, value) pairs sorted by u. Values outside the first and last u
    /// are held constant.
    pub points: Vec<[f64; 2]>,
}

impl Profile {
    pub fn constant(v: f64) -> Self {
        Self { points: vec![[0.0, v], [1.0, v]] }
    }

    pub fn from_points(mut points: Vec<[f64; 2]>) -> Self {
        points.sort_by(|a, b| a[0].total_cmp(&b[0]));
        Self { points }
    }

    pub fn eval(&self, u: f64) -> f64 {
        let p = &self.points;
        match p.len() {
            0 => 0.0,
            1 => p[0][1],
            _ => {
                if u <= p[0][0] {
                    return p[0][1];
                }
                for w in p.windows(2) {
                    if u <= w[1][0] {
                        let du = w[1][0] - w[0][0];
                        if du <= 0.0 {
                            return w[1][1];
                        }
                        let t = (u - w[0][0]) / du;
                        return w[0][1] + t * (w[1][1] - w[0][1]);
                    }
                }
                p[p.len() - 1][1]
            }
        }
    }

    pub fn min(&self) -> f64 {
        self.points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.points.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max)
    }
}

/// A centre-line resampled at equal arc-length spacing.
///
/// Open: `n + 1` stations from one end to the other. Closed: `n` stations,
/// the last joining back to the first.
#[derive(Clone, Debug)]
pub struct Centerline {
    pts: Vec<[f64; 2]>,
    tangents: Vec<[f64; 2]>,
    curvature: Vec<f64>,
    s: Vec<f64>,
    length: f64,
    closed: bool,
}

impl Centerline {
    /// Builds a Catmull-Rom spline through `control` and resamples it with
    /// `n` segments.
    pub fn new(control: &[[f64; 2]], closed: bool, n: usize) -> Self {
        let n = n.max(2);
        let dense = spline(control, closed, 64);
        // Cumulative arc length of the dense polyline.
        let mut cum = vec![0.0];
        let m = dense.len();
        let seg_count = if closed { m } else { m - 1 };
        for k in 0..seg_count {
            let a = dense[k];
            let b = dense[(k + 1) % m];
            cum.push(cum[k] + dist(a, b));
        }
        let length = *cum.last().unwrap();
        let count = if closed { n } else { n + 1 };
        let mut pts = Vec::with_capacity(count);
        let mut s = Vec::with_capacity(count);
        let mut k = 0;
        for i in 0..count {
            let target = length * i as f64 / n as f64;
            while k + 1 < cum.len() - 1 && cum[k + 1] < target {
                k += 1;
            }
            let seg = (cum[k + 1] - cum[k]).max(1e-300);
            let t = ((target - cum[k]) / seg).clamp(0.0, 1.0);
            let a = dense[k];
            let b = dense[(k + 1) % m];
            pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
            s.push(target);
        }
        // Tangents and curvature by finite differences on the resampled
        // points.
        let len = pts.len();
        let mut tangents = vec![[1.0, 0.0]; len];
        let mut curvature = vec![0.0; len];
        for i in 0..len {
            let (prev, next) = if closed {
                ((i + len - 1) % len, (i + 1) % len)
            } else {
                (i.saturating_sub(1), (i + 1).min(len - 1))
            };
            let d = [pts[next][0] - pts[prev][0], pts[next][1] - pts[prev][1]];
            let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-300);
            tangents[i] = [d[0] / l, d[1] / l];
        }
        for i in 0..len {
            let (prev, next) = if closed {
                ((i + len - 1) % len, (i + 1) % len)
            } else {
                (i.saturating_sub(1), (i + 1).min(len - 1))
            };
            if prev == i || next == i {
                continue;
            }
            let ta = tangents[prev];
            let tb = tangents[next];
            let cross = ta[0] * tb[1] - ta[1] * tb[0];
            let dot = ta[0] * tb[0] + ta[1] * tb[1];
            let angle = cross.atan2(dot);
            let ds = dist(pts[prev], pts[i]) + dist(pts[i], pts[next]);
            curvature[i] = angle / ds.max(1e-300);
        }
        if !closed && len > 2 {
            curvature[0] = curvature[1];
            curvature[len - 1] = curvature[len - 2];
        }
        Self { pts, tangents, curvature, s, length, closed }
    }

    pub fn len(&self) -> usize {
        self.pts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    pub fn closed(&self) -> bool {
        self.closed
    }

    pub fn length(&self) -> f64 {
        self.length
    }

    pub fn point(&self, i: usize) -> [f64; 2] {
        self.pts[i]
    }

    pub fn tangent(&self, i: usize) -> [f64; 2] {
        self.tangents[i]
    }

    /// Left normal: the tangent turned 90 degrees anticlockwise.
    pub fn normal(&self, i: usize) -> [f64; 2] {
        let t = self.tangents[i];
        [-t[1], t[0]]
    }

    /// Signed curvature (1/mm), positive when turning left.
    pub fn curvature(&self, i: usize) -> f64 {
        self.curvature[i]
    }

    pub fn s(&self, i: usize) -> f64 {
        self.s[i]
    }

    /// Fractional arc length of station i.
    pub fn u(&self, i: usize) -> f64 {
        if self.length > 0.0 { self.s[i] / self.length } else { 0.0 }
    }
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Interpolating cubic spline through the control points, parametrised by
/// chord length. Natural ends when open, periodic when closed. It is C2, so
/// curvature is continuous: kinks would otherwise show up in the pasta's
/// rest shape.
fn spline(c: &[[f64; 2]], closed: bool, per_segment: usize) -> Vec<[f64; 2]> {
    let n = c.len();
    if n < 2 || (closed && n < 3) {
        return c.to_vec();
    }
    let segs = if closed { n } else { n - 1 };
    // Knot spacing.
    let h: Vec<f64> = (0..segs).map(|k| dist(c[k], c[(k + 1) % n]).max(1e-12)).collect();
    // Second derivatives M at each knot, per coordinate.
    let mut second = vec![[0.0f64; 2]; n];
    if closed || n > 2 {
        let size = if closed { n } else { n - 2 };
        let mut a = vec![0.0f64; size * size];
        let mut rhs = vec![[0.0f64; 2]; size];
        for r in 0..size {
            let i = if closed { r } else { r + 1 }; // knot index
            let hp = h[(i + segs - 1) % segs]; // segment before knot i
            let hn = h[i % segs]; // segment after knot i
            let prev = (i + n - 1) % n;
            let next = (i + 1) % n;
            let col = |k: usize| -> Option<usize> {
                if closed { Some(k) } else if k >= 1 && k <= n - 2 { Some(k - 1) } else { None }
            };
            a[r * size + r] += 2.0 * (hp + hn);
            if let Some(cp) = col(prev) {
                a[r * size + cp] += hp;
            }
            if let Some(cn) = col(next) {
                a[r * size + cn] += hn;
            }
            for d in 0..2 {
                rhs[r][d] = 6.0 * ((c[next][d] - c[i][d]) / hn - (c[i][d] - c[prev][d]) / hp);
            }
        }
        let sol = solve_dense(a, rhs, size);
        for r in 0..size {
            let i = if closed { r } else { r + 1 };
            second[i] = sol[r];
        }
    }
    let mut out = Vec::with_capacity(segs * per_segment + 1);
    for k in 0..segs {
        let k1 = (k + 1) % n;
        let hk = h[k];
        for j in 0..per_segment {
            let t = hk * j as f64 / per_segment as f64;
            let a = (hk - t) / hk;
            let b = t / hk;
            let mut p = [0.0; 2];
            for d in 0..2 {
                p[d] = a * c[k][d]
                    + b * c[k1][d]
                    + ((a * a * a - a) * second[k][d] + (b * b * b - b) * second[k1][d]) * hk * hk / 6.0;
            }
            out.push(p);
        }
    }
    if !closed {
        out.push(c[n - 1]);
    }
    out
}

/// Gaussian elimination with partial pivoting for small systems with two
/// right-hand sides.
fn solve_dense(mut a: Vec<f64>, mut b: Vec<[f64; 2]>, n: usize) -> Vec<[f64; 2]> {
    for col in 0..n {
        let piv = (col..n).max_by(|&x, &y| a[x * n + col].abs().total_cmp(&a[y * n + col].abs())).unwrap();
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        if d.abs() < 1e-300 {
            continue;
        }
        for r in col + 1..n {
            let f = a[r * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[r * n + k] -= f * a[col * n + k];
            }
            for q in 0..2 {
                b[r][q] -= f * b[col][q];
            }
        }
    }
    for r in (0..n).rev() {
        for q in 0..2 {
            let mut v = b[r][q];
            for k in r + 1..n {
                v -= a[r * n + k] * b[k][q];
            }
            b[r][q] = v / a[r * n + r];
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_line_has_zero_curvature() {
        let c = Centerline::new(&[[0.0, 0.0], [10.0, 0.0]], false, 20);
        assert_eq!(c.len(), 21);
        assert!((c.length() - 10.0).abs() < 1e-9);
        for i in 0..c.len() {
            assert!(c.curvature(i).abs() < 1e-9);
            assert!((c.normal(i)[1] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn circle_curvature() {
        let r = 4.0;
        let ctrl: Vec<_> = (0..16)
            .map(|k| {
                let a = k as f64 * std::f64::consts::TAU / 16.0;
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        let c = Centerline::new(&ctrl, true, 200);
        assert!((c.length() - std::f64::consts::TAU * r).abs() / (std::f64::consts::TAU * r) < 2e-3);
        for i in 0..c.len() {
            assert!((c.curvature(i) - 1.0 / r).abs() < 0.02 / r, "{}", c.curvature(i));
        }
    }

    #[test]
    fn profile_interpolates() {
        let p = Profile::from_points(vec![[0.0, 1.0], [0.5, 2.0], [1.0, 1.0]]);
        assert_eq!(p.eval(0.25), 1.5);
        assert_eq!(p.eval(-1.0), 1.0);
        assert_eq!(p.eval(0.5), 2.0);
    }
}
