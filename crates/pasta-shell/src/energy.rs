//! Element energies of the discrete shell, with gradients and positive
//! semi-definite Hessian approximations.
//!
//! Stretching: St Venant–Kirchhoff on each triangle, written in terms of the
//! squared edge-length changes δ_k = |e_k|² − L̄_k² (Delingette's "triangular
//! springs"). The energy is δᵀ M δ, exact for a linear triangle. Its
//! Gauss–Newton Hessian is PSD by construction; the second-order term is
//! kept only for edges in tension, which keeps it PSD.
//!
//! Bending: a hinge energy on each interior edge, k (θ − θ̄)², with the
//! Gauss–Newton Hessian 2k ∇θ ∇θᵀ.

use glam::DVec3;

/// Rest data of one triangle. Edge k is opposite vertex k.
#[derive(Clone, Debug)]
pub struct TriRest {
    pub v: [u32; 3],
    pub rest_len2: [f64; 3],
    pub m: [[f64; 3]; 3],
    pub area: f64,
    pub thickness: f64,
}

/// Edge k joins vertices EDGE[k].
pub const EDGE: [(usize, usize); 3] = [(1, 2), (2, 0), (0, 1)];

impl TriRest {
    /// `mu` and `lambda` are plane-stress Lamé parameters (N/mm²).
    pub fn new(v: [u32; 3], rest_len2: [f64; 3], thickness: f64, mu: f64, lambda: f64) -> Self {
        let mut t = Self { v, rest_len2, m: [[0.0; 3]; 3], area: 0.0, thickness };
        t.update_rest(rest_len2, mu, lambda);
        t
    }

    /// Recomputes M from new rest lengths (used by creep).
    pub fn update_rest(&mut self, rest_len2: [f64; 3], mu: f64, lambda: f64) {
        // Lay the rest triangle flat: v0 at the origin, v1 on the x axis.
        let l0 = rest_len2[0].max(1e-30).sqrt(); // |v1 v2|
        let l1 = rest_len2[1].max(1e-30).sqrt(); // |v2 v0|
        let l2 = rest_len2[2].max(1e-30).sqrt(); // |v0 v1|
        let x2 = ((l1 * l1 + l2 * l2 - l0 * l0) / (2.0 * l2)).clamp(-l1, l1);
        let y2 = (l1 * l1 - x2 * x2).max((1e-3 * l1.min(l2)).powi(2)).sqrt();
        let p = [[0.0, 0.0], [l2, 0.0], [x2, y2]];
        self.area = 0.5 * l2 * y2;
        let e: Vec<[f64; 2]> = EDGE.iter().map(|&(a, b)| [p[b][0] - p[a][0], p[b][1] - p[a][1]]).collect();
        // Solve for symmetric B_k (a, b, c) with e_l^T B_k e_l = 1/2 [l == k].
        let rows: Vec<[f64; 3]> = e.iter().map(|q| [q[0] * q[0], 2.0 * q[0] * q[1], q[1] * q[1]]).collect();
        let inv = invert3(&rows);
        let mut b = [[0.0f64; 3]; 3]; // b[k] = (a, b, c)
        for k in 0..3 {
            for r in 0..3 {
                b[k][r] = 0.5 * inv[r][k];
            }
        }
        let tr = |q: &[f64; 3]| q[0] + q[2];
        let trprod = |p: &[f64; 3], q: &[f64; 3]| p[0] * q[0] + 2.0 * p[1] * q[1] + p[2] * q[2];
        let s = self.thickness * self.area;
        for j in 0..3 {
            for k in 0..3 {
                self.m[j][k] = s * (mu * trprod(&b[j], &b[k]) + 0.5 * lambda * tr(&b[j]) * tr(&b[k]));
            }
        }
        self.rest_len2 = rest_len2;
    }

    fn deltas(&self, x: &[DVec3; 3]) -> ([DVec3; 3], [f64; 3]) {
        let mut e = [DVec3::ZERO; 3];
        let mut d = [0.0; 3];
        for k in 0..3 {
            let (a, b) = EDGE[k];
            e[k] = x[b] - x[a];
            d[k] = e[k].length_squared() - self.rest_len2[k];
        }
        (e, d)
    }

    pub fn energy(&self, x: &[DVec3; 3]) -> f64 {
        let (_, d) = self.deltas(x);
        let mut w = 0.0;
        for j in 0..3 {
            for k in 0..3 {
                w += d[j] * self.m[j][k] * d[k];
            }
        }
        w
    }

    /// Gradient (9) and PSD Hessian (9x9, row-major) w.r.t. the three
    /// vertex positions.
    pub fn grad_hess(&self, x: &[DVec3; 3], g: &mut [f64; 9], h: &mut [f64; 81]) {
        let (e, d) = self.deltas(x);
        let md: [f64; 3] = std::array::from_fn(|j| (0..3).map(|k| self.m[j][k] * d[k]).sum());
        // ∂δ_k/∂x as 9-vectors.
        let mut jd = [[0.0f64; 9]; 3];
        for k in 0..3 {
            let (a, b) = EDGE[k];
            for c in 0..3 {
                jd[k][3 * b + c] = 2.0 * e[k][c];
                jd[k][3 * a + c] = -2.0 * e[k][c];
            }
        }
        g.iter_mut().for_each(|v| *v = 0.0);
        for k in 0..3 {
            for q in 0..9 {
                g[q] += 2.0 * md[k] * jd[k][q];
            }
        }
        h.iter_mut().for_each(|v| *v = 0.0);
        for j in 0..3 {
            for k in 0..3 {
                let c = 2.0 * self.m[j][k];
                if c == 0.0 {
                    continue;
                }
                for p in 0..9 {
                    let jp = jd[j][p];
                    if jp == 0.0 {
                        continue;
                    }
                    for q in 0..9 {
                        h[9 * p + q] += c * jp * jd[k][q];
                    }
                }
            }
        }
        for k in 0..3 {
            // Second-order term 2 (Mδ)_k ∇²δ_k, kept in tension only so the
            // element matrix stays PSD. (Keeping it in compression with a
            // per-element projection, or unprojected with a global shift,
            // was tried: neither was faster overall.)
            let c = 2.0 * md[k];
            if c <= 0.0 {
                continue;
            }
            let (a, b) = EDGE[k];
            for comp in 0..3 {
                let (ia, ib) = (3 * a + comp, 3 * b + comp);
                h[9 * ia + ia] += 2.0 * c;
                h[9 * ib + ib] += 2.0 * c;
                h[9 * ia + ib] -= 2.0 * c;
                h[9 * ib + ia] -= 2.0 * c;
            }
        }
    }
}

fn invert3(a: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let inv_det = 1.0 / det;
    let mut r = [[0.0; 3]; 3];
    r[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
    r[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
    r[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
    r[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det;
    r[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
    r[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
    r[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
    r[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
    r[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
    r
}

/// Hinge across edge (v[0], v[1]); v[2] completes triangle (v0, v1, v2) and
/// v[3] completes triangle (v1, v0, v3).
#[derive(Clone, Debug)]
pub struct Hinge {
    pub v: [u32; 4],
    pub rest_angle: f64,
    /// Energy weight: E = k (θ − θ̄)².
    pub k: f64,
}

/// Signed dihedral angle of the hinge (0 when flat).
pub fn dihedral(x: &[DVec3; 4]) -> f64 {
    let (a, b, p, q) = (x[0], x[1], x[2], x[3]);
    let e = b - a;
    let n1 = e.cross(p - a);
    let n2 = (a - b).cross(q - b);
    let el = e.length().max(1e-300);
    let y = n1.cross(n2).dot(e) / el;
    let xx = n1.dot(n2);
    y.atan2(xx)
}

/// Gradient of the dihedral angle w.r.t. the four vertices (Bridson et al.
/// 2003, with signs matching [`dihedral`]).
pub fn dihedral_grad(x: &[DVec3; 4]) -> [DVec3; 4] {
    let (a, b, p, q) = (x[0], x[1], x[2], x[3]);
    let e = b - a;
    let el = e.length().max(1e-300);
    let n1 = e.cross(p - a);
    let n2 = (a - b).cross(q - b);
    let n1s = n1.length_squared().max(1e-300);
    let n2s = n2.length_squared().max(1e-300);
    let gp = -el * n1 / n1s;
    let gq = -el * n2 / n2s;
    let ga = ((p - b).dot(e) / el) * n1 / n1s + ((q - b).dot(e) / el) * n2 / n2s;
    let gb = -((p - a).dot(e) / el) * n1 / n1s - ((q - a).dot(e) / el) * n2 / n2s;
    // The signs above are fixed against finite differences in the tests.
    [-ga, -gb, gp, gq]
}

impl Hinge {
    /// Bending weight for flexural rigidity `d` (N mm), rest edge length
    /// `len` and rest triangle areas, so that the hinges sum to
    /// (D/2) ∫ κ² dA. This is Wardetzky et al.'s 3|e|²/A_e with A_e taken as
    /// a third of the two triangles' area; the cylinder test checks it.
    pub fn weight(d: f64, len: f64, area1: f64, area2: f64) -> f64 {
        0.5 * d * len * len / (area1 + area2).max(1e-300)
    }

    pub fn energy(&self, x: &[DVec3; 4]) -> f64 {
        let t = wrap(dihedral(x) - self.rest_angle);
        self.k * t * t
    }

    /// Gradient (12) and Gauss–Newton Hessian (12x12, row-major).
    pub fn grad_hess(&self, x: &[DVec3; 4], g: &mut [f64; 12], h: &mut [f64; 144]) {
        let t = wrap(dihedral(x) - self.rest_angle);
        let dg = dihedral_grad(x);
        let mut jt = [0.0f64; 12];
        for v in 0..4 {
            for c in 0..3 {
                jt[3 * v + c] = dg[v][c];
            }
        }
        for p in 0..12 {
            g[p] = 2.0 * self.k * t * jt[p];
        }
        for p in 0..12 {
            for q in 0..12 {
                h[12 * p + q] = 2.0 * self.k * jt[p] * jt[q];
            }
        }
    }
}

/// Hessian of the dihedral angle (12x12, row-major) by central
/// differences of the exact gradient; symmetrised.
pub fn dihedral_hessian(x: &[DVec3; 4]) -> [f64; 144] {
    let scale = (x[1] - x[0]).length().max(1e-12);
    let eps = 1e-6 * scale;
    let mut h = [0.0; 144];
    for v in 0..4 {
        for c in 0..3 {
            let mut a = *x;
            let mut b = *x;
            a[v][c] += eps;
            b[v][c] -= eps;
            let ga = dihedral_grad(&a);
            let gb = dihedral_grad(&b);
            let col = 3 * v + c;
            for w in 0..4 {
                for d in 0..3 {
                    h[12 * (3 * w + d) + col] = (ga[w][d] - gb[w][d]) / (2.0 * eps);
                }
            }
        }
    }
    for i in 0..12 {
        for j in 0..i {
            let m = 0.5 * (h[12 * i + j] + h[12 * j + i]);
            h[12 * i + j] = m;
            h[12 * j + i] = m;
        }
    }
    h
}

/// Replaces a symmetric n x n matrix by its positive semi-definite part
/// (negative eigenvalues set to zero), using cyclic Jacobi rotations.
pub fn project_psd(a: &mut [f64], n: usize) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[n * i + i] = 1.0;
    }
    let norm: f64 = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return;
    }
    for _sweep in 0..30 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += m[n * p + q] * m[n * p + q];
            }
        }
        if off.sqrt() < 1e-14 * norm {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[n * p + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[n * q + q] - m[n * p + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m[n * k + p];
                    let mkq = m[n * k + q];
                    m[n * k + p] = c * mkp - s * mkq;
                    m[n * k + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[n * p + k];
                    let mqk = m[n * q + k];
                    m[n * p + k] = c * mpk - s * mqk;
                    m[n * q + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let vkp = v[n * k + p];
                    let vkq = v[n * k + q];
                    v[n * k + p] = c * vkp - s * vkq;
                    v[n * k + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let lam: Vec<f64> = (0..n).map(|i| m[n * i + i].max(0.0)).collect();
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += v[n * i + k] * lam[k] * v[n * j + k];
            }
            a[n * i + j] = s;
        }
    }
}

/// Wraps an angle difference into (−π, π].
pub fn wrap(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let mut r = a % t;
    if r > std::f64::consts::PI {
        r -= t;
    } else if r <= -std::f64::consts::PI {
        r += t;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_points() -> [DVec3; 3] {
        [DVec3::new(0.1, -0.2, 0.05), DVec3::new(1.3, 0.1, -0.1), DVec3::new(0.4, 1.1, 0.3)]
    }

    #[test]
    fn stretch_zero_at_rest_and_gradient_matches() {
        let x = tri_points();
        let rest: [f64; 3] = std::array::from_fn(|k| (x[EDGE[k].1] - x[EDGE[k].0]).length_squared());
        let t = TriRest::new([0, 1, 2], rest, 1.0, 10.0, 20.0);
        assert!(t.energy(&x).abs() < 1e-20);
        // Perturb and compare the gradient with central differences.
        let mut y = x;
        y[0] += DVec3::new(0.05, 0.02, -0.03);
        y[2] += DVec3::new(-0.04, 0.06, 0.01);
        let mut g = [0.0; 9];
        let mut h = [0.0; 81];
        t.grad_hess(&y, &mut g, &mut h);
        let eps = 1e-6;
        for v in 0..3 {
            for c in 0..3 {
                let mut a = y;
                let mut b = y;
                a[v][c] += eps;
                b[v][c] -= eps;
                let fd = (t.energy(&a) - t.energy(&b)) / (2.0 * eps);
                let an = g[3 * v + c];
                assert!((fd - an).abs() < 1e-6 * an.abs().max(1.0), "v{v} c{c}: fd {fd} an {an}");
            }
        }
    }

    #[test]
    fn stretch_matches_stvk_for_uniform_stretch() {
        // Equilateral triangle stretched uniformly by s: E = 1/2 (s² − 1) I,
        // W = h A (μ tr E² + λ/2 (tr E)²).
        let x0 = [DVec3::ZERO, DVec3::X, DVec3::new(0.5, 3f64.sqrt() / 2.0, 0.0)];
        let rest: [f64; 3] = std::array::from_fn(|k| (x0[EDGE[k].1] - x0[EDGE[k].0]).length_squared());
        let (mu, lambda, h) = (3.0, 5.0, 0.7);
        let t = TriRest::new([0, 1, 2], rest, h, mu, lambda);
        let s = 1.1;
        let x = x0.map(|p| p * s);
        let e = 0.5 * (s * s - 1.0);
        let expected = h * t.area * (mu * 2.0 * e * e + 0.5 * lambda * (2.0 * e).powi(2));
        assert!((t.energy(&x) - expected).abs() < 1e-12 * expected, "{} {}", t.energy(&x), expected);
    }

    fn hinge_points() -> [DVec3; 4] {
        [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.1, 0.0),
            DVec3::new(0.4, 0.9, 0.2),
            DVec3::new(0.6, -0.8, 0.35),
        ]
    }

    #[test]
    fn dihedral_gradient_matches_finite_differences() {
        let x = hinge_points();
        let g = dihedral_grad(&x);
        let eps = 1e-6;
        for v in 0..4 {
            for c in 0..3 {
                let mut a = x;
                let mut b = x;
                a[v][c] += eps;
                b[v][c] -= eps;
                let fd = (dihedral(&a) - dihedral(&b)) / (2.0 * eps);
                assert!((fd - g[v][c]).abs() < 1e-6, "v{v} c{c}: fd {fd} an {}", g[v][c]);
            }
        }
    }

    #[test]
    fn psd_projection() {
        // diag(2, -1) rotated by 30 degrees -> its PSD part is 2 u uᵀ.
        let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let u = [c, s];
        let w = [-s, c];
        let mut a: Vec<f64> = (0..4).map(|k| 2.0 * u[k / 2] * u[k % 2] - w[k / 2] * w[k % 2]).collect();
        project_psd(&mut a, 2);
        for k in 0..4 {
            assert!((a[k] - 2.0 * u[k / 2] * u[k % 2]).abs() < 1e-12);
        }
    }

    #[test]
    fn flat_hinge_has_zero_angle() {
        let x = [DVec3::ZERO, DVec3::X, DVec3::new(0.5, 1.0, 0.0), DVec3::new(0.5, -1.0, 0.0)];
        assert!(dihedral(&x).abs() < 1e-14);
    }

    /// Calibration of the hinge weight: a structured mesh bent into a
    /// cylinder of radius R must give (D/2) κ² A.
    #[test]
    fn hinge_weight_reproduces_cylinder_energy() {
        let (nx, ny, dx) = (40usize, 40usize, 0.1f64);
        let r = 3.0;
        let d = 2.0;
        for axis in 0..2 {
            let pos = |i: usize, j: usize| -> DVec3 {
                let (u, v) = (i as f64 * dx, j as f64 * dx);
                if axis == 0 {
                    DVec3::new(r * (u / r).sin(), v, r * (1.0 - (u / r).cos()))
                } else {
                    DVec3::new(u, r * (v / r).sin(), r * (1.0 - (v / r).cos()))
                }
            };
            let flat = |i: usize, j: usize| DVec3::new(i as f64 * dx, j as f64 * dx, 0.0);
            // Triangles with alternating diagonals.
            let mut tris = Vec::new();
            for i in 0..nx {
                for j in 0..ny {
                    let a = (i, j);
                    let b = (i + 1, j);
                    let c = (i + 1, j + 1);
                    let dd = (i, j + 1);
                    if (i + j) % 2 == 0 {
                        tris.push([a, b, c]);
                        tris.push([a, c, dd]);
                    } else {
                        tris.push([a, b, dd]);
                        tris.push([b, c, dd]);
                    }
                }
            }
            // Edge -> (triangle, opposite vertex).
            let mut map: std::collections::HashMap<((usize, usize), (usize, usize)), Vec<(usize, (usize, usize))>> =
                Default::default();
            for (t, tri) in tris.iter().enumerate() {
                for k in 0..3 {
                    let (a, b) = (tri[(k + 1) % 3], tri[(k + 2) % 3]);
                    let key = if a < b { (a, b) } else { (b, a) };
                    map.entry(key).or_default().push((t, tri[k]));
                }
            }
            let area_of = |t: &[(usize, usize); 3]| {
                let p = t.map(|(i, j)| flat(i, j));
                0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
            };
            let mut e = 0.0;
            let mut total_area = 0.0;
            for tri in &tris {
                total_area += area_of(tri);
            }
            for (key, adj) in &map {
                if adj.len() != 2 {
                    continue;
                }
                let (a, b) = *key;
                let (t1, p) = adj[0];
                let (t2, q) = adj[1];
                // Orient so that (a, b, p) is one triangle.
                let x = [pos(a.0, a.1), pos(b.0, b.1), pos(p.0, p.1), pos(q.0, q.1)];
                let len = (flat(a.0, a.1) - flat(b.0, b.1)).length();
                let k = Hinge::weight(d, len, area_of(&tris[t1]), area_of(&tris[t2]));
                let th = dihedral(&x);
                e += k * th * th;
            }
            // Exclude the boundary ring of hinges from the comparison area.
            let interior = ((nx - 1) as f64 * dx) * ((ny - 1) as f64 * dx);
            let expected = 0.5 * d / (r * r) * interior;
            let ratio = e / expected;
            println!("axis {axis}: discrete/continuum = {ratio:.4} (total area {total_area})");
            assert!((ratio - 1.0).abs() < 0.08, "axis {axis}: ratio {ratio}");
        }
    }
}

#[cfg(test)]
mod beam_shape {
    use super::*;

    /// Discrete bending energy of the exact cantilever shape on the strip
    /// mesh used by the simulator, against (D/2) W ∫ (y'')² dz.
    #[test]
    fn hinge_energy_of_beam_shape() {
        for (nx, ds) in [(16usize, 8.0f64 / 15.0), (32, 8.0 / 31.0)] {
            let nz = (12.0 / ds).round() as usize;
            let len = nz as f64 * ds;
            let w = (nx - 1) as f64 * ds;
            let a = 1e-6; // y = a (z⁴ − 4 L z³ + 6 L² z²), small slopes
            let y = |z: f64| a * (z.powi(4) - 4.0 * len * z.powi(3) + 6.0 * len * len * z * z);
            let ypp = |z: f64| a * (12.0 * z * z - 24.0 * len * z + 12.0 * len * len);
            let pos = |i: usize, j: usize| DVec3::new(i as f64 * ds, y(j as f64 * ds), j as f64 * ds);
            let flat = |i: usize, j: usize| DVec3::new(i as f64 * ds, 0.0, j as f64 * ds);
            let idx = |i: usize, j: usize| j * nx + i;
            let mut tris = Vec::new();
            for j in 0..nz {
                for i in 0..nx - 1 {
                    let (p, q, r, s) = (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
                    if (i + j) % 2 == 0 {
                        tris.push([p, q, r]);
                        tris.push([p, r, s]);
                    } else {
                        tris.push([p, q, s]);
                        tris.push([q, r, s]);
                    }
                }
            }
            let coord = |k: usize| (k % nx, k / nx);
            let mut open: std::collections::HashMap<(usize, usize), (usize, usize)> = Default::default();
            let d = 1.0;
            let mut e = 0.0;
            let area = |t: [usize; 3]| {
                let p = t.map(|k| {
                    let (i, j) = coord(k);
                    flat(i, j)
                });
                0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
            };
            for (ti, t) in tris.iter().enumerate() {
                for k in 0..3 {
                    let (u, v, opp) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                    if let Some(&(tj, p)) = open.get(&(v, u)) {
                        let quad = [v, u, p, opp].map(|k| {
                            let (i, j) = coord(k);
                            pos(i, j)
                        });
                        let (ui, uj) = coord(u);
                        let (vi, vj) = coord(v);
                        let l = (flat(ui, uj) - flat(vi, vj)).length();
                        let kk = Hinge::weight(d, l, area(tris[tj]), area(*t));
                        let th = dihedral(&quad);
                        e += kk * th * th;
                    }
                }
                for k in 0..3 {
                    open.insert((t[k], t[(k + 1) % 3]), (ti, t[(k + 2) % 3]));
                }
            }
            // Continuum: (D/2) W ∫ y''² dz, midpoint rule fine enough.
            let m = 4000;
            let cont: f64 = (0..m).map(|k| ypp((k as f64 + 0.5) * len / m as f64).powi(2)).sum::<f64>() * len / m as f64
                * 0.5
                * d
                * w;
            println!("nx {nx}: discrete/continuum = {:.4}", e / cont);
        }
    }
}

/// Bending energy of one face from a shape operator built out of its three
/// hinge angles (Grinspun, Gingold, Reisman and Zorin 2006):
///
/// S = Σ_k a_k Δθ_k t_k t_kᵀ,  a_k = |e_k| / (2 A),
/// E = (D A / 2) [ (1 − ν) tr S² + ν (tr S)² ],
///
/// where t_k is the unit in-plane normal of edge k in the rest triangle.
/// Coupling the three angles of a face makes the energy a proper tensor
/// quantity, which removes the soft zigzag modes that independent hinges
/// allow on structured meshes. Boundary edges have no hinge and contribute
/// nothing. E = Δθᵀ B Δθ with B from the rest geometry.
#[derive(Clone, Debug)]
pub struct FaceBend {
    /// Up to six nodes: the face's three vertices, then the opposite
    /// vertices of its hinges.
    pub nodes: Vec<u32>,
    /// For edge k (opposite vertex k): the hinge's four nodes as indices
    /// into `nodes`, in hinge order, and the hinge index.
    pub hinge: [Option<([u8; 4], u32)>; 3],
    pub b: [[f64; 3]; 3],
}

impl FaceBend {
    /// B from the rest edge lengths (squared), flexural rigidity and
    /// Poisson ratio.
    pub fn b_matrix(rest_len2: [f64; 3], d: f64, nu: f64) -> [[f64; 3]; 3] {
        let l0 = rest_len2[0].max(1e-30).sqrt();
        let l1 = rest_len2[1].max(1e-30).sqrt();
        let l2 = rest_len2[2].max(1e-30).sqrt();
        let x2 = ((l1 * l1 + l2 * l2 - l0 * l0) / (2.0 * l2)).clamp(-l1, l1);
        let y2 = (l1 * l1 - x2 * x2).max((1e-3 * l1.min(l2)).powi(2)).sqrt();
        let p = [[0.0, 0.0], [l2, 0.0], [x2, y2]];
        let area = 0.5 * l2 * y2;
        let mut t = [[0.0f64; 2]; 3];
        let mut a = [0.0f64; 3];
        for k in 0..3 {
            let (i, j) = EDGE[k];
            let e = [p[j][0] - p[i][0], p[j][1] - p[i][1]];
            let len = (e[0] * e[0] + e[1] * e[1]).sqrt().max(1e-300);
            t[k] = [-e[1] / len, e[0] / len];
            a[k] = len / (2.0 * area);
        }
        let mut b = [[0.0; 3]; 3];
        for j in 0..3 {
            for k in 0..3 {
                let tt = t[j][0] * t[k][0] + t[j][1] * t[k][1];
                b[j][k] = 0.5 * d * area * a[j] * a[k] * ((1.0 - nu) * tt * tt + nu);
            }
        }
        b
    }

    fn angles(&self, x: &[DVec3], rest: [f64; 3]) -> ([f64; 3], [Option<[DVec3; 4]>; 3]) {
        let mut d = [0.0; 3];
        let mut grads = [None; 3];
        for k in 0..3 {
            if let Some((loc, _)) = self.hinge[k] {
                let q = loc.map(|l| x[l as usize]);
                d[k] = wrap(dihedral(&q) - rest[k]);
                grads[k] = Some(dihedral_grad(&q));
            }
        }
        (d, grads)
    }

    pub fn energy(&self, x: &[DVec3], rest: [f64; 3]) -> f64 {
        let mut d = [0.0; 3];
        for k in 0..3 {
            if let Some((loc, _)) = self.hinge[k] {
                d[k] = wrap(dihedral(&loc.map(|l| x[l as usize])) - rest[k]);
            }
        }
        let mut e = 0.0;
        for j in 0..3 {
            for k in 0..3 {
                e += d[j] * self.b[j][k] * d[k];
            }
        }
        e
    }

    /// Gradient (3 × nodes) and Gauss–Newton Hessian (square, row-major).
    pub fn grad_hess(&self, x: &[DVec3], rest: [f64; 3], g: &mut [f64], h: &mut [f64]) {
        let n = 3 * self.nodes.len();
        let (d, grads) = self.angles(x, rest);
        // Jacobian rows of the three angles.
        let mut jac = [[0.0f64; 18]; 3];
        for k in 0..3 {
            if let (Some((loc, _)), Some(gr)) = (self.hinge[k], grads[k]) {
                for v in 0..4 {
                    for c in 0..3 {
                        jac[k][3 * loc[v] as usize + c] += gr[v][c];
                    }
                }
            }
        }
        let bd: [f64; 3] = std::array::from_fn(|j| (0..3).map(|k| self.b[j][k] * d[k]).sum());
        for p in 0..n {
            g[p] = 2.0 * (0..3).map(|j| bd[j] * jac[j][p]).sum::<f64>();
        }
        for p in 0..n {
            for q in 0..n {
                let mut s = 0.0;
                for j in 0..3 {
                    if jac[j][p] == 0.0 {
                        continue;
                    }
                    for k in 0..3 {
                        s += self.b[j][k] * jac[j][p] * jac[k][q];
                    }
                }
                h[n * p + q] = 2.0 * s;
            }
        }
    }
}

#[cfg(test)]
mod face_bend_tests {
    use super::*;

    /// Builds FaceBend elements for a structured grid and returns the total
    /// energy of the given positions.
    fn grid_energy(
        nx: usize,
        nz: usize,
        ds: f64,
        pattern: &dyn Fn(usize, usize) -> bool,
        pos: &dyn Fn(usize, usize) -> DVec3,
        nu: f64,
    ) -> f64 {
        let idx = |i: usize, j: usize| j * nx + i;
        let flat = |k: usize| DVec3::new((k % nx) as f64 * ds, 0.0, (k / nx) as f64 * ds);
        let mut tris = Vec::new();
        for j in 0..nz {
            for i in 0..nx - 1 {
                let (p, q, r, s) = (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
                if pattern(i, j) {
                    tris.push([p, q, r]);
                    tris.push([p, r, s]);
                } else {
                    tris.push([p, q, s]);
                    tris.push([q, r, s]);
                }
            }
        }
        // Directed edge -> (triangle, opposite).
        let mut map: std::collections::HashMap<(usize, usize), (usize, usize)> = Default::default();
        for (ti, t) in tris.iter().enumerate() {
            for k in 0..3 {
                map.insert((t[k], t[(k + 1) % 3]), (ti, t[(k + 2) % 3]));
            }
        }
        let x: Vec<DVec3> = (0..nx * (nz + 1)).map(|k| pos(k % nx, k / nx)).collect();
        let mut e = 0.0;
        for t in &tris {
            let rest: [f64; 3] = std::array::from_fn(|k| (flat(t[EDGE[k].1]) - flat(t[EDGE[k].0])).length_squared());
            let mut nodes: Vec<u32> = t.iter().map(|&v| v as u32).collect();
            let mut hinge = [None; 3];
            for k in 0..3 {
                let (a, b) = (t[EDGE[k].0], t[EDGE[k].1]);
                // This face traverses either (a, b) or (b, a); find the twin.
                let (u, v) = if map.get(&(a, b)).map(|&(ti, _)| tris[ti] == *t).unwrap_or(false) { (a, b) } else { (b, a) };
                if let Some(&(_, q)) = map.get(&(v, u)) {
                    let p = t.iter().copied().find(|&w| w != u && w != v).unwrap();
                    let quad = [u, v, p, q];
                    let loc = quad.map(|w| {
                        if let Some(pos) = nodes.iter().position(|&n| n as usize == w) {
                            pos as u8
                        } else {
                            nodes.push(w as u32);
                            (nodes.len() - 1) as u8
                        }
                    });
                    hinge[k] = Some((loc, 0));
                }
            }
            let fb = FaceBend { nodes: nodes.clone(), hinge, b: FaceBend::b_matrix(rest, 1.0, nu) };
            let xs: Vec<DVec3> = nodes.iter().map(|&n| x[n as usize]).collect();
            e += fb.energy(&xs, [0.0; 3]);
        }
        e
    }

    #[test]
    fn face_bend_cylinder_and_beam() {
        let ds = 0.1;
        let (nx, nz) = (41usize, 40usize);
        let patterns: [(&str, Box<dyn Fn(usize, usize) -> bool>); 2] =
            [("alt", Box::new(|i, j| (i + j) % 2 == 0)), ("same", Box::new(|_, _| true))];
        for (name, pat) in &patterns {
            for axis in 0..2 {
                let r = 3.0;
                let pos = |i: usize, j: usize| -> DVec3 {
                    let (u, v) = (i as f64 * ds, j as f64 * ds);
                    if axis == 0 {
                        DVec3::new(r * (u / r).sin(), r * (1.0 - (u / r).cos()), v)
                    } else {
                        DVec3::new(u, r * (1.0 - (v / r).cos()), r * (v / r).sin())
                    }
                };
                let e = grid_energy(nx, nz, ds, pat.as_ref(), &pos, 0.0);
                // Interior region only (boundary faces see fewer hinges).
                let full = 0.5 / (r * r) * ((nx - 1) as f64 * ds) * (nz as f64 * ds);
                println!("{name} cylinder axis {axis}: ratio {:.4}", e / full);
            }
        }
    }
}
