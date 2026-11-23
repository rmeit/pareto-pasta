//! Quadratic (10-node) tetrahedra with a compressible neo-Hookean material.
//!
//! Node order: corners 0..4, then the midpoints of edges (0,1), (1,2), (0,2),
//! (0,3), (1,3), (2,3). Integrals use the 4-point rule, which is exact for
//! the quadratic integrands of small-strain elasticity.

/// Local corner pairs of the six edges; node 4 + e sits on edge e.
pub const EDGES: [(usize, usize); 6] = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)];

/// The four faces, each listed so that its normal (right-hand rule) points
/// away from the opposite corner when the tet has positive volume.
pub const FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

const QA: f64 = 0.585_410_196_624_968_5;
const QB: f64 = 0.138_196_601_125_010_5;
pub const QP: [[f64; 4]; 4] = [[QA, QB, QB, QB], [QB, QA, QB, QB], [QB, QB, QA, QB], [QB, QB, QB, QA]];

/// Derivatives of the barycentric coordinates with respect to the
/// parametric coordinates (xi1, xi2, xi3) = (l1, l2, l3).
const DL: [[f64; 3]; 4] = [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Local index (4..10) of the mid-edge node between local corners a and b.
pub fn mid_node(a: usize, b: usize) -> usize {
    4 + EDGES.iter().position(|&(p, q)| (p == a && q == b) || (p == b && q == a)).expect("corners share an edge")
}

pub fn shape(l: &[f64; 4]) -> [f64; 10] {
    let mut n = [0.0; 10];
    for i in 0..4 {
        n[i] = l[i] * (2.0 * l[i] - 1.0);
    }
    for (e, &(p, q)) in EDGES.iter().enumerate() {
        n[4 + e] = 4.0 * l[p] * l[q];
    }
    n
}

fn dshape(l: &[f64; 4]) -> [[f64; 3]; 10] {
    let mut d = [[0.0; 3]; 10];
    for i in 0..4 {
        for j in 0..3 {
            d[i][j] = (4.0 * l[i] - 1.0) * DL[i][j];
        }
    }
    for (e, &(p, q)) in EDGES.iter().enumerate() {
        for j in 0..3 {
            d[4 + e][j] = 4.0 * (l[p] * DL[q][j] + l[q] * DL[p][j]);
        }
    }
    d
}

pub type M3 = [[f64; 3]; 3];

pub fn det3(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

pub fn inv3(m: &M3) -> Option<(M3, f64)> {
    let d = det3(m);
    if d == 0.0 || !d.is_finite() {
        return None;
    }
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let inv = [
        [c(1, 1, 2, 2) / d, -c(0, 1, 2, 2) / d, c(0, 1, 1, 2) / d],
        [-c(1, 0, 2, 2) / d, c(0, 0, 2, 2) / d, -c(0, 0, 1, 2) / d],
        [c(1, 0, 2, 1) / d, -c(0, 0, 2, 1) / d, c(0, 0, 1, 1) / d],
    ];
    Some((inv, d))
}

/// Rest geometry of one tet: shape-function gradients with respect to the
/// rest positions, and integration weights, at each quadrature point.
#[derive(Clone, Debug)]
pub struct Rest {
    pub grad: [[[f64; 3]; 10]; 4],
    pub w: [f64; 4],
}

impl Rest {
    /// `None` if the rest shape is degenerate or folded.
    pub fn new(x: &[[f64; 3]; 10]) -> Option<Rest> {
        let mut grad = [[[0.0; 3]; 10]; 4];
        let mut w = [0.0; 4];
        let mut sign = 0.0;
        for (q, l) in QP.iter().enumerate() {
            let dn = dshape(l);
            let mut j = [[0.0; 3]; 3];
            for a in 0..10 {
                for r in 0..3 {
                    for c in 0..3 {
                        j[r][c] += x[a][r] * dn[a][c];
                    }
                }
            }
            let (ji, d) = inv3(&j)?;
            if sign == 0.0 {
                sign = d.signum();
            } else if d.signum() != sign {
                return None;
            }
            for a in 0..10 {
                for k in 0..3 {
                    grad[q][a][k] = (0..3).map(|c| dn[a][c] * ji[c][k]).sum();
                }
            }
            w[q] = d.abs() / 24.0;
        }
        Some(Rest { grad, w })
    }

    pub fn volume(&self) -> f64 {
        self.w.iter().sum()
    }

    /// Integral of each shape function over the rest volume (consistent
    /// load of a unit body force).
    pub fn load(&self) -> [f64; 10] {
        let mut out = [0.0; 10];
        for (q, l) in QP.iter().enumerate() {
            let n = shape(l);
            for a in 0..10 {
                out[a] += self.w[q] * n[a];
            }
        }
        out
    }

    /// Lumped mass fractions (row-scaled diagonal): corners 1/36, edges 4/27
    /// of the volume. All positive, unlike row-sum lumping for this element.
    pub fn lumped(&self) -> [f64; 10] {
        let v = self.volume();
        let mut out = [4.0 * v / 27.0; 10];
        for o in out.iter_mut().take(4) {
            *o = v / 36.0;
        }
        out
    }

    fn deformation(&self, q: usize, x: &[[f64; 3]; 10]) -> M3 {
        let mut f = [[0.0; 3]; 3];
        for a in 0..10 {
            for i in 0..3 {
                for j in 0..3 {
                    f[i][j] += x[a][i] * self.grad[q][a][j];
                }
            }
        }
        f
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub mu: f64,
    pub lambda: f64,
}

impl Material {
    /// From Young's modulus (N/mm²) and Poisson's ratio.
    pub fn new(young: f64, nu: f64) -> Self {
        Self { mu: young / (2.0 * (1.0 + nu)), lambda: young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) }
    }

    /// Strain energy density; infinite for an inverted element.
    pub fn psi(&self, f: &M3) -> f64 {
        let j = det3(f);
        if !(j > 0.0) {
            return f64::INFINITY;
        }
        let i1: f64 = f.iter().flatten().map(|v| v * v).sum();
        let lj = j.ln();
        0.5 * self.mu * (i1 - 3.0) - self.mu * lj + 0.5 * self.lambda * lj * lj
    }

    /// First Piola stress and its derivative dP/dF (9 x 9, index 3i + J),
    /// optionally projected to be positive semi-definite.
    pub fn stress_tangent(&self, f: &M3, project: bool) -> Option<(M3, [f64; 81])> {
        let (finv, j) = inv3(f)?;
        if !(j > 0.0) {
            return None;
        }
        let lj = j.ln();
        // G = F^-T
        let g: M3 = std::array::from_fn(|i| std::array::from_fn(|jj| finv[jj][i]));
        let mut p = [[0.0; 3]; 3];
        for i in 0..3 {
            for jj in 0..3 {
                p[i][jj] = self.mu * (f[i][jj] - g[i][jj]) + self.lambda * lj * g[i][jj];
            }
        }
        let a = self.mu - self.lambda * lj;
        let mut c = [0.0; 81];
        for i in 0..3 {
            for jj in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        let mut v = a * g[i][l] * g[k][jj] + self.lambda * g[i][jj] * g[k][l];
                        if i == k && jj == l {
                            v += self.mu;
                        }
                        c[(3 * i + jj) * 9 + 3 * k + l] = v;
                    }
                }
            }
        }
        if project {
            project_psd9(&mut c);
        }
        Some((p, c))
    }
}

/// Clamps the negative eigenvalues of a symmetric 9 x 9 matrix to zero
/// (cyclic Jacobi).
pub fn project_psd9(a: &mut [f64; 81]) {
    let mut m = *a;
    let mut v = [0.0; 81];
    for i in 0..9 {
        v[i * 9 + i] = 1.0;
    }
    let scale: f64 = (0..9).map(|i| m[i * 9 + i].abs()).fold(0.0, f64::max).max(1e-300);
    for _ in 0..16 {
        let mut off = 0.0;
        for p in 0..9 {
            for q in p + 1..9 {
                off += m[p * 9 + q] * m[p * 9 + q];
            }
        }
        if off.sqrt() < 1e-13 * scale {
            break;
        }
        for p in 0..9 {
            for q in p + 1..9 {
                let apq = m[p * 9 + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * 9 + q] - m[p * 9 + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..9 {
                    let mkp = m[k * 9 + p];
                    let mkq = m[k * 9 + q];
                    m[k * 9 + p] = c * mkp - s * mkq;
                    m[k * 9 + q] = s * mkp + c * mkq;
                }
                for k in 0..9 {
                    let mpk = m[p * 9 + k];
                    let mqk = m[q * 9 + k];
                    m[p * 9 + k] = c * mpk - s * mqk;
                    m[q * 9 + k] = s * mpk + c * mqk;
                }
                for k in 0..9 {
                    let vkp = v[k * 9 + p];
                    let vkq = v[k * 9 + q];
                    v[k * 9 + p] = c * vkp - s * vkq;
                    v[k * 9 + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eig: [f64; 9] = std::array::from_fn(|i| m[i * 9 + i]);
    if eig.iter().all(|&e| e >= 0.0) {
        return;
    }
    for r in 0..9 {
        for c in 0..9 {
            a[r * 9 + c] = (0..9).map(|k| v[r * 9 + k] * eig[k].max(0.0) * v[c * 9 + k]).sum();
        }
    }
}

/// Elastic energy of one tet at positions x; infinite if any point inverts.
pub fn energy(rest: &Rest, mat: &Material, x: &[[f64; 3]; 10]) -> f64 {
    let mut e = 0.0;
    for q in 0..4 {
        e += rest.w[q] * mat.psi(&rest.deformation(q, x));
    }
    e
}

/// Gradient (30) and projected Hessian (30 x 30, row-major) of one tet.
/// Returns false if the element is inverted.
pub fn grad_hess(rest: &Rest, mat: &Material, x: &[[f64; 3]; 10], g: &mut [f64; 30], h: &mut [f64; 900]) -> bool {
    grad_hess_with(rest, mat, x, g, h, true)
}

/// As [`grad_hess`], with the projection of the material tangent optional.
pub fn grad_hess_with(
    rest: &Rest,
    mat: &Material,
    x: &[[f64; 3]; 10],
    g: &mut [f64; 30],
    h: &mut [f64; 900],
    project: bool,
) -> bool {
    g.fill(0.0);
    h.fill(0.0);
    for q in 0..4 {
        let f = rest.deformation(q, x);
        let Some((p, c)) = mat.stress_tangent(&f, project) else { return false };
        let w = rest.w[q];
        let gr = &rest.grad[q];
        for a in 0..10 {
            for i in 0..3 {
                g[3 * a + i] += w * (p[i][0] * gr[a][0] + p[i][1] * gr[a][1] + p[i][2] * gr[a][2]);
            }
        }
        // cb[b][(3i + J)][k] = sum_L C[iJ, kL] grad_b[L]
        let mut cb = [[[0.0; 3]; 9]; 10];
        for b in 0..10 {
            for ij in 0..9 {
                for k in 0..3 {
                    let row = ij * 9 + 3 * k;
                    cb[b][ij][k] = c[row] * gr[b][0] + c[row + 1] * gr[b][1] + c[row + 2] * gr[b][2];
                }
            }
        }
        for a in 0..10 {
            for b in 0..10 {
                for i in 0..3 {
                    for k in 0..3 {
                        let v = gr[a][0] * cb[b][3 * i][k] + gr[a][1] * cb[b][3 * i + 1][k] + gr[a][2] * cb[b][3 * i + 2][k];
                        h[(3 * a + i) * 30 + 3 * b + k] += w * v;
                    }
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight_tet(c: [[f64; 3]; 4]) -> [[f64; 3]; 10] {
        let mut x = [[0.0; 3]; 10];
        x[..4].copy_from_slice(&c);
        for (e, &(p, q)) in EDGES.iter().enumerate() {
            x[4 + e] = std::array::from_fn(|d| 0.5 * (c[p][d] + c[q][d]));
        }
        x
    }

    fn unit() -> [[f64; 3]; 10] {
        straight_tet([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 0.9, 0.1], [0.1, 0.2, 1.1]])
    }

    #[test]
    fn rest_volume_load_and_mass() {
        let x = unit();
        let r = Rest::new(&x).unwrap();
        let c: [[f64; 3]; 4] = std::array::from_fn(|i| x[i]);
        let e1: [f64; 3] = std::array::from_fn(|d| c[1][d] - c[0][d]);
        let e2: [f64; 3] = std::array::from_fn(|d| c[2][d] - c[0][d]);
        let e3: [f64; 3] = std::array::from_fn(|d| c[3][d] - c[0][d]);
        let v = det3(&[e1, e2, e3]).abs() / 6.0;
        assert!((r.volume() - v).abs() < 1e-14);
        let l = r.load();
        for a in 0..4 {
            assert!((l[a] + v / 20.0).abs() < 1e-14);
        }
        for a in 4..10 {
            assert!((l[a] - v / 5.0).abs() < 1e-14);
        }
        assert!((r.lumped().iter().sum::<f64>() - v).abs() < 1e-14);
    }

    #[test]
    fn rigid_motion_costs_nothing() {
        let x = unit();
        let r = Rest::new(&x).unwrap();
        let m = Material::new(1.0, 0.45);
        let (s, c) = (0.6f64.sin(), 0.6f64.cos());
        let y: [[f64; 3]; 10] = std::array::from_fn(|a| [c * x[a][0] - s * x[a][1] + 3.0, s * x[a][0] + c * x[a][1] - 1.0, x[a][2] + 2.0]);
        assert!(energy(&r, &m, &y).abs() < 1e-12);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let x = unit();
        let r = Rest::new(&x).unwrap();
        let m = Material::new(1.0, 0.4);
        let mut y = x;
        for (a, p) in y.iter_mut().enumerate() {
            for d in 0..3 {
                p[d] += 0.07 * ((a * 3 + d) as f64 * 1.3).sin();
            }
        }
        let mut g = [0.0; 30];
        let mut h = [0.0; 900];
        assert!(grad_hess(&r, &m, &y, &mut g, &mut h));
        let eps = 1e-6;
        for k in 0..30 {
            let mut yp = y;
            let mut ym = y;
            yp[k / 3][k % 3] += eps;
            ym[k / 3][k % 3] -= eps;
            let fd = (energy(&r, &m, &yp) - energy(&r, &m, &ym)) / (2.0 * eps);
            assert!((fd - g[k]).abs() < 1e-6 * (1.0 + g[k].abs()), "{k}: {fd} vs {}", g[k]);
        }
    }

    #[test]
    fn hessian_matches_finite_differences() {
        let x = unit();
        let r = Rest::new(&x).unwrap();
        let m = Material::new(1.0, 0.4);
        let mut y = x;
        for (a, p) in y.iter_mut().enumerate() {
            for d in 0..3 {
                p[d] += 0.01 * ((a * 3 + d) as f64 * 0.7).cos();
            }
        }
        let mut g0 = [0.0; 30];
        let mut h = [0.0; 900];
        grad_hess_with(&r, &m, &y, &mut g0, &mut h, false);
        let eps = 1e-6;
        for k in 0..30 {
            let mut yp = y;
            yp[k / 3][k % 3] += eps;
            let mut g1 = [0.0; 30];
            let mut h1 = [0.0; 900];
            grad_hess_with(&r, &m, &yp, &mut g1, &mut h1, false);
            for i in 0..30 {
                let fd = (g1[i] - g0[i]) / eps;
                assert!((fd - h[i * 30 + k]).abs() < 1e-4 * (1.0 + fd.abs()), "{i},{k}: {fd} vs {}", h[i * 30 + k]);
            }
        }
    }

    #[test]
    fn projection_keeps_psd_and_fixes_negative() {
        let mut a = [0.0; 81];
        for i in 0..9 {
            a[i * 9 + i] = i as f64 - 2.0;
        }
        a[1] = 0.5;
        a[9] = 0.5;
        project_psd9(&mut a);
        // x^T A x >= 0 for a few vectors.
        for s in 0..20 {
            let x: [f64; 9] = std::array::from_fn(|i| ((i + s) as f64 * 1.7).sin());
            let q: f64 = (0..9).map(|r| (0..9).map(|c| x[r] * a[r * 9 + c] * x[c]).sum::<f64>()).sum();
            assert!(q >= -1e-12);
        }
    }
}
