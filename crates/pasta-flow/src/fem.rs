//! Linear finite elements and a damped Newton method for the die flow.

use crate::{Drive, FlowParams, Rheology};
use pasta_die::TriMesh2;
use pasta_linalg::{SolveError, SpdPattern, par};
use rayon::prelude::*;

/// One set of energy parameters. Continuation walks through several.
#[derive(Clone, Copy, Debug)]
struct Stage {
    n: f64,
    k: f64,
    tau_y: f64,
    slip: bool,
    beta: f64,
    m: f64,
    dp: f64,
    /// Shear-rate regularisation (1/s).
    eps: f64,
    /// Slip-speed regularisation (mm/s).
    delta: f64,
}

#[derive(Clone, Debug, Default)]
pub struct SolveStats {
    pub newton_steps: usize,
    pub pressure_iterations: usize,
    pub cold_start: bool,
    pub seconds: f64,
}

#[derive(Clone, Debug)]
pub struct FlowSolution {
    /// Exit speed at each mesh node (mm/s).
    pub w: Vec<f64>,
    /// Pressure drop across the die (kPa).
    pub dp_kpa: f64,
    /// Total volume flow (mm^3/s).
    pub q_total: f64,
    /// Volume flow through each opening (mm^3/s).
    pub q_opening: Vec<f64>,
    /// Mean exit speed of each opening (mm/s).
    pub mean_speed: Vec<f64>,
    /// Shear rate in each triangle (1/s).
    pub tri_shear: Vec<f64>,
    /// Fraction of each opening's area moving as an unsheared plug.
    pub plug_fraction: Vec<f64>,
    /// Wall speed over the opening's mean speed, per boundary edge.
    pub wall_slip_ratio: Vec<f64>,
    /// Relative error estimate of the speed gradient (recovery based).
    pub error_estimate: f64,
    /// Final Newton decrement relative to the energy scale.
    pub residual: f64,
    pub converged: bool,
    pub stats: SolveStats,
    gamma_ref: f64,
    w_ref: f64,
    rheology: Rheology,
}

#[derive(Debug)]
pub enum FlowError {
    Cancelled,
    Solve(SolveError),
    Empty,
}

impl std::fmt::Display for FlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlowError::Cancelled => write!(f, "cancelled"),
            FlowError::Solve(e) => write!(f, "{e}"),
            FlowError::Empty => write!(f, "the die has no openings"),
        }
    }
}

impl std::error::Error for FlowError {}

impl From<SolveError> for FlowError {
    fn from(e: SolveError) -> Self {
        FlowError::Solve(e)
    }
}

/// Precomputed geometry and sparsity for one mesh and one slip setting.
pub struct FlowSolver {
    mesh: TriMesh2,
    area: Vec<f64>,
    grads: Vec<[[f64; 2]; 3]>,
    edge_len: Vec<f64>,
    no_slip: bool,
    dirichlet: Vec<bool>,
    pattern: SpdPattern,
    threads: usize,
}

const TRI_PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1)];
const EDGE_PAIRS: [(usize, usize); 3] = [(0, 0), (1, 1), (1, 0)];

fn lower(a: u32, b: u32) -> (usize, usize) {
    let (a, b) = (a as usize, b as usize);
    if a >= b { (a, b) } else { (b, a) }
}

impl FlowSolver {
    pub fn new(mesh: &TriMesh2, no_slip: bool, threads: usize) -> Result<Self, FlowError> {
        if mesh.tris.is_empty() {
            return Err(FlowError::Empty);
        }
        let n = mesh.nodes.len();
        let mut area = Vec::with_capacity(mesh.tris.len());
        let mut grads = Vec::with_capacity(mesh.tris.len());
        for t in &mesh.tris {
            let [p0, p1, p2] = t.map(|i| mesh.nodes[i as usize]);
            let a2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
            area.push(0.5 * a2);
            grads.push([
                [(p1[1] - p2[1]) / a2, (p2[0] - p1[0]) / a2],
                [(p2[1] - p0[1]) / a2, (p0[0] - p2[0]) / a2],
                [(p0[1] - p1[1]) / a2, (p1[0] - p0[0]) / a2],
            ]);
        }
        let edge_len = mesh
            .boundary
            .iter()
            .map(|e| {
                let a = mesh.nodes[e.a as usize];
                let b = mesh.nodes[e.b as usize];
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
            })
            .collect();
        let mut dirichlet = vec![false; n];
        if no_slip {
            for e in &mesh.boundary {
                dirichlet[e.a as usize] = true;
                dirichlet[e.b as usize] = true;
            }
        }
        let mut entries = Vec::with_capacity(n + 6 * mesh.tris.len() + 3 * mesh.boundary.len());
        for i in 0..n {
            entries.push((i, i));
        }
        for t in &mesh.tris {
            for &(a, b) in &TRI_PAIRS {
                entries.push(lower(t[a], t[b]));
            }
        }
        if !no_slip {
            for e in &mesh.boundary {
                let v = [e.a, e.b];
                for &(a, b) in &EDGE_PAIRS {
                    entries.push(lower(v[a], v[b]));
                }
            }
        }
        let pattern = SpdPattern::new(n, &entries)?;
        Ok(Self { mesh: mesh.clone(), area, grads, edge_len, no_slip, dirichlet, pattern, threads: threads.max(1) })
    }

    pub fn mesh(&self) -> &TriMesh2 {
        &self.mesh
    }

    pub fn no_slip(&self) -> bool {
        self.no_slip
    }

    /// Solves for the exit speed. `warm` is a previous solution on the same
    /// mesh; it is used as the starting point when given. `cancel` is
    /// polled between Newton steps.
    pub fn solve(
        &self,
        params: &FlowParams,
        warm: Option<&FlowSolution>,
        cancel: &(dyn Fn() -> bool + Sync),
    ) -> Result<FlowSolution, FlowError> {
        let t0 = std::time::Instant::now();
        let mut stats = SolveStats::default();
        let rh = params.rheology;
        let warm = warm.filter(|s| s.w.len() == self.mesh.nodes.len() && s.w.iter().all(|x| x.is_finite()));
        let (w, dp, gamma_ref, w_ref, converged, residual) = match params.drive {
            Drive::Pressure(dp) => {
                let (w, g, wr, conv, res) = self.solve_pressure(&rh, dp, warm, &mut stats, cancel)?;
                (w, dp, g, wr, conv, res)
            }
            Drive::FlowRate(q) => self.solve_flow_rate(&rh, q, warm, &mut stats, cancel)?,
        };
        stats.seconds = t0.elapsed().as_secs_f64();
        Ok(self.finish(w, dp, gamma_ref, w_ref, converged, residual, stats, rh))
    }

    fn solve_flow_rate(
        &self,
        rh: &Rheology,
        q_target: f64,
        warm: Option<&FlowSolution>,
        stats: &mut SolveStats,
        cancel: &(dyn Fn() -> bool + Sync),
    ) -> Result<(Vec<f64>, f64, f64, f64, bool, f64), FlowError> {
        let q_target = q_target.max(1e-12);
        let mut dp = warm.map(|s| s.dp_kpa).filter(|d| *d > 0.0).unwrap_or(1000.0);
        let mut start: Option<FlowSolution> = warm.cloned();
        let mut history: Vec<(f64, f64)> = Vec::new();
        let slope0 = 1.0 / rh.n.max(0.05);
        for it in 0..40 {
            stats.pressure_iterations = it + 1;
            let (w, g, wr, conv, res) = self.solve_pressure(rh, dp, start.as_ref(), stats, cancel)?;
            let q = self.total_flow(&w);
            history.push((dp.ln(), q.max(1e-300).ln()));
            let rel = q / q_target - 1.0;
            if rel.abs() < 1e-6 || it == 39 {
                return Ok((w, dp, g, wr, conv && rel.abs() < 1e-3, res));
            }
            // Secant step in log-log space; power-law slope to start.
            let slope = if history.len() >= 2 {
                let (x1, y1) = history[history.len() - 1];
                let (x0, y0) = history[history.len() - 2];
                let s = (y1 - y0) / (x1 - x0);
                if s.is_finite() && s > 0.2 { s.min(10.0 * slope0) } else { slope0 }
            } else {
                slope0
            };
            let step = ((q_target.ln() - q.max(1e-300).ln()) / slope).clamp(-4.0, 4.0);
            let new_dp = dp * step.exp();
            // Warm start: scale the speeds by the expected flow ratio.
            let scale = (step * slope).exp();
            let mut guess = self.finish(
                w.iter().map(|x| x * scale).collect(),
                new_dp,
                g,
                wr * scale,
                conv,
                res,
                SolveStats::default(),
                *rh,
            );
            guess.dp_kpa = new_dp;
            start = Some(guess);
            dp = new_dp;
        }
        unreachable!()
    }

    /// Returns (w, gamma_ref, w_ref, converged, residual).
    fn solve_pressure(
        &self,
        rh: &Rheology,
        dp: f64,
        warm: Option<&FlowSolution>,
        stats: &mut SolveStats,
        cancel: &(dyn Fn() -> bool + Sync),
    ) -> Result<(Vec<f64>, f64, f64, bool, f64), FlowError> {
        let slip = !self.no_slip;
        let final_stage = |g_ref: f64, w_ref: f64| Stage {
            n: rh.n,
            k: rh.k,
            tau_y: rh.tau_y,
            slip,
            beta: rh.beta,
            m: rh.m,
            dp,
            eps: 1e-3 * g_ref,
            delta: 1e-3 * w_ref,
        };
        if let Some(ws) = warm
            && ws.rheology.n > 0.0 {
                let mut w = ws.w.clone();
                let st = final_stage(ws.gamma_ref, ws.w_ref);
                let (conv, res) = self.newton(&mut w, &st, 1e-12, 60, stats, cancel)?;
                if conv {
                    let (g, wr) = self.scales(&w);
                    return Ok((w, g, wr, true, res));
                }
            }
        stats.cold_start = true;
        // Newtonian start with slip linearised, then rescaled twice so the
        // effective viscosity and wall friction match the real ones.
        let mut w = vec![0.0; self.mesh.nodes.len()];
        let mut g_ref: f64 = 1.0;
        let mut w_ref: f64 = 1.0;
        for _ in 0..3 {
            let st = Stage {
                n: 1.0,
                k: rh.k * g_ref.powf(rh.n - 1.0),
                tau_y: 0.0,
                slip,
                beta: rh.beta * w_ref.powf(rh.m - 1.0),
                m: 1.0,
                dp,
                eps: 1e-3 * g_ref,
                delta: 1e-3 * w_ref,
            };
            self.newton(&mut w, &st, 1e-14, 4, stats, cancel)?;
            let (g, wr) = self.scales(&w);
            g_ref = g.max(1e-12);
            w_ref = wr.max(1e-12);
        }
        let stages = 5;
        let mut last = (false, f64::INFINITY);
        for s in 1..=stages {
            let f = s as f64 / stages as f64;
            let n_s = 1.0 + f * (rh.n - 1.0);
            let m_s = 1.0 + f * (rh.m - 1.0);
            let st = Stage {
                n: n_s,
                k: rh.k * g_ref.powf(rh.n - n_s),
                tau_y: f * rh.tau_y,
                slip,
                beta: rh.beta * w_ref.powf(rh.m - m_s),
                m: m_s,
                dp,
                eps: g_ref * 10f64.powf(-1.0 - 2.0 * f),
                delta: w_ref * 10f64.powf(-1.0 - 2.0 * f),
            };
            let tol = if s == stages { 1e-12 } else { 1e-6 };
            last = self.newton(&mut w, &st, tol, 80, stats, cancel)?;
            let (g, wr) = self.scales(&w);
            g_ref = g.max(1e-12);
            w_ref = wr.max(1e-12);
        }
        Ok((w, g_ref, w_ref, last.0, last.1))
    }

    /// Area-weighted mean shear rate and mean speed.
    fn scales(&self, w: &[f64]) -> (f64, f64) {
        let mut ag = 0.0;
        let mut aw = 0.0;
        let mut at = 0.0;
        for (t, tri) in self.mesh.tris.iter().enumerate() {
            let g = self.grad_w(t, w);
            let a = self.area[t];
            ag += a * (g[0] * g[0] + g[1] * g[1]).sqrt();
            aw += a * (w[tri[0] as usize] + w[tri[1] as usize] + w[tri[2] as usize]).abs() / 3.0;
            at += a;
        }
        (ag / at, aw / at)
    }

    fn grad_w(&self, t: usize, w: &[f64]) -> [f64; 2] {
        let tri = self.mesh.tris[t];
        let g = &self.grads[t];
        let mut out = [0.0; 2];
        for q in 0..3 {
            let wi = w[tri[q] as usize];
            out[0] += wi * g[q][0];
            out[1] += wi * g[q][1];
        }
        out
    }

    fn total_flow(&self, w: &[f64]) -> f64 {
        self.mesh
            .tris
            .iter()
            .enumerate()
            .map(|(t, tri)| self.area[t] * (w[tri[0] as usize] + w[tri[1] as usize] + w[tri[2] as usize]) / 3.0)
            .sum()
    }

    fn energy(&self, w: &[f64], st: &Stage) -> f64 {
        let body = |t: usize| -> f64 {
            let tri = self.mesh.tris[t];
            let g = self.grad_w(t, w);
            let gam = (g[0] * g[0] + g[1] * g[1] + st.eps * st.eps).sqrt();
            let psi = st.k / (st.n + 1.0) * gam.powf(st.n + 1.0) + st.tau_y * gam;
            let wbar = (w[tri[0] as usize] + w[tri[1] as usize] + w[tri[2] as usize]) / 3.0;
            self.area[t] * (psi - st.dp / self.mesh.tri_land[t] * wbar)
        };
        let mut e: f64 = if self.threads > 1 {
            (0..self.mesh.tris.len()).into_par_iter().map(body).sum()
        } else {
            (0..self.mesh.tris.len()).map(body).sum()
        };
        if st.slip {
            for (k, edge) in self.mesh.boundary.iter().enumerate() {
                let beta = st.beta * edge.friction_scale;
                let (wa, wb) = (w[edge.a as usize], w[edge.b as usize]);
                for xi in [-1.0 / 3f64.sqrt(), 1.0 / 3f64.sqrt()] {
                    let wq = 0.5 * (1.0 - xi) * wa + 0.5 * (1.0 + xi) * wb;
                    e += 0.5 * self.edge_len[k] * beta / (st.m + 1.0) * (wq * wq + st.delta * st.delta).powf(0.5 * (st.m + 1.0));
                }
            }
        }
        e
    }

    /// Gradient and Hessian values in the pattern's entry order.
    fn assemble(&self, w: &[f64], st: &Stage, grad: &mut [f64], vals: &mut [f64]) {
        let n = self.mesh.nodes.len();
        let nt = self.mesh.tris.len();
        grad.iter_mut().for_each(|g| *g = 0.0);
        for i in 0..n {
            vals[i] = if self.dirichlet[i] { 1.0 } else { 0.0 };
        }
        let tri_vals = &mut vals[n..n + 6 * nt];
        let fill = |t: usize, out: &mut [f64]| {
            let tri = self.mesh.tris[t];
            let g = &self.grads[t];
            let gw = self.grad_w(t, w);
            let gam2 = gw[0] * gw[0] + gw[1] * gw[1] + st.eps * st.eps;
            let gam = gam2.sqrt();
            let dpsi = st.k * gam.powf(st.n) + st.tau_y;
            let ddpsi = st.n * st.k * gam.powf(st.n - 1.0);
            let a = dpsi / gam;
            let b = (ddpsi - a) / gam2;
            let area = self.area[t];
            for (slot, &(p, q)) in TRI_PAIRS.iter().enumerate() {
                let (ip, iq) = (tri[p] as usize, tri[q] as usize);
                if self.dirichlet[ip] || self.dirichlet[iq] {
                    out[slot] = 0.0;
                    continue;
                }
                let gp = g[p];
                let gq = g[q];
                let dot = gp[0] * gq[0] + gp[1] * gq[1];
                let proj = (gp[0] * gw[0] + gp[1] * gw[1]) * (gq[0] * gw[0] + gq[1] * gw[1]);
                out[slot] = area * (a * dot + b * proj);
            }
        };
        if self.threads > 1 {
            tri_vals.par_chunks_mut(6).enumerate().for_each(|(t, out)| fill(t, out));
        } else {
            tri_vals.chunks_mut(6).enumerate().for_each(|(t, out)| fill(t, out));
        }
        for t in 0..nt {
            let tri = self.mesh.tris[t];
            let g = &self.grads[t];
            let gw = self.grad_w(t, w);
            let gam = (gw[0] * gw[0] + gw[1] * gw[1] + st.eps * st.eps).sqrt();
            let a = (st.k * gam.powf(st.n) + st.tau_y) / gam;
            let force = st.dp / self.mesh.tri_land[t] / 3.0;
            for q in 0..3 {
                grad[tri[q] as usize] += self.area[t] * (a * (gw[0] * g[q][0] + gw[1] * g[q][1]) - force);
            }
        }
        if st.slip {
            let base = n + 6 * nt;
            for (k, edge) in self.mesh.boundary.iter().enumerate() {
                let beta = st.beta * edge.friction_scale;
                let ids = [edge.a as usize, edge.b as usize];
                let (wa, wb) = (w[ids[0]], w[ids[1]]);
                let mut h = [0.0; 3];
                for xi in [-1.0 / 3f64.sqrt(), 1.0 / 3f64.sqrt()] {
                    let nn = [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)];
                    let wq = nn[0] * wa + nn[1] * wb;
                    let r2 = wq * wq + st.delta * st.delta;
                    let d1 = beta * wq * r2.powf(0.5 * (st.m - 1.0));
                    let d2 = beta * r2.powf(0.5 * (st.m - 3.0)) * (st.delta * st.delta + st.m * wq * wq);
                    let wgt = 0.5 * self.edge_len[k];
                    grad[ids[0]] += wgt * d1 * nn[0];
                    grad[ids[1]] += wgt * d1 * nn[1];
                    h[0] += wgt * d2 * nn[0] * nn[0];
                    h[1] += wgt * d2 * nn[1] * nn[1];
                    h[2] += wgt * d2 * nn[0] * nn[1];
                }
                vals[base + 3 * k..base + 3 * k + 3].copy_from_slice(&h);
            }
        }
        for i in 0..n {
            if self.dirichlet[i] {
                grad[i] = 0.0;
            }
        }
    }

    /// Damped Newton. Returns (converged, final relative decrement).
    fn newton(
        &self,
        w: &mut [f64],
        st: &Stage,
        tol: f64,
        max_steps: usize,
        stats: &mut SolveStats,
        cancel: &(dyn Fn() -> bool + Sync),
    ) -> Result<(bool, f64), FlowError> {
        let n = w.len();
        for i in 0..n {
            if self.dirichlet[i] {
                w[i] = 0.0;
            }
        }
        let mut grad = vec![0.0; n];
        let mut vals = vec![0.0; self.pattern.num_entries()];
        let mut e = self.energy(w, st);
        let mut trial = vec![0.0; n];
        let mut last_dec = f64::INFINITY;
        for _ in 0..max_steps {
            if cancel() {
                return Err(FlowError::Cancelled);
            }
            self.assemble(w, st, &mut grad, &mut vals);
            let mut d: Vec<f64> = grad.iter().map(|g| -g).collect();
            self.pattern.solve(&vals, &mut d, par(self.threads))?;
            stats.newton_steps += 1;
            let slope: f64 = grad.iter().zip(&d).map(|(g, d)| g * d).sum();
            // Energy scale: the work done by the pressure.
            let scale = (e.abs()).max(1e-300);
            last_dec = -slope / scale;
            if !(slope < 0.0) || last_dec < tol {
                return Ok((true, last_dec.max(0.0)));
            }
            let mut t = 1.0;
            let mut accepted = false;
            for _ in 0..40 {
                for i in 0..n {
                    trial[i] = w[i] + t * d[i];
                }
                let et = self.energy(&trial, st);
                if et.is_finite() && et <= e + 1e-4 * t * slope {
                    w.copy_from_slice(&trial);
                    e = et;
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }
            if !accepted {
                // No descent possible at machine precision: converged as far
                // as it can be.
                return Ok((last_dec < 1e-6, last_dec));
            }
        }
        Ok((last_dec < tol.max(1e-9), last_dec))
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        w: Vec<f64>,
        dp: f64,
        gamma_ref: f64,
        w_ref: f64,
        converged: bool,
        residual: f64,
        stats: SolveStats,
        rheology: Rheology,
    ) -> FlowSolution {
        let m = &self.mesh;
        let nop = m.openings.len();
        let mut q_opening = vec![0.0; nop];
        let mut area_opening = vec![0.0; nop];
        let mut plug_area = vec![0.0; nop];
        let mut tri_shear = Vec::with_capacity(m.tris.len());
        let eps = 1e-3 * gamma_ref;
        for (t, tri) in m.tris.iter().enumerate() {
            let k = m.tri_opening[t] as usize;
            let a = self.area[t];
            q_opening[k] += a * (w[tri[0] as usize] + w[tri[1] as usize] + w[tri[2] as usize]) / 3.0;
            area_opening[k] += a;
            let g = self.grad_w(t, &w);
            let s = (g[0] * g[0] + g[1] * g[1]).sqrt();
            tri_shear.push(s);
            if rheology.tau_y > 0.0 && s < 10.0 * eps {
                plug_area[k] += a;
            }
        }
        let mean_speed: Vec<f64> = q_opening.iter().zip(&area_opening).map(|(q, a)| q / a.max(1e-300)).collect();
        let wall_slip_ratio = m
            .boundary
            .iter()
            .map(|e| 0.5 * (w[e.a as usize] + w[e.b as usize]) / mean_speed[e.opening as usize].max(1e-300))
            .collect();
        // Recovery-based error estimate.
        let n = m.nodes.len();
        let mut rec = vec![[0.0f64; 2]; n];
        let mut wt = vec![0.0f64; n];
        for (t, tri) in m.tris.iter().enumerate() {
            let g = self.grad_w(t, &w);
            for &i in tri {
                rec[i as usize][0] += self.area[t] * g[0];
                rec[i as usize][1] += self.area[t] * g[1];
                wt[i as usize] += self.area[t];
            }
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for (t, tri) in m.tris.iter().enumerate() {
            let g = self.grad_w(t, &w);
            let mut r = [0.0; 2];
            for &i in tri {
                let i = i as usize;
                r[0] += rec[i][0] / wt[i] / 3.0;
                r[1] += rec[i][1] / wt[i] / 3.0;
            }
            num += self.area[t] * ((r[0] - g[0]).powi(2) + (r[1] - g[1]).powi(2));
            den += self.area[t] * (g[0] * g[0] + g[1] * g[1]);
        }
        FlowSolution {
            q_total: q_opening.iter().sum(),
            plug_fraction: plug_area.iter().zip(&area_opening).map(|(p, a)| p / a.max(1e-300)).collect(),
            q_opening,
            mean_speed,
            tri_shear,
            wall_slip_ratio,
            error_estimate: (num / den.max(1e-300)).sqrt(),
            residual,
            converged,
            stats,
            gamma_ref,
            w_ref,
            rheology,
            dp_kpa: dp,
            w,
        }
    }
}

impl FlowSolution {
    /// Linear interpolation of the speed at a point inside triangle t.
    pub fn speed_in_triangle(&self, mesh: &TriMesh2, t: usize, p: [f64; 2]) -> f64 {
        let [a, b, c] = mesh.tris[t].map(|i| mesh.nodes[i as usize]);
        let det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        let l1 = ((p[0] - a[0]) * (c[1] - a[1]) - (p[1] - a[1]) * (c[0] - a[0])) / det;
        let l2 = ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / det;
        let l0 = 1.0 - l1 - l2;
        let [i0, i1, i2] = mesh.tris[t].map(|i| i as usize);
        l0 * self.w[i0] + l1 * self.w[i1] + l2 * self.w[i2]
    }
}
