use crate::energy::{EDGE, FaceBend, TriRest, dihedral, wrap};
use glam::DVec3;
use pasta_linalg::{Rng, SolveError, SpdPattern, par};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Exit-speed profile of one slot, as the shell needs it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StripInput {
    pub closed: bool,
    /// Station positions on the die face (mm).
    pub centre: Vec<[f64; 2]>,
    /// In-plane unit normal at each station.
    pub normal: Vec<[f64; 2]>,
    /// Exit speed after swell (mm/s).
    pub v: Vec<f64>,
    /// Sheet thickness after swell (mm).
    pub thickness: Vec<f64>,
    /// Natural curvature along the extrusion direction (1/mm), positive when
    /// the +normal face runs faster.
    pub bend_curvature: Vec<f64>,
}

impl StripInput {
    pub fn len(&self) -> usize {
        self.centre.len()
    }

    pub fn is_empty(&self) -> bool {
        self.centre.is_empty()
    }

    /// Resamples to `n` stations equally spaced in arc length.
    pub fn resample(&self, n: usize) -> StripInput {
        self.resample_refined(n, 1.0)
    }

    /// Resamples to `n` stations, placed more densely where the exit speed
    /// changes fastest: up to `max_ratio` times the density of the
    /// smoothest region. Ruffles live in the fast bands, which can be only a
    /// millimetre or two wide.
    pub fn resample_refined(&self, n: usize, max_ratio: f64) -> StripInput {
        let m = self.centre.len();
        let n = n.max(3);
        let seg = |k: usize| {
            let a = self.centre[k];
            let b = self.centre[(k + 1) % m];
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };
        let segs = if self.closed { m } else { m - 1 };
        // Station density on the input samples, from |dv/ds|.
        let vbar = self.v.iter().sum::<f64>() / m as f64;
        let mut grad = vec![0.0; m];
        for i in 0..m {
            let (a, b) = if self.closed {
                ((i + m - 1) % m, (i + 1) % m)
            } else {
                (i.saturating_sub(1), (i + 1).min(m - 1))
            };
            let ds = if self.closed || (i > 0 && i + 1 < m) { seg(a) + seg(i % segs.max(1)) } else { seg(i.min(segs - 1)) };
            grad[i] = ((self.v[b] - self.v[a]) / ds.max(1e-12)).abs() / vbar.max(1e-12);
        }
        // Smooth over about 3% of the stations so the spacing varies slowly.
        let w = ((m as f64 * 0.03) as isize).max(1);
        let idx = |i: isize| -> usize {
            if self.closed { i.rem_euclid(m as isize) as usize } else { i.clamp(0, m as isize - 1) as usize }
        };
        let smooth: Vec<f64> = (0..m as isize)
            .map(|i| (-w..=w).map(|o| grad[idx(i + o)]).fold(0.0, f64::max))
            .collect();
        let gmax = smooth.iter().cloned().fold(0.0, f64::max);
        let ratio = max_ratio.max(1.0);
        let density: Vec<f64> =
            smooth.iter().map(|g| if gmax > 0.0 { 1.0 + (ratio - 1.0) * g / gmax } else { 1.0 }).collect();
        let mut wcum = vec![0.0];
        for k in 0..segs {
            let l = seg(k);
            wcum.push(wcum[k] + l * 0.5 * (density[k] + density[(k + 1) % m]));
        }
        let total_w = wcum[segs];
        let mut out = StripInput {
            closed: self.closed,
            centre: Vec::with_capacity(n),
            normal: Vec::with_capacity(n),
            v: Vec::with_capacity(n),
            thickness: Vec::with_capacity(n),
            bend_curvature: Vec::with_capacity(n),
        };
        let mut k = 0;
        for q in 0..n {
            let target = if self.closed { total_w * q as f64 / n as f64 } else { total_w * q as f64 / (n - 1) as f64 };
            while k + 1 < segs && wcum[k + 1] < target {
                k += 1;
            }
            let len = (wcum[k + 1] - wcum[k]).max(1e-300);
            let t = ((target - wcum[k]) / len).clamp(0.0, 1.0);
            let k1 = (k + 1) % m;
            let lerp = |a: f64, b: f64| a + t * (b - a);
            out.centre.push([lerp(self.centre[k][0], self.centre[k1][0]), lerp(self.centre[k][1], self.centre[k1][1])]);
            let nx = lerp(self.normal[k][0], self.normal[k1][0]);
            let ny = lerp(self.normal[k][1], self.normal[k1][1]);
            let nl = (nx * nx + ny * ny).sqrt().max(1e-300);
            out.normal.push([nx / nl, ny / nl]);
            out.v.push(lerp(self.v[k], self.v[k1]));
            out.thickness.push(lerp(self.thickness[k], self.thickness[k1]));
            out.bend_curvature.push(lerp(self.bend_curvature[k], self.bend_curvature[k1]));
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Orientation {
    /// Extruded straight down: gravity along the extrusion direction.
    #[default]
    Down,
    /// Extruded horizontally over a support surface: gravity along −y.
    Horizontal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShellParams {
    /// Stations across the slot (columns of nodes).
    pub stations: usize,
    /// Stations are packed up to this many times more densely where the
    /// exit speed changes fastest (1 = evenly spaced).
    #[serde(default = "default_refine")]
    pub refine: f64,
    /// Elastic modulus of the fresh dough (kPa).
    pub young_kpa: f64,
    pub poisson: f64,
    /// Dough density (g/cm³).
    pub density: f64,
    pub orientation: Orientation,
    /// Length of the piece when it is cut (mm).
    pub cut_length: f64,
    /// Time steps per row of elements.
    pub substeps: usize,
    /// Creep time constant (s); rest shapes drift toward the current shape.
    /// None = purely elastic.
    pub creep_time: Option<f64>,
    /// Distance from the lowest point of the opening down to the support
    /// surface (mm), horizontal extrusion only.
    pub support_drop: f64,
    /// Size of the random kick that lets buckling start, as a fraction of
    /// the thickness.
    pub perturbation: f64,
    pub seed: u64,
    /// Newton iterations per time step at most.
    pub max_newton: usize,
    /// Numerical damping: weight of the pull toward the extrapolated shape,
    /// per unit area, in units of the flexural rigidity per mm⁴. Small
    /// values give a quasi-static path; it only has to pick a branch when
    /// the sheet buckles.
    pub damping: f64,
    /// After cutting, relax the piece with gravity off.
    pub relax_after_cut: bool,
    /// Threads for assembly and factorisation (1 = sequential).
    pub threads: usize,
}

fn default_refine() -> f64 {
    4.0
}

impl Default for ShellParams {
    fn default() -> Self {
        Self {
            stations: 48,
            refine: default_refine(),
            young_kpa: 50.0,
            poisson: 0.45,
            density: 1.3,
            orientation: Orientation::Down,
            cut_length: 80.0,
            substeps: 2,
            creep_time: None,
            support_drop: 5.0,
            perturbation: 0.02,
            seed: 1,
            max_newton: 10,
            damping: 1e-6,
            relax_after_cut: true,
            threads: 1,
        }
    }
}

#[derive(Debug)]
pub enum ShellError {
    Cancelled,
    Solve(SolveError),
    BadInput(String),
}

impl std::fmt::Display for ShellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellError::Cancelled => write!(f, "cancelled"),
            ShellError::Solve(e) => write!(f, "{e}"),
            ShellError::BadInput(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for ShellError {}

impl From<SolveError> for ShellError {
    fn from(e: SolveError) -> Self {
        ShellError::Solve(e)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ShellStats {
    pub steps: usize,
    pub rows: usize,
    pub newton_iterations: usize,
    pub pattern_builds: usize,
    pub unconverged_steps: usize,
    pub seconds: f64,
    pub nodes: usize,
    pub triangles: usize,
    /// Largest rest dihedral angle among frozen edges (rad), a check that
    /// the rest shape is what the die imposed.
    pub max_rest_angle: f64,
    /// Smallest and largest frozen rest edge length (mm).
    pub rest_edge_range: (f64, f64),
    /// Seconds spent assembling, factorising and evaluating energies.
    pub time_assemble: f64,
    pub time_solve: f64,
    pub time_energy: f64,
    /// Newton systems that needed a diagonal shift (indefinite Hessian).
    pub shifted_solves: usize,
}

/// Positions at one moment, for the timeline.
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub time: f64,
    /// Positions of the first `positions.len()` nodes.
    pub positions: Arc<Vec<[f32; 3]>>,
    /// Number of triangles that existed then (a prefix of `triangles`).
    pub triangles: usize,
}

/// Progress report passed to the caller during a run.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub time: f64,
    pub piece_length: f64,
    pub keyframe: Keyframe,
    pub stats: ShellStats,
}

#[derive(Clone, Debug)]
pub struct ShellResult {
    pub stations: usize,
    pub closed: bool,
    /// All triangles, in creation order (node indices).
    pub triangles: Vec<[u32; 3]>,
    /// Station of each node; row = index / stations.
    pub node_station: Vec<u32>,
    /// Shape when cut, hanging from the die.
    pub hanging: Vec<[f32; 3]>,
    /// Shape after relaxing with gravity off, if requested.
    pub relaxed: Option<Vec<[f32; 3]>>,
    /// Nodes that were outside the die when the piece was cut.
    pub outside: Vec<bool>,
    pub keyframes: Vec<Keyframe>,
    pub stats: ShellStats,
    /// The resampled input the run used.
    pub input: StripInput,
    pub thickness_at_node: Vec<f32>,
    pub speed_at_node: Vec<f32>,
}

impl ShellResult {
    /// Triangles whose three nodes are all outside the die (`outside` for
    /// the final shapes).
    pub fn outside_triangles(&self) -> Vec<[u32; 3]> {
        self.triangles
            .iter()
            .copied()
            .filter(|t| t.iter().all(|&i| self.outside.get(i as usize).copied().unwrap_or(false)))
            .collect()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeState {
    /// Inside the die, moved by the die.
    Driven,
    Free,
    /// Cut off with the die remnant.
    Removed,
}

const NO_OFF: u32 = u32::MAX;

struct TriElem {
    rest: TriRest,
    bend: FaceBend,
    frozen: bool,
    /// Value offsets of node-pair blocks (a, b) with global a >= b, for the
    /// stretching element (3 x 3) and the bending stencil (up to 6 x 6).
    off: [u32; 9],
    bend_off: [u32; 36],
    offsets_valid: bool,
}

/// Rest angle of one interior edge; the bending energy lives in the faces.
struct HingeElem {
    v: [u32; 4],
    rest_angle: f64,
    frozen: bool,
}

/// Pairwise block assembly into a fixed sparse pattern.
struct Pattern {
    rows_capacity: usize,
    /// Index of each unique node pair (a >= b); diagonal pairs come first,
    /// so the pair index of (i, i) is i.
    pair_index: std::collections::HashMap<(u32, u32), u32>,
    spd: SpdPattern,
    /// Offset of each pair's entries in the value array: 9 entries when
    /// a > b, 6 (lower triangle) when a == b.
    offset: Vec<u32>,
}

pub struct ShellSim {
    p: ShellParams,
    input: StripInput,
    n_s: usize,
    x: Vec<DVec3>,
    /// Positions at the end of the previous step.
    x_old: Vec<DVec3>,
    /// Where the damping pulls each node: the position extrapolated at
    /// constant velocity, so steady motion costs nothing.
    x_target: Vec<DVec3>,
    state: Vec<NodeState>,
    row_birth: Vec<f64>,
    /// Weight of each node (N), assigned when its elements leave the die.
    mass: Vec<f64>,
    /// Lumped rest area of each node (mm²).
    node_area: Vec<f64>,
    tris: Vec<TriElem>,
    hinges: Vec<HingeElem>,
    /// Directed edge -> triangle that owns it.
    edge_owner: std::collections::HashMap<(u32, u32), u32>,
    time: f64,
    dt_row: f64,
    dt: f64,
    z_spawn: f64,
    next_row: usize,
    first_release: Option<f64>,
    mu: f64,
    lambda: f64,
    young: f64,
    nu: f64,
    weight_density: f64,
    gravity: DVec3,
    support_y: f64,
    prox_coef: f64,
    /// Mean station spacing (mm).
    cell: f64,
    contact_k: f64,
    pattern: Option<Pattern>,
    rng: Rng,
    stats: ShellStats,
    keyframes: Vec<Keyframe>,
    cut: bool,
    gravity_on: bool,
}

/// Rows of headroom in the sparse pattern, so it is rebuilt only every
/// this many rows.
const ROW_CHUNK: usize = 24;

/// Systems larger than this use several threads for the factorisation.
const PAR_DOFS: usize = 12_000;

fn rest_len2(pts: &[DVec3; 3]) -> [f64; 3] {
    std::array::from_fn(|k| {
        let (a, b) = EDGE[k];
        (pts[b] - pts[a]).length_squared()
    })
}

impl ShellSim {
    pub fn new(input: &StripInput, params: &ShellParams) -> Result<Self, ShellError> {
        if input.len() < 3 {
            return Err(ShellError::BadInput("the slot profile needs at least 3 stations".into()));
        }
        if input.v.iter().any(|v| !(*v > 0.0)) {
            return Err(ShellError::BadInput("every station needs a positive exit speed".into()));
        }
        let n_s = params.stations.max(4);
        let input = input.resample_refined(n_s, params.refine);
        let p = params.clone();
        let segs = if input.closed { n_s } else { n_s - 1 };
        let mut ds = 0.0;
        for i in 0..segs {
            let a = input.centre[i];
            let b = input.centre[(i + 1) % n_s];
            ds += ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        }
        ds /= segs as f64;
        let mut vs = input.v.clone();
        vs.sort_by(|a, b| a.total_cmp(b));
        let v_med = vs[vs.len() / 2];
        let v_max = vs[vs.len() - 1];
        // Row spacing equals the station spacing at the median speed.
        let dt_row = ds / v_med;
        let young = p.young_kpa * 1e-3; // N/mm²
        let nu = p.poisson.clamp(0.0, 0.49);
        let mu = young / (2.0 * (1.0 + nu));
        let lambda = young * nu / (1.0 - nu * nu);
        // g/cm³ -> t/mm³ is 1e-9; times g = 9810 mm/s² gives N/mm³.
        let weight_density = p.density * 1e-9 * 9810.0;
        let gravity = match p.orientation {
            Orientation::Down => DVec3::new(0.0, 0.0, -1.0),
            Orientation::Horizontal => DVec3::new(0.0, -1.0, 0.0),
        };
        let y_min = input.centre.iter().map(|c| c[1]).fold(f64::INFINITY, f64::min);
        let h_mean = input.thickness.iter().sum::<f64>() / n_s as f64;
        let d_ref = young * h_mean.powi(3) / (12.0 * (1.0 - nu * nu));
        let m_ref = weight_density * h_mean * ds * ds;
        let mut sim = Self {
            input,
            n_s,
            x: Vec::new(),
            x_old: Vec::new(),
            x_target: Vec::new(),
            state: Vec::new(),
            row_birth: Vec::new(),
            mass: Vec::new(),
            node_area: Vec::new(),
            tris: Vec::new(),
            hinges: Vec::new(),
            edge_owner: Default::default(),
            time: 0.0,
            dt_row,
            dt: dt_row / p.substeps.max(1) as f64,
            z_spawn: 3.0 * v_max * dt_row,
            next_row: 0,
            first_release: None,
            mu,
            lambda,
            young,
            nu,
            weight_density,
            gravity,
            support_y: y_min - h_mean - p.support_drop,
            // Per unit area and independent of the mesh: D / (1 mm)^4.
            prox_coef: p.damping * d_ref,
            cell: ds,
            contact_k: m_ref / (0.02 * h_mean).powi(2),
            pattern: None,
            rng: Rng::new(p.seed),
            stats: ShellStats::default(),
            keyframes: Vec::new(),
            cut: false,
            gravity_on: true,
            p,
        };
        // Prime the die: rows already spread between the spawn plane and the
        // lip, so material starts leaving straight away.
        let rows_in_die = (sim.z_spawn / (v_max * dt_row)).floor() as usize;
        for k in 0..rows_in_die {
            let birth = -((rows_in_die - 1 - k) as f64) * dt_row;
            sim.spawn_row(birth);
        }
        Ok(sim)
    }

    fn flexural_rigidity(&self, h: f64) -> f64 {
        self.young * h.powi(3) / (12.0 * (1.0 - self.nu * self.nu))
    }

    fn total_nodes(&self) -> usize {
        self.x.len()
    }

    fn driven_position(&self, idx: usize, t: f64) -> DVec3 {
        let i = idx % self.n_s;
        let r = idx / self.n_s;
        let c = self.input.centre[i];
        DVec3::new(c[0], c[1], self.z_spawn - self.input.v[i] * (t - self.row_birth[r]))
    }

    /// Triangles of strip r (between rows r and r+1). Every quad has its
    /// diagonal in the same direction; with face-based bending the pattern
    /// does not need to alternate.
    fn strip_tris(&self, r: usize) -> Vec<[u32; 3]> {
        let n_s = self.n_s;
        let segs = if self.input.closed { n_s } else { n_s - 1 };
        let mut out = Vec::with_capacity(2 * segs);
        for i in 0..segs {
            let i1 = (i + 1) % n_s;
            let a = (r * n_s + i) as u32;
            let b = (r * n_s + i1) as u32;
            let c = ((r + 1) * n_s + i1) as u32;
            let d = ((r + 1) * n_s + i) as u32;
            out.push([a, b, c]);
            out.push([a, c, d]);
        }
        out
    }

    fn spawn_row(&mut self, birth: f64) {
        let r = self.next_row;
        self.next_row += 1;
        self.row_birth.push(birth);
        for i in 0..self.n_s {
            let idx = r * self.n_s + i;
            let pos = self.driven_position(idx, self.time.max(birth));
            self.x.push(pos);
            self.x_old.push(pos);
            self.x_target.push(pos);
            self.state.push(NodeState::Driven);
            self.mass.push(0.0);
            self.node_area.push(0.0);
        }
        self.stats.rows = self.next_row;
        if r == 0 {
            return;
        }
        // Bring the previous rows to the current time so the new elements
        // start from the die's geometry (molten updates keep them there).
        for idx in 0..self.total_nodes() {
            if self.state[idx] == NodeState::Driven {
                self.x[idx] = self.driven_position(idx, self.time.max(birth));
            }
        }
        for t in self.strip_tris(r - 1) {
            let ti = self.tris.len() as u32;
            let th = t.iter().map(|&i| self.input.thickness[i as usize % self.n_s]).sum::<f64>() / 3.0;
            let rest = rest_len2(&t.map(|i| self.x[i as usize]));
            let elem = TriRest::new(t, rest, th, self.mu, self.lambda);
            let bend = FaceBend { nodes: t.to_vec(), hinge: [None; 3], b: FaceBend::b_matrix(rest, self.flexural_rigidity(th), self.nu) };
            // Hinges with already-existing neighbours across each edge.
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                self.edge_owner.insert((a, b), ti);
            }
            self.tris.push(TriElem { rest: elem, bend, frozen: false, off: [NO_OFF; 9], bend_off: [NO_OFF; 36], offsets_valid: false });
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                if let Some(&other) = self.edge_owner.get(&(b, a)) {
                    // The neighbour traverses (b, a) with its third vertex p.
                    let ov = self.tris[other as usize].rest.v;
                    let p = ov.iter().copied().find(|&w| w != a && w != b).unwrap();
                    let q = t[(k + 2) % 3];
                    self.add_hinge([b, a, p, q], other, ti);
                }
            }
        }
        // Prune edge ownership of strips that can no longer gain neighbours.
        if r >= 3 {
            let old_row = (r - 3) as u32 * self.n_s as u32;
            self.edge_owner.retain(|&(a, b), _| a >= old_row || b >= old_row);
        }
    }

    /// Registers hinge [a, b, p, q] (face1 = (a, b, p), face2 = (b, a, q))
    /// with both faces.
    fn add_hinge(&mut self, quad: [u32; 4], face1: u32, face2: u32) {
        let hi = self.hinges.len() as u32;
        let mut h = HingeElem { v: quad, rest_angle: 0.0, frozen: false };
        h.rest_angle = self.hinge_rest_angle(&quad);
        self.hinges.push(h);
        for face in [face1, face2] {
            let tri = &mut self.tris[face as usize];
            let v = tri.rest.v;
            let k = (0..3)
                .find(|&k| {
                    let (i, j) = EDGE[k];
                    (v[i] == quad[0] && v[j] == quad[1]) || (v[i] == quad[1] && v[j] == quad[0])
                })
                .expect("hinge edge belongs to the face");
            let loc = quad.map(|w| {
                if let Some(pos) = tri.bend.nodes.iter().position(|&n| n == w) {
                    pos as u8
                } else {
                    tri.bend.nodes.push(w);
                    (tri.bend.nodes.len() - 1) as u8
                }
            });
            tri.bend.hinge[k] = Some((loc, hi));
            tri.offsets_valid = false;
        }
    }

    /// Rest angle from the die-shaped configuration, plus the natural bend
    /// from the speed bias across the gap.
    fn hinge_rest_angle(&self, v: &[u32; 4]) -> f64 {
        let pts: [DVec3; 4] = v.map(|i| self.x[i as usize]);
        let kb = v.iter().map(|&i| self.input.bend_curvature[i as usize % self.n_s]).sum::<f64>() / 4.0;
        if kb == 0.0 {
            return dihedral(&pts);
        }
        let z_mid = 0.5 * (pts[0].z + pts[1].z);
        let bent: [DVec3; 4] = std::array::from_fn(|q| {
            let i = v[q] as usize % self.n_s;
            let n = self.input.normal[i];
            let dz = pts[q].z - z_mid;
            pts[q] - 0.5 * kb * dz * dz * DVec3::new(n[0], n[1], 0.0)
        });
        dihedral(&bent)
    }

    /// Builds the sparse pattern for rows up to the current capacity.
    fn ensure_pattern(&mut self) -> Result<(), ShellError> {
        let need = self.next_row;
        let rebuild = match &self.pattern {
            Some(p) => p.rows_capacity < need,
            None => true,
        };
        if rebuild {
            let cap = need + ROW_CHUNK;
            let n_cap = cap * self.n_s;
            let mut pair_index: std::collections::HashMap<(u32, u32), u32> = Default::default();
            let mut pairs: Vec<(u32, u32)> = Vec::new();
            let mut add = |a: u32, b: u32| {
                let key = if a >= b { (a, b) } else { (b, a) };
                pair_index.entry(key).or_insert_with(|| {
                    pairs.push(key);
                    (pairs.len() - 1) as u32
                });
            };
            for i in 0..n_cap {
                add(i as u32, i as u32);
            }
            // Bending stencils: each face with the opposite vertices across
            // its edges.
            let all: Vec<[u32; 3]> = (0..cap.saturating_sub(1)).flat_map(|r| self.strip_tris(r)).collect();
            let mut owner: std::collections::HashMap<(u32, u32), usize> = Default::default();
            for (ti, t) in all.iter().enumerate() {
                for k in 0..3 {
                    owner.insert((t[k], t[(k + 1) % 3]), ti);
                }
            }
            for t in &all {
                let mut stencil: Vec<u32> = t.to_vec();
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    if let Some(&o) = owner.get(&(b, a)) {
                        let p = all[o].iter().copied().find(|&w| w != a && w != b).unwrap();
                        stencil.push(p);
                    }
                }
                for i in 0..stencil.len() {
                    for j in 0..=i {
                        add(stencil[i], stencil[j]);
                    }
                }
            }
            let mut entries = Vec::with_capacity(pairs.len() * 9);
            let mut offset = Vec::with_capacity(pairs.len());
            for &(a, b) in &pairs {
                offset.push(entries.len() as u32);
                let (a, b) = (a as usize, b as usize);
                for i in 0..3 {
                    for j in 0..3 {
                        if a == b && j > i {
                            continue;
                        }
                        entries.push((3 * a + i, 3 * b + j));
                    }
                }
            }
            let spd = SpdPattern::new(3 * n_cap, &entries)?;
            self.stats.pattern_builds += 1;
            self.pattern = Some(Pattern { rows_capacity: cap, pair_index, spd, offset });
            for t in &mut self.tris {
                t.offsets_valid = false;
            }
        }
        let pat = self.pattern.as_ref().unwrap();
        let lookup = |a: u32, b: u32| -> u32 {
            if a >= b { pat.offset[pat.pair_index[&(a, b)] as usize] } else { NO_OFF }
        };
        for t in &mut self.tris {
            if t.offsets_valid {
                continue;
            }
            for a in 0..3 {
                for b in 0..3 {
                    t.off[3 * a + b] = lookup(t.rest.v[a], t.rest.v[b]);
                }
            }
            let n = t.bend.nodes.len();
            t.bend_off = [NO_OFF; 36];
            for a in 0..n {
                for b in 0..n {
                    t.bend_off[n * a + b] = lookup(t.bend.nodes[a], t.bend.nodes[b]);
                }
            }
            t.offsets_valid = true;
        }
        Ok(())
    }

    fn piece_length(&self) -> f64 {
        match self.first_release {
            Some(t0) => {
                let mut vs = self.input.v.clone();
                vs.sort_by(|a, b| a.total_cmp(b));
                vs[vs.len() / 2] * (self.time - t0)
            }
            None => 0.0,
        }
    }

    /// Runs until the piece reaches the cut length, then (optionally)
    /// relaxes it. `progress` is called after every row; `cancel` is polled
    /// every Newton iteration.
    pub fn run(
        mut self,
        progress: &mut dyn FnMut(&Snapshot),
        cancel: &(dyn Fn() -> bool + Sync),
    ) -> Result<ShellResult, ShellError> {
        let t0 = std::time::Instant::now();
        let mut next_spawn = self.row_birth.last().copied().unwrap_or(0.0) + self.dt_row;
        while self.piece_length() < self.p.cut_length {
            if cancel() {
                return Err(ShellError::Cancelled);
            }
            self.time += self.dt;
            self.stats.steps += 1;
            let mut spawned = false;
            while self.time + 1e-12 >= next_spawn {
                self.spawn_row(next_spawn);
                next_spawn += self.dt_row;
                spawned = true;
            }
            self.drive_and_release();
            self.update_molten_rest();
            self.ensure_pattern()?;
            self.newton_step(cancel)?;
            self.creep();
            if spawned {
                let kf = self.keyframe();
                self.keyframes.push(kf.clone());
                self.stats.seconds = t0.elapsed().as_secs_f64();
                self.stats.nodes = self.total_nodes();
                self.stats.triangles = self.tris.len();
                progress(&Snapshot {
                    time: self.time,
                    piece_length: self.piece_length(),
                    keyframe: kf,
                    stats: self.stats.clone(),
                });
            }
        }
        // Settle the hanging shape with the die held still.
        for _ in 0..20 {
            self.x_old.clone_from(&self.x);
            let before = self.x.clone();
            self.newton_step(cancel)?;
            let moved = self.x.iter().zip(&before).map(|(a, b)| (*a - *b).length()).fold(0.0, f64::max);
            if moved < 1e-4 * self.cell {
                break;
            }
        }
        let outside: Vec<bool> = self.state.iter().map(|s| *s == NodeState::Free).collect();
        let hanging: Vec<[f32; 3]> = self.x.iter().map(|p| p.as_vec3().to_array()).collect();
        let relaxed = if self.p.relax_after_cut { Some(self.relax(cancel)?) } else { None };
        self.stats.seconds = t0.elapsed().as_secs_f64();
        self.stats.nodes = self.total_nodes();
        self.stats.triangles = self.tris.len();
        self.stats.max_rest_angle = self.hinges.iter().filter(|h| h.frozen).map(|h| h.rest_angle.abs()).fold(0.0, f64::max);
        let mut lo = f64::INFINITY;
        let mut hi: f64 = 0.0;
        for t in self.tris.iter().filter(|t| t.frozen) {
            for l2 in t.rest.rest_len2 {
                lo = lo.min(l2.sqrt());
                hi = hi.max(l2.sqrt());
            }
        }
        self.stats.rest_edge_range = (lo, hi);
        let kf = self.keyframe();
        self.keyframes.push(kf);
        let node_station: Vec<u32> = (0..self.total_nodes()).map(|i| (i % self.n_s) as u32).collect();
        let thickness_at_node = node_station.iter().map(|&i| self.input.thickness[i as usize] as f32).collect();
        let speed_at_node = node_station.iter().map(|&i| self.input.v[i as usize] as f32).collect();
        Ok(ShellResult {
            stations: self.n_s,
            closed: self.input.closed,
            triangles: self.tris.iter().map(|t| t.rest.v).collect(),
            node_station,
            hanging,
            relaxed,
            outside,
            keyframes: std::mem::take(&mut self.keyframes),
            stats: self.stats.clone(),
            input: self.input.clone(),
            thickness_at_node,
            speed_at_node,
        })
    }

    fn keyframe(&self) -> Keyframe {
        Keyframe {
            time: self.time,
            positions: Arc::new(self.x.iter().map(|p| p.as_vec3().to_array()).collect()),
            triangles: self.tris.len(),
        }
    }

    fn drive_and_release(&mut self) {
        for idx in 0..self.total_nodes() {
            if self.state[idx] != NodeState::Driven {
                continue;
            }
            let pos = self.driven_position(idx, self.time);
            if pos.z <= 0.0 {
                self.state[idx] = NodeState::Free;
                if self.first_release.is_none() {
                    self.first_release = Some(self.time);
                }
                // Keep the node on the lip and kick it a little along the
                // sheet normal so buckling can start.
                let i = idx % self.n_s;
                let n = self.input.normal[i];
                let amp = self.p.perturbation * self.input.thickness[i] * self.rng.symmetric();
                let p = DVec3::new(pos.x, pos.y, 0.0) + amp * DVec3::new(n[0], n[1], 0.0);
                self.x[idx] = p;
                // It was moving out of the die at its exit speed.
                self.x_old[idx] = p + DVec3::new(0.0, 0.0, self.input.v[i] * self.dt);
            } else {
                self.x[idx] = pos;
                self.x_old[idx] = pos;
            }
        }
    }

    /// Elements wholly inside the die take the die's shape as their rest
    /// shape. The first time a node leaves, the element freezes and its
    /// weight goes to its nodes.
    fn update_molten_rest(&mut self) {
        let (mu, lambda, nu) = (self.mu, self.lambda, self.nu);
        for k in 0..self.tris.len() {
            if self.tris[k].frozen {
                continue;
            }
            let v = self.tris[k].rest.v;
            if v.iter().all(|&i| self.state[i as usize] == NodeState::Driven) {
                let rest = rest_len2(&v.map(|i| self.x[i as usize]));
                let d = self.flexural_rigidity(self.tris[k].rest.thickness);
                let t = &mut self.tris[k];
                t.rest.update_rest(rest, mu, lambda);
                t.bend.b = FaceBend::b_matrix(rest, d, nu);
            } else {
                let t = &mut self.tris[k];
                t.frozen = true;
                let share = self.weight_density * t.rest.thickness * t.rest.area / 3.0;
                for &i in &t.rest.v {
                    self.mass[i as usize] += share;
                    self.node_area[i as usize] += t.rest.area / 3.0;
                }
            }
        }
        for k in 0..self.hinges.len() {
            if self.hinges[k].frozen {
                continue;
            }
            let v = self.hinges[k].v;
            if v.iter().all(|&i| self.state[i as usize] == NodeState::Driven) {
                self.hinges[k].rest_angle = self.hinge_rest_angle(&v);
            } else {
                self.hinges[k].frozen = true;
            }
        }
    }

    fn creep(&mut self) {
        let Some(tau) = self.p.creep_time else { return };
        if !(tau > 0.0) {
            return;
        }
        let f = (self.dt / tau).min(1.0);
        let (mu, lambda, nu) = (self.mu, self.lambda, self.nu);
        for k in 0..self.tris.len() {
            let t = &self.tris[k];
            if !t.frozen || t.rest.v.iter().any(|&i| self.state[i as usize] != NodeState::Free) {
                continue;
            }
            let cur = rest_len2(&t.rest.v.map(|i| self.x[i as usize]));
            let new: [f64; 3] = std::array::from_fn(|q| t.rest.rest_len2[q] + f * (cur[q] - t.rest.rest_len2[q]));
            let d = self.flexural_rigidity(t.rest.thickness);
            let t = &mut self.tris[k];
            t.rest.update_rest(new, mu, lambda);
            t.bend.b = FaceBend::b_matrix(new, d, nu);
        }
        for h in &mut self.hinges {
            if !h.frozen || h.v.iter().any(|&i| self.state[i as usize] != NodeState::Free) {
                continue;
            }
            let cur = dihedral(&h.v.map(|i| self.x[i as usize]));
            h.rest_angle += f * wrap(cur - h.rest_angle);
        }
    }

    fn live(&self, v: &[u32]) -> bool {
        v.iter().any(|&i| self.state[i as usize] == NodeState::Free)
            && v.iter().all(|&i| self.state[i as usize] != NodeState::Removed)
    }

    fn face_rest_angles(&self, t: &TriElem) -> [f64; 3] {
        std::array::from_fn(|k| t.bend.hinge[k].map(|(_, h)| self.hinges[h as usize].rest_angle).unwrap_or(0.0))
    }

    fn gather(&self, nodes: &[u32], x: &[DVec3]) -> Vec<DVec3> {
        nodes.iter().map(|&i| x[i as usize]).collect()
    }

    fn energy(&self, x: &[DVec3]) -> f64 {
        let elem: f64 = self
            .tris
            .par_iter()
            .map(|t| {
                let mut e = 0.0;
                if self.live(&t.rest.v) {
                    e += t.rest.energy(&t.rest.v.map(|i| x[i as usize]));
                }
                if self.live(&t.bend.nodes) {
                    e += t.bend.energy(&self.gather(&t.bend.nodes, x), self.face_rest_angles(t));
                }
                e
            })
            .sum();
        let mut e = elem;
        for i in 0..x.len() {
            if self.state[i] != NodeState::Free {
                continue;
            }
            if self.gravity_on {
                e -= self.mass[i] * self.gravity.dot(x[i]);
            }
            e += 0.5 * self.prox(i) * (x[i] - self.x_target[i]).length_squared();
            let (pz, ps) = self.penetrations(x[i]);
            e += self.contact_k / 3.0 * (pz.powi(3) + ps.powi(3));
        }
        e
    }

    fn prox(&self, i: usize) -> f64 {
        // While relaxing, damping only needs to pin the rigid-body motions.
        let scale = if self.cut { 1e-4 } else { 1.0 };
        self.prox_coef * self.node_area[i] * scale
    }

    /// Penetration into the die face and into the support surface (>= 0).
    fn penetrations(&self, p: DVec3) -> (f64, f64) {
        let pz = p.z.max(0.0);
        let ps = if self.p.orientation == Orientation::Horizontal && !self.cut {
            (self.support_y - p.y).max(0.0)
        } else {
            0.0
        };
        (pz, ps)
    }

    fn newton_step(&mut self, cancel: &(dyn Fn() -> bool + Sync)) -> Result<bool, ShellError> {
        self.newton_step_tol(cancel, 1e-2)
    }

    /// One time step. `tol` is the residual force allowed per degree of
    /// freedom, as a fraction of the heaviest node's weight. During growth a
    /// loose tolerance is enough because each step starts from the last one
    /// and errors do not accumulate; final shapes are solved tightly.
    fn newton_step_tol(&mut self, cancel: &(dyn Fn() -> bool + Sync), tol: f64) -> Result<bool, ShellError> {
        let n = self.total_nodes();
        if !self.state.contains(&NodeState::Free) {
            return Ok(true);
        }
        let n_cap = self.pattern.as_ref().unwrap().rows_capacity * self.n_s;
        let ndof = 3 * n_cap;
        // Predictor: constant velocity while growing; at rest while relaxing.
        for i in 0..n {
            if self.state[i] != NodeState::Free {
                continue;
            }
            let vel = if self.cut { DVec3::ZERO } else { self.x[i] - self.x_old[i] };
            self.x_old[i] = self.x[i];
            self.x_target[i] = self.x[i] + vel;
            self.x[i] = self.x_target[i];
        }
        let mut vals = vec![0.0; self.pattern.as_ref().unwrap().spd.num_entries()];
        let mut grad = vec![0.0; ndof];
        // Residual force allowed per degree of freedom: `tol` of the heaviest
        // node's weight, but no less than stretching one cell by 1e-10
        // (round-off).
        let h_mean = self.input.thickness.iter().sum::<f64>() / self.n_s as f64;
        let f_ref = (self.young * h_mean * self.cell * 1e-10).max(tol * self.mass.iter().cloned().fold(0.0, f64::max));
        let mut converged = false;
        let mut first_gmax: Option<f64> = None;
        let rel_tol = if tol >= 1e-2 { 1e-3 } else { 0.0 };
        let mut e = self.energy(&self.x);
        let mut trial = self.x.clone();
        for _ in 0..self.p.max_newton {
            if cancel() {
                return Err(ShellError::Cancelled);
            }
            let ta = std::time::Instant::now();
            self.assemble(&mut grad, &mut vals);
            self.stats.time_assemble += ta.elapsed().as_secs_f64();
            let gmax = grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
            if first_gmax.is_none() {
                first_gmax = Some(gmax);
            }
            // Converged in absolute terms, or (while growing) relative to
            // where this step started: the slow modes left over are picked
            // up by the following steps.
            let rel_ok = rel_tol > 0.0 && gmax < rel_tol * first_gmax.unwrap();
            if gmax < f_ref || rel_ok {
                converged = true;
                break;
            }
            let mut d: Vec<f64> = grad.iter().map(|g| -g).collect();
            let threads = if ndof > PAR_DOFS { self.p.threads } else { 1 };
            let ts = std::time::Instant::now();
            let shift = self.pattern.as_ref().unwrap().spd.solve(&vals, &mut d, par(threads))?;
            if shift > 0.0 {
                self.stats.shifted_solves += 1;
            }
            self.stats.time_solve += ts.elapsed().as_secs_f64();
            self.stats.newton_iterations += 1;
            let slope: f64 = grad.iter().zip(&d).map(|(g, d)| g * d).sum();
            if !(slope < 0.0) {
                break;
            }
            let mut t = 1.0;
            let mut accepted = false;
            for _ in 0..30 {
                for i in 0..n {
                    if self.state[i] == NodeState::Free {
                        trial[i] = self.x[i] + t * DVec3::new(d[3 * i], d[3 * i + 1], d[3 * i + 2]);
                    }
                }
                let te = std::time::Instant::now();
                let et = self.energy(&trial);
                self.stats.time_energy += te.elapsed().as_secs_f64();
                if et.is_finite() && et <= e + 1e-4 * t * slope {
                    self.x.copy_from_slice(&trial);
                    e = et;
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }
            if !accepted {
                break;
            }
        }
        if !converged {
            self.stats.unconverged_steps += 1;
        }
        Ok(converged)
    }

    fn assemble(&self, grad: &mut [f64], vals: &mut [f64]) {
        let pat = self.pattern.as_ref().unwrap();
        grad.iter_mut().for_each(|g| *g = 0.0);
        vals.iter_mut().for_each(|v| *v = 0.0);
        // Element gradients and Hessians in parallel, scattered in order.
        struct Part {
            k: usize,
            stretch: Option<([f64; 9], [f64; 81])>,
            bend: Option<(Vec<f64>, Vec<f64>)>,
        }
        let parts: Vec<Part> = self
            .tris
            .par_iter()
            .enumerate()
            .filter_map(|(k, t)| {
                let stretch = self.live(&t.rest.v).then(|| {
                    let mut g = [0.0; 9];
                    let mut h = [0.0; 81];
                    t.rest.grad_hess(&t.rest.v.map(|i| self.x[i as usize]), &mut g, &mut h);
                    (g, h)
                });
                let bend = self.live(&t.bend.nodes).then(|| {
                    let n = 3 * t.bend.nodes.len();
                    let mut g = vec![0.0; n];
                    let mut h = vec![0.0; n * n];
                    t.bend.grad_hess(&self.gather(&t.bend.nodes, &self.x), self.face_rest_angles(t), &mut g, &mut h);
                    (g, h)
                });
                (stretch.is_some() || bend.is_some()).then_some(Part { k, stretch, bend })
            })
            .collect();
        let scatter = |v: &[u32], off: &[u32], g: &[f64], h: &[f64], grad: &mut [f64], vals: &mut [f64]| {
            let nv = v.len();
            let w = 3 * nv;
            for a in 0..nv {
                let ia = v[a] as usize;
                if self.state[ia] != NodeState::Free {
                    continue;
                }
                for c in 0..3 {
                    grad[3 * ia + c] += g[3 * a + c];
                }
                for b in 0..nv {
                    let o = off[nv * a + b];
                    let ib = v[b] as usize;
                    if o == NO_OFF || self.state[ib] != NodeState::Free {
                        continue;
                    }
                    let o = o as usize;
                    let mut q = 0;
                    for i in 0..3 {
                        for j in 0..3 {
                            if ia == ib && j > i {
                                continue;
                            }
                            vals[o + q] += h[(3 * a + i) * w + 3 * b + j];
                            q += 1;
                        }
                    }
                }
            }
        };
        for part in &parts {
            let t = &self.tris[part.k];
            if let Some((g, h)) = &part.stretch {
                scatter(&t.rest.v, &t.off, g, h, grad, vals);
            }
            if let Some((g, h)) = &part.bend {
                scatter(&t.bend.nodes, &t.bend_off, g, h, grad, vals);
            }
        }
        // Node terms and constraints.
        let n_cap = pat.rows_capacity * self.n_s;
        let diag = [0, 2, 5]; // (0,0), (1,1), (2,2) in a lower diagonal block
        for i in 0..n_cap {
            let off = pat.offset[i] as usize;
            let free = i < self.state.len() && self.state[i] == NodeState::Free;
            if !free {
                for c in 0..3 {
                    vals[off + diag[c]] = 1.0;
                    grad[3 * i + c] = 0.0;
                }
                continue;
            }
            let c = self.prox(i);
            let dx = self.x[i] - self.x_target[i];
            for q in 0..3 {
                grad[3 * i + q] += c * dx[q];
                vals[off + diag[q]] += c;
            }
            if self.gravity_on {
                for q in 0..3 {
                    grad[3 * i + q] -= self.mass[i] * self.gravity[q];
                }
            }
            let (pz, ps) = self.penetrations(self.x[i]);
            if pz > 0.0 {
                grad[3 * i + 2] += self.contact_k * pz * pz;
                vals[off + diag[2]] += 2.0 * self.contact_k * pz;
            }
            if ps > 0.0 {
                grad[3 * i + 1] -= self.contact_k * ps * ps;
                vals[off + diag[1]] += 2.0 * self.contact_k * ps;
            }
        }
    }

    /// Cuts the piece at the die face and relaxes it with gravity off.
    fn relax(&mut self, cancel: &(dyn Fn() -> bool + Sync)) -> Result<Vec<[f32; 3]>, ShellError> {
        self.cut = true;
        self.gravity_on = false;
        for s in &mut self.state {
            if *s != NodeState::Free {
                *s = NodeState::Removed;
            }
        }
        // Each pseudo-step converges loosely; stop once the shape stops
        // moving (a ten-thousandth of a cell per step).
        for _ in 0..60 {
            let before = self.x.clone();
            self.newton_step(cancel)?;
            let moved = self.x.iter().zip(&before).map(|(a, b)| (*a - *b).length()).fold(0.0, f64::max);
            if moved < 1e-4 * self.cell {
                break;
            }
        }
        Ok(self.x.iter().map(|p| p.as_vec3().to_array()).collect())
    }
}
