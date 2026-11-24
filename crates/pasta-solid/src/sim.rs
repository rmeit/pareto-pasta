//! Growing solid: the extrudate as quadratic tetrahedra, added one row of
//! prisms at a time at the die face.
//!
//! Each section point k leaves the die at its own exit speed v_k. A row is
//! the material that crosses the die face at the same moment, so a row is
//! flat as it exits and fast points carry more length between rows. Each
//! layer of elements takes as its rest shape the die's kinematics at the
//! moment it is half out: a prism whose columns are v_k * dt_row long. That
//! mismatch in rest length between fast and slow columns is what makes the
//! piece curl, twist or ruffle.

use crate::element::{self, EDGES, FACES, Material, Rest, mid_node};
use crate::section::Section;
use glam::DVec3;
use pasta_linalg::{Rng, SolveError, SpdPattern, par};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Orientation {
    /// Extruded straight down: gravity along the extrusion direction.
    #[default]
    Down,
    /// Extruded horizontally over a support surface: gravity along -y.
    Horizontal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolidParams {
    /// Elastic modulus of the fresh dough (kPa).
    pub young_kpa: f64,
    pub poisson: f64,
    /// Dough density (g/cm³).
    pub density: f64,
    pub orientation: Orientation,
    /// Length of the piece when it is cut (mm), measured along the
    /// extrusion at the median exit speed.
    pub cut_length: f64,
    /// Element layers across the narrowest part of the opening.
    pub cells_across: usize,
    /// Element length along a slot as a multiple of its narrowest gap.
    pub station_aspect: f64,
    /// Row spacing along the extrusion as a multiple of the mean element
    /// edge in the section.
    pub row_aspect: f64,
    /// Time steps per row.
    pub substeps: usize,
    /// Distance from the opening down to the support surface (mm),
    /// horizontal extrusion only.
    pub support_drop: f64,
    /// Size of the random offset given to each row as it leaves the die,
    /// as a fraction of the element size, so buckling can start. The whole
    /// row moves sideways together and turns slightly about the extrusion
    /// axis; independent kicks per node would only squeeze the thickness.
    pub perturbation: f64,
    pub seed: u64,
    /// Newton iterations per time step at most.
    pub max_newton: usize,
    /// Extra damping toward the constant-velocity prediction, per unit
    /// volume, in units of Young's modulus per mm². Inertia alone is
    /// usually enough.
    pub damping: f64,
    /// After cutting, relax the piece with gravity off.
    pub relax_after_cut: bool,
    /// Threads for the factorisation (independent branches of the
    /// elimination tree run at the same time; 8 is about the useful limit).
    pub threads: usize,
    /// Clamp each material tangent to be positive semi-definite. Off by
    /// default: the exact Hessian (shifted by the solver when indefinite)
    /// converges about three times faster on buckling sheets.
    pub projected_newton: bool,
    /// For slots, give every point of the section the die-flow speed
    /// averaged across the gap (keeping its tilt from one wall to the
    /// other, which bends the sheet), smoothed along the slot over one gap.
    /// Just past the exit the slow skin at the walls is pulled along by
    /// the core, so the point speeds would overstate the difference. Off
    /// uses the point speeds, which fail when the dough sticks to the walls.
    /// Polygon openings always use the point speeds.
    pub gap_average: bool,
}

impl Default for SolidParams {
    fn default() -> Self {
        Self {
            young_kpa: 50.0,
            poisson: 0.45,
            density: 1.3,
            orientation: Orientation::Down,
            cut_length: 40.0,
            cells_across: 1,
            station_aspect: 0.75,
            row_aspect: 1.0,
            substeps: 2,
            support_drop: 5.0,
            perturbation: 0.02,
            seed: 1,
            max_newton: 10,
            damping: 0.0,
            relax_after_cut: true,
            threads: 1,
            projected_newton: false,
            gap_average: true,
        }
    }
}

#[derive(Debug)]
pub enum SolidError {
    Cancelled,
    Solve(SolveError),
    BadInput(String),
}

impl std::fmt::Display for SolidError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolidError::Cancelled => write!(f, "cancelled"),
            SolidError::Solve(e) => write!(f, "{e}"),
            SolidError::BadInput(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for SolidError {}

impl From<SolveError> for SolidError {
    fn from(e: SolveError) -> Self {
        SolidError::Solve(e)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SolidStats {
    pub steps: usize,
    pub rows: usize,
    pub newton_iterations: usize,
    pub pattern_builds: usize,
    pub unconverged_steps: usize,
    pub seconds: f64,
    pub nodes: usize,
    pub tets: usize,
    /// Unknowns in the last linear system.
    pub dofs: usize,
    pub time_assemble: f64,
    pub time_solve: f64,
    pub time_energy: f64,
    /// Newton systems that were indefinite and were solved with projected
    /// material tangents instead.
    pub shifted_solves: usize,
    /// Numeric factorisations, including failed attempts.
    #[serde(default)]
    pub factorisations: usize,
}

/// Positions at one moment, for the timeline.
#[derive(Clone, Debug)]
pub struct Keyframe {
    pub time: f64,
    /// Positions of the first `positions.len()` nodes.
    pub positions: Arc<Vec<[f32; 3]>>,
    /// Number of surface triangles that existed then (a prefix of
    /// `SolidResult::surface`).
    pub triangles: usize,
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub time: f64,
    pub piece_length: f64,
    pub keyframe: Keyframe,
    /// The growing surface so far (the keyframe uses all of it).
    pub surface: Arc<Vec<[u32; 3]>>,
    /// Exit speed (mm/s) and exit time (s) of every node so far.
    pub speed_at_node: Arc<Vec<f32>>,
    pub exit_time: Arc<Vec<f32>>,
    pub stats: SolidStats,
}

#[derive(Clone, Debug)]
pub struct SolidResult {
    /// Outer surface while growing: the sides and the first end, in
    /// creation order (the open end sits in the die). Each quadratic face
    /// is split into four triangles through its edge midpoints.
    pub surface: Vec<[u32; 3]>,
    /// Closed surface of the cut piece.
    pub piece_surface: Vec<[u32; 3]>,
    /// Shape when cut, hanging from the die.
    pub hanging: Vec<[f32; 3]>,
    /// Shape after relaxing with gravity off, if requested.
    pub relaxed: Option<Vec<[f32; 3]>>,
    /// Nodes that were outside the die when the piece was cut.
    pub outside: Vec<bool>,
    pub keyframes: Vec<Keyframe>,
    pub stats: SolidStats,
    pub section: Section,
    /// Exit speed of the material at each node (mm/s).
    pub speed_at_node: Vec<f32>,
    /// When each node left the die (s).
    pub exit_time: Vec<f32>,
    /// Row spacing at the median speed (mm).
    pub row_spacing: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeState {
    /// Inside the die, moved by the die.
    Driven,
    Free,
    /// Cut off with the die remnant.
    Removed,
}

const NO_OFF: u32 = u32::MAX;


/// Pseudo-steps allowed for settling the hanging piece and for relaxing it.
const SETTLE_STEPS: usize = 20;


struct Tet {
    v: [u32; 10],
    rest: Rest,
    /// Value offset of each node-pair block (a, b) with global a >= b.
    off: [u32; 100],
    offsets_valid: bool,
}

struct Pattern {
    rows_capacity: usize,
    n_nodes: usize,
    spd: SpdPattern,
    pairs: Vec<u64>,
    row_start: Vec<usize>,
    offset: Vec<u32>,
    /// Offset of each node's diagonal block.
    diag: Vec<u32>,
}

impl Pattern {
    fn lookup(&self, a: u32, b: u32) -> u32 {
        let key = (a as u64) << 32 | b as u64;
        let s = &self.pairs[self.row_start[a as usize]..self.row_start[a as usize + 1]];
        let k = s.binary_search(&key).expect("pair in pattern");
        self.offset[self.row_start[a as usize] + k]
    }
}

pub struct SolidSim {
    p: SolidParams,
    sec: Section,
    edge_map: HashMap<(u32, u32), u32>,
    /// Points per level; nodes are numbered level * np + point.
    np: usize,
    nc: usize,
    x: Vec<DVec3>,
    x_old: Vec<DVec3>,
    x_target: Vec<DVec3>,
    state: Vec<NodeState>,
    /// Lumped mass (t) and volume (mm³) of each node.
    mass: Vec<f64>,
    vol: Vec<f64>,
    /// Consistent weight of each node (N); corner nodes of quadratic
    /// elements carry a small negative share.
    weight: Vec<f64>,
    speed: Vec<f64>,
    exit_time: Vec<f64>,
    tets: Vec<Tet>,
    rows: usize,
    surface: Vec<[u32; 3]>,
    time: f64,
    dt_row: f64,
    dt: f64,
    v_median: f64,
    first_release: Option<f64>,
    mat: Material,
    young: f64,
    density: f64,
    weight_density: f64,
    gravity: DVec3,
    support_y: f64,
    cell: f64,
    contact_k: f64,
    pattern: Option<Pattern>,
    rng: Rng,
    stats: SolidStats,
    keyframes: Vec<Keyframe>,
    cut: bool,
    gravity_on: bool,
    /// Project each material tangent to be positive semi-definite.
    project: bool,
    /// Diagonal shift the last Newton system needed (0 if none).
    last_shift: f64,
    /// Release offsets per level: sideways shift (mm) and turn (rad).
    level_kick: Vec<[f64; 3]>,
    section_centre: [f64; 2],
    section_radius: f64,
}

impl SolidSim {
    pub fn new(section: &Section, params: &SolidParams) -> Result<Self, SolidError> {
        if section.tris.is_empty() {
            return Err(SolidError::BadInput("the section has no triangles".into()));
        }
        if section.speed.iter().any(|v| !(*v > 0.0)) {
            return Err(SolidError::BadInput("every section point needs a positive exit speed".into()));
        }
        let p = params.clone();
        let mut vs = section.speed.clone();
        vs.sort_by(|a, b| a.total_cmp(b));
        let v_median = vs[vs.len() / 2];
        let cell = section.mean_edge();
        let dt_row = p.row_aspect.max(0.05) * cell / v_median;
        let young = p.young_kpa * 1e-3;
        let nu = p.poisson.clamp(0.0, 0.49);
        let density = p.density * 1e-9; // t/mm³
        let weight_density = density * 9810.0; // N/mm³
        let gravity = match p.orientation {
            Orientation::Down => DVec3::new(0.0, 0.0, -1.0),
            Orientation::Horizontal => DVec3::new(0.0, -1.0, 0.0),
        };
        let y_min = section.points.iter().map(|q| q[1]).fold(f64::INFINITY, f64::min);
        let w_ref = weight_density * cell.powi(3);
        let mut sim = Self {
            edge_map: section.edge_map(),
            np: section.points.len(),
            nc: section.n_corners,
            sec: section.clone(),
            x: Vec::new(),
            x_old: Vec::new(),
            x_target: Vec::new(),
            state: Vec::new(),
            mass: Vec::new(),
            vol: Vec::new(),
            weight: Vec::new(),
            speed: Vec::new(),
            exit_time: Vec::new(),
            tets: Vec::new(),
            rows: 0,
            surface: Vec::new(),
            time: 0.0,
            dt_row,
            dt: dt_row / p.substeps.max(1) as f64,
            v_median,
            first_release: None,
            mat: Material::new(young, nu),
            young,
            density,
            weight_density,
            gravity,
            support_y: y_min - p.support_drop,
            cell,
            // The die face pushes back with a node's weight at 2% of a cell.
            contact_k: w_ref / (0.02 * cell).powi(2),
            pattern: None,
            rng: Rng::new(p.seed),
            stats: SolidStats::default(),
            keyframes: Vec::new(),
            cut: false,
            gravity_on: true,
            project: params.projected_newton,
            last_shift: 0.0,
            level_kick: Vec::new(),
            section_centre: {
                let n = section.n_corners as f64;
                section.points[..section.n_corners].iter().fold([0.0, 0.0], |a, q| [a[0] + q[0] / n, a[1] + q[1] / n])
            },
            section_radius: 0.0,
            p,
        };
        let c = sim.section_centre;
        sim.section_radius = section.points.iter().map(|q| ((q[0] - c[0]).powi(2) + (q[1] - c[1]).powi(2)).sqrt()).fold(0.0, f64::max);
        for r in 0..3 {
            sim.add_row(r)?;
        }
        Ok(sim)
    }

    fn tau(&self, r: usize) -> f64 {
        r as f64 * self.dt_row
    }

    /// Ends of a half-level point: the same corner on both rows, or the
    /// lower-numbered corner of an edge on the lower row and the other on
    /// the upper row.
    fn half_ends(&self, k: usize) -> (usize, usize) {
        if k < self.nc {
            (k, k)
        } else {
            let e = self.sec.edges[k - self.nc];
            (e[0] as usize, e[1] as usize)
        }
    }

    fn point_kin(&self, k: usize, r: usize, t: f64) -> DVec3 {
        let q = self.sec.points[k];
        DVec3::new(q[0], q[1], -self.sec.speed[k] * (t - self.tau(r)))
    }

    /// Where the die puts a node at time t (beyond the face this is the
    /// straight extrusion the material would follow if it were rigid).
    fn kinematic(&self, node: usize, t: f64) -> DVec3 {
        let level = node / self.np;
        let k = node % self.np;
        if level.is_multiple_of(2) {
            self.point_kin(k, level / 2, t)
        } else {
            let r = level / 2;
            let (i, j) = self.half_ends(k);
            0.5 * (self.point_kin(i, r, t) + self.point_kin(j, r + 1, t))
        }
    }

    fn node_speed(&self, node: usize) -> f64 {
        let level = node / self.np;
        let k = node % self.np;
        if level.is_multiple_of(2) {
            self.sec.speed[k]
        } else {
            let (i, j) = self.half_ends(k);
            0.5 * (self.sec.speed[i] + self.sec.speed[j])
        }
    }

    fn edge_index(&self, a: usize, b: usize) -> usize {
        let key = (a.min(b) as u32, a.max(b) as u32);
        self.edge_map[&key] as usize
    }

    /// Tets of layer r (between rows r and r+1). Each prism over a section
    /// triangle is split into three tets with every side diagonal running
    /// from the lower-numbered corner on row r to the higher-numbered corner
    /// on row r+1, so neighbouring prisms match.
    fn layer_tets(&self, r: usize) -> Vec<[u32; 10]> {
        let (np, nc) = (self.np, self.nc);
        let (lb, lh, lt) = (2 * r, 2 * r + 1, 2 * r + 2);
        let node = |level: usize, k: usize| (level * np + k) as u32;
        let mut out = Vec::with_capacity(3 * self.sec.tris.len());
        for t in &self.sec.tris {
            let mut c = [t[0] as usize, t[1] as usize, t[2] as usize];
            c.sort_unstable();
            let [a, b, cc] = c;
            let prisms: [[(bool, usize); 4]; 3] = [
                [(false, a), (false, b), (false, cc), (true, cc)],
                [(false, a), (false, b), (true, b), (true, cc)],
                [(false, a), (true, a), (true, b), (true, cc)],
            ];
            for tv in prisms {
                let mut n = [0u32; 10];
                for q in 0..4 {
                    let (top, k) = tv[q];
                    n[q] = node(if top { lt } else { lb }, k);
                }
                for (e, &(p, q)) in EDGES.iter().enumerate() {
                    let ((tp, kp), (tq, kq)) = (tv[p], tv[q]);
                    n[4 + e] = if tp == tq {
                        node(if tp { lt } else { lb }, nc + self.edge_index(kp, kq))
                    } else if kp == kq {
                        node(lh, kp)
                    } else {
                        let (kb, kt) = if tp { (kq, kp) } else { (kp, kq) };
                        debug_assert!(kb < kt);
                        node(lh, nc + self.edge_index(kb, kt))
                    };
                }
                out.push(n);
            }
        }
        out
    }

    fn add_row(&mut self, r: usize) -> Result<(), SolidError> {
        let first = if r == 0 { 0 } else { 2 * r - 1 };
        for level in first..=2 * r {
            for k in 0..self.np {
                let node = level * self.np + k;
                debug_assert_eq!(node, self.x.len());
                let pos = self.kinematic(node, self.time);
                self.x.push(pos);
                self.x_old.push(pos);
                self.x_target.push(pos);
                self.state.push(NodeState::Driven);
                self.mass.push(0.0);
                self.vol.push(0.0);
                self.weight.push(0.0);
                let v = self.node_speed(node);
                self.speed.push(v);
                // Leaves when its kinematic z reaches 0: z = z(0) - v t.
                let z0 = self.kinematic(node, 0.0).z;
                self.exit_time.push(z0 / v);
            }
        }
        self.rows = r + 1;
        self.stats.rows = self.rows;
        if r >= 1 {
            self.add_layer(r - 1)?;
        }
        Ok(())
    }

    fn add_layer(&mut self, r: usize) -> Result<(), SolidError> {
        let tf = 0.5 * (self.tau(r) + self.tau(r + 1));
        let boundary_pair = |sim: &Self, i: usize, j: usize| -> bool {
            sim.edge_map.get(&(i.min(j) as u32, i.max(j) as u32)).is_some_and(|&e| sim.sec.boundary[e as usize])
        };
        for v in self.layer_tets(r) {
            let xr: [[f64; 3]; 10] = v.map(|n| self.kinematic(n as usize, tf).to_array());
            let rest = Rest::new(&xr).ok_or_else(|| SolidError::BadInput("a rest element is degenerate".into()))?;
            let lumped = rest.lumped();
            let load = rest.load();
            for a in 0..10 {
                let n = v[a] as usize;
                self.mass[n] += self.density * lumped[a];
                self.vol[n] += lumped[a];
                self.weight[n] += self.weight_density * load[a];
            }
            // Outer faces: the sides along the outline, and the first end.
            for (f, face) in FACES.iter().enumerate() {
                let ks: [usize; 3] = face.map(|c| v[c] as usize % self.np);
                let levels: [usize; 3] = face.map(|c| v[c] as usize / self.np);
                let mut distinct = ks.to_vec();
                distinct.sort_unstable();
                distinct.dedup();
                let side = distinct.len() == 2 && boundary_pair(self, distinct[0], distinct[1]);
                let first_end = r == 0 && levels.iter().all(|&l| l == 0);
                if side || first_end {
                    let pos = |n: usize| DVec3::from_array(xr[n]);
                    self.surface.extend(face_triangles(&v, *face, f, &pos));
                }
            }
            self.tets.push(Tet { v, rest, off: [NO_OFF; 100], offsets_valid: false });
        }
        self.stats.tets = self.tets.len();
        Ok(())
    }

    fn ensure_pattern(&mut self) -> Result<(), SolidError> {
        let need = self.rows;
        let rebuild = self.pattern.as_ref().is_none_or(|p| p.rows_capacity < need);
        if rebuild {
            // No headroom: spare rows would be factorised (as zeros) in every
            // solve, which costs more than rebuilding the pattern each row.
            let cap = need;
            let n_nodes = (2 * cap - 1) * self.np;
            let mut pairs: Vec<u64> = (0..n_nodes as u64).map(|i| i << 32 | i).collect();
            for r in 0..cap - 1 {
                for v in self.layer_tets(r) {
                    for &a in &v {
                        for &b in &v {
                            if a > b {
                                pairs.push((a as u64) << 32 | b as u64);
                            }
                        }
                    }
                }
            }
            pairs.par_sort_unstable();
            pairs.dedup();
            let mut row_start = vec![0usize; n_nodes + 1];
            for &q in &pairs {
                row_start[(q >> 32) as usize + 1] += 1;
            }
            for i in 0..n_nodes {
                row_start[i + 1] += row_start[i];
            }
            let mut entries = Vec::with_capacity(pairs.len() * 9);
            let mut offset = Vec::with_capacity(pairs.len());
            for &q in &pairs {
                let (a, b) = ((q >> 32) as usize, (q & 0xffff_ffff) as usize);
                offset.push(entries.len() as u32);
                for i in 0..3 {
                    for j in 0..3 {
                        if a == b && j > i {
                            continue;
                        }
                        entries.push((3 * a + i, 3 * b + j));
                    }
                }
            }
            let spd = SpdPattern::new(3 * n_nodes, &entries)?;
            self.stats.pattern_builds += 1;
            // Within a row, pairs are sorted by column, so the diagonal is last.
            let diag = (0..n_nodes).map(|i| offset[row_start[i + 1] - 1]).collect();
            self.pattern = Some(Pattern { rows_capacity: cap, n_nodes, spd, pairs, row_start, offset, diag });
            for t in &mut self.tets {
                t.offsets_valid = false;
            }
        }
        let pat = self.pattern.as_ref().unwrap();
        self.tets.par_iter_mut().filter(|t| !t.offsets_valid).for_each(|t| {
            for a in 0..10 {
                for b in 0..10 {
                    let (ga, gb) = (t.v[a], t.v[b]);
                    t.off[10 * a + b] = if ga >= gb { pat.lookup(ga, gb) } else { NO_OFF };
                }
            }
            t.offsets_valid = true;
        });
        Ok(())
    }

    fn piece_length(&self) -> f64 {
        self.first_release.map_or(0.0, |t0| self.v_median * (self.time - t0))
    }

    /// Runs until the piece reaches the cut length, then (optionally)
    /// relaxes it. `progress` is called after every row; `cancel` is polled
    /// every Newton iteration.
    pub fn run(
        mut self,
        progress: &mut dyn FnMut(&Snapshot),
        cancel: &(dyn Fn() -> bool + Sync),
    ) -> Result<SolidResult, SolidError> {
        let t0 = std::time::Instant::now();
        while self.piece_length() < self.p.cut_length {
            if cancel() {
                return Err(SolidError::Cancelled);
            }
            self.time += self.dt;
            self.stats.steps += 1;
            // Keep two rows of material waiting in the die.
            let want = (self.time / self.dt_row).floor() as usize + 3;
            let mut spawned = false;
            while self.rows < want {
                self.add_row(self.rows)?;
                spawned = true;
            }
            self.drive_and_release();
            self.ensure_pattern()?;
            self.newton_step(cancel, 1e-2)?;
            if spawned {
                let kf = self.keyframe();
                self.keyframes.push(kf.clone());
                self.stats.seconds = t0.elapsed().as_secs_f64();
                self.stats.nodes = self.x.len();
                progress(&Snapshot {
                    time: self.time,
                    piece_length: self.piece_length(),
                    keyframe: kf,
                    surface: Arc::new(self.surface.clone()),
                    speed_at_node: Arc::new(self.speed.iter().map(|&v| v as f32).collect()),
                    exit_time: Arc::new(self.exit_time.iter().map(|&t| t as f32).collect()),
                    stats: self.stats.clone(),
                });
            }
        }
        // Settle the hanging shape with the die held still: pseudo-steps at
        // rest until the shape stops moving (a thousandth of an element).
        for _ in 0..SETTLE_STEPS {
            self.x_old.clone_from(&self.x);
            let before = self.x.clone();
            self.newton_step(cancel, 1e-2)?;
            if max_move(&self.x, &before) < 1e-3 * self.cell {
                break;
            }
        }
        let outside: Vec<bool> = self.state.iter().map(|s| *s == NodeState::Free).collect();
        let hanging: Vec<[f32; 3]> = self.x.iter().map(|q| q.as_vec3().to_array()).collect();
        let piece_surface = self.piece_surface();
        let relaxed = if self.p.relax_after_cut { Some(self.relax(cancel)?) } else { None };
        self.stats.seconds = t0.elapsed().as_secs_f64();
        self.stats.nodes = self.x.len();
        let kf = self.keyframe();
        self.keyframes.push(kf);
        Ok(SolidResult {
            surface: std::mem::take(&mut self.surface),
            piece_surface,
            hanging,
            relaxed,
            outside,
            keyframes: std::mem::take(&mut self.keyframes),
            stats: self.stats.clone(),
            section: self.sec.clone(),
            speed_at_node: self.speed.iter().map(|&v| v as f32).collect(),
            exit_time: self.exit_time.iter().map(|&t| t as f32).collect(),
            row_spacing: self.v_median * self.dt_row,
        })
    }

    fn keyframe(&self) -> Keyframe {
        Keyframe {
            time: self.time,
            positions: Arc::new(self.x.iter().map(|q| q.as_vec3().to_array()).collect()),
            triangles: self.surface.len(),
        }
    }

    /// Closed surface of the elements that are wholly outside the die.
    fn piece_surface(&self) -> Vec<[u32; 3]> {
        let outside = |t: &Tet| t.v.iter().all(|&n| self.state[n as usize] == NodeState::Free);
        let mut count: HashMap<[u32; 3], u32> = HashMap::new();
        for t in self.tets.iter().filter(|t| outside(t)) {
            for face in FACES {
                let mut key = face.map(|c| t.v[c]);
                key.sort_unstable();
                *count.entry(key).or_insert(0) += 1;
            }
        }
        let mut out = Vec::new();
        for t in self.tets.iter().filter(|t| outside(t)) {
            for (f, face) in FACES.iter().enumerate() {
                let mut key = face.map(|c| t.v[c]);
                key.sort_unstable();
                if count[&key] == 1 {
                    let pos = |n: usize| self.x[t.v[n] as usize];
                    out.extend(face_triangles(&t.v, *face, f, &|n| pos(n)));
                }
            }
        }
        out
    }

    fn drive_and_release(&mut self) {
        for i in 0..self.x.len() {
            if self.state[i] != NodeState::Driven {
                continue;
            }
            let pos = self.kinematic(i, self.time);
            if pos.z <= 0.0 {
                self.state[i] = NodeState::Free;
                if self.first_release.is_none() {
                    self.first_release = Some(self.time);
                }
                let k = self.kick(i);
                // The time step's predictor moves every free node from where
                // it was one step ago at its last velocity, so start a new
                // node one step back on its path out of the die; it then
                // lands where the die puts it now.
                self.x[i] = self.kinematic(i, self.time - self.dt) + k;
                self.x_old[i] = self.kinematic(i, self.time - 2.0 * self.dt) + k;
            } else {
                self.x[i] = pos;
                self.x_old[i] = pos;
            }
        }
    }

    /// The release offset of node i: its level's shared sideways shift plus
    /// a small turn about the section centroid.
    fn kick(&mut self, i: usize) -> DVec3 {
        let level = i / self.np;
        while self.level_kick.len() <= level {
            let a = self.p.perturbation * self.cell;
            let (dx, dy) = (a * self.rng.symmetric(), a * self.rng.symmetric());
            // A turn that moves the outermost point by about the same amount.
            let turn = a * self.rng.symmetric() / self.section_radius.max(1e-9);
            self.level_kick.push([dx, dy, turn]);
        }
        let [dx, dy, turn] = self.level_kick[level];
        let q = self.x[i];
        let (rx, ry) = (q.x - self.section_centre[0], q.y - self.section_centre[1]);
        DVec3::new(dx - turn * ry, dy + turn * rx, 0.0)
    }

    fn active(&self, t: &Tet) -> bool {
        t.v.iter().any(|&n| self.state[n as usize] == NodeState::Free)
            && t.v.iter().all(|&n| self.state[n as usize] != NodeState::Removed)
    }

    fn gather(&self, v: &[u32; 10], x: &[DVec3]) -> [[f64; 3]; 10] {
        v.map(|n| x[n as usize].to_array())
    }

    /// Spring to the predicted position: inertia (mass / dt²) plus optional
    /// damping.
    fn prox(&self, i: usize) -> f64 {
        self.mass[i] / (self.dt * self.dt) + self.p.damping * self.young * self.vol[i]
    }

    fn penetrations(&self, q: DVec3) -> (f64, f64) {
        if self.cut {
            return (0.0, 0.0);
        }
        let pz = q.z.max(0.0);
        let ps = if self.p.orientation == Orientation::Horizontal { (self.support_y - q.y).max(0.0) } else { 0.0 };
        (pz, ps)
    }

    fn energy(&self, x: &[DVec3]) -> f64 {
        let elastic: f64 = self
            .tets
            .par_iter()
            .filter(|t| self.active(t))
            .map(|t| element::energy(&t.rest, &self.mat, &self.gather(&t.v, x)))
            .sum();
        let mut e = elastic;
        if !e.is_finite() {
            return f64::INFINITY;
        }
        for i in 0..x.len() {
            if self.state[i] != NodeState::Free {
                continue;
            }
            if self.gravity_on {
                e -= self.weight[i] * self.gravity.dot(x[i]);
            }
            e += 0.5 * self.prox(i) * (x[i] - self.x_target[i]).length_squared();
            let (pz, ps) = self.penetrations(x[i]);
            e += self.contact_k / 3.0 * (pz.powi(3) + ps.powi(3));
        }
        e
    }

    /// One implicit time step. `tol` is the residual force allowed per
    /// degree of freedom as a fraction of the heaviest node's weight; while
    /// growing, a step also ends once the residual has fallen a thousandfold.
    fn newton_step(&mut self, cancel: &(dyn Fn() -> bool + Sync), tol: f64) -> Result<bool, SolidError> {
        let n = self.x.len();
        if !self.state.contains(&NodeState::Free) {
            return Ok(true);
        }
        let (n_nodes, n_entries) = {
            let pat = self.pattern.as_ref().unwrap();
            (pat.n_nodes, pat.spd.num_entries())
        };
        let ndof = 3 * n_nodes;
        for i in 0..n {
            if self.state[i] != NodeState::Free {
                continue;
            }
            let vel = if self.cut { DVec3::ZERO } else { self.x[i] - self.x_old[i] };
            self.x_old[i] = self.x[i];
            self.x_target[i] = self.x[i] + vel;
            self.x[i] = self.x_target[i];
        }
        let mut vals = vec![0.0; n_entries];
        let mut grad = vec![0.0; ndof];
        let w_max = self.weight.iter().fold(0.0f64, |m, w| m.max(w.abs()));
        let f_ref = (self.young * self.cell * self.cell * 1e-10).max(tol * w_max);
        let rel_tol = if tol >= 1e-2 { 1e-3 } else { 0.0 };
        let mut first_gmax: Option<f64> = None;
        let mut converged = false;
        let mut e = self.energy(&self.x);
        if !e.is_finite() {
            // The prediction inverted an element; start from the last shape.
            for i in 0..n {
                if self.state[i] == NodeState::Free {
                    self.x[i] = self.x_old[i];
                }
            }
            e = self.energy(&self.x);
        }
        let mut trial = self.x.clone();
        for _ in 0..self.p.max_newton {
            if cancel() {
                return Err(SolidError::Cancelled);
            }
            let ta = std::time::Instant::now();
            self.assemble(&mut grad, &mut vals, self.project);
            self.stats.time_assemble += ta.elapsed().as_secs_f64();
            let gmax = grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
            let g0 = *first_gmax.get_or_insert(gmax);
            if gmax < f_ref || (rel_tol > 0.0 && gmax < rel_tol * g0) {
                converged = true;
                break;
            }
            let ts = std::time::Instant::now();
            let f0 = self.pattern.as_ref().unwrap().spd.factorisations();
            let d = self.direction(&mut grad, &mut vals, self.p.threads)?;
            self.stats.factorisations += self.pattern.as_ref().unwrap().spd.factorisations() - f0;
            self.stats.time_solve += ts.elapsed().as_secs_f64();
            self.stats.newton_iterations += 1;
            self.stats.dofs = 3 * self.state.iter().filter(|s| **s == NodeState::Free).count();
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

    /// Newton direction from a fresh factorisation of the exact Hessian;
    /// where it is indefinite (the piece is buckling), from the Hessian with
    /// the material tangents projected to be positive semi-definite instead:
    /// a sure descent direction for one more factorisation, where searching
    /// for a diagonal shift can cost ten.
    fn direction(&mut self, grad: &mut [f64], vals: &mut [f64], threads: usize) -> Result<Vec<f64>, SolidError> {
        let mut d: Vec<f64> = grad.iter().map(|g| -g).collect();
        let spd = &self.pattern.as_ref().unwrap().spd;
        if !spd.try_solve(vals, &mut d, par(threads))? {
            self.stats.shifted_solves += 1;
            let ta = std::time::Instant::now();
            self.assemble(grad, vals, true);
            self.stats.time_assemble += ta.elapsed().as_secs_f64();
            d = grad.iter().map(|g| -g).collect();
            let spd = &self.pattern.as_ref().unwrap().spd;
            let first = 0.25 * self.last_shift;
            self.last_shift = spd.solve_from(vals, &mut d, par(threads), first)?;
        }
        Ok(d)
    }

    fn assemble(&self, grad: &mut [f64], vals: &mut [f64], project: bool) {
        let pat = self.pattern.as_ref().unwrap();
        grad.fill(0.0);
        vals.fill(0.0);
        let active: Vec<usize> = (0..self.tets.len()).filter(|&k| self.active(&self.tets[k])).collect();
        for chunk in active.chunks(2048) {
            let parts: Vec<(usize, Box<[f64; 30]>, Box<[f64; 900]>)> = chunk
                .par_iter()
                .filter_map(|&k| {
                    let t = &self.tets[k];
                    let mut g = Box::new([0.0; 30]);
                    let mut h = Box::new([0.0; 900]);
                    element::grad_hess_with(&t.rest, &self.mat, &self.gather(&t.v, &self.x), &mut g, &mut h, project)
                        .then_some((k, g, h))
                })
                .collect();
            for (k, g, h) in parts {
                let t = &self.tets[k];
                for a in 0..10 {
                    let ia = t.v[a] as usize;
                    if self.state[ia] != NodeState::Free {
                        continue;
                    }
                    for c in 0..3 {
                        grad[3 * ia + c] += g[3 * a + c];
                    }
                    for b in 0..10 {
                        let o = t.off[10 * a + b];
                        let ib = t.v[b] as usize;
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
                                vals[o + q] += h[(3 * a + i) * 30 + 3 * b + j];
                                q += 1;
                            }
                        }
                    }
                }
            }
        }
        let diag = [0, 2, 5]; // (0,0), (1,1), (2,2) in a lower diagonal block
        for i in 0..pat.n_nodes {
            let off = pat.diag[i] as usize;
            let free = i < self.state.len() && self.state[i] == NodeState::Free;
            if !free {
                // Decoupled rows; a diagonal at the dough's stiffness scale
                // keeps the solver's mean diagonal meaningful.
                for c in 0..3 {
                    vals[off + diag[c]] = self.young * self.cell;
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
                    grad[3 * i + q] -= self.weight[i] * self.gravity[q];
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
    fn relax(&mut self, cancel: &(dyn Fn() -> bool + Sync)) -> Result<Vec<[f32; 3]>, SolidError> {
        self.cut = true;
        self.gravity_on = false;
        for s in &mut self.state {
            if *s != NodeState::Free {
                *s = NodeState::Removed;
            }
        }
        for _ in 0..SETTLE_STEPS {
            let before = self.x.clone();
            self.newton_step(cancel, 1e-2)?;
            if max_move(&self.x, &before) < 1e-3 * self.cell {
                break;
            }
        }
        Ok(self.x.iter().map(|q| q.as_vec3().to_array()).collect())
    }
}

fn max_move(a: &[DVec3], b: &[DVec3]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (*p - *q).length()).fold(0.0, f64::max)
}

/// The four flat triangles of one quadratic face, wound so the normal
/// points away from the tet's opposite corner.
fn face_triangles(v: &[u32; 10], face: [usize; 3], f: usize, pos: &dyn Fn(usize) -> DVec3) -> [[u32; 3]; 4] {
    let opposite = f; // FACES[f] is the face opposite corner f
    let [mut a, b, mut c] = face;
    let n = (pos(b) - pos(a)).cross(pos(c) - pos(a));
    if n.dot(pos(opposite) - pos(a)) > 0.0 {
        std::mem::swap(&mut a, &mut c);
    }
    let (ab, bc, ca) = (mid_node(a, b), mid_node(b, c), mid_node(c, a));
    let g = |l: usize| v[l];
    [
        [g(a), g(ab), g(ca)],
        [g(ab), g(b), g(bc)],
        [g(ca), g(bc), g(c)],
        [g(ab), g(bc), g(ca)],
    ]
}
