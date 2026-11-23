use pasta_die::{Die, MeshOptions, TriMesh2, mesh_die};
use pasta_solid::{Section, SolidMeasurements, SolidParams, SolidResult, SolidSim, SolidStats};
use pasta_flow::{Drive, FlowParams, FlowSolution, FlowSolver, Readouts, Rheology, SlotProfile, profiles, readouts};
use pasta_shell::{Measurements, ShellParams, ShellResult, ShellSim, Snapshot, StripInput, measure};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// What pushes the dough through the die.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DriveSpec {
    /// Extruder output in grams per second, as a home extruder sets it.
    Output { grams_per_second: f64 },
    /// Fixed pressure drop (kPa).
    Pressure { kpa: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MeshQuality {
    /// For live dragging.
    Draft,
    #[default]
    Fine,
}

impl MeshQuality {
    pub fn options(self) -> MeshOptions {
        match self {
            MeshQuality::Draft => MeshOptions::draft(),
            MeshQuality::Fine => MeshOptions::fine(),
        }
    }
}

/// Everything needed to reproduce a run. Saved as RON.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub die: Die,
    pub rheology: Rheology,
    pub drive: DriveSpec,
    /// Thickness multiplier at the die exit; speed is divided by it.
    pub swell: f64,
    /// Prefactor C of the ruffle wavelength estimate C h^0.4 l^0.6.
    pub ruffle_c: f64,
    pub shell: ShellParams,
    /// The solid model of the extruded pasta (quadratic tets).
    #[serde(default)]
    pub solid: SolidParams,
    /// Which opening the pasta grows from.
    pub opening: usize,
    #[serde(default)]
    pub seed_note: Option<String>,
}

impl Project {
    pub fn from_preset(id: &str) -> Option<Self> {
        let p = pasta_die::preset(id)?;
        Some(Self::new(p.die))
    }

    pub fn new(die: Die) -> Self {
        Self {
            die,
            rheology: Rheology::default(),
            drive: DriveSpec::Output { grams_per_second: 0.5 },
            swell: 1.0,
            ruffle_c: 4.0,
            shell: ShellParams::default(),
            solid: SolidParams::default(),
            opening: 0,
            seed_note: None,
        }
    }

    pub fn flow_params(&self) -> FlowParams {
        let drive = match self.drive {
            // g/s divided by g/mm³ gives mm³/s.
            DriveSpec::Output { grams_per_second } => {
                Drive::FlowRate(grams_per_second / (self.shell.density * 1e-3).max(1e-12))
            }
            DriveSpec::Pressure { kpa } => Drive::Pressure(kpa),
        };
        FlowParams { rheology: self.rheology, drive }
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).expect("project serialises")
    }

    pub fn from_ron(s: &str) -> Result<Self, String> {
        ron::from_str(s).map_err(|e| e.to_string())
    }
}

/// Result of the die stage: mesh, speed map, profiles and readouts.
#[derive(Clone, Debug)]
pub struct DieOutput {
    pub mesh: Arc<TriMesh2>,
    pub flow: Arc<FlowSolution>,
    pub profiles: Vec<SlotProfile>,
    pub readouts: Vec<Readouts>,
    pub quality: MeshQuality,
    pub seconds: f64,
}

impl DieOutput {
    /// Profile of the given opening, if it is a slot.
    pub fn profile(&self, opening: usize) -> Option<&SlotProfile> {
        self.profiles.iter().find(|p| p.opening == opening)
    }

    pub fn readout(&self, opening: usize) -> Option<&Readouts> {
        self.readouts.iter().find(|r| r.opening == opening)
    }
}

/// Meshes the die and solves the flow. `warm` (a previous output with the
/// same node count) speeds up the solve.
pub fn solve_die(
    project: &Project,
    quality: MeshQuality,
    warm: Option<&DieOutput>,
    threads: usize,
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<DieOutput, String> {
    let t0 = std::time::Instant::now();
    let mesh = mesh_die(&project.die, &quality.options()).map_err(|e| e.to_string())?;
    let solver = FlowSolver::new(&mesh, project.rheology.no_slip, threads).map_err(|e| e.to_string())?;
    let warm = warm.filter(|w| w.mesh.nodes.len() == mesh.nodes.len()).map(|w| w.flow.as_ref());
    let flow = solver.solve(&project.flow_params(), warm, cancel).map_err(|e| e.to_string())?;
    let profiles = profiles(&mesh, &flow, project.swell);
    let readouts = profiles.iter().map(|p| readouts(p, project.ruffle_c)).collect();
    Ok(DieOutput {
        mesh: Arc::new(mesh),
        flow: Arc::new(flow),
        profiles,
        readouts,
        quality,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

/// The shell's view of a slot profile.
pub fn strip_input(p: &SlotProfile) -> StripInput {
    StripInput {
        closed: p.closed,
        centre: p.centre.clone(),
        normal: p.normal.clone(),
        v: p.v.clone(),
        thickness: p.thickness.clone(),
        bend_curvature: p.bend_curvature.clone(),
    }
}

/// Grows the pasta from the project's chosen opening.
pub fn run_shell(
    project: &Project,
    die: &DieOutput,
    progress: &mut dyn FnMut(&Snapshot),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<ShellResult, String> {
    let profile = die
        .profile(project.opening)
        .ok_or_else(|| format!("opening {} is not a slot, so it has no sheet to grow", project.opening + 1))?;
    let sim = ShellSim::new(&strip_input(profile), &project.shell).map_err(|e| e.to_string())?;
    sim.run(progress, cancel).map_err(|e| e.to_string())
}

/// Linear interpolation of the die-flow exit speed inside one opening. Points
/// on or just outside the opening's edge take the nearest triangle's value.
pub fn speed_sampler<'a>(mesh: &'a TriMesh2, w: &'a [f64], opening: usize) -> impl Fn([f64; 2]) -> f64 + 'a {
    let tris: Vec<usize> = (0..mesh.tris.len()).filter(|&t| mesh.tri_opening[t] as usize == opening).collect();
    move |p: [f64; 2]| {
        let mut best = (f64::NEG_INFINITY, [0.0; 3], 0usize);
        for &t in &tris {
            let [a, b, c] = mesh.tris[t].map(|i| mesh.nodes[i as usize]);
            let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
            let l1 = ((p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])) / det;
            let l2 = ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / det;
            let l = [1.0 - l1 - l2, l1, l2];
            let m = l[0].min(l[1]).min(l[2]);
            if m > best.0 {
                best = (m, l, t);
                if m >= -1e-12 {
                    break;
                }
            }
        }
        let l = best.1.map(|v| v.max(0.0));
        let s = l[0] + l[1] + l[2];
        let tri = mesh.tris[best.2];
        (0..3).map(|q| l[q] / s * w[tri[q] as usize]).sum()
    }
}

/// Exit speed of a slot as the dough leaves it: the gap-mean speed and its
/// linear tilt across the gap (w = v (1 + b t), t from the -normal wall to
/// the +normal wall) at each station, both smoothed along the slot over one
/// gap width so the end walls do not leave zero speed. Points are placed
/// on the nearest piece of centre line; speeds are before swell.
pub fn gap_mean_sampler(p: &SlotProfile) -> impl Fn([f64; 2]) -> f64 + '_ {
    let n = p.s.len();
    // Undo the profile's swell: v · thickness = mean · gap.
    let raw: Vec<f64> = (0..n).map(|i| p.v[i] * p.thickness[i] / p.gap[i]).collect();
    let flux: Vec<f64> = (0..n).map(|i| raw[i] * p.bias[i]).collect();
    // Piecewise-linear value at arc length s (wrapping for closed slots,
    // clamped to the ends for open ones).
    let at = |x: &[f64], s: f64| -> f64 {
        let s = if p.closed { s.rem_euclid(p.length) } else { s.clamp(p.s[0], p.s[n - 1]) };
        let k = p.s.partition_point(|&t| t <= s).clamp(1, n);
        let (s0, s1) = (p.s[k - 1], if k < n { p.s[k] } else { p.length });
        let x1 = if k < n { x[k] } else { x[0] };
        let f = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
        x[k - 1] + f * (x1 - x[k - 1])
    };
    let (mut v, mut b) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        // Mean over [s - g/2, s + g/2], cut at the ends of an open slot.
        let (mut lo, mut hi) = (p.s[i] - 0.5 * p.gap[i], p.s[i] + 0.5 * p.gap[i]);
        if !p.closed {
            lo = lo.max(p.s[0]);
            hi = hi.min(p.s[n - 1]);
        }
        const K: usize = 16;
        let (mut sw, mut swb) = (0.0, 0.0);
        for q in 0..K {
            let s = lo + (hi - lo) * (q as f64 + 0.5) / K as f64;
            sw += at(&raw, s);
            swb += at(&flux, s);
        }
        v[i] = sw / K as f64;
        b[i] = if sw > 0.0 { (swb / sw).clamp(-0.9, 0.9) } else { 0.0 };
    }
    let segs = if p.closed { n } else { n - 1 };
    move |q: [f64; 2]| {
        let mut best = (f64::INFINITY, 0usize, 0.0);
        for i in 0..segs {
            let (a, c) = (p.centre[i], p.centre[(i + 1) % n]);
            let d = [c[0] - a[0], c[1] - a[1]];
            let l2 = d[0] * d[0] + d[1] * d[1];
            let f = if l2 > 0.0 { (((q[0] - a[0]) * d[0] + (q[1] - a[1]) * d[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let e = [q[0] - a[0] - f * d[0], q[1] - a[1] - f * d[1]];
            let r = e[0] * e[0] + e[1] * e[1];
            if r < best.0 {
                best = (r, i, f);
            }
        }
        let (_, i, f) = best;
        let j = (i + 1) % n;
        let lerp = |x: &[f64]| x[i] + f * (x[j] - x[i]);
        let c = [p.centre[i][0] + f * (p.centre[j][0] - p.centre[i][0]), p.centre[i][1] + f * (p.centre[j][1] - p.centre[i][1])];
        let nrm = [p.normal[i][0] + f * (p.normal[j][0] - p.normal[i][0]), p.normal[i][1] + f * (p.normal[j][1] - p.normal[i][1])];
        let t = (((q[0] - c[0]) * nrm[0] + (q[1] - c[1]) * nrm[1]) / (0.5 * lerp(&p.gap))).clamp(-1.0, 1.0);
        lerp(&v) * (1.0 + lerp(&b) * t)
    }
}

/// Cross-section of the chosen opening for the solid model: meshed with
/// `solid.cells_across` elements across its narrowest part, with the exit
/// speed from the die-flow solve (gap-averaged for slots when
/// `solid.gap_average` is on) and die swell applied.
pub fn solid_section(project: &Project, die: &DieOutput) -> Result<Section, String> {
    let k = project.opening;
    let opening = project.die.openings.get(k).ok_or_else(|| format!("the die has no opening {}", k + 1))?;
    let single = Die { openings: vec![opening.clone()], ..project.die.clone() };
    let p = &project.solid;
    let cells = p.cells_across.max(1);
    let opts = MeshOptions { cells_across: cells, aspect: p.station_aspect * cells as f64, max_stations: 4000 };
    let m = mesh_die(&single, &opts).map_err(|e| e.to_string())?;
    let s = match die.profile(k) {
        Some(prof) if p.gap_average => Section::from_triangles(&m.nodes, &m.tris, gap_mean_sampler(prof)),
        prof => Section::from_triangles(&m.nodes, &m.tris, speed_sampler(&die.mesh, &die.flow.w, k)).map_err(|e| {
            if prof.is_some() {
                format!("{e}; the dough sticks to the walls, so turn on averaging across the gap")
            } else {
                format!("{e}; the dough sticks to the walls of this polygon opening, so give it more wall slip")
            }
        }),
    }?;
    Ok(s.swelled(project.swell))
}

/// Grows the pasta from the project's chosen opening as a solid.
pub fn run_solid(
    project: &Project,
    die: &DieOutput,
    progress: &mut dyn FnMut(&pasta_solid::Snapshot),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<SolidResult, String> {
    let section = solid_section(project, die)?;
    let sim = SolidSim::new(&section, &project.solid).map_err(|e| e.to_string())?;
    sim.run(progress, cancel).map_err(|e| e.to_string())
}

/// Summary of a full run, written as JSON by the command-line tool.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunReport {
    pub die: String,
    pub pressure_kpa: f64,
    pub flow_mm3_per_s: f64,
    pub flow_per_opening: Vec<f64>,
    pub mean_speed_per_opening: Vec<f64>,
    pub flow_error_estimate: f64,
    pub readouts: Vec<Readouts>,
    pub hanging: Option<Measurements>,
    pub relaxed: Option<Measurements>,
    pub shell_stats: Option<pasta_shell::ShellStats>,
    /// The solid piece when cut (hanging) and relaxed.
    #[serde(default)]
    pub piece_hanging: Option<SolidMeasurements>,
    #[serde(default)]
    pub piece_relaxed: Option<SolidMeasurements>,
    #[serde(default)]
    pub solid_stats: Option<SolidStats>,
}

impl RunReport {
    pub fn new(project: &Project, die: &DieOutput, shell: Option<&ShellResult>) -> Self {
        Self {
            die: project.die.name.clone(),
            pressure_kpa: die.flow.dp_kpa,
            flow_mm3_per_s: die.flow.q_total,
            flow_per_opening: die.flow.q_opening.clone(),
            mean_speed_per_opening: die.flow.mean_speed.clone(),
            flow_error_estimate: die.flow.error_estimate,
            readouts: die.readouts.clone(),
            hanging: shell.map(|s| measure(s, &s.hanging)),
            relaxed: shell.and_then(|s| s.relaxed.as_ref().map(|r| measure(s, r))),
            shell_stats: shell.map(|s| s.stats.clone()),
            piece_hanging: None,
            piece_relaxed: None,
            solid_stats: None,
        }
    }

    pub fn with_solid(mut self, r: &SolidResult) -> Self {
        self.piece_hanging = Some(pasta_solid::measure(r, &r.hanging));
        self.piece_relaxed = r.relaxed.as_ref().map(|x| pasta_solid::measure(r, x));
        self.solid_stats = Some(r.stats.clone());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_round_trips_through_ron() {
        let p = Project::from_preset("riccia").unwrap();
        let s = p.to_ron();
        let q = Project::from_ron(&s).unwrap();
        assert_eq!(p, q);
    }

    #[test]
    fn every_slot_preset_solves() {
        for preset in pasta_die::presets() {
            let p = Project::new(preset.die);
            let out = solve_die(&p, MeshQuality::Draft, None, 1, &|| false).unwrap();
            assert!(out.flow.converged, "{}", preset.id);
            let q = out.flow.q_total;
            let target = 0.5 / 1.3e-3;
            assert!((q - target).abs() / target < 1e-3, "{}: {q}", preset.id);
        }
    }

    #[test]
    fn gap_average_keeps_the_flow_and_survives_sticking() {
        for no_slip in [false, true] {
            let mut p = Project::from_preset("lasagnette").unwrap();
            p.rheology.no_slip = no_slip;
            let out = solve_die(&p, MeshQuality::Draft, None, 1, &|| false).unwrap();
            let s = solid_section(&p, &out).unwrap();
            assert!(s.speed.iter().all(|&v| v > 0.0));
            let flux = s.area() * s.mean_speed();
            let q = out.flow.q_total;
            assert!((flux - q).abs() / q < 0.03, "no_slip {no_slip}: section carries {flux}, die {q}");
            // The point speeds go to zero at sticking walls.
            p.solid.gap_average = false;
            assert_eq!(solid_section(&p, &out).is_err(), no_slip);
        }
    }

    #[test]
    fn every_slot_grows_a_section_when_stuck() {
        for preset in pasta_die::presets() {
            let mut p = Project::new(preset.die);
            p.rheology.no_slip = true;
            let out = solve_die(&p, MeshQuality::Draft, None, 1, &|| false).unwrap();
            for k in 0..p.die.openings.len() {
                if out.profile(k).is_none() {
                    continue;
                }
                p.opening = k;
                let s = solid_section(&p, &out).unwrap_or_else(|e| panic!("{} opening {k}: {e}", preset.id));
                let q = out.flow.q_opening[k];
                let flux = s.area() * s.mean_speed();
                assert!((flux - q).abs() / q < 0.05, "{} opening {k}: {flux} vs {q}", preset.id);
            }
        }
    }
}
