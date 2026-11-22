use pasta_die::{Die, MeshOptions, TriMesh2, mesh_die};
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
    /// Which slot opening the shell grows from.
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
        }
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
}
