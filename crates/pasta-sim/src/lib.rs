//! The Tier 1 pipeline: die → mesh → die flow → centre-line profile →
//! growing shell, plus project files, background jobs, sweeps and exports.

mod engine;
mod export;
mod project;
mod sweep;

pub use engine::{DieJob, Engine, ShellJob, ShellProgress};
pub use export::{write_obj, write_stl};
pub use project::{DieOutput, DriveSpec, MeshQuality, Project, RunReport, run_shell, solve_die, strip_input};
pub use sweep::{SweepParam, SweepRow, apply_param, sweep};

pub use pasta_die as die;
pub use pasta_flow as flow;
pub use pasta_shell as shell;
