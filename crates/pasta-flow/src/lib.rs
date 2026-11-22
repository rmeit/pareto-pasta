//! Die flow: the exit speed w(x, y) over each die opening.
//!
//! The dough moves straight through the die. w minimises the convex energy
//!
//! J(w) = ∫ [ K/(n+1) γ^(n+1) + τy γ − (Δp/L) w ] dA + ∫ β/(m+1) |w|^(m+1) ds,
//! with γ = |∇w|,
//!
//! where the first integral is over the openings (shear-thinning dough with a
//! yield stress, pushed by pressure Δp over land depth L) and the second is
//! over the walls (power-law slip). Units: mm, s, kPa.

mod fem;
mod reduce;

pub use fem::{FlowSolution, FlowSolver, SolveStats};
pub use reduce::{Readouts, SlotProfile, TubeBend, profiles, readouts};

use serde::{Deserialize, Serialize};

/// Dough and wall parameters for the die flow.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rheology {
    /// Shear-thinning index n (1 = Newtonian).
    pub n: f64,
    /// Consistency K (kPa s^n).
    pub k: f64,
    /// Yield stress (kPa).
    pub tau_y: f64,
    /// Dough sticks to the wall (no slip).
    pub no_slip: bool,
    /// Wall friction coefficient β (kPa (s/mm)^m): wall stress = β u^m.
    pub beta: f64,
    /// Slip exponent m.
    pub m: f64,
}

impl Default for Rheology {
    /// Mid-range pasta dough from the review (n 0.29 to 0.45, K 33 to 119
    /// kPa s^n). The wall values are uncalibrated placeholders.
    fn default() -> Self {
        Self { n: 0.4, k: 60.0, tau_y: 0.3, no_slip: false, beta: 30.0, m: 0.5 }
    }
}

/// What drives the flow.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Drive {
    /// Fixed pressure drop across the die (kPa).
    Pressure(f64),
    /// Fixed total volume flow through all openings (mm^3/s), as a home
    /// extruder sets it.
    FlowRate(f64),
}

impl Default for Drive {
    fn default() -> Self {
        // About 0.5 g/s of dough at 1.3 g/cm^3.
        Drive::FlowRate(385.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[derive(Default)]
pub struct FlowParams {
    pub rheology: Rheology,
    pub drive: Drive,
}

