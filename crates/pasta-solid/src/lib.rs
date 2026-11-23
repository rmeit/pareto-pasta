//! The extruded pasta as a growing solid of quadratic (10-node) tetrahedra.
//!
//! The die face is the plane z = 0 and the dough leaves along -z. The
//! cross-section is any triangulated opening (a slot, a Y, a ring with
//! ridges) with an exit speed at every point. Each row of the section
//! leaves the die at its own speeds; layers of prisms between rows are split
//! into tets whose rest shapes record how much length each column carried.
//! Quadratic elements bend correctly with one element through a wall, where
//! linear tets would be several times too stiff.

pub mod element;
mod measure;
mod section;
mod sim;

pub use measure::{SolidMeasurements, measure};
pub use section::Section;
pub use sim::{Keyframe, Orientation, Snapshot, SolidError, SolidParams, SolidResult, SolidSim, SolidStats};
