//! The extruded pasta as a growing thin elastic shell.
//!
//! The die face is the plane z = 0 and the dough leaves along −z. Each slot
//! station is a column of nodes; each time step the die pushes the column at
//! that station's exit speed. Material inside the die is stress-free in the
//! die's shape (it flows plastically), so every element's rest shape follows
//! the die geometry until one of its nodes leaves the die. From then on the
//! rest shape is frozen, apart from optional creep. Stations that exit faster
//! carry more length, which the sheet can only accommodate by curling,
//! bending or ruffling.

pub mod energy;
mod measure;
mod sim;

pub use measure::{Measurements, measure};
pub use sim::{
    Keyframe, Orientation, ShellError, ShellParams, ShellResult, ShellSim, ShellStats, Snapshot, StripInput,
};
