//! Die description: openings, gap and land-depth profiles, wall finish.
//!
//! All lengths are millimetres. A die face lies in the (x, y) plane and the
//! dough leaves it along -z.

mod export;
mod geometry;
mod mesh;
mod presets;

pub use export::{to_dxf, to_svg};
pub use geometry::{Centerline, Profile};
pub use mesh::{BoundaryEdge, MeshError, MeshOptions, OpeningMesh, SlotGrid, TriMesh2, mesh_die};
pub use presets::{Preset, preset, presets};

use serde::{Deserialize, Serialize};

/// Surface finish of the die channel. It scales the wall-slip coefficient.
///
/// The multipliers are placeholders until calibration with real dies: the
/// review found no slip data for pasta dough on bronze.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WallFinish {
    /// Rough bronze, as milled. The reference surface.
    #[default]
    Milled,
    /// Wire or sinker EDM finish.
    Edm,
    /// Polished brass or bronze.
    Polished,
    /// PTFE insert, the slippery extreme.
    Ptfe,
}

impl WallFinish {
    pub const ALL: [WallFinish; 4] = [WallFinish::Milled, WallFinish::Edm, WallFinish::Polished, WallFinish::Ptfe];

    /// Multiplier on the slip coefficient beta (larger = stickier).
    pub fn friction_scale(self) -> f64 {
        match self {
            WallFinish::Milled => 1.0,
            WallFinish::Edm => 0.7,
            WallFinish::Polished => 0.4,
            WallFinish::Ptfe => 0.1,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            WallFinish::Milled => "Milled bronze",
            WallFinish::Edm => "EDM",
            WallFinish::Polished => "Polished",
            WallFinish::Ptfe => "PTFE",
        }
    }
}

/// A thin slot described by its centre-line and a gap that may vary along it.
/// Ribbons, curved slots, dog-bones, wedges and tubes are all slots.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Slot {
    /// Control points of the centre-line (mm). A smooth spline passes
    /// through them.
    pub centerline: Vec<[f64; 2]>,
    /// Closed centre-line: a tube or ring.
    pub closed: bool,
    /// Gap across the slot (mm) against fractional arc length u in [0, 1].
    pub gap: Profile,
    /// Land depth (mm), the length of the channel in the extrusion
    /// direction, against fractional arc length.
    pub land: Profile,
    pub wall: WallFinish,
}

/// A free-form opening given as polygons. It gets a die-flow solve but no
/// shell: round holes, branched or chunky shapes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolygonOpening {
    pub outer: Vec<[f64; 2]>,
    pub holes: Vec<Vec<[f64; 2]>>,
    pub land: f64,
    pub wall: WallFinish,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Opening {
    Slot(Slot),
    Polygon(PolygonOpening),
}

impl Opening {
    pub fn wall(&self) -> WallFinish {
        match self {
            Opening::Slot(s) => s.wall,
            Opening::Polygon(p) => p.wall,
        }
    }

    pub fn as_slot(&self) -> Option<&Slot> {
        match self {
            Opening::Slot(s) => Some(s),
            Opening::Polygon(_) => None,
        }
    }
}

/// A die with one or more openings fed from a common chamber, so all
/// openings see the same pressure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Die {
    pub name: String,
    pub openings: Vec<Opening>,
    /// Die plate thickness (mm); the deepest land possible.
    pub thickness: f64,
    /// Diameter of the die plate (mm), as fitted to the extruder.
    pub plate_diameter: f64,
}

impl Die {
    /// Closed outlines of every opening: (opening index, polygon). A slot
    /// gives one outline, or two (outer and inner) for a tube.
    pub fn outlines(&self, samples: usize) -> Vec<(usize, Vec<[f64; 2]>)> {
        let mut out = Vec::new();
        for (k, o) in self.openings.iter().enumerate() {
            match o {
                Opening::Slot(s) => {
                    let c = Centerline::new(&s.centerline, s.closed, samples);
                    let walls = |side: f64| -> Vec<[f64; 2]> {
                        (0..c.len())
                            .map(|i| {
                                let h = s.gap.eval(c.u(i)) * 0.5 * side;
                                let p = c.point(i);
                                let n = c.normal(i);
                                [p[0] + h * n[0], p[1] + h * n[1]]
                            })
                            .collect()
                    };
                    let left = walls(1.0);
                    let mut right = walls(-1.0);
                    if s.closed {
                        out.push((k, left));
                        out.push((k, right));
                    } else {
                        right.reverse();
                        let mut poly = left;
                        poly.extend(right);
                        out.push((k, poly));
                    }
                }
                Opening::Polygon(p) => {
                    out.push((k, p.outer.clone()));
                    for h in &p.holes {
                        out.push((k, h.clone()));
                    }
                }
            }
        }
        out
    }

    /// Axis-aligned bounds of all openings: (min, max).
    pub fn bounds(&self) -> ([f64; 2], [f64; 2]) {
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for (_, poly) in self.outlines(128) {
            for p in poly {
                for d in 0..2 {
                    lo[d] = lo[d].min(p[d]);
                    hi[d] = hi[d].max(p[d]);
                }
            }
        }
        if !lo[0].is_finite() {
            return ([-1.0, -1.0], [1.0, 1.0]);
        }
        (lo, hi)
    }
}
