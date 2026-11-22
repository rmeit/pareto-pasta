use crate::{Die, Opening, PolygonOpening, Profile, Slot, WallFinish};

#[derive(Clone, Debug)]
pub struct Preset {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub die: Die,
}

fn die(name: &str, openings: Vec<Opening>) -> Die {
    Die { name: name.into(), openings, thickness: 15.0, plate_diameter: 60.0 }
}

fn straight(len: f64, gap: Profile, land: Profile) -> Opening {
    Opening::Slot(Slot {
        centerline: vec![[-len / 2.0, 0.0], [len / 2.0, 0.0]],
        closed: false,
        gap,
        land,
        wall: WallFinish::Milled,
    })
}

fn circle(cx: f64, cy: f64, r: f64, n: usize) -> Vec<[f64; 2]> {
    (0..n)
        .map(|k| {
            let a = k as f64 * std::f64::consts::TAU / n as f64;
            [cx + r * a.cos(), cy + r * a.sin()]
        })
        .collect()
}

/// Symmetric profile: `edge` over the outer `edge_frac` at each end,
/// `centre` in the middle, with ramps of `ramp_frac`.
fn ends_profile(centre: f64, edge: f64, edge_frac: f64, ramp_frac: f64) -> Profile {
    Profile::from_points(vec![
        [0.0, edge],
        [edge_frac, edge],
        [edge_frac + ramp_frac, centre],
        [1.0 - edge_frac - ramp_frac, centre],
        [1.0 - edge_frac, edge],
        [1.0, edge],
    ])
}

pub fn presets() -> Vec<Preset> {
    let mut v = Vec::new();
    v.push(Preset {
        id: "ribbon",
        name: "Ribbon",
        description: "Straight slot, 24 x 1.0 mm, uniform land. Should come out flat.",
        die: die("Ribbon", vec![straight(24.0, Profile::constant(1.0), Profile::constant(3.0))]),
    });
    // Dimensions from US 5,167,980: slot 24.56 mm, gap 0.53 mm, ends opened
    // to 0.74 mm, land 12.7 mm shortened to 11.13 mm over the last 2.79 mm.
    // The patent's own figures disagree in places, so treat as indicative.
    let len = 24.56;
    v.push(Preset {
        id: "riccia",
        name: "Ruffled edges",
        description: "Lasagna-edge die after US 5,167,980: wider gap and shorter land at both ends.",
        die: die(
            "Ruffled edges",
            vec![straight(
                len,
                ends_profile(0.53, 0.74, 0.74 / len, 0.15 / len),
                ends_profile(12.7, 11.13, 2.79 / len, 0.1 / len),
            )],
        ),
    });
    v.push(Preset {
        id: "dogbone",
        name: "Dog-bone",
        description: "Calibration die: 1.0 mm web with 1.3 mm lobes over 5 mm at each end.",
        die: die(
            "Dog-bone",
            vec![straight(24.0, ends_profile(1.0, 1.3, 5.0 / 24.0, 1.0 / 24.0), Profile::constant(3.0))],
        ),
    });
    v.push(Preset {
        id: "wedge",
        name: "Wedge",
        description: "Calibration die: gap tapers from 0.6 to 1.6 mm along 20 mm. Should curl sideways.",
        die: die(
            "Wedge",
            vec![straight(20.0, Profile::from_points(vec![[0.0, 0.6], [1.0, 1.6]]), Profile::constant(3.0))],
        ),
    });
    let arc: Vec<[f64; 2]> = (0..=6)
        .map(|k| {
            let a = (-60.0f64 + 20.0 * k as f64).to_radians();
            [12.0 * a.sin(), 12.0 * a.cos() - 12.0]
        })
        .collect();
    v.push(Preset {
        id: "curved",
        name: "Curved slot",
        description: "Arc of radius 12 mm through 120 degrees, gap 1.0 mm.",
        die: die(
            "Curved slot",
            vec![Opening::Slot(Slot {
                centerline: arc,
                closed: false,
                gap: Profile::constant(1.0),
                land: Profile::constant(3.0),
                wall: WallFinish::Milled,
            })],
        ),
    });
    v.push(Preset {
        id: "tube",
        name: "Tube",
        description: "Ring of radius 4 mm, gap 1.0 mm. Should come out straight.",
        die: die(
            "Tube",
            vec![Opening::Slot(Slot {
                centerline: circle(0.0, 0.0, 4.0, 16),
                closed: true,
                gap: Profile::constant(1.0),
                land: Profile::constant(3.0),
                wall: WallFinish::Milled,
            })],
        ),
    });
    v.push(Preset {
        id: "elbow",
        name: "Elbow tube",
        description: "Ring whose gap grows from 0.9 to 1.1 mm around it. The fast side is longer, so the tube bends.",
        die: die(
            "Elbow tube",
            vec![Opening::Slot(Slot {
                centerline: circle(0.0, 0.0, 4.0, 16),
                closed: true,
                gap: Profile::from_points(
                    (0..=16)
                        .map(|k| {
                            let u = k as f64 / 16.0;
                            [u, 0.9 + 0.1 * (1.0 - (std::f64::consts::TAU * u).cos())]
                        })
                        .collect(),
                ),
                land: Profile::constant(3.0),
                wall: WallFinish::Milled,
            })],
        ),
    });
    let gaps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6];
    v.push(Preset {
        id: "comb",
        name: "Comb",
        description: "Calibration die: six parallel 10 mm slots with gaps 0.6 to 1.6 mm. Weigh each strand.",
        die: die(
            "Comb",
            gaps.iter()
                .enumerate()
                .map(|(k, &g)| {
                    let x = -10.0 + 4.0 * k as f64;
                    Opening::Slot(Slot {
                        centerline: vec![[x, -5.0], [x, 5.0]],
                        closed: false,
                        gap: Profile::constant(g),
                        land: Profile::constant(3.0),
                        wall: WallFinish::Milled,
                    })
                })
                .collect(),
        ),
    });
    v.push(Preset {
        id: "holes",
        name: "Round holes",
        description: "Three round holes of radius 0.8, 1.0 and 1.2 mm. Flow only: strands are not sheets.",
        die: die(
            "Round holes",
            [(-4.0, 0.8), (0.0, 1.0), (4.0, 1.2)]
                .iter()
                .map(|&(x, r)| {
                    Opening::Polygon(PolygonOpening {
                        outer: circle(x, 0.0, r, 48),
                        holes: vec![],
                        land: 3.0,
                        wall: WallFinish::Milled,
                    })
                })
                .collect(),
        ),
    });
    v
}

pub fn preset(id: &str) -> Option<Preset> {
    presets().into_iter().find(|p| p.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeshOptions, mesh_die};

    #[test]
    fn every_preset_meshes() {
        for p in presets() {
            for opts in [MeshOptions::draft(), MeshOptions::fine()] {
                let m = mesh_die(&p.die, &opts).unwrap_or_else(|e| panic!("{}: {e}", p.id));
                assert!(m.min_area() > 0.0, "{}", p.id);
                assert_eq!(m.openings.len(), p.die.openings.len());
            }
        }
    }
}
