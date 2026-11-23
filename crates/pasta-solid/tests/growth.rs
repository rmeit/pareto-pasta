use pasta_solid::{Orientation, Section, SolidParams, SolidResult, SolidSim, measure};

/// Rectangle `width` (x) by `thick` (y), centred on the origin.
fn rect(width: f64, thick: f64, nw: usize, nh: usize, speed: impl Fn(f64, f64) -> f64) -> Section {
    let mut nodes = Vec::new();
    for j in 0..=nh {
        for i in 0..=nw {
            nodes.push([width * (i as f64 / nw as f64 - 0.5), thick * (j as f64 / nh as f64 - 0.5)]);
        }
    }
    let id = |i: usize, j: usize| (j * (nw + 1) + i) as u32;
    let mut tris = Vec::new();
    for j in 0..nh {
        for i in 0..nw {
            tris.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
            tris.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
        }
    }
    Section::from_triangles(&nodes, &tris, |p| speed(p[0], p[1])).unwrap()
}

fn run(s: &Section, p: &SolidParams) -> SolidResult {
    let t = std::time::Instant::now();
    let r = SolidSim::new(s, p).unwrap().run(&mut |_| {}, &|| false).unwrap();
    println!(
        "{:.1} s, {} rows, {} tets, {} dofs, {} Newton its, {} unconverged steps, {} shifted",
        t.elapsed().as_secs_f64(),
        r.stats.rows,
        r.stats.tets,
        r.stats.dofs,
        r.stats.newton_iterations,
        r.stats.unconverged_steps,
        r.stats.shifted_solves
    );
    r
}

#[test]
fn uniform_speed_stays_straight() {
    let s = rect(6.0, 0.8, 8, 1, |_, _| 10.0);
    let p = SolidParams { cut_length: 15.0, threads: 4, ..Default::default() };
    let r = run(&s, &p);
    let m = measure(&r, r.relaxed.as_ref().unwrap());
    println!("{m:?}");
    assert!((m.length - 15.0).abs() < 1.5, "{m:?}");
    assert!(m.bend_radius.is_none_or(|b| b > 1000.0), "{m:?}");
    assert!(m.twist_turns.abs() < 0.01, "{m:?}");
}

#[test]
fn linear_speed_curls_in_plane() {
    // v = v0 (1 + g x): the centre-line curls with radius 1 / g.
    let g = 0.02;
    let s = rect(6.0, 0.8, 10, 1, |x, _| 10.0 * (1.0 + g * x));
    let p = SolidParams { cut_length: 40.0, threads: 4, ..Default::default() };
    let r = run(&s, &p);
    let m = measure(&r, r.relaxed.as_ref().unwrap());
    println!("{m:?}");
    let radius = m.bend_radius.expect("curled");
    assert!((radius - 1.0 / g).abs() / (1.0 / g) < 0.05, "radius {radius}");
}

#[test]
fn cantilever_droop_matches_beam_theory() {
    // Horizontal extrusion of a stiff strip, thin along y, gravity along -y.
    // With nu = 0 the plate and beam answers agree: q L^4 / (8 E I). The
    // clamp is the last row of nodes still driven by the die, half a row
    // inside the face on average, so L is measured from there; the error
    // against the face itself must fall as the rows get shorter.
    let (w, h, young_kpa) = (8.0, 1.0, 2000.0);
    let mut errors = Vec::new();
    for nw in [8usize, 16] {
        let s = rect(w, h, nw, 1, |_, _| 10.0);
        let p = SolidParams {
            young_kpa,
            poisson: 0.0,
            orientation: Orientation::Horizontal,
            support_drop: 1e4,
            cut_length: 12.0,
            relax_after_cut: false,
            threads: 4,
            ..Default::default()
        };
        let r = run(&s, &p);
        // Oldest row (level 0) is the tip; average its corner nodes.
        let nc = r.section.n_corners;
        let tip_y = (0..nc).map(|k| r.hanging[k][1] as f64).sum::<f64>() / nc as f64;
        let tip_z = (0..nc).map(|k| r.hanging[k][2] as f64).sum::<f64>() / nc as f64;
        let q = 1.3e-9 * 9810.0 * w * h; // N/mm
        let ei = young_kpa * 1e-3 * w * h.powi(3) / 12.0;
        let beam = |l: f64| q * l.powi(4) / (8.0 * ei);
        let l = -tip_z;
        let raw = (-tip_y - beam(l)) / beam(l);
        let clamp = l + 0.5 * r.row_spacing;
        let shifted = (-tip_y - beam(clamp)) / beam(clamp);
        println!(
            "nw {nw}: row {:.3} mm, L {l:.2}, tip {:.5}, beam(L) {:.5} error {raw:+.4}, beam(L + row/2) error {shifted:+.4}",
            r.row_spacing,
            -tip_y,
            beam(l)
        );
        // The discrete clamp lies within the first row inside the face.
        assert!(-tip_y > 0.99 * beam(l) && -tip_y < 1.01 * beam(clamp), "{raw} {shifted}");
        errors.push((raw, shifted));
    }
    assert!(errors[1].0.abs() < 0.7 * errors[0].0.abs(), "error should fall with refinement: {errors:?}");
}
