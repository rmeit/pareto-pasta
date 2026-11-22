//! Growth tests with synthetic speed profiles (the plan's shell gates).

use pasta_shell::{Orientation, ShellParams, ShellSim, StripInput, measure};

fn ribbon(width: f64, n: usize, speed: impl Fn(f64) -> f64, h: f64) -> StripInput {
    let xs: Vec<f64> = (0..n).map(|i| -width / 2.0 + width * i as f64 / (n - 1) as f64).collect();
    StripInput {
        closed: false,
        centre: xs.iter().map(|&x| [x, 0.0]).collect(),
        normal: vec![[0.0, 1.0]; n],
        v: xs.iter().map(|&x| speed(x / (width / 2.0))).collect(),
        thickness: vec![h; n],
        bend_curvature: vec![0.0; n],
    }
}

fn run(input: &StripInput, p: &ShellParams) -> pasta_shell::ShellResult {
    let sim = ShellSim::new(input, p).unwrap();
    let mut last_print = 0;
    sim.run(
        &mut |s| {
            if s.stats.rows >= last_print + 40 {
                last_print = s.stats.rows;
                println!(
                    "  rows {} length {:.1} mm, {} Newton its, {:.2}s",
                    s.stats.rows, s.piece_length, s.stats.newton_iterations, s.stats.seconds
                );
            }
        },
        &|| false,
    )
    .unwrap()
}

fn params() -> ShellParams {
    ShellParams { stations: 24, cut_length: 30.0, threads: 4, ..Default::default() }
}

#[test]
fn uniform_speed_stays_flat() {
    let input = ribbon(12.0, 24, |_| 10.0, 1.0);
    let r = run(&input, &params());
    let m = measure(&r, r.relaxed.as_ref().unwrap());
    println!("uniform: {m:?} stats {:?}", r.stats);
    for e in [m.left_edge.as_ref().unwrap(), m.right_edge.as_ref().unwrap()] {
        assert!(e.wavelength.is_none(), "edge should be flat: {e:?}");
        assert!((e.length_ratio - 1.0).abs() < 0.01);
    }
    assert!(m.curl_radius.is_none());
    assert!((m.length - 30.0).abs() < 2.0, "{}", m.length);
}

#[test]
fn linear_speed_curls_in_plane() {
    // v = v0 (1 + g x): in-plane curl radius v0 / (v0 g) = 1/g at the centre.
    let width = 10.0;
    let g = 0.02; // per mm -> radius 50 mm
    let input = ribbon(width, 24, |u| 10.0 * (1.0 + g * u * width / 2.0), 1.0);
    let p = ShellParams { cut_length: 40.0, ..params() };
    let r = run(&input, &p);
    let m = measure(&r, r.relaxed.as_ref().unwrap());
    println!("linear: {m:?}");
    let radius = m.curl_radius.expect("should curl");
    assert!((radius - 1.0 / g).abs() / (1.0 / g) < 0.05, "radius {radius}");
    // In-plane: the edges stay flat.
    assert!(m.left_edge.as_ref().unwrap().wavelength.is_none());
}

#[test]
fn fast_edges_ruffle() {
    // Edges 25% faster than the centre over the outer fifth of a 24 mm ribbon.
    let input = ribbon(24.0, 48, |u| {
        let a = (u.abs() - 0.6).max(0.0) / 0.4;
        10.0 * (1.0 + 0.25 * a * a)
    }, 0.8);
    let p = ShellParams { stations: 32, cut_length: 45.0, orientation: Orientation::Down, ..params() };
    let r = run(&input, &p);
    let m = measure(&r, &r.hanging);
    println!("ruffle: {m:?} stats {:?}", r.stats);
    let e = m.left_edge.as_ref().unwrap();
    assert!(e.wavelength.is_some(), "expected ruffles: {e:?}");
    assert!(e.length_ratio > 1.1);
}

fn ring(r: f64, n: usize, speed: impl Fn(f64) -> f64, h: f64) -> StripInput {
    let th: Vec<f64> = (0..n).map(|i| std::f64::consts::TAU * i as f64 / n as f64).collect();
    StripInput {
        closed: true,
        centre: th.iter().map(|t| [r * t.cos(), r * t.sin()]).collect(),
        normal: th.iter().map(|t| [-t.sin(), t.cos()]).collect(),
        v: th.iter().map(|&t| speed(t)).collect(),
        thickness: vec![h; n],
        bend_curvature: vec![0.0; n],
    }
}

#[test]
fn uniform_tube_stays_straight() {
    let input = ring(4.0, 32, |_| 10.0, 1.0);
    let r = run(&input, &ShellParams { stations: 32, cut_length: 30.0, ..params() });
    let m = measure(&r, r.relaxed.as_ref().unwrap());
    println!("tube: {m:?}");
    assert!(m.curl_radius.is_none(), "{m:?}");
}

#[test]
fn one_sided_tube_bends() {
    // v = v0 (1 + a cos θ): the tube bends with curvature a / R.
    let (radius, a) = (4.0, 0.1);
    let input = ring(radius, 32, |t| 10.0 * (1.0 + a * t.cos()), 1.0);
    let r = run(&input, &ShellParams { stations: 32, cut_length: 40.0, ..params() });
    let m = measure(&r, r.relaxed.as_ref().unwrap());
    let expected = radius / a;
    println!("elbow: {m:?}, expected radius {expected}");
    let got = m.curl_radius.expect("should bend");
    assert!((got - expected).abs() / expected < 0.05, "{got}");
}

#[test]
fn cantilever_droop_matches_beam_theory() {
    // Horizontal extrusion, stiff enough for small deflections:
    // tip deflection q L^4 / (8 D) per unit width, L measured from the lip.
    // The discrete clamp acts at the last row still inside the die, one
    // cell above the lip, so the error against L should halve when the mesh
    // is refined, and L plus one cell should match within 5%.
    let (h, young_kpa, width) = (1.0, 2000.0, 8.0);
    let mut errors = Vec::new();
    for stations in [16usize, 32] {
        let input = ribbon(width, stations, |_| 10.0, h);
        let p = ShellParams {
            stations,
            young_kpa,
            poisson: 0.0,
            orientation: Orientation::Horizontal,
            support_drop: 1e4,
            cut_length: 12.0,
            relax_after_cut: false,
            ..params()
        };
        let r = run(&input, &p);
        let n_s = r.stations;
        // Oldest row = the tip.
        let tip_y = (0..n_s).map(|i| r.hanging[i][1] as f64).sum::<f64>() / n_s as f64;
        let tip_z = (0..n_s).map(|i| r.hanging[i][2] as f64).sum::<f64>() / n_s as f64;
        let cell = width / (stations - 1) as f64;
        let q = 1.3e-9 * 9810.0 * h; // N/mm²
        let d = young_kpa * 1e-3 * h.powi(3) / 12.0;
        let beam = |l: f64| q * l.powi(4) / (8.0 * d);
        let raw = ((-tip_y) - beam(-tip_z)).abs() / beam(-tip_z);
        let shifted = ((-tip_y) - beam(-tip_z + cell)).abs() / beam(-tip_z + cell);
        println!("cantilever: {stations} stations, L {:.2} tip {:.4} beam(L) {:.4} error {raw:.3}, beam(L + cell) error {shifted:.3}", -tip_z, -tip_y, beam(-tip_z));
        errors.push((raw, shifted));
    }
    assert!(errors[1].0 < 0.6 * errors[0].0, "error should fall with refinement: {errors:?}");
    for (_, shifted) in &errors {
        assert!(*shifted < 0.05, "{errors:?}");
    }
}

/// Plan gate: doubling the mesh changes ruffle wavelength and amplitude by
/// less than 10%. Slow (minutes), so ignored by default:
/// cargo test --release -p pasta-shell --test growth mesh_doubling -- --ignored --nocapture
#[test]
#[ignore]
fn mesh_doubling() {
    let speed = |u: f64| {
        let a = (u.abs() - 0.6).max(0.0) / 0.4;
        10.0 * (1.0 + 0.25 * a * a)
    };
    let mut out = Vec::new();
    for stations in [32usize, 64] {
        let input = ribbon(24.0, 256, speed, 0.8);
        let p = ShellParams { stations, cut_length: 60.0, refine: 1.0, ..params() };
        let r = run(&input, &p);
        let m = measure(&r, r.relaxed.as_ref().unwrap());
        let e = m.left_edge.clone().unwrap();
        println!("mesh doubling: {stations} stations: wavelength {:?} amplitude {:.3} length ratio {:.3} ({:.1} s)", e.wavelength, e.amplitude, e.length_ratio, r.stats.seconds);
        out.push(e);
    }
    let (a, b) = (&out[0], &out[1]);
    let dl = (a.wavelength.unwrap() - b.wavelength.unwrap()).abs() / b.wavelength.unwrap();
    let da = (a.amplitude - b.amplitude).abs() / b.amplitude;
    println!("mesh doubling: wavelength change {:.1}%, amplitude change {:.1}%", 100.0 * dl, 100.0 * da);
    assert!(dl < 0.1 && da < 0.1);
}
