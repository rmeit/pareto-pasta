//! Die-flow solver against analytic solutions (the plan's gate for the
//! die-flow milestone).

use pasta_die::{Die, MeshOptions, Opening, Profile, Slot, WallFinish, mesh_die, preset};
use pasta_flow::{Drive, FlowParams, FlowSolver, Rheology, profiles};

fn no_cancel() -> bool {
    false
}

fn slot_die(centerline: Vec<[f64; 2]>, closed: bool, gap: Profile, land: f64) -> Die {
    Die {
        name: "test".into(),
        openings: vec![Opening::Slot(Slot {
            centerline,
            closed,
            gap,
            land: Profile::constant(land),
            wall: WallFinish::Milled,
        })],
        thickness: 10.0,
        plate_diameter: 200.0,
    }
}

fn ring(r: f64, gap: f64) -> Die {
    let ctrl: Vec<_> = (0..64)
        .map(|k| {
            let a = k as f64 * std::f64::consts::TAU / 64.0;
            [r * a.cos(), r * a.sin()]
        })
        .collect();
    slot_die(ctrl, true, Profile::constant(gap), 2.0)
}

fn newtonian(no_slip: bool) -> Rheology {
    Rheology { n: 1.0, k: 1.0, tau_y: 0.0, no_slip, beta: 1.0, m: 1.0 }
}

fn solve(die: &Die, opts: MeshOptions, rh: Rheology, drive: Drive) -> (pasta_die::TriMesh2, pasta_flow::FlowSolution) {
    let mesh = mesh_die(die, &opts).unwrap();
    let solver = FlowSolver::new(&mesh, rh.no_slip, 1).unwrap();
    let sol = solver.solve(&FlowParams { rheology: rh, drive }, None, &no_cancel).unwrap();
    assert!(sol.converged, "did not converge: residual {}", sol.residual);
    (mesh, sol)
}

#[test]
fn newtonian_rectangular_duct() {
    // Duct a x b, no slip: Q = G b^3 a / (12 mu) [1 - 192 b/(pi^5 a) sum tanh(k pi a / 2b) / k^5].
    let (a, b, land, dp) = (10.0, 1.0, 2.0, 100.0);
    let die = slot_die(vec![[-a / 2.0, 0.0], [a / 2.0, 0.0]], false, Profile::constant(b), land);
    let g = dp / land;
    let mut sum = 0.0;
    for k in (1..200).step_by(2) {
        let k = k as f64;
        sum += (k * std::f64::consts::PI * a / (2.0 * b)).tanh() / k.powi(5);
    }
    let q_exact = g * b.powi(3) * a / 12.0 * (1.0 - 192.0 * b / (std::f64::consts::PI.powi(5) * a) * sum);
    let (_, sol) = solve(&die, MeshOptions::fine(), newtonian(true), Drive::Pressure(dp));
    let err = (sol.q_total - q_exact).abs() / q_exact;
    println!("duct: Q {} exact {} rel err {err:.2e}", sol.q_total, q_exact);
    assert!(err < 0.01);
}

fn power_law_slit_mean(n: f64, k: f64, h: f64, g: f64) -> f64 {
    n / (2.0 * n + 1.0) * (h / 2.0).powf(1.0 + 1.0 / n) * (g / k).powf(1.0 / n)
}

#[test]
fn power_law_slit() {
    // A ring of radius 50 with gap 1 is a slit without end effects; the
    // curvature correction is of order (h/2R)^2.
    let (h, land, dp, k) = (1.0, 2.0, 200.0, 2.0);
    let die = ring(50.0, h);
    let mut errs = Vec::new();
    for n in [1.0, 0.7, 0.5, 0.3] {
        let rh = Rheology { n, k, tau_y: 0.0, no_slip: true, beta: 1.0, m: 1.0 };
        let mut prev = f64::INFINITY;
        for opts in [MeshOptions { cells_across: 6, aspect: 4.0, max_stations: 1600 }, MeshOptions::fine()] {
            let (_, sol) = solve(&die, opts, rh, Drive::Pressure(dp));
            let exact = power_law_slit_mean(n, k, h, dp / land);
            let err = (sol.mean_speed[0] - exact).abs() / exact;
            println!("slit n={n} cells={}: mean {} exact {exact} err {err:.2e} steps {}", opts.cells_across, sol.mean_speed[0], sol.stats.newton_steps);
            assert!(err < prev, "error did not fall with refinement");
            prev = err;
        }
        errs.push(prev);
    }
    for e in errs {
        assert!(e < 0.02, "{e}");
    }
}

#[test]
fn power_law_slit_with_slip() {
    // Wall stress is G h / 2 whatever the dough; slip speed (tau_w / beta)^(1/m).
    let (h, land, dp, k, n, beta, m) = (1.0, 2.0, 200.0, 2.0, 0.4, 20.0, 0.5);
    let die = ring(50.0, h);
    let rh = Rheology { n, k, tau_y: 0.0, no_slip: false, beta, m };
    let (_, sol) = solve(&die, MeshOptions::fine(), rh, Drive::Pressure(dp));
    let g = dp / land;
    let us = (g * h / 2.0 / beta).powf(1.0 / m);
    let exact = us + power_law_slit_mean(n, k, h, g);
    let err = (sol.mean_speed[0] - exact).abs() / exact;
    println!("slip slit: mean {} exact {exact} (slip {us}) err {err:.2e}", sol.mean_speed[0]);
    assert!(err < 0.02);
}

fn station_speed(prof: &pasta_flow::SlotProfile, u: f64) -> f64 {
    let i = ((prof.v.len() - 1) as f64 * u).round() as usize;
    prof.v[i]
}

#[test]
fn dog_bone_newtonian_ratio() {
    let die = preset("dogbone").unwrap().die;
    let (mesh, sol) = solve(&die, MeshOptions::fine(), newtonian(true), Drive::Pressure(100.0));
    let prof = &profiles(&mesh, &sol, 1.0)[0];
    let ratio = station_speed(prof, 2.5 / 24.0) / station_speed(prof, 0.5);
    println!("dog-bone lobe/centre ratio {ratio}");
    assert!((ratio - 1.69).abs() < 0.02, "{ratio}");
}

#[test]
fn dog_bone_shear_thinning_converges_on_every_mesh() {
    let die = preset("dogbone").unwrap().die;
    let rh = Rheology { n: 0.3, k: 60.0, tau_y: 0.0, no_slip: true, beta: 1.0, m: 1.0 };
    for opts in [MeshOptions::draft(), MeshOptions { cells_across: 8, ..MeshOptions::fine() }, MeshOptions::fine()] {
        let (mesh, sol) = solve(&die, opts, rh, Drive::Pressure(1000.0));
        let prof = &profiles(&mesh, &sol, 1.0)[0];
        let ratio = station_speed(prof, 2.5 / 24.0) / station_speed(prof, 0.5);
        println!(
            "n=0.3 cells={}: {} Newton steps, {:.3}s, ratio {ratio:.3}, error estimate {:.3}",
            opts.cells_across, sol.stats.newton_steps, sol.stats.seconds, sol.error_estimate
        );
        // Power law, wide lobes: ratio tends to 1.3^(1 + 1/n).
        assert!(ratio > 2.0 && ratio < 1.3f64.powf(1.0 + 1.0 / 0.3) * 1.02);
    }
}

#[test]
fn strong_slip_flattens_the_profile() {
    let die = preset("dogbone").unwrap().die;
    let mut last = f64::INFINITY;
    for beta in [100.0, 1.0, 0.01] {
        let rh = Rheology { n: 0.4, k: 60.0, tau_y: 0.0, no_slip: false, beta, m: 0.5 };
        let (mesh, sol) = solve(&die, MeshOptions::draft(), rh, Drive::Pressure(1000.0));
        let prof = &profiles(&mesh, &sol, 1.0)[0];
        let ratio = station_speed(prof, 2.5 / 24.0) / station_speed(prof, 0.5);
        println!("beta {beta}: ratio {ratio}");
        assert!(ratio < last);
        last = ratio;
    }
    assert!(last < 1.05, "{last}");
}

#[test]
fn ratios_do_not_depend_on_k_and_pressure() {
    let die = preset("riccia").unwrap().die;
    let base = Rheology { n: 0.4, k: 60.0, tau_y: 0.0, no_slip: true, beta: 1.0, m: 1.0 };
    let (mesh, a) = solve(&die, MeshOptions::draft(), base, Drive::Pressure(1000.0));
    let (_, b) = solve(&die, MeshOptions::draft(), Rheology { k: 600.0, ..base }, Drive::Pressure(3000.0));
    let pa = &profiles(&mesh, &a, 1.0)[0];
    let pb = &profiles(&mesh, &b, 1.0)[0];
    let ma = pa.v.iter().sum::<f64>();
    let mb = pb.v.iter().sum::<f64>();
    for (x, y) in pa.v.iter().zip(&pb.v) {
        assert!((x / ma - y / mb).abs() < 1e-6 * (x / ma).abs().max(1e-3), "{} {}", x / ma, y / mb);
    }
}

#[test]
fn symmetric_die_symmetric_speeds() {
    let die = preset("riccia").unwrap().die;
    let (mesh, sol) = solve(&die, MeshOptions::draft(), Rheology::default(), Drive::Pressure(1000.0));
    let p = &profiles(&mesh, &sol, 1.0)[0];
    let n = p.v.len();
    for i in 0..n {
        let rel = (p.v[i] - p.v[n - 1 - i]).abs() / p.v[i].abs().max(1e-12);
        assert!(rel < 1e-6, "station {i}: {} vs {}", p.v[i], p.v[n - 1 - i]);
    }
    // A straight symmetric channel has no bias across the gap.
    for b in &p.bias {
        assert!(b.abs() < 1e-6, "{b}");
    }
}

#[test]
fn identical_openings_share_flow_equally() {
    let mut die = preset("ribbon").unwrap().die;
    let mut second = die.openings[0].clone();
    if let Opening::Slot(s) = &mut second {
        for p in &mut s.centerline {
            p[1] += 5.0;
        }
    }
    die.openings.push(second);
    let (_, sol) = solve(&die, MeshOptions::draft(), Rheology::default(), Drive::Pressure(1000.0));
    let rel = (sol.q_opening[0] - sol.q_opening[1]).abs() / sol.q_opening[0];
    assert!(rel < 1e-8, "{rel}");
}

#[test]
fn flow_rate_mode_hits_target() {
    let die = preset("comb").unwrap().die;
    let target = 400.0;
    let (_, sol) = solve(&die, MeshOptions::draft(), Rheology::default(), Drive::FlowRate(target));
    let rel = (sol.q_total - target).abs() / target;
    println!("flow rate: {} after {} pressure iterations, dp {}", sol.q_total, sol.stats.pressure_iterations, sol.dp_kpa);
    assert!(rel < 1e-3);
    // Wider comb slots carry more flow per unit area.
    for k in 1..sol.mean_speed.len() {
        assert!(sol.mean_speed[k] > sol.mean_speed[k - 1]);
    }
}

#[test]
fn warm_start_takes_fewer_steps() {
    let die = preset("riccia").unwrap().die;
    let mesh = mesh_die(&die, &MeshOptions::draft()).unwrap();
    let solver = FlowSolver::new(&mesh, false, 1).unwrap();
    let mut params = FlowParams::default();
    let cold = solver.solve(&params, None, &no_cancel).unwrap();
    params.rheology.n *= 1.05;
    let warm = solver.solve(&params, Some(&cold), &no_cancel).unwrap();
    println!("cold {} steps, warm {} steps", cold.stats.newton_steps, warm.stats.newton_steps);
    assert!(warm.converged);
    assert!(warm.stats.newton_steps < cold.stats.newton_steps);
}
