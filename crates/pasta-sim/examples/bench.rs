//! Timings against the plan's performance targets.
//! cargo run --release -p pasta-sim --example bench
use pasta_sim::{MeshQuality, Project, SweepParam, solve_die, sweep};
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let p = Project::from_preset("riccia").unwrap();
    for (label, q) in [("draft", MeshQuality::Draft), ("fine", MeshQuality::Fine)] {
        let mut cold = Vec::new();
        let mut warm = Vec::new();
        let mut nodes = 0;
        let mut last = None;
        for k in 0..7 {
            let mut pk = p.clone();
            pk.rheology.n = 0.4 + 0.002 * k as f64;
            let t = Instant::now();
            let o = solve_die(&pk, q, None, 1, &|| false).unwrap();
            cold.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            let w = solve_die(&pk, q, last.as_ref(), 1, &|| false).unwrap();
            if last.is_some() {
                warm.push(t.elapsed().as_secs_f64());
            }
            nodes = o.mesh.nodes.len();
            last = Some(w);
        }
        println!("die solve {label}: {nodes} unknowns, cold {:.1} ms, warm {:.1} ms (median, includes meshing and readouts)", 1e3 * median(cold), 1e3 * median(warm));
    }
    let o = solve_die(&p, MeshQuality::Fine, None, 1, &|| false).unwrap();
    let t = Instant::now();
    for _ in 0..100 {
        let profs = pasta_flow_profiles(&o, &p);
        std::hint::black_box(profs);
    }
    println!("reduction + readouts: {:.3} ms", t.elapsed().as_secs_f64() * 10.0);
    let params = vec![
        SweepParam::parse("n=0.3:0.5:10").unwrap(),
        SweepParam::parse("beta=10:100:10").unwrap(),
        SweepParam::parse("swell=1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9").unwrap(),
    ];
    let t = Instant::now();
    let rows = sweep(&p, &params, false, &|_, _| {});
    let secs = t.elapsed().as_secs_f64();
    println!("sweep: {} fine die solves in {:.1} s on {} threads ({:.1} ms each, wall)", rows.len(), secs, rayon::current_num_threads(), 1e3 * secs / rows.len() as f64);
}

fn pasta_flow_profiles(o: &pasta_sim::DieOutput, p: &Project) -> usize {
    let profs = pasta_sim::flow::profiles(&o.mesh, &o.flow, p.swell);
    profs.iter().map(|q| pasta_sim::flow::readouts(q, p.ruffle_c).ruffle_map.len()).sum()
}
