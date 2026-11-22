//! Runs a synthetic strip and writes OBJ files for inspection.
//! cargo run --release -p pasta-shell --example debug_strip -- <case> <outdir>
use pasta_shell::{ShellParams, ShellSim, StripInput, measure};
use std::io::Write;

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

fn write_obj(path: &str, pos: &[[f32; 3]], tris: &[[u32; 3]]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for p in pos {
        writeln!(f, "v {} {} {}", p[0], p[1], p[2]).unwrap();
    }
    for t in tris {
        writeln!(f, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("linear");
    let out = args.get(2).cloned().unwrap_or(".".into());
    let (input, cut) = match case {
        "uniform" => (ribbon(12.0, 24, |_| 10.0, 1.0), 30.0),
        "linear" => (ribbon(10.0, 24, |u| 10.0 * (1.0 + 0.02 * u * 5.0), 1.0), 40.0),
        "ruffle" => (
            ribbon(24.0, 48, |u| {
                let a = (u.abs() - 0.6).max(0.0) / 0.4;
                10.0 * (1.0 + 0.25 * a * a)
            }, 0.8),
            60.0,
        ),
        "cantilever" => (ribbon(8.0, 16, |_| 10.0, 1.0), 12.0),
        _ => panic!("unknown case"),
    };
    let mut p = ShellParams { stations: input.len(), cut_length: cut, threads: 1, ..Default::default() };
    if case == "cantilever" {
        p.young_kpa = 2000.0;
        p.poisson = 0.0;
        p.orientation = pasta_shell::Orientation::Horizontal;
        p.support_drop = 1e4;
        p.relax_after_cut = false;
        p.substeps = std::env::var("SUBSTEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    }
    let sim = ShellSim::new(&input, &p).unwrap();
    let r = sim
        .run(&mut |s| {
            if s.stats.rows % 20 == 0 {
                eprintln!("rows {} len {:.1} its {} unconv {} {:.2}s", s.stats.rows, s.piece_length, s.stats.newton_iterations, s.stats.unconverged_steps, s.stats.seconds);
            }
        }, &|| false)
        .unwrap();
    let tris = r.outside_triangles();
    write_obj(&format!("{out}/{case}_hanging.obj"), &r.hanging, &tris);
    if let Some(rel) = &r.relaxed {
        write_obj(&format!("{out}/{case}_relaxed.obj"), rel, &tris);
        println!("relaxed {:?}", measure(&r, rel));
    }
    println!("hanging {:?}", measure(&r, &r.hanging));
    println!("{:?}", r.stats);
    if case == "cantilever" {
        for row in 0..r.hanging.len() / r.stations {
            let k = row * r.stations + r.stations / 2;
            println!("prof {} {} {}", row, r.hanging[k][2], r.hanging[k][1]);
        }
        for row in [0usize, 6, 12, 18] {
            let ys: Vec<String> = (0..r.stations).map(|i| format!("{:.4}", r.hanging[row * r.stations + i][1])).collect();
            println!("row {row} y across width: {}", ys.join(" "));
        }
    }
    let n_s = r.stations;
    let rows = r.hanging.len() / n_s;
    for row in (0..rows).step_by(if case == "cantilever" { 1 } else { (rows / 12).max(1) }) {
        let free = (0..n_s).filter(|&i| r.outside[row * n_s + i]).count();
        let z: Vec<f32> = [0, n_s / 2, n_s - 1].iter().map(|&i| r.hanging[row * n_s + i][2]).collect();
        let x: Vec<f32> = [0, n_s / 2, n_s - 1].iter().map(|&i| r.hanging[row * n_s + i][0]).collect();
        let y: Vec<f32> = [0, n_s / 2, n_s - 1].iter().map(|&i| r.hanging[row * n_s + i][1]).collect();
        println!("row {row:3} free {free:2} z {z:?} y {y:?} x {x:?}");
    }
}
