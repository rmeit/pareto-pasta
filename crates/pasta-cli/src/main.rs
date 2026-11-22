//! `pasta`: headless runs of the Tier 1 model.

use clap::{Parser, Subcommand};
use pasta_sim::{MeshQuality, Project, RunReport, SweepParam, run_shell, solve_die, sweep, write_obj, write_stl};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "pasta", about = "Simulate pasta extrusion: die flow and the growing sheet")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List the built-in dies.
    Presets,
    /// Write a project file from a preset.
    New {
        preset: String,
        #[arg(short, long, default_value = "project.ron")]
        out: PathBuf,
    },
    /// Solve the die flow and print the readouts.
    Flow {
        project: PathBuf,
        /// Write the full report as JSON.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Write the centre-line profile of each slot as CSV.
        #[arg(long)]
        csv: Option<PathBuf>,
        #[arg(long)]
        draft: bool,
    },
    /// Die flow plus the growing shell.
    Run {
        project: PathBuf,
        #[arg(long)]
        obj: Option<PathBuf>,
        #[arg(long)]
        stl: Option<PathBuf>,
        /// Write the relaxed (gravity-free) shape too, with this OBJ path.
        #[arg(long)]
        relaxed_obj: Option<PathBuf>,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Cartesian sweep over parameters, one run per core.
    Sweep {
        project: PathBuf,
        /// name=a,b,c or name=start:stop:count; repeat for more parameters.
        #[arg(long = "set", required = true)]
        set: Vec<String>,
        /// Also grow the shell for each point (slow).
        #[arg(long)]
        shell: bool,
        #[arg(short, long, default_value = "sweep.csv")]
        out: PathBuf,
    },
    /// Export the die drawing for machining.
    ExportDie {
        project: PathBuf,
        #[arg(long)]
        svg: Option<PathBuf>,
        #[arg(long)]
        dxf: Option<PathBuf>,
    },
}

fn load(path: &PathBuf) -> Result<Project, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Project::from_ron(&s).map_err(|e| format!("{}: {e}", path.display()))
}

fn fmt_opt(v: Option<f64>, unit: &str) -> String {
    v.map(|x| format!("{x:.3} {unit}")).unwrap_or_else(|| "-".into())
}

fn print_flow(p: &Project, die: &pasta_sim::DieOutput) {
    let f = &die.flow;
    println!("{}: {} nodes, solved in {:.3} s ({} Newton steps)", p.die.name, die.mesh.nodes.len(), die.seconds, f.stats.newton_steps);
    println!("  pressure drop {:.1} kPa, total flow {:.1} mm³/s, error estimate {:.1}%", f.dp_kpa, f.q_total, 100.0 * f.error_estimate);
    for (k, (q, v)) in f.q_opening.iter().zip(&f.mean_speed).enumerate() {
        println!("  opening {}: flow {:.1} mm³/s ({:.1}%), mean speed {:.2} mm/s, plug {:.0}%", k + 1, q, 100.0 * q / f.q_total, v, 100.0 * f.plug_fraction[k]);
    }
    for r in &die.readouts {
        println!("  slot {}:", r.opening + 1);
        println!("    edge/centre speed {}", fmt_opt(r.edge_centre_ratio, ""));
        println!("    fastest/slowest   {:.3}", r.max_min_ratio);
        if let Some((rad, r2)) = r.curl_radius {
            println!("    curl radius       {rad:.1} mm (straight-line fit R² {r2:.2})");
        }
        println!("    bending radius    {}", fmt_opt(r.bend_radius, "mm"));
        if let Some(t) = &r.tube {
            println!("    tube bend         {:.1} mm radius ({:.1}% speed variation)", 1.0 / t.curvature.max(1e-12), 100.0 * t.amplitude);
        }
        println!("    ruffle margin     {}", fmt_opt(r.ruffle_margin, "× threshold"));
        println!("    wavelength        {} (C = {} uncalibrated)", fmt_opt(r.wavelength, "mm"), p.ruffle_c);
        println!("    amplitude         {}", fmt_opt(r.amplitude, "mm"));
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Presets => {
            for p in pasta_sim::die::presets() {
                println!("{:10} {}", p.id, p.description);
            }
        }
        Cmd::New { preset, out } => {
            let p = Project::from_preset(&preset).ok_or_else(|| format!("no preset {preset}; see `pasta presets`"))?;
            std::fs::write(&out, p.to_ron()).map_err(|e| e.to_string())?;
            println!("wrote {}", out.display());
        }
        Cmd::Flow { project, json, csv, draft } => {
            let p = load(&project)?;
            let q = if draft { MeshQuality::Draft } else { MeshQuality::Fine };
            let die = solve_die(&p, q, None, 4, &|| false)?;
            print_flow(&p, &die);
            if let Some(path) = json {
                let r = RunReport::new(&p, &die, None);
                std::fs::write(&path, serde_json::to_string_pretty(&r).unwrap()).map_err(|e| e.to_string())?;
            }
            if let Some(path) = csv {
                let mut s = String::from("opening,s_mm,x_mm,y_mm,gap_mm,thickness_mm,speed_mm_s,bias,bend_curvature_per_mm,ruffle_map\n");
                for (prof, r) in die.profiles.iter().zip(&die.readouts) {
                    for i in 0..prof.s.len() {
                        s.push_str(&format!(
                            "{},{},{},{},{},{},{},{},{},{}\n",
                            prof.opening + 1, prof.s[i], prof.centre[i][0], prof.centre[i][1], prof.gap[i], prof.thickness[i], prof.v[i], prof.bias[i], prof.bend_curvature[i], r.ruffle_map[i]
                        ));
                    }
                }
                std::fs::write(&path, s).map_err(|e| e.to_string())?;
            }
        }
        Cmd::Run { project, obj, stl, relaxed_obj, json } => {
            let p = load(&project)?;
            let die = solve_die(&p, MeshQuality::Fine, None, 4, &|| false)?;
            print_flow(&p, &die);
            let mut last = 0;
            let shell = run_shell(
                &p,
                &die,
                &mut |s| {
                    if s.stats.rows >= last + 25 {
                        last = s.stats.rows;
                        eprintln!("  growing: {:.1} mm, {} rows, {:.1} s", s.piece_length, s.stats.rows, s.stats.seconds);
                    }
                },
                &|| false,
            )?;
            let tris = shell.outside_triangles();
            let report = RunReport::new(&p, &die, Some(&shell));
            println!("  shell: {} nodes, {} triangles, {:.1} s, {} Newton iterations", shell.stats.nodes, shell.stats.triangles, shell.stats.seconds, shell.stats.newton_iterations);
            for (label, m) in [("hanging", &report.hanging), ("relaxed", &report.relaxed)] {
                if let Some(m) = m {
                    println!("  {label}: length {:.1} mm, curl radius {}", m.length, fmt_opt(m.curl_radius, "mm"));
                    for (side, e) in [("left", &m.left_edge), ("right", &m.right_edge)] {
                        if let Some(e) = e {
                            println!("    {side} edge: wavelength {}, amplitude {:.2} mm, length ratio {:.3}", fmt_opt(e.wavelength, "mm"), e.amplitude, e.length_ratio);
                        }
                    }
                }
            }
            if let Some(path) = obj {
                write_obj(&path, &shell.hanging, &tris).map_err(|e| e.to_string())?;
            }
            if let Some(path) = stl {
                write_stl(&path, &shell.hanging, &tris).map_err(|e| e.to_string())?;
            }
            if let (Some(path), Some(rel)) = (relaxed_obj, &shell.relaxed) {
                write_obj(&path, rel, &tris).map_err(|e| e.to_string())?;
            }
            if let Some(path) = json {
                std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
            }
        }
        Cmd::Sweep { project, set, shell, out } => {
            let p = load(&project)?;
            let params: Vec<SweepParam> = set.iter().map(|s| SweepParam::parse(s)).collect::<Result<_, _>>()?;
            let rows = sweep(&p, &params, shell, &|k, n| eprint!("\r  {k}/{n}"));
            eprintln!();
            let mut s = String::new();
            for prm in &params {
                s.push_str(&prm.name);
                s.push(',');
            }
            s.push_str("opening,pressure_kpa,mean_speed,edge_centre_ratio,max_min_ratio,curl_radius,bend_radius,ruffle_margin,wavelength_est,amplitude_est");
            if shell {
                s.push_str(",sim_length,sim_curl_radius,sim_left_wavelength,sim_left_amplitude,sim_left_length_ratio");
            }
            s.push_str(",seconds,error\n");
            let f = |v: Option<f64>| v.map(|x| format!("{x}")).unwrap_or_default();
            for row in &rows {
                let prefix: String = row.values.iter().map(|(_, v)| format!("{v},")).collect();
                match &row.report {
                    Some(rep) => {
                        for r in &rep.readouts {
                            s.push_str(&prefix);
                            s.push_str(&format!(
                                "{},{},{},{},{},{},{},{},{},{}",
                                r.opening + 1,
                                rep.pressure_kpa,
                                rep.mean_speed_per_opening[r.opening],
                                f(r.edge_centre_ratio),
                                r.max_min_ratio,
                                f(r.curl_radius.map(|c| c.0)),
                                f(r.bend_radius),
                                f(r.ruffle_margin),
                                f(r.wavelength),
                                f(r.amplitude)
                            ));
                            if shell {
                                let m = rep.relaxed.as_ref().or(rep.hanging.as_ref());
                                let le = m.and_then(|m| m.left_edge.as_ref());
                                s.push_str(&format!(
                                    ",{},{},{},{},{}",
                                    f(m.map(|m| m.length)),
                                    f(m.and_then(|m| m.curl_radius)),
                                    f(le.and_then(|e| e.wavelength)),
                                    f(le.map(|e| e.amplitude)),
                                    f(le.map(|e| e.length_ratio))
                                ));
                            }
                            s.push_str(&format!(",{},\n", row.seconds));
                        }
                    }
                    None => s.push_str(&format!("{prefix},,,,,,,,,,{},{}\n", row.seconds, row.error.clone().unwrap_or_default().replace(',', ";"))),
                }
            }
            std::fs::write(&out, s).map_err(|e| e.to_string())?;
            println!("wrote {} ({} points)", out.display(), rows.len());
        }
        Cmd::ExportDie { project, svg, dxf } => {
            let p = load(&project)?;
            if let Some(path) = svg {
                std::fs::write(&path, pasta_sim::die::to_svg(&p.die)).map_err(|e| e.to_string())?;
            }
            if let Some(path) = dxf {
                std::fs::write(&path, pasta_sim::die::to_dxf(&p.die)).map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
