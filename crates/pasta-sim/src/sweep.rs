//! Parameter sweeps: one run per core, each run single-threaded.

use crate::project::{DriveSpec, MeshQuality, Project, RunReport, run_shell, solve_die};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A parameter and the values to try.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepParam {
    pub name: String,
    pub values: Vec<f64>,
}

impl SweepParam {
    /// Parses "name=a,b,c" or "name=start:stop:count".
    pub fn parse(s: &str) -> Result<Self, String> {
        let (name, rhs) = s.split_once('=').ok_or_else(|| format!("expected name=values, got {s}"))?;
        let values = if rhs.contains(':') {
            let parts: Vec<&str> = rhs.split(':').collect();
            if parts.len() != 3 {
                return Err(format!("range must be start:stop:count, got {rhs}"));
            }
            let a: f64 = parts[0].parse().map_err(|e| format!("{e}"))?;
            let b: f64 = parts[1].parse().map_err(|e| format!("{e}"))?;
            let n: usize = parts[2].parse().map_err(|e| format!("{e}"))?;
            if n < 2 {
                vec![a]
            } else {
                (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
            }
        } else {
            rhs.split(',').map(|v| v.trim().parse::<f64>().map_err(|e| format!("{v}: {e}"))).collect::<Result<_, _>>()?
        };
        apply_param(&mut Project::from_preset("ribbon").unwrap(), name, values[0])?;
        Ok(Self { name: name.to_string(), values })
    }
}

/// Sets one named parameter on a project.
pub fn apply_param(p: &mut Project, name: &str, v: f64) -> Result<(), String> {
    match name {
        "n" => p.rheology.n = v,
        "k" => p.rheology.k = v,
        "tau_y" => p.rheology.tau_y = v,
        "beta" => p.rheology.beta = v,
        "m" => p.rheology.m = v,
        "no_slip" => p.rheology.no_slip = v != 0.0,
        "output" => p.drive = DriveSpec::Output { grams_per_second: v },
        "pressure" => p.drive = DriveSpec::Pressure { kpa: v },
        "swell" => p.swell = v,
        "ruffle_c" => p.ruffle_c = v,
        "young" => p.shell.young_kpa = v,
        "poisson" => p.shell.poisson = v,
        "density" => p.shell.density = v,
        "stations" => p.shell.stations = v as usize,
        "cut" => p.shell.cut_length = v,
        "creep" => p.shell.creep_time = if v > 0.0 { Some(v) } else { None },
        "seed" => p.shell.seed = v as u64,
        _ => {
            return Err(format!(
                "unknown parameter {name}; use one of n, k, tau_y, beta, m, no_slip, output, pressure, swell, ruffle_c, young, poisson, density, stations, cut, creep, seed"
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepRow {
    pub values: Vec<(String, f64)>,
    pub report: Option<RunReport>,
    pub error: Option<String>,
    pub seconds: f64,
}

/// Runs the Cartesian product of the parameters. With `shell`, each point
/// also grows the pasta (slow). `progress` gets the number finished.
pub fn sweep(
    base: &Project,
    params: &[SweepParam],
    shell: bool,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> Vec<SweepRow> {
    let mut points: Vec<Vec<(String, f64)>> = vec![Vec::new()];
    for p in params {
        let mut next = Vec::new();
        for pt in &points {
            for &v in &p.values {
                let mut q = pt.clone();
                q.push((p.name.clone(), v));
                next.push(q);
            }
        }
        points = next;
    }
    let total = points.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    points
        .into_par_iter()
        .map(|values| {
            let t0 = std::time::Instant::now();
            let mut p = base.clone();
            p.shell.threads = 1;
            let mut row = SweepRow { values: values.clone(), report: None, error: None, seconds: 0.0 };
            let mut run = || -> Result<RunReport, String> {
                for (name, v) in &values {
                    apply_param(&mut p, name, *v)?;
                }
                let die = solve_die(&p, MeshQuality::Fine, None, 1, &|| false)?;
                let shell_result = if shell { Some(run_shell(&p, &die, &mut |_| {}, &|| false)?) } else { None };
                Ok(RunReport::new(&p, &die, shell_result.as_ref()))
            };
            match run() {
                Ok(r) => row.report = Some(r),
                Err(e) => row.error = Some(e),
            }
            row.seconds = t0.elapsed().as_secs_f64();
            let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            progress(k, total);
            row
        })
        .collect()
}
