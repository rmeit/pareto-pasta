# pareto_pasta

Simulates pasta extrusion: given a die, predict the exit-speed map across each opening and the shape of the fresh pasta that grows from it, including ruffled edges, curls and bent tubes.

This is **Tier 1** of the plan: a 2D die-flow solve, a reduction to a speed profile along each slot, and a thin elastic shell grown one row at a time at the die. It runs on a laptop; the die flow is interactive, and the 3D pasta takes seconds to a few minutes depending on resolution.

## Build and run

Rust 1.96 or later.

```sh
cargo run --release -p pasta-app              # desktop app (opens the ruffled-edge die)
cargo run --release -p pasta-app -- my.ron    # open a project file
cargo run --release -p pasta-cli -- --help    # command-line tool, installed as `pasta`
cargo test --release --workspace              # tests (about 3 minutes)
```

### Desktop app

- **Die canvas (left):** the openings coloured by exit speed. Drag the round handles to reshape a slot; the heat map follows the cursor using a coarse mesh and a fine solve runs when you let go. Scroll to zoom, right-drag to pan, double-click to reset. Hover for a probe (speed, shear rate, gap, land depth). Click a slot to grow the pasta from it. Slots are drawn 4× wider than they are by default (`gap ×` control); this is display only.
- **Profile plot:** speed along the slot centre-line relative to the centre, with the ruffle-onset threshold.
- **3D view (centre):** the growing piece, drawn with matcap-style shading, distinct back faces, ground shadow, optional wireframe, slice plane and colour maps (exit speed, time since exit, thickness). Drag to orbit, right-drag to pan, scroll to zoom, double-click to fit, click to pick a point.
- **Parameters and readouts (right):** dough and wall law, extruder output or pressure, die swell, sheet stiffness, orientation, cut length, creep, resolution; readout cards turn amber or red as the ruffle margin passes 1 or 4.
- **Bottom bar:** Grow / Stop, "grow after each edit", timeline with Play, Hanging / Relaxed.
- **Keyboard:** `Ctrl+K` command palette, `Ctrl+Z` / `Ctrl+Shift+Z` undo and redo, `Ctrl+S` save, `Ctrl+O` open, `G` grow, `F` fit view, `W` wireframe, `Space` play.
- **Exports (File menu):** project (RON), die drawing (SVG, DXF with one layer per opening named by land depth), pasta surface (OBJ, STL), report (JSON), 3D view (PNG).
- **Compare:** "Pin for comparison" keeps the current pasta as a translucent ghost and its speed profile as a dashed line.

### Command line

```sh
pasta presets                                  # built-in dies
pasta new riccia -o riccia.ron                 # project from a preset
pasta flow riccia.ron --csv profile.csv        # die flow and readouts
pasta run riccia.ron --obj pasta.obj --relaxed-obj relaxed.obj --json report.json
pasta sweep riccia.ron --set beta=3,10,30,100 --set n=0.3:0.5:5 -o sweep.csv
pasta sweep riccia.ron --set swell=1,1.2 --shell -o shell_sweep.csv   # slow: grows each point
pasta export-die riccia.ron --svg die.svg --dxf die.dxf
```

Sweep parameters: `n k tau_y beta m no_slip output pressure swell ruffle_c young poisson density stations cut creep seed`. Sweeps run one point per core.

## How it works

| Stage | Crate | Method |
|---|---|---|
| Die description | `pasta-die` | Slots as a C2 spline centre-line with gap and land-depth profiles; free polygons for other openings. Structured meshes for slots, constrained Delaunay (spade) for polygons. |
| Die flow | `pasta-flow` | Exit speed w minimises a convex energy (shear-thinning dough with yield stress, pressure over land depth, power-law wall slip). Linear triangles, Newton with line search, continuation in n, sparse Cholesky (faer). Flow-rate mode finds the pressure by a secant search. |
| Reduction | `pasta-flow` | Mean speed, speed bias across the gap, thickness and curvature at each station; readouts (curl radius, bending radius, tube bend, ruffle map and margin, wavelength and amplitude estimates). |
| Growing shell | `pasta-shell` | Triangle mesh grown one row per die push. St Venant–Kirchhoff stretching per triangle; bending from a per-face shape operator built from hinge angles; gravity; contact with the die face and a support surface. Material inside the die takes the die's shape as its rest shape and freezes when it exits, so faster stations carry more length. Implicit steps solved by damped Newton with sparse Cholesky. |
| Pipeline | `pasta-sim` | Project files, background jobs (latest edit wins, cancellable), sweeps, exports. |
| Shared | `pasta-linalg` | Sparse SPD solves on a fixed pattern with reused workspace; a seeded random generator. |
| Front ends | `pasta-cli`, `pasta-app` | Command line; egui app with its own wgpu viewport. |

Units: mm, s, kPa (die flow) and N, mm (shell).

## Test status against the plan's gates

| Gate | Result |
|---|---|
| Newtonian rectangular duct vs series solution | 0.57% error (target 1%) |
| Power-law slit, n = 1 to 0.3, vs analytic | 0.3% to 1.2% at the fine mesh, falling with refinement (target 2%) |
| Slit with power-law slip vs analytic | 0.76% (target 2%) |
| Dog-bone, Newtonian, lobe/centre ratio | 1.687 (target 1.69 ± 0.02) |
| Dog-bone, n = 0.3, no slip | converges on every mesh, 26 to 31 Newton steps total including continuation |
| Strong slip flattens the profile | ratio 1.684 → 1.001 → 1.0000001 as β falls |
| K and Δp scaling leaves ratios unchanged | to 3×10⁻⁸ (target was 10⁻⁸; the gap is solver tolerance) |
| Symmetric die, identical openings, flow-rate mode | pass (flow within 0.1%) |
| Element gradients vs finite differences | pass |
| Hinge and face bending on a cylinder vs (D/2)κ²A | exact up to the boundary row |
| Uniform speed stays flat; tube stays straight | pass |
| Linear speed gradient curls in plane with radius v/\|v′\| | 50.2 mm vs 50.2 mm expected |
| Tube with one fast side bends with radius R/a | 40.7 mm vs 40 mm (1.7%) |
| Cantilever droop vs beam theory | within 0.1% with the clamp at the last row inside the die; error against the lip halves with each refinement |
| Edge-fast profile ruffles | yes: edges 19–21% longer than the centre-line |
| Mesh doubling changes ruffle statistics by < 10% | 32 → 64 stations: wavelength 1.2%, amplitude 0.4% (`mesh_doubling`, ignored by default because it takes 5 minutes) |
| Ruffle onset and wavelength scaling vs theory | not yet tested |
| Euler buckling load | not tested |

## Performance (24-core workstation, release build)

| Operation | Measured | Plan target |
|---|---|---|
| Die solve while dragging (315 unknowns) | 0.5 ms warm, 1.9 ms cold | under 16 ms |
| Die solve at rest (4,233 unknowns) | 9 ms warm, 32 ms cold | under 50 ms warm, 500 ms cold |
| Reduction and readouts | 0.013 ms | under 2 ms |
| Sweep of 1,000 die solves | 4.7 s | under 30 s |
| Shell, 32 stations × 60 rows | 14 to 22 s | |
| Shell, 64 stations × 120 rows | 283 s | |
| Shell, 48 stations × 150 rows (ruffled die, 76 mm) | about 100 s including relaxation | preview under 20 s at 64 × 200 — **missed** |

Shell time is dominated by sparse factorisation (about 30 ms per Newton iteration at 20k unknowns); multithreaded factorisation was slower at this size. The planned "frozen tail" (rows far from the die treated as rigid) is the next step for speed.

## Known limitations

- **Uncalibrated.** The wall law dominates: on the ruffled-edge die, β from 3 to 300 changes the edge excess from 0% to 107%. The default β = 30 is a placeholder. Calibrate with the comb, wedge and dog-bone dies (strand mass is the main measurement).
- **No self-contact.** Strongly ruffled sheets (edge excess above about 40%) can pass through themselves and crumple.
- **Elastic sheet.** Dough creep is available as a single relaxation time, off by default.
- **One-way coupling.** The sheet does not affect the die flow.
- **Thin slots only for the shell.** Polygon openings (round holes, branched shapes) get a die-flow solve only.
- **Ruffle wavelength.** On the patent die the simulated edge waves are much longer (about 50 mm) than the scaling-law estimate (6 mm with an uncalibrated C). Which is closer to real pasta is an open question for calibration.
- **Not implemented from the plan:** slider look-ahead, uncertainty bands, background mesh check, SVG/DXF import, photo overlay.
