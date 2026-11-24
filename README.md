# pareto_pasta

Simulates pasta extrusion: given a die, predict the exit-speed map across each opening and the shape of the fresh pasta that grows from it, including ruffled edges, curls and bent tubes.

This is **Tier 1** of the plan: a 2D die-flow solve across each opening, then the pasta grown one row at a time at the die as a **solid of quadratic (10-node) tetrahedra**. Any closed opening works: ribbons, curved slots, tubes, and branched or chunky shapes such as a Y. It runs on a laptop; the die flow is interactive, and the 3D pasta takes one to several minutes.

The first version grew a thin elastic shell from slot openings only. That model is still in `pasta-shell` with its tests, but the app and the command line now use the solid.

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
- **Parameters and readouts (right):** dough and wall law, extruder output or pressure, die swell, dough stiffness and Poisson's ratio, orientation, cut length, resolution (elements across the opening, element length, row spacing); readout cards turn amber or red as the ruffle margin passes 1 or 4. The simulated piece reports its size, twist (turns and pitch) and bend radius.
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
pasta sweep riccia.ron --set swell=1,1.2 --grow -o grow_sweep.csv   # slow: grows each point
pasta run rotini.ron --section-only --section-csv section.csv     # the solid's cross-section and exit speeds
pasta run lasagnette.ron --cut 30 --frames frames/ --frame-every 10   # growth as numbered OBJ files
pasta export-die riccia.ron --svg die.svg --dxf die.dxf
```

Sweep parameters: `n k tau_y beta m no_slip output pressure swell ruffle_c young poisson density cells element row cut seed gap_average`. Sweeps run one point per core. `pasta run` also takes `--cut` and `--cells` to override the project.

## How it works

| Stage | Crate | Method |
|---|---|---|
| Die description | `pasta-die` | Slots as a C2 spline centre-line with gap and land-depth profiles; free polygons (with holes) for other openings, with a constant land or a land depth that varies with distance from the centre. Structured meshes for slots, constrained Delaunay (spade) for polygons. |
| Die flow | `pasta-flow` | Exit speed w minimises a convex energy (shear-thinning dough with yield stress, pressure over land depth, power-law wall slip). Linear triangles, Newton with line search, continuation in n, sparse Cholesky (faer). Flow-rate mode finds the pressure by a secant search. |
| Reduction | `pasta-flow` | Mean speed, speed bias across the gap, thickness and curvature at each station; readouts (curl radius, bending radius, tube bend, ruffle map and margin, wavelength and amplitude estimates). |
| Growing solid | `pasta-solid` | The opening is meshed with one or more quadratic triangles across its thinnest part and every point takes its exit speed from the die flow. For slots that speed is averaged across the gap (keeping its tilt from wall to wall, which bends the sheet) and along the slot over one gap, since the slow skin at the walls is pulled along by the core just past the exit; this also lets dough that sticks to the walls grow. Polygon openings use the point speeds. Rows are the material that crosses the die face at the same moment; each layer of prisms between two rows is split into three quadratic tets (side diagonals chosen by node number, so neighbours match), with the die's kinematics at mid-layer as rest shape, so fast points carry more length. Compressible neo-Hookean material, consistent gravity loads, inertia, contact with the die face and a support surface. Implicit steps solved by Newton with the exact Hessian (shifted when indefinite) and sparse Cholesky. |
| Growing shell (previous model) | `pasta-shell` | Triangle mesh grown one row per die push. St Venant–Kirchhoff stretching per triangle; bending from a per-face shape operator built from hinge angles; gravity; contact with the die face and a support surface. Material inside the die takes the die's shape as its rest shape and freezes when it exits, so faster stations carry more length. Implicit steps solved by damped Newton with sparse Cholesky. |
| Pipeline | `pasta-sim` | Project files, background jobs (latest edit wins, cancellable), sweeps, exports. |
| Shared | `pasta-linalg` | Sparse SPD solves on a fixed pattern with reused workspace (faer's ordering and symbolic analysis). With several threads the numeric factorisation works on independent branches of the elimination tree at the same time, which faer does not; a seeded random generator. |
| Front ends | `pasta-cli`, `pasta-app` | Command line; egui app with its own wgpu viewport. |

Units: mm, s, kPa (die flow) and N, mm (pasta).

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
| Quadratic tet: gradient and Hessian vs finite differences; rigid motion costs nothing; volume, load and mass integrals | pass |
| Solid: uniform speed stays straight, no twist | pass (twist < 10⁻⁶ turns) |
| Solid: linear speed gradient curls in plane with radius v/\|v′\| | 50.02 mm vs 50.00 mm expected |
| Solid: cantilever droop vs beam theory | between the clamp at the die face and half a row inside it; error against the face falls from 19% to 8.7% as rows shrink from 1.1 to 0.78 mm |
| Solid: gap-averaged section carries the die's flow, including with walls stuck (every slot preset) | within 3% (lasagnette), 5% (all) |
| Solid: ribbon with the outer 1.5 mm 25% faster, no die | ruffles: centre flat, edges wave ±1.5 mm at 14.5 mm wavelength, edges 20% longer than the centre (not yet checked on a finer mesh) |
| Shell: hinge and face bending on a cylinder vs (D/2)κ²A | exact up to the boundary row |
| Shell: uniform speed stays flat; tube stays straight | pass |
| Shell: linear speed gradient curls in plane | 50.2 mm vs 50.2 mm expected |
| Shell: tube with one fast side bends with radius R/a | 40.7 mm vs 40 mm (1.7%) |
| Shell: cantilever droop vs beam theory | within 0.1% with the clamp at the last row inside the die |
| Shell: edge-fast profile ruffles; mesh doubling | edges 19–21% longer; 32 → 64 stations changes wavelength 1.2%, amplitude 0.4% |
| Ruffle onset and wavelength scaling vs theory | not yet tested |
| Euler buckling load | not tested |

## Example dies grown as solids

From `pasta new <preset>`, default dough and wall law (uncalibrated), one die solve then `pasta run`:

| Die | Section | Speed range | Result | Time |
|---|---|---|---|---|
| `lasagnette`: 20 mm ribbon, edges opened over 2.5 mm | 90 quadratic triangles, 0.57 mm | 27 to 38 mm/s (edges 1.37×) | 60 mm piece: the edges wave in a chain of saddles about 25–30 mm long; no fine frills | 8.4 min, 29k tets, 176k unknowns |
| `campanelle`: 240° arc, gap and land graded along it | 94 triangles, 0.59 mm | 12 to 44 mm/s (3.6×) | 30 mm piece: the C-shaped sheet coils and tapers into a twisted horn (0.4 turns) | 2.4 min, 15k tets |
| `rotini`: Y with curved, bulbed arms; land 8 mm at the core, 2.5 mm at the tips | 77 triangles, 0.80 mm (`--cells 3`) | 8 to 25 mm/s (tips 2.7×) | Three-armed helix, 1.45 turns in 19 mm (pitch about 13 mm) | 1.4 min, 11k tets |

What the die design taught: with the default wall slip the dough moves almost as a plug, so arm or edge shape alone barely changes the speed; land length is the strong lever (hence the radial land profile for polygon openings). Fast bands narrower than about a millimetre are dragged back by the slot's end walls. A 12 mm ribbon with 1.44× edges twists into a helicoid rather than ruffling; a 20 mm ribbon with 1.2–1.4× edges saddles, then waves.

## Performance (24-core workstation, release build)

| Operation | Measured | Plan target |
|---|---|---|
| Die solve while dragging (315 unknowns) | 0.5 ms warm, 1.9 ms cold | under 16 ms |
| Die solve at rest (4,233 unknowns) | 9 ms warm, 32 ms cold | under 50 ms warm, 500 ms cold |
| Reduction and readouts | 0.013 ms | under 2 ms |
| Sweep of 1,000 die solves | 4.7 s | under 30 s |
| Solid, 12 mm ribbon, 54 section triangles × 72 rows (67k unknowns) | 142 s | |
| Solid, 20 mm ribbon, 90 triangles × 110 rows (176k unknowns) | 503 s (before the speed-ups below) | |
| Solid, 20 mm ribbon with fast edges, 36 elements across, 40 mm (85k unknowns), 8 threads | 108 s (was 196 s) | |
| Shell, 32 stations × 60 rows | 14 to 22 s | |
| Shell, 64 stations × 120 rows | 283 s | |
| Shell, 48 stations × 150 rows (ruffled die, 76 mm) | about 100 s including relaxation | preview under 20 s at 64 × 200 — **missed** |

Both models spend over 90% of their time in sparse factorisation. Speed tests on the 85k-unknown ribbon: faer's threads only work inside dense blocks and made it 13% slower than one thread; keeping no spare rows in the pattern saves 17%; factorising independent branches of the elimination tree in parallel gives 1.43× at 8 threads (the tree's longest chain allows at most 1.9×; 16 threads add nothing); together 1.8×. Reusing a factorisation as a conjugate-gradient preconditioner lost (about 12 iterations per Newton step cost as much as the factorisations saved). Element assembly and the line search take about 6%, so hand-written SIMD there would not pay. The solid needs 5 to 8 Newton iterations per row; about one in twenty systems is indefinite while the piece buckles and is solved with projected material tangents instead. Run long jobs one at a time: several at once oversubscribe the cores. The planned "frozen tail" (rows far from the die treated as rigid) is the next step for speed.

## Known limitations

- **Uncalibrated.** The wall law dominates: on the ruffled-edge die, β from 3 to 300 changes the edge excess from 0% to 107%. The default β = 30 is a placeholder. Calibrate with the comb, wedge and dog-bone dies (strand mass is the main measurement).
- **No self-contact.** Coiling or strongly ruffled pieces (the campanelle, edge excess above about 40%) can pass through themselves.
- **Elastic dough.** The solid has no creep yet (the old shell had a single relaxation time).
- **One-way coupling.** The pasta does not affect the die flow.
- **Clamp at the die.** The last row of driven nodes sits up to a row inside the die face, so droop near the die is slightly overestimated until rows are short.
- **Ruffle wavelength.** On the patent die the simulated edge waves are much longer (about 50 mm) than the scaling-law estimate (6 mm with an uncalibrated C). Which is closer to real pasta is an open question for calibration.
- **Not implemented from the plan:** slider look-ahead, uncertainty bands, background mesh check, SVG/DXF import, photo overlay.
