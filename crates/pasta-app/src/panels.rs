//! Panels: top bar, die panel, parameters and readouts, bottom bar, the
//! overlay on the 3D view, and the command palette.

use crate::app::{App, ColorMode, Command, ShapeView};
use crate::viewport::Material;
use egui::{Color32, RichText};
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints};
use pasta_sim::die::{Opening, Profile, WallFinish};
use pasta_sim::solid::{Orientation, measure};
use pasta_sim::{DriveSpec, Project};

pub fn top_bar(app: &mut App, ui: &mut egui::Ui) -> Option<Command> {
    let mut cmd = None;
    egui::Panel::top("top").show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("pareto_pasta").strong());
            let name = app.path.as_ref().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "Untitled project".into());
            ui.label(RichText::new(name).weak());
            ui.separator();
            ui.menu_button("Presets", |ui| {
                for p in pasta_sim::die::presets() {
                    if ui.button(p.name).on_hover_text(p.description).clicked() {
                        let mut proj = Project::new(p.die);
                        // Keep the dough and pasta settings.
                        proj.rheology = app.project.rheology;
                        proj.drive = app.project.drive;
                        proj.swell = app.project.swell;
                        proj.ruffle_c = app.project.ruffle_c;
                        proj.solid = app.project.solid.clone();
                        app.load_project(proj, None);
                        ui.close();
                    }
                }
            });
            ui.menu_button("File", |ui| {
                for c in [Command::Open, Command::Save, Command::SaveAs] {
                    if ui.add(egui::Button::new(c.label()).shortcut_text(c.shortcut())).clicked() {
                        cmd = Some(c);
                        ui.close();
                    }
                }
                ui.separator();
                for c in [Command::ExportSvg, Command::ExportDxf, Command::ExportObj, Command::ExportStl, Command::ExportReport, Command::Screenshot] {
                    if ui.button(c.label()).clicked() {
                        cmd = Some(c);
                        ui.close();
                    }
                }
            });
            ui.add_enabled_ui(app.can_undo(), |ui| {
                if ui.button("Undo").on_hover_text("Ctrl+Z").clicked() {
                    cmd = Some(Command::Undo);
                }
            });
            ui.add_enabled_ui(app.can_redo(), |ui| {
                if ui.button("Redo").on_hover_text("Ctrl+Shift+Z").clicked() {
                    cmd = Some(Command::Redo);
                }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Commands  Ctrl+K").clicked() {
                    app.palette_open = true;
                    app.palette_query.clear();
                }
                if app.ghost.is_some() {
                    if ui.button("Clear comparison").clicked() {
                        cmd = Some(Command::ClearGhost);
                    }
                } else if ui.button("Pin for comparison").on_hover_text("Keep this result as a ghost while you edit").clicked() {
                    cmd = Some(Command::PinGhost);
                }
            });
        });
    });
    cmd
}

pub fn bottom_bar(app: &mut App, ui: &mut egui::Ui) -> Option<Command> {
    let mut cmd = None;
    egui::Panel::bottom("bottom").show(ui, |ui| {
        ui.horizontal(|ui| {
            if app.grow_running() {
                if ui.button("Stop").clicked() {
                    cmd = Some(Command::Stop);
                }
            } else if ui.button("Grow").on_hover_text("Grow the pasta from the selected slot (G)").clicked() {
                cmd = Some(Command::Grow);
            }
            ui.checkbox(&mut app.auto_grow, "Grow after each edit");
            ui.separator();
            let n = app.piece.as_ref().map(|s| s.1.keyframes.len()).unwrap_or(0);
            if n > 0 {
                let play = if app.playing { "Pause" } else { "Play" };
                if ui.button(play).on_hover_text("Play the growth (Space)").clicked() {
                    app.playing = !app.playing;
                }
                let mut k = app.timeline.unwrap_or(n - 1).min(n - 1);
                let r = ui.add(egui::Slider::new(&mut k, 0..=n - 1).show_value(false).text("timeline"));
                if r.changed() {
                    app.timeline = Some(k);
                    app.shape_view = ShapeView::Timeline;
                    app.playing = false;
                }
                if let Some(s) = &app.piece {
                    let t = s.1.keyframes.get(k).map(|f| f.time).unwrap_or(0.0);
                    ui.label(format!("{t:.2} s"));
                }
                ui.selectable_value(&mut app.shape_view, ShapeView::Hanging, "Hanging");
                ui.selectable_value(&mut app.shape_view, ShapeView::Relaxed, "Relaxed").on_hover_text("Cut from the die, gravity off");
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(status_text(app)).weak());
            });
        });
    });
    cmd
}

fn status_text(app: &App) -> String {
    let mut parts = Vec::new();
    if !app.status.is_empty() {
        parts.push(app.status.clone());
    }
    if let Some(e) = &app.die_error {
        parts.push(format!("Die: {e}"));
    } else if let Some(d) = &app.die {
        let f = &d.1.flow;
        parts.push(format!(
            "Die flow: {} unknowns, {} Newton steps, {:.0} ms, error ≈{:.0}%",
            d.1.mesh.nodes.len(),
            f.stats.newton_steps,
            d.1.seconds * 1e3,
            100.0 * f.error_estimate
        ));
    }
    if let Some(e) = &app.grow_error {
        parts.push(format!("Pasta: {e}"));
    } else if app.grow_running() {
        if let Some(p) = &app.grow_progress {
            parts.push(format!("Growing: {:.1} mm, {} rows, {:.1} s", p.snapshot.piece_length, p.snapshot.stats.rows, p.snapshot.stats.seconds));
        } else {
            parts.push("Growing…".into());
        }
    } else if let Some(s) = &app.piece {
        let st = &s.1.stats;
        parts.push(format!("Pasta: {} nodes, {} Newton iterations, {:.1} s", st.nodes, st.newton_iterations, st.seconds));
    }
    parts.join("   ·   ")
}

pub fn die_panel(app: &mut App, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Die").strong());
        ui.label(RichText::new(&app.project.die.name).weak());
    });
    ui.horizontal(|ui| {
        ui.label(RichText::new("Exit-speed heat map · drag points · scroll to zoom").weak().small());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(egui::DragValue::new(&mut app.canvas.exaggerate).range(1.0..=20.0).speed(0.1).prefix("gap ×"))
                .on_hover_text("Draw slots wider than they are, to see the speed across the gap. Display only.");
        });
    });
    let height = (ui.available_height() * 0.58).max(220.0);
    let die = app.die.as_ref().map(|d| &d.1);
    let (probe, clicked) = app.canvas.ui(ui, &mut app.project, die, height);
    if let Some(o) = clicked
        && app.project.die.openings.get(o).map(|op| op.as_slot().is_some()).unwrap_or(false) {
            app.project.opening = o;
        }
    if let Some(p) = probe {
        egui::Tooltip::always_open(ui.ctx().clone(), ui.layer_id(), ui.scope_id().with("probe"), egui::PopupAnchor::Pointer).show(|ui| {
            ui.label(p);
        });
    }
    ui.add_space(6.0);
    ui.label(RichText::new("Speed along the slot centre-line").strong());
    let Some(d) = &app.die else { return };
    let Some(prof) = d.1.profile(app.project.opening) else {
        ui.label("This opening is not a slot, so it has no centre-line profile. It still grows in 3D.");
        return;
    };
    let n = prof.v.len();
    let c = prof.v[n / 2].max(1e-12);
    let line: PlotPoints = prof.s.iter().zip(&prof.v).map(|(s, v)| [*s, *v / c]).collect();
    let h = prof.thickness[n / 2];
    let onset = 1.0 + 2.51 * (h / (0.5 * prof.length)).powi(2);
    let ghost: Option<PlotPoints> = app.ghost.as_ref().and_then(|g| g.profile.as_ref()).map(|p| {
        let gc = p[p.len() / 2][1].max(1e-12);
        p.iter().map(|q| [q[0], q[1] / gc]).collect()
    });
    let accent = ui.visuals().selection.stroke.color;
    Plot::new("profile")
        .height(ui.available_height().max(120.0))
        .legend(Legend::default())
        .x_axis_label("position along slot (mm)")
        .y_axis_label("speed ÷ centre speed")
        .allow_scroll(false)
        .show(ui, |pu| {
            pu.line(Line::new("speed", line).color(accent).width(2.0));
            if let Some(gp) = ghost {
                pu.line(Line::new("pinned", gp).color(Color32::GRAY).style(egui_plot::LineStyle::dashed_dense()));
            }
            if !prof.closed {
                pu.hline(HLine::new("ruffle onset", onset).color(Color32::from_rgb(0xD9, 0x6C, 0x2E)).style(egui_plot::LineStyle::dotted_dense()));
            }
        });
}

fn section(ui: &mut egui::Ui, title: &str, open: bool, add: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).strong()).default_open(open).show(ui, add);
}

fn slider(ui: &mut egui::Ui, v: &mut f64, range: std::ops::RangeInclusive<f64>, label: &str, unit: &str, help: &str) {
    ui.add(egui::Slider::new(v, range).text(label).suffix(unit).max_decimals(3)).on_hover_text(help);
}

fn log_slider(ui: &mut egui::Ui, v: &mut f64, range: std::ops::RangeInclusive<f64>, label: &str, unit: &str, help: &str) {
    ui.add(egui::Slider::new(v, range).logarithmic(true).text(label).suffix(unit).max_decimals(3)).on_hover_text(help);
}

fn profile_editor(ui: &mut egui::Ui, id: &str, p: &mut Profile, unit: &str) {
    egui::Grid::new(id).num_columns(3).spacing([6.0, 2.0]).show(ui, |ui| {
        ui.label(RichText::new("along slot").small().weak());
        ui.label(RichText::new(unit).small().weak());
        ui.end_row();
        let mut remove = None;
        for (k, pt) in p.points.iter_mut().enumerate() {
            ui.add(egui::DragValue::new(&mut pt[0]).range(0.0..=1.0).speed(0.002).max_decimals(3));
            ui.add(egui::DragValue::new(&mut pt[1]).range(0.01..=50.0).speed(0.005).max_decimals(3));
            if ui.small_button("×").clicked() {
                remove = Some(k);
            }
            ui.end_row();
        }
        if let Some(k) = remove
            && p.points.len() > 1 {
                p.points.remove(k);
            }
    });
    if ui.small_button("+ point").clicked() {
        let last = *p.points.last().unwrap_or(&[1.0, 1.0]);
        p.points.push([1.0, last[1]]);
    }
    p.points.sort_by(|a, b| a[0].total_cmp(&b[0]));
}

pub fn params_panel(app: &mut App, ui: &mut egui::Ui) {
    readouts(app, ui);
    ui.separator();
    let pr = &mut app.project;
    section(ui, "Die", false, |ui| {
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.text_edit_singleline(&mut pr.die.name);
        });
        let slots: Vec<usize> = pr.die.openings.iter().enumerate().filter(|(_, o)| o.as_slot().is_some()).map(|(k, _)| k).collect();
        if !slots.is_empty() {
            egui::ComboBox::from_label("Grows from")
                .selected_text(format!("Opening {}", pr.opening + 1))
                .show_ui(ui, |ui| {
                    for k in slots {
                        ui.selectable_value(&mut pr.opening, k, format!("Opening {}", k + 1));
                    }
                });
        }
        slider(ui, &mut pr.die.plate_diameter, 20.0..=120.0, "Plate diameter", " mm", "Die plate size, as fitted to the extruder");
        let k = pr.opening.min(pr.die.openings.len().saturating_sub(1));
        if let Some(op) = pr.die.openings.get_mut(k) {
            ui.label(RichText::new(format!("Opening {}", k + 1)).strong());
            let wall = match op {
                Opening::Slot(s) => &mut s.wall,
                Opening::Polygon(p) => &mut p.wall,
            };
            egui::ComboBox::from_label("Wall finish").selected_text(wall.label()).show_ui(ui, |ui| {
                for f in WallFinish::ALL {
                    ui.selectable_value(wall, f, f.label());
                }
            }).response.on_hover_text("Scales wall friction. The multipliers are placeholders until calibrated.");
            match op {
                Opening::Slot(s) => {
                    ui.checkbox(&mut s.closed, "Closed (tube)");
                    ui.label("Gap along the slot");
                    profile_editor(ui, "gap", &mut s.gap, "gap mm");
                    ui.label("Land depth along the slot");
                    profile_editor(ui, "land", &mut s.land, "land mm");
                }
                Opening::Polygon(p) => {
                    slider(ui, &mut p.land, 0.5..=30.0, "Land depth", " mm", "Length of the channel through the die");
                }
            }
        }
    });
    section(ui, "Dough and wall", true, |ui| {
        let r = &mut pr.rheology;
        slider(ui, &mut r.n, 0.15..=1.0, "Shear thinning n", "", "1 = Newtonian. Pasta dough 0.29 to 0.45 in the review.");
        log_slider(ui, &mut r.k, 1.0..=500.0, "Consistency K", " kPa·s^n", "Pasta dough 33 to 119 kPa·s^n in the review. Speed ratios do not depend on it without slip or yield stress.");
        log_slider(ui, &mut r.tau_y, 0.001..=10.0, "Yield stress", " kPa", "About 0.3 kPa for dough");
        ui.checkbox(&mut r.no_slip, "Dough sticks to the wall (no slip)");
        ui.add_enabled_ui(!r.no_slip, |ui| {
            log_slider(ui, &mut r.beta, 0.1..=1000.0, "Wall friction β", "", "Wall stress = β × slip speed^m (kPa, mm/s). Uncalibrated: the wall law matters more than anything else.");
            slider(ui, &mut r.m, 0.1..=1.0, "Slip exponent m", "", "Power of the slip law");
        });
    });
    section(ui, "Extruder", true, |ui| {
        let mut is_output = matches!(pr.drive, DriveSpec::Output { .. });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut is_output, true, "Output");
            ui.selectable_value(&mut is_output, false, "Pressure");
        });
        match (&mut pr.drive, is_output) {
            (DriveSpec::Output { grams_per_second }, true) => slider(ui, grams_per_second, 0.05..=5.0, "Output", " g/s", "Dough pushed through the die per second"),
            (DriveSpec::Pressure { kpa }, false) => log_slider(ui, kpa, 10.0..=100_000.0, "Pressure drop", " kPa", "Across the die"),
            (_, true) => pr.drive = DriveSpec::Output { grams_per_second: 0.5 },
            (_, false) => pr.drive = DriveSpec::Pressure { kpa: 5000.0 },
        }
    });
    section(ui, "Pasta", true, |ui| {
        slider(ui, &mut pr.swell, 0.8..=2.0, "Die swell", " x", "The section grows by this factor at the exit; speed falls by its square");
        let s = &mut pr.solid;
        log_slider(ui, &mut s.young_kpa, 1.0..=2000.0, "Stiffness E", " kPa", "Elastic modulus of fresh dough");
        slider(ui, &mut s.poisson, 0.0..=0.49, "Poisson's ratio", "", "Near 0.5 the dough is almost incompressible");
        slider(ui, &mut s.density, 0.8..=2.0, "Density", " g/cm³", "");
        egui::ComboBox::from_label("Extrusion").selected_text(match s.orientation {
            Orientation::Down => "Downward",
            Orientation::Horizontal => "Horizontal, onto a surface",
        }).show_ui(ui, |ui| {
            ui.selectable_value(&mut s.orientation, Orientation::Down, "Downward");
            ui.selectable_value(&mut s.orientation, Orientation::Horizontal, "Horizontal, onto a surface");
        });
        if s.orientation == Orientation::Horizontal {
            slider(ui, &mut s.support_drop, 0.0..=50.0, "Drop to surface", " mm", "From the bottom of the opening down to the support");
        }
        slider(ui, &mut s.cut_length, 5.0..=200.0, "Cut length", " mm", "The piece is cut at this length");
        ui.checkbox(&mut s.relax_after_cut, "Also relax the cut piece");
        ui.checkbox(&mut s.gap_average, "Average speed across the gap").on_hover_text(
            "Slots only. Past the exit the slow dough at the walls is pulled along by the core, so each point \
             takes the gap-mean speed (keeping its tilt from wall to wall). Needed when the dough sticks to the walls.",
        );
        egui::CollapsingHeader::new("Resolution").default_open(false).show(ui, |ui| {
            let mut c = s.cells_across as f64;
            slider(ui, &mut c, 1.0..=4.0, "Elements across", "", "Quadratic elements across the thinnest part of the opening; one is enough to bend correctly");
            s.cells_across = c.round() as usize;
            slider(ui, &mut s.station_aspect, 0.3..=3.0, "Element length", " × gap", "Element size along a slot, relative to its narrowest gap");
            slider(ui, &mut s.row_aspect, 0.3..=3.0, "Row spacing", " × element", "Spacing of rows along the pasta, relative to the section's element size");
            let mut sub = s.substeps as f64;
            slider(ui, &mut sub, 1.0..=8.0, "Steps per row", "", "");
            s.substeps = sub.round() as usize;
            ui.horizontal(|ui| {
                ui.label("Seed");
                ui.add(egui::DragValue::new(&mut s.seed));
            });
        });
    });
    section(ui, "Estimates", false, |ui| {
        slider(ui, &mut pr.ruffle_c, 0.5..=20.0, "Wavelength prefactor C", "", "λ ≈ C h^0.4 l^0.6. Fit to your pasta.");
    });
    section(ui, "View", false, |ui| {
        let v = &mut app.view;
        egui::ComboBox::from_label("Colour by").selected_text(app.color_mode.label()).show_ui(ui, |ui| {
            for m in ColorMode::ALL {
                ui.selectable_value(&mut app.color_mode, m, m.label());
            }
        });
        egui::ComboBox::from_label("Material").selected_text(format!("{:?}", v.material)).show_ui(ui, |ui| {
            for m in [Material::Clay, Material::Wax, Material::Flat] {
                ui.selectable_value(&mut v.material, m, format!("{m:?}"));
            }
        });
        ui.checkbox(&mut v.wireframe, "Wireframe (W)");
        ui.checkbox(&mut v.ground, "Ground and shadow");
        ui.checkbox(&mut v.show_colorbar, "Colour bar");
        let mut slice = v.slice.is_some();
        if ui.checkbox(&mut slice, "Slice plane").changed() {
            v.slice = if slice { Some(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])) } else { None };
        }
        if let Some((p, _)) = &mut v.slice {
            ui.add(egui::Slider::new(&mut p[0], -40.0..=40.0).text("slice x").suffix(" mm"));
        }
    });
}

fn card(ui: &mut egui::Ui, label: &str, value: String, level: u8, help: &str) {
    let (fill, stroke) = match level {
        2 => (Color32::from_rgba_unmultiplied(0xD9, 0x3B, 0x2E, 28), Color32::from_rgb(0xD9, 0x3B, 0x2E)),
        1 => (Color32::from_rgba_unmultiplied(0xE0, 0x8A, 0x1E, 28), Color32::from_rgb(0xE0, 0x8A, 0x1E)),
        _ => (Color32::TRANSPARENT, ui.visuals().widgets.noninteractive.bg_stroke.color),
    };
    egui::Frame::new().fill(fill).stroke(egui::Stroke::new(1.0, stroke)).corner_radius(6.0).inner_margin(egui::Margin::symmetric(8, 4)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(label);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(RichText::new(value).strong()));
        });
    })
    .response
    .on_hover_text(help);
}

fn readouts(app: &mut App, ui: &mut egui::Ui) {
    ui.label(RichText::new("Readouts").strong());
    let Some(d) = &app.die else {
        ui.label("Solving…");
        return;
    };
    let f = &d.1.flow;
    card(ui, "Pressure drop", format!("{:.0} kPa", f.dp_kpa), 0, "Across the die at the chosen output");
    if f.q_opening.len() > 1 {
        let shares: Vec<String> = f.q_opening.iter().map(|q| format!("{:.0}%", 100.0 * q / f.q_total)).collect();
        card(ui, "Flow share", shares.join(" / "), 0, "Share of the flow through each opening: weigh the strands to check");
    }
    if let Some(r) = d.1.readout(app.project.opening) {
        if let Some(x) = r.edge_centre_ratio {
            card(ui, "Edge ÷ centre speed", format!("{x:.3}"), 0, "Peak speed in the outer quarter of each side over the centre speed");
        }
        if let Some(m) = r.ruffle_margin {
            card(ui, "Ruffle margin", format!("{m:.1}×"), if m > 4.0 { 2 } else if m > 1.0 { 1 } else { 0 }, "Edge excess over the buckling threshold 2.51 (H/W)². Above 1 the edge should ruffle, above 4 strongly.");
        }
        if let (Some(l), Some(a)) = (r.wavelength, r.amplitude) {
            card(ui, "Ruffle estimate", format!("λ {l:.1} mm, A {a:.2} mm"), 0, "Scaling-law estimate; C is uncalibrated");
        }
        if let Some((rad, r2)) = r.curl_radius
            && r2 > 0.8 {
                card(ui, "Curl radius", format!("{rad:.0} mm"), 0, "In-plane curl v ÷ |v′| from a straight-line fit");
            }
        if let Some(b) = r.bend_radius {
            card(ui, "Bending radius", format!("{b:.0} mm"), 0, "From the speed difference between the two faces");
        }
        if let Some(t) = &r.tube {
            card(ui, "Tube bend radius", format!("{:.0} mm", 1.0 / t.curvature.max(1e-12)), 0, "From the speed variation around the tube");
        }
    }
    if let Some(pl) = f.plug_fraction.get(app.project.opening)
        && *pl > 0.01 {
            card(ui, "Plug flow", format!("{:.0}%", 100.0 * pl), 0, "Area moving as an unsheared block (yield stress)");
        }
    if let Some(s) = &app.piece {
        let r = &s.1;
        let pos = if app.shape_view == ShapeView::Relaxed { r.relaxed.as_ref().unwrap_or(&r.hanging) } else { &r.hanging };
        let m = measure(r, pos);
        ui.add_space(4.0);
        ui.label(RichText::new("Simulated pasta").strong());
        card(ui, "Size", format!("{:.1} × {:.1} × {:.1} mm", m.length, m.width, m.depth), 0, "Extents along the piece's principal axes");
        if m.twist_turns.abs() >= 0.05 {
            card(ui, "Twist", format!("{:+.2} turns", m.twist_turns), 0, &format!("Pitch {} (positive is counter-clockwise looking along the extrusion)", m.pitch.map(|p| format!("{p:.1} mm")).unwrap_or_else(|| "-".into())));
        }
        if let Some(b) = m.bend_radius {
            card(ui, "Bend radius", format!("{b:.0} mm"), 0, "Circle through the start, middle and end of the centre-line");
        }
        card(ui, "Model", format!("{} quadratic tets", r.stats.tets), 0, &format!("{} nodes, {:.0} s", r.stats.nodes, r.stats.seconds));
    }
    if let Some(info) = &app.pick_info {
        ui.add_space(4.0);
        ui.label(RichText::new("Picked point").strong());
        ui.label(info.as_str());
    }
}

pub fn view_overlay(app: &mut App, ui: &mut egui::Ui) {
    let rect = ui.max_rect();
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(egui::Rect::from_min_size(rect.min + egui::vec2(10.0, 10.0), egui::vec2(260.0, 30.0))));
    child.horizontal(|ui| {
        egui::ComboBox::from_id_salt("colour").selected_text(app.color_mode.label()).width(130.0).show_ui(ui, |ui| {
            for m in ColorMode::ALL {
                ui.selectable_value(&mut app.color_mode, m, m.label());
            }
        });
        for (label, cmd) in [("Front", Command::ViewFront), ("Side", Command::ViewSide), ("Top", Command::ViewTop), ("Iso", Command::ViewIso)] {
            if ui.small_button(label).clicked() {
                app.run_command(cmd, None);
            }
        }
    });
    if app.piece.is_none() && app.grow_progress.is_none() {
        let msg = if app.die.is_none() { "Solving the die…" } else { "Press Grow (G) to extrude the pasta" };
        ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, msg, egui::FontId::proportional(16.0), ui.visuals().weak_text_color());
    }
}

pub fn palette(app: &mut App, ctx: &egui::Context) -> Option<Command> {
    let mut chosen = None;
    let mut open = app.palette_open;
    egui::Window::new("Commands").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_TOP, [0.0, 80.0]).open(&mut open).show(ctx, |ui| {
        let r = ui.add(egui::TextEdit::singleline(&mut app.palette_query).hint_text("Type a command…").desired_width(360.0));
        r.request_focus();
        let q = app.palette_query.to_lowercase();
        let matches: Vec<Command> = Command::ALL.iter().copied().filter(|c| c.label().to_lowercase().contains(&q)).collect();
        if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            chosen = matches.first().copied();
        }
        for c in matches {
            ui.horizontal(|ui| {
                if ui.selectable_label(false, c.label()).clicked() {
                    chosen = Some(c);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(RichText::new(c.shortcut()).weak()));
            });
        }
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            app.palette_open = false;
        }
    });
    if chosen.is_some() {
        open = false;
    }
    app.palette_open = open && app.palette_open;
    chosen
}
