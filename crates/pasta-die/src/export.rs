//! Die drawings for machining: SVG for viewing and DXF for CAM.
//!
//! Each opening's outline goes on its own DXF layer, named with its land
//! depth range, because a 2D drawing cannot carry a depth that varies.

use crate::{Die, Opening};
use std::fmt::Write;

fn land_range(o: &Opening) -> (f64, f64) {
    match o {
        Opening::Slot(s) => (s.land.min(), s.land.max()),
        Opening::Polygon(p) => (p.land, p.land),
    }
}

fn land_label(lo: f64, hi: f64) -> String {
    if (hi - lo).abs() < 1e-9 { format!("{lo:.2}mm") } else { format!("{lo:.2}-{hi:.2}mm") }
}

/// SVG drawing in millimetres, y up as in the die coordinates.
pub fn to_svg(die: &Die) -> String {
    let r = die.plate_diameter * 0.5;
    let mut s = String::new();
    let _ = writeln!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{d}mm" height="{d}mm" viewBox="{x} {x} {d} {d}">"#,
        d = 2.0 * r,
        x = -r
    );
    let _ = writeln!(s, r#"<title>{}</title>"#, xml_escape(&die.name));
    let _ = writeln!(s, r#"<g transform="scale(1,-1)" fill="none" stroke="black" stroke-width="0.05">"#);
    let _ = writeln!(s, r#"<circle cx="0" cy="0" r="{r}" stroke-dasharray="0.5 0.5"/>"#);
    for (k, poly) in die.outlines(512) {
        let (lo, hi) = land_range(&die.openings[k]);
        let _ = write!(s, r#"<path data-opening="{}" data-land="{}" d=""#, k + 1, land_label(lo, hi));
        for (i, p) in poly.iter().enumerate() {
            let _ = write!(s, "{}{:.4},{:.4} ", if i == 0 { "M" } else { "L" }, p[0], p[1]);
        }
        let _ = writeln!(s, r#"Z"/>"#);
    }
    let _ = writeln!(s, "</g>");
    for (k, o) in die.openings.iter().enumerate() {
        let (lo, hi) = land_range(o);
        let _ = writeln!(
            s,
            r#"<text x="{}" y="{}" font-size="1.5" font-family="sans-serif">Opening {}: land {}</text>"#,
            -r + 1.0,
            -r + 2.5 + 2.0 * k as f64,
            k + 1,
            land_label(lo, hi)
        );
    }
    let _ = writeln!(s, "</svg>");
    s
}

fn xml_escape(t: &str) -> String {
    t.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Minimal ASCII DXF (R12-compatible entities): closed polylines, one layer
/// per opening, plus the plate circle.
pub fn to_dxf(die: &Die) -> String {
    let mut s = String::new();
    let mut pair = |code: i32, value: &str| {
        let _ = writeln!(s, "{code}\n{value}");
    };
    pair(0, "SECTION");
    pair(2, "HEADER");
    pair(9, "$INSUNITS");
    pair(70, "4"); // millimetres
    pair(0, "ENDSEC");
    pair(0, "SECTION");
    pair(2, "ENTITIES");
    pair(0, "CIRCLE");
    pair(8, "PLATE");
    pair(10, "0.0");
    pair(20, "0.0");
    pair(30, "0.0");
    pair(40, &format!("{:.4}", die.plate_diameter * 0.5));
    for (k, poly) in die.outlines(512) {
        let (lo, hi) = land_range(&die.openings[k]);
        let layer = format!("OPENING_{}_LAND_{}", k + 1, land_label(lo, hi).replace('.', "_"));
        pair(0, "POLYLINE");
        pair(8, &layer);
        pair(66, "1");
        pair(70, "1"); // closed
        for p in poly {
            pair(0, "VERTEX");
            pair(8, &layer);
            pair(10, &format!("{:.4}", p[0]));
            pair(20, &format!("{:.4}", p[1]));
            pair(30, "0.0");
        }
        pair(0, "SEQEND");
        pair(8, &layer);
    }
    pair(0, "ENDSEC");
    pair(0, "EOF");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dxf_has_one_polyline_per_outline() {
        let d = crate::preset("tube").unwrap().die;
        let dxf = to_dxf(&d);
        assert_eq!(dxf.matches("\nPOLYLINE\n").count(), 2);
        assert!(dxf.ends_with("EOF\n"));
        let svg = to_svg(&d);
        assert_eq!(svg.matches("<path").count(), 2);
    }
}
