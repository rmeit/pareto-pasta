//! Triangle meshes of die openings.
//!
//! Slots get a structured mesh: stations along the centre-line by layers
//! across the gap. That keeps triangles well shaped in thin slots and makes
//! the reduction to a centre-line profile exact. Polygon openings are
//! triangulated with constrained Delaunay refinement.

use crate::geometry::Centerline;
use crate::{Die, Opening, PolygonOpening, Slot};
use spade::{AngleLimit, ConstrainedDelaunayTriangulation, Point2, RefinementParameters, Triangulation};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshOptions {
    /// Triangle layers across the narrowest gap.
    pub cells_across: usize,
    /// Station spacing along a slot as a multiple of the layer thickness at
    /// the narrowest gap.
    pub aspect: f64,
    /// Upper limit on stations per slot.
    pub max_stations: usize,
}

impl MeshOptions {
    /// Coarse mesh for live dragging.
    pub fn draft() -> Self {
        Self { cells_across: 4, aspect: 3.0, max_stations: 400 }
    }

    /// Mesh used at rest.
    pub fn fine() -> Self {
        Self { cells_across: 16, aspect: 3.0, max_stations: 1600 }
    }
}

impl Default for MeshOptions {
    fn default() -> Self {
        Self::fine()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryEdge {
    pub a: u32,
    pub b: u32,
    pub opening: u32,
    /// Multiplier on the wall-slip coefficient from the wall finish.
    pub friction_scale: f64,
}

/// The structured grid of one slot.
#[derive(Clone, Debug)]
pub struct SlotGrid {
    pub node_offset: usize,
    /// Number of distinct stations (open: segments + 1; closed: segments).
    pub n_stations: usize,
    /// Number of node layers across the gap.
    pub n_layers: usize,
    pub closed: bool,
    pub length: f64,
    pub s: Vec<f64>,
    pub centre: Vec<[f64; 2]>,
    pub tangent: Vec<[f64; 2]>,
    pub normal: Vec<[f64; 2]>,
    pub curvature: Vec<f64>,
    pub gap: Vec<f64>,
    pub land: Vec<f64>,
}

impl SlotGrid {
    /// Node at station i and layer j (j = 0 is the -normal wall).
    pub fn node(&self, i: usize, j: usize) -> usize {
        self.node_offset + i * self.n_layers + j
    }

    /// Coordinate across the gap of layer j, in [-1, 1].
    pub fn layer_t(&self, j: usize) -> f64 {
        -1.0 + 2.0 * j as f64 / (self.n_layers - 1) as f64
    }
}

#[derive(Clone, Debug)]
pub struct OpeningMesh {
    pub nodes: std::ops::Range<usize>,
    pub tris: std::ops::Range<usize>,
    pub slot: Option<SlotGrid>,
    /// Open area (mm^2).
    pub area: f64,
}

/// Triangle mesh of all openings of a die.
#[derive(Clone, Debug, Default)]
pub struct TriMesh2 {
    pub nodes: Vec<[f64; 2]>,
    pub tris: Vec<[u32; 3]>,
    pub tri_opening: Vec<u32>,
    /// Land depth of each triangle (mm).
    pub tri_land: Vec<f64>,
    pub boundary: Vec<BoundaryEdge>,
    pub openings: Vec<OpeningMesh>,
}

impl TriMesh2 {
    pub fn tri_area(&self, t: usize) -> f64 {
        let [a, b, c] = self.tris[t].map(|i| self.nodes[i as usize]);
        0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    }

    /// Smallest triangle area divided by the largest, a quick quality check.
    pub fn min_area(&self) -> f64 {
        (0..self.tris.len()).map(|t| self.tri_area(t)).fold(f64::INFINITY, f64::min)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MeshError {
    /// The gap is wider than twice the radius of curvature somewhere, so the
    /// slot walls fold over.
    SlotFolds { opening: usize },
    TooFewPoints { opening: usize },
    Triangulation { opening: usize, message: String },
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::SlotFolds { opening } => {
                write!(f, "opening {}: the slot is too wide for how sharply it bends", opening + 1)
            }
            MeshError::TooFewPoints { opening } => write!(f, "opening {}: needs more points", opening + 1),
            MeshError::Triangulation { opening, message } => {
                write!(f, "opening {}: could not triangulate ({message})", opening + 1)
            }
        }
    }
}

impl std::error::Error for MeshError {}

pub fn mesh_die(die: &Die, opts: &MeshOptions) -> Result<TriMesh2, MeshError> {
    let mut m = TriMesh2::default();
    for (k, o) in die.openings.iter().enumerate() {
        match o {
            Opening::Slot(s) => mesh_slot(&mut m, k, s, opts)?,
            Opening::Polygon(p) => mesh_polygon(&mut m, k, p, opts)?,
        }
    }
    Ok(m)
}

fn mesh_slot(m: &mut TriMesh2, k: usize, slot: &Slot, opts: &MeshOptions) -> Result<(), MeshError> {
    if slot.centerline.len() < 2 || (slot.closed && slot.centerline.len() < 3) {
        return Err(MeshError::TooFewPoints { opening: k });
    }
    let n_layers = opts.cells_across.max(1) + 1;
    // Station count from the arc length and the narrowest gap.
    let probe = Centerline::new(&slot.centerline, slot.closed, 256);
    let min_gap = slot.gap.min().max(1e-6);
    let ds = min_gap / opts.cells_across.max(1) as f64 * opts.aspect;
    let segs = ((probe.length() / ds).ceil() as usize).clamp(8, opts.max_stations);
    let c = Centerline::new(&slot.centerline, slot.closed, segs);
    let n_st = c.len();

    let node_offset = m.nodes.len();
    let tri_start = m.tris.len();
    let mut grid = SlotGrid {
        node_offset,
        n_stations: n_st,
        n_layers,
        closed: slot.closed,
        length: c.length(),
        s: Vec::with_capacity(n_st),
        centre: Vec::with_capacity(n_st),
        tangent: Vec::with_capacity(n_st),
        normal: Vec::with_capacity(n_st),
        curvature: Vec::with_capacity(n_st),
        gap: Vec::with_capacity(n_st),
        land: Vec::with_capacity(n_st),
    };
    for i in 0..n_st {
        let u = c.u(i);
        let h = slot.gap.eval(u).max(1e-6);
        let kappa = c.curvature(i);
        if 0.5 * h * kappa.abs() >= 0.95 {
            return Err(MeshError::SlotFolds { opening: k });
        }
        grid.s.push(c.s(i));
        grid.centre.push(c.point(i));
        grid.tangent.push(c.tangent(i));
        grid.normal.push(c.normal(i));
        grid.curvature.push(kappa);
        grid.gap.push(h);
        grid.land.push(slot.land.eval(u).max(1e-6));
    }
    for i in 0..n_st {
        let p = grid.centre[i];
        let nrm = grid.normal[i];
        for j in 0..n_layers {
            let t = grid.layer_t(j) * 0.5 * grid.gap[i];
            m.nodes.push([p[0] + t * nrm[0], p[1] + t * nrm[1]]);
        }
    }
    let segs_along = if slot.closed { n_st } else { n_st - 1 };
    for i in 0..segs_along {
        let i1 = (i + 1) % n_st;
        let land = 0.5 * (grid.land[i] + grid.land[i1]);
        for j in 0..n_layers - 1 {
            let a = grid.node(i, j) as u32;
            let b = grid.node(i1, j) as u32;
            let cc = grid.node(i1, j + 1) as u32;
            let d = grid.node(i, j + 1) as u32;
            // Alternate the diagonal so the mesh has no preferred direction.
            let tris = if (i + j) % 2 == 0 { [[a, b, cc], [a, cc, d]] } else { [[a, b, d], [b, cc, d]] };
            for t in tris {
                m.tris.push(t);
                m.tri_opening.push(k as u32);
                m.tri_land.push(land);
            }
        }
    }
    let fs = slot.wall.friction_scale();
    let mut push_edge = |a: usize, b: usize| {
        m.boundary.push(BoundaryEdge { a: a as u32, b: b as u32, opening: k as u32, friction_scale: fs });
    };
    for i in 0..segs_along {
        let i1 = (i + 1) % n_st;
        push_edge(grid.node(i, 0), grid.node(i1, 0));
        push_edge(grid.node(i, n_layers - 1), grid.node(i1, n_layers - 1));
    }
    if !slot.closed {
        for j in 0..n_layers - 1 {
            push_edge(grid.node(0, j), grid.node(0, j + 1));
            push_edge(grid.node(n_st - 1, j), grid.node(n_st - 1, j + 1));
        }
    }
    let tris = tri_start..m.tris.len();
    // Orientation check: every triangle must have the same sign.
    let sign = m.tri_area(tri_start).signum();
    let mut area = 0.0;
    for t in tris.clone() {
        let a = m.tri_area(t);
        if a.signum() != sign || a == 0.0 {
            return Err(MeshError::SlotFolds { opening: k });
        }
        area += a.abs();
    }
    if sign < 0.0 {
        for t in tris.clone() {
            m.tris[t].swap(1, 2);
        }
    }
    m.openings.push(OpeningMesh { nodes: node_offset..m.nodes.len(), tris, slot: Some(grid), area });
    Ok(())
}

fn polygon_area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    (0..n).map(|i| p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1]).sum::<f64>() * 0.5
}

fn mesh_polygon(m: &mut TriMesh2, k: usize, poly: &PolygonOpening, opts: &MeshOptions) -> Result<(), MeshError> {
    if poly.outer.len() < 3 {
        return Err(MeshError::TooFewPoints { opening: k });
    }
    let perim: f64 = {
        let n = poly.outer.len();
        (0..n)
            .map(|i| {
                let a = poly.outer[i];
                let b = poly.outer[(i + 1) % n];
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
            })
            .sum()
    };
    let area_outer = polygon_area(&poly.outer).abs();
    let holes_area: f64 = poly.holes.iter().map(|h| polygon_area(h).abs()).sum();
    // Twice area over perimeter is the half-width of a thin strip and the
    // radius of a disc: a rough feature size.
    let feature = 2.0 * (area_outer - holes_area) / perim;
    let h = 2.0 * feature / opts.cells_across.max(1) as f64;
    let err = |e: spade::InsertionError| MeshError::Triangulation { opening: k, message: format!("{e:?}") };

    let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> = ConstrainedDelaunayTriangulation::new();
    let densify = |ring: &[[f64; 2]]| -> Vec<Point2<f64>> {
        let n = ring.len();
        let mut out = Vec::new();
        // An outline drawn much finer than the mesh would force tiny
        // triangles: resample it evenly at the mesh size instead.
        let seg: Vec<f64> = (0..n)
            .map(|i| {
                let (a, b) = (ring[i], ring[(i + 1) % n]);
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
            })
            .collect();
        let total: f64 = seg.iter().sum();
        if total / (n as f64) < 0.5 * h {
            let m = ((total / h).round() as usize).max(3);
            let (mut i, mut acc) = (0, 0.0);
            for q in 0..m {
                let target = total * q as f64 / m as f64;
                while acc + seg[i] < target && i + 1 < n {
                    acc += seg[i];
                    i += 1;
                }
                let t = ((target - acc) / seg[i].max(1e-300)).clamp(0.0, 1.0);
                let (a, b) = (ring[i], ring[(i + 1) % n]);
                out.push(Point2::new(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])));
            }
            return out;
        }
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            let l = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
            let pieces = ((l / h).ceil() as usize).max(1);
            for q in 0..pieces {
                let t = q as f64 / pieces as f64;
                out.push(Point2::new(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])));
            }
        }
        out
    };
    cdt.add_constraint_edges(densify(&poly.outer), true).map_err(err)?;
    for hole in &poly.holes {
        cdt.add_constraint_edges(densify(hole), true).map_err(err)?;
    }
    let result = cdt.refine(
        RefinementParameters::<f64>::new()
            .exclude_outer_faces(true)
            .with_angle_limit(AngleLimit::from_deg(25.0))
            .with_max_allowed_area(h * h * 0.6)
            .with_max_additional_vertices(200_000),
    );
    let excluded: std::collections::HashSet<_> = result.excluded_faces.iter().copied().collect();

    let node_offset = m.nodes.len();
    let tri_start = m.tris.len();
    let mut map = vec![u32::MAX; cdt.num_vertices()];
    let mut edge_count: std::collections::HashMap<(u32, u32), i32> = Default::default();
    let mut area = 0.0;
    for face in cdt.inner_faces() {
        if excluded.contains(&face.fix()) {
            continue;
        }
        let vs = face.vertices();
        let mut ids = [0u32; 3];
        for (q, v) in vs.iter().enumerate() {
            let vi = v.fix().index();
            if map[vi] == u32::MAX {
                let p = v.position();
                map[vi] = m.nodes.len() as u32;
                m.nodes.push([p.x, p.y]);
            }
            ids[q] = map[vi];
        }
        m.tris.push(ids);
        m.tri_opening.push(k as u32);
        let [pa, pb, pc] = ids.map(|i| m.nodes[i as usize]);
        m.tri_land.push(poly.land_at([(pa[0] + pb[0] + pc[0]) / 3.0, (pa[1] + pb[1] + pc[1]) / 3.0]).max(1e-6));
        let t = m.tris.len() - 1;
        let a = m.tri_area(t);
        if a < 0.0 {
            m.tris[t].swap(1, 2);
        }
        area += a.abs();
        for e in 0..3 {
            let (p, q) = (ids[e], ids[(e + 1) % 3]);
            *edge_count.entry((p.min(q), p.max(q))).or_insert(0) += 1;
        }
    }
    if m.tris.len() == tri_start {
        return Err(MeshError::Triangulation { opening: k, message: "no interior triangles".into() });
    }
    let fs = poly.wall.friction_scale();
    let mut bedges: Vec<_> = edge_count.into_iter().filter(|&(_, c)| c == 1).map(|(e, _)| e).collect();
    bedges.sort_unstable();
    for (a, b) in bedges {
        m.boundary.push(BoundaryEdge { a, b, opening: k as u32, friction_scale: fs });
    }
    m.openings.push(OpeningMesh { nodes: node_offset..m.nodes.len(), tris: tri_start..m.tris.len(), slot: None, area });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Profile, WallFinish};

    fn ribbon(len: f64, gap: f64) -> Die {
        Die {
            name: "t".into(),
            openings: vec![Opening::Slot(Slot {
                centerline: vec![[-len / 2.0, 0.0], [len / 2.0, 0.0]],
                closed: false,
                gap: Profile::constant(gap),
                land: Profile::constant(3.0),
                wall: WallFinish::Milled,
            })],
            thickness: 5.0,
            plate_diameter: 60.0,
        }
    }

    #[test]
    fn slot_area_matches() {
        let d = ribbon(20.0, 1.0);
        let m = mesh_die(&d, &MeshOptions::fine()).unwrap();
        let a: f64 = m.openings[0].area;
        assert!((a - 20.0).abs() < 1e-9, "{a}");
        // Boundary length = perimeter.
        let per: f64 = m
            .boundary
            .iter()
            .map(|e| {
                let a = m.nodes[e.a as usize];
                let b = m.nodes[e.b as usize];
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
            })
            .sum();
        assert!((per - 42.0).abs() < 1e-9, "{per}");
    }

    #[test]
    fn tube_has_no_end_caps() {
        let d = crate::preset("tube").unwrap().die;
        let m = mesh_die(&d, &MeshOptions::draft()).unwrap();
        let g = m.openings[0].slot.as_ref().unwrap();
        assert!(g.closed);
        assert_eq!(m.boundary.len(), 2 * g.n_stations);
        let r = 4.0;
        let expected = std::f64::consts::TAU * r * 1.0;
        assert!((m.openings[0].area - expected).abs() / expected < 0.01);
    }

    #[test]
    fn disc_polygon() {
        let r = 1.0;
        let outer: Vec<_> = (0..64)
            .map(|k| {
                let a = k as f64 * std::f64::consts::TAU / 64.0;
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        let d = Die {
            name: "disc".into(),
            openings: vec![Opening::Polygon(PolygonOpening {
                land_radial: None,
                outer,
                holes: vec![],
                land: 3.0,
                wall: WallFinish::Milled,
            })],
            thickness: 5.0,
            plate_diameter: 60.0,
        };
        let m = mesh_die(&d, &MeshOptions::fine()).unwrap();
        let exact = std::f64::consts::PI * (64.0 / std::f64::consts::TAU * (std::f64::consts::TAU / 64.0).sin());
        assert!((m.openings[0].area - exact).abs() < 1e-6, "{}", m.openings[0].area);
        assert!(m.tris.len() > 50);
        assert!(m.min_area() > 0.0);
    }
}
