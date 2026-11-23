use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cross-section of the extrudate as quadratic triangles: corner points
/// first, then one midpoint per edge, with the exit speed at every point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Section {
    pub points: Vec<[f64; 2]>,
    pub n_corners: usize,
    /// Corner pairs (i < j); the midpoint of edge e is point `n_corners + e`.
    pub edges: Vec<[u32; 2]>,
    /// Triangles by corner index, counter-clockwise.
    pub tris: Vec<[u32; 3]>,
    /// Exit speed at every point (mm/s).
    pub speed: Vec<f64>,
    /// Edges on the outline (used by one triangle only).
    pub boundary: Vec<bool>,
}

impl Section {
    /// Builds the quadratic section from linear triangles (only the nodes
    /// they use are kept) and a speed field.
    pub fn from_triangles(nodes: &[[f64; 2]], tris: &[[u32; 3]], speed: impl Fn([f64; 2]) -> f64) -> Result<Self, String> {
        if tris.is_empty() {
            return Err("the opening has no triangles".into());
        }
        let mut map: HashMap<u32, u32> = HashMap::new();
        let mut points: Vec<[f64; 2]> = Vec::new();
        let mut local = Vec::with_capacity(tris.len());
        for t in tris {
            let mut l = [0u32; 3];
            for (q, &v) in t.iter().enumerate() {
                l[q] = *map.entry(v).or_insert_with(|| {
                    points.push(nodes[v as usize]);
                    (points.len() - 1) as u32
                });
            }
            let [a, b, c] = l.map(|i| points[i as usize]);
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
            if area == 0.0 {
                return Err("the section mesh has a flat triangle".into());
            }
            if area < 0.0 {
                l.swap(1, 2);
            }
            local.push(l);
        }
        let n_corners = points.len();
        let mut edge_map: HashMap<(u32, u32), u32> = HashMap::new();
        let mut edges: Vec<[u32; 2]> = Vec::new();
        let mut count: Vec<u32> = Vec::new();
        for t in &local {
            for k in 0..3 {
                let (a, b) = (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]));
                let e = *edge_map.entry((a, b)).or_insert_with(|| {
                    edges.push([a, b]);
                    count.push(0);
                    (edges.len() - 1) as u32
                });
                count[e as usize] += 1;
            }
        }
        for e in &edges {
            let (p, q) = (points[e[0] as usize], points[e[1] as usize]);
            points.push([0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])]);
        }
        let speed: Vec<f64> = points.iter().map(|&p| speed(p)).collect();
        if speed.iter().any(|v| !(*v > 0.0) || !v.is_finite()) {
            return Err("every point of the section needs a positive exit speed".into());
        }
        Ok(Self { points, n_corners, edges, tris: local, speed, boundary: count.iter().map(|&c| c == 1).collect() })
    }

    pub fn edge_map(&self) -> HashMap<(u32, u32), u32> {
        self.edges.iter().enumerate().map(|(e, p)| ((p[0], p[1]), e as u32)).collect()
    }

    pub fn mean_edge(&self) -> f64 {
        let s: f64 = self
            .edges
            .iter()
            .map(|e| {
                let (p, q) = (self.points[e[0] as usize], self.points[e[1] as usize]);
                ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
            })
            .sum();
        s / self.edges.len().max(1) as f64
    }

    pub fn area(&self) -> f64 {
        self.tris
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| self.points[i as usize]);
                0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            })
            .sum()
    }

    /// Area-weighted mean exit speed (exact for the linear interpolant).
    pub fn mean_speed(&self) -> f64 {
        let mut q = 0.0;
        for t in &self.tris {
            let [a, b, c] = t.map(|i| self.points[i as usize]);
            let area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));
            q += area * t.iter().map(|&i| self.speed[i as usize]).sum::<f64>() / 3.0;
        }
        q / self.area()
    }

    /// Die swell: the section grows by `s` about its centroid and slows by
    /// s² so the flow rate is unchanged.
    pub fn swelled(&self, s: f64) -> Section {
        if s == 1.0 {
            return self.clone();
        }
        let n = self.n_corners as f64;
        let cx = self.points[..self.n_corners].iter().map(|p| p[0]).sum::<f64>() / n;
        let cy = self.points[..self.n_corners].iter().map(|p| p[1]).sum::<f64>() / n;
        let mut out = self.clone();
        for p in &mut out.points {
            *p = [cx + s * (p[0] - cx), cy + s * (p[1] - cy)];
        }
        for v in &mut out.speed {
            *v /= s * s;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_section() {
        let nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let s = Section::from_triangles(&nodes, &[[0, 1, 2], [0, 3, 2]], |p| 1.0 + p[0]).unwrap();
        assert_eq!(s.n_corners, 4);
        assert_eq!(s.edges.len(), 5);
        assert_eq!(s.points.len(), 9);
        assert_eq!(s.boundary.iter().filter(|b| **b).count(), 4);
        assert!((s.area() - 1.0).abs() < 1e-12);
        assert!((s.mean_speed() - 1.5).abs() < 1e-12);
    }
}
