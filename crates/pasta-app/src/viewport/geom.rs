//! CPU-side geometry: vertex normals, expansion to GPU vertices, bounds, picking, colour maps.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Non-indexed GPU vertex (one per triangle corner; barycentrics come from `vertex_index % 3`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub scalar: f32,
}

/// Thick-line vertex: a segment (a, b) plus (t, side) selecting the corner of its screen quad.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct LineVertex {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub ts: [f32; 2],
}

/// Area-weighted vertex normals into `out` (reuses its allocation).
pub(crate) fn vertex_normals(positions: &[[f32; 3]], indices: &[u32], out: &mut Vec<[f32; 3]>) {
    out.clear();
    out.resize(positions.len(), [0.0; 3]);
    for tri in indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        let p0 = Vec3::from(positions[i0]);
        let p1 = Vec3::from(positions[i1]);
        let p2 = Vec3::from(positions[i2]);
        // |cross| = 2 * area, so summing the raw cross product weights by area.
        let n = (p1 - p0).cross(p2 - p0);
        for i in [i0, i1, i2] {
            let o = &mut out[i];
            o[0] += n.x;
            o[1] += n.y;
            o[2] += n.z;
        }
    }
    for n in out.iter_mut() {
        let v = Vec3::from(*n).normalize_or_zero();
        *n = v.to_array();
    }
}

/// Expand an indexed mesh to per-corner GPU vertices into `out` (reuses its allocation).
pub(crate) fn expand(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    scalars: Option<&[f32]>,
    indices: &[u32],
    out: &mut Vec<GpuVertex>,
) {
    out.clear();
    out.reserve(indices.len());
    let n = positions.len();
    for tri in indices.as_chunks::<3>().0 {
        if tri.iter().any(|&i| i as usize >= n) {
            // keep vertex_index % 3 aligned: emit a degenerate triangle
            out.extend([GpuVertex::zeroed(); 3]);
            continue;
        }
        for &i in tri {
            let i = i as usize;
            out.push(GpuVertex {
                pos: positions[i],
                normal: normals[i],
                scalar: scalars.and_then(|s| s.get(i).copied()).unwrap_or(0.0),
            });
        }
    }
}

/// Axis-aligned bounds of a point set, or `None` if empty / non-finite.
pub(crate) fn bounds(positions: &[[f32; 3]]) -> Option<(Vec3, Vec3)> {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for p in positions {
        let v = Vec3::from(*p);
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    (lo.x <= hi.x).then_some((lo, hi))
}

/// Min/max of finite scalar values.
pub(crate) fn scalar_range(values: &[f32]) -> Option<(f32, f32)> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    (lo <= hi).then_some((lo, hi))
}

/// Closest ray–triangle hit (Möller–Trumbore, two-sided). Returns (triangle, t, u, v).
/// `keep` filters hit points (used for the slice plane).
pub(crate) fn raycast(
    positions: &[[f32; 3]],
    indices: &[u32],
    origin: Vec3,
    dir: Vec3,
    keep: impl Fn(Vec3) -> bool,
) -> Option<(usize, f32, f32, f32)> {
    let mut best: Option<(usize, f32, f32, f32)> = None;
    let n = positions.len();
    for (ti, tri) in indices.as_chunks::<3>().0.iter().enumerate() {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }
        let p0 = Vec3::from(positions[i0]);
        let e1 = Vec3::from(positions[i1]) - p0;
        let e2 = Vec3::from(positions[i2]) - p0;
        let pv = dir.cross(e2);
        let det = e1.dot(pv);
        if det.abs() < 1e-12 {
            continue;
        }
        let inv = 1.0 / det;
        let tv = origin - p0;
        let u = tv.dot(pv) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let qv = tv.cross(e1);
        let v = dir.dot(qv) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = e2.dot(qv) * inv;
        if t <= 0.0 || best.is_some_and(|b| t >= b.1) {
            continue;
        }
        if !keep(origin + dir * t) {
            continue;
        }
        best = Some((ti, t, u, v));
    }
    best
}

fn poly_viridis(t: f32) -> [f32; 3] {
    const C: [[f32; 3]; 7] = [
        [0.277_727_33, 0.005_407_344_5, 0.334_099_8],
        [0.105_093_04, 1.404_613_5, 1.384_590_2],
        [-0.330_861_83, 0.214_847_56, 0.095_095_16],
        [-4.634_230_6, -5.799_101, -19.332_441],
        [6.228_27, 14.179_933, 56.690_55],
        [4.776_385, -13.745_145, -65.353_03],
        [-5.435_456, 4.645_852_6, 26.312_435],
    ];
    let mut out = [0.0; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let mut acc = C[6][k];
        for c in C[..6].iter().rev() {
            acc = acc * t + c[k];
        }
        *o = acc;
    }
    out
}

fn poly_turbo(x: f32) -> [f32; 3] {
    let r4 = [0.135_721_38, 4.615_392_6, -42.660_324, 132.131_08];
    let g4 = [0.091_402_61, 2.194_188_4, 4.842_966_6, -14.185_033];
    let b4 = [0.106_673_3, 12.641_946, -60.582_047, 110.362_77];
    let r2 = [-152.942_4, 59.286_38];
    let g2 = [4.277_299, 2.829_566];
    let b2 = [-89.903_11, 27.348_25];
    let v4 = [1.0, x, x * x, x * x * x];
    let v2 = [v4[2] * v4[2], v4[3] * v4[2]];
    let d4 = |c: [f32; 4]| c[0] * v4[0] + c[1] * v4[1] + c[2] * v4[2] + c[3] * v4[3];
    let d2 = |c: [f32; 2]| c[0] * v2[0] + c[1] * v2[1];
    [d4(r4) + d2(r2), d4(g4) + d2(g2), d4(b4) + d2(b2)]
}

const COOLWARM: [[f32; 3]; 9] = [
    [0.2298, 0.2987, 0.7537],
    [0.3509, 0.4652, 0.8879],
    [0.4846, 0.6236, 0.9739],
    [0.6196, 0.7414, 0.9954],
    [0.8654, 0.8654, 0.8654],
    [0.9567, 0.7469, 0.6481],
    [0.9578, 0.6033, 0.4828],
    [0.8873, 0.4001, 0.3135],
    [0.7057, 0.0156, 0.1502],
];

fn piecewise_coolwarm(t: f32) -> [f32; 3] {
    let s = t * 8.0;
    let i = (s.floor() as usize).min(7);
    let f = s - i as f32;
    let (a, b) = (COOLWARM[i], COOLWARM[i + 1]);
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}

/// sRGB colour (0..1) of a colour map at `t` (clamped). Matches the WGSL implementations.
pub(crate) fn colormap_srgb(map: super::ColorMap, t: f32) -> [f32; 3] {
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    let c = match map {
        super::ColorMap::Viridis => poly_viridis(t),
        super::ColorMap::Coolwarm => piecewise_coolwarm(t),
        super::ColorMap::Turbo => poly_turbo(t),
    };
    c.map(|v| v.clamp(0.0, 1.0))
}

pub(crate) fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
