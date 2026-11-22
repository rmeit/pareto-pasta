use std::io::Write;

/// Wavefront OBJ with vertex positions (mm) and triangles. Only vertices
/// used by `tris` are written.
pub fn write_obj(path: &std::path::Path, positions: &[[f32; 3]], tris: &[[u32; 3]]) -> std::io::Result<()> {
    let (map, used) = compact(positions.len(), tris);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "# pareto_pasta shell, millimetres")?;
    for &i in &used {
        let p = positions[i as usize];
        writeln!(f, "v {} {} {}", p[0], p[1], p[2])?;
    }
    for t in tris {
        writeln!(f, "f {} {} {}", map[t[0] as usize] + 1, map[t[1] as usize] + 1, map[t[2] as usize] + 1)?;
    }
    f.flush()
}

/// Binary STL (mm). A sheet has no thickness here, so this is a surface for
/// viewing, not a solid for printing.
pub fn write_stl(path: &std::path::Path, positions: &[[f32; 3]], tris: &[[u32; 3]]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut header = [0u8; 80];
    let text = b"pareto_pasta shell surface, mm";
    header[..text.len()].copy_from_slice(text);
    f.write_all(&header)?;
    f.write_all(&(tris.len() as u32).to_le_bytes())?;
    for t in tris {
        let [a, b, c] = t.map(|i| positions[i as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let mut n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-30);
        n.iter_mut().for_each(|x| *x /= l);
        for x in n.iter().chain(a.iter()).chain(b.iter()).chain(c.iter()) {
            f.write_all(&x.to_le_bytes())?;
        }
        f.write_all(&[0u8; 2])?;
    }
    f.flush()
}

fn compact(n: usize, tris: &[[u32; 3]]) -> (Vec<u32>, Vec<u32>) {
    let mut map = vec![u32::MAX; n];
    let mut used = Vec::new();
    for t in tris {
        for &i in t {
            if map[i as usize] == u32::MAX {
                map[i as usize] = used.len() as u32;
                used.push(i);
            }
        }
    }
    (map, used)
}
