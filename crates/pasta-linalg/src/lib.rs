//! Sparse symmetric positive definite (SPD) solves on a fixed sparsity
//! pattern, plus a small deterministic random number generator.
//!
//! The solvers assemble their Hessians as a list of (row, col) entries in the
//! lower triangle, in a fixed order. [`SpdPattern`] analyses that list once
//! (fill-reducing ordering and symbolic Cholesky) and then factorises any
//! number of value vectors given in the same order. Duplicate entries are
//! summed.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{SymbolicCholesky, factorize_symbolic_cholesky};
use faer::sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat};
use faer::{Conj, MatMut, Side};

pub use faer::Par;

/// Parallelism to use for one factorisation.
pub fn par(threads: usize) -> Par {
    if threads <= 1 { Par::Seq } else { Par::rayon(threads) }
}

#[derive(Debug)]
pub enum SolveError {
    /// The matrix stayed indefinite after the largest diagonal shift.
    NotPositiveDefinite,
    Build(String),
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::NotPositiveDefinite => write!(f, "matrix is not positive definite"),
            SolveError::Build(s) => write!(f, "could not build matrix: {s}"),
        }
    }
}

impl std::error::Error for SolveError {}

/// Sparsity pattern and symbolic factorisation of an SPD matrix.
pub struct SpdPattern {
    n: usize,
    nnz_in: usize,
    sym: SymbolicSparseColMat<usize>,
    argsort: Argsort<usize>,
    chol: SymbolicCholesky<usize>,
    /// For each row, the index of one input entry on its diagonal.
    diag_entry: Vec<usize>,
    /// Factor values and scratch memory, reused between solves (faer
    /// overwrites them completely).
    workspace: std::sync::Mutex<Option<(Vec<f64>, MemBuffer, Par)>>,
}

impl SpdPattern {
    /// `entries` lists (row, col) with row >= col. Every diagonal entry must
    /// appear at least once.
    pub fn new(n: usize, entries: &[(usize, usize)]) -> Result<Self, SolveError> {
        let mut diag_entry = vec![usize::MAX; n];
        let idx: Vec<Pair<usize, usize>> = entries
            .iter()
            .enumerate()
            .map(|(k, &(r, c))| {
                assert!(r >= c && r < n, "entry ({r}, {c}) outside the lower triangle of {n}");
                if r == c && diag_entry[r] == usize::MAX {
                    diag_entry[r] = k;
                }
                Pair::new(r, c)
            })
            .collect();
        if let Some(r) = diag_entry.iter().position(|&k| k == usize::MAX) {
            return Err(SolveError::Build(format!("row {r} has no diagonal entry")));
        }
        let (sym, argsort) = SymbolicSparseColMat::try_new_from_indices(n, n, &idx)
            .map_err(|e| SolveError::Build(format!("{e:?}")))?;
        let chol = factorize_symbolic_cholesky(sym.as_ref(), Side::Lower, Default::default(), Default::default())
            .map_err(|e| SolveError::Build(format!("{e:?}")))?;
        Ok(Self { n, nnz_in: entries.len(), sym, argsort, chol, diag_entry, workspace: std::sync::Mutex::new(None) })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn num_entries(&self) -> usize {
        self.nnz_in
    }

    /// Solves A x = rhs in place, where A has `values` in the order of the
    /// entries given to [`SpdPattern::new`]. If A is not positive definite, a
    /// growing multiple of its mean diagonal is added until it is. Returns the
    /// shift that was used (0 when none was needed).
    pub fn solve(&self, values: &[f64], rhs: &mut [f64], par: Par) -> Result<f64, SolveError> {
        assert_eq!(values.len(), self.nnz_in);
        assert_eq!(rhs.len(), self.n);
        let mean_diag = {
            let s: f64 = self.diag_entry.iter().map(|&k| values[k].abs()).sum();
            (s / self.n.max(1) as f64).max(1e-300)
        };
        let mut guard = self.workspace.lock().unwrap_or_else(|e| e.into_inner());
        let reuse = matches!(&*guard, Some((_, _, p)) if *p == par);
        if !reuse {
            let req = self
                .chol
                .factorize_numeric_llt_scratch::<f64>(par, Default::default())
                .or(self.chol.solve_in_place_scratch::<f64>(1, par));
            *guard = Some((vec![0.0f64; self.chol.len_val()], MemBuffer::new(req), par));
        }
        let (lval, buf, _) = guard.as_mut().unwrap();
        let mut shift = 0.0;
        let mut shifted: Vec<f64> = Vec::new();
        for _ in 0..14 {
            let vals: &[f64] = if shift == 0.0 {
                values
            } else {
                shifted.clear();
                shifted.extend_from_slice(values);
                for &k in &self.diag_entry {
                    shifted[k] += shift;
                }
                &shifted
            };
            let a = SparseColMat::new_from_argsort(self.sym.clone(), &self.argsort, vals)
                .map_err(|e| SolveError::Build(format!("{e:?}")))?;
            let result = self.chol.factorize_numeric_llt(
                lval,
                a.as_ref(),
                Side::Lower,
                LltRegularization::default(),
                par,
                MemStack::new(buf),
                Default::default(),
            );
            if let Ok(llt) = result {
                let rhs_mat = MatMut::from_column_major_slice_mut(rhs, self.n, 1);
                llt.solve_in_place_with_conj(Conj::No, rhs_mat, par, MemStack::new(buf));
                return Ok(shift);
            }
            shift = if shift == 0.0 { 1e-10 * mean_diag } else { shift * 10.0 };
        }
        Err(SolveError::NotPositiveDefinite)
    }
}

/// SplitMix64: small, fast, deterministic. Used for the fixed-seed
/// perturbations that let buckling start.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    pub fn symmetric(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_laplacian_with_duplicates() {
        // 1D Laplacian with the diagonal split into two duplicate entries.
        let n = 50;
        let mut e = Vec::new();
        let mut v = Vec::new();
        for i in 0..n {
            e.push((i, i));
            v.push(1.0);
            e.push((i, i));
            v.push(1.0 + 1e-3);
            if i + 1 < n {
                e.push((i + 1, i));
                v.push(-1.0);
            }
        }
        let p = SpdPattern::new(n, &e).unwrap();
        let mut rhs = vec![1.0; n];
        let shift = p.solve(&v, &mut rhs, Par::Seq).unwrap();
        assert_eq!(shift, 0.0);
        // Check the residual.
        for i in 0..n {
            let mut ax = (2.0 + 1e-3) * rhs[i];
            if i > 0 {
                ax -= rhs[i - 1];
            }
            if i + 1 < n {
                ax -= rhs[i + 1];
            }
            assert!((ax - 1.0).abs() < 1e-8, "row {i}: {ax}");
        }
    }

    #[test]
    fn shifts_indefinite_matrix() {
        let e = [(0, 0), (1, 1), (1, 0)];
        let v = [1.0, 1.0, 2.0]; // eigenvalues -1 and 3
        let p = SpdPattern::new(2, &e).unwrap();
        let mut rhs = vec![1.0, 1.0];
        let shift = p.solve(&v, &mut rhs, Par::Seq).unwrap();
        assert!(shift >= 1.0 - 1e-9, "shift {shift}, x {rhs:?}");
    }
}
