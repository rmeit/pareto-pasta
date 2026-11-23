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
use faer::sparse::linalg::cholesky::{LltRef, SymbolicCholesky, factorize_symbolic_cholesky};
use faer::sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat};
use faer::{Conj, MatMut, Side};

pub use faer::Par;

mod tree;

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
    workspace: std::sync::Mutex<Option<Work>>,
    /// Numeric factorisations attempted so far, including failed ones.
    factorisations: std::sync::atomic::AtomicUsize,
    /// EXPERIMENT: tree-parallel numeric factorisation (None if faer chose
    /// a simplicial factor).
    tree: Option<tree::TreeFactor>,
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
        let tree = tree::TreeFactor::new(n, entries, &chol);
        Ok(Self { n, nnz_in: entries.len(), sym, argsort, chol, diag_entry, workspace: std::sync::Mutex::new(None), factorisations: Default::default(), tree })
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
        self.solve_from(values, rhs, par, 0.0)
    }

    /// Solves A x = rhs in place if A is positive definite; returns false
    /// (rhs untouched) if it is not, without trying any shift.
    pub fn try_solve(&self, values: &[f64], rhs: &mut [f64], par: Par) -> Result<bool, SolveError> {
        if !self.factorize(values, par)? {
            return Ok(false);
        }
        self.apply_inverse(rhs, par);
        Ok(true)
    }

    /// As [`SpdPattern::solve`], but starting the search for a shift at
    /// `first_shift` (an absolute value; 0 tries the matrix unshifted
    /// first). After an indefinite system, the previous shift is a much
    /// better start than zero: each failed attempt costs a factorisation.
    /// Shifts grow fourfold per attempt.
    pub fn solve_from(&self, values: &[f64], rhs: &mut [f64], par: Par, first_shift: f64) -> Result<f64, SolveError> {
        let shift = self.factorize_from(values, par, first_shift)?;
        self.apply_inverse(rhs, par);
        Ok(shift)
    }

    /// Factorises A and keeps the factor for [`SpdPattern::apply_inverse`].
    /// Returns false, and drops any kept factor, if A is not positive
    /// definite.
    pub fn factorize(&self, values: &[f64], par: Par) -> Result<bool, SolveError> {
        assert_eq!(values.len(), self.nnz_in);
        self.factorisations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut guard = self.work(par);
        let w = guard.as_mut().unwrap();
        w.valid = false;
        if let (Some(t), Ok(threads)) = (&self.tree, std::env::var("PASTA_TREE_THREADS")) {
            // EXPERIMENT
            let threads: usize = threads.parse().unwrap_or(8);
            let ok = t.factorize(values, &mut w.lval, threads);
            w.valid = ok;
            return Ok(ok);
        }
        let a = SparseColMat::new_from_argsort(self.sym.clone(), &self.argsort, values)
            .map_err(|e| SolveError::Build(format!("{e:?}")))?;
        let ok = self
            .chol
            .factorize_numeric_llt(
                &mut w.lval,
                a.as_ref(),
                Side::Lower,
                LltRegularization::default(),
                par,
                MemStack::new(&mut w.buf),
                Default::default(),
            )
            .is_ok();
        w.valid = ok;
        Ok(ok)
    }

    /// Factorises A plus the smallest diagonal shift (searched as in
    /// [`SpdPattern::solve_from`]) that makes it positive definite, keeps the
    /// factor, and returns the shift.
    pub fn factorize_from(&self, values: &[f64], par: Par, first_shift: f64) -> Result<f64, SolveError> {
        assert_eq!(values.len(), self.nnz_in);
        let mean_diag = {
            let s: f64 = self.diag_entry.iter().map(|&k| values[k].abs()).sum();
            (s / self.n.max(1) as f64).max(1e-300)
        };
        // A start below the smallest useful shift means "try unshifted".
        let mut shift = if first_shift >= 1e-8 * mean_diag { first_shift } else { 0.0 };
        let mut shifted: Vec<f64> = Vec::new();
        for _ in 0..24 {
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
            if self.factorize(vals, par)? {
                return Ok(shift);
            }
            shift = if shift == 0.0 { 1e-8 * mean_diag } else { shift * 4.0 };
        }
        Err(SolveError::NotPositiveDefinite)
    }

    /// EXPERIMENT: supernodes, total flops, and flops on the most expensive
    /// leaf-to-root path of the supernodal elimination tree (left-looking:
    /// each supernode's updates are counted where they are pulled in).
    pub fn tree_profile(&self) -> Option<(usize, f64, f64, usize)> {
        use faer::sparse::linalg::cholesky::SymbolicCholeskyRaw;
        let SymbolicCholeskyRaw::Supernodal(sym) = self.chol.raw() else { return None };
        let ns = sym.n_supernodes();
        let begin: Vec<usize> = sym.supernode_begin().to_vec();
        let end: Vec<usize> = sym.supernode_end().to_vec();
        let mut owner = vec![0usize; self.n];
        for s in 0..ns {
            for c in begin[s]..end[s] {
                owner[c] = s;
            }
        }
        let mut work = vec![0.0f64; ns];
        let mut parent = vec![usize::MAX; ns];
        for d in 0..ns {
            let pat: Vec<usize> = sym.supernode(d).pattern().to_vec();
            let c = (end[d] - begin[d]) as f64;
            let r = pat.len() as f64;
            work[d] += c * c * c / 3.0 + r * c * c;
            if let Some(&first) = pat.first() {
                parent[d] = owner[first];
            }
            // Pulls into each ancestor s: rows of d inside s (mid) and at or below s (bot).
            let mut i = 0;
            while i < pat.len() {
                let s = owner[pat[i]];
                let mut j = i;
                while j < pat.len() && owner[pat[j]] == s {
                    j += 1;
                }
                let mid = (j - i) as f64;
                let bot = (pat.len() - i) as f64;
                work[s] += 2.0 * c * (bot - 0.5 * mid) * mid;
                i = j;
            }
        }
        let total: f64 = work.iter().sum();
        // Children come before parents, so one forward pass gives path costs.
        let mut path = work.clone();
        let mut best = 0.0f64;
        let mut roots = 0;
        for s in 0..ns {
            if parent[s] == usize::MAX {
                roots += 1;
                best = best.max(path[s]);
            } else {
                let p = parent[s];
                let cand = path[s] + work[p];
                if cand > path[p] {
                    path[p] = cand;
                }
            }
        }
        Some((ns, total, best, roots))
    }

    /// Numeric factorisations attempted so far, including failed ones.
    pub fn factorisations(&self) -> usize {
        self.factorisations.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether a factor from [`SpdPattern::factorize`] is kept.
    pub fn has_factor(&self) -> bool {
        self.workspace.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|w| w.valid)
    }

    /// Replaces rhs by A⁻¹ rhs using the kept factor. Panics if there is none.
    pub fn apply_inverse(&self, rhs: &mut [f64], par: Par) {
        assert_eq!(rhs.len(), self.n);
        let mut guard = self.work(par);
        let w = guard.as_mut().unwrap();
        assert!(w.valid, "no factor kept");
        let llt = LltRef::<'_, usize, f64>::new(&self.chol, &w.lval);
        let rhs_mat = MatMut::from_column_major_slice_mut(rhs, self.n, 1);
        llt.solve_in_place_with_conj(Conj::No, rhs_mat, par, MemStack::new(&mut w.buf));
    }

    /// The workspace, with scratch sized for `par` (a kept factor survives a
    /// change of `par`).
    fn work(&self, par: Par) -> std::sync::MutexGuard<'_, Option<Work>> {
        let mut guard = self.workspace.lock().unwrap_or_else(|e| e.into_inner());
        if guard.as_ref().is_none_or(|w| w.par != par) {
            let req = self
                .chol
                .factorize_numeric_llt_scratch::<f64>(par, Default::default())
                .or(self.chol.solve_in_place_scratch::<f64>(1, par));
            let buf = MemBuffer::new(req);
            match guard.as_mut() {
                Some(w) => {
                    w.buf = buf;
                    w.par = par;
                }
                None => *guard = Some(Work { lval: vec![0.0f64; self.chol.len_val()], buf, par, valid: false }),
            }
        }
        guard
    }
}

struct Work {
    /// Factor values (faer overwrites them completely on each factorisation).
    lval: Vec<f64>,
    buf: MemBuffer,
    par: Par,
    /// Whether `lval` holds a complete factor.
    valid: bool,
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

    #[test]
    fn tree_factor_matches_faer() {
        // 27-point stencil on a 14^3 grid: dense enough for a supernodal factor.
        let m = 14;
        let id = |x: usize, y: usize, z: usize| (z * m + y) * m + x;
        let mut e = Vec::new();
        let mut v = Vec::new();
        let mut rng = Rng::new(3);
        for z in 0..m {
            for y in 0..m {
                for x in 0..m {
                    let a = id(x, y, z);
                    e.push((a, a));
                    v.push(30.0);
                    for dz in 0..2usize {
                        for dy in 0..3usize {
                            for dx in 0..3usize {
                                let (xx, yy, zz) = (x + dx, y + dy, z + dz);
                                if xx < 1 || yy < 1 || zz < 1 || xx > m || yy > m {
                                    continue;
                                }
                                let b = id(xx - 1, yy - 1, zz - 1);
                                if b < a {
                                    e.push((a, b));
                                    v.push(-0.5 + 0.1 * rng.symmetric());
                                }
                            }
                        }
                    }
                }
            }
        }
        let n = m * m * m;
        let p = SpdPattern::new(n, &e).unwrap();
        let tree = p.tree.as_ref().expect("supernodal");
        assert!(p.factorize(&v, Par::Seq).unwrap());
        let reference = p.workspace.lock().unwrap().as_ref().unwrap().lval.clone();
        for threads in [1, 4] {
            let mut l = vec![0.0; reference.len()];
            assert!(tree.factorize(&v, &mut l, threads));
            let err = l.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-10, "threads {threads}: max difference {err}");
        }
        // And faer's solve works on the tree factor.
        let mut l = vec![0.0; reference.len()];
        tree.factorize(&v, &mut l, 4);
        p.workspace.lock().unwrap().as_mut().unwrap().lval = l;
        let mut x = vec![1.0; n];
        p.apply_inverse(&mut x, Par::Seq);
        let mut ax = vec![0.0; n];
        for (k, &(r, c)) in e.iter().enumerate() {
            ax[r] += v[k] * x[c];
            if r != c {
                ax[c] += v[k] * x[r];
            }
        }
        assert!(ax.iter().all(|y| (y - 1.0).abs() < 1e-9));
    }
}
