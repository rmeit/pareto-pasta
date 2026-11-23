//! EXPERIMENT: left-looking supernodal Cholesky in which supernodes of
//! independent subtrees of the elimination tree are factorised at the same
//! time. It writes the factor in faer's own layout, so faer's solve uses it.
//! faer's own loop visits the supernodes in order and only uses threads
//! inside the dense blocks.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{LltRegularization, cholesky_in_place, cholesky_in_place_scratch};
use faer::linalg::matmul::matmul;
use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::sparse::linalg::cholesky::{SymbolicCholesky, SymbolicCholeskyRaw};
use faer::{Accum, MatMut, MatRef, Par};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const NONE: usize = usize::MAX;

pub(crate) struct TreeFactor {
    /// Lower triangle of the permuted matrix, by column.
    col_ptr: Vec<usize>,
    row: Vec<u32>,
    /// Position in `row` of each input entry.
    slot: Vec<u32>,
    begin: Vec<usize>,
    pat_ptr: Vec<usize>,
    pat: Vec<u32>,
    val_ptr: Vec<usize>,
    parent: Vec<usize>,
    n_children: Vec<usize>,
    /// Supernodes whose columns update each supernode.
    updaters: Vec<Vec<u32>>,
    leaves: Vec<usize>,
    n: usize,
    max_rows: usize,
    max_cols: usize,
    pool: Mutex<Option<(usize, Arc<rayon::ThreadPool>)>>,
    scratch: Mutex<Vec<Scratch>>,
}

struct Scratch {
    g2l: Vec<isize>,
    tmp: Vec<f64>,
    mem: MemBuffer,
}

#[derive(Clone, Copy)]
struct Shared(*mut f64);
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl TreeFactor {
    pub(crate) fn new(n: usize, entries: &[(usize, usize)], chol: &SymbolicCholesky<usize>) -> Option<Self> {
        let SymbolicCholeskyRaw::Supernodal(sym) = chol.raw() else { return None };
        let inv: Vec<usize> = match chol.perm() {
            Some(p) => p.arrays().1.to_vec(),
            None => (0..n).collect(),
        };
        let mut col_ptr = vec![0usize; n + 1];
        let pe: Vec<(usize, usize)> = entries
            .iter()
            .map(|&(r, c)| {
                let (a, b) = (inv[r], inv[c]);
                (a.max(b), a.min(b))
            })
            .collect();
        for &(_, j) in &pe {
            col_ptr[j + 1] += 1;
        }
        for j in 0..n {
            col_ptr[j + 1] += col_ptr[j];
        }
        let mut fill = col_ptr.clone();
        let mut row = vec![0u32; pe.len()];
        let mut slot = vec![0u32; pe.len()];
        for (k, &(i, j)) in pe.iter().enumerate() {
            row[fill[j]] = i as u32;
            slot[k] = fill[j] as u32;
            fill[j] += 1;
        }
        let ns = sym.n_supernodes();
        let begin: Vec<usize> = sym.supernode_begin().iter().copied().chain([n]).collect();
        let mut pat_ptr = vec![0usize; ns + 1];
        let mut pat = Vec::new();
        for s in 0..ns {
            pat.extend(sym.supernode(s).pattern().iter().map(|&r| r as u32));
            pat_ptr[s + 1] = pat.len();
        }
        let val_ptr: Vec<usize> = sym.col_ptr_for_val().to_vec();
        let mut owner = vec![0usize; n];
        for s in 0..ns {
            for c in begin[s]..begin[s + 1] {
                owner[c] = s;
            }
        }
        let mut parent = vec![NONE; ns];
        let mut n_children = vec![0usize; ns];
        let mut updaters: Vec<Vec<u32>> = vec![Vec::new(); ns];
        let (mut max_rows, mut max_cols) = (0, 0);
        for d in 0..ns {
            let dp = &pat[pat_ptr[d]..pat_ptr[d + 1]];
            let nc = begin[d + 1] - begin[d];
            max_cols = max_cols.max(nc);
            max_rows = max_rows.max(nc + dp.len());
            if let Some(&first) = dp.first() {
                parent[d] = owner[first as usize];
                n_children[parent[d]] += 1;
            }
            let mut last = NONE;
            for &r in dp {
                let s = owner[r as usize];
                if s != last {
                    updaters[s].push(d as u32);
                    last = s;
                }
            }
        }
        let leaves = (0..ns).filter(|&s| n_children[s] == 0).collect();
        Some(Self {
            col_ptr,
            row,
            slot,
            begin,
            pat_ptr,
            pat,
            val_ptr,
            parent,
            n_children,
            updaters,
            leaves,
            n,
            max_rows,
            max_cols,
            pool: Mutex::new(None),
            scratch: Mutex::new(Vec::new()),
        })
    }

    fn pool(&self, threads: usize) -> Arc<rayon::ThreadPool> {
        let mut g = self.pool.lock().unwrap();
        if g.as_ref().is_none_or(|(t, _)| *t != threads) {
            let p = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            *g = Some((threads, Arc::new(p)));
        }
        g.as_ref().unwrap().1.clone()
    }

    fn take_scratch(&self) -> Scratch {
        if let Some(s) = self.scratch.lock().unwrap().pop() {
            return s;
        }
        let req = cholesky_in_place_scratch::<f64>(self.max_cols, Par::Seq, Default::default());
        Scratch { g2l: vec![-1; self.n], tmp: vec![0.0; self.max_rows * self.max_cols], mem: MemBuffer::new(req) }
    }

    /// Factorises into `lval` (faer's supernodal layout). False if the
    /// matrix is not positive definite.
    pub(crate) fn factorize(&self, values: &[f64], lval: &mut [f64], threads: usize) -> bool {
        let mut vp = vec![0.0; self.row.len()];
        for (k, &v) in values.iter().enumerate() {
            vp[self.slot[k] as usize] += v;
        }
        let pending: Vec<AtomicUsize> = self.n_children.iter().map(|&c| AtomicUsize::new(c)).collect();
        let fail = AtomicBool::new(false);
        let out = Shared(lval.as_mut_ptr());
        let climb = |mut s: usize| {
            let mut sc = self.take_scratch();
            loop {
                if !fail.load(Ordering::Relaxed) && !self.compute(s, &vp, out, &mut sc) {
                    fail.store(true, Ordering::Relaxed);
                }
                let p = self.parent[s];
                if p == NONE || pending[p].fetch_sub(1, Ordering::AcqRel) != 1 {
                    break;
                }
                s = p;
            }
            self.scratch.lock().unwrap().push(sc);
        };
        if threads <= 1 {
            for &leaf in &self.leaves {
                climb(leaf);
            }
        } else {
            self.pool(threads).scope(|sc| {
                for &leaf in &self.leaves {
                    let climb = &climb;
                    sc.spawn(move |_| climb(leaf));
                }
            });
        }
        !fail.load(Ordering::Relaxed)
    }

    fn compute(&self, s: usize, vp: &[f64], out: Shared, sc: &mut Scratch) -> bool {
        let (s_start, s_end) = (self.begin[s], self.begin[s + 1]);
        let ncols = s_end - s_start;
        let spat = &self.pat[self.pat_ptr[s]..self.pat_ptr[s + 1]];
        let nrows = ncols + spat.len();
        for (i, &r) in spat.iter().enumerate() {
            sc.g2l[r as usize] = (i + ncols) as isize;
        }
        // SAFETY: supernode s's values are written only here, after every
        // descendant has finished (the pending counters order this), and
        // descendants' values are only read.
        let ls = unsafe { std::slice::from_raw_parts_mut(out.0.add(self.val_ptr[s]), self.val_ptr[s + 1] - self.val_ptr[s]) };
        ls.fill(0.0);
        for j in s_start..s_end {
            let jj = j - s_start;
            for q in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row[q] as usize;
                let ix = if i >= s_end { sc.g2l[i] as usize } else { i - s_start };
                ls[ix + jj * nrows] += vp[q];
            }
        }
        for &d in &self.updaters[s] {
            let d = d as usize;
            let dc = self.begin[d + 1] - self.begin[d];
            let dpat = &self.pat[self.pat_ptr[d]..self.pat_ptr[d + 1]];
            let dr = dc + dpat.len();
            let ld = unsafe { std::slice::from_raw_parts(out.0.add(self.val_ptr[d]) as *const f64, dr * dc) };
            let start = dpat.partition_point(|&r| (r as usize) < s_start);
            let mid = dpat[start..].partition_point(|&r| (r as usize) < s_end);
            let m = dpat.len() - start;
            let lmb = MatRef::from_column_major_slice_with_stride(&ld[dc + start..], m, dc, dr);
            let lmid = lmb.subrows(0, mid);
            let tmp = MatMut::from_column_major_slice_mut(&mut sc.tmp[..m * mid], m, mid);
            matmul(tmp, Accum::Replace, lmb, lmid.transpose(), 1.0, Par::Seq);
            for jj in 0..mid {
                let col = dpat[start + jj] as usize - s_start;
                for ii in jj..m {
                    let r = dpat[start + ii] as usize;
                    let rl = if ii < mid { r - s_start } else { sc.g2l[r] as usize };
                    ls[rl + col * nrows] -= sc.tmp[ii + jj * m];
                }
            }
        }
        let l = MatMut::from_column_major_slice_with_stride_mut(ls, nrows, ncols, nrows);
        let (mut top, bot) = l.split_at_row_mut(ncols);
        let ok = cholesky_in_place(
            top.as_mut(),
            LltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut sc.mem),
            Default::default(),
        )
        .is_ok();
        if ok {
            solve_lower_triangular_in_place(top.as_ref(), bot.transpose_mut(), Par::Seq);
        }
        for &r in spat {
            sc.g2l[r as usize] = -1;
        }
        ok
    }
}
