//! Truncated SVD for MPS bond compression.
//!
//! Computed by **one-sided (Hestenes) Jacobi SVD on A itself** — NOT by
//! eigendecomposing the normal equations A†A. Two reasons this matters for
//! deep-circuit MPS, both learned the hard way:
//!
//! 1. Forming A†A squares the condition number, so any singular value below
//!    ~√ε_mach·σ_max is lost to rounding. Deep, near-saturated bond matrices
//!    have dense spectra full of such values; on A directly the resolvable
//!    floor is ~ε_mach·σ_max instead.
//! 2. The normal-equations path reconstructed `U = A·V·S⁻¹`, dividing by tiny
//!    (and, from a non-converged Jacobi eigensolver, unreliable) σ. That
//!    manufactured non-orthogonal U — a *non-unitary* split that made state
//!    norms drift to 21.7 (should be exactly 1) on 11-qubit deep circuits at
//!    χ = 64, where truncation is mathematically impossible.
//!
//! One-sided Jacobi orthogonalises the COLUMNS of A by right-multiplying by
//! plane rotations; the converged column norms are the singular values, the
//! normalised columns are U, and the accumulated rotations are V. U and V are
//! orthonormal *by construction* — never reconstructed via S⁻¹ — so a split is
//! unitary up to genuine truncation, and the discarded singular-value weight is
//! reported (`SvdResultFlat::discarded_weight`) as a truncation certificate.
//!
//! Two parallel APIs:
//! - [`truncated_svd_flat`] is the row-major flat-buffer entry point.
//!   The MPS hot path and the CUDA backend both go through this; it
//!   avoids the nested `Vec<Vec<Complex64>>` allocation and copy that
//!   used to bracket every SVD call.
//! - [`truncated_svd`] is the nested-Vec API kept for back-compat. It
//!   defers to the flat path internally.

use num_complex::Complex64;

/// Result of SVD: A = U * diag(S) * Vt. Row-major flat buffers.
///
/// - `u` is `m × k` row-major: element `(i, k_idx)` lives at
///   `u[i * k + k_idx]` where `k = s.len()`.
/// - `vt` is `k × n` row-major: element `(k_idx, j)` lives at
///   `vt[k_idx * n + j]`.
/// - `discarded_weight` is the summed square of the singular values dropped by
///   truncation (Σ σ² over the tail past `max_rank`/`threshold`). It is the
///   standard MPS fidelity proxy: 0.0 means the split was exact, and a growing
///   value across a run is the honest signal that the bond dimension is too
///   small. Callers accumulate it to report/gate truncation error.
pub struct SvdResultFlat {
    pub u: Vec<Complex64>,
    pub s: Vec<f64>,
    pub vt: Vec<Complex64>,
    pub m: usize,
    pub n: usize,
    pub discarded_weight: f64,
}

/// Result of SVD: A = U * diag(S) * Vt. Nested-Vec form (legacy).
pub struct SvdResult {
    pub u: Vec<Vec<Complex64>>,  // m x k
    pub s: Vec<f64>,             // k singular values
    pub vt: Vec<Vec<Complex64>>, // k x n
}

impl From<SvdResultFlat> for SvdResult {
    fn from(f: SvdResultFlat) -> Self {
        let k = f.s.len();
        let mut u = vec![vec![Complex64::new(0.0, 0.0); k]; f.m];
        for i in 0..f.m {
            for j in 0..k {
                u[i][j] = f.u[i * k + j];
            }
        }
        let mut vt = vec![vec![Complex64::new(0.0, 0.0); f.n]; k];
        for i in 0..k {
            for j in 0..f.n {
                vt[i][j] = f.vt[i * f.n + j];
            }
        }
        SvdResult { u, s: f.s, vt }
    }
}

/// Compute truncated SVD of an m×n complex matrix laid out row-major
/// in a flat buffer. Row `i` starts at `matrix[i * stride]`; the row
/// has `n` complex elements. (Setting `stride == n` is the dense /
/// contiguous case — what the MPS Step-3 reshape produces.)
///
/// Keeps at most `max_rank` singular values strictly above `threshold`,
/// never fewer than 1.
pub fn truncated_svd_flat(
    matrix: &[Complex64],
    m: usize,
    n: usize,
    stride: usize,
    max_rank: usize,
    threshold: f64,
) -> SvdResultFlat {
    if m == 0 || n == 0 {
        return SvdResultFlat {
            u: vec![],
            s: vec![],
            vt: vec![],
            m,
            n,
            discarded_weight: 0.0,
        };
    }
    debug_assert!(stride >= n, "stride {stride} < n {n}");
    debug_assert!(matrix.len() >= (m - 1) * stride + n);

    // Dense contiguous m×n copy (drops any input stride padding).
    let mut a = Vec::with_capacity(m * n);
    for i in 0..m {
        a.extend_from_slice(&matrix[i * stride..i * stride + n]);
    }

    // One-sided Jacobi orthogonalises COLUMNS, so it wants rows ≥ cols. MPS
    // splits produce both shapes; for a wide matrix run on Aᴴ (which is tall)
    // and swap the U/V roles — A = UΣVᴴ ⇔ Aᴴ = VΣUᴴ. Either way `u_full` ends
    // up m×K, `vt_full` K×n, with K = min(m, n) singular values (unsorted).
    let k_full = m.min(n);
    let (u_full, s_full, vt_full): (Vec<Complex64>, Vec<f64>, Vec<Complex64>) = if m >= n {
        let (u, s, v) = one_sided_jacobi(&a, m, n); // u: m×n, v: n×n unitary
                                                    // vt = vᴴ : row k, col j ← conj(v[j][k])
        let mut vt = vec![Complex64::new(0.0, 0.0); n * n];
        for k in 0..n {
            for j in 0..n {
                vt[k * n + j] = v[j * n + k].conj();
            }
        }
        (u, s, vt)
    } else {
        // Aᴴ is n×m (tall): ah[j][i] = conj(a[i][j]).
        let mut ah = vec![Complex64::new(0.0, 0.0); n * m];
        for i in 0..m {
            for j in 0..n {
                ah[j * m + i] = a[i * n + j].conj();
            }
        }
        let (ub, s, vb) = one_sided_jacobi(&ah, n, m); // ub: n×m, vb: m×m unitary
                                                       // A = Vb Σ Ubᴴ ⇒ U_A = Vb (m×m), Vt_A = Ubᴴ (m×n): row k, col j ← conj(ub[j][k])
        let mut vt = vec![Complex64::new(0.0, 0.0); m * n];
        for k in 0..m {
            for j in 0..n {
                vt[k * n + j] = ub[j * m + k].conj();
            }
        }
        (vb, s, vt)
    };

    // Order the K singular values largest-first, then truncate: keep at most
    // `max_rank` strictly above `threshold`, never fewer than 1 (unchanged
    // semantics). The dropped tail's Σσ² is the truncation certificate.
    let mut order: Vec<usize> = (0..k_full).collect();
    // total_cmp (not partial_cmp().unwrap()) so a NaN/Inf σ from a pathological
    // input yields a defined order instead of a panic.
    order.sort_by(|&i, &j| s_full[j].total_cmp(&s_full[i]));
    let rank = order
        .iter()
        .take(max_rank)
        .take_while(|&&i| s_full[i] > threshold)
        .count()
        .max(1);
    let discarded_weight: f64 = order[rank..].iter().map(|&i| s_full[i] * s_full[i]).sum();

    let s: Vec<f64> = order[..rank].iter().map(|&i| s_full[i]).collect();
    let mut u = vec![Complex64::new(0.0, 0.0); m * rank];
    for row in 0..m {
        for (kc, &src) in order[..rank].iter().enumerate() {
            u[row * rank + kc] = u_full[row * k_full + src];
        }
    }
    let mut vt = vec![Complex64::new(0.0, 0.0); rank * n];
    for (kc, &src) in order[..rank].iter().enumerate() {
        for j in 0..n {
            vt[kc * n + j] = vt_full[src * n + j];
        }
    }

    SvdResultFlat {
        u,
        s,
        vt,
        m,
        n,
        discarded_weight,
    }
}

/// Compute truncated SVD of an m×n complex matrix.
/// Nested-Vec form retained for back-compat; defers to
/// [`truncated_svd_flat`].
pub fn truncated_svd(matrix: &[Vec<Complex64>], max_rank: usize, threshold: f64) -> SvdResult {
    let m = matrix.len();
    if m == 0 {
        return SvdResult {
            u: vec![],
            s: vec![],
            vt: vec![],
        };
    }
    let n = matrix[0].len();
    let mut flat = Vec::with_capacity(m * n);
    for row in matrix {
        flat.extend_from_slice(row);
    }
    truncated_svd_flat(&flat, m, n, n, max_rank, threshold).into()
}

/// One-sided (Hestenes) Jacobi SVD of a **tall or square** complex matrix `a`
/// (row-major, `rows × cols`, requires `rows ≥ cols`). Returns the thin SVD
/// `A = U Σ Vᴴ`:
/// - `u` is `rows × cols` row-major with orthonormal columns,
/// - `s` holds the `cols` singular values (UNSORTED, in column order),
/// - `v` is `cols × cols` row-major and unitary.
///
/// Columns of a working copy of A are rotated to mutual orthogonality by
/// right-multiplying plane rotations; at convergence each column's norm is a
/// singular value, the normalised column is the matching column of U, and the
/// accumulated rotations are V. U and V are orthonormal *by construction* — the
/// method never forms A†A and never reconstructs U via Σ⁻¹, which is exactly
/// what made the previous normal-equations kernel non-unitary on deep circuits.
/// Raw column base pointer that rayon may share across a round's tasks.
///
/// # Safety contract, stated once and load-bearing
///
/// Within one tournament ROUND every column index appears in exactly one
/// pair (the round-robin schedule is a partition), and a task touches only
/// its own two columns. Disjoint columns of a column-major buffer are
/// disjoint memory, so the aliasing rules hold for the same reason
/// `split_at_mut` would — the partition is just not expressible as slice
/// splits. Rounds are separated by rayon's join barrier, so no task ever
/// sees a column mid-rotation from another round.
#[derive(Clone, Copy)]
struct ColBase(*mut Complex64);
unsafe impl Send for ColBase {}
unsafe impl Sync for ColBase {}

/// One-sided (Hestenes) Jacobi, parallel across each round's DISJOINT
/// column pairs.
///
/// # Ordering: round-robin tournament, not the old cyclic (p, q) scan
///
/// The rotation set per sweep is identical; only the order changes. That is
/// what makes rounds parallelisable: the circle-method schedule pairs every
/// column exactly once per round, pairs touch disjoint columns, and disjoint
/// rotations commute EXACTLY — so the result is deterministic and
/// **independent of thread count** (asserted by
/// `thread_count_does_not_change_the_bits`), though it differs from the old
/// cyclic order in the last bits, as any FP reordering does. Convergence is
/// unchanged in kind: one-sided Jacobi under any fixed cyclic-by-rounds
/// ordering strictly reduces the off-diagonal Gram norm per rotation, and
/// the sweep-level "nothing rotated" criterion is untouched.
///
/// # Storage: column-major internally
///
/// The public API stays row-major; internally columns are made contiguous so
/// a pair's working set is two cache-friendly slices rather than two strided
/// walks — this is worth more than the parallelism at small χ and is what
/// makes the per-task unsafe justification a one-liner about disjoint
/// ranges. Transposes at entry/exit are O(rows·cols), noise against the
/// O(sweeps·cols²·rows) rotation work.
fn one_sided_jacobi(
    a: &[Complex64],
    rows: usize,
    cols: usize,
) -> (Vec<Complex64>, Vec<f64>, Vec<Complex64>) {
    use rayon::prelude::*;

    // Column-major working copies: column j of `b` at b[j*rows .. (j+1)*rows],
    // column j of `v` at v[j*cols .. (j+1)*cols].
    let mut b = vec![Complex64::new(0.0, 0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            b[j * rows + i] = a[i * cols + j];
        }
    }
    let mut v = vec![Complex64::new(0.0, 0.0); cols * cols];
    for j in 0..cols {
        v[j * cols + j] = Complex64::new(1.0, 0.0);
    }

    // BLOCK-round-robin: the tournament runs over BLOCKS of columns, not
    // single columns. A first-cut pair-level tournament scaled 1→4 threads
    // and then went BACKWARDS at 8-16 — measured on hea_31q_L16 (quiet box):
    // 22.2 s at 4 threads, 27.9 s at 16 with 129 s of SYS time, because a
    // 512-column matrix has 511 barriers per sweep and a run makes ~3M of
    // them. Blocks cut the barrier count by ~JACOBI_BLOCK× and fatten each
    // task from one O(rows) rotation to a block-pair's worth, which is the
    // difference between paying rayon's join per microsecond of work and per
    // half-millisecond of it.
    //
    // The schedule is a function of `cols` ONLY — never of the thread count —
    // so results stay deterministic and thread-count-invariant: block pairs
    // within a round touch disjoint columns, rotations inside a task run in
    // a fixed serial order, and the block size is a constant. (i, i) entries
    // are the diagonal round: each block orthogonalises its own interior.
    const JACOBI_BLOCK: usize = 16;
    let nblocks = cols.div_ceil(JACOBI_BLOCK);
    let bm = if nblocks.is_multiple_of(2) {
        nblocks
    } else {
        nblocks + 1
    };
    let mut rounds: Vec<Vec<(usize, usize)>> = Vec::new();
    rounds.push((0..nblocks).map(|i| (i, i)).collect());
    for r in 0..bm.saturating_sub(1) {
        let slot = |k: usize| -> usize {
            if k == 0 {
                0
            } else {
                1 + (r + k - 1) % (bm - 1)
            }
        };
        rounds.push(
            (0..bm / 2)
                .filter_map(|i| {
                    let (x, y) = (slot(i), slot(bm - 1 - i));
                    if x >= nblocks || y >= nblocks {
                        return None; // the bye slot
                    }
                    Some((x.min(y), x.max(y)))
                })
                .collect(),
        );
    }
    let block_range = |i: usize| (i * JACOBI_BLOCK, ((i + 1) * JACOBI_BLOCK).min(cols));

    // Converged when a whole sweep rotates nothing — the correct criterion,
    // unlike the old fixed 100-rotation cap that could not diagonalise a
    // 128-column matrix. Quadratic convergence, empirically ~6-10 sweeps; 60
    // is a safety ceiling with large margin, and converged inputs break out
    // early.
    let tol = 1e-14;
    let bp = ColBase(b.as_mut_ptr());
    let vp = ColBase(v.as_mut_ptr());
    // Grain gate, still load-bearing: a deep circuit spends MOST of its gates
    // at small χ while ramping up, and parallelism there is pure overhead —
    // measured 286 s of SYS time on hea_28q_L16 before the gate existed.
    // Below the gate the SAME schedule runs sequentially; tasks within a
    // round are independent, so serial-vs-parallel is bit-identical by
    // construction (the thread-count test pins 1 thread ≡ 8 threads, which
    // includes "inline" as a special case).
    let par = cols >= 128;
    for _sweep in 0..60 {
        let mut rotated = false;
        for tasks in &rounds {
            let one = move |&(bi, bj): &(usize, usize)| {
                // Bind the WRAPPERS, not their pointer fields: edition-2021
                // closure capture would otherwise capture the bare `*mut`
                // (not Send/Sync) instead of the vouched-for ColBase.
                let (bp, vp) = (bp, vp);
                let col = |j: usize| {
                    // SAFETY: see `ColBase` — block pairs within a round are
                    // a partition of the blocks, so every column this task
                    // touches belongs to it alone this round.
                    unsafe {
                        (
                            std::slice::from_raw_parts_mut(bp.0.add(j * rows), rows),
                            std::slice::from_raw_parts_mut(vp.0.add(j * cols), cols),
                        )
                    }
                };
                let (s0, e0) = block_range(bi);
                let mut did = false;
                if bi == bj {
                    for p in s0..e0 {
                        for q in (p + 1)..e0 {
                            let (colp, vcolp) = col(p);
                            let (colq, vcolq) = col(q);
                            did |= rotate_pair(colp, colq, vcolp, vcolq, tol);
                        }
                    }
                } else {
                    let (s1, e1) = block_range(bj);
                    for p in s0..e0 {
                        for q in s1..e1 {
                            let (colp, vcolp) = col(p);
                            let (colq, vcolq) = col(q);
                            did |= rotate_pair(colp, colq, vcolp, vcolq, tol);
                        }
                    }
                }
                did
            };
            let did = if par {
                tasks.par_iter().map(one).reduce(|| false, |x, y| x | y)
            } else {
                tasks.iter().map(one).fold(false, |x, y| x | y)
            };
            rotated |= did;
        }
        if !rotated {
            break;
        }
    }

    // Column norms are the singular values; normalise columns to form U.
    let mut s = vec![0.0f64; cols];
    for (j, sj) in s.iter_mut().enumerate() {
        let nrm: f64 = b[j * rows..(j + 1) * rows]
            .iter()
            .map(|z| z.norm_sqr())
            .sum();
        *sj = nrm.sqrt();
        if *sj > 1e-300 {
            let inv = 1.0 / *sj;
            for z in &mut b[j * rows..(j + 1) * rows] {
                *z *= inv;
            }
        }
        // A zero column (σ ≈ 0) is left zero; it is truncated by the caller.
    }

    // Back to the row-major layout the callers consume.
    let mut u = vec![Complex64::new(0.0, 0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            u[i * cols + j] = b[j * rows + i];
        }
    }
    let mut v_rm = vec![Complex64::new(0.0, 0.0); cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            v_rm[i * cols + j] = v[j * cols + i];
        }
    }
    (u, s, v_rm)
}

/// Gram + one plane rotation on a single column pair. The arithmetic is the
/// old cyclic loop's, verbatim — only the memory layout (contiguous columns)
/// and the caller's ordering changed.
fn rotate_pair(
    colp: &mut [Complex64],
    colq: &mut [Complex64],
    vcolp: &mut [Complex64],
    vcolq: &mut [Complex64],
    tol: f64,
) -> bool {
    let (mut app, mut aqq) = (0.0f64, 0.0f64);
    let mut apq = Complex64::new(0.0, 0.0);
    for (bip, biq) in colp.iter().zip(colq.iter()) {
        app += bip.norm_sqr();
        aqq += biq.norm_sqr();
        apq += bip.conj() * biq;
    }
    let apq_abs = apq.norm();
    if apq_abs < 1e-300 || apq_abs <= tol * (app * aqq).sqrt() {
        return false; // columns already orthogonal
    }
    // Real Jacobi angle on the phased 2×2 Hermitian Gram
    // [[app, apq], [conj(apq), aqq]]; this choice of t zeroes apq.
    let phase = apq / apq_abs; // e^{iθ}
    let tau = (aqq - app) / (2.0 * apq_abs);
    let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = c * t;
    let cs = Complex64::new(c, 0.0);
    let sp = phase.conj() * s; // conj(phase)·s
    let sq = phase * s;
    // col_p' = c·col_p − conj(phase)·s·col_q
    // col_q' = phase·s·col_p + c·col_q
    for (bip, biq) in colp.iter_mut().zip(colq.iter_mut()) {
        let (op, oq) = (*bip, *biq);
        *bip = cs * op - sp * oq;
        *biq = sq * op + cs * oq;
    }
    for (vip, viq) in vcolp.iter_mut().zip(vcolq.iter_mut()) {
        let (op, oq) = (*vip, *viq);
        *vip = cs * op - sp * oq;
        *viq = sq * op + cs * oq;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_identity() {
        let m = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let r = truncated_svd(&m, 2, 1e-10);
        assert_eq!(r.s.len(), 2);
        assert!((r.s[0] - 1.0).abs() < 1e-8);
        assert!((r.s[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_svd_rank1() {
        // [[1, 2], [2, 4]] has rank 1, singular values sqrt(25)=5 and 0
        let m = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            vec![Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
        ];
        let r = truncated_svd(&m, 2, 1e-10);
        assert!((r.s[0] - 5.0).abs() < 1e-8, "s[0] = {}", r.s[0]);
        // Second singular value should be ~0 or truncated
        if r.s.len() > 1 {
            assert!(r.s[1] < 1e-8, "s[1] = {}", r.s[1]);
        }
    }

    #[test]
    fn test_svd_reconstruction() {
        // Test that U * diag(S) * Vt ≈ original matrix
        let m = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)],
            vec![Complex64::new(3.0, -1.0), Complex64::new(4.0, 0.0)],
        ];
        let r = truncated_svd(&m, 2, 1e-10);

        // Reconstruct: sum_k s[k] * u[:,k] * vt[k,:]
        let rows = m.len();
        let cols = m[0].len();
        let mut recon = vec![vec![Complex64::new(0.0, 0.0); cols]; rows];
        for k in 0..r.s.len() {
            for i in 0..rows {
                for j in 0..cols {
                    recon[i][j] += r.u[i][k] * r.s[k] * r.vt[k][j];
                }
            }
        }

        for i in 0..rows {
            for j in 0..cols {
                let err = (m[i][j] - recon[i][j]).norm();
                assert!(err < 1e-6, "recon error at [{},{}]: {}", i, j, err);
            }
        }
    }

    // --- one-sided-Jacobi kernel: the properties the old normal-equations path
    // silently violated on deep circuits. ---

    /// Deterministic complex matrix generator (no rng dep in this crate).
    fn seeded_matrix(m: usize, n: usize, seed: u64) -> Vec<Complex64> {
        let mut state = seed;
        let mut next = || {
            // SplitMix64
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let z = z ^ (z >> 31);
            (z as f64) / (u64::MAX as f64) * 2.0 - 1.0
        };
        (0..m * n).map(|_| Complex64::new(next(), next())).collect()
    }

    /// max |(MᴴM − I)| over a `rows × cols` column-orthonormal buffer.
    fn orthonormality_defect(mat: &[Complex64], rows: usize, cols: usize) -> f64 {
        let mut worst = 0.0f64;
        for p in 0..cols {
            for q in 0..cols {
                let mut dot = Complex64::new(0.0, 0.0);
                for i in 0..rows {
                    dot += mat[i * cols + p].conj() * mat[i * cols + q];
                }
                let target = if p == q { 1.0 } else { 0.0 };
                worst = worst.max((dot - Complex64::new(target, 0.0)).norm());
            }
        }
        worst
    }

    fn assert_svd_valid(a: &[Complex64], m: usize, n: usize, seed: u64) {
        let r = truncated_svd_flat(a, m, n, n, m.min(n), 1e-30);
        let k = r.s.len();
        assert_eq!(k, m.min(n), "full rank expected for a generic matrix");
        // U (m×k) and Vᴴ (k×n) both have orthonormal rows/cols → V (n×k) too.
        assert!(
            orthonormality_defect(&r.u, m, k) < 1e-10,
            "UᴴU ≠ I (seed {seed}): defect {}",
            orthonormality_defect(&r.u, m, k)
        );
        // Rows of vt are orthonormal: build V (n×k) as vt conj-transpose.
        let mut vmat = vec![Complex64::new(0.0, 0.0); n * k];
        for kk in 0..k {
            for j in 0..n {
                vmat[j * k + kk] = r.vt[kk * n + j].conj();
            }
        }
        assert!(
            orthonormality_defect(&vmat, n, k) < 1e-10,
            "VᴴV ≠ I (seed {seed})"
        );
        // Reconstruction A ≈ U Σ Vᴴ.
        let mut worst = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = Complex64::new(0.0, 0.0);
                for kk in 0..k {
                    acc += r.u[i * k + kk] * r.s[kk] * r.vt[kk * n + j];
                }
                worst = worst.max((a[i * n + j] - acc).norm());
            }
        }
        assert!(worst < 1e-9, "reconstruction defect {worst} (seed {seed})");
    }

    #[test]
    fn jacobi_svd_is_unitary_tall_and_wide_at_bond_64_shapes() {
        // The exact shapes a χ=64 two-site split produces: 128×96 and 96×128.
        assert_svd_valid(&seeded_matrix(128, 96, 1), 128, 96, 1);
        assert_svd_valid(&seeded_matrix(96, 128, 2), 96, 128, 2);
        // Square and a couple of smaller odd shapes for good measure.
        assert_svd_valid(&seeded_matrix(64, 64, 3), 64, 64, 3);
        assert_svd_valid(&seeded_matrix(40, 12, 4), 40, 12, 4);
        assert_svd_valid(&seeded_matrix(12, 40, 5), 12, 40, 5);
    }

    /// The parallel-rounds contract: results are a FUNCTION OF THE SCHEDULE,
    /// never of the thread count. Disjoint-pair rotations touch disjoint
    /// columns and commute exactly, so 1 thread and 8 threads must produce
    /// bitwise-identical U, S, Vᵀ — the same machine-independent invariance
    /// the statevector kernels are held to (PLAN-SV-PERF S2-S5 split).
    #[test]
    fn thread_count_does_not_change_the_bits() {
        // An odd column count too, so the bye slot is exercised.
        for (m, n, seed) in [(96usize, 64usize, 0xABCDu64), (80, 33, 0x77)] {
            let a = seeded_matrix(m, n, seed);
            let run = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("pool")
                    .install(|| truncated_svd_flat(&a, m, n, n, n, 1e-30))
            };
            let r1 = run(1);
            let r8 = run(8);
            assert_eq!(r1.s.len(), r8.s.len(), "{m}x{n}");
            for (x, y) in r1.s.iter().zip(&r8.s) {
                assert_eq!(x.to_bits(), y.to_bits(), "sigma differs by thread count");
            }
            for (x, y) in r1.u.iter().zip(&r8.u) {
                assert_eq!(x.re.to_bits(), y.re.to_bits(), "U differs by thread count");
                assert_eq!(x.im.to_bits(), y.im.to_bits());
            }
            for (x, y) in r1.vt.iter().zip(&r8.vt) {
                assert_eq!(x.re.to_bits(), y.re.to_bits(), "Vt differs by thread count");
                assert_eq!(x.im.to_bits(), y.im.to_bits());
            }
        }
    }

    /// Convergence at the χ=256 shape the deep-circuit regime actually
    /// produces (512×512): the tournament ordering must still orthogonalise
    /// to the same defect bound as the old cyclic scan.
    #[test]
    fn jacobi_converges_at_the_chi_256_split_shape() {
        assert_svd_valid(&seeded_matrix(512, 512, 6), 512, 512, 6);
    }

    #[test]
    fn discarded_weight_equals_dropped_singular_squares() {
        // Diagonal matrix with a known spectrum: truncating to rank 2 must drop
        // exactly 2² + 1² = 5.
        let s_true = [4.0, 3.0, 2.0, 1.0];
        let n = s_true.len();
        let mut a = vec![Complex64::new(0.0, 0.0); n * n];
        for (i, &sv) in s_true.iter().enumerate() {
            a[i * n + i] = Complex64::new(sv, 0.0);
        }
        let r = truncated_svd_flat(&a, n, n, n, 2, 1e-12);
        assert_eq!(r.s.len(), 2);
        assert!((r.s[0] - 4.0).abs() < 1e-9 && (r.s[1] - 3.0).abs() < 1e-9);
        assert!(
            (r.discarded_weight - 5.0).abs() < 1e-9,
            "discarded_weight = {}",
            r.discarded_weight
        );
    }

    #[test]
    fn discarded_weight_is_zero_when_nothing_truncated() {
        let a = seeded_matrix(20, 8, 7);
        let r = truncated_svd_flat(&a, 20, 8, 8, 8, 1e-30);
        assert!(r.discarded_weight < 1e-12, "{}", r.discarded_weight);
    }
}
