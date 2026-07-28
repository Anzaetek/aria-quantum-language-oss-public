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
fn one_sided_jacobi(
    a: &[Complex64],
    rows: usize,
    cols: usize,
) -> (Vec<Complex64>, Vec<f64>, Vec<Complex64>) {
    let mut b = a.to_vec(); // working copy; its columns get rotated into U
    let mut v = vec![Complex64::new(0.0, 0.0); cols * cols];
    for i in 0..cols {
        v[i * cols + i] = Complex64::new(1.0, 0.0);
    }

    // Cyclic sweeps over every column pair. Converged when a whole sweep rotates
    // nothing — the correct criterion, unlike the old fixed 100-rotation cap
    // that could not diagonalise a 128-column matrix. One-sided Jacobi is
    // GLOBALLY convergent (each rotation strictly reduces the off-diagonal Gram
    // norm and never increases it), and converges quadratically — empirically
    // ~6-10 sweeps for matrices this size. 60 is therefore a safety ceiling with
    // large margin, not a hopeful cap; converged inputs break out early via
    // `rotated == false`, so the extra headroom is free.
    let tol = 1e-14;
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..cols {
            for q in (p + 1)..cols {
                // Gram entries of columns p, q of the working matrix.
                let (mut app, mut aqq) = (0.0f64, 0.0f64);
                let mut apq = Complex64::new(0.0, 0.0);
                for i in 0..rows {
                    let bip = b[i * cols + p];
                    let biq = b[i * cols + q];
                    app += bip.norm_sqr();
                    aqq += biq.norm_sqr();
                    apq += bip.conj() * biq;
                }
                let apq_abs = apq.norm();
                if apq_abs < 1e-300 || apq_abs <= tol * (app * aqq).sqrt() {
                    continue; // columns already orthogonal
                }
                rotated = true;
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
                for i in 0..rows {
                    let bip = b[i * cols + p];
                    let biq = b[i * cols + q];
                    b[i * cols + p] = cs * bip - sp * biq;
                    b[i * cols + q] = sq * bip + cs * biq;
                }
                for i in 0..cols {
                    let vip = v[i * cols + p];
                    let viq = v[i * cols + q];
                    v[i * cols + p] = cs * vip - sp * viq;
                    v[i * cols + q] = sq * vip + cs * viq;
                }
            }
        }
        if !rotated {
            break;
        }
    }

    // Column norms are the singular values; normalise columns to form U.
    let mut s = vec![0.0f64; cols];
    for j in 0..cols {
        let mut nrm = 0.0f64;
        for i in 0..rows {
            nrm += b[i * cols + j].norm_sqr();
        }
        s[j] = nrm.sqrt();
    }
    let mut u = b;
    for j in 0..cols {
        if s[j] > 1e-300 {
            let inv = 1.0 / s[j];
            for i in 0..rows {
                u[i * cols + j] *= inv;
            }
        }
        // A zero column (σ ≈ 0) is left zero; it is truncated by the caller.
    }
    (u, s, v)
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
