//! Truncated SVD for MPS bond compression.
//!
//! Implements a simple SVD via Jacobi one-sided rotation (suitable for
//! small matrices typical in MPS bonds, bond_dim <= 64).
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
pub struct SvdResultFlat {
    pub u: Vec<Complex64>,
    pub s: Vec<f64>,
    pub vt: Vec<Complex64>,
    pub m: usize,
    pub n: usize,
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
        };
    }
    debug_assert!(stride >= n, "stride {stride} < n {n}");
    debug_assert!(matrix.len() >= (m - 1) * stride + n);

    // For small matrices, use full SVD via eigendecomposition of A†A
    // A†A is n×n Hermitian positive semi-definite. Reordered loops
    // (k outer, i,j inner) so the hot inner pair of loads hits row k
    // sequentially — cache-friendlier than the (i,j outer, k inner)
    // form for typical MPS shapes (n ≪ m × stride).
    let mut ata = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for k in 0..m {
        let row = &matrix[k * stride..k * stride + n];
        for i in 0..n {
            let conj_i = row[i].conj();
            let ata_i = &mut ata[i];
            for j in 0..n {
                ata_i[j] += conj_i * row[j];
            }
        }
    }

    // Eigendecomposition of A†A via Jacobi iteration
    let (eigenvalues, eigenvectors) = hermitian_eigen(&ata, n);

    // Singular values = sqrt(eigenvalues), sorted descending
    let mut sv_pairs: Vec<(f64, usize)> = eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &ev)| (ev.max(0.0).sqrt(), i))
        .collect();
    sv_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Truncate
    let rank = sv_pairs
        .iter()
        .take(max_rank)
        .take_while(|(sv, _)| *sv > threshold)
        .count()
        .max(1); // Keep at least 1

    let s: Vec<f64> = sv_pairs[..rank].iter().map(|(sv, _)| *sv).collect();

    // V columns (right singular vectors) = eigenvectors of A†A
    // vt[k_idx][j] = eigenvectors[j][idx_k].conj(); flat row-major.
    let mut vt = vec![Complex64::new(0.0, 0.0); rank * n];
    for k in 0..rank {
        let idx = sv_pairs[k].1;
        for j in 0..n {
            vt[k * n + j] = eigenvectors[j][idx].conj();
        }
    }

    // U = A * V * S^{-1}; flat row-major, u[i * rank + k].
    let mut u = vec![Complex64::new(0.0, 0.0); m * rank];
    for i in 0..m {
        let row = &matrix[i * stride..i * stride + n];
        for k in 0..rank {
            if s[k] < 1e-15 {
                continue;
            }
            let idx = sv_pairs[k].1;
            let mut sum = Complex64::new(0.0, 0.0);
            for j in 0..n {
                sum += row[j] * eigenvectors[j][idx];
            }
            u[i * rank + k] = sum / s[k];
        }
    }

    SvdResultFlat { u, s, vt, m, n }
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

/// Eigendecomposition of a Hermitian matrix via Jacobi iteration.
/// Returns (eigenvalues, eigenvectors) where eigenvectors[i][j] is the
/// j-th component of the i-th eigenvector.
fn hermitian_eigen(matrix: &[Vec<Complex64>], n: usize) -> (Vec<f64>, Vec<Vec<Complex64>>) {
    if n == 1 {
        return (vec![matrix[0][0].re], vec![vec![Complex64::new(1.0, 0.0)]]);
    }

    // Work with a copy
    let mut a = matrix.to_vec();
    let mut v = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        v[i][i] = Complex64::new(1.0, 0.0);
    }

    // Jacobi iteration
    for _sweep in 0..100 {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i][j].norm();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-14 {
            break;
        }

        // Compute rotation to zero a[p][q]
        let apq = a[p][q];
        let app = a[p][p].re;
        let aqq = a[q][q].re;

        // Phase to make apq real
        let phase = if apq.norm() > 1e-30 {
            apq / apq.norm()
        } else {
            Complex64::new(1.0, 0.0)
        };
        let apq_real = apq.norm();

        // Jacobi angle
        let tau = (app - aqq) / (2.0 * apq_real);
        let t = if tau.abs() < 1e30 {
            let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
            sign / (tau.abs() + (1.0 + tau * tau).sqrt())
        } else {
            0.0
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Apply rotation: rows/cols p, q
        // With phase correction for complex case
        let phase_conj = phase.conj();

        for i in 0..n {
            let aip = a[i][p];
            let aiq = a[i][q];
            a[i][p] = Complex64::new(c, 0.0) * aip + Complex64::new(s, 0.0) * aiq * phase_conj;
            a[i][q] = -Complex64::new(s, 0.0) * aip * phase + Complex64::new(c, 0.0) * aiq;
        }
        for j in 0..n {
            let apj = a[p][j];
            let aqj = a[q][j];
            a[p][j] = Complex64::new(c, 0.0) * apj + Complex64::new(s, 0.0) * aqj * phase;
            a[q][j] = -Complex64::new(s, 0.0) * apj * phase_conj + Complex64::new(c, 0.0) * aqj;
        }

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = Complex64::new(c, 0.0) * vip + Complex64::new(s, 0.0) * viq * phase_conj;
            v[i][q] = -Complex64::new(s, 0.0) * vip * phase + Complex64::new(c, 0.0) * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i].re).collect();
    (eigenvalues, v)
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
}
