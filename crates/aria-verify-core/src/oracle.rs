// SPDX-License-Identifier: Apache-2.0
//! Pure-Rust classical oracles — the "ground truth" each quantum example is
//! checked against. No LAPACK / nalgebra: small textbook routines only, so the
//! default build stays dependency-clean. These deliberately use a DIFFERENT
//! method from the quantum side (direct linear algebra / brute force) so that
//! agreement is meaningful rather than circular.

use num_complex::Complex64;

/// Symmetric real eigenvalues via the cyclic Jacobi rotation method.
/// Returns the eigenvalues sorted descending. `mat` must be square symmetric.
#[allow(clippy::needless_range_loop)]
pub fn jacobi_eigenvalues(mat: &[Vec<f64>]) -> Vec<f64> {
    let n = mat.len();
    let mut a: Vec<Vec<f64>> = mat.to_vec();
    // Cyclic Jacobi sweeps. For the tiny matrices here this converges to
    // machine precision in a handful of sweeps; cap generously.
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                // Rotation angle that zeroes a[p][q].
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Apply the Givens rotation J^T A J to rows/cols p,q.
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut eig: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eig.sort_by(|x, y| y.partial_cmp(x).unwrap());
    eig
}

/// Singular values of a real matrix `m` (rows × cols), sorted descending.
/// Computed classically as `sqrt(eig(mᵀ m))` — the textbook definition.
#[allow(clippy::needless_range_loop)]
pub fn singular_values(m: &[Vec<f64>]) -> Vec<f64> {
    let rows = m.len();
    let cols = m[0].len();
    // gram = mᵀ m  (cols × cols, symmetric PSD)
    let mut gram = vec![vec![0.0; cols]; cols];
    for (i, gram_row) in gram.iter_mut().enumerate() {
        for (j, gram_ij) in gram_row.iter_mut().enumerate() {
            let mut s = 0.0;
            for k in 0..rows {
                s += m[k][i] * m[k][j];
            }
            *gram_ij = s;
        }
    }
    jacobi_eigenvalues(&gram)
        .into_iter()
        .map(|e| e.max(0.0).sqrt())
        .collect()
}

/// Pauli decomposition of a 2×2 Hermitian matrix `h` into real coefficients
/// `(a, bx, by, bz)` such that `h = a·I + bx·X + by·Y + bz·Z`.
/// `h[i][j]` are complex; for a Hermitian input the returned coeffs are real.
pub fn pauli_decompose_2x2(h: &[[Complex64; 2]; 2]) -> (f64, f64, f64, f64) {
    let a = (h[0][0].re + h[1][1].re) / 2.0;
    let bz = (h[0][0].re - h[1][1].re) / 2.0;
    let bx = (h[0][1].re + h[1][0].re) / 2.0;
    // X·(off-diag): h01 = bx - i·by  ⇒  by = Im(h10) = -Im(h01).
    let by = (h[1][0].im - h[0][1].im) / 2.0;
    (a, bx, by, bz)
}

/// Certified Neumann approximant of `1/x`: `p_d(x) = Σ_{k=0}^{d} (1−x)^k`,
/// evaluated by Horner on `r = 1−x`. This is the polynomial whose error is
/// proven sorry-free in `proofs/lean4/QuantumProofs/QSVT.lean` (`invPoly`,
/// `inv_poly_approx`) — the QSVT-inversion building block backing the shipped
/// `qsvt_invert.aria` / `hhl.aria` examples. Exact identity:
/// `p_d(x) = (1 − (1−x)^{d+1})/x`.
pub fn neumann_inv_poly(degree: usize, x: f64) -> f64 {
    let r = 1.0 - x;
    let mut acc = 1.0;
    for _ in 0..degree {
        acc = 1.0 + r * acc;
    }
    acc
}

/// The proven worst-case error of [`neumann_inv_poly`] on `[1/κ, 1]`:
/// `(1 − 1/κ)^{d+1} / (1/κ)`. Matches the Lean bound `(1−δ)^{d+1}/δ`
/// (`QSVT.inv_poly_approx`), saturated at the spectral edge `x = 1/κ`.
pub fn neumann_inv_error_bound(degree: usize, kappa: f64) -> f64 {
    let delta = 1.0 / kappa;
    (1.0 - delta).powi(degree as i32 + 1) / delta
}

/// Apply the n-qubit DFT matrix to a computational basis state `|x⟩`.
/// Returns the amplitude vector of `QFT|x⟩` with the SAME normalization and
/// bit-ordering convention as the omega statevector backend: amplitude of
/// `|k⟩` is `exp(2πi·x·k / N) / sqrt(N)`, `N = 2^n`.
pub fn qft_amplitudes(n: u32, x: u64) -> Vec<Complex64> {
    let dim = 1usize << n;
    let norm = 1.0 / (dim as f64).sqrt();
    let two_pi = std::f64::consts::TAU;
    (0..dim)
        .map(|k| {
            let phase = two_pi * (x as f64) * (k as f64) / (dim as f64);
            Complex64::from_polar(norm, phase)
        })
        .collect()
}

/// Brute-force MaxCut value of an undirected graph on `n` nodes with unit
/// edges. Returns `(best_cut_value, best_partition_bitmask)`.
pub fn brute_force_maxcut(n: usize, edges: &[(usize, usize)]) -> (usize, u64) {
    let mut best = 0usize;
    let mut best_mask = 0u64;
    for mask in 0u64..(1u64 << n) {
        let mut cut = 0usize;
        for &(i, j) in edges {
            if ((mask >> i) & 1) != ((mask >> j) & 1) {
                cut += 1;
            }
        }
        if cut > best {
            best = cut;
            best_mask = mask;
        }
    }
    (best, best_mask)
}

/// Smallest eigenvalue of a symmetric matrix.
pub fn min_eigenvalue(mat: &[Vec<f64>]) -> f64 {
    *jacobi_eigenvalues(mat).last().unwrap()
}

/// 2×2 Pauli matrix for `'I' | 'X' | 'Y' | 'Z'`.
fn pauli_2x2(p: char) -> [[Complex64; 2]; 2] {
    let z = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    match p {
        'I' => [[one, z], [z, one]],
        'X' => [[z, one], [one, z]],
        'Y' => [[z, -i], [i, z]],
        'Z' => [[one, z], [z, -one]],
        _ => panic!("unknown Pauli {p}"),
    }
}

/// Dense matrix of a Pauli-sum Hamiltonian on `n_qubits`. Each term is
/// `(coeff, [(qubit, pauli_char), ...])`; unlisted qubits get identity.
/// Qubit 0 is the least-significant index (matches the omega statevector).
pub fn pauli_sum_matrix(
    n_qubits: usize,
    terms: &[(f64, Vec<(usize, char)>)],
) -> Vec<Vec<Complex64>> {
    let dim = 1usize << n_qubits;
    let mut h = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    for (coeff, ops) in terms {
        // Per-qubit Pauli letter (default identity).
        let mut letters = vec!['I'; n_qubits];
        for &(q, p) in ops {
            letters[q] = p;
        }
        // Tensor product over qubits; qubit 0 is the LSB of the row/col index.
        for (row, h_row) in h.iter_mut().enumerate() {
            for (col, h_rc) in h_row.iter_mut().enumerate() {
                let mut amp = Complex64::new(*coeff, 0.0);
                for (q, letter) in letters.iter().enumerate() {
                    let r = (row >> q) & 1;
                    let c = (col >> q) & 1;
                    amp *= pauli_2x2(*letter)[r][c];
                    if amp == Complex64::new(0.0, 0.0) {
                        break;
                    }
                }
                *h_rc += amp;
            }
        }
    }
    h
}

/// Ground-state (minimum) energy of a Hermitian Pauli-sum Hamiltonian. The
/// matrix is real-symmetric for the Hamiltonians used here (even Y count per
/// term), so we diagonalize its real part with Jacobi.
pub fn ground_state_energy(n_qubits: usize, terms: &[(f64, Vec<(usize, char)>)]) -> f64 {
    let h = pauli_sum_matrix(n_qubits, terms);
    let real: Vec<Vec<f64>> = h.iter().map(|r| r.iter().map(|c| c.re).collect()).collect();
    min_eigenvalue(&real)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_matches_analytic_2x2() {
        // [[2,1],[1,3]] has eigenvalues (5 ± √5)/2.
        let m = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let ev = jacobi_eigenvalues(&m);
        let hi = (5.0 + 5.0_f64.sqrt()) / 2.0;
        let lo = (5.0 - 5.0_f64.sqrt()) / 2.0;
        assert!((ev[0] - hi).abs() < 1e-12, "{ev:?}");
        assert!((ev[1] - lo).abs() < 1e-12, "{ev:?}");
    }

    #[test]
    fn singular_values_symmetric() {
        // For a symmetric PD matrix, singular values == eigenvalues.
        let m = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let sv = singular_values(&m);
        let hi = (5.0 + 5.0_f64.sqrt()) / 2.0;
        let lo = (5.0 - 5.0_f64.sqrt()) / 2.0;
        assert!((sv[0] - hi).abs() < 1e-10, "{sv:?}");
        assert!((sv[1] - lo).abs() < 1e-10, "{sv:?}");
    }

    #[test]
    fn maxcut_triangle_is_two() {
        let (cut, _) = brute_force_maxcut(3, &[(0, 1), (1, 2), (0, 2)]);
        assert_eq!(cut, 2);
    }

    #[test]
    fn neumann_inverter_matches_lean_qsvt() {
        // Cross-check against proofs/lean4/QuantumProofs/QSVT.lean Numeric:
        // κ=2, d=8 ⇒ p_8(1/2)=511/256, bound=1/256, saturated at x=δ.
        let p = neumann_inv_poly(8, 0.5);
        assert!((p - 511.0 / 256.0).abs() < 1e-12, "p_8(1/2)={p}");
        let bound = neumann_inv_error_bound(8, 2.0);
        assert!((bound - 1.0 / 256.0).abs() < 1e-12, "bound={bound}");
        // exact identity p_d(x) = (1-(1-x)^{d+1})/x, and bound holds on [δ,1].
        for i in 0..=20 {
            let x = 0.5 + 0.5 * i as f64 / 20.0;
            let id = (1.0 - (1.0 - x).powi(9)) / x;
            assert!((neumann_inv_poly(8, x) - id).abs() < 1e-12);
            let err = (1.0 / x - neumann_inv_poly(8, x)).abs();
            assert!(err <= bound + 1e-12, "x={x} err={err} bound={bound}");
        }
    }
}
