//! Host-callable classical `solve(A, b)` façade.
//!
//! NUMERIC EXTRACTION of the gate-model toolkit's
//! `quantum-core/src/linalg/solver.rs`. The classical reference solver
//! ([`solve`] / [`solve_classical`], partial-pivot LU) ships here — it
//! is pure linear algebra and is the drop-in answer for `A⁻¹ b` plus
//! the LCU scaling factor `α`. The *quantum* circuit recipe
//! (`quantum_solve_circuit`: LCU block-encoding + QSVT inversion) stays
//! in the gate-model toolkit, since it builds an aria `Circuit`; this
//! runtime crate carries no circuit IR. `α` is computed directly from
//! the Pauli 1-norm ([`crate::block_encode::pauli_one_norm`]) rather
//! than from a block-encoding circuit — numerically identical.
//! KEEP-IN-SYNC with the toolkit source (`solve_classical`,
//! `gershgorin_condition_estimate` are byte-identical).

use crate::block_encode::pauli_one_norm;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Result of a [`solve`] call.
#[derive(Clone, Debug)]
pub struct SolveResult {
    /// The solution vector `x = A⁻¹ b`, normalized (unit ℓ²-norm).
    pub x_normalized: Array1<Complex64>,
    /// `‖x‖₂` (the magnitude that the normalized vector hides).
    pub norm: f64,
    /// 1-norm of the Pauli decomposition of `A` (LCU block-encoding
    /// scaling factor).
    pub alpha: f64,
}

/// Solve `A x = b` for a square dense complex matrix `A` (any
/// dimension, not just powers of 2). Uses partial-pivot LU as the
/// classical reference.
pub fn solve_classical(a: &Array2<Complex64>, b: &Array1<Complex64>) -> Array1<Complex64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "A must be square");
    assert_eq!(n, b.len(), "b must have length matching A");

    let mut m = a.clone();
    let mut rhs = b.clone();

    // Partial-pivot Gaussian elimination.
    for k in 0..n {
        // Find pivot row.
        let mut piv = k;
        let mut piv_val = m[(k, k)].norm();
        for i in (k + 1)..n {
            let v = m[(i, k)].norm();
            if v > piv_val {
                piv_val = v;
                piv = i;
            }
        }
        assert!(piv_val > 1e-12, "matrix is singular (or near-singular)");
        if piv != k {
            for c in k..n {
                let tmp = m[(k, c)];
                m[(k, c)] = m[(piv, c)];
                m[(piv, c)] = tmp;
            }
            let tmp = rhs[k];
            rhs[k] = rhs[piv];
            rhs[piv] = tmp;
        }
        // Eliminate below pivot.
        let diag = m[(k, k)];
        for i in (k + 1)..n {
            let factor = m[(i, k)] / diag;
            if factor.norm() == 0.0 {
                continue;
            }
            for c in k..n {
                let v = m[(k, c)];
                m[(i, c)] -= factor * v;
            }
            let v = rhs[k];
            rhs[i] -= factor * v;
        }
    }

    // Back-substitute.
    let mut x: Array1<Complex64> = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= m[(i, j)] * x[j];
        }
        x[i] = sum / m[(i, i)];
    }
    x
}

/// High-level façade matching the TODO item-4(c) signature:
/// `solve(A, b, ε) → (normalized direction, magnitude, α)`.
///
/// Uses the classical solver internally; the `ε` argument is reserved
/// for future use when a real QSVT-derived solution is wired in (it
/// will then bound the polynomial-approximation error of `1/x`).
pub fn solve(a: &Array2<Complex64>, b: &Array1<Complex64>, _eps: f64) -> SolveResult {
    let x = solve_classical(a, b);
    let norm: f64 = x.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    let x_normalized = if norm > 0.0 {
        x.mapv(|c| c / Complex64::new(norm, 0.0))
    } else {
        x.clone()
    };

    // For dimension ≥ 2 with power-of-two size, also report α (the LCU
    // block-encoding 1-norm) so callers can size ancillas / estimate
    // success probability for the quantum recipe. Computed from the
    // Pauli decomposition directly (no circuit needed).
    let alpha = if a.nrows().is_power_of_two() && a.nrows() >= 2 {
        pauli_one_norm(a)
    } else {
        f64::NAN
    };

    SolveResult {
        x_normalized,
        norm,
        alpha,
    }
}

/// Gershgorin-disk condition-number estimate. Crude, but only needs
/// `O(N²)` work. For diagonally-dominant matrices it's tight; for
/// general matrices it's an upper bound for κ. Used by the gate-model
/// toolkit's QSVT-degree heuristic; exposed here for host callers.
pub fn gershgorin_condition_estimate(a: &Array2<Complex64>) -> f64 {
    let n = a.nrows();
    let (mut max_r, mut min_r) = (f64::MIN, f64::MAX);
    for i in 0..n {
        let diag = a[(i, i)].norm();
        let off: f64 = (0..n).filter(|&j| j != i).map(|j| a[(i, j)].norm()).sum();
        let upper = diag + off;
        let lower = (diag - off).max(1e-6);
        if upper > max_r {
            max_r = upper;
        }
        if lower < min_r {
            min_r = lower;
        }
    }
    (max_r / min_r).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    fn cmplx(r: f64) -> Complex64 {
        Complex64::new(r, 0.0)
    }

    #[test]
    fn solve_diagonal() {
        let a: Array2<Complex64> = array![[cmplx(1.0), cmplx(0.0)], [cmplx(0.0), cmplx(2.0)]];
        let b: Array1<Complex64> = array![cmplx(1.0), cmplx(4.0)];
        let x = solve_classical(&a, &b);
        assert!((x[0].re - 1.0).abs() < 1e-9);
        assert!((x[1].re - 2.0).abs() < 1e-9);
    }

    #[test]
    fn solve_general_4x4() {
        // Well-conditioned random 4×4 (chosen by hand).
        let a: Array2<Complex64> = array![
            [cmplx(4.0), cmplx(1.0), cmplx(0.0), cmplx(0.0)],
            [cmplx(1.0), cmplx(3.0), cmplx(1.0), cmplx(0.0)],
            [cmplx(0.0), cmplx(1.0), cmplx(3.0), cmplx(1.0)],
            [cmplx(0.0), cmplx(0.0), cmplx(1.0), cmplx(2.0)],
        ];
        let b: Array1<Complex64> = array![cmplx(1.0), cmplx(2.0), cmplx(3.0), cmplx(4.0)];
        let x = solve_classical(&a, &b);
        // Verify A·x = b.
        let ax: Array1<Complex64> = a.dot(&x);
        for i in 0..4 {
            assert!(
                (ax[i] - b[i]).norm() < 1e-9,
                "row {i}: ax={}, b={}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn solve_returns_normalized_with_norm() {
        let a: Array2<Complex64> = array![[cmplx(2.0), cmplx(0.0)], [cmplx(0.0), cmplx(2.0)]];
        let b: Array1<Complex64> = array![cmplx(2.0), cmplx(0.0)];
        let res = solve(&a, &b, 1e-3);
        // Solution is x = (1, 0), norm = 1.
        assert!((res.norm - 1.0).abs() < 1e-9);
        assert!((res.x_normalized[0].re - 1.0).abs() < 1e-9);
        assert!(res.x_normalized[1].norm() < 1e-9);
        assert!(res.alpha.is_finite());
        // α = 1.5·I-part 1-norm: diag(2,2) = 2·I → α = 2.0.
        assert!((res.alpha - 2.0).abs() < 1e-9);
    }

    #[test]
    fn solve_complex_2x2() {
        // A = [[1, i/2], [-i/2, 1]] (Hermitian, eigenvalues 1±1/2).
        let a: Array2<Complex64> = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.5)],
            [Complex64::new(0.0, -0.5), Complex64::new(1.0, 0.0)],
        ];
        let b: Array1<Complex64> = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let x = solve_classical(&a, &b);
        let ax = a.dot(&x);
        for i in 0..2 {
            assert!((ax[i] - b[i]).norm() < 1e-9);
        }
    }

    #[test]
    fn gershgorin_estimate_diagonal_is_kappa() {
        // diag(1, 2): κ = 2/1 = 2.
        let a: Array2<Complex64> = array![[cmplx(1.0), cmplx(0.0)], [cmplx(0.0), cmplx(2.0)]];
        assert!((gershgorin_condition_estimate(&a) - 2.0).abs() < 1e-9);
    }
}
