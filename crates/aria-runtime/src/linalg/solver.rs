// SPDX-License-Identifier: Apache-2.0
//! Quantum circuit recipe for `A x = b` (LCU block-encoding + QSVT inversion).
//!
//! CIRCUIT half of the gate-model toolkit's
//! `quantum-core/src/linalg/solver.rs` (the classical reference solver —
//! `solve` / `solve_classical` — lives in [`omega_core::solver`]).
//! [`quantum_solve_circuit`] builds the QSVT circuit applying the
//! polynomial approximation of `1/x` to the block-encoded `A/α`.
//! KEEP-IN-SYNC with the toolkit source.

use crate::linalg::block_encode::{block_encode_dense, DenseBlockEncoding};
use crate::linalg::qsvt::qsvt_circuit;
use aria_core::ast::Circuit;
use ndarray::Array2;
use num_complex::Complex64;
use omega_core::chebyshev::qsvt_inversion_angles;
use omega_core::solver::gershgorin_condition_estimate;

/// Build the *quantum* circuit recipe for `A x = b`.
///
/// Returns `(qsvt_circ, dense_be, condition_number, qsvt_degree)`:
///
/// - `qsvt_circ` — the QSVT circuit applying the polynomial
///   approximation of `1/x` to the block-encoded `A/α`.
/// - `dense_be` — the LCU block encoding of `A` (gives access to the
///   Pauli terms and the block-encoding circuit separately).
/// - `condition_number` — singular-value bound used to set the
///   approximation domain `[1/κ, 1]`. Computed from a Gershgorin-disk
///   estimate; a real implementation would use SVD.
/// - `qsvt_degree` — polynomial degree used (heuristic from κ and ε).
///
/// Suitable for resource estimation; circuit accuracy depends on
/// the placeholder `qsvt_circuit` until the Wang-Lin angle reduction
/// is wired in (see `omega_core::chebyshev::qsvt_inversion_angles` doc).
pub fn quantum_solve_circuit(
    a: &Array2<Complex64>,
    eps: f64,
) -> (Circuit, DenseBlockEncoding, f64, usize) {
    let dense_be = block_encode_dense(a);
    let n_system = (a.nrows() as f64).log2().round() as usize;

    // Crude condition-number estimate via Gershgorin disks: the
    // spectrum lies in the union of disks centered at A[i,i] with
    // radius Σ_{j≠i} |A[i,j]|. Use this to bound κ ≤ max/min radius
    // ratio. For real applications, callers should pass κ directly.
    let kappa = gershgorin_condition_estimate(a);

    // Heuristic polynomial degree from arXiv:1511.02306 (Childs–Kothari–
    // Somma 2017): d = O(κ · log(κ/ε)). Cap at 64 for sanity.
    let d_raw = (kappa * (kappa / eps.max(1e-9)).ln()).max(4.0).round() as usize;
    let degree = d_raw.min(64);

    let angles = qsvt_inversion_angles(degree, kappa);
    let circuit = qsvt_circuit(n_system, &angles);
    (circuit, dense_be, kappa, degree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn cmplx(r: f64) -> Complex64 {
        Complex64::new(r, 0.0)
    }

    #[test]
    fn quantum_solve_circuit_builds() {
        let a: Array2<Complex64> = array![[cmplx(1.0), cmplx(0.0)], [cmplx(0.0), cmplx(2.0)]];
        let (circ, _be, kappa, degree) = quantum_solve_circuit(&a, 1e-3);
        assert!(kappa >= 1.0);
        assert!(degree >= 4);
        assert!(circ.gate_count() > 0);
        // QSVT uses 1 ancilla + n_system qubits.
        assert_eq!(circ.n_qubits(), 1 + 1);
    }
}
