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
use crate::linalg::qsvt::{inversion_angles, qsvt_circuit};
use aria_core::ast::Circuit;
use ndarray::Array2;
use num_complex::Complex64;
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
/// The QSVT phases come from the real angle finder
/// [`crate::linalg::qsvt::inversion_angles`] (Wang–Lin least-squares over the
/// exact QSP response), and the circuit interleaves them with the block-encoding
/// signal. Overall numeric fidelity is still bounded by the toy LCU SELECT in
/// [`block_encode_dense`], so this remains a resource-estimation recipe; the
/// angle/QSVT layer itself is numerically validated in the `qsvt` tests.
pub fn quantum_solve_circuit(
    a: &Array2<Complex64>,
    eps: f64,
) -> (Circuit, DenseBlockEncoding, f64, usize) {
    let dense_be = block_encode_dense(a);

    // Crude condition-number estimate via Gershgorin disks: the
    // spectrum lies in the union of disks centered at A[i,i] with
    // radius Σ_{j≠i} |A[i,j]|. Use this to bound κ ≤ max/min radius
    // ratio. For real applications, callers should pass κ directly.
    let kappa = gershgorin_condition_estimate(a);

    // Heuristic polynomial degree from arXiv:1511.02306 (Childs–Kothari–
    // Somma 2017): d = O(κ · log(κ/ε)). Cap at 64 for sanity.
    let d_raw = (kappa * (kappa / eps.max(1e-9)).ln()).max(4.0).round() as usize;
    let degree = d_raw.min(64);

    let phases = inversion_angles(degree, kappa);
    // The block encoding spans `n_ancilla + n_system` qubits; drive QSVT over the
    // same register so the composed signal is well-formed (ancilla = qubit 0).
    let n_qsvt_system = dense_be.circuit.n_qubits().saturating_sub(1);
    let circuit = qsvt_circuit(n_qsvt_system, &phases, &dense_be.circuit);
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
