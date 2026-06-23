// SPDX-License-Identifier: Apache-2.0
//! Quantum Singular Value Transformation (QSVT).
//!
//! QSVT applies a polynomial transformation to the singular values
//! of a block-encoded matrix. It's the foundation for optimal
//! quantum algorithms for linear algebra, search, and simulation.
//!
//! Given a block encoding of A with angles [φ₁, ..., φ_d]:
//! QSVT circuit = ∏ᵢ (signal_rotation(φᵢ) · signal_processing_unitary)
//!
//! The result block-encodes P(A) where P is a polynomial of degree d.
//!
//! Ported from the gate-model toolkit's `quantum-core/src/circuits/qsvt.rs`
//! (KEEP-IN-SYNC; only the `crate::ast` → `aria_core::ast` import differs).

use aria_core::ast::{Circuit, CircuitBuilder};
use std::f64::consts::PI;

/// Build a QSVT circuit for polynomial transformation.
///
/// - `n_system`: system qubits (block encoding uses 1 ancilla + n system)
/// - `angles`: QSVT phase angles [φ₁, ..., φ_d] defining the polynomial
///
/// The ancilla qubit is qubit 0, system qubits are 1..n.
pub fn qsvt_circuit(n_system: usize, angles: &[f64]) -> Circuit {
    let n_total = 1 + n_system; // ancilla + system
    let mut b = CircuitBuilder::new("qsvt", n_total, 0);

    for (i, &phi) in angles.iter().enumerate() {
        // Signal rotation on ancilla: e^{i*φ*Z}
        b.rz(0, 2.0 * phi);

        // Signal processing: controlled operation (placeholder)
        // In practice, this is the block encoding unitary or its complement
        if i % 2 == 0 {
            // "Even" step: apply block encoding
            // For a simple example, use controlled-RY as placeholder
            b.cx(0, 1);
        } else {
            // "Odd" step: apply conjugate block encoding
            b.cx(0, 1);
        }

        // Reflection on ancilla
        if i < angles.len() - 1 {
            b.h(0);
            b.z(0);
            b.h(0);
        }
    }

    b.build()
}

/// Compute QSVT angles for a simple polynomial.
///
/// For a degree-d polynomial approximation of sign(x):
/// angles ≈ [π/4, 0, ..., 0, π/4] (simplified).
// Index-loop bodies kept byte-identical to the toolkit source; OSS ci.sh
// runs clippy with `-D warnings`, so the range-loop lint is allowed locally.
#[allow(clippy::needless_range_loop)]
pub fn sign_function_angles(degree: usize) -> Vec<f64> {
    let mut angles = vec![0.0; degree + 1];
    angles[0] = PI / 4.0;
    angles[degree] = PI / 4.0;
    // Interior angles for the Chebyshev approximation
    for i in 1..degree {
        angles[i] = PI / (2.0 * degree as f64);
    }
    angles
}

/// Compute QSVT angles for matrix inversion (1/x).
///
/// Approximates 1/x on [1/κ, 1] where κ is the condition number.
#[allow(clippy::needless_range_loop)]
pub fn inversion_angles(degree: usize, kappa: f64) -> Vec<f64> {
    let mut angles = vec![0.0; degree + 1];
    // Simplified: use uniform angles scaled by 1/κ
    for i in 0..=degree {
        angles[i] = PI / (4.0 * kappa) * (if i % 2 == 0 { 1.0 } else { -1.0 });
    }
    angles[0] = PI / 4.0;
    angles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qsvt_circuit() {
        let angles = sign_function_angles(3);
        let circ = qsvt_circuit(1, &angles);
        assert_eq!(circ.n_qubits(), 2); // 1 ancilla + 1 system
        assert!(circ.gate_count() > 0);
    }

    #[test]
    fn test_sign_angles() {
        let angles = sign_function_angles(5);
        assert_eq!(angles.len(), 6);
        assert!((angles[0] - PI / 4.0).abs() < 1e-10);
        assert!((angles[5] - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_inversion_angles() {
        let angles = inversion_angles(7, 10.0);
        assert_eq!(angles.len(), 8);
    }

    #[test]
    fn test_qsvt_resources() {
        // QSVT with degree d uses O(d) gates
        for d in [3, 7, 15] {
            let angles = sign_function_angles(d);
            let circ = qsvt_circuit(1, &angles);
            // Gate count should scale linearly with degree
            assert!(circ.gate_count() >= d);
        }
    }
}
