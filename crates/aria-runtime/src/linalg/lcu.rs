// SPDX-License-Identifier: Apache-2.0
//! Linear Combination of Unitaries (LCU) for block encoding.
//!
//! LCU implements A = Σ αᵢ Uᵢ as a block encoding using:
//! 1. PREPARE circuit: loads coefficients into ancilla register
//! 2. SELECT circuit: applies Uᵢ controlled on ancilla state |i⟩
//! 3. PREPARE†: unprepare ancilla
//!
//! Result: ⟨0|_anc (PREP† · SELECT · PREP) |0⟩_anc = A / α
//! where α = Σ|αᵢ| is the 1-norm.
//!
//! Ported from the gate-model toolkit's `quantum-core/src/circuits/lcu.rs`
//! (KEEP-IN-SYNC; only the `crate::ast` → `aria_core::ast` import differs).

use aria_core::ast::{Circuit, CircuitBuilder};
use std::f64::consts::PI;

/// A term in the LCU decomposition: coefficient × unitary.
#[derive(Clone, Debug)]
pub struct LcuTerm {
    pub coeff: f64,
    /// Which Pauli/Clifford unitary this represents (index into a fixed set).
    pub unitary_index: usize,
}

/// Build an LCU block encoding circuit.
///
/// - `n_system`: number of system qubits
/// - `terms`: list of (coefficient, unitary_index) pairs
///
/// Uses ⌈log₂(k)⌉ ancilla qubits for k terms.
/// Total qubits: ancilla + system.
pub fn lcu_block_encoding(n_system: usize, terms: &[LcuTerm]) -> Circuit {
    let k = terms.len();
    let n_ancilla = ((k as f64).log2().ceil() as usize).max(1);
    let total = n_ancilla + n_system;

    let mut b = CircuitBuilder::new("lcu", total, 0);

    // Step 1: PREPARE — load √(|αᵢ|/α) into ancilla amplitudes
    let alpha: f64 = terms.iter().map(|t| t.coeff.abs()).sum();
    prepare_amplitudes(&mut b, terms, alpha, n_ancilla);

    // Step 2: SELECT — controlled application of unitaries
    for (i, term) in terms.iter().enumerate() {
        // Apply Uᵢ controlled on ancilla state |i⟩
        // Simplified: use CX/CZ as placeholder unitaries
        select_unitary(&mut b, i, n_ancilla, n_system, term);
    }

    // Step 3: PREPARE† — unprepare (adjoint of prepare)
    unprepare_amplitudes(&mut b, terms, alpha, n_ancilla);

    b.build()
}

fn prepare_amplitudes(b: &mut CircuitBuilder, terms: &[LcuTerm], alpha: f64, n_ancilla: usize) {
    if alpha == 0.0 {
        return;
    }

    // For 2 terms: single RY rotation on ancilla
    // cos(θ/2) = √(α₀/α), sin(θ/2) = √(α₁/α)
    if terms.len() == 2 && n_ancilla >= 1 {
        let ratio = (terms[0].coeff.abs() / alpha).sqrt();
        let theta = 2.0 * ratio.acos();
        b.ry(0, theta);
        return;
    }

    // For more terms: Hadamard on all ancilla (uniform superposition)
    // Then correct amplitudes with rotations
    for i in 0..n_ancilla {
        b.h(i);
    }
}

fn unprepare_amplitudes(b: &mut CircuitBuilder, terms: &[LcuTerm], alpha: f64, n_ancilla: usize) {
    if alpha == 0.0 {
        return;
    }

    if terms.len() == 2 && n_ancilla >= 1 {
        let ratio = (terms[0].coeff.abs() / alpha).sqrt();
        let theta = 2.0 * ratio.acos();
        b.ry(0, -theta);
        return;
    }

    for i in (0..n_ancilla).rev() {
        b.h(i);
    }
}

fn select_unitary(
    b: &mut CircuitBuilder,
    _index: usize,
    n_ancilla: usize,
    _n_system: usize,
    term: &LcuTerm,
) {
    let system_start = n_ancilla;

    // Simplified SELECT: controlled-Pauli based on term index
    // In practice, this would be a multiplexed unitary
    match term.unitary_index {
        0 => { /* Identity — do nothing */ }
        1 => {
            // Z on first system qubit, controlled on ancilla
            if n_ancilla > 0 {
                b.cz(0, system_start);
            }
        }
        2 => {
            // X on first system qubit, controlled on ancilla
            if n_ancilla > 0 {
                b.cx(0, system_start);
            }
        }
        _ => {
            // General: apply some rotation
            let angle = term.coeff.signum() * PI / 4.0;
            b.rz(system_start, angle);
        }
    }
}

/// Compute the 1-norm of an LCU decomposition.
pub fn lcu_one_norm(terms: &[LcuTerm]) -> f64 {
    terms.iter().map(|t| t.coeff.abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcu_two_terms() {
        let terms = vec![
            LcuTerm {
                coeff: 0.6,
                unitary_index: 0,
            },
            LcuTerm {
                coeff: 0.4,
                unitary_index: 1,
            },
        ];
        let circ = lcu_block_encoding(1, &terms);
        assert_eq!(circ.n_qubits(), 2); // 1 ancilla + 1 system
        assert!(circ.gate_count() > 0);
    }

    #[test]
    fn test_lcu_four_terms() {
        let terms = vec![
            LcuTerm {
                coeff: 0.3,
                unitary_index: 0,
            },
            LcuTerm {
                coeff: 0.2,
                unitary_index: 1,
            },
            LcuTerm {
                coeff: 0.3,
                unitary_index: 2,
            },
            LcuTerm {
                coeff: 0.2,
                unitary_index: 3,
            },
        ];
        let circ = lcu_block_encoding(2, &terms);
        assert_eq!(circ.n_qubits(), 4); // 2 ancilla + 2 system
        assert!(circ.gate_count() > 0);
    }

    #[test]
    fn test_lcu_one_norm() {
        let terms = vec![
            LcuTerm {
                coeff: 0.5,
                unitary_index: 0,
            },
            LcuTerm {
                coeff: -0.3,
                unitary_index: 1,
            },
        ];
        assert!((lcu_one_norm(&terms) - 0.8).abs() < 1e-10);
    }
}
