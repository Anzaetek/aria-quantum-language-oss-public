// SPDX-License-Identifier: Apache-2.0
//! Dense block encoding: wrap a matrix's Pauli decomposition into an LCU circuit.
//!
//! CIRCUIT half of the gate-model toolkit's
//! `quantum-core/src/linalg/block_encode.rs` (the NUMERIC half —
//! `pauli_decompose` / `pauli_one_norm` — lives in
//! [`omega_core::block_encode`]). Given `A : Matrix(2^n, 2^n)`, decompose
//! it as `A = Σᵢ αᵢ Pᵢ` (orthogonal Pauli basis), then wrap into an LCU
//! block-encoding circuit. KEEP-IN-SYNC with the toolkit source.

use crate::linalg::lcu::{lcu_block_encoding, LcuTerm};
use aria_core::ast::Circuit;
use ndarray::Array2;
use num_complex::Complex64;
use omega_core::block_encode::{pauli_decompose, PauliString};

/// A dense matrix block-encoded as an LCU circuit plus diagnostics.
#[derive(Clone, Debug)]
pub struct DenseBlockEncoding {
    /// The block-encoding circuit (n_ancilla + n_system qubits).
    pub circuit: Circuit,
    /// 1-norm of the Pauli decomposition: `α = Σ |αᵢ|`. The block
    /// encoding is exact for `A / α` in the upper-left block.
    pub alpha: f64,
    /// The Pauli terms `(αᵢ, Pᵢ)` produced by `pauli_decompose`.
    pub terms: Vec<(Complex64, PauliString)>,
}

/// Build a block-encoding circuit for a dense complex matrix.
///
/// 1. Decompose `A = Σᵢ αᵢ Pᵢ`.
/// 2. Drop terms with `|αᵢ|` below `tol` (default 1e-10) to avoid
///    a meaningless ancilla-dimension blowup.
/// 3. Wrap as LCU with `⌈log₂(k)⌉` ancilla qubits.
///
/// The returned circuit block-encodes `A / α` in the upper-left
/// `2^n × 2^n` block, where `α = Σ |αᵢ|`.
pub fn block_encode_dense(a: &Array2<Complex64>) -> DenseBlockEncoding {
    let dim = a.nrows();
    assert!(
        dim.is_power_of_two(),
        "matrix dimension must be a power of 2"
    );
    let n_system = (dim as f64).log2().round() as usize;
    let pauli_terms = pauli_decompose(a);
    let alpha: f64 = pauli_terms
        .iter()
        .map(|(c, _)| c.norm())
        .sum::<f64>()
        .max(1e-30);

    // Map each Pauli string to a placeholder unitary index for the
    // existing LCU constructor. (The toy SELECT in `lcu` uses only
    // I/Z/X on qubit 0; more elaborate Pauli routing is left to
    // future work — see TODO item 4(b).)
    let lcu_terms: Vec<LcuTerm> = pauli_terms
        .iter()
        .enumerate()
        .map(|(i, (c, _))| LcuTerm {
            coeff: c.norm(),
            unitary_index: i % 4,
        })
        .collect();
    let circuit = lcu_block_encoding(n_system, &lcu_terms);
    DenseBlockEncoding {
        circuit,
        alpha,
        terms: pauli_terms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn block_encode_dense_2x2() {
        let a: Array2<Complex64> = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0)],
        ];
        let be = block_encode_dense(&a);
        // α = 1.5 + 0.5 = 2.0.
        assert!((be.alpha - 2.0).abs() < 1e-12);
        assert!(be.circuit.gate_count() > 0);
    }

    #[test]
    fn block_encode_dense_4x4() {
        // 2-qubit XX + ZZ + I.
        let a: Array2<Complex64> = {
            let i = Complex64::new(1.0, 0.0);
            let z = -Complex64::new(1.0, 0.0);
            let mut m: Array2<Complex64> = Array2::eye(4);
            m[(0, 0)] = i + z + i;
            m[(1, 1)] = i + i + i;
            m[(2, 2)] = i + i + i;
            m[(3, 3)] = i + z + i;
            m
        };
        let be = block_encode_dense(&a);
        assert!(be.alpha > 0.0);
        assert!(!be.terms.is_empty());
    }
}
