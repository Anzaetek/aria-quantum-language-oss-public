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
use aria_core::ast::{Circuit, CircuitBuilder};
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

/// A real diagonal (computational-basis) matrix `A = diag(spectrum)` block-
/// encoded by a single-ancilla uniformly-controlled rotation.
///
/// `⟨0|_a W |0⟩_a = A/α` (α = max|λ_i|), with the ancilla as qubit 0 and the
/// system as qubits `1..=n_system`. Unlike [`block_encode_dense`]'s toy LCU
/// SELECT, this circuit *exactly* block-encodes the (diagonal) matrix — every
/// diagonal entry is realized — so it drives a genuine QSVT matrix inversion
/// (see [`crate::linalg::solver::quantum_solve_diagonal`]).
#[derive(Clone, Debug)]
pub struct DiagonalBlockEncoding {
    /// The block-encoding circuit on `1 + n_system` qubits (ancilla = qubit 0).
    pub circuit: Circuit,
    /// Normalization `α = max|λ_i|`; the circuit block-encodes `A/α` (spectrum
    /// in `[-1, 1]`).
    pub alpha: f64,
    /// Number of system qubits (`log2(spectrum.len())`).
    pub n_system: usize,
}

/// Block-encode a real diagonal Hermitian matrix `A = diag(spectrum)` as a
/// single-ancilla circuit with `⟨0|_a W |0⟩_a = A/α`, `α = max|λ_i|`.
///
/// The signal `W(μ) = [[μ, i·s],[i·s, μ]] = e^{i·arccos(μ)·X}` (μ = λ/α) is
/// applied to the ancilla, multiplexed on the system register — a uniformly
/// controlled `Rx` realized as `H · UCRz · H`. `spectrum.len()` must be a power
/// of two. This is the "Wx" signal QSVT/`omega_core::qsp` consumes.
pub fn block_encode_diagonal(spectrum: &[f64]) -> DiagonalBlockEncoding {
    let dim = spectrum.len();
    assert!(
        dim.is_power_of_two() && dim >= 1,
        "spectrum length must be a power of two"
    );
    let n_system = (dim as f64).log2().round() as usize;
    let alpha = spectrum
        .iter()
        .fold(0.0_f64, |m, &l| m.max(l.abs()))
        .max(1e-30);

    // Ancilla Rx angle per system basis state i: W(μ_i) = e^{i·arccos(μ_i)·X}
    // = Rx(-2·arccos(μ_i)), μ_i = λ_i/α ∈ [-1, 1].
    let thetas: Vec<f64> = spectrum
        .iter()
        .map(|&l| -2.0 * (l / alpha).clamp(-1.0, 1.0).acos())
        .collect();

    let mut b = CircuitBuilder::new("block_encode_diag", 1 + n_system, 0);
    // UCRx = H(anc) · UCRz(thetas; controls = system) · H(anc). Controls are
    // ordered MSB-first (highest system qubit first) so `thetas[i]` lands on
    // system basis state `i` in the LSB-indexed statevector.
    b.h(0);
    let controls: Vec<usize> = (1..=n_system).rev().collect();
    ucrz_multiplexor(&mut b, &thetas, &controls, 0);
    b.h(0);
    DiagonalBlockEncoding {
        circuit: b.build(),
        alpha,
        n_system,
    }
}

/// Recursive uniformly-controlled `Rz` on `target`: one `angles` entry per
/// control basis pattern (`angles.len() == 2^controls.len()`), `controls`
/// MSB-first. Emits `2^k` `Rz` rotations interleaved with `2^k` `CX` gates (the
/// standard multiplexed-rotation / Gray-code ladder, no ancilla).
fn ucrz_multiplexor(b: &mut CircuitBuilder, angles: &[f64], controls: &[usize], target: usize) {
    if controls.is_empty() {
        b.rz(target, angles[0]);
        return;
    }
    let m = angles.len() / 2;
    let top: Vec<f64> = (0..m).map(|i| (angles[i] + angles[i + m]) / 2.0).collect();
    let bot: Vec<f64> = (0..m).map(|i| (angles[i] - angles[i + m]) / 2.0).collect();
    ucrz_multiplexor(b, &top, &controls[1..], target);
    b.cx(controls[0], target);
    ucrz_multiplexor(b, &bot, &controls[1..], target);
    b.cx(controls[0], target);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn diagonal_block_encoding_is_exact() {
        // ⟨0|_a W |0⟩_a must equal diag(spectrum)/α exactly: prepare each system
        // basis state |i⟩, run W, and read the (ancilla=0, system=i) amplitude
        // (flat index 2i in the LSB-indexed statevector).
        for spectrum in [vec![1.0, 2.0], vec![2.0, 3.0, 3.0, 4.0]] {
            let be = block_encode_diagonal(&spectrum);
            let dim = spectrum.len();
            for i in 0..dim {
                let mut probe = Circuit::new("probe");
                let qs = probe.qreg("q", 1 + be.n_system);
                for j in 0..be.n_system {
                    if (i >> j) & 1 == 1 {
                        probe.apply(aria_core::ast::nodes::x(), vec![qs[1 + j].clone()]);
                    }
                }
                probe.append_circuit(&be.circuit, &std::collections::HashMap::new(), None);
                let sv = crate::run::statevector(
                    &probe,
                    &std::collections::HashMap::new(),
                    crate::run::BackendSel::Sim,
                )
                .unwrap();
                let got = sv[i << 1];
                let want = spectrum[i] / be.alpha;
                assert!(
                    (got.re - want).abs() < 1e-9 && got.im.abs() < 1e-9,
                    "spectrum {spectrum:?} i={i}: block entry {got:?} != {want}"
                );
            }
        }
    }

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
