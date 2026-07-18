// SPDX-License-Identifier: Apache-2.0
//! Quantum circuit recipe for `A x = b` (LCU block-encoding + QSVT inversion).
//!
//! CIRCUIT half of the gate-model toolkit's
//! `quantum-core/src/linalg/solver.rs` (the classical reference solver —
//! `solve` / `solve_classical` — lives in [`omega_core::solver`]).
//! [`quantum_solve_circuit`] builds the QSVT circuit applying the
//! polynomial approximation of `1/x` to the block-encoded `A/α`.
//! KEEP-IN-SYNC with the toolkit source.

use crate::linalg::block_encode::{
    block_encode_dense, block_encode_diagonal, DenseBlockEncoding, DiagonalBlockEncoding,
};
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

/// A genuine QSVT matrix-inversion circuit for a diagonal Hermitian system.
#[derive(Clone, Debug)]
pub struct QuantumInversion {
    /// The QSVT circuit `U_Φ` over `1 + n_system` qubits (ancilla = qubit 0).
    /// Its `⟨0|_a U_Φ |0⟩_a` block is the degree-`degree` QSP polynomial of
    /// `A/α`; its real part `(U_Φ + U_Φ†)/2` block equals `≈ (0.9/κ)·(A/α)⁻¹`,
    /// i.e. the (sub-normalized) inverse. Prepending `|b⟩` on the system and
    /// post-selecting the ancilla on `|0⟩` yields `∝ A⁻¹|b⟩` (real part taken
    /// via the standard `(U+U†)/2` LCU output wrapper).
    pub circuit: Circuit,
    /// The real diagonal block encoding the QSVT is built on.
    pub block_encoding: DiagonalBlockEncoding,
    /// Condition number `κ = max|λ|/min|λ|` used to set the domain `[1/κ, 1]`.
    pub kappa: f64,
    /// QSP polynomial degree used.
    pub degree: usize,
}

/// Build a genuine QSVT inversion circuit for a diagonal Hermitian matrix
/// `A = diag(spectrum)`: block-encode `A/α` exactly ([`block_encode_diagonal`]),
/// then apply the real QSP inversion polynomial ([`qsvt_circuit`] with
/// [`inversion_angles`]). Unlike [`quantum_solve_circuit`] (dense A, toy block
/// encoding, resource-estimation only), this genuinely inverts `A` — verified
/// end-to-end against the classical solution in the tests.
pub fn quantum_solve_diagonal(spectrum: &[f64], eps: f64) -> QuantumInversion {
    let be = block_encode_diagonal(spectrum);
    // κ from the (scaled) spectrum; degree from the Childs–Kothari–Somma bound
    // d = O(κ·log(κ/ε)), capped for tractability.
    let min_abs = spectrum
        .iter()
        .map(|l| l.abs())
        .filter(|&l| l > 1e-12)
        .fold(f64::INFINITY, f64::min);
    let kappa = (be.alpha / min_abs).max(1.0);
    let degree = (kappa * (kappa / eps.max(1e-9)).ln())
        .max(4.0)
        .round()
        .min(40.0) as usize;
    let phases = inversion_angles(degree, kappa);
    let circuit = qsvt_circuit(be.n_system, &phases, &be.circuit);
    QuantumInversion {
        circuit,
        block_encoding: be,
        kappa,
        degree,
    }
}

/// Build a genuine QSVT inversion circuit for a **dense** real symmetric
/// positive-definite matrix `A` (up to 4×4), via eigenbasis conjugation:
/// classically diagonalize `A = U·D·Uᵀ`, exactly block-encode `D`, and conjugate
/// it into `A`'s eigenbasis with the compiled rotation `U`. Works for any
/// condition number (κ only sets the degree). Restricted to real symmetric PD
/// `A` of dimension 2 or 4 (the class the diagonal inversion polynomial and the
/// 1-/2-qubit eigenbasis synthesis cover). Verified end-to-end vs classical solve.
pub fn quantum_solve_hermitian(a: &Array2<f64>, eps: f64) -> QuantumInversion {
    use crate::linalg::synth::{conjugated_block_encoding, eigh_symmetric};
    let dim = a.nrows();
    assert!(
        (dim == 2 || dim == 4) && a.ncols() == dim,
        "quantum_solve_hermitian supports 2×2 or 4×4 real symmetric A"
    );
    let (eigenvalues, u) = eigh_symmetric(a);
    let be_diag = block_encode_diagonal(&eigenvalues);
    let n_system = be_diag.n_system;
    let w_a = conjugated_block_encoding(&be_diag.circuit, &u, n_system);

    let min_abs = eigenvalues
        .iter()
        .map(|l| l.abs())
        .filter(|&l| l > 1e-12)
        .fold(f64::INFINITY, f64::min);
    let kappa = (be_diag.alpha / min_abs).max(1.0);
    let degree = (kappa * (kappa / eps.max(1e-9)).ln())
        .max(4.0)
        .round()
        .min(40.0) as usize;
    let phases = inversion_angles(degree, kappa);
    let circuit = qsvt_circuit(n_system, &phases, &w_a);
    QuantumInversion {
        circuit,
        block_encoding: DiagonalBlockEncoding {
            circuit: w_a,
            alpha: be_diag.alpha,
            n_system,
        },
        kappa,
        degree,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn cmplx(r: f64) -> Complex64 {
        Complex64::new(r, 0.0)
    }

    #[test]
    fn quantum_solve_diagonal_matches_classical() {
        use aria_core::ast::nodes::x;
        use aria_core::ast::Circuit;
        use num_complex::Complex64;
        use std::collections::HashMap;

        // Diagonal QSVT block P(μ_i) = ⟨0,i| U_Φ |0,i⟩, read off the simulated
        // circuit (system prepared in |i⟩, ancilla=0 amplitude at flat index 2i).
        fn diag_block(circuit: &Circuit, n_system: usize, i: usize) -> Complex64 {
            let mut probe = Circuit::new("probe");
            let qs = probe.qreg("q", 1 + n_system);
            for j in 0..n_system {
                if (i >> j) & 1 == 1 {
                    probe.apply(x(), vec![qs[1 + j].clone()]);
                }
            }
            probe.append_circuit(circuit, &HashMap::new(), None);
            let sv = crate::run::statevector(&probe, &HashMap::new(), crate::run::BackendSel::Sim)
                .unwrap();
            sv[i << 1]
        }

        // Well-conditioned diagonal systems, n_system = 1 and 2.
        for spectrum in [vec![1.0, 2.0], vec![2.0, 3.0, 3.0, 4.0]] {
            let dim = spectrum.len();
            let inv = quantum_solve_diagonal(&spectrum, 1e-2);
            // Real inversion operator's diagonal: Re(P)(μ_i) = ⟨(U+U†)/2⟩_ii.
            let re_p: Vec<f64> = (0..dim)
                .map(|i| diag_block(&inv.circuit, inv.block_encoding.n_system, i).re)
                .collect();

            // For several right-hand sides, the post-selected quantum solution
            // xq_i ∝ Re(P)_i · b_i must match the classical x_i = b_i/λ_i.
            let rhss = [
                vec![0.0, 1.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0, 1.0],
                vec![0.3, 0.9, 0.2, 0.7],
            ];
            for b in rhss {
                let b = &b[..dim];
                let q: Vec<f64> = (0..dim).map(|i| re_p[i] * b[i]).collect();
                let c: Vec<f64> = (0..dim).map(|i| b[i] / spectrum[i]).collect();
                let dot: f64 = q.iter().zip(&c).map(|(u, v)| u * v).sum();
                let nq: f64 = q.iter().map(|v| v * v).sum::<f64>().sqrt();
                let nc: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
                let fidelity = (dot / (nq * nc)).powi(2);
                assert!(
                    fidelity > 0.999,
                    "spectrum {spectrum:?} b {b:?}: solve fidelity {fidelity} (κ={}, d={})",
                    inv.kappa,
                    inv.degree
                );
            }
        }
    }

    #[test]
    fn quantum_solve_hermitian_matches_classical() {
        use aria_core::ast::nodes::x;
        use aria_core::ast::Circuit;
        use num_complex::Complex64;
        use std::collections::HashMap;

        // Full QSVT block P(A)_{ji} = ⟨0,j| U_Φ |0,i⟩ by simulating the circuit on
        // each system basis input |i⟩ and reading the ancilla=0 amplitudes.
        fn qsvt_block(circuit: &Circuit, n_system: usize) -> Vec<Vec<Complex64>> {
            let dim = 1usize << n_system;
            let mut block = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
            for i in 0..dim {
                let mut probe = Circuit::new("probe");
                let qs = probe.qreg("q", 1 + n_system);
                for j in 0..n_system {
                    if (i >> j) & 1 == 1 {
                        probe.apply(x(), vec![qs[1 + j].clone()]);
                    }
                }
                probe.append_circuit(circuit, &HashMap::new(), None);
                let sv =
                    crate::run::statevector(&probe, &HashMap::new(), crate::run::BackendSel::Sim)
                        .unwrap();
                for j in 0..dim {
                    block[j][i] = sv[j << 1];
                }
            }
            block
        }

        // Real symmetric positive-definite systems, dense (non-diagonal): 2×2, 4×4.
        let a2: Array2<f64> = array![[2.0, 0.5], [0.5, 3.0]];
        let a4: Array2<f64> = array![
            [4.0, 0.6, 0.2, 0.1],
            [0.6, 5.0, 0.3, 0.2],
            [0.2, 0.3, 6.0, 0.4],
            [0.1, 0.2, 0.4, 7.0],
        ];
        for a in [a2, a4] {
            let dim = a.nrows();
            let inv = quantum_solve_hermitian(&a, 1e-2);
            let n_system = inv.block_encoding.n_system;
            let block = qsvt_block(&inv.circuit, n_system);
            // Real inversion operator Re(P)(A) = (P + P†)/2.
            let re_p: Vec<Vec<f64>> = (0..dim)
                .map(|j| {
                    (0..dim)
                        .map(|k| 0.5 * (block[j][k].re + block[k][j].re))
                        .collect()
                })
                .collect();
            for b in [vec![1.0, 0.0, 0.0, 0.0], vec![0.3, 0.9, 0.5, 0.2]] {
                let b = &b[..dim];
                let q: Vec<f64> = (0..dim)
                    .map(|j| (0..dim).map(|k| re_p[j][k] * b[k]).sum())
                    .collect();
                let c = classical_solve_sym(&a, b);
                let dot: f64 = q.iter().zip(&c).map(|(u, v)| u * v).sum();
                let nq: f64 = q.iter().map(|v| v * v).sum::<f64>().sqrt();
                let nc: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
                let fidelity = (dot / (nq * nc)).powi(2);
                assert!(
                    fidelity > 0.999,
                    "dim {dim} b {b:?}: solve fidelity {fidelity} (κ={:.2}, d={})",
                    inv.kappa,
                    inv.degree
                );
            }
        }
    }

    /// Classical `A⁻¹ b` for a small symmetric matrix (Gaussian elimination).
    fn classical_solve_sym(a: &Array2<f64>, b: &[f64]) -> Vec<f64> {
        let n = a.nrows();
        let mut m = a.clone();
        let mut x = b.to_vec();
        for col in 0..n {
            let mut piv = col;
            for r in (col + 1)..n {
                if m[(r, col)].abs() > m[(piv, col)].abs() {
                    piv = r;
                }
            }
            if piv != col {
                for c in 0..n {
                    let t = m[(col, c)];
                    m[(col, c)] = m[(piv, c)];
                    m[(piv, c)] = t;
                }
                x.swap(col, piv);
            }
            let d = m[(col, col)];
            for r in 0..n {
                if r == col {
                    continue;
                }
                let f = m[(r, col)] / d;
                for c in col..n {
                    let t = m[(col, c)];
                    m[(r, c)] -= f * t;
                }
                x[r] -= f * x[col];
            }
        }
        (0..n).map(|i| x[i] / m[(i, i)]).collect()
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
