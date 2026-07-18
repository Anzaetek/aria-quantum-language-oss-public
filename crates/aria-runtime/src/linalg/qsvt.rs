// SPDX-License-Identifier: Apache-2.0
//! Quantum Singular Value Transformation (QSVT) — circuit construction.
//!
//! QSVT interleaves a block-encoding unitary `W` (the "signal") with QSP phase
//! rotations `S(φ)` to block-encode a polynomial `P(A)` of the block-encoded
//! matrix. This module builds the *circuit*; the QSP phase angles that realize a
//! target polynomial are computed by the numerically-validated angle finder in
//! [`omega_core::qsp`] (a Wang–Lin/Dong least-squares fit of the exact QSP
//! response over Chebyshev nodes), which mirrors the `QuantumProofs/QSP.lean`
//! `qsp` definition gate-for-gate.
//!
//! Conventions (identical to `omega_core::qsp` and `QSP.lean`, the "Wx" family):
//! * signal   `W(x) = [[x, i·s],[i·s, x]] = e^{i·arccos(x)·X}`  (s = √(1−x²));
//! * phase    `S(φ) = diag(e^{iφ}, e^{-iφ}) = e^{iφZ}`;
//! * unitary  `U_Φ = S(φ₀)·W·S(φ₁)·W·…·S(φ_{d-1})·W`   (`d` phases ⇒ degree ≤ d).
//!
//! The ancilla/flag qubit of the block encoding is qubit 0. For a single-ancilla
//! block encoding the QSVT projector-controlled phase `e^{iφ(2Π−I)}`
//! (`Π = |0⟩⟨0|` on the ancilla) is exactly `S(φ)` on qubit 0, i.e. `Rz(−2φ)`
//! with the standard `Rz(θ) = diag(e^{−iθ/2}, e^{iθ/2})`. The
//! `⟨0|_anc U_Φ |0⟩_anc` block then realizes `P(A)`; with [`inversion_angles`]
//! that polynomial approximates the (sub-normalized) inverse `≈ 1/x` on
//! `[1/κ, 1]` — validated against `qsp_response` in the tests below.
//!
//! (Previously this module emitted a fixed structural template with placeholder
//! `CX` "signal" steps and `[π/4, 0, …, 0, π/4]` "angles" that implemented no
//! polynomial at all. It now consumes the real angle finder and a real signal.)

use aria_core::ast::nodes::{rz, Circuit};
use omega_core::qsp::qsp_inversion_phases;
use std::collections::HashMap;

/// Build a QSVT circuit that applies the QSP phase sequence `phases` to the
/// block-encoding unitary `block_encoding` (the signal `W`).
///
/// - `n_system`: system qubits. The circuit spans `1 + n_system` qubits with the
///   ancilla/flag as qubit 0; `block_encoding` must use the same `q` register /
///   layout (ancilla first).
/// - `phases`: QSP phase angles `[φ₀, …, φ_{d-1}]` (e.g. from [`inversion_angles`]).
/// - `block_encoding`: the signal unitary `W`, a circuit over the same register.
///
/// Emits `U_Φ = S(φ₀)·W·…·S(φ_{d-1})·W` (Wx convention). The tests verify that
/// its `⟨0|·|0⟩` statevector amplitude equals `omega_core::qsp::qsp_response`.
pub fn qsvt_circuit(n_system: usize, phases: &[f64], block_encoding: &Circuit) -> Circuit {
    let n_total = 1 + n_system;
    let mut circ = Circuit::new("qsvt");
    let qubits = circ.qreg("q", n_total);
    let idmap: HashMap<aria_core::ast::nodes::Qubit, aria_core::ast::nodes::Qubit> = HashMap::new();
    // Gate (time) order is the matrix product read right-to-left: apply the
    // signal `W` first, then the phase `S(φ)`, ending with `S(φ₀)` applied last.
    for &phi in phases.iter().rev() {
        circ.append_circuit(block_encoding, &idmap, None); // signal W
        circ.apply(rz(-2.0 * phi), vec![qubits[0].clone()]); // phase S(φ) on the ancilla
    }
    circ
}

/// QSP phase angles implementing the (sub-normalized) matrix inversion
/// `≈ (1/κ)/x` on `[1/κ, 1]`, for `degree` phases.
///
/// Delegates to the numerically-validated angle finder
/// [`omega_core::qsp::qsp_inversion_phases`]: a least-squares fit of the exact
/// QSP response to the target `scale/x` over Chebyshev nodes. Feed the result to
/// [`qsvt_circuit`]. The `scale = 0.9/κ` keeps the target sub-normalized
/// (`|P| ≤ 1`), as QSVT requires; the realized block is that scaled inverse.
pub fn inversion_angles(degree: usize, kappa: f64) -> Vec<f64> {
    let kappa = kappa.max(1.0);
    // Sub-normalize with a little headroom below 1/κ so the fit is not pinned to
    // |P| = 1 at the x = 1/κ endpoint (which QSP cannot exceed).
    let scale = 0.9 / kappa;
    let iters = (1000 * degree.max(1)).clamp(4000, 20_000);
    qsp_inversion_phases(degree, kappa, scale, iters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_core::ast::CircuitBuilder;
    use omega_core::qsp::qsp_response;

    /// Angle layer: the fitted phases implement the scaled inverse `0.9/κ · 1/x`
    /// on `[1/κ, 1]` — the "validated vs 1/x" numeric anchor.
    #[test]
    fn inversion_angles_approximate_reciprocal() {
        let kappa = 2.0_f64;
        let degree = 8;
        let phases = inversion_angles(degree, kappa);
        assert_eq!(phases.len(), degree);
        let scale = 0.9 / kappa;
        let lo = 1.0 / kappa;
        let mut maxerr = 0.0_f64;
        for i in 0..=20 {
            let x = lo + (1.0 - lo) * i as f64 / 20.0;
            let got = qsp_response(&phases, x).re;
            maxerr = maxerr.max((got - scale / x).abs());
        }
        assert!(
            maxerr < 8e-2,
            "QSP inversion fit max err on [1/κ,1] = {maxerr}"
        );
    }

    /// Circuit layer: for a single-qubit signal `W(x) = e^{i·arccos(x)·X}`, the
    /// emitted QSVT circuit's `⟨0|U|0⟩` amplitude equals the exact QSP response
    /// `qsp_response(phases, x)` — i.e. the circuit really implements the QSP
    /// polynomial the phases encode (ties the circuit to the validated angles).
    #[test]
    fn qsvt_circuit_realizes_qsp_response() {
        let phases = [0.3_f64, -1.1, 0.7, 2.0];
        for &x in &[-0.8_f64, -0.3, 0.0, 0.4, 0.9] {
            // W(x) = e^{i·θ·X} = Rx(-2θ), θ = arccos(x).
            let theta = x.acos();
            let signal = CircuitBuilder::new("w", 1, 0).rx(0, -2.0 * theta).build();
            let circ = qsvt_circuit(0, &phases, &signal);
            let sv = crate::run::statevector(
                &circ,
                &std::collections::HashMap::new(),
                crate::run::BackendSel::Sim,
            )
            .unwrap();
            let want = qsp_response(&phases, x);
            assert!(
                (sv[0].re - want.re).abs() < 1e-9 && (sv[0].im - want.im).abs() < 1e-9,
                "x={x}: circuit ⟨0|U|0⟩ = {:?}, qsp_response = {:?}",
                sv[0],
                want
            );
        }
    }

    /// Structure: `d` phases ⇒ `d` signal blocks + `d` phase rotations, on the
    /// block encoding's register (`1 + n_system` qubits).
    #[test]
    fn qsvt_circuit_structure_scales_with_degree() {
        let n_system = 1;
        let signal = CircuitBuilder::new("w", 1 + n_system, 0)
            .h(0)
            .cx(0, 1)
            .build();
        let sig_gates = signal.gate_count();
        for d in [2usize, 4, 8] {
            let phases = vec![0.1; d];
            let circ = qsvt_circuit(n_system, &phases, &signal);
            assert_eq!(circ.n_qubits(), 1 + n_system);
            // d signal copies + d phase Rz gates.
            assert_eq!(circ.gate_count(), d * sig_gates + d);
        }
    }
}
