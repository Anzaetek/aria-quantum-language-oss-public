// SPDX-License-Identifier: Apache-2.0
//! `rxx` / `rzz` exported by `aria-core` must be readable by `omega-parser`,
//! and must mean the same thing Qiskit means by them.
//!
//! Measured before this landed:
//!
//! ```text
//!   RXX: emits "rxx(0.7) q[0], q[1];" -> REJECTED: unknown gate: rxx
//!   RYY: emits "ryy(0.7) q[0], q[1];" -> REJECTED: unknown gate: ryy
//!   RZZ: emits "rzz(0.7) q[0], q[1];" -> REJECTED: unknown gate: rzz
//! ```
//!
//! # Why a decomposition and not a `GateKind`
//!
//! These are interchange spellings, not new capabilities — the point is that a
//! QASM2 file round-trips, not that the engine gains a primitive. New variants
//! would cost seven backends and re-open the CUDA non-exhaustive-match problem
//! for a gate nobody asked to accelerate.
//!
//! # Why `ryy` is still refused
//!
//! Measured on qiskit 2.5.1 — and the finer measurement matters, because the
//! coarse version ("Qiskit accepts rxx/rzz") is only true of one loader:
//!
//! | spelling | strict `qasm2.loads` | legacy `from_qasm_str` |
//! |---|---|---|
//! | `rxx` | rejects | **accepts** |
//! | `rzz` | rejects | **accepts** |
//! | `cp`  | rejects | **accepts** |
//! | `ryy` | rejects | **rejects** |
//!
//! So `rxx`/`rzz` name real interoperability worth preserving, and `ryy` names
//! none: no Qiskit loader reads it. Teaching this parser to read `ryy` would
//! make the round trip work *for us alone* while every other toolchain still
//! could not load the file — worse than the status quo, because it would look
//! fixed. What to do about *emitting* it is a separate decision.
//!
//! # What this test would miss if written carelessly
//!
//! Asserting only that the source parses. A decomposition with a wrong sign, a
//! swapped control/target, or a dropped conjugator still parses. So every case
//! is contracted through the statevector backend and compared against the gate's
//! defining matrix.
//!
//! `RXX` and `RZZ` are both **symmetric in their two qubits**, so a
//! control/target swap is invisible on a symmetric input. The fixture therefore
//! starts from an **asymmetric state** (`h` on q0 only, then `t` on q0 to break
//! the remaining reflection) rather than relying on gate asymmetry — and
//! `asymmetry_of_the_fixture_is_load_bearing` proves the fixture can tell the
//! two qubits apart at all.

use aria_core::ast::nodes::*;
use aria_core::ast::qasm::to_qasm;
use num_complex::Complex64;
use omega_backend_statevector::StatevectorBackend;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

const THETA: f64 = 0.7;

/// `exp(-i·θ/2·(Z⊗Z))`, the qelib1 / Qiskit `RZZGate`.
/// Diagonal in the computational basis with phase `e^{∓iθ/2}`.
fn rzz_matrix(theta: f64) -> [[Complex64; 4]; 4] {
    let mut m = [[Complex64::new(0.0, 0.0); 4]; 4];
    for (k, row) in m.iter_mut().enumerate() {
        // Parity of the two bits: |00> and |11> get e^{-iθ/2}, the rest e^{+iθ/2}.
        let parity = ((k >> 1) & 1) ^ (k & 1);
        let sign = if parity == 0 { -1.0 } else { 1.0 };
        row[k] = Complex64::from_polar(1.0, sign * theta / 2.0);
    }
    m
}

/// `exp(-i·θ/2·(X⊗X))` = `(H⊗H) · RZZ(θ) · (H⊗H)`.
fn rxx_matrix(theta: f64) -> [[Complex64; 4]; 4] {
    let h = 1.0 / std::f64::consts::SQRT_2;
    // H⊗H, real and symmetric.
    let mut hh = [[0.0f64; 4]; 4];
    for (i, row) in hh.iter_mut().enumerate() {
        for (j, e) in row.iter_mut().enumerate() {
            let s0 = if (i >> 1) & 1 & ((j >> 1) & 1) == 1 { -1.0 } else { 1.0 };
            let s1 = if i & 1 & (j & 1) == 1 { -1.0 } else { 1.0 };
            *e = h * h * s0 * s1;
        }
    }
    let rzz = rzz_matrix(theta);
    let mut out = [[Complex64::new(0.0, 0.0); 4]; 4];
    for (i, orow) in out.iter_mut().enumerate() {
        for (j, oe) in orow.iter_mut().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..4 {
                acc += hh[i][k] * rzz[k][k] * hh[k][j];
            }
            *oe = acc;
        }
    }
    out
}

/// Prepare an asymmetric 2-qubit state, apply `gate`, return the amplitudes.
fn run(qasm_gate: &str) -> Vec<Complex64> {
    let src = format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\n\
         h q[0];\nt q[0];\n{qasm_gate}\n"
    );
    let ir = omega_parser::lower_to_ir(&src)
        .unwrap_or_else(|e| panic!("lowering failed for {qasm_gate:?}: {e}"));
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    match StatevectorBackend::new()
        .execute(&ir, &ParameterBinding::default(), &cfg)
        .unwrap_or_else(|e| panic!("simulation failed for {qasm_gate:?}: {e:?}"))
    {
        ExecResult::Statevector(sv) => sv,
        _ => unreachable!("shots: None must yield a Statevector"),
    }
}

/// The reference: apply the defining matrix to the same prepared state.
fn reference(m: &[[Complex64; 4]; 4]) -> Vec<Complex64> {
    // `h q[0]; t q[0];` on |00>. Qubit 0 is the LOW bit of the index, matching
    // the in-tree convention (`omega-core` keys states by u64 over the register).
    let h = 1.0 / std::f64::consts::SQRT_2;
    let mut psi = vec![Complex64::new(0.0, 0.0); 4];
    psi[0] = Complex64::new(h, 0.0);
    psi[1] = Complex64::from_polar(h, std::f64::consts::FRAC_PI_4);
    let mut out = vec![Complex64::new(0.0, 0.0); 4];
    for (i, o) in out.iter_mut().enumerate() {
        for (j, p) in psi.iter().enumerate() {
            *o += m[i][j] * p;
        }
    }
    out
}

fn max_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len(), "state dimension changed");
    a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max)
}

/// Both sides evaluate the same analytic expressions in f64, so the agreement
/// is a few ulps. A loose tolerance here would hide a real decomposition error.
const TOL: f64 = 1e-12;

#[test]
fn rzz_matches_its_defining_matrix() {
    let got = run(&format!("rzz({THETA}) q[0], q[1];"));
    let want = reference(&rzz_matrix(THETA));
    let d = max_diff(&got, &want);
    assert!(d <= TOL, "rzz({THETA}) differs from exp(-i θ/2 Z⊗Z) by {d:.3e}");
}

#[test]
fn rxx_matches_its_defining_matrix() {
    let got = run(&format!("rxx({THETA}) q[0], q[1];"));
    let want = reference(&rxx_matrix(THETA));
    let d = max_diff(&got, &want);
    assert!(d <= TOL, "rxx({THETA}) differs from exp(-i θ/2 X⊗X) by {d:.3e}");
}

/// The fixture must be able to tell q0 from q1, or the qubit-order check above
/// is decoration: `RXX` and `RZZ` are both symmetric in their two qubits, so
/// gate asymmetry cannot supply it and the *state* has to.
#[test]
fn asymmetry_of_the_fixture_is_load_bearing() {
    let straight = run(&format!("rzz({THETA}) q[0], q[1];"));
    let swapped = run(&format!("rz({THETA}) q[1];"));
    assert!(
        max_diff(&straight, &swapped) > 1e-6,
        "the fixture cannot distinguish these, so it could not catch a \
         control/target error either"
    );
}

/// What `aria-core` emits is what `omega-parser` reads — the actual round trip,
/// not a hand-written string.
#[test]
fn what_aria_core_emits_is_readable() {
    for (kind, n_ops) in [(GateKind::RXX, 7usize), (GateKind::RZZ, 3)] {
        let mut c = Circuit::new("c");
        let q = c.qreg("q", 2);
        c.apply(
            GateDef::with_params(kind, vec![THETA]),
            vec![q[0].clone(), q[1].clone()],
        );
        let text = to_qasm(&c).expect("emits");
        let ir = omega_parser::lower_to_ir(&text)
            .unwrap_or_else(|e| panic!("our own QASM2 is unreadable: {e}\n{text}"));
        assert_eq!(
            ir.ops.len(),
            n_ops,
            "{kind:?} decomposed into {} ops, expected {n_ops}:\n{text}",
            ir.ops.len()
        );
    }
}

/// `ryy` stays refused, and the refusal is the point rather than an oversight.
#[test]
fn ryy_is_still_refused_because_no_qiskit_loader_reads_it() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nryy(0.7) q[0], q[1];\n";
    assert!(
        omega_parser::lower_to_ir(src).is_err(),
        "reading `ryy` would fix the round trip for us alone: qiskit 2.5.1 \
         rejects it in BOTH the strict and the legacy loader, so the file would \
         still load nowhere else"
    );
}

/// Arity is refused rather than silently truncated.
#[test]
fn wrong_arity_is_refused() {
    let two_params = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nrzz(0.7, 0.2) q[0], q[1];\n";
    assert!(omega_parser::lower_to_ir(two_params).is_err(), "rzz takes 1 parameter");
    let one_qubit = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nrzz(0.7) q[0];\n";
    assert!(omega_parser::lower_to_ir(one_qubit).is_err(), "rzz acts on 2 qubits");
}

/// `inv @ rzz(θ) == rzz(−θ)`, and the decomposition must honour the modifier
/// rather than ignoring it — ignoring it parses fine and computes the wrong
/// state.
#[test]
fn the_inverse_modifier_is_honoured() {
    let fwd = run(&format!("rzz({THETA}) q[0], q[1];"));
    let inv = run(&format!("inv @ rzz({THETA}) q[0], q[1];"));
    let neg = run(&format!("rzz(-{THETA}) q[0], q[1];"));
    assert!(
        max_diff(&inv, &neg) <= TOL,
        "inv @ rzz(θ) should equal rzz(−θ)"
    );
    assert!(
        max_diff(&fwd, &inv) > 1e-6,
        "the fixture cannot see the inverse at all, so this proves nothing"
    );
}
