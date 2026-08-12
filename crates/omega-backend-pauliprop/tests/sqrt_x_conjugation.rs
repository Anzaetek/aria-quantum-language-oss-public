// SPDX-License-Identifier: Apache-2.0
//! `pauliprop`'s `√X` / `√X†` Clifford conjugation.
//!
//! # The direction trap
//!
//! `SingleImg` stores **`G† P G`**. The Lean proofs in
//! `proofs/lean4/QuantumProofs/SqrtX.lean` state the **`G P G†`** form. Those
//! are different, and reading one as the other swaps `sx` with `sxdg` — which
//! is a pure sign change on `fz`, produces a perfectly valid Pauli sum, and is
//! invisible to anything that only checks self-consistency.
//!
//! The table entries were therefore re-derived numerically and cross-checked
//! against the existing `CLIFF_S` comment (`S† X S = −Y = (−i)·raw(1,1)`),
//! which the same derivation reproduces exactly.
//!
//! The definitive check is the N-way expectation lane
//! (`omega-cli/tests/nway_expectation.rs`), which compares against Qiskit at
//! 1e-12 and moved `pauliprop` from 7 to 8 of 11 admitted fixtures when this
//! landed. These tests are the fast local guard on the same property.

use omega_backend_pauliprop::PauliPropBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind) -> GateOp {
    GateOp {
        gate,
        qubits: smallvec::smallvec![Qubit(0)],
        params: smallvec::smallvec![],
        classical_bit: None,
        condition: None,
    }
}

fn expect(ops: Vec<GateOp>, p: PauliOp) -> f64 {
    let mut ir = CircuitIR::new(1, CircuitType::GateBased);
    ir.ops = ops;
    PauliPropBackend::new()
        .expectation(
            &ir,
            &ParameterBinding::new(),
            &Observable {
                terms: vec![(1.0, vec![(0, p)])],
            },
        )
        .expect("pauliprop expectation")
}

/// Accepted at all — the point of the change. `sx` is Clifford, so a backend
/// built on Clifford conjugation should never have refused it.
#[test]
fn sx_and_sxdg_are_conjugatable() {
    for g in [GateKind::Sx, GateKind::Sxdg] {
        let mut ir = CircuitIR::new(1, CircuitType::GateBased);
        ir.ops = vec![op(g.clone())];
        PauliPropBackend::new()
            .expectation(
                &ir,
                &ParameterBinding::new(),
                &Observable {
                    terms: vec![(1.0, vec![(0, PauliOp::Z)])],
                },
            )
            .unwrap_or_else(|e| panic!("{g:?} is Clifford and must conjugate: {e}"));
    }
}

/// **The sign test.** `sx|0⟩` is the `−Y` eigenstate, `sxdg|0⟩` the `+Y` one.
///
/// This is the single assertion that catches reading the table in the wrong
/// direction, or copying `CLIFF_SX`'s `fz` into `CLIFF_SXDG`. Everything else
/// in this file passes under that mutation.
#[test]
fn sx_and_sxdg_give_opposite_y_signs() {
    let y_sx = expect(vec![op(GateKind::Sx)], PauliOp::Y);
    let y_sxdg = expect(vec![op(GateKind::Sxdg)], PauliOp::Y);
    assert!(
        (y_sx + 1.0).abs() < 1e-12,
        "sx|0> must be the -Y eigenstate: <Y> = {y_sx}, expected -1"
    );
    assert!(
        (y_sxdg - 1.0).abs() < 1e-12,
        "sxdg|0> must be the +Y eigenstate: <Y> = {y_sxdg}, expected +1"
    );
    // Stated as a difference too, so a uniform sign error on BOTH (a
    // direction mistake applied consistently) still fails here.
    assert!(
        (y_sx - y_sxdg + 2.0).abs() < 1e-12,
        "the two must differ by exactly -2, got {y_sx} and {y_sxdg}"
    );
}

/// `X` is the fixed axis of both.
#[test]
fn x_is_invariant() {
    for g in [GateKind::Sx, GateKind::Sxdg] {
        let v = expect(vec![op(GateKind::H), op(g.clone())], PauliOp::X);
        assert!((v - 1.0).abs() < 1e-12, "{g:?} must fix X: <X> = {v}");
    }
}

/// `sx·sx = X`, so `⟨Z⟩` flips to −1.
#[test]
fn sx_twice_is_x() {
    let v = expect(vec![op(GateKind::Sx), op(GateKind::Sx)], PauliOp::Z);
    assert!((v + 1.0).abs() < 1e-12, "sx;sx|0> = |1>, <Z> = {v}");
}

/// `sx; sxdg = I`.
#[test]
fn sx_then_sxdg_is_identity() {
    let v = expect(vec![op(GateKind::Sx), op(GateKind::Sxdg)], PauliOp::Z);
    assert!((v - 1.0).abs() < 1e-12, "sx;sxdg = I, <Z> = {v}");
}

/// Agreement with the stabilizer backend on the same circuits.
///
/// Weaker evidence than the Qiskit comparison in the N-way lane — two of our
/// own engines agreeing is exactly the configuration `FIXES_PLAN.md` K3 warns
/// about — but it is cheap, and the two encode the rule in genuinely different
/// ways: a bit-level tableau update versus a Pauli-image table with complex
/// factors. A shared misreading would have to survive both representations.
#[test]
fn agrees_with_the_stabilizer_backend() {
    use omega_backend_pauli::PauliBackend;
    for gates in [
        vec![GateKind::Sx],
        vec![GateKind::Sxdg],
        vec![GateKind::Sx, GateKind::Sx],
        vec![GateKind::H, GateKind::Sx],
        vec![GateKind::Sx, GateKind::H, GateKind::Sxdg],
    ] {
        let mut ir = CircuitIR::new(1, CircuitType::GateBased);
        ir.ops = gates.iter().cloned().map(op).collect();
        for p in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
            let obs = Observable {
                terms: vec![(1.0, vec![(0, p)])],
            };
            let a = PauliPropBackend::new()
                .expectation(&ir, &ParameterBinding::new(), &obs)
                .expect("pauliprop");
            let b = PauliBackend::new()
                .expectation(&ir, &ParameterBinding::new(), &obs)
                .expect("pauli");
            assert!(
                (a - b).abs() < 1e-12,
                "{gates:?} / {p:?}: pauliprop {a} vs stabilizer {b}"
            );
        }
    }
}
