//! `pauliprop` must REFUSE `Reset`, not silently skip it.
//!
//! This backend evolves observables by unitary conjugation. `reset q` is the
//! non-unitary channel `ρ → |0⟩⟨0|_q ⊗ Tr_q(ρ)`, which that picture cannot
//! express. It used to fall into the
//! `Id | Barrier | Measure | Reset => {}` no-op arm, so the backend answered a
//! *different* circuit than the one submitted — silently, with no warning:
//! after `Bell(q0,q1); reset q0` it reported ⟨Z₀⟩ = 0 where the channel gives
//! +1 (the reset simply never happened).
//!
//! A clean `Unsupported` lets the CLI dispatcher fall back to a backend that
//! models the channel, which is the whole point of the refusal.

use omega_backend_pauliprop::PauliPropBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn g(kind: GateKind, qubits: &[u32]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }
}

fn bell_reset0() -> CircuitIR {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[0]));
    c.add_op(g(GateKind::CX, &[0, 1]));
    c.add_op(g(GateKind::Reset, &[0]));
    c
}

#[test]
fn expectation_refuses_reset() {
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };
    let err = PauliPropBackend::new()
        .expectation(&bell_reset0(), &ParameterBinding::new(), &obs)
        .expect_err("pauliprop must refuse Reset, not silently skip it");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Reset"),
        "refusal should name Reset, got: {msg}"
    );
}

/// A circuit with no `Reset` is unaffected — the refusal must not be a blanket
/// rejection of non-unitary-looking ops. `Bell` has ⟨Z₀Z₁⟩ = 1.
#[test]
fn non_reset_circuits_still_run() {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[0]));
    c.add_op(g(GateKind::CX, &[0, 1]));
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
    };
    let v = PauliPropBackend::new()
        .expectation(&c, &ParameterBinding::new(), &obs)
        .expect("Bell must still run");
    assert!((v - 1.0).abs() < 1e-12, "⟨Z₀Z₁⟩ = {v}, want 1");
}
