// SPDX-License-Identifier: Apache-2.0
//! A gate naming one qubit twice, or a qubit past the register, is REFUSED
//! with a typed error — not run.
//!
//! Found by the 2026-09 review of the `Rbs` arm: `x q[0]; rbs(0.7) q[0],q[0];`
//! returned `⟨Z0⟩ = −1.0` and exit 0. Both generators collapse to a bare `Y`
//! on one qubit and `R_Y(θ)·R_Y(−θ)` is the identity, so the answer was the
//! input state's, silently. `cx q[0],q[0]` misbehaved the same way. The
//! statevector backend panics on the same input ("a two-qubit gate on one
//! qubit"); neither a wrong number nor a panic is an acceptable answer to a
//! malformed circuit, and nothing upstream validates it (`CircuitIR::add_op`
//! pushes unchecked, the parser checks arity only).

use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::error::OmegaError;
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn circuit(nq: u32, gate: GateKind, qs: &[u32], ps: &[f64]) -> CircuitIR {
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    c.ops.push(GateOp {
        gate: GateKind::X,
        qubits: [Qubit(0)].into_iter().collect(),
        params: Default::default(),
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate,
        qubits: qs.iter().map(|&q| Qubit(q)).collect(),
        params: ps.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    });
    c
}

fn z0() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    }
}

#[test]
fn a_two_qubit_gate_on_one_qubit_is_refused() {
    let be = PauliPropBackend::new();
    let params = ParameterBinding::new();
    for (gate, ps) in [
        (GateKind::Rbs, vec![0.7]),
        (GateKind::CX, vec![]),
        (GateKind::CRz, vec![0.3]),
        (GateKind::Swap, vec![]),
    ] {
        let c = circuit(2, gate.clone(), &[0, 0], &ps);
        match be.expectation(&c, &params, &z0()) {
            Err(OmegaError::InvalidCircuit(msg)) => {
                assert!(msg.contains("twice"), "{gate:?}: {msg}");
            }
            other => panic!("{gate:?} q0,q0 must be refused as InvalidCircuit, got {other:?}"),
        }
    }
}

#[test]
fn a_three_qubit_gate_with_a_repeated_qubit_is_refused() {
    let c = circuit(3, GateKind::CCX, &[0, 1, 0], &[]);
    let res = PauliPropBackend::new().expectation(&c, &ParameterBinding::new(), &z0());
    assert!(
        matches!(res, Err(OmegaError::InvalidCircuit(_))),
        "ccx q0,q1,q0 must be refused, got {res:?}"
    );
}

#[test]
fn a_gate_past_the_register_is_refused() {
    let c = circuit(2, GateKind::CX, &[0, 2], &[]);
    let res = PauliPropBackend::new().expectation(&c, &ParameterBinding::new(), &z0());
    assert!(
        matches!(res, Err(OmegaError::InvalidCircuit(_))),
        "cx q0,q2 on a 2-qubit register must be refused, got {res:?}"
    );
}

#[test]
fn distinct_qubits_still_run() {
    let c = circuit(2, GateKind::Rbs, &[0, 1], &[0.7]);
    let params = ParameterBinding::new();
    let got = PauliPropBackend::new()
        .expectation(&c, &params, &z0())
        .expect("well-formed circuit runs");
    let want = StatevectorBackend::new()
        .expectation(&c, &params, &z0())
        .expect("reference");
    assert!(
        (got - want).abs() < 1e-9,
        "pauliprop {got} vs statevector {want}"
    );
}
