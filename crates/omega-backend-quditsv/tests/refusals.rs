// SPDX-License-Identifier: Apache-2.0
//! **What this exact engine refuses, and that it refuses BEFORE allocating.**
//! (PLAN-QUDIT.md Q2 certificate story: exact is honest only because the
//! capacity gate refuses what the engine could not hold exactly.)

use omega_backend_quditsv::capacity::{self, MAX_PRODUCT_DIM};
use omega_backend_quditsv::QuditSvBackend;
use omega_core::circuit::{
    CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit, QuditRegister,
};
use omega_core::error::OmegaError;
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode, Observable};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, wires: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: wires.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

fn qudits(dims: &[u32], ops: Vec<GateOp>) -> CircuitIR {
    let mut c = CircuitIR::new(dims.len() as u32, CircuitType::GateBased);
    c.qudit_registers.push(QuditRegister {
        name: "q".into(),
        start: 0,
        dims: dims.to_vec(),
    });
    c.ops = ops;
    c
}

fn refused(c: &CircuitIR, cfg: &ExecConfig, needle: &str) -> String {
    match QuditSvBackend::new().execute(c, &ParameterBinding::new(), cfg) {
        Err(OmegaError::Unsupported(m)) => {
            assert!(m.starts_with("quditsv:"), "{m}");
            assert!(m.contains(needle), "message {m:?} lacks {needle:?}");
            m
        }
        other => panic!("expected Unsupported({needle}), got {other:?}"),
    }
}

fn analytic() -> ExecConfig {
    ExecConfig {
        shots: None,
        ..Default::default()
    }
}

/// `3^16 = 43_046_721 > 2^24`: refused by the ceiling with the product and
/// the ceiling both named. The test would take seconds and ~700 MiB if the
/// engine allocated first; it returns in microseconds because it does not.
#[test]
fn a_state_over_the_ceiling_is_refused_by_arithmetic_before_allocation() {
    let c = qudits(&[3; 16], vec![]);
    let t = std::time::Instant::now();
    let m = refused(&c, &analytic(), "43046721");
    assert!(m.contains(&MAX_PRODUCT_DIM.to_string()), "{m}");
    assert!(m.contains("3×3×3"), "{m}");
    assert!(
        t.elapsed().as_millis() < 100,
        "took {:?} — did it allocate?",
        t.elapsed()
    );
    // The same check, called directly, is the one the engine uses.
    assert!(capacity::check(&[3; 16]).is_err());
    assert_eq!(capacity::check(&[3, 2, 5]).unwrap(), 30);
}

#[test]
fn a_qubit_gate_on_a_qudit_wire_is_refused_by_name_with_the_alternatives() {
    let c = qudits(&[3, 3], vec![op(GateKind::CX, &[0, 1], &[])]);
    let m = refused(&c, &analytic(), "CX is a qubit gate");
    assert!(
        m.contains("dimension 3") && m.contains("csum") && m.contains("rxy"),
        "{m}"
    );
    let c = qudits(&[3], vec![op(GateKind::Rx, &[0], &[0.3])]);
    refused(&c, &analytic(), "Rx is a qubit gate");
    let c = qudits(&[3], vec![op(GateKind::T, &[0], &[])]);
    refused(&c, &analytic(), "T is a qubit gate");
    // …while the generalised ones run on the same wire.
    for g in [GateKind::H, GateKind::X, GateKind::Z] {
        assert!(QuditSvBackend::new()
            .execute(
                &qudits(&[3], vec![op(g, &[0], &[])]),
                &ParameterBinding::new(),
                &analytic()
            )
            .is_ok());
    }
}

#[test]
fn rxy_levels_are_validated() {
    let c = qudits(&[3], vec![op(GateKind::Rxy, &[0], &[0.0, 3.0, 0.1, 0.0])]);
    refused(&c, &analytic(), "j = 3 is not a level of a d = 3 wire");
    let c = qudits(&[3], vec![op(GateKind::Rxy, &[0], &[0.5, 1.0, 0.1, 0.0])]);
    refused(&c, &analytic(), "i = 0.5 is not a level");
    let c = qudits(&[3], vec![op(GateKind::Rxy, &[0], &[2.0, 1.0, 0.1, 0.0])]);
    let m = refused(&c, &analytic(), "i < j");
    assert!(m.contains("not silently swapped"), "{m}");
}

#[test]
fn sampling_a_qudit_circuit_is_refused_naming_the_register_and_the_way_out() {
    let c = qudits(&[3, 2], vec![op(GateKind::H, &[0], &[])]);
    let cfg = ExecConfig {
        shots: Some(100),
        ..Default::default()
    };
    let m = refused(&c, &cfg, "sampling a qudit circuit is not supported yet");
    assert!(
        m.contains("register 'q'") && m.contains("dimension 3") && m.contains("--statevector"),
        "{m}"
    );
}

#[test]
fn pauli_observables_are_refused_on_qudit_wires_and_fine_on_qubit_ones() {
    let c = qudits(
        &[3, 2],
        vec![op(GateKind::H, &[0], &[]), op(GateKind::X, &[1], &[])],
    );
    let z0 = Observable::z(0);
    let err = QuditSvBackend::new()
        .expectation(&c, &ParameterBinding::new(), &z0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Z0") && err.contains("dimension 3"), "{err}");
    // Z on the qubit wire of a mixed circuit is exact: X|0⟩ → ⟨Z⟩ = −1.
    let z1 = QuditSvBackend::new()
        .expectation(&c, &ParameterBinding::new(), &Observable::z(1))
        .unwrap();
    assert!((z1 + 1.0).abs() < 1e-12, "{z1}");
}

#[test]
fn channels_conditions_and_mid_circuit_measurement_are_refused() {
    let c = qudits(&[3], vec![op(GateKind::Reset, &[0], &[])]);
    refused(&c, &analytic(), "Reset is a non-unitary channel");
    let c = qudits(
        &[3],
        vec![op(GateKind::Measure, &[0], &[]), op(GateKind::X, &[0], &[])],
    );
    refused(&c, &analytic(), "after a measurement");
    let mut cond = op(GateKind::X, &[0], &[]);
    cond.condition = Some((0, 1, 1));
    let c = qudits(&[3], vec![cond]);
    refused(&c, &analytic(), "classically-conditioned");
    let c = qudits(&[3], vec![]);
    let cfg = ExecConfig {
        shots: None,
        mid_circuit_mode: MidCircuitMode::Collapse,
        ..Default::default()
    };
    refused(&c, &cfg, "mid-circuit collapse");
    let mut ph = CircuitIR::new(2, CircuitType::Photonic);
    ph.ops = vec![];
    refused(&ph, &analytic(), "photonic");
}
