// SPDX-License-Identifier: Apache-2.0
//! What the sector backend refuses, and — the point of deciding on the matrix
//! — what it admits although the gate NAME looks non-conserving.
use omega_backend_sector::SectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::error::OmegaError;
use omega_core::executor::{Backend, ExecConfig, MidCircuitMode, Observable};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

fn circuit(n: u32, ops: Vec<GateOp>) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.ops = ops;
    c
}

fn run(c: &CircuitIR) -> Result<f64, OmegaError> {
    SectorBackend::new().expectation(c, &ParameterBinding::new(), &Observable::z(0))
}

fn refused(c: &CircuitIR, needle: &str) {
    match run(c) {
        Err(OmegaError::Unsupported(m)) => {
            assert!(m.starts_with("sector backend:"), "{m}");
            assert!(m.contains(needle), "message {m:?} lacks {needle:?}");
        }
        other => panic!("expected Unsupported({needle}), got {other:?}"),
    }
}

#[test]
fn non_conserving_gates_are_refused_by_their_matrix() {
    refused(&circuit(2, vec![op(GateKind::H, &[0], &[])]), "H");
    refused(&circuit(2, vec![op(GateKind::Rx, &[0], &[0.3])]), "Rx");
    refused(
        &circuit(2, vec![op(GateKind::Ry, &[1], &[-0.3])]),
        "off-diagonal",
    );
    refused(&circuit(2, vec![op(GateKind::Sx, &[0], &[])]), "Sx");
    refused(&circuit(2, vec![op(GateKind::CX, &[0, 1], &[])]), "CX");
    refused(&circuit(2, vec![op(GateKind::CY, &[0, 1], &[])]), "CY");
    refused(
        &circuit(2, vec![op(GateKind::CU3, &[0, 1], &[0.5, 0.0, 0.0])]),
        "connects |00⟩/|11⟩ with |01⟩/|10⟩",
    );
    refused(&circuit(2, vec![op(GateKind::U2, &[0], &[0.1, 0.2])]), "U2");
    refused(
        &circuit(3, vec![op(GateKind::CCX, &[0, 1, 2], &[])]),
        "three-qubit",
    );
    refused(
        &circuit(3, vec![op(GateKind::CSwap, &[0, 1, 2], &[])]),
        "three-qubit",
    );
    refused(&circuit(2, vec![op(GateKind::Reset, &[0], &[])]), "Reset");
    refused(&circuit(2, vec![op(GateKind::Y, &[0], &[])]), "Y");
}

#[test]
fn zero_angle_rotations_are_admitted_because_their_matrix_is() {
    // Rx(0) = I, U3(0, φ, λ) = diag(1, e^{i(φ+λ)}), CU3(0, φ, λ) = cphase.
    for c in [
        circuit(
            2,
            vec![op(GateKind::X, &[0], &[]), op(GateKind::Rx, &[0], &[0.0])],
        ),
        circuit(
            2,
            vec![op(GateKind::X, &[0], &[]), op(GateKind::Ry, &[1], &[0.0])],
        ),
        circuit(
            2,
            vec![
                op(GateKind::X, &[0], &[]),
                op(GateKind::U3, &[0], &[0.0, 0.4, 0.5]),
            ],
        ),
        circuit(
            2,
            vec![
                op(GateKind::X, &[0], &[]),
                op(GateKind::CU3, &[1, 0], &[0.0, 0.4, 0.5]),
            ],
        ),
    ] {
        assert!((run(&c).unwrap() - (-1.0)).abs() < 1e-15, "{:?}", c.ops);
    }
}

#[test]
fn x_is_the_occupation_layer_and_nothing_else() {
    // First on its wire: fine, in any position of the op list.
    let ok = circuit(
        3,
        vec![
            op(GateKind::X, &[0], &[]),
            op(GateKind::Rbs, &[0, 1], &[0.3]),
            op(GateKind::X, &[2], &[]),
            op(GateKind::Rbs, &[1, 2], &[0.3]),
        ],
    );
    run(&ok).unwrap();
    // After a gate on the same wire: refused, even after a diagonal one.
    refused(
        &circuit(
            2,
            vec![op(GateKind::Z, &[0], &[]), op(GateKind::X, &[0], &[])],
        ),
        "occupation layer",
    );
    refused(
        &circuit(
            2,
            vec![
                op(GateKind::Rbs, &[0, 1], &[0.1]),
                op(GateKind::X, &[1], &[]),
            ],
        ),
        "occupation layer",
    );
    // Two X on one wire: the second is not first on its wire.
    refused(
        &circuit(
            2,
            vec![op(GateKind::X, &[0], &[]), op(GateKind::X, &[0], &[])],
        ),
        "occupation layer",
    );
    // Barrier and Id do not touch the wire.
    run(&circuit(
        2,
        vec![
            op(GateKind::Barrier, &[0, 1], &[]),
            op(GateKind::Id, &[0], &[]),
            op(GateKind::X, &[0], &[]),
        ],
    ))
    .unwrap();
}

#[test]
fn feed_forward_collapse_and_photonics_are_refused() {
    let mut cond = op(GateKind::Z, &[0], &[]);
    cond.condition = Some((0, 1, 1));
    refused(&circuit(2, vec![cond]), "conditioned");

    let c = circuit(2, vec![op(GateKind::X, &[0], &[])]);
    let r = SectorBackend::new().execute(
        &c,
        &ParameterBinding::new(),
        &ExecConfig {
            shots: Some(10),
            seed: Some(1),
            mid_circuit_mode: MidCircuitMode::Collapse,
        },
    );
    assert!(matches!(r, Err(OmegaError::Unsupported(_))), "{r:?}");

    let mut ph = CircuitIR::new(2, CircuitType::Photonic);
    ph.add_op(op(GateKind::PhaseShifter, &[0], &[0.1]));
    refused(&ph, "photonic");
}

#[test]
fn out_of_range_qubits_are_invalid_not_a_panic() {
    let c = circuit(2, vec![op(GateKind::Rbs, &[0, 5], &[0.1])]);
    assert!(matches!(run(&c), Err(OmegaError::InvalidCircuit(_))));
    let c = circuit(2, vec![op(GateKind::Rbs, &[1, 1], &[0.1])]);
    assert!(matches!(run(&c), Err(OmegaError::InvalidCircuit(_))));
    let c = circuit(2, vec![]);
    let o = Observable::z(7);
    assert!(SectorBackend::new()
        .expectation(&c, &ParameterBinding::new(), &o)
        .is_err());
}

#[test]
fn the_sector_cap_refuses_before_allocating() {
    let mut c = CircuitIR::new(64, CircuitType::GateBased);
    for q in 0..32 {
        c.add_op(op(GateKind::X, &[q], &[]));
    }
    match run(&c) {
        Err(OmegaError::Backend(m)) => assert!(m.contains("C(64, 32)"), "{m}"),
        other => panic!("{other:?}"),
    }
    let mut c = CircuitIR::new(65, CircuitType::GateBased);
    c.add_op(op(GateKind::X, &[0], &[]));
    refused(&c, "64 is the limit");
}
