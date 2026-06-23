//! End-to-end QPY round-trip fixtures.
//!
//! Each test builds a `CircuitIR` shaped after one of the
//! `verify-qiskit/fixtures/*` circuit families (bell, ghz, qft-like,
//! mid-circuit-meas, etc.), pipes it through `write_qpy_circuit_ir`,
//! reads it back via `read_qpy_circuit_ir`, and asserts that the
//! reconstructed circuit preserves the observable structure of the
//! original — gate kinds, qubit indices, parameter values under any
//! `ParameterBinding`, measurement targets, and conditional gates.
//!
//! These tests pin the writer's claim that "Concrete + symbolic
//! ParamExpr + single-clbit conditions round-trip end-to-end". They
//! sit alongside the per-module unit tests so a regression in any of
//! the writer/reader/builder chain trips here before it reaches the
//! verify-qiskit suite.

use omega_bridges::qpy::{read_qpy_circuit_ir, write_qpy_circuit_ir};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn op(gate: GateKind, qubits: &[u32], params: &[ParamExpr]) -> GateOp {
    let mut q = smallvec![];
    for &i in qubits {
        q.push(Qubit(i));
    }
    let mut p = smallvec![];
    for x in params {
        p.push(x.clone());
    }
    GateOp {
        gate,
        qubits: q,
        params: p,
        classical_bit: None,
        condition: None,
    }
}

#[test]
fn ghz_4q_round_trips_shape() {
    // H q[0]; CX q[0],q[1]; CX q[1],q[2]; CX q[2],q[3];
    let mut ir = CircuitIR::new(4, CircuitType::GateBased);
    ir.add_op(op(GateKind::H, &[0], &[]));
    ir.add_op(op(GateKind::CX, &[0, 1], &[]));
    ir.add_op(op(GateKind::CX, &[1, 2], &[]));
    ir.add_op(op(GateKind::CX, &[2, 3], &[]));

    let decoded = read_qpy_circuit_ir(&write_qpy_circuit_ir(&ir)).unwrap();
    assert_eq!(decoded.num_qubits, 4);
    assert_eq!(decoded.ops.len(), 4);
    assert_eq!(decoded.ops[0].gate, GateKind::H);
    assert_eq!(decoded.ops[1].gate, GateKind::CX);
    assert_eq!(decoded.ops[1].qubits[0], Qubit(0));
    assert_eq!(decoded.ops[1].qubits[1], Qubit(1));
    assert_eq!(decoded.ops[2].qubits[0], Qubit(1));
    assert_eq!(decoded.ops[3].qubits[1], Qubit(3));
}

#[test]
fn qft_like_3q_with_concrete_rotations_round_trips() {
    // H + a fan of CRz rotations followed by a SWAP — the rotation
    // angles are concrete so they bit-exact round-trip through the
    // b'f' INSTRUCTION_PARAM payload.
    let mut ir = CircuitIR::new(3, CircuitType::GateBased);
    ir.add_op(op(GateKind::H, &[0], &[]));
    ir.add_op(op(
        GateKind::CRz,
        &[0, 1],
        &[ParamExpr::Concrete(std::f64::consts::FRAC_PI_2)],
    ));
    ir.add_op(op(
        GateKind::CRz,
        &[0, 2],
        &[ParamExpr::Concrete(std::f64::consts::FRAC_PI_4)],
    ));
    ir.add_op(op(GateKind::H, &[1], &[]));
    ir.add_op(op(GateKind::Swap, &[0, 2], &[]));

    let decoded = read_qpy_circuit_ir(&write_qpy_circuit_ir(&ir)).unwrap();
    assert_eq!(decoded.ops.len(), 5);
    let binding = ParameterBinding::new();
    let theta_crz_01 = binding.resolve(&decoded.ops[1].params[0]).unwrap();
    let theta_crz_02 = binding.resolve(&decoded.ops[2].params[0]).unwrap();
    assert!(
        (theta_crz_01 - std::f64::consts::FRAC_PI_2).abs() < 1e-15,
        "CRz(0,1) angle must bit-exact round-trip"
    );
    assert!(
        (theta_crz_02 - std::f64::consts::FRAC_PI_4).abs() < 1e-15,
        "CRz(0,2) angle must bit-exact round-trip"
    );
}

#[test]
fn parametric_2q_ansatz_round_trips_symbol_structure() {
    // A 2-qubit hardware-efficient ansatz with shared parameters across
    // gates — exercises symbol-uuid deduplication: the writer must emit
    // one symbol-map entry per unique SymbolId regardless of how many
    // PARAM_EXPR_ELEM records reference it.
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.symbols.insert(0, "theta_0".to_string());
    ir.symbols.insert(1, "theta_1".to_string());
    ir.add_op(op(GateKind::Ry, &[0], &[ParamExpr::Symbol(0)]));
    ir.add_op(op(GateKind::Ry, &[1], &[ParamExpr::Symbol(1)]));
    ir.add_op(op(GateKind::CX, &[0, 1], &[]));
    // theta_0 + theta_1, exercising the binary-op encoder path.
    ir.add_op(op(
        GateKind::Rz,
        &[0],
        &[ParamExpr::Add(
            Box::new(ParamExpr::Symbol(0)),
            Box::new(ParamExpr::Symbol(1)),
        )],
    ));

    let decoded = read_qpy_circuit_ir(&write_qpy_circuit_ir(&ir)).unwrap();
    assert_eq!(decoded.ops.len(), 4);
    assert_eq!(decoded.symbols.len(), 2);
    // theta_0 + theta_1 must evaluate to the sum under any binding.
    let t0 = decoded
        .symbols
        .iter()
        .find(|(_, n)| n.as_str() == "theta_0")
        .map(|(id, _)| *id)
        .unwrap();
    let t1 = decoded
        .symbols
        .iter()
        .find(|(_, n)| n.as_str() == "theta_1")
        .map(|(id, _)| *id)
        .unwrap();
    let mut binding = ParameterBinding::new();
    binding.bind(t0, 0.3);
    binding.bind(t1, 1.4);
    let combined = binding.resolve(&decoded.ops[3].params[0]).unwrap();
    assert!(
        (combined - (0.3 + 1.4)).abs() < 1e-12,
        "Rz(theta_0 + theta_1) must evaluate to 1.7 under the test binding, got {combined}"
    );
}

#[test]
fn mid_circuit_measurement_with_conditional_x_round_trips() {
    // Mid-circuit-meas pattern: H q[0]; measure q[0] -> c[0]; if(c0==1) X q[1].
    // Exercises both classical_bit routing (Measure carg) and the
    // single-clbit condition path landed in stage 3b.
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.num_classical_bits = 1;
    ir.add_op(op(GateKind::H, &[0], &[]));
    ir.add_op(GateOp {
        gate: GateKind::Measure,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: Some(0),
        condition: None,
    });
    ir.add_op(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: Some((0, 1, 1)),
    });

    let decoded = read_qpy_circuit_ir(&write_qpy_circuit_ir(&ir)).unwrap();
    assert_eq!(decoded.ops.len(), 3);
    assert_eq!(decoded.ops[1].gate, GateKind::Measure);
    assert_eq!(decoded.ops[1].classical_bit, Some(0));
    assert_eq!(decoded.ops[2].gate, GateKind::X);
    assert_eq!(decoded.ops[2].condition, Some((0, 1, 1)));
}

#[test]
fn u3_phase_three_param_concrete_round_trips() {
    // U3(theta, phi, lambda) — exercises the 3-param INSTRUCTION_PARAM
    // sequence. Each scalar gets its own b'f' record so the reader
    // sees three independent payloads.
    let mut ir = CircuitIR::new(1, CircuitType::GateBased);
    ir.add_op(op(
        GateKind::U3,
        &[0],
        &[
            ParamExpr::Concrete(0.7),
            ParamExpr::Concrete(-0.4),
            ParamExpr::Concrete(1.2),
        ],
    ));

    let decoded = read_qpy_circuit_ir(&write_qpy_circuit_ir(&ir)).unwrap();
    let binding = ParameterBinding::new();
    let theta = binding.resolve(&decoded.ops[0].params[0]).unwrap();
    let phi = binding.resolve(&decoded.ops[0].params[1]).unwrap();
    let lam = binding.resolve(&decoded.ops[0].params[2]).unwrap();
    assert!((theta - 0.7).abs() < 1e-15);
    assert!((phi - -0.4).abs() < 1e-15);
    assert!((lam - 1.2).abs() < 1e-15);
}

#[test]
fn random_clifford_5q_round_trips_via_native_set() {
    // Random Clifford circuits in verify-qiskit/fixtures/08 use
    // H/S/Sdg/X/Y/Z/CX. Exercise the full single-qubit-no-param
    // family + CX, all back-to-back, to flush out any gate-name
    // mapping regression.
    let mut ir = CircuitIR::new(5, CircuitType::GateBased);
    for q in 0..5 {
        ir.add_op(op(GateKind::H, &[q], &[]));
    }
    for q in 0..5 {
        ir.add_op(op(GateKind::S, &[q], &[]));
        ir.add_op(op(GateKind::Sdg, &[q], &[]));
    }
    ir.add_op(op(GateKind::CX, &[0, 1], &[]));
    ir.add_op(op(GateKind::CX, &[1, 2], &[]));
    ir.add_op(op(GateKind::Y, &[3], &[]));
    ir.add_op(op(GateKind::Z, &[4], &[]));

    let decoded = read_qpy_circuit_ir(&write_qpy_circuit_ir(&ir)).unwrap();
    assert_eq!(decoded.num_qubits, 5);
    assert_eq!(decoded.ops.len(), ir.ops.len());
    for (i, (decoded_op, original_op)) in decoded.ops.iter().zip(ir.ops.iter()).enumerate() {
        assert_eq!(
            decoded_op.gate, original_op.gate,
            "op {i} gate kind must match after round-trip"
        );
        assert_eq!(
            decoded_op.qubits.as_slice(),
            original_op.qubits.as_slice(),
            "op {i} qubits must match after round-trip"
        );
    }
}

#[test]
fn negate_inside_add_round_trips_under_evaluation() {
    // `Rx(-(theta) + 1.0)` — nests Negate inside Add, exercising both
    // the Negate → Mul(_, -1) rewrite and the binary Add op together.
    let mut ir = CircuitIR::new(1, CircuitType::GateBased);
    ir.symbols.insert(0, "theta".to_string());
    let expr = ParamExpr::Add(
        Box::new(ParamExpr::Negate(Box::new(ParamExpr::Symbol(0)))),
        Box::new(ParamExpr::Concrete(1.0)),
    );
    ir.add_op(op(GateKind::Rx, &[0], &[expr]));

    let decoded = read_qpy_circuit_ir(&write_qpy_circuit_ir(&ir)).unwrap();
    let theta_id = decoded
        .symbols
        .iter()
        .find(|(_, n)| n.as_str() == "theta")
        .map(|(id, _)| *id)
        .unwrap();
    let mut binding = ParameterBinding::new();
    binding.bind(theta_id, 0.4);
    let v = binding.resolve(&decoded.ops[0].params[0]).unwrap();
    assert!(
        (v - (1.0 - 0.4)).abs() < 1e-12,
        "Rx(-theta + 1.0) with theta=0.4 must evaluate to 0.6, got {v}"
    );
}

#[test]
fn reset_and_barrier_pass_through() {
    // Reset and Barrier are no-param zero-carg gates with a varying
    // qubit count. Stage 2 added them; pin here that they survive a
    // round-trip in a realistic order.
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.add_op(op(GateKind::H, &[0], &[]));
    ir.add_op(op(GateKind::Reset, &[0], &[]));
    ir.add_op(op(GateKind::Barrier, &[0, 1], &[]));
    ir.add_op(op(GateKind::X, &[1], &[]));

    let decoded = read_qpy_circuit_ir(&write_qpy_circuit_ir(&ir)).unwrap();
    assert_eq!(decoded.ops.len(), 4);
    assert_eq!(decoded.ops[1].gate, GateKind::Reset);
    assert_eq!(decoded.ops[2].gate, GateKind::Barrier);
    assert_eq!(decoded.ops[2].qubits.len(), 2);
}
