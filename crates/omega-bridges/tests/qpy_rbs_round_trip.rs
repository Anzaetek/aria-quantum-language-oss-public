// SPDX-License-Identifier: Apache-2.0
//! `Rbs` across the pure-Rust QPY writer/reader pair.
//!
//! The writer spells `Rbs(θ)` as `XXPlusYYGate(−2θ, π/2)`; the reader
//! inverts it and REFUSES any other β. These tests pin the blob's spelling
//! at the byte level (so the Python side sees what the receipt on
//! `qpy/write.rs::qiskit_params` says it sees), the round trip, and the
//! refusal — by patching β in the blob, since no omega circuit can produce
//! a wrong β on its own.

use omega_bridges::qpy::{
    read_qpy_circuit_ir, write_qpy_circuit_ir, QpyError, RBS_XX_PLUS_YY_BETA,
};
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

fn x_then_rbs(theta: ParamExpr) -> CircuitIR {
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.add_op(op(GateKind::X, &[0], &[]));
    ir.add_op(op(GateKind::Rbs, &[0, 1], &[theta]));
    ir
}

fn count_subslice(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len()).filter(|w| *w == needle).count()
}

#[test]
fn the_blob_spells_rbs_as_xx_plus_yy_with_minus_two_theta_and_beta_half_pi() {
    let blob = write_qpy_circuit_ir(&x_then_rbs(ParamExpr::Concrete(0.3)));
    assert_eq!(
        count_subslice(&blob, b"XXPlusYYGate"),
        1,
        "one XXPlusYYGate instruction"
    );
    // Scalar params are the `b'f'` payload: an f64, little-endian.
    assert_eq!(
        count_subslice(&blob, &(-0.6f64).to_le_bytes()),
        1,
        "theta doubled AND negated"
    );
    assert_eq!(
        count_subslice(&blob, &RBS_XX_PLUS_YY_BETA.to_le_bytes()),
        1,
        "beta = pi/2"
    );
    assert_eq!(
        count_subslice(&blob, &0.3f64.to_le_bytes()),
        0,
        "raw theta must not leak through"
    );
}

#[test]
fn rbs_concrete_round_trips_bit_identically() {
    // −2·θ then −0.5·(that) are both exact in binary floating point, so the
    // claim is exactness, and the assertion is `to_bits`, not a tolerance.
    for theta in [0.3, -0.7, 1.1, 2.9] {
        let decoded = read_qpy_circuit_ir(&write_qpy_circuit_ir(&x_then_rbs(ParamExpr::Concrete(
            theta,
        ))))
        .unwrap();
        assert_eq!(decoded.ops.len(), 2);
        assert_eq!(decoded.ops[1].gate, GateKind::Rbs);
        assert_eq!(decoded.ops[1].qubits[0], Qubit(0));
        assert_eq!(decoded.ops[1].qubits[1], Qubit(1));
        match decoded.ops[1].params[0] {
            ParamExpr::Concrete(v) => assert_eq!(
                v.to_bits(),
                theta.to_bits(),
                "theta {theta} came back as {v}"
            ),
            ref other => panic!("expected a concrete theta, got {other:?}"),
        }
    }
}

#[test]
fn rbs_symbolic_round_trips_under_binding() {
    let mut ir = x_then_rbs(ParamExpr::Symbol(0));
    ir.symbols.insert(0, "theta".to_string());
    let decoded = read_qpy_circuit_ir(&write_qpy_circuit_ir(&ir)).unwrap();
    assert_eq!(decoded.ops[1].gate, GateKind::Rbs);
    let id = decoded
        .symbols
        .iter()
        .find(|(_, n)| n.as_str() == "theta")
        .map(|(id, _)| *id)
        .expect("symbol survives the wire");
    let mut binding = ParameterBinding::new();
    binding.bind(id, 0.7);
    let v = binding.resolve(&decoded.ops[1].params[0]).unwrap();
    assert!(
        (v - 0.7).abs() < 1e-12,
        "Mul(Mul(theta, -2), -0.5) must evaluate to theta, got {v}"
    );
}

#[test]
fn xx_plus_yy_with_beta_zero_is_refused_not_rounded_to_rbs() {
    // Qiskit's default β = 0 is a different unitary (−i·sin off-diagonals
    // where RBS has real ±sin). The reader must say so, not shrug it into
    // the nearest Rbs.
    let mut blob = write_qpy_circuit_ir(&x_then_rbs(ParamExpr::Concrete(0.3)));
    let beta = RBS_XX_PLUS_YY_BETA.to_le_bytes();
    let at = blob
        .windows(8)
        .position(|w| w == beta)
        .expect("beta payload present");
    blob[at..at + 8].copy_from_slice(&0.0f64.to_le_bytes());
    match read_qpy_circuit_ir(&blob) {
        Err(QpyError::Unsupported { what, .. }) => {
            assert!(
                what.contains("beta"),
                "refusal must name beta, got {what:?}"
            )
        }
        other => panic!("beta = 0 must be refused as Unsupported, got {other:?}"),
    }
}
