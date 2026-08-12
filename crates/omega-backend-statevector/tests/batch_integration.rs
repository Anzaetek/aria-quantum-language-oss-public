// SPDX-License-Identifier: Apache-2.0
//! The row-batched `expectation_batch` / `adjoint_gradient_batch` must be
//! bit-for-bit identical to the sequential per-binding loop — the guarantee
//! that lets a supervised trainer parallelise over data rows without
//! perturbing seeded (reproducible) results.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::params::ParameterBinding;

fn gate(kind: GateKind, qubits: &[u32], params: &[ParamExpr]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().cloned().collect(),
        classical_bit: None,
        condition: None,
    }
}

/// A 3-qubit parametric circuit with two feature symbols and one weight.
fn circuit() -> CircuitIR {
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    for (id, n) in [(0, "x_0"), (1, "x_1"), (2, "w")] {
        c.symbols.insert(id, n.to_string());
    }
    c.add_op(gate(GateKind::Ry, &[0], &[ParamExpr::Symbol(0)]));
    c.add_op(gate(GateKind::Ry, &[1], &[ParamExpr::Symbol(1)]));
    c.add_op(gate(GateKind::CX, &[0, 1], &[]));
    c.add_op(gate(GateKind::Rz, &[2], &[ParamExpr::Symbol(2)]));
    c.add_op(gate(GateKind::CX, &[1, 2], &[]));
    c
}

fn rows() -> Vec<ParameterBinding> {
    (0..64)
        .map(|i| {
            let t = i as f64 * 0.1;
            let mut b = ParameterBinding::new();
            b.bind(0, (t).sin());
            b.bind(1, (t * 0.7).cos());
            b.bind(2, 0.3 + 0.01 * t);
            b
        })
        .collect()
}

#[test]
fn expectation_batch_matches_sequential() {
    let backend = StatevectorBackend::new();
    let c = circuit();
    let obs = Observable::parse("0.5*Z0 + Z1 Z2").unwrap();
    let bnds = rows();
    let refs: Vec<&ParameterBinding> = bnds.iter().collect();

    let batched = backend.expectation_batch(&c, &refs, &obs).unwrap();
    let sequential: Vec<f64> = bnds
        .iter()
        .map(|b| backend.expectation(&c, b, &obs).unwrap())
        .collect();

    assert_eq!(batched.len(), sequential.len());
    for (i, (a, b)) in batched.iter().zip(&sequential).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "row {i}: {a} != {b} (not bit-identical)"
        );
    }
}

#[test]
fn adjoint_gradient_batch_matches_sequential() {
    let backend = StatevectorBackend::new();
    let c = circuit();
    let obs = Observable::parse("Z2").unwrap();
    let bnds = rows();
    let refs: Vec<&ParameterBinding> = bnds.iter().collect();

    let batched = backend.adjoint_gradient_batch(&c, &refs, &obs).unwrap();
    let sequential: Vec<_> = bnds
        .iter()
        .map(|b| backend.adjoint_gradient(&c, b, &obs).unwrap())
        .collect();

    assert_eq!(batched.len(), sequential.len());
    for (i, (a, b)) in batched.iter().zip(&sequential).enumerate() {
        let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
        assert_eq!(a.len(), b.len());
        for ((sa, ga), (sb, gb)) in a.iter().zip(b) {
            assert_eq!(sa, sb, "row {i}: symbol order differs");
            assert_eq!(
                ga.to_bits(),
                gb.to_bits(),
                "row {i} sym {sa}: gradient not bit-identical"
            );
        }
    }
}
