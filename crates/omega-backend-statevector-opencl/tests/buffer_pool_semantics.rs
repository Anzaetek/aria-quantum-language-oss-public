//! Smoke tests for `OpenClStatevectorBackend`'s internal `BufferPool`.
//!
//! Four anchors that mirror Metal's pool tests:
//!
//! 1. Pool starts empty — fresh backend has zero pooled buffers at
//!    every qubit count.
//! 2. After one `adjoint_gradient` call the pool holds exactly 3
//!    buffers at the circuit's qubit count (|φ⟩, |ν⟩, temp), and
//!    zero at every other count.
//! 3. After two adjoint calls on the same shape the pool stays at 3
//!    (LIFO reuse, no growth).
//! 4. Different qubit counts route to independent stacks — running
//!    on two shapes leaves both at 3 simultaneously.

#![cfg(feature = "opencl")]

use omega_backend_statevector_opencl::OpenClStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn tiny_param_circuit(n: u32) -> (CircuitIR, ParameterBinding, Observable) {
    let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
    circuit.symbols.insert(0, "theta".into());
    circuit.ops.push(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    if n >= 2 {
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(0), Qubit(1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    let mut params = ParameterBinding::new();
    params.bind(0, 0.31);
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };
    (circuit, params, obs)
}

#[test]
fn pool_starts_empty_on_fresh_backend() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    for n in 0..=8u32 {
        assert_eq!(
            backend.pool_size(n),
            0,
            "fresh backend pool must be empty at n={n}",
        );
    }
}

#[test]
fn pool_size_is_three_after_one_adjoint_call() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let (circuit, params, obs) = tiny_param_circuit(3);
    assert_eq!(backend.pool_size(3), 0);
    let _ = backend
        .adjoint_gradient(&circuit, &params, &obs)
        .expect("adjoint")
        .expect("Ok(Some)");
    assert_eq!(
        backend.pool_size(3),
        3,
        "after one adjoint call pool should hold |φ⟩ + |ν⟩ + temp at n=3",
    );
    // Other qubit counts stay empty.
    assert_eq!(backend.pool_size(4), 0);
    assert_eq!(backend.pool_size(2), 0);
}

#[test]
fn pool_steady_states_at_three_across_repeated_calls() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let (circuit, params, obs) = tiny_param_circuit(4);
    // Run the adjoint five times — pool size should stay at 3 after
    // the first call (LIFO reuse, no growth).
    for i in 0..5 {
        backend
            .adjoint_gradient(&circuit, &params, &obs)
            .expect("adjoint")
            .expect("Ok(Some)");
        assert_eq!(
            backend.pool_size(4),
            3,
            "after call {i} pool size must steady-state at 3, got {}",
            backend.pool_size(4),
        );
    }
}

#[test]
fn pool_uses_independent_stacks_per_qubit_count() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let (c3, p3, o3) = tiny_param_circuit(3);
    let (c5, p5, o5) = tiny_param_circuit(5);
    backend.adjoint_gradient(&c3, &p3, &o3).unwrap().unwrap();
    backend.adjoint_gradient(&c5, &p5, &o5).unwrap().unwrap();
    assert_eq!(backend.pool_size(3), 3);
    assert_eq!(backend.pool_size(5), 3);
    // Sanity: leasing 3q never reused a pooled 5q buffer (byte sizes
    // differ), so each stack carries its own three entries.
    assert_eq!(backend.pool_size(4), 0);
}
