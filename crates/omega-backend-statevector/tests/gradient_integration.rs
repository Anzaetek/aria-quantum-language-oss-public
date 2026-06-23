//! Integration tests for gradient methods: stochastic parameter-shift, Auto, adjoint rejection.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::gradient::{compute_gradient, GradMethod};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

#[test]
fn test_stochastic_grad_matches_analytic() {
    // On a measurement-free circuit, stochastic PSR should converge to analytic PSR.
    // Ry(θ)|0⟩: ⟨Z⟩ = cos(θ), d⟨Z⟩/dθ = -sin(θ)
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.symbols.insert(0, "theta".to_string());
    circuit.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });

    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };

    let theta = std::f64::consts::FRAC_PI_3;
    let mut params = ParameterBinding::new();
    params.bind(0, theta);

    // Stochastic with many shots should match analytic
    let grads = compute_gradient(
        &backend,
        &circuit,
        &params,
        &obs,
        &GradMethod::StochasticParameterShift { shots: 1 },
    )
    .unwrap();

    let expected = -theta.sin();
    // With no measurements, each shot gives the same answer, so even 1 shot is exact
    assert!(
        (grads[0].1 - expected).abs() < 1e-8,
        "stochastic grad = {} (expected {})",
        grads[0].1,
        expected
    );
}

#[test]
fn test_stochastic_grad_through_measurement() {
    // Ry(θ)|0⟩ → measure → reset → state is |0⟩
    // The gradient through measurement is zero for the post-measurement state,
    // but the stochastic method should still run without errors.
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.num_classical_bits = 1;
    circuit.symbols.insert(0, "theta".to_string());

    circuit.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    circuit.add_op(GateOp {
        gate: GateKind::Measure,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: Some(0),
        condition: None,
    });
    // Reset: if measured 1, flip back
    circuit.add_op(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: Some((0, 1, 1)),
    });

    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };

    let mut params = ParameterBinding::new();
    params.bind(0, 0.5);

    // Should run without error
    let grads = compute_gradient(
        &backend,
        &circuit,
        &params,
        &obs,
        &GradMethod::StochasticParameterShift { shots: 50 },
    )
    .unwrap();

    assert_eq!(grads.len(), 1);
    // After measure+reset, state is always |0⟩, so ⟨Z⟩ ≈ 1 regardless of θ
    // Gradient should be approximately 0
    assert!(
        grads[0].1.abs() < 0.3,
        "gradient through measure+reset should be ~0, got {}",
        grads[0].1
    );
}

#[test]
fn test_auto_selects_adjoint_for_unitary() {
    // No measurements → Auto should use Adjoint (which works for unitary circuits)
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.symbols.insert(0, "theta".to_string());
    circuit.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });

    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };

    let theta = 0.8;
    let mut params = ParameterBinding::new();
    params.bind(0, theta);

    let grads = compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Auto).unwrap();

    let expected = -theta.sin();
    assert!(
        (grads[0].1 - expected).abs() < 1e-8,
        "Auto grad = {} (expected {})",
        grads[0].1,
        expected
    );
}

#[test]
fn test_auto_selects_stochastic_for_measurements() {
    // Circuit with measurements → Auto should use StochasticParameterShift
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.num_classical_bits = 1;
    circuit.symbols.insert(0, "theta".to_string());

    circuit.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    circuit.add_op(GateOp {
        gate: GateKind::Measure,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: Some(0),
        condition: None,
    });

    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };

    let mut params = ParameterBinding::new();
    params.bind(0, 0.5);

    // Should succeed (Auto picks stochastic, not adjoint which would fail)
    let result = compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Auto);
    assert!(
        result.is_ok(),
        "Auto should handle circuits with measurements"
    );
}

#[test]
fn test_adjoint_rejects_measurements() {
    // Requesting Adjoint on circuit with measurements → clear error
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.num_classical_bits = 1;
    circuit.symbols.insert(0, "theta".to_string());

    circuit.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    circuit.add_op(GateOp {
        gate: GateKind::Measure,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: Some(0),
        condition: None,
    });

    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };

    let mut params = ParameterBinding::new();
    params.bind(0, 0.5);

    let result = compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Adjoint);
    assert!(
        result.is_err(),
        "Adjoint should reject circuits with measurements"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("mid-circuit measurements"),
        "Error should mention measurements: {err}"
    );
}
