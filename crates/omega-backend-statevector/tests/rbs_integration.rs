//! Integration tests for the RBS (Givens rotation) gate:
//! matrix action, Hamming-weight preservation, additivity, the exact
//! QASM decomposition, and gradient agreement across
//! parameter-shift (4-term Givens rule) / finite-difference / adjoint.
//!
//! RBS(θ) = exp(−i·θ/2·(Y⊗X − X⊗Y)): identity on {|00⟩, |11⟩},
//! [[cos θ, −sin θ], [sin θ, cos θ]] on span{|01⟩, |10⟩}
//! (basis label |ab⟩, a = op.qubits[0] = MSB of the gate matrix).

use num_complex::Complex64;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::gradient::{compute_gradient, GradMethod};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn gate(kind: GateKind, qubits: &[u32], params: &[ParamExpr]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().cloned().collect(),
        classical_bit: None,
        condition: None,
    }
}

fn statevector(circuit: &CircuitIR, params: &ParameterBinding) -> Vec<Complex64> {
    let backend = StatevectorBackend::new();
    let config = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    match backend.execute(circuit, params, &config).unwrap() {
        ExecResult::Statevector(sv) => sv,
        other => panic!("expected statevector, got {other:?}"),
    }
}

/// |01⟩ (a=qubit0=0, b=qubit1=1) → cos θ|01⟩ + sin θ|10⟩.
#[test]
fn test_rbs_rotates_the_single_excitation_subspace() {
    let theta = 0.7317;
    let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
    // Prepare |ab⟩ = |01⟩: qubit1 (=b) to |1⟩.
    circuit.add_op(gate(GateKind::X, &[1], &[]));
    circuit.add_op(gate(GateKind::Rbs, &[0, 1], &[ParamExpr::Concrete(theta)]));

    let sv = statevector(&circuit, &ParameterBinding::new());
    // Amplitude index: bit k ↔ qubit k. |a=0,b=1⟩ → index 0b10 = 2;
    // |a=1,b=0⟩ → index 0b01 = 1.
    assert!((sv[2].re - theta.cos()).abs() < 1e-12, "cos component");
    assert!((sv[1].re - theta.sin()).abs() < 1e-12, "sin component");
    assert!(sv[0].norm() < 1e-12 && sv[3].norm() < 1e-12);
    assert!(
        sv[1].im.abs() < 1e-12 && sv[2].im.abs() < 1e-12,
        "real gate"
    );
}

/// RBS acts as identity on the Hamming-weight-0 and -2 subspaces.
#[test]
fn test_rbs_preserves_hamming_weight() {
    let theta = 1.234;
    for prep in [vec![], vec![0u32, 1u32]] {
        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        for &q in &prep {
            circuit.add_op(gate(GateKind::X, &[q], &[]));
        }
        circuit.add_op(gate(GateKind::Rbs, &[0, 1], &[ParamExpr::Concrete(theta)]));
        let sv = statevector(&circuit, &ParameterBinding::new());
        let expect_idx = prep.iter().fold(0usize, |acc, &q| acc | (1 << q));
        assert!(
            (sv[expect_idx].re - 1.0).abs() < 1e-12,
            "|{expect_idx:02b}⟩ must be fixed by RBS"
        );
    }
}

/// RBS(θ₁)·RBS(θ₂) = RBS(θ₁+θ₂) on the same pair (same generator).
#[test]
fn test_rbs_angles_add() {
    let (t1, t2) = (0.41, -1.13);
    let mut c_two = CircuitIR::new(2, CircuitType::GateBased);
    c_two.add_op(gate(GateKind::X, &[1], &[]));
    c_two.add_op(gate(GateKind::Rbs, &[0, 1], &[ParamExpr::Concrete(t1)]));
    c_two.add_op(gate(GateKind::Rbs, &[0, 1], &[ParamExpr::Concrete(t2)]));

    let mut c_one = CircuitIR::new(2, CircuitType::GateBased);
    c_one.add_op(gate(GateKind::X, &[1], &[]));
    c_one.add_op(gate(
        GateKind::Rbs,
        &[0, 1],
        &[ParamExpr::Concrete(t1 + t2)],
    ));

    let b = ParameterBinding::new();
    let (sv2, sv1) = (statevector(&c_two, &b), statevector(&c_one, &b));
    for i in 0..4 {
        assert!((sv2[i] - sv1[i]).norm() < 1e-12, "additivity at index {i}");
    }
}

/// The QASM-export decomposition is exact:
/// RBS(θ) = (H⊗H)·CZ·(Ry(−θ)⊗Ry(θ))·CZ·(H⊗H)   (full angle θ).
#[test]
fn test_rbs_decomposition_matches_native_gate() {
    let theta = 0.937;
    // Non-trivial input state on 2 qubits.
    let prep = |c: &mut CircuitIR| {
        c.add_op(gate(GateKind::Ry, &[0], &[ParamExpr::Concrete(0.63)]));
        c.add_op(gate(GateKind::Ry, &[1], &[ParamExpr::Concrete(-1.21)]));
        c.add_op(gate(GateKind::CX, &[0, 1], &[]));
    };

    let mut native = CircuitIR::new(2, CircuitType::GateBased);
    prep(&mut native);
    native.add_op(gate(GateKind::Rbs, &[0, 1], &[ParamExpr::Concrete(theta)]));

    let mut decomp = CircuitIR::new(2, CircuitType::GateBased);
    prep(&mut decomp);
    decomp.add_op(gate(GateKind::H, &[0], &[]));
    decomp.add_op(gate(GateKind::H, &[1], &[]));
    decomp.add_op(gate(GateKind::CZ, &[0, 1], &[]));
    decomp.add_op(gate(GateKind::Ry, &[0], &[ParamExpr::Concrete(-theta)]));
    decomp.add_op(gate(GateKind::Ry, &[1], &[ParamExpr::Concrete(theta)]));
    decomp.add_op(gate(GateKind::CZ, &[0, 1], &[]));
    decomp.add_op(gate(GateKind::H, &[0], &[]));
    decomp.add_op(gate(GateKind::H, &[1], &[]));

    let b = ParameterBinding::new();
    let (sv_n, sv_d) = (statevector(&native, &b), statevector(&decomp, &b));
    for i in 0..4 {
        assert!(
            (sv_n[i] - sv_d[i]).norm() < 1e-12,
            "decomposition mismatch at index {i}: {} vs {}",
            sv_n[i],
            sv_d[i]
        );
    }
}

/// Build the 3-qubit test circuit: entangled prep, symbolic RBS, and a
/// trailing rotation so the RBS is mid-circuit.
fn rbs_grad_circuit() -> CircuitIR {
    let mut circuit = CircuitIR::new(3, CircuitType::GateBased);
    circuit.symbols.insert(0, "theta".to_string());
    circuit.add_op(gate(GateKind::Ry, &[0], &[ParamExpr::Concrete(0.4)]));
    circuit.add_op(gate(GateKind::X, &[1], &[]));
    circuit.add_op(gate(GateKind::CX, &[0, 2], &[]));
    circuit.add_op(gate(GateKind::Rbs, &[0, 1], &[ParamExpr::Symbol(0)]));
    circuit.add_op(gate(GateKind::Rbs, &[1, 2], &[ParamExpr::Symbol(0)]));
    circuit.add_op(gate(GateKind::Ry, &[2], &[ParamExpr::Concrete(-0.9)]));
    circuit
}

fn observable() -> Observable {
    Observable {
        terms: vec![
            (0.7, vec![(0, PauliOp::Z)]),
            (-0.4, vec![(1, PauliOp::Z), (2, PauliOp::Z)]),
            (0.25, vec![(0, PauliOp::X), (1, PauliOp::Y)]),
        ],
    }
}

/// The 4-term Givens parameter-shift rule matches finite differences
/// and adjoint AD — including the shared-symbol chain rule (θ drives
/// two RBS gates).
#[test]
fn test_rbs_parameter_shift_matches_fd_and_adjoint() {
    let backend = StatevectorBackend::new();
    let circuit = rbs_grad_circuit();
    let obs = observable();
    let mut params = ParameterBinding::new();
    params.bind(0, 0.83);

    let g_ps = compute_gradient(
        &backend,
        &circuit,
        &params,
        &obs,
        &GradMethod::ParameterShift,
    )
    .unwrap()[0]
        .1;
    let g_fd = compute_gradient(
        &backend,
        &circuit,
        &params,
        &obs,
        &GradMethod::FiniteDifference { epsilon: 1e-6 },
    )
    .unwrap()[0]
        .1;
    let g_ad =
        compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Adjoint).unwrap()[0].1;

    assert!(
        (g_ps - g_fd).abs() < 1e-5,
        "PSR {g_ps} vs finite-difference {g_fd}"
    );
    assert!((g_ps - g_ad).abs() < 1e-9, "PSR {g_ps} vs adjoint {g_ad}");
}

/// Pin the failure mode the Givens rule exists for: the naive 2-term
/// ±π/2 rule is *not* exact for RBS (frequencies {1, 2}).
#[test]
fn test_rbs_two_term_rule_would_be_wrong() {
    let backend = StatevectorBackend::new();
    let circuit = rbs_grad_circuit();
    let obs = observable();
    let mut params = ParameterBinding::new();
    params.bind(0, 0.83);

    let exact =
        compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Adjoint).unwrap()[0].1;

    // Hand-rolled 2-term global shift of the symbol.
    let shift = std::f64::consts::FRAC_PI_2;
    let expect = |v: f64| -> f64 {
        let mut p = ParameterBinding::new();
        p.bind(0, v);
        backend.expectation(&circuit, &p, &obs).unwrap()
    };
    let two_term = (expect(0.83 + shift) - expect(0.83 - shift)) / 2.0;

    assert!(
        (two_term - exact).abs() > 1e-3,
        "2-term rule unexpectedly matched ({two_term} vs {exact}); \
         the Givens 4-term rule would be redundant"
    );
}

/// Chain rule through ParamExpr::Mul: RBS(0.5·θ) gradient is half of
/// the slot derivative (and still needs the Givens rule internally).
#[test]
fn test_rbs_scaled_symbol_chain_rule() {
    let backend = StatevectorBackend::new();
    let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
    circuit.symbols.insert(0, "theta".to_string());
    circuit.add_op(gate(GateKind::X, &[1], &[]));
    circuit.add_op(gate(
        GateKind::Rbs,
        &[0, 1],
        &[ParamExpr::Mul(
            Box::new(ParamExpr::Concrete(0.5)),
            Box::new(ParamExpr::Symbol(0)),
        )],
    ));
    let obs = Observable {
        terms: vec![(1.0, vec![(1, PauliOp::Z)])],
    };
    let mut params = ParameterBinding::new();
    params.bind(0, 1.1);

    // ⟨Z₁⟩ after RBS(0.5θ) on |01⟩: state cos(0.5θ)|01⟩ + sin(0.5θ)|10⟩
    // → ⟨Z₁⟩ = sin²(0.5θ) − cos²(0.5θ) = −cos(θ); d/dθ = sin(θ).
    let expected = (1.1f64).sin();
    let g_ps = compute_gradient(
        &backend,
        &circuit,
        &params,
        &obs,
        &GradMethod::ParameterShift,
    )
    .unwrap()[0]
        .1;
    assert!(
        (g_ps - expected).abs() < 1e-9,
        "scaled-symbol PSR {g_ps} vs analytic {expected}"
    );
}
