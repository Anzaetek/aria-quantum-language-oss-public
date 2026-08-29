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

/// Helper: `Ry(theta) q0` followed by `measure q0 -> c0`.
///
/// The measurement is **inert** — nothing reads the bit it writes and nothing
/// touches the qubit again — so under the contract in
/// `omega_core::defer_measure` it is elided and the circuit is fully unitary.
fn ry_then_inert_measure(theta: f64) -> (CircuitIR, ParameterBinding) {
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
    let mut params = ParameterBinding::new();
    params.bind(0, theta);
    (circuit, params)
}

/// `Ry(theta) q0; measure q0 -> c0; H q0` — the measured qubit is reused
/// coherently, so there is **no** deferred form and no adjoint.
fn ry_measure_then_reuse(theta: f64) -> (CircuitIR, ParameterBinding) {
    let (mut circuit, params) = ry_then_inert_measure(theta);
    circuit.add_op(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    (circuit, params)
}

/// **`Auto` picks the EXACT method when the measurement can be deferred.**
///
/// This test was `test_auto_selects_stochastic_for_measurements` and asserted
/// only `result.is_ok()`, which is why it kept passing after the selection rule
/// changed underneath it: its name described a choice it never checked.
///
/// The rule now keys off deferrability rather than "contains a measurement".
/// `StochasticParameterShift` returns a NOISY gradient, so routing every measured
/// circuit to it gave up exactness on circuits that are exactly differentiable.
/// The assertion is therefore on the VALUE — `1e-8` against `−sin θ` is a
/// tolerance only an analytic method can meet; 100-shot sampling cannot come
/// close.
#[test]
fn auto_picks_the_exact_gradient_when_the_measurement_is_inert() {
    let backend = StatevectorBackend::new();
    let theta = 0.5;
    let (circuit, params) = ry_then_inert_measure(theta);
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };

    let grads = compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Auto)
        .expect("Auto must handle a circuit with an inert measurement");
    let expected = -theta.sin();
    assert!(
        (grads[0].1 - expected).abs() < 1e-8,
        "Auto gave {} for d<Z0>/dtheta at {theta}; the exact value is {expected}. \
         A discrepancy of ~1e-2 here means Auto fell back to 100-shot \
         StochasticParameterShift, which it should now do only when the circuit \
         cannot be deferred.",
        grads[0].1
    );
}

/// **`Auto` falls back to sampling when the circuit cannot be deferred — and the
/// sampled answer is RIGHT where the adjoint's would be wrong.**
///
/// `Ry(θ) q0; measure q0 -> c0; H q0` with `Z₀`. The measurement leaves a mixture
/// of `|0⟩` and `|1⟩`, and `H` maps both to `|±⟩`, each of which has
/// `⟨Z⟩ = 0`. So `⟨Z₀⟩ = 0` for **every** θ and the gradient is **0**.
///
/// Delete the measurement, as an adjoint sweep does, and the state is
/// `H·Ry(θ)|0⟩` with `⟨Z₀⟩ = sin θ`, whose derivative is `cos θ ≈ 0.878` at
/// θ = 0.5. Truth 0 against 0.878 — the two designs are a full unit apart on an
/// observable bounded in [−1, 1], so this needs no delicate tolerance.
///
/// The first draft of this test asserted that `Auto` would ERROR here, reasoning
/// that the forward expectation is refused. That was wrong:
/// `StochasticParameterShift` never calls `expectation` — it samples the collapse
/// path, which models this circuit correctly — so `Auto` produces the right number
/// rather than no number. Asserting an error would have demanded a refusal in
/// place of a correct answer.
#[test]
fn auto_falls_back_to_sampling_when_the_measurement_cannot_be_deferred() {
    let backend = StatevectorBackend::new();
    let theta = 0.5;
    let (circuit, params) = ry_measure_then_reuse(theta);
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };
    let grads = compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Auto)
        .expect("Auto must still answer a non-deferrable circuit, by sampling");
    let got = grads[0].1;
    // 100 shots per shifted evaluation, so this is a sampled estimate of an exact
    // zero. The band is generous because the alternative it must exclude sits at
    // 0.878, not at 0.05.
    assert!(
        got.abs() < 0.35,
        "d<Z0>/dtheta is exactly 0 for this circuit: the measurement leaves a \
         mixture and H sends both branches to |+-> where <Z> = 0, for every theta. \
         Got {got}. A value near cos({theta}) = {:.3} means the measurement was \
         differentiated away and the adjoint answered a circuit nobody wrote.",
        theta.cos()
    );
    assert!(
        (got - theta.cos()).abs() > 0.4,
        "got {got}, which is close to cos(theta) = {:.3} — the deleted-measurement \
         answer. This is the specific wrong value this test exists to exclude.",
        theta.cos()
    );
}

/// **`Adjoint` requested explicitly: exact for a deferrable circuit, refused
/// otherwise.**
///
/// This replaces `test_adjoint_rejects_measurements`, which used the inert-measure
/// circuit above and required an error. The refusal was correct only while the
/// adjoint could not handle measurements at all; now the measurement is elided and
/// the exact gradient is available, so refusing would be throwing away an answer
/// we have.
#[test]
fn explicit_adjoint_is_exact_when_deferrable_and_refuses_when_not() {
    let backend = StatevectorBackend::new();
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };

    let theta = 0.5;
    let (circuit, params) = ry_then_inert_measure(theta);
    let grads = compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Adjoint)
        .expect("an inert measurement is elided, so the adjoint applies");
    let expected = -theta.sin();
    assert!(
        (grads[0].1 - expected).abs() < 1e-8,
        "adjoint gave {} for d<Z0>/dtheta at {theta}; exact is {expected}",
        grads[0].1
    );

    let (circuit, params) = ry_measure_then_reuse(theta);
    let err = compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Adjoint)
        .expect_err("a coherently reused measured qubit has no adjoint")
        .to_string();
    assert!(
        err.contains("cannot be deferred") || err.contains("used again"),
        "the error should say WHY there is no adjoint, not just that there is \
         none; got: {err}"
    );
}
