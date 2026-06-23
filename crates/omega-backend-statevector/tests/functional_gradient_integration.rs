//! Phase 4 — Fast AD for functions of measurements.
//!
//! Eight tests covering:
//! - Functional::Qubo / Table → diagonal-observable construction
//! - DiagonalObservable gradients match adjoint-on-built-observable
//! - QUBO-cost gradient for a parameterised circuit
//! - ScoreFunction (REINFORCE) estimator unbiased on a small fixture
//! - Identity sanity, num_qubit guard.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::gradient::{
    compute_functional_gradient, compute_gradient, Functional, FunctionalGradMethod, GradMethod,
};
use omega_core::params::ParameterBinding;
use omega_core::qubo::Qubo;
use smallvec::smallvec;

fn ry_circuit_1q(symbol_id: SymbolId, name: &str) -> CircuitIR {
    let mut c = CircuitIR::new(1, CircuitType::GateBased);
    c.symbols.insert(symbol_id, name.to_string());
    c.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(symbol_id)],
        classical_bit: None,
        condition: None,
    });
    c
}

fn rx_pair_2q() -> CircuitIR {
    // Two independent Rx gates, no entanglement: bits sampled from
    // Bernoulli(sin²(θ_i/2)) marginals. Useful for closed-form QUBO gradient.
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.symbols.insert(0, "t0".to_string());
    c.symbols.insert(1, "t1".to_string());
    c.add_op(GateOp {
        gate: GateKind::Rx,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    c.add_op(GateOp {
        gate: GateKind::Rx,
        qubits: smallvec![Qubit(1)],
        params: smallvec![ParamExpr::Symbol(1)],
        classical_bit: None,
        condition: None,
    });
    c
}

#[test]
fn test_functional_qubo_to_diagonal_observable_matches_ising() {
    // The Functional::Qubo path must produce the same observable as
    // qubo.to_ising().to_observable() — the existing "manual" path.
    let mut q = Qubo::new(3);
    q.add(0, 0, -1.0);
    q.add(1, 1, -1.0);
    q.add(0, 1, 2.0);
    q.add(1, 2, -1.5);
    let f = Functional::Qubo(q.clone());
    let obs_via_functional = f.to_diagonal_observable();
    let obs_via_ising = q.to_ising().to_observable();

    // Same set of (coeff, paulis) terms, possibly reordered. Convert to
    // a canonical form for comparison.
    let canonicalise = |o: &Observable| -> Vec<(f64, Vec<(u32, u8)>)> {
        // Map PauliOp → u8 for Ord; we never use I in these terms so any
        // total order works.
        let pauli_rank = |p: &PauliOp| match p {
            PauliOp::I => 0,
            PauliOp::X => 1,
            PauliOp::Y => 2,
            PauliOp::Z => 3,
        };
        let mut v: Vec<(f64, Vec<(u32, u8)>)> = o
            .terms
            .iter()
            .map(|(c, t)| {
                let mut term: Vec<(u32, u8)> = t.iter().map(|(q, p)| (*q, pauli_rank(p))).collect();
                term.sort();
                (*c, term)
            })
            .collect();
        v.sort_by(|a, b| a.1.cmp(&b.1));
        v
    };
    assert_eq!(
        canonicalise(&obs_via_functional),
        canonicalise(&obs_via_ising)
    );
}

#[test]
fn test_functional_table_walsh_hadamard_roundtrip() {
    // Build a tiny f: {0,1}^2 → R, lift it to a Pauli-Z sum, and verify
    // that <x|O_f|x> = f(x) for all x.
    let rows = vec![
        (vec![false, false], 0.5),
        (vec![true, false], -1.2),
        (vec![false, true], 0.3),
        (vec![true, true], 2.7),
    ];
    let f = Functional::Table {
        num_qubits: 2,
        rows: rows.clone(),
    };
    let obs = f.to_diagonal_observable();

    // Evaluate <x|O|x> for each computational basis state.
    for (bits, expected) in &rows {
        // For each basis state, walk the Pauli-Z sum: Z_i contributes ±1 for x_i.
        let mut acc = 0.0;
        for (coeff, term) in &obs.terms {
            let mut sign = 1.0;
            for (qubit, op) in term {
                assert!(matches!(op, PauliOp::Z), "expected Z-only term");
                if bits[*qubit as usize] {
                    sign = -sign;
                }
            }
            acc += coeff * sign;
        }
        assert!(
            (acc - expected).abs() < 1e-10,
            "f({:?}) reconstructed = {} expected {}",
            bits,
            acc,
            expected
        );
    }
}

#[test]
fn test_diagonal_observable_gradient_matches_adjoint_on_built_observable() {
    // For a Functional::Qubo, computing the gradient through
    // FunctionalGradMethod::DiagonalObservable must agree (numerically)
    // with running adjoint AD on the Pauli-Z observable directly.
    let backend = StatevectorBackend::new();
    let circuit = rx_pair_2q();
    let mut params = ParameterBinding::new();
    params.bind(0, 0.7);
    params.bind(1, 1.1);

    let mut q = Qubo::new(2);
    q.add(0, 0, -1.0);
    q.add(1, 1, -1.0);
    q.add(0, 1, 2.0);
    let f = Functional::Qubo(q.clone());

    let g_functional = compute_functional_gradient(
        &backend,
        &circuit,
        &params,
        &f,
        &FunctionalGradMethod::DiagonalObservable,
    )
    .unwrap();

    let observable = q.to_ising().to_observable();
    let g_adjoint = compute_gradient(
        &backend,
        &circuit,
        &params,
        &observable,
        &GradMethod::Adjoint,
    )
    .unwrap();

    assert_eq!(g_functional.len(), g_adjoint.len());
    for ((id_a, va), (id_b, vb)) in g_functional.iter().zip(g_adjoint.iter()) {
        assert_eq!(id_a, id_b);
        assert!(
            (va - vb).abs() < 1e-10,
            "param {}: functional grad {} vs adjoint grad {}",
            id_a,
            va,
            vb
        );
    }
}

#[test]
fn test_diagonal_observable_qubo_grad_closed_form_2q() {
    // Closed-form check on Rx(θ_0) ⊗ Rx(θ_1).
    // |ψ⟩ = (cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩) on each qubit, so
    //   p(x_i = 1) = sin²(θ_i / 2)
    // For Qubo Q with diagonal -1 and (0,1)=2:
    //   E[f] = -p_0 - p_1 + 2 p_0 p_1
    // Gradient w.r.t. θ_0:
    //   dp_0/dθ_0 = sin(θ_0)/2
    //   dE/dθ_0 = (-1 + 2 p_1) · sin(θ_0)/2
    let backend = StatevectorBackend::new();
    let circuit = rx_pair_2q();
    let theta0 = 0.4;
    let theta1 = 1.3;
    let mut params = ParameterBinding::new();
    params.bind(0, theta0);
    params.bind(1, theta1);

    let mut q = Qubo::new(2);
    q.add(0, 0, -1.0);
    q.add(1, 1, -1.0);
    q.add(0, 1, 2.0);
    let f = Functional::Qubo(q);

    let grads = compute_functional_gradient(
        &backend,
        &circuit,
        &params,
        &f,
        &FunctionalGradMethod::DiagonalObservable,
    )
    .unwrap();

    let p0 = (theta0 / 2.0).sin().powi(2);
    let p1 = (theta1 / 2.0).sin().powi(2);
    let expected_d0 = (-1.0 + 2.0 * p1) * theta0.sin() / 2.0;
    let expected_d1 = (-1.0 + 2.0 * p0) * theta1.sin() / 2.0;

    assert!(
        (grads[0].1 - expected_d0).abs() < 1e-9,
        "dE/dθ0: got {}, expected {}",
        grads[0].1,
        expected_d0
    );
    assert!(
        (grads[1].1 - expected_d1).abs() < 1e-9,
        "dE/dθ1: got {}, expected {}",
        grads[1].1,
        expected_d1
    );
}

#[test]
fn test_diagonal_observable_table_gradient_matches_qubo() {
    // Express the same QUBO as a Table by enumerating bit-strings;
    // gradients must agree exactly with the Qubo path.
    let backend = StatevectorBackend::new();
    let circuit = rx_pair_2q();
    let mut params = ParameterBinding::new();
    params.bind(0, 0.6);
    params.bind(1, 1.0);

    let mut q = Qubo::new(2);
    q.add(0, 0, -1.0);
    q.add(1, 1, -1.0);
    q.add(0, 1, 2.0);
    let rows: Vec<(Vec<bool>, f64)> = (0..4)
        .map(|i| {
            let bits = vec![i & 1 == 1, (i >> 1) & 1 == 1];
            let v = q.evaluate(&bits);
            (bits, v)
        })
        .collect();

    let f_qubo = Functional::Qubo(q);
    let f_table = Functional::Table {
        num_qubits: 2,
        rows,
    };

    let g_qubo = compute_functional_gradient(
        &backend,
        &circuit,
        &params,
        &f_qubo,
        &FunctionalGradMethod::DiagonalObservable,
    )
    .unwrap();
    let g_table = compute_functional_gradient(
        &backend,
        &circuit,
        &params,
        &f_table,
        &FunctionalGradMethod::DiagonalObservable,
    )
    .unwrap();

    assert_eq!(g_qubo.len(), g_table.len());
    for ((_, vq), (_, vt)) in g_qubo.iter().zip(g_table.iter()) {
        assert!(
            (vq - vt).abs() < 1e-9,
            "qubo grad {} vs table grad {}",
            vq,
            vt
        );
    }
}

#[test]
fn test_diagonal_observable_identity_functional_zero_grad() {
    // f ≡ const: gradient must be exactly zero.
    let backend = StatevectorBackend::new();
    let circuit = ry_circuit_1q(0, "theta");
    let mut params = ParameterBinding::new();
    params.bind(0, 0.5);

    let f = Functional::Table {
        num_qubits: 1,
        rows: vec![(vec![false], 1.0), (vec![true], 1.0)],
    };
    let grads = compute_functional_gradient(
        &backend,
        &circuit,
        &params,
        &f,
        &FunctionalGradMethod::DiagonalObservable,
    )
    .unwrap();
    assert!(
        grads[0].1.abs() < 1e-12,
        "constant functional must yield zero gradient, got {}",
        grads[0].1
    );
}

#[test]
fn test_score_function_estimator_close_to_analytic() {
    // ScoreFunction (REINFORCE) is unbiased but noisy. Pick a regime
    // with reasonable variance: 1q Ry(θ) + linear functional f(0)=0,
    // f(1)=1, so E[f] = sin²(θ/2), dE/dθ = sin(θ)/2.
    let backend = StatevectorBackend::new();
    let circuit = ry_circuit_1q(0, "theta");
    let theta = 0.9;
    let mut params = ParameterBinding::new();
    params.bind(0, theta);

    let f = Functional::Table {
        num_qubits: 1,
        rows: vec![(vec![false], 0.0), (vec![true], 1.0)],
    };

    let grads = compute_functional_gradient(
        &backend,
        &circuit,
        &params,
        &f,
        &FunctionalGradMethod::ScoreFunction { shots: 4096 },
    )
    .unwrap();

    let expected = theta.sin() / 2.0;
    // Tolerance: for f ∈ {0,1}, var of estimator is bounded by f² ≈ 1; sample
    // stdev across 4096 shots and 1 param ≈ 1/sqrt(4096) ≈ 0.016. Use 5σ.
    assert!(
        (grads[0].1 - expected).abs() < 0.08,
        "score-function grad = {}, expected ≈ {} (analytic)",
        grads[0].1,
        expected
    );
}

#[test]
fn test_functional_num_qubits_mismatch_rejected() {
    // 1-qubit circuit, 2-qubit functional → graceful error, no panic.
    let backend = StatevectorBackend::new();
    let circuit = ry_circuit_1q(0, "theta");
    let mut params = ParameterBinding::new();
    params.bind(0, 0.5);

    let f = Functional::Table {
        num_qubits: 2,
        rows: vec![],
    };
    let err = compute_functional_gradient(
        &backend,
        &circuit,
        &params,
        &f,
        &FunctionalGradMethod::DiagonalObservable,
    )
    .unwrap_err();
    let s = format!("{:?}", err);
    assert!(
        s.contains("num_qubits") || s.contains("Unsupported"),
        "expected num_qubits-mismatch error, got: {}",
        s
    );
}
