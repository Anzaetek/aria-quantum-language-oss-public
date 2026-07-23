//! Integration tests for the parallelised parameter-shift rule
//! (arXiv:2606.03517): the trailing commuting-block gradients from a
//! single batched evaluation must equal serial parameter-shift and
//! adjoint AD, and the execution count must collapse to 1 per layer.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::gradient::{compute_gradient, GradMethod};
use omega_core::parallel_shift::parallel_parameter_shift_gradient;
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

/// A 4-qubit toy butterfly: unary data-load, one frozen inner layer,
/// then the trained final coupling layer of 2 disjoint RBS gates.
fn butterfly_circuit() -> CircuitIR {
    let mut c = CircuitIR::new(4, CircuitType::GateBased);
    for (id, name) in [(0, "phi_0"), (1, "phi_1")] {
        c.symbols.insert(id, name.to_string());
    }
    // Unary-ish load: |1000⟩ superposed a bit.
    c.add_op(gate(GateKind::X, &[0], &[]));
    c.add_op(gate(GateKind::Ry, &[2], &[ParamExpr::Concrete(0.35)]));
    // Frozen inner layer (concrete angles, stride-1 pairs).
    c.add_op(gate(GateKind::Rbs, &[0, 1], &[ParamExpr::Concrete(0.61)]));
    c.add_op(gate(GateKind::Rbs, &[2, 3], &[ParamExpr::Concrete(-0.42)]));
    // Trained coupling layer (stride-2 pairs) — the trailing block.
    c.add_op(gate(GateKind::Rbs, &[0, 2], &[ParamExpr::Symbol(0)]));
    c.add_op(gate(GateKind::Rbs, &[1, 3], &[ParamExpr::Symbol(1)]));
    c
}

fn observable() -> Observable {
    Observable {
        terms: vec![
            (1.0, vec![(0, PauliOp::Z)]),
            (-0.5, vec![(1, PauliOp::Z)]),
            (0.25, vec![(2, PauliOp::Z), (3, PauliOp::Z)]),
        ],
    }
}

fn bind() -> ParameterBinding {
    let mut params = ParameterBinding::new();
    params.bind(0, 0.53);
    params.bind(1, -1.07);
    params
}

#[test]
fn test_parallel_matches_serial_and_adjoint() {
    let backend = StatevectorBackend::new();
    let circuit = butterfly_circuit();
    let obs = observable();
    let params = bind();

    let (par, report) =
        parallel_parameter_shift_gradient(&backend, &circuit, &params, &obs, None).unwrap();
    let ser = compute_gradient(
        &backend,
        &circuit,
        &params,
        &obs,
        &GradMethod::ParameterShift,
    )
    .unwrap();
    let adj = compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Adjoint).unwrap();

    assert_eq!(par.len(), 2);
    for i in 0..2 {
        assert_eq!(par[i].0, ser[i].0);
        assert!(
            (par[i].1 - ser[i].1).abs() < 1e-9,
            "parallel {} vs serial {} for symbol {}",
            par[i].1,
            ser[i].1,
            par[i].0
        );
        assert!(
            (par[i].1 - adj[i].1).abs() < 1e-9,
            "parallel {} vs adjoint {} for symbol {}",
            par[i].1,
            adj[i].1,
            par[i].0
        );
    }

    // The whole trained layer costs ONE circuit execution (vs 4 per
    // gate = 8 for serial 4-term shifts).
    assert_eq!(report.block_symbols, 2);
    assert_eq!(report.fallback_symbols, 0);
    assert_eq!(report.circuit_executions, 1);
    assert_eq!(report.block_gates, 2);
}

#[test]
fn test_parallel_falls_back_for_out_of_block_symbols() {
    // Move symbol 0 in FRONT of an entangler so it can't sit in the
    // trailing block; symbol 1 stays in the block. Gradients must
    // still match the serial reference for both.
    let backend = StatevectorBackend::new();
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    c.symbols.insert(0, "early".to_string());
    c.symbols.insert(1, "late".to_string());
    c.add_op(gate(GateKind::X, &[0], &[]));
    c.add_op(gate(GateKind::Rbs, &[0, 1], &[ParamExpr::Symbol(0)]));
    c.add_op(gate(GateKind::CX, &[0, 2], &[]));
    c.add_op(gate(GateKind::Rbs, &[0, 1], &[ParamExpr::Symbol(1)]));

    let obs = Observable {
        terms: vec![(1.0, vec![(1, PauliOp::Z)]), (0.5, vec![(2, PauliOp::Z)])],
    };
    let mut params = ParameterBinding::new();
    params.bind(0, 0.91);
    params.bind(1, -0.33);

    let (par, report) =
        parallel_parameter_shift_gradient(&backend, &c, &params, &obs, None).unwrap();
    let ser = compute_gradient(&backend, &c, &params, &obs, &GradMethod::ParameterShift).unwrap();

    for i in 0..2 {
        assert!(
            (par[i].1 - ser[i].1).abs() < 1e-9,
            "symbol {}: parallel {} vs serial {}",
            par[i].0,
            par[i].1,
            ser[i].1
        );
    }
    assert_eq!(report.block_symbols, 1);
    assert_eq!(report.fallback_symbols, 1);
}

#[test]
fn test_parallel_respects_subset() {
    let backend = StatevectorBackend::new();
    let circuit = butterfly_circuit();
    let obs = observable();
    let params = bind();

    let only: std::collections::HashSet<SymbolId> = [1u32].into_iter().collect();
    let (par, _) =
        parallel_parameter_shift_gradient(&backend, &circuit, &params, &obs, Some(&only)).unwrap();
    assert_eq!(par.len(), 1);
    assert_eq!(par[0].0, 1);

    let ser = compute_gradient(
        &backend,
        &circuit,
        &params,
        &obs,
        &GradMethod::ParameterShift,
    )
    .unwrap();
    assert!((par[0].1 - ser[1].1).abs() < 1e-9);
}

#[test]
fn test_gradmethod_variant_routes_to_parallel() {
    let backend = StatevectorBackend::new();
    let circuit = butterfly_circuit();
    let obs = observable();
    let params = bind();

    let via_enum = compute_gradient(
        &backend,
        &circuit,
        &params,
        &obs,
        &GradMethod::ParallelParameterShift,
    )
    .unwrap();
    let ser = compute_gradient(
        &backend,
        &circuit,
        &params,
        &obs,
        &GradMethod::ParameterShift,
    )
    .unwrap();
    for i in 0..2 {
        assert!((via_enum[i].1 - ser[i].1).abs() < 1e-9);
    }
}

/// A Clifford readout layer after the trained block must NOT break the
/// one-execution property: generators are conjugated through the
/// suffix (i·[B·G_k·B†, H]) and the gradients still match serial
/// parameter-shift exactly.
#[test]
fn test_parallel_conjugates_through_clifford_suffix() {
    let backend = StatevectorBackend::new();
    // Exercise every supported suffix gate, including the phase-carrying
    // S/Sdg and the two-qubit entanglers.
    let suffixes: Vec<Vec<GateOp>> = vec![
        vec![gate(GateKind::H, &[0], &[]), gate(GateKind::H, &[2], &[])],
        vec![gate(GateKind::S, &[1], &[]), gate(GateKind::Sdg, &[3], &[])],
        vec![
            gate(GateKind::CX, &[0, 3], &[]),
            gate(GateKind::CZ, &[1, 2], &[]),
        ],
        vec![
            gate(GateKind::Swap, &[0, 2], &[]),
            gate(GateKind::Y, &[1], &[]),
            gate(GateKind::S, &[2], &[]),
            gate(GateKind::CX, &[2, 1], &[]),
            gate(GateKind::H, &[3], &[]),
        ],
    ];
    let obs = observable();
    let params = bind();
    for (case, suffix) in suffixes.into_iter().enumerate() {
        let mut circuit = butterfly_circuit();
        let n_suffix = suffix.len();
        for op in suffix {
            circuit.add_op(op);
        }
        let (par, report) =
            parallel_parameter_shift_gradient(&backend, &circuit, &params, &obs, None).unwrap();
        let ser = compute_gradient(
            &backend,
            &circuit,
            &params,
            &obs,
            &GradMethod::ParameterShift,
        )
        .unwrap();
        for i in 0..2 {
            assert!(
                (par[i].1 - ser[i].1).abs() < 1e-9,
                "case {case}: parallel {} vs serial {} for symbol {}",
                par[i].1,
                ser[i].1,
                par[i].0
            );
        }
        assert_eq!(report.block_symbols, 2, "case {case}");
        assert_eq!(
            report.circuit_executions, 1,
            "case {case}: still ONE execution"
        );
        assert_eq!(report.clifford_suffix_gates, n_suffix, "case {case}");
    }
}

/// A non-Clifford trailing gate (T) ends the suffix scan — the block
/// detection must fall back cleanly rather than mis-conjugate.
#[test]
fn test_non_clifford_suffix_falls_back() {
    let backend = StatevectorBackend::new();
    let mut circuit = butterfly_circuit();
    circuit.add_op(gate(GateKind::T, &[0], &[]));
    let obs = observable();
    let params = bind();
    let (par, report) =
        parallel_parameter_shift_gradient(&backend, &circuit, &params, &obs, None).unwrap();
    let ser = compute_gradient(
        &backend,
        &circuit,
        &params,
        &obs,
        &GradMethod::ParameterShift,
    )
    .unwrap();
    for i in 0..2 {
        assert!((par[i].1 - ser[i].1).abs() < 1e-9);
    }
    // T is not a supported suffix gate: the symbols escape the block and
    // take the serial path — correct, just not batched.
    assert_eq!(report.block_symbols, 0);
    assert_eq!(report.fallback_symbols, 2);
    assert_eq!(report.clifford_suffix_gates, 0);
}
