//! Integration tests for QML inference pipeline and circuit optimization.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::device::DeviceKind;
use omega_core::executor::*;
use omega_core::optimize::optimize;
use omega_core::params::ParameterBinding;
use omega_core::qml::*;
use smallvec::smallvec;

// ---- QML Inference ----

#[test]
fn test_qml_infer_expectation() {
    // Angle encode [π] → Ry(π)|0⟩ = |1⟩ → ⟨Z⟩ = -1
    let model = QmlModel {
        num_qubits: 1,
        encoding: Encoding::Angle,
        ansatz: CircuitIR::new(1, CircuitType::GateBased),
        measurement_qubits: vec![0],
        output_mode: OutputMode::Expectation,
    };

    let backend = StatevectorBackend::new();
    let params = ParameterBinding::new();
    let result = infer(&model, &[std::f64::consts::PI], &params, &backend).unwrap();

    assert_eq!(result.len(), 1);
    assert!(
        (result[0] - (-1.0)).abs() < 1e-8,
        "Ry(pi)|0> should give <Z> = -1, got {}",
        result[0]
    );
}

#[test]
fn test_qml_infer_with_ansatz() {
    // Angle encode [0] → |0⟩, then Ry(θ) ansatz → ⟨Z⟩ = cos(θ)
    let mut ansatz = CircuitIR::new(1, CircuitType::GateBased);
    ansatz.symbols.insert(0, "theta".to_string());
    ansatz.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });

    let model = QmlModel {
        num_qubits: 1,
        encoding: Encoding::Angle,
        ansatz,
        measurement_qubits: vec![0],
        output_mode: OutputMode::Expectation,
    };

    let backend = StatevectorBackend::new();
    let theta = 1.2;
    let mut params = ParameterBinding::new();
    params.bind(0, theta);

    let result = infer(&model, &[0.0], &params, &backend).unwrap();
    assert!(
        (result[0] - theta.cos()).abs() < 1e-8,
        "expected cos({}) = {}, got {}",
        theta,
        theta.cos(),
        result[0]
    );
}

#[test]
fn test_qml_infer_sample_mode_multi_qubit_correlation() {
    // Bell state on 2 qubits — H + CX (no explicit Measure ops; the
    // sampler reads the final amplitudes directly). The Bell state
    // (|00⟩+|11⟩)/√2 only ever produces basis states 00 and 11, so
    // bit 0 of the sampled bitstring equals bit 1 on every shot.
    // ⟨Z_0⟩ and ⟨Z_1⟩ must therefore be bit-for-bit equal across
    // any number of shots. Pins the M>1 measurement-qubit path of
    // `infer_sample` — the single-pass loop accumulates one running
    // sum per measurement qubit; if either qubit's accumulator
    // were mis-indexed (or `total` re-derived per qubit) the
    // exact-equality assertion would catch it.
    let mut ansatz = CircuitIR::new(2, CircuitType::GateBased);
    ansatz.add_op(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    ansatz.add_op(GateOp {
        gate: GateKind::CX,
        qubits: smallvec![Qubit(0), Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });

    let model = QmlModel {
        num_qubits: 2,
        encoding: Encoding::Angle,
        ansatz,
        measurement_qubits: vec![0, 1],
        output_mode: OutputMode::Sample { shots: 512 },
    };

    let backend = StatevectorBackend::new();
    // Encoding Ry(0) = identity, so the input doesn't perturb the
    // initial |00⟩ — H + CX then produces the canonical Bell state.
    let result = infer(&model, &[0.0, 0.0], &ParameterBinding::new(), &backend).unwrap();
    assert_eq!(result.len(), 2, "two measurement qubits → two values");
    assert!(
        (result[0] - result[1]).abs() < 1e-12,
        "Bell-state samples must produce identical ⟨Z_0⟩ / ⟨Z_1⟩, got {} vs {}",
        result[0],
        result[1]
    );
    // Each must lie within a generous statistical band around 0
    // (Bell expectation is exactly 0 in the limit).
    assert!(
        result[0].abs() < 0.25,
        "⟨Z_0⟩ on Bell samples should be near 0, got {}",
        result[0]
    );
}

#[test]
fn test_qml_infer_sample_mode() {
    // Encode [π] → |1⟩, measure → ⟨Z⟩ ≈ -1
    let mut ansatz = CircuitIR::new(1, CircuitType::GateBased);
    ansatz.num_classical_bits = 1;
    ansatz.add_op(GateOp {
        gate: GateKind::Measure,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: Some(0),
        condition: None,
    });

    let model = QmlModel {
        num_qubits: 1,
        encoding: Encoding::Angle,
        ansatz,
        measurement_qubits: vec![0],
        output_mode: OutputMode::Sample { shots: 100 },
    };

    let backend = StatevectorBackend::new();
    let result = infer(
        &model,
        &[std::f64::consts::PI],
        &ParameterBinding::new(),
        &backend,
    )
    .unwrap();
    assert!(
        (result[0] - (-1.0)).abs() < 0.1,
        "expected <Z> ~ -1 for |1>, got {}",
        result[0]
    );
}

#[test]
fn test_qml_midcircuit_measure_reset() {
    // Surrogate QML: encode → measure → conditional reset → Ry(θ)
    let mut ansatz = CircuitIR::new(1, CircuitType::GateBased);
    ansatz.num_classical_bits = 1;
    ansatz.add_op(GateOp {
        gate: GateKind::Measure,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: Some(0),
        condition: None,
    });
    ansatz.add_op(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: Some((0, 1, 1)),
    });
    ansatz.symbols.insert(0, "theta".to_string());
    ansatz.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });

    let model = QmlModel {
        num_qubits: 1,
        encoding: Encoding::Angle,
        ansatz,
        measurement_qubits: vec![0],
        output_mode: OutputMode::Sample { shots: 200 },
    };

    let backend = StatevectorBackend::new();
    let mut params = ParameterBinding::new();
    params.bind(0, 0.0);

    // Any data → measure+reset → |0⟩ → Ry(0) → |0⟩ → ⟨Z⟩ ≈ 1
    let result = infer(&model, &[1.5], &params, &backend).unwrap();
    assert!(
        (result[0] - 1.0).abs() < 0.3,
        "after measure+reset+Ry(0), <Z> should be ~1, got {}",
        result[0]
    );
}

// ---- Circuit Optimization ----

#[test]
fn test_optimize_preserves_semantics() {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.ops.push(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }); // cancels
    c.ops.push(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }); // stays
    c.ops.push(GateOp {
        gate: GateKind::CX,
        qubits: smallvec![Qubit(0), Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::Rz,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Concrete(0.3)],
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::Rz,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Concrete(0.5)],
        classical_bit: None,
        condition: None,
    }); // merges

    let sv_before = execute_sv(&c);

    let mut c_opt = c.clone();
    let removed = optimize(&mut c_opt);
    assert!(removed > 0);

    let sv_after = execute_sv(&c_opt);

    for (i, (a, b)) in sv_before.iter().zip(sv_after.iter()).enumerate() {
        assert!(
            (a - b).norm() < 1e-10,
            "amplitude[{i}] differs: before={a}, after={b}"
        );
    }
}

/// Verify optimization preserves semantics on a commutation-heavy circuit.
#[test]
fn test_optimize_preserves_semantics_commutation() {
    // Rz(0.3, q0) → CX(0,1) → Rz(0.5, q0) → H(q1)
    // Commuting Rz past CX then merging should give same result
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.ops.push(GateOp {
        gate: GateKind::Rz,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Concrete(0.3)],
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::CX,
        qubits: smallvec![Qubit(0), Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::Rz,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Concrete(0.5)],
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });

    assert_optimization_preserves(&c);
}

/// Verify optimization preserves semantics on a multi-gate cancellation circuit.
#[test]
fn test_optimize_preserves_semantics_multi_cancel() {
    // X X Y Y CZ CZ H on various qubits — lots of cancellations
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    c.ops.push(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(0)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }); // cancels
    c.ops.push(GateOp {
        gate: GateKind::CZ,
        qubits: smallvec![Qubit(0), Qubit(2)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::CZ,
        qubits: smallvec![Qubit(0), Qubit(2)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }); // cancels
    c.ops.push(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(2)],
        params: smallvec![ParamExpr::Concrete(1.1)],
        classical_bit: None,
        condition: None,
    });

    assert_optimization_preserves(&c);
}

/// Verify optimization preserves semantics on a QAOA circuit.
#[test]
fn test_optimize_preserves_semantics_qaoa() {
    use omega_core::qaoa::qaoa_circuit;
    use omega_core::qubo::Qubo;

    let mut q = Qubo::new(3);
    q.set(0, 1, 1.0);
    q.set(1, 2, -0.5);
    let ising = q.to_ising();
    let circuit = qaoa_circuit(&ising, 2);

    // QAOA circuits have symbolic params — bind them for execution
    let mut params = ParameterBinding::new();
    for &id in circuit.symbols.keys() {
        params.bind(id, 0.5);
    }

    let sv_before = execute_sv_with_params(&circuit, &params);

    let mut c_opt = circuit.clone();
    optimize(&mut c_opt);

    let sv_after = execute_sv_with_params(&c_opt, &params);
    for (i, (a, b)) in sv_before.iter().zip(sv_after.iter()).enumerate() {
        assert!(
            (a - b).norm() < 1e-10,
            "QAOA amplitude[{i}] differs: before={a}, after={b}"
        );
    }
}

fn assert_optimization_preserves(circuit: &CircuitIR) {
    let sv_before = execute_sv(circuit);
    let mut c_opt = circuit.clone();
    optimize(&mut c_opt);
    let sv_after = execute_sv(&c_opt);
    assert_eq!(sv_before.len(), sv_after.len());
    for (i, (a, b)) in sv_before.iter().zip(sv_after.iter()).enumerate() {
        assert!(
            (a - b).norm() < 1e-10,
            "amplitude[{i}] differs: before={a}, after={b}"
        );
    }
}

fn execute_sv(circuit: &CircuitIR) -> Vec<num_complex::Complex64> {
    execute_sv_with_params(circuit, &ParameterBinding::new())
}

fn execute_sv_with_params(
    circuit: &CircuitIR,
    params: &ParameterBinding,
) -> Vec<num_complex::Complex64> {
    let backend = StatevectorBackend::new();
    let config = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    backend
        .execute(circuit, params, &config)
        .unwrap()
        .statevector()
        .to_vec()
}

// ---- QML training (Phase 4) ----

#[test]
fn test_qml_trainer_converges_on_toy_regression() {
    // 2-qubit, 2-param Ry ansatz; learn to map angle-encoded x to a
    // pair of target ⟨Z_0⟩, ⟨Z_1⟩ values. CPU backend; tests that the
    // QmlTrainer.fit loop drives the loss down.
    let mut ansatz = CircuitIR::new(2, CircuitType::GateBased);
    ansatz.symbols.insert(0, "theta_0".to_string());
    ansatz.symbols.insert(1, "theta_1".to_string());
    ansatz.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    ansatz.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(1)],
        params: smallvec![ParamExpr::Symbol(1)],
        classical_bit: None,
        condition: None,
    });

    let model = QmlModel {
        num_qubits: 2,
        encoding: Encoding::Angle,
        ansatz,
        measurement_qubits: vec![0, 1],
        output_mode: OutputMode::Expectation,
    };

    // Synthetic training data: 8 random-ish points; targets are the
    // model's own output at a known parameter setting (so the trainer
    // is learning to recover those parameters).
    let true_params = [0.7_f64, -1.1];
    let backend = StatevectorBackend::new();
    let mut train_x: Vec<Vec<f64>> = Vec::new();
    let mut train_y: Vec<Vec<f64>> = Vec::new();
    for k in 0..8 {
        let x_a = (k as f64) * 0.31 - 0.4;
        let x_b = -(k as f64) * 0.17 + 0.2;
        let x = vec![x_a, x_b];
        let mut p = ParameterBinding::new();
        p.bind(0, true_params[0]);
        p.bind(1, true_params[1]);
        let y = infer(&model, &x, &p, &backend).unwrap();
        train_x.push(x);
        train_y.push(y);
    }

    // Initialise parameters away from the true ones; train.
    let mut params = ParameterBinding::new();
    params.bind(0, 0.0);
    params.bind(1, 0.0);
    let trainer = QmlTrainer::new(&model)
        .epochs(80)
        .learning_rate(0.4)
        .seed(7);
    let history = trainer
        .fit(&backend, &mut params, &train_x, &train_y)
        .expect("fit");

    // Loss must drop and the recovered parameters must be close to true.
    let first = history.loss_per_epoch.first().copied().unwrap_or(f64::NAN);
    let last = history.loss_per_epoch.last().copied().unwrap_or(f64::NAN);
    assert!(
        last < first * 0.1,
        "loss did not decrease enough: first={first}, last={last}"
    );
    let learned_0 = params.resolve(&ParamExpr::Symbol(0)).unwrap();
    let learned_1 = params.resolve(&ParamExpr::Symbol(1)).unwrap();
    let err_0 = (learned_0 - true_params[0]).abs();
    let err_1 = (learned_1 - true_params[1]).abs();
    // Toy gradient descent on a 2-param non-convex landscape — accept
    // anything within 0.1 of the true parameters.
    assert!(err_0 < 0.1, "theta_0 error = {err_0}");
    assert!(err_1 < 0.1, "theta_1 error = {err_1}");
}

#[test]
fn test_qml_trainer_optimize_matches_unoptimized() {
    // Pin: turning the per-point CircuitIR optimisation pass on
    // (default) must produce identical training trajectories vs
    // turning it off. The pass folds Ry(x_q) + Ry(theta_q) into a
    // single Ry(x_q + theta_q) which the adjoint chain-rules through
    // ParamExpr::Add — so loss_per_epoch and final params should
    // match to f64 precision.
    let mut ansatz = CircuitIR::new(3, CircuitType::GateBased);
    for s in 0..3u32 {
        ansatz.symbols.insert(s, format!("theta_{s}"));
    }
    // Layer 1: Ry per qubit — directly chains with the encoding's
    // Ry(x_q), so rotation-merging will fold them.
    for q in 0..3 {
        ansatz.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(q)],
            classical_bit: None,
            condition: None,
        });
    }
    // Entangling CX ladder, then a final un-merged layer of Rz.
    for q in 0..2 {
        ansatz.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(q), Qubit(q + 1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }

    let model = QmlModel {
        num_qubits: 3,
        encoding: Encoding::Angle,
        ansatz,
        measurement_qubits: vec![0, 2],
        output_mode: OutputMode::Expectation,
    };

    let backend = StatevectorBackend::new();
    let mut train_x: Vec<Vec<f64>> = Vec::new();
    let mut train_y: Vec<Vec<f64>> = Vec::new();
    for k in 0..6 {
        let x = vec![
            (k as f64) * 0.21 - 0.3,
            (k as f64) * 0.17 - 0.1,
            (k as f64) * -0.19 + 0.4,
        ];
        train_x.push(x);
        train_y.push(vec![0.3 - 0.05 * k as f64, -0.2 + 0.07 * k as f64]);
    }

    let make_params = || {
        let mut p = ParameterBinding::new();
        p.bind(0, 0.4);
        p.bind(1, -0.7);
        p.bind(2, 0.2);
        p
    };
    let make_trainer = || {
        QmlTrainer::new(&model)
            .epochs(15)
            .learning_rate(0.3)
            .seed(11)
    };

    let mut params_opt = make_params();
    let history_opt = make_trainer()
        .optimize(true)
        .fit(&backend, &mut params_opt, &train_x, &train_y)
        .expect("fit optimize=on");

    let mut params_raw = make_params();
    let history_raw = make_trainer()
        .optimize(false)
        .fit(&backend, &mut params_raw, &train_x, &train_y)
        .expect("fit optimize=off");

    // Loss curves must match closely (small ParamExpr::Add chain-
    // rule reorderings produce the same f64 result modulo summation
    // order — well within 1e-10).
    assert_eq!(
        history_opt.loss_per_epoch.len(),
        history_raw.loss_per_epoch.len()
    );
    for (e, (a, b)) in history_opt
        .loss_per_epoch
        .iter()
        .zip(history_raw.loss_per_epoch.iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-10,
            "epoch {e}: opt_loss = {a}, raw_loss = {b}, diff = {:.3e}",
            (a - b).abs()
        );
    }
    for s in 0..3u32 {
        let a = params_opt.resolve(&ParamExpr::Symbol(s)).unwrap();
        let b = params_raw.resolve(&ParamExpr::Symbol(s)).unwrap();
        assert!(
            (a - b).abs() < 1e-10,
            "sym {s}: optimize=on => {a}, optimize=off => {b}"
        );
    }
}

#[test]
fn test_qml_trainer_device_cpu_matches_cpu_backend() {
    // `.device(Cpu)` plus the CPU StatevectorBackend must train
    // without error (the validation should pass), and the loss must
    // decrease — pinning that the device check is not a no-op that
    // accidentally short-circuits the training loop.
    let mut ansatz = CircuitIR::new(1, CircuitType::GateBased);
    ansatz.symbols.insert(0, "theta".to_string());
    ansatz.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    let model = QmlModel {
        num_qubits: 1,
        encoding: Encoding::Angle,
        ansatz,
        measurement_qubits: vec![0],
        output_mode: OutputMode::Expectation,
    };

    let backend = StatevectorBackend::new();
    let train_x: Vec<Vec<f64>> = (0..4).map(|k| vec![(k as f64) * 0.3]).collect();
    let train_y: Vec<Vec<f64>> = (0..4).map(|k| vec![0.5 - 0.1 * k as f64]).collect();

    let mut params = ParameterBinding::new();
    params.bind(0, 0.0);
    let trainer = QmlTrainer::new(&model)
        .device(DeviceKind::Cpu)
        .epochs(20)
        .learning_rate(0.3)
        .seed(11);
    let history = trainer
        .fit(&backend, &mut params, &train_x, &train_y)
        .expect("Cpu trainer + Cpu backend must succeed");
    let first = history.loss_per_epoch.first().copied().unwrap();
    let last = history.loss_per_epoch.last().copied().unwrap();
    assert!(
        last < first,
        "loss did not decrease: first={first}, last={last}"
    );
}
