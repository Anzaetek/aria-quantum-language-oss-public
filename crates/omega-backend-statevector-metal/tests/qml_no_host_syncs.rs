//! Regression test: the QML gradient hot path must not pull the
//! full statevector to host.
//!
//! Counts `MetalState::read_state` calls before and after a real
//! QML training epoch on the Metal backend; the delta must be zero.
//!
//! Why this works: `read_state` is the *only* function that copies
//! the whole `2·dim·f32` device buffer back to host. Every other
//! Metal kernel either stays device-resident
//! (`apply_diagonal_pauli_sum`, the `apply_*` family) or pulls a
//! scalar reduction's partials (`inner_product`,
//! `pauli_expectation` — `num_tgs × 2 × f32` bytes, sub-µs at any
//! qubit count we support). So a zero-delta on this counter
//! certifies the hot path is free of bulk host syncs without
//! needing Instruments / MTLCommandBuffer tracing.
//!
//! Static-audit context — the only `read_state` site inside the
//! adjoint backward sweep (`adjoint.rs:146`) is the `else` branch
//! of `diagonal_pauli_terms`. The QML trainer's gradient observable
//! is `Σ 2·r_i · Z_{q_i}` — pure Z + identity — so the classifier
//! always returns `Some` and the host-pull branch is never taken.
//! This test pins that property at runtime against a real training
//! epoch.

#![cfg(all(target_os = "macos", feature = "metal"))]

use smallvec::smallvec;

use omega_backend_statevector_metal::{MetalState, MetalStatevectorBackend};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::params::ParameterBinding;
use omega_core::qml::{Encoding, OutputMode, QmlModel, QmlTrainer};

const NUM_QUBITS: u32 = 4;
const NUM_PARAMS: u32 = 6;
const NUM_TRAIN: usize = 4;
const NUM_EPOCHS: usize = 2;
const LR: f64 = 0.10;

fn build_model() -> QmlModel {
    // Minimal HEA — same shape as `qml_train_metal.rs` but smaller
    // (4q / 6p / 4 train pts × 2 epochs) so the test stays under a
    // second. Gate kinds (Ry, CX, Rz) match the QML hot path; only
    // the counts shrink.
    let mut ansatz = CircuitIR::new(NUM_QUBITS, CircuitType::GateBased);
    for s in 0..NUM_PARAMS {
        ansatz.symbols.insert(s, format!("theta_{s}"));
    }
    for q in 0..4 {
        ansatz.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(q)],
            classical_bit: None,
            condition: None,
        });
    }
    for q in 0..NUM_QUBITS - 1 {
        ansatz.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(q), Qubit(q + 1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    for (i, q) in (1..3u32).enumerate() {
        ansatz.add_op(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(4 + i as u32)],
            classical_bit: None,
            condition: None,
        });
    }
    QmlModel {
        num_qubits: NUM_QUBITS,
        encoding: Encoding::Angle,
        ansatz,
        measurement_qubits: vec![0, NUM_QUBITS - 1],
        output_mode: OutputMode::Expectation,
    }
}

fn build_training_set() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut train_x = Vec::with_capacity(NUM_TRAIN);
    let mut train_y = Vec::with_capacity(NUM_TRAIN);
    for k in 0..NUM_TRAIN {
        let t = k as f64 / NUM_TRAIN as f64;
        let x: Vec<f64> = (0..NUM_QUBITS)
            .map(|d| (t * 5.0 + d as f64 * 0.41).sin() * 1.2)
            .collect();
        let y0 = (t * 3.0 - 1.0).tanh();
        let y1 = (t * 4.0 + 0.5).cos() * 0.4;
        train_x.push(x);
        train_y.push(vec![y0, y1]);
    }
    (train_x, train_y)
}

fn fresh_params() -> ParameterBinding {
    let mut params = ParameterBinding::new();
    for s in 0..NUM_PARAMS {
        params.bind(s, ((s as f64) * 0.213 - 0.31).sin() * 0.3);
    }
    params
}

#[test]
fn qml_gradient_loop_does_not_pull_full_statevector_to_host() {
    let model = build_model();
    let (train_x, train_y) = build_training_set();
    let mut params = fresh_params();
    let backend = MetalStatevectorBackend::new().expect("Metal device");

    // Counter is process-global; treat it as a delta so any
    // earlier setup-side `read_state` calls don't poison the
    // assertion.
    let before = MetalState::read_state_call_count();
    let history = QmlTrainer::new(&model)
        .epochs(NUM_EPOCHS)
        .learning_rate(LR)
        .fit(&backend, &mut params, &train_x, &train_y)
        .expect("QML training on Metal");
    let after = MetalState::read_state_call_count();

    // Sanity-check that training actually ran.
    assert_eq!(history.loss_per_epoch.len(), NUM_EPOCHS, "epochs");

    // The key assertion: zero `read_state` calls during training.
    let delta = after - before;
    assert_eq!(
        delta, 0,
        "QML gradient hot path pulled the full statevector to host {delta} time(s). \
         Likely cause: a backward-sweep observable that doesn't classify as diagonal-Z \
         (so adjoint_gradient takes the `apply_observable_host` fallback at \
         adjoint.rs:146). The QML trainer's gradient observable is `Σ 2·r·Z` and \
         must always classify; investigate `diagonal_pauli_terms` and the trainer's \
         gradient-observable factory if this trips."
    );
}
