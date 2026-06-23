//! Functional integration test: QmlTrainer end-to-end on Metal vs CPU.
//!
//! Complements `adjoint_metal_matches_cpu_12q_hea` (which pins the
//! gradient correctness at one set of params) by exercising a full
//! training loop on Metal and confirming it converges to comparable
//! loss / parameters as the CPU backend on the same trainer.
//!
//! Scope kept small (4q / 6p / 8 training points / 30 epochs) so the
//! test fits inside `cargo test`'s default time budget. The
//! corresponding benchmark in `benches/qml_train_bench.rs` runs the
//! same shape at larger scale.

#![cfg(all(target_os = "macos", feature = "metal"))]

use smallvec::smallvec;

use omega_backend_statevector::StatevectorBackend;
use omega_backend_statevector_metal::MetalStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::device::DeviceKind;
use omega_core::error::OmegaError;
use omega_core::params::ParameterBinding;
use omega_core::qml::{Encoding, OutputMode, QmlModel, QmlTrainer};

const NUM_QUBITS: u32 = 4;
const NUM_PARAMS: u32 = 6;
const NUM_TRAIN: usize = 8;
const NUM_EPOCHS: usize = 30;
const LR: f64 = 0.10;

fn build_model() -> QmlModel {
    // 4-qubit HEA: 4 Ry on q0..3 (params 0..3), CX ladder, 2 Rz on
    // q1..2 (params 4..5). Output: ⟨Z_0⟩ + ⟨Z_3⟩ — two scalar targets.
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
    // Deterministic synthetic regression: 4-dim sinusoidal input,
    // 2-dim sinusoidal target. Same shape as the bench but smaller.
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
fn qml_trainer_metal_matches_cpu_after_full_training() {
    let model = build_model();
    let (train_x, train_y) = build_training_set();

    // Two runs from the same starting params — one on CPU, one on
    // Metal.  Both use the identical trainer / dataset / epochs / lr,
    // so any divergence in the final state has to come from the
    // gradient computation differing between the backends.
    let mut params_cpu = fresh_params();
    let cpu = StatevectorBackend::new();
    let history_cpu = QmlTrainer::new(&model)
        .epochs(NUM_EPOCHS)
        .learning_rate(LR)
        .fit(&cpu, &mut params_cpu, &train_x, &train_y)
        .expect("CPU training");

    let mut params_metal = fresh_params();
    let metal = MetalStatevectorBackend::new().expect("Metal device");
    let history_metal = QmlTrainer::new(&model)
        .epochs(NUM_EPOCHS)
        .learning_rate(LR)
        .fit(&metal, &mut params_metal, &train_x, &train_y)
        .expect("Metal training");

    // Both must report the same number of epochs.
    assert_eq!(history_cpu.loss_per_epoch.len(), NUM_EPOCHS, "cpu epochs");
    assert_eq!(
        history_metal.loss_per_epoch.len(),
        NUM_EPOCHS,
        "metal epochs"
    );

    // Both must converge — final loss strictly smaller than initial.
    let loss0_cpu = history_cpu.loss_per_epoch.first().copied().unwrap();
    let loss_n_cpu = history_cpu.loss_per_epoch.last().copied().unwrap();
    let loss0_metal = history_metal.loss_per_epoch.first().copied().unwrap();
    let loss_n_metal = history_metal.loss_per_epoch.last().copied().unwrap();
    assert!(
        loss_n_cpu < loss0_cpu,
        "CPU did not converge: loss[0] = {loss0_cpu}, loss[N] = {loss_n_cpu}"
    );
    assert!(
        loss_n_metal < loss0_metal,
        "Metal did not converge: loss[0] = {loss0_metal}, loss[N] = {loss_n_metal}"
    );

    // Per-epoch loss curves must agree to within f32 tolerance. The
    // QmlTrainer uses adjoint AD which Metal computes in f32; CPU
    // does it in f64. 1e-3 absolute is generous and well above the
    // ~1e-7 single-step gradient agreement reported by
    // `adjoint_metal_matches_cpu_12q_hea`.
    for (epoch, (lc, lm)) in history_cpu
        .loss_per_epoch
        .iter()
        .zip(history_metal.loss_per_epoch.iter())
        .enumerate()
    {
        let diff = (lc - lm).abs();
        assert!(
            diff < 1e-3,
            "epoch {epoch}: cpu loss {lc:.6} vs metal loss {lm:.6} (|Δ| = {diff:.3e})",
        );
    }

    // Final params must agree to within f32 tolerance — accumulated
    // gradient error over 30 epochs at LR = 0.1 lands well within
    // 5e-3 absolute on a 6-param model.
    for s in 0..NUM_PARAMS {
        let cpu_v = params_cpu
            .resolve(&ParamExpr::Symbol(s))
            .expect("cpu param");
        let metal_v = params_metal
            .resolve(&ParamExpr::Symbol(s))
            .expect("metal param");
        let diff = (cpu_v - metal_v).abs();
        assert!(
            diff < 5e-3,
            "param theta_{s}: cpu {cpu_v:.6} vs metal {metal_v:.6} (|Δ| = {diff:.3e})",
        );
    }
}

#[test]
fn qml_trainer_device_metal_matches_metal_backend() {
    // .device(Metal) + MetalStatevectorBackend must pass the device
    // check and produce a strictly decreasing loss curve.
    let model = build_model();
    let (train_x, train_y) = build_training_set();
    let mut params = fresh_params();
    let metal = MetalStatevectorBackend::new().expect("Metal device");
    let history = QmlTrainer::new(&model)
        .device(DeviceKind::Metal)
        .epochs(NUM_EPOCHS)
        .learning_rate(LR)
        .fit(&metal, &mut params, &train_x, &train_y)
        .expect(".device(Metal) + Metal backend must succeed");
    let first = history.loss_per_epoch.first().copied().unwrap();
    let last = history.loss_per_epoch.last().copied().unwrap();
    assert!(last < first, "loss didn't drop: first={first}, last={last}");
}

#[test]
fn qml_trainer_device_cpu_rejects_metal_backend() {
    // .device(Cpu) + MetalStatevectorBackend must fail fast with
    // OmegaError::Unsupported naming both sides of the mismatch.
    let model = build_model();
    let (train_x, train_y) = build_training_set();
    let mut params = fresh_params();
    let metal = MetalStatevectorBackend::new().expect("Metal device");
    let err = QmlTrainer::new(&model)
        .device(DeviceKind::Cpu)
        .epochs(NUM_EPOCHS)
        .learning_rate(LR)
        .fit(&metal, &mut params, &train_x, &train_y)
        .expect_err("Cpu trainer must refuse Metal backend");
    assert!(
        matches!(err, OmegaError::Unsupported(_)),
        "expected Unsupported, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(msg.contains("cpu") && msg.contains("metal"), "msg: {msg}");
}
