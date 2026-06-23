//! Phase 4 — QML training-loop benchmark.
//!
//! Two shapes:
//!   * `qml_train_18q_16p_32pts_10ep` — n=18 reference shape used
//!     to track session-over-session perf deltas during the 2026-05-08
//!     / 2026-05-09 perf push.
//!   * `qml_train_14q_16p_256pts_100ep` — the GPU_PLAN.md Phase 4c
//!     headline target shape. Smaller per-amp work (n=14) but 80×
//!     more train-pt-epochs than the n=18 shape; stresses dispatch /
//!     sync overhead more.
//!
//! Reproducer (full):
//!   cargo bench -p omega-backend-statevector-metal \
//!       --bench qml_train_bench --features metal
//!
//! The Phase 4c shape is heavy (~30+ min total wallclock at the
//! post-round-16 perf level). Filter to one shape with criterion's
//! standard `--bench` filter, e.g. `cargo bench -p ... -- 18q` to
//! capture only the n=18 numbers.
//!
//! Numbers stored at `benches/results/qml-train-aarch64-darwin.json`.

#![cfg(all(target_os = "macos", feature = "metal"))]

use criterion::{criterion_group, criterion_main, Criterion};
use smallvec::smallvec;

use omega_backend_statevector::StatevectorBackend;
use omega_backend_statevector_metal::MetalStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::Backend;
use omega_core::params::ParameterBinding;
use omega_core::qml::{Encoding, OutputMode, QmlModel, QmlTrainer};

/// HEA-shape parameters used by both bench functions. The Ry layer
/// covers `q0..(num_params/2)`; the Rz layer covers
/// `q4..(4 + num_params/2)`. Total `num_params` = 2× the per-layer
/// parameter count.
struct HeaShape {
    num_qubits: u32,
    num_params: u32,
    num_train: usize,
    num_epochs: usize,
}

const SHAPE_18Q: HeaShape = HeaShape {
    num_qubits: 18,
    num_params: 16,
    num_train: 32,
    num_epochs: 10,
};

const SHAPE_PHASE4C: HeaShape = HeaShape {
    num_qubits: 14,
    num_params: 16,
    num_train: 256,
    num_epochs: 100,
};

fn build_hea_model(shape: &HeaShape) -> QmlModel {
    let half = shape.num_params / 2;
    let mut ansatz = CircuitIR::new(shape.num_qubits, CircuitType::GateBased);
    for s in 0..shape.num_params {
        ansatz.symbols.insert(s, format!("theta_{s}"));
    }
    for q in 0..half {
        ansatz.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(q)],
            classical_bit: None,
            condition: None,
        });
    }
    for q in 0..shape.num_qubits - 1 {
        ansatz.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(q), Qubit(q + 1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    for (i, q) in (4..(4 + half)).enumerate() {
        ansatz.add_op(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(half + i as u32)],
            classical_bit: None,
            condition: None,
        });
    }

    QmlModel {
        num_qubits: shape.num_qubits,
        encoding: Encoding::Angle,
        ansatz,
        measurement_qubits: vec![0, shape.num_qubits - 1],
        output_mode: OutputMode::Expectation,
    }
}

fn build_training_set(shape: &HeaShape) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut train_x = Vec::with_capacity(shape.num_train);
    let mut train_y = Vec::with_capacity(shape.num_train);
    for k in 0..shape.num_train {
        let t = k as f64 / shape.num_train as f64;
        let mut x = Vec::with_capacity(shape.num_qubits as usize);
        for d in 0..shape.num_qubits {
            let v = (t * 6.0 + d as f64 * 0.31).sin() * 1.4;
            x.push(v);
        }
        let y0 = (t * 4.0 - 1.0).tanh();
        let y1 = (t * 5.0 + 0.7).cos() * 0.5;
        train_x.push(x);
        train_y.push(vec![y0, y1]);
    }
    (train_x, train_y)
}

fn fresh_params(shape: &HeaShape) -> ParameterBinding {
    let mut params = ParameterBinding::new();
    for s in 0..shape.num_params {
        let v = ((s as f64) * 0.317 - 0.84).sin() * 0.4;
        params.bind(s, v);
    }
    params
}

fn run_training(
    shape: &HeaShape,
    backend: &dyn Backend,
    train_x: &[Vec<f64>],
    train_y: &[Vec<f64>],
) {
    let model = build_hea_model(shape);
    let mut params = fresh_params(shape);
    let trainer = QmlTrainer::new(&model)
        .epochs(shape.num_epochs)
        .learning_rate(0.05);
    trainer
        .fit(backend, &mut params, train_x, train_y)
        .expect("training run");
}

fn bench_shape(c: &mut Criterion, shape: &HeaShape) {
    let (train_x, train_y) = build_training_set(shape);
    let bench_id = format!(
        "qml_train_{}q_{}p_{}pts_{}ep",
        shape.num_qubits, shape.num_params, shape.num_train, shape.num_epochs
    );
    let mut group = c.benchmark_group(bench_id);
    // Each iter runs the full training loop. Cap sample count at the
    // criterion minimum (10) so heavy shapes still finish in a
    // reasonable wallclock window.
    group.sample_size(10);

    let cpu = StatevectorBackend::new();
    group.bench_function("cpu", |b| {
        b.iter(|| run_training(shape, &cpu, &train_x, &train_y));
    });

    let metal = MetalStatevectorBackend::new().expect("Metal device");
    group.bench_function("metal", |b| {
        b.iter(|| run_training(shape, &metal, &train_x, &train_y));
    });

    group.finish();
}

fn bench_qml_train_18q(c: &mut Criterion) {
    bench_shape(c, &SHAPE_18Q);
}

fn bench_qml_train_phase4c(c: &mut Criterion) {
    bench_shape(c, &SHAPE_PHASE4C);
}

criterion_group!(benches, bench_qml_train_18q, bench_qml_train_phase4c);
criterion_main!(benches);
