//! Phase 4 — QML training-loop benchmark on CUDA.
//!
//! Mirrors `omega-backend-statevector-metal/benches/qml_train_bench.rs`
//! exactly so the n=18, 32-pt × 10-ep numbers compare apples-to-apples
//! across Metal (M-series mac) and CUDA (Linux+NVIDIA).
//!
//! Reproducer:
//!   cargo bench -p omega-backend-statevector-cuda \
//!       --bench qml_train_bench --features cuda
//!
//! Output: `benches/results/qml-train-x86_64-linux.json` (criterion
//! writes to `target/criterion/...` by default; the wrapper script
//! `benches/capture.sh` will land later to mirror Metal's
//! `benches/results/`).

#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use criterion::{criterion_group, criterion_main, Criterion};
use smallvec::smallvec;

use omega_backend_statevector::StatevectorBackend;
use omega_backend_statevector_cuda::CudaStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::Backend;
use omega_core::params::ParameterBinding;
use omega_core::qml::{Encoding, OutputMode, QmlModel, QmlTrainer};

// Same shape as the Metal bench so per-host numbers are comparable.
// 18-qubit / 16-param two-layer HEA, 32 points, 10 epochs.
//
// The Phase 4c headline shape (14q / 16p / 256pts / 100ep) can be
// dialled in at runtime via `OMEGA_QML_BENCH_SHAPE=phase4c`. Default
// stays at the round-15 baseline shape so cross-host comparisons
// against the Metal numbers in `qml-train-aarch64-darwin.json` work
// without re-flashing the constants.
const DEFAULT_NUM_QUBITS: u32 = 18;
const DEFAULT_NUM_PARAMS: u32 = 16;
const DEFAULT_NUM_TRAIN: usize = 32;
const DEFAULT_NUM_EPOCHS: usize = 10;

const PHASE4C_NUM_QUBITS: u32 = 14;
const PHASE4C_NUM_PARAMS: u32 = 16;
const PHASE4C_NUM_TRAIN: usize = 256;
const PHASE4C_NUM_EPOCHS: usize = 100;

#[derive(Clone, Copy)]
struct BenchShape {
    num_qubits: u32,
    num_params: u32,
    num_train: usize,
    num_epochs: usize,
    skip_cpu: bool,
}

fn shape_from_env() -> BenchShape {
    match std::env::var("OMEGA_QML_BENCH_SHAPE").as_deref() {
        Ok("phase4c") => BenchShape {
            num_qubits: PHASE4C_NUM_QUBITS,
            num_params: PHASE4C_NUM_PARAMS,
            num_train: PHASE4C_NUM_TRAIN,
            num_epochs: PHASE4C_NUM_EPOCHS,
            // CPU at 14q/16p/256pts/100ep extrapolates to ~150 s per
            // sample × 10 samples = 25 min. Skip the CPU arm for the
            // headline shape — the Phase 4c spec target is a CUDA-only
            // assertion, and the n=18 baseline already establishes the
            // CPU/CUDA ratio at ~110×.
            skip_cpu: true,
        },
        _ => BenchShape {
            num_qubits: DEFAULT_NUM_QUBITS,
            num_params: DEFAULT_NUM_PARAMS,
            num_train: DEFAULT_NUM_TRAIN,
            num_epochs: DEFAULT_NUM_EPOCHS,
            skip_cpu: false,
        },
    }
}

fn build_hea_model(shape: BenchShape) -> QmlModel {
    let mut ansatz = CircuitIR::new(shape.num_qubits, CircuitType::GateBased);
    for s in 0..shape.num_params {
        ansatz.symbols.insert(s, format!("theta_{s}"));
    }
    // Layer 1: 8 Ry on first 8 qubits — bench shape requires
    // num_qubits ≥ 8 (Phase 4c is 14, default is 18, both safe).
    for q in 0..8 {
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
    // Layer 2: 8 Rz starting from qubit 4. Phase 4c at n=14 has
    // qubits 0..13; default at n=18 has 0..17 — both fit q=4..11.
    for (i, q) in (4..12).enumerate() {
        ansatz.add_op(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(8 + i as u32)],
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

fn build_training_set(shape: BenchShape) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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

fn fresh_params(shape: BenchShape) -> ParameterBinding {
    let mut params = ParameterBinding::new();
    for s in 0..shape.num_params {
        let v = ((s as f64) * 0.317 - 0.84).sin() * 0.4;
        params.bind(s, v);
    }
    params
}

fn run_training(
    shape: BenchShape,
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

fn bench_qml_train(c: &mut Criterion) {
    let shape = shape_from_env();
    let (train_x, train_y) = build_training_set(shape);
    let bench_id = format!(
        "qml_train_{}q_{}p_{}pts_{}ep",
        shape.num_qubits, shape.num_params, shape.num_train, shape.num_epochs
    );
    let mut group = c.benchmark_group(bench_id);
    group.sample_size(10);

    if !shape.skip_cpu {
        let cpu = StatevectorBackend::new();
        group.bench_function("cpu", |b| {
            b.iter(|| run_training(shape, &cpu, &train_x, &train_y));
        });
    }

    let cuda = CudaStatevectorBackend::new().expect("CUDA device");
    group.bench_function("cuda", |b| {
        b.iter(|| run_training(shape, &cuda, &train_x, &train_y));
    });

    group.finish();
}

criterion_group!(benches, bench_qml_train);
criterion_main!(benches);
