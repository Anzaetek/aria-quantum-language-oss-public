//! Integration coverage for `QmlTrainer::device(DeviceKind::Cuda)`
//! against the real `CudaStatevectorBackend`. Mirror of the unit
//! tests in `omega-core/src/qml.rs` but against the actual backend
//! to confirm:
//! - `CudaStatevectorBackend::device()` returns `Cuda`.
//! - `QmlTrainer.device(DeviceKind::Cuda).fit(&cuda_backend, ..)`
//!   passes validation (no `InvalidCircuit("does not match")` error).
//! - `QmlTrainer.device(DeviceKind::Cpu).fit(&cuda_backend, ..)`
//!   surfaces the mismatch up front.
//!
//! Runs only when both `--features cuda` is on and a CUDA driver is
//! present at runtime (mirrors the gating in the lib tests).

#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use omega_backend_statevector_cuda::CudaStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::device::DeviceKind;
use omega_core::error::OmegaError;
use omega_core::executor::Backend;
use omega_core::params::ParameterBinding;
use omega_core::qml::{Encoding, OutputMode, QmlModel, QmlTrainer};
use smallvec::smallvec;

fn trivial_model() -> QmlModel {
    let mut ansatz = CircuitIR::new(2, CircuitType::GateBased);
    ansatz.symbols.insert(0, "theta".to_string());
    ansatz.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    QmlModel {
        num_qubits: 2,
        encoding: Encoding::Angle,
        ansatz,
        measurement_qubits: vec![0, 1],
        output_mode: OutputMode::Expectation,
    }
}

fn cuda_backend_or_skip() -> Option<CudaStatevectorBackend> {
    match CudaStatevectorBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: no CUDA backend on this host: {e:?}");
            None
        }
    }
}

#[test]
fn cuda_backend_reports_cuda_device_kind() {
    let Some(backend) = cuda_backend_or_skip() else {
        return;
    };
    assert_eq!(backend.device(), DeviceKind::Cuda);
}

#[test]
fn trainer_device_cuda_matches_cuda_backend() {
    let Some(backend) = cuda_backend_or_skip() else {
        return;
    };
    let model = trivial_model();
    let trainer = QmlTrainer::new(&model)
        .device(DeviceKind::Cuda)
        .epochs(1)
        .learning_rate(0.05);
    let mut params = ParameterBinding::new();
    params.bind(0, 0.1);
    let result = trainer.fit(&backend, &mut params, &[vec![0.1, 0.2]], &[vec![0.0, 0.0]]);
    // Whatever happens downstream (success or unrelated error), it
    // must NOT be the device-mismatch InvalidCircuit message.
    if let Err(OmegaError::InvalidCircuit(msg)) = &result {
        assert!(
            !msg.contains("does not match backend device"),
            "validation must pass with matching device; got: {msg}"
        );
    }
}

#[test]
fn trainer_device_cpu_with_cuda_backend_errors() {
    let Some(backend) = cuda_backend_or_skip() else {
        return;
    };
    let model = trivial_model();
    let trainer = QmlTrainer::new(&model)
        .device(DeviceKind::Cpu)
        .epochs(1)
        .learning_rate(0.05);
    let mut params = ParameterBinding::new();
    params.bind(0, 0.1);
    let err = trainer
        .fit(&backend, &mut params, &[vec![0.1, 0.2]], &[vec![0.0, 0.0]])
        .expect_err("CPU hint with CUDA backend must error");
    match err {
        OmegaError::Unsupported(msg) => {
            assert!(msg.contains("cuda"), "expected cuda in msg: {msg}");
            assert!(msg.contains("cpu"), "expected cpu in msg: {msg}");
        }
        other => panic!("expected Unsupported(device-mismatch), got {other:?}"),
    }
}
