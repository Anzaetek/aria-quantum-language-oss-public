//! Cross-backend statevector parity harness.
//!
//! Each GPU backend (Metal / OpenCL / CUDA) gets a parity test against
//! the CPU `StatevectorBackend` on a small but non-trivial circuit:
//! H + CX entangler + a ring of Rz / Ry rotations. Tests are gated by
//! the corresponding `--features` flag (defaults off so `cargo test
//! --workspace` on a vanilla host has no GPU dependency) and within
//! each test the backend constructor's `Unavailable` / `Runtime`
//! variants soft-skip the assertion — same convention the OpenCL
//! `execute_smoke` test uses. Hosts with the feature compiled in but
//! no GPU present still pass without false negatives.
//!
//! Acceptance criterion per backend: L2 distance to the CPU
//! statevector ≤ 1e-6. Tighter than verify-qiskit's 1e-12 since GPU
//! arithmetic on Metal is f32-only on Apple Silicon and we don't want
//! a per-backend ε ratchet here — the harness's job is "qualitatively
//! agree", not "byte-match".
//!
//! New backends slot in by adding a `#[cfg(feature = "$NEW")]` test
//! that follows the same construct-or-skip pattern below.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::CircuitIR;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;
use omega_parser::lower_to_ir;

/// 4-qubit entangling circuit with a 6-gate Rz/Ry "ansatz" tail.
/// Small enough to run in milliseconds on every backend; rich enough
/// that an f32 vs f64 divergence or a wrong-endian permutation would
/// blow past 1e-6 on at least one amplitude.
const PARITY_QASM: &str = r#"
OPENQASM 2.0;
qreg q[4];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
ry(0.31) q[0];
rz(0.42) q[1];
ry(0.55) q[2];
cx q[0], q[2];
rz(-0.21) q[0];
ry(0.77) q[3];
"#;

fn parity_circuit() -> CircuitIR {
    lower_to_ir(PARITY_QASM).expect("parity_circuit must parse")
}

fn cpu_statevector() -> Vec<num_complex::Complex64> {
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let backend = StatevectorBackend::new();
    match backend
        .execute(&parity_circuit(), &ParameterBinding::default(), &cfg)
        .unwrap()
    {
        ExecResult::Statevector(sv) => sv,
        _ => unreachable!("shots:None must yield Statevector"),
    }
}

/// L2 distance with conjugate of one side — kept as a single helper
/// so every backend's assertion reads the same way.
#[cfg(any(feature = "metal", feature = "opencl", feature = "cuda"))]
fn l2_distance(a: &[num_complex::Complex64], b: &[num_complex::Complex64]) -> f64 {
    assert_eq!(a.len(), b.len(), "statevector dimensions must match");
    let mut acc = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d.norm_sqr();
    }
    acc.sqrt()
}

#[cfg(any(feature = "metal", feature = "opencl", feature = "cuda"))]
const PARITY_TOL: f64 = 1e-6;

#[cfg(feature = "metal")]
#[test]
fn metal_matches_cpu_within_tol() {
    use omega_backend_statevector_metal::MetalStatevectorBackend;
    let backend = match MetalStatevectorBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Metal unavailable — skipping parity test ({e})");
            return;
        }
    };
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let gpu_sv = match backend
        .execute(&parity_circuit(), &ParameterBinding::default(), &cfg)
        .unwrap()
    {
        ExecResult::Statevector(sv) => sv,
        _ => unreachable!(),
    };
    let cpu_sv = cpu_statevector();
    let d = l2_distance(&cpu_sv, &gpu_sv);
    assert!(
        d <= PARITY_TOL,
        "Metal/CPU L2 distance {d} > tol {PARITY_TOL}"
    );
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_matches_cpu_within_tol() {
    use omega_backend_statevector_opencl::{OpenClError, OpenClStatevectorBackend};
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(OpenClError::Unavailable(_)) | Err(OpenClError::Runtime(_)) => {
            eprintln!("OpenCL unavailable — skipping parity test");
            return;
        }
        Err(e) => panic!("unexpected OpenCL init error: {e}"),
    };
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let gpu_sv = match backend
        .execute(&parity_circuit(), &ParameterBinding::default(), &cfg)
        .unwrap()
    {
        ExecResult::Statevector(sv) => sv,
        _ => unreachable!(),
    };
    let cpu_sv = cpu_statevector();
    let d = l2_distance(&cpu_sv, &gpu_sv);
    assert!(
        d <= PARITY_TOL,
        "OpenCL/CPU L2 distance {d} > tol {PARITY_TOL}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_cpu_within_tol() {
    use omega_backend_statevector_cuda::CudaStatevectorBackend;
    let backend = match CudaStatevectorBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("CUDA unavailable — skipping parity test ({e})");
            return;
        }
    };
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let gpu_sv = match backend
        .execute(&parity_circuit(), &ParameterBinding::default(), &cfg)
        .unwrap()
    {
        ExecResult::Statevector(sv) => sv,
        _ => unreachable!(),
    };
    let cpu_sv = cpu_statevector();
    let d = l2_distance(&cpu_sv, &gpu_sv);
    assert!(
        d <= PARITY_TOL,
        "CUDA/CPU L2 distance {d} > tol {PARITY_TOL}"
    );
}

#[test]
fn cpu_baseline_is_unit_norm() {
    // Sanity: the CPU reference state must itself be a valid statevector
    // before any GPU comparison is meaningful. Catches a broken
    // parity_circuit() before the cfg-gated tests obscure the failure.
    let sv = cpu_statevector();
    let norm_sq: f64 = sv.iter().map(|a| a.norm_sqr()).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-12,
        "CPU sv norm² = {norm_sq}, expected ~1"
    );
}
