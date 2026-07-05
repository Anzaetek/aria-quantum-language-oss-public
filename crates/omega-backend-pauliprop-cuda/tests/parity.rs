//! GPU-vs-CPU numeric parity for the Pauli-propagation branch accelerator.
//!
//! Only meaningful under `--features cuda` on a CUDA host. The GPU branch must
//! reproduce the CPU branch bit-for-bit up to floating-point rounding, on both
//! the exact and the `max_freq`-truncated engines.

#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_pauliprop_cuda::{cuda_branch, gpu_branch_count};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: params.iter().map(|p| ParamExpr::Concrete(*p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// Deep, entangling, non-Clifford circuit — a Trotter-like brickwall — so the
/// branch step produces many thousands of terms and the GPU path is exercised.
fn deep_circuit(nq: u32, layers: usize) -> CircuitIR {
    let mut ops = Vec::new();
    for q in 0..nq {
        ops.push(op(GateKind::H, &[q], &[]));
    }
    for l in 0..layers {
        for q in 0..nq - 1 {
            ops.push(op(GateKind::CX, &[q, q + 1], &[]));
        }
        for q in 0..nq {
            ops.push(op(GateKind::Rz, &[q], &[0.3 + 0.05 * l as f64]));
            ops.push(op(GateKind::Rx, &[q], &[0.2 + 0.05 * q as f64]));
        }
    }
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    c.ops = ops;
    c
}

fn obs() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (3, PauliOp::Z)])],
    }
}

/// Force the GPU path even on modest term counts so the test always exercises
/// the accelerator when a device is present.
fn force_gpu_min_zero() {
    // Safe: single-threaded test setup before any propagation runs.
    unsafe { std::env::set_var("PAULIPROP_GPU_MIN", "0") };
}

#[test]
fn gpu_branch_matches_cpu_exact() {
    force_gpu_min_zero();
    let c = deep_circuit(6, 5);
    let o = obs();
    let params = ParameterBinding::new();

    let exact = PauliPropBackend::new()
        .expectation(&c, &params, &o)
        .unwrap();

    let before = gpu_branch_count();
    let gpu = PauliPropBackend::new()
        .with_branch_hook(cuda_branch)
        .expectation(&c, &params, &o)
        .unwrap();

    if gpu_branch_count() == before {
        eprintln!("skipping: no CUDA device on this host (GPU branch never ran)");
        return;
    }
    assert!(
        (gpu - exact).abs() < 1e-9,
        "GPU branch expectation {gpu} != CPU {exact}"
    );
}

#[test]
fn gpu_branch_matches_cpu_with_max_freq() {
    force_gpu_min_zero();
    let c = deep_circuit(6, 5);
    let o = obs();
    let params = ParameterBinding::new();

    for max_freq in [2u32, 4, 8] {
        let cpu = PauliPropBackend::new()
            .max_freq(Some(max_freq))
            .expectation_with_budget(&c, &params, &o)
            .unwrap();
        let before = gpu_branch_count();
        let gpu = PauliPropBackend::new()
            .max_freq(Some(max_freq))
            .with_branch_hook(cuda_branch)
            .expectation_with_budget(&c, &params, &o)
            .unwrap();
        if gpu_branch_count() == before {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }
        // Value AND certified dropped-mass budget must match the CPU engine.
        assert!(
            (gpu.0 - cpu.0).abs() < 1e-9,
            "max_freq={max_freq}: GPU value {} != CPU {}",
            gpu.0,
            cpu.0
        );
        assert!(
            (gpu.1 - cpu.1).abs() < 1e-9,
            "max_freq={max_freq}: GPU dropped-mass {} != CPU {}",
            gpu.1,
            cpu.1
        );
    }
}
