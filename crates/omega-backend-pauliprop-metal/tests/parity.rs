//! GPU-vs-CPU numeric parity for the Metal Pauli-propagation branch accelerator.
//!
//! Only meaningful under `--features metal` on macOS. The GPU branch must
//! reproduce the CPU branch up to floating-point rounding (the coefficient math
//! is done on the CPU in f64 either way, so this is genuinely exact — only the
//! HashMap-order-dependent summation of the final expectation differs), on both
//! the exact and the `max_freq`-truncated engines.

#![cfg(all(target_os = "macos", feature = "metal"))]

use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_pauliprop_metal::{gpu_branch_count, metal_branch};
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

/// Wide, shallow circuit on `nq` qubits: H + a CX ladder (Clifford) plus a few
/// non-Clifford rotations placed to straddle the 32- and 64-qubit boundaries.
/// Cheap for pauliprop (few splits → bounded term count) but the packed
/// symplectic strings span multiple u64 words / uint lanes.
fn wide_circuit(nq: u32) -> CircuitIR {
    let mut ops = Vec::new();
    for q in 0..nq {
        ops.push(op(GateKind::H, &[q], &[]));
    }
    for q in 0..nq - 1 {
        ops.push(op(GateKind::CX, &[q, q + 1], &[]));
    }
    // A handful of non-Clifford rotations, deliberately on qubits in the low
    // lane (<32), the middle lane (32–63), and the high word (≥64).
    for &q in &[1u32, 20, 33, 50, 66] {
        if q < nq {
            ops.push(op(GateKind::Rz, &[q], &[0.35]));
            ops.push(op(GateKind::Rx, &[q], &[0.25]));
        }
    }
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    c.ops = ops;
    c
}

#[test]
fn gpu_branch_matches_cpu_wide_multiword() {
    // 70 qubits → 2 u64 words → 4 uint lanes per symplectic string. This is the
    // path the 6-qubit tests (and the CUDA suite) never hit: the Metal shader
    // folds each 64-bit accumulator across two uint lanes, so a string that
    // populates the high lane (qubits 32–69) must still parity-fold correctly.
    force_gpu_min_zero();
    let c = wide_circuit(70);
    // Observable straddling all four lanes: qubits 0, 40 (lane 1), 69 (word 1).
    let o = Observable {
        terms: vec![(
            1.0,
            vec![(0, PauliOp::Z), (40, PauliOp::Z), (69, PauliOp::Z)],
        )],
    };
    let params = ParameterBinding::new();

    let cpu = PauliPropBackend::new()
        .expectation(&c, &params, &o)
        .unwrap();
    let before = gpu_branch_count();
    let gpu = PauliPropBackend::new()
        .with_branch_hook(metal_branch)
        .expectation(&c, &params, &o)
        .unwrap();
    if gpu_branch_count() == before {
        eprintln!("skipping: no Metal device on this host");
        return;
    }
    assert!(
        (gpu - cpu).abs() < 1e-9,
        "wide multiword: GPU branch {gpu} != CPU {cpu}"
    );
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
        .with_branch_hook(metal_branch)
        .expectation(&c, &params, &o)
        .unwrap();

    if gpu_branch_count() == before {
        eprintln!("skipping: no Metal device on this host (GPU branch never ran)");
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
            .with_branch_hook(metal_branch)
            .expectation_with_budget(&c, &params, &o)
            .unwrap();
        if gpu_branch_count() == before {
            eprintln!("skipping: no Metal device on this host");
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
