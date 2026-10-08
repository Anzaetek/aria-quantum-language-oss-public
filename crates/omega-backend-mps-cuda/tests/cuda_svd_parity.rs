//! The CUDA bond-compression SVD against the CPU Jacobi kernel, on the shapes
//! `STATUS.md` §5.16 measured, and the dispatch counters that make a CPU
//! fallback visible.
//!
//! Gated on the `cuda` feature. Each test SKIPS, with the reason printed, only
//! when `CudaSvdContext::try_new()` fails — no device on this host. When a
//! device exists the test must FAIL, not pass vacuously, if the GPU path did
//! not run: each asserts `gpu_calls > 0` from the library's own counters (the
//! A10 form; a parity test that compared the CPU with a silent CPU fallback
//! would be green for the wrong reason).
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use std::cell::Cell;

use num_complex::Complex64;
use omega_backend_mps::svd::{truncated_svd_flat, SvdResultFlat};
use omega_backend_mps::MpsBackend;
use omega_backend_mps_cuda::{
    cuda_svd_dispatch, cuda_svd_flat, reset_cuda_svd_dispatch, CudaSvdContext,
};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

/// `true` when a device is present; otherwise prints why the test is skipped.
fn device_or_skip(test: &str) -> bool {
    match CudaSvdContext::try_new() {
        Ok(_) => true,
        Err(e) => {
            eprintln!("{test}: SKIPPED — no CUDA SVD context on this host: {e}");
            false
        }
    }
}

/// Identical to `mps_stage_profile`'s and `cuda_svd_share`'s circuit, so the
/// numbers here are the numbers in `STATUS.md` §5.16.
fn entangling_circuit(num_qubits: u32, depth: usize) -> CircuitIR {
    let mut circuit = CircuitIR::new(num_qubits, CircuitType::GateBased);
    let push = |c: &mut CircuitIR, gate: GateKind, qs: Vec<u32>, ps: Vec<f64>| {
        c.ops.push(GateOp {
            gate,
            qubits: qs.into_iter().map(Qubit).collect(),
            params: ps.into_iter().map(ParamExpr::Concrete).collect(),
            classical_bit: None,
            condition: None,
        });
    };
    for q in 0..num_qubits {
        push(&mut circuit, GateKind::H, vec![q], vec![]);
    }
    for d in 0..depth {
        let offset = d as u32 & 1;
        for q in (offset..num_qubits - 1).step_by(2) {
            push(&mut circuit, GateKind::CX, vec![q, q + 1], vec![]);
            push(&mut circuit, GateKind::Rz, vec![q + 1], vec![0.25]);
            push(&mut circuit, GateKind::CX, vec![q, q + 1], vec![]);
        }
        for q in 0..num_qubits {
            push(&mut circuit, GateKind::Rx, vec![q], vec![0.15]);
        }
    }
    circuit
}

fn z0() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0u32, PauliOp::Z)])],
    }
}

/// One run: `(⟨Z0⟩, discarded_weight)`. The refusal gate is lifted — this
/// compares certificates, it does not judge them.
fn run(chi: usize, depth: usize, svd: omega_backend_mps::SvdFlatFn) -> (f64, f64) {
    let backend = MpsBackend::new(chi)
        .with_max_discarded_weight(f64::INFINITY)
        .with_svd_fn(svd);
    let v = backend
        .expectation(
            &entangling_circuit(14, depth),
            &ParameterBinding::new(),
            &z0(),
        )
        .expect("expectation");
    (v, backend.last_run_stats().discarded_weight)
}

/// CPU run, then the GPU run through `cuda_svd_flat` with the counters reset
/// around it. Returns `(cpu, gpu)` pairs of `(⟨Z0⟩, discarded_weight)`.
fn cpu_and_gpu(chi: usize, depth: usize) -> ((f64, f64), (f64, f64)) {
    let cpu = run(chi, depth, truncated_svd_flat);
    reset_cuda_svd_dispatch();
    let gpu = run(chi, depth, cuda_svd_flat);
    let d = cuda_svd_dispatch();
    eprintln!(
        "14q depth={depth} chi={chi}: <Z0> cpu={:.6e} gpu={:.6e} |diff|={:.3e}; \
         discarded_weight cpu={:.6e} gpu={:.6e}; dispatch {d:?}",
        cpu.0,
        gpu.0,
        (cpu.0 - gpu.0).abs(),
        cpu.1,
        gpu.1
    );
    assert!(
        d.gpu_calls > 0,
        "a CUDA device is present but no SVD ran on it ({d:?}) — this would compare \
         the CPU with itself"
    );
    (cpu, gpu)
}

/// The truncating shape. Depth 12 is the shape named for this defect; depth 24
/// is the one whose `⟨Z⟩` STATUS quotes (−1.9921e-9 through `Zgesvda` against
/// −4.1442e-15 on the CPU). Depth 12's `⟨Z0⟩` is ~1e-15 on both paths and
/// cannot tell them apart; its certificate can (8.2e-2 relative under gesvda).
///
/// **Certificate tolerance is 1e-11 relative, not 1e-12**, measured rather
/// than chosen: at depth 12 the exact `Zgesvd` certificate agrees with the CPU
/// to 1.522e-12 relative (6.9e-14 at depth 24). The two kernels are different
/// exact algorithms — one-sided Jacobi resolves small σ to high RELATIVE
/// accuracy, bidiagonal QR to ~eps·σ_max ABSOLUTE — and the dropped tail is
/// made of the smallest kept-or-dropped σ, so the last digit of Σσ² is where
/// they differ. 1e-11 still sits nine orders below what the approximate solver
/// gave, and ten below a halved tail.
#[test]
fn gpu_svd_matches_cpu_under_truncation() {
    if !device_or_skip("gpu_svd_matches_cpu_under_truncation") {
        return;
    }
    for depth in [12usize, 24] {
        let ((vc, dc), (vg, dg)) = cpu_and_gpu(32, depth);
        assert!(
            dc > 1e-20,
            "depth {depth}: chi=32 is meant to truncate (a rounding tail reads ~1e-28), \
             CPU certificate {dc:e}"
        );
        assert!(
            (vc - vg).abs() <= 1e-12,
            "depth {depth}: <Z0> cpu {vc:e} vs gpu {vg:e} (|diff| {:e})",
            (vc - vg).abs()
        );
        let rel = (dc - dg).abs() / dc;
        assert!(
            rel <= 1e-11,
            "depth {depth}: discarded_weight cpu {dc:e} vs gpu {dg:e} (relative {rel:e})"
        );
    }
}

/// χ = 128 is exact at 14 qubits (the largest middle-cut bond is 2^7), so both
/// certificates must read exact — below the 1e-12 floor the CLI already uses
/// for "real truncation" — and the values agree to 1e-12. (A relative
/// comparison of two ~1e-28 rounding tails would measure noise.)
#[test]
fn gpu_svd_matches_cpu_when_exact() {
    if !device_or_skip("gpu_svd_matches_cpu_when_exact") {
        return;
    }
    // Depth 12 only: the exact case needs no second shape, and depth 24 at
    // chi=128 costs about a minute of CPU Jacobi in a debug build.
    {
        let depth = 12usize;
        let ((vc, dc), (vg, dg)) = cpu_and_gpu(128, depth);
        assert!(
            (vc - vg).abs() <= 1e-12,
            "depth {depth}: <Z0> cpu {vc:e} vs gpu {vg:e} (|diff| {:e})",
            (vc - vg).abs()
        );
        assert!(
            dc <= 1e-12 && dg <= 1e-12,
            "depth {depth}: exact run must certify exact: cpu {dc:e}, gpu {dg:e}"
        );
    }
}

thread_local! {
    static SPLITS: Cell<u64> = const { Cell::new(0) };
}

/// The CPU kernel, counting how many two-site splits the chain performs.
fn counting_cpu_svd(
    a: &[Complex64],
    m: usize,
    n: usize,
    lda: usize,
    max_rank: usize,
    threshold: f64,
) -> SvdResultFlat {
    SPLITS.with(|c| c.set(c.get() + 1));
    truncated_svd_flat(a, m, n, lda, max_rank, threshold)
}

/// Every SVD `cuda_svd_flat` performs is counted exactly once — on the GPU or
/// as a CPU fallback — so the two counters sum to the number of splits, and
/// the GPU one is non-zero on a host with a device.
#[test]
fn every_dispatch_is_counted() {
    if !device_or_skip("every_dispatch_is_counted") {
        return;
    }
    let (chi, depth) = (32, 12);
    SPLITS.with(|c| c.set(0));
    run(chi, depth, counting_cpu_svd);
    let splits = SPLITS.with(|c| c.get());
    assert!(splits > 0, "the chain performed no two-site splits");

    reset_cuda_svd_dispatch();
    run(chi, depth, cuda_svd_flat);
    let d = cuda_svd_dispatch();
    eprintln!("14q depth={depth} chi={chi}: {splits} splits; dispatch {d:?}");
    assert_eq!(
        d.gpu_calls + d.cpu_fallbacks,
        splits,
        "every split must be counted exactly once: {d:?} vs {splits} splits"
    );
    assert!(
        d.gpu_calls > 0,
        "a device is present but nothing ran on it: {d:?}"
    );
    assert_eq!(
        d.last_fallback.is_some(),
        d.cpu_fallbacks > 0,
        "a fallback must carry its reason, and only a fallback: {d:?}"
    );

    // And reset really zeroes it.
    reset_cuda_svd_dispatch();
    assert_eq!(cuda_svd_dispatch(), Default::default());
}
