//! GPU-vs-CPU numeric parity for the Pauli-propagation branch accelerator.
//!
//! Only meaningful under `--features cuda` on a CUDA host. The GPU branch must
//! reproduce the CPU branch bit-for-bit up to floating-point rounding, on both
//! the exact and the `max_freq`-truncated engines.

#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_pauliprop_cuda::{cuda_branch, gpu_branch_count, phase_times, reset_phase_times};
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

/// P2 baseline: what the GPU branch hook actually costs against the CPU branch
/// it replaces.
///
/// There is no published number for this anywhere — the CUDA arm has only ever
/// been gated on *agreement* (the two tests below), never on speed, so
/// "the GPU path is faster" has been an assumption. This measures it.
///
/// `#[ignore]`d because it is a timing test, matching the convention the
/// statevector crate already uses for `forward_graph_replay_perf_vs_naive`.
/// Run with `cargo test -p omega-backend-pauliprop-cuda --features cuda -- \
/// --ignored --nocapture`.
///
/// Read the result with the structure in mind: `branch_on_gpu` is a
/// `BranchHook`, so it takes `&mut PauliSum` — a HOST structure — and is called
/// once per rotation gate. Each call rebuilds the SoA from the term map,
/// performs ~15 device allocations, 7 uploads, a launch, a full
/// `synchronize()`, and 7 downloads. If the GPU margin is thin or negative at
/// moderate term counts, that fixed per-call overhead is why, and it is the
/// thing to attack — not the kernel.
#[test]
#[ignore = "perf timing — opt-in via `cargo test -- --ignored`"]
fn gpu_branch_vs_cpu_branch_timing() {
    force_gpu_min_zero();
    let params = ParameterBinding::new();
    let o = obs();

    // WARM-UP, discarded. Without it the first row carries one-time CUDA
    // context creation and NVRTC module load — ~0.5 s — and gets read as the
    // small-circuit result it is not. The earlier edition of this test reported
    // 0.11x at 6 qubits and that number was almost entirely context init.
    {
        let c = deep_circuit(6, 3);
        let _ = PauliPropBackend::new().expectation(&c, &params, &o);
        let _ = PauliPropBackend::new()
            .with_branch_hook(cuda_branch)
            .expectation(&c, &params, &o);
    }

    eprintln!(
        "{:>4}  {:>7}  {:>12}  {:>12}  {:>8}",
        "nq", "layers", "cpu", "gpu", "ratio"
    );
    // Widths chosen to cross the symplectic WORD boundary. `w = ceil(nq/64)`,
    // so everything at or below 64 qubits does its anticommute test in a
    // SINGLE u64 — there is almost no per-term work for the GPU to amortise
    // the transfers against. If a crossover exists it should appear past 64.
    for (nq, layers) in [
        (6u32, 5usize),
        (8, 6),
        (10, 6),
        (12, 6),
        (16, 5),
        (24, 4),
        (40, 3),
        (72, 3),
        (100, 3),
    ] {
        let c = deep_circuit(nq, layers);

        // Best of 3 per row. One sample is not a measurement, and the GPU arm
        // is the more variable of the two.
        let mut cpu = 0.0;
        let mut cpu_dt = std::time::Duration::MAX;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            cpu = PauliPropBackend::new()
                .expectation(&c, &params, &o)
                .unwrap();
            cpu_dt = cpu_dt.min(t0.elapsed());
        }

        let before = gpu_branch_count();
        reset_phase_times();
        let mut gpu = 0.0;
        let mut gpu_dt = std::time::Duration::MAX;
        for _ in 0..3 {
            let t1 = std::time::Instant::now();
            gpu = PauliPropBackend::new()
                .with_branch_hook(cuda_branch)
                .expectation(&c, &params, &o)
                .unwrap();
            gpu_dt = gpu_dt.min(t1.elapsed());
        }

        if gpu_branch_count() == before {
            eprintln!("skipping: no CUDA device on this host (GPU branch never ran)");
            return;
        }
        // A timing test that silently stopped agreeing would be worse than no
        // timing test, so keep the correctness assertion inside the loop.
        assert!(
            (gpu - cpu).abs() < 1e-9,
            "nq={nq}: GPU {gpu} != CPU {cpu} — a speed change altered the answer"
        );
        eprintln!(
            "{nq:>4}  {layers:>7}  {:>10.1?}  {:>10.1?}  {:>7.2}x  ({} gpu branches)",
            cpu_dt,
            gpu_dt,
            cpu_dt.as_secs_f64() / gpu_dt.as_secs_f64(),
            gpu_branch_count() - before
        );
        // Per-phase attribution — the whole point. Only populated under
        // PAULIPROP_GPU_PROFILE=1; without it these are all zero and the line
        // is skipped, so the timing above is not perturbed by instrumentation.
        let p = phase_times();
        if p.calls > 0 {
            let pct = |v: u64| 100.0 * v as f64 / p.total_ns().max(1) as f64;
            eprintln!(
                "        phases over {} calls: soa {:.0}% alloc {:.0}% upload {:.0}% \
                 launch+sync {:.0}% download {:.0}% merge {:.0}%  (sum {:.1?})",
                p.calls,
                pct(p.soa_ns),
                pct(p.alloc_ns),
                pct(p.upload_ns),
                pct(p.launch_sync_ns),
                pct(p.download_ns),
                pct(p.merge_ns),
                std::time::Duration::from_nanos(p.total_ns()),
            );
        }
    }
}

/// A LARGE truncating run followed by a SMALL one, on the same thread.
///
/// This is the case the device buffer pool introduces and that nothing else
/// covers. Buffers are grown and never shrunk, so after a big call every output
/// buffer is longer than the next small call needs, and its tail still holds the
/// previous call's data. If any consumer reads the whole buffer instead of the
/// live prefix, the stale tail is folded in — and for `odropped` that means
/// **inflating `dropped_mass`, the CERTIFIED truncation bound**, silently.
///
/// Verified to have teeth: changing the `odropped` view from `..num` to `..`
/// makes this test fail and only this one. The pre-existing parity tests pass
/// either way, because within a single propagation `num` never shrinks enough
/// to expose the tail — which is exactly why this needed writing.
#[test]
fn a_small_run_after_a_large_one_does_not_inherit_the_stale_tail() -> Result<(), String> {
    force_gpu_min_zero();
    let params = ParameterBinding::new();
    let o = obs();
    let max_freq = Some(3u32);

    // Big first: many terms with truncation on, purely to GROW the device buffer
    // pool. Its numeric result is intentionally discarded — at this aggressive a
    // cutoff the run legitimately REFUSES (discarded L1 mass far exceeds the
    // certified 1.0 ceiling), and a refusal still expands every branch on the GPU,
    // which is all this setup needs: oversized output buffers whose tails hold the
    // previous call's data. Unwrapping that expected refusal was the original bug.
    let big = deep_circuit(10, 6);
    let before = gpu_branch_count();
    let _ = PauliPropBackend::new()
        .max_freq(max_freq)
        .with_branch_hook(cuda_branch)
        .expectation_with_budget(&big, &params, &o);
    if gpu_branch_count() == before {
        eprintln!("SKIP (NOT A PASS): no CUDA device");
        return Ok(());
    }

    // Then small, on the same thread so it reuses those oversized buffers. This
    // run is comfortably within budget, so it must succeed; a refusal here is a
    // real failure, surfaced with its message rather than a bare unwrap.
    let small = deep_circuit(4, 2);
    let cpu = PauliPropBackend::new()
        .max_freq(max_freq)
        .expectation_with_budget(&small, &params, &o)
        .map_err(|e| format!("small CPU run should stay within budget: {e:?}"))?;
    let gpu = PauliPropBackend::new()
        .max_freq(max_freq)
        .with_branch_hook(cuda_branch)
        .expectation_with_budget(&small, &params, &o)
        .map_err(|e| format!("small GPU run should stay within budget: {e:?}"))?;

    assert!(
        (gpu.0 - cpu.0).abs() < 1e-9,
        "value after a larger run: GPU {} != CPU {}",
        gpu.0,
        cpu.0
    );
    // Print the numbers, not just PASS. `dropped_mass` is a CERTIFIED bound
    // that downstream code trusts, so the log should carry its value.
    eprintln!(
        "  dropped_mass after a larger run: GPU {:.15} vs CPU {:.15}  (|d| = {:.2e})",
        gpu.1,
        cpu.1,
        (gpu.1 - cpu.1).abs()
    );
    assert!(
        (gpu.1 - cpu.1).abs() < 1e-9,
        "DROPPED MASS after a larger run: GPU {} != CPU {}. A reused output \
         buffer's tail leaked into the certified error bound.",
        gpu.1,
        cpu.1
    );
    Ok(())
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
            .with_max_dropped_mass(Some(f64::INFINITY))
            .expectation_with_budget(&c, &params, &o)
            .unwrap();
        let before = gpu_branch_count();
        let gpu = PauliPropBackend::new()
            .max_freq(Some(max_freq))
            .with_max_dropped_mass(Some(f64::INFINITY))
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
