// SPDX-License-Identifier: Apache-2.0
//! Step 0 of `PLAN-CUDA-STREAMS.md`: does releasing the GIL actually buy
//! anything on CUDA, or does the single shared stream serialise it away?
//!
//! `147fecd` made `CudaStatevectorBackend` `Send + Sync`, which removes the
//! LANGUAGE-level obstacle to calling it from several host threads. It does not
//! touch the HARDWARE-level one: every call goes through one
//! `Arc<CudaStream>`, so two threads submit into the same queue and the driver
//! serialises them. Results stay correct — each launch owns its own
//! `StateBuffer` — but the work may not overlap.
//!
//! This measures whether that matters before anyone builds per-thread streams,
//! which is a real piece of work (the graph cache is instantiated against a
//! stream, so it needs per-stream re-instantiation or a per-thread cache that
//! multiplies device memory, and the `unsafe impl`'s Mutex argument would have
//! to be re-derived rather than ported).
//!
//! # What to expect, and why a negative result is a complete answer
//!
//! The GB10 has 48 SMs. A statevector kernel at any interesting width already
//! saturates them — a 24-qubit launch is 16.7M amplitudes. Stream concurrency
//! pays when kernels are too SMALL to fill the device, which is the opposite of
//! this workload. So the honest prediction is: a win at small widths, nothing
//! at large ones, and the interesting answer somewhere in between.
//!
//! `#[ignore]`d because it is a timing test, matching the convention used by
//! `gpu_branch_vs_cpu_branch_timing` and `forward_graph_replay_perf_vs_naive`.
//! Run with:
//! `cargo test --release -p omega-backend-statevector-cuda --features cuda \
//!   -- --ignored --nocapture stream_concurrency`
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use omega_backend_statevector_cuda::CudaStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;
use std::time::{Duration, Instant};

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: params.iter().map(|p| ParamExpr::Concrete(*p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// Layered, 2q-heavy — the shape that actually gets run, not a microbenchmark.
fn hea(nq: u32, layers: usize) -> CircuitIR {
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    for l in 0..layers {
        for q in 0..nq {
            c.ops.push(op(
                GateKind::Ry,
                &[q],
                &[0.2 + 0.05 * (q + l as u32) as f64],
            ));
            c.ops.push(op(GateKind::Rz, &[q], &[0.3 + 0.04 * q as f64]));
        }
        for q in 0..nq.saturating_sub(1) {
            c.ops.push(op(GateKind::CX, &[q, q + 1], &[]));
        }
    }
    c
}

fn obs() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
    }
}

#[test]
#[ignore = "perf timing — opt-in via `cargo test -- --ignored`"]
fn stream_concurrency_two_threads_vs_one() {
    let Ok(backend) = CudaStatevectorBackend::new() else {
        eprintln!("SKIP (NOT A PASS): no CUDA device");
        return;
    };
    let pb = ParameterBinding::new();
    let o = obs();

    // Warm-up: NVRTC compiles 26 kernels on first construction (~0.58 s). Left
    // in, the first row measures the compiler, not the GPU.
    let _ = backend.expectation(&hea(8, 2), &pb, &o);

    eprintln!(
        "{:>4} {:>7} {:>10} {:>12} {:>12} {:>9}",
        "nq", "layers", "calls", "1 thread", "2 threads", "speedup"
    );

    const CALLS: usize = 8;
    for (nq, layers) in [(10u32, 4usize), (14, 4), (18, 4), (20, 3), (22, 3), (24, 2)] {
        let c = hea(nq, layers);

        // Serial reference: CALLS runs on one thread.
        let mut serial = Duration::MAX;
        let mut want = 0.0;
        for _ in 0..3 {
            let t = Instant::now();
            for _ in 0..CALLS {
                want = backend.expectation(&c, &pb, &o).expect("serial");
            }
            serial = serial.min(t.elapsed());
        }

        // Same total work, split across two threads sharing ONE backend.
        let mut par = Duration::MAX;
        let mut got = 0.0;
        for _ in 0..3 {
            let t = Instant::now();
            let vals = std::thread::scope(|s| {
                let hs: Vec<_> = (0..2)
                    .map(|_| {
                        let (b, c, o, pb) = (&backend, &c, &o, &pb);
                        s.spawn(move || {
                            (0..CALLS / 2)
                                .map(|_| b.expectation(c, pb, o).expect("threaded"))
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                hs.into_iter()
                    .flat_map(|h| h.join().expect("thread"))
                    .collect::<Vec<_>>()
            });
            par = par.min(t.elapsed());
            got = vals[0];
        }

        // Correctness is not optional just because this is a timing test.
        assert!(
            (got - want).abs() < 1e-5,
            "nq={nq}: threaded {got} != serial {want} — sharing the backend \
             changed the answer, which matters far more than the timing"
        );

        eprintln!(
            "{nq:>4} {layers:>7} {CALLS:>10} {:>10.1?} {:>10.1?} {:>8.2}x",
            serial,
            par,
            serial.as_secs_f64() / par.as_secs_f64()
        );
    }

    eprintln!(
        "\n  A speedup near 1.00x means the single shared stream serialises the \
         two threads,\n  i.e. releasing the GIL buys correctness but not \
         parallelism. See PLAN-CUDA-STREAMS.md."
    );
}
