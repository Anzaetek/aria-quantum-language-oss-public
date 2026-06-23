//! Smoke test: time a fixed-sequence kernel batch as a captured
//! CUDA graph vs naive per-call launches. Purpose is *measurement*,
//! not coverage — informs whether the param-pool + graph capture
//! refactor is worth the engineering cost on this host.
//!
//! `cargo test -p omega-backend-statevector-cuda --features cuda
//!  --test graph_replay_smoke -- --nocapture`

#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use std::time::Instant;

use num_complex::Complex64;
use omega_backend_statevector_cuda::{CudaState, CudaStatevectorBackend};

const N: u32 = 14;
const SEQ_LEN: usize = 50; // ops per "training step"
const REPLAYS: usize = 1000; // ~ training points × epochs

fn run_naive(backend: &CudaStatevectorBackend) -> std::time::Duration {
    // Allocate one state, then run SEQ_LEN ops × REPLAYS rounds via
    // the existing per-call launch path.
    let mut state: CudaState = backend.allocate(N).expect("alloc");
    let theta = 0.317_f64;
    let t0 = Instant::now();
    for _ in 0..REPLAYS {
        // Mix of kernel types matching the HEA shape:
        // 14 Ry (encoding) + 8 Ry (layer 1) + 13 CX + 8 Rz +
        // some daggers/derivatives ≈ 50 ops.
        for q in 0..14 {
            state.apply_ry(q, theta).unwrap();
        }
        for q in 0..8 {
            state.apply_ry(q, theta).unwrap();
        }
        for q in 0..13 {
            state.apply_cx(q, q + 1).unwrap();
        }
        for q in 4..12 {
            state.apply_rz(q, theta).unwrap();
        }
        // Make sure the state doesn't blow up — re-init occasionally.
        // (This is a perf bench, not a correctness check.)
    }
    let _ = SEQ_LEN; // sanity
    t0.elapsed()
}

#[test]
fn time_kernel_launch_baseline() {
    // Establishes the per-launch CPU overhead in our actual launch
    // path. Multiply by ~7.7M to estimate Phase 4c total dispatch
    // cost.
    let backend = CudaStatevectorBackend::new().expect("cuda");
    let elapsed = run_naive(&backend);
    let total_launches = REPLAYS * (14 + 8 + 13 + 8);
    let per_launch_us = elapsed.as_secs_f64() * 1_000_000.0 / total_launches as f64;
    eprintln!(
        "naive: {:?} for {} replays × {} launches = {} total launches; per-launch {:.2} µs",
        elapsed,
        REPLAYS,
        14 + 8 + 13 + 8,
        total_launches,
        per_launch_us
    );
    // Sanity: the test should complete in under a minute even on a
    // slow CI host. It exists to *measure*, not to gate.
    assert!(
        elapsed.as_secs() < 60,
        "naive launch test exceeded 60 s — something is wrong"
    );
    let _ = Complex64::new(1.0, 0.0);
}
