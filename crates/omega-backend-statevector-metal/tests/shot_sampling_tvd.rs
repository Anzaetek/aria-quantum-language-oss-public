//! GPU shot-sampling correctness — `MetalState::sample_shots_gpu`.
//!
//! Acceptance gate per `NEXT_SESSION_PLAN.md` (item B): the
//! GPU sampler's empirical outcome distribution matches the
//! exact `|amp|²` distribution within statistical bounds at
//! `shots ≥ 10⁴`. We assert TVD ≤ 0.01 at 10⁵ shots — the
//! pure-statistical floor at this shot count is `1/√N ≈ 3e-3`,
//! so 1e-2 is a comfortable margin above the noise while still
//! tight enough to catch a broken sampler (a uniformly-broken
//! sampler on a Bell pair produces TVD ≈ 0.5).
//!
//! Determinism is pinned in a separate test: same seed → byte-for-
//! byte identical outcome counts.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::collections::HashMap;

use num_complex::Complex64;

use omega_backend_statevector_metal::MetalStatevectorBackend;

/// Total variation distance between an empirical count map and a
/// reference probability vector indexed by outcome.
fn tvd(counts: &HashMap<u64, u32>, probs: &[f64]) -> f64 {
    let total: u32 = counts.values().sum();
    let mut sum_abs = 0.0_f64;
    for (i, p) in probs.iter().enumerate() {
        let c = counts.get(&(i as u64)).copied().unwrap_or(0);
        let empirical = c as f64 / total as f64;
        sum_abs += (empirical - p).abs();
    }
    // Outcomes outside `probs.len()` should never appear; surface any
    // out-of-range slot as TVD contribution.
    for (&k, &c) in counts.iter() {
        if (k as usize) >= probs.len() {
            sum_abs += (c as f64) / (total as f64);
        }
    }
    0.5 * sum_abs
}

fn write_uniform_superposition(state: &mut omega_backend_statevector_metal::MetalState, k: usize) {
    let dim = 1usize << state.num_qubits();
    assert!(k <= dim, "k must fit in the state dimension");
    let amp = Complex64::new(1.0 / (k as f64).sqrt(), 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let mut amps = vec![zero; dim];
    amps[..k].fill(amp);
    state.write_state(&amps).expect("write_state");
}

#[test]
fn sample_shots_gpu_tvd_within_statistical_floor() {
    let backend = match MetalStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return, // no Metal device — skip.
    };
    // 4-qubit, k=5 non-trivial superposition: probability mass split
    // across 5 of 16 bins at 1/5 each. The 11 zero-probability bins
    // catch a sampler that uniformly draws over the full state dim
    // (a buggy "no CDF" sampler would hit them at uniform rate).
    let mut state = backend.allocate(4).expect("allocate");
    write_uniform_superposition(&mut state, 5);

    let mut true_probs = vec![0.0_f64; 16];
    true_probs[..5].fill(1.0 / 5.0);

    const SHOTS: u32 = 100_000;
    let counts = state
        .sample_shots_gpu(SHOTS, 0xCAFE_BABE_DEAD_BEEF)
        .expect("sample");
    let total: u32 = counts.values().sum();
    assert_eq!(total, SHOTS, "every shot must produce one outcome");

    let d = tvd(&counts, &true_probs);
    // 1/√100k ≈ 3.2e-3 — give it a 3× margin.
    assert!(
        d < 1e-2,
        "GPU sampler TVD {d:.5} exceeded 1e-2 budget at {SHOTS} shots",
    );
}

#[test]
fn sample_shots_gpu_is_deterministic_for_fixed_seed() {
    let backend = match MetalStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let mut state = backend.allocate(3).expect("allocate");
    write_uniform_superposition(&mut state, 7);

    const SHOTS: u32 = 10_000;
    const SEED: u64 = 0x1234_5678_9ABC_DEF0;
    let c1 = state.sample_shots_gpu(SHOTS, SEED).expect("sample 1");
    let c2 = state.sample_shots_gpu(SHOTS, SEED).expect("sample 2");
    assert_eq!(c1, c2, "identical seeds must produce identical count maps",);
}

#[test]
fn sample_shots_gpu_zero_shots_returns_empty_map() {
    let backend = match MetalStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let state = backend.allocate(2).expect("allocate");
    let counts = state.sample_shots_gpu(0, 42).expect("sample");
    assert!(counts.is_empty(), "zero shots yields empty count map");
}

#[test]
fn sample_shots_gpu_zero_state_collapses_to_basis_zero() {
    let backend = match MetalStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    // Fresh |0…0⟩ — single basis state should absorb every shot.
    let state = backend.allocate(3).expect("allocate");
    const SHOTS: u32 = 5_000;
    let counts = state.sample_shots_gpu(SHOTS, 99).expect("sample");
    assert_eq!(counts.get(&0).copied().unwrap_or(0), SHOTS);
    assert_eq!(counts.len(), 1);
}
