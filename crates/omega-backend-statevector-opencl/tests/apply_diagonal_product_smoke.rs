//! Smoke tests for `OpenClState::apply_diagonal_product` and the
//! `execute::apply_ops_fused` walker.
//!
//! Mirror of Metal's `apply_diagonal_product_*` and
//! `apply_ops_fused_*` test set. Pins three things:
//!
//! 1. Fused dispatch matches N sequential `apply_diagonal` calls
//!    bit-for-bit (modulo f32 round-off) — diagonal gates commute,
//!    so order doesn't matter, but the per-amplitude product must
//!    include every factor.
//! 2. Empty factor list is a no-op (no kernel enqueued).
//! 3. The fusion walker collapses a run of consecutive Rz gates into
//!    one fused dispatch. We pin the `OpenClStatevectorBackend::
//!    dispatch_count()` counter delta across an 8-Rz layer at 8
//!    (sequential) vs 1 (fused, via the walker through the execute
//!    layer).

#![cfg(feature = "opencl")]

use num_complex::Complex64;

use omega_backend_statevector_opencl::{OpenClState, OpenClStatevectorBackend};

fn alloc(num_qubits: u32) -> Option<(OpenClStatevectorBackend, OpenClState)> {
    let backend = OpenClStatevectorBackend::new().ok()?;
    let state = backend.allocate(num_qubits).expect("allocate");
    Some((backend, state))
}

fn random_state(num_qubits: u32, seed: u64) -> Vec<Complex64> {
    // Tiny LCG seeded for byte-stable test inputs — no extra dep.
    let dim = 1usize << num_qubits;
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(dim);
    let mut sum_sq = 0.0_f64;
    for _ in 0..dim {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let re = (s as i64 as f64) / (i64::MAX as f64);
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let im = (s as i64 as f64) / (i64::MAX as f64);
        out.push(Complex64::new(re, im));
        sum_sq += re * re + im * im;
    }
    let norm = sum_sq.sqrt();
    for c in &mut out {
        *c /= norm;
    }
    out
}

fn max_abs_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((x.re - y.re).abs()).max((x.im - y.im).abs()))
        .fold(0.0_f64, f64::max)
}

#[test]
fn apply_diagonal_product_matches_sequential_apply_diagonal() {
    // Mix Rz-style pairs on q0/q2 with an S gate on q4 — same shape
    // the Metal test uses.
    let Some((backend, _)) = alloc(5) else { return };
    let init = random_state(5, 0xFEED_FACE_u64);

    let factors = vec![
        (
            0u32,
            Complex64::from_polar(1.0, -0.31),
            Complex64::from_polar(1.0, 0.31),
        ),
        (
            2u32,
            Complex64::from_polar(1.0, -0.71),
            Complex64::from_polar(1.0, 0.71),
        ),
        (4u32, Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)),
    ];

    let mut sequential = backend.allocate(5).unwrap();
    sequential.write_state(&init).unwrap();
    for (q, d0, d1) in &factors {
        sequential.apply_diagonal(*q, *d0, *d1).unwrap();
    }

    let mut fused = backend.allocate(5).unwrap();
    fused.write_state(&init).unwrap();
    fused.apply_diagonal_product(&factors).unwrap();

    let max = max_abs_diff(&sequential.read_state(), &fused.read_state());
    assert!(
        max < 1e-5,
        "fused dispatch diverged from sequential: max abs diff = {max}",
    );
}

#[test]
fn apply_diagonal_product_with_empty_factors_is_noop() {
    let Some((_b, mut state)) = alloc(4) else {
        return;
    };
    let want = random_state(4, 0xBEEF_C0FF_u64);
    state.write_state(&want).unwrap();
    state.apply_diagonal_product(&[]).unwrap();
    let got = state.read_state();
    let max = max_abs_diff(&want, &got);
    assert!(max < 1e-6, "empty factor list must be no-op: diff = {max}");
}

#[test]
fn apply_diagonal_product_rejects_out_of_range_qubit() {
    let Some((_b, mut state)) = alloc(3) else {
        return;
    };
    let factors = vec![
        (1u32, Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)),
        (5u32, Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)),
    ];
    let err = state
        .apply_diagonal_product(&factors)
        .expect_err("expected QubitOutOfRange");
    let msg = err.to_string();
    assert!(msg.contains("out of range"), "got: {msg}");
}

#[test]
fn apply_diagonal_product_eight_rz_layer_matches_sequential() {
    let Some((backend, _)) = alloc(12) else {
        return;
    };
    let init = random_state(12, 0xCAFE_BABE_u64);

    let factors: Vec<(u32, Complex64, Complex64)> = (0..8)
        .map(|i| {
            let theta = 0.137 * (i as f64 + 1.0);
            let half = theta / 2.0;
            (
                4 + i as u32,
                Complex64::from_polar(1.0, -half),
                Complex64::from_polar(1.0, half),
            )
        })
        .collect();

    let mut sequential = backend.allocate(12).unwrap();
    sequential.write_state(&init).unwrap();
    for (q, d0, d1) in &factors {
        sequential.apply_diagonal(*q, *d0, *d1).unwrap();
    }

    let mut fused = backend.allocate(12).unwrap();
    fused.write_state(&init).unwrap();
    fused.apply_diagonal_product(&factors).unwrap();

    let max = max_abs_diff(&sequential.read_state(), &fused.read_state());
    assert!(max < 1e-5, "8-Rz fused vs sequential diff = {max}");
}

#[test]
fn apply_diagonal_product_is_one_dispatch_for_eight_factors() {
    // Pin the contract that load-bears the walker: one
    // `apply_diagonal_product` call = exactly one kernel dispatch.
    // Compared against eight sequential `apply_diagonal` calls on a
    // sibling buffer (which must show 8 dispatches), the test rules
    // out a regression where the fused kernel silently splits into
    // per-factor sub-dispatches. Per-buffer counters make this
    // parallel-safe — sibling tests in the same binary can't
    // contaminate the deltas.
    let Some((backend, _)) = alloc(12) else {
        return;
    };

    let factors: Vec<(u32, Complex64, Complex64)> = (0..8)
        .map(|i| {
            let theta = 0.137 * (i as f64 + 1.0);
            let half = theta / 2.0;
            (
                4 + i as u32,
                Complex64::from_polar(1.0, -half),
                Complex64::from_polar(1.0, half),
            )
        })
        .collect();

    let mut fused = backend.allocate(12).unwrap();
    let before_fused = fused.dispatch_count();
    fused.apply_diagonal_product(&factors).unwrap();
    let fused_delta = fused.dispatch_count() - before_fused;
    assert_eq!(
        fused_delta, 1,
        "fused dispatch must be one kernel, got {fused_delta}",
    );

    let mut sequential = backend.allocate(12).unwrap();
    let before_seq = sequential.dispatch_count();
    for (q, d0, d1) in &factors {
        sequential.apply_diagonal(*q, *d0, *d1).unwrap();
    }
    let seq_delta = sequential.dispatch_count() - before_seq;
    assert_eq!(
        seq_delta, 8,
        "sequential apply_diagonal must dispatch once per factor, got {seq_delta}",
    );
}
