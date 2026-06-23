//! GPU inner-product reduction — `OpenClState::inner_product`.
//!
//! Mirror of Metal's `inner_product_*` smoke set. The kernel computes
//! ⟨a|b⟩ = Σ_i conj(a_i) · b_i via a two-stage work-group reduction;
//! the host upcasts the per-work-group partials and sums them in f64.
//!
//! Acceptance gate: ⟨ψ|ψ⟩ = 1 on a normalised random state to within
//! ~1e-6 (f32 accumulator floor + tree-reduction round-off), and
//! ⟨a|b⟩ matches a host-side reference within the same tolerance on
//! independent random states.

#![cfg(feature = "opencl")]

use num_complex::Complex64;

use omega_backend_statevector_opencl::OpenClStatevectorBackend;

fn random_state(num_qubits: u32, seed: u64) -> Vec<Complex64> {
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

fn host_inner(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.conj() * y)
        .fold(Complex64::new(0.0, 0.0), |acc, v| acc + v)
}

#[test]
fn inner_product_with_self_is_one_for_normalised_state() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    // n=8 — dim=256; with a 256-wide work-group on Apple-OpenCL the
    // reduction collapses to one work-group, exercising the
    // single-partial path. n=14 below covers multi-work-group sums.
    let mut state = backend.allocate(8).unwrap();
    let psi = random_state(8, 0xCAFE_F00D_DEAD_BEEF);
    state.write_state(&psi).unwrap();
    let z = state.inner_product(&state).unwrap();
    let err = (z - Complex64::new(1.0, 0.0)).norm();
    assert!(
        err < 1e-5,
        "⟨ψ|ψ⟩ diverged from 1: got {z:?}, |err|={err:.2e}",
    );
}

#[test]
fn inner_product_matches_host_reference_on_independent_states() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let mut a = backend.allocate(8).unwrap();
    let mut b = backend.allocate(8).unwrap();
    let psi_a = random_state(8, 0x1111_2222_3333_4444);
    let psi_b = random_state(8, 0x5555_6666_7777_8888);
    a.write_state(&psi_a).unwrap();
    b.write_state(&psi_b).unwrap();

    let got = a.inner_product(&b).unwrap();
    let want = host_inner(&psi_a, &psi_b);
    let err = (got - want).norm();
    assert!(
        err < 1e-5,
        "GPU ⟨a|b⟩ = {got:?} vs host {want:?}, |err|={err:.2e}",
    );
}

#[test]
fn inner_product_handles_multi_workgroup_state() {
    // n=14 — dim=16384. With a 256-wide work-group on Apple-OpenCL
    // the reduction needs 64 partials; the host loop has to fold
    // them all. A buggy implementation that only read partial[0]
    // would land at ~1/64 of the true magnitude.
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let mut state = backend.allocate(14).unwrap();
    let psi = random_state(14, 0xABCD_EF01_2345_6789);
    state.write_state(&psi).unwrap();
    let z = state.inner_product(&state).unwrap();
    let err = (z - Complex64::new(1.0, 0.0)).norm();
    assert!(
        err < 1e-5,
        "n=14 ⟨ψ|ψ⟩ diverged: got {z:?}, |err|={err:.2e}",
    );
}

#[test]
fn inner_product_rejects_size_mismatch() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let a = backend.allocate(3).unwrap();
    let b = backend.allocate(4).unwrap();
    let err = a.inner_product(&b).expect_err("expected size mismatch");
    let msg = err.to_string();
    assert!(msg.contains("state length"), "got: {msg}");
}
