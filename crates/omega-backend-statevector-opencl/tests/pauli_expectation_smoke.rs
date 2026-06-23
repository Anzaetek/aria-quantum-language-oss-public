//! Smoke tests for `OpenClState::pauli_expectation` — the device
//! reduction that replaced the host-side `expectation_pauli` loop
//! in `execute::expectation`.
//!
//! Three anchors:
//!
//! 1. Sanity on |+⟩ — ⟨+|X|+⟩ = 1, ⟨+|Z|+⟩ = 0. Pins the X-mask /
//!    sign-mask wiring against a tiny analytic answer.
//! 2. Bell-state ⟨ZZ⟩ = 1 on (|00⟩+|11⟩)/√2 — Pins the multi-qubit
//!    Z-mask path (sign_mask has two Z bits set).
//! 3. Random 14q state against a CPU-side reference inner-product:
//!    pin the device reduction against an independent oracle. n=14
//!    exercises the multi-work-group sum path that
//!    `inner_product_handles_multi_workgroup_state` already pins for
//!    the inner_product kernel.

#![cfg(feature = "opencl")]

use num_complex::Complex64;

use omega_backend_statevector_opencl::{pauli_masks, OpenClStatevectorBackend};
use omega_core::executor::PauliOp;

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

/// Reference CPU pauli-string expectation — single-Pauli-string,
/// computes ⟨ψ|P|ψ⟩ via direct iteration over the basis. Same shape
/// as the old `execute::expectation_pauli` we replaced.
fn host_pauli(sv: &[Complex64], pauli_string: &[(u32, PauliOp)]) -> Complex64 {
    let dim = sv.len();
    let mut result = Complex64::new(0.0, 0.0);
    for i in 0..dim {
        let mut j = i;
        let mut coeff = Complex64::new(1.0, 0.0);
        for (qubit, pauli) in pauli_string {
            let q = *qubit as usize;
            let bit = (i >> q) & 1;
            match pauli {
                PauliOp::I => {}
                PauliOp::Z => {
                    if bit == 1 {
                        coeff *= -1.0;
                    }
                }
                PauliOp::X => {
                    j ^= 1 << q;
                }
                PauliOp::Y => {
                    j ^= 1 << q;
                    if bit == 0 {
                        coeff *= Complex64::new(0.0, 1.0);
                    } else {
                        coeff *= Complex64::new(0.0, -1.0);
                    }
                }
            }
        }
        result += sv[i].conj() * coeff * sv[j];
    }
    result
}

#[test]
fn pauli_x_on_plus_state_is_one() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    // n=2 plus state on qubit 0, qubit 1 = |0⟩.
    let inv_sqrt_2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut state = backend.allocate(2).unwrap();
    state
        .write_state(&[
            Complex64::new(inv_sqrt_2, 0.0),
            Complex64::new(inv_sqrt_2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ])
        .unwrap();
    let (x_mask, sign_mask, y_factor) = pauli_masks(&[(0u32, PauliOp::X)]);
    let z = state
        .pauli_expectation(x_mask, sign_mask, y_factor)
        .unwrap();
    let err = (z.re - 1.0).abs() + z.im.abs();
    assert!(err < 1e-5, "⟨+|X|+⟩ = {z:?}, err = {err:.2e}");
}

#[test]
fn pauli_zz_on_bell_state_is_one() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let inv_sqrt_2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut state = backend.allocate(2).unwrap();
    state
        .write_state(&[
            Complex64::new(inv_sqrt_2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(inv_sqrt_2, 0.0),
        ])
        .unwrap();
    let (x_mask, sign_mask, y_factor) = pauli_masks(&[(0u32, PauliOp::Z), (1u32, PauliOp::Z)]);
    let z = state
        .pauli_expectation(x_mask, sign_mask, y_factor)
        .unwrap();
    let err = (z.re - 1.0).abs() + z.im.abs();
    assert!(err < 1e-5, "Bell ⟨ZZ⟩ = {z:?}, err = {err:.2e}");
}

#[test]
fn pauli_y_on_plus_state_is_zero() {
    // ⟨+|Y|+⟩ = 0: Y mixes |0⟩↔|1⟩ with imaginary phases that average
    // out on the real-amplitude |+⟩. Pins the y_factor + sign_mask
    // path together — `pauli_masks` builds (x_mask=1, sign_mask=1,
    // y_factor=i) for a single Y on q0.
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let inv_sqrt_2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut state = backend.allocate(1).unwrap();
    state
        .write_state(&[
            Complex64::new(inv_sqrt_2, 0.0),
            Complex64::new(inv_sqrt_2, 0.0),
        ])
        .unwrap();
    let (x_mask, sign_mask, y_factor) = pauli_masks(&[(0u32, PauliOp::Y)]);
    let z = state
        .pauli_expectation(x_mask, sign_mask, y_factor)
        .unwrap();
    let err = z.norm();
    assert!(err < 1e-5, "⟨+|Y|+⟩ = {z:?}, err = {err:.2e}");
}

#[test]
fn pauli_expectation_matches_host_on_random_14q() {
    // n=14 ⇒ multi-work-group reduction. Drive a non-trivial Pauli
    // string (X on q3, Y on q7, Z on q11) and compare against the
    // host-side reference inner-product. The string mixes all three
    // non-I operators so a bug in any of x_mask / sign_mask /
    // y_factor would surface here.
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };
    let psi = random_state(14, 0xDEAD_BEEF_F00D_FACE);
    let mut state = backend.allocate(14).unwrap();
    state.write_state(&psi).unwrap();

    let pauli_string = vec![(3u32, PauliOp::X), (7u32, PauliOp::Y), (11u32, PauliOp::Z)];
    let (x_mask, sign_mask, y_factor) = pauli_masks(&pauli_string);
    let got = state
        .pauli_expectation(x_mask, sign_mask, y_factor)
        .unwrap();
    let want = host_pauli(&psi, &pauli_string);
    let err = (got - want).norm();
    assert!(err < 1e-5, "got {got:?}, want {want:?}, err = {err:.2e}");
}
