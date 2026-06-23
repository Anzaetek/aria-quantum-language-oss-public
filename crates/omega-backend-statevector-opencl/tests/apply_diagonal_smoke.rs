//! Smoke tests for `OpenClState::apply_diagonal` and
//! `OpenClState::apply_diagonal_2q`.
//!
//! Covers the gates the OpenCL `execute::apply_op` walker dispatches
//! through the diagonal fast paths (Z, S / Sdg, T / Tdg, Rz, U1 on
//! the 1q side; CZ, CRz on the 2q side). The intent is to pin both
//! the kernel arithmetic and the (qa low / qb high) bit ordering
//! against tiny analytic answers — same shape Metal's diagonal
//! coverage takes through the execute path, just exercised through
//! direct kernel calls so a future fusion-walker regression can't
//! quietly mask a per-kernel bug.

#![cfg(feature = "opencl")]

use std::f64::consts::FRAC_1_SQRT_2;
use std::f64::consts::PI;

use num_complex::Complex64;

use omega_backend_statevector_opencl::{OpenClState, OpenClStatevectorBackend};

fn alloc(num_qubits: u32) -> Option<(OpenClStatevectorBackend, OpenClState)> {
    let backend = OpenClStatevectorBackend::new().ok()?;
    let state = backend.allocate(num_qubits).expect("allocate");
    Some((backend, state))
}

/// Place each amplitude into the state via `write_state`. Used by the
/// tests to set up |+⟩ / |++⟩ / Bell starting states without an
/// `apply_h` round-trip (which would itself touch the kernel surface
/// under test).
fn set_state(state: &mut OpenClState, amps: &[Complex64]) {
    state.write_state(amps).expect("write_state");
}

fn assert_close(got: &[Complex64], want: &[Complex64], tol: f64) {
    assert_eq!(got.len(), want.len(), "dim mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let dr = (g.re - w.re).abs();
        let di = (g.im - w.im).abs();
        assert!(
            dr <= tol && di <= tol,
            "amp[{i}] got=({:.6}+{:.6}i) want=({:.6}+{:.6}i)",
            g.re,
            g.im,
            w.re,
            w.im
        );
    }
}

#[test]
fn apply_z_on_plus_yields_minus() {
    let Some((_b, mut s)) = alloc(1) else { return };
    let plus = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    set_state(&mut s, &plus);
    s.apply_diagonal(0, Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0))
        .expect("apply_z");
    let got = s.read_state();
    let want = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(-FRAC_1_SQRT_2, 0.0),
    ];
    assert_close(&got, &want, 1e-6);
}

#[test]
fn apply_s_then_sdg_is_identity() {
    let Some((_b, mut s)) = alloc(1) else { return };
    let plus = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    set_state(&mut s, &plus);
    s.apply_diagonal(0, Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0))
        .expect("S");
    s.apply_diagonal(0, Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0))
        .expect("Sdg");
    let got = s.read_state();
    assert_close(&got, &plus, 1e-6);
}

#[test]
fn apply_rz_on_plus_matches_analytical() {
    // Rz(θ)|+⟩ = (e^{-iθ/2}|0⟩ + e^{iθ/2}|1⟩) / √2.
    let Some((_b, mut s)) = alloc(1) else { return };
    let plus = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    set_state(&mut s, &plus);
    let theta = PI / 3.0;
    s.apply_diagonal(
        0,
        Complex64::from_polar(1.0, -theta / 2.0),
        Complex64::from_polar(1.0, theta / 2.0),
    )
    .expect("Rz");
    let got = s.read_state();
    let want = vec![
        FRAC_1_SQRT_2 * Complex64::from_polar(1.0, -theta / 2.0),
        FRAC_1_SQRT_2 * Complex64::from_polar(1.0, theta / 2.0),
    ];
    assert_close(&got, &want, 1e-6);
}

#[test]
fn apply_t_then_tdg_is_identity() {
    let Some((_b, mut s)) = alloc(1) else { return };
    let plus = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    set_state(&mut s, &plus);
    let pi4 = std::f64::consts::FRAC_PI_4;
    s.apply_diagonal(0, Complex64::new(1.0, 0.0), Complex64::from_polar(1.0, pi4))
        .expect("T");
    s.apply_diagonal(
        0,
        Complex64::new(1.0, 0.0),
        Complex64::from_polar(1.0, -pi4),
    )
    .expect("Tdg");
    let got = s.read_state();
    assert_close(&got, &plus, 1e-6);
}

#[test]
fn apply_u1_picks_up_phase_on_one() {
    // U1(λ)|+⟩ = (|0⟩ + e^{iλ}|1⟩) / √2.
    let Some((_b, mut s)) = alloc(1) else { return };
    let plus = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    set_state(&mut s, &plus);
    let lambda = 0.7;
    s.apply_diagonal(
        0,
        Complex64::new(1.0, 0.0),
        Complex64::from_polar(1.0, lambda),
    )
    .expect("U1");
    let got = s.read_state();
    let want = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        FRAC_1_SQRT_2 * Complex64::from_polar(1.0, lambda),
    ];
    assert_close(&got, &want, 1e-6);
}

#[test]
fn apply_cz_on_plus_plus_flips_sign_of_one_one() {
    // CZ|++⟩ = (|00⟩ + |01⟩ + |10⟩ - |11⟩) / 2.
    let Some((_b, mut s)) = alloc(2) else { return };
    let half = 0.5_f64;
    let plus_plus = vec![
        Complex64::new(half, 0.0),
        Complex64::new(half, 0.0),
        Complex64::new(half, 0.0),
        Complex64::new(half, 0.0),
    ];
    set_state(&mut s, &plus_plus);
    let one = Complex64::new(1.0, 0.0);
    s.apply_diagonal_2q(0, 1, one, one, one, Complex64::new(-1.0, 0.0))
        .expect("CZ");
    let got = s.read_state();
    let want = vec![
        Complex64::new(half, 0.0),
        Complex64::new(half, 0.0),
        Complex64::new(half, 0.0),
        Complex64::new(-half, 0.0),
    ];
    assert_close(&got, &want, 1e-6);
}

#[test]
fn apply_crz_on_bell_state_matches_analytical() {
    // Start from a Bell pair (|00⟩+|11⟩)/√2 — only |11⟩ is on the
    // controlled-target branch, so CRz(θ) leaves |00⟩ alone and
    // picks up e^{iθ/2} on |11⟩. Sanity check the (qa=control,
    // qb=target) → idx = bit_target*2 + bit_control ordering: d11
    // (idx 3, both bits = 1) is e^{iθ/2}.
    let Some((_b, mut s)) = alloc(2) else { return };
    let bell = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    set_state(&mut s, &bell);
    let theta = PI / 5.0;
    let one = Complex64::new(1.0, 0.0);
    let phm = Complex64::from_polar(1.0, -theta / 2.0);
    let php = Complex64::from_polar(1.0, theta / 2.0);
    // (qa=ctrl=0, qb=target=1, d00, d01, d10, d11) =
    // (0, 1, 1, e^{-iθ/2}, 1, e^{iθ/2}).
    s.apply_diagonal_2q(0, 1, one, phm, one, php).expect("CRz");
    let got = s.read_state();
    let want = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        FRAC_1_SQRT_2 * php,
    ];
    assert_close(&got, &want, 1e-6);
}

#[test]
fn apply_diagonal_rejects_qubit_out_of_range() {
    let Some((_b, mut s)) = alloc(2) else { return };
    let err = s
        .apply_diagonal(5, Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0))
        .expect_err("expected QubitOutOfRange");
    let msg = err.to_string();
    assert!(msg.contains("out of range"), "got: {msg}");
}

#[test]
fn apply_diagonal_2q_rejects_duplicate_qubits() {
    let Some((_b, mut s)) = alloc(3) else { return };
    let one = Complex64::new(1.0, 0.0);
    let err = s
        .apply_diagonal_2q(1, 1, one, one, one, one)
        .expect_err("expected DuplicateQubits");
    let msg = err.to_string();
    assert!(msg.contains("distinct qubits"), "got: {msg}");
}
