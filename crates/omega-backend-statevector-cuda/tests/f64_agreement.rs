// SPDX-License-Identifier: Apache-2.0
//! The point of the f64 path: agreement with an exact reference far tighter than
//! f32 can reach.
//!
//! f32 on this backend agrees with the f64 CPU statevector to ~3.5e-8..5e-7,
//! which is six orders looser than the 1e-9 this project gates cross-checks at.
//! These tests assert f64 lands near machine epsilon instead, against closed
//! forms computed independently in Rust.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use std::sync::Arc;

use cudarc::driver::CudaContext;
use omega_backend_statevector_cuda::f64_path::{KernelsF64, StateF64};

fn ry(theta: f64) -> [(f64, f64); 4] {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    [(c, 0.0), (-s, 0.0), (s, 0.0), (c, 0.0)]
}

fn setup(n: u32) -> Option<StateF64> {
    let ctx = CudaContext::new(0).ok()?;
    let kernels = Arc::new(KernelsF64::load(&ctx).expect("f64 kernels compile"));
    Some(StateF64::zero(&ctx, kernels, n).expect("alloc"))
}

#[test]
fn ry_rotation_matches_the_closed_form_to_machine_precision() {
    let Some(mut st) = setup(4) else {
        eprintln!("no CUDA device — skipping");
        return;
    };
    // <Z0> after RY(theta) on wire 0 is exactly cos(theta).
    let theta = 0.7;
    st.apply_1q(0, ry(theta)).unwrap();
    let got = st.expectation_z(0).unwrap();
    let want = theta.cos();
    let err = (got - want).abs();
    eprintln!("f64 GPU <Z0> = {got:.17}, exact = {want:.17}, |err| = {err:.3e}");
    // f32 would sit near 1e-7 here; assert three orders tighter than the 1e-9
    // this project gates at, which f32 cannot do.
    assert!(err < 1e-12, "f64 path only reached {err:.3e}");
}

#[test]
fn a_chain_of_rotations_stays_exact() {
    let Some(mut st) = setup(6) else {
        return;
    };
    // Independent rotations on distinct wires: <Z_q> = cos(theta_q) for each,
    // so the product state gives six exact checks from one run.
    let thetas = [0.3, 1.1, 2.4, 0.05, 3.0, 1.75];
    for (q, &t) in thetas.iter().enumerate() {
        st.apply_1q(q as u32, ry(t)).unwrap();
    }
    let mut worst = 0.0f64;
    for (q, &t) in thetas.iter().enumerate() {
        let err = (st.expectation_z(q as u32).unwrap() - t.cos()).abs();
        worst = worst.max(err);
    }
    eprintln!("worst |err| over 6 wires = {worst:.3e}");
    assert!(worst < 1e-12, "worst error {worst:.3e}");
}

#[test]
fn cnot_makes_a_bell_state_with_exact_zero_marginals() {
    let Some(mut st) = setup(2) else {
        return;
    };
    // H on wire 0 then CNOT(0,1): both marginals are exactly 0, and in f64 that
    // should be at the 1e-16 level rather than 1e-8.
    let h = 1.0 / 2.0f64.sqrt();
    st.apply_1q(0, [(h, 0.0), (h, 0.0), (h, 0.0), (-h, 0.0)])
        .unwrap();
    // CNOT as a 4x4 row-major complex matrix, control = qa.
    //
    // The kernel states its convention: `r = bit_qb * 2 + bit_qa`, i.e. qa is
    // the LOW bit. So with control = qa the non-trivial rows are 1 <-> 3, not
    // 2 <-> 3. Getting this backwards encodes CNOT with the control on qb, and
    // since wire 1 starts in |0> the control is simply never on — the run
    // returns <Z1> = 1.0 exactly and looks like a broken gate rather than a
    // mis-specified matrix.
    let mut u = [0.0f64; 32];
    let set = |u: &mut [f64; 32], row: usize, col: usize, re: f64| {
        u[2 * (row * 4 + col)] = re;
    };
    set(&mut u, 0, 0, 1.0); // |qb=0,qa=0> -> itself
    set(&mut u, 2, 2, 1.0); // |qb=1,qa=0> -> itself
    set(&mut u, 3, 1, 1.0); // |qb=0,qa=1> -> |qb=1,qa=1>
    set(&mut u, 1, 3, 1.0); // |qb=1,qa=1> -> |qb=0,qa=1>
    st.apply_2q(0, 1, u).unwrap();
    let z0 = st.expectation_z(0).unwrap();
    let z1 = st.expectation_z(1).unwrap();
    eprintln!("Bell marginals: <Z0> = {z0:.3e}, <Z1> = {z1:.3e}");
    assert!(z0.abs() < 1e-14 && z1.abs() < 1e-14, "{z0:.3e} {z1:.3e}");
    // And the state is normalised.
    let host = st.to_host().unwrap();
    let norm: f64 = host.chunks(2).map(|c| c[0] * c[0] + c[1] * c[1]).sum();
    assert!((norm - 1.0).abs() < 1e-14, "norm {norm}");
}
