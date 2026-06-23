//! End-to-end smoke for the OpenCL statevector backend's `apply_1q`
//! kernel.
//!
//! Skipped automatically when the `opencl` feature is off, when no
//! OpenCL ICD is present, or when the platform / device probe fails
//! at construction time. The CPU-only `cargo test --workspace`
//! invocation in `ci.sh` therefore stays clean on hosts without an
//! OpenCL toolkit installed.

#![cfg(feature = "opencl")]

use num_complex::Complex64;
use omega_backend_statevector_opencl::{OpenClError, OpenClStatevectorBackend};

fn frac_root_two() -> f64 {
    1.0 / 2.0_f64.sqrt()
}

#[test]
fn apply_h_on_one_qubit_state_lifts_to_plus_state() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(OpenClError::Unavailable(msg)) => {
            eprintln!("OpenCL unavailable ({msg}) — skipping apply_1q smoke");
            return;
        }
        Err(OpenClError::Runtime(msg)) => {
            eprintln!("OpenCL runtime error ({msg}) — skipping apply_1q smoke");
            return;
        }
        Err(e) => panic!("unexpected backend init error: {e}"),
    };

    let mut state = backend
        .allocate(1)
        .expect("allocate 1q buffer on a working OpenCL device");

    // Hadamard: U = (1/√2) * [[1, 1], [1, -1]]
    let r = frac_root_two();
    let u00 = Complex64::new(r, 0.0);
    let u01 = Complex64::new(r, 0.0);
    let u10 = Complex64::new(r, 0.0);
    let u11 = Complex64::new(-r, 0.0);
    state
        .apply_1q(0, u00, u01, u10, u11)
        .expect("apply_1q on a working OpenCL device");

    let amps = state.read_state();
    assert_eq!(amps.len(), 2);
    let tol = 1e-5_f64;
    assert!(
        (amps[0].re - r).abs() < tol && amps[0].im.abs() < tol,
        "amp[0] = {} expected ~{r}",
        amps[0]
    );
    assert!(
        (amps[1].re - r).abs() < tol && amps[1].im.abs() < tol,
        "amp[1] = {} expected ~{r}",
        amps[1]
    );
}

#[test]
fn apply_cnot_on_plus_zero_yields_bell_state() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(OpenClError::Unavailable(_)) | Err(OpenClError::Runtime(_)) => {
            eprintln!("OpenCL unavailable — skipping apply_2q CNOT smoke");
            return;
        }
        Err(e) => panic!("unexpected backend init error: {e}"),
    };

    let mut state = backend.allocate(2).expect("allocate 2q buffer");

    // H on qubit 0 — drives |00⟩ → (1/√2)(|00⟩ + |01⟩).
    let r = frac_root_two();
    let u00 = Complex64::new(r, 0.0);
    let u01 = Complex64::new(r, 0.0);
    let u10 = Complex64::new(r, 0.0);
    let u11 = Complex64::new(-r, 0.0);
    state.apply_1q(0, u00, u01, u10, u11).expect("apply H");

    // CNOT(control=0, target=1) — qubits in state ordering (qa, qb)
    // with qa low row-bit. Control qubit 0 is qa; target qubit 1 is
    // qb. With (qa low, qb high) row-major:
    //   row 0 (qb=0,qa=0) = |00⟩  → identity
    //   row 1 (qb=0,qa=1) = |01⟩  → flips target qubit 1 → |11⟩
    //   row 2 (qb=1,qa=0) = |10⟩  → identity
    //   row 3 (qb=1,qa=1) = |11⟩  → flips target qubit 1 → |01⟩
    // U is row-major; (re, im) interleaved.
    #[rustfmt::skip]
    let u_cnot: [f32; 32] = [
        // row 0: 1 0 0 0
        1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
        // row 1: 0 0 0 1
        0.0, 0.0,  0.0, 0.0,  0.0, 0.0,  1.0, 0.0,
        // row 2: 0 0 1 0
        0.0, 0.0,  0.0, 0.0,  1.0, 0.0,  0.0, 0.0,
        // row 3: 0 1 0 0
        0.0, 0.0,  1.0, 0.0,  0.0, 0.0,  0.0, 0.0,
    ];
    state.apply_2q(0, 1, &u_cnot).expect("apply CNOT");

    let amps = state.read_state();
    assert_eq!(amps.len(), 4);
    let tol = 1e-5_f64;
    // Expected: (1/√2)(|00⟩ + |11⟩). Index layout: q0 LSB.
    // |00⟩ = idx 0; |11⟩ = idx 3.
    assert!(
        (amps[0].re - r).abs() < tol && amps[0].im.abs() < tol,
        "amp[0] = {} expected ~{r}",
        amps[0]
    );
    assert!(amps[1].norm() < tol, "amp[1] = {}", amps[1]);
    assert!(amps[2].norm() < tol, "amp[2] = {}", amps[2]);
    assert!(
        (amps[3].re - r).abs() < tol && amps[3].im.abs() < tol,
        "amp[3] = {} expected ~{r}",
        amps[3]
    );
}

#[test]
fn apply_x_on_two_qubit_state_flips_target() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(OpenClError::Unavailable(_)) | Err(OpenClError::Runtime(_)) => {
            eprintln!("OpenCL unavailable — skipping apply_1q 2q smoke");
            return;
        }
        Err(e) => panic!("unexpected backend init error: {e}"),
    };

    let mut state = backend.allocate(2).expect("allocate 2q buffer");

    // X gate on qubit 0: |00⟩ → |01⟩ (qubit 0 is LSB).
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    state.apply_1q(0, zero, one, one, zero).expect("apply X");

    let amps = state.read_state();
    let tol = 1e-5_f64;
    // |0,1,0,0⟩ in |q1q0⟩ basis: index 1 is |q1=0, q0=1⟩.
    assert!(amps[0].norm() < tol, "amp[0] = {}", amps[0]);
    assert!(
        (amps[1].re - 1.0).abs() < tol && amps[1].im.abs() < tol,
        "amp[1] = {}",
        amps[1]
    );
    assert!(amps[2].norm() < tol, "amp[2] = {}", amps[2]);
    assert!(amps[3].norm() < tol, "amp[3] = {}", amps[3]);
}
