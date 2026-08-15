//! End-to-end `Backend::execute` smoke for the OpenCL statevector
//! backend. Builds a small `CircuitIR` directly (bypassing the QASM
//! parser since this crate doesn't depend on it) and asserts the
//! expected output for a Bell state run with both the statevector
//! result and counts mode.
//!
//! Skipped automatically when no usable OpenCL ICD is present on the
//! host — the constructor returns `OpenClError::Unavailable` /
//! `Runtime` and we exit green so `cargo test --workspace` stays
//! clean on hosts without OpenCL.

#![cfg(feature = "opencl")]

use omega_backend_statevector_opencl::{OpenClError, OpenClStatevectorBackend};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn h_op(qubit: u32) -> GateOp {
    GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(qubit)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }
}

fn cx_op(control: u32, target: u32) -> GateOp {
    GateOp {
        gate: GateKind::CX,
        qubits: smallvec![Qubit(control), Qubit(target)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }
}

fn make_bell_circuit() -> CircuitIR {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(h_op(0));
    c.add_op(cx_op(0, 1));
    c
}

#[test]
fn execute_bell_circuit_statevector_matches_expected() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(OpenClError::Unavailable(_)) | Err(OpenClError::Runtime(_)) => {
            eprintln!("OpenCL unavailable — skipping execute smoke");
            return;
        }
        Err(e) => panic!("unexpected backend init error: {e}"),
    };
    let circuit = make_bell_circuit();
    let params = ParameterBinding::default();
    let config = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let res = backend
        .execute(&circuit, &params, &config)
        .expect("Bell circuit should execute on OpenCL");
    let amps = match res {
        ExecResult::Statevector(v) => v,
        other => panic!("expected Statevector, got {other:?}"),
    };
    assert_eq!(amps.len(), 4);
    let r = 1.0 / 2.0_f64.sqrt();
    let tol = 1e-5_f64;
    // |00⟩ and |11⟩ at amplitude r; |01⟩ and |10⟩ at zero.
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
fn execute_bell_circuit_counts_only_have_correlated_outcomes() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(OpenClError::Unavailable(_)) | Err(OpenClError::Runtime(_)) => {
            eprintln!("OpenCL unavailable — skipping execute counts smoke");
            return;
        }
        Err(e) => panic!("unexpected backend init error: {e}"),
    };
    let circuit = make_bell_circuit();
    let params = ParameterBinding::default();
    const SHOTS: u32 = 4096;
    let config = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(42),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let res = backend
        .execute(&circuit, &params, &config)
        .expect("Bell circuit should execute on OpenCL");
    let counts = match res {
        ExecResult::Counts(c) => c,
        other => panic!("expected Counts, got {other:?}"),
    };
    let total: u32 = counts.values().sum();
    assert_eq!(total, SHOTS, "shots conserved");
    // Bell state: only 0b00 (idx 0) and 0b11 (idx 3) should appear.
    // Width 2 is part of the key: a bare `0` is not an `Outcome`, and an
    // outcome of the same value at another width is a different key.
    let at = |v: u64| {
        counts
            .get(&omega_core::outcome::Outcome::from_u64(v, 2))
            .copied()
            .unwrap_or(0)
    };
    let n00 = at(0);
    let n11 = at(3);
    let n01 = at(1);
    let n10 = at(2);
    assert_eq!(n01, 0, "Bell state must not produce |01⟩");
    assert_eq!(n10, 0, "Bell state must not produce |10⟩");
    let half = SHOTS / 2;
    let band = SHOTS / 4;
    assert!(
        (n00 as i32 - half as i32).unsigned_abs() < band,
        "n00 = {n00} too far from {half}"
    );
    assert!(
        (n11 as i32 - half as i32).unsigned_abs() < band,
        "n11 = {n11} too far from {half}"
    );
}

#[test]
fn expectation_z0z1_on_bell_state_is_one() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(OpenClError::Unavailable(_)) | Err(OpenClError::Runtime(_)) => {
            eprintln!("OpenCL unavailable — skipping expectation smoke");
            return;
        }
        Err(e) => panic!("unexpected backend init error: {e}"),
    };
    let circuit = make_bell_circuit();
    let params = ParameterBinding::default();
    // ⟨Z₀Z₁⟩ on (|00⟩ + |11⟩)/√2 = (+1·1 + +1·1)/2 = 1.
    let z0z1 = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
    };
    let val = backend
        .expectation(&circuit, &params, &z0z1)
        .expect("expectation should run on OpenCL");
    let tol = 1e-5_f64;
    assert!((val - 1.0).abs() < tol, "⟨Z₀Z₁⟩ = {val} expected ~1.0");
}

#[test]
fn expectation_multi_returns_one_value_per_observable() {
    let backend = match OpenClStatevectorBackend::new() {
        Ok(b) => b,
        Err(OpenClError::Unavailable(_)) | Err(OpenClError::Runtime(_)) => {
            eprintln!("OpenCL unavailable — skipping expectation_multi smoke");
            return;
        }
        Err(e) => panic!("unexpected backend init error: {e}"),
    };
    let circuit = make_bell_circuit();
    let params = ParameterBinding::default();
    // ⟨Z₀⟩ and ⟨Z₁⟩ on the Bell state (|00⟩+|11⟩)/√2: both 0
    // (equal +1 / -1 amplitudes).
    let obs = vec![Observable::z(0), Observable::z(1)];
    let vals = backend
        .expectation_multi(&circuit, &params, &obs)
        .expect("expectation_multi should run on OpenCL");
    assert_eq!(vals.len(), 2);
    let tol = 1e-5_f64;
    assert!(vals[0].abs() < tol, "⟨Z₀⟩ = {} expected ~0", vals[0]);
    assert!(vals[1].abs() < tol, "⟨Z₁⟩ = {} expected ~0", vals[1]);
}
