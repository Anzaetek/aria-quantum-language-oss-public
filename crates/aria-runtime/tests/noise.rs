// SPDX-License-Identifier: Apache-2.0
//! Regression + parity gate for the `--noise` path.
//!
//! The original bug: `--noise` was silently a no-op — a deterministic
//! `X; measure` circuit returned the *noiseless* `P(1) = 1` even with a noise
//! model set. These tests pin that every noise-capable backend now either
//! *applies* the model (moving the distribution) or *rejects* it loudly — never
//! silently returns the noiseless result — and that the numbers match the
//! analytic density-matrix values.

use std::collections::HashMap;

use aria_core::ast::{parse_aria, Circuit};
use aria_runtime::{expectation_noisy, parse_noise_model, run_counts_noisy, BackendSel};
use omega_core::executor::ExecResult;

/// Deterministic `X; measure` on one qubit — prepares |1⟩.
const DAMP: &str = r#"
circuit Damp() {
  qreg q[1]
  creg c[1]
  apply X on q[0]
  measure q[0] -> c[0]
}
"#;

fn damp_circuit() -> Circuit {
    parse_aria(DAMP)
        .expect("parse")
        .instantiate("Damp", &[])
        .expect("instantiate")
}

fn p1(res: &ExecResult, shots: u32) -> f64 {
    match res {
        ExecResult::Counts(c) => *c.get(&1).unwrap_or(&0) as f64 / shots as f64,
        other => panic!("expected counts, got {other:?}"),
    }
}

#[test]
fn readout_flip_moves_distribution_on_sampling_backends() {
    // The exact scenario from the bug report: readout_flip:0.5 on |1⟩ must give
    // ~50/50, not the noiseless P(1)=1, on BOTH trajectory samplers.
    let c = damp_circuit();
    let binds = HashMap::new();
    let model = parse_noise_model(r#"{"readout_flip":0.5}"#).unwrap();
    for sel in [BackendSel::Sim, BackendSel::Mps { chi: 64 }] {
        let res = run_counts_noisy(&c, &binds, 20000, Some(1), sel, &model).unwrap();
        let got = p1(&res, 20000);
        assert!(
            (got - 0.5).abs() < 0.02,
            "{:?}: readout_flip:0.5 should give P(1)≈0.5, got {got} (noiseless bug?)",
            sel
        );
    }
}

#[test]
fn amplitude_damping_matches_analytic_on_sampling_backends() {
    // amplitude_damping:0.5 on |1⟩ ⇒ P(1) = 1−γ = 0.5 (was silently 1.0).
    let c = damp_circuit();
    let binds = HashMap::new();
    let model = parse_noise_model(r#"{"amplitude_damping":0.5}"#).unwrap();
    for sel in [BackendSel::Sim, BackendSel::Mps { chi: 64 }] {
        let res = run_counts_noisy(&c, &binds, 20000, Some(2), sel, &model).unwrap();
        let got = p1(&res, 20000);
        assert!(
            (got - 0.5).abs() < 0.02,
            "{:?}: amplitude_damping:0.5 should give P(1)≈0.5, got {got}",
            sel
        );
    }
}

#[test]
fn noise_sampling_rejected_loudly_on_unsupported_backends() {
    // pauliprop / gpu / tch can't apply a noise model to sampled counts — they
    // must error, never silently drop it.
    let c = damp_circuit();
    let binds = HashMap::new();
    let model = parse_noise_model(r#"{"readout_flip":0.5}"#).unwrap();
    for sel in [BackendSel::PauliProp, BackendSel::Gpu, BackendSel::Tch] {
        let res = run_counts_noisy(&c, &binds, 100, Some(1), sel, &model);
        assert!(
            res.is_err(),
            "{:?} should reject --noise sampling loudly",
            sel
        );
    }
}

#[test]
fn pauliprop_noisy_expectation_is_exact() {
    // ⟨Z0⟩ of |1⟩ under amplitude damping γ is exactly 2γ−1 (deterministic).
    let c = damp_circuit();
    let binds = HashMap::new();
    let gamma = 0.3;
    let model = parse_noise_model(r#"{"amplitude_damping":0.3}"#).unwrap();
    let got = expectation_noisy(&c, "Z0", &binds, BackendSel::PauliProp, &model).unwrap();
    assert!(
        (got - (2.0 * gamma - 1.0)).abs() < 1e-12,
        "pauliprop noisy ⟨Z0⟩ = {got}, want {}",
        2.0 * gamma - 1.0
    );
    // The same request on a backend with only analytic expectations is rejected.
    assert!(expectation_noisy(&c, "Z0", &binds, BackendSel::Sim, &model).is_err());
}

#[test]
fn per_qubit_and_asymmetric_parse_round_trips() {
    // The calibrated-hardware forms parse; a typo is rejected (never silently 0).
    assert!(parse_noise_model(r#"{"amplitude_damping":[0.004,0.006]}"#).is_ok());
    assert!(parse_noise_model(r#"{"depolarizing":{"1q":0.001,"2q":0.012}}"#).is_ok());
    assert!(parse_noise_model(r#"{"readout":[{"p10":0.02,"p01":0.03}]}"#).is_ok());
    assert!(parse_noise_model(r#"{"readout_flip":0.02}"#).is_ok());
    // Both readout forms at once, and a typo'd key, are rejected.
    assert!(parse_noise_model(r#"{"readout_flip":0.02,"readout":0.02}"#).is_err());
    assert!(parse_noise_model(r#"{"reado":0.02}"#).is_err());
}
