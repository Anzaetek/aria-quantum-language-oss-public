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
        ExecResult::Counts(c) => {
            let w = c.keys().next().map(|o| o.width()).unwrap_or(1);
            *c.get(&omega_core::outcome::Outcome::from_u64(1, w))
                .unwrap_or(&0) as f64
                / shots as f64
        }
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

/// **A per-pair two-qubit rate must actually reach the sampled distribution.**
///
/// The per-pair table (`{"2q": {"0,1": …, "1,2": …, "default": …}}`) was
/// unit-tested at the level of *rate selection* — `at_gate` returns the right
/// number for the right pair — and nowhere else. That leaves the interesting
/// half unproven: whether the selected rate is the one the backend then
/// applies. A table that parses correctly and is then ignored, or one whose
/// value is read for the wrong pair, passes every rate-selection test and
/// changes no answer.
///
/// So: two CX gates on disjoint pairs in one circuit, one pair given a heavy
/// depolarizing rate and the other essentially none. If per-pair selection
/// reaches the sampler, the two qubits' marginals must come apart. If the
/// implementation collapsed to a single rate — either one — they could not.
#[test]
fn a_per_pair_two_qubit_rate_moves_the_distribution_it_names() {
    const SRC: &str = r#"
circuit Pairs() {
  qreg q[4]
  creg c[4]
  apply X on q[0]
  apply X on q[2]
  apply CX on q[0], q[1]
  apply CX on q[2], q[3]
  measure q[0] -> c[0]
  measure q[1] -> c[1]
  measure q[2] -> c[2]
  measure q[3] -> c[3]
}
"#;
    let circuit = parse_aria(SRC)
        .expect("parse")
        .instantiate("Pairs", &[])
        .expect("instantiate");
    let binds = HashMap::new();
    // Pair (0,1) is hammered; pair (2,3) is left almost clean. `default` is
    // low so an implementation that fell back to it for BOTH pairs would give
    // two clean marginals and fail.
    let model = parse_noise_model(r#"{"depolarizing":{"2q":{"0,1":0.5,"2,3":0.0,"default":0.0}}}"#)
        .expect("per-pair model must parse");

    const SHOTS: u32 = 20000;
    for sel in [BackendSel::Sim, BackendSel::Mps { chi: 64 }] {
        let res = run_counts_noisy(&circuit, &binds, SHOTS, Some(7), sel, &model).unwrap();
        let counts = match &res {
            ExecResult::Counts(c) => c,
            other => panic!("expected counts, got {other:?}"),
        };
        // Marginal P(bit = 1) for each measured qubit.
        let marginal = |bit: usize| -> f64 {
            counts
                .iter()
                .filter(|(o, _)| o.bit(bit as u32) == 1)
                .map(|(_, n)| *n as u64)
                .sum::<u64>() as f64
                / SHOTS as f64
        };
        // Noiseless truth: q0=1, q1=1 (CX from |1>), q2=1, q3=1.
        let noisy_pair = marginal(1); // target of the hammered CX
        let clean_pair = marginal(3); // target of the clean CX

        assert!(
            clean_pair > 0.98,
            "{sel:?}: pair (2,3) has rate 0 and must stay ~1.0, got {clean_pair} \
             — a rate is leaking onto a pair that was not given one"
        );
        assert!(
            noisy_pair < 0.9,
            "{sel:?}: pair (0,1) has rate 0.5 and must be visibly depolarized, \
             got {noisy_pair} — the per-pair rate is parsed but never applied"
        );
        assert!(
            clean_pair - noisy_pair > 0.05,
            "{sel:?}: the two pairs must come apart (clean {clean_pair}, noisy \
             {noisy_pair}); if they agree, per-pair selection collapsed to one rate"
        );
    }
}
