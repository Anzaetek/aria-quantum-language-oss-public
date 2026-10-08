// SPDX-License-Identifier: Apache-2.0
//! A10 for E2. Each test is aimed at one line a later edit could soften:
//! a tolerance in place of the integer, a short sample accepted, a
//! disagreement dropped, a Pauli string aimed at the wrong wire, a
//! sampler asked for fewer shots than the row claims, a smaller surface
//! code, a noise instruction skipped, a T that Stim is reported to have
//! refused when it did not.

use emu_compare_stim::assemble::{derive, timing};
use emu_compare_stim::gate::{
    account_shots, detector_noise_refusal, exact_clifford_integer, exact_gate,
};
use emu_compare_stim::stim_client::StimClient;
use emu_compare_stim::workload::{
    load_artifact, observable_to_stim_pauli, pauli_expectation, pauli_sample, qec_d5, to_stim,
    with_terminal_measures, without_measures, SEED,
};
use omega_emu_compare::Classification;

/// A registered skip, not a silent one: the five tests below need the stim
/// venv, and a clean checkout on any box does not have it. Failing there made
/// `cargo test --workspace` red on a fresh Linux clone, against this repo's
/// own rule elsewhere — a venv-gated leg announces its absence and skips.
/// Set `EMU_STIM_REQUIRE=1` (as a box claiming this lane should) to turn the
/// skip back into a failure, which is what stops the skip being silent.
fn stim_or_skip() -> Option<StimClient> {
    match StimClient::launch() {
        Ok(c) => Some(c),
        Err(e) => {
            if std::env::var("EMU_STIM_REQUIRE").as_deref() == Ok("1") {
                panic!("EMU_STIM_REQUIRE=1 but the stim arm is unavailable: {e}");
            }
            eprintln!("SKIP: stim arm unavailable, so this lane test cannot run: {e}");
            None
        }
    }
}

#[test]
fn exact_integer_gate_rejects_a_near_miss() {
    assert_eq!(exact_clifford_integer(1.0), Some(1));
    assert_eq!(exact_clifford_integer(-1.0), Some(-1));
    assert_eq!(exact_clifford_integer(0.0), Some(0));
    assert_eq!(exact_gate(1.0, 1.0).unwrap(), 1);
    assert_eq!(exact_gate(-1.0, -1.0).unwrap(), -1);
    assert_eq!(exact_gate(0.0, 0.0).unwrap(), 0);

    // One ulp-scale step past 1. A 1e-6 tolerance admits this; integer
    // equality does not.
    let near = 1.0 + 1e-8;
    assert!(
        exact_clifford_integer(near).is_none(),
        "a near miss is not an exact Clifford integer"
    );
    let err = exact_gate(1.0, near).expect_err("near miss");
    assert!(
        err.reason.contains("disagreement finding"),
        "{}",
        err.reason
    );

    // Shared and wrong is still wrong. Both arms at 0.5 agree with each
    // other and with nothing these circuits can return.
    assert!(exact_gate(0.5, 0.5).is_err());
}

#[test]
fn shot_count_accounting_rejects_a_short_sample() {
    account_shots(10_000, 10_000).unwrap();
    let err = account_shots(10_000, 9_999).expect_err("short sample");
    assert!(err.contains("shortfall"), "{err}");
    assert!(err.contains("10000"), "{err}");
    assert!(err.contains("9999"), "{err}");
}

#[test]
fn a_disagreement_is_a_finding_not_a_dropped_row() {
    let err = exact_gate(1.0, 0.0).expect_err("arms differ");
    assert!(
        err.reason.contains("disagreement finding"),
        "a differing pair must be named as a finding, got {}",
        err.reason
    );
    assert!(err.reason.contains("ours=1"), "{}", err.reason);
    assert!(err.reason.contains("stim=0"), "{}", err.reason);
    assert!(
        err.reason.contains("not dropped"),
        "the finding has to say it was kept, got {}",
        err.reason
    );
}

#[test]
fn peek_observable_expectation_matches_pauli_on_the_committed_brickwall() {
    let art = load_artifact("clifford-24q-d100").expect("committed brickwall");
    assert_eq!(art.qubits, 24);
    assert_eq!(art.depth, 100);
    let pauli = observable_to_stim_pauli("Z0", art.qubits).unwrap();
    // Independent of the converter's own concatenation: wire 0 is the
    // first character after the sign, and the rest are identity.
    assert_eq!(pauli, format!("+Z{}", "_".repeat(23)));

    let ours = pauli_expectation(&art.circuit, "Z0").unwrap();
    let Some(mut stim) = stim_or_skip() else {
        return;
    };
    stim.load("brick", &to_stim(&art.circuit).unwrap()).unwrap();
    let (theirs, _) = stim.expectation("brick", &pauli).unwrap();
    let v = exact_gate(ours, theirs as f64).unwrap_or_else(|d| panic!("{}", d.reason));
    assert!(v == -1 || v == 0 || v == 1);
}

#[test]
fn both_arms_account_for_every_requested_shot() {
    let art = load_artifact("clifford-24q-d100").unwrap();
    let shots = 16u32;
    let (observed, width) = pauli_sample(&art.circuit, shots, SEED, false).unwrap();
    account_shots(u64::from(shots), observed).unwrap();
    assert_eq!(width, 24, "the brickwall sample is the full register");

    let Some(mut stim) = stim_or_skip() else {
        return;
    };
    let text = to_stim(&with_terminal_measures(&art.circuit)).unwrap();
    let loaded = stim.load("samp", &text).unwrap();
    assert_eq!(loaded.num_measurements, 24);
    stim.compile("samp", SEED).unwrap();
    let sampled = stim.sample("samp", u64::from(shots)).unwrap();
    account_shots(u64::from(shots), sampled.shots_observed).unwrap();
    assert_eq!(sampled.num_measurements, 24);
}

#[test]
fn the_d5_memory_circuit_matches_stim_on_logical_z() {
    let qec = qec_d5().unwrap();
    assert_eq!(qec.qubits, 49);
    assert_eq!(qec.measures, 24);
    assert_eq!(qec.observable, "Z0Z5Z10Z15Z20");
    let bare = without_measures(&qec.circuit);
    let pauli = observable_to_stim_pauli(&qec.observable, qec.qubits).unwrap();
    let ours = pauli_expectation(&bare, &qec.observable).unwrap();
    let Some(mut stim) = stim_or_skip() else {
        return;
    };
    stim.load("qec", &to_stim(&bare).unwrap()).unwrap();
    let (theirs, _) = stim.expectation("qec", &pauli).unwrap();
    exact_gate(ours, theirs as f64).unwrap_or_else(|d| panic!("{}", d.reason));
}

#[test]
fn noise_and_detectors_are_a_refusal_not_a_skipped_instruction() {
    // Annotations and the Clifford skeleton are not a refusal by themselves.
    assert!(detector_noise_refusal(&["H", "CX", "TICK", "MR", "M", "R"]).is_err());
    let err = detector_noise_refusal(&["H", "DEPOLARIZE1"]).unwrap();
    assert!(err.contains("DEPOLARIZE1"), "{err}");
    let err = detector_noise_refusal(&["DETECTOR"]).unwrap();
    assert!(err.contains("DETECTOR"), "{err}");
    let err = detector_noise_refusal(&["H", "NOT_A_GATE"]).expect_err("unclassified");
    assert!(err.contains("unclassified"), "{err}");
    assert!(err.contains("NOT_A_GATE"), "{err}");
}

#[test]
fn stim_refuses_a_t_gate() {
    let Some(mut stim) = stim_or_skip() else {
        return;
    };
    let err = stim.refuse_t("H 0\nT 0").unwrap();
    assert!(err.contains("Gate not found"), "{err}");
    assert!(err.contains('T'), "{err}");
}

#[test]
fn a_large_stim_advantage_classifies_as_a_loss_and_a_tiny_one_is_overhead() {
    let ours = timing(&[1.0, 1.05, 1.1, 1.2, 1.3]).unwrap();
    let stim = timing(&[0.01, 0.011, 0.012, 0.013, 0.014]).unwrap();
    let d = derive(&ours, &stim, 1e-6);
    assert_eq!(d.classification, Classification::Loss);
    assert!(d.ratio < 1.0);
    assert!(!d.overhead_dominated, "0.01s is far above a 1e-6 floor");

    let tiny = timing(&[1e-5, 1.1e-5, 1.2e-5, 1.3e-5, 1.4e-5]).unwrap();
    let d = derive(&ours, &tiny, 1e-5);
    assert!(
        d.overhead_dominated,
        "under 10x the floor must be annotated"
    );
}

#[test]
fn committed_d25_circuit_is_what_stim_1_16_0_generates() {
    let Some(mut stim) = stim_or_skip() else {
        return;
    };
    let got = stim.d25_text().unwrap();
    let path = emu_compare_stim::workload::repo_root()
        .join("tools/emu_compare/stim/surface_d25_rotated_memory_z.stim");
    let committed =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(got["text"].as_str().unwrap(), committed);
    assert_eq!(got["qubits"].as_u64().unwrap(), 1324);
    let names: Vec<&str> = got["instructions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(names.contains(&"DEPOLARIZE1"), "{names:?}");
    assert!(names.contains(&"DETECTOR"), "{names:?}");
}
