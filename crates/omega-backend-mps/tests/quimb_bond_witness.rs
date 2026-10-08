// SPDX-License-Identifier: Apache-2.0
//! The bond witness has to refuse an under-entangled circuit.
//!
//! `contract_bench` warmed up with CNOT on |+⟩|+⟩, an identity, and printed a
//! speedup beside a counter that said the accelerated path had not run. The
//! fixture below is that circuit. A harness that reports a time for it is
//! the bug.

#[path = "../examples/mps_quimb_compare/quimb_lane.rs"]
#[allow(dead_code)]
mod quimb_lane;

use quimb_lane::{
    abs_gap, admit_saturating_row, classify_speed, cnot_on_plus_plus, exact_biased,
    format_capability_row, hea, min_median, parse_uptime_load1, passes_exact_value_gate,
    relative_error, requires_bond_saturation, row_void_at_load, run_ours, select_knob,
    speed_headline, timed_repeats, truncation_matched, CutoffMode, Knob, MapCell, MapObs,
    SpeedClass,
};

fn knob(cutoff: f64, mode: CutoffMode) -> Knob {
    Knob { cutoff, mode }
}

fn cell(chi: usize, our_gap: f64, rows: Vec<(Knob, usize, f64)>) -> MapCell {
    MapCell {
        chi,
        our_abs_gap: our_gap,
        obs: rows
            .into_iter()
            .map(|(knob, max_bond, gap)| MapObs {
                knob,
                max_bond,
                abs_gap: gap,
            })
            .collect(),
    }
}

/// DOES NOT CATCH: quimb returning `max_bond == chi` for a state whose Schmidt
/// spectrum never left bond 1. The witness trusts the number the arm reports.
#[test]
fn unsaturated_quimb_bond_is_refused_even_when_our_svd_witness_passes() {
    let err = admit_saturating_row(32, 1, 32, 10, 64).unwrap_err();
    assert!(
        err.contains("quimb max_bond"),
        "the refusal has to be the quimb bond, not a different witness: {err}"
    );
    assert!(admit_saturating_row(32, 32, 32, 10, 64).is_ok());
}

/// DOES NOT CATCH: a call-count witness. This circuit does execute an SVD
/// (one CX), so "calls > 0" accepts it. The bond is what refuses.
#[test]
fn ghost_cnot_on_plus_never_reaches_chi_and_is_refused() {
    let chi = 32;
    let built = cnot_on_plus_plus();
    let run = run_ours(&built.circuit, chi).expect("ghost evolution");
    assert!(
        run.svd_calls > 0,
        "the ghost does call the SVD; a call-count witness would time it"
    );
    assert!(
        run.max_bond < chi,
        "fixture is not under-entangled: bond {} reached chi {chi}",
        run.max_bond
    );
    let err = admit_saturating_row(chi, run.max_bond, run.max_bond, run.svd_calls, run.max_cols)
        .unwrap_err();
    assert!(
        err.contains("quimb max_bond"),
        "feeding the ghost's own bond in as quimb's report must refuse: {err}"
    );
    assert!(requires_bond_saturation(12));
    assert!(!requires_bond_saturation(4));
}

/// DOES NOT CATCH: both engines sharing a wrong angle. The gate compares our
/// MPS to our dense statevector on a circuit chi can hold exactly.
#[test]
fn exact_row_matches_dense_within_1e_10() {
    let built = exact_biased();
    assert_eq!(built.circuit.ops.len(), built.gates.len());
    let oracle = quimb_lane::dense_z0(&built.circuit).expect("dense oracle");
    assert!(
        oracle.abs() > 0.1,
        "the exact-row fixture must sit away from zero, oracle={oracle}"
    );
    let ours = run_ours(&built.circuit, 8).expect("mps");
    assert!(
        ours.discarded_weight < 1e-8,
        "4q at chi 8 must be exact, discarded {}",
        ours.discarded_weight
    );
    assert!(
        passes_exact_value_gate(ours.value, oracle),
        "rel {} value {} oracle {}",
        relative_error(ours.value, oracle),
        ours.value,
        oracle
    );
    assert!(!passes_exact_value_gate(0.0, 1.0));
    let _ = abs_gap(ours.value, oracle);
}

/// DOES NOT CATCH: a ratio computed from means instead of mins. The inputs
/// here are already mins and medians.
#[test]
fn tie_rule_is_pinned_and_1_16_is_not_a_win() {
    let (class, ratio) = classify_speed(1.00, 1.01, 1.16, 1.17);
    assert_eq!(class, SpeedClass::Tie);
    assert!((ratio - 1.16).abs() < 1e-12);
    let (win, _) = classify_speed(1.0, 1.05, 2.0, 2.1);
    assert_eq!(win, SpeedClass::Win);
    let (loss, _) = classify_speed(2.0, 2.1, 1.0, 1.05);
    assert_eq!(loss, SpeedClass::Loss);
    // 1.4× on the mins, but the intervals overlap.
    let (overlap, _) = classify_speed(1.0, 1.5, 1.4, 2.0);
    assert_eq!(overlap, SpeedClass::Tie);
    assert_eq!(speed_headline(4, SpeedClass::Win), "directional-only");
    assert_eq!(speed_headline(12, SpeedClass::Win), "win");
    assert_eq!(speed_headline(12, SpeedClass::Tie), "tie");
}

/// DOES NOT CATCH: the gaps being measured through `to_dense`. The selector
/// trusts the gaps it is given.
#[test]
fn selector_rejects_an_unsaturated_or_mismatched_cutoff() {
    let rank = knob(0.0, CutoffMode::Rsum2);
    let default = knob(1e-10, CutoffMode::Rsum2);
    let cells = vec![cell(32, 1e-3, vec![(default, 4, 1e-3), (rank, 32, 1.1e-3)])];
    let chosen = select_knob(&cells).expect("rank cap matches and saturates");
    assert_eq!(chosen, rank);

    let none = vec![cell(32, 1e-3, vec![(default, 4, 1e-3)])];
    assert!(select_knob(&none).is_err());

    let far = vec![cell(
        32,
        1e-3,
        vec![(rank, 32, 1e-1), (default, 32, 1.0e-3)],
    )];
    assert_eq!(select_knob(&far).unwrap(), default);

    assert!(truncation_matched(1e-4, 1.5e-4));
    assert!(!truncation_matched(1e-4, 1e-2));
    assert!(truncation_matched(0.0, 1e-12));
    assert!(!truncation_matched(0.0, 1e-3));
}

#[test]
fn capability_row_is_not_a_speed_class() {
    let text = format_capability_row(1.2e-2, 0.99, 0.98, 0.02);
    assert!(text.contains("kind=capability"));
    assert!(text.contains("not-a-speed-result"));
    assert!(text.contains("estimate"));
    assert!(!text.contains("kind=win"));
}

#[test]
fn load_repeat_and_median_rules() {
    // The lane gates on the shared per-host threshold, not a copy of it.
    let small = omega_emu_compare::load_void_above(10);
    let large = omega_emu_compare::load_void_above(32);
    assert!(!row_void_at_load(small, small));
    assert!(row_void_at_load(small + 0.01, small));
    assert!(!row_void_at_load(6.0, large), "load 6 on 32 cores is quiet");
    assert!(row_void_at_load(6.0, small), "load 6 on 10 cores is not");
    let mac = "23:42  up 22 days,  1:42, 4 users, load averages: 1.76 1.93 1.73";
    assert!((parse_uptime_load1(mac).unwrap() - 1.76).abs() < 1e-12);
    let linux = " 12:00:00 up 1 day,  2:00,  1 user,  load average: 0.50, 0.40, 0.30";
    assert!((parse_uptime_load1(linux).unwrap() - 0.50).abs() < 1e-12);
    assert_eq!(timed_repeats(30.0), 5);
    assert_eq!(timed_repeats(30.1), 3);
    assert_eq!(min_median(&[3.0, 1.0, 2.0]), Some((1.0, 2.0)));
}

/// The HEA list and the circuit are one builder. This does not catch a
/// builder that emits the same wrong angle into both.
#[test]
fn hea_gate_list_is_the_circuit() {
    let built = hea(14, 4);
    assert_eq!(built.circuit.ops.len(), built.gates.len());
    assert_eq!(built.n, 14);
    assert_eq!(built.depth, 4);
}

/// The published rows ingest the committed QASM, not a second builder.
/// DOES NOT CATCH: a manifest hash that was updated together with a wrong file.
#[test]
fn committed_hea_matches_the_brickwall_builder() {
    for &(n, depth) in quimb_lane::grid_shapes() {
        let art = quimb_lane::load_hea(n, depth).expect("committed HEA");
        assert_eq!(art.built.n, n);
        assert_eq!(art.built.depth, depth);
        assert_eq!(art.sha256.len(), 64);
    }
}

#[test]
fn every_grid_row_names_a_blind_spot() {
    for &(n, depth) in quimb_lane::grid_shapes() {
        for &chi in quimb_lane::grid_chis() {
            let spot = quimb_lane::blind_spot(n, depth, chi);
            assert!(
                spot.starts_with("does not catch"),
                "{n}q d{depth} chi {chi}: {spot}"
            );
        }
    }
}
