// SPDX-License-Identifier: Apache-2.0
//! A10 for E6. A speed classification on a ≤8-qubit row fails. A seeded
//! sign flip that the value does not feel fails. A one-sided gap fails.

use emu_compare_fermionic::ffsim_client::Ffsim;
use emu_compare_fermionic::gate::{
    admit, admit_label, hubbard_analytic, kitaev_analytic, kitaev_sector_refusal, rel_gap,
    sign_flip_detected, spread_flagged, RowLabel, BINDING_ROWS, SIGN_FLIP_FLOOR,
};
use emu_compare_fermionic::workload;
use omega_emu_compare::{Classification, GATE_REL_F64};

#[test]
fn a_speed_classification_on_a_binding_row_is_a_failure() {
    for id in BINDING_ROWS {
        admit_label(id, RowLabel::Capability).unwrap();
        for class in [
            Classification::Win,
            Classification::Loss,
            Classification::Tie,
        ] {
            let err = admit_label(id, RowLabel::Speed(class)).expect_err("speed label");
            assert!(
                err.contains("speed classification"),
                "{id} {class:?}: {err}"
            );
            assert!(
                err.contains("call overhead") || err.contains("binding"),
                "{err}"
            );
        }
    }
    // The mutation: the same function, with the label changed to a win.
    let err = admit_label("ferm-kitaev-n2", RowLabel::Speed(Classification::Win)).unwrap_err();
    assert!(err.contains("ferm-kitaev-n2"), "{err}");
    assert!(err.contains("Win"), "{err}");
}

#[test]
fn lucj_may_be_a_speed_row_and_n64_may_not() {
    admit_label("ferm-lucj16", RowLabel::Speed(Classification::Tie)).unwrap();
    admit_label("ferm-lucj16", RowLabel::Speed(Classification::Loss)).unwrap();
    let err = admit_label("ferm-kitaev-n64", RowLabel::Speed(Classification::Win)).unwrap_err();
    assert!(err.contains("2^64"), "{err}");
    admit_label("ferm-kitaev-n64", RowLabel::Capability).unwrap();
    assert!(!BINDING_ROWS.contains(&"ferm-lucj16"));
    assert!(!BINDING_ROWS.contains(&"ferm-kitaev-n64"));
}

#[test]
fn the_value_gate_is_two_sided() {
    let reference = -1.0;
    admit(reference, reference, reference).unwrap();
    // Inside the tolerance.
    let tiny = reference + 1e-12;
    admit(tiny, tiny, reference).unwrap();
    // Too high. A gate of either sign must fail.
    let high = reference + 1e-8;
    let err = admit(high, reference, reference).expect_err("high side");
    assert!(
        err.reason.contains("disagreement finding"),
        "{}",
        err.reason
    );
    // Too low. `got - reference <= tol` would admit this, because the
    // difference is negative. The absolute gap does not.
    let low = reference - 1e-8;
    let err = admit(low, reference, reference).expect_err("low side");
    assert!(
        err.reason.contains("disagreement finding"),
        "{}",
        err.reason
    );
    assert!(rel_gap(low, reference) > GATE_REL_F64);
    assert!(rel_gap(high, reference) > GATE_REL_F64);
}

#[test]
fn a_silent_sign_flip_is_not_detection() {
    // Equal values: the flip was sent to both arms, or it did nothing.
    let err = sign_flip_detected(-1.0, -1.0).unwrap_err();
    assert!(err.contains("NOT detected"), "{err}");
    // A floor of 0, or of 1e-6, would accept 1e-4. This floor does not.
    let quiet = -1.0 + 1e-4;
    let err = sign_flip_detected(-1.0, quiet).unwrap_err();
    assert!(err.contains("NOT detected"), "{err}");
    assert!(err.contains("unflipped oracle"), "{err}");
    let loud = sign_flip_detected(-1.0, 0.5).unwrap();
    assert!(loud > SIGN_FLIP_FLOOR);
}

#[test]
fn kitaev_refusal_text_has_to_name_particle_number() {
    kitaev_sector_refusal(
        "The given FermionOperator could not be converted because it does not conserve particle number",
    )
    .unwrap();
    let err = kitaev_sector_refusal("unsupported gate").unwrap_err();
    assert!(err.contains("particle number"), "{err}");
}

#[test]
fn spread_over_5x_is_flagged_and_a_modest_spread_is_not() {
    assert!(!spread_flagged(&[1.0, 1.2, 1.4, 2.0, 3.0]).unwrap());
    assert!(spread_flagged(&[1.0, 1.0, 6.0]).unwrap());
}

#[test]
fn closed_forms() {
    assert_eq!(kitaev_analytic(2), -1.0);
    assert_eq!(kitaev_analytic(8), -7.0);
    assert_eq!(kitaev_analytic(64), -63.0);
    let hub = hubbard_analytic();
    assert!((hub - (2.0 - 2.0 * 2.0_f64.sqrt())).abs() < 1e-15);
}

/// A registered skip, not a silent one, mirroring the stim lane: a clean
/// checkout on any box has no `.venv-ffsim`, and panicking there made
/// `cargo test --workspace` red on akilles from 9e3a6f4. `ci.sh` registers
/// the skip and exports `EMU_FFSIM_REQUIRE=1` on a box whose venv imports
/// ffsim, which turns the skip back into a failure where the lane can run.
fn ffsim_or_skip() -> Option<Ffsim> {
    match Ffsim::launch() {
        Ok(f) => Some(f),
        Err(e) => {
            if std::env::var("EMU_FFSIM_REQUIRE").as_deref() == Ok("1") {
                panic!("EMU_FFSIM_REQUIRE=1 but the ffsim arm is unavailable: {e}");
            }
            eprintln!("SKIP: ffsim arm unavailable, so this lane test cannot run: {e}");
            None
        }
    }
}

/// The Kitaev-64 sign flip needs no ffsim — its oracle is closed-form — so it
/// runs on every box rather than riding the venv-gated test below.
#[test]
fn kitaev64_feels_a_seeded_sign_flip_without_ffsim() {
    let n64 = workload::kitaev64(false).unwrap();
    admit(n64.energy, kitaev_analytic(64), kitaev_analytic(64)).unwrap();
    assert!(n64.exact && n64.final_terms == 63);
    let flipped = workload::kitaev64(true).unwrap().energy;
    sign_flip_detected(kitaev_analytic(64), flipped).unwrap();
}

#[test]
fn seeded_sign_flips_miss_the_unflipped_ffsim_value() {
    let Some(mut ffsim) = ffsim_or_skip() else {
        return;
    };

    let h2_theirs = ffsim.h2().unwrap()["value"].as_f64().unwrap();
    let h2_ref = workload::h2_reference().unwrap();
    admit(workload::h2(false).unwrap(), h2_theirs, h2_ref).unwrap();
    let h2_delta = sign_flip_detected(h2_theirs, workload::h2(true).unwrap()).unwrap();
    assert!(h2_delta > SIGN_FLIP_FLOOR);

    let hub_theirs = ffsim.hubbard().unwrap()["value"].as_f64().unwrap();
    admit(
        workload::hubbard(false).unwrap(),
        hub_theirs,
        hubbard_analytic(),
    )
    .unwrap();
    let hub_delta = sign_flip_detected(hub_theirs, workload::hubbard(true).unwrap()).unwrap();
    assert!(hub_delta > SIGN_FLIP_FLOOR);

    for n in [2u32, 8] {
        let got = workload::kitaev_small(n, false).unwrap();
        let body = ffsim.kitaev(n).unwrap();
        let theirs = body["value"].as_f64().unwrap();
        kitaev_sector_refusal(body["sector_refused"].as_str().unwrap()).unwrap();
        admit(got.statevector, theirs, kitaev_analytic(n)).unwrap();
        assert!(got.exact && got.dropped_mass == 0.0);
        let flipped = workload::kitaev_small(n, true).unwrap().statevector;
        // The oracle is the unflipped ffsim value, not a flipped circuit
        // sent back to ffsim.
        sign_flip_detected(theirs, flipped).unwrap();
    }

    let refusal = ffsim.kitaev64().unwrap();
    assert_eq!(refusal["ok"], false);
    assert_eq!(refusal["allocated"], false);
    kitaev_sector_refusal(refusal["error"].as_str().unwrap()).unwrap();

    let lucj = workload::lucj(false).unwrap();
    assert_eq!(lucj.sector_dim, 12870);
    assert_eq!(lucj.electrons, 8);
    let prepared = ffsim.lucj_prepare().unwrap();
    assert_eq!(prepared["dim"], 4900);
    let sample = ffsim.lucj().unwrap();
    let theirs = sample["value"].as_f64().unwrap();
    admit(lucj.sector, theirs, theirs).unwrap();
    assert!(rel_gap(lucj.sector, lucj.statevector) <= GATE_REL_F64);
    let flipped = workload::lucj(true).unwrap();
    sign_flip_detected(theirs, flipped.sector).unwrap();
}

fn ext(before: Option<f64>, during: f64) -> omega_emu_compare::ExternalCpu {
    omega_emu_compare::ExternalCpu {
        external_cores_during_row: during,
        window_s: 2.0,
        method: "test".into(),
        own_pids: vec![1],
        external_cores_before: before,
        before_window_s: before.map(|_| 2.0),
    }
}

/// §4.4a on the capability path: someone else's work before OR during the row
/// voids it; the lane's own load (which load1 cannot tell apart) does not
/// enter, because it is not in these numbers.
#[test]
fn a_capability_row_voids_on_external_work_before_or_during() {
    use emu_compare_fermionic::gate::external_void_reason;
    assert!(external_void_reason("r", &ext(Some(0.4), 0.6), 2.0).is_none());
    assert!(
        external_void_reason("r", &ext(Some(3.0), 0.6), 2.0).is_some(),
        "busy before"
    );
    assert!(
        external_void_reason("r", &ext(Some(0.4), 2.6), 2.0).is_some(),
        "busy during"
    );
    assert!(
        external_void_reason("r", &ext(None, 2.6), 2.0).is_some(),
        "no pre-row census, busy during"
    );
}
