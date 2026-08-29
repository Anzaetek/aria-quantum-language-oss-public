// SPDX-License-Identifier: Apache-2.0
//! The squeeze OPERATOR, anchored to the squeeze CONSTRUCTOR.
//!
//! `FockState::squeezed_vacuum` is already gated against piquasso by
//! `piquasso_xcheck.rs`. So `squeeze(r)` applied to a vacuum must reproduce it —
//! which pins the new operator's sign and normalisation to an externally
//! validated reference rather than to anyone's reading of a convention.
//!
//! That chain matters. Checking the operator against my own derivation of the
//! recurrence would be checking one reading of the literature against another
//! reading of the same literature. This repository has shipped exactly that
//! failure: the `Reset` channel was wrong in three backends in three different
//! bases, and every internal cross-backend gate passed because each pair
//! coincided in whatever basis was checked.

use num_complex::Complex64;
use omega_backend_cv::FockState;

/// The anchor. Column 0 of the operator IS the constructor's series.
#[test]
fn squeeze_on_vacuum_reproduces_the_constructor() {
    for &r in &[0.0, 0.1, 0.35, 0.5, 0.8, 1.0, -0.4] {
        let cutoff = 24;
        let want = FockState::squeezed_vacuum(r, cutoff).expect("constructor");
        let mut got = FockState::vacuum(cutoff).expect("vacuum");
        got.squeeze(r).expect("operator");

        let worst = want
            .amplitudes()
            .iter()
            .zip(got.amplitudes())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-12,
            "r={r}: the squeeze OPERATOR on vacuum disagrees with the squeeze \
             CONSTRUCTOR by {worst:.3e}. Column 0 of the operator is the \
             constructor's series by construction, so any difference is a sign \
             or normalisation error in the recurrence."
        );
    }
}

/// `⟨n⟩ = sinh²r` for a squeezed vacuum — an ANALYTIC reference, independent of
/// both our implementations.
///
/// The constructor test above would pass if operator and constructor were
/// identically wrong. This one cannot: `sinh²r` comes from the physics, not from
/// this repository.
#[test]
fn squeezed_vacuum_photon_number_matches_the_analytic_value() {
    for &r in &[0.1, 0.35, 0.5, 0.8] {
        // Cutoff generous enough that truncation is far below the tolerance;
        // <n> converges slowly in r, so this is checked at modest r only.
        let cutoff = 96;
        let mut st = FockState::vacuum(cutoff).expect("vacuum");
        st.squeeze(r).expect("squeeze");
        let want = r.sinh() * r.sinh();
        let got = st.expect_n(1e-6).expect("expect_n");
        assert!(
            (got - want).abs() < 1e-6,
            "r={r}: <n> = {got:.9} but sinh^2(r) = {want:.9}"
        );
    }
}

/// `S(-r)S(r) = I`. A round trip must return the state it started from.
///
/// Catches an error that is even in `r` — which the constructor anchor cannot,
/// since it only ever exercises the operator from vacuum.
#[test]
fn squeeze_then_unsqueeze_is_the_identity() {
    let cutoff = 64;
    let mut st = FockState::coherent(Complex64::new(0.4, -0.2), cutoff).expect("coherent");
    let before: Vec<Complex64> = st.amplitudes().to_vec();
    st.squeeze(0.3).expect("squeeze");
    st.squeeze(-0.3).expect("unsqueeze");
    let worst = before
        .iter()
        .zip(st.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        worst < 1e-9,
        "S(-r)S(r) is not the identity: worst amplitude drift {worst:.3e}"
    );
}

/// **The capability this was written for**: squeezing something that already has
/// structure. Previously impossible — squeeze existed only as a constructor, so
/// it could only ever be the first operation on a mode.
#[test]
fn squeeze_applies_to_a_state_that_already_has_structure() {
    let cutoff = 64;
    let mut st = FockState::coherent(Complex64::new(0.5, 0.0), cutoff).expect("coherent");
    st.squeeze(0.4).expect("squeeze after displace");

    // A squeezed coherent state is normalised, and squeezing a coherent state
    // raises <n> above |alpha|^2. Both are weak checks on their own; together
    // they catch a result that is merely a rescaled input.
    assert!(
        (st.norm_sqr() - 1.0).abs() < 1e-6,
        "norm drifted to {}",
        st.norm_sqr()
    );
    let n = st.expect_n(1e-6).expect("expect_n");
    let alpha_sq = 0.25;
    assert!(
        n > alpha_sq,
        "<n> = {n} did not rise above |alpha|^2 = {alpha_sq} after squeezing"
    );
}

/// A phase is REFUSED, not silently dropped.
///
/// There is no phased entry point at all — `squeeze` takes `f64`, so a caller
/// cannot express `S(r e^{iφ})` and get `S(r)` back by accident. This test
/// records that the absence is deliberate, and will fail to compile if someone
/// widens the signature without also widening the validation.
#[test]
fn the_signature_admits_no_phase() {
    fn assert_real_only(f: fn(&mut FockState, f64) -> Result<(), omega_backend_cv::CvError>) {
        let _ = f;
    }
    assert_real_only(FockState::squeeze);
}

/// Truncation is REPORTED, not hidden. Squeezing pushes population up the
/// ladder, so a tight cutoff must show up in `lost_norm`.
#[test]
fn a_tight_cutoff_is_reported_as_lost_mass() {
    let mut tight = FockState::vacuum(6).expect("vacuum");
    tight.squeeze(1.2).expect("squeeze");
    let mut roomy = FockState::vacuum(200).expect("vacuum");
    roomy.squeeze(1.2).expect("squeeze");

    assert!(
        tight.lost_norm() > 1e-3,
        "cutoff 6 at r=1.2 reported only {:.3e} lost — squeezing at that \
         strength puts real weight above |5>, so a near-zero figure means the \
         spill is not being measured",
        tight.lost_norm()
    );
    assert!(
        roomy.lost_norm() < tight.lost_norm(),
        "a roomier cutoff reported MORE loss ({:.3e}) than a tight one ({:.3e})",
        roomy.lost_norm(),
        tight.lost_norm()
    );
}
