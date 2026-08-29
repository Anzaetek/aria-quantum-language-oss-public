// SPDX-License-Identifier: Apache-2.0
//! The beamsplitter — the first operation that couples two modes.
//!
//! Before this, `MultiFockState` had `vacuum`, `displace`, `apply_single_mode`
//! and `expect_n`: a product basis with no way to entangle across it. Every
//! multi-mode state was independent modes that could never interact, so the
//! type could not express the canonical two-mode experiment at all.
//!
//! # Hong–Ou–Mandel is the test that matters
//!
//! Two indistinguishable photons entering a 50:50 beamsplitter on opposite
//! ports **never leave by opposite ports**: the `|1,1⟩ → |1,1⟩` amplitude
//! cancels exactly, and both photons exit together. `P(1,1) = 0` is not
//! approximately zero, it is zero by interference.
//!
//! That makes it a reference of a kind this crate mostly lacks: it comes from
//! the physics, not from piquasso and not from us. A test against another
//! implementation can only say the two agree; this one says the answer is
//! right. And it is a sharp test rather than a smoke test — a sign error in the
//! generator, a wrong √ factor, or a transposed block all break the
//! cancellation and leave visible weight on `|1,1⟩`.

use num_complex::Complex64;
use omega_backend_cv::multimode::MultiFockState;

const BUDGET: Option<usize> = Some(64 << 20);

fn fock(cutoff: usize, n_modes: usize, occ: &[usize]) -> MultiFockState {
    MultiFockState::number_state(cutoff, n_modes, occ, BUDGET).expect("number state")
}

fn prob(st: &MultiFockState, occ: &[usize]) -> f64 {
    let i = st.index_of(occ).expect("occupation in range");
    st.amplitudes()[i].norm_sqr()
}

/// **The Hong–Ou–Mandel dip.** `P(1,1) = 0` exactly, at 50:50.
#[test]
fn hong_ou_mandel_two_photons_never_exit_by_opposite_ports() {
    let mut st = fock(6, 2, &[1, 1]);
    st.beamsplitter(0, 1, std::f64::consts::FRAC_PI_4, 0.0)
        .expect("beamsplitter");

    let p11 = prob(&st, &[1, 1]);
    let p20 = prob(&st, &[2, 0]);
    let p02 = prob(&st, &[0, 2]);

    assert!(
        p11 < 1e-24,
        "HOM dip did not form: P(1,1) = {p11:.3e}, expected 0. The |1,1> \
         amplitude cancels by interference, so any visible weight here is a \
         sign error in the generator, a wrong sqrt factor, or a transposed \
         block — not truncation, which cannot reach a 2-photon block at \
         cutoff 6."
    );
    // The photons leave TOGETHER, half the time by each port.
    assert!((p20 - 0.5).abs() < 1e-12, "P(2,0) = {p20}, expected 0.5");
    assert!((p02 - 0.5).abs() < 1e-12, "P(0,2) = {p02}, expected 0.5");
    assert!(
        (p20 + p02 + p11 - 1.0).abs() < 1e-12,
        "probability is not conserved"
    );
}

/// The dip is specific to 50:50 — it must NOT appear at other angles.
///
/// Without this, a beamsplitter that always emptied `|1,1⟩` would pass the test
/// above. That is not a hypothetical failure mode: zeroing an amplitude is a
/// much easier bug to write than cancelling one.
#[test]
fn the_dip_is_specific_to_a_balanced_splitter() {
    for &theta in &[0.15, 0.5, 1.2] {
        let mut st = fock(6, 2, &[1, 1]);
        st.beamsplitter(0, 1, theta, 0.0).expect("beamsplitter");
        let p11 = prob(&st, &[1, 1]);
        // |<1,1|BS|1,1>|^2 = cos^2(2 theta) for this convention.
        let want = (2.0 * theta).cos().powi(2);
        assert!(
            (p11 - want).abs() < 1e-12,
            "theta={theta}: P(1,1) = {p11:.12} but cos^2(2 theta) = {want:.12}"
        );
        assert!(
            p11 > 1e-6,
            "theta={theta} is not balanced, so P(1,1) must NOT vanish; got {p11:.3e}"
        );
    }
}

/// A single photon splits by `sin²θ / cos²θ`, and total probability is kept.
#[test]
fn one_photon_splits_by_the_angle() {
    for &theta in &[0.0, 0.3, std::f64::consts::FRAC_PI_4, 1.1] {
        let mut st = fock(6, 2, &[1, 0]);
        st.beamsplitter(0, 1, theta, 0.0).expect("beamsplitter");
        let p10 = prob(&st, &[1, 0]);
        let p01 = prob(&st, &[0, 1]);
        assert!(
            (p10 - theta.cos().powi(2)).abs() < 1e-12,
            "theta={theta}: P(1,0) = {p10}, expected cos^2 = {}",
            theta.cos().powi(2)
        );
        assert!(
            (p01 - theta.sin().powi(2)).abs() < 1e-12,
            "theta={theta}: P(0,1) = {p01}, expected sin^2 = {}",
            theta.sin().powi(2)
        );
    }
}

/// `BS(-θ) BS(θ) = I`. Catches an error even in θ, which the angle tests above
/// cannot.
#[test]
fn a_beamsplitter_followed_by_its_inverse_is_the_identity() {
    let mut st = fock(7, 3, &[2, 1, 1]);
    let before: Vec<Complex64> = st.amplitudes().to_vec();
    st.beamsplitter(0, 2, 0.63, 0.41).expect("bs");
    // Genuinely changed something, or the round trip proves nothing.
    let moved = before
        .iter()
        .zip(st.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        moved > 0.1,
        "the beamsplitter did nothing: max move {moved:.3e}"
    );

    st.beamsplitter(0, 2, -0.63, 0.41).expect("inverse bs");
    let worst = before
        .iter()
        .zip(st.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        worst < 1e-12,
        "BS(-theta)BS(theta) is not the identity: worst drift {worst:.3e}"
    );
}

/// Modes NOT named by the beamsplitter must be untouched, and a spectator
/// carrying photons must stay exactly where it was.
#[test]
fn spectator_modes_are_untouched() {
    let mut st = fock(6, 3, &[1, 1, 3]);
    st.beamsplitter(0, 1, 0.7, 0.0).expect("bs");
    // Every surviving amplitude must still have 3 photons in mode 2.
    for (i, a) in st.amplitudes().iter().enumerate() {
        if a.norm_sqr() > 1e-18 {
            let occ = st.occupation_of(i);
            assert_eq!(
                occ[2], 3,
                "amplitude at {occ:?} has weight but mode 2 is no longer 3"
            );
        }
    }
}

/// Truncation is REPORTED. A block with `N >= cutoff` extends past the
/// representable corner, and the mass that lands there must show up in
/// `lost_norm` rather than vanishing silently.
#[test]
fn spill_past_the_cutoff_is_reported() {
    // |3,0> at cutoff 4: the N=3 block spans |3,0>..|0,3>, all representable.
    let mut ok = fock(4, 2, &[3, 0]);
    ok.beamsplitter(0, 1, 0.6, 0.0).expect("bs");
    assert!(
        ok.lost_norm() < 1e-15,
        "N=3 fits entirely under cutoff 4, so nothing should be lost; got {:.3e}",
        ok.lost_norm()
    );

    // |3,3> at cutoff 4: N=6 spans |6,0>..|0,6>, and only |3,3| .. is
    // representable — most of the block is outside.
    let mut lossy = fock(4, 2, &[3, 3]);
    lossy.beamsplitter(0, 1, 0.6, 0.0).expect("bs");
    assert!(
        lossy.lost_norm() > 1e-3,
        "N=6 at cutoff 4 puts most of the block outside the representable \
         corner, so the spill must be reported; got {:.3e}",
        lossy.lost_norm()
    );
}

/// Degenerate and out-of-range operands are refused.
#[test]
fn bad_operands_are_refused() {
    let mut st = fock(4, 2, &[1, 0]);
    assert!(st.beamsplitter(0, 0, 0.5, 0.0).is_err(), "same mode twice");
    assert!(
        st.beamsplitter(0, 9, 0.5, 0.0).is_err(),
        "mode out of range"
    );
    assert!(st.beamsplitter(0, 1, f64::NAN, 0.0).is_err(), "NaN theta");
    assert!(
        st.beamsplitter(0, 1, 0.5, f64::INFINITY).is_err(),
        "inf phi"
    );
}
