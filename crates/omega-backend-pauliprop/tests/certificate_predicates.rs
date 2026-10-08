// SPDX-License-Identifier: Apache-2.0
//! **The certificate's two predicates, pinned at their boundaries.**
//!
//! The mirror of `omega-backend-majoranaprop/tests/certificate_predicates.rs`,
//! travelling back the way the code came (plan §A2, §A3): majoranaprop was
//! forked from this crate and got the boundary tests first, because its
//! survivors were louder. This crate asserts `is_informative` in both
//! polarities through the engine (`truncation_gate.rs`), and that turned out
//! to be enough for every comparison mutant but one:
//!
//! ```text
//!   sim.rs:253:9   replace is_exact -> bool with true
//!   sim.rs:266:9   replace + with - in is_informative
//! ```
//!
//! `is_exact -> true` survived for the reason it survived one crate over: the
//! suite asserted it only where it is true. `+ -> -` is subtler. The predicate
//! is `m < R + |v|`, and `+` and `-` disagree exactly and only when `m` lies
//! inside the band `(R - |v|, R + |v|)`. That the mutant survived is the proof
//! that no engine case in the suite put `m` there — and it is not a corner:
//! that band is where a heavily truncated run of a large-`|v|` observable
//! lands, the case the vacuity gate exists for.
//!
//! Constructed certificates, because the predicate is pure and its inputs are
//! `pub`; `truncation_gate.rs` checks the same predicates on real runs, so the
//! literals cannot drift from what the engine produces.

use omega_backend_pauliprop::PauliPropCertificate;

/// A certificate with the two fields the predicates read set explicitly.
/// Everything else is inert filler — named here rather than `..Default::default()`
/// so a future field cannot silently join the predicate's inputs.
fn cert(dropped_mass: f64, observable_range: f64, value: f64) -> PauliPropCertificate {
    PauliPropCertificate {
        dropped_mass,
        observable_range,
        value,
        final_terms: 0,
        peak_terms: 0,
        coeff_min: 0.0,
        max_weight: None,
        max_freq: None,
        max_terms: None,
    }
}

/// `m < R + |v|` — strictly below, exactly at, strictly above; and the "below"
/// row sits inside the band where `+` and `-` disagree.
#[test]
fn is_informative_is_false_exactly_when_the_bound_excludes_nothing() {
    // R + |v| = 1.5 in every row, so only `m` moves.
    let (r, v) = (1.0, -0.5);

    assert!(
        cert(1.4, r, v).is_informative(),
        "m = 1.4 < R+|v| = 1.5: the interval [v-m, v+m] does not cover [-R, R], \
         so the bound still excludes something. This row is also inside \
         (R-|v|, R+|v|) = (0.5, 1.5): with `-` for `+` it reads 1.4 < 0.5 and \
         an informative run is reported vacuous"
    );
    assert!(
        !cert(1.5, r, v).is_informative(),
        "m = 1.5 == R+|v|: [v-m, v+m] exactly covers [-R, R]. Every value the \
         observable can take is consistent with this result, so it is vacuous. \
         This row is what makes `<` different from `<=`"
    );
    assert!(
        !cert(1.6, r, v).is_informative(),
        "m = 1.6 > R+|v|: vacuous with room to spare"
    );

    // |value|, not value: a negative value must widen the threshold, not
    // narrow it. With `value` used unsigned this row flips.
    assert!(
        cert(1.4, 1.0, 0.5).is_informative() && cert(1.4, 1.0, -0.5).is_informative(),
        "the sign of the value must not change the verdict"
    );
}

/// `dropped_mass == 0.0`. The suite asserted this only where it is true, which
/// is why `-> true` survived: nothing showed it could be false.
#[test]
fn is_exact_is_false_as_soon_as_anything_was_dropped() {
    assert!(cert(0.0, 1.0, 0.25).is_exact(), "nothing dropped");
    assert!(
        !cert(1e-18, 1.0, 0.25).is_exact(),
        "a dropped mass of 1e-18 is still not exact — `is_exact` is a \
         zero test, not a tolerance, because the caller decides what is \
         negligible and the certificate only reports"
    );
    // Exactness and informativeness are independent: an exact run is trivially
    // informative, but a truncated run can be either.
    assert!(cert(0.0, 1.0, 0.25).is_informative());
}
