// SPDX-License-Identifier: Apache-2.0
//! **The certificate's two predicates, pinned at their boundaries.**
//!
//! Found by `cargo-mutants` (plan §A2), and the finding is §A3 on top of it:
//! `omega-backend-majoranaprop` was forked structurally from
//! `omega-backend-pauliprop`, inherited `is_exact` / `is_informative`, and did
//! **not** inherit their tests. pauliprop asserts `is_informative` in both
//! polarities (`truncation_gate.rs:68` true, `:95` false); majoranaprop
//! asserted neither, and asserted `is_exact` only where it is *true*.
//!
//! Five mutants survived the whole suite:
//!
//! ```text
//!   engine.rs:81:9  replace is_exact       -> bool with true
//!   engine.rs:88:9  replace is_informative -> bool with true
//!   engine.rs:88:9  replace is_informative -> bool with false
//!   engine.rs:88:27 replace < with == in is_informative
//!   engine.rs:88:27 replace < with >  in is_informative
//! ```
//!
//! `is_informative` is not a detail: it is the vacuity gate, the thing that
//! separates "this bound excludes something" from "this bound is correct and
//! says nothing". Every claim made from a majoranaprop run rests on it, and
//! nothing could have told us it was wrong.
//!
//! # Why the boundary, and why constructed certificates
//!
//! The predicate is `dropped_mass < observable_range + |value|`. A single
//! informative case kills the `-> false` mutant and nothing else. To kill the
//! comparison mutants the cases must straddle the boundary **exactly**:
//!
//! | case | `m` vs `R+|v|` | correct | `<=` | `==` | `>` |
//! |---|---|---|---|---|---|---|
//! | below | `m < R+|v|` | true | true | false | false |
//! | at | `m == R+|v|` | **false** | true | true | false |
//! | above | `m > R+|v|` | false | false | false | true |
//!
//! Hitting `m == R+|v|` on the nose through the engine is not something a
//! circuit can be asked for, so these are constructed certificates — the
//! predicate is pure, and its inputs are `pub`. `an_engine_run_agrees_with_the_predicates`
//! keeps that honest by checking the same functions on a real propagation, so
//! the struct literals cannot drift from what the engine actually produces.

use omega_backend_majoranaprop::{MajoranaPropBackend, MajoranaPropCertificate};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;

/// A certificate with the two fields the predicates read set explicitly.
/// Everything else is inert filler — named here rather than `..Default::default()`
/// so a future field cannot silently join the predicate's inputs.
fn cert(dropped_mass: f64, observable_range: f64, value: f64) -> MajoranaPropCertificate {
    MajoranaPropCertificate {
        dropped_mass,
        observable_range,
        value,
        final_terms: 0,
        peak_terms: 0,
        coeff_min: 0.0,
        max_length: None,
        max_terms: None,
        branching_gates: 0,
        apriori_mse_ratio: None,
        seed_basis: omega_backend_majoranaprop::engine::SeedBasis::Pauli,
    }
}

/// `m < R + |v|` — strictly below, exactly at, and strictly above.
#[test]
fn is_informative_is_false_exactly_when_the_bound_excludes_nothing() {
    // R + |v| = 1.5 in every row, so only `m` moves.
    let (r, v) = (1.0, -0.5);

    assert!(
        cert(1.4, r, v).is_informative(),
        "m = 1.4 < R+|v| = 1.5: the interval [v-m, v+m] does not cover [-R, R], \
         so the bound still excludes something"
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

/// `apriori_mse_ratio >= 1.0`, and `None` when there is no length cut.
#[test]
fn apriori_is_vacuous_tracks_the_ratio_at_one() {
    let mut c = cert(0.0, 1.0, 0.0);
    assert_eq!(c.apriori_is_vacuous(), None, "no length cut ⇒ no prior");

    c.apriori_mse_ratio = Some(0.999);
    assert_eq!(c.apriori_is_vacuous(), Some(false));
    c.apriori_mse_ratio = Some(1.0);
    assert_eq!(
        c.apriori_is_vacuous(),
        Some(true),
        "a ratio of exactly 1 bounds the error by the whole range: vacuous"
    );
    c.apriori_mse_ratio = Some(1.001);
    assert_eq!(c.apriori_is_vacuous(), Some(true));
}

/// **Guard the guard.** The rows above are struct literals; if the engine
/// stopped populating these fields the way the predicates expect, they would
/// still pass. This runs a real propagation and checks the same two functions
/// against the numbers it actually produced.
#[test]
fn an_engine_run_agrees_with_the_predicates() {
    let mut c = CircuitIR::new(4, CircuitType::GateBased);
    let op = |k: GateKind, q: &[u32], p: &[f64]| GateOp {
        gate: k,
        qubits: q.iter().map(|&x| Qubit(x)).collect(),
        params: p.iter().map(|&x| ParamExpr::Concrete(x)).collect(),
        classical_bit: None,
        condition: None,
    };
    c.add_op(op(GateKind::X, &[0], &[]));
    c.add_op(op(GateKind::X, &[1], &[]));
    for (a, b) in [(0u32, 1u32), (1, 2), (2, 3)] {
        c.add_op(op(GateKind::Rbs, &[a, b], &[0.7]));
    }
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };
    let params = ParameterBinding::new();

    // Givens-only at a length-2 cut: exact by Thm 2(1), so both predicates
    // must agree that nothing was lost.
    let (_, exact) = MajoranaPropBackend::with_truncation(0.0, Some(2))
        .expectation_with_certificate(&c, &params, &obs)
        .unwrap();
    assert_eq!(exact.dropped_mass, 0.0);
    assert!(
        exact.is_exact(),
        "engine reported dropped_mass 0 but is_exact false"
    );
    assert!(
        exact.is_informative(),
        "an exact run is always informative: m = 0 < R + |v|"
    );

    // The predicates are functions of the reported fields, not of hidden
    // state: recomputing the condition by hand must match.
    assert_eq!(
        exact.is_informative(),
        exact.dropped_mass < exact.observable_range + exact.value.abs(),
        "is_informative drifted from the condition its doc states"
    );
    assert_eq!(exact.is_exact(), exact.dropped_mass == 0.0);
}
