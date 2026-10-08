// SPDX-License-Identifier: Apache-2.0
//! S2 test: the mutation ledger for the bound, and the local pins that keep a
//! defect in it from having to be found three files away.
//!
//! Every mutation below was applied to the source, the whole
//! `omega-backend-stabrank` suite was run with `--no-fail-fast`, and the file
//! was restored and checked byte-for-byte with `cmp`. Baseline: **60 tests,
//! all green**, this file's four included. Line numbers are as of the commit
//! that added this file. The method note from `mutation_s1_ledger.rs` is
//! obeyed and worth repeating: every campaign run touches the sources before
//! building, because cargo will otherwise reuse an integration-test binary
//! and the ledger will report a gap that is not there.
//!
//! # What this phase's mutations are about
//!
//! S0 and S1 mutate a *kernel*: a wrong phase, a wrong angle, a wrong
//! conjugation, and the question is whether some fixture's number moves. S2
//! mutates a *claim*. The bound is not an approximation to anything — it is
//! an assertion that the true error cannot exceed it — so the mutations here
//! split into two kinds, and both have to be caught by different means:
//!
//! * **Unsound**: the bound reports less than the derivation gives, so the
//!   certificate says something false. Entries P, Q and R. A fixture catches
//!   these only if its true error is a decent fraction of its bound, which is
//!   why `truncation_witness.rs` is built around concentrating the dropped
//!   mass in one branch rather than around looking realistic.
//! * **Underived**: the bound's arithmetic is untouched and its *inputs* stop
//!   meaning what the derivation assumed. Entry S — a renormalised truncated
//!   state — is the one the plan names, and it is invisible to every
//!   consistency check in this file, because `R·m·(2+m)` of the wrong `m` is
//!   still `R·m·(2+m)`.
//!
//! # The ledger
//!
//! **P — `backend.rs:391`:** `observable_range * m * (2.0 + m)` →
//! `... / 4.0`, i.e. the bound weakened by a factor of four. The plan's S2
//! mutation (ii).
//! Reddened, 2 tests: `the_instantiation_witness_asserts_all_three_conjuncts_together`
//! at `truncation_witness.rs:380` — conjunct (3) — on the cell that file
//! names `FIRST_SHARP_CELL`, `n = 4` seed 0, where `|Δ| = 1.0355e-1` against
//! a quartered bound of `8.7592e-2`; and this file's
//! `the_reported_bound_is_the_formula_its_doc_states` at line 216.
//! Stayed green: the vacuity gate's own fixture. At `m = 3.0832` the bound is
//! `15.673` and a quarter of it is `3.918`, still far above the ceiling of
//! `1`, so the refusal fires either way. That is the division of labour
//! between the two files: the gate test constrains the *predicate*, the
//! witness constrains the *arithmetic*, and neither is a substitute.
//!
//! **Q — `backend.rs:391`:** the same expression negated,
//! `-(observable_range * m * (2.0 + m))`.
//! Reddened, 3 tests: P's two, plus
//! `a_bound_that_excludes_nothing_is_refused_and_not_returned` at
//! `vacuity_refusal.rs:127`, which is the `Ok` arm — with a negative bound
//! nothing is ever vacuous, so the engine returned `0` for a run that
//! discarded 3.08 of coefficient mass, and the test failed on having been
//! handed a number at all rather than on the number's value.
//! Stayed green: `is_informative_is_false_exactly_when_the_bound_excludes_nothing`
//! and the rest of this file's predicate rows, which are struct literals and
//! do not care what the engine computes — that is what
//! `an_engine_run_agrees_with_the_predicates` is for, and it stays green too,
//! because a negative bound is still consistently below `R + |v|`.
//!
//! **R — `backend.rs:391`:** `observable_range * m * (2.0 + m)` →
//! `observable_range * m * m`, i.e. the factor of 2 dropped. This is the
//! Schrödinger-side bound collapsed to something closer to majoranaprop's
//! additive one, and §1.2 is the paragraph it contradicts.
//! Reddened, 2 tests: the same two as P. At the witness's `m = 0.162` the
//! bound falls from `3.5037e-1` to `2.6261e-2`, below the error at 39 of the
//! 40 cells.
//! Stayed green: the vacuity fixture again, for P's reason — `m² = 9.5` is
//! still vacuous.
//!
//! **S — `sum.rs:322`, the end of `StabilizerSum::truncate`:** three lines
//! added that divide every surviving coefficient by `norm_sqr().sqrt()`,
//! i.e. **the truncated state quietly renormalised**. The deliverable's
//! "no renormalization" rule, violated in the most plausible way a
//! well-meaning edit would violate it.
//! Reddened, 4 tests:
//! `truncation_deletes_branches_and_does_not_rescale_the_survivors`
//! (`sum.rs:710`, the bit-for-bit comparison against the uncut run) and
//! `a_chi_cut_keeps_the_heaviest_branches_and_banks_what_it_dropped`
//! (`sum.rs:679`: the four survivors carry `1.8222` where `c³ + 3c²s` is
//! `1.7688`); the witness at `truncation_witness.rs:349`, its closed-form
//! mass pin, `m = 0.16222` against the predicted `0.16205`; and
//! `a_bound_that_excludes_nothing_is_refused_and_not_returned` at
//! `vacuity_refusal.rs:168`, also a mass pin, and the loudest of the four —
//! `m = 4.5922` where the angle sequence gives `3.0832`, because rescaling
//! restores the coefficient to 1 before each of the twelve cuts so each one
//! discards a full `sin(π/8)`.
//!
//! Three things about S are worth stating, because they are the shape of this
//! whole class of defect.
//!
//! First, **the witness's conjunct (3) does not catch it.** Renormalising
//! moves the value *towards* the exact one — at the named cell, `0.72855`
//! divided by `‖ψ′‖ = √0.7285` lands within `1e-4` of the dense `1.0` — so
//! the measured error shrinks and stays inside the bound. An unsound
//! certificate that returns better numbers is exactly the kind this project's
//! contract is written against, and "the answer got closer" is not evidence.
//! What catches it is that the bound was derived for a state this is not.
//!
//! Second, **the closed-form mass pins are what catch it**, in three places,
//! and they catch it because they predict `m` from the angle sequence instead
//! of accepting whatever the engine banked. A test that only asserted
//! `state_dropped_mass > 0` would be green under S.
//!
//! Third, **`the_reported_bound_is_the_formula_its_doc_states` stays green**,
//! necessarily. It checks that the derived field is `R·m·(2+m)` of the
//! reported `R` and `m`; S changes `m` and leaves the formula alone. A
//! consistency pin cannot see a wrong input, and this is the entry that
//! measures that rather than assuming it.
//!
//! # One mutation each S2 fixture does not catch
//!
//! Not a hypothetical per fixture — the entry above that measured it.
//!
//! * `truncation_witness.rs`: **S**, at conjunct (3). Its mass pin is what
//!   reddens, not its error comparison, for the reason above. It is also
//!   blind to everything the S1 ledger's entry **I** covers — a scalar
//!   `StabilizerSum::clifford` applies to every branch alike is a global
//!   phase, and no expectation fixture at any `χ` can see one.
//! * `vacuity_refusal.rs`: **P** and **R**. Its `m` is 3.08, where the bound
//!   is vacuous under every weakening in this ledger and under plain `m` too.
//!   It constrains the gate, not the arithmetic.
//! * `mutation_s2_ledger.rs`, this file: **S**, and that is the third point
//!   above. Its predicate rows are also blind to P, Q and R, being struct
//!   literals; only its formula pin sees those.
//! * The whole S2 suite: a defect in `observable_l1_norm` that scales `R`.
//!   `R` multiplies the bound *and* sets the vacuity ceiling `R + |v|`, so a
//!   uniformly scaled `R` moves both sides of the gate together, and the
//!   witness's conjunct (3) has `R`-sized slack to absorb the rest. The S1
//!   ledger's entry **M** is where that field is pinned, by
//!   `three_way_fermionic.rs` comparing it against majoranaprop's own.

use omega_backend_stabrank::{SeedBasis, StabRankBackend, StabRankCertificate};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;

/// A certificate with only the fields the predicates read set meaningfully.
/// The rest is named explicitly rather than filled by a `Default`, so a field
/// that later joins the predicates' inputs cannot do it silently.
fn cert(expectation_error_bound: f64, observable_range: f64, value: f64) -> StabRankCertificate {
    StabRankCertificate {
        seed_basis: SeedBasis::Pauli,
        state_dropped_mass: 0.0,
        expectation_error_bound,
        final_chi: 0,
        peak_chi: 0,
        coeff_min: 0.0,
        max_chi: None,
        observable_range,
        max_branches: 0,
        truncated_norm_sqr: 1.0,
        value,
    }
}

fn gop(kind: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// Eight branches from three `T` gates, read out through a two-wire string.
/// A cut at five truncates; the same circuit with no cut does not.
fn fixture() -> (CircuitIR, Observable) {
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    for q in 0..3 {
        c.add_op(gop(GateKind::H, &[q], &[]));
    }
    for i in 0..3u32 {
        c.add_op(gop(GateKind::T, &[i], &[]));
        c.add_op(gop(GateKind::CX, &[i, (i + 1) % 3], &[]));
        c.add_op(gop(GateKind::H, &[(i + 2) % 3], &[]));
    }
    let obs = Observable {
        terms: vec![
            (0.75, vec![(0, PauliOp::Z), (1, PauliOp::X)]),
            (-0.5, vec![(2, PauliOp::Y)]),
        ],
    };
    (c, obs)
}

/// `expectation_error_bound` against `R·m·(2+m)` recomputed from the two raw
/// fields it is a function of, on a real truncated run.
///
/// This is the pin that makes a mutation in the formula *local*: without it a
/// wrong bound shows up as a conjunct failing in
/// `tests/truncation_witness.rs`, which is the right place to learn that the
/// bound is unsound and the wrong place to learn which expression is wrong.
///
/// It is also the invariant S3's JSON round-trip will rest on: a reader who
/// has `state_dropped_mass` and `observable_range` can re-derive the bound,
/// so the derived number can never drift from the raw one.
#[test]
fn the_reported_bound_is_the_formula_its_doc_states() {
    let (circuit, obs) = fixture();
    let params = ParameterBinding::new();
    let mut truncating_runs = 0usize;

    // Four cuts of the eight-branch sum, all of them still informative: at
    // `R = 1.25`, a cut to four discards `m = 0.462` and a bound of `1.42`,
    // which the vacuity gate refuses and `vacuity_refusal.rs` is about.
    for max_chi in [None, Some(7), Some(6), Some(5)] {
        let (_, cert) = StabRankBackend::with_truncation(0.0, max_chi)
            .expectation_with_certificate(&circuit, &params, &obs)
            .unwrap_or_else(|e| panic!("max_chi {max_chi:?}: {e}"));
        let m = cert.state_dropped_mass;
        let want = cert.observable_range * m * (2.0 + m);
        assert_eq!(
            cert.expectation_error_bound, want,
            "max_chi {max_chi:?}: the certificate reports a bound of {} where \
             R·m·(2+m) with its own R = {} and m = {m} is {want}. The derived \
             field has drifted from the raw ones it is documented as a \
             function of.",
            cert.expectation_error_bound, cert.observable_range
        );
        if m > 0.0 {
            truncating_runs += 1;
            assert!(
                cert.expectation_error_bound > 0.0,
                "max_chi {max_chi:?}: a positive mass gave a bound of {}",
                cert.expectation_error_bound
            );
        } else {
            assert_eq!(cert.expectation_error_bound, 0.0);
        }
    }
    assert!(
        truncating_runs >= 2,
        "only {truncating_runs} of the four settings truncated, so this test \
         mostly compared zero against zero"
    );
}

/// `expectation_error_bound < R + |v|` — strictly below, exactly at, and
/// strictly above.
///
/// The boundary rows are the point, and the reason they are constructed
/// certificates rather than engine runs is that `b == R + |v|` on the nose is
/// not something a circuit can be asked for. majoranaprop learned this from
/// `cargo-mutants`: five mutants of `is_exact`/`is_informative` survived its
/// whole suite, including `<` replaced by `==` and by `>`, because the only
/// cases asserted were ones where the predicate is true.
///
/// | case | `b` vs `R+|v|` | correct | `<=` | `==` | `>` |
/// |---|---|---|---|---|---|---|
/// | below | `b < R+|v|` | true | true | false | false |
/// | at | `b == R+|v|` | **false** | true | true | false |
/// | above | `b > R+|v|` | false | false | false | true |
#[test]
fn is_informative_is_false_exactly_when_the_bound_excludes_nothing() {
    // R + |v| = 1.5 in every row below, so only the bound moves.
    let (r, v) = (1.0, -0.5);

    assert!(
        cert(1.4, r, v).is_informative(),
        "b = 1.4 < R+|v| = 1.5: the interval [v-b, v+b] does not cover [-R, R], \
         so the bound still excludes something"
    );
    assert!(
        !cert(1.5, r, v).is_informative(),
        "b = 1.5 == R+|v|: [v-b, v+b] exactly covers [-R, R]. Every value the \
         observable can take is consistent with this result, so it is vacuous. \
         This row is what makes `<` different from `<=`"
    );
    assert!(
        !cert(1.6, r, v).is_informative(),
        "b = 1.6 > R+|v|: vacuous with room to spare"
    );
    // |value|, not value: a negative value must widen the threshold, not
    // narrow it. With `value` used signed this row flips.
    assert!(
        cert(1.4, 1.0, 0.5).is_informative() && cert(1.4, 1.0, -0.5).is_informative(),
        "the sign of the value must not change the verdict"
    );
}

/// `state_dropped_mass == 0.0` — a zero test, not a tolerance.
#[test]
fn is_exact_is_false_as_soon_as_anything_was_dropped() {
    let mut c = cert(0.0, 1.0, 0.25);
    assert!(c.is_exact(), "nothing dropped");
    c.state_dropped_mass = 1e-18;
    assert!(
        !c.is_exact(),
        "a dropped mass of 1e-18 is still not exact — the caller decides what \
         is negligible and the certificate only reports"
    );
    // Independent predicates: an exact run is trivially informative, a
    // truncated one can be either.
    assert!(cert(0.0, 1.0, 0.25).is_informative());
}

/// **Guard the guard.** The rows above are struct literals, and would still
/// pass if the engine stopped populating these fields the way the predicates
/// expect. This runs a real truncated propagation and checks the same two
/// functions against the numbers it actually produced.
#[test]
fn an_engine_run_agrees_with_the_predicates() {
    let (circuit, obs) = fixture();
    let params = ParameterBinding::new();

    let (_, exact) = StabRankBackend::new()
        .expectation_with_certificate(&circuit, &params, &obs)
        .expect("the exact run is small");
    assert_eq!(exact.state_dropped_mass, 0.0);
    assert!(exact.is_exact(), "zero mass but is_exact false");
    assert!(exact.is_informative(), "an exact run has b = 0 < R + |v|");
    assert_eq!(
        exact.is_informative(),
        exact.expectation_error_bound < exact.observable_range + exact.value.abs(),
        "is_informative drifted from the condition its doc states"
    );
    assert_eq!(exact.is_exact(), exact.state_dropped_mass == 0.0);

    let (value, cut) = StabRankBackend::with_truncation(0.0, Some(5))
        .expectation_with_certificate(&circuit, &params, &obs)
        .expect("this cut is still informative");
    assert!(cut.state_dropped_mass > 0.0, "the cut dropped nothing");
    assert!(!cut.is_exact(), "positive mass but is_exact true");
    assert_eq!(
        cut.is_informative(),
        cut.expectation_error_bound < cut.observable_range + cut.value.abs(),
        "is_informative drifted on a truncated run"
    );
    // The predicates read `cert.value`, and the caller reads the returned
    // one. They have to be the same number or the gate was applied to a
    // different result than the one handed back.
    assert_eq!(
        value, cut.value,
        "the certificate's value is not the returned one"
    );
}
