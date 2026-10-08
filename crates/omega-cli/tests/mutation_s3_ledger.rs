// SPDX-License-Identifier: Apache-2.0
//! S3 test: the mutation ledger for the CLI door and the cross-lane contract,
//! and the boundary rows the grid of engine runs cannot produce for itself.
//!
//! Every mutation below was applied to the source, the three S3 test binaries
//! were run with `--no-fail-fast`, and the file was restored and checked
//! byte-for-byte with `cmp`. Baseline: **15 tests, all green** —
//! `stabrank_cli_smoke.rs` (4), `stabrank_majoranaprop_intervals.rs` (3),
//! `stabrank_json_roundtrip.rs` (5) and this file's three, which are pure
//! arithmetic and are not reachable from `src/` (see the last section). The
//! twelve process-driven ones are what the entries below count. Line numbers
//! are as of the commit that
//! added this file. The S1 ledger's method note applies and is worth
//! repeating: every campaign run touches a *source* file before building, so
//! cargo cannot reuse a stale integration-test binary and report a gap that
//! is not there. Here that happens naturally — all four mutations are in
//! `src/`.
//!
//! # What this phase's mutations are about
//!
//! S0 and S1 mutate a kernel and ask whether a number moves. S2 mutates a
//! *claim* and asks whether an unsound bound is caught. S3 mutates neither:
//! the engine is finished and untouched, so what can go wrong here is a
//! **door** and a **document**. Three failure modes, and the ledger is
//! organised by them:
//!
//! * **The door is not there.** Entry A. This is the one the plan names for
//!   test (i) and it fails at an exit code, not at a tolerance.
//! * **The document disagrees with itself.** Entry C. The engine is correct
//!   and the JSON it emits is internally inconsistent — a derived field and
//!   the raw field it is a function of, drifted apart in one document.
//! * **The bound is unsound.** Entry B. Not S3's own code, but S3's
//!   interval test is the first place two independent engines are asked to
//!   agree about the same number, so this ledger measures what that test sees
//!   that S2's did not.
//!
//! And one entry that catches nothing here, on purpose: **D**, the
//! "underived" class, which is the honest answer to "what would these three
//! tests miss".
//!
//! # The ledger
//!
//! **A — `omega-cli/src/main.rs:2388`,** the expectation-mode dispatch arm:
//! `"stabrank" | "sr" => {` → `"stabrank_MUTANT" | "sr" => {`. The wiring
//! absent, in the only way it can be absent once the crate is a dependency:
//! `--backend stabrank` falls through to the `_` arm.
//! Reddened, **9 tests**. The named one is test (i)'s
//! `a_fermionic_observable_string_reaches_stabrank_and_matches_the_statevector`
//! at `stabrank_cli_smoke.rs:154`, and it failed exactly where the plan says
//! it should:
//!
//! ```text
//! omega-run --backend stabrank --expectation-fermionic exited Some(1), not 0.
//! ...
//! Expectation not supported for backend: stabrank
//! ```
//!
//! Also reddened: the other two value-comparing tests in that file, all four
//! stabrank-reading tests of `stabrank_json_roundtrip.rs`, and both
//! engine-reading tests of `stabrank_majoranaprop_intervals.rs`.
//! Stayed green, and each for a reason worth knowing:
//! `sampling_on_stabrank_is_refused_by_name`, because the *execute*-mode arm
//! is a second, separate `match` arm (`main.rs:2929`) and this mutation
//! touches only the expectation one — which is why the two refusals need two
//! tests; `the_majoranaprop_bound_is_saturated_on_the_headline_cell`, which
//! runs majoranaprop only; and
//! `the_block_is_absent_on_other_backends`, which asserts an **absence** —
//! with the arm gone the block is absent everywhere, and an absence assertion
//! cannot tell "absent where it should be" from "absent always". That last
//! one is the shape of a test that can only ever confirm, and it is paired
//! with `missing_keys_are_a_failure_not_a_default` for that reason.
//!
//! **B — `omega-backend-stabrank/src/backend.rs:391`:**
//! `observable_range * m * (2.0 + m)` → `observable_range * m * m`, the
//! factor of 2 dropped. This is the S2 ledger's entry **R** — the
//! Schrödinger-side bound collapsed towards majoranaprop's additive one, the
//! arithmetic form of the §1.2 error — re-measured here against S3's tests
//! rather than S2's.
//! Reddened, 3 tests. The named one is test (ii)'s
//! `the_two_engines_certified_intervals_intersect_and_contain_the_dense_value`
//! at `stabrank_majoranaprop_intervals.rs:333`, and the message is the
//! failure signal this phase exists to produce:
//!
//! ```text
//! UNSOUND BOUND in stabrank --max-chi 63: the certified interval is
//! [0.6670411908874498, 0.6670944592916883] and the dense value is
//! 0.6664213562373094, which is outside it. |v − dense| = 6.464689e-4
//! exceeds the reported bound 2.663420e-5 by 6.198e-4.
//! ```
//!
//! At `m = 3.14e-3` the mutation shrinks the bound by `(2+m)/m ≈ 637×`, which
//! is why containment fails by two orders of magnitude on a cell whose true
//! error is only 3.8% of the honest bound. Also reddened:
//! `the_two_masses_are_not_the_same_quantity` (the `R·(2+m)` factor pin) and
//! test (iii)'s round trip.
//! Stayed green: **all four tests of `stabrank_cli_smoke.rs`.** Every run
//! there is untruncated, so `m = 0` and the bound is `0` under either
//! expression. That is entry A's converse and the measured answer to "what
//! does the smoke test not catch".
//!
//! **C — `omega-cli/src/serialize.rs:447`:**
//! `"state_dropped_mass": cert.state_dropped_mass` →
//! `"state_dropped_mass": cert.state_dropped_mass * 0.5`. The engine is left
//! entirely correct and the **document** is made to contradict itself: the
//! derived bound no longer matches the raw mass printed beside it. This is the
//! defect class the round trip exists for — a stale, rounded or
//! independently-computed copy of an input — and it is invisible to every
//! library test, because none of them reads this document.
//! Reddened, 2 tests. The named one is test (iii)'s
//! `a_reader_rederives_the_bound_from_the_raw_fields_exactly` at
//! `stabrank_json_roundtrip.rs:256`:
//!
//! ```text
//! --max-chi Some("63"): the document reports a bound of 0.016986863648900685
//! where R·m·(2+m) of its own R = 2.7 and m = 0.0015703916154427289 is
//! 0.00848677327392054, a difference of 8.500e-3.
//! ```
//!
//! Also reddened: `the_two_masses_are_not_the_same_quantity`, which pins
//! `bound/m` against `R·(2+m)` and so is a second reader of the same
//! identity.
//! Stayed green, and this is the entry's real content: **test (ii)'s
//! intersection test itself.** It reads `expectation_error_bound` for the
//! interval's half-width and never recomputes it from the mass, so a mass
//! that drifted from the bound passes it untouched — the non-vacuity guard
//! only asks that the mass be positive, and half of a positive number is
//! positive. An interval test constrains the bound; it does not audit the
//! fields the bound was built from. Also green: all of `stabrank_cli_smoke.rs`,
//! whose runs report `m = 0`, and `0.5 × 0` is `0`.
//!
//! **D — `omega-backend-stabrank/src/sum.rs:320`,** in the `max_chi` arm of
//! `StabilizerSum::truncate`: `.map(|(c, _)| c.norm())` →
//! `.map(|(c, _)| 0.5 * c.norm())`, i.e. **the banked mass silently halved**.
//! The S2 ledger's "underived" class: the bound's arithmetic is untouched and
//! its *input* stops meaning what the derivation assumed. `R·m·(2+m)` of the
//! wrong `m` is still `R·m·(2+m)`.
//! Reddened **0 of the 12 process-driven S3 tests.** Every one stayed green. The
//! reported mass halves, the bound is recomputed consistently from the halved
//! mass, and stabrank's interval shrinks from
//! `[0.650081, 0.684055]` to `[0.658581, 0.675555]` — which still contains
//! the dense `0.666421` and still meets majoranaprop's interval, so test (ii)
//! passes; the two reported fields still satisfy the identity, so test (iii)
//! passes; and test (i) never truncates at all.
//! This mutation is **not** vacuous, which is what makes it evidence rather
//! than a story. It reddens two S2 tests:
//! `truncation_witness.rs:349`'s closed-form mass pin —
//!
//! ```text
//! n=4 seed=0 (circuit seed 0, observable 3): m = 0.08102658670431301 where
//! the angle sequence predicts 0.16205317340862602. The cut discarded a
//! different set of branches than the one the bound was reasoned about.
//! ```
//!
//! — and `vacuity_refusal.rs`'s
//! `a_bound_that_excludes_nothing_is_refused_and_not_returned`, whose own
//! mass pin fires. What catches a wrong input is a test that **predicts** the
//! input from the fixture's construction, not one that checks the outputs
//! against each other; S3 has no such test and does not need one, because S2
//! has three.
//!
//! # One mutation each S3 fixture does not catch
//!
//! Not hypotheticals — the entry above that measured it.
//!
//! * `stabrank_cli_smoke.rs`: **B**. All four tests stay green under an
//!   unsound bound, because every run in the file is exact and a bound of
//!   zero is correct under any formula. The file is about the door.
//! * `stabrank_majoranaprop_intervals.rs`: **C**, at its headline test. A
//!   mass that has drifted from the bound it is supposed to generate is
//!   invisible to an interval, which is built from the bound alone. (**D**
//!   likewise, for the same reason one layer down.)
//! * `stabrank_json_roundtrip.rs`: **D**. A consistency pin cannot see a
//!   wrong input, which is the third point of
//!   `omega-backend-stabrank/tests/mutation_s2_ledger.rs`'s entry **S** and
//!   is still true one layer out, in the JSON.
//! * `mutation_s3_ledger.rs`, this file: **all four**. Its rows are pure
//!   arithmetic on constructed intervals and never start a process, so no
//!   mutation in `src/` can reach them. That is deliberate — they exist to
//!   constrain the *predicates* the other three files apply, the way S2's
//!   struct-literal rows constrain `is_informative` — and it is the next
//!   section.
//!
//! # Why this file has rows of its own
//!
//! The grid in test (ii) is made of engine runs, and there are two boundary
//! cases it cannot ask a circuit for.
//!
//! **Two intervals that touch at exactly one point.** The grid's headline
//! cell comes close — majoranaprop's bound is saturated there, so the dense
//! value lands *on* an endpoint — but "the intersection is a single point" is
//! not something a fixture can be made to produce on demand. Written with
//! `hi > lo` instead of `hi >= lo`, the intersection predicate would call a
//! touching pair empty and report a bug in an engine that has none.
//!
//! **Containment that holds with equality.** `|v − dense| ≤ b` is the
//! soundness claim, and `≤` is not `<`: a bound is allowed to be exactly
//! attained, and majoranaprop's *is* attained on the headline cell to a
//! relative `4e-16`. Written with `<`, the containment check would call a
//! perfectly sound saturated bound unsound.
//!
//! This is majoranaprop's own `cargo-mutants` lesson in a new place: five
//! mutants of `is_exact`/`is_informative` survived its whole suite —
//! including `<` replaced by `==` and by `>` — because the only cases
//! asserted were ones where the predicate was comfortably true. The rows
//! below assert the uncomfortable ones.
//!
//! Two of those rows were written wrong on the first attempt and are kept,
//! corrected, with the measurement that corrected them — a ledger that only
//! records the mutations it predicted is the vacuous kind:
//!
//! * The one-ULP unsoundness row could not be expressed at a value of
//!   `0.525`: one ULP of `0.65` is coarser than one ULP of `0.125`, so the
//!   subtraction rounds the deliberate ULP of excess away and the row reads as
//!   sound. It is built at a value of `0.0`, where no cancellation occurs.
//! * `the_state_mass_is_not_a_half_width` first claimed that reading
//!   stabrank's raw mass as a half-width **excludes** the dense value on the
//!   measured cell. False: the error there is `6.46e-4` against a mass of
//!   `3.14e-3`, and across all seventeen informative cuts of this fixture
//!   `|v − dense| < m` on every one, worst ratio `0.322`. The raw mass is
//!   still not a bound — nothing caps the error at `m`, and the derivation
//!   permits `5.41×` it on this observable — but on this data a consumer
//!   reading the wrong field would have got away with it. That is the reason
//!   the rule is a rule and not a convention, and the test now says so with a
//!   constructed row at the permitted error instead of a measured one that
//!   does not exist.

/// An interval `[lo, hi]`, built from a certified `(value, bound)` pair the
/// way both engines' certificates are read.
fn interval(value: f64, bound: f64) -> (f64, f64) {
    (value - bound, value + bound)
}

/// The intersection predicate used by test (ii), isolated so its boundary can
/// be asserted. `>=`, not `>`: two intervals that meet at one point have a
/// non-empty intersection and the exact value may be that point.
fn intersects(a: (f64, f64), b: (f64, f64)) -> bool {
    let (lo, hi) = (a.0.max(b.0), a.1.min(b.1));
    hi >= lo
}

/// The soundness predicate used by test (ii). `<=`, not `<`: a bound may be
/// exactly attained, and on the measured headline cell majoranaprop's is.
fn contains(value: f64, bound: f64, exact: f64) -> bool {
    (value - exact).abs() <= bound
}

/// | case | `hi` vs `lo` | correct | `>` | `==` |
/// |---|---|---|---|---|
/// | overlapping | `hi > lo` | true | true | false |
/// | touching | `hi == lo` | **true** | **false** | true |
/// | disjoint | `hi < lo` | false | false | false |
#[test]
fn intervals_that_touch_at_one_point_do_intersect() {
    // Overlapping, with room: the comfortable case the grid is full of.
    assert!(
        intersects(interval(0.667, 0.017), interval(0.525, 0.141)),
        "the measured headline pair must overlap"
    );

    // Touching: [0.0, 1.0] and [1.0, 2.0]. The exact value could be 1.0, so
    // this is not an empty intersection and must not be reported as a bug in
    // an engine.
    let (a, b) = ((0.0, 1.0), (1.0, 2.0));
    assert!(
        intersects(a, b),
        "intervals {a:?} and {b:?} meet at exactly 1.0. Their intersection is \
         the single point {{1.0}}, which is a perfectly good certified \
         interval and may be where the exact value is. Written with `>` \
         instead of `>=` this row flips and the test reports an unsound bound \
         in an engine that has none."
    );
    assert!(intersects(b, a), "the predicate must be symmetric");

    // Disjoint by one ULP: the smallest real failure, and it must be caught.
    let gap = (1.0f64, f64::from_bits(1.0f64.to_bits() + 1));
    assert!(
        !intersects((0.0, gap.0), (gap.1, 2.0)),
        "an interval ending at {} and one starting at {} are disjoint; a \
         predicate that cannot see a one-ULP gap cannot see an unsound bound \
         either",
        gap.0,
        gap.1
    );
    assert!(
        !intersects((0.0, 0.4), (0.6, 1.0)),
        "disjoint with room to spare"
    );

    // Containment one way round is still an intersection — and is the common
    // case in the measured grid, where stabrank's interval is the narrower.
    assert!(
        intersects((0.3, 0.9), (0.4, 0.5)),
        "one interval inside the other"
    );
}

/// | case | `|v−exact|` vs `b` | correct | `<` |
/// |---|---|---|---|
/// | slack | below | true | true |
/// | saturated | **equal** | **true** | **false** |
/// | unsound | above | false | false |
#[test]
fn a_bound_that_is_exactly_attained_is_still_sound() {
    assert!(
        contains(0.6, 0.1, 0.55),
        "an error of 0.05 against a bound of 0.1 is sound with slack"
    );

    // Saturated, on the nose. This is the headline cell's majoranaprop leg:
    // the dropped monomials were exactly the hopping contribution, so the
    // error equals the bound and the exact value sits on the endpoint.
    assert!(
        contains(0.525, 0.125, 0.65),
        "an error of exactly 0.125 against a bound of exactly 0.125 is SOUND: \
         the claim is |Δ| ≤ b, a bound may be attained, and the measured \
         headline cell attains majoranaprop's to a relative 4e-16. Written \
         with `<` this row flips and a sound saturated bound is reported as a \
         defect."
    );

    // Unsound by one ULP, and it must be caught.
    //
    // Built at a value of 0.0 on purpose. At 0.525 the subtraction
    // `0.525 - 0.650000…` rounds to exactly 0.125 — one ULP of 0.65 is
    // coarser than one ULP of 0.125 — so the row cannot be expressed there at
    // all, which is a fact about f64 and not about the predicate. The first
    // attempt at this row was written at 0.525 and failed for that reason.
    let just_over = f64::from_bits(0.125f64.to_bits() + 1);
    assert!(
        !contains(0.0, 0.125, just_over),
        "an error of {just_over} against a bound of 0.125 is unsound by one \
         ULP and must be reported; a predicate too slack to see that cannot \
         see a real one either"
    );
    assert!(!contains(0.525, 0.125, 1.0), "unsound with room to spare");

    // The sign of the error must not matter: a value below the exact one is
    // as sound or unsound as the same distance above it.
    assert!(
        contains(0.6, 0.1, 0.55) == contains(0.5, 0.1, 0.55),
        "containment must be symmetric in the direction of the error"
    );
}

/// Reading stabrank's raw mass as a half-width is not a bound, and this is
/// the arithmetic that says so — §1.2's claim as a pure function.
///
/// # What this row does NOT claim, and the measurement that stopped it
///
/// The first version of this test asserted that the mass-as-half-width
/// interval **excludes** the dense value on the measured `--max-chi 63` cell.
/// That assertion failed, because it is false: the true error there is
/// `6.46e-4` and the mass is `3.14e-3`, so the too-narrow interval happens to
/// contain the right answer anyway. Sweeping all seventeen informative cuts
/// of this fixture, `|v − dense| < m` on **every one**, the largest ratio
/// being `0.322` at `--max-chi 61`.
///
/// That is worth recording rather than quietly deleting, because it is the
/// precise reason the rule is a rule. The raw mass is not a bound on
/// `|Δ⟨O⟩|` — the derivation does not support it, and nothing caps the error
/// at `m` — but on a given fixture it may well happen to exceed the error, so
/// a consumer who used it would get away with it and conclude they were
/// right. "It contained the answer on my data" is not evidence that a number
/// is a bound. The derivation permits an error of up to `R·(2+m) = 5.41×` the
/// mass on this very observable, and the last row below is that permitted
/// case: a sound run whose error the mass-as-half-width interval excludes.
#[test]
fn the_state_mass_is_not_a_half_width() {
    // Measured, `--max-chi 63` on the S3 fixture; see
    // `stabrank_majoranaprop_intervals.rs`.
    let dense = 0.6664213562373094;
    let value = 0.6670678250895691;
    let mass = 0.0031407832308854577;
    let r = 2.7;
    let bound = r * mass * (2.0 + mass);

    assert!(
        contains(value, bound, dense),
        "the derived bound {bound} must contain the dense value — this is the \
         same cell the interval test certifies"
    );
    // The factor between the two readings is the derivation's, not a
    // constant, and it is what a consumer loses by reading the wrong field.
    let factor = bound / mass;
    assert!(
        (factor - r * (2.0 + mass)).abs() < 1e-12,
        "bound/mass is {factor} where the derivation gives R·(2+m) = {}",
        r * (2.0 + mass)
    );
    assert!(
        factor > 5.0,
        "on this observable the mass-as-half-width interval is {factor}× too \
         narrow"
    );

    // On THIS cell the too-narrow interval still contains the answer, and the
    // row is asserted in that direction so nobody mistakes the reason the
    // rule exists. See the doc above.
    assert!(
        contains(value, mass, dense),
        "the measurement recorded in this test's doc has changed: the raw mass \
         {mass} no longer happens to exceed the true error {:.6e} on this \
         cell. Update the doc — the test's point survives either way, but its \
         stated reason would be stale.",
        (value - dense).abs()
    );

    // The permitted case, constructed: an error of 3m, which the derived bound
    // allows (it permits up to 5.41m) and the mass does not. This is the run
    // on which a consumer reading the wrong field reports a certified
    // interval the true answer is outside of.
    let permitted_error = 3.0 * mass;
    assert!(
        permitted_error < bound,
        "3m = {permitted_error} must be inside the derived bound {bound}, or \
         this row is not a sound run"
    );
    let exact = value - permitted_error;
    assert!(
        contains(value, bound, exact),
        "the derived bound must contain an error it explicitly permits"
    );
    assert!(
        !contains(value, mass, exact),
        "with the raw mass {mass} as the half-width, a run whose error is 3m — \
         which the derivation permits on this observable — reports the interval \
         [{}, {}], and the exact value {exact} is outside it. That is an \
         unsound certificate produced by reading an input to the bound as the \
         bound.",
        value - mass,
        value + mass
    );
}
