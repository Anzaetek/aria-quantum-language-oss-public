// SPDX-License-Identifier: Apache-2.0
//! S3 test (ii): **the cross-lane contract.** One job, run truncated on
//! majoranaprop and truncated on stabrank; the two certified intervals must
//! overlap, and both must contain the dense value.
//!
//! # What the contract is, and what it is not
//!
//! PLAN-MAJORANA-STIM.md §1.2, *What a combined certificate means*: when both
//! engines run the same job, each yields a certified interval `[v − b, v + b]`.
//! Both are sound, so the intersection is sound and tighter; **an empty
//! intersection is a bug in one engine**. That is the whole of what the two
//! lanes may do together, and this file is where it is checked.
//!
//! What they may **not** do is pool their dropped-mass fields. The two
//! numbers are different quantities in different pictures:
//!
//! * majoranaprop is Heisenberg and truncates the **observable**. Every
//!   Majorana monomial has `|⟨M⟩| ≤ 1`, so the discarded L1 mass bounds the
//!   error in `⟨O⟩` additively and *by itself*: `b = dropped_mass`. The field
//!   **is** the bound.
//! * stabrank is Schrödinger and truncates the **state**. A state enters an
//!   expectation twice — `⟨ψ|O|ψ⟩` is bilinear in it — so the same kind of L1
//!   mass `m` enters quadratically and scaled by the observable's range:
//!   `b = R·m·(2+m)`. The field is an **input** to the bound.
//!
//! There is no arithmetic that combines an observable-side mass with a
//! state-side mass, which is why the two certificates use different key
//! names and why nothing in this project feeds one engine's truncated output
//! into the other. `the_two_masses_are_not_the_same_quantity` asserts the
//! distinction on measured numbers rather than leaving it to prose: on the
//! headline cell majoranaprop's bound equals its mass to the last bit, while
//! stabrank's is `5.41×` its mass, and the two masses differ by a factor of
//! 45 on one and the same job.
//!
//! # Why this test is not vacuous
//!
//! A run that dropped nothing has a zero-width interval, contains the dense
//! value trivially, and overlaps everything. So every cell of the grid below
//! asserts, before anything else, that **both** engines truncated
//! (`state_dropped_mass > 0` and `dropped_mass > 0`, by a margin, not by
//! rounding) and that **both** moved the answer (`|v − dense| > 0`). The
//! grid was chosen by measurement for exactly that: four `--max-chi` cuts on
//! stabrank and three cuts on majoranaprop, all seven of which truncate, stay
//! informative, and return a value that is not the exact one.
//!
//! # The measured headline cell
//!
//! `--max-chi 63` against `--max-length 6`, on the fixture below, four
//! qubits, `χ = 64`:
//!
//! ```text
//! dense (statevector)      0.6664213562373094
//! stabrank      value      0.6670678250895691
//!               m          0.0031407832308854577   R = 2.7
//!               bound      0.016986863648900685  = R·m·(2+m)
//!               interval  [0.6500809614406684, 0.6840546887384698]  width 0.03397
//! majoranaprop  value      0.5249999999999999
//!               dropped    0.1414213562373095
//!               interval  [0.3835786437626904, 0.6664213562373094]  width 0.28284
//! intersection            [0.6500809614406684, 0.6664213562373094]  width 0.01634
//! ```
//!
//! Three things in that table are worth reading twice.
//!
//! **The intersection is strictly narrower than both inputs** — 0.01634
//! against 0.03397 and 0.28284 — so §1.2's "sound and tighter" is a measured
//! fact here and not an aspiration. It is a genuine two-sided cut: the lower
//! end comes from stabrank and the upper end from majoranaprop.
//!
//! **majoranaprop's bound is saturated.** `|v − dense| = 0.14142135623730945`
//! against a bound of `0.1414213562373095`: a ratio of
//! `0.9999999999999996`. The dropped Majorana monomials were exactly the
//! hopping contribution, so the error hits the bound to the last bit and the
//! dense value sits *on* the interval's upper endpoint. That is the sharpest
//! possible cell and it is why containment here is evidence: there is no
//! slack for an unsound bound to hide in on that side. It is also why
//! containment is asserted with a `CONTAIN_TOL` of 1e-12 rather than
//! exactly — the claim is `|v − dense| ≤ b`, which holds here with equality,
//! and an exact comparison would be a knife-edge on f64 summation order.
//! `the_majoranaprop_bound_is_saturated_on_the_headline_cell` pins the
//! saturation itself, so if the fixture ever stops being sharp this file says
//! so instead of quietly weakening.
//!
//! **The two engines took different doors.** majoranaprop's `seed_basis` is
//! `ladder` — its Majorana sum was seeded straight from the fermionic
//! operator (F3) — and stabrank's is `pauli`, because the Schrödinger seed is
//! the state `|0…0⟩` and the observable arrives mapped by Jordan–Wigner
//! (§1.4). That is not a disagreement, and the thing that must agree across
//! it is `R`: §1.2's standing pin is that the range is computed in the basis
//! actually read out, and both engines report `2.7`. A Pauli string and its
//! Majorana monomial differ by a sign only, so a mismatch there would mean
//! one engine's vacuity ceiling had moved.
//!
//! # If the intersection is ever empty
//!
//! It is a bug in one of the two engines, never a property of the job, and
//! the failure message prints both intervals and the dense value so the
//! unsound side can be identified from the output alone. The correct response
//! is to find the unsound bound. Widening a bound until the intervals meet
//! would convert the most valuable signal this phase can produce into a green
//! test.
//!
//! # One mutation this test does NOT catch
//!
//! **A silently halved banked mass.** Measured: with the accumulation in
//! `StabilizerSum::truncate` changed to bank `0.5·|cᵢ|` per dropped branch,
//! this file stays entirely green. Both of stabrank's reported fields move
//! together — `m` halves and so does `R·m·(2+m)` — so the interval shrinks to
//! `[0.6585809…, 0.6755546…]`, which still contains the dense value and still
//! meets majoranaprop's interval. The surviving mass is still positive, so
//! the non-vacuity guards pass too. What catches that mutation is the closed-
//! form mass pin in `omega-backend-stabrank/tests/truncation_witness.rs`,
//! which predicts `m` from the angle sequence instead of accepting whatever
//! the engine banked; this file can only check that the bound is *consistent*
//! with the mass it was given, and `stabrank_json_roundtrip.rs` does that
//! better. An interval test sees unsoundness in the arithmetic, not in the
//! inputs.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

/// The same Clifford + T + Clifford-angle-Givens workload as
/// `stabrank_cli_smoke.rs`, for the reason plan §1.4 gives: it is the overlap
/// regime, cheap for both engines, so both can be pushed into truncation on
/// one job instead of one of them refusing it.
const QASM: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n\
h q[0];\nh q[1];\nh q[2];\nh q[3];\nt q[0];\nt q[1];\nt q[2];\nt q[3];\n\
s q[1];\nrbs(1.5707963267948966) q[1],q[2];\ncx q[2],q[3];\nt q[0];\nt q[3];\n";

const FERMIONIC: &str = "1.25 [0^ 0] + -0.5 [2^ 2] + 0.75 [0^ 1] + 0.75 [1^ 0] \
                         + 0.4 [1^ 3] + 0.4 [3^ 1] + 0.6 [0^ 0 2^ 2]";

/// Both engines report `R = Σ|cᵢ|` over the post-JW Pauli terms, §1.2's pin.
const RANGE: f64 = 2.7;

/// Containment and overlap are claims of the form `x ≤ y`, and the headline
/// cell holds one of them with equality (see the module doc). The tolerance
/// is for f64 summation order at that boundary and nothing else: it is nine
/// orders below the smallest bound in the grid.
const CONTAIN_TOL: f64 = 1e-12;

/// Below this a reported mass is rounding, not a truncation, and the cell
/// would be the vacuous one this test exists to avoid.
const MASS_FLOOR: f64 = 1e-9;

/// stabrank cuts: `--max-chi` values that truncate the 64-branch
/// decomposition and stay informative. Measured — at `--max-chi 32` the bound
/// exceeds `R + |v|` and the engine refuses rather than returning a number,
/// which is S2's vacuity gate doing its job and not a cell for this grid.
const STABRANK_CUTS: [&str; 4] = ["63", "62", "60", "56"];

/// majoranaprop cuts, two axes of its own. `--max-length` is the Majorana
/// monomial length cut; `--truncate` is the coefficient floor shared with
/// pauliprop and stabrank. Both are used so this grid does not depend on one
/// truncation mechanism being the live one.
const MAJORANAPROP_CUTS: [(&str, &str); 3] = [
    ("--max-length", "6"),
    ("--max-length", "4"),
    ("--truncate", "0.2"),
];

/// One certified interval, whichever engine produced it.
struct Interval {
    label: String,
    value: f64,
    bound: f64,
    /// The engine's raw dropped-mass field. **Not** comparable across
    /// engines — that is this file's subject — so it is carried only to prove
    /// truncation fired and to be contrasted with `bound`.
    mass: f64,
    range: f64,
    seed_basis: String,
}

impl Interval {
    fn lo(&self) -> f64 {
        self.value - self.bound
    }
    fn hi(&self) -> f64 {
        self.value + self.bound
    }
    fn width(&self) -> f64 {
        2.0 * self.bound
    }
}

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_omega-run"))
}

fn write_fixture(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omega_stabrank_intervals_{}_{tag}.qasm",
        std::process::id()
    ));
    std::fs::write(&path, QASM).expect("write fixture");
    path
}

fn run_json(fixture: &Path, backend: &str, extra: &[&str]) -> Value {
    let mut args: Vec<&str> = vec![
        fixture.to_str().unwrap(),
        "--qasm-dialect",
        "lenient",
        "--backend",
        backend,
        "--expectation-fermionic",
        FERMIONIC,
        "--format",
        "json",
    ];
    args.extend_from_slice(extra);
    let out = Command::new(binary_path())
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn omega-run");
    assert_eq!(
        out.status.code(),
        Some(0),
        "omega-run --backend {backend} {extra:?} exited {:?}: a cell of this \
         grid stopped completing, so the grid is no longer the one that was \
         measured. stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("stdout is one JSON document")
}

fn f(doc: &Value, path: &[&str]) -> f64 {
    let mut cur = doc;
    for k in path {
        cur = &cur[*k];
    }
    cur.as_f64()
        .unwrap_or_else(|| panic!("{path:?} is not a number in {doc}"))
}

fn dense(fixture: &Path) -> f64 {
    f(&run_json(fixture, "statevector", &[]), &["value"])
}

fn stabrank_interval(fixture: &Path, max_chi: &str) -> Interval {
    let doc = run_json(fixture, "stabrank", &["--max-chi", max_chi]);
    let c = &doc["stabrank_truncation"];
    assert_eq!(
        c["informative"],
        Value::Bool(true),
        "stabrank --max-chi {max_chi} returned an uninformative certificate; \
         the engine is supposed to refuse rather than return one: {c}"
    );
    Interval {
        label: format!("stabrank --max-chi {max_chi}"),
        value: f(&doc, &["value"]),
        // The bound, NOT the mass. Reading `state_dropped_mass` here is the
        // precise error §1.2 is written against, and it would read as a
        // five-times-tighter interval than the derivation supports.
        bound: f(c, &["expectation_error_bound"]),
        mass: f(c, &["state_dropped_mass"]),
        range: f(c, &["observable_range"]),
        seed_basis: c["seed_basis"].as_str().unwrap_or_default().to_string(),
    }
}

fn majoranaprop_interval(fixture: &Path, flag: &str, arg: &str) -> Interval {
    let doc = run_json(fixture, "majoranaprop", &[flag, arg]);
    let c = &doc["majoranaprop_truncation"];
    assert_eq!(
        c["informative"],
        Value::Bool(true),
        "majoranaprop {flag} {arg} returned an uninformative certificate: {c}"
    );
    assert_eq!(
        c["dropped_mass_is_a_bound"],
        Value::Bool(true),
        "majoranaprop's mass is used directly as this interval's half-width \
         below, which is only right because the field is the bound: {c}"
    );
    Interval {
        label: format!("majoranaprop {flag} {arg}"),
        value: f(&doc, &["value"]),
        bound: f(c, &["dropped_mass"]),
        mass: f(c, &["dropped_mass"]),
        range: f(c, &["observable_range"]),
        seed_basis: c["seed_basis"].as_str().unwrap_or_default().to_string(),
    }
}

/// Both engines truncated, and both moved the answer. Without this a cell
/// that dropped nothing would pass every assertion in this file for free.
fn assert_cell_is_live(i: &Interval, dense: f64) {
    assert!(
        i.mass > MASS_FLOOR,
        "{}: dropped mass is {:.3e}, which is rounding and not a truncation. \
         A run that discarded nothing has a zero-width interval and overlaps \
         everything, so this cell would make the whole comparison vacuous.",
        i.label,
        i.mass
    );
    assert!(
        i.bound > MASS_FLOOR,
        "{}: bound is {:.3e}, so the interval has no width to speak of",
        i.label,
        i.bound
    );
    assert!(
        (i.value - dense).abs() > 1e-9,
        "{}: the truncated value {} is the dense value {dense} to within \
         {:.3e}. Truncation fired but changed nothing measurable, so \
         containment here says nothing about the bound.",
        i.label,
        i.value,
        (i.value - dense).abs()
    );
    assert!(
        (i.range - RANGE).abs() < 1e-12,
        "{}: observable_range is {} and not the recorded {RANGE}. §1.2's pin \
         is that R is computed in the basis actually read out; if it moved, \
         both the bound and the vacuity ceiling moved with it.",
        i.label,
        i.range
    );
}

/// Sound means the exact value is inside the certified interval. This is the
/// assertion an unsound bound fails, and it is checked per engine before the
/// intersection is formed, so the failure message names the guilty side.
fn assert_contains_dense(i: &Interval, dense: f64) {
    let err = (i.value - dense).abs();
    assert!(
        err <= i.bound + CONTAIN_TOL,
        "UNSOUND BOUND in {}: the certified interval is [{}, {}] and the dense \
         value is {dense}, which is outside it. |v − dense| = {err:.6e} \
         exceeds the reported bound {:.6e} by {:.3e}. A certificate in this \
         project is a bound and not an estimate, so this is a defect in that \
         engine, not a loose fixture.",
        i.label,
        i.lo(),
        i.hi(),
        i.bound,
        err - i.bound
    );
}

/// **The cross-lane contract.** Every pair of cuts: both engines truncated,
/// both intervals contain the dense value, and the two intervals overlap.
#[test]
fn the_two_engines_certified_intervals_intersect_and_contain_the_dense_value() {
    let path = write_fixture("grid");
    let d = dense(&path);

    let stabrank: Vec<Interval> = STABRANK_CUTS
        .iter()
        .map(|chi| stabrank_interval(&path, chi))
        .collect();
    let majoranaprop: Vec<Interval> = MAJORANAPROP_CUTS
        .iter()
        .map(|(flag, arg)| majoranaprop_interval(&path, flag, arg))
        .collect();

    // Different doors to the same job (§1.4), stated as an assertion because
    // a refactor that routed stabrank through a fermionic seed — or
    // majoranaprop through the Pauli image — would change what this grid is
    // comparing without changing a single number in it.
    assert!(
        stabrank.iter().all(|i| i.seed_basis == "pauli"),
        "stabrank reported a seed basis other than `pauli`"
    );
    assert!(
        majoranaprop.iter().all(|i| i.seed_basis == "ladder"),
        "majoranaprop did not take its direct fermionic door; the two engines \
         are then no longer reaching this job by different routes"
    );

    eprintln!("\n  dense (statevector) = {d:.16}");
    for i in stabrank.iter().chain(majoranaprop.iter()) {
        assert_cell_is_live(i, d);
        assert_contains_dense(i, d);
        eprintln!(
            "  {:<30} v = {:+.12}  b = {:.6e}  mass = {:.6e}  [{:.12}, {:.12}]",
            i.label,
            i.value,
            i.bound,
            i.mass,
            i.lo(),
            i.hi()
        );
    }

    let mut cells = 0usize;
    for s in &stabrank {
        for m in &majoranaprop {
            let (lo, hi) = (s.lo().max(m.lo()), s.hi().min(m.hi()));
            assert!(
                hi >= lo - CONTAIN_TOL,
                "EMPTY INTERSECTION — this is a bug in one of the two engines \
                 and not a property of the job.\n  {} : [{}, {}]\n  {} : [{}, \
                 {}]\n  dense: {d}\nBoth bounds are claimed to be sound, so \
                 the exact value lies in both intervals and they cannot be \
                 disjoint. Find the unsound bound; do NOT widen a bound to \
                 make them meet.",
                s.label,
                s.lo(),
                s.hi(),
                m.label,
                m.lo(),
                m.hi()
            );
            assert!(
                d >= lo - CONTAIN_TOL && d <= hi + CONTAIN_TOL,
                "the intersection [{lo}, {hi}] of {} and {} does not contain \
                 the dense value {d}, even though each interval was checked to \
                 contain it on its own — which is arithmetically impossible \
                 and means one of the two intervals changed under us",
                s.label,
                m.label
            );
            cells += 1;
        }
    }
    assert_eq!(
        cells,
        STABRANK_CUTS.len() * MAJORANAPROP_CUTS.len(),
        "the grid did not run"
    );

    // The headline cell, and §1.2's "sound and tighter": the first cut on
    // each side, whose intersection is strictly inside both inputs.
    let (s, m) = (&stabrank[0], &majoranaprop[0]);
    let (lo, hi) = (s.lo().max(m.lo()), s.hi().min(m.hi()));
    let width = hi - lo;
    eprintln!(
        "  intersection of [{}] and [{}] = [{:.16}, {:.16}], width {:.6e} \
         (inputs {:.6e} and {:.6e})",
        s.label,
        m.label,
        lo,
        hi,
        width,
        s.width(),
        m.width()
    );
    assert!(
        width < s.width() && width < m.width(),
        "the intersection is {width:.6e} wide against inputs of {:.6e} \
         ({}) and {:.6e} ({}). §1.2's claim is that the intersection is \
         *tighter*; on this cell it is a genuine two-sided cut — the lower end \
         from stabrank, the upper from majoranaprop — and if it has stopped \
         being one, this grid no longer demonstrates the claim and the cuts \
         need re-measuring.",
        s.width(),
        s.label,
        m.width(),
        m.label
    );
    assert!(
        cells > 0 && d > 0.0,
        "the fixture's value is not positive; the recorded numbers are stale"
    );

    let _ = std::fs::remove_file(&path);
}

/// The headline cell's majoranaprop bound is **saturated**, and this pins it.
///
/// Containment is only evidence where there is no slack to hide in. The
/// dropped Majorana monomials on this cut are exactly the hopping
/// contribution, so `|v − dense|` equals `dropped_mass` to the last bit and
/// the dense value sits on the interval's upper endpoint. If a future change
/// to the fixture or to the engine made that cell slack, every containment
/// assertion in this file would weaken silently — so the sharpness itself is
/// asserted rather than relied on.
#[test]
fn the_majoranaprop_bound_is_saturated_on_the_headline_cell() {
    let path = write_fixture("sharp");
    let d = dense(&path);
    let m = majoranaprop_interval(&path, "--max-length", "6");
    let err = (m.value - d).abs();
    let ratio = err / m.bound;
    eprintln!(
        "  {}: |v − dense| = {err:.17e}, bound = {:.17e}, ratio = {ratio:.16}",
        m.label, m.bound
    );
    assert!(
        ratio > 0.999,
        "{}: the true error is {err:.6e} against a bound of {:.6e}, a ratio of \
         {ratio:.6}. This cell was chosen because the bound is tight to \
         rounding there, which is what makes containment on it meaningful; at \
         this slack an unsound bound would pass unnoticed and the grid needs a \
         sharper cut.",
        m.label,
        m.bound
    );
    assert!(
        ratio <= 1.0 + CONTAIN_TOL,
        "{}: the true error EXCEEDS the bound, ratio {ratio}. Unsound.",
        m.label
    );
    let _ = std::fs::remove_file(&path);
}

/// §1.2 made measurable: the two engines' dropped-mass fields are not the
/// same quantity, so they cannot be pooled.
///
/// Two independent facts on one job, either of which fails if the fields are
/// ever conflated:
///
/// 1. majoranaprop's mass **is** its bound — the field and the half-width are
///    one number, because `|⟨M⟩| ≤ 1` makes the discarded L1 mass bound the
///    error by itself.
/// 2. stabrank's mass is **not** its bound — the bound is `R·m·(2+m)`, which
///    here is `5.41×` the mass, because `R = 2.7` and the state enters the
///    expectation twice.
///
/// So a consumer that read `state_dropped_mass` as an error budget would
/// claim an interval five times tighter than the derivation supports, and one
/// that summed the two masses would be adding a number in the Heisenberg
/// picture to a number in the Schrödinger one.
#[test]
fn the_two_masses_are_not_the_same_quantity() {
    let path = write_fixture("masses");
    let s = stabrank_interval(&path, "63");
    let m = majoranaprop_interval(&path, "--max-length", "6");

    assert_eq!(
        m.bound, m.mass,
        "majoranaprop's bound {} and its dropped_mass {} have come apart; the \
         additive-bound reading of that field is the whole reason it can be \
         used as a half-width",
        m.bound, m.mass
    );
    let factor = s.bound / s.mass;
    eprintln!(
        "  stabrank: m = {:.6e}, bound = {:.6e}, bound/m = {factor:.4} \
         (R·(2+m) = {:.4}); majoranaprop: mass = bound = {:.6e}",
        s.mass,
        s.bound,
        s.range * (2.0 + s.mass),
        m.mass
    );
    assert!(
        (factor - s.range * (2.0 + s.mass)).abs() < 1e-9,
        "stabrank's bound is {} on a mass of {}, a factor of {factor}, where \
         the derivation gives R·(2+m) = {}. The bound is not R·m·(2+m) of the \
         mass it reported.",
        s.bound,
        s.mass,
        s.range * (2.0 + s.mass)
    );
    assert!(
        factor > 2.0,
        "stabrank's bound is only {factor}× its mass. With R = {} the \
         derivation gives at least 2R; a factor near 1 would mean the engine \
         is reporting majoranaprop's additive bound, which is the §1.2 error \
         this project is written against.",
        s.range
    );
    assert!(
        (s.mass - m.mass).abs() > 1e-3,
        "the two engines banked masses of {} and {} on the same job. They are \
         measuring different things — one the discarded observable, the other \
         the discarded state — and numerically indistinguishable masses here \
         would suggest one field is being computed as the other.",
        s.mass,
        m.mass
    );
    let _ = std::fs::remove_file(&path);
}
