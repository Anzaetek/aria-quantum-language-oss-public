// SPDX-License-Identifier: Apache-2.0
//! S3 test (iii): **the derived number cannot drift from the raw ones.** A
//! reader of the JSON certificate re-derives `expectation_error_bound` from
//! `state_dropped_mass` and `observable_range` and gets the reported field
//! back, bit for bit.
//!
//! # Why bit for bit and not to a tolerance
//!
//! `R·m·(2+m)` is three f64 operations on two numbers that are in the same
//! document. A consumer who computes it gets exactly what the engine computed
//! — the same expression on the same inputs in the same precision — so the
//! assertion is `==` and not `(a-b).abs() < eps`. A tolerance here would be
//! slack for a wrong formula to live in: `R·m·m` differs from `R·m·(2+m)` by
//! a factor of `(2+m)/m`, which is `637×` at this fixture's mass, but
//! `R·m·(2+m)/(1+1e-13)` differs by nothing a tolerance of 1e-9 would see,
//! and that is still a bound that excludes something it should not.
//!
//! The point of emitting the raw inputs beside the derived field is that this
//! check is *possible for a consumer*. An emitter that computed the JSON's
//! bound independently of the JSON's mass — from a stale certificate, from a
//! different run, from a rounded copy — would be caught here and nowhere
//! else: no library test reads this document, and the interval test
//! (`stabrank_majoranaprop_intervals.rs`) reads only the bound, so a mass
//! field that drifted from it would pass there untouched.
//!
//! # The reader's own float parser is part of the claim, and it was the first
//! thing this test caught
//!
//! The bit-for-bit version of this test failed on its first run, and the
//! certificate was not at fault. `serde_json` without its `float_roundtrip`
//! feature — the default, and what this workspace builds — parses f64 with a
//! fast algorithm that can be **one ULP off**. On the `--max-chi 62` cell it
//! read the emitted `0.010723304703363117` back as `0.010723304703363115`,
//! and `R·m·(2+m)` of the altered mass missed the reported bound by
//! `6.9e-18`.
//!
//! The emitted *text* is exact: `serde_json` writes f64 via shortest
//! round-trip, so the document faithfully carries the engine's numbers. So
//! the numbers in the round-trip assertion below are extracted from the raw
//! stdout text and parsed with the standard library's `str::parse::<f64>()`,
//! which is correctly rounded. `Value`-based access is kept for the
//! structural checks, where one ULP is irrelevant.
//!
//! This is worth a consumer's attention rather than being a test-only
//! workaround: a reader who wants to re-derive the bound and compare it
//! exactly needs a correctly-rounded float parser. With a sloppy one the
//! comparison must carry a one-ULP tolerance — which is still four orders of
//! magnitude tighter than any wrong formula would need.
//!
//! # What else the block owes a reader, and is checked here
//!
//! * **Every key the derivation needs is present**, and no key is `null`.
//!   `missing_keys_are_a_failure_not_a_default` enumerates them, because
//!   `serde_json`'s `Value::Null.as_f64()` is `None` and a consumer that
//!   defaulted it to zero would silently read every truncated run as exact.
//! * **The block is absent, not zero-filled, on a backend that produces no
//!   stabilizer-rank certificate.** "No certificate" and "a certificate
//!   reading zero" have to stay distinguishable, which is the contract the
//!   three neighbouring `attach_*_certificate` functions already keep.
//! * **`truncated_norm_sqr` is not bounded above by 1.** S2 measured two of
//!   forty witness cells at `1.2285`: the branches of a decomposition are not
//!   orthogonal, so `‖ψ′‖² = 1 − ‖Δ‖² − 2Re⟨ψ′|Δ⟩` for the dropped part `Δ`,
//!   and a `Δ` overlapping the survivors negatively leaves the remainder
//!   *longer* than the state it came from. The grid below contains cells above
//!   and below 1 and asserts both, so a consumer-side "clamp it at 1" or an
//!   engine-side renormalisation would be caught rather than inferred. This
//!   is also the field by which a reader checks the no-renormalisation pin of
//!   §1.2, which the bound's derivation depends on.
//! * **`state_dropped_mass_is_a_bound` is `false`**, where majoranaprop's
//!   `dropped_mass_is_a_bound` is `true`. The two documents sit side by side
//!   in machine output and the flag is how a consumer that reads both learns
//!   which field it may use as a half-width.
//!
//! # One mutation this test does NOT catch
//!
//! **An unsound bound whose inputs moved with it.** Measured: with the
//! accumulation in `StabilizerSum::truncate` changed to bank `0.5·|cᵢ|` per
//! dropped branch, every assertion in this file stays green. `m` halves,
//! `R·m·(2+m)` is recomputed from the halved `m`, and the two agree exactly —
//! a consistency pin cannot see a wrong input, which is the third point of
//! `omega-backend-stabrank/tests/mutation_s2_ledger.rs`'s entry **S**. What
//! catches it is the closed-form mass pin in that crate's
//! `truncation_witness.rs`, which predicts `m` from the angle sequence rather
//! than accepting what the engine banked.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

/// The same workload as the other two S3 tests; see
/// `stabrank_cli_smoke.rs` for why it is this one.
const QASM: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n\
h q[0];\nh q[1];\nh q[2];\nh q[3];\nt q[0];\nt q[1];\nt q[2];\nt q[3];\n\
s q[1];\nrbs(1.5707963267948966) q[1],q[2];\ncx q[2],q[3];\nt q[0];\nt q[3];\n";

const FERMIONIC: &str = "1.25 [0^ 0] + -0.5 [2^ 2] + 0.75 [0^ 1] + 0.75 [1^ 0] \
                         + 0.4 [1^ 3] + 0.4 [3^ 1] + 0.6 [0^ 0 2^ 2]";

/// `None` is the exact run, which must round-trip too: `m = 0` gives a bound
/// of `0`, and a formula that produced anything else from a zero mass would
/// be reporting an error on a run that has none. The rest truncate.
const CUTS: [Option<&str>; 6] = [
    None,
    Some("63"),
    Some("62"),
    Some("60"),
    Some("56"),
    Some("48"),
];

/// Keys the block owes a reader, every one of them non-`null`.
const REQUIRED_KEYS: [&str; 16] = [
    "state_dropped_mass",
    "state_dropped_mass_is_a_bound",
    "expectation_error_bound",
    "expectation_error_bound_formula",
    "composes_with_majoranaprop_dropped_mass",
    "observable_range",
    "vacuous_at",
    "informative",
    "exact",
    "final_chi",
    "peak_chi",
    "coeff_min",
    "max_branches",
    "truncated_norm_sqr",
    "seed_basis",
    "max_chi",
];

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_omega-run"))
}

fn write_fixture(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omega_stabrank_roundtrip_{}_{tag}.qasm",
        std::process::id()
    ));
    std::fs::write(&path, QASM).expect("write fixture");
    path
}

fn run_text(fixture: &Path, backend: &str, extra: &[&str]) -> String {
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
        "omega-run --backend {backend} {extra:?} exited {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn run_json(fixture: &Path, backend: &str, extra: &[&str]) -> Value {
    let text = run_text(fixture, backend, extra);
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("stdout is not one JSON document ({e}): {text}"))
}

/// The raw stdout text, for the numbers that are compared exactly.
fn text_of(fixture: &Path, cut: Option<&str>) -> String {
    match cut {
        Some(chi) => run_text(fixture, "stabrank", &["--max-chi", chi]),
        None => run_text(fixture, "stabrank", &[]),
    }
}

/// The whole document. `value` lives at its top level and the certificate
/// block beside it, and `vacuous_at` is derived from both, so a helper that
/// returned only the block could not check that key.
fn doc_of(fixture: &Path, cut: Option<&str>) -> Value {
    run_json(
        fixture,
        "stabrank",
        &match cut {
            Some(chi) => vec!["--max-chi", chi],
            None => vec![],
        },
    )
}

fn cert_of(fixture: &Path, cut: Option<&str>) -> Value {
    doc_of(fixture, cut)["stabrank_truncation"].clone()
}

/// The f64 a JSON document's `"key":<number>` **text** denotes, parsed by the
/// standard library rather than by `serde_json`.
///
/// This exists for one reason, and it is the module doc's: `serde_json`'s
/// default float parser can be one ULP off, so it cannot be used to check an
/// exact identity between numbers it parsed. `str::parse::<f64>()` is
/// correctly rounded, and the emitted text is shortest-round-trip, so going
/// through the text is lossless.
///
/// The key is matched with its closing quote and colon, which is what keeps
/// `"state_dropped_mass":` from also matching
/// `"state_dropped_mass_is_a_bound":`.
fn raw_f64(text: &str, key: &str) -> f64 {
    let needle = format!("\"{key}\":");
    let start = text
        .find(&needle)
        .unwrap_or_else(|| panic!("no `{needle}` in the document:\n{text}"))
        + needle.len();
    let token: String = text[start..]
        .chars()
        .take_while(|c| matches!(c, '0'..='9' | '-' | '+' | '.' | 'e' | 'E'))
        .collect();
    token
        .parse()
        .unwrap_or_else(|e| panic!("`{key}` is `{token}`, which is not an f64 ({e})"))
}

fn num(cert: &Value, key: &str) -> f64 {
    cert[key]
        .as_f64()
        .unwrap_or_else(|| panic!("`{key}` is missing or not a number in {cert}"))
}

/// **The round trip.** `expectation_error_bound ==
/// observable_range · m · (2 + m)`, computed by the reader from the two raw
/// fields in the same document, with no tolerance.
#[test]
fn a_reader_rederives_the_bound_from_the_raw_fields_exactly() {
    let path = write_fixture("derive");
    let mut truncating = 0usize;

    for cut in CUTS {
        let text = text_of(&path, cut);
        let doc: Value = serde_json::from_str(&text).expect("one JSON document");
        let cert = doc["stabrank_truncation"].clone();
        // Read from the TEXT, not through `serde_json`'s float parser — see
        // `raw_f64` and the module doc. Everything compared with `==` below
        // comes through here.
        let m = raw_f64(&text, "state_dropped_mass");
        let r = raw_f64(&text, "observable_range");
        let reported = raw_f64(&text, "expectation_error_bound");
        // The reader's own arithmetic, in the order the field's documented
        // formula states it.
        let derived = r * m * (2.0 + m);
        assert_eq!(
            reported,
            derived,
            "--max-chi {cut:?}: the document reports a bound of {reported} \
             where R·m·(2+m) of its own R = {r} and m = {m} is {derived}, a \
             difference of {:.3e}. The derived field has drifted from the raw \
             fields it is documented as a function of, so a consumer \
             re-deriving it would disagree with the certificate it is reading.",
            reported - derived
        );

        // The formula is not only obeyed, it is stated in the document, so a
        // consumer need not have read this crate to know what to recompute.
        assert_eq!(
            cert["expectation_error_bound_formula"],
            Value::from("observable_range * m * (2 + m), m = state_dropped_mass"),
            "--max-chi {cut:?}: the block must say which expression a reader \
             should recompute; if the formula changes, this string changes with \
             it: {cert}"
        );

        // `vacuous_at` is the block's other derived key and owes the same
        // round trip, against the top-level `value` the run returned.
        let value = raw_f64(&text, "value");
        let vacuous_at = raw_f64(&text, "vacuous_at");
        assert_eq!(
            vacuous_at,
            r + value.abs(),
            "--max-chi {cut:?}: `vacuous_at` is {vacuous_at} where R + |value| \
             of this document's own R = {r} and value = {value} is {}. It is \
             the ceiling the engine's refusal compares the bound against, so a \
             reader must be able to recompute it.",
            r + value.abs()
        );
        // And `informative` must be the comparison it is documented as,
        // evaluated on the numbers in this same document.
        assert_eq!(
            cert["informative"],
            Value::Bool(reported < r + value.abs()),
            "--max-chi {cut:?}: `informative` is {} where \
             `expectation_error_bound < vacuous_at` on this document's numbers \
             is {}: {cert}",
            cert["informative"],
            reported < r + value.abs()
        );

        if m > 0.0 {
            truncating += 1;
            assert!(
                reported > 0.0,
                "--max-chi {cut:?}: a positive mass {m} produced a bound of \
                 {reported}"
            );
            assert_eq!(
                cert["exact"],
                Value::Bool(false),
                "--max-chi {cut:?}: mass {m} was dropped and the run still \
                 reports itself exact: {cert}"
            );
        } else {
            assert_eq!(
                reported, 0.0,
                "--max-chi {cut:?}: nothing was dropped and the bound is \
                 {reported}"
            );
            assert_eq!(cert["exact"], Value::Bool(true));
        }
    }

    assert!(
        truncating >= 4,
        "only {truncating} of {} cuts truncated, so this test mostly compared \
         zero against zero",
        CUTS.len()
    );
    let _ = std::fs::remove_file(&path);
}

/// Every key the block owes, present and non-`null`, on a truncated run and
/// on an exact one.
///
/// `Value::Null.as_f64()` is `None`, and a consumer that reached for
/// `.unwrap_or(0.0)` on a missing mass would read every truncated run as
/// exact. So absence is a failure here, not a default.
#[test]
fn missing_keys_are_a_failure_not_a_default() {
    let path = write_fixture("keys");
    for cut in [None, Some("60")] {
        let cert = cert_of(&path, cut);
        for key in REQUIRED_KEYS {
            assert!(
                cert.get(key).is_some(),
                "--max-chi {cut:?}: the certificate block has no `{key}`: {cert}"
            );
        }
        // `max_chi` is the one key that is legitimately `null` — on an
        // untruncated run there is no cut in force — and it must be `null`
        // rather than `0`, because `Some(0)` is a setting the engine refuses
        // by name.
        match cut {
            None => assert_eq!(
                cert["max_chi"],
                Value::Null,
                "an untruncated run must report `max_chi: null`, not a number: \
                 {cert}"
            ),
            Some(chi) => assert_eq!(
                cert["max_chi"].as_u64().map(|v| v.to_string()).as_deref(),
                Some(chi),
                "the cut in force must be reported as the number that was \
                 passed: {cert}"
            ),
        }
        for key in REQUIRED_KEYS {
            if key == "max_chi" {
                continue;
            }
            assert!(
                !cert[key].is_null(),
                "--max-chi {cut:?}: `{key}` is null, which a consumer reading \
                 it as a number will silently default: {cert}"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// The two flags that tell a consumer reading both engines' documents which
/// mass it may use as an error budget. stabrank's answer is **no**.
#[test]
fn the_block_says_its_mass_is_not_a_bound_and_does_not_compose() {
    let path = write_fixture("flags");
    let cert = cert_of(&path, Some("60"));
    assert_eq!(
        cert["state_dropped_mass_is_a_bound"],
        Value::Bool(false),
        "stabrank's mass is state-side and enters the error as R·m·(2+m); a \
         consumer told it is a bound would claim an interval {}× too tight: \
         {cert}",
        num(&cert, "observable_range") * (2.0 + num(&cert, "state_dropped_mass"))
    );
    assert_eq!(
        cert["composes_with_majoranaprop_dropped_mass"],
        Value::Bool(false),
        "the two engines' masses live in different pictures — one Heisenberg \
         and observable-side, one Schrödinger and state-side — and there is no \
         arithmetic that combines them. What two runs of one job give is an \
         intersection of intervals: {cert}"
    );
    // The neighbouring engine's opposite answer, read from a real run, so the
    // contrast is measured rather than asserted about absent code.
    let mp = run_json(&path, "majoranaprop", &["--max-length", "6"]);
    assert_eq!(
        mp["majoranaprop_truncation"]["dropped_mass_is_a_bound"],
        Value::Bool(true),
        "majoranaprop's field IS the bound; if that ever stopped being true \
         the two blocks would no longer be distinguishable by this flag"
    );
    assert!(
        mp["majoranaprop_truncation"]["state_dropped_mass"].is_null()
            && cert["dropped_mass"].is_null(),
        "the two blocks must not share a key name for their different \
         quantities: majoranaprop has `dropped_mass` and no \
         `state_dropped_mass`, stabrank the reverse"
    );
    let _ = std::fs::remove_file(&path);
}

/// `truncated_norm_sqr` is "different from 1", not "at most 1".
///
/// S2's correction, and the reason it is pinned at CLI level too: a consumer
/// that treats `> 1` as impossible will reject correct runs, and an engine
/// that renormalised `ψ′` would report `1.0` everywhere while satisfying a
/// bound nobody derived. The grid below contains cells on both sides.
#[test]
fn the_truncated_norm_is_not_bounded_above_by_one() {
    let path = write_fixture("norm");
    let exact = cert_of(&path, None);
    assert!(
        (num(&exact, "truncated_norm_sqr") - 1.0).abs() < 1e-9,
        "an untruncated run read out a state of norm² {}; nothing was dropped, \
         so it is the prepared state and must be a unit vector",
        num(&exact, "truncated_norm_sqr")
    );

    let mut above = Vec::new();
    let mut moved = 0usize;
    for chi in ["63", "62", "60", "56", "48"] {
        let cert = cert_of(&path, Some(chi));
        let n = num(&cert, "truncated_norm_sqr");
        if (n - 1.0).abs() > 1e-9 {
            moved += 1;
        }
        if n > 1.0 {
            above.push((chi, n));
        }
    }
    assert!(
        moved >= 4,
        "only {moved} of five cuts changed the norm at all; a truncating run \
         whose norm² stays 1.0 has been renormalised, which invalidates the \
         derivation the bound rests on"
    );
    assert!(
        !above.is_empty(),
        "no cut on this fixture left the truncated state with norm² above 1. \
         S2 measured that case — the branches are not orthogonal, so a dropped \
         part overlapping the survivors negatively leaves a longer remainder — \
         and this assertion is what keeps a `≤ 1` reading of the field from \
         looking safe. Either the engine started renormalising or the fixture \
         no longer exercises the case."
    );
    eprintln!("  cells with norm² > 1: {above:?}");
    let _ = std::fs::remove_file(&path);
}

/// Absent, not zero-filled, on a backend with no stabilizer-rank certificate.
/// "No certificate" and "a certificate reading zero" must stay apart, or a
/// consumer cannot tell an exact run from a build that does not report.
#[test]
fn the_block_is_absent_on_other_backends() {
    let path = write_fixture("absent");
    for backend in ["statevector", "majoranaprop", "pauliprop"] {
        let doc = run_json(&path, backend, &[]);
        assert!(
            doc.get("stabrank_truncation").is_none(),
            "--backend {backend} emitted a stabrank_truncation block: {doc}"
        );
        assert!(
            doc["value"].is_f64(),
            "--backend {backend} produced no value: {doc}"
        );
    }
    let _ = std::fs::remove_file(&path);
}
