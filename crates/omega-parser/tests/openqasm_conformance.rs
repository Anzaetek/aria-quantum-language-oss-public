// SPDX-License-Identifier: Apache-2.0
//! **Every official OpenQASM example is accounted for.**
//!
//! Walks `third_party/openqasm-examples/` — the specification's own corpus,
//! vendored byte-exact (see its `PROVENANCE.md`) — and asserts that each
//! program is either **parsed** or **refused with a message naming the
//! construct we do not support**.
//!
//! # The gate is coverage, not a score
//!
//! "N of 21 parse" is a subset report. What means something is that all 21 are
//! classified, and that the classification is checked in: a file moving between
//! categories — in *either* direction — fails here rather than drifting
//! unnoticed. Counting only the files we can do is the compared-zero-cells trap
//! this repository has found in the N-way anchor, the Perceval conventions lane
//! and the photonics width test.
//!
//! # A refusal must be diagnostic
//!
//! Every expected substring below was **read off an actual run**, not
//! predicted. That distinction earned its keep: a first draft of this table
//! guessed `cphase.qasm` would fail with "undefined qreg" (it declares no
//! registers). It does not — it dies on the Unicode `θ` in
//! `gate cphase(θ) a, b`, long before any register is looked up. Three of the
//! sixteen refusals were unnamed until the messages were measured:
//! `defcalgrammar` is a different keyword from `defcal`, `inverseqft1`'s cast
//! sits inside an `if`, and no keyword lookup can explain a non-ASCII
//! identifier.
//!
//! # Which reader this measures
//!
//! `omega_parser::lower_to_ir` — the lane that executes. This matters:
//! `aria_core::ast::qasm::from_qasm` is a **second, stricter** reader that
//! refuses register broadcast outright, so a conformance figure taken there
//! would differ. Naming the reader is part of the result.

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
enum Outcome {
    Parses,
    /// Refused, with a message containing this substring.
    RefusedNaming(&'static str),
}
use Outcome::{Parses, RefusedNaming};

/// The ledger. Editing this table is how a capability change gets recorded.
const EXPECTED: &[(&str, Outcome)] = &[
    // ---- Tier 1: within the gate-model profile ----
    ("rb.qasm", Parses),
    ("qft.qasm", Parses),
    ("qpt.qasm", Parses),
    ("teleport.qasm", Parses),
    ("inverseqft2.qasm", Parses),
    // ---- Tier 2: refused, each naming its own construct ----
    (
        "adder.qasm",
        RefusedNaming("classical variable declaration"),
    ),
    ("alignment.qasm", RefusedNaming("timing control")),
    ("arrays.qasm", RefusedNaming("array")),
    ("cphase.qasm", RefusedNaming("non-ASCII")),
    ("dd.qasm", RefusedNaming("timing control")),
    ("defcal.qasm", RefusedNaming("pulse-level calibration")),
    ("gateteleport.qasm", RefusedNaming("const")),
    // A cast inside the guard: `if(int[4](c) == 1)`. Matched on "cast" rather
    // than on the surrounding prose, which changed once already when the reader
    // gained the `c[i] == true/false` form and the message had to describe both.
    ("inverseqft1.qasm", RefusedNaming("a cast")),
    ("ipe.qasm", RefusedNaming("const")),
    ("msd.qasm", RefusedNaming("const")),
    ("qec.qasm", RefusedNaming("def")),
    ("rus.qasm", RefusedNaming("def")),
    ("scqec.qasm", RefusedNaming("const")),
    ("t1.qasm", RefusedNaming("timing control")),
    ("varteleport.qasm", RefusedNaming("const")),
    ("vqe.qasm", RefusedNaming("const")),
];

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("third_party/openqasm-examples")
}

/// The table must describe the directory exactly — no file unlisted, no entry
/// without a file. Without this, adding an example and forgetting to classify
/// it would leave it silently unchecked, which is the failure mode this whole
/// test exists to prevent.
#[test]
fn the_ledger_covers_every_vendored_program() {
    let dir = corpus_dir();
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".qasm"))
        .collect();
    on_disk.sort();

    let mut listed: Vec<String> = EXPECTED.iter().map(|(n, _)| n.to_string()).collect();
    listed.sort();

    assert_eq!(
        on_disk, listed,
        "the ledger and the vendored corpus have drifted (`stdgates.inc` is \
         excluded on purpose — it is an include, not a program)"
    );
    assert_eq!(listed.len(), 21, "the corpus is 21 programs");
}

/// The ledger itself.
#[test]
fn every_official_example_is_parsed_or_refused_by_name() {
    let dir = corpus_dir();
    let mut failures: Vec<String> = Vec::new();
    let (mut parsed, mut refused) = (0, 0);

    for (name, expected) in EXPECTED {
        let path = dir.join(name);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        match (omega_parser::lower_to_ir(&src), expected) {
            (Ok(ir), Parses) => {
                if ir.ops.is_empty() {
                    failures.push(format!(
                        "{name}: parsed to ZERO ops — an empty circuit is not a parse"
                    ));
                } else {
                    parsed += 1;
                }
            }
            (Ok(ir), RefusedNaming(what)) => failures.push(format!(
                "{name}: expected a refusal naming `{what}`, but it PARSED to {} ops. \
                 If this is a real capability gain, move it to Tier 1 here AND in \
                 PROVENANCE.md.",
                ir.ops.len()
            )),
            (Err(e), Parses) => {
                failures.push(format!("{name}: expected to parse, but was refused: {e}"))
            }
            (Err(e), RefusedNaming(what)) => {
                if e.contains(what) {
                    refused += 1;
                } else {
                    failures.push(format!(
                        "{name}: refused, but the message does not name `{what}`, so a \
                         user cannot act on it. Got: {e}"
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "OpenQASM conformance ledger — {} problem(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
    assert_eq!(
        parsed + refused,
        EXPECTED.len(),
        "every program must land in exactly one category"
    );
    assert_eq!(parsed, 5, "Tier 1 is five files");
    assert_eq!(refused, 16, "Tier 2 is sixteen files");
}

/// No refusal may be a bare pest position.
///
/// This is the property that makes the ledger worth having rather than a
/// tautology: `--> 9:7 = expected param_list_app` describes OUR grammar, not
/// the user's file, and every Tier-2 program produced exactly that before this
/// work. Asserted separately from the table so a future entry cannot be
/// "satisfied" by a substring that happens to appear in pest's rule list.
#[test]
fn no_refusal_is_only_a_position() {
    let dir = corpus_dir();
    for (name, expected) in EXPECTED {
        let RefusedNaming(_) = expected else { continue };
        let src = std::fs::read_to_string(dir.join(name)).expect("read");
        let e = omega_parser::lower_to_ir(&src).expect_err("Tier 2 must refuse");
        let first = e.lines().next().unwrap_or("");
        assert!(
            first.len() > 40 && !first.trim_start().starts_with("-->"),
            "{name}: the refusal opens with a bare position rather than an \
             explanation: {first}"
        );
    }
}

/// The vendored files must be the upstream bytes — a corpus quietly edited to
/// suit the reader tests the reader against itself.
#[test]
fn the_vendored_corpus_has_a_complete_checksum_manifest() {
    let sums =
        std::fs::read_to_string(corpus_dir().join("SHA256SUMS")).expect("SHA256SUMS must exist");
    let lines = sums.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(
        lines, 22,
        "SHA256SUMS must pin all 21 programs plus stdgates.inc, found {lines}"
    );
}
