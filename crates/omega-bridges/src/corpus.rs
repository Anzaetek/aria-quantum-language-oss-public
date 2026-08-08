// SPDX-License-Identifier: Apache-2.0
//! The shared QASM2 cross-check corpus, and the one definition of *which*
//! corpus that is.
//!
//! This lives in the library rather than in a test file for a specific reason.
//! The corpus locator **prefers a private tree** when the operator has it
//! checked out, so "the cross-check corpus" is not a fixed set of circuits —
//! it is a decision made at run time. Two harnesses that each carry their own
//! copy of that decision can silently diverge: one runs the audited vendored
//! fixtures, the other the unaudited private ones, and both report green.
//!
//! There is exactly one copy here, used by
//! `crates/omega-bridges/tests/cross_backend.rs` (the bridge-to-bridge L2
//! arms) and `crates/omega-cli/tests/nway_counts.rs` (the N-way counts
//! matrix, which needs the in-tree engines and so cannot live in this crate).
//!
//! [`Corpus::label`] is not decoration. A harness that does not print which
//! corpus it ran is reporting a number whose meaning the reader cannot
//! determine.

use std::path::{Path, PathBuf};

/// Where the cross-check fixtures came from, and what they are.
#[derive(Clone, Debug)]
pub struct Corpus {
    /// Human-readable provenance, e.g. `tests/fixtures/crosscheck`.
    pub label: &'static str,
    /// Absolute paths to every `*.qasm`, sorted for a stable report order.
    pub files: Vec<PathBuf>,
    /// True when this is the corpus vendored into (and audited by) this
    /// repository. False means a larger private tree took precedence, and any
    /// coverage claim this repository makes about the corpus does not apply.
    pub is_vendored: bool,
}

/// Every `*.qasm` under a directory tree, sorted.
fn walk_qasm(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "qasm") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The QASM2 corpus the cross-check harnesses scan.
///
/// Prefers the private `verify-qiskit/fixtures/` tree (369 files) when the
/// operator has it checked out beside this repo; that is the corpus the crate
/// docs and the Perceval/Bloqade thresholds refer to. It is **not vendored
/// into this repository**, so the fallback is the self-contained corpus under
/// `crates/omega-bridges/tests/fixtures/crosscheck/`.
pub fn crosscheck_corpus() -> Corpus {
    let private = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("verify-qiskit")
        .join("fixtures");
    if private.is_dir() {
        let files = walk_qasm(&private);
        if !files.is_empty() {
            return Corpus {
                label: "verify-qiskit/fixtures",
                files,
                is_vendored: false,
            };
        }
    }
    let vendored = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("crosscheck");
    Corpus {
        label: "tests/fixtures/crosscheck",
        files: walk_qasm(&vendored),
        is_vendored: true,
    }
}

/// Gate names a QASM2 source applies, plus any construct the shared lowering
/// refuses outright (`gate`, `opaque`, `if`) reported under its own keyword.
/// Declarations, `barrier`, and `measure`/`reset` (which every backend handles
/// structurally) are not gates and are skipped.
///
/// Deliberately the same shallow scan the Python converter does — it only has
/// to be *conservative*: a name this misses that the converter then refuses
/// shows up as a skipped fixture, never as a wrong number.
pub fn gates_used(qasm: &str) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    for raw_line in qasm.lines() {
        let line = raw_line.split("//").next().unwrap_or("");
        for stmt in line.split(';') {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }
            let head: String = stmt
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if head.is_empty() {
                continue;
            }
            match head.as_str() {
                "OPENQASM" | "include" | "qreg" | "creg" | "barrier" | "measure" | "reset" => {}
                other => {
                    out.insert(other.to_string());
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vendored corpus must be reachable from this crate's manifest dir.
    /// A path typo here would make every downstream harness silently compare
    /// nothing, which is the failure mode the whole module exists to prevent.
    #[test]
    fn vendored_corpus_is_locatable_and_non_empty() {
        let vendored = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("crosscheck");
        let files = walk_qasm(&vendored);
        assert!(
            files.len() >= 11,
            "vendored corpus at {} has {} files — expected at least 11",
            vendored.display(),
            files.len()
        );
    }

    #[test]
    fn gates_used_ignores_comments_and_declarations() {
        let src = "\
// measure this ccx thing
OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1]; // swap would go here
barrier q;
measure q[0] -> c[0];
";
        let gates = gates_used(src);
        // Only the two real gates. `measure`/`barrier`/declarations are
        // structural; `ccx` and `swap` appear ONLY in comments and must not
        // leak in — a fixture filter that reads comments would exclude
        // circuits from backends that could have run them.
        assert_eq!(
            gates.iter().cloned().collect::<Vec<_>>(),
            vec!["cx".to_string(), "h".to_string()]
        );
    }
}
