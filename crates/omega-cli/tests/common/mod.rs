// SPDX-License-Identifier: Apache-2.0
//! Shared venv discovery for the cross-check lanes.
//!
//! # Why this exists
//!
//! The repository had **two conventions for the same venv** and no one place
//! that knew both. `ci.sh`'s Qiskit differential stage reads
//! `./.venv-qiskit` at the repo root (overridable with `ARIA_QISKIT_PY`),
//! while the N-way and truncation-bound lanes each carried their own copy of a
//! resolver that looked only in `crates/omega-bridges/python/`.
//!
//! The consequence was not a failure, which is what makes it worth a module:
//! those lanes **self-skip** when the anchor is absent, print a note, and
//! report `ok`. So a host with a perfectly good venv in the location `ci.sh`
//! itself uses would run the N-way matrix, see green, and never learn that the
//! independent anchor never executed — leaving exactly the "our engines agree
//! with each other" configuration those tests exist to prevent. Observed on
//! this machine: the N-way counts lane reported 7 passed while comparing
//! against nothing.
//!
//! Resolution order, most explicit first:
//!
//! 1. `ARIA_QISKIT_PY` — the override `ci.sh` already documents (qiskit only).
//! 2. `crates/omega-bridges/python/.venv-<slug>` — the bridge-local convention.
//! 3. `<repo root>/.venv-<slug>` — the convention `ci.sh` uses.

use std::path::PathBuf;

fn bridge_python_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("omega-bridges")
        .join("python")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// Interpreter for the `<slug>` bridge venv, or a non-existent path when none
/// is installed (callers test `.exists()` and skip with a note).
///
/// Returning the *bridge-local* path when nothing is found is deliberate: it
/// keeps the existing "build it here" guidance in the skip messages pointing at
/// the same place it always did.
pub fn venv_python(slug: &str) -> PathBuf {
    if slug == "qiskit" {
        if let Ok(explicit) = std::env::var("ARIA_QISKIT_PY") {
            let p = PathBuf::from(explicit);
            if p.exists() {
                return p;
            }
        }
    }
    let local = bridge_python_dir()
        .join(format!(".venv-{slug}"))
        .join("bin")
        .join("python");
    if local.exists() {
        return local;
    }
    let root = repo_root()
        .join(format!(".venv-{slug}"))
        .join("bin")
        .join("python");
    if root.exists() {
        return root;
    }
    local
}
