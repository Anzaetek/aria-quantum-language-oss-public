// SPDX-License-Identifier: Apache-2.0
//! **`--expectation-fermionic` on majoranaprop seeds the engine from the
//! ladder terms, not from the JW Pauli image.** (PLAN-FERMIONIC F3)
//!
//! F1/F2 route every fermionic observable through Jordan–Wigner and hand the
//! Pauli sum to whichever backend was chosen. For majoranaprop that detour is
//! a waste: the engine's native basis IS the Majorana basis, and a ladder
//! operator maps into it directly (`engine::seed_from_fermionic`). F3 opens
//! that door in the CLI.
//!
//! The witness is the engine's, not the CLI's: `MajoranaPropCertificate`
//! carries `seed_basis`, written by the engine's `finish` from which door
//! called it. A flag set by the CLI next to the dispatch would pass whether or
//! not the engine took the direct door — that was the first draft of this
//! test, and it survived the mutation below. The certificate field does not.
//!
//! - the fermionic spelling on majoranaprop reports `seed_basis: "ladder"`;
//! - the hand-spelled Pauli image reports `"pauli"`;
//! - the two doors agree on the value and on the certificate's
//!   `observable_range` (the engine-level test `fermionic_seed.rs` (i) holds
//!   them bit-identical; here the tolerance is the CLI's float formatting).
//!
//! Mutation: point the `Some(op) =>` arm of the majoranaprop expectation
//! dispatch in `main.rs` back at `expectation_with_certificate` and the first
//! assertion goes red — the engine reports `"pauli"` because that is what it
//! was given.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::Value;

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_omega-run"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Same non-Clifford, entangling fixture as `fermionic_certificate_basis.rs`:
/// touches the Z-string qubit so the evolved sum is not the seed.
const QASM: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n\
x q[1];\nry(0.7) q[3];\ncx q[1],q[2];\nrz(0.4) q[2];\ncx q[3],q[1];\nry(0.3) q[1];\n";

const FERMIONIC: &str = "1 [1^ 3] + 1 [3^ 1]";
const PAULI_IMAGE: &str = "0.5*X1Z2X3+0.5*Y1Z2Y3";

fn write_fixture(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omega_fermionic_direct_seed_{}_{tag}.qasm",
        std::process::id()
    ));
    std::fs::write(&path, QASM).expect("write fixture");
    path
}

fn run_json(args: &[&str]) -> Value {
    let out = Command::new(binary_path())
        .args(args)
        .args(["--backend", "majoranaprop", "--format", "json"])
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn omega-run");
    assert!(
        out.status.success(),
        "omega-run {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("stdout is one JSON document")
}

fn seed_basis(doc: &Value) -> &str {
    doc["majoranaprop_truncation"]["seed_basis"]
        .as_str()
        .unwrap_or_else(|| panic!("no majoranaprop_truncation.seed_basis in {doc}"))
}

#[test]
fn fermionic_input_on_majoranaprop_is_seeded_from_the_ladder_terms() {
    let path = write_fixture("direct");
    let doc = run_json(&[path.to_str().unwrap(), "--expectation-fermionic", FERMIONIC]);
    assert_eq!(
        seed_basis(&doc),
        "ladder",
        "the engine was handed a Pauli image for a fermionic observable — \
         the JW detour is back: {doc}"
    );
    // The mapping doc still describes the JW image, because that is the
    // basis the certificate's L1 is taken in (§1.3) — direct seeding must
    // not erase that record.
    assert_eq!(
        doc["observable_mapping"]["mapping"],
        Value::from("jordan_wigner")
    );
    assert_eq!(
        doc["observable_mapping"]["input_basis"],
        Value::from("ladder")
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn both_doors_reach_the_same_value_and_certificate() {
    let path = write_fixture("both");
    let file = path.to_str().unwrap();
    let direct = run_json(&[file, "--expectation-fermionic", FERMIONIC]);
    let detour = run_json(&[file, "--expectation", PAULI_IMAGE]);
    assert_eq!(seed_basis(&direct), "ladder");
    assert_eq!(seed_basis(&detour), "pauli");
    assert!(
        detour["observable_mapping"].is_null(),
        "a Pauli spelling has no mapping doc: {detour}"
    );
    let (vd, vp) = (
        direct["value"].as_f64().expect("value"),
        detour["value"].as_f64().expect("value"),
    );
    assert!(
        vd.abs() > 1e-3,
        "fixture must not give a trivial expectation: {vd}"
    );
    assert!(
        (vd - vp).abs() < 1e-9,
        "direct door ({vd}) and JW detour ({vp}) disagree on the value"
    );
    for key in [
        "observable_range",
        "dropped_mass",
        "final_terms",
        "peak_terms",
    ] {
        let (a, b) = (
            &direct["majoranaprop_truncation"][key],
            &detour["majoranaprop_truncation"][key],
        );
        match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => assert!(
                (x - y).abs() < 1e-12,
                "certificate.{key}: direct {x} vs detour {y}"
            ),
            _ => assert_eq!(a, b, "certificate.{key} differs between the doors"),
        }
    }
    let _ = std::fs::remove_file(&path);
}
