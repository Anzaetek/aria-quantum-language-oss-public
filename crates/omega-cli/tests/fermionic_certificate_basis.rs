// SPDX-License-Identifier: Apache-2.0
//! **The majoranaprop certificate's `observable_range` is the L1 norm of the
//! *Pauli* image, not of the ladder coefficients.** (PLAN-FERMIONIC §1.3)
//!
//! `--expectation-fermionic` is the first path that puts a `FermionicOp` in the
//! CLI's hands upstream of the engine. Its ladder coefficients sit right next
//! to where the certificate is built, and a refactor that reaches for them
//! would move the vacuity gate with nothing to catch it — the certificate is
//! the one number in this project that is a bound rather than an estimate.
//!
//! `n_0` does not discriminate: ladder L1 = 1 and its JW image
//! `0.5·I − 0.5·Z0` also has Pauli L1 = 1. The non-adjacent hopping does:
//!
//! ```text
//! a†₁a₃ + h.c.   ladder: 1.0 [1^ 3] + 1.0 [3^ 1]          → L1 = 2.0
//!                JW:     0.5·X1Z2X3 + 0.5·Y1Z2Y3           → L1 = 1.0
//! ```
//!
//! A factor of two apart. The wrong basis reads 2.0 and both tests go red.

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

/// Non-Clifford, entangling, touching the Z-string qubit (2) so the
/// Heisenberg-evolved observable is not the seed and the value is not 0.
const QASM: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n\
x q[1];\nry(0.7) q[3];\ncx q[1],q[2];\nrz(0.4) q[2];\ncx q[3],q[1];\nry(0.3) q[1];\n";

const FERMIONIC: &str = "1 [1^ 3] + 1 [3^ 1]";
const PAULI_IMAGE: &str = "0.5*X1Z2X3+0.5*Y1Z2Y3";
const LADDER_L1: f64 = 2.0;
const PAULI_L1: f64 = 1.0;

fn write_fixture(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omega_fermionic_cert_basis_{}_{tag}.qasm",
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

fn observable_range(doc: &Value) -> f64 {
    doc["majoranaprop_truncation"]["observable_range"]
        .as_f64()
        .unwrap_or_else(|| panic!("no majoranaprop_truncation.observable_range in {doc}"))
}

#[test]
fn fermionic_certificate_range_is_the_pauli_l1_not_the_ladder_l1() {
    let path = write_fixture("fermionic");
    let doc = run_json(&[path.to_str().unwrap(), "--expectation-fermionic", FERMIONIC]);
    let range = observable_range(&doc);
    assert!(
        (range - LADDER_L1).abs() > 0.5,
        "observable_range = {range}: computed from the ladder coefficients (L1 = {LADDER_L1}), \
         not the Pauli image — the vacuity gate has moved"
    );
    assert!(
        (range - PAULI_L1).abs() < 1e-12,
        "observable_range = {range}, expected the JW image's L1 = {PAULI_L1}"
    );
    assert_eq!(
        doc["observable_mapping"]["mapping"],
        Value::from("jordan_wigner"),
        "the mapping the certificate was taken in must be recorded: {doc}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn fermionic_certificate_matches_the_hand_spelled_pauli_image() {
    let path = write_fixture("pauli");
    let file = path.to_str().unwrap();
    let fermionic = run_json(&[file, "--expectation-fermionic", FERMIONIC]);
    let pauli = run_json(&[file, "--expectation", PAULI_IMAGE]);
    let (rf, rp) = (observable_range(&fermionic), observable_range(&pauli));
    assert!(
        (rf - rp).abs() < 1e-12,
        "range differs between the fermionic spelling ({rf}) and its Pauli image ({rp})"
    );
    let (vf, vp) = (
        fermionic["value"].as_f64().expect("value"),
        pauli["value"].as_f64().expect("value"),
    );
    assert!(
        (vf - vp).abs() < 1e-9,
        "value differs between the fermionic spelling ({vf}) and its Pauli image ({vp})"
    );
    assert!(
        vf.abs() > 1e-3,
        "fixture must not give a trivial expectation: {vf}"
    );
    let _ = std::fs::remove_file(&path);
}
