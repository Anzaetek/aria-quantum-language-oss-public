//! Smoke tests for `omega-run --bridge`.
//!
//! These exercise the dispatcher plumbing without requiring a Python
//! venv. The default workspace build has no `bridge-*` features, so the
//! dispatcher returns `NotCompiled` — exactly the "off" path operators
//! see when they haven't opted into a bridge backend.

use std::path::PathBuf;
use std::process::{Command, Stdio};

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

fn fixture() -> PathBuf {
    // A self-contained Bell QASM shipped alongside the test. (The earlier path
    // pointed at a private `verify-qiskit/fixtures/` tree that isn't vendored
    // into this repo, so every test failed at file-read before reaching the
    // bridge dispatcher these tests actually exercise.)
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("bell_phi_plus.qasm")
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(binary_path())
        .args(args)
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn omega-run")
}

/// Only meaningful when the feature is ABSENT — the whole point is what
/// happens without it. Compiled in, `--bridge qiskit` succeeds and the
/// assertion `expected failure, got success` fires.
///
/// It was ungated, so `cargo test -p omega-cli --features bridge-qiskit` had
/// always failed here. Nothing noticed because ci.sh never tested omega-cli
/// with bridge features until the N-way matrix stage arrived — the same
/// hand-maintained-crate-list hole as FIXES_PLAN.md K9, one layer down.
#[cfg(not(feature = "bridge-qiskit"))]
#[test]
fn bridge_default_features_reports_not_compiled() {
    // Without `bridge-qiskit`/`bridge-perceval` compiled in, every
    // `--bridge <name>` invocation should fail with the typed
    // NotCompiled message that points operators at the rebuild flag.
    let path = fixture();
    let output = run(&[
        path.to_str().unwrap(),
        "--bridge",
        "qiskit",
        "--shots",
        "100",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "expected failure, got success");
    assert!(
        stderr.contains("not compiled in"),
        "expected NotCompiled message, got: {stderr}"
    );
    assert!(
        stderr.contains("--features bridge-qiskit"),
        "expected rebuild hint, got: {stderr}"
    );
}

#[test]
fn bridge_rejects_statevector_mode() {
    let path = fixture();
    let output = run(&[
        path.to_str().unwrap(),
        "--bridge",
        "qiskit",
        "--statevector",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("--bridge supports only the default sampling mode"),
        "expected mode-rejection notice, got: {stderr}"
    );
}

#[test]
fn bridge_rejects_unknown_backend_name() {
    let path = fixture();
    let output = run(&[
        path.to_str().unwrap(),
        "--bridge",
        "no-such-backend",
        "--shots",
        "100",
    ]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("Invalid --bridge value") && stderr.contains("unknown backend"),
        "expected unknown-backend message, got: {stderr}"
    );
}
