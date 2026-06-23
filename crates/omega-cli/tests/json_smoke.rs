//! End-to-end CLI smoke tests for --format json / jsonl.
//!
//! These invoke the built `omega-run` binary, pipe its stdout through `jq`,
//! and assert on the extracted fields. Skipped gracefully if `jq` is not on
//! PATH.

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn have_jq() -> bool {
    Command::new("jq")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn binary_path() -> PathBuf {
    // CARGO_BIN_EXE_omega-run is set by cargo for integration tests.
    PathBuf::from(env!("CARGO_BIN_EXE_omega-run"))
}

fn repo_root() -> PathBuf {
    // crates/omega-cli -> ../../ = repo root.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

fn run_pipe_jq(args: &[&str], jq_filter: &str) -> String {
    let mut omega = Command::new(binary_path())
        .args(args)
        .current_dir(repo_root())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn omega-run");

    let omega_stdout = omega.stdout.take().expect("omega stdout");
    let jq = Command::new("jq")
        .arg("-r")
        .arg(jq_filter)
        .stdin(omega_stdout)
        .output()
        .expect("spawn jq");

    let _ = omega.wait().expect("wait omega");
    assert!(jq.status.success(), "jq failed: {:?}", jq);
    String::from_utf8(jq.stdout).unwrap().trim().to_string()
}

#[test]
fn counts_json_total_matches_shots() {
    if !have_jq() {
        eprintln!("SKIP: jq not on PATH");
        return;
    }
    let total = run_pipe_jq(
        &[
            "examples/circuits/bell.qasm",
            "--shots",
            "256",
            "--seed",
            "7",
            "--format",
            "json",
        ],
        "[.counts[]] | add",
    );
    assert_eq!(total, "256", "counts sum should equal shots");
}

#[test]
fn statevector_json_has_right_length() {
    if !have_jq() {
        eprintln!("SKIP: jq not on PATH");
        return;
    }
    let len = run_pipe_jq(
        &[
            "examples/circuits/bell.qasm",
            "--statevector",
            "--format",
            "json",
        ],
        ".amplitudes | length",
    );
    assert_eq!(len, "4", "2-qubit statevector should have 4 amplitudes");
}

#[test]
fn expectation_json_bell_zz_is_one() {
    if !have_jq() {
        eprintln!("SKIP: jq not on PATH");
        return;
    }
    let value = run_pipe_jq(
        &[
            "examples/circuits/bell.qasm",
            "--expectation",
            "Z0Z1",
            "--format",
            "json",
        ],
        ".value",
    );
    let v: f64 = value.parse().unwrap();
    assert!(
        (v - 1.0).abs() < 1e-9,
        "⟨Z⊗Z⟩ on Bell state should be 1, got {}",
        v
    );
}

#[test]
fn pauliprop_expectation_matches_statevector_on_bell() {
    if !have_jq() {
        eprintln!("SKIP: jq not on PATH");
        return;
    }
    // The Pauli-propagation backend computes ⟨Z⊗Z⟩ on the Bell state via
    // Heisenberg propagation; it must equal +1 and match the statevector.
    let pp = run_pipe_jq(
        &[
            "examples/circuits/bell.qasm",
            "--backend",
            "pauliprop",
            "--expectation",
            "Z0Z1",
            "--format",
            "json",
        ],
        ".value",
    );
    let sv = run_pipe_jq(
        &[
            "examples/circuits/bell.qasm",
            "--backend",
            "statevector",
            "--expectation",
            "Z0Z1",
            "--format",
            "json",
        ],
        ".value",
    );
    let (p, s): (f64, f64) = (pp.parse().unwrap(), sv.parse().unwrap());
    assert!(
        (p - 1.0).abs() < 1e-9,
        "pauliprop ⟨Z⊗Z⟩ on Bell = {p}, expected 1"
    );
    assert!((p - s).abs() < 1e-9, "pauliprop {p} != statevector {s}");
}

#[test]
fn qpy_input_decodes_via_pure_rust_reader_to_bell_counts() {
    // End-to-end CLI test of the QPY auto-detect path
    // (`omega-cli/src/main.rs::is_qpy_input` → pure-Rust reader →
    // `lower_to_ir` → execute). The fixture is the same Bell QPY
    // blob the bridges crate uses for its reader-pin tests, only
    // here we go through the CLI binary so a regression in the
    // glue (extension detection, magic-byte detection,
    // prebuilt_circuit handoff) shows up.
    if !have_jq() {
        eprintln!("SKIP: jq not on PATH");
        return;
    }
    // Path is relative to the repo root (run_pipe_jq sets
    // current_dir to repo_root() — same convention as the QASM
    // tests above).
    let total = run_pipe_jq(
        &[
            "crates/omega-bridges/tests/fixtures/bell_qiskit_2_4_1.qpy",
            "--shots",
            "256",
            "--seed",
            "11",
            "--format",
            "json",
        ],
        "[.counts[]] | add",
    );
    assert_eq!(total, "256", "QPY-decoded Bell counts should sum to shots");

    // Bell-state signature: only |00⟩ and |11⟩ (encoded as 0b00 / 0b11).
    // The CLI emits MSB-first bit-strings (per its serializer); on a
    // 2-qubit register that's still "00" / "11". Anything else means
    // the QPY decode dropped a gate or rewired qubits.
    let keys = run_pipe_jq(
        &[
            "crates/omega-bridges/tests/fixtures/bell_qiskit_2_4_1.qpy",
            "--shots",
            "256",
            "--seed",
            "11",
            "--format",
            "json",
        ],
        ".counts | keys | sort | join(\",\")",
    );
    for k in keys.split(',') {
        assert!(
            k == "00" || k == "11",
            "QPY-decoded Bell state should only produce 00 / 11, got {k:?}"
        );
    }
}
