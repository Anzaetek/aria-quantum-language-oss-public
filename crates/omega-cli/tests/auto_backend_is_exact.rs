// SPDX-License-Identifier: Apache-2.0
//! **`--backend auto` never silently substitutes an approximate backend.**
//!
//! The obvious `auto` — the one the server already implements — falls back to
//! MPS at a fixed bond dimension for wide circuits. That is an approximation
//! chosen by a heuristic the user never saw, and MPS truncation is precisely
//! what this workspace spends a per-run certificate on reporting. An `auto`
//! that can quietly hand back a truncated answer is worse than no `auto`.
//!
//! So this one picks only among backends that are **exact** for the circuit in
//! front of it, always says which it picked, and refuses when no exact backend
//! fits rather than guessing.

use std::process::Command;

/// Returns (stdout, stdout+stderr, success).
///
/// The selection notice goes through the CLI's `info()`, which routes to
/// **stdout** in text mode and stderr in machine modes. What matters to these
/// tests is that the user is told at all, so they assert against the combined
/// streams rather than pinning which one it lands on.
fn omega_run(args: &[&str]) -> (String, String, bool) {
    let exe = env!("CARGO_BIN_EXE_omega-run");
    let out = Command::new(exe).args(args).output().expect("run");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let both = format!("{stdout}\n{stderr}");
    (stdout, both, out.status.success())
}

fn write(name: &str, body: &str) -> String {
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, body).expect("write fixture");
    path.to_string_lossy().into_owned()
}

/// A Clifford circuit far too wide for a dense state must route to the exact
/// stabilizer backend — and the choice must be reported, because the same
/// circuit elsewhere is exponential and the user deserves to know why it was
/// fast.
#[test]
fn a_wide_clifford_circuit_routes_to_the_exact_stabilizer_backend() {
    let mut src = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[120];\n");
    for i in 0..120 {
        src.push_str(&format!("h q[{i}];\n"));
    }
    for i in 0..119 {
        src.push_str(&format!("cx q[{i}], q[{}];\n", i + 1));
    }
    let path = write("auto_clifford120.qasm", &src);
    let (_out, err, ok) = omega_run(&[&path, "--backend", "auto", "--shots", "4", "--seed", "1"]);
    assert!(
        ok,
        "a 120-qubit Clifford circuit must run under auto:\n{err}"
    );
    assert!(
        err.contains("pauli"),
        "auto must report that it chose the stabilizer backend:\n{err}"
    );
}

/// A small non-Clifford circuit goes to the dense statevector, which is exact.
#[test]
fn a_small_non_clifford_circuit_routes_to_the_statevector() {
    let path = write(
        "auto_small_nc.qasm",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\nh q[0];\nt q[1];\ncx q[0], q[1];\n",
    );
    let (_out, err, ok) = omega_run(&[&path, "--backend", "auto", "--shots", "8", "--seed", "1"]);
    assert!(ok, "a small non-Clifford circuit must run:\n{err}");
    assert!(
        err.contains("statevector"),
        "auto must report the statevector choice:\n{err}"
    );
}

/// **The refusal is the point of the feature.** Non-Clifford and too wide for
/// a dense state: there is no exact backend, so `auto` must stop and name the
/// approximate options rather than pick one.
#[test]
fn a_circuit_with_no_exact_backend_is_refused_with_options() {
    let path = write(
        "auto_big_nc.qasm",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[40];\nh q[0];\nt q[1];\n",
    );
    let (_out, err, ok) = omega_run(&[&path, "--backend", "auto", "--shots", "4"]);
    assert!(!ok, "must refuse rather than approximate:\n{err}");
    assert!(
        err.contains("will not silently pick an approximate backend"),
        "the refusal must state the principle:\n{err}"
    );
    for option in ["mps:<chi>", "mps:auto", "pauliprop"] {
        assert!(
            err.contains(option),
            "the refusal must offer `{option}`:\n{err}"
        );
    }
}

/// `auto` must not disturb an explicit choice.
#[test]
fn an_explicit_backend_is_untouched() {
    let path = write(
        "auto_explicit.qasm",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\nh q[0];\ncx q[0], q[1];\n",
    );
    let (_out, err, ok) = omega_run(&[
        &path,
        "--backend",
        "statevector",
        "--shots",
        "8",
        "--seed",
        "1",
    ]);
    assert!(ok, "explicit backend must still work:\n{err}");
    assert!(
        !err.contains("auto:"),
        "auto must stay silent when it was not asked for:\n{err}"
    );
}
