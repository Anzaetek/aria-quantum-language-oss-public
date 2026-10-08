// SPDX-License-Identifier: Apache-2.0
//! **The `sector` backend is reachable from the CLI on a native `rbs` file,
//! and agrees with the dense statevector on it.**
//!
//! Two things this pins that the crate-level tests cannot:
//!
//! 1. The QASM path. `rbs` is not a qelib1 spelling; it is readable bare only
//!    under `--qasm-dialect lenient`, and it must lower to ONE native op —
//!    the emitter's `h/cz/ry` expansion contains `h`, which the sector
//!    backend refuses. Without this, the one engine built for hopping
//!    circuits could never be fed one from a file.
//! 2. The default dialect refuses it with the dialect message, not a bare
//!    `unknown gate`, so the way out (`lenient`, or a `gate` definition) is
//!    named.

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

/// One electron in mode 0, hopped part-way along a 4-mode chain.
const HOP_QASM: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n\
x q[0];\nrbs(0.7) q[0],q[1];\nrbs(0.3) q[1],q[2];\n";

/// Tagged per test: the harness runs tests in parallel and each one removes
/// its own fixture, so a shared path is a race.
fn write_fixture(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "omega_sector_smoke_{}_{tag}.qasm",
        std::process::id()
    ));
    std::fs::write(&path, HOP_QASM).expect("write fixture");
    path
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

/// Last `<O> = x` line of the expectation-mode text output.
fn expectation(args: &[&str]) -> f64 {
    let out = run(args);
    assert!(
        out.status.success(),
        "omega-run {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.contains("<O> ="))
        .unwrap_or_else(|| panic!("no `<O> =` line in output:\n{stdout}"));
    line.split('=')
        .nth(1)
        .unwrap()
        .trim()
        .parse()
        .expect("parse <O>")
}

#[test]
fn sector_matches_statevector_on_a_native_rbs_chain() {
    let path = write_fixture("agree");
    let file = path.to_str().unwrap();
    // One particle among four modes: Σ⟨Z_i⟩ = n_modes − 2·N = 2.
    let mut z_sum = 0.0;
    for ob in ["Z0", "Z1", "Z2", "Z3"] {
        let sector = expectation(&[
            file,
            "--backend",
            "sector",
            "--qasm-dialect",
            "lenient",
            "--expectation",
            ob,
        ]);
        let dense = expectation(&[
            file,
            "--backend",
            "statevector",
            "--qasm-dialect",
            "lenient",
            "--expectation",
            ob,
        ]);
        assert!(
            (sector - dense).abs() < 1e-9,
            "{ob}: sector {sector} vs statevector {dense}"
        );
        z_sum += sector;
    }
    assert!(
        (z_sum - 2.0).abs() < 1e-9,
        "particle number not conserved: Σ<Z> = {z_sum}"
    );
    // Mode 3 is never touched: exactly empty.
    let z3 = expectation(&[
        file,
        "--backend",
        "sector",
        "--qasm-dialect",
        "lenient",
        "--expectation",
        "Z3",
    ]);
    assert!(
        (z3 - 1.0).abs() < 1e-12,
        "untouched mode must be empty, got <Z3> = {z3}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn default_dialect_refuses_bare_rbs_and_names_the_way_out() {
    let path = write_fixture("refuse");
    let file = path.to_str().unwrap();
    let out = run(&[file, "--backend", "sector", "--expectation", "Z0"]);
    assert!(
        !out.status.success(),
        "a bare rbs must be refused in the default dialect"
    );
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(msg.contains("`rbs` is not defined in this file"), "{msg}");
    assert!(msg.contains("lenient"), "must name the way out: {msg}");
    assert!(!msg.contains("unknown gate"), "{msg}");
    let _ = std::fs::remove_file(&path);
}
