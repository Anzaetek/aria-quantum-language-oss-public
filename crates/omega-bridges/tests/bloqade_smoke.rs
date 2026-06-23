//! End-to-end smoke for the Bloqade (QuEra Aquila) bridge subprocess.
//!
//! Skipped automatically when the operator hasn't built the runner
//! venv (`make -C crates/omega-bridges/python bloqade-venv`).
//! Detection mirrors the Qiskit smoke: we look for the wrapper script
//! and the venv python; if either is missing, the test prints a
//! notice and exits green so `cargo test --workspace` doesn't require
//! Python on every contributor's machine.
//!
//! With the venv present, this asserts a Bell-state distribution
//! through `bloqade.pyqrack.StackMemorySimulator`: 1024 shots split
//! into `00` / `11` outcomes within a generous 25% statistical band
//! around 50/50.
//!
//! Without `--features bridge-bloqade` this file is a no-op stub.

#![cfg(feature = "bridge-bloqade")]

use omega_bridges::{run_qasm2, Backend, BridgeError};
use std::path::PathBuf;

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}

fn venv_python() -> PathBuf {
    runner_dir()
        .join(".venv-bloqade")
        .join("bin")
        .join("python")
}

fn wrapper() -> PathBuf {
    runner_dir().join("omega-bridge-bloqade-runner")
}

const BELL_QASM: &str = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"#;

#[test]
fn bloqade_runner_skip_when_no_venv() {
    if !wrapper().exists() {
        eprintln!(
            "wrapper script missing at {} — repo broken?",
            wrapper().display()
        );
        return;
    }
    std::env::set_var("OMEGA_BRIDGE_BLOQADE_CMD", wrapper());

    if venv_python().exists() {
        eprintln!("Bloqade venv present — asserting the Bell-state distribution");
        live_bloqade_bell();
        return;
    }

    eprintln!(
        "Skipping live Bloqade smoke: no venv at {}. \
         Build it with `make -C crates/omega-bridges/python bloqade-venv` to run.",
        venv_python().display()
    );

    let res = run_qasm2(Backend::Bloqade, BELL_QASM, 64, None);
    match res {
        Err(BridgeError::Unavailable(b, msg)) => {
            assert_eq!(b, Backend::Bloqade);
            assert!(
                msg.contains("python")
                    || msg.contains("bloqade")
                    || msg.contains("Bloqade")
                    || msg.contains("Errno"),
                "expected python/bloqade hint in Unavailable msg, got: {}",
                msg
            );
        }
        Ok(c) => panic!("got counts without venv: {:?}", c),
        Err(e) => panic!("expected Unavailable, got {:?}", e),
    }
}

/// With a working venv, the runner lifts the QASM2 source into the
/// kirin IR via `bloqade.qasm2.loads(..., returns="c")` and runs
/// `bloqade.pyqrack.StackMemorySimulator.task(...).batch_run(shots)`.
/// Bell state must produce only the `00` and `11` outcomes, each
/// within a 25% statistical band around 50/50 — generous enough that
/// the test doesn't flake but tight enough to catch a wrong-creg or
/// MSB/LSB inversion regression.
fn live_bloqade_bell() {
    const SHOTS: u32 = 1024;
    let res = run_qasm2(Backend::Bloqade, BELL_QASM, SHOTS, None);
    let counts = match res {
        Ok(c) => c,
        Err(BridgeError::Unavailable(b, msg)) => {
            // The local install may still be broken (e.g. operator
            // updated pyqrack without re-pinning). Surface a clear
            // skip rather than a hard failure so flaky local
            // environments don't gate `cargo test`.
            assert_eq!(b, Backend::Bloqade);
            eprintln!("Bloqade install reports unavailable: {msg} — skipping live assertion");
            return;
        }
        Err(e) => panic!("expected Ok counts or Unavailable, got {e:?}"),
    };

    let total: u32 = counts.values().copied().sum();
    assert!(
        total.abs_diff(SHOTS) <= 1,
        "counts total {total} should equal shots {SHOTS} (±1 for rounding)"
    );

    // The Bell state |00⟩+|11⟩ never produces |01⟩ or |10⟩.
    for (bits, n) in &counts {
        match bits.as_str() {
            "00" | "11" => {}
            other => panic!("unexpected outcome {other:?} count={n}"),
        }
    }

    let zero = *counts.get("00").expect("00 must appear in Bell counts");
    let one = *counts.get("11").expect("11 must appear in Bell counts");
    let half = SHOTS as f64 / 2.0;
    let band = 0.25 * SHOTS as f64;
    assert!(
        (zero as f64 - half).abs() < band,
        "00 count {zero} too far from {half}"
    );
    assert!(
        (one as f64 - half).abs() < band,
        "11 count {one} too far from {half}"
    );
}
