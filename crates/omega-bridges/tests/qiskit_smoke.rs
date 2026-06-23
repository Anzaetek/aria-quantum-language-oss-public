//! End-to-end smoke for the Qiskit bridge subprocess.
//!
//! Skipped automatically when the operator hasn't built the runner
//! venv (`make -C crates/omega-bridges/python qiskit-venv`). Detection
//! is conservative: we look for the wrapper script and the venv
//! python; if either is missing, the test prints a notice and exits
//! green so `cargo test --workspace` doesn't require Python on every
//! contributor's machine.
//!
//! Without `--features bridge-qiskit` this file is a no-op stub.

#![cfg(feature = "bridge-qiskit")]

use omega_bridges::{run_qasm2, Backend, BridgeError};
use std::path::PathBuf;

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}

fn venv_python() -> PathBuf {
    runner_dir().join(".venv-qiskit").join("bin").join("python")
}

fn wrapper() -> PathBuf {
    runner_dir().join("omega-bridge-qiskit-runner")
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
fn qiskit_runner_skip_when_no_venv() {
    if venv_python().exists() {
        eprintln!("Qiskit venv present — running the live test instead");
        live_qiskit_bell();
        return;
    }
    if !wrapper().exists() {
        eprintln!(
            "wrapper script missing at {} — repo broken?",
            wrapper().display()
        );
        return;
    }
    eprintln!(
        "Skipping live Qiskit smoke: no venv at {}. \
         Build it with `make -C crates/omega-bridges/python qiskit-venv` to run.",
        venv_python().display()
    );

    // Even without the venv, the dispatcher must surface a typed
    // error rather than panic. Forcing the wrapper path so PATH
    // discovery doesn't accidentally pick up a system python.
    std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", wrapper());
    let res = run_qasm2(Backend::Qiskit, BELL_QASM, 64, None);
    match res {
        Err(BridgeError::Unavailable(b, msg)) => {
            assert_eq!(b, Backend::Qiskit);
            // Either the wrapper itself complained (no python) or the
            // runner reported `qiskit-not-installed`. Both are
            // Unavailable from the operator's perspective.
            assert!(
                msg.contains("python")
                    || msg.contains("qiskit")
                    || msg.contains("Qiskit")
                    || msg.contains("Errno"),
                "expected python/qiskit hint in Unavailable msg, got: {}",
                msg
            );
        }
        Ok(c) => panic!("got counts without venv: {:?}", c),
        Err(e) => panic!("expected Unavailable, got {:?}", e),
    }
}

fn live_qiskit_bell() {
    std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", wrapper());
    let counts = run_qasm2(Backend::Qiskit, BELL_QASM, 4096, None)
        .expect("Qiskit runner must succeed when the venv is present");
    let total: u32 = counts.values().sum();
    assert_eq!(total, 4096, "shots should be conserved");
    let p00 = *counts.get("00").unwrap_or(&0);
    let p11 = *counts.get("11").unwrap_or(&0);
    let on_diag = p00 + p11;
    assert!(
        on_diag as f64 / total as f64 > 0.9,
        "Bell state should put ≥ 90% of shots on |00⟩ + |11⟩; got {}/{} ({:?})",
        on_diag,
        total,
        counts
    );
}

#[test]
fn qasm2_to_qpy_round_trips_through_pure_rust_reader() {
    // Skip when the venv isn't built — same convention as the bell
    // smoke above. Without Qiskit on the host there's nothing to
    // test on the writer side.
    if !venv_python().exists() {
        eprintln!(
            "Skipping qasm2_to_qpy round-trip: no venv at {}. \
             Build it with `make -C crates/omega-bridges/python qiskit-venv`.",
            venv_python().display()
        );
        return;
    }
    std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", wrapper());

    // Hand a Bell circuit's QASM2 to the runner; it returns the QPY
    // blob. Then feed the same blob into omega's pure-Rust QPY
    // reader and assert the resulting CircuitIR matches what the
    // QASM2 source would lower to (h on q0; cx q0, q1).
    let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"#;
    let qpy_bytes =
        omega_bridges::qasm2_to_qpy(qasm).expect("qasm2_to_qpy must succeed when venv is present");
    assert!(
        omega_bridges::is_qpy(&qpy_bytes),
        "qasm2_to_qpy must produce a blob that starts with the QISKIT magic"
    );

    let ir = omega_bridges::qpy::read_qpy_circuit_ir(&qpy_bytes)
        .expect("pure-Rust QPY reader must consume the just-written blob");
    use omega_core::circuit::{GateKind, Qubit};
    assert_eq!(ir.num_qubits, 2);
    assert_eq!(ir.ops.len(), 2);
    assert_eq!(ir.ops[0].gate, GateKind::H);
    assert_eq!(&ir.ops[0].qubits[..], &[Qubit(0)]);
    assert_eq!(ir.ops[1].gate, GateKind::CX);
    assert_eq!(&ir.ops[1].qubits[..], &[Qubit(0), Qubit(1)]);
}

#[test]
fn qasm2_to_qpy_rejects_empty_source() {
    // No venv needed — the InvalidInput guard is in the Rust side
    // before any subprocess work.
    let res = omega_bridges::qasm2_to_qpy("");
    match res {
        Err(BridgeError::InvalidInput(msg)) => {
            assert!(msg.contains("non-empty"), "got msg: {msg}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn qasm2_to_qpy_round_trips_3q_ghz() {
    // Tighten the writer's coverage past the 2-qubit Bell shape:
    // GHZ has a 3-qubit qreg, three gates total, and a longer
    // entangling chain. Pins that the QPY writer's circuit-header
    // (num_qubits = 3) + instruction args (cx on (q1, q2)) round
    // trip through the pure-Rust reader.
    if !venv_python().exists() {
        eprintln!("Skipping 3q GHZ round-trip: no venv");
        return;
    }
    std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", wrapper());

    let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
"#;
    let qpy_bytes = omega_bridges::qasm2_to_qpy(qasm).expect("qasm2_to_qpy must succeed for GHZ");
    assert!(omega_bridges::is_qpy(&qpy_bytes));

    let ir = omega_bridges::qpy::read_qpy_circuit_ir(&qpy_bytes)
        .expect("pure-Rust QPY reader must decode the GHZ blob");
    use omega_core::circuit::{GateKind, Qubit};
    assert_eq!(ir.num_qubits, 3);
    assert_eq!(ir.ops.len(), 3);
    assert_eq!(ir.ops[0].gate, GateKind::H);
    assert_eq!(&ir.ops[0].qubits[..], &[Qubit(0)]);
    assert_eq!(ir.ops[1].gate, GateKind::CX);
    assert_eq!(&ir.ops[1].qubits[..], &[Qubit(0), Qubit(1)]);
    assert_eq!(ir.ops[2].gate, GateKind::CX);
    assert_eq!(&ir.ops[2].qubits[..], &[Qubit(1), Qubit(2)]);
}

#[test]
fn qasm2_to_qpy_round_trips_rx_with_concrete_angle() {
    // Tightens coverage past the parameter-less Bell + GHZ fixtures:
    // an `rx(theta) q[0];` source forces the writer to encode an
    // INSTRUCTION_PARAM payload (concrete float, b'f' type byte).
    // The pure-Rust reader's float-param decoder then has to match
    // what Qiskit wrote. f64 IEEE binary is exact across the
    // round-trip, so `assert_eq!` on the resolved value is safe.
    if !venv_python().exists() {
        eprintln!("Skipping Rx round-trip: no venv");
        return;
    }
    std::env::set_var("OMEGA_BRIDGE_QISKIT_CMD", wrapper());

    let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(0.7853981633974483) q[0];
"#;
    let qpy_bytes = omega_bridges::qasm2_to_qpy(qasm).expect("qasm2_to_qpy must succeed for Rx");
    assert!(omega_bridges::is_qpy(&qpy_bytes));

    let ir = omega_bridges::qpy::read_qpy_circuit_ir(&qpy_bytes)
        .expect("pure-Rust QPY reader must decode the Rx blob");
    use omega_core::circuit::{GateKind, ParamExpr, Qubit};
    assert_eq!(ir.num_qubits, 1);
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(ir.ops[0].gate, GateKind::Rx);
    assert_eq!(&ir.ops[0].qubits[..], &[Qubit(0)]);
    assert_eq!(ir.ops[0].params.len(), 1);
    match &ir.ops[0].params[0] {
        ParamExpr::Concrete(v) => {
            assert!(
                (v - std::f64::consts::FRAC_PI_4).abs() < 1e-12,
                "expected π/4, got {v}"
            );
        }
        other => panic!("expected ParamExpr::Concrete, got {other:?}"),
    }
}
