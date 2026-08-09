// SPDX-License-Identifier: Apache-2.0
//! The Qiskit `expectation` bridge mode — exact ⟨O⟩, no shots.
//!
//! This is the anchor for the N-way **expectation** lane (`FIXES_PLAN.md`
//! Part K step 4), so its conventions have to be pinned harder than usual:
//! an anchor that is quietly wrong turns every engine's row red and sends the
//! reader hunting in the wrong crate.

#![cfg(feature = "bridge-qiskit")]

use omega_bridges::{expectation_qasm2, Backend, BridgeError, WireObservable};
use std::path::PathBuf;

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}
fn venv_python() -> PathBuf {
    runner_dir().join(".venv-qiskit").join("bin").join("python")
}
fn force_env() {
    std::env::set_var(
        "OMEGA_BRIDGE_QISKIT_CMD",
        runner_dir().join("omega-bridge-qiskit-runner"),
    );
}
fn obs(s: &str) -> WireObservable {
    vec![(s.to_string(), 1.0)]
}
const HDR: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";

macro_rules! skip_without_venv {
    () => {
        if !venv_python().exists() {
            eprintln!(
                "Qiskit venv missing at {} — skipping. Build with \
                 `make -C crates/omega-bridges/python qiskit-venv`.",
                venv_python().display()
            );
            return;
        }
        force_env();
    };
}

/// **The ordering pin.** Wire strings are LSB-first: leftmost char = qubit 0.
///
/// Qiskit's own `SparsePauliOp` is MSB-first, so the runner reverses. Getting
/// that backwards is invisible on `ZZ`, `XX`, or any palindromic term — which
/// is most of the obvious test cases, and is exactly how the counts lane
/// shipped a reversed-order bug past its own first test. So this uses `x q[0]`
/// on two qubits, where the two conventions give opposite answers.
#[test]
fn wire_observables_are_lsb_first() {
    skip_without_venv!();
    let qasm = format!("{HDR}qreg q[2];\nx q[0];");
    let vals = expectation_qasm2(Backend::Qiskit, &qasm, &[obs("ZI"), obs("IZ")])
        .expect("qiskit expectation");
    assert_eq!(vals.len(), 2);
    assert!(
        (vals[0] + 1.0).abs() < 1e-12,
        "\"ZI\" must name qubit 0 (which X flipped) -> -1, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 1.0).abs() < 1e-12,
        "\"IZ\" must name qubit 1 (untouched) -> +1, got {}",
        vals[1]
    );
}

/// Terminal measurements are stripped; multi-term observables sum correctly.
#[test]
fn terminal_measures_are_elided_and_terms_sum() {
    skip_without_venv!();
    let qasm = format!("{HDR}qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;");
    let combo: WireObservable = vec![("ZZ".into(), 0.5), ("XX".into(), 0.5)];
    let vals = expectation_qasm2(Backend::Qiskit, &qasm, &[obs("ZZ"), obs("XX"), combo])
        .expect("qiskit expectation");
    for (i, want) in [1.0, 1.0, 1.0].iter().enumerate() {
        assert!(
            (vals[i] - want).abs() < 1e-12,
            "value {i} = {} , expected {want}",
            vals[i]
        );
    }
}

/// A `reset` must be REFUSED, not answered.
///
/// Not defensive programming: `Statevector.from_instruction` on an entangled
/// reset returns **one stochastic trajectory**, measured as two distinct
/// states over 30 runs of Bell + `reset q[0]`. Answering would put a
/// nondeterministic value into the matrix, which reads as a flaky backend.
#[test]
fn reset_is_refused_because_the_anchor_would_be_nondeterministic() {
    skip_without_venv!();
    let qasm = format!("{HDR}qreg q[2];\nh q[0];\ncx q[0],q[1];\nreset q[0];");
    match expectation_qasm2(Backend::Qiskit, &qasm, &[obs("ZI")]) {
        Err(BridgeError::CannotExpress(_, msg)) => {
            assert!(msg.contains("reset"), "message should name reset: {msg}")
        }
        other => panic!("expected CannotExpress, got {other:?}"),
    }
}

/// A classically-conditioned gate is likewise refused.
#[test]
fn a_conditioned_gate_is_refused() {
    skip_without_venv!();
    let qasm = format!(
        "{HDR}qreg q[2];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\nif (c==1) x q[1];"
    );
    match expectation_qasm2(Backend::Qiskit, &qasm, &[obs("IZ")]) {
        Err(BridgeError::CannotExpress(_, msg)) => assert!(
            msg.contains("conditioned") || msg.contains("mixture"),
            "unhelpful message: {msg}"
        ),
        other => panic!("expected CannotExpress, got {other:?}"),
    }
}

/// Every non-Qiskit backend reports `CannotExpress`, not `NotCompiled`.
///
/// The distinction is the whole point of the step-2 taxonomy: the feature IS
/// compiled, the protocol simply carries no expectation path for those
/// backends. Reporting it as "not installed" would claim an environmental
/// excuse for a capability gap.
#[test]
fn other_backends_report_cannot_express_not_not_installed() {
    for b in [Backend::Perceval, Backend::Bloqade, Backend::Tsim, Backend::Ppvm] {
        let qasm = format!("{HDR}qreg q[1];\nh q[0];");
        match expectation_qasm2(b, &qasm, &[obs("Z")]) {
            Err(BridgeError::CannotExpress(got, _)) => assert_eq!(got, b),
            other => panic!("{b:?}: expected CannotExpress, got {other:?}"),
        }
    }
}

/// An empty observable list is rejected up front rather than spawning Python
/// to return an empty array.
#[test]
fn empty_observable_list_is_rejected() {
    let qasm = format!("{HDR}qreg q[1];\nh q[0];");
    assert!(matches!(
        expectation_qasm2(Backend::Qiskit, &qasm, &[]),
        Err(BridgeError::InvalidInput(_))
    ));
}
