// SPDX-License-Identifier: Apache-2.0
//! Stim's tableau as the **same-algorithm** anchor for `omega-backend-pauli`.
//!
//! Stim is *the* reference stabilizer simulator, so a disagreement on a
//! Clifford circuit is a defect by construction rather than a modelling
//! choice. `peek_observable_expectation` returns exact integers, so the
//! comparison carries no float slack at all.
//!
//! Reached through the tsim bridge because `bloqade-tsim` already puts Stim in
//! that venv — but it is **plain Stim**, not tsim's ZX engine. See
//! `tsim::expectation`.

#![cfg(all(feature = "bridge-tsim", feature = "bridge-qiskit"))]

use omega_bridges::{expectation_qasm2, Backend, BridgeError, WireObservable};
use std::path::PathBuf;

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}
fn venv(slug: &str) -> PathBuf {
    runner_dir().join(format!(".venv-{slug}")).join("bin").join("python")
}
fn force(slug: &str) {
    std::env::set_var(
        format!("OMEGA_BRIDGE_{}_CMD", slug.to_ascii_uppercase()),
        runner_dir().join(format!("omega-bridge-{slug}-runner")),
    );
}
fn obs(s: &str) -> WireObservable {
    vec![(s.to_string(), 1.0)]
}
const HDR: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";

macro_rules! skip_unless {
    ($($slug:expr),+) => {
        $(if !venv($slug).exists() {
            eprintln!("{} venv missing — skipping", $slug);
            return;
        } force($slug);)+
    };
}

/// **The Clifford guard, and the single most important test in this file.**
///
/// Plain Stim ACCEPTS our tag dialect and executes the base gate: `S[T] 0`
/// applies `S`, not `T`; `I[R_Z(theta=…)] 0` applies identity, not a rotation.
/// It does not refuse. So a non-Clifford circuit anchored on plain Stim would
/// return a confidently wrong reference and blame our backend for the gap.
///
/// Verified here by planting a `t` gate and requiring a typed refusal.
#[test]
fn a_non_clifford_circuit_is_refused_not_silently_mis_executed() {
    skip_unless!("tsim");
    let qasm = format!("{HDR}qreg q[1];\nh q[0];\nt q[0];");
    match expectation_qasm2(Backend::Tsim, &qasm, &[obs("Z")]) {
        Err(BridgeError::CannotExpress(_, msg)) => {
            assert!(
                msg.contains("not Clifford") && msg.contains("S[T]"),
                "the refusal must name the tagged instruction so the cause is \
                 obvious; got: {msg}"
            );
        }
        Ok(v) => panic!(
            "Stim returned {v:?} for a non-Clifford circuit — it executed the \
             tag's BASE gate. This is the silent-wrong-answer case the guard exists \
             to prevent."
        ),
        Err(e) => panic!("expected CannotExpress, got {e:?}"),
    }
}

/// Ordering pin, asymmetric. `stim.PauliString` is LSB-first like our wire
/// format; Qiskit's `SparsePauliOp` is MSB-first and its runner reverses.
#[test]
fn stim_reads_pauli_strings_lsb_first() {
    skip_unless!("tsim");
    let qasm = format!("{HDR}qreg q[2];\nx q[0];");
    let v = expectation_qasm2(Backend::Tsim, &qasm, &[obs("ZI"), obs("IZ")]).expect("stim");
    assert!((v[0] + 1.0).abs() < 1e-12, "\"ZI\" names q0 -> -1, got {}", v[0]);
    assert!((v[1] - 1.0).abs() < 1e-12, "\"IZ\" names q1 -> +1, got {}", v[1]);
}

/// Stim and Qiskit agree exactly on Clifford circuits — including `sx`/`sxdg`,
/// which Stim knows natively as `SQRT_X` / `SQRT_X_DAG`.
///
/// The tolerance is 0.0, not an epsilon: Stim's values are integers and
/// Qiskit's are exact on these states. Anything nonzero is a real discrepancy.
#[test]
fn stim_and_qiskit_agree_exactly_on_clifford_circuits() {
    skip_unless!("tsim", "qiskit");
    let cases: [(&str, &[&str]); 5] = [
        ("qreg q[1];\nh q[0];", &["X", "Y", "Z"]),
        ("qreg q[1];\nsx q[0];", &["X", "Y", "Z"]),
        ("qreg q[1];\nsxdg q[0];", &["X", "Y", "Z"]),
        ("qreg q[2];\nh q[0];\ncx q[0],q[1];", &["ZZ", "XX", "YY", "ZI", "IZ", "XY"]),
        ("qreg q[2];\nh q[0];\ns q[0];\ncx q[0],q[1];\nsxdg q[1];",
         &["ZZ", "XI", "IY", "XY", "YX", "ZX"]),
    ];
    let mut compared = 0;
    let mut worst = 0.0_f64;
    for (body, observables) in cases {
        let qasm = format!("{HDR}{body}");
        let wire: Vec<WireObservable> = observables.iter().map(|o| obs(o)).collect();
        let q = expectation_qasm2(Backend::Qiskit, &qasm, &wire).expect("qiskit");
        let s = match expectation_qasm2(Backend::Tsim, &qasm, &wire) {
            Ok(v) => v,
            Err(BridgeError::CannotExpress(_, m)) => {
                panic!("{body} is Clifford but Stim refused it: {m}")
            }
            Err(e) => panic!("stim failed on {body}: {e}"),
        };
        for (i, o) in observables.iter().enumerate() {
            let d = (s[i] - q[i]).abs();
            worst = worst.max(d);
            assert!(d < 1e-12, "{body} / {o}: stim {} vs qiskit {}", s[i], q[i]);
            compared += 1;
        }
    }
    eprintln!("stim vs qiskit: {compared} (circuit, observable) pairs, worst |Δ| = {worst:.3e}");
    assert!(compared >= 18, "only {compared} cells compared — coverage collapsed");
}

/// Stim's values really are integers — the property that makes this anchor
/// exact rather than float-limited.
///
/// Guards against a future "improvement" routing through `state_vector()`,
/// which is `complex64` (measured error 1.21e-08 on `1/√2`) and would silently
/// turn an exact anchor into a 1e-7 one.
#[test]
fn stim_returns_exact_integers() {
    skip_unless!("tsim");
    let qasm = format!("{HDR}qreg q[2];\nh q[0];\ncx q[0],q[1];");
    let v = expectation_qasm2(Backend::Tsim, &qasm, &[obs("ZZ"), obs("ZI"), obs("XX")])
        .expect("stim");
    for (i, x) in v.iter().enumerate() {
        assert_eq!(
            *x,
            x.round(),
            "value {i} = {x} is not an exact integer; a float-valued Stim path \
             (e.g. state_vector(), which is complex64) has crept in"
        );
    }
}
