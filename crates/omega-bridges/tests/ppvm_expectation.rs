// SPDX-License-Identifier: Apache-2.0
//! ppvm's `PauliSum` as the **same-algorithm** anchor for `pauliprop`.
//!
//! `docs/BRIDGES.md` has always described ppvm this way — "not a new
//! capability but an **independent numeric reference** that validates
//! pauliprop" — and nothing used it that way until the `expectation` mode
//! existed, because the counts protocol only reaches ppvm's *other* engine.
//!
//! Two anchors answer different questions, and the lane needs both:
//!
//! * **Qiskit `Statevector`** is exact ground truth by a *different* algorithm.
//!   It catches arithmetic errors.
//! * **ppvm `PauliSum`** is an independent implementation of the *same*
//!   algorithm. It catches a shared misunderstanding of the method — the class
//!   of defect ground truth alone would attribute to truncation.
//!
//! Two conventions here fail SILENTLY when reversed, so each has an asymmetric
//! pin rather than a palindromic one.

#![cfg(all(feature = "bridge-ppvm", feature = "bridge-qiskit"))]

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

/// **Ordering pin.** Wire strings are LSB-first on both sides — but Qiskit's
/// `SparsePauliOp` is MSB-first internally, so its runner reverses and ppvm's
/// must not. `x q[0]` distinguishes them; `ZZ` would not.
#[test]
fn both_anchors_read_pauli_strings_the_same_way() {
    skip_unless!("ppvm", "qiskit");
    let qasm = format!("{HDR}qreg q[2];\nx q[0];");
    let q = expectation_qasm2(Backend::Qiskit, &qasm, &[obs("ZI"), obs("IZ")]).expect("qiskit");
    let p = expectation_qasm2(Backend::Ppvm, &qasm, &[obs("ZI"), obs("IZ")]).expect("ppvm");
    assert!((q[0] + 1.0).abs() < 1e-12 && (q[1] - 1.0).abs() < 1e-12, "qiskit: {q:?}");
    assert!((p[0] + 1.0).abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12, "ppvm: {p:?}");
}

/// **Heisenberg-order pin.** ppvm must apply gates in REVERSE circuit order.
///
/// The fixture is chosen so the two orders differ: `h; rz(0.9)` with observable
/// `X` gives 0.6216 reversed and 1.0 forwards. An earlier attempt used
/// `rx(a); ry(b)` with `⟨Z⟩ = cos a·cos b`, which is symmetric in `a` and `b`
/// — both orders agreed and the test read as a pass.
#[test]
fn ppvm_applies_gates_in_heisenberg_order() {
    skip_unless!("ppvm", "qiskit");
    let qasm = format!("{HDR}qreg q[1];\nh q[0];\nrz(0.9) q[0];");
    let p = expectation_qasm2(Backend::Ppvm, &qasm, &[obs("X")]).expect("ppvm");
    assert!(
        (p[0] - 0.6216099682706644).abs() < 1e-9,
        "expected 0.62161 (reverse order); 1.0 would mean forward order. Got {}",
        p[0]
    );
    let q = expectation_qasm2(Backend::Qiskit, &qasm, &[obs("X")]).expect("qiskit");
    assert!((p[0] - q[0]).abs() < 1e-9, "ppvm {} vs qiskit {}", p[0], q[0]);
}

/// The two anchors agree across a spread of circuits and observables.
///
/// Includes `sx`/`sxdg` — ppvm has NATIVE `sqrt_x`/`sqrt_x_dag`, independent
/// support for treating them as first-class Clifford gates.
#[test]
fn ppvm_and_qiskit_agree_on_the_corpus_shapes() {
    skip_unless!("ppvm", "qiskit");
    let cases: [(&str, &[&str]); 5] = [
        ("qreg q[1];\nh q[0];", &["X", "Y", "Z"]),
        ("qreg q[1];\nsx q[0];", &["X", "Y", "Z"]),
        ("qreg q[1];\nsx q[0];\nsxdg q[0];", &["Z"]),
        ("qreg q[2];\nh q[0];\ncx q[0],q[1];", &["ZZ", "XX", "YY", "ZI", "IZ", "XY"]),
        ("qreg q[2];\nry(0.7) q[0];\ncx q[0],q[1];\nrz(1.1) q[1];",
         &["ZZ", "XI", "IY", "XY", "YX", "ZX"]),
    ];
    let mut compared = 0;
    let mut worst = 0.0_f64;
    for (body, observables) in cases {
        let qasm = format!("{HDR}{body}");
        let wire: Vec<WireObservable> = observables.iter().map(|o| obs(o)).collect();
        let q = expectation_qasm2(Backend::Qiskit, &qasm, &wire).expect("qiskit");
        let p = match expectation_qasm2(Backend::Ppvm, &qasm, &wire) {
            Ok(v) => v,
            // A typed refusal is a correct answer, not a failure.
            Err(BridgeError::CannotExpress(_, msg)) => {
                eprintln!("  ppvm cannot express {body}: {msg}");
                continue;
            }
            Err(e) => panic!("ppvm failed on {body}: {e}"),
        };
        for (i, o) in observables.iter().enumerate() {
            let d = (p[i] - q[i]).abs();
            if d > worst {
                worst = d;
            }
            assert!(d < 1e-9, "{body} / {o}: ppvm {} vs qiskit {}", p[i], q[i]);
            compared += 1;
        }
    }
    // Report the qualifying count: a differential check that silently compared
    // three cells reads like coverage (Part K4 trap 5).
    eprintln!("ppvm vs qiskit: {compared} (circuit, observable) pairs, worst |Δ| = {worst:.3e}");
    assert!(compared >= 15, "only {compared} cells compared — coverage collapsed");
}

/// Non-unitary constructs are refused by both, with the same taxonomy.
#[test]
fn both_anchors_refuse_the_same_constructs() {
    skip_unless!("ppvm", "qiskit");
    let reset = format!("{HDR}qreg q[2];\nh q[0];\ncx q[0],q[1];\nreset q[0];");
    for b in [Backend::Qiskit, Backend::Ppvm] {
        match expectation_qasm2(b, &reset, &[obs("ZI")]) {
            Err(BridgeError::CannotExpress(got, _)) => assert_eq!(got, b),
            other => panic!("{b:?}: expected CannotExpress, got {other:?}"),
        }
    }
}
