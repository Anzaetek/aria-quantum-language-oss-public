// SPDX-License-Identifier: Apache-2.0
//! **Our Pauli-propagation backend against an independent implementation of the
//! same algorithm.**
//!
//! `ppvm` is an external Rust implementation of Heisenberg-picture Pauli
//! propagation, citing the same source material as ours (arXiv:2505.21606 /
//! PauliPropagation.jl). That makes it the *same-algorithm* anchor —
//! a disagreement points at one of the two implementations rather than at a
//! modelling choice, which is exactly what a dense-statevector oracle cannot
//! tell you.
//!
//! # The gap this closes
//!
//! `ppvm_expectation.rs` opens with "ppvm's `PauliSum` as the **same-algorithm**
//! anchor for `pauliprop`". It compares **ppvm against Qiskit**. Our backend is
//! never constructed in it — grep for `PauliPropBackend` across
//! `crates/omega-bridges/` and the only hits are that doc comment and the
//! bridge source.
//!
//! So the anchor existed, was described as anchoring our backend, and did not.
//! Our `pauliprop` was validated only against a dense oracle built inside its
//! own test file — which catches arithmetic errors but cannot catch a shared
//! misreading of the algorithm, because there is no second reading present.
//!
//! # What a dense oracle cannot see, and this can
//!
//! Both implementations propagate observables backwards through the circuit and
//! truncate a Pauli sum. A convention error in *that* representation — the
//! conjugation direction, the Pauli-string bit order, the sign of an odd-Y
//! term — can be self-consistent inside one implementation. Comparing against a
//! statevector catches it only if the oracle is written independently enough,
//! and the oracle here lives in the same file as the backend.
//!
//! Two conventions are pinned explicitly below because both have bitten this
//! repository before: the LSB-first wire order (Qiskit's `SparsePauliOp` is
//! MSB-first internally, so its runner reverses and ppvm's must not), and the
//! Heisenberg reverse-order application.

#![cfg(feature = "bridge-ppvm")]

use omega_backend_pauliprop::PauliPropBackend;
use omega_bridges::{expectation_qasm2, Backend, BridgeError, WireObservable};
use omega_core::executor::{Backend as _, Observable};
use omega_core::params::ParameterBinding;
use std::path::PathBuf;

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}
fn venv(slug: &str) -> PathBuf {
    runner_dir()
        .join(format!(".venv-{slug}"))
        .join("bin")
        .join("python")
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

/// Dense wire form (`"ZI"`, LSB-first) -> our indexed form (`"Z0"`).
///
/// The two layers spell observables differently: the bridge wire is a DENSE
/// string with one character per qubit, ours is a sparse indexed list. Both are
/// LSB-first, so character `i` of the dense string is qubit `i` — that is the
/// convention `both_implementations_read_pauli_strings_lsb_first` pins, and
/// getting it backwards here would silently compare a different observable on
/// each side and could still "agree" on symmetric strings like `ZZ`.
fn dense_to_indexed(dense: &str) -> String {
    dense
        .chars()
        .enumerate()
        .filter(|(_, c)| *c != 'I')
        .map(|(i, c)| format!("{c}{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Our backend's expectation for one QASM2 body and one dense Pauli string.
fn ours(body: &str, pauli: &str) -> Result<f64, String> {
    let ir = omega_parser::lower_to_ir(&format!("{HDR}{body}"))
        .map_err(|e| format!("lower: {e}"))?;
    let indexed = dense_to_indexed(pauli);
    let o = Observable::parse(&indexed).map_err(|e| format!("observable {indexed:?}: {e}"))?;
    PauliPropBackend::new()
        .expectation(&ir, &ParameterBinding::new(), &o)
        .map_err(|e| e.to_string())
}

/// Circuits both implementations accept, spanning the shapes where a
/// convention error would hide.
///
/// Chosen so that no case is symmetric under the errors being hunted:
/// single-qubit rotations that distinguish X from Y, a Bell pair for two-qubit
/// strings, an asymmetric two-qubit circuit for qubit ORDER, and mixed strings
/// like `XY` where swapping the two factors changes the answer.
const CASES: &[(&str, &[&str])] = &[
    ("qreg q[1];\nh q[0];", &["X", "Y", "Z"]),
    ("qreg q[1];\nrx(0.7) q[0];", &["X", "Y", "Z"]),
    ("qreg q[1];\nry(0.9) q[0];\nrz(0.4) q[0];", &["X", "Y", "Z"]),
    ("qreg q[2];\nh q[0];\ncx q[0],q[1];", &["ZZ", "XX", "YY", "ZI", "IZ", "XY"]),
    (
        "qreg q[2];\nry(0.7) q[0];\ncx q[0],q[1];\nrz(1.1) q[1];",
        &["ZZ", "XI", "IY", "XY", "YX", "ZX"],
    ),
    (
        "qreg q[3];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\nrz(0.55) q[2];",
        &["ZZZ", "XXX", "ZII", "IIZ", "XYZ"],
    ),
];

/// The claim: two independent implementations of the same algorithm agree.
#[test]
fn our_pauliprop_agrees_with_ppvm_on_the_corpus() {
    if !venv("ppvm").exists() {
        eprintln!("ppvm venv missing — skipping the same-algorithm comparison");
        return;
    }
    force("ppvm");

    let (mut compared, mut worst) = (0usize, 0.0f64);
    let mut skipped: Vec<String> = Vec::new();

    for (body, observables) in CASES {
        let qasm = format!("{HDR}{body}");
        let wire: Vec<WireObservable> = observables.iter().map(|o| obs(o)).collect();
        let theirs = match expectation_qasm2(Backend::Ppvm, &qasm, &wire) {
            Ok(v) => v,
            // A typed refusal is a correct answer, not a failure — but it is
            // recorded, because a comparison that quietly skips everything
            // reads exactly like one that passed.
            Err(BridgeError::CannotExpress(_, msg)) => {
                skipped.push(format!("{body} (ppvm: {msg})"));
                continue;
            }
            Err(e) => panic!("ppvm failed on {body}: {e}"),
        };

        for (i, pauli) in observables.iter().enumerate() {
            let mine = match ours(body, pauli) {
                Ok(v) => v,
                Err(e) => {
                    skipped.push(format!("{body} / {pauli} (ours: {e})"));
                    continue;
                }
            };
            let d = (mine - theirs[i]).abs();
            worst = worst.max(d);
            assert!(
                d < 1e-9,
                "{body} / {pauli}: ours {mine} vs ppvm {} (|Δ| = {d:.3e}).\n\
                 Two independent implementations of the same algorithm disagree, so \
                 one of them is wrong — this is not a modelling difference.",
                theirs[i]
            );
            compared += 1;
        }
    }

    eprintln!(
        "pauliprop vs ppvm: {compared} (circuit, observable) pairs, worst |Δ| = {worst:.3e}; \
         {} skipped: {skipped:?}",
        skipped.len()
    );
    // Report the qualifying count. A differential check that compared three
    // cells reads like coverage.
    assert!(
        compared >= 20,
        "only {compared} cells compared (skipped: {skipped:?}) — coverage collapsed, \
         and a near-empty comparison is indistinguishable from a passing one"
    );
}

/// **Wire order.** Both sides must read Pauli strings LSB-first.
///
/// `x q[0]` distinguishes them; `ZZ` would not — under a reversed reading `ZZ`
/// is still `ZZ`, so a symmetric string cannot see this at all.
#[test]
fn both_implementations_read_pauli_strings_lsb_first() {
    if !venv("ppvm").exists() {
        eprintln!("ppvm venv missing — skipping");
        return;
    }
    force("ppvm");
    let body = "qreg q[2];\nx q[0];";
    let qasm = format!("{HDR}{body}");
    let theirs = expectation_qasm2(Backend::Ppvm, &qasm, &[obs("ZI"), obs("IZ")]).expect("ppvm");
    let mine = [ours(body, "ZI").expect("ours ZI"), ours(body, "IZ").expect("ours IZ")];

    // `x q[0]` flips qubit 0 only. LSB-first, "ZI" places Z on qubit 0.
    assert!(
        (mine[0] + 1.0).abs() < 1e-12 && (mine[1] - 1.0).abs() < 1e-12,
        "ours reads Pauli strings MSB-first: got {mine:?}, expected [-1, +1]"
    );
    assert!(
        (theirs[0] + 1.0).abs() < 1e-12 && (theirs[1] - 1.0).abs() < 1e-12,
        "ppvm reads Pauli strings MSB-first: got {theirs:?}"
    );
}

/// **Heisenberg order.** Both must apply gates in REVERSE circuit order when
/// propagating the observable.
///
/// The fixture is chosen so the two orders differ: `h; rz(0.9)` with observable
/// `X` gives ≈0.6216 reversed and 1.0 forwards. `rx(a); ry(b)` with `⟨Z⟩ =
/// cos a·cos b` would NOT work — it is symmetric in `a` and `b`, so both orders
/// agree and the test reads as a pass.
#[test]
fn both_implementations_propagate_in_heisenberg_order() {
    if !venv("ppvm").exists() {
        eprintln!("ppvm venv missing — skipping");
        return;
    }
    force("ppvm");
    let body = "qreg q[1];\nh q[0];\nrz(0.9) q[0];";
    let qasm = format!("{HDR}{body}");
    let theirs = expectation_qasm2(Backend::Ppvm, &qasm, &[obs("X")]).expect("ppvm");
    let mine = ours(body, "X").expect("ours");

    let reversed = 0.9f64.cos(); // ≈ 0.6216
    assert!(
        (mine - reversed).abs() < 1e-9,
        "ours gave {mine}, expected cos(0.9) = {reversed} — 1.0 would mean the \
         gates were applied in FORWARD order"
    );
    assert!(
        (theirs[0] - reversed).abs() < 1e-9,
        "ppvm gave {}, expected {reversed}",
        theirs[0]
    );
}

/// **Guard the guard.** The corpus must be able to tell the two implementations
/// apart at all — otherwise agreement is vacuous.
///
/// Every case must produce at least one expectation that is neither 0 nor ±1:
/// those three values are what a stub, a sign error, or a dropped rotation all
/// tend to produce, so a corpus made only of them would agree under several
/// wrong implementations.
#[test]
fn the_corpus_produces_non_trivial_values() {
    let mut nontrivial = 0usize;
    for (body, observables) in CASES {
        for pauli in *observables {
            if let Ok(v) = ours(body, pauli) {
                if v.abs() > 1e-6 && (v.abs() - 1.0).abs() > 1e-6 {
                    nontrivial += 1;
                }
            }
        }
    }
    assert!(
        nontrivial >= 5,
        "only {nontrivial} of the corpus's expectations are strictly between 0 and 1 \
         in magnitude; a corpus of 0s and ±1s agrees under a sign error or a dropped \
         rotation and proves little"
    );
}
