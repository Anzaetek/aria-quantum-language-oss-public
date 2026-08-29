// SPDX-License-Identifier: Apache-2.0
//! **The Aria front end must now ANSWER a feedforward expectation, not refuse
//! it.**
//!
//! `run::expectation` used to call `reject_feedforward_on_analytic_path` and
//! return an error for any circuit containing a `when c == v` gate. The reasoning
//! was sound at the time — the analytic backends evaluated without mid-circuit
//! collapse, so the guard was silently skipped and the answer was "plausible but
//! wrong" — and the error told the caller to *"rewrite the feedforward as
//! coherent control"*.
//!
//! `omega_core::defer_measure` now performs exactly that rewrite automatically,
//! so the refusal described a limitation that no longer existed. This file is
//! what makes its removal a tested change rather than a deletion: without it,
//! dropping the guard would be indistinguishable from dropping the check that
//! the answer is right.
//!
//! # Why these three observables
//!
//! `⟨Z₁⟩` and `⟨Z₀Z₁⟩` are diagonal and agree between the mixture and the naive
//! pure-state answer for one of them, so neither alone separates the designs.
//! `⟨X₀X₁⟩` is the discriminator: **0** for the mixture, **+1** for a deferral
//! that forgot to dephase, and it needs no tolerance because dephasing deletes
//! the term rather than computing a small number.

use aria_core::ast::parse_aria;
use aria_runtime::run::{expectation, BackendSel};
use std::collections::HashMap;

/// Teleport-style feedforward: `m[0]` drives an X on q1.
///
/// Deferred this is `H q0; CX q0,q1` — a Bell state — and the measurement
/// dephases q0.
const SRC: &str = r#"
circuit FF {
    qreg q[2]
    creg m[1]
    apply H on q[0]
    measure q[0] -> m[0]
    when m[0] == 1 { apply X on q[1] }
}
"#;

/// A measured qubit reused coherently. No deferred form exists.
const REUSED: &str = r#"
circuit RU {
    qreg q[1]
    creg m[1]
    apply H on q[0]
    measure q[0] -> m[0]
    apply H on q[0]
}
"#;

fn circuit(src: &str, name: &str) -> aria_core::ast::nodes::Circuit {
    parse_aria(src)
        .expect("parse")
        .instantiate(name, &[])
        .expect("instantiate")
}

fn expect(src: &str, name: &str, obs: &str) -> Result<f64, String> {
    expectation(&circuit(src, name), obs, &HashMap::new(), BackendSel::Sim)
}

#[test]
fn a_feedforward_circuit_is_answered_with_the_mixture_value() {
    // Both qubits always agree — q1 flips exactly when q0 measured 1.
    let zz = expect(SRC, "FF", "Z0 Z1").expect("feedforward is no longer refused");
    assert!(
        (zz - 1.0).abs() < 1e-12,
        "<Z0Z1> must be +1: the guard fires exactly when q0 is 1, so the two \
         qubits are perfectly correlated. Got {zz}. A value of 0 means the guard \
         was dropped and q1 stayed in |0>."
    );

    // Neither qubit carries a definite value on its own.
    let z1 = expect(SRC, "FF", "Z1").expect("runs");
    assert!(
        z1.abs() < 1e-12,
        "<Z1> must be 0 — the X fires on about half the branches. Got {z1}. \
         +1 means the guard never fired; -1 means it always did."
    );

    // THE discriminator.
    let xx = expect(SRC, "FF", "X0 X1").expect("runs");
    assert_eq!(
        xx, 0.0,
        "<X0X1> must be EXACTLY 0. This circuit is a mixture of |00> and |11>, \
         not a Bell state: the measurement destroyed the coherence between them. \
         +1 here means the measurement was deferred but the observable was never \
         dephased — the naive half of the transformation, which is a wrong answer \
         rather than an approximation."
    );
}

/// The removal of the refusal must not have turned into "answer everything".
///
/// A measured qubit used coherently afterwards has no deferred form, and this is
/// the reset-and-reuse shape — every QEC ancilla. It must still be refused, and
/// the message must name the construct, because a caller can act on that and not
/// on "unsupported".
#[test]
fn a_coherently_reused_measured_qubit_is_still_refused() {
    let err = expect(REUSED, "RU", "Z0")
        .expect_err("a measured qubit used coherently afterwards has no deferred form");
    assert!(
        err.contains("measured") && err.contains("used again"),
        "the refusal must name the construct; got: {err}"
    );
}
