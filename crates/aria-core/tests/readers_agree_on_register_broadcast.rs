// SPDX-License-Identifier: Apache-2.0
//! **The two QASM readers must accept the same language.**
//!
//! This repository has two independent OpenQASM 2.0 readers:
//!
//! * `aria_core::ast::qasm::from_qasm` — a line-oriented regex reader, behind
//!   the `aria import` subcommand;
//! * `omega_parser::lower_to_ir` — a pest grammar, behind `omega run`, the C
//!   FFI, the WASM host ABI and the HTTP server.
//!
//! They cannot both be linked into one binary (`aria-cli` has no `omega-parser`
//! dependency and `omega-cli` has no `aria-core` one), so nothing forces them
//! to agree and for a long time they did not: `from_qasm` refused register
//! broadcast outright while `omega-parser` expanded it. The same file imported
//! through one binary and errored through the other, and two programs from the
//! specification's own example corpus — `inverseqft2.qasm` and `qpt.qasm` —
//! were refused by `aria import` for a construct that is legal OpenQASM 2.0.
//!
//! # Why agreement is asserted on REFUSALS too
//!
//! Testing only that both accept the legal forms would let them drift on the
//! illegal ones, and that direction is worse. `cz q;` — a two-qubit gate over
//! one two-qubit register — is not defined by OpenQASM, and flattening it into
//! `cz q[0], q[1]` is a confident wrong answer rather than a refusal. Both
//! readers had that bug at some point, and a reader that "helpfully" accepts it
//! is silently executing a circuit the user did not write.
//!
//! # What this test does NOT claim
//!
//! Only that the two readers agree on *acceptance*, not that they produce
//! identical IR — they build different types, and the gate-count comparison
//! below is a coarse structural check rather than an equivalence proof.
//! Deeper agreement is what `crz_reset_spellable.rs` and
//! `every_emitted_gate_is_readable.rs` cover, by round-tripping through both.

/// `(source, what it is)` — every case legal OpenQASM 2.0, so BOTH readers must
/// accept it.
const BOTH_ACCEPT: &[(&str, &str)] = &[
    (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\nh q;\n",
        "single-qubit gate broadcast over a register",
    ),
    (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nh q;\n",
        "bare one-qubit register is the same as q[0]",
    ),
    (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nmeasure q -> c;\n",
        "equal-width measure broadcast",
    ),
    (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\nreset q;\n",
        "reset broadcast",
    ),
    (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n",
        "fully indexed, the form that always worked",
    ),
    (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncreg c[3];\nh q;\nreset q;\nmeasure q -> c;\n",
        "all three broadcast forms in one program",
    ),
];

/// Cases where broadcast has no meaning in OpenQASM. BOTH readers must refuse.
///
/// A reader that accepts these is not being lenient, it is inventing a
/// semantics the specification does not define.
const BOTH_REFUSE: &[(&str, &str)] = &[
    (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncz q;\n",
        "two-qubit gate over one register — NOT cz q[0],q[1]",
    ),
    (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncreg c[2];\nmeasure q -> c;\n",
        "mismatched measure widths — an error, not a truncation",
    ),
    (
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nccx q;\n",
        "three-qubit gate over one register",
    ),
];

#[test]
fn both_readers_accept_every_legal_broadcast_form() {
    let mut disagreements = Vec::new();
    for (src, what) in BOTH_ACCEPT {
        let aria = aria_core::ast::qasm::from_qasm(src);
        let omega = omega_parser::lower_to_ir(src);
        match (&aria, &omega) {
            (Ok(_), Ok(_)) => {}
            (Err(e), Ok(_)) => disagreements.push(format!(
                "{what}: omega-parser accepts it, aria-core REFUSES it — {e}"
            )),
            (Ok(_), Err(e)) => disagreements.push(format!(
                "{what}: aria-core accepts it, omega-parser REFUSES it — {e}"
            )),
            (Err(a), Err(o)) => disagreements.push(format!(
                "{what}: BOTH refuse a legal OpenQASM 2.0 program.\n    aria-core: {a}\n    omega-parser: {o}"
            )),
        }
    }
    assert!(
        disagreements.is_empty(),
        "the two readers disagree on {} legal program(s):\n  {}",
        disagreements.len(),
        disagreements.join("\n  ")
    );
}

#[test]
fn both_readers_refuse_every_meaningless_broadcast() {
    let mut disagreements = Vec::new();
    for (src, what) in BOTH_REFUSE {
        let aria = aria_core::ast::qasm::from_qasm(src);
        let omega = omega_parser::lower_to_ir(src);
        if aria.is_ok() {
            disagreements.push(format!(
                "{what}: aria-core ACCEPTED it. OpenQASM defines no such broadcast, so \
                 this is a circuit the user did not write being executed silently."
            ));
        }
        if omega.is_ok() {
            disagreements.push(format!(
                "{what}: omega-parser ACCEPTED it. OpenQASM defines no such broadcast."
            ));
        }
    }
    assert!(
        disagreements.is_empty(),
        "{} meaningless broadcast(s) accepted:\n  {}",
        disagreements.len(),
        disagreements.join("\n  ")
    );
}

/// The expansion must have the same SHAPE in both, not merely be accepted.
///
/// Acceptance alone would be satisfied by a reader that expanded `h q;` over a
/// 3-qubit register into one gate, or seven. Gate counts are coarse — the two
/// build different IR types and this is not an equivalence proof — but they
/// catch an off-by-one or a dropped tail, which is the realistic failure.
#[test]
fn both_readers_expand_a_broadcast_to_the_same_gate_count() {
    let cases: &[(&str, usize)] = &[
        ("qreg q[3];\nh q;\n", 3),
        ("qreg q[5];\nx q;\n", 5),
        ("qreg q[4];\nreset q;\n", 4),
        ("qreg q[2];\ncreg c[2];\nmeasure q -> c;\n", 2),
    ];
    for (body, want) in cases {
        let src = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n{body}");
        let a = aria_core::ast::qasm::from_qasm(&src)
            .unwrap_or_else(|e| panic!("aria-core refused {body:?}: {e}"));
        let o = omega_parser::lower_to_ir(&src)
            .unwrap_or_else(|e| panic!("omega-parser refused {body:?}: {e}"));
        // `instructions`, not `gate_count()`: the latter filters out Measure
        // and Reset as meta kinds, so it reads 0 for two of these four cases
        // and the comparison would silently pass on a broken expansion.
        assert_eq!(
            a.instructions.len(),
            *want,
            "aria-core expanded {body:?} to {} ops, expected {want}",
            a.instructions.len()
        );
        assert_eq!(
            o.ops.len(),
            *want,
            "omega-parser expanded {body:?} to {} ops, expected {want}",
            o.ops.len()
        );
    }
}
