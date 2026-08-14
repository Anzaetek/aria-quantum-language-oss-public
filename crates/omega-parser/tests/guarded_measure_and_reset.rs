// SPDX-License-Identifier: Apache-2.0
//! **A classical guard may wrap a `measure` or a `reset`, not only a gate.**
//!
//! `PLAN-EXPORT-INTEGRITY.md` P5 item 1, and independently reported by an
//! external bundle. Measured before this, against qiskit 2.5.1 on the same
//! three inputs:
//!
//! ```text
//!                     ours                          qiskit
//!   if (c==1) x       OK                            OK
//!   if (c==1) measure PARSE ERROR at 5:19           OK
//!   if (c==1) reset   "unknown gate: reset"         OK
//! ```
//!
//! Two different causes for what looked like one bug:
//!
//! * `measure` — the grammar's `if_stmt` admitted only `gate_app_stmt`, so it
//!   never parsed;
//! * `reset` — it *did* parse, because `reset q[0];` looks like a gate
//!   application, and then died at lowering. Fixing the grammar alone would
//!   have left it broken, because pest's `|` is an ORDERED choice and
//!   `gate_app_stmt` matched first. The specific forms have to come first.
//!
//! This workspace's own emitter produces guarded measures, so these were files
//! we could write and not read.

fn ir(src: &str) -> omega_core::circuit::CircuitIR {
    omega_parser::lower_to_ir(src).unwrap_or_else(|e| panic!("{src}\n-> {e}"))
}

const HDR: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n";

/// Each guarded form lowers to exactly one op of the right kind, carrying the
/// condition.
///
/// The assertion is on the CONDITION, not merely on "it parsed". A guarded
/// statement that lowered unconditioned would run on every shot instead of
/// some — the silent-drop failure in a different costume.
#[test]
fn each_guarded_form_lowers_with_its_condition() {
    use omega_core::circuit::GateKind;
    let cases: &[(&str, &str, GateKind)] = &[
        ("gate", "if (c==1) x q[0];", GateKind::X),
        ("measure", "if (c==1) measure q[0] -> c[1];", GateKind::Measure),
        ("reset", "if (c==1) reset q[0];", GateKind::Reset),
    ];
    for (name, body, want) in cases {
        let ir = ir(&format!("{HDR}{body}\n"));
        assert_eq!(ir.ops.len(), 1, "{name}: expected one op, got {}", ir.ops.len());
        assert_eq!(&ir.ops[0].gate, want, "{name}: wrong gate kind");
        assert!(
            ir.ops[0].condition.is_some(),
            "{name}: lowered WITHOUT the guard — it would run on every shot"
        );
    }
}

/// **Guard the guard.** The unguarded forms must still lower, and must NOT
/// carry a condition. Without this, a lowering that conditioned everything —
/// or one that dropped the `if` entirely — would satisfy the test above.
#[test]
fn the_unguarded_forms_are_unchanged() {
    use omega_core::circuit::GateKind;
    for (name, body, want) in [
        ("gate", "x q[0];", GateKind::X),
        ("measure", "measure q[0] -> c[1];", GateKind::Measure),
        ("reset", "reset q[0];", GateKind::Reset),
    ] {
        let ir = ir(&format!("{HDR}{body}\n"));
        assert_eq!(ir.ops.len(), 1, "{name}");
        assert_eq!(ir.ops[0].gate, want, "{name}: wrong kind — did the ordered choice change?");
        assert!(
            ir.ops[0].condition.is_none(),
            "{name}: an UNGUARDED statement acquired a condition"
        );
    }
}

/// `reset` must not be read as a gate application named "reset".
///
/// It was, before the grammar's alternatives were reordered, and the symptom
/// appeared at lowering rather than at parse — so a test that only checked
/// "does it parse" would have passed while the file was unreadable.
#[test]
fn reset_is_a_reset_not_a_gate_named_reset() {
    use omega_core::circuit::GateKind;
    let ir = ir(&format!("{HDR}if (c==1) reset q[0];\n"));
    assert_eq!(ir.ops[0].gate, GateKind::Reset);
}

/// A guarded BARRIER is refused — matching qiskit, which rejects it in both
/// loaders with "needed a gate application, measurement or reset".
///
/// It reaches the lowering because `barrier q[0];` matches `gate_app_stmt`, so
/// the grammar cannot exclude it without excluding real gates. Measured: before
/// this it lowered to a conditioned barrier, i.e. our reader admitted a
/// construct the reference implementation rejects — which is how this
/// workspace's own emitter once produced QASM that qiskit could not load.
#[test]
fn a_guarded_barrier_is_refused_as_qiskit_refuses_it() {
    let r = omega_parser::lower_to_ir(&format!("{HDR}if (c==1) barrier q[0];\n"));
    let e = match r {
        Ok(ir) => panic!(
            "accepted a guarded barrier, lowering it to {} op(s) — qiskit rejects this \
             in BOTH loaders, so the file would be unreadable elsewhere",
            ir.ops.len()
        ),
        Err(e) => e,
    };
    assert!(
        e.contains("barrier"),
        "the refusal must name the construct: {e}"
    );
}

/// The lowering must never emit a guarded statement WITHOUT its condition.
///
/// Widening the grammar without widening the lowering would have turned a parse
/// error into a silent drop — the statement simply absent — which is strictly
/// worse than refusing. Pinned by construction: every op in the guarded range
/// carries a condition.
#[test]
fn no_guarded_statement_lowers_unconditioned() {
    for body in [
        "if (c==1) x q[0];",
        "if (c==1) measure q[0] -> c[1];",
        "if (c==1) reset q[0];",
    ] {
        let ir = ir(&format!("{HDR}{body}\n"));
        assert!(!ir.ops.is_empty(), "{body}: dropped entirely");
        for op in &ir.ops {
            assert!(
                op.condition.is_some(),
                "{body}: lowered an op with no condition — it would run on every shot"
            );
        }
    }
}
