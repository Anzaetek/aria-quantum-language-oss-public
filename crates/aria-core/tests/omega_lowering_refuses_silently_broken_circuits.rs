// SPDX-License-Identifier: Apache-2.0
//! **`backends/omega.rs` must refuse what it cannot represent, not guess.**
//!
//! `PLAN-EXPORT-INTEGRITY.md` P6. Two silent corruptions, measured before the
//! fix:
//!
//! ```text
//!   x zz[7]  with only q[2] declared   ->  Ok, an X on QUBIT 0
//!   gate conditioned on an unmapped
//!   clbit                              ->  Ok, condition = None (unconditional)
//! ```
//!
//! Neither produced a diagnostic. The first executes a **different circuit**;
//! the second turns a gate that should fire on some shots into one that fires on
//! every shot.
//!
//! `aria-runtime/src/lower.rs` already returned errors for both, so this file
//! was the odd one out inside its own crate family — the same construct, two
//! philosophies, and the silent one feeding callers that panic by design.
//!
//! # Checked against the reference implementations
//!
//! Both are refusals in Qiskit 2.5.1 too, so this is not a house style:
//!
//! ```text
//!   x zz[0];        ->  QASM2ParseError: 'zz' is not defined in this scope
//!   if (zz==1) ...  ->  QASM2ParseError: 'zz' is not defined in this scope
//! ```
//!
//! # The one place we deliberately differ, and why
//!
//! Qiskit **accepts** `if (c == 2)` on `creg c[1]` (measured). That is a
//! REGISTER-valued comparison: well-formed over the register's value space, just
//! unsatisfiable. Aria's `condition` names a single `Clbit`, whose only values
//! are 0 and 1, so the same literal is out of domain rather than merely false —
//! and is refused. A register-wide comparison is a separate feature this
//! representation does not carry, and inventing it silently would be the defect
//! this file is about.

use aria_core::ast::nodes::*;
use aria_core::backends::omega::try_to_omega_ir;

fn conditioned(cond: Option<(Clbit, u64)>, declare_creg: bool) -> Circuit {
    let mut c = Circuit::new("c");
    let q = c.qreg("q", 1);
    if declare_creg {
        let _ = c.creg("m", 1);
    }
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::X),
        qubits: vec![q[0].clone()],
        clbits: vec![],
        condition: cond,
    });
    c
}

/// An undeclared qubit used to become qubit 0.
#[test]
fn an_undeclared_qubit_is_refused_not_retargeted_to_zero() {
    let mut c = Circuit::new("c");
    let _ = c.qreg("q", 2);
    c.apply(GateDef::new(GateKind::X), vec![Qubit::new("zz", 7)]);

    let err = match try_to_omega_ir(&c) {
        Ok(ir) => panic!(
            "lowered to {:?} — an X on qubit {} is a DIFFERENT circuit than the one \
             written, and nothing said so",
            ir.ops.iter().map(|o| o.qubits.clone()).collect::<Vec<_>>(),
            ir.ops[0].qubits[0]
        ),
        Err(e) => e,
    };
    assert!(err.contains("zz"), "the error must name the qubit: {err}");
    assert!(
        err.contains("undeclared") || err.contains("declare"),
        "and say what is wrong with it: {err}"
    );
}

/// A condition on an unmapped clbit used to be dropped, making the gate
/// unconditional — it then fired on every shot instead of some.
#[test]
fn a_condition_on_an_undeclared_clbit_is_refused_not_dropped() {
    let c = conditioned(Some((Clbit::new("zz", 3), 1)), false);
    let err = match try_to_omega_ir(&c) {
        Ok(ir) => panic!(
            "lowered with condition {:?} — a dropped guard makes the gate fire on \
             EVERY shot",
            ir.ops[0].condition
        ),
        Err(e) => e,
    };
    assert!(err.contains("zz"), "the error must name the bit: {err}");
}

/// A single classical bit holds 0 or 1, so `== 2` can never fire.
///
/// The message must explain the Qiskit difference rather than implying Qiskit
/// agrees — it does not, and a future reader comparing the two needs to know
/// this was a decision and not an oversight.
#[test]
fn a_condition_literal_above_one_is_refused_on_a_single_bit() {
    let mut c = Circuit::new("c");
    let q = c.qreg("q", 1);
    let cl = c.creg("m", 1);
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::X),
        qubits: vec![q[0].clone()],
        clbits: vec![],
        condition: Some((cl[0].clone(), 2)),
    });
    let err = try_to_omega_ir(&c).expect_err("`m[0] == 2` can never fire");
    assert!(err.contains("never fire"), "{err}");
    assert!(
        err.contains("Qiskit") || err.contains("qiskit"),
        "the message must state that this differs from Qiskit's register-valued \
         comparison, so the divergence is visible: {err}"
    );
}

/// **Guard the guard.** A lowering that refused everything would satisfy every
/// assertion above. Valid circuits — including a correctly-declared condition at
/// both legal literals — must still lower.
#[test]
fn valid_circuits_still_lower() {
    for (label, v) in [("condition == 0", 0u64), ("condition == 1", 1)] {
        let mut c = Circuit::new("c");
        let q = c.qreg("q", 1);
        let cl = c.creg("m", 1);
        c.instructions.push(Instruction {
            gate: GateDef::new(GateKind::X),
            qubits: vec![q[0].clone()],
            clbits: vec![],
            condition: Some((cl[0].clone(), v)),
        });
        let ir = try_to_omega_ir(&c)
            .unwrap_or_else(|e| panic!("{label}: a VALID circuit was refused: {e}"));
        assert_eq!(
            ir.ops[0].condition,
            Some((0u32, v)),
            "{label}: the condition must survive lowering intact"
        );
    }

    // And an ordinary unconditioned circuit.
    let c = conditioned(None, true);
    let ir = try_to_omega_ir(&c).expect("an unconditioned circuit must lower");
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(ir.ops[0].condition, None);
    assert_eq!(ir.ops[0].qubits, vec![0]);
}
