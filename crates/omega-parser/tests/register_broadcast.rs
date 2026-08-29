// SPDX-License-Identifier: Apache-2.0
//! **A bare register argument must never be silently reinterpreted.**
//!
//! `resolve_qubit` expands a bare register (`q`, no index) to all of its
//! qubits. Three callers then disagreed about what to do with the result, and
//! all three disagreements were silent.
//!
//! These are reachable from ordinary QASM2 — nothing here needs QASM3 — but
//! they were found while scoping conformance against the official OpenQASM
//! examples, which use the register-wide forms constantly. Our own emitter
//! never writes them, which is exactly why nothing exercised these paths.
//!
//! ```text
//!   D1  qubit[2] q; cz q;          -> silently became `cz q[0], q[1]`
//!   D2  measure q -> c;            -> `zip` truncated to the shorter register
//!   D3  gate g a,b {..} g q, r;    -> a register argument contributed only q[0],
//!                                     and EXTRA arguments were dropped unchecked
//! ```
//!
//! D1 is the one that returns numbers. The arity check ran on the **flattened**
//! operand count, so a 2-qubit gate applied to one 2-qubit register passed it
//! and produced a circuit the file never described. OpenQASM defines no such
//! broadcast.
//!
//! This is the same shape as the defect `gate_arity_is_validated.rs` records:
//! a validation that exists on one path and is bypassed on another.
//!
//! # What must NOT regress
//!
//! Over-refusal is a regression too. `barrier q;`, `reset q;` and
//! `measure q -> c;` at equal widths are legal, are used by the shipped
//! fixtures, and must keep working — they have their own statement arms that
//! loop correctly and never reach `lower_gate_app`.

use omega_parser::lower_to_ir;

fn err_of(src: &str, what: &str) -> String {
    match lower_to_ir(src) {
        Ok(ir) => panic!(
            "{what}: expected a refusal, but it lowered to {} ops — \
             a silent reinterpretation is the defect:\n{src}",
            ir.ops.len()
        ),
        Err(e) => e,
    }
}

// ---------------------------------------------------------------- D1

/// `cz q;` on a 2-qubit register used to become `cz q[0], q[1]`.
///
/// The flattened operand count is 2, which is `CZ`'s arity, so the check that
/// should have caught it waved it through. The answer was then confidently
/// wrong: a file describing one thing executed as another.
#[test]
fn a_two_qubit_gate_over_one_register_is_refused() {
    let src = "OPENQASM 2.0;\nqreg q[2];\ncz q;\n";
    let e = err_of(src, "cz over a 2-qubit register");
    assert!(
        e.contains("broadcast") || e.contains("register"),
        "the message must name the construct, not just the arity; got: {e}"
    );
}

/// The same trap at three qubits — `ccx q;` on `qreg q[3]`.
///
/// Worth its own case because it shows the coincidence is not special to two:
/// any register whose width happens to equal a gate's arity hit this.
#[test]
fn a_three_qubit_gate_over_one_register_is_refused() {
    let src = "OPENQASM 2.0;\nqreg q[3];\nccx q;\n";
    err_of(src, "ccx over a 3-qubit register");
}

/// A width that does *not* coincide was already refused — but for the wrong
/// reason (a raw arity mismatch). Pinned so the new message covers it too,
/// rather than leaving two different diagnostics for one construct.
#[test]
fn a_non_coinciding_width_is_still_refused() {
    let src = "OPENQASM 2.0;\nqreg q[3];\ncz q;\n";
    err_of(src, "cz over a 3-qubit register");
}

// ---------------------------------------------------------------- D2

/// `measure q -> c;` with mismatched widths silently produced the shorter
/// number of measurements, because the implementation zipped the two.
#[test]
fn a_measure_broadcast_with_mismatched_widths_is_refused() {
    let src = "OPENQASM 2.0;\nqreg q[4];\ncreg c[2];\nmeasure q -> c;\n";
    let e = err_of(src, "measure q[4] -> c[2]");
    assert!(
        e.contains('4') && e.contains('2'),
        "the message must name BOTH widths so the user can see which is wrong; got: {e}"
    );
}

/// Equal widths are legal and must keep working — this is the form the shipped
/// cross-check fixtures use.
#[test]
fn a_measure_broadcast_at_equal_widths_still_works() {
    let src = "OPENQASM 2.0;\nqreg q[3];\ncreg c[3];\nmeasure q -> c;\n";
    let ir = lower_to_ir(src).expect("equal-width measure broadcast is legal");
    assert_eq!(
        ir.ops.len(),
        3,
        "a 3-qubit measure broadcast must emit exactly 3 measurements"
    );
}

// ---------------------------------------------------------------- D3

/// A user-defined gate given a bare register used only that register's qubit 0.
///
/// The empty-body case applies nothing at all, so the body must be non-empty
/// for the discard to be observable — which is why this test has one.
#[test]
fn a_user_gate_given_a_wide_register_is_refused() {
    let src = "OPENQASM 2.0;\nqreg q[3];\ngate flip a { x a; }\nflip q;\n";
    err_of(src, "flip applied to a 3-qubit register");
}

/// Extra arguments were dropped with no check at all: the loop guarded only
/// `i < qubit_args.len()`, so a definition taking one qubit accepted three.
#[test]
fn too_many_arguments_to_a_user_gate_are_refused() {
    let src = "OPENQASM 2.0;\nqreg q[3];\ngate flip a { x a; }\nflip q[0], q[1], q[2];\n";
    let e = err_of(src, "flip called with 3 arguments");
    assert!(
        e.contains('1') && e.contains('3'),
        "the message must name expected and actual argument counts; got: {e}"
    );
}

/// Too FEW arguments were equally unchecked, and are the more dangerous half:
/// the missing qubit simply never entered the map, so the body silently lost
/// operations rather than gaining wrong ones.
#[test]
fn too_few_arguments_to_a_user_gate_are_refused() {
    let src = "OPENQASM 2.0;\nqreg q[2];\ngate pair a, b { cx a, b; }\npair q[0];\n";
    err_of(src, "pair called with 1 argument");
}

/// A correct call must still work, so the refusals above are about the defect
/// and not about user-defined gates in general.
#[test]
fn a_correctly_called_user_gate_still_works() {
    let src = "OPENQASM 2.0;\nqreg q[2];\ngate pair a, b { cx a, b; }\npair q[0], q[1];\n";
    let ir = lower_to_ir(src).expect("a well-formed user gate call must lower");
    assert_eq!(ir.ops.len(), 1, "the body is one cx");
}

// ------------------------------------------------- broadcast that is LEGAL

/// `barrier q;` and `reset q;` expand over the register and always did — they
/// have their own statement arms. Pinned so the D1 fix cannot over-reach into
/// them, which is the obvious way to break something while fixing this.
#[test]
fn barrier_and_reset_broadcast_are_unaffected() {
    let src = "OPENQASM 2.0;\nqreg q[3];\nreset q;\nbarrier q;\n";
    let ir = lower_to_ir(src).expect("reset/barrier broadcast is legal");
    let resets = ir
        .ops
        .iter()
        .filter(|o| matches!(o.gate, omega_core::circuit::GateKind::Reset))
        .count();
    assert_eq!(resets, 3, "reset over a 3-qubit register is 3 resets");
}

/// A **single-qubit** gate over a bare register is legal OpenQASM and expands
/// to one op per qubit. It did not lower before this work — the flattening sent
/// `h q;` into the arity check as a 4-operand `H`.
///
/// Required by the official `inverseqft2` and `qpt` examples.
#[test]
fn a_single_qubit_gate_broadcasts_to_one_op_per_qubit() {
    let src = "OPENQASM 2.0;\nqreg q[4];\nh q;\n";
    let ir = lower_to_ir(src).expect("single-qubit broadcast is legal OpenQASM");
    assert_eq!(
        ir.ops.len(),
        4,
        "h over a 4-qubit register must expand to 4 separate H ops, \
         not one H carrying 4 operands"
    );
    for (i, o) in ir.ops.iter().enumerate() {
        assert_eq!(
            o.qubits.len(),
            1,
            "op {i} must act on exactly one qubit, got {:?}",
            o.qubits
        );
    }
}
