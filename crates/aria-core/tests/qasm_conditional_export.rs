// SPDX-License-Identifier: Apache-2.0
//! `to_qasm` must carry a classical guard or refuse — never drop it silently.
//!
//! # The defect these pin (FIXES_PLAN.md Part H)
//!
//! The emitter had **zero references to `condition`**, so a guarded gate was
//! written out unguarded. Measured on `H q0; measure q0 -> c0; when c0 == 1
//! { X q1 }`: Qiskit running Aria's own export returned
//! `{"11": 2002, "10": 1998}` where the true distribution is
//! `{"00": 1995, "11": 2005}`. The correlation is destroyed, the file is valid
//! QASM, and the importer has no way to know anything was lost.
//!
//! That is the worst shape a defect can take here — silent on **both** sides —
//! and it is why `FIXES_PLAN.md` K4 trap 1 forbids conducting a differential
//! test *through* this export: both engines would agree, having lost the same
//! thing.
//!
//! # Why refusal, not best-effort
//!
//! Aria conditions on a single classical bit; QASM 2.0's `if` compares a whole
//! register. They coincide only at register size 1. On a wider register,
//! emitting `if (c == V)` asserts something *different* — trading a silent
//! drop for a silent change of meaning, which is worse because the output
//! looks right.

use aria_core::ast::{
    to_qasm, to_qasm3, Circuit, Clbit, GateDef, GateKind, Instruction, Qubit, RegisterDecl,
    RegisterKind,
};

/// `qreg q[2]; creg <name>[<size>]; h q0; measure q0 -> c0; when c0==1 { x q1 }`
fn feedforward(creg_name: &str, creg_size: usize) -> Circuit {
    let mut c = Circuit::new("Feedforward");
    c.registers.push(RegisterDecl {
        name: "q".into(),
        size: 2,
        kind: RegisterKind::Quantum,
    });
    c.registers.push(RegisterDecl {
        name: creg_name.into(),
        size: creg_size,
        kind: RegisterKind::Classical,
    });
    let q = |i: usize| Qubit::new("q", i);
    let cb = Clbit::new(creg_name, 0);

    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::H),
        qubits: vec![q(0)],
        clbits: vec![],
        condition: None,
    });
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::Measure),
        qubits: vec![q(0)],
        clbits: vec![cb.clone()],
        condition: None,
    });
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::X),
        qubits: vec![q(1)],
        clbits: vec![],
        condition: Some((cb, 1)),
    });
    c
}

/// **The guard reaches the file.** This is the assertion whose absence let the
/// defect ship: the exported text must contain the `if`, not just the `x`.
#[test]
fn a_single_bit_guard_is_exported_as_an_if() {
    let qasm = to_qasm(&feedforward("c", 1)).expect("size-1 creg is expressible");
    assert!(
        qasm.contains("if (c == 1) x q[1];"),
        "the guard must reach the file; got:\n{qasm}"
    );
    // And the guarded gate must NOT also appear unguarded anywhere.
    let unguarded = qasm
        .lines()
        .any(|l| l.trim() == "x q[1];");
    assert!(
        !unguarded,
        "found an UNGUARDED `x q[1];` — the guard was dropped, which is the \
         original defect. Full output:\n{qasm}"
    );
}

/// A wider register is REFUSED, and the message says why and what to do.
///
/// Emitting `if (c == 1)` here would assert the whole 2-bit register equals 1,
/// which is a different predicate from `c[0] == 1`.
#[test]
fn a_multi_bit_register_is_refused_rather_than_reinterpreted() {
    let err = to_qasm(&feedforward("c", 2))
        .expect_err("a size-2 creg cannot express a single-bit guard in QASM 2.0");
    for want in ["size 2", "different predicate", "to_qasm3"] {
        assert!(
            err.contains(want),
            "the refusal must explain the problem and the way out; missing {want:?} in:\n{err}"
        );
    }
}

/// The refusal must be load-bearing, not incidental: the SAME circuit without
/// the guard exports fine on a wide register.
///
/// Without this, an emitter that refused every multi-bit creg would pass the
/// test above.
#[test]
fn a_wide_register_without_a_guard_still_exports() {
    let mut c = feedforward("c", 2);
    for inst in &mut c.instructions {
        inst.condition = None;
    }
    let qasm = to_qasm(&c).expect("no guard, nothing to refuse");
    assert!(qasm.contains("creg c[2];"), "got:\n{qasm}");
    assert!(qasm.contains("x q[1];"), "got:\n{qasm}");
}

/// QASM 3 is the documented escape hatch, so it must actually be one.
///
/// `to_qasm3` is infallible today and does not emit `if` at all — so this
/// records the CURRENT state honestly rather than asserting a capability that
/// does not exist. If the refusal message points at `to_qasm3`, that path had
/// better at least accept the circuit.
#[test]
fn to_qasm3_accepts_what_qasm2_refuses() {
    let out = to_qasm3(&feedforward("c", 2));
    assert!(out.contains("OPENQASM 3.0;"), "got:\n{out}");
    // NOTE: to_qasm3 does not yet emit `if (c[i] == V)` either — its doc
    // comment says so ("no classical control flow"). The QASM2 refusal points
    // at it as the right *destination*, and closing that gap is tracked
    // separately. Asserting the `if` here would be asserting a feature that
    // does not exist, which is how a test starts lying about capability.
    assert!(
        !out.contains("if ("),
        "to_qasm3 has gained `if` emission — update the QASM2 refusal message \
         and this test, which deliberately records its ABSENCE"
    );
}

/// **Round-trip through the real parser**, not just a substring check.
///
/// Asserting that the emitted text *contains* `if (c == 1) x q[1];` only proves
/// we wrote what we meant to write. Re-parsing proves a consumer can read it
/// back and recovers the guard — which is the property the export exists for,
/// and the one the original defect broke while leaving the file perfectly
/// well-formed.
#[test]
fn an_exported_guard_survives_reparsing() {
    let qasm = to_qasm(&feedforward("c", 1)).expect("size-1 creg is expressible");
    let ir = omega_parser::lower_to_ir(&qasm).expect("our own export must re-parse");

    let guarded: Vec<_> = ir
        .ops
        .iter()
        .filter(|op| op.condition.is_some())
        .collect();
    assert_eq!(
        guarded.len(),
        1,
        "exactly one op should come back conditioned; got {} in:\n{qasm}",
        guarded.len()
    );
    // (start_bit, num_bits, expected) — a size-1 creg at bit 0, expecting 1.
    assert_eq!(guarded[0].condition, Some((0, 1, 1)));
    assert_eq!(
        format!("{:?}", guarded[0].gate),
        "X",
        "the X is the guarded op"
    );
}

/// A conditioned `barrier` is refused, because QASM 2.0's `if` takes a gate
/// application, measurement or reset — and a barrier is none of those.
///
/// **Measured against Qiskit 2.5.1**, not assumed: `if (c==1) barrier q;` is
/// rejected with "needed a gate application, measurement or reset", while
/// guarded gate / reset / measure are all accepted.
///
/// Our own pest grammar happens to ADMIT it (`barrier q` parses as a gate
/// application), so a round-trip through `omega-parser` alone would not have
/// caught this. The export exists for interchange, so the consumer that
/// matters is the other toolchain.
#[test]
fn a_conditioned_barrier_is_refused() {
    let mut c = feedforward("c", 1);
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::Barrier),
        qubits: vec![Qubit::new("q", 0), Qubit::new("q", 1)],
        clbits: vec![],
        condition: Some((Clbit::new("c", 0), 1)),
    });
    let err = to_qasm(&c).expect_err("a guarded barrier is not valid QASM 2.0");
    assert!(
        err.contains("barrier") && err.contains("gate application"),
        "the refusal must say why a barrier cannot be guarded; got:\n{err}"
    );
}

/// **The Aria emitter had the same defect**, and it was missed entirely by the
/// QASM work above.
///
/// `aria_emit.rs` had ZERO references to `condition`, so
/// `when m[0] == 1 { apply X on q[0] }` round-tripped as a bare
/// `apply X on q[0]` — the guard silently gone. Aria → Aria is the export a
/// reader is *most* likely to assume is faithful, which makes the silence
/// worse rather than better.
///
/// Found by cross-checking an external patch series against the in-tree
/// implementation: the QASM fix and this one are the same bug in two emitters,
/// and fixing one drew no attention to the other.
#[test]
fn the_aria_emitter_carries_the_guard_too() {
    use aria_core::ast::to_aria_source;
    let src = to_aria_source(&feedforward("c", 1), "Feedforward");
    assert!(
        src.contains("when c[0] == 1"),
        "the Aria emitter must wrap the guarded gate in `when`; got:\n{src}"
    );
    // And the guarded gate must not ALSO appear bare.
    let bare = src
        .lines()
        .any(|l| l.trim() == "apply X on q[1]");
    assert!(
        !bare,
        "found a BARE `apply X on q[1]` — the guard was dropped. Full output:\n{src}"
    );
}
