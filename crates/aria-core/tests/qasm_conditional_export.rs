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
        polarized: false,
    });
    c.registers.push(RegisterDecl {
        name: creg_name.into(),
        size: creg_size,
        kind: RegisterKind::Classical,
        polarized: false,
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

/// **QASM 3 is the escape hatch `to_qasm` names, so it must actually be one.**
///
/// This test previously asserted the ABSENCE of `if` in the QASM 3 path, and
/// documented that as honesty about a missing feature. It was honest about the
/// test and wrong about the product: `to_qasm`'s refusal message told users to
/// come here for single-bit guards, and `to_qasm3` emitted every guarded
/// instruction BARE. Following our own advice produced the silently wrong
/// circuit Part H measured — misdirection, which is worse than the silence it
/// replaced.
///
/// The absence-assertion was written as a canary that would "fail loudly when
/// the gap closes". It did its job; this is the inversion.
#[test]
fn to_qasm3_emits_the_single_bit_guard_qasm2_cannot() {
    // The 2-bit register QASM 2.0 must refuse: OpenQASM 3 addresses one bit.
    let c = feedforward("c", 2);
    assert!(
        to_qasm(&c).is_err(),
        "QASM 2.0 must still refuse a single-bit guard on a 2-bit register"
    );

    let out = to_qasm3(&c);
    assert!(out.contains("OPENQASM 3.0;"), "got:\n{out}");
    assert!(
        out.contains("if (c[0] == 1) x q[1];"),
        "QASM 3 must carry the guard the 2.0 path cannot express — that is the \
         entire premise of `to_qasm`'s refusal message. Got:\n{out}"
    );
    // And the guarded gate must not ALSO appear bare.
    assert!(
        !out.lines().any(|l| l.trim() == "x q[1];"),
        "found an UNGUARDED `x q[1];` — the guard was dropped:\n{out}"
    );
}

/// A size-1 register works too, so the fix is not special-cased to the wide
/// case that motivated it.
#[test]
fn to_qasm3_guards_a_size_one_register_as_well() {
    let out = to_qasm3(&feedforward("c", 1));
    assert!(
        out.contains("if (c[0] == 1) x q[1];"),
        "got:\n{out}"
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

/// **The Aria emitter's output must RE-PARSE**, not merely contain the right
/// substring.
///
/// `the_aria_emitter_carries_the_guard_too` is a substring check — the exact
/// standard its sibling `an_exported_guard_survives_reparsing` states three
/// lines away, and it could not see either of these:
///
/// 1. `RESET` was emitted as `-- reset q[0]`, a comment, so a channel that
///    changes measurement statistics vanished on round trip.
/// 2. A guarded comment line became `when m[0] == 1 { -- reset q[0] }`, and
///    since `--` runs to end-of-line the closing brace was commented out and
///    the file no longer parsed at all.
///
/// Both shipped in the commit that made RESET spellable.
#[test]
fn aria_emitter_output_reparses_with_guards_and_reset() {
    use aria_core::ast::{parse_aria_circuit, to_aria_source, GateKind as GK};

    let mut c = feedforward("c", 1);
    // A guarded RESET — pins defect 1.
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::Reset),
        qubits: vec![Qubit::new("q", 0)],
        clbits: vec![],
        condition: Some((Clbit::new("c", 0), 1)),
    });
    // A guarded BARRIER — pins defect 2. This is load-bearing and was missing:
    // once RESET became a real statement, NOTHING in the fixture emitted a
    // `--` comment any more, so the comment-wrapping mutation passed. The
    // Aria emitter still writes `-- barrier …`, so this is the case that
    // exercises the path.
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::Barrier),
        qubits: vec![Qubit::new("q", 0), Qubit::new("q", 1)],
        clbits: vec![],
        condition: Some((Clbit::new("c", 0), 1)),
    });

    let src = to_aria_source(&c, "Feedforward");
    let back = parse_aria_circuit(&src, "Feedforward").unwrap_or_else(|e| {
        panic!("the Aria emitter's own output must re-parse; got {e}\n\n{src}")
    });

    // The RESET survived as an instruction, not a comment.
    let resets: Vec<_> = back
        .instructions
        .iter()
        .filter(|i| i.gate.kind == GK::Reset)
        .collect();
    assert_eq!(
        resets.len(),
        1,
        "RESET must round-trip as an instruction. Emitting it as `-- reset` \
         deletes a channel silently. Source was:\n{src}"
    );

    // And the guards survived.
    let guarded = back.instructions.iter().filter(|i| i.condition.is_some()).count();
    assert!(
        guarded >= 1,
        "at least one guard must survive the round trip; source was:\n{src}"
    );
}
