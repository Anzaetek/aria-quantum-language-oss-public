// SPDX-License-Identifier: Apache-2.0
//! `CRZ` and `RESET` are spellable in the Aria language.
//!
//! Both had a `GateKind`, a QASM spelling, and (for RESET) an audited channel
//! in every backend — but no way to write them in Aria source. A language that
//! cannot express what its own backends execute is a gap, not a design choice.
//!
//! # Why CRZ is not CP
//!
//! `CP(λ) = diag(1, 1, 1, e^{iλ})`, `CRz(λ) = diag(1, 1, e^{−iλ/2}, e^{iλ/2})`.
//! They differ by a relative phase **on the controlled block**, not a global
//! one, so it is visible in any interference. `CRz` therefore maps to omega's
//! native `CRz` rather than being synthesised through `CU3` the way `CP` is —
//! and `crz_is_not_cp` pins that difference so a future "simplification" to
//! reuse the CP path fails loudly.

use aria_core::ast::{parse_aria_circuit, to_qasm, GateKind};

#[test]
fn crz_is_spellable_and_lowers() {
    const SRC: &str = r#"
circuit C {
    qreg q[2]
    apply CRZ(0.7) on q[0], q[1]
}
"#;
    let c = parse_aria_circuit(SRC, "C").expect("CRZ must parse");
    assert_eq!(c.instructions.len(), 1);
    let inst = &c.instructions[0];
    assert_eq!(inst.gate.kind, GateKind::CRz);
    assert_eq!(inst.qubits.len(), 2, "CRZ is control, target");
    assert!(
        (inst.gate.params[0].try_as_f64().unwrap() - 0.7).abs() < 1e-15,
        "the angle must survive parsing"
    );
}

/// The QASM lane spells it `crz`, both directions.
#[test]
fn crz_round_trips_through_qasm2() {
    const SRC: &str = r#"
circuit C {
    qreg q[2]
    apply CRZ(0.7) on q[0], q[1]
}
"#;
    let c = parse_aria_circuit(SRC, "C").unwrap();
    let qasm = to_qasm(&c).expect("CRZ is expressible in QASM 2.0");
    assert!(qasm.contains("crz"), "expected qelib1's `crz` in:\n{qasm}");

    let ir = omega_parser::lower_to_ir(&qasm).expect("our own export must re-parse");
    let crz: Vec<_> = ir
        .ops
        .iter()
        .filter(|op| format!("{:?}", op.gate) == "CRz")
        .collect();
    assert_eq!(crz.len(), 1, "exactly one CRz should come back from:\n{qasm}");
}

/// **CRZ must NOT be lowered as CP.** They differ by a relative phase on the
/// controlled block, which is observable.
///
/// Checked numerically rather than by reading the mapping: build both, contract
/// to statevectors through the same backend, and require they differ.
#[test]
fn crz_is_not_cp() {
    use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
    use omega_core::params::ParameterBinding;

    let sv_qasm = |qasm: &str| -> Vec<num_complex::Complex64> {
        let ir = omega_parser::lower_to_ir(qasm).unwrap();
        let cfg = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        match omega_backend_statevector::StatevectorBackend::new()
            .execute(&ir, &ParameterBinding::new(), &cfg)
            .unwrap()
        {
            ExecResult::Statevector(v) => v,
            other => panic!("expected a statevector, got {other:?}"),
        }
    };
    let sv = |src: &str| -> Vec<num_complex::Complex64> {
        let c = parse_aria_circuit(src, "C").unwrap();
        let qasm = to_qasm(&c).unwrap();
        let ir = omega_parser::lower_to_ir(&qasm).unwrap();
        let cfg = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        match omega_backend_statevector::StatevectorBackend::new()
            .execute(&ir, &ParameterBinding::new(), &cfg)
            .unwrap()
        {
            ExecResult::Statevector(v) => v,
            other => panic!("expected a statevector, got {other:?}"),
        }
    };
    // |+ +> so both amplitudes of the control are populated — on |00> the two
    // gates would agree, because neither acts.
    //
    // The comparison gate is written as `cu3(0,0,λ)` in QASM rather than as
    // Aria `CP`, because `omega-parser` has NO `cp` in its QASM2 gate table —
    // so `to_qasm` of a CP circuit produces a file our own parser rejects
    // ("unknown gate: cp"). That is a real round-trip gap, filed separately;
    // routing around it here keeps this test about CRZ-vs-CP semantics instead
    // of failing for an unrelated reason. `CP(λ) == CU3(0, 0, λ)` is exactly
    // how aria-core lowers CP anyway (backends/omega.rs).
    let head = "circuit C {\n    qreg q[2]\n    apply H on q[0]\n    apply H on q[1]\n";
    let a = sv(&format!("{head}    apply CRZ(0.7) on q[0], q[1]\n}}\n"));
    let b = sv_qasm(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\nh q[1];\n\
         cu3(0,0,0.7) q[0],q[1];\n",
    );

    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        worst > 1e-3,
        "CRZ and CP must differ (relative phase on the controlled block); \
         worst |Δamplitude| = {worst:.3e}. If this is ~0 the lowering is \
         routing CRZ through the CP path."
    );
}

/// `RESET` is spellable, and lowers to the same instruction `reset_qubit`
/// produces.
#[test]
fn reset_is_spellable() {
    const SRC: &str = r#"
circuit C {
    qreg q[1]
    creg c[2]
    apply X on q[0]
    measure q[0] -> c[0]
    apply RESET on q[0]
    measure q[0] -> c[1]
}
"#;
    let c = parse_aria_circuit(SRC, "C").expect("RESET must parse");
    let resets: Vec<_> = c
        .instructions
        .iter()
        .filter(|i| i.gate.kind == GateKind::Reset)
        .collect();
    assert_eq!(resets.len(), 1, "exactly one RESET");
    assert_eq!(resets[0].qubits.len(), 1);
}

/// **RESET is a channel, and it must actually reset.**
///
/// The parse test above would pass on a `RESET` that lowered to a no-op. This
/// runs the circuit: `X` then `RESET` must leave the qubit in |0>, so the
/// second measurement reads 0 on every shot. Without the reset it reads 1.
#[test]
fn reset_actually_clears_the_qubit() {
    use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
    use omega_core::params::ParameterBinding;

    const SRC: &str = r#"
circuit C {
    qreg q[1]
    creg c[1]
    apply X on q[0]
    apply RESET on q[0]
    measure q[0] -> c[0]
}
"#;
    let c = parse_aria_circuit(SRC, "C").unwrap();
    let qasm = to_qasm(&c).unwrap();
    let ir = omega_parser::lower_to_ir(&qasm).unwrap();
    let cfg = ExecConfig {
        shots: Some(512),
        seed: Some(7),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let ExecResult::Counts(counts) = omega_backend_statevector::StatevectorBackend::new()
        .execute(&ir, &ParameterBinding::new(), &cfg)
        .unwrap()
    else {
        panic!("expected counts")
    };
    assert_eq!(
        counts
            .get(&omega_core::outcome::Outcome::from_u64(
                0,
                counts.keys().next().map(|o| o.width()).unwrap_or(1),
            ))
            .copied()
            .unwrap_or(0),
        512,
        "X then RESET must leave |0> on every shot; got {counts:?}. A RESET \
         that lowered to a no-op would report 512 ones instead."
    );
}

/// **The two lowering paths, exercised directly.**
///
/// The tests above route through `to_qasm` → `omega_parser`, which does NOT
/// touch `aria_core::backends::omega::to_omega_ir` nor
/// `aria_runtime::lower` — the two maps this work actually edited.
///
/// Mutation-testing proved it: routing `CRz` through `CU3` in
/// `backends/omega.rs`, and turning `Reset` into a `Barrier` in
/// `aria-runtime/lower.rs`, both left every test above PASSING. The tests were
/// structurally blind to the code they were written for.
///
/// These go through `to_omega_ir` directly.
#[test]
fn to_omega_ir_maps_crz_natively_not_through_cu3() {
    use aria_core::backends::omega::to_omega_ir;
    const SRC: &str = r#"
circuit C {
    qreg q[2]
    apply CRZ(0.7) on q[0], q[1]
}
"#;
    let c = parse_aria_circuit(SRC, "C").unwrap();
    let ir = to_omega_ir(&c);
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(
        format!("{:?}", ir.ops[0].gate),
        "CRz",
        "CRZ must map to omega's NATIVE CRz. Mapping it to CU3 would be the \
         CP path, and CP(λ) = diag(1,1,1,e^{{iλ}}) differs from \
         CRz(λ) = diag(1,1,e^{{-iλ/2}},e^{{iλ/2}}) by a relative phase on the \
         controlled block — observable, not a global phase."
    );
    assert_eq!(ir.ops[0].params.len(), 1, "CRz takes one angle, not three");
}

#[test]
fn to_omega_ir_maps_reset_to_reset() {
    use aria_core::backends::omega::to_omega_ir;
    const SRC: &str = r#"
circuit C {
    qreg q[1]
    apply RESET on q[0]
}
"#;
    let c = parse_aria_circuit(SRC, "C").unwrap();
    let ir = to_omega_ir(&c);
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(
        format!("{:?}", ir.ops[0].gate),
        "Reset",
        "RESET must lower to the channel, not to a Barrier or a no-op"
    );
}

/// **The CP round trip closes** — `to_qasm` output is readable by our own
/// parser.
///
/// `aria-core` emitted `cp` for `GateKind::CP`, and `omega-parser` had no `cp`
/// entry, so exporting a CP circuit produced a file this repository could not
/// read back ("unknown gate: cp"). Only the round trip was broken, with both
/// ends looking healthy.
///
/// **The emitter now writes `cu1`, not `cp`**, and the original doc comment
/// here — "valid qelib1 which Qiskit accepted" — was wrong as stated. Measured
/// on qiskit 2.5.1: `cp` is **not** in qelib1, the strict `qasm2.loads` rejects
/// it, and only the legacy loader accepts it. `cu1` IS in qelib1, loads under
/// both, and is operator-identical (max|delta| = 0.000e+00 against qiskit's
/// native `cp`).
///
/// The parser still reads BOTH spellings — reading widely and writing narrowly
/// is the right asymmetry for an interchange format.
///
/// `CP(λ) == CU3(0, 0, λ)` exactly (`U3(0,0,λ) = diag(1, e^{iλ}) = P(λ)`), so
/// the parser resolves `cp` to CU3 and widens the single angle to three.
#[test]
fn cp_round_trips_through_our_own_parser() {
    const SRC: &str = r#"
circuit C {
    qreg q[2]
    apply CP(0.7) on q[0], q[1]
}
"#;
    let c = parse_aria_circuit(SRC, "C").unwrap();
    let qasm = to_qasm(&c).expect("CP is expressible in QASM 2.0");
    assert!(
        qasm.contains("cu1("),
        "expected the qelib1 spelling `cu1` (not `cp`, which strict qasm2.loads \
         rejects) in:\n{qasm}"
    );
    assert!(
        !qasm.contains("cp("),
        "`cp` is not in qelib1; emitting it makes the file strict-unloadable:\n{qasm}"
    );

    // The parser must still accept the OLD spelling from other producers —
    // qiskit's own `dumps` writes `cp`, so refusing it would break reading what
    // qiskit writes.
    let legacy = qasm.replace("cu1(", "cp(");
    omega_parser::lower_to_ir(&legacy)
        .expect("`cp` must still be READABLE even though we no longer write it");

    let ir = omega_parser::lower_to_ir(&qasm)
        .expect("our own export must re-parse — this is what #23 fixed");
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(format!("{:?}", ir.ops[0].gate), "CU3");
    assert_eq!(
        ir.ops[0].params.len(),
        3,
        "the single cp angle must be widened to CU3's three"
    );
}

/// The widened `cp` must be NUMERICALLY the controlled-phase gate, not merely
/// parse to something with three parameters.
///
/// Compares against `cu3(0,0,0.7)` written directly: identical states. And
/// against `crz(0.7)`: different, which is the same distinction
/// [`crz_is_not_cp`] makes from the other side.
#[test]
fn the_widened_cp_is_the_controlled_phase_gate() {
    use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
    use omega_core::params::ParameterBinding;

    let sv = |qasm: &str| -> Vec<num_complex::Complex64> {
        let ir = omega_parser::lower_to_ir(qasm).unwrap();
        let cfg = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        match omega_backend_statevector::StatevectorBackend::new()
            .execute(&ir, &ParameterBinding::new(), &cfg)
            .unwrap()
        {
            ExecResult::Statevector(v) => v,
            other => panic!("expected a statevector, got {other:?}"),
        }
    };
    let head = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\nh q[1];\n";
    let via_cp = sv(&format!("{head}cp(0.7) q[0],q[1];\n"));
    let via_cu3 = sv(&format!("{head}cu3(0,0,0.7) q[0],q[1];\n"));
    let via_crz = sv(&format!("{head}crz(0.7) q[0],q[1];\n"));

    let d_same = via_cp
        .iter()
        .zip(&via_cu3)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        d_same < 1e-15,
        "cp(λ) must equal cu3(0,0,λ) exactly; worst |Δ| = {d_same:.3e}"
    );

    let d_diff = via_cp
        .iter()
        .zip(&via_crz)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        d_diff > 1e-3,
        "cp and crz must DIFFER (relative phase on the controlled block); \
         worst |Δ| = {d_diff:.3e}"
    );
}

/// `cu1` is qelib1's other spelling of the same gate.
#[test]
fn cu1_is_accepted_as_the_same_gate() {
    let ir = omega_parser::lower_to_ir(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncu1(0.7) q[0],q[1];\n",
    )
    .expect("cu1 must parse");
    assert_eq!(format!("{:?}", ir.ops[0].gate), "CU3");
    assert_eq!(ir.ops[0].params.len(), 3);
}

/// A wrong arity is refused rather than silently widened to garbage.
#[test]
fn cp_with_the_wrong_arity_is_refused() {
    let err = omega_parser::lower_to_ir(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncp(0.1,0.2) q[0],q[1];\n",
    )
    .expect_err("cp takes exactly one angle");
    assert!(err.contains("1 parameter"), "unhelpful message: {err}");
}
