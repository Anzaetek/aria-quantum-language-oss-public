// SPDX-License-Identifier: Apache-2.0
//! **DITQASM registers lower to a `CircuitIR` that knows its wire
//! dimensions — against fixtures `mqt.qudits` wrote, not ones we did.**
//! (PLAN-QUDIT.md Q1, parser half.)
//!
//! Q0 committed eight files under `tests/fixtures/ditqasm/` as the byte
//! output of `mqt.qudits 0.5.2` (`PROVENANCE.json` records which two are
//! hand-written). A grammar that diverges from MQT's reading of its own
//! format fails here on bytes this repository did not author — the guard
//! against the QPY lane's writer-validated-by-its-own-reader defect
//! (`PLAN-OPEN-20260825.md:934-943`).
//!
//! What Q1 parses is the header token and the `qreg` dimension group, and
//! nothing else new. Gates that only DITQASM spells — `csum`, `rxy`, the
//! four-parameter `cx`, `ctl` — are outside Q1 by the plan's own cost
//! analysis, so the fixtures that use them are expected to fail AT THE
//! GATE, after the registers have been read: the register error must never
//! be what stops them.

use omega_core::circuit::QuditRegister;
use omega_parser::lower::lower_to_ir;
use std::path::PathBuf;

fn fixture(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ditqasm")
        .join(format!("{name}.qasm"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn dims_of(src: &str) -> Vec<u32> {
    lower_to_ir(src)
        .unwrap_or_else(|e| panic!("must lower: {e}\n{src}"))
        .wire_dims()
}

// ---------------------------------------------------------------------------
// The four §3e.1a answers, on MQT's bytes.
// ---------------------------------------------------------------------------

/// `qreg q [1][3];` — one qutrit. `h` on a qudit is DITQASM's Fourier gate
/// (Q0), which Q1 does not lower; the register must still be read first.
#[test]
fn qutrit_register_is_read_as_dimension_three() {
    let src = fixture("qutrit_h");
    let dims = dims_of(&src);
    assert_eq!(dims, vec![3], "{src}");
    let ir = lower_to_ir(&src).unwrap();
    assert_eq!(
        ir.qudit_registers,
        vec![QuditRegister {
            name: "q".into(),
            start: 0,
            dims: vec![3]
        }]
    );
    assert_eq!(ir.num_qubits, 1);
    let (reg, wire, d) = ir.first_qudit().expect("a qutrit is a qudit");
    assert_eq!((reg.name.as_str(), wire, d), ("q", 0, 3));
}

/// `qreg matter [2];` with no group is TWO qudits of dimension 2 — Q0's
/// first answer, from `qasm.py`'s `qdims = [2] * nq`. Not one qudit of
/// dimension 2, which is what the tutorial's `field [7][5,…]` suggested.
#[test]
fn bare_qreg_is_n_qubits_and_not_a_qudit_register() {
    let ir = lower_to_ir(&fixture("bare_qreg_input")).unwrap();
    assert_eq!(ir.num_qubits, 2);
    assert_eq!(ir.wire_dims(), vec![2, 2]);
    assert!(
        ir.qudit_registers.is_empty(),
        "a bare qreg declares nothing about dimension: {:?}",
        ir.qudit_registers
    );
    assert!(ir.first_qudit().is_none());
    assert!(ir.refuse_qudits("test").is_ok());
}

/// What MQT writes BACK for the bare form: `qreg matter [2][2,2];`. The
/// group is explicit and all-2, so it is recorded — the IR must not lie
/// about its source — and it is still not a qudit circuit.
#[test]
fn explicit_all_two_group_is_recorded_and_still_a_qubit_circuit() {
    let ir = lower_to_ir(&fixture("bare_qreg_reemitted")).unwrap();
    assert_eq!(ir.wire_dims(), vec![2, 2]);
    assert_eq!(ir.qudit_registers.len(), 1);
    assert_eq!(ir.qudit_registers[0].dims, vec![2, 2]);
    assert!(ir.first_qudit().is_none());
    assert!(ir.refuse_qudits("test").is_ok());
    // Both spellings of the bare register lower to the same wires and the
    // same gates: that is the equivalence MQT's re-emission asserts.
    let bare = lower_to_ir(&fixture("bare_qreg_input")).unwrap();
    assert_eq!(bare.num_qubits, ir.num_qubits);
    assert_eq!(bare.ops.len(), 2);
    // (the re-emitted file appends MQT's measures; the two `h` are shared)
    assert!(ir.ops.iter().filter(|o| o.qubits.len() == 1).count() >= 2);
}

/// Two named registers, mixed dimensions, non-contiguous: `matter [2][3,3]`
/// then `field [1][7]`. Wire offsets must follow declaration order exactly
/// as MQT's sitemap does, or `matter[1]` and `field[0]` swap identities.
#[test]
fn two_registers_keep_declaration_order_and_offsets() {
    // Q1 does not lower `csum`, so this file stops at its first gate; the
    // registers are read before any statement is lowered, and that is what
    // is under test. Use the header + declarations only.
    let src = fixture("two_regs");
    let decls: String = src
        .lines()
        .take_while(|l| !l.starts_with("h "))
        .map(|l| format!("{l}\n"))
        .collect();
    let ir = lower_to_ir(&decls).unwrap();
    assert_eq!(ir.num_qubits, 3);
    assert_eq!(ir.wire_dims(), vec![3, 3, 7]);
    assert_eq!(
        ir.qudit_registers,
        vec![
            QuditRegister {
                name: "matter".into(),
                start: 0,
                dims: vec![3, 3]
            },
            QuditRegister {
                name: "field".into(),
                start: 2,
                dims: vec![7]
            },
        ]
    );
    let (reg, wire, d) = ir.first_qudit().unwrap();
    assert_eq!((reg.name.as_str(), wire, d), ("matter", 0, 3));
}

/// The mixed-radix fixture: `[3][3,2,5]`. Wire 1 is a qubit inside a qudit
/// register — `wire_dim` must answer per wire, not per register.
#[test]
fn mixed_radix_register_answers_per_wire() {
    let src = fixture("mixed_325");
    let decls: String = src
        .lines()
        .take_while(|l| !l.starts_with("h "))
        .map(|l| format!("{l}\n"))
        .collect();
    let ir = lower_to_ir(&decls).unwrap();
    assert_eq!(ir.wire_dims(), vec![3, 2, 5]);
    assert_eq!(ir.wire_dim(1), 2);
    assert_eq!(ir.wire_dim(2), 5);
    assert_eq!(ir.wire_dim(99), 2, "an out-of-range wire is not a qudit");
}

// ---------------------------------------------------------------------------
// Every MQT-written fixture reads its registers. Whether the GATES lower is
// Q2+'s business; what Q1 promises is that no fixture fails on the header
// or the register line.
// ---------------------------------------------------------------------------

#[test]
fn every_fixture_gets_past_its_register_declarations() {
    let prov: serde_json::Value = serde_json::from_str(&fixture_raw("PROVENANCE.json")).unwrap();
    let files = prov["files"].as_object().expect("PROVENANCE.files");
    assert!(files.len() >= 8, "Q0 committed eight fixtures");
    for name in files.keys() {
        let src = fixture(name);
        let decls: String = src
            .lines()
            .take_while(|l| {
                let t = l.trim();
                t.is_empty()
                    || t.starts_with("DITQASM")
                    || t.starts_with("qreg")
                    || t.starts_with("creg")
                    || t.starts_with("//")
            })
            .map(|l| format!("{l}\n"))
            .collect();
        let ir = lower_to_ir(&decls).unwrap_or_else(|e| panic!("{name}: registers must read: {e}"));
        // PROVENANCE records `dims` for MQT-emitted files and `parsed_dims`
        // for the hand-written bare input; the two derived files
        // (`bare_qreg_reemitted`, `ctl_hand_corrected`) record neither and
        // are covered by name above. Only assert where Q0 wrote a number.
        let recorded = prov["files"][name]["dims"]
            .as_array()
            .or_else(|| prov["files"][name]["parsed_dims"].as_array())
            .map(|a| {
                a.iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect::<Vec<u32>>()
            });
        if let Some(want) = recorded {
            assert_eq!(ir.wire_dims(), want, "{name}: dims vs PROVENANCE");
        }
    }
}

fn fixture_raw(file: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ditqasm")
        .join(file);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

// ---------------------------------------------------------------------------
// Refusals in the parser, naming the register.
// ---------------------------------------------------------------------------

#[test]
fn a_dimension_group_of_the_wrong_length_is_refused_naming_the_register() {
    let err = lower_to_ir("DITQASM 2.0;\nqreg q [3][3,3];\n").unwrap_err();
    assert!(err.contains("qreg q"), "{err}");
    assert!(err.contains("size 3"), "{err}");
    assert!(err.contains("2 entries"), "{err}");
}

#[test]
fn a_dimension_below_two_is_refused() {
    for d in ["0", "1"] {
        let err = lower_to_ir(&format!("DITQASM 2.0;\nqreg q [1][{d}];\n")).unwrap_err();
        assert!(err.contains("qreg q"), "{err}");
        assert!(err.contains(&format!("dimension {d}")), "{err}");
    }
}

/// `OPENQASM 2.0;` never had a dimension group, and Q1 does not give it
/// one: the group is DITQASM's, and a QASM 2 reader that accepted it would
/// be a fourth dialect nobody else can read.
#[test]
fn openqasm_header_does_not_accept_a_dimension_group() {
    assert!(lower_to_ir("OPENQASM 2.0;\nqreg q [1][3];\n").is_err());
    // …and the plain QASM 2 register is unchanged.
    let ir = lower_to_ir("OPENQASM 2.0;\nqreg q[3];\n").unwrap();
    assert_eq!(ir.wire_dims(), vec![2, 2, 2]);
    assert!(ir.qudit_registers.is_empty());
}

/// The version is locked upstream to the literal `DITQASM 2.0;` (Q0). We
/// read the version through the same optional-minor rule QASM 2 uses, so
/// `DITQASM 2;` is tolerated as `OPENQASM 2;` is; what we do NOT do is
/// invent a `3.0`.
#[test]
fn ditqasm_header_is_the_only_new_token() {
    assert!(lower_to_ir("DITQASM 2.0;\nqreg q [1][3];\n").is_ok());
    assert!(lower_to_ir("DITQASM 2;\nqreg q [1][3];\n").is_ok());
}

// ---------------------------------------------------------------------------
// Q2: DITQASM's own gates lower under its header, and only there.
// ---------------------------------------------------------------------------

#[test]
fn rxy_and_csum_lower_under_the_ditqasm_header_only() {
    use omega_core::circuit::{GateKind, ParamExpr};
    let ir = lower_to_ir(
        "DITQASM 2.0;\nqreg q [2][3,5];\nrxy (0, 2, 1.2, 0.3) q[1];\ncsum q[0], q[1];\n",
    )
    .unwrap();
    assert_eq!(ir.ops.len(), 2);
    assert_eq!(ir.ops[0].gate, GateKind::Rxy);
    assert_eq!(ir.ops[0].qubits.len(), 1);
    let p: Vec<f64> = ir.ops[0]
        .params
        .iter()
        .map(|e| match e {
            ParamExpr::Concrete(v) => *v,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(p, vec![0.0, 2.0, 1.2, 0.3]);
    assert_eq!(ir.ops[1].gate, GateKind::CSum);
    assert_eq!(ir.ops[1].qubits.len(), 2);
    // Under OPENQASM the same names are not defined — and say why.
    let err = lower_to_ir("OPENQASM 2.0;\nqreg q[2];\ncsum q[0], q[1];\n").unwrap_err();
    assert!(err.contains("DITQASM") && err.contains("csum"), "{err}");
    let err = lower_to_ir("OPENQASM 2.0;\nqreg q[1];\nrxy (0, 1, 1.0, 0.0) q[0];\n").unwrap_err();
    assert!(err.contains("DITQASM") && err.contains("rxy"), "{err}");
    // Arity is checked like any builtin.
    let err = lower_to_ir("DITQASM 2.0;\nqreg q [1][3];\nrxy (0, 1) q[0];\n").unwrap_err();
    assert!(err.contains("4 parameter"), "{err}");
}

/// mqt.qudits writes `cx (l_a, l_b, ctrl, phi)` even on qubits (Q0): an
/// embedded two-level controlled gate that Q2 does not lower. The refusal
/// must name `csum` as the generalised CX rather than leave the user with
/// "cx takes 0 parameters, got 4".
#[test]
fn ditqasm_four_parameter_cx_is_refused_pointing_at_csum() {
    let err = lower_to_ir(&fixture("qubits_default")).unwrap_err();
    assert!(err.contains("csum"), "{err}");
    assert!(err.contains("two-level"), "{err}");
}

/// Two of Q0's MQT-written fixtures use only gates Q2 lowers, so they must
/// now lower COMPLETELY — registers and gates — not just past the
/// declarations.
#[test]
fn fixtures_within_q2s_gate_set_lower_completely() {
    use omega_core::circuit::GateKind;
    let ir = lower_to_ir(&fixture("qutrit_h")).unwrap();
    assert_eq!(ir.wire_dims(), vec![3]);
    let kinds: Vec<_> = ir.ops.iter().map(|o| o.gate.clone()).collect();
    assert_eq!(kinds, vec![GateKind::H, GateKind::Measure]);
    let ir = lower_to_ir(&fixture("two_regs")).unwrap();
    assert_eq!(ir.wire_dims(), vec![3, 3, 7]);
    let kinds: Vec<_> = ir.ops.iter().map(|o| o.gate.clone()).collect();
    assert_eq!(
        kinds,
        vec![
            GateKind::H,
            GateKind::CSum,
            GateKind::X,
            GateKind::Measure,
            GateKind::Measure,
            GateKind::Measure
        ]
    );
    // csum matter[1], field[0] → wires 1 and 2, control first.
    let cs = &ir.ops[1];
    assert_eq!(
        cs.qubits.iter().map(|q| q.0).collect::<Vec<_>>(),
        vec![1, 2]
    );
}
