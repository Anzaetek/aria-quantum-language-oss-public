// SPDX-License-Identifier: Apache-2.0
//! **This importer reads what this exporter writes, and follows a named Qiskit
//! dialect for the three spellings that are not in qelib1.**
//!
//! Two defects are pinned here, and the second one was hiding behind the first.
//!
//! # 1. The emitter's own output was unreadable by its own reader
//!
//! `to_qasm` writes a `gate` definition for every spelling qelib1 lacks —
//! `rxx`, `ryy`, `rzz`, `swap`, `cswap` — precisely so the file loads under
//! *both* Qiskit readers. `from_qasm` then rejected the definition line as
//! "multiple statements on line N", because a definition is one line carrying
//! several `;`. So all five gates failed `to_qasm` -> `from_qasm`, while Qiskit
//! read the very same file without complaint.
//!
//! # 2. The reader accepted exactly what the writer never emits
//!
//! Before the fix `from_qasm` accepted a **bare** `ryy` — which no Qiskit
//! reader accepts and which `to_qasm` never writes — and rejected the defined
//! form, which every reader accepts and which `to_qasm` always writes. Exactly
//! inverted.
//!
//! Measured on qiskit 2.5.1:
//!
//! ```text
//!               strict qasm2.loads   legacy from_qasm_str
//!   bare rxx    reject               accept
//!   bare rzz    reject               accept
//!   bare ryy    reject               reject
//!   with a gate def              accept (both)
//! ```

use aria_core::ast::nodes::*;
use aria_core::ast::qasm::{from_qasm, from_qasm_with_dialect, to_qasm, Qasm2Dialect};

fn circuit_with(kind: GateKind, n: usize, params: Vec<f64>) -> Circuit {
    let mut c = Circuit::new("C");
    c.qreg("q", n);
    c.apply(
        GateDef::with_params(kind, params),
        (0..n).map(|i| Qubit::new("q", i)).collect(),
    );
    c
}

fn bare(name: &str) -> String {
    format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\n{name}(0.7) q[0], q[1];\n")
}

/// **Every gate that needs a preamble definition must survive its own round
/// trip.** This is the regression: all five failed before.
#[test]
fn every_definition_carrying_gate_round_trips() {
    let cases = [
        (GateKind::RXX, 2, vec![0.7]),
        (GateKind::RYY, 2, vec![0.7]),
        (GateKind::RZZ, 2, vec![0.7]),
        (GateKind::SWAP, 2, vec![]),
        (GateKind::CSWAP, 3, vec![]),
    ];
    for (kind, n, params) in cases {
        let src = circuit_with(kind, n, params.clone());
        let text = to_qasm(&src).unwrap_or_else(|e| panic!("{kind:?} must export: {e}"));
        assert!(
            text.contains("gate "),
            "{kind:?} was expected to need a preamble definition; got:\n{text}"
        );
        let back = from_qasm(&text)
            .unwrap_or_else(|e| panic!("{kind:?}: emitter output must re-import: {e}\n{text}"));
        assert_eq!(
            back.instructions.len(),
            1,
            "{kind:?} round-tripped to {} instruction(s):\n{text}",
            back.instructions.len()
        );
        assert_eq!(
            back.instructions[0].gate.kind, kind,
            "{kind:?} came back as {:?}",
            back.instructions[0].gate.kind
        );
    }
}

/// A definition licenses the name in **every** dialect — including `Strict`,
/// which is the whole reason the emitter writes one.
#[test]
fn a_carried_definition_is_read_in_every_dialect() {
    for dialect in [
        Qasm2Dialect::Strict,
        Qasm2Dialect::Legacy,
        Qasm2Dialect::Lenient,
    ] {
        for kind in [GateKind::RXX, GateKind::RYY, GateKind::RZZ] {
            let text = to_qasm(&circuit_with(kind, 2, vec![0.7])).expect("export");
            from_qasm_with_dialect(&text, dialect)
                .unwrap_or_else(|e| panic!("{kind:?} in {dialect:?}: {e}"));
        }
    }
}

/// The acceptance matrix for a **bare** spelling, mirroring the parser lane's.
/// Both crates must answer identically or a file's readability would depend on
/// which reader it reached.
#[test]
fn bare_acceptance_matches_the_named_qiskit_reader() {
    let matrix = [
        (Qasm2Dialect::Strict, false, false, false),
        (Qasm2Dialect::Legacy, true, false, true),
        (Qasm2Dialect::Lenient, true, true, true),
    ];
    for (dialect, rxx_ok, ryy_ok, rzz_ok) in matrix {
        for (name, want) in [("rxx", rxx_ok), ("ryy", ryy_ok), ("rzz", rzz_ok)] {
            let got = from_qasm_with_dialect(&bare(name), dialect);
            assert_eq!(
                got.is_ok(),
                want,
                "bare `{name}` in {dialect:?}: expected ok={want}, got {got:?}"
            );
        }
    }
}

/// The default must be `Legacy`, and it must be the same default the parser
/// lane uses.
#[test]
fn the_default_dialect_is_legacy() {
    assert_eq!(Qasm2Dialect::default(), Qasm2Dialect::Legacy);
    assert!(from_qasm(&bare("rxx")).is_ok());
    assert!(from_qasm(&bare("rzz")).is_ok());
    assert!(
        from_qasm(&bare("ryy")).is_err(),
        "a bare `ryy` is refused by every Qiskit reader and must be refused here"
    );
}

/// A definition body that is **not** the canonical one must be refused, not
/// silently replaced by the built-in operator. Reading the name while ignoring
/// a body that means something else is the silent-substitution defect this
/// importer is repeatedly audited for.
#[test]
fn a_non_canonical_definition_body_is_refused_not_substituted() {
    // Structurally a valid definition, but the body is an X on each qubit —
    // nothing like a ZZ rotation.
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate rzz(param0) q0,q1 { x q0; x q1; }\nqreg q[2];\n\
               rzz(0.7) q[0], q[1];\n";
    let err = from_qasm(src).expect_err("a foreign body must not be silently ignored");
    assert!(
        err.contains("rzz") && (err.contains("body") || err.contains("implement")),
        "the refusal must name the gate and the body; got: {err}"
    );
}

/// Multi-line definitions are read too — the emitter writes one line, but a
/// hand-written or foreign file need not.
#[test]
fn a_multi_line_definition_is_read() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate rzz(param0) q0,q1 {\n  cx q0,q1;\n  rz(param0) q1;\n  cx q0,q1;\n}\n\
               qreg q[2];\nrzz(0.7) q[0], q[1];\n";
    let c = from_qasm(src).expect("a multi-line canonical definition must be read");
    assert_eq!(c.instructions.len(), 1);
    assert_eq!(c.instructions[0].gate.kind, GateKind::RZZ);
}

/// An unterminated definition must be an error, not a silent truncation of the
/// rest of the file.
#[test]
fn an_unterminated_definition_is_refused() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate rzz(param0) q0,q1 {\n  cx q0,q1;\nqreg q[2];\n";
    let err = from_qasm(src).expect_err("unterminated definition must be refused");
    assert!(err.contains("unterminated"), "got: {err}");
}

/// Stripping definitions must not shift line numbers in later diagnostics —
/// an error that points at the wrong line sends the reader to the wrong place.
#[test]
fn line_numbers_survive_definition_stripping() {
    // The bad statement is on line 6.
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
               gate rzz(param0) q0,q1 { cx q0,q1; rz(param0) q1; cx q0,q1; }\n\
               qreg q[2];\nrzz(0.7) q[0], q[1];\nnotagate q[0];\n";
    let err = from_qasm(src).expect_err("the bogus gate must be refused");
    assert!(
        err.contains("line 6"),
        "diagnostic must still point at line 6; got: {err}"
    );
}
