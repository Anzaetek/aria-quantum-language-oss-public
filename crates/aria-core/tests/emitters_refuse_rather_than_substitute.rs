// SPDX-License-Identifier: Apache-2.0
//! **An emitter must refuse what it cannot express, never quietly write
//! something else.**
//!
//! `to_qasm` was fixed for this: a symbolic angle used to leave through
//! `.try_as_f64().unwrap_or(0.0)` as a literal `0`, so `rz(theta)` was exported
//! as `rz(0)` — a well-formed file describing a *different circuit*, with no
//! diagnostic at either end. Its two siblings kept the defect:
//!
//! * `to_qasm3` — same `unwrap_or(0.0)`, plus `// unsupported gate:` comments
//!   for anything without a spelling.
//! * `to_aria_source` — same again, and this is the export where it matters
//!   most. That file's own header says Aria → Aria "is the one export where a
//!   reader is most likely to assume fidelity, which makes the silence worse
//!   rather than better", and then it emitted `-- unsupported gate:` anyway.
//!
//! A silently-zeroed angle is worse than an omitted gate: an omission at least
//! changes the operation count, while `rz(0)` looks exactly like a circuit
//! someone meant to write.

use aria_core::ast::expr::ParamExpr;
use aria_core::ast::nodes::*;
use aria_core::ast::{to_aria_source, to_qasm, to_qasm3};

/// The same defect, on the gate that **expands to its own statements**.
///
/// `RBS` never reaches the generic parameter path that
/// [`a_symbolic_angle_is_refused_by_every_emitter`] exercises: both QASM
/// emitters have a dedicated `GateKind::RBS` arm that writes the eight-line
/// `(H⊗H)·CZ·(Ry∓θ)·CZ·(H⊗H)` decomposition itself. Those arms kept
/// `.unwrap_or(0.0)` long after the generic path was fixed, so a symbolic angle
/// was emitted as `rbs(0)` — eight well-formed lines describing a *different*
/// circuit, with no diagnostic on either side.
///
/// It survived because the test above only ever used `RZ`. A fix verified
/// through one arm of a match says nothing about the others.
fn symbolic_rbs() -> Circuit {
    let mut c = Circuit::new("SymRbs");
    c.qreg("q", 2);
    c.instructions.push(Instruction {
        gate: GateDef {
            kind: GateKind::RBS,
            params: vec![ParamExpr::Symbol("theta".to_string())],
            label: None,
        },
        qubits: vec![Qubit::new("q", 0), Qubit::new("q", 1)],
        clbits: vec![],
        condition: None,
    });
    c
}

/// A symbolic **RBS** angle is refused by both QASM emitters, and in
/// particular is never written as `0`.
#[test]
fn a_symbolic_rbs_angle_is_refused_not_zeroed() {
    let c = symbolic_rbs();

    for (who, r) in [("to_qasm", to_qasm(&c)), ("to_qasm3", to_qasm3(&c))] {
        match r {
            Ok(text) => panic!(
                "{who} must refuse a symbolic RBS angle, but emitted a file. \
                 If it contains `rbs(0)` or `ry(0)` this is the silent-zero \
                 defect:\n{text}"
            ),
            Err(e) => {
                assert!(
                    e.contains("symbolic"),
                    "{who} must name the cause; got: {e}"
                );
                assert!(
                    e.to_lowercase().contains("bind"),
                    "{who} must say to bind the parameter; got: {e}"
                );
            }
        }
    }
}

/// The refusal must be about the *symbol*, not about RBS. A concrete angle
/// still emits, or the fix above is an over-refusal that silently dropped a
/// working gate.
#[test]
fn a_concrete_rbs_angle_still_emits() {
    let mut c = Circuit::new("Rbs");
    let q = c.qreg("q", 2);
    c.apply(GateDef::with_params(GateKind::RBS, vec![0.7]), q.to_vec());

    // The two emitters format differently on purpose — `to_qasm` pretty-prints
    // to fixed precision, `to_qasm3` writes the shortest round-tripping decimal
    // — so assert on the angle being *present and non-zero*, not on its
    // spelling. `rbs(0)` / `ry(0)` is the shape being excluded.
    for (who, text) in [
        (
            "to_qasm",
            to_qasm(&c).expect("concrete RBS must emit in 2.0"),
        ),
        (
            "to_qasm3",
            to_qasm3(&c).expect("concrete RBS must emit in 3.0"),
        ),
    ] {
        assert!(
            text.contains("0.7"),
            "{who}: the decomposition must carry the angle:\n{text}"
        );
        assert!(
            !text.contains("ry(0)") && !text.contains("rbs(0)"),
            "{who}: a zeroed angle is the defect this guards:\n{text}"
        );
    }
}

/// A circuit whose rotation angle is a free symbol, not a number.
fn symbolic_angle() -> Circuit {
    let mut c = Circuit::new("Sym");
    c.qreg("q", 1);
    c.instructions.push(Instruction {
        gate: GateDef {
            kind: GateKind::RZ,
            params: vec![ParamExpr::Symbol("theta".to_string())],
            label: None,
        },
        qubits: vec![Qubit::new("q", 0)],
        clbits: vec![],
        condition: None,
    });
    c
}

/// **All three emitters refuse a symbolic angle.** None may write `0`.
#[test]
fn a_symbolic_angle_is_refused_by_every_emitter() {
    let c = symbolic_angle();

    let q2 = to_qasm(&c);
    assert!(q2.is_err(), "to_qasm must refuse; got: {q2:?}");

    let q3 = to_qasm3(&c);
    assert!(q3.is_err(), "to_qasm3 must refuse; got: {q3:?}");

    let aria = to_aria_source(&c, "Sym");
    assert!(aria.is_err(), "to_aria_source must refuse; got: {aria:?}");

    // And each must say what to do about it, not merely fail.
    for (who, e) in [
        ("to_qasm", q2.unwrap_err()),
        ("to_qasm3", q3.unwrap_err()),
        ("to_aria_source", aria.unwrap_err()),
    ] {
        assert!(
            e.contains("symbolic"),
            "{who} must name the cause; got: {e}"
        );
        assert!(
            e.to_lowercase().contains("bind"),
            "{who} must say to bind the parameter; got: {e}"
        );
    }
}

/// The specific regression: the exported text must never contain a zeroed
/// rotation where a symbol was. Belt-and-braces against a future emitter that
/// returns `Ok` again.
#[test]
fn no_emitter_writes_a_zero_where_a_symbol_was() {
    let c = symbolic_angle();
    for out in [
        to_qasm(&c).ok(),
        to_qasm3(&c).ok(),
        to_aria_source(&c, "Sym").ok(),
    ]
    .into_iter()
    .flatten()
    {
        assert!(
            !out.contains("rz(0)") && !out.contains("RZ(0)"),
            "an emitter wrote a zeroed rotation for a symbolic angle:\n{out}"
        );
    }
}

/// **Concrete angles must still export.** A refusal that fires on ordinary
/// circuits would be worse than the bug it replaces.
#[test]
fn concrete_angles_still_export_everywhere() {
    let mut c = Circuit::new("Concrete");
    c.qreg("q", 2);
    c.apply(
        GateDef::with_params(GateKind::RZ, vec![0.7]),
        vec![Qubit::new("q", 0)],
    );
    c.apply(
        GateDef::with_params(GateKind::CX, vec![]),
        vec![Qubit::new("q", 0), Qubit::new("q", 1)],
    );
    to_qasm(&c).expect("qasm2 must still export a concrete circuit");
    to_qasm3(&c).expect("qasm3 must still export a concrete circuit");
    to_aria_source(&c, "Concrete").expect("aria must still export a concrete circuit");
}

/// **QASM3 carries a definition for the two-qubit rotations rather than
/// dropping them.**
///
/// `stdgates.inc` defines no `rxx`/`ryy`/`rzz` — that is a real property of the
/// OpenQASM 3 standard library, not an oversight here, and emitting the bare
/// names would produce a file a strict consumer rejects. The previous
/// behaviour was to comment them out, which produced a file that parses and is
/// missing an operation. Now the definition rides along, exactly as the 2.0
/// emitter already does.
#[test]
fn qasm3_defines_the_rotations_it_cannot_spell() {
    for (kind, name) in [
        (GateKind::RXX, "rxx"),
        (GateKind::RYY, "ryy"),
        (GateKind::RZZ, "rzz"),
    ] {
        let mut c = Circuit::new("Rot");
        c.qreg("q", 2);
        c.apply(
            GateDef::with_params(kind, vec![0.7]),
            vec![Qubit::new("q", 0), Qubit::new("q", 1)],
        );
        let out = to_qasm3(&c).unwrap_or_else(|e| panic!("{name} must export to QASM3: {e}"));
        assert!(
            out.contains(&format!("gate {name}(")),
            "{name} must carry its definition; got:\n{out}"
        );
        assert!(
            out.contains(&format!("{name}(")),
            "{name} must actually be called; got:\n{out}"
        );
        assert!(
            !out.contains("unsupported gate"),
            "{name} must not be commented out; got:\n{out}"
        );
    }
}

/// A gate with no spelling at all is a hard error in both fallible emitters —
/// never a comment.
#[test]
fn an_unspellable_gate_is_refused_not_commented() {
    let mut c = Circuit::new("Photonic");
    c.qreg("q", 2);
    c.apply(
        GateDef::with_params(GateKind::PolarizingBeamSplitter, vec![]),
        vec![Qubit::new("q", 0), Qubit::new("q", 1)],
    );
    let q3 = to_qasm3(&c);
    assert!(
        q3.is_err(),
        "an unspellable gate must be refused by to_qasm3; got:\n{q3:?}"
    );
    if let Err(e) = q3 {
        assert!(
            !e.contains("// unsupported"),
            "the refusal must be an error, not a comment: {e}"
        );
    }
}

/// Writes the QASM3 corpus for `tools/qiskit_xcheck/qasm3_dialect.py`, which
/// asserts the half this side cannot: that the file parses as **real**
/// OpenQASM 3 and builds the same operator qiskit's native gate does.
///
/// Deliberately not validated by feeding the output back through
/// `omega-parser`. That reader runs the QASM2 grammar and accepts names absent
/// from `stdgates.inc` (`cu1` among them), so a round trip through it would
/// certify a file a strict QASM3 consumer rejects — an emitter checked against
/// a reader that shares its blind spots.
///
/// # Why the corpus carries a CLASS
///
/// It used to be three files, each one gate, all unitary — so the Python side
/// could call `Operator(circuit)` unconditionally. That is exactly why the
/// invalid conditional shipped: `Operator()` **throws** on a circuit containing
/// `measure` or `reset`, so a corpus that can only hold unitaries can never
/// contain the emitter's riskiest output.
///
/// Each record is now `name<TAB>class<TAB>theta<TAB>path`, and the harness picks
/// its oracle from the class. `measured` files are checked for *loadability and
/// structure*; only `unitary` ones are compared as operators.
#[test]
fn write_the_qasm3_dialect_corpus() {
    use std::io::Write;
    const THETA: f64 = 0.7;
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target");
    std::fs::create_dir_all(&out_dir).expect("target dir");
    let mut index = String::new();

    // --- measured / conditional cases -------------------------------------
    //
    // The single-bit guard is the highest-risk thing this emitter writes and
    // had NO external check: it shipped as `if (c[0] == 1)`, which qiskit
    // refuses ("conditions must be 'bit == const bool' … not 'bit == const
    // int'"). Two Rust tests asserted that exact invalid string.
    for (name, circuit) in [
        ("conditional", conditional_circuit()),
        ("measured_ghz", measured_ghz()),
        ("reset_barrier", reset_and_barrier()),
    ] {
        let text = to_qasm3(&circuit).unwrap_or_else(|e| panic!("{name}: qasm3 export: {e}"));
        let path = out_dir.join(format!("qasm3_dialect_{name}.qasm3"));
        std::fs::write(&path, text).expect("write case");
        index.push_str(&format!("{name}\tmeasured\t0\t{}\n", path.display()));
    }

    for (kind, name) in [
        (GateKind::RXX, "rxx"),
        (GateKind::RYY, "ryy"),
        (GateKind::RZZ, "rzz"),
    ] {
        let mut c = Circuit::new("R");
        c.qreg("q", 2);
        c.apply(
            GateDef::with_params(kind, vec![THETA]),
            vec![Qubit::new("q", 0), Qubit::new("q", 1)],
        );
        let text = to_qasm3(&c).expect("qasm3 export");
        let path = out_dir.join(format!("qasm3_dialect_{name}.qasm3"));
        std::fs::write(&path, text).expect("write case");
        index.push_str(&format!("{name}\tunitary\t{THETA}\t{}\n", path.display()));
    }
    let mut f = std::fs::File::create(out_dir.join("qasm3_dialect_corpus.txt"))
        .expect("write corpus index");
    f.write_all(index.as_bytes()).expect("write corpus index");
}

/// `h q[0]; c[0] = measure q[0]; if (c[0] == true) x q[1];`
///
/// The emitter's riskiest output and, until now, the one with no external
/// check at all.
fn conditional_circuit() -> Circuit {
    let mut c = Circuit::new("Cond");
    c.qreg("q", 2);
    c.creg("c", 2);
    let cb = Clbit::new("c", 0);
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::H),
        qubits: vec![Qubit::new("q", 0)],
        clbits: vec![],
        condition: None,
    });
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::Measure),
        qubits: vec![Qubit::new("q", 0)],
        clbits: vec![cb.clone()],
        condition: None,
    });
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::X),
        qubits: vec![Qubit::new("q", 1)],
        clbits: vec![],
        condition: Some((cb, 1)),
    });
    c
}

/// A 3-qubit GHZ measured into a **non-identity, order-crossed** qubit→cbit
/// map: `q0→c2`, `q1→c1`, `q2→c0`.
///
/// The crossing is deliberate. `PLAN-WIDE-COUNTS.md` measured that with an
/// identity map the entire projection can be replaced by `cbit_of[q] = q` and
/// 145 tests still pass — so an identity fixture tests nothing about the
/// mapping.
fn measured_ghz() -> Circuit {
    let mut c = Circuit::new("Ghz");
    c.qreg("q", 3);
    c.creg("c", 3);
    c.apply(GateDef::new(GateKind::H), vec![Qubit::new("q", 0)]);
    c.apply(
        GateDef::new(GateKind::CX),
        vec![Qubit::new("q", 0), Qubit::new("q", 1)],
    );
    c.apply(
        GateDef::new(GateKind::CX),
        vec![Qubit::new("q", 1), Qubit::new("q", 2)],
    );
    for (q, b) in [(0usize, 2usize), (1, 1), (2, 0)] {
        c.instructions.push(Instruction {
            gate: GateDef::new(GateKind::Measure),
            qubits: vec![Qubit::new("q", q)],
            clbits: vec![Clbit::new("c", b)],
            condition: None,
        });
    }
    c
}

/// `reset` and `barrier` in one file — neither appears in any other corpus
/// entry, and `Operator()` rejects both, which is why they could not exist in
/// the corpus before it carried a class.
fn reset_and_barrier() -> Circuit {
    let mut c = Circuit::new("ResetBarrier");
    c.qreg("q", 2);
    c.creg("c", 2);
    c.apply(GateDef::new(GateKind::H), vec![Qubit::new("q", 0)]);
    c.apply(GateDef::new(GateKind::Reset), vec![Qubit::new("q", 0)]);
    c.apply(
        GateDef::new(GateKind::Barrier),
        vec![Qubit::new("q", 0), Qubit::new("q", 1)],
    );
    c.apply(
        GateDef::new(GateKind::CX),
        vec![Qubit::new("q", 0), Qubit::new("q", 1)],
    );
    c.instructions.push(Instruction {
        gate: GateDef::new(GateKind::Measure),
        qubits: vec![Qubit::new("q", 1)],
        clbits: vec![Clbit::new("c", 0)],
        condition: None,
    });
    c
}
