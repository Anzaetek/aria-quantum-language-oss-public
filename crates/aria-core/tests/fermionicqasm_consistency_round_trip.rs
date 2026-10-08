// SPDX-License-Identifier: Apache-2.0
//! F4.4 — writer round-trip. **Consistency only, not validation.**
//!
//! `parse(write(x)) == x` shows that this emitter and the readers agree with
//! each other. It does not show that a gate means what FermionicQASM says it
//! means. There is no external oracle for the spelling; the semantics evidence
//! is F4.3's ffsim gate (`ffsim_vs_fermionicqasm`), which compares lowered
//! circuits to ffsim and does not consult this writer. A green result here
//! must not be quoted as that.
//!
//! The writer refuses what the lowering refuses. Emitting cross-spin or
//! non-adjacent `givens`/`tunnel`, or `orbrot`, would produce text the reader
//! accepts and the lowering then rejects — a file that looks done and is not.

use aria_core::ast::expr::ParamExpr;
use aria_core::ast::fermionicqasm::{from_fermionicqasm, to_fermionicqasm};
use aria_core::ast::nodes::*;
use omega_parser::ast::{FermionicQasmProgram, FermionicQasmStmt};
use omega_parser::parse_fermionicqasm;
use std::path::PathBuf;

fn fixtures() -> Vec<(String, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../omega-parser/tests/fixtures/fermionicqasm");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("fqasm"))
        .collect();
    files.sort();
    let out: Vec<_> = files
        .iter()
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{name}: {e}"));
            (name, text)
        })
        .collect();
    assert_eq!(
        out.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        vec!["all_forms.fqasm", "spinless.fqasm"],
        "expected both FermionicQASM fixture programs"
    );
    out
}

/// Write → read → write, byte-identical.
///
/// The name is the claim. Agreement of our emitter with our reader is a
/// fixed point of the spelling. It is not a statement about ffsim, about the
/// Jordan–Wigner matrices, or about any circuit this text lowers to.
#[test]
fn write_read_write_of_every_fixture_is_byte_identical_consistency_not_validation() {
    for (name, src) in fixtures() {
        let circuit = from_fermionicqasm(&src).unwrap_or_else(|e| panic!("{name}: {e}"));
        let once = to_fermionicqasm(&circuit).unwrap_or_else(|e| panic!("{name}: emit: {e}"));
        let back =
            from_fermionicqasm(&once).unwrap_or_else(|e| panic!("{name}: re-read: {e}\n{once}"));
        let twice = to_fermionicqasm(&back).unwrap_or_else(|e| panic!("{name}: re-emit: {e}"));
        assert_eq!(
            once, twice,
            "{name}: write → read → write is not byte-identical.\n--- once ---\n{once}\n--- twice ---\n{twice}"
        );
        // The fixture's own bytes include comments, so they are not the
        // canonical emission. The programs have to be the same statements.
        let from_src = parse_fermionicqasm(&src).unwrap_or_else(|e| panic!("{name}: pest: {e}"));
        let from_emit =
            parse_fermionicqasm(&once).unwrap_or_else(|e| panic!("{name}: pest emit: {e}\n{once}"));
        assert_eq!(
            from_src, from_emit,
            "{name}: emission is not the same program as the fixture.\n{once}"
        );
        // And the lowering accepts it. A writer that emits text the lowering
        // rejects fails here, on the fixtures that are supposed to be legal.
        omega_parser::lower_to_ir(&once)
            .unwrap_or_else(|e| panic!("{name}: lowering refused the emission: {e}\n{once}"));
    }
}

/// Each spelling, separately, so a drift in one gate is that gate's failure
/// and not a single fixture assertion.
#[test]
fn every_emitted_spelling_is_read_back_consistency_not_validation() {
    spell("num", |c, m, _| {
        c.apply(
            GateDef::with_params(GateKind::P, vec![0.37]),
            vec![m[0].clone()],
        );
    });
    spell("numnum", |c, m, _| {
        c.apply(
            GateDef::with_params(GateKind::CP, vec![0.61]),
            vec![m[0].clone(), m[3].clone()],
        );
    });
    spell("givens", |c, m, _| {
        c.apply(
            GateDef::with_params(GateKind::RBS, vec![0.83]),
            vec![m[0].clone(), m[1].clone()],
        );
    });
    spell("tunnel", |c, m, _| {
        c.apply(
            GateDef::with_params(GateKind::Tunnel, vec![-0.29]),
            vec![m[2].clone(), m[3].clone()],
        );
    });
    spell("load", |c, m, _| {
        c.apply(GateDef::new(GateKind::Load), vec![m[1].clone()]);
    });
    spell("measure", |c, m, bits| {
        c.measure(&m[0], &bits[0]);
    });
}

fn spell(spelling: &str, build: impl FnOnce(&mut Circuit, &[Qubit], &[Clbit])) {
    let mut c = Circuit::new("c");
    let m = c.mode_reg("m", 4, false);
    let bits = c.creg("c", 2);
    build(&mut c, &m, &bits);
    let text = to_fermionicqasm(&c).unwrap_or_else(|e| panic!("{spelling}: {e}"));
    assert!(
        text.contains(spelling),
        "{spelling}: not in the emission:\n{text}"
    );
    let back = from_fermionicqasm(&text).unwrap_or_else(|e| panic!("{spelling}: {e}\n{text}"));
    let again = to_fermionicqasm(&back).unwrap_or_else(|e| panic!("{spelling}: {e}"));
    assert_eq!(
        text, again,
        "{spelling}: not a fixed point\n{text}\n{again}"
    );
    let ir = omega_parser::lower_to_ir(&text)
        .unwrap_or_else(|e| panic!("{spelling}: omega-parser refused it: {e}\n{text}"));
    assert!(!ir.ops.is_empty(), "{spelling}: lowered to nothing\n{text}");
    let pest = parse_fermionicqasm(&text).unwrap();
    assert!(
        program_mentions(&pest, spelling),
        "{spelling}: pest did not keep the statement:\n{text}"
    );
}

fn program_mentions(prog: &FermionicQasmProgram, spelling: &str) -> bool {
    prog.statements.iter().any(|s| match s {
        FermionicQasmStmt::Load { .. } => spelling == "load",
        FermionicQasmStmt::Measure { .. } => spelling == "measure",
        FermionicQasmStmt::GateApp(app) => app.name == spelling,
        _ => false,
    })
}

/// Non-adjacent `numnum` is legal. A blanket adjacency check would refuse it,
/// and this is the case that shows the hop check was not applied to number
/// operators.
#[test]
fn nonadjacent_numnum_is_emitted_and_lowered() {
    let mut c = Circuit::new("c");
    let m = c.mode_reg("m", 4, false);
    c.apply(
        GateDef::with_params(GateKind::CP, vec![0.7]),
        vec![m[0].clone(), m[2].clone()],
    );
    let text = to_fermionicqasm(&c).expect("non-adjacent numnum is legal");
    assert!(text.contains("numnum(0.7) m[0], m[2];"), "{text}");
    omega_parser::lower_to_ir(&text).unwrap_or_else(|e| panic!("lowering refused it: {e}\n{text}"));
}

#[test]
fn writer_refuses_nonadjacent_givens_and_tunnel() {
    for kind in [GateKind::RBS, GateKind::Tunnel] {
        let mut c = Circuit::new("c");
        let m = c.mode_reg("m", 4, false);
        c.apply(
            GateDef::with_params(kind, vec![0.7]),
            vec![m[0].clone(), m[2].clone()],
        );
        let err = to_fermionicqasm(&c).expect_err("non-adjacent hop must not be emitted");
        assert!(
            err.contains("Z string"),
            "{kind:?} must be refused as a Z-string gap, got: {err}"
        );
        assert!(
            !err.contains("unpinned") && !err.contains("cross-spin"),
            "{kind:?} on a spinless register is not the cross-spin refusal: {err}"
        );
    }
}

/// Adjacent across the block boundary (`m[1], m[2]` on `mode m[2] spin`).
/// `|p-q| == 1`, so the Z-string check would accept it. The refusal has to
/// be the unpinned cross-spin check. Same-spin pairs on the same register
/// still emit.
#[test]
fn writer_refuses_unpinned_cross_spin_givens_and_tunnel() {
    for kind in [GateKind::RBS, GateKind::Tunnel] {
        let mut c = Circuit::new("c");
        let m = c.mode_reg("m", 2, true);
        c.apply(
            GateDef::with_params(kind, vec![0.7]),
            vec![m[1].clone(), m[2].clone()],
        );
        let err = to_fermionicqasm(&c).expect_err("cross-spin hop must not be emitted");
        assert!(
            err.contains("unpinned") && err.contains("cross-spin"),
            "{kind:?} must be refused as unpinned cross-spin, got: {err}"
        );
        assert!(
            !err.contains("Z string"),
            "{kind:?} across the block boundary is adjacent; the Z-string message is the wrong refusal: {err}"
        );
    }

    let mut ok = Circuit::new("c");
    let m = ok.mode_reg("m", 2, true);
    ok.apply(
        GateDef::with_params(GateKind::RBS, vec![0.3]),
        vec![m[0].clone(), m[1].clone()],
    );
    ok.apply(
        GateDef::with_params(GateKind::Tunnel, vec![-0.45]),
        vec![m[2].clone(), m[3].clone()],
    );
    let text = to_fermionicqasm(&ok).expect("same-spin givens and tunnel are pinned");
    assert!(text.contains("mode m[2] spin;"), "{text}");
    omega_parser::lower_to_ir(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
}

#[test]
fn orbrot_is_not_in_1_0() {
    let src = "FERMIONICQASM 1.0;\nmode m[2];\norbrot(0.7) m[0], m[1];\n";
    let err = from_fermionicqasm(src).expect_err("orbrot must not parse as a 1.0 gate");
    assert!(
        err.contains("orbrot is not in FermionicQASM 1.0"),
        "the refusal must name the omission, got: {err}"
    );
    let pest = parse_fermionicqasm(src).expect("the grammar admits the name; the lowering refuses");
    let lowered = omega_parser::lower_to_ir(src).expect_err("lowering must refuse orbrot");
    assert!(
        lowered.contains("orbrot is not in FermionicQASM 1.0"),
        "{lowered}"
    );
    // The pest AST holding the name is not a license to emit it. Nothing
    // this writer prints contains the spelling.
    let _ = pest;
    let mut c = Circuit::new("c");
    let m = c.mode_reg("m", 2, false);
    c.apply(
        GateDef::with_params(GateKind::RBS, vec![0.7]),
        vec![m[0].clone(), m[1].clone()],
    );
    let text = to_fermionicqasm(&c).unwrap();
    assert!(!text.contains("orbrot"), "{text}");
}

#[test]
fn writer_refuses_a_load_after_a_gate_and_a_doubled_load() {
    let mut c = Circuit::new("c");
    let m = c.mode_reg("m", 2, false);
    c.apply(
        GateDef::with_params(GateKind::P, vec![0.2]),
        vec![m[0].clone()],
    );
    c.apply(GateDef::new(GateKind::Load), vec![m[1].clone()]);
    let err = to_fermionicqasm(&c).expect_err("load after a gate must not be emitted");
    assert!(err.contains("after a gate"), "{err}");

    let mut d = Circuit::new("c");
    let m = d.mode_reg("m", 2, false);
    d.apply(GateDef::new(GateKind::Load), vec![m[0].clone()]);
    d.apply(GateDef::new(GateKind::Load), vec![m[0].clone()]);
    let err = to_fermionicqasm(&d).expect_err("a doubled load must not be emitted");
    assert!(err.contains("already loaded"), "{err}");
}

#[test]
fn writer_refuses_a_qubit_gate_rather_than_commenting_it_out() {
    let mut c = Circuit::new("c");
    let m = c.mode_reg("m", 1, false);
    c.apply(GateDef::new(GateKind::H), vec![m[0].clone()]);
    let err = to_fermionicqasm(&c).expect_err("H is not a fermionic gate");
    assert!(err.contains("H"), "{err}");
    assert!(
        !err.contains("//"),
        "a comment is an emission that re-imports without the gate: {err}"
    );
}

#[test]
fn writer_refuses_an_expression_parameter_rather_than_evaluating_it() {
    let mut c = Circuit::new("c");
    let m = c.mode_reg("m", 1, false);
    c.apply(
        GateDef::with_exprs(GateKind::P, vec![ParamExpr::pi()]),
        vec![m[0].clone()],
    );
    let err = to_fermionicqasm(&c).expect_err("pi is not a FermionicQASM parameter spelling");
    assert!(err.contains("number") || err.contains("$name"), "{err}");
}

/// Symbols are in the grammar. A `$name` round-trips as a `$name`, not as 0.
#[test]
fn a_symbol_round_trips_as_a_symbol_consistency_not_validation() {
    let mut c = Circuit::new("c");
    let m = c.mode_reg("m", 1, false);
    c.apply(
        GateDef::with_exprs(GateKind::P, vec![ParamExpr::symbol("theta")]),
        vec![m[0].clone()],
    );
    let text = to_fermionicqasm(&c).unwrap();
    assert!(text.contains("num($theta) m[0];"), "{text}");
    let back = from_fermionicqasm(&text).unwrap();
    assert_eq!(to_fermionicqasm(&back).unwrap(), text);
}
