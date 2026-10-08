//! FERMIONICQASM 1.0 parser.
//!
//! This is an in-house format. It carries no authority of a standard: the
//! name has no prior art, and the grammar is not an adoption of anyone
//! else's dialect. Signs and spin order are pinned in
//! `tests/fixtures/fermionicqasm/conventions.json` (ffsim 0.0.84); this
//! module does not interpret them. Lowering is `lower_fermionicqasm`.

use pest::Parser;
use pest_derive::Parser;

use crate::ast::*;

#[derive(Parser)]
#[grammar = "fermionicqasm.pest"]
struct FermionicQasmParser;

/// The only version this reader accepts.
///
/// The grammar admits any `D` or `D.D` so a file from a later spec fails
/// here, where the message can name both the version it offered and the
/// version this reader implements. Same discipline as MQT's lock to the
/// literal `DITQASM 2.0;`. There is no external FermionicQASM corpus that
/// would justify accepting a neighbour version.
const ACCEPTED_VERSION: &str = "1.0";

/// Parse a FERMIONICQASM 1.0 source string into an AST.
pub fn parse_fermionicqasm(input: &str) -> Result<FermionicQasmProgram, String> {
    let pairs = FermionicQasmParser::parse(Rule::program, input).map_err(|e| format!("{e}"))?;

    let mut version = String::new();
    let mut statements = Vec::new();

    for pair in pairs {
        if pair.as_rule() != Rule::program {
            continue;
        }
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::header => {
                    for h in inner.into_inner() {
                        if h.as_rule() == Rule::version {
                            version = h.as_str().to_string();
                        }
                    }
                }
                Rule::statement => {
                    let stmt_inner = inner.into_inner().next().unwrap();
                    match stmt_inner.as_rule() {
                        Rule::mode_decl => {
                            let mut it = stmt_inner.into_inner();
                            let name = it.next().unwrap().as_str().to_string();
                            let size: u32 = it.next().unwrap().as_str().parse().unwrap();
                            // The `spin` marker is an optional trailing rule,
                            // so its presence is whether anything is left.
                            let spin = it.next().is_some();
                            statements.push(FermionicQasmStmt::ModeDecl { name, size, spin });
                        }
                        Rule::creg_decl => {
                            let mut it = stmt_inner.into_inner();
                            let name = it.next().unwrap().as_str().to_string();
                            let size: u32 = it.next().unwrap().as_str().parse().unwrap();
                            statements.push(FermionicQasmStmt::CregDecl { name, size });
                        }
                        Rule::load_stmt => {
                            let modes = parse_mode_list(stmt_inner.into_inner().next().unwrap());
                            statements.push(FermionicQasmStmt::Load { modes });
                        }
                        Rule::measure_stmt => {
                            let mut it = stmt_inner.into_inner();
                            let mode = parse_indexed(it.next().unwrap());
                            let cbit = parse_indexed(it.next().unwrap());
                            statements.push(FermionicQasmStmt::Measure { mode, cbit });
                        }
                        Rule::gate_app => {
                            let app = parse_gate_app(stmt_inner)?;
                            statements.push(FermionicQasmStmt::GateApp(app));
                        }
                        _ => {}
                    }
                }
                Rule::EOI => {}
                _ => {}
            }
        }
    }

    if version != ACCEPTED_VERSION {
        return Err(format!(
            "unsupported FERMIONICQASM version {version}; this reader accepts FERMIONICQASM {ACCEPTED_VERSION} only"
        ));
    }

    Ok(FermionicQasmProgram {
        version,
        statements,
    })
}

fn parse_gate_app(pair: pest::iterators::Pair<Rule>) -> Result<FermionicGateApp, String> {
    let mut name = String::new();
    let mut params = Vec::new();
    let mut modes = Vec::new();

    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::gate_name => {
                name = part.as_str().to_string();
            }
            Rule::param_list => {
                for p in part.into_inner() {
                    if p.as_rule() == Rule::param {
                        let inner = p.into_inner().next().unwrap();
                        match inner.as_rule() {
                            Rule::symbol => {
                                // Strip the '$' prefix.
                                let sym = inner.as_str();
                                let sym = sym.strip_prefix('$').unwrap_or(sym);
                                params.push(FermionicParam::Symbol(sym.to_string()));
                            }
                            Rule::number => {
                                let val: f64 = inner.as_str().parse().unwrap();
                                params.push(FermionicParam::Num(val));
                            }
                            _ => {}
                        }
                    }
                }
            }
            Rule::mode_list => {
                modes = parse_mode_list(part);
            }
            _ => {}
        }
    }

    Ok(FermionicGateApp {
        name,
        params,
        modes,
    })
}

fn parse_mode_list(pair: pest::iterators::Pair<Rule>) -> Vec<ModeRef> {
    pair.into_inner()
        .filter(|m| m.as_rule() == Rule::mode_ref)
        .map(parse_indexed)
        .collect()
}

fn parse_indexed(pair: pest::iterators::Pair<Rule>) -> ModeRef {
    let mut it = pair.into_inner();
    let reg = it.next().unwrap().as_str().to_string();
    let index: u32 = it.next().unwrap().as_str().parse().unwrap();
    ModeRef { reg, index }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn must_parse(src: &str) -> FermionicQasmProgram {
        parse_fermionicqasm(src).unwrap_or_else(|e| panic!("parse failed: {e}\n{src}"))
    }

    /// (i) The `.fqasm` fixtures parse as the statements they spell.
    ///
    /// A count of zero would pass with no files on disk. Parse success alone
    /// is not enough either: `load m[0];` is a legal gate application named
    /// `load` if `load_stmt` is missing, the same silent class as `creg`.
    #[test]
    fn fixture_files_parse() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fermionicqasm");
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("fqasm"))
            .collect();
        files.sort();
        let names: Vec<_> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec!["all_forms.fqasm".to_string(), "spinless.fqasm".to_string()],
            "expected the two FermionicQASM fixture programs in {}",
            dir.display()
        );
        for path in &files {
            let src =
                std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let prog = must_parse(&src);
            match path.file_name().and_then(|s| s.to_str()) {
                Some("all_forms.fqasm") => assert_all_forms(&prog),
                Some("spinless.fqasm") => assert_spinless(&prog),
                _ => unreachable!(),
            }
        }
    }

    fn assert_all_forms(prog: &FermionicQasmProgram) {
        assert_eq!(prog.version, "1.0");
        assert_eq!(prog.statements.len(), 9);
        assert_mode(&prog.statements[0], "m", 4, false);
        assert_mode(&prog.statements[1], "s", 2, true);
        assert_creg(&prog.statements[2], "c", 4);
        assert_load(&prog.statements[3], &[("m", 0), ("m", 1)]);
        assert_gate(
            &prog.statements[4],
            "num",
            &[ParamE::Sym("theta")],
            &[("m", 0)],
        );
        assert_gate(
            &prog.statements[5],
            "numnum",
            &[ParamE::Num(0.5)],
            &[("m", 0), ("m", 3)],
        );
        assert_gate(
            &prog.statements[6],
            "givens",
            &[ParamE::Num(-0.7)],
            &[("s", 0), ("s", 1)],
        );
        assert_gate(
            &prog.statements[7],
            "tunnel",
            &[ParamE::Sym("phi")],
            &[("s", 2), ("s", 3)],
        );
        assert_measure(&prog.statements[8], "m", 0, "c", 0);
    }

    fn assert_spinless(prog: &FermionicQasmProgram) {
        assert_eq!(prog.version, "1.0");
        assert_eq!(prog.statements.len(), 8);
        assert_mode(&prog.statements[0], "m", 4, false);
        assert_creg(&prog.statements[1], "c", 4);
        assert_load(&prog.statements[2], &[("m", 0), ("m", 1)]);
        assert_gate(&prog.statements[3], "num", &[ParamE::Num(0.7)], &[("m", 0)]);
        assert_gate(
            &prog.statements[4],
            "numnum",
            &[ParamE::Num(0.7)],
            &[("m", 0), ("m", 1)],
        );
        assert_gate(
            &prog.statements[5],
            "givens",
            &[ParamE::Num(0.7)],
            &[("m", 0), ("m", 1)],
        );
        assert_gate(
            &prog.statements[6],
            "tunnel",
            &[ParamE::Num(0.7)],
            &[("m", 1), ("m", 2)],
        );
        assert_measure(&prog.statements[7], "m", 3, "c", 3);
    }

    fn assert_mode(stmt: &FermionicQasmStmt, name: &str, size: u32, spin: bool) {
        match stmt {
            FermionicQasmStmt::ModeDecl {
                name: n,
                size: s,
                spin: sp,
            } => {
                assert_eq!(n, name);
                assert_eq!(*s, size);
                assert_eq!(*sp, spin);
            }
            other => panic!("expected ModeDecl {name}[{size}], got {other:?}"),
        }
    }

    fn assert_creg(stmt: &FermionicQasmStmt, name: &str, size: u32) {
        match stmt {
            FermionicQasmStmt::CregDecl { name: n, size: s } => {
                assert_eq!(n, name);
                assert_eq!(*s, size);
            }
            FermionicQasmStmt::GateApp(app) => {
                panic!("creg {name}[{size}]; parsed as a gate named `{}`", app.name)
            }
            other => panic!("expected CregDecl, got {other:?}"),
        }
    }

    fn assert_load(stmt: &FermionicQasmStmt, modes: &[(&str, u32)]) {
        match stmt {
            FermionicQasmStmt::Load { modes: got } => assert_refs(got, modes),
            FermionicQasmStmt::GateApp(app) => {
                panic!("load parsed as a gate named `{}`", app.name)
            }
            other => panic!("expected Load, got {other:?}"),
        }
    }

    fn assert_measure(stmt: &FermionicQasmStmt, reg: &str, index: u32, creg: &str, cbit: u32) {
        match stmt {
            FermionicQasmStmt::Measure { mode, cbit: bit } => {
                assert_eq!(mode.reg, reg);
                assert_eq!(mode.index, index);
                assert_eq!(bit.reg, creg);
                assert_eq!(bit.index, cbit);
            }
            other => panic!("expected Measure, got {other:?}"),
        }
    }

    enum ParamE {
        Sym(&'static str),
        Num(f64),
    }

    fn assert_gate(stmt: &FermionicQasmStmt, name: &str, params: &[ParamE], modes: &[(&str, u32)]) {
        match stmt {
            FermionicQasmStmt::GateApp(app) => {
                assert_eq!(app.name, name);
                assert_eq!(app.params.len(), params.len());
                for (got, exp) in app.params.iter().zip(params) {
                    match (got, exp) {
                        (FermionicParam::Symbol(s), ParamE::Sym(e)) => assert_eq!(s, e),
                        (FermionicParam::Num(n), ParamE::Num(e)) => assert_eq!(*n, *e),
                        _ => panic!("param mismatch in {name}: {got:?}"),
                    }
                }
                assert_refs(&app.modes, modes);
            }
            other => panic!("expected gate {name}, got {other:?}"),
        }
    }

    fn assert_refs(got: &[ModeRef], exp: &[(&str, u32)]) {
        assert_eq!(got.len(), exp.len());
        for (g, (reg, index)) in got.iter().zip(exp) {
            assert_eq!(g.reg, *reg);
            assert_eq!(g.index, *index);
        }
    }

    /// (ii) Ordered choice. `creg c[4];` must be a declaration. If
    /// `creg_decl` is moved below `gate_app`, this still parses — as a gate
    /// named `creg` — and the assertion fails. That is the silent misparse.
    #[test]
    fn creg_decl_is_not_a_gate_named_creg() {
        let prog = must_parse("FERMIONICQASM 1.0;\ncreg c[4];\n");
        assert_eq!(prog.statements.len(), 1);
        match &prog.statements[0] {
            FermionicQasmStmt::CregDecl { name, size } => {
                assert_eq!(name, "c");
                assert_eq!(*size, 4);
            }
            FermionicQasmStmt::GateApp(app) => {
                panic!(
                    "creg c[4]; parsed as a gate application named `{}` — \
                     creg_decl is below gate_app in the ordered choice",
                    app.name
                );
            }
            other => panic!("expected CregDecl, got {other:?}"),
        }
    }

    /// (iii) Version lock. The grammar accepts `1.1` and `2.0`; the check
    /// must refuse them and name both the offered version and `1.0`.
    /// Without the check, `expect_err` fails because the parse succeeds.
    #[test]
    fn version_lock_refuses_1_1_and_2_0() {
        for offered in ["1.1", "2.0"] {
            let src = format!("FERMIONICQASM {offered};\nmode m[1];\n");
            let err =
                parse_fermionicqasm(&src).expect_err("a version other than 1.0 must be refused");
            assert!(
                err.contains(offered),
                "message must name the offered version {offered}: {err}"
            );
            assert!(
                err.contains(ACCEPTED_VERSION),
                "message must name the accepted version {ACCEPTED_VERSION}: {err}"
            );
        }
        let ok = must_parse("FERMIONICQASM 1.0;\nmode m[1];\n");
        assert_eq!(ok.version, "1.0");
    }

    /// (iv) `measure m[0] -> c[0];` parses. Absent `measure_stmt` this is a
    /// pest error (the arrow is not a gate application), not a silent miss.
    #[test]
    fn measure_arrow_parses() {
        let prog = must_parse("FERMIONICQASM 1.0;\nmeasure m[0] -> c[0];\n");
        assert_eq!(prog.statements.len(), 1);
        match &prog.statements[0] {
            FermionicQasmStmt::Measure { mode, cbit } => {
                assert_eq!(mode.reg, "m");
                assert_eq!(mode.index, 0);
                assert_eq!(cbit.reg, "c");
                assert_eq!(cbit.index, 0);
            }
            other => panic!("expected Measure, got {other:?}"),
        }
    }
}
