use pest::Parser;
use pest_derive::Parser;

use crate::ast::*;

#[derive(Parser)]
#[grammar = "opticqasm.pest"]
struct OpticQasmParser;

/// Parse an OPTICQASM 1.0 source string into an AST.
pub fn parse_opticqasm(input: &str) -> Result<OpticQasmProgram, String> {
    let pairs = OpticQasmParser::parse(Rule::program, input).map_err(|e| format!("{}", e))?;

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
                        Rule::photon_decl => {
                            let mut it = stmt_inner.into_inner();
                            let name = it.next().unwrap().as_str().to_string();
                            let size: u32 = it.next().unwrap().as_str().parse().unwrap();
                            // The `pol` marker is an optional trailing rule, so
                            // its presence is simply whether anything is left.
                            let polarized = it.next().is_some();
                            statements.push(OpticQasmStmt::PhotonDecl {
                                name,
                                size,
                                polarized,
                            });
                        }
                        Rule::gate_app => {
                            let app = parse_gate_app(stmt_inner)?;
                            statements.push(OpticQasmStmt::GateApp(app));
                        }
                        _ => {}
                    }
                }
                Rule::EOI => {}
                _ => {}
            }
        }
    }

    Ok(OpticQasmProgram {
        version,
        statements,
    })
}

fn parse_gate_app(pair: pest::iterators::Pair<Rule>) -> Result<OpticGateApp, String> {
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
                                // Strip the '$' prefix
                                let sym = inner.as_str();
                                let sym = sym.strip_prefix('$').unwrap_or(sym);
                                params.push(OpticParam::Symbol(sym.to_string()));
                            }
                            Rule::number => {
                                let val: f64 = inner.as_str().parse().unwrap();
                                params.push(OpticParam::Num(val));
                            }
                            _ => {}
                        }
                    }
                }
            }
            Rule::mode_list => {
                for m in part.into_inner() {
                    if m.as_rule() == Rule::mode_ref {
                        let mut it = m.into_inner();
                        let reg = it.next().unwrap().as_str().to_string();
                        let index: u32 = it.next().unwrap().as_str().parse().unwrap();
                        modes.push(ModeRef { reg, index });
                    }
                }
            }
            _ => {}
        }
    }

    Ok(OpticGateApp {
        name,
        params,
        modes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_opticqasm() {
        let src = r#"OPTICQASM 1.0;
// My comment
photon q[6];
ps($phi0) q[0];
ps($phi1) q[1];
bs_rx($theta0, $phi_tr0) q[0], q[1];
bs_rx($theta1, $phi_tr1) q[2], q[3];
"#;
        let prog = parse_opticqasm(src).unwrap();
        assert_eq!(prog.version, "1.0");
        // 1 photon decl + 2 ps + 2 bs_rx = 5
        assert_eq!(prog.statements.len(), 5);

        // Check first gate
        match &prog.statements[1] {
            OpticQasmStmt::GateApp(app) => {
                assert_eq!(app.name, "ps");
                assert_eq!(app.params.len(), 1);
                assert_eq!(app.modes.len(), 1);
                assert_eq!(app.modes[0].index, 0);
                match &app.params[0] {
                    OpticParam::Symbol(s) => assert_eq!(s, "phi0"),
                    _ => panic!("expected symbol"),
                }
            }
            _ => panic!("expected gate app"),
        }
    }

    #[test]
    fn test_parse_full_mesh() {
        // From the specs
        let src = r#"OPTICQASM 1.0;
photon q[6];
ps($phi0) q[0];
ps($phi1) q[1];
ps($phi2) q[2];
ps($phi3) q[3];
ps($phi4) q[4];
ps($phi5) q[5];
bs_rx($theta0, $phi_tr0) q[0], q[1];
bs_rx($theta1, $phi_tr1) q[2], q[3];
bs_rx($theta2, $phi_tr2) q[4], q[5];
bs_rx($theta3, $phi_tr3) q[1], q[2];
bs_rx($theta4, $phi_tr4) q[3], q[4];
bs_rx($theta5, $phi_tr5) q[0], q[1];
bs_rx($theta6, $phi_tr6) q[2], q[3];
bs_rx($theta7, $phi_tr7) q[4], q[5];
bs_rx($theta8, $phi_tr8) q[1], q[2];
bs_rx($theta9, $phi_tr9) q[3], q[4];
bs_rx($theta10, $phi_tr10) q[0], q[1];
bs_rx($theta11, $phi_tr11) q[2], q[3];
bs_rx($theta12, $phi_tr12) q[4], q[5];
bs_rx($theta13, $phi_tr13) q[1], q[2];
bs_rx($theta14, $phi_tr14) q[3], q[4];
"#;
        let prog = parse_opticqasm(src).unwrap();
        // 1 photon + 6 ps + 15 bs_rx = 22
        assert_eq!(prog.statements.len(), 22);
    }
}
