use pest::Parser;
use pest_derive::Parser;

use crate::ast::*;

#[derive(Parser)]
#[grammar = "qasm2.pest"]
struct Qasm2Parser;

/// Parse a QASM 2.0 source string into an AST.
pub fn parse_qasm2(input: &str) -> Result<Qasm2Program, String> {
    let pairs = Qasm2Parser::parse(Rule::program, input).map_err(|e| format!("{}", e))?;

    let mut version = String::new();
    let mut statements = Vec::new();

    for pair in pairs {
        if pair.as_rule() == Rule::program {
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
                        if let Some(stmt) = parse_statement(inner)? {
                            statements.push(stmt);
                        }
                    }
                    Rule::EOI => {}
                    _ => {}
                }
            }
        }
    }

    Ok(Qasm2Program {
        version,
        statements,
    })
}

fn parse_statement(pair: pest::iterators::Pair<Rule>) -> Result<Option<Qasm2Stmt>, String> {
    let inner = pair.into_inner().next().ok_or("empty statement")?;

    match inner.as_rule() {
        Rule::include_stmt => {
            let s = inner.into_inner().next().unwrap().as_str();
            // Strip quotes
            let s = &s[1..s.len() - 1];
            Ok(Some(Qasm2Stmt::Include(s.to_string())))
        }
        Rule::qreg_decl => {
            let mut it = inner.into_inner();
            let name = it.next().unwrap().as_str().to_string();
            let size: u32 = it.next().unwrap().as_str().parse().unwrap();
            Ok(Some(Qasm2Stmt::QregDecl { name, size }))
        }
        Rule::creg_decl => {
            let mut it = inner.into_inner();
            let name = it.next().unwrap().as_str().to_string();
            let size: u32 = it.next().unwrap().as_str().parse().unwrap();
            Ok(Some(Qasm2Stmt::CregDecl { name, size }))
        }
        Rule::qubit_decl_v3 => {
            // QASM 3: `qubit[N] name;`. Lower to QregDecl — same
            // semantics as QASM 2's `qreg name[N];`, arg order reversed.
            let mut it = inner.into_inner();
            let size: u32 = it.next().unwrap().as_str().parse().unwrap();
            let name = it.next().unwrap().as_str().to_string();
            Ok(Some(Qasm2Stmt::QregDecl { name, size }))
        }
        Rule::bit_decl_v3 => {
            // QASM 3: `bit[N] name;`. Lower to CregDecl — same
            // semantics as QASM 2's `creg name[N];`, arg order reversed.
            let mut it = inner.into_inner();
            let size: u32 = it.next().unwrap().as_str().parse().unwrap();
            let name = it.next().unwrap().as_str().to_string();
            Ok(Some(Qasm2Stmt::CregDecl { name, size }))
        }
        Rule::gate_def => {
            let mut it = inner.into_inner();
            let name = it.next().unwrap().as_str().to_string();
            let mut params = Vec::new();
            let mut qubits = Vec::new();
            let mut body = Vec::new();

            for part in it {
                match part.as_rule() {
                    Rule::param_list_def => {
                        for p in part.into_inner() {
                            if p.as_rule() == Rule::ident {
                                params.push(p.as_str().to_string());
                            }
                        }
                    }
                    Rule::qubit_list_def => {
                        for q in part.into_inner() {
                            if q.as_rule() == Rule::ident {
                                qubits.push(q.as_str().to_string());
                            }
                        }
                    }
                    Rule::gate_body => {
                        for stmt in part.into_inner() {
                            if stmt.as_rule() == Rule::gate_app_stmt {
                                body.push(parse_gate_app(stmt)?);
                            }
                        }
                    }
                    _ => {}
                }
            }

            Ok(Some(Qasm2Stmt::GateDef(GateDef {
                name,
                params,
                qubits,
                body,
            })))
        }
        Rule::gate_app_stmt => {
            let app = parse_gate_app(inner)?;
            Ok(Some(Qasm2Stmt::GateApp(app)))
        }
        Rule::measure_stmt => {
            let mut it = inner.into_inner();
            let qubit = parse_qubit_ref(it.next().unwrap())?;
            let cbit = parse_cbit_ref(it.next().unwrap())?;
            Ok(Some(Qasm2Stmt::Measure { qubit, cbit }))
        }
        Rule::measure_assign_v3 => {
            // `c[i] = measure q[i];` — cbit first, then qubit (reversed vs the
            // QASM 2 arrow form), same `Measure` AST node.
            let mut it = inner.into_inner();
            let cbit = parse_cbit_ref(it.next().unwrap())?;
            let qubit = parse_qubit_ref(it.next().unwrap())?;
            Ok(Some(Qasm2Stmt::Measure { qubit, cbit }))
        }
        Rule::barrier_stmt => {
            let mut qubits = Vec::new();
            for part in inner.into_inner() {
                // qubit_list_app contains qubit_refs
                if part.as_rule() == Rule::qubit_list_app {
                    for qr in part.into_inner() {
                        if qr.as_rule() == Rule::qubit_ref {
                            qubits.push(parse_qubit_ref(qr)?);
                        }
                    }
                }
            }
            Ok(Some(Qasm2Stmt::Barrier(qubits)))
        }
        Rule::if_stmt => {
            let mut it = inner.into_inner();
            let creg = it.next().unwrap().as_str().to_string();
            let value: u64 = it.next().unwrap().as_str().parse().unwrap();
            // The guarded statement may be a gate application, a `measure` or
            // a `reset` — the grammar used to admit only the first, so
            // `if (c==1) measure q[0] -> c[0];` never parsed.
            let then_stmt = it.next().unwrap();
            let then = match then_stmt.as_rule() {
                Rule::gate_app_stmt => Qasm2Stmt::GateApp(parse_gate_app(then_stmt)?),
                Rule::measure_stmt => {
                    let mut m = then_stmt.into_inner();
                    let qubit = parse_qubit_ref(m.next().unwrap())?;
                    let cbit = parse_cbit_ref(m.next().unwrap())?;
                    Qasm2Stmt::Measure { qubit, cbit }
                }
                Rule::reset_stmt => {
                    let qr = then_stmt.into_inner().next().unwrap();
                    Qasm2Stmt::Reset(parse_qubit_ref(qr)?)
                }
                other => {
                    return Err(format!("unexpected statement after `if (...)`: {other:?}"))
                }
            };
            Ok(Some(Qasm2Stmt::If {
                creg,
                value,
                then: Box::new(then),
            }))
        }
        Rule::reset_stmt => {
            let qr = inner.into_inner().next().unwrap();
            let qubit = parse_qubit_ref(qr)?;
            Ok(Some(Qasm2Stmt::Reset(qubit)))
        }
        _ => Ok(None),
    }
}

fn parse_gate_app(pair: pest::iterators::Pair<Rule>) -> Result<GateApp, String> {
    let mut name = String::new();
    let mut params = Vec::new();
    let mut qubits = Vec::new();
    let mut modifiers = Vec::new();

    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::gate_modifier => {
                let inner_mod = part.into_inner().next().ok_or("empty gate_modifier")?;
                match inner_mod.as_rule() {
                    Rule::inv_modifier => modifiers.push(GateModifier::Inv),
                    Rule::pow_modifier => {
                        let n_pair = inner_mod
                            .into_inner()
                            .find(|p| p.as_rule() == Rule::integer)
                            .ok_or("pow modifier missing integer")?;
                        let n: i32 = n_pair
                            .as_str()
                            .parse()
                            .map_err(|e| format!("pow exponent: {e}"))?;
                        modifiers.push(GateModifier::Pow(n));
                    }
                    other => return Err(format!("unexpected modifier rule: {:?}", other)),
                }
            }
            Rule::gate_name => {
                name = part.as_str().to_string();
            }
            Rule::param_list_app => {
                for p in part.into_inner() {
                    if p.as_rule() == Rule::expr {
                        params.push(parse_expr(p)?);
                    }
                }
            }
            Rule::qubit_list_app => {
                for qr in part.into_inner() {
                    if qr.as_rule() == Rule::qubit_ref {
                        qubits.push(parse_qubit_ref(qr)?);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(GateApp {
        name,
        params,
        qubits,
        modifiers,
    })
}

fn parse_qubit_ref(pair: pest::iterators::Pair<Rule>) -> Result<QubitRef, String> {
    let mut it = pair.into_inner();
    let first = it.next().unwrap();
    let reg = first.as_str().to_string();
    match it.next() {
        Some(idx) => {
            let index: u32 = idx.as_str().parse().unwrap();
            Ok(QubitRef::Indexed { reg, index })
        }
        None => Ok(QubitRef::Register(reg)),
    }
}

fn parse_cbit_ref(pair: pest::iterators::Pair<Rule>) -> Result<CbitRef, String> {
    let mut it = pair.into_inner();
    let first = it.next().unwrap();
    let reg = first.as_str().to_string();
    match it.next() {
        Some(idx) => {
            let index: u32 = idx.as_str().parse().unwrap();
            Ok(CbitRef::Indexed { reg, index })
        }
        None => Ok(CbitRef::Register(reg)),
    }
}

fn parse_expr(pair: pest::iterators::Pair<Rule>) -> Result<Expr, String> {
    let mut terms: Vec<(Option<bool>, Expr)> = Vec::new(); // (negated?, expr)
    let mut ops: Vec<BinOp> = Vec::new();

    let mut negated = false;
    for part in pair.into_inner() {
        match part.as_rule() {
            Rule::prefix_op => {
                negated = true;
            }
            Rule::atom => {
                let mut atom = parse_atom(part)?;
                if negated {
                    atom = Expr::Neg(Box::new(atom));
                    negated = false;
                }
                terms.push((None, atom));
            }
            Rule::bin_op => {
                let op = match part.as_str() {
                    "+" => BinOp::Add,
                    "-" => BinOp::Sub,
                    "*" => BinOp::Mul,
                    "/" => BinOp::Div,
                    s => return Err(format!("unknown operator: {}", s)),
                };
                ops.push(op);
            }
            _ => {}
        }
    }

    if terms.is_empty() {
        return Err("empty expression".to_string());
    }

    // Build left-associative expression tree (no precedence climbing for simplicity)
    let mut result = terms.remove(0).1;
    for op in ops.into_iter() {
        let rhs = terms.remove(0).1;
        result = Expr::BinOp(Box::new(result), op, Box::new(rhs));
    }

    Ok(result)
}

fn parse_atom(pair: pest::iterators::Pair<Rule>) -> Result<Expr, String> {
    // Check if the atom text is literally "pi" (matched by the "pi" literal in the grammar)
    let atom_str = pair.as_str().trim();
    let inner = pair.into_inner().next();

    match inner {
        None => {
            // No sub-rule means it matched a literal like "pi"
            if atom_str == "pi" {
                Ok(Expr::Pi)
            } else {
                Err(format!("unexpected atom literal: {}", atom_str))
            }
        }
        Some(inner) => match inner.as_rule() {
            Rule::number => {
                let val: f64 = inner
                    .as_str()
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?;
                Ok(Expr::Num(val))
            }
            Rule::ident => {
                let name = inner.as_str();
                if name == "pi" {
                    Ok(Expr::Pi)
                } else {
                    Ok(Expr::Ident(name.to_string()))
                }
            }
            Rule::fn_call => {
                let mut it = inner.into_inner();
                let fn_name = it.next().unwrap().as_str().to_string();
                let arg = parse_expr(it.next().unwrap())?;
                Ok(Expr::FnCall(fn_name, Box::new(arg)))
            }
            Rule::expr => parse_expr(inner),
            _ => Err(format!("unexpected atom: {:?}", inner.as_rule())),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bell_state() {
        let src = r#"OPENQASM 2.0;
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
        let prog = parse_qasm2(src).unwrap();
        assert_eq!(prog.version, "2.0");
        assert_eq!(prog.statements.len(), 6);
    }

    #[test]
    fn test_parse_rotation() {
        let src = r#"OPENQASM 2.0;
qreg q[1];
rx(pi/2) q[0];
rz(3.14) q[0];
"#;
        let prog = parse_qasm2(src).unwrap();
        assert_eq!(prog.statements.len(), 3);
    }

    #[test]
    fn test_parse_gate_def() {
        let src = r#"OPENQASM 2.0;
qreg q[2];
gate mygate(a) p, q {
    rz(a) p;
    cx p, q;
}
mygate(pi/4) q[0], q[1];
"#;
        let prog = parse_qasm2(src).unwrap();
        // qreg decl, gate def, gate app
        assert_eq!(prog.statements.len(), 3);
    }
}
