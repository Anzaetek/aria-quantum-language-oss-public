use pest::Parser;
use pest_derive::Parser;

use crate::ast::*;

#[derive(Parser)]
#[grammar = "qasm2.pest"]
struct Qasm2Parser;

/// Parse a QASM 2.0 source string into an AST.
pub fn parse_qasm2(input: &str) -> Result<Qasm2Program, String> {
    let pairs =
        Qasm2Parser::parse(Rule::program, input).map_err(|e| enrich_parse_error(input, &e))?;

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

/// Name the OpenQASM 3 construct a parse failure actually tripped over.
///
/// A pest error is a position and a list of rules it wanted — `--> 9:7 …
/// expected param_list_app`. That is accurate and useless: it describes our
/// grammar, not the user's file. Measured across the specification's own
/// examples, every out-of-profile program failed with some variant of it, and
/// none said *which language feature* was the problem.
///
/// This is **purely diagnostic**. It never changes what is accepted — it runs
/// only on an error that has already happened, and it appends to the pest
/// output rather than replacing it, so the position is still there for anyone
/// who wants it.
///
/// The keyword is read from the line pest stopped on. It is not a
/// "does-this-look-like-QASM" sniff (see `lower_to_ir_with_dialect`, which
/// deliberately avoids one): nothing branches on the result, so a wrong guess
/// costs a less helpful message and nothing else.
fn enrich_parse_error(input: &str, e: &pest::error::Error<Rule>) -> String {
    use pest::error::LineColLocation;
    let line_no = match e.line_col {
        LineColLocation::Pos((l, _)) => l,
        LineColLocation::Span((l, _), _) => l,
    };
    let line = input.lines().nth(line_no.saturating_sub(1)).unwrap_or("");
    let first_word = line
        .trim_start()
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .find(|s| !s.is_empty())
        .unwrap_or("");

    if let Some(why) = unsupported_construct(first_word) {
        return format!("{why}\n\n{e}");
    }
    // Not a keyword we know. One more cause worth naming, because no keyword
    // lookup can find it: OpenQASM 3 permits Unicode identifiers and our
    // `ident` rule is ASCII-only. `cphase.qasm` declares `gate cphase(θ) a, b`
    // and failed with a caret under a character the reader simply cannot spell.
    if !line.is_ascii() {
        let offenders: String = line
            .chars()
            .filter(|c| !c.is_ascii())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        return format!(
            "this line contains non-ASCII character(s) `{offenders}`. OpenQASM 3 allows \
             Unicode identifiers; this reader's `ident` rule is ASCII-only, so a name \
             like `θ` cannot be read. Rename it (for example `theta`) before importing.\
             \n\n{e}"
        );
    }
    format!("{e}")
}

/// OpenQASM 3 constructs outside this reader's gate-model profile, each with a
/// message saying what it is and what to do instead.
///
/// Deliberately explicit rather than a catch-all: an unlisted keyword falls
/// back to the raw pest error, which is honest, whereas a generic "unsupported
/// feature" would claim knowledge we do not have.
fn unsupported_construct(keyword: &str) -> Option<String> {
    /// Stated once, so every message describes the same profile.
    const PROFILE: &str = "This reader implements the gate-model profile: register \
                           declarations, gate applications and `gate` definitions, \
                           `barrier`, `reset`, `measure`, and guards of the form \
                           `if (c == N)` or `if (c[i] == true/false)`.";

    let (what, advice) = match keyword {
        "def" => (
            "`def` (subroutine definition)",
            " Inline the subroutine before importing.",
        ),
        // `defcalgrammar` is a separate keyword and is how `defcal.qasm`
        // actually opens — matching only `defcal` missed the very file the
        // construct is named after.
        "defcal" | "cal" | "defcalgrammar" => (
            "pulse-level calibration (`defcal` / `cal` / `defcalgrammar`)",
            " There is no pulse model here, and no approximation that would not \
             silently change the circuit.",
        ),
        "extern" => ("`extern` (external function declaration)", ""),
        // Reached only when the `if_stmt` rule itself failed, so the condition
        // is neither supported form — typically a cast, as in
        // `if(int[4](c) == 1)` (`inverseqft1.qasm`), or an arithmetic
        // expression. A *mixed* comparison (`c[0] == 1`) does parse and is
        // refused later with a message naming which side is wrong, so it never
        // reaches here.
        "if" => (
            "an `if` guard that is neither `c == N` nor `c[i] == true/false` — a cast \
             or expression in the condition",
            " Compare a whole register to an integer, or a single bit to a boolean.",
        ),
        "for" | "while" => ("a loop (`for` / `while`)", " Unroll it before importing."),
        "const" => ("`const` (compile-time constant)", ""),
        "int" | "uint" | "float" | "angle" | "bool" | "complex" => {
            ("a classical variable declaration", "")
        }
        "duration" | "stretch" | "delay" | "durationof" => (
            "timing control (`duration` / `stretch` / `delay`)",
            " Gates here are ordered but not scheduled, so a duration cannot be \
             honoured and ignoring it would answer a different question.",
        ),
        "array" => ("`array` (classical array declaration)", ""),
        "let" => ("`let` (register alias)", ""),
        "input" | "output" => ("`input` / `output` (circuit parameters)", ""),
        "gphase" => ("`gphase` (global phase)", ""),
        "pragma" | "annotation" => ("a pragma or annotation", ""),
        _ => return None,
    };
    Some(format!(
        "OpenQASM 3 {what} is not in the supported subset. {PROFILE}{advice}"
    ))
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
            // QASM 3: `qubit[N] name;` or, for a single qubit, `qubit name;`.
            // Lowers to QregDecl — same semantics as QASM 2's `qreg name[N];`,
            // arg order reversed.
            let (name, size) = decl_v3_parts(inner)?;
            Ok(Some(Qasm2Stmt::QregDecl { name, size }))
        }
        Rule::bit_decl_v3 => {
            // QASM 3: `bit[N] name;` or `bit name;`.
            let (name, size) = decl_v3_parts(inner)?;
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
            // `cond_target` is either `c` (whole register) or `c[i]` (one bit).
            let target = it.next().ok_or("`if` without a condition target")?;
            let mut tparts = target.into_inner();
            let creg = tparts
                .next()
                .ok_or("`if` condition names no register")?
                .as_str()
                .to_string();
            let bit: Option<u32> = match tparts.next() {
                Some(idx) => Some(idx.as_str().parse().map_err(|_| {
                    format!(
                        "bit index `{}` in `if ({creg}[…])` is not a u32",
                        idx.as_str()
                    )
                })?),
                None => None,
            };
            // `cond_value` is an integer (`c == 1`) or a boolean (`c[0] == true`).
            let vpair = it.next().ok_or("`if` without a comparison value")?;
            let vtext = vpair.as_str();
            let value: u64 = match vtext {
                "true" => 1,
                "false" => 0,
                other => other.parse().map_err(|_| {
                    format!("`if ({creg} == {other})`: not an integer or boolean literal")
                })?,
            };
            // OpenQASM 3 pairs a bit with a bool and a bitarray with an int.
            // Mixing them is what made our own emitter's output unloadable, so
            // refuse rather than quietly accept a spelling qiskit rejects.
            let is_bool = matches!(vtext, "true" | "false");
            if let (Some(i), false) = (bit, is_bool) {
                return Err(format!(
                    "`if ({creg}[{i}] == {vtext})` compares a single BIT to an integer. \
                     OpenQASM 3 requires `bit == const bool` — write `{}` — or compare \
                     the whole register (`{creg} == {vtext}`). Strict consumers reject \
                     the mixed form.",
                    if value == 0 { "false" } else { "true" }
                ));
            }
            if bit.is_none() && is_bool {
                return Err(format!(
                    "`if ({creg} == {vtext})` compares a whole REGISTER to a boolean. \
                     OpenQASM 3 requires `bitarray == const int` — write \
                     `{creg} == {value}` — or address a single bit (`{creg}[0] == {vtext}`)."
                ));
            }
            // The guarded statement may be a gate application, a `measure` or
            // a `reset` — the grammar used to admit only the first, so
            // `if (c==1) measure q[0] -> c[0];` never parsed.
            let then_stmt = it.next().unwrap();
            // A braced block holds zero or more statements; the bare form is
            // the same thing with exactly one. Flattening both into a `Vec`
            // here means the lowering has a single path and the two spellings
            // cannot drift apart.
            let then = match then_stmt.as_rule() {
                Rule::if_block => then_stmt
                    .into_inner()
                    .map(parse_guarded_stmt)
                    .collect::<Result<Vec<_>, _>>()?,
                _ => vec![parse_guarded_stmt(then_stmt)?],
            };
            Ok(Some(Qasm2Stmt::If {
                creg,
                bit,
                value,
                then,
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

/// One statement inside an `if` — bare or inside a braced block.
///
/// Shared by both spellings deliberately: when the grammar was widened to admit
/// guarded `measure`/`reset`, the lowering's `if let` matched nothing and
/// emitted NOTHING, turning a parse error into a silent drop. Keeping one
/// function means adding a guardable form cannot repeat that.
fn parse_guarded_stmt(pair: pest::iterators::Pair<Rule>) -> Result<Qasm2Stmt, String> {
    match pair.as_rule() {
        Rule::gate_app_stmt => Ok(Qasm2Stmt::GateApp(parse_gate_app(pair)?)),
        Rule::measure_stmt => {
            let mut m = pair.into_inner();
            let qubit = parse_qubit_ref(m.next().ok_or("`measure` without a qubit")?)?;
            let cbit = parse_cbit_ref(m.next().ok_or("`measure` without a target bit")?)?;
            Ok(Qasm2Stmt::Measure { qubit, cbit })
        }
        Rule::reset_stmt => {
            let qr = pair.into_inner().next().ok_or("`reset` without a qubit")?;
            Ok(Qasm2Stmt::Reset(parse_qubit_ref(qr)?))
        }
        other => Err(format!("unexpected statement after `if (...)`: {other:?}")),
    }
}

/// Split a QASM 3 register declaration into `(name, size)`.
///
/// The size is optional — `qubit q;` is a single qubit — so the inner pairs are
/// either `[integer, ident]` or just `[ident]`. Reading them positionally with
/// `it.next()` would take the NAME as the size the moment the brackets are
/// absent, so the shape is inspected rather than assumed.
///
/// The width is parsed fallibly. It was `.parse().unwrap()`, which panics on an
/// integer too large for `u32` — and a parser that crashes on malformed input
/// cannot be pointed at anything untrusted. Same contract as
/// `check_gate_arity`: refuse with a message, never abort.
fn decl_v3_parts(pair: pest::iterators::Pair<Rule>) -> Result<(String, u32), String> {
    let parts: Vec<_> = pair.into_inner().collect();
    match parts.as_slice() {
        [name] => Ok((name.as_str().to_string(), 1)),
        [size, name] => {
            let s = size.as_str();
            let size: u32 = s.parse().map_err(|_| {
                format!(
                    "register size `{s}` is not a valid u32 (declaring `{}`)",
                    name.as_str()
                )
            })?;
            Ok((name.as_str().to_string(), size))
        }
        other => Err(format!(
            "malformed register declaration: expected `name` or `[size] name`, \
             got {} component(s)",
            other.len()
        )),
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
