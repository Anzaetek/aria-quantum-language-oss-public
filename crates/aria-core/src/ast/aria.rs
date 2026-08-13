//! Aria DSL parser.
//!
//! Parses the surface syntax documented in `examples/aria/README.md`
//! and produces a `Circuit` AST identical to what the QASM / JSON
//! importers emit. The parser is a single-pass recursive descent over
//! the tokens produced by [`Lexer`]. Sub-circuits referenced via
//! `oracle` are inlined at instantiation time.
//!
//! # Surface syntax
//!
//! ```text
//! @assert unitary
//! @prove "bell_correct" equiv { creates (|00> + |11>)/sqrt(2) }
//! circuit Bell {
//!     qreg q[2]
//!     creg c[2]
//!     apply H on q[0]
//!     apply CX on q[0], q[1]
//!     measure q -> c
//! }
//! ```
//!
//! Control flow: `repeat N { .. }`, `repeat i from a to b { .. }`,
//! `repeat i from a to b step s { .. }` (step may be negative),
//! `when expr { .. }` with compile-time or runtime conditions,
//! `oracle NAME(args) on q` (inlines a previously-parsed sub-circuit).
//!
//! Expressions: +, -, *, /, ^ (power), function calls (`sin`, `cos`,
//! `sqrt`, `bit`), numeric literals, identifiers, `pi`.
//!
//! # Entry points
//!
//! - [`parse_aria`] — parses a full source string into an
//!   [`AriaProgram`] containing every `circuit` template.
//! - [`AriaProgram::instantiate`] — materializes a named template
//!   with concrete integer parameter bindings. The returned `Circuit`
//!   is fully flattened (loops unrolled, oracles inlined).
//!
//! # Limitations
//!
//! - Conditional gates (`when m[0] == 1 { … }`) attach a per-instruction
//!   classical condition (`Instruction::condition`). Only the single-bit
//!   form `reg[i] == literal` is representable; any other runtime guard,
//!   and any nesting of runtime `when`, is a lowering error — the circuit
//!   model holds one condition per instruction, and executing a body whose
//!   guard was dropped would be silently wrong.
//! - `observable` blocks are parsed to a template but not lowered —
//!   observables live in `backends::omega::OmegaObservable`, which
//!   is a cross-boundary concern left to the caller.

use super::annotation::{Annotation, Property};
use super::expr::ParamExpr;
use super::nodes::*;
use std::collections::HashMap;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    StringLit(String),
    // Single-char / short operators
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Comma,
    Colon,
    Semicolon,
    At,
    Arrow, // ->
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Percent,
    EqEq,  // ==
    NotEq, // !=
    Le,    // <=
    Ge,    // >=
    Lt,
    Gt,
    Eq, // =
    // Keywords
    Circuit,
    Observable,
    Qreg,
    Creg,
    Let,
    Var,
    Apply,
    On,
    Repeat,
    From,
    To,
    Step,
    Measure,
    Oracle,
    When,
    Pi,
    Symbolic,
    True,
    False,
}

fn keyword(ident: &str) -> Option<Tok> {
    Some(match ident {
        "circuit" => Tok::Circuit,
        "observable" => Tok::Observable,
        "qreg" => Tok::Qreg,
        "creg" => Tok::Creg,
        "let" => Tok::Let,
        "var" => Tok::Var,
        "apply" => Tok::Apply,
        "on" => Tok::On,
        "repeat" => Tok::Repeat,
        "from" => Tok::From,
        "to" => Tok::To,
        "step" => Tok::Step,
        "measure" => Tok::Measure,
        "oracle" => Tok::Oracle,
        "when" => Tok::When,
        "pi" => Tok::Pi,
        "symbolic" => Tok::Symbolic,
        "true" => Tok::True,
        "false" => Tok::False,
        _ => return None,
    })
}

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
            line: 1,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        if c == b'\n' {
            self.line += 1;
        }
        Some(c)
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while let Some(c) = self.peek() {
                if c.is_ascii_whitespace() {
                    self.bump();
                } else {
                    break;
                }
            }
            // `--` line comment
            if self.pos + 1 < self.src.len()
                && self.src[self.pos] == b'-'
                && self.src[self.pos + 1] == b'-'
            {
                while let Some(c) = self.peek() {
                    if c == b'\n' {
                        break;
                    }
                    self.bump();
                }
            } else {
                break;
            }
        }
    }

    fn tokenize(&mut self) -> Result<Vec<(Tok, usize)>, String> {
        let mut out = Vec::new();
        loop {
            self.skip_ws_and_comments();
            let Some(c) = self.peek() else { break };
            let line = self.line;
            let tok = match c {
                b'{' => {
                    self.bump();
                    Tok::LBrace
                }
                b'}' => {
                    self.bump();
                    Tok::RBrace
                }
                b'[' => {
                    self.bump();
                    Tok::LBracket
                }
                b']' => {
                    self.bump();
                    Tok::RBracket
                }
                b'(' => {
                    self.bump();
                    Tok::LParen
                }
                b')' => {
                    self.bump();
                    Tok::RParen
                }
                b',' => {
                    self.bump();
                    Tok::Comma
                }
                b':' => {
                    self.bump();
                    Tok::Colon
                }
                b';' => {
                    self.bump();
                    Tok::Semicolon
                }
                b'@' => {
                    self.bump();
                    Tok::At
                }
                b'+' => {
                    self.bump();
                    Tok::Plus
                }
                b'*' => {
                    self.bump();
                    Tok::Star
                }
                b'/' => {
                    self.bump();
                    Tok::Slash
                }
                b'^' => {
                    self.bump();
                    Tok::Caret
                }
                b'%' => {
                    self.bump();
                    Tok::Percent
                }
                b'-' => {
                    self.bump();
                    if self.peek() == Some(b'>') {
                        self.bump();
                        Tok::Arrow
                    } else {
                        Tok::Minus
                    }
                }
                b'=' => {
                    self.bump();
                    if self.peek() == Some(b'=') {
                        self.bump();
                        Tok::EqEq
                    } else {
                        Tok::Eq
                    }
                }
                b'!' => {
                    self.bump();
                    if self.peek() == Some(b'=') {
                        self.bump();
                        Tok::NotEq
                    } else {
                        return Err(format!("line {line}: unexpected '!'"));
                    }
                }
                b'<' => {
                    self.bump();
                    if self.peek() == Some(b'=') {
                        self.bump();
                        Tok::Le
                    } else {
                        Tok::Lt
                    }
                }
                b'>' => {
                    self.bump();
                    if self.peek() == Some(b'=') {
                        self.bump();
                        Tok::Ge
                    } else {
                        Tok::Gt
                    }
                }
                b'"' => {
                    self.bump();
                    let start = self.pos;
                    while let Some(c) = self.peek() {
                        if c == b'"' {
                            break;
                        }
                        self.bump();
                    }
                    let s = std::str::from_utf8(&self.src[start..self.pos])
                        .map_err(|e| format!("line {line}: utf8: {e}"))?
                        .to_string();
                    if self.peek() != Some(b'"') {
                        return Err(format!("line {line}: unterminated string"));
                    }
                    self.bump();
                    Tok::StringLit(s)
                }
                c if c.is_ascii_digit() => {
                    let start = self.pos;
                    let mut saw_dot = false;
                    while let Some(c) = self.peek() {
                        if c.is_ascii_digit() {
                            self.bump();
                        } else if c == b'.' && !saw_dot {
                            // Peek next char to avoid consuming method-call `.`
                            // (not used in this DSL).
                            saw_dot = true;
                            self.bump();
                        } else {
                            break;
                        }
                    }
                    // Scientific-notation exponent: `e`/`E`, optional sign,
                    // then at least one digit (`1e-05`, `2.5E+3`). Consumed
                    // only when the full pattern matches, so an identifier
                    // right after a number (`1e`) still lexes as before.
                    let mut saw_exp = false;
                    if matches!(self.peek(), Some(b'e') | Some(b'E')) {
                        let mut j = self.pos + 1;
                        if matches!(self.src.get(j).copied(), Some(b'+') | Some(b'-')) {
                            j += 1;
                        }
                        if self.src.get(j).copied().is_some_and(|c| c.is_ascii_digit()) {
                            saw_exp = true;
                            while self.pos < j {
                                self.bump();
                            }
                            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                                self.bump();
                            }
                        }
                    }
                    let lit = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
                    if saw_dot || saw_exp {
                        let v: f64 = lit
                            .parse()
                            .map_err(|_| format!("line {line}: invalid float literal '{lit}'"))?;
                        // Rust's f64 parse yields Ok(inf) on exponent
                        // overflow (`1e999`); a non-finite angle silently
                        // NaN-poisons the whole state, so reject it here.
                        if !v.is_finite() {
                            return Err(format!(
                                "line {line}: float literal '{lit}' overflows f64"
                            ));
                        }
                        Tok::Float(v)
                    } else {
                        let v: i64 = lit
                            .parse()
                            .map_err(|_| format!("line {line}: invalid int literal '{lit}'"))?;
                        Tok::Int(v)
                    }
                }
                c if c.is_ascii_alphabetic() || c == b'_' => {
                    let start = self.pos;
                    while let Some(c) = self.peek() {
                        if c.is_ascii_alphanumeric() || c == b'_' {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                    let ident = std::str::from_utf8(&self.src[start..self.pos])
                        .unwrap()
                        .to_string();
                    if let Some(kw) = keyword(&ident) {
                        kw
                    } else {
                        Tok::Ident(ident)
                    }
                }
                b'|' => {
                    // `|ket>` sequences may appear inside @prove { … } blocks;
                    // treated as identifiers/markers — skip the character.
                    self.bump();
                    continue;
                }
                other => {
                    return Err(format!(
                        "line {}: unexpected character '{}'",
                        line, other as char
                    ));
                }
            };
            out.push((tok, line));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// AST (pre-instantiation templates)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Expr {
    IntLit(i64),
    FloatLit(f64),
    Pi,
    Ident(String),
    /// `name[idx]` — either register slot or symbolic-array slot.
    Index(String, Box<Expr>),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
    CmpEq(Box<Expr>, Box<Expr>),
    CmpNe(Box<Expr>, Box<Expr>),
    CmpLt(Box<Expr>, Box<Expr>),
    CmpLe(Box<Expr>, Box<Expr>),
    CmpGt(Box<Expr>, Box<Expr>),
    CmpGe(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug)]
enum Target {
    /// Bare register name (e.g. `q` in `measure q -> c`).
    Reg(String),
    /// Indexed register slot.
    Indexed(String, Expr),
}

#[derive(Clone, Debug)]
enum Stmt {
    Qreg {
        name: String,
        size: Expr,
    },
    Creg {
        name: String,
        size: Expr,
    },
    Let {
        name: String,
        value: Expr,
    },
    Symbolic {
        name: String,
        count: Expr,
    },
    Apply {
        gate: String,
        params: Vec<Expr>,
        qubits: Vec<Target>,
    },
    Measure {
        from: Target,
        to: Target,
    },
    MeasureAll {
        q_reg: String,
        c_reg: String,
    },
    Repeat {
        count: Expr,
        body: Vec<Stmt>,
    },
    RepeatRange {
        var: String,
        from: Expr,
        to: Expr,
        step: Expr,
        body: Vec<Stmt>,
    },
    When {
        cond: Expr,
        body: Vec<Stmt>,
    },
    Oracle {
        name: String,
        args: Vec<Expr>,
        qubits: Vec<Target>,
    },
}

#[derive(Clone, Debug)]
struct AnnotationAst {
    kind: String,
    content: String,
}

#[derive(Clone, Debug)]
pub struct CircuitTemplate {
    pub name: String,
    /// (param-name, type-name). Type-name is informational only.
    pub params: Vec<(String, String)>,
    annotations: Vec<AnnotationAst>,
    body: Vec<Stmt>,
}

/// Observable template. The body keeps the original token stream so
/// `to_omega_observable()` can re-parse it into a concrete
/// `OmegaObservable`. A space-joined debug dump is also kept in
/// `body_text` for human inspection.
#[derive(Clone, Debug)]
pub struct ObservableTemplate {
    pub name: String,
    /// Space-joined debug dump of the body tokens — for display only.
    pub body: String,
    body_tokens: Vec<Tok>,
}

impl ObservableTemplate {
    /// Lower this parsed observable into a runtime `OmegaObservable`.
    ///
    /// Supports the subset of Aria observable syntax actually used by
    /// the QML / VQE / QAOA examples:
    ///
    /// ```text
    /// observable H {
    ///     let g0 = -0.4804
    ///     let g1 =  0.3435
    ///     g0 * I
    ///   + g1 * Z(0)
    ///   + 0.5 * X(0) * X(1)
    ///   - 0.25 * Y(0) * Y(1) * Z(2)
    /// }
    /// ```
    ///
    /// Grammar (all tokens already split by `parse_observable`):
    ///
    /// ```text
    /// Body      := (Let | Term)+                         -- top-level is a sum
    /// Let       := 'let' IDENT '=' NumberExpr
    /// Term      := ('+' | '-')? Factor ('*' Factor)*
    /// Factor    := NumberExpr | 'I' | PauliOp
    /// PauliOp   := ('X' | 'Y' | 'Z') '(' INT ')'
    /// NumberExpr:= ['-'] (FLOAT | INT | IDENT)
    /// ```
    ///
    /// Returns the weighted Pauli-string representation that
    /// `crates/aria-core/src/backends/omega.rs::OmegaObservable`
    /// consumes.
    pub fn to_omega_observable(&self) -> Result<crate::backends::omega::OmegaObservable, String> {
        use crate::backends::omega::{OmegaObservable, OmegaPauliOp};

        let mut bindings: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let mut terms: Vec<(f64, Vec<(u32, OmegaPauliOp)>)> = Vec::new();

        let toks = &self.body_tokens;
        let mut i = 0usize;
        // Sign carried into each term (+1 for first, then flipped by
        // leading '+' / '-').
        let mut sign: f64 = 1.0;
        let mut first_term = true;
        while i < toks.len() {
            // `let NAME = VALUE` binding.
            if matches!(toks.get(i), Some(Tok::Let)) {
                i += 1;
                let name = match toks.get(i) {
                    Some(Tok::Ident(n)) => n.clone(),
                    _ => {
                        return Err(format!(
                            "observable '{}': expected identifier after `let`",
                            self.name
                        ));
                    }
                };
                i += 1;
                if !matches!(toks.get(i), Some(Tok::Eq)) {
                    return Err(format!(
                        "observable '{}': expected `=` after `let {name}`",
                        self.name
                    ));
                }
                i += 1;
                let (val, j) = parse_numeric(toks, i, &bindings)
                    .map_err(|e| format!("observable '{}': {e}", self.name))?;
                bindings.insert(name, val);
                i = j;
                continue;
            }

            // Sign separators between terms (not the first term).
            if !first_term {
                match toks.get(i) {
                    Some(Tok::Plus) => {
                        sign = 1.0;
                        i += 1;
                    }
                    Some(Tok::Minus) => {
                        sign = -1.0;
                        i += 1;
                    }
                    _ => {
                        return Err(format!(
                            "observable '{}': expected `+` or `-` between terms \
                             (got {:?} at token {})",
                            self.name,
                            toks.get(i),
                            i
                        ));
                    }
                }
            }

            // Parse one Term: (number | Pauli) ('*' (number | Pauli))*
            let mut coeff: f64 = sign;
            let mut paulis: Vec<(u32, OmegaPauliOp)> = Vec::new();
            loop {
                match toks.get(i) {
                    Some(Tok::Ident(name)) if is_pauli_letter(name) => {
                        // PauliOp
                        let op = pauli_from_letter(name);
                        i += 1;
                        if !matches!(toks.get(i), Some(Tok::LParen)) {
                            return Err(format!(
                                "observable '{}': expected `(` after Pauli `{name}`",
                                self.name
                            ));
                        }
                        i += 1;
                        let q = match toks.get(i) {
                            Some(Tok::Int(v)) if *v >= 0 => *v as u32,
                            _ => {
                                return Err(format!(
                                    "observable '{}': expected qubit index after `{name}(`",
                                    self.name
                                ));
                            }
                        };
                        i += 1;
                        if !matches!(toks.get(i), Some(Tok::RParen)) {
                            return Err(format!(
                                "observable '{}': expected `)` after `{name}({q}`",
                                self.name
                            ));
                        }
                        i += 1;
                        paulis.push((q, op));
                    }
                    Some(Tok::Ident(name)) if name.as_str() == "I" => {
                        // Identity — no Pauli to push.
                        i += 1;
                    }
                    _ => {
                        // Must be a numeric factor.
                        let (val, j) = parse_numeric(toks, i, &bindings)
                            .map_err(|e| format!("observable '{}': {e}", self.name))?;
                        coeff *= val;
                        i = j;
                    }
                }
                // Continue multiplying?
                if matches!(toks.get(i), Some(Tok::Star)) {
                    i += 1;
                    continue;
                }
                break;
            }
            terms.push((coeff, paulis));
            first_term = false;
        }

        Ok(OmegaObservable { terms })
    }
}

/// Parse a numeric atom at position `i` (optionally signed): either a
/// float/int literal or an identifier looked up in `bindings`.
fn parse_numeric(
    toks: &[Tok],
    i: usize,
    bindings: &std::collections::HashMap<String, f64>,
) -> Result<(f64, usize), String> {
    let mut j = i;
    let mut sign = 1.0;
    if matches!(toks.get(j), Some(Tok::Minus)) {
        sign = -1.0;
        j += 1;
    }
    let val = match toks.get(j) {
        Some(Tok::Float(f)) => *f,
        Some(Tok::Int(v)) => *v as f64,
        Some(Tok::Ident(name)) => *bindings
            .get(name.as_str())
            .ok_or_else(|| format!("unbound identifier `{name}`"))?,
        other => return Err(format!("expected number, got {other:?}")),
    };
    Ok((sign * val, j + 1))
}

fn is_pauli_letter(s: &str) -> bool {
    matches!(s, "X" | "Y" | "Z")
}

fn pauli_from_letter(s: &str) -> crate::backends::omega::OmegaPauliOp {
    use crate::backends::omega::OmegaPauliOp;
    match s {
        "X" => OmegaPauliOp::X,
        "Y" => OmegaPauliOp::Y,
        "Z" => OmegaPauliOp::Z,
        _ => unreachable!(),
    }
}

#[derive(Clone, Debug, Default)]
pub struct AriaProgram {
    pub circuits: Vec<CircuitTemplate>,
    pub observables: Vec<ObservableTemplate>,
}

impl AriaProgram {
    /// Look up a circuit template by name.
    pub fn circuit(&self, name: &str) -> Option<&CircuitTemplate> {
        self.circuits.iter().find(|c| c.name == name)
    }

    /// Look up an observable template by name.
    pub fn observable(&self, name: &str) -> Option<&ObservableTemplate> {
        self.observables.iter().find(|o| o.name == name)
    }

    /// Instantiate a named circuit with concrete integer parameter bindings.
    /// Parameters not supplied default to 0.
    pub fn instantiate(&self, name: &str, params: &[(&str, i64)]) -> Result<Circuit, String> {
        let tpl = self
            .circuit(name)
            .ok_or_else(|| format!("circuit '{name}' not found in Aria program"))?;
        let mut scope = Scope::default();
        // Declared params (default 0, else from bindings).
        for (pname, _) in &tpl.params {
            let mut val = 0i64;
            for (k, v) in params {
                if *k == pname {
                    val = *v;
                }
            }
            scope.ints.insert(pname.clone(), val);
        }
        let mut circ = Circuit::new(name);
        for a in &tpl.annotations {
            circ.annotate(lower_annotation(a));
        }
        lower_stmts(&tpl.body, &mut circ, &mut scope, self)?;
        Ok(circ)
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser {
    toks: Vec<(Tok, usize)>,
    pos: usize,
}

impl Parser {
    fn new(toks: Vec<(Tok, usize)>) -> Self {
        Self { toks, pos: 0 }
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|(t, _)| t)
    }

    fn peek2(&self) -> Option<&Tok> {
        self.toks.get(self.pos + 1).map(|(t, _)| t)
    }

    fn line(&self) -> usize {
        self.toks
            .get(self.pos)
            .map(|(_, l)| *l)
            .unwrap_or_else(|| self.toks.last().map(|(_, l)| *l).unwrap_or(0))
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos)?.0.clone();
        self.pos += 1;
        Some(t)
    }

    fn expect(&mut self, want: &Tok) -> Result<(), String> {
        let line = self.line();
        match self.peek() {
            Some(t) if std::mem::discriminant(t) == std::mem::discriminant(want) => {
                self.bump();
                Ok(())
            }
            Some(got) => Err(format!("line {line}: expected {want:?}, got {got:?}")),
            None => Err(format!("line {line}: expected {want:?}, got EOF")),
        }
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        let line = self.line();
        match self.bump() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(format!("line {line}: expected identifier, got {other:?}")),
        }
    }

    fn parse_program(&mut self) -> Result<AriaProgram, String> {
        let mut prog = AriaProgram::default();
        let mut pending_ann: Vec<AnnotationAst> = Vec::new();
        while let Some(tok) = self.peek() {
            match tok {
                Tok::At => {
                    pending_ann.push(self.parse_annotation()?);
                }
                Tok::Circuit => {
                    let mut tpl = self.parse_circuit()?;
                    // Attach preceding annotations.
                    tpl.annotations = std::mem::take(&mut pending_ann);
                    prog.circuits.push(tpl);
                }
                Tok::Observable => {
                    // Drop preceding annotations (observables don't carry them yet).
                    pending_ann.clear();
                    let obs = self.parse_observable()?;
                    prog.observables.push(obs);
                }
                other => {
                    return Err(format!(
                        "line {}: top-level: expected '@', 'circuit', or 'observable', got {:?}",
                        self.line(),
                        other
                    ));
                }
            }
        }
        Ok(prog)
    }

    fn parse_annotation(&mut self) -> Result<AnnotationAst, String> {
        self.expect(&Tok::At)?;
        let kind = self.expect_ident()?;
        // Slurp the rest of the annotation line — we capture raw content up
        // to the next '@' / 'circuit' / 'observable' at top level, or end
        // of logical line if a brace block follows.
        let start_line = self.line();
        let mut content = String::new();
        let mut depth = 0i32;
        while let Some((t, l)) = self.toks.get(self.pos).cloned() {
            if depth == 0 && l > start_line && matches!(t, Tok::At | Tok::Circuit | Tok::Observable)
            {
                break;
            }
            match &t {
                Tok::LBrace => depth += 1,
                Tok::RBrace => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                _ => {}
            }
            if !content.is_empty() {
                content.push(' ');
            }
            content.push_str(&format!("{t:?}"));
            self.pos += 1;
            if depth == 0 && l > start_line {
                // Consumed a token on a later line but outside any brace —
                // the annotation is finished.
                break;
            }
        }
        Ok(AnnotationAst { kind, content })
    }

    fn parse_circuit(&mut self) -> Result<CircuitTemplate, String> {
        self.expect(&Tok::Circuit)?;
        let name = self.expect_ident()?;
        let mut params = Vec::new();
        if matches!(self.peek(), Some(Tok::LParen)) {
            self.bump();
            while !matches!(self.peek(), Some(Tok::RParen)) {
                let pname = self.expect_ident()?;
                let ptype = if matches!(self.peek(), Some(Tok::Colon)) {
                    self.bump();
                    self.expect_ident()?
                } else {
                    "int".to_string()
                };
                params.push((pname, ptype));
                if matches!(self.peek(), Some(Tok::Comma)) {
                    self.bump();
                }
            }
            self.expect(&Tok::RParen)?;
        }
        self.expect(&Tok::LBrace)?;
        let body = self.parse_stmts()?;
        self.expect(&Tok::RBrace)?;
        Ok(CircuitTemplate {
            name,
            params,
            annotations: Vec::new(),
            body,
        })
    }

    fn parse_observable(&mut self) -> Result<ObservableTemplate, String> {
        self.expect(&Tok::Observable)?;
        let name = self.expect_ident()?;
        self.expect(&Tok::LBrace)?;
        // Slurp matching braces — we keep the body as a raw token dump
        // (for display) *and* a token vector (for structured re-parse).
        let mut depth = 1i32;
        let mut body = String::new();
        let mut body_tokens: Vec<Tok> = Vec::new();
        while depth > 0 {
            let Some((t, _l)) = self.toks.get(self.pos).cloned() else {
                return Err(format!("line {}: unterminated observable", self.line()));
            };
            self.pos += 1;
            match &t {
                Tok::LBrace => {
                    depth += 1;
                    body.push('{');
                    body_tokens.push(t);
                }
                Tok::RBrace => {
                    depth -= 1;
                    if depth > 0 {
                        body.push('}');
                        body_tokens.push(t);
                    }
                }
                other => {
                    if !body.is_empty() {
                        body.push(' ');
                    }
                    body.push_str(&format!("{other:?}"));
                    body_tokens.push(other.clone());
                }
            }
        }
        Ok(ObservableTemplate {
            name,
            body,
            body_tokens,
        })
    }

    fn parse_stmts(&mut self) -> Result<Vec<Stmt>, String> {
        let mut out = Vec::new();
        while let Some(tok) = self.peek() {
            if matches!(tok, Tok::RBrace) {
                break;
            }
            let s = self.parse_stmt()?;
            out.push(s);
        }
        Ok(out)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        let line = self.line();
        let tok = self
            .peek()
            .ok_or_else(|| format!("line {line}: unexpected EOF"))?;
        match tok {
            Tok::Qreg => {
                self.bump();
                let name = self.expect_ident()?;
                self.expect(&Tok::LBracket)?;
                let size = self.parse_expr()?;
                self.expect(&Tok::RBracket)?;
                Ok(Stmt::Qreg { name, size })
            }
            Tok::Creg => {
                self.bump();
                let name = self.expect_ident()?;
                self.expect(&Tok::LBracket)?;
                let size = self.parse_expr()?;
                self.expect(&Tok::RBracket)?;
                Ok(Stmt::Creg { name, size })
            }
            Tok::Let | Tok::Var => {
                self.bump();
                let name = self.expect_ident()?;
                self.expect(&Tok::Eq)?;
                // Special form: `let theta = symbolic[N]`.
                if matches!(self.peek(), Some(Tok::Symbolic)) {
                    self.bump();
                    self.expect(&Tok::LBracket)?;
                    let count = self.parse_expr()?;
                    self.expect(&Tok::RBracket)?;
                    return Ok(Stmt::Symbolic { name, count });
                }
                let value = self.parse_expr()?;
                Ok(Stmt::Let { name, value })
            }
            Tok::Apply => {
                self.bump();
                let gate = self.expect_ident()?;
                let mut params = Vec::new();
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.bump();
                    while !matches!(self.peek(), Some(Tok::RParen)) {
                        params.push(self.parse_expr()?);
                        if matches!(self.peek(), Some(Tok::Comma)) {
                            self.bump();
                        }
                    }
                    self.expect(&Tok::RParen)?;
                }
                self.expect(&Tok::On)?;
                let qubits = self.parse_target_list()?;
                Ok(Stmt::Apply {
                    gate,
                    params,
                    qubits,
                })
            }
            Tok::Measure => {
                self.bump();
                let from = self.parse_target()?;
                self.expect(&Tok::Arrow)?;
                let to = self.parse_target()?;
                match (&from, &to) {
                    (Target::Reg(q), Target::Reg(c)) => Ok(Stmt::MeasureAll {
                        q_reg: q.clone(),
                        c_reg: c.clone(),
                    }),
                    _ => Ok(Stmt::Measure { from, to }),
                }
            }
            Tok::Repeat => {
                self.bump();
                // Two forms:
                //   repeat EXPR { … }
                //   repeat IDENT from A to B [ step S ] { … }
                match (self.peek().cloned(), self.peek2().cloned()) {
                    (Some(Tok::Ident(_)), Some(Tok::From)) => {
                        let var = self.expect_ident()?;
                        self.expect(&Tok::From)?;
                        let from = self.parse_expr()?;
                        self.expect(&Tok::To)?;
                        let to = self.parse_expr()?;
                        let step = if matches!(self.peek(), Some(Tok::Step)) {
                            self.bump();
                            self.parse_expr()?
                        } else {
                            Expr::IntLit(1)
                        };
                        self.expect(&Tok::LBrace)?;
                        let body = self.parse_stmts()?;
                        self.expect(&Tok::RBrace)?;
                        Ok(Stmt::RepeatRange {
                            var,
                            from,
                            to,
                            step,
                            body,
                        })
                    }
                    _ => {
                        let count = self.parse_expr()?;
                        self.expect(&Tok::LBrace)?;
                        let body = self.parse_stmts()?;
                        self.expect(&Tok::RBrace)?;
                        Ok(Stmt::Repeat { count, body })
                    }
                }
            }
            Tok::When => {
                self.bump();
                let cond = self.parse_expr()?;
                self.expect(&Tok::LBrace)?;
                let body = self.parse_stmts()?;
                self.expect(&Tok::RBrace)?;
                Ok(Stmt::When { cond, body })
            }
            Tok::Oracle => {
                self.bump();
                let name = self.expect_ident()?;
                let mut args = Vec::new();
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.bump();
                    while !matches!(self.peek(), Some(Tok::RParen)) {
                        args.push(self.parse_expr()?);
                        if matches!(self.peek(), Some(Tok::Comma)) {
                            self.bump();
                        }
                    }
                    self.expect(&Tok::RParen)?;
                }
                self.expect(&Tok::On)?;
                let qubits = self.parse_target_list()?;
                Ok(Stmt::Oracle { name, args, qubits })
            }
            other => Err(format!(
                "line {line}: unexpected token {other:?} at statement start"
            )),
        }
    }

    fn parse_target_list(&mut self) -> Result<Vec<Target>, String> {
        let mut out = vec![self.parse_target()?];
        while matches!(self.peek(), Some(Tok::Comma)) {
            self.bump();
            out.push(self.parse_target()?);
        }
        Ok(out)
    }

    fn parse_target(&mut self) -> Result<Target, String> {
        let name = self.expect_ident()?;
        if matches!(self.peek(), Some(Tok::LBracket)) {
            self.bump();
            let idx = self.parse_expr()?;
            self.expect(&Tok::RBracket)?;
            Ok(Target::Indexed(name, idx))
        } else {
            Ok(Target::Reg(name))
        }
    }

    // Pratt-style expression parser.
    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.peek() {
                Some(Tok::EqEq) => "==",
                Some(Tok::NotEq) => "!=",
                Some(Tok::Lt) => "<",
                Some(Tok::Le) => "<=",
                Some(Tok::Gt) => ">",
                Some(Tok::Ge) => ">=",
                _ => break,
            };
            self.bump();
            let rhs = self.parse_add()?;
            lhs = match op {
                "==" => Expr::CmpEq(Box::new(lhs), Box::new(rhs)),
                "!=" => Expr::CmpNe(Box::new(lhs), Box::new(rhs)),
                "<" => Expr::CmpLt(Box::new(lhs), Box::new(rhs)),
                "<=" => Expr::CmpLe(Box::new(lhs), Box::new(rhs)),
                ">" => Expr::CmpGt(Box::new(lhs), Box::new(rhs)),
                ">=" => Expr::CmpGe(Box::new(lhs), Box::new(rhs)),
                _ => unreachable!(),
            };
        }
        Ok(lhs)
    }

    fn parse_add(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_mul()?;
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.bump();
                    let rhs = self.parse_mul()?;
                    lhs = Expr::Add(Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Minus) => {
                    self.bump();
                    let rhs = self.parse_mul()?;
                    lhs = Expr::Sub(Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_pow()?;
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.bump();
                    let rhs = self.parse_pow()?;
                    lhs = Expr::Mul(Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Slash) => {
                    self.bump();
                    let rhs = self.parse_pow()?;
                    lhs = Expr::Div(Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Percent) => {
                    self.bump();
                    let rhs = self.parse_pow()?;
                    lhs = Expr::Mod(Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_pow(&mut self) -> Result<Expr, String> {
        let base = self.parse_unary()?;
        if matches!(self.peek(), Some(Tok::Caret)) {
            self.bump();
            // Right-associative.
            let exp = self.parse_pow()?;
            Ok(Expr::Pow(Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if matches!(self.peek(), Some(Tok::Minus)) {
            self.bump();
            let e = self.parse_unary()?;
            return Ok(Expr::Neg(Box::new(e)));
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<Expr, String> {
        let line = self.line();
        let t = self
            .bump()
            .ok_or_else(|| format!("line {line}: unexpected EOF in expr"))?;
        match t {
            Tok::Int(v) => Ok(Expr::IntLit(v)),
            Tok::Float(v) => Ok(Expr::FloatLit(v)),
            Tok::Pi => Ok(Expr::Pi),
            Tok::True => Ok(Expr::IntLit(1)),
            Tok::False => Ok(Expr::IntLit(0)),
            Tok::LParen => {
                let e = self.parse_expr()?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            Tok::Ident(name) => {
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.bump();
                    let mut args = Vec::new();
                    while !matches!(self.peek(), Some(Tok::RParen)) {
                        args.push(self.parse_expr()?);
                        if matches!(self.peek(), Some(Tok::Comma)) {
                            self.bump();
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    Ok(Expr::Call(name, args))
                } else if matches!(self.peek(), Some(Tok::LBracket)) {
                    self.bump();
                    let idx = self.parse_expr()?;
                    self.expect(&Tok::RBracket)?;
                    Ok(Expr::Index(name, Box::new(idx)))
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            other => Err(format!(
                "line {line}: unexpected token {other:?} in expression"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Lowering templates → Circuit
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Debug)]
struct Scope {
    /// Compile-time integer bindings (loop vars, circuit params).
    ints: HashMap<String, i64>,
    /// Compile-time float bindings (`let x = 3.14`).
    floats: HashMap<String, f64>,
    /// Symbolic arrays: name → element count (symbols `{name}_{i}`).
    symbolic: HashMap<String, i64>,
}

#[derive(Clone, Debug)]
enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl Value {
    fn as_f64(&self) -> f64 {
        match self {
            Value::Int(i) => *i as f64,
            Value::Float(f) => *f,
            Value::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
    fn as_i64(&self) -> Result<i64, String> {
        match self {
            Value::Int(i) => Ok(*i),
            Value::Float(f) => {
                if (f - f.round()).abs() < 1e-9 {
                    Ok(f.round() as i64)
                } else {
                    Err(format!("expected integer, got float {f}"))
                }
            }
            Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        }
    }
    fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
        }
    }
}

fn eval_expr(e: &Expr, scope: &Scope) -> Result<Value, String> {
    match e {
        Expr::IntLit(i) => Ok(Value::Int(*i)),
        Expr::FloatLit(f) => Ok(Value::Float(*f)),
        Expr::Pi => Ok(Value::Float(PI)),
        Expr::Ident(name) => {
            if let Some(i) = scope.ints.get(name) {
                Ok(Value::Int(*i))
            } else if let Some(f) = scope.floats.get(name) {
                Ok(Value::Float(*f))
            } else {
                Err(format!("unbound identifier '{name}'"))
            }
        }
        Expr::Index(name, idx) => {
            // Symbolic-array element → we shouldn't reach here as a Value:
            // those are only meaningful as gate params, not as evaluated
            // numbers. If someone does arithmetic on `theta[0]`, fail loudly.
            if scope.symbolic.contains_key(name) {
                return Err(format!(
                    "cannot use symbolic array '{name}' as numeric value"
                ));
            }
            Err(format!(
                "'{name}[..]' is a register reference, not a numeric value (context: {idx:?})"
            ))
        }
        Expr::Neg(a) => {
            let v = eval_expr(a, scope)?;
            match v {
                Value::Int(i) => Ok(Value::Int(-i)),
                Value::Float(f) => Ok(Value::Float(-f)),
                Value::Bool(b) => Ok(Value::Int(if b { -1 } else { 0 })),
            }
        }
        Expr::Add(a, b) => arith(
            eval_expr(a, scope)?,
            eval_expr(b, scope)?,
            |x, y| x + y,
            i64::checked_add,
        ),
        Expr::Sub(a, b) => arith(
            eval_expr(a, scope)?,
            eval_expr(b, scope)?,
            |x, y| x - y,
            i64::checked_sub,
        ),
        Expr::Mul(a, b) => arith(
            eval_expr(a, scope)?,
            eval_expr(b, scope)?,
            |x, y| x * y,
            i64::checked_mul,
        ),
        Expr::Div(a, b) => {
            let x = eval_expr(a, scope)?;
            let y = eval_expr(b, scope)?;
            // Division on two ints: floor-divide (Python `//`-style). This
            // matches the surface-syntax convention — authors who want
            // float division write `2.0` explicitly for at least one side.
            if let (Value::Int(xi), Value::Int(yi)) = (&x, &y) {
                if *yi == 0 {
                    return Err("division by zero".into());
                }
                return Ok(Value::Int(xi.div_euclid(*yi)));
            }
            let yf = y.as_f64();
            if yf == 0.0 {
                return Err("division by zero".into());
            }
            Ok(Value::Float(x.as_f64() / yf))
        }
        Expr::Pow(a, b) => {
            let x = eval_expr(a, scope)?;
            let y = eval_expr(b, scope)?;
            if let (Value::Int(xi), Value::Int(yi)) = (&x, &y) {
                if *yi >= 0 && *yi <= 62 {
                    return Ok(Value::Int((*xi).pow(*yi as u32)));
                }
            }
            Ok(Value::Float(x.as_f64().powf(y.as_f64())))
        }
        Expr::Mod(a, b) => {
            let x = eval_expr(a, scope)?.as_i64()?;
            let y = eval_expr(b, scope)?.as_i64()?;
            if y == 0 {
                return Err("modulo by zero".into());
            }
            Ok(Value::Int(x.rem_euclid(y)))
        }
        Expr::Call(name, args) => eval_call(name, args, scope),
        Expr::CmpEq(a, b) => Ok(Value::Bool(
            (eval_expr(a, scope)?.as_f64() - eval_expr(b, scope)?.as_f64()).abs() < 1e-12,
        )),
        Expr::CmpNe(a, b) => Ok(Value::Bool(
            (eval_expr(a, scope)?.as_f64() - eval_expr(b, scope)?.as_f64()).abs() >= 1e-12,
        )),
        Expr::CmpLt(a, b) => Ok(Value::Bool(
            eval_expr(a, scope)?.as_f64() < eval_expr(b, scope)?.as_f64(),
        )),
        Expr::CmpLe(a, b) => Ok(Value::Bool(
            eval_expr(a, scope)?.as_f64() <= eval_expr(b, scope)?.as_f64(),
        )),
        Expr::CmpGt(a, b) => Ok(Value::Bool(
            eval_expr(a, scope)?.as_f64() > eval_expr(b, scope)?.as_f64(),
        )),
        Expr::CmpGe(a, b) => Ok(Value::Bool(
            eval_expr(a, scope)?.as_f64() >= eval_expr(b, scope)?.as_f64(),
        )),
    }
}

fn arith<F, G>(a: Value, b: Value, float_op: F, int_op: G) -> Result<Value, String>
where
    F: Fn(f64, f64) -> f64,
    G: Fn(i64, i64) -> Option<i64>,
{
    if let (Value::Int(x), Value::Int(y)) = (&a, &b) {
        if let Some(v) = int_op(*x, *y) {
            return Ok(Value::Int(v));
        }
    }
    Ok(Value::Float(float_op(a.as_f64(), b.as_f64())))
}

fn eval_call(name: &str, args: &[Expr], scope: &Scope) -> Result<Value, String> {
    let vals: Result<Vec<Value>, _> = args.iter().map(|a| eval_expr(a, scope)).collect();
    let vals = vals?;
    match (name, vals.as_slice()) {
        ("sin", [x]) => Ok(Value::Float(x.as_f64().sin())),
        ("cos", [x]) => Ok(Value::Float(x.as_f64().cos())),
        ("tan", [x]) => Ok(Value::Float(x.as_f64().tan())),
        // Inverse trig — kept in sync with the ParamExpr evaluator (ast/expr.rs) so a `let`
        // binding and a gate-angle expression fold identically (used by LCU PREPARE angles).
        ("asin", [x]) => Ok(Value::Float(x.as_f64().asin())),
        ("acos", [x]) => Ok(Value::Float(x.as_f64().acos())),
        ("atan", [x]) => Ok(Value::Float(x.as_f64().atan())),
        ("exp", [x]) => Ok(Value::Float(x.as_f64().exp())),
        ("log", [x]) => Ok(Value::Float(x.as_f64().ln())),
        ("sqrt", [x]) => Ok(Value::Float(x.as_f64().sqrt())),
        ("abs", [x]) => match x {
            Value::Int(i) => Ok(Value::Int(i.abs())),
            _ => Ok(Value::Float(x.as_f64().abs())),
        },
        ("bit", [n, k]) => {
            let n = n.as_i64()?;
            let k = k.as_i64()?;
            Ok(Value::Int((n >> k) & 1))
        }
        ("floor", [x]) => Ok(Value::Int(x.as_f64().floor() as i64)),
        ("ceil", [x]) => Ok(Value::Int(x.as_f64().ceil() as i64)),
        ("min", [a, b]) => {
            if let (Value::Int(x), Value::Int(y)) = (a, b) {
                Ok(Value::Int((*x).min(*y)))
            } else {
                Ok(Value::Float(a.as_f64().min(b.as_f64())))
            }
        }
        ("max", [a, b]) => {
            if let (Value::Int(x), Value::Int(y)) = (a, b) {
                Ok(Value::Int((*x).max(*y)))
            } else {
                Ok(Value::Float(a.as_f64().max(b.as_f64())))
            }
        }
        _ => Err(format!(
            "unknown function '{name}' with {} args",
            vals.len()
        )),
    }
}

fn eval_int(e: &Expr, scope: &Scope) -> Result<i64, String> {
    eval_expr(e, scope)?.as_i64()
}

/// Lower a gate-parameter expression into a `ParamExpr`. This does NOT
/// fully evaluate — it preserves symbols coming from `let theta =
/// symbolic[N]` declarations, so gates can retain symbolic parameters.
fn lower_param(e: &Expr, scope: &Scope) -> Result<ParamExpr, String> {
    match e {
        Expr::IntLit(i) => Ok(ParamExpr::Concrete(*i as f64)),
        Expr::FloatLit(f) => Ok(ParamExpr::Concrete(*f)),
        Expr::Pi => Ok(ParamExpr::Pi),
        Expr::Ident(name) => {
            if let Some(i) = scope.ints.get(name) {
                Ok(ParamExpr::Concrete(*i as f64))
            } else if let Some(f) = scope.floats.get(name) {
                Ok(ParamExpr::Concrete(*f))
            } else {
                Ok(ParamExpr::Symbol(name.clone()))
            }
        }
        Expr::Index(name, idx) => {
            // theta[k] where theta is a declared symbolic array.
            let count = *scope.symbolic.get(name).ok_or_else(|| {
                format!("'{name}[..]' used in gate param but not a symbolic array")
            })?;
            let k = eval_int(idx, scope)?;
            if k < 0 || k >= count {
                return Err(format!(
                    "symbolic array '{name}' index {k} out of range 0..{count}"
                ));
            }
            Ok(ParamExpr::Symbol(format!("{name}_{k}")))
        }
        Expr::Neg(a) => Ok(-lower_param(a, scope)?),
        Expr::Add(a, b) => Ok(lower_param(a, scope)? + lower_param(b, scope)?),
        Expr::Sub(a, b) => Ok(lower_param(a, scope)? + (-lower_param(b, scope)?)),
        Expr::Mul(a, b) => Ok(lower_param(a, scope)? * lower_param(b, scope)?),
        Expr::Div(a, b) => Ok(lower_param(a, scope)? / lower_param(b, scope)?),
        Expr::Pow(a, b) => {
            // No native pow in ParamExpr — if both sides are concrete,
            // fold; else fail (symbolic powers aren't expressible here).
            let af = try_concrete_f64(a, scope);
            let bf = try_concrete_f64(b, scope);
            if let (Some(x), Some(y)) = (af, bf) {
                Ok(ParamExpr::Concrete(x.powf(y)))
            } else {
                Err("symbolic exponentiation not supported in gate params".into())
            }
        }
        Expr::Mod(_, _) => Err("modulo not supported in gate params".into()),
        Expr::Call(name, args) => {
            let lowered: Result<Vec<ParamExpr>, _> =
                args.iter().map(|a| lower_param(a, scope)).collect();
            Ok(ParamExpr::FnCall(name.clone(), lowered?))
        }
        Expr::CmpEq(_, _)
        | Expr::CmpNe(_, _)
        | Expr::CmpLt(_, _)
        | Expr::CmpLe(_, _)
        | Expr::CmpGt(_, _)
        | Expr::CmpGe(_, _) => Err("comparison cannot be used as a gate parameter".into()),
    }
}

fn try_concrete_f64(e: &Expr, scope: &Scope) -> Option<f64> {
    eval_expr(e, scope).ok().map(|v| v.as_f64())
}

fn resolve_target(t: &Target, scope: &Scope) -> Result<Qubit, String> {
    match t {
        Target::Reg(name) => Ok(Qubit::new(name, 0)),
        Target::Indexed(name, idx) => {
            let i = eval_int(idx, scope)?;
            if i < 0 {
                return Err(format!("negative qubit index {i} on register '{name}'"));
            }
            Ok(Qubit::new(name, i as usize))
        }
    }
}

fn resolve_clbit(t: &Target, scope: &Scope) -> Result<Clbit, String> {
    match t {
        Target::Reg(name) => Ok(Clbit::new(name, 0)),
        Target::Indexed(name, idx) => {
            let i = eval_int(idx, scope)?;
            Ok(Clbit::new(name, i as usize))
        }
    }
}

fn gate_from_name(name: &str, params: Vec<ParamExpr>) -> Result<GateDef, String> {
    let kind = match name.to_ascii_uppercase().as_str() {
        "I" | "ID" => GateKind::I,
        "X" => GateKind::X,
        "Y" => GateKind::Y,
        "Z" => GateKind::Z,
        "H" => GateKind::H,
        "S" => GateKind::S,
        "SDG" => GateKind::Sdg,
        "T" => GateKind::T,
        "TDG" => GateKind::Tdg,
        "SX" => GateKind::SX,
        "RX" => GateKind::RX,
        "RY" => GateKind::RY,
        "RZ" => GateKind::RZ,
        "P" | "U1" => GateKind::P,
        "U" | "U3" => GateKind::U,
        "CX" | "CNOT" => GateKind::CX,
        "CY" => GateKind::CY,
        "CZ" => GateKind::CZ,
        "SWAP" => GateKind::SWAP,
        "CP" => GateKind::CP,
        "CRZ" => GateKind::CRz,
        // Reset is a CHANNEL, not a unitary, but it lives in GateKind, has a
        // builder (`Circuit::reset_qubit`), a QASM spelling in both
        // directions, and an audited implementation in every backend — it
        // simply had no Aria-language spelling. `apply RESET on q[i]` lowers
        // to exactly the instruction `reset_qubit` produces.
        "RESET" => GateKind::Reset,
        "CCX" | "TOFFOLI" => GateKind::CCX,
        "CSWAP" | "FREDKIN" => GateKind::CSWAP,
        "RXX" => GateKind::RXX,
        "RYY" => GateKind::RYY,
        "RZZ" => GateKind::RZZ,
        "RBS" => GateKind::RBS,
        other => return Err(format!("unknown gate '{other}'")),
    };
    Ok(GateDef::with_exprs(kind, params))
}

/// If `e` is a simple `reg[idx_lit] == int_lit` (or the symmetric form),
/// return the `(Clbit, expected)` pair to attach as a per-instruction
/// classical condition. Anything more complex returns `None`, which the
/// caller MUST turn into a lowering error: there is no way to attach such
/// a guard to the circuit model, and executing the body unconditionally
/// (the old fallback) produced a silently wrong circuit.
fn extract_simple_clbit_cond(e: &Expr, scope: &Scope) -> Option<(Clbit, u64)> {
    let (lhs, rhs) = match e {
        Expr::CmpEq(a, b) => (a.as_ref(), b.as_ref()),
        _ => return None,
    };
    let (idx_expr, val_expr) = match (lhs, rhs) {
        (Expr::Index(_, _), _) => (lhs, rhs),
        (_, Expr::Index(_, _)) => (rhs, lhs),
        _ => return None,
    };
    let (name, idx_box) = match idx_expr {
        Expr::Index(n, i) => (n.clone(), i),
        _ => return None,
    };
    let idx = eval_int(idx_box, scope).ok()?;
    if idx < 0 {
        return None;
    }
    let v = match val_expr {
        Expr::IntLit(v) => *v,
        _ => return None,
    };
    if v < 0 {
        return None;
    }
    Some((Clbit::new(&name, idx as usize), v as u64))
}

/// Is this `when` condition a RUNTIME guard (on measured classical bits) or a
/// compile-time one (on loop variables and constants)?
///
/// Decided by consulting the circuit's DECLARED classical registers.
///
/// It used to be decided by the register's NAME:
/// `name.starts_with('m') || name == "c"`. A creg called `flags` therefore
/// routed to compile-time evaluation and failed obscurely — while the emitter
/// happily wrote `when flags[0] == 1`, so the language could express a circuit
/// its own lowering mis-handled. Meanwhile a loop variable named `mask` was
/// treated as a measurement.
///
/// Same class as `RegisterDecl::polarized` being a flag rather than a naming
/// convention: a convention does not survive a rename, and the failure is
/// silent-adjacent.
fn is_runtime_cond(e: &Expr, circ: &Circuit) -> bool {
    let is_creg = |name: &str| {
        circ.registers
            .iter()
            .any(|r| r.kind == RegisterKind::Classical && r.name == name)
    };
    match e {
        Expr::Index(name, _) => is_creg(name),
        Expr::CmpEq(a, b)
        | Expr::CmpNe(a, b)
        | Expr::CmpLt(a, b)
        | Expr::CmpLe(a, b)
        | Expr::CmpGt(a, b)
        | Expr::CmpGe(a, b) => is_runtime_cond(a, circ) || is_runtime_cond(b, circ),
        Expr::Neg(a) => is_runtime_cond(a, circ),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
            is_runtime_cond(a, circ) || is_runtime_cond(b, circ)
        }
        _ => false,
    }
}

fn lower_stmts(
    stmts: &[Stmt],
    circ: &mut Circuit,
    scope: &mut Scope,
    prog: &AriaProgram,
) -> Result<(), String> {
    for s in stmts {
        lower_stmt(s, circ, scope, prog)?;
    }
    Ok(())
}

fn lower_stmt(
    s: &Stmt,
    circ: &mut Circuit,
    scope: &mut Scope,
    prog: &AriaProgram,
) -> Result<(), String> {
    match s {
        Stmt::Qreg { name, size } => {
            let n = eval_int(size, scope)?;
            if n < 0 {
                return Err(format!("negative qreg size {n}"));
            }
            circ.qreg(name, n as usize);
        }
        Stmt::Creg { name, size } => {
            let n = eval_int(size, scope)?;
            if n < 0 {
                return Err(format!("negative creg size {n}"));
            }
            circ.creg(name, n as usize);
        }
        Stmt::Let { name, value } => {
            let v = eval_expr(value, scope)?;
            match v {
                Value::Int(i) => {
                    scope.ints.insert(name.clone(), i);
                }
                Value::Float(f) => {
                    scope.floats.insert(name.clone(), f);
                }
                Value::Bool(b) => {
                    scope.ints.insert(name.clone(), if b { 1 } else { 0 });
                }
            }
        }
        Stmt::Symbolic { name, count } => {
            let n = eval_int(count, scope)?;
            scope.symbolic.insert(name.clone(), n);
        }
        Stmt::Apply {
            gate,
            params,
            qubits,
        } => {
            let lowered_params: Result<Vec<ParamExpr>, _> =
                params.iter().map(|e| lower_param(e, scope)).collect();
            let gd = gate_from_name(gate, lowered_params?)?;
            let qs: Result<Vec<Qubit>, _> =
                qubits.iter().map(|t| resolve_target(t, scope)).collect();
            circ.apply(gd, qs?);
        }
        Stmt::Measure { from, to } => {
            let q = resolve_target(from, scope)?;
            let c = resolve_clbit(to, scope)?;
            circ.measure(&q, &c);
        }
        Stmt::MeasureAll { q_reg, c_reg } => {
            let q_size = circ
                .registers
                .iter()
                .find(|r| r.name == *q_reg && r.kind == RegisterKind::Quantum)
                .map(|r| r.size)
                .ok_or_else(|| format!("quantum register '{q_reg}' not declared"))?;
            let c_size = circ
                .registers
                .iter()
                .find(|r| r.name == *c_reg && r.kind == RegisterKind::Classical)
                .map(|r| r.size)
                .ok_or_else(|| format!("classical register '{c_reg}' not declared"))?;
            let pairs = q_size.min(c_size);
            for i in 0..pairs {
                let q = Qubit::new(q_reg, i);
                let c = Clbit::new(c_reg, i);
                circ.measure(&q, &c);
            }
        }
        Stmt::Repeat { count, body } => {
            let n = eval_int(count, scope)?;
            for _ in 0..n {
                lower_stmts(body, circ, scope, prog)?;
            }
        }
        Stmt::RepeatRange {
            var,
            from,
            to,
            step,
            body,
        } => {
            let a = eval_int(from, scope)?;
            let b = eval_int(to, scope)?;
            let s = eval_int(step, scope)?;
            if s == 0 {
                return Err("repeat step cannot be zero".into());
            }
            let save = scope.ints.get(var).copied();
            let mut i = a;
            // Inclusive-to-inclusive range, matching the textual `repeat i
            // from 0 to n-1 { … }` form used throughout examples/aria/.
            loop {
                if s > 0 && i > b {
                    break;
                }
                if s < 0 && i < b {
                    break;
                }
                scope.ints.insert(var.clone(), i);
                lower_stmts(body, circ, scope, prog)?;
                i += s;
            }
            match save {
                Some(v) => {
                    scope.ints.insert(var.clone(), v);
                }
                None => {
                    scope.ints.remove(var);
                }
            }
        }
        Stmt::When { cond, body } => {
            let runtime = is_runtime_cond(cond, circ);
            if runtime {
                // A runtime guard must become a per-instruction classical
                // condition. If it cannot, refuse: lowering the body
                // unconditionally (the old behaviour, with the guard demoted
                // to a comment annotation nothing reads) executes it on
                // every shot — a silently wrong circuit.
                let Some((cl, v)) = extract_simple_clbit_cond(cond, scope) else {
                    return Err(format!(
                        "cannot lower `when {}`: a runtime `when` guard must have \
                         the form `reg[i] == literal` (or its mirror) to become a \
                         classically-controlled instruction; rewrite the guard or \
                         split it into supported single-bit comparisons",
                        describe_expr(cond)
                    ));
                };
                circ.annotate(Annotation::Comment(format!(
                    "conditional execution: when {}",
                    describe_expr(cond)
                )));
                let start = circ.instructions.len();
                lower_stmts(body, circ, scope, prog)?;
                for inst in &mut circ.instructions[start..] {
                    // The circuit model holds ONE classical condition per
                    // instruction. An instruction that already carries one
                    // came from a nested runtime `when`; stamping around it
                    // (the old behaviour) silently dropped THIS outer guard.
                    if let Some((prev_cl, prev_v)) = &inst.condition {
                        return Err(format!(
                            "nested runtime `when` is not supported: an instruction \
                             under `when {}` already carries the condition \
                             `{}[{}] == {}`, and the circuit model cannot represent \
                             the conjunction of both guards; flatten the guards into \
                             a single `reg[i] == literal` condition",
                            describe_expr(cond),
                            prev_cl.register,
                            prev_cl.index,
                            prev_v
                        ));
                    }
                    inst.condition = Some((cl.clone(), v));
                }
            } else {
                let v = eval_expr(cond, scope)?;
                if v.as_bool() {
                    lower_stmts(body, circ, scope, prog)?;
                }
            }
        }
        Stmt::Oracle { name, args, qubits } => {
            let tpl = prog
                .circuit(name)
                .ok_or_else(|| format!("oracle '{name}' not defined in Aria program"))?;
            // Evaluate args into int bindings for the callee scope.
            let mut callee = Scope::default();
            if tpl.params.len() != args.len() {
                return Err(format!(
                    "oracle '{name}' expects {} args, got {}",
                    tpl.params.len(),
                    args.len()
                ));
            }
            for ((pname, _), a) in tpl.params.iter().zip(args.iter()) {
                callee.ints.insert(pname.clone(), eval_int(a, scope)?);
            }
            // Lower the sub-circuit inline into `circ`. Rather than building
            // a separate Circuit (which has its own register namespace), we
            // walk the body directly in the caller's `circ`, mapping the
            // sub-circuit's register references onto the call-site targets
            // when possible. If the oracle doesn't declare its own registers
            // matching caller names, we emit them in the caller too.
            let caller_targets = qubits.clone();
            // Snapshot the register-list length so we can detect exactly
            // which registers the oracle body adds (by position, not name —
            // names repeat when the same oracle is called from a `repeat`
            // loop).
            let regs_len_before = circ.registers.len();
            let inst_len_before = circ.instructions.len();
            lower_stmts(&tpl.body, circ, &mut callee, prog)?;
            let new_reg_names: Vec<String> = circ.registers[regs_len_before..]
                .iter()
                .map(|r| r.name.clone())
                .collect();
            // Single-target oracle: map every oracle-local qubit reference
            // onto the caller's target register (optionally offset). This
            // matches the idiom `oracle NAME(args) on q` which passes the
            // whole caller register.
            if caller_targets.len() == 1 {
                let (caller_reg, caller_base) = match &caller_targets[0] {
                    Target::Reg(r) => (r.clone(), None),
                    Target::Indexed(r, idx) => (r.clone(), Some(eval_int(idx, scope)?)),
                };
                for inst in &mut circ.instructions[inst_len_before..] {
                    for q in &mut inst.qubits {
                        if new_reg_names.contains(&q.register) {
                            q.register = caller_reg.clone();
                            if let Some(base) = caller_base {
                                q.index += base as usize;
                            }
                        }
                    }
                }
            }
            // Drop the oracle-added register declarations regardless — the
            // caller already owns the backing qubits. If we left the extras
            // in place, `n_qubits()` would double-count.
            circ.registers.truncate(regs_len_before);
        }
    }
    Ok(())
}

fn describe_expr(e: &Expr) -> String {
    match e {
        Expr::IntLit(i) => i.to_string(),
        Expr::FloatLit(f) => f.to_string(),
        Expr::Pi => "pi".into(),
        Expr::Ident(n) => n.clone(),
        Expr::Index(n, idx) => format!("{n}[{}]", describe_expr(idx)),
        Expr::Neg(a) => format!("-{}", describe_expr(a)),
        Expr::Add(a, b) => format!("({} + {})", describe_expr(a), describe_expr(b)),
        Expr::Sub(a, b) => format!("({} - {})", describe_expr(a), describe_expr(b)),
        Expr::Mul(a, b) => format!("({} * {})", describe_expr(a), describe_expr(b)),
        Expr::Div(a, b) => format!("({} / {})", describe_expr(a), describe_expr(b)),
        Expr::Pow(a, b) => format!("({} ^ {})", describe_expr(a), describe_expr(b)),
        Expr::Mod(a, b) => format!("({} % {})", describe_expr(a), describe_expr(b)),
        Expr::Call(n, args) => {
            let s: Vec<String> = args.iter().map(describe_expr).collect();
            format!("{n}({})", s.join(", "))
        }
        Expr::CmpEq(a, b) => format!("{} == {}", describe_expr(a), describe_expr(b)),
        Expr::CmpNe(a, b) => format!("{} != {}", describe_expr(a), describe_expr(b)),
        Expr::CmpLt(a, b) => format!("{} < {}", describe_expr(a), describe_expr(b)),
        Expr::CmpLe(a, b) => format!("{} <= {}", describe_expr(a), describe_expr(b)),
        Expr::CmpGt(a, b) => format!("{} > {}", describe_expr(a), describe_expr(b)),
        Expr::CmpGe(a, b) => format!("{} >= {}", describe_expr(a), describe_expr(b)),
    }
}

fn lower_annotation(a: &AnnotationAst) -> Annotation {
    match a.kind.as_str() {
        "assert" if a.content.contains("Ident(\"unitary\")") => {
            Annotation::Assert(Property::Unitary)
        }
        "assert" if a.content.contains("Ident(\"self_inverse\")") => {
            Annotation::Assert(Property::SelfInverse)
        }
        "assert" if a.content.contains("Ident(\"hermitian\")") => {
            Annotation::Assert(Property::Hermitian)
        }
        "prove" => {
            // Best-effort: extract the quoted name.
            let name = a
                .content
                .split_once("StringLit(\"")
                .map(|x| x.1)
                .and_then(|s| s.split("\")").next())
                .unwrap_or("unnamed")
                .to_string();
            Annotation::Prove {
                name,
                property: Property::Custom(a.content.clone()),
            }
        }
        "bound" | "resource_bound" => Annotation::Comment(format!("resource bound: {}", a.content)),
        _ => Annotation::Comment(format!("@{}: {}", a.kind, a.content)),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse an Aria source string into an [`AriaProgram`].
pub fn parse_aria(src: &str) -> Result<AriaProgram, String> {
    let mut lex = Lexer::new(src);
    let toks = lex.tokenize()?;
    let mut parser = Parser::new(toks);
    parser.parse_program()
}

/// Convenience: parse an Aria source string and instantiate a single
/// circuit by name (no parameters).
pub fn parse_aria_circuit(src: &str, name: &str) -> Result<Circuit, String> {
    let prog = parse_aria(src)?;
    prog.instantiate(name, &[])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const BELL: &str = r#"
-- Bell state
@assert unitary
@prove "bell_correct" equiv { creates (00 + 11)/sqrt(2) }
@bound gate_count = 2

circuit Bell {
    qreg q[2]
    creg c[2]
    apply H on q[0]
    apply CX on q[0], q[1]
    measure q -> c
}
"#;

    #[test]
    fn parse_bell() {
        let prog = parse_aria(BELL).expect("parse");
        let circ = prog.instantiate("Bell", &[]).expect("instantiate");
        assert_eq!(circ.n_qubits(), 2);
        assert_eq!(circ.n_clbits(), 2);
        // H + CX + 2 measures
        assert_eq!(circ.instructions.len(), 4);
        assert_eq!(circ.instructions[0].gate.kind, GateKind::H);
        assert_eq!(circ.instructions[1].gate.kind, GateKind::CX);
        assert_eq!(circ.instructions[2].gate.kind, GateKind::Measure);
        // Annotations survived.
        assert!(circ
            .annotations
            .iter()
            .any(|a| matches!(a, Annotation::Assert(Property::Unitary))));
    }

    #[test]
    fn parse_scientific_notation_float_literals() {
        // Regression: `RY(1e-05)` used to lex as Int(1) + Ident("e") + ...,
        // silently executing as RY(1.0). The exponent must survive.
        const SRC: &str = r#"
circuit Sci {
    qreg q[1]
    apply RY(1e-05) on q[0]
    apply RY(2.5e-4) on q[0]
    apply RY(1E+3) on q[0]
    apply RY(1e5) on q[0]
}
"#;
        let prog = parse_aria(SRC).expect("parse");
        let circ = prog.instantiate("Sci", &[]).expect("instantiate");
        let angles: Vec<f64> = circ
            .instructions
            .iter()
            .map(|i| i.gate.params[0].try_as_f64().expect("concrete"))
            .collect();
        assert_eq!(angles, vec![1e-05, 2.5e-4, 1e3, 1e5]);
    }

    #[test]
    fn overflowing_float_literal_is_rejected() {
        // f64 parse yields Ok(inf) on `1e999`; accepting it NaN-poisons the
        // state downstream, so the lexer must reject non-finite literals.
        let src = "circuit Bad {\n  qreg q[1]\n  apply RY(1e999) on q[0]\n}\n";
        let err = parse_aria(src).unwrap_err();
        assert!(err.contains("overflows"), "got: {err}");
    }

    #[test]
    fn parse_qft_unroll() {
        const SRC: &str = r#"
circuit QFT(n: int) {
    qreg q[n]
    repeat i from 0 to n - 1 {
        apply H on q[i]
        repeat j from i + 1 to n - 1 {
            apply CP(pi / (2.0 ^ (j - i))) on q[j], q[i]
        }
    }
    repeat i from 0 to (n / 2) - 1 {
        apply SWAP on q[i], q[n - 1 - i]
    }
}
"#;
        let prog = parse_aria(SRC).unwrap();
        let c3 = prog.instantiate("QFT", &[("n", 3)]).unwrap();
        // For n=3: H+CP+CP, H+CP, H, + 1 swap = 7 ops
        assert_eq!(c3.n_qubits(), 3);
        assert_eq!(
            c3.instructions
                .iter()
                .filter(|i| i.gate.kind == GateKind::H)
                .count(),
            3
        );
        assert_eq!(
            c3.instructions
                .iter()
                .filter(|i| i.gate.kind == GateKind::CP)
                .count(),
            3
        );
        assert_eq!(
            c3.instructions
                .iter()
                .filter(|i| i.gate.kind == GateKind::SWAP)
                .count(),
            1
        );
    }

    #[test]
    fn parse_when_compile_time() {
        const SRC: &str = r#"
circuit G(marked: int) {
    qreg q[3]
    when bit(marked, 0) == 0 { apply X on q[0] }
    when bit(marked, 1) == 0 { apply X on q[1] }
    when bit(marked, 2) == 0 { apply X on q[2] }
}
"#;
        let prog = parse_aria(SRC).unwrap();
        // marked=5 → bit0=1, bit1=0, bit2=1 → only q[1] flipped
        let c = prog.instantiate("G", &[("marked", 5)]).unwrap();
        assert_eq!(
            c.instructions
                .iter()
                .filter(|i| i.gate.kind == GateKind::X)
                .count(),
            1
        );
        assert_eq!(c.instructions[0].qubits[0], Qubit::new("q", 1));

        // marked=0 → all three X's fire
        let c0 = prog.instantiate("G", &[("marked", 0)]).unwrap();
        assert_eq!(c0.gate_count(), 3);
    }

    #[test]
    fn parse_symbolic_array() {
        const SRC: &str = r#"
circuit Ansatz(n_layers: int) {
    qreg q[2]
    let theta = symbolic[4 * n_layers]
    repeat layer from 0 to n_layers - 1 {
        apply RY(theta[4 * layer + 0]) on q[0]
        apply RY(theta[4 * layer + 1]) on q[1]
        apply RZ(theta[4 * layer + 2]) on q[0]
        apply RZ(theta[4 * layer + 3]) on q[1]
        apply CX on q[0], q[1]
    }
}
"#;
        let prog = parse_aria(SRC).unwrap();
        let c = prog.instantiate("Ansatz", &[("n_layers", 2)]).unwrap();
        // 2 layers × 5 ops each = 10
        assert_eq!(c.instructions.len(), 10);
        // Symbols must be named theta_0..theta_7
        let syms = c.free_symbols();
        assert!(syms.contains("theta_0"));
        assert!(syms.contains("theta_7"));
        assert_eq!(syms.len(), 8);
    }

    #[test]
    fn parse_oracle_inline() {
        const SRC: &str = r#"
circuit Main {
    qreg q[3]
    oracle Flip on q
}
circuit Flip {
    qreg q[3]
    apply X on q[0]
    apply X on q[1]
}
"#;
        let prog = parse_aria(SRC).unwrap();
        let c = prog.instantiate("Main", &[]).unwrap();
        assert_eq!(
            c.instructions
                .iter()
                .filter(|i| i.gate.kind == GateKind::X)
                .count(),
            2
        );
    }

    #[test]
    fn parse_measure_all() {
        const SRC: &str = r#"
circuit C {
    qreg q[3]
    creg c[3]
    measure q -> c
}
"#;
        let c = parse_aria_circuit(SRC, "C").unwrap();
        assert_eq!(
            c.instructions
                .iter()
                .filter(|i| i.gate.kind == GateKind::Measure)
                .count(),
            3
        );
    }

    #[test]
    fn parse_unknown_gate_is_error() {
        const SRC: &str = r#"
circuit C {
    qreg q[1]
    apply NOPE on q[0]
}
"#;
        assert!(parse_aria_circuit(SRC, "C").is_err());
    }

    #[test]
    fn parse_when_runtime_cond_sets_instruction_condition() {
        const SRC: &str = r#"
circuit C {
    qreg q[1]
    creg m[1]
    apply H on q[0]
    measure q[0] -> m[0]
    when m[0] == 1 { apply X on q[0] }
}
"#;
        let c = parse_aria_circuit(SRC, "C").unwrap();
        // The X is emitted unconditionally but carries a per-instruction
        // condition pointing at m[0] == 1 for downstream backends.
        let xs: Vec<_> = c
            .instructions
            .iter()
            .filter(|i| i.gate.kind == GateKind::X)
            .collect();
        assert_eq!(xs.len(), 1);
        assert_eq!(xs[0].condition.as_ref(), Some(&(Clbit::new("m", 0), 1)));
        assert_eq!(
            c.instructions
                .iter()
                .filter(|i| i.gate.kind == GateKind::X)
                .count(),
            1
        );
        assert!(c
            .annotations
            .iter()
            .any(|a| matches!(a, Annotation::Comment(s) if s.contains("when"))));
    }

    #[test]
    fn when_runtime_cond_that_cannot_lower_is_an_error() {
        // None of these guards has a `reg[i] == literal` form, so none can
        // be attached to an instruction as a classical condition. They used
        // to lower the body UNCONDITIONALLY with the guard demoted to a
        // comment annotation — a silently wrong circuit. They must error.
        for (guard, frag) in [
            ("m[0] == m[1]", "m[0] == m[1]"),
            ("m[0] + m[1] == 2", "(m[0] + m[1]) == 2"),
            ("m[0] != 1", "m[0] != 1"),
            ("m[0] >= 1", "m[0] >= 1"),
        ] {
            let src = format!(
                r#"
circuit C {{
    qreg q[2]
    creg m[2]
    measure q[0] -> m[0]
    measure q[1] -> m[1]
    when {guard} {{ apply X on q[0] }}
}}
"#
            );
            let err = parse_aria_circuit(&src, "C")
                .expect_err("unrepresentable runtime `when` guard must not lower");
            assert!(err.contains("when"), "error must name the construct: {err}");
            assert!(
                err.contains(frag),
                "error must echo the guard `{frag}`: {err}"
            );
        }
    }

    #[test]
    fn nested_runtime_when_is_an_error() {
        // The circuit model holds a single classical condition per
        // instruction. Nesting used to let the INNER guard claim the slot
        // and the outer guard was silently dropped: X fired whenever
        // m[1] == 1 regardless of m[0]. It must be rejected instead.
        const SRC: &str = r#"
circuit C {
    qreg q[3]
    creg m[2]
    measure q[0] -> m[0]
    measure q[1] -> m[1]
    when m[0] == 1 { when m[1] == 1 { apply X on q[2] } }
}
"#;
        let err = parse_aria_circuit(SRC, "C")
            .expect_err("nested runtime `when` must not lower");
        assert!(err.contains("nested"), "{err}");
        assert!(err.contains("m[0] == 1"), "{err}");
    }

    #[test]
    fn compile_time_when_inside_runtime_when_still_lowers() {
        // A compile-time inner guard leaves no per-instruction condition,
        // so the outer runtime guard stamps the body as usual.
        const SRC: &str = r#"
circuit C {
    qreg q[2]
    creg m[1]
    measure q[0] -> m[0]
    when m[0] == 1 { when 1 == 1 { apply X on q[1] } }
}
"#;
        let c = parse_aria_circuit(SRC, "C").unwrap();
        let x = c
            .instructions
            .iter()
            .find(|i| i.gate.kind == GateKind::X)
            .expect("X lowered");
        assert_eq!(x.condition.as_ref(), Some(&(Clbit::new("m", 0), 1)));
    }

    #[test]
    fn observable_template_lowers_simple_terms() {
        const SRC: &str = r#"
observable H {
    0.5 * Z(0)
  - 0.25 * X(1)
}
"#;
        let prog = parse_aria(SRC).expect("parse");
        let obs = prog.observable("H").expect("H exists");
        let omega = obs.to_omega_observable().expect("lower");
        assert_eq!(omega.terms.len(), 2);
        assert!((omega.terms[0].0 - 0.5).abs() < 1e-12);
        assert_eq!(omega.terms[0].1.len(), 1);
        assert!((omega.terms[1].0 - (-0.25)).abs() < 1e-12);
    }

    #[test]
    fn observable_template_resolves_let_bindings_and_identity() {
        // Mirrors examples/aria/vqe_ansatz.aria H2 shape.
        const SRC: &str = r#"
observable H2 {
    let g0 = -0.4804
    let g1 =  0.3435

    g0 * I
  + g1 * Z(0)
  + 0.5 * X(0) * X(1)
}
"#;
        let prog = parse_aria(SRC).expect("parse");
        let omega = prog
            .observable("H2")
            .expect("H2 exists")
            .to_omega_observable()
            .expect("lower");
        assert_eq!(omega.terms.len(), 3);
        // term 0: identity with coefficient g0 = -0.4804
        assert!(omega.terms[0].1.is_empty());
        assert!((omega.terms[0].0 - (-0.4804)).abs() < 1e-12);
        // term 2: X(0) * X(1) with coefficient 0.5
        assert_eq!(omega.terms[2].1.len(), 2);
        assert!((omega.terms[2].0 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn observable_template_supports_three_qubit_string() {
        const SRC: &str = r#"
observable T {
  0.125 * X(0) * Y(1) * Z(2)
}
"#;
        let prog = parse_aria(SRC).expect("parse");
        let omega = prog.observable("T").unwrap().to_omega_observable().unwrap();
        assert_eq!(omega.terms.len(), 1);
        assert_eq!(omega.terms[0].1.len(), 3);
    }

    #[test]
    fn observable_template_errors_on_unbound_ident() {
        const SRC: &str = r#"
observable B {
    alpha * Z(0)
}
"#;
        let prog = parse_aria(SRC).expect("parse");
        let err = prog
            .observable("B")
            .unwrap()
            .to_omega_observable()
            .expect_err("should fail: alpha is unbound");
        assert!(err.contains("alpha"));
    }

    #[test]
    fn examples_aria_files_parse() {
        // Smoke-test each file under examples/aria/.
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/aria/");
        for (file, name, params) in &[
            ("bell.aria", "Bell", &[][..]),
            ("grover3.aria", "Grover3", &[("marked", 5)][..]),
            ("qft.aria", "QFT", &[("n", 3)][..]),
            ("teleport.aria", "Teleport", &[][..]),
            ("vqe_ansatz.aria", "VQEAnsatz", &[("n_layers", 2)][..]),
            ("deutsch_jozsa.aria", "DeutschJozsa", &[("n", 3)][..]),
            (
                "bernstein_vazirani.aria",
                "BernsteinVazirani",
                &[("n", 4), ("a", 11)][..],
            ),
            ("simon.aria", "Simon", &[("n", 3)][..]),
            ("superdense.aria", "Superdense", &[("b0", 1), ("b1", 0)][..]),
            ("swap_test.aria", "SwapTest", &[("n", 2)][..]),
            ("qpe.aria", "QPE", &[("t", 3)][..]),
            ("qaoa_maxcut.aria", "QAOAMaxCut", &[("n", 4), ("p", 2)][..]),
            ("shor_ecdlp.aria", "ShorECDLP", &[("n", 3), ("t", 6)][..]),
            ("qml_classifier.aria", "QMLClassifier", &[("L", 3)][..]),
            ("quantum_kernel.aria", "QuantumKernelMap", &[("n", 3)][..]),
            ("qgan.aria", "QGANGenerator", &[("n", 3), ("L", 2)][..]),
        ] {
            let path = format!("{root}{file}");
            let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            let prog = parse_aria(&src).unwrap_or_else(|e| panic!("parse {file}: {e}"));
            let c = prog
                .instantiate(name, params)
                .unwrap_or_else(|e| panic!("instantiate {file}::{name}: {e}"));
            assert!(c.n_qubits() >= 1, "{file} produced zero qubits");
        }
    }

    /// Seeded LCG → uniform `[0,1)`, for the parser/lowering fuzz test.
    fn fuzz_lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (1u64 << 31) as f64
    }

    #[test]
    fn fuzz_random_aria_parses_lowers_and_is_well_formed() {
        // Property (item 21f): a randomly-generated but grammatically-valid
        // `.aria` circuit always parses + lowers without panicking, and every
        // lowered instruction references in-range qubits of the single `q`
        // register — the lowering never invents an out-of-bounds wire, and the
        // non-meta gate count equals what we emitted. 1-/2-/3-qubit gates,
        // parametric and not, are all exercised. Closes the "only hand-written
        // examples ever parsed" gap the brutal reviews flagged.
        const ONE_Q: &[&str] = &["X", "Y", "Z", "H", "S", "SDG", "T", "TDG", "SX", "I"];
        const ONE_Q_P: &[&str] = &["RX", "RY", "RZ", "P"];
        const TWO_Q: &[&str] = &["CX", "CY", "CZ", "SWAP"];
        const TWO_Q_P: &[&str] = &["CP", "RXX", "RYY", "RZZ", "RBS"];
        const THREE_Q: &[&str] = &["CCX", "CSWAP"];

        let mut s = 0xFEED_5EEDu64;
        for _ in 0..400 {
            let n = 1 + (fuzz_lcg(&mut s) * 4.0) as usize; // 1..=4 qubits
            let ngates = 1 + (fuzz_lcg(&mut s) * 12.0) as usize; // 1..=12 gates
            let mut body = String::new();
            let mut applied = 0usize;
            for _ in 0..ngates {
                let max_arity = n.min(3);
                let arity = 1 + (fuzz_lcg(&mut s) * max_arity as f64) as usize; // 1..=max_arity
                                                                                // distinct qubit indices via a partial Fisher–Yates shuffle
                let mut idx: Vec<usize> = (0..n).collect();
                for k in 0..arity {
                    let j = k + (fuzz_lcg(&mut s) * (n - k) as f64) as usize; // k..=n-1
                    idx.swap(k, j);
                }
                let qs: Vec<String> = idx[..arity].iter().map(|q| format!("q[{q}]")).collect();
                let parametric = fuzz_lcg(&mut s) < 0.5;
                let pick = |set: &[&'static str], s: &mut u64| -> &'static str {
                    set[(fuzz_lcg(s) * set.len() as f64) as usize]
                };
                let (name, has_param) = match (arity, parametric) {
                    (1, false) => (pick(ONE_Q, &mut s), false),
                    (1, true) => (pick(ONE_Q_P, &mut s), true),
                    (2, false) => (pick(TWO_Q, &mut s), false),
                    (2, true) => (pick(TWO_Q_P, &mut s), true),
                    _ => (pick(THREE_Q, &mut s), false),
                };
                if has_param {
                    let theta = fuzz_lcg(&mut s) * std::f64::consts::TAU;
                    body.push_str(&format!(
                        "    apply {name}({theta:.5}) on {}\n",
                        qs.join(", ")
                    ));
                } else {
                    body.push_str(&format!("    apply {name} on {}\n", qs.join(", ")));
                }
                applied += 1;
            }
            let src = format!("circuit Fuzz {{\n    qreg q[{n}]\n{body}}}\n");
            let circ = parse_aria_circuit(&src, "Fuzz")
                .unwrap_or_else(|e| panic!("fuzz parse failed: {e}\n--- src ---\n{src}"));
            assert_eq!(circ.n_qubits(), n, "qubit count mismatch\n{src}");
            assert_eq!(circ.gate_count(), applied, "gate count mismatch\n{src}");
            for inst in &circ.instructions {
                for q in &inst.qubits {
                    assert!(q.index < n, "out-of-range qubit {} in\n{src}", q.index);
                    assert_eq!(q.register, "q", "unexpected register {}", q.register);
                }
            }
        }
    }
}
