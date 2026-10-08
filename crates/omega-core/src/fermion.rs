// SPDX-License-Identifier: Apache-2.0
//! Fermionic operators and their Jordan–Wigner image on qubits — §3c **T1**,
//! the layer above the gates the backends already execute.
//!
//! Before this module the repo had no fermionic surface at all:
//! `Observable::parse` speaks `X/Y/Z` + index and nothing else. This adds
//!
//! * [`FermionicOp`] — sums of ladder-operator products with complex
//!   coefficients, with product, adjoint and **normal ordering** (the
//!   canonical form that makes two spellings of one operator compare equal);
//! * [`FermionicOp::jordan_wigner`] — the map onto [`Observable`], refusing
//!   non-Hermitian input rather than silently dropping imaginary parts;
//! * [`FermionicOp::parse`] / [`Display`](std::fmt::Display) — the text
//!   spelling (PLAN-FERMIONIC F1), in **OpenFermion's
//!   `FermionOperator` convention**: `0.5 [1^ 3] +\n0.5 [3^ 1]`, `^` marking
//!   creation, a scalar spelled `c []`. Adopted rather than invented so it
//!   can be pinned against the library that owns it
//!   (`tests/fermion_spelling.rs`);
//! * [`cphase`] / [`givens`] — constructors for the two number-preserving
//!   gates, emitting the `CU3(0,0,θ)` and `Rbs` the backends already run.
//!
//! # Conventions, stated once and pinned by tests
//!
//! * Mode `j` **is** qubit `j`; `|0⟩` empty, `|1⟩` occupied.
//! * `a_j = Z_0 ⋯ Z_{j−1} · (X_j + iY_j)/2`, `a†_j = Z_0 ⋯ Z_{j−1} · (X_j − iY_j)/2`
//!   — the Z string sits on the **lower** indices, as in OpenFermion and
//!   Qiskit Nature's `JordanWignerMapper`. Consequently `n_j = (I − Z_j)/2`.
//! * A product `[l₀, l₁, …]` is the operator `l₀ · l₁ · ⋯` in the usual
//!   left-to-right algebraic order.
//! * Normal order is creation operators first, each group in **descending**
//!   mode index (OpenFermion's convention).
//!
//! Every one of these is checked against an explicitly built Fock-space
//! matrix in `tests/fermion_jw.rs`, exhaustively over all products up to
//! length three at four modes. That test is the §3c.4 leg-1 oracle; it tests
//! the *algebra*, and it is the only reference here that does not share code
//! with the thing it checks.

use std::collections::BTreeMap;

use num_complex::Complex64;
use smallvec::smallvec;

use crate::circuit::{GateKind, GateOp, ParamExpr, Qubit};
use crate::error::{OmegaError, Result};
use crate::executor::{Observable, PauliOp};

/// One ladder operator: `a_p` when `dagger == false`, `a†_p` when `true`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ladder {
    pub mode: u32,
    pub dagger: bool,
}

impl Ladder {
    /// `a†_p`.
    pub fn raise(mode: u32) -> Self {
        Self { mode, dagger: true }
    }
    /// `a_p`.
    pub fn lower(mode: u32) -> Self {
        Self {
            mode,
            dagger: false,
        }
    }
}

/// Coefficients with modulus below this are dropped when terms are merged.
pub const COEFF_EPS: f64 = 1e-12;

/// A sum of ladder-operator products with complex coefficients.
///
/// The representation is deliberately unnormalised: `terms` may contain the
/// same product twice or products that cancel. [`simplified`] merges,
/// [`normal_ordered`] canonicalises. Equality of two `FermionicOp`s as
/// *operators* is `a.normal_ordered() == b.normal_ordered()`, not `a == b`.
///
/// [`simplified`]: FermionicOp::simplified
/// [`normal_ordered`]: FermionicOp::normal_ordered
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FermionicOp {
    /// `(coefficient, product)`, the product read left to right.
    pub terms: Vec<(Complex64, Vec<Ladder>)>,
}

fn re(x: f64) -> Complex64 {
    Complex64::new(x, 0.0)
}

impl std::ops::Add for FermionicOp {
    type Output = Self;

    /// `self + other`: the concatenated term list (no simplification).
    fn add(mut self, other: Self) -> Self {
        self.terms.extend(other.terms);
        self
    }
}

/// A Jordan–Wigner image: Pauli terms as `(coefficient, [(qubit, Pauli)])`.
///
/// Named because `jordan_wigner_terms_counted` returns it beside a count, and
/// the tuple tips `clippy::type_complexity` over its threshold.
pub type PauliTerms = Vec<(Complex64, Vec<(u32, PauliOp)>)>;

impl FermionicOp {
    /// The zero operator.
    pub fn zero() -> Self {
        Self::default()
    }

    /// The identity (empty product, coefficient 1).
    pub fn identity() -> Self {
        Self::term(re(1.0), vec![])
    }

    /// A single term.
    pub fn term(coeff: Complex64, product: Vec<Ladder>) -> Self {
        Self {
            terms: vec![(coeff, product)],
        }
    }

    /// `a†_p`.
    pub fn raise(p: u32) -> Self {
        Self::term(re(1.0), vec![Ladder::raise(p)])
    }

    /// `a_p`.
    pub fn lower(p: u32) -> Self {
        Self::term(re(1.0), vec![Ladder::lower(p)])
    }

    /// Number operator `n_p = a†_p a_p`.
    pub fn number(p: u32) -> Self {
        Self::term(re(1.0), vec![Ladder::raise(p), Ladder::lower(p)])
    }

    /// Hopping term `t · (a†_p a_q + a†_q a_p)`. Hermitian for real `t`.
    /// With `p == q` this is `2t · n_p`, which is correct but probably not
    /// what was meant.
    pub fn hopping(p: u32, q: u32, t: f64) -> Self {
        Self {
            terms: vec![
                (re(t), vec![Ladder::raise(p), Ladder::lower(q)]),
                (re(t), vec![Ladder::raise(q), Ladder::lower(p)]),
            ],
        }
    }

    /// Density–density interaction `u · n_p n_q`.
    pub fn interaction(p: u32, q: u32, u: f64) -> Self {
        Self::term(
            re(u),
            vec![
                Ladder::raise(p),
                Ladder::lower(p),
                Ladder::raise(q),
                Ladder::lower(q),
            ],
        )
    }

    /// One more than the highest mode index named, or 0 for a scalar.
    pub fn num_modes(&self) -> u32 {
        self.terms
            .iter()
            .flat_map(|(_, prod)| prod.iter().map(|l| l.mode + 1))
            .max()
            .unwrap_or(0)
    }

    /// `c · self`.
    pub fn scale(mut self, c: Complex64) -> Self {
        for (coeff, _) in &mut self.terms {
            *coeff *= c;
        }
        self
    }

    /// Operator product `self · other` (every term of `self` left of every
    /// term of `other`).
    pub fn mul(&self, other: &Self) -> Self {
        let mut terms = Vec::with_capacity(self.terms.len() * other.terms.len());
        for (ca, pa) in &self.terms {
            for (cb, pb) in &other.terms {
                let mut prod = pa.clone();
                prod.extend_from_slice(pb);
                terms.push((ca * cb, prod));
            }
        }
        Self { terms }
    }

    /// Hermitian conjugate: reverse each product, swap raise/lower, conjugate
    /// the coefficient.
    pub fn dagger(&self) -> Self {
        Self {
            terms: self
                .terms
                .iter()
                .map(|(c, prod)| {
                    let rev = prod
                        .iter()
                        .rev()
                        .map(|l| Ladder {
                            mode: l.mode,
                            dagger: !l.dagger,
                        })
                        .collect();
                    (c.conj(), rev)
                })
                .collect(),
        }
    }

    /// Merge identical products and drop coefficients below [`COEFF_EPS`].
    /// Does **not** reorder anything; see [`normal_ordered`](Self::normal_ordered).
    pub fn simplified(&self) -> Self {
        let mut acc: BTreeMap<Vec<Ladder>, Complex64> = BTreeMap::new();
        for (c, prod) in &self.terms {
            *acc.entry(prod.clone()).or_insert_with(|| re(0.0)) += c;
        }
        Self {
            terms: acc
                .into_iter()
                .filter(|(_, c)| c.norm() >= COEFF_EPS)
                .map(|(p, c)| (c, p))
                .collect(),
        }
    }

    /// `true` if `a` may stand immediately before `b` in normal order.
    /// Equal operators are never in order: `a_p a_p = 0`.
    fn ordered(a: Ladder, b: Ladder) -> bool {
        if a == b {
            return false;
        }
        if a.dagger != b.dagger {
            return a.dagger;
        }
        a.mode > b.mode
    }

    /// `true` if every product is already in normal order.
    pub fn is_normal_ordered(&self) -> bool {
        self.terms
            .iter()
            .all(|(_, p)| p.windows(2).all(|w| Self::ordered(w[0], w[1])))
    }

    /// Rewrite into normal order using `{a_p, a†_q} = δ_pq`, `{a_p, a_q} = 0`,
    /// then merge. The result is the canonical form: two operators are equal
    /// iff their normal-ordered forms are equal term for term.
    pub fn normal_ordered(&self) -> Self {
        let mut done: Vec<(Complex64, Vec<Ladder>)> = Vec::new();
        let mut work = self.terms.clone();
        while let Some((c, ops)) = work.pop() {
            if c.norm() < COEFF_EPS {
                continue;
            }
            let violation =
                (0..ops.len().saturating_sub(1)).find(|&i| !Self::ordered(ops[i], ops[i + 1]));
            let Some(i) = violation else {
                done.push((c, ops));
                continue;
            };
            let (a, b) = (ops[i], ops[i + 1]);
            if a == b {
                // Nilpotent: a_p a_p = a†_p a†_p = 0.
                continue;
            }
            // Anticommute: X Y = −Y X (+ δ when it is a_p a†_p).
            let mut swapped = ops.clone();
            swapped.swap(i, i + 1);
            work.push((-c, swapped));
            if !a.dagger && b.dagger && a.mode == b.mode {
                let mut contracted = ops.clone();
                contracted.drain(i..i + 2);
                work.push((c, contracted));
            }
        }
        Self { terms: done }.simplified()
    }

    /// Jordan–Wigner image as a **complex** Pauli sum, with no Hermiticity
    /// requirement. Strings are sorted by qubit and identities omitted.
    pub fn jordan_wigner_terms(&self) -> Vec<(Complex64, Vec<(u32, PauliOp)>)> {
        self.jordan_wigner_terms_counted().0
    }

    /// [`jordan_wigner_terms`](Self::jordan_wigner_terms) plus the number of
    /// intermediate `(coefficient, Pauli string)` partial products the
    /// expansion built before merging — `Σ_terms (2^{k+1} − 2)` for `k`-ladder
    /// products, *observed* rather than computed, so a caller comparing this
    /// detour against a direct Majorana seed (majoranaprop, PLAN-FERMIONIC
    /// F3 test iii) is counting work that actually happened.
    pub fn jordan_wigner_terms_counted(&self) -> (PauliTerms, usize) {
        type PString = BTreeMap<u32, PauliOp>;
        let mut acc: BTreeMap<Vec<(u32, PauliOp)>, Complex64> = BTreeMap::new();
        let mut expanded = 0usize;
        for (coeff, prod) in &self.terms {
            let mut partial: Vec<(Complex64, PString)> = vec![(*coeff, PString::new())];
            for l in prod {
                let mut next = Vec::with_capacity(partial.len() * 2);
                for (c, s) in &partial {
                    for (bc, bs) in ladder_strings(*l) {
                        let (ph, out) = string_mul(s, &bs);
                        next.push((c * bc * ph, out));
                    }
                }
                expanded += next.len();
                partial = next;
            }
            for (c, s) in partial {
                let key: Vec<(u32, PauliOp)> = s.into_iter().collect();
                *acc.entry(key).or_insert_with(|| re(0.0)) += c;
            }
        }
        let terms = acc
            .into_iter()
            .filter(|(_, c)| c.norm() >= COEFF_EPS)
            .map(|(s, c)| (c, s))
            .collect();
        (terms, expanded)
    }

    /// Jordan–Wigner image as an [`Observable`].
    ///
    /// `Observable` carries real coefficients, so this is defined only for
    /// Hermitian operators and **refuses** otherwise. Dropping the imaginary
    /// parts would return a different operator with no signal — exactly the
    /// class of failure §3b.2 documents — so the refusal names the offending
    /// coefficient and the fix.
    pub fn jordan_wigner(&self) -> Result<Observable> {
        let mut terms = Vec::new();
        for (c, s) in self.jordan_wigner_terms() {
            if c.im.abs() > 1e-10 {
                let string = if s.is_empty() {
                    "I".to_string()
                } else {
                    s.iter()
                        .map(|(q, p)| format!("{p:?}{q}"))
                        .collect::<Vec<_>>()
                        .join("")
                };
                return Err(OmegaError::Unsupported(format!(
                    "the Jordan–Wigner image of this fermionic operator has a \
                     complex Pauli coefficient ({c} on {string}), so the operator \
                     is not Hermitian and cannot be an `Observable`, whose \
                     coefficients are real. Symmetrise it (`op + op.dagger()`) \
                     if that is what was meant, or use `jordan_wigner_terms` for \
                     the complex sum."
                )));
            }
            terms.push((c.re, s));
        }
        Ok(Observable { terms })
    }
}

// ---------------------------------------------------------------------------
// Text spelling — PLAN-FERMIONIC F1.
//
// The grammar is OpenFermion's `FermionOperator` string form, verbatim:
//
//     op    := '0' | term ( ('+' | '-') term )*
//     term  := [ coeff ] '[' ( mode '^'? )* ']'
//     coeff := python number: `1`, `-0.5`, `1e-3`, `2j`, `(1+2j)`
//
// `str(FermionOperator)` joins terms with " +\n" and prints a negative
// coefficient as `-0.5 [..]` after that `+`; both are accepted, as is a bare
// `-` between terms. Whitespace is free everywhere except inside a number.
// `Display` emits exactly OpenFermion's form so the two libraries can be fed
// each other's output (`tests/fermion_spelling.rs`, the cross-library pin).
//
// Refusals name the offending token; nothing is guessed or dropped.
// ---------------------------------------------------------------------------

impl std::fmt::Display for Ladder {
    /// `3^` for `a†_3`, `3` for `a_3`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.mode, if self.dagger { "^" } else { "" })
    }
}

/// A coefficient in Python's number spelling: `0.5`, `-1.0`, `2.0j`,
/// `(1.0+2.0j)`. Real parts use Rust's shortest round-trip `f64` form, so
/// `parse(display(op)) == op` exactly, not to a tolerance.
fn fmt_coeff(c: Complex64) -> String {
    if c.im == 0.0 {
        format!("{:?}", c.re)
    } else if c.re == 0.0 {
        format!("{:?}j", c.im)
    } else if c.im.is_sign_negative() {
        format!("({:?}-{:?}j)", c.re, -c.im)
    } else {
        format!("({:?}+{:?}j)", c.re, c.im)
    }
}

impl std::fmt::Display for FermionicOp {
    /// OpenFermion's `str(FermionOperator)`: terms as `coeff [p^ q ...]`
    /// joined by `" +\n"`, the empty product as `coeff []`, the zero operator
    /// as `0`. Term order is preserved; call
    /// [`normal_ordered`](Self::normal_ordered) first for a canonical text.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.terms.is_empty() {
            return f.write_str("0");
        }
        for (i, (c, prod)) in self.terms.iter().enumerate() {
            if i > 0 {
                f.write_str(" +\n")?;
            }
            let actions = prod
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            write!(f, "{} [{}]", fmt_coeff(*c), actions)?;
        }
        Ok(())
    }
}

impl std::str::FromStr for FermionicOp {
    type Err = OmegaError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

fn spelling_err(msg: String) -> OmegaError {
    OmegaError::Parse(format!("fermionic operator: {msg}"))
}

/// The text before a term's `[`: the joiner (`+` or `-`, mandatory after
/// the first term) and a coefficient in Python's number spelling — a real,
/// a pure imaginary `bj`, or `a±bj`, optionally in parentheses.
///
/// Terms are delimited by their brackets, as in OpenFermion's own reader, so
/// an unparenthesised `1e-3+2e-5j` is one coefficient, not two terms.
fn parse_coeff(raw: &str, term: &str, first: bool) -> Result<Complex64> {
    let bad = |what: &str| {
        spelling_err(format!(
            "coefficient '{}' in term '{}': {what} (expected a Python number such \
             as `0.5`, `-1e-3`, `2j` or `(1+2j)`, followed by `[...]`)",
            raw.trim(),
            term.trim()
        ))
    };
    let mut t: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    // The joiner. `+` is consumed; `-` stays as the coefficient's sign.
    // OpenFermion's own output is `+\n-0.5 [..]` for a negative term.
    if first {
        if let Some(rest) = t.strip_prefix('+') {
            t = rest.to_string();
        }
    } else {
        match t.chars().next() {
            Some('+') => t.remove(0),
            Some('-') => '-',
            _ => {
                return Err(spelling_err(format!(
                    "missing '+' or '-' before term '{}' (terms are joined by '+')",
                    term.trim()
                )))
            }
        };
    }
    if t.starts_with('+') {
        return Err(spelling_err(format!(
            "empty term (doubled sign) before '{}'",
            term.trim()
        )));
    }
    // A sign before parentheses negates the whole complex number. A sign
    // inside them belongs to the real part only: Python prints `-1j` as
    // `(-0-1j)`, and `-0-1j` is `(-0) + (-1)j`.
    let mut sign = 1.0;
    if t.starts_with("-(") {
        sign = -1.0;
        t.remove(0);
    }
    match (t.starts_with('('), t.ends_with(')')) {
        (true, true) => t = t[1..t.len() - 1].to_string(),
        (false, false) => {}
        _ => return Err(bad("unbalanced parentheses")),
    }
    if t.is_empty() {
        return Ok(re(sign));
    }
    if t == "-" {
        return Ok(re(-sign));
    }
    let real = |x: &str| -> Result<f64> {
        match x {
            "" | "+" => Ok(1.0),
            "-" => Ok(-1.0),
            _ => x.parse::<f64>().map_err(|e| bad(&e.to_string())),
        }
    };
    let value = if let Some(body) = t.strip_suffix('j') {
        // Last sign that is neither the leading sign nor an exponent's.
        let b = body.as_bytes();
        let split = (1..b.len())
            .rev()
            .find(|&i| matches!(b[i], b'+' | b'-') && !matches!(b[i - 1], b'e' | b'E'));
        match split {
            Some(i) => Complex64::new(real(&body[..i])?, real(&body[i..])?),
            None => Complex64::new(0.0, real(body)?),
        }
    } else if t.contains('j') {
        return Err(bad("'j' must be the last character of an imaginary part"));
    } else {
        re(t.parse::<f64>().map_err(|e| bad(&e.to_string()))?)
    };
    Ok(value * sign)
}

impl FermionicOp {
    /// Parse the OpenFermion spelling: `0.5 [1^ 3] + 0.5 [3^ 1]`.
    ///
    /// * `^` after a mode index is creation; a bare index is annihilation.
    /// * A missing coefficient is `1`; `[]` is the identity, so `2 []` is the
    ///   scalar 2; the string `0` is the zero operator.
    /// * Coefficients are Python numbers, complex included (`(1+2j) [0^ 1]`).
    /// * Terms are kept as written — no merging, no reordering. Use
    ///   [`normal_ordered`](Self::normal_ordered) for the canonical form and
    ///   [`jordan_wigner`](Self::jordan_wigner) for the qubit observable,
    ///   which is where a non-Hermitian spelling is refused.
    ///
    /// Malformed text is refused with the offending token named. This does
    /// **not** accept `Observable::parse`'s `0.5*Z0Z1` form, and vice versa:
    /// the two spellings name different algebras.
    pub fn parse(s: &str) -> Result<Self> {
        // A `#` line is a header, not an operator. The example files state
        // geometry and mode order there; OpenFermion's own `str()` has no
        // comments, and stripping these lines is what keeps `cat file`
        // readable without changing a coefficient.
        let stripped: String = s
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let trimmed = stripped.trim();
        if trimmed.is_empty() {
            return Err(spelling_err(
                "empty text (spell the zero operator as `0`, a scalar as `c []`)".into(),
            ));
        }
        if trimmed == "0" {
            return Ok(Self::zero());
        }
        let mut terms = Vec::new();
        let mut rest = trimmed;
        let mut first = true;
        loop {
            let Some(open) = rest.find('[') else {
                let tail = rest.trim();
                if tail.is_empty() {
                    break;
                }
                if first {
                    return Err(spelling_err(format!(
                        "term '{tail}' has no '[...]' ladder product; a scalar is spelled \
                         `c []`, a number operator `1 [p^ p]`"
                    )));
                }
                if tail.chars().all(|ch| matches!(ch, '+' | '-')) {
                    return Err(spelling_err(format!(
                        "empty term (trailing '{tail}') in '{trimmed}'"
                    )));
                }
                return Err(spelling_err(format!(
                    "unexpected '{tail}' after ']' in '{trimmed}' (terms are joined by '+')"
                )));
            };
            let Some(close_rel) = rest[open..].find(']') else {
                return Err(spelling_err(format!("unclosed '[' in '{}'", rest.trim())));
            };
            let close = open + close_rel;
            let t = rest[..=close].trim();
            let coeff = parse_coeff(&rest[..open], t, first)?;
            let mut prod = Vec::new();
            for tok in rest[open + 1..close].split_whitespace() {
                let (digits, dagger) = match tok.strip_suffix('^') {
                    Some(d) => (d, true),
                    None => (tok, false),
                };
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(spelling_err(format!(
                        "bad ladder token '{tok}' in term '{t}': expected a mode index \
                         like `3` (a_3) or `3^` (a†_3)"
                    )));
                }
                let mode: u32 = digits.parse().map_err(|e| {
                    spelling_err(format!("mode index '{digits}' in term '{t}': {e}"))
                })?;
                prod.push(Ladder { mode, dagger });
            }
            terms.push((coeff, prod));
            rest = &rest[close + 1..];
            first = false;
        }
        Ok(Self { terms })
    }
}

/// The two Pauli strings of one ladder operator with their coefficients:
/// `Z_{<j} X_j / 2` and `± i · Z_{<j} Y_j / 2` (`+` for `a`, `−` for `a†`).
fn ladder_strings(l: Ladder) -> [(Complex64, BTreeMap<u32, PauliOp>); 2] {
    let mut zs: BTreeMap<u32, PauliOp> = (0..l.mode).map(|k| (k, PauliOp::Z)).collect();
    let mut x = zs.clone();
    x.insert(l.mode, PauliOp::X);
    zs.insert(l.mode, PauliOp::Y);
    let y_coeff = if l.dagger {
        Complex64::new(0.0, -0.5)
    } else {
        Complex64::new(0.0, 0.5)
    };
    [(re(0.5), x), (y_coeff, zs)]
}

/// Single-qubit Pauli product `a · b = phase · out`.
fn pauli_mul(a: PauliOp, b: PauliOp) -> (Complex64, PauliOp) {
    use PauliOp::*;
    let one = re(1.0);
    let i = Complex64::new(0.0, 1.0);
    match (a, b) {
        (I, p) | (p, I) => (one, p),
        (X, X) | (Y, Y) | (Z, Z) => (one, I),
        (X, Y) => (i, Z),
        (Y, Z) => (i, X),
        (Z, X) => (i, Y),
        (Y, X) => (-i, Z),
        (Z, Y) => (-i, X),
        (X, Z) => (-i, Y),
    }
}

/// Product of two Pauli strings `s · t`, returning the phase and the string
/// with identities removed.
fn string_mul(
    s: &BTreeMap<u32, PauliOp>,
    t: &BTreeMap<u32, PauliOp>,
) -> (Complex64, BTreeMap<u32, PauliOp>) {
    let mut out = s.clone();
    let mut phase = re(1.0);
    for (&q, &tb) in t {
        match out.get(&q).copied() {
            None => {
                out.insert(q, tb);
            }
            Some(sa) => {
                let (ph, r) = pauli_mul(sa, tb);
                phase *= ph;
                if r == PauliOp::I {
                    out.remove(&q);
                } else {
                    out.insert(q, r);
                }
            }
        }
    }
    (phase, out)
}

/// `exp(iθ n_p n_q) = diag(1, 1, 1, e^{iθ})` — the controlled phase.
///
/// Not a new `GateKind`: `omega-parser` already lowers `cp`/`cu1`/`cphase` to
/// `CU3(0, 0, θ)`, which every backend executes and the statevector's
/// diagonal fast path recognises. This is the constructor that lowering
/// path never exposed. Symmetric in its two qubits.
pub fn cphase(p: u32, q: u32, theta: f64) -> GateOp {
    cphase_expr(p, q, ParamExpr::Concrete(theta))
}

/// [`cphase`] with a symbolic angle.
pub fn cphase_expr(p: u32, q: u32, theta: ParamExpr) -> GateOp {
    GateOp {
        gate: GateKind::CU3,
        qubits: smallvec![Qubit(p), Qubit(q)],
        params: smallvec![ParamExpr::Concrete(0.0), ParamExpr::Concrete(0.0), theta],
        classical_bit: None,
        condition: None,
    }
}

/// Fermionic Givens rotation `exp(θ (a†_p a_q − a†_q a_p))` between two
/// **adjacent** modes, emitted as `Rbs(θ)` on qubits `(p, q)`.
///
/// The identity `Rbs(θ)_{p,q} = exp(θ(a†_p a_q − a†_q a_p))` holds for
/// `|p − q| = 1` in either order (the Z string of the higher mode is exactly
/// the lower mode's `Z`, which the `(X ∓ iY)` factors absorb — derived and
/// pinned numerically in `tests/fermion_jw.rs`). For non-adjacent modes the
/// fermionic rotation carries a `Z` string over every mode strictly between,
/// which `Rbs` does not have, so emitting it anyway would run and return a
/// wrong number. **Refused** until the routing (Z-string decomposition or
/// swap network — §3c.0a) is specified and verified.
pub fn givens(p: u32, q: u32, theta: f64) -> Result<GateOp> {
    givens_expr(p, q, ParamExpr::Concrete(theta))
}

/// [`givens`] with a symbolic angle.
pub fn givens_expr(p: u32, q: u32, theta: ParamExpr) -> Result<GateOp> {
    if p.abs_diff(q) != 1 {
        return Err(OmegaError::Unsupported(format!(
            "givens({p}, {q}): a fermionic Givens rotation between non-adjacent \
             Jordan–Wigner modes needs a Z string over modes {}..={}, which the \
             adjacent-mode `Rbs` gate does not carry. Emitting `Rbs` here would run \
             and be wrong. Route through adjacent swaps or wait for the Z-string \
             decomposition (PLAN-OPEN-20260825 §3c.0a).",
            p.min(q) + 1,
            p.max(q).saturating_sub(1)
        )));
    }
    Ok(GateOp {
        gate: GateKind::Rbs,
        qubits: smallvec![Qubit(p), Qubit(q)],
        params: smallvec![theta],
        classical_bit: None,
        condition: None,
    })
}
