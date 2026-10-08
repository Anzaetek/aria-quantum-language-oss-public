// SPDX-License-Identifier: Apache-2.0
//! PLAN-FERMIONIC **F1** — the text spelling of `FermionicOp`.
//!
//! Three legs, as the plan states them:
//!
//! * **(i)** parse → `jordan_wigner()` → dense Fock matrix lives in
//!   `fermion_jw.rs` (`parsed_text_matches_the_oracle_…`), next to the
//!   oracle it uses, together with the deliberate wrong-string case.
//! * **(ii)** round trip: `parse(display(op)) == op`, exactly — not to a
//!   tolerance — and equal as operators under normal ordering. Here.
//! * **(iii)** the cross-library pin: the same strings through
//!   `openfermion.FermionOperator`, term-for-term coefficient agreement in
//!   both directions (our text → their parser; their `str()` → our parser).
//!   Skips out loud without a Python that imports `openfermion`: the FQE
//!   venv (`make -C crates/omega-bridges/python fqe-venv`) or
//!   `OMEGA_OPENFERMION_PYTHON=/path/to/python`.
//!
//! Also here: the refusals. Malformed text names the offending token, and
//! `Observable::parse` cannot read this spelling (so a parser that "worked"
//! by delegating to it would fail leg (i) outright).
use num_complex::Complex64;
use omega_core::executor::Observable;
use omega_core::fermion::{FermionicOp, Ladder};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn parse(s: &str) -> FermionicOp {
    FermionicOp::parse(s).unwrap_or_else(|e| panic!("parse({s:?}): {e}"))
}

fn refuse(s: &str) -> String {
    match FermionicOp::parse(s) {
        Ok(op) => panic!("parse({s:?}) accepted as {op:?}; it must refuse"),
        Err(e) => e.to_string(),
    }
}

/// The operators every round-trip test walks: real, complex, scalar,
/// duplicate products, non-adjacent Z strings, a length-4 product.
fn corpus() -> Vec<(&'static str, FermionicOp)> {
    vec![
        ("hopping", FermionicOp::hopping(1, 3, 0.5)),
        (
            "hubbard-like",
            FermionicOp::hopping(0, 3, -0.7)
                + FermionicOp::number(1).scale(c(0.3, 0.0))
                + FermionicOp::interaction(1, 2, 1.1),
        ),
        (
            "complex current",
            FermionicOp::raise(0)
                .mul(&FermionicOp::lower(1))
                .scale(c(1.0, 2.0))
                + FermionicOp::raise(1)
                    .mul(&FermionicOp::lower(0))
                    .scale(c(1.0, -2.0)),
        ),
        ("pure imaginary", FermionicOp::raise(2).scale(c(0.0, -1.0))),
        ("scalar", FermionicOp::identity().scale(c(2.0, 0.0))),
        (
            "scalar plus number",
            FermionicOp::identity().scale(c(-1.5, 0.0)) + FermionicOp::number(0),
        ),
        (
            "duplicate product",
            FermionicOp::number(0) + FermionicOp::number(0),
        ),
        (
            "tiny coefficient",
            FermionicOp::number(3).scale(c(1e-3, 0.0)),
        ),
        (
            "length four, unordered",
            FermionicOp::term(
                c(0.25, 0.0),
                vec![
                    Ladder::lower(2),
                    Ladder::raise(0),
                    Ladder::lower(0),
                    Ladder::raise(3),
                ],
            ),
        ),
        ("zero", FermionicOp::zero()),
    ]
}

// ---------------------------------------------------------------------------
// The spelling itself, pinned.
// ---------------------------------------------------------------------------

#[test]
fn display_is_openfermions_string_form() {
    assert_eq!(
        FermionicOp::hopping(1, 3, 0.5).to_string(),
        "0.5 [1^ 3] +\n0.5 [3^ 1]"
    );
    assert_eq!(
        FermionicOp::identity().scale(c(2.0, 0.0)).to_string(),
        "2.0 []"
    );
    assert_eq!(
        FermionicOp::number(0).scale(c(-1.0, 0.0)).to_string(),
        "-1.0 [0^ 0]"
    );
    assert_eq!(
        FermionicOp::raise(0)
            .mul(&FermionicOp::lower(1))
            .scale(c(1.0, 2.0))
            .to_string(),
        "(1.0+2.0j) [0^ 1]"
    );
    assert_eq!(
        FermionicOp::raise(0).scale(c(0.5, -0.25)).to_string(),
        "(0.5-0.25j) [0^]"
    );
    assert_eq!(
        FermionicOp::raise(2).scale(c(0.0, -1.0)).to_string(),
        "-1.0j [2^]"
    );
    assert_eq!(FermionicOp::zero().to_string(), "0");
    assert_eq!(Ladder::raise(7).to_string(), "7^");
    assert_eq!(Ladder::lower(7).to_string(), "7");
}

#[test]
fn parse_reads_the_documented_forms() {
    let hop = FermionicOp::hopping(1, 3, 0.5);
    // OpenFermion's own joiner, with and without the newline.
    assert_eq!(parse("0.5 [1^ 3] +\n0.5 [3^ 1]"), hop);
    assert_eq!(parse("0.5 [1^ 3] + 0.5 [3^ 1]"), hop);
    assert_eq!(parse("  0.5[1^ 3]+0.5[3^   1]  "), hop);
    // Missing coefficient is 1; `+` alone is 1; `-` alone is −1.
    assert_eq!(
        parse("[1^ 3]"),
        FermionicOp::term(c(1.0, 0.0), vec![Ladder::raise(1), Ladder::lower(3)])
    );
    assert_eq!(parse("- [0^]"), FermionicOp::raise(0).scale(c(-1.0, 0.0)));
    assert_eq!(parse("+ [0^]"), FermionicOp::raise(0));
    // A bare `-` between terms, and OpenFermion's `+\n-c` form.
    let want = FermionicOp::number(0) + FermionicOp::number(1).scale(c(-0.5, 0.0));
    assert_eq!(parse("1 [0^ 0] - 0.5 [1^ 1]"), want);
    assert_eq!(parse("1.0 [0^ 0] +\n-0.5 [1^ 1]"), want);
    // Exponents are not term separators.
    assert_eq!(
        parse("1e-3 [2^ 2]"),
        FermionicOp::number(2).scale(c(1e-3, 0.0))
    );
    assert_eq!(
        parse("1E+3 [2^ 2] + 2 []"),
        FermionicOp::number(2).scale(c(1e3, 0.0)) + FermionicOp::identity().scale(c(2.0, 0.0))
    );
    // Python complex spellings, including what `str(-1j)` prints.
    assert_eq!(
        parse("2j [0^ 1]"),
        FermionicOp::raise(0)
            .mul(&FermionicOp::lower(1))
            .scale(c(0.0, 2.0))
    );
    assert_eq!(
        parse("(1-2j) [1^ 0]"),
        FermionicOp::raise(1)
            .mul(&FermionicOp::lower(0))
            .scale(c(1.0, -2.0))
    );
    assert_eq!(
        parse("(-0-1j) [2^]"),
        FermionicOp::raise(2).scale(c(0.0, -1.0))
    );
    assert_eq!(
        parse("-(1+2j) [2^]"),
        FermionicOp::raise(2).scale(c(-1.0, -2.0))
    );
    assert_eq!(
        parse("j [2^] + -j [2]"),
        FermionicOp::raise(2).scale(c(0.0, 1.0)) + FermionicOp::lower(2).scale(c(0.0, -1.0))
    );
    assert_eq!(
        parse("1e-3+2e-5j [0]"),
        FermionicOp::lower(0).scale(c(1e-3, 2e-5))
    );
    // Scalars and zero.
    assert_eq!(parse("3.5 []"), FermionicOp::identity().scale(c(3.5, 0.0)));
    assert_eq!(parse("0"), FermionicOp::zero());
    // Term order and duplicates are kept as written — canonical form is
    // `normal_ordered`'s job, not the parser's.
    let dup = parse("1 [0^ 0] + 1 [0^ 0]");
    assert_eq!(dup.terms.len(), 2);
    assert_eq!(dup.simplified(), FermionicOp::number(0).scale(c(2.0, 0.0)));
    // `FromStr` is the same parser.
    assert_eq!(
        "0.5 [1^ 3]".parse::<FermionicOp>().unwrap(),
        parse("0.5 [1^ 3]")
    );
    // Example files carry a `#` header. It is not a coefficient.
    assert_eq!(
        parse("# nuclear repulsion on the identity\n0.713753990544915 [] +\n-1.25 [0^ 0]\n"),
        parse("0.713753990544915 [] + -1.25 [0^ 0]")
    );
}

// ---------------------------------------------------------------------------
// Leg (ii): round trip.
// ---------------------------------------------------------------------------

#[test]
fn parse_of_display_is_the_identity_exactly() {
    for (name, op) in corpus() {
        let text = op.to_string();
        let back = parse(&text);
        // Exact: same terms, same order, bit-identical coefficients. `{:?}`
        // on f64 is the shortest round-tripping decimal, so no tolerance.
        assert_eq!(
            back, op,
            "{name}: parse(display) changed the term list\n{text}"
        );
        // And as operators, which is what the plan's (ii) states.
        assert_eq!(
            back.normal_ordered(),
            op.normal_ordered(),
            "{name}: not the same operator after normal ordering"
        );
        // Canonical text round-trips to canonical text.
        let canon = op.normal_ordered();
        assert_eq!(parse(&canon.to_string()), canon, "{name}: canonical form");
    }
}

// ---------------------------------------------------------------------------
// Refusals name the token.
// ---------------------------------------------------------------------------

#[test]
fn malformed_text_is_refused_naming_the_token() {
    let e = refuse("0.5 [1^ x]");
    assert!(e.contains("'x'"), "{e}");
    let e = refuse("0.5 [1^ 3^^]");
    assert!(e.contains("'3^^'"), "{e}");
    let e = refuse("0.5 [^]");
    assert!(e.contains("'^'"), "{e}");
    let e = refuse("abc [1^]");
    assert!(e.contains("'abc'"), "{e}");
    let e = refuse("0.5*[1^ 3]");
    assert!(e.contains("'0.5*'"), "{e}");
    let e = refuse("1j2 [1^]");
    assert!(e.contains("'1j2'"), "{e}");
    let e = refuse("0.5 1^ 3");
    assert!(e.contains("'[...]'"), "{e}");
    let e = refuse("0.5 [1^ 3");
    assert!(e.contains("unclosed"), "{e}");
    let e = refuse("0.5 1^ 3]");
    assert!(e.contains("'[...]'"), "{e}");
    let e = refuse("(1+2j [1^]");
    assert!(e.contains("'(1+2j'") && e.contains("unbalanced"), "{e}");
    let e = refuse("1 [1^] extra");
    assert!(e.contains("'extra'"), "{e}");
    let e = refuse("1 [1^] + + 2 [2^]");
    assert!(e.contains("empty term"), "{e}");
    let e = refuse("1 [1^] +");
    assert!(e.contains("empty term"), "{e}");
    let e = refuse("1 [1^] 2 [2^]");
    assert!(e.contains("missing '+'") && e.contains("'2 [2^]'"), "{e}");
    let e = refuse("1 [1^] - -2 [2^]");
    assert!(e.contains("'- -2'"), "{e}");
    let e = refuse("");
    assert!(e.contains("empty"), "{e}");
    let e = refuse("   \n ");
    assert!(e.contains("empty"), "{e}");
    let e = refuse("1 [99999999999^]");
    assert!(e.contains("'99999999999'"), "{e}");
    // The Pauli spelling is a different algebra and is refused, not guessed.
    let e = refuse("0.5*Z0Z1");
    assert!(e.contains("'[...]'"), "{e}");
    // And the other way round — a parser that delegated to
    // `Observable::parse` could not have produced any of the above.
    assert!(Observable::parse("1 [1^ 3] + 1 [3^ 1]").is_err());
}

#[test]
fn non_hermitian_text_is_refused_where_it_becomes_an_observable() {
    // Parsing is not the place: `a†_0` is a fine operator.
    let op = parse("1 [0^]");
    let err = op.jordan_wigner().unwrap_err().to_string();
    assert!(err.contains("not Hermitian"), "{err}");
    // The fix the message names, spelled as text.
    assert!(parse("1 [0^] + 1 [0]").jordan_wigner().is_ok());
    // A complex coefficient on a Hermitian product is likewise refused.
    let err = parse("(0+1j) [0^ 0]")
        .jordan_wigner()
        .unwrap_err()
        .to_string();
    assert!(err.contains("not Hermitian"), "{err}");
    // …whereas `i(a†_0 a_1 − a†_1 a_0)` is Hermitian and maps.
    assert!(parse("1j [0^ 1] + -1j [1^ 0]").jordan_wigner().is_ok());
}

// ---------------------------------------------------------------------------
// Leg (iii): the cross-library pin.
// ---------------------------------------------------------------------------

fn openfermion_python() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OMEGA_OPENFERMION_PYTHON") {
        return Some(PathBuf::from(p));
    }
    let venv = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("omega-bridges")
        .join("python")
        .join(".venv-fqe")
        .join("bin")
        .join("python");
    venv.exists().then_some(venv)
}

const OF_SCRIPT: &str = r#"
import json, sys
from openfermion import FermionOperator
out = []
for s in json.load(sys.stdin):
    op = FermionOperator(s)
    terms = [[[[int(m), int(d)] for (m, d) in key], complex(v).real, complex(v).imag]
             for key, v in op.terms.items()]
    out.append({"terms": terms, "str": str(op)})
json.dump(out, sys.stdout)
"#;

type Terms = BTreeMap<Vec<(u32, bool)>, Complex64>;

fn ours(op: &FermionicOp) -> Terms {
    op.simplified()
        .terms
        .into_iter()
        .map(|(c, p)| (p.into_iter().map(|l| (l.mode, l.dagger)).collect(), c))
        .collect()
}

fn theirs(v: &Value) -> Terms {
    v["terms"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| {
            let coeff = c(t[1].as_f64().unwrap(), t[2].as_f64().unwrap());
            (coeff.norm() >= 1e-12).then(|| {
                let prod = t[0]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|l| (l[0].as_u64().unwrap() as u32, l[1].as_u64().unwrap() == 1))
                    .collect();
                (prod, coeff)
            })
        })
        .collect()
}

fn assert_same_terms(what: &str, a: &Terms, b: &Terms) {
    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>(),
        "{what}: different products\nours:   {a:?}\ntheirs: {b:?}"
    );
    for (k, ca) in a {
        let cb = b[k];
        assert!(
            (ca - cb).norm() < 1e-12,
            "{what}: coefficient on {k:?}: ours {ca}, theirs {cb}"
        );
    }
}

#[test]
fn openfermion_reads_our_text_and_we_read_its() {
    let Some(python) = openfermion_python() else {
        eprintln!(
            "no python with openfermion — skipping the cross-library pin. Build the FQE \
             venv (`make -C crates/omega-bridges/python fqe-venv`) or set \
             OMEGA_OPENFERMION_PYTHON."
        );
        return;
    };
    // Our `Display` for the whole corpus (except the zero operator: OpenFermion
    // prints it as `0` but reads `"0"` as the action string `a_0`, so that
    // string is not exchangeable and is pinned Rust-side only), plus hand
    // spellings in the forms OpenFermion documents.
    let mut cases: Vec<String> = corpus()
        .into_iter()
        .filter(|(name, _)| *name != "zero")
        .map(|(_, op)| op.to_string())
        .collect();
    cases.extend(
        [
            "1 [1^ 3] + 1 [3^ 1]",
            "[0^ 0]",
            "1.0 [0^ 0] +\n-0.5 [1^ 1]",
            "2j [0^ 1] + (1-2j) [1^ 0]",
            "1e-3 [2^ 2 1^ 1]",
            "3.5 []",
            "(-0-1j) [2^]",
        ]
        .map(String::from),
    );
    let mut child = Command::new(&python)
        .args(["-c", OF_SCRIPT])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn {}: {e}", python.display()));
    child
        .stdin
        .take()
        .unwrap()
        .write_all(json!(cases).to_string().as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "openfermion runner failed under {}:\n{}",
        python.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let results: Vec<Value> = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(results.len(), cases.len());
    for (s, r) in cases.iter().zip(&results) {
        // Direction 1: the same text, both parsers.
        let mine = ours(&parse(s));
        assert_same_terms(
            &format!("openfermion.FermionOperator({s:?})"),
            &mine,
            &theirs(r),
        );
        // Direction 2: OpenFermion's own rendering, our parser.
        let of_str = r["str"].as_str().unwrap();
        let back = ours(&parse(of_str));
        assert_same_terms(
            &format!("our parse of str(FermionOperator({s:?})) = {of_str:?}"),
            &back,
            &mine,
        );
        eprintln!("pinned: {s:?} ↔ {of_str:?}");
    }
}
