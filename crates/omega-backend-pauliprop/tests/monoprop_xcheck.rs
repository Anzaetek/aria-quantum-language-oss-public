// SPDX-License-Identifier: Apache-2.0
//! Differential cross-check against **monoprop** (`Algorithmiq/monoprop`),
//! an independent Pauli-propagation implementation from the authors of
//! arXiv:2503.18939. Same algorithm family as this crate, none of the code.
//!
//! The fixture is `tools/pp_cross_check/monoprop_fixture.jsonl`, generated
//! by `monoprop_ref.py`. Each case carries the circuit as **structured ops**;
//! this side renders them to OpenQASM 2 and lowers through `omega-parser`,
//! the Python side turns them into monoprop `ExpGate`s. Neither reads the
//! other's rendering, so the r⟨P⟩(θ) sign convention is exercised
//! independently on both sides rather than copied.
//!
//! Two kinds of case, with different claims:
//!
//! * **exact** (`cutoff == n_qubits`, no `lower_atol`): both engines are
//!   exact, so the values must agree to float noise. This is the load-bearing
//!   comparison.
//! * **truncated**: both engines drop terms, and their truncation rules are
//!   not guaranteed identical (monoprop's own benchmark page shows two codes
//!   in the same family retaining 109.4 M vs 79.4 M terms on one run). The
//!   only claim this side can make is its own certificate: `|ours − exact| ≤
//!   dropped_mass`. So the test checks monoprop's truncated value lies within
//!   our certificate of *our exact* value, and prints the raw gap so a
//!   tightening can be justified from evidence rather than assumed.
//!
//! # What this harness found, and what was done about it
//!
//! Found while building it, 2026-09-10: at an equal weight cap we discarded
//! MORE than monoprop. This crate has no native `rzz`/`rxx` gate and has
//! declined to add one — the parser lowers them to `cx·rz·cx` (resp. under an
//! `h` or `rx(±π/2)` conjugation). In the Heisenberg picture the first `cx`
//! takes a term through a transient weight-`w+1` string before the second `cx`
//! brings it back, so a weight cap applied between gates killed terms on the
//! strength of a weight that monoprop's native `ExpGate(ZZ)` never produces.
//! Measured then: `ZZ12` on the 4-qubit HEA at cap 2 left us **0 terms**
//! (dropped mass 1.0) where monoprop returned −0.239.
//!
//! Closed, 2026-09-23, by a peephole in `sim.rs` rather than by a new
//! `GateKind`: `propagate` recognises an adjacent `CX(a,b); Rz(θ) b; CX(a,b)`
//! and conjugates by the single generator `Z_a Z_b` instead, which is the same
//! unitary and skips the excursion. `tests/zz_peephole.rs` holds the pattern
//! match, the near-misses and the exactness. Measured on this fixture, before
//! → after:
//!
//! ```text
//!   hea4_l2_zz12_w3         err 2.776e-2 → 9.826e-3   gap to monoprop 1.794e-2 → 0.000e0
//!   hea4_l2_zz12_w3_atol01  err 3.090e-2 → 1.582e-2   gap to monoprop 2.263e-2 → 7.550e-3
//!   hea4_l2_x2_atol005      unchanged
//!   hea4_l2_zz12_atol01     unchanged
//!   hea4_l2_z0_atol005      unchanged
//! ```
//!
//! So on the one case that caps weight and nothing else, the two engines now
//! return the **same number to every printed digit** — the gap did not narrow,
//! it closed, and the `lower_atol.is_none()` branch of
//! `truncated_cases_lie_within_our_certificate` pins that at `EXACT_TOL`.
//! The mixed case narrowed 3x, and the three coefficient-only (`atol`) cases
//! did not move at all. That last row is the check on the explanation rather
//! than a null result: `cx` maps one Pauli to one Pauli with a unit-magnitude
//! factor, so it cannot change any coefficient's magnitude and a coefficient
//! cutoff was never exposed to the artefact. Only the weight axis was, and
//! only the weight axis moved.

use omega_backend_pauliprop::sim::PauliPropBackend;
use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;
use omega_parser::lower::lower_qasm2;
use omega_parser::parse_qasm2;
use serde_json::Value;

const EXACT_TOL: f64 = 1e-10;

struct Case {
    name: String,
    n_qubits: u32,
    ops: Vec<Value>,
    observable: Observable,
    cutoff: usize,
    lower_atol: Option<f64>,
    expectation: f64,
}

fn pauli(c: char) -> PauliOp {
    match c {
        'X' => PauliOp::X,
        'Y' => PauliOp::Y,
        'Z' => PauliOp::Z,
        other => panic!("fixture Pauli letter {other:?}"),
    }
}

fn load() -> (String, Vec<Case>) {
    let raw = include_str!("../../../tools/pp_cross_check/monoprop_fixture.jsonl");
    let mut meta = String::new();
    let mut cases = Vec::new();
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).expect("fixture line is JSON");
        if let Some(e) = v.get("error") {
            panic!("fixture records a generator error: {e}");
        }
        if let Some(m) = v.get("meta") {
            meta = m.to_string();
            continue;
        }
        let terms = v["observable"]
            .as_array()
            .expect("observable array")
            .iter()
            .map(|t| {
                let coeff = t[0].as_f64().expect("coeff");
                let letters = t[1].as_str().expect("letters");
                let qubits = t[2].as_array().expect("qubits");
                assert_eq!(letters.len(), qubits.len(), "letters/qubits length");
                let paulis = letters
                    .chars()
                    .zip(qubits)
                    .map(|(c, q)| (q.as_u64().expect("qubit") as u32, pauli(c)))
                    .collect();
                (coeff, paulis)
            })
            .collect();
        cases.push(Case {
            name: v["case"].as_str().expect("case").to_string(),
            n_qubits: v["n_qubits"].as_u64().expect("n_qubits") as u32,
            ops: v["ops"].as_array().expect("ops").clone(),
            observable: Observable { terms },
            cutoff: v["cutoff"].as_u64().expect("cutoff") as usize,
            lower_atol: v["lower_atol"].as_f64(),
            expectation: v["expectation"].as_f64().expect("expectation"),
        });
    }
    assert!(!cases.is_empty(), "fixture has no cases");
    (meta, cases)
}

/// Render the structured ops to OpenQASM 2. `{}` on an `f64` is Rust's
/// shortest round-trip form and never uses an exponent, so the parser sees
/// exactly the angle the fixture carries.
fn to_qasm(n: u32, ops: &[Value]) -> String {
    let mut s = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\n");
    for op in ops {
        let name = op["op"].as_str().expect("op name");
        let theta = op["theta"].as_f64().expect("theta");
        let qs: Vec<String> = op["q"]
            .as_array()
            .expect("q")
            .iter()
            .map(|q| format!("q[{}]", q.as_u64().expect("qubit")))
            .collect();
        assert!(
            matches!(name, "rx" | "ry" | "rz" | "rxx" | "rzz"),
            "op {name:?} is outside the subset this harness renders"
        );
        s.push_str(&format!("{name}({theta}) {};\n", qs.join(", ")));
    }
    s
}

fn is_exact(c: &Case) -> bool {
    c.cutoff >= c.n_qubits as usize && c.lower_atol.is_none()
}

#[test]
fn monoprop_fixture_is_sane() {
    let (meta, cases) = load();
    assert!(
        meta.contains("monoprop"),
        "meta line names the oracle: {meta}"
    );
    let exact = cases.iter().filter(|c| is_exact(c)).count();
    assert!(exact >= 4, "need several exact cases, have {exact}");
    assert!(cases.len() > exact, "need at least one truncated case");
    for c in &cases {
        c.observable
            .validate_qubits(c.n_qubits)
            .unwrap_or_else(|e| panic!("{}: observable out of range: {e}", c.name));
    }
}

#[test]
fn exact_cases_agree_with_monoprop() {
    let (_, cases) = load();
    let mut worst = (0.0_f64, String::new());
    for c in cases.iter().filter(|c| is_exact(c)) {
        let prog = parse_qasm2(&to_qasm(c.n_qubits, &c.ops))
            .unwrap_or_else(|e| panic!("{}: parse: {e}", c.name));
        let ir = lower_qasm2(&prog).unwrap_or_else(|e| panic!("{}: lower: {e}", c.name));
        let (ours, cert) = PauliPropBackend::new()
            .expectation_with_certificate(&ir, &ParameterBinding::new(), &c.observable)
            .unwrap_or_else(|e| panic!("{}: pauliprop: {e}", c.name));
        assert_eq!(cert.dropped_mass, 0.0, "{}: exact run dropped mass", c.name);
        let gap = (ours - c.expectation).abs();
        eprintln!(
            "{:<24} ours {ours:+.15} monoprop {:+.15} |Δ| {gap:.3e}",
            c.name, c.expectation
        );
        if gap > worst.0 {
            worst = (gap, c.name.clone());
        }
        assert!(
            gap <= EXACT_TOL,
            "{}: ours {ours} vs monoprop {} (|Δ| = {gap:.3e} > {EXACT_TOL:e})",
            c.name,
            c.expectation
        );
    }
    eprintln!("worst exact |Δ| = {:.3e} ({})", worst.0, worst.1);
}

#[test]
fn truncated_cases_lie_within_our_certificate() {
    let (_, cases) = load();
    for c in cases.iter().filter(|c| !is_exact(c)) {
        let prog = parse_qasm2(&to_qasm(c.n_qubits, &c.ops))
            .unwrap_or_else(|e| panic!("{}: parse: {e}", c.name));
        let ir = lower_qasm2(&prog).unwrap_or_else(|e| panic!("{}: lower: {e}", c.name));
        let binding = ParameterBinding::new();
        let (exact, _) = PauliPropBackend::new()
            .expectation_with_certificate(&ir, &binding, &c.observable)
            .unwrap_or_else(|e| panic!("{}: exact pauliprop: {e}", c.name));
        let max_weight = (c.cutoff < c.n_qubits as usize).then_some(c.cutoff);
        let (ours, cert) =
            PauliPropBackend::with_truncation(c.lower_atol.unwrap_or(0.0), max_weight)
                .expectation_with_certificate(&ir, &binding, &c.observable)
                .unwrap_or_else(|e| panic!("{}: truncated pauliprop: {e}", c.name));
        let ours_err = (ours - exact).abs();
        let theirs_err = (c.expectation - exact).abs();
        let gap = (ours - c.expectation).abs();
        eprintln!(
            "{:<24} exact {exact:+.12} ours {ours:+.12} (err {ours_err:.3e}) monoprop {:+.12} \
             (err {theirs_err:.3e}) ours-vs-monoprop {gap:.3e} dropped_mass {:.3e} terms {}",
            c.name, c.expectation, cert.dropped_mass, cert.final_terms
        );
        // Our own guarantee, restated here so a certificate regression shows
        // up next to the oracle numbers rather than in a separate test.
        assert!(
            ours_err <= cert.dropped_mass + 1e-12,
            "{}: our truncated value breaks our own certificate: err {ours_err:.3e} > {:.3e}",
            c.name,
            cert.dropped_mass
        );
        // A weight cap and nothing else: after the `cx·rz·cx` peephole the two
        // engines apply the SAME rule to the same terms, so they must now
        // agree to float noise rather than merely lie inside a certificate.
        // This is the regression pin for the whole exercise — it went from
        // 1.794e-2 apart to 0.000e0 — and it is deliberately the strictest
        // assertion in the file, because a peephole that stops firing shows up
        // here first and as a number rather than as a performance shrug.
        if c.lower_atol.is_none() {
            assert!(
                gap <= EXACT_TOL,
                "{}: a weight-capped run must now match monoprop exactly (the \
                 `cx·rz·cx` fold removed the transient the cap was killing), \
                 but ours {ours} vs monoprop {} is {gap:.3e} apart. Either the \
                 peephole stopped recognising this circuit's lowering, or a \
                 second truncation rule has diverged.",
                c.name,
                c.expectation
            );
        }
        // The comparison this test exists for. If monoprop's truncated value
        // is farther from the exact answer than our certificate allows for
        // *our* run at the same knobs, then either the knob mapping is wrong
        // or the two truncation rules differ by more than the certificate —
        // both worth stopping on.
        assert!(
            theirs_err <= cert.dropped_mass + 1e-12,
            "{}: monoprop {} is {theirs_err:.3e} from exact {exact}, outside our \
             dropped-mass certificate {:.3e} at the same knobs (cutoff {}, atol {:?})",
            c.name,
            c.expectation,
            cert.dropped_mass,
            c.cutoff,
            c.lower_atol
        );
    }
}
