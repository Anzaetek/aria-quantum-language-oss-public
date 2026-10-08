// SPDX-License-Identifier: Apache-2.0
//! **`CX(a,b); Rz(θ) b; CX(a,b)` folds to one `Z⊗Z` rotation, and that is not
//! a speed-up — it is accuracy.**
//!
//! `rzz(θ)` reaches this backend already lowered: the parser decomposes it to
//! `cx; rz; cx` rather than adding a `GateKind`, deliberately (a new variant
//! costs every other backend an arm, and the repo has declined it). `rxx` and
//! `ryy` lower to the *same* triple wrapped in single-qubit basis changes, so
//! all three arrive the same way.
//!
//! The decomposition is exact as a unitary. It is not exact as a *propagation*
//! once truncation is on. `CX(a,b)` maps `X_b → X_a X_b`, so between the two
//! `cx` a term sits one qubit heavier than it does on either side, and a
//! `max_weight` cap applied between gates kills it there. The term is not
//! heavy; the lowering is. Conjugating by the single generator `Z_a Z_b`
//! never visits the excursion.
//!
//! This was found by the monoprop cross-check (`tests/monoprop_xcheck.rs`),
//! which measured us discarding *more* than an independent implementation at
//! an equal weight cap, for exactly this reason. So the tests here are written
//! to measure the claim rather than to exercise the code:
//!
//! 1. with truncation off, folding changes no answer (`f64` ULP, against the
//!    dense statevector backend on both arms);
//! 2. with a weight cap that bites, folding discards **strictly less** mass
//!    and lands **strictly closer** to the statevector — the numbers are
//!    printed;
//! 3. four near-misses do not fold, and still answer correctly.

use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_statevector::sim::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;
use omega_parser::lower::lower_qasm2;
use omega_parser::parse_qasm2;

/// Agreement with the dense reference. Loose enough to be about the algorithm
/// rather than about float noise.
const SV_TOL: f64 = 1e-9;

/// How far the two arms of the *exact* run may sit apart.
///
/// Deliberately not bit-identity, and the distinction is the interesting part.
/// The fold is the same conjugation by the same unitary, so it produces the
/// same TERMS — but the readout sums them out of a `HashMap`, whose iteration
/// order differs between two maps built by different insert sequences, and
/// float addition is not associative. Measured over 54 observables on the
/// 4-qubit ansatz below: bit-identical on 19-32 of them run to run, worst gap
/// 1.110e-16 — one ULP, never more. `tests/out_of_support_skip.rs` declines
/// bit-identity for the same reason and says so at length.
///
/// So this is a few ULP, which is a real bound on "the fold changed nothing",
/// rather than a property the change does not have and that would fail in CI
/// for a reason that is not a defect.
const ULP_TOL: f64 = 1e-14;

fn g(kind: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

fn obs(paulis: &[(u32, PauliOp)]) -> Observable {
    Observable {
        terms: vec![(1.0, paulis.to_vec())],
    }
}

/// Every weight-2 Pauli observable on `n` qubits, plus every weight-1 one.
///
/// Weight 2 is the load-bearing set: a `ZZ`-generator error shows up on terms
/// that anticommute with `Z_a Z_b`, and a wrongly-folded triple (control and
/// target swapped, say) agrees with the truth on plenty of single-qubit
/// observables while being wrong on the pairs.
fn all_short_observables(n: u32) -> Vec<Observable> {
    let mut out = Vec::new();
    for q in 0..n {
        for p in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
            out.push(obs(&[(q, p)]));
        }
    }
    for a in 0..n {
        for b in (a + 1)..n {
            for pa in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
                for pb in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
                    out.push(obs(&[(a, pa), (b, pb)]));
                }
            }
        }
    }
    out
}

/// A 4-qubit hardware-efficient ansatz written with `rzz`/`rxx`, lowered
/// through the real parser so the `cx·rz·cx` the peephole has to recognise is
/// the one production emits — not one re-typed here.
///
/// `scale` multiplies every angle, so the same shape gives several genuinely
/// different circuits without a second hand-written fixture.
fn hea(scale: f64) -> CircuitIR {
    let a = |x: f64| x * scale;
    let src = format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n\
         ry({}) q[0];\nry({}) q[1];\nry({}) q[2];\nry({}) q[3];\n\
         rzz({}) q[0], q[1];\nrzz({}) q[1], q[2];\nrzz({}) q[2], q[3];\n\
         rx({}) q[0];\nrx({}) q[1];\nrx({}) q[2];\nrx({}) q[3];\n\
         rzz({}) q[0], q[1];\nrzz({}) q[1], q[2];\nrzz({}) q[2], q[3];\n",
        a(0.4),
        a(0.7),
        a(1.1),
        a(0.3),
        a(0.9),
        a(0.8),
        a(0.6),
        a(0.5),
        a(0.35),
        a(0.8),
        a(0.25),
        a(0.7),
        a(0.5),
        a(0.45),
    );
    lower_qasm2(&parse_qasm2(&src).expect("parse")).expect("lower")
}

/// The same shape with `rxx`, which lowers to the identical triple under an
/// `h⊗h` conjugation — so the peephole has to fire inside a basis change it
/// knows nothing about.
fn hea_xx() -> CircuitIR {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\n\
        ry(0.4) q[0];\nry(0.7) q[1];\nry(1.1) q[2];\nry(0.3) q[3];\n\
        rxx(0.9) q[0], q[1];\nrzz(0.8) q[1], q[2];\nrxx(0.6) q[2], q[3];\n\
        rx(0.5) q[0];\nrx(0.35) q[1];\nrx(0.8) q[2];\nrx(0.25) q[3];\n\
        rzz(0.7) q[0], q[1];\nrxx(0.5) q[1], q[2];\nrzz(0.45) q[2], q[3];\n";
    lower_qasm2(&parse_qasm2(src).expect("parse")).expect("lower")
}

// ---------------------------------------------------------------------------
// 1. Exactness
// ---------------------------------------------------------------------------

/// **With nothing truncated, the fold must change no answer.**
///
/// Both arms against the dense statevector backend, over every weight-1 and
/// weight-2 Pauli observable on four qubits, at four angle scales and on both
/// the `rzz` and the `rxx` lowering — 1080 comparisons. An engine that agreed
/// with itself but not with a second implementation would pass a
/// fold-vs-no-fold test alone, so the reference is the point.
#[test]
fn folding_changes_no_exact_answer() {
    let params = ParameterBinding::default();
    let sv = StatevectorBackend::new();
    let folded = PauliPropBackend::new();
    let plain = PauliPropBackend::new().with_zz_folding(false);

    let mut checked = 0usize;
    let mut nonzero = 0usize;
    let mut worst_arms = 0.0f64;
    let mut worst_sv = 0.0f64;

    let circuits: Vec<(String, CircuitIR)> = [0.35, 0.8, 1.0, 1.7]
        .iter()
        .map(|s| (format!("hea(zz)*{s}"), hea(*s)))
        .chain(std::iter::once(("hea(xx)".to_string(), hea_xx())))
        .collect();

    for (label, circuit) in &circuits {
        for o in all_short_observables(circuit.num_qubits) {
            let want = sv.expectation(circuit, &params, &o).expect("statevector");
            let with = folded.expectation(circuit, &params, &o).expect("folded");
            let without = plain.expectation(circuit, &params, &o).expect("unfolded");

            worst_arms = worst_arms.max((with - without).abs());
            worst_sv = worst_sv
                .max((with - want).abs())
                .max((without - want).abs());
            checked += 1;
            if want.abs() > 1e-6 {
                nonzero += 1;
            }

            assert!(
                (with - without).abs() <= ULP_TOL,
                "{label}: folding moved an EXACT answer by {:.3e} — the fold is \
                 only sound because `CX·Rz(θ)·CX` IS `exp(-iθ/2 Z⊗Z)`, so any \
                 gap beyond float noise means the pattern matched something \
                 else. folded {with} vs unfolded {without}",
                (with - without).abs()
            );
            assert!(
                (with - want).abs() <= SV_TOL,
                "{label}: folded {with} vs statevector {want} (|Δ| {:.3e})",
                (with - want).abs()
            );
            assert!(
                (without - want).abs() <= SV_TOL,
                "{label}: unfolded {without} vs statevector {want}"
            );
        }
    }

    eprintln!(
        "exactness: {checked} comparisons ({nonzero} with a non-trivial value), \
         worst fold-vs-unfold {worst_arms:.3e}, worst vs statevector {worst_sv:.3e}"
    );
    assert!(checked >= 200, "only {checked} comparisons");
    assert!(
        nonzero >= checked / 4,
        "only {nonzero}/{checked} observables had any signal — a test that mostly \
         compares zeros proves nothing"
    );
}

// ---------------------------------------------------------------------------
// 2. The win
// ---------------------------------------------------------------------------

/// Run one observable at one weight cap on both arms, printing the numbers.
///
/// Returns `(dropped_mass, |value − statevector|, final_terms)` for
/// (unfolded, folded).
#[allow(clippy::type_complexity)]
fn compare_at_cap(
    circuit: &CircuitIR,
    o: &Observable,
    label: &str,
    cap: usize,
) -> ((f64, f64, usize), (f64, f64, usize)) {
    let params = ParameterBinding::default();
    let want = StatevectorBackend::new()
        .expectation(circuit, &params, o)
        .expect("statevector");
    let mut arms = Vec::new();
    for fold in [false, true] {
        // The dropped-mass gate is disabled on purpose: the unfolded arm at
        // cap 2 discards EVERYTHING, which is precisely the state the gate
        // exists to refuse. Refusing it here would hide the measurement this
        // test is for.
        let (v, cert) = PauliPropBackend::with_truncation(0.0, Some(cap))
            .with_zz_folding(fold)
            .with_max_dropped_mass(Some(f64::INFINITY))
            .expectation_with_certificate(circuit, &params, o)
            .unwrap_or_else(|e| panic!("{label} cap {cap} fold={fold}: {e}"));
        eprintln!(
            "  {label} cap {cap} fold={fold:<5} value {v:+.10} \
             (exact {want:+.10}, err {:.4e}) dropped_mass {:.4e} terms {}",
            (v - want).abs(),
            cert.dropped_mass,
            cert.final_terms
        );
        arms.push((cert.dropped_mass, (v - want).abs(), cert.final_terms));
    }
    (arms[0], arms[1])
}

/// **At a weight cap that bites, the fold discards strictly less and lands
/// strictly closer.**
///
/// The reason this work exists. `max_weight = 2` on the 4-qubit ansatz is the
/// case the monoprop cross-check flagged: the unfolded run loses every term
/// (`dropped_mass` 1.0, value 0.0 — an expectation carrying no information at
/// all) because each `ZZ` term has to pass through weight 3 inside the `cx·rz·
/// cx` it was lowered to.
#[test]
fn a_weight_cap_discards_strictly_less_with_the_fold() {
    let c = hea(1.0);
    let o = obs(&[(1, PauliOp::Z), (2, PauliOp::Z)]);
    eprintln!("ZZ(1,2) on the 4-qubit rzz ansatz:");

    for cap in [2usize, 3] {
        let (plain, folded) = compare_at_cap(&c, &o, "ZZ12", cap);
        assert!(
            folded.0 < plain.0,
            "cap {cap}: folding discarded {:.4e}, not less than the {:.4e} the \
             cx·rz·cx triple discarded — the peephole is not firing, or the cap \
             was never binding",
            folded.0,
            plain.0
        );
        assert!(
            folded.1 < plain.1,
            "cap {cap}: folding is {:.4e} from the statevector answer, not closer \
             than the triple's {:.4e}. Less discarded mass that does not buy a \
             better number would mean the certificate and the error have come \
             apart",
            folded.1,
            plain.1
        );
        // The certificate has to keep covering the error on BOTH arms — a
        // cheaper answer that broke its own bound would be worse than the
        // expensive one.
        assert!(
            folded.1 <= folded.0 + 1e-12 && plain.1 <= plain.0 + 1e-12,
            "cap {cap}: error left the dropped-mass certificate"
        );
    }

    // The headline, stated as an assertion so it cannot quietly stop being
    // true: at cap 2 the triple keeps nothing.
    let (plain, folded) = compare_at_cap(&c, &o, "ZZ12", 2);
    assert_eq!(
        plain.2, 0,
        "the unfolded triple used to lose EVERY term at cap 2 (dropped mass 1.0). \
         It now keeps {} — the lowering changed, and the headline number in this \
         test's docs and in the commit message is stale.",
        plain.2
    );
    assert!(
        folded.2 > 0,
        "folding must leave something at cap 2; it left {}",
        folded.2
    );
}

/// **The win is not one lucky observable — but `dropped_mass` alone is the
/// wrong way to see it, and this test records why.**
///
/// The obvious claim, "folding always discards less", is FALSE, and it fails
/// for an instructive reason rather than a shallow one. `dropped_mass`
/// accumulates over gates. A run that the cap annihilates at layer 2 has
/// nothing left to discard at layers 3-6, so it stops accruing and finishes
/// with a *small-looking* number attached to an answer that is pure zero.
/// Measured on the 66 short observables of the ansatz at `max_weight = 2`:
/// folding discards strictly less on 26 of them and strictly more on 37.
///
/// The same trap sits one level further in. Per-observable |error| is 30 better
/// / 30 worse, because an unfolded run returning exactly `0.0` scores well on
/// every observable whose true value is near zero — a stopped clock, certified
/// at `dropped_mass = 1.0`, which is the state this crate's dropped-mass gate
/// exists to refuse rather than to report.
///
/// So the two aggregate claims below are the ones that survive contact with the
/// data, and both are about information rather than about the budget:
///
/// * the default (gated) engine REFUSES strictly fewer of these runs, because
///   fewer of them end with no information in them at all;
/// * total absolute error against the statevector is strictly lower.
#[test]
fn the_win_is_information_not_a_smaller_budget() {
    let params = ParameterBinding::default();
    let sv = StatevectorBackend::new();
    let c = hea(1.0);

    let mut refused = [0usize; 2];
    let mut surviving = [0usize; 2];
    let mut total_err = [0.0f64; 2];
    let mut mass_better = 0usize;
    let mut mass_worse = 0usize;
    let mut total = 0usize;

    for o in all_short_observables(4) {
        let want = sv.expectation(&c, &params, &o).expect("statevector");
        total += 1;
        let mut mass = [0.0f64; 2];
        for (arm, fold) in [false, true].into_iter().enumerate() {
            // The DEFAULT dropped-mass ceiling, deliberately: whether a run is
            // refused is the measurement.
            match PauliPropBackend::with_truncation(0.0, Some(2))
                .with_zz_folding(fold)
                .expectation_with_certificate(&c, &params, &o)
            {
                Ok((v, cert)) => {
                    surviving[arm] += cert.final_terms;
                    total_err[arm] += (v - want).abs();
                }
                Err(_) => {
                    refused[arm] += 1;
                    // A refused run contributes its worst case, so the error
                    // comparison cannot be won by refusing more.
                    total_err[arm] += want.abs();
                }
            }
            // Budget, separately, with the gate off so every row is available.
            let (_, cert) = PauliPropBackend::with_truncation(0.0, Some(2))
                .with_zz_folding(fold)
                .with_max_dropped_mass(Some(f64::INFINITY))
                .expectation_with_certificate(&c, &params, &o)
                .expect("ungated run");
            mass[arm] = cert.dropped_mass;
        }
        if mass[1] < mass[0] - 1e-12 {
            mass_better += 1;
        } else if mass[1] > mass[0] + 1e-12 {
            mass_worse += 1;
        }
    }

    eprintln!(
        "cap 2, {total} observables: refused {} -> {}, surviving terms {} -> {}, \
         total |error| {:.4} -> {:.4} (mean {:.4e} -> {:.4e}); per-observable \
         dropped_mass better on {mass_better}, worse on {mass_worse}",
        refused[0],
        refused[1],
        surviving[0],
        surviving[1],
        total_err[0],
        total_err[1],
        total_err[0] / total as f64,
        total_err[1] / total as f64,
    );

    assert!(
        refused[1] < refused[0],
        "the default engine refused {} runs with folding and {} without. Folding \
         is supposed to leave information where the triple left none; if it \
         refuses as many, the peephole is not firing on this ansatz.",
        refused[1],
        refused[0]
    );
    assert!(
        total_err[1] < total_err[0],
        "total |error| {:.6} with folding is not below {:.6} without. Refusals \
         are charged at |exact| on both arms, so this cannot be won by refusing \
         more — a regression here means the fold has stopped being exact.",
        total_err[1],
        total_err[0]
    );
    assert!(
        surviving[1] > surviving[0],
        "folding kept {} terms in total against {}; it can only ever keep more, \
         since the transient weight excursion it removes is the only thing the \
         cap was killing that the rotation does not produce",
        surviving[1],
        surviving[0]
    );
    // The non-monotonicity above is an observation, not an aspiration — pinned
    // so that a future reader who "fixes" it by asserting mass_worse == 0 finds
    // out here rather than in CI.
    assert!(
        mass_worse > 0,
        "per-observable dropped_mass is no longer non-monotone (better \
         {mass_better}, worse {mass_worse}). That would be a real improvement, \
         but this test's docs explain the opposite at length and must be rewritten."
    );
}

// ---------------------------------------------------------------------------
// 3. Near-misses
// ---------------------------------------------------------------------------

/// Build a 3-qubit circuit with a generic non-Clifford preamble, then `body`.
fn with_preamble(body: &[GateOp]) -> CircuitIR {
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    c.add_op(g(GateKind::Ry, &[0], &[0.6]));
    c.add_op(g(GateKind::Rx, &[1], &[1.05]));
    c.add_op(g(GateKind::Ry, &[2], &[0.45]));
    for op in body {
        c.add_op(op.clone());
    }
    c.add_op(g(GateKind::Rx, &[1], &[0.33]));
    c
}

/// A near-miss must (a) still answer correctly and (b) leave the truncated run
/// *identical* to the unfolded one — if the peephole had fired, the term count
/// and the discarded mass would have moved.
fn assert_does_not_fold(label: &str, circuit: &CircuitIR) {
    let params = ParameterBinding::default();
    let sv = StatevectorBackend::new();
    let mut identical = 0usize;
    for o in all_short_observables(circuit.num_qubits) {
        let want = sv.expectation(circuit, &params, &o).expect("statevector");
        let got = PauliPropBackend::new()
            .expectation(circuit, &params, &o)
            .expect("exact");
        assert!(
            (got - want).abs() <= SV_TOL,
            "{label}: pauliprop {got} vs statevector {want} (|Δ| {:.3e}). A \
             near-miss that folded would be answering a DIFFERENT unitary, and \
             this is the observable that notices.",
            (got - want).abs()
        );

        let mut arms = Vec::new();
        for fold in [false, true] {
            let (_, cert) = PauliPropBackend::with_truncation(0.0, Some(1))
                .with_zz_folding(fold)
                .with_max_dropped_mass(Some(f64::INFINITY))
                .expectation_with_certificate(circuit, &params, &o)
                .expect("truncated");
            arms.push((cert.dropped_mass, cert.final_terms));
        }
        assert_eq!(
            arms[0].1, arms[1].1,
            "{label}: the weight-capped run kept {} terms with folding on and {} \
             with it off. Identical work is the signature of a pattern that did \
             not match; a difference means it did.",
            arms[1].1, arms[0].1
        );
        assert!(
            (arms[0].0 - arms[1].0).abs() <= 1e-12,
            "{label}: dropped mass moved"
        );
        identical += 1;
    }
    eprintln!("{label}: {identical} observables, no fold observed");
}

/// The `Rz` on the CX **control** commutes with both CXs — it is a plain
/// `Rz(θ) a`, not a `ZZ` rotation. Folding it would be a different unitary.
#[test]
fn rz_on_the_control_does_not_fold() {
    let c = with_preamble(&[
        g(GateKind::CX, &[0, 1], &[]),
        g(GateKind::Rz, &[0], &[0.83]),
        g(GateKind::CX, &[0, 1], &[]),
    ]);
    assert_does_not_fold("rz-on-control", &c);
}

/// A second `CX` on a different pair. The target matches, the control does not.
#[test]
fn a_mismatched_cx_pair_does_not_fold() {
    let c = with_preamble(&[
        g(GateKind::CX, &[0, 1], &[]),
        g(GateKind::Rz, &[1], &[0.83]),
        g(GateKind::CX, &[2, 1], &[]),
    ]);
    assert_does_not_fold("mismatched-pair", &c);
}

/// Operands reversed on the way out: `CX(0,1) … CX(1,0)` is not `RZZ`.
#[test]
fn reversed_operands_do_not_fold() {
    let c = with_preamble(&[
        g(GateKind::CX, &[0, 1], &[]),
        g(GateKind::Rz, &[1], &[0.83]),
        g(GateKind::CX, &[1, 0], &[]),
    ]);
    assert_does_not_fold("reversed-operands", &c);
}

/// Something in between. The three ops must be adjacent; an `Rx` on the
/// control does not commute with the `CX`, so the window is not an `RZZ`.
#[test]
fn an_intervening_op_does_not_fold() {
    let c = with_preamble(&[
        g(GateKind::CX, &[0, 1], &[]),
        g(GateKind::Rx, &[0], &[0.41]),
        g(GateKind::Rz, &[1], &[0.83]),
        g(GateKind::CX, &[0, 1], &[]),
    ]);
    assert_does_not_fold("intervening-op", &c);
}

/// **The near-miss harness can tell the difference.**
///
/// Four tests above assert "nothing changed". That is only evidence if the
/// same harness reports a change on the pattern that *does* match — otherwise
/// a peephole deleted from the engine would pass all four.
#[test]
fn the_near_miss_harness_detects_a_real_fold() {
    let params = ParameterBinding::default();
    let c = with_preamble(&[
        g(GateKind::CX, &[0, 1], &[]),
        g(GateKind::Rz, &[1], &[0.83]),
        g(GateKind::CX, &[0, 1], &[]),
    ]);
    // `Z1` at cap 1, chosen from the sweep in `assert_does_not_fold`'s own
    // observable set: the unfolded triple loses every term (dropped 1.2701)
    // where the fold keeps two (dropped 0.2391). A weight-2 observable is the
    // WRONG probe here — at cap 1 it dies on the seed, before any gate runs,
    // and both arms then agree at zero for a reason that has nothing to do
    // with the peephole.
    let o = obs(&[(1, PauliOp::Z)]);
    let mut arms = Vec::new();
    for fold in [false, true] {
        let (_, cert) = PauliPropBackend::with_truncation(0.0, Some(1))
            .with_zz_folding(fold)
            .with_max_dropped_mass(Some(f64::INFINITY))
            .expectation_with_certificate(&c, &params, &o)
            .expect("truncated");
        arms.push((cert.dropped_mass, cert.final_terms));
    }
    eprintln!(
        "positive control, cap 1: unfolded dropped {:.4e} ({} terms), \
         folded dropped {:.4e} ({} terms)",
        arms[0].0, arms[0].1, arms[1].0, arms[1].1
    );
    assert!(
        arms[1].0 < arms[0].0 - 1e-12,
        "the matching triple must show a difference under a cap, or the four \
         near-miss tests are asserting nothing: unfolded {:.4e}, folded {:.4e}",
        arms[0].0,
        arms[1].0
    );
}

/// A conditioned gate inside the window must not be folded past its refusal.
///
/// The engine refuses classically-conditioned gates outright. If the peephole
/// matched on gate kind alone it would swallow the `Rz`'s guard and answer a
/// circuit nobody asked for — the exact failure mode `tests/conditional_refusal.rs`
/// was written for.
#[test]
fn a_conditioned_rz_inside_the_window_is_still_refused() {
    let mut rz = g(GateKind::Rz, &[1], &[0.83]);
    // `(start_bit, num_bits, expected)` — a 1-bit creg guard.
    rz.condition = Some((0, 1, 1));
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[0], &[]));
    c.add_op(g(GateKind::CX, &[0, 1], &[]));
    c.add_op(rz);
    c.add_op(g(GateKind::CX, &[0, 1], &[]));
    let err = PauliPropBackend::new()
        .expectation(&c, &ParameterBinding::default(), &obs(&[(1, PauliOp::Z)]))
        .expect_err("a conditioned gate must be refused, not folded away");
    let msg = err.to_string();
    assert!(
        msg.contains("classically-conditioned"),
        "refusal should name the guard, got: {msg}"
    );
}
