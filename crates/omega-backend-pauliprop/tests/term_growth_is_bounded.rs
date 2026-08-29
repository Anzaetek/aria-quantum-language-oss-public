//! **The Pauli sum has a ceiling, and crossing it is a refusal.**
//!
//! Every other truncation knob here (`coeff_min`, `max_weight`, `max_freq`) is
//! an approximation that discards amplitude and reports `dropped_mass`. Those
//! are correctly off by default: an exact engine should stay exact until asked.
//!
//! `max_terms` is different in kind. It is not a worse answer, it is a refusal,
//! and it is on by default because the failure it prevents is not an inaccurate
//! number — it is the machine.
//!
//! # Why coefficient truncation cannot save you here
//!
//! Each non-Clifford rotation splits every anticommuting term in two, weighted
//! `cos θ` and `sin θ`. For most angles one child is small and `--truncate`
//! prunes it. **A T gate is θ = π/4, where `cos θ = sin θ = 0.7071`** — the two
//! children are exactly equal, so no coefficient threshold can drop either
//! without dropping both. k such rotations means 2^k terms with nothing to
//! prune.
//!
//! # Why this was not needed before, and is now
//!
//! It was bounded by accident: the gates that generate T ladders (`CCX` and
//! friends) are refused as unsupported, so the explosion was unreachable. That
//! is an accident of coverage, not a design, and any work widening the gate set
//! removes it. The bound has to be explicit before that happens, not after.

use omega_backend_pauliprop::sim::DEFAULT_MAX_TERMS;
use omega_backend_pauliprop::PauliPropBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::params::ParameterBinding;

fn g(kind: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// A circuit engineered to branch.
///
/// Two things make this less obvious than it looks, and getting either wrong
/// produces a circuit that quietly does not branch at all:
///
/// 1. **Support.** With observable `Z0`, every other qubit carries identity,
///    and identity commutes with everything — so a T layer applied to a narrow
///    observable splits once and stops. The support has to be spread first.
/// 2. **Direction.** This backend is Heisenberg: `propagate` walks
///    `circuit.ops.iter().rev()`, so the *last* gate in the circuit is the
///    *first* one conjugated. A circuit written in the intuitive order spreads
///    the support after the T layer has already been consumed, which is exactly
///    the no-op case above.
///
/// So the ops are built in the order they must be *conjugated* — H to turn
/// `Z0` into `X0`, a CX ladder to spread it across all `n` qubits, then a T on
/// each qubit, now meeting an anticommuting factor everywhere and doubling the
/// sum each time — and then reversed on the way in.
fn branching_circuit(n: u32) -> CircuitIR {
    let mut conjugation_order = vec![g(GateKind::H, &[0], &[])];
    for q in 0..n - 1 {
        conjugation_order.push(g(GateKind::CX, &[q, q + 1], &[]));
    }
    for q in 0..n {
        conjugation_order.push(g(GateKind::T, &[q], &[]));
    }
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for op in conjugation_order.into_iter().rev() {
        c.add_op(op);
    }
    c
}

fn z0(_n: u32) -> Observable {
    // Sparse form: only the non-identity factors are listed.
    Observable {
        terms: vec![(1.0, vec![(0u32, PauliOp::Z)])],
    }
}

/// **A run that would explode is refused, and the refusal explains itself.**
///
/// Driven against a deliberately tiny ceiling rather than by building a
/// 2^24-term circuit. Testing the guard by actually allocating 16M terms would
/// consume the memory the guard exists to protect — the test would be the
/// incident. A small cap exercises the same code path in microseconds.
#[test]
fn a_branching_circuit_is_refused_rather_than_exhausting_memory() {
    let mut backend = PauliPropBackend::new();
    backend.max_terms = Some(8);
    let circuit = branching_circuit(12);
    let err = backend
        .expectation(&circuit, &ParameterBinding::default(), &z0(12))
        .expect_err("a 2^12-term expansion past an 8-term ceiling must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("terms"),
        "the refusal must say what ran away; got: {msg}"
    );
    // It must point at the knobs that make the run possible, not just say no.
    assert!(
        msg.contains("--truncate") || msg.contains("max-weight") || msg.contains("max-freq"),
        "the refusal must name a way forward; got: {msg}"
    );
}

/// **The ceiling must not break exact runs that fit.** A refusal that fires on
/// ordinary work is worse than the exhaustion it prevents, because it breaks
/// every working run rather than an unreachable one.
#[test]
fn ordinary_exact_runs_are_unaffected() {
    let backend = PauliPropBackend::new();
    // A few splits: 2^6 terms, nowhere near the ceiling.
    let circuit = branching_circuit(6);
    let v = backend
        .expectation(&circuit, &ParameterBinding::default(), &z0(6))
        .expect("a small branching circuit must still run exactly");
    assert!(v.is_finite(), "expectation must be a number, got {v}");
}

/// **Clifford circuits never branch, so the ceiling can never bite them** — at
/// any width. This is the backend's headline capability and it must stay
/// untouched by a bound aimed at non-Clifford growth.
#[test]
fn wide_clifford_circuits_are_untouched() {
    let backend = PauliPropBackend::new();
    let n = 400;
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for q in 0..n {
        c.add_op(g(GateKind::H, &[q], &[]));
    }
    for q in 0..n - 1 {
        c.add_op(g(GateKind::CX, &[q, q + 1], &[]));
    }
    backend
        .expectation(&c, &ParameterBinding::default(), &z0(n))
        .expect("a 400-qubit Clifford circuit must run: one Pauli stays one Pauli");
}

/// Truncation is the intended escape hatch from the term ceiling — but only
/// while it still leaves an answer behind.
///
/// **This test previously asserted the defect as a feature.** Its doc claimed
/// truncation gives "an approximation *with a reported bound* — not a silent
/// fallback", and its only assertion was `v.is_finite()`. What the configuration
/// below actually produced was `final_terms = 0`: every term discarded, `0.0`
/// returned, and a dropped-mass of 1.0 on an observable confined to [−1, 1].
/// A silent fallback, passing a test written to rule one out, because
/// `is_finite()` is true of `0.0`.
///
/// Both halves are asserted now: over-aggressive truncation must REFUSE, and
/// truncation that leaves something must complete with a bound that excludes
/// something.
#[test]
fn truncation_is_an_escape_hatch_only_while_it_leaves_an_answer() {
    let circuit = branching_circuit(12);
    let params = ParameterBinding::default();

    // Aggressive enough to discard the entire sum. The old expectation here
    // was `0.0`; the honest answer is a refusal.
    let mut starved = PauliPropBackend::with_truncation(1e-6, Some(3));
    starved.max_terms = Some(8);
    let err = starved
        .expectation(&circuit, &params, &z0(12))
        .expect_err("discarding every term must refuse, not return 0.0");
    assert!(
        format!("{err}").contains("excludes nothing"),
        "the refusal must say why: {err}"
    );

    // And with the ceiling lifted, confirm that is exactly what was happening —
    // the premise, asserted rather than assumed.
    let (v, cert) = starved
        .with_max_dropped_mass(Some(f64::INFINITY))
        .expectation_with_certificate(&circuit, &params, &z0(12))
        .expect("lifting the ceiling must let it through");
    assert_eq!(cert.final_terms, 0, "premise: the sum was emptied");
    assert_eq!(v, 0.0, "premise: an empty sum reads as 0.0");
    assert!(!cert.is_informative());

    // A truncation that still leaves terms: completes, and its bound is worth
    // reading. This is what the escape hatch actually looks like.
    //
    // A *weight* cap cannot demonstrate it on this circuit, and the reason is
    // worth recording: `Z0` propagates back through `H` to `X0` and then the CX
    // ladder spreads it across all `n` qubits at once, so the sum is
    // weight-`n` before the first `T`. Any cap below `n` deletes everything and
    // any cap at or above `n` cuts nothing — there is no partial regime to
    // find. Frequency truncation on a narrower register has one.
    let narrow = branching_circuit(6);
    let (v, cert) = PauliPropBackend::new()
        .max_freq(Some(4))
        .expectation_with_certificate(&narrow, &params, &z0(6))
        .expect("a truncation that leaves terms must complete");
    assert!(v.is_finite(), "expectation must be a number, got {v}");
    assert!(
        cert.is_informative(),
        "the escape hatch is only an escape if the answer means something: \
         dropped_mass {:.4e}, range {:.4e}, value {v}",
        cert.dropped_mass,
        cert.observable_range
    );
}

/// The ceiling is opt-outable for someone who knows their host, but the default
/// is on — the inverse of every other knob here, deliberately.
#[test]
fn the_ceiling_is_on_by_default_and_can_be_lifted() {
    assert_eq!(
        PauliPropBackend::new().max_terms,
        Some(DEFAULT_MAX_TERMS),
        "max_terms must default ON; it is a refusal, not an approximation"
    );
    let mut backend = PauliPropBackend::new();
    backend.max_terms = None;
    // Not exercised on a huge circuit here — the point of the test is that the
    // opt-out exists and is reachable, not to actually exhaust this machine.
    assert!(backend.max_terms.is_none());
}
