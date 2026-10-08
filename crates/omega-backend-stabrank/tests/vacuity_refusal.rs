// SPDX-License-Identifier: Apache-2.0
//! S2 test (iii): the **vacuity gate**. A bound that excludes nothing is not
//! a result, and the engine says so instead of handing back a number.
//!
//! `R·m·(2+m) ≥ R + |v|` means the certified interval `[v−b, v+b]` already
//! covers `[−R, R]`, the whole range `⟨O⟩` could possibly take. Every value
//! the observable can have is then consistent with the run, so the run has
//! measured nothing. Returning `v` anyway would be worse than returning an
//! error: the number looks like an answer, the certificate beside it is
//! *correct*, and only a reader who recomputes the interval learns that the
//! two together say "somewhere in the range".
//!
//! This is majoranaprop's `finish`, with this engine's arithmetic. The shape
//! is deliberately identical — refusal, not a return — because it is the
//! project's standing certificate contract and not a per-backend taste.
//!
//! # How this test fails if the gate is absent
//!
//! Every assertion below is on `is_err()` and on the refusal's text. Delete
//! the gate and the engine returns `Ok` with a meaningless value, and these
//! tests fail at `is_ok()` — not at a tolerance, and not by comparing the
//! meaningless value against anything.
//!
//! # The fixture
//!
//! Twelve `T` gates with `max_chi = 1`. That is the hardest cut the engine
//! can make, and the circuit is built to exceed the ceiling by a wide margin
//! rather than to sit beside it — the boundary itself is held by the second
//! test below, which accepts a lighter cut on the same construction.
//!
//! After every split both legs exist and the lighter one is discarded whole,
//! so `m = sin(π/8)·Σ_{k<12} cos^k(π/8) = 3.0832`, giving
//! `R·m·(2+m) = 15.673` against a ceiling of `R + |v| = 1`. It is also the
//! *cheapest* run in this crate: `χ = 1` throughout, one inner product per
//! observable term.
//!
//! The fixture is built on `T` rather than on the small angles of
//! `truncation_witness.rs` for the reason that file's doc comment measures
//! from the other side: naive sum-over-Cliffords truncation of twelve `T`
//! gates to a small `χ` **cannot** be informative, because the decomposition
//! carries an L1 mass of `1.307¹² = 24.74` and an informative bound may
//! discard at most 1.7% of it. The two fixtures are the same arithmetic read
//! in both directions, and this one is where that arithmetic is a refusal.
//!
//! # What this fixture does NOT catch
//!
//! Anything about the *value*, which it never looks at, and anything about
//! the `(2+m)` factor: at `m = 3.08` the gate fires under `R·m·m`, under
//! `R·m·(2+m)/4`, and under plain `m` alike. It is a test of the gate, not of
//! the bound — `truncation_witness.rs` is where the bound's arithmetic is
//! constrained, and the two cover each other.

use omega_backend_stabrank::{StabRankBackend, DEFAULT_MAX_BRANCHES};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;

const T_COUNT: usize = 12;

fn gop(kind: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// `t` `T` gates on three wires with Cliffords between them, so the branches
/// are genuinely different states and not `t` copies of one.
fn t_circuit(t: usize) -> CircuitIR {
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    for q in 0..3 {
        c.add_op(gop(GateKind::H, &[q], &[]));
    }
    for i in 0..t {
        c.add_op(gop(GateKind::T, &[(i % 3) as u32], &[]));
        c.add_op(gop(GateKind::H, &[((i + 1) % 3) as u32], &[]));
        c.add_op(gop(
            GateKind::CX,
            &[(i % 3) as u32, ((i + 2) % 3) as u32],
            &[],
        ));
    }
    c
}

/// The fixture built to exceed the ceiling: twelve `T` gates, cut to one
/// branch.
fn over_the_ceiling() -> CircuitIR {
    t_circuit(T_COUNT)
}

fn observable() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (2, PauliOp::X)])],
    }
}

/// `m = sin(π/8) · Σ_{k=0}^{11} cos^k(π/8)`, the closed form for a `max_chi`
/// of 1: each split scales the surviving coefficient by `cos(π/8)` and
/// discards `sin(π/8)` times what it had.
fn predicted_mass() -> f64 {
    use std::f64::consts::FRAC_PI_8;
    let (c, s) = (FRAC_PI_8.cos(), FRAC_PI_8.sin());
    let mut surviving = 1.0;
    let mut m = 0.0;
    for _ in 0..T_COUNT {
        m += s * surviving;
        surviving *= c;
    }
    m
}

/// The gate fires, the engine refuses, and the message names the knobs.
#[test]
fn a_bound_that_excludes_nothing_is_refused_and_not_returned() {
    let params = ParameterBinding::new();
    let circuit = over_the_ceiling();
    let obs = observable();
    let result = StabRankBackend::with_truncation(0.0, Some(1))
        .expectation_with_certificate(&circuit, &params, &obs);

    let err = match result {
        Err(e) => e,
        Ok((value, cert)) => panic!(
            "the engine returned {value} with expectation_error_bound = {} against \
             R + |value| = {}. The certified interval [{}, {}] contains the whole \
             a-priori range [-{}, {}], so this number excludes nothing and must \
             not have been returned.",
            cert.expectation_error_bound,
            cert.observable_range + value.abs(),
            value - cert.expectation_error_bound,
            value + cert.expectation_error_bound,
            cert.observable_range,
            cert.observable_range,
        ),
    };

    let msg = err.to_string();
    eprintln!("stabrank S2 vacuity refusal: {msg}");
    for needle in [
        // The knob that caused it, and the one that is not it.
        "max_chi",
        "coeff_min",
        "with_truncation",
        "with_max_branches",
        // The other engine, as majoranaprop's own refusal names this one.
        "majoranaprop",
        // The two numbers a reader needs to decide what to do next.
        "R·m·(2+m)",
    ] {
        assert!(
            msg.contains(needle),
            "the refusal must name {needle:?}; got: {msg}"
        );
    }

    // The mass it refused on, in closed form: the message's numbers have to
    // be the run's, not a constant someone typed into a format string.
    let m = predicted_mass();
    assert!(
        (3.0..3.1).contains(&m),
        "the fixture's mass drifted to {m}; it is the thing that makes the \
         bound vacuous and it is supposed to be ~3.084"
    );
    assert!(
        msg.contains(&format!("{m:.4e}")),
        "the refusal reports a dropped mass other than the run's {m:.4e}: {msg}"
    );
}

/// The gate is a boundary, not a mood: the same construction under a cut that
/// discards little enough comes back `Ok`, informative, and inside its bound.
///
/// Without this leg the test above is satisfied by an engine that refuses
/// every truncated run, which is a different bug with the same symptom.
///
/// The cut here is `χ ≤ 48` of six `T` gates' 64 branches — three quarters of
/// them kept, discarding `m ≈ 0.213` of the decomposition's 4.97 total mass,
/// for a bound of `0.472 < 1`. It is **not** the twelve-gate fixture under a
/// lighter cut, and the reason is the measurement that attempt produced: at
/// `max_chi = 2048`, half of the 4096 branches, the twelve-gate circuit still
/// discards `m = 1.7436` and is still refused, bound `6.5273` against a
/// ceiling of `1.2832`. Informativeness needs `m < √2 − 1 ≈ 0.414`, which on
/// that circuit means keeping roughly 3300 branches and paying 10.9M pairwise
/// overlaps for the privilege. That is the `1.307^t` mass growth of the naive
/// decomposition, stated as a run rather than as a remark.
#[test]
fn the_same_construction_is_accepted_once_the_cut_is_light_enough() {
    let params = ParameterBinding::new();
    let circuit = t_circuit(6);
    let obs = observable();

    let (value, cert) = StabRankBackend::with_truncation(0.0, Some(48))
        .expectation_with_certificate(&circuit, &params, &obs)
        .expect("a light cut on a six-gate circuit must be accepted");
    eprintln!(
        "stabrank S2 vacuity boundary: χ = {}, m = {:.6e}, bound = {:.6e}, \
         R + |v| = {:.6e}",
        cert.final_chi,
        cert.state_dropped_mass,
        cert.expectation_error_bound,
        cert.observable_range + value.abs()
    );
    assert_eq!(cert.final_chi, 48, "the cut must bind at 48 of 64 branches");
    assert!(
        cert.state_dropped_mass > 0.0,
        "the light cut dropped nothing, so it is not on the same side of the \
         same boundary as the refusal above"
    );
    assert!(cert.is_informative());
    assert!(!cert.is_exact());
    assert_eq!(cert.max_branches, DEFAULT_MAX_BRANCHES);
}

/// A truncation setting with no run behind it is refused where it is given
/// meaning, not silently reinterpreted as the nearest sensible one.
#[test]
fn a_ceiling_of_zero_branches_is_refused_by_name() {
    let params = ParameterBinding::new();
    let err = StabRankBackend::with_truncation(0.0, Some(0))
        .expectation_with_certificate(&over_the_ceiling(), &params, &observable())
        .expect_err("a χ ceiling of zero is not a state");
    assert!(
        err.to_string().contains("Some(1)"),
        "the refusal must say what to write instead; got: {err}"
    );

    let err = StabRankBackend::with_truncation(f64::NAN, None)
        .expectation_with_certificate(&over_the_ceiling(), &params, &observable())
        .expect_err("a NaN floor compares false against everything");
    assert!(
        err.to_string().contains("coeff_min"),
        "the refusal must name the knob; got: {err}"
    );
}
