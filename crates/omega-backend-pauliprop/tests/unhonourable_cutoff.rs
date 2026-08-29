// SPDX-License-Identifier: Apache-2.0
//! A requested cutoff that does not bound growth must SAY SO (3b.2 R4).
//!
//! The bug report measured `--truncate 1e-6` on a depth-16 ansatz hitting the
//! term ceiling anyway, and getting a refusal that never mentioned the cutoff:
//! *"`1e-6` neither completed nor warned; it simply hit the ceiling anyway. If
//! a requested cutoff does not bound growth, say so rather than failing as
//! though no cutoff had been given."*
//!
//! The two situations are different facts and the remedies point in opposite
//! directions — with no cutoff you add one, with an ineffective cutoff you
//! tighten the one you have — so a message that reads the same for both sends
//! half its readers the wrong way.

use omega_backend_pauliprop::PauliPropBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: params.iter().map(|p| ParamExpr::Concrete(*p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// T gates on every qubit: `cos θ = sin θ`, so coefficient truncation cannot
/// prune either branch. This is the worst case named in the refusal itself, and
/// it is what makes a small `--truncate` provably unable to bound growth.
///
/// **Gate ORDER is load-bearing and the first version of this got it wrong.**
/// pauliprop back-propagates the observable from the end of the circuit, and
/// `T` is diagonal, so it COMMUTES with `Z` — a `Z₀` observable meeting `T`
/// gates first passes straight through and never branches. That fixture
/// completed at 4096 terms and the tests failed on their own premise.
///
/// So the T layers go FIRST in circuit order (hit LAST in back-propagation),
/// with `H` and the CX ladder at the end: `Z₀` → `X₀` at the `H`, spread across
/// the register by the ladder, and only then does it reach the `T`s as an
/// X-type operator, which branches.
fn t_heavy(nq: u32, layers: usize) -> CircuitIR {
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    for _ in 0..layers {
        for q in 0..nq {
            c.ops.push(op(GateKind::T, &[q], &[]));
        }
    }
    for q in (0..nq - 1).rev() {
        c.ops.push(op(GateKind::CX, &[q, q + 1], &[]));
    }
    c.ops.push(op(GateKind::H, &[0], &[]));
    c
}

fn z0() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    }
}

/// BOTH halves in one test, deliberately.
///
/// Split apart, a change that hard-coded the "DESPITE" clause on would pass the
/// first half, and one that deleted it entirely would pass the second. Only
/// asserting the contrast pins that the message tracks what was actually
/// requested.
#[test]
fn a_cutoff_that_cannot_bound_growth_is_distinguished_from_no_cutoff() {
    let circuit = t_heavy(14, 6);
    let params = ParameterBinding::new();

    // (a) No cutoff. The refusal must NOT claim one was requested.
    let mut exact = PauliPropBackend::new();
    exact.max_terms = Some(4096);
    let bare = format!(
        "{}",
        exact
            .expectation(&circuit, &params, &z0())
            .expect_err("premise: this circuit must exceed a 4096-term ceiling")
    );
    assert!(
        !bare.contains("DESPITE"),
        "an exact run must not be told its cutoff failed — it has none: {bare}"
    );
    assert!(bare.contains("4096 ceiling"), "{bare}");

    // (b) A cutoff too small to bound T-gate branching. The refusal must name
    // it, and say it is being applied — the report's complaint was that the
    // message read as though no cutoff had been given.
    let mut truncating = PauliPropBackend::with_truncation(1e-6, None);
    truncating.max_terms = Some(4096);
    let told = format!(
        "{}",
        truncating
            .expectation(&circuit, &params, &z0())
            .expect_err("a cutoff of 1e-6 cannot bound T-gate branching")
    );
    assert!(
        told.contains("DESPITE"),
        "an ineffective cutoff must be distinguished from no cutoff: {told}"
    );
    assert!(
        told.contains("--truncate 1e-6"),
        "the message must name the cutoff actually in force: {told}"
    );
    assert!(
        told.contains("tightening it is the remedy"),
        "and must point the opposite way from 'add a cutoff': {told}"
    );
}

/// The other two truncation axes are named too, not just `--truncate`.
#[test]
fn the_weight_and_frequency_caps_are_named_when_they_are_the_ones_in_force() {
    let circuit = t_heavy(14, 6);
    let params = ParameterBinding::new();

    // A cap AT the register width cuts nothing — the honest R4 shape: a knob
    // was set and did not bound anything. (A cap of 12 on 14 qubits is the
    // other failure: it bounds growth so hard the sum empties, and the run is
    // then refused by the INFORMATIVENESS gate instead, never reaching the
    // term ceiling this test is about.)
    let mut by_weight = PauliPropBackend::with_truncation_freq(0.0, Some(14), None);
    by_weight.max_terms = Some(4096);
    let msg = format!(
        "{}",
        by_weight
            .expectation(&circuit, &params, &z0())
            .expect_err("a weight cap at the register width bounds nothing")
    );
    assert!(msg.contains("--max-weight 14"), "{msg}");
    assert!(msg.contains("DESPITE"), "{msg}");
    assert!(
        !msg.contains("--truncate 0"),
        "a coeff_min of 0.0 is NOT a cutoff in force and must not be listed: {msg}"
    );
}

/// R3, asserted rather than assumed: the refusal must state that the three
/// truncating options CHANGE THE ANSWER. The report's complaint was that it
/// "says nothing about `--truncate` or that truncating changes the answer".
#[test]
fn the_refusal_says_truncation_is_approximate() {
    let circuit = t_heavy(14, 6);
    let mut be = PauliPropBackend::new();
    be.max_terms = Some(4096);
    let msg = format!(
        "{}",
        be.expectation(&circuit, &ParameterBinding::new(), &z0())
            .expect_err("must refuse")
    );
    for needle in [
        "--truncate",
        "--max-weight",
        "--max-freq",
        "--max-terms",
        "CHANGE THE ANSWER",
        "dropped_mass",
    ] {
        assert!(msg.contains(needle), "refusal must name '{needle}': {msg}");
    }
}
