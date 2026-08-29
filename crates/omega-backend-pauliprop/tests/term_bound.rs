// SPDX-License-Identifier: Apache-2.0
//! `term_upper_bound` must actually BOUND the term count the engine reaches.
//!
//! A bound that is not a bound is a memory defect, not a pricing one: the
//! server reserves against this number, so under-pricing ends in an OOM while
//! over-pricing merely refuses a job that would have fitted. Every test here is
//! written with that asymmetry in mind — `peak <= bound` is the assertion that
//! matters, and the "is it useful" tests are secondary.
//!
//! The peak is sampled inside `branch`, i.e. MID-GATE. That is deliberate: a
//! `CCX` calls `branch` seven times while the engine's own ceiling check runs
//! once after the whole dispatch, so a gate-boundary sample could confirm a
//! bound the engine violates in flight.

use omega_backend_pauliprop::{
    branch_calls, peak_terms, reset_peak_terms, term_ceiling, term_upper_bound, PauliPropBackend,
    DEFAULT_MAX_TERMS,
};
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

/// Brickwall with rotations — grows the term count through `branch`.
fn brickwall(nq: u32, layers: usize) -> CircuitIR {
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    for q in 0..nq {
        c.ops.push(op(GateKind::H, &[q], &[]));
    }
    for l in 0..layers {
        for q in 0..nq.saturating_sub(1) {
            c.ops.push(op(GateKind::CX, &[q, q + 1], &[]));
        }
        for q in 0..nq {
            c.ops.push(op(GateKind::Rz, &[q], &[0.3 + 0.05 * l as f64]));
            c.ops.push(op(GateKind::Rx, &[q], &[0.2 + 0.05 * q as f64]));
        }
    }
    c
}

fn zz_obs() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
    }
}

fn run_and_peak(c: &CircuitIR, o: &Observable) -> usize {
    reset_peak_terms();
    let _ = PauliPropBackend::new()
        .expectation(c, &ParameterBinding::new(), o)
        .expect("propagation");
    peak_terms()
}

/// THE test. Run real circuits, record the real peak, assert the bound holds.
#[test]
fn the_bound_is_never_below_what_the_engine_reaches() {
    let o = zz_obs();
    let mut checked = 0;
    for (nq, layers) in [
        (4u32, 2usize),
        (4, 4),
        (6, 3),
        (8, 3),
        (10, 3),
        (12, 2),
        (16, 2),
        (24, 2),
    ] {
        let c = brickwall(nq, layers);
        let bound = term_upper_bound(&c, &o);
        let peak = run_and_peak(&c, &o) as u64;
        assert!(
            peak <= bound,
            "nq={nq} layers={layers}: engine reached {peak} terms but the bound \
             said {bound}. Under-pricing this is an OOM, not a refusal."
        );
        checked += 1;
    }
    assert!(checked >= 8, "corpus too thin: {checked}");
}

/// The bound must be USEFUL, not just safe. `return cap` everywhere would pass
/// the test above and change nothing — which is the state this work replaces.
#[test]
fn a_small_local_job_prices_far_below_the_cap() {
    let c = brickwall(4, 2);
    let bound = term_upper_bound(&c, &zz_obs());
    let cap = DEFAULT_MAX_TERMS as u64;
    assert!(
        bound < cap / 100,
        "a 4-qubit depth-2 circuit priced at {bound} terms against a cap of \
         {cap} — that is not far enough below to be worth computing"
    );
}

/// Width is nearly free: the same dynamics in a wider register must not raise
/// the price, because the light cone bounds the work, not the register.
#[test]
fn widening_the_register_does_not_raise_the_price() {
    let o = zz_obs();
    let narrow = term_upper_bound(&brickwall(8, 2), &o);
    let wide = term_upper_bound(&brickwall(24, 2), &o);
    assert_eq!(
        narrow, wide,
        "same depth, same observable, 8 vs 24 qubits priced differently \
         ({narrow} vs {wide}) — the cone, not the register, must set the price"
    );
}

/// Depth is NOT free — the bound must respond to what actually costs.
#[test]
fn depth_inside_the_cone_raises_the_price() {
    let o = zz_obs();
    let shallow = term_upper_bound(&brickwall(8, 1), &o);
    let deep = term_upper_bound(&brickwall(8, 3), &o);
    assert!(
        deep > shallow,
        "depth 1 priced at {shallow} and depth 3 at {deep}: adding non-Clifford \
         depth inside the cone must raise the bound"
    );
}

/// Every gate the dispatch accepts must be priced. Without this, adding a
/// branching gate to `sim.rs` silently under-prices every job using it.
///
/// The list mirrors the engine's supported set; a gate that is priced but not
/// dispatched is harmless, the reverse is not.
#[test]
fn every_dispatched_gate_is_priced() {
    use GateKind::*;
    for g in [
        H, X, Y, Z, S, Sdg, Sx, Sxdg, CX, CZ, CY, Swap, Id, Barrier, Measure, Rz, Rx, Ry, U1, T,
        Tdg, CRz, U3, CU3, CCX, CSwap,
    ] {
        assert!(
            branch_calls(&g).is_some(),
            "{g:?} is dispatched by the engine but has no branch-call count. \
             An unpriced branching gate under-prices every job that uses it."
        );
    }
}

/// The counts that were wrong once. `CSwap` was 6 in the first draft (a regex
/// miscount) when it is 7 — a 2x under-count per gate, in the OOM direction.
#[test]
fn the_branch_call_counts_are_the_ones_the_engine_makes() {
    use GateKind::*;
    assert_eq!(
        branch_calls(&H),
        Some(0),
        "Clifford gates cannot grow the sum"
    );
    assert_eq!(branch_calls(&CX), Some(0));
    assert_eq!(branch_calls(&Rz), Some(1));
    assert_eq!(branch_calls(&T), Some(1));
    assert_eq!(branch_calls(&CRz), Some(2));
    assert_eq!(branch_calls(&U3), Some(3));
    assert_eq!(
        branch_calls(&CCX),
        Some(7),
        "CCZ over the seven non-empty subsets of three qubits"
    );
    assert_eq!(
        branch_calls(&CSwap),
        Some(7),
        "CSwap is CX·CCX·CX and pays the SAME seven branchings — it was 6 in an \
         earlier draft, which under-priced every circuit using it by 2x"
    );
}

/// The ceiling is the CAP now, not the cap times a mid-gate factor.
///
/// The inverse of what this used to assert. The engine checked
/// `terms.len() > cap` once per gate AFTER the whole dispatch, so a sum could
/// enter `CCX`/`CSwap` at `cap - 1` and take seven consecutive doublings first
/// — a real peak of `2^7 · cap`, ~36 GB at n=40 against a 285 MB reservation.
/// It could exhaust memory BEFORE refusing, and the bound carried a matching
/// `2^7` so the governor would not under-reserve.
///
/// The check now runs inside `branch`'s expansion loop, so the compensation is
/// gone. What remains is a two-term transient: the test fires after a term's
/// children are added and one iteration adds at most two.
#[test]
fn the_ceiling_is_the_cap_plus_only_the_in_loop_transient() {
    let c = brickwall(20, 40);
    let bound = term_upper_bound(&c, &zz_obs());
    let cap = DEFAULT_MAX_TERMS as u64;

    assert!(
        bound >= cap,
        "a saturating circuit priced at {bound}, below the engine's cap {cap}"
    );
    assert_eq!(
        term_ceiling(),
        cap + 2,
        "the ceiling must be the cap plus the two-term in-loop transient. If it \
         grew back toward cap*2^7, the engine's check has moved back OUT of the \
         expansion loop and this bound must follow it."
    );
    assert!(
        bound <= cap + 2,
        "bound {bound} exceeds the ceiling {}",
        cap + 2
    );
}

/// The engine must REFUSE before allocating far past its own cap.
///
/// The defect the bound used to compensate for, tested directly. Peak is
/// sampled inside `branch` (mid-gate) because a gate-boundary sample would miss
/// exactly the excursion this guards against.
///
/// Two false starts, both test bugs that looked like engine bugs:
///   1. at 6 qubits there are exactly 4^6 = 4096 distinct Paulis, so a 4096 cap
///      can never be exceeded however deep the circuit runs;
///   2. a Z-only observable COMMUTES with CCX's Z-product generators, so CCX
///      alone grows nothing — `Rx` is what anticommutes with a Z term.
#[test]
fn a_saturating_run_refuses_without_blowing_past_the_cap() {
    let cap = 4096usize;
    let mut c = CircuitIR::new(12, CircuitType::GateBased);
    for q in 0..12u32 {
        c.ops.push(op(GateKind::H, &[q], &[]));
    }
    for l in 0..14u32 {
        for q in 0..11u32 {
            c.ops.push(op(GateKind::CX, &[q, q + 1], &[]));
        }
        for q in 0..12u32 {
            c.ops.push(op(GateKind::Rz, &[q], &[0.3 + 0.05 * l as f64]));
            c.ops.push(op(GateKind::Rx, &[q], &[0.2 + 0.05 * q as f64]));
        }
        c.ops.push(op(
            GateKind::CCX,
            &[l % 10, (l + 1) % 10 + 1, (l + 2) % 10 + 2],
            &[],
        ));
    }

    reset_peak_terms();
    let mut b = PauliPropBackend::new();
    b.max_terms = Some(cap);
    let r = b.expectation(&c, &ParameterBinding::new(), &zz_obs());
    let peak = peak_terms();

    assert!(r.is_err(), "this circuit should saturate a {cap}-term cap");
    assert!(
        peak <= cap + 2,
        "the engine reached {peak} terms against a {cap} cap before refusing. \
         The check is not inside the expansion loop: a CCX takes seven \
         consecutive doublings, which is how it exhausts memory BEFORE refusing."
    );
}

/// An unrecognised gate must price at the ceiling, not be skipped. Skipping
/// would under-price, which is the unsafe direction.
#[test]
fn an_unknown_gate_prices_at_the_ceiling() {
    assert!(
        branch_calls(&GateKind::Rbs).is_none(),
        "Rbs is not dispatched by pauliprop and must not be silently priced"
    );
    let mut c = brickwall(4, 1);
    c.ops.push(op(GateKind::Rbs, &[0, 1], &[0.3]));
    let bound = term_upper_bound(&c, &zz_obs());
    assert!(
        bound >= DEFAULT_MAX_TERMS as u64,
        "a circuit containing an unpriced gate came back at {bound}, below the \
         cap — an unknown gate must resolve toward the ceiling"
    );
}

/// Overflow must saturate toward the ceiling, never wrap toward zero.
///
/// Release builds do NOT panic here (no `[profile]` overrides): `1u64 << 64`
/// masks to 1 and `4u64.pow(32)` wraps to 0, and reserving zero is the worst
/// available outcome. Runs in both profiles; the release run is the one that
/// reflects what ships.
#[test]
fn a_huge_circuit_saturates_instead_of_wrapping() {
    // 400 rotations in the cone => 2^400 by the branching ceiling.
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    for i in 0..400 {
        c.ops.push(op(GateKind::Rz, &[0], &[0.1 * i as f64]));
    }
    let bound = term_upper_bound(&c, &zz_obs());
    assert!(
        bound > 0,
        "the bound wrapped to {bound} — reserving nothing is the one outcome \
         that guarantees an OOM"
    );
    // 2 qubits => at most 4^2 = 16 distinct Paulis, so the Pauli-space ceiling
    // should dominate and keep this tiny despite 400 branches.
    assert!(
        bound <= 16,
        "2 qubits hold at most 16 distinct Paulis, but the bound came back \
         {bound} — the 4^|cone| ceiling is not being applied"
    );
}
