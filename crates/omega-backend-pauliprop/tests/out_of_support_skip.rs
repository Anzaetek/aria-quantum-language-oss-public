// SPDX-License-Identifier: Apache-2.0
//! Gates acting outside the Pauli sum's support are skipped — and skipping them
//! changes nothing.
//!
//! # Why this needs a counter and not just a value comparison
//!
//! The skip is **semantically inert**: when no term has support on a gate's
//! qubits, the rebuild it replaces would produce a bit-identical map. So a test
//! that only compares expectation values passes whether or not the guard is
//! present, and would keep passing if the guard were deleted tomorrow. That is
//! a test with no teeth.
//!
//! `gates_skipped()` makes the optimisation observable, so these tests assert
//! BOTH halves: the answer is unchanged, AND the guard actually fired.
//!
//! # The property being defended
//!
//! Pauli propagation's whole claim is that it is width-unbounded — it runs at
//! 100+ qubits where a statevector cannot. That only holds if cost is bounded by
//! the OBSERVABLE'S LIGHT CONE rather than by the register. Measured on this
//! corpus, the term dynamics are identical at 10 and at 20 qubits; before the
//! skip, the insert count still rose 58% across that range purely from gates
//! outside the support. This test pins the fix.

use omega_backend_pauliprop::{gates_skipped, PauliPropBackend};
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

/// A brickwall on `active` qubits, embedded in a register of `nq`.
///
/// The gates on qubits >= `active` are the point: they are real gates on real
/// qubits, they just cannot affect an observable supported on 0..4 within this
/// depth, because the CX chain only spreads support one qubit per layer.
fn embedded(nq: u32, active: u32, layers: usize) -> CircuitIR {
    let mut ops = Vec::new();
    for q in 0..nq {
        ops.push(op(GateKind::H, &[q], &[]));
    }
    for l in 0..layers {
        for q in 0..nq - 1 {
            ops.push(op(GateKind::CX, &[q, q + 1], &[]));
        }
        for q in 0..nq {
            ops.push(op(GateKind::Rz, &[q], &[0.3 + 0.05 * l as f64]));
            ops.push(op(GateKind::Rx, &[q], &[0.2 + 0.05 * q as f64]));
        }
    }
    let _ = active;
    let mut c = CircuitIR::new(nq, CircuitType::GateBased);
    c.ops = ops;
    c
}

fn obs() -> Observable {
    Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (3, PauliOp::Z)])],
    }
}

/// Widening the register must not change the answer, and must not add work.
///
/// The observable is supported on {0, 3} and the CX chain spreads support one
/// qubit per layer, so at 4 layers nothing beyond ~qubit 7 can matter. Every
/// gate past that is skippable.
#[test]
fn widening_the_register_changes_neither_the_value_nor_the_work() {
    let params = ParameterBinding::new();
    let o = obs();

    let narrow = PauliPropBackend::new()
        .expectation(&embedded(10, 10, 4), &params, &o)
        .expect("narrow");

    let before = gates_skipped();
    let wide = PauliPropBackend::new()
        .expectation(&embedded(20, 10, 4), &params, &o)
        .expect("wide");
    let skipped = gates_skipped() - before;

    // Tolerance, NOT bit-identity — and the distinction is the interesting part.
    //
    // The skip is the identity on every TERM, but it changes the ORDER terms
    // are later summed in: a rebuilt `HashMap` iterates differently from one
    // that was left alone, and floating-point addition is not associative. The
    // observed gap is ~2 ULP. Asserting bit-identity here would be asserting a
    // property the change does not have, and it would fail for a reason that is
    // not a defect.
    //
    // (The pre-existing iteration order is already nondeterministic run to run,
    // so bit-identity was never on offer in the first place.)
    assert!(
        (narrow - wide).abs() < 1e-12,
        "embedding the same dynamics in a wider register changed the answer: \
         narrow {narrow} vs wide {wide} (delta {:.3e}). Reordering can move the \
         last bits; it cannot move this much.",
        (narrow - wide).abs()
    );

    // TEETH: without the guard this is 0 and the test fails.
    assert!(
        skipped > 0,
        "no gate was skipped on a 20-qubit register whose observable is \
         supported on {{0,3}} at depth 4. The out-of-support guard is not \
         firing, and the value assertion above cannot detect that on its own."
    );
}

/// The skip must not fire when every qubit really IS in the support.
///
/// The first version of this test asserted zero skips on a small register and
/// FAILED with 94 — because the premise was wrong, not the code. Propagation is
/// Heisenberg: it starts from the observable and the support GROWS. With an
/// observable on {0,3}, qubits 1 and 2 are genuinely outside it at the start, so
/// skipping gates there is correct.
///
/// To get a real no-skip case the observable has to cover the register from the
/// first insert, which is what this does. The mask never shrinks, so once every
/// bit is set no gate can ever be skipped.
#[test]
fn a_full_support_observable_skips_nothing() {
    let params = ParameterBinding::new();
    let full = Observable {
        terms: vec![(1.0, (0..4u32).map(|q| (q, PauliOp::Z)).collect::<Vec<_>>())],
    };

    let before = gates_skipped();
    let v = PauliPropBackend::new()
        .expectation(&embedded(4, 4, 3), &params, &full)
        .expect("full-support");
    let skipped = gates_skipped() - before;

    assert!(v.is_finite(), "expectation should be finite, got {v}");
    assert_eq!(
        skipped, 0,
        "{skipped} gate(s) were skipped although the observable is supported on \
         EVERY qubit of the register — the guard is dropping work it should be \
         doing."
    );
}

/// An independent implementation has to agree.
///
/// The tests above are self-consistency: they compare pauliprop against
/// pauliprop. A skip that is wrong in the same way in both arms would survive
/// all of them. The statevector backend shares no code with this engine, so it
/// is the check that can actually catch a bad skip condition.
#[test]
fn the_skip_agrees_with_an_independent_statevector() {
    use omega_backend_statevector::StatevectorBackend;

    let params = ParameterBinding::new();
    let o = obs();
    // 12 qubits: wide enough that the observable's light cone does NOT cover the
    // register at depth 3 (so skips fire), small enough for a dense reference.
    let c = embedded(12, 12, 3);

    let before = gates_skipped();
    let pp = PauliPropBackend::new()
        .expectation(&c, &params, &o)
        .expect("pauliprop");
    let skipped = gates_skipped() - before;
    let sv = StatevectorBackend::new()
        .expectation(&c, &params, &o)
        .expect("statevector");

    assert!(
        skipped > 0,
        "no gate was skipped, so this run does not exercise the guard at all"
    );
    assert!(
        (pp - sv).abs() < 1e-9,
        "pauliprop {pp} disagrees with the statevector reference {sv} \
         (delta {:.3e}) over a run in which {skipped} gates were skipped as \
         out-of-support. The skip condition is wrong.",
        (pp - sv).abs()
    );
}

/// The support mask is allowed to OVER-approximate but never to
/// under-approximate. Truncation removes terms without clearing their bits, so
/// after a truncating run the mask can claim support that no longer exists —
/// which costs a skipped skip, and must never cost correctness.
#[test]
fn truncation_leaves_the_mask_conservative_not_wrong() {
    let params = ParameterBinding::new();
    let o = obs();
    let c = embedded(12, 12, 5);

    let exact = PauliPropBackend::new()
        .expectation(&c, &params, &o)
        .expect("exact");

    for max_freq in [1u32, 2, 3] {
        let (v, dropped) = PauliPropBackend::new()
            .max_freq(Some(max_freq))
            // A deliberate sweep across aggressive caps, checking that the
            // support MASK stays conservative — the value is compared against
            // the exact one, never quoted on its own. At max_freq=1 the budget
            // is vacuous (~7.5 on a quantity in [-1, 1]), which is exactly what
            // the informativeness gate refuses to hand back as an ANSWER; here
            // it is evidence, so the ceiling is lifted explicitly.
            .with_max_dropped_mass(Some(f64::INFINITY))
            .expectation_with_budget(&c, &params, &o)
            .expect("truncated");
        // dropped_mass is a CERTIFIED bound on the error, so it must actually
        // bound it. If the mask under-approximated, work would vanish without
        // being charged to the budget and this would fail.
        assert!(
            (v - exact).abs() <= dropped + 1e-12,
            "max_freq={max_freq}: |{v} - {exact}| = {:.3e} exceeds the certified \
             dropped_mass bound {dropped:.3e}. Work went missing without being \
             charged to the error budget.",
            (v - exact).abs()
        );
    }
}
