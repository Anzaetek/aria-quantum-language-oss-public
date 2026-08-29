// SPDX-License-Identifier: Apache-2.0
//! `pauliprop` must never silently apply a classically-conditioned gate
//! unconditionally — it must either **answer the mixture correctly** or refuse.
//!
//! # This file changed direction, and the reason matters
//!
//! It originally required a *refusal*, because there was no representation of a
//! mixture to return. There is one now: `omega_core::defer_measure` defers the
//! measurement into a quantum control and dephases the observable, so this
//! backend answers the feedforward circuit exactly instead of abstaining.
//!
//! What did NOT change is the property under test: the unguarded answer must
//! never be passed off as the guarded one. The bar simply moved from "refuses"
//! to "gets it right", and the third test below — which measures truth by
//! sampling a different backend — is what makes that check independent of this
//! one's arithmetic.
//!
//! The `op.condition` guard in `sim.rs` is retained as a backstop. It is now
//! unreachable through `expectation`, since deferral removes every condition
//! before `propagate` sees the circuit, but `propagate` is not the only possible
//! caller and a live wrong answer is worse than dead code.
//!
//! # The defect
//!
//! Observable conjugation evolves `O → U†OU` for a **single** unitary `U`. A
//! guarded gate makes the circuit a classical mixture over measurement
//! outcomes, which is not one unitary and has no such representation.
//!
//! The backend had **zero references to `op.condition` anywhere in the crate**
//! — `sim.rs`, `pauli.rs`, `lib.rs` — while statevector (3 call sites), MPS (2)
//! and Pauli (1) all consult `condition_satisfied`. So a guarded gate ran
//! unconditionally and the result was reported with full confidence.
//!
//! This is the same failure mode the `Reset` refusal in `sim.rs` already guards
//! against, and its doc comment makes the identical argument: "Silently
//! skipping it answered a DIFFERENT circuit." Conditionals were the other half
//! and were missed.
//!
//! Found by the N-way matrix work (`FIXES_PLAN.md` K7).

use omega_backend_pauliprop::PauliPropBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::error::OmegaError;
use omega_core::executor::{Backend, Observable};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32], cbit: Option<u32>) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: smallvec::smallvec![],
        classical_bit: cbit,
        condition: None,
    }
}

/// `h q0; measure q0 -> c0; if (c==1) x q1` — the shape of
/// `12_feedforward_sometimes_false.qasm`.
fn feedforward() -> CircuitIR {
    let mut guarded_x = op(GateKind::X, &[1], None);
    guarded_x.condition = Some((0, 1, 1));
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.num_classical_bits = 1;
    ir.ops = vec![
        op(GateKind::H, &[0], None),
        op(GateKind::Measure, &[0], Some(0)),
        guarded_x,
    ];
    ir
}

/// **The feedforward circuit is now ANSWERED, and the answer is the mixture.**
///
/// `h q0; measure q0 -> c0; if(c==1) x q1` deferred is `h q0; cx q0,q1` — a Bell
/// state — and `⟨Z₁⟩` on a Bell state is **exactly 0**, which is the mixture's
/// value. The old silent behaviour gave −1.
///
/// Exact equality, not a tolerance: `Z₁` on a Bell state is a cancellation of
/// two terms of equal magnitude, not a limit.
#[test]
fn the_feedforward_circuit_is_answered_with_the_mixture_value() {
    let val = PauliPropBackend::new()
        .expectation(&feedforward(), &ParameterBinding::new(), &Observable::z(1))
        .expect("deferral makes this circuit expressible");
    assert!(
        val.abs() < 1e-12,
        "<Z1> for the feedforward mixture is 0; got {val}. -1 would mean the \
         guard was applied unconditionally (the original defect); +1 would mean \
         it was dropped."
    );
}

/// **A circuit that genuinely cannot be deferred is still refused, by name.**
///
/// `h q0; measure q0 -> c0; h q0` uses the measured qubit coherently afterwards,
/// so there is no deferred form: the measurement cannot move past work that
/// depends on it having happened. This is also the reset-and-reuse shape, i.e.
/// every QEC ancilla.
///
/// This was a **live wrong answer** until the shared rule landed. `Measure` was
/// grouped with `Id` and `Barrier` as a no-op, so this circuit was evaluated as
/// `h; h` = identity and returned `⟨Z₀⟩ = +1` where the truth is 0 — the
/// measurement destroys the coherence that the second `H` would otherwise
/// restore.
#[test]
fn a_coherently_reused_measured_qubit_is_refused_by_name() {
    let mut ir = CircuitIR::new(1, CircuitType::GateBased);
    ir.num_classical_bits = 1;
    ir.ops = vec![
        op(GateKind::H, &[0], None),
        op(GateKind::Measure, &[0], Some(0)),
        op(GateKind::H, &[0], None),
    ];
    let err = PauliPropBackend::new()
        .expectation(&ir, &ParameterBinding::new(), &Observable::z(0))
        .expect_err("a measured qubit used coherently afterwards is not deferrable");
    assert!(
        matches!(err, OmegaError::Unsupported(_)),
        "must be Unsupported — an honest abstention, which the N-way matrix \
         files as `cannot-express` rather than a failure: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("measured") && msg.contains("used again"),
        "the message must name the construct that caused the refusal; got: {msg}"
    );
}

/// The deferral must not have quietly become "ignore the guard".
///
/// Strip the condition and the same circuit gives a DIFFERENT answer: −1, since
/// the X now fires unconditionally. If deferral were secretly dropping the
/// guard, this value and the previous test's 0 would coincide, and both tests
/// would pass while the backend was wrong. The two together pin the gap.
#[test]
fn the_same_circuit_without_the_guard_is_accepted() {
    let mut ir = feedforward();
    for o in &mut ir.ops {
        o.condition = None;
    }
    let val = PauliPropBackend::new()
        .expectation(&ir, &ParameterBinding::new(), &Observable::z(1))
        .expect("an unguarded circuit must still work");
    // X applied unconditionally to q1 → ⟨Z₁⟩ = −1.
    assert!(
        (val + 1.0).abs() < 1e-12,
        "unguarded X on q1 gives <Z1> = -1, got {val}"
    );
}

/// What the silent behaviour actually produced, measured against a backend
/// that models the circuit correctly.
///
/// The old code ignored the condition, so for the GUARDED circuit it computed
/// the **unguarded** answer. Measured: `⟨Z₁⟩ = −1`, i.e. "q1 definitely
/// flipped". The correct answer for the mixture is `0` — the X fires on about
/// half the shots, so q1 carries no net Z polarisation.
///
/// A gap of **1.0 on an observable bounded in [−1, +1]**. Note that is the
/// largest error attainable *given the correct answer is 0*: the old value sat
/// at one endpoint of the range while the truth sat at its centre. (An earlier
/// draft of this test called it "wrong by a full sign" and asserted a
/// gap > 1.5, which is impossible here and simply failed — the phrase
/// described a +1 → −1 swing that does not occur on this circuit.)
///
/// The "correct" value comes from `StatevectorBackend`, not from arithmetic in
/// this file: a hand-computed expected value is a second implementation with no
/// reviewer.
#[test]
fn the_old_silent_answer_differed_from_truth_by_the_full_half_range() {
    use omega_core::executor::{ExecConfig, ExecResult, MidCircuitMode};

    let mut unguarded = feedforward();
    for o in &mut unguarded.ops {
        o.condition = None;
    }
    let silent_answer = PauliPropBackend::new()
        .expectation(&unguarded, &ParameterBinding::new(), &Observable::z(1))
        .expect("unguarded runs");
    assert!(
        (silent_answer + 1.0).abs() < 1e-12,
        "the old silent answer should be -1 (X applied unconditionally), got {silent_answer}"
    );

    // Truth for the GUARDED circuit, from a backend that honours the guard.
    // Collapse mode, so each shot is an independent trajectory.
    const SHOTS: u32 = 20_000;
    let guarded = feedforward();
    let config = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(7),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let ExecResult::Counts(counts) = omega_backend_statevector::StatevectorBackend::new()
        .execute(&guarded, &ParameterBinding::new(), &config)
        .expect("statevector runs the guarded circuit")
    else {
        panic!("expected counts")
    };
    // c0 records q0's measurement; the guard fires exactly when c0 == 1, and
    // that is also when q1 is flipped. So P(q1 = 1) = P(c0 = 1), and
    // <Z1> = 1 - 2*P(c0 = 1).
    let w = counts.keys().next().map(|o| o.width()).unwrap_or(1);
    let p1 = *counts
        .get(&omega_core::outcome::Outcome::from_u64(1, w))
        .unwrap_or(&0) as f64
        / SHOTS as f64;
    let truth = 1.0 - 2.0 * p1;
    assert!(
        truth.abs() < 0.02,
        "expected <Z1> ~ 0 from the statevector backend, got {truth} (p1 = {p1})"
    );

    let gap = (silent_answer - truth).abs();
    assert!(
        gap > 0.9,
        "the old silent answer ({silent_answer}) must differ from the measured \
         truth ({truth}) by ~1.0 on a [-1,+1] observable; got gap {gap}"
    );
}
