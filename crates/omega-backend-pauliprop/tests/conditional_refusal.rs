// SPDX-License-Identifier: Apache-2.0
//! `pauliprop` must REFUSE a classically-conditioned gate, not silently apply
//! it unconditionally.
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

#[test]
fn a_conditioned_gate_is_refused_with_a_typed_error() {
    let err = PauliPropBackend::new()
        .expectation(&feedforward(), &ParameterBinding::new(), &Observable::z(1))
        .expect_err("a guarded gate has no conjugation representation");
    assert!(
        matches!(err, OmegaError::Unsupported(_)),
        "must be Unsupported (a correct refusal, which the N-way matrix files \
         as `cannot-express`), not a generic error: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("conditioned") && msg.contains("statevector"),
        "the message must name the construct and point at a backend that \
         models it; got: {msg}"
    );
}

/// The refusal must be load-bearing, not incidental.
///
/// Strip the condition and the same circuit is accepted — so the error comes
/// from the guard specifically, not from `Measure`, the register, or the
/// observable. Without this, a backend that refused *everything* would pass the
/// test above.
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
/// draft of this test called it "wrong by a full sign" and asserted a gap
/// > 1.5, which is impossible here and simply failed — the phrase described a
/// +1 → −1 swing that does not occur on this circuit.)
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
    let p1 = *counts.get(&1).unwrap_or(&0) as f64 / SHOTS as f64;
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
