// SPDX-License-Identifier: Apache-2.0
//! **The acceptance test for P2, and it is a NON-DIAGONAL observable.**
//!
//! `PLAN-SIX-PROGRAMMES.md` P2 makes this the criterion that separates the three
//! candidate designs, because diagonal observables cannot tell them apart:
//!
//! | design | `⟨Z₀Z₁⟩` | `⟨X₀X₁⟩` |
//! |---|---|---|
//! | truth (collapse) | +1 | **0** |
//! | option (a), refuse non-diagonal | +1 | *refused* |
//! | naive deferral, no dephasing | +1 | **+1** |
//! | option (b), defer + dephase | +1 | **0** |
//!
//! Every existing fixture in the tree is Z-type, which is exactly why this
//! defect survived: the whole failure mode is off-diagonal.
//!
//! # The reference is external, and the off-diagonal half needs no tolerance
//!
//! `AerSimulator(method="density_matrix")` on `h q0; measure q0; if(c==1) x q1`
//! gives `⟨ZZ⟩ = 1.0`, `⟨XX⟩ = 0.0`, `⟨YY⟩ = 0.0` — measured 2026-08-19, against
//! the pure-Bell `1.0 / 1.0 / −1.0`.
//!
//! Aer samples one outcome per shot, so its *diagonal* values need averaging and
//! carry √N noise. Its *off-diagonal* values are exactly 0 at every shot count,
//! because each collapsed branch individually has zero X-expectation. So the
//! assertion below is **exact equality on a deleted term**, not a tolerance —
//! which is the strongest form available and the reason this test is worth more
//! than an averaged diagonal comparison.

use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::defer_measure::{defer_measurements, NotDeferrable};
use omega_core::executor::{Observable, PauliOp};

fn op(gate: GateKind, qubits: &[u32]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: Default::default(),
        classical_bit: None,
        condition: None,
    }
}

fn measure(q: u32, cbit: u32) -> GateOp {
    GateOp {
        gate: GateKind::Measure,
        qubits: [Qubit(q)].into_iter().collect(),
        params: Default::default(),
        classical_bit: Some(cbit),
        condition: None,
    }
}

/// The corpus fixture `12_feedforward_sometimes_false.qasm`, **all four ops**:
///
/// ```text
/// h q[0];  measure q[0] -> c[0];  if (c==1) x q[1];  measure q[1] -> c[0];
/// ```
///
/// # The last line is the whole point, and the first version of this file
/// omitted it
///
/// That trailing `measure q[1] -> c[0]` is a SECOND write to `c[0]`. The pass
/// originally built its `cbit -> qubit` map up front with last-write-wins, so
/// `if(c==1) x q[1]` resolved its control to **q1** and the pass emitted
/// `CX q1,q1` — which panics the statevector backend with "control and target
/// must differ". In `omega-server` that is a panic inside a request handler.
///
/// The defect was invisible here because this helper hand-copied the fixture and
/// stopped one line early, so the ambiguous case could not arise in the only
/// test looking for it. `crates/omega-cli/tests/deferred_expectation.rs` now
/// reads the `.qasm` from disk for exactly this reason; the hand-built version
/// is kept because it is what the unit-level rules below are written against,
/// but it must stay faithful.
fn feedforward() -> CircuitIR {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.num_classical_bits = 1;
    c.ops.push(op(GateKind::H, &[0]));
    c.ops.push(measure(0, 0));
    c.ops.push(GateOp {
        gate: GateKind::X,
        qubits: [Qubit(1)].into_iter().collect(),
        params: Default::default(),
        classical_bit: None,
        condition: Some((0, 1, 1)),
    });
    c.ops.push(measure(1, 0));
    c
}

fn pauli(terms: &[(u32, PauliOp)]) -> Observable {
    Observable {
        terms: vec![(1.0, terms.to_vec())],
    }
}

#[test]
fn deferral_turns_the_classical_control_into_a_cx() {
    let d = defer_measurements(&feedforward()).expect("deferrable");
    assert_eq!(d.measured, vec![0], "q0's measurement was deferred");
    // No measurement and no condition survive: the control represents both.
    assert!(!d
        .circuit
        .ops
        .iter()
        .any(|o| matches!(o.gate, GateKind::Measure)));
    assert!(d.circuit.ops.iter().all(|o| o.condition.is_none()));
    let kinds: Vec<_> = d.circuit.ops.iter().map(|o| o.gate.clone()).collect();
    assert_eq!(kinds, vec![GateKind::H, GateKind::CX], "h q0; cx q0,q1");
    let cx = d.circuit.ops.last().unwrap();
    assert_eq!(
        cx.qubits.iter().map(|q| q.0).collect::<Vec<_>>(),
        vec![0, 1],
        "the measured qubit becomes the CONTROL"
    );
}

/// **The acceptance criterion.** Off-diagonal terms on the deferred qubit are
/// deleted; diagonal ones survive.
#[test]
fn dephasing_deletes_off_diagonal_terms_on_the_measured_qubit() {
    let d = defer_measurements(&feedforward()).expect("deferrable");

    // ⟨Z₀Z₁⟩ survives — both designs agree here, which is why it cannot be the
    // acceptance test on its own.
    let zz = pauli(&[(0, PauliOp::Z), (1, PauliOp::Z)]);
    assert_eq!(
        zz.dephase(&d.measured).terms.len(),
        1,
        "Z on a measured qubit must survive dephasing"
    );

    // ⟨X₀X₁⟩ and ⟨Y₀Y₁⟩ are DELETED, so their expectation is exactly 0 — the
    // mixture's value. A naive deferral would return the Bell values ±1.
    for p in [PauliOp::X, PauliOp::Y] {
        let o = pauli(&[(0, p), (1, PauliOp::Z)]);
        assert!(
            o.dephase(&d.measured).terms.is_empty(),
            "{p:?} on the measured qubit must be deleted; leaving it returns the \
             pure-Bell value where the truth is the mixture's 0"
        );
    }

    // X or Y on a qubit that was NOT measured is untouched — dephasing is
    // per-qubit, not a blanket diagonal projection.
    let x1 = pauli(&[(1, PauliOp::X)]);
    assert_eq!(
        x1.dephase(&d.measured).terms.len(),
        1,
        "X on an unmeasured qubit must survive"
    );
}

/// Mixed-weight terms: a term dies if ANY measured qubit carries X or Y in it.
#[test]
fn a_term_dies_if_any_measured_qubit_carries_x_or_y() {
    let o = Observable {
        terms: vec![
            (1.0, vec![(0, PauliOp::Z), (1, PauliOp::X)]), // survives: q0 is Z
            (2.0, vec![(0, PauliOp::X), (1, PauliOp::X)]), // dies: q0 is X
            (3.0, vec![(0, PauliOp::I), (1, PauliOp::Y)]), // survives: q0 is I
            (4.0, vec![(0, PauliOp::Y)]),                  // dies
        ],
    };
    let out = o.dephase(&[0]);
    let weights: Vec<f64> = out.terms.iter().map(|(w, _)| *w).collect();
    assert_eq!(
        weights,
        vec![1.0, 3.0],
        "only the q0-diagonal terms survive"
    );
}

/// Dephasing on no qubits is the identity — the non-deferred path must be
/// untouched, or every existing expectation changes.
#[test]
fn dephasing_nothing_changes_nothing() {
    let o = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::X)]), (2.0, vec![(1, PauliOp::Y)])],
    };
    assert_eq!(o.dephase(&[]).terms, o.terms);
}

/// **Reset-and-reuse is refused, and that is the common case not a corner.**
///
/// It is the QEC ancilla pattern — `repcode_d3r3_5q` in this repo recycles q3/q4
/// three times. Deferral would move the measurement past work that depends on it
/// having happened.
#[test]
fn reset_and_reuse_is_refused_by_name() {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.num_classical_bits = 1;
    c.ops.push(op(GateKind::H, &[0]));
    c.ops.push(GateOp {
        gate: GateKind::Measure,
        qubits: [Qubit(0)].into_iter().collect(),
        params: Default::default(),
        classical_bit: Some(0),
        condition: None,
    });
    c.ops.push(op(GateKind::Reset, &[0]));
    c.ops.push(op(GateKind::H, &[0]));

    match defer_measurements(&c) {
        Err(NotDeferrable::QubitReusedAfterMeasure { qubit, .. }) => assert_eq!(qubit, 0),
        other => panic!("reset-and-reuse must be refused, got {other:?}"),
    }
}

/// A whole-register condition does not become one quantum control.
#[test]
fn a_multi_bit_condition_is_refused() {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.num_classical_bits = 2;
    c.ops.push(op(GateKind::H, &[0]));
    c.ops.push(GateOp {
        gate: GateKind::Measure,
        qubits: [Qubit(0)].into_iter().collect(),
        params: Default::default(),
        classical_bit: Some(0),
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::X,
        qubits: [Qubit(1)].into_iter().collect(),
        params: Default::default(),
        classical_bit: None,
        condition: Some((0, 2, 2)), // if (c == 2) over 2 bits
    });
    match defer_measurements(&c) {
        Err(NotDeferrable::ConditionNotSingleBitTrue {
            num_bits, expected, ..
        }) => {
            assert_eq!((num_bits, expected), (2, 2));
        }
        other => panic!("a whole-register predicate must be refused, got {other:?}"),
    }
}

/// A gate with no controlled form is refused rather than approximated.
#[test]
fn a_conditioned_gate_without_a_controlled_form_is_refused() {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.num_classical_bits = 1;
    c.ops.push(op(GateKind::H, &[0]));
    c.ops.push(GateOp {
        gate: GateKind::Measure,
        qubits: [Qubit(0)].into_iter().collect(),
        params: Default::default(),
        classical_bit: Some(0),
        condition: None,
    });
    c.ops.push(GateOp {
        gate: GateKind::H, // no CH in this pass
        qubits: [Qubit(1)].into_iter().collect(),
        params: Default::default(),
        classical_bit: None,
        condition: Some((0, 1, 1)),
    });
    assert!(matches!(
        defer_measurements(&c),
        Err(NotDeferrable::NoControlledForm { .. })
    ));
}

/// **A control bit written twice must resolve to the write that PRECEDES the
/// read.** The regression test for the panic described on `feedforward()`.
///
/// Asserted as a property of the emitted gate — control ≠ target — rather than
/// only as "the value is right", because the wrong answer here is not a slightly
/// off number: it is an invalid gate that aborts the process.
#[test]
fn a_cbit_written_twice_resolves_causally_and_never_emits_a_self_control() {
    let d = defer_measurements(&feedforward()).expect("fixture 12 must be deferrable");

    assert_eq!(
        d.measured,
        vec![0],
        "only q0's measurement is consequential — it is read by the condition. \
         q1's terminal measurement is inert and must NOT dephase, or this \
         contradicts the Qiskit anchor, which strips terminal measures and \
         answers with the pure state. Got {:?}",
        d.measured
    );

    for gop in &d.circuit.ops {
        let qs: Vec<u32> = gop.qubits.iter().map(|q| q.0).collect();
        let mut seen = qs.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            qs.len(),
            "the pass emitted {:?} on qubits {qs:?} — a gate with a repeated \
             qubit. `apply_cx` panics on control == target, so this is a process \
             abort and, in omega-server, a panic inside a request handler.",
            gop.gate
        );
    }

    let cx: Vec<Vec<u32>> = d
        .circuit
        .ops
        .iter()
        .filter(|o| matches!(o.gate, GateKind::CX))
        .map(|o| o.qubits.iter().map(|q| q.0).collect())
        .collect();
    assert_eq!(
        cx,
        vec![vec![0, 1]],
        "the deferred control must come from q0, the qubit whose measurement the \
         condition actually reads — not from q1, which overwrote the same bit \
         afterwards"
    );
}

/// **An inert measurement is elided, not dephased.**
///
/// `h q0; cx q0,q1; measure q0 -> c0; measure q1 -> c1` with nothing reading
/// either bit. The measurements have no consequence for any observable of the
/// prepared state, and `remove_final_measurements()` on the Qiskit side answers
/// with the pure state — so `⟨X₀X₁⟩` must stay **+1** here.
///
/// This is the case that would have broken 13 of the 14 crosscheck fixtures if
/// the pass dephased on every measured qubit, and it would have broken them
/// against the external oracle rather than against ourselves.
#[test]
fn an_inert_measurement_is_elided_and_leaves_the_observable_alone() {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.num_classical_bits = 2;
    c.ops.push(op(GateKind::H, &[0]));
    c.ops.push(op(GateKind::CX, &[0, 1]));
    c.ops.push(measure(0, 0));
    c.ops.push(measure(1, 1));

    let obs = pauli(&[(0, PauliOp::X), (1, PauliOp::X)]);
    let (circ, prepared) = omega_core::defer_measure::prepare_for_expectation(&c, &obs)
        .expect("terminal measures are deferrable");

    assert!(
        !circ.ops.iter().any(|o| matches!(o.gate, GateKind::Measure)),
        "the measurements should have been dropped"
    );
    assert_eq!(
        prepared.terms.len(),
        1,
        "X₀X₁ must SURVIVE: no measurement here is read by anything, so the \
         value is a property of the prepared Bell state and equals +1. Dephasing \
         it to 0 would contradict `remove_final_measurements` on the Qiskit side \
         for 13 of the 14 crosscheck fixtures."
    );
}

/// **A conditioned measurement is refused.**
///
/// `if (c==1) measure q1 -> c1` is legal OpenQASM 2.0 and is a measurement that
/// happens on only some branches. The pass used to handle the `Measure` arm
/// before looking at the condition, so it dropped the op and dephased q1
/// unconditionally — answering a circuit that measures always, for one that
/// measures sometimes.
#[test]
fn a_conditioned_measurement_is_refused() {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.num_classical_bits = 2;
    c.ops.push(op(GateKind::H, &[0]));
    c.ops.push(measure(0, 0));
    let mut m = measure(1, 1);
    m.condition = Some((0, 1, 1));
    c.ops.push(m);

    match defer_measurements(&c) {
        Err(NotDeferrable::ConditionedMeasure { at_op }) => assert_eq!(at_op, 2),
        Err(other) => panic!("refused, but for the wrong reason: {other}"),
        Ok(d) => panic!(
            "a conditionally-performed measurement was accepted and turned into \
             an unconditional dephase on {:?}",
            d.measured
        ),
    }
}
