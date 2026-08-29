// SPDX-License-Identifier: Apache-2.0
//! The principle of deferred measurement, as a circuit pass — **and only half of
//! the transformation.**
//!
//! # Read this before using it
//!
//! Deferring a measurement turns classical control into quantum control. That
//! step alone is **not** equivalent, and using it alone produces a *wrong
//! answer* rather than an approximation.
//!
//! `h q0; measure q0 -> c0; if(c==1) x q1`:
//!
//! | | state | `⟨Z₀Z₁⟩` | `⟨X₀X₁⟩` |
//! |---|---|---|---|
//! | truth (collapse) | mixture ½\|00⟩ + ½\|11⟩ | +1 | **0** |
//! | deferred only | pure Bell | +1 | **+1** |
//!
//! They agree on the diagonal observable and disagree off-diagonal. The
//! measurement is what recovers the mixture, so **the caller must dephase the
//! observable on every qubit in [`Deferred::measured`]** — see
//! [`crate::executor::Observable::dephase`]. The two are one transformation
//! split across two types, and this module's return value names the qubits
//! precisely so the second half cannot be forgotten silently.
//!
//! Verified before either half was written: `proofs/lean4/QuantumProofs/
//! DeferProbe.lean` (`trace_deph_left_eq_right`, axiom-clean) and
//! `AerSimulator(method="density_matrix")` on the fixture above.
//!
//! # Not every measurement is deferred: consequential vs INERT
//!
//! A measurement that nothing depends on — no later gate conditioned on the bit
//! it wrote, no later operation on the qubit — has no effect on any observable
//! of the *prepared state*, and is **elided** rather than dephased.
//!
//! That is not a shortcut, it is the contract, and getting it wrong was the
//! first draft's worst mistake. The external anchor
//! (`omega-bridges/python/qiskit_runner.py`) calls
//! `remove_final_measurements()` and then answers with the **pure** statevector,
//! and **all 14** crosscheck fixtures contain `measure`. Dephasing on inert
//! measurements would have contradicted the oracle on 13 of them — a
//! systematically wrong reference, which is worse than no reference. It would
//! also mean that adding a readout line to a VQE ansatz silently changes its
//! energy.
//!
//! So [`Deferred::measured`] holds exactly the qubits whose measurement was
//! **used as a control**. Everything else is dropped, and a circuit whose
//! measurements are all inert comes out of this pass with the same values it had
//! before it existed.
//!
//! # What is refused, and why refusing is the point
//!
//! Deferral is valid only if the measured qubit is not used coherently after the
//! measurement. **Reset-and-reuse is therefore not deferrable** — and that is
//! the ancilla-recycling pattern in every QEC circuit, including this repo's own
//! `repcode_d3r3_5q` fixture, so it is the common case rather than a corner.
//! Refused by name.

use crate::circuit::{CircuitIR, GateKind, GateOp, Qubit};

/// A circuit with its mid-circuit measurements deferred, plus the qubits whose
/// measurement moved — which the caller **must** dephase the observable on.
#[derive(Clone, Debug)]
pub struct Deferred {
    pub circuit: CircuitIR,
    /// Qubits whose measurement was **used as a control** — the consequential
    /// ones. Inert measurements are elided and do not appear here. Pass these to
    /// [`crate::executor::Observable::dephase`]; skipping that step is the wrong
    /// answer documented at the module level, not a missing optimisation.
    pub measured: Vec<u32>,
}

/// Why a circuit cannot be deferred. A refusal names the construct, because a
/// caller can act on that and not on "unsupported".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotDeferrable {
    /// The measured qubit is operated on coherently afterwards — e.g.
    /// reset-and-reuse. Deferral would move a measurement past work that
    /// depends on its having happened.
    QubitReusedAfterMeasure { qubit: u32, at_op: usize },
    /// A conditioned gate this pass has no controlled form for.
    NoControlledForm { gate: String, at_op: usize },
    /// A condition on a multi-bit register, or comparing against something
    /// other than 1. `if (c == 2)` over a 2-bit register is a predicate on the
    /// whole register and does not become a single quantum control.
    ConditionNotSingleBitTrue {
        num_bits: u32,
        expected: u64,
        at_op: usize,
    },
    /// A conditioned gate whose control bit was never written by a measurement
    /// in this circuit. Deferral has nothing to hang the control on.
    ConditionBitNeverMeasured { cbit: u32, at_op: usize },
    /// A `Measure` that is itself classically conditioned — `if (c==1) measure
    /// q1 -> c1`. Legal OpenQASM 2.0, and a measurement that only *sometimes*
    /// happens. Deferring it would dephase unconditionally, which answers a
    /// different circuit.
    ConditionedMeasure { at_op: usize },
}

impl std::fmt::Display for NotDeferrable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotDeferrable::QubitReusedAfterMeasure { qubit, at_op } => write!(
                f,
                "qubit {qubit} is measured and then used again at op {at_op} \
                 (e.g. reset-and-reuse). Deferral would move the measurement past \
                 work that depends on it having happened — this is the QEC \
                 ancilla-recycling pattern and it is genuinely not deferrable"
            ),
            NotDeferrable::NoControlledForm { gate, at_op } => write!(
                f,
                "the conditioned gate {gate} at op {at_op} has no controlled form \
                 in this pass. Refusing rather than approximating"
            ),
            NotDeferrable::ConditionNotSingleBitTrue {
                num_bits,
                expected,
                at_op,
            } => write!(
                f,
                "the condition at op {at_op} tests {num_bits} bit(s) against \
                 {expected}. Only a single bit compared to 1 becomes one quantum \
                 control; a whole-register predicate does not"
            ),
            NotDeferrable::ConditionBitNeverMeasured { cbit, at_op } => write!(
                f,
                "the gate at op {at_op} is conditioned on classical bit {cbit}, \
                 which no measurement in this circuit writes — there is no qubit \
                 to take the control from"
            ),
            NotDeferrable::ConditionedMeasure { at_op } => write!(
                f,
                "the measurement at op {at_op} is itself classically conditioned, \
                 so it happens on only some branches. Deferring it would dephase \
                 the qubit unconditionally — a different circuit"
            ),
        }
    }
}

/// The controlled form of a single-qubit gate, or `None` if this pass has none.
///
/// Deliberately small. `X` and `Z` are what conditioned circuits in this tree
/// actually use, and inventing a decomposition for the rest would be shipping
/// untested arithmetic behind a pass whose correctness argument is about
/// something else entirely.
fn controlled_form(gate: &GateKind) -> Option<GateKind> {
    match gate {
        GateKind::X => Some(GateKind::CX),
        GateKind::Z => Some(GateKind::CZ),
        GateKind::Y => Some(GateKind::CY),
        _ => None,
    }
}

/// Defer every mid-circuit measurement, or refuse naming the reason.
///
/// On success the returned circuit has **no** `Measure` ops and no conditions;
/// the measurements are represented by the controls, and the *dephasing* that
/// restores the mixture is the caller's second step.
pub fn defer_measurements(circuit: &CircuitIR) -> Result<Deferred, NotDeferrable> {
    // Precondition, checked before anything is rewritten so that a refusal is
    // cheap: a measured qubit must not be operated on afterwards. A second
    // measurement of the same qubit is idempotent in the computational basis and
    // is allowed; a coherent gate, or a reset, is not deferrable at all.
    for (i, op) in circuit.ops.iter().enumerate() {
        if !matches!(op.gate, GateKind::Measure) {
            continue;
        }
        // Checked BEFORE the measurement is otherwise handled. A conditioned
        // measurement happens on only some branches, and the earlier draft of
        // this pass dropped it and dephased unconditionally.
        if op.condition.is_some() {
            return Err(NotDeferrable::ConditionedMeasure { at_op: i });
        }
        let Some(mq) = op.qubits.first().map(|q| q.0) else {
            continue;
        };
        for (j, later) in circuit.ops.iter().enumerate().skip(i + 1) {
            if matches!(later.gate, GateKind::Id | GateKind::Barrier) {
                continue;
            }
            if matches!(later.gate, GateKind::Measure)
                && later.qubits.first().map(|q| q.0) == Some(mq)
            {
                continue;
            }
            if later.qubits.iter().any(|q| q.0 == mq) {
                return Err(NotDeferrable::QubitReusedAfterMeasure {
                    qubit: mq,
                    at_op: j,
                });
            }
        }
    }

    // `written_by` maps cbit -> qubit and is built **as the ops are walked**, so
    // a condition resolves to the measurement that most recently wrote its bit.
    //
    // Building it up front with last-write-wins was a real defect and not a
    // hypothetical one. `12_feedforward_sometimes_false.qasm` ends with a SECOND
    // `measure q[1] -> c[0]`, so `if(c==1) x q[1]` resolved its control to q1
    // and this pass emitted `CX q1,q1` — which panics the statevector backend
    // with "control and target must differ". The unit test missed it by
    // hand-copying the fixture without its final line, so the ambiguous case
    // could not arise in the only place that looked for it.
    let mut written_by: std::collections::HashMap<u32, u32> = Default::default();
    let mut out: Vec<GateOp> = Vec::with_capacity(circuit.ops.len());
    let mut measured: Vec<u32> = Vec::new();

    for (i, op) in circuit.ops.iter().enumerate() {
        if matches!(op.gate, GateKind::Measure) {
            if let (Some(cbit), Some(q)) = (op.classical_bit, op.qubits.first()) {
                written_by.insert(cbit, q.0);
            }
            // The op is dropped either way. Whether the qubit is DEPHASED is
            // decided below, by whether a later condition actually reads it —
            // an inert measurement changes no observable of the prepared state.
            continue;
        }

        match op.condition {
            None => out.push(op.clone()),
            Some((cbit, num_bits, expected)) => {
                if num_bits != 1 || expected != 1 {
                    return Err(NotDeferrable::ConditionNotSingleBitTrue {
                        num_bits,
                        expected,
                        at_op: i,
                    });
                }
                let Some(&ctrl) = written_by.get(&cbit) else {
                    return Err(NotDeferrable::ConditionBitNeverMeasured { cbit, at_op: i });
                };
                let Some(cgate) = controlled_form(&op.gate) else {
                    return Err(NotDeferrable::NoControlledForm {
                        gate: format!("{:?}", op.gate),
                        at_op: i,
                    });
                };
                // A control that is also a target would be an invalid gate.
                // Unreachable through the causal map above — the precondition
                // already refuses a measured qubit being touched again — but a
                // silently invalid circuit is worse than a refusal, so this is
                // checked rather than assumed.
                if op.qubits.iter().any(|q| q.0 == ctrl) {
                    return Err(NotDeferrable::QubitReusedAfterMeasure {
                        qubit: ctrl,
                        at_op: i,
                    });
                }
                // THIS is what makes the measurement consequential.
                if !measured.contains(&ctrl) {
                    measured.push(ctrl);
                }
                let mut qubits = smallvec::SmallVec::<[Qubit; 3]>::new();
                qubits.push(Qubit(ctrl));
                for q in op.qubits.iter() {
                    qubits.push(*q);
                }
                out.push(GateOp {
                    gate: cgate,
                    qubits,
                    params: op.params.clone(),
                    classical_bit: None,
                    condition: None,
                });
            }
        }
    }

    let mut c = circuit.clone();
    c.ops = out;
    Ok(Deferred {
        circuit: c,
        measured,
    })
}

/// Prepare a circuit and observable for an **expectation**.
///
/// This is the single definition of what `⟨O⟩` means for a circuit that
/// measures, and every backend's expectation path calls it.
///
/// * no `Measure` at all → the inputs are returned unchanged, so the
///   overwhelmingly common path is bit-for-bit what it was;
/// * measurements all inert → they are dropped and the observable is untouched,
///   which is what `remove_final_measurements` does on the Qiskit side;
/// * a measurement used as a classical control → deferred to a controlled gate,
///   and the observable dephased on that qubit;
/// * not deferrable → `Err`, and the caller **must** propagate it.
///
/// # Why this is one function and not a rule repeated at every call site
///
/// There are more than a dozen `Backend` expectation implementations —
/// statevector, MPS, pauliprop, the Clifford tableau, photonics, Metal, CUDA,
/// OpenCL, Torch — times `expectation`, `expectation_multi`,
/// `expectation_batch` and the fused gradient entry points. A rule copied into a
/// dozen places holds in eleven of them after the next change, and the
/// divergence is invisible from inside any one backend: each returns a
/// self-consistent number, and only a cross-engine comparison notices.
///
/// Convention is not enough at that width, so the rule lives here and the
/// enforcement is a test that runs the feedforward fixture through every backend
/// the build has.
///
/// # What the caller must NOT do
///
/// Swallow the error and fall back to the old behaviour. Returning the unitary
/// value for a circuit whose measurement has consequences is the defect this
/// exists to remove: it is a plausible number that is silently wrong, on the
/// path QML uses.
/// # Why the refusal is `Unsupported` and not `InvalidCircuit`
///
/// The N-way comparison lane maps `OmegaError::Unsupported` to `CannotExpress`
/// and everything else to a hard `Error` that reddens the engine's row
/// (`omega-cli/tests/nway_expectation.rs`). A circuit this pass cannot defer is
/// not malformed — it is a circuit this engine cannot express an expectation
/// for, which is exactly what that cell means. Choosing the wrong variant would
/// turn an honest abstention into a spurious failure.
pub fn prepare_for_expectation(
    circuit: &CircuitIR,
    observable: &crate::executor::Observable,
) -> crate::error::Result<(CircuitIR, crate::executor::Observable)> {
    if !circuit
        .ops
        .iter()
        .any(|op| matches!(op.gate, GateKind::Measure))
    {
        return Ok((circuit.clone(), observable.clone()));
    }
    let deferred = defer_measurements(circuit).map_err(|why| {
        crate::error::OmegaError::Unsupported(format!(
            "expectation is not defined for this circuit: {why}"
        ))
    })?;
    let dephased = observable.dephase(&deferred.measured);
    Ok((deferred.circuit, dephased))
}

/// [`prepare_for_expectation`] for the `expectation_multi` entry points.
///
/// One deferral, applied once, and every observable dephased on the **same**
/// qubit set. Preparing each observable independently would run the pass N times
/// for one answer, and — worse — would let two observables of the same call
/// disagree about which qubits were measured if the pass ever became
/// observable-dependent.
pub fn prepare_for_expectation_multi(
    circuit: &CircuitIR,
    observables: &[crate::executor::Observable],
) -> crate::error::Result<(CircuitIR, Vec<crate::executor::Observable>)> {
    if !circuit
        .ops
        .iter()
        .any(|op| matches!(op.gate, GateKind::Measure))
    {
        return Ok((circuit.clone(), observables.to_vec()));
    }
    let deferred = defer_measurements(circuit).map_err(|why| {
        crate::error::OmegaError::Unsupported(format!(
            "expectation is not defined for this circuit: {why}"
        ))
    })?;
    let dephased = observables
        .iter()
        .map(|o| o.dephase(&deferred.measured))
        .collect();
    Ok((deferred.circuit, dephased))
}
