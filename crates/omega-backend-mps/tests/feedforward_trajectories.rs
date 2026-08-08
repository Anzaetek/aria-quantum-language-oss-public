// SPDX-License-Identifier: Apache-2.0
//! `MpsBackend` must run one independent trajectory per shot whenever the
//! evolution is stochastic — including `Collapse`-mode mid-circuit measurement,
//! not only `Reset`.
//!
//! # The defect these pin
//!
//! `MpsBackend::execute` guarded its per-shot trajectory loop on
//! `circuit_has_reset(circuit)` alone. A circuit whose only stochasticity was a
//! collapse-mode `measure` therefore took the single-evolution path: **one**
//! trajectory ran, the measurement drew **one** outcome, and all `shots`
//! samples came from that one collapsed chain. A fair coin reported as
//! certainty.
//!
//! This is the same defect the CPU statevector backend fixed in `11888a9` and
//! that `NoisyMpsBackend` already guarded with `mps_collapses`. It never
//! propagated to the noiseless MPS backend, and nothing caught it: the two MPS
//! backends were never compared to each other on a conditional circuit, and
//! the noiseless one agreed with itself perfectly every time.
//!
//! It surfaced only when the N-way counts matrix
//! (`crates/omega-cli/tests/nway_counts.rs`) put it beside Qiskit.
//!
//! # Why these tests can fail
//!
//! Each was run against the pre-fix backend. `feedforward_is_not_deterministic`
//! saw `{0: 20000}`; `collapse_mode_keys_by_the_creg` saw keys wider than the
//! creg. Both are checked here in a form that the broken code cannot satisfy —
//! a distribution assertion, not a smoke test.

use std::collections::HashMap;

use omega_backend_mps::MpsBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

const SHOTS: u32 = 20_000;

fn op(gate: GateKind, qubits: &[u32], cbit: Option<u32>) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
        params: smallvec::smallvec![],
        classical_bit: cbit,
        condition: None,
    }
}

/// `h q0; measure q0 -> c0; if (c==1) x q1; measure q1 -> c0`
///
/// The `12_feedforward_sometimes_false.qasm` fixture. The guard is true on
/// about half the shots, so c0 ends at 1 about half the time.
fn feedforward_circuit() -> CircuitIR {
    let mut guarded_x = op(GateKind::X, &[1], None);
    guarded_x.condition = Some((0, 1, 1));

    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.num_classical_bits = 1;
    ir.ops = vec![
        op(GateKind::H, &[0], None),
        op(GateKind::Measure, &[0], Some(0)),
        guarded_x,
        op(GateKind::Measure, &[1], Some(0)),
    ];
    ir
}

fn run(ir: &CircuitIR, seed: u64) -> HashMap<u64, u32> {
    let config = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let res = MpsBackend::new(64)
        .execute(ir, &ParameterBinding::new(), &config)
        .expect("mps execute");
    match res {
        ExecResult::Counts(c) => c,
        _ => panic!("expected counts"),
    }
}

/// The headline: a superposition must not be reported as certainty.
///
/// Pre-fix this returned `{0: 20000}` at seed 7 — one trajectory's draw,
/// replayed 20000 times. Qiskit Aer on the same circuit gives ~50/50
/// (`{"0": 9895, "1": 10105}` at 20000 shots, measured).
#[test]
fn feedforward_is_not_deterministic() {
    let ir = feedforward_circuit();
    // Several seeds: a single seed could in principle draw a genuinely
    // lopsided sample, and more importantly the broken code's output depended
    // entirely on which way its ONE draw went — it would produce `{0: 20000}`
    // on some seeds and `{1: 20000}` on others. Both must fail.
    for seed in [7u64, 11, 12345, 0xDEADBEEF] {
        let counts = run(&ir, seed);
        assert_eq!(
            counts.len(),
            2,
            "seed {seed}: expected both outcomes, got {counts:?} — a single \
             outcome means every shot re-sampled ONE trajectory"
        );
        let total: u32 = counts.values().sum();
        assert_eq!(total, SHOTS);
        let p1 = *counts.get(&1).unwrap_or(&0) as f64 / total as f64;
        // 20000 shots, p = 1/2: sd = 0.0035, so ±0.02 is > 5 sigma.
        assert!(
            (p1 - 0.5).abs() < 0.02,
            "seed {seed}: P(c0=1) = {p1:.4}, expected ~0.5 — got {counts:?}"
        );
    }
}

/// Collapse-mode counts are keyed by the classical register, so a 1-bit creg
/// yields keys in `{0, 1}` however many qubits the circuit has.
///
/// Pre-fix the backend re-sampled the full 2-qubit chain, so keys could reach
/// 3. That is not merely a relabelling: it reports over a register the program
/// never asked about, and it silently disagrees with Qiskit, which reports over
/// the creg.
#[test]
fn collapse_mode_keys_by_the_creg() {
    let counts = run(&feedforward_circuit(), 7);
    for key in counts.keys() {
        assert!(
            *key <= 1,
            "key {key} exceeds the 1-bit creg — counts are keyed over the \
             qubit register, not the classical one. Full counts: {counts:?}"
        );
    }
}

/// The guard must actually be evaluated. With the condition removed the `X`
/// fires unconditionally and c0 ends at 1 on every shot — so this asserts the
/// fixture is discriminating, not just that the backend produces two outcomes
/// for some other reason.
#[test]
fn dropping_the_guard_changes_the_distribution() {
    let mut ir = feedforward_circuit();
    for op in &mut ir.ops {
        op.condition = None;
    }
    let counts = run(&ir, 7);
    assert_eq!(
        counts.get(&1).copied().unwrap_or(0),
        SHOTS,
        "with the guard removed the X always fires, so c0 must be 1 on every \
         shot; got {counts:?}. If this ever returns ~50/50 the conditional \
         test above has stopped discriminating."
    );
}

/// Skip mode is unaffected: no collapse, no creg keying, counts stay over the
/// qubit register. Guards the fix against over-reach.
#[test]
fn skip_mode_still_keys_by_the_qubit_register() {
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.num_classical_bits = 1;
    ir.ops = vec![
        op(GateKind::X, &[1], None),
        op(GateKind::Measure, &[1], Some(0)),
    ];
    let config = ExecConfig {
        shots: Some(1024),
        seed: Some(3),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let res = MpsBackend::new(64)
        .execute(&ir, &ParameterBinding::new(), &config)
        .expect("mps execute");
    let ExecResult::Counts(counts) = res else {
        panic!("expected counts")
    };
    // q1 = 1, q0 = 0 → basis index 0b10 = 2, over the QUBIT register.
    assert_eq!(counts.get(&2).copied().unwrap_or(0), 1024, "got {counts:?}");
}
