// SPDX-License-Identifier: Apache-2.0
//! `PauliBackend` collapse-mode counts must be keyed by the classical
//! register, not by a fresh measurement of every qubit.
//!
//! # The defect this pins
//!
//! The stabilizer backend evaluated classical conditions correctly — its
//! physics was right — and then, at the end of each shot, re-measured all `n`
//! qubits and used that as the counts key. So a 2-qubit circuit with a 1-bit
//! creg reported keys in `{0, 3}` where the creg values are `{0, 1}`.
//!
//! Two things made this hard to see. First, the backend agreed with itself
//! across runs, so no internal comparison flagged it. Second, any consumer that
//! formatted the key to creg width would **truncate** 3 to 1 and read the
//! result as correct — which is exactly what the N-way counts matrix did on its
//! first run, reporting `pauli` as fully in agreement while it was emitting
//! keys too wide to be creg values at all.
//!
//! Found via `crates/omega-cli/tests/nway_counts.rs`; the matrix now asserts
//! that every key fits its declared width rather than silently truncating.

use omega_backend_pauli::PauliBackend;
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

/// `h q0; measure q0 -> c0; if (c==1) x q1; measure q1 -> c0` — Clifford, so
/// the stabilizer backend can express it exactly.
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

fn run_collapse(
    ir: &CircuitIR,
    seed: u64,
) -> std::collections::HashMap<omega_core::outcome::Outcome, u32> {
    let config = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    match PauliBackend::new()
        .execute(ir, &ParameterBinding::new(), &config)
        .expect("pauli execute")
    {
        ExecResult::Counts(c) => c,
        _ => panic!("expected counts"),
    }
}

/// Every key must fit the 1-bit creg. Pre-fix this saw key 3.
#[test]
fn collapse_mode_keys_fit_the_creg_width() {
    let counts = run_collapse(&feedforward_circuit(), 7);
    for key in counts.keys() {
        assert!(
            key.width() == 1 && key.as_u64().unwrap_or(u64::MAX) <= 1,
            "key |{}> ({} bits) does not fit a 1-bit creg — the backend is \
             keying over the qubit register. Full counts: {counts:?}",
            key.to_bitstring(),
            key.width()
        );
    }
}

/// And the distribution is still the right one: c0 ends at 1 about half the
/// time. Keying correctly but computing the wrong physics would be no better.
#[test]
fn collapse_mode_distribution_matches_the_guard() {
    let counts = run_collapse(&feedforward_circuit(), 7);
    let total: u32 = counts.values().sum();
    assert_eq!(total, SHOTS);
    // Keys carry their own width now; build the probe at the creg's.
    let w = counts.keys().next().map(|o| o.width()).unwrap_or(1);
    let p1 = *counts
        .get(&omega_core::outcome::Outcome::from_u64(1, w))
        .unwrap_or(&0) as f64
        / total as f64;
    // Qiskit Aer on the same circuit: {"0": 9895, "1": 10105} at 20000 shots.
    // sd at p=1/2 is 0.0035, so ±0.02 is > 5 sigma.
    assert!(
        (p1 - 0.5).abs() < 0.02,
        "P(c0=1) = {p1:.4}, expected ~0.5 — got {counts:?}"
    );
}

/// Skip mode is untouched: no creg keying, counts stay over the qubit
/// register. Guards the fix against over-reach.
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
    let ExecResult::Counts(counts) = PauliBackend::new()
        .execute(&ir, &ParameterBinding::new(), &config)
        .expect("pauli execute")
    else {
        panic!("expected counts")
    };
    // q1 = 1, q0 = 0 → basis index 0b10 = 2, over the QUBIT register.
    let w = counts.keys().next().map(|o| o.width()).unwrap_or(2);
    assert_eq!(
        counts
            .get(&omega_core::outcome::Outcome::from_u64(2, w))
            .copied()
            .unwrap_or(0),
        1024,
        "got {counts:?}"
    );
}
