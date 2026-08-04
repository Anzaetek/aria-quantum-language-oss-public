//! `Reset` must implement the reset CHANNEL, not a coherent fold.
//!
//! `reset q` means "discard qubit q and substitute a fresh |0⟩". If q was
//! entangled, that entanglement must be **destroyed**, not transferred:
//!
//! ```text
//! ρ  →  |0⟩⟨0|_q ⊗ Tr_q(ρ)
//! ```
//!
//! Ground truth for `Bell(q0,q1); reset q0`, from Qiskit Aer's exact
//! `DensityMatrix` (qiskit 2.4.1 / aer 0.17.2):
//!
//! ```text
//! rho    = diag(0.5, 0, 0.5, 0)
//! rho_q1 = I/2            ⟨X₁⟩ = ⟨Y₁⟩ = ⟨Z₁⟩ = 0     ⟨Z₀⟩ = +1
//! ```
//!
//! q1 is left **maximally mixed** — a coin flip in every basis. A statevector
//! cannot hold that, so the ensemble has to come from independent per-shot
//! trajectories; `apply_reset` samples, projects, and flips accordingly.
//!
//! These three gates are the ones that discriminate a real reset from the
//! plausible near-misses. Each was verified against Aer's statevector,
//! stabilizer and matrix_product_state methods, which all agree:
//!
//! | gate | correct | coherent fold | post-selection |
//! |---|---|---|---|
//! | A  measure q1              | ~50/50 | ~50/50 ✓ | 0% ✗ |
//! | B  H q1 then measure q1    | ~50/50 | 0% ✗     | ~50/50 ✓ |
//! | C  reset \|−⟩ then measure | 0%     | 100% ✗   | 0% ✓ |
//!
//! No single gate catches every wrong implementation — A and B disagree in
//! *different* bases, which is exactly why the fold survived so long. Keep all
//! three.

use std::collections::HashMap;

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

const SHOTS: u32 = 4000;
/// ±5σ on 4000 Bernoulli(0.5) draws is ±158; 250 keeps the gate meaningful
/// (a wrong implementation misses by ~2000) without being seed-fragile.
const BAND: u32 = 250;

fn g(kind: GateKind, qubits: &[u32], bit: Option<u32>) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: smallvec![],
        classical_bit: bit,
        condition: None,
    }
}

fn counts(c: &CircuitIR, seed: u64) -> HashMap<u64, u32> {
    let cfg = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    StatevectorBackend::new()
        .execute(c, &ParameterBinding::new(), &cfg)
        .expect("execute")
        .counts()
        .clone()
}

/// Shots whose key has `bit` set.
fn ones(m: &HashMap<u64, u32>, bit: u64) -> u32 {
    m.iter().filter(|(k, _)| *k & bit != 0).map(|(_, v)| v).sum()
}

fn bell_reset0(n: u32) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[0], None));
    c.add_op(g(GateKind::CX, &[0, 1], None));
    c.add_op(g(GateKind::Reset, &[0], None));
    c
}

/// A — Bell, reset q0, measure q1 in Z. Correct: q1 is maximally mixed, 50/50.
/// Post-selection fails this (it would collapse q1 to |0⟩).
#[test]
fn reset_leaves_partner_mixed_in_z() {
    let m = counts(&bell_reset0(2), 7);
    let k = ones(&m, 0b10);
    assert!(
        k.abs_diff(SHOTS / 2) < BAND,
        "⟨Z₁⟩ after Bell+reset(q0): got {k}/{SHOTS} ones, want ~{}/{SHOTS} \
         (q1 must be maximally mixed, not collapsed)",
        SHOTS / 2
    );
}

/// B — Bell, reset q0, H q1, measure q1. Still 50/50: H maps I/2 to I/2.
/// The coherent fold fails this — it leaves q1 in |+⟩, so H gives |0⟩ always.
#[test]
fn reset_leaves_partner_mixed_in_x() {
    let mut c = bell_reset0(2);
    c.add_op(g(GateKind::H, &[1], None));
    let m = counts(&c, 7);
    let k = ones(&m, 0b10);
    assert!(
        k.abs_diff(SHOTS / 2) < BAND,
        "⟨X₁⟩ after Bell+reset(q0): got {k}/{SHOTS} ones, want ~{}/{SHOTS} \
         (a coherent fold would leave q1 = |+⟩ and give 0)",
        SHOTS / 2
    );
}

/// C — reset a qubit in |−⟩ = (|0⟩−|1⟩)/√2. It must read 0 with certainty.
/// The fold cancelled the two amplitudes to zero and left an all-zero state,
/// which sampled as |1⟩ on *every* shot.
#[test]
fn reset_of_minus_state_reads_zero() {
    let mut c = CircuitIR::new(1, CircuitType::GateBased);
    c.add_op(g(GateKind::X, &[0], None));
    c.add_op(g(GateKind::H, &[0], None)); // |−⟩
    c.add_op(g(GateKind::Reset, &[0], None));
    let m = counts(&c, 7);
    assert_eq!(
        ones(&m, 0b1),
        0,
        "reset(|−⟩) must give |0⟩ with certainty, got {m:?}"
    );
}

/// The reset qubit itself is |0⟩ on every trajectory, whatever it was before.
#[test]
fn reset_qubit_is_zero_on_every_shot() {
    let m = counts(&bell_reset0(2), 11);
    assert_eq!(
        ones(&m, 0b1),
        0,
        "q0 must be |0⟩ on every shot after reset, got {m:?}"
    );
}

/// Analytic expectation of a reset on an ENTANGLED qubit is a mixed-state
/// quantity and a statevector holds one trajectory, so it must refuse rather
/// than return a silently RNG-dependent number.
#[test]
fn analytic_expectation_refuses_entangled_reset() {
    let obs = Observable {
        terms: vec![(1.0, vec![(1, PauliOp::X)])],
    };
    let err = StatevectorBackend::new()
        .expectation(&bell_reset0(2), &ParameterBinding::new(), &obs)
        .expect_err("analytic expectation over Reset must refuse");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Reset") && msg.contains("entangled"),
        "refusal should name Reset and say why, got: {msg}"
    );
}

/// ...but a reset on an UNENTANGLED qubit is deterministic — projecting onto
/// |0⟩, or onto |1⟩ and flipping, give the same state — so the analytic answer
/// is exact and must still be served. X|0⟩ = |1⟩, reset → |0⟩, ⟨Z⟩ = +1.
#[test]
fn analytic_expectation_allows_unentangled_reset() {
    let mut c = CircuitIR::new(1, CircuitType::GateBased);
    c.add_op(g(GateKind::X, &[0], None));
    c.add_op(g(GateKind::Reset, &[0], None));
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };
    let v = StatevectorBackend::new()
        .expectation(&c, &ParameterBinding::new(), &obs)
        .expect("unentangled reset is deterministic — must not refuse");
    assert!((v - 1.0).abs() < 1e-12, "⟨Z⟩ after reset = {v}, want +1");
}

/// Same, but the qubit is in a superposition (|−⟩) yet still unentangled:
/// reset is deterministic, so this is exact too.
#[test]
fn analytic_expectation_allows_unentangled_superposition_reset() {
    let mut c = CircuitIR::new(1, CircuitType::GateBased);
    c.add_op(g(GateKind::X, &[0], None));
    c.add_op(g(GateKind::H, &[0], None)); // |−⟩
    c.add_op(g(GateKind::Reset, &[0], None));
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };
    let v = StatevectorBackend::new()
        .expectation(&c, &ParameterBinding::new(), &obs)
        .expect("unentangled reset is deterministic — must not refuse");
    assert!((v - 1.0).abs() < 1e-12, "⟨Z⟩ after reset(|−⟩) = {v}, want +1");
}
