//! `Reset` must implement the reset CHANNEL on the stabilizer backend too.
//!
//! Same three discriminating gates and the same Qiskit Aer ground truth as
//! `omega-backend-statevector/tests/reset_channel.rs` — see that file for the
//! derivation. Aer's `stabilizer` method agrees with its statevector and
//! matrix_product_state methods on all three.
//!
//! Gate A is the one this backend used to fail: forcing the measured outcome to
//! 0 (post-selection) instead of sampling drove the Bell partner to |0⟩ as
//! well, so q1 read 0 on every shot where the channel gives 50/50.

use std::collections::HashMap;

use omega_backend_pauli::PauliBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

const SHOTS: u32 = 4000;
const BAND: u32 = 250;

fn g(kind: GateKind, qubits: &[u32]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }
}

fn counts(c: &CircuitIR, seed: u64) -> HashMap<omega_core::outcome::Outcome, u32> {
    let cfg = ExecConfig {
        shots: Some(SHOTS),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    PauliBackend::new()
        .execute(c, &ParameterBinding::new(), &cfg)
        .expect("execute")
        .counts()
        .clone()
}

fn ones(m: &HashMap<omega_core::outcome::Outcome, u32>, bit: u64) -> u32 {
    // `bit` is a MASK in the old u64 convention; read the matching index.
    let idx = bit.trailing_zeros();
    m.iter()
        .filter(|(k, _)| k.bit(idx) == 1)
        .map(|(_, v)| v)
        .sum()
}

fn bell_reset0() -> CircuitIR {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[0]));
    c.add_op(g(GateKind::CX, &[0, 1]));
    c.add_op(g(GateKind::Reset, &[0]));
    c
}

/// A — partner is maximally mixed in Z. This is the post-selection gate.
#[test]
fn reset_leaves_partner_mixed_in_z() {
    let k = ones(&counts(&bell_reset0(), 7), 0b10);
    assert!(
        k.abs_diff(SHOTS / 2) < BAND,
        "stabilizer: got {k}/{SHOTS} ones on q1, want ~{} — forcing the reset \
         outcome to 0 post-selects and collapses the partner too",
        SHOTS / 2
    );
}

/// B — and in X.
#[test]
fn reset_leaves_partner_mixed_in_x() {
    let mut c = bell_reset0();
    c.add_op(g(GateKind::H, &[1]));
    let k = ones(&counts(&c, 7), 0b10);
    assert!(
        k.abs_diff(SHOTS / 2) < BAND,
        "stabilizer: got {k}/{SHOTS} ones on q1 after H, want ~{}",
        SHOTS / 2
    );
}

/// C — reset of |−⟩ must read 0 with certainty.
#[test]
fn reset_of_minus_state_reads_zero() {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(g(GateKind::X, &[0]));
    c.add_op(g(GateKind::H, &[0]));
    c.add_op(g(GateKind::Reset, &[0]));
    let m = counts(&c, 7);
    assert_eq!(
        ones(&m, 0b1),
        0,
        "stabilizer: reset(|−⟩) must be |0⟩, got {m:?}"
    );
}

/// The reset qubit is |0⟩ on every trajectory.
#[test]
fn reset_qubit_is_zero_on_every_shot() {
    let m = counts(&bell_reset0(), 11);
    assert_eq!(ones(&m, 0b1), 0, "stabilizer: q0 must be |0⟩, got {m:?}");
}

/// Analytic expectation over a Reset circuit is refused, not guessed.
#[test]
fn analytic_expectation_refuses_reset() {
    let obs = Observable {
        terms: vec![(1.0, vec![(1, PauliOp::Z)])],
    };
    let err = PauliBackend::new()
        .expectation(&bell_reset0(), &ParameterBinding::new(), &obs)
        .expect_err("analytic expectation over Reset must refuse");
    assert!(format!("{err:?}").contains("Reset"));
}
