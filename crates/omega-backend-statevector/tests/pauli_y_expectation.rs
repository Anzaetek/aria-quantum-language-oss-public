//! Regression tests for the odd-Y Pauli-expectation sign bug.
//!
//! `expectation_pauli` used to pair the basis-derived Y phase with the
//! wrong conjugation side, silently negating every Pauli string with an
//! ODD number of Y factors. Every shipped observable happened to have
//! even Y-parity (Z-strings, XX+YY Hamiltonians), so it survived until
//! the Clifford-suffix parallel-shift conjugation produced single-Y
//! gradient observables. These tests pin the textbook convention with
//! closed-form states.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn gate(kind: GateKind, qubits: &[u32], params: &[ParamExpr]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().cloned().collect(),
        classical_bit: None,
        condition: None,
    }
}

fn expect(circuit: &CircuitIR, obs: &Observable) -> f64 {
    StatevectorBackend::new()
        .expectation(circuit, &ParameterBinding::new(), obs)
        .unwrap()
}

/// |+i⟩ = S·H|0⟩ = (|0⟩ + i|1⟩)/√2 is the +1 eigenstate of Y.
#[test]
fn test_y_eigenstate_gives_plus_one() {
    let mut c = CircuitIR::new(1, CircuitType::GateBased);
    c.add_op(gate(GateKind::H, &[0], &[]));
    c.add_op(gate(GateKind::S, &[0], &[]));
    let y = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Y)])],
    };
    let v = expect(&c, &y);
    assert!((v - 1.0).abs() < 1e-12, "⟨+i|Y|+i⟩ = {v}, want +1");
}

/// Ry(θ)|0⟩ then Sdg·? — simpler: Rx(θ)|0⟩ has ⟨Y⟩ = −sin θ
/// (Rx rotates Z toward −Y in the Bloch picture).
#[test]
fn test_rx_rotation_y_profile() {
    let theta = 0.7113;
    let mut c = CircuitIR::new(1, CircuitType::GateBased);
    c.add_op(gate(GateKind::Rx, &[0], &[ParamExpr::Concrete(theta)]));
    let y = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Y)])],
    };
    let v = expect(&c, &y);
    assert!(
        (v + theta.sin()).abs() < 1e-12,
        "⟨Y⟩ after Rx({theta}) = {v}, want {}",
        -theta.sin()
    );
}

/// Mixed odd-Y two-qubit string: |ψ⟩ = (H⊗I)·CX-prep of a Bell-ish
/// state rotated so ⟨X₀Y₁⟩ is a clean closed form. Use
/// |ψ⟩ = CX(0,1)·(Ry(θ)⊗I)|00⟩ = cos(θ/2)|00⟩ + sin(θ/2)|11⟩:
///   ⟨X₀Y₁⟩ = 0 by symmetry, but ⟨Y₀Y₁⟩ = −sin θ and
///   ⟨Y₀X₁⟩ = 0; instead S on qubit 1 turns X₁ into Y₁:
/// simplest sharp check: state (|00⟩ + i|11⟩)/√2 via Ry(π/2), CX, S₁:
///   ⟨X₀Y₁⟩ = +1 exactly.
#[test]
fn test_two_qubit_single_y_string() {
    use std::f64::consts::FRAC_PI_2;
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(gate(GateKind::Ry, &[0], &[ParamExpr::Concrete(FRAC_PI_2)]));
    c.add_op(gate(GateKind::CX, &[0, 1], &[]));
    c.add_op(gate(GateKind::S, &[1], &[]));
    // |ψ⟩ = (|00⟩ + i|11⟩)/√2. X₀Y₁|00⟩ = X₀·(i|01⟩) = i|11⟩;
    // X₀Y₁|11⟩ = X₀·(−i|10⟩) = −i|00⟩.
    // ⟨ψ|X₀Y₁|ψ⟩ = conj(i/√2)·i·(1/√2) + conj(1/√2)·(−i)·(i/√2)
    //            = (−i·i + −i·i)/2 = 1.
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::X), (1, PauliOp::Y)])],
    };
    let v = expect(&c, &obs);
    assert!((v - 1.0).abs() < 1e-12, "⟨X₀Y₁⟩ = {v}, want +1");
}

/// Even-Y strings were always correct — pin one so the fix cannot
/// overshoot: Bell state (|00⟩+|11⟩)/√2 has ⟨Y₀Y₁⟩ = −1.
#[test]
fn test_even_y_string_unchanged() {
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(gate(GateKind::H, &[0], &[]));
    c.add_op(gate(GateKind::CX, &[0, 1], &[]));
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Y), (1, PauliOp::Y)])],
    };
    let v = expect(&c, &obs);
    assert!((v + 1.0).abs() < 1e-12, "⟨Y₀Y₁⟩ on Bell = {v}, want −1");
}
