// SPDX-License-Identifier: Apache-2.0
//! `StabilizerTableau::sx` / `sxdg` — the Clifford tableau rules.
//!
//! # Why this needs its own test
//!
//! A stabilizer backend **agrees with itself perfectly whatever rule it
//! implements**. A wrong sign condition still produces a well-formed
//! stabilizer state and a plausible distribution, so nothing internal can flag
//! it. That is the same configuration that hid the MPS trajectory defect
//! (`FIXES_PLAN.md` K7): self-consistency is not evidence.
//!
//! So the rules are checked against something outside this crate. Three
//! independent sources agree on the action:
//!
//! | gate | X → | Y → | Z → |
//! |---|---|---|---|
//! | `sx` (Stim `SQRT_X`) | +X | +Z | −Y |
//! | `sxdg` (Stim `SQRT_X_DAG`) | +X | −Z | +Y |
//!
//! 1. Stim's `Tableau::from_named_gate("SQRT_X")` — `X → +X`, `Z → −Y`.
//! 2. A direct conjugation `U·P·U†` computed from the matrices.
//! 3. `proofs/lean4/QuantumProofs/SqrtX.lean`, which *proves* it — and which
//!    was mutation-tested: flipping `sqrtX_conj_Z` to `= Y` makes Lean fail
//!    with `unsolved goals`.
//!
//! The closed form the tableau uses (`x' = x XOR z`, `z' = z`,
//! `sign ^= z AND NOT x` for `sx` / `z AND x` for `sxdg`) was derived from that
//! action and checked against all four Pauli inputs, **not** by analogy with
//! `s()` — note the two gates need *different* sign conditions, so copying one
//! to the other would be wrong in a way no self-consistent check could see.
//!
//! These tests reach the rules through the public backend surface: expectation
//! values of single-qubit Paulis after the gate, which is exactly `⟨0|U†PU|0⟩`.

use omega_backend_pauli::PauliBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind) -> GateOp {
    GateOp {
        gate,
        qubits: smallvec::smallvec![Qubit(0)],
        params: smallvec::smallvec![],
        classical_bit: None,
        condition: None,
    }
}

fn circuit(ops: Vec<GateOp>) -> CircuitIR {
    let mut ir = CircuitIR::new(1, CircuitType::GateBased);
    ir.ops = ops;
    ir
}

fn expect(ops: Vec<GateOp>, p: PauliOp) -> f64 {
    PauliBackend::new()
        .expectation(
            &circuit(ops),
            &ParameterBinding::new(),
            &Observable {
                terms: vec![(1.0, vec![(0, p)])],
            },
        )
        .expect("pauli expectation")
}

/// `sx` is accepted at all — the load-bearing consequence of making it a
/// first-class `GateKind` rather than aliasing to `U3`, which this backend
/// rejects outright as non-Clifford.
#[test]
fn the_stabilizer_backend_accepts_sx_and_sxdg() {
    for g in [GateKind::Sx, GateKind::Sxdg] {
        PauliBackend::new()
            .expectation(
                &circuit(vec![op(g.clone())]),
                &ParameterBinding::new(),
                &Observable {
                    terms: vec![(1.0, vec![(0, PauliOp::Z)])],
                },
            )
            .unwrap_or_else(|e| panic!("{g:?} must be Clifford-representable: {e}"));
    }
}

/// **`sx` applied to |0⟩ gives ⟨Y⟩ = −1.**
///
/// `|0⟩` is the `+Z` eigenstate. `sx` conjugation sends `Z → −Y`, so the state
/// after the gate is the `−Y` eigenstate: `⟨Z⟩ = 0`, `⟨Y⟩ = −1`, `⟨X⟩ = 0`.
///
/// This is the assertion that pins the SIGN, and it is the one that
/// distinguishes `sx` from `sxdg`. A rule copied from `sxdg` gives `+1` here.
#[test]
fn sx_on_zero_lands_on_the_minus_y_eigenstate() {
    let ops = vec![op(GateKind::Sx)];
    assert!(
        (expect(ops.clone(), PauliOp::Y) + 1.0).abs() < 1e-12,
        "sx|0> must be the -Y eigenstate: <Y> = {}, expected -1",
        expect(ops.clone(), PauliOp::Y)
    );
    assert!(expect(ops.clone(), PauliOp::Z).abs() < 1e-12, "<Z> must be 0");
    assert!(expect(ops, PauliOp::X).abs() < 1e-12, "<X> must be 0");
}

/// `sxdg` on |0⟩ gives `⟨Y⟩ = +1` — the OPPOSITE sign, which is why the two
/// tableau rules cannot share a sign condition.
#[test]
fn sxdg_on_zero_lands_on_the_plus_y_eigenstate() {
    let ops = vec![op(GateKind::Sxdg)];
    assert!(
        (expect(ops.clone(), PauliOp::Y) - 1.0).abs() < 1e-12,
        "sxdg|0> must be the +Y eigenstate: <Y> = {}, expected +1",
        expect(ops.clone(), PauliOp::Y)
    );
    assert!(expect(ops, PauliOp::Z).abs() < 1e-12, "<Z> must be 0");
}

/// `sx·sx = X`, so `sx; sx` on |0⟩ gives |1⟩ and `⟨Z⟩ = −1`.
///
/// The identity that makes the gate Clifford, exercised through the tableau
/// rather than the matrix.
#[test]
fn sx_twice_is_x() {
    let v = expect(vec![op(GateKind::Sx), op(GateKind::Sx)], PauliOp::Z);
    assert!((v + 1.0).abs() < 1e-12, "sx;sx|0> = |1>, <Z> = {v}, expected -1");
}

/// `sx; sxdg` is the identity, so `⟨Z⟩` returns to +1.
#[test]
fn sx_then_sxdg_is_identity() {
    let v = expect(vec![op(GateKind::Sx), op(GateKind::Sxdg)], PauliOp::Z);
    assert!((v - 1.0).abs() < 1e-12, "sx;sxdg = I, <Z> = {v}, expected +1");
    let w = expect(vec![op(GateKind::Sxdg), op(GateKind::Sx)], PauliOp::Z);
    assert!((w - 1.0).abs() < 1e-12, "sxdg;sx = I, <Z> = {w}, expected +1");
}

/// `X` is the fixed axis: `sx` on the `+X` eigenstate leaves `⟨X⟩ = +1`.
///
/// Guards against a rule that happens to get the Z and Y images right while
/// disturbing X — the component the `x' = x XOR z` update touches.
#[test]
fn x_is_invariant_under_sx() {
    // H|0> is the +X eigenstate.
    for g in [GateKind::Sx, GateKind::Sxdg] {
        let v = expect(vec![op(GateKind::H), op(g.clone())], PauliOp::X);
        assert!(
            (v - 1.0).abs() < 1e-12,
            "{g:?} must fix the X axis: <X> = {v}, expected +1"
        );
    }
}
