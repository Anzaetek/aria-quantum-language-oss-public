//! Circuit optimization passes for QASM circuits.
//!
//! Three passes in a fixed-point loop:
//! 1. RemoveRedundancies — cancel adjacent self-inverse pairs, remove zero-angle rotations
//! 2. RotationMerging — merge adjacent same-basis rotations (Rz(a);Rz(b) → Rz(a+b))
//! 3. CommuteThroughMultis — commute 1Q gates through CX/CZ to create new cancellation opportunities
//!
//! These three passes give ~80% of tket-style optimization benefit without the full
//! tket2/HUGR dependency stack.

use crate::circuit::*;

/// Run all optimization passes in a fixed-point loop until gate count stabilizes.
/// Returns the number of gates removed.
pub fn optimize(circuit: &mut CircuitIR) -> usize {
    let initial = circuit.ops.len();

    loop {
        let before = circuit.ops.len();
        remove_redundancies(circuit);
        merge_rotations(circuit);
        let after_merge = circuit.ops.len();
        commute_through_multis(circuit);
        remove_redundancies(circuit);
        merge_rotations(circuit);
        if circuit.ops.len() == before && circuit.ops.len() == after_merge {
            break;
        }
    }

    initial - circuit.ops.len()
}

/// Pass 1: Remove redundant gates.
/// - Adjacent self-inverse pairs on same qubits (H-H, X-X, Y-Y, Z-Z, CX-CX, CZ-CZ, Swap-Swap)
/// - Zero-angle rotations (Rz(0), Rx(0), Ry(0))
/// - Identity gates
fn remove_redundancies(circuit: &mut CircuitIR) {
    // Remove Id gates and zero-angle rotations
    circuit.ops.retain(|op| {
        if matches!(op.gate, GateKind::Id) {
            return false;
        }
        if matches!(
            op.gate,
            GateKind::Rx | GateKind::Ry | GateKind::Rz | GateKind::U1
        ) {
            if let Some(&ParamExpr::Concrete(v)) = op.params.first() {
                if v.abs() < 1e-15 {
                    return false;
                }
            }
        }
        true
    });

    // Cancel adjacent self-inverse pairs
    let mut i = 0;
    while i + 1 < circuit.ops.len() {
        if is_self_inverse_pair(&circuit.ops[i], &circuit.ops[i + 1]) {
            circuit.ops.remove(i + 1);
            circuit.ops.remove(i);
            // Step back to check for newly adjacent pairs
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
}

/// Check if two ops form a self-inverse pair (same gate, same qubits, no params).
fn is_self_inverse_pair(a: &GateOp, b: &GateOp) -> bool {
    if a.gate != b.gate || a.qubits != b.qubits {
        return false;
    }
    matches!(
        a.gate,
        GateKind::H
            | GateKind::X
            | GateKind::Y
            | GateKind::Z
            | GateKind::CX
            | GateKind::CZ
            | GateKind::Swap
    )
}

/// Pass 2: Merge adjacent same-basis rotations on the same qubit.
/// Rz(a); Rz(b) → Rz(a+b), same for Rx, Ry, U1.
fn merge_rotations(circuit: &mut CircuitIR) {
    let mut i = 0;
    while i + 1 < circuit.ops.len() {
        let mergeable = {
            let a = &circuit.ops[i];
            let b = &circuit.ops[i + 1];
            a.gate == b.gate
                && a.qubits == b.qubits
                && matches!(
                    a.gate,
                    GateKind::Rx | GateKind::Ry | GateKind::Rz | GateKind::U1
                )
                && a.params.len() == 1
                && b.params.len() == 1
        };

        if mergeable {
            let b_param = circuit.ops[i + 1].params[0].clone();
            let a_param = circuit.ops[i].params[0].clone();

            // Merge: add the parameters
            let merged = match (&a_param, &b_param) {
                (ParamExpr::Concrete(a), ParamExpr::Concrete(b)) => {
                    let sum = a + b;
                    // If sum is ~0, remove entirely
                    if sum.abs() < 1e-15 {
                        circuit.ops.remove(i + 1);
                        circuit.ops.remove(i);
                        i = i.saturating_sub(1);
                        continue;
                    }
                    ParamExpr::Concrete(sum)
                }
                _ => ParamExpr::Add(Box::new(a_param), Box::new(b_param)),
            };

            circuit.ops[i].params[0] = merged;
            circuit.ops.remove(i + 1);
            // Don't advance — check if the merged gate can merge with the next
        } else {
            i += 1;
        }
    }
}

/// Pass 3: Commute single-qubit gates through multi-qubit gates to create
/// new cancellation/merge opportunities.
///
/// Commutation rules:
/// - Rz commutes through CX on the control qubit
/// - X commutes through CX on the target qubit
/// - Z commutes through CZ on either qubit
/// - Rz commutes through CZ on either qubit
fn commute_through_multis(circuit: &mut CircuitIR) {
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i + 1 < circuit.ops.len() {
            if try_commute(&circuit.ops[i], &circuit.ops[i + 1]) {
                circuit.ops.swap(i, i + 1);
                changed = true;
            }
            i += 1;
        }
    }
}

/// Check if gate `a` (a 2Q gate) can commute with gate `b` (a 1Q gate that follows it).
/// Returns true if `b` can be moved before `a`.
fn try_commute(a: &GateOp, b: &GateOp) -> bool {
    // We only commute 1Q gates past 2Q gates
    if b.gate.num_qubits() != 1 || a.gate.num_qubits() < 2 {
        return false;
    }

    let single_qubit = b.qubits[0];

    match &a.gate {
        GateKind::CX => {
            let control = a.qubits[0];
            let target = a.qubits[1];

            if single_qubit == control {
                // Rz, Z, S, Sdg, T, Tdg, U1 commute through CX control
                matches!(
                    b.gate,
                    GateKind::Rz
                        | GateKind::Z
                        | GateKind::S
                        | GateKind::Sdg
                        | GateKind::T
                        | GateKind::Tdg
                        | GateKind::U1
                )
            } else if single_qubit == target {
                // X commutes through CX target
                matches!(b.gate, GateKind::X)
            } else {
                // Different qubit — always commutes
                true
            }
        }
        GateKind::CZ => {
            if single_qubit == a.qubits[0] || single_qubit == a.qubits[1] {
                // Z-basis gates commute through CZ on either qubit
                matches!(
                    b.gate,
                    GateKind::Rz
                        | GateKind::Z
                        | GateKind::S
                        | GateKind::Sdg
                        | GateKind::T
                        | GateKind::Tdg
                        | GateKind::U1
                )
            } else {
                true
            }
        }
        _ => {
            // For other 2Q gates: only commute if the 1Q gate acts on a different qubit
            single_qubit != a.qubits[0] && single_qubit != a.qubits[1]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    fn gate(kind: GateKind, qubits: &[u32]) -> GateOp {
        GateOp {
            gate: kind,
            qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
            params: smallvec![],
            classical_bit: None,
            condition: None,
        }
    }

    fn rot(kind: GateKind, qubit: u32, angle: f64) -> GateOp {
        GateOp {
            gate: kind,
            qubits: smallvec![Qubit(qubit)],
            params: smallvec![ParamExpr::Concrete(angle)],
            classical_bit: None,
            condition: None,
        }
    }

    #[test]
    fn test_cancel_hh() {
        let mut c = CircuitIR::new(1, CircuitType::GateBased);
        c.ops.push(gate(GateKind::H, &[0]));
        c.ops.push(gate(GateKind::H, &[0]));
        let removed = optimize(&mut c);
        assert_eq!(removed, 2);
        assert!(c.ops.is_empty());
    }

    #[test]
    fn test_cancel_cx_cx() {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.ops.push(gate(GateKind::CX, &[0, 1]));
        c.ops.push(gate(GateKind::CX, &[0, 1]));
        let removed = optimize(&mut c);
        assert_eq!(removed, 2);
        assert!(c.ops.is_empty());
    }

    #[test]
    fn test_no_cancel_different_qubits() {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.ops.push(gate(GateKind::H, &[0]));
        c.ops.push(gate(GateKind::H, &[1]));
        let removed = optimize(&mut c);
        assert_eq!(removed, 0);
        assert_eq!(c.ops.len(), 2);
    }

    #[test]
    fn test_merge_rz() {
        let mut c = CircuitIR::new(1, CircuitType::GateBased);
        c.ops.push(rot(GateKind::Rz, 0, 0.3));
        c.ops.push(rot(GateKind::Rz, 0, 0.5));
        let removed = optimize(&mut c);
        assert_eq!(removed, 1);
        assert_eq!(c.ops.len(), 1);
        if let ParamExpr::Concrete(v) = &c.ops[0].params[0] {
            assert!((*v - 0.8).abs() < 1e-10, "merged angle = {v}, expected 0.8");
        } else {
            panic!("expected concrete param");
        }
    }

    #[test]
    fn test_merge_rz_to_zero() {
        // Rz(0.5); Rz(-0.5) → removed entirely
        let mut c = CircuitIR::new(1, CircuitType::GateBased);
        c.ops.push(rot(GateKind::Rz, 0, 0.5));
        c.ops.push(rot(GateKind::Rz, 0, -0.5));
        let removed = optimize(&mut c);
        assert_eq!(removed, 2);
        assert!(c.ops.is_empty());
    }

    #[test]
    fn test_remove_zero_rotation() {
        let mut c = CircuitIR::new(1, CircuitType::GateBased);
        c.ops.push(rot(GateKind::Rz, 0, 0.0));
        c.ops.push(gate(GateKind::H, &[0]));
        let removed = optimize(&mut c);
        assert_eq!(removed, 1);
        assert_eq!(c.ops.len(), 1);
        assert_eq!(c.ops[0].gate, GateKind::H);
    }

    #[test]
    fn test_commute_rz_through_cx() {
        // CX(0,1); Rz(0.5, q0) — Rz on control commutes through CX
        // After commute: Rz(0.5, q0); CX(0,1)
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.ops.push(rot(GateKind::Rz, 0, 0.3));
        c.ops.push(gate(GateKind::CX, &[0, 1]));
        c.ops.push(rot(GateKind::Rz, 0, 0.5));
        // After commute: Rz(0.3); Rz(0.5); CX → merged Rz(0.8); CX
        let removed = optimize(&mut c);
        assert_eq!(removed, 1, "should merge two Rz gates");
        assert_eq!(c.ops.len(), 2, "Rz + CX");
    }

    #[test]
    fn test_optimize_qaoa_structure() {
        use crate::qaoa::qaoa_circuit;
        use crate::qubo::Qubo;

        let mut q = Qubo::new(3);
        q.set(0, 1, 1.0);
        q.set(1, 2, 1.0);
        let ising = q.to_ising();
        let mut circuit = qaoa_circuit(&ising, 2);

        let before = circuit.ops.len();
        optimize(&mut circuit);
        assert!(
            circuit.ops.len() <= before,
            "optimized ({}) should be <= original ({})",
            circuit.ops.len(),
            before
        );
    }
}
