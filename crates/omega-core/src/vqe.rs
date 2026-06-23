//! VQE (Variational Quantum Eigensolver) — hardware-efficient ansatz
//! for variational ground-state search.
//!
//! Sister module to [`crate::qaoa`]. Where QAOA bakes the cost
//! Hamiltonian into the circuit structure, VQE uses a problem-agnostic
//! ansatz and lets the optimiser shape the wavefunction. The harness
//! at `crates/omega-backend-statevector/tests/qubo_compare.rs` runs
//! both side-by-side on small QUBO instances against brute-force.
//!
//! Layer structure (executed `layers` times):
//!   1. `Ry(θ_q)` on every qubit (one fresh symbol per qubit).
//!   2. Linear `CX` entangling chain `CX(0,1), CX(1,2), …, CX(n-2,n-1)`.
//!
//! After all layers a final `Ry(θ_q)` per qubit closes the ansatz so
//! the post-entangling rotation is also trainable. Total free
//! parameters: `n * (layers + 1)`.

use smallvec::smallvec;
use std::collections::HashMap;

use crate::circuit::*;

/// Build a hardware-efficient VQE ansatz on `n` qubits with `layers`
/// rounds of (Ry layer + CX chain) followed by a final Ry layer.
///
/// Symbol naming: `ry_l{l}_{q}` for the qubit-`q` parameter in layer
/// `l` (0-indexed). Symbol IDs are assigned in `(layer, qubit)` row-
/// major order, matching the natural traversal for an optimiser that
/// flattens parameters into a `Vec<f64>`.
pub fn vqe_circuit(n: usize, layers: usize) -> CircuitIR {
    assert!(n >= 1, "VQE ansatz requires n >= 1");

    let mut ops = Vec::new();
    let mut symbols = HashMap::new();
    let mut next_sym: u32 = 0;

    for l in 0..=layers {
        for q in 0..n {
            let sid = next_sym;
            next_sym += 1;
            symbols.insert(sid, format!("ry_l{l}_{q}"));
            ops.push(GateOp {
                gate: GateKind::Ry,
                qubits: smallvec![Qubit(q as u32)],
                params: smallvec![ParamExpr::Symbol(sid)],
                classical_bit: None,
                condition: None,
            });
        }
        if l < layers {
            for q in 0..(n.saturating_sub(1)) {
                ops.push(GateOp {
                    gate: GateKind::CX,
                    qubits: smallvec![Qubit(q as u32), Qubit(q as u32 + 1)],
                    params: smallvec![],
                    classical_bit: None,
                    condition: None,
                });
            }
        }
    }

    CircuitIR {
        num_qubits: n as u32,
        num_classical_bits: 0,
        ops,
        circuit_type: CircuitType::GateBased,
        symbols,
        custom_gates: HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vqe_circuit_param_count() {
        let c = vqe_circuit(4, 2);
        assert_eq!(c.num_qubits, 4);
        assert_eq!(c.symbols.len(), 4 * (2 + 1));
        let cx_count = c
            .ops
            .iter()
            .filter(|op| matches!(op.gate, GateKind::CX))
            .count();
        // 2 layers × 3 CX per layer = 6
        assert_eq!(cx_count, 6);
    }

    #[test]
    fn vqe_circuit_zero_layers_is_single_ry_layer() {
        let c = vqe_circuit(3, 0);
        assert_eq!(c.symbols.len(), 3);
        assert!(c.ops.iter().all(|op| matches!(op.gate, GateKind::Ry)));
    }

    #[test]
    fn vqe_circuit_single_qubit_skips_cx() {
        let c = vqe_circuit(1, 3);
        assert_eq!(c.num_qubits, 1);
        assert!(c.ops.iter().all(|op| matches!(op.gate, GateKind::Ry)));
        assert_eq!(c.symbols.len(), 4);
    }
}
