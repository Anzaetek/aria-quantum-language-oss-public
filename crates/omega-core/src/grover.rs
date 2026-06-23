//! Grover's search algorithm, amplitude amplification, and noise-tolerant variants.
//!
//! Generates CircuitIR for Grover's algorithm and its generalisations:
//!   1. Standard Grover search with oracle for marked bitstrings
//!   2. Amplitude amplification with custom state preparation
//!   3. Approximate Grover (bounded iterations)
//!   4. Noise-tolerant Grover (adaptive iteration count under noise)
//!
//! Circuit structure per iteration:
//!   Oracle (phase-flip marked states) → Diffusion (2|s⟩⟨s| - I)
//!
//! Multi-controlled gates are decomposed into CX, CZ, CCX (all supported by backends).

use smallvec::smallvec;
use std::collections::HashMap;

use crate::circuit::*;

// ---------------------------------------------------------------------------
// Iteration count helpers
// ---------------------------------------------------------------------------

/// Optimal number of Grover iterations: ⌊π/4 × √(N/M)⌋.
///
/// `search_space` is N (total states, typically 2^n).
/// `num_solutions` is M (number of marked states).
pub fn optimal_iterations(search_space: usize, num_solutions: usize) -> usize {
    if num_solutions == 0 || num_solutions >= search_space {
        return 0;
    }
    let ratio = search_space as f64 / num_solutions as f64;
    (std::f64::consts::FRAC_PI_4 * ratio.sqrt()).floor() as usize
}

/// Noise-tolerant iteration count: caps at `1 / (2 × ε × gates_per_iter)` to avoid
/// accumulated noise washing out the amplitude amplification.
pub fn noise_tolerant_iterations(
    search_space: usize,
    num_solutions: usize,
    error_rate_per_gate: f64,
    gates_per_iteration: usize,
) -> usize {
    let ideal = optimal_iterations(search_space, num_solutions);
    if error_rate_per_gate <= 0.0 || gates_per_iteration == 0 {
        return ideal;
    }
    let noise_cap =
        (1.0 / (2.0 * error_rate_per_gate * gates_per_iteration as f64)).floor() as usize;
    ideal.min(noise_cap).max(1)
}

// ---------------------------------------------------------------------------
// Gate-level building blocks
// ---------------------------------------------------------------------------

/// Emit a multi-controlled-X (MCX) gate decomposed into CX / CCX.
///
/// For n controls:
///   - 1 control: CX
///   - 2 controls: CCX (Toffoli)
///   - n > 2: V-chain using `ancillae` (needs n-2 ancilla qubits)
fn emit_mcx(ops: &mut Vec<GateOp>, controls: &[Qubit], target: Qubit, ancillae: &[Qubit]) {
    match controls.len() {
        0 => {
            // No controls: just X on target
            ops.push(GateOp {
                gate: GateKind::X,
                qubits: smallvec![target],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        1 => {
            ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec![controls[0], target],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        2 => {
            ops.push(GateOp {
                gate: GateKind::CCX,
                qubits: smallvec![controls[0], controls[1], target],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        n => {
            // V-chain decomposition: requires n-2 ancillae
            assert!(
                ancillae.len() >= n - 2,
                "MCX with {} controls needs {} ancillae, got {}",
                n,
                n - 2,
                ancillae.len()
            );

            // Forward sweep: compute partial ANDs into ancillae
            // CCX(c0, c1, a0)
            ops.push(GateOp {
                gate: GateKind::CCX,
                qubits: smallvec![controls[0], controls[1], ancillae[0]],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
            // CCX(c_i, a_{i-2}, a_{i-1}) for i = 2..n-2
            for i in 2..n - 1 {
                ops.push(GateOp {
                    gate: GateKind::CCX,
                    qubits: smallvec![controls[i], ancillae[i - 2], ancillae[i - 1]],
                    params: smallvec![],
                    classical_bit: None,
                    condition: None,
                });
            }
            // Final CCX: last control + last ancilla -> target
            ops.push(GateOp {
                gate: GateKind::CCX,
                qubits: smallvec![controls[n - 1], ancillae[n - 3], target],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });

            // Reverse sweep: uncompute ancillae
            for i in (2..n - 1).rev() {
                ops.push(GateOp {
                    gate: GateKind::CCX,
                    qubits: smallvec![controls[i], ancillae[i - 2], ancillae[i - 1]],
                    params: smallvec![],
                    classical_bit: None,
                    condition: None,
                });
            }
            ops.push(GateOp {
                gate: GateKind::CCX,
                qubits: smallvec![controls[0], controls[1], ancillae[0]],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
    }
}

/// Emit a multi-controlled-Z (MCZ) gate.
///
/// MCZ flips the phase of the all-ones state |11...1⟩.
/// Decomposition: H(last) · MCX(rest → last) · H(last)
fn emit_mcz(ops: &mut Vec<GateOp>, qubits: &[Qubit], ancillae: &[Qubit]) {
    match qubits.len() {
        0 | 1 => {
            // MCZ on 0 or 1 qubit: Z gate (global phase for 0, Z for 1)
            if let Some(&q) = qubits.first() {
                ops.push(GateOp {
                    gate: GateKind::Z,
                    qubits: smallvec![q],
                    params: smallvec![],
                    classical_bit: None,
                    condition: None,
                });
            }
        }
        2 => {
            // CZ
            ops.push(GateOp {
                gate: GateKind::CZ,
                qubits: smallvec![qubits[0], qubits[1]],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        n => {
            // H on last qubit
            let last = qubits[n - 1];
            let controls: Vec<Qubit> = qubits[..n - 1].to_vec();

            ops.push(GateOp {
                gate: GateKind::H,
                qubits: smallvec![last],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
            emit_mcx(ops, &controls, last, ancillae);
            ops.push(GateOp {
                gate: GateKind::H,
                qubits: smallvec![last],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
    }
}

/// Emit oracle gates that phase-flip a specific computational basis state.
///
/// Strategy: X on qubits where the target bit is 0, then MCZ, then undo X.
#[allow(clippy::needless_range_loop)]
fn emit_oracle_mark_state(
    ops: &mut Vec<GateOp>,
    search_qubits: &[Qubit],
    target: u64,
    ancillae: &[Qubit],
) {
    let n = search_qubits.len();

    // X on qubits where target bit is 0
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            ops.push(GateOp {
                gate: GateKind::X,
                qubits: smallvec![search_qubits[i]],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
    }

    // MCZ on all search qubits
    emit_mcz(ops, search_qubits, ancillae);

    // Undo X gates
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            ops.push(GateOp {
                gate: GateKind::X,
                qubits: smallvec![search_qubits[i]],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
    }
}

/// Emit the Grover diffusion operator: 2|s⟩⟨s| - I.
///
/// Circuit: H → X → MCZ → X → H (on all search qubits).
fn emit_diffusion(ops: &mut Vec<GateOp>, search_qubits: &[Qubit], ancillae: &[Qubit]) {
    // H on all
    for &q in search_qubits {
        ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec![q],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    // X on all
    for &q in search_qubits {
        ops.push(GateOp {
            gate: GateKind::X,
            qubits: smallvec![q],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    // MCZ
    emit_mcz(ops, search_qubits, ancillae);
    // X on all
    for &q in search_qubits {
        ops.push(GateOp {
            gate: GateKind::X,
            qubits: smallvec![q],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    // H on all
    for &q in search_qubits {
        ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec![q],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a standard Grover search circuit.
///
/// `num_qubits` — number of search qubits (search space = 2^num_qubits).
/// `marked_states` — bitstrings to amplify.
/// `num_iterations` — if None, uses the optimal count.
///
/// Returns a CircuitIR with search qubits + any ancillae needed for MCX decomposition.
pub fn grover_circuit(
    num_qubits: u32,
    marked_states: &[u64],
    num_iterations: Option<usize>,
) -> CircuitIR {
    let n = num_qubits as usize;
    let search_space = 1usize << n;
    let iters =
        num_iterations.unwrap_or_else(|| optimal_iterations(search_space, marked_states.len()));

    // Ancillae needed for MCX: max(n-3, 0) when n >= 4
    let num_ancillae = if n >= 4 { n - 3 } else { 0 };
    let total_qubits = n + num_ancillae;

    let search_qubits: Vec<Qubit> = (0..n).map(|i| Qubit(i as u32)).collect();
    let ancillae: Vec<Qubit> = (n..total_qubits).map(|i| Qubit(i as u32)).collect();

    let mut ops = Vec::new();

    // Step 1: Hadamard on all search qubits → uniform superposition
    for &q in &search_qubits {
        ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec![q],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }

    // Step 2: Grover iterations
    for _ in 0..iters {
        // Oracle: phase-flip each marked state
        for &target in marked_states {
            emit_oracle_mark_state(&mut ops, &search_qubits, target, &ancillae);
        }
        // Diffusion
        emit_diffusion(&mut ops, &search_qubits, &ancillae);
    }

    CircuitIR {
        num_qubits: total_qubits as u32,
        num_classical_bits: 0,
        ops,
        circuit_type: CircuitType::GateBased,
        symbols: HashMap::new(),
        custom_gates: HashMap::new(),
    }
}

/// Build an amplitude amplification circuit with a custom state preparation unitary.
///
/// Generalises Grover by replacing H-initialisation with `state_prep`.
/// The diffusion operator becomes: state_prep · (2|0⟩⟨0| - I) · state_prep†.
///
/// `state_prep` — circuit preparing the initial state from |0⟩^n.
/// `marked_states` — bitstrings to amplify.
/// `num_iterations` — number of amplification rounds.
pub fn amplitude_amplification(
    state_prep: &CircuitIR,
    marked_states: &[u64],
    num_iterations: usize,
) -> CircuitIR {
    let n = state_prep.num_qubits as usize;
    let num_ancillae = if n >= 4 { n - 3 } else { 0 };
    let total_qubits = n + num_ancillae;

    let search_qubits: Vec<Qubit> = (0..n).map(|i| Qubit(i as u32)).collect();
    let ancillae: Vec<Qubit> = (n..total_qubits).map(|i| Qubit(i as u32)).collect();

    let mut ops = Vec::new();

    // Apply state preparation A|0⟩
    ops.extend(state_prep.ops.iter().cloned());

    for _ in 0..num_iterations {
        // Oracle: phase-flip marked states
        for &target in marked_states {
            emit_oracle_mark_state(&mut ops, &search_qubits, target, &ancillae);
        }

        // Reflection about |ψ⟩ = A|0⟩: A · (2|0⟩⟨0| - I) · A†
        // = A · (I - 2|0⟩⟨0|) negated = ...
        // Implemented as: A† → X all → MCZ → X all → A

        // A† (reverse state_prep)
        for op in state_prep.ops.iter().rev() {
            ops.push(adjoint_gate_op(op));
        }

        // X on all, MCZ, X on all (this is -(2|0⟩⟨0| - I) = I - 2|0⟩⟨0|...
        // but with the global phase it implements the correct reflection)
        for &q in &search_qubits {
            ops.push(GateOp {
                gate: GateKind::X,
                qubits: smallvec![q],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        emit_mcz(&mut ops, &search_qubits, &ancillae);
        for &q in &search_qubits {
            ops.push(GateOp {
                gate: GateKind::X,
                qubits: smallvec![q],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }

        // A (re-apply state_prep)
        ops.extend(state_prep.ops.iter().cloned());
    }

    CircuitIR {
        num_qubits: total_qubits as u32,
        num_classical_bits: 0,
        ops,
        circuit_type: CircuitType::GateBased,
        symbols: HashMap::new(),
        custom_gates: HashMap::new(),
    }
}

/// Build a noise-tolerant Grover circuit with adaptive iteration count.
///
/// `error_rate` — depolarizing error probability per gate.
/// Uses fewer iterations to avoid noise accumulation washing out amplification.
pub fn noise_tolerant_grover(num_qubits: u32, marked_states: &[u64], error_rate: f64) -> CircuitIR {
    let n = num_qubits as usize;
    let search_space = 1usize << n;

    // Estimate gates per iteration: oracle + diffusion
    // Oracle per marked state: ~2n X gates + MCZ gates
    // Diffusion: 2n H + 2n X + MCZ
    // MCZ on n qubits: ~2 + 2*(n-2) CCX for V-chain = ~2n gates
    let gates_per_iter = marked_states.len() * (2 * n + 2 * n) + 4 * n + 2 * n;

    let iters = noise_tolerant_iterations(
        search_space,
        marked_states.len(),
        error_rate,
        gates_per_iter,
    );

    grover_circuit(num_qubits, marked_states, Some(iters))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the adjoint (inverse) of a single gate operation.
/// Only handles gates used in typical state-prep circuits.
fn adjoint_gate_op(op: &GateOp) -> GateOp {
    let gate = match &op.gate {
        // Self-adjoint gates
        GateKind::H
        | GateKind::X
        | GateKind::Y
        | GateKind::Z
        | GateKind::CX
        | GateKind::CZ
        | GateKind::CY
        | GateKind::CCX
        | GateKind::Swap
        | GateKind::CSwap
        | GateKind::Id => op.gate.clone(),
        // S† ↔ S
        GateKind::S => GateKind::Sdg,
        GateKind::Sdg => GateKind::S,
        // T† ↔ T
        GateKind::T => GateKind::Tdg,
        GateKind::Tdg => GateKind::T,
        // Rotation gates: negate angle
        GateKind::Rx | GateKind::Ry | GateKind::Rz => op.gate.clone(),
        _ => op.gate.clone(),
    };

    // For rotation gates, negate all parameters
    let params = match &op.gate {
        GateKind::Rx | GateKind::Ry | GateKind::Rz | GateKind::CRz | GateKind::U1 => op
            .params
            .iter()
            .map(|p| ParamExpr::Negate(Box::new(p.clone())))
            .collect(),
        _ => op.params.clone(),
    };

    GateOp {
        gate,
        qubits: op.qubits.clone(),
        params,
        classical_bit: op.classical_bit,
        condition: op.condition,
    }
}

/// Count the number of gates in a Grover circuit (for estimating noise impact).
pub fn count_gates_per_iteration(num_qubits: u32, num_marked: usize) -> usize {
    let n = num_qubits as usize;
    // Oracle: per marked state, ~2n X + MCZ decomposition
    // MCZ: 2 H + MCX; MCX: 2*(n-2)+1 CCX for V-chain ≈ 2n-3 for n>=4, fewer otherwise
    let mcz_gates = if n <= 2 {
        1
    } else if n == 3 {
        3
    } else {
        2 + 2 * (n - 2) + 1
    };
    let oracle_per_state = 2 * n + mcz_gates;
    let diffusion = 2 * n + 2 * n + mcz_gates;
    num_marked * oracle_per_state + diffusion
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_iterations() {
        assert_eq!(optimal_iterations(4, 1), 1);
        assert_eq!(optimal_iterations(8, 1), 2);
        assert_eq!(optimal_iterations(16, 1), 3);
        assert_eq!(optimal_iterations(4, 0), 0);
        assert_eq!(optimal_iterations(4, 4), 0);
    }

    #[test]
    fn test_noise_tolerant_iterations_formula() {
        assert_eq!(noise_tolerant_iterations(8, 1, 0.0, 100), 2);
        let capped = noise_tolerant_iterations(1024, 1, 0.01, 50);
        let ideal = optimal_iterations(1024, 1);
        assert!(
            capped < ideal,
            "capped={} should be < ideal={}",
            capped,
            ideal
        );
        assert!(capped >= 1);
    }

    #[test]
    fn test_grover_circuit_structure_2qubit() {
        let circuit = grover_circuit(2, &[3], None);
        assert_eq!(circuit.num_qubits, 2);
        // First 2 ops should be H gates
        assert_eq!(circuit.ops[0].gate, GateKind::H);
        assert_eq!(circuit.ops[1].gate, GateKind::H);
        assert!(circuit.ops.len() > 2, "should have oracle + diffusion ops");
    }

    #[test]
    fn test_grover_circuit_structure_4qubit() {
        // 4 qubits → needs 1 ancilla
        let circuit = grover_circuit(4, &[15], None);
        assert_eq!(circuit.num_qubits, 5, "4 search + 1 ancilla");
    }

    #[test]
    fn test_grover_circuit_structure_5qubit() {
        let circuit = grover_circuit(5, &[21], None);
        assert_eq!(circuit.num_qubits, 7, "5 search + 2 ancillae");
    }

    #[test]
    fn test_noise_tolerant_fewer_ops() {
        let standard = grover_circuit(3, &[5], None);
        let noisy = noise_tolerant_grover(3, &[5], 0.01);
        assert!(
            noisy.ops.len() <= standard.ops.len(),
            "noise-tolerant ({} ops) should have <= standard ({} ops)",
            noisy.ops.len(),
            standard.ops.len()
        );
    }

    #[test]
    fn test_amplitude_amplification_structure() {
        let mut state_prep = CircuitIR::new(2, CircuitType::GateBased);
        state_prep.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Concrete(std::f64::consts::FRAC_PI_3)],
            classical_bit: None,
            condition: None,
        });
        state_prep.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec![Qubit(1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });

        let circuit = amplitude_amplification(&state_prep, &[3], 2);
        assert_eq!(circuit.num_qubits, 2);
        // Should start with the state_prep ops (Ry, H)
        assert_eq!(circuit.ops[0].gate, GateKind::Ry);
        assert_eq!(circuit.ops[1].gate, GateKind::H);
        assert!(circuit.ops.len() > 2);
    }
}
