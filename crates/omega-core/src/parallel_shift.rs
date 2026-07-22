//! Parallelised parameter-shift gradients for commuting trailing blocks
//! (Mathur, Barkoutsos, Yamada, Roetteler, Kerenidis — arXiv:2606.03517).
//!
//! When the last layer of a circuit is a *commuting block* — parametric
//! gates with mutually disjoint qubit supports, e.g. one butterfly layer
//! of RBS gates — every gate's generator `G_k` commutes with the whole
//! layer, and the gradient of `f(θ) = ⟨χ(θ)|H|χ(θ)⟩` factorises into
//! plain expectation values on the *unshifted* final state:
//!
//! ```text
//!     ∂f/∂θ_k = i·⟨χ| [G_k, H] |χ⟩
//! ```
//!
//! Because `G_k` and `H` are small Pauli sums, `i·[G_k, H]` is again a
//! Pauli-sum observable, so **all** block gradients come from a single
//! state preparation measured against `K` commuting-block observables —
//! the simulator analogue of the paper's "one set of circuit executions
//! per layer" (`Backend::expectation_multi` = one forward sweep). Serial
//! parameter-shift needs `4·K` executions for the same layer (4-term
//! Givens rule per RBS gate); this path needs `1`.
//!
//! Symbols with occurrences *outside* the trailing block fall back to
//! the ordinary per-slot shift rules — correctness first, speed second.
//! The [`ParallelShiftReport`] says exactly what happened, so callers
//! (e.g. the butterfly-QNN layer-wise trainer) can report honest
//! circuit-execution counts.

use std::collections::{HashMap, HashSet};

use crate::circuit::{CircuitIR, GateKind, ParamExpr, SymbolId};
use crate::error::{OmegaError, Result};
use crate::executor::{Backend, Observable, PauliOp};
use crate::params::ParameterBinding;

/// A Pauli string as `(qubit, op)` pairs — the same shape `Observable`
/// terms use. Kept sorted by qubit and free of explicit identities.
type PauliString = Vec<(u32, PauliOp)>;

/// Single-qubit Pauli product `a·b = phase · c` with `phase = i^k`.
/// Returns `(k mod 4, c)`, where `c = None` means identity.
fn pauli_mul_1q(a: PauliOp, b: PauliOp) -> (u8, Option<PauliOp>) {
    use PauliOp::*;
    match (a, b) {
        (I, I) => (0, None),
        (I, p) | (p, I) => (0, Some(p)),
        (X, X) | (Y, Y) | (Z, Z) => (0, None),
        // XY = iZ, YZ = iX, ZX = iY (cyclic); reversed order picks up −i.
        (X, Y) => (1, Some(Z)),
        (Y, Z) => (1, Some(X)),
        (Z, X) => (1, Some(Y)),
        (Y, X) => (3, Some(Z)),
        (Z, Y) => (3, Some(X)),
        (X, Z) => (3, Some(Y)),
    }
}

/// Product of two Pauli strings: `A·B = i^k · C`. Inputs need not be
/// sorted; the output is sorted by qubit with identities dropped.
fn pauli_mul(a: &PauliString, b: &PauliString) -> (u8, PauliString) {
    let mut by_qubit: HashMap<u32, PauliOp> = a.iter().cloned().collect();
    let mut k: u8 = 0;
    for &(q, op_b) in b {
        let op_a = by_qubit.remove(&q).unwrap_or(PauliOp::I);
        let (dk, prod) = pauli_mul_1q(op_a, op_b);
        k = (k + dk) % 4;
        if let Some(p) = prod {
            by_qubit.insert(q, p);
        }
    }
    let mut out: PauliString = by_qubit
        .into_iter()
        .filter(|(_, p)| !matches!(p, PauliOp::I))
        .collect();
    out.sort_by_key(|&(q, _)| q);
    (k, out)
}

/// Do two Pauli strings commute? They anticommute iff the number of
/// qubits where both act with *different* non-identity Paulis is odd.
fn pauli_commutes(a: &PauliString, b: &PauliString) -> bool {
    let a_map: HashMap<u32, PauliOp> = a.iter().cloned().collect();
    let mut anti = 0usize;
    for &(q, op_b) in b {
        if matches!(op_b, PauliOp::I) {
            continue;
        }
        if let Some(&op_a) = a_map.get(&q) {
            if !matches!(op_a, PauliOp::I) && op_a != op_b {
                anti += 1;
            }
        }
    }
    anti.is_multiple_of(2)
}

/// The Pauli-sum generator `G` of a gate `U(θ) = exp(−i·θ·G)`.
/// Note the *unhalved* convention: `Rx(θ) = exp(−i·θ·(X/2))` → `G = X/2`;
/// `RBS(θ) = exp(−i·θ·(Y⊗X − X⊗Y)/2)` → `G = (Y⊗X − X⊗Y)/2`.
fn gate_generator(gate: &GateKind, qubits: &[u32]) -> Option<Vec<(f64, PauliString)>> {
    match gate {
        GateKind::Rx => Some(vec![(0.5, vec![(qubits[0], PauliOp::X)])]),
        GateKind::Ry => Some(vec![(0.5, vec![(qubits[0], PauliOp::Y)])]),
        GateKind::Rz => Some(vec![(0.5, vec![(qubits[0], PauliOp::Z)])]),
        GateKind::Rbs => {
            let (a, b) = (qubits[0], qubits[1]);
            Some(vec![
                (0.5, vec![(a, PauliOp::Y), (b, PauliOp::X)]),
                (-0.5, vec![(a, PauliOp::X), (b, PauliOp::Y)]),
            ])
        }
        _ => None,
    }
}

/// Build the gradient observable `i·[G, H]` as a Pauli-sum `Observable`.
/// Every surviving term has a real coefficient (both `G` and `H` are
/// Hermitian, so `i·[G, H]` is Hermitian and Pauli products of
/// anticommuting strings carry a `±i` that the leading `i` cancels).
fn gradient_observable(generator: &[(f64, PauliString)], h: &Observable) -> Observable {
    let mut acc: HashMap<PauliString, f64> = HashMap::new();
    for (w, g_str) in generator {
        for (c, h_str) in &h.terms {
            let h_sorted: PauliString = {
                let mut s = h_str.clone();
                s.sort_by_key(|&(q, _)| q);
                s
            };
            if pauli_commutes(g_str, &h_sorted) {
                continue; // [G_t, H_t] = 0
            }
            // Anticommuting: [A, B] = 2AB, so i·w·c·[A, B] = 2·w·c·i·(i^k)·C.
            let (k, prod) = pauli_mul(g_str, &h_sorted);
            // i · i^k = i^(k+1); for anticommuting Hermitian Paulis k is
            // odd (1 or 3), so i^(k+1) ∈ {−1, +1} — real, as promised.
            let phase: f64 = match (k + 1) % 4 {
                0 => 1.0,
                2 => -1.0,
                _ => unreachable!("i·[G,H] must be Hermitian (k = {k})"),
            };
            *acc.entry(prod).or_insert(0.0) += 2.0 * w * c * phase;
        }
    }
    let terms: Vec<(f64, PauliString)> = acc
        .into_iter()
        .filter(|(_, c)| c.abs() > 1e-15)
        .map(|(s, c)| (c, s))
        .collect();
    Observable {
        terms: if terms.is_empty() {
            vec![(0.0, vec![])]
        } else {
            terms
        },
    }
}

/// One parametric slot inside the trailing commuting block.
struct BlockSlot {
    op_idx: usize,
    param_idx: usize,
    generator: Vec<(f64, PauliString)>,
}

/// Find the maximal trailing commuting block: scanning ops from the
/// end, keep gates that (a) have a known Pauli-sum generator or are
/// `Barrier`, and (b) act on qubits disjoint from every gate already
/// in the block. The first op that violates either rule ends the scan.
fn trailing_commuting_block(circuit: &CircuitIR) -> Vec<usize> {
    let mut used_qubits: HashSet<u32> = HashSet::new();
    let mut block = Vec::new();
    for (idx, op) in circuit.ops.iter().enumerate().rev() {
        if matches!(op.gate, GateKind::Barrier) {
            continue;
        }
        let qubits: Vec<u32> = op.qubits.iter().map(|q| q.0).collect();
        let has_generator = gate_generator(&op.gate, &qubits).is_some();
        let disjoint = qubits.iter().all(|q| !used_qubits.contains(q));
        if has_generator && disjoint {
            used_qubits.extend(qubits);
            block.push(idx);
        } else {
            break;
        }
    }
    block.reverse();
    block
}

/// What the parallel path did — for honest execution-count reporting.
#[derive(Clone, Debug, Default)]
pub struct ParallelShiftReport {
    /// Symbols whose every occurrence sat inside the trailing block —
    /// their gradients came from the single batched evaluation.
    pub block_symbols: usize,
    /// Symbols that fell back to serial per-slot shift rules.
    pub fallback_symbols: usize,
    /// Simulator forward passes ("circuit executions" in the hardware
    /// cost model): 1 for the batched block + the serial-shift
    /// evaluations of the fallback symbols.
    pub circuit_executions: usize,
    /// Number of gates in the detected trailing commuting block.
    pub block_gates: usize,
}

/// Gradient of `⟨H⟩` w.r.t. the given symbols (all free symbols when
/// `only` is `None`), using the parallel commuting-block rule for every
/// symbol confined to the trailing block and serial parameter-shift for
/// the rest. Returns `(symbol, gradient)` pairs sorted by symbol id
/// plus a [`ParallelShiftReport`].
pub fn parallel_parameter_shift_gradient(
    backend: &dyn Backend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    only: Option<&HashSet<SymbolId>>,
) -> Result<(Vec<(SymbolId, f64)>, ParallelShiftReport)> {
    if circuit
        .ops
        .iter()
        .any(|op| matches!(op.gate, GateKind::Measure))
    {
        return Err(OmegaError::Unsupported(
            "parallel parameter-shift requires a measurement-free circuit".into(),
        ));
    }

    let block_indices = trailing_commuting_block(circuit);
    let block_set: HashSet<usize> = block_indices.iter().copied().collect();

    // Slots per block op (a block gate may have several params in
    // principle; Rx/Ry/Rz/Rbs all have exactly one).
    let mut block_slots: Vec<BlockSlot> = Vec::new();
    for &op_idx in &block_indices {
        let op = &circuit.ops[op_idx];
        let qubits: Vec<u32> = op.qubits.iter().map(|q| q.0).collect();
        let generator =
            gate_generator(&op.gate, &qubits).expect("block membership implies a known generator");
        for param_idx in 0..op.params.len() {
            block_slots.push(BlockSlot {
                op_idx,
                param_idx,
                generator: generator.clone(),
            });
        }
    }

    // Classify each requested symbol: block-only or fallback.
    let mut symbol_ids: Vec<SymbolId> = circuit.symbols.keys().copied().collect();
    symbol_ids.sort_unstable();
    let wanted = |sym: SymbolId| only.map(|s| s.contains(&sym)).unwrap_or(true);

    // (symbol → its slots in the whole circuit)
    let slots_of = |sym: SymbolId| -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for (op_idx, op) in circuit.ops.iter().enumerate() {
            for (param_idx, p) in op.params.iter().enumerate() {
                if param_expr_uses(p, sym) {
                    v.push((op_idx, param_idx));
                }
            }
        }
        v
    };

    let mut block_syms: Vec<SymbolId> = Vec::new();
    let mut fallback_syms: Vec<SymbolId> = Vec::new();
    for &sym in &symbol_ids {
        if !wanted(sym) {
            continue;
        }
        let slots = slots_of(sym);
        if slots.is_empty() {
            continue; // inactive → gradient 0, emitted below
        }
        if slots.iter().all(|(op_idx, _)| block_set.contains(op_idx)) {
            block_syms.push(sym);
        } else {
            fallback_syms.push(sym);
        }
    }

    // --- Batched path: one expectation_multi over i·[G_k, H]. ---
    let mut grads: HashMap<SymbolId, f64> = HashMap::new();
    let mut executions = 0usize;
    if !block_syms.is_empty() {
        let grad_observables: Vec<Observable> = block_slots
            .iter()
            .map(|slot| gradient_observable(&slot.generator, observable))
            .collect();
        let slot_grads = backend.expectation_multi(circuit, params, &grad_observables)?;
        executions += 1; // one state preparation, K commuting observables

        for (slot, &slot_grad) in block_slots.iter().zip(slot_grads.iter()) {
            let pe = &circuit.ops[slot.op_idx].params[slot.param_idx];
            for &sym in &block_syms {
                let alpha = params.resolve_derivative(pe, sym)?;
                if alpha != 0.0 {
                    *grads.entry(sym).or_insert(0.0) += alpha * slot_grad;
                }
            }
        }
    }

    // --- Serial fallback for symbols escaping the block. ---
    if !fallback_syms.is_empty() {
        let subset: HashSet<SymbolId> = fallback_syms.iter().copied().collect();
        let serial = crate::gradient::compute_gradient_for(
            backend,
            circuit,
            params,
            observable,
            &crate::gradient::GradMethod::ParameterShift,
            Some(&subset),
        )?;
        for (sym, g) in serial {
            if subset.contains(&sym) {
                grads.insert(sym, g);
                // 4-term Givens (Rbs) and Banchi-Crooks (CRz/CU3) slots
                // cost 4 evaluations, 2-term slots cost 2; count the
                // conservative upper bound of 4 per occurrence.
                executions += 4 * slots_of(sym).len();
            }
        }
    }

    let report = ParallelShiftReport {
        block_symbols: block_syms.len(),
        fallback_symbols: fallback_syms.len(),
        circuit_executions: executions,
        block_gates: block_indices.len(),
    };

    let mut out: Vec<(SymbolId, f64)> = symbol_ids
        .iter()
        .filter(|&&s| wanted(s))
        .map(|&s| (s, grads.get(&s).copied().unwrap_or(0.0)))
        .collect();
    out.sort_by_key(|(id, _)| *id);
    Ok((out, report))
}

fn param_expr_uses(expr: &ParamExpr, sym: SymbolId) -> bool {
    match expr {
        ParamExpr::Concrete(_) => false,
        ParamExpr::Symbol(id) => *id == sym,
        ParamExpr::Negate(inner) => param_expr_uses(inner, sym),
        ParamExpr::Add(a, b) | ParamExpr::Mul(a, b) => {
            param_expr_uses(a, sym) || param_expr_uses(b, sym)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn pauli_mul_xy_is_i_z() {
        let (k, s) = pauli_mul(&vec![(0, PauliOp::X)], &vec![(0, PauliOp::Y)]);
        assert_eq!(k, 1);
        assert_eq!(s, vec![(0, PauliOp::Z)]);
    }

    #[test]
    fn pauli_mul_disjoint_concatenates() {
        let (k, s) = pauli_mul(&vec![(0, PauliOp::X)], &vec![(1, PauliOp::Z)]);
        assert_eq!(k, 0);
        assert_eq!(s, vec![(0, PauliOp::X), (1, PauliOp::Z)]);
    }

    #[test]
    fn pauli_commute_rules() {
        // X0 vs Z0 anticommute; X0Z1 vs Z0X1 commute (two anti sites).
        assert!(!pauli_commutes(
            &vec![(0, PauliOp::X)],
            &vec![(0, PauliOp::Z)]
        ));
        assert!(pauli_commutes(
            &vec![(0, PauliOp::X), (1, PauliOp::Z)],
            &vec![(0, PauliOp::Z), (1, PauliOp::X)]
        ));
    }

    #[test]
    fn rbs_gradient_observable_against_z_is_real_and_nonempty() {
        // G = (Y0X1 − X0Y1)/2, H = Z0. i[G,H] should be the Pauli sum
        // (X0X1 + Y0Y1) — the well-known RBS gradient observable.
        let gen = gate_generator(&GateKind::Rbs, &[0, 1]).unwrap();
        let h = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };
        let obs = gradient_observable(&gen, &h);
        assert_eq!(obs.terms.len(), 2, "terms: {:?}", obs.terms);
        for (c, s) in &obs.terms {
            assert_eq!(s.len(), 2, "two-qubit Pauli string expected");
            assert!((c.abs() - 1.0).abs() < 1e-12, "coeff {c}");
        }
    }

    #[test]
    fn trailing_block_stops_at_entangler() {
        let mut c = CircuitIR::new(4, crate::circuit::CircuitType::GateBased);
        let op = |gate, qs: &[u32]| crate::circuit::GateOp {
            gate,
            qubits: qs.iter().map(|&q| crate::circuit::Qubit(q)).collect(),
            params: smallvec![ParamExpr::Concrete(0.1)],
            classical_bit: None,
            condition: None,
        };
        let mut cx = op(GateKind::CX, &[0, 1]);
        cx.params.clear();
        c.add_op(op(GateKind::Rbs, &[0, 1])); // before the CX → excluded
        c.add_op(cx);
        c.add_op(op(GateKind::Rbs, &[0, 1]));
        c.add_op(op(GateKind::Rbs, &[2, 3]));
        let block = trailing_commuting_block(&c);
        assert_eq!(block, vec![2, 3]);
    }

    #[test]
    fn trailing_block_rejects_overlapping_supports() {
        let mut c = CircuitIR::new(3, crate::circuit::CircuitType::GateBased);
        let op = |gate, qs: &[u32]| crate::circuit::GateOp {
            gate,
            qubits: qs.iter().map(|&q| crate::circuit::Qubit(q)).collect(),
            params: smallvec![ParamExpr::Concrete(0.1)],
            classical_bit: None,
            condition: None,
        };
        c.add_op(op(GateKind::Rbs, &[0, 1]));
        c.add_op(op(GateKind::Rbs, &[1, 2])); // overlaps qubit 1
        let block = trailing_commuting_block(&c);
        assert_eq!(block, vec![1], "only the last gate fits the block");
    }
}
