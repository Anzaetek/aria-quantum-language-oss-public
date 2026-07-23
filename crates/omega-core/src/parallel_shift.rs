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

/// Is this gate a phase-tracked Clifford the suffix conjugation
/// supports? (Non-parametric only; T and CCX end the scan.)
fn is_suffix_clifford(gate: &GateKind) -> bool {
    matches!(
        gate,
        GateKind::H
            | GateKind::X
            | GateKind::Y
            | GateKind::Z
            | GateKind::S
            | GateKind::Sdg
            | GateKind::CX
            | GateKind::CZ
            | GateKind::Swap
    )
}

/// Find the maximal trailing structure `block ++ suffix`: an optional
/// run of non-parametric Clifford gates at the very end (the readout /
/// basis-change layer), preceded by the commuting block — parametric
/// gates with known Pauli-sum generators and pairwise-disjoint
/// supports. Returns `(block_indices, suffix_indices)`, both in
/// circuit order. Gradients of block gates stay one-execution: the
/// gradient observable is `i·[B·G_k·B†, H]` with the generator pushed
/// forward through the Clifford suffix `B` to the measurement frame.
fn trailing_commuting_block(circuit: &CircuitIR) -> (Vec<usize>, Vec<usize>) {
    let mut suffix = Vec::new();
    let mut block = Vec::new();
    let mut used_qubits: HashSet<u32> = HashSet::new();
    let mut in_suffix = true;
    for (idx, op) in circuit.ops.iter().enumerate().rev() {
        if matches!(op.gate, GateKind::Barrier) {
            continue;
        }
        if in_suffix {
            if is_suffix_clifford(&op.gate) {
                suffix.push(idx);
                continue;
            }
            in_suffix = false;
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
    suffix.reverse();
    (block, suffix)
}

/// Image of a single X or Z factor on `q` under conjugation by one
/// Clifford gate: `U · P_q · U†` as `(i-power, PauliString)`. Every
/// entry is a textbook stabilizer-tableau rule; Y factors never appear
/// here because callers decompose `Y = i·X·Z` first.
fn factor_image(gate: &GateKind, qs: &[u32], q: u32, is_x: bool) -> (u8, PauliString) {
    use PauliOp::*;
    let touched = qs.contains(&q);
    if !touched {
        return (0, vec![(q, if is_x { X } else { Z })]);
    }
    match gate {
        GateKind::H => (0, vec![(q, if is_x { Z } else { X })]),
        // S: X → Y (exactly), Z → Z. Sdg: X → −Y, Z → Z.
        GateKind::S if is_x => (0, vec![(q, Y)]),
        GateKind::Sdg if is_x => (2, vec![(q, Y)]),
        GateKind::S | GateKind::Sdg => (0, vec![(q, Z)]),
        // Paulis flip anticommuting factors.
        GateKind::X => (
            if is_x { 0 } else { 2 },
            vec![(q, if is_x { X } else { Z })],
        ),
        GateKind::Z => (
            if is_x { 2 } else { 0 },
            vec![(q, if is_x { X } else { Z })],
        ),
        GateKind::Y => (2, vec![(q, if is_x { X } else { Z })]),
        GateKind::CX => {
            let (c, t) = (qs[0], qs[1]);
            match (q == c, is_x) {
                (true, true) => (0, vec![(c, X), (t, X)]), // X_c → X_c X_t
                (true, false) => (0, vec![(c, Z)]),        // Z_c → Z_c
                (false, true) => (0, vec![(t, X)]),        // X_t → X_t
                (false, false) => (0, vec![(c, Z), (t, Z)]), // Z_t → Z_c Z_t
            }
        }
        GateKind::CZ => {
            let (a, b) = (qs[0], qs[1]);
            let other = if q == a { b } else { a };
            if is_x {
                // X_q → X_q Z_other
                let mut s = vec![(q, X), (other, Z)];
                s.sort_by_key(|&(qq, _)| qq);
                (0, s)
            } else {
                (0, vec![(q, Z)])
            }
        }
        GateKind::Swap => {
            let (a, b) = (qs[0], qs[1]);
            let dest = if q == a { b } else { a };
            (0, vec![(dest, if is_x { X } else { Z })])
        }
        other => unreachable!("factor_image on non-suffix gate {other:?}"),
    }
}

/// Conjugate a Pauli string through ONE Clifford gate: decompose each
/// site into X/Z factors (`Y = i·X·Z`), map every factor, and multiply
/// the images back together with the phase-exact [`pauli_mul`].
/// Returns `(i-power, string)`; the i-power is even iff the input
/// phase was even (Hermiticity is preserved), which the caller asserts.
fn conjugate_through_gate(
    phase: u8,
    string: &PauliString,
    gate: &GateKind,
    qs: &[u32],
) -> (u8, PauliString) {
    let mut k = phase;
    let mut acc: PauliString = Vec::new();
    for &(q, p) in string {
        let factors: Vec<(u8, PauliString)> = match p {
            PauliOp::I => continue,
            PauliOp::X => vec![factor_image(gate, qs, q, true)],
            PauliOp::Z => vec![factor_image(gate, qs, q, false)],
            PauliOp::Y => {
                // Y = i·X·Z
                k = (k + 1) % 4;
                vec![
                    factor_image(gate, qs, q, true),
                    factor_image(gate, qs, q, false),
                ]
            }
        };
        for (fk, fs) in factors {
            k = (k + fk) % 4;
            let (mk, ms) = pauli_mul(&acc, &fs);
            k = (k + mk) % 4;
            acc = ms;
        }
    }
    // Fold Y = i·X·Z back out: pauli_mul already emits Y with its phase
    // absorbed, so `acc` is a plain Pauli string and `k` the residue.
    (k, acc)
}

/// Conjugate a weighted Pauli-sum generator into the measurement frame:
/// `G̃ = B·G·B†` for the suffix unitary `B = U_n···U_1` (circuit order) —
/// the Schrödinger pushforward `∂f/∂θ = i·⟨ψ|[B·G·B†, H]|ψ⟩` at the
/// full final state `|ψ⟩ = B·L·A|0⟩`. Conjugate by `U_1` (the first
/// suffix gate) first. Direction pinned by a 1-qubit closed form
/// (Ry(θ) + S suffix, H = Z: BGB† = −X/2 gives f' = −sin θ) AND by the
/// parallel_shift_integration suffix tests against serial/FD — note the
/// direction can only be tested against a Y-sign-correct backend; the
/// odd-Y expectation bug this work uncovered made the opposite
/// direction look right at first (see tests/pauli_y_expectation.rs).
/// Phases must come back real (i^0 → +1, i^2 → −1); an odd residue
/// would break Hermiticity and panics.
fn conjugate_generator(
    generator: &[(f64, PauliString)],
    circuit: &CircuitIR,
    suffix: &[usize],
) -> Vec<(f64, PauliString)> {
    generator
        .iter()
        .map(|(w, s)| {
            let mut k: u8 = 0;
            let mut string = s.clone();
            for &idx in suffix {
                let op = &circuit.ops[idx];
                let qs: Vec<u32> = op.qubits.iter().map(|q| q.0).collect();
                let (nk, ns) = conjugate_through_gate(k, &string, &op.gate, &qs);
                k = nk;
                string = ns;
            }
            let sign = match k {
                0 => 1.0,
                2 => -1.0,
                _ => unreachable!("Clifford conjugation broke Hermiticity (i^{k})"),
            };
            (w * sign, string)
        })
        .collect()
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
    /// Non-parametric Clifford gates after the block (the readout /
    /// basis-change layer the generators were conjugated through).
    pub clifford_suffix_gates: usize,
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

    let (block_indices, suffix_indices) = trailing_commuting_block(circuit);
    let block_set: HashSet<usize> = block_indices.iter().copied().collect();

    // Slots per block op (a block gate may have several params in
    // principle; Rx/Ry/Rz/Rbs all have exactly one). Generators are
    // conjugated through the trailing Clifford suffix so the gradient
    // observables live at the measurement frame.
    let mut block_slots: Vec<BlockSlot> = Vec::new();
    for &op_idx in &block_indices {
        let op = &circuit.ops[op_idx];
        let qubits: Vec<u32> = op.qubits.iter().map(|q| q.0).collect();
        let generator =
            gate_generator(&op.gate, &qubits).expect("block membership implies a known generator");
        let generator = conjugate_generator(&generator, circuit, &suffix_indices);
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
        clifford_suffix_gates: suffix_indices.len(),
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
        let (block, suffix) = trailing_commuting_block(&c);
        assert_eq!(block, vec![2, 3]);
        assert!(suffix.is_empty());
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
        let (block, suffix) = trailing_commuting_block(&c);
        assert_eq!(block, vec![1], "only the last gate fits the block");
        assert!(suffix.is_empty());
    }
}
