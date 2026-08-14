//! Pauli/stabilizer backend for efficient Clifford circuit simulation.
//!
//! Supports Clifford gates (H, S, S†, X, Y, Z, CX, CY, CZ, SWAP) with
//! O(n²) time per gate and O(n²) memory. Measurement sampling is O(n²) per shot.
//! Non-Clifford gates (T, Rx, Ry, etc.) return an Unsupported error.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use omega_core::circuit::*;
use omega_core::error::{OmegaError, Result};
use omega_core::executor::*;
use omega_core::params::ParameterBinding;

use omega_core::executor::creg_to_u64;

use crate::stabilizer::pauli_mult_phase;
use crate::stabilizer::{PauliRow, StabilizerTableau};

/// Clifford stabilizer simulator.
pub struct PauliBackend;

impl Default for PauliBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PauliBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for PauliBackend {
    fn name(&self) -> &str {
        "pauli"
    }

    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult> {
        if circuit.circuit_type == CircuitType::Photonic {
            return Err(OmegaError::Unsupported(
                "Pauli backend does not support photonic circuits".into(),
            ));
        }

        let n = circuit.num_qubits as usize;

        // A shot outcome is keyed by a u64, so a register wider than 64
        // qubits cannot be represented and every high bit would be silently
        // dropped — a confident wrong answer, not a truncated one. Refused here
        // rather than at the key-construction site so the message names the
        // circuit, and refused in EVERY backend that can exceed 64 qubits
        // because this is a property of the result type, not of any simulator.
        if config.shots.is_some() {
            omega_core::executor::check_counts_width(
                omega_core::executor::counts_outcome_width(circuit, omega_core::executor::needs_collapse(circuit)
                    && config.mid_circuit_mode == omega_core::executor::MidCircuitMode::Collapse),
            )?;
        }

        match config.shots {
            Some(shots) => {
                // Sampling mode: efficient for stabilizer states
                let mut counts: HashMap<u64, u32> = HashMap::new();
                let mut rng: StdRng = match config.seed {
                    Some(seed) => StdRng::seed_from_u64(seed),
                    None => rand::make_rng::<StdRng>(),
                };

                for _ in 0..shots {
                    let mut tab = StabilizerTableau::zero_state(n);
                    let mut classical_bits = vec![0u8; circuit.num_classical_bits as usize];

                    if config.mid_circuit_mode == MidCircuitMode::Collapse {
                        // Mid-circuit measurement: interleave gates and measurements
                        for op in &circuit.ops {
                            if !op.condition_satisfied(&classical_bits) {
                                continue;
                            }

                            match &op.gate {
                                GateKind::Measure => {
                                    let q = op.qubits[0].0 as usize;
                                    let outcome = if tab.measure(q, &mut rng) { 1u8 } else { 0u8 };
                                    if let Some(cbit) = op.classical_bit {
                                        if (cbit as usize) < classical_bits.len() {
                                            classical_bits[cbit as usize] = outcome;
                                        }
                                    }
                                }
                                GateKind::Barrier => {}
                                _ => {
                                    apply_gate(&mut tab, op, params, &mut rng, false)?;
                                }
                            }
                        }
                    } else {
                        // Legacy: apply all gates, then measure at end
                        apply_circuit(&mut tab, circuit, params, &mut rng, false)?;
                    }

                    // In collapse mode the mid-circuit `measure` statements
                    // already ran and recorded outcomes into `classical_bits`.
                    // Key by the creg, matching the statevector and MPS
                    // backends (and Qiskit, which reports over the creg).
                    //
                    // Re-measuring every qubit here instead — which is what
                    // this did — keys by the FULL qubit register. On
                    // `12_feedforward_sometimes_false.qasm` (2 qubits, a 1-bit
                    // creg) it produced keys {0, 3} where the creg values are
                    // {0, 1}: right physics, wrong register, and a key too
                    // wide to be a creg value at all. Anything that then
                    // truncated to creg width would have read as agreement.
                    let bitstring = if config.mid_circuit_mode == MidCircuitMode::Collapse
                        && circuit.num_classical_bits > 0
                    {
                        creg_to_u64(&classical_bits)
                    } else {
                        let mut bits = 0u64;
                        for q in 0..n {
                            if tab.measure(q, &mut rng) {
                                bits |= 1 << q;
                            }
                        }
                        bits
                    };
                    *counts.entry(bitstring).or_insert(0) += 1;
                }

                Ok(ExecResult::Counts(counts))
            }
            None => {
                // Exact probabilities mode
                if n > 24 {
                    return Err(OmegaError::Unsupported(
                        "Pauli backend exact mode limited to <= 24 qubits".into(),
                    ));
                }

                let dim = 1usize << n;
                let mut probs = vec![0.0; dim];
                // For stabilizer states, there are exactly 2^k nonzero amplitudes
                // where k = number of non-identity stabilizers.
                // Compute by checking stabilizer constraints on each basis state.
                let mut tab = StabilizerTableau::zero_state(n);
                apply_circuit(&mut tab, circuit, params, &mut analytic_rng(), true)?;

                for basis in 0..dim {
                    probs[basis] = stabilizer_probability(&tab, basis, n);
                }

                Ok(ExecResult::Probabilities(probs))
            }
        }
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        let n = circuit.num_qubits as usize;
        let mut tab = StabilizerTableau::zero_state(n);
        apply_circuit(&mut tab, circuit, params, &mut analytic_rng(), true)?;

        let mut total = 0.0;
        for (coeff, pauli_terms) in &observable.terms {
            let val = stabilizer_expectation(&tab, pauli_terms);
            total += coeff * val;
        }
        Ok(total)
    }

    fn expectation_multi(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
    ) -> Result<Vec<f64>> {
        // Apply the circuit to a fresh stabilizer tableau once, then
        // evaluate every observable against the same `&tab`. The
        // default trait impl loops `expectation`, which rebuilds the
        // tableau via `apply_circuit` per call — N tableau rebuilds
        // when one suffices. Mirrors the CPU/MPS overrides.
        if observables.is_empty() {
            return Ok(Vec::new());
        }
        let n = circuit.num_qubits as usize;
        let mut tab = StabilizerTableau::zero_state(n);
        apply_circuit(&mut tab, circuit, params, &mut analytic_rng(), true)?;
        let mut out = Vec::with_capacity(observables.len());
        for obs in observables {
            let mut total = 0.0;
            for (coeff, pauli_terms) in &obs.terms {
                total += coeff * stabilizer_expectation(&tab, pauli_terms);
            }
            out.push(total);
        }
        Ok(out)
    }
}

/// Apply all gates in the circuit to the stabilizer tableau.
/// RNG for the analytic paths. Those paths refuse `Reset` first, and `Reset` is
/// the only consumer of randomness in `apply_gate`, so this is never drawn from;
/// it exists to satisfy the shared signature. Seeded for determinism.
fn analytic_rng() -> StdRng {
    StdRng::seed_from_u64(0)
}

fn apply_circuit(
    tab: &mut StabilizerTableau,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    rng: &mut impl Rng,
    analytic: bool,
) -> Result<()> {
    for op in &circuit.ops {
        apply_gate(tab, op, params, rng, analytic)?;
    }
    Ok(())
}

/// Apply a single gate to the tableau.
/// `rng` is consumed only by [`GateKind::Reset`], a stochastic channel rather
/// than a gate — see [`apply_reset_stabilizer`].
fn apply_gate(
    tab: &mut StabilizerTableau,
    op: &GateOp,
    _params: &ParameterBinding,
    rng: &mut impl Rng,
    analytic: bool,
) -> Result<()> {
    let q = |i: usize| op.qubits[i].0 as usize;

    match &op.gate {
        GateKind::H => tab.h(q(0)),
        GateKind::X => tab.x(q(0)),
        GateKind::Y => tab.y(q(0)),
        GateKind::Z => tab.z(q(0)),
        GateKind::S => tab.s(q(0)),
        GateKind::Sdg => tab.sdg(q(0)),
        GateKind::Sx => tab.sx(q(0)),
        GateKind::Sxdg => tab.sxdg(q(0)),
        GateKind::Id | GateKind::Barrier => {}

        // CX is natively supported
        GateKind::CX => tab.cx(q(0), q(1)),

        // CZ = H(tgt) · CX · H(tgt)
        GateKind::CZ => {
            tab.h(q(1));
            tab.cx(q(0), q(1));
            tab.h(q(1));
        }

        // CY = S†(tgt) · CX · S(tgt)
        GateKind::CY => {
            tab.sdg(q(1));
            tab.cx(q(0), q(1));
            tab.s(q(1));
        }

        // SWAP = CX(a,b) · CX(b,a) · CX(a,b)
        GateKind::Swap => {
            tab.cx(q(0), q(1));
            tab.cx(q(1), q(0));
            tab.cx(q(0), q(1));
        }

        GateKind::Reset => {
            apply_reset_stabilizer(tab, q(0), rng, analytic)?;
        }

        GateKind::Measure => {} // handled by caller (sampling loop or mid-circuit)

        // Non-Clifford gates
        GateKind::T
        | GateKind::Tdg
        | GateKind::Rx
        | GateKind::Ry
        | GateKind::Rz
        | GateKind::U1
        | GateKind::U2
        | GateKind::U3
        | GateKind::CRz
        | GateKind::CU3
        | GateKind::CCX
        | GateKind::CSwap => {
            return Err(OmegaError::Unsupported(format!(
                "gate {:?} is non-Clifford and not supported by the Pauli backend",
                op.gate
            )));
        }

        _ => {
            return Err(OmegaError::Unsupported(format!(
                "gate {:?} not supported by Pauli backend",
                op.gate
            )));
        }
    }

    Ok(())
}

/// Reset qubit `q` to |0⟩ in the stabilizer tableau — the reset **channel**
/// ρ → |0⟩⟨0|_q ⊗ Tr_q(ρ).
///
/// Measure `q` (collapsing the tableau onto a *sampled* outcome) and apply X
/// when that outcome is 1. `tab.measure` already handles both cases correctly:
/// a random outcome (some stabilizer anticommutes with Z_q) is drawn 50/50 and
/// the tableau updated, and a determined outcome is read off.
///
/// **Do not force the outcome to 0.** The previous implementation collapsed the
/// anticommuting row and then set the stabilizer to `+Z_q`, i.e. it
/// *post-selected* on measuring 0 rather than sampling. That is a different
/// channel: on Bell + `reset q0` it drove the partner to |0⟩ as well, so q1 read
/// 0 on all 4000 shots where the reset channel gives 50/50. It happens to agree
/// with the truth in the X basis, which is why cross-backend checks against the
/// (differently wrong) statevector fold never caught it. Pinned by
/// `tests/reset_channel.rs`.
///
/// `analytic` marks a `shots = None` run. There is no ensemble to average over
/// there, so a reset whose measurement outcome is *random* — i.e. some
/// stabilizer anticommutes with Z_q, meaning the qubit is not in a Z
/// eigenstate — is refused rather than silently collapsed one way. This is
/// deliberately conservative: an unentangled |+⟩ also has a random outcome yet
/// its reset is deterministic, and that case is refused too. The statevector
/// backend uses the sharper reduced-purity test; matching it here would need an
/// entanglement test on the tableau, which is left as a follow-up.
fn apply_reset_stabilizer(
    tab: &mut StabilizerTableau,
    q: usize,
    rng: &mut impl Rng,
    analytic: bool,
) -> Result<()> {
    if analytic && !reset_outcome_is_determined(tab, q) {
        return Err(OmegaError::Unsupported(format!(
            "pauli: analytic probabilities/expectation of Reset on qubit {q} is ill-defined — \
             the outcome is random, so the reset leaves the register in a mixed state that one \
             tableau cannot represent. Run with shots (each shot is an independent trajectory)."
        )));
    }
    if tab.measure(q, rng) {
        tab.x(q);
    }
    Ok(())
}

/// True when measuring `q` has a determined outcome: no stabilizer anticommutes
/// with Z_q, so the qubit sits in a Z eigenstate and reset is deterministic.
fn reset_outcome_is_determined(tab: &StabilizerTableau, q: usize) -> bool {
    let n = tab.n;
    let mut z_q = PauliRow::identity(n);
    z_q.z[q] = true;
    !(n..2 * n).any(|i| tab.rows[i].anticommutes(&z_q))
}

/// Compute the probability of measuring a particular basis state from a stabilizer state.
///
/// For a stabilizer state with stabilizers S_1, ..., S_n:
/// P(|b⟩) = Π_{k=1}^{n} (1 + (-1)^{s_k(b)} · eigenvalue_k) / 2
///
/// where s_k(b) is 1 if the stabilizer S_k gives eigenvalue -1 on |b⟩.
fn stabilizer_probability(tab: &StabilizerTableau, basis: usize, n: usize) -> f64 {
    // Fast reject, valid ONLY for DIAGONAL (X-free) stabilizers.
    //
    // `(-1)^sign · (-1)^{#Z on set bits}` is the eigenvalue of `S` on `|b⟩`
    // only when `S` has no X/Y component: otherwise `S|b⟩ ∝ |b ⊕ x_bits⟩ ≠ |b⟩`
    // and `S` has no eigenvalue on `|b⟩` at all. This loop used to run for
    // EVERY stabilizer, so a non-diagonal generator with a negative sign
    // annihilated basis states that carry real probability. Measured: 1576 of
    // 3000 random Clifford circuits returned a distribution that did not sum
    // to 1 — e.g. `[0, 0, 0.25, 0, 0.25, 0, 0, 0]` (sum 0.5) where the truth is
    // uniform 0.25 over four states. The comment below already said the formula
    // "only works when all stabilizers are diagonal"; nothing acted on it.
    //
    // The O(2^n) group enumeration further down is correct in general and is
    // self-sufficient — this is purely an early-out.
    for k in 0..n {
        let stab = tab.stabilizer(k);
        if stab.x.iter().any(|&x| x) {
            continue; // non-diagonal: no eigenvalue on |b⟩, cannot reject here
        }
        let mut parity = stab.sign;
        for q in 0..n {
            if (basis >> q) & 1 == 1 {
                parity ^= stab.z[q];
            }
        }
        // Eigenvalue -1 ⇒ |b⟩ is orthogonal to the stabilizer state.
        if parity {
            return 0.0;
        }
    }

    // But we also need to account for X components: they mix basis states.
    // For a general stabilizer state, the above only works when all stabilizers
    // are diagonal (only Z's). For non-diagonal stabilizers, we need a different approach.
    //
    // The correct formula: P(|b⟩) = |⟨b|ψ⟩|² = (1/2^n) * Σ_{S ∈ stabilizer group} ⟨b|S|b⟩
    // where the sum is over all 2^n elements of the stabilizer group.
    //
    // For efficiency, we compute this using the generators:
    // P(|b⟩) = (1/2^n) * Π_{k=1..n} (1 + eigenvalue(S_k, b))
    //
    // But this product formula only works when the stabilizers commute with Z measurements,
    // which isn't always the case.
    //
    // For the general case, use: P(b) = (1/2^n) * Σ_{g ∈ G} χ_b(g)
    // where χ_b(g) = ⟨b|g|b⟩ is the character of g on basis state b.
    //
    // For a Pauli g acting on |b⟩: g|b⟩ = (phase) * |b'⟩
    // ⟨b|g|b⟩ is nonzero only if b'=b, i.e., g has no X/Y on any qubit (or X/Y flips cancel).
    // Actually: g|b⟩ = phase * |b ⊕ x_bits(g)⟩
    // So ⟨b|g|b⟩ ≠ 0 only if x_bits(g) = 0 (g is diagonal, only I and Z).
    //
    // For a stabilizer group generated by S_1,...,S_n, we enumerate subsets.
    // This is O(2^n) which is exponential. For small n it's fine.

    // Enumerate all 2^n group elements (products of generators)
    let dim = 1u64 << n;
    let mut total = 0.0;

    for mask in 0..dim {
        // Compute the product of selected generators
        let mut x_bits = vec![false; n];
        let mut z_bits = vec![false; n];
        let mut phase_count = 0i32;
        let mut sign = false;

        for k in 0..n {
            if (mask >> k) & 1 == 1 {
                let stab = tab.stabilizer(k);
                // Multiply in this stabilizer
                for q in 0..n {
                    phase_count += pauli_mult_phase(x_bits[q], z_bits[q], stab.x[q], stab.z[q]);
                    x_bits[q] ^= stab.x[q];
                    z_bits[q] ^= stab.z[q];
                }
                sign ^= stab.sign;
            }
        }

        // Check if this group element is diagonal (no X flips)
        if x_bits.iter().any(|&x| x) {
            continue; // Non-diagonal, doesn't contribute
        }

        // Compute ⟨b|g|b⟩ for diagonal g = (-1)^sign * i^{Σ x·z} * Z^z_0 * Z^z_1 * ...
        // Since x_bits are all 0, the i^{Σ x·z} factor is 1.
        // g|b⟩ = (-1)^sign * (-1)^{Σ z_q * b_q} |b⟩
        let mut z_parity = sign;
        for q in 0..n {
            if z_bits[q] && (basis >> q) & 1 == 1 {
                z_parity = !z_parity;
            }
        }
        // Account for implicit i factors from Y terms (but x_bits are 0, so no Y terms)
        let implicit_phase = (phase_count % 4 + 4) % 4;
        let phase_sign = match implicit_phase {
            0 => 1.0,
            2 => -1.0,
            _ => 0.0, // imaginary — shouldn't happen for a valid stabilizer state
        };

        let eigenvalue = if z_parity { -phase_sign } else { phase_sign };
        total += eigenvalue;
    }

    total / (dim as f64)
}

/// Compute ⟨ψ|P|ψ⟩ for a Pauli observable on a stabilizer state.
///
/// Key property: for a stabilizer state |ψ⟩ and Pauli P:
/// - If P commutes with all stabilizers but is NOT in the stabilizer group: ⟨P⟩ = 0
/// - If P is in the stabilizer group: ⟨P⟩ = ±1 (determined by the sign)
/// - If P anti-commutes with any stabilizer: ⟨P⟩ = 0
fn stabilizer_expectation(tab: &StabilizerTableau, pauli_terms: &[(u32, PauliOp)]) -> f64 {
    let n = tab.n;

    // Build the Pauli operator from the terms
    let mut p_x = vec![false; n];
    let mut p_z = vec![false; n];
    for (q, op) in pauli_terms {
        let q = *q as usize;
        match op {
            PauliOp::I => {}
            PauliOp::X => p_x[q] = true,
            PauliOp::Y => {
                p_x[q] = true;
                p_z[q] = true;
            }
            PauliOp::Z => p_z[q] = true,
        }
    }

    let p = PauliRow {
        n,
        x: p_x,
        z: p_z,
        sign: false,
    };

    // Check if P anti-commutes with any stabilizer
    for k in 0..n {
        if tab.stabilizer(k).anticommutes(&p) {
            return 0.0;
        }
    }

    // P commutes with every stabilizer, so it lies in the group up to sign.
    // Express it as a product of generators by **Gaussian elimination over the
    // 2n-column symplectic representation** (columns `0..n` are X bits,
    // `n..2n` the Z bits).
    //
    // This was a single greedy pass with NO pivoting that multiplied by a
    // generator whenever *any* qubit carried a matching non-trivial Pauli. That
    // cannot reliably reduce a genuine group element to the identity, and on
    // failure the code fell through to `return 0.0` under the comment "in the
    // normalizer but not the group" — a failed ALGORITHM reported as physics.
    // `<P> = 0` is a legal expectation value, so nothing looked wrong.
    //
    // Measured before the fix: a Steane-encoded 2-qubit logical Grover circuit
    // gave `<Z_bar(patch 0)> = 0.000000` here while the statevector AND
    // Pauli-propagation backends both gave `+1.000000`.
    let bit = |x: &[bool], z: &[bool], c: usize| if c < n { x[c] } else { z[c - n] };

    // Row-echelon basis of the stabilizer group keyed by pivot column: each
    // `pivots[c]`, when present, has its leading 1 exactly at column `c`.
    let mut pivots: Vec<Option<PauliRow>> = vec![None; 2 * n];
    for k in 0..n {
        let mut row = tab.stabilizer(k).clone();
        for c in 0..2 * n {
            if !bit(&row.x, &row.z, c) {
                continue;
            }
            match &pivots[c] {
                Some(b) => {
                    let mut sign = row.sign;
                    mul_pauli_row(&mut row.x, &mut row.z, &mut sign, b);
                    row.sign = sign;
                }
                None => {
                    pivots[c] = Some(row);
                    break;
                }
            }
        }
    }

    let mut target_x = p.x.clone();
    let mut target_z = p.z.clone();
    let mut result_sign = false;
    for c in 0..2 * n {
        if !bit(&target_x, &target_z, c) {
            continue;
        }
        match &pivots[c] {
            Some(b) => mul_pauli_row(&mut target_x, &mut target_z, &mut result_sign, b),
            // Genuinely outside the group — unreachable for a full-rank
            // n-generator stabilizer state (the centralizer IS the group up to
            // phase), but now a real conclusion rather than a fallthrough.
            None => return 0.0,
        }
    }

    debug_assert!(
        !target_x.iter().any(|&x| x) && !target_z.iter().any(|&z| z),
        "elimination left a residue: the reduction is wrong, not the physics"
    );

    // P = (-1)^result_sign * (product of stabilizers), each +1 on |psi>.
    if result_sign {
        -1.0
    } else {
        1.0
    }
}

/// Multiply the Pauli `(x, z, sign)` in place by `row`, tracking the phase.
fn mul_pauli_row(x: &mut [bool], z: &mut [bool], sign: &mut bool, row: &PauliRow) {
    let mut phase = 0i32;
    for q in 0..row.n {
        phase += pauli_mult_phase(x[q], z[q], row.x[q], row.z[q]);
        x[q] ^= row.x[q];
        z[q] ^= row.z[q];
    }
    *sign ^= row.sign;
    if ((phase % 4) + 4) % 4 == 2 {
        *sign = !*sign;
    }
}


#[cfg(test)]
mod tests {
    /// **Regression: MEASUREMENT sampling and exact probabilities must agree
    /// with the dense reference.**
    ///
    /// The expectation test below did not cover these paths, which is how the
    /// duplicate phase table survived: commit 1c3ef82 fixed the copy in
    /// `sim.rs` (expectations) while `stabilizer.rs` kept an inverted copy
    /// driving `rowmult` / `measure` / `measure_prob`. A 3-qubit Clifford
    /// circuit then put **1000/1000 shots on zero-probability bitstrings**
    /// through the public `execute` API.
    ///
    /// Independently, the exact-probabilities fast-reject was applied to
    /// non-diagonal stabilizers, for which it is invalid: **1576 of 3000**
    /// random circuits returned an unnormalised distribution.
    #[test]
    fn measurement_and_probabilities_agree_with_dense_reference() {
        use omega_core::executor::{ExecConfig, ExecResult, MidCircuitMode};

        let mut seed = 0xBEEFu64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let be = PauliBackend::new();
        let analytic = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let mut checked = 0usize;
        let mut nondiag = 0usize;

        for _ in 0..300 {
            let n = 2 + (rnd() % 3) as u32;
            let depth = 4 + (rnd() % 10) as usize;
            let mut c = CircuitIR::new(n, CircuitType::GateBased);
            for _ in 0..depth {
                let q = (rnd() % n as u64) as u32;
                match rnd() % 6 {
                    0 => c.ops.push(make_op(GateKind::H, &[q])),
                    1 => c.ops.push(make_op(GateKind::S, &[q])),
                    2 => c.ops.push(make_op(GateKind::Sdg, &[q])),
                    3 => c.ops.push(make_op(GateKind::X, &[q])),
                    4 => c.ops.push(make_op(GateKind::Z, &[q])),
                    _ => {
                        let b = (rnd() % n as u64) as u32;
                        if q != b {
                            c.ops.push(make_op(GateKind::CX, &[q, b]));
                        }
                    }
                }
            }
            let truth = dense_probabilities(&c);

            // Exact-probabilities mode.
            if let Ok(ExecResult::Probabilities(p)) =
                be.execute(&c, &ParameterBinding::new(), &analytic)
            {
                checked += 1;
                let sum: f64 = p.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-9,
                    "probabilities sum to {sum}, not 1 (n={n})"
                );
                for (i, (a, b)) in p.iter().zip(truth.iter()).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-9,
                        "P({i}) = {a} vs dense {b} (n={n})"
                    );
                }
            }

            // Shot sampling must never land on a zero-probability outcome.
            let shots = ExecConfig {
                shots: Some(200),
                seed: Some(7),
                mid_circuit_mode: MidCircuitMode::Skip,
            };
            if let Ok(ExecResult::Counts(m)) = be.execute(&c, &ParameterBinding::new(), &shots) {
                for (k, cnt) in m {
                    assert!(
                        truth[k as usize] > 1e-12,
                        "{cnt} shots landed on bitstring {k} with true probability {} (n={n})",
                        truth[k as usize]
                    );
                }
            }
            // Track that the sweep actually exercises NON-DIAGONAL stabilizers,
            // the case the fast-reject got wrong. A sweep of only-diagonal
            // circuits would pass against the broken code.
            if truth.iter().filter(|&&p| p > 1e-12).count() > 1 {
                nondiag += 1;
            }
        }
        assert!(checked > 100, "only {checked} circuits reached probabilities mode");
        assert!(
            nondiag > 50,
            "only {nondiag} circuits had a spread support — the sweep is too degenerate \
             to exercise the non-diagonal path"
        );
    }

    /// Dense `|<b|psi>|^2` for a Clifford circuit — an independent oracle.
    fn dense_probabilities(c: &CircuitIR) -> Vec<f64> {
        use num_complex::Complex64;
        let n = c.num_qubits as usize;
        let dim = 1usize << n;
        let mut psi = vec![Complex64::new(0.0, 0.0); dim];
        psi[0] = Complex64::new(1.0, 0.0);
        let apply1 = |psi: &mut Vec<Complex64>, q: usize, m: [[Complex64; 2]; 2]| {
            let mut out = vec![Complex64::new(0.0, 0.0); psi.len()];
            for (i, &a) in psi.iter().enumerate() {
                if a == Complex64::new(0.0, 0.0) {
                    continue;
                }
                let b = (i >> q) & 1;
                out[i & !(1 << q)] += m[0][b] * a;
                out[i | (1 << q)] += m[1][b] * a;
            }
            *psi = out;
        };
        let z0 = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let r2 = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        for op in &c.ops {
            let q = op.qubits[0].0 as usize;
            match op.gate {
                GateKind::H => apply1(&mut psi, q, [[r2, r2], [r2, -r2]]),
                GateKind::X => apply1(&mut psi, q, [[z0, one], [one, z0]]),
                GateKind::Z => apply1(&mut psi, q, [[one, z0], [z0, -one]]),
                GateKind::S => apply1(&mut psi, q, [[one, z0], [z0, Complex64::new(0.0, 1.0)]]),
                GateKind::Sdg => apply1(&mut psi, q, [[one, z0], [z0, Complex64::new(0.0, -1.0)]]),
                GateKind::CX => {
                    let t = op.qubits[1].0 as usize;
                    let mut out = psi.clone();
                    for i in 0..dim {
                        if (i >> q) & 1 == 1 {
                            out[i ^ (1 << t)] = psi[i];
                        }
                    }
                    psi = out;
                }
                _ => panic!("dense oracle: unexpected gate {:?}", op.gate),
            }
        }
        psi.iter().map(|a| a.norm_sqr()).collect()
    }

    /// **Regression: `stabilizer_expectation` must agree with the exact
    /// statevector on random Clifford circuits, including Y observables.**
    ///
    /// Two defects lived here. The group-membership test was a single greedy
    /// pass with no pivoting, described in its own comment as Gaussian
    /// elimination; when it failed to reduce a genuine group element it fell
    /// through to `return 0.0` under "in the normalizer but not the group" — a
    /// failed ALGORITHM reported as physics, invisible because `<P> = 0` is a
    /// legal expectation. And `pauli_mult_phase` had the `X·Z`/`Z·X` rows
    /// inverted against the Aaronson–Gottesman `g` function, its own comment
    /// recording the doubt ("wait, X·Z = -iY") without acting on it.
    ///
    /// Measured before the fix: a Steane-encoded 2-qubit logical Grover circuit
    /// gave `<Z_bar(patch 0)> = 0.000000` here while the statevector and
    /// Pauli-propagation backends both gave `+1.000000`.
    ///
    /// Randomised rather than a fixed case, because the greedy reduction
    /// succeeded on many inputs — a hand-picked circuit could easily have
    /// passed against the broken code.
    #[test]
    fn expectation_agrees_with_statevector_on_random_clifford_circuits() {
        use omega_core::executor::PauliOp;

        // xorshift64: deterministic, no dev-dependency.
        let mut seed = 0xC0FFEEu64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let be = PauliBackend::new();
        let (mut nonzero, mut with_y) = (0usize, 0usize);

        for _ in 0..400 {
            let n = 2 + (rnd() % 4) as u32;
            let depth = 4 + (rnd() % 12) as usize;
            let mut c = CircuitIR::new(n, CircuitType::GateBased);
            for _ in 0..depth {
                let q = (rnd() % n as u64) as u32;
                match rnd() % 6 {
                    0 => c.ops.push(make_op(GateKind::H, &[q])),
                    1 => c.ops.push(make_op(GateKind::S, &[q])),
                    2 => c.ops.push(make_op(GateKind::Sdg, &[q])),
                    3 => c.ops.push(make_op(GateKind::X, &[q])),
                    4 => c.ops.push(make_op(GateKind::Z, &[q])),
                    _ => {
                        let b = (rnd() % n as u64) as u32;
                        if q != b {
                            c.ops.push(make_op(GateKind::CX, &[q, b]));
                        }
                    }
                }
            }
            let terms: Vec<(u32, PauliOp)> = (0..n)
                .map(|q| {
                    (
                        q,
                        match rnd() % 4 {
                            0 => PauliOp::I,
                            1 => PauliOp::X,
                            2 => PauliOp::Y,
                            _ => PauliOp::Z,
                        },
                    )
                })
                .collect();
            let has_y = terms.iter().any(|(_, p)| matches!(p, PauliOp::Y));
            let obs = Observable {
                terms: vec![(1.0, terms)],
            };

            let got = be
                .expectation(&c, &ParameterBinding::new(), &obs)
                .expect("stabilizer expectation");
            // Exact reference: build the statevector by dense simulation of the
            // same Clifford circuit and evaluate <psi|P|psi> directly.
            let want = dense_reference_expectation(&c, &obs);

            assert!(
                (got - want).abs() < 1e-9,
                "stabilizer {got} vs exact {want} on {n}q depth-{depth} circuit"
            );
            if want.abs() > 1e-9 {
                nonzero += 1;
                if has_y {
                    with_y += 1;
                }
            }
        }

        // Non-degeneracy: agreement on a set of all-zeros proves nothing, and
        // the Y observables are the ones the phase-table bug corrupted.
        assert!(nonzero > 20, "only {nonzero} non-trivial expectations sampled");
        assert!(with_y > 0, "no Y observable produced a non-zero expectation");
    }

    /// Dense `<psi|P|psi>` for a Clifford circuit — an independent oracle that
    /// shares no code with the tableau path.
    fn dense_reference_expectation(c: &CircuitIR, obs: &Observable) -> f64 {
        use num_complex::Complex64;
        use omega_core::executor::PauliOp;
        let n = c.num_qubits as usize;
        let dim = 1usize << n;
        let mut psi = vec![Complex64::new(0.0, 0.0); dim];
        psi[0] = Complex64::new(1.0, 0.0);
        let apply1 = |psi: &mut Vec<Complex64>, q: usize, m: [[Complex64; 2]; 2]| {
            let mut out = vec![Complex64::new(0.0, 0.0); psi.len()];
            for (i, &a) in psi.iter().enumerate() {
                if a == Complex64::new(0.0, 0.0) {
                    continue;
                }
                let b = (i >> q) & 1;
                let i0 = i & !(1 << q);
                let i1 = i | (1 << q);
                out[i0] += m[0][b] * a;
                out[i1] += m[1][b] * a;
            }
            *psi = out;
        };
        let z0 = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let r2 = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        for op in &c.ops {
            let q = op.qubits[0].0 as usize;
            match op.gate {
                GateKind::H => apply1(&mut psi, q, [[r2, r2], [r2, -r2]]),
                GateKind::X => apply1(&mut psi, q, [[z0, one], [one, z0]]),
                GateKind::Z => apply1(&mut psi, q, [[one, z0], [z0, -one]]),
                GateKind::S => apply1(&mut psi, q, [[one, z0], [z0, Complex64::new(0.0, 1.0)]]),
                GateKind::Sdg => apply1(&mut psi, q, [[one, z0], [z0, Complex64::new(0.0, -1.0)]]),
                GateKind::CX => {
                    let t = op.qubits[1].0 as usize;
                    let mut out = psi.clone();
                    for i in 0..dim {
                        if (i >> q) & 1 == 1 {
                            out[i ^ (1 << t)] = psi[i];
                        }
                    }
                    psi = out;
                }
                _ => panic!("reference oracle: unexpected gate {:?}", op.gate),
            }
        }
        let mut total = 0.0;
        for (coeff, terms) in &obs.terms {
            let mut acc = Complex64::new(0.0, 0.0);
            for i in 0..dim {
                let mut j = i;
                let mut w = Complex64::new(1.0, 0.0);
                for (qq, pp) in terms {
                    let qq = *qq as usize;
                    let bit = (i >> qq) & 1;
                    match pp {
                        PauliOp::I => {}
                        PauliOp::X => j ^= 1 << qq,
                        PauliOp::Z => {
                            if bit == 1 {
                                w = -w;
                            }
                        }
                        PauliOp::Y => {
                            j ^= 1 << qq;
                            w *= if bit == 0 {
                                Complex64::new(0.0, 1.0)
                            } else {
                                Complex64::new(0.0, -1.0)
                            };
                        }
                    }
                }
                acc += psi[j].conj() * w * psi[i];
            }
            total += coeff * acc.re;
        }
        total
    }

    use super::*;
    use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
    use omega_core::params::ParameterBinding;
    use smallvec::smallvec;

    fn make_op(gate: GateKind, qubits: &[u32]) -> GateOp {
        GateOp {
            gate,
            qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
            params: smallvec![],
            classical_bit: None,
            condition: None,
        }
    }

    fn empty_circuit(n: u32) -> CircuitIR {
        CircuitIR {
            num_qubits: n,
            num_classical_bits: 0,
            ops: vec![],
            circuit_type: CircuitType::GateBased,
            symbols: Default::default(),
            custom_gates: Default::default(),
        }
    }

    #[test]
    fn test_bell_state_sampling() {
        let mut circuit = empty_circuit(2);
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::CX, &[0, 1]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: Some(1000),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Skip,
        };

        let result = backend.execute(&circuit, &params, &config).unwrap();
        let counts = result.counts();

        // Bell state: only |00⟩ and |11⟩
        for bs in counts.keys() {
            assert!(*bs == 0 || *bs == 3, "unexpected bitstring: {bs}");
        }
        assert!(counts.contains_key(&0));
        assert!(counts.contains_key(&3));
    }

    #[test]
    fn test_ghz_sampling() {
        let mut circuit = empty_circuit(3);
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::CX, &[0, 1]));
        circuit.ops.push(make_op(GateKind::CX, &[1, 2]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: Some(1000),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Skip,
        };

        let result = backend.execute(&circuit, &params, &config).unwrap();
        let counts = result.counts();

        for bs in counts.keys() {
            assert!(*bs == 0 || *bs == 7, "unexpected bitstring: {bs}");
        }
    }

    #[test]
    fn test_x_gate_deterministic() {
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::X, &[0]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: Some(100),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Skip,
        };

        let result = backend.execute(&circuit, &params, &config).unwrap();
        let counts = result.counts();
        // X|0⟩ = |1⟩, always measure 1
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[&1], 100);
    }

    #[test]
    fn test_expectation_z_on_zero() {
        let circuit = empty_circuit(1);
        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };
        let val = backend.expectation(&circuit, &params, &obs).unwrap();
        assert!((val - 1.0).abs() < 1e-8, "expected 1.0, got {val}");
    }

    #[test]
    fn test_expectation_z_on_one() {
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::X, &[0]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };
        let val = backend.expectation(&circuit, &params, &obs).unwrap();
        assert!((val + 1.0).abs() < 1e-8, "expected -1.0, got {val}");
    }

    #[test]
    fn test_expectation_x_on_plus() {
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::H, &[0]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::X)])],
        };
        let val = backend.expectation(&circuit, &params, &obs).unwrap();
        assert!((val - 1.0).abs() < 1e-8, "expected 1.0, got {val}");
    }

    #[test]
    fn test_expectation_xx_bell() {
        let mut circuit = empty_circuit(2);
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::CX, &[0, 1]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        // ⟨XX⟩ on Bell state = 1
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::X), (1, PauliOp::X)])],
        };
        let val = backend.expectation(&circuit, &params, &obs).unwrap();
        assert!((val - 1.0).abs() < 1e-8, "expected 1.0, got {val}");
    }

    #[test]
    fn test_non_clifford_rejected() {
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::T, &[0]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: Some(1),
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        assert!(backend.execute(&circuit, &params, &config).is_err());
    }

    #[test]
    fn test_reset_from_one() {
        // X|0⟩ = |1⟩, then reset → |0⟩, measure should always be 0
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::X, &[0]));
        circuit.ops.push(make_op(GateKind::Reset, &[0]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: Some(100),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend.execute(&circuit, &params, &config).unwrap();
        let counts = result.counts();
        assert_eq!(counts.len(), 1, "should only have |0⟩");
        assert_eq!(counts[&0], 100);
    }

    #[test]
    fn test_reset_from_plus() {
        // H|0⟩ = |+⟩, then reset → |0⟩, measure should always be 0
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::Reset, &[0]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: Some(100),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend.execute(&circuit, &params, &config).unwrap();
        let counts = result.counts();
        assert_eq!(counts.len(), 1, "should only have |0⟩");
        assert_eq!(counts[&0], 100);
    }

    #[test]
    fn test_reset_expectation_z() {
        // X|0⟩ = |1⟩ → ⟨Z⟩ = -1; after reset → |0⟩ → ⟨Z⟩ = +1
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::X, &[0]));
        circuit.ops.push(make_op(GateKind::Reset, &[0]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };
        let val = backend.expectation(&circuit, &params, &obs).unwrap();
        assert!(
            (val - 1.0).abs() < 1e-8,
            "expected ⟨Z⟩=1 after reset, got {val}"
        );
    }

    #[test]
    fn test_exact_probabilities_bell() {
        let mut circuit = empty_circuit(2);
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::CX, &[0, 1]));

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };

        let result = backend.execute(&circuit, &params, &config).unwrap();
        match result {
            ExecResult::Probabilities(probs) => {
                assert!((probs[0] - 0.5).abs() < 1e-8, "|00⟩ prob = {}", probs[0]);
                assert!(probs[1].abs() < 1e-8, "|01⟩ prob = {}", probs[1]);
                assert!(probs[2].abs() < 1e-8, "|10⟩ prob = {}", probs[2]);
                assert!((probs[3] - 0.5).abs() < 1e-8, "|11⟩ prob = {}", probs[3]);
            }
            _ => panic!("expected Probabilities"),
        }
    }

    #[test]
    fn test_pauli_midcircuit_measure_deterministic() {
        // |0⟩ → measure → should always give 0
        let mut circuit = empty_circuit(1);
        circuit.num_classical_bits = 1;
        circuit.ops.push(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: Some(0),
            condition: None,
        });

        let backend = PauliBackend::new();
        let config = ExecConfig {
            shots: Some(100),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Collapse,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &config)
            .unwrap();
        let counts = result.counts();
        // Should only have |0⟩
        assert_eq!(counts.get(&0).copied().unwrap_or(0), 100);
    }

    #[test]
    fn test_pauli_midcircuit_conditional() {
        // `h q0; measure q0 -> c0; if (c==1) x q1; measure q1 -> c1`
        //
        // The feedforward correlates q1 with q0, so only |00⟩ and |11⟩ appear
        // and never |01⟩ or |10⟩ — that is the physics under test, unchanged.
        //
        // What changed is HOW it is observed. This test used to declare a
        // 1-bit creg, leave q1 unmeasured, and rely on the backend
        // re-measuring every qubit at the end and keying counts over the full
        // qubit register (giving 0 and 3). That keying was unique to this
        // backend: on the identical circuit, statevector and MPS both key by
        // the creg and return {0: 261, 1: 239} — as does Qiskit, which always
        // reports over the classical register. So the old assertion pinned
        // this backend's *divergence* from every other one.
        //
        // The fix is to record the correlation where the convention can carry
        // it: a 2-bit creg with q1 measured into c1. Strictly stronger — the
        // same property, checked through the shared counts convention rather
        // than around it.
        let mut circuit = empty_circuit(2);
        circuit.num_classical_bits = 2;
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: Some(0),
            condition: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::X,
            qubits: smallvec![Qubit(1)],
            params: smallvec![],
            classical_bit: None,
            condition: Some((0, 1, 1)),
        });
        circuit.ops.push(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec![Qubit(1)],
            params: smallvec![],
            classical_bit: Some(1),
            condition: None,
        });

        let backend = PauliBackend::new();
        let config = ExecConfig {
            shots: Some(500),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Collapse,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &config)
            .unwrap();
        let counts = result.counts();

        // creg values: c1c0 = 00 (0) and 11 (3). The anti-correlated 01 / 10
        // must be absent.
        let c00 = counts.get(&0).copied().unwrap_or(0);
        let c11 = counts.get(&3).copied().unwrap_or(0);
        assert_eq!(
            c00 + c11,
            500,
            "should only have |00⟩ and |11⟩, got {:?}",
            counts
        );
        // Both must actually occur — otherwise a backend that ignored the
        // guard entirely (q1 always 0, so always 00) would pass the line
        // above. 500 shots at p = 1/2: sd = 22 shots, so 150 is > 6 sigma
        // from either degenerate outcome.
        assert!(
            c00 > 150 && c11 > 150,
            "expected both outcomes near 250/250, got 00={c00} 11={c11} — a \
             single outcome means the guard was not evaluated per shot"
        );
    }

    #[test]
    fn test_expectation_multi_matches_per_observable_loop() {
        // GHZ state on 3 qubits: ⟨Z₀⟩ = 0 (independent qubits with
        // ±1 superposition), ⟨Z₀Z₁⟩ = 1, ⟨Z₀Z₂⟩ = 1, scaled-Z + 0.25
        // identity offset. Pin that the override returns the same
        // numbers as the default per-observable loop.
        let mut circuit = empty_circuit(3);
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::CX, &[0, 1]));
        circuit.ops.push(make_op(GateKind::CX, &[1, 2]));

        let observables = vec![
            Observable {
                terms: vec![(1.0, vec![(0, PauliOp::Z)])],
            },
            Observable {
                terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
            },
            Observable {
                terms: vec![(1.0, vec![(0, PauliOp::Z), (2, PauliOp::Z)])],
            },
            Observable {
                terms: vec![(0.5, vec![(2, PauliOp::Z)]), (0.25, vec![])],
            },
        ];

        let backend = PauliBackend::new();
        let params = ParameterBinding::new();
        let multi = backend
            .expectation_multi(&circuit, &params, &observables)
            .unwrap();
        assert_eq!(multi.len(), observables.len());
        for (obs, m) in observables.iter().zip(multi.iter()) {
            let single = backend.expectation(&circuit, &params, obs).unwrap();
            assert!(
                (m - single).abs() < 1e-12,
                "expectation_multi disagreed with expectation: {m} vs {single}"
            );
        }
    }

    #[test]
    fn test_expectation_multi_empty_returns_empty() {
        // Empty observable list short-circuits before the (cheap)
        // tableau build. Keeps the contract symmetric with the CPU /
        // MPS / Metal overrides.
        let circuit = empty_circuit(2);
        let backend = PauliBackend::new();
        let out = backend
            .expectation_multi(&circuit, &ParameterBinding::new(), &[])
            .unwrap();
        assert!(out.is_empty());
    }
}
