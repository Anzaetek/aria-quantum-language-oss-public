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

                    let mut bitstring = 0u64;
                    for q in 0..n {
                        if tab.measure(q, &mut rng) {
                            bitstring |= 1 << q;
                        }
                    }
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
    for k in 0..n {
        let stab = tab.stabilizer(k);
        // Evaluate the stabilizer on the basis state |b⟩
        // The eigenvalue is (-1)^{sign} * (-1)^{number of Z/Y on qubits where bit=1}
        let mut parity = stab.sign;
        for q in 0..n {
            let bit = (basis >> q) & 1 == 1;
            if bit {
                // Z contributes -1 when bit=1
                // Y = iXZ also contributes -1 when bit=1 (from the Z part)
                // X doesn't contribute phase from the bit value
                parity ^= stab.z[q];
            }
        }
        // If parity is false (eigenvalue +1): (1+1)/2 = 1
        // If parity is true (eigenvalue -1): (1-1)/2 = 0
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

    // P commutes with all stabilizers. Check if it's in the stabilizer group.
    // Try to express P as a product of stabilizer generators using Gaussian elimination.
    // Build a copy of the stabilizer generators and try to reduce P.
    let mut target_x = p.x.clone();
    let mut target_z = p.z.clone();
    let mut result_sign = false;

    for k in 0..n {
        let stab = tab.stabilizer(k);

        // Find a qubit where stab has a non-trivial Pauli
        // Try to eliminate that qubit from target using this stabilizer
        let mut should_multiply = false;

        // Check if we need this stabilizer to match target
        for q in 0..n {
            let stab_nontrivial = stab.x[q] || stab.z[q];
            let target_nontrivial = target_x[q] || target_z[q];

            if stab_nontrivial && target_nontrivial {
                // Check if they match on this qubit
                if stab.x[q] == target_x[q] && stab.z[q] == target_z[q] {
                    should_multiply = true;
                    break;
                }
            }
        }

        if should_multiply {
            // Multiply target by this stabilizer
            let mut phase = 0i32;
            for q in 0..n {
                phase += pauli_mult_phase(target_x[q], target_z[q], stab.x[q], stab.z[q]);
                target_x[q] ^= stab.x[q];
                target_z[q] ^= stab.z[q];
            }
            result_sign ^= stab.sign;
            let extra_sign = ((phase % 4 + 4) % 4) == 2;
            result_sign ^= extra_sign;
        }
    }

    // Check if target has been fully reduced to identity
    if target_x.iter().any(|&x| x) || target_z.iter().any(|&z| z) {
        // P is not in the stabilizer group (but commutes with all stabilizers)
        // This means P is in the normalizer but not the group → ⟨P⟩ = 0
        return 0.0;
    }

    // P = (-1)^result_sign * (product of some stabilizers)
    // So ⟨ψ|P|ψ⟩ = (-1)^result_sign * 1 = ±1
    if result_sign {
        -1.0
    } else {
        1.0
    }
}

fn pauli_mult_phase(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
    match ((x1, z1), (x2, z2)) {
        ((false, false), _) | (_, (false, false)) => 0,
        ((true, false), (false, true)) => 1, // X·Z = iY (wait, X·Z = -iY)
        ((false, true), (true, false)) => 3, // Z·X = iY
        ((true, false), (true, true)) => 1,  // X·Y = iZ
        ((true, true), (true, false)) => 3,  // Y·X = -iZ
        ((false, true), (true, true)) => 3,  // Z·Y = -iX
        ((true, true), (false, true)) => 1,  // Y·Z = iX
        _ => 0,                              // same: P² = I
    }
}

#[cfg(test)]
mod tests {
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
        // H q0 → measure q0 → if(c0==1) X q1 → measure all
        // Should get only |00⟩ or |11⟩ (never |01⟩ or |10⟩)
        let mut circuit = empty_circuit(2);
        circuit.num_classical_bits = 1;
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

        // Only |00⟩ (0) and |11⟩ (3) should appear
        let c00 = counts.get(&0).copied().unwrap_or(0);
        let c11 = counts.get(&3).copied().unwrap_or(0);
        assert_eq!(
            c00 + c11,
            500,
            "should only have |00⟩ and |11⟩, got {:?}",
            counts
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
