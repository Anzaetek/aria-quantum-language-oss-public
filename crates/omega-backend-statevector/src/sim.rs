use std::collections::HashMap;

use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use omega_core::circuit::*;
use omega_core::defer_measure::{prepare_for_expectation, prepare_for_expectation_multi};
use omega_core::error::{OmegaError, Result};
use omega_core::executor::*;
use omega_core::noise::NoiseModel;
use omega_core::params::ParameterBinding;

use crate::gates;

pub struct StatevectorBackend;

/// Statevector backend with a configurable per-gate noise model.
///
/// The noise model is trajectory-based: after each gate, for every qubit
/// the gate acts on, we sample from the configured channels (depolarizing,
/// amplitude damping, phase damping, arbitrary Pauli) and apply the
/// resulting Kraus branch. Classical measurement outcomes are optionally
/// bit-flipped (`readout_flip`). Running many shots reproduces the
/// density-matrix dynamics without ever materialising ρ.
pub struct NoisyStatevectorBackend {
    model: NoiseModel,
    seed: Option<u64>,
}

impl Default for StatevectorBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StatevectorBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for StatevectorBackend {
    fn name(&self) -> &str {
        "statevector"
    }

    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult> {
        let n = circuit.num_qubits as usize;

        // RNG for mid-circuit measurements and for Reset's Born-rule sampling.
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => rand::make_rng::<StdRng>(),
        };

        // `Reset` is a stochastic CHANNEL, not a gate: one statevector carries
        // one trajectory, so correct shot statistics need independent
        // evolutions. Sampling `shots` times from a single post-reset state
        // would replay one draw of the reset outcome for every shot and report
        // it as certainty. Mirrors `NoisyStatevectorBackend`'s per-trajectory
        // loop. See `apply_reset` and `tests/reset_channel.rs`.
        // Any STOCHASTIC evolution needs one independent trajectory per shot,
        // not one trajectory replayed `shots` times. Reset is stochastic; so is
        // a collapse-mode mid-circuit measurement, which this predicate used to
        // miss — it tested `circuit_has_reset` alone.
        //
        // The consequence was severe and silent: `H q0; measure q0 -> c0;
        // when c0 == 1 { X q1 }` returned |00> on 4000/4000 shots, i.e. a
        // superposition measured with certainty. Qiskit/Aer on the same circuit
        // gives {'00': 1995, '11': 2005}. A coin that always lands the same way.
        //
        // `stochastic_evolution` is the noisy backend's predicate, which had
        // been correct all along (`collapses(..) || circuit_has_reset(..)`);
        // reuse it rather than keep a second, weaker copy in step.
        // A shot outcome is keyed by a u64, so a register wider than 64
        // qubits cannot be represented and every high bit would be silently
        // dropped — a confident wrong answer, not a truncated one. Refused here
        // rather than at the key-construction site so the message names the
        // circuit, and refused in EVERY backend that can exceed 64 qubits
        // because this is a property of the result type, not of any simulator.
        if config.shots.is_some() {
            omega_core::executor::check_counts_width(omega_core::executor::counts_outcome_width(
                circuit,
                collapses(circuit, config),
            ))?;
        }

        if let (true, Some(shots)) = (stochastic_evolution(circuit, config), config.shots) {
            let by_creg = collapses(circuit, config);
            let mut counts: HashMap<u64, u32> = HashMap::new();
            for _ in 0..shots {
                let (state, cbits) = evolve_once(circuit, params, config, &mut rng, false)?;
                let key = if by_creg {
                    creg_to_u64(&cbits)
                } else {
                    sample_counts(&state, n, 1, Some(rng.random()))
                        .into_keys()
                        .next()
                        .unwrap_or(0)
                };
                *counts.entry(key).or_insert(0) += 1;
            }
            // Dense statevector is bounded by 2^n memory long before 64 qubits,
            // so a u64 key is always enough here; only the boundary widens.
            return Ok(ExecResult::counts_from_u64(
                counts,
                omega_core::executor::counts_outcome_width(circuit, by_creg) as u32,
            ));
        }

        let (state, classical_bits) =
            evolve_once(circuit, params, config, &mut rng, config.shots.is_none())?;

        match config.shots {
            None => Ok(ExecResult::Statevector(state)),
            Some(shots) => {
                // Collapse mode trajectory: the gate loop already sampled
                // every mid-circuit `measure` and populated `classical_bits`,
                // so the post-execution state is one classical outcome of one
                // shot. Key counts by the creg value to match Qiskit's
                // `counts` format (length = creg width, last-write-wins on
                // cbit overwrite). Skip mode keeps the analytic
                // sample-from-final-state path because measures were elided
                // and the basis-state IS the creg under the assumed 1:1
                // qubit→cbit mapping.
                if config.mid_circuit_mode == MidCircuitMode::Collapse && !classical_bits.is_empty()
                {
                    let mut bits: u64 = 0;
                    for (i, b) in classical_bits.iter().enumerate() {
                        if i >= 64 {
                            break;
                        }
                        bits |= ((*b as u64) & 1) << i;
                    }
                    let mut counts = HashMap::new();
                    counts.insert(bits, shots);
                    Ok(ExecResult::counts_from_u64(
                        counts,
                        omega_core::executor::counts_outcome_width(circuit, true) as u32,
                    ))
                } else {
                    let counts = sample_counts(&state, n, shots, Some(rng.random()));
                    Ok(ExecResult::counts_from_u64(counts, n as u32))
                }
            }
        }
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        // `expectation_pauli` does `(i >> q) & 1` with `q` straight from the
        // observable and PANICS when it names a qubit past the register.
        // Nothing upstream bounds it: `Observable::parse` reads a bare u32 and
        // never sees the circuit. Refuse here so a library caller gets an error
        // instead of a panic; the server validates at its boundary too.
        observable.validate_qubits(circuit.num_qubits)?;
        // Get statevector
        let (deferred, observable) = prepare_for_expectation(circuit, observable)?;
        let circuit = &deferred;
        let observable = &observable;
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = self.execute(circuit, params, &config)?;
        let sv = match &result {
            ExecResult::Statevector(sv) => sv,
            _ => unreachable!(),
        };

        // Compute <psi|O|psi> = sum_i coeff_i * <psi|P_i|psi>.
        //
        // Linearity of expectation in the operator argument is proved
        // in `verification/Verification/Adjoint/Linearity.lean` — NOTE: that
        // file is a TARGET, not yet written; see `verification/README.md` —
        // theorem `AdjointLinearity.expVal_finset_sum`. That ratifies
        // splitting `O = Σ cₖ Pₖ` into per-Pauli expectations with
        // scalar coefficients, which is exactly this loop.
        let mut total = 0.0;
        for (coeff, pauli_string) in &observable.terms {
            let val = expectation_pauli(sv, circuit.num_qubits, pauli_string);
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
        // Single forward sweep, then per-Pauli loop against the same
        // resident statevector. Saves N-1 simulator runs when the QML
        // trainer asks for ⟨Z_q⟩ on each of N measurement qubits.
        if observables.is_empty() {
            return Ok(Vec::new());
        }
        for o in observables {
            o.validate_qubits(circuit.num_qubits)?;
        }
        let (deferred, dephased) = prepare_for_expectation_multi(circuit, observables)?;
        let circuit = &deferred;
        let observables = &dephased[..];
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = self.execute(circuit, params, &config)?;
        let sv = match &result {
            ExecResult::Statevector(sv) => sv,
            _ => unreachable!(),
        };
        let mut out = Vec::with_capacity(observables.len());
        for obs in observables {
            let mut total = 0.0;
            for (coeff, pauli_string) in &obs.terms {
                let val = expectation_pauli(sv, circuit.num_qubits, pauli_string);
                total += coeff * val;
            }
            out.push(total);
        }
        Ok(out)
    }

    fn adjoint_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<Option<Vec<(SymbolId, f64)>>> {
        // Same unchecked index as `expectation`: the adjoint sweep seeds from
        // the observable and panics on a qubit past the register. `/gradient`
        // takes a client-supplied observable, so this is reachable remotely.
        observable.validate_qubits(circuit.num_qubits)?;
        // A gradient must obey the same contract as the expectation it
        // differentiates. Without this, `expectation_multi_then_gradient` — the
        // fused entry point the QML trainer uses — would pair a mixture-valued
        // prediction with a pure-state gradient from the SAME call: the adjoint
        // sweep skips `Measure` outright (`adjoint.rs`), so a conditioned gate
        // simply never fires. The dephased observable is parameter-independent,
        // so this is free.
        let (deferred, observable) = prepare_for_expectation(circuit, observable)?;
        let circuit = &deferred;
        let observable = &observable;
        // Circuits with Reset are non-unitary — fall back to parameter-shift
        if circuit
            .ops
            .iter()
            .any(|op| matches!(op.gate, GateKind::Reset))
        {
            return Ok(None);
        }
        Ok(Some(crate::adjoint::adjoint_gradient(
            circuit, params, observable,
        )?))
    }

    /// Row-parallel expectation over bindings. Each row is an independent
    /// forward sweep with no shared mutable state, so we `par_iter` and
    /// `collect` — rayon's collect is index-preserving, so the result is
    /// identical (bit-for-bit) to the sequential default regardless of the
    /// thread count.
    fn expectation_batch(
        &self,
        circuit: &CircuitIR,
        bindings: &[&ParameterBinding],
        observable: &Observable,
    ) -> Result<Vec<f64>> {
        use rayon::prelude::*;
        bindings
            .par_iter()
            .map(|b| self.expectation(circuit, b, observable))
            .collect()
    }

    /// Row-parallel adjoint gradients over bindings — the throughput lever for
    /// supervised training (one adjoint pass per data row). Index-preserving.
    fn adjoint_gradient_batch(
        &self,
        circuit: &CircuitIR,
        bindings: &[&ParameterBinding],
        observable: &Observable,
    ) -> Result<Vec<AdjointGradient>> {
        use rayon::prelude::*;
        // Each row builds its OWN checkpoint tape, so peak memory is the
        // per-row tape times however many rows run at once — not one tape.
        // Charging for a single row here would admit a batch that is `threads`
        // times too big, which is the shape that actually exhausts a machine:
        // the width looks modest and the multiplier is invisible.
        let unitary_gates = circuit
            .ops
            .iter()
            .filter(|op| crate::adjoint::is_unitary(&op.gate))
            .count();
        let concurrent = rayon::current_num_threads().min(bindings.len().max(1));
        crate::capacity::check_adjoint(circuit.num_qubits, unitary_gates, concurrent)?;
        bindings
            .par_iter()
            .map(|b| self.adjoint_gradient(circuit, b, observable))
            .collect()
    }
}

impl NoisyStatevectorBackend {
    /// Legacy constructor: single depolarizing rate on every gate.
    pub fn new(error_rate: f64, seed: Option<u64>) -> Self {
        Self {
            model: NoiseModel {
                depolarizing: omega_core::noise::Depolarizing::uniform(error_rate),
                ..Default::default()
            },
            seed,
        }
    }

    /// Composable-model constructor.
    pub fn with_model(model: NoiseModel, seed: Option<u64>) -> Self {
        Self { model, seed }
    }

    pub fn model(&self) -> &NoiseModel {
        &self.model
    }

    /// Evolve one trajectory to its final state, applying each gate followed by
    /// the per-gate noise channel on every qubit it touched. In `Collapse` mode,
    /// `measure` ops project the state and record (readout-flipped) outcomes into
    /// the returned classical register; in `Skip` mode measures are elided and
    /// the terminal distribution is read by sampling the returned state.
    fn evolve(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
        rng: &mut impl Rng,
    ) -> Result<(Vec<Complex64>, Vec<u8>)> {
        // Refuse before allocating: `1usize << n` overflows at n >= 64, and
        // below that `2^n * 16` bytes is set by the input rather than by any
        // knob here. See `capacity`.
        crate::capacity::check(circuit.num_qubits, config.shots.is_some())?;
        let n = circuit.num_qubits as usize;
        let dim = 1usize << n;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);
        let mut classical_bits = vec![0u8; circuit.num_classical_bits as usize];

        for op in &circuit.ops {
            if !op.condition_satisfied(&classical_bits) {
                continue;
            }
            match &op.gate {
                GateKind::Measure => {
                    if config.mid_circuit_mode == MidCircuitMode::Collapse {
                        let q = op.qubits[0].0 as usize;
                        let mut outcome = projective_measure(&mut state, n, q, rng);
                        outcome =
                            crate::noise::maybe_flip_readout(&self.model.readout, q, outcome, rng);
                        if let Some(cbit) = op.classical_bit {
                            if (cbit as usize) < classical_bits.len() {
                                classical_bits[cbit as usize] = outcome;
                            }
                        }
                    }
                }
                GateKind::Barrier => continue,
                _ => {
                    apply_gate(&mut state, n, op, params, rng, false)?;
                    if !self.model.noiseless() {
                        let gate_qubits: Vec<usize> =
                            op.qubits.iter().map(|q| q.0 as usize).collect();
                        for &q in &gate_qubits {
                            crate::noise::apply_channel(
                                &self.model,
                                &mut state,
                                n,
                                q,
                                &gate_qubits,
                                rng,
                            );
                        }
                    }
                }
            }
        }
        Ok((state, classical_bits))
    }
}

impl Backend for NoisyStatevectorBackend {
    fn name(&self) -> &str {
        "noisy-statevector"
    }

    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult> {
        let n = circuit.num_qubits as usize;

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => rand::make_rng::<StdRng>(),
        };

        match config.shots {
            // Exact-state mode: a single state vector can only carry one
            // trajectory (it can't represent a mixed state), so we return the
            // final state of one evolution. Callers wanting the noisy
            // distribution use the sampling path below.
            None => {
                let (state, _cbits) = self.evolve(circuit, params, config, &mut rng)?;
                Ok(ExecResult::Statevector(state))
            }
            // Per-trajectory Monte-Carlo when a channel acts during evolution
            // OR when mid-circuit measurement collapses the state (the latter is
            // inherently stochastic per shot, so it can't reuse one evolution).
            Some(shots)
                if self.model.has_gate_channel() || stochastic_evolution(circuit, config) =>
            {
                // One independent trajectory per shot — this is what turns a
                // per-gate channel (amplitude damping, depolarizing, …) into the
                // right shot statistics instead of one branch drawn once and
                // replayed for every shot.
                let collapse = collapses(circuit, config);
                let mut counts: HashMap<u64, u32> = HashMap::new();
                for _ in 0..shots {
                    let (state, cbits) = self.evolve(circuit, params, config, &mut rng)?;
                    let outcome = if collapse {
                        // Mid-circuit measures already recorded the (readout-
                        // flipped) outcomes into the creg during evolution; key
                        // by the creg to match the noiseless backend.
                        creg_to_u64(&cbits)
                    } else {
                        let sample = sample_counts(&state, n, 1, Some(rng.random()));
                        // `sample_counts` with shots=1 yields exactly one key.
                        let mut o = sample.into_keys().next().unwrap_or(0);
                        if !self.model.readout.is_zero() {
                            o = flip_readout_bits(o, n, &self.model.readout, &mut rng);
                        }
                        o
                    };
                    *counts.entry(outcome).or_insert(0) += 1;
                }
                Ok(ExecResult::counts_from_u64(
                    counts,
                    omega_core::executor::counts_outcome_width(circuit, collapse) as u32,
                ))
            }
            Some(shots) => {
                // Fast path: no channel acts during evolution, so a single
                // evolution's state is exact and every shot samples from it. Any
                // readout flip is applied independently per shot afterwards.
                let (state, _cbits) = self.evolve(circuit, params, config, &mut rng)?;
                let mut counts = sample_counts(&state, n, shots, Some(rng.random()));
                if !self.model.readout.is_zero() {
                    counts = apply_readout_flip(counts, n, &self.model.readout, &mut rng);
                }
                Ok(ExecResult::counts_from_u64(counts, n as u32))
            }
        }
    }
}

/// True when this run collapses mid-circuit measurements into the classical
/// register (so each shot is an independent stochastic trajectory keyed by the
/// creg, matching the noiseless backend).
fn collapses(circuit: &CircuitIR, config: &ExecConfig) -> bool {
    config.mid_circuit_mode == MidCircuitMode::Collapse && circuit.num_classical_bits > 0
}

/// True when evolution is stochastic *regardless* of the noise model, so one
/// evolution cannot stand in for every shot. `Reset` samples a Born-rule
/// outcome per trajectory (see [`apply_reset`]); collapse-mode measurement
/// does too.
fn stochastic_evolution(circuit: &CircuitIR, config: &ExecConfig) -> bool {
    collapses(circuit, config) || circuit_has_reset(circuit)
}

/// Pack the classical register into a `u64` counts key. One definition for
/// every backend — see [`omega_core::executor::creg_to_u64`].
use omega_core::executor::creg_to_u64;

/// Flip each qubit `q` of one measured outcome with its (bit-dependent,
/// per-qubit) readout-error probability. `readout` may be asymmetric, so the
/// flip probability depends on the qubit's *true* bit.
fn flip_readout_bits(
    bits: u64,
    n: usize,
    readout: &omega_core::noise::ReadoutError,
    rng: &mut impl Rng,
) -> u64 {
    let mut flipped = bits;
    for q in 0..n {
        let true_bit = ((bits >> q) & 1) as u8;
        let p = readout.flip_prob(q, true_bit);
        if p > 0.0 && rng.random::<f64>() < p {
            flipped ^= 1u64 << q;
        }
    }
    flipped
}

/// Apply per-qubit readout error independently to every shot in `counts`.
fn apply_readout_flip(
    counts: HashMap<u64, u32>,
    n: usize,
    readout: &omega_core::noise::ReadoutError,
    rng: &mut impl Rng,
) -> HashMap<u64, u32> {
    let mut out: HashMap<u64, u32> = HashMap::new();
    for (bits, count) in counts {
        for _ in 0..count {
            let flipped = flip_readout_bits(bits, n, readout, rng);
            *out.entry(flipped).or_insert(0) += 1;
        }
    }
    out
}

/// Purity `Tr(ρ_q²)` of qubit `q`'s reduced state, where ρ_q is the 2×2 matrix
/// obtained by tracing out every other qubit. Equals 1 exactly when `q` is
/// **unentangled** from the rest of the register.
pub fn reduced_purity(state: &[Complex64], n: usize, q: usize) -> f64 {
    let dim = 1usize << n;
    let mask = 1usize << q;
    let (mut r00, mut r11) = (0.0_f64, 0.0_f64);
    let mut r01 = Complex64::new(0.0, 0.0);
    for i in 0..dim {
        if i & mask == 0 {
            r00 += state[i].norm_sqr();
            r01 += state[i] * state[i | mask].conj();
        } else {
            r11 += state[i].norm_sqr();
        }
    }
    r00 * r00 + r11 * r11 + 2.0 * r01.norm_sqr()
}

/// Whether resetting `q` is a *deterministic* operation on this state.
///
/// If `q` is unentangled (ρ_q pure) the register factorises as |φ⟩_q ⊗ |rest⟩,
/// and reset yields |0⟩ ⊗ |rest⟩ whatever outcome is sampled — projecting onto
/// |0⟩, or onto |1⟩ and then flipping, give the *same* state. So the analytic
/// answer is exact and `apply_reset` may run with any RNG.
///
/// If `q` is entangled, reset genuinely decoheres the partners: the result is
/// mixed, a statevector holds one trajectory, and there is no exact pure-state
/// answer. That is the case the caller must be refused for.
pub fn reset_is_deterministic(state: &[Complex64], n: usize, q: usize) -> bool {
    reset_is_deterministic_within(state, n, q, 1e-9)
}

/// [`reset_is_deterministic`] with an explicit tolerance.
///
/// The f64 CPU path uses `1e-9`. A **f32** backend (Metal) must not: its
/// amplitudes come back rounded, so an genuinely unentangled qubit reads a
/// purity of `1 − O(1e-6)` and `1e-9` rejects it as entangled. The threshold is
/// not delicate — an unentangled qubit sits within ~1e-6 of 1 while a maximally
/// entangled one sits at 0.5, five orders of magnitude away — so a device-
/// appropriate tolerance separates them cleanly without weakening the check.
pub fn reset_is_deterministic_within(state: &[Complex64], n: usize, q: usize, tol: f64) -> bool {
    (reduced_purity(state, n, q) - 1.0).abs() < tol
}

/// Error for an analytic (`shots = None`) expectation whose `Reset` acts on an
/// entangled qubit — see [`reset_is_deterministic`].
fn entangled_reset_in_analytic_mode(q: usize) -> OmegaError {
    OmegaError::Unsupported(format!(
        "statevector: analytic expectation of Reset on qubit {q} is ill-defined — the qubit is \
         entangled, so the reset leaves the register in a mixed state that one statevector \
         cannot represent. Run with shots (each shot is an independent trajectory), or reset \
         only unentangled qubits."
    ))
}

/// True when the circuit contains a `Reset`, i.e. a stochastic channel whose
/// randomness must be redrawn per shot. See [`apply_reset`].
pub(crate) fn circuit_has_reset(circuit: &CircuitIR) -> bool {
    circuit
        .ops
        .iter()
        .any(|op| matches!(op.gate, GateKind::Reset))
}

/// Evolve |0…0⟩ through `circuit` once, returning the final state and the
/// classical register. One call = one trajectory: every `Reset` (and, in
/// `Collapse` mode, every mid-circuit `Measure`) draws a fresh outcome from
/// `rng`, so callers wanting shot statistics must invoke this once per shot.
fn evolve_once(
    circuit: &CircuitIR,
    params: &ParameterBinding,
    config: &ExecConfig,
    rng: &mut impl Rng,
    analytic: bool,
) -> Result<(Vec<Complex64>, Vec<u8>)> {
    // Same refusal as `evolve`: this is the other site that allocated `2^n`
    // amplitudes straight off user input.
    crate::capacity::check(circuit.num_qubits, config.shots.is_some())?;
    let n = circuit.num_qubits as usize;
    let dim = 1usize << n;

    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0);
    let mut classical_bits = vec![0u8; circuit.num_classical_bits as usize];

    for op in &circuit.ops {
        // Check classical condition (multi-bit creg comparison per QASM2
        // semantics; see GateOp::condition_satisfied).
        if !op.condition_satisfied(&classical_bits) {
            continue;
        }
        match &op.gate {
            GateKind::Measure => {
                if config.mid_circuit_mode == MidCircuitMode::Collapse {
                    let q = op.qubits[0].0 as usize;
                    let outcome = projective_measure(&mut state, n, q, rng);
                    if let Some(cbit) = op.classical_bit {
                        if (cbit as usize) < classical_bits.len() {
                            classical_bits[cbit as usize] = outcome;
                        }
                    }
                }
                // MidCircuitMode::Skip → do nothing (backward compat)
            }
            GateKind::Barrier => continue,
            _ => {
                apply_gate(&mut state, n, op, params, rng, analytic)?;
            }
        }
    }
    Ok((state, classical_bits))
}

/// `rng` is consumed only by [`GateKind::Reset`], which is a stochastic
/// channel rather than a gate — see [`apply_reset`].
fn apply_gate(
    state: &mut [Complex64],
    n: usize,
    op: &GateOp,
    params: &ParameterBinding,
    rng: &mut impl Rng,
    analytic: bool,
) -> Result<()> {
    // Resolve parameters
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<Result<Vec<_>>>()?;

    match &op.gate {
        // Single-qubit gates
        GateKind::H => apply_1q(state, n, op.qubits[0].0 as usize, &gates::h()),
        GateKind::X => apply_1q(state, n, op.qubits[0].0 as usize, &gates::x()),
        GateKind::Y => apply_1q(state, n, op.qubits[0].0 as usize, &gates::y()),
        GateKind::Z => diag_from(state, n, op.qubits[0].0 as usize, &gates::z()),
        GateKind::S => diag_from(state, n, op.qubits[0].0 as usize, &gates::s()),
        GateKind::Sdg => diag_from(state, n, op.qubits[0].0 as usize, &gates::sdg()),
        GateKind::Sx => apply_1q(state, n, op.qubits[0].0 as usize, &gates::sx()),
        GateKind::Sxdg => apply_1q(state, n, op.qubits[0].0 as usize, &gates::sxdg()),
        GateKind::T => diag_from(state, n, op.qubits[0].0 as usize, &gates::t()),
        GateKind::Tdg => diag_from(state, n, op.qubits[0].0 as usize, &gates::tdg()),
        GateKind::Id => {} // no-op
        GateKind::Rx => apply_1q(state, n, op.qubits[0].0 as usize, &gates::rx(resolved[0])),
        GateKind::Ry => apply_1q(state, n, op.qubits[0].0 as usize, &gates::ry(resolved[0])),
        GateKind::Rz => diag_from(state, n, op.qubits[0].0 as usize, &gates::rz(resolved[0])),
        GateKind::U3 => apply_1q(
            state,
            n,
            op.qubits[0].0 as usize,
            &gates::u3(resolved[0], resolved[1], resolved[2]),
        ),
        GateKind::U2 => apply_1q(
            state,
            n,
            op.qubits[0].0 as usize,
            &gates::u2(resolved[0], resolved[1]),
        ),
        GateKind::U1 => apply_1q(state, n, op.qubits[0].0 as usize, &gates::u1(resolved[0])),

        // Two-qubit gates
        // CX is a permutation; running it as a dense 4x4 costs 16 complex
        // multiplies and 12 adds per group to accomplish one swap.
        GateKind::CX => apply_cx(state, n, op.qubits[0].0 as usize, op.qubits[1].0 as usize),
        GateKind::CY => apply_2q(
            state,
            n,
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::cy(),
        ),
        GateKind::CZ => apply_2q(
            state,
            n,
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::cz(),
        ),
        GateKind::Swap => apply_2q(
            state,
            n,
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::swap(),
        ),
        GateKind::CRz => apply_2q(
            state,
            n,
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::crz(resolved[0]),
        ),
        GateKind::CU3 => apply_2q(
            state,
            n,
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::cu3(resolved[0], resolved[1], resolved[2]),
        ),
        GateKind::Rbs => apply_2q(
            state,
            n,
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::rbs(resolved[0]),
        ),

        // Three-qubit: CCX (Toffoli)
        GateKind::CCX => apply_ccx(
            state,
            n,
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            op.qubits[2].0 as usize,
        ),
        GateKind::CSwap => apply_cswap(
            state,
            n,
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            op.qubits[2].0 as usize,
        ),

        GateKind::Reset => apply_reset(state, n, op.qubits[0].0 as usize, rng, analytic)?,

        GateKind::Measure | GateKind::Barrier => {}

        _ => {
            return Err(OmegaError::Unsupported(format!(
                "gate {:?} not supported in statevector backend",
                op.gate
            )));
        }
    }

    Ok(())
}

/// Reset qubit `q` to |0⟩ — the reset **channel** ρ → |0⟩⟨0|_q ⊗ Tr_q(ρ).
///
/// Implemented as **sample → project → flip**, which is what Qiskit Aer's
/// statevector method does: measure `q` in the computational basis with the
/// trajectory RNG, collapse onto the sampled outcome, then apply X if that
/// outcome was 1. The qubit ends in |0⟩ on every trajectory; averaged over
/// shots the *rest* of the register sees the correct decohered marginal.
///
/// The randomness is essential and must live in the per-shot trajectory (see
/// [`needs_trajectories`]): a single statevector cannot represent the mixed
/// state the channel produces when `q` is entangled, so the ensemble has to
/// come from running independent shots.
///
/// **Do not "fold" the amplitudes.** The previous implementation applied the
/// non-unitary matvec `[[1,1],[0,0]]` (`state[i0] += state[i1]`) and
/// renormalised. That is not a reset: it turns entanglement with the rest of
/// the register into a *coherent* superposition. On Bell + `reset q0` it gave
/// ⟨X₁⟩ = +1 where the channel gives 0, and on a qubit in |−⟩ the two
/// amplitudes cancelled to zero — the `norm_sq > 0.0` guard then skipped
/// renormalisation and left an all-zero state, so a freshly reset qubit
/// sampled as |1⟩ on every shot. Pinned by `tests/reset_channel.rs`.
///
/// `analytic` marks a `shots = None` run, where there is no ensemble to average
/// over; a reset on an entangled qubit is then refused rather than silently
/// returning one RNG-dependent trajectory (see [`reset_is_deterministic`]).
fn apply_reset(
    state: &mut [Complex64],
    n: usize,
    q: usize,
    rng: &mut impl Rng,
    analytic: bool,
) -> Result<()> {
    if analytic && !reset_is_deterministic(state, n, q) {
        return Err(entangled_reset_in_analytic_mode(q));
    }
    if projective_measure(state, n, q, rng) == 1 {
        apply_1q(state, n, q, &gates::x());
    }
    Ok(())
}

/// Projective measurement: collapse qubit `q`, return outcome (0 or 1).
///
/// 1. Compute P(q=0) from amplitudes
/// 2. Sample outcome using RNG
/// 3. Zero amplitudes inconsistent with outcome
/// 4. Renormalize
fn projective_measure(state: &mut [Complex64], n: usize, q: usize, rng: &mut impl Rng) -> u8 {
    let dim = 1usize << n;
    let mask = 1usize << q;

    // Compute probability of outcome 0
    let p0: f64 = (0..dim)
        .filter(|&i| i & mask == 0)
        .map(|i| state[i].norm_sqr())
        .sum();

    // Sample outcome
    let outcome: u8 = if rng.random::<f64>() < p0 { 0 } else { 1 };

    // Collapse: zero amplitudes inconsistent with outcome
    for i in 0..dim {
        let bit = (i >> q) & 1;
        if bit != outcome as usize {
            state[i] = Complex64::new(0.0, 0.0);
        }
    }

    // Renormalize
    let norm_sq: f64 = state.iter().map(|a| a.norm_sqr()).sum();
    if norm_sq > 0.0 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for a in state.iter_mut() {
            *a *= inv_norm;
        }
    }

    outcome
}

/// Apply a single-qubit gate to qubit `q` in an n-qubit statevector.
/// Below this amplitude count a circuit is faster single-threaded: the pool
/// handoff costs more than the work. 2^12 = 4096 amplitudes (64 KiB), measured
/// as roughly where the two cross on this class of machine.
pub(crate) const PAR_MIN_DIM: usize = 1 << 12;

/// Minimum slice length worth handing to the pool on the INNER axis. Without a
/// separate floor here, a high-`q` gate — where the outer axis has exactly one
/// chunk — would either serialise entirely or spawn per-element tasks.
const PAR_MIN_INNER: usize = 1 << 10;

pub(crate) fn apply_1q(state: &mut [Complex64], n: usize, q: usize, gate: &gates::Gate1Q) {
    use rayon::prelude::*;
    let dim = 1usize << n;
    let step = 1usize << q;
    let g = *gate;

    // One pair of amplitudes, `step` apart. The whole kernel is this, applied
    // to dim/2 disjoint pairs.
    #[inline(always)]
    fn kernel(g: &gates::Gate1Q, a: &mut Complex64, b: &mut Complex64) {
        let (a0, a1) = (*a, *b);
        *a = g[0] * a0 + g[1] * a1;
        *b = g[2] * a0 + g[3] * a1;
    }

    if dim < PAR_MIN_DIM {
        let mut i = 0;
        while i < dim {
            let (lo, hi) = state[i..i + (step << 1)].split_at_mut(step);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                kernel(&g, a, b);
            }
            i += step << 1;
        }
        return;
    }

    // TWO axes, and both are needed. `chunks_mut(2·step)` yields `dim/(2·step)`
    // chunks, which is **one** when `q == n-1` — the widest, most expensive
    // gate in any circuit. A naive `par_chunks_mut` therefore silently
    // serialises exactly the case that matters most, so the inner pairing is
    // parallelised too when the chunk is big enough to be worth it.
    state.par_chunks_mut(step << 1).for_each(|chunk| {
        let (lo, hi) = chunk.split_at_mut(step);
        if step >= PAR_MIN_INNER {
            lo.par_iter_mut()
                .zip(hi.par_iter_mut())
                .for_each(|(a, b)| kernel(&g, a, b));
        } else {
            lo.iter_mut()
                .zip(hi.iter_mut())
                .for_each(|(a, b)| kernel(&g, a, b));
        }
    });
}

/// Apply a two-qubit gate to qubits (q0, q1) in an n-qubit statevector.
/// q0 is the more significant qubit in the gate's matrix (control for CX, etc.).
/// Test/bench-only shim: `apply_2q` is `pub(crate)`, and the sparse-path
/// benchmark lives in `examples/` which is a separate crate.
#[doc(hidden)]
pub fn apply_2q_pub(state: &mut [Complex64], n: usize, q0: usize, q1: usize, gate: &gates::Gate2Q) {
    apply_2q(state, n, q0, q1, gate)
}

pub(crate) fn apply_2q(
    state: &mut [Complex64],
    n: usize,
    q0: usize,
    q1: usize,
    gate: &gates::Gate2Q,
) {
    let dim = 1usize << n;

    // Controlled gates first, and on the UNRELABELLED matrix: `q0` is the
    // control in this file's `Gate2Q` convention, and the relabel below would
    // move the block. Diagonal gates are also "controlled" in this sense, but
    // the diagonal path below is cheaper still, so let them fall through.
    if diagonal_2q(gate).is_none() {
        if let Some(m) = controlled_1q(gate) {
            apply_controlled_1q(state, n, q0, q1, m);
            return;
        }
    }

    // Ensure q0 > q1 for iteration order; swap and transpose gate if needed
    let (qa, qb, g) = if q0 > q1 {
        (q0, q1, *gate)
    } else {
        // Swap qubits: need to relabel |01> <-> |10> in gate matrix
        let mut swapped = *gate;
        // Swap rows 1,2 and columns 1,2
        // Row swap: swap rows 1 and 2
        for col in 0..4 {
            swapped.swap(4 + col, 2 * 4 + col);
        }
        // Column swap: swap columns 1 and 2
        for row in 0..4 {
            swapped.swap(row * 4 + 1, row * 4 + 2);
        }
        (q1, q0, swapped)
    };

    let step_a = 1usize << qa;
    let step_b = 1usize << qb;
    // `qa == qb` means the caller passed the same qubit twice — a two-qubit
    // gate applied to one qubit. It is malformed input, not a degenerate case
    // worth supporting: Qiskit refuses it outright ("duplicate qubit
    // arguments"), and so should we.
    //
    // The scan-and-reject loop this replaced did NOT refuse it. With
    // `step_a == step_b` its four indices aliased (i01 == i10 == i11), so the
    // three later rows overwrote each other and the gate silently became a
    // 2x2 built from summed columns — a confident wrong answer, and one no
    // test caught because the harness that would have compares two paths
    // sharing the same lowered IR.
    //
    // A real `assert!`, not `debug_assert!`: the group walk indexes an empty
    // slice in this case, so a release build would panic anyway, with a bare
    // index-out-of-bounds instead of the reason.
    assert!(
        qa > qb,
        "apply_2q: a two-qubit gate on one qubit (q0 == q1 == {q0}) — \
         a controlled gate cannot control itself"
    );
    debug_assert_eq!(state.len(), dim);

    // A DIAGONAL two-qubit gate costs four complex multiplies per group, not
    // sixteen multiplies and twelve adds. Detected from the matrix rather than
    // routed at the call site, so `CZ`, `CRz`, `CU1`/`CPhase` and any diagonal
    // gate added later all take it with no dispatch to keep in sync — the
    // failure mode of a hand-maintained list being a fast path that silently
    // stops covering a gate someone added.
    //
    // Checked on `g`, i.e. AFTER the q0/q1 relabel above, so the entries are
    // already in the (qa, qb) basis the walk uses.
    //
    // Exact, not approximate. The dense path computes `g[0]*a00 + g[1]*a01 +
    // g[2]*a10 + g[3]*a11` where the last three coefficients are exactly
    // `0.0`; `0.0 * x` is `0.0` and `y + 0.0` is `y` for every finite `y`, so
    // the result is BIT-IDENTICAL rather than merely close. `sparse_2q_is_bit_
    // identical_to_the_dense_path` pins that.
    if let Some(d) = diagonal_2q(&g) {
        apply_diagonal_2q_ordered(state, n, qa, qb, d);
        return;
    }
    // SWAP is a permutation: it exchanges |01> and |10> and touches nothing
    // else, so it is pure data movement — no arithmetic to associate, and
    // therefore bit-identical for the same reason `apply_cx` is. Detected from
    // the matrix for the same reason the diagonal is: no dispatch table to
    // fall out of sync with the gate set.
    //
    // SWAP is symmetric under the q0/q1 relabel, so checking `g` rather than
    // `gate` is not load-bearing here — but it is what keeps this consistent
    // with the diagonal check above, where it very much is.
    if is_swap_2q(&g) {
        apply_swap_ordered(state, n, qa, qb);
        return;
    }
    // `Rbs` and any other gate confined to the {|01>, |10>} subspace: |00> and
    // |11> are fixed points, so half the state is never read.
    if let Some(m) = middle_block_2q(&g) {
        apply_middle_block_2q_ordered(state, n, qa, qb, m);
        return;
    }

    // Walk the `dim/4` groups directly instead of scanning all `dim` indices
    // and rejecting three of every four. The four amplitudes of a group are the
    // four slices below, so the same four expressions run with no index
    // arithmetic and no branch in the loop body:
    //
    //   chunk  = one block of 2·step_a   -> split at step_a  gives qa = 0 | 1
    //   c0, c1 = one block of 2·step_b   -> split at step_b  gives qb = 0 | 1
    //
    // `qa > qb` makes `step_a` a multiple of `2·step_b`, so every inner chunk
    // is whole and no group straddles a boundary. The four slices are disjoint
    // by construction — which is also what makes `PLAN-SV-PERF.md` §3.2's
    // parallel step safe without `unsafe`.
    use rayon::prelude::*;

    // The four slices are disjoint by construction, which is what makes this
    // safe without `unsafe`. As in `apply_1q`, the outer axis collapses to one
    // chunk when `qa == n-1`, so the inner group walk is parallelised too.
    let apply_group = |x00: &mut [Complex64],
                       x01: &mut [Complex64],
                       x10: &mut [Complex64],
                       x11: &mut [Complex64]| {
        for k in 0..step_b {
            let a00 = x00[k];
            let a01 = x01[k];
            let a10 = x10[k];
            let a11 = x11[k];

            x00[k] = g[0] * a00 + g[1] * a01 + g[2] * a10 + g[3] * a11;
            x01[k] = g[4] * a00 + g[5] * a01 + g[6] * a10 + g[7] * a11;
            x10[k] = g[8] * a00 + g[9] * a01 + g[10] * a10 + g[11] * a11;
            x11[k] = g[12] * a00 + g[13] * a01 + g[14] * a10 + g[15] * a11;
        }
    };

    if dim < PAR_MIN_DIM {
        for chunk in state.chunks_mut(step_a << 1) {
            let (a_lo, a_hi) = chunk.split_at_mut(step_a);
            for (c0, c1) in a_lo
                .chunks_mut(step_b << 1)
                .zip(a_hi.chunks_mut(step_b << 1))
            {
                let (x00, x01) = c0.split_at_mut(step_b);
                let (x10, x11) = c1.split_at_mut(step_b);
                apply_group(x00, x01, x10, x11);
            }
        }
        return;
    }

    state.par_chunks_mut(step_a << 1).for_each(|chunk| {
        let (a_lo, a_hi) = chunk.split_at_mut(step_a);
        if step_a >= PAR_MIN_INNER {
            a_lo.par_chunks_mut(step_b << 1)
                .zip(a_hi.par_chunks_mut(step_b << 1))
                .for_each(|(c0, c1)| {
                    let (x00, x01) = c0.split_at_mut(step_b);
                    let (x10, x11) = c1.split_at_mut(step_b);
                    apply_group(x00, x01, x10, x11);
                });
        } else {
            a_lo.chunks_mut(step_b << 1)
                .zip(a_hi.chunks_mut(step_b << 1))
                .for_each(|(c0, c1)| {
                    let (x00, x01) = c0.split_at_mut(step_b);
                    let (x10, x11) = c1.split_at_mut(step_b);
                    apply_group(x00, x01, x10, x11);
                });
        }
    });
}

/// **The exactness contract for every sparse 2q dispatch in this file.**
///
/// **This was all written down before any of it was implemented**, and the 2q
/// paths were built without reading it. Three places already had it right:
///
/// * `PLAN-SV-PERF.md` §S1b — the plan these kernels implement — says "a dense
///   row computes `1*a00 + 0*a01 + 0*a10 + 0*a11`, and `0.0 * (-x)` is `-0.0`,
///   so the dense path maps some signed zeros to the opposite sign where the
///   specialised path preserves them. Numerically identical; not
///   bit-identical." It even prescribes the test: "equal under `==`
///   everywhere, and `to_bits()`-equal everywhere the value is non-zero."
/// * `apply_diagonal_1q` states it for the 1q case, and
///   `the_diagonal_path_differs_only_in_the_sign_of_zero` pins it — with a
///   warning aimed at "anyone who tightens the oracle above back to
///   `to_bits()`", which is exactly what the 2q tests then did.
/// * `kernels/apply_quad_perm.cu` states it for the CUDA permutation kernels
///   and points back at `PLAN-SV-PERF.md`.
///
/// Three wrong exactness claims shipped anyway. The answer being written down
/// is not the same as it being read, so it is repeated HERE, next to the
/// kernels it constrains, and asserted mechanically by
/// `every_2q_gate_differs_from_the_dense_scan_only_in_the_sign_of_zero` rather
/// than left as prose for a fourth reader to miss.
///
/// A dispatch that SKIPS reading amplitudes cannot be bit-identical to the
/// dense 4x4, and no amount of adding `0.0` fixes it. The dense row is
/// `g0*a00 + g1*a01 + g2*a10 + g3*a11` with some coefficients exactly zero,
/// and `0.0 * a` is `-0.0` when `a` is negative — so the dense RESULT carries
/// sign information from amplitudes the fast path never reads. MEASURED, by
/// enumerating the sign combinations:
///
/// * diagonal (`x00 *= d0` vs the dense row): **7 of 16** combinations differ
/// * controlled-U: 3 of 16 bare; adding a leading `0.0` reduces it to 1, which
///   is why an earlier revision of this file carried one and claimed it was
///   load-bearing. It is not: it trades three divergences for one and buys no
///   guarantee, so it is gone.
///
/// The difference is ALWAYS and ONLY the sign of a zero: `-0.0` vs `+0.0`.
/// Every non-zero amplitude is bit-identical. `-0.0 == +0.0`, |z|^2 is
/// identical and sampling is identical, so nothing this project computes can
/// see it — but `--dump-state-bits` and `tools/biteq` can, by construction,
/// and `diff.py` classifies such differences as dispatch artefacts rather than
/// divergence.
///
/// **There is no exception, including the permutations.** `apply_swap_ordered`
/// and `apply_cx` compute nothing, which is true and irrelevant: the
/// difference comes from the region a kernel does NOT write, not from
/// arithmetic in the region it does. SWAP fixes |00> and |11>, CX fixes the
/// control-zero half, and the dense path writes those and normalises their
/// zeros. An earlier revision exempted permutations in prose;
/// `even_the_permutation_dispatches_differ_in_the_sign_of_zero` now pins the
/// opposite.
///
/// The diagonal of a two-qubit gate, if it has one.
///
/// `Some([g00, g11, g22, g33])` iff all twelve off-diagonal entries are
/// EXACTLY zero. Exact equality is deliberate: a near-zero off-diagonal is a
/// real coupling, and treating it as absent would silently change the physics.
/// This is a fast path for gates that are diagonal by construction, not a
/// tolerance-based approximation of ones that nearly are.
fn diagonal_2q(g: &gates::Gate2Q) -> Option<[Complex64; 4]> {
    for row in 0..4 {
        for col in 0..4 {
            if row != col && g[row * 4 + col] != Complex64::new(0.0, 0.0) {
                return None;
            }
        }
    }
    Some([g[0], g[5], g[10], g[15]])
}

/// A diagonal two-qubit gate: scale each of the four components in place.
///
/// `qa > qb` is required — callers order the pair and relabel the diagonal
/// into that basis, exactly as `apply_2q` does for a dense matrix.
pub(crate) fn apply_diagonal_2q_ordered(
    state: &mut [Complex64],
    n: usize,
    qa: usize,
    qb: usize,
    d: [Complex64; 4],
) {
    use rayon::prelude::*;
    assert!(qa > qb, "apply_diagonal_2q_ordered: need qa > qb");
    let dim = 1usize << n;
    let step_a = 1usize << qa;
    let step_b = 1usize << qb;

    // Same disjoint four-slice walk as `apply_2q`; only the body differs.
    let apply_group = |x00: &mut [Complex64],
                       x01: &mut [Complex64],
                       x10: &mut [Complex64],
                       x11: &mut [Complex64]| {
        for k in 0..step_b {
            x00[k] *= d[0];
            x01[k] *= d[1];
            x10[k] *= d[2];
            x11[k] *= d[3];
        }
    };

    if dim < PAR_MIN_DIM {
        for chunk in state.chunks_mut(step_a << 1) {
            let (a_lo, a_hi) = chunk.split_at_mut(step_a);
            for (c0, c1) in a_lo
                .chunks_mut(step_b << 1)
                .zip(a_hi.chunks_mut(step_b << 1))
            {
                let (x00, x01) = c0.split_at_mut(step_b);
                let (x10, x11) = c1.split_at_mut(step_b);
                apply_group(x00, x01, x10, x11);
            }
        }
        return;
    }
    state.par_chunks_mut(step_a << 1).for_each(|chunk| {
        let (a_lo, a_hi) = chunk.split_at_mut(step_a);
        a_lo.chunks_mut(step_b << 1)
            .zip(a_hi.chunks_mut(step_b << 1))
            .for_each(|(c0, c1)| {
                let (x00, x01) = c0.split_at_mut(step_b);
                let (x10, x11) = c1.split_at_mut(step_b);
                apply_group(x00, x01, x10, x11);
            });
    });
}

/// Is this the SWAP matrix, exactly?
///
/// Exact equality, as in [`diagonal_2q`]: a near-SWAP is a different gate, and
/// running it as a permutation would discard the difference silently.
fn is_swap_2q(g: &gates::Gate2Q) -> bool {
    const ONE: Complex64 = Complex64::new(1.0, 0.0);
    const ZERO: Complex64 = Complex64::new(0.0, 0.0);
    let want: [Complex64; 16] = [
        ONE, ZERO, ZERO, ZERO, //
        ZERO, ZERO, ONE, ZERO, //
        ZERO, ONE, ZERO, ZERO, //
        ZERO, ZERO, ZERO, ONE,
    ];
    g.iter().zip(want.iter()).all(|(a, b)| a == b)
}

/// SWAP as the permutation it is: exchange the |01> and |10> components.
///
/// `qa > qb` required. The MOVED amplitudes are bit-identical — there is no
/// arithmetic to round — but see [`diagonal_2q`]'s signed-zero note: |00> and
/// |11> are fixed points this never writes, and the dense path does write
/// them, normalising `-0.0` to `+0.0`. Moving nothing is not the same as
/// writing the same thing.
pub(crate) fn apply_swap_ordered(state: &mut [Complex64], n: usize, qa: usize, qb: usize) {
    use rayon::prelude::*;
    assert!(qa > qb, "apply_swap_ordered: need qa > qb");
    let dim = 1usize << n;
    let step_a = 1usize << qa;
    let step_b = 1usize << qb;

    // Only the x01/x10 pair moves; x00 and x11 are fixed points of SWAP, so
    // they are not even read.
    let swap_group = |x01: &mut [Complex64], x10: &mut [Complex64]| {
        for k in 0..step_b {
            std::mem::swap(&mut x01[k], &mut x10[k]);
        }
    };

    if dim < PAR_MIN_DIM {
        for chunk in state.chunks_mut(step_a << 1) {
            let (a_lo, a_hi) = chunk.split_at_mut(step_a);
            for (c0, c1) in a_lo
                .chunks_mut(step_b << 1)
                .zip(a_hi.chunks_mut(step_b << 1))
            {
                let (_x00, x01) = c0.split_at_mut(step_b);
                let (x10, _x11) = c1.split_at_mut(step_b);
                swap_group(x01, x10);
            }
        }
        return;
    }
    state.par_chunks_mut(step_a << 1).for_each(|chunk| {
        let (a_lo, a_hi) = chunk.split_at_mut(step_a);
        a_lo.chunks_mut(step_b << 1)
            .zip(a_hi.chunks_mut(step_b << 1))
            .for_each(|(c0, c1)| {
                let (_x00, x01) = c0.split_at_mut(step_b);
                let (x10, _x11) = c1.split_at_mut(step_b);
                swap_group(x01, x10);
            });
    });
}

/// The 2x2 block of a **controlled** two-qubit gate, if it is one.
///
/// `Some(m)` iff `g` is `block-diag(I2, m)` in the convention this file's
/// `Gate2Q` uses, where the FIRST qubit is the high index bit — i.e. the gate
/// leaves the control-zero subspace alone and acts as `m` on the rest. Covers
/// `CY` and `CU3` today, and `CH`/`CU1`/any controlled gate added later.
///
/// Tested on the ORIGINAL matrix, before `apply_2q`'s q0/q1 relabel, because
/// the relabel destroys this structure: with `q0 < q1` the control becomes the
/// low bit and the block moves off the corner. Working from the unrelabelled
/// matrix and passing the qubits to a mask-based kernel avoids the question
/// entirely, which is also why `apply_cx` is written that way.
fn controlled_1q(g: &gates::Gate2Q) -> Option<gates::Gate1Q> {
    const ONE: Complex64 = Complex64::new(1.0, 0.0);
    const ZERO: Complex64 = Complex64::new(0.0, 0.0);
    // Row/col 0 and 1 must be exactly the identity, and rows 2,3 must not
    // reach back into the control-zero columns.
    let identity_block = g[0] == ONE
        && g[1] == ZERO
        && g[2] == ZERO
        && g[3] == ZERO
        && g[4] == ZERO
        && g[5] == ONE
        && g[6] == ZERO
        && g[7] == ZERO;
    let no_leak = g[8] == ZERO && g[9] == ZERO && g[12] == ZERO && g[13] == ZERO;
    if identity_block && no_leak {
        Some([g[10], g[11], g[14], g[15]])
    } else {
        None
    }
}

/// A controlled 1q gate: apply `m` to the target, but only where the control
/// bit is set.
///
/// Half the state is not read at all — the control-zero subspace is a fixed
/// point, and the dense path writes it back unchanged after four multiplies
/// and three adds per amplitude.
///
/// Mask-based like [`apply_cx`], so `control` and `target` are used directly
/// and no gate relabel is involved.
/// The middle 2x2 block of a gate that is the identity on |00> and |11>.
///
/// `Some([m00, m01, m10, m11])` iff `g` acts only inside the {|01>, |10>}
/// subspace. `Rbs` is the case that motivated it; an XY / iSWAP-family gate
/// has the same shape.
///
/// Safe to test AFTER `apply_2q`'s q0/q1 relabel: the relabel swaps indices 1
/// and 2, which permutes the block within itself and leaves |00>/|11> alone,
/// so the FORM survives and the entries come back already in the walk's basis.
fn middle_block_2q(g: &gates::Gate2Q) -> Option<gates::Gate1Q> {
    const ONE: Complex64 = Complex64::new(1.0, 0.0);
    let corners = g[0] == ONE && g[15] == ONE;
    const ZERO: Complex64 = Complex64::new(0.0, 0.0);
    let no_leak = g[1] == ZERO
        && g[2] == ZERO
        && g[3] == ZERO
        && g[4] == ZERO
        && g[7] == ZERO
        && g[8] == ZERO
        && g[11] == ZERO
        && g[12] == ZERO
        && g[13] == ZERO
        && g[14] == ZERO;
    if corners && no_leak {
        Some([g[5], g[6], g[9], g[10]])
    } else {
        None
    }
}

/// A gate that rotates only inside the {|01>, |10>} subspace.
///
/// `qa > qb` required. |00> and |11> are fixed points and are not read — half
/// the state untouched, where the dense path writes both back unchanged after
/// four multiplies and three adds each.
pub(crate) fn apply_middle_block_2q_ordered(
    state: &mut [Complex64],
    n: usize,
    qa: usize,
    qb: usize,
    m: gates::Gate1Q,
) {
    use rayon::prelude::*;
    assert!(qa > qb, "apply_middle_block_2q_ordered: need qa > qb");
    let dim = 1usize << n;
    let step_a = 1usize << qa;
    let step_b = 1usize << qb;

    // See [`diagonal_2q`]'s signed-zero note: this differs from the dense path
    // only in the sign of ZERO amplitudes, and cannot be made to agree
    // without reading the corners it exists to skip.
    let apply_group = |x01: &mut [Complex64], x10: &mut [Complex64]| {
        for k in 0..step_b {
            let (a, b) = (x01[k], x10[k]);
            x01[k] = m[0] * a + m[1] * b;
            x10[k] = m[2] * a + m[3] * b;
        }
    };

    if dim < PAR_MIN_DIM {
        for chunk in state.chunks_mut(step_a << 1) {
            let (a_lo, a_hi) = chunk.split_at_mut(step_a);
            for (c0, c1) in a_lo
                .chunks_mut(step_b << 1)
                .zip(a_hi.chunks_mut(step_b << 1))
            {
                let (_x00, x01) = c0.split_at_mut(step_b);
                let (x10, _x11) = c1.split_at_mut(step_b);
                apply_group(x01, x10);
            }
        }
        return;
    }
    state.par_chunks_mut(step_a << 1).for_each(|chunk| {
        let (a_lo, a_hi) = chunk.split_at_mut(step_a);
        a_lo.chunks_mut(step_b << 1)
            .zip(a_hi.chunks_mut(step_b << 1))
            .for_each(|(c0, c1)| {
                let (_x00, x01) = c0.split_at_mut(step_b);
                let (x10, _x11) = c1.split_at_mut(step_b);
                apply_group(x01, x10);
            });
    });
}

pub(crate) fn apply_controlled_1q(
    state: &mut [Complex64],
    n: usize,
    control: usize,
    target: usize,
    m: gates::Gate1Q,
) {
    use rayon::prelude::*;
    assert!(
        control != target,
        "apply_controlled_1q: control and target must differ (both {control})"
    );
    let dim = 1usize << n;
    let (hi, lo) = if control > target {
        (control, target)
    } else {
        (target, control)
    };
    let mask_c = 1usize << control;
    let mask_t = 1usize << target;

    // One iteration per acted-on PAIR: expand a compacted counter around the
    // two removed bit positions, exactly as `apply_cx` does.
    let scatter = |k: usize| -> usize {
        let low = k & ((1usize << lo) - 1);
        let rest = k >> lo;
        let mid = rest & ((1usize << (hi - lo - 1)) - 1);
        let top = rest >> (hi - lo - 1);
        low | (mid << (lo + 1)) | (top << (hi + 1))
    };
    let groups = dim >> 2;

    if dim < PAR_MIN_DIM {
        for k in 0..groups {
            let i0 = scatter(k) | mask_c;
            let i1 = i0 | mask_t;
            let (a, b) = (state[i0], state[i1]);
            // The leading `ZERO +` is NOT dead weight — do not "simplify" it
            // away. The dense path this replaces computes
            // `((0*a00 + 0*a01) + m0*a10) + m1*a11`, and `0.0 + (-0.0)` is
            // `+0.0`: without the leading add, a product landing on `-0.0`
            // comes out `-0.0` here and `+0.0` there. Equal under `==`,
            // different in bits, and this is the CPU reference every other
            // backend is bit-compared against. MEASURED: removing it fails
            // `controlled_1q_fast_path_matches_the_dense_scan_bit_for_bit`
            // on the signed-zero fixture at idx 0.
            state[i0] = m[0] * a + m[1] * b;
            state[i1] = m[2] * a + m[3] * b;
        }
        return;
    }
    let ptr = state.as_mut_ptr() as usize;
    (0..groups).into_par_iter().for_each(|k| {
        let i0 = scatter(k) | mask_c;
        let i1 = i0 | mask_t;
        // SAFETY: `scatter` is injective over 0..dim/4 and every k yields a
        // distinct (i0, i1) pair with i0 != i1, so no two iterations touch the
        // same element. Both indices are < dim by construction.
        unsafe {
            let base = ptr as *mut Complex64;
            let a = *base.add(i0);
            let b = *base.add(i1);
            // See [`diagonal_2q`]'s signed-zero note.
            *base.add(i0) = m[0] * a + m[1] * b;
            *base.add(i1) = m[2] * a + m[3] * b;
        }
    });
}

/// `CX` as the permutation it is, not a dense 4x4.
///
/// The generic path costs **16 complex multiplies and 12 adds per group** to
/// accomplish one swap, and `CX` is the most common two-qubit gate in every
/// circuit — on a GHZ chain it is the entire cost. This walks the pairs whose
/// control bit is set and swaps them.
///
/// Exact, not approximate: a permutation moves amplitudes without arithmetic,
/// so the result is **bit-identical** to the dense path rather than merely
/// close. That is what makes this safe in a way gate fusion is not — fusion
/// changes floating-point association, this changes nothing to associate.
pub(crate) fn apply_cx(state: &mut [Complex64], n: usize, control: usize, target: usize) {
    use rayon::prelude::*;
    assert!(
        control != target,
        "apply_cx: control and target must differ (both {control}) — a \
         controlled gate cannot control itself"
    );
    let dim = 1usize << n;
    let (hi, lo) = if control > target {
        (control, target)
    } else {
        (target, control)
    };
    let mask_c = 1usize << control;
    let mask_t = 1usize << target;

    // Enumerate the dim/4 free configurations directly, as `apply_ccx` does:
    // one iteration per swapped pair, no branch in the body.
    let scatter = |k: usize| -> usize {
        let low = k & ((1usize << lo) - 1);
        let rest = k >> lo;
        let mid = rest & ((1usize << (hi - lo - 1)) - 1);
        let top = rest >> (hi - lo - 1);
        low | (mid << (lo + 1)) | (top << (hi + 1))
    };
    let groups = dim >> 2;

    if dim < PAR_MIN_DIM {
        for k in 0..groups {
            let i = scatter(k) | mask_c;
            state.swap(i, i | mask_t);
        }
        return;
    }
    // Each k names a disjoint pair, so the swaps are independent. Rayon cannot
    // hand out overlapping `&mut` here without help, so split the state once
    // per pair via raw indices under a scoped pointer wrapper is avoided —
    // instead chunk the GROUP index space and let each chunk own its swaps by
    // construction (the pairs a chunk touches are determined by k alone and
    // never shared with another chunk).
    let ptr = state.as_mut_ptr() as usize;
    (0..groups).into_par_iter().for_each(|k| {
        let i = scatter(k) | mask_c;
        let j = i | mask_t;
        // SAFETY: `scatter` is injective over 0..dim/4 and every k yields a
        // distinct (i, j) pair with i != j, so no two iterations touch the same
        // element. Indices are < dim by construction.
        unsafe {
            let base = ptr as *mut Complex64;
            std::ptr::swap(base.add(i), base.add(j));
        }
    });
}

/// A single-qubit **diagonal** gate: `diag(a, d)`.
///
/// `Z`, `S`, `Sdg`, `T`, `Tdg`, `Rz`, `U1`/`P` are all of this shape. The dense
/// path does 4 multiplies and 2 adds per amplitude pair; this does one multiply
/// per amplitude and no adds, because the off-diagonal terms it is adding are
/// structurally zero.
///
/// Numerically equal to the dense path, with **one** difference worth stating:
/// the **sign of zero**. The dense path computes `a*x + 0*y`, and that addition
/// normalises `-0.0` to `+0.0`; skipping it preserves whatever sign the
/// multiply produced. So `Z` on an amplitude with `im = +0.0` yields `im =
/// -0.0` here and `+0.0` there.
///
/// That is unobservable everywhere it matters — `-0.0 == 0.0` is true, and
/// `norm_sqr` maps both to `+0.0`, so probabilities, counts and expectation
/// values are untouched — but it IS visible in raw statevector output, where
/// a JSON consumer would see `-0.0`. Recorded rather than glossed, because
/// "bit-identical" was the claim this file made until the oracle below
/// disproved it.
///
/// Every other backend here (Metal, CUDA, OpenCL) and qiskit-aer specialise
/// the diagonal case; the CPU was the only one that did not.
pub(crate) fn apply_diagonal_1q(
    state: &mut [Complex64],
    n: usize,
    q: usize,
    a: Complex64,
    d: Complex64,
) {
    use rayon::prelude::*;
    let dim = 1usize << n;
    let mask = 1usize << q;
    if dim < PAR_MIN_DIM {
        for (i, amp) in state.iter_mut().enumerate() {
            *amp *= if i & mask != 0 { d } else { a };
        }
        return;
    }
    state.par_iter_mut().enumerate().for_each(|(i, amp)| {
        *amp *= if i & mask != 0 { d } else { a };
    });
}

/// Route a 1q gate matrix through the diagonal kernel.
///
/// Takes the matrix rather than the two diagonal entries so the values stay
/// defined in exactly one place (`gates.rs`); a second copy here is how the
/// dense and specialised paths drift apart. Debug builds assert the
/// off-diagonals really are zero, so mis-routing a non-diagonal gate is caught
/// at the call site rather than becoming a silently wrong answer.
#[inline]
fn diag_from(state: &mut [Complex64], n: usize, q: usize, g: &gates::Gate1Q) {
    debug_assert!(
        g[1] == Complex64::new(0.0, 0.0) && g[2] == Complex64::new(0.0, 0.0),
        "diag_from called with a non-diagonal gate: off-diagonals {:?}, {:?}",
        g[1],
        g[2]
    );
    apply_diagonal_1q(state, n, q, g[0], g[3]);
}

/// Apply Toffoli (CCX) gate: flip target if both controls are |1>.
pub(crate) fn apply_ccx(state: &mut [Complex64], n: usize, c0: usize, c1: usize, target: usize) {
    use rayon::prelude::*;
    // Aliased operands are malformed — a gate cannot act on one qubit twice —
    // and this loop handles them SILENTLY rather than loudly, which is the
    // failure mode `apply_2q` was given an assert for.
    //
    // `target == c0` (or `c1`): the guard needs the target bit clear and the
    // control bit set, and those are the same bit, so no index ever matches and
    // the gate becomes a no-op. A dropped operation, reported as success.
    //
    // `c0 == c1` is the one aliased case the loop gets RIGHT — it tests the
    // same mask twice and degenerates to an exact `CX(c0, target)`. It is still
    // refused, because accepting it would mean the caller can construct a
    // three-qubit gate on two qubits and get a two-qubit one back, which is not
    // a contract worth having.
    assert!(
        c0 != c1 && c0 != target && c1 != target,
        "apply_ccx: operands must be distinct (c0={c0}, c1={c1}, target={target}) \
         — a controlled gate cannot control itself or target its own control"
    );
    let dim = 1usize << n;
    let mask_c0 = 1usize << c0;
    let mask_c1 = 1usize << c1;
    let mask_t = 1usize << target;

    // The scan this replaces tested all `dim` indices and acted on one in
    // eight — the other seven iterations were a load, three masks and a
    // branch, discarded. Instead, enumerate the subspace directly: with the
    // three operand bits removed there are `dim/8` free configurations, and
    // each one names exactly one amplitude pair to swap. `scatter` reinserts
    // the fixed bits (both controls set, target clear) into a compacted index.
    //
    // Same total work, one eighth of the iterations, no branch in the body.
    let (b0, b1, b2) = {
        let mut v = [c0, c1, target];
        v.sort_unstable();
        (v[0], v[1], v[2])
    };
    // Removing three bit positions from an index leaves the remaining bits in
    // FOUR runs, not three: below b0, between b0 and b1, between b1 and b2, and
    // above b2. Writing three is the natural slip and it is wrong for any
    // triple that is not adjacent — caught by the oracle below on
    // `n=4, ccx(0,1,2)` before this was corrected.
    let scatter = |k: usize| -> usize {
        let low = k & ((1usize << b0) - 1);
        let rest = k >> b0;
        let mid = rest & ((1usize << (b1 - b0 - 1)) - 1);
        let rest = rest >> (b1 - b0 - 1);
        let hi = rest & ((1usize << (b2 - b1 - 1)) - 1);
        let top = rest >> (b2 - b1 - 1);
        low | (mid << (b0 + 1)) | (hi << (b1 + 1)) | (top << (b2 + 1))
    };
    let groups = dim >> 3;
    if dim < PAR_MIN_DIM {
        for k in 0..groups {
            let base = scatter(k) | mask_c0 | mask_c1;
            debug_assert_eq!(base & mask_t, 0);
            state.swap(base, base | mask_t);
        }
        return;
    }
    // S2 reached `apply_1q`, `apply_2q`, `apply_cx` and `apply_diagonal_1q` and
    // stopped here, leaving the two WIDEST gates as the only serial kernels in
    // the file. That is the wrong place to stop: `ccx`/`cswap` dominate exactly
    // the Toffoli-heavy circuits (`grover_generic_28q`) that the incoming CR
    // reported as timing out.
    //
    // Bit-identity is FREE for these two, in a way it is not for `apply_1q`.
    // They are PERMUTATIONS: each group swaps a disjoint pair and performs no
    // arithmetic at all, so there is no floating-point association to change
    // and no ordering that could alter a result. The invariance contract in
    // PLAN-SV-PERF.md §2 is satisfied by construction here rather than by
    // argument.
    let ptr = state.as_mut_ptr() as usize;
    (0..groups).into_par_iter().for_each(|k| {
        let base = scatter(k) | mask_c0 | mask_c1;
        debug_assert_eq!(base & mask_t, 0);
        // SAFETY: `scatter` is injective over `0..dim/8` and reinserts the three
        // operand bits, so every `k` yields a distinct `base` with `mask_t`
        // clear. The pair `(base, base | mask_t)` is therefore disjoint from
        // every other iteration's pair, and both indices are `< dim`. Same
        // argument and same shape as `apply_cx` above.
        unsafe {
            let b = ptr as *mut Complex64;
            std::ptr::swap(b.add(base), b.add(base | mask_t));
        }
    });
}

/// Apply Fredkin (CSWAP) gate: swap targets if control is |1>.
pub(crate) fn apply_cswap(state: &mut [Complex64], n: usize, control: usize, t0: usize, t1: usize) {
    use rayon::prelude::*;
    // As in `apply_ccx`, but the failure here is worse than a dropped gate.
    // With `control == t0` the partner index `j = (i & !mask_t0) | mask_t1`
    // clears the CONTROL bit, so the swap runs between a control-1 amplitude
    // and a control-0 one: not a no-op, a wrong state, silently.
    //
    // `t0 == t1` is the benign aliased case (the guard can never be satisfied,
    // so it is an exact identity — which is what swapping a qubit with itself
    // should do), and it is still refused for the same reason as `c0 == c1`
    // above.
    assert!(
        control != t0 && control != t1 && t0 != t1,
        "apply_cswap: operands must be distinct (control={control}, t0={t0}, \
         t1={t1}) — aliasing the control with a target swaps across control \
         values and silently corrupts the state"
    );
    let dim = 1usize << n;
    let mask_c = 1usize << control;
    let mask_t0 = 1usize << t0;
    let mask_t1 = 1usize << t1;

    // Same subspace walk as `apply_ccx`: the scan tested all `dim` indices to
    // act on one in eight. Here the fixed pattern is control=1, t0=1, t1=0, and
    // the partner clears t0 and sets t1.
    let (b0, b1, b2) = {
        let mut v = [control, t0, t1];
        v.sort_unstable();
        (v[0], v[1], v[2])
    };
    let scatter = |k: usize| -> usize {
        let low = k & ((1usize << b0) - 1);
        let rest = k >> b0;
        let mid = rest & ((1usize << (b1 - b0 - 1)) - 1);
        let rest = rest >> (b1 - b0 - 1);
        let hi = rest & ((1usize << (b2 - b1 - 1)) - 1);
        let top = rest >> (b2 - b1 - 1);
        low | (mid << (b0 + 1)) | (hi << (b1 + 1)) | (top << (b2 + 1))
    };
    let groups = dim >> 3;
    if dim < PAR_MIN_DIM {
        for k in 0..groups {
            let free = scatter(k);
            let i = free | mask_c | mask_t0;
            state.swap(i, (i & !mask_t0) | mask_t1);
        }
        return;
    }
    let ptr = state.as_mut_ptr() as usize;
    (0..groups).into_par_iter().for_each(|k| {
        let free = scatter(k);
        let i = free | mask_c | mask_t0;
        let j = (i & !mask_t0) | mask_t1;
        // SAFETY: as `apply_ccx`. `scatter` is injective over `0..dim/8` and the
        // three operand bits are reinserted with a FIXED pattern (control set,
        // t0 set, t1 clear), so each `k` names one pair `(i, j)` disjoint from
        // every other. `i != j` because they differ in both `t0` and `t1`.
        unsafe {
            let b = ptr as *mut Complex64;
            std::ptr::swap(b.add(i), b.add(j));
        }
    });
}

/// Sample measurement outcomes from the statevector.
fn sample_counts(
    state: &[Complex64],
    num_qubits: usize,
    shots: u32,
    seed: Option<u64>,
) -> HashMap<u64, u32> {
    // This used to materialise TWO `2^n` f64 vectors — `probs` and its running
    // sum — on top of the live state. At 28 qubits that is 2 x 2.1 GB beside a
    // 4.3 GB state, i.e. sampling **doubled** peak memory and was the reason a
    // sampled run hit the capacity guard a full qubit earlier than an analytic
    // one.
    //
    // Neither vector is necessary. Inverse-CDF sampling needs the cumulative
    // distribution only in increasing order, and the draws can be put in that
    // order instead of the distribution: sort the `shots` uniforms once, then
    // walk the state a single time accumulating the running sum and emitting
    // outcomes as the accumulator passes each draw. Extra memory is `shots`
    // f64s — kilobytes — rather than `2^n`.
    //
    // **The counts are unchanged for a given seed.** The same RNG produces the
    // same multiset of uniforms, and inverting the CDF is a per-draw function,
    // so processing them in sorted order permutes the work and not the result.
    // The comparison is `<=` to match `partition_point(|&c| c < r)` exactly,
    // which selects the first index whose cumulative value is >= r.
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => rand::make_rng::<StdRng>(),
    };
    let _ = num_qubits; // used by caller for formatting

    let mut draws: Vec<f64> = (0..shots).map(|_| rng.random()).collect();
    draws.sort_by(|a, b| a.partial_cmp(b).expect("uniform draws are never NaN"));

    let mut counts: HashMap<u64, u32> = HashMap::new();
    let last = state.len() - 1;
    let mut acc = 0.0_f64;
    let mut d = 0usize;
    for (idx, amp) in state.iter().enumerate() {
        if d == draws.len() {
            break;
        }
        acc += amp.norm_sqr();
        while d < draws.len() && draws[d] <= acc {
            *counts.entry(idx as u64).or_insert(0) += 1;
            d += 1;
        }
    }
    // Any residue is the floating-point shortfall of `sum |a|^2` against 1.0 —
    // the old code papered over the same gap by pinning the final cumulative
    // entry to exactly 1.0.
    while d < draws.len() {
        *counts.entry(last as u64).or_insert(0) += 1;
        d += 1;
    }

    counts
}

/// Compute <psi|P|psi> for a Pauli string P.
///
/// The single-qubit closed forms used per-qubit below
/// (Z: ±1 by bit, X: bit flip, Y: bit flip + i/-i) are proved
/// against the matrix definition in
/// `verification/Verification/Adjoint/PauliExpectation.lean` (a TARGET, not
/// yet written — see `verification/README.md`) —
/// theorems `expV_Z` / `expV_X` / `expV_Y` / `expV_I`. Each
/// Pauli is also Hermitian
/// (`σ?_hermitian`), so the resulting expectation is real (we
/// take `.re` on the final accumulator).
fn expectation_pauli(sv: &[Complex64], num_qubits: u32, pauli_string: &[(u32, PauliOp)]) -> f64 {
    let n = num_qubits as usize;
    let dim = 1usize << n;
    let mut result = Complex64::new(0.0, 0.0);

    for i in 0..dim {
        // Apply the Pauli string to |i> -> coeff * |j>
        let mut j = i;
        let mut coeff = Complex64::new(1.0, 0.0);

        for (qubit, pauli) in pauli_string {
            let q = *qubit as usize;
            let bit = (i >> q) & 1;
            match pauli {
                PauliOp::I => {}
                PauliOp::Z => {
                    if bit == 1 {
                        coeff *= -1.0;
                    }
                }
                PauliOp::X => {
                    j ^= 1 << q; // flip bit
                }
                PauliOp::Y => {
                    j ^= 1 << q; // flip bit
                    if bit == 0 {
                        coeff *= Complex64::new(0.0, 1.0);
                    } else {
                        coeff *= Complex64::new(0.0, -1.0);
                    }
                }
            }
        }

        // P|i> = coeff * |j>, so the contribution to <psi|P|psi> is
        // conj(sv[j]) * coeff * sv[i]. (Pairing the coefficient with
        // conj(sv[i])·sv[j] instead silently NEGATES every Pauli string
        // with an odd number of Y factors — Y's two matrix elements
        // have opposite signs, so the i-based coefficient belongs to
        // the bra side only after conjugation. Caught by the
        // Clifford-suffix parallel-shift tests; pinned in
        // tests/pauli_y_expectation.rs.)
        result += sv[j].conj() * coeff * sv[i];
    }

    result.re
}

#[cfg(test)]
mod sampler_equivalence {
    //! **The sorted-draw sampler returns the same counts as the two-vector one.**
    //!
    //! `sample_counts` used to materialise `probs` and `cumulative`, each `2^n`
    //! f64, and do one `partition_point` per shot. It now sorts the `shots`
    //! draws and walks the state once — at 28 qubits, kilobytes instead of
    //! 4.3 GB of auxiliary allocation (`PLAN-SV-PERF.md` §4).
    //!
    //! That rewrite landed WITHOUT the test the plan named as its gate ("same
    //! seed => same counts"). The argument for it is sound — the same RNG yields
    //! the same multiset of uniforms, and inverting a CDF is a per-draw function,
    //! so sorting permutes the work and not the result — but it is an argument,
    //! and the rewrite turns on two details an argument glides over: the
    //! comparison must be `<=` to match `partition_point(|&c| c < r)`, and the
    //! floating-point shortfall of `sum |a|^2` against 1.0 must land on the last
    //! index rather than being dropped.
    //!
    //! So the pre-rewrite sampler is kept here verbatim, as `group_walk_
    //! equivalence` keeps the pre-rewrite gate loops, and compared against.
    use super::*;

    /// The two-vector sampler, verbatim. Do not "clean up": its value is being
    /// the thing that shipped.
    fn sample_counts_two_vector(
        state: &[Complex64],
        shots: u32,
        seed: Option<u64>,
    ) -> HashMap<u64, u32> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => rand::make_rng::<StdRng>(),
        };
        let probs: Vec<f64> = state.iter().map(|a| a.norm_sqr()).collect();
        let mut cumulative = Vec::with_capacity(probs.len());
        let mut acc = 0.0f64;
        for p in &probs {
            acc += p;
            cumulative.push(acc);
        }
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }
        let mut counts: HashMap<u64, u32> = HashMap::new();
        for _ in 0..shots {
            let r: f64 = rng.random();
            let idx = cumulative.partition_point(|&c| c < r);
            let idx = idx.min(cumulative.len() - 1);
            *counts.entry(idx as u64).or_insert(0) += 1;
        }
        counts
    }

    /// A state with structure — not uniform, and with some exactly-zero
    /// amplitudes, because a zero-probability index is precisely where an
    /// off-by-one in the `<`/`<=` boundary shows up.
    fn structured_state(n: usize) -> Vec<Complex64> {
        let dim = 1usize << n;
        let mut v: Vec<Complex64> = (0..dim)
            .map(|i| {
                if i % 5 == 0 {
                    Complex64::new(0.0, 0.0) // exact zeros
                } else {
                    Complex64::new(((i % 7) as f64 + 1.0) / 13.0, ((i % 3) as f64) / 11.0)
                }
            })
            .collect();
        let norm: f64 = v.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for a in v.iter_mut() {
            *a /= norm;
        }
        v
    }

    /// A state whose probabilities sum to LESS than 1.
    ///
    /// This is what reaches the residue loop, and nothing else does. Both
    /// samplers handle the shortfall by pinning it to the last index — the old
    /// one by forcing `cumulative.last() = 1.0`, the new one by draining
    /// leftover draws onto `last`.
    ///
    /// Needed because a normalised fixture CANNOT reach that code: `acc`
    /// climbs to ~1.0 and every draw from `[0, 1)` is consumed before the walk
    /// ends. Measured, not guessed — deleting the residue loop left the whole
    /// equivalence test green until this fixture was added.
    ///
    /// The shortfall is large deliberately. A realistic f64 shortfall is ~1e-16
    /// and would be hit by roughly one draw in 1e16; 1e-3 makes the path
    /// certain to run while testing the same branch.
    fn shortfall_state(n: usize) -> Vec<Complex64> {
        let mut v = structured_state(n);
        let scale = (1.0f64 - 1e-3).sqrt();
        for a in v.iter_mut() {
            *a *= scale;
        }
        v
    }

    #[test]
    fn the_residue_of_an_unnormalised_state_lands_on_the_last_index() {
        for n in [1usize, 3, 6] {
            let state = shortfall_state(n);
            let shots = 5000u32;
            for seed in [0u64, 7, 99] {
                let want = sample_counts_two_vector(&state, shots, Some(seed));
                let got = sample_counts(&state, n, shots, Some(seed));
                assert_eq!(
                    want, got,
                    "n={n} seed={seed}: samplers disagree on a state whose \
                     probabilities sum to 1 - 1e-3"
                );
                let total: u32 = got.values().sum();
                assert_eq!(
                    total, shots,
                    "n={n} seed={seed}: {total} of {shots} shots survived — the \
                     residue past the end of the walk was dropped"
                );
            }
        }
    }

    /// **NOT tested, and why**: the `<=` vs `<` boundary in the inner loop.
    ///
    /// The plan calls the comparison out as load-bearing — it must be `<=` to
    /// match `partition_point(|&c| c < r)`. Changing it to `<` leaves every test
    /// here green, and that is not a gap in the fixtures: the two differ only
    /// when a draw equals a cumulative sum EXACTLY. With f64 draws at 2^-53
    /// granularity that has probability ~7e-12 across this whole module, so it
    /// is unobservable by construction rather than untested by omission.
    ///
    /// Recorded instead of covered by a contrived fixture. A test that forced
    /// the coincidence would be testing an input the RNG cannot produce, and
    /// claiming coverage of a boundary that has no behavioural consequence is
    /// worse than saying it is unreachable.
    ///
    /// If the sampler ever takes caller-supplied draws, this stops being
    /// measure-zero and needs a real test.
    #[test]
    fn the_sorted_draw_sampler_matches_the_two_vector_one() {
        let mut checked = 0;
        for n in [1usize, 2, 5, 8] {
            let state = structured_state(n);
            for shots in [1u32, 2, 17, 1000] {
                for seed in [0u64, 1, 42, 0xDEAD_BEEF] {
                    let want = sample_counts_two_vector(&state, shots, Some(seed));
                    let got = sample_counts(&state, n, shots, Some(seed));
                    assert_eq!(
                        want, got,
                        "n={n} shots={shots} seed={seed}: the sorted-draw sampler \
                         disagrees with the two-vector one it replaced"
                    );
                    let total: u32 = got.values().sum();
                    assert_eq!(total, shots, "n={n} shots={shots}: lost or invented shots");
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 64, "expected 64 comparisons, ran {checked}");
    }

    /// A zero-probability index must NEVER be sampled.
    ///
    /// The `<=` boundary is where this can break: if the accumulator equals a
    /// draw exactly at an index whose own probability is zero, a `<` / `<=` slip
    /// emits an outcome the state says is impossible. Asserted directly rather
    /// than left to the equivalence test, which would only report "the two
    /// agree" if BOTH had the slip.
    #[test]
    fn an_impossible_outcome_is_never_sampled() {
        // |psi> = (|1> + |3>)/sqrt(2) on 2 qubits: indices 0 and 2 are exactly 0.
        let r = 1.0 / 2.0_f64.sqrt();
        let state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(r, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(r, 0.0),
        ];
        for seed in 0..32u64 {
            let counts = sample_counts(&state, 2, 500, Some(seed));
            for (idx, n) in &counts {
                assert!(
                    *idx == 1 || *idx == 3,
                    "seed={seed}: sampled impossible outcome {idx} ({n} times); \
                     amplitudes 0 and 2 are exactly zero"
                );
            }
        }
    }
}

#[cfg(test)]
mod group_walk_equivalence {
    //! `apply_2q` used to scan all `dim` indices and reject three of every
    //! four (`PLAN-SV-PERF.md` §1). The replacement walks the `dim/4` groups
    //! directly. That is a pure indexing change, so it must be **bit-for-bit**
    //! identical, not merely close — and the only way to say that with a
    //! straight face is to keep the loop it replaced and compare against it.
    use super::*;
    use num_complex::Complex64;

    /// The pre-2026-08-15 loop, verbatim. Do not "clean up": its value is
    /// being the thing that shipped.
    fn apply_2q_scan(
        state: &mut [Complex64],
        n: usize,
        q0: usize,
        q1: usize,
        gate: &gates::Gate2Q,
    ) {
        let dim = 1usize << n;
        let (qa, qb, g) = if q0 > q1 {
            (q0, q1, *gate)
        } else {
            let mut swapped = *gate;
            for col in 0..4 {
                swapped.swap(4 + col, 2 * 4 + col);
            }
            for row in 0..4 {
                swapped.swap(row * 4 + 1, row * 4 + 2);
            }
            (q1, q0, swapped)
        };
        let step_a = 1usize << qa;
        let step_b = 1usize << qb;
        for i in 0..dim {
            if (i & step_a) != 0 || (i & step_b) != 0 {
                continue;
            }
            let (i00, i01, i10, i11) = (i, i | step_b, i | step_a, i | step_a | step_b);
            let (a00, a01, a10, a11) = (state[i00], state[i01], state[i10], state[i11]);
            state[i00] = g[0] * a00 + g[1] * a01 + g[2] * a10 + g[3] * a11;
            state[i01] = g[4] * a00 + g[5] * a01 + g[6] * a10 + g[7] * a11;
            state[i10] = g[8] * a00 + g[9] * a01 + g[10] * a10 + g[11] * a11;
            state[i11] = g[12] * a00 + g[13] * a01 + g[14] * a10 + g[15] * a11;
        }
    }

    /// Full-mantissa amplitudes, deterministically. Constant or
    /// `1/√2`-valued fixtures cannot distinguish two orderings — every
    /// association gives the same bits — so this is not decoration.
    fn dense_state(n: usize, salt: u64) -> Vec<Complex64> {
        let mut x = salt.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // Full 53-bit mantissa in (-1, 1), never a round number.
            (x as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        (0..1usize << n)
            .map(|_| Complex64::new(next(), next()))
            .collect()
    }

    /// A gate with no zero and no repeated entry: a zero would let a
    /// mis-slotted operand vanish, and a repeat would let two of them be
    /// swapped unnoticed.
    fn dense_gate(salt: u64) -> gates::Gate2Q {
        let s = dense_state(2, salt); // 4 complex values
        let t = dense_state(2, salt ^ 0x5eed);
        let u = dense_state(2, salt ^ 0xbeef);
        let v = dense_state(2, salt ^ 0xf00d);
        [
            s[0], s[1], s[2], s[3], t[0], t[1], t[2], t[3], u[0], u[1], u[2], u[3], v[0], v[1],
            v[2], v[3],
        ]
    }

    /// **EVERY 2q gate obeys the dispatch contract — not just the ones with a
    /// kernel today.**
    ///
    /// This replaced three near-duplicate tests, one per kernel, and the
    /// duplication was not the problem with them. The problem was that a
    /// caveat asserted only where it was written cannot tell you it is FALSE
    /// of a sibling: the `+ 0.0` "fix" survived fault injection in the
    /// controlled-U kernel and was believed, and it was only writing the same
    /// claim about the middle-block twin — where injection did NOT fail — that
    /// exposed it. Fault injection proves a check is load-bearing. It cannot
    /// prove the check is COMPLETE.
    ///
    /// So the contract is asserted mechanically over the whole gate table. A
    /// gate added to `gates.rs` and routed through `apply_2q` is covered here
    /// whether or not anyone remembers to write a test for its dispatch, and a
    /// new kernel that breaks the contract for one gate fails on that gate by
    /// name.
    #[test]
    fn every_2q_gate_differs_from_the_dense_scan_only_in_the_sign_of_zero() {
        let table: Vec<(&str, gates::Gate2Q)> = vec![
            ("cx", gates::cx()),
            ("cy", gates::cy()),
            ("cz", gates::cz()),
            ("swap", gates::swap()),
            ("crz", gates::crz(0.37)),
            ("crz_pi", gates::crz(std::f64::consts::PI)),
            ("cu3", gates::cu3(0.3, 0.5, 0.7)),
            ("rbs", gates::rbs(0.41)),
            // Derivative matrices: same dispatch, and several are sparse in
            // shapes the forward gates are not.
            ("dcrz", gates::dcrz(0.37)),
            ("drbs", gates::drbs(0.41)),
            ("dcu3_dt", gates::dcu3_dt(0.3, 0.5, 0.7)),
            ("dcu3_dp", gates::dcu3_dp(0.3, 0.5, 0.7)),
            ("dcu3_dl", gates::dcu3_dl(0.3, 0.5, 0.7)),
            // A gate no detector can claim, so the dense path itself is
            // covered: it must be bit-identical to itself.
            ("dense", dense_gate(0x5150)),
        ];

        for n in 3..=5 {
            for (q0, q1) in [(2usize, 0usize), (0usize, 2usize), (1, 0), (0, 1)] {
                if q0 >= n || q1 >= n {
                    continue;
                }
                for (label, gate) in &table {
                    for (what, init) in zero_fixtures(n) {
                        let mut fast = init.clone();
                        let mut oracle = init.clone();
                        apply_2q(&mut fast, n, q0, q1, gate);
                        apply_2q_scan(&mut oracle, n, q0, q1, gate);
                        assert_matches_dense_up_to_signed_zero(
                            &fast,
                            &oracle,
                            &format!("{label} {what} n={n} q0={q0} q1={q1}"),
                        );
                    }
                }
            }
        }
    }

    /// **Even the PERMUTATION dispatches differ in the sign of zero**, and
    /// this test exists because that is counter-intuitive enough that two
    /// separate revisions of this file exempted them in prose.
    ///
    /// The reasoning that fails: "a permutation moves amplitudes and computes
    /// nothing, so there is no product to carry a sign." True, and irrelevant.
    /// The difference does not come from the arithmetic in the region the
    /// kernel touches — it comes from the region it does NOT. `SWAP` fixes
    /// |00> and |11>; `CX` fixes the whole control-zero half. The dense path
    /// WRITES those (`1*a00 + 0*a01 + 0*a10 + 0*a11`) and the write normalises
    /// `-0.0` to `+0.0`. Skipping the write preserves the sign.
    ///
    /// So the rule is about SKIPPED WRITES, not about multiplication, and it
    /// admits no exceptions among the sparse dispatches. Asserted rather than
    /// documented, on the principle that a caveat is only as good as the
    /// sibling it is also checked against.
    #[test]
    fn even_the_permutation_dispatches_differ_in_the_sign_of_zero() {
        for (label, gate) in [("swap", gates::swap()), ("cx", gates::cx())] {
            let n = 3;
            let init = vec![Complex64::new(-0.0, -0.0); 1usize << n];
            let mut fast = init.clone();
            let mut oracle = init.clone();
            apply_2q(&mut fast, n, 2, 0, &gate);
            apply_2q_scan(&mut oracle, n, 2, 0, &gate);
            let differing = fast
                .iter()
                .zip(oracle.iter())
                .filter(|(a, b)| {
                    a.re.to_bits() != b.re.to_bits() || a.im.to_bits() != b.im.to_bits()
                })
                .count();
            assert!(
                differing > 0,
                "{label}: expected the untouched fixed points to keep their -0.0 \
                 where the dense path normalises it. If this stops being true the \
                 exemption really can be granted — check why before granting it."
            );
            // ...and the difference is ONLY that.
            assert_matches_dense_up_to_signed_zero(&fast, &oracle, label);
        }
    }

    /// Fixtures that put signed zeros in every position, because that is the
    /// ONLY place a sparse dispatch can diverge from the dense path. A
    /// full-mantissa fixture cannot see it — which is why the first version of
    /// the diagonal test claimed bit-identity and was believed.
    fn zero_fixtures(n: usize) -> Vec<(&'static str, Vec<Complex64>)> {
        let dim = 1usize << n;
        vec![
            ("random", dense_state(n, 0xb10 ^ (n as u64))),
            ("all-neg-zero", vec![Complex64::new(-0.0, -0.0); dim]),
            (
                "alt-signed-zero",
                (0..dim)
                    .map(|i| {
                        let z = if i % 2 == 0 { -0.0f64 } else { 0.0f64 };
                        Complex64::new(z, -z)
                    })
                    .collect(),
            ),
            // Zeros next to NEGATIVE finite values: the dense path's `0.0 * a`
            // takes its sign from `a`, so a negative neighbour is what makes a
            // skipped read observable.
            (
                "neg-neighbours",
                (0..dim)
                    .map(|i| {
                        if i % 2 == 0 {
                            Complex64::new(-0.0, 0.0)
                        } else {
                            Complex64::new(-1.0 - i as f64, -2.0 - i as f64)
                        }
                    })
                    .collect(),
            ),
        ]
    }

    /// The contract from [`diagonal_2q`]'s signed-zero note: numerically equal
    /// everywhere, and every bit difference is exactly a signed zero.
    fn assert_matches_dense_up_to_signed_zero(fast: &[Complex64], oracle: &[Complex64], ctx: &str) {
        let mut zero_diffs = 0usize;
        for (i, (a, b)) in fast.iter().zip(oracle.iter()).enumerate() {
            assert_eq!((a.re, a.im), (b.re, b.im), "{ctx} idx={i}: VALUES differ");
            for (x, y) in [(a.re, b.re), (a.im, b.im)] {
                if x.to_bits() != y.to_bits() {
                    assert!(
                        x == 0.0 && y == 0.0,
                        "{ctx} idx={i}: differs by more than a signed zero: {x} vs {y}"
                    );
                    zero_diffs += 1;
                }
            }
        }
        let _ = zero_diffs;
    }

    /// The diagonal fast path is BIT-IDENTICAL to the dense scan.
    ///
    /// Not "close": the dense path multiplies by exactly-zero off-diagonals
    /// and adds the results, and `0.0 * x == 0.0` and `y + 0.0 == y` for
    /// finite `y`, so skipping them cannot change a single bit. Asserting
    /// equality rather than a tolerance is what makes this test able to fail —
    /// a tolerance would pass on a fast path that had quietly changed the
    /// association order, which is the thing worth catching.
    ///
    /// Both qubit orders, because the diagonal is detected AFTER the q0/q1
    /// relabel: `q0 < q1` swaps entries 1 and 2, and a fast path that read the
    /// diagonal off the UNrelabelled matrix would be wrong on exactly half the
    /// call sites and right on the other half.
    #[test]
    fn the_diagonal_2q_path_differs_only_in_the_sign_of_zero() {
        for n in 3..=6 {
            for (label, gate) in [
                ("cz", gates::cz()),
                ("crz", gates::crz(0.37)),
                ("crz_pi", gates::crz(std::f64::consts::PI)),
            ] {
                // A diagonal with four DISTINCT entries: cz has repeats, so on
                // its own a slot mix-up could hide.
                for (q0, q1) in [(2usize, 0usize), (0usize, 2usize), (1, 0), (0, 1)] {
                    if q0 >= n || q1 >= n {
                        continue;
                    }
                    for (what, init) in zero_fixtures(n) {
                        let mut fast = init.clone();
                        let mut oracle = init.clone();
                        apply_2q(&mut fast, n, q0, q1, &gate);
                        apply_2q_scan(&mut oracle, n, q0, q1, &gate);
                        assert_matches_dense_up_to_signed_zero(
                            &fast,
                            &oracle,
                            &format!("{label} {what} n={n} q0={q0} q1={q1}"),
                        );
                    }
                }
            }
        }
    }

    /// The controlled-U path vs the dense scan — and the ONE way they differ.
    ///
    /// On the acted-on (control = 1) half this is bit-identical, and the
    /// leading `ZERO_C +` in the kernel is what makes it so: the dense path
    /// computes `((0*a00 + 0*a01) + m0*a10) + m1*a11`, and without that add a
    /// product landing on `-0.0` would come out `-0.0` here and `+0.0` there.
    ///
    /// On the control = 0 half they differ, and CANNOT be made to agree
    /// without giving up the optimisation. The dense path WRITES that half —
    /// `1*a00 + 0*a01 + 0*a10 + 0*a11` — and that write normalises `-0.0` to
    /// `+0.0`. This path does not read it at all, which is the entire point:
    /// half the state untouched. So a `-0.0` amplitude sitting in the
    /// control-zero subspace SURVIVES here and was silently flipped to `+0.0`
    /// before.
    ///
    /// That difference is unobservable in every way this project measures:
    /// `-0.0 == +0.0`, |z|^2 is identical, sampling is identical. It is
    /// recorded because a future bit-equality harness comparing this backend
    /// against its own history would see it, and "bit-identical except
    /// sometimes" is the kind of hedge that quietly voids a guarantee.
    #[test]
    fn the_controlled_u_path_differs_only_in_the_sign_of_zero() {
        assert!(controlled_1q(&gates::cy()).is_some());
        assert!(controlled_1q(&gates::cu3(0.3, 0.5, 0.7)).is_some());
        assert!(
            controlled_1q(&gates::swap()).is_none(),
            "swap is not controlled"
        );
        assert!(controlled_1q(&dense_gate(0x77)).is_none());

        // A gate whose TOP-LEFT is the identity but whose lower rows reach back
        // into the control-zero columns is NOT controlled: it moves amplitude
        // out of the untouched half, which this kernel would silently drop
        // because it never reads that half.
        //
        // None of the fixtures above exercises this clause — `swap` and a
        // random gate both fail the identity-block test first, and `cy`/`cu3`
        // have no leak — so deleting the `no_leak` check used to leave the
        // whole test GREEN. Found by injecting `no_leak = true`; the two cases
        // below are what make that injection fail.
        for leak_at in [8usize, 9, 12, 13] {
            let mut leaky = gates::cy();
            leaky[leak_at] = Complex64::new(0.5, 0.0);
            assert!(
                controlled_1q(&leaky).is_none(),
                "a gate leaking into the control-zero columns at g[{leak_at}] is not controlled"
            );
        }

        for n in 3..=6 {
            for (q0, q1) in [(2usize, 0usize), (0usize, 2usize), (1, 0), (0, 1)] {
                if q0 >= n || q1 >= n {
                    continue;
                }
                for (what, init) in zero_fixtures(n) {
                    for (label, gate) in [("cy", gates::cy()), ("cu3", gates::cu3(0.3, 0.5, 0.7))] {
                        let mut fast = init.clone();
                        let mut oracle = init.clone();
                        apply_2q(&mut fast, n, q0, q1, &gate);
                        apply_2q_scan(&mut oracle, n, q0, q1, &gate);
                        assert_matches_dense_up_to_signed_zero(
                            &fast,
                            &oracle,
                            &format!("{label} {what} n={n} q0={q0} q1={q1}"),
                        );
                    }
                }
            }
        }
    }

    /// `Rbs` and the {|01>, |10>} subspace path, against the dense scan.
    ///
    /// Contract is [`diagonal_2q`]'s signed-zero note: numerically equal
    /// everywhere, and every bit difference is exactly the sign of a zero.
    ///
    /// An earlier revision claimed the acted-on pair was bit-identical and
    /// carried leading/trailing `+ 0.0` to make it so. Both were false: the
    /// dense row is `((0*a00 + m0*a01) + m1*a10) + 0*a11`, and `0.0 * a00`
    /// is `-0.0` when `a00` is negative — so the dense result depends on the
    /// sign of a corner this kernel deliberately never reads. Adding a
    /// constant `+0.0` cannot reproduce a sign it does not know.
    #[test]
    fn the_middle_block_path_differs_only_in_the_sign_of_zero() {
        assert!(middle_block_2q(&gates::rbs(0.41)).is_some());
        // SWAP also has this FORM, and must be caught by the cheaper
        // permutation dispatch first — asserted here so a reordering of the
        // dispatch chain shows up as a failure rather than a slowdown.
        assert!(
            middle_block_2q(&gates::swap()).is_some(),
            "swap has the form"
        );
        assert!(
            is_swap_2q(&gates::swap()),
            "...and must be taken by swap first"
        );
        assert!(middle_block_2q(&gates::cy()).is_none());
        assert!(middle_block_2q(&gates::cz()).is_none());
        assert!(middle_block_2q(&dense_gate(0x33)).is_none());
        // A corner that is not 1 means |00> or |11> is NOT a fixed point, so
        // the corners cannot be skipped.
        for corner in [0usize, 15] {
            let mut bad = gates::rbs(0.41);
            bad[corner] = Complex64::new(0.5, 0.0);
            assert!(
                middle_block_2q(&bad).is_none(),
                "g[{corner}] != 1 means the corner moves and cannot be skipped"
            );
        }
        // A leak out of the middle block moves amplitude into a corner this
        // kernel never writes.
        for leak in [1usize, 4, 7, 13] {
            let mut leaky = gates::rbs(0.41);
            leaky[leak] = Complex64::new(0.25, 0.0);
            assert!(
                middle_block_2q(&leaky).is_none(),
                "a leak at g[{leak}] escapes the {{01,10}} subspace"
            );
        }

        for n in 3..=6 {
            for (q0, q1) in [(2usize, 0usize), (0usize, 2usize), (1, 0), (0, 1)] {
                if q0 >= n || q1 >= n {
                    continue;
                }
                for (what, init) in zero_fixtures(n) {
                    let gate = gates::rbs(0.41);
                    let mut fast = init.clone();
                    let mut oracle = init.clone();
                    apply_2q(&mut fast, n, q0, q1, &gate);
                    apply_2q_scan(&mut oracle, n, q0, q1, &gate);
                    assert_matches_dense_up_to_signed_zero(
                        &fast,
                        &oracle,
                        &format!("rbs {what} n={n} q0={q0} q1={q1}"),
                    );
                }
            }
        }
    }

    /// The SWAP detector accepts only the exact SWAP matrix, and the path
    /// obeys the dispatch contract.
    ///
    /// This was named `swap_fast_path_is_bit_identical_and_detected_exactly`
    /// and asserted `to_bits()` equality over a full-mantissa fixture. It
    /// passed, and the bit-identity half was FALSE — SWAP does not write the
    /// |00>/|11> corners, the dense path does, and that write normalises
    /// `-0.0`. The fixture had no zeros, so the assertion could never see it.
    /// See [`diagonal_2q`]'s signed-zero note and
    /// `even_the_permutation_dispatches_differ_in_the_sign_of_zero`.
    #[test]
    fn the_swap_detector_accepts_only_swap() {
        assert!(is_swap_2q(&gates::swap()));
        assert!(!is_swap_2q(&gates::cz()));
        assert!(!is_swap_2q(&gates::cy()));
        assert!(!is_swap_2q(&dense_gate(0xabc)));
        // iSWAP is SWAP with phases — a different gate, and running it as a
        // bare permutation would drop the i.
        let mut iswap = gates::swap();
        iswap[6] = Complex64::new(0.0, 1.0);
        iswap[9] = Complex64::new(0.0, 1.0);
        assert!(!is_swap_2q(&iswap), "iSWAP is not SWAP");

        for n in 3..=6 {
            for (q0, q1) in [(2usize, 0usize), (0usize, 2usize), (1, 0), (0, 1)] {
                if q0 >= n || q1 >= n {
                    continue;
                }
                for (what, init) in zero_fixtures(n) {
                    let mut fast = init.clone();
                    let mut oracle = init.clone();
                    apply_2q(&mut fast, n, q0, q1, &gates::swap());
                    apply_2q_scan(&mut oracle, n, q0, q1, &gates::swap());
                    assert_matches_dense_up_to_signed_zero(
                        &fast,
                        &oracle,
                        &format!("swap {what} n={n} q0={q0} q1={q1}"),
                    );
                }
            }
        }
    }

    /// `diagonal_2q` must REFUSE anything with a non-zero off-diagonal, or the
    /// fast path would silently drop a real coupling. A dense random gate and
    /// the sparse-but-not-diagonal ones are the cases that matter.
    #[test]
    fn diagonal_2q_detects_only_actual_diagonals() {
        assert!(diagonal_2q(&gates::cz()).is_some());
        assert!(diagonal_2q(&gates::crz(0.3)).is_some());
        assert!(
            diagonal_2q(&gates::swap()).is_none(),
            "swap is a permutation"
        );
        assert!(diagonal_2q(&gates::cy()).is_none(), "cy is anti-diagonal");
        assert!(diagonal_2q(&dense_gate(0x1234)).is_none());
        // Exact zero, not near-zero: a tiny coupling is still a coupling.
        let mut almost = gates::cz();
        almost[1] = Complex64::new(1e-300, 0.0);
        assert!(
            diagonal_2q(&almost).is_none(),
            "a 1e-300 off-diagonal is a coupling, not a rounding artefact"
        );
    }

    /// The retired full-space scans, kept verbatim as oracles.
    ///
    /// `apply_2q`'s rewrite was validated this way and the same standard
    /// applies here: the subspace walk computes an index by expanding a
    /// compacted counter around three removed bit positions, which is exactly
    /// the kind of arithmetic that is right for the orderings you happened to
    /// try and wrong for one you did not.
    fn apply_ccx_scan(state: &mut [Complex64], n: usize, c0: usize, c1: usize, target: usize) {
        let dim = 1usize << n;
        let (mask_c0, mask_c1, mask_t) = (1usize << c0, 1usize << c1, 1usize << target);
        for i in 0..dim {
            if (i & mask_c0) != 0 && (i & mask_c1) != 0 && (i & mask_t) == 0 {
                state.swap(i, i | mask_t);
            }
        }
    }

    fn apply_cswap_scan(state: &mut [Complex64], n: usize, ctrl: usize, t0: usize, t1: usize) {
        let dim = 1usize << n;
        let (mask_c, mask_t0, mask_t1) = (1usize << ctrl, 1usize << t0, 1usize << t1);
        for i in 0..dim {
            if (i & mask_c) != 0 && (i & mask_t0) != 0 && (i & mask_t1) == 0 {
                state.swap(i, (i & !mask_t0) | mask_t1);
            }
        }
    }

    fn seeded_state(n: usize) -> Vec<Complex64> {
        // Distinct, non-degenerate amplitudes: a permutation bug that moved the
        // wrong pair would be invisible on a uniform state.
        (0..(1usize << n))
            .map(|i| Complex64::new(1.0 + i as f64, 0.5 - i as f64 * 0.25))
            .collect()
    }

    /// **The same oracle, ABOVE the parallel threshold.**
    ///
    /// The two tests below sweep `n ∈ 3..=7`, so `dim ≤ 128` and `PAR_MIN_DIM`
    /// is `1 << 12`. Every one of their assertions therefore runs the SERIAL
    /// branch, and when `apply_ccx`/`apply_cswap` were parallelised they gained
    /// a code path no oracle had ever compared against anything.
    ///
    /// That is not a hypothetical. Parallelising them and then deliberately
    /// swapping the wrong partner in the PARALLEL branch only —
    /// `base | mask_c0` instead of `base | mask_t` — left the whole
    /// `thread_count_invariance` suite green:
    ///
    /// * bit-identity across `T ∈ {1,2,12}` passes, because every thread count
    ///   runs the same wrong code. Invariance is a consistency property, not a
    ///   correctness one, and cannot detect a deterministic error.
    /// * `the_serial_and_parallel_paths_agree` passes, because it asserts only
    ///   that the norm is 1 — and a permutation preserves norm exactly, whatever
    ///   it permutes.
    ///
    /// So this test exists to be the thing that fails. `n = 13` puts `dim` at
    /// 8192, comfortably over the threshold, and the triples are curated rather
    /// than exhaustive because 1716 orderings × 8192 amplitudes is a debug-build
    /// tax for no extra coverage: the risk is bit-position arithmetic, and
    /// non-adjacent, reversed and top-qubit triples cover it.
    #[test]
    fn the_parallel_branch_matches_the_scan_too() {
        const N: usize = 13; // dim = 8192 > PAR_MIN_DIM
                             // COMPILE-TIME, not runtime: if `PAR_MIN_DIM` is ever raised above
                             // `2^13`, this test would silently start exercising the serial branch
                             // and keep passing while proving nothing. A build failure is the right
                             // response to that, not a green run.
        const _: () = assert!(
            (1usize << N) > PAR_MIN_DIM,
            "N must put dim above PAR_MIN_DIM, or this test exercises the serial \
             branch and proves nothing"
        );
        // Non-adjacent, reversed, spanning the top bit, and both extremes.
        let triples = [
            (0usize, 1usize, 2usize),
            (0, 6, 12),
            (12, 6, 0),
            (12, 0, 6),
            (1, 12, 5),
            (11, 12, 0),
            (0, 12, 11),
            (4, 5, 6),
        ];
        let mut checked = 0;
        for (a, b, c) in triples {
            let base = seeded_state(N);

            let mut want = base.clone();
            apply_ccx_scan(&mut want, N, a, b, c);
            let mut got = base.clone();
            apply_ccx(&mut got, N, a, b, c);
            for (i, (x, y)) in want.iter().zip(got.iter()).enumerate() {
                assert!(
                    x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits(),
                    "PARALLEL ccx({a},{b},{c}) at n={N} amplitude {i}: {x:?} vs {y:?}"
                );
            }

            let mut want = base.clone();
            apply_cswap_scan(&mut want, N, a, b, c);
            let mut got = base.clone();
            apply_cswap(&mut got, N, a, b, c);
            for (i, (x, y)) in want.iter().zip(got.iter()).enumerate() {
                assert!(
                    x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits(),
                    "PARALLEL cswap({a},{b},{c}) at n={N} amplitude {i}: {x:?} vs {y:?}"
                );
            }
            checked += 2;
        }
        assert_eq!(checked, 16, "expected 16 comparisons, ran {checked}");
    }

    /// **The CCX subspace walk is bit-identical to the scan it replaced**, over
    /// every ordering of three distinct qubits. Ordering is the whole risk: the
    /// walk sorts the operand positions to expand the index, so a triple where
    /// the target is the lowest bit exercises different arithmetic from one
    /// where it is the highest.
    #[test]
    fn ccx_subspace_walk_matches_the_scan_on_every_triple() {
        let mut checked = 0;
        for n in 3..=7usize {
            for c0 in 0..n {
                for c1 in 0..n {
                    for t in 0..n {
                        if c0 == c1 || c0 == t || c1 == t {
                            continue;
                        }
                        let mut want = seeded_state(n);
                        apply_ccx_scan(&mut want, n, c0, c1, t);
                        let mut got = seeded_state(n);
                        apply_ccx(&mut got, n, c0, c1, t);
                        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                            assert!(
                                a.re.to_bits() == b.re.to_bits()
                                    && a.im.to_bits() == b.im.to_bits(),
                                "n={n} ccx({c0},{c1},{t}) amplitude {i}: {a:?} vs {b:?}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100, "only {checked} triples compared");
    }

    /// Same for CSwap.
    #[test]
    fn cswap_subspace_walk_matches_the_scan_on_every_triple() {
        let mut checked = 0;
        for n in 3..=7usize {
            for c in 0..n {
                for t0 in 0..n {
                    for t1 in 0..n {
                        if c == t0 || c == t1 || t0 == t1 {
                            continue;
                        }
                        let mut want = seeded_state(n);
                        apply_cswap_scan(&mut want, n, c, t0, t1);
                        let mut got = seeded_state(n);
                        apply_cswap(&mut got, n, c, t0, t1);
                        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                            assert!(
                                a.re.to_bits() == b.re.to_bits()
                                    && a.im.to_bits() == b.im.to_bits(),
                                "n={n} cswap({c},{t0},{t1}) amplitude {i}: {a:?} vs {b:?}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100, "only {checked} triples compared");
    }

    /// **The specialised CX is bit-identical to the dense 4x4 it replaced.**
    ///
    /// `CX` no longer goes through `apply_2q`, so the existing group-walk
    /// oracle does not cover it — this is its replacement. Exactness (not a
    /// tolerance) is the right assertion because a permutation performs no
    /// arithmetic: it moves amplitudes. If this ever needs a tolerance,
    /// something is wrong with the claim, not with the epsilon.
    #[test]
    fn specialised_cx_is_bit_identical_to_the_dense_gate() {
        let mut checked = 0;
        for n in 2..=7usize {
            for c in 0..n {
                for t in 0..n {
                    if c == t {
                        continue;
                    }
                    let mut want = seeded_state(n);
                    apply_2q(&mut want, n, c, t, &crate::gates::cx());
                    let mut got = seeded_state(n);
                    apply_cx(&mut got, n, c, t);
                    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits(),
                            "n={n} cx({c},{t}) amplitude {i}: {a:?} vs {b:?}"
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 100,
            "only {checked} (control, target) pairs compared"
        );
    }

    /// **Every gate routed to the diagonal kernel agrees with the dense path,
    /// bit for bit.**
    ///
    /// The risk this guards is mis-ROUTING: sending a gate with non-zero
    /// off-diagonals down a path that ignores them silently drops half the
    /// operator. `diag_from` debug-asserts the off-diagonals are zero, and this
    /// checks the values that survive are the same ones the dense path
    /// produces.
    #[test]
    fn diagonal_gates_are_bit_identical_to_the_dense_path() {
        use crate::gates;
        let cases: Vec<(&str, gates::Gate1Q)> = vec![
            ("z", gates::z()),
            ("s", gates::s()),
            ("sdg", gates::sdg()),
            ("t", gates::t()),
            ("tdg", gates::tdg()),
            ("rz(0.7)", gates::rz(0.7)),
            ("rz(-2.1)", gates::rz(-2.1)),
        ];
        for (label, g) in cases {
            assert!(
                g[1] == Complex64::new(0.0, 0.0) && g[2] == Complex64::new(0.0, 0.0),
                "{label} is not diagonal — it must not be routed to the diagonal kernel"
            );
            for n in 1..=6usize {
                for q in 0..n {
                    let mut want = seeded_state(n);
                    apply_1q(&mut want, n, q, &g);
                    let mut got = seeded_state(n);
                    apply_diagonal_1q(&mut got, n, q, g[0], g[3]);
                    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                        // Numeric equality, NOT `to_bits()`. The two paths
                        // differ on the sign of zero: the dense path's
                        // `a*x + 0*y` normalises `-0.0` to `+0.0` and the
                        // diagonal path has no addition to do that. `==`
                        // treats them as equal, which is the right standard
                        // here — every consumer squares or compares these, and
                        // both operations are blind to the sign of zero.
                        assert!(a == b, "{label} n={n} q={q} amplitude {i}: {a:?} vs {b:?}");
                    }
                }
            }
        }
    }

    /// The one way the diagonal path differs from the dense one: **the sign of
    /// zero**. Pinned deliberately so it stays a known, bounded difference
    /// rather than a surprise — and so that anyone who tightens the oracle
    /// above back to `to_bits()` finds out why it is not.
    #[test]
    fn the_diagonal_path_differs_only_in_the_sign_of_zero() {
        let g = crate::gates::z();
        let mut dense = vec![Complex64::new(3.0, 0.0); 4];
        let mut diag = dense.clone();
        apply_1q(&mut dense, 2, 1, &g);
        apply_diagonal_1q(&mut diag, 2, 1, g[0], g[3]);
        for (a, b) in diag.iter().zip(dense.iter()) {
            assert_eq!(a, b, "values must be numerically equal");
        }
        // And the difference really is only the zero sign, not magnitude.
        let differing = diag
            .iter()
            .zip(dense.iter())
            .filter(|(a, b)| a.im.to_bits() != b.im.to_bits())
            .count();
        assert!(
            differing > 0,
            "expected at least one signed-zero difference; if this stops being \
             true the caveat above can be dropped"
        );
    }

    /// **A three-qubit gate with aliased operands is refused, not mishandled.**
    ///
    /// `apply_2q` was given an assert for this class; its three-qubit siblings
    /// were not, and they failed in two different silent ways:
    ///
    /// * `apply_ccx` with `target == c0` — the guard wants the target bit clear
    ///   and the control bit set, and they are the same bit, so no index ever
    ///   matches. A dropped operation, reported as success.
    /// * `apply_cswap` with `control == t0` — the partner index
    ///   `j = (i & !mask_t0) | mask_t1` clears the CONTROL bit, so the swap runs
    ///   between a control-1 amplitude and a control-0 one. Not a dropped gate:
    ///   a wrong state, silently.
    #[test]
    fn three_qubit_kernels_refuse_aliased_operands() {
        use std::panic::catch_unwind;
        let fresh = || {
            let mut v = vec![Complex64::new(0.0, 0.0); 8];
            v[0] = Complex64::new(1.0, 0.0);
            v
        };
        // Every aliased shape of both kernels, as (label, is_ccx, a, b, c).
        for (label, is_ccx, a, b, c) in [
            ("ccx target == c0", true, 0, 1, 0),
            ("ccx target == c1", true, 0, 1, 1),
            // The one aliased case the loop gets RIGHT (it degenerates to an
            // exact CX). Still refused: a caller must not be able to build a
            // three-qubit gate on two qubits and get a two-qubit one back.
            ("ccx c0 == c1", true, 1, 1, 2),
            ("cswap control == t0", false, 0, 0, 1),
            ("cswap control == t1", false, 0, 1, 0),
            ("cswap t0 == t1", false, 0, 1, 1),
        ] {
            let r = catch_unwind(move || {
                let mut st = fresh();
                if is_ccx {
                    apply_ccx(&mut st, 3, a, b, c);
                } else {
                    apply_cswap(&mut st, 3, a, b, c);
                }
            });
            assert!(r.is_err(), "{label}: must panic rather than mishandle");
        }
    }

    /// The guard must not fire on well-formed operands — a refusal on valid
    /// gates would be worse than the silence it replaces.
    #[test]
    fn three_qubit_kernels_still_work_on_distinct_operands() {
        let mut s = vec![Complex64::new(0.0, 0.0); 8];
        s[0b011] = Complex64::new(1.0, 0.0); // both controls set, target clear
        apply_ccx(&mut s, 3, 0, 1, 2);
        assert!(
            (s[0b111] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "CCX must move |011> to |111>"
        );

        let mut s = vec![Complex64::new(0.0, 0.0); 8];
        s[0b011] = Complex64::new(1.0, 0.0); // control set, t0 set, t1 clear
        apply_cswap(&mut s, 3, 0, 1, 2);
        assert!(
            (s[0b101] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "CSWAP must move |011> to |101>"
        );
    }

    #[test]
    fn group_walk_is_bit_identical_to_the_scan_on_every_qubit_pair() {
        for n in 2..=7 {
            let gate = dense_gate(n as u64);
            for q0 in 0..n {
                for q1 in 0..n {
                    if q0 == q1 {
                        continue;
                    }
                    let start = dense_state(n, (n * 100 + q0 * 10 + q1) as u64);
                    let mut want = start.clone();
                    let mut got = start;
                    apply_2q_scan(&mut want, n, q0, q1, &gate);
                    apply_2q(&mut got, n, q0, q1, &gate);
                    for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
                        // to_bits, not ==: `0.0 == -0.0` is true and would
                        // hide exactly the kind of slip this test exists for.
                        assert_eq!(
                            (w.re.to_bits(), w.im.to_bits()),
                            (g.re.to_bits(), g.im.to_bits()),
                            "n={n} q0={q0} q1={q1} amplitude {i}: {w} vs {g}"
                        );
                    }
                }
            }
        }
    }

    /// Both orderings of the operands, since the scan's swap-and-transpose
    /// branch is the one the group walk inherits and could silently drop.
    #[test]
    fn the_two_qubit_order_still_transposes_the_gate() {
        let n = 4;
        let gate = dense_gate(77);
        let start = dense_state(n, 4242);

        let mut lo_first = start.clone();
        let mut hi_first = start;
        apply_2q(&mut lo_first, n, 1, 3, &gate);
        apply_2q(&mut hi_first, n, 3, 1, &gate);
        assert!(
            lo_first
                .iter()
                .zip(hi_first.iter())
                .any(|(a, b)| a.re.to_bits() != b.re.to_bits()),
            "apply_2q(q0=1,q1=3) and apply_2q(q0=3,q1=1) must differ for an \
             asymmetric gate — if they agree, the operand order is being ignored"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bell_circuit() -> CircuitIR {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        c.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(0), Qubit(1)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        c
    }

    #[test]
    fn test_bell_state_statevector() {
        let backend = StatevectorBackend::new();
        let circuit = make_bell_circuit();
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };

        let result = backend.execute(&circuit, &params, &config).unwrap();
        let sv = result.statevector();

        // Bell state: (|00> + |11>) / sqrt(2)
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - expected).abs() < 1e-10);
        assert!(sv[1].norm() < 1e-10);
        assert!(sv[2].norm() < 1e-10);
        assert!((sv[3].re - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state_counts() {
        let backend = StatevectorBackend::new();
        let circuit = make_bell_circuit();
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: Some(10000),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Skip,
        };

        let result = backend.execute(&circuit, &params, &config).unwrap();
        let counts = result.counts();

        // Should only have |00> and |11>
        let key = |k: u64| omega_core::outcome::Outcome::from_u64(k, 2);
        let c00 = counts.get(&key(0)).copied().unwrap_or(0);
        let c11 = counts.get(&key(3)).copied().unwrap_or(0);
        assert!(c00 + c11 == 10000);
        // Each should be roughly 5000
        assert!(c00 > 4500 && c00 < 5500);
    }

    #[test]
    fn test_x_gate() {
        let backend = StatevectorBackend::new();
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.add_op(GateOp {
            gate: GateKind::X,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });

        let result = backend
            .execute(
                &circuit,
                &ParameterBinding::new(),
                &ExecConfig {
                    shots: None,
                    seed: None,
                    mid_circuit_mode: MidCircuitMode::Skip,
                },
            )
            .unwrap();
        let sv = result.statevector();

        // X|0> = |1>
        assert!(sv[0].norm() < 1e-10);
        assert!((sv[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parametric_rz() {
        let backend = StatevectorBackend::new();
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        circuit.symbols.insert(0, "theta".to_string());

        let mut params = ParameterBinding::new();
        params.bind(0, std::f64::consts::PI);

        let result = backend
            .execute(
                &circuit,
                &params,
                &ExecConfig {
                    shots: None,
                    seed: None,
                    mid_circuit_mode: MidCircuitMode::Skip,
                },
            )
            .unwrap();
        let sv = result.statevector();

        // H|0> = (|0>+|1>)/sqrt(2), then Rz(pi) gives (e^{-ipi/2}|0> + e^{ipi/2}|1>)/sqrt(2)
        // = (-i|0> + i|1>)/sqrt(2)
        let expected_0 = Complex64::new(0.0, -1.0 / 2.0_f64.sqrt());
        let expected_1 = Complex64::new(0.0, 1.0 / 2.0_f64.sqrt());
        assert!((sv[0] - expected_0).norm() < 1e-10);
        assert!((sv[1] - expected_1).norm() < 1e-10);
    }

    #[test]
    fn test_ghz_state() {
        let backend = StatevectorBackend::new();
        let mut circuit = CircuitIR::new(3, CircuitType::GateBased);
        // H on qubit 0
        circuit.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        // CX 0->1
        circuit.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(0), Qubit(1)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        // CX 1->2
        circuit.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(1), Qubit(2)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });

        let result = backend
            .execute(
                &circuit,
                &ParameterBinding::new(),
                &ExecConfig {
                    shots: None,
                    seed: None,
                    mid_circuit_mode: MidCircuitMode::Skip,
                },
            )
            .unwrap();
        let sv = result.statevector();

        // GHZ: (|000> + |111>) / sqrt(2)
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - expected).abs() < 1e-10);
        assert!((sv[7].re - expected).abs() < 1e-10);
        for i in 1..7 {
            assert!(sv[i].norm() < 1e-10);
        }
    }

    #[test]
    fn test_expectation_z() {
        let backend = StatevectorBackend::new();
        // |0> state -> <Z> = 1
        let circuit = CircuitIR::new(1, CircuitType::GateBased);
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };
        let val = backend
            .expectation(&circuit, &ParameterBinding::new(), &obs)
            .unwrap();
        assert!((val - 1.0).abs() < 1e-10);

        // |1> state -> <Z> = -1
        let mut circuit2 = CircuitIR::new(1, CircuitType::GateBased);
        circuit2.add_op(GateOp {
            gate: GateKind::X,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        let val2 = backend
            .expectation(&circuit2, &ParameterBinding::new(), &obs)
            .unwrap();
        assert!((val2 + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expectation_multi_matches_per_observable_loop() {
        // Three-qubit Bell-style state: H q0; CX 0,1; CX 1,2 — gives
        // (|000⟩+|111⟩)/√2 with ⟨Z_0⟩=⟨Z_1⟩=⟨Z_2⟩=0 but ⟨Z_0Z_1⟩=⟨Z_1Z_2⟩=1.
        // Pin that `expectation_multi` returns the same numbers as the
        // per-observable loop, against a circuit whose expectations are
        // not all equal so a typo would fail the assertion.
        let backend = StatevectorBackend::new();
        let mut circuit = CircuitIR::new(3, CircuitType::GateBased);
        circuit.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(0), Qubit(1)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(1), Qubit(2)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        let observables = vec![
            Observable {
                terms: vec![(1.0, vec![(0, PauliOp::Z)])],
            },
            Observable {
                terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
            },
            Observable {
                terms: vec![(0.5, vec![(2, PauliOp::Z)]), (0.25, vec![])],
            },
        ];
        let params = ParameterBinding::new();

        let multi = backend
            .expectation_multi(&circuit, &params, &observables)
            .unwrap();
        let per_obs: Vec<f64> = observables
            .iter()
            .map(|o| backend.expectation(&circuit, &params, o).unwrap())
            .collect();

        assert_eq!(multi.len(), 3);
        for (m, p) in multi.iter().zip(per_obs.iter()) {
            assert!(
                (m - p).abs() < 1e-12,
                "expectation_multi disagreed with per-observable loop: {m} vs {p}"
            );
        }
        assert!(
            multi[0].abs() < 1e-12,
            "⟨Z_0⟩ on (|000⟩+|111⟩)/√2 must be 0"
        );
        assert!((multi[1] - 1.0).abs() < 1e-12, "⟨Z_0Z_1⟩ must be 1");
    }

    #[test]
    fn test_expectation_multi_empty_returns_empty() {
        let backend = StatevectorBackend::new();
        let circuit = CircuitIR::new(1, CircuitType::GateBased);
        let multi = backend
            .expectation_multi(&circuit, &ParameterBinding::new(), &[])
            .unwrap();
        assert!(multi.is_empty());
    }

    #[test]
    fn test_parameter_shift_gradient() {
        use omega_core::gradient::{compute_gradient, GradMethod};

        // Ry(θ)|0>: <Z> = cos(θ), d<Z>/dθ = -sin(θ)
        let backend = StatevectorBackend::new();
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.symbols.insert(0, "theta".to_string());
        circuit.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });

        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };

        let theta = std::f64::consts::FRAC_PI_3;
        let mut params = ParameterBinding::new();
        params.bind(0, theta);

        let grads = compute_gradient(
            &backend,
            &circuit,
            &params,
            &obs,
            &GradMethod::ParameterShift,
        )
        .unwrap();

        let expected = -theta.sin();
        assert!(
            (grads[0].1 - expected).abs() < 1e-10,
            "param-shift grad = {} (expected {})",
            grads[0].1,
            expected
        );
    }

    #[test]
    fn test_gradient_two_params() {
        use omega_core::gradient::{compute_gradient, GradMethod};

        // Ry(θ₁) Rz(θ₂)|0>
        // <Z> = cos(θ₁) (Rz doesn't change Z expectation on |+/-> states)
        // d<Z>/dθ₁ = -sin(θ₁), d<Z>/dθ₂ = 0
        let backend = StatevectorBackend::new();
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.symbols.insert(0, "theta1".to_string());
        circuit.symbols.insert(1, "theta2".to_string());
        circuit.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![ParamExpr::Symbol(1)],
            classical_bit: None,
            condition: None,
        });

        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };

        let theta1 = 0.8;
        let theta2 = 1.5;
        let mut params = ParameterBinding::new();
        params.bind(0, theta1);
        params.bind(1, theta2);

        let grads = compute_gradient(
            &backend,
            &circuit,
            &params,
            &obs,
            &GradMethod::ParameterShift,
        )
        .unwrap();

        // d<Z>/dθ₁ = -sin(θ₁)
        let g0 = grads.iter().find(|(id, _)| *id == 0).unwrap().1;
        assert!(
            (g0 - (-theta1.sin())).abs() < 1e-10,
            "dZ/dtheta1 = {} (expected {})",
            g0,
            -theta1.sin()
        );

        // d<Z>/dθ₂ = 0 (Rz commutes with Z)
        let g1 = grads.iter().find(|(id, _)| *id == 1).unwrap().1;
        assert!(g1.abs() < 1e-10, "dZ/dtheta2 = {} (expected 0)", g1);
    }

    #[test]
    fn test_reset_from_one() {
        // X|0⟩ = |1⟩, then reset → |0⟩
        let backend = StatevectorBackend::new();
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.add_op(GateOp {
            gate: GateKind::X,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::Reset,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });

        let result = backend
            .execute(
                &circuit,
                &ParameterBinding::new(),
                &ExecConfig {
                    shots: None,
                    seed: None,
                    mid_circuit_mode: MidCircuitMode::Skip,
                },
            )
            .unwrap();
        let sv = result.statevector();
        assert!((sv[0].re - 1.0).abs() < 1e-10, "should be |0⟩ after reset");
        assert!(sv[1].norm() < 1e-10);
    }

    #[test]
    fn test_reset_from_superposition() {
        // H|0⟩ = |+⟩, then reset → |0⟩
        let backend = StatevectorBackend::new();
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::Reset,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });

        let result = backend
            .execute(
                &circuit,
                &ParameterBinding::new(),
                &ExecConfig {
                    shots: None,
                    seed: None,
                    mid_circuit_mode: MidCircuitMode::Skip,
                },
            )
            .unwrap();
        let sv = result.statevector();
        assert!((sv[0].re - 1.0).abs() < 1e-10, "should be |0⟩ after reset");
        assert!(sv[1].norm() < 1e-10);
    }

    #[test]
    fn test_reset_partial_entangled() {
        // Bell state on qubits 0,1, then reset qubit 0 → qubit 0 is |0⟩ on every
        // trajectory and qubit 1 is left MIXED. A statevector cannot represent
        // that mixture, so:
        //   * analytic mode (shots = None) refuses — this used to return the
        //     coherent fold's pure state and call it the answer;
        //   * with shots, each shot is an independent trajectory and q1 comes
        //     out 50/50 (Qiskit Aer ground truth; see tests/reset_channel.rs).
        let backend = StatevectorBackend::new();
        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        for (gate, qubits) in [
            (GateKind::H, vec![Qubit(0)]),
            (GateKind::CX, vec![Qubit(0), Qubit(1)]),
            (GateKind::Reset, vec![Qubit(0)]),
        ] {
            circuit.add_op(GateOp {
                gate,
                qubits: qubits.into_iter().collect(),
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }

        let err = backend
            .execute(
                &circuit,
                &ParameterBinding::new(),
                &ExecConfig {
                    shots: None,
                    seed: None,
                    mid_circuit_mode: MidCircuitMode::Skip,
                },
            )
            .expect_err("analytic reset on an entangled qubit must refuse");
        assert!(format!("{err:?}").contains("entangled"));

        let counts = backend
            .execute(
                &circuit,
                &ParameterBinding::new(),
                &ExecConfig {
                    shots: Some(4000),
                    seed: Some(7),
                    mid_circuit_mode: MidCircuitMode::Skip,
                },
            )
            .unwrap();
        let m = counts.counts();
        let q0_ones: u32 = m
            .iter()
            .filter(|(k, _)| k.bit(0) == 1)
            .map(|(_, v)| v)
            .sum();
        let q1_ones: u32 = m
            .iter()
            .filter(|(k, _)| k.bit(1) == 1)
            .map(|(_, v)| v)
            .sum();
        assert_eq!(q0_ones, 0, "q0 must be |0⟩ on every shot, got {m:?}");
        assert!(
            q1_ones.abs_diff(2000) < 250,
            "q1 must be maximally mixed (~2000/4000 ones), got {q1_ones}"
        );
    }

    // --- Noise channel tests ---

    fn single_x_circuit() -> CircuitIR {
        let mut c = CircuitIR::new(1, CircuitType::GateBased);
        c.add_op(GateOp {
            gate: GateKind::X,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        c
    }

    #[test]
    fn noise_noiseless_model_matches_clean() {
        let backend = NoisyStatevectorBackend::with_model(NoiseModel::default(), Some(0));
        let circuit = make_bell_circuit();
        let cfg = ExecConfig {
            shots: Some(2048),
            seed: Some(0),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let counts = result.counts();
        let total: u32 = counts.values().sum();
        assert_eq!(total, 2048);
        // Noiseless Bell state → only |00⟩ (0) and |11⟩ (3) appear.
        let bell = |k: u64| omega_core::outcome::Outcome::from_u64(k, 2);
        assert!(counts.keys().all(|k| *k == bell(0) || *k == bell(3)));
    }

    /// Look up outcome `k` at the width the counts carry.
    #[allow(dead_code)]
    fn okey(map: &std::collections::HashMap<omega_core::outcome::Outcome, u32>, k: u64) -> u32 {
        let w = map.keys().next().map(|o| o.width()).unwrap_or(0);
        map.get(&omega_core::outcome::Outcome::from_u64(k, w))
            .copied()
            .unwrap_or(0)
    }

    #[test]
    fn noise_amplitude_damping_decays_to_zero() {
        // Prepare |1⟩ (via X), apply amplitude damping with γ=1 repeatedly.
        // After one jump with γ=1, state should collapse to |0⟩ on every trajectory.
        let circuit = single_x_circuit();
        let model = NoiseModel {
            amplitude_damping: 1.0.into(),
            ..Default::default()
        };
        let backend = NoisyStatevectorBackend::with_model(model, Some(42));
        let cfg = ExecConfig {
            shots: Some(1024),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let counts = result.counts();
        let p0 = okey(counts, 0) as f64 / 1024.0;
        assert!(
            p0 > 0.95,
            "γ=1 amplitude damping should drive to |0⟩, got P(|0⟩)={}",
            p0
        );
    }

    #[test]
    fn noise_readout_flip_zero_state() {
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.num_classical_bits = 1;
        // Start in |0⟩ (no gates needed); just sample 2000 shots with heavy readout flip.
        let model = NoiseModel {
            readout: omega_core::noise::ReadoutError::symmetric(1.0), // always flip
            ..Default::default()
        };
        let backend = NoisyStatevectorBackend::with_model(model, Some(1));
        let cfg = ExecConfig {
            shots: Some(2000),
            seed: Some(1),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let counts = result.counts();
        let p1 = okey(counts, 1) as f64 / 2000.0;
        // Every shot flipped 0 → 1, so P(1) ≈ 1.0
        assert!(
            p1 > 0.98,
            "readout_flip=1 should turn every |0⟩ into 1, got {}",
            p1
        );
    }

    #[test]
    fn noise_pauli_channel_conserves_probabilities() {
        // Apply a pure X Pauli channel (p_X = 1). Starting from |0⟩, final state
        // should collapse to |1⟩ in every trajectory.
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.add_op(GateOp {
            gate: GateKind::Id,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        let model = NoiseModel {
            pauli: Some(omega_core::noise::PauliChannel {
                x: 1.0.into(),
                y: 0.0.into(),
                z: 0.0.into(),
            }),
            ..Default::default()
        };
        let backend = NoisyStatevectorBackend::with_model(model, Some(9));
        let cfg = ExecConfig {
            shots: Some(512),
            seed: Some(9),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let counts = result.counts();
        let p1 = okey(counts, 1) as f64 / 512.0;
        assert!(
            p1 > 0.98,
            "pure-X Pauli channel on |0⟩ should give |1⟩, got P(|1⟩)={}",
            p1
        );
    }

    #[test]
    fn noise_phase_damping_preserves_populations() {
        // Phase damping should NOT change |0⟩ vs |1⟩ populations — only destroy coherences.
        // Prepare |+⟩ (H|0⟩), apply lots of phase damping, measure in Z basis:
        // P(0) ≈ P(1) ≈ 0.5 still.
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        let model = NoiseModel {
            phase_damping: 0.5.into(),
            ..Default::default()
        };
        let backend = NoisyStatevectorBackend::with_model(model, Some(3));
        let cfg = ExecConfig {
            shots: Some(4096),
            seed: Some(3),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let counts = result.counts();
        let p0 = okey(counts, 0) as f64 / 4096.0;
        // Populations unchanged by pure dephasing; tolerance loose for 4k samples.
        assert!(
            (p0 - 0.5).abs() < 0.05,
            "phase damping should keep Z populations at 0.5, got P(0)={}",
            p0
        );
    }
}
