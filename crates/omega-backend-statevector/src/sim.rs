use std::collections::HashMap;

use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use omega_core::circuit::*;
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
            return Ok(ExecResult::Counts(counts));
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
                    Ok(ExecResult::Counts(counts))
                } else {
                    let counts = sample_counts(&state, n, shots, Some(rng.random()));
                    Ok(ExecResult::Counts(counts))
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
        // Get statevector
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
                Ok(ExecResult::Counts(counts))
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
                Ok(ExecResult::Counts(counts))
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
        GateKind::Z => apply_1q(state, n, op.qubits[0].0 as usize, &gates::z()),
        GateKind::S => apply_1q(state, n, op.qubits[0].0 as usize, &gates::s()),
        GateKind::Sdg => apply_1q(state, n, op.qubits[0].0 as usize, &gates::sdg()),
        GateKind::T => apply_1q(state, n, op.qubits[0].0 as usize, &gates::t()),
        GateKind::Tdg => apply_1q(state, n, op.qubits[0].0 as usize, &gates::tdg()),
        GateKind::Id => {} // no-op
        GateKind::Rx => apply_1q(state, n, op.qubits[0].0 as usize, &gates::rx(resolved[0])),
        GateKind::Ry => apply_1q(state, n, op.qubits[0].0 as usize, &gates::ry(resolved[0])),
        GateKind::Rz => apply_1q(state, n, op.qubits[0].0 as usize, &gates::rz(resolved[0])),
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
        GateKind::CX => apply_2q(
            state,
            n,
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::cx(),
        ),
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
pub(crate) fn apply_1q(state: &mut [Complex64], n: usize, q: usize, gate: &gates::Gate1Q) {
    let dim = 1usize << n;
    let step = 1usize << q;

    let mut i = 0;
    while i < dim {
        for j in i..i + step {
            let i0 = j;
            let i1 = j + step;
            let a0 = state[i0];
            let a1 = state[i1];
            state[i0] = gate[0] * a0 + gate[1] * a1;
            state[i1] = gate[2] * a0 + gate[3] * a1;
        }
        i += step << 1;
    }
}

/// Apply a two-qubit gate to qubits (q0, q1) in an n-qubit statevector.
/// q0 is the more significant qubit in the gate's matrix (control for CX, etc.).
pub(crate) fn apply_2q(
    state: &mut [Complex64],
    n: usize,
    q0: usize,
    q1: usize,
    gate: &gates::Gate2Q,
) {
    let dim = 1usize << n;

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

    for i in 0..dim {
        // Only process basis states where both target bits are 0
        if (i & step_a) != 0 || (i & step_b) != 0 {
            continue;
        }
        let i00 = i;
        let i01 = i | step_b;
        let i10 = i | step_a;
        let i11 = i | step_a | step_b;

        let a00 = state[i00];
        let a01 = state[i01];
        let a10 = state[i10];
        let a11 = state[i11];

        state[i00] = g[0] * a00 + g[1] * a01 + g[2] * a10 + g[3] * a11;
        state[i01] = g[4] * a00 + g[5] * a01 + g[6] * a10 + g[7] * a11;
        state[i10] = g[8] * a00 + g[9] * a01 + g[10] * a10 + g[11] * a11;
        state[i11] = g[12] * a00 + g[13] * a01 + g[14] * a10 + g[15] * a11;
    }
}

/// Apply Toffoli (CCX) gate: flip target if both controls are |1>.
fn apply_ccx(state: &mut [Complex64], n: usize, c0: usize, c1: usize, target: usize) {
    let dim = 1usize << n;
    let mask_c0 = 1usize << c0;
    let mask_c1 = 1usize << c1;
    let mask_t = 1usize << target;

    for i in 0..dim {
        // Only process states where both controls are 1 and target is 0
        if (i & mask_c0) != 0 && (i & mask_c1) != 0 && (i & mask_t) == 0 {
            let j = i | mask_t;
            state.swap(i, j);
        }
    }
}

/// Apply Fredkin (CSWAP) gate: swap targets if control is |1>.
fn apply_cswap(state: &mut [Complex64], n: usize, control: usize, t0: usize, t1: usize) {
    let dim = 1usize << n;
    let mask_c = 1usize << control;
    let mask_t0 = 1usize << t0;
    let mask_t1 = 1usize << t1;

    for i in 0..dim {
        // Control is 1, t0 is 1, t1 is 0 -> swap with t0=0, t1=1
        if (i & mask_c) != 0 && (i & mask_t0) != 0 && (i & mask_t1) == 0 {
            let j = (i & !mask_t0) | mask_t1;
            state.swap(i, j);
        }
    }
}

/// Sample measurement outcomes from the statevector.
fn sample_counts(
    state: &[Complex64],
    num_qubits: usize,
    shots: u32,
    seed: Option<u64>,
) -> HashMap<u64, u32> {
    let probs: Vec<f64> = state.iter().map(|a| a.norm_sqr()).collect();

    // Build cumulative distribution
    let mut cumulative = Vec::with_capacity(probs.len());
    let mut sum = 0.0;
    for p in &probs {
        sum += p;
        cumulative.push(sum);
    }
    // Normalize for floating point errors
    if let Some(last) = cumulative.last_mut() {
        *last = 1.0;
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => rand::make_rng::<StdRng>(),
    };

    let mut counts = HashMap::new();
    let _ = num_qubits; // used by caller for formatting

    for _ in 0..shots {
        let r: f64 = rng.random();
        let idx = cumulative.partition_point(|&c| c < r);
        let bitstring = idx.min(probs.len() - 1) as u64;
        *counts.entry(bitstring).or_insert(0) += 1;
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
        let c00 = counts.get(&0).copied().unwrap_or(0);
        let c11 = counts.get(&3).copied().unwrap_or(0);
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
        let q0_ones: u32 = m.iter().filter(|(k, _)| *k & 1 != 0).map(|(_, v)| v).sum();
        let q1_ones: u32 = m.iter().filter(|(k, _)| *k & 2 != 0).map(|(_, v)| v).sum();
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
        assert!(counts.keys().all(|k| *k == 0 || *k == 3));
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
        let p0 = *counts.get(&0).unwrap_or(&0) as f64 / 1024.0;
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
        let p1 = *counts.get(&1).unwrap_or(&0) as f64 / 2000.0;
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
        let p1 = *counts.get(&1).unwrap_or(&0) as f64 / 512.0;
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
        let p0 = *counts.get(&0).unwrap_or(&0) as f64 / 4096.0;
        // Populations unchanged by pure dephasing; tolerance loose for 4k samples.
        assert!(
            (p0 - 0.5).abs() < 0.05,
            "phase damping should keep Z populations at 0.5, got P(0)={}",
            p0
        );
    }
}
