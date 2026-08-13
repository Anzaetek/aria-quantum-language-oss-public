//! MPS (Matrix Product State) backend for circuit simulation.
//!
//! Uses tensor network contraction with truncated SVD to efficiently
//! simulate circuits with limited entanglement (low bond dimension).

use std::collections::HashMap;
use std::sync::Mutex;

use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use omega_core::circuit::*;
use omega_core::error::{OmegaError, Result};
use omega_core::executor::*;
use omega_core::noise::{NoiseModel, ReadoutError};
use omega_core::params::ParameterBinding;

use crate::gates;
use crate::mps::Mps;
use crate::svd::truncated_svd_flat;
use crate::{Contract2qFn, SvdFlatFn};

/// Truncation certificate for the most recent run through an [`MpsBackend`].
///
/// `discarded_weight` is the accumulated relative singular-value weight thrown
/// away across every two-site split (0.0 = exact, e.g. χ ≥ 2^(n/2)); a growing
/// value is the honest signal the bond dimension is under-provisioned — the
/// standard MPS fidelity proxy. `max_bond_reached` is the largest post-
/// truncation bond dimension seen, i.e. how close the run came to saturating
/// `max_bond_dim`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MpsRunStats {
    pub discarded_weight: f64,
    pub max_bond_reached: usize,
}

/// Accumulated discarded weight above which a result is refused rather than
/// returned.
///
/// # Why there is a default at all
///
/// `discarded_weight` is the standard MPS fidelity proxy and this backend has
/// always computed it correctly — and then returned the state regardless, with
/// the number printed in a banner nothing consulted. Measured on a 19-qubit
/// chain:
///
/// ```text
///   circuit          discarded    TVD vs exact
///   qft_16           0.000e0      1.9e-14      correct
///   w_xy_grid_16q    9.773e-30    4.7e-15      correct
///   wide_chain_19q   6.586e0      5.1e-01      HALF THE DISTRIBUTION WRONG
/// ```
///
/// The signal separates the good runs from the bad one by thirty orders of
/// magnitude, so gating on it costs nothing in the cases that work.
///
/// # Why this value
///
/// The discarded weight bounds the infidelity, so `1e-6` means "at most about
/// one part in a million of the state was thrown away" — far above the `1e-30`
/// the honest runs above produce, and far below the `6.586` that produced a
/// useless answer. It is a ceiling on *silent* approximation, not a limit on
/// what the backend can do: [`MpsBackend::with_max_discarded_weight`] raises it
/// for callers who want an approximation and say so.
///
/// Note the observed values do not sit near this boundary from either side,
/// which is the point — a threshold chosen to sit between two clusters thirty
/// decades apart is not a tuned constant.
pub const DEFAULT_MAX_DISCARDED_WEIGHT: f64 = 1e-6;

/// MPS-based quantum circuit simulator.
///
/// The `max_bond_dim` parameter controls the trade-off between accuracy and
/// memory/time: higher values are more accurate but use more resources.
/// For states with bond dimension <= max_bond_dim, the simulation is exact.
pub struct MpsBackend {
    pub max_bond_dim: usize,
    /// Bond-compression SVD provider handed to every `Mps` this backend runs.
    /// CPU Jacobi by default; `with_svd_fn` swaps in a GPU `gesvdj` accelerator
    /// (see `omega-backend-mps-cuda`). Wiring here means the whole MPS circuit —
    /// adjacent gates and the SWAP network for distant gates — runs its
    /// truncations on the GPU, with a transparent CPU fallback.
    svd_fn: SvdFlatFn,
    /// Optional two-site-gate accelerator handed to every `Mps` this backend
    /// runs (the Metal θ-contraction; see `omega-backend-mps-metal`). `None` =
    /// the built-in contract+SVD path. Transparent CPU fall-through, so wiring
    /// it under a `metal` build never changes results below the GPU threshold.
    contract_fn: Option<Contract2qFn>,
    /// Adaptive truncation threshold ε: when `Some`, each split keeps the
    /// singular values above `ε · σ_max` up to `max_bond_dim` (the ceiling),
    /// so the bond grows with entanglement instead of always filling the cap.
    /// `None` = fixed-rank truncation at `max_bond_dim`.
    adaptive_eps: Option<f64>,
    /// Refuse a result whose accumulated discarded weight exceeds this.
    /// See [`MpsBackend::with_max_discarded_weight`] and [`DEFAULT_MAX_DISCARDED_WEIGHT`].
    max_discarded_weight: f64,
    /// Truncation certificate of the most recent `execute`/`expectation`.
    /// Interior-mutable because the `Backend` trait methods take `&self`;
    /// under batch parallelism it reflects one (arbitrary) row's run. Read via
    /// [`Self::last_run_stats`]. Never printed (library crates stay silent).
    stats: Mutex<MpsRunStats>,
}

impl MpsBackend {
    pub fn new(max_bond_dim: usize) -> Self {
        Self {
            max_bond_dim,
            svd_fn: truncated_svd_flat,
            contract_fn: None,
            adaptive_eps: None,
            max_discarded_weight: DEFAULT_MAX_DISCARDED_WEIGHT,
            stats: Mutex::new(MpsRunStats::default()),
        }
    }

    /// Raise (or lower) the discarded-weight ceiling this backend will return a
    /// result under. `f64::INFINITY` restores the old behaviour of returning
    /// whatever the truncation produced.
    ///
    /// Deliberately explicit: an approximate MPS result is a legitimate thing to
    /// want, but it should be *asked for*, not arrived at by default.
    pub fn with_max_discarded_weight(mut self, w: f64) -> Self {
        self.max_discarded_weight = w;
        self
    }

    /// Refuse a run whose truncation certificate says the state is not the one
    /// the circuit describes.
    fn check_truncation(&self) -> Result<()> {
        let st = self.last_run_stats();
        if st.discarded_weight > self.max_discarded_weight {
            return Err(OmegaError::Unsupported(format!(
                "MPS truncation discarded {:.3e} of the state (bond reached {}, ceiling \
                 {:.3e}). The result would not approximate the circuit: measured on a \
                 19-qubit chain, a discarded weight of 6.586 gave a total-variation \
                 distance of 0.51 from the exact distribution — half the distribution \
                 wrong, returned without complaint. Raise the bond dimension, or call \
                 `with_max_discarded_weight` to accept an approximation deliberately.",
                st.discarded_weight, st.max_bond_reached, self.max_discarded_weight
            )));
        }
        Ok(())
    }

    /// Enable adaptive bond truncation with relative singular-value threshold
    /// `eps` — the bond grows with the actual entanglement up to `max_bond_dim`
    /// (the hard ceiling), instead of always filling it. This is what
    /// `--backend mps:auto` selects.
    pub fn with_adaptive(mut self, eps: f64) -> Self {
        self.adaptive_eps = Some(eps);
        self
    }

    /// The truncation certificate of the most recent run — the discarded
    /// singular-value weight (fidelity proxy) and the peak bond dimension. Lets
    /// an embedder holding a concrete `MpsBackend` tell "χ was plenty" from
    /// "χ threw away half the state" without any CLI or trait change.
    pub fn last_run_stats(&self) -> MpsRunStats {
        *self.stats.lock().unwrap()
    }

    /// Record a finished run's truncation stats (last-writer-wins).
    fn record_stats(&self, mps: &Mps) {
        *self.stats.lock().unwrap() = MpsRunStats {
            discarded_weight: mps.discarded_weight,
            max_bond_reached: mps.max_bond_reached,
        };
    }

    /// Evolve |0…0⟩ through `circuit` once, returning the final chain and the
    /// classical register. One call = one trajectory: every `Reset` (and, in
    /// `Collapse` mode, every mid-circuit `Measure`) draws a fresh outcome from
    /// `rng`, so callers wanting shot statistics must invoke this per shot.
    fn evolve_once(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
        rng: &mut StdRng,
    ) -> Result<(Mps, Vec<u8>)> {
        let n = circuit.num_qubits as usize;
        let mut mps = Mps::zero_state(n, self.max_bond_dim);
        mps.set_svd_fn(self.svd_fn);
        if let Some(cf) = self.contract_fn {
            mps.set_contract_fn(cf);
        }
        if let Some(eps) = self.adaptive_eps {
            mps.set_adaptive_eps(eps);
        }
        let mut classical_bits = vec![0u8; circuit.num_classical_bits as usize];

        for op in &circuit.ops {
            if !op.condition_satisfied(&classical_bits) {
                continue;
            }
            match &op.gate {
                GateKind::Measure => {
                    if config.mid_circuit_mode == MidCircuitMode::Collapse {
                        let q = op.qubits[0].0 as usize;
                        let outcome = mps.measure_site(q, rng);
                        if let Some(cbit) = op.classical_bit {
                            if (cbit as usize) < classical_bits.len() {
                                classical_bits[cbit as usize] = outcome;
                            }
                        }
                    }
                }
                GateKind::Barrier => continue,
                _ => {
                    apply_gate_mps(&mut mps, op, params, rng)?;
                }
            }
        }
        Ok((mps, classical_bits))
    }

    /// Route this backend's bond-compression SVDs through `f` (e.g. the CUDA
    /// `gesvdj` accelerator). Falls back to CPU inside `f` when no GPU is
    /// present, so callers can wire it unconditionally under a `cuda` build.
    pub fn with_svd_fn(mut self, f: SvdFlatFn) -> Self {
        self.svd_fn = f;
        self
    }

    /// Route each adjacent two-qubit gate through the accelerator `f` (the Metal
    /// θ-contraction). Falls back to the built-in path inside `f` when no Metal
    /// device is present or the bond is below its threshold, so callers can wire
    /// it unconditionally under a `metal` build.
    pub fn with_contract_fn(mut self, f: Contract2qFn) -> Self {
        self.contract_fn = Some(f);
        self
    }
}

impl Backend for MpsBackend {
    fn name(&self) -> &str {
        "mps"
    }

    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult> {
        if circuit.circuit_type == CircuitType::Photonic {
            return Err(OmegaError::Unsupported(
                "MPS backend does not support photonic circuits".into(),
            ));
        }

        let mut rng: StdRng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng::<StdRng>(),
        };

        // Any STOCHASTIC evolution needs one independent trajectory per shot,
        // not one trajectory replayed `shots` times.
        //
        // `Reset` is a stochastic CHANNEL, not a gate (see `apply_reset_mps`),
        // and a `Collapse`-mode mid-circuit measurement is stochastic too.
        // This predicate tested `circuit_has_reset` ALONE, which is the exact
        // defect the CPU statevector backend fixed in `11888a9` and documents
        // at `omega-backend-statevector/src/sim.rs`, and which
        // `NoisyMpsBackend` already guards with `mps_collapses`. It never
        // propagated here.
        //
        // Consequence, measured on
        // `omega-bridges/tests/fixtures/crosscheck/12_feedforward_sometimes_false.qasm`
        // (`h q0; measure q0 -> c0; if (c==1) x q1; measure q1 -> c0`) at
        // 20000 shots, seed 7:
        //
        //   MpsBackend before: {0: 20000}          — certainty
        //   Qiskit Aer:        {0: ~9900, 1: ~10100} — a fair coin
        //
        // One trajectory drew c0 = 0, the guarded X never fired, and every
        // one of the 20000 shots then sampled that same collapsed chain. A
        // superposition reported as deterministic, silently. Found by the
        // N-way counts matrix (`omega-cli/tests/nway_counts.rs`) — no
        // in-tree-only comparison could have caught it, because the *noisy*
        // MPS backend was right and this one was wrong in a way that still
        // agreed with itself.
        let by_creg = mps_collapses(circuit, config);
        // A shot outcome is keyed by a u64, so a register wider than 64
        // qubits cannot be represented and every high bit would be silently
        // dropped — a confident wrong answer, not a truncated one. Refused here
        // rather than at the key-construction site so the message names the
        // circuit, and refused in EVERY backend that can exceed 64 qubits
        // because this is a property of the result type, not of any simulator.
        if config.shots.is_some() {
            omega_core::executor::check_counts_width(circuit.num_qubits as usize)?;
        }

        if let (true, Some(shots)) = (by_creg || circuit_has_reset(circuit), config.shots) {
            let mut counts = HashMap::new();
            for _ in 0..shots {
                let (mps, cbits) = self.evolve_once(circuit, params, config, &mut rng)?;
                self.record_stats(&mps);
                // In collapse mode the creg IS the outcome: the measures
                // already happened during evolution, and re-sampling the
                // final chain would report the post-collapse qubit state
                // rather than what was recorded. Matches the statevector and
                // noisy-MPS backends.
                let outcome = if by_creg {
                    mps_creg_to_u64(&cbits)
                } else {
                    let envs = mps.right_environments();
                    mps.sample_with_envs(&envs, &mut rng)
                };
                *counts.entry(outcome).or_insert(0) += 1;
            }
            // Checked AFTER the trajectories: `record_stats` is last-writer-wins
            // and the ceiling is about the run as a whole, not one shot.
            self.check_truncation()?;
            return Ok(ExecResult::Counts(counts));
        }

        let (mps, _classical_bits) = self.evolve_once(circuit, params, config, &mut rng)?;
        self.record_stats(&mps);
        // The truncation certificate gates the RESULT, in every mode. It was
        // computed and printed but consulted by nothing, so a run that
        // discarded 6.5x the state returned a distribution half of which was
        // wrong — with the evidence on screen.
        self.check_truncation()?;

        match config.shots {
            None => {
                let sv = mps.to_statevector();
                Ok(ExecResult::Statevector(sv))
            }
            Some(shots) => {
                // Right environments depend only on the final state, not on
                // the outcomes drawn — compute once, reuse for every shot.
                let envs = mps.right_environments();
                let mut counts = HashMap::new();
                for _ in 0..shots {
                    let outcome = mps.sample_with_envs(&envs, &mut rng);
                    *counts.entry(outcome).or_insert(0) += 1;
                }
                Ok(ExecResult::Counts(counts))
            }
        }
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        reject_reset_in_analytic_mode(circuit)?;
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

        let n = circuit.num_qubits;
        let mut total = 0.0;
        for (coeff, pauli_string) in &observable.terms {
            let val = expectation_pauli(sv, n, pauli_string);
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
        // Contract the MPS once into a flat statevector, then evaluate
        // every observable against the same `&sv`. The default trait
        // impl loops `expectation`, which re-runs the full MPS contract
        // per observable — N copies of the heavy sweep when the QML
        // trainer asks for ⟨Z_q⟩ on N measurement qubits. Mirrors the
        // CPU StatevectorBackend's override.
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
        let n = circuit.num_qubits;
        let mut out = Vec::with_capacity(observables.len());
        for obs in observables {
            let mut total = 0.0;
            for (coeff, pauli_string) in &obs.terms {
                total += coeff * expectation_pauli(sv, n, pauli_string);
            }
            out.push(total);
        }
        Ok(out)
    }
}

/// MPS backend with a per-gate + readout noise model (trajectory Monte-Carlo).
///
/// Mirrors [`NoisyStatevectorBackend`](omega_backend_statevector) but on a
/// matrix-product state. Noise **composes** with the same GPU accelerators as
/// [`MpsBackend`]: the channels are cheap single-site CPU operations, while the
/// heavy two-qubit θ-contraction / bond SVD still runs on the device (Metal /
/// CUDA) via the wired `contract_fn` / `svd_fn`.
///
/// Each shot is one independent trajectory: unitary channels (depolarizing,
/// dephasing, Pauli) are unravelled as a random single-qubit gate; amplitude
/// damping uses the local quantum-jump probability `γ·P(qubit = 1)` computed
/// from the MPS. Aggregating trajectories reproduces the density-matrix result.
pub struct NoisyMpsBackend {
    pub max_bond_dim: usize,
    svd_fn: SvdFlatFn,
    contract_fn: Option<Contract2qFn>,
    model: NoiseModel,
}

impl NoisyMpsBackend {
    /// Build with the given bond dimension and noise model (CPU SVD, no GPU
    /// contraction hook). Wire accelerators with [`Self::with_svd_fn`] /
    /// [`Self::with_contract_fn`], exactly as for [`MpsBackend`].
    pub fn with_model(max_bond_dim: usize, model: NoiseModel) -> Self {
        Self {
            max_bond_dim,
            svd_fn: truncated_svd_flat,
            contract_fn: None,
            model,
        }
    }

    /// Route bond-compression SVDs through `f` (e.g. the CUDA `gesvdj`
    /// accelerator); composes with the noise trajectories.
    pub fn with_svd_fn(mut self, f: SvdFlatFn) -> Self {
        self.svd_fn = f;
        self
    }

    /// Route adjacent two-qubit gates through `f` (the Metal θ-contraction);
    /// composes with the noise trajectories.
    pub fn with_contract_fn(mut self, f: Contract2qFn) -> Self {
        self.contract_fn = Some(f);
        self
    }

    fn fresh_mps(&self, n: usize) -> Mps {
        let mut mps = Mps::zero_state(n, self.max_bond_dim);
        mps.set_svd_fn(self.svd_fn);
        if let Some(cf) = self.contract_fn {
            mps.set_contract_fn(cf);
        }
        mps
    }

    /// Evolve one noisy trajectory to a final MPS (and the classical register,
    /// for `Collapse`-mode mid-circuit measurement).
    fn evolve(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
        rng: &mut StdRng,
    ) -> Result<(Mps, Vec<u8>)> {
        let n = circuit.num_qubits as usize;
        let mut mps = self.fresh_mps(n);
        let mut classical_bits = vec![0u8; circuit.num_classical_bits as usize];

        for op in &circuit.ops {
            if !op.condition_satisfied(&classical_bits) {
                continue;
            }
            match &op.gate {
                GateKind::Measure => {
                    if config.mid_circuit_mode == MidCircuitMode::Collapse {
                        let q = op.qubits[0].0 as usize;
                        let mut outcome = mps.measure_site(q, rng);
                        if !self.model.readout.is_zero() {
                            let p = self.model.readout.flip_prob(q, outcome);
                            if p > 0.0 && rng.random::<f64>() < p {
                                outcome ^= 1;
                            }
                        }
                        if let Some(cbit) = op.classical_bit {
                            if (cbit as usize) < classical_bits.len() {
                                classical_bits[cbit as usize] = outcome;
                            }
                        }
                    }
                }
                GateKind::Barrier => continue,
                _ => {
                    apply_gate_mps(&mut mps, op, params, rng)?;
                    if self.model.has_gate_channel() {
                        let gate_qubits: Vec<usize> =
                            op.qubits.iter().map(|q| q.0 as usize).collect();
                        for &q in &gate_qubits {
                            apply_mps_channel(&mut mps, &self.model, q, &gate_qubits, rng);
                        }
                    }
                }
            }
        }
        Ok((mps, classical_bits))
    }
}

impl Backend for NoisyMpsBackend {
    fn name(&self) -> &str {
        "noisy-mps"
    }

    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult> {
        if circuit.circuit_type == CircuitType::Photonic {
            return Err(OmegaError::Unsupported(
                "MPS backend does not support photonic circuits".into(),
            ));
        }
        let n = circuit.num_qubits as usize;
        let mut rng: StdRng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng::<StdRng>(),
        };

        match config.shots {
            // One trajectory's final state (a single MPS can't carry a mixture).
            None => {
                let (mps, _cbits) =
                    self.evolve(circuit, params, &config_skip_shots(config), &mut rng)?;
                Ok(ExecResult::Statevector(mps.to_statevector()))
            }
            // Per-trajectory when a channel acts during evolution OR when
            // mid-circuit measurement collapses the state (inherently stochastic
            // per shot).
            Some(shots) if self.model.has_gate_channel() || mps_collapses(circuit, config) => {
                let collapse = mps_collapses(circuit, config);
                let mut counts = HashMap::new();
                for _ in 0..shots {
                    let (mps, cbits) = self.evolve(circuit, params, config, &mut rng)?;
                    let outcome = if collapse {
                        // Mid-circuit measures recorded (readout-flipped) outcomes
                        // into the creg during evolution; key by the creg.
                        mps_creg_to_u64(&cbits)
                    } else {
                        let mut o = mps.sample(&mut rng);
                        if !self.model.readout.is_zero() {
                            o = flip_all_readout(o, n, &self.model.readout, &mut rng);
                        }
                        o
                    };
                    *counts.entry(outcome).or_insert(0) += 1;
                }
                Ok(ExecResult::Counts(counts))
            }
            Some(shots) => {
                // Readout-only: no channel acts during evolution, so a single
                // evolution is exact and every shot samples from it.
                let (mps, _cbits) = self.evolve(circuit, params, config, &mut rng)?;
                let envs = mps.right_environments();
                let mut counts = HashMap::new();
                for _ in 0..shots {
                    let mut outcome = mps.sample_with_envs(&envs, &mut rng);
                    if !self.model.readout.is_zero() {
                        outcome = flip_all_readout(outcome, n, &self.model.readout, &mut rng);
                    }
                    *counts.entry(outcome).or_insert(0) += 1;
                }
                Ok(ExecResult::Counts(counts))
            }
        }
    }
}

/// A copy of `config` forced to statevector (no-shots) mode.
fn config_skip_shots(config: &ExecConfig) -> ExecConfig {
    let mut c = config.clone();
    c.shots = None;
    c
}

/// Apply the per-gate noise channel to qubit `q` of the MPS. Order matches the
/// statevector backend (depolarizing, Pauli, amplitude damping, phase damping).
fn apply_mps_channel<R: Rng>(
    mps: &mut Mps,
    model: &NoiseModel,
    q: usize,
    gate_qubits: &[usize],
    rng: &mut R,
) {
    // Depolarizing: with prob p, a uniformly chosen X/Y/Z (unitary unravelling).
    // A two-qubit gate's pair selects a per-pair rate when one is configured.
    let p = model.depolarizing.at_gate(q, gate_qubits);
    if p > 0.0 && rng.random::<f64>() < p {
        match (rng.random::<f64>() * 3.0) as u8 {
            0 => mps.apply_1q(q, &gates::x()),
            1 => mps.apply_1q(q, &gates::y()),
            _ => mps.apply_1q(q, &gates::z()),
        }
    }

    // Explicit Pauli channel.
    if let Some(pauli) = &model.pauli {
        let (px, py, pz) = (pauli.x.at(q), pauli.y.at(q), pauli.z.at(q));
        if px + py + pz > 0.0 {
            let r = rng.random::<f64>();
            if r < px {
                mps.apply_1q(q, &gates::x());
            } else if r < px + py {
                mps.apply_1q(q, &gates::y());
            } else if r < px + py + pz {
                mps.apply_1q(q, &gates::z());
            }
        }
    }

    // Amplitude damping: local quantum jump with p_jump = γ·P(qubit = 1).
    let gamma = model.amplitude_damping.at(q);
    if gamma > 0.0 {
        apply_amplitude_damping_mps(mps, q, gamma, rng);
    }

    // Phase damping λ: apply Z with probability λ/2 (unitary unravelling).
    let lambda = model.phase_damping.at(q);
    if lambda > 0.0 && rng.random::<f64>() < 0.5 * lambda {
        mps.apply_1q(q, &gates::z());
    }
}

/// Amplitude damping on qubit `q` of the MPS via the quantum-jump method.
///
/// `p_jump = γ·P(qubit = 1)`, with `P(1) = (1 − ⟨Z_q⟩)/2` read from the MPS.
/// On a jump, `E₁ ∝ |0⟩⟨1|` transfers the excitation to |0⟩; otherwise
/// `E₀ = diag(1, √(1−γ))` attenuates it. Either branch renormalises globally
/// (the local norm is wrong for an entangled site).
fn apply_amplitude_damping_mps<R: Rng>(mps: &mut Mps, q: usize, gamma: f64, rng: &mut R) {
    let norm_sq = mps_norm_sq(mps);
    if norm_sq <= 0.0 {
        return;
    }
    let z = site_z_numerator(mps, q) / norm_sq;
    let p1 = ((1.0 - z) / 2.0).clamp(0.0, 1.0);
    if p1 <= 0.0 {
        return;
    }
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    if rng.random::<f64>() < gamma * p1 {
        // Jump: E₁ = [[0, √γ], [0, 0]] (|1⟩ → √γ|0⟩); global renorm absorbs √γ.
        mps.apply_1q(q, &[zero, Complex64::new(gamma.sqrt(), 0.0), zero, zero]);
    } else {
        // No jump: E₀ = diag(1, √(1−γ)).
        mps.apply_1q(
            q,
            &[one, zero, zero, Complex64::new((1.0 - gamma).sqrt(), 0.0)],
        );
    }
    renormalize_mps(mps);
}

/// Scale the whole MPS to unit norm by rescaling one site tensor.
fn renormalize_mps(mps: &mut Mps) {
    let norm_sq = mps_norm_sq(mps);
    if norm_sq > 0.0 {
        let inv = 1.0 / norm_sq.sqrt();
        for a in mps.tensors[0].data.iter_mut() {
            *a *= inv;
        }
    }
}

/// Unnormalised `⟨ψ|Z_q|ψ⟩` via transfer-matrix contraction (Z contributes a
/// `(-1)^s` sign at the physical index of site `q`). Divide by `mps_norm_sq`
/// for the true expectation.
fn site_z_numerator(mps: &Mps, q: usize) -> f64 {
    let zero = Complex64::new(0.0, 0.0);
    let mut env = vec![Complex64::new(1.0, 0.0)];
    let mut env_dim = 1usize;
    for (site, t) in mps.tensors.iter().enumerate() {
        let bl = t.bond_left;
        let br = t.bond_right;
        let mut new_env = vec![zero; br * br];
        for l1 in 0..bl {
            for l2 in 0..bl {
                let e = env[l1 * env_dim + l2];
                if e.norm_sqr() < 1e-30 {
                    continue;
                }
                for s in 0..2 {
                    let sign = if site == q && s == 1 { -1.0 } else { 1.0 };
                    for r1 in 0..br {
                        let a1 = t.get(l1, s, r1).conj();
                        for r2 in 0..br {
                            new_env[r1 * br + r2] += e * a1 * t.get(l2, s, r2) * sign;
                        }
                    }
                }
            }
        }
        env = new_env;
        env_dim = br;
    }
    env[0].re
}

/// True when this run collapses mid-circuit measurements into the classical
/// register (each shot an independent stochastic trajectory keyed by the creg).
fn mps_collapses(circuit: &CircuitIR, config: &ExecConfig) -> bool {
    config.mid_circuit_mode == MidCircuitMode::Collapse && circuit.num_classical_bits > 0
}

/// Pack the classical register into a `u64` counts key. One definition for
/// every backend — see [`omega_core::executor::creg_to_u64`].
use omega_core::executor::creg_to_u64 as mps_creg_to_u64;

/// Flip each qubit of a sampled outcome with its per-qubit (possibly
/// asymmetric) readout-error probability.
fn flip_all_readout<R: Rng>(bits: u64, n: usize, readout: &ReadoutError, rng: &mut R) -> u64 {
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

/// True when the circuit contains a `Reset`, whose randomness must be redrawn
/// per shot. See [`apply_reset_mps`].
fn circuit_has_reset(circuit: &CircuitIR) -> bool {
    circuit
        .ops
        .iter()
        .any(|op| matches!(op.gate, GateKind::Reset))
}

/// Analytic (`shots = None`) expectation over a `Reset` circuit is not a
/// well-defined pure-state quantity — the channel leaves the register mixed and
/// one chain holds a single trajectory, so the answer would silently depend on
/// an RNG draw. Refuse instead. Mirrors the CPU statevector backend.
fn reject_reset_in_analytic_mode(circuit: &CircuitIR) -> Result<()> {
    if circuit_has_reset(circuit) {
        return Err(OmegaError::Unsupported(
            "mps: analytic expectation of a circuit containing Reset is ill-defined \
             (Reset produces a mixed state); run with shots so each shot is an independent \
             trajectory"
                .into(),
        ));
    }
    Ok(())
}

/// `rng` is consumed only by [`GateKind::Reset`], a stochastic channel rather
/// than a gate — see [`apply_reset_mps`].
fn apply_gate_mps(
    mps: &mut Mps,
    op: &GateOp,
    params: &ParameterBinding,
    rng: &mut impl Rng,
) -> Result<()> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<Result<Vec<_>>>()?;

    match &op.gate {
        // Single-qubit gates (no params)
        GateKind::H => mps.apply_1q(op.qubits[0].0 as usize, &gates::h()),
        GateKind::X => mps.apply_1q(op.qubits[0].0 as usize, &gates::x()),
        GateKind::Y => mps.apply_1q(op.qubits[0].0 as usize, &gates::y()),
        GateKind::Z => mps.apply_1q(op.qubits[0].0 as usize, &gates::z()),
        GateKind::S => mps.apply_1q(op.qubits[0].0 as usize, &gates::s()),
        GateKind::Sdg => mps.apply_1q(op.qubits[0].0 as usize, &gates::sdg()),
        GateKind::Sx => mps.apply_1q(op.qubits[0].0 as usize, &gates::sx()),
        GateKind::Sxdg => mps.apply_1q(op.qubits[0].0 as usize, &gates::sxdg()),
        GateKind::T => mps.apply_1q(op.qubits[0].0 as usize, &gates::t()),
        GateKind::Tdg => mps.apply_1q(op.qubits[0].0 as usize, &gates::tdg()),
        GateKind::Id => {}

        // Single-qubit gates (parametric)
        GateKind::Rx => mps.apply_1q(op.qubits[0].0 as usize, &gates::rx(resolved[0])),
        GateKind::Ry => mps.apply_1q(op.qubits[0].0 as usize, &gates::ry(resolved[0])),
        GateKind::Rz => mps.apply_1q(op.qubits[0].0 as usize, &gates::rz(resolved[0])),
        GateKind::U3 => mps.apply_1q(
            op.qubits[0].0 as usize,
            &gates::u3(resolved[0], resolved[1], resolved[2]),
        ),
        GateKind::U2 => mps.apply_1q(
            op.qubits[0].0 as usize,
            &gates::u2(resolved[0], resolved[1]),
        ),
        GateKind::U1 => mps.apply_1q(op.qubits[0].0 as usize, &gates::u1(resolved[0])),

        // Two-qubit gates
        GateKind::CX => mps.apply_2q_distant(
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::cx(),
        ),
        GateKind::CY => mps.apply_2q_distant(
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::cy(),
        ),
        GateKind::CZ => mps.apply_2q_distant(
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::cz(),
        ),
        GateKind::Swap => mps.apply_2q_distant(
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::swap(),
        ),
        GateKind::CRz => mps.apply_2q_distant(
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::crz(resolved[0]),
        ),
        GateKind::CU3 => mps.apply_2q_distant(
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::cu3(resolved[0], resolved[1], resolved[2]),
        ),
        GateKind::Rbs => mps.apply_2q_distant(
            op.qubits[0].0 as usize,
            op.qubits[1].0 as usize,
            &gates::rbs(resolved[0]),
        ),

        // Three-qubit gates: decompose into 1Q + 2Q operations
        GateKind::CCX => {
            apply_toffoli_decomposed(mps, op.qubits[0].0, op.qubits[1].0, op.qubits[2].0);
        }
        GateKind::CSwap => {
            apply_fredkin_decomposed(mps, op.qubits[0].0, op.qubits[1].0, op.qubits[2].0);
        }

        GateKind::Reset => {
            apply_reset_mps(mps, op.qubits[0].0 as usize, rng);
        }

        GateKind::Measure | GateKind::Barrier => {}

        _ => {
            return Err(OmegaError::Unsupported(format!(
                "gate {:?} not supported in MPS backend",
                op.gate
            )));
        }
    }

    Ok(())
}

/// Reset qubit `q` to |0⟩ in the MPS representation — the reset **channel**
/// ρ → |0⟩⟨0|_q ⊗ Tr_q(ρ).
///
/// Sample → project → flip, matching the CPU statevector's `apply_reset` and
/// Qiskit Aer's matrix_product_state method: `measure_site` draws the outcome
/// from the correct environment-aware marginal and collapses the chain, then X
/// returns the site to |0⟩. One call = one trajectory, so the caller must
/// re-evolve per shot (see the `circuit_has_reset` branch in `execute`).
///
/// **Do not "fold" the amplitudes.** The previous implementation applied the
/// non-unitary map `[[1,1],[0,0]]` and renormalised globally, which is a
/// *coherent* fold, not a reset: on Bell + `reset q0` it left q1 in |+⟩
/// (⟨X₁⟩ = +1 instead of 0), and on a qubit in |−⟩ the amplitudes cancelled to
/// zero so the reset qubit sampled as |1⟩ every shot. Pinned by
/// `tests/reset_channel.rs`.
fn apply_reset_mps(mps: &mut Mps, q: usize, rng: &mut impl Rng) {
    if mps.measure_site(q, rng) == 1 {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        mps.apply_1q(q, &[zero, one, one, zero]); // X
    }
}

/// Compute ⟨ψ|ψ⟩ for the MPS via transfer-matrix contraction.
fn mps_norm_sq(mps: &Mps) -> f64 {
    // Transfer matrix contraction from left to right.
    // env[l1, l2] starts as 1x1 identity, then for each site q:
    //   env'[r1, r2] = Σ_{l1,l2,s} env[l1,l2] * conj(A[l1,s,r1]) * A[l2,s,r2]
    let mut env = vec![Complex64::new(1.0, 0.0)]; // 1x1 = [[1]]
    let mut env_dim = 1usize;

    for t in &mps.tensors {
        let bl = t.bond_left;
        let br = t.bond_right;
        let mut new_env = vec![Complex64::new(0.0, 0.0); br * br];
        for l1 in 0..bl {
            for l2 in 0..bl {
                let e = env[l1 * env_dim + l2];
                if e.norm_sqr() < 1e-30 {
                    continue;
                }
                for s in 0..2 {
                    for r1 in 0..br {
                        let a1 = t.get(l1, s, r1).conj();
                        for r2 in 0..br {
                            new_env[r1 * br + r2] += e * a1 * t.get(l2, s, r2);
                        }
                    }
                }
            }
        }
        env = new_env;
        env_dim = br;
    }

    env[0].re
}

/// Toffoli (CCX) decomposed into 1Q and 2Q gates.
/// Standard decomposition: 6 CNOTs + H + T/Tdg gates.
fn apply_toffoli_decomposed(mps: &mut Mps, c0: u32, c1: u32, tgt: u32) {
    let c0 = c0 as usize;
    let c1 = c1 as usize;
    let tgt = tgt as usize;

    mps.apply_1q(tgt, &gates::h());
    mps.apply_2q_distant(c1, tgt, &gates::cx());
    mps.apply_1q(tgt, &gates::tdg());
    mps.apply_2q_distant(c0, tgt, &gates::cx());
    mps.apply_1q(tgt, &gates::t());
    mps.apply_2q_distant(c1, tgt, &gates::cx());
    mps.apply_1q(tgt, &gates::tdg());
    mps.apply_2q_distant(c0, tgt, &gates::cx());
    mps.apply_1q(c1, &gates::t());
    mps.apply_1q(tgt, &gates::t());
    mps.apply_1q(tgt, &gates::h());
    mps.apply_2q_distant(c0, c1, &gates::cx());
    mps.apply_1q(c0, &gates::t());
    mps.apply_1q(c1, &gates::tdg());
    mps.apply_2q_distant(c0, c1, &gates::cx());
}

/// Fredkin (CSwap) decomposed: CX + Toffoli + CX.
fn apply_fredkin_decomposed(mps: &mut Mps, ctrl: u32, a: u32, b: u32) {
    let a_u = a as usize;
    let b_u = b as usize;

    mps.apply_2q_distant(b_u, a_u, &gates::cx());
    apply_toffoli_decomposed(mps, ctrl, a, b);
    mps.apply_2q_distant(b_u, a_u, &gates::cx());
}

/// Compute <psi|P|psi> for a single Pauli string.
fn expectation_pauli(sv: &[Complex64], num_qubits: u32, paulis: &[(u32, PauliOp)]) -> f64 {
    let n = num_qubits as usize;
    let dim = 1usize << n;

    // Apply Pauli string to |psi>, compute <psi|P|psi>
    let mut result = Complex64::new(0.0, 0.0);
    let i_unit = Complex64::new(0.0, 1.0);

    for basis in 0..dim {
        let mut phase = Complex64::new(1.0, 0.0);
        let mut target_basis = basis;

        for (q, op) in paulis {
            let q = *q as usize;
            let bit = (target_basis >> q) & 1;
            match op {
                PauliOp::I => {}
                PauliOp::X => {
                    target_basis ^= 1 << q;
                }
                PauliOp::Y => {
                    target_basis ^= 1 << q;
                    // Y|0> = i|1>, Y|1> = -i|0>
                    if bit == 0 {
                        phase *= i_unit;
                    } else {
                        phase *= -i_unit;
                    }
                }
                PauliOp::Z => {
                    if bit == 1 {
                        phase *= Complex64::new(-1.0, 0.0);
                    }
                }
            }
        }

        // P|basis> = phase * |target>, so the contribution to
        // <psi|P|psi> is conj(sv[target]) * phase * sv[basis].
        // (Pairing the basis-derived phase with conj(sv[basis]) instead
        // negates every odd-Y Pauli string — mirrors the statevector
        // backend's fix; pinned by tests below.)
        result += sv[target_basis].conj() * phase * sv[basis];
    }

    result.re
}

#[cfg(test)]
mod tests {
    use super::*;
    use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
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
    fn test_bell_state_matches_statevector() {
        let mut circuit = empty_circuit(2);
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::CX, &[0, 1]));

        let backend = MpsBackend::new(64);
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };

        let result = backend.execute(&circuit, &params, &config).unwrap();
        let sv = result.statevector();

        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - expected).abs() < 1e-8);
        assert!(sv[1].norm() < 1e-8);
        assert!(sv[2].norm() < 1e-8);
        assert!((sv[3].re - expected).abs() < 1e-8);
    }

    #[test]
    fn test_ghz3_sampling() {
        let mut circuit = empty_circuit(3);
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::CX, &[0, 1]));
        circuit.ops.push(make_op(GateKind::CX, &[1, 2]));

        let backend = MpsBackend::new(64);
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: Some(1000),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Skip,
        };

        let result = backend.execute(&circuit, &params, &config).unwrap();
        let counts = result.counts();

        // GHZ state: only |000> and |111> should appear
        for bitstring in counts.keys() {
            assert!(
                *bitstring == 0 || *bitstring == 7,
                "unexpected bitstring: {bitstring}"
            );
        }
        assert!(counts.contains_key(&0));
        assert!(counts.contains_key(&7));
    }

    #[test]
    fn test_toffoli_decomposition() {
        // |110> --CCX--> |111>
        let mut circuit = empty_circuit(3);
        circuit.ops.push(make_op(GateKind::X, &[0]));
        circuit.ops.push(make_op(GateKind::X, &[1]));
        circuit.ops.push(make_op(GateKind::CCX, &[0, 1, 2]));

        let backend = MpsBackend::new(64);
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };

        let result = backend.execute(&circuit, &params, &config).unwrap();
        let sv = result.statevector();

        // Should be |111> = index 7
        assert!(
            (sv[7].norm() - 1.0).abs() < 1e-6,
            "expected |111>, got sv[7]={}",
            sv[7]
        );
        for i in 0..8 {
            if i != 7 {
                assert!(sv[i].norm() < 1e-6, "sv[{}] = {} should be ~0", i, sv[i]);
            }
        }
    }

    #[test]
    fn test_expectation_z() {
        // |1> state: <Z> should be -1
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::X, &[0]));

        let backend = MpsBackend::new(64);
        let params = ParameterBinding::new();
        let observable = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };

        let val = backend.expectation(&circuit, &params, &observable).unwrap();
        assert!((val + 1.0).abs() < 1e-8, "expected -1, got {val}");
    }

    #[test]
    fn test_reset_from_one() {
        // X|0⟩ = |1⟩, then reset → |0⟩
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::X, &[0]));
        circuit.ops.push(make_op(GateKind::Reset, &[0]));

        let backend = MpsBackend::new(64);
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend.execute(&circuit, &params, &config).unwrap();
        let sv = result.statevector();
        assert!(
            (sv[0].re - 1.0).abs() < 1e-8,
            "should be |0⟩ after reset, got {}",
            sv[0]
        );
        assert!(sv[1].norm() < 1e-8);
    }

    #[test]
    fn test_reset_from_superposition() {
        // H|0⟩ = |+⟩, then reset → |0⟩
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::Reset, &[0]));

        let backend = MpsBackend::new(64);
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend.execute(&circuit, &params, &config).unwrap();
        let sv = result.statevector();
        assert!((sv[0].re - 1.0).abs() < 1e-8, "should be |0⟩ after reset");
        assert!(sv[1].norm() < 1e-8);
    }

    #[test]
    fn test_reset_partial_bell() {
        // Bell state, reset qubit 0 → qubit 0 in |0⟩
        let mut circuit = empty_circuit(2);
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::CX, &[0, 1]));
        circuit.ops.push(make_op(GateKind::Reset, &[0]));

        let backend = MpsBackend::new(64);
        let params = ParameterBinding::new();
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend.execute(&circuit, &params, &config).unwrap();
        let sv = result.statevector();
        // qubit 0 (bit 0) should be 0 → indices with bit 0 set should be zero
        assert!(sv[1].norm() < 1e-8, "|01⟩ should be zero");
        assert!(sv[3].norm() < 1e-8, "|11⟩ should be zero");
        let norm_sq: f64 = sv.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm_sq - 1.0).abs() < 1e-8, "state should be normalised");
    }

    #[test]
    fn test_mps_midcircuit_measure_zero() {
        // |0⟩ → measure → should give |0⟩
        let mut circuit = empty_circuit(1);
        circuit.num_classical_bits = 1;
        circuit.ops.push(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: Some(0),
            condition: None,
        });

        let backend = MpsBackend::new(64);
        let config = ExecConfig {
            shots: None,
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Collapse,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &config)
            .unwrap();
        let sv = result.statevector();
        assert!(
            (sv[0].re - 1.0).abs() < 1e-8,
            "should be |0⟩ after measuring |0⟩"
        );
    }

    #[test]
    fn test_mps_midcircuit_measure_bell() {
        // Bell state → measure q0 → q1 collapses correspondingly
        let mut circuit = empty_circuit(2);
        circuit.num_classical_bits = 1;
        circuit.ops.push(make_op(GateKind::H, &[0]));
        circuit.ops.push(make_op(GateKind::CX, &[0, 1]));
        circuit.ops.push(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: Some(0),
            condition: None,
        });

        let backend = MpsBackend::new(64);

        for seed in 0..50 {
            let config = ExecConfig {
                shots: None,
                seed: Some(seed),
                mid_circuit_mode: MidCircuitMode::Collapse,
            };
            let result = backend
                .execute(&circuit, &ParameterBinding::new(), &config)
                .unwrap();
            let sv = result.statevector();
            // Should be either |00⟩ or |11⟩
            assert!(sv[1].norm_sqr() < 1e-8, "seed {seed}: |01⟩ should be 0");
            assert!(sv[2].norm_sqr() < 1e-8, "seed {seed}: |10⟩ should be 0");
        }
    }

    #[test]
    fn test_expectation_multi_matches_per_observable_loop() {
        // Pin that the override returns the same numbers as the
        // default (per-observable loop) implementation. GHZ state on
        // 3 qubits gives a non-trivial mix of correlated Pauli
        // expectations: ⟨Z₀⟩ = 0, ⟨Z₀Z₁⟩ = 1, ⟨Z₀Z₂⟩ = 1, plus a
        // weighted-sum observable to exercise the inner coeff loop.
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

        let backend = MpsBackend::new(64);
        let params = ParameterBinding::new();
        let multi = backend
            .expectation_multi(&circuit, &params, &observables)
            .unwrap();
        assert_eq!(multi.len(), observables.len());
        for (obs, m) in observables.iter().zip(multi.iter()) {
            let single = backend.expectation(&circuit, &params, obs).unwrap();
            assert!(
                (m - single).abs() < 1e-10,
                "expectation_multi disagreed with expectation: {m} vs {single}"
            );
        }
    }

    // ----- noise (trajectory) parity vs analytic density-matrix values -----

    use omega_core::noise::{Depolarizing, NoiseModel, Rate};

    fn p1_of(counts: &HashMap<u64, u32>, shots: u32) -> f64 {
        *counts.get(&1).unwrap_or(&0) as f64 / shots as f64
    }

    #[test]
    fn noise_mps_amplitude_damping_matches_analytic() {
        // X|0⟩ = |1⟩ then amplitude damping γ: P(1) = 1 − γ.
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::X, &[0]));
        let gamma = 0.5_f64;
        let backend = NoisyMpsBackend::with_model(
            64,
            NoiseModel {
                amplitude_damping: gamma.into(),
                ..Default::default()
            },
        );
        let cfg = ExecConfig {
            shots: Some(20000),
            seed: Some(7),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let p1 = p1_of(result.counts(), 20000);
        assert!((p1 - (1.0 - gamma)).abs() < 0.02, "MPS ampdamp P(1)={p1}");
    }

    #[test]
    fn noise_mps_depolarizing_matches_analytic() {
        // X|0⟩ = |1⟩ then depolarizing p: P(1) = 1 − 2p/3.
        let mut circuit = empty_circuit(1);
        circuit.ops.push(make_op(GateKind::X, &[0]));
        let p = 0.3_f64;
        let backend = NoisyMpsBackend::with_model(
            64,
            NoiseModel {
                depolarizing: Depolarizing::uniform(p),
                ..Default::default()
            },
        );
        let cfg = ExecConfig {
            shots: Some(20000),
            seed: Some(11),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let p1 = p1_of(result.counts(), 20000);
        assert!(
            (p1 - (1.0 - 2.0 * p / 3.0)).abs() < 0.02,
            "MPS depol P(1)={p1}"
        );
    }

    #[test]
    fn noise_mps_per_qubit_amplitude_damping() {
        // Two qubits both |1⟩; only q1 damps (γ=0.8): P(q0=1)≈1, P(q1=1)≈0.2.
        let mut circuit = empty_circuit(2);
        circuit.ops.push(make_op(GateKind::X, &[0]));
        circuit.ops.push(make_op(GateKind::X, &[1]));
        let backend = NoisyMpsBackend::with_model(
            64,
            NoiseModel {
                amplitude_damping: Rate::PerQubit(vec![0.0, 0.8]),
                ..Default::default()
            },
        );
        let cfg = ExecConfig {
            shots: Some(20000),
            seed: Some(5),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let counts = result.counts();
        let shots = 20000.0_f64;
        let mut q0_one = 0.0_f64;
        let mut q1_one = 0.0_f64;
        for (bits, c) in counts.iter() {
            if bits & 1 != 0 {
                q0_one += *c as f64;
            }
            if bits & 2 != 0 {
                q1_one += *c as f64;
            }
        }
        assert!(
            (q0_one / shots - 1.0).abs() < 0.01,
            "P(q0=1)={}",
            q0_one / shots
        );
        assert!(
            (q1_one / shots - 0.2).abs() < 0.02,
            "P(q1=1)={}",
            q1_one / shots
        );
    }

    #[test]
    fn test_expectation_multi_empty_returns_empty() {
        // Empty observable list short-circuits before the heavy MPS
        // contract — pin both that no error fires and that no work
        // happens (an erroneously eager `execute` would still succeed
        // here, but the `Vec::new()` shape gates against an empty
        // result vector with bogus zeros).
        let circuit = empty_circuit(2);
        let backend = MpsBackend::new(64);
        let out = backend
            .expectation_multi(&circuit, &ParameterBinding::new(), &[])
            .unwrap();
        assert!(out.is_empty());
    }

    // --- deep-circuit unitarity + truncation-certificate regression ---
    //
    // The normal-equations SVD kernel produced non-unitary splits on deep
    // circuits (state norm → 21.7 at χ=64 where truncation is impossible). All
    // pre-existing MPS tests are shallow, so nothing caught it. These exercise
    // the truncation-free regime (χ = 2^(n/2) ⇒ exact) and the truncating one.

    /// SplitMix64 → deterministic angles with no rng dependency.
    fn brick_angle(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let z = z ^ (z >> 31);
        (z as f64) / (u64::MAX as f64) * std::f64::consts::TAU
    }

    /// Random-angle brickwork: `layers` of (RY on every qubit, then a CX ladder
    /// on alternating even/odd bonds) — RY + adjacent CX only, exactly the
    /// report's stress circuit. Deterministic in `seed`.
    fn brickwork(n: u32, layers: usize, seed: u64) -> CircuitIR {
        let mut c = empty_circuit(n);
        let mut st = seed;
        for layer in 0..layers {
            for q in 0..n {
                let mut op = make_op(GateKind::Ry, &[q]);
                op.params = smallvec![ParamExpr::Concrete(brick_angle(&mut st))];
                c.ops.push(op);
            }
            let start = (layer % 2) as u32;
            let mut q = start;
            while q + 1 < n {
                c.ops.push(make_op(GateKind::CX, &[q, q + 1]));
                q += 2;
            }
        }
        c
    }

    fn statevector_of(circuit: &CircuitIR, chi: usize) -> Vec<Complex64> {
        let backend = MpsBackend::new(chi);
        let result = backend
            .execute(
                circuit,
                &ParameterBinding::new(),
                &ExecConfig {
                    shots: None,
                    seed: None,
                    mid_circuit_mode: MidCircuitMode::Skip,
                },
            )
            .unwrap();
        result.statevector().to_vec()
    }

    /// Apply a RY+CX brickwork IR to a fresh Mps and return it — so the test
    /// can inspect the RAW (un-normalized) state, unlike execute→to_statevector.
    fn brickwork_mps(circuit: &CircuitIR, chi: usize) -> crate::mps::Mps {
        let mut mps = crate::mps::Mps::zero_state(circuit.num_qubits as usize, chi);
        for op in &circuit.ops {
            match &op.gate {
                GateKind::Ry => {
                    let theta = match &op.params[0] {
                        ParamExpr::Concrete(v) => *v,
                        _ => panic!("expected concrete angle"),
                    };
                    mps.apply_1q(op.qubits[0].0 as usize, &crate::gates::ry(theta));
                }
                GateKind::CX => {
                    mps.apply_2q_distant(
                        op.qubits[0].0 as usize,
                        op.qubits[1].0 as usize,
                        &crate::gates::cx(),
                    );
                }
                _ => panic!("brickwork_mps only handles RY + CX"),
            }
        }
        mps
    }

    #[test]
    fn deep_brickwork_is_unitary_at_exact_bond_dimension() {
        // n=8, χ=16=2^(n/2): every Schmidt rank fits, so truncation is
        // impossible and the state MUST stay exactly normalized through depth.
        // Check the RAW norm ⟨ψ|ψ⟩ from the tensors — NOT to_statevector, which
        // divides by ‖ψ‖ and so reads 1 even for the old non-unitary kernel
        // (norm 21.7 → 21.7/21.7 = 1). The raw norm is what the bug corrupted.
        // 60 layers ≈ 900 ops.
        let circuit = brickwork(8, 60, 0xBEEF);
        let raw_norm_sq = brickwork_mps(&circuit, 16).norm_sqr();
        assert!(
            (raw_norm_sq - 1.0).abs() < 1e-10,
            "raw ‖ψ‖² = {raw_norm_sq} (expected exactly 1 at exact χ; a non-unitary split drifts it)"
        );
    }

    /// Independent dense reference simulator (RY + CX only) — deliberately not
    /// another MPS, so the parity check has a genuinely separate ground truth.
    /// Qubit q is bit q, LSB-first (matches `to_statevector`).
    fn dense_reference(circuit: &CircuitIR) -> Vec<Complex64> {
        let n = circuit.num_qubits as usize;
        let dim = 1usize << n;
        let mut sv = vec![Complex64::new(0.0, 0.0); dim];
        sv[0] = Complex64::new(1.0, 0.0);
        for op in &circuit.ops {
            match &op.gate {
                GateKind::Ry => {
                    let theta = match &op.params[0] {
                        ParamExpr::Concrete(v) => *v,
                        _ => panic!("reference expects concrete angles"),
                    };
                    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
                    let q = op.qubits[0].0 as usize;
                    let bit = 1usize << q;
                    let mut out = sv.clone();
                    for i in 0..dim {
                        if i & bit == 0 {
                            let a = sv[i];
                            let b = sv[i | bit];
                            out[i] = c * a - s * b;
                            out[i | bit] = s * a + c * b;
                        }
                    }
                    sv = out;
                }
                GateKind::CX => {
                    let (ctrl, tgt) = (op.qubits[0].0 as usize, op.qubits[1].0 as usize);
                    let (cb, tb) = (1usize << ctrl, 1usize << tgt);
                    let mut out = sv.clone();
                    for i in 0..dim {
                        if i & cb != 0 {
                            out[i] = sv[i ^ tb];
                        }
                    }
                    sv = out;
                }
                _ => panic!("reference only implements RY + CX"),
            }
        }
        sv
    }

    #[test]
    fn deep_brickwork_matches_dense_reference_at_exact_bond_dimension() {
        // χ=8=2^(6/2) is exact for n=6, so mps must match an INDEPENDENT dense
        // simulator to ~1e-9 across 80 layers.
        let circuit = brickwork(6, 80, 0x1234);
        let reference = dense_reference(&circuit);
        let got = statevector_of(&circuit, 8);
        assert_eq!(got.len(), reference.len());
        let worst = got
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-9, "max amplitude Δ vs dense reference = {worst}");
    }

    #[test]
    fn under_provisioned_bond_reports_discarded_weight() {
        // Same deep circuit, χ=2 forces heavy truncation: discarded_weight must
        // be clearly positive, and readout normalization keeps the returned
        // statevector at ‖ψ‖² = 1 even though the raw truncated state is lossy.
        let circuit = brickwork(8, 60, 0xBEEF);
        // This test EXISTS to under-provision the bond, so it must opt into
        // the approximation explicitly — which is the point of the ceiling:
        // an approximate MPS result is legitimate when asked for, and a defect
        // when arrived at silently.
        let backend = MpsBackend::new(2).with_max_discarded_weight(f64::INFINITY);
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
        let stats = backend.last_run_stats();
        assert!(
            stats.discarded_weight > 1e-3,
            "expected real truncation at χ=2, got discarded_weight {}",
            stats.discarded_weight
        );
        assert!(stats.max_bond_reached <= 2);
        let norm_sq: f64 = sv.iter().map(|z| z.norm_sqr()).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-9,
            "readout normalization should keep ‖ψ‖² = 1, got {norm_sq}"
        );
    }

    #[test]
    fn exact_bond_reports_negligible_discarded_weight() {
        let circuit = brickwork(8, 60, 0xBEEF);
        let backend = MpsBackend::new(16); // 2^(8/2), exact
        backend
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
        let stats = backend.last_run_stats();
        assert!(
            stats.discarded_weight < 1e-12,
            "exact χ must not truncate, got {}",
            stats.discarded_weight
        );
    }

    #[test]
    fn adaptive_bond_matches_exact_within_its_tolerance_and_uses_smaller_bond() {
        // A tight adaptive tolerance under a generous ceiling must reproduce the
        // exact statevector to ~its own ε, while keeping the bond at or below the
        // exact requirement — i.e. it grows only as far as the entanglement needs.
        let circuit = brickwork(8, 60, 0xBEEF);
        let exact = statevector_of(&circuit, 16); // 2^(8/2), exact reference
        // Adaptive mode truncates by design; the assertion below is that it
        // stays within its stated tolerance, so the ceiling is lifted here and
        // the tolerance does the gating.
        let backend = MpsBackend::new(16)
            .with_adaptive(1e-8)
            .with_max_discarded_weight(f64::INFINITY);
        let cfg = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let got = result.statevector();
        let worst = got
            .iter()
            .zip(&exact)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-6, "adaptive vs exact max Δ = {worst}");
        let stats = backend.last_run_stats();
        assert!(stats.max_bond_reached <= 16, "must respect the ceiling");
        // Coarsening ε discards at least as much weight (monotonic) — this deep
        // brickwork scrambles to a near-flat Schmidt spectrum, so even a coarse
        // ε may still fill the bond; the honest invariant is monotone discard.
        // Same reason: a coarse ε is chosen here precisely to discard a lot.
        let coarse = MpsBackend::new(16)
            .with_adaptive(1e-1)
            .with_max_discarded_weight(f64::INFINITY);
        coarse
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let coarse_stats = coarse.last_run_stats();
        assert!(coarse_stats.max_bond_reached <= 16, "ceiling respected");
        assert!(
            coarse_stats.discarded_weight >= stats.discarded_weight,
            "coarser ε must discard ≥ finer ε ({} vs {})",
            coarse_stats.discarded_weight,
            stats.discarded_weight
        );
    }

    #[test]
    fn adaptive_keeps_a_small_bond_on_low_entanglement() {
        // A shallow circuit generates little entanglement, so adaptive mode
        // under a generous ceiling must keep the bond far below it — the point
        // of "grow only as the state needs".
        let circuit = brickwork(8, 3, 0x77);
        let cfg = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let adaptive = MpsBackend::new(64).with_adaptive(1e-8);
        adaptive
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let stats = adaptive.last_run_stats();
        assert!(
            stats.max_bond_reached < 64,
            "adaptive should not fill the ceiling on a shallow circuit (bond {})",
            stats.max_bond_reached
        );
        // And it stays accurate: matches the fixed-χ=64 (exact) statevector.
        let exact = statevector_of(&circuit, 64);
        let result = adaptive
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .unwrap();
        let got = result.statevector();
        let worst = got
            .iter()
            .zip(&exact)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-6, "adaptive shallow vs exact Δ = {worst}");
    }
}
