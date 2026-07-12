//! MPS (Matrix Product State) backend for circuit simulation.
//!
//! Uses tensor network contraction with truncated SVD to efficiently
//! simulate circuits with limited entanglement (low bond dimension).

use std::collections::HashMap;

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
}

impl MpsBackend {
    pub fn new(max_bond_dim: usize) -> Self {
        Self {
            max_bond_dim,
            svd_fn: truncated_svd_flat,
            contract_fn: None,
        }
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

        let n = circuit.num_qubits as usize;
        let mut mps = Mps::zero_state(n, self.max_bond_dim);
        mps.set_svd_fn(self.svd_fn);
        if let Some(cf) = self.contract_fn {
            mps.set_contract_fn(cf);
        }
        let mut classical_bits = vec![0u8; circuit.num_classical_bits as usize];
        let mut rng: StdRng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng::<StdRng>(),
        };

        for op in &circuit.ops {
            if !op.condition_satisfied(&classical_bits) {
                continue;
            }

            match &op.gate {
                GateKind::Measure => {
                    if config.mid_circuit_mode == MidCircuitMode::Collapse {
                        let q = op.qubits[0].0 as usize;
                        let outcome = mps.measure_site(q, &mut rng);
                        if let Some(cbit) = op.classical_bit {
                            if (cbit as usize) < classical_bits.len() {
                                classical_bits[cbit as usize] = outcome;
                            }
                        }
                    }
                }
                GateKind::Barrier => continue,
                _ => {
                    apply_gate_mps(&mut mps, op, params)?;
                }
            }
        }

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
                    apply_gate_mps(&mut mps, op, params)?;
                    if self.model.has_gate_channel() {
                        let arity = op.qubits.len();
                        for qb in &op.qubits {
                            apply_mps_channel(&mut mps, &self.model, qb.0 as usize, arity, rng);
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
                let (mps, _cbits) = self.evolve(circuit, params, &config_skip_shots(config), &mut rng)?;
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
    arity: usize,
    rng: &mut R,
) {
    // Depolarizing: with prob p, a uniformly chosen X/Y/Z (unitary unravelling).
    let p = model.depolarizing.at(q, arity);
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
        mps.apply_1q(q, &[one, zero, zero, Complex64::new((1.0 - gamma).sqrt(), 0.0)]);
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

/// Pack the classical register (bit `i` = cbit `i`) into a `u64` counts key,
/// matching the statevector backend's collapse-mode encoding.
fn mps_creg_to_u64(classical_bits: &[u8]) -> u64 {
    let mut bits = 0u64;
    for (i, b) in classical_bits.iter().enumerate() {
        if i >= 64 {
            break;
        }
        bits |= ((*b as u64) & 1) << i;
    }
    bits
}

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

fn apply_gate_mps(mps: &mut Mps, op: &GateOp, params: &ParameterBinding) -> Result<()> {
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

        // Three-qubit gates: decompose into 1Q + 2Q operations
        GateKind::CCX => {
            apply_toffoli_decomposed(mps, op.qubits[0].0, op.qubits[1].0, op.qubits[2].0);
        }
        GateKind::CSwap => {
            apply_fredkin_decomposed(mps, op.qubits[0].0, op.qubits[1].0, op.qubits[2].0);
        }

        GateKind::Reset => {
            apply_reset_mps(mps, op.qubits[0].0 as usize);
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

/// Reset qubit `q` to |0⟩ in the MPS representation.
///
/// Applies the non-unitary |0⟩⟨0| + |0⟩⟨1| map and renormalises via global norm.
fn apply_reset_mps(mps: &mut Mps, q: usize) {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    // Reset "gate": |0⟩⟨0| + |0⟩⟨1| = [[1, 1], [0, 0]]
    mps.apply_1q(q, &[one, one, zero, zero]);

    // Renormalise using the global norm (local norm is wrong for entangled states)
    let norm_sq = mps_norm_sq(mps);
    if norm_sq > 0.0 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for a in mps.tensors[q].data.iter_mut() {
            *a *= inv_norm;
        }
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
        let mut coeff = sv[basis].conj();
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
                        coeff *= i_unit;
                    } else {
                        coeff *= -i_unit;
                    }
                }
                PauliOp::Z => {
                    if bit == 1 {
                        coeff *= Complex64::new(-1.0, 0.0);
                    }
                }
            }
        }

        result += coeff * sv[target_basis];
    }

    result.re
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
        let result = backend.execute(&circuit, &ParameterBinding::new(), &cfg).unwrap();
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
        let result = backend.execute(&circuit, &ParameterBinding::new(), &cfg).unwrap();
        let p1 = p1_of(result.counts(), 20000);
        assert!((p1 - (1.0 - 2.0 * p / 3.0)).abs() < 0.02, "MPS depol P(1)={p1}");
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
        let result = backend.execute(&circuit, &ParameterBinding::new(), &cfg).unwrap();
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
        assert!((q0_one / shots - 1.0).abs() < 0.01, "P(q0=1)={}", q0_one / shots);
        assert!((q1_one / shots - 0.2).abs() < 0.02, "P(q1=1)={}", q1_one / shots);
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
}
