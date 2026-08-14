//! Photonics simulation backend.
//!
//! Executes photonic circuits (OPTICQASM) by:
//! 1. Building the unitary transfer matrix from ps/bs_rx components
//! 2. Computing output Fock-state distribution via SLOS

use std::collections::HashMap;

use omega_core::circuit::*;
use omega_core::error::{OmegaError, Result};
use omega_core::executor::*;
use omega_core::params::ParameterBinding;

use crate::components::{self, PhotonicOp};
use crate::slos;

/// Photonic circuit simulator backend.
///
/// Simulates discrete-variable photonic circuits using Fock-space evolution.
/// Supports phase shifters (ps) and beam splitters (bs_rx).
pub struct PhotonicsBackend {
    /// Default input Fock state. If None, uses |1,1,...,1,0,...,0> with
    /// photons in the first n modes where n = num_modes/2.
    pub default_input: Option<slos::FockState>,
}

impl Default for PhotonicsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PhotonicsBackend {
    pub fn new() -> Self {
        Self {
            default_input: None,
        }
    }

    /// Create with a specific input Fock state.
    pub fn with_input(input: slos::FockState) -> Self {
        Self {
            default_input: Some(input),
        }
    }

    /// Extract PhotonicOps from a CircuitIR.
    fn extract_ops(circuit: &CircuitIR, params: &ParameterBinding) -> Result<Vec<PhotonicOp>> {
        let mut ops = Vec::new();

        for gate_op in &circuit.ops {
            match &gate_op.gate {
                GateKind::PhaseShifter => {
                    let phi = params.resolve(&gate_op.params[0])?;
                    let mode = gate_op.qubits[0].0 as usize;
                    ops.push(PhotonicOp::PhaseShifter { mode, phi });
                }
                GateKind::BeamSplitterRx => {
                    let theta = params.resolve(&gate_op.params[0])?;
                    let phi = params.resolve(&gate_op.params[1])?;
                    let mode0 = gate_op.qubits[0].0 as usize;
                    let mode1 = gate_op.qubits[1].0 as usize;
                    ops.push(PhotonicOp::BeamSplitterRx {
                        mode0,
                        mode1,
                        theta,
                        phi,
                    });
                }
                GateKind::Barrier => continue,
                GateKind::Measure => continue,
                other => {
                    return Err(OmegaError::Unsupported(format!(
                        "gate {:?} not supported in photonics backend",
                        other
                    )));
                }
            }
        }

        Ok(ops)
    }

    /// Get the input Fock state, using default if not configured.
    fn get_input(&self, num_modes: usize) -> slos::FockState {
        if let Some(ref input) = self.default_input {
            return input.clone();
        }
        // Default: one photon in each of the first ceil(num_modes/2) modes
        let n_photons = num_modes.div_ceil(2);
        let mut input = vec![0u32; num_modes];
        for i in 0..n_photons.min(num_modes) {
            input[i] = 1;
        }
        input
    }
}

impl Backend for PhotonicsBackend {
    fn name(&self) -> &str {
        "photonics"
    }

    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> Result<ExecResult> {
        if circuit.circuit_type != CircuitType::Photonic {
            return Err(OmegaError::InvalidCircuit(
                "photonics backend requires a Photonic circuit".into(),
            ));
        }

        let num_modes = circuit.num_qubits as usize;
        let ops = Self::extract_ops(circuit, params)?;

        // Build the unitary transfer matrix
        let unitary = components::build_unitary(num_modes, &ops);

        // Get input Fock state
        let input = self.get_input(num_modes);

        // Run SLOS to get output distribution
        let distribution = slos::slos_full(&unitary, &input);

        match config.shots {
            None => {
                // Return probability distribution as "Probabilities" variant
                // Encode Fock states as bitstrings for compatibility with ExecResult
                // (this is a simplification — proper Fock state output would be better)
                Ok(ExecResult::Probabilities(
                    distribution.iter().map(|(_, p)| *p).collect(),
                ))
            }
            Some(shots) => {
                // Sample from the distribution
                let counts = sample_from_distribution(&distribution, shots, config.seed)?;
                // Photonic keys index Fock basis states, not qubits, and the
                // dense distribution is bounded well below 2^64 entries.
                Ok(ExecResult::counts_from_u64(counts, circuit.num_qubits))
            }
        }
    }

    /// Photonic adjoint gradient via cached parameter-shift.
    ///
    /// `BeamSplitterRx(θ)` and `PhaseShifter(φ)` are both 2π-periodic in
    /// their real parameters, so the exact parameter-shift rule applies:
    ///   grad_θ E = (E(θ + π/2) - E(θ - π/2)) / 2.
    ///
    /// The improvement over the core's fallback parameter-shift is that
    /// this implementation lives inside the backend and can share the
    /// input Fock state / observable evaluation harness across all
    /// parameter shifts, avoiding the per-parameter backend lookup cost.
    fn adjoint_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<Option<Vec<(omega_core::circuit::SymbolId, f64)>>> {
        if circuit.circuit_type != CircuitType::Photonic {
            return Ok(None);
        }
        // Collect the symbols this circuit depends on. We only support
        // directly-bound single-symbol parameters for now (no compound
        // expressions) — compound expressions fall back to the core's
        // parameter-shift machinery, which already uses
        // `ParamExpr::differentiate`.
        use omega_core::circuit::ParamExpr;
        let mut direct_symbols: Vec<omega_core::circuit::SymbolId> = Vec::new();
        for op in &circuit.ops {
            for param in &op.params {
                if let ParamExpr::Symbol(s) = param {
                    if !direct_symbols.contains(s) {
                        direct_symbols.push(*s);
                    }
                }
            }
        }
        if direct_symbols.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let shift = std::f64::consts::FRAC_PI_2;
        let mut out: Vec<(omega_core::circuit::SymbolId, f64)> =
            Vec::with_capacity(direct_symbols.len());

        for sym in direct_symbols {
            let base = params.get(sym).ok_or_else(|| OmegaError::UnboundSymbol {
                id: sym,
                name: format!("sym_{sym}"),
            })?;

            let mut p_plus = params.clone();
            p_plus.bind(sym, base + shift);
            let e_plus = self.expectation(circuit, &p_plus, observable)?;

            let mut p_minus = params.clone();
            p_minus.bind(sym, base - shift);
            let e_minus = self.expectation(circuit, &p_minus, observable)?;

            out.push((sym, (e_plus - e_minus) / 2.0));
        }

        Ok(Some(out))
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> Result<f64> {
        if circuit.circuit_type != CircuitType::Photonic {
            return Err(OmegaError::InvalidCircuit(
                "photonics backend requires a Photonic circuit".into(),
            ));
        }

        let num_modes = circuit.num_qubits as usize;
        let ops = Self::extract_ops(circuit, params)?;
        let unitary = components::build_unitary(num_modes, &ops);
        let input = self.get_input(num_modes);
        let distribution = slos::slos_full(&unitary, &input);

        Ok(evaluate_observable_against_distribution(
            &distribution,
            observable,
        ))
    }

    fn expectation_multi(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
    ) -> Result<Vec<f64>> {
        // Build the M×M Reck unitary and run SLOS once, then evaluate
        // each observable against the cached Fock-state distribution.
        // Both `build_unitary` and `slos::slos_full` are O(M³) /
        // O(2ⁿ·M²) respectively; the default trait impl loops
        // `expectation`, paying both costs per observable. Symmetric
        // to the CPU / MPS / Pauli / Metal overrides.
        if observables.is_empty() {
            return Ok(Vec::new());
        }
        if circuit.circuit_type != CircuitType::Photonic {
            return Err(OmegaError::InvalidCircuit(
                "photonics backend requires a Photonic circuit".into(),
            ));
        }
        let num_modes = circuit.num_qubits as usize;
        let ops = Self::extract_ops(circuit, params)?;
        let unitary = components::build_unitary(num_modes, &ops);
        let input = self.get_input(num_modes);
        let distribution = slos::slos_full(&unitary, &input);
        Ok(observables
            .iter()
            .map(|obs| evaluate_observable_against_distribution(&distribution, obs))
            .collect())
    }
}

/// Photonic Pauli-Z cost: each term `(coeff, [(mode, PauliOp::Z), …])`
/// contributes `coeff · Σ_state prob(state) · Π_term_qubits sign(state[mode])`,
/// where `sign = -1` when the mode has ≥ 1 photon, `+1` for vacuum.
/// `I` / `X` / `Y` are no-ops on the Fock-state probability semantics.
fn evaluate_observable_against_distribution(
    distribution: &[(slos::FockState, f64)],
    observable: &Observable,
) -> f64 {
    let mut total = 0.0;
    for (coeff, pauli_string) in &observable.terms {
        let mut term_val = 0.0;
        for (state, prob) in distribution {
            let mut sign = 1.0;
            for (qubit, pauli) in pauli_string {
                let mode = *qubit as usize;
                let n = if mode < state.len() { state[mode] } else { 0 };
                match pauli {
                    PauliOp::Z if n > 0 => {
                        sign *= -1.0;
                    }
                    _ => {}
                }
            }
            term_val += sign * prob;
        }
        total += coeff * term_val;
    }
    total
}

/// Largest mode count the u64 nibble encoding can represent without collision.
const MAX_ENCODABLE_MODES: usize = 16;
/// Largest photon count per mode the 4-bit nibble can hold.
const MAX_PHOTONS_PER_MODE: u32 = 15;

/// Sample from a Fock-state probability distribution.
/// Returns counts keyed by a hash of the Fock state (encoded as bitstring).
fn sample_from_distribution(
    distribution: &[(slos::FockState, f64)],
    shots: u32,
    seed: Option<u64>,
) -> Result<HashMap<u64, u32>> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => rand::make_rng::<StdRng>(),
    };

    // Build cumulative distribution
    let mut cumulative = Vec::with_capacity(distribution.len());
    let mut sum = 0.0;
    for (_, p) in distribution {
        sum += p;
        cumulative.push(sum);
    }
    if let Some(last) = cumulative.last_mut() {
        *last = 1.0;
    }

    // Encode Fock states as u64: pack photon counts into nibbles, so at most
    // 16 modes with at most 15 photons each.
    //
    // Both limits are checked UP FRONT and refused, because exceeding either
    // used to be silent: modes past 16 were dropped with a bare `break` and
    // counts above 15 were masked with `& 0xF`. Either way, distinct Fock
    // states collapse onto the same key and the caller receives a plausible
    // histogram that is quietly wrong — the worst failure available, and worse
    // than an error, since nothing downstream can detect it.
    //
    // This got sharper with polarization: a `pol` register uses TWO optical
    // modes per spatial mode, so the 16-mode ceiling is reached at 8 spatial
    // modes.
    for (i, state) in distribution.iter().enumerate() {
        if state.0.len() > MAX_ENCODABLE_MODES {
            return Err(OmegaError::InvalidCircuit(format!(
                "shots-mode photonic sampling encodes Fock states in a u64 \
                 (4 bits per mode), so it supports at most {MAX_ENCODABLE_MODES} \
                 modes; this circuit has {}. Note a polarized register uses two \
                 optical modes per spatial mode. Run without --shots for the \
                 analytic distribution, which has no such limit.",
                state.0.len()
            )));
        }
        if let Some(&n) = state.0.iter().find(|&&n| n > MAX_PHOTONS_PER_MODE) {
            return Err(OmegaError::InvalidCircuit(format!(
                "shots-mode photonic sampling allows at most \
                 {MAX_PHOTONS_PER_MODE} photons per mode (4-bit nibble \
                 encoding); configuration {i} has {n}"
            )));
        }
    }

    let encode_fock = |state: &slos::FockState| -> u64 {
        let mut val = 0u64;
        for (i, &n) in state.iter().enumerate() {
            val |= (n as u64) << (i * 4);
        }
        val
    };

    let mut counts = HashMap::new();
    for _ in 0..shots {
        let r: f64 = rng.random();
        let idx = cumulative
            .partition_point(|&c| c < r)
            .min(distribution.len() - 1);
        let key = encode_fock(&distribution[idx].0);
        *counts.entry(key).or_insert(0) += 1;
    }

    Ok(counts)
}

/// Format a Fock-state-encoded u64 back to a readable string.
pub fn decode_fock_string(encoded: u64, num_modes: usize) -> String {
    let mut modes = Vec::with_capacity(num_modes);
    for i in 0..num_modes {
        modes.push(((encoded >> (i * 4)) & 0xF) as u32);
    }
    format!(
        "|{}>",
        modes
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use omega_core::params::ParameterBinding;
    use smallvec::smallvec;

    fn make_simple_photonic_circuit() -> (CircuitIR, ParameterBinding) {
        // 2-mode circuit: PS(pi/4) on mode 0, then 50:50 BS
        let mut circuit = CircuitIR::new(2, CircuitType::Photonic);

        circuit.add_op(GateOp {
            gate: GateKind::PhaseShifter,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Concrete(std::f64::consts::FRAC_PI_4)],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::BeamSplitterRx,
            qubits: smallvec![Qubit(0), Qubit(1)],
            params: smallvec![
                ParamExpr::Concrete(std::f64::consts::FRAC_PI_4),
                ParamExpr::Concrete(0.0)
            ],
            classical_bit: None,
            condition: None,
        });

        (circuit, ParameterBinding::new())
    }

    #[test]
    fn test_photonics_backend_executes() {
        let (circuit, params) = make_simple_photonic_circuit();
        let backend = PhotonicsBackend::with_input(vec![1, 0]);

        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend.execute(&circuit, &params, &config).unwrap();
        match &result {
            ExecResult::Probabilities(probs) => {
                let total: f64 = probs.iter().sum();
                assert!((total - 1.0).abs() < 1e-8, "total prob = {}", total);
            }
            _ => panic!("expected Probabilities result"),
        }
    }

    #[test]
    fn test_photonics_hom_effect() {
        // |1,1> through 50:50 BS -> bunching
        let mut circuit = CircuitIR::new(2, CircuitType::Photonic);
        circuit.add_op(GateOp {
            gate: GateKind::BeamSplitterRx,
            qubits: smallvec![Qubit(0), Qubit(1)],
            params: smallvec![
                ParamExpr::Concrete(std::f64::consts::FRAC_PI_4),
                ParamExpr::Concrete(0.0)
            ],
            classical_bit: None,
            condition: None,
        });

        let backend = PhotonicsBackend::with_input(vec![1, 1]);
        let config = ExecConfig {
            shots: Some(1000),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &config)
            .unwrap();
        let counts = result.counts();

        // Encode |1,1> = 0x11 = 17
        // Photonic keys pack 4 bits of occupancy per mode; the width is the
        // mode count, as `counts_from_u64` records it.
        let w = counts.keys().next().map(|o| o.width()).unwrap_or(2);
        let key = |k: u64| omega_core::outcome::Outcome::from_u64(k, w);
        let count_11 = counts.get(&key(0x11)).copied().unwrap_or(0);
        assert_eq!(count_11, 0, "HOM: |1,1> should have 0 counts");

        // All shots should be |2,0> or |0,2>
        let count_20 = counts.get(&key(0x02)).copied().unwrap_or(0); // 2 in mode 0
        let count_02 = counts.get(&key(0x20)).copied().unwrap_or(0); // 2 in mode 1
        assert_eq!(count_20 + count_02, 1000);
    }

    #[test]
    fn test_parametric_photonic_circuit() {
        // Circuit with symbolic parameters
        let mut circuit = CircuitIR::new(2, CircuitType::Photonic);
        circuit.symbols.insert(0, "theta".to_string());
        circuit.symbols.insert(1, "phi".to_string());

        circuit.add_op(GateOp {
            gate: GateKind::BeamSplitterRx,
            qubits: smallvec![Qubit(0), Qubit(1)],
            params: smallvec![ParamExpr::Symbol(0), ParamExpr::Symbol(1)],
            classical_bit: None,
            condition: None,
        });

        let mut params = ParameterBinding::new();
        params.bind(0, std::f64::consts::FRAC_PI_4);
        params.bind(1, 0.0);

        let backend = PhotonicsBackend::with_input(vec![1, 0]);
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let result = backend.execute(&circuit, &params, &config).unwrap();
        match result {
            ExecResult::Probabilities(probs) => {
                let total: f64 = probs.iter().sum();
                assert!((total - 1.0).abs() < 1e-8);
            }
            _ => panic!("expected Probabilities"),
        }
    }

    #[test]
    fn test_rejects_gate_based_circuit() {
        let circuit = CircuitIR::new(2, CircuitType::GateBased);
        let backend = PhotonicsBackend::new();
        let result = backend.execute(&circuit, &ParameterBinding::new(), &ExecConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_fock_string() {
        assert_eq!(decode_fock_string(0x12, 2), "|2,1>");
        assert_eq!(decode_fock_string(0x00, 3), "|0,0,0>");
        assert_eq!(decode_fock_string(0x102, 3), "|2,0,1>");
    }

    #[test]
    fn test_photonic_gradient_parameter_shift() {
        // Parametric photonic circuit: PS(θ) on mode 0 then BS on modes 0,1
        // Verify parameter-shift gradient matches finite-difference
        use omega_core::gradient::{compute_gradient, GradMethod};

        let mut circuit = CircuitIR::new(2, CircuitType::Photonic);
        circuit.symbols.insert(0, "theta".to_string());
        circuit.add_op(GateOp {
            gate: GateKind::PhaseShifter,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::BeamSplitterRx,
            qubits: smallvec![Qubit(0), Qubit(1)],
            params: smallvec![
                ParamExpr::Concrete(std::f64::consts::FRAC_PI_4),
                ParamExpr::Concrete(0.0)
            ],
            classical_bit: None,
            condition: None,
        });

        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        };

        let backend = PhotonicsBackend::with_input(vec![1, 0]);
        let mut params = ParameterBinding::new();
        params.bind(0, 0.5);

        // Parameter-shift gradient
        let grad_ps = compute_gradient(
            &backend,
            &circuit,
            &params,
            &obs,
            &GradMethod::ParameterShift,
        )
        .unwrap();

        // Finite-difference gradient
        let grad_fd = compute_gradient(
            &backend,
            &circuit,
            &params,
            &obs,
            &GradMethod::FiniteDifference { epsilon: 1e-5 },
        )
        .unwrap();

        assert!(
            (grad_ps[0].1 - grad_fd[0].1).abs() < 1e-4,
            "PSR grad = {}, FD grad = {} — should match",
            grad_ps[0].1,
            grad_fd[0].1
        );
    }

    #[test]
    fn test_expectation_multi_matches_per_observable_loop() {
        // Pin that the override agrees with N independent `expectation`
        // calls on a parametric photonic circuit. The shared
        // `build_unitary` + `slos_full` is the heavy work; the
        // per-observable loop must see the same Fock distribution
        // both routes consume.
        let (circuit, params) = make_simple_photonic_circuit();
        let backend = PhotonicsBackend::with_input(vec![1, 0]);

        // Three observables exercising single-Z, two-mode Z, and a
        // weighted-sum + identity-offset shape on the photonic
        // distribution (Z = -1 sign per mode with ≥ 1 photon).
        let observables = vec![
            Observable {
                terms: vec![(1.0, vec![(0, PauliOp::Z)])],
            },
            Observable {
                terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
            },
            Observable {
                terms: vec![(0.5, vec![(1, PauliOp::Z)]), (0.25, vec![])],
            },
        ];

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
        // Empty observables short-circuits before the SLOS / unitary
        // build. The early-return also bypasses the photonic-circuit
        // type check, so an empty list won't fire `InvalidCircuit`
        // even on a non-photonic circuit — match the trait contract
        // (no observables to evaluate ⇒ trivially success).
        let (circuit, params) = make_simple_photonic_circuit();
        let backend = PhotonicsBackend::with_input(vec![1, 0]);
        let out = backend.expectation_multi(&circuit, &params, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_expectation_multi_rejects_gate_based_circuit() {
        // The non-empty path keeps the `Photonic`-only guard so a
        // mis-routed gate-based circuit gets a typed error rather
        // than building a degenerate Reck unitary.
        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        circuit.add_op(GateOp {
            gate: GateKind::PhaseShifter,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Concrete(0.5)],
            classical_bit: None,
            condition: None,
        });
        let backend = PhotonicsBackend::with_input(vec![1, 0]);
        let observables = vec![Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        }];
        match backend.expectation_multi(&circuit, &ParameterBinding::new(), &observables) {
            Err(OmegaError::InvalidCircuit(_)) => {}
            other => panic!("expected InvalidCircuit error, got {other:?}"),
        }
    }
}
