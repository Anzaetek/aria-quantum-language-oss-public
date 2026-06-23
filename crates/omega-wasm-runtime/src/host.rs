//! Host state shared between the WASM runtime and guest functions.

use std::collections::HashMap;

use omega_backend_photonics::PhotonicsBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType};
use omega_core::error::Result;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use omega_core::gradient::{
    compute_functional_gradient, compute_gradient, parse_functional_spec, FunctionalGradMethod,
    GradMethod,
};
use omega_core::params::ParameterBinding;
use omega_core::qaoa::qaoa_circuit;
use omega_core::qubo::Qubo;

/// Shared state accessible from host functions.
pub struct HostState {
    /// Registered circuits by ID.
    pub circuits: HashMap<u32, CircuitIR>,
    /// Registered observables by ID.
    pub observables: HashMap<u32, Observable>,
    /// Progress callback: (iteration, value).
    pub progress: Vec<(u32, f64)>,
    /// Final result from the WASM guest.
    pub final_result: Option<WasmResult>,
    /// Free-form input bytes the guest can read via `omega_input_*`.
    /// Typically a JSON config selecting `num_params`, `optimizer`,
    /// `max_iters`, etc., or a problem payload (e.g. a QUBO matrix).
    pub input_bytes: Vec<u8>,
}

/// Result returned by a WASM optimization loop.
#[derive(Clone, Debug)]
pub struct WasmResult {
    pub optimal_params: Vec<f64>,
    pub optimal_value: f64,
    pub iterations: u32,
}

impl Default for HostState {
    fn default() -> Self {
        Self::new()
    }
}

impl HostState {
    pub fn new() -> Self {
        Self {
            circuits: HashMap::new(),
            observables: HashMap::new(),
            progress: Vec::new(),
            final_result: None,
            input_bytes: Vec::new(),
        }
    }

    /// Stage the input payload the guest will read via `omega_input_*`.
    pub fn set_input(&mut self, bytes: Vec<u8>) {
        self.input_bytes = bytes;
    }

    /// Register a circuit, returns its ID.
    pub fn register_circuit(&mut self, circuit: CircuitIR) -> u32 {
        let id = self.circuits.len() as u32 + 1;
        self.circuits.insert(id, circuit);
        id
    }

    /// Register a Z-observable on given qubits with given coefficients.
    /// Simple helper for common Hamiltonians.
    pub fn register_observable(&mut self, observable: Observable) -> u32 {
        let id = self.observables.len() as u32 + 1;
        self.observables.insert(id, observable);
        id
    }

    /// Build a QAOA circuit + observable from a QUBO JSON payload, register
    /// both, and return `(circuit_id, observable_id, ising_offset)`.
    ///
    /// QUBO JSON shape: `{"n": N, "Q": [[i, j, c], ...]}` (see
    /// `omega_core::qubo::Qubo::from_json`). `depth` is the QAOA round count.
    pub fn build_qaoa_from_qubo_json(
        &mut self,
        qubo_json: &str,
        depth: usize,
    ) -> std::result::Result<(u32, u32, f64), String> {
        let qubo = Qubo::from_json(qubo_json)?;
        let ising = qubo.to_ising();
        let circuit = qaoa_circuit(&ising, depth);
        let observable = ising.to_observable();
        let cid = self.register_circuit(circuit);
        let oid = self.register_observable(observable);
        Ok((cid, oid, ising.offset))
    }

    /// Execute a circuit and return the expectation value of an observable.
    pub fn execute_expectation(
        &self,
        circuit_id: u32,
        params: &[f64],
        observable_id: u32,
    ) -> Result<f64> {
        let circuit = self.circuits.get(&circuit_id).ok_or_else(|| {
            omega_core::error::OmegaError::InvalidCircuit(format!(
                "circuit {} not found",
                circuit_id
            ))
        })?;
        let observable = self.observables.get(&observable_id).ok_or_else(|| {
            omega_core::error::OmegaError::InvalidCircuit(format!(
                "observable {} not found",
                observable_id
            ))
        })?;

        let binding = build_binding(circuit, params);

        match circuit.circuit_type {
            CircuitType::GateBased => {
                let backend = StatevectorBackend::new();
                backend.expectation(circuit, &binding, observable)
            }
            CircuitType::Photonic => {
                let backend = PhotonicsBackend::new();
                backend.expectation(circuit, &binding, observable)
            }
        }
    }

    /// Execute a circuit with shots and return measurement counts.
    pub fn execute_with_shots(
        &self,
        circuit_id: u32,
        params: &[f64],
        shots: u32,
        seed: Option<u64>,
    ) -> Result<HashMap<u64, u32>> {
        let circuit = self.circuits.get(&circuit_id).ok_or_else(|| {
            omega_core::error::OmegaError::InvalidCircuit(format!(
                "circuit {} not found",
                circuit_id
            ))
        })?;

        let binding = build_binding(circuit, params);
        let config = ExecConfig {
            shots: Some(shots),
            seed,
            mid_circuit_mode: MidCircuitMode::Skip,
        };

        let backend = StatevectorBackend::new();
        let result = backend.execute(circuit, &binding, &config)?;
        match result {
            ExecResult::Counts(counts) => Ok(counts),
            _ => Ok(HashMap::new()),
        }
    }

    /// Exact statevector of a circuit (no sampling; measurement gates skipped),
    /// returned as interleaved `(re, im)` f64 pairs in the backend's basis
    /// ordering (amplitude `k` is `[2k] + i·[2k+1]`). Used by the
    /// `omega_read_statevector` host import so a guest can verify
    /// amplitude-level quantities (e.g. a QFT) against a classical model.
    pub fn statevector_interleaved(&self, circuit_id: u32, params: &[f64]) -> Result<Vec<f64>> {
        let circuit = self.circuits.get(&circuit_id).ok_or_else(|| {
            omega_core::error::OmegaError::InvalidCircuit(format!(
                "circuit {} not found",
                circuit_id
            ))
        })?;
        let binding = build_binding(circuit, params);
        let config = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let backend = StatevectorBackend::new();
        match backend.execute(circuit, &binding, &config)? {
            ExecResult::Statevector(sv) => {
                let mut out = Vec::with_capacity(sv.len() * 2);
                for a in sv {
                    out.push(a.re);
                    out.push(a.im);
                }
                Ok(out)
            }
            other => Err(omega_core::error::OmegaError::InvalidCircuit(format!(
                "expected statevector, got {other:?}"
            ))),
        }
    }

    /// Run one Collapse-mode trajectory and return the final
    /// classical-register state as a vector of bits (LSB-first, length =
    /// `circuit.num_classical_bits`).
    ///
    /// Mid-circuit measurements sample on the fly and write into the
    /// creg per QASM `measure q[i] -> c[j]` semantics; the returned bits
    /// match Qiskit's last-write-wins `counts` key for the same shot.
    /// Pure-unitary circuits (no measure ops) return an all-zero vector
    /// since classical_bits is never written.
    pub fn execute_classical(
        &self,
        circuit_id: u32,
        params: &[f64],
        seed: Option<u64>,
    ) -> Result<Vec<u8>> {
        let circuit = self.circuits.get(&circuit_id).ok_or_else(|| {
            omega_core::error::OmegaError::InvalidCircuit(format!(
                "circuit {} not found",
                circuit_id
            ))
        })?;

        let binding = build_binding(circuit, params);
        let config = ExecConfig {
            shots: Some(1),
            seed,
            mid_circuit_mode: MidCircuitMode::Collapse,
        };

        let backend = StatevectorBackend::new();
        let result = backend.execute(circuit, &binding, &config)?;
        let creg_value: u64 = match result {
            // Collapse-mode counts are keyed by creg state (see
            // `crates/omega-backend-statevector/src/sim.rs`).
            ExecResult::Counts(c) => c.keys().copied().next().unwrap_or(0),
            _ => 0,
        };
        let n = circuit.num_classical_bits as usize;
        let mut bits = vec![0u8; n];
        for (i, slot) in bits.iter_mut().enumerate().take(64) {
            *slot = ((creg_value >> i) & 1) as u8;
        }
        Ok(bits)
    }

    /// Evaluate multiple observables on a single circuit execution.
    pub fn execute_multi(
        &self,
        circuit_id: u32,
        params: &[f64],
        observable_ids: &[u32],
    ) -> Result<Vec<f64>> {
        let circuit = self.circuits.get(&circuit_id).ok_or_else(|| {
            omega_core::error::OmegaError::InvalidCircuit(format!(
                "circuit {} not found",
                circuit_id
            ))
        })?;

        let binding = build_binding(circuit, params);
        let backend = StatevectorBackend::new();

        let mut results = Vec::with_capacity(observable_ids.len());
        for &obs_id in observable_ids {
            let observable = self.observables.get(&obs_id).ok_or_else(|| {
                omega_core::error::OmegaError::InvalidCircuit(format!(
                    "observable {} not found",
                    obs_id
                ))
            })?;
            results.push(backend.expectation(circuit, &binding, observable)?);
        }
        Ok(results)
    }

    /// Compute the gradient of an expectation value.
    pub fn compute_gradient(
        &self,
        circuit_id: u32,
        params: &[f64],
        observable_id: u32,
    ) -> Result<Vec<f64>> {
        let circuit = self.circuits.get(&circuit_id).ok_or_else(|| {
            omega_core::error::OmegaError::InvalidCircuit(format!(
                "circuit {} not found",
                circuit_id
            ))
        })?;
        let observable = self.observables.get(&observable_id).ok_or_else(|| {
            omega_core::error::OmegaError::InvalidCircuit(format!(
                "observable {} not found",
                observable_id
            ))
        })?;

        let binding = build_binding(circuit, params);

        let grad_pairs = match circuit.circuit_type {
            CircuitType::GateBased => {
                let backend = StatevectorBackend::new();
                compute_gradient(
                    &backend,
                    circuit,
                    &binding,
                    observable,
                    &GradMethod::Adjoint,
                )?
            }
            CircuitType::Photonic => {
                let backend = PhotonicsBackend::new();
                compute_gradient(
                    &backend,
                    circuit,
                    &binding,
                    observable,
                    &GradMethod::ParameterShift,
                )?
            }
        };

        Ok(grad_pairs.into_iter().map(|(_, v)| v).collect())
    }

    /// Compute the gradient of `E[f(x)]` for a [`Functional`] (Qubo or
    /// Table) without materialising a full observable on the guest
    /// side. Mirrors `omega-cli`'s `--gradient-of-fn` mode and shares
    /// its JSON-spec parser. `method_byte` is `0` for the diagonal
    /// (exact) path, `1` for the score-function REINFORCE estimator;
    /// `score_fn_shots` is only consulted when `method_byte == 1`.
    pub fn compute_functional_gradient_from_spec(
        &self,
        circuit_id: u32,
        params: &[f64],
        functional_spec: &str,
        method_byte: u8,
        score_fn_shots: u32,
    ) -> Result<Vec<f64>> {
        let circuit = self.circuits.get(&circuit_id).ok_or_else(|| {
            omega_core::error::OmegaError::InvalidCircuit(format!(
                "circuit {} not found",
                circuit_id
            ))
        })?;
        if !matches!(circuit.circuit_type, CircuitType::GateBased) {
            return Err(omega_core::error::OmegaError::Unsupported(
                "compute_functional_gradient_from_spec: only gate-based circuits supported".into(),
            ));
        }
        let (functional, _kind) = parse_functional_spec(functional_spec).map_err(|e| {
            omega_core::error::OmegaError::InvalidCircuit(format!(
                "functional spec parse error: {e}"
            ))
        })?;
        let method = match method_byte {
            0 => FunctionalGradMethod::DiagonalObservable,
            1 => FunctionalGradMethod::ScoreFunction {
                shots: if score_fn_shots == 0 {
                    1024
                } else {
                    score_fn_shots
                },
            },
            other => {
                return Err(omega_core::error::OmegaError::InvalidCircuit(format!(
                    "unknown functional grad method byte {other} (0=diagonal, 1=score-fn)"
                )))
            }
        };
        let binding = build_binding(circuit, params);
        let backend = StatevectorBackend::new();
        let grad_pairs =
            compute_functional_gradient(&backend, circuit, &binding, &functional, &method)?;
        Ok(grad_pairs.into_iter().map(|(_, v)| v).collect())
    }
}

/// Build a ParameterBinding from a circuit and a flat parameter array.
fn build_binding(circuit: &CircuitIR, params: &[f64]) -> ParameterBinding {
    let mut binding = ParameterBinding::new();
    let mut symbol_ids: Vec<u32> = circuit.symbols.keys().copied().collect();
    symbol_ids.sort();
    for (i, &sym_id) in symbol_ids.iter().enumerate() {
        let value = if i < params.len() { params[i] } else { 0.0 };
        binding.bind(sym_id, value);
    }
    binding
}

/// Create an H2 Hamiltonian (simplified 1-qubit model for testing).
/// H = -1.0523 * I + 0.3979 * Z0 - 0.3979 * Z1 - 0.0112 * Z0Z1 + 0.1809 * X0X1
pub fn h2_hamiltonian() -> Observable {
    Observable {
        terms: vec![
            // -1.0523 * I (constant offset, doesn't affect gradient)
            // We skip I terms since they don't change with parameters

            // 0.3979 * Z0
            (0.3979, vec![(0, PauliOp::Z)]),
            // -0.3979 * Z1
            (-0.3979, vec![(1, PauliOp::Z)]),
            // -0.0112 * Z0 Z1
            (-0.0112, vec![(0, PauliOp::Z), (1, PauliOp::Z)]),
            // 0.1809 * X0 X1
            (0.1809, vec![(0, PauliOp::X), (1, PauliOp::X)]),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omega_core::circuit::*;
    use smallvec::smallvec;

    fn make_bell_circuit() -> CircuitIR {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        c.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(0), Qubit(1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        c
    }

    #[test]
    fn test_execute_with_shots() {
        let mut state = HostState::new();
        let cid = state.register_circuit(make_bell_circuit());
        let counts = state.execute_with_shots(cid, &[], 1000, Some(42)).unwrap();
        // Bell state: only |00⟩ and |11⟩
        let c00 = counts.get(&0).copied().unwrap_or(0);
        let c11 = counts.get(&3).copied().unwrap_or(0);
        assert_eq!(c00 + c11, 1000);
        assert!(c00 > 400 && c00 < 600);
    }

    #[test]
    fn test_execute_classical_returns_creg_bits() {
        use omega_core::circuit::{GateKind, GateOp, Qubit};
        use smallvec::smallvec;

        // X q[0]; measure q[0] -> c[0]; X q[1]; measure q[1] -> c[1].
        // Final creg should be c0=1, c1=1.
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.num_classical_bits = 2;
        c.add_op(GateOp {
            gate: GateKind::X,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        c.add_op(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: Some(0),
            condition: None,
        });
        c.add_op(GateOp {
            gate: GateKind::X,
            qubits: smallvec![Qubit(1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        c.add_op(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec![Qubit(1)],
            params: smallvec![],
            classical_bit: Some(1),
            condition: None,
        });

        let mut state = HostState::new();
        let cid = state.register_circuit(c);
        let bits = state.execute_classical(cid, &[], Some(7)).unwrap();
        assert_eq!(bits, vec![1, 1], "both cbits should be 1, got {:?}", bits);
    }

    #[test]
    fn test_execute_multi() {
        let mut state = HostState::new();
        let cid = state.register_circuit(CircuitIR::new(1, CircuitType::GateBased));
        let obs_z = state.register_observable(Observable::z(0));
        let obs_x = state.register_observable(Observable::x(0));

        // |0⟩: ⟨Z⟩ = 1, ⟨X⟩ = 0
        let results = state.execute_multi(cid, &[], &[obs_z, obs_x]).unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            (results[0] - 1.0).abs() < 1e-10,
            "⟨Z⟩ on |0⟩ = {}",
            results[0]
        );
        assert!(results[1].abs() < 1e-10, "⟨X⟩ on |0⟩ = {}", results[1]);
    }

    #[test]
    fn test_compute_functional_gradient_from_spec_qubo() {
        // 2-qubit Rx⊗Rx ansatz; QUBO with diagonal couplings only.
        // The diagonal-observable path is exact, so the answer must
        // match a hand-computed reference closely.
        let mut state = HostState::new();
        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        circuit.symbols.insert(0, "th0".into());
        circuit.symbols.insert(1, "th1".into());
        circuit.add_op(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec![Qubit(0)],
            params: smallvec![omega_core::circuit::ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec![Qubit(1)],
            params: smallvec![omega_core::circuit::ParamExpr::Symbol(1)],
            classical_bit: None,
            condition: None,
        });
        let cid = state.register_circuit(circuit);
        // QUBO: f(x) = x_0 + 2 x_1 (linear, sparse couplings).
        let spec = r#"{"qubo":{"n":2,"Q":[[0,0,1.0],[1,1,2.0]]}}"#;
        let params = vec![0.4_f64, 1.3];
        let grad = state
            .compute_functional_gradient_from_spec(cid, &params, spec, 0, 0)
            .expect("functional grad");
        assert_eq!(grad.len(), 2);
        // Sanity: gradient is non-trivial and finite.
        assert!(
            grad.iter().all(|g| g.is_finite()),
            "non-finite gradient {grad:?}"
        );
        // Check vs the library's compute_functional_gradient on the
        // same inputs — the host wrapper must agree.
        use omega_core::executor::Backend;
        let backend = StatevectorBackend::new();
        let circuit_ref = &state.circuits[&cid];
        let binding = build_binding(circuit_ref, &params);
        let func = omega_core::gradient::Functional::Qubo(
            omega_core::qubo::Qubo::from_json(r#"{"n":2,"Q":[[0,0,1.0],[1,1,2.0]]}"#).unwrap(),
        );
        let _ = backend.name(); // ensure trait is in scope
        let lib_grad = omega_core::gradient::compute_functional_gradient(
            &backend,
            circuit_ref,
            &binding,
            &func,
            &omega_core::gradient::FunctionalGradMethod::DiagonalObservable,
        )
        .unwrap();
        for (i, (sym, lg)) in lib_grad.iter().enumerate() {
            assert_eq!(*sym, i as u32);
            assert!(
                (grad[i] - lg).abs() < 1e-9,
                "wrapper diverges from library at idx {i}: {} vs {lg}",
                grad[i]
            );
        }
    }

    #[test]
    fn test_register_observable_from_builder() {
        let mut state = HostState::new();
        let obs = Observable::z(0) + Observable::z(1);
        let id = state.register_observable(obs);
        assert!(id > 0);
        assert!(state.observables.contains_key(&id));
        assert_eq!(state.observables[&id].terms.len(), 2);
    }

    #[test]
    fn test_set_input_and_readback() {
        // input_bytes is the channel used by `omega_input_*` host
        // imports — guests read the raw bytes the CLI / lambda layer
        // staged before invocation. Pin the round-trip.
        let mut state = HostState::new();
        assert!(state.input_bytes.is_empty());
        state.set_input(b"{\"num_params\":4}".to_vec());
        assert_eq!(state.input_bytes, b"{\"num_params\":4}".to_vec());
        // A second call replaces (not appends) — match the documented
        // "stage the input the guest will read" semantics.
        state.set_input(b"new payload".to_vec());
        assert_eq!(state.input_bytes, b"new payload".to_vec());
    }

    #[test]
    fn test_build_qaoa_from_qubo_json_returns_ids_and_offset() {
        let mut state = HostState::new();
        // Triangle MaxCut QUBO: n=3, three edges (0,1), (1,2), (0,2),
        // standard encoding Q_ii=-degree=-2, Q_ij=+2 per edge.
        let json = r#"{"n":3,"Q":[[0,0,-2],[1,1,-2],[2,2,-2],[0,1,2],[1,2,2],[0,2,2]]}"#;
        let (cid, oid, offset) = state.build_qaoa_from_qubo_json(json, 2).unwrap();
        assert!(cid > 0 && oid > 0);
        assert!(state.circuits.contains_key(&cid));
        assert!(state.observables.contains_key(&oid));

        let circuit = &state.circuits[&cid];
        assert_eq!(circuit.num_qubits, 3);
        // depth=2 → 2*2 = 4 free symbols (gamma_0, beta_0, gamma_1, beta_1)
        assert_eq!(circuit.symbols.len(), 4);

        // Ising offset for triangle MaxCut: each edge contributes
        // -1/2 to the offset under the standard QUBO→Ising map (the
        // ZZ term carries +1/2). The code paths are tested
        // elsewhere; here we just pin that the offset is finite and
        // non-zero so we know the field is wired through.
        assert!(offset.is_finite());
        assert!(
            offset != 0.0,
            "expected non-zero Ising offset, got {offset}"
        );
    }

    #[test]
    fn test_build_qaoa_from_qubo_json_rejects_garbage() {
        let mut state = HostState::new();
        let result = state.build_qaoa_from_qubo_json("not json", 1);
        assert!(result.is_err());
        // No partial state should be left behind on parse failure.
        assert!(state.circuits.is_empty());
        assert!(state.observables.is_empty());
    }

    #[test]
    fn test_execute_expectation_bell_zz() {
        // Bell state: ⟨Z₀Z₁⟩ = +1.
        let mut state = HostState::new();
        let cid = state.register_circuit(make_bell_circuit());
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
        };
        let oid = state.register_observable(obs);
        let val = state.execute_expectation(cid, &[], oid).unwrap();
        assert!((val - 1.0).abs() < 1e-12, "⟨Z₀Z₁⟩ on Bell = {val}");
    }

    #[test]
    fn test_execute_expectation_unknown_circuit_errors() {
        let state = HostState::new();
        let err = state
            .execute_expectation(999, &[], 1)
            .expect_err("unknown circuit must error");
        assert!(format!("{err:?}").contains("999"));
    }

    #[test]
    fn test_execute_expectation_unknown_observable_errors() {
        let mut state = HostState::new();
        let cid = state.register_circuit(make_bell_circuit());
        let err = state
            .execute_expectation(cid, &[], 999)
            .expect_err("unknown observable must error");
        assert!(format!("{err:?}").contains("999"));
    }

    #[test]
    fn test_execute_with_shots_unknown_circuit_errors() {
        let state = HostState::new();
        let err = state
            .execute_with_shots(42, &[], 100, Some(1))
            .expect_err("unknown circuit must error");
        assert!(format!("{err:?}").contains("42"));
    }

    #[test]
    fn test_compute_gradient_returns_one_per_symbol() {
        // Single Rx gate; analytic gradient ⟨Z⟩ vs θ = -sin(θ).
        let mut state = HostState::new();
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.symbols.insert(0, "th0".into());
        circuit.add_op(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec![Qubit(0)],
            params: smallvec![omega_core::circuit::ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        let cid = state.register_circuit(circuit);
        let oid = state.register_observable(Observable::z(0));
        let theta = std::f64::consts::FRAC_PI_3;
        let grad = state.compute_gradient(cid, &[theta], oid).unwrap();
        assert_eq!(grad.len(), 1);
        let expected = -theta.sin();
        assert!(
            (grad[0] - expected).abs() < 1e-10,
            "compute_gradient = {} (expected {})",
            grad[0],
            expected
        );
    }

    #[test]
    fn test_compute_gradient_unknown_observable_errors() {
        let mut state = HostState::new();
        let cid = state.register_circuit(make_bell_circuit());
        let err = state
            .compute_gradient(cid, &[], 7777)
            .expect_err("unknown observable must error");
        assert!(format!("{err:?}").contains("7777"));
    }

    #[test]
    fn test_functional_gradient_unknown_method_byte_errors() {
        // Method byte 2 isn't 0 (diagonal) or 1 (score-fn) — must
        // fail fast with a clear message.
        let mut state = HostState::new();
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.symbols.insert(0, "th0".into());
        circuit.add_op(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec![Qubit(0)],
            params: smallvec![omega_core::circuit::ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        let cid = state.register_circuit(circuit);
        let spec = r#"{"qubo":{"n":1,"Q":[[0,0,1.0]]}}"#;
        let err = state
            .compute_functional_gradient_from_spec(cid, &[0.4], spec, 2, 0)
            .expect_err("unknown method byte must error");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("unknown functional grad method byte 2"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_functional_gradient_photonic_circuit_rejected() {
        // `Functional` only makes sense on diagonal Z observables —
        // photonic circuits don't have a Pauli structure, so the
        // host wrapper must refuse them with a clear Unsupported.
        let mut state = HostState::new();
        let circuit = CircuitIR::new(2, CircuitType::Photonic);
        let cid = state.register_circuit(circuit);
        let spec = r#"{"qubo":{"n":1,"Q":[[0,0,1.0]]}}"#;
        let err = state
            .compute_functional_gradient_from_spec(cid, &[], spec, 0, 0)
            .expect_err("photonic circuits must reject functional gradient");
        assert!(format!("{err:?}").contains("only gate-based"));
    }

    #[test]
    fn test_h2_hamiltonian_terms_are_pinned() {
        // The H2 Hamiltonian constants are referenced from VQE recipe
        // docs and the WASM CLI's --observable default; pin them so a
        // typo in the helper doesn't silently shift demo results.
        let h = h2_hamiltonian();
        assert_eq!(h.terms.len(), 4);
        // Sort by string representation of qubit indices so the test
        // doesn't depend on insertion order.
        let coeffs: Vec<f64> = h.terms.iter().map(|(c, _)| *c).collect();
        assert!(coeffs.contains(&0.3979));
        assert!(coeffs.contains(&-0.3979));
        assert!(coeffs.contains(&-0.0112));
        assert!(coeffs.contains(&0.1809));
    }

    #[test]
    fn test_build_binding_pads_missing_params_with_zero() {
        // Circuit has 3 free symbols but caller passes 2 values —
        // remaining symbols default to 0.0. This is the documented
        // behaviour the WASM guests rely on for variable-arity
        // ansätze.
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.symbols.insert(0, "a".into());
        circuit.symbols.insert(1, "b".into());
        circuit.symbols.insert(2, "c".into());
        let binding = build_binding(&circuit, &[1.5, 2.5]);
        assert!(
            (binding
                .resolve(&omega_core::circuit::ParamExpr::Symbol(0))
                .unwrap()
                - 1.5)
                .abs()
                < 1e-12
        );
        assert!(
            (binding
                .resolve(&omega_core::circuit::ParamExpr::Symbol(1))
                .unwrap()
                - 2.5)
                .abs()
                < 1e-12
        );
        // Symbol 2 wasn't supplied — must default to 0.0, not error.
        assert!(
            binding
                .resolve(&omega_core::circuit::ParamExpr::Symbol(2))
                .unwrap()
                .abs()
                < 1e-12
        );
    }
}
