//! MPS (Matrix Product State) backend for circuit simulation.
//!
//! Uses tensor network contraction with truncated SVD to efficiently
//! simulate circuits with limited entanglement (low bond dimension).

use std::collections::HashMap;

use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::SeedableRng;

use omega_core::circuit::*;
use omega_core::error::{OmegaError, Result};
use omega_core::executor::*;
use omega_core::params::ParameterBinding;

use crate::gates;
use crate::mps::Mps;

/// MPS-based quantum circuit simulator.
///
/// The `max_bond_dim` parameter controls the trade-off between accuracy and
/// memory/time: higher values are more accurate but use more resources.
/// For states with bond dimension <= max_bond_dim, the simulation is exact.
pub struct MpsBackend {
    pub max_bond_dim: usize,
}

impl MpsBackend {
    pub fn new(max_bond_dim: usize) -> Self {
        Self { max_bond_dim }
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
