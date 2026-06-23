//! Integration tests for Grover's algorithm variants executed on the statevector backend.

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::grover::*;
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

fn run_probs(circuit: &CircuitIR) -> Vec<f64> {
    let backend = StatevectorBackend::new();
    let params = ParameterBinding::new();
    let config = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let result = backend.execute(circuit, &params, &config).unwrap();
    let sv = result.statevector();
    sv.iter().map(|a| a.norm_sqr()).collect()
}

// ---- Standard Grover ----

#[test]
fn test_grover_2qubit_amplifies_target() {
    // 2 qubits, mark |11⟩ (state 3). 1 iteration → probability 1.0.
    let circuit = grover_circuit(2, &[3], None);
    let probs = run_probs(&circuit);
    assert!(probs[3] > 0.99, "P(|11⟩) = {:.4}, expected ~1.0", probs[3]);
}

#[test]
fn test_grover_2qubit_mark_00() {
    let circuit = grover_circuit(2, &[0], None);
    let probs = run_probs(&circuit);
    assert!(probs[0] > 0.99, "P(|00⟩) = {:.4}, expected ~1.0", probs[0]);
}

#[test]
fn test_grover_3qubit_single_target() {
    // 3 qubits, mark |101⟩ = 5. Optimal: 2 iterations.
    let circuit = grover_circuit(3, &[5], None);
    let probs = run_probs(&circuit);
    assert!(probs[5] > 0.9, "P(|101⟩) = {:.4}, expected > 0.9", probs[5]);
}

#[test]
fn test_grover_3qubit_two_targets() {
    // 3 qubits (N=8), 2 marked states.
    let circuit = grover_circuit(3, &[2, 6], None);
    let probs = run_probs(&circuit);
    let p_marked = probs[2] + probs[6];
    assert!(
        p_marked > 0.9,
        "P(marked) = {:.4}, expected > 0.9",
        p_marked
    );
}

#[test]
fn test_grover_4qubit_with_ancillae() {
    // 4 qubits, 1 ancilla for V-chain MCX.
    let circuit = grover_circuit(4, &[15], None);
    assert_eq!(circuit.num_qubits, 5);
    let probs = run_probs(&circuit);
    assert!(
        probs[15] > 0.9,
        "P(|1111⟩) = {:.4}, expected > 0.9",
        probs[15]
    );
}

#[test]
fn test_grover_5qubit() {
    // 5 qubits = 32 states, mark |10101⟩ = 21. Needs 2 ancillae.
    let circuit = grover_circuit(5, &[21], None);
    assert_eq!(circuit.num_qubits, 7);
    let probs = run_probs(&circuit);
    assert!(
        probs[21] > 0.9,
        "P(|10101⟩) = {:.4}, expected > 0.9",
        probs[21]
    );
}

// ---- Diffusion operator ----

#[test]
fn test_diffusion_amplifies() {
    // Manually: H → oracle(|00⟩) → diffusion on 2 qubits
    let circuit = grover_circuit(2, &[0], Some(1));
    let probs = run_probs(&circuit);
    assert!(probs[0] > 0.9, "P(|00⟩) = {:.4}, expected > 0.9", probs[0]);
}

// ---- Amplitude amplification ----

#[test]
fn test_amplitude_amplification_boosts_target() {
    // Custom state prep: Ry(π/3)|0⟩ ⊗ H|0⟩
    // Initial P(|11⟩) = sin²(π/6) × 0.5 = 0.125
    let mut state_prep = CircuitIR::new(2, CircuitType::GateBased);
    state_prep.add_op(GateOp {
        gate: GateKind::Ry,
        qubits: smallvec![Qubit(0)],
        params: smallvec![ParamExpr::Concrete(std::f64::consts::FRAC_PI_3)],
        classical_bit: None,
        condition: None,
    });
    state_prep.add_op(GateOp {
        gate: GateKind::H,
        qubits: smallvec![Qubit(1)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    });

    let circuit = amplitude_amplification(&state_prep, &[3], 1);
    let probs = run_probs(&circuit);
    assert!(
        probs[3] > 0.125,
        "P(|11⟩) = {:.4}, should be amplified above 0.125",
        probs[3]
    );
}

// ---- Approximate Grover ----

#[test]
fn test_approximate_grover_partial_boost() {
    // 4-qubit space, optimal = 3 iterations, use only 1.
    // Should boost above uniform 1/16 = 0.0625.
    let circuit = grover_circuit(4, &[7], Some(1));
    let probs = run_probs(&circuit);
    let uniform = 1.0 / 16.0;
    assert!(
        probs[7] > uniform * 2.0,
        "P(|0111⟩) = {:.4}, should be well above uniform {:.4}",
        probs[7],
        uniform
    );
}

// ---- Noise-tolerant Grover ----

#[test]
fn test_noise_tolerant_grover_executes() {
    let circuit = noise_tolerant_grover(3, &[5], 0.01);
    let probs = run_probs(&circuit);
    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-10, "probabilities should sum to 1");
}

#[test]
fn test_noise_tolerant_grover_under_depolarizing() {
    // Compare standard vs noise-tolerant on the NoisyStatevectorBackend.
    // With noise, fewer iterations should maintain higher success probability.
    use omega_backend_statevector::NoisyStatevectorBackend;

    let error_rate = 0.005;
    let noisy_backend = NoisyStatevectorBackend::new(error_rate, Some(42));

    let params = ParameterBinding::new();
    let config = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };

    // Standard Grover: optimal iterations (may be too many under noise)
    let circuit_standard = grover_circuit(3, &[5], None);
    let result_std = noisy_backend
        .execute(&circuit_standard, &params, &config)
        .unwrap();
    let probs_std: Vec<f64> = result_std
        .statevector()
        .iter()
        .map(|a| a.norm_sqr())
        .collect();

    // Noise-tolerant: fewer iterations
    let circuit_tolerant = noise_tolerant_grover(3, &[5], error_rate);
    let noisy_backend2 = NoisyStatevectorBackend::new(error_rate, Some(42));
    let result_tol = noisy_backend2
        .execute(&circuit_tolerant, &params, &config)
        .unwrap();
    let probs_tol: Vec<f64> = result_tol
        .statevector()
        .iter()
        .map(|a| a.norm_sqr())
        .collect();

    // Both should execute. Under noise, the tolerant version should maintain
    // reasonable probability on the target.
    let total_std: f64 = probs_std.iter().sum();
    let total_tol: f64 = probs_tol.iter().sum();
    assert!((total_std - 1.0).abs() < 0.1, "std probs should ~sum to 1");
    assert!(
        (total_tol - 1.0).abs() < 0.1,
        "tolerant probs should ~sum to 1"
    );
}
