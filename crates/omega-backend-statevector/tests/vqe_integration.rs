//! Focused correctness tests for `omega_core::vqe::vqe_circuit`.
//!
//! The harness in `qubo_compare.rs` shows VQE converges on real QUBO
//! instances; these tests pin the underlying contract (ansatz
//! expressivity and adjoint-AD gradient correctness) on observables
//! with analytically known ground-state energies.

use std::collections::HashMap;

use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::CircuitIR;
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::gradient::{compute_gradient, GradMethod};
use omega_core::params::ParameterBinding;
use omega_core::vqe::vqe_circuit;

/// Single qubit, observable ⟨Z⟩. Ground state |1⟩ at energy -1.
/// vqe_circuit(1, 0) is a single Ry(θ); ⟨Z⟩ = cos(θ); minimum at θ=π.
#[test]
fn vqe_single_qubit_z() {
    let backend = StatevectorBackend::new();
    let circuit = vqe_circuit(1, 0);
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z)])],
    };

    let symbols = sorted_symbols(&circuit);
    assert_eq!(symbols.len(), 1);

    let final_energy = run_gd(&backend, &circuit, &obs, &symbols, &[0.5], 60, 0.2);
    assert!(
        (final_energy - (-1.0)).abs() < 1e-4,
        "VQE on single-qubit Z should reach -1, got {final_energy}"
    );
}

/// Two qubits, observable Z₀Z₁. Ground states |01⟩ and |10⟩ both at -1.
/// vqe_circuit(2, 1) has 4 params + one CX; expressive enough for an
/// anti-correlated state.
#[test]
fn vqe_two_qubit_zz_antiferro() {
    let backend = StatevectorBackend::new();
    let circuit = vqe_circuit(2, 1);
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
    };

    let symbols = sorted_symbols(&circuit);
    assert_eq!(symbols.len(), 4);

    // Random asymmetric init so we don't land on a fixed point at θ=0.
    let init: Vec<f64> = symbols
        .iter()
        .enumerate()
        .map(|(i, _)| ((i + 1) as f64) * 0.371)
        .collect();
    let final_energy = run_gd(&backend, &circuit, &obs, &symbols, &init, 200, 0.15);
    assert!(
        (final_energy - (-1.0)).abs() < 1e-3,
        "VQE on Z₀Z₁ should reach -1, got {final_energy}"
    );
}

/// Two qubits, observable -Z₀ - Z₁ (negative sum of local fields).
/// Minimisation drives both qubits to ⟨Z⟩=+1 (state |00⟩), so the
/// minimum is -1 + -1 = -2. Both qubits must independently reach
/// θ=0; the entangling CX layer should not obstruct product-state
/// convergence.
#[test]
fn vqe_two_qubit_field_sum() {
    let backend = StatevectorBackend::new();
    let circuit = vqe_circuit(2, 1);
    let obs = Observable {
        terms: vec![(-1.0, vec![(0, PauliOp::Z)]), (-1.0, vec![(1, PauliOp::Z)])],
    };

    let symbols = sorted_symbols(&circuit);
    let init: Vec<f64> = vec![0.4, 0.6, 0.3, 0.5];
    let final_energy = run_gd(&backend, &circuit, &obs, &symbols, &init, 200, 0.15);
    assert!(
        (final_energy - (-2.0)).abs() < 1e-3,
        "VQE on -Z₀-Z₁ should reach -2 (state |00⟩), got {final_energy}"
    );
}

// ---------------------------------------------------------------------

fn sorted_symbols(circuit: &CircuitIR) -> Vec<u32> {
    let mut ids: Vec<u32> = circuit.symbols.keys().copied().collect();
    ids.sort();
    ids
}

fn run_gd(
    backend: &StatevectorBackend,
    circuit: &CircuitIR,
    obs: &Observable,
    symbols: &[u32],
    init: &[f64],
    max_iters: usize,
    lr: f64,
) -> f64 {
    let mut values = init.to_vec();
    let mut last = f64::INFINITY;
    for iter in 1..=max_iters {
        let mut pb = ParameterBinding::new();
        for (sid, v) in symbols.iter().zip(values.iter()) {
            pb.bind(*sid, *v);
        }
        let grads = compute_gradient(backend, circuit, &pb, obs, &GradMethod::Adjoint)
            .expect("adjoint gradient");
        let map: HashMap<u32, f64> = grads.into_iter().collect();
        for (i, sid) in symbols.iter().enumerate() {
            let g = *map.get(sid).unwrap_or(&0.0);
            values[i] -= lr * g;
        }
        let cost = backend
            .expectation(circuit, &pb, obs)
            .unwrap_or(f64::INFINITY);
        if iter > 5 && (last - cost).abs() < 1e-10 {
            return cost;
        }
        last = cost;
    }
    last
}
