// SPDX-License-Identifier: Apache-2.0
//! Kitaev chain at the sweet spot, t = Δ = 1, μ = 0.
//!
//! The Jordan–Wigner image is −Σ_j X_j X_{j+1}. The product state with |+⟩
//! on every mode is a ground state, energy −(N−1). Each row is that circuit
//! through `expectation_fermionic_with_certificate`. A π/2 rotation does not
//! split a Majorana monomial, so every row is exact (`dropped_mass` 0). The
//! row asserts that; printing the number is not the check.

use std::time::Instant;

use omega_backend_majoranaprop::MajoranaPropBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::fermion::FermionicOp;
use omega_core::params::ParameterBinding;

fn gate(kind: GateKind, wire: u32) -> GateOp {
    GateOp {
        gate: kind,
        qubits: std::iter::once(Qubit(wire)).collect(),
        params: Default::default(),
        classical_bit: None,
        condition: None,
    }
}

/// −(a†_j a_{j+1} + h.c.) + (a_j a_{j+1} + h.c.) on each bond.
fn kitaev(n: u32) -> FermionicOp {
    let mut op = FermionicOp::zero();
    for j in 0..n - 1 {
        op = op + FermionicOp::hopping(j, j + 1, -1.0);
        let pair = FermionicOp::lower(j).mul(&FermionicOp::lower(j + 1))
            + FermionicOp::raise(j + 1).mul(&FermionicOp::raise(j));
        op = op + pair;
    }
    op
}

fn prep(n: u32) -> CircuitIR {
    let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
    for j in 0..n {
        circuit.add_op(gate(GateKind::H, j));
    }
    circuit
}

fn main() {
    let backend = MajoranaPropBackend::new();
    let params = ParameterBinding::new();
    println!(
        "{:>6} {:>14} {:>14} {:>12} {:>10} {:>10}",
        "N", "energy", "analytic", "|delta|", "monomials", "us"
    );
    for n in 2u32..=128 {
        let op = kitaev(n);
        let circuit = prep(n);
        let analytic = -((n - 1) as f64);
        let started = Instant::now();
        let (energy, cert) = backend
            .expectation_fermionic_with_certificate(&circuit, &params, &op)
            .expect("kitaev expectation");
        let us = started.elapsed().as_secs_f64() * 1.0e6;
        let delta = (energy - analytic).abs();
        println!(
            "{n:>6} {energy:>14.8} {analytic:>14.8} {delta:>12.3e} {:>10} {us:>10.1}",
            cert.final_terms
        );
        assert!(
            delta < 1e-9,
            "N={n}: energy {energy} vs analytic {analytic}"
        );
        assert_eq!(
            cert.dropped_mass, 0.0,
            "N={n}: dropped_mass {}",
            cert.dropped_mass
        );
        assert!(cert.is_exact(), "N={n} certificate is not exact");
        assert_eq!(
            cert.final_terms,
            (n - 1) as usize,
            "N={n}: one monomial per bond"
        );
    }
}
