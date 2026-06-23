// SPDX-License-Identifier: Apache-2.0
//! vqe_ansatz — ground-state energy of the H₂ molecule (minimal basis, the
//! standard 2-qubit Hamiltonian).
//!
//! WHAT: the lowest eigenvalue E₀ of the explicit H₂ Hamiltonian below.
//! QUANTUM: minimize ⟨ψ(θ)|H|ψ(θ)⟩ over VQEAnsatz(n_layers=2); the optimizer
//!   loop runs inside vqe.wasm.
//! CLASSICAL: exact smallest eigenvalue of the 4×4 Hamiltonian matrix (Jacobi).
//! CHECK: |E_vqe − E₀| ≤ 1e-3.

use aria_verify_core::{banner, harness, oracle, resolve, Observable, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    // H₂ Hamiltonian, both as the omega Pauli-sum string (quantum) and as a
    // term list (classical oracle). Kept identical, side by side.
    let h2_str = "-0.4804*I0+0.3435*Z0+-0.4347*Z1+0.5716*Z0Z1+0.0910*X0X1+0.0910*Y0Y1";
    let terms: Vec<(f64, Vec<(usize, char)>)> = vec![
        (-0.4804, vec![]),
        (0.3435, vec![(0, 'Z')]),
        (-0.4347, vec![(1, 'Z')]),
        (0.5716, vec![(0, 'Z'), (1, 'Z')]),
        (0.0910, vec![(0, 'X'), (1, 'X')]),
        (0.0910, vec![(0, 'Y'), (1, 'Y')]),
    ];

    let guest = "vqe";
    let transport = resolve(transport_override, guest);
    banner::header(
        "vqe_ansatz",
        "ground-state energy E₀ of the H₂ Hamiltonian",
        &transport.label(guest),
    );
    println!("  H = {h2_str}");

    let classical_e0 = oracle::ground_state_energy(2, &terms);

    let lowered = harness::load_lowered("vqe_ansatz.aria", "VQEAnsatz", &[("n_layers", 2)])?;
    let n_params = lowered.symbol_ids.len().max(1);
    let obs = Observable::parse(h2_str)?;
    let (e_vqe, _params) = harness::minimize(
        transport,
        guest,
        lowered.ir,
        obs,
        n_params,
        vec![0.1; n_params],
        600,
        0.2,
    )?;

    Ok(banner::report_scalar(
        "vqe_ansatz",
        "VQE min energy",
        e_vqe,
        "exact min eigenvalue",
        classical_e0,
        1e-3,
    ))
}
