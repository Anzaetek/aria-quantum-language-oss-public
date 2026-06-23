// SPDX-License-Identifier: Apache-2.0
//! qsvd — singular values of a fixed matrix M.
//!
//! WHAT: the singular values of the explicit 2×2 matrix M below.
//! QUANTUM: M is symmetric, so its singular values are √(eigenvalues of MᵀM).
//!   We variationally find the LARGEST eigenvalue λ₁ of G = MᵀM by minimizing
//!   ⟨−G⟩ over the 1-qubit ansatz in qsvd.aria (a VQE; G is written as a Pauli
//!   sum a·I + bx·X + by·Y + bz·Z). The optimizer loop runs inside vqe.wasm.
//!   The second singular value follows from the 2×2 trace identity
//!   λ₂ = tr(G) − λ₁, and σᵢ = √λᵢ.
//! CLASSICAL: a pure-Rust Jacobi SVD of the very same M.
//! CHECK: the two singular-value vectors agree to ≤ 1e-3.

use aria_verify_core::{
    banner, harness, oracle, resolve, Complex64, Observable, Transport, Verdict,
};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    // The matrix being SVD-ized — stated explicitly so it is obvious.
    let m = [[2.0_f64, 1.0], [1.0, 3.0]];
    let m_vec: Vec<Vec<f64>> = m.iter().map(|r| r.to_vec()).collect();

    let guest = "vqe";
    let transport = resolve(transport_override, guest);
    banner::header(
        "qsvd",
        "singular values of the fixed matrix M (variational, via MᵀM)",
        &transport.label(guest),
    );
    println!("  M = [[{:.3}, {:.3}],", m[0][0], m[0][1]);
    println!("       [{:.3}, {:.3}]]", m[1][0], m[1][1]);

    // Classical oracle: Jacobi SVD of M.
    let classical_sv = oracle::singular_values(&m_vec);

    // G = MᵀM (2×2, real symmetric).
    let mut g = [[0.0_f64; 2]; 2];
    for (i, g_row) in g.iter_mut().enumerate() {
        for (j, g_ij) in g_row.iter_mut().enumerate() {
            *g_ij = (0..2).map(|k| m[k][i] * m[k][j]).sum();
        }
    }
    let trace_g = g[0][0] + g[1][1];

    // Pauli decomposition of G (Hermitian 2×2).
    let g_c = [
        [Complex64::new(g[0][0], 0.0), Complex64::new(g[0][1], 0.0)],
        [Complex64::new(g[1][0], 0.0), Complex64::new(g[1][1], 0.0)],
    ];
    let (a, bx, by, bz) = oracle::pauli_decompose_2x2(&g_c);

    // Observable for the optimizer = −G (so minimizing it maximizes ⟨G⟩).
    let mut terms = vec![format!("{:.12}*I0", -a)];
    if bx.abs() > 1e-12 {
        terms.push(format!("{:.12}*X0", -bx));
    }
    if by.abs() > 1e-12 {
        terms.push(format!("{:.12}*Y0", -by));
    }
    if bz.abs() > 1e-12 {
        terms.push(format!("{:.12}*Z0", -bz));
    }
    let neg_g = Observable::parse(&terms.join("+"))?;

    // The variational ansatz lives in the shipped qsvd.aria.
    let lowered = harness::load_lowered("qsvd.aria", "QsvdAnsatz", &[])?;
    let n_params = lowered.symbol_ids.len().max(1);

    // Minimize ⟨−G⟩ → −λ₁. The loop runs in vqe.wasm (or native fallback).
    let (min_val, _params) = harness::minimize(
        transport,
        guest,
        lowered.ir,
        neg_g,
        n_params,
        vec![0.5; n_params],
        400,
        0.3,
    )?;
    let lambda_max = -min_val;
    let lambda_min = (trace_g - lambda_max).max(0.0);
    let mut quantum_sv = vec![lambda_max.max(0.0).sqrt(), lambda_min.sqrt()];
    quantum_sv.sort_by(|x, y| y.partial_cmp(x).unwrap());

    Ok(banner::report_values(
        "qsvd",
        "variational σ via λ(MᵀM)",
        &quantum_sv,
        "Jacobi SVD",
        &classical_sv,
        1e-3,
    ))
}
