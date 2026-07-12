// SPDX-License-Identifier: Apache-2.0
//! Encoded 2-qubit Grover on the Steane `[[7,1,3]]` code.
//!
//! Grover's oracle + diffuser for 2 qubits are pure Clifford (H, X, CZ), so the
//! algorithm runs **exactly** on encoded logical qubits. One iteration finds the
//! marked state with certainty, so the logical read-out is deterministic:
//! `⟨Z̄_i⟩ = (−1)^{marked_i}`. We check that for every marked state on the
//! Pauli-propagation backend and confirm the exact statevector agrees — i.e. the
//! algorithm survives the code and both simulators concur.

use aria_qec::ecc::SimBackend;
use aria_qec::logical::{
    compile_physical, logical_grover2, logical_z_expectation, SteaneTransversal,
};
use aria_verify_core::{banner, Transport, Verdict};

pub fn run(_transport: Transport) -> Result<Verdict, String> {
    banner::header(
        "qec-grover",
        "2-qubit Grover on encoded Steane [[7,1,3]] logical qubits — logical ⟨Z̄⟩ = (−1)^marked",
        "native (aria-qec transversal; pauliprop + statevector)",
    );

    let code = SteaneTransversal::new();
    // For each marked state we record the two logical ⟨Z̄⟩ (must equal the ideal
    // ±1) plus the pauliprop−statevector gap (must be 0). All entries compared to
    // their target within 1e-6.
    let mut quantum: Vec<f64> = Vec::new();
    let mut ideal: Vec<f64> = Vec::new();

    for marked in 0u8..4 {
        let prog = compile_physical(&logical_grover2(marked), &code);
        let z0 = logical_z_expectation(&prog, 0, SimBackend::PauliProp)?;
        let z1 = logical_z_expectation(&prog, 1, SimBackend::PauliProp)?;
        let sv0 = logical_z_expectation(&prog, 0, SimBackend::Statevector)?;
        let sv1 = logical_z_expectation(&prog, 1, SimBackend::Statevector)?;
        let want0 = if marked & 1 == 1 { -1.0 } else { 1.0 };
        let want1 = if (marked >> 1) & 1 == 1 { -1.0 } else { 1.0 };
        let ok = (z0 - want0).abs() < 1e-6 && (z1 - want1).abs() < 1e-6;
        println!(
            "  marked=|{:02b}⟩: ⟨Z̄_0⟩={z0:+.6} ⟨Z̄_1⟩={z1:+.6}  (sv {sv0:+.6},{sv1:+.6})  {}",
            marked,
            if ok { "✓ recovered" } else { "✗" }
        );
        // Correctness: encoded ⟨Z̄⟩ vs ideal ±1.
        quantum.push(z0);
        ideal.push(want0);
        quantum.push(z1);
        ideal.push(want1);
        // Backend cross-check: pauliprop − statevector vs 0.
        quantum.push(z0 - sv0);
        ideal.push(0.0);
        quantum.push(z1 - sv1);
        ideal.push(0.0);
    }

    Ok(banner::report_values(
        "qec-grover",
        "encoded ⟨Z̄⟩ (4 marked) + pauliprop−statevector gap",
        &quantum,
        "ideal ±1 / 0",
        &ideal,
        1e-6,
    ))
}
