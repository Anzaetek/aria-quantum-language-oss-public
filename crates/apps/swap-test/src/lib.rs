// SPDX-License-Identifier: Apache-2.0
//! swap_test — overlap of two states from one ancilla measurement.
//!
//! WHAT: |⟨ψ|φ⟩|² via P(ancilla=0) = ½ + ½|⟨ψ|φ⟩|².
//! QUANTUM: run swap_test.aria, but flip the first qubit of the SECOND register
//!   to |−⟩ so the two product states are ORTHOGONAL; sample the ancilla and
//!   read P(ancilla=0) (omega_app counts mode).
//! CLASSICAL: ⟨+|−⟩·⟨+|+⟩ = 0 ⇒ overlap 0 ⇒ P(0)=0.5.
//! CHECK: P(0)_measured ≈ ½+½|⟨ψ|φ⟩|² within ±0.02.

use aria_verify_core::{banner, harness, resolve, util, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let n: i64 = 2;
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "swap_test",
        "fidelity |⟨ψ|φ⟩|² of two registers via the ancilla",
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("swap_test.aria", "SwapTest", &[("n", n)])?;
    // Flip the first qubit of the SECOND register (global qubit n+1) so that
    // after its Hadamard it becomes |−⟩ while the first register stays |+⟩.
    // Then ⟨ψ|φ⟩ = ⟨+|−⟩·⟨+|+⟩ = 0 — orthogonal states, a non-trivial check.
    let mut ir = lowered.ir.clone();
    harness::prepend_basis_state(&mut ir, 1u64 << (n as u64 + 1));
    println!("  ψ = |+⟩^{n} (reg 1),  φ = |−⟩⊗|+⟩^{} (reg 2)", n - 1);
    let (payload, _) =
        harness::execute_report(transport, ir, harness::AppMode::Counts { shots: 8192 }, &[])?;
    let counts = util::counts_from_payload(&payload);
    // Counts are keyed by the full qubit register; the ancilla is qubit 0, so
    // take the MARGINAL probability that bit 0 is zero.
    let total = util::total_shots(&counts) as f64;
    let p0 = counts
        .iter()
        .filter(|(k, _)| k & 1 == 0)
        .map(|(_, c)| *c as f64)
        .sum::<f64>()
        / total.max(1.0);

    // Orthogonal product states ⇒ overlap 0 ⇒ P(0) = 0.5.
    let overlap_sq = 0.0;
    let p0_classical = 0.5 + 0.5 * overlap_sq;

    Ok(banner::report_scalar(
        "swap_test",
        "P(ancilla=0), 8192 shots",
        p0,
        "½+½|⟨ψ|φ⟩|²",
        p0_classical,
        0.02,
    ))
}
