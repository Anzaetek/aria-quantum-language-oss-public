// SPDX-License-Identifier: Apache-2.0
//! cqs — Hadamard-test overlap estimation (the quantum primitive of CQS).
//!
//! WHAT: CQS (qa-cqs, arXiv:1909.07344) builds its Gram matrix from overlaps
//!   Re⟨ψ|P|ψ⟩ estimated by Hadamard tests. This runs one such test through
//!   omega: |ψ⟩=RY(π/3)|0⟩, P=Z ⇒ Re⟨ψ|Z|ψ⟩ = cos(π/3) = 0.5.
//! QUANTUM: cqs.aria (HadamardTestZ) sampled on the omega runtime; the ancilla
//!   reads 0 with probability (1+Re⟨Z⟩)/2, so Re⟨Z⟩ = 2·P(anc=0) − 1.
//! CLASSICAL: ⟨ψ|Z|ψ⟩ via omega-core::cqs::apply_pauli on |ψ⟩ = [cos(π/6), sin(π/6)].
//! CHECK: the sampled overlap matches the classical value within ±0.05 (shots).
//!   (The full CQS linear solve from many such overlaps is the Rust module.)

use aria_verify_core::{banner, harness, resolve, util, Complex64, Transport, Verdict};
use omega_core::cqs::apply_pauli;

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "cqs",
        "Hadamard-test overlap Re⟨ψ|Z|ψ⟩ (CQS's Gram-matrix primitive)",
        &transport.label(guest),
    );

    let shots = 8192u32;
    let lowered = harness::load_lowered("cqs.aria", "HadamardTestZ", &[])?;
    let (payload, _) = harness::execute_report(
        transport,
        lowered.ir.clone(),
        harness::AppMode::Counts { shots },
        &[],
    )?;
    let counts = util::counts_from_payload(&payload);
    let total = util::total_shots(&counts) as f64;
    // Ancilla is qubit 0 (the only measured creg bit); P(anc=0) = share with bit0=0.
    let p0 = counts
        .iter()
        .filter(|(k, _)| k & 1 == 0)
        .map(|(_, c)| *c as f64)
        .sum::<f64>()
        / total.max(1.0);
    let re_z_quantum = 2.0 * p0 - 1.0;

    // Classical overlap via the CQS primitive: |ψ⟩ = RY(π/3)|0⟩ = [cos π/6, sin π/6].
    let (c6, s6) = (
        (std::f64::consts::PI / 6.0).cos(),
        (std::f64::consts::PI / 6.0).sin(),
    );
    let psi = vec![Complex64::new(c6, 0.0), Complex64::new(s6, 0.0)];
    let zpsi = apply_pauli(&[3], &psi);
    let re_z_classical: f64 = psi.iter().zip(&zpsi).map(|(a, b)| (a.conj() * b).re).sum();

    println!("  P(anc=0) = {p0:.4}  ⇒  Re⟨Z⟩_quantum = {re_z_quantum:+.4}");
    Ok(banner::report_scalar(
        "cqs",
        "Re⟨ψ|Z|ψ⟩ (Hadamard test, 8192 shots)",
        re_z_quantum,
        "cos(π/3) via apply_pauli",
        re_z_classical,
        0.05,
    ))
}
