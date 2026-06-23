// SPDX-License-Identifier: Apache-2.0
//! qsp — Quantum Signal Processing: the d-fold signal product is a Chebyshev T_d.
//!
//! WHAT: the QSP circuit interleaves the signal W(x)=RX(θ) with phase rotations
//!   RZ(φ_k). With every phase φ_k = 0 it collapses to W(x)^d = RX(d·θ), whose
//!   ⟨Z⟩ readout is cos(d·θ) = T_d(cos θ) — the Chebyshev baseline of QSP.
//! QUANTUM: run qsp.aria (d=4) binding all φ_k = 0 and the signal θ; read ⟨Z₀⟩.
//! CLASSICAL: ⟨Z₀⟩ = cos(d·θ), exactly (convention-robust — cos is even).
//! CHECK: |⟨Z₀⟩ − cos(d·θ)| ≤ 1e-9.
//!
//! The general phase ⇒ degree-d polynomial correspondence (forward + converse)
//! is proven sorry-free in proofs/lean4/QuantumProofs/QSP.lean.

use aria_verify_core::{banner, harness, resolve, Observable, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let d = 4i64;
    let theta = 0.7_f64;
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "qsp",
        "Quantum Signal Processing — zero-phase ⟨Z₀⟩ vs the Chebyshev cos(d·θ)",
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("qsp.aria", "QSP", &[("d", d)])?;
    // Bind every signal-processing phase φ_k = 0 and the signal symbol = θ, in
    // the runtime's parameter order (free symbols sorted by SymbolId), so the
    // product collapses to W(θ)^d = RX(d·θ).
    let mut ids: Vec<(u32, String)> = lowered
        .symbol_ids
        .iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect();
    ids.sort_by_key(|(id, _)| *id);
    let params: Vec<f64> = ids
        .iter()
        .map(|(_, name)| if name == "signal_0" { theta } else { 0.0 })
        .collect();

    let (z, _) = harness::execute_report(
        transport,
        lowered.ir,
        harness::AppMode::Expectations(vec![Observable::z(0)]),
        &params,
    )?;
    let measured = z[0];
    let classical = (d as f64 * theta).cos();

    println!("  d = {d}, θ = {theta}, all φ_k = 0");
    Ok(banner::report_scalar(
        "qsp",
        "⟨Z₀⟩ with all QSP phases = 0",
        measured,
        "cos(d·θ) = T_d(cos θ)",
        classical,
        1e-9,
    ))
}
