// SPDX-License-Identifier: Apache-2.0
//! bell — Bell state (|00⟩+|11⟩)/√2 vs the analytic amplitudes.
//!
//! WHAT: prepare H·CX|00⟩ and read the EXACT statevector (measurements skipped).
//! QUANTUM: run bell.aria through the omega runtime, statevector mode.
//! CLASSICAL: amplitudes (1/√2, 0, 0, 1/√2) ⇒ probabilities (½, 0, 0, ½).
//! CHECK: |aᵢ|² == [½, 0, 0, ½] within 1e-9.

use aria_verify_core::{banner, harness, resolve, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "bell",
        "Bell state (|00⟩+|11⟩)/√2 — exact statevector probabilities",
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("bell.aria", "Bell", &[])?;
    let (payload, _) =
        harness::execute_report(transport, lowered.ir, harness::AppMode::Statevector, &[])?;
    let probs: Vec<f64> = payload
        .chunks_exact(2)
        .map(|c| c[0] * c[0] + c[1] * c[1])
        .collect();

    let classical = vec![0.5, 0.0, 0.0, 0.5];
    Ok(banner::report_values(
        "bell",
        "|aᵢ|² of H·CX|00⟩",
        &probs,
        "(½,0,0,½)",
        &classical,
        1e-9,
    ))
}
