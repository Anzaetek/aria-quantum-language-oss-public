// SPDX-License-Identifier: Apache-2.0
//! ghz — GHZ state (|000⟩+|111⟩)/√2 vs the analytic amplitudes.
//!
//! WHAT: prepare H·CX·CX|000⟩ and read the EXACT statevector.
//! QUANTUM: run ghz.aria through the omega runtime, statevector mode.
//! CLASSICAL: only |000⟩ and |111⟩ have weight ½ each; all others 0.
//! CHECK: |aᵢ|² == [½, 0, 0, 0, 0, 0, 0, ½] within 1e-9.

use aria_verify_core::{banner, harness, resolve, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "ghz",
        "GHZ state (|000⟩+|111⟩)/√2 — exact statevector probabilities",
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("ghz.aria", "GHZ", &[])?;
    let (payload, _) =
        harness::execute_report(transport, lowered.ir, harness::AppMode::Statevector, &[])?;
    let probs: Vec<f64> = payload
        .chunks_exact(2)
        .map(|c| c[0] * c[0] + c[1] * c[1])
        .collect();

    let mut classical = vec![0.0; 8];
    classical[0] = 0.5;
    classical[7] = 0.5;
    Ok(banner::report_values(
        "ghz",
        "|aᵢ|² of the 3-qubit GHZ state",
        &probs,
        "(½ at |000⟩, ½ at |111⟩)",
        &classical,
        1e-9,
    ))
}
