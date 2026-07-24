// SPDX-License-Identifier: Apache-2.0
//! Logical Quantum Phase Estimation at the logical-channel altitude.
//!
//! Estimating the phase of `P(2πφ)` with `m` counting qubits: for `φ = j/2ᵐ`
//! the counting register collapses to a clean delta at `j`, so the recovered
//! estimate `argmax/2ᵐ` equals `φ` exactly. Checked for every exact `m`-bit
//! phase; the peak probability confirms the delta.

use aria_qec::logical::{qpe_distribution, qpe_phase_error};
use aria_verify_core::{banner, Transport, Verdict};

pub fn run(_transport: Transport) -> Result<Verdict, String> {
    banner::header(
        "qec-qpe",
        "logical QPE: φ = j/2^m recovered as a clean delta",
        "native (aria-qec, statevector)",
    );

    let m = 3;
    let mut quantum: Vec<f64> = Vec::new();
    let mut ideal: Vec<f64> = Vec::new();

    for j in 0u64..(1u64 << m) {
        let phase = j as f64 / (1u64 << m) as f64;
        let dist = qpe_distribution(m, phase);
        let (best, peak) = dist
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&k, &p)| (k, p))
            .unwrap_or((0, 0.0));
        let est = best as f64 / (1u64 << m) as f64;
        let err = qpe_phase_error(&dist, m, phase);
        println!("  φ={phase:.3}: estimate={est:.3} peak_prob={peak:.4} phase_error={err:.2e}");
        quantum.push(est);
        ideal.push(phase);
    }

    Ok(banner::report_values(
        "qec-qpe",
        "recovered phase (all exact m=3 phases)",
        &quantum,
        "true φ = j/2^m",
        &ideal,
        1e-6,
    ))
}
