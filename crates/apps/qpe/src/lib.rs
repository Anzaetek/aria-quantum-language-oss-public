// SPDX-License-Identifier: Apache-2.0
//! qpe — quantum phase estimation recovers a baked eigenphase.
//!
//! WHAT: the template bakes φ = 1/8 into the controlled-phase cascade and runs
//!   an exact inverse-QFT, so the t-bit counting register collapses (prob 1)
//!   onto the binary expansion of φ.
//! QUANTUM: run qpe.aria (t = 3), sample the counting creg, take the argmax.
//! CLASSICAL: φ = 1/8 is exactly 3-bit representable ⇒ recovered φ̂ = 1/8.
//! CHECK: recovered φ̂ == 0.125 within 1e-9 (the outcome is deterministic).

use aria_verify_core::{banner, harness, resolve, util, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let t: i64 = 3;
    let phi_baked = 0.125;
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "qpe",
        "quantum phase estimation — recovered eigenphase φ̂ vs the baked φ = 1/8",
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("qpe.aria", "QPE", &[("t", t)])?;
    let (payload, _) = harness::execute_report(
        transport,
        lowered.ir,
        harness::AppMode::Counts { shots: 1024 },
        &[],
    )?;
    let counts = util::counts_from_payload(&payload);
    let key = util::argmax_outcome(&counts);

    // The counting register reads the binary fraction φ = 0.c₀c₁…c_{t-1}, i.e.
    // bit i of the outcome carries weight 2^-(i+1).
    let phi_hat: f64 = (0..t)
        .map(|i| ((key >> i) & 1) as f64 / (1u64 << (i + 1)) as f64)
        .sum();
    println!(
        "  argmax outcome = {key:0width$b}  (P = {:.4})",
        util::prob_of(&counts, key),
        width = t as usize
    );
    Ok(banner::report_scalar(
        "qpe",
        "recovered phase φ̂",
        phi_hat,
        "baked φ = 1/8",
        phi_baked,
        1e-9,
    ))
}
