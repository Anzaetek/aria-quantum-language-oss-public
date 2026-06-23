// SPDX-License-Identifier: Apache-2.0
//! grover3 — amplitude amplification finds a marked item.
//!
//! WHAT: Grover search on 3 qubits (8 items) for a marked index.
//! QUANTUM: run grover3.aria for k=2 iterations, sample 8192 shots, read the
//!   probability of the marked outcome (omega_app counts mode).
//! CLASSICAL: the analytic Grover success probability sin²((2k+1)θ),
//!   θ = arcsin(1/√8), and the marked index itself (which the search must
//!   return as the most-likely outcome).
//! CHECK: argmax == marked (exact) AND P(marked) ≈ analytic within ±0.05.

use aria_verify_core::{banner, harness, resolve, util, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let marked: u64 = 5;
    let k = 2.0_f64;
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "grover3",
        &format!("Grover search on 8 items for the marked index {marked}"),
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("grover3.aria", "Grover3", &[("marked", marked as i64)])?;
    let (payload, _) = harness::execute_report(
        transport,
        lowered.ir,
        harness::AppMode::Counts { shots: 8192 },
        &[],
    )?;
    let counts = util::counts_from_payload(&payload);
    let amax = util::argmax_outcome(&counts);
    let p_marked = util::prob_of(&counts, marked);

    let theta = (1.0 / 8.0_f64.sqrt()).asin();
    let p_analytic = ((2.0 * k + 1.0) * theta).sin().powi(2);

    println!("  most-likely outcome = {amax} (marked = {marked})");
    if amax != marked {
        return Err(format!(
            "grover did not return the marked item (got {amax}, want {marked})"
        ));
    }
    Ok(banner::report_scalar(
        "grover3",
        "measured P(marked), 8192 shots",
        p_marked,
        "analytic sin²((2k+1)θ)",
        p_analytic,
        0.05,
    ))
}
