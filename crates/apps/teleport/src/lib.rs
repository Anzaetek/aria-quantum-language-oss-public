// SPDX-License-Identifier: Apache-2.0
//! teleport — move an unknown qubit via a Bell pair + 2 classical bits.
//!
//! WHAT: Bob's output qubit equals Alice's input.
//! QUANTUM: prepare Alice's payload |1⟩, run teleport.aria (mid-circuit measure
//!   then classically-conditioned corrections), then measure Bob's qubit q[2].
//!   Repeat over 8 measurement trajectories (omega_app classical_bits mode,
//!   Collapse semantics).
//! CLASSICAL: teleportation is exact ⇒ q[2] must equal the input on EVERY
//!   trajectory.
//! CHECK: fraction of trajectories with q[2]==input is 1.0, exactly.

use aria_verify_core::{banner, harness, resolve, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let input_bit: u64 = 1;
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "teleport",
        &format!("teleport |{input_bit}⟩ from Alice (q0) to Bob (q2)"),
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("teleport.aria", "Teleport", &[])?;
    let mut ir = lowered.ir.clone();
    harness::prepend_basis_state(&mut ir, input_bit); // Alice's payload = |input_bit⟩
    harness::append_measure(&mut ir, 2, 2); // read Bob's qubit into cbit 2

    let trajectories = 8;
    let mut matched = 0;
    for seed in 0..trajectories {
        let (payload, _) = harness::execute_report(
            transport,
            ir.clone(),
            harness::AppMode::ClassicalBits { seed: seed as i64 },
            &[],
        )?;
        let bob = *payload.get(2).unwrap_or(&0.0) as u64;
        if bob == input_bit {
            matched += 1;
        }
    }
    println!("  {matched}/{trajectories} trajectories delivered q[2] = {input_bit}");

    Ok(banner::report_scalar(
        "teleport",
        "fraction of trajectories with q[2]==input",
        matched as f64 / trajectories as f64,
        "exact teleportation",
        1.0,
        0.0,
    ))
}
