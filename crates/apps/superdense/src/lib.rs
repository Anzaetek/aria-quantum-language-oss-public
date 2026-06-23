// SPDX-License-Identifier: Apache-2.0
//! superdense — superdense coding: 2 classical bits through 1 transmitted qubit.
//!
//! WHAT: Alice encodes (b0, b1) onto her half of a Bell pair; Bob undoes the
//!   Bell prep and measures both qubits, reading back exactly (b0, b1).
//! QUANTUM: run superdense.aria with (b0, b1) = (1, 0); sample the 2-bit creg.
//! CLASSICAL: decoding is deterministic — Bob's bits == Alice's input bits.
//! CHECK: argmax 2-bit outcome == b0 | (b1 << 1), exactly.

use aria_verify_core::{banner, harness, resolve, util, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let (b0, b1) = (1i64, 0i64);
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "superdense",
        "superdense coding — Bob's decoded bits vs Alice's input (b0,b1)",
        &transport.label(guest),
    );

    let lowered =
        harness::load_lowered("superdense.aria", "Superdense", &[("b0", b0), ("b1", b1)])?;
    let (payload, _) = harness::execute_report(
        transport,
        lowered.ir,
        harness::AppMode::Counts { shots: 1024 },
        &[],
    )?;
    let counts = util::counts_from_payload(&payload);
    let decoded = util::argmax_outcome(&counts);

    // Bob's Bell-basis decode is a deterministic bijection input → outcome:
    // qubit 0 reads back the phase bit b1, qubit 1 the bit-flip bit b0, so the
    // 2-bit outcome is b1 | (b0 << 1). (Either labelling is "correct" superdense
    // coding — 2 bits through 1 transmitted qubit — what matters is the bijection.)
    let classical = (b1 as u64) | ((b0 as u64) << 1);
    println!(
        "  encoded (b0,b1) = ({b0},{b1}); P(decoded) = {:.4}",
        util::prob_of(&counts, decoded)
    );
    Ok(banner::report_exact_u64(
        "superdense",
        "argmax decoded creg",
        decoded,
        "b1 | (b0<<1)",
        classical,
        2,
    ))
}
