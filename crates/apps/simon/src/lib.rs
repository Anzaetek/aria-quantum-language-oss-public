// SPDX-License-Identifier: Apache-2.0
//! simon — Simon's algorithm: each sample y is orthogonal to the hidden s.
//!
//! WHAT: the baked oracle has period s = 1ⁿ (all ones). The algorithm
//!   guarantees every measured query string y satisfies y·s = 0 (mod 2);
//!   for s = 1ⁿ that means y has EVEN Hamming weight.
//! QUANTUM: run simon.aria (n = 3), sample the n-bit query creg many times.
//! CLASSICAL: every y must satisfy y·s = 0 ⇒ the satisfied fraction is 1.0.
//! CHECK: fraction(y·s == 0) == 1.0 within 1e-9.

use aria_verify_core::{banner, harness, resolve, util, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let n: i64 = 3;
    let s: u64 = (1u64 << n) - 1; // baked period s = 1ⁿ
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "simon",
        "Simon's algorithm — every measured y is ⊥ to the hidden period s",
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("simon.aria", "Simon", &[("n", n)])?;
    let (payload, _) = harness::execute_report(
        transport,
        lowered.ir,
        harness::AppMode::Counts { shots: 4096 },
        &[],
    )?;
    let counts = util::counts_from_payload(&payload);
    let total = util::total_shots(&counts) as f64;
    let ok_shots: f64 = counts
        .iter()
        .filter(|(y, _)| ((y & s).count_ones() & 1) == 0) // y·s = 0 (mod 2)
        .map(|(_, c)| *c as f64)
        .sum();
    let fraction = ok_shots / total.max(1.0);

    println!(
        "  s = {s:0width$b}, distinct y's = {}",
        counts.len(),
        width = n as usize
    );
    Ok(banner::report_scalar(
        "simon",
        "fraction of samples with y·s = 0",
        fraction,
        "all y ⊥ s",
        1.0,
        1e-9,
    ))
}
