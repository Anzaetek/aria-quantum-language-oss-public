// SPDX-License-Identifier: Apache-2.0
//! Logical QFT at the logical-channel altitude.
//!
//! The QFT's controlled-phase gates are not transversal, so we simulate the
//! ideal K-qubit logical unitary directly and check the two defining numeric
//! facts: `QFT|input⟩` is uniform over the computational basis, and
//! `QFT ∘ QFT⁻¹ = identity` (the round trip recovers the input exactly).

use aria_qec::logical::{grover_success_prob, qft_distribution, qft_roundtrip_distribution};
use aria_verify_core::{banner, Transport, Verdict};

pub fn run(_transport: Transport) -> Result<Verdict, String> {
    banner::header(
        "qec-qft",
        "logical QFT: QFT|input⟩ uniform + QFT∘QFT⁻¹ = identity",
        "native (aria-qec, statevector)",
    );

    let n = 4;
    let mut quantum: Vec<f64> = Vec::new();
    let mut ideal: Vec<f64> = Vec::new();

    // Round trip recovers each input basis state with probability 1.
    for input in [0u64, 5, 11, 15] {
        let rt = qft_roundtrip_distribution(n, input);
        let recovered = grover_success_prob(&rt, input);
        println!("  QFT∘QFT⁻¹ |{input:>2}⟩ → recovered prob {recovered:.9}");
        quantum.push(recovered);
        ideal.push(1.0);
    }

    // QFT of a basis state is uniform: max deviation from 1/2^n.
    let uni = qft_distribution(n, 5);
    let unit = 1.0 / (1u64 << n) as f64;
    let maxdev = (0u64..(1 << n))
        .map(|v| (uni.get(&v).copied().unwrap_or(0.0) - unit).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "  QFT|5⟩ uniform max-deviation from 1/{} = {maxdev:.2e}",
        1u64 << n
    );
    quantum.push(maxdev);
    ideal.push(0.0);

    Ok(banner::report_values(
        "qec-qft",
        "roundtrip recovery (4 inputs) + uniformity",
        &quantum,
        "ideal 1 / 0",
        &ideal,
        1e-9,
    ))
}
