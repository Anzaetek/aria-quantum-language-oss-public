// SPDX-License-Identifier: Apache-2.0
//! deutsch_jozsa — decide constant vs balanced in one query.
//!
//! WHAT: classify f(x)=x₀⊕…⊕x_{n-1} as constant or balanced.
//! QUANTUM: run deutsch_jozsa.aria; for a balanced f the query register reads
//!   all-ones (non-zero ⇒ balanced) — read via omega_app classical_bits.
//! CLASSICAL: evaluate the truth table — parity is balanced, signature = 2ⁿ−1.
//! CHECK: measured register == 2ⁿ−1, exactly (and ≠ 0 ⇒ balanced).

use aria_verify_core::{banner, harness, resolve, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let n: i64 = 3;
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "deutsch_jozsa",
        "classify f(x)=parity(x) as constant or balanced (one query)",
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("deutsch_jozsa.aria", "DeutschJozsa", &[("n", n)])?;
    let (_payload, value) = harness::execute_report(
        transport,
        lowered.ir,
        harness::AppMode::ClassicalBits { seed: 7 },
        &[],
    )?;
    // Classical truth table: parity over n bits is balanced ⇒ all-ones signature.
    let expected = (1u64 << n) - 1;
    println!(
        "  measured register {} ⇒ {}",
        value as u64,
        if value as u64 != 0 {
            "BALANCED"
        } else {
            "CONSTANT"
        }
    );

    Ok(banner::report_exact_u64(
        "deutsch_jozsa",
        "measured register (balanced ⇒ all-ones)",
        value as u64,
        "truth table ⇒ 2ⁿ−1",
        expected,
        n as usize,
    ))
}
