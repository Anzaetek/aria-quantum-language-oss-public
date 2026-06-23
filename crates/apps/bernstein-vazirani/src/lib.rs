// SPDX-License-Identifier: Apache-2.0
//! bernstein_vazirani — recover a hidden bit-string in one query.
//!
//! WHAT: recover the hidden string a from the oracle f(x) = a·x (mod 2).
//! QUANTUM: run bernstein_vazirani.aria once and read the measured register
//!   (omega_app classical_bits mode).
//! CLASSICAL: the hidden string a (the input we encoded).
//! CHECK: recovered == a, exactly.

use aria_verify_core::{banner, harness, resolve, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let n: i64 = 4;
    let a: u64 = 5; // 0101
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "bernstein_vazirani",
        &format!("recover hidden a={a} from f(x)=a·x mod 2, in one query"),
        &transport.label(guest),
    );

    let lowered = harness::load_lowered(
        "bernstein_vazirani.aria",
        "BernsteinVazirani",
        &[("n", n), ("a", a as i64)],
    )?;
    let (_payload, value) = harness::execute_report(
        transport,
        lowered.ir,
        harness::AppMode::ClassicalBits { seed: 7 },
        &[],
    )?;

    Ok(banner::report_exact_u64(
        "bernstein_vazirani",
        "recovered string",
        value as u64,
        "hidden a",
        a,
        n as usize,
    ))
}
