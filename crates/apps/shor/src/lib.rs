// SPDX-License-Identifier: Apache-2.0
//! shor — Shor's algorithm factors N = 15 with base a = 11 (compiled, order-2 case).
//!
//! WHAT: quantum period-finding on U|y⟩ = |11·y mod 15⟩ with a t = 3 counting register, then the
//!   classical post-processing (continued fractions → period → gcd) that turns the measured phase
//!   into factors of 15.
//! QUANTUM: run shor.aria, read the exact statevector, marginalize onto the counting register.
//! CLASSICAL: the true multiplicative order of 11 mod 15 is r = 2, and 15 = 3 · 5.
//! CHECK: the counting distribution peaks exactly at k ∈ {0, 4} (phases {0, 1/2}); the non-trivial
//!   peak gives r = 2^t / gcd(k, 2^t) = 2, and gcd(11^{r/2} ± 1, 15) = {3, 5} — the factors. Every
//!   quantity is deterministic, so the check is exact (tol 1e-9).

use aria_verify_core::{banner, harness, resolve, Transport, Verdict};

const N: u64 = 15;
const A: u64 = 11;
const T: u32 = 3; // counting qubits

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "shor",
        "Shor factoring of N=15 (a=11): counting-register phase → period r → factors {3,5}",
        &transport.label(guest),
    );

    // Exact statevector; qubit order is c[0..2] then w[0..3], with c[0] the low bit.
    let lowered = harness::load_lowered("shor.aria", "Shor15", &[])?;
    let (flat, _) =
        harness::execute_report(transport, lowered.ir, harness::AppMode::Statevector, &[])?;
    let amps: Vec<(f64, f64)> = flat.chunks_exact(2).map(|c| (c[0], c[1])).collect();

    // Marginal probability over the counting register (low T bits of the state index).
    let dim = 1usize << T;
    let mut marg = vec![0.0_f64; dim];
    for (i, (re, im)) in amps.iter().enumerate() {
        marg[i & (dim - 1)] += re * re + im * im;
    }
    let peaks: Vec<usize> = (0..dim).filter(|&k| marg[k] > 1e-6).collect();
    println!(
        "  counting peaks: {:?}  (probs {:?})",
        peaks,
        peaks
            .iter()
            .map(|&k| (marg[k] * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );

    // Non-trivial peak → phase → period r = 2^t / gcd(k, 2^t).
    let k = *peaks
        .iter()
        .find(|&&k| k != 0)
        .ok_or("no non-trivial peak found")?;
    let r = (dim as u64) / gcd(k as u64, dim as u64);
    // Even r ⇒ factors gcd(a^{r/2} ± 1, N).
    let half = A.pow((r / 2) as u32) % N;
    let mut f = [gcd(half + 1, N), gcd((half + N - 1) % N, N)];
    f.sort_unstable();
    println!(
        "  measured phase = {k}/{dim} = {:.3}  ⇒  r = {r}  ⇒  factors {{{}, {}}}",
        k as f64 / dim as f64,
        f[0],
        f[1]
    );

    // Classical oracle: 15 = 3 · 5, and the recovered period is 2.
    let quantum = vec![k as f64 / dim as f64, r as f64, f[0] as f64, f[1] as f64];
    let classical = vec![0.5, 2.0, 3.0, 5.0];
    Ok(banner::report_values(
        "shor",
        "phase, r, factors (quantum)",
        &quantum,
        "1/2, 2, {3,5} (classical)",
        &classical,
        1e-9,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_and_factors_of_11_mod_15() {
        // 11^2 = 121 = 1 mod 15 ⇒ order 2
        assert_eq!(A.pow(2) % N, 1);
        assert_eq!(gcd(11 + 1, 15), 3);
        assert_eq!(gcd(11 - 1, 15), 5);
    }
}
