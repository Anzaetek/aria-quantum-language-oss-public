// SPDX-License-Identifier: Apache-2.0
//! Dump a rotated surface code + a seeded batch of code-capacity error trials,
//! each already decoded by aria-qec's exact minimum-weight decoder, as
//! newline-delimited JSON. Consumed by `tools/qec_cross_check/check_decoder.py`,
//! which re-decodes the *identical* error samples with PyMatching (the
//! field-standard MWPM decoder) and asserts the two decoders agree on the
//! logical outcome shot-for-shot.
//!
//! Usage: cargo run -p aria-qec --example dump_surface_code -- <d> <p> <shots> <seed>
//!   line 1        : header JSON (code structure + params + our logical rates)
//!   lines 2..N+1  : one trial "x_err;z_err;our_x_flip;our_z_flip"
//!                   (x_err / z_err are space-separated data-qubit indices)

use std::collections::BTreeSet;

use aria_qec::ecc::codes::SurfaceCode;
use aria_qec::ecc::mwpm::decode_mwpm_correction;
use aria_qec::logical::noise::NoiseModel;

fn parity(error: &[usize], checks: &[Vec<usize>]) -> Vec<u8> {
    let eset: BTreeSet<usize> = error.iter().copied().collect();
    checks
        .iter()
        .map(|c| (c.iter().filter(|q| eset.contains(q)).count() % 2) as u8)
        .collect()
}

fn sym_diff(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut set: BTreeSet<usize> = a.iter().copied().collect();
    for &q in b {
        if !set.remove(&q) {
            set.insert(q);
        }
    }
    set.into_iter().collect()
}

fn odd_overlap(error: &[usize], op: &BTreeSet<usize>) -> bool {
    error.iter().filter(|q| op.contains(q)).count() % 2 == 1
}

fn jarr(v: &[usize]) -> String {
    let items: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", items.join(","))
}

fn jarr2(v: &[Vec<usize>]) -> String {
    let items: Vec<String> = v.iter().map(|c| jarr(c)).collect();
    format!("[{}]", items.join(","))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let p: f64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.05);
    let shots: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(7);

    let code = SurfaceCode::new(d);
    let x_checks = code.x_checks().to_vec();
    let z_checks = code.z_checks().to_vec();
    let lx: BTreeSet<usize> = code.logical_x().into_iter().collect();
    let lz: BTreeSet<usize> = code.logical_z().into_iter().collect();
    // Independent per-qubit X/Z channel at rate p (the code-capacity model the
    // two CSS sectors decode independently), matching NoiseModel::depolarizing.
    let noise = NoiseModel::depolarizing(p);

    // Header.
    println!(
        "{{\"d\":{},\"n_data\":{},\"x_checks\":{},\"z_checks\":{},\"logical_x\":{},\"logical_z\":{},\"p_bit\":{},\"p_phase\":{},\"shots\":{}}}",
        d,
        code.n_data(),
        jarr2(&x_checks),
        jarr2(&z_checks),
        jarr(&code.logical_x()),
        jarr(&code.logical_z()),
        noise.p_bit,
        noise.p_phase,
        shots,
    );

    let mut rng = seed ^ 0xA5A5_5A5A_DEAD_BEEF;
    for _ in 0..shots {
        let (x_err, z_err) = noise.sample_data_errors(code.n_data(), &mut rng);
        // Decoder input order: X-checks (detect Z), then Z-checks (detect X).
        let mut full = parity(&z_err, &x_checks);
        full.extend(parity(&x_err, &z_checks));
        let corr = decode_mwpm_correction(&code, &full);
        let residual_x = sym_diff(&x_err, &corr.x_flips);
        let residual_z = sym_diff(&z_err, &corr.z_flips);
        let our_x_flip = odd_overlap(&residual_x, &lz) as u8; // residual X vs logical Z
        let our_z_flip = odd_overlap(&residual_z, &lx) as u8; // residual Z vs logical X
        let xe: Vec<String> = x_err.iter().map(|q| q.to_string()).collect();
        let ze: Vec<String> = z_err.iter().map(|q| q.to_string()).collect();
        println!(
            "{};{};{};{}",
            xe.join(" "),
            ze.join(" "),
            our_x_flip,
            our_z_flip
        );
    }
}
