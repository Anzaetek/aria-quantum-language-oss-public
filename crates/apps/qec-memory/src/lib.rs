// SPDX-License-Identifier: Apache-2.0
//! Surface-code logical memory under a hardware-flavored noise model.
//!
//! A code-capacity Monte-Carlo of the rotated surface code under a neutral-atom
//! (ZZ-biased) Pauli channel. Two numeric facts, both seeded/reproducible:
//! (1) **distance suppression** — the logical-error rate at `d=5` is below `d=3`
//! (the sub-threshold signature); (2) the **extracted effective logical channel**
//! is phase-biased (`p_lz > p_lx`), reflecting the ZZ bias.

use aria_qec::ecc::codes::SurfaceCode;
use aria_qec::logical::{extract_surface_memory_channel, surface_memory_rate, NoiseModel};
use aria_verify_core::{banner, Transport, Verdict};

pub fn run(_transport: Transport) -> Result<Verdict, String> {
    banner::header(
        "qec-memory",
        "surface-code memory: pL(d=5) < pL(d=3) + ZZ-biased effective channel",
        "native (aria-qec code-capacity Monte-Carlo)",
    );

    let noise = NoiseModel::neutral_atom(0.06);
    // 8k shots keeps the demo CI-fast (~15s) while the suppression gap
    // (pL≈0.063 vs 0.044) and the phase bias (p_lz≈0.038 vs p_lx≈0.006) are both
    // many Monte-Carlo sigmas wide, so the two inequalities hold decisively.
    let shots = 8_000u32;
    let seed = 7u64;

    let r3 = surface_memory_rate(&SurfaceCode::new(3), &noise, shots, seed);
    let r5 = surface_memory_rate(&SurfaceCode::new(5), &noise, shots, seed);
    let ch = extract_surface_memory_channel(&SurfaceCode::new(5), &noise, shots, seed);

    println!("  neutral-atom p=0.06, {shots} shots:");
    println!("    d=3 logical_rate = {r3:.6}");
    println!("    d=5 logical_rate = {r5:.6}   (must be < d=3)");
    println!(
        "    d=5 effective channel: p_lx={:.6} p_ly={:.6} p_lz={:.6}   (p_lz must exceed p_lx)",
        ch.p_lx, ch.p_ly, ch.p_lz
    );

    // Encode the two inequalities as non-negative margins that must be 0:
    //   (r5 − r3).max(0) = 0  ⟺  d=5 suppresses below d=3
    //   (p_lx − p_lz).max(0) = 0  ⟺  the channel is phase (ZZ) biased
    let suppression_violation = (r5 - r3).max(0.0);
    let bias_violation = (ch.p_lx - ch.p_lz).max(0.0);

    Ok(banner::report_values(
        "qec-memory",
        "[distance-suppression violation, phase-bias violation]",
        &[suppression_violation, bias_violation],
        "both zero",
        &[0.0, 0.0],
        1e-9,
    ))
}
