//! Effective logical Pauli channel extraction + logical-level composition.
//!
//! The two-level validation methodology of arXiv:2509.18294 (Fig. 2): instead of
//! expanding every logical qubit to its physical data qubits for a long
//! computation, *extract* the effective logical Pauli channel of a gadget once
//! (by running the physical gadget under noise + decoding, and measuring the
//! residual logical Pauli distribution), then compose those channels cheaply at
//! the logical level. Here we extract a surface-code **memory round** channel
//! and validate that composing it over `R` rounds reproduces the physical
//! multi-round logical-error rate.

use crate::ecc::codes::SurfaceCode;

use super::memory::memory_round_outcome;
use super::noise::{splitmix64, NoiseModel};

/// Effective logical Pauli channel of a gadget: the probabilities that it
/// induces a logical X, Y (= both), or Z error. `p_i = 1 - (p_lx+p_ly+p_lz)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EffectiveLogicalChannel {
    pub p_lx: f64,
    pub p_ly: f64,
    pub p_lz: f64,
}

impl EffectiveLogicalChannel {
    /// Total logical error probability.
    pub fn total(&self) -> f64 {
        self.p_lx + self.p_ly + self.p_lz
    }
}

/// Extract the effective logical channel of one surface-code memory round under
/// `noise` by Monte-Carlo: classify each round's residual as logical
/// X-only / Z-only / both (Y).
pub fn extract_surface_memory_channel(
    code: &SurfaceCode,
    noise: &NoiseModel,
    shots: u32,
    seed: u64,
) -> EffectiveLogicalChannel {
    let x_checks = code.x_checks().to_vec();
    let z_checks = code.z_checks().to_vec();
    // Same seed derivation as surface_memory_rate ⇒ identical error draws, so
    // the channel's total error equals the single-round failure rate exactly.
    let mut rng = seed ^ 0xA5A5_5A5A_DEAD_BEEF;
    let (mut nx, mut ny, mut nz) = (0u32, 0u32, 0u32);
    for _ in 0..shots {
        let (xf, zf) = memory_round_outcome(code, &x_checks, &z_checks, noise, &mut rng);
        match (xf, zf) {
            (true, true) => ny += 1,
            (true, false) => nx += 1,
            (false, true) => nz += 1,
            (false, false) => {}
        }
    }
    let n = shots as f64;
    EffectiveLogicalChannel {
        p_lx: nx as f64 / n,
        p_ly: ny as f64 / n,
        p_lz: nz as f64 / n,
    }
}

/// Physical multi-round logical-error rate: run `rounds` physical memory rounds
/// (each decoded), XOR the per-round logical flips into a running frame, and
/// count a failure if the net logical frame is non-trivial.
pub fn surface_memory_rate_rounds(
    code: &SurfaceCode,
    noise: &NoiseModel,
    rounds: usize,
    shots: u32,
    seed: u64,
) -> f64 {
    let x_checks = code.x_checks().to_vec();
    let z_checks = code.z_checks().to_vec();
    let mut rng = seed ^ 0xA5A5_5A5A_DEAD_BEEF;
    let mut failures = 0u32;
    for _ in 0..shots {
        let (mut fx, mut fz) = (false, false);
        for _ in 0..rounds {
            let (xf, zf) = memory_round_outcome(code, &x_checks, &z_checks, noise, &mut rng);
            fx ^= xf;
            fz ^= zf;
        }
        if fx || fz {
            failures += 1;
        }
    }
    failures as f64 / shots as f64
}

/// Logical-level composition: apply the extracted channel `rounds` times to a
/// single logical qubit (sampling a logical X/Y/Z each round into a frame),
/// count a failure if the net frame is non-trivial. Cheap — no physical qubits.
pub fn channel_rate_rounds(
    ch: &EffectiveLogicalChannel,
    rounds: usize,
    shots: u32,
    seed: u64,
) -> f64 {
    let mut rng = seed ^ 0x0C0F_FEE0_1234_5678;
    let mut failures = 0u32;
    for _ in 0..shots {
        let (mut fx, mut fz) = (false, false);
        for _ in 0..rounds {
            let r = splitmix64(&mut rng);
            // Partition [0,1): X, then Y, then Z, then identity.
            if r < ch.p_lx {
                fx ^= true;
            } else if r < ch.p_lx + ch.p_ly {
                fx ^= true;
                fz ^= true;
            } else if r < ch.total() {
                fz ^= true;
            }
        }
        if fx || fz {
            failures += 1;
        }
    }
    failures as f64 / shots as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracted_channel_total_matches_memory_rate() {
        // The channel's total logical error == single-round memory failure rate.
        let code = SurfaceCode::new(3);
        let noise = NoiseModel::neutral_atom(0.06);
        let ch = extract_surface_memory_channel(&code, &noise, 60_000, 3);
        let direct = super::super::memory::surface_memory_rate(&code, &noise, 60_000, 3);
        assert!((ch.total() - direct).abs() < 1e-9, "{} vs {}", ch.total(), direct);
    }

    #[test]
    fn neutral_atom_channel_is_phase_biased() {
        // ZZ-biased noise ⇒ logical Z (phase) errors dominate logical X.
        let code = SurfaceCode::new(3);
        let ch = extract_surface_memory_channel(&code, &NoiseModel::neutral_atom(0.08), 80_000, 5);
        assert!(ch.p_lz > ch.p_lx, "p_lz {} should exceed p_lx {}", ch.p_lz, ch.p_lx);
    }

    #[test]
    fn channel_composition_matches_physical_multiround() {
        // Extract the 1-round channel, then predict the R-round logical rate from
        // logical-level composition; it must match a direct physical R-round
        // simulation within Monte-Carlo error (two-level validation).
        let code = SurfaceCode::new(3);
        let noise = NoiseModel::depolarizing(0.05);
        let rounds = 8;
        let shots = 40_000u32;

        let ch = extract_surface_memory_channel(&code, &noise, 80_000, 9);
        let predicted = channel_rate_rounds(&ch, rounds, shots, 21);
        let physical = surface_memory_rate_rounds(&code, &noise, rounds, shots, 21);

        // 3σ binomial tolerance on `shots`.
        let se = (physical * (1.0 - physical) / shots as f64).sqrt().max(1e-4);
        assert!(
            (predicted - physical).abs() < 3.0 * se + 3e-3,
            "channel {predicted} vs physical {physical} (3σ={:.4})",
            3.0 * se
        );
    }
}
