//! Code-capacity logical-memory Monte-Carlo → logical-error-rate curves.
//!
//! A pure (no-backend) simulation of a surface-code *memory*: draw an
//! independent per-data-qubit Pauli error from a [`NoiseModel`], compute the two
//! CSS-sector syndromes algebraically, decode with the exact MWPM decoder
//! ([`decode_mwpm_correction`]), and classify a logical failure. This reproduces
//! the decode logic of [`crate::ecc::run::decode_trial`] without invoking a
//! quantum backend, so it runs enough shots to resolve sub-threshold logical
//! rates and their distance scaling.
//!
//! The backend-based path in `ecc::run` already proves a real simulator
//! reproduces these syndromes bit-for-bit; here we only need the (deterministic,
//! algebraic) syndrome, so we skip it for speed.

use std::collections::BTreeSet;

use crate::ecc::codes::SurfaceCode;
use crate::ecc::mwpm::decode_mwpm_correction;

use super::noise::NoiseModel;

/// Symmetric difference of two qubit-index sets (XOR of Pauli supports).
fn sym_diff(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut set: BTreeSet<usize> = a.iter().copied().collect();
    for &q in b {
        if !set.remove(&q) {
            set.insert(q);
        }
    }
    set.into_iter().collect()
}

/// Parity of an error support against a set of checks.
fn parity(error: &[usize], checks: &[Vec<usize>]) -> Vec<u8> {
    let eset: BTreeSet<usize> = error.iter().copied().collect();
    checks
        .iter()
        .map(|c| (c.iter().filter(|q| eset.contains(q)).count() % 2) as u8)
        .collect()
}

/// Logical flips of a residual error: `(logical_X_flip, logical_Z_flip)`. A
/// residual X error flips the logical qubit iff it anticommutes with logical Z
/// (odd overlap with its support); symmetrically for residual Z vs logical X.
fn logical_flips(code: &SurfaceCode, residual_x: &[usize], residual_z: &[usize]) -> (bool, bool) {
    let lz: BTreeSet<usize> = code.logical_z().into_iter().collect();
    let lx: BTreeSet<usize> = code.logical_x().into_iter().collect();
    let x_flip = residual_x.iter().filter(|q| lz.contains(q)).count() % 2 == 1;
    let z_flip = residual_z.iter().filter(|q| lx.contains(q)).count() % 2 == 1;
    (x_flip, z_flip)
}

/// One code-capacity memory round: sample errors, decode, return the logical
/// flips `(logical_X, logical_Z)` induced after correction. Shared by the
/// logical-rate estimator and the channel extractor.
pub(crate) fn memory_round_outcome(
    code: &SurfaceCode,
    x_checks: &[Vec<usize>],
    z_checks: &[Vec<usize>],
    noise: &NoiseModel,
    rng: &mut u64,
) -> (bool, bool) {
    let (x_err, z_err) = noise.sample_data_errors(code.n_data(), rng);
    // Full syndrome in decoder order: X-checks (detect Z), then Z-checks
    // (detect X) — matching ecc::run::decode_trial.
    let mut full = parity(&z_err, x_checks);
    full.extend(parity(&x_err, z_checks));
    let corr = decode_mwpm_correction(code, &full);
    let residual_x = sym_diff(&x_err, &corr.x_flips);
    let residual_z = sym_diff(&z_err, &corr.z_flips);
    logical_flips(code, &residual_x, &residual_z)
}

/// Estimate the logical-error rate of a surface-code memory under `noise`.
/// Errors are drawn from a seeded stream so the estimate is reproducible.
pub fn surface_memory_rate(code: &SurfaceCode, noise: &NoiseModel, shots: u32, seed: u64) -> f64 {
    let x_checks = code.x_checks().to_vec();
    let z_checks = code.z_checks().to_vec();
    let mut rng = seed ^ 0xA5A5_5A5A_DEAD_BEEF;
    let mut failures = 0u32;
    for _ in 0..shots {
        let (xf, zf) = memory_round_outcome(code, &x_checks, &z_checks, noise, &mut rng);
        if xf || zf {
            failures += 1;
        }
    }
    failures as f64 / shots as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical::metrics::{sub_threshold_slope, LogicalErrorCurve};

    #[test]
    fn ideal_noise_never_fails() {
        let code = SurfaceCode::new(3);
        let rate = surface_memory_rate(&code, &NoiseModel::ideal(), 5_000, 1);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn larger_distance_suppresses_logical_error() {
        // Below threshold, d=5 must beat d=3 under a neutral-atom (ZZ-biased)
        // channel. (d=7 suppression is covered by the demo's larger sweep.)
        let noise = NoiseModel::neutral_atom(0.06);
        let r3 = surface_memory_rate(&SurfaceCode::new(3), &noise, 40_000, 7);
        let r5 = surface_memory_rate(&SurfaceCode::new(5), &noise, 40_000, 7);
        assert!(r5 < r3, "d=5 ({r5}) should suppress below d=3 ({r3})");
    }

    #[test]
    fn sub_threshold_slope_matches_distance() {
        // For d=3, correcting ⌊(d-1)/2⌋=1 error, the leading logical rate scales
        // ~ p^2, so the log-log slope should be ≈ (d+1)/2 = 2.
        let d = 3;
        let code = SurfaceCode::new(d);
        let mut curve = LogicalErrorCurve::default();
        for &p in &[0.04_f64, 0.06, 0.09] {
            let noise = NoiseModel::depolarizing(p);
            let pl = surface_memory_rate(&code, &noise, 60_000, 11);
            curve.push(p, d, pl);
        }
        let slope = sub_threshold_slope(&curve, d).expect("slope");
        assert!(
            (1.3..=2.8).contains(&slope),
            "d=3 log-log slope = {slope}, expected ≈ 2"
        );
    }
}
