//! Partial-distinguishability SLOS — photons that are neither perfectly
//! indistinguishable nor fully distinguishable.
//!
//! Plain [`crate::slos::slos_full`] assumes perfect indistinguishability: the
//! output probability is `|Per(U_{s→t})|² / (∏ sᵢ! ∏ tⱼ!)`, so interference
//! (Hong–Ou–Mandel bunching) is maximal. Real photons carry internal degrees of
//! freedom (spectral / temporal / polarisation) that make them partially
//! distinguishable, controlled by an indistinguishability `η ∈ [0,1]`:
//!   * `η = 1` — perfectly indistinguishable (the usual SLOS, full interference);
//!   * `η = 0` — fully distinguishable (classical particles, no interference):
//!     the probability is `Per(|U_{s→t}|²) / (∏ sᵢ! ∏ tⱼ!)`.
//!
//! For the two-photon regime the partial-distinguishability output is the exact
//! convex mixture of the two limits (the interference term is weighted by the
//! pairwise overlap `η = |⟨φ₁|φ₂⟩|²`):
//!
//! ```text
//!   P_η(t) = [ (1−η)·Per(|U_{s→t}|²) + η·|Per(U_{s→t})|² ] / (∏ sᵢ! ∏ tⱼ!)
//! ```
//!
//! This is exact for n ≤ 2 photons (the regime of the quantum-enhanced-kernel
//! experiments) and the standard two-source mean-field model beyond it. The
//! photonic quantum-enhanced kernel `κ(x₁,x₂) = P_η(no-collision)` is then a
//! tunable function of `η`, recovering the classical kernel at `η = 0`.

use num_complex::Complex64;

use crate::permanent::permanent;
use crate::slos::{enumerate_fock_states, total_photons, FockState};

fn factorial(n: u32) -> f64 {
    (1..=n as u64).product::<u64>() as f64
}

fn fock_to_modes(state: &FockState) -> Vec<usize> {
    let mut modes = Vec::new();
    for (mode, &count) in state.iter().enumerate() {
        for _ in 0..count {
            modes.push(mode);
        }
    }
    modes
}

fn scattering_submatrix(
    u: &[Vec<Complex64>],
    rows: &[usize],
    cols: &[usize],
) -> Vec<Vec<Complex64>> {
    rows.iter()
        .map(|&r| cols.iter().map(|&c| u[r][c]).collect())
        .collect()
}

/// Probability of output Fock state `output` for input `input` at
/// indistinguishability `eta` (clamped to `[0,1]`).
pub fn partial_probability(
    u: &[Vec<Complex64>],
    input: &FockState,
    output: &FockState,
    eta: f64,
) -> f64 {
    let eta = eta.clamp(0.0, 1.0);
    if total_photons(input) != total_photons(output) {
        return 0.0;
    }
    let cols = fock_to_modes(input);
    let rows = fock_to_modes(output);
    let n = cols.len();
    if n == 0 {
        return 1.0; // vacuum → vacuum
    }
    let sub = scattering_submatrix(u, &rows, &cols);
    // Indistinguishable term: |Per(sub)|².
    let p_indist = permanent(&sub).norm_sqr();
    // Distinguishable term: Per(|sub|²) (entrywise modulus-squared, real).
    let sub_abs2: Vec<Vec<Complex64>> = sub
        .iter()
        .map(|row| {
            row.iter()
                .map(|z| Complex64::new(z.norm_sqr(), 0.0))
                .collect()
        })
        .collect();
    let p_dist = permanent(&sub_abs2).re;
    let norm: f64 = input
        .iter()
        .chain(output.iter())
        .map(|&k| factorial(k))
        .product();
    ((1.0 - eta) * p_dist + eta * p_indist) / norm
}

/// Full output distribution at indistinguishability `eta`. Probabilities sum to
/// 1 for a unitary `u` (both limits are normalised, so any convex mixture is).
pub fn slos_partial(u: &[Vec<Complex64>], input: &FockState, eta: f64) -> Vec<(FockState, f64)> {
    let m = u.len();
    let n = total_photons(input);
    enumerate_fock_states(m, n)
        .into_iter()
        .filter_map(|t| {
            let p = partial_probability(u, input, &t, eta);
            (p > 1e-15).then_some((t, p))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{self, PhotonicOp};
    use std::f64::consts::FRAC_PI_4;

    fn bs5050() -> Vec<Vec<Complex64>> {
        components::build_unitary(
            2,
            &[PhotonicOp::BeamSplitterRx {
                mode0: 0,
                mode1: 1,
                theta: FRAC_PI_4,
                phi: 0.0,
            }],
        )
    }

    #[test]
    fn eta_one_recovers_hom_dip() {
        // Indistinguishable: P(1,1)=0, P(2,0)=P(0,2)=0.5.
        let u = bs5050();
        assert!(partial_probability(&u, &vec![1, 1], &vec![1, 1], 1.0) < 1e-12);
        assert!((partial_probability(&u, &vec![1, 1], &vec![2, 0], 1.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn eta_zero_is_classical() {
        // Distinguishable particles on a 50:50 BS: P(1,1)=0.5, P(2,0)=P(0,2)=0.25.
        let u = bs5050();
        assert!((partial_probability(&u, &vec![1, 1], &vec![1, 1], 0.0) - 0.5).abs() < 1e-12);
        assert!((partial_probability(&u, &vec![1, 1], &vec![2, 0], 0.0) - 0.25).abs() < 1e-12);
        assert!((partial_probability(&u, &vec![1, 1], &vec![0, 2], 0.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn coincidence_scales_linearly_with_eta() {
        // The coincidence P(1,1) = 0.5·(1−η): full dip at η=1, classical 0.5 at η=0.
        let u = bs5050();
        for &eta in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let p11 = partial_probability(&u, &vec![1, 1], &vec![1, 1], eta);
            assert!((p11 - 0.5 * (1.0 - eta)).abs() < 1e-12, "eta={eta}: {p11}");
        }
    }

    #[test]
    fn distribution_normalised_any_eta() {
        let u = components::build_unitary(
            3,
            &[
                PhotonicOp::BeamSplitterRx {
                    mode0: 0,
                    mode1: 1,
                    theta: 0.7,
                    phi: 0.3,
                },
                PhotonicOp::BeamSplitterRx {
                    mode0: 1,
                    mode1: 2,
                    theta: 0.9,
                    phi: 0.0,
                },
            ],
        );
        for &eta in &[0.0, 0.5, 0.972, 1.0] {
            let total: f64 = slos_partial(&u, &vec![1, 1, 0], eta)
                .iter()
                .map(|(_, p)| p)
                .sum();
            assert!((total - 1.0).abs() < 1e-10, "eta={eta}: Σp={total}");
        }
    }
}
