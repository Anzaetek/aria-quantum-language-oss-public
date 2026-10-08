// SPDX-License-Identifier: Apache-2.0
//! The transcription of arXiv:2503.18939v4 Theorem 1 (eqs. 8–9), checked
//! for the properties the paper states: a seamless regime switch at `L*`,
//! decay in the cutoff while `w₀ < N/e`, and the size regime where it is
//! informative at all.

use omega_backend_majoranaprop::{theorem1_mse_ratio, theorem1_mse_ratio_f};
use std::f64::consts::E;

#[test]
fn the_two_regimes_meet_at_l_star() {
    for (n, w0) in [(28.0, 4.0), (52.0, 6.0), (100.0, 8.0), (12.0, 2.0)] {
        let l_star = 2.0 * n * (5.0 * E * w0 * 4f64.powf(w0)).ln();
        let below = theorem1_mse_ratio_f(n, l_star, w0);
        let above = theorem1_mse_ratio_f(n, l_star + 1e-9, w0);
        assert!(
            (below - above).abs() < 1e-9 * below.max(1e-300),
            "N={n} w0={w0}: {below} vs {above} at L*={l_star}"
        );
        // eq. 9 at L* collapses to eq. 8's second term exactly.
        let eq8 = 2f64.powf(-(n - 1.0)) + (E * w0 / n).powf(w0 / 2.0);
        assert!((below - eq8).abs() < 1e-12 * eq8, "{below} vs {eq8}");
    }
}

#[test]
fn deeper_circuits_only_tighten_and_the_cutoff_helps_below_n_over_e() {
    let n = 100.0;
    // Depth: non-increasing in L everywhere (flat below L*, decaying above).
    let mut prev = f64::INFINITY;
    for l in (0..40_000).step_by(500) {
        let r = theorem1_mse_ratio_f(n, l as f64, 6.0);
        assert!(r <= prev + 1e-15, "L={l}: {r} > {prev}");
        prev = r;
    }
    // Cutoff: (e·w₀/N)^{w₀/2} is minimised at w₀ = N/e² ≈ 13.5 and crosses 1
    // at w₀ = N/e ≈ 36.8. Strictly decreasing up to the minimum…
    let mut prev = f64::INFINITY;
    for w0 in 2..=13 {
        let r = theorem1_mse_ratio(100, 10, w0).unwrap();
        assert!(r < prev, "w0={w0}: {r} ≥ {prev}");
        prev = r;
    }
    // …then it turns around: a longer cut buys nothing past N/e²…
    let at_13 = theorem1_mse_ratio(100, 10, 13).unwrap();
    let at_24 = theorem1_mse_ratio(100, 10, 24).unwrap();
    assert!(
        at_24 > at_13,
        "past N/e² the bound must grow: {at_24} vs {at_13}"
    );
    // …and past N/e it is vacuous outright.
    assert!(theorem1_mse_ratio(100, 10, 36).unwrap() < 1.0);
    assert!(theorem1_mse_ratio(100, 10, 40).unwrap() > 1.0);
}

#[test]
fn informative_at_chemistry_sizes_vacuous_at_test_sizes() {
    // N = 52, w₀ = 6: the paper's largest active space, (6e/52)^3 ≈ 0.031.
    let r = theorem1_mse_ratio(52, 300, 6).unwrap();
    assert!(r < 0.05, "{r}");
    // N = 12, w₀ = 4: (4e/12)^2 ≈ 0.82 — a prior that excludes almost nothing.
    let r = theorem1_mse_ratio(12, 20, 4).unwrap();
    assert!(r > 0.8 && r < 1.0, "{r}");
    // N = 4, w₀ = 4: e^2 > 1 — vacuous, and must be reported as such.
    assert!(theorem1_mse_ratio(4, 20, 4).unwrap() >= 1.0);
    // Degenerate inputs give no number rather than a wrong one.
    assert!(theorem1_mse_ratio(0, 1, 4).is_none());
    assert!(theorem1_mse_ratio(10, 1, 1).is_none());
}
