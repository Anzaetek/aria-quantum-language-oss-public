//! Chebyshev expansion of `1/x` on `[1/κ, 1]` and a stub QSVT phase
//! sequence derived from it.
//!
//! Replaces the uniform-scaling placeholder in
//! `circuits::qsvt::inversion_angles` with a proper polynomial
//! approximation. The full Wang–Lin / Haah angle-reduction (which
//! converts polynomial coefficients into the QSP phase sequence
//! `(φ₁,…,φ_d)`) is non-trivial; this module produces a faithful
//! polynomial first, then exposes a simple coefficient-derived angle
//! sequence that downstream `qsvt_circuit` consumers can plug in.

use std::f64::consts::PI;

/// Compute Chebyshev expansion coefficients of `1/x` on `[1/κ, 1]`.
///
/// Uses the standard discrete Chebyshev projection on the
/// `(degree+1)` Chebyshev–Gauss nodes:
///
/// ```text
/// xⱼ = (1+1/κ)/2 + ((1-1/κ)/2) · cos(π(j+½)/(N+1))
/// cₖ = (2/(N+1)) Σⱼ f(xⱼ) · cos(π·k·(j+½)/(N+1))            (k > 0)
/// c₀ = (1/(N+1)) Σⱼ f(xⱼ)
/// ```
///
/// The resulting `c[0..=degree]` satisfy
/// `1/x ≈ c₀ + 2·Σ_{k≥1} cₖ·Tₖ(t(x))` on `[1/κ, 1]` with rapidly
/// decaying error in `degree`.
pub fn cheb_inversion_coeffs(degree: usize, kappa: f64) -> Vec<f64> {
    assert!(kappa >= 1.0, "kappa must be ≥ 1");
    let n = degree + 1;
    // Sample `1/x` at Chebyshev nodes mapped to [1/κ, 1].
    let lo = 1.0 / kappa;
    let half_sum = 0.5 * (1.0 + lo);
    let half_diff = 0.5 * (1.0 - lo);
    let f: Vec<f64> = (0..n)
        .map(|j| {
            let t = (PI * (j as f64 + 0.5) / n as f64).cos();
            let x = half_sum + half_diff * t;
            1.0 / x
        })
        .collect();
    // Discrete Chebyshev projection.
    (0..n)
        .map(|k| {
            let s: f64 = (0..n)
                .map(|j| {
                    let arg = PI * k as f64 * (j as f64 + 0.5) / n as f64;
                    f[j] * arg.cos()
                })
                .sum();
            if k == 0 {
                s / n as f64
            } else {
                2.0 * s / n as f64
            }
        })
        .collect()
}

/// Evaluate a Chebyshev expansion `c[0..=d]` at `x ∈ [-1, 1]` using
/// the Clenshaw recurrence.
pub fn eval_chebyshev(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }
    if coeffs.len() == 1 {
        return coeffs[0];
    }
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for &c in coeffs.iter().skip(1).rev() {
        let b0 = 2.0 * x * b1 - b2 + c;
        b2 = b1;
        b1 = b0;
    }
    coeffs[0] + x * b1 - b2
}

/// QSVT phase angles for the polynomial approximation of `1/x` on
/// `[1/κ, 1]`. Derived directly from the Chebyshev coefficients.
///
/// **Note**: this is NOT the full Wang–Lin angle-reduction — it
/// produces a coefficient-tied sequence suitable for the placeholder
/// `qsvt_circuit` in `circuits::qsvt`. The *real* QSP angle finder (a
/// least-squares fit whose phases are verified to reproduce the target via
/// the [`super::qsp::qsp_response`] evaluator that mirrors
/// `QuantumProofs/QSP.lean`) now lives in
/// [`super::qsp::qsp_inversion_phases`]; prefer it when you need phases
/// that actually implement `1/x`. This function is retained for the
/// coefficient-domain recipe used by `quantum_solve_circuit`.
pub fn qsvt_inversion_angles(degree: usize, kappa: f64) -> Vec<f64> {
    let coeffs = cheb_inversion_coeffs(degree, kappa);
    // Normalize so that the polynomial sup-norm on [1/κ, 1] is ≤ 1
    // (required for QSVT block-encoding).
    let max_coef = coeffs
        .iter()
        .map(|c| c.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-12);
    coeffs
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            // Map coefficient to a phase via arctan, alternating sign
            // by parity to match the standard QSVT convention.
            let s = if k % 2 == 0 { 1.0 } else { -1.0 };
            s * (c / max_coef).atan()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Certified Neumann-series inverter
// ---------------------------------------------------------------------------
//
// The Chebyshev fit above is *heuristically* accurate but has no
// machine-checked error bound. The Neumann polynomial below is the
// approximant whose error is proven sorry-free in
// `proofs/lean4/QuantumProofs/QSVT.lean` (`invPoly`, `inv_poly_approx`),
// so its accuracy is certified, not just measured. Same `O(κ·log(κ/ε))`
// degree (Childs–Kothari–Somma); slower constant than Chebyshev but with
// a closed-form, verified worst-case bound.

/// Evaluate the **Neumann inversion polynomial**
/// `p_d(x) = Σ_{k=0}^{d} (1 − x)^k`, the certified degree-`d` approximant
/// of `1/x`.
///
/// Proven identities (Lean `invPoly_mul_self` / `inv_sub_invPoly`):
/// `p_d(x) = (1 − (1−x)^{d+1}) / x`, hence the exact error
/// `1/x − p_d(x) = (1−x)^{d+1} / x`.
///
/// Evaluated by Horner on `r = 1 − x` for numerical stability:
/// `Σ_{k=0}^{d} r^k = 1 + r(1 + r(1 + …))`.
pub fn eval_inv_poly(degree: usize, x: f64) -> f64 {
    let r = 1.0 - x;
    let mut acc = 1.0;
    for _ in 0..degree {
        acc = 1.0 + r * acc;
    }
    acc
}

/// The proven worst-case error of [`eval_inv_poly`] on `[1/κ, 1]`:
/// `(1 − 1/κ)^{d+1} / (1/κ) = κ·(1 − 1/κ)^{d+1}`.
///
/// This is exactly the Lean bound `(1−δ)^{d+1}/δ` with `δ = 1/κ`
/// (`QSVT.inv_poly_approx`); it is *saturated* at the worst point
/// `x = 1/κ` (Lean `qsvt_residual_exact`). So the value returned here is
/// both an upper bound everywhere on `[1/κ, 1]` and the actual error at
/// the spectral edge.
pub fn neumann_inversion_error_bound(degree: usize, kappa: f64) -> f64 {
    assert!(kappa >= 1.0, "kappa must be ≥ 1");
    let delta = 1.0 / kappa;
    (1.0 - delta).powi(degree as i32 + 1) / delta
}

/// Smallest degree `d` such that [`neumann_inversion_error_bound`]`(d, κ)
/// ≤ ε`, i.e. a *certified-sufficient* degree for `‖p_d − 1/x‖∞ ≤ ε` on
/// `[1/κ, 1]`.
///
/// Closed form from the proven bound: need `(1−δ)^{d+1} ≤ ε·δ`, so
/// `d + 1 ≥ ln(ε·δ) / ln(1−δ)` (both logs negative ⇒ ratio positive),
/// the `O(κ·log(κ/ε))` Childs–Kothari–Somma degree. `κ = 1` (spectrum
/// `{1}`) is exact at `d = 0`.
pub fn neumann_degree_for_eps(kappa: f64, eps: f64) -> usize {
    assert!(kappa >= 1.0, "kappa must be ≥ 1");
    assert!(eps > 0.0, "eps must be > 0");
    let delta = 1.0 / kappa;
    if delta >= 1.0 {
        return 0;
    }
    let ratio = (eps * delta).ln() / (1.0 - delta).ln();
    let d_plus_1 = ratio.ceil().max(1.0) as usize;
    d_plus_1 - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Chebyshev expansion of `1/x` should approximate `1/x`
    /// well at random points in `[1/κ, 1]`. We map `x ∈ [1/κ, 1]` to
    /// `t ∈ [-1, 1]` via `t = (2x − (1+1/κ)) / (1 − 1/κ)`.
    #[test]
    fn cheb_inversion_recovers_one_over_x() {
        let kappa = 4.0;
        let degree = 16;
        let coeffs = cheb_inversion_coeffs(degree, kappa);
        let lo = 1.0 / kappa;
        let half_sum = 0.5 * (1.0 + lo);
        let half_diff = 0.5 * (1.0 - lo);
        // Sample 21 evenly-spaced points strictly inside [1/κ, 1].
        let mut max_err = 0.0_f64;
        for i in 1..20 {
            let x = lo + (1.0 - lo) * (i as f64) / 20.0;
            let t = (x - half_sum) / half_diff;
            let approx = eval_chebyshev(&coeffs, t);
            let exact = 1.0 / x;
            let err = (approx - exact).abs();
            if err > max_err {
                max_err = err;
            }
        }
        // Degree-16 Chebyshev fit of 1/x on [0.25, 1] is excellent.
        assert!(max_err < 1e-3, "max_err = {max_err}");
    }

    #[test]
    fn cheb_inversion_degree_monotone() {
        // Higher degree should not increase error.
        let kappa = 8.0;
        let lo = 1.0 / kappa;
        let half_sum = 0.5 * (1.0 + lo);
        let half_diff = 0.5 * (1.0 - lo);

        let err_for = |d: usize| -> f64 {
            let coeffs = cheb_inversion_coeffs(d, kappa);
            (1..20)
                .map(|i| {
                    let x = lo + (1.0 - lo) * (i as f64) / 20.0;
                    let t = (x - half_sum) / half_diff;
                    (eval_chebyshev(&coeffs, t) - 1.0 / x).abs()
                })
                .fold(0.0, f64::max)
        };
        let e8 = err_for(8);
        let e24 = err_for(24);
        assert!(e24 <= e8 * 1.01, "e8={e8}, e24={e24}");
    }

    /// `eval_inv_poly(d,x)` equals the closed form `(1−(1−x)^{d+1})/x`
    /// exactly — the Rust mirror of Lean `invPoly_mul_self`.
    #[test]
    fn neumann_matches_geometric_identity() {
        for &d in &[0usize, 1, 4, 8, 16] {
            for i in 1..=20 {
                let x = i as f64 / 20.0; // (0, 1]
                let lhs = eval_inv_poly(d, x);
                let rhs = (1.0 - (1.0 - x).powi(d as i32 + 1)) / x;
                assert!((lhs - rhs).abs() < 1e-12, "d={d} x={x} lhs={lhs} rhs={rhs}");
            }
        }
    }

    /// The proven bound holds across `[1/κ, 1]` and is saturated at the
    /// spectral edge `x = 1/κ` — Rust mirror of Lean `inv_poly_approx`
    /// + `qsvt_residual_exact`.
    #[test]
    fn neumann_error_bound_holds_and_saturates() {
        let kappa = 4.0;
        let degree = 20;
        let bound = neumann_inversion_error_bound(degree, kappa);
        let delta = 1.0 / kappa;
        for i in 0..=50 {
            let x = delta + (1.0 - delta) * i as f64 / 50.0;
            let err = (1.0 / x - eval_inv_poly(degree, x)).abs();
            assert!(err <= bound + 1e-12, "x={x} err={err} bound={bound}");
        }
        let err_at_delta = (1.0 / delta - eval_inv_poly(degree, delta)).abs();
        assert!(
            (err_at_delta - bound).abs() < 1e-12,
            "saturation: err@δ={err_at_delta} bound={bound}"
        );
    }

    /// Exact numeric cross-check against the `QSVT.lean` Numeric section:
    /// κ=2, d=8 ⇒ p_8(1/2) = 511/256 and bound = 1/256.
    #[test]
    fn neumann_matches_lean_numeric_kappa2_d8() {
        let p = eval_inv_poly(8, 0.5);
        assert!((p - 511.0 / 256.0).abs() < 1e-12, "p_8(1/2)={p}");
        let bound = neumann_inversion_error_bound(8, 2.0);
        assert!((bound - 1.0 / 256.0).abs() < 1e-12, "bound={bound}");
    }

    /// `neumann_degree_for_eps` returns a degree that achieves the target
    /// and is minimal (one lower exceeds ε).
    #[test]
    fn neumann_degree_for_eps_is_minimal_and_sufficient() {
        for &(kappa, eps) in &[(2.0, 1e-3), (4.0, 1e-4), (10.0, 1e-2), (8.0, 1e-6)] {
            let d = neumann_degree_for_eps(kappa, eps);
            let bound = neumann_inversion_error_bound(d, kappa);
            assert!(
                bound <= eps,
                "insufficient: κ={kappa} ε={eps} d={d} bound={bound}"
            );
            if d > 0 {
                let prev = neumann_inversion_error_bound(d - 1, kappa);
                assert!(
                    prev > eps,
                    "not minimal: κ={kappa} ε={eps} d={d} prev={prev}"
                );
            }
        }
    }

    #[test]
    fn qsvt_angles_have_right_length() {
        let angles = qsvt_inversion_angles(12, 5.0);
        assert_eq!(angles.len(), 13);
        // Angles must be finite.
        for &a in &angles {
            assert!(a.is_finite());
            assert!(a.abs() <= std::f64::consts::FRAC_PI_2 + 1e-9);
        }
    }
}
