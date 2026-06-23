//! Quantum Signal Processing (QSP) response evaluation + phase finding.
//!
//! Replaces the placeholder `qsvt_inversion_angles` (which produced an
//! `atan`-of-Chebyshev-coefficients sequence that does **not** actually
//! implement `1/x`) with:
//!
//!   1. [`qsp_response`] — the exact QSP unitary response `U_Φ(x)[0][0]`,
//!      matching the formal `QuantumProofs/QSP.lean` `qsp` definition
//!      gate-for-gate (`U = S(φ₀)·W·S(φ₁)·W·…`). This is the *correct* QSP
//!      semantics: the phase sequence implements a polynomial in `x`
//!      (forward QSP theorem, proven sorry-free in Lean).
//!   2. [`qsp_inversion_phases`] — a Wang–Lin/Dong-style **angle finder**:
//!      least-squares fit of the QSP response to a target polynomial
//!      (here the scaled `1/x`) over Chebyshev nodes, by finite-difference
//!      gradient descent.
//!
//! Numeric anchors (see tests): the all-zero phase sequence reproduces the
//! Chebyshev polynomial `T_d` exactly (`W^d[0][0] = cos(d·arccos x) =
//! T_d(x)`), and the fitted phases round-trip to the target within a
//! stated tolerance.

use num_complex::Complex64;

type M2 = [[Complex64; 2]; 2];

#[inline]
fn mm(a: M2, b: M2) -> M2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// The QSP signal (X-rotation) operator `W(x) = [[x, i·s],[i·s, x]]`,
/// `s = √(1−x²)`. For `x ∈ [-1, 1]` this is `e^{i·arccos(x)·X}`.
#[inline]
fn signal(x: f64) -> M2 {
    let s = (1.0 - x * x).max(0.0).sqrt();
    [
        [Complex64::new(x, 0.0), Complex64::new(0.0, s)],
        [Complex64::new(0.0, s), Complex64::new(x, 0.0)],
    ]
}

/// The QSP phase operator `S(φ) = diag(e^{iφ}, e^{-iφ})`.
#[inline]
fn phase_mat(phi: f64) -> M2 {
    [
        [Complex64::from_polar(1.0, phi), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, -phi)],
    ]
}

/// Evaluate the QSP unitary `U_Φ(x) = S(φ₀)·W·S(φ₁)·W·…·S(φ_{d-1})·W` and
/// return its top-left entry `U_Φ(x)[0][0]`. Matches the Lean
/// `QuantumProofs.QSP.qsp` definition exactly: `d` phases ⇒ `d` signal
/// operators ⇒ a degree-`≤ d` polynomial in `x`.
pub fn qsp_response(phases: &[f64], x: f64) -> Complex64 {
    let w = signal(x);
    let mut acc: M2 = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    // qsp (φ :: rest) = S(φ) · W · qsp rest, so fold from the last phase.
    for &phi in phases.iter().rev() {
        acc = mm(phase_mat(phi), mm(w, acc));
    }
    acc[0][0]
}

/// The real polynomial implemented by a QSP phase sequence: `Re U_Φ(x)[0][0]`.
pub fn qsp_signal_poly(phases: &[f64], x: f64) -> f64 {
    qsp_response(phases, x).re
}

/// Chebyshev polynomial of the first kind `T_d(x)` (reference oracle).
pub fn chebyshev_t(d: usize, x: f64) -> f64 {
    // Stable recurrence on [-1, 1] (and beyond).
    if d == 0 {
        return 1.0;
    }
    let (mut tkm1, mut tk) = (1.0, x);
    for _ in 1..d {
        let tkp1 = 2.0 * x * tk - tkm1;
        tkm1 = tk;
        tk = tkp1;
    }
    tk
}

/// Max-norm error of a QSP phase sequence against a target real function,
/// sampled over `n` Chebyshev nodes on `[lo, 1]`.
pub fn qsp_fit_error<F: Fn(f64) -> f64>(phases: &[f64], target: &F, lo: f64, n: usize) -> f64 {
    cheb_nodes(lo, n)
        .into_iter()
        .map(|x| (qsp_signal_poly(phases, x) - target(x)).abs())
        .fold(0.0_f64, f64::max)
}

/// Chebyshev–Gauss nodes mapped into `[lo, 1]`.
fn cheb_nodes(lo: f64, n: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    let half_sum = 0.5 * (1.0 + lo);
    let half_diff = 0.5 * (1.0 - lo);
    (0..n)
        .map(|j| half_sum + half_diff * (PI * (j as f64 + 0.5) / n as f64).cos())
        .collect()
}

/// Fit QSP phases so that `Re U_Φ(x)[0][0] ≈ target(x)` on `[lo, 1]`, by
/// finite-difference gradient descent over Chebyshev nodes. Deterministic
/// (fixed zero/`π`-free init), returns the phase vector of length `degree`.
///
/// This is the Wang–Lin/Dong angle-finding step: it turns a target
/// polynomial into the QSP phase sequence that realizes it. General
/// high-degree robustness (Haah complementary-polynomial root finding) is a
/// further refinement; gradient descent is reliable for the moderate
/// degrees used by well-conditioned inversion.
pub fn fit_qsp_phases<F: Fn(f64) -> f64>(
    target: &F,
    degree: usize,
    lo: f64,
    iters: usize,
    lr: f64,
) -> Vec<f64> {
    let nodes = cheb_nodes(lo, (2 * degree + 1).max(8));
    // Init: tiny symmetric seed (all-zero would have zero gradient for odd
    // targets at the symmetric point; a small ramp breaks the symmetry).
    let mut phi: Vec<f64> = (0..degree)
        .map(|k| 0.01 * ((k as f64) - (degree as f64) / 2.0))
        .collect();
    let eps = 1e-6;
    let mut step = lr;
    let mut prev_loss = f64::INFINITY;
    let loss = |p: &[f64]| -> f64 {
        nodes
            .iter()
            .map(|&x| {
                let d = qsp_signal_poly(p, x) - target(x);
                d * d
            })
            .sum::<f64>()
    };
    for _ in 0..iters {
        // Central-difference gradient.
        let mut grad = vec![0.0_f64; degree];
        for k in 0..degree {
            let saved = phi[k];
            phi[k] = saved + eps;
            let lp = loss(&phi);
            phi[k] = saved - eps;
            let lm = loss(&phi);
            phi[k] = saved;
            grad[k] = (lp - lm) / (2.0 * eps);
        }
        for k in 0..degree {
            phi[k] -= step * grad[k];
        }
        let l = loss(&phi);
        // Simple adaptive step: back off on divergence, gently grow on progress.
        if l > prev_loss {
            step *= 0.5;
        } else {
            step *= 1.05;
        }
        prev_loss = l;
        if l < 1e-18 {
            break;
        }
    }
    phi
}

/// Find QSP phases implementing the scaled inverse `x ↦ (scale)/x` on
/// `[1/κ, 1]`. `scale` keeps the target sub-normalized (`|P| ≤ 1`); a value
/// `≤ 1/κ` is safe. Returns the phase sequence of length `degree`.
pub fn qsp_inversion_phases(degree: usize, kappa: f64, scale: f64, iters: usize) -> Vec<f64> {
    let lo = 1.0 / kappa;
    let target = |x: f64| scale / x;
    fit_qsp_phases(&target, degree, lo, iters, 0.02)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anchor: all-zero phases give `W^d`, whose `[0][0]` entry is exactly
    /// the Chebyshev polynomial `T_d(x)` (`= cos(d·arccos x)`), with zero
    /// imaginary part. Deterministic, hand-verifiable.
    #[test]
    fn zero_phases_give_chebyshev() {
        for d in 0..=10 {
            let phases = vec![0.0; d];
            for i in 0..=20 {
                let x = -1.0 + 2.0 * i as f64 / 20.0;
                let r = qsp_response(&phases, x);
                assert!(
                    (r.re - chebyshev_t(d, x)).abs() < 1e-9,
                    "d={d} x={x} re={} T_d={}",
                    r.re,
                    chebyshev_t(d, x)
                );
                assert!(r.im.abs() < 1e-9, "d={d} x={x} im={}", r.im);
            }
        }
    }

    /// QSP unitarity: `|U[0][0]|² + |U[0][1]|² = 1` for any phases (each
    /// factor is unitary, so the product is).
    #[test]
    fn qsp_response_is_unitary_row() {
        let phases = [0.3, -1.1, 0.7, 2.0, -0.5];
        for i in 0..=10 {
            let x = -1.0 + 2.0 * i as f64 / 10.0;
            let w = signal(x);
            let mut acc: M2 = [
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            ];
            for &phi in phases.iter().rev() {
                acc = mm(phase_mat(phi), mm(w, acc));
            }
            let row = acc[0][0].norm_sqr() + acc[0][1].norm_sqr();
            assert!((row - 1.0).abs() < 1e-9, "x={x} rownorm={row}");
        }
    }

    /// Round-trip: a single phase implements `x ↦ cos(2φ)·x + …`; fitting
    /// the target `0.6·x` (degree 1) recovers phases reproducing it.
    #[test]
    fn fit_degree_one_linear() {
        let target = |x: f64| 0.6 * x;
        let phases = fit_qsp_phases(&target, 1, -1.0, 1000, 0.05);
        let err = qsp_fit_error(&phases, &target, -1.0, 21);
        assert!(err < 1e-3, "degree-1 fit err={err}");
    }

    /// Round-trip on a Chebyshev target `0.9·T_3(x)` (degree 3): the fitted
    /// phases reproduce it within tolerance.
    #[test]
    fn fit_degree_three_chebyshev() {
        let target = |x: f64| 0.9 * chebyshev_t(3, x);
        let phases = fit_qsp_phases(&target, 3, -1.0, 4000, 0.02);
        let err = qsp_fit_error(&phases, &target, -1.0, 31);
        assert!(err < 5e-2, "degree-3 fit err={err}");
    }

    /// Inversion round-trip: fitted phases approximate the scaled `1/x` on
    /// `[1/κ, 1]` (κ=2) within a documented tolerance.
    #[test]
    fn fit_inversion_kappa2() {
        let kappa = 2.0;
        let scale = 0.4; // sub-normalized target 0.4/x ∈ [0.4, 0.8] on [0.5,1]
        let phases = qsp_inversion_phases(8, kappa, scale, 10000);
        let target = |x: f64| scale / x;
        let err = qsp_fit_error(&phases, &target, 1.0 / kappa, 25);
        assert!(err < 5e-2, "inversion fit err={err}");
    }
}
