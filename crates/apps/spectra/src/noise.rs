// SPDX-License-Identifier: Apache-2.0
//! Noise-robustness scan of the Heisenberg certification.
//!
//! The `spectra` harness certifies a genuine quantum advantage on the
//! disordered-Heisenberg substrate under an *ideal* simulator. The obvious
//! next question for any advantage claim is: **does it survive noise, and at
//! what error rate does it vanish?** This module answers it honestly.
//!
//! The dynamics-matched quantum lane is trained once on the noiseless
//! simulator (the deployed model). Its per-row correlator is then re-evaluated
//! through the PauliProp backend under a sweep of per-gate depolarizing rates —
//! PauliProp folds the noise channel into the Heisenberg-adjoint expectation
//! *exactly* (no trajectory sampling), and reproduces the statevector
//! correlator to machine precision at zero noise. Depolarizing shrinks every
//! Pauli-string expectation toward zero, so the correlator (and with it the
//! quantum score's discriminating power) decays with noise while the classical
//! lanes, computed on the same rows, stay fixed. The certification gate
//! `CI_lo(Δ AUC) > 0` therefore flips from CERTIFIED to REFUSED at some
//! crossover rate — that rate is the reported robustness margin.

use omega_backend_pauliprop::PauliPropBackend;
use omega_core::noise::{Depolarizing, NoiseModel, Rate};

use crate::lanes::{self, auc, bootstrap_delta_ci_lo, logreg_fit, logreg_score};

/// A decoherence channel to sweep. PauliProp folds each one's Heisenberg
/// adjoint into the expectation exactly, so the robustness margin can be
/// reported for more than one noise model — showing the crossover is not an
/// artifact of a single channel choice.
#[derive(Clone, Copy)]
pub enum Channel {
    /// Depolarizing after every gate (arity-aware): the standard worst-case
    /// per-gate error. Rate = per-gate depolarizing probability.
    Depolarizing,
    /// Amplitude damping (T1 relaxation, |1⟩ → |0⟩) after every gate. Rate = γ.
    AmplitudeDamping,
    /// Phase damping (T2 dephasing, loss of X/Y coherence, Z untouched) after
    /// every gate. Rate = λ.
    PhaseDamping,
}

impl Channel {
    pub fn label(self) -> &'static str {
        match self {
            Channel::Depolarizing => "per-gate depolarizing",
            Channel::AmplitudeDamping => "per-gate amplitude damping (T1)",
            Channel::PhaseDamping => "per-gate phase damping (T2)",
        }
    }

    fn model(self, rate: f64) -> NoiseModel {
        match self {
            Channel::Depolarizing => NoiseModel {
                depolarizing: Depolarizing::uniform(rate),
                ..Default::default()
            },
            Channel::AmplitudeDamping => NoiseModel {
                amplitude_damping: Rate::Uniform(rate),
                ..Default::default()
            },
            Channel::PhaseDamping => NoiseModel {
                phase_damping: Rate::Uniform(rate),
                ..Default::default()
            },
        }
    }
}

/// The best-scoring classical lane on a given split — the baseline the quantum
/// lane must beat. Fits the same five competitors as the main certification
/// panel (LogReg, Fourier GAM, GA2M, boosted stumps, order-matched JOINT) and
/// keeps the one with the highest test AUC.
pub struct ClassicalBest {
    pub name: &'static str,
    pub auc: f64,
    pub scores: Vec<f64>,
}

pub fn classical_best(
    train_x: &[Vec<f64>],
    train_y: &[f64],
    test_x: &[Vec<f64>],
    test_y: &[f64],
) -> ClassicalBest {
    // Each lane maps rows through a feature builder, then a shared logistic
    // head — identical to tier2_certify's panel, minus the printing.
    let fit_lane = |basis: &dyn Fn(&[f64]) -> Vec<f64>| -> Vec<f64> {
        let w = logreg_fit(
            &train_x.iter().map(|p| basis(p)).collect::<Vec<_>>(),
            train_y,
            1e-3,
            250,
            0.4,
        );
        test_x.iter().map(|p| logreg_score(&w, &basis(p))).collect()
    };

    let logreg = fit_lane(&|p: &[f64]| p.to_vec());
    let gam_map = lanes::gam_basis(train_x, train_y);
    let gam = fit_lane(&|p: &[f64]| gam_map(p));
    let ga2m_map = lanes::ga2m_basis(train_x, train_y);
    let ga2m = fit_lane(&|p: &[f64]| ga2m_map(p));
    let stumps = lanes::stumps_fit(train_x, train_y, 150, 0.1);
    let hgb: Vec<f64> = test_x
        .iter()
        .map(|p| lanes::stumps_score(&stumps, p))
        .collect();
    let joint_scan = lanes::joint_basis(train_x, train_y);
    let joint = fit_lane(&|p: &[f64]| (joint_scan.basis)(p));

    let lanes_scored: [(&'static str, Vec<f64>); 5] = [
        ("logreg", logreg),
        ("gam", gam),
        ("ga2m", ga2m),
        ("stumps", hgb),
        ("joint", joint),
    ];
    let mut best = ClassicalBest {
        name: "logreg",
        auc: -1.0,
        scores: vec![],
    };
    for (name, scores) in lanes_scored {
        let a = auc(&scores, test_y);
        if a > best.auc {
            best = ClassicalBest {
                name,
                auc: a,
                scores,
            };
        }
    }
    best
}

/// A PauliProp backend carrying `channel` at strength `rate`.
pub fn channel_backend(channel: Channel, rate: f64) -> PauliPropBackend {
    // coeff_min 1e-8 prunes only negligible Pauli terms, keeping the raw
    // correlator within ~1e-8 of the statevector value at zero noise (the
    // affine head then keeps the score within ~1e-6); noise is applied via its
    // exact Heisenberg adjoint. Nonzero rates shrink coefficients, so fewer
    // terms survive truncation and the sweep speeds up.
    PauliPropBackend::with_truncation(1e-8, None).with_noise(channel.model(rate))
}

/// One point of the noise sweep.
pub struct NoisePoint {
    pub rate: f64,
    pub auc_q: f64,
    pub ci_lo: f64,
    pub certified: bool,
}

/// The interpolated crossover rate: where the certification statistic
/// `CI_lo(Δ)` crosses zero — the CERTIFIED ↔ REFUSED boundary — between the
/// last certified point and the first refused one. Linear in `CI_lo` vs rate,
/// which is far tighter than "largest certified grid rate". Returns `None` if
/// the sweep never crosses (still certified at the top).
pub fn crossover_rate(points: &[NoisePoint]) -> Option<f64> {
    for w in points.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if a.certified && !b.certified {
            // a.ci_lo > 0 ≥ b.ci_lo, so the fraction lands in [0, 1).
            let t = a.ci_lo / (a.ci_lo - b.ci_lo);
            return Some(a.rate + t * (b.rate - a.rate));
        }
    }
    None
}

/// Bootstrap the quantum-vs-best-classical certification gate at one rate.
pub fn certify_point(
    rate: f64,
    quantum_scores: &[f64],
    classical: &ClassicalBest,
    test_y: &[f64],
    boot_reps: usize,
    seed: u64,
) -> NoisePoint {
    let auc_q = auc(quantum_scores, test_y);
    let ci_lo = bootstrap_delta_ci_lo(quantum_scores, &classical.scores, test_y, boot_reps, seed);
    NoisePoint {
        rate,
        auc_q,
        ci_lo,
        certified: ci_lo > 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(rate: f64, ci_lo: f64) -> NoisePoint {
        NoisePoint {
            rate,
            auc_q: 0.0,
            ci_lo,
            certified: ci_lo > 0.0,
        }
    }

    #[test]
    fn crossover_interpolates_the_ci_zero() {
        // CI_lo goes +0.1 (rate 0.02) → −0.1 (rate 0.04); the zero sits halfway.
        let points = [pt(0.0, 0.4), pt(0.02, 0.1), pt(0.04, -0.1)];
        let x = crossover_rate(&points).unwrap();
        assert!((x - 0.03).abs() < 1e-12, "expected 0.03, got {x}");
    }

    #[test]
    fn crossover_takes_the_first_transition() {
        // Only the first CERTIFIED→REFUSED edge counts, even if noise re-crosses.
        let points = [pt(0.0, 0.2), pt(0.1, -0.05), pt(0.2, 0.01), pt(0.3, -0.2)];
        let x = crossover_rate(&points).unwrap();
        // between 0.0 (+0.2) and 0.1 (−0.05): 0.2/(0.2+0.05)=0.8 → 0.08
        assert!((x - 0.08).abs() < 1e-12, "expected 0.08, got {x}");
    }

    #[test]
    fn no_crossover_when_always_certified() {
        let points = [pt(0.0, 0.4), pt(0.02, 0.3), pt(0.04, 0.2)];
        assert!(crossover_rate(&points).is_none());
    }
}
