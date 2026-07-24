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
use omega_core::noise::{Depolarizing, NoiseModel};

use crate::lanes::{self, auc, bootstrap_delta_ci_lo, logreg_fit, logreg_score};

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

/// A per-gate uniform depolarizing model at rate `p`.
pub fn depolarizing_backend(p: f64) -> PauliPropBackend {
    // coeff_min 1e-8 prunes only negligible Pauli terms, keeping the raw
    // correlator within ~1e-8 of the statevector value at zero noise (the
    // affine head then keeps the score within ~1e-6); noise is applied via its
    // exact Heisenberg adjoint. Nonzero rates shrink coefficients, so fewer
    // terms survive truncation and the sweep speeds up.
    PauliPropBackend::with_truncation(1e-8, None).with_noise(NoiseModel {
        depolarizing: Depolarizing::uniform(p),
        ..Default::default()
    })
}

/// One point of the noise sweep.
pub struct NoisePoint {
    pub rate: f64,
    pub auc_q: f64,
    pub ci_lo: f64,
    pub certified: bool,
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
