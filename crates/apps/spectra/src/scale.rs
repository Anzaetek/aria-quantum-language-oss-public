// SPDX-License-Identifier: Apache-2.0
//! spectra_scaling — the substrate at growing system sizes (the
//! QML_ROADMAP "toward the 13–19-qubit crossover" step, cf.
//! arXiv:2607.15815 §resource separation: exact classical simulation of
//! the scrambling substrate costs ~2^{γ·n} per sample while hardware
//! cost grows linearly in gate count).
//!
//! For n ∈ {7, 9, 11, 13} sites this harness:
//!   1. builds the n-site disordered chain (the `arch::build_ir`
//!      generalisation of spectra_heisenberg.aria) and asserts the
//!      |+⟩^⊗n eigenstate invariant (⟨Σ ZZ⟩ = 0 at φ = 0, Trotter-exact)
//!      — the correctness gate at every size;
//!   2. generates a labelled dataset and, for n ≤ 11, runs the quick
//!      gap check: DMQ (trainable couplings) vs the cheap classical
//!      panel (Fourier GAM + boosted stumps) on identical splits — the
//!      quantum–classical AUC gap must persist as n grows;
//!   3. records the deterministic per-sample simulation work
//!      (gate-ops × 2^n amplitude touches) and the measured wall-clock,
//!      fits the wall-clock growth exponent, and prints the paper-style
//!      crossover extrapolation (advisory — machine-dependent timings
//!      never gate the verdict).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use aria_verify_core::data::SplitMix64;
use aria_verify_core::Observable;
use omega_core::circuit::SymbolId;
use omega_core::executor::Backend;
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::params::ParameterBinding;

use crate::arch;

pub struct SizeReport {
    pub n_sites: usize,
    pub invariant: f64,
    pub work_per_sample: f64,
    pub secs_per_sample: f64,
    pub gap: Option<(f64, f64)>, // (dmq_auc, best_classical_auc)
}

fn chain_bonds(n: usize) -> Vec<(u32, u32)> {
    (0..n as u32 - 1).map(|k| (k, k + 1)).collect()
}

fn correlator(n: usize) -> Observable {
    Observable {
        terms: (0..n as u32 - 1)
            .map(|k| {
                (
                    1.0,
                    vec![
                        (k, omega_core::executor::PauliOp::Z),
                        (k + 1, omega_core::executor::PauliOp::Z),
                    ],
                )
            })
            .collect(),
    }
}

fn uniform_phase(rng: &mut SplitMix64) -> f64 {
    (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI
}

/// One size point: invariant check, dataset, optional gap check, timing.
#[allow(clippy::too_many_arguments)]
pub fn run_size(
    n_sites: usize,
    steps: usize,
    dt: f64,
    backend: &dyn Backend,
    seed: u64,
    n_samples: usize,
    gap_check: bool,
    dmq_epochs: usize,
) -> Result<SizeReport, String> {
    let bonds = chain_bonds(n_sites);
    let (ir, jt_ids, pt_ids) = arch::build_ir_sized(&bonds, steps, n_sites);
    let obs = correlator(n_sites);
    let mut rng = SplitMix64(seed ^ (n_sites as u64) << 8);
    let couplings: Vec<f64> = (0..bonds.len()).map(|_| 0.5 + rng.next_f64()).collect();

    let eval = |couplings: &[f64], phases: &[f64]| -> Result<f64, String> {
        let mut b = ParameterBinding::new();
        for (&id, &j) in jt_ids.iter().zip(couplings) {
            b.bind(id, j * dt);
        }
        for (&id, &p) in pt_ids.iter().zip(phases) {
            b.bind(id, p * dt);
        }
        backend
            .expectation(&ir, &b, &obs)
            .map_err(|e| e.to_string())
    };

    // 1. Correctness invariant (exact at every n).
    let invariant = eval(&couplings, &vec![0.0; n_sites])?;

    // 2. Dataset + timing.
    let phases: Vec<Vec<f64>> = (0..n_samples)
        .map(|_| (0..n_sites).map(|_| uniform_phase(&mut rng)).collect())
        .collect();
    let t0 = Instant::now();
    let z: Vec<f64> = phases
        .iter()
        .map(|p| eval(&couplings, p))
        .collect::<Result<_, _>>()?;
    let secs_per_sample = t0.elapsed().as_secs_f64() / n_samples as f64;
    let mut sorted = z.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tau = sorted[sorted.len() / 2];
    let y: Vec<f64> = z
        .iter()
        .map(|&v| if v > tau { 1.0 } else { -1.0 })
        .collect();
    // Deterministic work model: every gate op touches all 2^n amplitudes.
    let work_per_sample = ir.ops.len() as f64 * (1u64 << n_sites) as f64;

    // 3. Optional gap check (DMQ vs cheap classical panel).
    let gap = if gap_check {
        let half = n_samples / 2;
        let (trx, try_) = (&phases[..half], &y[..half]);
        let (tex, tey) = (&phases[half..], &y[half..]);
        // Classical: trained-frequency GAM + boosted stumps.
        let gam_map = crate::lanes::gam_basis(trx, try_);
        let gam_w = crate::lanes::logreg_fit(
            &trx.iter().map(|p| gam_map(p)).collect::<Vec<_>>(),
            try_,
            1e-3,
            200,
            0.4,
        );
        let gam_scores: Vec<f64> = tex
            .iter()
            .map(|p| crate::lanes::logreg_score(&gam_w, &gam_map(p)))
            .collect();
        let stumps = crate::lanes::stumps_fit(trx, try_, 120, 0.1);
        let stump_scores: Vec<f64> = tex
            .iter()
            .map(|p| crate::lanes::stumps_score(&stumps, p))
            .collect();
        let best_classical =
            crate::lanes::auc(&gam_scores, tey).max(crate::lanes::auc(&stump_scores, tey));

        // DMQ: flat-init couplings, adjoint + Adam, affine head.
        let mut learned = vec![1.0; bonds.len()];
        let mut head = vec![0.0, 0.0];
        let trainable: HashSet<SymbolId> = jt_ids.iter().copied().collect();
        let (mut am, mut av): (HashMap<u32, f64>, HashMap<u32, f64>) =
            (HashMap::new(), HashMap::new());
        for epoch in 0..dmq_epochs {
            let raw: Vec<f64> = trx
                .iter()
                .map(|p| eval(&learned, p))
                .collect::<Result<_, _>>()?;
            let feats: Vec<Vec<f64>> = raw.iter().map(|&v| vec![v]).collect();
            head = aria_verify_core::data::ridge_regression(&feats, try_, 1e-6)?;
            let a = head[0];
            let mut acc: HashMap<u32, f64> = HashMap::new();
            for ((p, &yi), &rv) in trx.iter().zip(try_).zip(&raw) {
                let r = aria_verify_core::data::ridge_predict(&head, &[rv]) - yi;
                let mut b = ParameterBinding::new();
                for (&id, &j) in jt_ids.iter().zip(&learned) {
                    b.bind(id, j * dt);
                }
                for (&id, &ph) in pt_ids.iter().zip(p.iter()) {
                    b.bind(id, ph * dt);
                }
                let grads = compute_gradient_for(
                    backend,
                    &ir,
                    &b,
                    &obs,
                    &GradMethod::Adjoint,
                    Some(&trainable),
                )
                .map_err(|e| e.to_string())?;
                for (id, g) in grads {
                    *acc.entry(id).or_insert(0.0) += 2.0 * r * a * dt * g / trx.len() as f64;
                }
            }
            let (b1, b2, eps) = (0.9, 0.999, 1e-8);
            let t = (epoch + 1) as f64;
            for (k, &id) in jt_ids.iter().enumerate() {
                let g = acc.get(&id).copied().unwrap_or(0.0);
                let mi = am.entry(id).or_insert(0.0);
                *mi = b1 * *mi + (1.0 - b1) * g;
                let vi = av.entry(id).or_insert(0.0);
                *vi = b2 * *vi + (1.0 - b2) * g * g;
                let m_hat = *mi / (1.0 - b1.powf(t));
                let v_hat = *vi / (1.0 - b2.powf(t));
                learned[k] -= 0.05 * m_hat / (v_hat.sqrt() + eps);
            }
        }
        let raw: Vec<f64> = trx
            .iter()
            .map(|p| eval(&learned, p))
            .collect::<Result<_, _>>()?;
        let feats: Vec<Vec<f64>> = raw.iter().map(|&v| vec![v]).collect();
        head = aria_verify_core::data::ridge_regression(&feats, try_, 1e-6)?;
        let mut scores = Vec::with_capacity(tex.len());
        for p in tex {
            scores.push(aria_verify_core::data::ridge_predict(
                &head,
                &[eval(&learned, p)?],
            ));
        }
        Some((crate::lanes::auc(&scores, tey), best_classical))
    } else {
        None
    };

    Ok(SizeReport {
        n_sites,
        invariant,
        work_per_sample,
        secs_per_sample,
        gap,
    })
}
