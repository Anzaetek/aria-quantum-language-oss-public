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

/// The depolarizing crossover of the certification at ONE system size — the
/// scaling×noise data point. Trains the dynamics-matched lane noiselessly
/// (statevector), then re-scores it through the exact-noisy PauliProp backend
/// over a depolarizing-rate grid and interpolates where the certification gate
/// `CI_lo(Δ AUC) > 0` crosses zero.
pub struct SizeCrossover {
    pub n_sites: usize,
    pub eval_rows: usize,
    pub noiseless_auc_q: f64,
    pub classical_name: &'static str,
    pub classical_auc: f64,
    /// `max |statevector − PauliProp(0)|` over the eval scores, when the
    /// rate-0 PauliProp sanity was run at this size (only the smallest size,
    /// to keep the sweep affordable — the noiseless reference is otherwise the
    /// statevector directly).
    pub sanity_diff: Option<f64>,
    pub points: Vec<crate::noise::NoisePoint>,
    pub crossover: Option<f64>,
    /// Circuit gate count (ops in the lowered IR) — the physical driver of the
    /// noise-decay constant: each gate applies one depolarizing multiplier.
    pub gate_count: usize,
    /// (rate, signal amplitude) where amplitude = std of the quantum scores
    /// across eval rows. Depolarizing collapses this exponentially,
    /// A(r) ≈ A(0)·exp(−κ·r); κ (fit by the caller) is the draw-robust noise
    /// sensitivity, governed by gate count rather than the starting margin.
    pub signal: Vec<(f64, f64)>,
    /// Wall-clock of the nonzero-rate PauliProp sweep (illustrates the cost
    /// wall that caps the exact method's reach).
    pub noisy_secs: f64,
}

/// Population std of a score vector — the "signal amplitude" that decoherence
/// collapses toward zero.
fn amplitude(scores: &[f64]) -> f64 {
    let n = scores.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = scores.iter().sum::<f64>() / n;
    (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n).sqrt()
}

/// Build the labelled substrate dataset at `n_sites`: fixed disorder draw, then
/// median-thresholded sign of the bond correlator (identical recipe to
/// [`run_size`], factored so the noise harness shares it).
#[allow(clippy::type_complexity)]
fn sized_dataset(
    n_sites: usize,
    steps: usize,
    dt: f64,
    backend: &dyn Backend,
    seed: u64,
    n_samples: usize,
) -> Result<
    (
        omega_core::circuit::CircuitIR,
        Vec<u32>,
        Vec<u32>,
        Observable,
        Vec<Vec<f64>>,
        Vec<f64>,
    ),
    String,
> {
    let bonds = chain_bonds(n_sites);
    let (ir, jt_ids, pt_ids) = arch::build_ir_sized(&bonds, steps, n_sites);
    let obs = correlator(n_sites);
    let couplings = nested_couplings(bonds.len(), seed);
    let mut rng = SplitMix64(seed ^ 0xDA7A);
    let eval = |phases: &[f64]| -> Result<f64, String> {
        let mut b = ParameterBinding::new();
        for (&id, &j) in jt_ids.iter().zip(&couplings) {
            b.bind(id, j * dt);
        }
        for (&id, &p) in pt_ids.iter().zip(phases) {
            b.bind(id, p * dt);
        }
        backend
            .expectation(&ir, &b, &obs)
            .map_err(|e| e.to_string())
    };
    let phases: Vec<Vec<f64>> = (0..n_samples)
        .map(|_| (0..n_sites).map(|_| uniform_phase(&mut rng)).collect())
        .collect();
    let z: Vec<f64> = phases.iter().map(|p| eval(p)).collect::<Result<_, _>>()?;
    let mut sorted = z.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tau = sorted[sorted.len() / 2];
    let y: Vec<f64> = z
        .iter()
        .map(|&v| if v > tau { 1.0 } else { -1.0 })
        .collect();
    Ok((ir, jt_ids, pt_ids, obs, phases, y))
}

#[allow(clippy::too_many_arguments)]
pub fn run_size_noise(
    n_sites: usize,
    steps: usize,
    dt: f64,
    sv: &(dyn Backend + Sync),
    seed: u64,
    n_samples: usize,
    per_class: usize,
    rates: &[f64],
    coeff_min: f64,
    boot_reps: usize,
    epochs: usize,
    sanity: bool,
) -> Result<SizeCrossover, String> {
    use crate::noise::{self, Channel};
    use crate::qnn::DmqLane;
    assert_eq!(rates.first(), Some(&0.0), "rate grid must start at 0.0");

    let (ir, jt_ids, pt_ids, obs, phases, y) =
        sized_dataset(n_sites, steps, dt, sv, seed, n_samples)?;

    // Class-balanced train / eval split: eval takes the last `per_class` rows
    // of each class (a paired bootstrap needs both classes present); train is
    // everything before. Deterministic — the dataset order is seed-fixed.
    let mut rng = SplitMix64(seed ^ 0xE7A1 ^ n_sites as u64);
    let mut by_class = |want: f64| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..y.len()).filter(|&i| y[i] == want).collect();
        for k in (1..idx.len()).rev() {
            let j = (rng.next_f64() * (k + 1) as f64) as usize % (k + 1);
            idx.swap(k, j);
        }
        idx
    };
    let (pos, neg) = (by_class(1.0), by_class(-1.0));
    if pos.len() <= per_class || neg.len() <= per_class {
        return Err(format!(
            "n={n_sites}: too few samples per class ({} / {}) for {per_class} eval rows",
            pos.len(),
            neg.len()
        ));
    }
    let take = |src: &[Vec<f64>], idx: &[usize]| -> Vec<Vec<f64>> {
        idx.iter().map(|&i| src[i].clone()).collect()
    };
    let take1 = |src: &[f64], idx: &[usize]| -> Vec<f64> { idx.iter().map(|&i| src[i]).collect() };
    let tr_idx: Vec<usize> = pos[..pos.len() - per_class]
        .iter()
        .chain(&neg[..neg.len() - per_class])
        .copied()
        .collect();
    let ev_idx: Vec<usize> = pos[pos.len() - per_class..]
        .iter()
        .chain(&neg[neg.len() - per_class..])
        .copied()
        .collect();
    let (trx, try_) = (take(&phases, &tr_idx), take1(&y, &tr_idx));
    let (tex, tey) = (take(&phases, &ev_idx), take1(&y, &ev_idx));

    // Train the deployed model on the ideal simulator, pick its classical rival.
    let mut dmq = DmqLane::new(&ir, &jt_ids, &pt_ids, dt, &obs);
    dmq.fit(sv, &trx, &try_, epochs, 0.05)?;
    let classical = noise::classical_best(&trx, &try_, &tex, &tey);
    let sv_scores = dmq.scores(sv, &tex)?;
    let noiseless_auc_q = crate::lanes::auc(&sv_scores, &tey);

    // Optional PauliProp(0) ≈ statevector sanity (only the cheapest size).
    let sanity_diff = if sanity {
        let pp0 = dmq.scores_par(&noise::channel_backend(Channel::Depolarizing, 0.0), &tex)?;
        Some(
            sv_scores
                .iter()
                .zip(&pp0)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max),
        )
    } else {
        None
    };

    // Sweep: rate 0 reuses the exact statevector scores; nonzero rates run the
    // exact-noisy PauliProp adjoint at the looser (still <1% accurate) floor.
    let t0 = Instant::now();
    let mut points = Vec::new();
    let mut signal = Vec::new();
    for &r in rates {
        let qs = if r == 0.0 {
            sv_scores.clone()
        } else {
            dmq.scores_par(
                &noise::channel_backend_trunc(Channel::Depolarizing, r, coeff_min),
                &tex,
            )?
        };
        signal.push((r, amplitude(&qs)));
        points.push(noise::certify_point(
            r,
            &qs,
            &classical,
            &tey,
            boot_reps,
            seed ^ 0x51,
        ));
    }
    let noisy_secs = t0.elapsed().as_secs_f64();
    let crossover = noise::crossover_rate(&points);

    Ok(SizeCrossover {
        n_sites,
        eval_rows: tex.len(),
        noiseless_auc_q,
        classical_name: classical.name,
        classical_auc: classical.auc,
        sanity_diff,
        points,
        crossover,
        gate_count: ir.ops.len(),
        signal,
        noisy_secs,
    })
}

/// Least-squares decay constant κ of the signal amplitude: slope of
/// −ln A(r) vs r over the points with A above a small floor (the collapse can
/// take A below where a log is meaningful). Returns `None` if fewer than two
/// usable points remain. Draw-robust: κ is a *relative* decay, independent of
/// the rate-0 amplitude A(0).
pub fn decay_constant(signal: &[(f64, f64)]) -> Option<f64> {
    let floor = 1e-4;
    let pts: Vec<(f64, f64)> = signal
        .iter()
        .filter(|&&(_, a)| a > floor)
        .map(|&(r, a)| (r, -a.ln()))
        .collect();
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f64;
    let (sx, sy): (f64, f64) = (pts.iter().map(|p| p.0).sum(), pts.iter().map(|p| p.1).sum());
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    let denom = n * sxx - sx * sx;
    (denom.abs() > 1e-18).then_some((n * sxy - sx * sy) / denom)
}

/// Nested disorder: the SAME realization extended by one bond per size, so the
/// only variable across sizes is the added gates. Seeded independently of n, so
/// the first (n−1) couplings match for every n — a controlled scaling
/// experiment, not a fresh draw at each size.
fn nested_couplings(n_bonds: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64(seed ^ 0xC0DE);
    (0..n_bonds).map(|_| 0.5 + rng.next_f64()).collect()
}

/// One point of the κ-scaling scan: the depolarizing decay constant of the
/// substrate's bond-correlator signal at `n_sites`, measured directly from the
/// generator's true (nested) couplings — no training needed, since κ is a
/// property of the circuit and the observable, and the trained lane recovers
/// the true couplings anyway. Cheap: only the nonzero rates call PauliProp,
/// and only over `n_rows` inputs.
pub struct KappaPoint {
    pub n_sites: usize,
    pub gate_count: usize,
    pub signal: Vec<(f64, f64)>,
    pub kappa: Option<f64>,
    pub secs: f64,
}

pub fn generator_kappa(
    n_sites: usize,
    steps: usize,
    dt: f64,
    seed: u64,
    n_rows: usize,
    rates: &[f64],
    coeff_min: f64,
) -> Result<KappaPoint, String> {
    use rayon::prelude::*;
    let bonds = chain_bonds(n_sites);
    let (ir, jt_ids, pt_ids) = arch::build_ir_sized(&bonds, steps, n_sites);
    let obs = correlator(n_sites);
    let couplings = nested_couplings(bonds.len(), seed);
    // Fixed input rows (n-dependent dimension, deterministic seed).
    let mut prng = SplitMix64(seed ^ 0xA11CE ^ n_sites as u64);
    let rows: Vec<Vec<f64>> = (0..n_rows)
        .map(|_| (0..n_sites).map(|_| uniform_phase(&mut prng)).collect())
        .collect();
    let corr = |backend: &(dyn Backend + Sync), phases: &[f64]| -> Result<f64, String> {
        let mut b = ParameterBinding::new();
        for (&id, &j) in jt_ids.iter().zip(&couplings) {
            b.bind(id, j * dt);
        }
        for (&id, &p) in pt_ids.iter().zip(phases) {
            b.bind(id, p * dt);
        }
        backend
            .expectation(&ir, &b, &obs)
            .map_err(|e| e.to_string())
    };
    let t0 = Instant::now();
    let mut signal = Vec::new();
    for &r in rates {
        let vals: Vec<f64> = if r == 0.0 {
            let sv = omega_backend_statevector::StatevectorBackend::new();
            rows.iter()
                .map(|p| corr(&sv, p))
                .collect::<Result<_, _>>()?
        } else {
            let bk = crate::noise::channel_backend_trunc(
                crate::noise::Channel::Depolarizing,
                r,
                coeff_min,
            );
            rows.par_iter()
                .map(|p| corr(&bk, p))
                .collect::<Result<_, _>>()?
        };
        signal.push((r, amplitude(&vals)));
    }
    let secs = t0.elapsed().as_secs_f64();
    let kappa = decay_constant(&signal);
    Ok(KappaPoint {
        n_sites,
        gate_count: ir.ops.len(),
        signal,
        kappa,
        secs,
    })
}
