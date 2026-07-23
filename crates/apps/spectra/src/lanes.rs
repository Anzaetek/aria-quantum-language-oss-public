// SPDX-License-Identifier: Apache-2.0
//! The SPECTRA classical lane panel (arXiv:2607.15815 Tier 2) plus the
//! shared metrics: five classical competitors on identical splits —
//! LogReg, trained-frequency Fourier GAM, GA2M (pairwise harmonic
//! products), boosted stumps (the HGB stand-in), and the order-matched
//! JOINT lane whose supervised periodogram scan hunts explicit k-way
//! `cos(Σ f_j·φ_j + b)` terms. Completeness of the panel is what makes
//! refusal on the sparse pocket *correct* rather than a lane gap.
//!
//! Everything is pure Rust and deterministic; per-lane feature builders
//! return closures so train and test rows go through identical maps.

use aria_verify_core::data::SplitMix64;

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Rank-based AUC of `scores` against ±1 labels (ties get half credit).
pub fn auc(scores: &[f64], y: &[f64]) -> f64 {
    let mut pos = 0.0;
    let mut neg = 0.0;
    let mut won = 0.0;
    for (i, &si) in scores.iter().enumerate() {
        if y[i] < 0.0 {
            continue;
        }
        pos += 1.0;
        for (j, &sj) in scores.iter().enumerate() {
            if y[j] > 0.0 {
                continue;
            }
            won += if si > sj {
                1.0
            } else if si == sj {
                0.5
            } else {
                0.0
            };
        }
    }
    for &yj in y {
        if yj < 0.0 {
            neg += 1.0;
        }
    }
    if pos == 0.0 || neg == 0.0 {
        return 0.5;
    }
    won / (pos * neg)
}

/// Paired stratified bootstrap of `Δ = auc(a) − auc(b)` over the SAME
/// test rows; returns the 2.5th-percentile lower confidence bound.
pub fn bootstrap_delta_ci_lo(
    scores_a: &[f64],
    scores_b: &[f64],
    y: &[f64],
    reps: usize,
    seed: u64,
) -> f64 {
    let n = y.len();
    let pos: Vec<usize> = (0..n).filter(|&i| y[i] > 0.0).collect();
    let neg: Vec<usize> = (0..n).filter(|&i| y[i] < 0.0).collect();
    let mut rng = SplitMix64(seed);
    let mut deltas = Vec::with_capacity(reps);
    for _ in 0..reps {
        // Stratified: resample positives and negatives separately.
        let mut idx = Vec::with_capacity(n);
        for _ in 0..pos.len() {
            idx.push(pos[(rng.next_f64() * pos.len() as f64) as usize % pos.len()]);
        }
        for _ in 0..neg.len() {
            idx.push(neg[(rng.next_f64() * neg.len() as f64) as usize % neg.len()]);
        }
        let ya: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let sa: Vec<f64> = idx.iter().map(|&i| scores_a[i]).collect();
        let sb: Vec<f64> = idx.iter().map(|&i| scores_b[i]).collect();
        deltas.push(auc(&sa, &ya) - auc(&sb, &ya));
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    deltas[(0.025 * reps as f64) as usize]
}

// ---------------------------------------------------------------------------
// Logistic regression (the shared linear head of every classical lane)
// ---------------------------------------------------------------------------

/// L2-regularised logistic regression via full-batch gradient descent.
/// Returns `[w..., b]`; deterministic (zero init, fixed schedule).
pub fn logreg_fit(x: &[Vec<f64>], y: &[f64], l2: f64, iters: usize, lr: f64) -> Vec<f64> {
    let d = x[0].len();
    let n = x.len() as f64;
    let mut w = vec![0.0; d + 1];
    for _ in 0..iters {
        let mut g = vec![0.0; d + 1];
        for (xi, &yi) in x.iter().zip(y) {
            let z = logreg_score(&w, xi);
            // dL/dz for label ±1: −y·σ(−y·z)
            let m = -yi / (1.0 + (yi * z).exp());
            for (k, &xk) in xi.iter().enumerate() {
                g[k] += m * xk / n;
            }
            g[d] += m / n;
        }
        for k in 0..d {
            g[k] += l2 * w[k];
        }
        for (wk, gk) in w.iter_mut().zip(&g) {
            *wk -= lr * gk;
        }
    }
    w
}

pub fn logreg_score(w: &[f64], x: &[f64]) -> f64 {
    x.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f64>() + w[w.len() - 1]
}

// ---------------------------------------------------------------------------
// Frequency scanning (shared by GAM / GA2M / JOINT and the Tier-1 screen)
// ---------------------------------------------------------------------------

/// The continuum frequency grid every scan uses: 0.25 … 8.0 in steps of
/// 0.05 — fine enough to land within 0.05 of any generator frequency.
pub fn freq_grid() -> Vec<f64> {
    (5..=160).map(|k| k as f64 * 0.05).collect()
}

/// Periodogram power of the labels against one phase column at
/// frequency `f`: corr(cos)² + corr(sin)².
pub fn periodogram_power(phase_col: &[f64], y: &[f64], f: f64) -> f64 {
    let n = phase_col.len() as f64;
    let (mut c, mut s) = (0.0, 0.0);
    for (&p, &yi) in phase_col.iter().zip(y) {
        c += (f * p).cos() * yi / n;
        s += (f * p).sin() * yi / n;
    }
    c * c + s * s
}

/// Top-`k` frequencies for one feature by periodogram power.
fn top_frequencies(phase_col: &[f64], y: &[f64], k: usize) -> Vec<f64> {
    let mut scored: Vec<(f64, f64)> = freq_grid()
        .into_iter()
        .map(|f| (periodogram_power(phase_col, y, f), f))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.into_iter().take(k).map(|(_, f)| f).collect()
}

fn column(phases: &[Vec<f64>], j: usize) -> Vec<f64> {
    phases.iter().map(|p| p[j]).collect()
}

// ---------------------------------------------------------------------------
// Lane feature maps
// ---------------------------------------------------------------------------

/// Trained-frequency GAM basis: per feature, cos/sin at its top-2
/// periodogram frequencies (learned on the TRAIN split only).
pub fn gam_basis(train_phases: &[Vec<f64>], train_y: &[f64]) -> impl Fn(&[f64]) -> Vec<f64> {
    let d = train_phases[0].len();
    let freqs: Vec<Vec<f64>> = (0..d)
        .map(|j| top_frequencies(&column(train_phases, j), train_y, 2))
        .collect();
    move |p: &[f64]| {
        let mut out = Vec::with_capacity(4 * p.len());
        for (j, &pj) in p.iter().enumerate() {
            for &f in &freqs[j] {
                out.push((f * pj).cos());
                out.push((f * pj).sin());
            }
        }
        out
    }
}

/// GA2M basis: the GAM basis plus all pairwise products of each
/// feature's top-1 harmonic pair — explicit order-2 interactions.
pub fn ga2m_basis(train_phases: &[Vec<f64>], train_y: &[f64]) -> impl Fn(&[f64]) -> Vec<f64> {
    let d = train_phases[0].len();
    let top1: Vec<f64> = (0..d)
        .map(|j| top_frequencies(&column(train_phases, j), train_y, 1)[0])
        .collect();
    let gam = gam_basis(train_phases, train_y);
    move |p: &[f64]| {
        let mut out = gam(p);
        for a in 0..p.len() {
            for b in (a + 1)..p.len() {
                let (ca, sa) = ((top1[a] * p[a]).cos(), (top1[a] * p[a]).sin());
                let (cb, sb) = ((top1[b] * p[b]).cos(), (top1[b] * p[b]).sin());
                out.extend_from_slice(&[ca * cb, ca * sb, sa * cb, sa * sb]);
            }
        }
        out
    }
}

/// Order-matched JOINT lane: the GA2M basis extended with explicit
/// 3-way terms cos/sin(f_a·φ_a + f_b·φ_b + f_c·φ_c) found by a
/// supervised periodogram scan over feature triples × a coarse joint
/// frequency grid. For d > 4 the triple scan is capped to the 4
/// highest-single-power features (the cap is printed by the caller —
/// no silent truncation).
/// A feature map from a phase row to lane features.
pub type FeatureMap = Box<dyn Fn(&[f64]) -> Vec<f64>>;

pub struct JointScanResult {
    pub basis: FeatureMap,
    /// The discovered joint term: feature triple + refined frequencies —
    /// printed by the caller so a run shows WHAT the scan found (on the
    /// sparse pocket it should be the planted (3.7, 5.1, 6.8)).
    pub triple: [usize; 3],
    pub freqs: [f64; 3],
    pub best_triple_power: f64,
    pub capped_features: Option<usize>,
}

pub fn joint_basis(train_phases: &[Vec<f64>], train_y: &[f64]) -> JointScanResult {
    let d = train_phases[0].len();
    // Coarse joint grid: 1.0 … 8.0 step 0.35 (21 values) — lands within
    // 0.175 of any generator frequency; the linear head absorbs the
    // residual detuning over the ±π window.
    let grid: Vec<f64> = (0..21).map(|k| 1.0 + 0.35 * k as f64).collect();
    let (feat_pool, capped): (Vec<usize>, Option<usize>) = if d > 4 {
        let mut by_power: Vec<(f64, usize)> = (0..d)
            .map(|j| {
                let col = column(train_phases, j);
                let p = freq_grid()
                    .into_iter()
                    .map(|f| periodogram_power(&col, train_y, f))
                    .fold(0.0, f64::max);
                (p, j)
            })
            .collect();
        by_power.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        (
            by_power.into_iter().take(4).map(|(_, j)| j).collect(),
            Some(4),
        )
    } else {
        ((0..d).collect(), None)
    };

    // Scan triples for the strongest joint response.
    let n = train_phases.len() as f64;
    let mut best: (f64, [usize; 3], [f64; 3]) = (0.0, [0, 0, 0], [0.0, 0.0, 0.0]);
    for ia in 0..feat_pool.len() {
        for ib in (ia + 1)..feat_pool.len() {
            for ic in (ib + 1)..feat_pool.len() {
                let (a, b_, c_) = (feat_pool[ia], feat_pool[ib], feat_pool[ic]);
                for &fa in &grid {
                    for &fb in &grid {
                        for &fc in &grid {
                            let (mut cc, mut ss) = (0.0, 0.0);
                            for (p, &yi) in train_phases.iter().zip(train_y) {
                                let u = fa * p[a] + fb * p[b_] + fc * p[c_];
                                cc += u.cos() * yi / n;
                                ss += u.sin() * yi / n;
                            }
                            let pow = cc * cc + ss * ss;
                            if pow > best.0 {
                                best = (pow, [a, b_, c_], [fa, fb, fc]);
                            }
                        }
                    }
                }
            }
        }
    }
    // Two-stage refinement: coordinate-wise fine scan (±0.35, step
    // 0.05, two rounds) around the coarse winner — lands within 0.025
    // of any generator frequency without a cubic fine grid.
    let (_, tri, mut fs) = best;
    let n_ = train_phases.len() as f64;
    let joint_power = |fs: &[f64; 3]| -> f64 {
        let (mut cc, mut ss) = (0.0, 0.0);
        for (p, &yi) in train_phases.iter().zip(train_y) {
            let u = fs[0] * p[tri[0]] + fs[1] * p[tri[1]] + fs[2] * p[tri[2]];
            cc += u.cos() * yi / n_;
            ss += u.sin() * yi / n_;
        }
        cc * cc + ss * ss
    };
    for _round in 0..2 {
        for coord in 0..3 {
            let base = fs[coord];
            let mut best_local = (joint_power(&fs), base);
            for step in -7..=7 {
                let cand = base + 0.05 * step as f64;
                if cand <= 0.2 {
                    continue;
                }
                let mut trial = fs;
                trial[coord] = cand;
                let pw = joint_power(&trial);
                if pw > best_local.0 {
                    best_local = (pw, cand);
                }
            }
            fs[coord] = best_local.1;
        }
    }
    let best_power = joint_power(&fs);
    let ga2m = ga2m_basis(train_phases, train_y);
    let basis = Box::new(move |p: &[f64]| {
        let mut out = ga2m(p);
        let u = fs[0] * p[tri[0]] + fs[1] * p[tri[1]] + fs[2] * p[tri[2]];
        out.push(u.cos());
        out.push(u.sin());
        out
    });
    JointScanResult {
        basis,
        triple: tri,
        freqs: fs,
        best_triple_power: best_power,
        capped_features: capped,
    }
}

// ---------------------------------------------------------------------------
// Boosted stumps (the axis-aligned HGB stand-in)
// ---------------------------------------------------------------------------

pub struct Stump {
    feature: usize,
    threshold: f64,
    left: f64,
    right: f64,
}

/// LS-Boost with depth-1 trees on quantile thresholds — the axis-aligned
/// partitioner whose failure mode (C4: high joint frequency) the
/// Fourier wall predicts.
pub fn stumps_fit(x: &[Vec<f64>], y: &[f64], rounds: usize, lr: f64) -> Vec<Stump> {
    let d = x[0].len();
    let n = x.len();
    // 15 quantile thresholds per feature.
    let thresholds: Vec<Vec<f64>> = (0..d)
        .map(|j| {
            let mut col: Vec<f64> = x.iter().map(|r| r[j]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (1..16).map(|q| col[q * n / 16]).collect()
        })
        .collect();
    let mut residual: Vec<f64> = y.to_vec();
    let mut model = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let mut best = (f64::INFINITY, 0usize, 0.0, 0.0, 0.0);
        for j in 0..d {
            for &t in &thresholds[j] {
                let (mut sl, mut nl, mut sr, mut nr) = (0.0, 0.0, 0.0, 0.0);
                for (r, &res) in x.iter().zip(&residual) {
                    if r[j] <= t {
                        sl += res;
                        nl += 1.0;
                    } else {
                        sr += res;
                        nr += 1.0;
                    }
                }
                if nl == 0.0 || nr == 0.0 {
                    continue;
                }
                let (ml, mr) = (sl / nl, sr / nr);
                // SSE reduction of this split.
                let sse: f64 = x
                    .iter()
                    .zip(&residual)
                    .map(|(r, &res)| {
                        let m = if r[j] <= t { ml } else { mr };
                        (res - m) * (res - m)
                    })
                    .sum();
                if sse < best.0 {
                    best = (sse, j, t, ml, mr);
                }
            }
        }
        let stump = Stump {
            feature: best.1,
            threshold: best.2,
            left: lr * best.3,
            right: lr * best.4,
        };
        for (r, res) in x.iter().zip(residual.iter_mut()) {
            *res -= if r[stump.feature] <= stump.threshold {
                stump.left
            } else {
                stump.right
            };
        }
        model.push(stump);
    }
    model
}

pub fn stumps_score(model: &[Stump], x: &[f64]) -> f64 {
    model
        .iter()
        .map(|s| {
            if x[s.feature] <= s.threshold {
                s.left
            } else {
                s.right
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_orders_perfect_and_random() {
        let y = [1.0, 1.0, -1.0, -1.0];
        assert!((auc(&[0.9, 0.8, 0.2, 0.1], &y) - 1.0).abs() < 1e-12);
        assert!((auc(&[0.1, 0.2, 0.8, 0.9], &y) - 0.0).abs() < 1e-12);
        assert!((auc(&[0.5, 0.5, 0.5, 0.5], &y) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn logreg_separates_a_linear_problem() {
        let x: Vec<Vec<f64>> = (0..40).map(|i| vec![(i as f64 - 19.5) / 10.0]).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| if r[0] > 0.0 { 1.0 } else { -1.0 })
            .collect();
        let w = logreg_fit(&x, &y, 1e-4, 300, 0.5);
        let scores: Vec<f64> = x.iter().map(|r| logreg_score(&w, r)).collect();
        assert!(auc(&scores, &y) > 0.99);
    }

    #[test]
    fn stumps_fit_a_step_function() {
        let x: Vec<Vec<f64>> = (0..60).map(|i| vec![i as f64 / 59.0]).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| if r[0] > 0.6 { 1.0 } else { -1.0 })
            .collect();
        let m = stumps_fit(&x, &y, 40, 0.3);
        let scores: Vec<f64> = x.iter().map(|r| stumps_score(&m, r)).collect();
        assert!(auc(&scores, &y) > 0.99);
    }

    #[test]
    fn periodogram_peaks_at_the_generating_frequency() {
        let mut rng = SplitMix64(7);
        let phases: Vec<f64> = (0..400)
            .map(|_| (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI)
            .collect();
        let y: Vec<f64> = phases
            .iter()
            .map(|&p| if (3.7 * p).cos() > 0.0 { 1.0 } else { -1.0 })
            .collect();
        let p_gen = periodogram_power(&phases, &y, 3.7);
        let p_off = periodogram_power(&phases, &y, 1.3);
        assert!(p_gen > 5.0 * p_off, "gen {p_gen} vs off {p_off}");
    }
}
