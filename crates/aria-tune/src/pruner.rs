// SPDX-License-Identifier: Apache-2.0
//! Pruners: stop a trial early once its intermediate reports say it cannot win.
//!
//! A pruner only ever sees what the caller [`report`](crate::Study::report)s —
//! a step index, a running value, and named auxiliary metrics. It never
//! computes anything itself. That keeps this crate dependency-free (no tensor
//! library, no PCA) while still supporting rich domain gates: the caller
//! computes `grad_norm` / `eff_rank` / whatever and hands them over as `aux`.

use crate::sampler::Direction;
use crate::trial::{Trial, TrialState};

/// Decides whether a running trial should stop.
pub trait Pruner {
    /// `history` is every trial recorded so far, including `trial` itself.
    fn should_prune(&mut self, trial: &Trial, history: &[Trial], direction: Direction) -> bool;
    fn name(&self) -> &'static str;
}

/// Never prunes.
#[derive(Debug, Clone, Default)]
pub struct NoPruner;

impl Pruner for NoPruner {
    fn should_prune(&mut self, _t: &Trial, _h: &[Trial], _d: Direction) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "none"
    }
}

// ---------------------------------------------------------------------------
// Median
// ---------------------------------------------------------------------------

/// Prune when a trial's value at step `s` is worse than the median of every
/// completed trial's value at that same step.
///
/// Two warmups keep it from firing on noise: `warmup_trials` completed trials
/// must exist before any pruning, and each trial gets `warmup_steps` reports
/// of grace.
#[derive(Debug, Clone)]
pub struct MedianPruner {
    pub warmup_trials: usize,
    pub warmup_steps: usize,
}

impl Default for MedianPruner {
    fn default() -> Self {
        Self {
            warmup_trials: 3,
            warmup_steps: 2,
        }
    }
}

impl MedianPruner {
    pub fn new(warmup_trials: usize, warmup_steps: usize) -> Self {
        Self {
            warmup_trials,
            warmup_steps,
        }
    }
}

fn median(mut xs: Vec<f64>) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    Some(if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    })
}

impl Pruner for MedianPruner {
    fn should_prune(&mut self, trial: &Trial, history: &[Trial], direction: Direction) -> bool {
        let Some(last) = trial.steps.last() else {
            return false;
        };
        if trial.steps.len() <= self.warmup_steps {
            return false;
        }
        let completed: Vec<&Trial> = history
            .iter()
            .filter(|t| matches!(t.state, TrialState::Complete(_)) && t.id != trial.id)
            .collect();
        if completed.len() < self.warmup_trials {
            return false;
        }
        // Compare against the peers' value at the SAME step.
        let peers: Vec<f64> = completed
            .iter()
            .filter_map(|t| t.steps.iter().find(|s| s.step == last.step))
            .map(|s| s.value)
            .collect();
        let Some(med) = median(peers) else {
            return false;
        };
        // Prune when strictly worse than the median.
        !direction.at_least_as_good(last.value, med)
    }

    fn name(&self) -> &'static str {
        "median"
    }
}

// ---------------------------------------------------------------------------
// Successive halving
// ---------------------------------------------------------------------------

/// Asynchronous successive halving: at each rung step, keep only the best
/// `1/eta` fraction of the trials that have reached it.
#[derive(Debug, Clone)]
pub struct SuccessiveHalving {
    /// Step indices at which the cut happens.
    pub rungs: Vec<usize>,
    /// Keep the top `1/eta` at each rung.
    pub eta: f64,
    pub min_at_rung: usize,
}

impl SuccessiveHalving {
    pub fn new(rungs: Vec<usize>, eta: f64) -> Self {
        Self {
            rungs,
            eta: eta.max(1.0001),
            min_at_rung: 2,
        }
    }
}

impl Pruner for SuccessiveHalving {
    fn should_prune(&mut self, trial: &Trial, history: &[Trial], direction: Direction) -> bool {
        let Some(last) = trial.steps.last() else {
            return false;
        };
        if !self.rungs.contains(&last.step) {
            return false;
        }
        // Everyone else who reached this rung.
        let peers: Vec<f64> = history
            .iter()
            .filter(|t| t.id != trial.id)
            .filter_map(|t| t.steps.iter().find(|s| s.step == last.step))
            .map(|s| s.value)
            .collect();
        if peers.len() + 1 < self.min_at_rung {
            return false;
        }
        let keep = (((peers.len() + 1) as f64) / self.eta).ceil().max(1.0) as usize;
        // Rank among peers: how many are strictly better?
        let better = peers
            .iter()
            .filter(|p| !direction.at_least_as_good(last.value, **p))
            .count();
        better >= keep
    }

    fn name(&self) -> &'static str {
        "successive_halving"
    }
}

// ---------------------------------------------------------------------------
// Gate pruner
// ---------------------------------------------------------------------------

/// Domain gates on caller-reported `aux` metrics, in the style of the
/// finance_lab Optuna study.
///
/// Four independent gates, any of which prunes:
/// * **vanishing gradient** — `aux["grad_norm"]` below a floor;
/// * **no skill** — the reported value at or below a chance baseline;
/// * **loss slope** — no improvement of at least `min_delta` across a window;
/// * **expressivity collapse** — `aux["eff_rank"]` too low or
///   `aux["top2_cumevr"]` too high, against thresholds **calibrated from the
///   first `warmup_trials` completed trials** rather than guessed.
///
/// `relaxation` (≥ 1) widens the calibrated collapse thresholds. Quantum
/// trials legitimately sit at lower effective rank than classical ones — a
/// weight-preserving ansatz confines the state to one sector — so gating them
/// on a classically-calibrated threshold would prune the whole family.
#[derive(Debug, Clone)]
pub struct GatePruner {
    pub warmup_trials: usize,
    pub warmup_steps: usize,
    pub grad_norm_floor: f64,
    /// Chance level of the reported metric (e.g. `1/n_classes`); `None` disables.
    pub no_skill_floor: Option<f64>,
    pub loss_slope_min_delta: f64,
    pub loss_slope_window: usize,
    pub relaxation: f64,
    /// Higher-is-better for the reported value (mirrors the study direction).
    pub maximize_value: bool,
    calib: Vec<(f64, f64)>,
    thresholds: Option<(f64, f64)>,
}

impl Default for GatePruner {
    fn default() -> Self {
        Self {
            warmup_trials: 15,
            warmup_steps: 3,
            grad_norm_floor: 1e-6,
            no_skill_floor: None,
            loss_slope_min_delta: 1e-4,
            loss_slope_window: 5,
            relaxation: 1.0,
            maximize_value: true,
            calib: Vec::new(),
            thresholds: None,
        }
    }
}

impl GatePruner {
    pub fn new(warmup_trials: usize) -> Self {
        Self {
            warmup_trials,
            ..Default::default()
        }
    }

    pub fn with_relaxation(mut self, relaxation: f64) -> Self {
        self.relaxation = relaxation.max(1.0);
        self
    }
    pub fn with_grad_norm_floor(mut self, floor: f64) -> Self {
        self.grad_norm_floor = floor;
        self
    }
    pub fn with_no_skill_floor(mut self, floor: f64) -> Self {
        self.no_skill_floor = Some(floor);
        self
    }
    pub fn with_loss_slope(mut self, min_delta: f64, window: usize) -> Self {
        self.loss_slope_min_delta = min_delta;
        self.loss_slope_window = window.max(2);
        self
    }

    /// Have the collapse thresholds been calibrated yet?
    pub fn is_calibrated(&self) -> bool {
        self.thresholds.is_some()
    }

    /// The calibrated `(eff_rank_floor, top2_ceiling)`, once available.
    pub fn thresholds(&self) -> Option<(f64, f64)> {
        self.thresholds
    }

    /// Absorb completed trials into the calibration set, and once
    /// `warmup_trials` are in, fix the thresholds.
    fn calibrate(&mut self, history: &[Trial]) {
        if self.thresholds.is_some() {
            return;
        }
        self.calib = history
            .iter()
            .filter(|t| matches!(t.state, TrialState::Complete(_)))
            .filter_map(|t| {
                let last = t.steps.last()?;
                Some((last.aux_get("eff_rank")?, last.aux_get("top2_cumevr")?))
            })
            .collect();
        if self.calib.len() < self.warmup_trials {
            return;
        }
        let n = self.calib.len() as f64;
        let mean = |f: fn(&(f64, f64)) -> f64, c: &[(f64, f64)]| -> f64 {
            c.iter().map(f).sum::<f64>() / n
        };
        let std = |f: fn(&(f64, f64)) -> f64, c: &[(f64, f64)], m: f64| -> f64 {
            (c.iter().map(|v| (f(v) - m).powi(2)).sum::<f64>() / n).sqrt()
        };
        let m_er = mean(|v| v.0, &self.calib);
        let s_er = std(|v| v.0, &self.calib, m_er);
        let m_t2 = mean(|v| v.1, &self.calib);
        let s_t2 = std(|v| v.1, &self.calib, m_t2);
        // One standard deviation out, then widened by `relaxation`.
        let er_floor = (m_er - s_er) / self.relaxation;
        let t2_ceiling = ((m_t2 + s_t2) * self.relaxation).min(1.0);
        self.thresholds = Some((er_floor, t2_ceiling));
    }
}

impl Pruner for GatePruner {
    fn should_prune(&mut self, trial: &Trial, history: &[Trial], _direction: Direction) -> bool {
        self.calibrate(history);
        let Some(last) = trial.steps.last() else {
            return false;
        };
        if trial.steps.len() <= self.warmup_steps {
            return false;
        }

        // 1. Vanishing gradient.
        if let Some(g) = last.aux_get("grad_norm") {
            if g < self.grad_norm_floor {
                return true;
            }
        }
        // 2. No skill against a chance baseline.
        if let Some(floor) = self.no_skill_floor {
            let hopeless = if self.maximize_value {
                last.value <= floor
            } else {
                last.value >= floor
            };
            if hopeless {
                return true;
            }
        }
        // 3. Flat loss slope over the window.
        if trial.steps.len() >= self.loss_slope_window {
            let w = &trial.steps[trial.steps.len() - self.loss_slope_window..];
            let (first, lastv) = (w[0].value, w[w.len() - 1].value);
            let improvement = if self.maximize_value {
                lastv - first
            } else {
                first - lastv
            };
            if improvement < self.loss_slope_min_delta {
                return true;
            }
        }
        // 4. Expressivity collapse — only once calibrated.
        if let Some((er_floor, t2_ceiling)) = self.thresholds {
            if let Some(er) = last.aux_get("eff_rank") {
                if er < er_floor {
                    return true;
                }
            }
            if let Some(t2) = last.aux_get("top2_cumevr") {
                if t2 > t2_ceiling {
                    return true;
                }
            }
        }
        false
    }

    fn name(&self) -> &'static str {
        "gate"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trial::StepReport;

    fn trial_with(id: usize, values: &[f64], state: TrialState) -> Trial {
        let mut t = Trial::new(id, vec![0], vec![]);
        for (i, v) in values.iter().enumerate() {
            t.steps.push(StepReport {
                step: i,
                value: *v,
                aux: vec![],
            });
        }
        t.state = state;
        t
    }

    fn with_aux(id: usize, values: &[f64], aux: &[(&str, f64)], state: TrialState) -> Trial {
        let mut t = trial_with(id, values, state);
        if let Some(last) = t.steps.last_mut() {
            last.aux = aux.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        }
        t
    }

    #[test]
    fn median_prunes_the_plateaued_trial_and_spares_the_improving_one() {
        // Accept: MedianPruner prunes a plateaued trial, spares an improving one.
        let peers: Vec<Trial> = (0..4)
            .map(|i| trial_with(i, &[0.5, 0.6, 0.7, 0.8], TrialState::Complete(0.8)))
            .collect();

        let mut p = MedianPruner::new(3, 2);
        let plateaued = trial_with(9, &[0.5, 0.5, 0.5, 0.5], TrialState::Running);
        let mut hist = peers.clone();
        hist.push(plateaued.clone());
        assert!(
            p.should_prune(&plateaued, &hist, Direction::Maximize),
            "plateaued trial survived"
        );

        let improving = trial_with(10, &[0.5, 0.7, 0.85, 0.95], TrialState::Running);
        let mut hist2 = peers.clone();
        hist2.push(improving.clone());
        assert!(
            !p.should_prune(&improving, &hist2, Direction::Maximize),
            "improving trial was pruned"
        );
    }

    #[test]
    fn median_respects_both_warmups() {
        let mut p = MedianPruner::new(3, 2);
        let bad = trial_with(9, &[0.0, 0.0, 0.0, 0.0], TrialState::Running);
        // Too few completed peers ⇒ no pruning even though the trial is awful.
        let few: Vec<Trial> = (0..2)
            .map(|i| trial_with(i, &[1.0, 1.0, 1.0, 1.0], TrialState::Complete(1.0)))
            .collect();
        assert!(!p.should_prune(&bad, &few, Direction::Maximize));

        // Enough peers, but the trial is still inside its step grace period.
        let many: Vec<Trial> = (0..4)
            .map(|i| trial_with(i, &[1.0, 1.0, 1.0, 1.0], TrialState::Complete(1.0)))
            .collect();
        let young = trial_with(9, &[0.0, 0.0], TrialState::Running);
        assert!(!p.should_prune(&young, &many, Direction::Maximize));
    }

    #[test]
    fn median_handles_minimisation() {
        let mut p = MedianPruner::new(2, 1);
        let peers: Vec<Trial> = (0..3)
            .map(|i| trial_with(i, &[1.0, 0.5, 0.2], TrialState::Complete(0.2)))
            .collect();
        // Higher loss than the median ⇒ prune when minimising.
        let worse = trial_with(9, &[1.0, 0.9, 0.9], TrialState::Running);
        let mut h = peers.clone();
        h.push(worse.clone());
        assert!(p.should_prune(&worse, &h, Direction::Minimize));
        let better = trial_with(10, &[1.0, 0.3, 0.05], TrialState::Running);
        let mut h2 = peers;
        h2.push(better.clone());
        assert!(!p.should_prune(&better, &h2, Direction::Minimize));
    }

    #[test]
    fn successive_halving_cuts_the_bottom_at_a_rung() {
        let mut p = SuccessiveHalving::new(vec![2], 2.0);
        let peers: Vec<Trial> = (0..4)
            .map(|i| trial_with(i, &[0.0, 0.0, 0.9 - 0.1 * i as f64], TrialState::Running))
            .collect();
        // Worst of five at the rung ⇒ pruned.
        let worst = trial_with(9, &[0.0, 0.0, 0.1], TrialState::Running);
        let mut h = peers.clone();
        h.push(worst.clone());
        assert!(p.should_prune(&worst, &h, Direction::Maximize));
        // Best ⇒ kept.
        let best = trial_with(10, &[0.0, 0.0, 0.99], TrialState::Running);
        let mut h2 = peers;
        h2.push(best.clone());
        assert!(!p.should_prune(&best, &h2, Direction::Maximize));
    }

    #[test]
    fn successive_halving_only_fires_on_rung_steps() {
        let mut p = SuccessiveHalving::new(vec![5], 2.0);
        let peers: Vec<Trial> = (0..4)
            .map(|i| trial_with(i, &[0.9, 0.9, 0.9], TrialState::Running))
            .collect();
        let worst = trial_with(9, &[0.0, 0.0, 0.0], TrialState::Running);
        let mut h = peers;
        h.push(worst.clone());
        // Last step is 2, not a rung ⇒ untouched.
        assert!(!p.should_prune(&worst, &h, Direction::Maximize));
    }

    #[test]
    fn gate_calibration_activates_only_after_warmup() {
        // Accept: GatePruner calibration activates only after warmup.
        let mut p = GatePruner::new(15).with_loss_slope(-1e9, 5); // disable slope gate
                                                                  // A collapsed trial (rank 1) that WOULD trip the collapse gate.
        let collapsed = with_aux(
            99,
            &[0.5, 0.5, 0.5, 0.5, 0.5],
            &[("eff_rank", 1.0), ("top2_cumevr", 1.0)],
            TrialState::Running,
        );

        // Fewer than 15 completed trials ⇒ not calibrated, no collapse gating.
        let short: Vec<Trial> = (0..10)
            .map(|i| {
                with_aux(
                    i,
                    &[0.5; 5],
                    &[("eff_rank", 4.0), ("top2_cumevr", 0.5)],
                    TrialState::Complete(0.8),
                )
            })
            .collect();
        assert!(!p.should_prune(&collapsed, &short, Direction::Maximize));
        assert!(!p.is_calibrated(), "calibrated too early");

        // With 15 completed trials the thresholds fix and the gate fires.
        let full: Vec<Trial> = (0..15)
            .map(|i| {
                with_aux(
                    i,
                    &[0.5; 5],
                    &[("eff_rank", 4.0), ("top2_cumevr", 0.5)],
                    TrialState::Complete(0.8),
                )
            })
            .collect();
        assert!(p.should_prune(&collapsed, &full, Direction::Maximize));
        assert!(p.is_calibrated());
        let (er, t2) = p.thresholds().unwrap();
        assert!((er - 4.0).abs() < 1e-9, "eff_rank floor {er}");
        assert!((t2 - 0.5).abs() < 1e-9, "top2 ceiling {t2}");
    }

    #[test]
    fn gate_relaxation_widens_the_calibrated_thresholds() {
        let build = |relax: f64| {
            let mut p = GatePruner::new(4)
                .with_relaxation(relax)
                .with_loss_slope(-1e9, 5);
            let hist: Vec<Trial> = (0..4)
                .map(|i| {
                    with_aux(
                        i,
                        &[0.5; 5],
                        &[("eff_rank", 4.0), ("top2_cumevr", 0.4)],
                        TrialState::Complete(0.8),
                    )
                })
                .collect();
            let probe = with_aux(9, &[0.5; 5], &[], TrialState::Running);
            p.should_prune(&probe, &hist, Direction::Maximize);
            p.thresholds().unwrap()
        };
        let (er1, t21) = build(1.0);
        let (er2, t22) = build(2.0);
        assert!(er2 < er1, "relaxation must LOWER the eff_rank floor");
        assert!(t22 > t21, "relaxation must RAISE the top2 ceiling");
    }

    #[test]
    fn gate_vanishing_gradient_and_no_skill() {
        let mut p = GatePruner::new(1000) // never calibrates — isolate the gates
            .with_grad_norm_floor(1e-4)
            .with_loss_slope(-1e9, 5);
        let flat_grad = with_aux(
            1,
            &[0.6, 0.6, 0.6, 0.6],
            &[("grad_norm", 1e-9)],
            TrialState::Running,
        );
        assert!(p.should_prune(&flat_grad, &[], Direction::Maximize));

        let healthy = with_aux(
            2,
            &[0.6, 0.7, 0.8, 0.9],
            &[("grad_norm", 1.0)],
            TrialState::Running,
        );
        assert!(!p.should_prune(&healthy, &[], Direction::Maximize));

        let mut q = GatePruner::new(1000)
            .with_no_skill_floor(0.5)
            .with_loss_slope(-1e9, 5);
        let chance = trial_with(3, &[0.5, 0.5, 0.5, 0.5], TrialState::Running);
        assert!(q.should_prune(&chance, &[], Direction::Maximize));
    }

    #[test]
    fn gate_loss_slope_needs_real_improvement() {
        let mut p = GatePruner::new(1000).with_loss_slope(0.01, 4);
        let creeping = trial_with(1, &[0.50, 0.501, 0.502, 0.503], TrialState::Running);
        assert!(p.should_prune(&creeping, &[], Direction::Maximize));
        let learning = trial_with(2, &[0.50, 0.60, 0.70, 0.80], TrialState::Running);
        assert!(!p.should_prune(&learning, &[], Direction::Maximize));
    }

    #[test]
    fn no_pruner_never_prunes() {
        let mut p = NoPruner;
        let awful = trial_with(1, &[0.0; 20], TrialState::Running);
        assert!(!p.should_prune(&awful, &[], Direction::Maximize));
    }
}
