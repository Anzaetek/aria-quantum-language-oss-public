// SPDX-License-Identifier: Apache-2.0
//! `aria-tune` — a small, **dependency-free** optimisation engine for tuning
//! Aria programs.
//!
//! The loop is `ask` / `report` / `tell`:
//!
//! ```
//! use aria_tune::{Direction, MedianPruner, Space, Study, TpeSampler};
//!
//! let space = Space::new()
//!     .int("layers", 1, 4, 1)
//!     .log_float("lr", 1e-3, 3e-1, 8)
//!     .categorical("optimizer", &["gd", "adam"]);
//!
//! let mut study = Study::new(space, Direction::Maximize)
//!     .with_sampler(Box::new(TpeSampler::new(20260729)))
//!     .with_pruner(Box::new(MedianPruner::default()));
//!
//! for _ in 0..12 {
//!     let trial = study.ask();
//!     let layers = trial.int("layers").unwrap();
//!     let lr = trial.float("lr").unwrap();
//!     let mut acc = 0.0;
//!     for epoch in 0..5 {
//!         acc = 1.0 - (lr - 0.05).abs() - 0.02 * (layers as f64 - 3.0).abs()
//!             + 0.01 * epoch as f64;
//!         study.report(trial.id, epoch, acc, &[("grad_norm".into(), 0.1)]);
//!         if study.should_prune(trial.id) { break; }
//!     }
//!     study.tell(trial.id, acc);
//! }
//! assert!(study.best().is_some());
//! ```
//!
//! # Design
//!
//! * **No dependencies.** Samplers, pruners, RNG, CSV and JSON are all
//!   in-tree, so a study is reproducible bit-for-bit from its seed and the
//!   crate builds anywhere the workspace does.
//! * **Everything is a grid.** [`Space`] discretises every dimension —
//!   integer, float, log-float, categorical — to an index grid, so one
//!   categorical TPE covers continuous dimensions too and two trials are
//!   equal exactly when their index vectors are.
//! * **Pruners only read what you report.** They never compute a metric
//!   themselves; the caller supplies `grad_norm`, `eff_rank` and friends as
//!   `aux`. That is what keeps a rich domain gate ([`GatePruner`]) free of
//!   any tensor or linear-algebra dependency.

pub mod pruner;
pub mod rng;
pub mod sampler;
pub mod space;
pub mod study;
pub mod trial;

pub use pruner::{GatePruner, MedianPruner, NoPruner, Pruner, SuccessiveHalving};
pub use rng::Rng;
pub use sampler::{Direction, GridSampler, Observation, RandomSampler, Sampler, TpeSampler};
pub use space::{Param, ParamValue, Space};
pub use study::Study;
pub use trial::{StepReport, Trial, TrialState};

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: a study that must find the peak of a separable objective
    /// while pruning the hopeless trials.
    #[test]
    fn end_to_end_study_finds_the_optimum_and_prunes() {
        let space = Space::new()
            .int("n", 2, 10, 2)
            .log_float("lr", 1e-3, 1e0, 7)
            .categorical("opt", &["gd", "adam"]);

        let mut study = Study::new(space, Direction::Maximize)
            .with_sampler(Box::new(TpeSampler::new(11)))
            .with_pruner(Box::new(MedianPruner::new(4, 1)));

        // Objective peaks at n = 6, lr ≈ 1e-1, opt = adam.
        let objective = |n: i64, lr: f64, opt: &str| -> f64 {
            let a = -0.05 * (n as f64 - 6.0).abs();
            let b = -0.5 * (lr.log10() + 1.0).abs();
            let c = if opt == "adam" { 0.2 } else { 0.0 };
            0.6 + a + b + c
        };

        for _ in 0..40 {
            let t = study.ask();
            let (n, lr, opt) = (
                t.int("n").unwrap(),
                t.float("lr").unwrap(),
                t.cat("opt").unwrap().to_string(),
            );
            let target = objective(n, lr, &opt);
            let mut value = 0.0;
            for epoch in 0..6 {
                // Converge toward the target.
                value = target * (1.0 - 0.7_f64.powi(epoch + 1));
                study.report(t.id, epoch as usize, value, &[("grad_norm".into(), 0.5)]);
                if study.should_prune(t.id) {
                    break;
                }
            }
            study.tell(t.id, value);
        }

        assert_eq!(study.trials().len(), 40);
        assert!(study.n_pruned() >= 1, "nothing was pruned");
        let best = study.best().expect("a completed trial");
        assert!(
            (best.int("n").unwrap() - 6).abs() <= 2,
            "n = {:?} far from the optimum",
            best.int("n")
        );
        // The winner must beat the field, and land in the objective's
        // positive region — which is a small slice of this space, so getting
        // there is evidence of search rather than luck.
        assert!(best.state.score().unwrap() > study.mean_score().unwrap());
        assert!(
            best.state.score().unwrap() > 0.0,
            "best score {:?} never reached the good region",
            best.state.score()
        );
    }

    #[test]
    fn tpe_outperforms_random_on_a_space_too_large_to_brute_force() {
        // The engine's reason to exist. The space must be big enough that 40
        // random draws cannot cover it — on a small grid random simply
        // enumerates most of the points and there is nothing to beat.
        let objective = |a: i64, b: f64, c: i64| {
            -0.05 * (a as f64 - 9.0).abs() - (b.log10() + 1.0).abs() - 0.05 * (c as f64 - 3.0).abs()
        };
        let space = || {
            Space::new()
                .int("a", 1, 12, 1)
                .log_float("b", 1e-3, 1e0, 12)
                .int("c", 1, 12, 1)
        };
        assert_eq!(space().size(), 12 * 12 * 12, "space shrank");

        let run = |sampler: Box<dyn Sampler>| -> f64 {
            let mut s = Study::new(space(), Direction::Maximize).with_sampler(sampler);
            for _ in 0..40 {
                let t = s.ask();
                let v = objective(
                    t.int("a").unwrap(),
                    t.float("b").unwrap(),
                    t.int("c").unwrap(),
                );
                s.tell(t.id, v);
            }
            s.best().unwrap().state.score().unwrap()
        };
        let seeds = 12u64;
        let (mut tpe, mut rnd) = (0.0, 0.0);
        for seed in 0..seeds {
            tpe += run(Box::new(TpeSampler::new(seed)));
            rnd += run(Box::new(RandomSampler::new(seed)));
        }
        let (tpe, rnd) = (tpe / seeds as f64, rnd / seeds as f64);
        assert!(tpe >= rnd, "TPE mean best {tpe} < random mean best {rnd}");
    }

    #[test]
    fn a_study_with_no_trials_is_well_defined() {
        let s = Study::new(Space::new().int("n", 1, 3, 1), Direction::Maximize);
        assert!(s.best().is_none());
        assert!(s.mean_score().is_none());
        assert_eq!(s.n_pruned(), 0);
        assert_eq!(s.to_csv().trim_end(), "id,state,score,steps,n");
        assert!(s.to_json().contains("\"trials\":[]"));
    }
}
