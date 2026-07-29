// SPDX-License-Identifier: Apache-2.0
//! The `ask` / `report` / `tell` loop.

use crate::pruner::{NoPruner, Pruner};
use crate::sampler::{Direction, Observation, RandomSampler, Sampler};
use crate::space::Space;
use crate::trial::{StepReport, Trial, TrialState};

/// An optimisation study over a [`Space`].
///
/// ```
/// use aria_tune::{Direction, Space, Study, TpeSampler};
/// let mut study = Study::new(Space::new().int("n", 2, 8, 2), Direction::Maximize)
///     .with_sampler(Box::new(TpeSampler::new(7)));
/// for _ in 0..10 {
///     let t = study.ask();
///     let n = t.int("n").unwrap();
///     study.tell(t.id, -(n as f64 - 6.0).abs());
/// }
/// assert_eq!(study.best().unwrap().int("n"), Some(6));
/// ```
pub struct Study {
    space: Space,
    direction: Direction,
    sampler: Box<dyn Sampler>,
    pruner: Box<dyn Pruner>,
    trials: Vec<Trial>,
}

impl Study {
    pub fn new(space: Space, direction: Direction) -> Self {
        Self {
            space,
            direction,
            sampler: Box::new(RandomSampler::new(0)),
            pruner: Box::new(NoPruner),
            trials: Vec::new(),
        }
    }

    pub fn with_sampler(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.sampler = sampler;
        self
    }
    pub fn with_pruner(mut self, pruner: Box<dyn Pruner>) -> Self {
        self.pruner = pruner;
        self
    }

    pub fn space(&self) -> &Space {
        &self.space
    }
    pub fn direction(&self) -> Direction {
        self.direction
    }
    pub fn trials(&self) -> &[Trial] {
        &self.trials
    }
    pub fn sampler_name(&self) -> &'static str {
        self.sampler.name()
    }
    pub fn pruner_name(&self) -> &'static str {
        self.pruner.name()
    }

    /// Completed observations, in trial order — what the sampler learns from.
    /// Pruned and running trials are excluded: a pruned trial's score is a
    /// lower bound, not a measurement, and feeding it in would teach the
    /// sampler that the region is worse than it has been shown to be.
    fn history(&self) -> Vec<Observation> {
        self.trials
            .iter()
            .filter_map(|t| t.state.score().map(|s| (t.combo.clone(), s)))
            .collect()
    }

    /// Propose the next trial and register it as running.
    pub fn ask(&mut self) -> Trial {
        let history = self.history();
        let combo = self.sampler.suggest(&self.space, &history, self.direction);
        let params = self.space.decode(&combo);
        let trial = Trial::new(self.trials.len(), combo, params);
        self.trials.push(trial.clone());
        trial
    }

    /// Record an intermediate value (plus any auxiliary metrics) for a trial.
    pub fn report(&mut self, id: usize, step: usize, value: f64, aux: &[(String, f64)]) {
        if let Some(t) = self.trials.get_mut(id) {
            t.steps.push(StepReport {
                step,
                value,
                aux: aux.to_vec(),
            });
        }
    }

    /// Ask the pruner whether trial `id` should stop. Marks it `Pruned` when
    /// the answer is yes, so a caller that ignores the return value still ends
    /// up with a consistent study.
    pub fn should_prune(&mut self, id: usize) -> bool {
        let Some(trial) = self.trials.get(id).cloned() else {
            return false;
        };
        if !matches!(trial.state, TrialState::Running) {
            return false;
        }
        let verdict = self
            .pruner
            .should_prune(&trial, &self.trials, self.direction);
        if verdict {
            if let Some(t) = self.trials.get_mut(id) {
                t.state = TrialState::Pruned;
            }
        }
        verdict
    }

    /// Finish a trial with its final score. A pruned trial stays pruned.
    ///
    /// A **non-finite** score (`NaN`/`±inf` — a diverged run) is recorded as
    /// [`TrialState::Pruned`], not `Complete`. It is not a measurement, and
    /// letting it through would corrupt everything downstream: `NaN` compares
    /// false against every value, so it could be returned by [`Self::best`]
    /// purely by position, it sorts arbitrarily in the sampler's ranking, and
    /// it would serialise as invalid JSON.
    pub fn tell(&mut self, id: usize, score: f64) {
        if let Some(t) = self.trials.get_mut(id) {
            if matches!(t.state, TrialState::Running) {
                t.state = if score.is_finite() {
                    TrialState::Complete(score)
                } else {
                    TrialState::Pruned
                };
            }
        }
    }

    /// The best completed trial.
    pub fn best(&self) -> Option<&Trial> {
        self.trials
            .iter()
            .filter(|t| t.state.score().is_some())
            .reduce(|a, b| {
                let (sa, sb) = (a.state.score().unwrap(), b.state.score().unwrap());
                if self.direction.at_least_as_good(sa, sb) {
                    a
                } else {
                    b
                }
            })
    }

    /// Mean score over completed trials.
    pub fn mean_score(&self) -> Option<f64> {
        let scores: Vec<f64> = self.trials.iter().filter_map(|t| t.state.score()).collect();
        if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }

    pub fn n_pruned(&self) -> usize {
        self.trials
            .iter()
            .filter(|t| matches!(t.state, TrialState::Pruned))
            .count()
    }
    pub fn n_complete(&self) -> usize {
        self.trials
            .iter()
            .filter(|t| matches!(t.state, TrialState::Complete(_)))
            .count()
    }

    /// One header row plus one row per trial.
    pub fn to_csv(&self) -> String {
        let mut out = String::from("id,state,score,steps");
        for name in self.space.names() {
            out.push(',');
            out.push_str(name);
        }
        out.push('\n');
        for t in &self.trials {
            out.push_str(&format!(
                "{},{},{},{}",
                t.id,
                t.state.as_str(),
                t.state
                    .score()
                    .map(|s| format!("{s:.6}"))
                    .unwrap_or_default(),
                t.steps.len()
            ));
            for (_, v) in &t.params {
                out.push(',');
                out.push_str(&csv_escape(&v.to_string()));
            }
            out.push('\n');
        }
        out
    }

    /// Hand-rolled JSON — this crate deliberately has no dependencies.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\"direction\":\"");
        out.push_str(match self.direction {
            Direction::Maximize => "maximize",
            Direction::Minimize => "minimize",
        });
        out.push_str("\",\"sampler\":\"");
        out.push_str(self.sampler.name());
        out.push_str("\",\"pruner\":\"");
        out.push_str(self.pruner.name());
        out.push_str("\",\"trials\":[");
        for (i, t) in self.trials.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"id\":{},\"state\":\"{}\"",
                t.id,
                t.state.as_str()
            ));
            if let Some(s) = t.state.score() {
                out.push_str(&format!(",\"score\":{}", json_number(s)));
            }
            out.push_str(",\"params\":{");
            for (j, (name, v)) in t.params.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str(&format!("\"{}\":", json_escape(name)));
                match v {
                    crate::space::ParamValue::Str(s) => {
                        out.push_str(&format!("\"{}\"", json_escape(s)))
                    }
                    crate::space::ParamValue::Float(f) => out.push_str(&json_number(*f)),
                    crate::space::ParamValue::Int(i) => out.push_str(&i.to_string()),
                }
            }
            out.push_str("}}");
        }
        out.push_str("]}");
        out
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// JSON has no `NaN` or `Infinity` literal — Rust's `Display` for `f64` emits
/// `NaN` / `inf`, which every JSON parser rejects. Non-finite values become
/// `null`, the standard representation.
fn json_number(v: f64) -> String {
    if v.is_finite() {
        v.to_string()
    } else {
        "null".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruner::MedianPruner;
    use crate::sampler::{GridSampler, TpeSampler};

    fn space() -> Space {
        Space::new()
            .int("n", 2, 8, 2)
            .categorical("opt", &["gd", "adam"])
    }

    #[test]
    fn ask_tell_records_and_ranks() {
        let mut s =
            Study::new(space(), Direction::Maximize).with_sampler(Box::new(GridSampler::new()));
        for _ in 0..8 {
            let t = s.ask();
            let score = t.int("n").unwrap() as f64
                + if t.cat("opt") == Some("adam") {
                    1.0
                } else {
                    0.0
                };
            s.tell(t.id, score);
        }
        assert_eq!(s.trials().len(), 8);
        assert_eq!(s.n_complete(), 8);
        let best = s.best().unwrap();
        assert_eq!(best.int("n"), Some(8));
        assert_eq!(best.cat("opt"), Some("adam"));
        assert_eq!(best.state.score(), Some(9.0));
    }

    #[test]
    fn best_respects_minimisation() {
        let mut s =
            Study::new(space(), Direction::Minimize).with_sampler(Box::new(GridSampler::new()));
        for _ in 0..8 {
            let t = s.ask();
            let n = t.int("n").unwrap() as f64;
            s.tell(t.id, n);
        }
        assert_eq!(s.best().unwrap().int("n"), Some(2));
    }

    #[test]
    fn csv_has_one_row_per_trial_plus_a_header() {
        // Accept: CSV rows == trials.
        let mut s =
            Study::new(space(), Direction::Maximize).with_sampler(Box::new(GridSampler::new()));
        for _ in 0..5 {
            let t = s.ask();
            s.tell(t.id, t.int("n").unwrap() as f64);
        }
        let csv = s.to_csv();
        let lines: Vec<&str> = csv.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 6, "header + 5 trials");
        assert_eq!(lines[0], "id,state,score,steps,n,opt");
        assert_eq!(lines[1..].len(), s.trials().len());
        for (i, line) in lines[1..].iter().enumerate() {
            assert!(
                line.starts_with(&format!("{i},complete,")),
                "row {i}: {line}"
            );
            assert_eq!(line.split(',').count(), 6);
        }
    }

    #[test]
    fn json_round_trips_the_shape() {
        let mut s =
            Study::new(space(), Direction::Minimize).with_sampler(Box::new(GridSampler::new()));
        let t = s.ask();
        s.tell(t.id, 1.5);
        let t2 = s.ask();
        s.report(t2.id, 0, 0.1, &[]);
        let json = s.to_json();
        assert!(json.starts_with("{\"direction\":\"minimize\""));
        assert!(json.contains("\"sampler\":\"grid\""));
        assert!(json.contains("\"pruner\":\"none\""));
        assert!(json.contains("\"score\":1.5"));
        assert!(json.contains("\"opt\":\"gd\""));
        // The running trial has no score field.
        assert_eq!(json.matches("\"score\"").count(), 1);
        assert_eq!(json.matches("\"id\"").count(), 2);
    }

    #[test]
    fn pruned_trials_are_excluded_from_the_sampler_history() {
        // A pruned trial's score is a lower bound, not a measurement.
        let mut s = Study::new(space(), Direction::Maximize)
            .with_sampler(Box::new(GridSampler::new()))
            .with_pruner(Box::new(MedianPruner::new(1, 0)));
        let a = s.ask();
        s.report(a.id, 0, 0.9, &[]);
        s.tell(a.id, 0.9);
        let b = s.ask();
        s.report(b.id, 0, 0.1, &[]);
        assert!(s.should_prune(b.id), "clearly-worse trial not pruned");
        // tell() must not resurrect it.
        s.tell(b.id, 0.1);
        assert_eq!(s.trials()[b.id].state, TrialState::Pruned);
        assert_eq!(s.n_pruned(), 1);
        assert_eq!(s.n_complete(), 1);
        assert_eq!(s.history().len(), 1);
        assert_eq!(s.mean_score(), Some(0.9));
    }

    #[test]
    fn should_prune_is_idempotent_and_ignores_unknown_ids() {
        let mut s =
            Study::new(space(), Direction::Maximize).with_pruner(Box::new(MedianPruner::new(1, 0)));
        assert!(!s.should_prune(999));
        let a = s.ask();
        s.report(a.id, 0, 1.0, &[]);
        s.tell(a.id, 1.0);
        let b = s.ask();
        s.report(b.id, 0, 0.0, &[]);
        assert!(s.should_prune(b.id));
        // Already pruned ⇒ false the second time (not a re-prune).
        assert!(!s.should_prune(b.id));
    }

    #[test]
    fn a_full_tpe_study_is_reproducible() {
        let run = || {
            let mut s = Study::new(space(), Direction::Maximize)
                .with_sampler(Box::new(TpeSampler::new(42)));
            for _ in 0..20 {
                let t = s.ask();
                let score =
                    t.int("n").unwrap() as f64 - if t.cat("opt") == Some("gd") { 3.0 } else { 0.0 };
                s.tell(t.id, score);
            }
            s.to_csv()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn csv_escapes_awkward_categorical_values() {
        let space = Space::new().categorical("k", &["plain", "with,comma"]);
        let mut s =
            Study::new(space, Direction::Maximize).with_sampler(Box::new(GridSampler::new()));
        for _ in 0..2 {
            let t = s.ask();
            s.tell(t.id, 0.0);
        }
        let csv = s.to_csv();
        assert!(csv.contains("\"with,comma\""), "{csv}");
    }
}
