// SPDX-License-Identifier: Apache-2.0
//! Trials and their intermediate reports.

use crate::space::ParamValue;

/// One intermediate report from a running trial.
#[derive(Debug, Clone, PartialEq)]
pub struct StepReport {
    pub step: usize,
    pub value: f64,
    /// Caller-computed auxiliary metrics (`grad_norm`, `eff_rank`, …). The
    /// pruners read these by name; this crate never computes them.
    pub aux: Vec<(String, f64)>,
}

impl StepReport {
    /// Look up an auxiliary metric by name.
    pub fn aux_get(&self, key: &str) -> Option<f64> {
        self.aux.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }
}

/// Lifecycle of a trial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrialState {
    Running,
    Complete(f64),
    Pruned,
}

impl TrialState {
    pub fn as_str(&self) -> &'static str {
        match self {
            TrialState::Running => "running",
            TrialState::Complete(_) => "complete",
            TrialState::Pruned => "pruned",
        }
    }
    pub fn score(&self) -> Option<f64> {
        match self {
            TrialState::Complete(s) => Some(*s),
            _ => None,
        }
    }
}

/// One point of the search space, its reports, and its outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    pub id: usize,
    /// Grid indices — the canonical identity of the point.
    pub combo: Vec<usize>,
    /// Decoded, named values.
    pub params: Vec<(String, ParamValue)>,
    pub steps: Vec<StepReport>,
    pub state: TrialState,
}

impl Trial {
    pub fn new(id: usize, combo: Vec<usize>, params: Vec<(String, ParamValue)>) -> Self {
        Self {
            id,
            combo,
            params,
            steps: Vec::new(),
            state: TrialState::Running,
        }
    }

    /// Named parameter value.
    pub fn get(&self, name: &str) -> Option<&ParamValue> {
        self.params.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
    /// Named integer parameter (integers only).
    pub fn int(&self, name: &str) -> Option<i64> {
        self.get(name).and_then(|v| v.as_int())
    }
    /// Named numeric parameter (integers widen to float).
    pub fn float(&self, name: &str) -> Option<f64> {
        self.get(name).and_then(|v| v.as_float())
    }
    /// Named categorical choice.
    pub fn cat(&self, name: &str) -> Option<&str> {
        self.get(name).and_then(|v| v.as_str())
    }

    /// The most recent reported value.
    pub fn last_value(&self) -> Option<f64> {
        self.steps.last().map(|s| s.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Trial {
        Trial::new(
            3,
            vec![1, 0],
            vec![
                ("n".into(), ParamValue::Int(8)),
                ("lr".into(), ParamValue::Float(0.01)),
                ("opt".into(), ParamValue::Str("adam".into())),
            ],
        )
    }

    #[test]
    fn typed_accessors() {
        let t = sample();
        assert_eq!(t.int("n"), Some(8));
        assert_eq!(t.float("lr"), Some(0.01));
        assert_eq!(t.cat("opt"), Some("adam"));
        // Integers widen to float, categoricals do not convert.
        assert_eq!(t.float("n"), Some(8.0));
        assert_eq!(t.int("lr"), None);
        assert_eq!(t.cat("n"), None);
        assert_eq!(t.int("missing"), None);
    }

    #[test]
    fn aux_lookup_and_last_value() {
        let mut t = sample();
        assert_eq!(t.last_value(), None);
        t.steps.push(StepReport {
            step: 0,
            value: 0.4,
            aux: vec![("grad_norm".into(), 0.02)],
        });
        t.steps.push(StepReport {
            step: 1,
            value: 0.7,
            aux: vec![],
        });
        assert_eq!(t.last_value(), Some(0.7));
        assert_eq!(t.steps[0].aux_get("grad_norm"), Some(0.02));
        assert_eq!(t.steps[0].aux_get("nope"), None);
        assert_eq!(t.steps[1].aux_get("grad_norm"), None);
    }

    #[test]
    fn state_reporting() {
        assert_eq!(TrialState::Running.as_str(), "running");
        assert_eq!(TrialState::Pruned.as_str(), "pruned");
        assert_eq!(TrialState::Complete(0.9).as_str(), "complete");
        assert_eq!(TrialState::Complete(0.9).score(), Some(0.9));
        assert_eq!(TrialState::Running.score(), None);
        assert_eq!(TrialState::Pruned.score(), None);
    }
}
