// SPDX-License-Identifier: Apache-2.0
//! The search space.
//!
//! Every dimension — integer, float, log-float, or categorical — is
//! **discretised to an index grid**. That is the design decision the whole
//! crate rests on: a trial is then just a `Vec<usize>` of grid indices, so one
//! categorical TPE implementation covers continuous dimensions too, samplers
//! never need per-type special cases, and two trials are equal exactly when
//! their index vectors are.
//!
//! The cost is that a continuous dimension is only ever explored at its `res`
//! grid points. For tuning work that is the right trade: it makes the study
//! finite, enumerable by `GridSampler`, and reproducible.

/// One dimension of the search space.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    /// Integers `lo, lo+step, …` up to and including at most `hi`.
    Int { lo: i64, hi: i64, step: i64 },
    /// `res` evenly spaced values from `lo` to `hi` inclusive.
    Float { lo: f64, hi: f64, res: usize },
    /// `res` geometrically spaced values from `lo` to `hi` inclusive.
    /// Both bounds must be strictly positive.
    LogFloat { lo: f64, hi: f64, res: usize },
    /// A fixed list of named choices.
    Categorical(Vec<String>),
}

/// A concrete value drawn from a [`Param`].
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Str(String),
}

impl ParamValue {
    /// The integer value, or `None` for other kinds.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParamValue::Int(v) => Some(*v),
            _ => None,
        }
    }
    /// The float value; integers widen, categoricals do not convert.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParamValue::Float(v) => Some(*v),
            ParamValue::Int(v) => Some(*v as f64),
            ParamValue::Str(_) => None,
        }
    }
    /// The categorical choice, or `None` for numeric kinds.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParamValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::Int(v) => write!(f, "{v}"),
            ParamValue::Float(v) => write!(f, "{v}"),
            ParamValue::Str(s) => write!(f, "{s}"),
        }
    }
}

impl Param {
    /// Number of grid points. Always ≥ 1, so no dimension can be empty.
    pub fn cardinality(&self) -> usize {
        match self {
            Param::Int { lo, hi, step } => {
                let step = (*step).max(1);
                if hi < lo {
                    1
                } else {
                    (((hi - lo) / step) + 1) as usize
                }
            }
            Param::Float { res, .. } | Param::LogFloat { res, .. } => (*res).max(1),
            Param::Categorical(choices) => choices.len().max(1),
        }
    }

    /// The value at grid index `i` (clamped into range).
    pub fn value(&self, i: usize) -> ParamValue {
        let k = self.cardinality();
        let i = i.min(k - 1);
        match self {
            Param::Int { lo, step, .. } => ParamValue::Int(lo + (i as i64) * (*step).max(1)),
            Param::Float { lo, hi, .. } => {
                if k == 1 {
                    ParamValue::Float(*lo)
                } else {
                    ParamValue::Float(lo + (hi - lo) * i as f64 / (k - 1) as f64)
                }
            }
            Param::LogFloat { lo, hi, .. } => {
                // A non-positive bound would make `ln` produce NaN/-inf and
                // silently poison every downstream comparison, so fall back to
                // a linear grid rather than emit NaN. `validate` rejects such
                // a dimension up front; this is the belt-and-braces path for a
                // `Param` built by hand rather than through `Space`.
                if k == 1 || *lo <= 0.0 || *hi <= 0.0 {
                    ParamValue::Float(*lo)
                } else {
                    let (a, b) = (lo.ln(), hi.ln());
                    ParamValue::Float((a + (b - a) * i as f64 / (k - 1) as f64).exp())
                }
            }
            Param::Categorical(choices) => {
                ParamValue::Str(choices.get(i).cloned().unwrap_or_default())
            }
        }
    }

    /// Reject dimensions that cannot describe a usable grid.
    ///
    /// These are programmer errors, not runtime conditions, and each one
    /// otherwise fails *silently*: a non-positive `LogFloat` bound yields NaN
    /// for every point, and a zero or negative `Int` step quietly becomes 1,
    /// handing back a different space than the caller asked for.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Param::Int { lo, hi, step } => {
                if *step <= 0 {
                    return Err(format!("Int step must be ≥ 1, got {step}"));
                }
                if hi < lo {
                    return Err(format!("Int range is empty: lo {lo} > hi {hi}"));
                }
                Ok(())
            }
            Param::Float { lo, hi, res } => {
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(format!("Float bounds must be finite, got {lo}..{hi}"));
                }
                if *res == 0 {
                    return Err("Float res must be ≥ 1".into());
                }
                Ok(())
            }
            Param::LogFloat { lo, hi, res } => {
                // Finiteness first, so the positivity check below cannot be
                // handed a NaN — `NaN <= 0.0` is false, which would let a NaN
                // bound through and put NaN at every grid point.
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(format!("LogFloat bounds must be finite, got {lo}..{hi}"));
                }
                if *lo <= 0.0 || *hi <= 0.0 {
                    return Err(format!(
                        "LogFloat bounds must be strictly positive, got {lo}..{hi}"
                    ));
                }
                if *res == 0 {
                    return Err("LogFloat res must be ≥ 1".into());
                }
                Ok(())
            }
            Param::Categorical(choices) => {
                if choices.is_empty() {
                    return Err("Categorical needs at least one choice".into());
                }
                Ok(())
            }
        }
    }
}

/// A named list of dimensions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Space {
    dims: Vec<(String, Param)>,
}

impl Space {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a dimension. Chainable.
    ///
    /// # Panics
    /// If the dimension is malformed (see [`Param::validate`]). Building a
    /// space is compile-time-shaped work, and the alternatives — a silent
    /// NaN grid or a silently different step — are far worse to debug than a
    /// loud failure at construction. Use [`Space::try_add`] to handle it.
    pub fn add(self, name: &str, param: Param) -> Self {
        self.try_add(name, param)
            .unwrap_or_else(|e| panic!("aria-tune: invalid dimension `{name}`: {e}"))
    }

    /// Fallible [`Space::add`].
    pub fn try_add(mut self, name: &str, param: Param) -> Result<Self, String> {
        param.validate()?;
        self.dims.push((name.to_string(), param));
        Ok(self)
    }

    pub fn int(self, name: &str, lo: i64, hi: i64, step: i64) -> Self {
        self.add(name, Param::Int { lo, hi, step })
    }
    pub fn float(self, name: &str, lo: f64, hi: f64, res: usize) -> Self {
        self.add(name, Param::Float { lo, hi, res })
    }
    pub fn log_float(self, name: &str, lo: f64, hi: f64, res: usize) -> Self {
        self.add(name, Param::LogFloat { lo, hi, res })
    }
    pub fn categorical(self, name: &str, choices: &[&str]) -> Self {
        self.add(
            name,
            Param::Categorical(choices.iter().map(|s| s.to_string()).collect()),
        )
    }

    pub fn len(&self) -> usize {
        self.dims.len()
    }
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.dims.iter().map(|(n, _)| n.as_str())
    }
    pub fn params(&self) -> impl Iterator<Item = &Param> {
        self.dims.iter().map(|(_, p)| p)
    }
    pub fn get(&self, name: &str) -> Option<&Param> {
        self.dims.iter().find(|(n, _)| n == name).map(|(_, p)| p)
    }

    /// Grid cardinality of each dimension — what samplers actually index.
    pub fn dims(&self) -> Vec<usize> {
        self.dims.iter().map(|(_, p)| p.cardinality()).collect()
    }

    /// Total number of distinct points, saturating rather than overflowing.
    pub fn size(&self) -> usize {
        self.dims()
            .into_iter()
            .try_fold(1usize, |acc, k| acc.checked_mul(k))
            .unwrap_or(usize::MAX)
    }

    /// Decode an index vector into named values.
    pub fn decode(&self, combo: &[usize]) -> Vec<(String, ParamValue)> {
        self.dims
            .iter()
            .enumerate()
            .map(|(d, (name, p))| {
                let i = combo.get(d).copied().unwrap_or(0);
                (name.clone(), p.value(i))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_grid_endpoints_and_step() {
        let p = Param::Int {
            lo: 4,
            hi: 8,
            step: 2,
        };
        assert_eq!(p.cardinality(), 3);
        assert_eq!(p.value(0), ParamValue::Int(4));
        assert_eq!(p.value(1), ParamValue::Int(6));
        assert_eq!(p.value(2), ParamValue::Int(8));
        // Out-of-range indices clamp rather than panic.
        assert_eq!(p.value(99), ParamValue::Int(8));
        // A step that does not divide the span stops at or below `hi`.
        let q = Param::Int {
            lo: 1,
            hi: 6,
            step: 2,
        };
        assert_eq!(q.cardinality(), 3);
        assert_eq!(q.value(2), ParamValue::Int(5));
    }

    #[test]
    fn float_grid_hits_both_bounds() {
        let p = Param::Float {
            lo: -1.0,
            hi: 3.0,
            res: 5,
        };
        assert_eq!(p.cardinality(), 5);
        assert_eq!(p.value(0).as_float().unwrap(), -1.0);
        assert_eq!(p.value(4).as_float().unwrap(), 3.0);
        assert!((p.value(2).as_float().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_float_grid_is_geometric() {
        let p = Param::LogFloat {
            lo: 1e-3,
            hi: 1e-1,
            res: 3,
        };
        let v: Vec<f64> = (0..3).map(|i| p.value(i).as_float().unwrap()).collect();
        assert!((v[0] - 1e-3).abs() < 1e-12);
        assert!((v[1] - 1e-2).abs() < 1e-12, "mid {}", v[1]);
        assert!((v[2] - 1e-1).abs() < 1e-12);
        // Successive ratios are constant — that is what "log" buys.
        assert!(((v[1] / v[0]) - (v[2] / v[1])).abs() < 1e-9);
    }

    #[test]
    fn degenerate_dimensions_stay_usable() {
        // res = 1 or an empty categorical must not produce a zero-size grid,
        // or a sampler would divide by zero.
        assert_eq!(
            Param::Float {
                lo: 2.0,
                hi: 5.0,
                res: 1
            }
            .cardinality(),
            1
        );
        assert_eq!(Param::Categorical(vec![]).cardinality(), 1);
        assert_eq!(
            Param::Int {
                lo: 5,
                hi: 1,
                step: 1
            }
            .cardinality(),
            1
        );
    }

    #[test]
    fn space_decodes_named_values() {
        let s = Space::new()
            .int("n", 4, 8, 2)
            .log_float("lr", 1e-3, 1e-1, 3)
            .categorical("opt", &["gd", "adam"]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.dims(), vec![3, 3, 2]);
        assert_eq!(s.size(), 18);
        let vals = s.decode(&[2, 0, 1]);
        assert_eq!(vals[0], ("n".to_string(), ParamValue::Int(8)));
        assert!((vals[1].1.as_float().unwrap() - 1e-3).abs() < 1e-12);
        assert_eq!(vals[2].1.as_str().unwrap(), "adam");
        // A short combo decodes with index 0, not a panic.
        assert_eq!(s.decode(&[]).len(), 3);
    }

    #[test]
    fn size_saturates_instead_of_overflowing() {
        let mut s = Space::new();
        for i in 0..40 {
            s = s.int(&format!("d{i}"), 0, 1_000_000, 1);
        }
        assert_eq!(s.size(), usize::MAX);
    }
}
