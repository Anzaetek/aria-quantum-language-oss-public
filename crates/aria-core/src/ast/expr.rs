//! Symbolic parameter expressions.
//!
//! Replaces concrete-only `f64` parameters with an expression tree
//! supporting symbols, arithmetic, and standard math functions.
//! Matches omega-functions' `ParamExpr` and Aria DSL's `Expr`.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::f64::consts::PI;
use std::fmt;

/// A symbolic or concrete parameter expression.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ParamExpr {
    /// Concrete numeric value.
    Concrete(f64),
    /// Named symbolic parameter (e.g., "theta", "phi").
    Symbol(String),
    /// The constant π.
    Pi,
    /// Negation: -expr.
    Neg(Box<ParamExpr>),
    /// Addition: a + b.
    Add(Box<ParamExpr>, Box<ParamExpr>),
    /// Multiplication: a * b.
    Mul(Box<ParamExpr>, Box<ParamExpr>),
    /// Division: a / b.
    Div(Box<ParamExpr>, Box<ParamExpr>),
    /// Function call: sin(x), cos(x), exp(x), sqrt(x).
    FnCall(String, Vec<ParamExpr>),
}

impl ParamExpr {
    pub fn symbol(name: &str) -> Self {
        Self::Symbol(name.to_string())
    }

    pub fn pi() -> Self {
        Self::Pi
    }

    /// Check if this expression is a concrete value (no symbols).
    pub fn is_concrete(&self) -> bool {
        match self {
            Self::Concrete(_) | Self::Pi => true,
            Self::Symbol(_) => false,
            Self::Neg(a) => a.is_concrete(),
            Self::Add(a, b) | Self::Mul(a, b) | Self::Div(a, b) => {
                a.is_concrete() && b.is_concrete()
            }
            Self::FnCall(_, args) => args.iter().all(|a| a.is_concrete()),
        }
    }

    /// Collect all free symbol names.
    pub fn free_symbols(&self) -> HashSet<String> {
        let mut syms = HashSet::new();
        self.collect_symbols(&mut syms);
        syms
    }

    fn collect_symbols(&self, out: &mut HashSet<String>) {
        match self {
            Self::Symbol(s) => {
                out.insert(s.clone());
            }
            Self::Neg(a) => a.collect_symbols(out),
            Self::Add(a, b) | Self::Mul(a, b) | Self::Div(a, b) => {
                a.collect_symbols(out);
                b.collect_symbols(out);
            }
            Self::FnCall(_, args) => {
                for a in args {
                    a.collect_symbols(out);
                }
            }
            Self::Concrete(_) | Self::Pi => {}
        }
    }

    /// Evaluate to a concrete f64, resolving symbols via bindings.
    pub fn eval(&self, bindings: &std::collections::HashMap<String, f64>) -> Result<f64, String> {
        match self {
            Self::Concrete(v) => Ok(*v),
            Self::Pi => Ok(PI),
            Self::Symbol(s) => bindings
                .get(s)
                .copied()
                .ok_or_else(|| format!("unbound symbol: {s}")),
            Self::Neg(a) => Ok(-a.eval(bindings)?),
            Self::Add(a, b) => Ok(a.eval(bindings)? + b.eval(bindings)?),
            Self::Mul(a, b) => Ok(a.eval(bindings)? * b.eval(bindings)?),
            Self::Div(a, b) => {
                let denom = b.eval(bindings)?;
                if denom == 0.0 {
                    return Err("division by zero".into());
                }
                Ok(a.eval(bindings)? / denom)
            }
            Self::FnCall(name, args) => {
                let vals: Result<Vec<f64>, _> = args.iter().map(|a| a.eval(bindings)).collect();
                let vals = vals?;
                match (name.as_str(), vals.as_slice()) {
                    ("sin", [x]) => Ok(x.sin()),
                    ("cos", [x]) => Ok(x.cos()),
                    ("tan", [x]) => Ok(x.tan()),
                    ("exp", [x]) => Ok(x.exp()),
                    ("log", [x]) => Ok(x.ln()),
                    ("sqrt", [x]) => Ok(x.sqrt()),
                    ("abs", [x]) => Ok(x.abs()),
                    ("asin", [x]) => Ok(x.asin()),
                    ("acos", [x]) => Ok(x.acos()),
                    ("atan", [x]) => Ok(x.atan()),
                    _ => Err(format!("unknown function: {name}")),
                }
            }
        }
    }

    /// Try to evaluate without bindings (only works for concrete expressions).
    pub fn try_as_f64(&self) -> Option<f64> {
        self.eval(&std::collections::HashMap::new()).ok()
    }
}

// Backward compatibility: f64 → ParamExpr
impl From<f64> for ParamExpr {
    fn from(v: f64) -> Self {
        Self::Concrete(v)
    }
}

impl From<i32> for ParamExpr {
    fn from(v: i32) -> Self {
        Self::Concrete(v as f64)
    }
}

// Arithmetic operators
impl std::ops::Add for ParamExpr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::Add(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Mul for ParamExpr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Div for ParamExpr {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::Div(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Neg for ParamExpr {
    type Output = Self;
    fn neg(self) -> Self {
        Self::Neg(Box::new(self))
    }
}

impl fmt::Display for ParamExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Concrete(v) => write!(f, "{v}"),
            Self::Symbol(s) => write!(f, "${s}"),
            Self::Pi => write!(f, "pi"),
            Self::Neg(a) => write!(f, "-({a})"),
            Self::Add(a, b) => write!(f, "({a} + {b})"),
            Self::Mul(a, b) => write!(f, "({a} * {b})"),
            Self::Div(a, b) => write!(f, "({a} / {b})"),
            Self::FnCall(name, args) => {
                let args_str: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{name}({})", args_str.join(", "))
            }
        }
    }
}

impl PartialEq for ParamExpr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Concrete(a), Self::Concrete(b)) => (a - b).abs() < 1e-15,
            (Self::Symbol(a), Self::Symbol(b)) => a == b,
            (Self::Pi, Self::Pi) => true,
            _ => false, // structural equality only for simple cases
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_concrete() {
        let e = ParamExpr::from(3.14);
        assert!(e.is_concrete());
        assert_eq!(e.try_as_f64(), Some(3.14));
        assert!(e.free_symbols().is_empty());
    }

    #[test]
    fn test_symbol() {
        let e = ParamExpr::symbol("theta");
        assert!(!e.is_concrete());
        assert!(e.free_symbols().contains("theta"));

        let mut bindings = HashMap::new();
        bindings.insert("theta".to_string(), 1.5);
        assert_eq!(e.eval(&bindings).unwrap(), 1.5);
    }

    #[test]
    fn test_arithmetic() {
        let e = ParamExpr::Pi / ParamExpr::from(4.0);
        assert!(e.is_concrete());
        assert!((e.try_as_f64().unwrap() - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_symbolic_arithmetic() {
        let e = ParamExpr::symbol("a") + ParamExpr::symbol("b");
        assert!(!e.is_concrete());
        let syms = e.free_symbols();
        assert!(syms.contains("a"));
        assert!(syms.contains("b"));

        let mut bindings = HashMap::new();
        bindings.insert("a".to_string(), 2.0);
        bindings.insert("b".to_string(), 3.0);
        assert_eq!(e.eval(&bindings).unwrap(), 5.0);
    }

    #[test]
    fn test_fn_call() {
        let e = ParamExpr::FnCall("sin".into(), vec![ParamExpr::Pi / ParamExpr::from(2.0)]);
        assert!((e.try_as_f64().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_unbound_error() {
        let e = ParamExpr::symbol("x");
        assert!(e.eval(&HashMap::new()).is_err());
    }

    #[test]
    fn test_display() {
        let e = ParamExpr::symbol("theta");
        assert_eq!(format!("{e}"), "$theta");
        let e2 = ParamExpr::Pi / ParamExpr::from(4.0);
        assert!(format!("{e2}").contains("pi"));
    }
}
