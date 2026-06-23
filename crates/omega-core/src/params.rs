use std::collections::HashMap;

use crate::circuit::{ParamExpr, SymbolId};
use crate::error::{OmegaError, Result};

/// Maps symbol IDs to concrete f64 values for circuit execution.
#[derive(Clone, Debug, Default)]
pub struct ParameterBinding {
    values: HashMap<SymbolId, f64>,
}

impl ParameterBinding {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind(&mut self, symbol: SymbolId, value: f64) {
        self.values.insert(symbol, value);
    }

    pub fn get(&self, symbol: SymbolId) -> Option<f64> {
        self.values.get(&symbol).copied()
    }

    /// Recursively resolve a parameter expression to a concrete value.
    pub fn resolve(&self, expr: &ParamExpr) -> Result<f64> {
        match expr {
            ParamExpr::Concrete(v) => Ok(*v),
            ParamExpr::Symbol(id) => {
                self.values
                    .get(id)
                    .copied()
                    .ok_or_else(|| OmegaError::UnboundSymbol {
                        id: *id,
                        name: format!("sym_{}", id),
                    })
            }
            ParamExpr::Negate(inner) => Ok(-self.resolve(inner)?),
            ParamExpr::Add(a, b) => Ok(self.resolve(a)? + self.resolve(b)?),
            ParamExpr::Mul(a, b) => Ok(self.resolve(a)? * self.resolve(b)?),
        }
    }

    /// Resolve the derivative d(expr)/d(symbol) at the current bindings.
    pub fn resolve_derivative(&self, expr: &ParamExpr, symbol: SymbolId) -> Result<f64> {
        self.resolve(&expr.differentiate(symbol))
    }
}

impl ParamExpr {
    /// Symbolic derivative of this expression with respect to the given symbol.
    /// Returns a new ParamExpr tree representing d(self)/d(symbol).
    ///
    /// Verified against the analytic derivative in
    /// `verification/Verification/Adjoint/ChainRule.lean` —
    /// theorem `differentiate_correct`. Each match arm below has
    /// the corresponding `HasDerivAt` step in the Lean induction.
    pub fn differentiate(&self, symbol: SymbolId) -> ParamExpr {
        match self {
            ParamExpr::Concrete(_) => ParamExpr::Concrete(0.0),
            ParamExpr::Symbol(id) => {
                if *id == symbol {
                    ParamExpr::Concrete(1.0)
                } else {
                    ParamExpr::Concrete(0.0)
                }
            }
            ParamExpr::Negate(inner) => ParamExpr::Negate(Box::new(inner.differentiate(symbol))),
            ParamExpr::Add(a, b) => ParamExpr::Add(
                Box::new(a.differentiate(symbol)),
                Box::new(b.differentiate(symbol)),
            ),
            ParamExpr::Mul(a, b) => {
                // Product rule: d(a*b)/dx = da/dx * b + a * db/dx
                ParamExpr::Add(
                    Box::new(ParamExpr::Mul(Box::new(a.differentiate(symbol)), b.clone())),
                    Box::new(ParamExpr::Mul(a.clone(), Box::new(b.differentiate(symbol)))),
                )
            }
        }
    }
}

impl From<Vec<(SymbolId, f64)>> for ParameterBinding {
    fn from(pairs: Vec<(SymbolId, f64)>) -> Self {
        let mut pb = Self::new();
        for (id, val) in pairs {
            pb.bind(id, val);
        }
        pb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differentiate_concrete() {
        let expr = ParamExpr::Concrete(2.5);
        let d = expr.differentiate(0);
        let pb = ParameterBinding::new();
        assert_eq!(pb.resolve(&d).unwrap(), 0.0);
    }

    #[test]
    fn test_differentiate_symbol_match() {
        let expr = ParamExpr::Symbol(0);
        let d = expr.differentiate(0);
        let pb = ParameterBinding::new();
        assert_eq!(pb.resolve(&d).unwrap(), 1.0);
    }

    #[test]
    fn test_differentiate_symbol_no_match() {
        let expr = ParamExpr::Symbol(1);
        let d = expr.differentiate(0);
        let pb = ParameterBinding::new();
        assert_eq!(pb.resolve(&d).unwrap(), 0.0);
    }

    #[test]
    fn test_differentiate_negate() {
        // d(-x)/dx = -1
        let expr = ParamExpr::Negate(Box::new(ParamExpr::Symbol(0)));
        let d = expr.differentiate(0);
        let pb = ParameterBinding::new();
        assert_eq!(pb.resolve(&d).unwrap(), -1.0);
    }

    #[test]
    fn test_differentiate_add() {
        // d(x + 3)/dx = 1
        let expr = ParamExpr::Add(
            Box::new(ParamExpr::Symbol(0)),
            Box::new(ParamExpr::Concrete(3.0)),
        );
        let d = expr.differentiate(0);
        let pb = ParameterBinding::new();
        assert_eq!(pb.resolve(&d).unwrap(), 1.0);
    }

    #[test]
    fn test_differentiate_mul_constant_times_symbol() {
        // d(2.5 * x)/dx = 2.5  (QAOA-style expression)
        let expr = ParamExpr::Mul(
            Box::new(ParamExpr::Concrete(2.5)),
            Box::new(ParamExpr::Symbol(0)),
        );
        let mut pb = ParameterBinding::new();
        pb.bind(0, 1.0);
        assert!((pb.resolve_derivative(&expr, 0).unwrap() - 2.5).abs() < 1e-15);
    }

    #[test]
    fn test_differentiate_product_rule() {
        // d(x * y)/dx = y  where y is symbol 1
        let expr = ParamExpr::Mul(
            Box::new(ParamExpr::Symbol(0)),
            Box::new(ParamExpr::Symbol(1)),
        );
        let mut pb = ParameterBinding::new();
        pb.bind(0, 3.0);
        pb.bind(1, 7.0);
        // d/dx(x*y) = 1*y + x*0 = y = 7.0
        assert!((pb.resolve_derivative(&expr, 0).unwrap() - 7.0).abs() < 1e-15);
        // d/dy(x*y) = 0*y + x*1 = x = 3.0
        assert!((pb.resolve_derivative(&expr, 1).unwrap() - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_differentiate_qaoa_compound() {
        // QAOA uses: Mul(Concrete(2.0 * J), Symbol(gamma)) where J = 0.5
        // d/d(gamma) = 2.0 * J = 1.0
        let j = 0.5;
        let expr = ParamExpr::Mul(
            Box::new(ParamExpr::Concrete(2.0 * j)),
            Box::new(ParamExpr::Symbol(0)),
        );
        let mut pb = ParameterBinding::new();
        pb.bind(0, 0.3); // gamma value doesn't matter for derivative
        assert!((pb.resolve_derivative(&expr, 0).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_differentiate_nested() {
        // d(2 * (x + 3))/dx = 2
        let expr = ParamExpr::Mul(
            Box::new(ParamExpr::Concrete(2.0)),
            Box::new(ParamExpr::Add(
                Box::new(ParamExpr::Symbol(0)),
                Box::new(ParamExpr::Concrete(3.0)),
            )),
        );
        let mut pb = ParameterBinding::new();
        pb.bind(0, 5.0);
        assert!((pb.resolve_derivative(&expr, 0).unwrap() - 2.0).abs() < 1e-15);
    }
}
