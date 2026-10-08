use std::collections::HashMap;

use crate::circuit::{fallback_symbol_name, CircuitIR, ParamExpr, SymbolId};
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

    /// Bind a flat value vector to `circuit`'s free symbols in ascending
    /// symbol-ID order — the convention shared by the CLI's `--params`, the
    /// WASM host ABI and the verification harnesses.
    ///
    /// Refuses unless `values.len()` equals the number of free symbols.
    /// Before this existed each caller zipped the two lists itself and bound
    /// whatever was missing to 0.0, which turns a short vector into a
    /// plausible wrong expectation value instead of an error.
    pub fn from_flat(circuit: &CircuitIR, values: &[f64]) -> Result<Self> {
        let ids = circuit.sorted_symbol_ids();
        if values.len() != ids.len() {
            return Err(OmegaError::ParameterCount {
                expected: ids.len(),
                got: values.len(),
                unbound: ids[values.len().min(ids.len())..]
                    .iter()
                    .map(|&id| circuit.symbol_name(id))
                    .collect(),
            });
        }
        Ok(ids
            .into_iter()
            .zip(values.iter().copied())
            .collect::<Vec<_>>()
            .into())
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
                        name: fallback_symbol_name(*id),
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
    /// NOT verified in Lean, despite what this said until 2026-09-04: it
    /// claimed `Verification/Adjoint/ChainRule.lean::differentiate_correct`
    /// and a matching `HasDerivAt` step per match arm. No such file exists
    /// here or in the private monorepo. The symbolic derivative is checked
    /// numerically, by the gradient tests that compare adjoint AD against
    /// parameter-shift.
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
    use crate::circuit::CircuitType;

    fn three_symbol_circuit() -> CircuitIR {
        // Inserted out of ID order: binding must go by sorted ID, not by
        // HashMap iteration order.
        let mut c = CircuitIR::new(1, CircuitType::GateBased);
        c.symbols.insert(7, "c".into());
        c.symbols.insert(3, "a".into());
        c.symbols.insert(5, "b".into());
        c
    }

    #[test]
    fn from_flat_binds_in_sorted_id_order() {
        let c = three_symbol_circuit();
        let pb = ParameterBinding::from_flat(&c, &[1.5, 2.5, 3.5]).unwrap();
        assert_eq!(pb.get(3), Some(1.5));
        assert_eq!(pb.get(5), Some(2.5));
        assert_eq!(pb.get(7), Some(3.5));
    }

    #[test]
    fn from_flat_refuses_short_vector_naming_unbound() {
        let c = three_symbol_circuit();
        match ParameterBinding::from_flat(&c, &[1.5]) {
            Err(OmegaError::ParameterCount {
                expected,
                got,
                unbound,
            }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 1);
                assert_eq!(unbound, vec!["b".to_string(), "c".to_string()]);
            }
            other => panic!("expected ParameterCount, got {other:?}"),
        }
        let msg = ParameterBinding::from_flat(&c, &[1.5])
            .unwrap_err()
            .to_string();
        assert!(msg.contains("got 1 value(s)"), "{msg}");
        assert!(msg.contains("3 free parameter(s)"), "{msg}");
        assert!(msg.contains("[b, c]"), "{msg}");
    }

    #[test]
    fn from_flat_refuses_long_vector() {
        let c = three_symbol_circuit();
        match ParameterBinding::from_flat(&c, &[1.0, 2.0, 3.0, 4.0]) {
            Err(OmegaError::ParameterCount {
                expected,
                got,
                unbound,
            }) => {
                assert_eq!((expected, got), (3, 4));
                assert!(unbound.is_empty());
            }
            other => panic!("expected ParameterCount, got {other:?}"),
        }
    }

    #[test]
    fn from_flat_empty_circuit_accepts_only_empty_vector() {
        let c = CircuitIR::new(1, CircuitType::GateBased);
        assert!(ParameterBinding::from_flat(&c, &[]).is_ok());
        assert!(ParameterBinding::from_flat(&c, &[0.0]).is_err());
    }

    #[test]
    fn symbol_name_falls_back_to_sym_id() {
        let c = three_symbol_circuit();
        assert_eq!(c.symbol_name(3), "a");
        assert_eq!(c.symbol_name(99), "sym_99");
        assert_eq!(fallback_symbol_name(99), "sym_99");
    }

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
