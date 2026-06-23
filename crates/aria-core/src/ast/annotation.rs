//! Circuit annotations: constraints, contracts, proof obligations.
//!
//! Annotations attach formal specifications to circuits and gates.
//! They are exported as:
//! - QASM comments (`// @assert unitary`)
//! - Lean 4 theorem stubs
//! - Rocq/Coq theorem stubs
//! - Runtime validation (resource bounds)
//!
//! Inspired by fpga-meta-compiler's inline assert/assume/cover
//! and Qbricks' parametric specifications.

use super::expr::ParamExpr;
use serde::{Deserialize, Serialize};

/// An annotation on a circuit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Annotation {
    /// Constraint: must hold (checked at export/compile time).
    Assert(Property),
    /// Assumption: assumed to hold (precondition on input state).
    Assume(Property),

    /// Contract: precondition.
    Requires(Property),
    /// Contract: postcondition.
    Ensures(Property),

    /// Proof obligation: exported to Lean4/Rocq as a named theorem.
    Prove { name: String, property: Property },

    /// Invariant: for hybrid classical-quantum loops.
    Invariant(Property),

    /// Resource bound: metric ≤ bound.
    ResourceBound { metric: String, bound: ParamExpr },

    /// Free-form comment (passthrough to QASM/Lean4/Rocq).
    Comment(String),
}

/// A formal property of a circuit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Property {
    /// U†U = I.
    Unitary,
    /// U² = I.
    SelfInverse,
    /// U = U†.
    Hermitian,
    /// This circuit is equivalent to another (same unitary).
    Equiv { description: String },
    /// Output state matches expected for given input.
    OutputState { input: String, expected: String },
    /// Probability bound: P(qubit=outcome) op bound.
    Probability {
        qubit: usize,
        outcome: u8,
        op: CmpOp,
        bound: ParamExpr,
    },
    /// Custom property (raw Lean4/Rocq expression).
    Custom(String),
}

/// Comparison operator for probability bounds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CmpOp {
    Ge,
    Le,
    Eq,
    Gt,
    Lt,
}

impl std::fmt::Display for Annotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Assert(p) => write!(f, "@assert {p}"),
            Self::Assume(p) => write!(f, "@assume {p}"),
            Self::Requires(p) => write!(f, "@requires {p}"),
            Self::Ensures(p) => write!(f, "@ensures {p}"),
            Self::Prove { name, property } => write!(f, "@prove \"{name}\" {property}"),
            Self::Invariant(p) => write!(f, "@invariant {p}"),
            Self::ResourceBound { metric, bound } => {
                write!(f, "@resource_bound {metric} <= {bound}")
            }
            Self::Comment(s) => write!(f, "// {s}"),
        }
    }
}

impl std::fmt::Display for Property {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unitary => write!(f, "unitary"),
            Self::SelfInverse => write!(f, "self_inverse"),
            Self::Hermitian => write!(f, "hermitian"),
            Self::Equiv { description } => write!(f, "equiv({description})"),
            Self::OutputState { input, expected } => write!(f, "output({input} → {expected})"),
            Self::Probability {
                qubit,
                outcome,
                op,
                bound,
            } => {
                write!(f, "P(q{qubit}={outcome}) {op} {bound}")
            }
            Self::Custom(s) => write!(f, "custom({s})"),
        }
    }
}

impl std::fmt::Display for CmpOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ge => write!(f, ">="),
            Self::Le => write!(f, "<="),
            Self::Eq => write!(f, "=="),
            Self::Gt => write!(f, ">"),
            Self::Lt => write!(f, "<"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annotation_display() {
        let ann = Annotation::Assert(Property::Unitary);
        assert_eq!(format!("{ann}"), "@assert unitary");

        let ann2 = Annotation::Prove {
            name: "qft_correct".into(),
            property: Property::Equiv {
                description: "dft_matrix".into(),
            },
        };
        assert!(format!("{ann2}").contains("qft_correct"));
    }

    #[test]
    fn test_resource_bound() {
        let ann = Annotation::ResourceBound {
            metric: "t_count".into(),
            bound: ParamExpr::from(15.0),
        };
        assert!(format!("{ann}").contains("t_count"));
    }

    #[test]
    fn test_probability_bound() {
        let prop = Property::Probability {
            qubit: 0,
            outcome: 1,
            op: CmpOp::Ge,
            bound: ParamExpr::from(0.99),
        };
        assert!(format!("{prop}").contains(">="));
    }
}
