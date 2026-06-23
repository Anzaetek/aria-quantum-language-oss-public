// SPDX-License-Identifier: Apache-2.0
//! # aria-core
//!
//! Front end for the **Aria Quantum Language**: a lexer, recursive-descent
//! parser, circuit/observable AST, compile-time parameter expressions, and the
//! backend-agnostic gate-model [`Circuit`](ast::Circuit) IR.
//!
//! The pipeline is:
//!
//! ```text
//! source ──parse_aria──▶ AriaProgram ──instantiate(name, params)──▶ Circuit
//!        Circuit ──bind_params──▶ Circuit ──to_omega_ir──▶ OmegaCircuitIR
//! ```
//!
//! Execution is intentionally *not* part of this crate — a `Circuit` lowers to
//! the omega wire IR and is run by a pluggable backend (pure-Rust statevector
//! by default; libtorch / GPU / remote as optional plugins).
pub mod ast;
pub mod backends;

pub use ast::{
    AriaProgram, Circuit, CircuitTemplate, ObservableTemplate, parse_aria, parse_aria_circuit,
};
