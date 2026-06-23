// SPDX-License-Identifier: Apache-2.0
//! Quantum linear-algebra circuit recipes (block encoding + QSVT).
//!
//! The CIRCUIT-construction half of the gate-model toolkit's linalg stack:
//! it builds aria `Circuit`s, so it lives here in the runtime (which depends
//! on both `aria_core` for the `Circuit`/`CircuitBuilder` IR and `omega_core`
//! for the numeric Pauli decomposition + angle-finding). The pure-numeric
//! half — `pauli_decompose`, classical `solve`, `cheb`/`qsp` angle-finding —
//! ships in `omega_core::{block_encode, solver, chebyshev, qsp}`.
//!
//! - [`lcu`] — Linear-Combination-of-Unitaries block-encoding circuit.
//! - [`qsvt`] — Quantum Singular Value Transformation circuit.
//! - [`block_encode`] — wrap a dense matrix's Pauli decomposition into LCU.
//! - [`solver`] — end-to-end `A x = b` quantum circuit recipe.

pub mod block_encode;
pub mod lcu;
pub mod qsvt;
pub mod solver;

pub use block_encode::{block_encode_dense, DenseBlockEncoding};
pub use lcu::{lcu_block_encoding, lcu_one_norm, LcuTerm};
pub use qsvt::{inversion_angles, qsvt_circuit, sign_function_angles};
pub use solver::quantum_solve_circuit;
