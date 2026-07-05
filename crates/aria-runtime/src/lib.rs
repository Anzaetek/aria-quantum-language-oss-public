// SPDX-License-Identifier: Apache-2.0
//! # aria-runtime
//!
//! The execution layer for the Aria Quantum Language. It [`lower`]s an Aria
//! [`Circuit`](aria_core::ast::Circuit) into omega-core's in-process
//! `CircuitIR` and runs it on a pluggable backend that implements
//! `omega_core::executor::Backend`.
//!
//! The default backend is the pure-Rust statevector simulator; MPS, GPU,
//! libtorch (`tch`), and remote backends are added as optional features in
//! later phases and selected via [`BackendSel`].
pub mod linalg;
pub mod lower;
pub mod run;
pub mod train;

#[cfg(feature = "remote")]
pub mod remote;

pub use lower::{lower, Lowered};
pub use run::{
    counts_width, expectation, expectation_pauliprop, run_counts, statevector, BackendSel,
    PauliPropTruncation,
};
pub use train::{train_expectation, TrainConfig, TrainResult};

#[cfg(feature = "remote")]
pub use remote::{run_counts_remote, Remote};
