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
//!
//! ## Using Aria as a library (lower once, bind many)
//!
//! The canonical "train / evaluate an Aria circuit from Rust" flow — parse and
//! lower ONCE, then bind parameters by [`SymbolId`](omega_core::circuit::SymbolId)
//! and drive the public `omega_core::executor::Backend` trait. Full guide:
//! `docs/LIBRARY.md`.
//!
//! ```
//! use aria_core::ast::parse_aria;
//! use aria_runtime::lower::lower;
//! use omega_backend_statevector::StatevectorBackend;
//! use omega_core::executor::{Backend, Observable};
//! use omega_core::gradient::{compute_gradient_for, GradMethod};
//! use omega_core::params::ParameterBinding;
//!
//! // 1. Parse + instantiate + lower ONCE.
//! let src = "circuit M() { qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0] }";
//! let circuit = parse_aria(src).unwrap().instantiate("M", &[]).unwrap();
//! let low = lower(&circuit).unwrap();
//!
//! // 2. Bind parameters by SymbolId (symbol_ids is the name → id map).
//! let t_id = low.symbol_ids["t_0"];
//! let mut binding = ParameterBinding::new();
//! binding.bind(t_id, std::f64::consts::FRAC_PI_3);
//!
//! // 3. Expectation ⟨Z⟩ = cos(t) through the Backend trait.
//! let backend = StatevectorBackend::new();
//! let z = Observable::parse("Z0").unwrap();
//! let exp = backend.expectation(&low.ir, &binding, &z).unwrap();
//! assert!((exp - (std::f64::consts::FRAC_PI_3).cos()).abs() < 1e-12);
//!
//! // 4. Exact gradient ∂⟨Z⟩/∂t via adjoint AD — one call, all symbols.
//! let grads = compute_gradient_for(
//!     &backend, &low.ir, &binding, &z, &GradMethod::Adjoint, None,
//! ).unwrap();
//! let dz_dt = grads.iter().find(|(id, _)| *id == t_id).unwrap().1;
//! assert!((dz_dt + (std::f64::consts::FRAC_PI_3).sin()).abs() < 1e-9);
//! ```
pub mod linalg;
pub mod lower;
pub mod model;
pub mod run;
pub mod train;
pub mod train_supervised;

#[cfg(feature = "remote")]
pub mod remote;

pub use lower::{lower, Lowered};
pub use model::{ModelMetadata, TrainedModel};
pub use omega_backend_mps::MpsRunStats;
pub use run::{
    counts_width, expectation, expectation_noisy, expectation_pauliprop, expectation_with_gradient,
    parse_noise_model, run_counts, run_counts_noisy, statevector, take_last_mps_stats, BackendSel,
    GradMethod, PauliPropTruncation,
};
pub use train::{train_expectation, Optimizer, TrainConfig, TrainResult};
pub use train_supervised::{train_supervised, Loss, SupervisedConfig, SupervisedResult};

#[cfg(feature = "remote")]
pub use remote::{expectation_remote, run_counts_remote, Remote};
