// SPDX-License-Identifier: Apache-2.0
//! Fermionic lane of the emulator comparison.
//!
//! One binary, one Python process for ffsim. The ≤8-qubit rows are value
//! gates published as capability rows. LUCJ-16 is the only timed row.
//! Kitaev n64 is majoranaprop's certificate against a refusal.

pub mod assemble;
pub mod ffsim_client;
pub mod fqe;
pub mod gate;
pub mod measure;
pub mod workload;
