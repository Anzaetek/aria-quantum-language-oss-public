// SPDX-License-Identifier: Apache-2.0
//! Dense CPU lane of the emulator comparison (PLAN-EMULATOR-COMPARISON §7 E4).
//!
//! Ours is `omega_backend_statevector::StatevectorBackend` (f64) on the
//! library path. The competitors are qulacs 0.6.14 (f64, matched) and qsim
//! 0.22.1 (f32 — every qsim row is precision-unmatched and annotated, §4.5),
//! each one long-lived Python process per knob setting. The Aer CPU oracle
//! supplies the reference value and the reference final state for the value
//! gate; it is never timed.
//!
//! One binary, no runner framework (§8). The checks that make a row honest
//! live in [`gate`] so a test can redden them one at a time.

pub mod assemble;
pub mod client;
pub mod gate;
pub mod measure;
pub mod workload;
