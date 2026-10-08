// SPDX-License-Identifier: Apache-2.0
//! Dense GPU lane of the emulator comparison (PLAN-EMULATOR-COMPARISON §7 E5,
//! akilles only).
//!
//! Ours is `omega-backend-statevector-cuda` (f32) on the library path. The
//! competitor is cuStateVec 1.15 driven directly through cuquantum-python, in
//! the fastest variant that passes the value and state gates; qiskit-aer-gpu
//! 0.15.1 is the sanity floor at the same fusion setting. Every GPU arm is a
//! child process, started after the row's device census and gone before its
//! after-reading, so "the card was ours" is checked rather than asserted.
//!
//! One binary, no runner framework (§8).

pub mod census;
pub mod client;
pub mod measure;
