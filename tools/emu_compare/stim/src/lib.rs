// SPDX-License-Identifier: Apache-2.0
//! Stabilizer lane of the emulator comparison.
//!
//! One binary, one Python process for Stim. Not a runner framework: the
//! rows this lane owes are written out in `measure`, and the checks that
//! make a row honest live in [`gate`] so a test can redden them one at a
//! time.

pub mod assemble;
pub mod gate;
pub mod measure;
pub mod stim_client;
pub mod workload;
