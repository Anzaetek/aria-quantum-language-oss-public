// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `noise` harness. See src/lib.rs.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("noise", aria_app_noise::run)
}
