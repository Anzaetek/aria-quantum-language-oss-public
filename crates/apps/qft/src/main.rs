// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `qft` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("qft", aria_app_qft::run)
}
