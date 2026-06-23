// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `grover3` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("grover3", aria_app_grover3::run)
}
