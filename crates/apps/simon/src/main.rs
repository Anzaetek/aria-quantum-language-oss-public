// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `simon` harness. See src/lib.rs.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("simon", aria_app_simon::run)
}
