// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `bell` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("bell", aria_app_bell::run)
}
