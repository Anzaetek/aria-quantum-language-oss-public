// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `bernstein_vazirani` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("bernstein_vazirani", aria_app_bernstein_vazirani::run)
}
