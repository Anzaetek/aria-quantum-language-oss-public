// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `qaoa_maxcut` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("qaoa_maxcut", aria_app_qaoa_maxcut::run)
}
