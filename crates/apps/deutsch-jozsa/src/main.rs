// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `deutsch_jozsa` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("deutsch_jozsa", aria_app_deutsch_jozsa::run)
}
