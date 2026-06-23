// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `swap_test` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("swap_test", aria_app_swap_test::run)
}
