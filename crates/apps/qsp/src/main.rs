// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `qsp` harness. See src/lib.rs.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("qsp", aria_app_qsp::run)
}
