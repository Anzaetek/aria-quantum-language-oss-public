// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `qpe` harness. See src/lib.rs.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("qpe", aria_app_qpe::run)
}
