// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `qec-grover` harness. See src/lib.rs.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("qec-grover", aria_app_qec_grover::run)
}
