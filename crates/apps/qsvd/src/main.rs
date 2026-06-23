// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `qsvd` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("qsvd", aria_app_qsvd::run)
}
