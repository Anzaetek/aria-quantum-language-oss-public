// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `teleport` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("teleport", aria_app_teleport::run)
}
