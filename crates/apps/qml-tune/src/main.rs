// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `qml_tune` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("qml_tune", aria_app_qml_tune::run)
}
