// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `qml_classifier` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("qml_classifier", aria_app_qml_classifier::run)
}
