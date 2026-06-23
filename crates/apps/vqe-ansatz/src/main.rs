// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `vqe_ansatz` harness. See src/lib.rs for the harness.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("vqe_ansatz", aria_app_vqe_ansatz::run)
}
