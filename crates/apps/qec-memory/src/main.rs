// SPDX-License-Identifier: Apache-2.0
//! Standalone runner for the `qec-memory` harness. See src/lib.rs.
use std::process::ExitCode;

fn main() -> ExitCode {
    aria_verify_core::run_main("qec-memory", aria_app_qec_memory::run)
}
