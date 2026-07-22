// SPDX-License-Identifier: Apache-2.0
//! Shared toolkit for the Aria example **application harnesses**.
//!
//! Each shipped example lives in its own crate under `crates/apps/<name>` and
//! is tiny and readable because everything reusable is here:
//!   * [`banner`]  — the "what is computed" + PASS/FAIL reporting,
//!   * [`oracle`]  — pure-Rust classical references (Jacobi SVD/eig, DFT, …),
//!   * [`harness`] — `.aria → lower → run THROUGH the omega WASM runtime`,
//!   * [`util`]    — small decode/format helpers,
//!   * [`socket`]  — drive a running omega-server (feature `remote`).
//!
//! An app crate writes one `pub fn run(Transport) -> Result<Verdict, String>`
//! and a 3-line bin that calls [`run_main`].

pub mod banner;
pub mod data;
pub mod harness;
pub mod oracle;
pub mod sim;
#[cfg(feature = "remote")]
pub mod socket;
pub mod util;

pub use banner::Verdict;
pub use harness::Transport;

// Convenience re-exports so an app crate depends ONLY on aria-verify-core.
pub use num_complex::Complex64;
pub use omega_core::executor::Observable;
pub use omega_wasm_runtime::host::HostState;

use std::process::ExitCode;

/// A single example runner: takes the chosen transport, returns its verdict.
pub type ExampleFn = fn(Transport) -> Result<Verdict, String>;

/// Resolve the transport: honor a forced `Native`, else use wasm if the guest
/// binary is built (falling back to native otherwise).
pub fn resolve(over: Transport, guest: &str) -> Transport {
    if over == Transport::Native {
        Transport::Native
    } else {
        harness::transport_for(guest)
    }
}

/// Standalone entry point for an app crate's bin: parse `--native`, run the
/// single example, and exit 0 iff it PASSED.
pub fn run_main(name: &str, run: ExampleFn) -> ExitCode {
    let native = std::env::args().any(|a| a == "--native");
    let t = if native {
        Transport::Native
    } else {
        Transport::WasmInProcess
    };
    match run(t) {
        Ok(v) if v.ok() => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("\n{name}: ERROR — {e}");
            ExitCode::FAILURE
        }
    }
}
