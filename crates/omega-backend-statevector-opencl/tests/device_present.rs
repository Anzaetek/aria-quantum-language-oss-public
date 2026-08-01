//! Guard against a vacuously-green OpenCL CI stage.
//!
//! Every other integration test in this crate skips itself when
//! `OpenClStatevectorBackend::new()` can't reach a device:
//!
//! ```ignore
//! let backend = match OpenClStatevectorBackend::new() {
//!     Ok(b) => b,
//!     Err(_) => return,
//! };
//! ```
//!
//! That is the right default — `cargo test --all-features` on a host with
//! no OpenCL ICD should not fail — but it means `cargo test --features
//! opencl` reports "ok" on a box where not one kernel ever ran. `ci.sh`'s
//! `ARIA_OPENCL=1` stage is a *claim* that a device exists, so it sets
//! `ARIA_OPENCL_REQUIRE_DEVICE=1` and this test turns that silent skip into
//! a loud failure.
//!
//! Unset (the default), this test is itself a no-op.

#![cfg(feature = "opencl")]

use omega_backend_statevector_opencl::OpenClStatevectorBackend;

#[test]
fn device_present_when_required() {
    if std::env::var("ARIA_OPENCL_REQUIRE_DEVICE").as_deref() != Ok("1") {
        return;
    }
    if let Err(e) = OpenClStatevectorBackend::new() {
        panic!(
            "ARIA_OPENCL_REQUIRE_DEVICE=1 but no OpenCL device is reachable: {e:?}\n\
             Every other test in this crate would have silently skipped, so the \
             CI stage would have passed without executing a single kernel. Either \
             install/repair the OpenCL ICD or drop ARIA_OPENCL=1."
        );
    }
}
