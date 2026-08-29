// SPDX-License-Identifier: Apache-2.0
//! **Which backends can cross a thread boundary.**
//!
//! A compile-time measurement, not a runtime test: if a type is not
//! `Send + Sync`, this file does not build. It exists because a whole piece of
//! work depends on the answer and the answer is not readable from the source.
//!
//! Lives in `omega-cli` because that is the crate that already depends on every
//! backend — `omega-core` defines the trait, so the backends depend on *it* and
//! it cannot see them.
//!
//! # Why it matters
//!
//! `aria-py` holds the GIL for the entire duration of every simulation —
//! `allow_threads` appears nowhere in `bindings/` or `crates/` — so two Python
//! threads calling `expectation` serialise completely. Releasing the GIL needs
//! the closure passed to `Python::allow_threads` to be `Send`, which needs the
//! `&dyn Backend` it captures to be `Sync`.
//!
//! The GPU backends each hold an `imp::DeviceHandle` (Metal and OpenCL also an
//! `Arc<imp::BufferPool>`), and **none of those crates declares
//! `unsafe impl Send`/`Sync`** for it — the only such impls in the tree are for
//! an internal CUDA callback context. So whether they qualify depends on their
//! underlying FFI types' auto-traits, which reading cannot settle.
//!
//! Narrowing the bound to the `aria-py` boundary does not avoid the question:
//! `make_backend` there constructs `gpu`, `gpu:cuda`, `gpu:metal` and
//! `gpu:opencl`, so every GPU backend is implicated either way.
//!
//! # Coverage is by cargo feature, and the gaps are NAMED
//!
//! Each GPU backend is asserted only when its feature is on, so this file
//! compiles everywhere while telling the truth about what it checked. A
//! feature-gated assertion that silently vanishes is the green-when-absent trap
//! this repository has found repeatedly, so [`report_what_was_measured`] prints
//! the coverage and the untested set is visible in the output rather than
//! inferred from an absence.
//!
//! **CUDA cannot be compiled on a machine without the toolkit**, so on such a
//! machine this file does not measure it. Landing a `Send + Sync` bound that
//! CUDA might fail, from a machine that cannot compile CUDA, is exactly the
//! hazard `PLATFORM-OWNERSHIP.md` exists to prevent.

/// The whole measurement: instantiating this is the assertion.
fn assert_send_sync<T: Send + Sync>() {}

/// CPU backends — always built, so always measured.
#[test]
fn cpu_backends_are_send_sync() {
    assert_send_sync::<omega_backend_statevector::StatevectorBackend>();
    assert_send_sync::<omega_backend_statevector::NoisyStatevectorBackend>();
    assert_send_sync::<omega_backend_mps::MpsBackend>();
    assert_send_sync::<omega_backend_mps::NoisyMpsBackend>();
    assert_send_sync::<omega_backend_pauli::PauliBackend>();
    assert_send_sync::<omega_backend_pauliprop::PauliPropBackend>();
    assert_send_sync::<omega_backend_photonics::PhotonicsBackend>();
}

/// A boxed trait object is what `aria-py` actually stores, and it is a stricter
/// requirement than the concrete types qualifying: the bound has to be spelled
/// on the trait object for it to hold.
#[test]
fn a_boxed_backend_can_be_declared_send_sync() {
    assert_send_sync::<Box<dyn omega_core::executor::Backend + Send + Sync>>();
    // A shared reference must be `Send` — that is what lets it cross into an
    // `allow_threads` closure. `&T: Send` requires `T: Sync`.
    assert_send_sync::<&(dyn omega_core::executor::Backend + Send + Sync)>();
}

#[cfg(feature = "metal")]
#[test]
fn metal_backend_is_send_sync() {
    assert_send_sync::<omega_backend_statevector_metal::MetalStatevectorBackend>();
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_backend_is_send_sync() {
    assert_send_sync::<omega_backend_statevector_opencl::OpenClStatevectorBackend>();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_backend_is_send_sync() {
    assert_send_sync::<omega_backend_statevector_cuda::CudaStatevectorBackend>();
}

/// Print what this run actually measured, and name what it did not.
///
/// Without this, a machine with no GPU features runs three tests, passes, and
/// says nothing about the four backends it never looked at — which reads as
/// coverage. The point of the programme this measurement serves is to know
/// whether a `Send + Sync` bound is safe *everywhere*, and "safe on the ones we
/// compiled" is a different claim.
#[test]
fn report_what_was_measured() {
    let mut measured = vec!["cpu (7 backends)", "Box<dyn Backend + Send + Sync>"];
    let mut unmeasured = Vec::new();

    if cfg!(feature = "metal") {
        measured.push("metal");
    } else {
        unmeasured.push("metal (build with --features metal)");
    }
    if cfg!(feature = "opencl") {
        measured.push("opencl");
    } else {
        unmeasured.push("opencl (build with --features opencl)");
    }
    if cfg!(feature = "cuda") {
        measured.push("cuda");
    } else {
        unmeasured.push("cuda (needs the CUDA toolkit — run on the GPU host)");
    }

    eprintln!("Send + Sync measured for: {}", measured.join(", "));
    if unmeasured.is_empty() {
        eprintln!("Send + Sync coverage is COMPLETE on this host.");
    } else {
        eprintln!(
            "NOT measured on this host: {}\n\
             A `Send + Sync` bound cannot be declared safe for these from here.",
            unmeasured.join(", ")
        );
    }
}
