//! Integration coverage for the CUDA → CPU fallback path (#10):
//! - `CudaStatevectorBackend::allocate` surfaces device OOM as
//!   `CudaError::OutOfMemory`, mapped through to
//!   `OmegaError::OutOfMemory` (distinct from generic `Backend(_)`).
//! - `QmlTrainer::fit` catches the OOM mid-run and continues on the
//!   backend-supplied `cpu_fallback()` (default-on, opt out via
//!   `QmlTrainer::cpu_fallback_on_oom(false)`).
//!
//! The OOM is provoked by requesting `num_qubits` close to the
//! per-allocation cap (`CudaStatevectorBackend::MAX_QUBITS`) and
//! relying on cudarc / cuMemAlloc to refuse a buffer of that size on
//! the host's GPU. On hosts where the request happens to fit (e.g.,
//! a 96 GB Blackwell when MAX_QUBITS gates the size below the
//! capacity), the assertion that the allocation surfaces the OOM
//! variant is skipped, but the trainer-level fallback test stays
//! valid via the omega-core unit tests (no real GPU needed there).

#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use omega_backend_statevector_cuda::{CudaError, CudaStatevectorBackend};
use omega_core::error::OmegaError;

#[test]
fn allocate_at_max_qubits_either_succeeds_or_returns_oom() {
    let Ok(backend) = CudaStatevectorBackend::new() else {
        eprintln!("skipping: no CUDA backend");
        return;
    };
    // MAX_QUBITS is the per-allocation cap; pushing right at it is
    // the practical OOM threshold on the smallest target devices.
    // On a 96 GB Blackwell the alloc will succeed, which we accept —
    // the assertion is "the path's failure mode is the OOM variant,
    // not a generic Driver error."
    let n = CudaStatevectorBackend::MAX_QUBITS;
    match backend.allocate(n) {
        Ok(_) => {
            eprintln!("note: alloc at MAX_QUBITS={n} succeeded on this host; OOM path untested");
        }
        Err(CudaError::OutOfMemory { num_qubits, reason }) => {
            assert_eq!(num_qubits, n);
            assert!(!reason.is_empty(), "OOM reason should carry detail");
            // Round-trip through OmegaError: must land on the
            // OutOfMemory variant (not Backend).
            let oe: OmegaError = CudaError::OutOfMemory {
                num_qubits: n,
                reason,
            }
            .into();
            assert!(matches!(oe, OmegaError::OutOfMemory(_)));
        }
        Err(other) => {
            panic!("expected OutOfMemory or success, got {other:?}");
        }
    }
}

#[test]
fn allocate_refused_above_max_qubits_is_not_oom() {
    let Ok(backend) = CudaStatevectorBackend::new() else {
        return;
    };
    // Past MAX_QUBITS the backend short-circuits with
    // `AllocationRefused` — that's a *static* "you asked for too
    // much", not a device OOM, and the trainer shouldn't try to
    // recover from it.
    let n = CudaStatevectorBackend::MAX_QUBITS + 1;
    let err = match backend.allocate(n) {
        Ok(_) => panic!("alloc above MAX_QUBITS must refuse"),
        Err(e) => e,
    };
    assert!(matches!(err, CudaError::AllocationRefused { .. }));
    // The OmegaError path also stays on the generic side (not
    // OutOfMemory), so the trainer falls through to a hard error.
    let oe: OmegaError = err.into();
    assert!(!matches!(oe, OmegaError::OutOfMemory(_)));
}
