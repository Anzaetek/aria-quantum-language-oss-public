// SPDX-License-Identifier: Apache-2.0
//! The GPU backends must agree on their shared magic numbers.
//!
//! Same shape as `mps_selector_agrees_with_backend.rs`, which pins the MPS
//! chi/ceiling/eps trio: constants that exist in more than one crate are
//! asserted equal, so drift goes red instead of silently splitting behaviour
//! by vendor. (`omega-server`'s `UNLIMITED_FLOOR` pair was deduplicated to one
//! spelling instead — an intra-crate copy needs no cross-crate pin.)
//!
//! Equality, not literals: the *value* 28 may legitimately change (it is a
//! 2 GiB-allocation guard), but it must change on every vendor at once.

#[test]
fn statevector_gpu_backends_share_max_qubits() {
    use omega_backend_statevector_cuda::CudaStatevectorBackend as Cuda;
    use omega_backend_statevector_metal::MetalStatevectorBackend as Metal;
    use omega_backend_statevector_opencl::OpenClStatevectorBackend as OpenCl;

    assert_eq!(
        Metal::MAX_QUBITS,
        Cuda::MAX_QUBITS,
        "the 2 GiB statevector allocation guard must refuse at the same size on \
         Metal and CUDA — a circuit is not 'too big' on one vendor only"
    );
    assert_eq!(
        Metal::MAX_QUBITS,
        OpenCl::MAX_QUBITS,
        "the 2 GiB statevector allocation guard must refuse at the same size on \
         Metal and OpenCL — a circuit is not 'too big' on one vendor only"
    );
}

#[test]
fn pauliprop_accelerators_share_min_terms_threshold() {
    assert_eq!(
        omega_backend_pauliprop_metal::DEFAULT_MIN_TERMS,
        omega_backend_pauliprop_cuda::DEFAULT_MIN_TERMS,
        "the small-sum CPU/GPU crossover must be the same on both accelerators — \
         `PAULIPROP_GPU_MIN` unset must mean one thing"
    );
}
