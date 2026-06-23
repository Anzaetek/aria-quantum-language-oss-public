//! OpenCL (cross-vendor) statevector backend.
//!
//! **Status:** `apply_1q` kernel + state I/O lands here. The public
//! surface mirrors the Metal and CUDA crates so the workspace dispatches
//! through the same [`omega_core::device::DeviceKind`] surface; on a
//! build without `--features opencl` (or a host with no usable ICD)
//! the constructor returns [`OpenClError::Unavailable`] and callers
//! fall back to CPU. The remaining kernel slices (`apply_2q`,
//! `inner_product`, `pauli_expectation`, adjoint sweep) land
//! incrementally on top.
//!
//! ## Why a separate crate from CUDA / Metal
//!
//! OpenCL is the only one of the three backends with a runtime
//! ICD-loader story — the `ocl` crate itself compiles on Linux,
//! macOS, and Windows; what differs across hosts is whether
//! `clGetPlatformIDs` returns anything useful. So the feature gate
//! controls *whether* we link `ocl` (no behaviour without it), but
//! the *runtime* availability check happens inside `OpenClStatevectorBackend::new`
//! — same shape as Metal / CUDA, just with a runtime probe instead
//! of a compile-time `cfg(target_os)` guard.
//!
//! ## State layout (planned)
//!
//! Statevector lives in one OpenCL `Buffer<f32>` — interleaved
//! `(re, im, re, im, …)` — matching the Metal and CUDA backends'
//! memory layout. f32 throughout the device, f64 round-tripped at the
//! host boundary. CPU backend stays f64; cross-checks tolerate ~1e-6
//! round-off (Phase 1 Metal validated the same threshold against the
//! 386-fixture verify-qiskit corpus).

use thiserror::Error;

#[cfg(feature = "opencl")]
mod adjoint;
#[cfg(feature = "opencl")]
mod execute;
#[cfg(feature = "opencl")]
mod imp;
#[cfg(feature = "opencl")]
mod kernels;

#[cfg(feature = "opencl")]
use num_complex::Complex64;

#[cfg(feature = "opencl")]
pub use execute::pauli_masks;

use omega_core::circuit::{CircuitIR, SymbolId};
use omega_core::error::{OmegaError, Result as OmegaResult};
use omega_core::executor::{
    Backend, ExecConfig, ExecResult, ExpectationsAndGradient, GradientObservableFactory, Observable,
};
use omega_core::params::ParameterBinding;

/// Errors specific to the OpenCL backend. Map to
/// [`omega_core::error::OmegaError::Backend`] when surfaced through the
/// `Backend` trait.
#[derive(Debug, Error)]
pub enum OpenClError {
    /// Build was missing `--features opencl`, or the feature is
    /// compiled in but no usable OpenCL platform / device is visible
    /// to the loader. Callers should fall back to the CPU backend.
    #[error("OpenCL backend unavailable: {0}")]
    Unavailable(&'static str),
    /// Requested `num_qubits` exceeds the per-allocation cap or the
    /// resulting buffer would be larger than the conservative ceiling.
    #[error("OpenCL allocation refused: {reason} (num_qubits={num_qubits})")]
    AllocationRefused {
        num_qubits: u32,
        reason: &'static str,
    },
    /// Caller passed a state slice whose length doesn't match
    /// `2^num_qubits`.
    #[error("state length mismatch: expected {expected}, got {got}")]
    StateLengthMismatch { expected: usize, got: usize },
    /// Kernel build failed at runtime. Shouldn't happen for the
    /// kernels we ship; if it does the OpenCL ICD or device is
    /// mis-configured.
    #[error("OpenCL kernel `{kernel}` failed to build: {reason}")]
    KernelCompile {
        kernel: &'static str,
        reason: String,
    },
    /// Caller asked for a qubit index that is outside `[0, num_qubits)`.
    #[error("qubit index {qubit} out of range (num_qubits={num_qubits})")]
    QubitOutOfRange { qubit: u32, num_qubits: u32 },
    /// Two-qubit operation called with the same qubit on both args.
    #[error("two-qubit op needs distinct qubits, got qa = qb = {qubit}")]
    DuplicateQubits { qubit: u32 },
    /// Runtime ICD / driver error surfaced from the underlying `ocl`
    /// crate. Carried as an owned `String` because the upstream
    /// errors are dynamic. Most callers should map this back to
    /// `OmegaError::Backend` and let the operator decide whether to
    /// fall back to CPU.
    #[error("OpenCL runtime error: {0}")]
    Runtime(String),
}

impl From<OpenClError> for OmegaError {
    fn from(e: OpenClError) -> Self {
        OmegaError::Backend(e.to_string())
    }
}

// ---------------------------------------------------------------------
// Public API surfaces
// ---------------------------------------------------------------------

/// Stateless OpenCL device handle.
///
/// Construct once with [`OpenClStatevectorBackend::new`]; allocate
/// per-circuit state via [`Self::allocate`]. Implements
/// [`omega_core::executor::Backend`] so it slots into the same
/// dispatch the CPU backend uses.
///
/// `new` discovers the first usable platform / device, builds the
/// embedded kernel program, and constructs an
/// [`ocl::CommandQueue`]. The handle is cheap to keep around;
/// per-circuit `OpenClState` instances clone it under refcount.
pub struct OpenClStatevectorBackend {
    #[cfg(feature = "opencl")]
    handle: imp::DeviceHandle,
    #[cfg(feature = "opencl")]
    pool: std::sync::Arc<imp::BufferPool>,
    #[cfg(not(feature = "opencl"))]
    _private: (),
}

/// Per-circuit statevector held in an OpenCL `Buffer<f32>`. Created
/// by [`OpenClStatevectorBackend::allocate`]. The inner
/// [`imp::StateBuffer`] drops cleanly via `ocl::Buffer`'s own `Drop`
/// (which calls `clReleaseMemObject`), so no `ManuallyDrop` shim is
/// needed here. Metal's mirror struct uses `ManuallyDrop` because
/// its drop may transfer the buffer back to a `BufferPool`; the
/// OpenCL backend has no pool today.
pub struct OpenClState {
    #[cfg(feature = "opencl")]
    inner: imp::StateBuffer,
    #[cfg(not(feature = "opencl"))]
    _private: (),
}

impl OpenClStatevectorBackend {
    /// Open the system's default OpenCL platform / device and compile
    /// the kernel library. On a build that doesn't include the OpenCL
    /// feature — or on a host with no usable ICD — returns
    /// `Err(OpenClError::Unavailable)` rather than panicking.
    pub fn new() -> Result<Self, OpenClError> {
        #[cfg(feature = "opencl")]
        {
            let handle = imp::DeviceHandle::new()?;
            let pool = std::sync::Arc::new(imp::BufferPool::new());
            Ok(Self { handle, pool })
        }
        #[cfg(not(feature = "opencl"))]
        {
            Err(OpenClError::Unavailable("rebuild with --features opencl"))
        }
    }

    /// Allocate a fresh statevector for `num_qubits` qubits, initialised
    /// to `|0…0⟩`. Refuses allocations above
    /// [`OpenClStatevectorBackend::MAX_QUBITS`].
    pub fn allocate(&self, num_qubits: u32) -> Result<OpenClState, OpenClError> {
        #[cfg(feature = "opencl")]
        {
            let inner = self.handle.allocate(num_qubits)?;
            Ok(OpenClState { inner })
        }
        #[cfg(not(feature = "opencl"))]
        {
            let _ = num_qubits;
            Err(OpenClError::Unavailable("rebuild with --features opencl"))
        }
    }

    /// Static availability check. Mirrors
    /// [`omega_core::device::DeviceKind::is_available`] for the OpenCL
    /// branch. Reports compile-time presence of the feature; does
    /// *not* probe the runtime ICD (use [`Self::new`] for that).
    pub const fn is_available() -> bool {
        cfg!(feature = "opencl")
    }

    /// Hard cap on `num_qubits` for a single allocation. Above this the
    /// statevector would exceed 2 GiB on f32-complex; we'd rather
    /// refuse cleanly than thrash. Mirrors the Metal / CUDA crates.
    pub const MAX_QUBITS: u32 = 28;

    /// Number of `StateBuffer`s currently pooled in the backend's
    /// internal `BufferPool` at the given qubit count. Surfaces the
    /// pool's running cache size — the QML trainer's adjoint hot
    /// path is expected to steady-state at three buffers per qubit
    /// count after the first epoch (|φ⟩ + |ν⟩ + scratch). Tests
    /// pin this contract via deltas across `adjoint_gradient`
    /// invocations.
    #[cfg(feature = "opencl")]
    pub fn pool_size(&self, num_qubits: u32) -> usize {
        self.pool.pooled_count(num_qubits)
    }
}

impl OpenClState {
    /// Number of qubits this state was allocated for.
    pub fn num_qubits(&self) -> u32 {
        #[cfg(feature = "opencl")]
        {
            self.inner.num_qubits
        }
        #[cfg(not(feature = "opencl"))]
        {
            0
        }
    }

    /// Read the statevector back to host-side `Complex64`. Crosses the
    /// f32 → f64 boundary; values match the buffer to within f32
    /// precision.
    #[cfg(feature = "opencl")]
    pub fn read_state(&self) -> Vec<Complex64> {
        self.inner.read_state()
    }

    /// Overwrite the statevector with a host-side `Complex64` slice.
    /// Slice length must be `2^num_qubits`.
    #[cfg(feature = "opencl")]
    pub fn write_state(&mut self, state: &[Complex64]) -> Result<(), OpenClError> {
        self.inner.write_state(state)
    }

    /// Apply a single-qubit unitary `U = [[u00, u01], [u10, u11]]` to
    /// `qubit`. Same byte layout as the Metal / CUDA `apply_1q` paths
    /// so a host-side gate-matrix builder feeds this kernel
    /// unchanged.
    #[cfg(feature = "opencl")]
    pub fn apply_1q(
        &mut self,
        qubit: u32,
        u00: Complex64,
        u01: Complex64,
        u10: Complex64,
        u11: Complex64,
    ) -> Result<(), OpenClError> {
        self.inner.apply_1q(qubit, u00, u01, u10, u11)
    }

    /// Apply a row-major 4x4 unitary on `(qa, qb)`, qa low / qb high
    /// row-bit. Caller passes 32 floats: 16 complex (re, im) entries.
    /// Same convention as the Metal / CUDA backends.
    #[cfg(feature = "opencl")]
    pub fn apply_2q(&mut self, qa: u32, qb: u32, u: &[f32; 32]) -> Result<(), OpenClError> {
        self.inner.apply_2q(qa, qb, u)
    }

    /// Apply a diagonal 1q gate `U = diag(d0, d1)` — Z / S / Sdg / T /
    /// Tdg / Rz / U1 and their derivatives. See
    /// [`crate::imp::StateBuffer::apply_diagonal`].
    #[cfg(feature = "opencl")]
    pub fn apply_diagonal(
        &mut self,
        qubit: u32,
        d0: Complex64,
        d1: Complex64,
    ) -> Result<(), OpenClError> {
        self.inner.apply_diagonal(qubit, d0, d1)
    }

    /// Apply a diagonal 2q gate `U = diag(d00, d01, d10, d11)` — CRz
    /// / CZ and their derivatives. See
    /// [`crate::imp::StateBuffer::apply_diagonal_2q`].
    #[cfg(feature = "opencl")]
    pub fn apply_diagonal_2q(
        &mut self,
        qa: u32,
        qb: u32,
        d00: Complex64,
        d01: Complex64,
        d10: Complex64,
        d11: Complex64,
    ) -> Result<(), OpenClError> {
        self.inner.apply_diagonal_2q(qa, qb, d00, d01, d10, d11)
    }

    /// Fused-diagonal-product dispatch — apply N independent diagonal
    /// 1q gates in a single kernel launch. Each factor is `(qubit, d0,
    /// d1)`. See [`crate::imp::StateBuffer::apply_diagonal_product`]
    /// for kernel detail. Used by the diagonal-fusion walker.
    #[cfg(feature = "opencl")]
    pub fn apply_diagonal_product(
        &mut self,
        factors: &[(u32, Complex64, Complex64)],
    ) -> Result<(), OpenClError> {
        self.inner.apply_diagonal_product(factors)
    }

    /// Compute ⟨self|other⟩ on the GPU. Two-stage work-group
    /// reduction (matches Metal's pattern); the host sums the
    /// per-work-group partials. See
    /// [`crate::imp::StateBuffer::inner_product`] for kernel detail.
    #[cfg(feature = "opencl")]
    pub fn inner_product(&self, other: &OpenClState) -> Result<Complex64, OpenClError> {
        self.inner.inner_product(&other.inner)
    }

    /// Fused Pauli-string expectation ⟨ψ|P|ψ⟩ on the GPU. `x_mask`,
    /// `sign_mask`, and `y_factor` come from
    /// `omega_backend_statevector_opencl::pauli_masks` (re-export of
    /// `execute::pauli_masks`). See
    /// [`crate::imp::StateBuffer::pauli_expectation`] for kernel
    /// detail.
    #[cfg(feature = "opencl")]
    pub fn pauli_expectation(
        &self,
        x_mask: u32,
        sign_mask: u32,
        y_factor: Complex64,
    ) -> Result<Complex64, OpenClError> {
        self.inner.pauli_expectation(x_mask, sign_mask, y_factor)
    }

    /// Device-resident statevector copy: `dst ← self`. Used by the
    /// adjoint AD path to clone `|φ⟩` into a scratch state before
    /// applying a derivative gate. See
    /// [`crate::imp::StateBuffer::copy_into`].
    #[cfg(feature = "opencl")]
    pub fn copy_into(&self, dst: &OpenClState) -> Result<(), OpenClError> {
        self.inner.copy_into(&dst.inner)
    }

    /// Per-state running tally of kernel dispatches against this
    /// buffer. Bumped at every `apply_*` and every shot-sampling
    /// kernel stage. Used by tests to pin fusion / batching contracts
    /// — e.g. an 8-Rz layer should collapse to one
    /// `apply_diagonal_product` dispatch, not 8 separate
    /// `apply_diagonal` dispatches.
    #[cfg(feature = "opencl")]
    pub fn dispatch_count(&self) -> u64 {
        self.inner.dispatch_count()
    }

    /// GPU-resident shot sampling — Philox4×32 + Hillis-Steele scan +
    /// CDF binary-search. Returns the per-outcome counts the same way
    /// `MetalState::sample_shots_gpu` does. See
    /// [`crate::imp::StateBuffer::sample_shots_gpu`] for the pipeline
    /// detail.
    #[cfg(feature = "opencl")]
    pub fn sample_shots_gpu(
        &self,
        shots: u32,
        seed: u64,
    ) -> Result<std::collections::HashMap<u64, u32>, OpenClError> {
        self.inner.sample_shots_gpu(shots, seed)
    }
}

// ---------------------------------------------------------------------
// Backend trait impl
// ---------------------------------------------------------------------

impl Backend for OpenClStatevectorBackend {
    fn name(&self) -> &str {
        "opencl-statevector"
    }

    fn device(&self) -> omega_core::device::DeviceKind {
        omega_core::device::DeviceKind::OpenCl
    }

    /// CPU rescue path for `QmlTrainer::fit` on allocation refusal.
    /// Mirrors the Metal / CUDA pattern. Returns the workspace CPU
    /// `StatevectorBackend`, which the trainer uses for the
    /// remainder of training after emitting one stderr notice.
    fn cpu_fallback(&self) -> Option<Box<dyn omega_core::executor::Backend>> {
        Some(Box::new(
            omega_backend_statevector::StatevectorBackend::new(),
        ))
    }

    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> OmegaResult<ExecResult> {
        #[cfg(feature = "opencl")]
        {
            execute::run(&self.handle, circuit, params, config)
        }
        #[cfg(not(feature = "opencl"))]
        {
            let _ = (circuit, params, config);
            Err(OpenClError::Unavailable("rebuild with --features opencl").into())
        }
    }

    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> OmegaResult<f64> {
        #[cfg(feature = "opencl")]
        {
            execute::expectation(&self.handle, circuit, params, observable)
        }
        #[cfg(not(feature = "opencl"))]
        {
            let _ = (circuit, params, observable);
            Err(OpenClError::Unavailable("rebuild with --features opencl").into())
        }
    }

    fn expectation_multi(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
    ) -> OmegaResult<Vec<f64>> {
        #[cfg(feature = "opencl")]
        {
            execute::expectation_multi(&self.handle, circuit, params, observables)
        }
        #[cfg(not(feature = "opencl"))]
        {
            let _ = (circuit, params, observables);
            Err(OpenClError::Unavailable("rebuild with --features opencl").into())
        }
    }

    fn adjoint_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> OmegaResult<Option<Vec<(SymbolId, f64)>>> {
        #[cfg(feature = "opencl")]
        {
            adjoint::adjoint_gradient(&self.handle, &self.pool, circuit, params, observable)
        }
        #[cfg(not(feature = "opencl"))]
        {
            let _ = (circuit, params, observable);
            Ok(None)
        }
    }

    fn expectation_multi_then_gradient(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _observables: &[Observable],
        _gradient_observable_factory: GradientObservableFactory<'_>,
    ) -> OmegaResult<ExpectationsAndGradient> {
        Err(OpenClError::Unavailable(
            "OpenCL expectation_multi_then_gradient path not yet implemented",
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "opencl"))]
    fn new_returns_unavailable_on_default_build() {
        // Default (no --features opencl): constructor must surface
        // OpenClError::Unavailable cleanly, not panic.
        match OpenClStatevectorBackend::new() {
            Err(OpenClError::Unavailable(_)) => {}
            Ok(_) => panic!("expected Unavailable on default build, got Ok(_)"),
            Err(e) => panic!("expected Unavailable on default build, got {e:?}"),
        }
    }

    #[test]
    fn is_available_matches_feature_flag() {
        // Compile-time mirror of DeviceKind::is_available's OpenCl
        // branch. With --features opencl this is true; without, false.
        assert_eq!(
            OpenClStatevectorBackend::is_available(),
            cfg!(feature = "opencl")
        );
    }

    #[test]
    fn max_qubits_matches_other_gpu_backends() {
        // The cap is shared across the GPU backends; if one moves the
        // others should too. Spotting a divergence here saves a future
        // operator wondering why one backend refuses at n=28 while
        // another silently OOMs.
        assert_eq!(OpenClStatevectorBackend::MAX_QUBITS, 28);
    }
}
