//! Metal (Apple Silicon) statevector backend.
//!
//! **Status:** Phase 1, step 6a — `Backend` trait wired. The crate
//! compiles on every platform but only does real work when built with
//! `--features metal` *and* the target is macOS. Elsewhere the
//! constructor returns [`MetalError::Unavailable`] so callers can fall
//! back to CPU per the [`omega_core::device::DeviceKind`] contract.
//!
//! ## Architecture
//!
//! Two public types:
//! - [`MetalStatevectorBackend`] — stateless device handle. Owns the
//!   `MTLDevice`, command queue, and pre-compiled MSL kernels. Cheap
//!   to keep around; this is what implements
//!   [`omega_core::executor::Backend`].
//! - [`MetalState`] — per-circuit statevector. Holds the
//!   shared-mode `MTLBuffer` plus `num_qubits`, plus its own
//!   refcounted clones of device/queue/kernels so it doesn't borrow
//!   back to the handle. Direct callers (tests, the future QML
//!   trainer) keep one alive across many gate applications;
//!   `Backend::execute` allocates one per call.
//!
//! ## State layout
//!
//! Statevector lives in one [`metal::Buffer`] with
//! [`metal::MTLResourceOptions::StorageModeShared`] — Apple Silicon's
//! unified memory makes shared-mode buffers zero-copy from host. We
//! store amplitudes as **interleaved f32** pairs (`re, im, re, im, …`).
//! GPUs are dramatically faster at f32 than f64 and parameter sweeps
//! train within the resulting precision. The CPU backend stays f64;
//! cross-checks tolerate ~1e-6 round-off.

use thiserror::Error;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod adjoint;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod imp;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod kernels;

#[cfg(all(target_os = "macos", feature = "metal"))]
use std::mem::ManuallyDrop;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::Arc;

#[cfg(all(target_os = "macos", feature = "metal"))]
use num_complex::Complex64;

use omega_core::circuit::CircuitIR;
#[cfg(all(target_os = "macos", feature = "metal"))]
use omega_core::circuit::GateKind;
use omega_core::circuit::SymbolId;
use omega_core::error::{OmegaError, Result as OmegaResult};
#[cfg(all(target_os = "macos", feature = "metal"))]
use omega_core::executor::MidCircuitMode;
use omega_core::executor::{
    Backend, ExecConfig, ExecResult, ExpectationsAndGradient, GradientObservableFactory, Observable,
};
use omega_core::params::ParameterBinding;

/// Errors specific to the Metal backend. Map to
/// [`omega_core::error::OmegaError::Backend`] when surfaced through the
/// `Backend` trait.
#[derive(Debug, Error)]
pub enum MetalError {
    /// Build was missing `--features metal` or the target isn't macOS.
    /// Callers should fall back to the CPU backend.
    #[error("Metal backend unavailable: {0}")]
    Unavailable(&'static str),
    /// Requested `num_qubits` exceeds the per-allocation cap or the
    /// resulting buffer would be larger than the conservative ceiling.
    #[error("Metal allocation refused: {reason} (num_qubits={num_qubits})")]
    AllocationRefused {
        num_qubits: u32,
        reason: &'static str,
    },
    /// Caller passed a state slice whose length doesn't match
    /// `2^num_qubits`.
    #[error("state length mismatch: expected {expected}, got {got}")]
    StateLengthMismatch { expected: usize, got: usize },
    /// The requested operation is not representable on this backend — e.g. an
    /// analytic `Reset` on an entangled qubit, whose true result is a mixed
    /// state that one statevector cannot hold. Mirrors the CPU backend's
    /// `OmegaError::Unsupported` so both refuse the same inputs.
    #[error("{0}")]
    Unsupported(String),
    /// MSL kernel failed to compile or pipeline state creation failed.
    /// This shouldn't happen for the shaders we ship; if it does the
    /// build is broken or the OS Metal toolchain is mis-configured.
    #[error("Metal kernel `{kernel}` failed to compile: {reason}")]
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
}

impl From<MetalError> for OmegaError {
    fn from(e: MetalError) -> Self {
        match e {
            // `Unsupported` means "this backend cannot express that", which is
            // a different thing from "this backend failed" — callers branch on
            // it to fall back. Mapping it to `Backend` (as every variant used
            // to be) made Metal refuse DIFFERENTLY from the CPU for identical
            // input: CPU returned `Unsupported`, Metal `Backend`.
            MetalError::Unsupported(msg) => OmegaError::Unsupported(msg),
            other => OmegaError::Backend(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------
// Public API surfaces
// ---------------------------------------------------------------------

/// Stateless Apple Silicon Metal device handle.
///
/// Construct once with [`MetalStatevectorBackend::new`]; allocate
/// per-circuit state via [`Self::allocate`]. Implements
/// [`omega_core::executor::Backend`] so it slots into the same
/// dispatch the CPU backend uses.
///
/// Internally maintains a [`imp::BufferPool`] that recycles
/// `MTLBuffer`s across calls so the QML trainer's
/// `adjoint_gradient` hot path doesn't pay a fresh GPU allocation
/// per training point. The pool is opt-in via [`Self::lease`];
/// [`Self::allocate`] keeps non-pooled semantics for callers that
/// want owning buffers (tests, ad-hoc state construction).
pub struct MetalStatevectorBackend {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    handle: imp::DeviceHandle,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pool: Arc<imp::BufferPool>,
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    _private: (),
}

/// Per-circuit statevector held in an `MTLBuffer`. Created by
/// [`MetalStatevectorBackend::allocate`] (owning) or
/// [`MetalStatevectorBackend::lease`] (returns to the backend pool
/// on drop).
pub struct MetalState {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) inner: ManuallyDrop<imp::StateBuffer>,
    /// When `Some`, this state was leased from the pool and the
    /// inner buffer goes back to it on drop. When `None`, the inner
    /// buffer drops normally (releasing the `MTLBuffer`).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) pool_return: Option<Arc<imp::BufferPool>>,
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    _private: (),
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Drop for MetalState {
    fn drop(&mut self) {
        // Safety: `inner` is taken exactly once, in this Drop impl.
        // After the take, `self.inner` must not be accessed again,
        // and `self` itself is being dropped so it cannot be.
        let buf = unsafe { ManuallyDrop::take(&mut self.inner) };
        if let Some(pool) = self.pool_return.take() {
            pool.return_buffer(buf);
        }
        // else: `buf` falls out of scope and the `MTLBuffer` releases.
    }
}

impl MetalStatevectorBackend {
    /// Open the system default `MTLDevice` and compile the kernel
    /// library. On a build that doesn't include the Metal toolchain,
    /// returns `Err(MetalError::Unavailable)` rather than panicking.
    pub fn new() -> Result<Self, MetalError> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let handle = imp::DeviceHandle::new()?;
            Ok(Self {
                handle,
                pool: Arc::new(imp::BufferPool::new()),
            })
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            Err(MetalError::Unavailable(
                "rebuild on macOS with --features metal",
            ))
        }
    }

    /// Allocate a fresh statevector for `num_qubits` qubits, initialised
    /// to `|0…0⟩`. Refuses allocations above
    /// [`MetalStatevectorBackend::MAX_QUBITS`]. The returned state is
    /// owning — its `MTLBuffer` is released on drop. Use
    /// [`Self::lease`] when the buffer should be recycled across
    /// repeated calls (e.g. inside the QML training loop).
    pub fn allocate(&self, num_qubits: u32) -> Result<MetalState, MetalError> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let inner = self.handle.allocate(num_qubits)?;
            Ok(MetalState {
                inner: ManuallyDrop::new(inner),
                pool_return: None,
            })
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = num_qubits;
            Err(MetalError::Unavailable(
                "rebuild on macOS with --features metal",
            ))
        }
    }

    /// Lease a statevector from the backend's internal buffer pool,
    /// initialised to `|0…0⟩`. On drop the underlying `MTLBuffer`
    /// returns to the pool keyed by `num_qubits` instead of being
    /// released — the next `lease(num_qubits)` reuses it without
    /// hitting the Metal allocator.
    ///
    /// First-time leases at a given `num_qubits` allocate fresh.
    /// Subsequent calls draw from the pool; a single-threaded QML
    /// training loop converges to one allocation per buffer kind in
    /// the first epoch and pays zero allocations thereafter.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn lease(&self, num_qubits: u32) -> Result<MetalState, MetalError> {
        let inner = self.pool.lease(&self.handle, num_qubits)?;
        Ok(MetalState {
            inner: ManuallyDrop::new(inner),
            pool_return: Some(Arc::clone(&self.pool)),
        })
    }

    /// Stub for non-metal builds — surfaces the same
    /// `MetalError::Unavailable` as `new`. Keeps callers off the
    /// `cfg` branch.
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn lease(&self, num_qubits: u32) -> Result<MetalState, MetalError> {
        let _ = num_qubits;
        Err(MetalError::Unavailable(
            "rebuild on macOS with --features metal",
        ))
    }

    /// Static availability check. Mirrors
    /// [`omega_core::device::DeviceKind::is_available`] for the Metal
    /// branch.
    pub const fn is_available() -> bool {
        cfg!(all(target_os = "macos", feature = "metal"))
    }

    /// Hard cap on `num_qubits` for a single allocation. Above this the
    /// statevector would exceed 2 GiB on f32-complex; we'd rather
    /// refuse cleanly than thrash. Real available-memory probing lands
    /// with the QML trainer wiring (Phase 4).
    pub const MAX_QUBITS: u32 = 28;
}

impl MetalState {
    /// Number of qubits this state was allocated for.
    pub fn num_qubits(&self) -> u32 {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            self.inner.num_qubits
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            0
        }
    }

    /// Read the statevector back to host-side `Complex64`. Crosses the
    /// f32→f64 boundary; values match the buffer to within f32 precision.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn read_state(&self) -> Vec<Complex64> {
        self.inner.read_state()
    }

    /// Process-global count of `MetalState::read_state` calls since
    /// the program started. Used by the no-host-syncs regression
    /// test (`tests/qml_no_host_syncs.rs`): read this before and
    /// after one training epoch; the delta must be zero because
    /// the QML hot path's gradient observable is diagonal-Z, so the
    /// adjoint backward sweep takes the GPU-only
    /// `apply_diagonal_pauli_sum` path rather than the host
    /// fallback.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn read_state_call_count() -> u64 {
        imp::READ_STATE_CALL_COUNT.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Overwrite the statevector with a host-side `Complex64` slice.
    /// Slice length must be `2^num_qubits`. Truncates each amplitude to
    /// f32.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn write_state(&mut self, state: &[Complex64]) -> Result<(), MetalError> {
        self.inner.write_state(state)
    }

    /// Compute ⟨self|other⟩ on the GPU. Both states must have the
    /// same `num_qubits`. Uses a two-stage reduction kernel; only
    /// the per-threadgroup partials cross the host boundary, summed
    /// on CPU. ~1 host sync per call regardless of state size.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn inner_product(&self, other: &MetalState) -> Result<Complex64, MetalError> {
        self.inner.inner_product(&other.inner)
    }

    /// GPU-resident shot sampling — pulls `shots` measurement
    /// outcomes from the device-side statevector without reading the
    /// full state back to host. See
    /// [`crate::imp::StateBuffer::sample_shots_gpu`] for the kernel
    /// pipeline (compute-probs → Hillis-Steele scan → Philox4×32
    /// sample). Returns a per-outcome count map keyed by basis-state
    /// index; total counts == `shots`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn sample_shots_gpu(
        &self,
        shots: u32,
        seed: u64,
    ) -> Result<std::collections::HashMap<u64, u32>, MetalError> {
        self.inner.sample_shots_gpu(shots, seed)
    }

    /// Fused Pauli-string expectation on the GPU: computes
    /// `⟨ψ|P|ψ⟩` for `P = ⊗_q σ^{p_q}_q` in a single dispatch.
    /// Replaces the previous "clone state → apply σ → inner_product"
    /// trio with one kernel; cuts per-term host-syncs from ~3 to 1.
    /// Caller passes the precomputed masks via `pauli_masks`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn pauli_expectation(
        &self,
        x_mask: u32,
        sign_mask: u32,
        y_factor: Complex64,
    ) -> Result<Complex64, MetalError> {
        self.inner.pauli_expectation(x_mask, sign_mask, y_factor)
    }

    /// Apply a diagonal Pauli-sum observable
    /// `O = Σ_k coeffs[k] · Z^{(sign_masks[k])}` to `self`, writing
    /// `O|ψ⟩` into `dst`. `dst` must already be allocated with the
    /// same `num_qubits`. Each `sign_mask` has a 1 bit at every
    /// qubit position the Z acts on (identity term ⇒ mask 0).
    /// Replaces the host-side `apply_observable_host` initialization
    /// in the adjoint loop's ν setup for QML-style observables.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_diagonal_pauli_sum(
        &self,
        dst: &MetalState,
        terms: &[(u32, f32)],
    ) -> Result<(), MetalError> {
        self.inner.apply_diagonal_pauli_sum(&dst.inner, terms)
    }

    /// Apply N independent diagonal 1q gates as a single fused
    /// dispatch — `state[i] *= ∏_k diag_k[bit_qubit_k(i)]`. Saves
    /// N-1 GPU dispatch round-trips when consecutive diagonal gates
    /// can be fused (e.g. an HEA layer's 8 Rz gates on disjoint
    /// qubits collapse to one dispatch). Each factor is
    /// `(qubit, d0, d1)` — diagonal gates commute, so factor order
    /// is unconstrained.
    ///
    /// Empty `factors` is a no-op. Out-of-range `qubit` returns
    /// `MetalError::QubitOutOfRange`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_diagonal_product(
        &self,
        factors: &[(u32, Complex64, Complex64)],
    ) -> Result<(), MetalError> {
        self.inner.apply_diagonal_product(factors)
    }

    /// Overwrite `dst` with this state's amplitudes. Both buffers
    /// must have the same `num_qubits`. Used by the adjoint AD loop
    /// to recycle a single per-op scratch buffer instead of
    /// allocating a fresh one for every parameter — saves a GPU
    /// allocation + free per parameter (~16 per training point on
    /// the standard 16-param HEA).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn copy_into(&self, dst: &MetalState) -> Result<(), MetalError> {
        if self.inner.num_qubits != dst.inner.num_qubits {
            let dim_src = 1usize << self.inner.num_qubits;
            let dim_dst = 1usize << dst.inner.num_qubits;
            return Err(MetalError::StateLengthMismatch {
                expected: dim_src,
                got: dim_dst,
            });
        }
        // Host memcpy — flush pending GPU work on both buffers so the
        // copy doesn't race in-flight kernels.
        self.inner.end_batch_if_open();
        dst.inner.end_batch_if_open();
        let dim = 1usize << self.inner.num_qubits;
        let bytes = dim * 8;
        let src = self.inner.state.contents() as *const u8;
        let dst = dst.inner.state.contents() as *mut u8;
        // Safety: both buffers are shared-mode with `2*dim*4` bytes,
        // matching `num_qubits` checked above. The pointers come
        // from the same device + are non-overlapping (different
        // MTLBuffer handles).
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, bytes);
        }
        Ok(())
    }

    // ---- diagonal / 1q / 2q kernel wrappers --------------------------

    /// Apply a diagonal single-qubit gate `diag(d0, d1)` to `qubit`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_diagonal(
        &self,
        qubit: u32,
        d0: Complex64,
        d1: Complex64,
    ) -> Result<(), MetalError> {
        self.inner.apply_diagonal(qubit, d0, d1)
    }

    /// Pauli-Z on `qubit` — diag(1, -1).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_z(&self, qubit: u32) -> Result<(), MetalError> {
        self.apply_diagonal(qubit, Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0))
    }

    /// S phase gate on `qubit` — diag(1, i).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_s(&self, qubit: u32) -> Result<(), MetalError> {
        self.apply_diagonal(qubit, Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0))
    }

    /// Sdg (S†) on `qubit` — diag(1, -i).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_sdg(&self, qubit: u32) -> Result<(), MetalError> {
        self.apply_diagonal(qubit, Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0))
    }

    /// T phase gate on `qubit` — diag(1, e^{iπ/4}).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_t(&self, qubit: u32) -> Result<(), MetalError> {
        self.apply_diagonal(
            qubit,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
        )
    }

    /// Tdg (T†) on `qubit` — diag(1, e^{-iπ/4}).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_tdg(&self, qubit: u32) -> Result<(), MetalError> {
        self.apply_diagonal(
            qubit,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4),
        )
    }

    /// Rz(θ) on `qubit` — diag(e^{-iθ/2}, e^{iθ/2}).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_rz(&self, qubit: u32, theta: f64) -> Result<(), MetalError> {
        self.apply_diagonal(
            qubit,
            Complex64::from_polar(1.0, -theta / 2.0),
            Complex64::from_polar(1.0, theta / 2.0),
        )
    }

    /// U1(λ) on `qubit` — diag(1, e^{iλ}).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_u1(&self, qubit: u32, lambda: f64) -> Result<(), MetalError> {
        self.apply_diagonal(
            qubit,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, lambda),
        )
    }

    /// `apply_1q` reading from `self`, writing to `dst`. Sister of
    /// `apply_1q` (in-place); used by the adjoint backward sweep's
    /// per-parameter derivative apply to skip the prior
    /// `copy_into(self → dst) + apply_1q(dst, ...)` roundtrip.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_1q_into(
        &self,
        dst: &MetalState,
        qubit: u32,
        u: &[Complex64; 4],
    ) -> Result<(), MetalError> {
        self.inner.apply_1q_into(&dst.inner, qubit, u)
    }

    /// `apply_diagonal` reading from `self`, writing to `dst`. Same
    /// rationale as `apply_1q_into` but for the diagonal-fast-path.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_diagonal_into(
        &self,
        dst: &MetalState,
        qubit: u32,
        d0: Complex64,
        d1: Complex64,
    ) -> Result<(), MetalError> {
        self.inner.apply_diagonal_into(&dst.inner, qubit, d0, d1)
    }

    /// dRz/dθ from `self` into `dst`. Sister of `apply_drz` (in-place).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_drz_into(
        &self,
        dst: &MetalState,
        qubit: u32,
        theta: f64,
    ) -> Result<(), MetalError> {
        let neg_half_i = Complex64::new(0.0, -0.5);
        let pos_half_i = Complex64::new(0.0, 0.5);
        let d0 = neg_half_i * Complex64::from_polar(1.0, -theta / 2.0);
        let d1 = pos_half_i * Complex64::from_polar(1.0, theta / 2.0);
        self.apply_diagonal_into(dst, qubit, d0, d1)
    }

    /// dU1/dλ from `self` into `dst`. Sister of `apply_du1` (in-place).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_du1_into(
        &self,
        dst: &MetalState,
        qubit: u32,
        lambda: f64,
    ) -> Result<(), MetalError> {
        let d0 = Complex64::new(0.0, 0.0);
        let d1 = Complex64::new(0.0, 1.0) * Complex64::from_polar(1.0, lambda);
        self.apply_diagonal_into(dst, qubit, d0, d1)
    }

    /// dRz/dθ on `qubit` —
    /// `(1/2) · diag(-i·e^{-iθ/2}, i·e^{iθ/2})`.
    ///
    /// Dispatches via `apply_diagonal` instead of the generic
    /// `apply_1q` because the derivative is itself diagonal — half
    /// the per-amplitude memory traffic and skip the host-side 4-
    /// element `Gate1Q` build. Matches `gates::drz(theta)` modulo
    /// the off-diagonal zeros that `apply_1q` would multiply by
    /// anyway.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_drz(&self, qubit: u32, theta: f64) -> Result<(), MetalError> {
        let neg_half_i = Complex64::new(0.0, -0.5);
        let pos_half_i = Complex64::new(0.0, 0.5);
        let d0 = neg_half_i * Complex64::from_polar(1.0, -theta / 2.0);
        let d1 = pos_half_i * Complex64::from_polar(1.0, theta / 2.0);
        self.apply_diagonal(qubit, d0, d1)
    }

    /// dU1/dλ on `qubit` — `diag(0, i·e^{iλ})`.
    ///
    /// Same diagonal-fast-path rationale as `apply_drz`. The d0
    /// factor is 0 — that amplitude gets zeroed in place by the
    /// `apply_diagonal` kernel, which matches `gates::du1_dl`'s
    /// shape exactly.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_du1(&self, qubit: u32, lambda: f64) -> Result<(), MetalError> {
        let d0 = Complex64::new(0.0, 0.0);
        let d1 = Complex64::new(0.0, 1.0) * Complex64::from_polar(1.0, lambda);
        self.apply_diagonal(qubit, d0, d1)
    }

    /// Apply a diagonal 2q gate `U = diag(d00, d01, d10, d11)` to
    /// `(qa, qb)`. Index ordering matches `apply_2q`:
    /// row r = bit_qb*2 + bit_qa.
    ///
    /// Half the per-amplitude memory traffic vs the generic
    /// `apply_2q` matvec, no 4x4 sum-of-products. Use for any 2q
    /// gate diagonal in the computational basis (CRz, CZ, dCRz, …).
    /// `qa == qb` returns `MetalError::DuplicateQubits`; out-of-range
    /// qubits return `MetalError::QubitOutOfRange`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_diagonal_2q(
        &self,
        qa: u32,
        qb: u32,
        d00: Complex64,
        d01: Complex64,
        d10: Complex64,
        d11: Complex64,
    ) -> Result<(), MetalError> {
        self.inner.apply_diagonal_2q(qa, qb, d00, d01, d10, d11)
    }

    /// dCRz/dθ on `(qc, qt)` — `diag(0, (-i/2)·e^{-iθ/2}, 0,
    /// (i/2)·e^{iθ/2})` under Metal's qb-high / qa-low convention,
    /// where qa=qc (control) and qb=qt (target). Routes through
    /// `apply_diagonal_2q` instead of the generic `apply_2q` matvec.
    /// Mirrors `apply_crz`'s qubit-ordering convention.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_dcrz(&self, qc: u32, qt: u32, theta: f64) -> Result<(), MetalError> {
        let neg_half_i = Complex64::new(0.0, -0.5);
        let pos_half_i = Complex64::new(0.0, 0.5);
        let phn = neg_half_i * Complex64::from_polar(1.0, -theta / 2.0);
        let php = pos_half_i * Complex64::from_polar(1.0, theta / 2.0);
        let z = Complex64::new(0.0, 0.0);
        // qa=qc, qb=qt → row r = bit_qt*2 + bit_qc:
        //   r=0 (qt=0, qc=0): unchanged → 0 (derivative)
        //   r=1 (qt=0, qc=1): phn factor
        //   r=2 (qt=1, qc=0): unchanged → 0
        //   r=3 (qt=1, qc=1): php factor
        self.apply_diagonal_2q(qc, qt, z, phn, z, php)
    }

    /// Apply a generic 1q unitary `U = [[u00, u01], [u10, u11]]`
    /// (row-major).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_1q(&self, qubit: u32, u: &[Complex64; 4]) -> Result<(), MetalError> {
        self.inner.apply_1q(qubit, u)
    }

    /// Hadamard on `qubit` — H = (1/√2)[[1,1],[1,-1]].
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_h(&self, qubit: u32) -> Result<(), MetalError> {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let u = [
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(-s, 0.0),
        ];
        self.apply_1q(qubit, &u)
    }

    /// Pauli-X on `qubit`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_x(&self, qubit: u32) -> Result<(), MetalError> {
        let u = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        self.apply_1q(qubit, &u)
    }

    /// Pauli-Y on `qubit`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_y(&self, qubit: u32) -> Result<(), MetalError> {
        let u = [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ];
        self.apply_1q(qubit, &u)
    }

    /// Rx(θ) on `qubit`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_rx(&self, qubit: u32, theta: f64) -> Result<(), MetalError> {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let u = [
            Complex64::new(c, 0.0),
            Complex64::new(0.0, -s),
            Complex64::new(0.0, -s),
            Complex64::new(c, 0.0),
        ];
        self.apply_1q(qubit, &u)
    }

    /// Ry(θ) on `qubit`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_ry(&self, qubit: u32, theta: f64) -> Result<(), MetalError> {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let u = [
            Complex64::new(c, 0.0),
            Complex64::new(-s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(c, 0.0),
        ];
        self.apply_1q(qubit, &u)
    }

    /// U3(θ, φ, λ) on `qubit` — IBM/Qiskit convention.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_u3(
        &self,
        qubit: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
    ) -> Result<(), MetalError> {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let u = [
            Complex64::new(c, 0.0),
            -Complex64::from_polar(s, lambda),
            Complex64::from_polar(s, phi),
            Complex64::from_polar(c, phi + lambda),
        ];
        self.apply_1q(qubit, &u)
    }

    /// Apply a generic 2q unitary (4×4 row-major, row index =
    /// `bit_qb*2 + bit_qa`).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_2q(&self, qa: u32, qb: u32, u: &[Complex64; 16]) -> Result<(), MetalError> {
        self.inner.apply_2q(qa, qb, u)
    }

    /// CX with control = `qc`, target = `qt`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_cx(&self, qc: u32, qt: u32) -> Result<(), MetalError> {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        #[rustfmt::skip]
        let u = [
            o, z, z, z,
            z, z, z, o,
            z, z, o, z,
            z, o, z, z,
        ];
        self.apply_2q(qc, qt, &u)
    }

    /// CY with control = `qc`, target = `qt`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_cy(&self, qc: u32, qt: u32) -> Result<(), MetalError> {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let pi = Complex64::new(0.0, 1.0);
        let mi = Complex64::new(0.0, -1.0);
        // qa = qc (low), qb = qt (high). CY: when qc=1, apply Y to qt.
        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩.
        // Inputs: row 0=|qt=0,qc=0⟩, row 1=|qt=0,qc=1⟩,
        //         row 2=|qt=1,qc=0⟩, row 3=|qt=1,qc=1⟩.
        // CY actions:
        //   |qt=0,qc=0⟩ → unchanged                   ⇒ U[0,0]=1
        //   |qt=0,qc=1⟩ → i|qt=1,qc=1⟩                 ⇒ U[3,1]=i
        //   |qt=1,qc=0⟩ → unchanged                   ⇒ U[2,2]=1
        //   |qt=1,qc=1⟩ → -i|qt=0,qc=1⟩                ⇒ U[1,3]=-i
        #[rustfmt::skip]
        let u = [
            o, z, z, z,
            z, z, z, mi,
            z, z, o, z,
            z, pi, z, z,
        ];
        self.apply_2q(qc, qt, &u)
    }

    /// CZ on `(qa, qb)`. Diagonal: |11⟩ picks up -1.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_cz(&self, qa: u32, qb: u32) -> Result<(), MetalError> {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let m = Complex64::new(-1.0, 0.0);
        #[rustfmt::skip]
        let u = [
            o, z, z, z,
            z, o, z, z,
            z, z, o, z,
            z, z, z, m,
        ];
        self.apply_2q(qa, qb, &u)
    }

    /// SWAP on `(qa, qb)`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_swap(&self, qa: u32, qb: u32) -> Result<(), MetalError> {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        #[rustfmt::skip]
        let u = [
            o, z, z, z,
            z, z, o, z,
            z, o, z, z,
            z, z, z, o,
        ];
        self.apply_2q(qa, qb, &u)
    }

    /// RBS(θ) / Givens rotation on `(qa, qb)`: identity on {|00⟩, |11⟩}, a real
    /// 2×2 rotation on span{|01⟩, |10⟩}.
    ///
    /// Metal's `apply_2q` shares CUDA's basis order (identical `apply_cx`
    /// matrices), which swaps indices 1↔2 relative to the CPU `gates::rbs`
    /// builder. RBS is sign-antisymmetric on the off-diagonal block, so the
    /// +sin/−sin land at [1][2]/[2][1] — byte-for-byte the CUDA `apply_rbs`
    /// matrix. The f32 result tracks the CPU f64 amplitudes to ~1e-6.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_rbs(&self, qa: u32, qb: u32, theta: f64) -> Result<(), MetalError> {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let cv = Complex64::new(theta.cos(), 0.0);
        let sv = Complex64::new(theta.sin(), 0.0);
        let nsv = Complex64::new(-theta.sin(), 0.0);
        #[rustfmt::skip]
        let u = [
            o, z,   z,   z,
            z, cv,  sv,  z,
            z, nsv, cv,  z,
            z, z,   z,   o,
        ];
        self.apply_2q(qa, qb, &u)
    }

    /// Reset qubit `q` to |0⟩ — the **channel**, matching the CPU
    /// `apply_reset` (projective measure, then `X` if the outcome was 1).
    ///
    /// This used to fold each |1⟩ amplitude into its paired |0⟩ slot
    /// (`new0 = old0 + old1`) and renormalise, described as "matching the CPU
    /// `apply_reset` exactly". It no longer does: the CPU backend was corrected
    /// (Reset is a channel, not a coherent fold) and Metal kept the superseded
    /// semantics. The fold is wrong even on an *unentangled* qubit — on |−⟩ it
    /// gives `old0 + old1 = 0`, so the `norm_sq > 0` guard silently skipped the
    /// renormalisation and left the register at zero amplitude.
    ///
    /// **Caveat on "the same predicate":** Metal calls the CPU's
    /// `reset_is_deterministic_within` but with a LOOSER tolerance (`1e-4` vs
    /// `1e-9`), so the acceptance sets are NOT equal — Metal accepts a qubit at
    /// purity `1 − 8·10⁻⁶`, which is genuinely entangled, and returns a pure
    /// state for it (dropping a branch of weight up to `5·10⁻⁵`). Sharing the
    /// function prevents the *formula* drifting, not the *policy*. An earlier
    /// comment here claimed the two "cannot drift apart"; that was wrong. The
    /// gap is recorded in `LE2_ASSUMPTION_LEDGER.md` A6.
    ///
    /// It also had **no entanglement guard**. `apply_ops_fused` carries no RNG,
    /// so Metal can only do the deterministic case; on an entangled qubit the
    /// true post-reset state is *mixed* and no single statevector represents
    /// it. The CPU refuses that in analytic mode — Metal silently returned a
    /// pure-state answer. Now both refuse, via the *same* predicate
    /// (`omega_backend_statevector::sim::reset_is_deterministic`), so the two
    /// cannot drift apart.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_reset(&self, q: u32) -> Result<(), MetalError> {
        self.apply_reset_with(q, None)
    }

    /// `P(qubit q reads 0)`, computed ON DEVICE from `⟨Z_q⟩` — no readback.
    ///
    /// `⟨Z_q⟩ = p0 − p1` and `p0 + p1 = 1`, so `p0 = (1 + ⟨Z_q⟩)/2`. Clamped
    /// because the f32 reduction can land a hair outside `[0,1]`. Mirrors
    /// CUDA's `reset_p0`. The trajectory path uses this instead of
    /// `read_state`, which would pull `2^n` amplitudes to the host **per shot**.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn reset_p0(&self, q: u32) -> Result<f64, MetalError> {
        let one = Complex64::new(1.0, 0.0);
        let z_q = self.pauli_expectation(0, 1u32 << q, one)?.re;
        Ok((0.5 * (1.0 + z_q)).clamp(0.0, 1.0))
    }

    /// [`Self::apply_reset`] with an explicit trajectory branch.
    ///
    /// `branch = None` is the ANALYTIC contract: only a state whose reset has a
    /// single pure-state answer is representable, so an entangled qubit is
    /// refused. `branch = Some(outcome_one)` is one **trajectory** — the caller
    /// has already sampled the outcome, so an entangled reset is fine and the
    /// guard does not apply. Mirrors CUDA's `reset_rng: Option<&mut StdRng>`.
    ///
    /// Without this split the guard fired in SHOTS mode too, where the CPU and
    /// MPS backends happily run per-shot trajectories: a Bell + `Reset q0` at
    /// 512 shots returned counts on CPU and an error on Metal, while the error
    /// text advised "run with shots" — which is what the caller was doing.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_reset_with(&self, q: u32, branch: Option<bool>) -> Result<(), MetalError> {
        let n = self.num_qubits() as usize;
        // The purity check needs the amplitudes — but ONLY the analytic path
        // performs it. A trajectory (`branch = Some(..)`) skips the readback
        // entirely, so shots mode never pulls `2^n` amplitudes per shot.
        let host = if branch.is_none() {
            self.read_state()
        } else {
            Vec::new()
        };
        // f32 tolerance: Metal amplitudes come back rounded, so an unentangled
        // qubit reads purity 1 − O(1e-6) and the CPU's f64 threshold (1e-9)
        // would reject it. A maximally entangled qubit reads 0.5, so 1e-4
        // separates the two by five orders of magnitude.
        if branch.is_none()
            && !omega_backend_statevector::sim::reset_is_deterministic_within(
                &host,
                n,
                q as usize,
                1e-4,
            )
        {
            return Err(MetalError::Unsupported(format!(
                "metal statevector: Reset on qubit {q} is ill-defined here — the qubit is \
                 entangled, so the reset leaves the register in a mixed state that one \
                 statevector cannot represent. Run with shots (each shot is an independent \
                 trajectory), or reset only unentangled qubits."
            )));
        }

        // Unentangled case: the two branches agree up to global phase, so the
        // branch choice is free (p0 need NOT be 0 or 1 — `|+>` is accepted with
        // p0 = 0.5; requiring p0 ∈ {0,1} is CUDA's stricter, divergent rule).
        let p0 = match branch {
            // Already sampled by the caller from an on-device `reset_p0`.
            Some(_) => self.reset_p0(q)?,
            None => {
                let mask = 1usize << q;
                host.iter()
                    .enumerate()
                    .filter(|(i, _)| i & mask == 0)
                    .map(|(_, a)| a.norm_sqr())
                    .sum()
            }
        };

        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        // A caller-supplied branch wins; otherwise (deterministic case) either
        // branch gives the same state up to global phase, so take p0 >= 1/2.
        // A zero-weight branch is unreachable — force the other one so the
        // renormalisation stays finite (matches the CPU and CUDA backends).
        let outcome_is_one = match branch {
            Some(b) => {
                if b && p0 >= 1.0 - 1e-12 {
                    false
                } else if !b && p0 <= 1e-12 {
                    true
                } else {
                    b
                }
            }
            None => p0 < 0.5,
        };
        if outcome_is_one {
            self.apply_diagonal(q, zero, one)?; // keep |1>, kill |0>
        } else {
            self.apply_diagonal(q, one, zero)?; // keep |0>, kill |1>
        }
        let norm_sq = self.pauli_expectation(0, 0, one)?.re;
        if norm_sq > 0.0 {
            let inv = Complex64::new(1.0 / norm_sq.sqrt(), 0.0);
            self.apply_diagonal(q, inv, inv)?;
        }
        if outcome_is_one {
            self.apply_x(q)?;
        }
        Ok(())
    }

    /// CRz(θ) with control = `qc`, target = `qt`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_crz(&self, qc: u32, qt: u32, theta: f64) -> Result<(), MetalError> {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let phn = Complex64::from_polar(1.0, -theta / 2.0);
        let php = Complex64::from_polar(1.0, theta / 2.0);
        #[rustfmt::skip]
        let u = [
            o,   z,   z,   z,
            z,   phn, z,   z,
            z,   z,   o,   z,
            z,   z,   z,   php,
        ];
        self.apply_2q(qc, qt, &u)
    }

    /// Toffoli (CCX): control1, control2 → target. Decomposition into
    /// `apply_h / apply_t / apply_tdg / apply_cx` from Nielsen-Chuang
    /// figure 4.9 — exact unitary equivalent of the standard 3-qubit
    /// gate, no extra ancillas. 15 ops total.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_ccx(&self, qc1: u32, qc2: u32, qt: u32) -> Result<(), MetalError> {
        // Lifted directly from
        // omega-backend-statevector::gates::{ccx via apply_ccx_gate}
        // — same decomposition Nielsen-Chuang use, modulo the qubit
        // labelling.
        self.apply_h(qt)?;
        self.apply_cx(qc2, qt)?;
        self.apply_tdg(qt)?;
        self.apply_cx(qc1, qt)?;
        self.apply_t(qt)?;
        self.apply_cx(qc2, qt)?;
        self.apply_tdg(qt)?;
        self.apply_cx(qc1, qt)?;
        self.apply_t(qc2)?;
        self.apply_t(qt)?;
        self.apply_h(qt)?;
        self.apply_cx(qc1, qc2)?;
        self.apply_t(qc1)?;
        self.apply_tdg(qc2)?;
        self.apply_cx(qc1, qc2)?;
        Ok(())
    }

    /// Fredkin gate (CSwap): if control = 1, swap targets t1 and t2.
    /// Decomposed via three CXs and one CCX.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_cswap(&self, qc: u32, qt1: u32, qt2: u32) -> Result<(), MetalError> {
        // CSwap(c, a, b) = CX(b, a); CCX(c, a, b); CX(b, a).
        self.apply_cx(qt2, qt1)?;
        self.apply_ccx(qc, qt1, qt2)?;
        self.apply_cx(qt2, qt1)?;
        Ok(())
    }

    /// CU3(θ, φ, λ) with control = `qc`, target = `qt`.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_cu3(
        &self,
        qc: u32,
        qt: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
    ) -> Result<(), MetalError> {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let cu_00 = Complex64::new(c, 0.0);
        let cu_01 = -Complex64::from_polar(s, lambda);
        let cu_10 = Complex64::from_polar(s, phi);
        let cu_11 = Complex64::from_polar(c, phi + lambda);
        // qa=qc, qb=qt. CU3: qc=0 → identity on qt; qc=1 → U3(θ,φ,λ) on qt.
        // row 0 (|qt=0,qc=0⟩): unchanged                       ⇒ U[0,0]=1
        // row 1 (|qt=0,qc=1⟩): U3 acting on qt — out_qt=0 part:
        //                      from in qt=0: c*(in row 1)
        //                      from in qt=1: -e^{iλ}s*(in row 3) ⇒ U[1,1]=c, U[1,3]=-e^{iλ}s
        // row 2 (|qt=1,qc=0⟩): unchanged                       ⇒ U[2,2]=1
        // row 3 (|qt=1,qc=1⟩): out_qt=1 part:
        //                      from in qt=0: e^{iφ}s*(in row 1)
        //                      from in qt=1: e^{i(φ+λ)}c*(in row 3) ⇒ U[3,1]=e^{iφ}s, U[3,3]=e^{i(φ+λ)}c
        #[rustfmt::skip]
        let u = [
            o,    z,     z,    z,
            z,    cu_00, z,    cu_01,
            z,    z,     o,    z,
            z,    cu_10, z,    cu_11,
        ];
        self.apply_2q(qc, qt, &u)
    }
}

// ---------------------------------------------------------------------
// Backend trait — wires the metal handle into the workspace dispatch
// ---------------------------------------------------------------------

impl Backend for MetalStatevectorBackend {
    fn name(&self) -> &str {
        "metal-statevector"
    }

    fn device(&self) -> omega_core::device::DeviceKind {
        omega_core::device::DeviceKind::Metal
    }

    /// CPU rescue path for `QmlTrainer::fit` when the Metal allocator
    /// refuses at n ≥ 22 on memory-tight devices. The returned
    /// `StatevectorBackend` is the same CPU sim the workspace ships
    /// — pinned by every CPU integration test — so the rescue is
    /// behaviour-equivalent to "user re-ran with the CPU backend"
    /// from the gradient-correctness side, only slower.
    fn cpu_fallback(&self) -> Option<Box<dyn omega_core::executor::Backend>> {
        Some(Box::new(
            omega_backend_statevector::StatevectorBackend::new(),
        ))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> OmegaResult<ExecResult> {
        let n = circuit.num_qubits;
        let state = self.lease(n)?;

        // Reset is applied in-sequence by `apply_ops_fused` (deterministic
        // fold + renormalise, matching the CPU backend). Mid-circuit
        // measurement with collapse isn't on the GPU yet — refuse so the CLI
        // dispatcher can fall back to the CPU backend deterministically. (CCX
        // and CSwap are decomposed into 1q+2q kernels by `apply_op`.)
        for op in &circuit.ops {
            if let GateKind::Measure = &op.gate {
                if config.mid_circuit_mode == MidCircuitMode::Collapse {
                    return Err(OmegaError::Unsupported(
                        "metal: mid-circuit measurement not yet implemented (Phase 1 step 8)"
                            .into(),
                    ));
                }
            }
        }

        // Apply each gate via the fused walker. Conditions are
        // evaluated against the same initial classical-bits image as
        // the CPU backend; without mid-circuit support those bits
        // stay zero throughout, so any `condition (cbit, expected
        // != 0)` skips its op. Consecutive unconditional diagonal
        // gates (Z / S / Sdg / T / Tdg / Rz / U1) collapse into one
        // `apply_diagonal_product` dispatch — saves N-1 round-trips
        // per N-long fusion run (e.g. an HEA layer's 8 Rz fan).
        let classical_bits = vec![0u8; circuit.num_classical_bits as usize];
        // A circuit with `Reset` is NOT unitary, so "evolve once then sample the
        // final state" — which is all this backend's shot path does — is
        // invalid: the true result is a mixture over trajectories. In shots
        // mode we therefore DELEGATE to the CPU statevector backend, which runs
        // one trajectory per shot.
        //
        // This is a fallback, not a fix, and it is deliberate. I first
        // implemented per-shot GPU trajectories (lease → evolve → sample, with
        // the reset branch drawn from an RNG). It is correct — verified at 16
        // shots against the CPU, same support, no impossible outcomes — but it
        // BLOCKS at 0% CPU from a few hundred shots onward, and draining the
        // batch `apply_ops_fused` leaves open after a Reset did not resolve it.
        // Shipping a hang is worse than the bug it replaces, so the delegation
        // stands until that is root-caused. See LIMITATIONS.md.
        //
        // Before this, the analytic entanglement guard fired in shots mode too:
        // Bell + `Reset q0` at 512 shots returned counts on the CPU and an
        // error on Metal, whose text advised "run with shots" — which is what
        // the caller was already doing.
        if config.shots.is_some()
            && circuit
                .ops
                .iter()
                .any(|op| matches!(op.gate, GateKind::Reset))
        {
            return omega_backend_statevector::StatevectorBackend::new()
                .execute(circuit, params, config);
        }

        apply_ops_fused(&state, &circuit.ops, params, |op| {
            // The walker's `condition_skip` predicate returns `true`
            // to skip the op. Since `classical_bits` is all-zeros
            // (no mid-circuit measurement on Metal), any condition
            // requiring a non-zero value should skip.
            !op.condition_satisfied(&classical_bits)
        }, None)?;

        match config.shots {
            None => {
                // Full statevector requested — the host pull is intrinsic.
                let amps = state.read_state();
                Ok(ExecResult::Statevector(amps))
            }
            Some(shots) => {
                // GPU shot sampling via Philox4×32 + Hillis-Steele CDF
                // scan. The host↔device traffic shrinks from `2·dim·f32`
                // bytes (`read_state`) to `shots·4` bytes (outcomes), so
                // for large-n shot-mode CLI runs the state stays GPU-
                // resident. Seed defaults: a u64 sampled from
                // `rand::make_rng` when `config.seed` is None so two
                // runs with the same unspecified seed still diverge
                // statistically (matches the CPU sampler's contract).
                let seed = config.seed.unwrap_or_else(|| {
                    use rand::rngs::StdRng;
                    use rand::RngExt;
                    let mut rng: StdRng = rand::make_rng();
                    rng.random::<u64>()
                });
                let counts = state.sample_shots_gpu(shots, seed)?;
                Ok(ExecResult::Counts(counts))
            }
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn execute(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _config: &ExecConfig,
    ) -> OmegaResult<ExecResult> {
        Err(OmegaError::Backend(
            "metal backend not available on this build (rebuild on macOS with --features metal)"
                .into(),
        ))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> OmegaResult<f64> {
        // GPU-side expectation: run forward, get |ψ⟩ resident on GPU,
        // then evaluate each Pauli term in *one* fused kernel
        // (`pauli_expectation`) that computes `⟨ψ|P|ψ⟩` directly
        // from the masks (x/y/z bits + (-i)^{|Y|} prefactor) without
        // cloning the state or stepping σ kernels per qubit.
        //
        // Cuts per-term host-syncs from ~3 (state-clone + N σ-applies
        // + inner_product) to 1 (single reduction → partials → host
        // sum). At ~100 Pauli terms / 20q that's ~100 syncs total
        // instead of ~300, plus we skip the per-term state clone.
        let n = circuit.num_qubits;
        let psi = self.lease(n)?;
        // Reset is applied in-sequence by `apply_ops_fused` (deterministic).
        // Forward sweep with diagonal-gate fusion (consecutive Z / S /
        // Sdg / T / Tdg / Rz / U1 collapse into one
        // `apply_diagonal_product` dispatch). Measure ops are silently
        // skipped — they're no-ops in the expectation path.
        apply_ops_fused(&psi, &circuit.ops, params, |op| {
            matches!(&op.gate, GateKind::Measure)
        }, None)?;

        let mut total = 0.0_f64;
        for (coeff, pauli_string) in &observable.terms {
            let (x_mask, sign_mask, y_factor) = pauli_masks(pauli_string);
            let ip = psi.pauli_expectation(x_mask, sign_mask, y_factor)?;
            // ⟨ψ|P|ψ⟩ for Hermitian P is real; the imag part is
            // round-off (~ 1e-7 at f32).
            total += coeff * ip.re;
        }
        Ok(total)
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn expectation(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _observable: &Observable,
    ) -> OmegaResult<f64> {
        Err(OmegaError::Backend(
            "metal backend not available on this build (rebuild on macOS with --features metal)"
                .into(),
        ))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn expectation_multi(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
    ) -> OmegaResult<Vec<f64>> {
        // Single forward sweep, then evaluate every observable against
        // the resident on-device |ψ⟩ via the same fused
        // `pauli_expectation` reduction the per-observable path uses.
        // Saves N-1 forward sweeps when the trainer asks for ⟨Z_q⟩ on
        // each of N measurement qubits per training point.
        if observables.is_empty() {
            return Ok(Vec::new());
        }

        let n = circuit.num_qubits;
        let psi = self.lease(n)?;
        // Reset is applied in-sequence by `apply_ops_fused` (deterministic).
        apply_ops_fused(&psi, &circuit.ops, params, |op| {
            matches!(&op.gate, GateKind::Measure)
        }, None)?;

        let mut out = Vec::with_capacity(observables.len());
        for obs in observables {
            let mut total = 0.0_f64;
            for (coeff, pauli_string) in &obs.terms {
                let (x_mask, sign_mask, y_factor) = pauli_masks(pauli_string);
                let ip = psi.pauli_expectation(x_mask, sign_mask, y_factor)?;
                total += coeff * ip.re;
            }
            out.push(total);
        }
        Ok(out)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn adjoint_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> OmegaResult<Option<Vec<(SymbolId, f64)>>> {
        // Circuits with Reset are non-unitary — no adjoint. Decline (Ok(None))
        // so the runtime falls back to parameter-shift over the (now
        // reset-capable) forward `expectation`. Mirrors the CPU/CUDA backends.
        if circuit
            .ops
            .iter()
            .any(|op| matches!(op.gate, GateKind::Reset))
        {
            return Ok(None);
        }
        adjoint::adjoint_gradient(self, circuit, params, observable)
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn adjoint_gradient(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _observable: &Observable,
    ) -> OmegaResult<Option<Vec<(SymbolId, f64)>>> {
        // No GPU build → no adjoint; caller falls back to param-shift.
        Ok(None)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn expectation_multi_then_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
        gradient_observable_factory: GradientObservableFactory<'_>,
    ) -> OmegaResult<ExpectationsAndGradient> {
        // Trainer hot path: compute predictions and gradient with a
        // *single* forward sweep on the GPU. Default trait impl does
        // two sweeps (one in `expectation_multi`, one inside
        // `adjoint_gradient`); fusing them saves ~3 ms per train pt
        // on n=18.
        //
        // If the circuit contains a non-unitary op, fall back to the
        // default path (separate calls) so the parameter-shift
        // fallback in `adjoint_gradient` still triggers correctly.
        if circuit
            .ops
            .iter()
            .any(|op| matches!(&op.gate, GateKind::Reset))
        {
            return Err(OmegaError::Unsupported(
                "metal expectation_multi_then_gradient: Reset not supported".into(),
            ));
        }
        if circuit
            .ops
            .iter()
            .any(|op| matches!(&op.gate, GateKind::Measure))
        {
            // Forward sweep silently skips Measure (matches expectation_multi's
            // behaviour); but adjoint requires fully-unitary ops. Detect
            // this and fall back to the default split path.
            let predictions = self.expectation_multi(circuit, params, observables)?;
            let obs = gradient_observable_factory(&predictions);
            let gradient = self.adjoint_gradient(circuit, params, &obs)?;
            return Ok((predictions, gradient));
        }

        // Forward sweep — the leased phi state goes into both the
        // expectation reductions (read-only) and the adjoint backward
        // sweep (where it's daggered down).
        let n = circuit.num_qubits;
        let phi = self.lease(n)?;
        apply_ops_fused(&phi, &circuit.ops, params, |op| {
            matches!(&op.gate, GateKind::Measure)
        }, None)?;

        // Predictions: pauli_expectation per observable on the
        // resident on-device |ψ⟩. Mirrors expectation_multi.
        let mut predictions = Vec::with_capacity(observables.len());
        for obs in observables {
            let mut total = 0.0_f64;
            for (coeff, pauli_string) in &obs.terms {
                let (x_mask, sign_mask, y_factor) = pauli_masks(pauli_string);
                let ip = phi.pauli_expectation(x_mask, sign_mask, y_factor)?;
                total += coeff * ip.re;
            }
            predictions.push(total);
        }

        // Build the gradient observable from the predictions, then
        // hand off to the adjoint backward sweep with the existing
        // forward state. Skips the second forward sweep entirely.
        let gradient_obs = gradient_observable_factory(&predictions);
        let gradient = adjoint::adjoint_gradient_with_forward_state(
            self,
            circuit,
            params,
            &gradient_obs,
            phi,
        )?;

        Ok((predictions, gradient))
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn expectation_multi_then_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
        gradient_observable_factory: GradientObservableFactory<'_>,
    ) -> OmegaResult<ExpectationsAndGradient> {
        // No GPU build → defer to the default trait impl by calling
        // the constituent methods explicitly (default body inlined
        // since trait dispatch through `&dyn Backend` would be the
        // same code path).
        let predictions = self.expectation_multi(circuit, params, observables)?;
        let obs = gradient_observable_factory(&predictions);
        let gradient = self.adjoint_gradient(circuit, params, &obs)?;
        Ok((predictions, gradient))
    }
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// Map a single `GateOp` to the right kernel call. Mirrors the
/// dispatch in `omega-backend-statevector::sim::apply_gate`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn apply_op(
    state: &MetalState,
    op: &omega_core::circuit::GateOp,
    params: &ParameterBinding,
) -> OmegaResult<()> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;

    let q0 = || op.qubits[0].0;
    let q1 = || op.qubits[1].0;

    let res: Result<(), MetalError> = match &op.gate {
        // No-ops at the kernel level
        GateKind::Id | GateKind::Barrier => Ok(()),
        GateKind::Measure => Ok(()), // skip mode — already filtered above

        // Single-qubit gates
        GateKind::H => state.apply_h(q0()),
        GateKind::X => state.apply_x(q0()),
        GateKind::Y => state.apply_y(q0()),
        GateKind::Z => state.apply_z(q0()),
        GateKind::S => state.apply_s(q0()),
        GateKind::Sdg => state.apply_sdg(q0()),
        GateKind::T => state.apply_t(q0()),
        GateKind::Tdg => state.apply_tdg(q0()),

        GateKind::Rx => state.apply_rx(q0(), resolved[0]),
        GateKind::Ry => state.apply_ry(q0(), resolved[0]),
        GateKind::Rz => state.apply_rz(q0(), resolved[0]),
        GateKind::U1 => state.apply_u1(q0(), resolved[0]),
        // U2(φ, λ) = U3(π/2, φ, λ)
        GateKind::U2 => state.apply_u3(q0(), std::f64::consts::FRAC_PI_2, resolved[0], resolved[1]),
        GateKind::U3 => state.apply_u3(q0(), resolved[0], resolved[1], resolved[2]),

        // Two-qubit gates
        GateKind::CX => state.apply_cx(q0(), q1()),
        GateKind::CY => state.apply_cy(q0(), q1()),
        GateKind::CZ => state.apply_cz(q0(), q1()),
        GateKind::Swap => state.apply_swap(q0(), q1()),
        GateKind::CRz => state.apply_crz(q0(), q1(), resolved[0]),
        GateKind::CU3 => state.apply_cu3(q0(), q1(), resolved[0], resolved[1], resolved[2]),
        GateKind::Rbs => state.apply_rbs(q0(), q1(), resolved[0]),

        // Three-qubit gates — decomposed via existing 1q+2q kernels.
        GateKind::CCX => state.apply_ccx(q0(), q1(), op.qubits[2].0),
        GateKind::CSwap => state.apply_cswap(q0(), q1(), op.qubits[2].0),

        // Reset stays rejected — non-unitary, requires the projection
        // logic that step 8 wires up.
        GateKind::Reset => {
            return Err(OmegaError::Unsupported(format!(
                "metal: {:?} should have been filtered before apply_op",
                op.gate
            )));
        }

        // Photonic / custom — no native Metal kernel; the CPU statevector
        // backend handles these. (RBS now runs natively via `apply_rbs`.)
        GateKind::PhaseShifter | GateKind::BeamSplitterRx | GateKind::Custom(_) => {
            return Err(OmegaError::Unsupported(format!(
                "metal-statevector: gate {:?} is not supported on this backend",
                op.gate
            )));
        }
    };

    res.map_err(OmegaError::from)
}

/// Classify `op` as a fusion-eligible diagonal 1q gate. Returns
/// `Some((qubit, d0, d1))` when the gate is unconditional and
/// diagonal in the computational basis (Z, S/Sdg, T/Tdg, Rz, U1);
/// `None` for anything that needs the full `apply_op` path.
///
/// Conditional gates (`op.condition.is_some()`) are always rejected
/// even when the gate itself is diagonal — fusion can't move across
/// a classical predicate without losing the per-op skip semantics.
/// Id / Barrier / Measure aren't classified as factors either: they
/// are no-ops at the kernel level and emitting an identity factor
/// would just waste a slot in the per-amp shader loop.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn diagonal_factor(
    op: &omega_core::circuit::GateOp,
    params: &ParameterBinding,
) -> OmegaResult<Option<(u32, Complex64, Complex64)>> {
    if op.condition.is_some() {
        return Ok(None);
    }
    let q = op.qubits[0].0;
    let factor = match &op.gate {
        GateKind::Z => (q, Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)),
        GateKind::S => (q, Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)),
        GateKind::Sdg => (q, Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0)),
        GateKind::T => (
            q,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
        ),
        GateKind::Tdg => (
            q,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4),
        ),
        GateKind::Rz => {
            let theta = params.resolve(&op.params[0])?;
            (
                q,
                Complex64::from_polar(1.0, -theta / 2.0),
                Complex64::from_polar(1.0, theta / 2.0),
            )
        }
        GateKind::U1 => {
            let lambda = params.resolve(&op.params[0])?;
            (
                q,
                Complex64::new(1.0, 0.0),
                Complex64::from_polar(1.0, lambda),
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(factor))
}

/// Apply a sequence of `GateOp`s to `state`, batching consecutive
/// fusion-eligible diagonal gates into single `apply_diagonal_product`
/// dispatches. Non-diagonal gates flush the pending factor list and
/// fall through to the per-op `apply_op` path.
///
/// The `condition_skip` predicate decides whether each op should be
/// skipped (mirrors the caller's `if condition fails { continue }`
/// loop). Skipped ops are bypassed without disturbing the pending
/// fusion run.
///
/// Saves N-1 dispatch round-trips per N-long fusion run. The HEA
/// bench's second layer (8 Rz on q4..q11) collapses from 8 dispatches
/// to 1; QAOA layers' Rz fans similarly batch.
///
/// Takes `IntoIterator<Item = &GateOp>` so the same helper drives
/// the forward sweep (`&[GateOp]` from `&Vec<GateOp>`) and the
/// adjoint forward sweep (`Vec<&GateOp>` filtered for unitary ops —
/// caller passes `iter().copied()`).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn apply_ops_fused<'a, I>(
    state: &MetalState,
    ops: I,
    params: &ParameterBinding,
    mut condition_skip: impl FnMut(&omega_core::circuit::GateOp) -> bool,
    mut reset_rng: Option<&mut rand::rngs::StdRng>,
) -> OmegaResult<()>
where
    I: IntoIterator<Item = &'a omega_core::circuit::GateOp>,
{
    let mut pending: Vec<(u32, Complex64, Complex64)> = Vec::new();

    let flush =
        |state: &MetalState, pending: &mut Vec<(u32, Complex64, Complex64)>| -> OmegaResult<()> {
            match pending.len() {
                0 => Ok(()),
                1 => {
                    let (q, d0, d1) = pending[0];
                    state.apply_diagonal(q, d0, d1).map_err(OmegaError::from)?;
                    pending.clear();
                    Ok(())
                }
                _ => {
                    state
                        .apply_diagonal_product(pending)
                        .map_err(OmegaError::from)?;
                    pending.clear();
                    Ok(())
                }
            }
        };

    // Open a command-buffer batch so every kernel encoded inside the
    // walker shares a single `commit + wait_until_completed`. At n=18
    // the HEA forward sweep is ~44 dispatches; without batching that's
    // 44 host↔GPU sync roundtrips. With batching it's one. The
    // matched `end_batch` flushes pending GPU work before returning.
    state.inner.begin_batch();
    let walk_result = (|| -> OmegaResult<()> {
        for op in ops {
            if condition_skip(op) {
                continue;
            }
            // Id and Barrier are kernel-level no-ops and shouldn't break
            // an in-flight fusion run. Skip them without flushing so a
            // `Rz; Id; Rz; Rz` sequence still collapses to one fused
            // dispatch (common after compiler-inserted barriers between
            // optimisation passes).
            if matches!(&op.gate, GateKind::Id | GateKind::Barrier) {
                continue;
            }
            // Reset is non-unitary and full-state: flush fused diagonals, then
            // apply it in sequence. `apply_reset`'s `pauli_expectation` calls
            // `end_batch_if_open` (committing the just-flushed dispatches before
            // it reads the norm), so the open batch is drained correctly;
            // reopen one so subsequent gates keep batching. Kept out of
            // `apply_op` so the adjoint dagger never treats Reset as unitary.
            if matches!(&op.gate, GateKind::Reset) {
                flush(state, &mut pending)?;
                let q = op.qubits[0].0;
                // `Some(rng)` = one trajectory (shots mode): sample the branch,
                // so an entangled reset is fine. `None` = analytic: only a
                // deterministic reset is representable and `apply_reset_with`
                // refuses otherwise.
                let branch = match reset_rng.as_deref_mut() {
                    Some(rng) => {
                        use rand::RngExt;
                        let p0 = state.reset_p0(q).map_err(OmegaError::from)?;
                        Some(rng.random::<f64>() >= p0)
                    }
                    None => None,
                };
                state
                    .apply_reset_with(q, branch)
                    .map_err(OmegaError::from)?;
                state.inner.begin_batch();
                continue;
            }
            match diagonal_factor(op, params)? {
                Some(factor) => pending.push(factor),
                None => {
                    flush(state, &mut pending)?;
                    apply_op(state, op, params)?;
                }
            }
        }
        flush(state, &mut pending)?;
        Ok(())
    })();
    state.inner.end_batch();
    walk_result
}

/// Pauli-string expectation on a host-side statevector. Kept for
/// reference / future fallback use; the production path is the GPU
/// inner_product reduction in `Backend::expectation`.
/// Compute the (x_mask, sign_mask, y_factor) triple consumed by
/// the `pauli_expectation` kernel from a Pauli string.
///
/// * `x_mask` has bit `q` set if qubit `q` carries X or Y (those
///   flip the basis bit). Used per-thread as `j = i XOR x_mask`.
/// * `sign_mask` has bit `q` set if qubit `q` carries Y or Z (each
///   contributes `(-1)^bit_q(i)` to the phase). The kernel uses
///   `(-1)^popcount(i & sign_mask)` per index.
/// * `y_factor = (-i)^{|Y|}` — the global Y-count prefactor folded in
///   inside the kernel before the reduction. It is `(-i)`, not `(+i)`,
///   per Y: the kernel forms the matrix element `P[i, i^x]`, which for
///   a Y qubit is `(-i)·(-1)^bit_i` (`Y|0⟩ = i|1⟩`, `Y|1⟩ = -i|0⟩`),
///   and the `(-1)^bit` half is already carried by `sign_mask`.
///
/// I qubits are no-ops and contribute to none of the three.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn pauli_masks(pauli_string: &[(u32, omega_core::executor::PauliOp)]) -> (u32, u32, Complex64) {
    use omega_core::executor::PauliOp;
    let mut x_mask: u32 = 0;
    let mut sign_mask: u32 = 0;
    let mut y_count: u32 = 0;
    for &(q, ref p) in pauli_string {
        let bit = 1u32 << q;
        match p {
            PauliOp::I => {}
            PauliOp::X => {
                x_mask |= bit;
            }
            PauliOp::Y => {
                x_mask |= bit;
                sign_mask |= bit;
                y_count += 1;
            }
            PauliOp::Z => {
                sign_mask |= bit;
            }
        }
    }
    // Per-Y prefactor is (-i)^|Y|, NOT i^|Y| — the kernel forms
    // conj(ψ[i])·ψ[i^x]·phase, and for a Y qubit P[i,i^x] = (-i)·(-1)^bit_i
    // (Y|0⟩=i|1⟩, Y|1⟩=-i|0⟩). Using i^|Y| silently negates every Pauli string
    // with an ODD number of Y factors (see the CPU `expectation_pauli` note).
    let y_factor = match y_count & 3 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, -1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, 1.0),
    };
    (x_mask, sign_mask, y_factor)
}

#[allow(dead_code)]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn cpu_pauli_expectation(
    state: &[Complex64],
    num_qubits: u32,
    pauli_string: &[(u32, omega_core::executor::PauliOp)],
) -> f64 {
    use omega_core::executor::PauliOp;
    let dim = 1usize << num_qubits;
    // Sum over computational-basis indices. For each amplitude, the
    // Pauli string maps |i⟩ → c_i * |j⟩ where c_i is a phase factor
    // and j has some bits flipped. Acc ⟨ψ|P|ψ⟩ = sum_i conj(ψ_i) * c_i * ψ_j.
    let mut acc = Complex64::new(0.0, 0.0);
    for i in 0..dim {
        let mut j = i;
        let mut phase = Complex64::new(1.0, 0.0);
        for &(q, ref p) in pauli_string {
            let bit = (i >> q) & 1;
            match p {
                PauliOp::I => {}
                PauliOp::X => {
                    j ^= 1usize << q;
                }
                PauliOp::Y => {
                    j ^= 1usize << q;
                    if bit == 0 {
                        phase *= Complex64::new(0.0, 1.0); // i
                    } else {
                        phase *= Complex64::new(0.0, -1.0); // -i
                    }
                }
                PauliOp::Z => {
                    if bit == 1 {
                        phase = -phase;
                    }
                }
            }
        }
        // P|i⟩ = phase·|j⟩, so the contribution to ⟨ψ|P|ψ⟩ is
        // conj(ψ[j])·phase·ψ[i] — the `bit`-keyed phase is a KET-side
        // coefficient and belongs with ψ[i], not ψ[j]. Pairing it as
        // conj(ψ[i])·phase·ψ[j] instead silently negates every string with
        // an odd number of Y factors (Y's two matrix elements have opposite
        // signs; X and Z are symmetric, so they are unaffected). This
        // oracle carried that error and so agreed with the old `i^|Y|`
        // kernel — two sign bugs cancelling. Matches the production CPU
        // `sim::expectation_pauli` and its `tests/pauli_y_expectation.rs`.
        acc += state[j].conj() * phase * state[i];
    }
    // ⟨ψ|P|ψ⟩ for Hermitian P is real; tiny imag is round-off.
    acc.re
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_matches_build_config() {
        let expected = cfg!(all(target_os = "macos", feature = "metal"));
        assert_eq!(MetalStatevectorBackend::is_available(), expected);
    }

    #[test]
    fn constructor_succeeds_iff_available() {
        let res = MetalStatevectorBackend::new();
        if MetalStatevectorBackend::is_available() {
            assert!(res.is_ok(), "metal-feature build must construct OK");
        } else {
            match res {
                Err(MetalError::Unavailable(_)) => {}
                Ok(_) => panic!("backend constructed on non-metal build"),
                Err(e) => panic!("expected Unavailable, got {e:?}"),
            }
        }
    }

    #[test]
    fn allocate_refuses_too_many_qubits() {
        let backend = match MetalStatevectorBackend::new() {
            Ok(b) => b,
            Err(MetalError::Unavailable(_)) => return,
            Err(e) => panic!("unexpected error: {e:?}"),
        };
        let res = backend.allocate(MetalStatevectorBackend::MAX_QUBITS + 1);
        match res {
            Err(MetalError::AllocationRefused { .. }) => {}
            Ok(_) => panic!("expected AllocationRefused"),
            Err(e) => panic!("expected AllocationRefused, got {e:?}"),
        }
    }

    // -- macOS+metal-only behavioural tests -------------------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn new_backend() -> MetalStatevectorBackend {
        MetalStatevectorBackend::new().expect("device")
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn new_state(n: u32) -> MetalState {
        new_backend().allocate(n).expect("alloc")
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn lease_returns_zero_initialised_state_like_allocate() {
        // First lease at a given num_qubits hits the slow path
        // (BufferPool stack is empty → allocate fresh). The state
        // must still be |0…0⟩ — same contract as `allocate`.
        let backend = new_backend();
        for &n in &[2u32, 5, 8] {
            let state = backend.lease(n).expect("lease");
            let v = state.read_state();
            assert_eq!(v.len(), 1usize << n);
            assert!((v[0] - Complex64::new(1.0, 0.0)).norm() < 1e-6);
            for amp in &v[1..] {
                assert!(amp.norm() < 1e-6, "amp = {amp:?}");
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn read_state_call_count_increments_on_each_read() {
        // Sanity-pins the no-host-syncs regression test's mechanism:
        // every `read_state` call must bump the counter. Without
        // this guard, a wiring bug could silently make the
        // regression test trivially pass by never moving the
        // counter at all. The counter is process-global and other
        // tests can run in parallel — so we assert `>= 1` per call
        // rather than strict equality (any concurrent test's
        // `read_state` only inflates the delta).
        let backend = new_backend();
        let state = backend.lease(3).expect("lease");
        let before = MetalState::read_state_call_count();
        let _ = state.read_state();
        let mid = MetalState::read_state_call_count();
        let _ = state.read_state();
        let after = MetalState::read_state_call_count();
        assert!(
            mid >= before + 1,
            "first read must bump counter (before={before}, mid={mid})"
        );
        assert!(
            after >= mid + 1,
            "second read must bump counter (mid={mid}, after={after})"
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn lease_pool_recycles_buffer_across_drop_cycle() {
        // Lease → drop returns the buffer to the pool. A second
        // lease at the same num_qubits pulls from the pool (no
        // fresh MTLBuffer allocation) and the state must be
        // re-initialised to |0…0⟩ even after the previous lessee
        // wrote into it.
        let backend = new_backend();
        let n = 5u32;
        assert_eq!(backend.pool.pooled_count(n), 0);

        // First lease + scribble + drop. After drop the pool stack
        // should hold one buffer.
        {
            let mut state = backend.lease(n).expect("first lease");
            // Pollute the state so the recycle reset is observable.
            let dim = 1usize << n;
            let scribbled: Vec<Complex64> = (0..dim)
                .map(|i| Complex64::new(i as f64 * 0.01, 0.0))
                .collect();
            state.write_state(&scribbled).expect("write");
            // Buffer is in flight — pool stack is still empty.
            assert_eq!(backend.pool.pooled_count(n), 0);
        }
        assert_eq!(backend.pool.pooled_count(n), 1);

        // Second lease pulls from the pool. Reset must zero out the
        // scribble. A fresh allocation here would also return
        // |0…0⟩, so the count check above is what proves recycling.
        let state = backend.lease(n).expect("second lease");
        assert_eq!(backend.pool.pooled_count(n), 0);
        let v = state.read_state();
        assert!((v[0] - Complex64::new(1.0, 0.0)).norm() < 1e-6);
        for amp in &v[1..] {
            assert!(amp.norm() < 1e-6, "leftover scribble: {amp:?}");
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn lease_pool_keys_separate_qubit_counts_independently() {
        // Pool is keyed by num_qubits — a 6q drop cannot satisfy a
        // 4q lease (different byte sizes). Confirm both stacks
        // accumulate independently and a lease at one count doesn't
        // pull from another.
        let backend = new_backend();
        {
            let _a = backend.lease(4).expect("lease 4q");
            let _b = backend.lease(6).expect("lease 6q");
        }
        assert_eq!(backend.pool.pooled_count(4), 1);
        assert_eq!(backend.pool.pooled_count(6), 1);

        // Lease 4q — drains the 4q stack only.
        let _c = backend.lease(4).expect("release 4q");
        assert_eq!(backend.pool.pooled_count(4), 0);
        assert_eq!(backend.pool.pooled_count(6), 1);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn allocate_does_not_return_to_pool_on_drop() {
        // `allocate(n)` is the non-pool entry point — its
        // MetalState carries `pool_return: None`, so the buffer
        // releases on drop instead of pooling. Confirm the pool
        // stays empty across allocate→drop.
        let backend = new_backend();
        {
            let _state = backend.allocate(4).expect("alloc");
        }
        assert_eq!(backend.pool.pooled_count(4), 0);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn fresh_backend_initialised_to_zero_state() {
        for &n in &[2u32, 4, 8] {
            let dim = 1usize << n;
            let state = new_state(n);
            let v = state.read_state();
            assert_eq!(v.len(), dim);
            assert!((v[0] - Complex64::new(1.0, 0.0)).norm() < 1e-6);
            for amp in &v[1..] {
                assert!(amp.norm() < 1e-6, "amp = {amp:?}");
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn write_then_read_roundtrip_within_f32_precision() {
        for &n in &[2u32, 4, 8, 12] {
            let dim = 1usize << n;
            let mut state = new_state(n);
            let want: Vec<Complex64> = (0..dim)
                .map(|i| {
                    let re = ((i as f64) * 0.0007 + 0.0123).sin();
                    let im = ((i as f64) * 0.0011 - 0.0456).cos();
                    Complex64::new(re, im)
                })
                .collect();
            state.write_state(&want).expect("write");
            let got = state.read_state();
            assert_eq!(got.len(), want.len());
            let max = want
                .iter()
                .zip(got.iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0_f64, f64::max);
            assert!(max < 1e-6, "n={n}, max abs diff = {max}");
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn write_state_rejects_wrong_length() {
        let mut state = new_state(3);
        let bad = vec![Complex64::new(0.0, 0.0); 7];
        match state.write_state(&bad) {
            Err(MetalError::StateLengthMismatch { expected, got }) => {
                assert_eq!(expected, 8);
                assert_eq!(got, 7);
            }
            other => panic!("expected StateLengthMismatch, got {other:?}"),
        }
    }

    // ---- diagonal kernel ----------------------------------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn cpu_apply_diagonal(state: &mut [Complex64], qubit: u32, d0: Complex64, d1: Complex64) {
        for (i, amp) in state.iter_mut().enumerate() {
            let bit = (i >> qubit) & 1;
            *amp *= if bit == 0 { d0 } else { d1 };
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn random_state(n: u32, seed: u64) -> Vec<Complex64> {
        let dim = 1usize << n;
        let mut x: u64 = seed.wrapping_mul(0x9E3779B97F4A7C15);
        let mut next_f = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let mut s: Vec<Complex64> = (0..dim)
            .map(|_| Complex64::new(next_f(), next_f()))
            .collect();
        let norm: f64 = s.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for c in &mut s {
            *c /= norm;
        }
        s
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn max_abs_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0_f64, f64::max)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_z_on_plus_state_yields_minus_state() {
        let mut state = new_state(1);
        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        state
            .write_state(&[
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ])
            .expect("write");
        state.apply_z(0).expect("Z");
        let got = state.read_state();
        let expect = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ];
        assert!(max_abs_diff(&got, &expect) < 1e-6);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_matches_cpu_oracle_random() {
        let n = 5u32;
        let mut state = new_state(n);
        let initial = random_state(n, 0xC0FFEEu64);
        state.write_state(&initial).expect("write");
        let mut cpu = initial;
        let ops: Vec<(u32, Complex64, Complex64)> = vec![
            (
                0,
                Complex64::from_polar(1.0, -0.35),
                Complex64::from_polar(1.0, 0.35),
            ),
            (3, Complex64::new(1.0, 0.0), Complex64::from_polar(1.0, 1.3)),
            (
                1,
                Complex64::new(1.0, 0.0),
                Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
            ),
            (4, Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)),
            (2, Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)),
        ];
        for &(q, d0, d1) in &ops {
            state.apply_diagonal(q, d0, d1).expect("diag");
            cpu_apply_diagonal(&mut cpu, q, d0, d1);
        }
        let got = state.read_state();
        let max = max_abs_diff(&got, &cpu);
        assert!(max < 1e-6, "max abs diff = {max}");
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_drz_matches_apply_1q_drz_oracle() {
        // `apply_drz` is the diagonal-fast-path for the dRz/dθ
        // derivative apply in `adjoint_gradient`. Pin that it
        // produces the same per-amplitude result as the previous
        // `apply_1q(&gates::drz(theta))` route on a random state +
        // a sweep of θ values exercising both the small-angle and
        // half-revolution regimes.
        use omega_backend_statevector::gates;

        let n = 6u32;
        let backend = new_backend();
        let initial = random_state(n, 0xDEADBEEFu64);

        for &theta in &[0.0, 0.123, 0.5, 1.234, std::f64::consts::PI, -0.7] {
            for &q in &[0u32, 2, 5] {
                let mut fast = backend.allocate(n).expect("alloc fast");
                fast.write_state(&initial).expect("write fast");
                fast.apply_drz(q, theta).expect("apply_drz");

                let mut slow = backend.allocate(n).expect("alloc slow");
                slow.write_state(&initial).expect("write slow");
                slow.apply_1q(q, &gates::drz(theta)).expect("apply_1q drz");

                let got = fast.read_state();
                let want = slow.read_state();
                let max = max_abs_diff(&got, &want);
                assert!(
                    max < 1e-6,
                    "apply_drz vs apply_1q(drz) mismatch at q={q} theta={theta}: {max}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_du1_matches_apply_1q_du1_oracle() {
        // Sister test for U1 — same diagonal-fast-path rationale.
        // The d0 factor is 0, so this also pins that
        // apply_diagonal correctly zeroes the |0⟩-amplitude
        // component (it would otherwise be a silent precision
        // bug).
        use omega_backend_statevector::gates;

        let n = 5u32;
        let backend = new_backend();
        let initial = random_state(n, 0xFEEDFACEu64);

        for &lambda in &[0.0, 0.4, 1.5, std::f64::consts::FRAC_PI_3, -0.9] {
            for &q in &[0u32, 1, 4] {
                let mut fast = backend.allocate(n).expect("alloc fast");
                fast.write_state(&initial).expect("write fast");
                fast.apply_du1(q, lambda).expect("apply_du1");

                let mut slow = backend.allocate(n).expect("alloc slow");
                slow.write_state(&initial).expect("write slow");
                slow.apply_1q(q, &gates::du1_dl(lambda))
                    .expect("apply_1q du1_dl");

                let got = fast.read_state();
                let want = slow.read_state();
                let max = max_abs_diff(&got, &want);
                assert!(
                    max < 1e-6,
                    "apply_du1 vs apply_1q(du1_dl) mismatch at q={q} lambda={lambda}: {max}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn batched_sweep_matches_unbatched_sweep_bit_identical() {
        // Pin the round-12 begin_batch/end_batch invariant: applying
        // a sequence of kernels inside an open batch must produce a
        // bit-for-bit identical result to applying them without
        // batching. The whole point of the slice is correctness
        // preserved + dispatch overhead saved; this is the regression
        // gate.
        let n = 6u32;
        let backend = new_backend();
        let initial = random_state(n, 0xBA7C45E7u64);

        // Sequence covers all 5 in-place kernel paths the batch
        // services: apply_diagonal (Rz), apply_diagonal_product
        // (multi-Rz fusion via apply_ops_fused isn't tested here —
        // direct apply_diagonal calls), apply_1q (Rx), apply_2q (CX
        // synthesised via the apply_cx wrapper which dispatches
        // apply_2q internally), apply_diagonal_2q (dCRz / synthetic).
        fn run_sequence(state: &MetalState) {
            state.apply_rx(0, 0.7).unwrap();
            state.apply_ry(2, -1.1).unwrap();
            state.apply_rz(1, 0.4).unwrap();
            state.apply_cx(0, 1).unwrap();
            state.apply_cx(2, 3).unwrap();
            state
                .apply_diagonal_2q(
                    1,
                    4,
                    Complex64::new(1.0, 0.0),
                    Complex64::from_polar(1.0, 0.55),
                    Complex64::new(1.0, 0.0),
                    Complex64::from_polar(1.0, -0.33),
                )
                .unwrap();
            state.apply_h(5).unwrap();
            state.apply_rz(4, 0.95).unwrap();
        }

        // Unbatched reference.
        let mut unbatched = backend.allocate(n).expect("alloc unbatched");
        unbatched.write_state(&initial).expect("write unbatched");
        run_sequence(&unbatched);
        let unbatched_amps = unbatched.read_state();

        // Batched run.
        let mut batched = backend.allocate(n).expect("alloc batched");
        batched.write_state(&initial).expect("write batched");
        batched.inner.begin_batch();
        run_sequence(&batched);
        batched.inner.end_batch();
        let batched_amps = batched.read_state();

        // Bit-identical — same kernels, same inputs, same scheduling
        // order; the only difference is one cmd_buf vs many.
        assert_eq!(unbatched_amps.len(), batched_amps.len());
        for (i, (u, b)) in unbatched_amps.iter().zip(batched_amps.iter()).enumerate() {
            assert_eq!(
                u.re.to_bits(),
                b.re.to_bits(),
                "amplitude {i} re differs: unbatched {} vs batched {}",
                u.re,
                b.re
            );
            assert_eq!(
                u.im.to_bits(),
                b.im.to_bits(),
                "amplitude {i} im differs: unbatched {} vs batched {}",
                u.im,
                b.im
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn nested_begin_batch_is_idempotent() {
        // Calling begin_batch while one is already open must not
        // create a second cmd_buf or lose the in-flight encoder.
        // The second begin_batch is a no-op; the original encoder
        // continues accumulating dispatches.
        let backend = new_backend();
        let mut state = backend.allocate(4).expect("alloc");
        state
            .write_state(&[Complex64::new(0.5, 0.0); 16])
            .expect("write");
        state.inner.begin_batch();
        state.apply_rz(0, 0.3).unwrap();
        // Second begin — should be a no-op.
        state.inner.begin_batch();
        state.apply_rz(1, 0.7).unwrap();
        state.inner.end_batch();
        // After end_batch, no batch should be pending. A subsequent
        // read_state must succeed without flushing anything.
        let _ = state.read_state();
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn read_state_implicitly_flushes_open_batch() {
        // If a caller forgets end_batch, host-side reads must still
        // observe the post-kernel state — `read_state` flushes any
        // open batch via `end_batch_if_open`. Catches misuse.
        let backend = new_backend();
        let mut state = backend.allocate(3).expect("alloc");
        // Initialize to |0...0⟩ (already done by allocate); apply X
        // on q0 to flip amp[0] → amp[1]. Without flushing the read
        // would still see the pre-kernel |0...0⟩ state.
        state.inner.begin_batch();
        state.apply_x(0).unwrap();
        // Note: deliberately *no* end_batch call.
        let amps = state.read_state();
        // |001⟩ in MSB-first with q0 = LSB → amp[1] = 1
        assert!(
            (amps[1] - Complex64::new(1.0, 0.0)).norm() < 1e-6,
            "implicit flush failed; amps[1] = {}",
            amps[1]
        );
        assert!(
            amps[0].norm() < 1e-6,
            "amps[0] should be 0 after X; got {}",
            amps[0]
        );
        // Confirm state is also writable after — no stuck batch.
        state.write_state(&[Complex64::new(0.0, 0.0); 8]).unwrap();
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_2q_matches_apply_2q_full_matrix_oracle() {
        // `apply_diagonal_2q(qa, qb, d00, d01, d10, d11)` must match
        // `apply_2q(qa, qb, U)` on the same initial state when U is
        // the corresponding diagonal-only 4x4 matrix. Sweep over
        // adjacent (qa, qb), distant (qa, qb), and a few d-quads
        // mixing pure phases, mixed magnitudes, and zero entries
        // (the dCRz pattern).
        let n = 6u32;
        let backend = new_backend();
        let initial = random_state(n, 0xABCDEF01u64);

        let cases = [
            // (qa, qb, [d00, d01, d10, d11])
            (
                0u32,
                1u32,
                [
                    Complex64::from_polar(1.0, 0.3),
                    Complex64::from_polar(1.0, -0.7),
                    Complex64::from_polar(1.0, 1.1),
                    Complex64::from_polar(1.0, 0.0),
                ],
            ),
            (
                2,
                4,
                [
                    Complex64::new(0.0, 0.0),
                    Complex64::from_polar(1.0, 0.5),
                    Complex64::new(0.0, 0.0),
                    Complex64::from_polar(1.0, -0.5),
                ],
            ), // dCRz-shape
            (
                5,
                0,
                [
                    Complex64::new(0.5, 0.5),
                    Complex64::new(-0.25, 0.75),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(1.0, 0.0),
                ],
            ),
        ];

        for &(qa, qb, ds) in &cases {
            let z = Complex64::new(0.0, 0.0);
            #[rustfmt::skip]
            let u = [
                ds[0], z,     z,     z,
                z,     ds[1], z,     z,
                z,     z,     ds[2], z,
                z,     z,     z,     ds[3],
            ];

            let mut fast = backend.allocate(n).expect("alloc fast");
            fast.write_state(&initial).expect("write fast");
            fast.apply_diagonal_2q(qa, qb, ds[0], ds[1], ds[2], ds[3])
                .expect("apply_diagonal_2q");

            let mut slow = backend.allocate(n).expect("alloc slow");
            slow.write_state(&initial).expect("write slow");
            slow.apply_2q(qa, qb, &u).expect("apply_2q");

            let got = fast.read_state();
            let want = slow.read_state();
            let max = max_abs_diff(&got, &want);
            assert!(max < 1e-6, "qa={qa} qb={qb}: max abs diff = {max}");
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_2q_rejects_duplicate_qubits() {
        let state = new_state(4);
        let z = Complex64::new(1.0, 0.0);
        match state.apply_diagonal_2q(2, 2, z, z, z, z) {
            Err(MetalError::DuplicateQubits { qubit }) => assert_eq!(qubit, 2),
            other => panic!("expected DuplicateQubits, got {other:?}"),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_2q_rejects_oob_qubit() {
        let state = new_state(3);
        let z = Complex64::new(1.0, 0.0);
        match state.apply_diagonal_2q(3, 0, z, z, z, z) {
            Err(MetalError::QubitOutOfRange { qubit, num_qubits }) => {
                assert_eq!(qubit, 3);
                assert_eq!(num_qubits, 3);
            }
            other => panic!("expected QubitOutOfRange, got {other:?}"),
        }
        match state.apply_diagonal_2q(0, 3, z, z, z, z) {
            Err(MetalError::QubitOutOfRange { qubit, num_qubits }) => {
                assert_eq!(qubit, 3);
                assert_eq!(num_qubits, 3);
            }
            other => panic!("expected QubitOutOfRange, got {other:?}"),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_dcrz_matches_apply_2q_dcrz_oracle() {
        // `apply_dcrz` is the diagonal-fast-path for the dCRz/dθ
        // derivative apply. Pin that it matches the previous route
        // `apply_2q(perm_2q_to_metal(&gates::dcrz(theta)))` on a
        // random state across {θ, qc, qt} sweeps. perm_2q_to_metal
        // is the row/col swap that converts CPU's control-high
        // 4x4 layout to Metal's qb-high; mirror it locally for the
        // oracle.
        use omega_backend_statevector::gates;

        fn perm_2q_to_metal_local(g: &[Complex64; 16]) -> [Complex64; 16] {
            let perm = [0usize, 2, 1, 3];
            let mut out = [Complex64::new(0.0, 0.0); 16];
            for r in 0..4 {
                for c in 0..4 {
                    out[r * 4 + c] = g[perm[r] * 4 + perm[c]];
                }
            }
            out
        }

        let n = 5u32;
        let backend = new_backend();
        let initial = random_state(n, 0x5A5A5A5Au64);

        for &theta in &[0.0, 0.1, 0.7, 1.5, std::f64::consts::PI, -0.4] {
            for &(qc, qt) in &[(0u32, 1u32), (2, 4), (3, 0), (1, 4)] {
                let mut fast = backend.allocate(n).expect("alloc fast");
                fast.write_state(&initial).expect("write fast");
                fast.apply_dcrz(qc, qt, theta).expect("apply_dcrz");

                let mut slow = backend.allocate(n).expect("alloc slow");
                slow.write_state(&initial).expect("write slow");
                let u = perm_2q_to_metal_local(&gates::dcrz(theta));
                slow.apply_2q(qc, qt, &u).expect("apply_2q dcrz");

                let got = fast.read_state();
                let want = slow.read_state();
                let max = max_abs_diff(&got, &want);
                assert!(
                    max < 1e-6,
                    "apply_dcrz vs apply_2q(dcrz) mismatch at qc={qc} qt={qt} theta={theta}: {max}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_rejects_oob_qubit() {
        let state = new_state(3);
        match state.apply_z(3) {
            Err(MetalError::QubitOutOfRange { qubit, num_qubits }) => {
                assert_eq!(qubit, 3);
                assert_eq!(num_qubits, 3);
            }
            other => panic!("expected QubitOutOfRange, got {other:?}"),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_rz_pi_on_plus_yields_minus_i_plus_i() {
        let mut state = new_state(1);
        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        state
            .write_state(&[
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ])
            .expect("write");
        state.apply_rz(0, std::f64::consts::PI).expect("Rz");
        let got = state.read_state();
        let expect = vec![
            Complex64::new(0.0, -inv_sqrt2),
            Complex64::new(0.0, inv_sqrt2),
        ];
        let max = max_abs_diff(&got, &expect);
        assert!(max < 1e-6, "max abs diff = {max}");
    }

    // ---- 1q kernel -----------------------------------------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn cpu_apply_1q(state: &mut [Complex64], qubit: u32, u: &[Complex64; 4]) {
        let dim = state.len();
        let mask = 1usize << qubit;
        let mut i = 0usize;
        while i < dim {
            if (i & mask) != 0 {
                i += 1;
                continue;
            }
            let i0 = i;
            let i1 = i | mask;
            let a = state[i0];
            let b = state[i1];
            state[i0] = u[0] * a + u[1] * b;
            state[i1] = u[2] * a + u[3] * b;
            i += 1;
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_h_on_zero_yields_plus_state() {
        let state = new_state(1);
        state.apply_h(0).expect("H");
        let got = state.read_state();
        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        let expect = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ];
        assert!(max_abs_diff(&got, &expect) < 1e-6);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_x_on_zero_yields_one_state() {
        let state = new_state(1);
        state.apply_x(0).expect("X");
        let got = state.read_state();
        let expect = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        assert!(max_abs_diff(&got, &expect) < 1e-6);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_1q_matches_cpu_oracle_random_unitaries() {
        for &n in &[3u32, 5, 8, 10] {
            let mut state = new_state(n);
            let mut cpu = random_state(n, 0xDEADBEEFu64 ^ (n as u64));
            state.write_state(&cpu).expect("write");
            let ops: Vec<(u32, [Complex64; 4])> = vec![
                (0, h_unitary()),
                (2.min(n - 1), u3_unitary(0.7, 1.1, -0.4)),
                (1.min(n - 1), x_unitary()),
                (0, ry_unitary(2.3)),
                (n - 1, h_unitary()),
                (n - 1, u3_unitary(-1.7, 0.4, 2.1)),
            ];
            for (q, u) in &ops {
                state.apply_1q(*q, u).expect("apply");
                cpu_apply_1q(&mut cpu, *q, u);
            }
            let got = state.read_state();
            let max = max_abs_diff(&got, &cpu);
            assert!(max < 1e-6, "n={n}, max diff = {max}");
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn h_unitary() -> [Complex64; 4] {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        [
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(-s, 0.0),
        ]
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn x_unitary() -> [Complex64; 4] {
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn ry_unitary(theta: f64) -> [Complex64; 4] {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        [
            Complex64::new(c, 0.0),
            Complex64::new(-s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(c, 0.0),
        ]
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn u3_unitary(theta: f64, phi: f64, lambda: f64) -> [Complex64; 4] {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        [
            Complex64::new(c, 0.0),
            -Complex64::from_polar(s, lambda),
            Complex64::from_polar(s, phi),
            Complex64::from_polar(c, phi + lambda),
        ]
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_1q_rejects_oob_qubit() {
        let state = new_state(2);
        match state.apply_h(2) {
            Err(MetalError::QubitOutOfRange { qubit, num_qubits }) => {
                assert_eq!(qubit, 2);
                assert_eq!(num_qubits, 2);
            }
            other => panic!("expected QubitOutOfRange, got {other:?}"),
        }
    }

    // ---- 2q kernel -----------------------------------------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn cpu_apply_2q(state: &mut [Complex64], qa: u32, qb: u32, u: &[Complex64; 16]) {
        assert_ne!(qa, qb, "cpu_apply_2q needs distinct qubits");
        let dim = state.len();
        let mask_a = 1usize << qa;
        let mask_b = 1usize << qb;
        let qmin = qa.min(qb) as usize;
        let qmax = qa.max(qb) as usize;
        let mid_count = qmax - qmin - 1;
        let low_mask = (1usize << qmin) - 1;
        let mid_mask = if mid_count == 0 {
            0
        } else {
            (1usize << mid_count) - 1
        };
        let quads = dim / 4;
        for tid in 0..quads {
            let low = tid & low_mask;
            let mid = ((tid >> qmin) & mid_mask) << (qmin + 1);
            let high = (tid >> (qmax - 1)) << (qmax + 1);
            let i00 = low | mid | high;
            let i01 = i00 | mask_a;
            let i10 = i00 | mask_b;
            let i11 = i00 | mask_a | mask_b;
            let v = [state[i00], state[i01], state[i10], state[i11]];
            let mut o = [Complex64::new(0.0, 0.0); 4];
            for r in 0..4 {
                for c in 0..4 {
                    o[r] += u[r * 4 + c] * v[c];
                }
            }
            state[i00] = o[0];
            state[i01] = o[1];
            state[i10] = o[2];
            state[i11] = o[3];
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn h_then_cx_yields_bell_state() {
        let state = new_state(2);
        state.apply_h(0).expect("H");
        state.apply_cx(0, 1).expect("CX");
        let got = state.read_state();
        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        let expect = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ];
        assert!(max_abs_diff(&got, &expect) < 1e-6);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn cz_on_plus_plus_yields_phase_flip_on_11() {
        let state = new_state(2);
        state.apply_h(0).expect("H");
        state.apply_h(1).expect("H");
        state.apply_cz(0, 1).expect("CZ");
        let got = state.read_state();
        let expect = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(-0.5, 0.0),
        ];
        assert!(max_abs_diff(&got, &expect) < 1e-6);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn swap_exchanges_amplitudes() {
        let mut state = new_state(2);
        let mut s = vec![Complex64::new(0.0, 0.0); 4];
        s[1] = Complex64::new(1.0, 0.0);
        state.write_state(&s).expect("write");
        state.apply_swap(0, 1).expect("SWAP");
        let got = state.read_state();
        let expect = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        assert!(max_abs_diff(&got, &expect) < 1e-6);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_2q_matches_cpu_oracle_random() {
        for &n in &[3u32, 5, 8] {
            let mut state = new_state(n);
            let mut cpu = random_state(n, 0xBADCAFE0u64 ^ (n as u64));
            state.write_state(&cpu).expect("write");
            let h_n = if n >= 4 { n - 2 } else { 1 };
            let ops: Vec<(&'static str, u32, u32, [Complex64; 16])> = vec![
                ("CX", 0, h_n.max(1), build_cx_matrix()),
                ("CZ", 1.min(n - 1), n - 1, build_cz_matrix()),
                ("SWAP", 0, 1.min(n - 1), build_swap_matrix()),
                ("CRz", 0, n - 1, build_crz_matrix(0.7)),
            ];
            for (label, qa, qb, u) in &ops {
                if *qa == *qb {
                    continue;
                }
                state
                    .apply_2q(*qa, *qb, u)
                    .unwrap_or_else(|e| panic!("{} on {qa},{qb} failed: {e}", label));
                cpu_apply_2q(&mut cpu, *qa, *qb, u);
            }
            let got = state.read_state();
            let max = max_abs_diff(&got, &cpu);
            assert!(max < 1e-6, "n={n}, max diff = {max}");
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn build_cx_matrix() -> [Complex64; 16] {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        [
            o, z, z, z, //
            z, z, z, o, //
            z, z, o, z, //
            z, o, z, z,
        ]
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn build_cz_matrix() -> [Complex64; 16] {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let m = Complex64::new(-1.0, 0.0);
        [
            o, z, z, z, //
            z, o, z, z, //
            z, z, o, z, //
            z, z, z, m, //
        ]
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn build_swap_matrix() -> [Complex64; 16] {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        [
            o, z, z, z, //
            z, z, o, z, //
            z, o, z, z, //
            z, z, z, o, //
        ]
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn build_crz_matrix(theta: f64) -> [Complex64; 16] {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let phn = Complex64::from_polar(1.0, -theta / 2.0);
        let php = Complex64::from_polar(1.0, theta / 2.0);
        [
            o, z, z, z, //
            z, phn, z, z, //
            z, z, o, z, //
            z, z, z, php,
        ]
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_2q_rejects_duplicate_qubits() {
        let state = new_state(3);
        match state.apply_cx(1, 1) {
            Err(MetalError::DuplicateQubits { qubit }) => assert_eq!(qubit, 1),
            other => panic!("expected DuplicateQubits, got {other:?}"),
        }
    }

    // ---- Backend-trait integration ------------------------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn backend_execute_bell_circuit_matches_cpu() {
        // Hand-build a Bell IR (H q0; CX q0 q1) and execute under both
        // the metal Backend trait and the CPU StatevectorBackend; states
        // must agree to 1e-6.
        use omega_core::circuit::{CircuitIR, CircuitType, GateOp, Qubit};
        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        circuit.ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            condition: None,
            classical_bit: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(0), Qubit(1)],
            params: smallvec::smallvec![],
            condition: None,
            classical_bit: None,
        });

        let pb = ParameterBinding::new();
        let cfg = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let metal_backend = new_backend();
        let metal_result = metal_backend
            .execute(&circuit, &pb, &cfg)
            .expect("metal execute");
        let metal_sv = match &metal_result {
            ExecResult::Statevector(sv) => sv.clone(),
            _ => panic!("expected Statevector"),
        };

        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        let expect = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ];
        assert!(max_abs_diff(&metal_sv, &expect) < 1e-6);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn backend_execute_with_shots_returns_counts() {
        use omega_core::circuit::{CircuitIR, CircuitType, GateOp, Qubit};
        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        circuit.ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            condition: None,
            classical_bit: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(0), Qubit(1)],
            params: smallvec::smallvec![],
            condition: None,
            classical_bit: None,
        });
        let pb = ParameterBinding::new();
        let cfg = ExecConfig {
            shots: Some(2048),
            seed: Some(42),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let backend = new_backend();
        let result = backend.execute(&circuit, &pb, &cfg).expect("execute");
        let counts = match &result {
            ExecResult::Counts(c) => c,
            _ => panic!("expected Counts"),
        };
        // Bell state: shots split between |00⟩ (=0) and |11⟩ (=3).
        let total: u32 = counts.values().sum();
        assert_eq!(total, 2048);
        let on_diag = counts.get(&0).copied().unwrap_or(0) + counts.get(&3).copied().unwrap_or(0);
        assert!(
            on_diag as f64 / total as f64 > 0.95,
            "expected ≥ 95% on |00⟩+|11⟩, got {on_diag}/{total}: {counts:?}"
        );
    }

    // ---- inner_product GPU reduction (step 7) -------------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn inner_product_self_equals_norm_squared() {
        // ⟨ψ|ψ⟩ on a normalised pseudo-random state must equal 1
        // within f32 round-off.
        for &n in &[3u32, 6, 10, 12] {
            let mut state = new_state(n);
            let s = random_state(n, 0xA1B2C3D4u64 ^ (n as u64));
            state.write_state(&s).expect("write");
            let ip = state.inner_product(&state).expect("ip");
            assert!(
                (ip.re - 1.0).abs() < 1e-5,
                "n={n}: ⟨ψ|ψ⟩.re = {} (want 1)",
                ip.re
            );
            assert!(ip.im.abs() < 1e-5, "n={n}: ⟨ψ|ψ⟩.im = {} (want 0)", ip.im);
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn inner_product_orthogonal_states_are_zero() {
        // ⟨0…0|1…1⟩ = 0.
        let n: u32 = 4;
        let dim = 1usize << n;
        let mut a = new_state(n);
        let mut b = new_state(n);
        let mut sa = vec![Complex64::new(0.0, 0.0); dim];
        sa[0] = Complex64::new(1.0, 0.0);
        let mut sb = vec![Complex64::new(0.0, 0.0); dim];
        sb[dim - 1] = Complex64::new(1.0, 0.0);
        a.write_state(&sa).unwrap();
        b.write_state(&sb).unwrap();
        let ip = a.inner_product(&b).expect("ip");
        assert!(ip.norm() < 1e-6, "expected ~0, got {ip:?}");
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn inner_product_matches_cpu_dot_product() {
        // GPU vs the same dot product computed host-side. Two
        // pseudo-random normalised states; max abs diff < 1e-5.
        let n: u32 = 8;
        let mut a = new_state(n);
        let mut b = new_state(n);
        let sa = random_state(n, 0x11111111u64);
        let sb = random_state(n, 0x22222222u64);
        a.write_state(&sa).unwrap();
        b.write_state(&sb).unwrap();
        let gpu_ip = a.inner_product(&b).expect("ip");
        let cpu_ip: Complex64 = sa.iter().zip(sb.iter()).map(|(x, y)| x.conj() * y).sum();
        let diff = (gpu_ip - cpu_ip).norm();
        assert!(diff < 1e-5, "gpu = {gpu_ip:?}, cpu = {cpu_ip:?}");
    }

    // ---- pauli_expectation fused kernel -------------------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn pauli_expectation_matches_cpu_oracle() {
        // GPU `pauli_expectation` vs the host-side `cpu_pauli_expectation`
        // reference on a pseudo-random normalised state for assorted
        // Pauli strings (Z-only, X-only, Y-only, mixed). f32 vs f64
        // tolerance: 1e-5.
        use omega_core::executor::PauliOp;

        let n: u32 = 8;
        let mut state = new_state(n);
        let s = random_state(n, 0xCAFEBEEFu64);
        state.write_state(&s).unwrap();

        // Mix of strings: identity, single-qubit each Pauli, two-qubit
        // ZZ and XX, three-qubit XYZ. Hits every branch of pauli_masks.
        let strings: Vec<Vec<(u32, PauliOp)>> = vec![
            vec![],
            vec![(0, PauliOp::Z)],
            vec![(0, PauliOp::X)],
            vec![(0, PauliOp::Y)],
            vec![(0, PauliOp::Z), (1, PauliOp::Z)],
            vec![(0, PauliOp::X), (3, PauliOp::X)],
            vec![(2, PauliOp::Y), (5, PauliOp::Y)],
            vec![(0, PauliOp::X), (1, PauliOp::Y), (2, PauliOp::Z)],
            vec![(0, PauliOp::Y), (3, PauliOp::Y), (6, PauliOp::Y)],
            vec![
                (0, PauliOp::X),
                (1, PauliOp::Y),
                (2, PauliOp::Z),
                (3, PauliOp::I),
                (4, PauliOp::X),
                (5, PauliOp::Z),
            ],
        ];

        for (idx, p_str) in strings.iter().enumerate() {
            let cpu = cpu_pauli_expectation(&s, n, p_str);
            let (x_mask, sign_mask, y_factor) = pauli_masks(p_str);
            let gpu = state
                .pauli_expectation(x_mask, sign_mask, y_factor)
                .expect("pauli_expectation");
            let diff = (gpu.re - cpu).abs();
            assert!(
                diff < 1e-5,
                "string #{idx}: cpu = {cpu}, gpu.re = {}, diff = {diff}",
                gpu.re
            );
            // Hermitian Pauli ⇒ imag part should be ~0.
            assert!(
                gpu.im.abs() < 1e-5,
                "string #{idx}: gpu.im = {} (Hermitian Pauli expected real)",
                gpu.im
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn pauli_expectation_bell_state_classic_observables() {
        // Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2:
        //   ⟨Z⊗Z⟩ = +1, ⟨X⊗X⟩ = +1, ⟨Y⊗Y⟩ = -1, ⟨Z⊗I⟩ = 0.
        // These are the textbook values everyone learns; they pin
        // down the kernel's sign / Y-prefactor handling.
        use omega_core::executor::PauliOp;

        let n: u32 = 2;
        let mut state = new_state(n);
        let inv_sqrt2 = (1.0_f64 / 2.0_f64).sqrt();
        let bell = vec![
            Complex64::new(inv_sqrt2, 0.0), // |00⟩
            Complex64::new(0.0, 0.0),       // |01⟩
            Complex64::new(0.0, 0.0),       // |10⟩
            Complex64::new(inv_sqrt2, 0.0), // |11⟩
        ];
        state.write_state(&bell).unwrap();

        type Case<'a> = (&'a str, Vec<(u32, PauliOp)>, f64);
        let cases: &[Case<'_>] = &[
            ("ZZ", vec![(0, PauliOp::Z), (1, PauliOp::Z)], 1.0),
            ("XX", vec![(0, PauliOp::X), (1, PauliOp::X)], 1.0),
            ("YY", vec![(0, PauliOp::Y), (1, PauliOp::Y)], -1.0),
            ("ZI", vec![(0, PauliOp::Z)], 0.0),
        ];
        for (label, p_str, want) in cases {
            let (x_mask, sign_mask, y_factor) = pauli_masks(p_str);
            let gpu = state
                .pauli_expectation(x_mask, sign_mask, y_factor)
                .expect("pauli_expectation");
            assert!(
                (gpu.re - want).abs() < 1e-5,
                "{label}: got {} want {want}",
                gpu.re
            );
        }
    }

    // ---- Adjoint AD integration tests ---------------------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn adjoint_metal_matches_cpu_12q_hea() {
        // GPU_PLAN.md Phase 1 target: 12-qubit / 16-parameter
        // hardware-efficient ansatz. Two layers of single-qubit
        // rotations + a CX entangling ladder, sharing 16 free symbols.
        // Metal adjoint must match CPU within 1e-5 (relaxed from
        // 1e-10 because the GPU layout is f32; CPU stays f64).
        use omega_backend_statevector::StatevectorBackend;
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};
        use omega_core::executor::PauliOp;

        let n: u32 = 12;
        let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
        for s in 0..16u32 {
            circuit.symbols.insert(s, format!("theta_{s}"));
        }
        // Layer 1: Ry on first 8 qubits (params 0..8).
        for q in 0..8 {
            circuit.ops.push(GateOp {
                gate: GateKind::Ry,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(q)],
                classical_bit: None,
                condition: None,
            });
        }
        // Linear CX entangling ladder.
        for q in 0..n - 1 {
            circuit.ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec::smallvec![Qubit(q), Qubit(q + 1)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        // Layer 2: Rz on the last 8 qubits (params 8..16).
        for (i, q) in (4..n).enumerate() {
            circuit.ops.push(GateOp {
                gate: GateKind::Rz,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(8 + i as u32)],
                classical_bit: None,
                condition: None,
            });
        }

        let mut params = ParameterBinding::new();
        for s in 0..16u32 {
            let v = ((s as f64) * 0.317 - 0.84).sin() * 1.4;
            params.bind(s, v);
        }
        let obs = Observable {
            terms: vec![
                (1.0, vec![(0, PauliOp::Z)]),
                (0.5, vec![(5, PauliOp::Z)]),
                (-0.7, vec![(0, PauliOp::Z), (11, PauliOp::Z)]),
                (0.3, vec![(3, PauliOp::X)]),
            ],
        };

        let cpu = StatevectorBackend::new();
        let cpu_grads = cpu
            .adjoint_gradient(&circuit, &params, &obs)
            .expect("cpu adjoint")
            .expect("cpu has adjoint");

        let metal = MetalStatevectorBackend::new().expect("metal");
        let metal_grads = metal
            .adjoint_gradient(&circuit, &params, &obs)
            .expect("metal adjoint")
            .expect("metal has adjoint");

        assert_eq!(cpu_grads.len(), 16);
        assert_eq!(metal_grads.len(), 16);
        let mut max_diff = 0.0_f64;
        for ((sa, ga), (sb, gb)) in cpu_grads.iter().zip(metal_grads.iter()) {
            assert_eq!(sa, sb);
            let diff = (ga - gb).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 1e-5,
                "symbol {sa}: cpu = {ga:.10}, metal = {gb:.10}, diff = {diff:.3e}"
            );
        }
        eprintln!("12q/16-param HEA: max abs diff vs CPU adjoint = {max_diff:.3e}");
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn reset_matches_cpu() {
        // Reset is a CHANNEL (projective measure, then X if the outcome was 1),
        // not a coherent fold. Metal must agree with the CPU backend on BOTH
        // halves of that:
        //
        //   (a) an UNENTANGLED reset is deterministic and both must produce the
        //       same state — including the |−⟩ case, where the old coherent
        //       fold gave `old0 + old1 = 0` and silently skipped the
        //       renormalise, leaving zero amplitude;
        //   (b) an ENTANGLED reset in analytic mode is ill-defined (the true
        //       result is mixed) and both must REFUSE. Metal used to return a
        //       pure-state answer here while the CPU errored.
        //
        // The original version of this test asserted (a) using an entangled
        // circuit, so it broke as soon as the CPU gained its guard.
        use omega_backend_statevector::StatevectorBackend;
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};

        let push = |c: &mut CircuitIR, gate, qubits: &[u32], params: &[f64]| {
            c.ops.push(GateOp {
                gate,
                qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
                params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
                classical_bit: None,
                condition: None,
            });
        };
        let params = ParameterBinding::new();
        let cfg = ExecConfig {
            shots: None,
            ..ExecConfig::default()
        };
        let cpu = StatevectorBackend::new();
        let metal = MetalStatevectorBackend::new().expect("metal");

        // (a) Unentangled resets, including |−⟩ (H then Reset) and |1⟩ (X then
        // Reset) — the two the coherent fold got wrong.
        for (name, prep) in [
            ("|+>", vec![(GateKind::H, 0u32, vec![])]),
            ("|->", vec![(GateKind::X, 0u32, vec![]), (GateKind::H, 0u32, vec![])]),
            ("|1>", vec![(GateKind::X, 0u32, vec![])]),
            ("Ry", vec![(GateKind::Ry, 0u32, vec![0.7])]),
        ] {
            let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
            for (g, q, ps) in &prep {
                push(&mut circuit, g.clone(), &[*q], ps);
            }
            // Qubit 1 stays in a non-trivial product state so a wrong reset on
            // qubit 0 cannot be masked by an all-|0> register.
            push(&mut circuit, GateKind::Ry, &[1], &[0.4]);
            push(&mut circuit, GateKind::Reset, &[0], &[]);
            push(&mut circuit, GateKind::H, &[1], &[]);

            let a = match cpu.execute(&circuit, &params, &cfg).expect("cpu") {
                ExecResult::Statevector(v) => v,
                _ => panic!("expected statevector"),
            };
            let b = match metal.execute(&circuit, &params, &cfg).expect("metal") {
                ExecResult::Statevector(v) => v,
                _ => panic!("expected statevector"),
            };
            assert_eq!(a.len(), b.len(), "{name}");
            // Compare UP TO GLOBAL PHASE. Reset ends at |0>, but via a
            // projective measurement whose branch the CPU picks with an RNG:
            // on |−⟩ (p0 = 0.5) the outcome-1 branch renormalises to −|0> and
            // then applies X, so the CPU may return −|0>⊗rest where Metal —
            // which has no RNG and always takes the p0 ≥ 0.5 branch — returns
            // +|0>⊗rest. That is a global phase, physically the same state.
            // An amplitude-wise diff reports ~1.67 for it, which is what the
            // first version of this assertion did.
            let overlap: Complex64 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| x.conj() * y)
                .fold(Complex64::new(0.0, 0.0), |acc, z| acc + z);
            assert!(
                (overlap.norm() - 1.0).abs() < 1e-4,
                "{name}: |<cpu|metal>| = {:.6}, expected 1 (equal up to global phase)",
                overlap.norm()
            );
            // Non-degenerate: the state must not be all-zero, which is exactly
            // what the old fold produced for |−⟩.
            let norm: f64 = b.iter().map(|z| z.norm_sqr()).sum();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "{name}: metal norm {norm:.6}, expected 1 (a zeroed register would pass a \
                 pure difference check against an equally-zeroed CPU state)"
            );
        }

        // (b) Entangled reset in analytic mode: BOTH must refuse.
        let mut ent = CircuitIR::new(2, CircuitType::GateBased);
        push(&mut ent, GateKind::H, &[0], &[]);
        push(&mut ent, GateKind::CX, &[0, 1], &[]);
        push(&mut ent, GateKind::Reset, &[0], &[]);
        let cpu_err = cpu.execute(&ent, &params, &cfg).is_err();
        let metal_err = metal.execute(&ent, &params, &cfg).is_err();
        assert!(cpu_err, "cpu accepted an entangled analytic Reset");
        assert!(
            metal_err,
            "metal accepted an entangled analytic Reset — it returns a pure state \
             where the truth is mixed, while the CPU refuses"
        );
    }


    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn expectation_matches_cpu_pauli_string() {
        // Odd-Y regression: the observable's `-0.4·⟨Y2⟩` term (|Y| = 1) pins the
        // `(-i)^|Y|` Pauli prefactor. With the old `i^|Y|` it came out sign-
        // flipped (~0.5 error). Mirrors the CUDA `expectation_matches_cpu_pauli_string`.
        use omega_backend_statevector::StatevectorBackend;
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};
        use omega_core::executor::PauliOp;

        let n: u32 = 5;
        let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
        circuit.symbols.insert(0, "theta".into());
        circuit.ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec::smallvec![Qubit(2)],
            params: smallvec::smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(0), Qubit(3)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });

        let mut params = ParameterBinding::new();
        params.bind(0, 0.731);
        let obs = Observable {
            terms: vec![
                (1.5, vec![(0, PauliOp::X), (3, PauliOp::Z)]),
                (-0.4, vec![(2, PauliOp::Y)]),
                (0.6, vec![(1, PauliOp::Z), (4, PauliOp::Z)]),
            ],
        };

        let cpu = StatevectorBackend::new();
        let cpu_e = cpu.expectation(&circuit, &params, &obs).expect("cpu");
        let metal = MetalStatevectorBackend::new().expect("metal");
        let metal_e = metal.expectation(&circuit, &params, &obs).expect("metal");
        assert!(
            (cpu_e - metal_e).abs() < 1e-5,
            "cpu = {cpu_e}, metal = {metal_e}, diff = {:.3e}",
            (cpu_e - metal_e).abs()
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn adjoint_metal_matches_cpu() {
        // Small parameterized ansatz: 4 qubits, 6 parameters. Tests the
        // full Rx/Ry/Rz + CX adjoint path against the CPU reference.
        // Target tolerance 1e-5 (relaxed from the CPU's 1e-10 because
        // the GPU layout is f32; CPU stays f64).
        use omega_backend_statevector::StatevectorBackend;
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};
        use omega_core::executor::PauliOp;

        let n: u32 = 4;
        let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
        // Symbol IDs 0..6 for parameters.
        for s in 0..6u32 {
            circuit.symbols.insert(s, format!("theta_{s}"));
        }
        // First layer: Ry on each qubit, parameterised.
        for q in 0..n {
            circuit.ops.push(GateOp {
                gate: GateKind::Ry,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(q)],
                classical_bit: None,
                condition: None,
            });
        }
        // Entangling CX ring.
        for q in 0..n {
            circuit.ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec::smallvec![Qubit(q), Qubit((q + 1) % n)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        // Second layer: Rx + Rz on q0 and q1, two more params.
        circuit.ops.push(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![ParamExpr::Symbol(4)],
            classical_bit: None,
            condition: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec::smallvec![Qubit(1)],
            params: smallvec::smallvec![ParamExpr::Symbol(5)],
            classical_bit: None,
            condition: None,
        });

        let mut params = ParameterBinding::new();
        // Pseudo-arbitrary parameter values.
        let values = [0.32_f64, -1.17, 0.81, 1.43, -0.55, 0.93];
        for (s, v) in values.iter().enumerate() {
            params.bind(s as u32, *v);
        }

        // Observable: Z_0 + 0.5 · Z_0 Z_2 + 0.3 · X_1.
        let obs = Observable {
            terms: vec![
                (1.0, vec![(0, PauliOp::Z)]),
                (0.5, vec![(0, PauliOp::Z), (2, PauliOp::Z)]),
                (0.3, vec![(1, PauliOp::X)]),
            ],
        };

        let cpu = StatevectorBackend::new();
        let cpu_grads = cpu
            .adjoint_gradient(&circuit, &params, &obs)
            .expect("cpu adjoint")
            .expect("cpu has adjoint");

        let metal = MetalStatevectorBackend::new().expect("metal");
        let metal_grads = metal
            .adjoint_gradient(&circuit, &params, &obs)
            .expect("metal adjoint")
            .expect("metal has adjoint");

        assert_eq!(cpu_grads.len(), metal_grads.len());
        for ((sa, ga), (sb, gb)) in cpu_grads.iter().zip(metal_grads.iter()) {
            assert_eq!(sa, sb, "symbol id mismatch");
            assert!(
                (ga - gb).abs() < 1e-5,
                "symbol {sa}: cpu = {ga:.10}, metal = {gb:.10}, diff = {:.3e}",
                (ga - gb).abs()
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn ccx_truth_table_matches_classical_toffoli() {
        // For each computational-basis input |c1 c2 t⟩, CCX should flip
        // t iff c1 = c2 = 1. Verify on all 8 inputs against the
        // analytical truth table.
        for input in 0..8u32 {
            let c1 = (input >> 2) & 1;
            let c2 = (input >> 1) & 1;
            let t = input & 1;
            let expect_t = if c1 == 1 && c2 == 1 { t ^ 1 } else { t };
            let expect_idx = (c1 << 2) | (c2 << 1) | expect_t;

            let mut state = new_state(3);
            let mut s = vec![Complex64::new(0.0, 0.0); 8];
            // qubit indices in our convention: amplitude index =
            // q2 << 2 | q1 << 1 | q0. Pick q0=t, q1=c2, q2=c1 so the
            // bit pattern matches |c1 c2 t⟩ as written above.
            let amp_idx = ((c1 as usize) << 2) | ((c2 as usize) << 1) | (t as usize);
            s[amp_idx] = Complex64::new(1.0, 0.0);
            state.write_state(&s).expect("write");
            state.apply_ccx(2, 1, 0).expect("CCX");

            let got = state.read_state();
            for (idx, amp) in got.iter().enumerate() {
                let want = if idx == expect_idx as usize { 1.0 } else { 0.0 };
                assert!(
                    (amp.norm() - want).abs() < 1e-5,
                    "input {input:03b}: expected amp at {expect_idx:03b} = {want}, got idx {idx} = {amp:?}"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    #[allow(clippy::identity_op)] // bit-pattern shows intent: qc<<2 | qa<<1 | qb
    fn cswap_swaps_targets_when_control_one() {
        // |c=1, a=0, b=1⟩ → |c=1, a=1, b=0⟩ ; |c=0, ...⟩ unchanged.
        // Amplitude index = qc<<2 | qa<<1 | qb (q0=b, q1=a, q2=c).
        let mut state = new_state(3);
        let mut s = vec![Complex64::new(0.0, 0.0); 8];
        let in_idx = (1usize << 2) | (0usize << 1) | 1usize;
        s[in_idx] = Complex64::new(1.0, 0.0);
        state.write_state(&s).expect("write");
        state.apply_cswap(2, 1, 0).expect("CSwap");
        let got = state.read_state();
        let out_idx = (1usize << 2) | (1usize << 1) | 0usize;
        assert!(
            (got[out_idx].norm() - 1.0).abs() < 1e-5,
            "expected swap, got {got:?}"
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn backend_rejects_collapse_measurement_with_unsupported() {
        // The CLI dispatcher falls back to the CPU statevector backend on
        // `Unsupported`, so what this backend refuses is load-bearing. Reset
        // is no longer refused — it runs the deterministic fold-and-
        // renormalise (see `reset_matches_cpu`) — but mid-circuit measurement
        // WITH COLLAPSE still has no GPU implementation and must keep failing
        // loudly rather than silently returning an uncollapsed state.
        use omega_core::circuit::{CircuitIR, CircuitType, GateOp, Qubit};
        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.ops.push(GateOp {
            gate: GateKind::Measure,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            condition: None,
            classical_bit: Some(0),
        });
        let pb = ParameterBinding::new();
        let cfg = ExecConfig {
            shots: None,
            seed: None,
            mid_circuit_mode: MidCircuitMode::Collapse,
        };
        let backend = new_backend();
        let res = backend.execute(&circuit, &pb, &cfg);
        assert!(matches!(res, Err(OmegaError::Unsupported(_))));
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_pauli_sum_matches_host_observable_for_z_only() {
        // Build a known state on 3 qubits and apply a Z-only Pauli-sum
        // observable via the GPU kernel; compare against a CPU oracle
        // that does the same arithmetic. They must agree within f32
        // round-off (Metal is f32 internally).
        let backend = new_backend();
        let mut psi_state = backend.allocate(3).expect("alloc psi");
        // Hand-rolled non-trivial state: equal-magnitude amps with
        // different complex phases per basis state.
        let dim = 1usize << 3;
        let mut amps = Vec::with_capacity(dim);
        for i in 0..dim {
            let theta = (i as f64) * 0.31415;
            amps.push(Complex64::new(theta.cos(), theta.sin()) / (dim as f64).sqrt());
        }
        psi_state.write_state(&amps).expect("write psi");

        // Observable: 1.5 · Z₀ + (-0.7) · Z₁·Z₂ + 0.25 · I.
        let dst_state = backend.allocate(3).expect("alloc dst");
        let terms: Vec<(u32, f32)> = vec![(0b001, 1.5), (0b110, -0.7), (0b000, 0.25)];
        psi_state
            .apply_diagonal_pauli_sum(&dst_state, &terms)
            .expect("apply_diagonal_pauli_sum");
        let gpu_nu = dst_state.read_state();

        // CPU reference: ν[i] = ψ[i] · Σ_k c_k · (-1)^popcount(i & s_k).
        let cpu_nu: Vec<Complex64> = (0..dim)
            .map(|i| {
                let mut scale = 0.0f64;
                for (mask, coeff) in &terms {
                    let parity = ((i as u32) & mask).count_ones() & 1;
                    if parity == 0 {
                        scale += *coeff as f64;
                    } else {
                        scale -= *coeff as f64;
                    }
                }
                amps[i] * scale
            })
            .collect();

        for (g, c) in gpu_nu.iter().zip(cpu_nu.iter()) {
            assert!(
                (g.re - c.re).abs() < 1e-5 && (g.im - c.im).abs() < 1e-5,
                "GPU diag-Pauli-sum diverged from CPU oracle: {g:?} vs {c:?}"
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_pauli_sum_with_empty_terms_zero_fills_dst() {
        // Empty term list ⇒ ν = 0|ψ⟩ = 0 across all amplitudes.
        // Uses the `write_bytes(0)` short-circuit in the imp.rs path.
        let backend = new_backend();
        let mut psi_state = backend.allocate(2).expect("alloc psi");
        psi_state
            .write_state(&[
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
            ])
            .expect("write");
        let dst_state = backend.allocate(2).expect("alloc dst");
        psi_state
            .apply_diagonal_pauli_sum(&dst_state, &[])
            .expect("empty terms must succeed");
        let nu = dst_state.read_state();
        for amp in &nu {
            assert!(
                amp.norm() < 1e-9,
                "empty observable must zero ν, got {amp:?}"
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn expectation_multi_matches_per_observable_loop() {
        // 3-qubit Bell-style state, then evaluate three observables
        // (single-Z, two-qubit Z⊗Z, scaled-Z + identity offset). The
        // batched call must agree with N independent `expectation`
        // calls bit-for-bit (within f32 round-off — Metal is f32
        // internally, so we use 1e-5).
        use omega_core::circuit::{CircuitIR, CircuitType, GateOp, Qubit};
        use omega_core::executor::{Observable, PauliOp};

        let mut circuit = CircuitIR::new(3, CircuitType::GateBased);
        circuit.ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            condition: None,
            classical_bit: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(0), Qubit(1)],
            params: smallvec::smallvec![],
            condition: None,
            classical_bit: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(1), Qubit(2)],
            params: smallvec::smallvec![],
            condition: None,
            classical_bit: None,
        });
        let observables = vec![
            Observable {
                terms: vec![(1.0, vec![(0, PauliOp::Z)])],
            },
            Observable {
                terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
            },
            Observable {
                terms: vec![(0.5, vec![(2, PauliOp::Z)]), (0.25, vec![])],
            },
        ];
        let pb = ParameterBinding::new();
        let backend = new_backend();
        let multi = backend
            .expectation_multi(&circuit, &pb, &observables)
            .unwrap();
        assert_eq!(multi.len(), 3);
        for (obs, m) in observables.iter().zip(multi.iter()) {
            let single = backend.expectation(&circuit, &pb, obs).unwrap();
            assert!(
                (m - single).abs() < 1e-5,
                "expectation_multi disagreed with expectation: {m} vs {single}"
            );
        }
    }

    // ---- copy_into adjoint scratch helper -----------------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn copy_into_overwrites_dst_with_src_amplitudes() {
        // Write a pseudo-random normalised state to `src`, allocate a
        // fresh `dst` (which initialises to |0…0⟩), then `copy_into`.
        // Reading `dst` back must match `src` exactly — copy_into is a
        // raw memcpy of the f32 buffer, no kernel arithmetic.
        for &n in &[2u32, 5, 8, 10] {
            let backend = new_backend();
            let mut src = backend.allocate(n).expect("alloc src");
            let dst = backend.allocate(n).expect("alloc dst");
            let want = random_state(n, 0xC0_FFEE_u64 ^ (n as u64));
            src.write_state(&want).expect("write src");
            src.copy_into(&dst).expect("copy_into");
            let got = dst.read_state();
            let max = max_abs_diff(&want, &got);
            assert!(max < 1e-6, "n={n}: max abs diff = {max}");
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn copy_into_does_not_modify_source() {
        // The adjoint inner loop relies on |φ⟩ surviving the
        // copy_into → derivative → inner_product round so that the
        // next sym in the same op still operates on the same |φ⟩.
        // Pin that the copy is a true read-only snapshot of `src`.
        let n: u32 = 6;
        let backend = new_backend();
        let mut src = backend.allocate(n).expect("alloc src");
        let dst = backend.allocate(n).expect("alloc dst");
        let want = random_state(n, 0xDEAD_BEEF_u64);
        src.write_state(&want).expect("write src");
        src.copy_into(&dst).expect("copy_into");
        let src_after = src.read_state();
        let max = max_abs_diff(&want, &src_after);
        assert!(max < 1e-6, "src mutated by copy_into: max abs diff = {max}");
    }

    // ---- apply_diagonal_product fused-gate kernel --------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_product_matches_sequential_apply_diagonal() {
        // Pin that one fused dispatch produces bit-for-bit (within
        // f32) the same state as N sequential `apply_diagonal` calls.
        // Diagonal gates commute, so order doesn't matter, but the
        // factors must each contribute their bit-conditioned `d0` /
        // `d1` to the per-amplitude product.
        let n: u32 = 5;
        let backend = new_backend();
        // Pick a non-trivial mix of diagonal factors hitting
        // different qubits with different (re, im) phases — Rz-style
        // pairs (d0 = e^{-iθ/2}, d1 = e^{iθ/2}) plus one scaling pair.
        let factors = vec![
            (
                0u32,
                Complex64::from_polar(1.0, -0.31),
                Complex64::from_polar(1.0, 0.31),
            ), // Rz(0.62) on q0
            (
                2u32,
                Complex64::from_polar(1.0, -0.71),
                Complex64::from_polar(1.0, 0.71),
            ), // Rz(1.42) on q2
            (4u32, Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)), // S gate on q4
        ];

        // Reference: write the same input state to a fresh buffer
        // and apply factors sequentially via apply_diagonal.
        let want_init = random_state(n, 0xFEED_FACE_u64);
        let mut sequential = backend.allocate(n).unwrap();
        sequential.write_state(&want_init).unwrap();
        for (q, d0, d1) in &factors {
            sequential.apply_diagonal(*q, *d0, *d1).unwrap();
        }

        // Fused: same input + the new method.
        let mut fused = backend.allocate(n).unwrap();
        fused.write_state(&want_init).unwrap();
        fused.apply_diagonal_product(&factors).unwrap();

        let seq_state = sequential.read_state();
        let fused_state = fused.read_state();
        let max = max_abs_diff(&seq_state, &fused_state);
        assert!(
            max < 1e-5,
            "fused dispatch diverged from sequential: max abs diff = {max}"
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_product_with_empty_factors_is_noop() {
        // Empty factor list ⇒ identity; the state buffer must come
        // back unchanged. Matches the empty-observable convention
        // shared with `apply_diagonal_pauli_sum`.
        let n: u32 = 4;
        let backend = new_backend();
        let mut state = backend.allocate(n).unwrap();
        let want = random_state(n, 0xBEEF_C0FFu64);
        state.write_state(&want).unwrap();
        state.apply_diagonal_product(&[]).unwrap();
        let got = state.read_state();
        let max = max_abs_diff(&want, &got);
        // f32 round-trip on write_state / read_state — match the
        // existing write/read roundtrip pin's tolerance.
        assert!(max < 1e-6, "empty factor list must be no-op: diff = {max}");
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_product_rejects_out_of_range_qubit() {
        // A qubit index ≥ num_qubits in any factor must surface as a
        // typed `QubitOutOfRange` before any kernel dispatch — matches
        // the per-call apply_diagonal contract.
        let n: u32 = 3;
        let backend = new_backend();
        let state = backend.allocate(n).unwrap();
        let factors = vec![
            (1u32, Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)),
            (5u32, Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)), // out of range
        ];
        match state.apply_diagonal_product(&factors) {
            Err(MetalError::QubitOutOfRange { qubit, num_qubits }) => {
                assert_eq!(qubit, 5);
                assert_eq!(num_qubits, n);
            }
            other => panic!("expected QubitOutOfRange, got {other:?}"),
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_diagonal_product_eight_rz_layer_matches_sequential() {
        // The HEA bench's canonical layer: 8 Rz gates on q4..q11
        // (different angles per slot). At n=12 this exercises the
        // dim=4096 fused dispatch path with a realistic factor count.
        let n: u32 = 12;
        let backend = new_backend();
        let init = random_state(n, 0xCAFE_BABEu64);

        // Eight independent Rz angles — same shape the QML trainer
        // emits via the second HEA layer.
        let factors: Vec<(u32, Complex64, Complex64)> = (0..8)
            .map(|i| {
                let theta = 0.137 * (i as f64 + 1.0);
                let half = theta / 2.0;
                (
                    4 + i as u32,
                    Complex64::from_polar(1.0, -half),
                    Complex64::from_polar(1.0, half),
                )
            })
            .collect();

        // Reference: sequential apply_diagonal calls.
        let mut sequential = backend.allocate(n).unwrap();
        sequential.write_state(&init).unwrap();
        for (q, d0, d1) in &factors {
            sequential.apply_diagonal(*q, *d0, *d1).unwrap();
        }

        // Fused: one dispatch.
        let mut fused = backend.allocate(n).unwrap();
        fused.write_state(&init).unwrap();
        fused.apply_diagonal_product(&factors).unwrap();

        let max = max_abs_diff(&sequential.read_state(), &fused.read_state());
        assert!(max < 1e-5, "8-Rz fused vs sequential diff = {max}");
    }

    // ---- diagonal_factor + apply_ops_fused walker --------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    use omega_core::circuit::{GateOp, ParamExpr};

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn make_op(gate: GateKind, qubits: &[u32], params: Vec<ParamExpr>) -> GateOp {
        use omega_core::circuit::Qubit;
        GateOp {
            gate,
            qubits: qubits.iter().map(|q| Qubit(*q)).collect(),
            params: params.into_iter().collect(),
            classical_bit: None,
            condition: None,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn diagonal_factor_classifies_diagonal_gates() {
        // Z / S / Sdg / T / Tdg / Rz / U1 must all classify; the
        // returned (d0, d1) must match what each gate's diagonal
        // form prescribes. Bind theta + lambda explicitly so the
        // Rz / U1 cases can be checked numerically.
        let mut params = ParameterBinding::new();
        params.bind(0, std::f64::consts::FRAC_PI_3);
        params.bind(1, std::f64::consts::FRAC_PI_4);

        // Z on q2 → (1, -1)
        let op = make_op(GateKind::Z, &[2], vec![]);
        let (q, d0, d1) = super::diagonal_factor(&op, &params).unwrap().unwrap();
        assert_eq!(q, 2);
        assert!((d0 - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!((d1 - Complex64::new(-1.0, 0.0)).norm() < 1e-12);

        // S on q0 → (1, i)
        let op = make_op(GateKind::S, &[0], vec![]);
        let (_, _, d1) = super::diagonal_factor(&op, &params).unwrap().unwrap();
        assert!((d1 - Complex64::new(0.0, 1.0)).norm() < 1e-12);

        // Sdg → (1, -i)
        let op = make_op(GateKind::Sdg, &[0], vec![]);
        let (_, _, d1) = super::diagonal_factor(&op, &params).unwrap().unwrap();
        assert!((d1 - Complex64::new(0.0, -1.0)).norm() < 1e-12);

        // T → (1, e^{iπ/4})
        let op = make_op(GateKind::T, &[0], vec![]);
        let (_, _, d1) = super::diagonal_factor(&op, &params).unwrap().unwrap();
        let want = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
        assert!((d1 - want).norm() < 1e-12);

        // Tdg → (1, e^{-iπ/4})
        let op = make_op(GateKind::Tdg, &[0], vec![]);
        let (_, _, d1) = super::diagonal_factor(&op, &params).unwrap().unwrap();
        let want = Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4);
        assert!((d1 - want).norm() < 1e-12);

        // Rz(π/3) on q1 → (e^{-iπ/6}, e^{iπ/6})
        let op = make_op(GateKind::Rz, &[1], vec![ParamExpr::Symbol(0)]);
        let (q, d0, d1) = super::diagonal_factor(&op, &params).unwrap().unwrap();
        assert_eq!(q, 1);
        let want_d0 = Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_6);
        let want_d1 = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_6);
        assert!((d0 - want_d0).norm() < 1e-12);
        assert!((d1 - want_d1).norm() < 1e-12);

        // U1(π/4) on q3 → (1, e^{iπ/4})
        let op = make_op(GateKind::U1, &[3], vec![ParamExpr::Symbol(1)]);
        let (q, d0, d1) = super::diagonal_factor(&op, &params).unwrap().unwrap();
        assert_eq!(q, 3);
        assert!((d0 - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!((d1 - Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4)).norm() < 1e-12);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn diagonal_factor_rejects_non_diagonal_gates() {
        // X / Y / H / Rx / Ry / U2 / U3 / CX / CY / CZ / Swap /
        // CRz / CU3 / CCX / CSwap / Id / Barrier / Measure all
        // must classify as None — the fusion walker has to fall
        // through to the per-op `apply_op` path.
        let params = ParameterBinding::new();
        let mut p = ParameterBinding::new();
        p.bind(0, 0.5);

        for gate in &[
            GateKind::X,
            GateKind::Y,
            GateKind::H,
            GateKind::Id,
            GateKind::Barrier,
            GateKind::Measure,
        ] {
            let op = make_op(gate.clone(), &[0], vec![]);
            assert!(
                super::diagonal_factor(&op, &params).unwrap().is_none(),
                "{gate:?} must classify as None"
            );
        }

        // Parametric non-diagonals
        let op = make_op(GateKind::Rx, &[0], vec![ParamExpr::Symbol(0)]);
        assert!(super::diagonal_factor(&op, &p).unwrap().is_none());
        let op = make_op(GateKind::Ry, &[0], vec![ParamExpr::Symbol(0)]);
        assert!(super::diagonal_factor(&op, &p).unwrap().is_none());

        // Two-qubit
        let op = make_op(GateKind::CX, &[0, 1], vec![]);
        assert!(super::diagonal_factor(&op, &params).unwrap().is_none());
        let op = make_op(GateKind::CZ, &[0, 1], vec![]);
        assert!(super::diagonal_factor(&op, &params).unwrap().is_none());

        // Three-qubit
        let op = make_op(GateKind::CCX, &[0, 1, 2], vec![]);
        assert!(super::diagonal_factor(&op, &params).unwrap().is_none());
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn diagonal_factor_rejects_conditional_diagonal() {
        // A conditional diagonal gate (e.g. `if c[0] == 1 then Z`)
        // can't fuse — the walker needs to evaluate the condition
        // per op and skip / dispatch independently.
        let mut op = make_op(GateKind::Z, &[0], vec![]);
        op.condition = Some((0, 1, 1));
        let params = ParameterBinding::new();
        assert!(
            super::diagonal_factor(&op, &params).unwrap().is_none(),
            "conditional diagonal must classify as None even though Z is diagonal"
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_ops_fused_interleaved_diagonal_and_non_diagonal() {
        // Mixed sequence: H q0; Z q1; S q1; Rz(θ) q2; X q0; Z q0;
        // — H and X break the diagonal run; the Z-S-Rz triple in
        // between fuses; the trailing single-Z doesn't (only one
        // factor). The walker must produce the same state as
        // applying each op individually via apply_op.
        let n: u32 = 3;
        let backend = new_backend();

        let mut params = ParameterBinding::new();
        params.bind(0, 0.7);
        let ops = vec![
            make_op(GateKind::H, &[0], vec![]),
            make_op(GateKind::Z, &[1], vec![]),
            make_op(GateKind::S, &[1], vec![]),
            make_op(GateKind::Rz, &[2], vec![ParamExpr::Symbol(0)]),
            make_op(GateKind::X, &[0], vec![]),
            make_op(GateKind::Z, &[0], vec![]),
        ];

        // Reference: per-op apply.
        let want_init = random_state(n, 0xBEEFu64);
        let mut want = backend.allocate(n).unwrap();
        want.write_state(&want_init).unwrap();
        for op in &ops {
            super::apply_op(&want, op, &params).unwrap();
        }

        // Fused: walker.
        let mut got = backend.allocate(n).unwrap();
        got.write_state(&want_init).unwrap();
        super::apply_ops_fused(&got, &ops, &params, |_| false, None).unwrap();

        let max = max_abs_diff(&want.read_state(), &got.read_state());
        assert!(
            max < 1e-5,
            "fused walker diverged from per-op reference: {max}"
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_ops_fused_condition_skip_predicate_bypasses_ops() {
        // The condition_skip closure controls which ops the walker
        // visits. With a closure that skips the middle op, the
        // resulting state must match an explicit two-op application
        // (the kept ops only). Pins that skipped ops don't disturb
        // the pending fusion run either — first kept op is Rz, the
        // skipped one is also Rz (would be fusion-eligible if not
        // skipped), and the trailing op is H (non-fusable). A bug
        // that accidentally batched the skipped factor would give
        // a different state.
        let n: u32 = 3;
        let backend = new_backend();
        let mut params = ParameterBinding::new();
        params.bind(0, 0.4);
        params.bind(1, 0.9); // ← only this op should be skipped
        params.bind(2, 1.3);
        let ops = vec![
            make_op(GateKind::Rz, &[0], vec![ParamExpr::Symbol(0)]),
            make_op(GateKind::Rz, &[1], vec![ParamExpr::Symbol(1)]), // skipped
            make_op(GateKind::H, &[2], vec![]),
        ];

        let init = random_state(n, 0xCAFEu64);

        // Reference: only apply ops 0 and 2.
        let mut want = backend.allocate(n).unwrap();
        want.write_state(&init).unwrap();
        super::apply_op(&want, &ops[0], &params).unwrap();
        super::apply_op(&want, &ops[2], &params).unwrap();

        // Walker with condition_skip on op[1].
        let mut got = backend.allocate(n).unwrap();
        got.write_state(&init).unwrap();
        super::apply_ops_fused(&got, &ops, &params, |op| {
            // Skip the middle Rz only.
            matches!(&op.gate, GateKind::Rz)
                && op
                    .params
                    .first()
                    .map(|p| matches!(p, ParamExpr::Symbol(1)))
                    .unwrap_or(false)
        }, None)
        .unwrap();

        let max = max_abs_diff(&want.read_state(), &got.read_state());
        assert!(max < 1e-5, "condition-skipped walker diverged: {max}");
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_ops_fused_id_and_barrier_dont_break_fusion_run() {
        // Pin that Id / Barrier are transparent to the walker — a
        // sequence `Rz; Id; Rz; Barrier; Rz` should still collapse
        // into ONE fused `apply_diagonal_product` dispatch rather
        // than three. The numerical check uses the per-op apply_op
        // reference (where Id/Barrier are also no-ops); a regression
        // that flushed mid-run would still pass numerically but
        // would emit three separate dispatches. Catching the
        // dispatch-count regression directly would need GPU
        // instrumentation we don't ship, so this test is structural:
        // confirms the result still matches per-op reference, which
        // keeps the optimization honest.
        let n: u32 = 3;
        let backend = new_backend();

        let mut params = ParameterBinding::new();
        params.bind(0, 0.31);
        params.bind(1, 0.62);
        params.bind(2, 0.93);
        let ops = vec![
            make_op(GateKind::Rz, &[0], vec![ParamExpr::Symbol(0)]),
            make_op(GateKind::Id, &[0], vec![]),
            make_op(GateKind::Rz, &[1], vec![ParamExpr::Symbol(1)]),
            make_op(GateKind::Barrier, &[0], vec![]),
            make_op(GateKind::Rz, &[2], vec![ParamExpr::Symbol(2)]),
        ];

        let want_init = random_state(n, 0xFADEDBEEu64);

        // Per-op reference (apply_op handles Id / Barrier as Ok no-op).
        let mut want = backend.allocate(n).unwrap();
        want.write_state(&want_init).unwrap();
        for op in &ops {
            super::apply_op(&want, op, &params).unwrap();
        }

        // Fused walker.
        let mut got = backend.allocate(n).unwrap();
        got.write_state(&want_init).unwrap();
        super::apply_ops_fused(&got, &ops, &params, |_| false, None).unwrap();

        let max = max_abs_diff(&want.read_state(), &got.read_state());
        assert!(
            max < 1e-5,
            "Id/Barrier-transparent fusion diverged from per-op: {max}"
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn apply_ops_fused_empty_op_list_is_noop() {
        let n: u32 = 4;
        let backend = new_backend();
        let mut state = backend.allocate(n).unwrap();
        let want = random_state(n, 0xDEADu64);
        state.write_state(&want).unwrap();
        let params = ParameterBinding::new();
        super::apply_ops_fused(&state, &[], &params, |_| false, None).unwrap();
        let got = state.read_state();
        let max = max_abs_diff(&want, &got);
        assert!(max < 1e-6, "empty op list must not modify state: {max}");
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn copy_into_with_mismatched_num_qubits_returns_typed_error() {
        // The byte-size check guards against writing past dst's
        // allocation when the caller hands a smaller scratch. A typed
        // `StateLengthMismatch` is the contract; pin both directions
        // (src bigger than dst, and src smaller than dst) since either
        // would corrupt the buffer differently.
        let backend = new_backend();

        let src_big = backend.allocate(4).expect("alloc src");
        let dst_small = backend.allocate(3).expect("alloc dst");
        match src_big.copy_into(&dst_small) {
            Err(MetalError::StateLengthMismatch { expected, got }) => {
                assert_eq!(expected, 16);
                assert_eq!(got, 8);
            }
            other => panic!("expected StateLengthMismatch, got {other:?}"),
        }

        let src_small = backend.allocate(3).expect("alloc src");
        let dst_big = backend.allocate(4).expect("alloc dst");
        match src_small.copy_into(&dst_big) {
            Err(MetalError::StateLengthMismatch { expected, got }) => {
                assert_eq!(expected, 8);
                assert_eq!(got, 16);
            }
            other => panic!("expected StateLengthMismatch, got {other:?}"),
        }
    }
}

