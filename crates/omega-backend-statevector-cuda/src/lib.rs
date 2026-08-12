//! CUDA (NVIDIA, Linux/Windows) statevector backend.
//!
//! **Status:** Phase 2, step 1 — `Backend` trait wired. The crate
//! compiles on every platform but only does real work when built with
//! `--features cuda` *and* the target is Linux or Windows. Elsewhere
//! the constructor returns [`CudaError::Unavailable`] so callers can
//! fall back to CPU per the [`omega_core::device::DeviceKind`]
//! contract.
//!
//! ## Architecture
//!
//! Mirrors the Metal backend exactly so the cross-cutting wiring
//! (executor dispatch, `apply_op` walker, fusion + adjoint paths) is
//! identical:
//! - [`CudaStatevectorBackend`] — stateless device handle. Owns the
//!   CUDA context, default stream, and the NVRTC-compiled kernel
//!   library. Cheap to keep around; this is what implements
//!   [`omega_core::executor::Backend`].
//! - [`CudaState`] — per-circuit statevector. Holds one
//!   [`cudarc::driver::CudaSlice<f32>`] (interleaved (re, im) f32) plus
//!   `num_qubits`, plus its own refcounted clones of context/stream/
//!   kernels so it doesn't borrow back to the handle.
//!
//! ## State layout
//!
//! Statevector lives in one f32-complex CudaSlice — interleaved
//! `(re, im, re, im, …)` — matching the Metal backend's memory layout.
//! GPUs are dramatically faster at f32 than f64 and parameter sweeps
//! train within the resulting precision. The CPU backend stays f64;
//! cross-checks tolerate ~1e-6 round-off (Phase 1 Metal validated the
//! same threshold against the 386-fixture verify-qiskit corpus).

// Trajectory RNG for `Reset` — only reachable on a real CUDA build.
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
use rand::rngs::StdRng;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
use rand::{RngExt, SeedableRng};
use thiserror::Error;

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
mod adjoint;
// `forward_graph` lives behind `allow(dead_code)` because the only
// in-tree consumers right now are the lib's `tests` module — the
// eventual production consumer is the next-slice
// `expectation_multi_then_gradient_graph` override.
#[allow(dead_code)]
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
mod forward_graph;
// `backward_graph` (TrainStepGraph) — captures forward + ν init +
// adjoint backward into one CudaGraph. Hot-path wiring lands in D4
// (expectation_multi_then_gradient_via_graph). Until then only the
// lib's tests module exercises it.
#[allow(dead_code)]
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
mod backward_graph;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
pub mod f64_path;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
mod imp;
// `pub` so the dual-precision compile test can reach `Precision` and the
// source list; still gated, because the module uses cudarc.
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
pub mod kernels;

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
use num_complex::Complex64;

use omega_core::circuit::CircuitIR;
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
use omega_core::circuit::GateKind;
use omega_core::circuit::SymbolId;
use omega_core::error::{OmegaError, Result as OmegaResult};
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
use omega_core::executor::MidCircuitMode;
use omega_core::executor::{
    Backend, ExecConfig, ExecResult, ExpectationsAndGradient, GradientObservableFactory, Observable,
};
use omega_core::params::ParameterBinding;

/// Errors specific to the CUDA backend. Map to
/// [`omega_core::error::OmegaError::Backend`] when surfaced through the
/// `Backend` trait.
#[derive(Debug, Error)]
pub enum CudaError {
    /// Build was missing `--features cuda` or the target isn't
    /// Linux/Windows. Callers should fall back to the CPU backend.
    #[error("CUDA backend unavailable: {0}")]
    Unavailable(&'static str),
    /// Requested `num_qubits` exceeds the per-allocation cap or the
    /// resulting buffer would be larger than the conservative ceiling.
    #[error("CUDA allocation refused: {reason} (num_qubits={num_qubits})")]
    AllocationRefused {
        num_qubits: u32,
        reason: &'static str,
    },
    /// Caller passed a state slice whose length doesn't match
    /// `2^num_qubits`.
    #[error("state length mismatch: expected {expected}, got {got}")]
    StateLengthMismatch { expected: usize, got: usize },
    /// NVRTC kernel compile or module load failed. Shouldn't happen
    /// for the kernels we ship; if it does the CUDA toolchain is
    /// mis-configured or the GPU is incompatible.
    #[error("CUDA kernel `{kernel}` failed to compile: {reason}")]
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
    /// Device returned `CUDA_ERROR_OUT_OF_MEMORY`. Distinguished from
    /// the generic `Driver(_)` variant so the trainer-level fallback
    /// in `QmlTrainer.fit` can switch to CPU mid-run instead of
    /// failing the whole training loop. See `imp::DeviceHandle::allocate`
    /// for the cudarc `DriverError` → `OutOfMemory` mapping.
    #[error("CUDA out of memory at num_qubits={num_qubits}: {reason}")]
    OutOfMemory { num_qubits: u32, reason: String },
    /// Generic CUDA driver / runtime error.
    #[error("CUDA driver error: {0}")]
    Driver(String),
}

impl From<CudaError> for OmegaError {
    fn from(e: CudaError) -> Self {
        match e {
            CudaError::OutOfMemory { .. } => OmegaError::OutOfMemory(e.to_string()),
            other => OmegaError::Backend(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------
// Public API surfaces
// ---------------------------------------------------------------------

/// Stateless CUDA device handle.
///
/// Construct once with [`CudaStatevectorBackend::new`]; allocate
/// per-circuit state via [`Self::allocate`]. Implements
/// [`omega_core::executor::Backend`] so it slots into the same
/// dispatch the CPU backend uses.
pub struct CudaStatevectorBackend {
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    handle: imp::DeviceHandle,
    /// Single-entry cache: the most-recently-built TrainStepGraph
    /// keyed by (circuit shape, gradient-obs sign masks). The QML
    /// trainer hits the same shape on every training point so a
    /// one-entry cache is enough for hot-path amortization.
    /// Populated lazily on the first call to
    /// `expectation_multi_then_gradient` whose circuit + observable
    /// are graph-compatible (Z-only obs, supported gate set).
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    train_step_graph_cache: std::sync::Mutex<Option<(u64, backward_graph::TrainStepGraph)>>,
    #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
    _private: (),
}

/// Per-circuit statevector held in a `CudaSlice<f32>`. Created by
/// [`CudaStatevectorBackend::allocate`].
pub struct CudaState {
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub(crate) inner: imp::StateBuffer,
    #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
    _private: (),
}

impl CudaStatevectorBackend {
    /// Open device 0 + the default stream and NVRTC-compile the
    /// kernel library. On a build that doesn't include the CUDA
    /// runtime, returns `Err(CudaError::Unavailable)` rather than
    /// panicking.
    pub fn new() -> Result<Self, CudaError> {
        #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
        {
            let handle = imp::DeviceHandle::new()?;
            Ok(Self {
                handle,
                train_step_graph_cache: std::sync::Mutex::new(None),
            })
        }
        #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
        {
            Err(CudaError::Unavailable(
                "rebuild on Linux/Windows with --features cuda",
            ))
        }
    }

    /// Allocate a fresh statevector for `num_qubits` qubits, initialised
    /// to `|0…0⟩`. Refuses allocations above
    /// [`CudaStatevectorBackend::MAX_QUBITS`].
    pub fn allocate(&self, num_qubits: u32) -> Result<CudaState, CudaError> {
        #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
        {
            let inner = self.handle.allocate(num_qubits)?;
            Ok(CudaState { inner })
        }
        #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
        {
            let _ = num_qubits;
            Err(CudaError::Unavailable(
                "rebuild on Linux/Windows with --features cuda",
            ))
        }
    }

    /// Static availability check. Mirrors
    /// [`omega_core::device::DeviceKind::is_available`] for the CUDA
    /// branch.
    pub const fn is_available() -> bool {
        cfg!(all(
            any(target_os = "linux", target_os = "windows"),
            feature = "cuda"
        ))
    }

    /// Non-graph fallback for the residual-gradient path. Used by
    /// `expectation_multi_then_gradient` when we've already consumed
    /// the FnOnce factory (to extract y_labels) but the
    /// `TrainStepGraph` capture refused (e.g. unsupported gate
    /// kind). Reconstructs the gradient observable from
    /// `(measurement qubits implicit in observables, y_labels)`,
    /// runs forward + per-Z predictions outside the graph, then
    /// runs `adjoint_gradient_with_forward_state` for gradients.
    /// Defensive — practically unreachable on QmlTrainer ansatzes
    /// the rest of the suite supports.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    fn expectation_multi_then_gradient_residual_fallback(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
        y_labels: &[f32],
    ) -> OmegaResult<ExpectationsAndGradient> {
        use omega_core::executor::PauliOp;
        let n = circuit.num_qubits;
        let inner = self
            .handle
            .allocate(n)
            .map_err(|e| OmegaError::Backend(format!("cuda alloc psi: {e}")))?;
        let mut psi = CudaState { inner };
        apply_ops_fused(
            &mut psi,
            &circuit.ops,
            params,
            |op| matches!(&op.gate, GateKind::Measure),
            None,
        )?;
        let mut predictions = Vec::with_capacity(observables.len());
        for obs in observables {
            let mut total = 0.0_f64;
            for (coeff, pauli_string) in &obs.terms {
                let (x_mask, sign_mask, y_factor) = pauli_masks(pauli_string);
                let ip = psi.pauli_expectation(x_mask, sign_mask, y_factor)?;
                total += coeff * ip.re;
            }
            predictions.push(total);
        }
        // Reconstruct the residual gradient observable from
        // y_labels: coeff[i] = 2 * (predictions[i] - y_labels[i]).
        let mut terms = Vec::with_capacity(observables.len());
        for (i, obs) in observables.iter().enumerate() {
            let q = obs.terms[0].1[0].0;
            let coeff = 2.0 * (predictions[i] - y_labels[i] as f64);
            if coeff != 0.0 {
                terms.push((coeff, vec![(q, PauliOp::Z)]));
            }
        }
        let gradient_obs = Observable { terms };
        let gradient = adjoint::adjoint_gradient_with_forward_state(
            &self.handle,
            circuit,
            params,
            &gradient_obs,
            psi,
        )?;
        Ok((predictions, gradient))
    }

    /// Hard cap on `num_qubits` for a single allocation. Above this the
    /// statevector would exceed 2 GiB on f32-complex; we'd rather
    /// refuse cleanly than thrash. Mirrors Metal's MAX_QUBITS — kernel
    /// indexing assumes 32-bit masks (popcount on u32) so 28 is the
    /// hard ceiling regardless of host VRAM.
    pub const MAX_QUBITS: u32 = 28;
}

impl CudaState {
    pub fn num_qubits(&self) -> u32 {
        #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
        {
            self.inner.num_qubits
        }
        #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
        {
            0
        }
    }

    /// Read the statevector back to host-side `Complex64`. Crosses
    /// the f32→f64 boundary; values match the buffer to within f32
    /// precision.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn read_state(&self) -> Result<Vec<Complex64>, CudaError> {
        self.inner.read_state()
    }

    /// Process-global count of `CudaState::read_state` calls since
    /// the program started. Used by the no-host-syncs regression
    /// test (`tests/qml_no_host_syncs.rs`): read this before and
    /// after one training epoch; the delta must be zero because
    /// the QML hot path's gradient observable is diagonal-Z, so the
    /// adjoint backward sweep takes the GPU-only
    /// `apply_diagonal_pauli_sum` path rather than the host
    /// fallback. Mirrors `MetalState::read_state_call_count`.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn read_state_call_count() -> u64 {
        imp::READ_STATE_CALL_COUNT.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Overwrite the statevector with a host-side `Complex64` slice.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn write_state(&mut self, state: &[Complex64]) -> Result<(), CudaError> {
        self.inner.write_state(state)
    }

    /// Compute ⟨self|other⟩ on the GPU.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn inner_product(&self, other: &CudaState) -> Result<Complex64, CudaError> {
        self.inner.inner_product(&other.inner)
    }

    /// Fused Pauli-string expectation `⟨ψ|P|ψ⟩` in one dispatch.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn pauli_expectation(
        &self,
        x_mask: u32,
        sign_mask: u32,
        y_factor: Complex64,
    ) -> Result<Complex64, CudaError> {
        self.inner.pauli_expectation(x_mask, sign_mask, y_factor)
    }

    /// Apply a diagonal Pauli-sum observable `O = Σ_k c_k · Z^{(s_k)}`
    /// to `self`, writing `O|ψ⟩` into `dst`.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_diagonal_pauli_sum(
        &self,
        dst: &mut CudaState,
        terms: &[(u32, f32)],
    ) -> Result<(), CudaError> {
        self.inner.apply_diagonal_pauli_sum(&mut dst.inner, terms)
    }

    /// Apply N independent diagonal 1q gates as a single fused
    /// dispatch.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_diagonal_product(
        &mut self,
        factors: &[(u32, Complex64, Complex64)],
    ) -> Result<(), CudaError> {
        self.inner.apply_diagonal_product(factors)
    }

    /// Overwrite `dst` with this state's amplitudes.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn copy_into(&self, dst: &mut CudaState) -> Result<(), CudaError> {
        self.inner.copy_into(&mut dst.inner)
    }

    // ---- diagonal / 1q / 2q kernel wrappers --------------------------

    /// Apply a diagonal single-qubit gate `diag(d0, d1)` to `qubit`.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_diagonal(
        &mut self,
        qubit: u32,
        d0: Complex64,
        d1: Complex64,
    ) -> Result<(), CudaError> {
        self.inner.apply_diagonal(qubit, d0, d1)
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_z(&mut self, qubit: u32) -> Result<(), CudaError> {
        self.apply_diagonal(qubit, Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0))
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_s(&mut self, qubit: u32) -> Result<(), CudaError> {
        self.apply_diagonal(qubit, Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0))
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_sdg(&mut self, qubit: u32) -> Result<(), CudaError> {
        self.apply_diagonal(qubit, Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0))
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_t(&mut self, qubit: u32) -> Result<(), CudaError> {
        self.apply_diagonal(
            qubit,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
        )
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_tdg(&mut self, qubit: u32) -> Result<(), CudaError> {
        self.apply_diagonal(
            qubit,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4),
        )
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_rz(&mut self, qubit: u32, theta: f64) -> Result<(), CudaError> {
        self.apply_diagonal(
            qubit,
            Complex64::from_polar(1.0, -theta / 2.0),
            Complex64::from_polar(1.0, theta / 2.0),
        )
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_u1(&mut self, qubit: u32, lambda: f64) -> Result<(), CudaError> {
        self.apply_diagonal(
            qubit,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, lambda),
        )
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_1q(&mut self, qubit: u32, u: &[Complex64; 4]) -> Result<(), CudaError> {
        self.inner.apply_1q(qubit, u)
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_h(&mut self, qubit: u32) -> Result<(), CudaError> {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let u = [
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(-s, 0.0),
        ];
        self.apply_1q(qubit, &u)
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_x(&mut self, qubit: u32) -> Result<(), CudaError> {
        let u = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        self.apply_1q(qubit, &u)
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_y(&mut self, qubit: u32) -> Result<(), CudaError> {
        let u = [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ];
        self.apply_1q(qubit, &u)
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_rx(&mut self, qubit: u32, theta: f64) -> Result<(), CudaError> {
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

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_ry(&mut self, qubit: u32, theta: f64) -> Result<(), CudaError> {
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

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_u3(
        &mut self,
        qubit: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
    ) -> Result<(), CudaError> {
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

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_2q(&mut self, qa: u32, qb: u32, u: &[Complex64; 16]) -> Result<(), CudaError> {
        self.inner.apply_2q(qa, qb, u)
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_cx(&mut self, qc: u32, qt: u32) -> Result<(), CudaError> {
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

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_cy(&mut self, qc: u32, qt: u32) -> Result<(), CudaError> {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let pi = Complex64::new(0.0, 1.0);
        let mi = Complex64::new(0.0, -1.0);
        #[rustfmt::skip]
        let u = [
            o, z, z, z,
            z, z, z, mi,
            z, z, o, z,
            z, pi, z, z,
        ];
        self.apply_2q(qc, qt, &u)
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_cz(&mut self, qa: u32, qb: u32) -> Result<(), CudaError> {
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

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_swap(&mut self, qa: u32, qb: u32) -> Result<(), CudaError> {
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

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_rbs(&mut self, qa: u32, qb: u32, theta: f64) -> Result<(), CudaError> {
        // Reconfigurable Beam Splitter / Givens rotation: identity on
        // {|00>, |11>}, a real 2x2 rotation on span{|01>, |10>}.
        //
        // `apply_2q`'s basis order swaps indices 1<->2 relative to the CPU
        // `gates::rbs` builder (this is exactly the `perm_2q_to_cuda`
        // permutation [0,2,1,3] the adjoint applies to CPU 2q matrices). RBS is
        // sign-antisymmetric on the off-diagonal block, so under that swap
        // `gates::rbs(theta)` becomes the block below — the +sin/-sin sit at
        // [1][2]/[2][1] rather than the CPU [1][2]=-sin/[2][1]=+sin. Verified
        // numerically against the CPU statevector by
        // `gpu_cuda_agrees_with_sim_on_rbs`.
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

    /// Reset qubit `q` to |0⟩ — the reset **channel** ρ → |0⟩⟨0|_q ⊗ Tr_q(ρ),
    /// matching the CPU `apply_reset`.
    ///
    /// **Sample → project → flip**, with the Born probability computed on the
    /// device. `u` is a uniform draw in `[0,1)` supplied by the caller's
    /// trajectory RNG, so the randomness stays with the shot loop that owns it
    /// (the GPU state object is not the right place for an RNG).
    ///
    /// `p0 = P(q reads 0)` comes from the existing fused Pauli-expectation
    /// reduction: `⟨Z_q⟩ = p0 − p1` and `p0 + p1 = 1`, so
    /// `p0 = (1 + ⟨Z_q⟩)/2` — Pauli masks `(x_mask = 0, sign_mask = 1<<q,
    /// y_factor = 1)`. No new kernel, and the only host transfer is that one
    /// scalar, which the previous implementation already paid for its norm
    /// readback.
    ///
    /// Projection, renormalisation and the conditional X then fuse into a
    /// *single* `apply_1q`, since `new0 = u00·a0 + u01·a1`, `new1 = u10·a0 +
    /// u11·a1`:
    ///
    /// * outcome 0 → `[[1/√p0, 0], [0, 0]]`  (keep |0⟩, renormalise)
    /// * outcome 1 → `[[0, 1/√p1], [0, 0]]`  (keep |1⟩, renormalise, and move
    ///   it into the |0⟩ slot — that *is* the conditional X)
    ///
    /// **Do not "fold" the amplitudes.** The previous implementation applied
    /// `[[1,1],[0,0]]` and renormalised, mirroring the CPU backend's old bug:
    /// that is a *coherent* fold, which turns entanglement into a fake
    /// superposition (⟨X₁⟩ = +1 after Bell + reset instead of 0) and
    /// annihilates a |−⟩ state. See `omega-backend-statevector`'s `apply_reset`
    /// and `tests/reset_channel.rs`.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_reset(&mut self, q: u32, u: f64) -> Result<(), CudaError> {
        let p0 = self.reset_p0(q)?;
        self.apply_reset_outcome(q, p0, u >= p0)
    }

    /// `P(qubit q reads 0)`, computed on the device from `⟨Z_q⟩`.
    ///
    /// Callers use this to decide the reset branch, and analytic (`shots =
    /// None`) callers use it to detect a *deterministic* reset (`p0` is 0 or 1
    /// ⇒ the qubit sits in a Z eigenstate) — the only case where a reset has a
    /// well-defined pure-state answer.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn reset_p0(&mut self, q: u32) -> Result<f64, CudaError> {
        let one = Complex64::new(1.0, 0.0);
        // ⟨Z_q⟩ = p0 − p1 and p0 + p1 = 1 ⇒ p0 = (1 + ⟨Z_q⟩)/2. Clamp: the f32
        // reduction can land a hair outside [0,1] and a negative under the
        // sqrt below would poison the state.
        let z_q = self.pauli_expectation(0, 1u32 << q, one)?.re;
        Ok((0.5 * (1.0 + z_q)).clamp(0.0, 1.0))
    }

    /// Collapse `q` onto `outcome_one`, renormalise, and move the surviving
    /// amplitude into the |0⟩ slot — the project/renormalise/conditional-X of
    /// the reset channel, fused into one `apply_1q`.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_reset_outcome(
        &mut self,
        q: u32,
        p0: f64,
        outcome_one: bool,
    ) -> Result<(), CudaError> {
        let zero = Complex64::new(0.0, 0.0);
        let p1 = 1.0 - p0;
        // A zero-weight branch is unreachable; force the other outcome so the
        // renormalisation stays finite (matches the CPU backend).
        let outcome_one = if p0 <= 0.0 {
            true
        } else if p1 <= 0.0 {
            false
        } else {
            outcome_one
        };
        let inv = Complex64::new(1.0 / if outcome_one { p1 } else { p0 }.sqrt(), 0.0);
        let m = if outcome_one {
            [zero, inv, zero, zero]
        } else {
            [inv, zero, zero, zero]
        };
        self.apply_1q(q, &m)
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_crz(&mut self, qc: u32, qt: u32, theta: f64) -> Result<(), CudaError> {
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

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_ccx(&mut self, qc1: u32, qc2: u32, qt: u32) -> Result<(), CudaError> {
        // Same Nielsen-Chuang decomposition Metal uses; identical
        // sequence so f32 round-off is bit-equivalent.
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

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_cswap(&mut self, qc: u32, qt1: u32, qt2: u32) -> Result<(), CudaError> {
        self.apply_cx(qt2, qt1)?;
        self.apply_ccx(qc, qt1, qt2)?;
        self.apply_cx(qt2, qt1)?;
        Ok(())
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    pub fn apply_cu3(
        &mut self,
        qc: u32,
        qt: u32,
        theta: f64,
        phi: f64,
        lambda: f64,
    ) -> Result<(), CudaError> {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let cu_00 = Complex64::new(c, 0.0);
        let cu_01 = -Complex64::from_polar(s, lambda);
        let cu_10 = Complex64::from_polar(s, phi);
        let cu_11 = Complex64::from_polar(c, phi + lambda);
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
// Backend trait — wires the CUDA handle into the workspace dispatch
// ---------------------------------------------------------------------

impl Backend for CudaStatevectorBackend {
    fn name(&self) -> &str {
        "cuda-statevector"
    }

    fn device(&self) -> omega_core::device::DeviceKind {
        omega_core::device::DeviceKind::Cuda
    }

    /// CPU rescue path for `QmlTrainer::fit` when the CUDA allocator
    /// refuses at n ≥ 22 on memory-tight devices. Mirrors the
    /// Metal backend's override — returns the workspace's CPU
    /// `StatevectorBackend`, which the trainer uses for the
    /// remainder of training after emitting one stderr notice.
    fn cpu_fallback(&self) -> Option<Box<dyn omega_core::executor::Backend>> {
        Some(Box::new(
            omega_backend_statevector::StatevectorBackend::new(),
        ))
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    fn execute(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        config: &ExecConfig,
    ) -> OmegaResult<ExecResult> {
        let n = circuit.num_qubits;
        let mut state = self.allocate(n)?;

        // Reset is handled in-sequence by `apply_ops_fused` (deterministic
        // fold + renormalise, matching the CPU backend). Mid-circuit
        // measurement with collapse still isn't supported on GPU — refuse so
        // the CLI dispatcher can fall back to CPU.
        for op in &circuit.ops {
            if let GateKind::Measure = &op.gate {
                if config.mid_circuit_mode == MidCircuitMode::Collapse {
                    return Err(OmegaError::Unsupported(
                        "cuda: mid-circuit measurement not yet implemented".into(),
                    ));
                }
            }
        }

        let classical_bits = vec![0u8; circuit.num_classical_bits as usize];

        // `Reset` is a stochastic CHANNEL (see `CudaState::apply_reset`): one
        // statevector carries one trajectory, so correct shot statistics need
        // independent evolutions. Sampling every shot from a single post-reset
        // state would replay one draw of the reset outcome as certainty.
        // Mirrors the CPU statevector backend. The device buffer is reused
        // across shots and re-zeroed via `write_state`.
        let has_reset = circuit
            .ops
            .iter()
            .any(|op| matches!(op.gate, GateKind::Reset));
        if let (true, Some(shots)) = (has_reset, config.shots) {
            let mut rng = match config.seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => rand::make_rng::<StdRng>(),
            };
            let dim = 1usize << n;
            let mut zero = vec![Complex64::new(0.0, 0.0); dim];
            zero[0] = Complex64::new(1.0, 0.0);
            let mut counts: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
            for _ in 0..shots {
                state.write_state(&zero)?;
                apply_ops_fused(
                    &mut state,
                    &circuit.ops,
                    params,
                    |op| !op.condition_satisfied(&classical_bits),
                    Some(&mut rng),
                )?;
                let one = state.inner.sample_counts_on_device(1, Some(rng.random()))?;
                if let Some(k) = one.into_keys().next() {
                    *counts.entry(k).or_insert(0) += 1;
                }
            }
            return Ok(ExecResult::Counts(counts));
        }

        apply_ops_fused(
            &mut state,
            &circuit.ops,
            params,
            |op| {
                // Skip the gate when its classical condition is NOT
                // satisfied. `GateOp::condition_satisfied` handles both
                // single-bit `if(c == V)` and multi-bit cregs (post-merge
                // `Option<(start_bit, num_bits, expected)>` shape).
                !op.condition_satisfied(&classical_bits)
            },
            None,
        )?;

        // Shot mode skips the full statevector readback — only the
        // per-amplitude probabilities (one f32 per amp = half the
        // bytes) cross the PCIe link. Statevector mode still needs
        // the full complex64 amps.
        match config.shots {
            None => {
                let amps = state.read_state()?;
                Ok(ExecResult::Statevector(amps))
            }
            Some(shots) => {
                // Fully on-device sampler — recursive multi-block CDF
                // scan + CURAND uniforms + per-shot binary search.
                // Handles any n the statevector itself fits at.
                let counts = state.inner.sample_counts_on_device(shots, config.seed)?;
                Ok(ExecResult::Counts(counts))
            }
        }
    }

    #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
    fn execute(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _config: &ExecConfig,
    ) -> OmegaResult<ExecResult> {
        Err(OmegaError::Backend(
            "cuda backend not available on this build (rebuild on Linux/Windows with --features cuda)".into(),
        ))
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    fn expectation(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> OmegaResult<f64> {
        let n = circuit.num_qubits;
        let mut psi = self.allocate(n)?;
        // Reset is applied in-sequence by `apply_ops_fused` (deterministic).
        apply_ops_fused(
            &mut psi,
            &circuit.ops,
            params,
            |op| matches!(&op.gate, GateKind::Measure),
            None,
        )?;

        let mut total = 0.0_f64;
        for (coeff, pauli_string) in &observable.terms {
            let (x_mask, sign_mask, y_factor) = pauli_masks(pauli_string);
            let ip = psi.pauli_expectation(x_mask, sign_mask, y_factor)?;
            total += coeff * ip.re;
        }
        Ok(total)
    }

    #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
    fn expectation(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _observable: &Observable,
    ) -> OmegaResult<f64> {
        Err(OmegaError::Backend(
            "cuda backend not available on this build (rebuild on Linux/Windows with --features cuda)".into(),
        ))
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    fn expectation_multi(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
    ) -> OmegaResult<Vec<f64>> {
        if observables.is_empty() {
            return Ok(Vec::new());
        }

        let n = circuit.num_qubits;
        let mut psi = self.allocate(n)?;
        // Reset is applied in-sequence by `apply_ops_fused` (deterministic).
        apply_ops_fused(
            &mut psi,
            &circuit.ops,
            params,
            |op| matches!(&op.gate, GateKind::Measure),
            None,
        )?;

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

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    fn adjoint_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observable: &Observable,
    ) -> OmegaResult<Option<Vec<(SymbolId, f64)>>> {
        // Circuits with Reset are non-unitary — no adjoint. Decline (Ok(None))
        // so the runtime falls back to parameter-shift, which runs the
        // (now reset-capable) forward `expectation`. Mirrors the CPU backend.
        if circuit
            .ops
            .iter()
            .any(|op| matches!(op.gate, GateKind::Reset))
        {
            return Ok(None);
        }
        adjoint::adjoint_gradient(&self.handle, circuit, params, observable)
    }

    #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
    fn adjoint_gradient(
        &self,
        _circuit: &CircuitIR,
        _params: &ParameterBinding,
        _observable: &Observable,
    ) -> OmegaResult<Option<Vec<(SymbolId, f64)>>> {
        Ok(None)
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    fn expectation_multi_then_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
        gradient_observable_factory: GradientObservableFactory<'_>,
    ) -> OmegaResult<ExpectationsAndGradient> {
        if circuit
            .ops
            .iter()
            .any(|op| matches!(&op.gate, GateKind::Reset))
        {
            return Err(OmegaError::Unsupported(
                "cuda expectation_multi_then_gradient: Reset not supported".into(),
            ));
        }
        if circuit
            .ops
            .iter()
            .any(|op| matches!(&op.gate, GateKind::Measure))
        {
            // Forward sweep silently skips Measure (matches expectation_multi);
            // adjoint requires fully-unitary ops. Fall back to the default
            // split path (expectation_multi then adjoint_gradient).
            let predictions = self.expectation_multi(circuit, params, observables)?;
            let obs = gradient_observable_factory(&predictions);
            let gradient = self.adjoint_gradient(circuit, params, &obs)?;
            return Ok((predictions, gradient));
        }

        // Stage E fast path: when the prediction `observables` are
        // each a single-Z term (the QML trainer's residual-gradient
        // shape) and the circuit is graph-compatible, use the
        // captured `TrainStepGraph` for forward + per-Z predictions
        // + host_func coeff derivation + ν init + backward —
        // entirely on the GPU side, with one cuGraphLaunch replay
        // instead of ~280 cuLaunchKernel calls plus a redundant
        // forward sweep for predictions.
        //
        // Precondition (cheap, decided on `observables` shape so we
        // don't consume the FnOnce factory if we'll fall back):
        // every prediction observable must be a single Z on a
        // single qubit with coefficient 1.0. The QmlTrainer
        // residual-gradient hot path satisfies this trivially —
        // `observables[i] = Z_q_i`.
        let trainer_residual_z = observables
            .iter()
            .all(|obs| obs.terms.len() == 1 && is_single_z_unit_coeff(&obs.terms[0]));
        if trainer_residual_z && !observables.is_empty() {
            // Extract y_labels by calling factory with zero
            // predictions. For a residual-shaped factory
            //   coeffs[i] = 2 · (y_hat[i] - y_label[i])
            // ⇒ factory(zeros)[i] = -2 · y_label[i]
            // ⇒ y_label[i] = -factory(zeros)[i] / 2.
            //
            // This consumes the FnOnce factory; we commit to the
            // graph path from this point. If the factory output
            // doesn't match the residual shape (e.g. terms are
            // missing because their coeff was exactly zero, or
            // term order doesn't match observables), we fall back
            // gracefully by detecting the mismatch and routing
            // through the non-graph path with a fresh
            // adjoint_gradient call.
            let zeros = vec![0.0_f64; observables.len()];
            let gradient_obs_zero = gradient_observable_factory(&zeros);
            // Build a y_label_per_qubit map keyed by qubit. The
            // factory may DROP terms whose coefficient is exactly
            // 0 (the QML trainer does this — see
            // `omega-core::qml::QmlTrainer::fit`). A missing term
            // just means coeff = 0 ⇒ y_label = 0 for that
            // observable's qubit. Default the y_labels Vec to 0
            // and only override for terms that show up in the
            // factory output.
            let mut y_labels = vec![0.0_f32; observables.len()];
            let mut residual_shape_ok = true;
            'outer: for (coeff, pauli_string) in &gradient_obs_zero.terms {
                if pauli_string.len() != 1 {
                    residual_shape_ok = false;
                    break;
                }
                let (q, ref op) = pauli_string[0];
                if !matches!(op, omega_core::executor::PauliOp::Z) {
                    residual_shape_ok = false;
                    break;
                }
                // Find the matching observable index by qubit.
                for (i, obs) in observables.iter().enumerate() {
                    if obs.terms.len() == 1 && obs.terms[0].1.len() == 1 && obs.terms[0].1[0].0 == q
                    {
                        y_labels[i] = (-coeff / 2.0) as f32;
                        continue 'outer;
                    }
                }
                // Term has a qubit not in observables — factory
                // output doesn't match the prediction set.
                residual_shape_ok = false;
                break;
            }
            if !residual_shape_ok {
                return Err(OmegaError::Unsupported(
                    "cuda expectation_multi_then_gradient: gradient observable factory \
                     output isn't single-Z residual-shaped; cannot use graph and factory \
                     is already consumed"
                        .into(),
                ));
            }

            // Build a synthetic gradient observable with exactly
            // `observables.len()` terms (one Z per measurement
            // qubit, placeholder coeff 1.0). This is what the
            // captured graph's `n_outputs` is sized to, regardless
            // of which terms the factory dropped — the per-replay
            // host_func computes the actual coeffs from
            // `(predictions, y_labels)` and writes them into the
            // graph's coeffs pool.
            let synthetic_grad_obs = Observable {
                terms: observables
                    .iter()
                    .map(|obs| {
                        let q = obs.terms[0].1[0].0;
                        (1.0_f64, vec![(q, omega_core::executor::PauliOp::Z)])
                    })
                    .collect(),
            };
            let key = train_step_graph_key(circuit, &synthetic_grad_obs);
            let mut cache = self
                .train_step_graph_cache
                .lock()
                .expect("train_step_graph_cache poisoned");
            let needs_build = match cache.as_ref() {
                Some((cached_key, _)) => *cached_key != key,
                None => true,
            };
            if needs_build {
                match backward_graph::TrainStepGraph::capture(
                    &self.handle,
                    circuit,
                    &synthetic_grad_obs,
                ) {
                    Ok(g) => {
                        *cache = Some((key, g));
                    }
                    Err(_) => {
                        // Capture refused (unsupported gate).
                        // Fall back to the residual-fallback path
                        // that reconstructs the gradient_obs from
                        // y_labels.
                        drop(cache);
                        return self.expectation_multi_then_gradient_residual_fallback(
                            circuit,
                            params,
                            observables,
                            &y_labels,
                        );
                    }
                }
            }
            let graph = &mut cache.as_mut().expect("cache populated above").1;
            let (predictions, gradient) = graph.replay_with_y_labels(circuit, params, &y_labels)?;
            return Ok((predictions, Some(gradient)));
        }

        // observables aren't single-Z residual shape — generic
        // non-graph path via factory(predictions).
        let n = circuit.num_qubits;
        let inner = self
            .handle
            .allocate(n)
            .map_err(|e| OmegaError::Backend(format!("cuda alloc psi: {e}")))?;
        let mut psi = CudaState { inner };
        apply_ops_fused(
            &mut psi,
            &circuit.ops,
            params,
            |op| matches!(&op.gate, GateKind::Measure),
            None,
        )?;

        let mut predictions = Vec::with_capacity(observables.len());
        for obs in observables {
            let mut total = 0.0_f64;
            for (coeff, pauli_string) in &obs.terms {
                let (x_mask, sign_mask, y_factor) = pauli_masks(pauli_string);
                let ip = psi.pauli_expectation(x_mask, sign_mask, y_factor)?;
                total += coeff * ip.re;
            }
            predictions.push(total);
        }

        let gradient_obs = gradient_observable_factory(&predictions);
        let gradient = adjoint::adjoint_gradient_with_forward_state(
            &self.handle,
            circuit,
            params,
            &gradient_obs,
            psi,
        )?;

        Ok((predictions, gradient))
    }

    #[cfg(not(all(any(target_os = "linux", target_os = "windows"), feature = "cuda")))]
    fn expectation_multi_then_gradient(
        &self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        observables: &[Observable],
        gradient_observable_factory: GradientObservableFactory<'_>,
    ) -> OmegaResult<ExpectationsAndGradient> {
        // Non-CUDA build: emulate the default trait body explicitly so
        // the fallback path stays predictable.
        let predictions = self.expectation_multi(circuit, params, observables)?;
        let obs = gradient_observable_factory(&predictions);
        let gradient = self.adjoint_gradient(circuit, params, &obs)?;
        Ok((predictions, gradient))
    }
}

// ---------------------------------------------------------------------
// Helpers — fusion walker, op dispatch, sampling, pauli masks
// ---------------------------------------------------------------------

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
pub(crate) fn apply_op(
    state: &mut CudaState,
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

    let res: Result<(), CudaError> = match &op.gate {
        GateKind::Id | GateKind::Barrier => Ok(()),
        GateKind::Measure => Ok(()),

        GateKind::H => state.apply_h(q0()),
        GateKind::X => state.apply_x(q0()),
        GateKind::Y => state.apply_y(q0()),
        GateKind::Z => state.apply_z(q0()),
        GateKind::S => state.apply_s(q0()),
        GateKind::Sdg => state.apply_sdg(q0()),
        // √X / √X† are non-diagonal Cliffords; apply the exact Qiskit matrix
        // (the CPU sim uses the same `gates::sx`/`sxdg`), not the U3 alias that
        // differs by a global e^{iπ/4}.
        GateKind::Sx => state.apply_1q(q0(), &omega_backend_statevector::gates::sx()),
        GateKind::Sxdg => state.apply_1q(q0(), &omega_backend_statevector::gates::sxdg()),
        GateKind::T => state.apply_t(q0()),
        GateKind::Tdg => state.apply_tdg(q0()),

        GateKind::Rx => state.apply_rx(q0(), resolved[0]),
        GateKind::Ry => state.apply_ry(q0(), resolved[0]),
        GateKind::Rz => state.apply_rz(q0(), resolved[0]),
        GateKind::U1 => state.apply_u1(q0(), resolved[0]),
        GateKind::U2 => state.apply_u3(q0(), std::f64::consts::FRAC_PI_2, resolved[0], resolved[1]),
        GateKind::U3 => state.apply_u3(q0(), resolved[0], resolved[1], resolved[2]),

        GateKind::CX => state.apply_cx(q0(), q1()),
        GateKind::CY => state.apply_cy(q0(), q1()),
        GateKind::CZ => state.apply_cz(q0(), q1()),
        GateKind::Swap => state.apply_swap(q0(), q1()),
        GateKind::CRz => state.apply_crz(q0(), q1(), resolved[0]),
        GateKind::CU3 => state.apply_cu3(q0(), q1(), resolved[0], resolved[1], resolved[2]),
        GateKind::Rbs => state.apply_rbs(q0(), q1(), resolved[0]),

        GateKind::CCX => state.apply_ccx(q0(), q1(), op.qubits[2].0),
        GateKind::CSwap => state.apply_cswap(q0(), q1(), op.qubits[2].0),

        GateKind::Reset => {
            return Err(OmegaError::Unsupported(format!(
                "cuda: {:?} should have been filtered before apply_op",
                op.gate
            )));
        }

        GateKind::PhaseShifter | GateKind::BeamSplitterRx | GateKind::Custom(_) => {
            return Err(OmegaError::Unsupported(format!(
                "cuda-statevector: gate {:?} is not supported on this backend",
                op.gate
            )));
        }
    };

    res.map_err(OmegaError::from)
}

/// Classify `op` as a fusion-eligible diagonal 1q gate. Same logic as
/// the Metal walker's `diagonal_factor`.
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
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

/// Apply a sequence of `GateOp`s with consecutive-diagonal-gate
/// fusion. Mirrors the Metal walker exactly (fusion semantics carry
/// over — diagonal gates commute, Id/Barrier are transparent, the
/// pending list is flushed by every non-diagonal op).
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
///
/// `reset_rng` carries the trajectory randomness for `Reset`. `Some(rng)` is a
/// shot run: each reset samples its Born outcome. `None` is an analytic
/// (`shots = None`) run, where a reset is only well-defined if it is
/// deterministic — otherwise the register ends up mixed and one statevector
/// cannot represent it, so we refuse rather than pick a branch silently.
pub(crate) fn apply_ops_fused<'a, I>(
    state: &mut CudaState,
    ops: I,
    params: &ParameterBinding,
    mut condition_skip: impl FnMut(&omega_core::circuit::GateOp) -> bool,
    mut reset_rng: Option<&mut StdRng>,
) -> OmegaResult<()>
where
    I: IntoIterator<Item = &'a omega_core::circuit::GateOp>,
{
    let mut pending: Vec<(u32, Complex64, Complex64)> = Vec::new();

    for op in ops {
        if condition_skip(op) {
            continue;
        }
        if matches!(&op.gate, GateKind::Id | GateKind::Barrier) {
            continue;
        }
        // Reset is non-unitary and full-state: flush any fused diagonals
        // first, then apply it in sequence. Kept out of `apply_op` (whose
        // Reset arm stays a "filtered upstream" guard) so the adjoint dagger
        // path never silently treats it as unitary.
        if matches!(&op.gate, GateKind::Reset) {
            flush_pending(state, &mut pending)?;
            let q = op.qubits[0].0;
            let p0 = state.reset_p0(q).map_err(OmegaError::from)?;
            let outcome_one = match reset_rng.as_deref_mut() {
                Some(rng) => rng.random::<f64>() >= p0,
                None => {
                    // Analytic run: only a determined outcome is representable.
                    //
                    // KNOWN DIVERGENCE (recorded 2026-08-05, NOT device-verified
                    // — this arm cannot be compiled or run on the Mac dev box).
                    // This refuses whenever the OUTCOME is random; the CPU
                    // backend (`sim::reset_is_deterministic`) refuses whenever
                    // the qubit is ENTANGLED, and Metal now matches the CPU.
                    // They differ on an unentangled superposition:
                    //
                    //   |+> unentangled : purity 1, p0 = 0.5
                    //     CPU/Metal  -> ALLOW (both branches land on |0>(x)rest,
                    //                   so the RESULT is deterministic even
                    //                   though the outcome is not)
                    //     CUDA       -> REFUSE  <-- false rejection
                    //
                    // Entangled cases are refused by all three, so this is a
                    // usability/consistency defect (a valid circuit errors out),
                    // not a wrong-answer defect. The fix is to adopt the purity
                    // criterion — `omega_backend_statevector::sim::
                    // reset_is_deterministic_within(&state.read_state()?, n, q,
                    // 1e-4)` — which needs a dependency on the CPU crate and a
                    // device readback, and must be verified ON a CUDA box.
                    // See LIMITATIONS.md and verification/.../Reset.lean (T1).
                    if p0 > 1e-6 && p0 < 1.0 - 1e-6 {
                        return Err(OmegaError::Unsupported(format!(
                            "cuda: analytic expectation of Reset on qubit {q} is ill-defined — \
                             the outcome is random (p0 = {p0:.6}), so the reset leaves the \
                             register in a mixed state that one statevector cannot represent. \
                             Run with shots (each shot is an independent trajectory)."
                        )));
                    }
                    p0 <= 0.5
                }
            };
            state
                .apply_reset_outcome(q, p0, outcome_one)
                .map_err(OmegaError::from)?;
            continue;
        }
        match diagonal_factor(op, params)? {
            Some(factor) => pending.push(factor),
            None => {
                flush_pending(state, &mut pending)?;
                apply_op(state, op, params)?;
            }
        }
    }
    flush_pending(state, &mut pending)?;
    Ok(())
}

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
fn flush_pending(
    state: &mut CudaState,
    pending: &mut Vec<(u32, Complex64, Complex64)>,
) -> OmegaResult<()> {
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
}

/// True iff `term` is a single-Z Pauli with coefficient 1.0 — the
/// shape the QML trainer uses for prediction observables. Used as a
/// cheap, FnOnce-factory-preserving precondition check before
/// committing to the graph path in
/// `expectation_multi_then_gradient`.
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
fn is_single_z_unit_coeff(term: &(f64, Vec<(u32, omega_core::executor::PauliOp)>)) -> bool {
    use omega_core::executor::PauliOp;
    (term.0 - 1.0).abs() < 1e-15 && term.1.len() == 1 && matches!(term.1[0].1, PauliOp::Z)
}

/// Extract per-term f32 coefficients from `obs` if every term is
/// Z-only (or identity); returns `None` if any X/Y component is
/// present. Kept for diagnostic / future use; the current
/// `expectation_multi_then_gradient` path detects Z-only via the
/// per-term `is_single_z_unit_coeff` check on `observables` and
/// extracts coefficients from `factory(zeros)` directly.
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
#[allow(dead_code)]
fn z_only_coeffs(obs: &Observable) -> Option<Vec<f32>> {
    use omega_core::executor::PauliOp;
    let mut out = Vec::with_capacity(obs.terms.len());
    for (coeff, pauli_string) in &obs.terms {
        for &(_q, ref p) in pauli_string {
            match p {
                PauliOp::I | PauliOp::Z => {}
                PauliOp::X | PauliOp::Y => return None,
            }
        }
        out.push(*coeff as f32);
    }
    Some(out)
}

/// Stable u64 hash of (circuit gate kinds + qubit indices + symbol
/// layout, gradient observable's per-term sign masks). Two calls
/// with the same circuit shape AND same observable shape produce the
/// same key; coefficient values are NOT part of the key (they're the
/// per-replay-mutable input).
#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
fn train_step_graph_key(circuit: &CircuitIR, obs: &Observable) -> u64 {
    use omega_core::executor::PauliOp;
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    circuit.num_qubits.hash(&mut h);
    circuit.num_classical_bits.hash(&mut h);
    circuit.ops.len().hash(&mut h);
    for op in &circuit.ops {
        std::mem::discriminant(&op.gate).hash(&mut h);
        for q in &op.qubits {
            q.0.hash(&mut h);
        }
        // Param expression structure (without concrete values) so
        // shape-only changes (e.g. compound expressions vs flat
        // symbols) re-key. Hash a shape tag per ParamExpr variant.
        for pe in &op.params {
            hash_param_expr_shape(pe, &mut h);
        }
        op.condition.hash(&mut h);
    }
    let mut sym_keys: Vec<_> = circuit.symbols.keys().copied().collect();
    sym_keys.sort_unstable();
    sym_keys.hash(&mut h);
    obs.terms.len().hash(&mut h);
    for (_coeff, pauli_string) in &obs.terms {
        for &(q, ref p) in pauli_string {
            q.hash(&mut h);
            let tag: u8 = match p {
                PauliOp::I => 0,
                PauliOp::X => 1,
                PauliOp::Y => 2,
                PauliOp::Z => 3,
            };
            tag.hash(&mut h);
        }
        // Term-end sentinel so two adjacent terms aren't
        // indistinguishable from one merged.
        u8::MAX.hash(&mut h);
    }
    h.finish()
}

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
fn hash_param_expr_shape<H: std::hash::Hasher>(expr: &omega_core::circuit::ParamExpr, h: &mut H) {
    use omega_core::circuit::ParamExpr;
    use std::hash::Hash;
    match expr {
        ParamExpr::Concrete(_) => 0u8.hash(h),
        ParamExpr::Symbol(s) => {
            1u8.hash(h);
            s.hash(h);
        }
        ParamExpr::Negate(a) => {
            2u8.hash(h);
            hash_param_expr_shape(a, h);
        }
        ParamExpr::Add(a, b) => {
            3u8.hash(h);
            hash_param_expr_shape(a, h);
            hash_param_expr_shape(b, h);
        }
        ParamExpr::Mul(a, b) => {
            4u8.hash(h);
            hash_param_expr_shape(a, h);
            hash_param_expr_shape(b, h);
        }
    }
}

#[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
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
    // Per-Y prefactor is (-i)^|Y|, NOT i^|Y|. The kernel forms
    // conj(ψ[i])·ψ[i^x]·phase, so `phase` must equal the matrix element
    // P[i, i^x]; for a Y qubit that is (-i)·(-1)^bit_i (Y|0⟩=i|1⟩, Y|1⟩=-i|0⟩),
    // i.e. the global Y prefactor is (-i)^|Y|. Using i^|Y| silently negates
    // every Pauli string with an ODD number of Y factors (matches the CPU
    // `expectation_pauli` warning in omega-backend-statevector/src/sim.rs).
    let y_factor = match y_count & 3 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, -1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, 1.0),
    };
    (x_mask, sign_mask, y_factor)
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_matches_build_config() {
        let expected = cfg!(all(
            any(target_os = "linux", target_os = "windows"),
            feature = "cuda"
        ));
        assert_eq!(CudaStatevectorBackend::is_available(), expected);
    }

    #[test]
    fn constructor_succeeds_iff_available() {
        let res = CudaStatevectorBackend::new();
        if CudaStatevectorBackend::is_available() {
            // The Linux+NVIDIA CI host must construct OK; if this
            // fails the GPU is missing or CUDA toolchain is broken.
            assert!(
                res.is_ok(),
                "cuda-feature build must construct OK; err={:?}",
                res.err()
            );
        } else {
            match res {
                Err(CudaError::Unavailable(_)) => {}
                Ok(_) => panic!("backend constructed on non-cuda build"),
                Err(e) => panic!("expected Unavailable, got {e:?}"),
            }
        }
    }

    #[test]
    fn allocate_refuses_too_many_qubits() {
        let backend = match CudaStatevectorBackend::new() {
            Ok(b) => b,
            Err(CudaError::Unavailable(_)) => return,
            Err(e) => panic!("unexpected error: {e:?}"),
        };
        let res = backend.allocate(CudaStatevectorBackend::MAX_QUBITS + 1);
        match res {
            Err(CudaError::AllocationRefused { .. }) => {}
            Ok(_) => panic!("expected AllocationRefused"),
            Err(e) => panic!("expected AllocationRefused, got {e:?}"),
        }
    }

    // ---- behavioural tests gated on linux+cuda ---------------------

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    fn new_backend() -> CudaStatevectorBackend {
        CudaStatevectorBackend::new().expect("device")
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    fn new_state(n: u32) -> CudaState {
        new_backend().allocate(n).expect("alloc")
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn fresh_backend_initialised_to_zero_state() {
        for &n in &[2u32, 4, 8] {
            let dim = 1usize << n;
            let state = new_state(n);
            let v = state.read_state().expect("read");
            assert_eq!(v.len(), dim);
            assert!((v[0] - Complex64::new(1.0, 0.0)).norm() < 1e-6);
            for amp in &v[1..] {
                assert!(amp.norm() < 1e-6, "amp = {amp:?}");
            }
        }
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
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
            let got = state.read_state().expect("read");
            assert_eq!(got.len(), want.len());
            let max = want
                .iter()
                .zip(got.iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0_f64, f64::max);
            assert!(max < 1e-5, "max round-trip error {max}");
        }
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn hadamard_on_zero_yields_uniform_superposition() {
        let mut state = new_state(2);
        state.apply_h(0).expect("H(0)");
        state.apply_h(1).expect("H(1)");
        let v = state.read_state().expect("read");
        let amp = 0.5_f64;
        for a in &v {
            assert!((a.re - amp).abs() < 1e-5, "expected re={amp}, got {a:?}");
            assert!(a.im.abs() < 1e-5, "expected im=0, got {a:?}");
        }
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn sx_pins_exact_phase_and_dagger_composes_to_identity() {
        // Drives the apply_op / apply_op_dagger Sx arms directly. Pins the
        // EXACT Qiskit √X phase — the U3(π/2,−π/2,π/2) alias would differ by
        // e^{−iπ/4} and fail the first two asserts.
        let op = |gate| omega_core::circuit::GateOp {
            gate,
            qubits: smallvec::smallvec![omega_core::circuit::Qubit(0)],
            params: Default::default(),
            classical_bit: None,
            condition: None,
        };
        let params = ParameterBinding::new();

        // √X|0⟩ = ½[(1+i), (1−i)].
        let mut state = new_state(1);
        apply_op(&mut state, &op(GateKind::Sx), &params).expect("Sx");
        let v = state.read_state().expect("read");
        assert!((v[0] - Complex64::new(0.5, 0.5)).norm() < 1e-5, "amp0 = {:?}", v[0]);
        assert!((v[1] - Complex64::new(0.5, -0.5)).norm() < 1e-5, "amp1 = {:?}", v[1]);

        // √X·√X = X: |0⟩ → |1⟩.
        apply_op(&mut state, &op(GateKind::Sx), &params).expect("Sx^2");
        let v = state.read_state().expect("read");
        assert!(v[0].norm() < 1e-5, "amp0 = {:?}", v[0]);
        assert!((v[1] - Complex64::new(1.0, 0.0)).norm() < 1e-5, "amp1 = {:?}", v[1]);

        // √X†·√X = I: a fresh |0⟩ round-trips back to |0⟩.
        let mut state = new_state(1);
        apply_op(&mut state, &op(GateKind::Sx), &params).expect("Sx");
        apply_op(&mut state, &op(GateKind::Sxdg), &params).expect("Sxdg");
        let v = state.read_state().expect("read");
        assert!((v[0] - Complex64::new(1.0, 0.0)).norm() < 1e-5, "amp0 = {:?}", v[0]);
        assert!(v[1].norm() < 1e-5, "amp1 = {:?}", v[1]);
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn cx_creates_bell_pair() {
        let mut state = new_state(2);
        state.apply_h(0).expect("H");
        state.apply_cx(0, 1).expect("CX");
        let v = state.read_state().expect("read");
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((v[0].re - s).abs() < 1e-5);
        assert!(v[1].norm() < 1e-5);
        assert!(v[2].norm() < 1e-5);
        assert!((v[3].re - s).abs() < 1e-5);
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn pauli_z_negates_one_amplitude() {
        let mut state = new_state(3);
        // |+++⟩
        state.apply_h(0).unwrap();
        state.apply_h(1).unwrap();
        state.apply_h(2).unwrap();
        let v0 = state.read_state().unwrap();
        state.apply_z(1).unwrap();
        let v1 = state.read_state().unwrap();
        // amplitudes where bit 1 is set should flip sign.
        for i in 0..(1usize << 3) {
            let bit1 = (i >> 1) & 1;
            let expected = if bit1 == 1 { -v0[i] } else { v0[i] };
            assert!((v1[i] - expected).norm() < 1e-5, "i={i}");
        }
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn expectation_multi_then_gradient_matches_split_path() {
        // Pin the trainer-hot-path override against the split
        // (expectation_multi + adjoint_gradient) path on a small
        // parameterized HEA: predictions and gradients should agree
        // bit-for-bit modulo f32 round-off.
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};
        use omega_core::executor::PauliOp;

        let n: u32 = 4;
        let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
        for s in 0..4u32 {
            circuit.symbols.insert(s, format!("theta_{s}"));
        }
        for q in 0..n {
            circuit.ops.push(GateOp {
                gate: GateKind::Ry,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(q)],
                classical_bit: None,
                condition: None,
            });
        }
        for q in 0..n - 1 {
            circuit.ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec::smallvec![Qubit(q), Qubit(q + 1)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        let mut params = ParameterBinding::new();
        for s in 0..4u32 {
            params.bind(s, ((s as f64) * 0.41 - 0.7).sin());
        }
        let observables: Vec<Observable> = (0..n)
            .map(|q| Observable {
                terms: vec![(1.0, vec![(q, PauliOp::Z)])],
            })
            .collect();
        // Trainer-style residual gradient: 2·(y_hat - y)·Z_q with
        // synthetic targets.
        let targets = [0.1_f64, -0.3, 0.5, -0.2];
        let factory: GradientObservableFactory<'_> = Box::new(move |y_hat: &[f64]| Observable {
            terms: y_hat
                .iter()
                .zip(targets.iter())
                .enumerate()
                .map(|(q, (h, y))| (2.0 * (h - y), vec![(q as u32, PauliOp::Z)]))
                .collect(),
        });

        let cuda = CudaStatevectorBackend::new().expect("cuda");

        // Reference: split path.
        let preds_ref = cuda
            .expectation_multi(&circuit, &params, &observables)
            .expect("split predictions");
        let factory_ref: GradientObservableFactory<'_> =
            Box::new(move |y_hat: &[f64]| Observable {
                terms: y_hat
                    .iter()
                    .zip(targets.iter())
                    .enumerate()
                    .map(|(q, (h, y))| (2.0 * (h - y), vec![(q as u32, PauliOp::Z)]))
                    .collect(),
            });
        let obs_ref = factory_ref(&preds_ref);
        let grad_ref = cuda
            .adjoint_gradient(&circuit, &params, &obs_ref)
            .expect("split gradient")
            .expect("has gradient");

        // Fused path.
        let (preds, grad) = cuda
            .expectation_multi_then_gradient(&circuit, &params, &observables, factory)
            .expect("fused");
        let grad = grad.expect("has gradient");

        assert_eq!(preds.len(), preds_ref.len());
        for (a, b) in preds.iter().zip(preds_ref.iter()) {
            assert!((a - b).abs() < 1e-6, "preds {a} vs {b}");
        }
        assert_eq!(grad.len(), grad_ref.len());
        for ((sa, ga), (sb, gb)) in grad.iter().zip(grad_ref.iter()) {
            assert_eq!(sa, sb);
            assert!((ga - gb).abs() < 1e-6, "grad {sa}: {ga} vs {gb}");
        }
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn reset_matches_cpu() {
        // Mid-circuit Reset on the GPU must reproduce the CPU backend's reset
        // CHANNEL, not the old coherent fold. Reset a qubit out of an
        // entangled, non-|0> state (Bell + RY) then keep computing, so a wrong
        // reset propagates into the final distribution.
        //
        // The channel leaves the register MIXED, so there is no analytic
        // statevector to compare — this used to diff `shots: None` amplitudes,
        // which both backends now (correctly) refuse. Compare shot
        // distributions instead: each shot is an independent trajectory.
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
        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        push(&mut circuit, GateKind::H, &[0], &[]);
        push(&mut circuit, GateKind::CX, &[0, 1], &[]);
        push(&mut circuit, GateKind::Ry, &[1], &[0.7]);
        push(&mut circuit, GateKind::Reset, &[0], &[]);
        push(&mut circuit, GateKind::H, &[1], &[]);

        let params = ParameterBinding::new();
        let cpu = StatevectorBackend::new();
        let cuda = CudaStatevectorBackend::new().expect("cuda");

        // Analytic mode is ill-defined for a reset on an entangled qubit and
        // must be refused by BOTH backends rather than silently answered.
        let analytic = ExecConfig {
            shots: None,
            ..ExecConfig::default()
        };
        assert!(
            cpu.execute(&circuit, &params, &analytic).is_err(),
            "cpu must refuse analytic entangled reset"
        );
        assert!(
            cuda.execute(&circuit, &params, &analytic).is_err(),
            "cuda must refuse analytic entangled reset"
        );

        // Shot distributions must agree. Independent RNG streams, so compare
        // frequencies within a statistical band, not bit-for-bit.
        const SHOTS: u32 = 8000;
        let shot_cfg = |seed| ExecConfig {
            shots: Some(SHOTS),
            seed: Some(seed),
            ..ExecConfig::default()
        };
        let freq = |r: ExecResult| -> [f64; 4] {
            let m = r.counts().clone();
            let mut f = [0.0; 4];
            for (k, v) in m {
                f[(k as usize) & 3] = v as f64 / SHOTS as f64;
            }
            f
        };
        let a = freq(cpu.execute(&circuit, &params, &shot_cfg(7)).expect("cpu"));
        let b = freq(
            cuda.execute(&circuit, &params, &shot_cfg(11))
                .expect("cuda"),
        );
        // 5 sigma on 8000 draws at p<=0.5 is ~0.028; 0.04 keeps the gate sharp
        // (the old fold misses whole outcomes) without being seed-fragile.
        for i in 0..4 {
            assert!(
                (a[i] - b[i]).abs() < 0.04,
                "cuda vs cpu reset distribution differs at |{i:02b}>: cpu {:.4} gpu {:.4} \
                 (full: cpu {a:?} gpu {b:?})",
                a[i],
                b[i]
            );
        }
        // q0 was reset: it must read 0 on every shot, on both backends.
        assert!(a[1] + a[3] < 1e-12, "cpu: q0 not reset, {a:?}");
        assert!(b[1] + b[3] < 1e-12, "gpu: q0 not reset, {b:?}");
    }

    /// The three circuits that discriminate a real reset CHANNEL from the
    /// plausible near-misses, pinned to Qiskit Aer's exact `DensityMatrix`
    /// (qiskit 2.4.1 / aer 0.17.2) for `Bell(q0,q1); reset q0`:
    ///
    /// ```text
    /// rho = diag(0.5, 0, 0.5, 0)   rho_q1 = I/2   <X_1> = <Y_1> = <Z_1> = 0
    /// ```
    ///
    /// | gate | correct | coherent fold | post-selection |
    /// |---|---|---|---|
    /// | A  measure q1           | ~50/50 | ~50/50 | 0%     |
    /// | B  H q1 then measure q1 | ~50/50 | 0%     | ~50/50 |
    /// | C  reset \|-> then measure | 0%  | 100%   | 0%     |
    ///
    /// A and B fail in DIFFERENT bases, so no single gate is sufficient — that
    /// is exactly why the fold survived cross-backend "agreement" checks. Keep
    /// all three. Mirrors `omega-backend-statevector/tests/reset_channel.rs`.
    #[test]
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    fn reset_channel_matches_aer_ground_truth() {
        use omega_core::circuit::{CircuitType, GateOp, Qubit};
        const SHOTS: u32 = 4000;
        const BAND: u32 = 250;

        let push = |c: &mut CircuitIR, gate, qubits: &[u32]| {
            c.ops.push(GateOp {
                gate,
                qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
                params: Default::default(),
                classical_bit: None,
                condition: None,
            });
        };
        let cuda = CudaStatevectorBackend::new().expect("cuda");
        let run = |c: &CircuitIR| -> [u32; 4] {
            let cfg = ExecConfig {
                shots: Some(SHOTS),
                seed: Some(7),
                ..ExecConfig::default()
            };
            let m = cuda
                .execute(c, &ParameterBinding::new(), &cfg)
                .expect("cuda execute")
                .counts()
                .clone();
            let mut out = [0u32; 4];
            for (k, v) in m {
                out[(k as usize) & 3] += v;
            }
            out
        };
        let ones = |f: [u32; 4], bit: usize| -> u32 {
            (0..4).filter(|i| i & bit != 0).map(|i| f[i]).sum()
        };

        // A — partner maximally mixed in Z (post-selection would give 0).
        let mut a = CircuitIR::new(2, CircuitType::GateBased);
        push(&mut a, GateKind::H, &[0]);
        push(&mut a, GateKind::CX, &[0, 1]);
        push(&mut a, GateKind::Reset, &[0]);
        let ka = ones(run(&a), 2);
        assert!(
            ka.abs_diff(SHOTS / 2) < BAND,
            "cuda A: q1 ones {ka}/{SHOTS}, want ~{}",
            SHOTS / 2
        );

        // B — and in X (the coherent fold would leave q1 = |+> and give 0).
        let mut b = a.clone();
        push(&mut b, GateKind::H, &[1]);
        let kb = ones(run(&b), 2);
        assert!(
            kb.abs_diff(SHOTS / 2) < BAND,
            "cuda B: q1 ones {kb}/{SHOTS}, want ~{}",
            SHOTS / 2
        );

        // C — reset of |-> must read 0 with certainty (the fold read 1 always).
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        push(&mut c, GateKind::X, &[0]);
        push(&mut c, GateKind::H, &[0]);
        push(&mut c, GateKind::Reset, &[0]);
        let kc = ones(run(&c), 1);
        assert_eq!(kc, 0, "cuda C: reset(|->) must read 0, got {kc} ones");

        // The reset qubit itself is |0> on every trajectory.
        assert_eq!(ones(run(&a), 1), 0, "cuda: q0 must be |0> after reset");
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn execute_matches_cpu_bell_pair() {
        use omega_backend_statevector::StatevectorBackend;
        use omega_core::circuit::{CircuitType, GateOp, Qubit};

        let mut circuit = CircuitIR::new(2, omega_core::circuit::CircuitType::GateBased);
        let _ = CircuitType::GateBased;
        circuit.ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(0), Qubit(1)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        let params = ParameterBinding::new();
        // shots = None ⇒ exact statevector. ExecConfig::default has
        // shots = 1024 which would route both backends down the
        // sampler.
        let cfg = ExecConfig {
            shots: None,
            ..ExecConfig::default()
        };
        let cpu = StatevectorBackend::new();
        let cuda = CudaStatevectorBackend::new().expect("cuda");
        let cpu_res = cpu.execute(&circuit, &params, &cfg).expect("cpu execute");
        let cuda_res = cuda.execute(&circuit, &params, &cfg).expect("cuda execute");
        match (cpu_res, cuda_res) {
            (ExecResult::Statevector(a), ExecResult::Statevector(b)) => {
                assert_eq!(a.len(), b.len());
                let max = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).norm())
                    .fold(0.0f64, f64::max);
                assert!(max < 1e-5, "max amp diff {max}");
            }
            _ => panic!("expected statevectors"),
        }
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn adjoint_cuda_matches_cpu_12q_hea() {
        // GPU_PLAN.md Phase 2 target — mirrors Metal's
        // `adjoint_metal_matches_cpu_12q_hea` exactly so the same
        // numerics are pinned across both GPU backends. 12q/16 params,
        // tolerance 1e-5 (relaxed from the CPU's 1e-10 because the
        // GPU layout is f32).
        use omega_backend_statevector::StatevectorBackend;
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};
        use omega_core::executor::PauliOp;

        let n: u32 = 12;
        let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
        for s in 0..16u32 {
            circuit.symbols.insert(s, format!("theta_{s}"));
        }
        for q in 0..8 {
            circuit.ops.push(GateOp {
                gate: GateKind::Ry,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(q)],
                classical_bit: None,
                condition: None,
            });
        }
        // Non-parametric √X / √X† in the middle of the ansatz: adds no gradient
        // symbols but forces the adjoint's Sx/Sxdg arms (forward Sx→sx, dagger
        // Sx→sxdg) to agree with the CPU reference bit-for-bit.
        for (q, gate) in [(0u32, GateKind::Sx), (1u32, GateKind::Sxdg)] {
            circuit.ops.push(GateOp {
                gate,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        for q in 0..n - 1 {
            circuit.ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec::smallvec![Qubit(q), Qubit(q + 1)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
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

        let cuda = CudaStatevectorBackend::new().expect("cuda");
        let cuda_grads = cuda
            .adjoint_gradient(&circuit, &params, &obs)
            .expect("cuda adjoint")
            .expect("cuda has adjoint");

        assert_eq!(cpu_grads.len(), 16);
        assert_eq!(cuda_grads.len(), 16);
        let mut max_diff = 0.0_f64;
        for ((sa, ga), (sb, gb)) in cpu_grads.iter().zip(cuda_grads.iter()) {
            assert_eq!(sa, sb);
            let diff = (ga - gb).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 1e-5,
                "symbol {sa}: cpu = {ga:.10}, cuda = {gb:.10}, diff = {diff:.3e}"
            );
        }
        eprintln!("12q/16-param HEA: max abs diff vs CPU adjoint = {max_diff:.3e}");
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn expectation_matches_cpu_pauli_string() {
        // Cross-check fused pauli_expectation kernel against CPU on a
        // mixed Pauli observable that exercises X/Y/Z components.
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
        let cuda = CudaStatevectorBackend::new().expect("cuda");
        let cuda_e = cuda.expectation(&circuit, &params, &obs).expect("cuda");
        assert!(
            (cpu_e - cuda_e).abs() < 1e-5,
            "cpu = {cpu_e}, cuda = {cuda_e}, diff = {:.3e}",
            (cpu_e - cuda_e).abs()
        );
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    #[ignore = "perf timing — runs ~30 s; opt-in via `cargo test -- --ignored`"]
    fn forward_graph_in_trainer_shape_perf() {
        // Times an end-to-end trainer-shape loop:
        //   for k in 0..N: forward(circuit, params_k) → predictions
        // using (a) backend.execute (naive), (b) ForwardGraph.replay
        // followed by read_state. The shape mirrors what the QML
        // trainer's hot path does when computing predictions —
        // forward sweep + readout. Backward sweep is not measured
        // here (it's currently NOT graph-captured; that's the next
        // slice's lever).
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};
        use omega_core::executor::PauliOp;
        use std::time::Instant;

        const N: u32 = 14;
        const ITERS: usize = 5000;

        let mut circuit = CircuitIR::new(N, CircuitType::GateBased);
        for s in 0..16u32 {
            circuit.symbols.insert(s, format!("theta_{s}"));
        }
        for q in 0..8u32 {
            circuit.ops.push(GateOp {
                gate: GateKind::Ry,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(q)],
                classical_bit: None,
                condition: None,
            });
        }
        for q in 0..N - 1 {
            circuit.ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec::smallvec![Qubit(q), Qubit(q + 1)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        for (i, q) in (4..12u32).enumerate() {
            circuit.ops.push(GateOp {
                gate: GateKind::Rz,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(8 + i as u32)],
                classical_bit: None,
                condition: None,
            });
        }

        let backend = new_backend();
        // Per-measurement-qubit observables, like the trainer.
        let observables: Vec<Observable> = vec![0u32, N - 1]
            .into_iter()
            .map(|q| Observable {
                terms: vec![(1.0, vec![(q, PauliOp::Z)])],
            })
            .collect();
        let mut params = ParameterBinding::new();

        // Naive: backend.expectation_multi via the existing path.
        let t0 = Instant::now();
        for k in 0..ITERS {
            for s in 0..16u32 {
                params.bind(s, ((s as f64 + k as f64) * 0.317 - 0.84).sin() * 0.4);
            }
            let _ = backend
                .expectation_multi(&circuit, &params, &observables)
                .unwrap();
        }
        backend.handle.stream.synchronize().unwrap();
        let naive = t0.elapsed();

        // Graph: replay forward, then run the same per-Z reductions
        // on the resulting state. We use the StateBuffer's
        // pauli_expectation directly.
        let mut graph = forward_graph::ForwardGraph::capture(&backend.handle, &circuit).unwrap();
        let t1 = Instant::now();
        for k in 0..ITERS {
            for s in 0..16u32 {
                params.bind(s, ((s as f64 + k as f64) * 0.317 - 0.84).sin() * 0.4);
            }
            let state = graph.replay(&circuit, &params).unwrap();
            for obs in &observables {
                for (_coeff, pauli_string) in &obs.terms {
                    let (x_mask, sign_mask, y_factor) = pauli_masks(pauli_string);
                    let _ = state
                        .pauli_expectation(x_mask, sign_mask, y_factor)
                        .unwrap();
                }
            }
        }
        let graph_t = t1.elapsed();

        eprintln!(
            "trainer-forward naive: {:?}  ({:.2} µs/iter)",
            naive,
            naive.as_secs_f64() * 1e6 / ITERS as f64
        );
        eprintln!(
            "trainer-forward graph: {:?}  ({:.2} µs/iter)",
            graph_t,
            graph_t.as_secs_f64() * 1e6 / ITERS as f64
        );
        eprintln!(
            "speedup: {:.2}× (graph/naive = {:.3})",
            naive.as_secs_f64() / graph_t.as_secs_f64(),
            graph_t.as_secs_f64() / naive.as_secs_f64()
        );
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    #[ignore = "perf timing — runs ~10 s; opt-in via `cargo test -- --ignored`"]
    fn forward_graph_replay_perf_vs_naive() {
        // Time `REPLAYS` forward sweeps via the graph path vs naive
        // per-call kernel launches. Pins the empirical per-launch
        // saving on this host. Also exposes whether graph replay
        // wins enough to pay back the per-replay memcpy_htod
        // (3 pools, ~50 entries total at this shape — KB-class).
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};
        use std::time::Instant;

        const N: u32 = 14;
        const REPLAYS: usize = 5000;

        let mut circuit = CircuitIR::new(N, CircuitType::GateBased);
        for s in 0..16u32 {
            circuit.symbols.insert(s, format!("theta_{s}"));
        }
        for q in 0..8u32 {
            circuit.ops.push(GateOp {
                gate: GateKind::Ry,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(q)],
                classical_bit: None,
                condition: None,
            });
        }
        for q in 0..N - 1 {
            circuit.ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec::smallvec![Qubit(q), Qubit(q + 1)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        for (i, q) in (4..12u32).enumerate() {
            circuit.ops.push(GateOp {
                gate: GateKind::Rz,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(8 + i as u32)],
                classical_bit: None,
                condition: None,
            });
        }

        let backend = new_backend();
        let cfg = ExecConfig {
            shots: None,
            ..ExecConfig::default()
        };
        let mut params = ParameterBinding::new();

        // Naive timing: REPLAYS full executes via the by-value path.
        let t0 = Instant::now();
        for k in 0..REPLAYS {
            for s in 0..16u32 {
                params.bind(s, ((s as f64 + k as f64) * 0.317 - 0.84).sin() * 0.4);
            }
            let _ = backend.execute(&circuit, &params, &cfg).unwrap();
        }
        // Make sure all GPU work flushed before we stop the clock.
        backend.handle.stream.synchronize().unwrap();
        let naive = t0.elapsed();

        // Graph timing: capture once, replay REPLAYS times.
        let mut graph =
            forward_graph::ForwardGraph::capture(&backend.handle, &circuit).expect("capture");
        let t1 = Instant::now();
        for k in 0..REPLAYS {
            for s in 0..16u32 {
                params.bind(s, ((s as f64 + k as f64) * 0.317 - 0.84).sin() * 0.4);
            }
            let _ = graph.replay(&circuit, &params).unwrap();
        }
        // Sync the graph's stream before we stop the clock.
        graph.synchronize().unwrap();
        let graph_t = t1.elapsed();

        let ops = circuit.ops.len();
        eprintln!(
            "naive: {:?}  ({} replays × {} ops = {} launches; {:.2} µs/launch)",
            naive,
            REPLAYS,
            ops,
            REPLAYS * ops,
            naive.as_secs_f64() * 1e6 / (REPLAYS * ops) as f64
        );
        eprintln!(
            "graph: {:?}  ({:.2} µs/replay incl. memcpy_htod)",
            graph_t,
            graph_t.as_secs_f64() * 1e6 / REPLAYS as f64
        );
        eprintln!(
            "speedup: {:.2}× (graph/naive = {:.3})",
            naive.as_secs_f64() / graph_t.as_secs_f64(),
            graph_t.as_secs_f64() / naive.as_secs_f64()
        );
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn forward_graph_replay_matches_naive_forward() {
        // Capture the forward sweep for an HEA-shape circuit, replay
        // with a fixed param binding, and compare the resulting
        // statevector against the same circuit run through the
        // by-value kernel path. Both should agree to f32 precision.
        // Pins that:
        //   (a) the captured graph executes the kernels in the right
        //       order with the right params,
        //   (b) the pooled kernel arithmetic matches the by-value
        //       arithmetic on every gate kind we currently dispatch.
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};

        let n: u32 = 6;
        let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
        for s in 0..4u32 {
            circuit.symbols.insert(s, format!("theta_{s}"));
        }
        // Layer 1: 4 Ry on q0..q3.
        for q in 0..4 {
            circuit.ops.push(GateOp {
                gate: GateKind::Ry,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(q)],
                classical_bit: None,
                condition: None,
            });
        }
        // Entangling CX ladder.
        for q in 0..n - 1 {
            circuit.ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec::smallvec![Qubit(q), Qubit(q + 1)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        // Layer 2: H on q4, Rz on q5 (mixes 1q apply_1q + diagonal).
        circuit.ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(4)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec::smallvec![Qubit(5)],
            params: smallvec::smallvec![ParamExpr::Concrete(0.731)],
            classical_bit: None,
            condition: None,
        });

        let mut params = ParameterBinding::new();
        for s in 0..4u32 {
            params.bind(s, ((s as f64) * 0.41 - 0.7).sin());
        }

        let backend = new_backend();
        let cfg = ExecConfig {
            shots: None,
            ..ExecConfig::default()
        };
        let ref_result = backend
            .execute(&circuit, &params, &cfg)
            .expect("naive execute");
        let v_ref = match ref_result {
            ExecResult::Statevector(v) => v,
            _ => panic!("expected statevector"),
        };

        let mut graph = forward_graph::ForwardGraph::capture(&backend.handle, &circuit)
            .expect("capture forward graph");
        // read_state synchronizes internally before pulling host
        // bytes — no separate graph.synchronize() needed.
        let state = graph.replay(&circuit, &params).expect("replay");
        let v_graph = state.read_state().expect("read state");

        assert_eq!(v_ref.len(), v_graph.len());
        let max = v_ref
            .iter()
            .zip(v_graph.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(max < 1e-5, "graph vs naive max diff = {max:.3e}");

        // Replay again with different params — graph should re-use
        // the captured kernel sequence with new pool contents.
        let mut params2 = ParameterBinding::new();
        for s in 0..4u32 {
            params2.bind(s, -((s as f64) * 0.23 + 0.1).cos());
        }
        let v_ref2 = match backend
            .execute(&circuit, &params2, &cfg)
            .expect("naive execute 2")
        {
            ExecResult::Statevector(v) => v,
            _ => panic!("expected statevector"),
        };
        let state2 = graph.replay(&circuit, &params2).expect("replay 2");
        let v_graph2 = state2.read_state().expect("read state 2");
        let max2 = v_ref2
            .iter()
            .zip(v_graph2.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(max2 < 1e-5, "graph replay #2 max diff = {max2:.3e}");
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn train_step_graph_matches_naive_backward() {
        // Capture + replay an HEA-shape circuit with a Z-only
        // gradient observable; gradients must match the naive
        // adjoint_gradient path within 1e-5 (the same tolerance
        // adjoint_cuda_matches_cpu_12q_hea uses — atomic-add
        // ordering is non-deterministic but bounded).
        use omega_core::circuit::{CircuitType, GateOp, ParamExpr, Qubit};
        use omega_core::executor::PauliOp;

        let n: u32 = 8;
        let mut circuit = CircuitIR::new(n, CircuitType::GateBased);
        for s in 0..6u32 {
            circuit.symbols.insert(s, format!("theta_{s}"));
        }
        // Layer 1: 4 Ry on q0..q3.
        for q in 0..4 {
            circuit.ops.push(GateOp {
                gate: GateKind::Ry,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(q)],
                classical_bit: None,
                condition: None,
            });
        }
        // Non-parametric √X / √X† in the graph-capture path: exercises the
        // Sx/Sxdg arms of classify_op_kernel + forward_1q_matrix against the
        // naive adjoint (dagger derived by conj-transpose).
        for (q, gate) in [(0u32, GateKind::Sx), (2u32, GateKind::Sxdg)] {
            circuit.ops.push(GateOp {
                gate,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        // CX ladder.
        for q in 0..n - 1 {
            circuit.ops.push(GateOp {
                gate: GateKind::CX,
                qubits: smallvec::smallvec![Qubit(q), Qubit(q + 1)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        // Layer 2: 2 Rz on q4, q5.
        for (i, q) in (4..6u32).enumerate() {
            circuit.ops.push(GateOp {
                gate: GateKind::Rz,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![ParamExpr::Symbol(4 + i as u32)],
                classical_bit: None,
                condition: None,
            });
        }

        let mut params = ParameterBinding::new();
        for s in 0..6u32 {
            params.bind(s, ((s as f64) * 0.317 - 0.84).sin() * 0.4);
        }

        // Z-only gradient observable, two terms — like the trainer's
        // residual gradient `Σ 2(y_hat - y)·Z_q`.
        let coeffs: Vec<f32> = vec![0.7, -0.3];
        let template = Observable {
            terms: vec![
                (coeffs[0] as f64, vec![(0u32, PauliOp::Z)]),
                (coeffs[1] as f64, vec![(n - 1, PauliOp::Z)]),
            ],
        };

        let backend = new_backend();

        // Compute predictions outside the graph so we can derive
        // y_label values that, after the captured host_func runs
        // `coeffs_graph = 2·(graph_preds - y_label)`, reconstruct
        // the test's intended `coeffs`.
        let pred_obs: Vec<Observable> = template
            .terms
            .iter()
            .map(|(_, p)| Observable {
                terms: vec![(1.0, p.clone())],
            })
            .collect();
        let predictions = backend
            .expectation_multi(&circuit, &params, &pred_obs)
            .expect("predictions");
        let y_labels: Vec<f32> = predictions
            .iter()
            .zip(coeffs.iter())
            .map(|(&p, &c)| p as f32 - c / 2.0)
            .collect();

        // Naive reference path with the test's `template` Observable.
        let grad_ref = backend
            .adjoint_gradient(&circuit, &params, &template)
            .expect("naive adjoint")
            .expect("has gradient");

        // Graph path: replay_with_y_labels reconstructs the same
        // gradient observable inside the graph via the host_func
        // callback.
        let mut graph =
            backward_graph::TrainStepGraph::capture(&backend.handle, &circuit, &template)
                .expect("TrainStepGraph capture");
        let (_graph_predictions, grad) = graph
            .replay_with_y_labels(&circuit, &params, &y_labels)
            .expect("replay");

        assert_eq!(grad.len(), grad_ref.len());
        let mut max_diff = 0.0_f64;
        for ((sa, ga), (sb, gb)) in grad.iter().zip(grad_ref.iter()) {
            assert_eq!(sa, sb);
            let diff = (ga - gb).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 1e-5,
                "sym {sa}: graph = {ga:.10}, naive = {gb:.10}, diff = {diff:.3e}"
            );
        }
        eprintln!("graph vs naive max diff = {max_diff:.3e}");

        // Replay #2 with different coeffs to exercise pool updates.
        // Re-derive y_labels for the new target coeffs.
        let coeffs2: Vec<f32> = vec![1.4, 0.5];
        let template2 = Observable {
            terms: vec![
                (coeffs2[0] as f64, vec![(0u32, PauliOp::Z)]),
                (coeffs2[1] as f64, vec![(n - 1, PauliOp::Z)]),
            ],
        };
        let y_labels2: Vec<f32> = predictions
            .iter()
            .zip(coeffs2.iter())
            .map(|(&p, &c)| p as f32 - c / 2.0)
            .collect();
        let grad_ref2 = backend
            .adjoint_gradient(&circuit, &params, &template2)
            .expect("naive #2")
            .expect("has gradient #2");
        let (_p2, grad2) = graph
            .replay_with_y_labels(&circuit, &params, &y_labels2)
            .expect("replay #2");
        for ((sa, ga), (sb, gb)) in grad2.iter().zip(grad_ref2.iter()) {
            assert_eq!(sa, sb);
            assert!(
                (ga - gb).abs() < 1e-5,
                "replay #2 sym {sa}: graph = {ga:.10}, naive = {gb:.10}"
            );
        }

        // Replay #3 with different params (same coeffs as #1) so
        // forward / dagger / derivative pools all change. Predictions
        // are also different under the new params, so re-derive the
        // y_labels.
        let mut params3 = ParameterBinding::new();
        for s in 0..6u32 {
            params3.bind(s, ((s as f64) * -0.21 + 0.6).cos() * 0.3);
        }
        let predictions3 = backend
            .expectation_multi(&circuit, &params3, &pred_obs)
            .expect("predictions #3");
        let y_labels3: Vec<f32> = predictions3
            .iter()
            .zip(coeffs.iter())
            .map(|(&p, &c)| p as f32 - c / 2.0)
            .collect();
        let grad_ref3 = backend
            .adjoint_gradient(&circuit, &params3, &template)
            .expect("naive #3")
            .expect("has gradient #3");
        let (_p3, grad3) = graph
            .replay_with_y_labels(&circuit, &params3, &y_labels3)
            .expect("replay #3");
        for ((sa, ga), (sb, gb)) in grad3.iter().zip(grad_ref3.iter()) {
            assert_eq!(sa, sb);
            assert!(
                (ga - gb).abs() < 1e-5,
                "replay #3 sym {sa}: graph = {ga:.10}, naive = {gb:.10}"
            );
        }
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn diagonal_pauli_sum_via_pool_matches_by_value() {
        // Pin the pooled-pool wrapper against the existing
        // by-value apply_diagonal_pauli_sum on a Z-only observable.
        // Same kernel runs in both paths; the wrapper just lets the
        // CUDA-graph capture caller hold the sign_masks / coeffs
        // pools across replays.
        use cudarc::driver::CudaSlice;
        use num_complex::Complex64;

        let backend = new_backend();
        let n: u32 = 5;
        let dim = 1usize << n;

        // Random-ish input state.
        let psi_amps: Vec<Complex64> = (0..dim)
            .map(|i| {
                let re = ((i as f64) * 0.13 - 0.4).sin();
                let im = ((i as f64) * 0.07 + 0.2).cos();
                Complex64::new(re, im)
            })
            .collect();
        let mut psi = backend.allocate(n).unwrap();
        psi.inner.write_state(&psi_amps).unwrap();

        // Z-only observable: 0.7·Z_0 + (-0.3)·Z_2 + 0.5·Z_1·Z_3.
        // sign_mask bits are set at qubits the Z-term acts on.
        let terms_by_value: Vec<(u32, f32)> = vec![(0b00001, 0.7), (0b00100, -0.3), (0b01010, 0.5)];

        // Reference: existing by-value path.
        let mut nu_ref = backend.allocate(n).unwrap();
        psi.inner
            .apply_diagonal_pauli_sum(&mut nu_ref.inner, &terms_by_value)
            .unwrap();
        let v_ref = nu_ref.read_state().unwrap();

        // Pool path: pre-allocate sign_masks + coeffs pools, populate,
        // launch via wrapper.
        let stream = backend.handle.stream.clone();
        let sign_masks: Vec<u32> = terms_by_value.iter().map(|(m, _)| *m).collect();
        let coeffs: Vec<f32> = terms_by_value.iter().map(|(_, c)| *c).collect();
        let sign_masks_pool: CudaSlice<u32> = stream.clone_htod(&sign_masks).unwrap();
        let coeffs_pool: CudaSlice<f32> = stream.clone_htod(&coeffs).unwrap();
        let mut nu = backend.allocate(n).unwrap();
        psi.inner
            .apply_diagonal_pauli_sum_via_pool(
                &mut nu.inner,
                &sign_masks_pool,
                &coeffs_pool,
                terms_by_value.len() as u32,
            )
            .unwrap();
        let v = nu.read_state().unwrap();

        assert_eq!(v.len(), v_ref.len());
        let max = v
            .iter()
            .zip(v_ref.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(max < 1e-6, "pool path diff vs by-value = {max:.3e}");

        // Now update coeffs in place (simulates per-train-pt change)
        // and verify the second invocation reflects the new values.
        let coeffs2: Vec<f32> = vec![1.4, -0.6, 1.0];
        let coeffs_pool2: CudaSlice<f32> = stream.clone_htod(&coeffs2).unwrap();
        let terms_by_value2: Vec<(u32, f32)> = sign_masks
            .iter()
            .zip(coeffs2.iter())
            .map(|(&m, &c)| (m, c))
            .collect();
        let mut nu2 = backend.allocate(n).unwrap();
        psi.inner
            .apply_diagonal_pauli_sum_via_pool(
                &mut nu2.inner,
                &sign_masks_pool,
                &coeffs_pool2,
                terms_by_value2.len() as u32,
            )
            .unwrap();
        let v2 = nu2.read_state().unwrap();
        let mut nu_ref2 = backend.allocate(n).unwrap();
        psi.inner
            .apply_diagonal_pauli_sum(&mut nu_ref2.inner, &terms_by_value2)
            .unwrap();
        let v_ref2 = nu_ref2.read_state().unwrap();
        let max2 = v2
            .iter()
            .zip(v_ref2.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(max2 < 1e-6, "second-call coeff update diff = {max2:.3e}");
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn pauli_z_expectation_to_slot_matches_pauli_expectation() {
        use cudarc::driver::CudaSlice;
        use num_complex::Complex64;

        let backend = new_backend();
        let n: u32 = 5;
        let dim = 1usize << n;

        let psi_amps: Vec<Complex64> = (0..dim)
            .map(|i| {
                let re = ((i as f64) * 0.13 - 0.4).sin();
                let im = ((i as f64) * 0.07 + 0.2).cos();
                Complex64::new(re, im)
            })
            .collect();
        let mut psi = backend.allocate(n).unwrap();
        psi.inner.write_state(&psi_amps).unwrap();

        for q in 0..n {
            let sign_mask = 1u32 << q;
            let ip = psi
                .inner
                .pauli_expectation(0, sign_mask, Complex64::new(1.0, 0.0))
                .unwrap();
            let want = ip.re;

            let stream = backend.handle.stream.clone();
            let mut predictions_dev: CudaSlice<f64> = stream.alloc_zeros::<f64>(1).unwrap();
            psi.inner
                .pauli_z_expectation_to_slot(&mut predictions_dev, sign_mask, 0)
                .unwrap();
            let host: Vec<f64> = stream.clone_dtoh(&predictions_dev).unwrap();
            stream.synchronize().unwrap();
            let got = host[0];

            let diff = (want - got).abs();
            assert!(
                diff < 1e-5,
                "qubit {q}: pauli_z_to_slot {got:.10} vs pauli_expectation {want:.10}, diff = {diff:.3e}"
            );
        }
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn fused_inner_product_accumulate_matches_naive() {
        // Cross-check the fused inner_product_accumulate_pooled
        // kernel against the existing inner_product_deferred + host
        // accumulate path. For a known nu, temp, chain, sym_idx
        // config, both must produce the same grad slot value within
        // f32 inner-product precision (atomicAdd nondeterminism is
        // bounded by 1e-5 on this scale).
        use cudarc::driver::CudaSlice;
        use num_complex::Complex64;

        let backend = new_backend();
        let n: u32 = 5;
        let dim = 1usize << n;

        // Build two random-ish states.
        let nu_amps: Vec<Complex64> = (0..dim)
            .map(|i| {
                let re = ((i as f64) * 0.13 - 0.4).sin();
                let im = ((i as f64) * 0.07 + 0.2).cos();
                Complex64::new(re, im)
            })
            .collect();
        let temp_amps: Vec<Complex64> = (0..dim)
            .map(|i| {
                let re = ((i as f64) * -0.21 + 0.6).sin();
                let im = ((i as f64) * 0.11 - 0.3).cos();
                Complex64::new(re, im)
            })
            .collect();
        let mut nu = backend.allocate(n).unwrap();
        let mut temp = backend.allocate(n).unwrap();
        nu.inner.write_state(&nu_amps).unwrap();
        temp.inner.write_state(&temp_amps).unwrap();

        // Naive reference: inner_product_deferred + 2*Re*chain.
        let chain: f64 = 0.731;
        let pending = nu.inner.inner_product_deferred(&temp.inner).unwrap();
        backend.handle.stream.synchronize().unwrap();
        let ip = pending.reduce();
        let want: f64 = 2.0 * ip.re * chain;

        // Fused path: pre-allocate grad_dev (one slot), chain pool
        // (one entry = chain), sym pool (one entry = 0). Run kernel,
        // sync, read back grad_dev[0].
        let stream = backend.handle.stream.clone();
        let mut grad_dev: CudaSlice<f64> = stream.alloc_zeros::<f64>(1).unwrap();
        let chain_pool: CudaSlice<f64> = stream.clone_htod(&[chain]).unwrap();
        let sym_pool: CudaSlice<u32> = stream.clone_htod(&[0u32]).unwrap();
        nu.inner
            .inner_product_accumulate_via_pool(
                &temp.inner,
                &mut grad_dev,
                &chain_pool,
                &sym_pool,
                0,
            )
            .unwrap();
        let grad_host: Vec<f64> = stream.clone_dtoh(&grad_dev).unwrap();
        stream.synchronize().unwrap();
        let got = grad_host[0];

        let diff = (want - got).abs();
        assert!(
            diff < 1e-5,
            "fused accumulate mismatch: want={want:.10}, got={got:.10}, diff={diff:.3e}"
        );

        // Two-call accumulation: same kernel called twice with two
        // different chains should sum into the same slot.
        let chain2: f64 = -0.42;
        let want2 = want + 2.0 * ip.re * chain2;
        let chain_pool2: CudaSlice<f64> = stream.clone_htod(&[chain2]).unwrap();
        let sym_pool2: CudaSlice<u32> = stream.clone_htod(&[0u32]).unwrap();
        nu.inner
            .inner_product_accumulate_via_pool(
                &temp.inner,
                &mut grad_dev,
                &chain_pool2,
                &sym_pool2,
                0,
            )
            .unwrap();
        let grad_host2: Vec<f64> = stream.clone_dtoh(&grad_dev).unwrap();
        stream.synchronize().unwrap();
        let got2 = grad_host2[0];
        let diff2 = (want2 - got2).abs();
        assert!(
            diff2 < 1e-5,
            "fused accumulate (2 calls) mismatch: want={want2:.10}, got={got2:.10}, diff={diff2:.3e}"
        );
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn pooled_param_kernels_match_by_value_kernels() {
        // Cross-check the pooled-param variants against the by-value
        // path on the same gate sequence. Bit-for-bit equality is
        // expected — the kernels do the same arithmetic on the same
        // inputs; only the parameter delivery differs.
        use cudarc::driver::CudaSlice;

        let backend = new_backend();
        let n: u32 = 6;

        // Build a small gate sequence: H + Rx(theta) + CX + Rz(phi).
        let theta = 0.731_f64;
        let phi = -1.234_f64;

        // Reference: by-value path.
        let mut state_ref = backend.allocate(n).expect("alloc ref");
        state_ref.apply_h(0).unwrap();
        state_ref.apply_rx(2, theta).unwrap();
        state_ref.apply_cx(0, 3).unwrap();
        state_ref.apply_rz(4, phi).unwrap();
        let v_ref = state_ref.read_state().unwrap();

        // Pooled: stage every kernel call's params into a pool, then
        // launch via apply_*_via_pool slot lookup.
        let mut state = backend.allocate(n).expect("alloc pooled");
        let stream = state.inner.handle.stream.clone();

        // H = (1/√2)[[1,1],[1,-1]]
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let h_mat = [
            num_complex::Complex64::new(s, 0.0),
            num_complex::Complex64::new(s, 0.0),
            num_complex::Complex64::new(s, 0.0),
            num_complex::Complex64::new(-s, 0.0),
        ];
        // Rx(theta) = [[c, -is], [-is, c]]
        let c = (theta / 2.0).cos();
        let sn = (theta / 2.0).sin();
        let rx_mat = [
            num_complex::Complex64::new(c, 0.0),
            num_complex::Complex64::new(0.0, -sn),
            num_complex::Complex64::new(0.0, -sn),
            num_complex::Complex64::new(c, 0.0),
        ];
        let pool_1q: Vec<imp::Apply1qParams> = vec![
            imp::Apply1qParams::from_matrix(0, &h_mat),
            imp::Apply1qParams::from_matrix(2, &rx_mat),
        ];
        let pool_1q_dev: CudaSlice<imp::Apply1qParams> =
            stream.clone_htod(&pool_1q).expect("upload 1q pool");

        // CX(0,3) — same matrix Metal/CUDA already use.
        let z = num_complex::Complex64::new(0.0, 0.0);
        let o = num_complex::Complex64::new(1.0, 0.0);
        #[rustfmt::skip]
        let cx_mat = [
            o, z, z, z,
            z, z, z, o,
            z, z, o, z,
            z, o, z, z,
        ];
        let pool_2q: Vec<imp::Apply2qParams> = vec![imp::Apply2qParams::from_matrix(0, 3, &cx_mat)];
        let pool_2q_dev: CudaSlice<imp::Apply2qParams> =
            stream.clone_htod(&pool_2q).expect("upload 2q pool");

        // Rz(phi) = diag(e^{-i phi/2}, e^{i phi/2})
        let rz_d0 = num_complex::Complex64::from_polar(1.0, -phi / 2.0);
        let rz_d1 = num_complex::Complex64::from_polar(1.0, phi / 2.0);
        let pool_diag: Vec<imp::DiagonalParams> =
            vec![imp::DiagonalParams::from_complex(4, rz_d0, rz_d1)];
        let pool_diag_dev: CudaSlice<imp::DiagonalParams> =
            stream.clone_htod(&pool_diag).expect("upload diag pool");

        state.inner.apply_1q_via_pool(&pool_1q_dev, 0).unwrap();
        state.inner.apply_1q_via_pool(&pool_1q_dev, 1).unwrap();
        state.inner.apply_2q_via_pool(&pool_2q_dev, 0).unwrap();
        state
            .inner
            .apply_diagonal_via_pool(&pool_diag_dev, 0)
            .unwrap();

        let v = state.read_state().unwrap();
        assert_eq!(v.len(), v_ref.len());
        let max = v
            .iter()
            .zip(v_ref.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(max < 1e-6, "pooled vs by-value max diff = {max:.3e}");
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn shot_sampling_uses_device_probabilities() {
        // Bell pair via the shot-sampling path: must produce a
        // |00⟩+|11⟩ distribution (no |01⟩ or |10⟩) at high shot
        // count. Indirectly confirms the on-device sampler
        // (`sample_counts_on_device` — recursive CDF scan + CURAND
        // + per-shot binary search) is wired into `Backend::execute`
        // correctly.
        use omega_core::circuit::{CircuitType, GateOp, Qubit};

        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        circuit.ops.push(GateOp {
            gate: GateKind::H,
            qubits: smallvec::smallvec![Qubit(0)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.ops.push(GateOp {
            gate: GateKind::CX,
            qubits: smallvec::smallvec![Qubit(0), Qubit(1)],
            params: smallvec::smallvec![],
            classical_bit: None,
            condition: None,
        });

        let backend = new_backend();
        let cfg = ExecConfig {
            shots: Some(8192),
            seed: Some(0x1234_5678),
            ..ExecConfig::default()
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .expect("execute");
        let counts = match result {
            ExecResult::Counts(c) => c,
            _ => panic!("expected counts"),
        };
        // Bell only ever produces |00⟩ (idx 0) or |11⟩ (idx 3).
        let n00 = *counts.get(&0).unwrap_or(&0);
        let n11 = *counts.get(&3).unwrap_or(&0);
        let n_other: u32 = counts
            .iter()
            .filter(|(&k, _)| k != 0 && k != 3)
            .map(|(_, v)| *v)
            .sum();
        assert_eq!(n_other, 0, "Bell shouldn't produce |01⟩ or |10⟩");
        assert_eq!(n00 + n11, 8192);
        // ~50/50 split; conservative tolerance because we want this
        // to be deterministic given the seed.
        let bias = (n00 as i64 - n11 as i64).abs();
        assert!(
            bias < 500,
            "Bell |00⟩ vs |11⟩ should be balanced; got {n00} / {n11} (bias {bias})"
        );
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn on_device_sampler_multi_block_scan_uniform_superposition() {
        // Exercises the two-pass CDF scan: 12-qubit uniform
        // superposition (H on every qubit) has dim=4096 > 1024, so
        // pass-1 spans 4 blocks × 1024 threads. Multinomial check:
        // every basis state should appear and total = shots.
        use omega_core::circuit::{CircuitType, GateOp, Qubit};

        const N: u32 = 12;
        const DIM: u64 = 1u64 << N;
        const SHOTS: u32 = 1 << 20;

        let mut circuit = CircuitIR::new(N, CircuitType::GateBased);
        for q in 0..N {
            circuit.ops.push(GateOp {
                gate: GateKind::H,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }

        let backend = new_backend();
        let cfg = ExecConfig {
            shots: Some(SHOTS),
            seed: Some(0xDEAD_BEEF),
            ..ExecConfig::default()
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .expect("execute");
        let counts = match result {
            ExecResult::Counts(c) => c,
            _ => panic!("expected counts"),
        };
        let total: u32 = counts.values().sum();
        assert_eq!(total, SHOTS);
        assert_eq!(counts.len() as u64, DIM, "uniform covers all bins");
        let expected = SHOTS as f64 / DIM as f64;
        for (k, &c) in &counts {
            assert!(
                (c as f64 - expected).abs() < 160.0,
                "bin {k} count {c} too far from {expected}"
            );
        }
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn on_device_sampler_two_level_scan_n22_uniform() {
        // Exercises the second recursion level of the multi-block
        // scan: n=22 → dim=4M → num_blocks=4096 > 1024 (the
        // single-block threshold). The recursion must scan the
        // 4096-element block_totals via a sub-pass (4 sub-blocks),
        // then a single-block top.
        use omega_core::circuit::{CircuitType, GateOp, Qubit};

        if cudarc::driver::CudaContext::new(0).is_err() {
            eprintln!("skipping: no CUDA device on this host");
            return;
        }

        const N: u32 = 22;
        const DIM: u64 = 1u64 << N;
        const SHOTS: u32 = 2 * 1024 * 1024;

        let mut circuit = CircuitIR::new(N, CircuitType::GateBased);
        for q in 0..N {
            circuit.ops.push(GateOp {
                gate: GateKind::H,
                qubits: smallvec::smallvec![Qubit(q)],
                params: smallvec::smallvec![],
                classical_bit: None,
                condition: None,
            });
        }

        let backend = new_backend();
        let cfg = ExecConfig {
            shots: Some(SHOTS),
            seed: Some(0xC0FFEE),
            ..ExecConfig::default()
        };
        let result = backend
            .execute(&circuit, &ParameterBinding::new(), &cfg)
            .expect("execute");
        let counts = match result {
            ExecResult::Counts(c) => c,
            _ => panic!("expected counts"),
        };
        let total: u32 = counts.values().sum();
        assert_eq!(total, SHOTS);
        for &k in counts.keys() {
            assert!(k < DIM, "basis index {k} out of range");
        }
        // SHOTS/DIM ≈ 0.5 per bin; ~ DIM·(1 − e⁻⁰·⁵) bins observed
        // ≈ 1.65M ± 5%.
        let unique = counts.len() as u64;
        let expected_unique = (DIM as f64 * (1.0 - (-0.5_f64).exp())) as u64;
        let slack = expected_unique / 20;
        assert!(
            unique.abs_diff(expected_unique) < slack,
            "unique={unique}, expected ~{expected_unique} (±{slack})"
        );
    }

    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]
    #[test]
    fn copy_into_replicates_state() {
        let mut a = new_state(4);
        let mut b = new_state(4);
        a.apply_h(0).unwrap();
        a.apply_h(2).unwrap();
        a.copy_into(&mut b).unwrap();
        let va = a.read_state().unwrap();
        let vb = b.read_state().unwrap();
        for (x, y) in va.iter().zip(vb.iter()) {
            assert!((x - y).norm() < 1e-6);
        }
    }
}
