//! macOS+`metal`-feature implementation of the statevector buffer.
//!
//! Only compiled when both `target_os = "macos"` and `feature = "metal"`
//! are set; the parent crate gates this module behind the same `cfg`.
//! Everything `unsafe` in the crate lives here so the public lib stays
//! safe.
//!
//! ## Architecture
//!
//! Two layers:
//! - [`DeviceHandle`] — owns the `MTLDevice`, `MTLCommandQueue`, and
//!   the pre-compiled `KernelLibrary`. Stateless re: circuit
//!   dimensions. Cheap to clone (the metal-rs handles are
//!   refcounted).
//! - [`StateBuffer`] — owns one statevector `MTLBuffer` plus a
//!   per-instance `num_qubits`. Holds its own clones of device/queue/
//!   kernels so it operates without borrowing back to the handle.
//!
//! The `Backend` trait wiring constructs a fresh `StateBuffer` per
//! `execute` call so it can serve any circuit, while direct callers
//! (tests, the to-be-written QML trainer) can keep a `StateBuffer`
//! alive across many gate applications without re-allocating.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Process-global counter for `StateBuffer::read_state` calls.
/// Used by the no-host-syncs regression test to confirm the QML
/// gradient hot path never pulls the full statevector to host:
/// the test reads this counter before/after a training epoch and
/// asserts the delta is zero. The atomic add is sub-ns and runs
/// regardless of build flags; `read_state` itself already pays a
/// `commit + wait_until_completed` and copies `2·dim·f32` bytes
/// host-side, so the fetch_add is in the noise.
pub(crate) static READ_STATE_CALL_COUNT: AtomicU64 = AtomicU64::new(0);

use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder, Device, MTLResourceOptions, MTLSize,
};
use num_complex::Complex64;

use crate::kernels::KernelLibrary;
use crate::{MetalError, MetalStatevectorBackend};

/// Bytes per amplitude when stored as f32 complex (re, im).
const COMPLEX_F32_BYTES: u64 = 8;

/// Stateless device handle — opens MTLDevice + queue + kernels once,
/// hands out fresh `StateBuffer`s per circuit. Cheap to clone.
#[derive(Clone)]
pub(crate) struct DeviceHandle {
    pub device: Device,
    pub queue: CommandQueue,
    pub kernels: KernelLibrary,
}

impl DeviceHandle {
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::Unavailable(
            "MTLDevice::system_default returned None",
        ))?;
        let queue = device.new_command_queue();
        let kernels = KernelLibrary::new(&device)?;
        Ok(Self {
            device,
            queue,
            kernels,
        })
    }

    /// Allocate a fresh statevector buffer initialised to `|0…0⟩`.
    pub fn allocate(&self, num_qubits: u32) -> Result<StateBuffer, MetalError> {
        if num_qubits > MetalStatevectorBackend::MAX_QUBITS {
            return Err(MetalError::AllocationRefused {
                num_qubits,
                reason: "exceeds MAX_QUBITS (would allocate > 2 GiB)",
            });
        }

        let dim: u64 = 1u64 << num_qubits;
        let bytes = dim
            .checked_mul(COMPLEX_F32_BYTES)
            .ok_or(MetalError::AllocationRefused {
                num_qubits,
                reason: "byte size overflow",
            })?;

        let state = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);

        let ptr = state.contents() as *mut u8;
        if ptr.is_null() {
            return Err(MetalError::AllocationRefused {
                num_qubits,
                reason: "MTLBuffer.contents() returned null",
            });
        }
        // Safety: bytes matches the request and ptr is non-null.
        unsafe {
            std::ptr::write_bytes(ptr, 0u8, bytes as usize);
            (ptr as *mut f32).write(1.0);
        }

        Ok(StateBuffer {
            handle: self.clone(),
            state,
            num_qubits,
            batch: RefCell::new(None),
            partials_buf: RefCell::new(None),
        })
    }
}

/// Per-circuit statevector. Holds a clone of the device handle so the
/// gate-apply methods don't need a borrow back to it.
///
/// `batch` controls command-buffer batching: when `Some`, kernel
/// applies append to the open `BatchHandle`'s encoder instead of
/// each kernel building its own cmd_buf and committing+waiting.
/// Set by [`Self::begin_batch`], drained by [`Self::end_batch`]
/// (which `end_encoding` + commit + wait once for the entire run).
/// `RefCell` because the kernel methods take `&self` — the batch
/// state is interior-mutable per buffer instance, never shared.
pub(crate) struct StateBuffer {
    pub handle: DeviceHandle,
    pub state: Buffer,
    pub num_qubits: u32,
    pub(crate) batch: RefCell<Option<BatchHandle>>,
    /// Reusable partials buffer for two-stage reduction kernels
    /// (`inner_product`, `pauli_expectation`). Lazily allocated on
    /// first use; sized for the largest threadgroup count seen so
    /// far. Buffer is refcounted (cheap clone) and shared-mode, so
    /// the kernel writes here directly and the host reads back
    /// without staging. Avoids an `MTLBuffer` allocation per
    /// reduction call — at 81k inner_products per QML bench
    /// iteration the per-call alloc was a measurable fraction of
    /// wallclock.
    pub(crate) partials_buf: RefCell<Option<Buffer>>,
}

/// Open command buffer + compute encoder shared across many kernel
/// dispatches. Each kernel call appends `set_pipeline / set_buffer /
/// set_bytes / dispatch_threads` to the encoder; metal serializes
/// them in encoding order via the implicit memory barrier between
/// dispatches in the same compute encoder. One commit + wait
/// amortizes the per-cmd_buf scheduling overhead across the run.
///
/// `dispatch_count` lets `end_batch` short-circuit when no dispatches
/// were encoded — useful when a batch is opened "just in case" and
/// the caller never actually dispatched anything (e.g.
/// `adjoint_gradient`'s loop pre-opens phi/nu batches at the top
/// and re-opens them after each inner_product flush, but the very
/// first iteration's nu batch may still be empty when inner_product
/// fires). An empty cmd_buf should drop without a commit/wait
/// roundtrip — both the encoder and the cmd_buf release cleanly
/// without scheduling any GPU work.
///
/// **Autoreleasepool note (`adjoint::adjoint_gradient_inner`):**
/// Each `BatchHandle` holds owned references to `MTLCommandBuffer`
/// and `MTLComputeCommandEncoder`, but Metal hands these out as
/// autoreleased Objective-C objects. Long-running loops that open /
/// close many batches without draining a surrounding
/// `objc::rc::autoreleasepool` accumulate these NSObjects under
/// the queue's 64-outstanding cap, and `commandBuffer` eventually
/// blocks forever on its dispatch semaphore. The fix lives at the
/// caller — see `adjoint_gradient_inner`'s per-iteration
/// `drain_autorelease` wrapper.
pub(crate) struct BatchHandle {
    cmd_buf: CommandBuffer,
    encoder: ComputeCommandEncoder,
    dispatch_count: u32,
}

/// Mirrors `DiagonalParams` in `shaders/apply_diagonal.metal`. Layout
/// must stay in sync — kernel reads via `constant DiagonalParams &`.
#[repr(C)]
struct DiagonalParams {
    qubit: u32,
    d0_re: f32,
    d0_im: f32,
    d1_re: f32,
    d1_im: f32,
}

/// Mirrors `DiagonalPauliSumParams` in
/// `shaders/apply_diagonal_pauli_sum.metal`.
#[repr(C)]
struct DiagonalPauliSumParams {
    num_terms: u32,
}

/// Mirrors `DiagonalProductParams` in
/// `shaders/apply_diagonal_product.metal`.
#[repr(C)]
struct DiagonalProductParams {
    num_factors: u32,
}

/// Mirrors `Diagonal2qParams` in `shaders/apply_diagonal_2q.metal`.
/// 4 complex entries (re, im interleaved) covering the 2-qubit
/// diagonal: d00, d01, d10, d11 — index r = bit_qb*2 + bit_qa.
#[repr(C)]
struct Diagonal2qParams {
    qa: u32,
    qb: u32,
    d: [f32; 8],
}

/// Mirrors `Apply1qParams` in `shaders/apply_1q.metal`. The 8 floats
/// after `qubit` are the row-major 2x2 unitary in (re, im) pairs.
#[repr(C)]
struct Apply1qParams {
    qubit: u32,
    u00_re: f32,
    u00_im: f32,
    u01_re: f32,
    u01_im: f32,
    u10_re: f32,
    u10_im: f32,
    u11_re: f32,
    u11_im: f32,
}

/// Mirrors `Apply1qIntoParams` in `shaders/apply_1q_into.metal`.
/// Same payload as `Apply1qParams` but used by the read-src /
/// write-dst kernel variant.
#[repr(C)]
struct Apply1qIntoParams {
    qubit: u32,
    u00_re: f32,
    u00_im: f32,
    u01_re: f32,
    u01_im: f32,
    u10_re: f32,
    u10_im: f32,
    u11_re: f32,
    u11_im: f32,
}

/// Mirrors `DiagonalIntoParams` in `shaders/apply_diagonal_into.metal`.
#[repr(C)]
struct DiagonalIntoParams {
    qubit: u32,
    d0_re: f32,
    d0_im: f32,
    d1_re: f32,
    d1_im: f32,
}

/// Mirrors `Apply2qParams` in `shaders/apply_2q.metal`. 16 complex
/// entries (re, im interleaved) of the row-major 4x4 unitary.
#[repr(C)]
struct Apply2qParams {
    qa: u32,
    qb: u32,
    u: [f32; 32],
}

/// Mirrors `ScanParams` in `shaders/shot_sample.metal`.
#[repr(C)]
struct ScanParams {
    stride: u32,
    dim: u32,
}

/// Mirrors `SampleParams` in `shaders/shot_sample.metal`. Packs the
/// u64 seed as two u32 halves matching the Philox4×32 key shape.
#[repr(C)]
struct SampleParams {
    seed_lo: u32,
    seed_hi: u32,
    shots: u32,
    dim: u32,
}

impl StateBuffer {
    /// Open a kernel-dispatch batch. Subsequent kernel calls (the
    /// in-place applies — `apply_diagonal`, `apply_diagonal_2q`,
    /// `apply_diagonal_product`, `apply_1q`, `apply_2q`) append to
    /// a shared command buffer instead of each call paying its own
    /// `commit + wait_until_completed` roundtrip. Called from
    /// `apply_ops_fused` to wrap the entire forward sweep — at n=18
    /// the HEA bench saves ~120 commit/wait cycles per iteration.
    ///
    /// `end_batch` flushes pending work (`end_encoding` + commit +
    /// wait). `begin_batch` is idempotent if called while a batch is
    /// already open (the existing batch keeps accumulating).
    /// Kernels that need a host-visible result (`inner_product`,
    /// `pauli_expectation`, `read_state`, `apply_diagonal_pauli_sum`'s
    /// dst-write, `copy_into`) are *not* batched — they keep their
    /// self-contained encode+commit+wait shape because the caller
    /// needs the side effect immediately. If they're called while a
    /// batch is open they implicitly flush via `end_batch_if_open`.
    pub fn begin_batch(&self) {
        let mut batch = self.batch.borrow_mut();
        if batch.is_some() {
            // Idempotent: nested begin_batch is a no-op.
            return;
        }
        let cmd_buf = self.handle.queue.new_command_buffer().to_owned();
        let encoder = cmd_buf.new_compute_command_encoder().to_owned();
        *batch = Some(BatchHandle {
            cmd_buf,
            encoder,
            dispatch_count: 0,
        });
    }

    /// Flush any pending batch — `end_encoding` + commit + wait. No-op
    /// if no batch is open. Idempotent. If the open batch has zero
    /// dispatches encoded (`dispatch_count == 0`), skip the commit/wait
    /// roundtrip — just end_encoding and drop.
    pub fn end_batch(&self) {
        let mut batch = self.batch.borrow_mut();
        if let Some(b) = batch.take() {
            b.encoder.end_encoding();
            if b.dispatch_count > 0 {
                b.cmd_buf.commit();
                b.cmd_buf.wait_until_completed();
            }
            // else: no GPU work scheduled; drop the cmd_buf cleanly.
        }
    }

    /// Internal — flush any open batch *before* a non-batched kernel
    /// (`inner_product`, `pauli_expectation`, host-visible reads). The
    /// state buffer is the shared resource; pending writes to it must
    /// land before the next op reads them. Also skips commit/wait if
    /// the batch is empty (mirrors `end_batch`).
    pub(crate) fn end_batch_if_open(&self) {
        let mut batch = self.batch.borrow_mut();
        if let Some(b) = batch.take() {
            b.encoder.end_encoding();
            if b.dispatch_count > 0 {
                b.cmd_buf.commit();
                b.cmd_buf.wait_until_completed();
            }
        }
    }

    /// Lazy-init the cached partials buffer for two-stage reduction
    /// kernels. Returns a cheap-clone of the underlying ObjC handle.
    /// Re-allocates only when the requested size exceeds the current
    /// capacity (so a kernel asking for fewer threadgroups than a
    /// previous call gets the same buffer).
    fn ensure_partials_buf(&self, bytes: u64) -> Buffer {
        let mut slot = self.partials_buf.borrow_mut();
        let needs_alloc = slot.as_ref().is_none_or(|b| b.length() < bytes);
        if needs_alloc {
            *slot = Some(
                self.handle
                    .device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
            );
        }
        slot.as_ref().unwrap().clone()
    }

    /// Run `f(encoder)` on either the open batch's encoder (no
    /// commit) or a fresh self-contained cmd_buf + encoder
    /// (commit + wait). Hides the batched/unbatched distinction
    /// from each kernel method. Increments `dispatch_count` when
    /// appending to a batch so `end_batch` can short-circuit empty
    /// batches.
    fn with_compute_encoder<F>(&self, f: F)
    where
        F: FnOnce(&metal::ComputeCommandEncoderRef),
    {
        let mut batch = self.batch.borrow_mut();
        if let Some(b) = batch.as_mut() {
            f(&b.encoder);
            b.dispatch_count += 1;
        } else {
            drop(batch);
            let cmd_buf = self.handle.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            f(encoder);
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }
    }

    /// Apply a diagonal 1q gate `diag(d0, d1)` on `qubit` in place.
    pub fn apply_diagonal(
        &self,
        qubit: u32,
        d0: Complex64,
        d1: Complex64,
    ) -> Result<(), MetalError> {
        if qubit >= self.num_qubits {
            return Err(MetalError::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        let params = DiagonalParams {
            qubit,
            d0_re: d0.re as f32,
            d0_im: d0.im as f32,
            d1_re: d1.re as f32,
            d1_im: d1.im as f32,
        };
        let dim: u64 = 1u64 << self.num_qubits;

        let pipeline = self.handle.kernels.apply_diagonal.clone();
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        let tg_size = MTLSize {
            width: max_tg.min(dim),
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: dim,
            height: 1,
            depth: 1,
        };
        self.with_compute_encoder(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.state), 0);
            encoder.set_bytes(
                1,
                std::mem::size_of::<DiagonalParams>() as u64,
                (&params as *const DiagonalParams) as *const c_void,
            );
            encoder.dispatch_threads(grid_size, tg_size);
        });
        Ok(())
    }

    /// Diagonal 1q gate `U = diag(d0, d1)` reading from `self`,
    /// writing to `dst`. Sister of `apply_diagonal` (in-place);
    /// used by the adjoint backward sweep's per-parameter dRz /
    /// dU1 derivative apply to skip the prior two-step
    /// `copy_into(self → dst)` followed by `apply_diagonal(dst,
    /// ...)`. The kernel reads `self.state` via the implicit
    /// memory barrier (so any pending daggers in `self`'s open
    /// batch land before the read) and writes the derivative
    /// result directly to `dst`.
    pub fn apply_diagonal_into(
        &self,
        dst: &StateBuffer,
        qubit: u32,
        d0: Complex64,
        d1: Complex64,
    ) -> Result<(), MetalError> {
        if dst.num_qubits != self.num_qubits {
            return Err(MetalError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << dst.num_qubits,
            });
        }
        if qubit >= self.num_qubits {
            return Err(MetalError::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        let params = DiagonalIntoParams {
            qubit,
            d0_re: d0.re as f32,
            d0_im: d0.im as f32,
            d1_re: d1.re as f32,
            d1_im: d1.im as f32,
        };
        let dim: u64 = 1u64 << self.num_qubits;

        let pipeline = self.handle.kernels.apply_diagonal_into.clone();
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        let tg_size = MTLSize {
            width: max_tg.min(dim),
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: dim,
            height: 1,
            depth: 1,
        };
        // Encoded into self's batch (or self-contained). The kernel
        // writes to dst.state — metal cmd_bufs can target any
        // buffer, not just the one whose batch we're using.
        self.with_compute_encoder(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.state), 0);
            encoder.set_buffer(1, Some(&dst.state), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<DiagonalIntoParams>() as u64,
                (&params as *const DiagonalIntoParams) as *const c_void,
            );
            encoder.dispatch_threads(grid_size, tg_size);
        });
        Ok(())
    }

    /// Apply a diagonal 2q gate `U = diag(d00, d01, d10, d11)` to
    /// `(qa, qb)` in place. Index ordering matches `apply_2q`:
    /// row r = bit_qb*2 + bit_qa, so d00 hits the (qa=0, qb=0) branch,
    /// d01 hits (qa=1, qb=0), d10 hits (qa=0, qb=1), d11 hits
    /// (qa=1, qb=1).
    ///
    /// Covers CRz / CZ / dCRz / any diagonal-in-CB 2q gate. Half the
    /// per-amplitude memory traffic vs the generic `apply_2q` matvec
    /// (one complex read+write per amplitude vs four-of-each per
    /// quad), and skips the 4x4 sum-of-products entirely. `qa == qb`
    /// returns `MetalError::QubitsMustDiffer`; out-of-range qubits
    /// return `MetalError::QubitOutOfRange`.
    pub fn apply_diagonal_2q(
        &self,
        qa: u32,
        qb: u32,
        d00: Complex64,
        d01: Complex64,
        d10: Complex64,
        d11: Complex64,
    ) -> Result<(), MetalError> {
        if qa >= self.num_qubits {
            return Err(MetalError::QubitOutOfRange {
                qubit: qa,
                num_qubits: self.num_qubits,
            });
        }
        if qb >= self.num_qubits {
            return Err(MetalError::QubitOutOfRange {
                qubit: qb,
                num_qubits: self.num_qubits,
            });
        }
        if qa == qb {
            return Err(MetalError::DuplicateQubits { qubit: qa });
        }

        let params = Diagonal2qParams {
            qa,
            qb,
            d: [
                d00.re as f32,
                d00.im as f32,
                d01.re as f32,
                d01.im as f32,
                d10.re as f32,
                d10.im as f32,
                d11.re as f32,
                d11.im as f32,
            ],
        };
        let dim: u64 = 1u64 << self.num_qubits;

        let pipeline = self.handle.kernels.apply_diagonal_2q.clone();
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        let tg_size = MTLSize {
            width: max_tg.min(dim),
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: dim,
            height: 1,
            depth: 1,
        };
        self.with_compute_encoder(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.state), 0);
            encoder.set_bytes(
                1,
                std::mem::size_of::<Diagonal2qParams>() as u64,
                (&params as *const Diagonal2qParams) as *const c_void,
            );
            encoder.dispatch_threads(grid_size, tg_size);
        });
        Ok(())
    }

    /// Apply a diagonal Pauli-sum observable `O = Σ_k c_k · Z^{(s_k)}`
    /// to `psi`, writing the result into `dst`. Each term is a
    /// `(sign_mask, coeff)` pair where `sign_mask` has a 1 bit at
    /// every qubit position the Z acts on (identity term has
    /// `sign_mask = 0`). `dst` and `self` must have the same
    /// `num_qubits` and may alias (in-place update).
    ///
    /// Replaces the host-side `O|ψ⟩` initialization in the adjoint
    /// loop's ν setup for QML-style observables (one Z per
    /// measurement qubit). General observables with X/Y components
    /// still need the host-side path.
    pub fn apply_diagonal_pauli_sum(
        &self,
        dst: &StateBuffer,
        terms: &[(u32, f32)],
    ) -> Result<(), MetalError> {
        if dst.num_qubits != self.num_qubits {
            return Err(MetalError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << dst.num_qubits,
            });
        }
        // Two-buffer kernel — flush both source and dst pending work
        // so the read-from-self / write-to-dst semantics match the
        // host-side `apply_observable_host` it replaces.
        self.end_batch_if_open();
        dst.end_batch_if_open();

        let dim: u64 = 1u64 << self.num_qubits;
        if terms.is_empty() {
            // Empty observable: O = 0, so ν = O|ψ⟩ = 0. Zero-fill
            // dst directly — no kernel needed. Matches the host
            // path's behaviour (`apply_observable_host` over an empty
            // term list returns a zero vector).
            let bytes = (dim as usize) * 8;
            let dst_ptr = dst.state.contents() as *mut u8;
            unsafe {
                std::ptr::write_bytes(dst_ptr, 0, bytes);
            }
            return Ok(());
        }
        let params = DiagonalPauliSumParams {
            num_terms: terms.len() as u32,
        };
        let sign_masks: Vec<u32> = terms.iter().map(|(m, _)| *m).collect();
        let coeffs: Vec<f32> = terms.iter().map(|(_, c)| *c).collect();

        let cmd_buf = self.handle.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.handle.kernels.apply_diagonal_pauli_sum);
        encoder.set_buffer(0, Some(&self.state), 0);
        encoder.set_buffer(1, Some(&dst.state), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<DiagonalPauliSumParams>() as u64,
            (&params as *const DiagonalPauliSumParams) as *const c_void,
        );
        encoder.set_bytes(
            3,
            (sign_masks.len() * std::mem::size_of::<u32>()) as u64,
            sign_masks.as_ptr() as *const c_void,
        );
        encoder.set_bytes(
            4,
            (coeffs.len() * std::mem::size_of::<f32>()) as u64,
            coeffs.as_ptr() as *const c_void,
        );

        let max_tg = self
            .handle
            .kernels
            .apply_diagonal_pauli_sum
            .max_total_threads_per_threadgroup();
        let tg_width = max_tg.min(dim);
        let tg_size = MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: dim,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        Ok(())
    }

    /// Apply N independent diagonal 1q gates as a single fused
    /// dispatch. Each `(qubit, d0, d1)` factor multiplies amplitude
    /// `i` by `d0` (qubit-bit 0) or `d1` (qubit-bit 1). All factors
    /// applied in one kernel pass — saves N-1 dispatch round-trips
    /// vs looping `apply_diagonal`.
    ///
    /// Diagonal gates commute, so the factor order doesn't affect
    /// correctness. The shader iterates the factor list per amplitude;
    /// branch divergence on the bit lookup is negligible on Apple GPUs.
    /// Empty `factors` is a no-op (matches `apply_diagonal_pauli_sum`'s
    /// empty-terms convention).
    ///
    /// Caller is responsible for:
    /// - Picking `(d0, d1)` pairs for each qubit's intended gate
    ///   (e.g. `Rz(θ)` → `(e^{-iθ/2}, e^{iθ/2})`).
    /// - Confirming each `qubit` is in `[0, num_qubits)`.
    pub fn apply_diagonal_product(
        &self,
        factors: &[(u32, Complex64, Complex64)],
    ) -> Result<(), MetalError> {
        if factors.is_empty() {
            return Ok(());
        }
        for &(q, _, _) in factors {
            if q >= self.num_qubits {
                return Err(MetalError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: self.num_qubits,
                });
            }
        }

        let dim: u64 = 1u64 << self.num_qubits;
        let params = DiagonalProductParams {
            num_factors: factors.len() as u32,
        };
        let qubits: Vec<u32> = factors.iter().map(|(q, _, _)| *q).collect();
        let d0_factors: Vec<[f32; 2]> = factors
            .iter()
            .map(|(_, d0, _)| [d0.re as f32, d0.im as f32])
            .collect();
        let d1_factors: Vec<[f32; 2]> = factors
            .iter()
            .map(|(_, _, d1)| [d1.re as f32, d1.im as f32])
            .collect();

        let pipeline = self.handle.kernels.apply_diagonal_product.clone();
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        let tg_size = MTLSize {
            width: max_tg.min(dim),
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: dim,
            height: 1,
            depth: 1,
        };
        self.with_compute_encoder(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.state), 0);
            encoder.set_bytes(
                1,
                std::mem::size_of::<DiagonalProductParams>() as u64,
                (&params as *const DiagonalProductParams) as *const c_void,
            );
            encoder.set_bytes(
                2,
                (qubits.len() * std::mem::size_of::<u32>()) as u64,
                qubits.as_ptr() as *const c_void,
            );
            encoder.set_bytes(
                3,
                (d0_factors.len() * std::mem::size_of::<[f32; 2]>()) as u64,
                d0_factors.as_ptr() as *const c_void,
            );
            encoder.set_bytes(
                4,
                (d1_factors.len() * std::mem::size_of::<[f32; 2]>()) as u64,
                d1_factors.as_ptr() as *const c_void,
            );
            encoder.dispatch_threads(grid_size, tg_size);
        });
        Ok(())
    }

    /// Apply a generic 1q unitary `U = [[u00, u01], [u10, u11]]`.
    pub fn apply_1q(&self, qubit: u32, u: &[Complex64; 4]) -> Result<(), MetalError> {
        if qubit >= self.num_qubits {
            return Err(MetalError::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        let params = Apply1qParams {
            qubit,
            u00_re: u[0].re as f32,
            u00_im: u[0].im as f32,
            u01_re: u[1].re as f32,
            u01_im: u[1].im as f32,
            u10_re: u[2].re as f32,
            u10_im: u[2].im as f32,
            u11_re: u[3].re as f32,
            u11_im: u[3].im as f32,
        };
        let pairs: u64 = 1u64 << (self.num_qubits - 1);

        let pipeline = self.handle.kernels.apply_1q.clone();
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        let tg_size = MTLSize {
            width: max_tg.min(pairs),
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: pairs,
            height: 1,
            depth: 1,
        };
        self.with_compute_encoder(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.state), 0);
            encoder.set_bytes(
                1,
                std::mem::size_of::<Apply1qParams>() as u64,
                (&params as *const Apply1qParams) as *const c_void,
            );
            encoder.dispatch_threads(grid_size, tg_size);
        });
        Ok(())
    }

    /// Generic 1q gate reading from `self`, writing to `dst`.
    /// Sister of `apply_1q` (in-place). Used by the adjoint backward
    /// sweep's per-parameter dRx / dRy / dU2 / dU3 derivative apply
    /// to skip the prior `copy_into(self → dst) + apply_1q(dst, ...)`
    /// two-step. Saves one commit+wait per parameter (the host-
    /// memcpy phi flush goes away — the kernel reads phi.state via
    /// the in-encoder memory barrier and writes derivative result
    /// directly to dst).
    pub fn apply_1q_into(
        &self,
        dst: &StateBuffer,
        qubit: u32,
        u: &[Complex64; 4],
    ) -> Result<(), MetalError> {
        if dst.num_qubits != self.num_qubits {
            return Err(MetalError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << dst.num_qubits,
            });
        }
        if qubit >= self.num_qubits {
            return Err(MetalError::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        let params = Apply1qIntoParams {
            qubit,
            u00_re: u[0].re as f32,
            u00_im: u[0].im as f32,
            u01_re: u[1].re as f32,
            u01_im: u[1].im as f32,
            u10_re: u[2].re as f32,
            u10_im: u[2].im as f32,
            u11_re: u[3].re as f32,
            u11_im: u[3].im as f32,
        };
        let pairs: u64 = 1u64 << (self.num_qubits - 1);

        let pipeline = self.handle.kernels.apply_1q_into.clone();
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        let tg_size = MTLSize {
            width: max_tg.min(pairs),
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: pairs,
            height: 1,
            depth: 1,
        };
        self.with_compute_encoder(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.state), 0);
            encoder.set_buffer(1, Some(&dst.state), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<Apply1qIntoParams>() as u64,
                (&params as *const Apply1qIntoParams) as *const c_void,
            );
            encoder.dispatch_threads(grid_size, tg_size);
        });
        Ok(())
    }

    /// Apply a generic 2q unitary `U` (4x4 row-major, index =
    /// bit_qb*2 + bit_qa) to qubits (qa, qb).
    pub fn apply_2q(&self, qa: u32, qb: u32, u: &[Complex64; 16]) -> Result<(), MetalError> {
        if qa >= self.num_qubits {
            return Err(MetalError::QubitOutOfRange {
                qubit: qa,
                num_qubits: self.num_qubits,
            });
        }
        if qb >= self.num_qubits {
            return Err(MetalError::QubitOutOfRange {
                qubit: qb,
                num_qubits: self.num_qubits,
            });
        }
        if qa == qb {
            return Err(MetalError::DuplicateQubits { qubit: qa });
        }
        if self.num_qubits < 2 {
            return Err(MetalError::QubitOutOfRange {
                qubit: qb,
                num_qubits: self.num_qubits,
            });
        }
        let mut u_flat = [0.0f32; 32];
        for (k, c) in u.iter().enumerate() {
            u_flat[2 * k] = c.re as f32;
            u_flat[2 * k + 1] = c.im as f32;
        }
        let params = Apply2qParams { qa, qb, u: u_flat };
        let quads: u64 = 1u64 << (self.num_qubits - 2);

        let pipeline = self.handle.kernels.apply_2q.clone();
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        let tg_size = MTLSize {
            width: max_tg.min(quads),
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: quads,
            height: 1,
            depth: 1,
        };
        self.with_compute_encoder(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.state), 0);
            encoder.set_bytes(
                1,
                std::mem::size_of::<Apply2qParams>() as u64,
                (&params as *const Apply2qParams) as *const c_void,
            );
            encoder.dispatch_threads(grid_size, tg_size);
        });
        Ok(())
    }

    /// Compute ⟨self|other⟩ on the GPU via a two-stage threadgroup
    /// reduction. The kernel writes one float2 partial per
    /// threadgroup; the host sums the (small) partial array. Both
    /// states must have the same `num_qubits`.
    pub fn inner_product(&self, other: &StateBuffer) -> Result<Complex64, MetalError> {
        if self.num_qubits != other.num_qubits {
            return Err(MetalError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << other.num_qubits,
            });
        }
        let dim: u64 = 1u64 << self.num_qubits;
        let pipeline = self.handle.kernels.inner_product.clone();
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        // Pick the largest power-of-two threadgroup size ≤ max_tg and
        // ≤ dim. Power-of-two is required by the in-shader reduction.
        let mut tg_width: u64 = 1;
        while tg_width * 2 <= max_tg.min(dim) {
            tg_width *= 2;
        }
        let num_tgs = dim.div_ceil(tg_width);

        let partials_bytes = num_tgs * 8; // float2 = 8 bytes
        let partials_buf = self.ensure_partials_buf(partials_bytes);
        let tg_size = MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: dim,
            height: 1,
            depth: 1,
        };

        // Encode the reduction dispatch into either an open batch on
        // one of the two states, or a fresh cmd_buf. Riding on an
        // existing batch means the partial dispatches inside it
        // (e.g. the per-param `apply_op_derivative` write to
        // `temp_state`) and the reduction read coalesce into one
        // commit + wait — saves one host↔GPU sync per param in the
        // adjoint backward sweep.
        //
        // Defensive: if BOTH states have open batches we'd need to
        // merge them, which Metal doesn't support across cmd_bufs.
        // Close `other` first in that case so we can ride on `self`.
        let encode = |encoder: &metal::ComputeCommandEncoderRef| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&self.state), 0);
            encoder.set_buffer(1, Some(&other.state), 0);
            encoder.set_buffer(2, Some(&partials_buf), 0);
            // Threadgroup memory: tg_width float2 entries = 8 bytes each.
            encoder.set_threadgroup_memory_length(0, tg_width * 8);
            encoder.dispatch_threads(grid_size, tg_size);
        };

        let self_open = self.batch.borrow().is_some();
        let other_open = other.batch.borrow().is_some();

        if self_open && other_open {
            // Two separate cmd_bufs in flight — close `other` so we
            // can ride on `self`'s. (Round 14 + later: nu may carry
            // accumulated daggers from earlier param-less iterations
            // when the QML adjoint loop pre-opens phi/nu batches; in
            // that case round 13's "ride on self=nu" path commits
            // those daggers in the same cmd_buf as the reduction.)
            other.end_batch_if_open();
        }

        let self_open = self.batch.borrow().is_some();
        let other_open = other.batch.borrow().is_some();

        if self_open {
            let mut slot = self.batch.borrow_mut();
            let handle = slot.take().unwrap();
            encode(&handle.encoder);
            handle.encoder.end_encoding();
            // The batch always commits here — we just appended the
            // reduction dispatch, which counts. Whether the batch
            // had pre-existing daggers (`dispatch_count` > 0) or
            // was opened empty doesn't matter: the reduction itself
            // is now scheduled and the partials read needs the wait.
            handle.cmd_buf.commit();
            handle.cmd_buf.wait_until_completed();
        } else if other_open {
            let mut slot = other.batch.borrow_mut();
            let handle = slot.take().unwrap();
            encode(&handle.encoder);
            handle.encoder.end_encoding();
            handle.cmd_buf.commit();
            handle.cmd_buf.wait_until_completed();
        } else {
            let cmd_buf = self.handle.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encode(encoder);
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }

        // Sum the partials host-side.
        let ptr = partials_buf.contents() as *const f32;
        let mut acc_re: f64 = 0.0;
        let mut acc_im: f64 = 0.0;
        unsafe {
            for i in 0..num_tgs as usize {
                acc_re += ptr.add(2 * i).read() as f64;
                acc_im += ptr.add(2 * i + 1).read() as f64;
            }
        }
        Ok(Complex64::new(acc_re, acc_im))
    }

    /// Fused Pauli-string expectation `⟨ψ|P|ψ⟩` in one kernel
    /// dispatch. Skips the previous "clone state → apply σ → inner
    /// product" trio: per-thread, conj(ψ[i]) · sign · ψ[i XOR x_mask]
    /// with `sign = (-1)^popcount(i & sign_mask)`, then a global
    /// `(-i)^{|Y|}` prefactor folded in via `y_factor`.
    ///
    /// `x_mask`, `sign_mask`, and `y_factor` are computed by the
    /// caller (see `lib.rs::expectation` and `lib.rs::pauli_masks`).
    pub fn pauli_expectation(
        &self,
        x_mask: u32,
        sign_mask: u32,
        y_factor: Complex64,
    ) -> Result<Complex64, MetalError> {
        // Reduction reads `self.state` — flush pending writes first.
        self.end_batch_if_open();
        let dim: u64 = 1u64 << self.num_qubits;
        let pipeline = &self.handle.kernels.pauli_expectation;
        let max_tg = pipeline.max_total_threads_per_threadgroup();
        let mut tg_width: u64 = 1;
        while tg_width * 2 <= max_tg.min(dim) {
            tg_width *= 2;
        }
        let num_tgs = dim.div_ceil(tg_width);

        let partials_bytes = num_tgs * 8;
        let partials_buf = self.ensure_partials_buf(partials_bytes);

        let cmd_buf = self.handle.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.state), 0);
        encoder.set_buffer(1, Some(&partials_buf), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &x_mask as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &sign_mask as *const u32 as *const c_void,
        );
        let yf: [f32; 2] = [y_factor.re as f32, y_factor.im as f32];
        encoder.set_bytes(
            4,
            (std::mem::size_of::<f32>() * 2) as u64,
            yf.as_ptr() as *const c_void,
        );
        encoder.set_threadgroup_memory_length(0, tg_width * 8);

        let tg_size = MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: dim,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let ptr = partials_buf.contents() as *const f32;
        let mut acc_re: f64 = 0.0;
        let mut acc_im: f64 = 0.0;
        unsafe {
            for i in 0..num_tgs as usize {
                acc_re += ptr.add(2 * i).read() as f64;
                acc_im += ptr.add(2 * i + 1).read() as f64;
            }
        }
        Ok(Complex64::new(acc_re, acc_im))
    }

    /// GPU-resident shot sampling. Pipeline: (1) `shot_probs` writes
    /// `|state[i]|²` per amplitude. (2) `shot_scan_step` runs a
    /// Hillis-Steele inclusive prefix-sum, ping-ponging over two device
    /// buffers across `⌈log₂(dim)⌉` passes; one of the buffers holds
    /// the CDF after the loop. (3) `shot_sample` spawns one thread per
    /// shot, draws a uniform via Philox4×32 keyed by the host-supplied
    /// seed, and binary-searches the CDF for the outcome bin.
    ///
    /// Returns the per-outcome counts. The only host↔device sync is
    /// the final `outcomes` read (`shots × 4` bytes), so this avoids
    /// the `2·dim·f32`-byte full-statevector pull a host-side sampler
    /// would do.
    ///
    /// Acceptance gate (`tests/shot_sampling_tvd.rs`): empirical TVD
    /// vs the analytical `|amp|²` distribution stays inside the
    /// statistical floor at shots ≥ 10⁴; f32 amplitude precision is
    /// the dominant error source.
    pub fn sample_shots_gpu(&self, shots: u32, seed: u64) -> Result<HashMap<u64, u32>, MetalError> {
        // Flush pending writes — kernel chain reads `self.state`.
        self.end_batch_if_open();

        if shots == 0 {
            return Ok(HashMap::new());
        }

        let dim: u64 = 1u64 << self.num_qubits;
        let probs_bytes = dim * 4; // f32 per amplitude
        let outcomes_bytes = (shots as u64) * 4; // u32 per shot

        let dev = &self.handle.device;
        let probs_a = dev.new_buffer(probs_bytes, MTLResourceOptions::StorageModeShared);
        let probs_b = dev.new_buffer(probs_bytes, MTLResourceOptions::StorageModeShared);
        let outcomes = dev.new_buffer(outcomes_bytes, MTLResourceOptions::StorageModeShared);

        let cmd_buf = self.handle.queue.new_command_buffer();

        // Stage 1: |amp|² → probs_a.
        {
            let pipeline = &self.handle.kernels.shot_probs;
            let max_tg = pipeline.max_total_threads_per_threadgroup();
            let tg = MTLSize {
                width: max_tg.min(dim),
                height: 1,
                depth: 1,
            };
            let grid = MTLSize {
                width: dim,
                height: 1,
                depth: 1,
            };
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&self.state), 0);
            encoder.set_buffer(1, Some(&probs_a), 0);
            encoder.dispatch_threads(grid, tg);
            encoder.end_encoding();
        }

        // Stage 2: Hillis-Steele scan. n passes for dim = 2^n. For
        // dim = 1 (num_qubits = 0) the loop body doesn't execute and
        // probs_a already trivially holds the singleton CDF.
        let num_passes = self.num_qubits;
        let mut active_is_a = true;
        for pass in 0..num_passes {
            let stride: u32 = 1u32 << pass;
            let p = ScanParams {
                stride,
                dim: dim as u32,
            };
            let (in_buf, out_buf) = if active_is_a {
                (&probs_a, &probs_b)
            } else {
                (&probs_b, &probs_a)
            };
            let pipeline = &self.handle.kernels.shot_scan_step;
            let max_tg = pipeline.max_total_threads_per_threadgroup();
            let tg = MTLSize {
                width: max_tg.min(dim),
                height: 1,
                depth: 1,
            };
            let grid = MTLSize {
                width: dim,
                height: 1,
                depth: 1,
            };
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(in_buf), 0);
            encoder.set_buffer(1, Some(out_buf), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<ScanParams>() as u64,
                (&p as *const ScanParams) as *const c_void,
            );
            encoder.dispatch_threads(grid, tg);
            encoder.end_encoding();
            active_is_a = !active_is_a;
        }
        let cdf = if active_is_a { &probs_a } else { &probs_b };

        // Stage 3: Philox sample.
        {
            let pipeline = &self.handle.kernels.shot_sample;
            let max_tg = pipeline.max_total_threads_per_threadgroup();
            let shots_u64 = shots as u64;
            let tg = MTLSize {
                width: max_tg.min(shots_u64),
                height: 1,
                depth: 1,
            };
            let grid = MTLSize {
                width: shots_u64,
                height: 1,
                depth: 1,
            };
            let p = SampleParams {
                seed_lo: seed as u32,
                seed_hi: (seed >> 32) as u32,
                shots,
                dim: dim as u32,
            };
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(cdf), 0);
            encoder.set_buffer(1, Some(&outcomes), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<SampleParams>() as u64,
                (&p as *const SampleParams) as *const c_void,
            );
            encoder.dispatch_threads(grid, tg);
            encoder.end_encoding();
        }

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Aggregate outcomes into a HashMap host-side. shots × 4 bytes.
        let outcomes_ptr = outcomes.contents() as *const u32;
        let mut counts: HashMap<u64, u32> = HashMap::new();
        // Safety: outcomes is `shots * 4` bytes, shared-mode, written
        // exclusively by the `shot_sample` kernel which has completed
        // (we waited on the cmd_buf). Each thread tid in [0, shots)
        // wrote exactly once to outcomes[tid].
        unsafe {
            for i in 0..shots as usize {
                let o = outcomes_ptr.add(i).read() as u64;
                *counts.entry(o).or_insert(0) += 1;
            }
        }
        Ok(counts)
    }

    pub fn read_state(&self) -> Vec<Complex64> {
        // Counter increments before the flush + memcpy so a test
        // observing `READ_STATE_CALL_COUNT` sees this call even if
        // the loop is interrupted mid-copy. Relaxed ordering — the
        // counter is informational, not used for control flow.
        READ_STATE_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        // Host read — flush any pending GPU writes first.
        self.end_batch_if_open();
        let dim = 1usize << self.num_qubits;
        let ptr = self.state.contents() as *const f32;
        let mut out = Vec::with_capacity(dim);
        // Safety: backing buffer is `2 * dim * f32` bytes, the same
        // layout we wrote on construction / via write_state.
        unsafe {
            for i in 0..dim {
                let re = ptr.add(2 * i).read() as f64;
                let im = ptr.add(2 * i + 1).read() as f64;
                out.push(Complex64::new(re, im));
            }
        }
        out
    }

    pub fn write_state(&self, state: &[Complex64]) -> Result<(), MetalError> {
        let dim = 1usize << self.num_qubits;
        if state.len() != dim {
            return Err(MetalError::StateLengthMismatch {
                expected: dim,
                got: state.len(),
            });
        }
        // Host write — flush pending GPU work so we don't race the
        // memcpy against an in-flight kernel.
        self.end_batch_if_open();
        let ptr = self.state.contents() as *mut f32;
        // Safety: same layout invariant as read_state.
        unsafe {
            for (i, amp) in state.iter().enumerate() {
                ptr.add(2 * i).write(amp.re as f32);
                ptr.add(2 * i + 1).write(amp.im as f32);
            }
        }
        Ok(())
    }

    /// Re-initialise the statevector to `|0…0⟩` in place. Host-side
    /// memset over the shared-mode `MTLBuffer`; no kernel dispatch.
    /// Used by [`BufferPool::lease`] to recycle pooled buffers between
    /// the QML trainer's per-training-point adjoint sweeps.
    pub fn reset_to_zero(&self) {
        // Host write — flush any pending kernel that targets this
        // buffer first so the memset doesn't race in-flight work.
        self.end_batch_if_open();
        let dim = 1usize << self.num_qubits;
        let bytes = dim * COMPLEX_F32_BYTES as usize;
        let ptr = self.state.contents() as *mut u8;
        // Safety: shared-mode buffer of `2*dim*4 = bytes` bytes,
        // matching `num_qubits`. Mirrors the post-allocation
        // initialisation in `DeviceHandle::allocate`.
        unsafe {
            std::ptr::write_bytes(ptr, 0u8, bytes);
            (ptr as *mut f32).write(1.0);
        }
    }
}

/// LIFO buffer pool keyed by `num_qubits`. Recycles `StateBuffer`s
/// across `MetalStatevectorBackend` calls so the QML trainer's hot
/// path (`adjoint_gradient` × N training points × M epochs) doesn't
/// pay a fresh `MTLBuffer` allocation per call.
///
/// Per-`adjoint_gradient` invocation needs three buffers (forward
/// |φ⟩, adjoint |ν⟩, scratch derivative state) — at 32 training
/// points × 100 epochs that's ~9600 allocations otherwise. The pool
/// converges to a steady-state size of 3 per `num_qubits` after the
/// first epoch.
///
/// Buffers are reset to `|0…0⟩` on lease (cheap host memset on a
/// shared-mode `MTLBuffer`); on drop of the leasing [`MetalState`]
/// they're returned to the pool. Different qubit counts route to
/// independent stacks — leasing 14q never reuses a pooled 16q
/// buffer (the byte sizes differ).
pub(crate) struct BufferPool {
    entries: Mutex<HashMap<u32, Vec<StateBuffer>>>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Lease a `StateBuffer` for `num_qubits` qubits. Pulls from the
    /// pooled stack when available (resetting to `|0…0⟩`); otherwise
    /// allocates a fresh buffer through `handle`.
    pub fn lease(&self, handle: &DeviceHandle, num_qubits: u32) -> Result<StateBuffer, MetalError> {
        {
            let mut entries = self.entries.lock().expect("BufferPool mutex poisoned");
            if let Some(stack) = entries.get_mut(&num_qubits) {
                if let Some(buf) = stack.pop() {
                    drop(entries);
                    buf.reset_to_zero();
                    return Ok(buf);
                }
            }
        }
        handle.allocate(num_qubits)
    }

    /// Return a `StateBuffer` to the pool. Called by `MetalState::drop`
    /// when `pool_return` is set.
    pub fn return_buffer(&self, buf: StateBuffer) {
        let n = buf.num_qubits;
        let mut entries = self.entries.lock().expect("BufferPool mutex poisoned");
        entries.entry(n).or_default().push(buf);
    }

    /// Inspect the number of pooled buffers for a given qubit count.
    /// Test-only — unit tests use it to assert pool semantics.
    #[cfg(test)]
    pub fn pooled_count(&self, num_qubits: u32) -> usize {
        let entries = self.entries.lock().expect("BufferPool mutex poisoned");
        entries.get(&num_qubits).map(|v| v.len()).unwrap_or(0)
    }
}
