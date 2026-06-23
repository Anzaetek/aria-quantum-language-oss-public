//! Linux/Windows + `cuda`-feature implementation of the statevector
//! buffer.
//!
//! Only compiled when both `target_os = "linux"|"windows"` and
//! `feature = "cuda"` are set; the parent crate gates this module
//! behind the same `cfg`.
//!
//! Mirrors `omega-backend-statevector-metal/src/imp.rs`. Two layers:
//! - [`DeviceHandle`] — owns the [`CudaContext`], default
//!   [`CudaStream`], and the pre-compiled [`KernelLibrary`]. Cheap to
//!   clone (`Arc` everything).
//! - [`StateBuffer`] — owns one statevector [`CudaSlice<f32>`] (laid
//!   out as interleaved (re, im) f32 pairs, identical to Metal so the
//!   shared CPU gate-derivative builders carry over) plus a
//!   per-instance `num_qubits`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cudarc::driver::sys::CUresult;
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};
use num_complex::Complex64;

use crate::kernels::KernelLibrary;
use crate::{CudaError, CudaStatevectorBackend};

/// Process-global counter for `StateBuffer::read_state` calls. Mirrors
/// the Metal backend's counter — `read_state` is the only CUDA path
/// that pulls the full `2·dim·f32` device buffer back to host, so a
/// zero-delta across a QML training epoch certifies the gradient hot
/// path is host-sync-free at that granularity. The atomic add is
/// sub-ns; the dtoh memcpy + stream synchronize that `read_state`
/// itself pays dwarf it. Relaxed ordering — informational only.
pub(crate) static READ_STATE_CALL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Map a cudarc `DriverError` from an allocation call into the right
/// `CudaError` variant: `CUDA_ERROR_OUT_OF_MEMORY` becomes
/// `CudaError::OutOfMemory` (so the trainer can fall back to CPU),
/// everything else stays in `CudaError::Driver` (an opaque hard
/// failure).
fn driver_error_to_cuda_error(num_qubits: u32, op: &str, e: DriverError) -> CudaError {
    if e.0 == CUresult::CUDA_ERROR_OUT_OF_MEMORY {
        CudaError::OutOfMemory {
            num_qubits,
            reason: format!("{op}: {e}"),
        }
    } else {
        CudaError::Driver(format!("{op}: {e}"))
    }
}

/// Bytes per amplitude when stored as f32 complex (re, im).
const COMPLEX_F32_BYTES: u64 = 8;

/// Cap blocks at 256 threads — comfortably within the 1024 hard limit
/// and matches typical occupancy for memory-bound kernels.
const BLOCK_THREADS: u32 = 256;

/// Stateless device handle — opens CUDA context + stream + kernels
/// once, hands out fresh [`StateBuffer`]s per circuit.
#[derive(Clone)]
pub(crate) struct DeviceHandle {
    // Holding the Arc<CudaContext> keeps the device context alive
    // for the lifetime of every StateBuffer cloned off this handle —
    // dropping it would invalidate the loaded modules referenced by
    // `kernels`. Not read directly outside of debugging.
    #[allow(dead_code)]
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub kernels: KernelLibrary,
}

impl DeviceHandle {
    pub fn new() -> Result<Self, CudaError> {
        let ctx = CudaContext::new(0)
            .map_err(|e| CudaError::Driver(format!("CudaContext::new(0) failed: {e}")))?;
        let stream = ctx.default_stream();
        let kernels = KernelLibrary::new(&ctx)?;
        Ok(Self {
            ctx,
            stream,
            kernels,
        })
    }

    /// Allocate a fresh statevector buffer initialised to `|0…0⟩`.
    /// Layout: 2·dim f32 (re, im interleaved) — identical to the
    /// Metal backend.
    pub fn allocate(&self, num_qubits: u32) -> Result<StateBuffer, CudaError> {
        if num_qubits > CudaStatevectorBackend::MAX_QUBITS {
            return Err(CudaError::AllocationRefused {
                num_qubits,
                reason: "exceeds MAX_QUBITS (would allocate > 2 GiB)",
            });
        }

        let dim: u64 = 1u64 << num_qubits;
        let _bytes = dim
            .checked_mul(COMPLEX_F32_BYTES)
            .ok_or(CudaError::AllocationRefused {
                num_qubits,
                reason: "byte size overflow",
            })?;

        // Length in f32 (re, im) — twice the amplitude count.
        let len_f32 = (dim as usize) * 2;
        let mut state = self
            .stream
            .alloc_zeros::<f32>(len_f32)
            .map_err(|e| driver_error_to_cuda_error(num_qubits, "alloc_zeros", e))?;
        // Set state[0] = 1.0+0i — the |0…0⟩ amplitude. We do this by
        // copying a single 1.0 into the first f32 (the real part).
        let one = [1.0f32];
        {
            let mut head = state.slice_mut(0..1);
            self.stream
                .memcpy_htod(&one[..], &mut head)
                .map_err(|e| CudaError::Driver(format!("memcpy_htod init: {e}")))?;
        }

        Ok(StateBuffer {
            handle: self.clone(),
            state,
            num_qubits,
        })
    }
}

/// Per-circuit statevector. Holds a clone of the device handle so the
/// gate-apply methods don't need to borrow back to it.
pub(crate) struct StateBuffer {
    pub handle: DeviceHandle,
    pub state: CudaSlice<f32>,
    pub num_qubits: u32,
}

// ---- Kernel parameter structs --------------------------------------
//
// Each #[repr(C)] struct mirrors the matching `extern "C" struct` in
// the .cu source. cudarc's `PushKernelArg<&T>` impl for `T: DeviceRepr`
// passes a pointer to the struct via the kernel argument list, and
// CUDA copies it by value into the kernel parameter buffer at launch.

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct DiagonalParams {
    pub qubit: u32,
    pub d0_re: f32,
    pub d0_im: f32,
    pub d1_re: f32,
    pub d1_im: f32,
}
unsafe impl DeviceRepr for DiagonalParams {}
unsafe impl ValidAsZeroBits for DiagonalParams {}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DiagonalPauliSumParams {
    pub num_terms: u32,
}
unsafe impl DeviceRepr for DiagonalPauliSumParams {}
unsafe impl ValidAsZeroBits for DiagonalPauliSumParams {}

/// Stage G6: per-term entry for the `pauli_z_chain_accumulate`
/// kernel. The chain pool holds one entry per Pauli-Z gradient
/// term (one per RZ in the trailing chain on Phase 4c). Each entry
/// records the qubit `Z_q` acts on, the destination grad_dev slot
/// `sym`, and the autodiff chain factor `chain` for the term —
/// updated host-side per replay just like the existing chain_pool.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct PauliZChainEntry {
    pub qubit: u32,
    pub sym: u32,
    pub chain: f64,
}
unsafe impl DeviceRepr for PauliZChainEntry {}
unsafe impl ValidAsZeroBits for PauliZChainEntry {}

#[repr(C)]
#[derive(Clone, Copy)]
struct DiagonalProductParams {
    num_factors: u32,
}
unsafe impl DeviceRepr for DiagonalProductParams {}
unsafe impl ValidAsZeroBits for DiagonalProductParams {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Apply1qParams {
    pub qubit: u32,
    pub u00_re: f32,
    pub u00_im: f32,
    pub u01_re: f32,
    pub u01_im: f32,
    pub u10_re: f32,
    pub u10_im: f32,
    pub u11_re: f32,
    pub u11_im: f32,
}
unsafe impl DeviceRepr for Apply1qParams {}
unsafe impl ValidAsZeroBits for Apply1qParams {}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Apply2qParams {
    pub qa: u32,
    pub qb: u32,
    pub u: [f32; 32],
}
impl Default for Apply2qParams {
    fn default() -> Self {
        Self {
            qa: 0,
            qb: 1,
            u: [0.0; 32],
        }
    }
}
unsafe impl DeviceRepr for Apply2qParams {}
unsafe impl ValidAsZeroBits for Apply2qParams {}

// The following constructors + apply_*_via_pool methods are
// consumed by the upcoming ForwardGraph slice. Until that wiring
// lands clippy sees them as unused — silence the lib gate; the
// integration test in lib.rs's tests module exercises every one.
#[allow(dead_code)]
impl DiagonalParams {
    pub fn from_complex(qubit: u32, d0: Complex64, d1: Complex64) -> Self {
        Self {
            qubit,
            d0_re: d0.re as f32,
            d0_im: d0.im as f32,
            d1_re: d1.re as f32,
            d1_im: d1.im as f32,
        }
    }
}

#[allow(dead_code)]
impl Apply1qParams {
    pub fn from_matrix(qubit: u32, u: &[Complex64; 4]) -> Self {
        Self {
            qubit,
            u00_re: u[0].re as f32,
            u00_im: u[0].im as f32,
            u01_re: u[1].re as f32,
            u01_im: u[1].im as f32,
            u10_re: u[2].re as f32,
            u10_im: u[2].im as f32,
            u11_re: u[3].re as f32,
            u11_im: u[3].im as f32,
        }
    }
}

#[allow(dead_code)]
impl Apply2qParams {
    pub fn from_matrix(qa: u32, qb: u32, u: &[Complex64; 16]) -> Self {
        let mut flat = [0.0f32; 32];
        for (k, c) in u.iter().enumerate() {
            flat[2 * k] = c.re as f32;
            flat[2 * k + 1] = c.im as f32;
        }
        Self { qa, qb, u: flat }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Float2 {
    re: f32,
    im: f32,
}
unsafe impl DeviceRepr for Float2 {}
unsafe impl ValidAsZeroBits for Float2 {}

/// Sync-deferred inner_product result. The wrapped host `Vec<f32>`
/// is filled asynchronously by the GPU after
/// [`StateBuffer::inner_product_deferred`] returns; reading it before
/// the owning stream is synchronized is UB. The only way to extract
/// the scalar `Complex64` is via [`Self::reduce`], which the caller
/// must invoke *after* `stream.synchronize()` has run (either
/// explicitly or via another host-blocking op on the same stream).
#[must_use = "PendingInnerProduct must be reduced after a stream sync; dropping it discards the partial buffer mid-flight"]
pub(crate) struct PendingInnerProduct {
    host: Vec<f32>,
    // Keep the device-side partials buffer alive past the
    // `inner_product_deferred` call. cudarc's `CudaSlice` drop is
    // safe re: in-flight memcpys (it uses `free_async` on hosts
    // that support memory pools, and `synchronize+free_sync`
    // otherwise) — but on the no-async-alloc path the implicit
    // sync defeats the deferral. Holding the slice in this struct
    // means it doesn't drop until `reduce()` consumes self, which
    // happens AFTER the caller has done its single end-of-loop
    // synchronize().
    _partials_dev: CudaSlice<f32>,
}

impl PendingInnerProduct {
    /// Sum the partials host-side. Caller is responsible for ensuring
    /// the stream that wrote the partials has synchronized — otherwise
    /// the bytes read here are uninitialized and the result is UB.
    pub fn reduce(self) -> Complex64 {
        let mut acc_re: f64 = 0.0;
        let mut acc_im: f64 = 0.0;
        for i in (0..self.host.len()).step_by(2) {
            acc_re += self.host[i] as f64;
            acc_im += self.host[i + 1] as f64;
        }
        Complex64::new(acc_re, acc_im)
    }
}

/// Recursive multi-block inclusive prefix-sum (CDF) of `data[0..len]`
/// on device. Composes the `cdf_block_scan_pass1` +
/// `cdf_block_scan_pass2_from_inclusive` kernels.
///
/// At each level:
///   - If the array fits in one block, `pass1` alone produces a
///     full inclusive scan and we're done.
///   - Otherwise: pass1 produces per-block inclusive scans + a
///     `block_totals` array; we recurse on `block_totals` (so it
///     too becomes inclusive); then pass2 adds the prior-block
///     total to every element.
///
/// Recursion depth is `log_BLOCK_SIZE(dim)` — for BLOCK_SIZE=1024:
/// depth 1 covers dim ≤ 1024 (n ≤ 10), depth 2 covers dim ≤ 1024²
/// (n ≤ 20), depth 3 covers dim ≤ 1024³ = 2³⁰ (n ≤ 30). Bounded
/// well above any practical statevector size.
fn cdf_inclusive_scan_recursive(
    handle: &DeviceHandle,
    data: &mut CudaSlice<f32>,
    len: u64,
    block_size: u32,
) -> Result<(), CudaError> {
    if len == 0 {
        return Ok(());
    }
    let stream = &handle.stream;
    let num_blocks_u64 = len.div_ceil(block_size as u64);
    let num_blocks: u32 = num_blocks_u64
        .try_into()
        .map_err(|_| CudaError::Driver(format!("num_blocks {num_blocks_u64} overflows u32")))?;

    // `block_totals` of length `num_blocks` accumulates per-block
    // inclusive sums. When `num_blocks == 1` we still allocate one
    // slot because pass1 always writes the trailing block total.
    let mut block_totals = stream
        .alloc_zeros::<f32>(num_blocks as usize)
        .map_err(|e| CudaError::Driver(format!("alloc block_totals: {e}")))?;

    // Pass 1 — per-block inclusive scan.
    {
        let func = handle.kernels.cdf_block_scan_pass1.clone();
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: block_size * 4,
        };
        let mut builder = stream.launch_builder(&func);
        builder.arg(&mut *data).arg(&mut block_totals).arg(&len);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch cdf_block_scan_pass1: {e}")))?;
        }
    }

    if num_blocks == 1 {
        // Single block — pass1 alone is the full inclusive scan.
        return Ok(());
    }

    // Recurse so `block_totals` holds the inclusive scan of itself.
    cdf_inclusive_scan_recursive(handle, &mut block_totals, num_blocks_u64, block_size)?;

    // Pass 2 — add the prior block's inclusive total back into each
    // element. `block_totals` is read as INCLUSIVE; the kernel
    // converts to the exclusive offset (`block_totals[b-1]` for
    // `b > 0`, 0 for `b == 0`).
    {
        let func = handle.kernels.cdf_block_scan_pass2_from_inclusive.clone();
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder.arg(&mut *data).arg(&block_totals).arg(&len);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                CudaError::Driver(format!("launch cdf_block_scan_pass2_from_inclusive: {e}"))
            })?;
        }
    }

    Ok(())
}

/// Pick `(grid_dim, block_dim)` for a kernel that should launch
/// `total` threads. block_dim is a power of two ≤ [`BLOCK_THREADS`]
/// so the in-kernel reduction (which assumes power-of-two `blockDim.x`)
/// stays correct.
fn launch_dims(total: u64) -> (u32, u32) {
    let block: u32 = if total >= BLOCK_THREADS as u64 {
        BLOCK_THREADS
    } else {
        // Largest power-of-two ≤ total, ≥ 1.
        let mut b: u32 = 1;
        while (b as u64) * 2 <= total {
            b *= 2;
        }
        b.max(1)
    };
    let grid = total.div_ceil(block as u64);
    let grid: u32 = grid.try_into().expect("grid_dim overflow");
    (grid, block)
}

impl StateBuffer {
    /// On-device shot sampler. Replaces the host CDF + per-shot
    /// loop with:
    ///
    /// 1. `compute_probabilities` (existing) → per-amp f32 `|ψ|²`
    /// 2. Recursive multi-block inclusive scan
    ///    (`cdf_inclusive_scan_recursive`) → CDF in place
    /// 3. CURAND `fill_with_uniform` → `shots` f32 uniforms on device
    /// 4. `sample_from_cdf` → per-shot binary search + atomicAdd into
    ///    a dense `counts` buffer
    /// 5. `memcpy_dtoh` of `counts` → host `HashMap<u64, u32>`
    ///
    /// The scan recurses on per-block totals (block_size = 1024) so
    /// supported `dim` is bounded only by available device memory
    /// for the `f32 probs[dim] + u32 counts[dim]` allocations. On a
    /// 32 GB device with f32 amps the statevector itself caps
    /// `n ≤ 31` long before the scan does.
    ///
    /// `BLOCK_SIZE = 1024` must stay aligned with the shared-memory
    /// Hillis-Steele scan in `cdf_scan.cu`.
    pub fn sample_counts_on_device(
        &self,
        shots: u32,
        seed: Option<u64>,
    ) -> Result<std::collections::HashMap<u64, u32>, CudaError> {
        const BLOCK_SIZE: u32 = 1024;

        if shots == 0 {
            return Ok(std::collections::HashMap::new());
        }
        let dim: u64 = 1u64 << self.num_qubits;
        if dim == 0 {
            return Ok(std::collections::HashMap::new());
        }

        let stream = &self.handle.stream;

        // Step 1: compute_probabilities → f32 array of length dim.
        let mut probs_dev = stream
            .alloc_zeros::<f32>(dim as usize)
            .map_err(|e| CudaError::Driver(format!("alloc probs: {e}")))?;
        let (grid, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let func = self.handle.kernels.compute_probabilities.clone();
            let mut builder = stream.launch_builder(&func);
            builder.arg(&self.state).arg(&mut probs_dev).arg(&dim);
            unsafe {
                builder
                    .launch(cfg)
                    .map_err(|e| CudaError::Driver(format!("launch compute_probabilities: {e}")))?;
            }
        }

        // Step 2: recursive multi-block inclusive scan → CDF in place.
        cdf_inclusive_scan_recursive(&self.handle, &mut probs_dev, dim, BLOCK_SIZE)?;
        // After the scan, the last entry of probs_dev is ~Σ|ψ|² ≈ 1.0.
        // Pin it to 1.0 host-side via a one-shot patch — keeps f32
        // rounding from dropping a shot on the floor (matches the
        // existing host-CDF behaviour).
        {
            let one = [1.0_f32];
            let mut tail = probs_dev.slice_mut((dim as usize - 1)..(dim as usize));
            stream
                .memcpy_htod(&one[..], &mut tail)
                .map_err(|e| CudaError::Driver(format!("memcpy cdf tail=1.0: {e}")))?;
        }

        // Step 3: curand uniforms on device. Seed is optional;
        // `make_rng` matches the host-fallback's seed-from-OS path.
        let resolved_seed = seed.unwrap_or_else(|| {
            use rand::{RngExt, SeedableRng};
            rand::rngs::StdRng::seed_from_u64(0);
            // Use a system-entropy seed when caller didn't pin one,
            // so independent runs are not bitwise identical.
            let mut r = rand::make_rng::<rand::rngs::StdRng>();
            r.random::<u64>()
        });
        let mut uniforms = stream
            .alloc_zeros::<f32>(shots as usize)
            .map_err(|e| CudaError::Driver(format!("alloc uniforms: {e}")))?;
        {
            let rng = cudarc::curand::CudaRng::new(resolved_seed, stream.clone())
                .map_err(|e| CudaError::Driver(format!("curand new: {e:?}")))?;
            rng.fill_with_uniform(&mut uniforms)
                .map_err(|e| CudaError::Driver(format!("curand fill_with_uniform: {e:?}")))?;
        }

        // Step 4: per-shot binary search + atomicAdd into counts.
        let mut counts_dev = stream
            .alloc_zeros::<u32>(dim as usize)
            .map_err(|e| CudaError::Driver(format!("alloc counts: {e}")))?;
        {
            let func = self.handle.kernels.sample_from_cdf.clone();
            let shots_u64 = shots as u64;
            let (sgrid, sblock) = launch_dims(shots_u64);
            let cfg_sample = LaunchConfig {
                grid_dim: (sgrid, 1, 1),
                block_dim: (sblock, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&func);
            builder
                .arg(&probs_dev)
                .arg(&uniforms)
                .arg(&mut counts_dev)
                .arg(&dim)
                .arg(&shots_u64);
            unsafe {
                builder
                    .launch(cfg_sample)
                    .map_err(|e| CudaError::Driver(format!("launch sample_from_cdf: {e}")))?;
            }
        }

        // Step 5: dtoh counts and sparsify into a HashMap.
        let counts_host: Vec<u32> = stream
            .clone_dtoh(&counts_dev)
            .map_err(|e| CudaError::Driver(format!("memcpy counts: {e}")))?;
        stream
            .synchronize()
            .map_err(|e| CudaError::Driver(format!("sync sample: {e}")))?;
        let mut out: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
        for (idx, &count) in counts_host.iter().enumerate() {
            if count > 0 {
                out.insert(idx as u64, count);
            }
        }
        Ok(out)
    }

    pub fn read_state(&self) -> Result<Vec<Complex64>, CudaError> {
        // Counter bumps before the dtoh memcpy so a test observing
        // `READ_STATE_CALL_COUNT` records the call even if the memcpy
        // errors out. Relaxed ordering — counter is informational.
        READ_STATE_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        let dim = 1usize << self.num_qubits;
        let host: Vec<f32> = self
            .handle
            .stream
            .clone_dtoh(&self.state)
            .map_err(|e| CudaError::Driver(format!("memcpy_dtov read_state: {e}")))?;
        self.handle
            .stream
            .synchronize()
            .map_err(|e| CudaError::Driver(format!("sync read_state: {e}")))?;
        let mut out = Vec::with_capacity(dim);
        for i in 0..dim {
            out.push(Complex64::new(host[2 * i] as f64, host[2 * i + 1] as f64));
        }
        Ok(out)
    }

    pub fn write_state(&mut self, state: &[Complex64]) -> Result<(), CudaError> {
        let dim = 1usize << self.num_qubits;
        if state.len() != dim {
            return Err(CudaError::StateLengthMismatch {
                expected: dim,
                got: state.len(),
            });
        }
        let mut host: Vec<f32> = Vec::with_capacity(2 * dim);
        for amp in state {
            host.push(amp.re as f32);
            host.push(amp.im as f32);
        }
        self.handle
            .stream
            .memcpy_htod(&host[..], &mut self.state)
            .map_err(|e| CudaError::Driver(format!("memcpy_htod write_state: {e}")))?;
        self.handle
            .stream
            .synchronize()
            .map_err(|e| CudaError::Driver(format!("sync write_state: {e}")))?;
        Ok(())
    }

    /// Overwrite `dst` with this state's amplitudes. Both buffers must
    /// have the same `num_qubits`.
    pub fn copy_into(&self, dst: &mut StateBuffer) -> Result<(), CudaError> {
        if self.num_qubits != dst.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << dst.num_qubits,
            });
        }
        self.handle
            .stream
            .memcpy_dtod(&self.state, &mut dst.state)
            .map_err(|e| CudaError::Driver(format!("memcpy_dtod copy_into: {e}")))?;
        Ok(())
    }

    /// Pooled-param variant of [`apply_diagonal`]. Reads the gate
    /// params from `pool[slot]` at execution time. Lets the
    /// CUDA-graph capture path stamp the same kernel call into a
    /// graph once and reuse it across replays — the per-replay
    /// param update happens by writing fresh DiagonalParams into
    /// `pool[slot]` via memcpy_htod, NOT by re-recording the graph.
    /// Caller is responsible for sizing/populating the pool ahead
    /// of dispatch.
    #[allow(dead_code)] // Wired up in the next ForwardGraph slice.
    pub fn apply_diagonal_via_pool(
        &mut self,
        pool: &CudaSlice<DiagonalParams>,
        slot: u32,
    ) -> Result<(), CudaError> {
        let dim: u64 = 1u64 << self.num_qubits;
        let (grid, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_diagonal_pooled.clone();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&mut self.state).arg(pool).arg(&slot).arg(&dim);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_diagonal_pooled: {e}")))?;
        }
        Ok(())
    }

    /// Stage G6: dual-state chained pooled-diagonal apply.
    /// Walks `pool[start..start+count]` and multiplies BOTH
    /// `self.state` (φ) and `other.state` (ν) through every
    /// diagonal in one launch. Used by the captured backward
    /// sweep to fuse the entire RZ chain's φ + ν daggers into
    /// one graph node.
    pub fn apply_diagonal_chain_dual_via_pool(
        &mut self,
        other: &mut StateBuffer,
        pool: &CudaSlice<DiagonalParams>,
        start: u32,
        count: u32,
    ) -> Result<(), CudaError> {
        if self.num_qubits != other.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << other.num_qubits,
            });
        }
        let dim: u64 = 1u64 << self.num_qubits;
        let (grid, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_diagonal_chain_dual_pooled.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&mut self.state)
            .arg(&mut other.state)
            .arg(pool)
            .arg(&start)
            .arg(&count)
            .arg(&dim);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                CudaError::Driver(format!("launch apply_diagonal_chain_dual_pooled: {e}"))
            })?;
        }
        Ok(())
    }

    /// Stage G7: per-op Pauli-Y triple fusion. One kernel:
    ///   1. computes `chain_pool[accum_slot] · Im⟨ν|Y_q|φ⟩` and
    ///      atomic-adds into `grad_dev[sym_idx_pool[accum_slot]]`,
    ///   2. applies `dagger_1q_pool[dagger_slot]` (which is U†_op)
    ///      to BOTH `self.state` (φ) and `nu.state` (ν), writing
    ///      back the post-dagger amplitudes.
    ///
    /// Replaces the (DaggerPhi1q + Deriv1qInnerProductAccumulate +
    /// DaggerNu1q) triple for an RY param'd op with a single
    /// captured graph node.
    #[allow(clippy::too_many_arguments)] // pool refs + slots inherent to this fused kernel
    pub fn pauli_y_accumulate_then_dagger_both_via_pool(
        &mut self,
        nu: &mut StateBuffer,
        dagger_pool: &CudaSlice<Apply1qParams>,
        dagger_slot: u32,
        grad_dev: &mut CudaSlice<f64>,
        chain_pool: &CudaSlice<f64>,
        sym_idx_pool: &CudaSlice<u32>,
        accum_slot: u32,
    ) -> Result<(), CudaError> {
        if self.num_qubits != nu.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << nu.num_qubits,
            });
        }
        if self.num_qubits == 0 {
            return Err(CudaError::QubitOutOfRange {
                qubit: 0,
                num_qubits: self.num_qubits,
            });
        }
        let pairs: u64 = 1u64 << (self.num_qubits - 1);
        let (grid, block) = launch_dims(pairs);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            // Shared memory: blockDim.x f32 entries × 4 bytes (per-pair partials, scalar reduction).
            shared_mem_bytes: block * 4,
        };
        let stream = self.handle.stream.clone();
        let func = self
            .handle
            .kernels
            .pauli_y_accumulate_then_dagger_both_pooled
            .clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&mut self.state)
            .arg(&mut nu.state)
            .arg(dagger_pool)
            .arg(&dagger_slot)
            .arg(grad_dev)
            .arg(chain_pool)
            .arg(sym_idx_pool)
            .arg(&accum_slot)
            .arg(&pairs);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                CudaError::Driver(format!(
                    "launch pauli_y_accumulate_then_dagger_both_pooled: {e}"
                ))
            })?;
        }
        Ok(())
    }

    /// Stage G6: chained Pauli-Z gradient accumulator. Computes
    /// `gradient[sym_k] += chain_k · Im⟨ν|Z_{q_k}|φ⟩` for every
    /// entry in `pool[start..start+count]`, in a single launch.
    /// `self` is φ and `nu` is ν. The pool is updated host-side
    /// per replay (chain factors change with `params`).
    pub fn pauli_z_chain_accumulate(
        &self,
        nu: &StateBuffer,
        grad_dev: &mut CudaSlice<f64>,
        pool: &CudaSlice<PauliZChainEntry>,
        start: u32,
        count: u32,
    ) -> Result<(), CudaError> {
        if self.num_qubits != nu.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << nu.num_qubits,
            });
        }
        if count == 0 {
            return Ok(());
        }
        let dim: u64 = 1u64 << self.num_qubits;
        let (grid_x, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, count, 1),
            block_dim: (block, 1, 1),
            // Shared memory: blockDim.x float2 entries × 8 bytes.
            shared_mem_bytes: block * 8,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.pauli_z_chain_accumulate.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&self.state)
            .arg(&nu.state)
            .arg(grad_dev)
            .arg(pool)
            .arg(&start)
            .arg(&count)
            .arg(&dim);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch pauli_z_chain_accumulate: {e}")))?;
        }
        Ok(())
    }

    /// Stage G5: chained pooled-diagonal apply. Folds K consecutive
    /// pooled-diagonal launches (`pool[start..start+count]`) into
    /// one kernel that walks each amplitude once and multiplies
    /// through K diagonal factors. K-1 fewer graph nodes per
    /// captured replay vs emitting K separate
    /// `apply_diagonal_via_pool` calls.
    pub fn apply_diagonal_chain_via_pool(
        &mut self,
        pool: &CudaSlice<DiagonalParams>,
        start: u32,
        count: u32,
    ) -> Result<(), CudaError> {
        let dim: u64 = 1u64 << self.num_qubits;
        let (grid, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_diagonal_chain_pooled.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&mut self.state)
            .arg(pool)
            .arg(&start)
            .arg(&count)
            .arg(&dim);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                CudaError::Driver(format!("launch apply_diagonal_chain_pooled: {e}"))
            })?;
        }
        Ok(())
    }

    /// Pooled-param variant of [`apply_1q`]. See
    /// [`apply_diagonal_via_pool`] for the graph-replay rationale.
    #[allow(dead_code)]
    pub fn apply_1q_via_pool(
        &mut self,
        pool: &CudaSlice<Apply1qParams>,
        slot: u32,
    ) -> Result<(), CudaError> {
        if self.num_qubits == 0 {
            return Err(CudaError::QubitOutOfRange {
                qubit: 0,
                num_qubits: self.num_qubits,
            });
        }
        let pairs: u64 = 1u64 << (self.num_qubits - 1);
        let (grid, block) = launch_dims(pairs);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_1q_pooled.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&mut self.state)
            .arg(pool)
            .arg(&slot)
            .arg(&pairs);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_1q_pooled: {e}")))?;
        }
        Ok(())
    }

    /// Stage G1: fused `copy_into` + `apply_1q_via_pool`. Reads
    /// `self.state` (the source — typically φ in the captured
    /// backward sweep) and writes `dst.state[i] = (U · self.state)[i]`
    /// with U fetched from `pool[slot]`. Replaces the
    /// `memcpy_dtod` + `apply_1q_via_pool` pair the captured graph
    /// used to emit for every CopyPhiToTemp + Deriv1q step, halving
    /// the node count on those backward steps.
    pub fn apply_1q_from_to_via_pool(
        &self,
        dst: &mut StateBuffer,
        pool: &CudaSlice<Apply1qParams>,
        slot: u32,
    ) -> Result<(), CudaError> {
        if self.num_qubits != dst.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << dst.num_qubits,
            });
        }
        if self.num_qubits == 0 {
            return Err(CudaError::QubitOutOfRange {
                qubit: 0,
                num_qubits: self.num_qubits,
            });
        }
        let pairs: u64 = 1u64 << (self.num_qubits - 1);
        let (grid, block) = launch_dims(pairs);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_1q_from_to_pooled.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&self.state)
            .arg(&mut dst.state)
            .arg(pool)
            .arg(&slot)
            .arg(&pairs);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_1q_from_to_pooled: {e}")))?;
        }
        Ok(())
    }

    /// Per-qubit ⟨Z⟩ expectation written to a device-side
    /// predictions slot. Captured by the training-step graph after
    /// the forward sweep so the host_func callback can derive the
    /// gradient observable's residual coefficients without breaking
    /// the captured pipeline. `predictions_dev[slot]` is
    /// atomically accumulated — the caller is responsible for
    /// zeroing the slot before each replay (the captured graph
    /// memsets the predictions buffer at the start of every
    /// replay).
    #[allow(dead_code)]
    pub fn pauli_z_expectation_to_slot(
        &self,
        predictions_dev: &mut CudaSlice<f64>,
        sign_mask: u32,
        slot: u32,
    ) -> Result<(), CudaError> {
        let dim: u64 = 1u64 << self.num_qubits;
        let (grid, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            // Shared memory: blockDim.x f32 entries × 4 bytes.
            shared_mem_bytes: block * 4,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.pauli_z_expectation_to_slot.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&self.state)
            .arg(predictions_dev)
            .arg(&sign_mask)
            .arg(&slot)
            .arg(&dim);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                CudaError::Driver(format!("launch pauli_z_expectation_to_slot: {e}"))
            })?;
        }
        Ok(())
    }

    /// Pooled-pool variant of [`apply_diagonal_pauli_sum`]. Takes
    /// pre-allocated `sign_masks_pool` (constant per circuit shape)
    /// and `coeffs_pool` (one f32 per term, updated per training
    /// point with `2·(y_hat - y_label)`). The existing
    /// `apply_diagonal_pauli_sum.cu` kernel already reads both
    /// arrays from device pointers; this wrapper just lets the
    /// graph-capture caller hold the pools across replays instead
    /// of `clone_htod`-ing fresh buffers per call.
    ///
    /// Caller is responsible for sizing the pools to `num_terms`
    /// and for populating `coeffs_pool` (and, on first call,
    /// `sign_masks_pool`) before invoking.
    #[allow(dead_code)]
    pub fn apply_diagonal_pauli_sum_via_pool(
        &self,
        dst: &mut StateBuffer,
        sign_masks_pool: &CudaSlice<u32>,
        coeffs_pool: &CudaSlice<f32>,
        num_terms: u32,
    ) -> Result<(), CudaError> {
        if dst.num_qubits != self.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << dst.num_qubits,
            });
        }
        let dim: u64 = 1u64 << self.num_qubits;
        if num_terms == 0 {
            // Same convention as the existing apply_diagonal_pauli_sum:
            // empty observable ⇒ zero-fill dst.
            self.handle
                .stream
                .memset_zeros(&mut dst.state)
                .map_err(|e| CudaError::Driver(format!("memset_zeros: {e}")))?;
            return Ok(());
        }
        let params = DiagonalPauliSumParams { num_terms };
        let (grid, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_diagonal_pauli_sum.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&self.state)
            .arg(&mut dst.state)
            .arg(&params)
            .arg(sign_masks_pool)
            .arg(coeffs_pool)
            .arg(&dim);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                CudaError::Driver(format!("launch apply_diagonal_pauli_sum_via_pool: {e}"))
            })?;
        }
        Ok(())
    }

    /// Fused inner_product + atomic gradient-accumulate. Reads the
    /// chain factor + symbol slot from device-side pools; on
    /// completion `grad_dev[chain_pool[slot].sym_idx]` is `+=
    /// 2·Re⟨self|temp⟩ · chain_pool[slot].chain`.
    ///
    /// Used by the BackwardGraph capture path so the per-(op, sym)
    /// host roundtrip in the adjoint inner loop disappears entirely.
    /// `chain_pool` (f64) and `sym_idx_pool` (u32) are populated by
    /// the host before each replay from `params.resolve_derivative`
    /// and the per-(op, sym) symbol id.
    #[allow(dead_code)]
    pub fn inner_product_accumulate_via_pool(
        &self,
        temp: &StateBuffer,
        grad_dev: &mut CudaSlice<f64>,
        chain_pool: &CudaSlice<f64>,
        sym_idx_pool: &CudaSlice<u32>,
        slot: u32,
    ) -> Result<(), CudaError> {
        if self.num_qubits != temp.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << temp.num_qubits,
            });
        }
        let dim: u64 = 1u64 << self.num_qubits;
        let (grid, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            // Shared memory: blockDim.x float2 entries × 8 bytes.
            shared_mem_bytes: block * 8,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.inner_product_accumulate_pooled.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&self.state)
            .arg(&temp.state)
            .arg(grad_dev)
            .arg(chain_pool)
            .arg(sym_idx_pool)
            .arg(&slot)
            .arg(&dim);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                CudaError::Driver(format!("launch inner_product_accumulate_pooled: {e}"))
            })?;
        }
        Ok(())
    }

    /// Stage G2 — triple fusion of `CopyPhiToTemp + Deriv1q +
    /// Accumulate`. Reads φ (`self`) + ν (`nu`), forms the deriv'd
    /// amplitudes from `deriv_1q_pool[deriv_slot]` per pair-thread,
    /// contracts immediately with ν, block-reduces, and atomic-adds
    /// `2 · chain_pool[accum_slot] · Re⟨ν|∂U·φ⟩` into
    /// `grad_dev[sym_idx_pool[accum_slot]]`. Skips materialising
    /// `temp` entirely.
    #[allow(clippy::too_many_arguments)] // pool refs + slots inherent to this fused kernel
    pub fn apply_1q_inner_product_accumulate_via_pool(
        &self,
        nu: &StateBuffer,
        deriv_1q_pool: &CudaSlice<Apply1qParams>,
        deriv_slot: u32,
        grad_dev: &mut CudaSlice<f64>,
        chain_pool: &CudaSlice<f64>,
        sym_idx_pool: &CudaSlice<u32>,
        accum_slot: u32,
    ) -> Result<(), CudaError> {
        if self.num_qubits != nu.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << nu.num_qubits,
            });
        }
        if self.num_qubits == 0 {
            return Err(CudaError::QubitOutOfRange {
                qubit: 0,
                num_qubits: self.num_qubits,
            });
        }
        let pairs: u64 = 1u64 << (self.num_qubits - 1);
        let (grid, block) = launch_dims(pairs);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            // Shared memory: blockDim.x float2 entries × 8 bytes.
            shared_mem_bytes: block * 8,
        };
        let stream = self.handle.stream.clone();
        let func = self
            .handle
            .kernels
            .apply_1q_inner_product_accumulate_pooled
            .clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&self.state)
            .arg(&nu.state)
            .arg(deriv_1q_pool)
            .arg(&deriv_slot)
            .arg(grad_dev)
            .arg(chain_pool)
            .arg(sym_idx_pool)
            .arg(&accum_slot)
            .arg(&pairs);
        unsafe {
            builder.launch(cfg).map_err(|e| {
                CudaError::Driver(format!(
                    "launch apply_1q_inner_product_accumulate_pooled: {e}"
                ))
            })?;
        }
        Ok(())
    }

    /// Pooled-param variant of [`apply_2q`]. See
    /// [`apply_diagonal_via_pool`] for the graph-replay rationale.
    #[allow(dead_code)]
    pub fn apply_2q_via_pool(
        &mut self,
        pool: &CudaSlice<Apply2qParams>,
        slot: u32,
    ) -> Result<(), CudaError> {
        if self.num_qubits < 2 {
            return Err(CudaError::QubitOutOfRange {
                qubit: 0,
                num_qubits: self.num_qubits,
            });
        }
        let quads: u64 = 1u64 << (self.num_qubits - 2);
        let (grid, block) = launch_dims(quads);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_2q_pooled.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&mut self.state)
            .arg(pool)
            .arg(&slot)
            .arg(&quads);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_2q_pooled: {e}")))?;
        }
        Ok(())
    }

    /// Stage G4: dual-state pooled 1q gate. 1q analogue of
    /// [`apply_2q_pooled_dual`] — applies U from `pool[slot]` to
    /// both `self` and `other` in one launch. Fuses
    /// `DaggerPhi1q` + `DaggerNu1q` for non-parameterised 1q ops
    /// (H, X, Y, Z, S, T on the Phase 4c HEA shape).
    pub fn apply_1q_pooled_dual(
        &mut self,
        other: &mut StateBuffer,
        pool: &CudaSlice<Apply1qParams>,
        slot: u32,
    ) -> Result<(), CudaError> {
        if self.num_qubits != other.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << other.num_qubits,
            });
        }
        if self.num_qubits == 0 {
            return Err(CudaError::QubitOutOfRange {
                qubit: 0,
                num_qubits: self.num_qubits,
            });
        }
        let pairs: u64 = 1u64 << (self.num_qubits - 1);
        let (grid, block) = launch_dims(pairs);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_1q_pooled_dual.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&mut self.state)
            .arg(&mut other.state)
            .arg(pool)
            .arg(&slot)
            .arg(&pairs);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_1q_pooled_dual: {e}")))?;
        }
        Ok(())
    }

    /// Stage G3: dual-state pooled 2q gate. Applies the same U
    /// from `pool[slot]` to both `self` and `other` in one kernel
    /// call. Used by the captured backward sweep to fuse
    /// `DaggerPhi2q` + `DaggerNu2q` for non-parameterised 2q ops
    /// (CNOT/CZ on Phase 4c HEA shape) — replaces two graph nodes
    /// with one.
    pub fn apply_2q_pooled_dual(
        &mut self,
        other: &mut StateBuffer,
        pool: &CudaSlice<Apply2qParams>,
        slot: u32,
    ) -> Result<(), CudaError> {
        if self.num_qubits != other.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << other.num_qubits,
            });
        }
        if self.num_qubits < 2 {
            return Err(CudaError::QubitOutOfRange {
                qubit: 0,
                num_qubits: self.num_qubits,
            });
        }
        let quads: u64 = 1u64 << (self.num_qubits - 2);
        let (grid, block) = launch_dims(quads);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_2q_pooled_dual.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&mut self.state)
            .arg(&mut other.state)
            .arg(pool)
            .arg(&slot)
            .arg(&quads);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_2q_pooled_dual: {e}")))?;
        }
        Ok(())
    }

    pub fn apply_diagonal(
        &mut self,
        qubit: u32,
        d0: Complex64,
        d1: Complex64,
    ) -> Result<(), CudaError> {
        if qubit >= self.num_qubits {
            return Err(CudaError::QubitOutOfRange {
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
        let (grid, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_diagonal.clone();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&mut self.state).arg(&params).arg(&dim);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_diagonal: {e}")))?;
        }
        Ok(())
    }

    pub fn apply_diagonal_pauli_sum(
        &self,
        dst: &mut StateBuffer,
        terms: &[(u32, f32)],
    ) -> Result<(), CudaError> {
        if dst.num_qubits != self.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << dst.num_qubits,
            });
        }

        let dim: u64 = 1u64 << self.num_qubits;
        if terms.is_empty() {
            self.handle
                .stream
                .memset_zeros(&mut dst.state)
                .map_err(|e| CudaError::Driver(format!("memset_zeros: {e}")))?;
            return Ok(());
        }
        let params = DiagonalPauliSumParams {
            num_terms: terms.len() as u32,
        };
        let sign_masks: Vec<u32> = terms.iter().map(|(m, _)| *m).collect();
        let coeffs: Vec<f32> = terms.iter().map(|(_, c)| *c).collect();

        let sign_dev = self
            .handle
            .stream
            .clone_htod(&sign_masks)
            .map_err(|e| CudaError::Driver(format!("upload sign_masks: {e}")))?;
        let coeffs_dev = self
            .handle
            .stream
            .clone_htod(&coeffs)
            .map_err(|e| CudaError::Driver(format!("upload coeffs: {e}")))?;

        let (grid, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_diagonal_pauli_sum.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&self.state)
            .arg(&mut dst.state)
            .arg(&params)
            .arg(&sign_dev)
            .arg(&coeffs_dev)
            .arg(&dim);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_diagonal_pauli_sum: {e}")))?;
        }
        Ok(())
    }

    pub fn apply_diagonal_product(
        &mut self,
        factors: &[(u32, Complex64, Complex64)],
    ) -> Result<(), CudaError> {
        if factors.is_empty() {
            return Ok(());
        }
        for &(q, _, _) in factors {
            if q >= self.num_qubits {
                return Err(CudaError::QubitOutOfRange {
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
        let d0_factors: Vec<f32> = factors
            .iter()
            .flat_map(|(_, d0, _)| [d0.re as f32, d0.im as f32])
            .collect();
        let d1_factors: Vec<f32> = factors
            .iter()
            .flat_map(|(_, _, d1)| [d1.re as f32, d1.im as f32])
            .collect();

        let qubits_dev = self
            .handle
            .stream
            .clone_htod(&qubits)
            .map_err(|e| CudaError::Driver(format!("upload qubits: {e}")))?;
        let d0_dev = self
            .handle
            .stream
            .clone_htod(&d0_factors)
            .map_err(|e| CudaError::Driver(format!("upload d0: {e}")))?;
        let d1_dev = self
            .handle
            .stream
            .clone_htod(&d1_factors)
            .map_err(|e| CudaError::Driver(format!("upload d1: {e}")))?;

        let (grid, block) = launch_dims(dim);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_diagonal_product.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&mut self.state)
            .arg(&params)
            .arg(&qubits_dev)
            .arg(&d0_dev)
            .arg(&d1_dev)
            .arg(&dim);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_diagonal_product: {e}")))?;
        }
        Ok(())
    }

    pub fn apply_1q(&mut self, qubit: u32, u: &[Complex64; 4]) -> Result<(), CudaError> {
        if qubit >= self.num_qubits {
            return Err(CudaError::QubitOutOfRange {
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
        let (grid, block) = launch_dims(pairs);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_1q.clone();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&mut self.state).arg(&params).arg(&pairs);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_1q: {e}")))?;
        }
        Ok(())
    }

    pub fn apply_2q(&mut self, qa: u32, qb: u32, u: &[Complex64; 16]) -> Result<(), CudaError> {
        if qa >= self.num_qubits {
            return Err(CudaError::QubitOutOfRange {
                qubit: qa,
                num_qubits: self.num_qubits,
            });
        }
        if qb >= self.num_qubits {
            return Err(CudaError::QubitOutOfRange {
                qubit: qb,
                num_qubits: self.num_qubits,
            });
        }
        if qa == qb {
            return Err(CudaError::DuplicateQubits { qubit: qa });
        }
        if self.num_qubits < 2 {
            return Err(CudaError::QubitOutOfRange {
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
        let (grid, block) = launch_dims(quads);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.apply_2q.clone();
        let mut builder = stream.launch_builder(&func);
        builder.arg(&mut self.state).arg(&params).arg(&quads);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch apply_2q: {e}")))?;
        }
        Ok(())
    }

    /// Compute ⟨self|other⟩ via a two-stage block reduction. The
    /// per-block partials cross to host and sum on CPU (the count
    /// is small — ≤ a few thousand for n ≤ 28).
    pub fn inner_product(&self, other: &StateBuffer) -> Result<Complex64, CudaError> {
        let pending = self.inner_product_deferred(other)?;
        // Sync now and reduce on host — same shape as the
        // deferred variant's eventual sync.
        self.handle
            .stream
            .synchronize()
            .map_err(|e| CudaError::Driver(format!("sync inner_product: {e}")))?;
        Ok(pending.reduce())
    }

    /// Same kernel + memcpy as [`inner_product`] but **does not
    /// synchronize the stream**. Returns a [`PendingInnerProduct`]
    /// whose host buffer is being filled asynchronously; the caller
    /// must call [`crate::imp::DeviceHandle::synchronize`] (or any
    /// other host-blocking op on the same stream) before invoking
    /// `pending.reduce()`. Used by the adjoint loop to batch all
    /// per-(op, sym) inner_products into a single end-of-loop sync —
    /// at n=14 this collapses ~960 syncs per training step down to 1.
    pub fn inner_product_deferred(
        &self,
        other: &StateBuffer,
    ) -> Result<PendingInnerProduct, CudaError> {
        if self.num_qubits != other.num_qubits {
            return Err(CudaError::StateLengthMismatch {
                expected: 1usize << self.num_qubits,
                got: 1usize << other.num_qubits,
            });
        }
        let dim: u64 = 1u64 << self.num_qubits;
        let (grid, block) = launch_dims(dim);
        let num_partials = grid as usize;
        let mut partials = self
            .handle
            .stream
            .alloc_zeros::<f32>(num_partials * 2)
            .map_err(|e| CudaError::Driver(format!("alloc partials: {e}")))?;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            // Shared memory: blockDim.x float2 = 8 bytes each.
            shared_mem_bytes: block * 8,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.inner_product.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&self.state)
            .arg(&other.state)
            .arg(&mut partials)
            .arg(&dim);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch inner_product: {e}")))?;
        }
        // Async memcpy_dtoh → host Vec. The Vec is allocated and its
        // bytes are being filled by the GPU asynchronously; reading
        // before a stream sync is UB. The PendingInnerProduct wrapper
        // makes that contract explicit at the type level: the only
        // way to get an f64 out is via reduce(), which the caller is
        // expected to invoke after a sync.
        let host: Vec<f32> = self
            .handle
            .stream
            .clone_dtoh(&partials)
            .map_err(|e| CudaError::Driver(format!("memcpy partials: {e}")))?;
        Ok(PendingInnerProduct {
            host,
            _partials_dev: partials,
        })
    }

    pub fn pauli_expectation(
        &self,
        x_mask: u32,
        sign_mask: u32,
        y_factor: Complex64,
    ) -> Result<Complex64, CudaError> {
        let dim: u64 = 1u64 << self.num_qubits;
        let (grid, block) = launch_dims(dim);
        let num_partials = grid as usize;
        let mut partials = self
            .handle
            .stream
            .alloc_zeros::<f32>(num_partials * 2)
            .map_err(|e| CudaError::Driver(format!("alloc partials: {e}")))?;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 8,
        };
        let yf = Float2 {
            re: y_factor.re as f32,
            im: y_factor.im as f32,
        };
        let stream = self.handle.stream.clone();
        let func = self.handle.kernels.pauli_expectation.clone();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&self.state)
            .arg(&mut partials)
            .arg(&x_mask)
            .arg(&sign_mask)
            .arg(&yf)
            .arg(&dim);
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| CudaError::Driver(format!("launch pauli_expectation: {e}")))?;
        }
        let host: Vec<f32> = self
            .handle
            .stream
            .clone_dtoh(&partials)
            .map_err(|e| CudaError::Driver(format!("memcpy partials: {e}")))?;
        self.handle
            .stream
            .synchronize()
            .map_err(|e| CudaError::Driver(format!("sync pauli_expectation: {e}")))?;
        let mut acc_re: f64 = 0.0;
        let mut acc_im: f64 = 0.0;
        for i in 0..num_partials {
            acc_re += host[2 * i] as f64;
            acc_im += host[2 * i + 1] as f64;
        }
        Ok(Complex64::new(acc_re, acc_im))
    }
}
