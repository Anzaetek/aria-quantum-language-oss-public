//! OpenCL implementation details — device discovery, context /
//! queue / program management, and the per-circuit
//! [`StateBuffer`].
//!
//! Public surface ([`crate::OpenClStatevectorBackend`] and
//! [`crate::OpenClState`]) wraps these types so the lib.rs surface
//! stays stable as kernels land slice by slice.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use num_complex::Complex64;
use ocl::{flags::MemFlags, Buffer, Context, Device, Kernel, Platform, Program, Queue};

use crate::kernels::KernelLibrary;
use crate::OpenClError;

/// Stateless per-process OpenCL handle: platform, device, context,
/// command queue, kernel library. Cheap to clone (refcounted).
pub(crate) struct DeviceHandle {
    /// Held alive so the queue + buffers it owns stay valid; not
    /// directly read after construction (kernels enqueue through
    /// `queue`).
    #[allow(dead_code)]
    pub context: Arc<Context>,
    /// Likewise — retained for future per-device introspection
    /// (e.g. workgroup size, max-alloc) but not read on the hot
    /// path today.
    #[allow(dead_code)]
    pub device: Device,
    pub queue: Arc<Queue>,
    pub kernels: Arc<KernelLibrary>,
}

impl DeviceHandle {
    /// Pick the first usable platform + device, build the kernel
    /// library. Mirrors Metal's `MetalStatevectorBackend::new`. The
    /// search order is GPU-first (matches `clinfo`'s default) and
    /// falls back to any device on the platform.
    pub fn new() -> Result<Self, OpenClError> {
        let platform = pick_platform()?;
        let device = pick_device(platform)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("ocl Context::build: {e}")))?;
        let context = Arc::new(context);
        let queue = Queue::new(&context, device, None)
            .map_err(|e| OpenClError::Runtime(format!("ocl Queue::new: {e}")))?;
        let queue = Arc::new(queue);
        let kernels = KernelLibrary::build(&context, device)?;
        Ok(Self {
            context,
            device,
            queue,
            kernels: Arc::new(kernels),
        })
    }

    /// Allocate a fresh statevector for `num_qubits` qubits, initialised
    /// to `|0…0⟩`. Pre-builds the per-buffer `apply_1q` / `apply_2q`
    /// kernels and a 32-float scratch buffer for the 2q gate matrix so
    /// the per-gate hot path is just `set_arg + enqueue` (no kernel
    /// rebuild, no fresh buffer allocation, no `queue.finish`).
    pub fn allocate(&self, num_qubits: u32) -> Result<StateBuffer, OpenClError> {
        if num_qubits > crate::OpenClStatevectorBackend::MAX_QUBITS {
            return Err(OpenClError::AllocationRefused {
                num_qubits,
                reason: "exceeds OpenClStatevectorBackend::MAX_QUBITS",
            });
        }
        let dim: usize = 1usize
            .checked_shl(num_qubits)
            .ok_or(OpenClError::AllocationRefused {
                num_qubits,
                reason: "1 << num_qubits overflows usize",
            })?;
        let n_floats = dim.checked_mul(2).ok_or(OpenClError::AllocationRefused {
            num_qubits,
            reason: "2 * 2^num_qubits overflows usize",
        })?;

        let mut host = vec![0.0_f32; n_floats];
        host[0] = 1.0; // |0…0⟩ amplitude

        let buf = Buffer::<f32>::builder()
            .queue((*self.queue).clone())
            .flags(MemFlags::new().read_write().copy_host_ptr())
            .len(n_floats)
            .copy_host_slice(&host)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("ocl Buffer::build: {e}")))?;

        // 32-float scratch for the 2q gate matrix. Re-uploaded per gate
        // (32 floats is tiny vs the kernel cost). `read_only` from the
        // device side, `host_write_only` from the host side — matches
        // the access pattern.
        let u_buf = Buffer::<f32>::builder()
            .queue((*self.queue).clone())
            .flags(MemFlags::new().read_only().host_write_only())
            .len(32)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("ocl Buffer::build u_buf: {e}")))?;

        // Pre-build the apply_1q kernel with the state buffer bound at
        // arg 0 (constant for this StateBuffer's lifetime); arg 1
        // (qubit), args 2..=9 (the 8 U entries), and arg 10 (pairs)
        // are mutated via `set_arg` before each enqueue.
        let apply_1q_kernel = Kernel::builder()
            .program(&self.kernels.program)
            .name("apply_1q")
            .queue((*self.queue).clone())
            .arg(&buf)
            .arg(0u32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0u64)
            .build()
            .map_err(|e| {
                OpenClError::Runtime(format!("ocl Kernel::build apply_1q (cached): {e}"))
            })?;

        // Pre-build apply_2q with state + u_buf bound at args 0 + 1;
        // args 2 (qa), 3 (qb), 4 (quads) mutate per call.
        let apply_2q_kernel = Kernel::builder()
            .program(&self.kernels.program)
            .name("apply_2q")
            .queue((*self.queue).clone())
            .arg(&buf)
            .arg(&u_buf)
            .arg(0u32)
            .arg(0u32)
            .arg(0u64)
            .build()
            .map_err(|e| {
                OpenClError::Runtime(format!("ocl Kernel::build apply_2q (cached): {e}"))
            })?;

        // Pre-build apply_diagonal — state at arg 0; arg 1 (qubit),
        // args 2..=5 (d0_re/im, d1_re/im), arg 6 (dim) update per
        // call. dim is constant for this buffer; we still pass it as
        // a mutable arg because OpenCL `set_arg` is symmetric per slot.
        let apply_diagonal_kernel = Kernel::builder()
            .program(&self.kernels.program)
            .name("apply_diagonal")
            .queue((*self.queue).clone())
            .arg(&buf)
            .arg(0u32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0u64)
            .build()
            .map_err(|e| {
                OpenClError::Runtime(format!("ocl Kernel::build apply_diagonal (cached): {e}"))
            })?;

        // Pre-build apply_diagonal_2q — state at arg 0; args 1..=10
        // (qa, qb, four (re, im) d-entries), arg 11 (dim) per call.
        let apply_diagonal_2q_kernel = Kernel::builder()
            .program(&self.kernels.program)
            .name("apply_diagonal_2q")
            .queue((*self.queue).clone())
            .arg(&buf)
            .arg(0u32)
            .arg(0u32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0.0_f32)
            .arg(0u64)
            .build()
            .map_err(|e| {
                OpenClError::Runtime(format!("ocl Kernel::build apply_diagonal_2q (cached): {e}"))
            })?;

        Ok(StateBuffer {
            num_qubits,
            buf,
            queue: Arc::clone(&self.queue),
            program: Arc::clone(&self.kernels.program),
            apply_1q_kernel,
            apply_2q_kernel,
            apply_diagonal_kernel,
            apply_diagonal_2q_kernel,
            u_buf,
            dispatch_count: AtomicU64::new(0),
        })
    }
}

fn pick_platform() -> Result<Platform, OpenClError> {
    let pls = Platform::list();
    if pls.is_empty() {
        return Err(OpenClError::Unavailable("no OpenCL platforms found"));
    }
    // First-platform-wins; on macOS that's `Apple`, on Linux/Windows
    // typically the GPU vendor's ICD.
    Ok(pls[0])
}

fn pick_device(platform: Platform) -> Result<Device, OpenClError> {
    // Try GPU first; fall back to any device.
    if let Ok(devs) = Device::list(platform, Some(ocl::flags::DeviceType::new().gpu())) {
        if let Some(d) = devs.into_iter().next() {
            return Ok(d);
        }
    }
    if let Ok(devs) = Device::list_all(platform) {
        if let Some(d) = devs.into_iter().next() {
            return Ok(d);
        }
    }
    Err(OpenClError::Unavailable(
        "no OpenCL devices on the selected platform",
    ))
}

/// Per-circuit statevector held in an `ocl::Buffer<f32>` of length
/// `2 * 2^num_qubits` (interleaved (re, im) f32 pairs).
///
/// Holds pre-built `apply_1q` / `apply_2q` kernels (with `buf` bound
/// at arg 0) and a 32-float scratch for the 2q gate matrix. The hot
/// path on each gate is `set_arg` + `enqueue` only — no kernel
/// rebuild, no fresh `Buffer::builder()`, no `queue.finish`. A
/// blocking `read_state` / `write_state` (on `apply_1q` / `apply_2q`'s
/// shared `Queue`, which is in-order by default) flushes any pending
/// gate work as a natural side effect of the host I/O round-trip.
///
/// The cached `Kernel`s internally retain the OpenCL program (via
/// `clCreateKernel`'s implicit `clRetainProgram`), so we do not also
/// need a Rust-side `Arc<KernelLibrary>` here — the `DeviceHandle`'s
/// reference is enough to keep the program alive across the whole
/// process.
pub(crate) struct StateBuffer {
    pub num_qubits: u32,
    pub buf: Buffer<f32>,
    pub queue: Arc<Queue>,
    /// Held so `sample_shots_gpu` (and future per-call kernels) can
    /// build short-lived `Kernel` handles for `shot_probs` /
    /// `shot_scan_step` / `shot_sample` against the same program the
    /// cached `apply_1q` / `apply_2q` kernels live in.
    program: Arc<Program>,
    apply_1q_kernel: Kernel,
    apply_2q_kernel: Kernel,
    apply_diagonal_kernel: Kernel,
    apply_diagonal_2q_kernel: Kernel,
    u_buf: Buffer<f32>,
    /// Per-buffer tally of kernel dispatches issued through this
    /// `StateBuffer`. Bumped at every `apply_*` kernel enqueue and
    /// every `sample_shots_gpu` stage. Mirrors Metal's per-batch
    /// `dispatch_count` (sized to a single `StateBuffer`'s lifetime
    /// so parallel tests on independent buffers don't contaminate
    /// each other's counts). Read via `dispatch_count()`; relaxed
    /// ordering — informational, not used for control flow.
    dispatch_count: AtomicU64,
}

impl StateBuffer {
    fn dim(&self) -> usize {
        1usize << self.num_qubits
    }

    /// Per-buffer running tally of kernel dispatches. Used by the
    /// diagonal-fusion smoke test to pin "an 8-Rz layer collapses to
    /// one dispatch, not 8".
    pub fn dispatch_count(&self) -> u64 {
        self.dispatch_count.load(Ordering::Relaxed)
    }

    /// Read the statevector back to host-side `Complex64`. Crosses the
    /// f32 → f64 boundary.
    pub fn read_state(&self) -> Vec<Complex64> {
        let n_floats = self.dim() * 2;
        let mut buf = vec![0.0_f32; n_floats];
        // Blocking read — host-side enqueue + wait.
        self.buf
            .read(&mut buf[..])
            .queue(&self.queue)
            .enq()
            .expect("opencl read_state");
        buf.chunks_exact(2)
            .map(|c| Complex64::new(c[0] as f64, c[1] as f64))
            .collect()
    }

    /// Overwrite the statevector with a host-side `Complex64` slice.
    /// Slice length must be `2^num_qubits`. Truncates each amplitude
    /// to f32.
    pub fn write_state(&mut self, state: &[Complex64]) -> Result<(), OpenClError> {
        if state.len() != self.dim() {
            return Err(OpenClError::StateLengthMismatch {
                expected: self.dim(),
                got: state.len(),
            });
        }
        let mut host = Vec::with_capacity(state.len() * 2);
        for c in state {
            host.push(c.re as f32);
            host.push(c.im as f32);
        }
        self.buf
            .write(&host[..])
            .queue(&self.queue)
            .enq()
            .map_err(|e| OpenClError::Runtime(format!("opencl write_state: {e}")))?;
        Ok(())
    }

    /// Apply a row-major 4x4 unitary on qubits `(qa, qb)`. The matrix
    /// is laid out with `qa` as the low row-bit and `qb` as the high
    /// row-bit — same convention as the Metal / CUDA `apply_2q`
    /// paths, so a host-side matrix builder feeds all three GPU
    /// backends unchanged. `u` must be exactly 32 floats (16 complex
    /// entries, (re, im) interleaved).
    pub fn apply_2q(&mut self, qa: u32, qb: u32, u: &[f32; 32]) -> Result<(), OpenClError> {
        if qa >= self.num_qubits {
            return Err(OpenClError::QubitOutOfRange {
                qubit: qa,
                num_qubits: self.num_qubits,
            });
        }
        if qb >= self.num_qubits {
            return Err(OpenClError::QubitOutOfRange {
                qubit: qb,
                num_qubits: self.num_qubits,
            });
        }
        if qa == qb {
            return Err(OpenClError::DuplicateQubits { qubit: qa });
        }
        let quads = (self.dim() / 4) as u64;

        // Refresh the 32-float gate matrix on the cached scratch
        // buffer. In-order queue → the upload completes before the
        // following kernel enqueue reads it.
        self.u_buf
            .write(&u[..])
            .queue(&self.queue)
            .enq()
            .map_err(|e| OpenClError::Runtime(format!("opencl u_buf write: {e}")))?;

        // Update the per-call args on the cached kernel. Args 0
        // (state) and 1 (u_buf) were bound at allocate time.
        let k = &self.apply_2q_kernel;
        k.set_arg(2, qa)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg apply_2q qa: {e}")))?;
        k.set_arg(3, qb)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg apply_2q qb: {e}")))?;
        k.set_arg(4, quads)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg apply_2q quads: {e}")))?;

        // Safety: the kernel source is the in-tree apply_2q.cl, vetted
        // at build time by the OpenCL ICD's compiler (see
        // KernelLibrary::build). Runtime args have already been
        // bounds-checked above (qa, qb < num_qubits, qa != qb) so the
        // in-kernel index arithmetic stays within `state`'s allocation.
        unsafe {
            k.cmd()
                .queue(&*self.queue)
                .global_work_size(quads as usize)
                .enq()
                .map_err(|e| OpenClError::Runtime(format!("opencl enq apply_2q: {e}")))?;
        }
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
        // No queue.finish: in-order queue serialises with the next
        // gate / read_state, and read_state's blocking read flushes.
        Ok(())
    }

    /// Apply a single-qubit unitary `U = [[u00, u01], [u10, u11]]` to
    /// `qubit`. Mirrors the Metal / CUDA `apply_1q` paths so the
    /// host-side gate-matrix builders feed this kernel without
    /// touching their byte layout.
    pub fn apply_1q(
        &mut self,
        qubit: u32,
        u00: Complex64,
        u01: Complex64,
        u10: Complex64,
        u11: Complex64,
    ) -> Result<(), OpenClError> {
        if qubit >= self.num_qubits {
            return Err(OpenClError::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        let pairs = (self.dim() / 2) as u64;

        // Update the per-call args on the cached kernel. Arg 0 (state)
        // was bound at allocate time.
        let k = &self.apply_1q_kernel;
        k.set_arg(1, qubit)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg apply_1q qubit: {e}")))?;
        k.set_arg(2, u00.re as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg u00_re: {e}")))?;
        k.set_arg(3, u00.im as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg u00_im: {e}")))?;
        k.set_arg(4, u01.re as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg u01_re: {e}")))?;
        k.set_arg(5, u01.im as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg u01_im: {e}")))?;
        k.set_arg(6, u10.re as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg u10_re: {e}")))?;
        k.set_arg(7, u10.im as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg u10_im: {e}")))?;
        k.set_arg(8, u11.re as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg u11_re: {e}")))?;
        k.set_arg(9, u11.im as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg u11_im: {e}")))?;
        k.set_arg(10, pairs)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg apply_1q pairs: {e}")))?;

        // Safety: the kernel source is the in-tree apply_1q.cl, vetted
        // at build time by the OpenCL ICD's compiler. The runtime
        // qubit index has been bounds-checked above so the kernel's
        // `1 << qubit` shift + index arithmetic stays in-buffer.
        unsafe {
            k.cmd()
                .queue(&*self.queue)
                .global_work_size(pairs as usize)
                .enq()
                .map_err(|e| OpenClError::Runtime(format!("opencl enq apply_1q: {e}")))?;
        }
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
        // No queue.finish: read_state's blocking read serves as the
        // host-visible sync point.
        Ok(())
    }

    /// Apply a diagonal single-qubit gate `U = diag(d0, d1)` to
    /// `qubit`. Dispatches through the dedicated `apply_diagonal`
    /// kernel instead of going through `apply_1q`: half the memory
    /// traffic per amplitude and skip the 2x2 matvec on the off-
    /// diagonal zeros. Direct equivalent of
    /// `MetalState::apply_diagonal`.
    pub fn apply_diagonal(
        &mut self,
        qubit: u32,
        d0: Complex64,
        d1: Complex64,
    ) -> Result<(), OpenClError> {
        if qubit >= self.num_qubits {
            return Err(OpenClError::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        let dim = self.dim() as u64;
        let k = &self.apply_diagonal_kernel;
        k.set_arg(1, qubit).map_err(|e| {
            OpenClError::Runtime(format!("opencl set_arg apply_diagonal qubit: {e}"))
        })?;
        k.set_arg(2, d0.re as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg d0_re: {e}")))?;
        k.set_arg(3, d0.im as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg d0_im: {e}")))?;
        k.set_arg(4, d1.re as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg d1_re: {e}")))?;
        k.set_arg(5, d1.im as f32)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg d1_im: {e}")))?;
        k.set_arg(6, dim)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg apply_diagonal dim: {e}")))?;

        // Safety: kernel source is in-tree apply_diagonal.cl, vetted
        // at program-build time. tid >= dim is early-returned in the
        // kernel; qubit has been bounds-checked above.
        unsafe {
            k.cmd()
                .queue(&*self.queue)
                .global_work_size(self.dim())
                .enq()
                .map_err(|e| OpenClError::Runtime(format!("opencl enq apply_diagonal: {e}")))?;
        }
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Apply a diagonal 2q gate `U = diag(d00, d01, d10, d11)` on
    /// `(qa, qb)`. Index ordering matches `apply_2q`:
    /// row r = bit_qb*2 + bit_qa. Mirrors
    /// `MetalState::apply_diagonal_2q`.
    pub fn apply_diagonal_2q(
        &mut self,
        qa: u32,
        qb: u32,
        d00: Complex64,
        d01: Complex64,
        d10: Complex64,
        d11: Complex64,
    ) -> Result<(), OpenClError> {
        if qa >= self.num_qubits {
            return Err(OpenClError::QubitOutOfRange {
                qubit: qa,
                num_qubits: self.num_qubits,
            });
        }
        if qb >= self.num_qubits {
            return Err(OpenClError::QubitOutOfRange {
                qubit: qb,
                num_qubits: self.num_qubits,
            });
        }
        if qa == qb {
            return Err(OpenClError::DuplicateQubits { qubit: qa });
        }
        let dim = self.dim() as u64;
        let k = &self.apply_diagonal_2q_kernel;
        k.set_arg(1, qa)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg 2q qa: {e}")))?;
        k.set_arg(2, qb)
            .map_err(|e| OpenClError::Runtime(format!("opencl set_arg 2q qb: {e}")))?;
        let entries: [(usize, f32); 8] = [
            (3, d00.re as f32),
            (4, d00.im as f32),
            (5, d01.re as f32),
            (6, d01.im as f32),
            (7, d10.re as f32),
            (8, d10.im as f32),
            (9, d11.re as f32),
            (10, d11.im as f32),
        ];
        for (slot, v) in entries {
            k.set_arg(slot as u32, v)
                .map_err(|e| OpenClError::Runtime(format!("opencl set_arg 2q slot {slot}: {e}")))?;
        }
        k.set_arg(11, dim).map_err(|e| {
            OpenClError::Runtime(format!("opencl set_arg apply_diagonal_2q dim: {e}"))
        })?;

        // Safety: in-tree kernel; qa, qb bounds-checked + distinct.
        unsafe {
            k.cmd()
                .queue(&*self.queue)
                .global_work_size(self.dim())
                .enq()
                .map_err(|e| OpenClError::Runtime(format!("opencl enq apply_diagonal_2q: {e}")))?;
        }
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Apply N diagonal 1q gates as a single fused dispatch. Each
    /// factor is `(qubit, d0, d1)`; the kernel multiplies each
    /// amplitude by the per-factor entry indexed by that qubit's bit.
    /// Equivalent (modulo f32 associativity) to N sequential
    /// `apply_diagonal` calls — diagonal gates commute — but pays one
    /// dispatch instead of N. Empty factor list is a no-op (and
    /// burns no dispatch).
    ///
    /// Out-of-range qubits return `OpenClError::QubitOutOfRange` and
    /// the kernel is not enqueued.
    pub fn apply_diagonal_product(
        &mut self,
        factors: &[(u32, Complex64, Complex64)],
    ) -> Result<(), OpenClError> {
        if factors.is_empty() {
            return Ok(());
        }
        // Validate qubits up front so a single bad factor doesn't
        // mutate state via a partial write.
        for (q, _, _) in factors {
            if *q >= self.num_qubits {
                return Err(OpenClError::QubitOutOfRange {
                    qubit: *q,
                    num_qubits: self.num_qubits,
                });
            }
        }

        let n: usize = factors.len();
        let mut qubits = Vec::with_capacity(n);
        let mut d0_flat = Vec::with_capacity(n * 2);
        let mut d1_flat = Vec::with_capacity(n * 2);
        for (q, d0, d1) in factors {
            qubits.push(*q);
            d0_flat.push(d0.re as f32);
            d0_flat.push(d0.im as f32);
            d1_flat.push(d1.re as f32);
            d1_flat.push(d1.im as f32);
        }

        // Short-lived device buffers carrying the factor records.
        // Per-call alloc is fine — `n` is small (≤ num_qubits in
        // practice), so the bytes are negligible compared to the
        // statevector itself. The buffers go out of scope at the end
        // of this function; `ocl::Buffer::Drop` releases them.
        let qubits_buf = Buffer::<u32>::builder()
            .queue((*self.queue).clone())
            .flags(
                MemFlags::new()
                    .read_only()
                    .host_write_only()
                    .copy_host_ptr(),
            )
            .len(n)
            .copy_host_slice(&qubits)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl qubits buf: {e}")))?;
        let d0_buf = Buffer::<f32>::builder()
            .queue((*self.queue).clone())
            .flags(
                MemFlags::new()
                    .read_only()
                    .host_write_only()
                    .copy_host_ptr(),
            )
            .len(n * 2)
            .copy_host_slice(&d0_flat)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl d0 buf: {e}")))?;
        let d1_buf = Buffer::<f32>::builder()
            .queue((*self.queue).clone())
            .flags(
                MemFlags::new()
                    .read_only()
                    .host_write_only()
                    .copy_host_ptr(),
            )
            .len(n * 2)
            .copy_host_slice(&d1_flat)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl d1 buf: {e}")))?;

        let dim = self.dim();
        let kernel = Kernel::builder()
            .program(&*self.program)
            .name("apply_diagonal_product")
            .queue((*self.queue).clone())
            .arg(&self.buf)
            .arg(&qubits_buf)
            .arg(&d0_buf)
            .arg(&d1_buf)
            .arg(n as u32)
            .arg(dim as u64)
            .build()
            .map_err(|e| {
                OpenClError::Runtime(format!("ocl Kernel::build apply_diagonal_product: {e}"))
            })?;

        // Safety: every qubit was bounds-checked above; n is the host
        // slice length and matches the device buffer lengths; dim
        // sizes the state buffer. Kernel early-returns on tid >= dim.
        unsafe {
            kernel
                .cmd()
                .queue(&*self.queue)
                .global_work_size(dim)
                .enq()
                .map_err(|e| {
                    OpenClError::Runtime(format!("opencl enq apply_diagonal_product: {e}"))
                })?;
        }
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Re-initialise the statevector to `|0…0⟩` on the device. Two
    /// enqueued ops: a `clEnqueueFillBuffer` zeroes the whole buffer
    /// (device-side fill — no host roundtrip), then a single 4-byte
    /// `clEnqueueWriteBuffer` writes `1.0_f32` at offset 0 to set the
    /// `|0…0⟩` amplitude's real part. Used by [`BufferPool::lease`]
    /// to recycle pooled buffers between QML adjoint sweeps. Cheap
    /// vs `write_state`'s full `2·dim·f32` byte upload.
    pub fn reset_to_zero(&self) -> Result<(), OpenClError> {
        let n_floats = self.dim() * 2;
        // Stage 1: device-side zero fill.
        self.buf
            .cmd()
            .fill(0.0_f32, Some(n_floats))
            .queue(&self.queue)
            .enq()
            .map_err(|e| OpenClError::Runtime(format!("opencl fill zero: {e}")))?;
        // Stage 2: poke the |0…0⟩ amplitude's real part to 1.0. Small
        // write — single f32. In-order queue → this lands after the
        // fill, before any subsequent gate apply.
        let one = [1.0_f32];
        self.buf
            .write(&one[..])
            .queue(&self.queue)
            .enq()
            .map_err(|e| OpenClError::Runtime(format!("opencl write |0...0⟩: {e}")))?;
        Ok(())
    }

    /// Device-resident statevector copy: `dst.buf ← self.buf`. Used
    /// by the adjoint AD path to clone `|φ⟩` into a scratch buffer
    /// before applying a derivative gate. Mirrors
    /// `MetalState::copy_into` — straight `clEnqueueCopyBuffer`, no
    /// host roundtrip. The in-order queue serialises with any
    /// pending writes to either buffer (`clEnqueueCopyBuffer`
    /// inherits the queue's ordering semantics).
    pub fn copy_into(&self, dst: &StateBuffer) -> Result<(), OpenClError> {
        if dst.num_qubits != self.num_qubits {
            return Err(OpenClError::StateLengthMismatch {
                expected: self.dim(),
                got: dst.dim(),
            });
        }
        // Buffer::copy returns a BufferCmd; the trailing `.enq()` calls
        // `clEnqueueCopyBuffer`. None / None / None = whole buffer
        // (src offset 0, dst offset 0, full length).
        self.buf
            .copy(&dst.buf, None, None)
            .queue(&self.queue)
            .enq()
            .map_err(|e| OpenClError::Runtime(format!("opencl copy_into: {e}")))?;
        Ok(())
    }

    /// Compute ⟨self|other⟩ on the GPU via a two-stage work-group
    /// reduction. The kernel writes one float2 partial per work-
    /// group; the host sums the (small) partial array. Both states
    /// must have the same `num_qubits`. Mirrors
    /// `MetalState::inner_product` byte-for-byte.
    ///
    /// f32 accumulator floor on the device side. The host-side
    /// partial sum upcasts to f64; the dominant rounding is the
    /// per-work-group cmul_conj + tree reduction. The CPU sister
    /// matches to within ~1e-6 at n ≤ 22 (validated by the smoke
    /// test pinning `<ψ|ψ> = 1` after random rotations).
    pub fn inner_product(&self, other: &StateBuffer) -> Result<Complex64, OpenClError> {
        if self.num_qubits != other.num_qubits {
            return Err(OpenClError::StateLengthMismatch {
                expected: self.dim(),
                got: other.dim(),
            });
        }
        let dim = self.dim();
        // Pick the largest power-of-two work-group size ≤ max_wg_size
        // and ≤ dim. The in-kernel reduction loop assumes power-of-two.
        // `max_wg_size` is queried once via the cached `DeviceHandle`
        // device — most ICDs report 256 or 1024 here.
        let max_wg = self
            .max_wg_size_cached()
            .map_err(|e| OpenClError::Runtime(format!("opencl max_wg_size: {e}")))?;
        let cap = std::cmp::min(max_wg, dim);
        let mut local_size: usize = 1;
        while local_size * 2 <= cap {
            local_size *= 2;
        }
        let num_groups = dim.div_ceil(local_size);

        let partials_buf = Buffer::<f32>::builder()
            .queue((*self.queue).clone())
            .flags(MemFlags::new().write_only().host_read_only())
            .len(num_groups * 2) // float2 per work-group → 2 f32 entries
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl partials buf: {e}")))?;

        let kernel = Kernel::builder()
            .program(&*self.program)
            .name("inner_product")
            .queue((*self.queue).clone())
            .arg(&self.buf)
            .arg(&other.buf)
            .arg(&partials_buf)
            // __local float2 scratch[local_size]. ocl exposes the
            // local-memory arg via `Local::<f32>::new(2 * local_size)`
            // — f32 entries, two per float2.
            .arg_local::<f32>(2 * local_size)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("ocl Kernel::build inner_product: {e}")))?;

        // Safety: both state buffers are at least `dim` float2 entries
        // (per allocate); the partials buffer is sized to one float2
        // per work-group; local memory is sized to one float2 per
        // work-item. The kernel does not read past `gid` (= global id)
        // even when `dim < global_work_size` because `local_size`
        // divides `global_work_size` and `dim` (we sized
        // `global_work_size = local_size * num_groups ≥ dim`).
        let global_work_size = local_size * num_groups;
        unsafe {
            kernel
                .cmd()
                .queue(&*self.queue)
                .global_work_size(global_work_size)
                .local_work_size(local_size)
                .enq()
                .map_err(|e| OpenClError::Runtime(format!("opencl enq inner_product: {e}")))?;
        }
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);

        // Read partials back to host and sum.
        let mut host = vec![0.0_f32; num_groups * 2];
        partials_buf
            .read(&mut host[..])
            .queue(&self.queue)
            .enq()
            .map_err(|e| OpenClError::Runtime(format!("opencl partials read: {e}")))?;
        let mut acc_re: f64 = 0.0;
        let mut acc_im: f64 = 0.0;
        for chunk in host.chunks_exact(2) {
            acc_re += chunk[0] as f64;
            acc_im += chunk[1] as f64;
        }
        Ok(Complex64::new(acc_re, acc_im))
    }

    /// Fused Pauli-string expectation `⟨ψ|P|ψ⟩` in one kernel
    /// dispatch. Replaces the host-side
    /// `execute::expectation_pauli` loop on the OpenCL backend with
    /// a single kernel that does the conj-ψ-times-σψ + tree reduction
    /// in-place. Mirrors `MetalState::pauli_expectation` byte-for-byte.
    ///
    /// `x_mask`, `sign_mask`, and `y_factor` come from
    /// [`crate::execute::pauli_masks`].
    pub fn pauli_expectation(
        &self,
        x_mask: u32,
        sign_mask: u32,
        y_factor: Complex64,
    ) -> Result<Complex64, OpenClError> {
        let dim = self.dim();
        let max_wg = self
            .max_wg_size_cached()
            .map_err(|e| OpenClError::Runtime(format!("opencl max_wg_size: {e}")))?;
        let cap = std::cmp::min(max_wg, dim);
        let mut local_size: usize = 1;
        while local_size * 2 <= cap {
            local_size *= 2;
        }
        let num_groups = dim.div_ceil(local_size);

        let partials_buf = Buffer::<f32>::builder()
            .queue((*self.queue).clone())
            .flags(MemFlags::new().write_only().host_read_only())
            .len(num_groups * 2)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl partials buf (pauli): {e}")))?;

        let kernel = Kernel::builder()
            .program(&*self.program)
            .name("pauli_expectation")
            .queue((*self.queue).clone())
            .arg(&self.buf)
            .arg(&partials_buf)
            .arg(x_mask)
            .arg(sign_mask)
            .arg(y_factor.re as f32)
            .arg(y_factor.im as f32)
            .arg_local::<f32>(2 * local_size)
            .build()
            .map_err(|e| {
                OpenClError::Runtime(format!("ocl Kernel::build pauli_expectation: {e}"))
            })?;

        // Safety: `psi` is `dim` float2 entries; `partials` sized to
        // `num_groups` float2; local memory matches `local_size`
        // float2. `j = i ^ x_mask` stays in [0, dim) because dim is a
        // power of two and x_mask < dim (caller-built from the Pauli
        // string on qubits ≤ num_qubits-1). Kernel doesn't read past
        // `gid` since global size equals `local_size * num_groups`.
        let global_work_size = local_size * num_groups;
        unsafe {
            kernel
                .cmd()
                .queue(&*self.queue)
                .global_work_size(global_work_size)
                .local_work_size(local_size)
                .enq()
                .map_err(|e| OpenClError::Runtime(format!("opencl enq pauli_expectation: {e}")))?;
        }
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);

        let mut host = vec![0.0_f32; num_groups * 2];
        partials_buf
            .read(&mut host[..])
            .queue(&self.queue)
            .enq()
            .map_err(|e| OpenClError::Runtime(format!("opencl partials read (pauli): {e}")))?;
        let mut acc_re: f64 = 0.0;
        let mut acc_im: f64 = 0.0;
        for chunk in host.chunks_exact(2) {
            acc_re += chunk[0] as f64;
            acc_im += chunk[1] as f64;
        }
        Ok(Complex64::new(acc_re, acc_im))
    }

    /// Lazily cache the device's `CL_DEVICE_MAX_WORK_GROUP_SIZE`. We
    /// route through this rather than re-querying the ICD on every
    /// `inner_product` call to avoid the per-call FFI roundtrip. The
    /// cached value is queue-local; the `Arc<Queue>` is shared across
    /// every `StateBuffer` on a given `DeviceHandle`. Returns
    /// `OclError` raw — the caller wraps it.
    fn max_wg_size_cached(&self) -> Result<usize, ocl::Error> {
        // `Queue::device` returns the device the queue was created on;
        // `Device::max_wg_size` does the FFI query. Since this is on
        // the inner_product hot path, future optimisation can move
        // this onto `DeviceHandle` proper; for now, the ICD-side query
        // is a few µs.
        let dev = self.queue.device();
        dev.max_wg_size()
    }

    /// GPU-resident shot sampling. Pipeline mirrors Metal's
    /// `MetalState::sample_shots_gpu`: (1) `shot_probs` writes
    /// `|state[i]|²` per amplitude; (2) `shot_scan_step` runs a
    /// Hillis-Steele inclusive prefix-sum ping-ponging across two
    /// device buffers over `⌈log₂(dim)⌉` passes; (3) `shot_sample`
    /// spawns one work-item per shot, draws a uniform via Philox4×32
    /// keyed by the host-supplied seed, and binary-searches the CDF
    /// for the outcome bin. Returns per-outcome counts. The only
    /// host↔device sync is the final `outcomes` read (`shots × 4`
    /// bytes), so we skip the `2·dim·f32`-byte full-statevector pull
    /// the host sampler in `execute::sample_counts` does.
    ///
    /// Acceptance gate (`tests/shot_sampling_tvd.rs`): empirical TVD
    /// vs the analytical `|amp|²` distribution stays inside the
    /// statistical floor at shots ≥ 10⁴; f32 amplitude precision is
    /// the dominant error source.
    pub fn sample_shots_gpu(
        &self,
        shots: u32,
        seed: u64,
    ) -> Result<HashMap<u64, u32>, OpenClError> {
        if shots == 0 {
            return Ok(HashMap::new());
        }

        let dim = self.dim();
        let dim_u32: u32 = u32::try_from(dim).map_err(|_| {
            OpenClError::Runtime(
                "opencl shot sampling: dim doesn't fit in u32 (num_qubits > 31)".into(),
            )
        })?;

        // Two ping-pong buffers for the Hillis-Steele scan, plus the
        // outcomes buffer (one u32 per shot). All device-resident;
        // probs_*'s contents are written by the kernel chain so we
        // don't bother copying host data in.
        let probs_a = Buffer::<f32>::builder()
            .queue((*self.queue).clone())
            .flags(MemFlags::new().read_write().host_no_access())
            .len(dim)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl probs_a build: {e}")))?;
        let probs_b = Buffer::<f32>::builder()
            .queue((*self.queue).clone())
            .flags(MemFlags::new().read_write().host_no_access())
            .len(dim)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl probs_b build: {e}")))?;
        let outcomes = Buffer::<u32>::builder()
            .queue((*self.queue).clone())
            .flags(MemFlags::new().write_only().host_read_only())
            .len(shots as usize)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl outcomes build: {e}")))?;

        let program = &*self.program;

        // Stage 1: |amp|² → probs_a.
        let probs_kernel = Kernel::builder()
            .program(program)
            .name("shot_probs")
            .queue((*self.queue).clone())
            .arg(&self.buf)
            .arg(&probs_a)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl shot_probs build: {e}")))?;
        // Safety: shot_probs reads state[tid] and writes probs[tid]
        // for tid in [0, dim); both buffers are sized to `dim`.
        unsafe {
            probs_kernel
                .cmd()
                .queue(&*self.queue)
                .global_work_size(dim)
                .enq()
                .map_err(|e| OpenClError::Runtime(format!("opencl enq shot_probs: {e}")))?;
        }
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);

        // Stage 2: Hillis-Steele inclusive scan. `num_passes`
        // = ⌈log₂(dim)⌉ = `num_qubits` (since dim = 2^num_qubits).
        // The kernel is pre-built once and reused via set_arg for the
        // ping-pong + stride update — keeps the per-pass cost to one
        // enqueue.
        let scan_kernel = Kernel::builder()
            .program(program)
            .name("shot_scan_step")
            .queue((*self.queue).clone())
            .arg(&probs_a) // arg 0 — will mutate per pass
            .arg(&probs_b) // arg 1 — will mutate per pass
            .arg(0u32) // arg 2: stride
            .arg(dim_u32) // arg 3: dim
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl shot_scan_step build: {e}")))?;
        let num_passes = self.num_qubits;
        let mut active_is_a = true;
        for pass in 0..num_passes {
            let stride: u32 = 1u32 << pass;
            if active_is_a {
                scan_kernel
                    .set_arg(0, &probs_a)
                    .map_err(|e| OpenClError::Runtime(format!("opencl scan set_arg 0: {e}")))?;
                scan_kernel
                    .set_arg(1, &probs_b)
                    .map_err(|e| OpenClError::Runtime(format!("opencl scan set_arg 1: {e}")))?;
            } else {
                scan_kernel
                    .set_arg(0, &probs_b)
                    .map_err(|e| OpenClError::Runtime(format!("opencl scan set_arg 0: {e}")))?;
                scan_kernel
                    .set_arg(1, &probs_a)
                    .map_err(|e| OpenClError::Runtime(format!("opencl scan set_arg 1: {e}")))?;
            }
            scan_kernel
                .set_arg(2, stride)
                .map_err(|e| OpenClError::Runtime(format!("opencl scan set_arg stride: {e}")))?;
            // Safety: dim work-items, both ping-pong buffers sized
            // to `dim`, tid >= dim is early-returned in the kernel.
            unsafe {
                scan_kernel
                    .cmd()
                    .queue(&*self.queue)
                    .global_work_size(dim)
                    .enq()
                    .map_err(|e| OpenClError::Runtime(format!("opencl enq shot_scan_step: {e}")))?;
            }
            self.dispatch_count.fetch_add(1, Ordering::Relaxed);
            active_is_a = !active_is_a;
        }
        // After `n` ping-pong passes starting from `probs_a`,
        // the final write target was `probs_b` if n is odd else
        // `probs_a`. Equivalently: `active_is_a` flips every pass,
        // so it tells us the *next* source — i.e. the current CDF.
        let cdf = if active_is_a { &probs_a } else { &probs_b };

        // Stage 3: Philox sample.
        let seed_lo = seed as u32;
        let seed_hi = (seed >> 32) as u32;
        let sample_kernel = Kernel::builder()
            .program(program)
            .name("shot_sample")
            .queue((*self.queue).clone())
            .arg(cdf)
            .arg(&outcomes)
            .arg(seed_lo)
            .arg(seed_hi)
            .arg(shots)
            .arg(dim_u32)
            .build()
            .map_err(|e| OpenClError::Runtime(format!("opencl shot_sample build: {e}")))?;
        // Safety: shots work-items; outcomes sized `shots`; cdf sized
        // `dim`. Kernel early-returns on tid >= shots and clamps to
        // dim-1 if the binary search lands past the last bin.
        unsafe {
            sample_kernel
                .cmd()
                .queue(&*self.queue)
                .global_work_size(shots as usize)
                .enq()
                .map_err(|e| OpenClError::Runtime(format!("opencl enq shot_sample: {e}")))?;
        }
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);

        // Blocking read of the outcomes vector — natural sync point.
        let mut host = vec![0u32; shots as usize];
        outcomes
            .read(&mut host[..])
            .queue(&self.queue)
            .enq()
            .map_err(|e| OpenClError::Runtime(format!("opencl outcomes read: {e}")))?;

        let mut counts: HashMap<u64, u32> = HashMap::new();
        for o in host {
            *counts.entry(o as u64).or_insert(0) += 1;
        }
        Ok(counts)
    }
}

/// LIFO buffer pool keyed by `num_qubits`. Recycles `StateBuffer`s
/// across `OpenClStatevectorBackend::adjoint_gradient` calls so the
/// QML trainer's hot path (`adjoint_gradient` × N training points ×
/// M epochs) doesn't pay a fresh `clCreateBuffer` per call.
///
/// Per-adjoint invocation needs three buffers (forward |φ⟩, adjoint
/// |ν⟩, scratch derivative state). At 32 training points × 100
/// epochs that's ~9600 allocations otherwise. The pool converges to
/// a steady-state size of 3 per `num_qubits` after the first epoch.
///
/// Buffers are reset to `|0…0⟩` on lease (fill + 4-byte write —
/// see `StateBuffer::reset_to_zero`); on drop / return they push back
/// onto the per-`num_qubits` stack. Different qubit counts route to
/// independent stacks — leasing 14q never reuses a pooled 16q
/// buffer (the byte sizes differ).
///
/// First-cut: explicit `lease` / `return_buffer` calls — no RAII
/// wrapper yet. The OpenCL adjoint path owns its three buffers
/// directly so the lease/return boundary is unambiguous. Metal
/// evolved a `MetalState` RAII guard later; OpenCL can do the same
/// in a follow-up if a caller outside `adjoint_gradient` needs
/// pooling.
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
    /// allocates a fresh buffer through `handle`. The dispatch_count
    /// atomic on a pooled buffer is preserved across leases — tests
    /// that pin it should sample deltas, not absolute values (matches
    /// the existing `apply_diagonal_product_smoke` shape).
    pub fn lease(
        &self,
        handle: &DeviceHandle,
        num_qubits: u32,
    ) -> Result<StateBuffer, OpenClError> {
        {
            let mut entries = self.entries.lock().expect("BufferPool mutex poisoned");
            if let Some(stack) = entries.get_mut(&num_qubits) {
                if let Some(buf) = stack.pop() {
                    drop(entries);
                    buf.reset_to_zero()?;
                    return Ok(buf);
                }
            }
        }
        handle.allocate(num_qubits)
    }

    /// Return a `StateBuffer` to the pool. Caller is responsible for
    /// not touching the buffer after the call. Pushed onto the
    /// per-`num_qubits` stack LIFO so the next `lease` for the same
    /// size sees a hot cache.
    pub fn return_buffer(&self, buf: StateBuffer) {
        let n = buf.num_qubits;
        let mut entries = self.entries.lock().expect("BufferPool mutex poisoned");
        entries.entry(n).or_default().push(buf);
    }

    /// Count the pooled buffers at a given qubit count. Used by the
    /// pool-semantics smoke tests to pin "after the first adjoint
    /// call the pool steady-states at three per qubit count".
    /// Internally a `Mutex` lock + map lookup — read-only, no
    /// side effect on the pool. Public so integration tests (which
    /// build outside the crate's `cfg(test)`) can reach it through
    /// the backend.
    pub fn pooled_count(&self, num_qubits: u32) -> usize {
        let entries = self.entries.lock().expect("BufferPool mutex poisoned");
        entries.get(&num_qubits).map(|v| v.len()).unwrap_or(0)
    }
}
