//! CUDA-graph capture+replay for the full per-training-point sweep
//! (forward + ν init + adjoint backward).
//!
//! ## Why
//!
//! `forward_graph::ForwardGraph` only captures the forward sweep —
//! ~17% of the trainer's per-step time. The backward sweep (per-op
//! dagger of φ and ν, per-param copy_into + derivative apply +
//! `inner_product`) is the rest, and `inner_product` had to host-sync
//! per `(op, sym)` to read its scalar result. That sync defeats
//! per-launch-cost reduction.
//!
//! The two pieces that close the gap (D1, D2):
//!
//! * `inner_product_accumulate_pooled` — fused reduction that
//!   atomic-adds `2·Re⟨ν|temp⟩·chain` into a device-side gradient
//!   slot. No host roundtrip per `(op, sym)`.
//! * `apply_diagonal_pauli_sum_via_pool` — pooled wrapper so the
//!   gradient observable's per-train-pt coefficients update via
//!   in-place memcpy_htod between replays.
//!
//! With those, the entire per-training-point sequence (forward + ν
//! init + backward + gradient accumulate) becomes a deterministic
//! kernel-launch sequence whose ONLY per-replay change is pool
//! contents — exactly what CUDA Graphs can amortize.
//!
//! ## Captured sequence (one graph per circuit shape)
//!
//! 1. memset state to zero, memcpy 1.0 to state[0] → state = |0⟩
//! 2. memset grad_dev to zero
//! 3. Forward sweep: walk `unitary_ops`, dispatch
//!    `apply_*_via_pool(forward_pool, slot)` per op
//! 4. ν = O|ψ⟩ via `apply_diagonal_pauli_sum_via_pool` (Z-only obs)
//! 5. Backward sweep, walking `unitary_ops` reversed:
//!    a. `apply_*_via_pool(dagger_pool, slot)` on φ
//!    b. For each `(op, param_idx, sym)`:
//!    i.   memcpy_dtod φ → temp
//!    ii.  `apply_*_via_pool(deriv_pool, slot)` on temp
//!    iii. `inner_product_accumulate_pooled` reads (ν, temp),
//!    writes into `grad_dev[sym_slot]` via atomic add
//!    c. `apply_*_via_pool(dagger_pool, slot)` on ν
//! 6. memcpy_dtoh grad_dev → grad_host (the only host-visible side
//!    effect of the replay, beyond the captured state buffer)
//!
//! ## Per-replay host work
//!
//! * Resolve every `(op, param_idx, sym)`'s chain factor
//! * Compute every per-op gate matrix (forward, dagger, derivative)
//! * Compute the gradient observable's per-term coefficients
//! * memcpy_htod into the corresponding pools + reset state + grad
//! * `graph.launch()` then sync, read grad_host
//!
//! Aside from the host CPU work above, one `graph.launch()` replays
//! the entire kernel sequence with replay overhead ~0.3 µs/node
//! instead of ~2.1 µs/cuLaunchKernel — the lever the Phase 4c
//! 16 → ≤4 s gap is held open by.
//!
//! ## Scope of this slice
//!
//! Same gate set the ForwardGraph supports (H/X/Y/Z/S/Sdg/T/Tdg/Rx/
//! Ry/Rz/U1/U2/U3/CX/CY/CZ/Swap/CRz/CU3). 3q gates (CCX/CSwap),
//! Reset, Measure, conditional gates, Barrier, and the photonic /
//! Custom kinds are rejected at build time. The gradient observable
//! must be Z-only (the QML trainer's residual gradient is). The
//! trainer's general expectation_multi_then_gradient still falls
//! back to the non-graph path for any of these unsupported shapes.
//!
//! Symbolic-param chain rule supports `Symbol(s)`, `Concrete(_)`
//! (chain = 0 ⇒ skipped), `Add(...)`, `Mul(...)`, `Negate(...)` —
//! enough for the trainer's flat-Symbol ansatzes plus the
//! rotation-merged `Add(Concrete, Symbol)` shape.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Replay-time profiling counters. Enabled when env var
/// `OMEGA_TRAINSTEP_PROFILE=1` is set the first time
/// `replay_with_y_labels` is called. Each counter accumulates
/// elapsed nanoseconds across all replays in the process; the
/// totals are printed (and reset) when `OMEGA_TRAINSTEP_PROFILE_DUMP`
/// is set or by calling [`dump_replay_profile`] directly.
///
/// Used as one-shot instrumentation while closing the Phase 4c
/// gap — not part of the public surface.
#[allow(dead_code)]
static REPLAY_PROFILE_INIT: AtomicBool = AtomicBool::new(false);
#[allow(dead_code)]
static REPLAY_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
#[allow(dead_code)]
static REPLAY_NS_RESOLVE: AtomicU64 = AtomicU64::new(0);
#[allow(dead_code)]
static REPLAY_NS_UPLOAD: AtomicU64 = AtomicU64::new(0);
#[allow(dead_code)]
static REPLAY_NS_LAUNCH: AtomicU64 = AtomicU64::new(0);
#[allow(dead_code)]
static REPLAY_NS_GRAD_DTOH: AtomicU64 = AtomicU64::new(0);
#[allow(dead_code)]
static REPLAY_COUNT: AtomicU64 = AtomicU64::new(0);

#[allow(dead_code)]
fn replay_profile_enabled() -> bool {
    if !REPLAY_PROFILE_INIT.swap(true, Ordering::Relaxed) {
        let on = std::env::var_os("OMEGA_TRAINSTEP_PROFILE").is_some();
        REPLAY_PROFILE_ENABLED.store(on, Ordering::Relaxed);
    }
    REPLAY_PROFILE_ENABLED.load(Ordering::Relaxed)
}

/// Dump (and reset) the replay-profile counters to stderr. Safe to
/// call when profiling is disabled (no-op).
pub fn dump_replay_profile() {
    if !REPLAY_PROFILE_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let n = REPLAY_COUNT.swap(0, Ordering::Relaxed);
    if n == 0 {
        return;
    }
    let resolve = REPLAY_NS_RESOLVE.swap(0, Ordering::Relaxed);
    let upload = REPLAY_NS_UPLOAD.swap(0, Ordering::Relaxed);
    let launch = REPLAY_NS_LAUNCH.swap(0, Ordering::Relaxed);
    let dtoh = REPLAY_NS_GRAD_DTOH.swap(0, Ordering::Relaxed);
    let total = resolve + upload + launch + dtoh;
    eprintln!(
        "TrainStepGraph replay profile (n={}): total {:.2}ms = resolve {:.2}ms ({:.1}%) + upload {:.2}ms ({:.1}%) + launch+sync {:.2}ms ({:.1}%) + grad_dtoh {:.2}ms ({:.1}%)",
        n,
        total as f64 / 1e6,
        resolve as f64 / 1e6,
        100.0 * resolve as f64 / total.max(1) as f64,
        upload as f64 / 1e6,
        100.0 * upload as f64 / total.max(1) as f64,
        launch as f64 / 1e6,
        100.0 * launch as f64 / total.max(1) as f64,
        dtoh as f64 / 1e6,
        100.0 * dtoh as f64 / total.max(1) as f64,
    );
}

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode_enum};
use cudarc::driver::{CudaContext, CudaGraph, CudaSlice, LaunchConfig, PushKernelArg};
use num_complex::Complex64;

use omega_backend_statevector::gates;
use omega_core::circuit::{CircuitIR, GateKind, GateOp, ParamExpr, SymbolId};
use omega_core::error::{OmegaError, Result as OmegaResult};
use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;

use crate::imp::{Apply1qParams, Apply2qParams, DeviceHandle, DiagonalParams, StateBuffer};
use crate::CudaError;

/// Static structure of a captured training step. The "shape" of a
/// captured graph — gate kinds + qubits + symbol layout — that's
/// fixed across replays. Per-replay only the pool CONTENTS change.
struct GraphPlan {
    /// One entry per kernel launch in the captured sequence,
    /// in the order issued during capture.
    entries: Vec<PlanEntry>,
    /// Number of slots in each pool — sized so every plan entry's
    /// `slot` is in range.
    n_forward_diag: usize,
    n_forward_1q: usize,
    n_forward_2q: usize,
    n_dagger_diag: usize,
    n_dagger_1q: usize,
    n_dagger_2q: usize,
    n_deriv_1q: usize,
    n_deriv_2q: usize,
    n_accumulate: usize,
    /// Number of unique symbols in the circuit. Equals the size of
    /// `grad_dev` and the length of `sym_to_slot.values()`.
    num_symbols: usize,
    /// Map from circuit `SymbolId` → its slot in `grad_dev`. Stable
    /// across replays.
    sym_to_slot: HashMap<SymbolId, u32>,
    /// Z-only gradient observable shape: per-term sign mask. Constant
    /// per circuit shape; coefficients update per replay.
    gradient_obs_sign_masks: Vec<u32>,
    /// Per-(op, param_idx, sym) record of how to populate the
    /// accumulate pool's `chain_pool` entry from the binding —
    /// captured at build time so replays just call
    /// `params.resolve_derivative` per accumulate slot.
    accumulate_chain_recipe: Vec<ChainRecipe>,
    /// For each backward derivative slot: which (op_idx, param_idx)
    /// it corresponds to. Used to recompute the derivative matrix
    /// per replay.
    deriv_1q_recipe: Vec<DerivRecipe>,
    deriv_2q_recipe: Vec<DerivRecipe>,
    /// For each forward / dagger slot: which `op_idx` it owns.
    forward_diag_op: Vec<usize>,
    forward_1q_op: Vec<usize>,
    forward_2q_op: Vec<usize>,
    dagger_diag_op: Vec<usize>,
    dagger_1q_op: Vec<usize>,
    dagger_2q_op: Vec<usize>,
    /// Stage G6: the trailing run of param'd RZ ops on disjoint
    /// qubits whose gradient + dagger blocks have been collapsed
    /// into the chain-fused `PauliZChainAccumulate` +
    /// `DaggerBothDiagChain` pair. Each entry records the
    /// information needed to populate the chain pools per replay.
    /// Empty when the circuit has no trailing chain that qualifies.
    pauli_z_chain_recipe: Vec<PauliZChainRecipe>,
}

#[derive(Clone, Copy)]
struct ChainRecipe {
    /// op index in `circuit.ops` whose param_expr is the source of
    /// the chain rule.
    op_idx: usize,
    /// which param slot of that op is being differentiated.
    param_idx: usize,
    /// the symbol id whose gradient slot this accumulate writes
    /// into.
    sym_id: SymbolId,
}

#[derive(Clone, Copy)]
struct DerivRecipe {
    op_idx: usize,
    param_idx: usize,
}

/// Stage G6: per-RZ-chain-entry recipe. Captures the info needed
/// to (re)populate the `pauli_z_chain_pool` and
/// `pauli_z_chain_dagger_pool` per replay — the qubit (constant
/// per circuit shape, but stored alongside for kernel reads), the
/// destination grad_dev slot for the gradient, and the (op, param)
/// pair for the autodiff chain factor + the dagger matrix.
#[derive(Clone, Copy)]
struct PauliZChainRecipe {
    op_idx: usize,
    param_idx: usize,
    sym_id: SymbolId,
    qubit: u32,
}

#[derive(Clone, Copy)]
enum PlanEntry {
    ForwardDiag {
        slot: u32,
    },
    Forward1q {
        slot: u32,
    },
    Forward2q {
        slot: u32,
    },
    /// Stage E: per-measurement-qubit ⟨Z⟩ written to
    /// predictions_dev[slot]. Captured between the forward sweep
    /// and the residual-coefficient builder.
    PredictZ {
        sign_mask: u32,
        slot: u32,
    },
    /// Stage F2: in-graph residual-coefficient builder kernel.
    /// Reads `predictions_dev` (filled by PredictZ) + `y_label_dev`
    /// (populated host→device per replay), writes
    /// `nu_coeffs_pool[i] = 2·(predictions[i] - y_label[i])`. Pure-
    /// device replacement for the Stage E
    /// `cuLaunchHostFunc` host callback + surrounding memcpy
    /// chain — no CUDA worker thread context switch, three graph
    /// nodes collapsed into one.
    ComputeResidualCoeffs,
    NuInit,
    DaggerPhiDiag {
        slot: u32,
    },
    DaggerPhi1q {
        slot: u32,
    },
    DaggerPhi2q {
        slot: u32,
    },
    CopyPhiToTemp,
    Deriv1q {
        slot: u32,
    },
    Deriv2q {
        slot: u32,
    },
    /// Stage G1: fused `CopyPhiToTemp + Deriv1q`. One kernel reads
    /// φ and writes (∂U · φ) to temp. Replaces a `memcpy_dtod`
    /// graph node + an `apply_1q_pooled` graph node with a single
    /// `apply_1q_from_to_pooled` node — saves one graph node per
    /// 1q parameterised gate (≈16 nodes on the n=14/16-param HEA).
    #[allow(dead_code)] // Stage G2 supersedes this for the 1q path.
    CopyPhiAndDeriv1q {
        slot: u32,
    },
    /// Stage G2: triple fusion `CopyPhiToTemp + Deriv1q +
    /// Accumulate`. One kernel reads φ + ν, forms the deriv'd
    /// amplitudes from `deriv_1q_pool[deriv_slot]` per pair-thread,
    /// contracts with ν, block-reduces, and atomic-adds
    /// `2 · chain · Re⟨ν|∂U·φ⟩` into `grad_dev[sym]`. Replaces the
    /// (`CopyPhiAndDeriv1q` + `Accumulate`) pair with one node —
    /// saves another ~16 nodes on the 1q-deriv path.
    Deriv1qInnerProductAccumulate {
        deriv_slot: u32,
        accum_slot: u32,
    },
    Accumulate {
        slot: u32,
    },
    DaggerNuDiag {
        slot: u32,
    },
    DaggerNu1q {
        slot: u32,
    },
    DaggerNu2q {
        slot: u32,
    },
    /// Stage G3: dual-state DaggerBoth2q. One kernel applies U†
    /// from `dagger_2q_pool[slot]` to BOTH `phi` and `nu` in one
    /// pass — emitted instead of the separate `DaggerPhi2q` +
    /// `DaggerNu2q` pair when the underlying 2q op has NO
    /// symbol-bearing parameters (CNOT, CZ on Phase 4c). The plan
    /// allocates a single shared dagger_2q slot for the pair so the
    /// pool memcpy stays the same shape.
    DaggerBoth2q {
        slot: u32,
    },
    /// Stage G4: 1q analogue of `DaggerBoth2q`. Emitted for
    /// non-parameterised 1q ops (H, X, Y, Z, S, T) where the per-
    /// op deriv loop is empty so DaggerPhi1q + DaggerNu1q are
    /// adjacent and fuse-able.
    DaggerBoth1q {
        slot: u32,
    },
    /// Stage G5: chained forward-diagonal apply. Walks
    /// `forward_diag_pool[start..start+count]` and applies all
    /// `count` diagonals to φ in one kernel. Replaces a run of
    /// `count` consecutive `ForwardDiag` entries with one node,
    /// saving `count - 1` graph nodes (`count = 8` on the Phase 4c
    /// HEA Layer 2 RZ chain → -7 nodes).
    ForwardDiagChain {
        start: u32,
        count: u32,
    },
    /// Stage G6: chained Pauli-Z gradient accumulator. Computes
    /// `gradient[sym_k] += chain_k · Im⟨ν|Z_{q_k}|φ⟩` for every
    /// chain term in `pauli_z_chain_pool[0..count]`, in a single
    /// launch. Uses pre-dagger (φ_n, ν_n) for ALL terms via the
    /// commutation identity (Z_q commutes with later RZ daggers
    /// in the chain).
    PauliZChainAccumulate {
        count: u32,
    },
    /// Stage G6: dual-state diagonal chain dagger. Walks
    /// `pauli_z_chain_dagger_pool[0..count]` and applies all K
    /// daggers to BOTH φ and ν in one kernel. Emitted right after
    /// `PauliZChainAccumulate` so subsequent backward iterations
    /// see the post-dagger states.
    DaggerBothDiagChain {
        count: u32,
    },
    /// Stage G7: per-op Pauli-Y triple fusion. Replaces
    /// (DaggerPhi1q + Deriv1qInnerProductAccumulate + DaggerNu1q)
    /// for parameterised RY ops with one kernel — uses the
    /// generator identity ∂U·U† = -i/2·Y_q and writes φ_{k-1} +
    /// ν_{k-1} as part of the same launch.
    Pauli1qYDaggerBothFused {
        dagger_slot: u32,
        accum_slot: u32,
    },
}

/// Userdata for the `cuLaunchHostFunc` host callback. Holds raw
/// pointers into stable-address Vec<> data inside the
/// `TrainStepGraph` struct. Once allocated, the Vec data pointers
/// don't move (Rust moves the Vec field but heap data stays put);
/// the Box wrapping this struct gives a stable address for the
/// userData pointer the captured graph hands the callback.
// Stage F2 retired the cuLaunchHostFunc host callback in favour of
// a pure-device `compute_residual_coeffs` kernel — see the docstring
// on that kernel. The HostCallbackContext + host_residual_coeffs_callback
// types below are kept dead-code-allowed as a record of the
// approach, since they exercise a different cudarc capture pattern
// (cuLaunchHostFunc / pinned host memory bridging GPU phases) that
// could be useful for non-residual factories down the line.
#[allow(dead_code)]
struct HostCallbackContext {
    num_outputs: usize,
    predictions_host: *const f64,
    y_label_host: *const f32,
    coeffs_host: *mut f32,
}

// Send/Sync are required because the captured graph runs the host
// function on a CUDA-managed worker thread that's distinct from
// the calling thread. The raw pointers point at memory owned by
// TrainStepGraph (which itself is `&mut` accessed only on the
// calling thread between graph.launch() invocations) — so the
// callback running on a worker thread is safe as long as the
// host_func runs only as part of graph.launch and the graph is not
// concurrently re-launched.
unsafe impl Send for HostCallbackContext {}
unsafe impl Sync for HostCallbackContext {}

/// Page-locked host buffer with manual lifetime management. We
/// can't use cudarc's safe `PinnedHostSlice` here because its
/// `as_ptr`/`as_mut_ptr` accessors call `cuEventSynchronize` on an
/// internal event that was never recorded, returning
/// CUDA_ERROR_INVALID_VALUE. Bypassing the cudarc wrapper and
/// going straight to `malloc_host` + `free_host` gives us a raw
/// pointer we can hand to the host_func callback (whose userData
/// must be a stable pointer to host memory) and a `&mut [T]` that
/// cudarc's `HostSlice<T> for [T]` impl already supports for
/// captured `memcpy_htod` / `memcpy_dtoh`.
///
/// Allocated with flags=0 (cuMemAllocHost-equivalent default) —
/// ordinary pinned memory, not write-combined. Both host reads
/// and host writes are cached, which the host_func callback
/// needs (it reads predictions then writes coeffs).
struct PinnedBuf<T> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T> Send for PinnedBuf<T> {}
unsafe impl<T> Sync for PinnedBuf<T> {}

impl<T> PinnedBuf<T> {
    fn new(len: usize) -> Result<Self, CudaError> {
        let bytes = len.max(1) * std::mem::size_of::<T>();
        let raw = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
            .map_err(|e| CudaError::Driver(format!("malloc_host: {e}")))?;
        let ptr = raw as *mut T;
        // Zero-init so first replay reads aren't surprised by
        // uninitialised contents.
        unsafe {
            std::ptr::write_bytes(ptr, 0u8, len.max(1));
        }
        Ok(Self {
            ptr,
            len: len.max(1),
        })
    }

    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        // Best-effort free; ignore errors during teardown.
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr as *mut std::ffi::c_void) };
    }
}

/// extern "C" trampoline that the captured graph invokes per
/// replay. Reads `predictions_host` + `y_label_host`, writes the
/// MSE-residual gradient observable's coefficients to
/// `coeffs_host`. Pure CPU; no CUDA calls.
#[allow(dead_code)]
unsafe extern "C" fn host_residual_coeffs_callback(user_data: *mut std::ffi::c_void) {
    let ctx = unsafe { &*(user_data as *const HostCallbackContext) };
    for i in 0..ctx.num_outputs {
        let pred = unsafe { *ctx.predictions_host.add(i) };
        let label = unsafe { *ctx.y_label_host.add(i) };
        let coeff = 2.0_f32 * (pred as f32 - label);
        unsafe {
            *ctx.coeffs_host.add(i) = coeff;
        }
    }
}

/// Captured forward + ν-init + adjoint sequence. See module docs.
pub(crate) struct TrainStepGraph {
    plan: GraphPlan,
    handle: DeviceHandle, // graph stream + kernels (graph-stream variant)
    phi: StateBuffer,
    nu: StateBuffer,
    temp: StateBuffer,

    forward_diag_pool: CudaSlice<DiagonalParams>,
    forward_1q_pool: CudaSlice<Apply1qParams>,
    forward_2q_pool: CudaSlice<Apply2qParams>,
    nu_sign_masks_pool: CudaSlice<u32>,
    nu_coeffs_pool: CudaSlice<f32>,
    dagger_diag_pool: CudaSlice<DiagonalParams>,
    dagger_1q_pool: CudaSlice<Apply1qParams>,
    dagger_2q_pool: CudaSlice<Apply2qParams>,
    deriv_1q_pool: CudaSlice<Apply1qParams>,
    deriv_2q_pool: CudaSlice<Apply2qParams>,
    chain_pool: CudaSlice<f64>,
    sym_idx_pool: CudaSlice<u32>,
    /// Stage G6: per-RZ-chain-entry pool keyed by chain order,
    /// holding (qubit, sym, chain) per term. Populated host-side
    /// per replay (chain factor varies with `params`); the captured
    /// `pauli_z_chain_accumulate` kernel reads it directly to
    /// derive each term's gradient contribution.
    pauli_z_chain_pool: CudaSlice<crate::imp::PauliZChainEntry>,
    /// Stage G6: dagger-matrix pool for the chain. K entries (one
    /// `DiagonalParams` per chain RZ), updated host-side per replay
    /// just like `dagger_diag_pool`. Read by the
    /// `apply_diagonal_chain_dual_pooled` kernel.
    pauli_z_chain_dagger_pool: CudaSlice<DiagonalParams>,
    grad_dev: CudaSlice<f64>,
    /// Stage E — predictions buffer (one f64 per measurement
    /// qubit). Filled by the in-graph `pauli_z_expectation_to_slot`
    /// kernels; consumed by `compute_residual_coeffs` (device-side)
    /// and by the post-replay `clone_dtoh` for the trainer's
    /// `predictions` return value.
    predictions_dev: CudaSlice<f64>,
    /// Stage F2 — device-side y_labels buffer. Populated host→
    /// device per replay (one small memcpy_htod issued by
    /// `replay_with_y_labels` before `graph.launch()`). The
    /// captured `compute_residual_coeffs` kernel reads this plus
    /// `predictions_dev` and writes coefficients into
    /// `nu_coeffs_pool` — entirely on-device, no host_func.
    y_label_dev: CudaSlice<f32>,
    /// Pageable host staging Vec for y_labels. `replay_with_y_labels`
    /// writes the per-replay values here, then issues one
    /// memcpy_htod into `y_label_dev`. Plain Vec is fine because
    /// the memcpy happens OUTSIDE the captured graph.
    y_label_host: Vec<f32>,

    forward_diag_host: Vec<DiagonalParams>,
    forward_1q_host: Vec<Apply1qParams>,
    forward_2q_host: Vec<Apply2qParams>,
    nu_coeffs_host: Vec<f32>,
    dagger_diag_host: Vec<DiagonalParams>,
    dagger_1q_host: Vec<Apply1qParams>,
    dagger_2q_host: Vec<Apply2qParams>,
    deriv_1q_host: Vec<Apply1qParams>,
    deriv_2q_host: Vec<Apply2qParams>,
    chain_host: Vec<f64>,
    grad_host: Vec<f64>,
    /// Stage G6 staging: host buffers for `pauli_z_chain_pool` +
    /// `pauli_z_chain_dagger_pool`. Sized to
    /// `plan.pauli_z_chain_recipe.len()`. Plain Vec — the memcpy
    /// runs outside the captured graph.
    pauli_z_chain_host: Vec<crate::imp::PauliZChainEntry>,
    pauli_z_chain_dagger_host: Vec<DiagonalParams>,

    graph: CudaGraph,
}

impl TrainStepGraph {
    /// Build + capture the graph for `circuit` paired with the
    /// fixed-shape gradient observable `gradient_obs_template`. The
    /// observable's term VALUES (coefficients) are ignored — only
    /// the per-term sign masks are extracted from it for the
    /// captured `apply_diagonal_pauli_sum_via_pool` call. Real
    /// coefficients arrive at replay time via
    /// [`Self::replay_with_coeffs`].
    ///
    /// Returns `Err(Unsupported)` when the circuit shape (Reset /
    /// Measure / 3q / conditional / Barrier / photonic / Custom) or
    /// the observable shape (any X/Y component) isn't supported by
    /// this slice — the trainer falls back to the non-graph adjoint
    /// path in that case.
    pub fn capture(
        handle: &DeviceHandle,
        circuit: &CircuitIR,
        gradient_obs_template: &Observable,
    ) -> OmegaResult<Self> {
        let plan = build_plan(circuit, gradient_obs_template)?;

        // Same dance as ForwardGraph: legacy default stream can't
        // be captured, and cudarc 0.19 per-slice event tracking
        // invalidates capture, so toggle event tracking off across
        // the capture-window allocations and back on after.
        unsafe {
            handle.ctx.disable_event_tracking();
        }
        let graph_stream = handle
            .ctx
            .new_stream()
            .map_err(|e| OmegaError::Backend(format!("new_stream for train graph: {e}")))?;
        let graph_handle = DeviceHandle {
            ctx: handle.ctx.clone(),
            stream: graph_stream.clone(),
            kernels: handle.kernels.clone(),
        };

        let n = circuit.num_qubits;
        let phi = graph_handle
            .allocate(n)
            .map_err(|e| OmegaError::Backend(format!("alloc phi: {e}")))?;
        let nu = graph_handle
            .allocate(n)
            .map_err(|e| OmegaError::Backend(format!("alloc nu: {e}")))?;
        let temp = graph_handle
            .allocate(n)
            .map_err(|e| OmegaError::Backend(format!("alloc temp: {e}")))?;

        let forward_diag_pool = graph_stream
            .alloc_zeros::<DiagonalParams>(plan.n_forward_diag.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc forward_diag pool: {e}")))?;
        let forward_1q_pool = graph_stream
            .alloc_zeros::<Apply1qParams>(plan.n_forward_1q.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc forward_1q pool: {e}")))?;
        let forward_2q_pool = graph_stream
            .alloc_zeros::<Apply2qParams>(plan.n_forward_2q.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc forward_2q pool: {e}")))?;
        let nu_sign_masks_pool = graph_stream
            .alloc_zeros::<u32>(plan.gradient_obs_sign_masks.len().max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc nu sign_masks pool: {e}")))?;
        let nu_coeffs_pool = graph_stream
            .alloc_zeros::<f32>(plan.gradient_obs_sign_masks.len().max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc nu coeffs pool: {e}")))?;
        let dagger_diag_pool = graph_stream
            .alloc_zeros::<DiagonalParams>(plan.n_dagger_diag.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc dagger_diag pool: {e}")))?;
        let dagger_1q_pool = graph_stream
            .alloc_zeros::<Apply1qParams>(plan.n_dagger_1q.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc dagger_1q pool: {e}")))?;
        let dagger_2q_pool = graph_stream
            .alloc_zeros::<Apply2qParams>(plan.n_dagger_2q.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc dagger_2q pool: {e}")))?;
        let deriv_1q_pool = graph_stream
            .alloc_zeros::<Apply1qParams>(plan.n_deriv_1q.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc deriv_1q pool: {e}")))?;
        let deriv_2q_pool = graph_stream
            .alloc_zeros::<Apply2qParams>(plan.n_deriv_2q.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc deriv_2q pool: {e}")))?;
        let chain_pool = graph_stream
            .alloc_zeros::<f64>(plan.n_accumulate.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc chain pool: {e}")))?;
        let sym_idx_pool = graph_stream
            .alloc_zeros::<u32>(plan.n_accumulate.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc sym_idx pool: {e}")))?;
        // Stage G6 — Pauli-Z chain pools. Sized to the trailing
        // chain length (typically 8 on Phase 4c HEA; 0 for shapes
        // without a qualifying trailing chain). Allocate at
        // .max(1) to keep the CudaSlice non-empty.
        let n_pauli_z_chain = plan.pauli_z_chain_recipe.len();
        let pauli_z_chain_pool = graph_stream
            .alloc_zeros::<crate::imp::PauliZChainEntry>(n_pauli_z_chain.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc pauli_z_chain pool: {e}")))?;
        let pauli_z_chain_dagger_pool = graph_stream
            .alloc_zeros::<DiagonalParams>(n_pauli_z_chain.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc pauli_z_chain_dagger pool: {e}")))?;
        let grad_dev = graph_stream
            .alloc_zeros::<f64>(plan.num_symbols.max(1))
            .map_err(|e| OmegaError::Backend(format!("alloc grad_dev: {e}")))?;
        // Stage E — predictions_dev (one f64 per measurement
        // qubit). Captured pauli_z_expectation_to_slot kernels
        // atomic-add into this buffer; captured memcpy_dtoh ferries
        // it to predictions_host for the host_func.
        let n_outputs = plan.gradient_obs_sign_masks.len().max(1);
        let predictions_dev = graph_stream
            .alloc_zeros::<f64>(n_outputs)
            .map_err(|e| OmegaError::Backend(format!("alloc predictions_dev: {e}")))?;
        // Stage F2 — y_label device buffer must also be allocated
        // inside the disable_event_tracking window, otherwise the
        // alloc event gets waited on inside the capture loop and
        // invalidates the captured graph.
        let y_label_dev = graph_stream
            .alloc_zeros::<f32>(n_outputs)
            .map_err(|e| OmegaError::Backend(format!("alloc y_label_dev: {e}")))?;

        unsafe {
            handle.ctx.enable_event_tracking();
        }

        // Pre-size host vectors to match the plan; populate with
        // zero / default entries so capture-time launches see valid
        // memory at slot 0..n.
        let mut forward_diag_host = vec![DiagonalParams::default(); plan.n_forward_diag.max(1)];
        let _ = &mut forward_diag_host; // silence rust-analyzer's mut-but-not-mut warn
        let forward_1q_host = vec![Apply1qParams::default(); plan.n_forward_1q.max(1)];
        let forward_2q_host = vec![Apply2qParams::default(); plan.n_forward_2q.max(1)];
        let nu_coeffs_host = vec![0.0f32; plan.gradient_obs_sign_masks.len().max(1)];
        let dagger_diag_host = vec![DiagonalParams::default(); plan.n_dagger_diag.max(1)];
        let dagger_1q_host = vec![Apply1qParams::default(); plan.n_dagger_1q.max(1)];
        let dagger_2q_host = vec![Apply2qParams::default(); plan.n_dagger_2q.max(1)];
        let deriv_1q_host = vec![Apply1qParams::default(); plan.n_deriv_1q.max(1)];
        let deriv_2q_host = vec![Apply2qParams::default(); plan.n_deriv_2q.max(1)];
        let chain_host = vec![0.0f64; plan.n_accumulate.max(1)];
        let grad_host = vec![0.0f64; plan.num_symbols.max(1)];
        // Stage G6 — host staging for the Pauli-Z chain pools.
        let pauli_z_chain_host =
            vec![crate::imp::PauliZChainEntry::default(); n_pauli_z_chain.max(1)];
        let pauli_z_chain_dagger_host = vec![DiagonalParams::default(); n_pauli_z_chain.max(1)];

        // Stage F2 — host staging buffer for y_labels. The matching
        // device buffer (`y_label_dev`) is allocated above inside
        // the disable_event_tracking window so its alloc event
        // doesn't bleed into the capture.
        let y_label_host = vec![0.0_f32; n_outputs];

        // Upload the static sign_masks pool — its contents are
        // constant across all replays.
        let mut nu_sign_masks_pool = nu_sign_masks_pool;
        if !plan.gradient_obs_sign_masks.is_empty() {
            graph_stream
                .memcpy_htod(&plan.gradient_obs_sign_masks[..], &mut nu_sign_masks_pool)
                .map_err(|e| OmegaError::Backend(format!("upload sign_masks pool: {e}")))?;
        }
        // Upload the static sym_idx pool — also constant per shape.
        let mut sym_idx_pool = sym_idx_pool;
        if !plan.accumulate_chain_recipe.is_empty() {
            let sym_idx: Vec<u32> = plan
                .accumulate_chain_recipe
                .iter()
                .map(|r| *plan.sym_to_slot.get(&r.sym_id).expect("sym slot exists"))
                .collect();
            graph_stream
                .memcpy_htod(&sym_idx[..], &mut sym_idx_pool)
                .map_err(|e| OmegaError::Backend(format!("upload sym_idx pool: {e}")))?;
        }
        graph_stream
            .synchronize()
            .map_err(|e| OmegaError::Backend(format!("sync after static uploads: {e}")))?;

        // Now bind the per-replay-mutable state into a local scope
        // we'll reuse for the captured launches and for the
        // returned struct.
        let mut phi = phi;
        let mut nu = nu;
        let mut temp = temp;
        let mut forward_diag_pool = forward_diag_pool;
        let mut forward_1q_pool = forward_1q_pool;
        let mut forward_2q_pool = forward_2q_pool;
        let mut nu_coeffs_pool = nu_coeffs_pool;
        let mut dagger_diag_pool = dagger_diag_pool;
        let mut dagger_1q_pool = dagger_1q_pool;
        let mut dagger_2q_pool = dagger_2q_pool;
        let mut deriv_1q_pool = deriv_1q_pool;
        let mut deriv_2q_pool = deriv_2q_pool;
        let chain_pool = chain_pool;
        let mut grad_dev = grad_dev;
        let mut predictions_dev = predictions_dev;
        let y_label_dev = y_label_dev;
        let mut pauli_z_chain_pool = pauli_z_chain_pool;
        let mut pauli_z_chain_dagger_pool = pauli_z_chain_dagger_pool;
        let _ = (
            &mut forward_diag_pool,
            &mut forward_1q_pool,
            &mut forward_2q_pool,
            &mut dagger_diag_pool,
            &mut dagger_1q_pool,
            &mut dagger_2q_pool,
            &mut deriv_1q_pool,
            &mut deriv_2q_pool,
            &mut pauli_z_chain_pool,
            &mut pauli_z_chain_dagger_pool,
        ); // placate borrow checker; pools mutated only in replay

        // Begin capture; walk the plan; end capture.
        graph_stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| OmegaError::Backend(format!("begin_capture: {e}")))?;

        let result: Result<(), CudaError> = (|| {
            // 1. Reset state to |0...0⟩ inside the captured graph
            //    so every replay starts fresh.
            graph_stream
                .memset_zeros(&mut phi.state)
                .map_err(|e| CudaError::Driver(format!("capture phi memset: {e}")))?;
            // Tiny memcpy of the |0⟩ amplitude. STATE_INIT is a
            // static so the captured host pointer stays valid.
            const STATE_INIT: [f32; 1] = [1.0];
            {
                let mut head = phi.state.slice_mut(0..1);
                graph_stream
                    .memcpy_htod(&STATE_INIT[..], &mut head)
                    .map_err(|e| CudaError::Driver(format!("capture phi init: {e}")))?;
            }
            // 2. Reset grad_dev + predictions_dev to zero (atomic
            // adds need a clean slate every replay).
            graph_stream
                .memset_zeros(&mut grad_dev)
                .map_err(|e| CudaError::Driver(format!("capture grad memset: {e}")))?;
            graph_stream
                .memset_zeros(&mut predictions_dev)
                .map_err(|e| CudaError::Driver(format!("capture predictions memset: {e}")))?;

            for entry in &plan.entries {
                match *entry {
                    PlanEntry::ForwardDiag { slot } => {
                        phi.apply_diagonal_via_pool(&forward_diag_pool, slot)?;
                    }
                    PlanEntry::ForwardDiagChain { start, count } => {
                        phi.apply_diagonal_chain_via_pool(&forward_diag_pool, start, count)?;
                    }
                    PlanEntry::Forward1q { slot } => {
                        phi.apply_1q_via_pool(&forward_1q_pool, slot)?;
                    }
                    PlanEntry::Forward2q { slot } => {
                        phi.apply_2q_via_pool(&forward_2q_pool, slot)?;
                    }
                    PlanEntry::PredictZ { sign_mask, slot } => {
                        phi.pauli_z_expectation_to_slot(&mut predictions_dev, sign_mask, slot)?;
                    }
                    PlanEntry::ComputeResidualCoeffs => {
                        // Pure-device residual-coefficient builder.
                        // Reads predictions_dev + y_label_dev,
                        // writes nu_coeffs_pool. Replaces Stage E's
                        // memcpy_dtoh + cuLaunchHostFunc + memcpy_htod
                        // chain with a single kernel launch — no
                        // CUDA worker thread context switch, three
                        // graph nodes collapsed into one.
                        let n_outputs_u32 = plan.gradient_obs_sign_masks.len() as u32;
                        let cfg = LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (n_outputs_u32.max(1), 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let func = handle.kernels.compute_residual_coeffs.clone();
                        let mut builder = graph_stream.launch_builder(&func);
                        builder
                            .arg(&predictions_dev)
                            .arg(&y_label_dev)
                            .arg(&mut nu_coeffs_pool)
                            .arg(&n_outputs_u32);
                        unsafe {
                            builder.launch(cfg).map_err(|e| {
                                CudaError::Driver(format!("launch compute_residual_coeffs: {e}"))
                            })?;
                        }
                    }
                    PlanEntry::NuInit => {
                        phi.apply_diagonal_pauli_sum_via_pool(
                            &mut nu,
                            &nu_sign_masks_pool,
                            &nu_coeffs_pool,
                            plan.gradient_obs_sign_masks.len() as u32,
                        )?;
                    }
                    PlanEntry::DaggerPhiDiag { slot } => {
                        phi.apply_diagonal_via_pool(&dagger_diag_pool, slot)?;
                    }
                    PlanEntry::DaggerPhi1q { slot } => {
                        phi.apply_1q_via_pool(&dagger_1q_pool, slot)?;
                    }
                    PlanEntry::DaggerPhi2q { slot } => {
                        phi.apply_2q_via_pool(&dagger_2q_pool, slot)?;
                    }
                    PlanEntry::CopyPhiToTemp => {
                        phi.copy_into(&mut temp)?;
                    }
                    PlanEntry::Deriv1q { slot } => {
                        temp.apply_1q_via_pool(&deriv_1q_pool, slot)?;
                    }
                    PlanEntry::Deriv2q { slot } => {
                        temp.apply_2q_via_pool(&deriv_2q_pool, slot)?;
                    }
                    PlanEntry::CopyPhiAndDeriv1q { slot } => {
                        phi.apply_1q_from_to_via_pool(&mut temp, &deriv_1q_pool, slot)?;
                    }
                    PlanEntry::Deriv1qInnerProductAccumulate {
                        deriv_slot,
                        accum_slot,
                    } => {
                        phi.apply_1q_inner_product_accumulate_via_pool(
                            &nu,
                            &deriv_1q_pool,
                            deriv_slot,
                            &mut grad_dev,
                            &chain_pool,
                            &sym_idx_pool,
                            accum_slot,
                        )?;
                    }
                    PlanEntry::Accumulate { slot } => {
                        nu.inner_product_accumulate_via_pool(
                            &temp,
                            &mut grad_dev,
                            &chain_pool,
                            &sym_idx_pool,
                            slot,
                        )?;
                    }
                    PlanEntry::DaggerNuDiag { slot } => {
                        nu.apply_diagonal_via_pool(&dagger_diag_pool, slot)?;
                    }
                    PlanEntry::DaggerNu1q { slot } => {
                        nu.apply_1q_via_pool(&dagger_1q_pool, slot)?;
                    }
                    PlanEntry::DaggerNu2q { slot } => {
                        nu.apply_2q_via_pool(&dagger_2q_pool, slot)?;
                    }
                    PlanEntry::DaggerBoth2q { slot } => {
                        phi.apply_2q_pooled_dual(&mut nu, &dagger_2q_pool, slot)?;
                    }
                    PlanEntry::DaggerBoth1q { slot } => {
                        phi.apply_1q_pooled_dual(&mut nu, &dagger_1q_pool, slot)?;
                    }
                    PlanEntry::PauliZChainAccumulate { count } => {
                        phi.pauli_z_chain_accumulate(
                            &nu,
                            &mut grad_dev,
                            &pauli_z_chain_pool,
                            0,
                            count,
                        )?;
                    }
                    PlanEntry::DaggerBothDiagChain { count } => {
                        phi.apply_diagonal_chain_dual_via_pool(
                            &mut nu,
                            &pauli_z_chain_dagger_pool,
                            0,
                            count,
                        )?;
                    }
                    PlanEntry::Pauli1qYDaggerBothFused {
                        dagger_slot,
                        accum_slot,
                    } => {
                        phi.pauli_y_accumulate_then_dagger_both_via_pool(
                            &mut nu,
                            &dagger_1q_pool,
                            dagger_slot,
                            &mut grad_dev,
                            &chain_pool,
                            &sym_idx_pool,
                            accum_slot,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        let graph_opt = graph_stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| OmegaError::Backend(format!("end_capture: {e}")))?;
        result.map_err(OmegaError::from)?;
        let graph =
            graph_opt.ok_or_else(|| OmegaError::Backend("end_capture returned no graph".into()))?;

        if std::env::var_os("OMEGA_TRAINSTEP_PROFILE").is_some() {
            eprintln!(
                "TrainStepGraph captured: {} plan entries (forward_diag={}, forward_1q={}, forward_2q={}, dagger_diag={}, dagger_1q={}, dagger_2q={}, deriv_1q={}, deriv_2q={}, accumulate={}, predict_z={}, nu_init=1, residual_coeffs=1)",
                plan.entries.len(),
                plan.n_forward_diag,
                plan.n_forward_1q,
                plan.n_forward_2q,
                plan.n_dagger_diag,
                plan.n_dagger_1q,
                plan.n_dagger_2q,
                plan.n_deriv_1q,
                plan.n_deriv_2q,
                plan.n_accumulate,
                plan.gradient_obs_sign_masks.len(),
            );
        }

        Ok(Self {
            plan,
            handle: graph_handle,
            phi,
            nu,
            temp,
            forward_diag_pool,
            forward_1q_pool,
            forward_2q_pool,
            nu_sign_masks_pool,
            nu_coeffs_pool,
            dagger_diag_pool,
            dagger_1q_pool,
            dagger_2q_pool,
            deriv_1q_pool,
            deriv_2q_pool,
            chain_pool,
            sym_idx_pool,
            pauli_z_chain_pool,
            pauli_z_chain_dagger_pool,
            grad_dev,
            predictions_dev,
            y_label_dev,
            y_label_host,
            forward_diag_host,
            forward_1q_host,
            forward_2q_host,
            nu_coeffs_host,
            dagger_diag_host,
            dagger_1q_host,
            dagger_2q_host,
            deriv_1q_host,
            deriv_2q_host,
            chain_host,
            grad_host,
            pauli_z_chain_host,
            pauli_z_chain_dagger_host,
            graph,
        })
    }

    /// Replay the captured training step with fresh `params` and
    /// the per-output `y_labels`. The graph computes predictions
    /// in-flight, the host-func callback derives the residual
    /// gradient observable's coefficients (`2·(y_hat - y_label)`),
    /// and the backward sweep accumulates gradients.
    ///
    /// Returns `(predictions, gradients_sorted_by_symbol)`. The
    /// predictions are read from the graph's `predictions_host`
    /// buffer; the gradients are `clone_dtoh`-ed from `grad_dev`.
    #[allow(clippy::type_complexity)]
    pub fn replay_with_y_labels(
        &mut self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
        y_labels: &[f32],
    ) -> OmegaResult<(Vec<f64>, Vec<(SymbolId, f64)>)> {
        if y_labels.len() != self.plan.gradient_obs_sign_masks.len() {
            return Err(OmegaError::InvalidCircuit(format!(
                "TrainStepGraph: y_label count {} != sign_masks count {}",
                y_labels.len(),
                self.plan.gradient_obs_sign_masks.len()
            )));
        }
        let profile = replay_profile_enabled();
        let t_resolve_start = if profile { Some(Instant::now()) } else { None };
        // Stage forward / dagger / derivative / chain pools — same
        // as replay_with_coeffs but skipping the explicit coeffs
        // upload (the host_func computes coeffs from
        // predictions + y_label inside the graph).
        for (slot, &op_idx) in self.plan.forward_diag_op.iter().enumerate() {
            self.forward_diag_host[slot] = resolve_diagonal_params(&circuit.ops[op_idx], params)?;
        }
        for (slot, &op_idx) in self.plan.forward_1q_op.iter().enumerate() {
            self.forward_1q_host[slot] = resolve_apply_1q_params(&circuit.ops[op_idx], params)?;
        }
        for (slot, &op_idx) in self.plan.forward_2q_op.iter().enumerate() {
            self.forward_2q_host[slot] = resolve_apply_2q_params(&circuit.ops[op_idx], params)?;
        }
        for (slot, &op_idx) in self.plan.dagger_diag_op.iter().enumerate() {
            self.dagger_diag_host[slot] =
                resolve_dagger_diagonal_params(&circuit.ops[op_idx], params)?;
        }
        for (slot, &op_idx) in self.plan.dagger_1q_op.iter().enumerate() {
            self.dagger_1q_host[slot] =
                resolve_dagger_apply_1q_params(&circuit.ops[op_idx], params)?;
        }
        for (slot, &op_idx) in self.plan.dagger_2q_op.iter().enumerate() {
            self.dagger_2q_host[slot] =
                resolve_dagger_apply_2q_params(&circuit.ops[op_idx], params)?;
        }
        for (slot, recipe) in self.plan.deriv_1q_recipe.iter().enumerate() {
            self.deriv_1q_host[slot] = resolve_deriv_apply_1q_params(
                &circuit.ops[recipe.op_idx],
                params,
                recipe.param_idx,
            )?;
        }
        for (slot, recipe) in self.plan.deriv_2q_recipe.iter().enumerate() {
            self.deriv_2q_host[slot] = resolve_deriv_apply_2q_params(
                &circuit.ops[recipe.op_idx],
                params,
                recipe.param_idx,
            )?;
        }
        for (slot, recipe) in self.plan.accumulate_chain_recipe.iter().enumerate() {
            let chain = params.resolve_derivative(
                &circuit.ops[recipe.op_idx].params[recipe.param_idx],
                recipe.sym_id,
            )?;
            self.chain_host[slot] = chain;
        }
        // Stage G6: per-replay update for the Pauli-Z chain pools.
        // For each chain RZ, populate (qubit, sym_slot, chain) into
        // `pauli_z_chain_host`, and resolve the dagger matrix into
        // `pauli_z_chain_dagger_host`. Both pools sit at fixed
        // offsets — the kernel reads start=0, count=chain_len.
        for (slot, recipe) in self.plan.pauli_z_chain_recipe.iter().enumerate() {
            let chain = params.resolve_derivative(
                &circuit.ops[recipe.op_idx].params[recipe.param_idx],
                recipe.sym_id,
            )?;
            let sym_slot = *self
                .plan
                .sym_to_slot
                .get(&recipe.sym_id)
                .expect("chain sym must have grad_dev slot");
            self.pauli_z_chain_host[slot] = crate::imp::PauliZChainEntry {
                qubit: recipe.qubit,
                sym: sym_slot,
                chain,
            };
            self.pauli_z_chain_dagger_host[slot] =
                resolve_dagger_diagonal_params(&circuit.ops[recipe.op_idx], params)?;
        }
        // Stage F2: stage y_labels host-side, then memcpy to
        // device. The captured `compute_residual_coeffs` kernel
        // reads y_label_dev to derive the residual coefficients.
        for (slot, &y) in y_labels.iter().enumerate() {
            self.y_label_host[slot] = y;
        }

        // Upload pools.
        let stream = self.handle.stream.clone();
        let t_upload_start = if profile {
            // Sync first so the resolve timing isn't smeared with
            // any prior stream work.
            stream.synchronize().ok();
            let now = Instant::now();
            if let Some(t0) = t_resolve_start {
                REPLAY_NS_RESOLVE
                    .fetch_add(now.duration_since(t0).as_nanos() as u64, Ordering::Relaxed);
            }
            Some(now)
        } else {
            None
        };
        stream
            .memcpy_htod(&self.y_label_host[..], &mut self.y_label_dev)
            .map_err(|e| OmegaError::Backend(format!("upload y_label_dev: {e}")))?;
        if !self.forward_diag_host.is_empty() {
            stream
                .memcpy_htod(&self.forward_diag_host[..], &mut self.forward_diag_pool)
                .map_err(|e| OmegaError::Backend(format!("upload forward_diag: {e}")))?;
        }
        if !self.forward_1q_host.is_empty() {
            stream
                .memcpy_htod(&self.forward_1q_host[..], &mut self.forward_1q_pool)
                .map_err(|e| OmegaError::Backend(format!("upload forward_1q: {e}")))?;
        }
        if !self.forward_2q_host.is_empty() {
            stream
                .memcpy_htod(&self.forward_2q_host[..], &mut self.forward_2q_pool)
                .map_err(|e| OmegaError::Backend(format!("upload forward_2q: {e}")))?;
        }
        if !self.dagger_diag_host.is_empty() {
            stream
                .memcpy_htod(&self.dagger_diag_host[..], &mut self.dagger_diag_pool)
                .map_err(|e| OmegaError::Backend(format!("upload dagger_diag: {e}")))?;
        }
        if !self.dagger_1q_host.is_empty() {
            stream
                .memcpy_htod(&self.dagger_1q_host[..], &mut self.dagger_1q_pool)
                .map_err(|e| OmegaError::Backend(format!("upload dagger_1q: {e}")))?;
        }
        if !self.dagger_2q_host.is_empty() {
            stream
                .memcpy_htod(&self.dagger_2q_host[..], &mut self.dagger_2q_pool)
                .map_err(|e| OmegaError::Backend(format!("upload dagger_2q: {e}")))?;
        }
        if !self.deriv_1q_host.is_empty() {
            stream
                .memcpy_htod(&self.deriv_1q_host[..], &mut self.deriv_1q_pool)
                .map_err(|e| OmegaError::Backend(format!("upload deriv_1q: {e}")))?;
        }
        if !self.deriv_2q_host.is_empty() {
            stream
                .memcpy_htod(&self.deriv_2q_host[..], &mut self.deriv_2q_pool)
                .map_err(|e| OmegaError::Backend(format!("upload deriv_2q: {e}")))?;
        }
        if !self.chain_host.is_empty() {
            stream
                .memcpy_htod(&self.chain_host[..], &mut self.chain_pool)
                .map_err(|e| OmegaError::Backend(format!("upload chain: {e}")))?;
        }
        // Stage G6: upload the chain pools when the trailing RZ
        // chain is non-empty (skipped on shapes with no chain).
        if !self.plan.pauli_z_chain_recipe.is_empty() {
            stream
                .memcpy_htod(&self.pauli_z_chain_host[..], &mut self.pauli_z_chain_pool)
                .map_err(|e| OmegaError::Backend(format!("upload pauli_z_chain: {e}")))?;
            stream
                .memcpy_htod(
                    &self.pauli_z_chain_dagger_host[..],
                    &mut self.pauli_z_chain_dagger_pool,
                )
                .map_err(|e| OmegaError::Backend(format!("upload pauli_z_chain_dagger: {e}")))?;
        }

        let t_launch_start = if profile {
            stream.synchronize().ok();
            let now = Instant::now();
            if let Some(t0) = t_upload_start {
                REPLAY_NS_UPLOAD
                    .fetch_add(now.duration_since(t0).as_nanos() as u64, Ordering::Relaxed);
            }
            Some(now)
        } else {
            None
        };

        // Replay.
        self.graph
            .launch()
            .map_err(|e| OmegaError::Backend(format!("graph launch: {e}")))?;

        if profile {
            stream.synchronize().ok();
            if let Some(t0) = t_launch_start {
                REPLAY_NS_LAUNCH.fetch_add(
                    Instant::now().duration_since(t0).as_nanos() as u64,
                    Ordering::Relaxed,
                );
            }
        }
        let t_dtoh_start = if profile { Some(Instant::now()) } else { None };

        // Pull gradients back to host.
        let host: Vec<f64> = stream
            .clone_dtoh(&self.grad_dev)
            .map_err(|e| OmegaError::Backend(format!("memcpy grad_dev: {e}")))?;
        stream
            .synchronize()
            .map_err(|e| OmegaError::Backend(format!("sync: {e}")))?;
        self.grad_host = host;
        if profile {
            if let Some(t0) = t_dtoh_start {
                REPLAY_NS_GRAD_DTOH.fetch_add(
                    Instant::now().duration_since(t0).as_nanos() as u64,
                    Ordering::Relaxed,
                );
            }
            REPLAY_COUNT.fetch_add(1, Ordering::Relaxed);
        }

        // Stage F2: predictions live on the device after replay.
        // One memcpy_dtoh of the small predictions_dev buffer to
        // host gives us the trainer's `predictions` return value.
        let predictions: Vec<f64> = stream
            .clone_dtoh(&self.predictions_dev)
            .map_err(|e| OmegaError::Backend(format!("memcpy predictions: {e}")))?;

        // Map gradients to circuit symbol order.
        let mut grad_out: Vec<(SymbolId, f64)> = circuit
            .symbols
            .keys()
            .map(|&sym| {
                let slot = *self
                    .plan
                    .sym_to_slot
                    .get(&sym)
                    .expect("circuit symbol must have a slot");
                (sym, self.grad_host[slot as usize])
            })
            .collect();
        grad_out.sort_by_key(|(s, _)| *s);
        Ok((predictions, grad_out))
    }
}

impl Drop for TrainStepGraph {
    fn drop(&mut self) {
        // Dump the replay profile when the cached graph drops
        // (typically when the backend is dropped at end of bench).
        // No-op if profiling is disabled.
        dump_replay_profile();
    }
}

/// Stage G6: detect the trailing run of param'd RZ ops at the end
/// of the forward sweep that qualify for the chain restructuring.
/// Returns the list of (op_idx, param_idx, sym_id, qubit) for the
/// chain, in forward order. Empty when no such chain exists.
///
/// Conditions for an op to join the chain:
///   - is `RZ`
///   - has exactly one symbol-bearing param expression
///   - acts on a qubit not yet seen in the chain (disjoint qubits)
///
/// We walk forward ops from the END backwards, stopping at the
/// first non-qualifying op. The result is reversed so the chain is
/// in forward order. Phase 4c HEA's Layer 2 (8 RZs on q=4..11) is
/// the canonical case.
fn detect_trailing_pauli_z_chain(unitary_ops: &[(usize, &GateOp)]) -> Vec<PauliZChainRecipe> {
    let mut acc: Vec<PauliZChainRecipe> = Vec::new();
    let mut seen_qubits: std::collections::HashSet<u32> = Default::default();
    for &(op_idx, op) in unitary_ops.iter().rev() {
        if matches!(&op.gate, GateKind::Id | GateKind::Barrier) {
            continue;
        }
        if !matches!(&op.gate, GateKind::Rz) {
            break;
        }
        // Must have exactly one symbol-bearing param.
        let mut chain_param: Option<(usize, SymbolId)> = None;
        let mut has_value_only = false;
        for (param_idx, p) in op.params.iter().enumerate() {
            let syms = collect_symbols(p);
            if syms.is_empty() {
                has_value_only = true;
                continue;
            }
            if syms.len() != 1 || chain_param.is_some() {
                chain_param = None;
                break;
            }
            chain_param = Some((param_idx, syms[0]));
        }
        let (param_idx, sym_id) = match chain_param {
            Some(p) if !has_value_only => p,
            _ => break,
        };
        let qubit = op.qubits[0].0;
        if !seen_qubits.insert(qubit) {
            // duplicate qubit — chain breaks (commutation no longer
            // holds for repeated qubit RZs since we'd be fusing
            // angles on the same qubit anyway, which is fine
            // mathematically but the commutation argument with
            // _later_ ops in the chain assumes disjoint qubits).
            break;
        }
        acc.push(PauliZChainRecipe {
            op_idx,
            param_idx,
            sym_id,
            qubit,
        });
    }
    acc.reverse();
    acc
}

/// Fuse consecutive `ForwardDiag` plan entries with sequential
/// slots into a single `ForwardDiagChain`. Stage G5 post-processing
/// — the original walker pushes one entry per diagonal op; this
/// pass merges back-to-back runs (e.g. the HEA's Layer 2 RZ chain)
/// into one launch.
fn fuse_forward_diag_runs(entries: Vec<PlanEntry>) -> Vec<PlanEntry> {
    let mut out: Vec<PlanEntry> = Vec::with_capacity(entries.len());
    let mut i = 0;
    while i < entries.len() {
        if let PlanEntry::ForwardDiag { slot: s0 } = entries[i] {
            let start = s0;
            let mut j = i + 1;
            let mut expected = s0 + 1;
            while j < entries.len() {
                if let PlanEntry::ForwardDiag { slot } = entries[j] {
                    if slot == expected {
                        expected += 1;
                        j += 1;
                        continue;
                    }
                }
                break;
            }
            let count = j - i;
            if count >= 2 {
                out.push(PlanEntry::ForwardDiagChain {
                    start,
                    count: count as u32,
                });
            } else {
                out.push(entries[i]);
            }
            i = j;
        } else {
            out.push(entries[i]);
            i += 1;
        }
    }
    out
}

/// Z-only Pauli-sum extractor for the gradient observable. Same
/// shape as `adjoint::diagonal_pauli_terms` but returns only the
/// sign masks (coeffs are per-replay).
fn extract_gradient_obs_sign_masks(obs: &Observable) -> OmegaResult<Vec<u32>> {
    let mut out = Vec::with_capacity(obs.terms.len());
    for (_coeff, pauli_string) in &obs.terms {
        let mut sign_mask: u32 = 0;
        for &(q, ref p) in pauli_string {
            match p {
                PauliOp::I => {}
                PauliOp::Z => sign_mask |= 1u32 << q,
                PauliOp::X | PauliOp::Y => {
                    return Err(OmegaError::Unsupported(
                        "TrainStepGraph: gradient observable must be Z-only (X/Y not supported \
                         in graph capture; trainer falls back to non-graph path)"
                            .into(),
                    ));
                }
            }
        }
        out.push(sign_mask);
    }
    Ok(out)
}

fn build_plan(circuit: &CircuitIR, gradient_obs_template: &Observable) -> OmegaResult<GraphPlan> {
    let gradient_obs_sign_masks = extract_gradient_obs_sign_masks(gradient_obs_template)?;

    // Validate every op + classify forward/backward kernel kinds.
    for op in &circuit.ops {
        if op.condition.is_some() {
            return Err(OmegaError::Unsupported(
                "TrainStepGraph: conditional gates not supported (yet)".into(),
            ));
        }
        match &op.gate {
            GateKind::Reset
            | GateKind::Measure
            | GateKind::CCX
            | GateKind::CSwap
            | GateKind::PhaseShifter
            | GateKind::BeamSplitterRx
            | GateKind::Rbs
            | GateKind::Custom(_) => {
                return Err(OmegaError::Unsupported(format!(
                    "TrainStepGraph: gate {:?} not supported in this slice",
                    op.gate
                )));
            }
            GateKind::Id | GateKind::Barrier => {
                // These are no-ops at the kernel level — we'll skip
                // them in the capture walker.
            }
            _ => {}
        }
    }

    // Stable symbol → slot map. Sort by SymbolId so the slot index
    // is reproducible across captures.
    let mut sym_ids: Vec<SymbolId> = circuit.symbols.keys().copied().collect();
    sym_ids.sort_unstable();
    let sym_to_slot: HashMap<SymbolId, u32> = sym_ids
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as u32))
        .collect();
    let num_symbols = sym_ids.len();

    let unitary_ops: Vec<(usize, &GateOp)> = circuit
        .ops
        .iter()
        .enumerate()
        .filter(|(_, op)| {
            !matches!(
                &op.gate,
                GateKind::Measure | GateKind::Barrier | GateKind::Reset
            )
        })
        .collect();

    // Stage G6: detect a trailing pure-RZ chain at the end of the
    // forward sweep that qualifies for the commutation-based
    // restructuring. The chain's ops still appear in the forward
    // sweep (so the forward state is computed correctly), but they
    // are EXCLUDED from per-op backward processing — we emit a
    // single `PauliZChainAccumulate` + `DaggerBothDiagChain` pair
    // at the START of the backward sweep instead. Empty when no
    // such chain exists.
    let pauli_z_chain_recipe = detect_trailing_pauli_z_chain(&unitary_ops);
    let chain_op_set: std::collections::HashSet<usize> =
        pauli_z_chain_recipe.iter().map(|r| r.op_idx).collect();

    let mut entries = Vec::new();
    let mut forward_diag_op = Vec::new();
    let mut forward_1q_op = Vec::new();
    let mut forward_2q_op = Vec::new();
    let mut dagger_diag_op = Vec::new();
    let mut dagger_1q_op = Vec::new();
    let mut dagger_2q_op = Vec::new();
    let mut deriv_1q_recipe: Vec<DerivRecipe> = Vec::new();
    let mut deriv_2q_recipe: Vec<DerivRecipe> = Vec::new();
    let mut accumulate_chain_recipe: Vec<ChainRecipe> = Vec::new();

    // Forward sweep entries (in op order).
    for &(op_idx, op) in &unitary_ops {
        if matches!(&op.gate, GateKind::Id) {
            continue;
        }
        let kind = classify_op_kernel(op)?;
        match kind {
            KernelKind::Diagonal => {
                let slot = forward_diag_op.len() as u32;
                forward_diag_op.push(op_idx);
                entries.push(PlanEntry::ForwardDiag { slot });
            }
            KernelKind::Apply1q => {
                let slot = forward_1q_op.len() as u32;
                forward_1q_op.push(op_idx);
                entries.push(PlanEntry::Forward1q { slot });
            }
            KernelKind::Apply2q => {
                let slot = forward_2q_op.len() as u32;
                forward_2q_op.push(op_idx);
                entries.push(PlanEntry::Forward2q { slot });
            }
        }
    }

    // Stage E: predictions inside the graph.
    //
    // For each gradient-observable term (each is a single-Z on one
    // qubit, by the Z-only check in extract_gradient_obs_sign_masks),
    // we run pauli_z_expectation_to_slot, atomic-adding ⟨Z_q⟩ into
    // predictions_dev[i]. The sign_mask is the same one ν init uses.
    // Slot index `i` matches the term order so the host_func and
    // memcpy_htod that follow line up correctly.
    for (i, &sign_mask) in gradient_obs_sign_masks.iter().enumerate() {
        entries.push(PlanEntry::PredictZ {
            sign_mask,
            slot: i as u32,
        });
    }
    // Stage F2: pure-device residual-coefficient builder kernel.
    // Reads predictions_dev + y_label_dev (populated host→device
    // per replay), writes nu_coeffs_pool. Replaces Stage E's
    // memcpy_dtoh + host_func + memcpy_htod chain with one kernel
    // node — no CUDA worker thread context switch.
    entries.push(PlanEntry::ComputeResidualCoeffs);

    // ν init (reads nu_coeffs_pool the previous memcpy just wrote).
    entries.push(PlanEntry::NuInit);

    // Stage G6: emit the chain-fused accumulate + dual-dagger pair
    // at the START of the backward sweep, BEFORE any per-op work
    // for non-chain ops. Both kernels read pre-dagger (φ_n, ν_n)
    // and after the dagger leaves both states transformed to the
    // post-RZ-chain state, ready for the per-op backward of the
    // remaining (non-chain) ops.
    if !pauli_z_chain_recipe.is_empty() {
        entries.push(PlanEntry::PauliZChainAccumulate {
            count: pauli_z_chain_recipe.len() as u32,
        });
        entries.push(PlanEntry::DaggerBothDiagChain {
            count: pauli_z_chain_recipe.len() as u32,
        });
    }

    // Backward sweep (reverse op order). Per op:
    //   1. dagger φ via the dagger pool
    //   2. for each (param_idx, sym_id): copy_into + derivative + accumulate
    //   3. dagger ν via the dagger pool
    //
    // Chain ops are skipped — they were handled by the
    // `PauliZChainAccumulate` + `DaggerBothDiagChain` pair above.
    for &(op_idx, op) in unitary_ops.iter().rev() {
        if matches!(&op.gate, GateKind::Id) {
            continue;
        }
        if chain_op_set.contains(&op_idx) {
            continue;
        }
        let kind = classify_op_kernel(op)?;
        // Stage G3: detect ops with no symbol-bearing params so we
        // can fuse DaggerPhi + DaggerNu into one dual-state launch
        // for the 2q case (CNOT/CZ on Phase 4c). The phi+nu
        // daggers are emitted at the start AND end of the per-op
        // block; for non-param ops the inner deriv/accumulate loop
        // is empty so they're adjacent and fuse-able.
        let op_has_params = op.params.iter().any(|p| !collect_symbols(p).is_empty());
        if !op_has_params {
            match kind {
                KernelKind::Apply2q => {
                    let slot = dagger_2q_op.len() as u32;
                    dagger_2q_op.push(op_idx);
                    entries.push(PlanEntry::DaggerBoth2q { slot });
                    continue;
                }
                KernelKind::Apply1q => {
                    let slot = dagger_1q_op.len() as u32;
                    dagger_1q_op.push(op_idx);
                    entries.push(PlanEntry::DaggerBoth1q { slot });
                    continue;
                }
                KernelKind::Diagonal => {}
            }
        }
        // Stage G7: per-op Pauli-Y triple fusion for parameterised
        // RY ops (single-symbol param, single-qubit). Replaces the
        // (DaggerPhi1q + Deriv1qInnerProductAccumulate + DaggerNu1q)
        // triple with one fused kernel — uses ∂RY·RY† = -i/2·Y_q.
        if matches!(&op.gate, GateKind::Ry) && op.params.len() == 1 {
            let syms = collect_symbols(&op.params[0]);
            if syms.len() == 1 {
                let sym_id = syms[0];
                let dagger_slot = dagger_1q_op.len() as u32;
                dagger_1q_op.push(op_idx);
                let accum_slot = accumulate_chain_recipe.len() as u32;
                accumulate_chain_recipe.push(ChainRecipe {
                    op_idx,
                    param_idx: 0,
                    sym_id,
                });
                entries.push(PlanEntry::Pauli1qYDaggerBothFused {
                    dagger_slot,
                    accum_slot,
                });
                continue;
            }
        }
        // Dagger φ.
        match kind {
            KernelKind::Diagonal => {
                let slot = dagger_diag_op.len() as u32;
                dagger_diag_op.push(op_idx);
                entries.push(PlanEntry::DaggerPhiDiag { slot });
            }
            KernelKind::Apply1q => {
                let slot = dagger_1q_op.len() as u32;
                dagger_1q_op.push(op_idx);
                entries.push(PlanEntry::DaggerPhi1q { slot });
            }
            KernelKind::Apply2q => {
                let slot = dagger_2q_op.len() as u32;
                dagger_2q_op.push(op_idx);
                entries.push(PlanEntry::DaggerPhi2q { slot });
            }
        }

        // Per-param derivative + accumulate.
        for (param_idx, param_expr) in op.params.iter().enumerate() {
            let syms = collect_symbols(param_expr);
            for sym_id in syms {
                // derivative apply — kind matches the op. The kernel
                // kind for the *derivative* matrix is always apply_1q
                // for 1q-parameterised gates and apply_2q for
                // 2q-parameterised gates (CRz / CU3). Diagonal-
                // derivative-as-apply_1q is inefficient but
                // correctness-preserving — same path the existing
                // adjoint module uses.
                let deriv_kind = match &op.gate {
                    GateKind::Rx
                    | GateKind::Ry
                    | GateKind::Rz
                    | GateKind::U1
                    | GateKind::U2
                    | GateKind::U3 => KernelKind::Apply1q,
                    GateKind::CRz | GateKind::CU3 => KernelKind::Apply2q,
                    other => {
                        return Err(OmegaError::Unsupported(format!(
                            "TrainStepGraph: derivative path for {other:?} not supported"
                        )));
                    }
                };
                match deriv_kind {
                    KernelKind::Apply1q => {
                        // Stage G2 fusion: emit the triple-fused
                        // copy+deriv+accumulate kernel. Skips the
                        // temp materialisation entirely on the 1q
                        // path. Saves another graph node per
                        // 1q-parameterised gate vs Stage G1.
                        let deriv_slot = deriv_1q_recipe.len() as u32;
                        deriv_1q_recipe.push(DerivRecipe { op_idx, param_idx });
                        let accum_slot = accumulate_chain_recipe.len() as u32;
                        accumulate_chain_recipe.push(ChainRecipe {
                            op_idx,
                            param_idx,
                            sym_id,
                        });
                        entries.push(PlanEntry::Deriv1qInnerProductAccumulate {
                            deriv_slot,
                            accum_slot,
                        });
                    }
                    KernelKind::Apply2q => {
                        // 2q derivative path still uses the split
                        // CopyPhiToTemp + Deriv2q + Accumulate
                        // chain because we don't have a fused 2q
                        // triple kernel yet (and 2q parameterised
                        // gates are rarer in the QML trainer's HEA
                        // shape).
                        entries.push(PlanEntry::CopyPhiToTemp);
                        let slot = deriv_2q_recipe.len() as u32;
                        deriv_2q_recipe.push(DerivRecipe { op_idx, param_idx });
                        entries.push(PlanEntry::Deriv2q { slot });
                        let accum_slot = accumulate_chain_recipe.len() as u32;
                        accumulate_chain_recipe.push(ChainRecipe {
                            op_idx,
                            param_idx,
                            sym_id,
                        });
                        entries.push(PlanEntry::Accumulate { slot: accum_slot });
                    }
                    KernelKind::Diagonal => unreachable!(),
                }
            }
        }

        // Dagger ν.
        match kind {
            KernelKind::Diagonal => {
                let slot = dagger_diag_op.len() as u32;
                dagger_diag_op.push(op_idx);
                entries.push(PlanEntry::DaggerNuDiag { slot });
            }
            KernelKind::Apply1q => {
                let slot = dagger_1q_op.len() as u32;
                dagger_1q_op.push(op_idx);
                entries.push(PlanEntry::DaggerNu1q { slot });
            }
            KernelKind::Apply2q => {
                let slot = dagger_2q_op.len() as u32;
                dagger_2q_op.push(op_idx);
                entries.push(PlanEntry::DaggerNu2q { slot });
            }
        }
    }

    // Stage G5: post-process the entries list. Collapse runs of
    // ≥ 2 consecutive `ForwardDiag` entries with sequential slots
    // into one `ForwardDiagChain { start, count }`. The pool layout
    // doesn't change; only the launch shape does.
    let entries = fuse_forward_diag_runs(entries);

    Ok(GraphPlan {
        n_forward_diag: forward_diag_op.len(),
        n_forward_1q: forward_1q_op.len(),
        n_forward_2q: forward_2q_op.len(),
        n_dagger_diag: dagger_diag_op.len(),
        n_dagger_1q: dagger_1q_op.len(),
        n_dagger_2q: dagger_2q_op.len(),
        n_deriv_1q: deriv_1q_recipe.len(),
        n_deriv_2q: deriv_2q_recipe.len(),
        n_accumulate: accumulate_chain_recipe.len(),
        num_symbols,
        sym_to_slot,
        gradient_obs_sign_masks,
        accumulate_chain_recipe,
        deriv_1q_recipe,
        deriv_2q_recipe,
        forward_diag_op,
        forward_1q_op,
        forward_2q_op,
        dagger_diag_op,
        dagger_1q_op,
        dagger_2q_op,
        pauli_z_chain_recipe,
        entries,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KernelKind {
    Diagonal,
    Apply1q,
    Apply2q,
}

fn classify_op_kernel(op: &GateOp) -> OmegaResult<KernelKind> {
    Ok(match &op.gate {
        GateKind::Z
        | GateKind::S
        | GateKind::Sdg
        | GateKind::T
        | GateKind::Tdg
        | GateKind::Rz
        | GateKind::U1 => KernelKind::Diagonal,
        GateKind::H
        | GateKind::X
        | GateKind::Y
        | GateKind::Sx
        | GateKind::Sxdg
        | GateKind::Rx
        | GateKind::Ry
        | GateKind::U2
        | GateKind::U3 => KernelKind::Apply1q,
        GateKind::CX
        | GateKind::CY
        | GateKind::CZ
        | GateKind::Swap
        | GateKind::CRz
        | GateKind::CU3 => KernelKind::Apply2q,
        other => {
            return Err(OmegaError::Unsupported(format!(
                "TrainStepGraph: gate {other:?} not classifiable into a kernel kind"
            )));
        }
    })
}

fn collect_symbols(expr: &ParamExpr) -> Vec<SymbolId> {
    fn walk(e: &ParamExpr, out: &mut Vec<SymbolId>) {
        match e {
            ParamExpr::Concrete(_) => {}
            ParamExpr::Symbol(s) => out.push(*s),
            ParamExpr::Add(a, b) | ParamExpr::Mul(a, b) => {
                walk(a, out);
                walk(b, out);
            }
            ParamExpr::Negate(a) => walk(a, out),
        }
    }
    let mut out = Vec::new();
    walk(expr, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

// ---- Per-replay matrix resolvers --------------------------------

fn resolve_diagonal_params(op: &GateOp, params: &ParameterBinding) -> OmegaResult<DiagonalParams> {
    let q = op.qubits[0].0;
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let (d0, d1) = match &op.gate {
        GateKind::Z => (Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)),
        GateKind::S => (Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)),
        GateKind::Sdg => (Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0)),
        GateKind::T => (
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
        ),
        GateKind::Tdg => (
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4),
        ),
        GateKind::Rz => (
            Complex64::from_polar(1.0, -resolved[0] / 2.0),
            Complex64::from_polar(1.0, resolved[0] / 2.0),
        ),
        GateKind::U1 => (
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, resolved[0]),
        ),
        other => {
            return Err(OmegaError::Unsupported(format!(
                "resolve_diagonal_params: gate {other:?} not handled"
            )));
        }
    };
    Ok(DiagonalParams::from_complex(q, d0, d1))
}

fn resolve_apply_1q_params(op: &GateOp, params: &ParameterBinding) -> OmegaResult<Apply1qParams> {
    let q = op.qubits[0].0;
    let mat = forward_1q_matrix(op, params)?;
    Ok(Apply1qParams::from_matrix(q, &mat))
}

fn resolve_apply_2q_params(op: &GateOp, params: &ParameterBinding) -> OmegaResult<Apply2qParams> {
    let qa = op.qubits[0].0;
    let qb = op.qubits[1].0;
    let mat = forward_2q_matrix(op, params)?;
    Ok(Apply2qParams::from_matrix(qa, qb, &mat))
}

fn forward_1q_matrix(op: &GateOp, params: &ParameterBinding) -> OmegaResult<[Complex64; 4]> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let s = std::f64::consts::FRAC_1_SQRT_2;
    Ok(match &op.gate {
        GateKind::H => [
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(-s, 0.0),
        ],
        GateKind::X => [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        GateKind::Y => [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
        // √X / √X† — exact Qiskit matrices (½[[1±i,1∓i],[1∓i,1±i]]), the dagger
        // is derived generically by conj-transpose downstream.
        GateKind::Sx => [
            Complex64::new(0.5, 0.5),
            Complex64::new(0.5, -0.5),
            Complex64::new(0.5, -0.5),
            Complex64::new(0.5, 0.5),
        ],
        GateKind::Sxdg => [
            Complex64::new(0.5, -0.5),
            Complex64::new(0.5, 0.5),
            Complex64::new(0.5, 0.5),
            Complex64::new(0.5, -0.5),
        ],
        GateKind::Rx => {
            let c = (resolved[0] / 2.0).cos();
            let sn = (resolved[0] / 2.0).sin();
            [
                Complex64::new(c, 0.0),
                Complex64::new(0.0, -sn),
                Complex64::new(0.0, -sn),
                Complex64::new(c, 0.0),
            ]
        }
        GateKind::Ry => {
            let c = (resolved[0] / 2.0).cos();
            let sn = (resolved[0] / 2.0).sin();
            [
                Complex64::new(c, 0.0),
                Complex64::new(-sn, 0.0),
                Complex64::new(sn, 0.0),
                Complex64::new(c, 0.0),
            ]
        }
        GateKind::U2 => u3_mat(std::f64::consts::FRAC_PI_2, resolved[0], resolved[1]),
        GateKind::U3 => u3_mat(resolved[0], resolved[1], resolved[2]),
        other => {
            return Err(OmegaError::Unsupported(format!(
                "forward_1q_matrix: gate {other:?} not handled"
            )));
        }
    })
}

fn u3_mat(theta: f64, phi: f64, lambda: f64) -> [Complex64; 4] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        Complex64::new(c, 0.0),
        -Complex64::from_polar(s, lambda),
        Complex64::from_polar(s, phi),
        Complex64::from_polar(c, phi + lambda),
    ]
}

fn forward_2q_matrix(op: &GateOp, params: &ParameterBinding) -> OmegaResult<[Complex64; 16]> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    Ok(match &op.gate {
        GateKind::CX => {
            #[rustfmt::skip]
            let m = [
                o, z, z, z,
                z, z, z, o,
                z, z, o, z,
                z, o, z, z,
            ];
            m
        }
        GateKind::CY => {
            let pi = Complex64::new(0.0, 1.0);
            let mi = Complex64::new(0.0, -1.0);
            #[rustfmt::skip]
            let m = [
                o, z, z, z,
                z, z, z, mi,
                z, z, o, z,
                z, pi, z, z,
            ];
            m
        }
        GateKind::CZ => {
            let m_ = Complex64::new(-1.0, 0.0);
            #[rustfmt::skip]
            let m = [
                o, z, z, z,
                z, o, z, z,
                z, z, o, z,
                z, z, z, m_,
            ];
            m
        }
        GateKind::Swap => {
            #[rustfmt::skip]
            let m = [
                o, z, z, z,
                z, z, o, z,
                z, o, z, z,
                z, z, z, o,
            ];
            m
        }
        GateKind::CRz => {
            let theta = resolved[0];
            let phn = Complex64::from_polar(1.0, -theta / 2.0);
            let php = Complex64::from_polar(1.0, theta / 2.0);
            #[rustfmt::skip]
            let m = [
                o, z, z, z,
                z, phn, z, z,
                z, z, o, z,
                z, z, z, php,
            ];
            m
        }
        GateKind::CU3 => {
            let theta = resolved[0];
            let phi = resolved[1];
            let lambda = resolved[2];
            let c = (theta / 2.0).cos();
            let s = (theta / 2.0).sin();
            let cu_00 = Complex64::new(c, 0.0);
            let cu_01 = -Complex64::from_polar(s, lambda);
            let cu_10 = Complex64::from_polar(s, phi);
            let cu_11 = Complex64::from_polar(c, phi + lambda);
            #[rustfmt::skip]
            let m = [
                o, z, z, z,
                z, cu_00, z, cu_01,
                z, z, o, z,
                z, cu_10, z, cu_11,
            ];
            m
        }
        other => {
            return Err(OmegaError::Unsupported(format!(
                "forward_2q_matrix: gate {other:?} not handled"
            )));
        }
    })
}

// ---- Dagger param resolvers (apply U_k†) ------------------------

fn resolve_dagger_diagonal_params(
    op: &GateOp,
    params: &ParameterBinding,
) -> OmegaResult<DiagonalParams> {
    // For diagonal gates, U† = diag(conj(d0), conj(d1)).
    let mut p = resolve_diagonal_params(op, params)?;
    p.d0_im = -p.d0_im;
    p.d1_im = -p.d1_im;
    Ok(p)
}

fn resolve_dagger_apply_1q_params(
    op: &GateOp,
    params: &ParameterBinding,
) -> OmegaResult<Apply1qParams> {
    let q = op.qubits[0].0;
    let u = forward_1q_matrix(op, params)?;
    // (U†)[r,c] = conj(U[c,r])
    let ud = [u[0].conj(), u[2].conj(), u[1].conj(), u[3].conj()];
    Ok(Apply1qParams::from_matrix(q, &ud))
}

fn resolve_dagger_apply_2q_params(
    op: &GateOp,
    params: &ParameterBinding,
) -> OmegaResult<Apply2qParams> {
    let qa = op.qubits[0].0;
    let qb = op.qubits[1].0;
    let u = forward_2q_matrix(op, params)?;
    let mut ud = [Complex64::new(0.0, 0.0); 16];
    for r in 0..4 {
        for c in 0..4 {
            ud[r * 4 + c] = u[c * 4 + r].conj();
        }
    }
    Ok(Apply2qParams::from_matrix(qa, qb, &ud))
}

// ---- Derivative param resolvers ---------------------------------

fn resolve_deriv_apply_1q_params(
    op: &GateOp,
    params: &ParameterBinding,
    param_idx: usize,
) -> OmegaResult<Apply1qParams> {
    let q = op.qubits[0].0;
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let mat = match (&op.gate, param_idx) {
        (GateKind::Rx, 0) => gates::drx(resolved[0]),
        (GateKind::Ry, 0) => gates::dry(resolved[0]),
        (GateKind::Rz, 0) => gates::drz(resolved[0]),
        (GateKind::U1, 0) => gates::du1_dl(resolved[0]),
        (GateKind::U2, 0) => gates::du2_dp(resolved[0], resolved[1]),
        (GateKind::U2, 1) => gates::du2_dl(resolved[0], resolved[1]),
        (GateKind::U3, 0) => gates::du3_dt(resolved[0], resolved[1], resolved[2]),
        (GateKind::U3, 1) => gates::du3_dp(resolved[0], resolved[1], resolved[2]),
        (GateKind::U3, 2) => gates::du3_dl(resolved[0], resolved[1], resolved[2]),
        (other, idx) => {
            return Err(OmegaError::Unsupported(format!(
                "resolve_deriv_apply_1q_params: no derivative for {other:?} param_idx={idx}"
            )));
        }
    };
    Ok(Apply1qParams::from_matrix(q, &mat))
}

fn resolve_deriv_apply_2q_params(
    op: &GateOp,
    params: &ParameterBinding,
    param_idx: usize,
) -> OmegaResult<Apply2qParams> {
    let qa = op.qubits[0].0;
    let qb = op.qubits[1].0;
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let cpu_mat = match (&op.gate, param_idx) {
        (GateKind::CRz, 0) => gates::dcrz(resolved[0]),
        (GateKind::CU3, 0) => gates::dcu3_dt(resolved[0], resolved[1], resolved[2]),
        (GateKind::CU3, 1) => gates::dcu3_dp(resolved[0], resolved[1], resolved[2]),
        (GateKind::CU3, 2) => gates::dcu3_dl(resolved[0], resolved[1], resolved[2]),
        (other, idx) => {
            return Err(OmegaError::Unsupported(format!(
                "resolve_deriv_apply_2q_params: no derivative for {other:?} param_idx={idx}"
            )));
        }
    };
    // CPU `Gate2Q` uses `q_first`-high indexing; CUDA apply_2q uses
    // `qb`-high. The same row/col swap (perm [0, 2, 1, 3]) the
    // existing adjoint module applies.
    let perm = [0usize, 2, 1, 3];
    let mut out = [Complex64::new(0.0, 0.0); 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = cpu_mat[perm[r] * 4 + perm[c]];
        }
    }
    Ok(Apply2qParams::from_matrix(qa, qb, &out))
}

// Marker function so clippy doesn't complain about unused imports
// during the period the trainer wiring (D4) hasn't landed yet.
#[allow(dead_code)]
fn _hold_ctx_alive(c: &Arc<CudaContext>) -> &Arc<CudaContext> {
    c
}
