//! CUDA-graph capture+replay path for fixed-shape forward sweeps.
//!
//! ## Why
//!
//! On the Phase 4c QML bench (n=14, 16 params, 256 pts × 100 ep) the
//! per-`cuLaunchKernel` CPU overhead (~2.08 µs measured by
//! `tests/graph_replay_smoke.rs`) accounts for ~16 s of pure dispatch
//! CPU time — i.e. the entire wallclock. Capturing the kernel
//! sequence into a `CudaGraph` once and replaying it via
//! `cuGraphLaunch` knocks per-launch overhead to ~0.3 µs.
//!
//! ## How
//!
//! The forward sweep is a deterministic walk over `circuit.ops`. Each
//! op maps to one (or a few) of the pooled-param kernels
//! [`crate::imp::StateBuffer::apply_diagonal_via_pool`],
//! [`apply_1q_via_pool`], [`apply_2q_via_pool`]. We allocate one
//! device-side pool per kernel type, pre-walk the circuit to assign
//! a per-op slot index, then `begin_capture` the stream, issue the
//! kernel sequence reading from the pools, and `end_capture` to
//! produce a `CudaGraph`.
//!
//! Per replay:
//! 1. Resolve every op's gate matrix from the current
//!    `ParameterBinding` into host-side `Vec<DiagonalParams>` /
//!    `Vec<Apply1qParams>` / `Vec<Apply2qParams>`.
//! 2. `memcpy_htod` the three vecs into the corresponding pools.
//! 3. `graph.launch()` — the kernels run in their captured order,
//!    each loading its params from `pool[slot]` at execution time.
//! 4. The state buffer (owned by the graph) holds the result; caller
//!    reads it back via `replay()`'s returned `&CudaState`.
//!
//! ## Scope (this slice)
//!
//! Forward-sweep only. HEA-relevant gate set: H, X, Y, Z, S, Sdg, T,
//! Tdg, Rx, Ry, Rz, U1, U2, U3, CX, CY, CZ, Swap, CRz, CU3. Reset /
//! Measure / Barrier / Id are rejected (the trainer hot path doesn't
//! use them). 3q gates (CCX, CSwap) and the bridge gate types
//! (PhaseShifter, BeamSplitterRx, Custom) are also rejected — adding
//! them is mechanical but each multiplies plan entries per op.
//!
//! Backward-sweep / adjoint capture is a follow-up; the plan walker
//! could be reused with a "dagger=true" flag and inverted op order.

use std::sync::Arc;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode_enum};
use cudarc::driver::{CudaGraph, CudaSlice};
use num_complex::Complex64;

use crate::imp::{Apply1qParams, Apply2qParams, DeviceHandle, DiagonalParams, StateBuffer};
use crate::CudaError;
use omega_core::circuit::{CircuitIR, GateKind, GateOp};
use omega_core::error::{OmegaError, Result as OmegaResult};
use omega_core::params::ParameterBinding;

/// Per-op record describing which pool + slot a captured kernel
/// reads from. Indexed by the order in which kernels were issued
/// during capture, so replay walks the same sequence and writes
/// the resolved gate params into the same slots.
#[derive(Clone, Copy)]
enum PlanEntry {
    Diagonal { slot: u32, op_idx: usize },
    Apply1q { slot: u32, op_idx: usize },
    Apply2q { slot: u32, op_idx: usize },
}

/// Captured forward sweep — owns its state buffer and pools, can
/// be replayed many times with different params.
pub(crate) struct ForwardGraph {
    state: StateBuffer,
    diagonal_pool_dev: CudaSlice<DiagonalParams>,
    apply_1q_pool_dev: CudaSlice<Apply1qParams>,
    apply_2q_pool_dev: CudaSlice<Apply2qParams>,
    diagonal_host: Vec<DiagonalParams>,
    apply_1q_host: Vec<Apply1qParams>,
    apply_2q_host: Vec<Apply2qParams>,
    plan: Vec<PlanEntry>,
    graph: CudaGraph,
}

impl ForwardGraph {
    /// Capture the forward sweep for `circuit` starting from
    /// `|0…0⟩`. Caller must reuse the same circuit *shape* on every
    /// replay (gate sequence + qubits + per-op kind); only param
    /// VALUES may differ.
    pub fn capture(handle: &DeviceHandle, circuit: &CircuitIR) -> OmegaResult<Self> {
        let plan = build_plan(circuit)?;

        let n_diag = plan
            .iter()
            .filter(|p| matches!(p, PlanEntry::Diagonal { .. }))
            .count()
            .max(1);
        let n_1q = plan
            .iter()
            .filter(|p| matches!(p, PlanEntry::Apply1q { .. }))
            .count()
            .max(1);
        let n_2q = plan
            .iter()
            .filter(|p| matches!(p, PlanEntry::Apply2q { .. }))
            .count()
            .max(1);

        // Two CUDA-Graph quirks we have to thread through:
        //
        // 1. The legacy default stream (returned by
        //    `ctx.default_stream()`) doesn't support
        //    `cuStreamBeginCapture` at all — we'd get
        //    `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`. Need a fresh
        //    non-default stream.
        // 2. cudarc's per-slice read/write event tracking (enabled
        //    once `new_stream` flips the ctx into multi-stream mode)
        //    inserts `cuStreamWaitEvent` + `cuEventRecord` calls
        //    around every kernel launch. Those calls invalidate
        //    capture (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`) on
        //    every cudarc 0.19 path I've tried — the event objects
        //    were created in non-capture mode and cuStreamBeginCapture
        //    rejects them. Disabling event tracking BEFORE the
        //    pool/state allocations means those slices come back
        //    with `read = None` / `write = None`, so the launch
        //    builder skips the wait/record entirely. Single-stream
        //    serialization inside the captured graph is enough to
        //    keep ordering correct (CUDA streams execute their
        //    queue in order — that's exactly what gets recorded as
        //    graph dependencies).
        //
        // Caveat: `disable_event_tracking` is `unsafe` because it
        // affects every CudaSlice allocated AFTER this point in the
        // process. We pair it with an immediate re-enable after the
        // capture-time allocations to keep the side-effect
        // window short. Slices allocated outside that window keep
        // their normal event tracking.
        unsafe {
            handle.ctx.disable_event_tracking();
        }
        let graph_stream = handle
            .ctx
            .new_stream()
            .map_err(|e| OmegaError::Backend(format!("new_stream for graph: {e}")))?;
        let graph_handle = DeviceHandle {
            ctx: handle.ctx.clone(),
            stream: graph_stream.clone(),
            kernels: handle.kernels.clone(),
        };

        let diagonal_pool_dev = graph_stream
            .alloc_zeros::<DiagonalParams>(n_diag)
            .map_err(|e| OmegaError::Backend(format!("alloc diagonal pool: {e}")))?;
        let apply_1q_pool_dev = graph_stream
            .alloc_zeros::<Apply1qParams>(n_1q)
            .map_err(|e| OmegaError::Backend(format!("alloc 1q pool: {e}")))?;
        let apply_2q_pool_dev = graph_stream
            .alloc_zeros::<Apply2qParams>(n_2q)
            .map_err(|e| OmegaError::Backend(format!("alloc 2q pool: {e}")))?;

        let mut state = graph_handle
            .allocate(circuit.num_qubits)
            .map_err(|e| OmegaError::Backend(format!("alloc state: {e}")))?;
        // Re-enable event tracking now that the capture-window
        // slices have been created with `read = None` / `write = None`.
        unsafe {
            handle.ctx.enable_event_tracking();
        }

        // Stream capture mode = THREAD_LOCAL so other threads' work
        // on the global default stream isn't accidentally captured
        // into our graph.
        graph_stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| OmegaError::Backend(format!("begin_capture: {e}")))?;

        // Issue every kernel from the plan reading from pool[slot].
        // Capture mode means none of these execute — they just
        // record kernel-launch nodes into the graph.
        let result: Result<(), CudaError> = (|| {
            for entry in &plan {
                match *entry {
                    PlanEntry::Diagonal { slot, .. } => {
                        state.apply_diagonal_via_pool(&diagonal_pool_dev, slot)?;
                    }
                    PlanEntry::Apply1q { slot, .. } => {
                        state.apply_1q_via_pool(&apply_1q_pool_dev, slot)?;
                    }
                    PlanEntry::Apply2q { slot, .. } => {
                        state.apply_2q_via_pool(&apply_2q_pool_dev, slot)?;
                    }
                }
            }
            Ok(())
        })();
        // Always end capture even if launches fail — otherwise the
        // stream stays in capture mode and poisons every subsequent
        // call.
        let graph_opt = graph_stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| OmegaError::Backend(format!("end_capture: {e}")))?;
        result.map_err(OmegaError::from)?;
        let graph =
            graph_opt.ok_or_else(|| OmegaError::Backend("end_capture returned no graph".into()))?;

        Ok(Self {
            state,
            diagonal_pool_dev,
            apply_1q_pool_dev,
            apply_2q_pool_dev,
            diagonal_host: vec![DiagonalParams::default(); n_diag],
            apply_1q_host: vec![Apply1qParams::default(); n_1q],
            apply_2q_host: vec![Apply2qParams::default(); n_2q],
            plan,
            graph,
        })
    }

    /// Replay the captured sweep with fresh `params`. Resets the
    /// owned state to `|0…0⟩` first, stages new gate matrices into
    /// the pools, then launches the graph. Returns a borrow of the
    /// post-sweep `StateBuffer` — caller can `read_state()` from it
    /// or feed it to subsequent (non-graph) work.
    pub fn replay(
        &mut self,
        circuit: &CircuitIR,
        params: &ParameterBinding,
    ) -> OmegaResult<&mut StateBuffer> {
        // 1. Resolve all per-op params + write into host pool vecs.
        for entry in &self.plan {
            match *entry {
                PlanEntry::Diagonal { slot, op_idx } => {
                    let op = &circuit.ops[op_idx];
                    self.diagonal_host[slot as usize] = resolve_diagonal_params(op, params)?;
                }
                PlanEntry::Apply1q { slot, op_idx } => {
                    let op = &circuit.ops[op_idx];
                    self.apply_1q_host[slot as usize] = resolve_apply_1q_params(op, params)?;
                }
                PlanEntry::Apply2q { slot, op_idx } => {
                    let op = &circuit.ops[op_idx];
                    self.apply_2q_host[slot as usize] = resolve_apply_2q_params(op, params)?;
                }
            }
        }

        // 2. Reset state to |0…0⟩. memset_zeros is async on the
        // stream; the subsequent memcpy_htod's queue after it.
        let stream = self.state.handle.stream.clone();
        stream
            .memset_zeros(&mut self.state.state)
            .map_err(|e| OmegaError::Backend(format!("memset_zeros: {e}")))?;
        let one = [1.0f32];
        {
            let mut head = self.state.state.slice_mut(0..1);
            stream
                .memcpy_htod(&one[..], &mut head)
                .map_err(|e| OmegaError::Backend(format!("memcpy init: {e}")))?;
        }

        // 3. Upload param frames to device pools. These pools are
        // the same CudaSlices the captured kernels point at, so the
        // graph replay reads the new values.
        stream
            .memcpy_htod(&self.diagonal_host[..], &mut self.diagonal_pool_dev)
            .map_err(|e| OmegaError::Backend(format!("upload diagonal pool: {e}")))?;
        stream
            .memcpy_htod(&self.apply_1q_host[..], &mut self.apply_1q_pool_dev)
            .map_err(|e| OmegaError::Backend(format!("upload 1q pool: {e}")))?;
        stream
            .memcpy_htod(&self.apply_2q_host[..], &mut self.apply_2q_pool_dev)
            .map_err(|e| OmegaError::Backend(format!("upload 2q pool: {e}")))?;

        // 4. Replay the captured kernel sequence.
        self.graph
            .launch()
            .map_err(|e| OmegaError::Backend(format!("graph launch: {e}")))?;

        Ok(&mut self.state)
    }

    /// Synchronize the underlying graph stream so the host can read
    /// post-replay results. Replay queues all work async; callers
    /// reading state on host normally pick up the sync via
    /// `read_state` (which clones-then-synchronizes), but this
    /// method exists for callers who need an explicit barrier
    /// (e.g. timing harnesses).
    #[allow(dead_code)]
    pub fn synchronize(&self) -> OmegaResult<()> {
        self.state
            .handle
            .stream
            .synchronize()
            .map_err(|e| OmegaError::Backend(format!("sync: {e}")))
    }
}

/// Walk `circuit.ops` and decide which pool + slot each gate maps
/// to. Each gate kind dispatches to exactly one of the three pools
/// in this slice (no decomposition into multi-kernel sequences yet).
fn build_plan(circuit: &CircuitIR) -> OmegaResult<Vec<PlanEntry>> {
    let mut out = Vec::with_capacity(circuit.ops.len());
    let mut diag_slot: u32 = 0;
    let mut q1_slot: u32 = 0;
    let mut q2_slot: u32 = 0;
    for (op_idx, op) in circuit.ops.iter().enumerate() {
        if op.condition.is_some() {
            return Err(OmegaError::Unsupported(
                "cuda forward graph: conditional gates not supported in graph capture (yet)".into(),
            ));
        }
        let entry = match &op.gate {
            // Diagonal 1q gates (one apply_diagonal kernel each).
            GateKind::Z
            | GateKind::S
            | GateKind::Sdg
            | GateKind::T
            | GateKind::Tdg
            | GateKind::Rz
            | GateKind::U1 => {
                let e = PlanEntry::Diagonal {
                    slot: diag_slot,
                    op_idx,
                };
                diag_slot += 1;
                e
            }
            // Generic 1q gates → apply_1q.
            GateKind::H
            | GateKind::X
            | GateKind::Y
            | GateKind::Sx
            | GateKind::Sxdg
            | GateKind::Rx
            | GateKind::Ry
            | GateKind::U2
            | GateKind::U3 => {
                let e = PlanEntry::Apply1q {
                    slot: q1_slot,
                    op_idx,
                };
                q1_slot += 1;
                e
            }
            // 2q gates → apply_2q. CRz / CU3 / CY: same kernel, just
            // different matrix.
            GateKind::CX
            | GateKind::CY
            | GateKind::CZ
            | GateKind::Swap
            | GateKind::CRz
            | GateKind::CU3 => {
                let e = PlanEntry::Apply2q {
                    slot: q2_slot,
                    op_idx,
                };
                q2_slot += 1;
                e
            }
            // Out of scope for this slice. Caller falls back to the
            // non-graph forward path (or the QmlTrainer keeps using
            // `expectation_multi_then_gradient` directly).
            other => {
                return Err(OmegaError::Unsupported(format!(
                    "cuda forward graph: gate {other:?} not yet supported in graph capture"
                )));
            }
        };
        out.push(entry);
    }
    Ok(out)
}

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
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let mat: [Complex64; 4] = match &op.gate {
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
        // √X / √X† — exact Qiskit matrices, NOT the U3 alias (differs by e^{iπ/4}).
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
        GateKind::U2 => {
            let phi = resolved[0];
            let lambda = resolved[1];
            // U2 = U3(π/2, φ, λ); reuse U3 builder.
            u3_matrix(std::f64::consts::FRAC_PI_2, phi, lambda)
        }
        GateKind::U3 => u3_matrix(resolved[0], resolved[1], resolved[2]),
        other => {
            return Err(OmegaError::Unsupported(format!(
                "resolve_apply_1q_params: gate {other:?} not handled"
            )));
        }
    };
    Ok(Apply1qParams::from_matrix(q, &mat))
}

fn u3_matrix(theta: f64, phi: f64, lambda: f64) -> [Complex64; 4] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        Complex64::new(c, 0.0),
        -Complex64::from_polar(s, lambda),
        Complex64::from_polar(s, phi),
        Complex64::from_polar(c, phi + lambda),
    ]
}

fn resolve_apply_2q_params(op: &GateOp, params: &ParameterBinding) -> OmegaResult<Apply2qParams> {
    let qa = op.qubits[0].0;
    let qb = op.qubits[1].0;
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    let mat: [Complex64; 16] = match &op.gate {
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
                "resolve_apply_2q_params: gate {other:?} not handled"
            )));
        }
    };
    Ok(Apply2qParams::from_matrix(qa, qb, &mat))
}

// Suppress: `params: &ParameterBinding` is used by both 1q and 2q
// resolvers, but the diagonal one's `params` arg is technically only
// read inside resolve_diagonal_params for the Rz/U1 cases — clippy
// doesn't see it as never-used because it IS read via match arms.
// No lint expected; this comment is a breadcrumb if one fires later.
#[allow(dead_code)]
const _: () = ();

// Hold an Arc<DeviceHandle> — not strictly required since we already
// have everything we need via `state.handle`, but having a top-level
// tag makes refactors easier.
#[allow(dead_code)]
fn _handle_ref(state: &StateBuffer) -> Arc<cudarc::driver::CudaContext> {
    state.handle.ctx.clone()
}
