//! Adjoint differentiation on Metal.
//!
//! Phase 1, step 9a — first cut. Algorithm mirrors
//! `omega-backend-statevector::adjoint` (Jones-style adjoint):
//!
//!   forward:  |ψ⟩ = U_M·…·U_1 |0⟩
//!   backward: |φ⟩ ← |ψ⟩,  |ν⟩ ← O|ψ⟩
//!   for k = M down to 1:
//!       |φ⟩ ← U_k† |φ⟩
//!       for each parameter θ_p in U_k:
//!           temp = (∂U_k/∂θ_p) |φ⟩
//!           g_p += chain · 2 Re ⟨ν|temp⟩
//!       |ν⟩ ← U_k† |ν⟩
//!
//! Forward gates run on GPU. Backward dagger applies reuse the same GPU
//! kernels (with the conjugate-transposed matrix). Derivative-applies
//! also run on GPU (the apply_1q/apply_2q kernels don't require unitary
//! input). The inner product step ships partials to host via the
//! `inner_product` reduction kernel (step 9c).
//!
//! Observable application `O|ψ⟩` runs on GPU when the observable is
//! diagonal in the computational basis (only Z and identity factors —
//! the QML trainer's gradient observable is exactly this case). The
//! `apply_diagonal_pauli_sum` kernel computes ν[i] = ψ[i] · Σ_k c_k ·
//! (-1)^popcount(i & sign_mask_k) in one dispatch with no host sync.
//! General observables (X / Y components) still take the host
//! read / apply / write path.

use std::collections::HashMap;

use num_complex::Complex64;

use omega_backend_statevector::gates;
use omega_core::circuit::{CircuitIR, GateKind, GateOp, ParamExpr, SymbolId};
use omega_core::error::{OmegaError, Result as OmegaResult};
use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;

use crate::{MetalState, MetalStatevectorBackend};

/// Compute gradients of ⟨ψ(θ)|O|ψ(θ)⟩ via adjoint AD on Metal.
/// Returns `Ok(None)` if the circuit contains a non-unitary op
/// (Reset / Measure under Collapse) — caller falls back to
/// parameter-shift, matching the CPU contract.
///
/// All three working buffers (forward |φ⟩, adjoint |ν⟩, scratch
/// derivative state) come from the backend's [`imp::BufferPool`]
/// via [`MetalStatevectorBackend::lease`]. After the first epoch
/// the QML trainer's hot path runs allocation-free at the
/// statevector level.
pub(crate) fn adjoint_gradient(
    backend: &MetalStatevectorBackend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
) -> OmegaResult<Option<Vec<(SymbolId, f64)>>> {
    adjoint_gradient_inner(backend, circuit, params, observable, None)
}

/// Variant of [`adjoint_gradient`] that takes a caller-provided
/// post-forward `|ψ⟩` state instead of leasing + computing it. Used
/// by [`MetalStatevectorBackend::expectation_multi_then_gradient`]
/// to avoid running the forward sweep twice (once for predictions,
/// once for gradients) — a ~9% Metal wallclock savings on the QML
/// bench at the post-round-13 baseline.
///
/// `pre_forward_state.num_qubits()` must equal `circuit.num_qubits`.
pub(crate) fn adjoint_gradient_with_forward_state(
    backend: &MetalStatevectorBackend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    pre_forward_state: crate::MetalState,
) -> OmegaResult<Option<Vec<(SymbolId, f64)>>> {
    adjoint_gradient_inner(
        backend,
        circuit,
        params,
        observable,
        Some(pre_forward_state),
    )
}

fn adjoint_gradient_inner(
    backend: &MetalStatevectorBackend,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    pre_forward_state: Option<crate::MetalState>,
) -> OmegaResult<Option<Vec<(SymbolId, f64)>>> {
    if circuit.ops.iter().any(|op| !is_unitary(&op.gate)) {
        return Ok(None);
    }

    let n = circuit.num_qubits;
    let unitary_ops: Vec<&GateOp> = circuit
        .ops
        .iter()
        .filter(|op| is_unitary(&op.gate))
        .collect();

    // Forward sweep on GPU. The fusion walker collapses consecutive
    // unconditional diagonal gates (Z / S / Sdg / T / Tdg / Rz / U1)
    // into a single `apply_diagonal_product` dispatch — the second
    // HEA layer's eight Rz gates on q4..q11 go from eight dispatches
    // to one. `unitary_ops` is `Vec<&GateOp>`; `.iter().copied()`
    // produces the `Item = &GateOp` iterator the helper expects.
    //
    // When the caller has already computed the post-forward state
    // (the trainer's `expectation_multi_then_gradient` path), reuse
    // it instead of redoing the sweep.
    let phi_state = match pre_forward_state {
        Some(s) => {
            if s.num_qubits() != n {
                return Err(OmegaError::Backend(format!(
                    "metal adjoint_gradient: pre-forward state size mismatch \
                     (expected {n} qubits, got {})",
                    s.num_qubits()
                )));
            }
            s
        }
        None => {
            let phi = backend
                .lease(n)
                .map_err(|e| OmegaError::Backend(format!("metal lease phi: {e}")))?;
            crate::apply_ops_fused(&phi, unitary_ops.iter().copied(), params, |_| false, None)?;
            phi
        }
    };

    // Initialize ν = O|ψ⟩. When the observable is diagonal in the
    // computational basis (only Z and identity Pauli factors — the
    // QML trainer's case), use a single GPU kernel that computes
    // ν[i] = ψ[i] · Σ_k c_k · (-1)^popcount(i & sign_mask_k) without
    // ever touching the host. General observables (X / Y components)
    // still pay the read/apply/write roundtrip.
    let mut nu_state = backend
        .lease(n)
        .map_err(|e| OmegaError::Backend(format!("metal lease nu: {e}")))?;
    if let Some(diag_terms) = diagonal_pauli_terms(observable) {
        phi_state
            .apply_diagonal_pauli_sum(&nu_state, &diag_terms)
            .map_err(|e| OmegaError::Backend(format!("metal apply_diagonal_pauli_sum: {e}")))?;
    } else {
        let psi_host = phi_state.read_state();
        let nu_host = apply_observable_host(&psi_host, n, observable);
        nu_state
            .write_state(&nu_host)
            .map_err(|e| OmegaError::Backend(format!("metal write nu: {e}")))?;
    }

    // Lease one reusable scratch buffer for the per-parameter
    // derivative state. Recycled across every (op, param_idx, sym)
    // iteration via `phi_state.copy_into(&temp_state)`. The lease
    // returns to the pool on drop at the end of this function — the
    // next adjoint call reuses it without an MTLBuffer allocation.
    let temp_state = backend
        .lease(n)
        .map_err(|e| OmegaError::Backend(format!("metal lease temp: {e}")))?;

    // Backward sweep.
    //
    // Round 16: phi.daggers and nu.daggers from consecutive
    // parameter-less ops (CX layer, encoding Ry layer) chain into
    // their respective state's batch instead of paying a per-op
    // commit+wait. phi's batch is flushed by `copy_into` inside the
    // param loop; nu's batch is flushed by `inner_product`'s ride-on
    // logic (which encodes the reduction into nu's encoder when nu
    // has pending work). Both batches re-open after each flush so
    // the next iter's daggers continue to chain. End_batch on both
    // at function exit cleans up any tail daggers from the final
    // param-less cluster (encoding Ry in the HEA backward sweep).
    //
    // Empty-batch dispatches (the very first iter's nu batch when
    // nu hasn't been daggered yet) are skipped via the
    // `dispatch_count` short-circuit in `end_batch_if_open` — no
    // wasted commit+wait.
    phi_state.inner.begin_batch();
    nu_state.inner.begin_batch();
    let mut gradients: HashMap<SymbolId, f64> = HashMap::new();
    // Each backward-loop iteration opens 2-3 fresh `MTLCommandBuffer`s
    // (the per-param `phi.end_batch + begin_batch` pair plus the
    // ride-on `inner_product`). Metal hands them out as autoreleased
    // NSObjects from the queue; without a draining
    // `objc::rc::autoreleasepool` they pile up under the queue's
    // 64-outstanding cap, and `commandBuffer` eventually blocks
    // forever on its dispatch semaphore. The QML training bench
    // doesn't trip this because n=4/6p × 30 epochs stays under the
    // limit; the QUBO/QAOA harness at K4 depth=3 (~30
    // parameterised gate-ops × 3 cmd_bufs each ≈ 90) does. Wrap
    // each iteration in its own pool so command buffers and
    // encoders allocated inside (~3 per iter) drop at the iteration
    // boundary.
    for op in unitary_ops.iter().rev() {
        let result = drain_autorelease(|| {
            // |φ⟩ ← U_k† |φ⟩  (encoded into phi's batch)
            apply_op_dagger(&phi_state, op, params)?;

            // For each parameter slot in U_k, accumulate gradient.
            for (param_idx, param_expr) in op.params.iter().enumerate() {
                let syms = collect_symbols(param_expr);
                for sym_id in syms {
                    let chain = params.resolve_derivative(param_expr, sym_id)?;
                    if chain.abs() < 1e-30 {
                        continue;
                    }
                    apply_op_derivative_into(&phi_state, &temp_state, op, params, param_idx)?;
                    phi_state.inner.end_batch_if_open();
                    phi_state.inner.begin_batch();
                    let ip = nu_state
                        .inner_product(&temp_state)
                        .map_err(|e| OmegaError::Backend(format!("metal inner_product: {e}")))?;
                    nu_state.inner.begin_batch();
                    *gradients.entry(sym_id).or_insert(0.0) += 2.0 * ip.re * chain;
                }
            }

            // |ν⟩ ← U_k† |ν⟩  (encoded into nu's batch)
            apply_op_dagger(&nu_state, op, params)?;
            Ok::<(), OmegaError>(())
        });
        result?;
    }
    // Flush any tail daggers from the last param-less cluster
    // (encoding Ry layer in the HEA backward sweep — the loop ends
    // with no inner_product to drain nu, and no copy_into to drain
    // phi).
    phi_state.inner.end_batch();
    nu_state.inner.end_batch();

    let mut result: Vec<(SymbolId, f64)> = circuit
        .symbols
        .keys()
        .map(|&sym| (sym, gradients.get(&sym).copied().unwrap_or(0.0)))
        .collect();
    result.sort_by_key(|(id, _)| *id);
    Ok(Some(result))
}

fn is_unitary(gate: &GateKind) -> bool {
    !matches!(
        gate,
        GateKind::Measure | GateKind::Barrier | GateKind::Reset
    )
}

/// Run `f()` inside an `objc::rc::autoreleasepool`. On macOS the
/// pool drains autoreleased Objective-C objects (notably
/// `MTLCommandBuffer` / `MTLComputeCommandEncoder`) on scope exit,
/// preventing accumulation under the queue's 64-outstanding cap.
/// On non-macOS targets this is the identity function — the metal
/// feature compiles to a stub backend there anyway.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn drain_autorelease<R, F: FnOnce() -> R>(f: F) -> R {
    objc::rc::autoreleasepool(f)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn drain_autorelease<R, F: FnOnce() -> R>(f: F) -> R {
    f()
}

/// Recurse through a `ParamExpr`, collecting every free symbol that
/// appears. ParamExpr currently has 5 variants: Concrete, Symbol,
/// Negate, Add, Mul.
fn collect_symbols(expr: &ParamExpr) -> Vec<SymbolId> {
    fn walk(e: &ParamExpr, out: &mut Vec<SymbolId>) {
        match e {
            ParamExpr::Concrete(_) => {}
            ParamExpr::Symbol(s) => out.push(*s),
            ParamExpr::Add(a, b) | ParamExpr::Mul(a, b) => {
                walk(a, out);
                walk(b, out);
            }
            ParamExpr::Negate(a) => {
                walk(a, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(expr, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

/// Apply U_k† to a `MetalState` in place. For involutions (H, X, Y, Z,
/// CX, CY, CZ, SWAP, CCX, CSwap) the dagger is the same op. For
/// S↔Sdg / T↔Tdg use the partner. For parameterized gates with simple
/// adjoints (Rx, Ry, Rz, U1, CRz) negate the angle. For the rest
/// (U2, U3, CU3) build the matrix and conjugate-transpose.
fn apply_op_dagger(state: &MetalState, op: &GateOp, params: &ParameterBinding) -> OmegaResult<()> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let q0 = op.qubits[0].0;

    let res: Result<(), crate::MetalError> = match &op.gate {
        // Self-adjoint involutions
        GateKind::H => state.apply_h(q0),
        GateKind::X => state.apply_x(q0),
        GateKind::Y => state.apply_y(q0),
        GateKind::Z => state.apply_z(q0),
        GateKind::Id | GateKind::Barrier => Ok(()),

        // Pairs
        GateKind::S => state.apply_sdg(q0),
        GateKind::Sdg => state.apply_s(q0),
        // sx† = sxdg (verified `sx @ sxdg == I` to 0.000e+00).
        GateKind::Sx => state.apply_1q(q0, &[Complex64::new(0.5, -0.5), Complex64::new(0.5, 0.5), Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5)]),
        GateKind::Sxdg => state.apply_1q(q0, &[Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5), Complex64::new(0.5, -0.5), Complex64::new(0.5, 0.5)]),
        GateKind::T => state.apply_tdg(q0),
        GateKind::Tdg => state.apply_t(q0),

        // Parameterized 1q with negated-angle dagger
        GateKind::Rx => state.apply_rx(q0, -resolved[0]),
        GateKind::Ry => state.apply_ry(q0, -resolved[0]),
        GateKind::Rz => state.apply_rz(q0, -resolved[0]),
        GateKind::U1 => state.apply_u1(q0, -resolved[0]),

        // Parameterized 1q via matrix conj-transpose
        GateKind::U2 => apply_1q_dagger_via_matrix(state, q0, &gates::u2(resolved[0], resolved[1])),
        GateKind::U3 => {
            apply_1q_dagger_via_matrix(state, q0, &gates::u3(resolved[0], resolved[1], resolved[2]))
        }

        // 2q gates
        GateKind::CX => state.apply_cx(q0, op.qubits[1].0),
        GateKind::CY => state.apply_cy(q0, op.qubits[1].0),
        GateKind::CZ => state.apply_cz(q0, op.qubits[1].0),
        GateKind::Swap => state.apply_swap(q0, op.qubits[1].0),
        GateKind::CRz => state.apply_crz(q0, op.qubits[1].0, -resolved[0]),
        GateKind::CU3 => apply_2q_dagger_via_matrix(
            state,
            q0,
            op.qubits[1].0,
            &perm_2q_to_metal(&gates::cu3(resolved[0], resolved[1], resolved[2])),
        ),
        // RBS(θ)† = RBS(−θ) (real orthogonal Givens rotation).
        GateKind::Rbs => state.apply_rbs(q0, op.qubits[1].0, -resolved[0]),

        // 3q involutions — applying CCX / CSwap again undoes them.
        GateKind::CCX => state.apply_ccx(q0, op.qubits[1].0, op.qubits[2].0),
        GateKind::CSwap => state.apply_cswap(q0, op.qubits[1].0, op.qubits[2].0),

        // Reset / Measure / photonic / Custom rejected upstream (no native
        // Metal kernel; the CPU statevector backend handles these).
        GateKind::Reset
        | GateKind::Measure
        | GateKind::PhaseShifter
        | GateKind::BeamSplitterRx
        | GateKind::Custom(_) => {
            return Err(OmegaError::Unsupported(format!(
                "metal adjoint dagger: unsupported gate {:?}",
                op.gate
            )));
        }
    };
    res.map_err(OmegaError::from)
}

/// Apply (∂U_k / ∂θ_p) reading from `src`, writing to `dst`, where
/// p = `param_idx`. Sister of the in-place
/// `apply_op_derivative`-shaped logic but using src+dst kernels —
/// eliminates the prior `copy_into(src → dst) + apply_op_derivative
/// (dst, ...)` two-step. The kernel reads src.state via the
/// in-encoder memory barrier (so any pending daggers in src's open
/// batch land before the read) and writes to dst.state directly.
///
/// For gate kinds that don't yet have an `_into` variant (CU3 and
/// most U2/U3 paths), falls back to `copy_into + in-place
/// derivative` — preserves correctness, no perf gain but also no
/// regression. The HEA bench's hot path (Ry, Rz) hits the
/// fast-path branches.
fn apply_op_derivative_into(
    src: &MetalState,
    dst: &MetalState,
    op: &GateOp,
    params: &ParameterBinding,
    param_idx: usize,
) -> OmegaResult<()> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let q0 = op.qubits[0].0;
    let res: Result<(), crate::MetalError> = match (&op.gate, param_idx) {
        (GateKind::Rx, 0) => src.apply_1q_into(dst, q0, &gates::drx(resolved[0])),
        (GateKind::Ry, 0) => src.apply_1q_into(dst, q0, &gates::dry(resolved[0])),
        (GateKind::Rz, 0) => src.apply_drz_into(dst, q0, resolved[0]),
        (GateKind::U1, 0) => src.apply_du1_into(dst, q0, resolved[0]),
        (GateKind::U2, 0) => src.apply_1q_into(dst, q0, &gates::du2_dp(resolved[0], resolved[1])),
        (GateKind::U2, 1) => src.apply_1q_into(dst, q0, &gates::du2_dl(resolved[0], resolved[1])),
        (GateKind::U3, 0) => src.apply_1q_into(
            dst,
            q0,
            &gates::du3_dt(resolved[0], resolved[1], resolved[2]),
        ),
        (GateKind::U3, 1) => src.apply_1q_into(
            dst,
            q0,
            &gates::du3_dp(resolved[0], resolved[1], resolved[2]),
        ),
        (GateKind::U3, 2) => src.apply_1q_into(
            dst,
            q0,
            &gates::du3_dl(resolved[0], resolved[1], resolved[2]),
        ),
        // 2q derivatives don't have _into kernels yet — fall back to
        // the legacy copy_into + in-place path for CU3 / CRz. CRz's
        // dCRz is the round-9b apply_diagonal_2q fast-path; CU3
        // takes apply_2q. Both still in-place.
        (GateKind::CRz, 0)
        | (GateKind::CU3, 0)
        | (GateKind::CU3, 1)
        | (GateKind::CU3, 2)
        | (GateKind::Rbs, 0) => {
            // Slow path: copy then in-place. Caller is responsible
            // for the memcpy here so this branch stays a single
            // function call.
            src.copy_into(dst).map_err(OmegaError::from)?;
            return apply_op_derivative(dst, op, params, param_idx);
        }
        _ => {
            return Err(OmegaError::Unsupported(format!(
                "metal adjoint: no derivative-into for {:?} param_idx={param_idx}",
                op.gate
            )));
        }
    };
    res.map_err(OmegaError::from)
}

/// Apply (∂U_k / ∂θ_p) to a `MetalState` in place, where p = `param_idx`.
/// Kept for the CU3 fallback path inside `apply_op_derivative_into`.
fn apply_op_derivative(
    state: &MetalState,
    op: &GateOp,
    params: &ParameterBinding,
    param_idx: usize,
) -> OmegaResult<()> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let q0 = op.qubits[0].0;
    let res: Result<(), crate::MetalError> = match (&op.gate, param_idx) {
        // 1q derivatives — Gate1Q = [Complex64; 4] matches apply_1q's
        // expected layout exactly; no permutation needed.
        (GateKind::Rx, 0) => state.apply_1q(q0, &gates::drx(resolved[0])),
        (GateKind::Ry, 0) => state.apply_1q(q0, &gates::dry(resolved[0])),
        // Rz / U1 derivatives are diagonal — route through the
        // `apply_diagonal` kernel (half the per-amplitude memory
        // traffic vs the generic `apply_1q` matvec, no host-side
        // `Gate1Q` build).
        (GateKind::Rz, 0) => state.apply_drz(q0, resolved[0]),
        (GateKind::U1, 0) => state.apply_du1(q0, resolved[0]),
        (GateKind::U2, 0) => state.apply_1q(q0, &gates::du2_dp(resolved[0], resolved[1])),
        (GateKind::U2, 1) => state.apply_1q(q0, &gates::du2_dl(resolved[0], resolved[1])),
        (GateKind::U3, 0) => {
            state.apply_1q(q0, &gates::du3_dt(resolved[0], resolved[1], resolved[2]))
        }
        (GateKind::U3, 1) => {
            state.apply_1q(q0, &gates::du3_dp(resolved[0], resolved[1], resolved[2]))
        }
        (GateKind::U3, 2) => {
            state.apply_1q(q0, &gates::du3_dl(resolved[0], resolved[1], resolved[2]))
        }
        // 2q derivatives — CPU gates::* uses control-high indexing,
        // Metal apply_2q uses qb-high indexing; perm_2q_to_metal swaps
        // rows/cols 1 and 2 to convert.
        // CRz derivative is diagonal; route through apply_diagonal_2q
        // (half the per-amplitude memory traffic vs apply_2q).
        (GateKind::CRz, 0) => state.apply_dcrz(q0, op.qubits[1].0, resolved[0]),
        (GateKind::CU3, 0) => state.apply_2q(
            q0,
            op.qubits[1].0,
            &perm_2q_to_metal(&gates::dcu3_dt(resolved[0], resolved[1], resolved[2])),
        ),
        (GateKind::CU3, 1) => state.apply_2q(
            q0,
            op.qubits[1].0,
            &perm_2q_to_metal(&gates::dcu3_dp(resolved[0], resolved[1], resolved[2])),
        ),
        (GateKind::CU3, 2) => state.apply_2q(
            q0,
            op.qubits[1].0,
            &perm_2q_to_metal(&gates::dcu3_dl(resolved[0], resolved[1], resolved[2])),
        ),
        // dRBS/dθ — same [0,2,1,3] perm as the other CPU 2q derivative
        // matrices (matches the forward `apply_rbs` convention).
        (GateKind::Rbs, 0) => state.apply_2q(
            q0,
            op.qubits[1].0,
            &perm_2q_to_metal(&gates::drbs(resolved[0])),
        ),
        _ => {
            return Err(OmegaError::Unsupported(format!(
                "metal adjoint: no derivative for {:?} param_idx={param_idx}",
                op.gate
            )));
        }
    };
    res.map_err(OmegaError::from)
}

fn apply_1q_dagger_via_matrix(
    state: &MetalState,
    qubit: u32,
    u: &[Complex64; 4],
) -> Result<(), crate::MetalError> {
    // 2x2 conjugate-transpose: (U†)[r,c] = conj(U[c,r])
    let ud = [u[0].conj(), u[2].conj(), u[1].conj(), u[3].conj()];
    state.apply_1q(qubit, &ud)
}

fn apply_2q_dagger_via_matrix(
    state: &MetalState,
    qa: u32,
    qb: u32,
    u: &[Complex64; 16],
) -> Result<(), crate::MetalError> {
    let mut ud = [Complex64::new(0.0, 0.0); 16];
    for r in 0..4 {
        for c in 0..4 {
            ud[r * 4 + c] = u[c * 4 + r].conj();
        }
    }
    state.apply_2q(qa, qb, &ud)
}

/// CPU `gates::Gate1Q` is `[Complex64; 4]` in the same ordering Metal's
/// `apply_1q` expects (row-major u00, u01, u10, u11). No permutation
/// needed for 1q matrices.
///
/// CPU `gates::Gate2Q` uses the basis labelling |q_first q_second⟩ —
/// q_first contributes the *high* bit of the row/col index. Metal's
/// `apply_2q(qa, qb, u)` uses `row = bit_qb*2 + bit_qa` — qb is high,
/// qa is low. Both are consistent if the caller hands `qa=q_first`,
/// `qb=q_second`, but the *matrix entries* still need the high/low
/// bit positions swapped, because:
///   CPU row index = q_first*2 + q_second
///   Metal row index = qb*2 + qa = q_second*2 + q_first
/// So the row/col indices are bit-swapped. Permutation σ on 4 indices
/// that swaps positions 1 and 2 (i.e. swaps the high/low bits) does
/// the conversion.
fn perm_2q_to_metal(g: &gates::Gate2Q) -> [Complex64; 16] {
    let perm = [0usize, 2, 1, 3];
    let mut out = [Complex64::new(0.0, 0.0); 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = g[perm[r] * 4 + perm[c]];
        }
    }
    out
}

/// Apply an observable `O = Σ c_k P_k` to a host-side statevector,
/// returning O|ψ⟩. Mirrors `omega-backend-statevector::adjoint::apply_observable`.
fn apply_observable_host(state: &[Complex64], num_qubits: u32, obs: &Observable) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); state.len()];
    for (coeff, pauli_string) in &obs.terms {
        let term = apply_pauli_string(state, num_qubits, pauli_string);
        for (o, t) in out.iter_mut().zip(term.iter()) {
            *o += Complex64::new(*coeff, 0.0) * t;
        }
    }
    out
}

fn apply_pauli_string(
    state: &[Complex64],
    num_qubits: u32,
    pauli_string: &[(u32, PauliOp)],
) -> Vec<Complex64> {
    let dim = 1usize << num_qubits;
    let mut out = vec![Complex64::new(0.0, 0.0); dim];
    for (j, amp) in state.iter().enumerate().take(dim) {
        let mut i = j;
        let mut phase = Complex64::new(1.0, 0.0);
        for &(q, ref p) in pauli_string {
            let bit = (j >> q) & 1;
            match p {
                PauliOp::I => {}
                PauliOp::X => {
                    i ^= 1usize << q;
                }
                PauliOp::Y => {
                    i ^= 1usize << q;
                    if bit == 0 {
                        phase *= Complex64::new(0.0, 1.0);
                    } else {
                        phase *= Complex64::new(0.0, -1.0);
                    }
                }
                PauliOp::Z => {
                    if bit == 1 {
                        phase = -phase;
                    }
                }
            }
        }
        // O|j⟩ = phase · |i⟩, so out[i] gets phase * state[j]
        out[i] += phase * amp;
    }
    out
}

// `inner_product_host` removed in step 9c — superseded by the GPU
// `inner_product` reduction kernel landed in step 7. Kept here as a
// historical note: the implementation was a one-liner
// `a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()` that pulled
// 2·dim·8 bytes per parameter to host before reducing.

/// Classify an `Observable` as diagonal-only: every term must consist
/// solely of `Z` and `I` Pauli factors. Returns `Some` of the
/// `(sign_mask, coeff_f32)` pairs the Metal `apply_diagonal_pauli_sum`
/// kernel consumes when the observable qualifies; `None` otherwise.
///
/// `sign_mask` has a 1 bit at every qubit the term's Z acts on.
/// Identity-only terms collapse to `sign_mask = 0` (constant scaling).
/// Coefficients are downcast to f32 to match the kernel's input
/// type — the QML trainer's residuals are already f32-precision-
/// representable in practice, and the f64→f32 step gates the
/// off-diagonal-Pauli fallback so noisier observables route through
/// the host path.
fn diagonal_pauli_terms(obs: &Observable) -> Option<Vec<(u32, f32)>> {
    let mut out = Vec::with_capacity(obs.terms.len());
    for (coeff, pauli_string) in &obs.terms {
        let mut sign_mask: u32 = 0;
        for &(q, ref p) in pauli_string {
            match p {
                PauliOp::I => {}
                PauliOp::Z => sign_mask |= 1u32 << q,
                PauliOp::X | PauliOp::Y => return None,
            }
        }
        out.push((sign_mask, *coeff as f32));
    }
    Some(out)
}

#[cfg(test)]
mod diagonal_classifier_tests {
    use super::*;

    #[test]
    fn diagonal_observable_with_single_z_term_classifies() {
        let obs = Observable {
            terms: vec![(2.5, vec![(3, PauliOp::Z)])],
        };
        let terms = diagonal_pauli_terms(&obs).expect("Z-only must classify");
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].0, 1u32 << 3);
        assert_eq!(terms[0].1, 2.5);
    }

    #[test]
    fn diagonal_observable_with_multiple_z_terms_collects_each() {
        // QML trainer's typical observable: Σ_i 2·r_i · Z_{q_i}.
        let obs = Observable {
            terms: vec![
                (1.0, vec![(0, PauliOp::Z)]),
                (-0.5, vec![(2, PauliOp::Z)]),
                (0.25, vec![(1, PauliOp::Z), (3, PauliOp::Z)]),
            ],
        };
        let terms = diagonal_pauli_terms(&obs).expect("multi-Z must classify");
        assert_eq!(terms.len(), 3);
        assert_eq!(terms[0], (0b0001, 1.0));
        assert_eq!(terms[1], (0b0100, -0.5));
        assert_eq!(terms[2], (0b1010, 0.25));
    }

    #[test]
    fn identity_only_term_collapses_to_zero_mask() {
        // A constant offset (identity-only term) — the kernel applies
        // it as a uniform scalar across every amplitude.
        let obs = Observable {
            terms: vec![(0.7, vec![])],
        };
        let terms = diagonal_pauli_terms(&obs).expect("identity must classify");
        assert_eq!(terms, vec![(0, 0.7)]);
    }

    #[test]
    fn x_component_falls_back_to_host_path() {
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)]), (1.0, vec![(1, PauliOp::X)])],
        };
        assert!(diagonal_pauli_terms(&obs).is_none());
    }

    #[test]
    fn y_component_falls_back_to_host_path() {
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Y)])],
        };
        assert!(diagonal_pauli_terms(&obs).is_none());
    }

    #[test]
    fn empty_observable_classifies_as_diagonal() {
        // Caller short-circuits the empty-observable case before the
        // kernel dispatch, but the classifier should report it as
        // "trivially diagonal" rather than rejecting it as
        // not-yet-supported.
        let obs = Observable { terms: vec![] };
        let terms = diagonal_pauli_terms(&obs).expect("empty must classify");
        assert!(terms.is_empty());
    }
}
