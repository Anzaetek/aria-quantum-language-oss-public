//! Adjoint differentiation on CUDA — port of
//! `omega-backend-statevector-metal/src/adjoint.rs`.
//!
//! Algorithm mirrors the CPU + Metal versions:
//!
//!   forward:  |ψ⟩ = U_M·…·U_1 |0⟩
//!   backward: |φ⟩ ← |ψ⟩,  |ν⟩ ← O|ψ⟩
//!   for k = M down to 1:
//!       |φ⟩ ← U_k† |φ⟩
//!       for each parameter θ_p in U_k:
//!           temp = (∂U_k/∂θ_p) |φ⟩
//!           g_p += chain · 2·Re ⟨ν|temp⟩
//!       |ν⟩ ← U_k† |ν⟩
//!
//! Forward + dagger + derivative applies all run on GPU. Inner
//! product uses the two-stage block reduction kernel — only the
//! per-block partials cross to host.
//!
//! Observable application `O|ψ⟩` runs entirely on GPU when the
//! observable is diagonal (Z + I only) — the QML trainer's gradient
//! observable hits this path. General observables (X / Y components)
//! still take the host read / apply / write roundtrip.
//!
//! ## Multi-stream overlap
//!
//! The current cut runs the full adjoint loop on the default stream;
//! a follow-up will fork a worker stream so the per-parameter
//! derivative-apply + inner_product launches overlap with the next
//! op's `phi` dagger sweep. The kernels and synchronization points
//! are already structured to admit that change without reshaping the
//! algorithm.

use std::collections::HashMap;

use num_complex::Complex64;

use omega_backend_statevector::gates;
use omega_core::circuit::{CircuitIR, GateKind, GateOp, ParamExpr, SymbolId};
use omega_core::error::{OmegaError, Result as OmegaResult};
use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;

use crate::imp::DeviceHandle;
use crate::CudaState;

pub(crate) fn adjoint_gradient(
    handle: &DeviceHandle,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
) -> OmegaResult<Option<Vec<(SymbolId, f64)>>> {
    adjoint_gradient_inner(handle, circuit, params, observable, None)
}

/// Variant of [`adjoint_gradient`] that takes a caller-provided
/// post-forward `|ψ⟩` state instead of recomputing it. Used by
/// [`crate::CudaStatevectorBackend::expectation_multi_then_gradient`]
/// to skip the second forward sweep when the trainer asks for
/// predictions and gradients in one go (mirrors Metal round 15).
pub(crate) fn adjoint_gradient_with_forward_state(
    handle: &DeviceHandle,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    pre_forward_state: CudaState,
) -> OmegaResult<Option<Vec<(SymbolId, f64)>>> {
    adjoint_gradient_inner(handle, circuit, params, observable, Some(pre_forward_state))
}

fn adjoint_gradient_inner(
    handle: &DeviceHandle,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
    pre_forward_state: Option<CudaState>,
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

    // Forward sweep — reuse the caller's post-forward state when one
    // was provided, otherwise compute it via the fusion walker.
    let mut phi_state = match pre_forward_state {
        Some(s) => {
            if s.num_qubits() != n {
                return Err(OmegaError::Backend(format!(
                    "cuda adjoint_gradient: pre-forward state size mismatch \
                     (expected {n} qubits, got {})",
                    s.num_qubits()
                )));
            }
            s
        }
        None => {
            let phi_inner = handle
                .allocate(n)
                .map_err(|e| OmegaError::Backend(format!("cuda alloc phi: {e}")))?;
            let mut phi_state = CudaState { inner: phi_inner };
            crate::apply_ops_fused(&mut phi_state, unitary_ops.iter().copied(), params, |_| {
                false
            })?;
            phi_state
        }
    };

    // Initialize ν = O|ψ⟩.
    let nu_inner = handle
        .allocate(n)
        .map_err(|e| OmegaError::Backend(format!("cuda alloc nu: {e}")))?;
    let mut nu_state = CudaState { inner: nu_inner };
    if let Some(diag_terms) = diagonal_pauli_terms(observable) {
        phi_state
            .apply_diagonal_pauli_sum(&mut nu_state, &diag_terms)
            .map_err(|e| OmegaError::Backend(format!("cuda apply_diagonal_pauli_sum: {e}")))?;
    } else {
        let psi_host = phi_state
            .read_state()
            .map_err(|e| OmegaError::Backend(format!("cuda read psi: {e}")))?;
        let nu_host = apply_observable_host(&psi_host, n, observable);
        nu_state
            .write_state(&nu_host)
            .map_err(|e| OmegaError::Backend(format!("cuda write nu: {e}")))?;
    }

    // Reusable scratch for per-parameter derivative state — one
    // alloc total, recycled via copy_into.
    let temp_inner = handle
        .allocate(n)
        .map_err(|e| OmegaError::Backend(format!("cuda alloc temp: {e}")))?;
    let mut temp_state = CudaState { inner: temp_inner };

    // Per-(op, sym) inner_product launches stream-async via
    // `inner_product_deferred` — partials memcpy_dtoh queues but the
    // host bytes aren't valid until we sync below. We stash the
    // (sym_id, chain, PendingInnerProduct) tuples and reduce after
    // ONE end-of-loop synchronize() so the per-step host-sync count
    // collapses from ~num_params·num_ops down to 1. At n=14 / 16 params
    // / 51-op HEA that's ~480 syncs eliminated per training step,
    // which is the dominant per-step overhead at small statevector
    // sizes (per-launch latency ~tens of µs vs ~ µs of per-amp work).
    let mut gradients: HashMap<SymbolId, f64> = HashMap::new();
    let mut pending: Vec<(SymbolId, f64, crate::imp::PendingInnerProduct)> = Vec::new();
    for op in unitary_ops.iter().rev() {
        // |φ⟩ ← U_k† |φ⟩
        apply_op_dagger(&mut phi_state, op, params)?;

        for (param_idx, param_expr) in op.params.iter().enumerate() {
            let syms = collect_symbols(param_expr);
            for sym_id in syms {
                let chain = params.resolve_derivative(param_expr, sym_id)?;
                if chain.abs() < 1e-30 {
                    continue;
                }
                phi_state
                    .copy_into(&mut temp_state)
                    .map_err(|e| OmegaError::Backend(format!("cuda copy_into: {e}")))?;
                apply_op_derivative(&mut temp_state, op, params, param_idx)?;
                let pending_ip = nu_state
                    .inner
                    .inner_product_deferred(&temp_state.inner)
                    .map_err(|e| OmegaError::Backend(format!("cuda inner_product: {e}")))?;
                pending.push((sym_id, chain, pending_ip));
            }
        }

        // |ν⟩ ← U_k† |ν⟩
        apply_op_dagger(&mut nu_state, op, params)?;
    }

    // One sync to flush all pending memcpy_dtoh's, then reduce
    // host-side. After this point every PendingInnerProduct's host
    // buffer is fully populated.
    handle
        .stream
        .synchronize()
        .map_err(|e| OmegaError::Backend(format!("cuda final sync: {e}")))?;
    for (sym_id, chain, pending_ip) in pending {
        let ip = pending_ip.reduce();
        *gradients.entry(sym_id).or_insert(0.0) += 2.0 * ip.re * chain;
    }

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

fn apply_op_dagger(
    state: &mut CudaState,
    op: &GateOp,
    params: &ParameterBinding,
) -> OmegaResult<()> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let q0 = op.qubits[0].0;

    let res: Result<(), crate::CudaError> = match &op.gate {
        GateKind::H => state.apply_h(q0),
        GateKind::X => state.apply_x(q0),
        GateKind::Y => state.apply_y(q0),
        GateKind::Z => state.apply_z(q0),
        GateKind::Id | GateKind::Barrier => Ok(()),

        GateKind::S => state.apply_sdg(q0),
        GateKind::Sdg => state.apply_s(q0),
        GateKind::T => state.apply_tdg(q0),
        GateKind::Tdg => state.apply_t(q0),

        GateKind::Rx => state.apply_rx(q0, -resolved[0]),
        GateKind::Ry => state.apply_ry(q0, -resolved[0]),
        GateKind::Rz => state.apply_rz(q0, -resolved[0]),
        GateKind::U1 => state.apply_u1(q0, -resolved[0]),

        GateKind::U2 => apply_1q_dagger_via_matrix(state, q0, &gates::u2(resolved[0], resolved[1])),
        GateKind::U3 => {
            apply_1q_dagger_via_matrix(state, q0, &gates::u3(resolved[0], resolved[1], resolved[2]))
        }

        GateKind::CX => state.apply_cx(q0, op.qubits[1].0),
        GateKind::CY => state.apply_cy(q0, op.qubits[1].0),
        GateKind::CZ => state.apply_cz(q0, op.qubits[1].0),
        GateKind::Swap => state.apply_swap(q0, op.qubits[1].0),
        GateKind::CRz => state.apply_crz(q0, op.qubits[1].0, -resolved[0]),
        GateKind::CU3 => apply_2q_dagger_via_matrix(
            state,
            q0,
            op.qubits[1].0,
            &perm_2q_to_cuda(&gates::cu3(resolved[0], resolved[1], resolved[2])),
        ),

        GateKind::CCX => state.apply_ccx(q0, op.qubits[1].0, op.qubits[2].0),
        GateKind::CSwap => state.apply_cswap(q0, op.qubits[1].0, op.qubits[2].0),

        GateKind::Reset
        | GateKind::Measure
        | GateKind::PhaseShifter
        | GateKind::BeamSplitterRx
        | GateKind::Rbs
        | GateKind::Custom(_) => {
            return Err(OmegaError::Unsupported(format!(
                "cuda adjoint dagger: unsupported gate {:?}",
                op.gate
            )));
        }
    };
    res.map_err(OmegaError::from)
}

fn apply_op_derivative(
    state: &mut CudaState,
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
    let res: Result<(), crate::CudaError> = match (&op.gate, param_idx) {
        (GateKind::Rx, 0) => state.apply_1q(q0, &gates::drx(resolved[0])),
        (GateKind::Ry, 0) => state.apply_1q(q0, &gates::dry(resolved[0])),
        (GateKind::Rz, 0) => state.apply_1q(q0, &gates::drz(resolved[0])),
        (GateKind::U1, 0) => state.apply_1q(q0, &gates::du1_dl(resolved[0])),
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
        (GateKind::CRz, 0) => state.apply_2q(
            q0,
            op.qubits[1].0,
            &perm_2q_to_cuda(&gates::dcrz(resolved[0])),
        ),
        (GateKind::CU3, 0) => state.apply_2q(
            q0,
            op.qubits[1].0,
            &perm_2q_to_cuda(&gates::dcu3_dt(resolved[0], resolved[1], resolved[2])),
        ),
        (GateKind::CU3, 1) => state.apply_2q(
            q0,
            op.qubits[1].0,
            &perm_2q_to_cuda(&gates::dcu3_dp(resolved[0], resolved[1], resolved[2])),
        ),
        (GateKind::CU3, 2) => state.apply_2q(
            q0,
            op.qubits[1].0,
            &perm_2q_to_cuda(&gates::dcu3_dl(resolved[0], resolved[1], resolved[2])),
        ),
        _ => {
            return Err(OmegaError::Unsupported(format!(
                "cuda adjoint: no derivative for {:?} param_idx={param_idx}",
                op.gate
            )));
        }
    };
    res.map_err(OmegaError::from)
}

fn apply_1q_dagger_via_matrix(
    state: &mut CudaState,
    qubit: u32,
    u: &[Complex64; 4],
) -> Result<(), crate::CudaError> {
    let ud = [u[0].conj(), u[2].conj(), u[1].conj(), u[3].conj()];
    state.apply_1q(qubit, &ud)
}

fn apply_2q_dagger_via_matrix(
    state: &mut CudaState,
    qa: u32,
    qb: u32,
    u: &[Complex64; 16],
) -> Result<(), crate::CudaError> {
    let mut ud = [Complex64::new(0.0, 0.0); 16];
    for r in 0..4 {
        for c in 0..4 {
            ud[r * 4 + c] = u[c * 4 + r].conj();
        }
    }
    state.apply_2q(qa, qb, &ud)
}

/// CPU `gates::Gate2Q` uses |q_first q_second⟩ ordering — q_first is
/// the *high* row/col bit. The CUDA `apply_2q(qa, qb, u)` follows the
/// Metal convention `row = bit_qb*2 + bit_qa` (qb high, qa low). Same
/// permutation σ that swaps the high/low bits as Metal — see the
/// Metal adjoint's `perm_2q_to_metal` for the full derivation.
fn perm_2q_to_cuda(g: &gates::Gate2Q) -> [Complex64; 16] {
    let perm = [0usize, 2, 1, 3];
    let mut out = [Complex64::new(0.0, 0.0); 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = g[perm[r] * 4 + perm[c]];
        }
    }
    out
}

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
        out[i] += phase * amp;
    }
    out
}

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
    fn x_component_falls_back_to_host_path() {
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)]), (1.0, vec![(1, PauliOp::X)])],
        };
        assert!(diagonal_pauli_terms(&obs).is_none());
    }

    #[test]
    fn empty_observable_classifies_as_diagonal() {
        let obs = Observable { terms: vec![] };
        let terms = diagonal_pauli_terms(&obs).expect("empty must classify");
        assert!(terms.is_empty());
    }
}
