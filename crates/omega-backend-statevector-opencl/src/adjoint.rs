//! Adjoint differentiation on OpenCL.
//!
//! Jones-style adjoint AD, byte-compatible with the Metal / CPU
//! implementations:
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
//! Forward sweep runs through the fusion walker (`apply_ops_fused`)
//! so consecutive diagonal gates (Z / S / Sdg / T / Tdg / Rz / U1)
//! collapse into one `apply_diagonal_product` dispatch. Backward
//! daggers reuse the same gate kernels (the dagger of a Rx/Ry/Rz/U1
//! is the same kernel with a negated angle; S↔Sdg, T↔Tdg are
//! partner ops; H/X/Y/Z/CX/CY/CZ/Swap/CCX/CSwap are involutions).
//! Derivative applies use `copy_into(phi → temp)` + in-place
//! derivative on `temp` — first-cut shape; the per-gate `_into`
//! variants Metal evolved later can land in a follow-up. Inner
//! product ships partials to host through the GPU `inner_product`
//! kernel (no full statevector roundtrip).
//!
//! Observable application `O|ψ⟩` takes the host read / apply / write
//! path. Metal's `apply_diagonal_pauli_sum` device kernel that
//! handles the Z-only QML case stays unported in this slice — a
//! dedicated kernel + classifier can land in a follow-up; the host
//! path is correct, just slower on QML training. The cross-CPU
//! parity test pins correctness; the QML perf gap closes with the
//! follow-up.
//!
//! Non-unitary ops (Reset / Measure) cause early `Ok(None)` so the
//! caller (`QmlTrainer`) falls back to parameter-shift — same shape
//! as the CPU contract.

use std::collections::HashMap;

use num_complex::Complex64;

use omega_backend_statevector::gates;
use omega_core::circuit::{CircuitIR, GateKind, GateOp, ParamExpr, SymbolId};
use omega_core::error::{OmegaError, Result as OmegaResult};
use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;

use crate::imp::{BufferPool, DeviceHandle, StateBuffer};
use crate::OpenClError;

pub(crate) fn adjoint_gradient(
    handle: &DeviceHandle,
    pool: &BufferPool,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
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

    // Lease the three working buffers (|φ⟩, |ν⟩, temp) from the
    // backend's pool. After the first adjoint call the pool steady-
    // states at three buffers per qubit count, so subsequent calls
    // are allocation-free at the OpenCL buffer level. Buffers are
    // reset to `|0…0⟩` on lease (device-side `clEnqueueFillBuffer`
    // + a single 4-byte poke) — much cheaper than the prior
    // per-call `clCreateBuffer + copy_host_ptr` shape.
    let mut phi = pool.lease(handle, n)?;
    let mut nu = pool.lease(handle, n)?;
    let mut temp = pool.lease(handle, n)?;

    // Forward sweep — `|φ⟩ ← U_M·…·U_1 |0⟩`. The unitary_ops slice is
    // already filtered, so the predicate is a const `false` skip.
    let unitary_owned: Vec<GateOp> = unitary_ops.iter().map(|&op| op.clone()).collect();
    crate::execute::apply_ops_fused(&mut phi, &unitary_owned, params, |_| false)?;

    // Initialise ν = O|ψ⟩ via the host path. For the QML trainer's
    // diagonal-only observable case Metal has a fused
    // `apply_diagonal_pauli_sum` kernel that skips this roundtrip;
    // landing that on OpenCL is a follow-up — for now the host pass
    // is correct (cross-CPU parity test pins it) at the cost of one
    // 2·dim·f32 byte read.
    let psi_host = phi.read_state();
    let nu_host = apply_observable_host(&psi_host, n, observable);
    nu.write_state(&nu_host).map_err(OmegaError::from)?;

    // Backward sweep.
    let mut gradients: HashMap<SymbolId, f64> = HashMap::new();
    for op in unitary_ops.iter().rev() {
        // |φ⟩ ← U_k† |φ⟩
        apply_op_dagger(&mut phi, op, params)?;

        for (param_idx, param_expr) in op.params.iter().enumerate() {
            let syms = collect_symbols(param_expr);
            for sym_id in syms {
                let chain = params.resolve_derivative(param_expr, sym_id)?;
                if chain.abs() < 1e-30 {
                    continue;
                }
                // temp ← (∂U_k/∂θ_p) |φ⟩
                phi.copy_into(&temp).map_err(OmegaError::from)?;
                apply_op_derivative_inplace(&mut temp, op, params, param_idx)?;
                let ip = nu.inner_product(&temp).map_err(OmegaError::from)?;
                *gradients.entry(sym_id).or_insert(0.0) += 2.0 * ip.re * chain;
            }
        }

        // |ν⟩ ← U_k† |ν⟩
        apply_op_dagger(&mut nu, op, params)?;
    }

    // Return all three buffers to the pool — next call's lease
    // reuses them with a `reset_to_zero` instead of paying a fresh
    // `clCreateBuffer`. Read state is irrelevant; we never touch
    // these buffers again after return.
    pool.return_buffer(phi);
    pool.return_buffer(nu);
    pool.return_buffer(temp);

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

/// Collect every free symbol that appears in a `ParamExpr`. Mirrors
/// Metal's `collect_symbols` and the CPU version exactly.
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

/// Apply U_k† to a `StateBuffer` in place. Mirrors Metal's
/// `apply_op_dagger` byte-for-byte: involutions (H, X, Y, Z, CX, CY,
/// CZ, Swap, CCX, CSwap) re-apply themselves; pairs (S↔Sdg, T↔Tdg)
/// swap; parameterised Rx/Ry/Rz/U1/CRz negate the angle; U2/U3/CU3
/// build the matrix and conjugate-transpose.
fn apply_op_dagger(
    state: &mut StateBuffer,
    op: &GateOp,
    params: &ParameterBinding,
) -> OmegaResult<()> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;
    let q0 = op.qubits[0].0;

    let res: Result<(), OpenClError> = match &op.gate {
        // Self-adjoint involutions
        GateKind::H => state.apply_1q(
            q0,
            gates::h()[0],
            gates::h()[1],
            gates::h()[2],
            gates::h()[3],
        ),
        GateKind::X => state.apply_1q(
            q0,
            gates::x()[0],
            gates::x()[1],
            gates::x()[2],
            gates::x()[3],
        ),
        GateKind::Y => state.apply_1q(
            q0,
            gates::y()[0],
            gates::y()[1],
            gates::y()[2],
            gates::y()[3],
        ),
        GateKind::Z => {
            state.apply_diagonal(q0, Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0))
        }
        GateKind::Id | GateKind::Barrier => Ok(()),

        // Pairs
        GateKind::S => {
            state.apply_diagonal(q0, Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0))
        }
        GateKind::Sdg => {
            state.apply_diagonal(q0, Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0))
        }
        GateKind::T => state.apply_diagonal(
            q0,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4),
        ),
        GateKind::Tdg => state.apply_diagonal(
            q0,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
        ),

        // Parameterised 1q with negated-angle dagger
        GateKind::Rx => {
            let g = gates::rx(-resolved[0]);
            state.apply_1q(q0, g[0], g[1], g[2], g[3])
        }
        GateKind::Ry => {
            let g = gates::ry(-resolved[0]);
            state.apply_1q(q0, g[0], g[1], g[2], g[3])
        }
        GateKind::Rz => state.apply_diagonal(
            q0,
            Complex64::from_polar(1.0, resolved[0] / 2.0),
            Complex64::from_polar(1.0, -resolved[0] / 2.0),
        ),
        GateKind::U1 => state.apply_diagonal(
            q0,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -resolved[0]),
        ),

        // Parameterised 1q via matrix conj-transpose
        GateKind::U2 => {
            let g = gates::u2(resolved[0], resolved[1]);
            let gd = [g[0].conj(), g[2].conj(), g[1].conj(), g[3].conj()];
            state.apply_1q(q0, gd[0], gd[1], gd[2], gd[3])
        }
        GateKind::U3 => {
            let g = gates::u3(resolved[0], resolved[1], resolved[2]);
            let gd = [g[0].conj(), g[2].conj(), g[1].conj(), g[3].conj()];
            state.apply_1q(q0, gd[0], gd[1], gd[2], gd[3])
        }

        // 2q gates — involutions and parameterised
        GateKind::CX => {
            let q1 = op.qubits[1].0;
            let g = perm_to_kernel(&gates::cx());
            state.apply_2q(q0, q1, &g)
        }
        GateKind::CY => {
            let q1 = op.qubits[1].0;
            let g = perm_to_kernel(&gates::cy());
            state.apply_2q(q0, q1, &g)
        }
        GateKind::CZ => {
            // CZ is diagonal and self-adjoint.
            let q1 = op.qubits[1].0;
            let one = Complex64::new(1.0, 0.0);
            state.apply_diagonal_2q(q0, q1, one, one, one, Complex64::new(-1.0, 0.0))
        }
        GateKind::Swap => {
            let q1 = op.qubits[1].0;
            let g = perm_to_kernel(&gates::swap());
            state.apply_2q(q0, q1, &g)
        }
        GateKind::CRz => {
            // CRz(-θ): swap d01↔d11's phase signs.
            let q1 = op.qubits[1].0;
            let theta = -resolved[0];
            let one = Complex64::new(1.0, 0.0);
            let phm = Complex64::from_polar(1.0, -theta / 2.0);
            let php = Complex64::from_polar(1.0, theta / 2.0);
            state.apply_diagonal_2q(q0, q1, one, phm, one, php)
        }
        GateKind::CU3 => {
            // CU3† via conjugate-transpose of the full 4x4.
            let q1 = op.qubits[1].0;
            let g = gates::cu3(resolved[0], resolved[1], resolved[2]);
            let mut gd = [Complex64::new(0.0, 0.0); 16];
            for r in 0..4 {
                for c in 0..4 {
                    gd[r * 4 + c] = g[c * 4 + r].conj();
                }
            }
            // gd is in CPU q_first-high convention; permute to kernel.
            let permuted = perm_to_kernel_from_complex(&gd);
            state.apply_2q(q0, q1, &permuted)
        }

        // 3q gates — not yet supported on the OpenCL backend.
        GateKind::CCX | GateKind::CSwap => {
            return Err(OmegaError::Unsupported(format!(
                "opencl adjoint dagger: 3q gate {:?} not yet decomposed",
                op.gate
            )));
        }

        // Photonic / RBS / custom / non-unitary rejected upstream. (RBS has a
        // GPU adjoint on CUDA and Metal but no OpenCL kernel — see the
        // matching arm in `execute::apply_op`.)
        GateKind::Reset
        | GateKind::Measure
        | GateKind::Rbs
        | GateKind::PhaseShifter
        | GateKind::BeamSplitterRx
        | GateKind::Custom(_) => {
            return Err(OmegaError::Unsupported(format!(
                "opencl adjoint dagger: unsupported gate {:?}",
                op.gate
            )));
        }
    };
    res.map_err(OmegaError::from)
}

/// Apply (∂U_k / ∂θ_p) to a `StateBuffer` in place. The caller is
/// responsible for having already memcpy'd the source state into
/// this buffer (`phi.copy_into(temp)`). First-cut: in-place
/// derivative on the scratch buffer. Metal's `_into` variants are a
/// follow-up; the dispatch counts are equivalent — one extra
/// `copy_into` per param vs Metal's batched copy-then-derivative.
fn apply_op_derivative_inplace(
    state: &mut StateBuffer,
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

    let res: Result<(), OpenClError> = match (&op.gate, param_idx) {
        // 1q derivatives. Rx/Ry are not diagonal; route through
        // apply_1q. Rz/U1 are diagonal — route through apply_diagonal
        // for the half-traffic fast path (mirrors Metal).
        (GateKind::Rx, 0) => {
            let g = gates::drx(resolved[0]);
            state.apply_1q(q0, g[0], g[1], g[2], g[3])
        }
        (GateKind::Ry, 0) => {
            let g = gates::dry(resolved[0]);
            state.apply_1q(q0, g[0], g[1], g[2], g[3])
        }
        (GateKind::Rz, 0) => {
            // dRz/dθ = (1/2) diag(-i e^{-iθ/2}, i e^{iθ/2})
            let theta = resolved[0];
            let d0 = Complex64::new(0.0, -0.5) * Complex64::from_polar(1.0, -theta / 2.0);
            let d1 = Complex64::new(0.0, 0.5) * Complex64::from_polar(1.0, theta / 2.0);
            state.apply_diagonal(q0, d0, d1)
        }
        (GateKind::U1, 0) => {
            // dU1/dλ = diag(0, i e^{iλ})
            let lambda = resolved[0];
            let d0 = Complex64::new(0.0, 0.0);
            let d1 = Complex64::new(0.0, 1.0) * Complex64::from_polar(1.0, lambda);
            state.apply_diagonal(q0, d0, d1)
        }
        (GateKind::U2, 0) => {
            let g = gates::du2_dp(resolved[0], resolved[1]);
            state.apply_1q(q0, g[0], g[1], g[2], g[3])
        }
        (GateKind::U2, 1) => {
            let g = gates::du2_dl(resolved[0], resolved[1]);
            state.apply_1q(q0, g[0], g[1], g[2], g[3])
        }
        (GateKind::U3, 0) => {
            let g = gates::du3_dt(resolved[0], resolved[1], resolved[2]);
            state.apply_1q(q0, g[0], g[1], g[2], g[3])
        }
        (GateKind::U3, 1) => {
            let g = gates::du3_dp(resolved[0], resolved[1], resolved[2]);
            state.apply_1q(q0, g[0], g[1], g[2], g[3])
        }
        (GateKind::U3, 2) => {
            let g = gates::du3_dl(resolved[0], resolved[1], resolved[2]);
            state.apply_1q(q0, g[0], g[1], g[2], g[3])
        }
        // dCRz/dθ — diagonal in CB. (1/2) diag(0, -i e^{-iθ/2}, 0, i e^{iθ/2}).
        (GateKind::CRz, 0) => {
            let q1 = op.qubits[1].0;
            let theta = resolved[0];
            let z = Complex64::new(0.0, 0.0);
            let d01 = Complex64::new(0.0, -0.5) * Complex64::from_polar(1.0, -theta / 2.0);
            let d11 = Complex64::new(0.0, 0.5) * Complex64::from_polar(1.0, theta / 2.0);
            state.apply_diagonal_2q(q0, q1, z, d01, z, d11)
        }
        (GateKind::CU3, 0) => {
            let q1 = op.qubits[1].0;
            let g = perm_to_kernel(&gates::dcu3_dt(resolved[0], resolved[1], resolved[2]));
            state.apply_2q(q0, q1, &g)
        }
        (GateKind::CU3, 1) => {
            let q1 = op.qubits[1].0;
            let g = perm_to_kernel(&gates::dcu3_dp(resolved[0], resolved[1], resolved[2]));
            state.apply_2q(q0, q1, &g)
        }
        (GateKind::CU3, 2) => {
            let q1 = op.qubits[1].0;
            let g = perm_to_kernel(&gates::dcu3_dl(resolved[0], resolved[1], resolved[2]));
            state.apply_2q(q0, q1, &g)
        }
        _ => {
            return Err(OmegaError::Unsupported(format!(
                "opencl adjoint: no derivative for {:?} param_idx={param_idx}",
                op.gate
            )));
        }
    };
    res.map_err(OmegaError::from)
}

/// CPU-convention 4x4 gate matrix → kernel-convention 32-float
/// (qb-high / qa-low) byte layout. Same permutation `execute.rs::
/// perm_to_kernel` uses, but exposed here so the dagger / derivative
/// dispatchers don't need to re-import it.
fn perm_to_kernel(g: &gates::Gate2Q) -> [f32; 32] {
    let perm = [0usize, 2, 1, 3];
    let mut out = [0.0_f32; 32];
    for r in 0..4 {
        for c in 0..4 {
            let src: Complex64 = g[perm[r] * 4 + perm[c]];
            out[2 * (r * 4 + c)] = src.re as f32;
            out[2 * (r * 4 + c) + 1] = src.im as f32;
        }
    }
    out
}

fn perm_to_kernel_from_complex(g: &[Complex64; 16]) -> [f32; 32] {
    let perm = [0usize, 2, 1, 3];
    let mut out = [0.0_f32; 32];
    for r in 0..4 {
        for c in 0..4 {
            let src = g[perm[r] * 4 + perm[c]];
            out[2 * (r * 4 + c)] = src.re as f32;
            out[2 * (r * 4 + c) + 1] = src.im as f32;
        }
    }
    out
}

/// Apply an observable `O = Σ c_k P_k` to a host-side statevector.
/// Direct port of Metal's `apply_observable_host` / the CPU
/// reference. The QML trainer's typical Z-only observable can later
/// route through a dedicated `apply_diagonal_pauli_sum` device
/// kernel; the host path stays correct in the meantime.
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
