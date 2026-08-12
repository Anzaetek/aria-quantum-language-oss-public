//! `Backend::execute` glue for the OpenCL statevector backend.
//!
//! Walks the circuit gate-by-gate, dispatches each unconditional /
//! satisfied op through the matching `apply_1q` / `apply_2q` kernel,
//! and either reads back the full statevector or samples shot counts
//! at the end. Mirrors the Metal / CUDA `execute` paths but skips
//! their fused-diagonal walker and their lease-pooled buffers — those
//! land in follow-up commits once the kernel surface grows beyond
//! the basic 1q + 2q pair.
//!
//! Mid-circuit measurement, `Reset`, and conditional gates whose
//! creg has been written by a prior `Measure` aren't supported yet.
//! The dispatcher rejects circuits that need them so the caller's
//! `--device opencl` selection falls back to CPU deterministically
//! rather than silently producing wrong outputs.

use num_complex::Complex64;
use omega_backend_statevector::gates;
use omega_core::circuit::{CircuitIR, GateKind, GateOp};
use omega_core::error::{OmegaError, Result as OmegaResult};
use omega_core::executor::{ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use omega_core::params::ParameterBinding;

use crate::imp::{DeviceHandle, StateBuffer};

/// Run a circuit on the OpenCL backend. Mirrors
/// `MetalStatevectorBackend::execute` minus the mid-circuit
/// measurement / reset / fusion paths (still TODO on the OpenCL side).
pub(crate) fn run(
    handle: &DeviceHandle,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    config: &ExecConfig,
) -> OmegaResult<ExecResult> {
    let n = circuit.num_qubits;
    let mut state = handle.allocate(n)?;

    // Up-front rejection for the gates the OpenCL backend doesn't
    // support yet. Letting them fall through to `apply_op` would
    // surface a less obvious error mid-walk; cleaner to refuse
    // up-front and let the caller fall back to CPU.
    for op in &circuit.ops {
        match &op.gate {
            GateKind::Reset => {
                return Err(OmegaError::Unsupported(
                    "opencl: Reset is non-unitary; not yet implemented".into(),
                ));
            }
            GateKind::Measure if config.mid_circuit_mode == MidCircuitMode::Collapse => {
                return Err(OmegaError::Unsupported(
                    "opencl: mid-circuit measurement not yet implemented".into(),
                ));
            }
            _ => {}
        }
    }

    // No mid-circuit measurement → classical bits stay zero throughout.
    let classical_bits = vec![0u8; circuit.num_classical_bits as usize];
    apply_ops_fused(&mut state, &circuit.ops, params, |op| {
        !op.condition_satisfied(&classical_bits)
    })?;

    match config.shots {
        None => Ok(ExecResult::Statevector(state.read_state())),
        Some(shots) => {
            // GPU-resident sampler — Philox4×32 + Hillis-Steele scan +
            // CDF binary search. Removes the full statevector
            // host-roundtrip the prior `sample_counts` did (`2·dim·f32`
            // bytes); only the per-shot outcome buffer crosses to the
            // host.
            //
            // Seed handling matches Metal's: an explicit
            // `ExecConfig::seed` is forwarded verbatim; absent it we
            // synthesise a fresh u64 from a host RNG so successive
            // shot-runs aren't bit-identical. The CPU `execute` path
            // uses the same construction (`rand::make_rng()`), so
            // tests that snapshot one backend's shot histogram and
            // compare to another can use a fixed seed.
            let resolved_seed = match config.seed {
                Some(s) => s,
                None => {
                    use rand::rngs::StdRng;
                    use rand::RngExt;
                    let mut host_rng: StdRng = rand::make_rng::<StdRng>();
                    host_rng.random()
                }
            };
            let counts = state
                .sample_shots_gpu(shots, resolved_seed)
                .map_err(OmegaError::from)?;
            Ok(ExecResult::Counts(counts))
        }
    }
}

/// Apply one circuit op to the device state. Single-qubit gates
/// resolve to `apply_1q`; two-qubit gates build the row-major matrix
/// via `gates::*`, swap rows/cols to the (qa low, qb high) Metal /
/// CUDA / OpenCL convention via `perm_to_kernel`, and dispatch
/// `apply_2q`. Three-qubit gates (CCX, CSwap) and photonic / custom
/// kinds are rejected — same gates the Metal backend used to reject
/// before its 1q+2q decomposition shipped; they'll land here too
/// once the basic forward path stabilises.
fn apply_op(state: &mut StateBuffer, op: &GateOp, params: &ParameterBinding) -> OmegaResult<()> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<OmegaResult<Vec<_>>>()?;

    let q0 = || op.qubits[0].0;
    let q1 = || op.qubits[1].0;

    match &op.gate {
        // No-ops at the kernel level
        GateKind::Id | GateKind::Barrier => Ok(()),
        GateKind::Measure => Ok(()), // skip mode — already filtered above

        // Single-qubit gates — diagonal-in-CB ones (Z, S, Sdg, T, Tdg,
        // Rz, U1) dispatch through the `apply_diagonal` fast path
        // (half the per-amplitude memory traffic + skips the 2x2
        // matvec on off-diagonal zeros). The rest go through `apply_1q`.
        GateKind::H => apply_1q_from_matrix(state, q0(), &gates::h()),
        GateKind::X => apply_1q_from_matrix(state, q0(), &gates::x()),
        GateKind::Y => apply_1q_from_matrix(state, q0(), &gates::y()),
        GateKind::Z => state
            .apply_diagonal(q0(), Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0))
            .map_err(OmegaError::from),
        GateKind::S => state
            .apply_diagonal(q0(), Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0))
            .map_err(OmegaError::from),
        GateKind::Sdg => state
            .apply_diagonal(q0(), Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0))
            .map_err(OmegaError::from),
        // NOT diagonal, so these cannot use the apply_diagonal fast path that
        // S/Sdg/T/Tdg take — they need the general 1q kernel with the exact
        // matrix (routing via u3 would add the e^{iπ/4} global phase).
        GateKind::Sx => apply_1q_from_matrix(state, q0(), &gates::sx()),
        GateKind::Sxdg => apply_1q_from_matrix(state, q0(), &gates::sxdg()),
        GateKind::T => state
            .apply_diagonal(
                q0(),
                Complex64::new(1.0, 0.0),
                Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
            )
            .map_err(OmegaError::from),
        GateKind::Tdg => state
            .apply_diagonal(
                q0(),
                Complex64::new(1.0, 0.0),
                Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4),
            )
            .map_err(OmegaError::from),
        GateKind::Rx => apply_1q_from_matrix(state, q0(), &gates::rx(resolved[0])),
        GateKind::Ry => apply_1q_from_matrix(state, q0(), &gates::ry(resolved[0])),
        GateKind::Rz => state
            .apply_diagonal(
                q0(),
                Complex64::from_polar(1.0, -resolved[0] / 2.0),
                Complex64::from_polar(1.0, resolved[0] / 2.0),
            )
            .map_err(OmegaError::from),
        GateKind::U1 => state
            .apply_diagonal(
                q0(),
                Complex64::new(1.0, 0.0),
                Complex64::from_polar(1.0, resolved[0]),
            )
            .map_err(OmegaError::from),
        GateKind::U2 => apply_1q_from_matrix(state, q0(), &gates::u2(resolved[0], resolved[1])),
        GateKind::U3 => apply_1q_from_matrix(
            state,
            q0(),
            &gates::u3(resolved[0], resolved[1], resolved[2]),
        ),

        // Two-qubit gates — diagonal-in-CB ones (CZ, CRz) go through
        // `apply_diagonal_2q` for the same reason as the 1q case. The
        // generic dense 4x4 ones feed through `apply_2q` with the
        // convention swap (CPU is qa-high / OpenCL is qa-low) handled
        // by `perm_to_kernel`.
        GateKind::CX => apply_2q_from_matrix(state, q0(), q1(), &gates::cx()),
        GateKind::CY => apply_2q_from_matrix(state, q0(), q1(), &gates::cy()),
        GateKind::CZ => {
            // diag(1, 1, 1, -1) under (bit_qb, bit_qa) ordering — only
            // the (1,1) entry picks up the minus.
            let one = Complex64::new(1.0, 0.0);
            state
                .apply_diagonal_2q(q0(), q1(), one, one, one, Complex64::new(-1.0, 0.0))
                .map_err(OmegaError::from)
        }
        GateKind::Swap => apply_2q_from_matrix(state, q0(), q1(), &gates::swap()),
        GateKind::CRz => {
            // CRz(θ) on (qc=q0, qt=q1) — diagonal in CB. Mirror Metal's
            // `apply_crz_diagonal` byte-for-byte: pass (qa=qc, qb=qt,
            // d00, d01, d10, d11) = (control, target, 1, e^{-iθ/2}, 1,
            // e^{iθ/2}). The kernel indexes d by idx = bit_qb*2 +
            // bit_qa = bit_target*2 + bit_control, so d01 picks up
            // (target=0, control=1) → e^{-iθ/2} and d11 picks up
            // (target=1, control=1) → e^{iθ/2}, leaving the
            // control=0 amplitudes untouched.
            let theta = resolved[0];
            let one = Complex64::new(1.0, 0.0);
            let phm = Complex64::from_polar(1.0, -theta / 2.0);
            let php = Complex64::from_polar(1.0, theta / 2.0);
            state
                .apply_diagonal_2q(q0(), q1(), one, phm, one, php)
                .map_err(OmegaError::from)
        }
        GateKind::CU3 => apply_2q_from_matrix(
            state,
            q0(),
            q1(),
            &gates::cu3(resolved[0], resolved[1], resolved[2]),
        ),

        // Three-qubit + photonic + custom — defer to a follow-up
        // commit.
        GateKind::CCX | GateKind::CSwap => Err(OmegaError::Unsupported(format!(
            "opencl-statevector: 3q gate {:?} not yet decomposed",
            op.gate
        ))),
        GateKind::Reset => Err(OmegaError::Unsupported(
            "opencl: Reset is non-unitary; should have been filtered before apply_op".into(),
        )),
        // Photonic / RBS / custom — no OpenCL kernel. RBS runs natively on the
        // CUDA and Metal statevector backends, but not here; per LIMITATIONS.md
        // this backend surfaces a clean "unsupported gate" error so the CLI
        // dispatcher falls back to the CPU statevector backend (which supports
        // RBS and its Givens parameter-shift rule).
        GateKind::Rbs | GateKind::PhaseShifter | GateKind::BeamSplitterRx | GateKind::Custom(_) => {
            Err(OmegaError::Unsupported(format!(
                "opencl-statevector: gate {:?} is not supported on this backend",
                op.gate
            )))
        }
    }
}

fn apply_1q_from_matrix(state: &mut StateBuffer, qubit: u32, g: &gates::Gate1Q) -> OmegaResult<()> {
    state
        .apply_1q(qubit, g[0], g[1], g[2], g[3])
        .map_err(OmegaError::from)
}

fn apply_2q_from_matrix(
    state: &mut StateBuffer,
    qa: u32,
    qb: u32,
    g: &gates::Gate2Q,
) -> OmegaResult<()> {
    let permuted = perm_to_kernel(g);
    state.apply_2q(qa, qb, &permuted).map_err(OmegaError::from)
}

/// Convert a CPU-convention 4x4 gate matrix (q_first*2 + q_second
/// row indexing — first qubit is high) to the kernel convention
/// (qb*2 + qa — qb high, qa low). Same permutation Metal applies via
/// `perm_2q_to_metal`. Direct port; permutation `[0, 2, 1, 3]` swaps
/// the bit pair in row + column indices simultaneously.
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

/// Run a single forward sweep, then evaluate one observable on the
/// resulting device-resident statevector via the GPU
/// `pauli_expectation` reduction kernel. Sums `O = Σ cₖ Pₖ` term by
/// term, dispatching one kernel per Pauli string. Removes the
/// full-statevector host roundtrip the prior host-loop sampler did.
pub(crate) fn expectation(
    handle: &DeviceHandle,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
) -> OmegaResult<f64> {
    let state = run_to_state_buffer(handle, circuit, params)?;
    let mut total = 0.0_f64;
    for (coeff, pauli_string) in &observable.terms {
        let (x_mask, sign_mask, y_factor) = pauli_masks(pauli_string);
        let z = state
            .pauli_expectation(x_mask, sign_mask, y_factor)
            .map_err(OmegaError::from)?;
        total += coeff * z.re;
    }
    Ok(total)
}

/// Same forward sweep, multiple observables. The Backend trait's
/// default loops `expectation`, which would re-run the forward sweep
/// per observable. Mirror the Metal / CPU optimisation: one forward
/// sweep, fan out per Pauli string via the GPU reduction kernel.
pub(crate) fn expectation_multi(
    handle: &DeviceHandle,
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observables: &[Observable],
) -> OmegaResult<Vec<f64>> {
    if observables.is_empty() {
        return Ok(Vec::new());
    }
    let state = run_to_state_buffer(handle, circuit, params)?;
    let mut out = Vec::with_capacity(observables.len());
    for obs in observables {
        let mut total = 0.0_f64;
        for (coeff, pauli_string) in &obs.terms {
            let (x_mask, sign_mask, y_factor) = pauli_masks(pauli_string);
            let z = state
                .pauli_expectation(x_mask, sign_mask, y_factor)
                .map_err(OmegaError::from)?;
            total += coeff * z.re;
        }
        out.push(total);
    }
    Ok(out)
}

/// Run the forward sweep and return the device-resident
/// `StateBuffer`. Same shape as `run`'s statevector branch but skips
/// the device→host `read_state` roundtrip so expectation kernels can
/// read the state directly. Mid-circuit measurement / Reset still
/// rejected (the OpenCL backend doesn't support them yet) before
/// any work is enqueued.
fn run_to_state_buffer(
    handle: &DeviceHandle,
    circuit: &CircuitIR,
    params: &ParameterBinding,
) -> OmegaResult<crate::imp::StateBuffer> {
    for op in &circuit.ops {
        match &op.gate {
            GateKind::Reset => {
                return Err(OmegaError::Unsupported(
                    "opencl: Reset is non-unitary; not yet implemented".into(),
                ));
            }
            GateKind::Measure => {
                return Err(OmegaError::Unsupported(
                    "opencl: mid-circuit measurement not yet implemented".into(),
                ));
            }
            _ => {}
        }
    }
    let mut state = handle.allocate(circuit.num_qubits)?;
    let classical_bits = vec![0u8; circuit.num_classical_bits as usize];
    apply_ops_fused(&mut state, &circuit.ops, params, |op| {
        !op.condition_satisfied(&classical_bits)
    })?;
    Ok(state)
}

/// Build the (`x_mask`, `sign_mask`, `y_factor`) triple the
/// `pauli_expectation` kernel consumes from a Pauli string.
///
/// * `x_mask` has bit `q` set if qubit `q` carries X or Y (those
///   flip the basis bit). The kernel uses `j = i XOR x_mask`.
/// * `sign_mask` has bit `q` set if qubit `q` carries Y or Z (each
///   contributes `(-1)^bit_q(i)` to the phase). The kernel uses
///   `(-1)^popcount(i & sign_mask)`.
/// * `y_factor = (-i)^{|Y|}` — the global Y-count prefactor folded in
///   inside the kernel before the reduction sums. It is `(-i)`, not
///   `(+i)`, per Y: the kernel forms the matrix element `P[i, i^x]`,
///   which for a Y qubit is `(-i)·(-1)^bit_i` (`Y|0⟩ = i|1⟩`,
///   `Y|1⟩ = -i|0⟩`), and the `(-1)^bit` half is in `sign_mask`.
///
/// I qubits are no-ops. Direct port of Metal's `pauli_masks`.
pub fn pauli_masks(pauli_string: &[(u32, PauliOp)]) -> (u32, u32, Complex64) {
    let mut x_mask: u32 = 0;
    let mut sign_mask: u32 = 0;
    let mut y_count: u32 = 0;
    for (q, p) in pauli_string {
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

/// Classify `op` as a fusion-eligible diagonal 1q gate. Returns
/// `Some((qubit, d0, d1))` when the gate is unconditional and
/// diagonal in the computational basis (Z, S/Sdg, T/Tdg, Rz, U1);
/// `None` for anything that needs the full `apply_op` path. Direct
/// port of Metal's `diagonal_factor` — same gates classified the
/// same way so OpenCL and Metal collapse the same fusion runs.
///
/// Conditional gates (`op.condition.is_some()`) are always rejected
/// even when the gate itself is diagonal: fusion cannot move across
/// a classical predicate without losing the per-op skip semantics.
/// Id / Barrier / Measure aren't classified as factors either — they
/// are no-ops at the kernel level and emitting an identity factor
/// would just waste a slot in the per-amp shader loop.
fn diagonal_factor(
    op: &GateOp,
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
/// fusion-eligible diagonal gates into a single
/// `apply_diagonal_product` dispatch. Non-diagonal gates flush the
/// pending factor list and fall through to the per-op `apply_op`
/// path.
///
/// `condition_skip` decides whether each op should be skipped
/// (mirrors the caller's `if !condition_satisfied { continue; }`
/// loop). Skipped ops bypass the walker without disturbing the
/// pending fusion run — same shape Metal uses.
///
/// Saves N-1 dispatches per N-long fusion run. The HEA bench's
/// second layer (8 Rz on disjoint qubits) collapses from 8
/// dispatches to 1; QAOA Rz fans batch the same way.
pub(crate) fn apply_ops_fused(
    state: &mut crate::imp::StateBuffer,
    ops: &[GateOp],
    params: &ParameterBinding,
    mut condition_skip: impl FnMut(&GateOp) -> bool,
) -> OmegaResult<()> {
    let mut pending: Vec<(u32, Complex64, Complex64)> = Vec::new();

    fn flush(
        state: &mut crate::imp::StateBuffer,
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

    for op in ops {
        if condition_skip(op) {
            continue;
        }
        // Id and Barrier are kernel-level no-ops and shouldn't break
        // an in-flight fusion run. Skip without flushing so
        // `Rz; Id; Rz; Rz` still collapses to one fused dispatch —
        // common after compiler-inserted barriers between optimisation
        // passes. Matches Metal's walker exactly.
        if matches!(&op.gate, GateKind::Id | GateKind::Barrier) {
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
}
