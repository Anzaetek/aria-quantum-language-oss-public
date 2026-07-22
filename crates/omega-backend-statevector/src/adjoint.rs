//! Adjoint differentiation for statevector-based gradient computation.
//!
//! Computes all parameter gradients in a single forward + backward pass,
//! giving O(1) cost in the number of parameters (vs O(2p) for parameter-shift).
//!
//! ## Formal verification cross-reference
//!
//! Each step of the algorithm is ratified by a Lean 4 theorem in
//! `verification/Verification/Adjoint/`:
//!
//! * Per-gate matrix derivatives (`apply_gate_derivative` reaches into
//!   `gates::drx` / `gates::dcrz` / `gates::du3_dp` / `gates::du3_dl`):
//!   * Rx → `ParamShiftRx.drx_eq_neg_half_i_x_rx`
//!   * Ry → `ParamShiftRy.dry_eq_neg_half_i_y_ry`
//!   * Rz → `ParamShiftRz.drz_eq_neg_half_i_z_rz`
//!   * U1 → `ParamShiftU1.du1_dlambda_eq_i_u1_p1`
//!   * U2 (φ, λ) → specialisation of `ParamShiftU3` at θ = π/2;
//!     `gates::du2_dp` / `gates::du2_dl` directly call the U3
//!     derivative path, so the U3 theorems apply unchanged.
//!   * CRz → `ParamShiftCRz.dcrz_eq_neg_half_i_m_crz`
//!   * U3 (φ, λ slots) → `ParamShiftU3.du3_dphi_eq_i_p1_u3`,
//!     `ParamShiftU3.du3_dlambda_eq_i_u3_p1`
//!   * U3 (θ slot) → `ParamShiftU3.du3_dtheta_eq_half_u3_shifted`
//!     (parameter-shift form: `dU3/dθ = (1/2)·U3(θ+π)`) plus the
//!     adjoint-step corollary
//!     `ParamShiftU3.u3_theta_adjoint_step_eq_param_shift`. The
//!     "single-generator Hermitian form" matrix identity stays open
//!     as a future refinement; not load-bearing for the
//!     parameter-shift derivation `Adjoint/AdjointEqShift.lean`
//!     will use.
//!   * CU3 (φ, λ, θ slots) → `ParamShiftCU3.dcu3_dphi_eq_i_p11_cu3`,
//!     `ParamShiftCU3.dcu3_dlambda_eq_i_cu3_p11`,
//!     `ParamShiftCU3.dcu3_dtheta_eq_half_q_ctl_cu3_shifted`.
//!     The φ / λ slots use the `P11 = |11⟩⟨11|` projector
//!     (controlled lift of U3's `P₁ = diag(0, 1)`); the θ slot
//!     uses the wider `Q_ctl = diag(0, 0, 1, 1)` projector
//!     because the upper-left identity block of CU3(θ+π) needs
//!     to be zeroed.
//! * Symbolic chain rule (`params.rs::ParamExpr::differentiate`,
//!   called via `resolve_derivative`):
//!   `ChainRule.differentiate_correct` /
//!   `ChainRule.deriv_evaluate_eq`.
//! * Composition through an outer real-valued function — the
//!   gate-angle → expectation chain that `apply_gate_derivative`
//!   materialises by multiplying the gate-derivative inner product
//!   by the chain factor `params.resolve_derivative(expr, sym)`:
//!   `Composition.composition_correct` /
//!   `Composition.deriv_composition_eq`.
//! * Linearity in the operator argument (used by
//!   `apply_observable` + the gradient accumulation across
//!   `observable.terms`):
//!   `Linearity.expVal_linear` / `Linearity.expVal_finset_sum`.
//! * Single-qubit Pauli expectation closed forms (used by every
//!   per-qubit step inside `apply_observable` and `pauli`):
//!   `PauliExpectation.expV_{Z,X,Y,I}` plus the four
//!   `σ?_hermitian` witnesses.
//! * Hermitian-implies-real-expectation (justifies the `2·Re(...)`
//!   projection in the gradient accumulator below — for Hermitian
//!   observables the imaginary part the algorithm discards is
//!   round-off, not signal):
//!   `HermitianReal.expVal_real_of_hermitian` /
//!   `HermitianReal.im_expVal_zero_of_hermitian`.
//! * Conjugation preserves Hermiticity — `U† O U` is Hermitian
//!   whenever `O` is, so the "effective observable" the backward
//!   sweep carries through every step is itself a valid input
//!   to the real-expectation lemma above:
//!   `Conjugation.conjugation_preserves_hermitian` /
//!   `Conjugation.adjoint_state_observable_hermitian`.
//! * Unitary invariance of inner products — `⟨Uψ|Uφ⟩ = ⟨ψ|φ⟩`
//!   when `U†U = I`. The forward sweep doesn't re-normalise
//!   between gates because every standard gate ships as a
//!   unitary by construction; this lemma is the algebraic
//!   warrant for that:
//!   `UnitaryInvariance.inner_invariant_under_unitary` /
//!   `UnitaryInvariance.norm_sq_invariant_under_unitary`.
//!
//! `ci.sh` runs `lake build` against the umbrella
//! `verification/Verification.lean` which transitively imports
//! every file under `Verification/Adjoint/`. Removing or breaking
//! any cited theorem fails CI before the Rust adjoint code can
//! merge.

use std::collections::HashMap;

use num_complex::Complex64;

use omega_core::circuit::*;
use omega_core::error::{OmegaError, Result};
use omega_core::executor::{Observable, PauliOp};
use omega_core::params::ParameterBinding;

use crate::gates;
use crate::sim::{apply_1q, apply_2q};

/// Compute gradients of ⟨ψ(θ)|O|ψ(θ)⟩ via adjoint differentiation.
///
/// Algorithm:
///   Forward:  |ψ₀⟩ → U₁ → |ψ₁⟩ → ... → Uₙ → |ψₙ⟩  (store checkpoints)
///   Backward: |λ⟩ = O|ψₙ⟩, then for i=n..1:
///     grad[k] += 2·Re(⟨λ|dUᵢ/dθₖ|ψᵢ₋₁⟩) · chain_factor
///     |λ⟩ = Uᵢ†|λ⟩
pub fn adjoint_gradient(
    circuit: &CircuitIR,
    params: &ParameterBinding,
    observable: &Observable,
) -> Result<Vec<(SymbolId, f64)>> {
    let n = circuit.num_qubits as usize;
    let dim = 1usize << n;

    // Collect unitary gate indices (skip Measure, Barrier, Reset)
    let unitary_ops: Vec<(usize, &GateOp)> = circuit
        .ops
        .iter()
        .enumerate()
        .filter(|(_, op)| is_unitary(&op.gate))
        .collect();

    // --- Forward pass: store all checkpoints ---
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0);

    let mut checkpoints: Vec<Vec<Complex64>> = Vec::with_capacity(unitary_ops.len() + 1);
    checkpoints.push(state.clone());

    for (_, op) in &unitary_ops {
        apply_gate_forward(&mut state, n, op, params)?;
        checkpoints.push(state.clone());
    }

    // --- Initialize lambda = O|ψₙ⟩ ---
    let mut lambda = apply_observable(&state, n, observable);

    // --- Backward pass ---
    let mut gradients: HashMap<SymbolId, f64> = HashMap::new();

    for (step, (_, op)) in unitary_ops.iter().enumerate().rev() {
        let psi_i = &checkpoints[step]; // state before gate was applied

        // Accumulate gradient for each parametric slot
        for (param_idx, param_expr) in op.params.iter().enumerate() {
            let syms = collect_symbols(param_expr);
            for sym_id in syms {
                let chain = params.resolve_derivative(param_expr, sym_id)?;
                if chain.abs() < 1e-30 {
                    continue;
                }

                let du_psi = apply_gate_derivative(psi_i, n, op, params, param_idx)?;
                let ip = inner_product(&lambda, &du_psi);
                let contribution = 2.0 * ip.re * chain;

                *gradients.entry(sym_id).or_insert(0.0) += contribution;
            }
        }

        // |λ⟩ = Uᵢ† |λ⟩
        apply_adjoint_gate(&mut lambda, n, op, params)?;
    }

    // Return sorted by symbol ID, including zeros for inactive symbols
    let mut result: Vec<(SymbolId, f64)> = circuit
        .symbols
        .keys()
        .map(|&sym| (sym, gradients.get(&sym).copied().unwrap_or(0.0)))
        .collect();
    result.sort_by_key(|(id, _)| *id);
    Ok(result)
}

/// Check if a gate is unitary (differentiable in the adjoint sense).
fn is_unitary(gate: &GateKind) -> bool {
    !matches!(
        gate,
        GateKind::Measure | GateKind::Barrier | GateKind::Reset
    )
}

/// Apply a gate in the forward direction.
fn apply_gate_forward(
    state: &mut [Complex64],
    n: usize,
    op: &GateOp,
    params: &ParameterBinding,
) -> Result<()> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<Result<Vec<_>>>()?;

    let q0 = op.qubits[0].0 as usize;

    match &op.gate {
        // Non-parametric 1Q
        GateKind::H => apply_1q(state, n, q0, &gates::h()),
        GateKind::X => apply_1q(state, n, q0, &gates::x()),
        GateKind::Y => apply_1q(state, n, q0, &gates::y()),
        GateKind::Z => apply_1q(state, n, q0, &gates::z()),
        GateKind::S => apply_1q(state, n, q0, &gates::s()),
        GateKind::Sdg => apply_1q(state, n, q0, &gates::sdg()),
        GateKind::T => apply_1q(state, n, q0, &gates::t()),
        GateKind::Tdg => apply_1q(state, n, q0, &gates::tdg()),
        GateKind::Id => {}

        // Parametric 1Q
        GateKind::Rx => apply_1q(state, n, q0, &gates::rx(resolved[0])),
        GateKind::Ry => apply_1q(state, n, q0, &gates::ry(resolved[0])),
        GateKind::Rz => apply_1q(state, n, q0, &gates::rz(resolved[0])),
        GateKind::U1 => apply_1q(state, n, q0, &gates::u1(resolved[0])),
        GateKind::U2 => apply_1q(state, n, q0, &gates::u2(resolved[0], resolved[1])),
        GateKind::U3 => apply_1q(
            state,
            n,
            q0,
            &gates::u3(resolved[0], resolved[1], resolved[2]),
        ),

        // 2Q gates
        GateKind::CX => apply_2q(state, n, q0, op.qubits[1].0 as usize, &gates::cx()),
        GateKind::CY => apply_2q(state, n, q0, op.qubits[1].0 as usize, &gates::cy()),
        GateKind::CZ => apply_2q(state, n, q0, op.qubits[1].0 as usize, &gates::cz()),
        GateKind::Swap => apply_2q(state, n, q0, op.qubits[1].0 as usize, &gates::swap()),
        GateKind::CRz => apply_2q(
            state,
            n,
            q0,
            op.qubits[1].0 as usize,
            &gates::crz(resolved[0]),
        ),
        GateKind::CU3 => apply_2q(
            state,
            n,
            q0,
            op.qubits[1].0 as usize,
            &gates::cu3(resolved[0], resolved[1], resolved[2]),
        ),
        GateKind::Rbs => apply_2q(
            state,
            n,
            q0,
            op.qubits[1].0 as usize,
            &gates::rbs(resolved[0]),
        ),

        // 3Q gates: decompose (same as main sim)
        GateKind::CCX => {
            let q1 = op.qubits[1].0 as usize;
            let q2 = op.qubits[2].0 as usize;
            apply_ccx_forward(state, n, q0, q1, q2);
        }
        GateKind::CSwap => {
            let q1 = op.qubits[1].0 as usize;
            let q2 = op.qubits[2].0 as usize;
            apply_cswap_forward(state, n, q0, q1, q2);
        }

        _ => {
            return Err(OmegaError::Unsupported(format!(
                "adjoint: unsupported gate {:?}",
                op.gate
            )));
        }
    }

    Ok(())
}

/// Apply dU/d(param_idx) to a state, returning the result (does not modify input).
fn apply_gate_derivative(
    state: &[Complex64],
    n: usize,
    op: &GateOp,
    params: &ParameterBinding,
    param_idx: usize,
) -> Result<Vec<Complex64>> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<Result<Vec<_>>>()?;

    let q0 = op.qubits[0].0 as usize;
    let mut out = state.to_vec();

    match (&op.gate, param_idx) {
        (GateKind::Rx, 0) => apply_1q(&mut out, n, q0, &gates::drx(resolved[0])),
        (GateKind::Ry, 0) => apply_1q(&mut out, n, q0, &gates::dry(resolved[0])),
        (GateKind::Rz, 0) => apply_1q(&mut out, n, q0, &gates::drz(resolved[0])),
        (GateKind::U1, 0) => apply_1q(&mut out, n, q0, &gates::du1_dl(resolved[0])),
        (GateKind::U2, 0) => apply_1q(&mut out, n, q0, &gates::du2_dp(resolved[0], resolved[1])),
        (GateKind::U2, 1) => apply_1q(&mut out, n, q0, &gates::du2_dl(resolved[0], resolved[1])),
        (GateKind::U3, 0) => apply_1q(
            &mut out,
            n,
            q0,
            &gates::du3_dt(resolved[0], resolved[1], resolved[2]),
        ),
        (GateKind::U3, 1) => apply_1q(
            &mut out,
            n,
            q0,
            &gates::du3_dp(resolved[0], resolved[1], resolved[2]),
        ),
        (GateKind::U3, 2) => apply_1q(
            &mut out,
            n,
            q0,
            &gates::du3_dl(resolved[0], resolved[1], resolved[2]),
        ),
        (GateKind::CRz, 0) => {
            apply_2q(
                &mut out,
                n,
                q0,
                op.qubits[1].0 as usize,
                &gates::dcrz(resolved[0]),
            );
        }
        (GateKind::CU3, 0) => {
            apply_2q(
                &mut out,
                n,
                q0,
                op.qubits[1].0 as usize,
                &gates::dcu3_dt(resolved[0], resolved[1], resolved[2]),
            );
        }
        (GateKind::CU3, 1) => {
            apply_2q(
                &mut out,
                n,
                q0,
                op.qubits[1].0 as usize,
                &gates::dcu3_dp(resolved[0], resolved[1], resolved[2]),
            );
        }
        (GateKind::CU3, 2) => {
            apply_2q(
                &mut out,
                n,
                q0,
                op.qubits[1].0 as usize,
                &gates::dcu3_dl(resolved[0], resolved[1], resolved[2]),
            );
        }
        (GateKind::Rbs, 0) => {
            apply_2q(
                &mut out,
                n,
                q0,
                op.qubits[1].0 as usize,
                &gates::drbs(resolved[0]),
            );
        }
        _ => {
            return Err(OmegaError::Unsupported(format!(
                "adjoint: no derivative for {:?} param_idx={}",
                op.gate, param_idx
            )));
        }
    }

    Ok(out)
}

/// Apply U† to state in-place.
fn apply_adjoint_gate(
    state: &mut [Complex64],
    n: usize,
    op: &GateOp,
    params: &ParameterBinding,
) -> Result<()> {
    let resolved: Vec<f64> = op
        .params
        .iter()
        .map(|p| params.resolve(p))
        .collect::<Result<Vec<_>>>()?;

    let q0 = op.qubits[0].0 as usize;

    match &op.gate {
        // Non-parametric 1Q (self-adjoint or known adjoints)
        GateKind::H => apply_1q(state, n, q0, &gates::h()), // H† = H
        GateKind::X => apply_1q(state, n, q0, &gates::x()), // X† = X
        GateKind::Y => apply_1q(state, n, q0, &gates::y()), // Y† = Y
        GateKind::Z => apply_1q(state, n, q0, &gates::z()), // Z† = Z
        GateKind::S => apply_1q(state, n, q0, &gates::sdg()), // S† = Sdg
        GateKind::Sdg => apply_1q(state, n, q0, &gates::s()), // Sdg† = S
        GateKind::T => apply_1q(state, n, q0, &gates::tdg()), // T† = Tdg
        GateKind::Tdg => apply_1q(state, n, q0, &gates::t()), // Tdg† = T
        GateKind::Id => {}

        // Parametric 1Q: compute gate, then adjoint
        GateKind::Rx => apply_1q(state, n, q0, &gates::adjoint_1q(&gates::rx(resolved[0]))),
        GateKind::Ry => apply_1q(state, n, q0, &gates::adjoint_1q(&gates::ry(resolved[0]))),
        GateKind::Rz => apply_1q(state, n, q0, &gates::adjoint_1q(&gates::rz(resolved[0]))),
        GateKind::U1 => apply_1q(state, n, q0, &gates::adjoint_1q(&gates::u1(resolved[0]))),
        GateKind::U2 => apply_1q(
            state,
            n,
            q0,
            &gates::adjoint_1q(&gates::u2(resolved[0], resolved[1])),
        ),
        GateKind::U3 => apply_1q(
            state,
            n,
            q0,
            &gates::adjoint_1q(&gates::u3(resolved[0], resolved[1], resolved[2])),
        ),

        // 2Q gates
        GateKind::CX => apply_2q(state, n, q0, op.qubits[1].0 as usize, &gates::cx()), // CX† = CX
        GateKind::CY => apply_2q(
            state,
            n,
            q0,
            op.qubits[1].0 as usize,
            &gates::adjoint_2q(&gates::cy()),
        ),
        GateKind::CZ => apply_2q(state, n, q0, op.qubits[1].0 as usize, &gates::cz()), // CZ† = CZ
        GateKind::Swap => apply_2q(state, n, q0, op.qubits[1].0 as usize, &gates::swap()), // SWAP† = SWAP
        GateKind::CRz => apply_2q(
            state,
            n,
            q0,
            op.qubits[1].0 as usize,
            &gates::adjoint_2q(&gates::crz(resolved[0])),
        ),
        GateKind::CU3 => apply_2q(
            state,
            n,
            q0,
            op.qubits[1].0 as usize,
            &gates::adjoint_2q(&gates::cu3(resolved[0], resolved[1], resolved[2])),
        ),
        // RBS(θ)† = RBS(−θ)
        GateKind::Rbs => apply_2q(
            state,
            n,
            q0,
            op.qubits[1].0 as usize,
            &gates::rbs(-resolved[0]),
        ),

        // 3Q gates
        GateKind::CCX => {
            // CCX† = CCX (self-adjoint)
            let q1 = op.qubits[1].0 as usize;
            let q2 = op.qubits[2].0 as usize;
            apply_ccx_forward(state, n, q0, q1, q2);
        }
        GateKind::CSwap => {
            // CSwap† = CSwap (self-adjoint)
            let q1 = op.qubits[1].0 as usize;
            let q2 = op.qubits[2].0 as usize;
            apply_cswap_forward(state, n, q0, q1, q2);
        }

        _ => {
            return Err(OmegaError::Unsupported(format!(
                "adjoint: unsupported gate {:?}",
                op.gate
            )));
        }
    }

    Ok(())
}

/// Compute O|ψ⟩ where O = Σ cᵢ Pᵢ is a sum of Pauli strings.
fn apply_observable(state: &[Complex64], n: usize, observable: &Observable) -> Vec<Complex64> {
    let dim = 1usize << n;
    let mut result = vec![Complex64::new(0.0, 0.0); dim];
    let i_unit = Complex64::new(0.0, 1.0);

    for (coeff, pauli_string) in &observable.terms {
        let coeff_c = Complex64::new(*coeff, 0.0);
        for basis in 0..dim {
            let mut target = basis;
            let mut phase = coeff_c;

            for (q, op) in pauli_string {
                let q = *q as usize;
                let bit = (basis >> q) & 1;
                match op {
                    PauliOp::I => {}
                    PauliOp::X => {
                        target ^= 1 << q;
                    }
                    PauliOp::Y => {
                        target ^= 1 << q;
                        phase *= if bit == 0 { i_unit } else { -i_unit };
                    }
                    PauliOp::Z => {
                        if bit == 1 {
                            phase *= -Complex64::new(1.0, 0.0);
                        }
                    }
                }
            }

            result[target] += phase * state[basis];
        }
    }

    result
}

/// ⟨a|b⟩ = Σᵢ conj(aᵢ) · bᵢ
fn inner_product(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai.conj() * bi).sum()
}

/// Collect all SymbolIds referenced in a ParamExpr.
fn collect_symbols(expr: &ParamExpr) -> Vec<SymbolId> {
    let mut syms = Vec::new();
    collect_symbols_inner(expr, &mut syms);
    syms.sort();
    syms.dedup();
    syms
}

fn collect_symbols_inner(expr: &ParamExpr, out: &mut Vec<SymbolId>) {
    match expr {
        ParamExpr::Symbol(id) => out.push(*id),
        ParamExpr::Negate(inner) => collect_symbols_inner(inner, out),
        ParamExpr::Add(a, b) | ParamExpr::Mul(a, b) => {
            collect_symbols_inner(a, out);
            collect_symbols_inner(b, out);
        }
        ParamExpr::Concrete(_) => {}
    }
}

/// CCX (Toffoli) — direct implementation.
fn apply_ccx_forward(state: &mut [Complex64], n: usize, c0: usize, c1: usize, target: usize) {
    let dim = 1usize << n;
    let mask_c0 = 1usize << c0;
    let mask_c1 = 1usize << c1;
    let mask_t = 1usize << target;

    for i in 0..dim {
        if (i & mask_c0) != 0 && (i & mask_c1) != 0 && (i & mask_t) == 0 {
            let j = i | mask_t;
            state.swap(i, j);
        }
    }
}

/// CSwap (Fredkin) — direct implementation.
fn apply_cswap_forward(state: &mut [Complex64], n: usize, ctrl: usize, t0: usize, t1: usize) {
    let dim = 1usize << n;
    let mask_c = 1usize << ctrl;
    let mask_t0 = 1usize << t0;
    let mask_t1 = 1usize << t1;

    for i in 0..dim {
        if (i & mask_c) != 0 && (i & mask_t0) != 0 && (i & mask_t1) == 0 {
            let j = (i & !mask_t0) | mask_t1;
            state.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StatevectorBackend;
    use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
    use omega_core::executor::PauliOp;
    use omega_core::gradient::{compute_gradient, GradMethod};
    use omega_core::params::ParameterBinding;
    use smallvec::smallvec;

    fn z_observable() -> Observable {
        Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z)])],
        }
    }

    #[test]
    fn test_adjoint_ry_single_gate() {
        // d⟨Z⟩/dθ = -sin(θ) for Ry(θ)|0⟩
        let theta = std::f64::consts::FRAC_PI_3;

        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.symbols.insert(0, "theta".to_string());
        circuit.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });

        let mut params = ParameterBinding::new();
        params.bind(0, theta);

        let grads = adjoint_gradient(&circuit, &params, &z_observable()).unwrap();
        let expected = -theta.sin();
        assert!(
            (grads[0].1 - expected).abs() < 1e-10,
            "AD grad = {} (expected {})",
            grads[0].1,
            expected
        );
    }

    #[test]
    fn test_adjoint_matches_param_shift() {
        // Ry(θ₁) on q0, Rz(θ₂) on q0: compare AD to parameter-shift
        let theta1 = 0.8;
        let theta2 = 1.5;

        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.symbols.insert(0, "theta1".to_string());
        circuit.symbols.insert(1, "theta2".to_string());
        circuit.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Symbol(1)],
            classical_bit: None,
            condition: None,
        });

        let mut params = ParameterBinding::new();
        params.bind(0, theta1);
        params.bind(1, theta2);

        let backend = StatevectorBackend::new();
        let obs = z_observable();

        let ad_grads = adjoint_gradient(&circuit, &params, &obs).unwrap();
        let ps_grads = compute_gradient(
            &backend,
            &circuit,
            &params,
            &obs,
            &GradMethod::ParameterShift,
        )
        .unwrap();

        for (ad, ps) in ad_grads.iter().zip(ps_grads.iter()) {
            assert_eq!(ad.0, ps.0);
            assert!(
                (ad.1 - ps.1).abs() < 1e-10,
                "sym {}: AD={}, PS={}",
                ad.0,
                ad.1,
                ps.1
            );
        }
    }

    #[test]
    fn test_adjoint_two_qubit_entangling() {
        // Ry(θ) on q0, CX(q0,q1), measure ⟨ZZ⟩
        let theta = 1.2;

        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        circuit.symbols.insert(0, "theta".to_string());
        circuit.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(0), Qubit(1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });

        let mut params = ParameterBinding::new();
        params.bind(0, theta);

        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
        };

        let backend = StatevectorBackend::new();
        let ad_grads = adjoint_gradient(&circuit, &params, &obs).unwrap();
        let ps_grads = compute_gradient(
            &backend,
            &circuit,
            &params,
            &obs,
            &GradMethod::ParameterShift,
        )
        .unwrap();

        assert!(
            (ad_grads[0].1 - ps_grads[0].1).abs() < 1e-10,
            "AD={}, PS={}",
            ad_grads[0].1,
            ps_grads[0].1
        );
    }

    #[test]
    fn test_param_shift_matches_adjoint_for_crz() {
        // Regression test for a previously latent bug: CRz's generator
        // `M/2` has spectrum {0, 0, ±1/2} → expectation has *two*
        // frequencies ({1/2, 1}). The 2-term ±π/2 rule that
        // `parameter_shift_gradient` used to apply uniformly is wrong on
        // the 1/2-frequency block-off-diagonal entries (it would require
        // `sin(π/4) = 1/2`, false). After the per-slot fix the CRz slot
        // uses the 4-term Banchi-Crooks variant; this test pins that
        // path against the (independently Lean'd) adjoint AD.
        //
        // Circuit: a state with weight on both control branches, then
        // CRz(s), then a non-diagonal observable. `H` on the control
        // gives 50/50 |0⟩/|1⟩ and a non-diagonal `X⊗Z` observable
        // surfaces the cross-block (Δ = ±1/2) entries that distinguish
        // the broken 2-term rule from the correct 4-term one.
        let s = 0.7;
        let t = 1.1;

        let mut circuit = CircuitIR::new(2, CircuitType::GateBased);
        circuit.symbols.insert(0, "s".to_string());
        circuit.symbols.insert(1, "t".to_string());
        circuit.add_op(GateOp {
            gate: GateKind::H,
            qubits: smallvec![Qubit(0)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(1)],
            params: smallvec![ParamExpr::Symbol(1)],
            classical_bit: None,
            condition: None,
        });
        circuit.add_op(GateOp {
            gate: GateKind::CRz,
            qubits: smallvec![Qubit(0), Qubit(1)],
            params: smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });

        let mut params = ParameterBinding::new();
        params.bind(0, s);
        params.bind(1, t);

        let backend = StatevectorBackend::new();
        let obs = Observable {
            terms: vec![(1.0, vec![(0, PauliOp::X), (1, PauliOp::Z)])],
        };

        let ad_grads = adjoint_gradient(&circuit, &params, &obs).unwrap();
        let ps_grads = compute_gradient(
            &backend,
            &circuit,
            &params,
            &obs,
            &GradMethod::ParameterShift,
        )
        .unwrap();
        let fd_grads = compute_gradient(
            &backend,
            &circuit,
            &params,
            &obs,
            &GradMethod::FiniteDifference { epsilon: 1e-6 },
        )
        .unwrap();

        // PSR ↔ AD: should be exact (both analytic), tolerance 1e-10.
        for (ad, ps) in ad_grads.iter().zip(ps_grads.iter()) {
            assert_eq!(ad.0, ps.0);
            assert!(
                (ad.1 - ps.1).abs() < 1e-10,
                "sym {}: AD={}, PSR={} (Δ = {:e})",
                ad.0,
                ad.1,
                ps.1,
                ad.1 - ps.1
            );
        }
        // PSR ↔ FD: cross-check independent of the AD path, looser
        // tolerance from the finite-difference truncation.
        for (ps, fd) in ps_grads.iter().zip(fd_grads.iter()) {
            assert_eq!(ps.0, fd.0);
            assert!(
                (ps.1 - fd.1).abs() < 1e-4,
                "sym {}: PSR={}, FD={} (Δ = {:e})",
                ps.0,
                ps.1,
                fd.1,
                ps.1 - fd.1
            );
        }
    }

    #[test]
    fn test_adjoint_qaoa_compound_expressions() {
        // QAOA uses compound expressions like Mul(Concrete(2*J), Symbol(gamma)).
        // Verify AD against finite differences (parameter-shift doesn't handle
        // compound expressions correctly because the π/2 shift is in symbol space,
        // not in the resolved angle space).
        use omega_core::qaoa::qaoa_circuit;
        use omega_core::qubo::Qubo;

        let mut q = Qubo::new(2);
        q.set(0, 0, -1.0);
        q.set(1, 1, -1.0);
        q.set(0, 1, 2.0);
        let ising = q.to_ising();
        let circuit = qaoa_circuit(&ising, 1);
        let observable = ising.to_observable();

        let mut params = ParameterBinding::new();
        for &id in circuit.symbols.keys() {
            params.bind(id, 0.5);
        }

        let backend = StatevectorBackend::new();
        let ad_grads = adjoint_gradient(&circuit, &params, &observable).unwrap();

        // Compare to finite differences
        let fd_grads = compute_gradient(
            &backend,
            &circuit,
            &params,
            &observable,
            &GradMethod::FiniteDifference { epsilon: 1e-7 },
        )
        .unwrap();

        for (ad, fd) in ad_grads.iter().zip(fd_grads.iter()) {
            assert_eq!(ad.0, fd.0);
            assert!(
                (ad.1 - fd.1).abs() < 1e-5,
                "sym {} ({}): AD={}, FD={}",
                ad.0,
                circuit.symbols.get(&ad.0).unwrap_or(&"?".to_string()),
                ad.1,
                fd.1
            );
        }
    }

    #[test]
    fn test_adjoint_via_grad_method() {
        // Verify GradMethod::Adjoint works through the dispatch
        let theta = 0.7;

        let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
        circuit.symbols.insert(0, "theta".to_string());
        circuit.add_op(GateOp {
            gate: GateKind::Ry,
            qubits: smallvec![Qubit(0)],
            params: smallvec![ParamExpr::Symbol(0)],
            classical_bit: None,
            condition: None,
        });

        let mut params = ParameterBinding::new();
        params.bind(0, theta);

        let backend = StatevectorBackend::new();
        let obs = z_observable();

        let grads =
            compute_gradient(&backend, &circuit, &params, &obs, &GradMethod::Adjoint).unwrap();
        let expected = -theta.sin();
        assert!(
            (grads[0].1 - expected).abs() < 1e-10,
            "GradMethod::Adjoint: {} (expected {})",
            grads[0].1,
            expected
        );
    }

    #[test]
    fn test_gate_derivative_accuracy() {
        // Verify dRy/dθ against finite differences
        let eps = 1e-7;
        for theta in &[0.0, 0.5, 1.0, 2.0, std::f64::consts::PI] {
            let g_plus = gates::ry(theta + eps);
            let g_minus = gates::ry(theta - eps);
            let dg = gates::dry(*theta);
            for i in 0..4 {
                let fd = (g_plus[i] - g_minus[i]) / (2.0 * eps);
                assert!(
                    (dg[i] - fd).norm() < 1e-5,
                    "dRy[{}] at theta={}: AD={}, FD={}",
                    i,
                    theta,
                    dg[i],
                    fd
                );
            }
        }
    }
}
