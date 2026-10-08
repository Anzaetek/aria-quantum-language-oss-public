// SPDX-License-Identifier: Apache-2.0
//! The predicates that admit or refuse a dense row, each separately testable.

use omega_emu_compare::{Classification, GATE_REL_F32, GATE_REL_F64, WIN_RATIO_THRESHOLD};

/// §4.6, built from the rule rather than tagged afterwards.
pub fn classify(
    ratio: f64,
    ours_min: f64,
    ours_med: f64,
    comp_min: f64,
    comp_med: f64,
) -> Classification {
    let disjoint = ours_med < comp_min || comp_med < ours_min;
    let beyond = ratio >= WIN_RATIO_THRESHOLD || ratio <= 1.0 / WIN_RATIO_THRESHOLD;
    if beyond && disjoint {
        if ratio > 1.0 {
            Classification::Win
        } else {
            Classification::Loss
        }
    } else {
        Classification::Tie
    }
}

/// Relative gap with scale at least 1, as the other lanes use.
pub fn rel_gap(got: f64, reference: f64) -> f64 {
    (got - reference).abs() / reference.abs().max(1.0)
}

/// Below this the oracle value is treated as degenerate for the circuit.
///
/// A value gate on a quantity that is zero for the circuit cannot fail: two
/// arms that both return 0 agree. That shape has bitten this campaign three
/// times (⟨Z₀⟩ on the pinned HEA and Clifford circuits; ⟨P⟩ = 0 on stabilizer
/// states; an identity `rbs`). So the cell is refused unless the oracle's
/// value is clearly non-zero — far above both arms' tolerance.
pub const NONTRIVIAL_MIN_ABS: f64 = 1e-3;

/// Refuse a cell whose oracle value cannot discriminate.
pub fn nontrivial(oracle_value: f64) -> Result<(), String> {
    if oracle_value.abs() < NONTRIVIAL_MIN_ABS || !oracle_value.is_finite() {
        return Err(format!(
            "the oracle value {oracle_value:e} is below {NONTRIVIAL_MIN_ABS:e}: a value gate on \
             a quantity that is zero for the circuit cannot fail — refusing the cell"
        ));
    }
    Ok(())
}

/// The state gate's tolerance for an arm of the given precision.
pub fn state_tol(f64_arm: bool) -> f64 {
    if f64_arm {
        GATE_REL_F64
    } else {
        GATE_REL_F32
    }
}

/// §4.3a: two-sided `|1−F| ≤ tol` AND phase-aligned `max|Δψ| ≤ tol`.
pub fn state_gate(
    label: &str,
    one_minus_f_abs: f64,
    max_abs_diff: f64,
    tol: f64,
) -> Result<(), String> {
    if !(one_minus_f_abs.is_finite() && max_abs_diff.is_finite()) {
        return Err(format!("{label}: state gate numbers are not finite"));
    }
    if one_minus_f_abs > tol {
        return Err(format!(
            "{label}: |1−F| = {one_minus_f_abs:.3e} > {tol:.0e} against the Aer double state"
        ));
    }
    if max_abs_diff > tol {
        return Err(format!(
            "{label}: phase-aligned max|Δψ| = {max_abs_diff:.3e} > {tol:.0e} against the Aer double state"
        ));
    }
    Ok(())
}

/// Sampling gate: two-sided per-qubit z-score of the `<Z>` estimate against
/// the exact marginal, `|m̂ − m| / sqrt((1 − m²)/shots) ≤ 5` on every qubit.
/// Returns the worst |z|.
pub const Z_BOUND: f64 = 5.0;

pub fn z_gate(label: &str, estimate: &[f64], exact: &[f64], shots: u32) -> Result<f64, String> {
    if estimate.len() != exact.len() || estimate.is_empty() {
        return Err(format!(
            "{label}: {} estimates against {} exact marginals",
            estimate.len(),
            exact.len()
        ));
    }
    let mut worst: f64 = 0.0;
    for (q, (e, m)) in estimate.iter().zip(exact).enumerate() {
        let var = (1.0 - m * m).max(1e-12) / f64::from(shots);
        let z = (e - m).abs() / var.sqrt();
        if !z.is_finite() || z > Z_BOUND {
            return Err(format!(
                "{label}: qubit {q} <Z> estimate {e:.4} vs exact {m:.4} is {z:.2} standard \
                 errors away (bound {Z_BOUND})"
            ));
        }
        worst = worst.max(z);
    }
    Ok(worst)
}

/// §4.3a: on a loaded box an OpenMP arm collapses rather than slows.
pub const SPREAD_FLAG: f64 = 5.0;

/// `max/min` across a row's repeats, and whether it trips the flag.
pub fn spread(times: &[f64]) -> (f64, bool) {
    let mn = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = times.iter().cloned().fold(0.0_f64, f64::max);
    let r = mx / mn;
    (r, r > SPREAD_FLAG)
}
