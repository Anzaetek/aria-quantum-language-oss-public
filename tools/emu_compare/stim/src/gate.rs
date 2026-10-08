// SPDX-License-Identifier: Apache-2.0
//! The two checks a stabilizer speed row is not allowed to skip.
//!
//! Clifford expectation values are exact integers in {-1, 0, 1}. A
//! tolerance here would admit a tableau that has left the stabilizer
//! group, which is the failure `peek_observable_expectation` exists to
//! make impossible. Sampling rows additionally require the shots that
//! came back to be the shots that were asked for, on each arm: a sampler
//! that silently returns a shorter batch is the shape this lane is here
//! to catch.

use omega_emu_compare::Classification;

/// Stim instructions that are the d=25 workload and are not a Clifford
/// gate. Skipping one of these and running the rest would time a
/// different computation and call it the memory experiment.
const REFUSED_INSTRUCTIONS: &[&str] = &[
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
];

/// Clifford and measure/reset instructions this backend can say in its
/// own gates. `MR` is measure-reset, which is `M` then `R` here; it is
/// not the thing the capability row refuses.
const CLIFFORD_INSTRUCTIONS: &[&str] = &[
    "H", "S", "S_DAG", "X", "Y", "Z", "I", "CX", "CY", "CZ", "SWAP", "M", "MX", "MY", "MZ", "R",
    "MR",
];

/// Annotations and the repeat block. They are not gates. `REPEAT` is
/// structural: the body's instruction names are also in the set, and
/// those are what get classified.
const STRUCTURAL_INSTRUCTIONS: &[&str] = &["REPEAT", "TICK", "QUBIT_COORDS", "SHIFT_COORDS"];

/// Why a pair of expectation values is not a speed row.
#[derive(Debug, Clone, PartialEq)]
pub struct Disagreement {
    /// Our arm's value, as returned.
    pub ours: f64,
    /// Stim's value, as returned.
    pub stim: f64,
    /// The sentence the void row publishes. Both numbers are in it; a
    /// finding that names only one side is how a disagreement gets dropped
    /// in prose.
    pub reason: String,
}

/// `Some(-1 | 0 | 1)` when `v` is that value with no rounding.
pub fn exact_clifford_integer(v: f64) -> Option<i8> {
    if v == 0.0 {
        Some(0)
    } else if v == 1.0 {
        Some(1)
    } else if v == -1.0 {
        Some(-1)
    } else {
        None
    }
}

/// Admit the pair only when both sides are the same exact integer.
///
/// Agreement at a non-integer (two arms both returning 0.5) is still a
/// finding: on these circuits the integer is the physics, and a shared
/// non-integer is a shared departure from it.
pub fn exact_gate(ours: f64, stim: f64) -> Result<i8, Disagreement> {
    match (exact_clifford_integer(ours), exact_clifford_integer(stim)) {
        (Some(a), Some(b)) if a == b => Ok(a),
        _ => Err(Disagreement {
            ours,
            stim,
            reason: format!(
                "disagreement finding: ours={ours} stim={stim} (§4.3); the row is not dropped"
            ),
        }),
    }
}

/// The shots that came back are the shots that were requested.
pub fn account_shots(requested: u64, observed: u64) -> Result<(), String> {
    if observed != requested {
        Err(format!(
            "shot-count shortfall: requested {requested}, observed {observed}"
        ))
    } else {
        Ok(())
    }
}

/// What the d=25 instruction set does to this backend.
///
/// `Ok` is the refusal text — the workload contains noise or detector
/// instructions, which is the capability fact. `Err` is an unclassified
/// name, or a circuit that had neither: both mean we must not publish a
/// row that claims to have looked at the memory experiment.
pub fn detector_noise_refusal(names: &[&str]) -> Result<String, String> {
    let mut refused = Vec::new();
    let mut unknown = Vec::new();
    for name in names {
        if REFUSED_INSTRUCTIONS.contains(name) {
            refused.push(*name);
        } else if CLIFFORD_INSTRUCTIONS.contains(name) || STRUCTURAL_INSTRUCTIONS.contains(name) {
        } else {
            unknown.push(*name);
        }
    }
    if !unknown.is_empty() {
        return Err(format!(
            "unclassified stim instructions {unknown:?}; not skipping them and not calling the circuit sampled"
        ));
    }
    if refused.is_empty() {
        return Err(
            "the d=25 circuit contained no noise or detector instruction; it is not the workload"
                .into(),
        );
    }
    Ok(format!(
        "pauli backend cannot express {}; it has no noise channel and no detector or logical-observable sampler",
        refused.join(", ")
    ))
}

/// §4.6, the same predicates the schema re-derives. Duplicated here so the
/// row is built from the rule rather than tagged afterwards.
pub fn classify(
    ratio: f64,
    ours_min: f64,
    ours_med: f64,
    stim_min: f64,
    stim_med: f64,
) -> Classification {
    let disjoint = ours_med < stim_min || stim_med < ours_min;
    let beyond = ratio >= omega_emu_compare::WIN_RATIO_THRESHOLD
        || ratio <= 1.0 / omega_emu_compare::WIN_RATIO_THRESHOLD;
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

/// Relative gap against a reference, with scale at least 1 so a zero
/// reference does not divide by zero. Exact ±1 and 0 then have a gap
/// equal to the absolute difference.
pub fn rel_gap(got: f64, reference: f64) -> f64 {
    let scale = reference.abs().max(1.0);
    (got - reference).abs() / scale
}

// ---------------------------------------------------------------------------
// §4.3c: a value gate must be one a wrong implementation could fail
// ---------------------------------------------------------------------------

/// One observable inside a value gate, with both arms' answers.
#[derive(Debug, Clone, PartialEq)]
pub struct GateObs {
    /// What to call it in the witness, e.g. `Z0` or `U Z0 U+`.
    pub label: String,
    /// Our `Observable` syntax.
    pub obs: String,
    /// The signed stim PauliString.
    pub pauli: String,
    /// What the construction predicts, when it predicts anything. `+1` for
    /// the Heisenberg image of a Z-type Pauli, `-1` for its negation; `None`
    /// for the row's own observable, which is whatever it measures.
    pub predicted: Option<i8>,
    /// Our arm.
    pub ours: f64,
    /// Theirs.
    pub stim: i64,
    /// The dense oracle, where one is affordable.
    pub oracle: Option<f64>,
}

/// §4.3c: the values a trivial implementation returns without computing.
pub const DEGENERATE: [i8; 3] = [0, 1, -1];

/// Refuse a gate whose observables all came back the same stub-reachable
/// value.
///
/// This is the lane-side copy of the rule `omega_emu_compare::check`
/// enforces at write time, and it is here on purpose: the writer's refusal
/// arrives after the row has been measured, which is too late to say *what
/// to measure instead*. Failing here names the fix.
///
/// The receipt is this lane's own output. Six published rows — both Clifford
/// brickwall expectations, both their sampling rows, and both surface-d5
/// rows — gated on a single value: `0` for `<Z0>` on the brickwall, `1` for
/// the logical `Z` on the syndrome circuit. Every field of those rows was
/// filled in correctly. The observable was reasonable and the circuit was
/// reasonable; their combination made the gate read `0 == 0`, which a
/// backend returning a constant passes.
pub fn discriminating(values: &[i8]) -> Result<(), String> {
    if values.is_empty() {
        return Err("a value gate with no observable pins nothing (§4.3c)".to_string());
    }
    let first = values[0];
    if values.iter().any(|v| *v != first) {
        return Ok(());
    }
    if !DEGENERATE.contains(&first) {
        return Ok(());
    }
    Err(format!(
        "degenerate value gate: all {} observables came back {first}, which is a value a \
         backend computing nothing returns (§4.3c). Widen the gate: the Heisenberg image \
         U P U-dagger is +1 by construction and its negation is -1, so [measured, +1, -1] \
         discriminates at the cost of two extra peeks per arm, outside the timed region",
        values.len()
    ))
}

#[cfg(test)]
mod discriminating_tests {
    use super::*;

    #[test]
    fn the_six_published_gates_are_refused() {
        // Exactly the shapes that were on main: four rows at 0, two at 1.
        for v in [0i8, 1, -1] {
            let err = discriminating(&[v]).expect_err("a lone degenerate value");
            assert!(err.contains("degenerate value gate"), "{err}");
            assert!(
                err.contains("Heisenberg"),
                "the refusal must name the fix: {err}"
            );
        }
    }

    #[test]
    fn repeating_a_degenerate_value_does_not_rescue_it() {
        assert!(discriminating(&[0, 0, 0]).is_err());
        assert!(discriminating(&[1, 1]).is_err());
    }

    #[test]
    fn the_widened_gate_is_admitted() {
        discriminating(&[0, 1, -1]).expect("[0, 1, -1] cannot be passed by a constant");
        discriminating(&[1, 1, -1]).expect("the surface row's shape: logical 1, image 1, -1");
    }

    #[test]
    fn an_empty_gate_is_refused() {
        assert!(discriminating(&[]).is_err());
    }
}
