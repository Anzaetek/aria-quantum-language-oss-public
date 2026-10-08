// SPDX-License-Identifier: Apache-2.0
//! The checks a fermionic row is not allowed to skip.
//!
//! The ≤8-qubit fixtures are capability rows. A speed classification on one
//! is a failure here, before a row is built: at that size a wall-clock
//! measures call overhead. The value gate is two-sided absolute relative
//! error at 1e-10. The sign-flip pin keeps the unflipped oracle fixed and
//! requires the flipped arm to miss it by more than 1e-2; agreement of two
//! flipped circuits would not.

use omega_emu_compare::{Classification, RowBody, GATE_REL_F64};

/// Kitaev n2, Kitaev n8, Hubbard2, H2. Not LUCJ, and not Kitaev n64.
pub const BINDING_ROWS: &[&str] = &[
    "ferm-kitaev-n2",
    "ferm-kitaev-n8",
    "ferm-hubbard2",
    "ferm-h2-ground",
];

/// Same floor as the bridge harness's tunnel-θ pin. A smaller floor admits
/// a flip the state barely felt.
pub const SIGN_FLIP_FLOOR: f64 = 1e-2;

/// §4.3a: flag, do not drop, when the repeats spread by more than this.
pub const SPREAD_FLAG: f64 = 5.0;

/// What the harness is about to publish, as far as the blur-guard cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowLabel {
    /// No ratio and no classification.
    Capability,
    /// A win, a loss, or a tie. All three are speed classifications.
    Speed(Classification),
}

/// Why a pair of energies is not admitted.
#[derive(Debug, Clone, PartialEq)]
pub struct Disagreement {
    /// Our value.
    pub ours: f64,
    /// The competitor's value.
    pub theirs: f64,
    /// The reference both were held to.
    pub reference: f64,
    /// The sentence a void row publishes.
    pub reason: String,
}

/// `|got − reference| / max(|reference|, 1)`.
///
/// Absolute, so a value on either side of the reference fails the same way.
/// Scale at least 1 keeps a zero reference from dividing by zero; none of
/// this lane's references are zero.
pub fn rel_gap(got: f64, reference: f64) -> f64 {
    let scale = reference.abs().max(1.0);
    (got - reference).abs() / scale
}

/// Both arms against the reference, and against each other.
///
/// A one-sided check `got − reference ≤ tol` admits a value that is too low.
/// This one fails both directions.
pub fn admit(ours: f64, theirs: f64, reference: f64) -> Result<(f64, f64), Disagreement> {
    let ours_gap = rel_gap(ours, reference);
    let theirs_gap = rel_gap(theirs, reference);
    let cross = rel_gap(ours, theirs);
    if ours_gap <= GATE_REL_F64 && theirs_gap <= GATE_REL_F64 && cross <= GATE_REL_F64 {
        Ok((ours_gap, theirs_gap))
    } else {
        Err(Disagreement {
            ours,
            theirs,
            reference,
            reason: format!(
                "disagreement finding: ours={ours} ffsim={theirs} reference={reference} \
                 gaps {ours_gap:.3e}/{theirs_gap:.3e} cross {cross:.3e} over {GATE_REL_F64:.0e} \
                 (§4.3, two-sided); the row is not dropped"
            ),
        })
    }
}

/// The flipped arm against the unflipped oracle.
///
/// `Ok` is the absolute delta, which has to exceed [`SIGN_FLIP_FLOOR`].
/// Sending the flipped circuit to both arms and comparing them is not this
/// function: the oracle argument is the unflipped value.
pub fn sign_flip_detected(unflipped_oracle: f64, flipped_ours: f64) -> Result<f64, String> {
    let delta = (flipped_ours - unflipped_oracle).abs();
    if delta > SIGN_FLIP_FLOOR {
        Ok(delta)
    } else {
        Err(format!(
            "flipping the seeded sign was NOT detected against the unflipped oracle. \
             |Δ| = {delta:e} (oracle {unflipped_oracle}, flipped {flipped_ours}). \
             Same-circuit agreement would hide this; the oracle has to stay the pinned one."
        ))
    }
}

/// The ≤8-qubit rows, and Kitaev n64, have no speed classification.
pub fn admit_label(row_id: &str, label: RowLabel) -> Result<(), String> {
    if BINDING_ROWS.contains(&row_id) {
        if let RowLabel::Speed(class) = label {
            return Err(format!(
                "speed classification {class:?} on {row_id} is a test failure: at <= 8 qubits \
                 a wall-clock measures call overhead (§3, §4.7)"
            ));
        }
    }
    if row_id == "ferm-kitaev-n64" {
        if let RowLabel::Speed(class) = label {
            return Err(format!(
                "speed classification {class:?} on ferm-kitaev-n64: dense competitors face \
                 2^64 and FQE's sector does not contain the pairing state"
            ));
        }
    }
    Ok(())
}

/// The published body agrees with [`admit_label`].
pub fn admit_body(row_id: &str, body: &RowBody) -> Result<(), String> {
    let label = match body {
        RowBody::Capability(_) => RowLabel::Capability,
        RowBody::Speed(speed) => RowLabel::Speed(speed.classification),
    };
    admit_label(row_id, label)
}

/// ffsim's sector path names this phrase when it refuses pairing.
pub fn kitaev_sector_refusal(message: &str) -> Result<(), String> {
    if message.contains("does not conserve particle number") {
        Ok(())
    } else {
        Err(format!(
            "ffsim's refusal did not say the operator fails to conserve particle number: {message}"
        ))
    }
}

/// §4.6, duplicated so the row is built from the rule.
pub fn classify(
    ratio: f64,
    ours_min: f64,
    ours_med: f64,
    theirs_min: f64,
    theirs_med: f64,
) -> Classification {
    let disjoint = ours_med < theirs_min || theirs_med < ours_min;
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

/// True when the repeats are too spread to read as one engine (§4.3a).
pub fn spread_flagged(samples: &[f64]) -> Result<bool, String> {
    if samples.is_empty() {
        return Err("no samples".into());
    }
    let mut min = f64::MAX;
    let mut max = 0.0_f64;
    for t in samples {
        if !t.is_finite() || *t <= 0.0 {
            return Err(format!("non-positive sample {t}"));
        }
        min = min.min(*t);
        max = max.max(*t);
    }
    Ok(max / min > SPREAD_FLAG)
}

/// 1-minute load strictly above `void_above` voids the row. The threshold is
/// [`omega_emu_compare::load_void_above`] for the host, not a copy of it.
pub fn row_void_at_load(load_1min: f64, void_above: f64) -> bool {
    omega_emu_compare::void_at_load(load_1min, void_above)
}

/// Ground energy of the sweet-spot Kitaev chain, −(N−1).
pub fn kitaev_analytic(n: u32) -> f64 {
    -((n - 1) as f64)
}

/// Two-electron ground energy of the 2-site Hubbard model at t = 1, U = 4.
pub fn hubbard_analytic() -> f64 {
    2.0 - 2.0 * 2.0_f64.sqrt()
}

/// §4.4a for a capability row, shared with the other lanes.
pub use omega_emu_compare::external_void_reason;
