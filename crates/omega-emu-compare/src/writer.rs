// SPDX-License-Identifier: Apache-2.0
//! The JSON-lines writer, and the consistency checks that sit in front of it.
//!
//! Two layers, deliberately:
//!
//! * **serde refuses an unwitnessed row.** [`crate::Witness`],
//!   [`crate::PathWitnesses`] and [`crate::GitRev`] fail inside `Serialize`,
//!   so `serde_json::to_string` on a row with an absent witness or a stale
//!   binary errors with no help from this module. A lane that bypasses the
//!   writer still cannot publish such a row.
//! * **the writer refuses an inconsistent one.** Whether `classification`
//!   agrees with the `derivation` inputs, whether the load is inside the
//!   threshold the row itself declares, whether a GPU row has its device
//!   census — these are cross-field facts, and checking them here keeps them
//!   readable instead of spread across hand-written `Serialize` impls.
//!
//! Not a harness. Opens a file, appends a line, flushes. §8 caps this phase at
//! "a serde struct and a JSONL writer, not a benchmark harness", and the cap
//! is the deliverable as much as the code is.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::load::{LOAD_VOID_ABOVE_LARGE_HOST, LOAD_VOID_ABOVE_SMALL_HOST, SMALL_HOST_MAX_CORES};
use crate::row::{
    Classification, FloorVerdict, Lane, Row, RowBody, SpeedRow, ValueGate, VoidRow,
    DEGENERATE_GATE_VALUES, DEGENERATE_GATE_WINDOW, GPU_UTIL_VOID_PCT, MIN_REPEATS,
    OVERHEAD_FLOOR_MULTIPLE, OVERHEAD_RULE, TIE_RULE, WIN_RATIO_THRESHOLD,
};

/// Relative slack when re-deriving a field from the row's own inputs. Tight
/// enough to catch a changed statistic, loose enough not to trip on the
/// round-trip of an f64 through JSON.
const REDERIVE_REL_TOL: f64 = 1e-9;

/// Why a row was not written.
#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    /// The row contradicts itself or the fixed policy. Every reason is named.
    #[error("refused {row_id}: {}", reasons.join("; "))]
    Refused {
        /// The row that was refused.
        row_id: String,
        /// Every reason, so one pass names them all.
        reasons: Vec<String>,
    },
    /// The row could not be serialized — an absent witness, a void rev pair.
    #[error("{row_id} does not serialize: {source}")]
    NotSerializable {
        /// The row that was refused.
        row_id: String,
        /// Serde's own message.
        #[source]
        source: serde_json::Error,
    },
    /// The file would not take the line.
    #[error("writing {path}: {source}")]
    Io {
        /// The target file.
        path: String,
        /// The underlying failure.
        #[source]
        source: std::io::Error,
    },
}

/// Appends rows to one `.jsonl` file.
pub struct RowWriter {
    path: String,
    out: BufWriter<File>,
}

impl RowWriter {
    /// Open (or create) a JSONL file and append to it. Appending matters: §7
    /// requires results to accrete as they are measured, "losses included at
    /// the moment they are measured, not in a final editing pass", so Ruling 3
    /// is enforced by ordering rather than by intention.
    pub fn append(path: impl AsRef<Path>) -> Result<Self, WriteError> {
        let path = path.as_ref();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|source| WriteError::Io {
                path: path.display().to_string(),
                source,
            })?;
        Ok(Self {
            path: path.display().to_string(),
            out: BufWriter::new(file),
        })
    }

    /// Write one row, or refuse it with every reason named.
    pub fn write_row(&mut self, row: &Row) -> Result<(), WriteError> {
        let reasons = check(row);
        if !reasons.is_empty() {
            return Err(WriteError::Refused {
                row_id: row.row_id.clone(),
                reasons,
            });
        }
        let line = serde_json::to_string(row).map_err(|source| WriteError::NotSerializable {
            row_id: row.row_id.clone(),
            source,
        })?;
        self.line(&line)
    }

    /// Write a void row. Never refused: a void row's reason is its whole
    /// value, and a refusal with nowhere to land becomes a silent drop.
    pub fn write_void(&mut self, void: &VoidRow) -> Result<(), WriteError> {
        let line = serde_json::to_string(void).map_err(|source| WriteError::NotSerializable {
            row_id: void.row_id.clone(),
            source,
        })?;
        self.line(&line)
    }

    fn line(&mut self, line: &str) -> Result<(), WriteError> {
        let io = |source| WriteError::Io {
            path: self.path.clone(),
            source,
        };
        writeln!(self.out, "{line}").map_err(io)?;
        self.out.flush().map_err(io)
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= REDERIVE_REL_TOL * a.abs().max(b.abs()).max(1.0)
}

/// Every way this row contradicts itself or the policy of §4, named.
///
/// Absent witnesses are *not* listed here: those are serde's job, and
/// duplicating them would invite the two lists to disagree. [`Row::refusals`]
/// is the diagnostic for that half.
pub fn check(row: &Row) -> Vec<String> {
    let mut out = Vec::new();
    if !row.precision.matched && row.precision.note.is_none() {
        out.push(
            "precision.matched is false with no note: §4.5 requires an unmatched row to \
             carry its annotation, and such a row can be published as a loss but never \
             quoted as a win"
                .to_string(),
        );
    }
    match row.circuit.quantity {
        crate::row::Quantity::Sampling if row.circuit.shots.is_none() => {
            out.push("circuit.quantity is sampling with no shots".to_string());
        }
        crate::row::Quantity::Expectation if row.circuit.observable.is_none() => {
            out.push("circuit.quantity is expectation with no observable".to_string());
        }
        _ => {}
    }
    if let RowBody::Speed(speed) = &row.body {
        check_speed(row.lane, speed, &mut out);
    }
    out
}

fn check_speed(lane: Lane, speed: &SpeedRow, out: &mut Vec<String>) {
    if let Some(gate) = speed.value_gate.get() {
        if !gate.passed {
            out.push(
                "value_gate.passed is false: §4.3 admits no untimed-but-published row — \
                 a failing gate is reported as a disagreement finding (a VoidRow), not \
                 as a speed row"
                    .to_string(),
            );
        } else if gate.ours_gap > gate.tolerance_rel || gate.competitor_gap > gate.tolerance_rel {
            out.push(format!(
                "value_gate says passed with gaps {:.3e}/{:.3e} over its own tolerance {:.3e}",
                gate.ours_gap, gate.competitor_gap, gate.tolerance_rel
            ));
        }
        check_gate_discriminates(gate, out);
    }

    if let Some(load) = speed.load.get() {
        match &load.external_cpu {
            // §4.4a: the box was quiet when the row started (load1_before), and
            // nobody else's work ran during it (external cores). load1_after is
            // recorded but not tested: the lane's own threads raise it.
            Some(ext) => {
                // The pre-row gate, in whichever currency the row carries. A
                // row with a pre-row census is tested on it; a row from before
                // that field existed is tested on `load1_before`, exactly as
                // it was when it was admitted.
                match ext.external_cores_before {
                    Some(before) => {
                        if !(before.is_finite() && ext.before_window_s.is_some_and(|w| w > 0.0)) {
                            out.push(
                                "external_cores_before is not a finite core count over a \
                                 positive window"
                                    .into(),
                            );
                        } else if before > load.void_above {
                            out.push(format!(
                                "work outside the lane used {before:.2} cores in the census \
                                 before the row, over its void threshold {:.2}: §4.4a voids \
                                 and re-runs",
                                load.void_above
                            ));
                        }
                    }
                    None => {
                        if load.load1_before > load.void_above {
                            out.push(format!(
                                "load1 before the row was {:.2} against its void threshold \
                                 {:.2}: §4.4 voids and re-runs, it does not publish",
                                load.load1_before, load.void_above
                            ));
                        }
                    }
                }
                if !(ext.external_cores_during_row.is_finite() && ext.window_s > 0.0) {
                    out.push(
                        "external_cpu is not a finite core count over a positive window".into(),
                    );
                } else if ext.external_cores_during_row > load.void_above {
                    out.push(format!(
                        "work outside the lane's tree used {:.2} cores during the row, over \
                         its void threshold {:.2}: §4.4a voids and re-runs",
                        ext.external_cores_during_row, load.void_above
                    ));
                }
                if ext.own_pids.is_empty() {
                    out.push(
                        "external_cpu names no own pids: the lane would count as external".into(),
                    );
                }
            }
            None => {
                let worst = load.load1_before.max(load.load1_after);
                if worst > load.void_above {
                    out.push(format!(
                        "load1 reached {worst:.2} against the row's own void threshold \
                         {:.2}: §4.4 voids and re-runs, it does not publish",
                        load.void_above
                    ));
                }
            }
        }
        // The row's threshold must be one §4.4 allows. Without this a lane
        // that wrote its own number would be voided against that number and
        // nothing else; the source scan cannot see every spelling of one.
        if load.void_above != LOAD_VOID_ABOVE_SMALL_HOST
            && load.void_above != LOAD_VOID_ABOVE_LARGE_HOST
        {
            out.push(format!(
                "load.void_above is {} but §4.4 allows only {LOAD_VOID_ABOVE_SMALL_HOST} \
                 (<= {SMALL_HOST_MAX_CORES} cores) or {LOAD_VOID_ABOVE_LARGE_HOST}; take it \
                 from load_void_above",
                load.void_above
            ));
        }
        match (&load.gpu, lane) {
            (None, Lane::DenseGpu) => out.push(
                "a GPU row with no device census: GPU contention is invisible to the \
                 load average, and §4.4's addendum exists because the card was at 98% \
                 while the 1-minute load reported the CPU side only"
                    .to_string(),
            ),
            (Some(gpu), _) => {
                if !gpu.holders_before.is_empty() || !gpu.holders_after.is_empty() {
                    out.push(format!(
                        "the GPU was not ours: {:?} before, {:?} after",
                        gpu.holders_before, gpu.holders_after
                    ));
                }
                let util = gpu.util_before_pct.max(gpu.util_after_pct);
                if util > gpu.void_above_pct {
                    out.push(format!(
                        "GPU utilisation reached {util}% against the row's threshold {}%",
                        gpu.void_above_pct
                    ));
                }
                if gpu.void_above_pct != GPU_UTIL_VOID_PCT {
                    out.push(format!(
                        "gpu.void_above_pct is {} but §4.4 pins {GPU_UTIL_VOID_PCT}",
                        gpu.void_above_pct
                    ));
                }
            }
            (None, _) => {}
        }
    }

    for (label, arm) in [("ours", &speed.ours), ("competitor", &speed.competitor)] {
        if let Some(t) = arm.timing.get() {
            if t.repeats < MIN_REPEATS {
                out.push(format!(
                    "{label}.timing.repeats is {} : §4.4 fixes N = 5 (3 for rows over \
                     30 s), and under 3 there is no median-min spread to witness noise with",
                    t.repeats
                ));
            }
            if !(t.min_s.is_finite() && t.median_s.is_finite()) || t.min_s <= 0.0 {
                out.push(format!(
                    "{label}.timing is not a pair of positive finite seconds"
                ));
            } else if t.min_s > t.median_s {
                out.push(format!(
                    "{label}.timing.min_s {} exceeds median_s {}",
                    t.min_s, t.median_s
                ));
            }
            if !t.warmup_discarded {
                out.push(format!(
                    "{label}.timing.warmup_discarded is false: §4.4 fixes one untimed \
                     warm-up per (arm, row), paying allocator, rayon pool, NVRTC/JIT and \
                     Python import outside the timed region"
                ));
            }
            if !t.interleaved {
                out.push(format!(
                    "{label}.timing.interleaved is false: §4.4 fixes A/B/A/B so drift \
                     hits both arms"
                ));
            }
        }
        if let Some(fixed) = arm.fixed_cost_s.get() {
            if !fixed.is_finite() || *fixed < 0.0 {
                out.push(format!("{label}.fixed_cost_s is not a non-negative number"));
            }
        }
    }

    check_derivation(speed, out);
}

/// §4.3c: a value gate must be one a wrong implementation could fail.
///
/// The gate's job is to certify that both arms computed the same *answer*. It
/// does that by pinning values — so it only does it when the pinned values are
/// values something had to compute. A gate that pins nothing but `0.0` across
/// every observable it names is passed by a backend that computes nothing and
/// returns `0.0`, which is the A10 defect in its original form: a test that
/// cannot fail is not evidence.
///
/// The check reads `ours_value`, because our backend is the thing under test.
/// It accepts a scalar or any nesting of arrays and objects (sampling rows
/// carry a per-qubit `<Z_q>` vector), collects the numeric leaves, and refuses
/// when every leaf is the same member of [`DEGENERATE_GATE_VALUES`].
///
/// What it does **not** refuse: a gate with one degenerate value among others.
/// `[0, 1, -1]` discriminates — no constant passes it — and on a stabilizer
/// state that vector is free, since `<Z0>` is `0`, the Heisenberg image
/// `U Z0 U†` is `+1`, and its negation is `-1`. The fix for a degenerate gate
/// is to widen it, not to drop the row.
fn check_gate_discriminates(gate: &ValueGate, out: &mut Vec<String>) {
    let leaves = numeric_leaves(&gate.ours_value);
    if leaves.is_empty() {
        out.push(
            "value_gate.ours_value carries no number: a gate with nothing pinned is              passed by every implementation, correct or not (§4.3c)"
                .to_string(),
        );
        return;
    }
    let first = leaves[0];
    let constant = leaves
        .iter()
        .all(|v| (v - first).abs() <= DEGENERATE_GATE_WINDOW);
    if !constant {
        return;
    }
    let degenerate = DEGENERATE_GATE_VALUES
        .iter()
        .any(|d| (first - d).abs() <= DEGENERATE_GATE_WINDOW);
    if !degenerate {
        return;
    }
    out.push(format!(
        "value_gate pins the single degenerate value {first} across all {} of its          observables: a backend that computes nothing and returns {first} passes this          gate, so it is not evidence (A10, §4.3c). Widen the gate to a vector with at          least two distinct values — on a stabilizer state <Z0> is 0, the Heisenberg          image U Z0 U-dagger is +1, and its negation is -1, so [0, 1, -1] costs one          extra peek per arm",
        leaves.len()
    ));
}

/// Every number reachable in a gate value, in document order. Scalars, arrays
/// and objects all appear in `ours_value` across the lanes, so the walk is
/// generic rather than per-lane.
fn numeric_leaves(v: &serde_json::Value) -> Vec<f64> {
    let mut out = Vec::new();
    walk_numeric(v, &mut out);
    out
}

fn walk_numeric(v: &serde_json::Value, out: &mut Vec<f64>) {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                out.push(f);
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| walk_numeric(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| walk_numeric(x, out)),
        _ => {}
    }
}

fn check_derivation(speed: &SpeedRow, out: &mut Vec<String>) {
    let (Some(d), Some(ours), Some(theirs)) = (
        speed.derivation.get(),
        speed.ours.timing.get(),
        speed.competitor.timing.get(),
    ) else {
        return;
    };

    if d.tie_rule != TIE_RULE {
        out.push(
            "derivation.tie_rule is not the §4.6 rule verbatim: the rule was pinned \
             before any number existed, and a row carrying a different wording is the \
             threshold moving after the fact (§8)"
                .to_string(),
        );
    }
    if d.overhead_rule != OVERHEAD_RULE {
        out.push("derivation.overhead_rule is not the §4.7 rule verbatim".to_string());
    }
    if !close(d.win_ratio_threshold, WIN_RATIO_THRESHOLD) {
        out.push(format!(
            "derivation.win_ratio_threshold is {} but §4.6 pins {WIN_RATIO_THRESHOLD}",
            d.win_ratio_threshold
        ));
    }
    if !close(d.ours_min_median_s[0], ours.min_s) || !close(d.ours_min_median_s[1], ours.median_s) {
        out.push("derivation.ours_min_median_s does not match ours.timing".to_string());
    }
    if !close(d.competitor_min_median_s[0], theirs.min_s)
        || !close(d.competitor_min_median_s[1], theirs.median_s)
    {
        out.push("derivation.competitor_min_median_s does not match competitor.timing".to_string());
    }

    let ratio = theirs.min_s / ours.min_s;
    if !close(d.ratio_competitor_over_ours, ratio) {
        out.push(format!(
            "derivation.ratio_competitor_over_ours is {} but competitor.min / ours.min \
             is {ratio}",
            d.ratio_competitor_over_ours
        ));
    }
    if !close(speed.ratio_competitor_over_ours, ratio) {
        out.push(format!(
            "ratio_competitor_over_ours is {} but competitor.min / ours.min is {ratio}",
            speed.ratio_competitor_over_ours
        ));
    }

    let disjoint = ours.median_s < theirs.min_s || theirs.median_s < ours.min_s;
    if d.intervals_disjoint != disjoint {
        out.push(format!(
            "derivation.intervals_disjoint is {} but [{}, {}] and [{}, {}] are {}",
            d.intervals_disjoint,
            ours.min_s,
            ours.median_s,
            theirs.min_s,
            theirs.median_s,
            if disjoint { "disjoint" } else { "overlapping" }
        ));
    }
    let beyond = ratio >= WIN_RATIO_THRESHOLD || ratio <= 1.0 / WIN_RATIO_THRESHOLD;
    if d.ratio_beyond_threshold != beyond {
        out.push(format!(
            "derivation.ratio_beyond_threshold is {} but the ratio {ratio} is {}",
            d.ratio_beyond_threshold,
            if beyond { "past" } else { "inside" }
        ));
    }
    let want = if beyond && disjoint {
        if ratio > 1.0 {
            Classification::Win
        } else {
            Classification::Loss
        }
    } else {
        Classification::Tie
    };
    if speed.classification != want {
        out.push(format!(
            "classification is {:?} but the derivation inputs give {want:?} \
             (ratio {ratio:.4}, intervals {})",
            speed.classification,
            if disjoint { "disjoint" } else { "overlapping" }
        ));
    }

    if let Some(floor) = speed.competitor.per_call_floor_s.get() {
        if !close(d.competitor_per_call_floor_s, *floor) {
            out.push(
                "derivation.competitor_per_call_floor_s does not match \
                 competitor.per_call_floor_s"
                    .to_string(),
            );
        }
        if *floor > 0.0 {
            let multiple = theirs.min_s / floor;
            if !close(d.competitor_min_over_floor, multiple) {
                out.push(format!(
                    "derivation.competitor_min_over_floor is {} but competitor.min / \
                     floor is {multiple}",
                    d.competitor_min_over_floor
                ));
            }
            let dominated = multiple < OVERHEAD_FLOOR_MULTIPLE;
            if speed.overhead_dominated != dominated {
                out.push(format!(
                    "overhead_dominated is {} but the competitor's min is {multiple:.2}x \
                     its own per-call floor, and §4.7 annotates under {OVERHEAD_FLOOR_MULTIPLE}x",
                    speed.overhead_dominated
                ));
            }
        }
    }

    match (&speed.floor_check, d.floor_arm_min_s) {
        (Some(fc), Some(min_s)) => {
            if !close(fc.min_s, min_s) {
                out.push("derivation.floor_arm_min_s does not match floor_check.min_s".to_string());
            }
            let want = if theirs.min_s > fc.min_s {
                FloorVerdict::LoopIsTheFinding
            } else {
                FloorVerdict::Ok
            };
            if fc.verdict != want {
                out.push(format!(
                    "floor_check.verdict is {:?} but the competitor's min {} against the \
                     floor arm's {} gives {want:?}",
                    fc.verdict, theirs.min_s, fc.min_s
                ));
            }
        }
        (Some(_), None) => out.push(
            "floor_check is present but derivation.floor_arm_min_s is absent, so the \
             verdict cannot be re-derived from the row"
                .to_string(),
        ),
        (None, Some(_)) => out.push(
            "derivation.floor_arm_min_s is present with no floor_check to derive".to_string(),
        ),
        (None, None) => {}
    }
}
