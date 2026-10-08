// SPDX-License-Identifier: Apache-2.0
//! Build a schema row from measured numbers. Classification, the ratio,
//! and the overhead flag are computed here from the timings, so a caller
//! cannot tag a loss as a win after seeing it.

use std::collections::BTreeMap;

use omega_emu_compare::{
    Arm, Classification, Derivation, Direction, Lane, LoadRecord, PathWitness, PathWitnesses,
    Precision, Quantity, Row, RowBody, SpeedArm, SpeedRow, Threads, Timing, ValueGate, Witness,
    Workload, OVERHEAD_FLOOR_MULTIPLE, OVERHEAD_RULE, TIE_RULE, WIN_RATIO_THRESHOLD,
};

use crate::gate::classify;

pub struct Derived {
    pub classification: Classification,
    pub ratio: f64,
    pub overhead_dominated: bool,
    pub derivation: Derivation,
}

pub fn derive(ours: &Timing, stim: &Timing, floor_s: f64) -> Derived {
    let ratio = stim.min_s / ours.min_s;
    let multiple = stim.min_s / floor_s;
    Derived {
        classification: classify(ratio, ours.min_s, ours.median_s, stim.min_s, stim.median_s),
        ratio,
        overhead_dominated: floor_s > 0.0 && multiple < OVERHEAD_FLOOR_MULTIPLE,
        derivation: Derivation {
            tie_rule: TIE_RULE.to_string(),
            win_ratio_threshold: WIN_RATIO_THRESHOLD,
            ours_min_median_s: [ours.min_s, ours.median_s],
            competitor_min_median_s: [stim.min_s, stim.median_s],
            ratio_competitor_over_ours: ratio,
            intervals_disjoint: ours.median_s < stim.min_s || stim.median_s < ours.min_s,
            ratio_beyond_threshold: ratio >= WIN_RATIO_THRESHOLD
                || ratio <= 1.0 / WIN_RATIO_THRESHOLD,
            overhead_rule: OVERHEAD_RULE.to_string(),
            competitor_per_call_floor_s: floor_s,
            competitor_min_over_floor: multiple,
            floor_arm_min_s: None,
        },
    }
}

pub fn min_median(times: &[f64]) -> Result<(f64, f64), String> {
    if times.len() < 3 {
        return Err(format!("{} repeats, §4.4 requires at least 3", times.len()));
    }
    for t in times {
        if !t.is_finite() || *t <= 0.0 {
            return Err(format!("non-positive sample {t}"));
        }
    }
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    Ok((sorted[0], median))
}

pub fn timing(times: &[f64]) -> Result<Timing, String> {
    let (min_s, median_s) = min_median(times)?;
    Ok(Timing {
        min_s,
        median_s,
        repeats: times.len() as u32,
        warmup_discarded: true,
        interleaved: true,
    })
}

pub fn arm(
    name: &str,
    versions: BTreeMap<String, String>,
    threads: u32,
    blas: &str,
    knobs: BTreeMap<String, String>,
    witnesses: Vec<PathWitness>,
) -> Arm {
    Arm {
        name: name.to_string(),
        versions: Witness::present(versions),
        threads: Witness::present(Threads {
            threads,
            blas: blas.to_string(),
        }),
        knobs,
        executed_path: PathWitnesses::new(witnesses),
    }
}

pub fn speed_arm(
    who: Arm,
    times: &[f64],
    fixed_cost_s: f64,
    floor_s: f64,
) -> Result<SpeedArm, String> {
    if !fixed_cost_s.is_finite() || fixed_cost_s < 0.0 {
        return Err(format!(
            "fixed cost {fixed_cost_s} is not a non-negative time"
        ));
    }
    Ok(SpeedArm {
        arm: who,
        timing: Witness::present(timing(times)?),
        fixed_cost_s: Witness::present(fixed_cost_s),
        per_call_floor_s: Witness::present(floor_s),
    })
}

pub fn clifford_precision() -> Precision {
    Precision {
        ours: "f64".to_string(),
        competitor: "exact-integer".to_string(),
        matched: true,
        note: Some(
            "both arms are exact on this Clifford circuit; the value gate is integer \
             equality, stricter than the f64 1e-10 relative rule. Stim's number is \
             peek_observable_expectation (§4.3)."
                .to_string(),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn speed_row(
    row_id: &str,
    recorded: &str,
    git: omega_emu_compare::GitRev,
    circuit: Workload,
    ours: SpeedArm,
    competitor: SpeedArm,
    value_gate: ValueGate,
    load: LoadRecord,
    notes: Vec<String>,
) -> Result<Row, String> {
    let ours_t = ours.timing.get().ok_or("ours timing absent")?.clone();
    let stim_t = competitor.timing.get().ok_or("stim timing absent")?.clone();
    let floor = *competitor
        .per_call_floor_s
        .get()
        .ok_or("stim floor absent")?;
    let derived = derive(&ours_t, &stim_t, floor);
    Ok(Row {
        row_id: row_id.to_string(),
        lane: Lane::Stabilizer,
        phase: "E2".to_string(),
        recorded: recorded.to_string(),
        box_name: Witness::present(omega_emu_compare::host_name()?),
        git: Witness::present(git),
        circuit,
        precision: clifford_precision(),
        body: RowBody::Speed(Box::new(SpeedRow {
            ours,
            competitor,
            context_arms: BTreeMap::new(),
            value_gate: Witness::present(value_gate),
            load: Witness::present(load),
            ratio_competitor_over_ours: derived.ratio,
            classification: derived.classification,
            overhead_dominated: derived.overhead_dominated,
            floor_check: None,
            derivation: Witness::present(derived.derivation),
        })),
        notes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn capability_row(
    row_id: &str,
    recorded: &str,
    git: omega_emu_compare::GitRev,
    circuit: Workload,
    direction: Direction,
    claim: &str,
    evidence: &str,
    ours: Arm,
    competitor: Arm,
    refusal: &str,
    notes: Vec<String>,
    matched: bool,
    precision_note: &str,
) -> Result<Row, String> {
    Ok(Row {
        row_id: row_id.to_string(),
        lane: Lane::Stabilizer,
        phase: "E2".to_string(),
        recorded: recorded.to_string(),
        box_name: Witness::present(omega_emu_compare::host_name()?),
        git: Witness::present(git),
        circuit,
        precision: Precision {
            ours: "f64".to_string(),
            competitor: "exact-integer".to_string(),
            matched,
            note: Some(precision_note.to_string()),
        },
        body: RowBody::Capability(Box::new(omega_emu_compare::CapabilityRow {
            direction,
            claim: claim.to_string(),
            evidence: Witness::present(evidence.to_string()),
            ours,
            competitor,
            refusal: Witness::present(refusal.to_string()),
        })),
        notes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn workload(
    id: &str,
    file: &str,
    sha256: &str,
    generator: &str,
    qubits: u32,
    depth: u32,
    shots: Option<u32>,
    quantity: Quantity,
    observable: Option<&str>,
) -> Workload {
    Workload {
        id: id.to_string(),
        file: file.to_string(),
        sha256: Witness::present(sha256.to_string()),
        generator: generator.to_string(),
        seed: None,
        qubits,
        depth,
        chi: None,
        shots,
        quantity,
        observable: observable.map(|s| s.to_string()),
    }
}

pub fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

pub fn witness(name: &str, observed: &str) -> PathWitness {
    PathWitness {
        name: name.to_string(),
        observed: observed.to_string(),
        asserted: true,
    }
}
