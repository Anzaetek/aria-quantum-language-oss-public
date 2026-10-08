// SPDX-License-Identifier: Apache-2.0
//! Build schema rows from measured numbers. Classification, ratio and the
//! overhead flag are computed here from the timings, so a caller cannot tag a
//! loss as a win after seeing it.

use std::collections::BTreeMap;

use omega_emu_compare::{
    Arm, Derivation, GitRev, Lane, LoadRecord, PathWitness, PathWitnesses, Precision, Quantity,
    Row, RowBody, SpeedArm, SpeedRow, Threads, Timing, ValueGate, Witness, Workload,
    OVERHEAD_FLOOR_MULTIPLE, OVERHEAD_RULE, TIE_RULE, WIN_RATIO_THRESHOLD,
};

use crate::gate::classify;

pub const PHASE: &str = "E4";

/// Which competitor a row is against; decides the precision annotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Competitor {
    Qulacs,
    Qsim,
}

pub fn precision(c: Competitor) -> Precision {
    match c {
        Competitor::Qulacs => Precision {
            ours: "f64".into(),
            competitor: "f64".into(),
            matched: true,
            note: Some(
                "both arms complex128 (§4.5: our CPU statevector is f64, qulacs is f64)".into(),
            ),
        },
        Competitor::Qsim => Precision {
            ours: "f64".into(),
            competitor: "f32".into(),
            matched: false,
            note: Some(
                "qsimcirq 0.22.1 computes in complex64: measured on akilles against a \
                 complex128 reference, |1-F| 1.7e-8..6.3e-7 and phase-aligned max|dpsi| \
                 9e-9..8e-8, where a double path sits near 1e-14; QSimOptions has no \
                 precision knob. Per §4.5 this row is evidence when we LOSE and is never \
                 quoted as a win."
                    .into(),
            ),
        },
    }
}

pub fn timing(times: &[f64]) -> Result<Timing, String> {
    if times.len() < omega_emu_compare::MIN_REPEATS as usize {
        return Err(format!("{} repeats, §4.4 requires at least 3", times.len()));
    }
    let mut sorted = times.to_vec();
    for t in &sorted {
        if !t.is_finite() || *t <= 0.0 {
            return Err(format!("non-positive sample {t}"));
        }
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    Ok(Timing {
        min_s: sorted[0],
        median_s: median,
        repeats: n as u32,
        warmup_discarded: true,
        interleaved: true,
    })
}

pub fn arm(
    name: &str,
    versions: BTreeMap<String, String>,
    threads: u32,
    knobs: BTreeMap<String, String>,
    witnesses: Vec<PathWitness>,
) -> Arm {
    Arm {
        name: name.to_string(),
        versions: Witness::present(versions),
        threads: Witness::present(Threads {
            threads,
            blas: "none".to_string(),
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

pub fn derivation(ours: &Timing, comp: &Timing, floor_s: f64) -> (f64, bool, Derivation) {
    let ratio = comp.min_s / ours.min_s;
    let multiple = comp.min_s / floor_s;
    let d = Derivation {
        tie_rule: TIE_RULE.to_string(),
        win_ratio_threshold: WIN_RATIO_THRESHOLD,
        ours_min_median_s: [ours.min_s, ours.median_s],
        competitor_min_median_s: [comp.min_s, comp.median_s],
        ratio_competitor_over_ours: ratio,
        intervals_disjoint: ours.median_s < comp.min_s || comp.median_s < ours.min_s,
        ratio_beyond_threshold: ratio >= WIN_RATIO_THRESHOLD || ratio <= 1.0 / WIN_RATIO_THRESHOLD,
        overhead_rule: OVERHEAD_RULE.to_string(),
        competitor_per_call_floor_s: floor_s,
        competitor_min_over_floor: multiple,
        floor_arm_min_s: None,
    };
    (
        ratio,
        floor_s > 0.0 && multiple < OVERHEAD_FLOOR_MULTIPLE,
        d,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn speed_row(
    row_id: &str,
    recorded: &str,
    box_name: &str,
    git: GitRev,
    circuit: Workload,
    competitor_kind: Competitor,
    ours: SpeedArm,
    competitor: SpeedArm,
    value_gate: ValueGate,
    load: LoadRecord,
    notes: Vec<String>,
) -> Result<Row, String> {
    let o = ours.timing.get().ok_or("ours timing absent")?.clone();
    let c = competitor
        .timing
        .get()
        .ok_or("competitor timing absent")?
        .clone();
    let floor = *competitor
        .per_call_floor_s
        .get()
        .ok_or("competitor floor absent")?;
    let (ratio, overhead, d) = derivation(&o, &c, floor);
    Ok(Row {
        row_id: row_id.to_string(),
        lane: Lane::DenseCpu,
        phase: PHASE.to_string(),
        recorded: recorded.to_string(),
        box_name: Witness::present(box_name.to_string()),
        git: Witness::present(git),
        circuit,
        precision: precision(competitor_kind),
        body: RowBody::Speed(Box::new(SpeedRow {
            ours,
            competitor,
            context_arms: BTreeMap::new(),
            value_gate: Witness::present(value_gate),
            load: Witness::present(load),
            ratio_competitor_over_ours: ratio,
            classification: classify(ratio, o.min_s, o.median_s, c.min_s, c.median_s),
            overhead_dominated: overhead,
            floor_check: None,
            derivation: Witness::present(d),
        })),
        notes,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn workload(
    art: &crate::workload::Artifact,
    quantity: Quantity,
    shots: Option<u32>,
    observable: Option<&str>,
) -> Workload {
    Workload {
        id: art.id.clone(),
        file: art.file.clone(),
        sha256: Witness::present(art.sha256.clone()),
        generator: art.generator.clone(),
        seed: art.seed,
        qubits: art.qubits,
        depth: art.depth,
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
