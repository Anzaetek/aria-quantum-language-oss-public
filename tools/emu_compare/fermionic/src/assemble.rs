// SPDX-License-Identifier: Apache-2.0
//! Build a schema row from measured numbers. A binding-dominated id cannot
//! be given a speed classification: [`crate::gate::admit_body`] runs before
//! the row is returned.

use std::collections::BTreeMap;

use omega_emu_compare::{
    Arm, CapabilityRow, Classification, Derivation, Direction, GateReference, Lane, LoadRecord,
    PathWitness, PathWitnesses, Precision, Quantity, Row, RowBody, SpeedArm, SpeedRow, Threads,
    Timing, ValueGate, Witness, Workload, GATE_REL_F64, OVERHEAD_FLOOR_MULTIPLE, OVERHEAD_RULE,
    TIE_RULE, WIN_RATIO_THRESHOLD,
};

use crate::gate::{admit_body, classify};

pub struct Derived {
    pub classification: Classification,
    pub ratio: f64,
    pub overhead_dominated: bool,
    pub derivation: Derivation,
}

pub fn derive(ours: &Timing, theirs: &Timing, floor_s: f64) -> Derived {
    let ratio = theirs.min_s / ours.min_s;
    let multiple = theirs.min_s / floor_s;
    Derived {
        classification: classify(
            ratio,
            ours.min_s,
            ours.median_s,
            theirs.min_s,
            theirs.median_s,
        ),
        ratio,
        overhead_dominated: floor_s > 0.0 && multiple < OVERHEAD_FLOOR_MULTIPLE,
        derivation: Derivation {
            tie_rule: TIE_RULE.to_string(),
            win_ratio_threshold: WIN_RATIO_THRESHOLD,
            ours_min_median_s: [ours.min_s, ours.median_s],
            competitor_min_median_s: [theirs.min_s, theirs.median_s],
            ratio_competitor_over_ours: ratio,
            intervals_disjoint: ours.median_s < theirs.min_s || theirs.median_s < ours.min_s,
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

pub fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

pub fn witness(name: &str, observed: impl Into<String>) -> PathWitness {
    PathWitness {
        name: name.to_string(),
        observed: observed.into(),
        asserted: true,
    }
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
    Ok(SpeedArm {
        arm: who,
        timing: Witness::present(timing(times)?),
        fixed_cost_s: Witness::present(fixed_cost_s),
        per_call_floor_s: Witness::present(floor_s),
    })
}

pub fn f64_precision(note: &str) -> Precision {
    Precision {
        ours: "f64".to_string(),
        competitor: "f64".to_string(),
        matched: true,
        note: Some(note.to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn workload(
    id: &str,
    file: &str,
    sha256: &str,
    generator: &str,
    seed: Option<u64>,
    qubits: u32,
    depth: u32,
    observable: &str,
) -> Workload {
    Workload {
        id: id.to_string(),
        file: file.to_string(),
        sha256: Witness::present(sha256.to_string()),
        generator: generator.to_string(),
        seed,
        qubits,
        depth,
        chi: None,
        shots: None,
        quantity: Quantity::Expectation,
        observable: Some(observable.to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn capability(
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
    precision: Precision,
    notes: Vec<String>,
) -> Result<Row, String> {
    let row = Row {
        row_id: row_id.to_string(),
        lane: Lane::Fermionic,
        phase: "E6".to_string(),
        recorded: recorded.to_string(),
        box_name: Witness::present(omega_emu_compare::host_name()?),
        git: Witness::present(git),
        circuit,
        precision,
        body: RowBody::Capability(Box::new(CapabilityRow {
            direction,
            claim: claim.to_string(),
            evidence: Witness::present(evidence.to_string()),
            ours,
            competitor,
            refusal: Witness::present(refusal.to_string()),
        })),
        notes,
    };
    admit_body(row_id, &row.body)?;
    Ok(row)
}

#[allow(clippy::too_many_arguments)]
pub fn speed(
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
    let theirs_t = competitor
        .timing
        .get()
        .ok_or("ffsim timing absent")?
        .clone();
    let floor = *competitor
        .per_call_floor_s
        .get()
        .ok_or("ffsim floor absent")?;
    let derived = derive(&ours_t, &theirs_t, floor);
    if !value_gate.passed {
        return Err(format!(
            "{row_id}: value gate failed; publish a void row, not a speed row"
        ));
    }
    let row = Row {
        row_id: row_id.to_string(),
        lane: Lane::Fermionic,
        phase: "E6".to_string(),
        recorded: recorded.to_string(),
        box_name: Witness::present(omega_emu_compare::host_name()?),
        git: Witness::present(git),
        circuit,
        precision: f64_precision(
            "both arms are f64. The gate is |Δ| / max(|reference|, 1) ≤ 1e-10, two-sided.",
        ),
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
    };
    admit_body(row_id, &row.body)?;
    Ok(row)
}

pub fn lucj_gate(ours: f64, theirs: f64) -> ValueGate {
    let scale = ours.abs().max(theirs.abs()).max(1.0);
    let gap = (ours - theirs).abs() / scale;
    ValueGate {
        tolerance_rel: GATE_REL_F64,
        reference: GateReference::ArmVsArm {
            note: "16 modes is under the 26q line, and a dense statevector of our QASM \
                   shares our lowering. The independent number is ffsim's fused UCJ, \
                   which does not read the file. Both gaps are |sector − ffsim| / scale. \
                   A separate statevector of the same QASM is asserted equal to the \
                   sector value and is not this reference."
                .to_string(),
        },
        ours_value: serde_json::json!(ours),
        competitor_value: serde_json::json!(theirs),
        ours_gap: gap,
        competitor_gap: gap,
        truncation: None,
        passed: gap <= GATE_REL_F64,
    }
}

#[cfg(test)]
mod tests {
    use super::{arm, capability, f64_precision, map, speed, speed_arm, witness, workload};
    use omega_emu_compare::{GitRev, LoadRecord, RowWriter};

    fn rev() -> GitRev {
        GitRev::matched("0123456789abcdef0123456789abcdef01234567")
    }

    fn who() -> omega_emu_compare::Arm {
        arm(
            "arm",
            map(&[("v", "1")]),
            1,
            "none",
            map(&[("k", "v")]),
            vec![witness("pin", "1"), witness("gate", "ok")],
        )
    }

    fn sha() -> String {
        "ab".repeat(32)
    }

    #[test]
    fn a_binding_id_serializes_as_capability_and_refuses_a_speed_body() {
        let row = capability(
            "ferm-kitaev-n2",
            "2026-10-07",
            rev(),
            workload(
                "kitaev-n2",
                "kitaev_n2.qasm",
                &sha(),
                "gen",
                None,
                2,
                1,
                "H",
            ),
            omega_emu_compare::Direction::OursOnly,
            "correctness, not a speed row",
            "both arms returned -1",
            who(),
            who(),
            "speed classification refused: a wall-clock measures call overhead",
            f64_precision("f64"),
            vec!["binding-dominated capability/correctness row".into()],
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("ferm-cap-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut writer = RowWriter::append(&path).unwrap();
        writer
            .write_row(&row)
            .expect("a capability row for kitaev n2 serializes");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"capability\""), "{text}");
        assert!(!text.contains("\"classification\""), "{text}");
        let _ = std::fs::remove_file(&path);

        let gate = super::lucj_gate(-1.0, -1.0);
        let err = speed(
            "ferm-kitaev-n2",
            "2026-10-07",
            rev(),
            workload(
                "kitaev-n2",
                "kitaev_n2.qasm",
                &sha(),
                "gen",
                None,
                2,
                1,
                "H",
            ),
            speed_arm(who(), &[1.0, 1.1, 1.2, 1.3, 1.4], 0.0, 0.01).unwrap(),
            speed_arm(who(), &[2.0, 2.1, 2.2, 2.3, 2.4], 0.0, 0.01).unwrap(),
            gate,
            LoadRecord {
                load1_before: 0.2,
                load1_after: 0.3,
                void_above: omega_emu_compare::load_void_above(10),
                cooldown_s: None,
                hostgate_cap: omega_emu_compare::hostgate_cap_from(Some("2G")).ok(),
                gpu: None,
                external_cpu: None,
            },
            Vec::new(),
        )
        .expect_err("speed body");
        assert!(err.contains("speed classification"), "{err}");
        assert!(err.contains("ferm-kitaev-n2"), "{err}");
    }

    #[test]
    fn committed_jsonl_keeps_binding_rows_unclassified() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("rows.jsonl");
        let text = std::fs::read_to_string(&path).expect("rows.jsonl is the published lane");
        let binding = [
            "ferm-kitaev-n2",
            "ferm-kitaev-n8",
            "ferm-hubbard2",
            "ferm-h2-ground",
        ];
        let mut seen = std::collections::BTreeSet::new();
        for line in text.lines().filter(|l| !l.is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).expect("jsonl");
            let id = v["row_id"].as_str().expect("row_id").to_string();
            let body = v["body"].as_object().expect("body");
            if binding.contains(&id.as_str()) {
                assert!(
                    body.contains_key("capability"),
                    "{id} published a speed body"
                );
                assert!(!body.contains_key("speed"), "{id}");
                assert!(
                    body["capability"].get("classification").is_none(),
                    "{id} carries a classification"
                );
            } else if id == "ferm-kitaev-n64" {
                assert!(body.contains_key("capability"), "{id}");
                assert!(!body.contains_key("speed"), "{id}");
            } else if id == "ferm-lucj16" {
                assert_eq!(body["speed"]["classification"], "loss", "{id}");
                assert_eq!(body["speed"]["value_gate"]["passed"], true);
                assert_eq!(body["speed"]["overhead_dominated"], true);
            } else {
                panic!("unexpected row {id}");
            }
            seen.insert(id);
        }
        for id in binding {
            assert!(seen.contains(id), "missing {id}");
        }
        assert!(seen.contains("ferm-kitaev-n64"));
        assert!(seen.contains("ferm-lucj16"));
    }
}
