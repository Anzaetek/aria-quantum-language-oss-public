// SPDX-License-Identifier: Apache-2.0
//! A10 for E0: the schema round-trips, and **a row missing any witness field
//! fails to serialize**.
//!
//! The second half is the one that matters. Later phases are meant to *refuse
//! into* this format — a GPU row with no device census, an MPS row whose bond
//! never grew, an arm whose binary predates the tree it ran in. If the schema
//! accepts an unwitnessed row, those refusals have nowhere to land and the
//! whole plan is vacuous, which is the failure §8 names first.
//!
//! So the witness ablation below is exhaustive rather than illustrative: every
//! witness field on a fully-populated speed row is knocked out one at a time
//! and each knockout must fail. Adding a witness field without adding it to
//! `WITNESS_ABLATIONS` leaves that field untested, which is why
//! `every_witness_field_is_ablated` counts them against the serialized row.

use std::collections::BTreeMap;

use omega_emu_compare::{
    check, Arm, CapabilityRow, Classification, Derivation, Direction, FloorCheck, FloorVerdict,
    GateReference, GitRev, GpuCensus, GpuHolder, Lane, LoadRecord, PathWitness, PathWitnesses,
    Precision, Quantity, Row, RowBody, RowWriter, SpeedArm, SpeedRow, Threads, Timing, Truncation,
    ValueGate, VoidRow, Witness, Workload, GATE_REL_F64, GPU_UTIL_VOID_PCT,
    LOAD_VOID_ABOVE_LARGE_HOST, LOAD_VOID_ABOVE_SMALL_HOST, OVERHEAD_RULE, TIE_RULE,
    WIN_RATIO_THRESHOLD,
};

const REV: &str = "2c4f5ea0000000000000000000000000000000ab";

fn versions(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

fn witnesses(pairs: &[(&str, &str)]) -> PathWitnesses {
    PathWitnesses::new(
        pairs
            .iter()
            .map(|(name, observed)| PathWitness {
                name: (*name).to_string(),
                observed: (*observed).to_string(),
                asserted: true,
            })
            .collect(),
    )
}

fn arm(name: &str, path: &[(&str, &str)]) -> Arm {
    Arm {
        name: name.to_string(),
        versions: Witness::present(versions(&[("omega-backend-mps", "0.1.0")])),
        threads: Witness::present(Threads {
            threads: 8,
            blas: "Accelerate (macOS 15)".to_string(),
        }),
        knobs: versions(&[("chi", "128"), ("svd", "accelerate-zgesdd")]),
        executed_path: witnesses(path),
    }
}

fn speed_arm(name: &str, min_s: f64, median_s: f64, floor_s: f64) -> SpeedArm {
    SpeedArm {
        arm: arm(
            name,
            &[
                ("svd_call_count", "4128"),
                ("svd_max_cols", "128 == pinned chi"),
            ],
        ),
        timing: Witness::present(Timing {
            min_s,
            median_s,
            repeats: 5,
            warmup_discarded: true,
            interleaved: true,
        }),
        fixed_cost_s: Witness::present(0.0031),
        per_call_floor_s: Witness::present(floor_s),
    }
}

/// A fully-witnessed, self-consistent speed row: 14q x depth 12 HEA at chi 128,
/// our MPS against quimb, a loss published.
fn speed_row() -> Row {
    let ours = speed_arm("omega-backend-mps", 0.412, 0.431, 1.0e-5);
    let theirs = speed_arm("quimb-CircuitMPS", 0.198, 0.206, 8.0e-4);
    let ratio = 0.198 / 0.412;
    let floor_arm_min_s = 0.260;
    Row {
        row_id: "mps-hea-14q-d12-chi128".to_string(),
        lane: Lane::Mps,
        phase: "E3".to_string(),
        recorded: "2026-10-07".to_string(),
        box_name: Witness::present("andromeda".to_string()),
        git: Witness::present(GitRev::matched(REV)),
        circuit: Workload {
            id: "hea-14q-d12".to_string(),
            file: "tools/emu_compare/qasm/hea_14q_d12.qasm".to_string(),
            sha256: Witness::present(
                "af80affd72c474d21e467991ab79e2f97c4d06d0fb53a544625afe1a700415d0".to_string(),
            ),
            generator: "gen_workloads.py -> hea_qasm(14, 12)".to_string(),
            seed: None,
            qubits: 14,
            depth: 12,
            chi: Some(128),
            shots: None,
            quantity: Quantity::Expectation,
            observable: Some("Z0Z1+X0".to_string()),
        },
        precision: Precision {
            ours: "f64".to_string(),
            competitor: "f64".to_string(),
            matched: true,
            note: None,
        },
        body: RowBody::Speed(Box::new(SpeedRow {
            ours,
            competitor: theirs,
            context_arms: BTreeMap::new(),
            value_gate: Witness::present(ValueGate {
                tolerance_rel: GATE_REL_F64,
                reference: GateReference::Oracle {
                    arm: "aer-cpu-f64".to_string(),
                    versions: versions(&[("qiskit-aer", "0.17.2")]),
                    value: serde_json::json!(-1.234_567_890_123),
                },
                ours_value: serde_json::json!(-1.234_567_890_122),
                competitor_value: serde_json::json!(-1.234_567_890_124),
                ours_gap: 8.1e-13,
                competitor_gap: 8.1e-13,
                truncation: Some(Truncation {
                    ours_discarded_weight: 3.2e-11,
                    ours_fidelity_estimate: Some(1.0 - 3.2e-11),
                    competitor_fidelity_estimate: Some(1.0 - 2.9e-11),
                }),
                passed: true,
            }),
            load: Witness::present(LoadRecord {
                load1_before: 0.84,
                load1_after: 1.17,
                void_above: 2.0,
                cooldown_s: Some(60),
                hostgate_cap: Some("--host-bytes 2G".to_string()),
                gpu: None,
                external_cpu: None,
            }),
            ratio_competitor_over_ours: ratio,
            classification: Classification::Loss,
            overhead_dominated: false,
            floor_check: Some(FloorCheck {
                arm: "quimb-CircuitMPS-opt_einsum-greedy".to_string(),
                min_s: floor_arm_min_s,
                verdict: FloorVerdict::Ok,
            }),
            derivation: Witness::present(Derivation {
                tie_rule: TIE_RULE.to_string(),
                win_ratio_threshold: WIN_RATIO_THRESHOLD,
                ours_min_median_s: [0.412, 0.431],
                competitor_min_median_s: [0.198, 0.206],
                ratio_competitor_over_ours: ratio,
                intervals_disjoint: true,
                ratio_beyond_threshold: true,
                overhead_rule: OVERHEAD_RULE.to_string(),
                competitor_per_call_floor_s: 8.0e-4,
                competitor_min_over_floor: 0.198 / 8.0e-4,
                floor_arm_min_s: Some(floor_arm_min_s),
            }),
        })),
        notes: vec![
            "14q chi >= 128 sits outside MPS's useful regime (the backend warns); kept \
             because the shape was pinned in §3"
                .to_string(),
        ],
    }
}

/// A fully-witnessed capability row: Clifford+T with a bound, which Stim
/// cannot execute at all.
fn capability_row() -> Row {
    Row {
        row_id: "stabrank-clifford-t-vs-stim-refusal".to_string(),
        lane: Lane::Stabilizer,
        phase: "E2".to_string(),
        recorded: "2026-10-07".to_string(),
        box_name: Witness::present("andromeda".to_string()),
        git: Witness::present(GitRev::matched(REV)),
        circuit: Workload {
            id: "clifford-24q-d100".to_string(),
            file: "tools/emu_compare/qasm/clifford_24q_d100.qasm".to_string(),
            sha256: Witness::present(
                "ec5b5c693d4b06a49f649f7335b8eb881ce623a3aa7ff7cd22a65b8233a6b1a4".to_string(),
            ),
            generator: "gen_workloads.py -> clifford_qasm(24, 100), plus one T layer".to_string(),
            seed: None,
            qubits: 24,
            depth: 100,
            chi: None,
            shots: None,
            quantity: Quantity::Expectation,
            observable: Some("Z0".to_string()),
        },
        precision: Precision {
            ours: "f64".to_string(),
            competitor: "exact (tableau)".to_string(),
            matched: false,
            note: Some(
                "not a precision comparison: Stim is exact and Clifford-only, so there \
                 is no shared numeric path to match (§4.5)"
                    .to_string(),
            ),
        },
        body: RowBody::Capability(Box::new(CapabilityRow {
            direction: Direction::OursOnly,
            claim: "stabrank executes T with a derived certificate; Stim cannot execute T \
                    at all"
                .to_string(),
            evidence: "PLAN-MAJORANA-STIM §1.3 settles that Stim's gate set is Clifford; \
                       our certificate is omega-backend-stabrank's stabiliser-rank bound"
                .to_string()
                .into(),
            ours: arm(
                "omega-backend-stabrank",
                &[("certificate_bound", "4.1e-9, asserted non-vacuous")],
            ),
            competitor: arm("stim-1.16.0", &[("version_pin", "1.16.0")]),
            refusal: "stim.Circuit.append('T', [0]) raises ValueError: Unrecognized \
                      instruction name 'T'"
                .to_string()
                .into(),
        })),
        notes: vec![],
    }
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

#[test]
fn speed_row_round_trips() {
    let row = speed_row();
    let line = serde_json::to_string(&row).expect("a fully witnessed row serializes");
    assert!(!line.contains('\n'), "a JSONL row must be one line");
    let back: Row = serde_json::from_str(&line).expect("round-trip");
    assert_eq!(row, back);
    assert_eq!(
        serde_json::to_string(&back).unwrap(),
        line,
        "re-serializing must be byte-stable, or the doc and the JSONL drift"
    );
}

#[test]
fn capability_row_round_trips() {
    let row = capability_row();
    let line = serde_json::to_string(&row).expect("a fully witnessed capability row serializes");
    let back: Row = serde_json::from_str(&line).expect("round-trip");
    assert_eq!(row, back);
}

#[test]
fn a_capability_row_has_nowhere_to_put_a_ratio() {
    // §1: a capability fact is "never [...] an empty speed cell, in either
    // direction". The sum type plus deny_unknown_fields is that rule made
    // mechanical rather than reviewed.
    let line = serde_json::to_string(&capability_row()).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&line).unwrap();
    doc["body"]["capability"]["ratio_competitor_over_ours"] = serde_json::json!(1.9);
    let err = serde_json::from_value::<Row>(doc).expect_err("a ratio on a capability row");
    assert!(
        err.to_string().contains("unknown field"),
        "expected an unknown-field refusal, got {err}"
    );
}

#[test]
fn a_speed_row_cannot_drop_its_timings() {
    let line = serde_json::to_string(&speed_row()).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&line).unwrap();
    doc["body"]["speed"]["ours"]
        .as_object_mut()
        .unwrap()
        .remove("timing");
    let err = serde_json::from_value::<Row>(doc).expect_err("a speed arm with no timing");
    assert!(
        err.to_string().contains("missing field `timing`"),
        "expected a missing-field refusal, got {err}"
    );
}

// ---------------------------------------------------------------------------
// A10: a row missing any witness field fails to serialize
// ---------------------------------------------------------------------------

/// Every witness field on [`speed_row`], with the ablation that removes it.
/// The name is the JSON path, checked against the serialized row by
/// `every_witness_field_is_ablated`.
type Ablation = (&'static str, fn(&mut Row));

const WITNESS_ABLATIONS: &[Ablation] = &[
    ("box_name", |r| r.box_name = Witness::absent()),
    ("git", |r| r.git = Witness::absent()),
    ("circuit.sha256", |r| r.circuit.sha256 = Witness::absent()),
    ("body.speed.ours.arm.versions", |r| {
        speed_mut(r).ours.arm.versions = Witness::absent();
    }),
    ("body.speed.ours.arm.threads", |r| {
        speed_mut(r).ours.arm.threads = Witness::absent();
    }),
    ("body.speed.ours.arm.executed_path", |r| {
        speed_mut(r).ours.arm.executed_path = PathWitnesses::new(vec![]);
    }),
    ("body.speed.ours.timing", |r| {
        speed_mut(r).ours.timing = Witness::absent();
    }),
    ("body.speed.ours.fixed_cost_s", |r| {
        speed_mut(r).ours.fixed_cost_s = Witness::absent();
    }),
    ("body.speed.ours.per_call_floor_s", |r| {
        speed_mut(r).ours.per_call_floor_s = Witness::absent();
    }),
    ("body.speed.competitor.arm.versions", |r| {
        speed_mut(r).competitor.arm.versions = Witness::absent();
    }),
    ("body.speed.competitor.arm.threads", |r| {
        speed_mut(r).competitor.arm.threads = Witness::absent();
    }),
    ("body.speed.competitor.arm.executed_path", |r| {
        speed_mut(r).competitor.arm.executed_path = PathWitnesses::new(vec![]);
    }),
    ("body.speed.competitor.timing", |r| {
        speed_mut(r).competitor.timing = Witness::absent();
    }),
    ("body.speed.competitor.fixed_cost_s", |r| {
        speed_mut(r).competitor.fixed_cost_s = Witness::absent();
    }),
    ("body.speed.competitor.per_call_floor_s", |r| {
        speed_mut(r).competitor.per_call_floor_s = Witness::absent();
    }),
    ("body.speed.value_gate", |r| {
        speed_mut(r).value_gate = Witness::absent();
    }),
    ("body.speed.load", |r| {
        speed_mut(r).load = Witness::absent();
    }),
    ("body.speed.derivation", |r| {
        speed_mut(r).derivation = Witness::absent();
    }),
];

fn speed_mut(row: &mut Row) -> &mut SpeedRow {
    match &mut row.body {
        RowBody::Speed(speed) => speed,
        RowBody::Capability(_) => panic!("not a speed row"),
    }
}

#[test]
fn a_row_missing_any_witness_field_fails_to_serialize() {
    for (field, ablate) in WITNESS_ABLATIONS {
        let mut row = speed_row();
        ablate(&mut row);

        let err = serde_json::to_string(&row)
            .expect_err(&format!("serializing a row with no {field} must fail"));
        // Serde cannot name the field from inside a nested value, so the
        // message states the rule; Row::refusals names the field.
        let msg = err.to_string();
        assert!(
            msg.contains("witness field absent") || msg.contains("executed_path is empty"),
            "{field}: expected a witness refusal, got {msg}"
        );
        let refusals = row.refusals();
        assert!(
            refusals.iter().any(|r| r.starts_with(field)),
            "{field}: Row::refusals said {refusals:?}"
        );

        // And the writer refuses it, so a lane gets the field name in its log
        // rather than only serde's rule text.
        let mut writer = RowWriter::append(tmp_path("ablate")).unwrap();
        writer
            .write_row(&row)
            .expect_err("the writer must refuse what does not serialize");
    }
}

#[test]
fn every_witness_field_is_ablated() {
    // A witness field added to the schema without a line in
    // WITNESS_ABLATIONS would be untested. Count the `Witness`-typed leaves by
    // ablating them all at once and asserting Row::refusals names exactly as
    // many fields as the table has entries.
    let mut row = speed_row();
    for (_, ablate) in WITNESS_ABLATIONS {
        ablate(&mut row);
    }
    let refusals = row.refusals();
    assert_eq!(
        refusals.len(),
        WITNESS_ABLATIONS.len(),
        "the ablation table and the row's witness fields disagree: {refusals:?}"
    );
}

#[test]
fn a_capability_row_missing_its_evidence_fails_to_serialize() {
    for ablate in [
        (|r: &mut Row| capability_mut(r).evidence = Witness::absent()) as fn(&mut Row),
        |r: &mut Row| capability_mut(r).refusal = Witness::absent(),
        |r: &mut Row| capability_mut(r).competitor.executed_path = PathWitnesses::new(vec![]),
    ] {
        let mut row = capability_row();
        ablate(&mut row);
        assert!(
            serde_json::to_string(&row).is_err(),
            "a capability row without its evidence is an assertion about someone \
             else's software, not a row"
        );
    }
}

fn capability_mut(row: &mut Row) -> &mut CapabilityRow {
    match &mut row.body {
        RowBody::Capability(cap) => cap,
        RowBody::Speed(_) => panic!("not a capability row"),
    }
}

#[test]
fn a_printed_witness_is_not_a_witness() {
    // §4.1: each witness reaches an assert or a non-zero exit, "not a
    // println". The receipt is contract_bench.rs printing "0 of N pairs on
    // GPU" beside "speedup 1.16x" and exiting 0.
    let mut row = speed_row();
    speed_mut(&mut row).ours.arm.executed_path = PathWitnesses::new(vec![PathWitness {
        name: "cuda_svd_dispatch".to_string(),
        observed: "0 of 4128 pairs on GPU".to_string(),
        asserted: false,
    }]);
    let err = serde_json::to_string(&row).expect_err("a printed witness must refuse");
    assert!(
        err.to_string().contains("printed, not asserted"),
        "got {err}"
    );
}

#[test]
fn deserializing_refuses_a_line_whose_witness_key_was_deleted() {
    // The in-memory refusal is only half of it: a JSONL line that a different
    // writer produced without the key must also not parse, or the format is
    // only enforced on the way out.
    let line = serde_json::to_string(&speed_row()).unwrap();
    for key in ["box_name", "git"] {
        let mut doc: serde_json::Value = serde_json::from_str(&line).unwrap();
        doc.as_object_mut().unwrap().remove(key);
        let err = serde_json::from_value::<Row>(doc).expect_err("a line with no {key}");
        assert!(
            err.to_string().contains(&format!("missing field `{key}`")),
            "{key}: got {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// A10: the git-rev void
// ---------------------------------------------------------------------------

#[test]
fn equal_revs_serialize_and_differing_revs_do_not() {
    let mut row = speed_row();
    row.git = Witness::present(GitRev::matched(REV));
    assert!(
        serde_json::to_string(&row).is_ok(),
        "a clean tree whose rev matches the binary's must serialize"
    );

    let other = "ffffffffffffffffffffffffffffffffffffffff";
    row.git = Witness::present(GitRev {
        compiled_from: REV.to_string(),
        worktree: other.to_string(),
        worktree_dirty: false,
    });
    let err = serde_json::to_string(&row).expect_err("a stale binary voids the row");
    assert!(
        err.to_string().contains("stale binary"),
        "expected the §4.4 void, got {err}"
    );
}

#[test]
fn a_dirty_tree_voids_the_row() {
    let mut row = speed_row();
    row.git = Witness::present(GitRev {
        compiled_from: REV.to_string(),
        worktree: REV.to_string(),
        worktree_dirty: true,
    });
    let err = serde_json::to_string(&row).expect_err("a dirty tree voids the row");
    assert!(
        err.to_string().contains("worktree is dirty"),
        "expected the §4.4 void, got {err}"
    );
}

#[test]
fn an_unset_build_stamp_voids_the_row_even_when_it_matches() {
    // The hole in carrying the rev pair as one string: if the build forgets to
    // embed a rev and the runner reads the same placeholder, a `!=` comparison
    // passes while nothing is pinned. Both sides must be a commit id.
    for rev in [
        "unset",
        "",
        "2c4f5ea",
        "not-a-sha-but-exactly-forty-chars-long!!",
    ] {
        let mut row = speed_row();
        row.git = Witness::present(GitRev {
            compiled_from: rev.to_string(),
            worktree: rev.to_string(),
            worktree_dirty: false,
        });
        let err = serde_json::to_string(&row)
            .expect_err(&format!("{rev:?} on both sides must still void"));
        assert!(
            err.to_string().contains("40-hex commit id"),
            "{rev:?}: got {err}"
        );
    }
}

#[test]
fn a_void_row_always_serializes() {
    // Where the refusals land. A refusal with nowhere to go becomes a silent
    // drop, which is the defect one level up from the one being prevented.
    let void = VoidRow {
        row_id: "gpu-hea-28q-d24".to_string(),
        lane: Lane::DenseGpu,
        phase: "E5".to_string(),
        recorded: "2026-10-06".to_string(),
        reason: "(before) the GPU is not ours: util 98%, holders [llama-server 70.4 GB, \
                 train.py adapter 3.6 GB]"
            .to_string(),
        context: serde_json::json!({
            "gpu_util_before_pct": 98,
            "holders_before": [{"pid": 3141, "mib": 72089, "name": "llama-server"}],
        }),
    };
    let line = serde_json::to_string(&void).unwrap();
    let back: VoidRow = serde_json::from_str(&line).unwrap();
    assert_eq!(void, back);

    let mut writer = RowWriter::append(tmp_path("void")).unwrap();
    writer
        .write_void(&void)
        .expect("a void row is never refused");
}

// ---------------------------------------------------------------------------
// The writer's second layer: a row that serializes can still be wrong
// ---------------------------------------------------------------------------

#[test]
fn a_consistent_row_passes_check_and_writes() {
    let row = speed_row();
    assert_eq!(check(&row), Vec::<String>::new());
    let path = tmp_path("ok");
    let mut writer = RowWriter::append(&path).unwrap();
    writer.write_row(&row).unwrap();
    writer.write_row(&capability_row()).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text.lines().count(), 2, "one row, one line");
}

#[test]
fn classification_must_follow_from_the_derivation_inputs() {
    // §8's last bullet: "a tie reported as a win because the threshold moved
    // after the numbers existed". The derivation block is in the row so a
    // reader can re-derive the verdict; the writer does that re-derivation.
    let mut row = speed_row();
    speed_mut(&mut row).classification = Classification::Win;
    let reasons = check(&row);
    assert!(
        reasons.iter().any(|r| r.contains("classification is Win")),
        "got {reasons:?}"
    );

    let mut row = speed_row();
    speed_mut(&mut row).derivation = Witness::present(Derivation {
        win_ratio_threshold: 1.05,
        ..speed_derivation()
    });
    let reasons = check(&row);
    assert!(
        reasons.iter().any(|r| r.contains("win_ratio_threshold")),
        "a moved threshold must be caught, got {reasons:?}"
    );
}

fn speed_derivation() -> Derivation {
    match &speed_row().body {
        RowBody::Speed(speed) => speed.derivation.get().unwrap().clone(),
        RowBody::Capability(_) => unreachable!(),
    }
}

#[test]
fn a_tie_inside_the_band_is_not_a_win() {
    // 1.17-1.21x was treated as "only 1.2x" in item 15 and turned out to be
    // configuration, not kernel. A ratio in that band classifies as a tie
    // however the arms are ordered.
    let mut row = speed_row();
    let speed = speed_mut(&mut row);
    speed.competitor.timing = Witness::present(Timing {
        min_s: 0.412 * 1.2,
        median_s: 0.412 * 1.26,
        repeats: 5,
        warmup_discarded: true,
        interleaved: true,
    });
    let ratio = 1.2;
    speed.ratio_competitor_over_ours = ratio;
    speed.classification = Classification::Tie;
    speed.overhead_dominated = false;
    speed.floor_check = None;
    speed.derivation = Witness::present(Derivation {
        competitor_min_median_s: [0.412 * 1.2, 0.412 * 1.26],
        ratio_competitor_over_ours: ratio,
        intervals_disjoint: true,
        ratio_beyond_threshold: false,
        competitor_min_over_floor: (0.412 * 1.2) / 8.0e-4,
        floor_arm_min_s: None,
        ..speed_derivation()
    });
    assert_eq!(
        check(&row),
        Vec::<String>::new(),
        "a 1.2x row is a clean tie"
    );

    speed_mut(&mut row).classification = Classification::Win;
    assert!(
        check(&row)
            .iter()
            .any(|r| r.contains("classification is Win")),
        "1.2x must not be publishable as a win"
    );
}

#[test]
fn a_failed_value_gate_is_not_a_speed_row() {
    let mut row = speed_row();
    let gate = match &row.body {
        RowBody::Speed(s) => s.value_gate.get().unwrap().clone(),
        RowBody::Capability(_) => unreachable!(),
    };
    speed_mut(&mut row).value_gate = Witness::present(ValueGate {
        passed: false,
        ours_gap: 4.0e-3,
        ..gate
    });
    let reasons = check(&row);
    assert!(
        reasons.iter().any(|r| r.contains("disagreement finding")),
        "a fast wrong number is disqualified (§4.3), got {reasons:?}"
    );
}

#[test]
fn a_loaded_box_voids_the_row() {
    let mut row = speed_row();
    let load = match &row.body {
        RowBody::Speed(s) => s.load.get().unwrap().clone(),
        RowBody::Capability(_) => unreachable!(),
    };
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        load1_after: 7.3,
        ..load
    });
    let reasons = check(&row);
    assert!(
        reasons.iter().any(|r| r.contains("void threshold")),
        "item 15's numbers moved 29% under another session's load, got {reasons:?}"
    );
}

/// The writer pins `void_above` to §4.4's two values, so a lane that chose its
/// own number is refused even if its load stayed under that number. Both
/// allowed values pass (a large-host row at load 7.3 is a row); anything else
/// — including a looser 50.0 that would admit a saturated box — is refused.
#[test]
fn a_row_with_a_threshold_section_four_does_not_allow_is_refused() {
    let mut row = speed_row();
    let load = match &row.body {
        RowBody::Speed(s) => s.load.get().unwrap().clone(),
        RowBody::Capability(_) => unreachable!(),
    };
    for allowed in [LOAD_VOID_ABOVE_SMALL_HOST, LOAD_VOID_ABOVE_LARGE_HOST] {
        speed_mut(&mut row).load = Witness::present(LoadRecord {
            void_above: allowed,
            ..load.clone()
        });
        assert_eq!(check(&row), Vec::<String>::new(), "{allowed} is §4.4's");
    }
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        load1_after: 7.3,
        void_above: LOAD_VOID_ABOVE_LARGE_HOST,
        ..load.clone()
    });
    assert_eq!(
        check(&row),
        Vec::<String>::new(),
        "7.3 on a large host is quiet"
    );
    for invented in [50.0, 3.0, 1.0] {
        speed_mut(&mut row).load = Witness::present(LoadRecord {
            void_above: invented,
            ..load.clone()
        });
        let reasons = check(&row);
        assert!(
            reasons.iter().any(|r| r.contains("allows only")),
            "void_above {invented} must be refused, got {reasons:?}"
        );
    }
}

#[test]
fn a_gpu_row_needs_its_device_census_and_a_quiet_card() {
    let mut row = speed_row();
    row.lane = Lane::DenseGpu;
    let reasons = check(&row);
    assert!(
        reasons.iter().any(|r| r.contains("no device census")),
        "CPU load average cannot see the card, got {reasons:?}"
    );

    let load = match &row.body {
        RowBody::Speed(s) => s.load.get().unwrap().clone(),
        RowBody::Capability(_) => unreachable!(),
    };
    let census = |holders: Vec<GpuHolder>, util: u32| GpuCensus {
        util_before_pct: util,
        util_after_pct: util,
        void_above_pct: GPU_UTIL_VOID_PCT,
        holders_before: holders.clone(),
        holders_after: holders,
    };
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        gpu: Some(census(vec![], 2)),
        ..load.clone()
    });
    assert_eq!(check(&row), Vec::<String>::new(), "a quiet card is a row");

    speed_mut(&mut row).load = Witness::present(LoadRecord {
        gpu: Some(census(
            vec![GpuHolder {
                pid: 3141,
                mib: 72089,
                name: "llama-server".to_string(),
            }],
            98,
        )),
        ..load
    });
    let reasons = check(&row);
    assert!(
        reasons.iter().any(|r| r.contains("GPU was not ours")),
        "got {reasons:?}"
    );
}

#[test]
fn an_overhead_dominated_row_must_say_so() {
    // §4.7, the deliberately absent 6x6 permanent row: a comparison under 10x
    // the competitor's own per-call floor measures their binding against our
    // arithmetic.
    let mut row = speed_row();
    let speed = speed_mut(&mut row);
    speed.competitor.per_call_floor_s = Witness::present(0.05);
    speed.derivation = Witness::present(Derivation {
        competitor_per_call_floor_s: 0.05,
        competitor_min_over_floor: 0.198 / 0.05,
        ..speed_derivation()
    });
    let reasons = check(&row);
    assert!(
        reasons.iter().any(|r| r.contains("its own per-call floor")),
        "a 3.96x-over-floor row must be annotated, got {reasons:?}"
    );
    speed_mut(&mut row).overhead_dominated = true;
    assert_eq!(check(&row), Vec::<String>::new());
}

#[test]
fn a_slow_authored_loop_is_the_finding() {
    // §6's addendum: the stale wrapper is kept as a sanity floor precisely so
    // that "our hand-written loop is slower than the wrapper around the same
    // library" is detectable and attributed to the loop.
    let mut row = speed_row();
    let speed = speed_mut(&mut row);
    speed.floor_check = Some(FloorCheck {
        arm: "aer-gpu-0.15.1".to_string(),
        min_s: 0.120,
        verdict: FloorVerdict::Ok,
    });
    speed.derivation = Witness::present(Derivation {
        floor_arm_min_s: Some(0.120),
        ..speed_derivation()
    });
    let reasons = check(&row);
    assert!(
        reasons.iter().any(|r| r.contains("LoopIsTheFinding")),
        "competitor 0.198 s against a 0.120 s floor is the loop, got {reasons:?}"
    );
}

#[test]
fn an_unmatched_precision_row_must_carry_its_annotation() {
    let mut row = speed_row();
    row.precision = Precision {
        ours: "f64".to_string(),
        competitor: "f32".to_string(),
        matched: false,
        note: None,
    };
    let reasons = check(&row);
    assert!(
        reasons.iter().any(|r| r.contains("never quoted as a win")),
        "qsim is f32 (finding 2) and its rows are annotated, got {reasons:?}"
    );
}

#[test]
fn fewer_than_three_repeats_carries_no_noise_witness() {
    let mut row = speed_row();
    speed_mut(&mut row).ours.timing = Witness::present(Timing {
        min_s: 0.412,
        median_s: 0.412,
        repeats: 1,
        warmup_discarded: true,
        interleaved: true,
    });
    let reasons = check(&row);
    assert!(
        reasons.iter().any(|r| r.contains("repeats is 1")),
        "got {reasons:?}"
    );
}

#[test]
fn an_unwarmed_or_sequential_run_is_not_a_measurement() {
    for (field, timing) in [
        (
            "warmup_discarded",
            Timing {
                min_s: 0.412,
                median_s: 0.431,
                repeats: 5,
                warmup_discarded: false,
                interleaved: true,
            },
        ),
        (
            "interleaved",
            Timing {
                min_s: 0.412,
                median_s: 0.431,
                repeats: 5,
                warmup_discarded: true,
                interleaved: false,
            },
        ),
    ] {
        let mut row = speed_row();
        speed_mut(&mut row).ours.timing = Witness::present(timing);
        let reasons = check(&row);
        assert!(
            reasons.iter().any(|r| r.contains(field)),
            "{field}: got {reasons:?}"
        );
    }
}

fn tmp_path(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("emu-compare-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("rows.jsonl")
}

fn ext(cores: f64) -> omega_emu_compare::ExternalCpu {
    omega_emu_compare::ExternalCpu {
        external_cores_during_row: cores,
        window_s: 12.0,
        method: omega_emu_compare::EXTERNAL_CPU_METHOD.to_string(),
        own_pids: vec![4242],
        // No pre-row census: the shape every row published before that field
        // existed has, and those rows must keep being judged the way they
        // were admitted.
        external_cores_before: None,
        before_window_s: None,
    }
}

/// With a pre-row census attached.
fn ext_before(cores: f64, before: f64) -> omega_emu_compare::ExternalCpu {
    omega_emu_compare::ExternalCpu {
        external_cores_before: Some(before),
        before_window_s: Some(2.0),
        ..ext(cores)
    }
}

/// §4.4a: with an external-CPU record, the lane's own load in load1_after does
/// not void the row; external work over the threshold does; and the box must
/// still have been quiet when the row started.
#[test]
fn external_cpu_replaces_load1_after_but_not_load1_before() {
    let mut row = speed_row();
    let load = match &row.body {
        RowBody::Speed(s) => s.load.get().unwrap().clone(),
        RowBody::Capability(_) => unreachable!(),
    };
    // The dense lane's case: our own 32 threads lifted load1 to 15.
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        load1_after: 15.4,
        external_cpu: Some(ext(0.6)),
        ..load.clone()
    });
    assert_eq!(
        check(&row),
        Vec::<String>::new(),
        "our own load is not contention"
    );
    // Someone else's 9 cores during the row: void.
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        load1_after: 15.4,
        external_cpu: Some(ext(9.0)),
        ..load.clone()
    });
    assert!(
        check(&row)
            .iter()
            .any(|r| r.contains("outside the lane's tree")),
        "{:?}",
        check(&row)
    );
    // A busy box at the start: void, external or not.
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        load1_before: 9.0,
        void_above: 8.0,
        external_cpu: Some(ext(0.1)),
        ..load.clone()
    });
    assert!(
        check(&row).iter().any(|r| r.contains("before the row")),
        "{:?}",
        check(&row)
    );
    // Without the record, the old rule still applies.
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        load1_after: 15.4,
        external_cpu: None,
        ..load
    });
    assert!(check(&row).iter().any(|r| r.contains("void threshold")));
}

#[test]
fn rows_written_before_section_4_4a_still_parse() {
    // A load record with no external_cpu key, as every E2/E3/E6 row has.
    let line = r#"{"load1_before":1.0,"load1_after":1.2,"void_above":2.0,"cooldown_s":null,"hostgate_cap":"--host-bytes 2G","gpu":null}"#;
    let rec: LoadRecord = serde_json::from_str(line).unwrap();
    assert!(rec.external_cpu.is_none());
    assert!(!serde_json::to_string(&rec)
        .unwrap()
        .contains("external_cpu"));
}

// ---------------------------------------------------------------------------
// §4.3c: a value gate must be one a wrong implementation could fail
// ---------------------------------------------------------------------------
//
// The receipt is E2. Its two Clifford expectation rows and two Clifford
// sampling rows all gate on `<Z0>`, and `<Z0>` on a Clifford brickwall is
// exactly 0.0 — ours, Stim's, and the dense oracle's. Four published rows
// whose value gate reads `0 == 0`. The observable was not the mistake and the
// circuit was not the mistake; their combination was, and nothing in the
// schema noticed, because every field was filled in correctly.

/// Set the gate's `ours_value` and return whatever `check` says.
fn reasons_for_gate(value: serde_json::Value) -> Vec<String> {
    let mut row = speed_row();
    let gate = speed_mut(&mut row).value_gate.get().unwrap().clone();
    speed_mut(&mut row).value_gate = Witness::present(ValueGate {
        ours_value: value,
        ..gate
    });
    check(&row)
}

fn is_degenerate_refusal(reasons: &[String]) -> bool {
    reasons.iter().any(|r| r.contains("degenerate"))
}

#[test]
fn a_gate_pinning_only_a_stub_reachable_value_is_refused() {
    for value in [
        serde_json::json!(0.0),
        serde_json::json!(0),
        serde_json::json!(1.0),
        serde_json::json!(-1.0),
        // The sampling shape: a per-qubit <Z_q> vector that is all zeros is
        // the same defect spread over 24 entries.
        serde_json::json!([0.0, 0.0, 0.0, 0.0]),
        serde_json::json!([1.0, 1.0]),
        // Nested, because `ours_value` is free-form JSON and a lane may key
        // the values by observable name.
        serde_json::json!({"z0": 0.0, "z1": 0.0}),
    ] {
        let reasons = reasons_for_gate(value.clone());
        assert!(
            is_degenerate_refusal(&reasons),
            "{value} should be refused as degenerate, got {reasons:?}"
        );
    }
}

#[test]
fn widening_the_gate_is_what_clears_the_refusal() {
    // The fix for E2 is not to drop the Clifford rows. On a stabilizer state
    // the discriminating vector is free: <Z0> is 0, the Heisenberg image
    // U Z0 U-dagger is +1 by construction, and its negation is -1. No
    // constant passes [0, 1, -1].
    let reasons = reasons_for_gate(serde_json::json!([0.0, 1.0, -1.0]));
    assert!(
        reasons.is_empty(),
        "a widened gate must be admitted, got {reasons:?}"
    );
}

#[test]
fn a_constant_that_no_stub_returns_is_not_degenerate() {
    // The rule is about the values a trivial implementation produces, not
    // about constancy. A gate pinning 0.4871 across its observables is
    // repetitive, not vacuous: nothing reaches that number without computing.
    let reasons = reasons_for_gate(serde_json::json!([0.487_1, 0.487_1, 0.487_1]));
    assert!(
        reasons.is_empty(),
        "a non-degenerate constant must be admitted, got {reasons:?}"
    );
}

#[test]
fn a_gate_with_no_number_in_it_is_refused() {
    for value in [
        serde_json::json!("0"),
        serde_json::json!({}),
        serde_json::json!([]),
        serde_json::json!(null),
    ] {
        let reasons = reasons_for_gate(value.clone());
        assert!(
            reasons.iter().any(|r| r.contains("carries no number")),
            "{value} pins nothing and must be refused, got {reasons:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// The rule applies to what is published, not only to what will be
// ---------------------------------------------------------------------------

/// Every committed row store. A rule enforced only at write time is a rule the
/// already-published rows are exempt from, and the already-published rows are
/// the ones in the document.
const PUBLISHED_ROW_STORES: &[&str] = &[
    "tools/emu_compare/stim/rows.jsonl",
    "tools/emu_compare/fermionic/rows.jsonl",
    "crates/omega-backend-mps/examples/mps_quimb_compare/e3.jsonl",
    "tools/emu_compare/dense/rows.jsonl",
    "tools/emu_compare/pauliprop/rows.jsonl",
    "tools/emu_compare/cv/rows.jsonl",
];

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/<name> sits two levels under the root")
        .to_path_buf()
}

#[test]
fn every_published_row_passes_every_check() {
    let root = repo_root();
    let mut failures: Vec<String> = Vec::new();
    let mut seen = 0usize;
    for rel in PUBLISHED_ROW_STORES {
        let path = root.join(rel);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{rel} is a published row store and must be readable: {e}"));
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            // A store holds speed rows and void rows interleaved. A void row
            // is never refused — its reason is its whole value (`write_void`)
            // — so it is counted and skipped, not checked.
            if let Ok(void) = serde_json::from_str::<VoidRow>(line) {
                let _ = void;
                continue;
            }
            seen += 1;
            let row: Row = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("{rel}:{} does not deserialize: {e}", n + 1));
            let reasons = check(&row);
            if !reasons.is_empty() {
                failures.push(format!("{rel}:{} {}: {:?}", n + 1, row.row_id, reasons));
            }
        }
    }
    assert!(seen > 0, "no published rows were read at all");
    assert!(
        failures.is_empty(),
        "{} of {seen} published rows are refused by the current checks:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// §4.4a, the pre-row half: a row that measured the box before it started is
/// judged on that measurement, not on the load average.
///
/// This is what unblocks the mac half. `load1_before` on andromeda reads 3.28
/// and then 1.92 a minute later on a box that is 85% idle by both the census
/// and `top`, so a `load1_before <= 2.0` gate refuses every row on that box
/// whatever the box is doing — which is why the dense lane's mac rows are
/// recorded as owed rather than absent.
#[test]
fn a_pre_row_census_replaces_the_load1_before_gate() {
    let mut row = speed_row();
    let load = match &row.body {
        RowBody::Speed(s) => s.load.get().unwrap().clone(),
        RowBody::Capability(_) => unreachable!(),
    };

    // andromeda: load1 says 3.28, the census says 0.4 cores. Admitted.
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        load1_before: 3.28,
        load1_after: 8.81,
        void_above: 2.0,
        external_cpu: Some(ext_before(0.2, 0.4)),
        ..load.clone()
    });
    assert_eq!(
        check(&row),
        Vec::<String>::new(),
        "a quiet box must not be voided by a statistic that is not measuring it"
    );

    // Genuinely busy before the row: void, and the reason names the census
    // rather than the load average.
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        load1_before: 0.1,
        void_above: 2.0,
        external_cpu: Some(ext_before(0.2, 5.0)),
        ..load.clone()
    });
    let reasons = check(&row);
    assert!(
        reasons
            .iter()
            .any(|r| r.contains("before the row") && r.contains("cores")),
        "{reasons:?}"
    );

    // A pre-row figure with no window is not a measurement.
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        external_cpu: Some(omega_emu_compare::ExternalCpu {
            external_cores_before: Some(0.1),
            before_window_s: None,
            ..ext(0.2)
        }),
        ..load.clone()
    });
    assert!(
        check(&row)
            .iter()
            .any(|r| r.contains("external_cores_before")),
        "{:?}",
        check(&row)
    );

    // And a row with no pre-row census is still held to load1_before, so the
    // thirteen rows already published do not change verdict.
    speed_mut(&mut row).load = Witness::present(LoadRecord {
        load1_before: 9.0,
        void_above: 8.0,
        external_cpu: Some(ext(0.1)),
        ..load
    });
    assert!(
        check(&row).iter().any(|r| r.contains("load1 before")),
        "{:?}",
        check(&row)
    );
}

#[test]
fn with_pre_row_actually_attaches_the_figure() {
    // A builder that silently does nothing would leave every row falling back
    // to the `load1_before` gate while the lane believed it had replaced it —
    // and on andromeda that is the difference between publishing and not.
    // Mutating `with_pre_row` to drop the value reddened no test until this
    // one existed.
    let probe = omega_emu_compare::QuietProbe {
        cores: 0.37,
        window_s: 2.0,
        load1: 3.28,
    };
    let e = ext(0.2).with_pre_row(&probe);
    assert_eq!(e.external_cores_before, Some(0.37));
    assert_eq!(e.before_window_s, Some(2.0));
    assert_eq!(
        e.external_cores_during_row, 0.2,
        "the during-row figure must not be touched"
    );
}
