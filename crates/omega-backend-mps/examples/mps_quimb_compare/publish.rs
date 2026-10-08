// SPDX-License-Identifier: Apache-2.0
//! Rows for this lane, written through `omega-emu-compare`'s writer.
//!
//! A speed row is emitted only when both dense gaps are within the schema's
//! 1e-10 relative gate. A truncating row that misses that gate is a void row
//! with the times, the gaps and both discarded-weight numbers in its context.
//! The writer will not accept the other way around, and this lane does not
//! add a field to make it.

use std::collections::BTreeMap;
use std::process::Command;

use omega_emu_compare::{
    Arm, CapabilityRow, Classification, Derivation, Direction, GateReference, Lane, LoadRecord,
    PathWitness, PathWitnesses, Precision, Quantity, Row, RowBody, RowWriter, SpeedArm, SpeedRow,
    Threads, Timing, Truncation, ValueGate, VoidRow, Witness, Workload, GATE_REL_F64,
    OVERHEAD_FLOOR_MULTIPLE, OVERHEAD_RULE, TIE_RULE, WIN_RATIO_THRESHOLD,
};
use serde_json::{json, Value};

use crate::quimb_lane::{
    blind_spot, classify_speed, relative_error, requires_bond_saturation, Artifact, Knob,
    SpeedClass,
};

/// What [`LanePublisher::open`] needs besides the quimb hello line.
pub struct PublishOpen {
    pub path: String,
    pub rev: String,
    pub hostgate_cap: String,
    /// §4.4's load threshold the rows were gated on, from
    /// `omega_emu_compare::load_void_above` for this host.
    pub void_above: f64,
    pub ours_floor_s: f64,
    pub quimb_floor_s: f64,
    pub ours_threads: u32,
    pub svd_delegate: String,
    pub knob: Knob,
}

pub struct LanePublisher {
    writer: RowWriter,
    rev: String,
    box_name: String,
    recorded: String,
    hostgate_cap: String,
    void_above: f64,
    ours_floor_s: f64,
    quimb_floor_s: f64,
    quimb_versions: BTreeMap<String, String>,
    quimb_threads: u32,
    quimb_blas: String,
    ours_threads: u32,
    svd_delegate: String,
    knob: Knob,
}

/// One measured shape, whether or not the value gate admits it as a speed row.
pub struct TimedFacts {
    pub artifact: Artifact,
    pub chi: usize,
    pub oracle: f64,
    pub ours_value: f64,
    pub quimb_value: f64,
    pub ours_bond: usize,
    pub quimb_bond: usize,
    pub ours_discarded: f64,
    pub ours_fidelity: f64,
    pub quimb_fidelity: f64,
    pub quimb_error: f64,
    pub ours_calls: usize,
    pub ours_cols: usize,
    pub ours_min: f64,
    pub ours_med: f64,
    pub quimb_min: f64,
    pub quimb_med: f64,
    pub repeats: u32,
    pub ours_warmup_s: f64,
    pub quimb_warmup_s: f64,
    pub load_before: f64,
    pub load_after: f64,
    pub cooldown_s: u32,
    /// §4.4a: CPU used outside the lane, before and during the row.
    pub external_cpu: Option<omega_emu_compare::ExternalCpu>,
}

impl LanePublisher {
    pub fn open(cfg: PublishOpen, hello: &Value) -> Result<Self, String> {
        if cfg.rev.len() != 40 || !cfg.rev.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!(
                "rev {:?} is not 40 hex; refusing to publish",
                cfg.rev
            ));
        }
        if !(cfg.ours_floor_s.is_finite() && cfg.ours_floor_s > 0.0) {
            return Err(format!(
                "our per-call floor {} is not a positive time",
                cfg.ours_floor_s
            ));
        }
        if !(cfg.quimb_floor_s.is_finite() && cfg.quimb_floor_s > 0.0) {
            return Err(format!(
                "quimb per-call floor {} is not a positive time",
                cfg.quimb_floor_s
            ));
        }
        let quimb_versions =
            string_map(hello.get("versions").ok_or("quimb hello has no versions")?)?;
        let quimb_threads = hello
            .get("threads")
            .and_then(Value::as_u64)
            .ok_or("quimb hello has no threads")? as u32;
        let quimb_blas = hello
            .get("blas")
            .and_then(Value::as_str)
            .ok_or("quimb hello has no blas")?
            .to_string();
        let box_name = Command::new("hostname")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or("hostname failed")?;
        let recorded = Command::new("date")
            .args(["+%F"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() == 10)
            .ok_or("date +%F failed")?;
        let writer = RowWriter::append(&cfg.path).map_err(|e| e.to_string())?;
        Ok(Self {
            writer,
            rev: cfg.rev,
            box_name,
            recorded,
            hostgate_cap: cfg.hostgate_cap,
            void_above: cfg.void_above,
            ours_floor_s: cfg.ours_floor_s,
            quimb_floor_s: cfg.quimb_floor_s,
            quimb_versions,
            quimb_threads,
            quimb_blas,
            ours_threads: cfg.ours_threads,
            svd_delegate: cfg.svd_delegate,
            knob: cfg.knob,
        })
    }

    pub fn write_speed(&mut self, facts: &TimedFacts) -> Result<(), String> {
        let row = self.speed_row(facts)?;
        self.writer.write_row(&row).map_err(|e| e.to_string())
    }

    pub fn write_void(&mut self, row_id: &str, reason: &str, context: Value) -> Result<(), String> {
        let void = VoidRow {
            row_id: row_id.to_string(),
            lane: Lane::Mps,
            phase: "E3".into(),
            recorded: self.recorded.clone(),
            reason: reason.to_string(),
            context,
        };
        self.writer.write_void(&void).map_err(|e| e.to_string())
    }

    pub fn write_capability(&mut self, facts: &TimedFacts) -> Result<(), String> {
        let row = self.capability_row(facts)?;
        self.writer.write_row(&row).map_err(|e| e.to_string())
    }

    fn speed_row(&self, facts: &TimedFacts) -> Result<Row, String> {
        let ours_gap = relative_error(facts.ours_value, facts.oracle);
        let quimb_gap = relative_error(facts.quimb_value, facts.oracle);
        if ours_gap > GATE_REL_F64 || quimb_gap > GATE_REL_F64 {
            return Err(
                "speed row refused: a dense gap past 1e-10 is a void row, not a speed row".into(),
            );
        }
        let ratio = facts.quimb_min / facts.ours_min;
        let (class, _) = classify_speed(
            facts.ours_min,
            facts.ours_med,
            facts.quimb_min,
            facts.quimb_med,
        );
        let classification = match class {
            SpeedClass::Win => Classification::Win,
            SpeedClass::Loss => Classification::Loss,
            SpeedClass::Tie => Classification::Tie,
        };
        let disjoint = facts.ours_med < facts.quimb_min || facts.quimb_med < facts.ours_min;
        let beyond = ratio >= WIN_RATIO_THRESHOLD || ratio <= 1.0 / WIN_RATIO_THRESHOLD;
        let multiple = facts.quimb_min / self.quimb_floor_s;
        Ok(Row {
            row_id: speed_id(facts),
            lane: Lane::Mps,
            phase: "E3".into(),
            recorded: self.recorded.clone(),
            box_name: Witness::present(self.box_name.clone()),
            git: Witness::present(omega_emu_compare::GitRev::matched(self.rev.clone())),
            circuit: self.workload(facts),
            precision: Precision {
                ours: "f64".into(),
                competitor: "f64".into(),
                matched: true,
                note: None,
            },
            body: RowBody::Speed(Box::new(SpeedRow {
                ours: self.speed_arm(facts, true)?,
                competitor: self.speed_arm(facts, false)?,
                context_arms: BTreeMap::new(),
                value_gate: Witness::present(self.gate(facts, ours_gap, quimb_gap, true)),
                load: Witness::present(self.load(facts)),
                ratio_competitor_over_ours: ratio,
                classification,
                overhead_dominated: multiple < OVERHEAD_FLOOR_MULTIPLE,
                floor_check: None,
                derivation: Witness::present(Derivation {
                    tie_rule: TIE_RULE.to_string(),
                    win_ratio_threshold: WIN_RATIO_THRESHOLD,
                    ours_min_median_s: [facts.ours_min, facts.ours_med],
                    competitor_min_median_s: [facts.quimb_min, facts.quimb_med],
                    ratio_competitor_over_ours: ratio,
                    intervals_disjoint: disjoint,
                    ratio_beyond_threshold: beyond,
                    overhead_rule: OVERHEAD_RULE.to_string(),
                    competitor_per_call_floor_s: self.quimb_floor_s,
                    competitor_min_over_floor: multiple,
                    floor_arm_min_s: None,
                }),
            })),
            notes: self.notes(facts, ours_gap, quimb_gap),
        })
    }

    fn capability_row(&self, facts: &TimedFacts) -> Result<Row, String> {
        let claim = "Our MPS attaches an accumulated discarded-weight certificate \
            (the per-split relative discarded singular-value weight the refusal ceiling \
            is written in) alongside a fidelity_estimate. Both are estimates: this MPS is \
            non-canonical, and discarded_weight is not a proven bound on |Δ⟨O⟩|. quimb \
            CircuitMPS reports fidelity_estimate and error_estimate, which its docstring \
            describes as estimates from the state norm, and does not report that \
            accumulated discarded-weight certificate. This is a capability fact, not a speed result."
            .to_string();
        let evidence = format!(
            "14q depth 12 chi {}: ours discarded_weight={:.6e} fidelity_estimate={:.6e}; \
             quimb fidelity_estimate={:.6e} error_estimate={:.6e}; the quimb arm's JSON \
             has no discarded_weight field",
            facts.chi,
            facts.ours_discarded,
            facts.ours_fidelity,
            facts.quimb_fidelity,
            facts.quimb_error
        );
        let refusal = "quimb CircuitMPS.fidelity_estimate and error_estimate are norm-based \
            estimates (quimb's docstring). The arm returns those two numbers and no \
            accumulated discarded-weight certificate."
            .to_string();
        Ok(Row {
            row_id: "mps-capability-discarded-weight".into(),
            lane: Lane::Mps,
            phase: "E3".into(),
            recorded: self.recorded.clone(),
            box_name: Witness::present(self.box_name.clone()),
            git: Witness::present(omega_emu_compare::GitRev::matched(self.rev.clone())),
            circuit: self.workload(facts),
            precision: Precision {
                ours: "f64".into(),
                competitor: "f64".into(),
                matched: true,
                note: None,
            },
            body: RowBody::Capability(Box::new(CapabilityRow {
                direction: Direction::OursOnly,
                claim,
                evidence: Witness::present(evidence),
                ours: self.identity_arm(facts, true, false)?,
                competitor: self.identity_arm(facts, false, false)?,
                refusal: Witness::present(refusal),
            })),
            notes: vec![
                "capability, not a speed result".into(),
                "fidelity_estimate on both arms is an estimate; discarded_weight is not a proven |Δ⟨O⟩| bound".into(),
                blind_spot(facts.artifact.built.n, facts.artifact.built.depth, facts.chi).into(),
            ],
        })
    }

    fn workload(&self, facts: &TimedFacts) -> Workload {
        Workload {
            id: facts.artifact.id.clone(),
            file: facts.artifact.file.clone(),
            sha256: Witness::present(facts.artifact.sha256.clone()),
            generator: facts.artifact.generator.clone(),
            seed: None,
            qubits: facts.artifact.built.n,
            depth: facts.artifact.built.depth as u32,
            chi: Some(facts.chi as u32),
            shots: None,
            quantity: Quantity::Expectation,
            observable: Some("Z0".into()),
        }
    }

    fn gate(&self, facts: &TimedFacts, ours_gap: f64, quimb_gap: f64, passed: bool) -> ValueGate {
        let mut versions = BTreeMap::new();
        versions.insert("omega-backend-statevector".into(), "0.1.0".into());
        ValueGate {
            tolerance_rel: GATE_REL_F64,
            reference: GateReference::Oracle {
                arm: "omega-backend-statevector".into(),
                versions,
                value: json!(facts.oracle),
            },
            ours_value: json!(facts.ours_value),
            competitor_value: json!(facts.quimb_value),
            ours_gap,
            competitor_gap: quimb_gap,
            truncation: Some(Truncation {
                ours_discarded_weight: facts.ours_discarded,
                ours_fidelity_estimate: Some(facts.ours_fidelity),
                competitor_fidelity_estimate: Some(facts.quimb_fidelity),
            }),
            passed,
        }
    }

    fn load(&self, facts: &TimedFacts) -> LoadRecord {
        LoadRecord {
            load1_before: facts.load_before,
            load1_after: facts.load_after,
            void_above: self.void_above,
            cooldown_s: Some(facts.cooldown_s),
            hostgate_cap: Some(self.hostgate_cap.clone()),
            gpu: None,
            external_cpu: facts.external_cpu.clone(),
        }
    }

    fn notes(&self, facts: &TimedFacts, ours_gap: f64, quimb_gap: f64) -> Vec<String> {
        let n = facts.artifact.built.n;
        let depth = facts.artifact.built.depth;
        let mut notes = vec![
            format!(
                "dense absolute gap ours {:.6e} quimb {:.6e}; relative gap ours {:.6e} quimb {:.6e}",
                (facts.ours_value - facts.oracle).abs(),
                (facts.quimb_value - facts.oracle).abs(),
                ours_gap,
                quimb_gap
            ),
            format!(
                "quimb error_estimate {:.6e} (norm-based estimate, not a schema field)",
                facts.quimb_error
            ),
            format!(
                "quimb cutoff {} mode {} selected against the dense oracle before timing",
                self.knob.cutoff,
                self.knob.mode.as_str()
            ),
            blind_spot(n, depth, facts.chi).into(),
        ];
        if depth < 12 {
            notes.push(
                "depth-4 row is directional only and is never a headline, whatever the classification".into(),
            );
        }
        if n == 14 && facts.chi >= 128 {
            notes.push(
                "14q at chi >= 128 sits outside MPS's useful regime and is kept only because the grid pinned it".into(),
            );
        }
        if requires_bond_saturation(depth) {
            notes.push(
                "akilles 32-core replicate not run; this depth is SVD-dominated, which is where core topology can invert the ranking".into(),
            );
        }
        notes
    }

    fn speed_arm(&self, facts: &TimedFacts, ours: bool) -> Result<SpeedArm, String> {
        let (min_s, med_s, fixed) = if ours {
            (facts.ours_min, facts.ours_med, facts.ours_warmup_s)
        } else {
            (facts.quimb_min, facts.quimb_med, facts.quimb_warmup_s)
        };
        // One source for this predicate. It was inlined here and at the note
        // above while `requires_bond_saturation` already existed and was the
        // copy the test pinned, so a mutation here reddened nothing: the test
        // was watching a different line than the publisher read.
        let require_saturation = requires_bond_saturation(facts.artifact.built.depth);
        Ok(SpeedArm {
            arm: self.identity_arm(facts, ours, require_saturation)?,
            timing: Witness::present(Timing {
                min_s,
                median_s: med_s,
                repeats: facts.repeats,
                warmup_discarded: true,
                interleaved: true,
            }),
            fixed_cost_s: Witness::present(fixed),
            per_call_floor_s: Witness::present(if ours {
                self.ours_floor_s
            } else {
                self.quimb_floor_s
            }),
        })
    }

    fn identity_arm(
        &self,
        facts: &TimedFacts,
        ours: bool,
        require_saturation: bool,
    ) -> Result<Arm, String> {
        if ours {
            if facts.ours_calls == 0 {
                return Err(
                    "our SVD call count is 0; not publishing an arm that did not run it".into(),
                );
            }
            if require_saturation
                && (facts.ours_bond != facts.chi || facts.ours_cols != 2 * facts.chi)
            {
                return Err("our bond or SVD width missed the cap; not a speed arm".into());
            }
            let mut path = vec![
                witness("svd_call_count", facts.ours_calls.to_string()),
                witness(
                    "expectation_not_to_statevector",
                    "MpsBackend::with_svd_fn+expectation",
                ),
            ];
            if require_saturation {
                path.push(witness("max_bond", facts.ours_bond.to_string()));
                path.push(witness("svd_max_cols", facts.ours_cols.to_string()));
            }
            let mut knobs = BTreeMap::new();
            knobs.insert("chi".into(), facts.chi.to_string());
            knobs.insert("adaptive".into(), "off".into());
            knobs.insert("sigma_floor".into(), "1e-14".into());
            knobs.insert("max_discarded_weight".into(), "inf".into());
            knobs.insert("svd_certificate".into(), "custom".into());
            knobs.insert("svd_delegate".into(), self.svd_delegate.clone());
            let mut versions = BTreeMap::new();
            versions.insert(
                "omega-backend-mps".into(),
                env!("CARGO_PKG_VERSION").to_string(),
            );
            versions.insert("rustc".into(), rustc_version());
            Ok(Arm {
                name: "omega-backend-mps".into(),
                versions: Witness::present(versions),
                threads: Witness::present(Threads {
                    threads: self.ours_threads,
                    blas: format!("{} (certificate custom; shim delegates)", self.svd_delegate),
                }),
                knobs,
                executed_path: PathWitnesses::new(path),
            })
        } else {
            if require_saturation && facts.quimb_bond != facts.chi {
                return Err("quimb bond missed the cap; not a speed arm".into());
            }
            let mut path = vec![
                witness("timed_path", "CircuitMPS.apply_gate+local_expectation"),
                witness("to_dense_called", "false"),
            ];
            if require_saturation {
                path.push(witness("quimb_max_bond", facts.quimb_bond.to_string()));
            }
            let mut knobs = BTreeMap::new();
            knobs.insert("max_bond".into(), facts.chi.to_string());
            knobs.insert("cutoff".into(), format!("{}", self.knob.cutoff));
            knobs.insert("cutoff_mode".into(), self.knob.mode.as_str().into());
            knobs.insert("renorm".into(), "false".into());
            knobs.insert("dtype".into(), "complex128".into());
            knobs.insert("normalized".into(), "true".into());
            knobs.insert(
                "thread_env".into(),
                "OMP_NUM_THREADS,VECLIB_MAXIMUM_THREADS,OPENBLAS_NUM_THREADS,NUMBA_NUM_THREADS unset".into(),
            );
            Ok(Arm {
                name: "quimb-CircuitMPS".into(),
                versions: Witness::present(self.quimb_versions.clone()),
                threads: Witness::present(Threads {
                    threads: self.quimb_threads,
                    blas: self.quimb_blas.clone(),
                }),
                knobs,
                executed_path: PathWitnesses::new(path),
            })
        }
    }
}

pub fn speed_id(facts: &TimedFacts) -> String {
    format!("mps-{}-chi{}", facts.artifact.id, facts.chi)
}

pub fn measured_context(facts: &TimedFacts, kind: &str) -> Value {
    let ours_gap = relative_error(facts.ours_value, facts.oracle);
    let quimb_gap = relative_error(facts.quimb_value, facts.oracle);
    json!({
        "kind": kind,
        "not_a_speed_classification": true,
        "n": facts.artifact.built.n,
        "depth": facts.artifact.built.depth,
        "chi": facts.chi,
        "circuit_id": facts.artifact.id,
        "circuit_sha256": facts.artifact.sha256,
        "oracle": facts.oracle,
        "ours_value": facts.ours_value,
        "quimb_value": facts.quimb_value,
        "ours_gap_rel": ours_gap,
        "quimb_gap_rel": quimb_gap,
        "ours_gap_abs": (facts.ours_value - facts.oracle).abs(),
        "quimb_gap_abs": (facts.quimb_value - facts.oracle).abs(),
        "ours_discarded_weight": facts.ours_discarded,
        "ours_fidelity_estimate": facts.ours_fidelity,
        "quimb_fidelity_estimate": facts.quimb_fidelity,
        "quimb_error_estimate": facts.quimb_error,
        "ours_max_bond": facts.ours_bond,
        "quimb_max_bond": facts.quimb_bond,
        "ours_min_s": facts.ours_min,
        "ours_median_s": facts.ours_med,
        "quimb_min_s": facts.quimb_min,
        "quimb_median_s": facts.quimb_med,
        "repeats": facts.repeats,
        "load_before": facts.load_before,
        "load_after": facts.load_after,
        "blind_spot": blind_spot(facts.artifact.built.n, facts.artifact.built.depth, facts.chi),
    })
}

fn witness(name: &str, observed: impl ToString) -> PathWitness {
    PathWitness {
        name: name.to_string(),
        observed: observed.to_string(),
        asserted: true,
    }
}

fn string_map(v: &Value) -> Result<BTreeMap<String, String>, String> {
    let obj = v.as_object().ok_or("versions is not an object")?;
    let mut out = BTreeMap::new();
    for (k, val) in obj {
        let s = val
            .as_str()
            .ok_or_else(|| format!("version {k} is not a string"))?;
        out.insert(k.clone(), s.to_string());
    }
    if out.is_empty() {
        return Err("versions map is empty".into());
    }
    Ok(out)
}

fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}
