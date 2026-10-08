// SPDX-License-Identifier: Apache-2.0
//! The GPU lane's measurement (PLAN-EMULATOR-COMPARISON §7 E5).
//!
//! Per cell:
//! 1. the Aer CPU oracle (f64) gives the observable's reference value and the
//!    reference final state, up to 26 qubits (§4.3); a degenerate reference
//!    refuses the cell (`emu_compare_dense::gate::nontrivial`);
//! 2. GATE STAGE, untimed as evidence: every candidate runs alone on the card
//!    (ours; cuStateVec unfused / fused / fused-f64; Aer-GPU unfused / fused /
//!    fused-f64), with a warm-up, a sweep of 3, the value gap and the state gate;
//! 3. ADMISSION: ours must pass; the competitor is cuStateVec's FASTEST variant
//!    that passes the gate (§8's "competitor's favour", bounded by §4.3); the
//!    Aer floor runs at the same fusion setting; f64 variants are controls and
//!    never admitted;
//! 4. THE ROW: device census before on an empty card; the three arms started;
//!    one discarded warm-up each; a per-pid device-memory witness; N = 5
//!    interleaved repeats (3 over 30 s); the arms quit; the card released; the
//!    device census after; the external-CPU census over the row. A row whose
//!    census voids it is re-run, and published as void after three attempts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use emu_compare_dense::assemble::{arm, speed_arm, witness, workload};
use emu_compare_dense::gate::{classify, nontrivial, rel_gap, spread, state_gate, state_tol};
use emu_compare_dense::workload::{load_artifact, observable_string, observable_terms, Artifact};
use omega_emu_compare::{
    CpuCensus, Derivation, FloorCheck, FloorVerdict, GateReference, GitRev, Lane, LoadRecord,
    PathWitness, Precision, Quantity, Row, RowBody, RowWriter, SpeedRow, ValueGate, VoidRow,
    Witness, GATE_REL_F32, GATE_REL_F64, OVERHEAD_FLOOR_MULTIPLE, OVERHEAD_RULE, TIE_RULE,
    WIN_RATIO_THRESHOLD,
};
use serde_json::{json, Value};

use crate::census;
use crate::client::{Engine, GpuArm, Variant};

pub const PHASE: &str = "E5";

/// The lane's cells, smallest first so a cap kill leaves the rows that finished.
pub const EXPECTATION_CELLS: &[&str] = &[
    "hea-24q-d24",
    "random1-24q-d20",
    "hea-26q-d24",
    "random1-26q-d20",
    "hea-28q-d24",
    "random1-28q-d20",
];
/// §4.3: a dense f64 oracle is affordable up to here; above, the gate is
/// arm-vs-arm (against the two f64 controls, which must agree with each other).
pub const ORACLE_MAX_QUBITS: u32 = 26;
const ORACLE_THREADS: u32 = 16;
const SWEEP_REPEATS: usize = 3;
const REPEATS: usize = 5;
const MAX_ATTEMPTS: u32 = 3;
const FLOOR_CALLS: usize = 5;
/// §4.1's negative-space tell: a GPU arm with no fixed startup cost anywhere
/// (process start + CUDA context + first-call excess) is not on a GPU.
pub const FIXED_COST_MIN_S: f64 = 0.05;
/// The two f64 controls must agree this closely to stand in for the oracle.
const F64_CONTROLS_AGREE: f64 = 1e-10;
/// How long a row waits for someone else's job to leave the card.
const CARD_WAIT: Duration = Duration::from_secs(600);

pub const CANDIDATES: &[Engine] = &[
    Engine::Ours,
    Engine::Custatevec(Variant::Unfused),
    Engine::Custatevec(Variant::Fused),
    Engine::Custatevec(Variant::FusedF64),
    Engine::AerGpu(Variant::Unfused),
    Engine::AerGpu(Variant::Fused),
    Engine::AerGpu(Variant::FusedF64),
];

/// What the gate stage learned about one candidate.
#[derive(Clone, Debug)]
pub struct Cand {
    pub engine: Engine,
    pub startup_s: f64,
    pub build_s: f64,
    pub first_s: f64,
    pub sweep: Vec<f64>,
    pub floor_s: f64,
    pub value: f64,
    pub value_gap: f64,
    pub state: Option<(f64, f64)>,
    pub passes: bool,
    pub why_not: Option<String>,
    pub versions: BTreeMap<String, String>,
    pub witnesses: Vec<PathWitness>,
}

impl Cand {
    pub fn sweep_min(&self) -> f64 {
        self.sweep.iter().cloned().fold(f64::INFINITY, f64::min)
    }
    pub fn fixed_cost_s(&self) -> f64 {
        self.startup_s + (self.first_s - self.sweep_min()).max(0.0)
    }
}

/// §8 bounded by §4.3: the fastest cuStateVec variant that passed, f64 never.
pub fn admit(cands: &[Cand]) -> Option<Variant> {
    cands
        .iter()
        .filter(|c| matches!(c.engine, Engine::Custatevec(v) if !v.is_f64()) && c.passes)
        .min_by(|a, b| a.sweep_min().partial_cmp(&b.sweep_min()).expect("finite"))
        .and_then(|c| match c.engine {
            Engine::Custatevec(v) => Some(v),
            _ => None,
        })
}

fn min_of(mut f: impl FnMut() -> Result<f64, String>) -> Result<f64, String> {
    let mut m = f64::INFINITY;
    for _ in 0..FLOOR_CALLS {
        m = m.min(f()?);
    }
    Ok(m)
}

struct Ctx {
    git: GitRev,
    recorded: String,
    box_name: String,
    device: String,
    void_above: f64,
    hostgate_cap: String,
    writer: RowWriter,
    tmp: PathBuf,
    oracle: GpuArm,
    oracle_versions: BTreeMap<String, String>,
}

pub fn append_void(out: &Path, row_id: &str, reason: &str) -> Result<(), String> {
    if !out.exists() {
        return Err(format!(
            "{} does not exist; nothing to append a void to",
            out.display()
        ));
    }
    let void = VoidRow {
        row_id: row_id.to_string(),
        lane: Lane::DenseGpu,
        phase: PHASE.to_string(),
        recorded: today()?,
        reason: reason.to_string(),
        context: json!({"source": "hostgate --watch killed the process tree"}),
    };
    let mut writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    writer.write_void(&void).map_err(|e| e.to_string())
}

pub fn measure(out: &Path, only: Option<&str>, append: bool) -> Result<(), String> {
    if out.exists() && !append {
        return Err(format!(
            "{} exists; refusing to append over a previous lane file",
            out.display()
        ));
    }
    let git = git_snap()?;
    let recorded = today()?;
    let box_name = omega_emu_compare::host_name()?;
    let cores = omega_emu_compare::host_cores()?;
    let void_above = omega_emu_compare::load_void_above(cores);
    let hostgate_cap = omega_emu_compare::hostgate_cap()?;
    let device = census::device_name()?;
    eprintln!(
        "gpu lane on {box_name} ({device}) at {recorded} rev {}; load void above {void_above} ({cores} cores); cap {hostgate_cap}",
        git.worktree
    );
    if let Some(o) = only {
        for x in o.split(',').map(str::trim) {
            if !EXPECTATION_CELLS.contains(&x) {
                return Err(format!("--only names unknown cell {x}"));
            }
        }
    }
    let tmp = std::env::temp_dir().join(format!("emu-gpu-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let oracle = GpuArm::launch(Engine::Oracle {
        threads: ORACLE_THREADS,
    })?;
    let oracle_versions = oracle.versions()?;
    let writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    let mut ctx = Ctx {
        git,
        recorded,
        box_name,
        device,
        void_above,
        hostgate_cap,
        writer,
        tmp,
        oracle,
        oracle_versions,
    };
    for id in EXPECTATION_CELLS {
        if only.is_none_or(|o| o.split(',').any(|x| x.trim() == *id)) {
            ctx.cell(id)?;
        }
    }
    let _ = std::fs::remove_dir_all(&ctx.tmp);
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn row_id(id: &str) -> String {
    format!("gpu-{id}-o-expectation-vs-custatevec")
}

/// Wait (bounded) until nobody else holds the card. The gate stage needs a
/// card to itself too: a shared card changes no value, but it can OOM an arm.
fn wait_card() -> Result<census::Before, String> {
    let start = Instant::now();
    loop {
        let b = census::before()?;
        if census::quiet(&b) {
            return Ok(b);
        }
        if start.elapsed() > CARD_WAIT {
            return Err(format!(
                "the card stayed busy for {} s: util {}%, holders {:?}",
                CARD_WAIT.as_secs(),
                b.util_pct,
                b.holders
            ));
        }
        eprintln!(
            "card not ours (util {}%, holders {:?}); waiting",
            b.util_pct, b.holders
        );
        std::thread::sleep(Duration::from_secs(30));
    }
}

impl Ctx {
    fn cell(&mut self, id: &str) -> Result<(), String> {
        let art = load_artifact(id)?;
        let n = art.qubits;
        let terms = observable_terms(n);
        eprintln!("BEGIN {id} ({n}q, depth {})", art.depth);

        // 1. Oracle.
        let reference = if n <= ORACLE_MAX_QUBITS {
            self.oracle.load(id, &art.path)?;
            let ref_npy = self.tmp.join(format!("{id}-aer.npy"));
            let r = self
                .oracle
                .call(json!({"op": "reference", "name": id, "terms": terms, "out": ref_npy}))?;
            let v = r["value"].as_f64().ok_or("oracle value missing")?;
            nontrivial(v)?;
            eprintln!("  oracle O = {v:+.15}");
            Some((v, ref_npy))
        } else {
            None
        };

        // 2. Gate stage.
        let mut cands = Vec::new();
        for &engine in CANDIDATES {
            wait_card()?;
            cands.push(self.candidate(
                engine,
                id,
                &art,
                &terms,
                reference.as_ref().map(|r| r.1.as_path()),
            )?);
        }
        let (ref_value, gate_ref) = match &reference {
            Some((v, _)) => (
                *v,
                GateReference::Oracle {
                    arm: "aer-cpu-f64".into(),
                    versions: self.oracle_versions.clone(),
                    value: json!(v),
                },
            ),
            None => {
                let f = |e: Engine| {
                    cands
                        .iter()
                        .find(|c| c.engine == e)
                        .map(|c| c.value)
                        .ok_or("missing f64 control")
                };
                let a = f(Engine::Custatevec(Variant::FusedF64))?;
                let b = f(Engine::AerGpu(Variant::FusedF64))?;
                if rel_gap(a, b) > F64_CONTROLS_AGREE {
                    return self.void(&row_id(id), &format!(
                        "disagreement finding: above the {ORACLE_MAX_QUBITS}q oracle line the two f64 controls must \
                         agree to {F64_CONTROLS_AGREE:e}, and cuStateVec-f64 {a:+.15} vs Aer-GPU-f64 {b:+.15} do not"));
                }
                (a, GateReference::ArmVsArm { note: format!(
                    "{n}q exceeds the {ORACLE_MAX_QUBITS}q dense-oracle line (§4.3): the reference is cuStateVec at f64 \
                     ({a:+.15}), which agrees with Aer-GPU at f64 ({b:+.15}) to {:.1e}; no state gate at this size",
                    rel_gap(a, b)) })
            }
        };
        for c in &mut cands {
            c.value_gap = rel_gap(c.value, ref_value);
            let tol = if c.engine.is_f64() {
                GATE_REL_F64
            } else {
                GATE_REL_F32
            };
            let mut why = Vec::new();
            if c.value_gap > tol {
                why.push(format!("value gap {:.3e} > {tol:.0e}", c.value_gap));
            }
            if let Some((omf, md)) = c.state {
                if let Err(e) = state_gate(&c.engine.label(), omf, md, state_tol(c.engine.is_f64()))
                {
                    why.push(e);
                }
            }
            if c.fixed_cost_s() < FIXED_COST_MIN_S && !matches!(c.engine, Engine::Oracle { .. }) {
                why.push(format!("fixed startup cost {:.3}s < {FIXED_COST_MIN_S}s: a GPU arm with no startup cost is suspect (§4.1)", c.fixed_cost_s()));
            }
            c.passes = why.is_empty();
            c.why_not = (!why.is_empty()).then(|| why.join("; "));
            eprintln!(
                "  {:28} sweep min {:.6e}s gap {:.3e} state {:?} fixed {:.3}s -> {}",
                c.engine.label(),
                c.sweep_min(),
                c.value_gap,
                c.state,
                c.fixed_cost_s(),
                c.why_not.as_deref().unwrap_or("passes")
            );
        }
        if let Some((_, p)) = &reference {
            let _ = std::fs::remove_file(p);
        }
        let sweep_table: Vec<String> = cands
            .iter()
            .map(|c| {
                format!(
                    "{}: sweep min {:.6e}s, gap {:.3e}{}, fixed {:.3}s, {}",
                    c.engine.label(),
                    c.sweep_min(),
                    c.value_gap,
                    c.state
                        .map(|(a, b)| format!(", |1-F| {a:.3e} max|dpsi| {b:.3e}"))
                        .unwrap_or_default(),
                    c.fixed_cost_s(),
                    c.why_not
                        .as_deref()
                        .map(|w| format!("REFUSED ({w})"))
                        .unwrap_or_else(|| "passes".into())
                )
            })
            .collect();

        // 3. Admission.
        let ours = cands
            .iter()
            .find(|c| c.engine == Engine::Ours)
            .cloned()
            .ok_or("no ours")?;
        if !ours.passes {
            return self.void(
                &row_id(id),
                &format!(
                    "disagreement finding: ours {}",
                    ours.why_not.unwrap_or_default()
                ),
            );
        }
        let Some(variant) = admit(&cands) else {
            return self.void(
                &row_id(id),
                &format!(
                    "disagreement finding: no cuStateVec variant passes the gate: {}",
                    sweep_table.join(" | ")
                ),
            );
        };

        // 4. The row.
        let mut last = String::new();
        for attempt in 1..=MAX_ATTEMPTS {
            match self.row(
                id,
                &art,
                &terms,
                ref_value,
                &gate_ref,
                &cands,
                variant,
                &sweep_table,
            )? {
                Ok(()) => return Ok(()),
                Err(why) => {
                    eprintln!("  {}: attempt {attempt}: {why}", row_id(id));
                    last = why;
                }
            }
        }
        self.void(
            &row_id(id),
            &format!("{MAX_ATTEMPTS} attempts, last: {last}"),
        )
    }

    fn candidate(
        &mut self,
        engine: Engine,
        id: &str,
        art: &Artifact,
        terms: &Value,
        ref_npy: Option<&Path>,
    ) -> Result<Cand, String> {
        let mut a = GpuArm::launch(engine)?;
        let build_s = a.load(id, &art.path)?;
        let (_, first_s) = a.expectation(id, terms)?;
        let mut sweep = Vec::new();
        let mut value = f64::NAN;
        for _ in 0..SWEEP_REPEATS {
            let (v, t) = a.expectation(id, terms)?;
            sweep.push(t);
            value = v;
        }
        let floor_s = min_of(|| a.floor())?;
        let state = match ref_npy {
            Some(r) => {
                let out = self.tmp.join(format!(
                    "{id}-{}.npy",
                    engine.label().replace(['[', ']'], "_")
                ));
                a.write_state(id, &out)?;
                let v = self
                    .oracle
                    .call(json!({"op": "compare", "ref": r, "other": out}))?;
                let _ = std::fs::remove_file(&out);
                Some((
                    v["one_minus_f_abs"]
                        .as_f64()
                        .ok_or("compare without |1-F|")?,
                    v["max_abs_diff"]
                        .as_f64()
                        .ok_or("compare without max|dpsi|")?,
                ))
            }
            None => None,
        };
        let versions = a.versions()?;
        let witnesses = a.witnesses()?;
        let startup_s = a.startup_s;
        a.quit()?;
        Ok(Cand {
            engine,
            startup_s,
            build_s,
            first_s,
            sweep,
            floor_s,
            value,
            value_gap: f64::NAN,
            state,
            passes: false,
            why_not: None,
            versions,
            witnesses,
        })
    }

    /// One attempt at the row. Outer `Err`: the lane cannot continue. Inner
    /// `Err`: this attempt was voided by a census and may be re-run.
    #[allow(clippy::too_many_arguments)]
    fn row(
        &mut self,
        id: &str,
        art: &Artifact,
        terms: &Value,
        ref_value: f64,
        gate_ref: &GateReference,
        cands: &[Cand],
        variant: Variant,
        sweep_table: &[String],
    ) -> Result<Result<(), String>, String> {
        let n = art.qubits;
        let state_mib = (1u64 << n) * 8 / (1 << 20); // complex64
        let before = match wait_card() {
            Ok(b) => b,
            Err(e) => return Ok(Err(e)),
        };
        let load_before = wait_quiet(self.void_above)?;
        let engines = [
            Engine::Ours,
            Engine::Custatevec(variant),
            Engine::AerGpu(variant),
        ];
        // ONE GPU process at a time. Two CUDA processes coexisting on this
        // card slow each other asymmetrically, by up to 45% and by start
        // order (measured on akilles: ours 97 ms alone, 146 ms when started
        // after a cuStateVec process; cuStateVec 125 ms alone, 168 ms when
        // started after ours), while each arm alone is steady across fresh
        // processes (8 each: 96-98 ms, 126-130 ms). So each repeat starts an
        // arm, loads, discards one warm-up, times one call, takes the arm's
        // device census, and quits it before the next arm starts; the arms
        // interleave per repeat (A/B/C/A/B/C, §4.4). Reaped arms are the
        // lane's own tree in the CPU census (cutime/cstime).
        let cpu = CpuCensus::start(&[std::process::id(), self.oracle.pid()])?;
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); 3];
        let mut values: Vec<Vec<f64>> = vec![Vec::new(); 3];
        let mut fixed_all: Vec<Vec<f64>> = vec![Vec::new(); 3];
        let mut held_all: Vec<Vec<u64>> = vec![Vec::new(); 3];
        let mut versions: Vec<BTreeMap<String, String>> = Vec::new();
        let mut hello_w: Vec<Vec<PathWitness>> = Vec::new();
        let mut want = REPEATS;
        while times[0].len() < want {
            for (i, &e) in engines.iter().enumerate() {
                let mut a = GpuArm::launch(e)?;
                a.load(id, &art.path)?;
                let (_, first) = a.expectation(id, terms)?; // discarded warm-up
                let (v, t) = a.expectation(id, terms)?;
                // The arm must be ON the card, holding at least its state
                // (Aer releases its state after a run and keeps its context:
                // >= 1 MiB). A failure aborts the lane whole (§7 E5).
                let need = if matches!(e, Engine::AerGpu(_)) {
                    1
                } else {
                    state_mib
                };
                let got = census::per_pid_mib()?.get(&a.pid()).copied().unwrap_or(0);
                if got < need {
                    return Err(format!(
                        "ABORT (§7 E5: the lane aborts whole on a failed device census): {} pid {} holds {got} MiB \
                         of device memory, needs >= {need} MiB: this arm is not running on the card",
                        e.label(),
                        a.pid()
                    ));
                }
                let f = a.startup_s + (first - t).max(0.0);
                if f < FIXED_COST_MIN_S {
                    return Err(format!(
                        "ABORT: {} fixed startup cost {f:.3}s < {FIXED_COST_MIN_S}s: a GPU arm with no startup \
                         cost is suspect (§4.1)",
                        e.label()
                    ));
                }
                if versions.len() == i {
                    versions.push(a.versions()?);
                    hello_w.push(a.witnesses()?);
                }
                a.quit()?;
                times[i].push(t);
                values[i].push(v);
                fixed_all[i].push(f);
                held_all[i].push(got);
            }
            eprintln!(
                "  repeat {} ours {:.6e}s cusv {:.6e}s aer {:.6e}s",
                times[0].len(),
                times[0].last().unwrap(),
                times[1].last().unwrap(),
                times[2].last().unwrap()
            );
            if times[0].len() >= 3 && times.iter().flatten().any(|t| *t > 30.0) {
                want = want.min(3);
            }
        }
        let external = cpu.finish()?;
        let gpu = census::after(before)?;
        let load_after = load1()?;
        if let Some(why) = census::void_reason(&gpu) {
            return Ok(Err(why));
        }
        if omega_emu_compare::void_at_load(external.external_cores_during_row, self.void_above) {
            return Ok(Err(format!(
                "{:.2} cores of work outside the lane during the row",
                external.external_cores_during_row
            )));
        }
        let median = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            s[s.len() / 2]
        };
        let fixed: Vec<f64> = fixed_all.iter().map(|f| median(f)).collect();
        let floors: Vec<f64> = engines
            .iter()
            .map(|e| {
                cands
                    .iter()
                    .find(|c| c.engine == *e)
                    .map(|c| c.floor_s)
                    .expect("candidate")
            })
            .collect();
        let device_w: Vec<PathWitness> = (0..3)
            .map(|i| {
                let need = if matches!(engines[i], Engine::AerGpu(_)) {
                    1
                } else {
                    state_mib
                };
                witness(
                    "device_census_pid",
                    &format!(
                        "every repeat's process held >= {need} MiB on the card: {:?} MiB",
                        held_all[i]
                    ),
                )
            })
            .collect();

        // Value gate on every repeat (§4.3), the same tolerance as the gate stage.
        let gaps: Vec<f64> = values
            .iter()
            .map(|vs| {
                vs.iter()
                    .map(|v| rel_gap(*v, ref_value))
                    .fold(0.0, f64::max)
            })
            .collect();
        if gaps[0] > GATE_REL_F32 || gaps[1] > GATE_REL_F32 {
            return Ok(Err(format!(
                "value gate failed in the row: ours {:.3e}, competitor {:.3e}",
                gaps[0], gaps[1]
            )));
        }
        let mut notes = Vec::new();
        let spreads: Vec<(f64, bool)> = times.iter().map(|t| spread(t)).collect();
        if spreads.iter().any(|s| s.1) {
            return Ok(Err(format!(
                "a repeat spread exceeds 5x: {:?} (§4.3a)",
                spreads.iter().map(|s| s.0).collect::<Vec<_>>()
            )));
        }

        // Assemble.
        let gate_w = |label: &str, g: f64, c: &Cand| -> Vec<PathWitness> {
            let mut w = vec![witness(
                "value_gate",
                &format!("max rel gap over the row {g:.3e} <= {GATE_REL_F32:.0e} vs {label}"),
            )];
            if let Some((omf, md)) = c.state {
                w.push(witness(
                    "state_gate_vs_aer_double",
                    &format!("|1-F|={omf:.3e} max|dpsi|={md:.3e} <= {GATE_REL_F32:.0e}"),
                ));
            }
            w
        };
        let ref_label = match gate_ref {
            GateReference::Oracle { .. } => "aer-cpu-f64",
            _ => "the f64 controls",
        };
        let cand = |e: Engine| {
            cands
                .iter()
                .find(|c| c.engine == e)
                .expect("every engine was a candidate")
        };
        let mut speed_arms = Vec::new();
        for i in 0..3 {
            let mut w = hello_w[i].clone();
            w.push(device_w[i].clone());
            w.extend(gate_w(ref_label, gaps[i], cand(engines[i])));
            let a = arm(
                &engines[i].label(),
                versions[i].clone(),
                engines[i].threads(),
                engines[i].knobs(),
                w,
            );
            speed_arms.push(speed_arm(a, &times[i], fixed[i], floors[i])?);
        }
        let floor_arm = speed_arms.pop().expect("three arms");
        let comp = speed_arms.pop().expect("three arms");
        let ours = speed_arms.pop().expect("three arms");

        // Context: every candidate not timed in the row, from its gate-stage sweep.
        let mut context = BTreeMap::new();
        let floor_min = floor_arm.timing.get().expect("present").min_s;
        context.insert(
            format!("{} (floor, timed in the row)", engines[2].label()),
            floor_arm,
        );
        for c in cands {
            if engines.contains(&c.engine) {
                continue;
            }
            let mut w = c.witnesses.clone();
            w.push(witness(
                "gate_stage",
                c.why_not
                    .as_deref()
                    .map(|x| format!("REFUSED: {x}"))
                    .as_deref()
                    .unwrap_or("passes"),
            ));
            let a = arm(
                &c.engine.label(),
                c.versions.clone(),
                c.engine.threads(),
                c.engine.knobs(),
                w,
            );
            let mut sa = speed_arm(a, &c.sweep, c.fixed_cost_s(), c.floor_s)?;
            if let Some(t) = sa.timing.get().cloned() {
                sa.timing = Witness::present(omega_emu_compare::Timing {
                    interleaved: false,
                    ..t
                });
            }
            context.insert(
                format!(
                    "{} (context, not a row: gate-stage sweep, not interleaved)",
                    c.engine.label()
                ),
                sa,
            );
        }

        let o = ours.timing.get().expect("present").clone();
        let t = comp.timing.get().expect("present").clone();
        let comp_floor = *comp.per_call_floor_s.get().expect("present");
        let ratio = t.min_s / o.min_s;
        let multiple = t.min_s / comp_floor;
        let verdict = if t.min_s > floor_min {
            FloorVerdict::LoopIsTheFinding
        } else {
            FloorVerdict::Ok
        };
        let derivation = Derivation {
            tie_rule: TIE_RULE.to_string(),
            win_ratio_threshold: WIN_RATIO_THRESHOLD,
            ours_min_median_s: [o.min_s, o.median_s],
            competitor_min_median_s: [t.min_s, t.median_s],
            ratio_competitor_over_ours: ratio,
            intervals_disjoint: o.median_s < t.min_s || t.median_s < o.min_s,
            ratio_beyond_threshold: ratio >= WIN_RATIO_THRESHOLD
                || ratio <= 1.0 / WIN_RATIO_THRESHOLD,
            overhead_rule: OVERHEAD_RULE.to_string(),
            competitor_per_call_floor_s: comp_floor,
            competitor_min_over_floor: multiple,
            floor_arm_min_s: Some(floor_min),
        };
        let classification = classify(ratio, o.min_s, o.median_s, t.min_s, t.median_s);

        notes.push(format!(
            "observable O = {} ({} terms); reference {ref_value:+.15} ({ref_label})",
            observable_string(n),
            2 * n - 1
        ));
        notes.push(format!("admitted competitor: cuStateVec {} — the FASTEST variant that passes the gate (§8 in the competitor's favour, bounded by §4.3); f64 variants are controls, never admitted", variant.as_str()));
        notes.push(format!(
            "gate stage (each candidate alone on the card): {}",
            sweep_table.join(" | ")
        ));
        let fused = cand(Engine::Custatevec(Variant::Fused));
        if variant == Variant::Unfused && !fused.passes {
            notes.push(format!(
                "cuStateVec fused runs in {:.3e}s ({:.2}x ours' min {:.3e}s) and misses the gate: {}. This row says \"we beat \
                 every cuStateVec configuration that meets the accuracy gate\", not \"we beat cuStateVec\"; the fused miss is a \
                 statement about this circuit (on RANDOM-1 the fused path's drift was far smaller), not about the configuration",
                fused.sweep_min(), o.min_s / fused.sweep_min(), o.min_s, fused.why_not.as_deref().unwrap_or("")));
        }
        notes.push(format!(
            "fixed cost, reported separately from steady state (§7 E5), median over the row's launches: ours {:.3}s, \
             cuStateVec {:.3}s, Aer floor {:.3}s (process start + CUDA context + first-call excess); Aer's untimed \
             warm-up is load-bearing: its first call JITs sm_90 PTX for sm_120",
            fixed[0], fixed[1], fixed[2]
        ));
        notes.push(
            "one GPU process at a time: every repeat starts an arm, discards one warm-up, times one call and quits it \
             before the next arm starts. Two CUDA processes coexisting on this card slow each other asymmetrically, by \
             up to 45% and by start order (ours 97 ms alone, 146 ms when started after a cuStateVec process; cuStateVec \
             125 ms alone, 168 ms when started after ours, HEA 24q), while each arm alone is steady across fresh \
             processes (96-98 ms and 126-130 ms over 8 each)"
                .to_string(),
        );
        notes.push(format!(
            "device census: card {} ; holders and utilisation before the arms started and after they exited; the after \
             utilisation is the min of {} samples taken {} s after the last arm exited (read immediately, the window still \
             covers our own arms)", self.device, census::UTIL_SAMPLES, census::SETTLE_S));
        notes.push(format!("max/min across repeats: ours {:.2}x, cuStateVec {:.2}x, Aer floor {:.2}x (flag above 5x, §4.3a)", spreads[0].0, spreads[1].0, spreads[2].0));
        if id.starts_with("hea-") {
            notes.push("HEA at depth 24 and 24/26/28 qubits is a shape added for the GPU lane, not one §3 pinned".into());
        }

        let value_gate = ValueGate {
            tolerance_rel: GATE_REL_F32,
            reference: gate_ref.clone(),
            ours_value: json!(values[0][0]),
            competitor_value: json!(values[1][0]),
            ours_gap: gaps[0],
            competitor_gap: gaps[1],
            truncation: None,
            passed: true,
        };
        let row = Row {
            row_id: row_id(id),
            lane: Lane::DenseGpu,
            phase: PHASE.to_string(),
            recorded: self.recorded.clone(),
            box_name: Witness::present(self.box_name.clone()),
            git: Witness::present(self.git.clone()),
            circuit: workload(art, Quantity::Expectation, None, Some(&observable_string(n))),
            precision: Precision {
                ours: "f32".into(),
                competitor: "f32".into(),
                matched: true,
                note: Some("both arms complex64: our CUDA arm is f32 (CudaSlice<f32>); cuStateVec at CUDA_C_32F/COMPUTE_32F; \
                            the Aer floor at precision=\"single\" (Aer-GPU defaults to double on this card). Precision is \
                            witnessed by the gap to the f64 reference, not by dtype: every arm returns complex128".into()),
            },
            body: RowBody::Speed(Box::new(SpeedRow {
                ours,
                competitor: comp,
                context_arms: context,
                value_gate: Witness::present(value_gate),
                load: Witness::present(LoadRecord {
                    load1_before: load_before,
                    load1_after: load_after,
                    void_above: self.void_above,
                    cooldown_s: None,
                    hostgate_cap: Some(self.hostgate_cap.clone()),
                    gpu: Some(gpu),
                    external_cpu: Some(external),
                }),
                ratio_competitor_over_ours: ratio,
                classification,
                overhead_dominated: comp_floor > 0.0 && multiple < OVERHEAD_FLOOR_MULTIPLE,
                floor_check: Some(FloorCheck { arm: engines[2].label(), min_s: floor_min, verdict }),
                derivation: Witness::present(derivation),
            })),
            notes,
        };
        self.publish(&row)?;
        Ok(Ok(()))
    }

    fn publish(&mut self, row: &Row) -> Result<(), String> {
        let problems = omega_emu_compare::check(row);
        if !problems.is_empty() {
            return Err(format!("{} refused by check(): {problems:?}", row.row_id));
        }
        self.writer.write_row(row).map_err(|e| e.to_string())?;
        if let RowBody::Speed(s) = &row.body {
            let o = s.ours.timing.get().expect("present");
            let t = s.competitor.timing.get().expect("present");
            eprintln!(
                "ROW {}\t{:?}\tratio={:.4}\tours_min={:.6e}\tcomp_min={:.6e}\tfloor={:?}",
                row.row_id,
                s.classification,
                s.ratio_competitor_over_ours,
                o.min_s,
                t.min_s,
                s.floor_check.as_ref().map(|f| &f.verdict)
            );
        }
        Ok(())
    }

    fn void(&mut self, row_id: &str, reason: &str) -> Result<(), String> {
        eprintln!("VOID {row_id}: {reason}");
        let void = VoidRow {
            row_id: row_id.to_string(),
            lane: Lane::DenseGpu,
            phase: PHASE.to_string(),
            recorded: self.recorded.clone(),
            reason: reason.to_string(),
            context: json!({"box": self.box_name, "device": self.device}),
        };
        self.writer.write_void(&void).map_err(|e| e.to_string())
    }
}

fn git_snap() -> Result<GitRev, String> {
    let rev = GitRev {
        compiled_from: option_env!("OMEGA_EMU_GIT_REV").unwrap_or("").to_string(),
        worktree: git(&["rev-parse", "HEAD"])?,
        worktree_dirty: !git(&["status", "--porcelain"])?.is_empty(),
    };
    if let Some(why) = rev.void_reason() {
        return Err(why);
    }
    Ok(rev)
}

fn git(args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(emu_compare_dense::workload::repo_root())
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn today() -> Result<String, String> {
    let out = Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .map_err(|e| format!("date: {e}"))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn load1() -> Result<f64, String> {
    let s = std::fs::read_to_string("/proc/loadavg").map_err(|e| format!("/proc/loadavg: {e}"))?;
    s.split_whitespace()
        .next()
        .ok_or("empty /proc/loadavg")?
        .parse()
        .map_err(|e| format!("load: {e}"))
}

fn wait_quiet(void_above: f64) -> Result<f64, String> {
    let start = Instant::now();
    loop {
        let load = load1()?;
        if !omega_emu_compare::void_at_load(load, void_above) {
            return Ok(load);
        }
        if start.elapsed() > Duration::from_secs(600) {
            return Err(format!("load1 {load} stayed above {void_above} for 10 min"));
        }
        eprintln!("load1 {load} > {void_above}, waiting");
        std::thread::sleep(Duration::from_secs(20));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(engine: Engine, sweep: f64, passes: bool) -> Cand {
        Cand {
            engine,
            startup_s: 0.3,
            build_s: 0.0,
            first_s: sweep,
            sweep: vec![sweep; 3],
            floor_s: 1e-5,
            value: 0.0,
            value_gap: 0.0,
            state: None,
            passes,
            why_not: None,
            versions: BTreeMap::new(),
            witnesses: vec![],
        }
    }

    #[test]
    fn the_fastest_passing_variant_is_admitted() {
        let c = vec![
            cand(Engine::Custatevec(Variant::Unfused), 0.166, true),
            cand(Engine::Custatevec(Variant::Fused), 0.033, true),
        ];
        assert_eq!(
            admit(&c),
            Some(Variant::Fused),
            "in the competitor's favour when fused passes"
        );
    }

    #[test]
    fn a_failing_variant_is_never_admitted_however_fast() {
        let c = vec![
            cand(Engine::Custatevec(Variant::Unfused), 0.166, true),
            cand(Engine::Custatevec(Variant::Fused), 0.033, false),
        ];
        assert_eq!(
            admit(&c),
            Some(Variant::Unfused),
            "§4.3: a fast wrong number is disqualified"
        );
    }

    #[test]
    fn an_f64_control_is_never_the_competitor() {
        let c = vec![
            cand(Engine::Custatevec(Variant::Unfused), 0.166, false),
            cand(Engine::Custatevec(Variant::FusedF64), 0.105, true),
        ];
        assert_eq!(admit(&c), None, "controls are context, never rows");
    }

    #[test]
    fn fixed_cost_counts_startup_and_first_call_excess() {
        let mut c = cand(Engine::Ours, 0.122, true);
        c.startup_s = 0.25;
        c.first_s = 0.40;
        assert!((c.fixed_cost_s() - (0.25 + 0.278)).abs() < 1e-12);
    }
}
