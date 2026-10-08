// SPDX-License-Identifier: Apache-2.0
//! The measurement (§4.4). Per circuit: the Aer oracle's reference and the
//! non-triviality check; every arm's final state gated against the oracle's
//! (untimed, §4.3a); a knob sweep per arm, fastest setting kept (§8, in the
//! competitor's favour, recorded); then one row per competitor, each with one
//! discarded warm-up per arm, N = 5 interleaved repeats (3 over 30 s), and the
//! 1-minute load before and after. A row whose load crosses the host's
//! threshold is re-run, and published as void after three attempts.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use omega_core::executor::Observable;
use omega_emu_compare::{
    CpuCensus, GateReference, GitRev, Lane, LoadRecord, PathWitness, Quantity, Row, RowBody,
    RowWriter, ValueGate, VoidRow, GATE_REL_F32, GATE_REL_F64,
};
use serde_json::{json, Value};

use crate::assemble::{self, arm, map, speed_arm, speed_row, witness, Competitor, PHASE};
use crate::client::{pair, seconds, Engine, PyArm};
use crate::gate::{nontrivial, rel_gap, spread, state_gate, state_tol, z_gate, Z_BOUND};
use crate::workload::{
    load_artifact, observable_string, observable_terms, write_npy, z_marginals, Artifact, Ours,
    OBSERVABLE_NAME, OURS_VERSION, SEED, SHOTS,
};

/// §3's dense expectation cells, in run order (largest last, so a cap kill
/// leaves the rows that finished).
pub const EXPECTATION_CELLS: &[&str] = &[
    "hea-14q-d4",
    "hea-14q-d12",
    "hea-14q-d24",
    "hea-20q-d12",
    "hea-20q-d24",
    "random1-24q-d20",
    "random1-26q-d20",
];
/// §3's one sampling cell.
pub const SAMPLING_CELL: &str = "random1-20q-d20";

/// Thread counts swept for every arm: the physical and the logical core count
/// of akilles (16C/32T). The fastest is kept per arm and per circuit.
pub const THREAD_SWEEP: &[u32] = &[16, 32];
/// qsim's `max_fused_gate_size` values swept (qsim's documented range).
pub const FUSE_SWEEP: &[u32] = &[2, 3, 4];
/// The oracle's thread count. It is never timed.
const ORACLE_THREADS: u32 = 16;
/// Timed repeats per sweep setting (min kept).
const SWEEP_REPEATS: usize = 3;
const MAX_ATTEMPTS: u32 = 3;
/// A per-call floor is the minimum of this many calls (§4.7): one call can be
/// a cold OpenMP or allocator start, which would mark a row overhead-dominated
/// against a floor it never pays again.
const FLOOR_CALLS: usize = 5;

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
    void_above: f64,
    hostgate_cap: String,
    writer: RowWriter,
    tmp: PathBuf,
    oracle: PyArm,
    oracle_versions: std::collections::BTreeMap<String, String>,
    ours: Vec<Ours>,
    qulacs: Vec<PyArm>,
    qsim: Vec<PyArm>,
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
        lane: Lane::DenseCpu,
        phase: PHASE.to_string(),
        recorded: today()?,
        reason: reason.to_string(),
        context: json!({"source": "hostgate --watch killed the process tree"}),
    };
    let mut writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    writer.write_void(&void).map_err(|e| e.to_string())
}

/// `only`: a comma-separated list of cell ids (expectation cells and/or the
/// sampling cell). `append`: add to an existing lane file — every row carries
/// its own git stamp, so rows from two runs remain separately attributable.
pub fn measure(out: &Path, only: Option<&str>, append: bool) -> Result<(), String> {
    if out.exists() && !append {
        return Err(format!(
            "{} exists; refusing to append over a previous lane file",
            out.display()
        ));
    }
    let git = git_snap()?;
    let recorded = today()?;
    let box_name = hostname()?;
    let cores = omega_emu_compare::host_cores()?;
    let void_above = omega_emu_compare::load_void_above(cores);
    let hostgate_cap = omega_emu_compare::hostgate_cap()?;
    eprintln!(
        "dense lane on {box_name} at {recorded} rev {}; load void above {void_above} ({cores} cores); cap {hostgate_cap}",
        git.worktree
    );
    let tmp = std::env::temp_dir().join(format!("emu-dense-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;

    let oracle = PyArm::launch(Engine::AerOracle {
        threads: ORACLE_THREADS,
    })?;
    let oracle_versions = oracle.versions()?;
    let ours = THREAD_SWEEP
        .iter()
        .map(|&t| Ours::new(t))
        .collect::<Result<Vec<_>, _>>()?;
    let qulacs = THREAD_SWEEP
        .iter()
        .map(|&t| PyArm::launch(Engine::Qulacs { threads: t }))
        .collect::<Result<Vec<_>, _>>()?;
    let mut qsim = Vec::new();
    for &t in THREAD_SWEEP {
        for &f in FUSE_SWEEP {
            qsim.push(PyArm::launch(Engine::Qsim {
                threads: t,
                fuse: f,
            })?);
        }
    }
    // §6: qsim rows are akilles-only; its SIMD refusal already ran at launch.
    if box_name != "akilles" {
        return Err(format!(
            "qsim rows are akilles-only by §6; this host is {box_name}"
        ));
    }
    let writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    let mut ctx = Ctx {
        git,
        recorded,
        box_name,
        void_above,
        hostgate_cap,
        writer,
        tmp,
        oracle,
        oracle_versions,
        ours,
        qulacs,
        qsim,
    };
    let wanted = |id: &str| only.is_none_or(|o| o.split(',').any(|x| x.trim() == id));
    if let Some(o) = only {
        for x in o.split(',') {
            let x = x.trim();
            if !EXPECTATION_CELLS.contains(&x) && x != SAMPLING_CELL {
                return Err(format!("--only names unknown cell {x}"));
            }
        }
    }
    for id in EXPECTATION_CELLS {
        if wanted(id) {
            ctx.expectation_cell(id)?;
        }
    }
    if wanted(SAMPLING_CELL) {
        ctx.sampling_cell(SAMPLING_CELL)?;
    }
    let _ = std::fs::remove_dir_all(&ctx.tmp);
    eprintln!("wrote {}", out.display());
    Ok(())
}

/// Which arm a timed closure drives.
#[derive(Clone, Copy)]
enum Who {
    Ours(usize),
    Qulacs(usize),
    Qsim(usize),
}

struct Choice {
    idx: usize,
    sweep: String,
}

impl Ctx {
    /// §4.4a: the lane's own tree — this process and every arm it launched.
    fn own_pids(&self) -> Vec<u32> {
        let mut v = vec![std::process::id(), self.oracle.pid()];
        v.extend(self.qulacs.iter().map(PyArm::pid));
        v.extend(self.qsim.iter().map(PyArm::pid));
        v
    }

    fn expectation_cell(&mut self, id: &str) -> Result<(), String> {
        let art = load_artifact(id)?;
        let n = art.qubits;
        let terms = observable_terms(n);
        let obs = Observable::parse(&observable_string(n))?;
        obs.validate_qubits(n).map_err(|e| e.to_string())?;
        eprintln!("BEGIN {id} ({n}q, depth {})", art.depth);

        // Our lowering time is this arm's setup cost (the competitors report
        // theirs from their own builders).
        let t = std::time::Instant::now();
        let _ = load_artifact(id)?;
        let ours_build = t.elapsed().as_secs_f64();

        self.oracle.load(id, &art.path)?;
        let mut qulacs_build = Vec::new();
        for a in &mut self.qulacs {
            qulacs_build.push(a.load(id, &art.path)?);
        }
        let mut qsim_build = Vec::new();
        for a in &mut self.qsim {
            qsim_build.push(a.load(id, &art.path)?);
        }

        // Oracle: value + reference state; refuse a degenerate cell.
        let ref_npy = self.tmp.join(format!("{id}-aer.npy"));
        let r = self
            .oracle
            .call(json!({"op": "reference", "name": id, "terms": terms, "out": ref_npy}))?;
        let oracle_value = r["value"].as_f64().ok_or("oracle value missing")?;
        nontrivial(oracle_value)?;
        eprintln!("  oracle {OBSERVABLE_NAME} = {oracle_value:+.12}");

        // Sweeps (fastest setting per arm, recorded).
        let ours_c = self.sweep(
            id,
            &terms,
            &obs,
            &art,
            (0..self.ours.len()).map(Who::Ours).collect(),
        )?;
        let qul_c = self.sweep(
            id,
            &terms,
            &obs,
            &art,
            (0..self.qulacs.len()).map(Who::Qulacs).collect(),
        )?;
        let qsim_c = self.sweep(
            id,
            &terms,
            &obs,
            &art,
            (0..self.qsim.len()).map(Who::Qsim).collect(),
        )?;

        // State gates for the settings that will be timed (§4.3a, mandatory).
        let ours_state = self.ours[ours_c.idx].state(&art.circuit)?;
        let ours_npy = self.tmp.join(format!("{id}-ours.npy"));
        write_npy(&ours_npy, &ours_state)?;
        drop(ours_state);
        let ours_sg = self.state_check("ours", &ref_npy, &ours_npy, true)?;
        let qul_npy = self.tmp.join(format!("{id}-qulacs.npy"));
        self.qulacs[qul_c.idx].write_state(id, &qul_npy)?;
        let qul_sg = self.state_check("qulacs", &ref_npy, &qul_npy, true)?;
        let qsim_npy = self.tmp.join(format!("{id}-qsim.npy"));
        self.qsim[qsim_c.idx].write_state(id, &qsim_npy)?;
        let qsim_sg = self.state_check("qsim", &ref_npy, &qsim_npy, false)?;
        for p in [&ref_npy, &ours_npy, &qul_npy, &qsim_npy] {
            let _ = std::fs::remove_file(p);
        }

        // qsim's in-call translation, measured for the record.
        let translate_s =
            seconds(&self.qsim[qsim_c.idx].call(json!({"op": "translate", "name": id}))?)?;
        let api = self.qsim[qsim_c.idx]
            .call(json!({"op": "expectation_api", "name": id, "terms": terms}))?;
        let (api_v, api_s) = pair(&api, "value", "seconds")?;

        let workload = assemble::workload(
            &art,
            Quantity::Expectation,
            None,
            Some(&observable_string(n)),
        );
        // A failed state gate voids that arm's rows; ours failing voids both.
        let ours_sg = match ours_sg {
            Ok(w) => w,
            Err(e) => {
                for c in ["qulacs", "qsim"] {
                    self.void(
                        &format!("dense-{id}-o-expectation-vs-{c}"),
                        &format!("ours: {e}"),
                    )?;
                }
                return Ok(());
            }
        };
        for (comp, comp_idx, sweep_note, build_s, sg) in [
            (
                Competitor::Qulacs,
                qul_c.idx,
                qul_c.sweep.clone(),
                qulacs_build[qul_c.idx],
                qul_sg.clone(),
            ),
            (
                Competitor::Qsim,
                qsim_c.idx,
                qsim_c.sweep.clone(),
                qsim_build[qsim_c.idx],
                qsim_sg.clone(),
            ),
        ] {
            let comp_who = match comp {
                Competitor::Qulacs => Who::Qulacs(comp_idx),
                Competitor::Qsim => Who::Qsim(comp_idx),
            };
            let sg = match sg {
                Ok(w) => w,
                Err(e) => {
                    let c = if comp == Competitor::Qulacs {
                        "qulacs"
                    } else {
                        "qsim"
                    };
                    self.void(&format!("dense-{id}-o-expectation-vs-{c}"), &e)?;
                    continue;
                }
            };
            let mut extra = Vec::new();
            if art.depth <= 4 {
                extra.push(
                    "§3: depth-4 rows are sub-millisecond and directional only, never headlines"
                        .to_string(),
                );
            }
            extra.extend(vec![
                format!("observable O = {OBSERVABLE_NAME}; oracle (Aer CPU double) value {oracle_value:+.15}"),
                format!("ours sweep (min of {SWEEP_REPEATS} after a warm-up): {}", ours_c.sweep),
                format!("competitor sweep: {sweep_note}"),
                format!("state gate (untimed, vs Aer double): ours {}; competitor {}", ours_sg.1, sg.1),
            ]);
            if comp == Competitor::Qsim {
                extra.push(format!(
                    "qsim public API simulate_expectation_values (translation inside) took {api_s:.6e}s \
                     for value {api_v:+.12}; its cirq->qsim translation alone {translate_s:.6e}s. The row \
                     times the kernel on a pre-translated circuit, as §4.2 times Aer on a pre-transpiled one."
                ));
            }
            self.expectation_row(
                id,
                &art,
                &workload,
                &terms,
                &obs,
                oracle_value,
                ours_c.idx,
                ours_build,
                ours_sg.0.clone(),
                comp,
                comp_who,
                build_s,
                sg.0.clone(),
                extra,
            )?;
        }
        Ok(())
    }

    /// Time every setting of one arm (one warm-up, then min of 3) and keep the
    /// fastest. Returns the chosen index and a printable sweep table.
    fn sweep(
        &mut self,
        id: &str,
        terms: &Value,
        obs: &Observable,
        art: &Artifact,
        settings: Vec<Who>,
    ) -> Result<Choice, String> {
        let mut table = Vec::new();
        let mut best: Option<(usize, f64)> = None;
        for who in settings {
            let f = |s: &mut Self| s.run(who, id, terms, obs, art).map(|(_, t)| t);
            let _ = f(self)?; // warm-up
            let mut mn = f64::INFINITY;
            for _ in 0..SWEEP_REPEATS {
                mn = mn.min(f(self)?);
            }
            let (label, idx) = self.label(who);
            table.push(format!("{label} {mn:.6e}s"));
            if best.is_none_or(|(_, b)| mn < b) {
                best = Some((idx, mn));
            }
        }
        let (idx, _) = best.ok_or("empty sweep")?;
        Ok(Choice {
            idx,
            sweep: table.join(", "),
        })
    }

    fn label(&self, who: Who) -> (String, usize) {
        match who {
            Who::Ours(i) => (format!("ours[t={}]", self.ours[i].threads), i),
            Who::Qulacs(i) => (self.qulacs[i].engine.label(), i),
            Who::Qsim(i) => (self.qsim[i].engine.label(), i),
        }
    }

    /// One timed expectation of one arm: `(value, seconds)`.
    fn run(
        &mut self,
        who: Who,
        id: &str,
        terms: &Value,
        obs: &Observable,
        art: &Artifact,
    ) -> Result<(f64, f64), String> {
        match who {
            Who::Ours(i) => self.ours[i].expectation(&art.circuit, obs),
            Who::Qulacs(i) => self.qulacs[i].expectation(id, terms),
            Who::Qsim(i) => self.qsim[i].expectation(id, terms),
        }
    }

    /// Outer `Err`: the comparison could not run. Inner `Err`: it ran and the
    /// arm failed the gate — a disagreement finding for that arm's rows.
    #[allow(clippy::type_complexity)]
    fn state_check(
        &mut self,
        label: &str,
        reference: &Path,
        other: &Path,
        f64_arm: bool,
    ) -> Result<Result<(PathWitness, String), String>, String> {
        let v = self
            .oracle
            .call(json!({"op": "compare", "ref": reference, "other": other}))?;
        let (omf, md) = pair(&v, "one_minus_f_abs", "max_abs_diff")?;
        let tol = state_tol(f64_arm);
        if let Err(e) = state_gate(label, omf, md, tol) {
            return Ok(Err(format!(
                "disagreement finding (state gate, untimed): {e}"
            )));
        }
        let observed = format!("|1-F|={omf:.3e} max|dpsi|={md:.3e} <= {tol:.0e} vs aer-cpu-f64");
        Ok(Ok((
            witness("state_gate_vs_aer_double", &observed),
            format!("{label} {observed}"),
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn expectation_row(
        &mut self,
        id: &str,
        art: &Artifact,
        workload: &omega_emu_compare::Workload,
        terms: &Value,
        obs: &Observable,
        oracle_value: f64,
        ours_idx: usize,
        ours_build: f64,
        ours_state_w: PathWitness,
        comp: Competitor,
        comp_who: Who,
        comp_build: f64,
        comp_state_w: PathWitness,
        mut notes: Vec<String>,
    ) -> Result<(), String> {
        let comp_name = match comp {
            Competitor::Qulacs => "qulacs",
            Competitor::Qsim => "qsim",
        };
        let row_id = format!("dense-{id}-o-expectation-vs-{comp_name}");
        let ours_floor = min_of(|| self.ours[ours_idx].floor())?;
        let comp_floor = match comp_who {
            Who::Qulacs(i) => min_of(|| self.qulacs[i].floor())?,
            Who::Qsim(i) => min_of(|| self.qsim[i].floor())?,
            Who::Ours(_) => unreachable!(),
        };
        for attempt in 1..=MAX_ATTEMPTS {
            let before = wait_quiet(self.void_above)?;
            let census = CpuCensus::start(&self.own_pids())?;
            // One untimed warm-up per (arm, row).
            let (_, ours_first) = self.run(Who::Ours(ours_idx), id, terms, obs, art)?;
            let (_, comp_first) = self.run(comp_who, id, terms, obs, art)?;
            let mut n = 5usize;
            let (mut a, mut b, mut av, mut bv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            while a.len() < n {
                let (va, ta) = self.run(Who::Ours(ours_idx), id, terms, obs, art)?;
                let (vb, tb) = self.run(comp_who, id, terms, obs, art)?;
                eprintln!(
                    "  {row_id} repeat {} ours {ta:.6e}s {comp_name} {tb:.6e}s",
                    a.len() + 1
                );
                a.push(ta);
                b.push(tb);
                av.push(va);
                bv.push(vb);
                if a.len() >= 3 && (a.iter().chain(&b).any(|t| *t > 30.0)) {
                    n = n.min(3);
                }
            }
            let external = census.finish()?;
            let after = load1()?;
            if omega_emu_compare::void_at_load(external.external_cores_during_row, self.void_above)
            {
                eprintln!(
                    "  {row_id}: {:.2} cores of work outside the lane during the row > {} (attempt {attempt}); re-running",
                    external.external_cores_during_row, self.void_above
                );
                continue;
            }
            // Value gate: every repeat's value, both arms, against the oracle.
            let comp_tol = match comp {
                Competitor::Qulacs => GATE_REL_F64,
                Competitor::Qsim => GATE_REL_F32,
            };
            let og = av
                .iter()
                .map(|v| rel_gap(*v, oracle_value))
                .fold(0.0, f64::max);
            let cg = bv
                .iter()
                .map(|v| rel_gap(*v, oracle_value))
                .fold(0.0, f64::max);
            if og > GATE_REL_F64 || cg > comp_tol {
                let reason = format!(
                    "disagreement finding: oracle {oracle_value:+.15}, ours gap {og:.3e} (tol {GATE_REL_F64:.0e}), \
                     {comp_name} gap {cg:.3e} (tol {comp_tol:.0e})"
                );
                return self.void(&row_id, &reason);
            }
            let gate = ValueGate {
                tolerance_rel: comp_tol.max(GATE_REL_F64),
                reference: GateReference::Oracle {
                    arm: "aer-cpu-f64".into(),
                    versions: self.oracle_versions.clone(),
                    value: json!(oracle_value),
                },
                ours_value: json!(av[0]),
                competitor_value: json!(bv[0]),
                ours_gap: og,
                competitor_gap: cg,
                truncation: None,
                passed: true,
            };
            let (sa, fa) = spread(&a);
            let (sb, fb) = spread(&b);
            notes.push(format!(
                "first call (untimed warm-up) ours {ours_first:.6e}s, {comp_name} {comp_first:.6e}s; \
                 setup (circuit build, untimed) ours {ours_build:.6e}s, {comp_name} {comp_build:.6e}s"
            ));
            notes.push(format!("max/min across repeats: ours {sa:.2}x, {comp_name} {sb:.2}x (flag above 5x, §4.3a)"));
            if fa || fb {
                notes.push("FLAGGED: a repeat spread exceeds 5x (§4.3a: an OpenMP arm on a loaded box collapses rather than slows)".into());
            }
            let ours_t = self.ours[ours_idx].threads;
            let mut ours_w = self.ours[ours_idx].witness()?;
            ours_w.push(ours_state_w.clone());
            ours_w.push(witness(
                "value_gate_vs_aer_double",
                &format!("max rel gap {og:.3e} <= {GATE_REL_F64:.0e}"),
            ));
            let ours_arm = arm(
                "omega-backend-statevector",
                map(&[("omega-backend-statevector", OURS_VERSION)]),
                ours_t,
                map(&[
                    ("api", "StatevectorBackend::expectation"),
                    ("rayon_threads", &ours_t.to_string()),
                ]),
                ours_w,
            );
            let (comp_engine, mut comp_w, comp_versions) = {
                let a = match comp_who {
                    Who::Qulacs(i) => &self.qulacs[i],
                    Who::Qsim(i) => &self.qsim[i],
                    Who::Ours(_) => unreachable!(),
                };
                (a.engine.clone(), a.witnesses()?, a.versions()?)
            };
            comp_w.push(comp_state_w.clone());
            comp_w.push(witness(
                "value_gate_vs_aer_double",
                &format!("max rel gap {cg:.3e} <= {comp_tol:.0e}"),
            ));
            let comp_arm = arm(
                &comp_engine.label(),
                comp_versions,
                comp_engine.threads(),
                comp_engine.knobs(),
                comp_w,
            );
            let load = LoadRecord {
                load1_before: before,
                load1_after: after,
                void_above: self.void_above,
                cooldown_s: None,
                hostgate_cap: Some(self.hostgate_cap.clone()),
                gpu: None,
                external_cpu: Some(external.clone()),
            };
            let row = speed_row(
                &row_id,
                &self.recorded,
                &self.box_name,
                self.git.clone(),
                workload.clone(),
                comp,
                speed_arm(ours_arm, &a, ours_build, ours_floor)?,
                speed_arm(comp_arm, &b, comp_build, comp_floor)?,
                gate,
                load,
                notes,
            )?;
            return self.publish(&row);
        }
        self.void(
            &row_id,
            &format!(
                "work outside the lane stayed above {} cores after {MAX_ATTEMPTS} attempts",
                self.void_above
            ),
        )
    }

    fn sampling_cell(&mut self, id: &str) -> Result<(), String> {
        let art = load_artifact(id)?;
        let n = art.qubits;
        eprintln!("BEGIN {id} sampling ({n}q, {SHOTS} shots)");
        self.oracle.load(id, &art.path)?;
        let r = self
            .oracle
            .call(json!({"op": "reference", "name": id, "terms": observable_terms(n)}))?;
        let exact: Vec<f64> = r["z_marginals"]
            .as_array()
            .ok_or("oracle z_marginals missing")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN))
            .collect();
        // Sampling compares against qulacs only: §3's sampling row exercises
        // the sampling path, and qsim's sampler is not part of E4's deliverable.
        let qi = 0;
        let oi = 0;
        let t = std::time::Instant::now();
        let ours_state = self.ours[oi].state(&art.circuit)?;
        let ours_setup = t.elapsed().as_secs_f64();
        self.qulacs[qi].load(id, &art.path)?;
        let ev = self.qulacs[qi].call(json!({"op": "evolve", "name": id}))?;
        let qul_setup = seconds(&ev)?;
        let ours_floor = min_of(|| self.ours[oi].floor_sample())?;
        let qul_floor = min_of(|| self.qulacs[qi].floor_sample())?;
        let row_id = format!("dense-{id}-sampling-{SHOTS}-vs-qulacs");
        for _attempt in 1..=MAX_ATTEMPTS {
            let before = wait_quiet(self.void_above)?;
            let census = CpuCensus::start(&self.own_pids())?;
            let _ = self.ours[oi].sample(&ours_state, n, SHOTS, SEED)?;
            let _ = self.qulacs[qi]
                .call(json!({"op": "sample", "name": id, "shots": SHOTS, "seed": SEED}))?;
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let (mut zo, mut zc): (f64, f64) = (0.0, 0.0);
            // Seed 0's per-qubit <Z> estimates: the numbers the gate pins (§4.3c).
            let (mut est0, mut est0_c): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
            for rep in 0..5 {
                let (counts, ta) = self.ours[oi].sample(&ours_state, n, SHOTS, SEED + rep)?;
                let est = z_marginals(&counts, n, SHOTS);
                zo = zo.max(z_gate("ours", &est, &exact, SHOTS)?);
                if est0.is_empty() {
                    est0 = est.clone();
                }
                let v = self.qulacs[qi].call(
                    json!({"op": "sample", "name": id, "shots": SHOTS, "seed": SEED + rep}),
                )?;
                let tb = seconds(&v)?;
                let est_c: Vec<f64> = v["z_marginals"]
                    .as_array()
                    .ok_or("qulacs z_marginals missing")?
                    .iter()
                    .map(|x| x.as_f64().unwrap_or(f64::NAN))
                    .collect();
                zc = zc.max(z_gate("qulacs", &est_c, &exact, SHOTS)?);
                if est0_c.is_empty() {
                    est0_c = est_c.clone();
                }
                eprintln!(
                    "  {row_id} repeat {} ours {ta:.6e}s qulacs {tb:.6e}s",
                    rep + 1
                );
                a.push(ta);
                b.push(tb);
            }
            let external = census.finish()?;
            let after = load1()?;
            if omega_emu_compare::void_at_load(external.external_cores_during_row, self.void_above)
            {
                continue;
            }
            let gate = ValueGate {
                tolerance_rel: Z_BOUND,
                reference: GateReference::Oracle {
                    arm: "aer-cpu-f64 exact per-qubit <Z> marginals".into(),
                    versions: self.oracle_versions.clone(),
                    value: json!(exact),
                },
                ours_value: json!(est0),
                competitor_value: json!(est0_c),
                ours_gap: zo,
                competitor_gap: zc,
                truncation: None,
                passed: true,
            };
            let (sa, _) = spread(&a);
            let (sb, _) = spread(&b);
            let notes = vec![
                format!(
                    "sampling gate: two-sided per-qubit z-score of the <Z> estimate against Aer's exact \
                     marginal, |z| <= {Z_BOUND} on every qubit and every repeat (seeds {SEED}..{}); value_gate's \
                     tolerance_rel and gaps carry that z-score, not a relative error; ours_value and \
                     competitor_value are the seed-0 per-qubit <Z> estimates the gate pins (§4.3c)",
                    SEED + 4
                ),
                format!(
                    "timed: sampling only, on an already-evolved state (§4.2); evolution is setup: ours \
                     {ours_setup:.6e}s, qulacs {qul_setup:.6e}s"
                ),
                format!("max/min across repeats: ours {sa:.2}x, qulacs {sb:.2}x"),
            ];
            let ours_t = self.ours[oi].threads;
            let mut ow = self.ours[oi].witness()?;
            ow.push(witness(
                "z_gate_vs_aer_marginals",
                &format!("max |z| {zo:.3} <= {Z_BOUND}"),
            ));
            let ours_arm = arm(
                "omega-backend-statevector",
                map(&[("omega-backend-statevector", OURS_VERSION)]),
                ours_t,
                map(&[
                    ("api", "sim::sample_from_state"),
                    ("rayon_threads", &ours_t.to_string()),
                ]),
                ow,
            );
            let mut cw = self.qulacs[qi].witnesses()?;
            cw.push(witness(
                "z_gate_vs_aer_marginals",
                &format!("max |z| {zc:.3} <= {Z_BOUND}"),
            ));
            let mut ck = self.qulacs[qi].engine.knobs();
            ck.insert(
                "timed_call".into(),
                "QuantumState.sampling(shots, seed) on an evolved state".into(),
            );
            let comp_arm = arm(
                &self.qulacs[qi].engine.label(),
                self.qulacs[qi].versions()?,
                self.qulacs[qi].engine.threads(),
                ck,
                cw,
            );
            let load = LoadRecord {
                load1_before: before,
                load1_after: after,
                void_above: self.void_above,
                cooldown_s: None,
                hostgate_cap: Some(self.hostgate_cap.clone()),
                gpu: None,
                external_cpu: Some(external.clone()),
            };
            let row = speed_row(
                &row_id,
                &self.recorded,
                &self.box_name,
                self.git.clone(),
                assemble::workload(&art, Quantity::Sampling, Some(SHOTS), None),
                Competitor::Qulacs,
                speed_arm(ours_arm, &a, ours_setup, ours_floor)?,
                speed_arm(comp_arm, &b, qul_setup, qul_floor)?,
                gate,
                load,
                notes,
            )?;
            return self.publish(&row);
        }
        self.void(
            &row_id,
            &format!(
                "work outside the lane stayed above {} cores after {MAX_ATTEMPTS} attempts",
                self.void_above
            ),
        )
    }

    fn publish(&mut self, row: &Row) -> Result<(), String> {
        let problems = omega_emu_compare::check(row);
        if !problems.is_empty() {
            return Err(format!("{} refused by check(): {problems:?}", row.row_id));
        }
        self.writer.write_row(row).map_err(|e| e.to_string())?;
        announce(row);
        Ok(())
    }

    fn void(&mut self, row_id: &str, reason: &str) -> Result<(), String> {
        eprintln!("VOID {row_id}: {reason}");
        let void = VoidRow {
            row_id: row_id.to_string(),
            lane: Lane::DenseCpu,
            phase: PHASE.to_string(),
            recorded: self.recorded.clone(),
            reason: reason.to_string(),
            context: json!({"box": self.box_name}),
        };
        self.writer.write_void(&void).map_err(|e| e.to_string())
    }
}

fn announce(row: &Row) {
    if let RowBody::Speed(s) = &row.body {
        let o = s.ours.timing.get().expect("present");
        let t = s.competitor.timing.get().expect("present");
        let load = s.load.get().expect("present");
        eprintln!(
            "ROW {}\t{:?}\tratio={:.4}\tours_min={:.6e}\tours_med={:.6e}\tcomp_min={:.6e}\tcomp_med={:.6e}\tload {}→{}\toverhead={}",
            row.row_id, s.classification, s.ratio_competitor_over_ours, o.min_s, o.median_s, t.min_s,
            t.median_s, load.load1_before, load.load1_after, s.overhead_dominated,
        );
    }
}

/// Untimed: what each arm returns on one cell, and the oracle. Not a row.
pub fn probe(id: &str) -> Result<(), String> {
    let art = load_artifact(id)?;
    let n = art.qubits;
    let terms = observable_terms(n);
    let obs = Observable::parse(&observable_string(n))?;
    let mut oracle = PyArm::launch(Engine::AerOracle {
        threads: ORACLE_THREADS,
    })?;
    oracle.load(id, &art.path)?;
    let r = oracle.call(json!({"op": "reference", "name": id, "terms": terms}))?;
    eprintln!("oracle {}", r["value"]);
    let ours = Ours::new(16)?;
    eprintln!("ours {:?}", ours.expectation(&art.circuit, &obs)?);
    for e in [
        Engine::Qulacs { threads: 16 },
        Engine::Qsim {
            threads: 16,
            fuse: 2,
        },
    ] {
        let mut a = PyArm::launch(e)?;
        a.load(id, &art.path)?;
        eprintln!("{} {:?}", a.engine.label(), a.expectation(id, &terms)?);
    }
    Ok(())
}

fn git_snap() -> Result<GitRev, String> {
    let compiled = option_env!("OMEGA_EMU_GIT_REV").unwrap_or("").to_string();
    let worktree = git(&["rev-parse", "HEAD"])?;
    let porcelain = git(&["status", "--porcelain"])?;
    let rev = GitRev {
        compiled_from: compiled,
        worktree: worktree.trim().to_string(),
        worktree_dirty: !porcelain.trim().is_empty(),
    };
    if let Some(why) = rev.void_reason() {
        return Err(why);
    }
    Ok(rev)
}

fn git(args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(crate::workload::repo_root())
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

fn hostname() -> Result<String, String> {
    let out = Command::new("hostname")
        .output()
        .map_err(|e| format!("hostname: {e}"))?;
    let h = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if h.is_empty() {
        return Err("empty hostname: a row must name its box".into());
    }
    Ok(h)
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
    let start = std::time::Instant::now();
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
