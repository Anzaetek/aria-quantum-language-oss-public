// SPDX-License-Identifier: Apache-2.0
//! The measurement (§4.4, §4.4a): value gate against the dense oracle first,
//! one discarded warm-up per arm, N = 5 interleaved repeats (3 over 30 s),
//! load1 before (must be quiet) and the CPU used outside the lane's own tree
//! during the row (must be under the host threshold).

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use omega_emu_compare::{
    Arm, Classification, CpuCensus, Derivation, GateReference, GitRev, Lane, LoadRecord,
    PathWitness, PathWitnesses, Precision, Quantity, Row, RowBody, RowWriter, SpeedArm, SpeedRow,
    Threads, Timing, ValueGate, VoidRow, Witness, Workload, GATE_REL_F32, GATE_REL_F64,
    OVERHEAD_FLOOR_MULTIPLE, OVERHEAD_RULE, TIE_RULE, WIN_RATIO_THRESHOLD,
};
use serde_json::json;

use crate::monoprop::Monoprop;
use crate::workload::{
    fixture_exact_cells, hea_cell, oracle, ours, ours_floor, ours_witness, Cell, OursKnobs,
    OURS_VERSION,
};

pub const PHASE: &str = "E7";
/// §2's matched-accuracy tolerance, pinned before either arm's HEA sweep was
/// timed: the existing f32 gate constant, |rel err| ≤ 1e-6 against the dense
/// oracle. Reused rather than chosen, so it could not be picked to suit a
/// number.
pub const MATCHED_REL: f64 = GATE_REL_F32;
/// The mapped truncation grid, identical on both arms: weight cap (None = no
/// cap, i.e. n) and coefficient floor (0 = none). Ours: `max_weight`,
/// `coeff_min`. monoprop: `cutoff`, `lower_atol`.
pub const WEIGHTS: &[Option<usize>] = &[None, Some(10), Some(8), Some(6), Some(5), Some(4)];
pub const ATOLS: &[f64] = &[0.0, 1e-10, 1e-8, 1e-6];
const HEA: &str = "hea-14q-d12";
const MAX_ATTEMPTS: u32 = 3;

struct Ctx {
    git: GitRev,
    recorded: String,
    box_name: String,
    void_above: f64,
    hostgate_cap: String,
    writer: RowWriter,
    mp: Monoprop,
}

/// monoprop's truncation setting, in its knob names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpKnobs {
    pub cutoff: u32,
    pub lower_atol: Option<f64>,
}

/// One grid point on both arms: weight cap `w` (None = n) and coefficient
/// floor `a` (0 = none), in each engine's own knob names.
pub fn mapped(n: u32, w: Option<usize>, a: f64) -> (OursKnobs, MpKnobs) {
    (
        OursKnobs {
            coeff_min: a,
            max_weight: w.filter(|w| *w < n as usize),
        },
        MpKnobs {
            cutoff: w.map(|w| w as u32).unwrap_or(n).min(n),
            lower_atol: (a > 0.0).then_some(a),
        },
    )
}

pub fn measure(out: &Path, append: bool) -> Result<(), String> {
    if out.exists() && !append {
        return Err(format!(
            "{} exists; pass --append to add to it",
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
        "pauliprop lane on {box_name} {recorded} rev {}; void above {void_above}",
        git.worktree
    );
    let mp = Monoprop::launch()?;
    let writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    let mut ctx = Ctx {
        git,
        recorded,
        box_name,
        void_above,
        hostgate_cap,
        writer,
        mp,
    };
    for cell in fixture_exact_cells()? {
        ctx.exact_row(&cell)?;
    }
    let hea = hea_cell(HEA)?;
    ctx.exact_row(&hea)?;
    ctx.matched_row(&hea)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

impl Ctx {
    fn load(&mut self, cell: &Cell) -> Result<(), String> {
        let mut req = cell.monoprop_load.clone();
        req["name"] = json!(cell.id);
        self.mp.call(req).map(|_| ())
    }

    fn mp_run(&mut self, cell: &Cell, k: MpKnobs) -> Result<(f64, f64, u64), String> {
        self.mp
            .expectation(&cell.id, &cell.terms, k.cutoff, k.lower_atol)
    }

    fn exact_row(&mut self, cell: &Cell) -> Result<(), String> {
        self.load(cell)?;
        let (ok, mk) = mapped(cell.qubits, None, 0.0);
        let reference = oracle(cell)?;
        let notes = vec![format!(
            "both arms exact: ours max_weight None, coeff_min 0; monoprop cutoff {} (= n), lower_atol None",
            cell.qubits
        )];
        self.row(
            cell,
            &format!(
                "{}-exact-vs-monoprop",
                cell.id.trim_end_matches("-vs-monoprop")
            ),
            ok,
            mk,
            reference,
            GATE_REL_F64,
            notes,
        )
    }

    /// §2: both arms tuned on one mapped grid to |rel err| ≤ MATCHED_REL, the
    /// fastest admissible setting each, then timed against each other.
    fn matched_row(&mut self, cell: &Cell) -> Result<(), String> {
        let reference = oracle(cell)?;
        let n = cell.qubits;
        let mut ours_best: Option<(f64, OursKnobs, f64)> = None;
        let mut mp_best: Option<(f64, MpKnobs, f64)> = None;
        let mut table = Vec::new();
        for &w in WEIGHTS {
            for &a in ATOLS {
                let (ok, mk) = mapped(n, w, a);
                let label = format!(
                    "w={} atol={a:e}",
                    w.map(|x| x.to_string()).unwrap_or("n".into())
                );
                // Ours: one warm-up, min of 3. A refusal (vacuous bound) is inadmissible.
                let ours_cell = match (0..4)
                    .map(|_| ours(cell, ok))
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(runs) => {
                        let v = runs[0].0;
                        let t = runs[1..].iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
                        let err = rel(v, reference);
                        let admissible = err <= MATCHED_REL;
                        if admissible && ours_best.is_none_or(|b| t < b.0) {
                            ours_best = Some((t, ok, err));
                        }
                        format!(
                            "ours err {err:.1e} {t:.3e}s dm {:.1e}{}",
                            runs[0].2.dropped_mass,
                            if admissible { "" } else { " ✗" }
                        )
                    }
                    Err(e) => format!("ours refused ({})", e.chars().take(40).collect::<String>()),
                };
                let mut runs = Vec::new();
                for _ in 0..4 {
                    runs.push(self.mp_run(cell, mk)?);
                }
                let v = runs[0].0;
                let t = runs[1..].iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
                let err = rel(v, reference);
                let admissible = err <= MATCHED_REL;
                if admissible && mp_best.is_none_or(|b| t < b.0) {
                    mp_best = Some((t, mk, err));
                }
                table.push(format!(
                    "{label}: {ours_cell}; monoprop err {err:.1e} {t:.3e}s{}",
                    if admissible { "" } else { " ✗" }
                ));
            }
        }
        for l in &table {
            eprintln!("  grid {l}");
        }
        let (_, ok, _) = ours_best.ok_or("no admissible setting on our side")?;
        let (_, mk, _) = mp_best.ok_or("no admissible setting on monoprop's side")?;
        let notes = vec![
            format!(
                "matched ACCURACY (§2): each arm's fastest setting on one mapped grid with |rel err| ≤ {MATCHED_REL:e} \
                 against the dense oracle. Chosen: ours max_weight {:?} coeff_min {:e}; monoprop cutoff {} lower_atol {:?}. \
                 The knobs are not one contract (tests/monoprop_xcheck.rs), so the rows compare equal accuracy, not equal knobs.",
                ok.max_weight, ok.coeff_min, mk.cutoff, mk.lower_atol
            ),
            format!("grid (one warm-up, min of 3 per setting; ✗ = outside {MATCHED_REL:e}): {}", table.join(" | ")),
        ];
        self.row(
            cell,
            &format!("pp-{}-o-matched-1e-6-vs-monoprop", cell.id),
            ok,
            mk,
            reference,
            MATCHED_REL,
            notes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn row(
        &mut self,
        cell: &Cell,
        row_id: &str,
        ok: OursKnobs,
        mk: MpKnobs,
        reference: f64,
        tol: f64,
        mut notes: Vec<String>,
    ) -> Result<(), String> {
        let row_id = if row_id.starts_with("pp-") {
            row_id.to_string()
        } else {
            format!("pp-{row_id}")
        };
        let ours_floor = (0..5)
            .map(|_| ours_floor())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let mut mf = f64::INFINITY;
        for _ in 0..5 {
            mf = mf.min(self.mp.floor()?);
        }
        for attempt in 1..=MAX_ATTEMPTS {
            let before = wait_quiet(self.void_above)?;
            let census = CpuCensus::start(&[std::process::id(), self.mp.pid()])?;
            let (_, ours_first, _) = ours(cell, ok)?;
            let (_, mp_first, _) = self.mp_run(cell, mk)?;
            let (mut a, mut b, mut av, mut bv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut last_cert = None;
            let mut mp_terms = 0;
            let mut n = 5usize;
            while a.len() < n {
                let (va, ta, cert) = ours(cell, ok)?;
                let (vb, tb, terms) = self.mp_run(cell, mk)?;
                a.push(ta);
                b.push(tb);
                av.push(va);
                bv.push(vb);
                last_cert = Some(cert);
                mp_terms = terms;
                if a.len() >= 3 && a.iter().chain(&b).any(|t| *t > 30.0) {
                    n = n.min(3);
                }
            }
            let external = census.finish()?;
            let after = load1()?;
            if omega_emu_compare::void_at_load(external.external_cores_during_row, self.void_above)
            {
                eprintln!(
                    "  {row_id}: {:.2} external cores (attempt {attempt}); re-running",
                    external.external_cores_during_row
                );
                continue;
            }
            let og = av.iter().map(|v| rel(*v, reference)).fold(0.0, f64::max);
            let cg = bv.iter().map(|v| rel(*v, reference)).fold(0.0, f64::max);
            if og > tol || cg > tol {
                return self.void(&row_id, &format!(
                    "disagreement finding: oracle {reference:+.15}, ours gap {og:.3e}, monoprop gap {cg:.3e}, tol {tol:e}"
                ));
            }
            let cert = last_cert.expect("ran");
            if ok.max_weight.is_none() && ok.coeff_min == 0.0 && cert.dropped_mass != 0.0 {
                return self.void(
                    &row_id,
                    &format!("an exact run reported dropped mass {}", cert.dropped_mass),
                );
            }
            notes.push(format!(
                "ours: dropped_mass {:.3e} (a bound on |Δ⟨O⟩|), final terms {}, peak terms {}; monoprop: {mp_terms} terms, no error quantity reported",
                cert.dropped_mass, cert.final_terms, cert.peak_terms
            ));
            notes.push(format!(
                "first call (untimed warm-up): ours {ours_first:.6e}s, monoprop {mp_first:.6e}s"
            ));
            let ours_t = timing(&a)?;
            let comp_t = timing(&b)?;
            let ratio = comp_t.min_s / ours_t.min_s;
            let multiple = comp_t.min_s / mf;
            let deriv = Derivation {
                tie_rule: TIE_RULE.into(),
                win_ratio_threshold: WIN_RATIO_THRESHOLD,
                ours_min_median_s: [ours_t.min_s, ours_t.median_s],
                competitor_min_median_s: [comp_t.min_s, comp_t.median_s],
                ratio_competitor_over_ours: ratio,
                intervals_disjoint: ours_t.median_s < comp_t.min_s
                    || comp_t.median_s < ours_t.min_s,
                ratio_beyond_threshold: ratio >= WIN_RATIO_THRESHOLD
                    || ratio <= 1.0 / WIN_RATIO_THRESHOLD,
                overhead_rule: OVERHEAD_RULE.into(),
                competitor_per_call_floor_s: mf,
                competitor_min_over_floor: multiple,
                floor_arm_min_s: None,
            };
            let class = if deriv.ratio_beyond_threshold && deriv.intervals_disjoint {
                if ratio > 1.0 {
                    Classification::Win
                } else {
                    Classification::Loss
                }
            } else {
                Classification::Tie
            };
            let mut ow = ours_witness()?;
            ow.push(w(
                "value_gate_vs_dense_f64",
                &format!("max rel gap {og:.3e} <= {tol:e}"),
            ));
            ow.push(w(
                "certificate_dropped_mass",
                &format!("{:.3e}", cert.dropped_mass),
            ));
            let mut cw = self.mp.witnesses()?;
            cw.push(w(
                "value_gate_vs_dense_f64",
                &format!("max rel gap {cg:.3e} <= {tol:e}"),
            ));
            let ours_arm = Arm {
                name: "omega-backend-pauliprop".into(),
                versions: Witness::present(map(&[("omega-backend-pauliprop", OURS_VERSION)])),
                threads: Witness::present(Threads {
                    threads: 1,
                    blas: "none".into(),
                }),
                knobs: map(&[
                    ("api", "PauliPropBackend::expectation_with_certificate"),
                    ("coeff_min", &format!("{:e}", ok.coeff_min)),
                    ("max_weight", &format!("{:?}", ok.max_weight)),
                    (
                        "threads",
                        "1 (single-threaded by design: no rayon in the engine)",
                    ),
                ]),
                executed_path: PathWitnesses::new(ow),
            };
            let comp_arm = Arm {
                name: "monoprop-0.9.0".into(),
                versions: Witness::present(self.mp.versions()),
                threads: Witness::present(Threads { threads: self.mp.threads(), blas: "none".into() }),
                knobs: map(&[
                    ("timed_call", "PauliPropagator(obs, [], cutoff, lower_atol) + propagate + expectation_value"),
                    ("cutoff", &mk.cutoff.to_string()),
                    ("lower_atol", &format!("{:?}", mk.lower_atol)),
                    ("threads", "CPU affinity (monoprop has no thread knob; 6.1 ms on 32 CPUs vs 14.7 ms on 1, exact 14q HEA)"),
                ]),
                executed_path: PathWitnesses::new(cw),
            };
            let row = Row {
                row_id: row_id.clone(),
                lane: Lane::Pauliprop,
                phase: PHASE.into(),
                recorded: self.recorded.clone(),
                box_name: Witness::present(self.box_name.clone()),
                git: Witness::present(self.git.clone()),
                circuit: Workload {
                    id: cell.id.clone(),
                    file: cell.file.clone(),
                    sha256: Witness::present(cell.sha256.clone()),
                    generator: cell.generator.clone(),
                    seed: cell.seed,
                    qubits: cell.qubits,
                    depth: cell.depth,
                    chi: None,
                    shots: None,
                    quantity: Quantity::Expectation,
                    observable: Some(cell.observable_text.clone()),
                },
                precision: Precision {
                    ours: "f64".into(),
                    competitor: "f64".into(),
                    matched: true,
                    note: Some("both engines propagate double-precision coefficients; the exact rows agree with the dense oracle to <= 1e-12".into()),
                },
                body: RowBody::Speed(Box::new(SpeedRow {
                    ours: SpeedArm {
                        arm: ours_arm,
                        timing: Witness::present(ours_t),
                        fixed_cost_s: Witness::present(0.0),
                        per_call_floor_s: Witness::present(ours_floor),
                    },
                    competitor: SpeedArm {
                        arm: comp_arm,
                        timing: Witness::present(comp_t),
                        fixed_cost_s: Witness::present(0.0),
                        per_call_floor_s: Witness::present(mf),
                    },
                    context_arms: BTreeMap::new(),
                    value_gate: Witness::present(ValueGate {
                        tolerance_rel: tol,
                        reference: GateReference::Oracle {
                            arm: "omega-backend-statevector (dense f64)".into(),
                            versions: map(&[("omega-backend-statevector", OURS_VERSION)]),
                            value: json!(reference),
                        },
                        ours_value: json!(av[0]),
                        competitor_value: json!(bv[0]),
                        ours_gap: og,
                        competitor_gap: cg,
                        truncation: None,
                        passed: true,
                    }),
                    load: Witness::present(LoadRecord {
                        load1_before: before,
                        load1_after: after,
                        void_above: self.void_above,
                        cooldown_s: None,
                        hostgate_cap: Some(self.hostgate_cap.clone()),
                        gpu: None,
                        external_cpu: Some(external),
                    }),
                    ratio_competitor_over_ours: ratio,
                    classification: class,
                    overhead_dominated: multiple < OVERHEAD_FLOOR_MULTIPLE,
                    floor_check: None,
                    derivation: Witness::present(deriv),
                })),
                notes,
            };
            let problems = omega_emu_compare::check(&row);
            if !problems.is_empty() {
                return Err(format!("{row_id} refused by check(): {problems:?}"));
            }
            self.writer.write_row(&row).map_err(|e| e.to_string())?;
            eprintln!(
                "ROW {row_id}\t{class:?}\tratio={ratio:.4}\tours_min={:.6e}\tmonoprop_min={:.6e}\toverhead={}",
                a.iter().cloned().fold(f64::INFINITY, f64::min),
                b.iter().cloned().fold(f64::INFINITY, f64::min),
                multiple < OVERHEAD_FLOOR_MULTIPLE
            );
            return Ok(());
        }
        self.void(
            &row_id,
            &format!(
                "work outside the lane stayed above {} cores after {MAX_ATTEMPTS} attempts",
                self.void_above
            ),
        )
    }

    fn void(&mut self, row_id: &str, reason: &str) -> Result<(), String> {
        eprintln!("VOID {row_id}: {reason}");
        self.writer
            .write_void(&VoidRow {
                row_id: row_id.into(),
                lane: Lane::Pauliprop,
                phase: PHASE.into(),
                recorded: self.recorded.clone(),
                reason: reason.into(),
                context: json!({"box": self.box_name}),
            })
            .map_err(|e| e.to_string())
    }
}

fn rel(v: f64, r: f64) -> f64 {
    (v - r).abs() / r.abs().max(1.0)
}

fn w(name: &str, observed: &str) -> PathWitness {
    PathWitness {
        name: name.into(),
        observed: observed.into(),
        asserted: true,
    }
}

fn map(p: &[(&str, &str)]) -> BTreeMap<String, String> {
    p.iter().map(|(k, v)| ((*k).into(), (*v).into())).collect()
}

fn timing(t: &[f64]) -> Result<Timing, String> {
    if t.len() < 3 || t.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(format!("bad timings {t:?}"));
    }
    let mut s = t.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let m = if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2])
    };
    Ok(Timing {
        min_s: s[0],
        median_s: m,
        repeats: s.len() as u32,
        warmup_discarded: true,
        interleaved: true,
    })
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
        .current_dir(crate::workload::repo_root())
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn hostname() -> Result<String, String> {
    let out = Command::new("hostname")
        .output()
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn today() -> Result<String, String> {
    let out = Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn load1() -> Result<f64, String> {
    let s = std::fs::read_to_string("/proc/loadavg").map_err(|e| e.to_string())?;
    s.split_whitespace()
        .next()
        .ok_or("empty loadavg")?
        .parse()
        .map_err(|e| format!("{e}"))
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
        std::thread::sleep(Duration::from_secs(20));
    }
}

/// Untimed: each arm's value on every cell against the oracle. Not a row.
pub fn probe() -> Result<(), String> {
    let mut mp = Monoprop::launch()?;
    let mut cells = fixture_exact_cells()?;
    cells.push(hea_cell(HEA)?);
    for c in &cells {
        let mut req = c.monoprop_load.clone();
        req["name"] = json!(c.id);
        mp.call(req)?;
        let r = oracle(c)?;
        let (o, ot, cert) = ours(
            c,
            OursKnobs {
                coeff_min: 0.0,
                max_weight: None,
            },
        )?;
        let (m, mt, _) = mp.expectation(&c.id, &c.terms, c.qubits, None)?;
        eprintln!(
            "{:<32} oracle {r:+.15} ours {o:+.15} ({:.1e}, {ot:.3e}s, dm {:.1e}) monoprop {m:+.15} ({:.1e}, {mt:.3e}s)",
            c.id, rel(o, r), cert.dropped_mass, rel(m, r)
        );
    }
    Ok(())
}

pub fn append_void(out: &Path, row_id: &str, reason: &str) -> Result<(), String> {
    let mut writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    writer
        .write_void(&VoidRow {
            row_id: row_id.into(),
            lane: Lane::Pauliprop,
            phase: PHASE.into(),
            recorded: today()?,
            reason: reason.into(),
            context: json!({"source": "hostgate --watch killed the process tree"}),
        })
        .map_err(|e| e.to_string())
}
