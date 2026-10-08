// SPDX-License-Identifier: Apache-2.0
//! MPS lane, quimb `CircuitMPS` against our `expectation` path.
//!
//! Timed region, both arms: the gate sweep plus a local Z expectation.
//! Ours is `MpsBackend::with_svd_fn` + `expectation`. Theirs is
//! `CircuitMPS.apply_gate` + `local_expectation` (one-site 2×2 reduced
//! density matrix). Neither side calls `to_statevector` / `to_dense` inside
//! the timer.
//!
//! The akilles 32-core replicate is not this binary. Depth ≥ 12 is where the
//! SVD dominates, and that is where a 32-wide host can invert an M4 ranking;
//! depth 4 does not saturate the bond and is not the row that would flip.
//! No GPU.

#[allow(dead_code)]
mod publish;
#[allow(dead_code)]
mod quimb_lane;

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use omega_emu_compare::{CpuCensus, ExternalCpu, QuietProbe};
use serde_json::{json, Value};

use publish::{measured_context, speed_id, LanePublisher, PublishOpen, TimedFacts};
use quimb_lane::{
    abs_gap, admit_saturating_row, blind_spot, classify_speed, cnot_on_plus_plus, dense_z0,
    empty_pair, format_capability_row, grid_chis, grid_shapes, heavy_row, knob_sweep, load_hea,
    mapping_shapes, min_median, parse_uptime_load1, relative_error, requires_bond_saturation,
    row_void_at_load, run_ours, run_ours_adaptive, select_knob, speed_headline, timed_repeats,
    truncation_matched, Artifact, CutoffMode, Knob, MapCell, MapObs,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    let path = args
        .next()
        .unwrap_or_else(|| "/tmp/e3-quimb-map.json".to_string());
    match cmd.as_str() {
        "witness" => witness(),
        "map" => map(&path),
        "time" => time(&path),
        _ => {
            eprintln!("usage: mps_quimb_compare witness|map [FILE]|time FILE");
            std::process::exit(2);
        }
    }
}

fn compiled_rev() -> &'static str {
    env!("OMEGA_BUILD_REV")
}

fn git_stdout(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn runtime_stamp() -> String {
    let Some(rev) = git_stdout(&["rev-parse", "HEAD"]) else {
        return "unknown".into();
    };
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .is_some_and(|o| !o.stdout.iter().all(|b| b.is_ascii_whitespace()));
    if dirty {
        format!("{rev}-dirty")
    } else {
        rev
    }
}

/// A timed row is published only from a clean tree whose compile stamp is
/// the checkout it is running in.
fn require_publishable_rev() {
    let compiled = compiled_rev();
    let runtime = runtime_stamp();
    if compiled != runtime || compiled == "unknown" || compiled.ends_with("-dirty") {
        eprintln!("rev witness refused: compiled={compiled} runtime={runtime}");
        std::process::exit(2);
    }
    println!("rev={compiled}");
}

fn bridges_python() -> (PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../omega-bridges/python");
    (
        root.join(".venv-quimb/bin/python"),
        root.join("quimb_mps_arm.py"),
    )
}

struct QuimbProc {
    child: std::process::Child,
    stdin: std::io::BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    hello: Value,
    next_id: u64,
}

struct QuimbEval {
    expectation: f64,
    max_bond: usize,
    fidelity_estimate: f64,
    error_estimate: f64,
    seconds: f64,
}

impl QuimbProc {
    fn spawn() -> Result<Self, String> {
        let (py, script) = bridges_python();
        if !py.is_file() {
            return Err(format!(
                "quimb venv is not a venv: {} is missing. make quimb-venv, which imports quimb before it stamps .installed",
                py.display()
            ));
        }
        let probe = Command::new(&py)
            .args(["-c", "import quimb, quimb.tensor"])
            .output()
            .map_err(|e| e.to_string())?;
        if !probe.status.success() {
            return Err(format!(
                "quimb venv failed its import probe: {}",
                String::from_utf8_lossy(&probe.stderr)
            ));
        }
        let mut child = Command::new(&py)
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONWARNINGS", "ignore")
            .spawn()
            .map_err(|e| e.to_string())?;
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));
        let stdin = std::io::BufWriter::new(child.stdin.take().expect("stdin piped"));
        let mut proc = Self {
            child,
            stdin,
            stdout,
            hello: Value::Null,
            next_id: 1,
        };
        let hello = proc.read_value()?;
        let z0 = hello
            .get("self_check_z0")
            .and_then(Value::as_f64)
            .ok_or("quimb hello missing self_check_z0")?;
        if (z0 - 1.0).abs() > 1e-8 {
            return Err(format!("quimb |00> <Z_0> self-check was {z0}, want 1"));
        }
        if hello.get("to_dense_called").and_then(Value::as_bool) == Some(true) {
            return Err("quimb arm reported to_dense during startup".into());
        }
        proc.hello = hello;
        Ok(proc)
    }

    /// The quimb process's pid, counted as the lane's own tree by §4.4a.
    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn read_value(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        let n = self
            .stdout
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("quimb arm closed stdout".into());
        }
        serde_json::from_str(&line).map_err(|e| format!("quimb JSON: {e} in {line}"))
    }

    fn eval(
        &mut self,
        n: u32,
        chi: usize,
        knob: Knob,
        gates: &[quimb_lane::GateRec],
    ) -> Result<QuimbEval, String> {
        let id = self.next_id;
        self.next_id += 1;
        let gates = gates
            .iter()
            .map(|g| {
                json!({
                    "name": g.name,
                    "qubits": g.qubits,
                    "params": g.params,
                })
            })
            .collect::<Vec<_>>();
        let req = json!({
            "id": id,
            "n": n,
            "chi": chi,
            "cutoff": knob.cutoff,
            "cutoff_mode": knob.mode.as_str(),
            "gates": gates,
        });
        writeln!(self.stdin, "{req}").map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;
        let resp = self.read_value()?;
        if let Some(err) = resp.get("error") {
            return Err(format!("quimb job {id}: {err}"));
        }
        if resp.get("timed_path").and_then(Value::as_str)
            != Some("CircuitMPS.apply_gate+local_expectation")
        {
            return Err(format!(
                "quimb job {id} did not name the MPS expectation path"
            ));
        }
        if resp.get("to_dense_called").and_then(Value::as_bool) != Some(false) {
            return Err(format!("quimb job {id} reported to_dense"));
        }
        Ok(QuimbEval {
            expectation: resp["expectation"].as_f64().ok_or("expectation")?,
            max_bond: resp["max_bond"].as_u64().ok_or("max_bond")? as usize,
            fidelity_estimate: resp["fidelity_estimate"].as_f64().ok_or("fidelity")?,
            error_estimate: resp["error_estimate"].as_f64().ok_or("error_estimate")?,
            seconds: resp["seconds"].as_f64().ok_or("seconds")?,
        })
    }
}

impl Drop for QuimbProc {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, r#"{{"op":"quit"}}"#);
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

fn load1() -> Result<f64, String> {
    let out = Command::new("uptime").output().map_err(|e| e.to_string())?;
    parse_uptime_load1(&String::from_utf8_lossy(&out.stdout))
}

/// §4.4's threshold for this host, computed once so the gate and the
/// published `void_above` cannot disagree.
fn void_above() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        omega_emu_compare::host_load_void_above().unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2);
        })
    })
}

/// §4.4a: wait until work outside the lane (this process and the quimb
/// process) is under the host threshold, judged by the external-CPU census,
/// not the 1-minute load average.
fn wait_quiet(quimb_pid: u32) -> QuietProbe {
    omega_emu_compare::wait_quiet_external(void_above(), &own_pids(quimb_pid), load1)
        .unwrap_or_else(|e| {
            eprintln!("{e}; not publishing a number");
            std::process::exit(2);
        })
}

fn own_pids(quimb_pid: u32) -> Vec<u32> {
    vec![std::process::id(), quimb_pid]
}

fn cooldown_if_heavy(n: u32, depth: usize) {
    if heavy_row(n, depth) {
        eprintln!("cooldown 60s after {n}q depth {depth}");
        std::thread::sleep(Duration::from_secs(60));
    }
}

fn mode_of(s: &str) -> Result<CutoffMode, String> {
    match s {
        "rsum2" => Ok(CutoffMode::Rsum2),
        "rel" => Ok(CutoffMode::Rel),
        "abs" => Ok(CutoffMode::Abs),
        other => Err(format!("unknown cutoff mode {other}")),
    }
}

fn knob_json(k: Knob) -> Value {
    json!({"cutoff": k.cutoff, "cutoff_mode": k.mode.as_str()})
}

fn hostname() -> String {
    Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn witness() {
    println!("rev_compiled={}", compiled_rev());
    println!("rev_runtime={}", runtime_stamp());
    let built = cnot_on_plus_plus();
    let chi = 32usize;
    let ours = run_ours(&built.circuit, chi).expect("our ghost");
    println!(
        "ours ghost max_bond={} svd_calls={} max_cols={} discarded={:.3e}",
        ours.max_bond, ours.svd_calls, ours.max_cols, ours.discarded_weight
    );
    assert!(ours.svd_calls > 0, "ghost must execute an SVD");
    assert!(ours.max_bond < chi, "ghost bond reached chi");
    let mut quimb = QuimbProc::spawn().expect("quimb");
    let q = quimb
        .eval(
            built.n,
            chi,
            Knob {
                cutoff: 0.0,
                mode: CutoffMode::Rsum2,
            },
            &built.gates,
        )
        .expect("quimb ghost");
    println!(
        "quimb ghost max_bond={} fidelity_estimate={:.6e} error_estimate={:.6e}",
        q.max_bond, q.fidelity_estimate, q.error_estimate
    );
    match admit_saturating_row(
        chi,
        q.max_bond,
        ours.max_bond,
        ours.svd_calls,
        ours.max_cols,
    ) {
        Err(reason) => {
            println!("ROW kind=refused fixture=cnot-on-plus-plus chi={chi}");
            println!("reason={reason}");
            println!(
                "blind_spot=does not catch a witness that refuses every row; saturated admission is a separate test"
            );
        }
        Ok(()) => {
            eprintln!("witness failed: the ghost was admitted");
            std::process::exit(1);
        }
    }
}

fn map(path: &str) {
    println!("rev_compiled={}", compiled_rev());
    println!("rev_runtime={}", runtime_stamp());
    println!("host={}", hostname());
    println!("rayon_threads={}", rayon::current_num_threads());
    println!(
        "load_void_above={} cores={}",
        void_above(),
        omega_emu_compare::host_cores().unwrap_or(0)
    );
    println!(
        "svd_delegate={}",
        quimb_lane::selected_svd_kernel().as_str()
    );
    let mut quimb = QuimbProc::spawn().expect("quimb");
    println!("quimb_hello={}", quimb.hello);
    let mut cells_json = Vec::new();
    let mut cells = Vec::new();
    for &(n, depth) in mapping_shapes() {
        for &chi in grid_chis() {
            let load_before = wait_quiet(quimb.pid()).load1;
            let artifact = load_hea(n, depth).expect("committed HEA");
            let built = &artifact.built;
            println!(
                "MAP artifact id={} file={} sha256={}",
                artifact.id, artifact.file, artifact.sha256
            );
            let oracle = dense_z0(&built.circuit).expect("dense");
            let ours = run_ours(&built.circuit, chi).expect("ours");
            let adaptive = run_ours_adaptive(&built.circuit, chi).expect("adaptive");
            let our_gap = abs_gap(ours.value, oracle);
            let adaptive_gap = abs_gap(adaptive.value, oracle);
            println!(
                "MAP cell n={n} depth={depth} chi={chi} load_before={load_before:.2} oracle={oracle:.12e} ours={:.12e} our_gap={our_gap:.3e} our_bond={} our_discarded={:.3e} our_fidelity={:.6e} adaptive_eps={:.1e} adaptive_gap={adaptive_gap:.3e} adaptive_bond={}",
                ours.value,
                ours.max_bond,
                ours.discarded_weight,
                ours.fidelity_estimate,
                omega_backend_mps::select::AUTO_EPS,
                adaptive.max_bond
            );
            let mut obs = Vec::new();
            let mut obs_json = Vec::new();
            for knob in knob_sweep() {
                let q = quimb.eval(n, chi, knob, &built.gates).expect("quimb map");
                let gap = abs_gap(q.expectation, oracle);
                println!(
                    "MAP quimb n={n} depth={depth} chi={chi} cutoff={} mode={} value={:.12e} gap={gap:.3e} max_bond={} fidelity={:.6e} error={:.6e} matched={} saturated={}",
                    knob.cutoff,
                    knob.mode.as_str(),
                    q.expectation,
                    q.max_bond,
                    q.fidelity_estimate,
                    q.error_estimate,
                    truncation_matched(our_gap, gap),
                    q.max_bond == chi
                );
                obs.push(MapObs {
                    knob,
                    max_bond: q.max_bond,
                    abs_gap: gap,
                });
                obs_json.push(json!({
                    "cutoff": knob.cutoff,
                    "cutoff_mode": knob.mode.as_str(),
                    "max_bond": q.max_bond,
                    "abs_gap": gap,
                    "expectation": q.expectation,
                    "fidelity_estimate": q.fidelity_estimate,
                    "error_estimate": q.error_estimate,
                }));
            }
            let load_after = load1().expect("uptime");
            println!("MAP cell_done n={n} depth={depth} chi={chi} load_after={load_after:.2}");
            cells.push(MapCell {
                chi,
                our_abs_gap: our_gap,
                obs,
            });
            cells_json.push(json!({
                "n": n,
                "depth": depth,
                "chi": chi,
                "oracle": oracle,
                "ours_expectation": ours.value,
                "our_abs_gap": our_gap,
                "our_max_bond": ours.max_bond,
                "our_discarded_weight": ours.discarded_weight,
                "our_fidelity_estimate": ours.fidelity_estimate,
                "adaptive_eps": omega_backend_mps::select::AUTO_EPS,
                "adaptive_abs_gap": adaptive_gap,
                "adaptive_max_bond": adaptive.max_bond,
                "adaptive_discarded_weight": adaptive.discarded_weight,
                "load_before": load_before,
                "load_after": load_after,
                "obs": obs_json,
            }));
            cooldown_if_heavy(n, depth);
        }
    }
    let selected = select_knob(&cells);
    let doc = json!({
        "compiled_rev": compiled_rev(),
        "runtime_rev": runtime_stamp(),
        "selected": selected.as_ref().ok().map(|k| knob_json(*k)),
        "select_error": selected.as_ref().err().cloned(),
        "cells": cells_json,
    });
    std::fs::write(path, serde_json::to_string_pretty(&doc).unwrap()).expect("write map");
    match selected {
        Ok(k) => println!(
            "MAP selected cutoff={} mode={} file={path}",
            k.cutoff,
            k.mode.as_str()
        ),
        Err(e) => {
            eprintln!("MAP selected none: {e}");
            std::process::exit(1);
        }
    }
}

fn jsonl_path() -> PathBuf {
    // The default sits beside the script, which is where the results doc says
    // a lane's JSONL lives. A measurement run sets E3_JSONL outside the
    // worktree: creating the file inside the tree would make every later row
    // a dirty-tree measurement while the row claimed it was clean.
    if let Ok(path) = std::env::var("E3_JSONL") {
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/mps_quimb_compare/e3.jsonl")
}

fn hostgate_cap() -> String {
    omega_emu_compare::hostgate_cap().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    })
}

fn measure_floors(quimb: &mut QuimbProc) -> (f64, f64) {
    let built = empty_pair();
    let knob = Knob {
        cutoff: 0.0,
        mode: CutoffMode::Rsum2,
    };
    let chi = 2usize;
    run_ours(&built.circuit, chi).expect("our floor warmup");
    quimb
        .eval(built.n, chi, knob, &built.gates)
        .expect("quimb floor warmup");
    let mut ours = Vec::new();
    let mut qs = Vec::new();
    for _ in 0..3 {
        ours.push(run_ours(&built.circuit, chi).expect("our floor").seconds);
        qs.push(
            quimb
                .eval(built.n, chi, knob, &built.gates)
                .expect("quimb floor")
                .seconds,
        );
    }
    let (omin, _) = min_median(&ours).expect("our floor samples");
    let (qmin, _) = min_median(&qs).expect("quimb floor samples");
    println!(
        "FLOOR ours_min={omin:.6e} quimb_min={qmin:.6e} circuit=2q-no-gates chi=2 N=3 warmup_discarded=1 interleaved=1"
    );
    (omin, qmin)
}

fn time(path: &str) {
    require_publishable_rev();
    let cap = hostgate_cap();
    let raw = std::fs::read_to_string(path).expect("map file");
    let doc: Value = serde_json::from_str(&raw).expect("map json");
    if doc["compiled_rev"].as_str() != Some(compiled_rev()) {
        eprintln!("map file rev does not match this binary; re-run map");
        std::process::exit(2);
    }
    let sel = doc.get("selected").filter(|v| !v.is_null());
    let Some(sel) = sel else {
        eprintln!("map file has no selected cutoff; refusing to time");
        std::process::exit(2);
    };
    let knob = Knob {
        cutoff: sel["cutoff"].as_f64().expect("cutoff"),
        mode: mode_of(sel["cutoff_mode"].as_str().expect("mode")).unwrap(),
    };
    println!("host={}", hostname());
    println!("rayon_threads={}", rayon::current_num_threads());
    println!(
        "load_void_above={} cores={}",
        void_above(),
        omega_emu_compare::host_cores().unwrap_or(0)
    );
    println!(
        "svd_delegate={} certificate_kernel=custom (shim over the delegate)",
        quimb_lane::selected_svd_kernel().as_str()
    );
    let mut quimb = QuimbProc::spawn().expect("quimb");
    println!("quimb_hello={}", quimb.hello);
    println!(
        "knob cutoff={} mode={} (from the mapping, not from a default)",
        knob.cutoff,
        knob.mode.as_str()
    );
    let (ours_floor, quimb_floor) = measure_floors(&mut quimb);
    let jsonl = jsonl_path();
    if jsonl.exists() {
        std::fs::remove_file(&jsonl).expect("replace previous jsonl");
    }
    let mut publisher = LanePublisher::open(
        PublishOpen {
            path: jsonl.to_str().expect("jsonl path").to_string(),
            rev: compiled_rev().to_string(),
            hostgate_cap: cap.clone(),
            void_above: void_above(),
            ours_floor_s: ours_floor,
            quimb_floor_s: quimb_floor,
            ours_threads: rayon::current_num_threads() as u32,
            // The kernel that was INSTALLED, not the one this platform
            // defaults to. Those differ exactly when `E3_SVD_KERNEL` is set,
            // which is the case the field has to be able to report or the
            // override would be unobservable in the row it changes.
            svd_delegate: quimb_lane::selected_svd_kernel().as_str().to_string(),
            knob,
        },
        &quimb.hello,
    )
    .expect("row writer");
    println!("jsonl={}", jsonl.display());
    println!("hostgate_cap={cap}");
    let mut capability_done = false;
    let mut cooldown_before = 0u32;
    for &(n, depth) in grid_shapes() {
        for &chi in grid_chis() {
            run_timed_row(
                &mut quimb,
                &mut publisher,
                RowJob {
                    n,
                    depth,
                    chi,
                    knob,
                    cooldown_s: cooldown_before,
                },
                &mut capability_done,
            );
            if heavy_row(n, depth) {
                cooldown_if_heavy(n, depth);
                cooldown_before = 60;
            } else {
                cooldown_before = 0;
            }
        }
    }
}

struct Warm {
    artifact: Artifact,
    oracle: f64,
    ours_value: f64,
    ours_bond: usize,
    ours_discarded: f64,
    ours_fidelity: f64,
    ours_calls: usize,
    ours_cols: usize,
    ours_seconds: f64,
    q_value: f64,
    q_bond: usize,
    q_fidelity: f64,
    q_error: f64,
    q_seconds: f64,
    load_before: f64,
    /// §4.4a: the pre-row probe and the census over the row (warm-up to last repeat).
    probe: QuietProbe,
    census: Option<CpuCensus>,
}

fn warmup(quimb: &mut QuimbProc, n: u32, depth: usize, chi: usize, knob: Knob) -> Warm {
    let probe = wait_quiet(quimb.pid());
    let census = CpuCensus::start(&own_pids(quimb.pid())).expect("census");
    let load_before = probe.load1;
    let artifact = load_hea(n, depth).expect("committed HEA");
    let oracle = dense_z0(&artifact.built.circuit).expect("dense");
    let ours = run_ours(&artifact.built.circuit, chi).expect("ours warmup");
    let q = quimb
        .eval(n, chi, knob, &artifact.built.gates)
        .expect("quimb warmup");
    Warm {
        artifact,
        oracle,
        ours_value: ours.value,
        ours_bond: ours.max_bond,
        ours_discarded: ours.discarded_weight,
        ours_fidelity: ours.fidelity_estimate,
        ours_calls: ours.svd_calls,
        ours_cols: ours.max_cols,
        ours_seconds: ours.seconds,
        q_value: q.expectation,
        q_bond: q.max_bond,
        q_fidelity: q.fidelity_estimate,
        q_error: q.error_estimate,
        q_seconds: q.seconds,
        load_before,
        probe,
        census: Some(census),
    }
}

struct Samples {
    ours_min: f64,
    ours_med: f64,
    quimb_min: f64,
    quimb_med: f64,
    repeats: u32,
    load_after: f64,
    cooldown_s: u32,
    external_cpu: Option<ExternalCpu>,
}

fn no_time(load_after: f64, cooldown_s: u32) -> Samples {
    Samples {
        ours_min: 0.0,
        ours_med: 0.0,
        quimb_min: 0.0,
        quimb_med: 0.0,
        repeats: 0,
        load_after,
        cooldown_s,
        external_cpu: None,
    }
}

fn facts_of(warm: &Warm, chi: usize, samples: Samples) -> TimedFacts {
    TimedFacts {
        artifact: warm.artifact.clone(),
        chi,
        oracle: warm.oracle,
        ours_value: warm.ours_value,
        quimb_value: warm.q_value,
        ours_bond: warm.ours_bond,
        quimb_bond: warm.q_bond,
        ours_discarded: warm.ours_discarded,
        ours_fidelity: warm.ours_fidelity,
        quimb_fidelity: warm.q_fidelity,
        quimb_error: warm.q_error,
        ours_calls: warm.ours_calls,
        ours_cols: warm.ours_cols,
        ours_min: samples.ours_min,
        ours_med: samples.ours_med,
        quimb_min: samples.quimb_min,
        quimb_med: samples.quimb_med,
        repeats: samples.repeats,
        ours_warmup_s: warm.ours_seconds,
        quimb_warmup_s: warm.q_seconds,
        load_before: warm.load_before,
        load_after: samples.load_after,
        cooldown_s: samples.cooldown_s,
        external_cpu: samples.external_cpu,
    }
}

struct RowJob {
    n: u32,
    depth: usize,
    chi: usize,
    knob: Knob,
    cooldown_s: u32,
}

fn publish_void(publisher: &mut LanePublisher, facts: &TimedFacts, reason: &str) {
    let id = speed_id(facts);
    publisher
        .write_void(&id, reason, measured_context(facts, "void"))
        .unwrap_or_else(|e| panic!("void row {id}: {e}"));
    println!("JSONL void id={id}");
}

fn run_timed_row(
    quimb: &mut QuimbProc,
    publisher: &mut LanePublisher,
    job: RowJob,
    capability_done: &mut bool,
) {
    let RowJob {
        n,
        depth,
        chi,
        knob,
        cooldown_s,
    } = job;
    if runtime_stamp() != compiled_rev() {
        eprintln!(
            "rev witness refused mid-grid: compiled={} runtime={}",
            compiled_rev(),
            runtime_stamp()
        );
        std::process::exit(2);
    }
    for try_n in 0..2 {
        let mut warm = warmup(quimb, n, depth, chi, knob);
        let our_gap = abs_gap(warm.ours_value, warm.oracle);
        let q_gap = abs_gap(warm.q_value, warm.oracle);
        let our_rel = relative_error(warm.ours_value, warm.oracle);
        let q_rel = relative_error(warm.q_value, warm.oracle);
        println!(
            "PRE n={n} depth={depth} chi={chi} artifact={} load_before={:.2} oracle={:.12e} ours={:.12e} quimb={:.12e} our_gap={our_gap:.3e} quimb_gap={q_gap:.3e} our_bond={} quimb_bond={} our_discarded={:.3e} our_fidelity={:.6e} quimb_fidelity={:.6e} quimb_error={:.6e} our_rel={our_rel:.3e} quimb_rel={q_rel:.3e}",
            warm.artifact.id,
            warm.load_before,
            warm.oracle,
            warm.ours_value,
            warm.q_value,
            warm.ours_bond,
            warm.q_bond,
            warm.ours_discarded,
            warm.ours_fidelity,
            warm.q_fidelity,
            warm.q_error,
        );
        if !*capability_done && n == 14 && depth == 12 && chi == 32 {
            println!(
                "{}",
                format_capability_row(
                    warm.ours_discarded,
                    warm.ours_fidelity,
                    warm.q_fidelity,
                    warm.q_error
                )
            );
            let cap_facts = facts_of(&warm, chi, no_time(warm.load_before, cooldown_s));
            publisher
                .write_capability(&cap_facts)
                .unwrap_or_else(|e| panic!("capability row: {e}"));
            println!("JSONL capability id=mps-capability-discarded-weight");
            println!(
                "blind_spot=does not prove discarded_weight bounds the expectation error; this MPS is non-canonical and the engine does not claim that bound"
            );
            *capability_done = true;
        }
        let load_after_now = load1().expect("uptime");
        if requires_bond_saturation(depth) {
            if let Err(reason) = admit_saturating_row(
                chi,
                warm.q_bond,
                warm.ours_bond,
                warm.ours_calls,
                warm.ours_cols,
            ) {
                println!("ROW kind=refused n={n} depth={depth} chi={chi}");
                println!("reason={reason}");
                println!("blind_spot={}", blind_spot(n, depth, chi));
                println!("load_after={load_after_now:.2}");
                let facts = facts_of(&warm, chi, no_time(load_after_now, cooldown_s));
                publish_void(publisher, &facts, &reason);
                return;
            }
        }
        // The schema's gate is relative 1e-10. A truncating row that misses it
        // is still timed when the two dense gaps are the same work, and
        // published as a void row rather than a speed classification.
        let exact =
            our_rel <= omega_emu_compare::GATE_REL_F64 && q_rel <= omega_emu_compare::GATE_REL_F64;
        if !exact && !truncation_matched(our_gap, q_gap) {
            let reason = format!(
                "dense gaps are not the same work (ours {our_gap:.3e}, quimb {q_gap:.3e}); not timed"
            );
            println!("ROW kind=disagreement n={n} depth={depth} chi={chi}");
            println!("reason={reason}");
            println!("blind_spot={}", blind_spot(n, depth, chi));
            println!("load_after={load_after_now:.2}");
            let facts = facts_of(&warm, chi, no_time(load_after_now, cooldown_s));
            publish_void(publisher, &facts, &reason);
            return;
        }
        let repeats = timed_repeats(warm.ours_seconds.max(warm.q_seconds));
        let mut ours_s = Vec::new();
        let mut quimb_s = Vec::new();
        for i in 0..repeats {
            let o = run_ours(&warm.artifact.built.circuit, chi).expect("ours timed");
            ours_s.push(o.seconds);
            let q = quimb
                .eval(n, chi, knob, &warm.artifact.built.gates)
                .expect("quimb timed");
            quimb_s.push(q.seconds);
            if requires_bond_saturation(depth) && (q.max_bond != chi || o.max_bond != chi) {
                let reason = format!(
                    "bond left the cap during timed repeat {i}: ours {} quimb {}",
                    o.max_bond, q.max_bond
                );
                println!("ROW kind=refused n={n} depth={depth} chi={chi}");
                println!("reason={reason}");
                let facts = facts_of(&warm, chi, no_time(load1().expect("uptime"), cooldown_s));
                publish_void(publisher, &facts, &reason);
                return;
            }
            eprintln!(
                "sample {i} n={n} depth={depth} chi={chi} ours_s={:.6} quimb_s={:.6}",
                o.seconds, q.seconds
            );
        }
        let load_after = load1().expect("uptime");
        let external = warm
            .census
            .take()
            .expect("census started in warmup")
            .finish()
            .expect("census")
            .with_pre_row(&warm.probe);
        let ext_before = external.external_cores_before.unwrap_or(f64::NAN);
        let ext_during = external.external_cores_during_row;
        if row_void_at_load(warm.probe.cores, void_above())
            || row_void_at_load(ext_during, void_above())
        {
            eprintln!(
                "void: outside the lane {ext_before:.2} cores before, {ext_during:.2} during (load1 {:.2} → {load_after:.2}) try={try_n}",
                warm.load_before
            );
            if try_n == 0 {
                std::thread::sleep(Duration::from_secs(60));
                continue;
            }
            let reason = format!(
                "work outside the lane above {} cores (before {ext_before:.2}, during {ext_during:.2}; load1 {:.2} → {load_after:.2})",
                void_above(),
                warm.load_before
            );
            println!("ROW kind=void n={n} depth={depth} chi={chi}");
            println!("reason={reason}");
            let (omin, omed) = min_median(&ours_s).unwrap_or((0.0, 0.0));
            let (qmin, qmed) = min_median(&quimb_s).unwrap_or((0.0, 0.0));
            let facts = facts_of(
                &warm,
                chi,
                Samples {
                    ours_min: omin,
                    ours_med: omed,
                    quimb_min: qmin,
                    quimb_med: qmed,
                    repeats: repeats as u32,
                    load_after,
                    cooldown_s,
                    external_cpu: Some(external.clone()),
                },
            );
            publish_void(publisher, &facts, &reason);
            return;
        }
        let (omin, omed) = min_median(&ours_s).expect("samples");
        let (qmin, qmed) = min_median(&quimb_s).expect("samples");
        let (class, ratio) = classify_speed(omin, omed, qmin, qmed);
        let headline = speed_headline(depth, class);
        let facts = facts_of(
            &warm,
            chi,
            Samples {
                ours_min: omin,
                ours_med: omed,
                quimb_min: qmin,
                quimb_med: qmed,
                repeats: repeats as u32,
                load_after,
                cooldown_s,
                external_cpu: Some(external),
            },
        );
        if exact {
            publisher
                .write_speed(&facts)
                .unwrap_or_else(|e| panic!("speed row: {e}"));
            println!(
                "ROW kind=speed-exact class={headline} n={n} depth={depth} chi={chi} N={repeats} ours_min={omin:.6} ours_med={omed:.6} quimb_min={qmin:.6} quimb_med={qmed:.6} ratio_quimb_over_ours={ratio:.4} load_before={:.2} load_after={load_after:.2}",
                warm.load_before
            );
            println!("JSONL speed id={}", speed_id(&facts));
        } else {
            let reason = format!(
                "dense relative gaps {our_rel:.3e}/{q_rel:.3e} exceed 1e-10, so this is not a speed row; the times below are the matched-truncation measurement and are not a classification"
            );
            publish_void(publisher, &facts, &reason);
            println!(
                "ROW kind=void-truncating class=not-a-speed-classification directional={headline} n={n} depth={depth} chi={chi} N={repeats} ours_min={omin:.6} ours_med={omed:.6} quimb_min={qmin:.6} quimb_med={qmed:.6} ratio_quimb_over_ours={ratio:.4} load_before={:.2} load_after={load_after:.2}",
                warm.load_before
            );
        }
        println!(
            "GAPS our_discarded={:.3e} our_fidelity={:.6e} quimb_fidelity={:.6e} quimb_error={:.6e} dense_gap_ours={our_gap:.3e} dense_gap_quimb={q_gap:.3e}",
            warm.ours_discarded, warm.ours_fidelity, warm.q_fidelity, warm.q_error
        );
        println!("blind_spot={}", blind_spot(n, depth, chi));
        return;
    }
}
