// SPDX-License-Identifier: Apache-2.0
//! The CV row of the emulator comparison (§3, E7): one capability/correctness
//! row, `omega-backend-cv` against piquasso 8.0.1 on a two-mode squeezing +
//! interferometer + photon-number case from the committed fixture.
//!
//! Not a speed row. piquasso's hot path is NumPy plus numba, not a tuned CV
//! simulator, and at two modes and cutoff 14 both arms' times are measured
//! against their own per-call floors and printed. The row refuses a speed
//! classification and says why in its own numbers (§4.7, as E6 did for its
//! ≤ 8-mode fermionic cells).

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use omega_backend_cv::multimode::MultiFockState;
use omega_emu_compare::{
    Arm, CapabilityRow, CpuCensus, Direction, GitRev, Lane, PathWitness, PathWitnesses, Precision,
    Quantity, Row, RowBody, RowWriter, Threads, Witness, Workload,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const CASE: &str = "tms_phased_on_photons";
const FIXTURE: &str = "tools/cv_cross_check/piquasso_multimode_fixture.jsonl";
const BUDGET: Option<usize> = Some(256 << 20);
const GATE: f64 = 1e-10;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

fn sh(cmd: &str, args: &[&str]) -> Result<String, String> {
    let o = Command::new(cmd)
        .args(args)
        .current_dir(root())
        .output()
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&o.stdout).trim().to_string())
}

struct Py {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    out: BufReader<std::process::ChildStdout>,
}

impl Py {
    fn call(&mut self, req: Value) -> Result<Value, String> {
        writeln!(self.stdin, "{req}").map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;
        let mut l = String::new();
        self.out.read_line(&mut l).map_err(|e| e.to_string())?;
        let v: Value = serde_json::from_str(&l).map_err(|e| format!("{e}: {l}"))?;
        if let Some(e) = v.get("error") {
            return Err(format!("piquasso: {e}"));
        }
        Ok(v)
    }
}

fn ours(case: &Value) -> Result<(Vec<f64>, f64), String> {
    let n = case["n_modes"].as_u64().ok_or("n_modes")? as usize;
    let cutoff = case["cutoff"].as_u64().ok_or("cutoff")? as usize;
    let prep: Vec<usize> = case["prep"]
        .as_array()
        .ok_or("prep")?
        .iter()
        .map(|x| x.as_u64().unwrap_or(0) as usize)
        .collect();
    let t = Instant::now();
    let mut st =
        MultiFockState::number_state(cutoff, n, &prep, BUDGET).map_err(|e| format!("{e:?}"))?;
    for op in case["ops"].as_array().ok_or("ops")? {
        let f = |k: &str| op[k].as_f64().unwrap_or(f64::NAN);
        let u = |k: &str| op[k].as_u64().unwrap_or(0) as usize;
        match op["op"].as_str().unwrap_or("") {
            "beamsplitter" => st
                .beamsplitter(u("a"), u("b"), f("theta"), f("phi"))
                .map_err(|e| format!("{e:?}"))?,
            "squeezing2" => st
                .squeezing2(u("a"), u("b"), f("r"), f("phi"))
                .map_err(|e| format!("{e:?}"))?,
            other => return Err(format!("op {other} not in this row's case")),
        }
    }
    let m = (0..n)
        .map(|i| st.expect_n(i).map_err(|e| format!("{e:?}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((m, t.elapsed().as_secs_f64()))
}

fn ours_floor() -> Result<f64, String> {
    let t = Instant::now();
    let st = MultiFockState::number_state(2, 1, &[0], BUDGET).map_err(|e| format!("{e:?}"))?;
    let v = st.expect_n(0).map_err(|e| format!("{e:?}"))?;
    let dt = t.elapsed().as_secs_f64();
    if v.abs() > 1e-12 {
        return Err(format!("floor <n> of vacuum is {v}"));
    }
    Ok(dt)
}

fn min_med(t: &[f64]) -> (f64, f64) {
    let mut s = t.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[0], s[s.len() / 2])
}

fn run(out: &Path) -> Result<(), String> {
    let compiled = option_env!("OMEGA_EMU_GIT_REV").unwrap_or("").to_string();
    let git = GitRev {
        compiled_from: compiled,
        worktree: sh("git", &["rev-parse", "HEAD"])?,
        worktree_dirty: !sh("git", &["status", "--porcelain"])?.is_empty(),
    };
    if let Some(why) = git.void_reason() {
        return Err(why);
    }
    let bytes = std::fs::read(root().join(FIXTURE)).map_err(|e| e.to_string())?;
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let case: Value = String::from_utf8(bytes)
        .map_err(|e| e.to_string())?
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|v| v["case"] == CASE)
        .ok_or("case missing from fixture")?;
    let fixture_n: Vec<f64> = case["mean_n"]
        .as_array()
        .ok_or("mean_n")?
        .iter()
        .filter_map(Value::as_f64)
        .collect();

    let py = std::env::var("EMU_CV_PY").map_err(|_| "EMU_CV_PY unset".to_string())?;
    // piquasso 8.0.1's numba on-disk cache does not reload on akilles (numba
    // 0.68.0): the first process compiles and writes it, every later process
    // aborts with "LLVM ERROR: Symbol not found: .numba.unresolved$...hafnian".
    // So each run gets an empty cache directory and compiles in its untimed
    // warm-up, which the row reports as piquasso's first call.
    let numba_cache = std::env::temp_dir().join(format!("emu-cv-numba-{}", std::process::id()));
    std::fs::create_dir_all(&numba_cache).map_err(|e| e.to_string())?;
    let mut child = Command::new(py)
        .env("NUMBA_CACHE_DIR", &numba_cache)
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("cv_arm.py"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;
    let stdin = child.stdin.take().unwrap();
    let outp = BufReader::new(child.stdout.take().unwrap());
    let mut pyc = Py {
        child,
        stdin,
        out: outp,
    };
    let hello = pyc.call(json!({"op": "hello"}))?;
    let req = json!({"op": "run", "cutoff": case["cutoff"], "n_modes": case["n_modes"], "prep": case["prep"], "ops": case["ops"]});

    let census = CpuCensus::start(&[std::process::id(), pyc.child.id()])?;
    let (_, ours_first) = ours(&case)?;
    let pf = pyc.call(req.clone())?;
    let pq_first = pf["seconds"].as_f64().unwrap_or(f64::NAN);
    let (mut a, mut b, mut ov, mut pv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..5 {
        let (m, t) = ours(&case)?;
        a.push(t);
        ov = m;
        let r = pyc.call(req.clone())?;
        b.push(r["seconds"].as_f64().ok_or("seconds")?);
        pv = r["mean_n"]
            .as_array()
            .ok_or("mean_n")?
            .iter()
            .filter_map(Value::as_f64)
            .collect();
    }
    let ext = census.finish()?;
    let of = (0..5)
        .map(|_| ours_floor())
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let mut pf_min = f64::INFINITY;
    for _ in 0..5 {
        pf_min = pf_min.min(
            pyc.call(json!({"op": "floor"}))?["seconds"]
                .as_f64()
                .unwrap_or(f64::INFINITY),
        );
    }
    let _ = writeln!(pyc.stdin, r#"{{"op":"quit"}}"#);

    let gap = |x: &[f64], y: &[f64]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max)
    };
    let g_live = gap(&ov, &pv);
    let g_fix = gap(&ov, &fixture_n);
    if g_live > GATE || g_fix > GATE {
        return Err(format!(
            "value gate failed: ours {ov:?} piquasso {pv:?} fixture {fixture_n:?}"
        ));
    }
    let (om, omed) = min_med(&a);
    let (pm, pmed) = min_med(&b);
    let evidence = format!(
        "<n_0>,<n_1>: ours {:.15},{:.15}; piquasso live {:.15},{:.15}; fixture {:.15},{:.15}; max |Δ| ours-vs-piquasso {g_live:.3e}, ours-vs-fixture {g_fix:.3e} (≤ {GATE:e}). \
         Timings (min/median of 5, interleaved, one warm-up each): ours {om:.3e}/{omed:.3e} s, piquasso {pm:.3e}/{pmed:.3e} s; \
         per-call floors (min of 5): ours {of:.3e} s, piquasso {pf_min:.3e} s; first calls ours {ours_first:.3e} s, piquasso {pq_first:.3e} s (numba JIT). \
         External CPU during the measurement {:.2} cores over {:.2} s.",
        ov[0], ov[1], pv[0], pv[1], fixture_n[0], fixture_n[1], ext.external_cores_during_row, ext.window_s
    );
    let spread = pmed / pm;
    let spread_clause = if spread > 5.0 {
        format!("its timings are unstable, median/min {spread:.1}x across the five repeats (§4.3a flags above 5x)")
    } else {
        format!("median/min across the five repeats was {spread:.1}x, under §4.3a's 5x flag")
    };
    let refusal = format!(
        "speed classification refused. (1) piquasso's hot path is NumPy plus numba, not a tuned CV simulator, so a ratio \
         would measure our Rust against NumPy (§2). (2) Its first call is a {pq_first:.2} s numba compile; {spread_clause}. \
         (3) Two modes at cutoff 14 says nothing about CV simulation at scale. It is NOT overhead-dominated by §4.7: \
         piquasso's min is {:.1}x its own per-call floor and ours {:.1}x ours. The ratio of mins, {:.3}, is printed for \
         completeness and is not a result.",
        pm / pf_min,
        om / of,
        pm / om
    );
    let w = |n: &str, o: String| PathWitness {
        name: n.into(),
        observed: o,
        asserted: true,
    };
    let row = Row {
        row_id: "cv-tms-phased-on-photons-mean-n".into(),
        lane: Lane::Cv,
        phase: "E7".into(),
        recorded: sh("date", &["+%Y-%m-%d"])?,
        box_name: Witness::present(sh("hostname", &[])?),
        git: Witness::present(git),
        circuit: Workload {
            id: CASE.into(),
            file: FIXTURE.into(),
            sha256: Witness::present(sha),
            generator: "piquasso_multimode_ref.py case tms_phased_on_photons".into(),
            seed: None,
            qubits: 2,
            depth: 2,
            chi: None,
            shots: None,
            quantity: Quantity::Expectation,
            observable: Some("<n_0>, <n_1> (per-mode mean photon number)".into()),
        },
        precision: Precision {
            ours: "f64".into(),
            competitor: "f64".into(),
            matched: true,
            note: Some("both complex128 Fock amplitudes; piquasso truncates by total photon number, ours per mode, and the case keeps total photon number under the cutoff, so the bases agree on its support".into()),
        },
        body: RowBody::Capability(Box::new(CapabilityRow {
            direction: Direction::OursOnly,
            claim: "Correctness on a two-mode squeezing + interferometer state, not a speed row: <n_i> agrees with piquasso live and with the committed fixture to ~1e-15.".into(),
            evidence: Witness::present(evidence),
            ours: Arm {
                name: "omega-backend-cv".into(),
                versions: Witness::present(BTreeMap::from([("omega-backend-cv".to_string(), env!("CARGO_PKG_VERSION").to_string())])),
                threads: Witness::present(Threads { threads: 1, blas: "none".into() }),
                knobs: BTreeMap::from([("api".to_string(), "MultiFockState::number_state + squeezing2 + beamsplitter + expect_n".to_string())]),
                executed_path: PathWitnesses::new(vec![w("value_gate_vs_piquasso_and_fixture", format!("{:.3e}", g_live.max(g_fix)))]),
            },
            competitor: Arm {
                name: "piquasso-8.0.1".into(),
                versions: Witness::present(hello["versions"].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").into())).collect()).unwrap_or_default()),
                threads: Witness::present(Threads { threads: 1, blas: "numpy default".into() }),
                knobs: BTreeMap::from([("simulator".to_string(), "PureFockSimulator(d=2, cutoff=14)".to_string())]),
                executed_path: PathWitnesses::new(vec![
                    w("piquasso_version", hello["versions"]["piquasso"].as_str().unwrap_or("").into()),
                    w("simulator_class", hello["simulator"].as_str().unwrap_or("").into()),
                ]),
            },
            refusal: Witness::present(refusal),
        })),
        notes: vec![
            format!("the census of work outside the lane's tree read {:.2} cores (§4.4a)", ext.external_cores_during_row),
            "piquasso ran with a fresh NUMBA_CACHE_DIR: its numba on-disk cache does not reload on this box (numba 0.68.0), so it compiles once per process, inside the untimed warm-up".into(),
        ],
    };
    let problems = omega_emu_compare::check(&row);
    if !problems.is_empty() {
        return Err(format!("check(): {problems:?}"));
    }
    let mut writer = RowWriter::append(out).map_err(|e| e.to_string())?;
    writer.write_row(&row).map_err(|e| e.to_string())?;
    eprintln!("CAP cv-tms-phased-on-photons-mean-n\n  {}", row_text(&row));
    Ok(())
}

fn row_text(r: &Row) -> String {
    match &r.body {
        RowBody::Capability(c) => format!(
            "{}\n  {}",
            c.evidence.get().cloned().unwrap_or_default(),
            c.refusal.get().cloned().unwrap_or_default()
        ),
        _ => String::new(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("usage: emu-compare-cv --out PATH");
            std::process::exit(2)
        });
    if let Err(e) = run(Path::new(&out)) {
        eprintln!("emu-compare-cv: {e}");
        std::process::exit(1);
    }
}
