// SPDX-License-Identifier: Apache-2.0
//! One Python process per (engine, knob setting). Imports and circuit
//! construction happen in the process before any timed call; the seconds in
//! a response are `perf_counter` around exactly the call the row names.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use omega_emu_compare::PathWitness;
use serde_json::{json, Value};

/// Which Python program, and the knobs pinned for its whole life.
#[derive(Clone, Debug, PartialEq)]
pub enum Engine {
    /// qulacs, OpenMP threads.
    Qulacs { threads: u32 },
    /// qsim, `cpu_threads` and `max_fused_gate_size`.
    Qsim { threads: u32, fuse: u32 },
    /// The Aer CPU double oracle; never timed.
    AerOracle { threads: u32 },
}

impl Engine {
    fn script(&self) -> &'static str {
        match self {
            Engine::Qulacs { .. } => "qulacs_arm.py",
            Engine::Qsim { .. } => "qsim_arm.py",
            Engine::AerOracle { .. } => "aer_oracle.py",
        }
    }

    /// Threads this process was pinned to.
    pub fn threads(&self) -> u32 {
        match self {
            Engine::Qulacs { threads }
            | Engine::Qsim { threads, .. }
            | Engine::AerOracle { threads } => *threads,
        }
    }

    /// Knobs printed into the row.
    pub fn knobs(&self) -> BTreeMap<String, String> {
        let mut k = BTreeMap::new();
        match self {
            Engine::Qulacs { threads } => {
                k.insert("OMP_NUM_THREADS".into(), threads.to_string());
                k.insert(
                    "timed_call".into(),
                    "QuantumState(n) + update_quantum_state + Observable.get_expectation_value"
                        .into(),
                );
            }
            Engine::Qsim { threads, fuse } => {
                k.insert("cpu_threads".into(), threads.to_string());
                k.insert("max_fused_gate_size".into(), fuse.to_string());
                k.insert(
                    "timed_call".into(),
                    "qsim_simulate_expectation_values kernel on a pre-translated circuit".into(),
                );
            }
            Engine::AerOracle { threads } => {
                k.insert("max_parallel_threads".into(), threads.to_string());
                k.insert("precision".into(), "double".into());
            }
        }
        k
    }

    /// Short label for logs and sweep tables.
    pub fn label(&self) -> String {
        match self {
            Engine::Qulacs { threads } => format!("qulacs[t={threads}]"),
            Engine::Qsim { threads, fuse } => format!("qsim[t={threads},f={fuse}]"),
            Engine::AerOracle { threads } => format!("aer[t={threads}]"),
        }
    }
}

/// A live Python arm.
pub struct PyArm {
    pub engine: Engine,
    child: Child,
    stdin: std::io::BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    pub hello: Value,
}

impl PyArm {
    /// Start the process and run its `hello`, which asserts the arm's own
    /// preconditions (version pins, device, SIMD path) and exits non-zero if
    /// any fails.
    pub fn launch(engine: Engine) -> Result<Self, String> {
        let python = python_path()?;
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut cmd = Command::new(&python);
        cmd.arg(dir.join(engine.script()))
            .current_dir(&dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONWARNINGS", "ignore")
            .env("PYTHONPATH", &dir)
            .env("EMU_DENSE_THREADS", engine.threads().to_string());
        if let Engine::Qsim { fuse, .. } = engine {
            cmd.env("EMU_DENSE_QSIM_FUSE", fuse.to_string());
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", engine.label()))?;
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));
        let stdin = std::io::BufWriter::new(child.stdin.take().expect("stdin piped"));
        let mut arm = Self {
            engine,
            child,
            stdin,
            stdout,
            hello: Value::Null,
        };
        let hello = arm.call(json!({"op": "hello"}))?;
        let threads = hello.get("threads").and_then(Value::as_u64);
        if threads != Some(u64::from(arm.engine.threads())) {
            return Err(format!(
                "{} reports threads {threads:?}, launched with {}",
                arm.engine.label(),
                arm.engine.threads()
            ));
        }
        arm.hello = hello;
        Ok(arm)
    }

    /// The arm process's pid, counted as the lane's own tree by §4.4a's census.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn call(&mut self, req: Value) -> Result<Value, String> {
        writeln!(self.stdin, "{req}").map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        let n = self
            .stdout
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            return Err(format!(
                "{} closed stdout (it refused or crashed; see its stderr above)",
                self.engine.label()
            ));
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|e| format!("{} JSON: {e} in {line}", self.engine.label()))?;
        if let Some(err) = value.get("error") {
            return Err(format!("{} {}: {err}", self.engine.label(), req["op"]));
        }
        Ok(value)
    }

    /// Versions the arm reported at `hello`.
    pub fn versions(&self) -> Result<BTreeMap<String, String>, String> {
        let obj = self
            .hello
            .get("versions")
            .and_then(Value::as_object)
            .ok_or("hello has no versions")?;
        Ok(obj
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
            .collect())
    }

    /// The executed-path witnesses the arm asserted at `hello`. Each one was
    /// an assert in the arm (it exits otherwise), so `asserted` is carried
    /// through, and an arm that reports a printed-only witness is refused.
    pub fn witnesses(&self) -> Result<Vec<PathWitness>, String> {
        let list = self
            .hello
            .get("witnesses")
            .and_then(Value::as_array)
            .ok_or("hello has no witnesses")?;
        let mut out = Vec::new();
        for w in list {
            let name = w["name"].as_str().ok_or("witness without name")?;
            let observed = w["observed"].as_str().ok_or("witness without observed")?;
            if w["asserted"].as_bool() != Some(true) {
                return Err(format!(
                    "{} witness {name} was not asserted (§4.1)",
                    self.engine.label()
                ));
            }
            out.push(PathWitness {
                name: name.to_string(),
                observed: observed.to_string(),
                asserted: true,
            });
        }
        Ok(out)
    }

    pub fn load(&mut self, name: &str, path: &std::path::Path) -> Result<f64, String> {
        let v = self.call(json!({"op": "load", "name": name, "path": path}))?;
        Ok(v.get("build_s").and_then(Value::as_f64).unwrap_or(0.0))
    }

    /// One timed expectation: `(value, seconds)`.
    pub fn expectation(&mut self, name: &str, terms: &Value) -> Result<(f64, f64), String> {
        let v = self.call(json!({"op": "expectation", "name": name, "terms": terms}))?;
        pair(&v, "value", "seconds")
    }

    pub fn floor(&mut self) -> Result<f64, String> {
        let v = self.call(json!({"op": "floor"}))?;
        seconds(&v)
    }

    pub fn floor_sample(&mut self) -> Result<f64, String> {
        let v = self.call(json!({"op": "floor_sample"}))?;
        seconds(&v)
    }

    pub fn write_state(&mut self, name: &str, out: &std::path::Path) -> Result<(), String> {
        self.call(json!({"op": "state", "name": name, "out": out}))
            .map(|_| ())
    }
}

impl Drop for PyArm {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, r#"{{"op":"quit"}}"#);
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

pub fn seconds(v: &Value) -> Result<f64, String> {
    v.get("seconds")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("response has no seconds: {v}"))
}

pub fn pair(v: &Value, a: &str, b: &str) -> Result<(f64, f64), String> {
    let x = v
        .get(a)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("response has no {a}: {v}"))?;
    let y = v
        .get(b)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("response has no {b}: {v}"))?;
    Ok((x, y))
}

/// The lane venv's interpreter, from `EMU_DENSE_PY`. No default path: a lane
/// that silently picked some other venv would publish versions it did not
/// pin (§6 requires `requirements-compare-dense.txt`).
pub fn python_path() -> Result<PathBuf, String> {
    let p = std::env::var("EMU_DENSE_PY").map_err(|_| {
        "EMU_DENSE_PY is unset: point it at the venv built from \
         tools/emu_compare/dense/requirements-compare-dense.txt"
            .to_string()
    })?;
    let p = PathBuf::from(p);
    if !p.is_file() {
        return Err(format!("EMU_DENSE_PY {} is not a file", p.display()));
    }
    Ok(p)
}
