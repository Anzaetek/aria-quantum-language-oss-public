// SPDX-License-Identifier: Apache-2.0
//! Every GPU arm is a child process speaking JSON lines (the protocol of
//! `tools/emu_compare/dense/dense_common.py`). Ours is the sibling binary
//! `emu-compare-gpu-ours`; the competitors are Python programs, each in the
//! venv its pins require. One process per (engine, variant), started after the
//! row's "before" device census and gone before its "after" one.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Instant;

use omega_emu_compare::PathWitness;
use serde_json::{json, Value};

/// A competitor's fusion and precision setting, pinned for its process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Variant {
    /// One kernel application per gate, f32.
    Unfused,
    /// The engine's fusion, f32.
    Fused,
    /// Fusion at f64: a control, never admitted as a row's competitor.
    FusedF64,
}

impl Variant {
    pub fn as_str(self) -> &'static str {
        match self {
            Variant::Unfused => "unfused",
            Variant::Fused => "fused",
            Variant::FusedF64 => "fused-f64",
        }
    }
    pub fn is_f64(self) -> bool {
        self == Variant::FusedF64
    }
}

/// Which program, and its pinned knobs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Engine {
    /// `omega-backend-statevector-cuda`, f32.
    Ours,
    /// cuStateVec 1.15 through cuquantum-python, written from NVIDIA's samples.
    Custatevec(Variant),
    /// qiskit-aer-gpu 0.15.1 with cuStateVec: the sanity floor.
    AerGpu(Variant),
    /// Aer CPU, double: the oracle. Never timed, never on the card.
    Oracle { threads: u32 },
}

impl Engine {
    pub fn label(&self) -> String {
        match self {
            Engine::Ours => "omega-cuda-f32".into(),
            Engine::Custatevec(v) => format!("custatevec[{}]", v.as_str()),
            Engine::AerGpu(v) => format!("aer-gpu-0.15.1[{}]", v.as_str()),
            Engine::Oracle { threads } => format!("aer-cpu-f64[t={threads}]"),
        }
    }

    /// Host threads the process is pinned to. GPU arms drive the card from
    /// one host thread; the number is asserted back at `hello`.
    pub fn threads(&self) -> u32 {
        match self {
            Engine::Oracle { threads } => *threads,
            _ => 1,
        }
    }

    pub fn is_f64(&self) -> bool {
        match self {
            Engine::Custatevec(v) | Engine::AerGpu(v) => v.is_f64(),
            Engine::Ours => false,
            Engine::Oracle { .. } => true,
        }
    }

    pub fn knobs(&self) -> BTreeMap<String, String> {
        let mut k = BTreeMap::new();
        let mut put = |a: &str, b: &str| {
            k.insert(a.to_string(), b.to_string());
        };
        match self {
            Engine::Ours => {
                put(
                    "api",
                    "Backend::expectation on CudaStatevectorBackend (library path, not the CLI)",
                );
                put(
                    "precision",
                    "f32 state (CudaSlice<f32>), f64 reduction of the Pauli sum",
                );
                put(
                    "fusion",
                    "the backend's own apply_ops_fused, inside the timed call",
                );
                put("state_alloc", "inside the timed call");
            }
            Engine::Custatevec(v) => {
                put("variant", v.as_str());
                put(
                    "precision",
                    if v.is_f64() {
                        "CUDA_C_64F, COMPUTE_64F"
                    } else {
                        "CUDA_C_32F, COMPUTE_32F"
                    },
                );
                put("matrices", "device-resident, uploaded before the clock");
                put("workspace", "memory_handler.py pattern: default pool, release threshold raised, mallocAsync on a dedicated stream");
                put("state_alloc", "at load, outside the timed call (ours allocates inside: in cuStateVec's favour)");
                put(
                    "fusion",
                    match v {
                        Variant::Unfused => "none: one apply_matrix per gate (gate_application.py)",
                        _ => "host-side, untimed, greedy into <=2-qubit dense blocks",
                    },
                );
                put("timed_call", "initialize_state_vector + apply_matrix per block + compute_expectations_on_pauli_basis, synchronised");
            }
            Engine::AerGpu(v) => {
                put("variant", v.as_str());
                put("precision", if v.is_f64() { "double" } else { "single" });
                put("cuStateVec_enable", "True");
                put(
                    "fusion_enable",
                    if *v == Variant::Unfused {
                        "False"
                    } else {
                        "True (Aer default)"
                    },
                );
                put(
                    "timed_call",
                    "AerSimulator.run(...).result() on a circuit transpiled before the clock",
                );
                put(
                    "warm_up",
                    "load-bearing: the first call JITs sm_90 PTX for sm_120",
                );
            }
            Engine::Oracle { threads } => {
                put("max_parallel_threads", &threads.to_string());
                put("precision", "double");
            }
        }
        k
    }

    fn command(&self) -> Result<Command, String> {
        let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dense = here.join("../dense");
        let py = |var: &str, what: &str| -> Result<PathBuf, String> {
            let p = std::env::var(var)
                .map_err(|_| format!("{var} is unset: point it at the venv python for {what}"))?;
            let p = PathBuf::from(p);
            if !p.is_file() {
                return Err(format!("{var} {} is not a file", p.display()));
            }
            Ok(p)
        };
        let mut cmd = match self {
            Engine::Ours => {
                let exe = std::env::current_exe().map_err(|e| e.to_string())?;
                let ours = exe.with_file_name("emu-compare-gpu-ours");
                if !ours.is_file() {
                    return Err(format!(
                        "{} is missing: build with `--features cuda` (run.sh does)",
                        ours.display()
                    ));
                }
                Command::new(ours)
            }
            Engine::Custatevec(_) => {
                let mut c = Command::new(py("EMU_GPU_CUSV_PY", "cuquantum-python 26.9.0")?);
                c.arg(here.join("custatevec_arm.py"));
                c
            }
            Engine::AerGpu(_) => {
                let mut c = Command::new(py(
                    "EMU_GPU_AER_PY",
                    "qiskit 1.4.6 + qiskit-aer-gpu 0.15.1",
                )?);
                c.arg(here.join("aer_gpu_arm.py"));
                c
            }
            Engine::Oracle { .. } => {
                let mut c =
                    Command::new(py("EMU_GPU_ORACLE_PY", "qiskit 2.5.2 + qiskit-aer 0.17.2")?);
                c.arg(dense.join("aer_oracle.py"));
                c
            }
        };
        let pythonpath =
            std::env::join_paths([here.clone(), dense.clone()]).map_err(|e| e.to_string())?;
        cmd.current_dir(&here)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONWARNINGS", "ignore")
            .env("PYTHONPATH", pythonpath)
            .env("EMU_DENSE_THREADS", self.threads().to_string());
        if let Engine::Custatevec(v) | Engine::AerGpu(v) = self {
            cmd.env("EMU_GPU_VARIANT", v.as_str());
        }
        Ok(cmd)
    }
}

/// A live arm.
pub struct GpuArm {
    pub engine: Engine,
    child: Child,
    stdin: std::io::BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    pub hello: Value,
    /// Spawn to the end of `hello`: interpreter or binary start, imports, CUDA
    /// context. Part of the arm's fixed cost, reported separately (§7 E5).
    pub startup_s: f64,
    quit: bool,
}

impl GpuArm {
    pub fn launch(engine: Engine) -> Result<Self, String> {
        let t = Instant::now();
        let mut cmd = engine.command()?;
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
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
            startup_s: 0.0,
            quit: false,
        };
        let hello = arm.call(json!({"op": "hello"}))?;
        let threads = hello.get("threads").and_then(Value::as_u64);
        if threads != Some(u64::from(engine.threads())) {
            return Err(format!(
                "{} reports threads {threads:?}, launched with {}",
                engine.label(),
                engine.threads()
            ));
        }
        arm.hello = hello;
        arm.startup_s = t.elapsed().as_secs_f64();
        Ok(arm)
    }

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
                "{} closed stdout (it refused or crashed; see its stderr)",
                self.engine.label()
            ));
        }
        let v: Value = serde_json::from_str(&line)
            .map_err(|e| format!("{} JSON: {e} in {line}", self.engine.label()))?;
        if let Some(err) = v.get("error") {
            return Err(format!("{} {}: {err}", self.engine.label(), req["op"]));
        }
        Ok(v)
    }

    pub fn versions(&self) -> Result<BTreeMap<String, String>, String> {
        let obj = self
            .hello
            .get("versions")
            .and_then(Value::as_object)
            .ok_or("hello has no versions")?;
        Ok(obj
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| v.to_string()),
                )
            })
            .collect())
    }

    /// Witnesses the arm asserted at `hello`; a printed-only one is refused (§4.1).
    pub fn witnesses(&self) -> Result<Vec<PathWitness>, String> {
        let list = self
            .hello
            .get("witnesses")
            .and_then(Value::as_array)
            .ok_or("hello has no witnesses")?;
        list.iter()
            .map(|w| {
                let name = w["name"].as_str().ok_or("witness without name")?;
                let observed = w["observed"].as_str().ok_or("witness without observed")?;
                if w["asserted"].as_bool() != Some(true) {
                    return Err(format!(
                        "{} witness {name} was not asserted (§4.1)",
                        self.engine.label()
                    ));
                }
                Ok(PathWitness {
                    name: name.into(),
                    observed: observed.into(),
                    asserted: true,
                })
            })
            .collect()
    }

    pub fn load(&mut self, name: &str, path: &std::path::Path) -> Result<f64, String> {
        let v = self.call(json!({"op": "load", "name": name, "path": path}))?;
        Ok(v.get("build_s").and_then(Value::as_f64).unwrap_or(0.0))
    }

    /// One timed expectation: `(value, seconds)`.
    pub fn expectation(&mut self, name: &str, terms: &Value) -> Result<(f64, f64), String> {
        let v = self.call(json!({"op": "expectation", "name": name, "terms": terms}))?;
        let val = v
            .get("value")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("no value: {v}"))?;
        let s = v
            .get("seconds")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("no seconds: {v}"))?;
        if !(val.is_finite() && s.is_finite() && s > 0.0) {
            return Err(format!(
                "{} returned value {val}, seconds {s}",
                self.engine.label()
            ));
        }
        Ok((val, s))
    }

    pub fn floor(&mut self) -> Result<f64, String> {
        let v = self.call(json!({"op": "floor"}))?;
        v.get("seconds")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("no seconds: {v}"))
    }

    pub fn write_state(&mut self, name: &str, out: &std::path::Path) -> Result<(), String> {
        self.call(json!({"op": "state", "name": name, "out": out}))
            .map(|_| ())
    }

    /// Quit and wait, so the card is released before the "after" census.
    pub fn quit(mut self) -> Result<(), String> {
        self.quit = true;
        let _ = writeln!(self.stdin, r#"{{"op":"quit"}}"#);
        let _ = self.stdin.flush();
        let st = self.child.wait().map_err(|e| e.to_string())?;
        if !st.success() {
            return Err(format!("{} exited {st}", self.engine.label()));
        }
        Ok(())
    }
}

impl Drop for GpuArm {
    fn drop(&mut self) {
        if !self.quit {
            let _ = writeln!(self.stdin, r#"{{"op":"quit"}}"#);
            let _ = self.stdin.flush();
            let _ = self.child.wait();
        }
    }
}
