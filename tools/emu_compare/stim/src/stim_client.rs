// SPDX-License-Identifier: Apache-2.0
//! One long-lived Stim process. Import, and the ready-handshake peek,
//! happen before any row is timed.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};

use serde_json::json;

pub struct StimClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    /// From the ready handshake, asserted there.
    pub version: String,
    pub python: String,
    pub numpy: String,
    pub linkage: String,
    pub smoke_x: i64,
    pub smoke_z: i64,
}

pub struct Loaded {
    pub parse_s: f64,
    pub num_qubits: u32,
    pub num_measurements: u32,
}

pub struct Sampled {
    pub seconds: f64,
    pub shots_observed: u64,
    pub num_measurements: u32,
}

impl StimClient {
    pub fn launch() -> Result<Self, String> {
        let python = python_path()?;
        let script = script_path();
        let mut child = Command::new(&python)
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("PYTHONUNBUFFERED", "1")
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", python.display()))?;
        let stdin = child.stdin.take().ok_or("stim arm has no stdin")?;
        let stdout = child.stdout.take().ok_or("stim arm has no stdout")?;
        let mut client = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            version: String::new(),
            python: String::new(),
            numpy: String::new(),
            linkage: String::new(),
            smoke_x: 0,
            smoke_z: 0,
        };
        let ready = client.call(json!({"cmd": "ready"}))?;
        client.version = ready["stim"].as_str().unwrap_or("").to_string();
        client.python = ready["python"].as_str().unwrap_or("").to_string();
        client.numpy = ready["numpy"].as_str().unwrap_or("").to_string();
        client.linkage = ready["linkage"].as_str().unwrap_or("").to_string();
        client.smoke_x = ready["smoke_x"].as_i64().ok_or("ready missing smoke_x")?;
        client.smoke_z = ready["smoke_z"].as_i64().ok_or("ready missing smoke_z")?;
        if client.version != "1.16.0" {
            return Err(format!("stim version {}, pinned 1.16.0", client.version));
        }
        if client.smoke_x != 1 || client.smoke_z != 0 {
            return Err(format!(
                "peek smoke on H|0> returned <X>={} <Z>={}, expected 1 and 0",
                client.smoke_x, client.smoke_z
            ));
        }
        // A wheel that links OpenMP or a BLAS has a thread knob. Leaving it
        // at one thread would be the strawman §8 names. This wheel does not
        // link either; the assert is what makes that a witness.
        let lower = client.linkage.to_ascii_lowercase();
        for needle in ["libomp", "libiomp", "accelerate", "openblas"] {
            if lower.contains(needle) {
                return Err(format!(
                    "stim links {needle}; refusing to time it without setting that knob in its favor (§8). linkage:\n{}",
                    client.linkage
                ));
            }
        }
        Ok(client)
    }

    pub fn load(&mut self, name: &str, circuit: &str) -> Result<Loaded, String> {
        let v = self.call(json!({"cmd": "load", "name": name, "circuit": circuit}))?;
        Ok(Loaded {
            parse_s: v["parse_s"].as_f64().ok_or("load missing parse_s")?,
            num_qubits: v["num_qubits"].as_u64().ok_or("load missing num_qubits")? as u32,
            num_measurements: v["num_measurements"]
                .as_u64()
                .ok_or("load missing num_measurements")? as u32,
        })
    }

    pub fn expectation(&mut self, name: &str, pauli: &str) -> Result<(i64, f64), String> {
        let v = self.call(json!({"cmd": "expectation", "name": name, "pauli": pauli}))?;
        let value = v["value"].as_i64().ok_or("expectation missing value")?;
        let seconds = v["seconds"].as_f64().ok_or("expectation missing seconds")?;
        Ok((value, seconds))
    }

    /// The arm process's pid, for §4.4a's own-tree census.
    ///
    /// `Some` always, while the client is alive; the option exists so a caller
    /// cannot accidentally depend on it after the child is reaped. Counting
    /// the arm as ours matters: stim's CPU is somebody else's work by every
    /// other definition, and if the census booked it as external then every
    /// row where the competitor did real work would void.
    pub fn pid(&self) -> Option<u32> {
        Some(self.child.id())
    }

    /// `U P U-dagger` for the named circuit's Clifford, as a signed stim
    /// PauliString, plus its weight.
    ///
    /// Derived on the competitor's side on purpose: the gate's discriminating
    /// observable must not come from the backend under test, or agreeing with
    /// it proves nothing. The predicted value `+1` is a theorem about the
    /// construction, so a wrong conjugation does not silently pass — it moves
    /// the peek off `+1` and voids the row.
    pub fn heisenberg_image(&mut self, name: &str, pauli: &str) -> Result<(String, u64), String> {
        let v = self.call(json!({"cmd": "heisenberg_image", "name": name, "pauli": pauli}))?;
        let image = v["image"]
            .as_str()
            .ok_or("heisenberg_image missing image")?
            .to_string();
        let weight = v["weight"]
            .as_u64()
            .ok_or("heisenberg_image missing weight")?;
        Ok((image, weight))
    }

    pub fn compile(&mut self, name: &str, seed: u64) -> Result<f64, String> {
        let v = self.call(json!({"cmd": "compile", "name": name, "seed": seed}))?;
        v["compile_s"]
            .as_f64()
            .ok_or("compile missing compile_s".to_string())
    }

    pub fn sample(&mut self, name: &str, shots: u64) -> Result<Sampled, String> {
        let v = self.call(json!({"cmd": "sample", "name": name, "shots": shots}))?;
        let sampled = Sampled {
            seconds: v["seconds"].as_f64().ok_or("sample missing seconds")?,
            shots_observed: v["shots_observed"]
                .as_u64()
                .ok_or("sample missing shots_observed")?,
            num_measurements: v["num_measurements"]
                .as_u64()
                .ok_or("sample missing num_measurements")? as u32,
        };
        crate::gate::account_shots(shots, sampled.shots_observed)?;
        Ok(sampled)
    }

    pub fn floor_expectation(&mut self) -> Result<f64, String> {
        let v = self.call(json!({"cmd": "floor_expectation"}))?;
        v["seconds"]
            .as_f64()
            .ok_or("floor missing seconds".to_string())
    }

    pub fn floor_sample(&mut self) -> Result<f64, String> {
        let v = self.call(json!({"cmd": "floor_sample"}))?;
        v["seconds"]
            .as_f64()
            .ok_or("floor missing seconds".to_string())
    }

    /// Stim's refusal of a circuit that contains `T`. `Ok` is the
    /// exception text, and only when Stim actually raised.
    pub fn refuse_t(&mut self, circuit: &str) -> Result<String, String> {
        let v = self.call(json!({"cmd": "refuse_t", "circuit": circuit}))?;
        if v["refused"].as_bool() != Some(true) {
            return Err("stim accepted a T gate".into());
        }
        let err = v["error"].as_str().unwrap_or("").to_string();
        if err.is_empty() || !err.contains('T') {
            return Err(format!("stim refusal text does not name T: {err}"));
        }
        Ok(err)
    }

    pub fn d25_text(&mut self) -> Result<serde_json::Value, String> {
        self.call(json!({"cmd": "d25_text"}))
    }

    pub fn d25_sample(&mut self, shots: u64) -> Result<serde_json::Value, String> {
        let v = self.call(json!({"cmd": "d25_sample", "shots": shots}))?;
        let observed = v["shots_observed"]
            .as_u64()
            .ok_or("d25 missing shots_observed")?;
        let obs = v["obs_shots_observed"]
            .as_u64()
            .ok_or("d25 missing obs shots")?;
        crate::gate::account_shots(shots, observed)?;
        crate::gate::account_shots(shots, obs)?;
        Ok(v)
    }

    fn call(&mut self, req: serde_json::Value) -> Result<serde_json::Value, String> {
        let line = serde_json::to_string(&req).map_err(|e| e.to_string())?;
        writeln!(self.stdin, "{line}").map_err(|e| format!("write to stim: {e}"))?;
        self.stdin.flush().map_err(|e| format!("flush stim: {e}"))?;
        let mut buf = String::new();
        let n = self
            .stdout
            .read_line(&mut buf)
            .map_err(|e| format!("read from stim: {e}"))?;
        if n == 0 {
            return Err("stim arm closed stdout".into());
        }
        let v: serde_json::Value = serde_json::from_str(&buf)
            .map_err(|e| format!("stim reply is not json ({e}): {buf}"))?;
        if v["ok"].as_bool() != Some(true) {
            return Err(v["error"].as_str().unwrap_or("stim error").to_string());
        }
        Ok(v)
    }
}

impl Drop for StimClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn python_path() -> Result<PathBuf, String> {
    if let Some(p) = std::env::var_os("STIM_PYTHON") {
        return Ok(PathBuf::from(p));
    }
    let p = crate::workload::repo_root().join("tools/qec_cross_check/.venv/bin/python");
    if !p.exists() {
        return Err(format!(
            "no stim python at {}. Install with: python3 -m venv tools/qec_cross_check/.venv && tools/qec_cross_check/.venv/bin/python -m pip install -r tools/emu_compare/requirements-compare-stim.txt",
            p.display()
        ));
    }
    Ok(p)
}

pub fn script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("stim_arm.py")
}

/// Min of `n` calls after one discarded warmup. The floor's own noise
/// should not be a single cold call.
pub fn min_of(n: usize, mut f: impl FnMut() -> Result<f64, String>) -> Result<f64, String> {
    let _ = f()?;
    let mut best = f64::MAX;
    for _ in 0..n {
        let t = f()?;
        if t < best {
            best = t;
        }
    }
    if !best.is_finite() || best <= 0.0 {
        return Err(format!("floor {best} is not a positive finite time"));
    }
    Ok(best)
}
