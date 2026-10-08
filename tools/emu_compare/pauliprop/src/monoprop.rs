// SPDX-License-Identifier: Apache-2.0
//! The monoprop process. Imports and circuit construction happen before any
//! timed call; `seconds` in a response is the propagator construction,
//! `propagate` and `expectation_value` — exactly one expectation's work.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use omega_emu_compare::PathWitness;
use serde_json::{json, Value};

pub struct Monoprop {
    child: Child,
    stdin: std::io::BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    pub hello: Value,
}

impl Monoprop {
    pub fn launch() -> Result<Self, String> {
        let py = std::env::var("EMU_PAULIPROP_PY").map_err(|_| {
            "EMU_PAULIPROP_PY is unset: point it at the venv built from \
             tools/emu_compare/pauliprop/requirements-compare-pauliprop.txt"
                .to_string()
        })?;
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut child = Command::new(&py)
            .arg(dir.join("monoprop_arm.py"))
            .current_dir(&dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONWARNINGS", "ignore")
            .spawn()
            .map_err(|e| format!("spawn monoprop: {e}"))?;
        let stdout = BufReader::new(child.stdout.take().expect("piped"));
        let stdin = std::io::BufWriter::new(child.stdin.take().expect("piped"));
        let mut m = Self {
            child,
            stdin,
            stdout,
            hello: Value::Null,
        };
        m.hello = m.call(json!({"op": "hello"}))?;
        Ok(m)
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn call(&mut self, req: Value) -> Result<Value, String> {
        writeln!(self.stdin, "{req}").map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;
        // monoprop's C++ core may print placement lines on stdout; skip any
        // line that is not a JSON object from the arm.
        loop {
            let mut line = String::new();
            if self
                .stdout
                .read_line(&mut line)
                .map_err(|e| e.to_string())?
                == 0
            {
                return Err("monoprop closed stdout (it refused or crashed)".into());
            }
            if !line.trim_start().starts_with('{') {
                continue;
            }
            let v: Value =
                serde_json::from_str(&line).map_err(|e| format!("monoprop JSON: {e} in {line}"))?;
            if let Some(e) = v.get("error") {
                return Err(format!("monoprop {}: {e}", req["op"]));
            }
            return Ok(v);
        }
    }

    pub fn threads(&self) -> u32 {
        self.hello["threads"].as_u64().unwrap_or(0) as u32
    }

    pub fn versions(&self) -> BTreeMap<String, String> {
        self.hello["versions"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn witnesses(&self) -> Result<Vec<PathWitness>, String> {
        let list = self.hello["witnesses"]
            .as_array()
            .ok_or("hello has no witnesses")?;
        list.iter()
            .map(|w| {
                if w["asserted"].as_bool() != Some(true) {
                    return Err(format!("monoprop witness {} not asserted", w["name"]));
                }
                Ok(PathWitness {
                    name: w["name"].as_str().unwrap_or("").into(),
                    observed: w["observed"].as_str().unwrap_or("").into(),
                    asserted: true,
                })
            })
            .collect()
    }

    /// One timed expectation: `(value, seconds, terms)`.
    pub fn expectation(
        &mut self,
        name: &str,
        terms: &Value,
        cutoff: u32,
        lower_atol: Option<f64>,
    ) -> Result<(f64, f64, u64), String> {
        let v = self.call(json!({"op": "expectation", "name": name, "terms": terms, "cutoff": cutoff, "lower_atol": lower_atol}))?;
        Ok((
            v["value"].as_f64().ok_or("no value")?,
            v["seconds"].as_f64().ok_or("no seconds")?,
            v["terms"].as_u64().unwrap_or(0),
        ))
    }

    pub fn floor(&mut self) -> Result<f64, String> {
        self.call(json!({"op": "floor"}))?["seconds"]
            .as_f64()
            .ok_or_else(|| "no seconds".into())
    }
}

impl Drop for Monoprop {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, r#"{{"op":"quit"}}"#);
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}
