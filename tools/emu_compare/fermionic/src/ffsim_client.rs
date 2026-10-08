// SPDX-License-Identifier: Apache-2.0
//! One ffsim process for the lane. Imports happen at startup, outside every
//! timed region. The seconds in a response are the contraction only.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use serde_json::{json, Value};

pub struct Ffsim {
    child: Child,
    stdin: std::io::BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    pub hello: Value,
}

impl Ffsim {
    pub fn launch() -> Result<Self, String> {
        let python = python_path();
        if !python.is_file() {
            return Err(format!(
                "ffsim venv is not a venv: {} is missing",
                python.display()
            ));
        }
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ffsim_arm.py");
        let mut child = Command::new(&python)
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONWARNINGS", "ignore")
            .spawn()
            .map_err(|e| format!("spawn ffsim: {e}"))?;
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));
        let stdin = std::io::BufWriter::new(child.stdin.take().expect("stdin piped"));
        let mut proc = Self {
            child,
            stdin,
            stdout,
            hello: Value::Null,
        };
        let hello = proc.call(json!({"op": "hello"}))?;
        if hello
            .get("self_check_sector_refused")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(format!(
                "ffsim hello did not refuse the Kitaev sector path: {hello}"
            ));
        }
        let energy = hello
            .get("self_check_kitaev_n2")
            .and_then(Value::as_f64)
            .ok_or("ffsim hello missing self_check_kitaev_n2")?;
        if (energy - (-1.0)).abs() > 1e-10 {
            return Err(format!("ffsim Kitaev n2 self-check was {energy}, want -1"));
        }
        proc.hello = hello;
        Ok(proc)
    }

    /// The ffsim process's pid, counted as the lane's own tree by §4.4a's census.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn version(&self, name: &str) -> Result<String, String> {
        self.hello
            .get(name)
            .and_then(Value::as_str)
            .map(|s| s.to_string())
            .ok_or_else(|| format!("ffsim hello missing {name}"))
    }

    fn call(&mut self, req: Value) -> Result<Value, String> {
        writeln!(self.stdin, "{req}").map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        let n = self
            .stdout
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("ffsim closed stdout".into());
        }
        let value: Value =
            serde_json::from_str(&line).map_err(|e| format!("ffsim JSON: {e} in {line}"))?;
        if value.get("ok").and_then(Value::as_bool) == Some(false) && req["op"] != "kitaev64" {
            return Err(format!(
                "ffsim {}: {}",
                req["op"],
                value.get("error").and_then(Value::as_str).unwrap_or(&line)
            ));
        }
        Ok(value)
    }

    pub fn h2(&mut self) -> Result<Value, String> {
        self.call(json!({"op": "h2"}))
    }

    pub fn hubbard(&mut self) -> Result<Value, String> {
        self.call(json!({"op": "hubbard"}))
    }

    pub fn kitaev(&mut self, n: u32) -> Result<Value, String> {
        self.call(json!({"op": "kitaev", "n": n}))
    }

    pub fn kitaev64(&mut self) -> Result<Value, String> {
        self.call(json!({"op": "kitaev64"}))
    }

    pub fn lucj_prepare(&mut self) -> Result<Value, String> {
        self.call(json!({"op": "lucj_prepare"}))
    }

    pub fn lucj(&mut self) -> Result<Value, String> {
        self.call(json!({"op": "lucj"}))
    }

    pub fn floor(&mut self) -> Result<f64, String> {
        let v = self.call(json!({"op": "floor"}))?;
        v.get("seconds")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("floor response has no seconds: {v}"))
    }
}

impl Drop for Ffsim {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, r#"{{"op":"quit"}}"#);
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

fn python_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../crates/omega-bridges/python/.venv-ffsim/bin/python")
}

pub fn number(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing {key} in {v}"))
}

pub fn text(v: &Value, key: &str) -> Result<String, String> {
    v.get(key)
        .and_then(Value::as_str)
        .map(|s| s.to_string())
        .ok_or_else(|| format!("missing {key} in {v}"))
}
