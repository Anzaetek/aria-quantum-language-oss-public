// SPDX-License-Identifier: Apache-2.0
//! Our arm of the GPU lane, as a process of its own.
//!
//! It is a child of the lane binary, like the competitors, for one reason: the
//! row's device census (§4.4) voids any row whose card has a holder before or
//! after it, with no exemption for the lane's own processes. So every arm that
//! touches the card must be started after the "before" census and gone before
//! the "after" one, and an in-process CUDA context could not be.
//!
//! JSON lines on stdin/stdout, the same protocol as the Python arms
//! (`tools/emu_compare/dense/dense_common.py`): `hello`, `load`,
//! `expectation`, `state`, `floor`, `quit`. The `seconds` in a response is the
//! wall time of exactly `Backend::expectation` on a circuit lowered before the
//! clock started (§4.2); the CLI is not used because `omega-run --device cuda
//! --expectation` is refused (that mode has no CUDA dispatch).

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::time::Instant;

use omega_backend_statevector_cuda::CudaStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType};
use omega_core::executor::{Backend, ExecConfig, ExecResult, Observable};
use omega_core::params::ParameterBinding;
use omega_parser::lower::{lower_to_ir_with_dialect, Qasm2Dialect};
use serde_json::{json, Value};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn observable_from_terms(terms: &Value) -> Result<Observable, String> {
    let list = terms.as_array().ok_or("terms is not an array")?;
    let mut parts = Vec::with_capacity(list.len());
    for t in list {
        let coeff = t[0].as_f64().ok_or("term without a coefficient")?;
        let mut s = String::new();
        for p in t[1].as_array().ok_or("term without paulis")? {
            let q = p[0].as_u64().ok_or("pauli without qubit")?;
            let op = p[1].as_str().ok_or("pauli without letter")?;
            s.push_str(&format!("{op}{q}"));
        }
        parts.push(format!("{coeff}*{s}"));
    }
    Observable::parse(&parts.join("+"))
}

struct Arm {
    backend: CudaStatevectorBackend,
    circuits: HashMap<String, CircuitIR>,
}

impl Arm {
    fn expectation(&self, circuit: &CircuitIR, obs: &Observable) -> Result<(f64, f64), String> {
        let t = Instant::now();
        let v = self
            .backend
            .expectation(circuit, &ParameterBinding::new(), obs)
            .map_err(|e| e.to_string())?;
        let dt = t.elapsed().as_secs_f64();
        if !v.is_finite() {
            return Err(format!("our expectation is {v}"));
        }
        Ok((v, dt))
    }

    fn handle(&mut self, req: &Value) -> Result<Value, String> {
        match req["op"].as_str().unwrap_or("") {
            "hello" => {
                let name = self.backend.name().to_string();
                if name != "cuda-statevector" {
                    return Err(format!(
                        "our backend is {name:?}, the GPU lane is the CUDA statevector"
                    ));
                }
                Ok(json!({
                    "versions": {"omega-backend-statevector-cuda": VERSION},
                    "threads": 1,
                    "witnesses": [
                        {"name": "backend_name", "observed": name, "asserted": true},
                        {"name": "device_backend_constructed",
                         "observed": "CudaStatevectorBackend::new() succeeded (it has no CPU fallback)",
                         "asserted": true},
                    ],
                }))
            }
            "load" => {
                let path = req["path"].as_str().ok_or("load without path")?;
                let name = req["name"].as_str().ok_or("load without name")?.to_string();
                let t = Instant::now();
                let src = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
                let c = lower_to_ir_with_dialect(&src, Qasm2Dialect::Legacy)
                    .map_err(|e| format!("{path}: {e}"))?;
                let build_s = t.elapsed().as_secs_f64();
                self.circuits.insert(name, c);
                Ok(json!({"build_s": build_s}))
            }
            "expectation" => {
                let name = req["name"].as_str().ok_or("expectation without name")?;
                let c = self
                    .circuits
                    .get(name)
                    .ok_or_else(|| format!("{name} not loaded"))?;
                let obs = observable_from_terms(&req["terms"])?;
                obs.validate_qubits(c.num_qubits)
                    .map_err(|e| e.to_string())?;
                let (v, dt) = self.expectation(c, &obs)?;
                Ok(json!({"value": v, "seconds": dt}))
            }
            "state" => {
                let name = req["name"].as_str().ok_or("state without name")?;
                let out = req["out"].as_str().ok_or("state without out")?;
                let c = self
                    .circuits
                    .get(name)
                    .ok_or_else(|| format!("{name} not loaded"))?;
                let cfg = ExecConfig {
                    shots: None,
                    ..Default::default()
                };
                let st = match self
                    .backend
                    .execute(c, &ParameterBinding::new(), &cfg)
                    .map_err(|e| e.to_string())?
                {
                    ExecResult::Statevector(v) => v,
                    _ => return Err("execute(shots: None) did not return a statevector".into()),
                };
                emu_compare_dense::workload::write_npy(std::path::Path::new(out), &st)?;
                Ok(json!({}))
            }
            "floor" => {
                let c = CircuitIR::new(1, CircuitType::GateBased);
                let obs = Observable::parse("Z0")?;
                let (v, dt) = self.expectation(&c, &obs)?;
                if (v - 1.0).abs() > 1e-6 {
                    return Err(format!("floor <Z0> on |0> is {v}"));
                }
                Ok(json!({"seconds": dt}))
            }
            other => Err(format!("unknown op {other:?}")),
        }
    }
}

fn main() {
    let backend = match CudaStatevectorBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("emu-compare-gpu-ours: no CUDA device ({e}); this arm never falls back");
            std::process::exit(3);
        }
    };
    let mut arm = Arm {
        backend,
        circuits: HashMap::new(),
    };
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(out, "{}", json!({"error": format!("bad JSON: {e}")}));
                let _ = out.flush();
                continue;
            }
        };
        if req["op"] == "quit" {
            let _ = writeln!(out, "{}", json!({"ok": true}));
            let _ = out.flush();
            return;
        }
        let resp = match arm.handle(&req) {
            Ok(Value::Object(mut m)) => {
                m.insert("ok".into(), json!(true));
                Value::Object(m)
            }
            Ok(other) => json!({"ok": true, "value": other}),
            Err(e) => json!({"error": e}),
        };
        let _ = writeln!(out, "{resp}");
        let _ = out.flush();
    }
}
