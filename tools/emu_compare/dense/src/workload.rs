// SPDX-License-Identifier: Apache-2.0
//! The pinned circuits, the pinned observable, and our arm.
//!
//! Every arm ingests the committed QASM bytes (§3), hash-checked against
//! `MANIFEST.json`. Ours lowers them once, before any clock, exactly as the
//! competitors build their circuit objects before theirs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use num_complex::Complex64;
use omega_backend_statevector::sim::sample_from_state;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType};
use omega_core::executor::{Backend, ExecConfig, ExecResult, Observable};
use omega_core::params::ParameterBinding;
use omega_parser::lower::{lower_to_ir_with_dialect, Qasm2Dialect};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// The observable every dense expectation row measures:
/// `O = Σᵢ ZᵢZᵢ₊₁ + Σᵢ Xᵢ` (2n−1 terms; E5's choice).
///
/// Not `Z0`: on every committed HEA artifact `⟨Zᵢ⟩ = 0` for every qubit to
/// rounding (max |Zᵢ| ≤ 1.4e-16), so a `Z0` gate passes any arm that returns
/// zero — including a broken one. `O` is ≈ +15 / +21 on HEA and ≈ +0.05..0.1 on
/// RANDOM-1, and it touches every qubit, so no gate is outside its backward
/// light cone.
pub const OBSERVABLE_NAME: &str = "sum_i Z_i Z_{i+1} + sum_i X_i";

/// Sampling row: shots and seed (§3: one shots=1000 row at 20q).
pub const SHOTS: u32 = 1000;
pub const SEED: u64 = 0;

/// Our crate's version string, as the workspace pins it.
pub const OURS_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A committed artifact and the IR our arm executes.
#[derive(Clone)]
pub struct Artifact {
    pub id: String,
    pub file: String,
    pub path: PathBuf,
    pub sha256: String,
    pub generator: String,
    pub seed: Option<u64>,
    pub qubits: u32,
    pub depth: u32,
    pub circuit: CircuitIR,
}

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repo root from tools/emu_compare/dense")
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Load a manifest artifact; refuse if bytes, hash or qubit count disagree.
pub fn load_artifact(id: &str) -> Result<Artifact, String> {
    let root = repo_root();
    let manifest_path = root.join("tools/emu_compare/MANIFEST.json");
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let art = doc["artifacts"]
        .get(id)
        .ok_or_else(|| format!("no artifact {id} in MANIFEST.json"))?;
    let s = |k: &str| -> Result<String, String> {
        art[k]
            .as_str()
            .map(|x| x.to_string())
            .ok_or_else(|| format!("{id}: manifest has no {k}"))
    };
    let file = s("file")?;
    let sha = s("sha256")?;
    let generator = s("generator")?;
    let qubits = art["qubits"].as_u64().ok_or("artifact has no qubits")? as u32;
    let depth = art["depth"].as_u64().ok_or("artifact has no depth")? as u32;
    let seed = art["seed"].as_u64();
    if art["dialect"].as_str().unwrap_or("legacy") != "legacy" {
        return Err(format!("{id}: this lane ingests legacy QASM only"));
    }
    let path = root.join(&file);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let got = sha256_hex(&bytes);
    if got != sha {
        return Err(format!("{id}: file sha256 {got} != manifest {sha}"));
    }
    let src = String::from_utf8(bytes).map_err(|e| e.to_string())?;
    let circuit = lower_to_ir_with_dialect(&src, Qasm2Dialect::Legacy)
        .map_err(|e| format!("{id}: lowering: {e}"))?;
    if circuit.num_qubits != qubits {
        return Err(format!(
            "{id}: IR has {} qubits, manifest says {qubits}",
            circuit.num_qubits
        ));
    }
    Ok(Artifact {
        id: id.to_string(),
        file,
        path,
        sha256: sha,
        generator,
        seed,
        qubits,
        depth,
        circuit,
    })
}

/// `O` in our parser's syntax, e.g. `Z0Z1+Z1Z2+X0+X1+X2` for n = 3.
pub fn observable_string(n: u32) -> String {
    let mut parts: Vec<String> = (0..n.saturating_sub(1))
        .map(|i| format!("Z{i}Z{}", i + 1))
        .collect();
    parts.extend((0..n).map(|i| format!("X{i}")));
    parts.join("+")
}

/// `O` as the arms' JSON terms: `[[coeff, [[qubit, "Z"], ...]], ...]`.
pub fn observable_terms(n: u32) -> Value {
    let mut terms = Vec::new();
    for i in 0..n.saturating_sub(1) {
        terms.push(json!([1.0, [[i, "Z"], [i + 1, "Z"]]]));
    }
    for i in 0..n {
        terms.push(json!([1.0, [[i, "X"]]]));
    }
    Value::Array(terms)
}

/// Our arm, pinned to a rayon pool of `threads`.
pub struct Ours {
    pub threads: u32,
    pool: rayon::ThreadPool,
    backend: StatevectorBackend,
}

impl Ours {
    pub fn new(threads: u32) -> Result<Self, String> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .build()
            .map_err(|e| format!("rayon pool of {threads}: {e}"))?;
        let ours = Self {
            threads,
            pool,
            backend: StatevectorBackend::new(),
        };
        ours.witness()?;
        Ok(ours)
    }

    /// The arm's preconditions, asserted: the backend is the statevector
    /// backend (not an MPS or other backend through name drift), and the
    /// rayon pool its kernels run in has exactly the pinned thread count.
    pub fn witness(&self) -> Result<Vec<omega_emu_compare::PathWitness>, String> {
        let name = self.backend.name().to_string();
        if name != "statevector" {
            return Err(format!(
                "our backend is {name:?}, the dense lane is statevector"
            ));
        }
        let inside = self.pool.install(rayon::current_num_threads);
        if inside != self.threads as usize {
            return Err(format!(
                "rayon reports {inside} threads inside a pool pinned to {}",
                self.threads
            ));
        }
        Ok(vec![
            omega_emu_compare::PathWitness {
                name: "backend_name".into(),
                observed: name,
                asserted: true,
            },
            omega_emu_compare::PathWitness {
                name: "rayon_current_num_threads".into(),
                observed: inside.to_string(),
                asserted: true,
            },
        ])
    }

    /// One timed expectation through the library path (§4.2): `(value, s)`.
    pub fn expectation(&self, circuit: &CircuitIR, obs: &Observable) -> Result<(f64, f64), String> {
        self.pool.install(|| {
            let start = std::time::Instant::now();
            let v = self
                .backend
                .expectation(circuit, &ParameterBinding::new(), obs)
                .map_err(|e| e.to_string())?;
            let dt = start.elapsed().as_secs_f64();
            if !v.is_finite() {
                return Err(format!("our expectation is {v}"));
            }
            Ok((v, dt))
        })
    }

    /// The evolved state, untimed (value gate and sampling setup).
    pub fn state(&self, circuit: &CircuitIR) -> Result<Vec<Complex64>, String> {
        self.pool.install(|| {
            let cfg = ExecConfig {
                shots: None,
                ..Default::default()
            };
            match self
                .backend
                .execute(circuit, &ParameterBinding::new(), &cfg)
                .map_err(|e| e.to_string())?
            {
                ExecResult::Statevector(v) => Ok(v),
                _ => Err("execute(shots: None) did not return a statevector".into()),
            }
        })
    }

    /// One timed sampling call on an evolved state: `(counts, seconds)`.
    pub fn sample(
        &self,
        state: &[Complex64],
        n: u32,
        shots: u32,
        seed: u64,
    ) -> Result<(HashMap<u64, u32>, f64), String> {
        self.pool.install(|| {
            let start = std::time::Instant::now();
            let counts = sample_from_state(state, n as usize, shots, Some(seed));
            let dt = start.elapsed().as_secs_f64();
            let got: u64 = counts.values().map(|c| u64::from(*c)).sum();
            if got != u64::from(shots) {
                return Err(format!("asked {shots} shots, sampler returned {got}"));
            }
            Ok((counts, dt))
        })
    }

    /// Smallest invocation of our expectation path: 1 qubit, no gates, `<Z0>`.
    pub fn floor(&self) -> Result<f64, String> {
        let circuit = CircuitIR::new(1, CircuitType::GateBased);
        let obs = Observable::parse("Z0")?;
        let (v, dt) = self.expectation(&circuit, &obs)?;
        if (v - 1.0).abs() > 1e-12 {
            return Err(format!("floor <Z0> on |0> is {v}"));
        }
        Ok(dt)
    }

    /// Smallest invocation of our sampling path: one shot of |0>.
    pub fn floor_sample(&self) -> Result<f64, String> {
        let state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let (counts, dt) = self.sample(&state, 1, 1, SEED)?;
        if counts.get(&0) != Some(&1) {
            return Err(format!("floor sample of |0> is {counts:?}"));
        }
        Ok(dt)
    }
}

/// Per-qubit `<Z>` estimates from counts (qubit 0 = LOW bit).
pub fn z_marginals(counts: &HashMap<u64, u32>, n: u32, shots: u32) -> Vec<f64> {
    (0..n)
        .map(|q| {
            let ones: u64 = counts
                .iter()
                .filter(|(k, _)| (**k >> q) & 1 == 1)
                .map(|(_, c)| u64::from(*c))
                .sum();
            1.0 - 2.0 * ones as f64 / f64::from(shots)
        })
        .collect()
}

/// Write a state as a complex128 `.npy` (format 1.0, little-endian), qubit 0
/// the LOW bit, for the oracle's untimed state comparison.
pub fn write_npy(path: &Path, state: &[Complex64]) -> Result<(), String> {
    use std::io::Write;
    let mut header = format!(
        "{{'descr': '<c16', 'fortran_order': False, 'shape': ({},), }}",
        state.len()
    );
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
    let io = |e: std::io::Error| e.to_string();
    w.write_all(b"\x93NUMPY\x01\x00").map_err(io)?;
    w.write_all(&(header.len() as u16).to_le_bytes())
        .map_err(io)?;
    w.write_all(header.as_bytes()).map_err(io)?;
    for a in state {
        w.write_all(&a.re.to_le_bytes()).map_err(io)?;
        w.write_all(&a.im.to_le_bytes()).map_err(io)?;
    }
    w.flush().map_err(io)
}
