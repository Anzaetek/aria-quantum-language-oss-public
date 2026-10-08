// SPDX-License-Identifier: Apache-2.0
//! Circuits, observables, our arm and the oracle.

use std::path::{Path, PathBuf};

use omega_backend_pauliprop::sim::{PauliPropBackend, PauliPropCertificate};
use omega_core::circuit::{CircuitIR, CircuitType};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;
use omega_parser::lower::{lower_qasm2, lower_to_ir_with_dialect, Qasm2Dialect};
use omega_parser::parse_qasm2;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const OURS_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repo root")
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One cell: the circuit both arms run, the observable, and how monoprop gets it.
pub struct Cell {
    pub id: String,
    pub file: String,
    pub sha256: String,
    pub generator: String,
    pub seed: Option<u64>,
    pub qubits: u32,
    pub depth: u32,
    pub circuit: CircuitIR,
    /// `[[coeff, [[qubit, "X"|"Y"|"Z"], ...]], ...]`, for monoprop.
    pub terms: Value,
    pub observable: Observable,
    /// The observable as our parser spells it (printed into the row).
    pub observable_text: String,
    /// How monoprop loads it: `{"op": "load_ops", "n", "ops"}` or `{"op": "load_qasm", "path"}`.
    pub monoprop_load: Value,
}

fn letter(p: PauliOp) -> &'static str {
    match p {
        PauliOp::X => "X",
        PauliOp::Y => "Y",
        PauliOp::Z => "Z",
        PauliOp::I => "I",
    }
}

fn terms_json(obs: &Observable) -> Value {
    Value::Array(
        obs.terms
            .iter()
            .map(|(c, ps)| {
                json!([
                    c,
                    ps.iter()
                        .map(|(q, p)| json!([q, letter(*p)]))
                        .collect::<Vec<_>>()
                ])
            })
            .collect(),
    )
}

fn obs_text(obs: &Observable) -> String {
    obs.terms
        .iter()
        .map(|(c, ps)| {
            let s: String = ps
                .iter()
                .map(|(q, p)| format!("{}{q}", letter(*p)))
                .collect();
            format!("{c}*{s}")
        })
        .collect::<Vec<_>>()
        .join("+")
}

/// The exact cases of the committed monoprop fixture (`cutoff == n`, no
/// `lower_atol`). The truncated cases are not timed: at equal knobs the two
/// engines do not do the same work (see `tests/monoprop_xcheck.rs`), and the
/// HEA matched-accuracy row is where truncation is compared.
pub fn fixture_exact_cells() -> Result<Vec<Cell>, String> {
    let rel = "tools/pp_cross_check/monoprop_fixture.jsonl";
    let path = repo_root().join(rel);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let sha = sha256_hex(&bytes);
    let mut out = Vec::new();
    for line in String::from_utf8(bytes).map_err(|e| e.to_string())?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        if v.get("meta").is_some() {
            continue;
        }
        let n = v["n_qubits"].as_u64().ok_or("n_qubits")? as u32;
        let cutoff = v["cutoff"].as_u64().ok_or("cutoff")? as u32;
        if cutoff < n || !v["lower_atol"].is_null() {
            continue;
        }
        let name = v["case"].as_str().ok_or("case")?.to_string();
        let ops = v["ops"].as_array().ok_or("ops")?.clone();
        let mut terms = Vec::new();
        for t in v["observable"].as_array().ok_or("observable")? {
            let c = t[0].as_f64().ok_or("coeff")?;
            let letters = t[1].as_str().ok_or("letters")?;
            let qs = t[2].as_array().ok_or("qubits")?;
            let ps = letters
                .chars()
                .zip(qs)
                .map(|(l, q)| {
                    let p = match l {
                        'X' => PauliOp::X,
                        'Y' => PauliOp::Y,
                        'Z' => PauliOp::Z,
                        o => return Err(format!("letter {o}")),
                    };
                    Ok((q.as_u64().ok_or("qubit")? as u32, p))
                })
                .collect::<Result<Vec<_>, String>>()?;
            terms.push((c, ps));
        }
        let observable = Observable { terms };
        let qasm = to_qasm(n, &ops)?;
        let prog = parse_qasm2(&qasm).map_err(|e| format!("{name}: parse: {e}"))?;
        let circuit = lower_qasm2(&prog).map_err(|e| format!("{name}: lower: {e}"))?;
        out.push(Cell {
            // The fixture's own "_exact" suffix is dropped: the row id adds
            // "-exact" itself, and the truncated cases are never rows.
            id: format!(
                "pp-fixture-{}",
                name.trim_end_matches("_exact").replace('_', "-")
            ),
            file: rel.to_string(),
            sha256: sha.clone(),
            generator: format!("monoprop_ref.py case {name}"),
            seed: None,
            qubits: n,
            depth: ops.len() as u32,
            circuit,
            terms: terms_json(&observable),
            observable_text: obs_text(&observable),
            observable,
            monoprop_load: json!({"op": "load_ops", "n": n, "ops": ops}),
        });
    }
    Ok(out)
}

/// The structured ops as OpenQASM 2, the same rendering
/// `crates/omega-backend-pauliprop/tests/monoprop_xcheck.rs` uses.
pub fn to_qasm(n: u32, ops: &[Value]) -> Result<String, String> {
    let mut s = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\n");
    for op in ops {
        let name = op["op"].as_str().ok_or("op name")?;
        if !matches!(name, "rx" | "ry" | "rz" | "rxx" | "rzz") {
            return Err(format!("op {name} outside the rendered subset"));
        }
        let theta = op["theta"].as_f64().ok_or("theta")?;
        let qs: Vec<String> = op["q"]
            .as_array()
            .ok_or("q")?
            .iter()
            .map(|q| format!("q[{}]", q.as_u64().unwrap_or(0)))
            .collect();
        s.push_str(&format!("{name}({theta}) {};\n", qs.join(", ")));
    }
    Ok(s)
}

/// The §3 HEA artifact, hash-checked, with E4's observable
/// `O = Σ ZᵢZᵢ₊₁ + Σ Xᵢ` (⟨Zᵢ⟩ = 0 on every HEA artifact, so a single-Z gate
/// would be vacuous).
pub fn hea_cell(id: &str) -> Result<Cell, String> {
    let root = repo_root();
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("tools/emu_compare/MANIFEST.json"))
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let art = manifest["artifacts"]
        .get(id)
        .ok_or_else(|| format!("no artifact {id}"))?;
    let file = art["file"].as_str().ok_or("file")?.to_string();
    let sha = art["sha256"].as_str().ok_or("sha256")?.to_string();
    let path = root.join(&file);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if sha256_hex(&bytes) != sha {
        return Err(format!("{id}: sha256 mismatch against MANIFEST.json"));
    }
    let n = art["qubits"].as_u64().ok_or("qubits")? as u32;
    let circuit = lower_to_ir_with_dialect(
        &String::from_utf8(bytes).map_err(|e| e.to_string())?,
        Qasm2Dialect::Legacy,
    )
    .map_err(|e| format!("{id}: lowering: {e}"))?;
    let mut parts: Vec<String> = (0..n - 1).map(|i| format!("Z{i}Z{}", i + 1)).collect();
    parts.extend((0..n).map(|i| format!("X{i}")));
    let observable = Observable::parse(&parts.join("+"))?;
    Ok(Cell {
        id: id.to_string(),
        file,
        sha256: sha,
        generator: art["generator"].as_str().unwrap_or("").to_string(),
        seed: art["seed"].as_u64(),
        qubits: n,
        depth: art["depth"].as_u64().unwrap_or(0) as u32,
        circuit,
        terms: terms_json(&observable),
        observable_text: parts.join("+"),
        observable,
        monoprop_load: json!({"op": "load_qasm", "path": path}),
    })
}

/// Our truncation setting, in our knob names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OursKnobs {
    pub coeff_min: f64,
    pub max_weight: Option<usize>,
}

/// One timed expectation of ours: `(value, seconds, certificate)`.
pub fn ours(cell: &Cell, k: OursKnobs) -> Result<(f64, f64, PauliPropCertificate), String> {
    let backend = PauliPropBackend::with_truncation(k.coeff_min, k.max_weight);
    let t = std::time::Instant::now();
    let (v, cert) = backend
        .expectation_with_certificate(&cell.circuit, &ParameterBinding::new(), &cell.observable)
        .map_err(|e| e.to_string())?;
    let dt = t.elapsed().as_secs_f64();
    Ok((v, dt, cert))
}

/// Asserted: the backend is pauliprop, and it is single-threaded by design.
pub fn ours_witness() -> Result<Vec<omega_emu_compare::PathWitness>, String> {
    let b = PauliPropBackend::new();
    let name = b.name().to_string();
    if name != "pauliprop" {
        return Err(format!("our backend is {name:?}, this lane is pauliprop"));
    }
    Ok(vec![omega_emu_compare::PathWitness {
        name: "backend_name".into(),
        observed: name,
        asserted: true,
    }])
}

/// Smallest invocation of our timed path: one qubit, no gates, `<Z0>`.
pub fn ours_floor() -> Result<f64, String> {
    let circuit = CircuitIR::new(1, CircuitType::GateBased);
    let obs = Observable::parse("Z0")?;
    let backend = PauliPropBackend::new();
    let t = std::time::Instant::now();
    let (v, _) = backend
        .expectation_with_certificate(&circuit, &ParameterBinding::new(), &obs)
        .map_err(|e| e.to_string())?;
    let dt = t.elapsed().as_secs_f64();
    if (v - 1.0).abs() > 1e-12 {
        return Err(format!("floor <Z0> is {v}"));
    }
    Ok(dt)
}

/// The value gate's oracle: dense f64 statevector, a different algorithm from
/// both arms.
pub fn oracle(cell: &Cell) -> Result<f64, String> {
    omega_backend_statevector::StatevectorBackend::new()
        .expectation(&cell.circuit, &ParameterBinding::new(), &cell.observable)
        .map_err(|e| e.to_string())
}
