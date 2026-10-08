// SPDX-License-Identifier: Apache-2.0
//! The pinned circuits, ingested once and handed to both arms.
//!
//! The brickwall bytes are the committed QASM. The d=5 memory circuit is
//! `SurfaceCode::syndrome_circuit` at distance 5 — the rotated-surface
//! extraction the qec cross-check's code is — lowered through the same
//! `to_omega_core_ir` that crate's simulator path uses. Stim text is
//! emitted from that IR, so the two arms cannot be handed two circuits.

use std::path::{Path, PathBuf};

use omega_backend_pauli::PauliBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable};
use omega_core::params::ParameterBinding;
use omega_parser::lower::{lower_to_ir_with_dialect, Qasm2Dialect};
use sha2::{Digest, Sha256};

use crate::gate::{account_shots, exact_gate};

/// Shots on every sampling row. Pinned in §3 for the brickwall; the d=5
/// row uses the same count rather than a smaller one.
pub const SHOTS: u32 = 10_000;

/// Sampling seed, both arms. The streams are not compared; the seed is
/// printed so a row can be re-run.
pub const SEED: u64 = 0;

/// A committed artifact, plus the IR both arms execute.
#[derive(Clone)]
pub struct Artifact {
    /// Manifest id.
    pub id: String,
    /// Repo-relative path.
    pub file: String,
    /// sha256 of the file bytes.
    pub sha256: String,
    /// Generator line from the manifest.
    pub generator: String,
    /// Qubits, from the manifest and checked against the IR.
    pub qubits: u32,
    /// Depth, from the manifest.
    pub depth: u32,
    /// Lowered in the manifest's dialect.
    pub circuit: CircuitIR,
}

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repo root from tools/emu_compare/stim")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Load a manifest artifact and refuse if the bytes, the hash, or the
/// qubit count disagree with the IR.
pub fn load_artifact(id: &str) -> Result<Artifact, String> {
    let root = repo_root();
    let manifest_path = root.join("tools/emu_compare/MANIFEST.json");
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let doc: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let art = doc["artifacts"]
        .get(id)
        .ok_or_else(|| format!("no artifact {id} in MANIFEST.json"))?;
    let file = art["file"]
        .as_str()
        .ok_or("artifact has no file")?
        .to_string();
    let sha = art["sha256"]
        .as_str()
        .ok_or("artifact has no sha256")?
        .to_string();
    let generator = art["generator"]
        .as_str()
        .ok_or("artifact has no generator")?
        .to_string();
    let qubits = art["qubits"].as_u64().ok_or("artifact has no qubits")? as u32;
    let depth = art["depth"].as_u64().ok_or("artifact has no depth")? as u32;
    let dialect = art["dialect"].as_str().unwrap_or("legacy");
    if dialect != "legacy" {
        return Err(format!(
            "{id} is dialect {dialect}; this lane ingests legacy QASM"
        ));
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
        sha256: sha,
        generator,
        qubits,
        depth,
        circuit,
    })
}

fn gate(kind: GateKind, qubits: &[u32], classical_bit: Option<u32>) -> GateOp {
    let mut op = GateOp {
        gate: kind,
        qubits: Default::default(),
        params: Default::default(),
        classical_bit,
        condition: None,
    };
    for q in qubits {
        op.qubits.push(Qubit(*q));
    }
    op
}

/// Stim program text. One line per IR op, IR qubit order, which for CX is
/// control then target — the order the committed QASM has.
pub fn to_stim(circuit: &CircuitIR) -> Result<String, String> {
    let mut lines = Vec::with_capacity(circuit.ops.len());
    for op in &circuit.ops {
        let q = |i: usize| -> Result<u32, String> {
            op.qubits
                .get(i)
                .map(|q| q.0)
                .ok_or_else(|| format!("{:?} missing qubit {i}", op.gate))
        };
        let line = match &op.gate {
            GateKind::H => format!("H {}", q(0)?),
            GateKind::S => format!("S {}", q(0)?),
            GateKind::Sdg => format!("S_DAG {}", q(0)?),
            GateKind::X => format!("X {}", q(0)?),
            GateKind::Y => format!("Y {}", q(0)?),
            GateKind::Z => format!("Z {}", q(0)?),
            GateKind::Id => format!("I {}", q(0)?),
            GateKind::CX => format!("CX {} {}", q(0)?, q(1)?),
            GateKind::CY => format!("CY {} {}", q(0)?, q(1)?),
            GateKind::CZ => format!("CZ {} {}", q(0)?, q(1)?),
            GateKind::Swap => format!("SWAP {} {}", q(0)?, q(1)?),
            GateKind::Measure => format!("M {}", q(0)?),
            GateKind::Reset => format!("R {}", q(0)?),
            GateKind::Barrier => continue,
            other => {
                return Err(format!(
                    "no stim lowering for {other:?}; refusing to drop the gate"
                ))
            }
        };
        lines.push(line);
    }
    Ok(lines.join("\n"))
}

/// Drop measurements. Expectation rows on both arms see this circuit:
/// Stim's `do` would otherwise collapse, and our `expectation` would
/// defer, and those are not the same computation.
pub fn without_measures(circuit: &CircuitIR) -> CircuitIR {
    let mut out = circuit.clone();
    out.ops.retain(|op| op.gate != GateKind::Measure);
    out.num_classical_bits = 0;
    out
}

/// Terminal computational-basis measurement of every qubit, which is what
/// our sampler does implicitly when the circuit has no `measure`. Stim
/// has no implicit final measure, so the sampling arm gets these lines.
pub fn with_terminal_measures(circuit: &CircuitIR) -> CircuitIR {
    let mut out = circuit.clone();
    out.num_classical_bits = circuit.num_qubits;
    for q in 0..circuit.num_qubits {
        out.ops.push(gate(GateKind::Measure, &[q], Some(q)));
    }
    out
}

/// The brickwall plus one `T` on qubit 0. The file bytes are not rewritten.
pub fn append_t_on_zero(circuit: &CircuitIR) -> CircuitIR {
    let mut out = circuit.clone();
    out.ops.push(gate(GateKind::T, &[0], None));
    out
}

/// `Z0` → `+Z___…` of length `n`, wire 0 on the left. Stim's PauliString
/// is LSB-first, the same order as our wire indices.
pub fn observable_to_stim_pauli(observable: &str, n: u32) -> Result<String, String> {
    let parsed = Observable::parse(observable)?;
    if parsed.terms.len() != 1 {
        return Err(format!(
            "{observable}: stim peek takes one Pauli term, got {}",
            parsed.terms.len()
        ));
    }
    let (coeff, sites) = &parsed.terms[0];
    if *coeff != 1.0 {
        return Err(format!("{observable}: coefficient {coeff}, expected 1"));
    }
    let mut chars = vec!['_'; n as usize];
    for (q, op) in sites {
        let ch = match op {
            omega_core::executor::PauliOp::X => 'X',
            omega_core::executor::PauliOp::Y => 'Y',
            omega_core::executor::PauliOp::Z => 'Z',
            omega_core::executor::PauliOp::I => '_',
        };
        let idx = *q as usize;
        if idx >= chars.len() {
            return Err(format!("{observable}: qubit {q} past n={n}"));
        }
        chars[idx] = ch;
    }
    Ok(format!("+{}", chars.into_iter().collect::<String>()))
}

/// A signed stim PauliString → our `Observable` syntax.
///
/// `"+XZ_Y"` → `"X0Z1Y3"`; `"-XZ_Y"` → `"-1*X0Z1Y3"`. The inverse of
/// [`observable_to_stim_pauli`], needed because the gate's discriminating
/// observable is derived on the stim side (see
/// `StimArm::heisenberg_image`) and then has to be measured on ours.
///
/// Refuses `i` and `-i` signs. stim admits them and our `Observable` has no
/// imaginary coefficient, so converting one would quietly drop the phase. A
/// conjugation of a Hermitian Pauli by a Clifford is Hermitian, so the sign
/// is real on every string this is called with — the refusal is for the case
/// where that stops being true.
pub fn stim_pauli_to_observable(p: &str) -> Result<String, String> {
    let (sign, body) = match p.chars().next() {
        Some('+') => (1.0_f64, &p[1..]),
        Some('-') => (-1.0_f64, &p[1..]),
        Some('i') => return Err(format!("{p}: imaginary sign has no Observable encoding")),
        Some('X' | 'Y' | 'Z' | '_') => (1.0, p),
        _ => return Err(format!("{p}: not a stim PauliString")),
    };
    if body.starts_with('i') {
        return Err(format!("{p}: imaginary sign has no Observable encoding"));
    }
    let mut frags = String::new();
    for (q, ch) in body.chars().enumerate() {
        match ch {
            'X' | 'Y' | 'Z' => frags.push_str(&format!("{ch}{q}")),
            '_' | 'I' => {}
            other => return Err(format!("{p}: bad Pauli letter '{other}'")),
        }
    }
    if frags.is_empty() {
        return Err(format!("{p}: the identity is not a gate observable"));
    }
    Ok(if sign < 0.0 {
        format!("-1*{frags}")
    } else {
        frags
    })
}

/// Negate a signed stim PauliString: `"+XZ"` ↔ `"-XZ"`.
pub fn negate_stim_pauli(p: &str) -> Result<String, String> {
    match p.chars().next() {
        Some('+') => Ok(format!("-{}", &p[1..])),
        Some('-') => Ok(format!("+{}", &p[1..])),
        Some('X' | 'Y' | 'Z' | '_') => Ok(format!("-{p}")),
        _ => Err(format!("{p}: not a signed stim PauliString")),
    }
}

pub fn pauli_expectation(circuit: &CircuitIR, observable: &str) -> Result<f64, String> {
    let obs = Observable::parse(observable)?;
    let backend = PauliBackend::new();
    if backend.name() != "pauli" {
        return Err(format!(
            "backend name is {}, expected pauli",
            backend.name()
        ));
    }
    backend
        .expectation(circuit, &ParameterBinding::new(), &obs)
        .map_err(|e| e.to_string())
}

pub fn statevector_expectation(circuit: &CircuitIR, observable: &str) -> Result<f64, String> {
    let obs = Observable::parse(observable)?;
    let backend = omega_backend_statevector::StatevectorBackend::new();
    backend
        .expectation(circuit, &ParameterBinding::new(), &obs)
        .map_err(|e| e.to_string())
}

/// Sample, and return `(shots observed, outcome width)`.
///
/// `collapse` is the syndrome circuit's mode: its measures are
/// mid-circuit and the key is the creg, which is the path `aria-qec`
/// already runs. The brickwall has no measures; `collapse` is false and
/// the backend measures every qubit at the end.
pub fn pauli_sample(
    circuit: &CircuitIR,
    shots: u32,
    seed: u64,
    collapse: bool,
) -> Result<(u64, u32), String> {
    let backend = PauliBackend::new();
    let cfg = ExecConfig {
        shots: Some(shots),
        seed: Some(seed),
        mid_circuit_mode: if collapse {
            MidCircuitMode::Collapse
        } else {
            MidCircuitMode::Skip
        },
    };
    let result = backend
        .execute(circuit, &ParameterBinding::new(), &cfg)
        .map_err(|e| e.to_string())?;
    let counts = match result {
        ExecResult::Counts(c) => c,
        _ => return Err("pauli sample did not return counts".into()),
    };
    let observed: u64 = counts.values().map(|c| u64::from(*c)).sum();
    account_shots(u64::from(shots), observed)?;
    let width = counts.keys().next().map(|o| o.width()).unwrap_or(0);
    Ok((observed, width))
}

/// Timed wrappers. The value is retained so a release build cannot delete
/// the call.
pub fn time_pauli_expectation(circuit: &CircuitIR, observable: &str) -> Result<f64, String> {
    let start = std::time::Instant::now();
    let v = pauli_expectation(circuit, observable)?;
    let dt = start.elapsed().as_secs_f64();
    // Touch the value. On these circuits it is an integer; the gate has
    // already admitted the row, so a drift here is a bug, not a tolerance.
    let _ = exact_gate(v, v).map_err(|d| d.reason)?;
    Ok(dt)
}

pub fn time_pauli_sample(
    circuit: &CircuitIR,
    shots: u32,
    seed: u64,
    collapse: bool,
) -> Result<f64, String> {
    let start = std::time::Instant::now();
    let _ = pauli_sample(circuit, shots, seed, collapse)?;
    Ok(start.elapsed().as_secs_f64())
}

/// The d=5 rotated-surface syndrome extraction, one round.
pub struct QecMemory {
    /// Data qubits plus ancillas, measures included.
    pub circuit: CircuitIR,
    /// Logical Z, the memory observable: product of Z on the leftmost column.
    pub observable: String,
    pub qubits: u32,
    pub measures: u32,
}

pub fn qec_d5() -> Result<QecMemory, String> {
    use aria_qec::ecc::codes::{QECCode, SurfaceCode};
    use aria_qec::ecc::to_omega_core_ir;

    let code = SurfaceCode::new(5);
    let circuit = to_omega_core_ir(&code.syndrome_circuit());
    let observable: String = code.logical_z().iter().map(|q| format!("Z{q}")).collect();
    let measures = circuit
        .ops
        .iter()
        .filter(|op| op.gate == GateKind::Measure)
        .count() as u32;
    if circuit.num_qubits != 49 {
        return Err(format!(
            "d=5 syndrome circuit has {} qubits, expected 25 data + 24 ancilla",
            circuit.num_qubits
        ));
    }
    if observable != "Z0Z5Z10Z15Z20" {
        return Err(format!("logical Z is {observable}, expected Z0Z5Z10Z15Z20"));
    }
    if measures != 24 {
        return Err(format!(
            "d=5 syndrome circuit has {measures} measures, expected 24"
        ));
    }
    Ok(QecMemory {
        qubits: circuit.num_qubits,
        circuit,
        observable,
        measures,
    })
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(sha256_hex(&bytes))
}

/// Smallest invocation of our expectation path: one qubit, no gates, `<Z>`.
pub fn floor_pauli_expectation() -> Result<f64, String> {
    let circuit = CircuitIR::new(1, CircuitType::GateBased);
    time_pauli_expectation(&circuit, "Z0")
}

/// Smallest invocation of our sampling path: one shot of one-qubit H.
pub fn floor_pauli_sample() -> Result<f64, String> {
    let mut circuit = CircuitIR::new(1, CircuitType::GateBased);
    circuit.ops.push(gate(GateKind::H, &[0], None));
    time_pauli_sample(&circuit, 1, SEED, false)
}

#[cfg(test)]
mod pauli_conversion_tests {
    use super::*;
    use omega_core::executor::{Observable, PauliOp};

    #[test]
    fn a_signed_stim_string_round_trips_through_our_observable() {
        // The gate's discriminating observable arrives as a stim PauliString
        // and has to be measured on our side, so this conversion is the join
        // between the two arms. Getting the index convention wrong here would
        // measure a different operator and still look like agreement failure
        // rather than a bug, so it is pinned both ways.
        let o = stim_pauli_to_observable("+XZ_Y").unwrap();
        assert_eq!(o, "X0Z1Y3");
        let parsed = Observable::parse(&o).unwrap();
        assert_eq!(parsed.terms.len(), 1);
        assert_eq!(parsed.terms[0].0, 1.0);
        assert_eq!(
            parsed.terms[0].1,
            vec![(0, PauliOp::X), (1, PauliOp::Z), (3, PauliOp::Y)]
        );
        assert_eq!(observable_to_stim_pauli(&o, 4).unwrap(), "+XZ_Y");
    }

    #[test]
    fn the_sign_survives_into_the_coefficient() {
        let o = stim_pauli_to_observable("-ZZ").unwrap();
        assert_eq!(o, "-1*Z0Z1");
        let parsed = Observable::parse(&o).unwrap();
        assert_eq!(parsed.terms[0].0, -1.0);
        assert_eq!(parsed.terms[0].1, vec![(0, PauliOp::Z), (1, PauliOp::Z)]);
    }

    #[test]
    fn an_unsigned_string_is_read_as_positive() {
        assert_eq!(stim_pauli_to_observable("Z__").unwrap(), "Z0");
    }

    #[test]
    fn an_imaginary_sign_is_refused_rather_than_silently_dropped() {
        // Our Observable has no imaginary coefficient. Accepting `iXZ` would
        // measure XZ and lose the phase, which is the quiet-wrong-answer
        // shape this whole lane exists to catch.
        for p in ["iXZ", "-iXZ"] {
            let err = stim_pauli_to_observable(p).expect_err(p);
            assert!(err.contains("imaginary"), "{p}: {err}");
        }
    }

    #[test]
    fn the_identity_is_not_a_gate_observable() {
        // `<I> = 1` on every state, so it would be a degenerate gate entry
        // dressed up as a third observable.
        assert!(stim_pauli_to_observable("+____").is_err());
    }

    #[test]
    fn negation_flips_the_sign_in_both_directions() {
        assert_eq!(negate_stim_pauli("+XZ").unwrap(), "-XZ");
        assert_eq!(negate_stim_pauli("-XZ").unwrap(), "+XZ");
        assert_eq!(negate_stim_pauli("XZ").unwrap(), "-XZ");
        // Twice is the identity, which is the property the gate relies on:
        // the third observable must be exactly minus the second.
        let once = negate_stim_pauli("+XYZ").unwrap();
        assert_eq!(negate_stim_pauli(&once).unwrap(), "+XYZ");
    }
}
