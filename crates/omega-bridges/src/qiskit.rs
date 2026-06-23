//! Qiskit bridge — run a QASM2 source through Qiskit + qiskit-aer.
//!
//! Implementation lives in `crates/omega-bridges/python/qiskit_runner.py`
//! (invoked via the wrapper script `omega-bridge-qiskit-runner`). The
//! Rust side is a thin transport that ships a JSON request over stdin
//! and parses the JSON response on stdout — keeping the workspace
//! Python-free. See `docs/BRIDGES.md` for the operator install steps.

#[cfg(feature = "bridge-qiskit")]
use crate::runner::{
    invoke_runner, run_subprocess, ParseAs, ParsedResponse, RunnerRequest, RunnerSpec,
};
use crate::{Backend, BridgeError, Counts, NoiseConfig};

#[cfg(not(feature = "bridge-qiskit"))]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Qiskit, "qiskit"))
}

#[cfg(feature = "bridge-qiskit")]
pub fn run(qasm: &str, shots: u32, noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    let spec = RunnerSpec::new(Backend::Qiskit, "qiskit");
    let req = RunnerRequest {
        qasm,
        shots,
        noise,
        input_fock: None,
    };
    run_subprocess(&spec, &req)
}

#[cfg(not(feature = "bridge-qiskit"))]
pub fn qpy_to_qasm2(_qpy_bytes: &[u8]) -> Result<String, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Qiskit, "qiskit"))
}

/// Decode a QPY blob to QASM2 by routing through the Qiskit
/// subprocess. Lets omega-parser take a `.qpy` file by leaning on
/// Qiskit's own `qiskit.qpy.load` for the actual deserialisation —
/// version compatibility stays Qiskit's responsibility, no
/// thousand-line pure-Rust QPY parser required for the bring-up.
///
/// The runner rejects QPY blobs containing more than one circuit so
/// the caller doesn't silently get only the first.
#[cfg(feature = "bridge-qiskit")]
pub fn qpy_to_qasm2(qpy_bytes: &[u8]) -> Result<String, BridgeError> {
    use base64::Engine;
    if qpy_bytes.len() < 6 || &qpy_bytes[..6] != b"QISKIT" {
        return Err(BridgeError::InvalidInput(
            "input is not a QPY blob (magic bytes missing)".into(),
        ));
    }
    let qpy_b64 = base64::engine::general_purpose::STANDARD.encode(qpy_bytes);
    let payload = serde_json::json!({
        "mode": "qpy_to_qasm2",
        "qpy_b64": qpy_b64,
    })
    .to_string();
    let spec = RunnerSpec::new(Backend::Qiskit, "qiskit");
    match invoke_runner(&spec, &payload, ParseAs::Qasm2String)? {
        ParsedResponse::Qasm2(s) => Ok(s),
        other => Err(BridgeError::Backend(
            Backend::Qiskit,
            format!("internal: qpy_to_qasm2 received unexpected response: {other:?}"),
        )),
    }
}

#[cfg(not(feature = "bridge-qiskit"))]
pub fn qasm2_to_qpy(_qasm: &str) -> Result<Vec<u8>, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Qiskit, "qiskit"))
}

/// Encode a QASM2 source as a QPY blob via the Qiskit subprocess.
/// Symmetric to [`qpy_to_qasm2`]: the runner uses `qasm2.loads` to
/// build a `QuantumCircuit` and `qpy.dump` to serialise it. The
/// returned bytes start with the `b"QISKIT"` magic header — the
/// same shape omega's pure-Rust QPY reader consumes — so a caller
/// can round-trip `omega-parsed QASM → QPY → CircuitIR` to validate
/// the writer, or hand the QPY blob to downstream Qiskit-only
/// tooling.
#[cfg(feature = "bridge-qiskit")]
pub fn qasm2_to_qpy(qasm: &str) -> Result<Vec<u8>, BridgeError> {
    if qasm.trim().is_empty() {
        return Err(BridgeError::InvalidInput(
            "qasm2_to_qpy: source must be non-empty".into(),
        ));
    }
    let payload = serde_json::json!({
        "mode": "qasm2_to_qpy",
        "qasm": qasm,
    })
    .to_string();
    let spec = RunnerSpec::new(Backend::Qiskit, "qiskit");
    match invoke_runner(&spec, &payload, ParseAs::QpyBytes)? {
        ParsedResponse::Qpy(b) => Ok(b),
        other => Err(BridgeError::Backend(
            Backend::Qiskit,
            format!("internal: qasm2_to_qpy received unexpected response: {other:?}"),
        )),
    }
}
