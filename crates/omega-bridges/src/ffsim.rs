//! ffsim (IBM) bridge — the **differential anchor for the fermionic
//! surface** (`PLAN-OPEN-20260825.md` §3c.0d / §3c.0e item 2).
//!
//! Implementation lives in `crates/omega-bridges/python/ffsim_runner.py`
//! (invoked via the wrapper script `omega-bridge-ffsim-runner`). See
//! `docs/BRIDGES.md` for venv setup.
//!
//! ## Why this bridge is not on the QASM2 wire
//!
//! Every other backend takes QASM2 in. `Rbs` — the gate this anchor exists
//! to check — has no QASM2 spelling, and lowering it to CX + rotations does
//! not help: ffsim accepts a circuit only when every gate is
//! Hamming-weight preserving *and recognisable as such*, so a decomposed
//! `Rbs` is either rejected or silently routed to a general simulator,
//! which is the thing we were trying not to compare against.
//!
//! So the circuit travels as a QPY blob from the pure-Rust writer
//! ([`crate::qpy::write_qpy_circuit_ir`]), base64 inside the JSON request.
//! `Rbs(θ)` arrives as `XXPlusYYGate(−2θ, π/2)` — the mapping
//! `qpy/write.rs::qiskit_params` documents with its numeric receipt and
//! `tests/qpy_rbs_vs_qiskit.rs` pins through Qiskit's own gate semantics.
//!
//! ## What the runner does with it
//!
//! `ffsim_runner.py` peels the leading X layer into a
//! `PrepareSlaterDeterminantSpinlessJW` (the occupation), rewrites
//! `CU3(0,0,λ)` to `CPhaseGate(λ)` (exact — qiskit 2.5.2 `Operator` delta
//! 0.0), **refuses** anything that is not Hamming-weight preserving with
//! `ffsim-unsupported-gate`, evolves in ffsim's number-conserving basis,
//! and only then expands to a 2^n statevector to evaluate the wire
//! observables through the SAME `SparsePauliOp` path `qiskit_runner.py`
//! uses. The two anchors therefore differ only in the evolution engine —
//! which is what makes their agreement evidence.
//!
//! Counts mode is `CannotExpress`, not `Unavailable`: ffsim is a
//! reference, not a sampler anyone picks.

#[cfg(feature = "bridge-ffsim")]
use crate::runner::{invoke_runner, ParseAs, ParsedResponse, RunnerSpec};
use crate::{Backend, BridgeError, Counts, NoiseConfig, WireObservable};
use omega_core::circuit::CircuitIR;

#[cfg(not(feature = "bridge-ffsim"))]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Ffsim, "ffsim"))
}

/// Always `CannotExpress`: a capability gap, not a missing install.
#[cfg(feature = "bridge-ffsim")]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::CannotExpress(
        Backend::Ffsim,
        "ffsim is an expectation-only reference for the fermionic surface; it has no \
         counts mode — use `omega_bridges::ffsim::expectation` with a CircuitIR"
            .into(),
    ))
}

/// The QASM2 entry point every other backend has. For ffsim it is a
/// capability gap by construction, and the message says where the real
/// entry point is.
pub fn expectation_qasm2(
    _qasm: &str,
    _observables: &[WireObservable],
) -> Result<Vec<f64>, BridgeError> {
    if !cfg!(feature = "bridge-ffsim") {
        return Err(BridgeError::NotCompiled(Backend::Ffsim, "ffsim"));
    }
    Err(BridgeError::CannotExpress(
        Backend::Ffsim,
        "ffsim takes a CircuitIR over QPY, not QASM2: `Rbs` has no QASM2 spelling and \
         decomposing it defeats the comparison — call `omega_bridges::ffsim::expectation`"
            .into(),
    ))
}

#[cfg(not(feature = "bridge-ffsim"))]
pub fn expectation(
    _ir: &CircuitIR,
    _observables: &[WireObservable],
) -> Result<Vec<f64>, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Ffsim, "ffsim"))
}

/// Exact `⟨O⟩` of the state `ir` prepares, evolved by ffsim in the
/// fixed-particle-number basis. Analytic — no shots — so compare at ~1e-12.
///
/// `observables` use the wire format documented on [`crate::PauliTerm`]:
/// dense, LSB-first Pauli strings of length `ir.num_qubits`.
///
/// Refusals are typed so the matrix cannot mistake them for coverage:
/// a gate the Rust QPY writer cannot spell, or a multi-bit condition it
/// cannot encode, is `CannotExpress` here rather than a panic inside the
/// writer; a gate ffsim will not evolve (anything not Hamming-weight
/// preserving, or an `X` outside the leading occupation layer) comes back
/// from the runner as `ffsim-unsupported-gate`, also `CannotExpress`.
#[cfg(feature = "bridge-ffsim")]
pub fn expectation(
    ir: &CircuitIR,
    observables: &[WireObservable],
) -> Result<Vec<f64>, BridgeError> {
    use base64::Engine;
    if observables.is_empty() {
        return Err(BridgeError::InvalidInput(
            "observables must not be empty".into(),
        ));
    }
    if ir.num_qubits == 0 {
        return Err(BridgeError::InvalidInput("circuit has no qubits".into()));
    }
    // What the writer would panic on, refused up front as the capability gap
    // it is. A panic across a bridge reads as a defect in the bridge.
    for op in &ir.ops {
        if crate::qpy::gate_kind_to_qiskit_name(&op.gate).is_none() {
            return Err(BridgeError::CannotExpress(
                Backend::Ffsim,
                format!("gate {:?} has no spelling in the Rust QPY writer", op.gate),
            ));
        }
        if let Some((start, num_bits, _)) = op.condition {
            if num_bits != 1 {
                return Err(BridgeError::CannotExpress(
                    Backend::Ffsim,
                    format!(
                        "condition on {num_bits} bits from {start}: the QPY writer encodes \
                         single-clbit conditions only (and ffsim refuses conditions anyway)"
                    ),
                ));
            }
        }
    }
    let qpy = crate::qpy::write_qpy_circuit_ir(ir);
    let payload = serde_json::json!({
        "mode": "expectation",
        "qpy_b64": base64::engine::general_purpose::STANDARD.encode(&qpy),
        "observables": observables,
    })
    .to_string();
    let spec = RunnerSpec::new(Backend::Ffsim, "ffsim");
    match invoke_runner(&spec, &payload, ParseAs::Values)? {
        ParsedResponse::Values(v) => {
            if v.len() != observables.len() {
                return Err(BridgeError::Backend(
                    Backend::Ffsim,
                    format!(
                        "runner returned {} values for {} observables",
                        v.len(),
                        observables.len()
                    ),
                ));
            }
            Ok(v)
        }
        other => Err(BridgeError::Backend(
            Backend::Ffsim,
            format!("unexpected response shape: {other:?}"),
        )),
    }
}
