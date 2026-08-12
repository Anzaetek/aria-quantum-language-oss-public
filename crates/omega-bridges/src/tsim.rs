//! QuEra `tsim` bridge — ZX stabilizer-rank sampling.
//!
//! Unlike [`crate::ppvm`], this is a genuine new capability: stabilizer-rank
//! decomposition samples noisy Clifford+T circuits at scales the MPS backend
//! cannot reach, and nothing in this repository implements that method.
//!
//! **tsim is a strong QEC tool; the limitation is this surface, not the tool.**
//! It carries the full Stim v1.13 instruction set with its noise channels and —
//! the part that matters for QEC — detectors and observables, which is what
//! magic-state distillation/cultivation and surface-code studies need.
//!
//! The bridge protocol, however, carries QASM2 in and counts out. Detector and
//! observable records have no expression in that shape, so *through this door*
//! tsim arrives as a plain noisy sampler with its most useful QEC feature
//! unreachable.
//!
//! So for QEC investigation, drive tsim directly through its own API rather than
//! through here; extending this protocol with a detector/observable response is
//! scoped work worth doing only if QEC sampling becomes recurring in this repo.
//! See `docs/BRIDGES.md` and `fixes/TSIM-PPVM.md`.
//!
//! Runner: `python/tsim_runner.py` via the `omega-bridge-tsim-runner` wrapper.
//! tsim pulls JAX, so it gets its own venv; a missing install is
//! [`BridgeError::Unavailable`], never a hard error.

#[cfg(feature = "bridge-tsim")]
use crate::runner::{invoke_runner, run_subprocess, ParseAs, ParsedResponse, RunnerRequest, RunnerSpec};
use crate::{Backend, BridgeError, Counts, NoiseConfig};

#[cfg(not(feature = "bridge-tsim"))]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Tsim, "tsim"))
}

#[cfg(not(feature = "bridge-tsim"))]
pub fn expectation(
    _qasm: &str,
    _observables: &[crate::WireObservable],
) -> Result<Vec<f64>, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Tsim, "tsim"))
}

/// Exact Clifford expectation values via **plain Stim's tableau**.
///
/// The same-algorithm anchor for the in-tree `omega-backend-pauli` stabilizer
/// backend. Stim is *the* reference stabilizer simulator, so a disagreement is
/// a defect by construction rather than a modelling difference.
///
/// **This uses Stim, not tsim.** tsim's ZX stabilizer-rank engine exists to
/// handle non-Clifford circuits, which is irrelevant here: what an anchor
/// needs is exactness, and `peek_observable_expectation` returns exact
/// integers (+1 / −1 / 0). It is routed through this bridge only because
/// `bloqade-tsim` already pulls Stim into the venv.
///
/// **Clifford-only, enforced on the emitted Stim text.** Our lowering writes
/// `S[T] 0` and `I[R_Z(theta=…)] 0` for non-Clifford gates, and plain Stim
/// parses the bracket as an annotation on the base gate — applying S instead
/// of T, identity instead of a rotation — **without complaint**. The runner
/// refuses any lowering containing a `[` tag rather than returning a
/// confidently wrong reference, and reports `tsim-not-supported`, which the
/// step-2 taxonomy maps to `CannotExpress`.
#[cfg(feature = "bridge-tsim")]
pub fn expectation(
    qasm: &str,
    observables: &[crate::WireObservable],
) -> Result<Vec<f64>, BridgeError> {
    if observables.is_empty() {
        return Err(BridgeError::InvalidInput(
            "observables must not be empty".into(),
        ));
    }
    let payload = serde_json::json!({
        "mode": "expectation",
        "qasm": qasm,
        "observables": observables,
    })
    .to_string();
    let spec = RunnerSpec::new(Backend::Tsim, "tsim");
    match invoke_runner(&spec, &payload, ParseAs::Values)? {
        ParsedResponse::Values(v) => {
            if v.len() != observables.len() {
                return Err(BridgeError::Backend(
                    Backend::Tsim,
                    format!("returned {} values for {} observables", v.len(), observables.len()),
                ));
            }
            Ok(v)
        }
        other => Err(BridgeError::Backend(
            Backend::Tsim,
            format!("unexpected response shape: {other:?}"),
        )),
    }
}

#[cfg(feature = "bridge-tsim")]
pub fn run(qasm: &str, shots: u32, noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    let spec = RunnerSpec::new(Backend::Tsim, "tsim");
    let req = RunnerRequest {
        qasm,
        shots,
        noise,
        input_fock: None,
    };
    run_subprocess(&spec, &req)
}
