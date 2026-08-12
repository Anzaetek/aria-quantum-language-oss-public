//! QuEra `ppvm` bridge — an independent validator for `omega-backend-pauliprop`.
//!
//! ppvm propagates a magnitude-truncated `PauliSum` in the Heisenberg picture.
//! That is the **same algorithm family** as the in-tree
//! `omega-backend-pauliprop` (whose own docs cite arXiv:2505.21606 /
//! PauliPropagation.jl), so this bridge is not here to add a capability — it is
//! here so pauliprop can be checked against an implementation nobody in this
//! repository wrote.
//!
//! That distinction matters more than it sounds. This project has already
//! shipped a defect that every *internal* cross-backend agreement gate missed,
//! because each pair of backends happened to coincide in the basis being
//! checked. Two implementations of the same idea agreeing is evidence; one
//! implementation agreeing with itself is not.
//!
//! Runner: `python/ppvm_runner.py` via the `omega-bridge-ppvm-runner` wrapper.
//! A missing install is [`BridgeError::Unavailable`], never a hard error, so the
//! default `./ci.sh` stays green on a machine without ppvm built.
//!
//! ppvm is itself Rust (`ppvm-pauli-sum`), so a later revision could drop the
//! subprocess and validate in-process via a git Cargo dependency. The bridge is
//! the cheap uniform start, not the only possible shape.

#[cfg(feature = "bridge-ppvm")]
use crate::runner::{invoke_runner, run_subprocess, ParseAs, ParsedResponse, RunnerRequest, RunnerSpec};
use crate::{Backend, BridgeError, Counts, NoiseConfig};

#[cfg(not(feature = "bridge-ppvm"))]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Ppvm, "ppvm"))
}

#[cfg(not(feature = "bridge-ppvm"))]
pub fn expectation(
    _qasm: &str,
    _observables: &[crate::WireObservable],
    _trunc: Option<(f64, Option<usize>)>,
) -> Result<Vec<f64>, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Ppvm, "ppvm"))
}

/// Expectation values via ppvm's `PauliSum` — Heisenberg propagation, the
/// **same algorithm family** as `omega-backend-pauliprop`.
///
/// This is the reason ppvm is vendored at all, and until now nothing could use
/// it that way: the counts protocol reaches only ppvm's *other* engine
/// (`GeneralizedTableau`). Qiskit's `Statevector` anchors the lane on exact
/// ground truth via a **different** algorithm; this anchors `pauliprop`
/// against an independent implementation of the **same** one, which is what
/// catches a shared misunderstanding of the method rather than an arithmetic
/// slip.
///
/// `trunc` is `(min_abs_coeff, max_pauli_weight)`, mirroring
/// `PauliPropBackend::with_truncation`, so the *truncation behaviour* is
/// comparable and not just the exact result.
#[cfg(feature = "bridge-ppvm")]
pub fn expectation(
    qasm: &str,
    observables: &[crate::WireObservable],
    trunc: Option<(f64, Option<usize>)>,
) -> Result<Vec<f64>, BridgeError> {
    if observables.is_empty() {
        return Err(BridgeError::InvalidInput(
            "observables must not be empty".into(),
        ));
    }
    let mut payload = serde_json::json!({
        "mode": "expectation",
        "qasm": qasm,
        "observables": observables,
    });
    if let Some((min_abs, max_w)) = trunc {
        payload["min_abs_coeff"] = serde_json::json!(min_abs);
        if let Some(w) = max_w {
            payload["max_pauli_weight"] = serde_json::json!(w);
        }
    }
    let spec = RunnerSpec::new(Backend::Ppvm, "ppvm");
    match invoke_runner(&spec, &payload.to_string(), ParseAs::Values)? {
        ParsedResponse::Values(v) => {
            if v.len() != observables.len() {
                return Err(BridgeError::Backend(
                    Backend::Ppvm,
                    format!("returned {} values for {} observables", v.len(), observables.len()),
                ));
            }
            Ok(v)
        }
        other => Err(BridgeError::Backend(
            Backend::Ppvm,
            format!("unexpected response shape: {other:?}"),
        )),
    }
}

#[cfg(feature = "bridge-ppvm")]
pub fn run(qasm: &str, shots: u32, noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    let spec = RunnerSpec::new(Backend::Ppvm, "ppvm");
    let req = RunnerRequest {
        qasm,
        shots,
        noise,
        input_fock: None,
    };
    run_subprocess(&spec, &req)
}
