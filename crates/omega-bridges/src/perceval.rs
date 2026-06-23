//! Perceval (Quandela) bridge.
//!
//! Implementation lives in `crates/omega-bridges/python/perceval_runner.py`
//! (invoked via the wrapper script `omega-bridge-perceval-runner`).
//! See `docs/BRIDGES.md` for venv setup.
//!
//! The runner accepts both QASM2 (gate-based, dual-rail-encoded via
//! `QiskitConverter`) and OPTICQASM (native photonic — `ps`, `bs_rx`,
//! `bs_ry` over `photon q[N]`). Auto-detection happens on the Python
//! side from the source header.

#[cfg(feature = "bridge-perceval")]
use crate::runner::{run_subprocess, RunnerRequest, RunnerSpec};
use crate::{Backend, BridgeError, Counts, NoiseConfig};

#[cfg(not(feature = "bridge-perceval"))]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Perceval, "perceval"))
}

#[cfg(feature = "bridge-perceval")]
pub fn run(qasm: &str, shots: u32, noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    let spec = RunnerSpec::new(Backend::Perceval, "perceval");
    let req = RunnerRequest {
        qasm,
        shots,
        noise,
        input_fock: None,
    };
    run_subprocess(&spec, &req)
}

#[cfg(not(feature = "bridge-perceval"))]
pub fn run_opticqasm(
    _source: &str,
    _shots: u32,
    _input_fock: Option<&[u32]>,
    _noise: Option<&NoiseConfig>,
) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Perceval, "perceval"))
}

/// Run a native OPTICQASM source through Perceval. Output counts are
/// keyed by Fock-state strings (comma-separated, e.g. `"1,0,1,0"`)
/// rather than the qubit bit-strings used by gate-based runners — the
/// natural output for a photonic simulator.
#[cfg(feature = "bridge-perceval")]
pub fn run_opticqasm(
    source: &str,
    shots: u32,
    input_fock: Option<&[u32]>,
    noise: Option<&NoiseConfig>,
) -> Result<Counts, BridgeError> {
    let spec = RunnerSpec::new(Backend::Perceval, "perceval");
    let req = RunnerRequest {
        qasm: source,
        shots,
        noise,
        input_fock,
    };
    run_subprocess(&spec, &req)
}
