//! Bloqade (QuEra Aquila) bridge.
//!
//! Implementation lives in `crates/omega-bridges/python/bloqade_runner.py`
//! (invoked via the wrapper script `omega-bridge-bloqade-runner`).
//! See `docs/BRIDGES.md` for venv setup.
//!
//! Today the runner accepts QASM2 (gate mode, routed through
//! [`bloqade-circuit`](https://github.com/QuEraComputing/bloqade-circuit)
//! to the Aquila gate set + a state-vector simulator). The analog AHS
//! mode (time-dependent Rabi/detuning waveforms over an atom-position
//! lattice, via the [`bloqade`](https://github.com/QuEraComputing/bloqade)
//! package) is the next step — for now the wrapper returns
//! `BridgeError::Unavailable` for `mode == "ahs"` requests so the
//! dispatch surface stays stable while the input shape is finalised.

#[cfg(feature = "bridge-bloqade")]
use crate::runner::{run_subprocess, RunnerRequest, RunnerSpec};
use crate::{Backend, BridgeError, Counts, NoiseConfig};

#[cfg(not(feature = "bridge-bloqade"))]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Bloqade, "bloqade"))
}

#[cfg(feature = "bridge-bloqade")]
pub fn run(qasm: &str, shots: u32, noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    let spec = RunnerSpec::new(Backend::Bloqade, "bloqade");
    let req = RunnerRequest {
        qasm,
        shots,
        noise,
        input_fock: None,
    };
    run_subprocess(&spec, &req)
}
