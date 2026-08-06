//! QuEra `tsim` bridge — ZX stabilizer-rank sampling.
//!
//! Unlike [`crate::ppvm`], this is a genuine new capability: stabilizer-rank
//! decomposition samples noisy Clifford+T circuits at scales the MPS backend
//! cannot reach, and nothing in this repository implements that method.
//!
//! **Known limit, stated rather than discovered later.** The bridge protocol
//! carries QASM2 in and counts out. tsim's distinctive value for QEC —
//! *detector* and *observable* sampling — has no expression in that shape, so
//! through this surface tsim is a plain noisy sampler. A detector-aware
//! extension is separate work, and only worth doing if QEC sampling actually
//! lands here. See `fixes/TSIM-PPVM.md`.
//!
//! Runner: `python/tsim_runner.py` via the `omega-bridge-tsim-runner` wrapper.
//! tsim pulls JAX, so it gets its own venv; a missing install is
//! [`BridgeError::Unavailable`], never a hard error.

#[cfg(feature = "bridge-tsim")]
use crate::runner::{run_subprocess, RunnerRequest, RunnerSpec};
use crate::{Backend, BridgeError, Counts, NoiseConfig};

#[cfg(not(feature = "bridge-tsim"))]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Tsim, "tsim"))
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
