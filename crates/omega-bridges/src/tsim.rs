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
