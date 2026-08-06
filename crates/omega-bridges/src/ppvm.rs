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
use crate::runner::{run_subprocess, RunnerRequest, RunnerSpec};
use crate::{Backend, BridgeError, Counts, NoiseConfig};

#[cfg(not(feature = "bridge-ppvm"))]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Ppvm, "ppvm"))
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
