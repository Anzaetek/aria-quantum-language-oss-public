//! Cirq bridge — placeholder.
//!
//! Reserved for a follow-up commit once #27 prioritises this backend.
//! Today `--features bridge-cirq` returns
//! [`BridgeError::Unavailable`].

use crate::{Backend, BridgeError, Counts, NoiseConfig};

#[cfg(not(feature = "bridge-cirq"))]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Cirq, "cirq"))
}

#[cfg(feature = "bridge-cirq")]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::Unavailable(
        Backend::Cirq,
        "Cirq bridge not yet implemented".to_string(),
    ))
}
