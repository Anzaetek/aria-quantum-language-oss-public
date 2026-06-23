//! Qadence bridge — placeholder.
//!
//! Reserved for a follow-up commit once #27 prioritises this backend.
//! Today `--features bridge-qadence` returns
//! [`BridgeError::Unavailable`].

use crate::{Backend, BridgeError, Counts, NoiseConfig};

#[cfg(not(feature = "bridge-qadence"))]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::NotCompiled(Backend::Qadence, "qadence"))
}

#[cfg(feature = "bridge-qadence")]
pub fn run(_qasm: &str, _shots: u32, _noise: Option<&NoiseConfig>) -> Result<Counts, BridgeError> {
    Err(BridgeError::Unavailable(
        Backend::Qadence,
        "Qadence bridge not yet implemented".to_string(),
    ))
}
