// SPDX-License-Identifier: Apache-2.0
//! Refuse a mixed-radix state this backend should not hold, **before** a
//! single amplitude is allocated.
//!
//! Two gates, in the shape `omega-backend-statevector::capacity` and
//! `omega-backend-cv::capacity` both use:
//!
//! 1. A hard ceiling on the product dimension, independent of the host.
//!    This is a reference engine: past `2^24` amplitudes (256 MiB) it is no
//!    longer the calculation it exists for, and a machine with a terabyte
//!    should still refuse — "it fits" is not "it is the right engine".
//! 2. The host's available memory, so a run that would fit the ceiling but
//!    not the box refuses rather than driving it into swap. Overridable
//!    with `OMEGA_QUDITSV_OVERSUBSCRIBE=1` for someone who knows better.
//!
//! The product is computed in `u128` with checked arithmetic: `Π dᵢ` for
//! mixed dimensions has no closed form to reason about, and an overflow
//! that wrapped to a small number would admit exactly the run that must be
//! refused.

use omega_core::error::{OmegaError, Result};
use omega_core::hostmem;

/// `Complex64` — two `f64`.
pub const BYTES_PER_AMPLITUDE: u128 = 16;

/// Hard ceiling on `Π dᵢ`: `2^24` amplitudes, 256 MiB.
pub const MAX_PRODUCT_DIM: u128 = 1 << 24;

/// Set to `1` to skip the host-memory comparison (the ceiling still holds).
pub const OVERSUBSCRIBE_VAR: &str = "OMEGA_QUDITSV_OVERSUBSCRIBE";

/// `Π dᵢ`, or `None` on overflow (which is already a refusal).
pub fn product_dim(dims: &[u32]) -> Option<u128> {
    dims.iter()
        .try_fold(1u128, |acc, &d| acc.checked_mul(d as u128))
}

/// The refusal, or the product dimension as a `usize` ready to allocate.
pub fn check(dims: &[u32]) -> Result<usize> {
    let spell = || {
        dims.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("×")
    };
    let Some(dim) = product_dim(dims) else {
        return Err(OmegaError::Unsupported(format!(
            "quditsv: the product dimension {} overflows a 128-bit integer; this state \
             cannot be indexed, let alone allocated",
            spell()
        )));
    };
    if dim > MAX_PRODUCT_DIM {
        return Err(OmegaError::Unsupported(format!(
            "quditsv: the product dimension {} = {dim} amplitudes ({}) exceeds this \
             backend's hard ceiling of {MAX_PRODUCT_DIM} amplitudes ({}); it is a dense \
             reference engine for laptop-sized qudit circuits, refusing rather than \
             allocating. Fewer wires or smaller dimensions; the MPS lane is the one \
             that scales (PLAN-QUDIT.md Q3).",
            spell(),
            hostmem::human_bytes(dim * BYTES_PER_AMPLITUDE),
            hostmem::human_bytes(MAX_PRODUCT_DIM * BYTES_PER_AMPLITUDE),
        )));
    }
    if std::env::var(OVERSUBSCRIBE_VAR).is_ok_and(|v| v == "1") {
        return Ok(dim as usize);
    }
    if let Some(available) = hostmem::available_bytes() {
        let need = dim * BYTES_PER_AMPLITUDE;
        if need > available as u128 {
            return Err(OmegaError::Unsupported(format!(
                "quditsv: {} amplitudes for dimensions {} need {} but only {} is \
                 available on this host — refusing rather than driving it into swap. \
                 Set {OVERSUBSCRIBE_VAR}=1 if you know this host can take it.",
                dim,
                spell(),
                hostmem::human_bytes(need),
                hostmem::human_bytes(available as u128),
            )));
        }
    }
    Ok(dim as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_is_exact_and_checked() {
        assert_eq!(product_dim(&[]), Some(1));
        assert_eq!(product_dim(&[3, 2, 5]), Some(30));
        assert_eq!(product_dim(&[2; 24]), Some(1 << 24));
        assert_eq!(product_dim(&[u32::MAX; 5]), None);
    }

    #[test]
    fn the_ceiling_refuses_before_the_host_is_consulted() {
        // 3^16 = 43_046_721 > 2^24: refused by arithmetic, with the product
        // and the ceiling both in the message.
        let err = check(&[3; 16]).unwrap_err().to_string();
        assert!(err.contains("43046721"), "{err}");
        assert!(err.contains(&MAX_PRODUCT_DIM.to_string()), "{err}");
        assert!(err.contains("3×3×3"), "{err}");
        // Exactly at the ceiling is admitted by the ceiling.
        assert_eq!(
            check(&[2; 24]).map_err(|e| e.to_string()).ok(),
            Some(1 << 24)
        );
        assert_eq!(check(&[3, 2, 5]).unwrap(), 30);
    }
}
