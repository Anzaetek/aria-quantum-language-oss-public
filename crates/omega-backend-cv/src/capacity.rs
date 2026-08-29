// SPDX-License-Identifier: Apache-2.0
//! **What a multi-mode CV state costs, and the one way to get it catastrophically
//! wrong.**
//!
//! A CV mode has an infinite Fock ladder, truncated at `cutoff`. `n` modes
//! combine as a tensor product, so the state is `cutoff^n` amplitudes. At the
//! envelope this backend targets — `n_modes <= 6`, `cutoff <= 8` — that is
//! `8^6 = 262_144` amplitudes, **4 MiB** of `Complex64`. Laptop-sized.
//!
//! # The trap this module exists to prevent
//!
//! Truncated displacement and squeezing are **not unitary**: cutting the ladder
//! clips amplitude that should flow in from higher levels. The single-mode code
//! handles that by computing in a larger space and cutting back —
//! `const PAD: usize = 96; let dim = cutoff + PAD;` in `FockState::displace`.
//! For one mode that is 104 amplitudes instead of 8, and free.
//!
//! The obvious multi-mode generalisation is to build the padded operator and
//! tensor it up. **Do not.** The padding is per mode, so it enters the product
//! as a power:
//!
//! ```text
//!   product basis          8^6   =           262_144 amplitudes =  4.00 MiB
//!   PADDED product      (8+96)^6 = 1_265_319_018_496 amplitudes = 18.41 TiB
//!                                                       ratio: 4.83e6 x
//! ```
//!
//! That is the allocation that takes a machine down, and it is reachable by
//! writing the *natural* extension of code that is correct in one mode. The
//! padded dimension must stay an **inner summation index**, never a factor in
//! the product — see [`mode_local_temp_amplitudes`].
//!
//! # Why this module has no host-memory probe
//!
//! This crate ships with **one** runtime dependency (`num-complex`) on purpose:
//! it is meant to be embeddable, and `Cargo.toml` says so. Reaching for
//! `omega_core::hostmem` here would put a dependency in the embed path that no
//! embedder can opt out of. So the arithmetic is pure and the *caller* supplies
//! a byte budget it obtained however it likes. A backend that knows about the
//! host passes `hostmem::available_bytes()`; an embedder passes its own number.

use crate::CvError;

/// `Complex64` — two `f64`.
pub const BYTES_PER_AMPLITUDE: usize = 16;

/// Hard ceiling on the product dimension, independent of how much memory the
/// host happens to have.
///
/// `2^26` amplitudes is 1 GiB of `Complex64`. Above this a CV run is not a
/// laptop-sized calculation and the truncation error almost certainly dominates
/// anything the extra dimension buys, so refusing is more useful than trying:
/// the envelope this backend was designed for is four megabytes.
///
/// A hard cap as well as a budget check, because a machine with a terabyte of
/// RAM should still refuse — "it fits" is not the same as "it is the right
/// calculation".
pub const MAX_PRODUCT_DIM: usize = 1 << 26;

/// Amplitudes in the product basis: `cutoff^n_modes`.
///
/// `None` on overflow rather than a wrapped value. `checked_pow` matters here:
/// `20^15` wraps a `u64` and would report a *small* requirement for an
/// impossible state — a refusal that admits.
pub fn product_dim(cutoff: usize, n_modes: usize) -> Option<usize> {
    if n_modes == 0 {
        return Some(0);
    }
    cutoff.checked_pow(n_modes as u32)
}

/// Bytes for the product-basis state vector, or `None` on overflow.
pub fn product_bytes(cutoff: usize, n_modes: usize) -> Option<usize> {
    product_dim(cutoff, n_modes)?.checked_mul(BYTES_PER_AMPLITUDE)
}

/// Amplitudes a **mode-local** application of a padded operator needs.
///
/// This is the whole point of the module. A single-mode operator touches one
/// mode's index, so it contracts a `cutoff x padded_dim` rectangle against that
/// one axis while every other axis is carried along unchanged. The worst
/// intermediate is `cutoff^(n-1) * padded_dim` — at `cutoff = 8`, `n = 6`,
/// `padded_dim = 104` that is 3_407_872 amplitudes, **52 MiB**, against 18.41
/// TiB for the padded product.
///
/// Compare with [`padded_product_amplitudes`], which exists only to be shown to
/// be absurd.
pub fn mode_local_temp_amplitudes(
    cutoff: usize,
    n_modes: usize,
    padded_dim: usize,
) -> Option<usize> {
    if n_modes == 0 {
        return Some(0);
    }
    product_dim(cutoff, n_modes - 1)?.checked_mul(padded_dim)
}

/// The size of the mistake: `(cutoff + pad)^n_modes`.
///
/// **Never allocate this.** It is here so the cost can be named in a message and
/// pinned in a test, rather than discovered by a machine falling over. Returns
/// `None` on overflow, which for realistic pads and mode counts it frequently
/// does — itself a useful signal about the scale involved.
pub fn padded_product_amplitudes(cutoff: usize, n_modes: usize, pad: usize) -> Option<usize> {
    product_dim(cutoff.checked_add(pad)?, n_modes)
}

/// Refuse a multi-mode state that cannot or should not be built.
///
/// `budget_bytes` is what the caller is willing to spend; `None` skips the
/// memory check and applies only the hard ceiling. See the module docs for why
/// this crate does not probe the host itself.
pub fn check(cutoff: usize, n_modes: usize, budget_bytes: Option<usize>) -> Result<(), CvError> {
    if cutoff == 0 {
        return Err(CvError::CutoffTooSmall { cutoff, needed: 1 });
    }
    let Some(dim) = product_dim(cutoff, n_modes) else {
        return Err(CvError::Unrepresentable {
            what: "cutoff^n_modes overflows a usize: this state cannot be indexed, \
                   let alone allocated",
        });
    };
    if dim > MAX_PRODUCT_DIM {
        return Err(CvError::Unrepresentable {
            what: "the product basis exceeds this backend's hard ceiling of 2^26 \
                   amplitudes (1 GiB). Reduce the cutoff or the mode count — the \
                   designed envelope is n_modes <= 6, cutoff <= 8, which is 4 MiB",
        });
    }
    if let (Some(budget), Some(bytes)) = (budget_bytes, product_bytes(cutoff, n_modes)) {
        if bytes > budget {
            return Err(CvError::Unrepresentable {
                what: "the product basis does not fit the memory budget the caller \
                       allowed. Reduce the cutoff or the mode count",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The designed envelope is small, and saying so in a test keeps it honest.
    #[test]
    fn the_designed_envelope_is_four_megabytes() {
        assert_eq!(product_dim(8, 6), Some(262_144));
        assert_eq!(product_bytes(8, 6), Some(4 * 1024 * 1024));
        assert!(check(8, 6, Some(16 * 1024 * 1024)).is_ok());
    }

    /// **The trap, pinned.** Lifting the padded operator into the product basis
    /// is 4.8 million times the state it is supposed to act on.
    #[test]
    fn the_padded_product_is_absurd_and_mode_local_is_not() {
        let padded = padded_product_amplitudes(8, 6, 96).expect("104^6 fits a usize");
        assert_eq!(padded, 1_265_319_018_496);
        // 18.41 TiB.
        assert_eq!(padded * BYTES_PER_AMPLITUDE, 20_245_104_295_936);

        let local = mode_local_temp_amplitudes(8, 6, 8 + 96).expect("8^5 * 104");
        assert_eq!(local, 3_407_872); // 52 MiB
        assert!(
            padded / local > 300_000,
            "if these were comparable the whole constraint would be noise"
        );
    }

    /// A padded product that overflows must report `None`, not a wrapped value.
    /// A wrapped size looks SMALL, so it would pass a budget check and then
    /// allocate — a refusal that admits is worse than no refusal.
    #[test]
    fn overflow_is_none_not_a_wrapped_small_number() {
        assert_eq!(product_dim(usize::MAX, 2), None);
        assert_eq!(padded_product_amplitudes(1000, 40, 96), None);
        assert!(check(1000, 40, None).is_err());
    }

    /// The hard ceiling applies even when the host has the memory: "it fits" is
    /// not "it is the right calculation".
    #[test]
    fn the_hard_ceiling_applies_regardless_of_budget() {
        let huge = Some(usize::MAX);
        assert!(
            check(64, 6, huge).is_err(),
            "64^6 is 6.9e10 amplitudes — refused on the ceiling, not the budget"
        );
    }

    /// A budget smaller than the state refuses, and the same shape passes with a
    /// larger one — so the check is about the budget rather than the shape.
    #[test]
    fn the_budget_is_load_bearing() {
        assert!(
            check(8, 6, Some(1024)).is_err(),
            "4 MiB does not fit in 1 KiB"
        );
        assert!(check(8, 6, Some(8 * 1024 * 1024)).is_ok());
    }

    /// Degenerate inputs do not panic.
    #[test]
    fn degenerate_shapes_are_errors_not_panics() {
        assert!(
            check(0, 4, None).is_err(),
            "a zero cutoff represents nothing"
        );
        assert_eq!(product_dim(8, 0), Some(0));
        assert_eq!(mode_local_temp_amplitudes(8, 0, 104), Some(0));
    }
}
