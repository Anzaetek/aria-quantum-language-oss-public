//! Single-patch quantum error-correcting codes + decoding.
//!
//! Code construction ([`codes`]) and MWPM decoding ([`mwpm`]) are pure Rust; the
//! [`run`] execution glue (added in the execution layer) runs syndrome
//! extraction on a selectable omega backend.

pub mod codes;
pub mod mwpm;
pub mod transform;

pub use codes::*;
pub use mwpm::{decode_mwpm, decode_mwpm_correction, Correction};
pub use transform::*;
