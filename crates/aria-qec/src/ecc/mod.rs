//! Single-patch quantum error-correcting codes + decoding.
//!
//! Code construction ([`codes`]) and MWPM decoding ([`mwpm`]) are pure Rust; the
//! [`run`] execution glue (added in the execution layer) runs syndrome
//! extraction on a selectable omega backend.

pub mod codes;
pub mod mwpm;
pub mod run;
pub mod transform;

pub use codes::*;
pub use mwpm::{decode_mwpm, decode_mwpm_correction, Correction};
pub use run::{
    bitflip_syndrome, decode_trial, monte_carlo, phaseflip_syndrome, run_counts, syndrome_bits,
    to_omega_core_ir, MonteCarlo, SimBackend, Trial,
};
pub use transform::*;
