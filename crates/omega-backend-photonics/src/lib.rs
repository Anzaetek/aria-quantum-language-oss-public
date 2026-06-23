#![allow(clippy::needless_range_loop)]

pub mod components;
pub mod decompose;
pub mod distinguishability;
pub mod fockstate;
pub mod mbqc;
pub mod permanent;
pub mod reservoir;
pub mod sim;
pub mod slos;
pub mod unitary_grad;

pub use distinguishability::{partial_probability, slos_partial};
pub use fockstate::FockKet;
pub use mbqc::{simulate_pattern_deterministic as simulate_mbqc_pattern, Pattern as MbqcPattern};
pub use reservoir::{MemristiveReservoir, ReservoirStep};
pub use sim::PhotonicsBackend;
