#![allow(clippy::needless_range_loop)]

mod sim;
mod stabilizer;

pub use sim::PauliBackend;
pub use stabilizer::pauli_mult_phase;
