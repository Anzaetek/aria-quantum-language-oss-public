#![allow(clippy::needless_range_loop)]

pub(crate) mod adjoint;
// `gates` is promoted to pub so the Metal backend can reuse the same
// forward + derivative matrix builders without duplicating the math.
// Module re-exports `Gate1Q`, `Gate2Q` plus h/x/y/z/.../drx/dry/drz/...
// — the small set of pure-math functions we need cross-backend.
pub mod gates;
pub mod noise;
mod sim;

pub use noise::{NoiseModel, PauliRates};
pub use sim::NoisyStatevectorBackend;
pub use sim::StatevectorBackend;
