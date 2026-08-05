#![allow(clippy::needless_range_loop)]

pub(crate) mod adjoint;
// `gates` is promoted to pub so the Metal backend can reuse the same
// forward + derivative matrix builders without duplicating the math.
// Module re-exports `Gate1Q`, `Gate2Q` plus h/x/y/z/.../drx/dry/drz/...
// — the small set of pure-math functions we need cross-backend.
pub mod gates;
pub mod noise;
pub mod sim;

// The noise data model is shared across backends (see `omega_core::noise`);
// re-exported here for the callers that construct it alongside this backend.
pub use omega_core::noise::{Depolarizing, NoiseModel, PauliChannel, Rate, ReadoutError};
pub use sim::NoisyStatevectorBackend;
pub use sim::StatevectorBackend;
