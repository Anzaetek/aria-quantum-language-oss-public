//! Pauli-propagation backend (prototype) — a Heisenberg-picture tree simulator.
//!
//! A fourth omega simulation scheme alongside statevector / stabilizer / MPS,
//! following the Pauli-propagation framework (arXiv:2505.21606,
//! `PauliPropagation.jl`). It evolves an **observable** backward through the
//! circuit as a truncated sum (tree) of weighted Pauli strings and reads off an
//! **expectation value** — see `PAULI_PROPAGATION_PLAN.md` (item 17) in the
//! `quantum` repo.
//!
//! Cost model: exact and width-unbounded for **Clifford** circuits (a single
//! Pauli never branches), a tunable approximation for non-Clifford gates.

pub mod pauli;
pub mod sim;

pub use pauli::{PauliKey, PauliSum, Weighted};
pub use sim::{pack_bits, unpack_bits, BranchHook, PauliPropBackend};
