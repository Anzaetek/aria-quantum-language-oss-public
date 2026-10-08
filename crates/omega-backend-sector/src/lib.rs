// SPDX-License-Identifier: Apache-2.0
//! Exact simulation of number-conserving qubit circuits in one Hamming-weight
//! sector — `PLAN-OPEN-20260825.md` §3c T2.
//!
//! `C(n, k)` amplitudes instead of `2^n`. Under Jordan–Wigner the weight-`k`
//! strings are the `k`-particle Fock sector, so a fermionic circuit built from
//! `omega_core::fermion` (`givens`, `cphase`) on an occupation layer of `X`s
//! runs here exactly, with the same gate matrices as the dense backend and
//! the same Jordan–Wigner observables. See [`sim::SectorBackend`].

pub mod basis;
pub mod sim;

pub use basis::{binomial, Sector};
pub use sim::{SectorBackend, MAX_DENSE_READOUT_QUBITS, MAX_DIM};
