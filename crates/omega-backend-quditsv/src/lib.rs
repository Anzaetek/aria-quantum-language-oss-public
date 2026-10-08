// SPDX-License-Identifier: Apache-2.0
//! Exact dense simulation of **mixed-radix** circuits — `PLAN-QUDIT.md` Q2.
//!
//! `Π dᵢ` amplitudes for wires of local dimension `dᵢ`, read from
//! `CircuitIR::wire_dims()` (Q1). Wire 0 is the least significant digit,
//! exactly as qubit 0 is the least significant bit in every qubit engine
//! here, so on an all-`d = 2` circuit this state is the dense qubit
//! statevector index for index — which is what makes the `d = 2`
//! differential a test of the addressing rather than of a convention.
//!
//! This is the in-tree **reference** for qudits, in the role
//! `omega-backend-sector` plays for particle-conserving circuits: small,
//! exact, deliberately laptop-sized (a hard ceiling on the product
//! dimension, refused **before** allocating), no GPU, no gradients, no
//! noise. Q3's MPS lane is measured against it.
//!
//! What it evolves: the generalised one-wire gates `H` → Fourier `F_d`,
//! `X` → shift `X_d`, `Z` → clock `Z_d` (mqt.qudits' semantics, verified in
//! Q0 — each is the qubit gate at `d = 2`); `rxy(i, j, θ, φ)` on a level
//! pair; `csum` (the generalised CX); and every qubit gate on `d = 2`
//! wires, embedded from the statevector backend's own matrix table. A qubit
//! gate on a `d > 2` wire is refused by name — `CX` on a qutrit is not a
//! thing, `csum` is.
//!
//! Certificate: **exact**, bound zero. Nothing is discarded, ever; the
//! capacity gate is what makes that honest.

pub mod capacity;
pub mod gates;
pub mod sim;

pub use capacity::{product_dim, MAX_PRODUCT_DIM};
pub use sim::{QuditSvBackend, QuditSvCertificate};
