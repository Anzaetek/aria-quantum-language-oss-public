// SPDX-License-Identifier: Apache-2.0
//! Majorana propagation (MP) — Heisenberg back-propagation of an observable
//! written in the **Majorana monomial** basis, for Jordan–Wigner-encoded
//! fermionic circuits. Companion to `omega-backend-pauliprop`, which does the
//! same thing in the Pauli basis; the two engines answer the same question
//! (`⟨0|U† O U|0⟩`) with the same certificate contract, and differ only in
//! the basis they truncate in.
//!
//! Why a second basis: a fermionic Givens rotation `Rbs(θ)` on modes `p,q`
//! is TWO Majorana rotations (`iγ_{2p}γ_{2q}` and `iγ_{2p+1}γ_{2q+1}`) and
//! each maps a Majorana monomial to at most two monomials whose **length**
//! (Majorana count) does not grow. In the Pauli basis the same gate's
//! generator `Y_pX_q − X_pY_q` grows Pauli *weight* by the whole JW string
//! between `p` and `q`. So truncating by Majorana length is the natural
//! low-rank cut for fermionic dynamics, and the dropped mass is a genuine
//! bound on `|Δ⟨O⟩|` for exactly the reason it is in pauliprop: every
//! Hermitian Majorana monomial squares to the identity, so `|⟨M⟩| ≤ 1`.
//!
//! [`majorana`] is the monomial algebra (keys, products, commutation, Fock
//! readout), verified exhaustively against dense matrices for small mode
//! counts in `tests/majorana_oracle.rs`. [`engine`] is the propagation on
//! top of it, pinned against the statevector backend and exact pauliprop in
//! `tests/engine_vs_statevector.rs`.

pub mod engine;
pub mod majorana;

pub use engine::{
    observable_l1_norm, theorem1_mse_ratio, theorem1_mse_ratio_f, MajoranaPropBackend,
    MajoranaPropCertificate, MajoranaSum, DEFAULT_MAX_TERMS,
};
pub use majorana::MajoranaKey;
