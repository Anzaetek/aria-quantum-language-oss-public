// SPDX-License-Identifier: Apache-2.0
//! Stabilizer-rank simulation (`stabrank`) — Schrödinger-picture companion
//! to `omega-backend-majoranaprop`, whose cost axis is the count of
//! non-Clifford gates rather than the propagated term count.
//!
//! This crate is PLAN-MAJORANA-STIM.md phases **S0** (the kernel), **S1**
//! (the sum over Cliffords) and **S2** (truncation, and the bound).
//!
//! S0 is [`ChForm`]: a stabilizer state in the CH form of Bravyi, Browne,
//! Calpin, Campbell, Gosset and Howard (Quantum 3, 181 (2019)), with Clifford
//! gate application, the amplitude `⟨x|φ⟩` **with its exact global phase**,
//! and the phase-sensitive pairwise quantities `⟨φ|φ′⟩` and `⟨φ|P|φ′⟩`.
//!
//! The phase is the point. A stabilizer tableau — including the one in
//! `omega-backend-pauli` — represents the state by its stabilizer group and
//! so loses the overall phase, which is harmless for a single state and
//! fatal for a decomposition `|ψ⟩ = Σᵢ cᵢ|φᵢ⟩`, whose readout
//! `Σᵢⱼ c̄ᵢcⱼ⟨φᵢ|O|φⱼ⟩` is built entirely out of relative phases between
//! different `|φᵢ⟩`. Nothing here is "a tableau with a phase bolted on":
//! the state is stored as `ω U_C U_H |s⟩` and every update rule is derived
//! for that form.
//!
//! S1 is [`StabilizerSum`]: that decomposition, and the χ² readout that uses
//! it. One mechanism covers every non-Clifford gate — a Pauli rotation
//! `exp(−iφ/2 P) = cos(φ/2)·I − i sin(φ/2)·P` splits one branch into two, both
//! Clifford — so `T`, `T†` and a generic `Rz(θ)` differ only in `φ`, and
//! `χ = 2^t` is the naive stabilizer rank of plan §1.1.
//!
//! S2 is the knob that makes `χ` affordable and the certificate that makes it
//! honest: [`StabilizerSum::truncate`] drops the lightest branches under
//! `coeff_min` and `max_chi` and banks their L1 mass `m`, and
//! [`StabRankCertificate`] reports that `m` **and** the derived bound
//! `R·m·(2+m)` on `|Δ⟨O⟩|`, whose derivation — and whose difference from
//! majoranaprop's additive `dropped_mass` — is that struct's doc comment. The
//! truncated state is not renormalised, because the bound is derived for the
//! raw one, and a run whose bound excludes nothing is refused rather than
//! returned.
//!
//! S3 adds no engine code at all — it is the CLI door (`--backend stabrank`,
//! both observable spellings through the existing F2 dispatch, and a JSON
//! certificate block), plus the one thing a reader of two certificates needs
//! told in words: `CROSS-LANE-INTERVALS.md`, beside this crate's `Cargo.toml`.
//! Its subject is the field above. majoranaprop's `dropped_mass` **is** an
//! error bound and this engine's `state_dropped_mass` is **not** one, so the
//! two do not compose under any arithmetic; what two runs of one job license
//! is **intersecting their certified intervals**, which is sound, tighter than
//! either, and empty only if one of the two engines has an unsound bound.
//!
//! The S0 mutation ledger (`tests/mutation_phase_table.rs`) records why the
//! two phases needed separate test suites: a defect that changes only a
//! *global* phase leaves every single-state expectation and the entire Stim
//! leg green, because a tableau has no global phase and `⟨P⟩` on one state is
//! invariant under one. In a sum over Cliffords those phases stop cancelling
//! — every cross term `⟨φᵢ|O|φⱼ⟩` is a relative phase — so S1's suite is
//! where such a defect becomes visible, and `tests/mutation_s1_ledger.rs`
//! measures that it does.
//!
//! Stim is not a dependency and never will be (plan §1.3): it has no
//! stabilizer-state inner product, no exact global phase, and cannot execute
//! `T` at all. It is the independent *oracle* for the Clifford fragment and
//! nothing else.
//!
//! [`StabRankBackend`] is the expectation-value door. It refuses `execute()`,
//! naming the backend to use instead, exactly as majoranaprop does, and
//! returns a [`StabRankCertificate`] alongside the value.

// The kernel is dense F_2 linear algebra over `n × n` bit matrices, indexed
// by wire on both axes. `for q in 0..n` with several arrays indexed by `q`
// is the readable form here and is what `omega-backend-pauli` does for the
// same reason.
#![allow(clippy::needless_range_loop)]

mod backend;
mod chform;
mod sum;

pub use backend::{SeedBasis, StabRankBackend, StabRankCertificate, DEFAULT_MAX_BRANCHES};
pub use chform::{BraProgram, ChForm};
pub use sum::{observable_l1_norm, PauliSite, StabilizerSum};
