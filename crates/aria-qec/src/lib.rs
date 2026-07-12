//! # aria-qec — transversal low-overhead quantum error correction
//!
//! A port of the `quantum` toolkit's `ecc/` + `logical/` layer onto the Aria /
//! omega backends. It provides:
//!
//! * [`ecc`] — single-patch QEC codes (repetition, Steane `[[7,1,3]]`, rotated
//!   surface `[[d²,1,d]]`) with a minimum-weight (MWPM) decoder and a selectable
//!   simulator backend (statevector / stabilizer / MPS / Pauli-propagation).
//! * [`logical`] — a transversal logical-circuit layer: multi-patch geometry,
//!   transversal Clifford + non-Clifford gadgets, hardware-flavored noise models
//!   (neutral-atom / trapped-ion), 15-to-1 magic-state distillation, code-capacity
//!   memory Monte-Carlo, effective-logical-channel extraction, and encoded
//!   QFT / QPE / Grover.
//!
//! It builds on `aria_core::ast` (circuit builder) + `aria_core::backends::omega`
//! (lowering) + `omega_core` (executor) + the `omega-backend-*` crates.
pub mod ecc;
pub mod logical;
