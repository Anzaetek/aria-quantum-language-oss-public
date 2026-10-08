// SPDX-License-Identifier: Apache-2.0
//! Pauliprop lane of the emulator comparison (PLAN-EMULATOR-COMPARISON §7 E7).
//!
//! Ours is `omega_backend_pauliprop::PauliPropBackend` through its gated
//! `expectation_with_certificate`. The competitor is monoprop 0.9.0
//! (Algorithmiq, C++ core), promoted from oracle (§2). The oracle for every
//! value gate is `omega_backend_statevector` (dense f64), an independent
//! algorithm from both arms.
//!
//! Rows: the committed exact monoprop fixture circuits, re-run live and timed;
//! the 14q × depth-12 HEA exact on both sides; and the same HEA under the
//! matched-ACCURACY rule of §2 — the two engines' truncation knobs are not one
//! contract (`tests/monoprop_xcheck.rs` documents where they differ), so each
//! arm's fastest setting on one mapped grid that meets one pinned accuracy
//! is timed, and the grid is printed into the row.

pub mod measure;
pub mod monoprop;
pub mod workload;
