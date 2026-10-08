// SPDX-License-Identifier: Apache-2.0
//! Row schema and JSON-lines writer for the emulator comparison.
//!
//! The comparison is PLAN-EMULATOR-COMPARISON (STATUS §5 item 17): our
//! backends measured against **competitors**, not the oracles they are already
//! wired against. Its results live in one document,
//! `docs/EMULATOR-COMPARISON.md`, backed by one JSONL file per lane written
//! through [`RowWriter`].
//!
//! This crate is **one shape for every lane**. It exists because the shape was
//! about to be settled by whichever lane happened to produce a row first, and
//! two shapes for one comparison is the failure that makes the whole exercise
//! unreadable: rows that cannot be put in a table together, and a doc whose
//! prose cites fields that only half the rows have.
//!
//! # What this crate refuses
//!
//! A row that cannot name the code, the box, the file, or the evidence that the
//! path on its label is the path that ran **does not serialize**. That is the
//! one load-bearing property here, and it is a type-level one:
//!
//! ```
//! use omega_emu_compare::Witness;
//!
//! // A witness that was taken serializes as its value.
//! assert_eq!(serde_json::to_string(&Witness::present(3u8)).unwrap(), "3");
//! // One that was not does not serialize at all.
//! assert!(serde_json::to_string(&Witness::<u8>::absent()).is_err());
//! ```
//!
//! The reason is the A10 defect class this estate keeps relapsing into — *the
//! accelerated path never ran and the harness said so*, three receipts in one
//! day (`PLAN-OPEN-20260825.md:2556-2605`). §8 of the plan restates it as the
//! condition that would make the comparison worthless: "Any row without its
//! executed-path witness is not a row." A schema that merely has a field for
//! the witness does not enforce that; a schema where the absent case has no
//! JSON encoding does.
//!
//! A row that *does* serialize can still be wrong about itself, so [`check`]
//! re-derives every derived field from the `derivation` block the row carries
//! and refuses a row that disagrees with its own inputs.
//!
//! # What this crate is not
//!
//! Not a runner, not a plugin surface, not a config format. §8 caps the
//! comparison at "one results doc, ~46 rows total, one script per lane, no
//! runner framework, no plugin abstraction, no YAML" — because the standard
//! cuts both ways, and a large harness nobody trusts is as bad as an
//! unmeasured claim. Lanes are shell and Python scripts under
//! `tools/emu_compare*/`; they build a [`Row`] and hand it to [`RowWriter`].
//!
//! # The pinned workloads
//!
//! `tools/emu_compare/qasm/` holds the §3 circuit set, written by
//! `tools/emu_compare/gen_workloads.py` and hashed in
//! `tools/emu_compare/MANIFEST.json`. Every arm ingests the same file; the HEA
//! export in particular is the contract, not a per-arm reimplementation, and
//! `tests/workload_artifacts.rs` pins each exported file against the
//! `entangling_circuit` it claims to be.

#![deny(missing_docs)]

mod cap;
mod external;
mod load;
mod row;
mod witness;
mod writer;

pub use cap::{host_name, hostgate_cap, hostgate_cap_from, HOSTGATE_CAP_ENV};
pub use external::{
    external_cpu_sampled, external_cpu_since, external_void_reason, parse_ps_table, parse_ps_time,
    ps_own_cpu_s, ps_total_cpu_s, wait_quiet_external, CpuCensus, PsRow, QuietProbe,
    CENSUS_MIN_WINDOW_S, EXTERNAL_CPU_METHOD, EXTERNAL_CPU_METHOD_PS, PROBE_WINDOW_S,
};
pub use load::{
    host_cores, host_load_void_above, load_void_above, void_at_load, LOAD_VOID_ABOVE_LARGE_HOST,
    LOAD_VOID_ABOVE_SMALL_HOST, SMALL_HOST_MAX_CORES,
};
pub use row::{
    Arm, CapabilityRow, Classification, Derivation, Direction, ExternalCpu, FloorCheck,
    FloorVerdict, GateReference, GpuCensus, GpuHolder, Lane, LoadRecord, Precision, Quantity, Row,
    RowBody, SpeedArm, SpeedRow, Threads, Timing, Truncation, ValueGate, VoidRow, Workload,
    GATE_REL_F32, GATE_REL_F64, GPU_UTIL_VOID_PCT, MIN_REPEATS, OVERHEAD_FLOOR_MULTIPLE,
    OVERHEAD_RULE, TIE_RULE, WIN_RATIO_THRESHOLD,
};
pub use witness::{GitRev, PathWitness, PathWitnesses, Witness, ABSENT_WITNESS};
pub use writer::{check, RowWriter, WriteError};
