// SPDX-License-Identifier: Apache-2.0
//! **Every gate `aria-core` can emit as QASM2 must be readable by
//! `omega-parser` — enumerated exhaustively, not sampled.**
//!
//! This is the guard for a whole class of defect rather than one instance. Two
//! gate tables in two crates drift apart, and the symptom is silent: the export
//! succeeds, the file is well-formed, and only a re-import fails — often in some
//! other tool, months later. Found instances, every one by a hand diff and none
//! by a test:
//!
//! | spelling | found | state |
//! |---|---|---|
//! | `cp` | hand diff | fixed — widened to `CU3` |
//! | `rxx` | hand diff | fixed — decomposed in the parser |
//! | `rzz` | hand diff | fixed — decomposed in the parser |
//! | `ryy` | hand diff | **still unreadable, deliberately — see below** |
//!
//! # Why an exhaustive `match` and not a source scrape
//!
//! An earlier draft of this guard read the two crates' *source* and diffed the
//! string literals. That version could not see anything the tables did not
//! spell — a wrong arity, a parameter dropped in the widening, a gate emitted
//! from a different arm. Worse, it silently stops working the moment either
//! table changes shape, which is precisely when a guard is needed.
//!
//! The `match` in `spec()` below has no `_` arm. Adding a `GateKind` variant
//! **fails to compile** until someone states what QASM2 does with it. That is
//! the mechanism; the assertions are secondary.

use aria_core::ast::nodes::*;
use aria_core::ast::qasm::to_qasm;

/// What a gate needs to be built, and what we expect of its QASM2 round trip.
enum Spec {
    /// Emitted and readable. Carries (param count, qubit count).
    RoundTrips(usize, usize),
    /// Emitted, but no consumer can read it back. The string is the reason, and
    /// it must be a reason about the *ecosystem*, not about our convenience.
    EmittedButUnreadable(usize, usize, &'static str),
    /// Not part of the QASM2 lane at all.
    NotQasm2,
}

/// **Exhaustive on purpose — do not add a `_` arm.**
fn spec(kind: &GateKind) -> Spec {
    use GateKind::*;
    match kind {
        // Single-qubit, no parameters.
        I | X | Y | Z | H | S | Sdg | T | Tdg | SX => Spec::RoundTrips(0, 1),
        // Parametric single-qubit.
        RX | RY | RZ | P => Spec::RoundTrips(1, 1),
        U => Spec::RoundTrips(3, 1),
        // Two-qubit, no parameters.
        CX | CY | CZ | SWAP => Spec::RoundTrips(0, 2),
        // Parametric two-qubit.
        CP | CRz => Spec::RoundTrips(1, 2),
        // Decomposed by `omega-parser` into cx/rz (+ h conjugators) rather than
        // becoming IR primitives — interchange spellings, not new capabilities.
        RXX | RZZ => Spec::RoundTrips(1, 2),
        RYY => Spec::EmittedButUnreadable(
            1,
            2,
            "qiskit 2.5.1 rejects `ryy` in BOTH the strict `qasm2.loads` and the \
             legacy `from_qasm_str` — it is not in qelib1. Teaching omega-parser \
             to read it would fix the round trip for us ALONE while every other \
             toolchain still could not load the file: strictly worse than the \
             gap, because it would look fixed. The real fix is to stop EMITTING \
             it (export the rx(pi/2)-conjugated rzz, which every consumer reads) \
             — a change to the emitter, tracked separately.",
        ),
        RBS => Spec::NotQasm2, // Givens rotation; no qelib1 spelling.
        // Three-qubit.
        CCX | CSWAP => Spec::RoundTrips(0, 3),
        // Structural / non-unitary: covered by their own round-trip tests
        // (`qasm_conditional_export.rs`, `crz_reset_spellable.rs`), not here —
        // they are not gate applications with a parameter list.
        Barrier | Reset | Measure => Spec::NotQasm2,
        // Photonic: the OPTICQASM lane, guarded by
        // `opticqasm_readable_by_parser.rs` and `opticqasm_reader_agreement.rs`.
        BeamSplitter | PhaseShifter | Squeezing | Displacement | Kerr => Spec::NotQasm2,
    }
}

/// Every variant, listed once. Kept beside `spec()` so the compiler's
/// exhaustiveness check and this list are read together.
const ALL: &[GateKind] = &[
    GateKind::I, GateKind::X, GateKind::Y, GateKind::Z, GateKind::H,
    GateKind::S, GateKind::Sdg, GateKind::T, GateKind::Tdg, GateKind::SX,
    GateKind::RX, GateKind::RY, GateKind::RZ, GateKind::P, GateKind::U,
    GateKind::CX, GateKind::CY, GateKind::CZ, GateKind::SWAP,
    GateKind::RXX, GateKind::RYY, GateKind::RZZ, GateKind::CP, GateKind::CRz,
    GateKind::RBS, GateKind::CCX, GateKind::CSWAP,
    GateKind::Barrier, GateKind::Reset, GateKind::Measure,
    GateKind::BeamSplitter, GateKind::PhaseShifter,
    GateKind::Squeezing, GateKind::Displacement, GateKind::Kerr,
];

/// Distinct, generic parameter values: no coincidence (0, π/2, equal values)
/// can make a swapped or dropped parameter invisible.
const PARAMS: [f64; 3] = [0.37, 0.61, 0.83];

fn emit(kind: &GateKind, n_params: usize, n_qubits: usize) -> Result<String, String> {
    let mut c = Circuit::new("c");
    let q = c.qreg("q", n_qubits);
    c.apply(
        GateDef::with_params(kind.clone(), PARAMS[..n_params].to_vec()),
        q.to_vec(),
    );
    to_qasm(&c)
}

#[test]
fn every_qasm2_gate_aria_core_emits_is_read_back_by_omega_parser() {
    let mut broken = Vec::new();
    let mut checked = 0usize;

    for kind in ALL {
        let (np, nq) = match spec(kind) {
            Spec::RoundTrips(np, nq) => (np, nq),
            Spec::EmittedButUnreadable(..) | Spec::NotQasm2 => continue,
        };
        let text = match emit(kind, np, nq) {
            Ok(t) => t,
            Err(e) => {
                broken.push(format!("{kind:?}: declared as round-tripping but NOT EMITTED: {e}"));
                continue;
            }
        };
        match omega_parser::lower_to_ir(&text) {
            Ok(ir) if ir.ops.is_empty() => broken.push(format!(
                "{kind:?}: parsed to ZERO ops — an empty circuit is not a round trip:\n{text}"
            )),
            Ok(_) => checked += 1,
            Err(e) => broken.push(format!("{kind:?}: emitted but UNREADABLE: {e}\n{text}")),
        }
    }

    assert!(
        broken.is_empty(),
        "aria-core emits QASM2 that omega-parser cannot read:\n{}\n\n\
         Fix by teaching `omega-parser` the spelling (a widening in \
         `lower_gate_app` if arities differ, as `cp` does, or a decomposition, \
         as `rxx`/`rzz` do) — or by removing it from the emitter if no consumer \
         accepts it.",
        broken.join("\n")
    );
    assert!(
        checked >= 20,
        "only {checked} gates were actually round-tripped; the enumeration has \
         drifted and this guard is no longer covering the table"
    );
}

/// The exemption list is exactly one entry, and it stays visible.
///
/// If `ryy` ever becomes readable — because the emitter switched to the
/// conjugated `rzz`, or because qelib1 grew it — this test fails and the entry
/// moves to `RoundTrips`. That is the correct way to find out, rather than an
/// exemption quietly outliving its reason.
#[test]
fn the_only_unreadable_export_is_ryy_and_the_reason_still_holds() {
    let mut exempt = Vec::new();
    for kind in ALL {
        if let Spec::EmittedButUnreadable(np, nq, why) = spec(kind) {
            exempt.push(format!("{kind:?}"));
            assert!(
                why.len() > 80,
                "{kind:?}: an exemption needs a stated reason about the ecosystem, \
                 not a placeholder"
            );
            let text = emit(kind, np, nq).expect("declared as emitted");
            assert!(
                omega_parser::lower_to_ir(&text).is_err(),
                "{kind:?} is now READABLE — move it to Spec::RoundTrips and delete \
                 the exemption:\n{text}"
            );
        }
    }
    assert_eq!(
        exempt,
        vec!["RYY"],
        "the set of knowingly-unreadable exports changed; each one is a file this \
         workspace can write and nothing can read, so the list must not grow \
         quietly"
    );
}
