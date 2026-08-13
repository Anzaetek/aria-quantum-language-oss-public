// SPDX-License-Identifier: Apache-2.0
//! Emits one QASM2 file per gate for `tools/qiskit_xcheck/qasm2_dialect.py`,
//! which checks that **Qiskit loads our QASM2 and agrees on what it means**.
//!
//! The Rust half deliberately does almost nothing: it writes the corpus and
//! asserts the things that need no Python (every gate emits; the file is
//! well-formed; our own parser reads it back). The claim that requires Qiskit —
//! *the operator is the same one Qiskit builds natively* — lives in the Python
//! harness and runs under `ARIA_QISKIT_XCHECK=1`, because it needs the venv.
//!
//! # Which Qiskit dialect
//!
//! `qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS`, not the strict qelib1 parser.
//! That is not a convenience: Qiskit's **own** `qasm2.dumps` output does not
//! survive its strict loader either. Measured on qiskit 2.5.1, round-tripping
//! Qiskit's own emission through `qasm2.loads`:
//!
//! ```text
//!   gate    qiskit strict   aria strict
//!   ryy     FAIL            FAIL
//!   rxx     FAIL            FAIL
//!   rzz     FAIL            FAIL
//!   cp      FAIL            FAIL
//!   sx      FAIL            FAIL
//!   p       FAIL            FAIL
//!   cswap   FAIL            FAIL
//!   crz     OK              OK
//!   ccx     OK              OK
//!   u       FAIL            OK      <- the one divergence, see below
//! ```
//!
//! So demanding strict conformance would demand something Qiskit does not do.
//! Aligning with "Qiskit's QASM2" means aligning with the dialect Qiskit
//! actually writes and reads.
//!
//! # We now aim HIGHER than qiskit's own output
//!
//! The table above was the starting point, not the target. Since qiskit cannot
//! reload its own emission, matching it byte-for-byte would inherit that defect.
//! Instead every spelling is emitted in a form the **strict** parser accepts,
//! which is a superset of what qiskit's legacy mode reads:
//!
//! ```text
//!   p      -> u1        (in qelib1; operator-identical, max|delta| = 0)
//!   cp     -> cu1       (in qelib1; operator-identical, max|delta| = 0)
//!   swap   -> gate def  (`swap` is NOT in qelib1 — measured, not assumed)
//!   cswap  -> gate def
//!   rxx    -> gate def
//!   rzz    -> gate def
//!   ryy    -> gate def, qelib1-only body (qiskit's uses `sxdg`, not in qelib1)
//!   u      -> u3        (qiskit emits `u`, which is not in qelib1)
//! ```
//!
//! Result: **24 of 25 gates load in `qasm2.loads`**, against 3 of 10 before.
//!
//! # The one that cannot be fixed: `sx`
//!
//! `sx` is not in qelib1 and **cannot be defined in it**. The natural body
//! `sdg a; h a; sdg a;` is `u3(pi/2, -pi/2, pi/2)`, which differs from `sx` by a
//! global phase `e^{i*pi/4}` — measured as max|delta| = 5.412e-01 against
//! qiskit's native `sx`. QASM 2.0 has no syntax for a global phase, so an exact
//! `sx` is not expressible. A bare `sx` (legacy-only, exactly what qiskit emits)
//! is the honest choice; emitting the u3 form would silently substitute a
//! different operator.

use aria_core::ast::nodes::*;
use aria_core::ast::qasm::to_qasm;

/// Distinct, generic parameters: no 0, no π/2, no repeats, so a swapped or
/// dropped argument cannot coincide with the right answer. These MUST match
/// `NATIVE` in `qasm2_dialect.py` — the Python side asserts every gate here has
/// a reference, so a mismatch fails rather than silently skipping.
fn corpus() -> Vec<(&'static str, GateKind, Vec<f64>, usize)> {
    vec![
        ("ryy", GateKind::RYY, vec![0.7], 2),
        ("rxx", GateKind::RXX, vec![0.7], 2),
        ("rzz", GateKind::RZZ, vec![0.7], 2),
        ("cp", GateKind::CP, vec![0.7], 2),
        ("crz", GateKind::CRz, vec![0.7], 2),
        ("sx", GateKind::SX, vec![], 1),
        ("u", GateKind::U, vec![0.3, 0.4, 0.5], 1),
        ("p", GateKind::P, vec![0.7], 1),
        ("ccx", GateKind::CCX, vec![], 3),
        ("cswap", GateKind::CSWAP, vec![], 3),
        ("cx", GateKind::CX, vec![], 2),
        ("cy", GateKind::CY, vec![], 2),
        ("cz", GateKind::CZ, vec![], 2),
        ("swap", GateKind::SWAP, vec![], 2),
        ("rx", GateKind::RX, vec![0.7], 1),
        ("ry", GateKind::RY, vec![0.7], 1),
        ("rz", GateKind::RZ, vec![0.7], 1),
        ("h", GateKind::H, vec![], 1),
        ("s", GateKind::S, vec![], 1),
        ("sdg", GateKind::Sdg, vec![], 1),
        ("t", GateKind::T, vec![], 1),
        ("tdg", GateKind::Tdg, vec![], 1),
        ("x", GateKind::X, vec![], 1),
        ("y", GateKind::Y, vec![], 1),
        ("z", GateKind::Z, vec![], 1),
    ]
}

fn emit_one(kind: GateKind, params: Vec<f64>, nq: usize) -> String {
    let mut c = Circuit::new("c");
    let q = c.qreg("q", nq);
    c.apply(GateDef::with_params(kind, params), q.to_vec());
    to_qasm(&c).expect("every gate in the corpus must be emittable")
}

/// Writes the corpus where the Python harness expects it, and asserts what can
/// be asserted without Qiskit.
#[test]
fn emit_the_qiskit_dialect_corpus() {
    let mut out = String::new();
    for (name, kind, params, nq) in corpus() {
        let text = emit_one(kind.clone(), params, nq);
        // Our own parser must read it back. This is not the Qiskit claim — it
        // is the cheap half, and it localises a failure to the emitter before
        // the Python harness even runs.
        let ir = omega_parser::lower_to_ir(&text)
            .unwrap_or_else(|e| panic!("{name}: our own parser cannot read it: {e}\n{text}"));
        assert!(
            !ir.ops.is_empty(),
            "{name}: parsed to zero ops — an empty circuit is not a round trip:\n{text}"
        );
        out.push_str(&format!("===={name}\n{text}"));
    }

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/qasm2_dialect_corpus.txt");
    std::fs::create_dir_all(path.parent().unwrap()).expect("create target dir");
    std::fs::write(&path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// `ryy` carries a **qelib1-only** `gate` definition.
///
/// This test previously pinned qiskit's definition byte-for-byte, on the
/// reasoning that copying what qiskit writes IS the alignment. Measuring
/// strict-loadability reversed it: qiskit's body uses `sxdg`, **`sxdg` is not in
/// qelib1**, and qiskit's own strict parser therefore rejects qiskit's own
/// output (`'sxdg' is not defined in this scope`).
///
/// The `rx(±pi/2)`-conjugated body uses only qelib1 gates, loads in BOTH qiskit
/// parsers, and is exact — max|delta| = 1.214e-16 against `RYYGate(0.7)`.
/// Matching the spelling would have cost the property that made matching worth
/// doing.
///
/// Without any definition the export loads NOWHERE: qiskit strict, qiskit
/// legacy and `omega-parser` all refused a bare `ryy(θ)`.
#[test]
fn ryy_is_emitted_with_a_qelib1_only_gate_definition() {
    let text = emit_one(GateKind::RYY, vec![0.7], 2);
    assert!(
        text.contains("gate ryy(param0) q0,q1 {"),
        "the preamble must define ryy:\n{text}"
    );
    assert!(
        !text.contains("sxdg") && !text.contains("sx "),
        "the body must avoid sx/sxdg — neither is in qelib1, which is exactly why \
         qiskit's own definition is not strict-loadable:\n{text}"
    );
    // The definition must be usable, not merely present.
    let ir = omega_parser::lower_to_ir(&text).expect("must parse");
    assert_eq!(
        ir.ops.len(),
        7,
        "the ryy body should expand to rx,rx,cx,rz,cx,rx,rx:\n{text}"
    );
}

/// The definition is emitted only when used — as `qasm2.dumps` does. Emitting
/// it unconditionally would put a `gate ryy` line in every file we produce,
/// which Qiskit does not do and which would make our output differ from its for
/// no reason.
#[test]
fn the_ryy_definition_is_not_emitted_when_unused() {
    let text = emit_one(GateKind::CX, vec![], 2);
    assert!(
        !text.contains("gate ryy"),
        "a circuit with no ryy must not carry its definition:\n{text}"
    );
}
