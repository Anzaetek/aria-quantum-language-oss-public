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
//! # The one deliberate divergence: `u`
//!
//! Qiskit emits `u(θ,φ,λ)`, which is **not** in qelib1, so its own strict loader
//! rejects it. We emit `u3(θ,φ,λ)`, which **is** in qelib1 and loads under both
//! parsers. That is a divergence in the direction of *more* portability, and it
//! is kept on purpose: matching Qiskit here would mean deliberately emitting a
//! spelling that fewer tools accept, for no gain. Qiskit's legacy loader reads
//! `u3` and produces the identical operator — which the Python half asserts.

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

/// `ryy` carries Qiskit's own `gate` definition, byte-for-byte.
///
/// Without it the export loads **nowhere** — measured: qiskit's strict loader,
/// qiskit's legacy loader, and `omega-parser` all refused a bare `ryy(θ)`.
#[test]
fn ryy_is_emitted_with_qiskits_own_gate_definition() {
    let text = emit_one(GateKind::RYY, vec![0.7], 2);
    assert!(
        text.contains(
            "gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; \
             cx q0,q1; sx q0; sx q1; }"
        ),
        "the preamble must reproduce `qasm2.dumps`'s definition verbatim — \
         an equivalent decomposition of our own choosing is not alignment:\n{text}"
    );
    // And the definition must actually be usable, not just present.
    let ir = omega_parser::lower_to_ir(&text).expect("must parse");
    assert_eq!(
        ir.ops.len(),
        7,
        "the ryy body should expand to sxdg,sxdg,cx,rz,cx,sx,sx:\n{text}"
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
