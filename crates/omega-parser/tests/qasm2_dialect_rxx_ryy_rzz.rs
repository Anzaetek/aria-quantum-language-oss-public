// SPDX-License-Identifier: Apache-2.0
//! **`rxx` / `ryy` / `rzz` follow a named Qiskit dialect, and the operator is
//! Qiskit's.**
//!
//! None of the three is in `qelib1.inc`, so whether a *bare* one (no preceding
//! `gate` definition) is readable is a property of the reader, not of QASM.
//! Qiskit ships two readers that disagree, measured on 2.5.1:
//!
//! ```text
//!               strict `qasm2.loads`   legacy `from_qasm_str`
//!   bare rxx    reject                 accept
//!   bare rzz    reject                 accept
//!   bare ryy    reject                 reject      <- both refuse
//!   with a gate def                accept (both)
//! ```
//!
//! `Qasm2Dialect` makes that choice explicit; `Legacy` is the default because it
//! is the more permissive of the two readers Qiskit actually ships.
//!
//! The acceptance matrix is only half of it. A reader that accepts the name and
//! builds the *wrong operator* is worse than one that refuses, so the second
//! half of this file pins each decomposition to `exp(-i θ/2 P⊗P)` — Qiskit's
//! `RXXGate` / `RYYGate` / `RZZGate` convention, verified against Qiskit itself
//! at max|Δ| ≤ 5.551e-17 (see `tools/qiskit_xcheck/`).

use omega_core::circuit::{GateKind, ParamExpr};
use omega_parser::lower::{lower_to_ir_with_dialect, Qasm2Dialect};

fn bare(name: &str) -> String {
    format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\n{name}(0.7) q[0], q[1];\n")
}

/// The canonical definitions this workspace's emitter writes. Every Qiskit
/// reader accepts these, and so must every dialect here.
fn with_def(name: &str) -> String {
    let body = match name {
        "rxx" => "h q0; h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q0; h q1;",
        "ryy" => {
            "rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(param0) q1; cx q0,q1; \
                  rx(-pi/2) q0; rx(-pi/2) q1;"
        }
        "rzz" => "cx q0,q1; rz(param0) q1; cx q0,q1;",
        other => panic!("no canonical definition for {other}"),
    };
    format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
         gate {name}(param0) q0,q1 {{ {body} }}\nqreg q[2];\n{name}(0.7) q[0], q[1];\n"
    )
}

/// **The acceptance matrix is the contract.** Each cell is what the named
/// Qiskit reader does, so a change here is a change in which files this
/// workspace can interchange with.
#[test]
fn bare_acceptance_matches_the_named_qiskit_reader() {
    // (dialect, rxx, ryy, rzz)
    let matrix = [
        (Qasm2Dialect::Strict, false, false, false),
        (Qasm2Dialect::Legacy, true, false, true),
        (Qasm2Dialect::Lenient, true, true, true),
    ];
    for (dialect, rxx_ok, ryy_ok, rzz_ok) in matrix {
        for (name, want_ok) in [("rxx", rxx_ok), ("ryy", ryy_ok), ("rzz", rzz_ok)] {
            let got = lower_to_ir_with_dialect(&bare(name), dialect);
            assert_eq!(
                got.is_ok(),
                want_ok,
                "bare `{name}` in {dialect:?}: expected ok={want_ok}, got {got:?}"
            );
        }
    }
}

/// `Legacy` is the default, so the no-argument entry point must agree with it.
/// Pinning this separately stops the default from drifting silently.
#[test]
fn the_default_dialect_is_legacy() {
    assert_eq!(Qasm2Dialect::default(), Qasm2Dialect::Legacy);
    assert!(omega_parser::lower_to_ir(&bare("rxx")).is_ok());
    assert!(omega_parser::lower_to_ir(&bare("rzz")).is_ok());
    assert!(omega_parser::lower_to_ir(&bare("ryy")).is_err());
}

/// **A file that defines the gate is readable in every dialect** — that is the
/// whole reason the emitter writes definitions. If this regresses, this
/// workspace's own exported files stop round-tripping in `Strict`.
#[test]
fn a_carried_definition_is_read_in_every_dialect() {
    for dialect in [
        Qasm2Dialect::Strict,
        Qasm2Dialect::Legacy,
        Qasm2Dialect::Lenient,
    ] {
        for name in ["rxx", "ryy", "rzz"] {
            let ir = lower_to_ir_with_dialect(&with_def(name), dialect)
                .unwrap_or_else(|e| panic!("`{name}` with its own definition in {dialect:?}: {e}"));
            assert!(
                !ir.ops.is_empty(),
                "`{name}` with a definition in {dialect:?} produced no ops"
            );
        }
    }
}

/// The refusal must say *why* and *what to do*. It used to be the same bare
/// `unknown gate: ryy` a typo produces, which sends the reader hunting for a
/// misspelling that is not there.
#[test]
fn the_refusal_explains_itself() {
    let err = omega_parser::lower_to_ir(&bare("ryy")).expect_err("bare ryy is refused by default");
    for needle in ["ryy", "legacy", "qelib1", "gate ryy(param0)", "lenient"] {
        assert!(
            err.contains(needle),
            "refusal must mention `{needle}`; got:\n{err}"
        );
    }
    // And it must NOT read as an unknown identifier.
    assert!(
        !err.starts_with("unknown gate"),
        "refusal still reads as a typo report:\n{err}"
    );
}

/// The placeholder `GateKind::CX` that lets the dialect (not the name table)
/// decide must never reach the op list. A refused `ryy` that silently became a
/// bare `CX` would be a wrong operator, not an error — the failure mode this
/// guard exists for.
#[test]
fn a_refused_spelling_never_emits_a_placeholder_cx() {
    for dialect in [Qasm2Dialect::Strict, Qasm2Dialect::Legacy] {
        for name in ["rxx", "ryy", "rzz"] {
            if let Ok(ir) = lower_to_ir_with_dialect(&bare(name), dialect) {
                // Accepted: must be a real decomposition, not one lone CX.
                assert!(
                    ir.ops.len() > 1,
                    "`{name}` in {dialect:?} lowered to {} op(s) — that is the \
                     placeholder leaking, not a decomposition",
                    ir.ops.len()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The operator itself: exp(-i θ/2 P⊗P), which is Qiskit's convention.
// ---------------------------------------------------------------------------

type C = (f64, f64);

fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn cadd(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}

/// 4x4 dense matrix of a lowered 2-qubit circuit, by applying it to each of the
/// four basis states. Little-endian qubit 0 = least significant bit, matching
/// `omega-core`'s convention.
fn dense_4x4(ops: &[omega_core::circuit::GateOp]) -> [[C; 4]; 4] {
    let mut out = [[(0.0, 0.0); 4]; 4];
    for (col, item) in out.iter_mut().enumerate() {
        let mut psi = [(0.0, 0.0); 4];
        psi[col] = (1.0, 0.0);
        for op in ops {
            apply(&mut psi, op);
        }
        *item = psi;
    }
    // `out[col]` currently holds the image of basis `col`; transpose so
    // `m[row][col]` reads as a matrix element.
    let mut m = [[(0.0, 0.0); 4]; 4];
    for (col, image) in out.iter().enumerate() {
        for (row, amp) in image.iter().enumerate() {
            m[row][col] = *amp;
        }
    }
    m
}

fn theta_of(op: &omega_core::circuit::GateOp) -> f64 {
    match op.params.first() {
        Some(ParamExpr::Concrete(v)) => *v,
        other => panic!("expected a concrete angle, got {other:?}"),
    }
}

fn apply(psi: &mut [C; 4], op: &omega_core::circuit::GateOp) {
    let q = |i: usize| op.qubits[i].0 as usize;
    match &op.gate {
        GateKind::H => {
            let s = std::f64::consts::FRAC_1_SQRT_2;
            one_q(psi, q(0), [(s, 0.0), (s, 0.0), (s, 0.0), (-s, 0.0)]);
        }
        GateKind::Rx => {
            let (c, s) = ((theta_of(op) / 2.0).cos(), (theta_of(op) / 2.0).sin());
            one_q(psi, q(0), [(c, 0.0), (0.0, -s), (0.0, -s), (c, 0.0)]);
        }
        GateKind::Rz => {
            let h = theta_of(op) / 2.0;
            one_q(
                psi,
                q(0),
                [
                    (h.cos(), -h.sin()),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (h.cos(), h.sin()),
                ],
            );
        }
        GateKind::CX => {
            let (c, t) = (q(0), q(1));
            let mut next = *psi;
            for (i, slot) in next.iter_mut().enumerate() {
                let src = if (i >> c) & 1 == 1 { i ^ (1 << t) } else { i };
                *slot = psi[src];
            }
            *psi = next;
        }
        other => panic!("decomposition emitted an unexpected gate: {other:?}"),
    }
}

fn one_q(psi: &mut [C; 4], q: usize, m: [C; 4]) {
    let bit = 1usize << q;
    for i in 0..4 {
        if i & bit != 0 {
            continue;
        }
        let (a, b) = (psi[i], psi[i | bit]);
        psi[i] = cadd(cmul(m[0], a), cmul(m[1], b));
        psi[i | bit] = cadd(cmul(m[2], a), cmul(m[3], b));
    }
}

/// Reference `exp(-i θ/2 P⊗P)` for P in {X, Y, Z}, built from the closed form
/// `cos(θ/2)·I - i·sin(θ/2)·(P⊗P)` — valid because `(P⊗P)² = I`.
fn reference(pauli: char, theta: f64) -> [[C; 4]; 4] {
    let p: [[C; 2]; 2] = match pauli {
        'x' => [[(0.0, 0.0), (1.0, 0.0)], [(1.0, 0.0), (0.0, 0.0)]],
        'y' => [[(0.0, 0.0), (0.0, -1.0)], [(0.0, 1.0), (0.0, 0.0)]],
        'z' => [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (-1.0, 0.0)]],
        other => panic!("no such Pauli: {other}"),
    };
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let mut m = [[(0.0, 0.0); 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            // Little-endian: bit 0 is qubit 0. P⊗P is symmetric under the
            // factor order, so the kron order cannot hide an error here.
            let kron = cmul(p[(row >> 1) & 1][(col >> 1) & 1], p[row & 1][col & 1]);
            let mut v = cmul((0.0, -s), kron);
            if row == col {
                v = cadd(v, (c, 0.0));
            }
            m[row][col] = v;
        }
    }
    m
}

/// **Each spelling lowers to Qiskit's operator, not merely to something with the
/// right name.** `ryy` is the one that matters most: it is new here, and the
/// obvious `sdg`/`h` conjugation differs from Qiskit's by a global phase.
#[test]
fn every_spelling_lowers_to_the_qiskit_operator() {
    const THETA: f64 = 0.7;
    const TOL: f64 = 1e-12;
    for (name, pauli) in [("rxx", 'x'), ("ryy", 'y'), ("rzz", 'z')] {
        // Lenient so all three take the bare path under test; the definition
        // path is covered above.
        let ir = lower_to_ir_with_dialect(&bare(name), Qasm2Dialect::Lenient)
            .unwrap_or_else(|e| panic!("`{name}` must lower in Lenient: {e}"));
        let got = dense_4x4(&ir.ops);
        let want = reference(pauli, THETA);
        let mut worst = 0.0_f64;
        for r in 0..4 {
            for c in 0..4 {
                let d = ((got[r][c].0 - want[r][c].0).powi(2)
                    + (got[r][c].1 - want[r][c].1).powi(2))
                .sqrt();
                worst = worst.max(d);
            }
        }
        assert!(
            worst < TOL,
            "`{name}({THETA})` differs from exp(-i θ/2 {}⊗{}) by {worst:.3e}",
            pauli.to_ascii_uppercase(),
            pauli.to_ascii_uppercase()
        );
    }
}

/// The fixture must be able to tell the three apart — if `reference` returned
/// the same matrix for every Pauli, the test above would pass on a reader that
/// lowered all three identically.
#[test]
fn the_reference_distinguishes_the_three() {
    const THETA: f64 = 0.7;
    let (x, y, z) = (
        reference('x', THETA),
        reference('y', THETA),
        reference('z', THETA),
    );
    let differs = |a: &[[C; 4]; 4], b: &[[C; 4]; 4]| {
        (0..4).any(|r| {
            (0..4).any(|c| {
                (a[r][c].0 - b[r][c].0).abs() > 1e-9 || (a[r][c].1 - b[r][c].1).abs() > 1e-9
            })
        })
    };
    assert!(differs(&x, &y), "XX and YY references coincide");
    assert!(differs(&y, &z), "YY and ZZ references coincide");
    assert!(differs(&x, &z), "XX and ZZ references coincide");
}
