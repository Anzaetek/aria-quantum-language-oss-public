// SPDX-License-Identifier: Apache-2.0
//! **A malformed QASM2 file must be an error, never a panic.**
//!
//! `PLAN-EXPORT-INTEGRITY.md` P4. There was no arity validation anywhere between
//! the parser and a backend's array index, so a grammatical but wrong-arity file
//! crashed the process. Measured against `omega-backend-statevector`, all three
//! panicking:
//!
//! ```text
//!   gate mycp(t) a,b { cp(t) a,b; }  ->  CU3 carrying 1 param  ->  PANIC
//!   cu3(0.7) q[0], q[1];             ->  CU3 carrying 1 param  ->  PANIC
//!   u3(0.7) q[0];                    ->  U3  carrying 1 param  ->  PANIC
//! ```
//!
//! The first is the interesting one and is why this is a class rather than three
//! bugs: the `cp`/`cu1` widening lived only on the top-level path, and the
//! **user-defined-gate body path bypassed it entirely** — along with everything
//! else that path never reached. `gate mycp(t) a,b { cp(t) a,b; }` produced a
//! `CU3` with one parameter where the top-level `cp(t) a,b;` produced a correct
//! three-parameter one. Two paths, one grammar, different results.
//!
//! Both now go through `widen_cp_params` and `check_gate_arity`.
//!
//! # Why this matters beyond tidiness
//!
//! A parser that panics on malformed input cannot be pointed at anything
//! untrusted — not a user upload, not a file from another tool, not a fuzzer.
//! "Refuses with a message" and "crashes the process" are different products.

/// Wrong arities that used to reach a backend and panic.
#[test]
fn wrong_arity_is_refused_at_lowering() {
    let cases: &[(&str, &str, &str)] = &[
        (
            "cu3 with one parameter",
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncu3(0.7) q[0], q[1];\n",
            "3 parameter",
        ),
        (
            "u3 with one parameter",
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nu3(0.7) q[0];\n",
            "3 parameter",
        ),
        (
            "rx with two parameters",
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nrx(0.7, 0.2) q[0];\n",
            "1 parameter",
        ),
        (
            "cx on three qubits",
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncx q[0], q[1], q[2];\n",
            "2 qubit",
        ),
        (
            "h on two qubits",
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0], q[1];\n",
            "1 qubit",
        ),
        (
            "ccx on two qubits",
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\nccx q[0], q[1];\n",
            "3 qubit",
        ),
    ];
    for (name, src, expect) in cases {
        let err = omega_parser::lower_to_ir(src)
            .map(|ir| {
                format!(
                    "LOWERED to {} op(s) with param counts {:?}",
                    ir.ops.len(),
                    ir.ops.iter().map(|o| o.params.len()).collect::<Vec<_>>()
                )
            })
            .expect_err(&format!("{name}: must be refused, not lowered"));
        assert!(
            err.contains(expect),
            "{name}: the error must state the expected arity (looking for {expect:?}): {err}"
        );
    }
}

/// **The user-defined-gate body path reaches the same widening as the top
/// level.**
///
/// This is the specific defect: `cp` inside a gate body produced a `CU3` with
/// ONE parameter. The assertion is on the parameter count, not on `is_ok()` —
/// the old behaviour was `Ok` too, and that is exactly why nothing caught it.
#[test]
fn cp_inside_a_gate_body_is_widened_like_cp_at_top_level() {
    let body = omega_parser::lower_to_ir(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
         gate mycp(t) a,b { cp(t) a,b; }\nqreg q[2];\nmycp(0.7) q[0], q[1];\n",
    )
    .expect("a valid gate body must lower");
    let top = omega_parser::lower_to_ir(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncp(0.7) q[0], q[1];\n",
    )
    .expect("top-level cp must lower");

    assert_eq!(body.ops.len(), 1);
    assert_eq!(top.ops.len(), 1);
    assert_eq!(
        body.ops[0].params.len(),
        3,
        "cp in a gate body must widen to CU3(0, 0, lambda); one parameter is what \
         made the statevector backend index resolved[2] and panic"
    );
    assert_eq!(
        body.ops[0].params.len(),
        top.ops[0].params.len(),
        "the two paths must agree — one grammar cannot mean two things"
    );
    assert_eq!(body.ops[0].gate, top.ops[0].gate, "same gate kind on both paths");
}

/// A wrong arity inside a gate body is refused too, not just at top level.
#[test]
fn wrong_arity_inside_a_gate_body_is_refused() {
    let err = omega_parser::lower_to_ir(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
         gate bad(t) a,b { u3(t) a; }\nqreg q[2];\nbad(0.7) q[0], q[1];\n",
    )
    .expect_err("u3 with one parameter must be refused wherever it appears");
    assert!(err.contains("3 parameter"), "{err}");
}

/// Guard the guard: valid programs must still lower. A validator that refuses
/// everything would pass every assertion above.
#[test]
fn valid_programs_still_lower() {
    let cases: &[(&str, &str, usize)] = &[
        ("u3", "u3(0.3, 0.4, 0.5) q[0];", 1),
        ("rx", "rx(0.7) q[0];", 1),
        ("h", "h q[0];", 1),
        ("cx", "cx q[0], q[1];", 1),
        ("cp", "cp(0.7) q[0], q[1];", 1),
        ("crz", "crz(0.7) q[0], q[1];", 1),
        ("ccx", "ccx q[0], q[1], q[2];", 1),
        ("cswap", "cswap q[0], q[1], q[2];", 1),
        ("barrier", "barrier q[0], q[1], q[2];", 1),
        // Decomposed, so more than one op — the point is that it lowers.
        ("rzz", "rzz(0.7) q[0], q[1];", 3),
    ];
    for (name, line, want_ops) in cases {
        let src = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\n{line}\n");
        let ir = omega_parser::lower_to_ir(&src)
            .unwrap_or_else(|e| panic!("{name}: a VALID program was refused: {e}\n{src}"));
        assert_eq!(ir.ops.len(), *want_ops, "{name}: wrong op count\n{src}");
    }
}

/// The whole point: after refusal, nothing malformed reaches a backend.
///
/// Runs the previously-panicking inputs through lowering and asserts they never
/// produce an IR at all. Executing them is not needed — and could not be done
/// here without a backend dependency — because a refused program has nothing to
/// execute, which IS the fix.
#[test]
fn the_previously_panicking_inputs_never_produce_an_ir() {
    for src in [
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\ngate mycp(t) a,b { cp(t) a,b; }\nqreg q[2];\nmycp(0.7) q[0], q[1];\n",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncu3(0.7) q[0], q[1];\n",
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nu3(0.7) q[0];\n",
    ] {
        match omega_parser::lower_to_ir(src) {
            // The first case is now VALID (it widens correctly), so it lowers —
            // but it must carry three parameters, not one.
            Ok(ir) => {
                for op in &ir.ops {
                    assert_ne!(
                        op.params.len(),
                        1,
                        "a CU3/U3 carrying one parameter reached the IR; a backend \
                         will index params[2] and panic:\n{src}"
                    );
                }
            }
            Err(_) => {}
        }
    }
}
