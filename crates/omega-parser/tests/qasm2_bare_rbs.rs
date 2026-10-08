// SPDX-License-Identifier: Apache-2.0
//! **A bare `rbs(θ) a,b` is readable in the `lenient` dialect only, and lowers
//! to ONE native `GateKind::Rbs` — never to its `h/cz/ry` expansion.**
//!
//! `rbs` is this workspace's Givens / reconfigurable-beam-splitter rotation:
//! the Hamming-weight-preserving hopping primitive the `sector` backend admits
//! natively. It is not in `qelib1.inc` and no Qiskit reader knows the name, so
//! it sits in the same "we can read it, nobody else can" bucket as a bare
//! `ryy` — `Qasm2Dialect::Lenient`, opt-in, never the default.
//!
//! Why it must stay one op: the emitter's decomposition (`h; h; cz; ry(-θ);
//! ry(θ); cz; h; h`) contains `h`, which does not conserve particle number, so
//! a reader that expanded `rbs` would hand the sector backend a circuit it is
//! obliged to REFUSE. The one spelling that engine exists for would be the one
//! it could never run.

use omega_core::circuit::{GateKind, ParamExpr};
use omega_parser::lower::{lower_to_ir_with_dialect, Qasm2Dialect};

fn bare_rbs() -> String {
    "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nrbs(0.7) q[0], q[1];\n".to_string()
}

fn rbs_in_body() -> String {
    "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\
     gate hop(t) a,b { rbs(t) a,b; }\n\
     qreg q[2];\nhop(0.7) q[0], q[1];\n"
        .to_string()
}

#[test]
fn lenient_lowers_bare_rbs_to_one_native_op_with_the_angle() {
    let ir = lower_to_ir_with_dialect(&bare_rbs(), Qasm2Dialect::Lenient)
        .expect("lenient must read a bare rbs");
    assert_eq!(ir.ops.len(), 1, "rbs must not be decomposed: {:?}", ir.ops);
    let op = &ir.ops[0];
    assert_eq!(op.gate, GateKind::Rbs);
    assert_eq!(op.qubits.len(), 2);
    assert_eq!(op.params.len(), 1);
    match &op.params[0] {
        ParamExpr::Concrete(v) => assert!((v - 0.7).abs() < 1e-15, "angle mangled: {v}"),
        other => panic!("expected a concrete angle, got {other:?}"),
    }
}

#[test]
fn legacy_and_strict_refuse_bare_rbs_with_the_dialect_message() {
    for dialect in [Qasm2Dialect::Legacy, Qasm2Dialect::Strict] {
        let err = lower_to_ir_with_dialect(&bare_rbs(), dialect)
            .expect_err("a bare rbs outside lenient must be refused");
        assert!(
            err.contains("`rbs` is not defined in this file"),
            "must say undefined-in-file, not `unknown gate`: {err}"
        );
        assert!(
            err.contains("native hopping gate"),
            "must say Qiskit has no such name, so the user does not go looking: {err}"
        );
        assert!(err.contains("lenient"), "must name the way out: {err}");
        assert!(!err.contains("unknown gate"), "{err}");
    }
}

/// A `gate` body calling `rbs` follows the same rule as a top-level `rbs`:
/// inlined natively under lenient, refused (with the same message) elsewhere.
#[test]
fn rbs_inside_a_gate_body_follows_the_dialect() {
    let ir = lower_to_ir_with_dialect(&rbs_in_body(), Qasm2Dialect::Lenient)
        .expect("lenient must inline a body rbs");
    assert_eq!(ir.ops.len(), 1);
    assert_eq!(ir.ops[0].gate, GateKind::Rbs);

    let err = lower_to_ir_with_dialect(&rbs_in_body(), Qasm2Dialect::Legacy)
        .expect_err("legacy must refuse a body rbs");
    assert!(err.contains("`rbs` is not defined in this file"), "{err}");
}

/// The existing three are untouched by the addition.
#[test]
fn rxx_rzz_still_read_in_legacy_and_ryy_still_does_not() {
    let src = |g: &str| {
        format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\n{g}(0.7) q[0], q[1];\n")
    };
    assert!(lower_to_ir_with_dialect(&src("rxx"), Qasm2Dialect::Legacy).is_ok());
    assert!(lower_to_ir_with_dialect(&src("rzz"), Qasm2Dialect::Legacy).is_ok());
    assert!(lower_to_ir_with_dialect(&src("ryy"), Qasm2Dialect::Legacy).is_err());
}
