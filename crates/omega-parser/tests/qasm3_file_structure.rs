// SPDX-License-Identifier: Apache-2.0
//! **The reader accepts the file structure the OpenQASM 3 specification uses.**
//!
//! Three rules, each of which rejected real files from the specification's own
//! example corpus (`third_party/openqasm-examples/`, vendored byte-exact):
//!
//! | rule | who it rejected |
//! |---|---|
//! | header mandatory | **20 of 21** programs — they carry no `OPENQASM` line |
//! | `version` requires `D.D` | `teleport.qasm`, which writes `OPENQASM 3;` |
//! | `COMMENT` handles only `//` | `adder.qasm`, `vqe.qasm` — both open with `/* */` |
//!
//! OpenQASM 3 states the version statement is optional. The corpus takes it at
//! its word, so this was not a tolerance question: the reader could not read
//! the language's own examples, and every one of them failed before a single
//! statement was examined.
//!
//! # The header gate lived ABOVE the grammar
//!
//! Relaxing the pest rule alone changed nothing. `lower_to_ir_with_dialect`
//! inspects the first non-comment line and refused anything not starting with
//! `OPENQASM`/`OPTICQASM` *before* the parser ran. That is why the fix touches
//! the dispatcher too, and why the measurement below is of `lower_to_ir` rather
//! than of the grammar in isolation.
//!
//! # Why there is no format-sniffing heuristic
//!
//! A tempting alternative was to guess ("does it contain `include`, `qreg`,
//! `qubit`?"). That is one more predicate to keep in step with the grammar, and
//! predicates drifting from the code they describe is a defect class this
//! repository has hit repeatedly. Instead there is one path: headerless input
//! is read as OpenQASM 3, and if that fails the error says so. Genuinely
//! unknown input still gets an actionable message — see
//! [`unknown_input_still_explains_itself`].

use omega_parser::lower_to_ir;

/// No version statement at all — the shape 20 of the 21 official programs use.
#[test]
fn a_file_with_no_version_statement_parses() {
    let src = "include \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\n";
    let ir = lower_to_ir(src).expect("the version statement is optional in OpenQASM 3");
    assert_eq!(ir.num_qubits, 2);
    assert_eq!(ir.ops.len(), 2);
}

/// `OPENQASM 3;` — a major version with no minor. Written by `teleport.qasm`,
/// and it used to fail at the `.` with a bare `--> 2:10`.
#[test]
fn a_major_only_version_parses() {
    let src = "OPENQASM 3;\ninclude \"stdgates.inc\";\nqubit[1] q;\nh q[0];\n";
    let ir = lower_to_ir(src).expect("`OPENQASM 3;` is legal");
    assert_eq!(ir.ops.len(), 1);
}

/// The existing spellings must keep working — this is an additive change.
#[test]
fn the_established_headers_still_parse() {
    for src in [
        "OPENQASM 2.0;\nqreg q[1];\nh q[0];\n",
        "OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n",
    ] {
        let ir = lower_to_ir(src).unwrap_or_else(|e| panic!("must still parse: {e}\n{src}"));
        assert_eq!(ir.ops.len(), 1);
    }
}

/// Block comments, including one *before* the header — the position
/// `adder.qasm` and `vqe.qasm` put theirs in, which also exercises the
/// dispatcher's own comment skipping.
#[test]
fn block_comments_are_accepted_anywhere() {
    let src = "/*\n * a leading block comment\n */\ninclude \"stdgates.inc\";\n\
               qubit[2] q; /* inline */\nh q[0]; // trailing line comment\n\
               /* multi\n   line */\ncx q[0], q[1];\n";
    let ir = lower_to_ir(src).expect("block comments are part of the language");
    assert_eq!(ir.num_qubits, 2);
    assert_eq!(
        ir.ops.len(),
        2,
        "comments must not swallow or add statements"
    );
}

/// An unterminated block comment must not silently eat the rest of the file.
#[test]
fn an_unterminated_block_comment_is_an_error() {
    let src = "qubit[1] q;\n/* never closed\nh q[0];\n";
    assert!(
        lower_to_ir(src).is_err(),
        "an unclosed `/*` must fail, not consume the remaining statements"
    );
}

/// Relaxing the header must not turn genuine garbage into a bare pest
/// position. The message names what was assumed, so a user pointing the reader
/// at the wrong kind of file learns why.
#[test]
fn unknown_input_still_explains_itself() {
    let e = lower_to_ir("this is not a circuit at all\n").expect_err("garbage must refuse");
    assert!(
        e.contains("header"),
        "the message must say a header was absent; got: {e}"
    );
    assert!(
        e.contains("OpenQASM 3"),
        "the message must name what it tried instead; got: {e}"
    );
}

/// OPTICQASM is untouched: its own grammar requires its header, so a headerless
/// file was never going to be one, and the new fallback must not capture it.
#[test]
fn opticqasm_still_routes_to_its_own_lane() {
    let src = "OPTICQASM 1.0;\nmode m[2];\nbs m[0], m[1];\n";
    // Whether this particular body is valid is that lane's business; what
    // matters here is that it was NOT read as OpenQASM.
    if let Err(e) = lower_to_ir(src) {
        assert!(
            !e.contains("read as OpenQASM 3"),
            "an OPTICQASM file must not fall through to the QASM lane; got: {e}"
        );
    }
}

/// Scalar declarations — `qubit q;` / `bit c;`, written by `qpt.qasm` and
/// `teleport.qasm`.
///
/// These did not fail *as declarations*: pest's ordered choice fell through to
/// `gate_app_stmt`, which matched `qubit` as a GATE NAME and `q` as its
/// operand, so the file parsed and died at lowering with `unknown gate: qubit`.
/// A rule that does not match is not a rule that refuses — the same trap the
/// grammar already records for `reset`.
#[test]
fn scalar_register_declarations_parse() {
    let src = "qubit q;\nbit c;\nh q;\nc = measure q;\n";
    let ir = lower_to_ir(src).expect("`qubit q;` declares a single qubit");
    assert_eq!(ir.num_qubits, 1, "a scalar declaration is width 1");
    assert_eq!(ir.num_classical_bits, 1);
}

/// Sized and scalar forms must coexist — the size became optional, not absent.
#[test]
fn sized_and_scalar_declarations_coexist() {
    let src = "qubit[3] a;\nqubit b;\nbit[2] c;\nbit d;\nh b;\n";
    let ir = lower_to_ir(src).expect("both forms are legal");
    assert_eq!(ir.num_qubits, 4, "3 + 1");
    assert_eq!(ir.num_classical_bits, 3, "2 + 1");
}

/// `cphase(λ)` is the OpenQASM 3 spelling of `cp`/`cu1`. Asserted to be the
/// SAME operator rather than merely accepted: all three widen to `CU3(0, 0, λ)`,
/// so they must lower to identical op streams.
#[test]
fn cphase_is_the_same_operator_as_cp_and_cu1() {
    let lower = |name: &str| {
        let src = format!("qubit[2] q;\n{name}(0.7) q[0], q[1];\n");
        lower_to_ir(&src).unwrap_or_else(|e| panic!("`{name}` must lower: {e}"))
    };
    let reference = lower("cp");
    for name in ["cphase", "cu1"] {
        let got = lower(name);
        assert_eq!(
            got.ops.len(),
            reference.ops.len(),
            "`{name}` must produce the same op count as `cp`"
        );
        for (i, (a, b)) in got.ops.iter().zip(&reference.ops).enumerate() {
            assert_eq!(a.gate, b.gate, "`{name}` op {i} differs in gate kind");
            assert_eq!(a.qubits, b.qubits, "`{name}` op {i} differs in operands");
            // `ParamExpr` has no `PartialEq`, so compare its rendering — which
            // is the property that matters here anyway: the widened
            // `(0, 0, lambda)` triple must be identical.
            assert_eq!(
                format!("{:?}", a.params),
                format!("{:?}", b.params),
                "`{name}` op {i} differs in parameters"
            );
        }
    }
}

/// A wrong parameter count on the new spelling is refused like the others,
/// rather than silently widening whatever it was given.
#[test]
fn cphase_with_the_wrong_arity_is_refused() {
    let src = "qubit[2] q;\ncphase(0.1, 0.2) q[0], q[1];\n";
    let e = lower_to_ir(src).expect_err("cphase takes exactly one angle");
    assert!(
        e.contains("cphase"),
        "the message must name the gate; got: {e}"
    );
}

/// A braced `if` body — `if (c==1) { x q[0]; }`. OpenQASM 3 allows either
/// form; `teleport.qasm`'s own comment says "braces optional in this case".
///
/// **Parsing it is not the claim — applying the guard is.** A braced body that
/// parsed and then emitted its statements *unconditioned* would be a silently
/// wrong circuit, which is exactly the defect the guarded-`measure`/`reset`
/// work already had to fix once. So this asserts on `condition`, not on op
/// count.
#[test]
fn a_braced_if_body_carries_the_guard() {
    let src = "qubit[2] q;\nbit[1] c;\nh q[0];\nmeasure q[0] -> c[0];\nif (c==1) { x q[1]; }\n";
    let ir = lower_to_ir(src).expect("a braced if body is legal OpenQASM 3");
    let x = ir
        .ops
        .iter()
        .find(|o| matches!(o.gate, omega_core::circuit::GateKind::X))
        .expect("the guarded X must be emitted");
    assert!(
        x.condition.is_some(),
        "the X inside the braces must carry the guard; unconditioned it is a \
         different circuit"
    );
}

/// **Every** statement in the block is guarded, not just the first. A block is
/// where "patch the first op" and "patch all of them" first diverge, and the
/// bare form cannot distinguish them.
#[test]
fn every_statement_in_a_braced_block_is_guarded() {
    let src = "qubit[3] q;\nbit[1] c;\nh q[0];\nmeasure q[0] -> c[0];\n\
               if (c==1) { x q[1]; y q[2]; z q[1]; }\n";
    let ir = lower_to_ir(src).expect("a multi-statement block is legal");
    let guarded = ir.ops.iter().filter(|o| o.condition.is_some()).count();
    assert_eq!(
        guarded, 3,
        "all three statements in the block must be conditioned, got {guarded}"
    );
}

/// The bare and braced spellings must produce the same circuit — otherwise one
/// of them is wrong and the corpus only exercises whichever we happened to try.
#[test]
fn braced_and_bare_guards_agree() {
    let base = "qubit[2] q;\nbit[1] c;\nh q[0];\nmeasure q[0] -> c[0];\n";
    let bare = lower_to_ir(&format!("{base}if (c==1) x q[1];\n")).expect("bare form");
    let braced = lower_to_ir(&format!("{base}if (c==1) {{ x q[1]; }}\n")).expect("braced form");
    assert_eq!(bare.ops.len(), braced.ops.len(), "same op count");
    for (i, (a, b)) in bare.ops.iter().zip(&braced.ops).enumerate() {
        assert_eq!(a.gate, b.gate, "op {i} gate differs");
        assert_eq!(a.condition, b.condition, "op {i} guard differs");
    }
}

/// An empty block guards nothing, and the existing "produced no operation"
/// refusal must catch it rather than letting the guard evaporate.
#[test]
fn an_empty_braced_block_is_refused() {
    let src = "qubit[2] q;\nbit[1] c;\nmeasure q[0] -> c[0];\nif (c==1) { }\n";
    assert!(
        lower_to_ir(src).is_err(),
        "an empty guarded block produces no operation and must be refused, \
         not silently dropped"
    );
}

/// **The single-bit guard our own emitter writes.**
///
/// `to_qasm3` produces `if (c[0] == true) x q[1];` and this reader refused it —
/// we could not read the file we write. Closing that is the point of the
/// indexed `cond_target`.
///
/// The guard must condition on ONE bit, not the register: reusing the
/// whole-register tuple would assert the entire register equals the value, a
/// different predicate and exactly the reinterpretation `to_qasm` refuses to
/// perform on the export side.
#[test]
fn a_single_bit_guard_conditions_on_that_bit_only() {
    let src = "qubit[2] q;\nbit[4] c;\nmeasure q[0] -> c[2];\nif (c[2] == true) x q[1];\n";
    let ir = lower_to_ir(src).expect("`c[2] == true` is legal OpenQASM 3");
    let x = ir
        .ops
        .iter()
        .find(|o| matches!(o.gate, omega_core::circuit::GateKind::X))
        .expect("the guarded X must be emitted");
    let (start, width, value) = x.condition.expect("the X must carry the guard");
    assert_eq!(width, 1, "a single-bit guard conditions on ONE bit");
    assert_eq!(start, 2, "it must be bit 2, not the register base");
    assert_eq!(value, 1, "`true` is 1");
}

/// `false` is honoured as 0, not treated as "absent".
#[test]
fn a_false_guard_is_value_zero() {
    let src = "qubit[2] q;\nbit[1] c;\nmeasure q[0] -> c[0];\nif (c[0] == false) x q[1];\n";
    let ir = lower_to_ir(src).expect("`== false` is legal");
    let x = ir
        .ops
        .iter()
        .find(|o| matches!(o.gate, omega_core::circuit::GateKind::X))
        .expect("guarded X");
    assert_eq!(x.condition.expect("guard").2, 0, "`false` is 0");
}

/// The whole-register form keeps working and keeps its own width.
#[test]
fn a_whole_register_guard_is_unchanged() {
    let src = "qreg q[2];\ncreg c[3];\nmeasure q[0] -> c[0];\nif (c == 1) x q[1];\n";
    let ir = lower_to_ir(src).expect("QASM 2.0's whole-register guard still works");
    let x = ir
        .ops
        .iter()
        .find(|o| matches!(o.gate, omega_core::circuit::GateKind::X))
        .expect("guarded X");
    let (start, width, value) = x.condition.expect("guard");
    assert_eq!((start, width, value), (0, 3, 1), "the whole 3-bit register");
}

/// **The mixed forms are refused**, because strict OpenQASM 3 consumers reject
/// them — `bit` pairs with `bool`, `bitarray` with `int`. Accepting
/// `c[0] == 1` here would have entrenched on the read side exactly the invalid
/// spelling the emitter was just fixed for, and made the round trip look
/// healthy while producing files qiskit will not load.
#[test]
fn mixing_bit_with_int_or_register_with_bool_is_refused() {
    let cases = [
        (
            "qubit[2] q;\nbit[2] c;\nmeasure q[0] -> c[0];\nif (c[0] == 1) x q[1];\n",
            "single BIT to an integer",
        ),
        (
            "qubit[2] q;\nbit[2] c;\nmeasure q[0] -> c[0];\nif (c == true) x q[1];\n",
            "whole REGISTER to a boolean",
        ),
    ];
    for (src, expect) in cases {
        let e = lower_to_ir(src).expect_err("a mixed comparison must be refused");
        assert!(
            e.contains(expect),
            "the message must say which side is wrong (looking for {expect:?}); got: {e}"
        );
        // And it must show the caller what to write instead.
        assert!(
            e.contains("write") || e.contains("address a single bit"),
            "the refusal must offer the correct spelling; got: {e}"
        );
    }
}

/// An out-of-range bit index is refused rather than silently conditioning on
/// whatever sits past the register.
#[test]
fn an_out_of_range_guard_bit_is_refused() {
    let src = "qubit[2] q;\nbit[2] c;\nmeasure q[0] -> c[0];\nif (c[7] == true) x q[1];\n";
    let e = lower_to_ir(src).expect_err("bit 7 does not exist in a 2-bit creg");
    assert!(
        e.contains("out of range"),
        "the message must say the index is out of range; got: {e}"
    );
}

/// The end-to-end claim, against the specification's own corpus: `rb.qasm` is
/// the first official example to parse, and it does so only because all three
/// rules above were relaxed together.
#[test]
fn an_official_example_parses_end_to_end() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("third_party/openqasm-examples/rb.qasm");
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let ir = lower_to_ir(&src).unwrap_or_else(|e| panic!("rb.qasm is Tier 1 and must parse: {e}"));
    assert_eq!(ir.num_qubits, 2, "rb.qasm declares qubit[2] q");
    assert!(
        ir.ops.len() >= 8,
        "rb.qasm has h/cz/s/z plus barriers and two measures, got {} ops",
        ir.ops.len()
    );
}
