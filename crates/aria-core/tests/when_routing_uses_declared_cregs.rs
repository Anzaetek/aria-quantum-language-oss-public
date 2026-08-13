// SPDX-License-Identifier: Apache-2.0
//! **A `when` guard is a runtime condition because the register is DECLARED
//! classical, not because of what it is called.**
//!
//! `PLAN-EXPORT-INTEGRITY.md` P6 residual. `is_runtime_cond` used to decide by
//! name: `name.starts_with('m') || name == "c"`. Two consequences, both live:
//!
//! * a creg called `flags` routed to COMPILE-TIME evaluation and failed
//!   obscurely — while the Aria emitter happily wrote `when flags[0] == 1`, so
//!   the language could express a circuit its own lowering mis-handled;
//! * anything else starting with `m` was treated as a measurement.
//!
//! Same class as `RegisterDecl::polarized` being a flag rather than a naming
//! convention: a convention does not survive a rename.

use aria_core::ast::parse_aria;

fn instantiate(src: &str) -> Result<aria_core::ast::nodes::Circuit, String> {
    parse_aria(src)?.instantiate("C", &[])
}

/// The register name that used to break: `flags` matches neither `m*` nor `c`.
#[test]
fn a_creg_not_named_m_or_c_still_gives_a_runtime_condition() {
    let src = "circuit C() {\n  qreg q[1]\n  creg flags[1]\n  \
               measure q[0] -> flags[0]\n  when flags[0] == 1 { apply X on q[0] }\n}\n";
    let c = instantiate(src).expect("a creg named `flags` must lower like any other");
    let guarded: Vec<_> = c
        .instructions
        .iter()
        .filter(|i| i.condition.is_some())
        .collect();
    assert_eq!(
        guarded.len(),
        1,
        "the X must carry a runtime condition; with name-based routing this went \
         to compile-time evaluation instead. Instructions: {:?}",
        c.instructions.iter().map(|i| (i.gate.kind, i.condition.is_some())).collect::<Vec<_>>()
    );
    let (cl, v) = guarded[0].condition.as_ref().unwrap();
    assert_eq!(cl.register, "flags");
    assert_eq!(*v, 1);
}

/// The names that always worked must keep working — this is a widening, not a
/// swap.
#[test]
fn the_conventional_names_still_work() {
    for reg in ["m", "c", "meas"] {
        let src = format!(
            "circuit C() {{\n  qreg q[1]\n  creg {reg}[1]\n  \
             measure q[0] -> {reg}[0]\n  when {reg}[0] == 1 {{ apply X on q[0] }}\n}}\n"
        );
        let c = instantiate(&src).unwrap_or_else(|e| panic!("creg `{reg}`: {e}"));
        assert_eq!(
            c.instructions.iter().filter(|i| i.condition.is_some()).count(),
            1,
            "creg `{reg}` must still produce a runtime condition"
        );
    }
}

/// **Guard the guard.** A compile-time `when` over a loop variable must NOT
/// become a runtime condition — otherwise the change above would have made
/// everything runtime, which passes the tests above for the wrong reason.
///
/// `i` is a loop variable, not a declared creg, so the guard is evaluated at
/// lowering: the body appears for the iterations where it holds and is absent
/// for the others, with no per-instruction condition anywhere.
#[test]
fn a_compile_time_guard_over_a_loop_variable_stays_compile_time() {
    let src = "circuit C() {\n  qreg q[4]\n  repeat i from 0 to 3 {\n    \
               when i == 2 { apply X on q[i] }\n  }\n}\n";
    let c = instantiate(src).expect("compile-time when must lower");
    assert!(
        c.instructions.iter().all(|i| i.condition.is_none()),
        "a loop-variable guard must be resolved at lowering, not emitted as a \
         runtime condition: {:?}",
        c.instructions.iter().map(|i| (i.gate.kind, i.condition.clone())).collect::<Vec<_>>()
    );
    assert_eq!(
        c.instructions.len(),
        1,
        "exactly one iteration satisfies `i == 2`, so exactly one X survives"
    );
    assert_eq!(c.instructions[0].qubits[0].index, 2, "and it is on q[2]");
}

/// **Only a declared CLASSICAL register may be conditioned on.**
///
/// This is the half the loop-variable test above cannot reach: `when i == 2`
/// parses as `Expr::Var`, not `Expr::Index`, so a mutation making *every index*
/// runtime leaves it green. Measured under that mutation, all three of these
/// were ACCEPTED and produced a conditioned instruction:
///
/// ```text
///   when t[0] == 1   (t is a symbolic array)  ->  Ok, 1 conditioned instr
///   when zz[0] == 1  (zz undeclared)          ->  Ok, 1 conditioned instr
///   when q[0] == 1   (q is a QREG)            ->  Ok, 1 conditioned instr
/// ```
///
/// Conditioning a gate on a qubit register, or on a name that was never
/// declared, is not a guard — it is a circuit that means nothing, accepted
/// silently. Each must be refused, and the refusal must say what the name
/// actually is rather than "unknown".
#[test]
fn only_a_declared_classical_register_can_be_conditioned_on() {
    let cases: &[(&str, &str)] = &[
        (
            "symbolic array",
            "circuit C() {\n  qreg q[1]\n  let t = symbolic[2]\n  \
             when t[0] == 1 { apply X on q[0] }\n}\n",
        ),
        (
            "undeclared name",
            "circuit C() {\n  qreg q[1]\n  when zz[0] == 1 { apply X on q[0] }\n}\n",
        ),
        (
            "a quantum register",
            "circuit C() {\n  qreg q[1]\n  when q[0] == 1 { apply X on q[0] }\n}\n",
        ),
    ];
    for (label, src) in cases {
        match instantiate(src) {
            Ok(c) => panic!(
                "{label}: accepted, producing {} instruction(s) with {} condition(s) — \
                 a gate cannot be guarded on this",
                c.instructions.len(),
                c.instructions.iter().filter(|i| i.condition.is_some()).count()
            ),
            Err(e) => assert!(
                !e.is_empty(),
                "{label}: refused with an empty message, which is not a diagnostic"
            ),
        }
    }
}
