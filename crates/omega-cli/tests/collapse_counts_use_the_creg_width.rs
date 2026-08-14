// SPDX-License-Identifier: Apache-2.0
//! **A counts key is as wide as the register that produced it — which is not
//! always the qubit count.**
//!
//! In collapse mode the mid-circuit `measure`s have already run, so the CREG is
//! the outcome (`creg_to_u64`); in skip mode the whole qubit register is
//! sampled. Gating and rendering on `num_qubits` in both cases is wrong for the
//! first.
//!
//! Measured before this: a 1024-qubit circuit with a mid-circuit measure and
//! feed-forward into `creg c[2]` was REFUSED, because the guard asked whether
//! 1024 > 64 rather than whether the two-bit outcome fits. That is the natural
//! shape of a large run — measure a handful of qubits out of many — and it does
//! not need a wider key at all.
//!
//! The skip-mode refusal is NOT a bug and is pinned below: there the backend
//! genuinely samples all 1024 qubits, so the key really is 1024 bits wide.

use omega_core::executor::counts_outcome_width;

fn ir(src: &str) -> omega_core::circuit::CircuitIR {
    omega_parser::lower_to_ir(src).expect("lower")
}

/// Collapse: the width is the classical register, not the qubit count.
#[test]
fn collapse_width_is_the_creg_not_the_qubit_count() {
    let c = ir("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1024];\ncreg c[2];\n\
                h q[0];\nmeasure q[0] -> c[0];\nif (c==1) x q[1];\nmeasure q[1] -> c[1];\n");
    assert_eq!(
        counts_outcome_width(&c, true),
        2,
        "two measured bits is a two-bit outcome, whatever the register size"
    );
}

/// Skip: the width IS the qubit count, because the whole register is sampled.
///
/// Without this, "use the creg everywhere" would look correct and would silently
/// under-report the width on the sampling path — re-creating the original
/// truncation defect from the other side.
#[test]
fn skip_width_is_the_qubit_count() {
    let c = ir("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[70];\ncreg c[2];\n\
                h q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n");
    assert_eq!(
        counts_outcome_width(&c, false),
        70,
        "skip mode samples the whole register, so the key is that wide"
    );
}

/// The two modes must actually differ on the same circuit — otherwise both
/// assertions above could hold under a function that ignored its argument.
#[test]
fn the_two_modes_disagree_and_that_is_the_point() {
    let c = ir("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[70];\ncreg c[2];\n\
                h q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n");
    assert_ne!(
        counts_outcome_width(&c, true),
        counts_outcome_width(&c, false),
        "if the mode made no difference this whole distinction would be noise"
    );
}

/// A circuit with no measurements has no creg-keyed outcome at all.
#[test]
fn collapse_width_is_zero_without_measurements() {
    let c = ir("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[8];\nh q[0];\n");
    assert_eq!(counts_outcome_width(&c, true), 0);
    assert_eq!(counts_outcome_width(&c, false), 8);
}
