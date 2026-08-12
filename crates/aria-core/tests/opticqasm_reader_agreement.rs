// SPDX-License-Identifier: Apache-2.0
//! **The two OPTICQASM readers must agree on what is valid.**
//!
//! This workspace reads OPTICQASM twice: `aria_core::ast::opticqasm::from_opticqasm`
//! (regex, into the `aria-core` AST) and `omega_parser::parse_opticqasm` +
//! `lower_opticqasm` / `lower_opticqasm_cv` (pest, into the omega IRs). Two
//! readers of one dialect that disagree about validity is the same defect class
//! as an emitter whose output no reader accepts — a file is "valid OPTICQASM"
//! only relative to whichever one you happened to call.
//!
//! An adversarial review found they disagreed on five inputs, and in every case
//! **`aria-core` was the silent one**:
//!
//! | input | `aria-core` before | `omega-parser` |
//! |---|---|---|
//! | `OPTICQASM 1.0; photon q[2]; ps(0.5) q[0];` on one line | `Ok`, **0 regs, 0 ops** | 2 statements |
//! | `ps(0.5) q[0]; ps(0.7) q[1];` on one line | `Ok`, **1 gate**, param 0.5, modes [0,1] | 2 gates |
//! | `ps(0.5) zz[7];`, no `zz` declared | `Ok`, invented the register | refused |
//! | `bs_rx(1.2, 0.3) q[0];` | `Ok`, one-mode beam splitter | refused (after this change) |
//! | `kerr(0.1) garbage q[0];` | `Ok`, ignored the garbage | refused |
//!
//! Rather than fix those five and hope, this test pins the agreement itself.
//!
//! # What is compared, and what is deliberately not
//!
//! Only **accept vs reject**. The two produce different data types and have
//! genuinely different scopes — `omega-parser` implements `hwp`/`pbs` and the
//! `pol` register marker, which the `aria-core` AST cannot represent yet
//! (PLAN-OPTICQASM-INTEGRITY O4). Those live in `KNOWN_ASYMMETRIC` with the
//! reason stated, so the exemption is visible and finite instead of a silently
//! weakened assertion.

use aria_core::ast::opticqasm::from_opticqasm;
use omega_parser::{lower_opticqasm_cv, parse_opticqasm};

/// Does the pest lane accept this source under *either* profile?
///
/// Either, not both: `ps` is linear-optical and a phase-space rotation, so it
/// lowers on both, while `kerr` lowers only on CV and `hwp` only on DV. "Some
/// profile in this workspace can execute it" is the property that matters.
fn omega_accepts(src: &str) -> Result<(), String> {
    let program = parse_opticqasm(src).map_err(|e| format!("parse: {e}"))?;
    let dv = omega_parser::lower::lower_opticqasm(&program);
    let cv = lower_opticqasm_cv(&program);
    match (dv, cv) {
        (Err(d), Err(c)) => Err(format!("DV: {d} | CV: {c}")),
        _ => Ok(()),
    }
}

/// Sources both readers must ACCEPT.
const ACCEPT: &[(&str, &str)] = &[
    ("plain", "OPTICQASM 1.0;\nphoton q[2];\nps(0.5) q[0];\n"),
    (
        "two gates",
        "OPTICQASM 1.0;\nphoton q[2];\nps(0.5) q[0];\nbs_rx(1.2, 0.3) q[0], q[1];\n",
    ),
    // The grammar's WHITESPACE includes `\n`, so newlines carry no meaning.
    // This is the input that used to yield `Ok(0 regs, 0 ops)`.
    (
        "whole file on one line",
        "OPTICQASM 1.0; photon q[2]; ps(0.5) q[0];",
    ),
    (
        "two statements on one line",
        "OPTICQASM 1.0;\nphoton q[2];\nps(0.5) q[0]; ps(0.7) q[1];\n",
    ),
    (
        "trailing comment",
        "OPTICQASM 1.0;\nphoton q[2];\nps(0.5) q[0]; // a note\n",
    ),
    (
        "leading comment",
        "OPTICQASM 1.0;\n// a note\nphoton q[2];\nps(0.5) q[0];\n",
    ),
    ("cv gates", "OPTICQASM 1.0;\nphoton q[1];\nsqueeze(0.4, 0.2) q[0];\nkerr(0.15) q[0];\n"),
    ("bs alias", "OPTICQASM 1.0;\nphoton q[2];\nbs(1.2, 0.3) q[0], q[1];\n"),
    ("two registers", "OPTICQASM 1.0;\nphoton a[1];\nphoton b[1];\nps(0.5) b[0];\n"),
    ("negative parameter", "OPTICQASM 1.0;\nphoton q[1];\ndisplace(-0.7, -0.1) q[0];\n"),
];

/// Sources both readers must REJECT.
const REJECT: &[(&str, &str)] = &[
    (
        "undefined register",
        "OPTICQASM 1.0;\nphoton q[2];\nps(0.5) zz[7];\n",
    ),
    (
        "mode out of range",
        "OPTICQASM 1.0;\nphoton q[2];\nps(0.5) q[9];\n",
    ),
    (
        "too few modes",
        "OPTICQASM 1.0;\nphoton q[2];\nbs_rx(1.2, 0.3) q[0];\n",
    ),
    (
        "too many modes",
        "OPTICQASM 1.0;\nphoton q[2];\nps(0.5) q[0], q[1];\n",
    ),
    (
        "too many parameters",
        "OPTICQASM 1.0;\nphoton q[1];\nps(0.5, 0.7) q[0];\n",
    ),
    (
        "too few parameters",
        "OPTICQASM 1.0;\nphoton q[2];\nbs_rx(1.2) q[0], q[1];\n",
    ),
    ("unknown gate", "OPTICQASM 1.0;\nphoton q[1];\nwibble(0.5) q[0];\n"),
    (
        "junk in the mode list",
        "OPTICQASM 1.0;\nphoton q[1];\nkerr(0.1) garbage q[0];\n",
    ),
    ("no header", "photon q[1];\nps(0.5) q[0];\n"),
    (
        "bs_ry — named by the grammar, implemented nowhere",
        "OPTICQASM 1.0;\nphoton q[2];\nbs_ry(1.2, 0.3) q[0], q[1];\n",
    ),
];

/// Genuine, bounded scope differences. Each names the reason.
const KNOWN_ASYMMETRIC: &[(&str, &str, &str)] = &[
    (
        "polarization register",
        "OPTICQASM 1.0;\nphoton q[2] pol;\nps(0.5) q[0];\n",
        "the aria-core AST cannot carry the H/V mode mapping (O4), so it refuses \
         rather than silently reinterpret every mode index",
    ),
    (
        "half-wave plate",
        "OPTICQASM 1.0;\nphoton q[1] pol;\nhwp(0.4) q[0];\n",
        "aria-core has no polarization GateKind (O4); omega-parser expands hwp \
         into phase shifters and beam splitters",
    ),
];

#[test]
fn both_readers_accept_the_same_valid_sources() {
    let mut failures = Vec::new();
    for (name, src) in ACCEPT {
        let aria = from_opticqasm(src);
        let omega = omega_accepts(src);
        match (&aria, &omega) {
            (Ok(_), Ok(())) => {}
            (a, o) => failures.push(format!(
                "{name}: aria-core={} omega-parser={}\n    {src:?}",
                a.as_ref().map(|_| "Ok".into()).unwrap_or_else(|e| e.clone()),
                o.as_ref().map(|_| "Ok".to_string()).unwrap_or_else(|e| e.clone()),
            )),
        }
        // Accepting is not enough: `Ok` with an empty circuit is how the
        // one-line case passed for two years' worth of reasoning. Every
        // accepted source here has at least one gate.
        if let Ok(c) = &aria {
            if c.instructions.is_empty() {
                failures.push(format!(
                    "{name}: aria-core returned Ok with ZERO operations — this is the \
                     `Ok(0 regs, 0 ops)` defect, not an acceptance\n    {src:?}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "readers disagree:\n{}", failures.join("\n"));
}

#[test]
fn both_readers_reject_the_same_invalid_sources() {
    let mut failures = Vec::new();
    for (name, src) in REJECT {
        let aria = from_opticqasm(src);
        let omega = omega_accepts(src);
        if aria.is_ok() || omega.is_ok() {
            failures.push(format!(
                "{name}: aria-core={} omega-parser={}\n    {src:?}",
                if aria.is_ok() { "ACCEPTED" } else { "rejected" },
                if omega.is_ok() { "ACCEPTED" } else { "rejected" },
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "a reader accepted something invalid:\n{}",
        failures.join("\n")
    );
}

/// Pins the exemptions so they cannot quietly grow. If one of these starts
/// working on both sides — because O4 landed — this test fails and the case
/// moves into `ACCEPT`, which is the correct way to find out.
#[test]
fn the_known_scope_differences_are_exactly_these() {
    for (name, src, why) in KNOWN_ASYMMETRIC {
        assert!(
            from_opticqasm(src).is_err(),
            "{name}: aria-core now accepts this — if O4 landed, move it to ACCEPT ({why})"
        );
        assert!(
            omega_accepts(src).is_ok(),
            "{name}: omega-parser no longer accepts this, so it is not a scope \
             difference any more ({why})"
        );
    }
}

/// The counts are asserted so a corpus that quietly shrinks to nothing cannot
/// masquerade as agreement.
#[test]
fn the_corpus_is_not_empty() {
    assert!(ACCEPT.len() >= 10, "accept corpus shrank to {}", ACCEPT.len());
    assert!(REJECT.len() >= 10, "reject corpus shrank to {}", REJECT.len());
}
