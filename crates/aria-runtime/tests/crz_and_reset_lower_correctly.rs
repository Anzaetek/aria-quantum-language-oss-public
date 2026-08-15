// SPDX-License-Identifier: Apache-2.0
//! **`aria-runtime`'s own lowering of `CRz` and `Reset`, through its own entry
//! points.**
//!
//! `PLAN-EXPORT-INTEGRITY.md` P2. Commit `b0745c0` claimed these two arms were
//! covered. They were not: the tests it added call
//! `aria_core::backends::omega::to_omega_ir` — a **different crate and a
//! different function** — while `aria_runtime::lower`'s `AKind::CRz` and
//! `AKind::Reset` arms had no coverage at all.
//!
//! Re-measured on 2026-08-13, mutating `crates/aria-runtime/src/lower.rs`:
//!
//! ```text
//!   AKind::Reset => (OKind::Barrier, ...)   ->  0 failing targets
//!   AKind::CRz   => (OKind::CU3,     ...)   ->  0 failing targets
//! ```
//!
//! Both mutations passed the entire `aria-runtime` suite, so the earlier claim
//! was wrong and is corrected in the commit that adds this file.
//!
//! These tests go through `run_counts` and `expectation` — the runtime's public
//! surface — rather than through the lowering table, because testing a table by
//! reading the table is what produced the original gap.

use aria_core::ast::parse_aria;
use aria_runtime::run::{expectation, run_counts, BackendSel};
use omega_core::executor::ExecResult;
use std::collections::HashMap;

fn circuit(src: &str, name: &str) -> aria_core::ast::nodes::Circuit {
    parse_aria(src)
        .expect("parse")
        .instantiate(name, &[])
        .expect("instantiate")
}

fn counts(src: &str, name: &str, shots: u32) -> HashMap<String, u64> {
    match run_counts(
        &circuit(src, name),
        &HashMap::new(),
        shots,
        Some(0xC0FFEE),
        BackendSel::Sim,
    )
    .expect("run_counts")
    {
        ExecResult::Counts(c) => c
            .into_iter()
            // Matches the old `{k:b}`: leading zeros stripped, but never to
            // the empty string — an all-zero outcome is "0".
            .map(|(k, v)| {
                let t = k.to_bitstring();
                let t = t.trim_start_matches('0');
                (
                    if t.is_empty() {
                        "0".to_string()
                    } else {
                        t.to_string()
                    },
                    v as u64,
                )
            })
            .collect(),
        other => panic!("expected Counts, got {other:?}"),
    }
}

/// **`Reset` must actually reset**, not act as a `Barrier`.
///
/// The mutation this catches is `AKind::Reset => (OKind::Barrier, ...)`, which
/// leaves the qubit wherever it was. So the circuit puts the qubit in `|1>`
/// first: with a real reset every shot measures 0, with a barrier every shot
/// measures 1. Total separation, no statistics needed.
///
/// A fixture starting from `|0>` would be **invariant** under exactly this
/// mutation — reset-of-zero and barrier-of-zero are the same thing — which is
/// the shape of defect this repository keeps producing.
#[test]
fn reset_returns_the_qubit_to_zero_and_is_not_a_barrier() {
    let src = "circuit R() {\n  qreg q[1]\n  creg c[1]\n  apply X on q[0]\n  \
               apply RESET on q[0]\n  measure q[0] -> c[0]\n}\n";
    let c = counts(src, "R", 256);
    let zeros = *c.get("0").unwrap_or(&0);
    assert_eq!(
        zeros, 256,
        "after X then RESET every shot must read 0; got {c:?}. If this reads 256 \
         ones, RESET lowered to something that leaves the state alone."
    );
}

/// Guard the guard: without the RESET the same circuit must read all ones.
///
/// Without this, a `RESET` that somehow produced a *constant zero* circuit —
/// or an `X` that silently did nothing — would satisfy the test above.
#[test]
fn the_reset_fixture_would_read_ones_without_the_reset() {
    let src = "circuit NoR() {\n  qreg q[1]\n  creg c[1]\n  apply X on q[0]\n  \
               measure q[0] -> c[0]\n}\n";
    let c = counts(src, "NoR", 256);
    assert_eq!(
        *c.get("1").unwrap_or(&0),
        256,
        "the fixture must reach |1> before the reset, or the reset test proves \
         nothing; got {c:?}"
    );
}

/// **`CRz` must be `CRz`, not `CP` or `CU3`.**
///
/// `CRz(λ) = diag(1, 1, e^{−iλ/2}, e^{+iλ/2})` while `CP(λ) = diag(1, 1, 1,
/// e^{iλ})`. They differ by a relative phase `e^{−iλ/2}` on the controlled
/// block — not a global phase, so it is visible in interference, but **not** in
/// a computational-basis measurement of either qubit.
///
/// So this measures an interference observable: with the control in `|+>` and
/// the target in `|+>`, `<X0>` depends on the phase the controlled block
/// applies. A basis-state fixture would be invariant under the substitution and
/// would prove nothing.
#[test]
fn crz_is_not_cp_and_the_difference_is_observable() {
    // Control q[0] in |+>, target q[1] in |+>.
    let src = |gate: &str| {
        format!(
            "circuit C() {{\n  qreg q[2]\n  apply H on q[0]\n  apply H on q[1]\n  \
             apply {gate} on q[0], q[1]\n}}\n"
        )
    };
    let crz = expectation(
        &circuit(&src("CRZ(1.1)"), "C"),
        "X0",
        &HashMap::new(),
        BackendSel::Sim,
    )
    .expect("expectation");

    // The reference value for CRz, computed from the definition rather than
    // taken from a run: with both qubits in |+>, CRz(λ) leaves <X0> at
    // cos(λ/2) — the control picks up the average of the two conditional
    // phases e^{∓iλ/2}.
    let want = (1.1f64 / 2.0).cos();
    assert!(
        (crz - want).abs() < 1e-12,
        "<X0> after CRz(1.1) should be cos(0.55) = {want:.12}, got {crz:.12}"
    );

    // And the fixture must actually distinguish CRz from CP: if these agreed,
    // the assertion above would pass under the substitution too.
    let cp = expectation(
        &circuit(&src("CP(1.1)"), "C"),
        "X0",
        &HashMap::new(),
        BackendSel::Sim,
    )
    .expect("expectation");
    assert!(
        (crz - cp).abs() > 1e-6,
        "CRz and CP give the same <X0> on this fixture ({crz:.12} vs {cp:.12}), so \
         it cannot catch one being lowered as the other"
    );
}

/// `CRz`'s parameter must reach the backend. A lowering that dropped or zeroed
/// it would still produce a `CRz` op and pass the kind check above at λ = 0.
#[test]
fn the_crz_angle_reaches_the_backend() {
    let at = |lambda: f64| {
        let src = format!(
            "circuit C() {{\n  qreg q[2]\n  apply H on q[0]\n  apply H on q[1]\n  \
             apply CRZ({lambda}) on q[0], q[1]\n}}\n"
        );
        expectation(&circuit(&src, "C"), "X0", &HashMap::new(), BackendSel::Sim)
            .expect("expectation")
    };
    let (a, b) = (at(0.4), at(1.9));
    assert!(
        (a - b).abs() > 1e-6,
        "two different CRz angles gave the same <X0> ({a:.12} vs {b:.12}); the \
         parameter is not reaching the backend"
    );
    assert!((a - (0.4f64 / 2.0).cos()).abs() < 1e-12);
    assert!((b - (1.9f64 / 2.0).cos()).abs() < 1e-12);
}
