// SPDX-License-Identifier: Apache-2.0
//! **Counts are keyed by a `u64`, so more than 64 qubits must be refused — not
//! silently truncated.**
//!
//! Reported as "MPS returns physically wrong counts on GHZ at 128 qubits", with
//! the diagnosis "an indexing/ordering boundary between 64 and 128". Both halves
//! were wrong, and the tests here are shaped by what it took to find that out.
//!
//! Measured before the fix, sweeping the boundary one qubit at a time:
//!
//! ```text
//!    63 qubits:  |0…0>, |1…1>          correct
//!    64 qubits:  |0…0>, |1…1>          correct
//!    65 qubits:  |0…0>, |0 1^64>       ONE leading zero
//!    66 qubits:  |0…0>, |00 1^64>      TWO leading zeros
//!   128 qubits:  |0…0>, |0^64 1^64>    the reported symptom
//! ```
//!
//! One lost bit per qubit above 64 — a 64-bit container, not a truncation
//! threshold and not an ordering pathology. And **not MPS**: the same GHZ at 65
//! qubits truncated identically on the stabilizer backend, so every backend
//! returning counts was affected.
//!
//! # Why the boundary is swept rather than sampled
//!
//! Testing 64 and 128 only — which is what the original report exercised —
//! cannot distinguish "u64 key" from "ordering bug", because both produce a
//! wrong 128-qubit answer and a right 64-qubit one. The one-qubit steps at 65
//! and 66 are what make the cause unambiguous, so they are what is pinned.
//!
//! # Why the assertion is on key CONTENTS
//!
//! A GHZ state yields exactly **two** distinct outcomes whether or not the key
//! truncates — the truncated result also has two keys. So an assertion on the
//! number of outcomes, or on the shot total, passes on broken output. The
//! assertion has to be that the two keys are all-zeros and all-ones.

use omega_backend_mps::MpsBackend;
use omega_backend_pauli::PauliBackend;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;

/// GHZ over `n` qubits: `h q[0]`, a CX chain, then measure every qubit.
fn ghz(n: usize) -> String {
    let mut s = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\ncreg c[{n}];\nh q[0];\n");
    for i in 0..n - 1 {
        s.push_str(&format!("cx q[{i}], q[{}];\n", i + 1));
    }
    for i in 0..n {
        s.push_str(&format!("measure q[{i}] -> c[{i}];\n"));
    }
    s
}

fn cfg(shots: u32) -> ExecConfig {
    ExecConfig {
        shots: Some(shots),
        seed: Some(1),
        mid_circuit_mode: MidCircuitMode::Collapse,
    }
}

fn run(backend: &dyn Backend, n: usize) -> omega_core::error::Result<ExecResult> {
    let ir = omega_parser::lower_to_ir(&ghz(n)).expect("GHZ must lower");
    backend.execute(&ir, &ParameterBinding::default(), &cfg(200))
}

fn backends() -> Vec<(&'static str, Box<dyn Backend>)> {
    vec![
        ("mps", Box::new(MpsBackend::new(64)) as Box<dyn Backend>),
        ("pauli", Box::new(PauliBackend::new()) as Box<dyn Backend>),
    ]
}

/// At and below 64 qubits the outcomes must be exactly all-zeros and all-ones.
#[test]
fn ghz_counts_are_correct_up_to_the_key_width() {
    for (name, b) in backends() {
        for n in [63usize, 64] {
            let res = run(b.as_ref(), n)
                .unwrap_or_else(|e| panic!("{name} at {n} qubits must run: {e}"));
            let ExecResult::Counts(c) = res else {
                panic!("{name}: expected Counts")
            };

            let all_ones: u64 = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
            let mut keys: Vec<u64> = c.keys().copied().collect();
            keys.sort_unstable();
            assert_eq!(
                keys,
                vec![0u64, all_ones],
                "{name} at {n} qubits: a GHZ admits exactly |0…0> and |1…1>, got \
                 {keys:?}. Asserting on the NUMBER of outcomes would pass here \
                 even when the key is truncated, which is why the contents are \
                 checked."
            );
        }
    }
}

/// Above 64 qubits every backend must REFUSE, at each of the steps that
/// exposed the cause.
#[test]
fn more_than_64_qubits_is_refused_not_truncated() {
    for (name, b) in backends() {
        for n in [65usize, 66, 128] {
            match run(b.as_ref(), n) {
                Ok(ExecResult::Counts(c)) => {
                    let mut keys: Vec<u64> = c.keys().copied().collect();
                    keys.sort_unstable();
                    panic!(
                        "{name} at {n} qubits returned counts with keys {keys:?} — \
                         a {n}-bit outcome does not fit a 64-bit key, so these are \
                         wrong, not truncated"
                    );
                }
                Ok(other) => panic!("{name} at {n}: expected Counts or an error, got {other:?}"),
                Err(e) => {
                    let m = e.to_string();
                    assert!(
                        m.contains("64") && m.contains(&n.to_string()),
                        "{name} at {n}: the refusal must state both the limit and \
                         the actual width, got: {m}"
                    );
                }
            }
        }
    }
}

/// **Guard the guard.** The refusal must be about the COUNTS KEY, not about
/// large circuits in general — an expectation value over the same 128-qubit
/// circuit is unaffected and must still work.
///
/// Without this, "refuse everything above 64 qubits" would satisfy the test
/// above while destroying the regime MPS and the stabilizer backend exist for.
#[test]
fn the_refusal_is_about_counts_not_about_size() {
    let n = 128;
    // Same circuit shape, but no measurements and no shots — an analytic
    // expectation, which never touches the counts key.
    let mut src = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\nh q[0];\n");
    for i in 0..n - 1 {
        src.push_str(&format!("cx q[{i}], q[{}];\n", i + 1));
    }
    let ir = omega_parser::lower_to_ir(&src).expect("lower");

    let obs = omega_core::executor::Observable::parse("Z0").expect("observable");
    let b = PauliBackend::new();
    let v = b
        .expectation(&ir, &ParameterBinding::default(), &obs)
        .expect("a 128-qubit expectation must still work — the u64 limit is a \
                 property of the counts KEY, not of the simulation");
    assert!(
        v.abs() < 1e-9,
        "<Z0> on a 128-qubit GHZ is 0 by symmetry, got {v}"
    );
}
