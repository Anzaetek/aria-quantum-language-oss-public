// SPDX-License-Identifier: Apache-2.0
//! **A GHZ over n qubits has exactly two outcomes, at every n — including
//! 1024.**
//!
//! This file used to assert the opposite: that above 64 qubits every backend
//! must REFUSE, because `ExecResult::Counts` was keyed by a `u64` and every bit
//! above 63 was silently dropped:
//!
//! ```text
//!   63 qubits:  |0…0>, |1…1>        correct
//!   64 qubits:  |0…0>, |1…1>        correct
//!   65 qubits:  |0…0>, |0 1^64>     one leading zero lost, silently
//! ```
//!
//! The refusal was step 1 of PLAN-WIDE-COUNTS and was never the goal — it
//! forbade exactly the regime MPS exists for. The key is now an `Outcome`,
//! which carries as many words as it needs, so the same fixtures assert
//! CORRECTNESS at the widths that used to be turned away.
//!
//! # What could make this pass for the wrong reason
//!
//! * **Asserting on the NUMBER of outcomes.** A truncated GHZ still has exactly
//!   two keys — that is precisely how the original defect hid. Every assertion
//!   below is on the key CONTENTS.
//! * **Testing only GHZ.** Its all-ones outcome differs above bit 64 only
//!   trivially. `mixed_width_outcomes_differ_above_bit_64` uses a fixture whose
//!   outcomes differ in a single high bit and nowhere else.

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
    // Shots scale DOWN with width. A GHZ has exactly two outcomes, so a handful
    // of shots proves as much as hundreds — and the stabilizer backend measures
    // every one of n qubits per shot, which was measured at ~1.1 s/shot at
    // n = 1024. At 200 shots this file took 338 s; the assertions are on the
    // key CONTENTS, not on statistics, so the count buys nothing.
    let shots = if n > 512 { 4 } else if n > 128 { 20 } else { 200 };
    backend.execute(&ir, &ParameterBinding::default(), &cfg(shots))
}

fn backends() -> Vec<(&'static str, Box<dyn Backend>)> {
    vec![
        ("mps", Box::new(MpsBackend::new(64)) as Box<dyn Backend>),
        ("pauli", Box::new(PauliBackend::new()) as Box<dyn Backend>),
    ]
}

/// A GHZ admits exactly `|0…0>` and `|1…1>` — at 63, 64, 65, 128, 256 and 1024.
///
/// 65 is the first width the old `u64` key could not represent; 1024 is the
/// requirement this whole change exists for. The assertion is on the key
/// CONTENTS at each width, because a truncated GHZ still has two keys.
#[test]
fn ghz_counts_are_correct_at_every_width() {
    for (name, b) in backends() {
        for n in [63usize, 64, 65, 128, 256, 1024] {
            let res = run(b.as_ref(), n)
                .unwrap_or_else(|e| panic!("{name} at {n} qubits must run: {e}"));
            let ExecResult::Counts(c) = res else {
                panic!("{name}: expected Counts")
            };

            let mut keys: Vec<String> = c.keys().map(|o| o.to_bitstring()).collect();
            keys.sort();
            let (zeros, ones) = ("0".repeat(n), "1".repeat(n));

            // Every key must be one of the two. This is the assertion that a
            // TRUNCATED key fails, and it holds at any shot count.
            for k in &keys {
                assert!(
                    *k == zeros || *k == ones,
                    "{name} at {n} qubits: |{k}> is not a GHZ outcome. Asserting \
                     on the NUMBER of outcomes would pass here even when the key \
                     is truncated, which is why the contents are checked."
                );
            }
            // Both must actually OCCUR — otherwise "only |1…1> ever appears"
            // would satisfy the check above. Only asserted where the shot count
            // makes it near-certain: at n > 512 the run is 4 shots (the
            // stabilizer backend measures every qubit, ~1.1 s/shot at 1024), and
            // 2^-4 of the time a fair GHZ gives one outcome four times.
            if n <= 512 {
                assert_eq!(
                    keys,
                    vec![zeros, ones],
                    "{name} at {n} qubits: both |0…0> and |1…1> must occur"
                );
            }
            for o in c.keys() {
                assert_eq!(
                    o.width() as usize, n,
                    "{name} at {n}: key |{}> is {} bits wide",
                    o.to_bitstring(), o.width()
                );
            }
        }
    }
}

/// **A fixture whose outcomes differ ONLY above bit 64.**
///
/// GHZ is the easiest possible wide case and was the fixture that hid the
/// original defect. Here qubit 0 is `|0>` and qubit 70 is in superposition, so
/// the two outcomes are identical in every bit the old `u64` key could hold and
/// differ in exactly one bit it could not. Under truncation both collapse to
/// the same key and the histogram has ONE entry.
#[test]
fn mixed_width_outcomes_differ_above_bit_64() {
    let n = 80usize;
    let mut src = format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\ncreg c[{n}];\nh q[70];\n"
    );
    for i in 0..n {
        src.push_str(&format!("measure q[{i}] -> c[{i}];\n"));
    }
    let ir = omega_parser::lower_to_ir(&src).expect("lower");
    for (name, b) in backends() {
        let ExecResult::Counts(c) = b
            .execute(&ir, &ParameterBinding::default(), &cfg(400))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
        else {
            panic!("{name}: expected Counts")
        };
        let mut keys: Vec<String> = c.keys().map(|o| o.to_bitstring()).collect();
        keys.sort();
        let zeros = "0".repeat(n);
        // MSB-first, so bit 70 is at index n-1-70 = 9.
        let mut one = vec![b'0'; n];
        one[n - 1 - 70] = b'1';
        let one = String::from_utf8(one).unwrap();
        assert_eq!(
            keys,
            vec![zeros, one],
            "{name}: the two outcomes differ only at bit 70. One entry here \
             means the key was truncated to 64 bits and they collapsed together."
        );
    }
}

/// **The stabilizer guard and its sampler must answer the same question.**
///
/// The guard tested `needs_collapse(circuit) && mode == Collapse`; the sampler
/// keys on the creg for `mode == Collapse && num_classical_bits > 0`. They
/// diverge for a wide collapse-mode circuit that declares a creg but contains
/// no `measure`: `needs_collapse` is false, so the guard priced the outcome at
/// `num_qubits` and refused above 64 — while the sampler would have built a
/// perfectly representable creg-width key.
///
/// Over-refusal rather than a wrong answer, so it is the mild form. But it is
/// the same guard/sampler split that produced *wrong answers* in the MPS
/// backend twice, and the guard's own comment already claimed the two used the
/// "same predicate".
#[test]
fn a_wide_collapse_circuit_with_a_creg_but_no_measures_is_not_refused() {
    use omega_backend_pauli::PauliBackend;
    use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
    use omega_core::params::ParameterBinding;

    // 70 qubits, a 2-bit creg, no `measure` anywhere.
    let mut src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[70];\ncreg c[2];\n".to_string();
    for i in 0..70 {
        src.push_str(&format!("h q[{i}];\n"));
    }
    let ir = omega_parser::lower_to_ir(&src).expect("lower");
    let cfg = ExecConfig {
        shots: Some(50),
        seed: Some(2),
        mid_circuit_mode: MidCircuitMode::Collapse,
    };
    let r = PauliBackend::new().execute(&ir, &ParameterBinding::default(), &cfg);
    let counts = match r {
        Ok(ExecResult::Counts(c)) => c,
        Ok(o) => panic!("{o:?}"),
        Err(e) => panic!(
            "refused a run the sampler would have keyed on a 2-bit creg: {e}\n\
             The guard priced the outcome at num_qubits because `needs_collapse` \
             is false without a measure, while the sampler keys on the creg for \
             any collapse-mode circuit that has one."
        ),
    };
    // Nothing was measured, so every shot records the creg's initial value.
    assert_eq!(counts.values().sum::<u32>(), 50);
    let wide: Vec<String> = counts
        .keys()
        .filter(|o| o.as_u64().unwrap_or(u64::MAX) > 0b11)
        .map(|o| o.to_bitstring())
        .collect();
    assert!(wide.is_empty(), "keys outside the 2-bit creg: {wide:?}");
}
