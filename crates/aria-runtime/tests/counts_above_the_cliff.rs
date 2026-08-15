// SPDX-License-Identifier: Apache-2.0
//! **`aria-runtime` above 64 qubits, on both the noiseless and the noisy path.**
//!
//! `run_counts` and `run_counts_noisy` used to call `project_counts_onto_creg`
//! unconditionally. Above the cliff the backend already returns a creg-keyed
//! key, so the projection re-read qubit positions out of it — a second
//! projection of an already-projected key. The fix is an early return when
//! `counts_keyed_on_creg`, which is only sound if the backend really does key
//! on the creg.
//!
//! Both halves need a test at THIS layer, and neither had one: the widest
//! fixture in the crate was a 1-qubit circuit with `creg c[65]`, which tests
//! the refusal path and never crosses the cliff with a working run. Reverting
//! either early return left the whole workspace green.
//!
//! The noisy path is the sharper case. `NoisyMpsBackend` did NOT project, so
//! the early return handed back raw full-register keys and the previous
//! (double) projection had been masking it — a fix that removed a different
//! gate. Both are fixed; this pins them together, because they are only
//! correct as a pair.

use aria_core::ast::parse_aria;
use aria_runtime::run::{run_counts, run_counts_noisy, BackendSel};
use omega_core::executor::ExecResult;
use omega_core::noise::{NoiseModel, ReadoutError};
use std::collections::HashMap;

/// 70 qubits; the reported pair is entangled with **unequal marginals**
/// (p = 0.85 / 0.15) and reported to crossed classical bits.
///
/// The asymmetry is load-bearing. A Bell pair's support {00, 11} and an
/// anti-correlated pair's {01, 10} are both closed under swapping the two
/// classical bits, so neither can detect a bit-order defect — measured, twice.
fn src(n: usize, a: usize, b: usize) -> String {
    let mut s = format!("circuit Wide {{\n    qreg q[{n}]\n    creg c[2]\n\n");
    // The |+> filler must skip the reported pair: applying H to both AFTER the
    // entangling gates maps the state onto a correlated one, whose support is
    // the palindrome this fixture exists to avoid.
    for i in 0..n {
        if i != a && i != b {
            s.push_str(&format!("    apply H on q[{i}]\n"));
        }
    }
    s.push_str(&format!("    apply RY(0.7954) on q[{a}]\n"));
    s.push_str(&format!("    apply CX on q[{a}], q[{b}]\n"));
    s.push_str(&format!("    apply X on q[{b}]\n"));
    s.push_str(&format!(
        "    measure q[{a}] -> c[1]\n    measure q[{b}] -> c[0]\n}}\n"
    ));
    s
}

fn circuit_pair(n: usize, a: usize, b: usize) -> aria_core::ast::nodes::Circuit {
    parse_aria(&src(n, a, b))
        .expect("parse")
        .instantiate("Wide", &[])
        .expect("instantiate")
}

fn circuit(n: usize) -> aria_core::ast::nodes::Circuit {
    circuit_pair(n, 40, 41)
}

fn check(counts: HashMap<omega_core::outcome::Outcome, u32>, shots: u32, lo: f64, what: &str) {
    let wide: Vec<String> = counts
        .keys()
        .filter(|o| o.width() > 2 || o.as_u64().unwrap_or(u64::MAX) > 0b11)
        .map(|o| o.to_bitstring())
        .collect();
    assert!(
        wide.is_empty(),
        "{what}: {} keys exceed the 2-bit creg (e.g. {:?}). Either the backend \
         returned a full-register key, or a creg-keyed one was projected a \
         SECOND time and the bits were re-read at qubit positions.",
        wide.len(),
        wide.iter().take(3).collect::<Vec<_>>()
    );
    let total: u32 = counts.values().sum();
    assert_eq!(total, shots, "{what}: shots went missing");
    let w = counts.keys().next().map(|o| o.width()).unwrap_or(2);
    let f01 = *counts
        .get(&omega_core::outcome::Outcome::from_u64(0b01, w))
        .unwrap_or(&0) as f64
        / total as f64;
    assert!(
        f01 > lo,
        "{what}: |01> took {f01:.3} of the shots, expected ~0.85. Near 0.15 \
         means the two classical bits are swapped; near 0.25 means the key is \
         reading unmeasured qubits. Counts: {counts:?}"
    );
}

fn counts_of(
    r: Result<ExecResult, String>,
    what: &str,
) -> HashMap<omega_core::outcome::Outcome, u32> {
    match r.unwrap_or_else(|e| panic!("{what}: {e}")) {
        ExecResult::Counts(c) => c,
        o => panic!("{what}: {o:?}"),
    }
}

#[test]
fn run_counts_keys_on_the_creg_at_seventy_qubits() {
    let r = run_counts(
        &circuit(70),
        &HashMap::new(),
        400,
        Some(7),
        BackendSel::Mps { chi: 8 },
    );
    check(counts_of(r, "run_counts"), 400, 0.75, "run_counts");
}

#[test]
fn run_counts_noisy_keys_on_the_creg_at_seventy_qubits() {
    let mut model = NoiseModel::default();
    model.readout = ReadoutError::symmetric(0.02);
    let r = run_counts_noisy(
        &circuit(70),
        &HashMap::new(),
        400,
        Some(7),
        BackendSel::Mps { chi: 8 },
        &model,
    );
    // 2% readout on each reported bit softens 0.85 to ~0.82.
    check(
        counts_of(r, "run_counts_noisy"),
        400,
        0.70,
        "run_counts_noisy",
    );
}

/// **Guard the guard: below the cliff nothing moved.**
///
/// The early return fires only when `counts_keyed_on_creg`, which is false for
/// a narrow skip-mode circuit — so the existing projection must still run and
/// still produce a 2-bit key from a 20-qubit register.
#[test]
fn below_the_cliff_the_projection_still_runs() {
    let c = circuit_pair(20, 10, 11);
    let counts = counts_of(
        run_counts(
            &c,
            &HashMap::new(),
            400,
            Some(7),
            BackendSel::Mps { chi: 8 },
        ),
        "narrow",
    );
    check(counts, 400, 0.75, "narrow");
}
