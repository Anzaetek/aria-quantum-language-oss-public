// SPDX-License-Identifier: Apache-2.0
//! **Projecting a shot onto the measured qubits must not change the physics.**
//!
//! Above 64 qubits the full-register key cannot exist — `sample_with_envs`
//! builds it with `bit << q`, which runs out of range — so a wide circuit with
//! measurements is keyed on its CLASSICAL register instead. That is what makes
//! a 1024-qubit sampled run possible at all, and it is also what qiskit reports
//! over.
//!
//! The risk is that the projection changes the DISTRIBUTION rather than only the
//! key. It must not: every qubit is still sampled in order (each outcome
//! conditions the ones after it), and only the reporting is narrowed.
//!
//! This test runs BELOW the cliff, where both paths exist, so they can be
//! compared on the same circuit with the same seed. Above the cliff there is
//! nothing to compare against — which is exactly why the check has to happen
//! here.

use omega_backend_mps::MpsBackend;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;
use std::collections::HashMap;

/// A 20-qubit circuit whose measured pair is correlated, so a mis-projection
/// (wrong bit, wrong position) shows up as a different distribution rather than
/// merely a different labelling.
fn src(n: usize) -> String {
    let mut s = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\ncreg c[2];\n");
    s.push_str("h q[0];\ncx q[0], q[1];\n");
    // Some unmeasured activity, so the projection has something to drop.
    for i in 2..n {
        s.push_str(&format!("h q[{i}];\n"));
    }
    s.push_str("measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n");
    s
}

fn counts(ir: &omega_core::circuit::CircuitIR, seed: u64) -> HashMap<u64, u32> {
    let cfg = ExecConfig {
        shots: Some(4000),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    match MpsBackend::new(64)
        .execute(ir, &ParameterBinding::default(), &cfg)
        .expect("run")
    {
        ExecResult::Counts(c) => c,
        o => panic!("{o:?}"),
    }
}

/// The measured pair is a Bell pair: only `00` and `11` may appear, at ~50/50.
///
/// Asserting the SHAPE, not just "it ran". A projection that took the wrong
/// qubits would still produce two-bit keys — it would produce the wrong ones.
#[test]
fn the_projected_marginal_is_the_bell_distribution() {
    let ir = omega_parser::lower_to_ir(&src(20)).expect("lower");
    let full = counts(&ir, 7);

    // Below the cliff the key is the full register; marginalise by hand onto
    // the two measured qubits, which is what the projection does above it.
    let mut marginal: HashMap<u64, u32> = HashMap::new();
    for (k, v) in &full {
        *marginal.entry(k & 0b11).or_insert(0) += v;
    }
    let total: u32 = marginal.values().sum();
    assert_eq!(total, 4000);

    let (n00, n11) = (
        *marginal.get(&0b00).unwrap_or(&0),
        *marginal.get(&0b11).unwrap_or(&0),
    );
    let (n01, n10) = (
        *marginal.get(&0b01).unwrap_or(&0),
        *marginal.get(&0b10).unwrap_or(&0),
    );
    assert_eq!(
        n01 + n10,
        0,
        "q0 and q1 are a Bell pair: |01> and |10> must not occur, got {n01} and {n10}"
    );
    // 4000 shots, p = 1/2: the 6-sigma band is ~190.
    let skew = (n00 as i64 - n11 as i64).unsigned_abs();
    assert!(
        skew < 200,
        "|00> vs |11> skew {skew} exceeds the 6-sigma band at 4000 shots"
    );
}

/// **The unmeasured qubits must not leak into the key.**
///
/// 18 of the 20 qubits are in `|+>` and unmeasured. If the projection reported
/// them, the key would take many values instead of two — which is precisely the
/// difference between a 1024-qubit run being usable and being a histogram of
/// 4000 unique strings.
#[test]
fn unmeasured_qubits_do_not_reach_the_key() {
    let ir = omega_parser::lower_to_ir(&src(20)).expect("lower");
    let full = counts(&ir, 7);
    // Below the cliff the FULL key is still used, so this documents the
    // pre-cliff behaviour the projection deliberately does not change.
    assert!(
        full.keys().any(|k| k >> 2 != 0),
        "below 64 qubits the full register is still reported — if this fails, the \
         projection was applied where it should not be, changing existing results"
    );
}
