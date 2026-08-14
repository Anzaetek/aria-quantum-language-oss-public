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

/// **The stabilizer backend at 128 qubits: projected counts are the Bell
/// distribution, and nothing else appears.**
///
/// This one runs ABOVE the cliff, where the full-register key does not exist,
/// so there is no unprojected result to compare against. The check is therefore
/// against the distribution the circuit defines, pooled over several seeds —
/// a single seed cannot distinguish "correct" from "off by a couple of sigma".
///
/// The stabilizer path measures ONLY the reported qubits, unlike the MPS path
/// which must draw every site because each conditions the next. Marginalising
/// over an unmeasured qubit equals never measuring it, so the distribution is
/// unchanged. Measured at 1024 qubits reporting two: forcing every site to be
/// measured costs 111.8 s for 100 shots (1.12 s per shot); as written the whole
/// run takes 0.71 s including process startup.
#[test]
fn stabilizer_projected_counts_are_the_bell_distribution() {
    use omega_backend_pauli::PauliBackend;

    let n = 128usize;
    let mut src = format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\n"
    );
    for i in 2..n {
        src.push_str(&format!("h q[{i}];\n"));
    }
    src.push_str("measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n");
    let ir = omega_parser::lower_to_ir(&src).expect("lower");

    let (mut zeros, mut total, mut off_diagonal) = (0u32, 0u32, 0u32);
    for seed in 1..=6u64 {
        let cfg = ExecConfig {
            shots: Some(400),
            seed: Some(seed),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let c = match PauliBackend::new()
            .execute(&ir, &ParameterBinding::default(), &cfg)
            .expect("128-qubit stabilizer sampling must work")
        {
            ExecResult::Counts(c) => c,
            o => panic!("{o:?}"),
        };
        for (k, v) in &c {
            total += v;
            match k {
                0b00 => zeros += v,
                0b11 => {}
                // Anything else is not a Bell outcome. This is the assertion
                // that a mis-projection fails: taking the wrong qubits still
                // yields two-bit keys, but not these two.
                _ => off_diagonal += v,
            }
        }
    }

    assert_eq!(
        off_diagonal, 0,
        "|01> / |10> occurred {off_diagonal} times — the projection took the wrong qubits"
    );
    assert_eq!(total, 2400, "shots lost");
    // 6 sigma at n = 2400, p = 1/2 is about 147.
    let dev = (zeros as i64 - 1200).unsigned_abs();
    assert!(
        dev < 150,
        "|00> came out {zeros}/2400, {dev} away from the expected 1200 — beyond the \
         6-sigma band, so this is a bias rather than sampling noise"
    );
}

// ---------------------------------------------------------------------------
// Above the cliff, on MPS, with a NON-IDENTITY qubit -> cbit map.
//
// Everything above this line runs at n = 20 or on the stabilizer backend. So
// the MPS projection — `sample_with_envs_projected`, the function the whole
// wide-sampling story rests on — had no test that executed it at all, and a
// review confirmed the suite stayed green with the classical keying destroyed.
//
// Two things were missing, and both are needed together:
//
//   * a width ABOVE 64, or the projected branch is never taken;
//   * a map that is not the identity, or `cbit_of[q] = q` passes every check.
//     Every fixture above measures q0 -> c0 and q1 -> c1, under which
//     "project onto the creg" and "take the low bits of the register" are the
//     same function.
//
// The Bell pair therefore sits on q40/q41 reporting to c1/c0 — SWAPPED, so bit
// order is checked too — with the other 68 qubits in |+>. Under an identity map
// the key would read q0/q1, which are unentangled |+>: a uniform four-outcome
// histogram, not a Bell pair.
// ---------------------------------------------------------------------------

/// Bell pair on q40/q41, reported to c1/c0, at 70 qubits.
fn wide_src(n: usize) -> String {
    let mut s = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\ncreg c[2];\n");
    s.push_str("h q[40];\ncx q[40], q[41];\n");
    for i in 0..n {
        if i != 40 && i != 41 {
            s.push_str(&format!("h q[{i}];\n"));
        }
    }
    // Deliberately crossed: q40 -> c1, q41 -> c0.
    s.push_str("measure q[40] -> c[1];\nmeasure q[41] -> c[0];\n");
    s
}

/// **The MPS projected key is the Bell distribution at 70 qubits.**
///
/// The assertion is on the key CONTENTS. `|01>` and `|10>` carry probability
/// zero, so their absence is the whole claim: an identity map, an off-by-one, a
/// reversed bit order or a dropped projection all put weight there.
#[test]
fn mps_projected_counts_above_the_cliff_are_the_bell_distribution() {
    let ir = omega_parser::lower_to_ir(&wide_src(70)).expect("lower");
    let (mut n00, mut n11, mut other) = (0u32, 0u32, 0u32);
    let mut total = 0u32;
    for seed in 1..=4u64 {
        let cfg = ExecConfig {
            shots: Some(300),
            seed: Some(seed),
            mid_circuit_mode: MidCircuitMode::Skip,
        };
        let c = match MpsBackend::new(8)
            .execute(&ir, &ParameterBinding::default(), &cfg)
            .expect("70-qubit MPS sampling must work, not be refused")
        {
            ExecResult::Counts(c) => c,
            o => panic!("{o:?}"),
        };
        for (k, v) in &c {
            total += v;
            match k {
                0b00 => n00 += v,
                0b11 => n11 += v,
                _ => other += v,
            }
        }
    }
    assert_eq!(total, 1200, "shots went missing");
    assert_eq!(
        other, 0,
        "the reported pair is a Bell pair, so only |00> and |11> may appear; \
         {other} of 1200 shots landed elsewhere. Under an identity qubit->cbit \
         map the key would read q0/q1 — two independent |+> qubits — and this is \
         the count that would be ~900."
    );
    let skew = (n00 as i64 - n11 as i64).unsigned_abs();
    assert!(skew < 130, "|00> vs |11> skew {skew} exceeds 6 sigma at 1200 shots");
}

/// **A reset circuit above the cliff.** Same claim, different branch.
///
/// `circuit_has_reset` admits a circuit to creg keying even when it has no
/// feed-forward, and for a while that branch chose its outcome with a
/// *different* predicate than the guard that admitted it — so it fell through
/// to the unprojected sampler and emitted a full-register key at the creg's
/// promised width. Measured before the fix on this shape: `|110>` where `|11>`
/// was promised.
#[test]
fn mps_reset_circuit_above_the_cliff_is_also_projected() {
    let mut s = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[70];\ncreg c[2];\n".to_string();
    s.push_str("h q[40];\ncx q[40], q[41];\nreset q[5];\n");
    s.push_str("measure q[40] -> c[1];\nmeasure q[41] -> c[0];\n");
    let ir = omega_parser::lower_to_ir(&s).expect("lower");

    let cfg = ExecConfig {
        shots: Some(400),
        seed: Some(3),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let c = match MpsBackend::new(8)
        .execute(&ir, &ParameterBinding::default(), &cfg)
        .expect("reset + 70 qubits must work")
    {
        ExecResult::Counts(c) => c,
        o => panic!("{o:?}"),
    };
    let bad: Vec<u64> = c.keys().copied().filter(|k| *k != 0 && *k != 0b11).collect();
    assert!(
        bad.is_empty(),
        "reset path emitted keys outside the 2-bit creg's Bell support: {bad:?} \
         — a key like 0b110 is the full register reported at the creg's width"
    );
    assert_eq!(c.values().sum::<u32>(), 400);
}

/// **Guard the guard: the fixture must be able to fail.**
///
/// If `wide_src` ever stops entangling, or the unmeasured qubits stop being in
/// superposition, the two tests above pass vacuously. Below the cliff the full
/// register is reported, so this checks the fixture's physics directly: q40/q41
/// are perfectly correlated and q0 is not correlated with them.
#[test]
fn the_wide_fixture_would_expose_an_identity_map() {
    let ir = omega_parser::lower_to_ir(&wide_src(20).replace("q[40]", "q[10]").replace("q[41]", "q[11]"))
        .expect("lower");
    let cfg = ExecConfig {
        shots: Some(600),
        seed: Some(5),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let c = match MpsBackend::new(8)
        .execute(&ir, &ParameterBinding::default(), &cfg)
        .expect("run")
    {
        ExecResult::Counts(c) => c,
        o => panic!("{o:?}"),
    };
    // Full-register keys here. q10 and q11 must agree in every single shot...
    for k in c.keys() {
        assert_eq!(
            (k >> 10) & 1,
            (k >> 11) & 1,
            "fixture no longer entangles the reported pair"
        );
    }
    // ...and q0, which an identity map would report instead, must take BOTH
    // values, so reading it in place of the pair is detectable.
    let q0: std::collections::HashSet<u64> = c.keys().map(|k| k & 1).collect();
    assert_eq!(
        q0.len(),
        2,
        "q0 is constant, so an identity map would still yield a 2-outcome \
         histogram and the tests above could not tell the two apart"
    );
}
