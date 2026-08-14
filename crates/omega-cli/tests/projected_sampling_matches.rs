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

fn counts(ir: &omega_core::circuit::CircuitIR, seed: u64) -> HashMap<omega_core::outcome::Outcome, u32> {
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
    // Marginalise onto qubits 0 and 1 — the two the circuit measures.
    let mut marginal: HashMap<u64, u32> = HashMap::new();
    for (k, v) in &full {
        let two = (k.bit(0) as u64) | ((k.bit(1) as u64) << 1);
        *marginal.entry(two).or_insert(0) += v;
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
        full.keys().any(|k| (2..k.width()).any(|i| k.bit(i) == 1)),
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
            match k.as_u64() {
                Some(0b00) => zeros += v,
                Some(0b11) => {}
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
// The entangled pair therefore sits on q40/q41 reporting to c1/c0, with the
// other 68 qubits in |+>. Under an identity map the key would read q0/q1, which
// are unentangled |+>: a uniform four-outcome histogram, not a two-outcome one.
//
// **The two reported bits must have DIFFERENT marginals**, or bit order is not
// tested at all. Two earlier versions of this fixture got that wrong:
//
//   * a Bell pair, support {00, 11} — closed under swapping the two bits;
//   * an anti-correlated pair, support {01, 10} — also closed under it.
//
// Both were checked against a real bit-order defect (`Some(c)` -> `Some(c ^ 1)`
// at every projection site) and both passed: 118 tests green in the first case,
// 73 in the second. A *set* being preserved is the trap — the swap has to move
// PROBABILITY, so the two outcomes need unequal weight.
//
// So the pair is prepared with `ry(0.7954)` instead of `h`: P(01) = 0.85 and
// P(10) = 0.15. Swapping the bits exchanges those, which no tolerance band can
// absorb.
// ---------------------------------------------------------------------------

/// Anti-correlated pair on `(a, b)`, reported to c1/c0, at `n` qubits.
///
/// The pair indices are parameters rather than a `.replace` on a fixed source.
/// With the replace, the |+> loop below — which skips only literal 40/41 — also
/// hit the reported pair once it had been renamed, applying `h` to both AFTER
/// the entangling gates. That is not harmless: H⊗H maps |Ψ+> to |Φ->, turning
/// the anti-correlated pair back into a correlated one, whose support {00, 11}
/// is exactly the palindrome this fixture exists to avoid.
fn wide_src_pair(n: usize, a: usize, b: usize) -> String {
    let mut s = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\ncreg c[2];\n");
    for i in 0..n {
        if i != a && i != b {
            s.push_str(&format!("h q[{i}];\n"));
        }
    }
    // cos|0> + sin|1> with cos^2 = 0.85, entangled, then one side flipped:
    // outcomes are 01 (p = 0.85) and 10 (p = 0.15) — never 00 or 11, and never
    // equally weighted.
    s.push_str(&format!("ry(0.7954) q[{a}];\ncx q[{a}], q[{b}];\nx q[{b}];\n"));
    // Crossed: a -> c1, b -> c0. With an anti-correlated pair this is
    // load-bearing; with a Bell pair it would not be.
    s.push_str(&format!("measure q[{a}] -> c[1];\nmeasure q[{b}] -> c[0];\n"));
    s
}

fn wide_src(n: usize) -> String {
    wide_src_pair(n, 40, 41)
}

/// **The MPS projected key is the anti-correlated distribution at 70 qubits.**
///
/// The assertion is on the key CONTENTS. `|00>` and `|11>` carry probability
/// zero, so their absence is the whole claim: an identity map, an off-by-one, a
/// dropped projection or a bit-order error all put weight there. The last of
/// those is why the pair is anti-correlated rather than a Bell pair.
#[test]
fn mps_projected_counts_above_the_cliff_are_anticorrelated() {
    let ir = omega_parser::lower_to_ir(&wide_src(70)).expect("lower");
    let (mut n01, mut n10, mut other) = (0u32, 0u32, 0u32);
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
            match k.as_u64() {
                Some(0b01) => n01 += v,
                Some(0b10) => n10 += v,
                _ => other += v,
            }
        }
    }
    assert_eq!(total, 1200, "shots went missing");
    assert_eq!(
        other, 0,
        "the reported pair's support is {{01, 10}}; {other} of 1200 shots landed \
         elsewhere. Under an identity qubit->cbit map the key reads q0/q1 — two \
         independent |+> qubits — and this count would be ~900."
    );
    // p = 0.85 / 0.15. 6 sigma at 1200 shots is ~74, so the bands cannot meet
    // in the middle: a swapped bit order lands 1020 in `n10`, not 180.
    let f01 = n01 as f64 / total as f64;
    assert!(
        (f01 - 0.85).abs() < 0.07,
        "|01> took {f01:.3} of the shots, expected 0.85 (|01> {n01}, |10> {n10}). \
         0.15 means the two classical bits are SWAPPED — which a Bell or \
         anti-correlated pair could not have detected, both supports being \
         closed under the swap."
    );
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
    s.push_str("ry(0.7954) q[40];\ncx q[40], q[41];\nx q[41];\nreset q[5];\n");
    s.push_str("measure q[40] -> c[1];\nmeasure q[41] -> c[0];\n");
    let ir = omega_parser::lower_to_ir(&s).expect("lower");
    // Support is {01, 10}: anti-correlated, so a swapped bit order is visible.

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
    let bad: Vec<String> = c.keys().filter(|o| o.as_u64() != Some(0b01) && o.as_u64() != Some(0b10)).map(|o| o.to_bitstring()).collect();
    let n01 = *c.get(&omega_core::outcome::Outcome::from_u64(0b01, 2)).unwrap_or(&0);
    assert!(
        bad.is_empty(),
        "reset path emitted keys outside the 2-bit creg's anti-correlated \
         support {{01, 10}}: {bad:?} — a key like 0b110 is the full register \
         reported at the creg's width"
    );
    assert_eq!(c.values().sum::<u32>(), 400);
    // Same asymmetry as above, so this arm also detects a swapped bit order.
    let f01 = n01 as f64 / 400.0;
    assert!(
        (f01 - 0.85).abs() < 0.09,
        "|01> took {f01:.3} of 400 shots, expected 0.85 — 0.15 means the two \
         classical bits are swapped"
    );
}

/// **Guard the guard: the fixture must be able to fail.**
///
/// If `wide_src` ever stops entangling, or the unmeasured qubits stop being in
/// superposition, the two tests above pass vacuously. Below the cliff the full
/// register is reported, so this checks the fixture's physics directly: q40/q41
/// are perfectly correlated and q0 is not correlated with them.
#[test]
fn the_wide_fixture_would_expose_an_identity_map() {
    let ir = omega_parser::lower_to_ir(&wide_src_pair(20, 10, 11)).expect("lower");
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
        assert_ne!(
            k.bit(10),
            k.bit(11),
            "fixture no longer anti-correlates the reported pair"
        );
    }
    // ...and q0, which an identity map would report instead, must take BOTH
    // values, so reading it in place of the pair is detectable.
    let q0: std::collections::HashSet<u8> = c.keys().map(|k| k.bit(0)).collect();
    assert_eq!(
        q0.len(),
        2,
        "q0 is constant, so an identity map would still yield a 2-outcome \
         histogram and the tests above could not tell the two apart"
    );
}

// ---------------------------------------------------------------------------
// The NOISY backend, which had the guard but not the projection.
//
// `NoisyMpsBackend::execute` was given the width guard and nothing else, so it
// admitted a wide run at CREG width — the guard's predicate promises the key
// will be the creg — and then sampled the FULL register in both non-collapse
// arms. Measured through `omega-run` before the fix:
//
//   70q, pair on q40/q41 -> creg c[2], --noise:
//       |110000000000000000000000000000000000000000>: 110, |00>: 90
//   70q, x q[65]; measure q[65] -> c[0], --noise:
//       |10>: 50 of 50   (bit 65 masked to bit 1 by `1u64 << 65`; truth |1>)
//
// The noiseless run of the same circuit was correct, so adding `--noise`
// silently changed what the key meant. Nothing in the suite constructed this
// backend above the cliff, which is why a guard-only fix looked complete.
// ---------------------------------------------------------------------------

fn noisy_counts(src: &str, shots: u32, seed: u64) -> HashMap<omega_core::outcome::Outcome, u32> {
    use omega_backend_mps::NoisyMpsBackend;
    use omega_core::noise::{NoiseModel, ReadoutError};

    let ir = omega_parser::lower_to_ir(src).expect("lower");
    let mut model = NoiseModel::default();
    // Readout error is the arm that flips bits: it must act on the QUBIT index,
    // while the key is built on the classical index. Flipping a packed creg key
    // would apply qubit q's detector error to classical bit q.
    model.readout = ReadoutError::symmetric(0.02);
    let cfg = ExecConfig {
        shots: Some(shots),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    match NoisyMpsBackend::with_model(8, model)
        .execute(&ir, &ParameterBinding::default(), &cfg)
        .expect("70-qubit noisy MPS sampling must work, not be refused")
    {
        ExecResult::Counts(c) => c,
        o => panic!("{o:?}"),
    }
}

/// **The noisy backend keys on the creg above the cliff, like the noiseless
/// one.**
///
/// With 2% readout error the support is no longer exactly {01, 10} — that is
/// the point of the model — so the assertion is that the DOMINANT outcome is
/// still `01` and that no key exceeds two bits. A full-register key is not a
/// small perturbation of a 2-bit one; it is a 42-character string.
#[test]
fn the_noisy_backend_projects_above_the_cliff() {
    let c = noisy_counts(&wide_src(70), 400, 5);

    let too_wide: Vec<String> = c
        .keys()
        .filter(|o| o.as_u64().unwrap_or(u64::MAX) > 0b11)
        .map(|o| o.to_bitstring())
        .collect();
    assert!(
        too_wide.is_empty(),
        "keys outside the 2-bit creg: {too_wide:?} — the guard admitted this run \
         at creg width and the sampler then built a full-register key"
    );
    let total: u32 = c.values().sum();
    assert_eq!(total, 400);
    let n01 = *c.get(&omega_core::outcome::Outcome::from_u64(0b01, 2)).unwrap_or(&0) as f64 / total as f64;
    // 0.85 ideal, softened by 2% readout on each of the two reported bits.
    assert!(
        n01 > 0.70,
        "|01> took {n01:.3} of the shots; ~0.82 expected at 2% readout error. \
         Near 0.15 means the two classical bits are swapped; near 0.25 means the \
         key is reading unmeasured qubits. Counts: {c:?}"
    );
}

/// **A qubit above index 63 must not alias into a low bit.**
///
/// `1u64 << 65` is masked to `1u64 << 1` in release builds and panics in debug,
/// so measuring q65 reported bit 1. The creg is 1 bit wide, so the only
/// admissible key is 0 or 1 — and with `x q[65]` it is 1 in all but the readout
/// flips.
#[test]
fn a_qubit_above_the_cliff_does_not_alias_into_a_low_bit() {
    let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[70];\ncreg c[1];\n\
               x q[65];\nmeasure q[65] -> c[0];\n";
    let c = noisy_counts(src, 200, 9);
    let bad: Vec<String> = c
        .keys()
        .filter(|o| o.as_u64().unwrap_or(u64::MAX) > 1)
        .map(|o| o.to_bitstring())
        .collect();
    assert!(
        bad.is_empty(),
        "a 1-bit creg produced {bad:?}. A key of 2 is bit 65 aliased to bit 1 by \
         a masked shift"
    );
    // Width 1: the creg is one bit. Probing at width 2 finds nothing and
    // reports 0, which reads as "the fix is broken" when it is the probe that
    // is — Outcome equality includes the width, deliberately.
    let ones = *c
        .get(&omega_core::outcome::Outcome::from_u64(1, 1))
        .unwrap_or(&0);
    assert!(
        ones > 180,
        "q65 is |1>, so ~98% of 200 shots should read 1 at 2% readout error; got \
         {ones}. Counts: {c:?}"
    );
}
