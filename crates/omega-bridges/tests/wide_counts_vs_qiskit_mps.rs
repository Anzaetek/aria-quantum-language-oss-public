// SPDX-License-Identifier: Apache-2.0
//! **Our wide sampling against Qiskit Aer's matrix-product-state simulator.**
//!
//! `ExecResult::Counts` is keyed by `Outcome` now, so a 1024-qubit shot is
//! representable and the 64-qubit refusal is gone. The tests that landed with
//! that change assert against in-tree oracles and analytic expectations: a GHZ
//! has two outcomes, an 80-qubit fixture differs in exactly one high bit. Those
//! catch truncation and bit-order defects.
//!
//! What they cannot catch is a **shared misreading** — a convention we and our
//! own oracle both get wrong the same way. That needs a second implementation,
//! and above ~30 qubits a dense statevector is not available to be one. Aer's
//! MPS method is: an independent implementation of the same approximation, at
//! the same width.
//!
//! This is the counts analogue of `pauliprop_vs_ppvm.rs`, which closed the same
//! gap for the expectation lane.
//!
//! # Conventions, pinned rather than assumed
//!
//! * **The two sides use OPPOSITE bit orders, on purpose, and this test
//!   converts.** Qiskit reports MSB-first, but the bridge runner deliberately
//!   normalises every backend to **LSB-first** (leftmost character = qubit 0)
//!   so that Perceval, ppvm and Qiskit all speak one convention on the wire.
//!   `Outcome::to_bitstring` is **MSB-first**, because that is what a user
//!   reads. So `bridge_to_msb_first` reverses.
//!
//!   This was not free: comparing the two raw gave a worst per-qubit `P(1)`
//!   difference of **0.0677** — ~17σ at 8000 shots, indistinguishable from a
//!   real defect. Reversing one side gives **0.0040**. The fixtures are
//!   asymmetric precisely so that this shows up: a palindromic support agrees
//!   under either reading, which is how a bit-order defect stayed invisible
//!   twice in this repository.
//! * Aer's bond dimension is left at its default (effectively unbounded) so the
//!   reference is EXACT. Matching a finite bond on both sides would compare two
//!   different approximations, and a disagreement would be ambiguous between
//!   them.
//!
//! # Why the broad circuit is compared on MARGINALS, not on a histogram
//!
//! A full-histogram TVD is only meaningful when the support is small relative
//! to the shot count. The first version of this file used TVD on every fixture
//! and reported **TVD 0.91 on a 70-qubit rotation circuit** — which looked
//! exactly like a serious defect and was nothing of the kind. Measured: that
//! circuit produces **3375 distinct outcomes from 4000 shots**, so two
//! independent samples of it share almost no keys and the TVD is ≈1 whether or
//! not both engines are correct.
//!
//! So: histogram TVD for small-support fixtures (a GHZ has two outcomes), and
//! per-qubit marginals plus pairwise correlations for broad ones. Those
//! converge at 4000 shots and are what actually discriminates.

#![cfg(feature = "bridge-qiskit")]

use omega_backend_mps::MpsBackend;
use omega_core::executor::{Backend as _, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::params::ParameterBinding;
use std::collections::HashMap;
use std::path::PathBuf;

fn venv() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join(".venv-qiskit")
        .join("bin")
        .join("python")
}
fn force_runner() {
    std::env::set_var(
        "OMEGA_BRIDGE_QISKIT_CMD",
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("python")
            .join("omega-bridge-qiskit-runner"),
    );
}

/// Reverse a bridge key (LSB-first) into our MSB-first order.
///
/// Not cosmetic: see the module comment. The bridge wire is LSB-first for every
/// backend by design; `Outcome::to_bitstring` is MSB-first.
fn bridge_to_msb_first(m: HashMap<String, u32>) -> HashMap<String, u32> {
    m.into_iter()
        .map(|(k, v)| (k.chars().rev().collect::<String>(), v))
        .collect()
}

/// Our MPS counts, as MSB-first bitstrings.
fn ours(qasm: &str, shots: u32, seed: u64, chi: usize) -> HashMap<String, u32> {
    let ir = omega_parser::lower_to_ir(qasm).expect("lower");
    let cfg = ExecConfig {
        shots: Some(shots),
        seed: Some(seed),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    match MpsBackend::new(chi)
        .execute(&ir, &ParameterBinding::default(), &cfg)
        .expect("our MPS must run")
    {
        ExecResult::Counts(c) => c
            .into_iter()
            .map(|(o, n)| (o.to_bitstring(), n))
            .collect(),
        o => panic!("{o:?}"),
    }
}

/// Total variation distance between two count histograms.
fn tvd(a: &HashMap<String, u32>, b: &HashMap<String, u32>) -> f64 {
    let ta: f64 = a.values().map(|&v| v as f64).sum();
    let tb: f64 = b.values().map(|&v| v as f64).sum();
    let mut keys: std::collections::HashSet<&String> = a.keys().collect();
    keys.extend(b.keys());
    0.5 * keys
        .iter()
        .map(|k| {
            let pa = *a.get(*k).unwrap_or(&0) as f64 / ta;
            let pb = *b.get(*k).unwrap_or(&0) as f64 / tb;
            (pa - pb).abs()
        })
        .sum::<f64>()
}

/// GHZ over `n` qubits, every qubit measured.
fn ghz(n: usize) -> String {
    let mut s = format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\ncreg c[{n}];\nh q[0];\n"
    );
    for i in 0..n - 1 {
        s.push_str(&format!("cx q[{i}], q[{}];\n", i + 1));
    }
    for i in 0..n {
        s.push_str(&format!("measure q[{i}] -> c[{i}];\n"));
    }
    s
}

/// Low-entanglement but MANY outcomes: small-angle rotations on every qubit and
/// a light CX chain. Unlike GHZ this has a broad distribution, so agreeing on
/// it is evidence about the distribution rather than about two special keys.
fn spread(n: usize) -> String {
    let mut s = format!(
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\ncreg c[{n}];\n"
    );
    for i in 0..n {
        s.push_str(&format!("ry({}) q[{i}];\n", 0.35 + 0.03 * (i % 7) as f64));
    }
    for i in (0..n - 1).step_by(2) {
        s.push_str(&format!("cx q[{i}], q[{}];\n", i + 1));
    }
    for i in 0..n {
        s.push_str(&format!("measure q[{i}] -> c[{i}];\n"));
    }
    s
}

/// Per-qubit `P(1)`, indexed MSB-first as the keys are.
fn marginals(m: &HashMap<String, u32>) -> Vec<f64> {
    let width = m.keys().next().map(|k| k.len()).unwrap_or(0);
    let total: f64 = m.values().map(|&v| v as f64).sum();
    let mut out = vec![0.0; width];
    for (k, v) in m {
        for (i, ch) in k.chars().enumerate() {
            if ch == '1' {
                out[i] += *v as f64;
            }
        }
    }
    out.iter_mut().for_each(|x| *x /= total.max(1.0));
    out
}

/// `<Z_i Z_j>` for adjacent index pairs, MSB-first. Marginals alone cannot see
/// a correlation defect: two engines can agree on every single-qubit
/// probability while disagreeing completely about how the qubits are linked.
fn zz_adjacent(m: &HashMap<String, u32>) -> Vec<f64> {
    let width = m.keys().next().map(|k| k.len()).unwrap_or(0);
    let total: f64 = m.values().map(|&v| v as f64).sum();
    let mut out = vec![0.0; width.saturating_sub(1)];
    for (k, v) in m {
        let b: Vec<i32> = k.chars().map(|c| if c == '1' { -1 } else { 1 }).collect();
        for i in 0..out.len() {
            out[i] += (b[i] * b[i + 1]) as f64 * *v as f64;
        }
    }
    out.iter_mut().for_each(|x| *x /= total.max(1.0));
    out
}

/// **Small support: compare the whole histogram.** A GHZ admits two outcomes,
/// so 4000 shots resolve the distribution and TVD is the right instrument.
#[test]
fn our_mps_agrees_with_qiskit_mps_on_narrow_support_circuits() {
    if !venv().exists() {
        eprintln!("qiskit venv missing — skipping");
        return;
    }
    force_runner();
    let shots = 4000u32;
    let (mut compared, mut worst) = (0usize, 0.0f64);
    let mut skipped: Vec<String> = Vec::new();

    for (name, qasm) in [("ghz-70", ghz(70)), ("ghz-128", ghz(128))] {
        let theirs = match omega_bridges::qiskit::run_mps(&qasm, shots, Some(7), None) {
            Ok(c) => bridge_to_msb_first(c),
            Err(e) => {
                skipped.push(format!("{name} ({e})"));
                continue;
            }
        };
        let mine = ours(&qasm, shots, 7, 64);
        assert_eq!(
            mine.keys().next().map(|k| k.len()),
            theirs.keys().next().map(|k| k.len()),
            "{name}: key widths differ"
        );
        // Both engines are exact here (a GHZ needs bond 2), so the only spread
        // is sampling noise on a fair coin: 6 sigma at 4000 shots is ~0.05.
        let d = tvd(&mine, &theirs);
        worst = worst.max(d);
        compared += 1;
        assert!(
            d < 0.06,
            "{name}: TVD {d:.4}. Both engines are exact at bond 2, so this is a \
             disagreement about the distribution or the bit order, not \
             truncation.\n  ours: {:?}\n  theirs: {:?}",
            top3(&mine),
            top3(&theirs)
        );
    }
    eprintln!("narrow-support: {compared} circuits, worst TVD {worst:.4}; skipped {skipped:?}");
    assert!(compared >= 2, "only {compared} compared (skipped {skipped:?})");
}

/// **Broad support: compare marginals and adjacent correlations.**
///
/// 3375 distinct outcomes from 4000 shots means a histogram comparison measures
/// sampling noise, not agreement. These statistics converge.
#[test]
fn our_mps_agrees_with_qiskit_mps_on_broad_distributions() {
    if !venv().exists() {
        eprintln!("qiskit venv missing — skipping");
        return;
    }
    force_runner();
    let shots = 8000u32;
    let (mut compared, mut worst_m, mut worst_z) = (0usize, 0.0f64, 0.0f64);
    let mut skipped: Vec<String> = Vec::new();

    for (name, qasm) in [("spread-70", spread(70)), ("spread-100", spread(100))] {
        let theirs = match omega_bridges::qiskit::run_mps(&qasm, shots, Some(7), None) {
            Ok(c) => bridge_to_msb_first(c),
            Err(e) => {
                skipped.push(format!("{name} ({e})"));
                continue;
            }
        };
        let mine = ours(&qasm, shots, 7, 64);

        let (mm, mt) = (marginals(&mine), marginals(&theirs));
        assert_eq!(mm.len(), mt.len(), "{name}: widths differ");
        let dm = mm
            .iter()
            .zip(&mt)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let (zm, zt) = (zz_adjacent(&mine), zz_adjacent(&theirs));
        let dz = zm
            .iter()
            .zip(&zt)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        worst_m = worst_m.max(dm);
        worst_z = worst_z.max(dz);
        compared += 1;

        // 8000 shots: a binomial standard error is at most 0.0056, so 6 sigma
        // is ~0.034. The band below leaves room for the worst of ~100 qubits.
        assert!(
            dm < 0.05,
            "{name}: worst per-qubit P(1) difference {dm:.4}.\n  ours[..6]:   {:?}\n  theirs[..6]: {:?}",
            &mm[..6.min(mm.len())],
            &mt[..6.min(mt.len())]
        );
        assert!(
            dz < 0.06,
            "{name}: worst adjacent <Z_i Z_j> difference {dz:.4} — the engines \
             agree on single-qubit probabilities but not on how the qubits are \
             correlated"
        );
    }
    eprintln!(
        "broad: {compared} circuits, worst |ΔP(1)| {worst_m:.4}, worst |Δ<ZZ>| \
         {worst_z:.4}; skipped {skipped:?}"
    );
    assert!(compared >= 2, "only {compared} compared (skipped {skipped:?})");
}

fn top3(m: &HashMap<String, u32>) -> Vec<(String, u32)> {
    let mut v: Vec<_> = m.iter().map(|(k, n)| (k.clone(), *n)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v.truncate(3);
    v
}

/// **The two conventions, pinned on an asymmetric fixture.**
///
/// `x q[0]` on 3 qubits is `100` LSB-first and `001` MSB-first. A GHZ, or any
/// palindromic support, reads the same either way and cannot pin this — which
/// is exactly why a bit-order defect survived twice here.
///
/// If this test starts failing because the bridge changed to MSB-first, the fix
/// is to delete `bridge_to_msb_first`, not to flip this assertion.
#[test]
fn the_bridge_is_lsb_first_and_ours_is_msb_first() {
    if !venv().exists() {
        eprintln!("qiskit venv missing — skipping");
        return;
    }
    force_runner();
    let qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncreg c[3];\n\
                x q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n";
    let raw = omega_bridges::qiskit::run_mps(qasm, 64, Some(1), None).expect("qiskit");
    assert_eq!(
        raw.keys().collect::<Vec<_>>(),
        vec!["100"],
        "the bridge wire is LSB-first: qubit 0 set is the LEFTMOST character"
    );

    let mine = ours(qasm, 64, 1, 8);
    assert_eq!(
        mine.keys().collect::<Vec<_>>(),
        vec!["001"],
        "ours is MSB-first: qubit 0 is the RIGHTMOST character"
    );

    // ...and the conversion reconciles them, which is what every comparison in
    // this file depends on.
    assert_eq!(
        bridge_to_msb_first(raw).keys().collect::<Vec<_>>(),
        mine.keys().collect::<Vec<_>>()
    );
}

/// **Guard the guard: the comparison can fail.**
///
/// If our counts were compared against themselves, or the TVD always read 0,
/// the agreement above would be vacuous. Two genuinely different distributions
/// must produce a TVD the assertion would reject.
#[test]
fn the_tvd_separates_different_distributions() {
    let a: HashMap<String, u32> = [("00".to_string(), 100), ("11".to_string(), 100)]
        .into_iter()
        .collect();
    let b: HashMap<String, u32> = [("01".to_string(), 100), ("10".to_string(), 100)]
        .into_iter()
        .collect();
    assert!((tvd(&a, &a) - 0.0).abs() < 1e-12, "a distribution matches itself");
    assert!(
        tvd(&a, &b) > 0.99,
        "disjoint supports must give TVD ~1, got {}",
        tvd(&a, &b)
    );
}
