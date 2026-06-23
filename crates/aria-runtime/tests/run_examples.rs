// SPDX-License-Identifier: Apache-2.0
//! Numeric end-to-end gate for `aria run`: parse → instantiate → lower → execute
//! on the pure-Rust statevector backend, every result checked against a golden
//! value within tolerance. No GUI, numbers only.

use std::collections::HashMap;

use aria_core::ast::{parse_aria, Circuit};
use aria_runtime::{expectation, run_counts, statevector, BackendSel};
use num_complex::Complex64;

const SIM: BackendSel = BackendSel::Sim;
const TOL: f64 = 1e-10;

fn no_binds() -> HashMap<String, f64> {
    HashMap::new()
}

fn example(file: &str, name: &str, ints: &[(&str, i64)]) -> Circuit {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/aria")
        .join(file);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {file}: {e}"));
    let prog = parse_aria(&src).unwrap_or_else(|e| panic!("parse {file}: {e}"));
    prog.instantiate(name, ints)
        .unwrap_or_else(|e| panic!("instantiate {name}: {e}"))
}

fn inline(src: &str, name: &str) -> Circuit {
    parse_aria(src)
        .unwrap_or_else(|e| panic!("parse: {e}"))
        .instantiate(name, &[])
        .unwrap_or_else(|e| panic!("instantiate {name}: {e}"))
}

fn assert_amp(sv: &[Complex64], idx: usize, expect: Complex64) {
    let got = sv[idx];
    assert!(
        (got - expect).norm() < TOL,
        "amp[{idx}]: got {got}, expected {expect}"
    );
}

#[test]
fn bell_statevector_is_exact() {
    let c = example("bell.aria", "Bell", &[]);
    let sv = statevector(&c, &no_binds(), SIM).unwrap();
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert_eq!(sv.len(), 4);
    assert_amp(&sv, 0b00, Complex64::new(s, 0.0));
    assert_amp(&sv, 0b01, Complex64::new(0.0, 0.0));
    assert_amp(&sv, 0b10, Complex64::new(0.0, 0.0));
    assert_amp(&sv, 0b11, Complex64::new(s, 0.0));
}

#[test]
fn bell_zz_expectation_is_one() {
    let c = example("bell.aria", "Bell", &[]);
    let zz = expectation(&c, "Z0 Z1", &no_binds(), SIM).unwrap();
    assert!((zz - 1.0).abs() < TOL, "<Z0 Z1> = {zz}");
}

#[test]
fn bell_counts_are_balanced_and_correlated() {
    let c = example("bell.aria", "Bell", &[]);
    let res = run_counts(&c, &no_binds(), 8192, Some(7), SIM).unwrap();
    // Only |00> and |11> may appear; balanced within 5%.
    if let omega_core::executor::ExecResult::Counts(map) = res {
        let total: u64 = map.values().map(|&v| v as u64).sum();
        assert_eq!(total, 8192);
        let n00 = *map.get(&0b00).unwrap_or(&0) as f64;
        let n11 = *map.get(&0b11).unwrap_or(&0) as f64;
        assert_eq!(n00 + n11, 8192.0, "no |01>/|10> outcomes for a Bell state");
        assert!((n00 / 8192.0 - 0.5).abs() < 0.05, "P(00)={}", n00 / 8192.0);
        assert!((n11 / 8192.0 - 0.5).abs() < 0.05, "P(11)={}", n11 / 8192.0);
    } else {
        panic!("expected counts");
    }
}

#[test]
fn qft_on_zero_is_uniform() {
    let c = example("qft.aria", "QFT", &[("n", 3)]);
    let sv = statevector(&c, &no_binds(), SIM).unwrap();
    assert_eq!(sv.len(), 8);
    let u = 1.0 / (8.0f64).sqrt();
    for (i, amp) in sv.iter().enumerate() {
        assert!(
            (amp - Complex64::new(u, 0.0)).norm() < TOL,
            "QFT|000> amp[{i}] = {amp}, expected {u}"
        );
    }
}

#[test]
fn controlled_phase_is_exact() {
    // X q0; X q1; CP(λ) phases |11> by e^{iλ}. Verifies CP → CU3(0,0,λ).
    let src = |lam: &str| {
        format!(
            "circuit C() {{\n  qreg q[2]\n  apply X on q[0]\n  apply X on q[1]\n  apply CP({lam}) on q[0], q[1]\n}}\n"
        )
    };
    let c = inline(&src("pi/2"), "C");
    let sv = statevector(&c, &no_binds(), SIM).unwrap();
    assert_amp(&sv, 0b11, Complex64::new(0.0, 1.0)); // e^{i pi/2} = i

    let c = inline(&src("pi"), "C");
    let sv = statevector(&c, &no_binds(), SIM).unwrap();
    assert_amp(&sv, 0b11, Complex64::new(-1.0, 0.0)); // e^{i pi} = -1
}

#[test]
fn mps_backend_agrees_with_sim_on_bell() {
    // Same Bell circuit through a different pluggable backend (MPS) must yield
    // the same physics: only |00>/|11>, balanced — no |01>/|10>.
    let c = example("bell.aria", "Bell", &[]);
    let res = run_counts(&c, &no_binds(), 8192, Some(2), BackendSel::Mps).unwrap();
    if let omega_core::executor::ExecResult::Counts(map) = res {
        let n00 = *map.get(&0b00).unwrap_or(&0) as f64;
        let n11 = *map.get(&0b11).unwrap_or(&0) as f64;
        assert_eq!(n00 + n11, 8192.0, "MPS Bell produced |01>/|10> outcomes");
        assert!((n00 / 8192.0 - 0.5).abs() < 0.05);
    } else {
        panic!("expected counts");
    }
}

#[test]
fn ghz_statevector_is_exact() {
    let src = "circuit Ghz() {\n  qreg q[3]\n  apply H on q[0]\n  apply CX on q[0], q[1]\n  apply CX on q[1], q[2]\n}\n";
    let c = inline(src, "Ghz");
    let sv = statevector(&c, &no_binds(), SIM).unwrap();
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert_eq!(sv.len(), 8);
    assert_amp(&sv, 0b000, Complex64::new(s, 0.0));
    assert_amp(&sv, 0b111, Complex64::new(s, 0.0));
    for i in 1..7 {
        assert_amp(&sv, i, Complex64::new(0.0, 0.0));
    }
}

#[cfg(feature = "metal")]
#[test]
fn gpu_metal_agrees_with_sim_on_qft() {
    // The Metal GPU statevector must reproduce the CPU statevector exactly.
    let c = example("qft.aria", "QFT", &[("n", 3)]);
    let cpu = statevector(&c, &no_binds(), BackendSel::Sim).unwrap();
    let gpu = statevector(&c, &no_binds(), BackendSel::Gpu).unwrap();
    assert_eq!(cpu.len(), gpu.len());
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).norm() < 1e-6, "amp[{i}]: sim {a} vs gpu {b}");
    }
}

#[cfg(feature = "tch")]
#[test]
fn tch_backend_agrees_with_sim_on_qft() {
    // libtorch statevector must reproduce the pure-Rust statevector on an
    // asymmetric circuit (QFT|000>), confirming identical qubit/bit ordering.
    let c = example("qft.aria", "QFT", &[("n", 3)]);
    let cpu = statevector(&c, &no_binds(), BackendSel::Sim).unwrap();
    let tch = statevector(&c, &no_binds(), BackendSel::Tch).unwrap();
    assert_eq!(cpu.len(), tch.len());
    for (i, (a, b)) in cpu.iter().zip(tch.iter()).enumerate() {
        assert!((a - b).norm() < 1e-9, "amp[{i}]: sim {a} vs tch {b}");
    }
}
