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
fn partial_measurement_counts_are_keyed_over_creg() {
    // Regression: `--shots` used to report the full qubit register no matter
    // what `measure → creg` said. With creg c[1] and only q[0] measured,
    // counts must have exactly the outcomes |0>/|1> with P(|1>) = sin²(0.4636)
    // = 0.2000, regardless of the unmeasured H qubit.
    let src = "circuit Partial() {\n  qreg q[2]\n  creg c[1]\n  apply RY(0.927295218) on q[0]\n  apply H on q[1]\n  measure q[0] -> c[0]\n}\n";
    let c = inline(src, "Partial");
    assert_eq!(aria_runtime::counts_width(&c, &no_binds()), 1);
    for sel in [BackendSel::Sim, BackendSel::Mps] {
        let res = run_counts(&c, &no_binds(), 8192, Some(5), sel).unwrap();
        if let omega_core::executor::ExecResult::Counts(map) = res {
            let total: u64 = map.values().map(|&v| v as u64).sum();
            assert_eq!(total, 8192);
            assert!(
                map.keys().all(|&k| k <= 1),
                "{}: expected creg-width outcomes, got {map:?}",
                sel.name()
            );
            let p1 = *map.get(&1).unwrap_or(&0) as f64 / 8192.0;
            assert!(
                (p1 - 0.2).abs() < 0.02,
                "{}: P(|1>) = {p1}, expected 0.20",
                sel.name()
            );
        } else {
            panic!("expected counts");
        }
    }
}

#[test]
fn counts_width_agrees_with_projection_for_symbolic_params() {
    // Regression: counts_width used to lower the UNBOUND circuit, so a
    // parameter shape that only lowers after binding (sin of a symbol) made
    // it fall back to full-register width while run_counts still projected —
    // creg keys printed as if they were full-register basis states.
    let src = "circuit Partial() {\n  qreg q[2]\n  creg c[1]\n  let t = symbolic[1]\n  apply RY(sin(t[0])) on q[0]\n  apply H on q[1]\n  measure q[0] -> c[0]\n}\n";
    let c = inline(src, "Partial");
    let binds: HashMap<String, f64> = [("t_0".to_string(), 2.0)].into();
    assert_eq!(aria_runtime::counts_width(&c, &binds), 1);
    let res = run_counts(&c, &binds, 4096, Some(5), SIM).unwrap();
    if let omega_core::executor::ExecResult::Counts(map) = res {
        assert!(map.keys().all(|&k| k <= 1), "creg-width keys, got {map:?}");
        // P(|1>) = sin²(sin(2)/2) ≈ 0.1935
        let p1 = *map.get(&1).unwrap_or(&0) as f64 / 4096.0;
        let expect = (2.0f64.sin() / 2.0).sin().powi(2);
        assert!(
            (p1 - expect).abs() < 0.03,
            "P(|1>) = {p1}, expected {expect}"
        );
    } else {
        panic!("expected counts");
    }
}

#[test]
fn measure_into_clbit_64_or_higher_is_a_loud_error() {
    // Counts keys are u64: projecting onto creg bit 64 must error, not
    // overflow the shift (debug panic / silently masked key in release).
    let src = "circuit Wide() {\n  qreg q[1]\n  creg c[65]\n  apply H on q[0]\n  measure q[0] -> c[64]\n}\n";
    let c = inline(src, "Wide");
    let err = run_counts(&c, &no_binds(), 16, Some(1), SIM).unwrap_err();
    assert!(err.contains("64"), "error should name the bad index: {err}");
}

#[test]
fn permuted_measure_targets_follow_the_mapping() {
    // measure q[0] -> c[1] and q[1] -> c[0]: creg bit 1 carries the RY qubit
    // (P=0.2), creg bit 0 carries the H qubit (P=0.5).
    let src = "circuit Perm() {\n  qreg q[2]\n  creg c[2]\n  apply RY(0.927295218) on q[0]\n  apply H on q[1]\n  measure q[0] -> c[1]\n  measure q[1] -> c[0]\n}\n";
    let c = inline(src, "Perm");
    let res = run_counts(&c, &no_binds(), 8192, Some(5), SIM).unwrap();
    if let omega_core::executor::ExecResult::Counts(map) = res {
        let p = |mask: u64| {
            map.iter()
                .filter(|(k, _)| *k & mask != 0)
                .map(|(_, &v)| v as f64)
                .sum::<f64>()
                / 8192.0
        };
        let p_c1 = p(0b10); // RY qubit, sin²(0.4636) = 0.2
        let p_c0 = p(0b01); // H qubit, 0.5
        assert!((p_c1 - 0.2).abs() < 0.02, "P(c1=1) = {p_c1}, expected 0.20");
        assert!((p_c0 - 0.5).abs() < 0.02, "P(c0=1) = {p_c0}, expected 0.50");
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

#[test]
fn pauliprop_backend_matches_sim_expectations() {
    // Pauli propagation is exact on Clifford circuits: Bell ⟨Z0 Z1⟩ = 1 and
    // ⟨Z0⟩ = 0, identical to the statevector backend.
    let c = example("bell.aria", "Bell", &[]);
    for obs in ["Z0 Z1", "Z0"] {
        let pp = expectation(&c, obs, &no_binds(), BackendSel::PauliProp).unwrap();
        let sv = expectation(&c, obs, &no_binds(), SIM).unwrap();
        assert!((pp - sv).abs() < TOL, "<{obs}>: pauliprop {pp} vs sim {sv}");
    }
    // Non-Clifford single-qubit rotation: ⟨Z0⟩ = cos(0.7).
    let c = inline(
        "circuit R() {\n  qreg q[1]\n  apply RY(0.7) on q[0]\n}\n",
        "R",
    );
    let z = expectation(&c, "Z0", &no_binds(), BackendSel::PauliProp).unwrap();
    assert!((z - 0.7f64.cos()).abs() < TOL, "<Z0> = {z}");
}

#[test]
fn pauliprop_backend_rejects_sampling_with_clear_error() {
    // Expectation-only backend: run_counts must fail loudly, not mis-sample.
    let c = example("bell.aria", "Bell", &[]);
    let err = run_counts(&c, &no_binds(), 128, Some(1), BackendSel::PauliProp).unwrap_err();
    assert!(
        err.contains("expectation"),
        "error should say the backend is expectation-only, got: {err}"
    );
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

#[cfg(feature = "cuda")]
#[test]
fn gpu_cuda_agrees_with_sim_on_qft() {
    // The CUDA GPU statevector must reproduce the CPU statevector. Kernels
    // are f32-interleaved, so the tolerance is looser than the exact-f64 CPU
    // path but far tighter than any physically meaningful deviation. QFT|0000>
    // is asymmetric across all amplitudes, so this also pins qubit/bit order.
    let c = example("qft.aria", "QFT", &[("n", 4)]);
    let cpu = statevector(&c, &no_binds(), BackendSel::Sim).unwrap();
    let gpu = statevector(&c, &no_binds(), BackendSel::Gpu).unwrap();
    assert_eq!(cpu.len(), gpu.len());
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).norm() < 1e-5, "amp[{i}]: sim {a} vs gpu {b}");
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
