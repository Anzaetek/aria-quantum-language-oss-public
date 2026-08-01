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
    let res = run_counts(&c, &no_binds(), 8192, Some(2), BackendSel::Mps { chi: 64 }).unwrap();
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
    for sel in [BackendSel::Sim, BackendSel::Mps { chi: 64 }] {
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

/// Asymmetric RBS probe: an entangling H/RY prelude then RBS on adjacent
/// (0,1),(1,2) and non-adjacent (0,2) pairs. The RBS off-diagonal block is
/// sign-antisymmetric — `RBS(θ)` and `RBS(−θ)` differ only in the sign of the
/// |01⟩↔|10⟩ coupling — so an amplitude match against the CPU pins the GPU
/// gate's sign convention AND qubit order, not just its magnitude.
#[cfg(any(feature = "cuda", feature = "metal"))]
const RBS_PROBE_SRC: &str = "circuit RbsProbe() {\n\
    qreg q[3]\n\
    apply H on q[0]\n\
    apply RY(0.9) on q[1]\n\
    apply RBS(0.7) on q[0], q[1]\n\
    apply RBS(1.3) on q[1], q[2]\n\
    apply RY(0.4) on q[2]\n\
    apply RBS(2.1) on q[0], q[2]\n\
}\n";

#[cfg(feature = "cuda")]
#[test]
fn gpu_cuda_agrees_with_sim_on_rbs() {
    // RBS was previously refused on the CUDA statevector backend; it now runs
    // the Givens rotation on span{|01>,|10>} via the generic 2q apply, matching
    // the CPU f64 amplitudes to f32 round-off.
    let c = inline(RBS_PROBE_SRC, "RbsProbe");
    let cpu = statevector(&c, &no_binds(), BackendSel::Sim).unwrap();
    let gpu = statevector(&c, &no_binds(), BackendSel::Gpu).unwrap();
    assert_eq!(cpu.len(), gpu.len());
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).norm() < 1e-5, "amp[{i}]: sim {a} vs gpu {b}");
    }
}

#[cfg(feature = "metal")]
#[test]
fn gpu_metal_agrees_with_sim_on_rbs() {
    // Metal mirror of `gpu_cuda_agrees_with_sim_on_rbs` — RBS now runs on the
    // Metal statevector backend (f32) and must reproduce the CPU amplitudes.
    let c = inline(RBS_PROBE_SRC, "RbsProbe");
    let cpu = statevector(&c, &no_binds(), BackendSel::Sim).unwrap();
    let gpu = statevector(&c, &no_binds(), BackendSel::Gpu).unwrap();
    assert_eq!(cpu.len(), gpu.len());
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).norm() < 1e-6, "amp[{i}]: sim {a} vs gpu {b}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_mps_cuda_agrees_with_sim() {
    // Under a cuda build, `--backend mps` routes bond-compression SVD through
    // the cuSOLVER gesvdj accelerator (CPU fallback if no device). The GPU-SVD
    // MPS statevector must reproduce the exact CPU statevector. QFT|00000> is
    // fully entangling, so the SVD path is genuinely exercised (bond dim > 1).
    let c = example("qft.aria", "QFT", &[("n", 5)]);
    let cpu = statevector(&c, &no_binds(), BackendSel::Sim).unwrap();
    let mps = statevector(&c, &no_binds(), BackendSel::Mps { chi: 64 }).unwrap();
    assert_eq!(cpu.len(), mps.len());
    for (i, (a, b)) in cpu.iter().zip(mps.iter()).enumerate() {
        // Native-f64 gesvdj → the 1e-10 forward tolerance is reachable.
        assert!((a - b).norm() < 1e-10, "amp[{i}]: sim {a} vs mps(gpu) {b}");
    }
}

/// Brickwork of `RY`+`CX` over `n` qubits and `layers` layers — genuinely
/// entangling, so the MPS middle bond climbs toward its cap. For `n = 12` the
/// middle Schmidt rank can never exceed 2^6 = 64 (the MPS bond cap), so there is
/// no truncation and the MPS state equals the exact statevector up to the f32
/// round-off of the Metal contraction.
#[cfg(feature = "metal")]
fn brickwork(n: usize, layers: usize) -> String {
    let mut s = format!("circuit BW() {{\n  qreg q[{n}]\n");
    for i in 0..n {
        s += &format!("  apply H on q[{i}]\n");
    }
    for l in 0..layers {
        let mut i = l % 2;
        while i + 1 < n {
            let theta = 0.3 + 0.1 * l as f64 + 0.05 * i as f64;
            s += &format!("  apply RY({theta}) on q[{i}]\n");
            s += &format!("  apply CX on q[{i}], q[{}]\n", i + 1);
            i += 2;
        }
    }
    s += "}\n";
    s
}

#[cfg(feature = "metal")]
#[test]
fn gpu_mps_metal_agrees_with_sim() {
    // Under a metal build, `--backend mps` routes the two-site θ-contraction
    // through the Metal GPU (SVD stays on CPU). It engages only above the
    // bond-dim threshold (32), so we need a wide, fully-entangling circuit — a
    // 12-qubit RY+CX brickwork drives the middle bond up to the 64 cap, well
    // past 32, with no truncation (rank ≤ 2^6 = 64). The GPU contraction runs in
    // f32 (Apple has no native f64), so the tolerance is the f32 accumulation
    // floor — far tighter than any physically meaningful drift.
    // Lower the GPU bond threshold so the Metal contraction engages on this
    // moderate circuit deterministically (production keeps the tuned 32). The
    // path still runs the real f32 kernel — this only changes *when* it's used.
    unsafe { std::env::set_var("MPS_METAL_MIN_BOND", "4") };
    let src = brickwork(12, 8);
    let c = inline(&src, "BW");
    let before = omega_backend_mps_metal::metal_contraction_count();
    let cpu = statevector(&c, &no_binds(), BackendSel::Sim).unwrap();
    let mps = statevector(&c, &no_binds(), BackendSel::Mps { chi: 64 }).unwrap();
    let ran = omega_backend_mps_metal::metal_contraction_count() - before;
    if ran == 0 {
        eprintln!("skipping assertion: no Metal device (GPU contraction never ran)");
        return;
    }
    eprintln!("metal contractions dispatched: {ran}");
    assert_eq!(cpu.len(), mps.len());
    let max_diff = cpu
        .iter()
        .zip(mps.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    eprintln!("max amplitude diff sim vs mps(metal): {max_diff:.3e}");
    assert!(
        max_diff < 1e-3,
        "sim vs mps(metal) max amplitude diff {max_diff:.3e} exceeds f32 floor"
    );
}

#[cfg(feature = "metal")]
#[test]
fn gpu_pauliprop_metal_agrees_with_sim() {
    // Under a metal build, `--backend pauliprop` offloads the symplectic branch
    // step to the Metal GPU while keeping f64 coefficients on the CPU — so it is
    // exact, not approximate. Force the GPU path on so the accelerator runs even
    // at modest term counts.
    unsafe { std::env::set_var("PAULIPROP_GPU_MIN", "0") };
    // GHZ is Clifford — no branch splits, so it never exercises the GPU branch
    // hook; it's here only as an exact cross-check of the two backends.
    let c = example("ghz.aria", "GHZ", &[]);
    let pp = expectation(&c, "Z0 Z1", &no_binds(), BackendSel::PauliProp).unwrap();
    let sv = expectation(&c, "Z0 Z1", &no_binds(), SIM).unwrap();
    assert!(
        (pp - sv).abs() < 1e-9,
        "GHZ <Z0 Z1>: pauliprop {pp} vs sim {sv}"
    );

    // Non-Clifford circuit: RY/RZ make `branch()` fire, so the Metal hook runs.
    // Read the counter around *this* run and skip the assertion if it never
    // moved — otherwise the test would silently pass on pauliprop-CPU vs
    // statevector-CPU even when the GPU path never executed.
    let c2 = inline(
        "circuit T() {\n  qreg q[4]\n  apply H on q[0]\n  apply CX on q[0], q[1]\n  \
         apply RY(0.6) on q[2]\n  apply CX on q[1], q[2]\n  apply RZ(0.4) on q[3]\n  \
         apply CX on q[2], q[3]\n}\n",
        "T",
    );
    let before = omega_backend_pauliprop_metal::gpu_branch_count();
    let pp2 = expectation(&c2, "Z0 Z3", &no_binds(), BackendSel::PauliProp).unwrap();
    if omega_backend_pauliprop_metal::gpu_branch_count() == before {
        eprintln!("skipping assertion: no Metal device (GPU branch never ran)");
        return;
    }
    let sv2 = expectation(&c2, "Z0 Z3", &no_binds(), SIM).unwrap();
    assert!(
        (pp2 - sv2).abs() < 1e-9,
        "<Z0 Z3>: pauliprop {pp2} vs sim {sv2}"
    );
}

#[test]
fn expectation_with_gradient_matches_analytic_derivative() {
    use aria_runtime::{expectation_with_gradient, GradMethod};

    // RY(t_0) RY(t_1) on |0>, observable Z0. Only t_0 acts on q0, and
    // ⟨Z0⟩ = cos(t_0), so ∂⟨Z0⟩/∂t_0 = −sin(t_0) and ∂/∂t_1 = 0 exactly.
    let src = "circuit G() {\n  qreg q[1]\n  let t = symbolic[2]\n  apply RY(t[0]) on q[0]\n  \
         apply RY(t[1]) on q[0]\n}\n";
    let c = inline(src, "G");
    // Combined angle is t_0 + t_1; ⟨Z0⟩ = cos(t_0 + t_1), so both partials are
    // −sin(t_0 + t_1). Pick values whose sum is a clean angle.
    let binds: HashMap<String, f64> = [("t_0".to_string(), 0.7), ("t_1".to_string(), 0.5)].into();
    let sum = 1.2_f64;

    // Adjoint and parameter-shift must both hit the analytic value/gradient.
    for method in [GradMethod::Adjoint, GradMethod::ParameterShift] {
        let (value, grad) = expectation_with_gradient(&c, "Z0", &binds, SIM, method, None).unwrap();
        assert!(
            (value - sum.cos()).abs() < 1e-9,
            "value {value} vs cos {}",
            sum.cos()
        );
        assert_eq!(grad.len(), 2, "both symbols returned");
        for name in ["t_0", "t_1"] {
            assert!(
                (grad[name] + sum.sin()).abs() < 1e-7,
                "∂/∂{name} = {} vs analytic {}",
                grad[name],
                -sum.sin()
            );
        }
    }

    // `only` restricts the returned gradient to the named subset.
    let (_, only_grad) =
        expectation_with_gradient(&c, "Z0", &binds, SIM, GradMethod::Adjoint, Some(&["t_1"]))
            .unwrap();
    assert_eq!(only_grad.len(), 1, "only t_1 requested");
    assert!(only_grad.contains_key("t_1") && !only_grad.contains_key("t_0"));

    // An unknown binding name is a clear error, not a silent no-op.
    let bad: HashMap<String, f64> = [("nope".to_string(), 1.0)].into();
    let err =
        expectation_with_gradient(&c, "Z0", &bad, SIM, GradMethod::Adjoint, None).unwrap_err();
    assert!(err.contains("unknown symbol 'nope'"), "got: {err}");
}

/// Symbolic RBS circuit for GPU adjoint-gradient parity. `X` first lifts the
/// weight-0 vacuum into span{|01>,|10>} so the Hamming-weight-preserving RBS
/// actually rotates it; a trailing RY gives a second non-trivial partial.
#[cfg(any(feature = "cuda", feature = "metal"))]
const RBS_GRAD_SRC: &str = "circuit RbsGrad() {\n\
    qreg q[2]\n\
    let t = symbolic[2]\n\
    apply X on q[0]\n\
    apply RBS(t[0]) on q[0], q[1]\n\
    apply RY(t[1]) on q[1]\n\
}\n";

#[cfg(feature = "cuda")]
#[test]
fn gpu_cuda_rbs_gradient_agrees_with_sim() {
    // The CUDA adjoint now differentiates RBS (dRBS/dθ via the shared
    // perm_2q_to_cuda(drbs) path); value AND both partials must match the CPU
    // adjoint to f32 round-off.
    use aria_runtime::{expectation_with_gradient, GradMethod};
    let c = inline(RBS_GRAD_SRC, "RbsGrad");
    let binds: HashMap<String, f64> = [("t_0".to_string(), 0.6), ("t_1".to_string(), 0.9)].into();
    let (vc, gc) =
        expectation_with_gradient(&c, "Z1", &binds, SIM, GradMethod::Adjoint, None).unwrap();
    let (vg, gg) =
        expectation_with_gradient(&c, "Z1", &binds, BackendSel::Gpu, GradMethod::Adjoint, None)
            .unwrap();
    assert!((vc - vg).abs() < 1e-4, "value: sim {vc} vs gpu {vg}");
    for name in ["t_0", "t_1"] {
        assert!(
            (gc[name] - gg[name]).abs() < 1e-4,
            "∂/∂{name}: sim {} vs gpu {}",
            gc[name],
            gg[name]
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn gpu_metal_rbs_gradient_agrees_with_sim() {
    // Metal mirror: dRBS/dθ via perm_2q_to_metal(drbs) through the in-place
    // (copy_into) derivative path. Value and both partials match the CPU adjoint.
    use aria_runtime::{expectation_with_gradient, GradMethod};
    let c = inline(RBS_GRAD_SRC, "RbsGrad");
    let binds: HashMap<String, f64> = [("t_0".to_string(), 0.6), ("t_1".to_string(), 0.9)].into();
    let (vc, gc) =
        expectation_with_gradient(&c, "Z1", &binds, SIM, GradMethod::Adjoint, None).unwrap();
    let (vg, gg) =
        expectation_with_gradient(&c, "Z1", &binds, BackendSel::Gpu, GradMethod::Adjoint, None)
            .unwrap();
    assert!((vc - vg).abs() < 1e-5, "value: sim {vc} vs gpu {vg}");
    for name in ["t_0", "t_1"] {
        assert!(
            (gc[name] - gg[name]).abs() < 1e-5,
            "∂/∂{name}: sim {} vs gpu {}",
            gc[name],
            gg[name]
        );
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
