// SPDX-License-Identifier: Apache-2.0
//! Numeric gate for `aria train`: pure-Rust variational optimization of an Aria
//! ansatz's `symbolic[k]` parameters converges to a known minimum.

use aria_core::ast::{parse_aria, Circuit};
use aria_runtime::{train_expectation, BackendSel, TrainConfig};

fn inline(src: &str, name: &str) -> Circuit {
    parse_aria(src)
        .unwrap_or_else(|e| panic!("parse: {e}"))
        .instantiate(name, &[])
        .unwrap_or_else(|e| panic!("instantiate {name}: {e}"))
}

#[test]
fn single_qubit_ry_minimizes_z_to_minus_one() {
    // RY(θ)|0⟩ ⇒ ⟨Z⟩ = cos θ, minimized to −1 at θ = π.
    let src = "circuit Ry1() {\n  qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0]\n}\n";
    let c = inline(src, "Ry1");
    let cfg = TrainConfig {
        steps: 300,
        lr: 0.2,
        seed: 3,
        init_scale: 1.0,
        ..Default::default()
    };
    let r = train_expectation(&c, "Z0", &cfg, BackendSel::Sim).unwrap();
    assert!(
        r.final_value < -0.999,
        "did not converge to ⟨Z⟩=−1: final {}",
        r.final_value
    );
    assert!(
        r.final_value < r.history[0] - 0.5,
        "objective did not decrease substantially: {} -> {}",
        r.history[0],
        r.final_value
    );
}

#[test]
fn two_qubit_anticorrelation_minimizes_zz() {
    // RY(a)q0, RY(b)q1 ⇒ ⟨Z0 Z1⟩ = cos a · cos b, minimized to −1
    // (one qubit |0⟩, the other |1⟩).
    let src = "circuit Ry2() {\n  qreg q[2]\n  let t = symbolic[2]\n  apply RY(t[0]) on q[0]\n  apply RY(t[1]) on q[1]\n}\n";
    let c = inline(src, "Ry2");
    let cfg = TrainConfig {
        steps: 400,
        lr: 0.2,
        seed: 11,
        init_scale: 1.0,
        ..Default::default()
    };
    let r = train_expectation(&c, "Z0 Z1", &cfg, BackendSel::Sim).unwrap();
    assert!(
        r.final_value < -0.999,
        "did not converge to ⟨Z0 Z1⟩=−1: final {}",
        r.final_value
    );
}

#[test]
fn vqe_ansatz_reaches_h2_ground_state_energy() {
    // The shipped hardware-efficient ansatz, trained against the 2-qubit H₂
    // Hamiltonian, recovers the exact minimum eigenvalue (−1.851199) — a real
    // pure-Rust VQE, no libtorch. Identity term written as `I0` (index skipped).
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/aria/vqe_ansatz.aria");
    let src = std::fs::read_to_string(&path).unwrap();
    let prog = parse_aria(&src).unwrap();
    let circuit = prog.instantiate("VQEAnsatz", &[("n_layers", 2)]).unwrap();
    let h2 = "-0.4804*I0+0.3435*Z0+-0.4347*Z1+0.5716*Z0Z1+0.0910*X0X1+0.0910*Y0Y1";
    let cfg = TrainConfig {
        steps: 600,
        lr: 0.1,
        seed: 7,
        init_scale: 1.0,
        ..Default::default()
    };
    let r = train_expectation(&circuit, h2, &cfg, BackendSel::Sim).unwrap();
    let exact_min = -1.851199;
    assert!(
        (r.final_value - exact_min).abs() < 1e-3,
        "VQE energy {} did not reach H₂ ground state {exact_min}",
        r.final_value
    );
}

#[test]
fn monotone_history_is_nonincreasing_overall() {
    let src = "circuit Ry1() {\n  qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0]\n}\n";
    let c = inline(src, "Ry1");
    let cfg = TrainConfig {
        steps: 200,
        lr: 0.1,
        seed: 5,
        init_scale: 0.8,
        ..Default::default()
    };
    let r = train_expectation(&c, "Z0", &cfg, BackendSel::Sim).unwrap();
    // Gradient descent on a convex-in-basin objective: end well below start.
    assert!(r.final_value <= r.history[0]);
}

#[test]
fn frozen_symbols_keep_their_pinned_values() {
    // Freeze t[0] at 0.3; only t[1] trains. ⟨Z0⟩ stays cos(0.3) no
    // matter what t[1] does (t[1] acts on q1), and ⟨Z1⟩ minimizes.
    let src = "circuit Ry2() {\n  qreg q[2]\n  let t = symbolic[2]\n  apply RY(t[0]) on q[0]\n  apply RY(t[1]) on q[1]\n}\n";
    let c = inline(src, "Ry2");
    let cfg = TrainConfig {
        steps: 300,
        lr: 0.2,
        seed: 11,
        frozen: vec!["t_0".into()],
        init: [("t_0".to_string(), 0.3)].into_iter().collect(),
        ..Default::default()
    };
    let r = train_expectation(&c, "Z0 Z1", &cfg, aria_runtime::BackendSel::Sim).unwrap();
    assert!(
        (r.params["t_0"] - 0.3).abs() < 1e-12,
        "frozen t_0 moved: {}",
        r.params["t_0"]
    );
    // ⟨Z0 Z1⟩ = cos(0.3)·cos(t1) → minimum is −cos(0.3).
    let expected = -(0.3f64).cos();
    assert!(
        (r.final_value - expected).abs() < 1e-3,
        "expected {expected}, got {}",
        r.final_value
    );
}

#[test]
fn freezing_unknown_symbol_is_an_error() {
    let src = "circuit Ry1() {\n  qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0]\n}\n";
    let c = inline(src, "Ry1");
    let cfg = TrainConfig {
        frozen: vec!["nope".into()],
        ..Default::default()
    };
    let err = train_expectation(&c, "Z0", &cfg, aria_runtime::BackendSel::Sim).unwrap_err();
    assert!(err.contains("unknown symbol 'nope'"), "err: {err}");
}

#[test]
fn adam_converges_on_the_single_qubit_problem() {
    let src = "circuit Ry1() {\n  qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0]\n}\n";
    let c = inline(src, "Ry1");
    let cfg = TrainConfig {
        steps: 300,
        lr: 0.1,
        seed: 3,
        optimizer: aria_runtime::Optimizer::adam(),
        ..Default::default()
    };
    let r = train_expectation(&c, "Z0", &cfg, aria_runtime::BackendSel::Sim).unwrap();
    assert!(
        r.final_value < -0.999,
        "Adam did not converge: final {}",
        r.final_value
    );
}
