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
    };
    let r = train_expectation(&c, "Z0", &cfg, BackendSel::Sim).unwrap();
    // Gradient descent on a convex-in-basin objective: end well below start.
    assert!(r.final_value <= r.history[0]);
}
