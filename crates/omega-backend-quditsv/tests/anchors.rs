// SPDX-License-Identifier: Apache-2.0
//! **Leg (i): analytic anchors a qubit engine cannot fake.** (PLAN-QUDIT.md Q2)
//!
//! Every assertion here needs amplitude at level 2 (or 4) to EXIST. A `d = 2`
//! delegation, a wrong stride, or a gate applied to the wrong wire fails on
//! the value — `F_3|0⟩` is not uniform over two amplitudes, `X_3` does not
//! cycle `|2⟩` back to `|0⟩` if there is no `|2⟩`, and `rxy(0,1)` cannot be
//! the identity on a level that is not there. This is what makes the
//! `d = 2` differential (leg ii) falsifiable rather than a tautology.

use num_complex::Complex64;
use omega_backend_quditsv::sim::run;
use omega_backend_quditsv::QuditSvBackend;
use omega_core::circuit::{
    CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit, QuditRegister,
};
use omega_core::executor::{Backend, ExecConfig, ExecResult};
use omega_core::params::ParameterBinding;
use std::f64::consts::PI;

fn op(gate: GateKind, wires: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: wires.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// A circuit on wires of the given dimensions, declared as one register.
fn qudits(dims: &[u32], ops: Vec<GateOp>) -> CircuitIR {
    let mut c = CircuitIR::new(dims.len() as u32, CircuitType::GateBased);
    c.qudit_registers.push(QuditRegister {
        name: "q".into(),
        start: 0,
        dims: dims.to_vec(),
    });
    c.ops = ops;
    c
}

fn amps(c: &CircuitIR) -> Vec<Complex64> {
    run(c, &ParameterBinding::new()).unwrap().amp
}

fn close(a: Complex64, b: Complex64) -> bool {
    (a - b).norm() < 1e-12
}

fn omega(d: u32) -> Complex64 {
    Complex64::from_polar(1.0, 2.0 * PI / d as f64)
}

#[test]
fn fourier_of_the_ground_state_is_uniform_over_all_three_levels() {
    let a = amps(&qudits(&[3], vec![op(GateKind::H, &[0], &[])]));
    assert_eq!(a.len(), 3, "a qutrit has three amplitudes");
    let u = Complex64::new(1.0 / 3f64.sqrt(), 0.0);
    for (k, x) in a.iter().enumerate() {
        assert!(close(*x, u), "level {k}: {x}");
    }
}

#[test]
fn shift_cycles_level_two_back_to_zero() {
    // X_3 X_3 |0⟩ = |2⟩, then X_3 |2⟩ = |0⟩. Only works if |2⟩ exists.
    let two = amps(&qudits(
        &[3],
        vec![op(GateKind::X, &[0], &[]), op(GateKind::X, &[0], &[])],
    ));
    assert!(close(two[2], Complex64::new(1.0, 0.0)), "{two:?}");
    assert!(close(two[0], Complex64::new(0.0, 0.0)) && close(two[1], Complex64::new(0.0, 0.0)));
    let back = amps(&qudits(
        &[3],
        vec![
            op(GateKind::X, &[0], &[]),
            op(GateKind::X, &[0], &[]),
            op(GateKind::X, &[0], &[]),
        ],
    ));
    assert!(close(back[0], Complex64::new(1.0, 0.0)), "{back:?}");
}

#[test]
fn clock_puts_omega_to_the_k_on_level_k() {
    // Z_3 F_3 |0⟩ = (1/√3) Σ ω^k |k⟩.
    let a = amps(&qudits(
        &[3],
        vec![op(GateKind::H, &[0], &[]), op(GateKind::Z, &[0], &[])],
    ));
    let u = 1.0 / 3f64.sqrt();
    for (k, x) in a.iter().enumerate() {
        assert!(close(*x, omega(3).powu(k as u32) * u), "level {k}: {x}");
    }
}

#[test]
fn rxy_on_levels_zero_one_is_the_identity_on_level_two() {
    // |2⟩ prepared by two shifts; rxy(0,1,θ,φ) must leave it alone for any θ, φ.
    let a = amps(&qudits(
        &[3],
        vec![
            op(GateKind::X, &[0], &[]),
            op(GateKind::X, &[0], &[]),
            op(GateKind::Rxy, &[0], &[0.0, 1.0, 1.3, 0.4]),
        ],
    ));
    assert!(close(a[2], Complex64::new(1.0, 0.0)), "{a:?}");
    // …and on |0⟩ it rotates within {0, 1}: mqt.qudits' value for
    // rxy(0, 1, 0.7, 0.3) |0⟩, taken from its simulator in Q2 (see
    // tools/quditsv_xcheck): [0.939373, 0.101333 − 0.327583i, 0].
    let a = amps(&qudits(
        &[3],
        vec![op(GateKind::Rxy, &[0], &[0.0, 1.0, 0.7, 0.3])],
    ));
    assert!(
        (a[0] - Complex64::new(0.939373, 0.0)).norm() < 1e-6,
        "{a:?}"
    );
    assert!(
        (a[1] - Complex64::new(0.101333, -0.327583)).norm() < 1e-6,
        "{a:?}"
    );
    assert!(close(a[2], Complex64::new(0.0, 0.0)), "{a:?}");
}

#[test]
fn csum_adds_the_control_digit_to_the_target_mod_d() {
    // control = |2⟩ (wire 0), target = |0⟩ (wire 1): → target |2⟩.
    // mqt.qudits: index 8 of 9 in ITS big-endian order; ours is wire 0 least
    // significant, so digits (c=2, t=2) sit at 2·1 + 2·3 = 8 as well.
    let a = amps(&qudits(
        &[3, 3],
        vec![
            op(GateKind::X, &[0], &[]),
            op(GateKind::X, &[0], &[]),
            op(GateKind::CSum, &[0, 1], &[]),
        ],
    ));
    assert!(close(a[2 + 2 * 3], Complex64::new(1.0, 0.0)), "{a:?}");
    // and a mixed pair: control d=3 at |2⟩, target d=5 at |4⟩ → 4+2 mod 5 = |1⟩.
    let a = amps(&qudits(
        &[3, 5],
        vec![
            op(GateKind::X, &[0], &[]),
            op(GateKind::X, &[0], &[]),
            op(GateKind::X, &[1], &[]),
            op(GateKind::X, &[1], &[]),
            op(GateKind::X, &[1], &[]),
            op(GateKind::X, &[1], &[]),
            op(GateKind::CSum, &[0, 1], &[]),
        ],
    ));
    // index = digit0 + digit1·d0 = 2 + 1·3
    assert!(close(a[2 + 3], Complex64::new(1.0, 0.0)), "{a:?}");
}

/// Mixed radix `[3, 2, 5]`: a gate on wire 2 must move amplitude in steps
/// of `3·2 = 6`, and leave digits 0 and 1 alone. This is the stride test,
/// and it is the one a `2^n` addressing error fails.
#[test]
fn mixed_radix_strides_address_the_right_wire() {
    let a = amps(&qudits(&[3, 2, 5], vec![op(GateKind::X, &[2], &[])]));
    assert_eq!(a.len(), 30);
    assert!(close(a[6], Complex64::new(1.0, 0.0)), "{a:?}");
    // wire 0 (d=3) then wire 1 (d=2): X X X_2 → digits (1, 1, 0) = 1 + 3 = 4
    let a = amps(&qudits(
        &[3, 2, 5],
        vec![op(GateKind::X, &[0], &[]), op(GateKind::X, &[1], &[])],
    ));
    assert!(close(a[1 + 3], Complex64::new(1.0, 0.0)), "{a:?}");
    // Fourier on the d=5 wire from |0⟩: five equal amplitudes at stride 6.
    let a = amps(&qudits(&[3, 2, 5], vec![op(GateKind::H, &[2], &[])]));
    let u = Complex64::new(1.0 / 5f64.sqrt(), 0.0);
    for k in 0..5 {
        assert!(close(a[6 * k], u), "k={k}: {:?}", a[6 * k]);
    }
    let mass: f64 = a.iter().map(|x| x.norm_sqr()).sum();
    assert!((mass - 1.0).abs() < 1e-12);
}

#[test]
fn the_backend_door_returns_the_full_product_state() {
    let c = qudits(&[3, 2], vec![op(GateKind::H, &[0], &[])]);
    let cfg = ExecConfig {
        shots: None,
        ..Default::default()
    };
    let r = QuditSvBackend::new()
        .execute(&c, &ParameterBinding::new(), &cfg)
        .unwrap();
    let ExecResult::Statevector(sv) = r else {
        panic!("expected a statevector");
    };
    assert_eq!(sv.len(), 6);
    let (out, cert) = QuditSvBackend::new()
        .execute_with_certificate(&c, &ParameterBinding::new(), &cfg)
        .unwrap();
    assert!(matches!(out, ExecResult::Statevector(_)));
    assert!(cert.exact);
    assert_eq!(cert.dropped_mass, 0.0);
    assert_eq!(cert.dims, vec![3, 2]);
    assert_eq!(cert.product_dim, 6);
}
