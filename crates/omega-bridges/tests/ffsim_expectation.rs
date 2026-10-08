// SPDX-License-Identifier: Apache-2.0
//! The ffsim `expectation` bridge — the fermionic differential anchor
//! (`PLAN-OPEN-20260825.md` §3c.0e item 2).
//!
//! Numbers are checked against closed forms derived from `gates::rbs` /
//! `gates::cu3` by hand (worked in the doc comments), on observables chosen
//! so that the sign of θ, the sign of λ, and the qarg order each flip at
//! least one value. Agreement here means ffsim's evolution in the
//! number-conserving basis and omega's dense matrices describe the same
//! unitary — which is the claim T1 needed an outside witness for.
//!
//! Skips out loud without `python/.venv-ffsim`. The refusal tests need no
//! venv and always run.

#![cfg(feature = "bridge-ffsim")]

use omega_bridges::ffsim;
use omega_bridges::{expectation_qasm2, run_qasm2, Backend, BridgeError, WireObservable};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use smallvec::smallvec;
use std::path::PathBuf;

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}
fn venv_python() -> PathBuf {
    runner_dir().join(".venv-ffsim").join("bin").join("python")
}
fn force_env() {
    std::env::set_var(
        "OMEGA_BRIDGE_FFSIM_CMD",
        runner_dir().join("omega-bridge-ffsim-runner"),
    );
}
macro_rules! skip_without_venv {
    () => {
        if !venv_python().exists() {
            eprintln!(
                "ffsim venv missing at {} — skipping. Build with \
                 `make -C crates/omega-bridges/python ffsim-venv`.",
                venv_python().display()
            );
            return;
        }
        force_env();
    };
}

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    let mut q = smallvec![];
    for &i in qubits {
        q.push(Qubit(i));
    }
    let mut p = smallvec![];
    for &x in params {
        p.push(ParamExpr::Concrete(x));
    }
    GateOp {
        gate,
        qubits: q,
        params: p,
        classical_bit: None,
        condition: None,
    }
}
fn obs(s: &str) -> WireObservable {
    vec![(s.to_string(), 1.0)]
}
fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-12,
            "observable {i}: ffsim {g}, closed form {w}"
        );
    }
}

/// `X q0; Rbs(θ)(q0,q1)`: |10⟩ → `−sin θ|01⟩ + cos θ|10⟩`, so
/// ⟨Z0⟩ = −cos 2θ, ⟨Z1⟩ = cos 2θ, ⟨X0X1⟩ = −sin 2θ (sign-sensitive).
#[test]
fn one_electron_rbs_matches_the_closed_form() {
    skip_without_venv!();
    let theta = 0.3;
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.add_op(op(GateKind::X, &[0], &[]));
    ir.add_op(op(GateKind::Rbs, &[0, 1], &[theta]));
    let got = ffsim::expectation(&ir, &[obs("ZI"), obs("IZ"), obs("XX")]).unwrap();
    let (c, s) = ((2.0 * theta).cos(), (2.0 * theta).sin());
    assert_close(&got, &[-c, c, -s]);
}

/// Two electrons, all three fermionic primitives, worked by hand:
///
/// ```text
/// X q0; X q1                      |110⟩                       (q0 q1 q2)
/// Rbs(θ) (q1,q2)                  −sin θ|101⟩ + cos θ|110⟩
/// CU3(0,0,λ) (q0,q1) = CPhase(λ)  −sin θ|101⟩ + cos θ e^{iλ}|110⟩
/// Swap (q0,q2)                    −sin θ|101⟩ + cos θ e^{iλ}|011⟩
/// ```
///
/// ⟨Z0⟩ = cos 2θ, ⟨Z1⟩ = −cos 2θ, ⟨Z2⟩ = −1,
/// ⟨X0X1⟩ = −sin 2θ cos λ (sees θ's sign and λ's magnitude),
/// ⟨Y0X1⟩ = +sin 2θ sin λ (sees λ's sign).
#[test]
fn two_electrons_rbs_cphase_swap_match_the_closed_form() {
    skip_without_venv!();
    let (theta, lambda) = (0.3, 0.7);
    let mut ir = CircuitIR::new(3, CircuitType::GateBased);
    ir.add_op(op(GateKind::X, &[0], &[]));
    ir.add_op(op(GateKind::X, &[1], &[]));
    ir.add_op(op(GateKind::Rbs, &[1, 2], &[theta]));
    ir.add_op(op(GateKind::CU3, &[0, 1], &[0.0, 0.0, lambda]));
    ir.add_op(op(GateKind::Swap, &[0, 2], &[]));
    let got = ffsim::expectation(
        &ir,
        &[obs("ZII"), obs("IZI"), obs("IIZ"), obs("XXI"), obs("YXI")],
    )
    .unwrap();
    let (c, s) = ((2.0 * theta).cos(), (2.0 * theta).sin());
    assert_close(&got, &[c, -c, -1.0, -s * lambda.cos(), s * lambda.sin()]);
}

#[test]
fn a_hadamard_is_refused_as_cannot_express() {
    skip_without_venv!();
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.add_op(op(GateKind::H, &[0], &[]));
    match ffsim::expectation(&ir, &[obs("ZI")]) {
        Err(BridgeError::CannotExpress(Backend::Ffsim, msg)) => {
            assert!(msg.contains("ffsim-unsupported-gate"), "{msg}")
        }
        other => panic!("H must be a typed refusal, got {other:?}"),
    }
}

#[test]
fn an_x_after_the_occupation_layer_is_refused() {
    skip_without_venv!();
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.add_op(op(GateKind::X, &[0], &[]));
    ir.add_op(op(GateKind::Rbs, &[0, 1], &[0.3]));
    ir.add_op(op(GateKind::X, &[1], &[]));
    assert!(matches!(
        ffsim::expectation(&ir, &[obs("ZI")]),
        Err(BridgeError::CannotExpress(Backend::Ffsim, _))
    ));
}

/// A photonic gate the Rust QPY writer cannot spell is refused BEFORE the
/// writer is asked (it would panic), and without needing a venv.
#[test]
fn a_gate_the_writer_cannot_spell_is_refused_before_the_subprocess() {
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.add_op(op(GateKind::PhaseShifter, &[0], &[0.1]));
    match ffsim::expectation(&ir, &[obs("ZI")]) {
        Err(BridgeError::CannotExpress(Backend::Ffsim, msg)) => {
            assert!(msg.contains("PhaseShifter"), "{msg}")
        }
        other => panic!("expected CannotExpress, got {other:?}"),
    }
}

#[test]
fn counts_mode_is_a_capability_gap_not_an_install_problem() {
    assert!(matches!(
        run_qasm2(Backend::Ffsim, "OPENQASM 2.0;", 10, None),
        Err(BridgeError::CannotExpress(Backend::Ffsim, _))
    ));
}

#[test]
fn qasm2_expectation_points_at_the_real_entry_point() {
    match expectation_qasm2(Backend::Ffsim, "OPENQASM 2.0;", &[obs("Z")]) {
        Err(BridgeError::CannotExpress(Backend::Ffsim, msg)) => {
            assert!(msg.contains("ffsim::expectation"), "{msg}")
        }
        other => panic!("expected CannotExpress, got {other:?}"),
    }
}
