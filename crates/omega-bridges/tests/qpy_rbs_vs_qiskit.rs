// SPDX-License-Identifier: Apache-2.0
//! The `Rbs → XXPlusYYGate(−2θ, π/2)` spelling, checked through **Qiskit's
//! own gate semantics** rather than against a matrix we typed in.
//!
//! Path: `CircuitIR` → Rust QPY writer → `qpy_to_qasm2` (Qiskit decodes the
//! blob and re-emits QASM2, decomposing `xx_plus_yy` itself) → the qiskit
//! `expectation` mode. Nothing on the omega side interprets the gate after
//! the writer, so what comes back is what Qiskit thinks `XXPlusYYGate` means
//! — compared with the closed form from `gates::rbs`.
//!
//! The observable that matters is `⟨X0 X1⟩ = −sin 2θ`: the sign flips under
//! `+2θ`, and under swapping the qargs. `⟨Z⟩` cannot see either mistake.

#![cfg(feature = "bridge-qiskit")]

use omega_bridges::qiskit::qpy_to_qasm2;
use omega_bridges::qpy::write_qpy_circuit_ir;
use omega_bridges::{expectation_qasm2, Backend, WireObservable};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use smallvec::smallvec;
use std::path::PathBuf;

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}
fn venv_python() -> PathBuf {
    runner_dir().join(".venv-qiskit").join("bin").join("python")
}
macro_rules! skip_without_venv {
    () => {
        if !venv_python().exists() {
            eprintln!(
                "Qiskit venv missing at {} — skipping. Build with \
                 `make -C crates/omega-bridges/python qiskit-venv`.",
                venv_python().display()
            );
            return;
        }
        std::env::set_var(
            "OMEGA_BRIDGE_QISKIT_CMD",
            runner_dir().join("omega-bridge-qiskit-runner"),
        );
    };
}

fn op(gate: GateKind, qubits: &[u32], params: &[ParamExpr]) -> GateOp {
    let mut q = smallvec![];
    for &i in qubits {
        q.push(Qubit(i));
    }
    let mut p = smallvec![];
    for x in params {
        p.push(x.clone());
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

const THETA: f64 = 0.3;

fn through_qiskit(rbs_qargs: &[u32]) -> Vec<f64> {
    let mut ir = CircuitIR::new(2, CircuitType::GateBased);
    ir.add_op(op(GateKind::X, &[0], &[]));
    ir.add_op(op(GateKind::Rbs, rbs_qargs, &[ParamExpr::Concrete(THETA)]));
    let qasm = qpy_to_qasm2(&write_qpy_circuit_ir(&ir)).expect("Qiskit decodes the writer's blob");
    assert!(
        qasm.contains("xx_plus_yy"),
        "the QASM2 Qiskit emits must carry the gate by name (as a custom definition), got:\n{qasm}"
    );
    expectation_qasm2(Backend::Qiskit, &qasm, &[obs("ZI"), obs("IZ"), obs("XX")]).unwrap()
}

fn assert_close(got: &[f64], want: &[f64]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-12,
            "observable {i}: qiskit {g}, closed form {w}"
        );
    }
}

/// `X q0; Rbs(θ) (q0, q1)`. omega's matrix (q0 = MSB) sends |10⟩ to
/// `−sin θ |01⟩ + cos θ |10⟩`, so ⟨Z0⟩ = −cos 2θ, ⟨Z1⟩ = +cos 2θ,
/// ⟨X0 X1⟩ = −sin 2θ.
#[test]
fn rbs_through_qiskit_matches_omega_closed_form_including_the_sign() {
    skip_without_venv!();
    let (c, s) = ((2.0 * THETA).cos(), (2.0 * THETA).sin());
    assert_close(&through_qiskit(&[0, 1]), &[-c, c, -s]);
}

/// Same gate, qargs reversed: now |10⟩ (in q0,q1 order) is the gate's |01⟩,
/// which goes to `cos θ |01⟩ + sin θ |10⟩` — ⟨X0 X1⟩ flips to +sin 2θ. If
/// the writer or Qiskit ignored qarg order, this and the test above could
/// not both pass.
#[test]
fn rbs_on_reversed_qargs_flips_the_sign_sensitive_observable() {
    skip_without_venv!();
    let (c, s) = ((2.0 * THETA).cos(), (2.0 * THETA).sin());
    assert_close(&through_qiskit(&[1, 0]), &[-c, c, s]);
}
