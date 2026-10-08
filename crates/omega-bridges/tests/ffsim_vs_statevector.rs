// SPDX-License-Identifier: Apache-2.0
//! ffsim against OUR statevector backend on the SAME `CircuitIR`, read out
//! through the SAME Jordan–Wigner observables — the differential anchor
//! `PLAN-OPEN-20260825.md` §3c.0e item 2 exists for.
//!
//! `ffsim_expectation.rs` checks ffsim against closed forms on two and three
//! qubits. This file checks T1's mapper (`omega_core::fermion`: `givens`,
//! `cphase`, `jordan_wigner`) against ffsim: circuits built only from T1's
//! own constructors, in several particle-number sectors, on observables
//! chosen so that a Z-string sign, a hopping sign, or a phase from `cphase`
//! each flip at least one number. ffsim evolves in the number-conserving
//! basis and never sees omega's matrices; the statevector backend applies
//! omega's matrices and never sees ffsim. Agreement at 1e-9 is the claim
//! that they describe the same unitary and the same operator ordering.
//!
//! Skips out loud without `python/.venv-ffsim`.

#![cfg(feature = "bridge-ffsim")]

use num_complex::Complex64;
use omega_backend_statevector::StatevectorBackend;
use omega_bridges::{ffsim, WireObservable};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend as _, Observable, PauliOp};
use omega_core::fermion::{cphase, givens, FermionicOp, Ladder};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;
use std::path::PathBuf;

fn runner_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}
fn venv_python() -> PathBuf {
    runner_dir().join(".venv-ffsim").join("bin").join("python")
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
        std::env::set_var(
            "OMEGA_BRIDGE_FFSIM_CMD",
            runner_dir().join("omega-bridge-ffsim-runner"),
        );
    };
}

fn x(q: u32) -> GateOp {
    GateOp {
        gate: GateKind::X,
        qubits: smallvec![Qubit(q)],
        params: smallvec![],
        classical_bit: None,
        condition: None,
    }
}
fn rbs(p: u32, q: u32, theta: f64) -> GateOp {
    GateOp {
        gate: GateKind::Rbs,
        qubits: smallvec![Qubit(p), Qubit(q)],
        params: smallvec![ParamExpr::Concrete(theta)],
        classical_bit: None,
        condition: None,
    }
}

/// `Observable` (indexed, sparse) → the bridge wire form (dense, LSB-first:
/// character `i` is qubit `i`), one string per term, coefficient carried.
fn to_wire(obs: &Observable, n: u32) -> WireObservable {
    obs.terms
        .iter()
        .map(|(coeff, paulis)| {
            let mut s = vec![b'I'; n as usize];
            for (q, p) in paulis {
                s[*q as usize] = match p {
                    PauliOp::I => b'I',
                    PauliOp::X => b'X',
                    PauliOp::Y => b'Y',
                    PauliOp::Z => b'Z',
                };
            }
            (String::from_utf8(s).unwrap(), *coeff)
        })
        .collect()
}

/// `i (a†_p a_q − a†_q a_p)` — the current operator. Hermitian, and its
/// Jordan–Wigner image is `(X_p Z… Y_q − Y_p Z… X_q)/2`: sensitive to the
/// phase `cphase` leaves behind, which `hopping` (the real part) is blind to.
fn current(p: u32, q: u32) -> FermionicOp {
    let t = FermionicOp::term(Complex64::i(), vec![Ladder::raise(p), Ladder::lower(q)]);
    let d = t.dagger();
    t + d
}

/// Every observable the comparison reads. On 4 modes.
fn observables() -> Vec<(&'static str, Observable)> {
    let jw = |o: FermionicOp| o.jordan_wigner().unwrap();
    let mut v = vec![];
    for p in 0..4 {
        v.push(("n_p", jw(FermionicOp::number(p))));
    }
    for (p, q) in [(0, 1), (1, 2), (2, 3)] {
        v.push(("hop adjacent", jw(FermionicOp::hopping(p, q, 1.0))));
        v.push(("current adjacent", jw(current(p, q))));
    }
    // Z string over one and over two modes.
    v.push(("hop 0-2 (Z on 1)", jw(FermionicOp::hopping(0, 2, 1.0))));
    v.push(("hop 0-3 (Z on 1,2)", jw(FermionicOp::hopping(0, 3, 0.5))));
    v.push(("current 1-3 (Z on 2)", jw(current(1, 3))));
    v.push(("n_0 n_1", jw(FermionicOp::interaction(0, 1, 1.0))));
    v.push(("n_1 n_3", jw(FermionicOp::interaction(1, 3, 1.0))));
    // One many-term "Hamiltonian", so the wire's per-term coefficients and the
    // sum are exercised together, not one Pauli string at a time.
    let h = FermionicOp::hopping(0, 1, -1.0)
        + FermionicOp::hopping(1, 2, -1.0)
        + FermionicOp::hopping(2, 3, -1.0)
        + FermionicOp::interaction(0, 1, 0.5)
        + FermionicOp::interaction(2, 3, 0.5)
        + FermionicOp::number(2).scale(Complex64::new(-0.3, 0.0));
    v.push(("Hubbard-like sum", jw(h)));
    v
}

/// A circuit from T1's constructors only, deep enough that every mode and
/// both `givens` orientations take part, on top of the given occupation.
fn layered(occupied: &[u32]) -> CircuitIR {
    let mut ir = CircuitIR::new(4, CircuitType::GateBased);
    for &m in occupied {
        ir.add_op(x(m));
    }
    ir.add_op(givens(0, 1, 0.3).unwrap());
    ir.add_op(givens(2, 3, -0.45).unwrap());
    ir.add_op(cphase(1, 2, 0.7));
    ir.add_op(givens(1, 2, 0.52).unwrap());
    ir.add_op(cphase(0, 3, -0.9));
    ir.add_op(givens(3, 2, 0.21).unwrap()); // reversed qubit order
    ir.add_op(cphase(0, 1, 1.3));
    ir.add_op(givens(1, 0, -0.8).unwrap()); // reversed, negative
    ir
}

fn statevector(ir: &CircuitIR, obs: &[Observable]) -> Vec<f64> {
    let sv = StatevectorBackend::new();
    obs.iter()
        .map(|o| {
            o.validate_qubits(ir.num_qubits).unwrap();
            sv.expectation(ir, &ParameterBinding::new(), o).unwrap()
        })
        .collect()
}

fn compare(ir: &CircuitIR, named: &[(&str, Observable)], what: &str) {
    let obs: Vec<Observable> = named.iter().map(|(_, o)| o.clone()).collect();
    let wires: Vec<WireObservable> = obs.iter().map(|o| to_wire(o, ir.num_qubits)).collect();
    let ours = statevector(ir, &obs);
    let theirs = ffsim::expectation(ir, &wires).unwrap();
    assert_eq!(ours.len(), theirs.len());
    let mut worst = 0.0f64;
    for (i, ((name, _), (a, b))) in named.iter().zip(ours.iter().zip(&theirs)).enumerate() {
        let d = (a - b).abs();
        worst = worst.max(d);
        assert!(
            d < 1e-9,
            "{what}, observable {i} ({name}): statevector {a}, ffsim {b}, |Δ| = {d:e}"
        );
    }
    eprintln!(
        "{what}: {} observables agree, worst |Δ| = {worst:e}",
        named.len()
    );
}

#[test]
fn t1_layer_agrees_with_ffsim_in_every_sector() {
    skip_without_venv!();
    let named = observables();
    for occupied in [&[0][..], &[0, 2][..], &[1, 2][..], &[0, 1, 3][..]] {
        compare(
            &layered(occupied),
            &named,
            &format!("occupied {occupied:?}"),
        );
    }
}

/// The empty sector is not a degenerate case for the bridge: ffsim's
/// `nelec = 0` basis has one state, and every observable must still come
/// back — numbers all 0 except the identity-bearing ones.
#[test]
fn the_empty_sector_still_round_trips() {
    skip_without_venv!();
    compare(&layered(&[]), &observables(), "occupied []");
}

/// `Rbs` on NON-adjacent qubits. omega's `givens` refuses `(0, 2)` because
/// the fermionic rotation would carry a Z on mode 1; a bare `Rbs(0,2)` is
/// the QUBIT gate, no Z string. ffsim reaches the same reading through its
/// swap-defect path (`_apply_xx_plus_yy` → `_apply_swap_defect`): it
/// simulates the Qiskit gate, not the fermionic rotation. With mode 1
/// occupied the two readings differ by a sign, so this pins that both sides
/// take the qubit one — and pins the closed form, which is the fermionic
/// one with the sign flipped:
///
/// ```text
/// X0; X1; Rbs(θ)(0,2)  on |1_0 1_1 0_2⟩ → cos θ |110⟩ − sin θ |011⟩
/// ⟨X0 X2⟩ = ⟨Y0 Y2⟩ = −sin 2θ           (qubit pair, mode 1 untouched)
/// ⟨a†_0 a_2 + h.c.⟩ = ⟨(X0 Z1 X2 + Y0 Z1 Y2)/2⟩ = +sin 2θ   (Z1 = −1)
/// ```
#[test]
fn a_non_adjacent_rbs_is_the_qubit_gate_on_both_sides() {
    skip_without_venv!();
    let theta = 0.4;
    let mut ir = CircuitIR::new(3, CircuitType::GateBased);
    ir.add_op(x(0));
    ir.add_op(x(1));
    ir.add_op(rbs(0, 2, theta));
    let jw = |o: FermionicOp| o.jordan_wigner().unwrap();
    let named = vec![
        ("n_0", jw(FermionicOp::number(0))),
        ("n_1", jw(FermionicOp::number(1))),
        ("n_2", jw(FermionicOp::number(2))),
        ("hop 0-2 (Z on 1)", jw(FermionicOp::hopping(0, 2, 1.0))),
        (
            "X0 X2 (plain Pauli)",
            Observable {
                terms: vec![(1.0, vec![(0, PauliOp::X), (2, PauliOp::X)])],
            },
        ),
    ];
    compare(&ir, &named, "non-adjacent Rbs");
    let got = statevector(
        &ir,
        &named.iter().map(|(_, o)| o.clone()).collect::<Vec<_>>(),
    );
    let s = (2.0 * theta).sin();
    let want = [theta.cos().powi(2), 1.0, theta.sin().powi(2), s, -s];
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-9,
            "closed form, observable {i} ({}): got {g}, want {w}",
            named[i].0
        );
    }
}
