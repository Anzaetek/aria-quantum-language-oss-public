// SPDX-License-Identifier: Apache-2.0
//! The fermionic layer (§3c T1) end to end through the statevector backend.
//!
//! `omega-core/tests/fermion_jw.rs` proves the algebra against a dense oracle
//! but never runs a backend. This file closes the loop: the `cphase` and
//! `givens` constructors are executed by the real engine and read out through
//! Jordan–Wigner observables, with sign-sensitive expected values worked out
//! by hand from the fermionic identities. If the gate the backend applies
//! differed from the doc'd matrix, or the Z-string convention differed from
//! the backend's bit order, these numbers would not come out.
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::*;
use omega_core::fermion::{cphase, givens, FermionicOp};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, qubits: &[u32]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: Default::default(),
        classical_bit: None,
        condition: None,
    }
}

fn circuit(n: u32, ops: Vec<GateOp>) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    c.ops = ops;
    c
}

fn expect(c: &CircuitIR, obs: &Observable) -> f64 {
    obs.validate_qubits(c.num_qubits).unwrap();
    StatevectorBackend::new()
        .expectation(c, &ParameterBinding::new(), obs)
        .unwrap()
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() < 1e-9, "{what}: got {a}, expected {b}");
}

#[test]
fn cphase_is_exp_i_theta_n_p_n_q() {
    // q1 occupied, q0 in (|0⟩ + |1⟩)/√2: the phase lands on |11⟩ only, so
    // ⟨X0⟩ = cos θ and ⟨Y0⟩ = sin θ. With q1 empty nothing happens.
    let theta = 0.7;
    for (p, q) in [(0, 1), (1, 0)] {
        let occupied = circuit(
            2,
            vec![
                op(GateKind::X, &[1]),
                op(GateKind::H, &[0]),
                cphase(p, q, theta),
            ],
        );
        close(
            expect(&occupied, &Observable::x(0)),
            theta.cos(),
            "⟨X0⟩ occupied",
        );
        close(
            expect(&occupied, &Observable::y(0)),
            theta.sin(),
            "⟨Y0⟩ occupied",
        );
        let empty = circuit(2, vec![op(GateKind::H, &[0]), cphase(p, q, theta)]);
        close(expect(&empty, &Observable::x(0)), 1.0, "⟨X0⟩ empty");
        close(expect(&empty, &Observable::y(0)), 0.0, "⟨Y0⟩ empty");
    }
}

#[test]
fn hopping_across_an_occupied_mode_flips_sign_through_the_z_string() {
    // (|1_0⟩ + |1_2⟩)/√2 in modes 0 and 2. a†_0 a_2 + h.c. has eigenvalue +1
    // on it — unless mode 1 is occupied, when the Z string turns it to −1.
    // That sign is the whole content of Jordan–Wigner.
    let hop = FermionicOp::hopping(0, 2, 1.0).jordan_wigner().unwrap();
    let prepare = |occupy_middle: bool| {
        let mut ops = vec![
            op(GateKind::H, &[0]),
            op(GateKind::CX, &[0, 2]),
            op(GateKind::X, &[2]),
        ];
        if occupy_middle {
            ops.insert(0, op(GateKind::X, &[1]));
        }
        circuit(3, ops)
    };
    close(expect(&prepare(false), &hop), 1.0, "mode 1 empty");
    close(expect(&prepare(true), &hop), -1.0, "mode 1 occupied");
}

#[test]
fn givens_rotates_one_particle_between_adjacent_modes_with_the_documented_sign() {
    // exp(θ(a†_0 a_1 − a†_1 a_0)) |1_0⟩ = cos θ |1_0⟩ − sin θ |1_1⟩, so
    // ⟨n_1⟩ = sin²θ and ⟨a†_0 a_1 + h.c.⟩ = −sin 2θ. The second is the one
    // that would silently pass with the wrong sign of θ in the first.
    let theta = 0.61;
    let c = circuit(2, vec![op(GateKind::X, &[0]), givens(0, 1, theta).unwrap()]);
    let n1 = FermionicOp::number(1).jordan_wigner().unwrap();
    let n0 = FermionicOp::number(0).jordan_wigner().unwrap();
    let hop = FermionicOp::hopping(0, 1, 1.0).jordan_wigner().unwrap();
    close(expect(&c, &n1), theta.sin().powi(2), "⟨n_1⟩");
    close(expect(&c, &n0), theta.cos().powi(2), "⟨n_0⟩");
    close(expect(&c, &hop), -(2.0 * theta).sin(), "⟨a†_0 a_1 + h.c.⟩");

    // Reversed qubit order is the reversed rotation.
    let r = circuit(2, vec![op(GateKind::X, &[0]), givens(1, 0, theta).unwrap()]);
    close(expect(&r, &n1), theta.sin().powi(2), "⟨n_1⟩ reversed");
    close(expect(&r, &hop), (2.0 * theta).sin(), "⟨hop⟩ reversed");
}
