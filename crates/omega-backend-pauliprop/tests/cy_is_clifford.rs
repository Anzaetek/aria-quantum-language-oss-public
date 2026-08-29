// SPDX-License-Identifier: Apache-2.0
//! **`CY` conjugates as a Clifford, and the phase is right.**
//!
//! `CY` used to fall into pauliprop's catch-all and be refused as unsupported.
//! That cost a whole gate for no reason: `CY` is Clifford, so it maps one Pauli
//! to exactly one Pauli — no branching, no truncation, no `dropped_mass`. The
//! alternative that suggests itself, decomposing it into `S`/`CX`/`Sdg`, would
//! be three ops instead of one and is easy to get backwards (`CY = (I⊗S)·CX·
//! (I⊗S†)`, not the reverse).
//!
//! # Why this test is numerical rather than a table comparison
//!
//! The image table was derived by hand, and one of its four entries carries a
//! phase: this encoding stores `Y` as the bit pair `(x=1, z=1)`, which is
//! literally `X·Z = −i·Y`, so the `Xc → Xc⊗Y` image needs a compensating
//! factor of `i`. A sign or factor error there produces a **wrong expectation
//! value**, not a crash, and it would agree with the correct answer on every
//! observable that happens to avoid the affected term.
//!
//! So the check is against an independent engine — the dense statevector
//! backend — over observables chosen to include the ones a phase error would
//! corrupt.

use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_statevector::sim::StatevectorBackend;
use omega_core::circuit::*;
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn g(kind: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

fn obs(paulis: &[(u32, PauliOp)]) -> Observable {
    Observable {
        terms: vec![(1.0, paulis.to_vec())],
    }
}

/// Circuits that put real amplitude on every Pauli axis before the `CY`, so a
/// wrong image cannot hide behind a zero coefficient.
fn circuits() -> Vec<(&'static str, CircuitIR)> {
    let mut out = Vec::new();

    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[0], &[]));
    c.add_op(g(GateKind::CY, &[0, 1], &[]));
    out.push(("h;cy", c));

    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[0], &[]));
    c.add_op(g(GateKind::S, &[0], &[]));
    c.add_op(g(GateKind::H, &[1], &[]));
    c.add_op(g(GateKind::CY, &[0, 1], &[]));
    out.push(("h;s;h;cy", c));

    // Rotations off the axes: the coefficients are then generic, so a phase
    // error cannot cancel numerically.
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    c.add_op(g(GateKind::Ry, &[0], &[0.7]));
    c.add_op(g(GateKind::Rx, &[1], &[1.1]));
    c.add_op(g(GateKind::Rz, &[2], &[0.3]));
    c.add_op(g(GateKind::CY, &[0, 1], &[]));
    c.add_op(g(GateKind::CY, &[1, 2], &[]));
    out.push(("ry;rx;rz;cy;cy", c));

    // Reversed operand order — the control/target images are different, so a
    // transposed table passes the symmetric cases and fails this one.
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(g(GateKind::Ry, &[0], &[0.9]));
    c.add_op(g(GateKind::Ry, &[1], &[0.4]));
    c.add_op(g(GateKind::CY, &[1, 0], &[]));
    out.push(("ry;ry;cy(1,0)", c));

    out
}

/// **Every Pauli observable must agree with the dense statevector backend.**
#[test]
fn cy_expectations_match_the_statevector_backend() {
    let pp = PauliPropBackend::new();
    let sv = StatevectorBackend::new();
    let params = ParameterBinding::default();
    const TOL: f64 = 1e-9;

    let mut checked = 0;
    let mut nonzero = 0;
    for (label, circuit) in circuits() {
        let n = circuit.num_qubits;
        // Every single-qubit Pauli, plus every weight-2 pair — the Y ones are
        // what a phase error in the Xc image corrupts.
        let mut observables: Vec<Vec<(u32, PauliOp)>> = Vec::new();
        for q in 0..n {
            for p in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
                observables.push(vec![(q, p)]);
            }
        }
        for a in 0..n {
            for b in (a + 1)..n {
                for pa in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
                    for pb in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
                        observables.push(vec![(a, pa), (b, pb)]);
                    }
                }
            }
        }

        for paulis in observables {
            let o = obs(&paulis);
            let want = sv
                .expectation(&circuit, &params, &o)
                .expect("statevector reference");
            let got = pp
                .expectation(&circuit, &params, &o)
                .unwrap_or_else(|e| panic!("{label}: pauliprop must handle CY: {e}"));
            assert!(
                (got - want).abs() < TOL,
                "{label}, observable {paulis:?}: pauliprop {got} vs statevector \
                 {want} (delta {:.3e})",
                (got - want).abs()
            );
            checked += 1;
            if want.abs() > 1e-6 {
                nonzero += 1;
            }
        }
    }

    // Non-degeneracy: agreement on a set of zeros proves nothing, and a phase
    // error shows up only on terms that are actually populated.
    // The corpus yields 81 comparisons; the bar exists to catch a future
    // edit that silently empties it, not to assert a particular count.
    assert!(checked >= 80, "only {checked} observables compared");
    assert!(
        nonzero > 10,
        "only {nonzero} observables had a non-zero value — the corpus is too \
         degenerate to catch a phase error"
    );
}

/// `CY` must cost exactly one Pauli per Pauli. If it ever became a
/// decomposition into rotations it would branch, and this backend's whole
/// advantage on Clifford circuits would quietly disappear.
#[test]
fn cy_never_branches() {
    let pp = PauliPropBackend::new();
    let params = ParameterBinding::default();
    // A wide Clifford chain of CY: exact and cheap, or not Clifford handling.
    let n = 200;
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for q in 0..n {
        c.add_op(g(GateKind::H, &[q], &[]));
    }
    for q in 0..n - 1 {
        c.add_op(g(GateKind::CY, &[q, q + 1], &[]));
    }
    let o = obs(&[(0, PauliOp::Z)]);
    let v = pp
        .expectation(&c, &params, &o)
        .expect("a 200-qubit CY chain must stay exact and cheap");
    assert!(v.is_finite(), "expectation must be a number, got {v}");
}

/// **`U3` expands exactly, and the global phase it drops really is dropped by
/// conjugation.**
///
/// `U3(θ,φ,λ) = e^{i(φ+λ)/2}·Rz(φ)·Ry(θ)·Rz(λ)` — the equality holds only up to
/// that phase. It is exact *here* because `O -> U†OU` cancels it, which is
/// precisely why the expansion belongs inside this backend rather than in the
/// shared lowering, where the same rewrite would hand a phase-shifted circuit
/// to the statevector backend.
///
/// Checked against the statevector backend, which does implement `U3`
/// natively — so this compares the expansion against the real gate, not
/// against another copy of the same decomposition.
#[test]
fn u3_matches_the_statevector_backend() {
    let pp = PauliPropBackend::new();
    let sv = StatevectorBackend::new();
    let params = ParameterBinding::default();
    let mut nonzero = 0;
    for (t, f, l) in [
        (0.7, 1.1, 0.3),
        (1.9, -0.4, 2.2),
        (std::f64::consts::FRAC_PI_2, 0.0, 0.0),
        (0.0, 0.5, -0.5),
    ] {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.add_op(g(GateKind::H, &[0], &[]));
        c.add_op(g(GateKind::U3, &[0], &[t, f, l]));
        c.add_op(g(GateKind::CX, &[0, 1], &[]));
        c.add_op(g(GateKind::U3, &[1], &[l, t, f]));
        for q in 0..2u32 {
            for p in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
                let o = obs(&[(q, p)]);
                let want = sv.expectation(&c, &params, &o).expect("reference");
                let got = pp
                    .expectation(&c, &params, &o)
                    .unwrap_or_else(|e| panic!("U3({t},{f},{l}): {e}"));
                assert!(
                    (got - want).abs() < 1e-9,
                    "U3({t},{f},{l}) q{q} {p:?}: pauliprop {got} vs statevector {want}"
                );
                if want.abs() > 1e-6 {
                    nonzero += 1;
                }
            }
        }
    }
    assert!(
        nonzero > 5,
        "only {nonzero} non-trivial values — corpus too degenerate"
    );
}

/// **The diagonal `CU3(0,0,λ)` — what `cp`/`cu1` lower to — is exact.**
#[test]
fn the_controlled_phase_form_of_cu3_matches() {
    let pp = PauliPropBackend::new();
    let sv = StatevectorBackend::new();
    let params = ParameterBinding::default();
    let mut nonzero = 0;
    for lam in [0.3, 1.7, -0.9] {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.add_op(g(GateKind::H, &[0], &[]));
        c.add_op(g(GateKind::H, &[1], &[]));
        c.add_op(g(GateKind::CU3, &[0, 1], &[0.0, 0.0, lam]));
        for q in 0..2u32 {
            for p in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
                let o = obs(&[(q, p)]);
                let want = sv.expectation(&c, &params, &o).expect("reference");
                let got = pp
                    .expectation(&c, &params, &o)
                    .unwrap_or_else(|e| panic!("CP({lam}): {e}"));
                assert!(
                    (got - want).abs() < 1e-9,
                    "CP({lam}) q{q} {p:?}: pauliprop {got} vs statevector {want}"
                );
                if want.abs() > 1e-6 {
                    nonzero += 1;
                }
            }
        }
    }
    assert!(nonzero > 3, "only {nonzero} non-trivial values");
}

/// **A general `CU3` must be REFUSED, not silently treated as a phase.**
///
/// This is the trap: `cp` arrives widened to `CU3(0,0,λ)`, so an implementation
/// keyed on `GateKind::CU3` that applies the controlled-phase identity without
/// checking θ and φ returns a wrong number for every genuine `cu3` — with no
/// error anywhere. The guard is the whole point of the arm.
#[test]
fn a_general_cu3_is_refused_rather_than_mistaken_for_a_phase() {
    let pp = PauliPropBackend::new();
    let params = ParameterBinding::default();
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[0], &[]));
    c.add_op(g(GateKind::CU3, &[0, 1], &[0.7, 0.2, 0.3]));
    let err = pp
        .expectation(&c, &params, &obs(&[(0, PauliOp::Z)]))
        .expect_err("a general CU3 must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("diagonal") || msg.contains("CU3(0, 0"),
        "the refusal must explain which form is supported; got: {msg}"
    );
    assert!(
        msg.contains("statevector") || msg.contains("MPS"),
        "the refusal should name a backend that can do it; got: {msg}"
    );
}

/// **CCX and CSwap via CCZ — seven COMMUTING rotations, and the signs are
/// right.**
///
/// The textbook route is 6 CX plus a 7-gate T ladder. That is the wrong one
/// here: those 6 CX are full Clifford passes over the whole sum for no benefit,
/// and the T ladder's generators do not commute with the CX between them, so
/// the Pauli weight scrambles as it goes. `CCZ` is seven DIAGONAL rotations at
/// ±π/4 whose generators are all products of `Z` — and products of `Z` all
/// commute, so there is nothing between them to scramble.
///
/// What that buys is structure, not cheapness. At θ = π/4, `cos θ = sin θ`, so
/// both children of each split carry equal weight and coefficient truncation
/// cannot prune either. A Toffoli is genuinely expensive for Pauli propagation;
/// the `max_terms` ceiling is what stops that from becoming a hung machine.
///
/// The seven signs come from expanding `CCZ = exp(iπ/8·(1−Z₁)(1−Z₂)(1−Z₃))` and
/// are wrong-answer material if mis-derived — which is why this compares
/// against the dense statevector backend rather than against the derivation.
#[test]
fn ccx_and_cswap_match_the_statevector_backend() {
    let pp = PauliPropBackend::new();
    let sv = StatevectorBackend::new();
    let params = ParameterBinding::default();
    const TOL: f64 = 1e-9;

    let mut checked = 0;
    let mut nonzero = 0;
    for (label, circuit) in [
        ("ccx", {
            let mut c = CircuitIR::new(3, CircuitType::GateBased);
            c.add_op(g(GateKind::H, &[0], &[]));
            c.add_op(g(GateKind::H, &[1], &[]));
            c.add_op(g(GateKind::Ry, &[2], &[0.6]));
            c.add_op(g(GateKind::CCX, &[0, 1, 2], &[]));
            c
        }),
        ("ccx reversed operands", {
            let mut c = CircuitIR::new(3, CircuitType::GateBased);
            c.add_op(g(GateKind::Ry, &[0], &[1.2]));
            c.add_op(g(GateKind::H, &[1], &[]));
            c.add_op(g(GateKind::H, &[2], &[]));
            c.add_op(g(GateKind::CCX, &[2, 1, 0], &[]));
            c
        }),
        ("cswap", {
            let mut c = CircuitIR::new(3, CircuitType::GateBased);
            c.add_op(g(GateKind::H, &[0], &[]));
            c.add_op(g(GateKind::Ry, &[1], &[0.8]));
            c.add_op(g(GateKind::Rx, &[2], &[0.5]));
            c.add_op(g(GateKind::CSwap, &[0, 1, 2], &[]));
            c
        }),
    ] {
        for q in 0..3u32 {
            for p in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
                let o = obs(&[(q, p)]);
                let want = sv.expectation(&circuit, &params, &o).expect("reference");
                let got = pp
                    .expectation(&circuit, &params, &o)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert!(
                    (got - want).abs() < TOL,
                    "{label} q{q} {p:?}: pauliprop {got} vs statevector {want} \
                     (delta {:.3e})",
                    (got - want).abs()
                );
                checked += 1;
                if want.abs() > 1e-6 {
                    nonzero += 1;
                }
            }
        }
    }
    assert!(checked >= 27, "only {checked} observables compared");
    assert!(
        nonzero > 3,
        "only {nonzero} non-trivial values — a sign error hides on zeros"
    );
}
