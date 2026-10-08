// SPDX-License-Identifier: Apache-2.0
//! S1 test (ii), and the standing rule of plan §2: the **fixture-symmetry**
//! pair. A `t → s` and a `t → tdg` mutation must each move a measured
//! expectation by a stated nonzero amount, and the fixture has to be built so
//! that they do.
//!
//! The trap §2 names, and K8 recorded as a `t → tdg` mutation scoring exactly
//! `0.00e+00`: on `|+⟩`, a `Z`-rotation by `θ` gives
//!
//! ```text
//! ⟨X⟩ = cos θ        ⟨Y⟩ = sin θ
//! ```
//!
//! and `cos` is **even**. `⟨X⟩` is therefore the same number for `T` and for
//! `T†` — not close, equal to the last bit — so a fixture that measures `⟨X⟩`
//! alone scores a `t → tdg` mutation at zero and reports a clean bill. `sin`
//! is odd and separates them by `2 sin(π/4) = √2`.
//!
//! The rule is "measure **both**", not "measure `⟨Y⟩`", and the two halves of
//! the first test are why. Start from `|+i⟩ = SH|0⟩` instead and the same
//! rotation gives `⟨X⟩ = −sin θ`, `⟨Y⟩ = cos θ`: the blindness changes legs.
//! Neither observable subsumes the other, so dropping either one leaves a
//! blind spot — just a different one.
//!
//! Every value below is a closed form, asserted as such, and cross-checked
//! against the dense statevector on the spot by [`measure`]: a *gap* between
//! two numbers is worth nothing if either end is wrong.
//!
//! The mutations here are applied to the **circuit**, so these numbers hold
//! without any source edit. `mutation_s1_ledger.rs` is the separate leg that
//! edits the engine.
//!
//! # Which mutation this fixture catches, and which it does not
//!
//! Catches: any confusion of a Z-rotation's sign or magnitude — `T` run as
//! `S`, as `T†`, as `Z` or as the identity — both on a single branch pair and,
//! through the interferometer, when the answer is assembled from 1024 cross
//! terms between 32 branches.
//!
//! Does NOT catch: anything that moves both legs of a pair by the same
//! amount on every circuit here. A defect in `h_gate`, or in the readout's
//! handling of an observable's coefficient, would shift the absolute values
//! and leave every *gap* intact. The gaps are what this file asserts; the
//! absolute values are asserted too, against their closed forms, which is
//! what closes that hole — and `clifford_t_vs_statevector.rs` is what holds
//! them against the oracle across circuits nobody chose by hand.

use omega_backend_stabrank::StabRankBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, Observable};
use omega_core::params::ParameterBinding;

const TOL: f64 = 1e-12;

/// `1/√2`, which every number in this file is built from.
const R: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn op(kind: &GateKind, qubits: &[u32]) -> GateOp {
    GateOp {
        gate: kind.clone(),
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: Default::default(),
        classical_bit: None,
        condition: None,
    }
}

/// `⟨P⟩` through stabrank, cross-checked against the dense oracle on the
/// spot, with the branch count returned so a fixture can state it.
fn measure(c: &CircuitIR, obs: &Observable, what: &str) -> (f64, usize) {
    let (got, cert) = StabRankBackend::new()
        .expectation_with_certificate(c, &ParameterBinding::new(), obs)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let want = StatevectorBackend::new()
        .expectation(c, &ParameterBinding::new(), obs)
        .expect("dense oracle");
    assert!(
        (got - want).abs() < TOL,
        "{what}: stabrank {got} vs dense {want} (χ = {})",
        cert.final_chi
    );
    (got, cert.final_chi)
}

/// Read `⟨X₀⟩` and `⟨Y₀⟩` off a circuit built by `make` with `gate` in its
/// slot, and return them with the branch count.
fn xy(make: &dyn Fn(&GateKind) -> CircuitIR, gate: &GateKind) -> (f64, f64, usize) {
    let c = make(gate);
    let (x, chi) = measure(&c, &Observable::x(0), &format!("⟨X₀⟩ with {gate:?}"));
    let (y, _) = measure(&c, &Observable::y(0), &format!("⟨Y₀⟩ with {gate:?}"));
    (x, y, chi)
}

/// Assert a triple of readings against its closed form, then assert the two
/// mutation gaps against theirs.
///
/// `blind_leg_is_x` names the even leg, the one `t → tdg` must not move, and
/// `blind_tol` is how much "not move" is allowed to be. On a one-split
/// fixture it is **0.0** — the two runs sum four identical products in the
/// same order and the equality is bit-for-bit, so anything looser would leave
/// room for the fixture to be weakly sensitive, which is not the claim. At
/// χ = 32 the same quantity is a sum of 1024 products and bit-equality is not
/// on offer; there the tolerance is stated and the measured value printed.
#[allow(clippy::too_many_arguments)]
fn assert_pair(
    label: &str,
    (tx, ty): (f64, f64),
    (sx, sy): (f64, f64),
    (dx, dy): (f64, f64),
    want: [(f64, f64); 3],
    blind_leg_is_x: bool,
    blind_tol: f64,
) {
    for (what, got, expected) in [
        ("⟨X₀⟩ with T", tx, want[0].0),
        ("⟨Y₀⟩ with T", ty, want[0].1),
        ("⟨X₀⟩ with S", sx, want[1].0),
        ("⟨Y₀⟩ with S", sy, want[1].1),
        ("⟨X₀⟩ with T†", dx, want[2].0),
        ("⟨Y₀⟩ with T†", dy, want[2].1),
    ] {
        assert!(
            (got - expected).abs() < TOL,
            "{label}: {what} = {got}, expected {expected}"
        );
    }

    let (blind, seeing) = if blind_leg_is_x {
        ((tx, dx, "⟨X₀⟩"), (ty, dy, "⟨Y₀⟩"))
    } else {
        ((ty, dy, "⟨Y₀⟩"), (tx, dx, "⟨X₀⟩"))
    };
    assert!(
        (blind.0 - blind.1).abs() <= blind_tol,
        "{label}: {} is the even leg here, so T and T† must agree on it to \
         {blind_tol}; they differ by {:.3e}. If they differ by more, this \
         fixture is not the §2 trap and the other leg stops being the thing \
         that carries the test.",
        blind.2,
        (blind.0 - blind.1).abs()
    );
    assert!(
        ((seeing.0 - seeing.1).abs() - 2.0 * R).abs() < TOL,
        "{label}: t → tdg must move {} by {}; measured {}",
        seeing.2,
        2.0 * R,
        (seeing.0 - seeing.1).abs()
    );

    // t → s moves both legs, by two different stated amounts: `|cos(π/4) −
    // cos(π/2)| = 1/√2` on one and `|sin(π/4) − sin(π/2)| = 1 − 1/√2` on the
    // other. Which leg gets which is the same swap as above.
    let (big, small) = if blind_leg_is_x {
        ((tx - sx).abs(), (ty - sy).abs())
    } else {
        ((ty - sy).abs(), (tx - sx).abs())
    };
    assert!(
        (big - R).abs() < TOL,
        "{label}: t → s must move the even leg by {R}; measured {big}"
    );
    assert!(
        (small - (1.0 - R)).abs() < TOL,
        "{label}: t → s must move the odd leg by {}; measured {small}",
        1.0 - R
    );
}

/// `H` (or `SH`) then the gate under test, on one wire. χ = 2.
fn on_plus(gate: &GateKind) -> CircuitIR {
    let mut c = CircuitIR::new(1, CircuitType::GateBased);
    c.add_op(op(&GateKind::H, &[0]));
    c.add_op(op(gate, &[0]));
    c
}

fn on_plus_i(gate: &GateKind) -> CircuitIR {
    let mut c = on_plus(&GateKind::S);
    c.add_op(op(gate, &[0]));
    c
}

/// The §2 trap, and the fact that makes the rule "both" rather than "⟨Y⟩".
#[test]
fn on_plus_x_is_blind_to_t_versus_tdg_and_on_plus_i_it_is_y_that_is_blind() {
    // |+⟩: ⟨X⟩ = cos θ (even, blind), ⟨Y⟩ = sin θ.
    let (tx, ty, chi) = xy(&on_plus, &GateKind::T);
    let (sx, sy, _) = xy(&on_plus, &GateKind::S);
    let (dx, dy, _) = xy(&on_plus, &GateKind::Tdg);
    assert_eq!(chi, 2, "one T is one 2-term split");
    assert_pair(
        "|+⟩",
        (tx, ty),
        (sx, sy),
        (dx, dy),
        [(R, R), (0.0, 1.0), (R, -R)],
        true,
        0.0,
    );
    eprintln!(
        "stabrank §2 on |+⟩  (χ = {chi}): t→s moves ⟨X⟩ {:.6} ⟨Y⟩ {:.6}; \
         t→tdg moves ⟨X⟩ {:.3e} ⟨Y⟩ {:.6}",
        (tx - sx).abs(),
        (ty - sy).abs(),
        (tx - dx).abs(),
        (ty - dy).abs()
    );

    // |+i⟩ = SH|0⟩: ⟨X⟩ = −sin θ, ⟨Y⟩ = cos θ (even, blind). The legs swap.
    let (tx, ty, _) = xy(&on_plus_i, &GateKind::T);
    let (sx, sy, _) = xy(&on_plus_i, &GateKind::S);
    let (dx, dy, _) = xy(&on_plus_i, &GateKind::Tdg);
    assert_pair(
        "|+i⟩",
        (tx, ty),
        (sx, sy),
        (dx, dy),
        [(-R, R), (-1.0, 0.0), (R, R)],
        false,
        0.0,
    );
    eprintln!(
        "stabrank §2 on |+i⟩ (χ = 2): t→s moves ⟨X⟩ {:.6} ⟨Y⟩ {:.6}; \
         t→tdg moves ⟨X⟩ {:.6} ⟨Y⟩ {:.3e}",
        (tx - sx).abs(),
        (ty - sy).abs(),
        (tx - dx).abs(),
        (ty - dy).abs()
    );
}

/// A GHZ interferometer: entangle, phase, un-entangle, read wire 0.
///
/// `(|000⟩ + |111⟩)/√2`, then four `T` gates and the slot, then the
/// entangling pair run backwards. The `|000⟩` branch picks up nothing and the
/// `|111⟩` branch picks up every phase, so the two un-entangle onto wire 0 as
/// `(|0⟩ + φ|1⟩)/√2` with `φ = e^{4iπ/4}·g = −g`, `g` the phase the slot gate
/// puts on `|1⟩`. Then `⟨X₀⟩ = Re φ` and `⟨Y₀⟩ = Im φ` — the `|+⟩` fixture's
/// own numbers, up to the overall `−`, reached through four splits.
///
/// Four and not three: `−g` is what makes `Re φ` even in the slot angle again,
/// so this is the §2 trap itself rather than a circuit that merely contains a
/// T gate. χ is 32 with a `T` in the slot and 16 with an `S`, and the value is
/// assembled from 1024 cross terms `⟨φᵢ|O|φⱼ⟩` — which is the point: a phase
/// that is merely *global per branch* cancels on one state and does not cancel
/// here.
fn interferometer(gate: &GateKind) -> CircuitIR {
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    c.add_op(op(&GateKind::H, &[0]));
    c.add_op(op(&GateKind::CX, &[0, 1]));
    c.add_op(op(&GateKind::CX, &[1, 2]));
    for q in [0u32, 1, 2, 0] {
        c.add_op(op(&GateKind::T, &[q]));
    }
    // The slot under mutation.
    c.add_op(op(gate, &[2]));
    c.add_op(op(&GateKind::CX, &[1, 2]));
    c.add_op(op(&GateKind::CX, &[0, 1]));
    c
}

/// The same trap, downstream of four splits and 1024 cross terms.
#[test]
fn the_trap_survives_into_the_interfering_fixture_at_chi_thirty_two() {
    let (tx, ty, tchi) = xy(&interferometer, &GateKind::T);
    let (sx, sy, schi) = xy(&interferometer, &GateKind::S);
    let (dx, dy, dchi) = xy(&interferometer, &GateKind::Tdg);
    assert_eq!(
        (tchi, schi, dchi),
        (32, 16, 32),
        "four T gates plus the slot: 2^5 with a T in it, 2^4 with an S"
    );
    // φ = −g. T: −e^{iπ/4}. S: −i. T†: −e^{−iπ/4}.
    assert_pair(
        "interferometer",
        (tx, ty),
        (sx, sy),
        (dx, dy),
        [(-R, -R), (0.0, -1.0), (-R, R)],
        true,
        1e-13,
    );
    eprintln!(
        "stabrank §2 interfering (χ = {tchi}, {} cross terms): t→s moves ⟨X₀⟩ \
         {:.6} ⟨Y₀⟩ {:.6}; t→tdg moves ⟨X₀⟩ {:.3e} ⟨Y₀⟩ {:.6}",
        tchi * tchi,
        (tx - sx).abs(),
        (ty - sy).abs(),
        (tx - dx).abs(),
        (ty - dy).abs()
    );
}
