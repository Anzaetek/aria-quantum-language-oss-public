// SPDX-License-Identifier: Apache-2.0
//! S1 test (iv): **three-way agreement** between stabrank, majoranaprop and
//! the dense statevector on a Clifford-angle-Givens + T fermionic workload,
//! at 1e-10 — and `observable_range` computed post-JW and asserted against
//! majoranaprop's.
//!
//! # Why this workload and not another
//!
//! Plan §1.4 puts the two engines in complementary cost regimes and names the
//! place they overlap. majoranaprop is Heisenberg: a Givens rotation `Rbs(θ)`
//! is free at **any** angle because it preserves Majorana length, and a `T`
//! branches but exactly. stabrank is Schrödinger: a Clifford gate is free,
//! and `Rbs(θ)` is Clifford exactly at multiples of `π/2`. So a circuit of
//! quarter-turn Givens rotations and `T` gates is cheap for both and —
//! the part that makes it a test rather than a demonstration — **exact** for
//! both. Neither engine truncates here, so a disagreement is a defect and
//! cannot be explained away as a bound.
//!
//! Both certificates are asserted exact (`is_exact()`), which is the
//! instantiation witness in its negative form: the claim "both engines are
//! exact on this workload" is checked rather than assumed, and if a future
//! change made either one truncate this fixture, it would say so here instead
//! of quietly widening its own error.
//!
//! # The `observable_range` pin
//!
//! §1.2's standing pin is that `R` is computed **in the basis actually read
//! out**. The two engines read out in different ones and arrive at the same
//! number, and that is the assertion:
//!
//! * stabrank is Schrödinger, so there is no ladder-native seeding to do on
//!   the state side (§1.4): the `FermionicOp` is mapped by Jordan–Wigner to a
//!   Pauli `Observable` — exactly, so nothing is lost — and `R = Σ|cᵢ|` over
//!   *those* terms. That is the F2 dispatch the CLI will use at S3, done here
//!   by the test because at S1 the engine has no fermionic door.
//! * majoranaprop's ladder door seeds the Majorana basis directly and reports
//!   `R = Σ|c_b|` over monomials.
//!
//! A Pauli string and its Majorana monomial differ by a sign only, so the two
//! sums must agree. This is also the test that keeps
//! [`omega_backend_stabrank::observable_l1_norm`] from drifting away from
//! majoranaprop's identically-defined `observable_l1_norm`: the two are
//! separate functions in sibling crates with no dependency edge between them,
//! and an assertion is what ties them together rather than a shared symbol.
//!
//! # Which mutation this fixture catches, and which it does not
//!
//! Catches: any disagreement between the two engines' pictures — a wrong JW
//! sign on one side, a Givens rotation that is Clifford in one engine's table
//! and not the other's, a `T` executed at the wrong angle in either. Because
//! three independent implementations have to meet, a defect has to be present
//! in two of them in the same direction to hide.
//!
//! Does NOT catch: a defect in the Jordan–Wigner map itself. All three legs
//! read the *same* `Observable` (stabrank and statevector literally; and
//! majoranaprop's ladder seed is checked against its own Pauli door in
//! `omega-backend-majoranaprop/tests/fermionic_seed.rs`, not here), so a JW
//! map that produced the wrong Pauli image would move all three legs together
//! and this fixture would stay green. It also says nothing about
//! generic-angle Givens, which stabrank splits rather than applies — that is
//! `clifford_t_vs_statevector.rs`'s gate-table sweep.

use num_complex::Complex64;
use omega_backend_majoranaprop::MajoranaPropBackend;
use omega_backend_stabrank::{observable_l1_norm, StabRankBackend, StabilizerSum};
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::Backend;
use omega_core::fermion::{FermionicOp, Ladder};
use omega_core::params::ParameterBinding;

const TOL: f64 = 1e-10;

fn gop(kind: &GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind.clone(),
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// One mode per qubit under Jordan–Wigner. Four is enough for the dense leg
/// to be instant and enough for a hopping term to be non-adjacent.
const MODES: u32 = 4;

/// `X_q` under JW creates a particle on mode `q` out of the vacuum, so the
/// prepared occupation is the list of wires given an `X`. Particle-number
/// preserving Givens rotations on the vacuum do nothing at all, which is why
/// the preparation comes first.
///
/// The `H` layer is there for a reason worth stating. A quarter-turn Givens
/// rotation is a *signed permutation* of the occupation basis and a `T` is
/// diagonal, so a circuit of those two started from a computational basis
/// state never leaves one: every expectation collapses to a coefficient of
/// the observable, every branch of the decomposition is a copy of every
/// other, and three engines agree for free. Staying inside the
/// number-preserving Clifford-angle gate set cannot avoid this — building a
/// superposition of occupations needs a Givens rotation off the `π/2` grid,
/// which is exactly what this workload may not contain. So the superposition
/// is made with Hadamards, which are Clifford, free for stabrank, and in
/// majoranaprop's table; the circuit stops conserving particle number and the
/// observable stops being trivially diagonal, which is the point.
/// `the_workload_is_not_trivially_diagonal` holds that gain on the record.
fn workload(
    occupied: &[u32],
    superposed: &[u32],
    t_wires: &[u32],
    quarter_turns: &[(u32, u32, u32)],
) -> CircuitIR {
    use std::f64::consts::FRAC_PI_2;
    let mut c = CircuitIR::new(MODES, CircuitType::GateBased);
    for &q in occupied {
        c.add_op(gop(&GateKind::X, &[q], &[]));
    }
    for &q in superposed {
        c.add_op(gop(&GateKind::H, &[q], &[]));
    }
    for &q in superposed {
        c.add_op(gop(&GateKind::CX, &[q, (q + 1) % MODES], &[]));
    }
    for &(p, q, k) in quarter_turns {
        c.add_op(gop(&GateKind::Rbs, &[p, q], &[k as f64 * FRAC_PI_2]));
    }
    for &q in t_wires {
        c.add_op(gop(&GateKind::T, &[q], &[]));
    }
    // A second Givens layer after the phases, so the T gates sit *between*
    // rotations rather than at the end where they would only dress the
    // readout basis. Its angle is `k + 1`, not `k`: `Rbs(q, p, θ)` is
    // `Rbs(p, q, −θ)` — the generator `Y_pX_q − X_pY_q` is antisymmetric in
    // the two wires — so repeating `k` here would make the second layer the
    // first one's inverse, the two would cancel, and the whole workload would
    // stop depending on `k` at all.
    for &(p, q, k) in quarter_turns {
        c.add_op(gop(&GateKind::Rbs, &[q, p], &[(k + 1) as f64 * FRAC_PI_2]));
    }
    for &q in t_wires {
        c.add_op(gop(&GateKind::Tdg, &[q], &[]));
        c.add_op(gop(&GateKind::T, &[(q + 1) % MODES], &[]));
    }
    c
}

/// A Hermitian fermionic observable with number, hopping and interaction
/// terms — three different Majorana lengths, so the Heisenberg leg has
/// something to branch on.
fn observable() -> FermionicOp {
    let c = |x: f64| Complex64::new(x, 0.0);
    let hop = |p: u32, q: u32, w: f64| {
        let t = FermionicOp::term(c(w), vec![Ladder::raise(p), Ladder::lower(q)]);
        let td = t.clone().dagger();
        t + td
    };
    FermionicOp::term(c(1.25), vec![Ladder::raise(0), Ladder::lower(0)])
        + FermionicOp::term(c(-0.5), vec![Ladder::raise(2), Ladder::lower(2)])
        + hop(0, 1, 0.75)
        + hop(1, 3, 0.4)
        + FermionicOp::term(
            c(0.6),
            vec![
                Ladder::raise(0),
                Ladder::lower(0),
                Ladder::raise(2),
                Ladder::lower(2),
            ],
        )
}

/// One workload, three engines, two certificates. Returns the agreed value so
/// a caller can check the sweep is not reading the same number every time.
fn three_way(c: &CircuitIR, what: &str) -> f64 {
    let params = ParameterBinding::new();
    let fermionic = observable();
    let pauli = fermionic
        .jordan_wigner()
        .expect("the observable is Hermitian, so its JW image is real");

    let (stab, stab_cert) = StabRankBackend::new()
        .expectation_with_certificate(c, &params, &pauli)
        .unwrap_or_else(|e| panic!("{what}: stabrank refused the workload: {e}"));
    let (maj, maj_cert) = MajoranaPropBackend::new()
        .expectation_fermionic_with_certificate(c, &params, &fermionic)
        .unwrap_or_else(|e| panic!("{what}: majoranaprop refused the workload: {e}"));
    let dense = StatevectorBackend::new()
        .expectation(c, &params, &pauli)
        .expect("dense oracle");

    // Both engines exact: the premise of the comparison, asserted.
    assert!(
        stab_cert.is_exact() && stab_cert.state_dropped_mass == 0.0,
        "{what}: stabrank truncated something on a workload S1 cannot truncate"
    );
    assert!(
        maj_cert.is_exact() && maj_cert.dropped_mass == 0.0,
        "{what}: majoranaprop dropped {:.3e} of coefficient mass, so the two \
         engines are no longer being compared on equal terms",
        maj_cert.dropped_mass
    );

    for (a, b, pair) in [
        (stab, dense, "stabrank vs statevector"),
        (maj, dense, "majoranaprop vs statevector"),
        (stab, maj, "stabrank vs majoranaprop"),
    ] {
        assert!(
            (a - b).abs() < TOL,
            "{what}: {pair} disagree — {a} vs {b}, |Δ| = {:.3e}",
            (a - b).abs()
        );
    }

    // §1.2's pin: the same R, computed in two different bases.
    let post_jw = observable_l1_norm(&pauli);
    assert!(
        (stab_cert.observable_range - post_jw).abs() < 1e-15,
        "{what}: stabrank's observable_range {} is not Σ|cᵢ| over the Pauli \
         terms it actually read out ({post_jw})",
        stab_cert.observable_range
    );
    assert!(
        (stab_cert.observable_range - maj_cert.observable_range).abs() < 1e-12,
        "{what}: the two engines report different ranges for the same \
         observable — stabrank {} (post-JW Pauli), majoranaprop {} (Majorana \
         monomials). A Pauli string and its Majorana monomial differ by a sign \
         only, so these must agree.",
        stab_cert.observable_range,
        maj_cert.observable_range
    );
    assert!(
        stab.abs() <= stab_cert.observable_range + TOL,
        "{what}: ⟨O⟩ = {stab} is outside its own a-priori range"
    );

    eprintln!(
        "stabrank three-way {what}: value {stab:.12} (majoranaprop {maj:.12}, \
         dense {dense:.12}), χ = {} / peak {}, majoranaprop terms {} / peak {}, \
         R = {:.6} both sides",
        stab_cert.final_chi,
        stab_cert.peak_chi,
        maj_cert.final_terms,
        maj_cert.peak_terms,
        stab_cert.observable_range
    );
    stab
}

/// The headline cell: half-filled, superposed, four quarter-turn Givens
/// rotations at mixed `k`, nine T gates.
#[test]
fn clifford_angle_givens_plus_t_agrees_three_ways() {
    let c = workload(
        &[0, 2],
        &[1, 3],
        &[0, 1, 3],
        &[(0, 1, 1), (1, 2, 3), (2, 3, 1), (0, 3, 2)],
    );
    three_way(&c, "half filling, k = (1,3,1,2)");
}

/// The same workload across every occupation and every quarter turn, so the
/// agreement is not a property of one cell. `k = 0` is included: there the
/// Givens layer is the identity and the circuit is Clifford+T alone, which is
/// the degenerate end of the overlap region and should still agree.
#[test]
fn the_agreement_holds_across_occupations_and_quarter_turns() {
    let mut values = Vec::new();
    for occupied in [
        vec![],
        vec![0u32],
        vec![1, 2],
        vec![0, 1, 2],
        vec![0, 1, 2, 3],
    ] {
        // The superposition and phase layers move with the occupation too, so
        // the twenty cells are twenty circuits rather than one circuit seen
        // from five starting points.
        let m = occupied.len() as u32;
        for k in 0..4u32 {
            let c = workload(
                &occupied,
                &[m % MODES, (m + 2) % MODES],
                &[(m + k) % MODES, (m + k + 1) % MODES],
                &[(0, 1, k), (1, 2, k + 1), (2, 3, k + 2), (0, 2, k + 3)],
            );
            values.push(three_way(&c, &format!("occupied {occupied:?}, k = {k}")));
        }
    }
    assert_eq!(values.len(), 20, "the sweep did not cover its grid");
    let distinct = {
        let mut v = values.clone();
        v.sort_by(f64::total_cmp);
        v.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        v.len()
    };
    eprintln!("stabrank three-way sweep: {distinct} distinct values over 20 cells");
    // Measured: 6 distinct values. The bar sits one below so that an honest
    // reshuffle of the grid does not redden it, but a collapse to "every cell
    // returns the same number" — three engines agreeing once, reported twenty
    // times — still does.
    assert!(
        distinct >= 5,
        "only {distinct} distinct values over 20 cells — the sweep is reading \
         nearly the same number everywhere, and three engines agreeing on one \
         number twenty times is one piece of evidence, not twenty"
    );
}

/// How many branches of a decomposition are the same state as the first, up
/// to the global phase the CH form carries: `|⟨φᵢ|φ₀⟩| = 1`.
fn copies_of_branch_zero(sum: &StabilizerSum) -> usize {
    let first = &sum.branches()[0].1;
    sum.branches()
        .iter()
        .filter(|(_, b)| (b.inner_product(first).norm() - 1.0).abs() < 1e-9)
        .count()
}

/// The non-triviality the `H` layer buys, held on the record.
///
/// Without it the state is a computational basis state throughout, every
/// branch of the decomposition is a copy of the others, and the fermionic
/// observable's expectation is one of its own coefficients. Both facts are
/// measured here against the same workload with the layer removed, so the
/// difference between a fixture that tests something and a fixture that
/// cannot fail is a number in this file rather than a claim in a comment.
#[test]
fn the_workload_is_not_trivially_diagonal() {
    let turns = [(0u32, 1u32, 1u32), (1, 2, 3), (2, 3, 1), (0, 3, 2)];
    let params = ParameterBinding::new();
    let pauli = observable().jordan_wigner().expect("Hermitian");
    let backend = StabRankBackend::new();

    let flat = workload(&[0, 2], &[], &[0, 1, 3], &turns);
    let flat_sum = backend.simulate_sum(&flat, &params).expect("executable");
    let flat_copies = copies_of_branch_zero(&flat_sum);
    let flat_value = backend
        .expectation(&flat, &params, &pauli)
        .expect("executable");
    assert_eq!(
        flat_copies,
        flat_sum.chi(),
        "without the superposition layer every branch should be the same state"
    );

    let live = workload(&[0, 2], &[1, 3], &[0, 1, 3], &turns);
    let live_sum = backend.simulate_sum(&live, &params).expect("executable");
    let live_copies = copies_of_branch_zero(&live_sum);
    let live_value = backend
        .expectation(&live, &params, &pauli)
        .expect("executable");
    eprintln!(
        "stabrank workload non-triviality: without the H layer {flat_copies} of \
         {} branches are copies of branch 0 and ⟨O⟩ = {flat_value:.12}; with it, \
         {live_copies} of {} and ⟨O⟩ = {live_value:.12}",
        flat_sum.chi(),
        live_sum.chi()
    );
    assert!(
        live_copies * 2 < live_sum.chi(),
        "{live_copies} of {} branches are still copies of branch 0; the \
         superposition layer is not doing its job",
        live_sum.chi()
    );
    assert!(
        (live_value - flat_value).abs() > 0.1,
        "the H layer moved ⟨O⟩ by only {:.3e}, so the fixture is still reading \
         a coefficient of the observable back to itself",
        (live_value - flat_value).abs()
    );
}

/// The cost claim that makes this workload the overlap region rather than
/// just a circuit both engines happen to survive: stabrank's rank is `2^t`
/// and **no larger**, so the four Givens rotations in each layer — eight in
/// all, each two Pauli rotations — cost nothing at a quarter turn. Off the
/// grid they would cost `4^8` on their own.
#[test]
fn the_givens_layer_is_free_for_stabrank_at_a_quarter_turn() {
    let t_wires = [0u32, 2];
    // `workload` emits one T per wire in `t_wires`, then one Tdg and one T
    // per wire in the tail: 2 + 2 + 2 = 6 non-Clifford gates.
    let c = workload(
        &[0, 2],
        &[1, 3],
        &t_wires,
        &[(0, 1, 1), (1, 2, 3), (2, 3, 1), (0, 3, 2)],
    );
    assert_eq!(c.t_count(), 6, "the fixture's T count is part of the claim");
    let sum = StabRankBackend::new()
        .simulate_sum(&c, &ParameterBinding::new())
        .expect("the workload is executable");
    assert_eq!(
        sum.chi(),
        64,
        "χ must be 2^6 — exactly the T gates, and nothing from the eight \
         quarter-turn Givens rotations, which are Clifford (plan §1.4)"
    );
    assert_eq!(sum.peak_chi(), 64);
}
