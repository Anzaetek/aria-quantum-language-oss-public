// SPDX-License-Identifier: Apache-2.0
//! S1 test (iii): the **count pin**, and the rest of the certificate.
//!
//! `final_chi` on a `t`-gate circuit must exceed 1 and be at most `2^t`. The
//! pin is A10 in the form `fermionic_seed.rs` (iii) uses it: it is an
//! *exposed-count* contract, and the thing it rules out is an engine that
//! produced the right number by some other means. A backend that quietly
//! called the dense statevector has no branches to count; one that counted a
//! variable it incremented itself would still have to produce 2^t distinct
//! CH forms that sum to the right state, which is what the tests below ask
//! for rather than taking `final_chi` on trust:
//!
//! * `final_chi` is `branches().len()`, read off the decomposition;
//! * every branch satisfies the CH-form invariants;
//! * the branches are pairwise **distinct states** — `|⟨φᵢ|φⱼ⟩| < 1` for
//!   `i ≠ j` — so a rank padded with copies to hit `2^t` fails;
//! * and the same run's value still matches the dense oracle.
//!
//! The upper half of the pin, `χ ≤ 2^t`, is the plan's. The exact equality
//! `χ = 2^t` is asserted separately and for a different reason: at S1 nothing
//! is dropped and nothing is merged, so the naive rank is what comes out.
//! S2 truncates and will lower it; a later phase that merges branches would
//! lower it too. Either is a deliberate change to a named assertion rather
//! than a silent drift.
//!
//! # Which mutation this fixture catches, and which it does not
//!
//! Catches: a split that does not split (χ stuck at 1), a split that fires on
//! a Clifford gate (χ growing where §1.4 says it must not), a `peak_chi` that
//! is not a peak, a dropped branch, a duplicated branch, and any certificate
//! field computed from the wrong thing.
//!
//! Does NOT catch: anything about the branch *coefficients*. χ counts the
//! branches and says nothing about `cᵢ`, so an engine that got every
//! coefficient wrong would pass every count here — the `dense` cross-check
//! inside `exercise` is what stops that, and `clifford_t_vs_statevector.rs`
//! is what does it at scale.

use omega_backend_stabrank::{SeedBasis, StabRankBackend, DEFAULT_MAX_BRANCHES};
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

const TOL: f64 = 1e-10;

fn op(kind: &GateKind, qubits: &[u32]) -> GateOp {
    gop(kind, qubits, &[])
}

fn gop(kind: &GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind.clone(),
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// A circuit on 4 wires with exactly `t` `T` gates, spread over the wires and
/// separated by entangling Cliffords so no two branches coincide.
fn with_t_gates(t: usize) -> CircuitIR {
    let n = 4u32;
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for q in 0..n {
        c.add_op(op(&GateKind::H, &[q]));
    }
    for i in 0..t {
        let q = (i as u32) % n;
        c.add_op(op(&GateKind::T, &[q]));
        c.add_op(op(&GateKind::CX, &[q, (q + 1) % n]));
        c.add_op(op(&GateKind::H, &[(q + 1) % n]));
    }
    c
}

/// A three-term observable, so `observable_range` has something to add up.
fn observable() -> Observable {
    Observable {
        terms: vec![
            (1.0, vec![(0, PauliOp::X), (1, PauliOp::Z)]),
            (-0.5, vec![(2, PauliOp::Y)]),
            (0.75, vec![(0, PauliOp::Z), (3, PauliOp::X)]),
        ],
    }
}

/// `Σ|cᵢ|` of [`observable`], written out rather than summed by the code
/// under test.
const OBSERVABLE_RANGE: f64 = 1.0 + 0.5 + 0.75;

/// Run a circuit both ways and check everything the certificate claims about
/// the decomposition against the decomposition itself.
fn exercise(c: &CircuitIR, what: &str) -> (usize, usize) {
    let backend = StabRankBackend::new();
    let obs = observable();
    let (value, cert) = backend
        .expectation_with_certificate(c, &ParameterBinding::new(), &obs)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let sum = backend
        .simulate_sum(c, &ParameterBinding::new())
        .unwrap_or_else(|e| panic!("{what}: {e}"));

    assert_eq!(
        cert.final_chi,
        sum.branches().len(),
        "{what}: final_chi must be the number of branches there are, not a \
         number the engine remembers"
    );
    assert!(
        cert.peak_chi >= cert.final_chi,
        "{what}: peak χ {} is below final χ {}",
        cert.peak_chi,
        cert.final_chi
    );
    assert_eq!(
        cert.peak_chi, cert.final_chi,
        "{what}: at S1 nothing is ever removed from the sum, so the peak is \
         the end"
    );
    for (i, (_, branch)) in sum.branches().iter().enumerate() {
        branch
            .check_invariants()
            .unwrap_or_else(|e| panic!("{what}: branch {i}: {e}"));
    }

    // The certificate's own contract.
    assert_eq!(cert.seed_basis, SeedBasis::Pauli);
    assert_eq!(cert.seed_basis.as_str(), "pauli");
    assert_eq!(
        cert.state_dropped_mass, 0.0,
        "{what}: an S1 run truncates nothing"
    );
    assert!(cert.is_exact(), "{what}: an S1 run is exact");
    assert_eq!(cert.max_branches, DEFAULT_MAX_BRANCHES);
    assert!(
        (cert.observable_range - OBSERVABLE_RANGE).abs() < 1e-15,
        "{what}: observable_range {} is not Σ|cᵢ| = {OBSERVABLE_RANGE}",
        cert.observable_range
    );
    assert_eq!(
        cert.value, value,
        "{what}: the certificate reports the value"
    );
    assert!(
        value.abs() <= cert.observable_range + TOL,
        "{what}: ⟨O⟩ = {value} is outside its own a-priori range"
    );

    // And the value is right, so none of the counting above was bought by
    // getting the physics wrong.
    let want = StatevectorBackend::new()
        .expectation(c, &ParameterBinding::new(), &obs)
        .expect("dense oracle");
    assert!(
        (value - want).abs() < TOL,
        "{what}: stabrank {value} vs dense {want}"
    );
    (cert.final_chi, cert.peak_chi)
}

/// The pin itself, over `t = 0..=8`, plus the growth law between cells.
#[test]
fn final_chi_exceeds_one_and_is_at_most_two_to_the_t() {
    let mut previous = 0usize;
    for t in 0..=8usize {
        let c = with_t_gates(t);
        assert_eq!(
            c.t_count(),
            t,
            "the fixture must contain the T gates the pin is about"
        );
        let (chi, _) = exercise(&c, &format!("t={t}"));
        let ceiling = 1usize << t;
        assert!(
            chi <= ceiling,
            "t={t}: χ = {chi} exceeds 2^t = {ceiling}. The split is 2-term; a \
             rank above 2^t means branches are being created somewhere else."
        );
        if t > 0 {
            assert!(
                chi > 1,
                "t={t}: χ = {chi}. A circuit with a T gate in it has more than \
                 one stabilizer branch — an engine that reports 1 here is not \
                 running a decomposition, and one that secretly ran the dense \
                 statevector has no branches to report at all."
            );
            assert_eq!(
                chi,
                2 * previous,
                "t={t}: each T gate is one 2-term split, so χ doubles"
            );
        } else {
            assert_eq!(chi, 1, "a Clifford circuit is one branch");
        }
        previous = chi;
        eprintln!("stabrank count pin t={t}: χ = {chi}, ceiling 2^t = {ceiling}");
    }
}

/// Every `T` lands on a wire that is in no Z eigenstate, so neither leg of
/// any split is a copy of the other — see
/// [`the_naive_rank_is_not_the_minimal_rank`] for what happens when they are.
fn with_distinct_branches(t: usize) -> CircuitIR {
    let n = 5u32;
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for q in 0..n {
        c.add_op(op(&GateKind::H, &[q]));
    }
    for i in 0..t {
        let q = (i as u32) % n;
        c.add_op(op(&GateKind::T, &[q]));
        c.add_op(op(&GateKind::CZ, &[q, (q + 1) % n]));
    }
    c
}

/// `χ = 2^t` and the branches are 2^t **different** states. A decomposition
/// padded with copies of one branch satisfies every count in the test above
/// and fails here.
#[test]
fn the_branches_are_distinct_states_and_there_are_exactly_two_to_the_t_of_them() {
    for t in 1..=5usize {
        let c = with_distinct_branches(t);
        let sum = StabRankBackend::new()
            .simulate_sum(&c, &ParameterBinding::new())
            .expect("the fixture is executable");
        assert_eq!(sum.chi(), 1usize << t, "t={t}: S1 neither drops nor merges");
        let mut worst = 0.0f64;
        for (i, (_, a)) in sum.branches().iter().enumerate() {
            for (j, (_, b)) in sum.branches().iter().enumerate() {
                let overlap = a.inner_product(b).norm();
                if i == j {
                    assert!(
                        (overlap - 1.0).abs() < 1e-12,
                        "t={t}: branch {i} is not normalised (|⟨φ|φ⟩| = {overlap})"
                    );
                } else {
                    worst = worst.max(overlap);
                    assert!(
                        overlap < 1.0 - 1e-9,
                        "t={t}: branches {i} and {j} are the same state \
                         (|⟨φᵢ|φⱼ⟩| = {overlap}); a rank padded with copies is \
                         not a rank"
                    );
                }
            }
        }
        eprintln!(
            "stabrank distinctness t={t}: {} branches, largest off-diagonal \
             |⟨φᵢ|φⱼ⟩| = {worst:.6}",
            sum.chi()
        );
    }
}

/// **χ = 2^t is the naive rank, not the minimal one**, and the plan's pin is
/// `χ ≤ 2^t` for exactly this reason.
///
/// `T` on a wire that happens to sit in a Z eigenstate is a phase and nothing
/// else: `Z_q|φ⟩ = ±|φ⟩`, so the 2-term split produces two branches holding
/// the *same state*, whose coefficients then recombine into that phase. The
/// answer is right — the split is an identity at every angle — but the rank
/// is twice what the decomposition needs.
///
/// `with_t_gates` does this at `t = 2`: its second `T` lands on a wire that
/// the preceding `H` has just put in a Z eigenstate. Measured below rather
/// than described, because it is the kind of fact that quietly stops being
/// true and then nobody knows why a later phase's rank is different.
///
/// Nothing here asks S1 to merge. Recognising the degeneracy means comparing
/// states, which is `O(n³)` per pair against a split that costs `O(1)`, and
/// the saving is a property of the circuit rather than of the engine. It is
/// recorded so that S2, which truncates by coefficient mass, is read against
/// the right baseline.
#[test]
fn the_naive_rank_is_not_the_minimal_rank() {
    let sum = StabRankBackend::new()
        .simulate_sum(&with_t_gates(2), &ParameterBinding::new())
        .expect("the fixture is executable");
    assert_eq!(sum.chi(), 4, "the engine still splits twice");
    let mut distinct: Vec<usize> = Vec::new();
    for (i, (_, a)) in sum.branches().iter().enumerate() {
        if !distinct
            .iter()
            .any(|&j| (a.inner_product(&sum.branches()[j].1).norm() - 1.0).abs() < 1e-9)
        {
            distinct.push(i);
        }
    }
    eprintln!(
        "stabrank naive vs minimal rank: χ = {} branches spanning {} distinct \
         states on a fixture whose second T lands on a Z eigenstate",
        sum.chi(),
        distinct.len()
    );
    assert_eq!(
        distinct.len(),
        2,
        "the fixture is built so that the second split is degenerate; if this \
         is no longer 2, either the fixture changed or the engine started \
         merging, and `χ = 2^t` elsewhere in this file should be re-read"
    );
}

/// §1.4's cost claim at the backend door: a Clifford gate is free, including
/// the parameterised ones at a quarter turn. `rz(π/4)` splits and `rz(π/2)`
/// does not, and both are exact.
#[test]
fn a_clifford_angle_rotation_costs_no_branch() {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let build = |angles: &[(GateKind, f64)]| {
        let mut c = CircuitIR::new(4, CircuitType::GateBased);
        for q in 0..4u32 {
            c.add_op(op(&GateKind::H, &[q]));
        }
        c.add_op(op(&GateKind::CX, &[0, 1]));
        for (kind, theta) in angles {
            let wires: &[u32] = if *kind == GateKind::Rbs {
                &[1, 2]
            } else {
                &[1]
            };
            c.add_op(gop(kind, wires, &[*theta]));
        }
        c
    };
    for (kind, clifford_angles) in [
        (GateKind::Rz, [0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]),
        (GateKind::Rx, [0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]),
        (GateKind::Ry, [0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]),
        (GateKind::U1, [0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]),
        (GateKind::Rbs, [0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]),
    ] {
        let gates: Vec<_> = clifford_angles.iter().map(|&a| (kind.clone(), a)).collect();
        let c = build(&gates);
        let (chi, peak) = exercise(&c, &format!("{kind:?} at four quarter turns"));
        assert_eq!(
            (chi, peak),
            (1, 1),
            "{kind:?} at multiples of π/2 is Clifford and must not branch — \
             that is the cost claim of plan §1.4, and an engine that branches \
             here is exponential in the wrong thing"
        );

        // The same gate off the grid does branch, so the test above is not
        // passing because the gate is being ignored.
        let c = build(&[(kind.clone(), FRAC_PI_4)]);
        let (chi, _) = exercise(&c, &format!("{kind:?} at π/4"));
        let want = if kind == GateKind::Rbs { 4 } else { 2 };
        assert_eq!(
            chi, want,
            "{kind:?} at π/4 is not Clifford and must split (rbs is two Pauli \
             rotations, hence two splits)"
        );
    }
}

/// The ceiling refuses through the backend door, and names a way out. It is
/// a memory stop, not a truncation knob: nothing is dropped and no mass is
/// reported, because the run does not finish.
#[test]
fn the_branch_ceiling_refuses_through_the_door() {
    let c = with_t_gates(6);
    let backend = StabRankBackend::new().with_max_branches(16);
    let err = backend
        .expectation_with_certificate(&c, &ParameterBinding::new(), &observable())
        .expect_err("64 branches do not fit under a ceiling of 16");
    let msg = err.to_string();
    assert!(
        msg.contains("with_max_branches") && msg.contains("majoranaprop"),
        "the refusal must name the knob and the engine with the other cost \
         axis; got: {msg}"
    );
    // Raised, the same run returns, and the ceiling is recorded.
    let (_, cert) = StabRankBackend::new()
        .with_max_branches(64)
        .expectation_with_certificate(&c, &ParameterBinding::new(), &observable())
        .expect("64 branches fit under a ceiling of 64");
    assert_eq!((cert.final_chi, cert.max_branches), (64, 64));
    assert!(cert.is_exact(), "raising a ceiling does not cost accuracy");
}
