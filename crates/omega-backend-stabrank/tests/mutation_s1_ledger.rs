// SPDX-License-Identifier: Apache-2.0
//! S1 test (v): the mutation ledger, and the local pin that makes its
//! central finding checkable rather than merely recorded.
//!
//! Every mutation below was applied to the source, the whole
//! `omega-backend-stabrank` suite was run with `--no-fail-fast`, and the file
//! was restored and checked byte-for-byte with `cmp`. Baseline: **47 tests,
//! all green**, this file's two included. Line numbers are as of the commit
//! that added this file.
//!
//! One note on method, because it cost a full campaign. The first pass ran
//! the mutations back to back and cargo did not relink every integration test
//! binary between them: mutation G came back with 7 reddened tests where a
//! single clean run gave 17, and the missing ten were the ones in targets
//! that had been silently reused. Every number below is from a campaign that
//! touches the sources before each build. A mutation ledger that measures a
//! stale binary is worse than none, since it reports a gap that is not there
//! and would send the next reader looking for a fixture to strengthen.
//!
//! # The finding that governs this phase
//!
//! S0's ledger closed on entry **E** — the `e^{iπ/4}` branch factor in
//! `desuperpose` conjugated — which reddened exactly three S0 tests and left
//! the entire Stim leg and every expectation green. The reasoning offered for
//! why S1 would catch it is that a global phase on one branch is a *relative*
//! phase between two, and a sum over Cliffords contracts branches against
//! each other.
//!
//! **That reasoning is wrong for a Clifford+T circuit, and the measurement is
//! entry J.** The two branches of a `T` split differ by `Z_q`;
//! `ChForm::z_gate` touches `γ` alone and leaves `s`, `v` and the `F/G/M`
//! tableau untouched; `h_gate` consults exactly those; so both branches take
//! *identical* `desuperpose` decisions for the rest of the run and accumulate
//! the same factor. The defect stays global and the χ² readout cancels it
//! exactly. Splitting over Cliffords is not sufficient — the splits have to
//! leave the `Z` axis, which is why
//! `an_off_axis_split_is_what_makes_a_per_branch_phase_reach_the_readout`
//! exists and why it is the only expectation fixture in this crate that
//! reddens under E.
//!
//! The same structure governs entry I, and more strongly: `sqrt_pauli`'s
//! leftover scalar is applied through `StabilizerSum::clifford`, which by
//! construction touches every branch equally. No expectation fixture can ever
//! see a defect there, at any χ, on any circuit. The amplitude legs are not a
//! second opinion on that class, they are the only one.
//!
//! # The ledger
//!
//! **F — `sum.rs:102`, `quarter_turns`:** `k.rem_euclid(8.0)` →
//! `k.rem_euclid(4.0)`, i.e. the rotation's period read as `2π` instead of
//! `4π`.
//! Reddened, 3 tests: both `quarter_turns` unit tests and
//! `every_non_clifford_gate_in_the_s1_table_is_exercised_against_the_dense_oracle`,
//! on `Rz(2π)`, which is `−I` and came back `+I`.
//! Stayed green: every `T`/`T†` fixture in the crate. `π/4` is not on the
//! grid at all, so the whole Clifford+T lane is untouched — this is a defect
//! only a fixture that *writes down a quarter turn* can see, and the gate
//! table sweep is the one that does.
//!
//! **G — `backend.rs:279`, the `T` arm:** `FRAC_PI_4` → `2.0 * FRAC_PI_4`,
//! i.e. `T` executed as `S` up to phase. §2's `t → s` trap.
//! Reddened, 17 tests — the widest in this ledger: every random band, both
//! fixture-symmetry tests, four of the five branch-count pins, two of the
//! three-way fermionic tests, both of this file's pins, and
//! `a_t_gate_is_refused_by_the_single_branch_door_rather_than_approximated`
//! (the gate stops splitting, so the single-branch door stops refusing it).
//! `final_chi_exceeds_one_and_is_at_most_two_to_the_t` is the count pin doing
//! exactly its job: χ collapses to 1.
//! Stayed green: `the_workload_is_not_trivially_diagonal` and
//! `a_clifford_angle_rotation_costs_no_branch`, which both assert that
//! something is *Clifford*, and this mutation makes one more thing Clifford.
//!
//! This is the entry that found a weakness rather than confirming one. On the
//! first version of the suite it left `deep_t_counts_still_match_the_dense_statevector`
//! green: at the fixed seed that band carried, all four cells had `⟨O⟩ = 0`,
//! so thirty-two non-Clifford gates could become Clifford without moving a
//! number that was zero either way. `⟨P⟩` on a stabilizer state is `0` or
//! `±1` and `0` is the common case; 163 of the broad band's 240 cells were
//! the same. Both bands now pick their cells with the dense oracle, which is
//! `2^n` and does not care about the T count, and every cell has
//! `|⟨O⟩| > 0.1`.
//!
//! **H — `backend.rs:279-280`, the `T` arm:** `FRAC_PI_4` → `-FRAC_PI_4` in
//! both the rotation and the scalar, i.e. `T` executed as `T†`. §2's
//! `t → tdg` trap, and K8's.
//! Reddened, 7 tests. The named one is
//! `on_plus_x_is_blind_to_t_versus_tdg_and_on_plus_i_it_is_y_that_is_blind`,
//! at its `⟨Y₀⟩` leg: stabrank `−0.7071067811865475` against the dense
//! `+0.7071067811865476`. Its `⟨X₀⟩` leg is green under the same mutation,
//! which is the §2 rule measured rather than asserted — `cos θ` cannot tell
//! the two apart and `sin θ` can. The interfering fixture reddens the same
//! way at χ = 32.
//! Stayed green: every branch-count pin. χ is `2^t` either way, so the count
//! leg is blind to this by construction — which is why the count pin is not
//! the only pin.
//!
//! **I — `sum.rs:174`, `sqrt_pauli`'s leftover scalar:**
//! `if dagger { 1.0 } else { -1.0 }` → `{ -1.0 } else { 1.0 }`, i.e.
//! `e^{∓iπ/4}` conjugated.
//! Reddened, 2 tests:
//! `the_clifford_path_and_the_split_path_agree_with_the_rotation_matrix` and
//! `every_non_clifford_gate_in_the_s1_table_is_exercised_against_the_dense_oracle`.
//! Stayed green: **every expectation fixture in the crate**, necessarily —
//! see the section above. Also the whole Clifford+T lane, which never emits a
//! quarter-turn Pauli rotation and so never calls `sqrt_pauli` at all.
//!
//! **J — `chform.rs:463`, `desuperpose`:** `Complex64::new(ISQRT2, ISQRT2)` →
//! `(ISQRT2, -ISQRT2)`. **S0 ledger entry E, reproduced.**
//! Reddened, 6 tests: S0's three, plus
//! `clifford_t_amplitudes_match_the_dense_statevector`,
//! `every_non_clifford_gate_in_the_s1_table_is_exercised_against_the_dense_oracle`,
//! and — the one that is the point —
//! `an_off_axis_split_is_what_makes_a_per_branch_phase_reach_the_readout`,
//! which is an *expectation* fixture. S0 had none that could see this.
//! Stayed green: the entire Clifford+T expectation lane, for the structural
//! reason given above, and `backend_expectation_matches_the_dense_backend`,
//! still, at χ = 1.
//!
//! **K — `sum.rs:296`, `split`:** `Complex64::new(0.0, -(phi/2.0).sin())` →
//! `(0.0, (phi/2.0).sin())`, i.e. `cos·I + i sin·P` for `cos·I − i sin·P`.
//! Reddened, 10 tests: every random band, both fixture-symmetry tests, the
//! rotation-matrix unit test, and `a_clifford_angle_rotation_costs_no_branch`
//! (whose `norm_sqr` and value legs both move).
//! Stayed green: **all four three-way fermionic tests**, and that is the
//! clearest "would not catch" in this ledger. Flipping the sign of the flip
//! leg conjugates every branch coefficient, so a value that comes out real
//! and symmetric across the decomposition does not move; those fixtures read
//! out a Hermitian fermionic observable on a workload whose Givens layer is
//! at quarter turns, and they land on exactly such values. Three independent
//! engines agreeing is strong evidence about the *mapping* between pictures
//! and weak evidence about the sign conventions inside one of them. The
//! random bands are what cover this, and the three-way fixture is not a
//! substitute for them.
//!
//! **L — `sum.rs:348`, `pauli_expectations`:** `let bra = ci.conj();` →
//! `let bra = *ci;`, i.e. the χ² sum taken without conjugating the bra
//! coefficient.
//! Reddened, 15 tests, and ten of them through the `IMAG_TOL` refusal in
//! `expectation_with_certificate` rather than through a tolerance: the Gram
//! matrix stops being Hermitian and the engine says so, with imaginary parts
//! from `3.536e-1` up to `1.000e0`. That refusal is live only because the
//! readout sums the *whole* χ² square instead of half of it; building the
//! symmetry in would have turned this mutation into a clean wrong number.
//! Stayed green: the amplitude leg's per-amplitude comparisons, which never
//! enter the readout (its `norm_sqr` call does, and that is what reddens the
//! test).
//!
//! **M — `sum.rs:87`, `observable_l1_norm`:** `c.abs()` → `*c`, i.e. `Σcᵢ`
//! for `Σ|cᵢ|`.
//! Reddened, 4 tests: both three-way fermionic cells that compare
//! `observable_range` across the two engines, and the two branch-count pins
//! that assert it against a hand-written constant.
//! Stayed green: every value comparison in the crate. `observable_range` does
//! not enter `value`; it is a claim about the observable, and only a fixture
//! that checks the claim can see it. At S1 that is all it can be — the field
//! becomes load-bearing at S2, where `R` multiplies the dropped mass.
//!
//! **N — `sum.rs:302`, `split`'s growth accounting:**
//! `self.branches.len() * (2 - usize::from(drop_keep || drop_flip))` →
//! `self.branches.len()`, i.e. the ceiling tested against the χ before the
//! split rather than after.
//! Reddened, 1 test: `the_branch_ceiling_refuses_and_names_the_way_out`.
//! Stayed green: `the_branch_ceiling_refuses_through_the_door`, the
//! integration pin for the same ceiling. It overshoots far enough that the
//! undercounted ceiling still trips, one rotation later — so the unit test is
//! what makes this mutation local, exactly as the phase-table derivation was
//! at S0.
//!
//! **O — `chform.rs:578`, `bra_program`:** `omega_conj: self.omega.conj()` →
//! `self.omega`.
//! Reddened, 13 tests, including the S0 kernel's own
//! `a_state_overlaps_itself_at_exactly_one`,
//! `clifford_expectations_equal_stims_exactly` and
//! `clifford_expectations_land_on_exact_integers` — this is the one S1
//! addition the S0 Stim leg can see, because it breaks the overlap at χ = 1
//! too.
//! Stayed green: the fixture-symmetry pair and every branch-count pin but
//! one. Those fixtures reach the readout through states whose `ω` is real,
//! where conjugating it is the identity. They are fixtures with hand-computed
//! expected values, which is what makes them sharp on `T` versus `T†` and
//! blind here; the random bands are the other half of that trade.
//!
//! # One mutation each fixture does not catch
//!
//! Not a hypothetical per fixture — the entry above that measured it.
//!
//! * `clifford_t_vs_statevector.rs`, the Clifford+T bands: **I**, and **J**
//!   on everything except the off-axis band. Both are phases that stay
//!   common to every branch of a `Z`-axis split, and a band that only emits
//!   `T` and `T†` cannot produce any other kind.
//! * `clifford_t_vs_statevector.rs`, the off-axis band: **F**. It uses three
//!   fixed generic angles and never writes down a quarter turn, so the
//!   Clifford-angle table is not on its path at all.
//! * `fixture_symmetry.rs`: **O**. Both fixtures reach the readout through
//!   states whose `ω` is real, and conjugating a real number is the identity.
//!   They are also blind to **F**, **M** and **N** — a two-fixture pair with
//!   hand-computed values is sharp on one axis by design.
//! * `branch_count_pin.rs`: **H**. χ is `2^t` whether the engine runs `T` or
//!   `T†`, so the count leg cannot distinguish them at any depth. This is the
//!   fixture pair §2 exists for: the count pin and the symmetry pin are each
//!   blind to what the other sees.
//! * `three_way_fermionic.rs`: **K**, all four tests. Three engines meeting
//!   is evidence about the mapping between pictures, not about a sign
//!   convention inside the χ² sum that leaves a real symmetric value alone.
//! * `mutation_s1_ledger.rs`, this file: **J**, the entry it was written to
//!   explain. Its pins are about how a phase propagates through the readout,
//!   which is arithmetic over `branches()`, and they do not care where the
//!   phase on a branch came from.
//! * The whole suite: a defect that multiplies the *entire* sum by one
//!   constant. Every expectation is bilinear and every amplitude leg starts
//!   from `ω = 1` on both sides. `global_phase_pin.rs` is the S0 fixture that
//!   covers that class, by pinning an absolute phase against a constant
//!   computed by hand.
//!
//! # What this file pins locally
//!
//! A ledger that only records is a comment. The two tests below write the χ²
//! readout a second time, from the public branch API, and use it to check the
//! claim the ledger turns on: a phase applied to *every* branch cancels, and
//! the same phase applied to *one* does not. That is the whole reason entries
//! I and J behave the way they do, and it is measured here rather than
//! argued.
//!
//! What it does NOT pin: the structural claim itself — that a `Z` split keeps
//! both branches on the same `desuperpose` path. That is a statement about
//! which lines of `h_gate` run, and nothing public exposes it. It is held up
//! instead by the contrast between entries J's two halves, which is evidence
//! of a different kind and is why both bands exist.

use num_complex::Complex64;
use omega_backend_stabrank::{PauliSite, StabRankBackend, StabilizerSum};
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::PauliOp;
use omega_core::params::ParameterBinding;

fn op(kind: GateKind, qubits: &[u32]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: Default::default(),
        classical_bit: None,
        condition: None,
    }
}

/// A χ = 8 Clifford+T decomposition with entanglement before and after the
/// non-Clifford gates, so the branches are not all the same state.
fn fixture() -> StabilizerSum {
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    for g in [GateKind::H, GateKind::S] {
        c.add_op(op(g, &[0]));
    }
    c.add_op(op(GateKind::H, &[1]));
    c.add_op(op(GateKind::CX, &[0, 1]));
    c.add_op(op(GateKind::T, &[0]));
    c.add_op(op(GateKind::H, &[2]));
    c.add_op(op(GateKind::CZ, &[1, 2]));
    c.add_op(op(GateKind::T, &[1]));
    c.add_op(op(GateKind::H, &[0]));
    c.add_op(op(GateKind::Tdg, &[2]));
    c.add_op(op(GateKind::CX, &[2, 0]));
    StabRankBackend::new()
        .simulate_sum(&c, &ParameterBinding::new())
        .expect("the fixture is executable")
}

/// `Σᵢⱼ conj(cᵢ·zᵢ)·(cⱼ·zⱼ)·⟨φᵢ|P|φⱼ⟩`, written from the public branch API
/// rather than taken from the engine, with a per-branch phase `z` the caller
/// supplies.
///
/// `z ≡ 1` reproduces [`StabilizerSum::pauli_expectation`], and the first
/// test below is that it does — an independent readout that agreed with the
/// engine only because it *was* the engine would pin nothing.
fn readout(sum: &StabilizerSum, sites: &[PauliSite], z: impl Fn(usize) -> Complex64) -> Complex64 {
    let branches = sum.branches();
    let programs: Vec<_> = branches.iter().map(|(_, s)| s.bra_program()).collect();
    let kets: Vec<_> = branches
        .iter()
        .map(|(_, s)| {
            let mut k = s.clone();
            for &(q, p) in sites {
                match p {
                    PauliOp::I => {}
                    PauliOp::X => k.x_gate(q),
                    PauliOp::Y => k.y_gate(q),
                    PauliOp::Z => k.z_gate(q),
                }
            }
            k
        })
        .collect();

    let mut acc = Complex64::new(0.0, 0.0);
    for (i, (ci, _)) in branches.iter().enumerate() {
        for (j, (cj, _)) in branches.iter().enumerate() {
            acc += (ci * z(i)).conj() * (cj * z(j)) * programs[i].overlap(&kets[j]);
        }
    }
    acc
}

const TERMS: [&[PauliSite]; 4] = [
    &[],
    &[(0, PauliOp::X)],
    &[(1, PauliOp::Y), (2, PauliOp::Z)],
    &[(0, PauliOp::Z), (1, PauliOp::X), (2, PauliOp::Y)],
];

/// The second readout against the engine's, before it is used to claim
/// anything.
#[test]
fn the_readout_written_here_agrees_with_the_engines() {
    let sum = fixture();
    assert_eq!(sum.chi(), 8, "the fixture must actually branch");
    let one = |_: usize| Complex64::new(1.0, 0.0);
    for sites in TERMS {
        let mine = readout(&sum, sites, one);
        let theirs = sum.pauli_expectation(sites);
        assert!(
            (mine - theirs).norm() < 1e-13,
            "{sites:?}: this file's χ² sum {mine} against the engine's {theirs}"
        );
    }
}

/// The claim entries I and J turn on.
///
/// A phase common to every branch appears as `conj(z)·z` in every one of the
/// χ² terms and so cancels. It cancels *algebraically*, not bit-for-bit:
/// `conj(z)·z` for `z = e^{iπ/4}` is `1.0000000000000002` in `f64`, because
/// `cos(π/4)² + sin(π/4)²` is not `1.0` there. The bar is therefore a
/// rounding bar and is stated as one — measured at `1.7e-17`, fourteen orders
/// of magnitude below the bar on the other leg, so the two claims do not meet
/// in the middle.
///
/// A phase on a single branch does not cancel, and the amount it moves the
/// value by is recorded rather than merely asserted non-zero: `1.07e-2` and
/// `4.44e-3` on the two terms that can see it.
///
/// Two kinds of term are excluded from that leg, and counted so the exclusion
/// cannot quietly empty it. One is a term whose value is zero on this
/// fixture — nothing can move what is not there. The other is the identity,
/// `⟨ψ|ψ⟩`, which is blind to a per-branch phase for a reason worth keeping:
/// the branches here are near enough to mutually orthogonal that only the
/// diagonal `i = j` terms survive, and on those the phase cancels whatever it
/// is. Normalisation is not a phase check, and `norm_sqr` elsewhere in this
/// crate should not be read as one.
#[test]
fn a_phase_on_every_branch_cancels_and_a_phase_on_one_does_not() {
    let sum = fixture();
    let one = |_: usize| Complex64::new(1.0, 0.0);
    // The factor `desuperpose` applies, and the one mutation J conjugates.
    let global = |_: usize| Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let single = |i: usize| {
        if i == 0 {
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4)
        } else {
            Complex64::new(1.0, 0.0)
        }
    };

    let mut worst_common = 0.0f64;
    let mut smallest_relative = f64::INFINITY;
    let mut live = 0usize;
    for sites in TERMS {
        let base = readout(&sum, sites, one);
        let common = readout(&sum, sites, global);
        let relative = readout(&sum, sites, single);
        eprintln!(
            "stabrank phase pin {sites:?}: ⟨P⟩ = {base:.6}, common phase moves it \
             by {:.3e}, a phase on branch 0 alone by {:.3e}",
            (common - base).norm(),
            (relative - base).norm()
        );
        worst_common = worst_common.max((common - base).norm());
        if !sites.is_empty() && base.norm() > 0.05 {
            live += 1;
            smallest_relative = smallest_relative.min((relative - base).norm());
        }
    }
    assert!(
        live >= 2,
        "only {live} of the {} terms have a non-zero value on this fixture",
        TERMS.len()
    );
    assert!(
        worst_common < 1e-15,
        "a phase common to all 8 branches moved the readout by {worst_common:.3e}, \
         which is past rounding: the χ² sum is not bilinear in the coefficients"
    );
    assert!(
        smallest_relative > 1e-3,
        "a phase on one branch moved the readout by only {smallest_relative:.3e}, \
         so this fixture does not in fact distinguish a relative phase from a \
         global one and cannot stand behind the ledger's entries I and J"
    );
}
