// SPDX-License-Identifier: Apache-2.0
//! S2 test (i): the **instantiation witness** — PLAN-MAJORANA-STIM.md §2's
//! standing rule, and the point of this phase.
//!
//! The rule is written against the vacuous-adequacy-theorem lesson: three
//! sibling-repo theorems type-checked sorry-free with an unsatisfiable
//! premise. A certificate claim ships with a concrete run where all three of
//!
//! 1. `state_dropped_mass > 0` — truncation actually fired;
//! 2. the bound is informative, `expectation_error_bound < R + |value|`;
//! 3. `0 < |value − dense| ≤ expectation_error_bound`
//!
//! hold **together**, because each one alone can pass for a reason that has
//! nothing to do with the bound being right. A run where nothing was dropped
//! satisfies (3) trivially; a bound of `10¹⁰⁰` satisfies (3) and fails (2); a
//! value that happens to be exact satisfies (2) and (3) and proves that the
//! truncation did not reach the observable rather than that the bound holds.
//! Every cell of the sweep below asserts all three, and the error floor in
//! (3) is what makes it a measurement.
//!
//! # The fixture, and why its angles are what they are
//!
//! `t = 12` non-Clifford gates at `n ≤ 10`, cut to `χ = 3`: a `T`, a `T†` and
//! ten `Rz(0.002)`, each followed by a short random Clifford layer, with a
//! Hadamard layer at each end. Untruncated this is `χ = 2¹² = 4096` branches;
//! the cut holds `χ` at 3 and peak `χ` at 6.
//!
//! The ten small angles are not padding, and they are not a free choice.
//! Being informative means `R·m·(2+m) < R + |v|`, and `|v| ≤ R` always, so
//! **no run with `m ≥ √3 − 1 ≈ 0.732` can be informative at all**, and one
//! with a small value needs `m < √2 − 1 ≈ 0.414`. Twelve `T` gates carry a
//! total L1 coefficient mass of `(cos π/8 + sin π/8)¹² = 24.74`, so an
//! informative bound permits discarding at most 1.7% of it — which means
//! *keeping* some 3300 of the 4096 branches. "Twelve `T` gates" and "small
//! `χ`" are not simultaneously satisfiable under naive sum-over-Cliffords
//! truncation, and that is a fact about the naive decomposition (whose mass
//! grows as `1.307^t`), not about the bound. The fixture spends its mass
//! budget on two `π/4` splits and keeps the other ten cheap: `m = 0.162053`,
//! of which `0.146447 = sin²(π/8)` is the single branch the first cut
//! dropped and `0.015607` is the ten small ones together.
//!
//! # Why the mass is deliberately concentrated in one branch
//!
//! A bound nothing can get near cannot detect being weakened, and mutation
//! (ii) — `R·m·(2+m)/4` — is exactly that test. The chain
//! `|Δ⟨O⟩| ≤ ‖O‖·‖ψ−ψ′‖·(‖ψ‖+‖ψ′‖) ≤ R·m·(2+m)` has three places to be
//! loose, and the fixture closes two of them:
//!
//! * `‖O‖ ≤ R` is an **equality** here: every observable is a single Pauli
//!   string with coefficient 1, so `‖O‖ = R = 1`. A weighted multi-term
//!   observable would give the bound slack worth nothing.
//! * `‖ψ − ψ′‖ ≤ m` is an equality only when the dropped branches are
//!   parallel; `N` comparable orthogonal ones lose a factor `√N`. The cut at
//!   `χ = 3` after the second `π/4` split drops **one** branch carrying 90% of
//!   the mass, so this step is nearly tight. Measured: with three `π/4`
//!   splits and the cut at six — the same mass spread over three branches —
//!   only 11 of 52 cells stayed above a quarter of their bound, against 21 of
//!   40 here.
//! * Cauchy–Schwarz on `⟨ψ|O|ψ−ψ′⟩` is the one left, and it is seed
//!   dependent: `⟨φ_dropped|P|ψ⟩` between stabilizer states is `0` or a power
//!   of `1/√2` times a phase. That is why this is a sweep and not a cell.
//!
//! The ratio `|Δ| / bound` over the 40 cells runs from `0.0612` to `0.7748`,
//! and 21 of the 40 clear a quarter. Two of them are named —
//! [`FIRST_SHARP_CELL`], where a quartered bound reddens conjunct (3), and
//! [`SHARPEST_CELL`], which measures how much weakening this fixture can see
//! at all.
//!
//! # What this fixture does NOT catch
//!
//! The S1 ledger's entry **I** — `sqrt_pauli`'s leftover `e^{∓iπ/4}`
//! conjugated. That scalar is applied through `StabilizerSum::clifford`,
//! which touches every branch equally, so it is a global phase and **no**
//! expectation fixture at any `χ` can see it; the amplitude legs of
//! `clifford_t_vs_statevector.rs` are the only cover. Truncation does not
//! change that: dropping branches does not make a common factor relative.
//!
//! It is also blind to a defect in the `max_branches` ceiling, which it never
//! approaches, and to the vacuity gate, which it never trips — those are
//! `sum.rs`'s unit tests and `vacuity_refusal.rs`.

use omega_backend_stabrank::StabRankBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

/// Non-Clifford gates in the witness circuit: `2^12 = 4096` branches exact.
const T_COUNT: usize = 12;
/// How many of them are `π/4` splits — a `T` and a `T†`. The rest are
/// [`TINY`], for the mass-budget reason in this file's doc comment.
const HEAVY: usize = 2;
/// The small `Rz` angle. Small enough that ten of its splits together cost
/// a tenth of what one `π/4` split costs.
const TINY: f64 = 0.002;
/// The truncating ceiling. Three branches, so the cut after the second `π/4`
/// split discards exactly one — the concentration the bound's sharpness
/// needs.
const MAX_CHI: usize = 3;

/// **The cell that reddens mutation (ii)**, `R·m·(2+m)/4`: the first in sweep
/// order whose true error exceeds a quarter of its bound, so it is where a
/// quartered bound stops containing the error and conjunct (3) fails. Named
/// so the mutation has a cell to re-run rather than a sweep to bisect.
/// Measured: `|Δ| = 1.0355e-1` against a quartered bound of `8.7592e-2`.
const FIRST_SHARP_CELL: (u32, u64) = (4, 0);
/// The sweep's **sharpest** cell, `|Δ| = 2.7145e-1` of a `3.5037e-1` bound.
/// It is the one that says how much weakening this fixture can see at all: a
/// bound divided by anything over `1/0.7748 = 1.29` fails here.
const SHARPEST_CELL: (u32, u64) = (4, 3);
/// What [`FIRST_SHARP_CELL`] has to clear for a quartered bound to be caught.
const SHARP_RATIO: f64 = 0.25;
/// What [`SHARPEST_CELL`] measured, as a floor. Both are asserted: the first
/// is the one the mutation trips, the second is the fixture's reach.
const SHARPEST_RATIO: f64 = 0.77;

/// Deterministic, so a failure names a cell a reader can re-run.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, k: usize) -> usize {
        (self.next_u64() % k as u64) as usize
    }
}

const CLIFFORD_1Q: [GateKind; 6] = [
    GateKind::H,
    GateKind::S,
    GateKind::Sdg,
    GateKind::X,
    GateKind::Y,
    GateKind::Z,
];
const CLIFFORD_2Q: [GateKind; 3] = [GateKind::CX, GateKind::CY, GateKind::CZ];
/// All three axes, so the small splits are not all on `Z`. A `Z`-axis split
/// leaves both branches on the same `desuperpose` path (the S1 ledger's entry
/// J), and a fixture that only ever split along `Z` would be making a weaker
/// circuit than it looks like it is making.
const AXES: [GateKind; 3] = [GateKind::Rx, GateKind::Ry, GateKind::Rz];

fn gop(kind: &GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind.clone(),
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

fn clifford_layer(c: &mut CircuitIR, n: u32, depth: usize, rng: &mut Lcg) {
    for _ in 0..depth {
        if n >= 2 && rng.below(3) == 0 {
            let a = rng.below(n as usize) as u32;
            let mut b = rng.below(n as usize) as u32;
            while b == a {
                b = rng.below(n as usize) as u32;
            }
            c.add_op(gop(
                &CLIFFORD_2Q[rng.below(CLIFFORD_2Q.len())],
                &[a, b],
                &[],
            ));
        } else {
            let a = rng.below(n as usize) as u32;
            c.add_op(gop(&CLIFFORD_1Q[rng.below(CLIFFORD_1Q.len())], &[a], &[]));
        }
    }
}

/// `t` non-Clifford gates — the first [`HEAVY`] of them a `T` and a `T†`, the
/// rest `Rz`/`Rx`/`Ry` at [`TINY`] — interleaved with random Clifford layers.
///
/// The heavy splits come first on purpose. Truncation runs after every split,
/// so the order decides which branch the cut finds smallest: with the two
/// `π/4` splits first, `χ` reaches 4 and the cut drops the one branch that
/// took the minority leg of both, `sin²(π/8)`. Putting them last would spread
/// the same mass over the small splits' branches and loosen the bound.
fn witness_circuit(n: u32, seed: u64, t: usize) -> CircuitIR {
    let mut rng = Lcg(0x5212_0000 + seed * 7919 + u64::from(n) * 131);
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for q in 0..n {
        c.add_op(gop(&GateKind::H, &[q], &[]));
    }
    clifford_layer(&mut c, n, 3 + n as usize, &mut rng);
    for i in 0..HEAVY {
        let a = rng.below(n as usize) as u32;
        let kind = if i == 0 { GateKind::T } else { GateKind::Tdg };
        c.add_op(gop(&kind, &[a], &[]));
        clifford_layer(&mut c, n, 3, &mut rng);
    }
    for _ in 0..(t - HEAVY) {
        let a = rng.below(n as usize) as u32;
        c.add_op(gop(&AXES[rng.below(3)], &[a], &[TINY]));
        clifford_layer(&mut c, n, 2, &mut rng);
    }
    for q in 0..n {
        c.add_op(gop(&GateKind::H, &[q], &[]));
    }
    c
}

/// How many gates in the circuit split a branch in two: everything but the
/// Cliffords. Counted from the IR rather than assumed, so the `χ = 2^t` the
/// doc comment claims is a property of the fixture and not of its author.
fn non_clifford_count(c: &CircuitIR) -> usize {
    c.ops
        .iter()
        .filter(|op| {
            matches!(
                op.gate,
                GateKind::T | GateKind::Tdg | GateKind::Rx | GateKind::Ry | GateKind::Rz
            )
        })
        .count()
}

/// A single Pauli string with coefficient 1, so `R = ‖O‖ = 1` exactly — the
/// first of the bound's three slack points, closed. Weight 1 to 3: `⟨P⟩` on a
/// near-stabilizer state vanishes identically more often the longer `P` is,
/// and a cell where both engines return zero is not a comparison.
fn pauli_string(n: u32, j: usize) -> Observable {
    let letters = [PauliOp::X, PauliOp::Y, PauliOp::Z];
    let mut rng = Lcg(0x0b5e_0000 + j as u64 * 2654435761 + u64::from(n));
    let weight = 1 + rng.below(3).min(n as usize - 1);
    let mut wires: Vec<u32> = Vec::new();
    while wires.len() < weight {
        let q = rng.below(n as usize) as u32;
        if !wires.contains(&q) {
            wires.push(q);
        }
    }
    wires.sort_unstable();
    let sites: Vec<(u32, PauliOp)> = wires
        .into_iter()
        .map(|q| (q, letters[rng.below(3)]))
        .collect();
    Observable {
        terms: vec![(1.0, sites)],
    }
}

/// The nearest cell to `(n, seed)` whose **dense** expectation is not zero.
///
/// S1's mutation G is why this search exists and why the threshold is `0.1`
/// rather than `0`: `⟨P⟩` on a stabilizer state is `0` or `±1` and `0` is the
/// common case, so a grid of fixed cells is mostly cells where both engines
/// return zero and agree for free. Measured on the first version of this
/// file, 13 of 40 cells had no live observable among 24 candidates at all.
/// The dense oracle is `2^n` and does not care about `t` (§1.3's keystone),
/// so this is paid for in the cheap currency.
///
/// Here it does a second job. Conjunct (3) needs `|value − dense| > 0`, and a
/// cell whose exact value is zero is a cell where truncation has very little
/// to move.
fn live_cell(n: u32, seed: u64) -> (u64, usize, CircuitIR, Observable, f64) {
    let dense = StatevectorBackend::new();
    let params = ParameterBinding::new();
    for bump in 0..16u64 {
        let s = seed + 1009 * bump;
        let c = witness_circuit(n, s, T_COUNT);
        for j in 0..24usize {
            let o = pauli_string(n, j);
            let want = dense.expectation(&c, &params, &o).expect("dense oracle");
            if want.abs() > 0.1 {
                return (s, j, c, o, want);
            }
        }
    }
    panic!("n={n} seed={seed}: no cell within 16 seed bumps has |⟨O⟩| > 0.1")
}

/// `m` re-derived in closed form from the angle sequence, independently of the
/// engine.
///
/// This is the strong form of conjunct (1): "something was dropped" would be
/// satisfied by a cut that threw away the wrong branches. The coefficient
/// magnitudes in this decomposition do not depend on the Clifford layers at
/// all — a Clifford moves no coefficient — so the whole cut is predictable:
/// the two `π/4` splits give `{c², cs, cs, s²}` for `c = cos(π/8)`,
/// `s = sin(π/8)`, a ceiling of 3 drops `s²`, and each small split thereafter
/// drops `sin(θ/2)` times the surviving mass and scales it by `cos(θ/2)`.
fn predicted_dropped_mass() -> f64 {
    use std::f64::consts::FRAC_PI_8;
    let (c, s) = (FRAC_PI_8.cos(), FRAC_PI_8.sin());
    let (ct, st) = ((TINY / 2.0).cos(), (TINY / 2.0).sin());
    assert_eq!(
        MAX_CHI, 3,
        "the derivation below is written for a cut at three"
    );
    let mut surviving = c * c + 2.0 * c * s;
    let mut m = s * s;
    for _ in 0..(T_COUNT - HEAVY) {
        m += st * surviving;
        surviving *= ct;
    }
    m
}

/// **The witness.** Forty cells, every one of them asserting all three
/// conjuncts, plus the two structural facts that make the conjuncts mean what
/// they say: the state was not renormalised, and `χ` was held at the ceiling
/// rather than never reaching it.
///
/// If the measured error ever exceeds the bound this test fails and names the
/// cell. That is an unsound certificate, and no tolerance here may be
/// loosened to make it pass.
#[test]
fn the_instantiation_witness_asserts_all_three_conjuncts_together() {
    let params = ParameterBinding::new();
    let predicted = predicted_dropped_mass();
    let mut cells = 0usize;
    let mut sharp = 0usize;
    let mut worst_ratio = 0.0f64;
    let mut smallest_error = f64::INFINITY;
    let mut closest_to_unit_norm = f64::INFINITY;
    let mut first_sharp_ratio = None;
    let mut sharpest_ratio = None;

    for n in [4u32, 6, 8, 10] {
        for seed in 0..10u64 {
            let (s, j, circuit, observable, dense) = live_cell(n, seed);
            let at = format!("n={n} seed={seed} (circuit seed {s}, observable {j})");
            assert_eq!(
                non_clifford_count(&circuit),
                T_COUNT,
                "{at}: the fixture must carry {T_COUNT} branching gates, so that \
                 χ = 2^{T_COUNT} is what the cut at {MAX_CHI} is cutting"
            );

            let (value, cert) = StabRankBackend::with_truncation(0.0, Some(MAX_CHI))
                .expectation_with_certificate(&circuit, &params, &observable)
                .unwrap_or_else(|e| panic!("{at}: the truncating engine refused: {e}"));

            // (1) Truncation fired — and dropped the branches it was supposed
            // to, not merely some.
            assert!(
                cert.state_dropped_mass > 0.0,
                "{at}: state_dropped_mass is {}, so nothing was discarded and \
                 this run witnesses nothing about a truncation bound",
                cert.state_dropped_mass
            );
            assert!(
                (cert.state_dropped_mass - predicted).abs() < 1e-12,
                "{at}: m = {} where the angle sequence predicts {predicted}. The \
                 cut discarded a different set of branches than the one the \
                 bound was reasoned about.",
                cert.state_dropped_mass
            );
            assert!(!cert.is_exact(), "{at}: is_exact with a positive m");

            // (2) The bound excludes something.
            assert!(
                cert.is_informative(),
                "{at}: bound {} against R + |v| = {}",
                cert.expectation_error_bound,
                cert.observable_range + value.abs()
            );
            assert!(
                cert.expectation_error_bound < cert.observable_range + value.abs(),
                "{at}: is_informative disagrees with the condition its doc states"
            );

            // (3) The true error is non-zero and inside the bound.
            let error = (value - dense).abs();
            assert!(
                error > 0.0,
                "{at}: stabrank returned the dense value {dense} bit-for-bit \
                 after discarding a mass of {}. Either the dropped branches \
                 never reached this observable — in which case the cell is not \
                 a witness — or the engine did not truncate.",
                cert.state_dropped_mass
            );
            assert!(
                error <= cert.expectation_error_bound,
                "UNSOUND CERTIFICATE at {at}: |value − dense| = {error:.6e} \
                 exceeds expectation_error_bound = {:.6e} (m = {}, R = {}, \
                 value = {value}, dense = {dense}). The bound is not a bound. \
                 Do not widen it; the derivation or the accounting is wrong.",
                cert.expectation_error_bound,
                cert.state_dropped_mass,
                cert.observable_range
            );

            // The truncated state is raw. A renormalising engine reports
            // exactly 1.0 here at every cell, which is the one value this
            // cannot be.
            assert!(
                (cert.truncated_norm_sqr - 1.0).abs() > 1e-3,
                "{at}: ⟨ψ′|ψ′⟩ = {} after discarding a mass of {}. The bound's \
                 (2+m) factor comes from ‖ψ′‖ ≤ ‖ψ‖ + m, so a renormalised \
                 state satisfies a different inequality than the one reported.",
                cert.truncated_norm_sqr,
                cert.state_dropped_mass
            );
            assert_eq!(cert.final_chi, MAX_CHI, "{at}: χ off its ceiling");
            assert_eq!(
                cert.peak_chi,
                2 * MAX_CHI,
                "{at}: peak χ must be the width before the cut"
            );
            assert_eq!(cert.max_chi, Some(MAX_CHI));
            assert_eq!(cert.coeff_min, 0.0);

            let ratio = error / cert.expectation_error_bound;
            cells += 1;
            sharp += usize::from(ratio > SHARP_RATIO);
            worst_ratio = worst_ratio.max(ratio);
            smallest_error = smallest_error.min(error);
            closest_to_unit_norm = closest_to_unit_norm.min((cert.truncated_norm_sqr - 1.0).abs());
            if (n, seed) == FIRST_SHARP_CELL || (n, seed) == SHARPEST_CELL {
                eprintln!(
                    "stabrank S2 witness, named cell {at}: m = {}, bound = {}, \
                     |value − dense| = {error:.6e}, ratio = {ratio:.4}, \
                     ⟨ψ′|ψ′⟩ = {}",
                    cert.state_dropped_mass, cert.expectation_error_bound, cert.truncated_norm_sqr
                );
            }
            if (n, seed) == FIRST_SHARP_CELL {
                first_sharp_ratio = Some(ratio);
            }
            if (n, seed) == SHARPEST_CELL {
                sharpest_ratio = Some(ratio);
            }
        }
    }

    eprintln!(
        "stabrank S2 witness: {cells} cells, m = {predicted}, bound = {}, \
         |Δ|/bound from {:.4} to {worst_ratio:.4}, {sharp} above {SHARP_RATIO}, \
         smallest |Δ| = {smallest_error:.4e}, closest ⟨ψ′|ψ′⟩ to 1 off by {:.4}",
        predicted * (2.0 + predicted),
        smallest_error / (predicted * (2.0 + predicted)),
        closest_to_unit_norm
    );
    assert_eq!(cells, 40, "the sweep did not cover its grid");

    // Mutation (ii): the bound weakened to R·m·(2+m)/4. These two assertions
    // are what keep that from surviving. Conjunct (3) above is where it
    // actually reddens — at FIRST_SHARP_CELL, in sweep order — and these say
    // the sweep still contains a cell sharp enough for it to. Without one the
    // bound would be unfalsifiable from below: any amount of weakening would
    // leave every error in the sweep inside the weakened bound.
    let first = first_sharp_ratio.expect("FIRST_SHARP_CELL is in the grid");
    assert!(
        first > SHARP_RATIO,
        "the named cell n={} seed={} reaches only {first:.4} of its bound, so a \
         bound quartered at backend.rs would still contain its error and \
         conjunct (3) would not fire there.",
        FIRST_SHARP_CELL.0,
        FIRST_SHARP_CELL.1
    );
    let sharpest = sharpest_ratio.expect("SHARPEST_CELL is in the grid");
    assert!(
        sharpest > SHARPEST_RATIO,
        "the sharpest cell n={} seed={} reaches {sharpest:.4} of its bound, \
         below the {SHARPEST_RATIO} it measured. That number is this fixture's \
         reach: it can only see a bound inflated or weakened by more than \
         1/{sharpest:.4}.",
        SHARPEST_CELL.0,
        SHARPEST_CELL.1
    );
}

/// The control: the same construction with the knobs off is exact, and says
/// so in every field.
///
/// `t` is 5 rather than 12 because an untruncated `t = 12` run is 4096
/// branches and 16.7M pairwise overlaps. What this rules out is the reading
/// under which the sweep above measures a constant engine error rather than a
/// truncation error: at `χ = 32` the same kind of circuit agrees with the
/// dense oracle to 1e-10, and at `χ = 3` it does not.
#[test]
fn with_the_knobs_off_the_same_construction_is_exact_and_its_bound_is_zero() {
    let params = ParameterBinding::new();
    let dense = StatevectorBackend::new();
    let mut compared = 0usize;
    for n in [4u32, 6] {
        for seed in 0..4u64 {
            let circuit = witness_circuit(n, seed, 5);
            let observable = pauli_string(n, (seed as usize + n as usize) % 24);
            let want = dense
                .expectation(&circuit, &params, &observable)
                .expect("dense oracle");
            let at = format!("n={n} seed={seed}");

            let (value, cert) = StabRankBackend::new()
                .expectation_with_certificate(&circuit, &params, &observable)
                .unwrap_or_else(|e| panic!("{at}: {e}"));
            assert_eq!(cert.state_dropped_mass, 0.0, "{at}");
            assert_eq!(cert.expectation_error_bound, 0.0, "{at}");
            assert!(cert.is_exact() && cert.is_informative(), "{at}");
            assert_eq!(cert.max_chi, None, "{at}");
            assert_eq!(cert.final_chi, 32, "{at}: five splits are 2^5 branches");
            assert!(
                (cert.truncated_norm_sqr - 1.0).abs() < 1e-10,
                "{at}: an exact run must still be a unit vector, got {}",
                cert.truncated_norm_sqr
            );
            assert!(
                (value - want).abs() < 1e-10,
                "{at}: exact run {value} against dense {want}"
            );

            // And the same circuit truncated does drop something, so the
            // difference between the two runs is the knob and not the depth.
            let (_, cut) = StabRankBackend::with_truncation(0.0, Some(MAX_CHI))
                .expectation_with_certificate(&circuit, &params, &observable)
                .unwrap_or_else(|e| panic!("{at} truncated: {e}"));
            assert!(
                cut.state_dropped_mass > 0.0,
                "{at}: the cut at {MAX_CHI} dropped nothing from a 32-branch sum"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 8);
}
