// SPDX-License-Identifier: Apache-2.0
//! S1 test (i): random **Clifford + T** circuits against the dense
//! statevector — the expectation through the χ² readout at 1e-10, and the
//! amplitudes of the whole decomposition as complex numbers.
//!
//! With `T` unsupported this file does not fail on a tolerance, it fails at
//! `Err`: every call below unwraps, and the S0 engine refused `T` by name.
//! That is the plan's point about this fixture — it cannot be satisfied by a
//! loosened bound.
//!
//! The dense backend is the binding oracle and is entitled to be: its gate
//! matrices are the literal textbook ones (`gates::t()` is `diag(1,
//! e^{iπ/4})`, `gates::rz` is `diag(e^{−iθ/2}, e^{iθ/2})`), so it carries the
//! true global phase and not merely the true ray. §1.3's keystone is why it
//! can certify this engine at all: dense cost is `2^n` **independent of t**,
//! so at `n ≤ 10` it certifies any T-count.
//!
//! # How the sweep is shaped, and why it is not a full grid
//!
//! The two axes have different costs. The dense oracle is `2^n`; the χ²
//! readout is `4^t · poly(n)` — `χ = 2^t` branches and a Gram matrix over
//! them. Measured in this crate's debug profile, one observable term costs
//! 98 ms at `n = 10, t = 6` and 37 s at `n = 10, t = 10`. So both axes are
//! swept to the plan's stated limits, but not simultaneously everywhere: a
//! broad band covers every width to `n = 10` at `t ≤ 5`, a deep band pushes
//! `t` to 8 at four widths, and the corner `n = 10, t = 10` is one named
//! cell of its own. The corner is what makes the limits real rather than
//! nominal, which is why it is here and not trimmed.
//!
//! Every cell is chosen by the oracle rather than written down, through
//! [`live_cell`]. `⟨P⟩` on a stabilizer state is `0` or `±1` and `0` is the
//! common case, so a grid of fixed seeds is mostly cells where both engines
//! return zero and agree for free. That is not a hypothetical: measured on
//! the first version of this file, 163 of the broad band's 240 cells and all
//! four of the deep band's were exactly zero, and the deep band stayed green
//! under a mutation that executed every `T` as an `S`.
//!
//! # Which mutation this fixture catches, and which it does not
//!
//! Catches: everything the S0 amplitude sweep caught, plus `T` and `T†` run
//! at the wrong angle or the wrong sign, the split's `−i sin` coefficient,
//! and the Clifford-angle table — each measured in `mutation_s1_ledger.rs`.
//!
//! Does NOT catch: the S0 ledger's entry **E**, the `e^{iπ/4}` branch factor
//! in `desuperpose` conjugated, anywhere except the off-axis band below. The
//! reasoning that a sum over Cliffords turns a global phase into a relative
//! one is sound and does not apply here: the two branches of a `T` split
//! differ by `Z`, `ChForm::z_gate` touches `γ` and nothing `h_gate` reads, so
//! both branches take the same `desuperpose` decisions forever and the defect
//! stays global. The ledger measures that, and
//! `an_off_axis_split_is_what_makes_a_per_branch_phase_reach_the_readout` is
//! the band that does see it.
//!
//! It also does not catch a defect multiplying the whole sum by one constant
//! — the expectation leg is bilinear and the amplitude leg starts from
//! `ω = 1` on both sides — which is `global_phase_pin.rs`'s job; or an error
//! confined to a gate the generator never emits, which is why
//! `every_non_clifford_gate_in_the_s1_table_is_exercised_against_the_dense_oracle`
//! exists.

use num_complex::Complex64;
use omega_backend_stabrank::StabRankBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, Observable, PauliOp};
use omega_core::params::ParameterBinding;

const TOL: f64 = 1e-10;

const CLIFFORD_1Q: [GateKind; 8] = [
    GateKind::H,
    GateKind::S,
    GateKind::Sdg,
    GateKind::X,
    GateKind::Y,
    GateKind::Z,
    GateKind::Sx,
    GateKind::Sxdg,
];
const CLIFFORD_2Q: [GateKind; 4] = [GateKind::CX, GateKind::CY, GateKind::CZ, GateKind::Swap];
/// Both signs, so a sweep that passed by executing `T†` as `T` would have to
/// pass twice over, once in each direction.
const NON_CLIFFORD: [GateKind; 2] = [GateKind::T, GateKind::Tdg];

/// Deterministic, so a failure names a seed a reader can re-run.
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

/// `depth` random Clifford gates, one- and two-wire mixed.
fn random_clifford_body(n: u32, depth: usize, rng: &mut Lcg) -> Vec<GateOp> {
    let mut ops: Vec<GateOp> = Vec::with_capacity(depth);
    for _ in 0..depth {
        if n >= 2 && rng.below(3) == 0 {
            let a = rng.below(n as usize) as u32;
            let mut b = rng.below(n as usize) as u32;
            while b == a {
                b = rng.below(n as usize) as u32;
            }
            ops.push(op(&CLIFFORD_2Q[rng.below(CLIFFORD_2Q.len())], &[a, b]));
        } else {
            let a = rng.below(n as usize) as u32;
            ops.push(op(&CLIFFORD_1Q[rng.below(CLIFFORD_1Q.len())], &[a]));
        }
    }
    ops
}

/// A random Clifford circuit of `depth` gates with exactly `t` `T`/`T†`
/// gates spliced into it at random positions.
///
/// The T gates go *between* Clifford gates rather than in a block at the
/// end: a block would leave every branch differing by a Pauli on the same
/// final state, which is the one arrangement where the cross terms are
/// easiest to get right by accident.
fn random_clifford_t(n: u32, depth: usize, t: usize, rng: &mut Lcg) -> CircuitIR {
    let mut ops = random_clifford_body(n, depth, rng);
    for _ in 0..t {
        let at = rng.below(ops.len() + 1);
        let a = rng.below(n as usize) as u32;
        ops.insert(at, op(&NON_CLIFFORD[rng.below(NON_CLIFFORD.len())], &[a]));
    }
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for o in ops {
        c.add_op(o);
    }
    c
}

/// The same construction splitting **off** the `Z` axis: `r` generic-angle
/// `Rx`, `Ry` or `Rbs` gates spliced into a Clifford body, with a Hadamard
/// layer at the end.
///
/// The Hadamard layer is not padding. `desuperpose` runs from `h_gate`, so
/// without something after the rotations the branches never reach the step
/// whose phase this circuit exists to expose.
fn random_off_axis(n: u32, depth: usize, r: usize, rng: &mut Lcg) -> CircuitIR {
    const ANGLES: [f64; 4] = [0.37, 1.1, -0.8, 2.3];
    let mut ops = random_clifford_body(n, depth, rng);
    for _ in 0..r {
        let at = rng.below(ops.len() + 1);
        let a = rng.below(n as usize) as u32;
        let th = ANGLES[rng.below(ANGLES.len())];
        if n >= 2 && rng.below(3) == 0 {
            let mut b = rng.below(n as usize) as u32;
            while b == a {
                b = rng.below(n as usize) as u32;
            }
            ops.insert(at, gop(&GateKind::Rbs, &[a, b], &[th]));
        } else {
            let kind = if rng.below(2) == 0 {
                GateKind::Rx
            } else {
                GateKind::Ry
            };
            ops.insert(at, gop(&kind, &[a], &[th]));
        }
    }
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for o in ops {
        c.add_op(o);
    }
    for q in 0..n {
        c.add_op(op(&GateKind::H, &[q]));
    }
    c
}

fn dense_state(c: &CircuitIR) -> Vec<Complex64> {
    let cfg = ExecConfig {
        shots: None,
        ..ExecConfig::default()
    };
    StatevectorBackend::new()
        .execute(c, &ParameterBinding::new(), &cfg)
        .expect("statevector oracle")
        .statevector()
        .to_vec()
}

/// A rotating set of observables, so the sweep reads out along more than one
/// axis. `k = 0` is a single `Z`, which a circuit can happen to make trivial;
/// the rest are multi-wire and weighted.
fn observable_for(n: u32, k: usize) -> Observable {
    let letters = [PauliOp::X, PauliOp::Y, PauliOp::Z];
    let string = |pick: &dyn Fn(usize) -> PauliOp, step: usize| -> Vec<(u32, PauliOp)> {
        (0..n as usize)
            .step_by(step)
            .map(|q| (q as u32, pick(q)))
            .collect()
    };
    match k % 4 {
        0 => Observable::z(0),
        1 => Observable {
            terms: vec![
                (1.0, vec![(0, PauliOp::X)]),
                (-0.5, vec![(n - 1, PauliOp::Y)]),
            ],
        },
        2 => Observable {
            terms: vec![(0.75, string(&|q| letters[q % 3], 1))],
        },
        // A Pauli string names each wire at most once; on one wire the second
        // term collapses to that wire alone rather than repeating it.
        _ => Observable {
            terms: vec![
                (1.0, string(&|q| letters[(q + 1) % 3], 2)),
                (
                    0.25,
                    if n == 1 {
                        vec![(0, PauliOp::Z)]
                    } else {
                        vec![(0, PauliOp::Z), (n - 1, PauliOp::Z)]
                    },
                ),
            ],
        },
    }
}

/// The circuit and observable of one cell, deterministic in `(n, t, seed, k)`
/// so the dense oracle can be asked about a cell before stabrank runs it.
fn cell(n: u32, t: usize, seed: u64, k: usize) -> (CircuitIR, Observable) {
    let mut rng = Lcg(0x51a2_0000 + seed * 7919 + u64::from(n) * 131 + t as u64);
    let c = random_clifford_t(n, 5 + 3 * n as usize, t, &mut rng);
    (c, observable_for(n, k))
}

/// The nearest cell to `(n, t, seed, k)` whose dense expectation is not zero,
/// as `(seed, k)`.
///
/// This search is the difference between a sweep and a sweep that can fail.
/// `⟨P⟩` on a stabilizer state is `0` or `±1`, and `0` is the common case:
/// the string anticommutes with a stabilizer and the expectation vanishes
/// identically. T gates make the value fractional but do nothing to stop it
/// from vanishing. Measured on the first version of this band — fixed seeds,
/// a fixed rotation of observables — **163 of 240 cells returned exactly
/// zero on both engines**, and the four deep cells returned zero on all four.
/// Those cells are two engines agreeing that something is zero, which is true
/// and is not evidence that either one executed a `T`; the band stayed green
/// under a mutation that ran every `T` as an `S`.
///
/// The dense oracle is `2^n` and does not care how many T gates a circuit has
/// (§1.3's keystone), so the search is paid for in the cheap currency and
/// stabrank still runs exactly once per cell.
fn live_cell(n: u32, t: usize, seed: u64, k: usize) -> (u64, usize) {
    let dense = StatevectorBackend::new();
    (0..32u64)
        .find_map(|bump| {
            (0..4usize).find_map(|dk| {
                let (s, k) = (seed + 101 * bump, (k + dk) % 4);
                let (c, obs) = cell(n, t, s, k);
                let v = dense
                    .expectation(&c, &ParameterBinding::new(), &obs)
                    .expect("dense oracle");
                (v.abs() > 0.1).then_some((s, k))
            })
        })
        .unwrap_or_else(|| panic!("n={n} t={t}: no cell near seed {seed} has ⟨O⟩ ≠ 0"))
}

/// One cell: run both engines, assert, and return `|Δ|` and the dense value.
fn compare(n: u32, t: usize, seed: u64, k: usize) -> (f64, f64) {
    let (c, obs) = cell(n, t, seed, k);
    let (got, cert) = StabRankBackend::new()
        .expectation_with_certificate(&c, &ParameterBinding::new(), &obs)
        .unwrap_or_else(|e| {
            panic!(
                "n={n} t={t} seed={seed}: stabrank refused a Clifford+T circuit. \
                 An engine that cannot execute T fails here, at Err, not at a \
                 tolerance. {e}"
            )
        });
    let want = StatevectorBackend::new()
        .expectation(&c, &ParameterBinding::new(), &obs)
        .expect("dense oracle");
    let d = (got - want).abs();
    assert!(
        d < TOL,
        "n={n} t={t} seed={seed} obs={k}: stabrank {got} vs dense {want}, \
         |Δ| = {d:.3e} (χ = {}, peak χ = {})",
        cert.final_chi,
        cert.peak_chi
    );
    assert!(cert.is_exact(), "an S1 run must be exact");
    assert_eq!(cert.state_dropped_mass, 0.0);
    (d, want)
}

/// The broad band: every width to the plan's `n = 10`, `t` to 5, four seeds,
/// the four observables in rotation.
#[test]
fn random_clifford_t_expectations_match_the_dense_statevector() {
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    for n in 1..=10u32 {
        for t in 0..=5usize {
            for seed in 0..4u64 {
                let (seed, k) = live_cell(n, t, seed, (seed as usize + t) % 4);
                let (d, want) = compare(n, t, seed, k);
                assert!(
                    want.abs() > 0.1,
                    "n={n} t={t} seed={seed}: the oracle handed back a cell with \
                     ⟨O⟩ = {want}, which both engines would agree on for free"
                );
                if d > worst {
                    worst = d;
                    worst_at = format!("n={n} t={t} seed={seed} obs={k}");
                }
                compared += 1;
            }
        }
    }
    eprintln!(
        "stabrank Clifford+T broad band: {compared} values, all with |⟨O⟩| > 0.1, \
         worst |Δ| = {worst:.3e} at {worst_at}"
    );
    assert!(compared >= 240, "only {compared} cells compared");
    assert!(
        worst > 0.0,
        "every value agreed bit-for-bit across {compared} cells computed by two \
         different algorithms. That is what an 'engine' that secretly called the \
         dense backend would report."
    );
}

/// The deep band: `t` past the point where χ is the whole cost, at four
/// widths. One seed each — these cells are seconds, not milliseconds.
///
/// The cell is picked by [`live_cell`], and this band is where the reason for
/// that was measured: at the fixed seed it used to carry, all four cells had
/// `⟨O⟩ = 0`, so the band stayed green under a mutation that executed every
/// `T` as an `S` — thirty-two non-Clifford gates replaced by Clifford ones,
/// and four comparisons of zero against zero did not notice.
#[test]
fn deep_t_counts_still_match_the_dense_statevector() {
    let mut worst = 0.0f64;
    for (n, t) in [(2u32, 8usize), (4, 8), (6, 8), (8, 7)] {
        let (seed, k) = live_cell(n, t, 11, 2);
        let (d, want) = compare(n, t, seed, k);
        eprintln!(
            "stabrank deep cell n={n} t={t} seed={seed} obs={k}: ⟨O⟩ = {want:.9}, \
             |Δ| = {d:.3e}"
        );
        assert!(want.abs() > 0.1, "the oracle picked a trivial cell");
        worst = worst.max(d);
    }
    assert!(worst > 0.0, "every deep cell agreed bit-for-bit");
}

/// The corner the plan names: `n = 10`, `t = 10`. χ = 1024 branches and a
/// Gram matrix of 1,048,576 inner products, against a dense oracle that does
/// not care how many T gates there are (§1.3's keystone). One observable
/// term, because in this crate's debug profile this cell is ~37 s.
///
/// The seed is **picked by the oracle**, not written down: a random
/// Clifford+T circuit read out through a Pauli string lands on `⟨O⟩ = 0`
/// often enough that a fixed seed is likely to buy a cell where both engines
/// return exactly zero and agree for free. The dense run is `2^n` and cheap,
/// so the search is cheap; stabrank runs once, on a cell whose answer is
/// known in advance to be worth comparing.
#[test]
fn the_corner_of_the_sweep_is_the_plans_own_limit() {
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Y), (3, PauliOp::Z), (7, PauliOp::X)])],
    };
    let dense = StatevectorBackend::new();
    let (seed, c, want) = (0u64..64)
        .find_map(|seed| {
            let mut rng = Lcg(0x0c04_b1a5_0000_0000 + seed);
            let c = random_clifford_t(10, 35, 10, &mut rng);
            let want = dense
                .expectation(&c, &ParameterBinding::new(), &obs)
                .expect("dense oracle");
            (want.abs() > 0.2).then_some((seed, c, want))
        })
        .expect("no seed in 64 gave a non-trivial dense expectation");

    let (got, cert) = StabRankBackend::new()
        .expectation_with_certificate(&c, &ParameterBinding::new(), &obs)
        .expect("n = 10, t = 10 is inside the plan's stated limits");
    let d = (got - want).abs();
    eprintln!(
        "stabrank corner n=10 t=10 seed={seed}: χ = {}, value {got}, dense {want}, \
         |Δ| = {d:.3e}",
        cert.final_chi
    );
    assert_eq!(cert.final_chi, 1024, "ten T gates are 2^10 branches");
    assert!(
        want.abs() > 0.2,
        "the corner cell must not be the free agreement at ⟨O⟩ = 0"
    );
    assert!(
        d < TOL,
        "n=10 t=10 seed={seed}: stabrank {got} vs dense {want}, |Δ| = {d:.3e}"
    );
}

/// The band above splits along `Z` and nothing else, and that is a blind
/// spot with a proof behind it rather than a suspicion.
///
/// The two branches of a `T` split differ by `Z_q`, and `ChForm::z_gate`
/// touches `γ` alone — not `s`, not `v`, not the `F/G/M` tableau. Every later
/// gate therefore acts identically on the parts of the two branches that
/// `h_gate` reads, so the two take *the same* `desuperpose` decisions for the
/// rest of the run and pick up the same number of `e^{iπ/4}` factors. A
/// defect in that factor is then one scalar multiplying every branch: a
/// global phase, which a bilinear readout cancels exactly. That is S0 ledger
/// entry **E**, and measured against this suite, a Clifford+T expectation
/// does not see it either — the sum over Cliffords is not enough on its own.
///
/// What is enough is a split **off** the `Z` axis. `Rx`, `Ry` and `Rbs` at a
/// generic angle branch by `X` and `Y`, those do move `s`, so the branches
/// diverge in `desuperpose` and the defect becomes a *relative* phase that
/// survives the readout. This fixture is that circuit, and the S1 ledger
/// records that it reddens under E where the T band above stays green.
#[test]
fn an_off_axis_split_is_what_makes_a_per_branch_phase_reach_the_readout() {
    let dense = StatevectorBackend::new();
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    for n in 2..=5u32 {
        for r in 1..=4usize {
            for seed in 0..3u64 {
                let (c, obs, want) = (0u64..32)
                    .find_map(|bump| {
                        let mut rng =
                            Lcg(0x0ffa_0000 + (seed + bump) * 7919 + u64::from(n) * 131 + r as u64);
                        let c = random_off_axis(n, 4 + 2 * n as usize, r, &mut rng);
                        (0..4usize).find_map(|dk| {
                            let obs = observable_for(n, (r + seed as usize + dk) % 4);
                            let want = dense
                                .expectation(&c, &ParameterBinding::new(), &obs)
                                .expect("dense oracle");
                            (want.abs() > 0.1).then_some((c.clone(), obs, want))
                        })
                    })
                    .unwrap_or_else(|| panic!("n={n} r={r} seed={seed}: no live cell"));

                let (got, cert) = StabRankBackend::new()
                    .expectation_with_certificate(&c, &ParameterBinding::new(), &obs)
                    .unwrap_or_else(|e| panic!("n={n} r={r} seed={seed}: {e}"));
                let d = (got - want).abs();
                assert!(
                    d < TOL,
                    "n={n} r={r} seed={seed}: stabrank {got} vs dense {want}, \
                     |Δ| = {d:.3e} (χ = {})",
                    cert.final_chi
                );
                assert!(cert.final_chi > 1, "an off-axis rotation must branch");
                if d > worst {
                    worst = d;
                    worst_at = format!("n={n} r={r} seed={seed}");
                }
                compared += 1;
            }
        }
    }
    eprintln!(
        "stabrank off-axis band: {compared} values, all with |⟨O⟩| > 0.1, \
         worst |Δ| = {worst:.3e} at {worst_at}"
    );
    assert_eq!(compared, 48, "the band did not cover its grid");
    assert!(worst > 0.0, "every off-axis cell agreed bit-for-bit");
}

/// The amplitude leg: the decomposition summed back into a statevector,
/// compared as **complex numbers**. Taking a modulus here would let a
/// phase-blind sum pass, which is the one thing this lane exists to rule out.
///
/// `O(χ·2^n)` rather than `O(χ²)`, so `t` can go further than the expectation
/// legs allow — and this is the leg that sees a per-branch global phase,
/// because the branches are summed before anything is contracted.
#[test]
fn clifford_t_amplitudes_match_the_dense_statevector() {
    let mut compared = 0usize;
    let mut live = 0usize;
    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    for n in 1..=6u32 {
        for t in 0..=10usize {
            for seed in 0..3u64 {
                let mut rng = Lcg(0xa117_0000 + seed * 977 + u64::from(n) * 31 + t as u64);
                let c = random_clifford_t(n, 5 + 3 * n as usize, t, &mut rng);
                let sum = StabRankBackend::new()
                    .simulate_sum(&c, &ParameterBinding::new())
                    .unwrap_or_else(|e| panic!("n={n} t={t} seed={seed}: {e}"));
                for (_, branch) in sum.branches() {
                    branch
                        .check_invariants()
                        .unwrap_or_else(|e| panic!("n={n} t={t} seed={seed}: {e}"));
                }
                let got = sum.to_statevector();
                let want = dense_state(&c);
                assert_eq!(got.len(), want.len());
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    let d = (a - b).norm();
                    if d > worst {
                        worst = d;
                        worst_at = format!("n={n} t={t} seed={seed} basis={i}");
                    }
                    assert!(
                        d < TOL,
                        "n={n} t={t} seed={seed} basis state {i}: stabrank {a} vs \
                         dense {b}, |Δ| = {d:.3e}. This compares the complex \
                         number, so a phase-only disagreement reddens here."
                    );
                    compared += 1;
                    live += usize::from(b.im.abs() > 1e-12);
                }
                // `norm_sqr` goes through the χ² readout with the identity
                // observable — `Σᵢⱼ c̄ᵢcⱼ⟨φᵢ|φⱼ⟩` — so it checks the pairwise
                // machinery, not just the amplitudes above. It also costs χ²,
                // which is why it is spent on the cells where χ is small and
                // the deep cells carry the amplitude leg alone.
                if sum.chi() <= 64 {
                    let norm = sum.norm_sqr();
                    assert!(
                        (norm - 1.0).abs() < TOL,
                        "n={n} t={t} seed={seed}: ‖ψ‖² = {norm}, so the branch \
                         coefficients do not reassemble a state"
                    );
                }
            }
        }
    }
    eprintln!(
        "stabrank Clifford+T amplitudes: {compared} values ({live} with a \
         non-zero imaginary part), worst |Δ| = {worst:.3e} at {worst_at}"
    );
    assert_eq!(
        compared,
        3 * 11 * (2 + 4 + 8 + 16 + 32 + 64),
        "the sweep did not cover every cell of its grid"
    );
    // The count that matters for *this* leg is not how many amplitudes are
    // non-zero but how many are genuinely complex. A sum whose per-branch
    // phases were wrong still gets every real amplitude of a Clifford circuit
    // right; the ones with an imaginary part are the cells where the phase
    // bookkeeping has to be correct to agree. Measured: 1023 of 4158.
    assert!(
        live >= 500,
        "only {live} of {compared} amplitudes have an imaginary part, so this \
         sweep is mostly blind to the phase bookkeeping it exists to check"
    );
    assert!(worst > 0.0, "every amplitude agreed bit-for-bit");
}

/// The sweep above only ever emits `T` and `T†`. Every other entry the S1
/// gate table added is pinned here, alone, against the dense oracle, as a
/// complex amplitude — so the table is exercised rather than merely written.
///
/// Each gate appears at a generic angle **and** at a quarter turn, because
/// those are two different code paths (the 2-term split and the in-place
/// Clifford application) that have to produce the same operator.
#[test]
fn every_non_clifford_gate_in_the_s1_table_is_exercised_against_the_dense_oracle() {
    use std::f64::consts::{FRAC_PI_2, PI};
    let angles = [
        0.37,
        1.0,
        -0.8,
        FRAC_PI_2,
        PI,
        3.0 * FRAC_PI_2,
        2.0 * PI,
        0.0,
    ];
    let one_q = [GateKind::Rz, GateKind::U1, GateKind::Rx, GateKind::Ry];
    let mut seen = 0usize;
    for kind in &one_q {
        for theta in angles {
            // Dressed so the wire is an eigenstate of nothing.
            let mut c = CircuitIR::new(2, CircuitType::GateBased);
            c.add_op(op(&GateKind::H, &[0]));
            c.add_op(op(&GateKind::S, &[0]));
            c.add_op(op(&GateKind::H, &[1]));
            c.add_op(op(&GateKind::CX, &[1, 0]));
            c.add_op(gop(kind, &[0], &[theta]));
            check_against_dense(&c, &format!("{kind:?}({theta})"));
            seen += 1;
        }
    }
    for theta in angles {
        let mut c = CircuitIR::new(3, CircuitType::GateBased);
        c.add_op(op(&GateKind::H, &[0]));
        c.add_op(op(&GateKind::X, &[1]));
        c.add_op(op(&GateKind::S, &[1]));
        c.add_op(op(&GateKind::H, &[2]));
        c.add_op(gop(&GateKind::Rbs, &[0, 2], &[theta]));
        c.add_op(gop(&GateKind::Rbs, &[1, 2], &[theta]));
        check_against_dense(&c, &format!("Rbs({theta})"));
        seen += 1;
    }
    for kind in &NON_CLIFFORD {
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.add_op(op(&GateKind::H, &[0]));
        c.add_op(op(&GateKind::H, &[1]));
        c.add_op(op(&GateKind::CZ, &[0, 1]));
        c.add_op(op(kind, &[0]));
        c.add_op(op(&GateKind::H, &[0]));
        c.add_op(op(kind, &[1]));
        check_against_dense(&c, &format!("{kind:?}"));
        seen += 1;
    }
    assert_eq!(seen, 4 * 8 + 8 + 2);
}

fn check_against_dense(c: &CircuitIR, what: &str) {
    let sum = StabRankBackend::new()
        .simulate_sum(c, &ParameterBinding::new())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    for (_, branch) in sum.branches() {
        branch
            .check_invariants()
            .unwrap_or_else(|e| panic!("{what}: {e}"));
    }
    let got = sum.to_statevector();
    let want = dense_state(c);
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - b).norm() < TOL,
            "{what} amplitude {i}: stabrank {a} vs dense {b}. The comparison is \
             complex, so a global phase this gate owes and did not pay reddens here."
        );
    }
}
