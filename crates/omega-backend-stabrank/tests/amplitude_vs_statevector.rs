// SPDX-License-Identifier: Apache-2.0
//! S0 test (i): every amplitude of a random Clifford circuit, against the
//! dense statevector, **as a complex number** — real and imaginary parts
//! both, at 1e-12.
//!
//! The dense backend is the binding oracle here and is entitled to be: its
//! gate matrices are the exact textbook ones (`gates::h`, `gates::s`,
//! `gates::sx` is the true `√X`, not `U3(π/2,−π/2,π/2)`), so it carries the
//! true global phase and not merely the true ray. Comparing `|amplitude|`
//! would make this file pass against a phase-blind kernel, which is the
//! whole thing S0 exists to rule out; nothing below takes a modulus.
//!
//! # Which mutation this fixture catches, and which it does not
//!
//! Catches: any error in the CH-form update rules that changes a *relative*
//! phase between basis states, which is every entry of the Aaronson–
//! Gottesman phase table reached by `plain_mult_phase`, every `γ`
//! increment, and the `e^{±iπ/4}` factors in `desuperpose` — see
//! `mutation_phase_table.rs`.
//!
//! Does NOT catch: an error that multiplies the *whole* state by a constant
//! phase and nothing else. Both legs of this comparison start from the same
//! `|0…0⟩` and the comparison is amplitude by amplitude, so a bug that
//! inserts, say, a spurious `i` into `ω` on every `h_gate` would be caught,
//! but one that scales `ω` once at construction would not — the circuits
//! here always begin at `ω = 1`. `global_phase_pin.rs` is the fixture that
//! pins an absolute phase against a hand-computed constant, and
//! `inner_product_closure.rs` is the one that sees a phase on one of two
//! *different* states, which is the quantity stabilizer rank actually sums.

use num_complex::Complex64;
use omega_backend_stabrank::{ChForm, StabRankBackend};
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig};
use omega_core::params::ParameterBinding;

const TOL: f64 = 1e-12;

/// The Clifford generators the S0 gate table accepts. Two-qubit entries are
/// listed once; the generator picks the wire pair.
const ONE_Q: [GateKind; 8] = [
    GateKind::H,
    GateKind::S,
    GateKind::Sdg,
    GateKind::X,
    GateKind::Y,
    GateKind::Z,
    GateKind::Sx,
    GateKind::Sxdg,
];
const TWO_Q: [GateKind; 4] = [GateKind::CX, GateKind::CY, GateKind::CZ, GateKind::Swap];

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
    GateOp {
        gate: kind.clone(),
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: Default::default(),
        classical_bit: None,
        condition: None,
    }
}

fn random_clifford(n: u32, depth: usize, rng: &mut Lcg) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for _ in 0..depth {
        if n >= 2 && rng.below(3) == 0 {
            let a = rng.below(n as usize) as u32;
            let mut b = rng.below(n as usize) as u32;
            while b == a {
                b = rng.below(n as usize) as u32;
            }
            c.add_op(op(&TWO_Q[rng.below(TWO_Q.len())], &[a, b]));
        } else {
            let a = rng.below(n as usize) as u32;
            c.add_op(op(&ONE_Q[rng.below(ONE_Q.len())], &[a]));
        }
    }
    c
}

fn dense(c: &CircuitIR) -> Vec<Complex64> {
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

fn kernel(c: &CircuitIR) -> ChForm {
    StabRankBackend::new()
        .simulate(c, &ParameterBinding::new())
        .expect("the circuit is Clifford, so the kernel must accept it")
}

/// (i). 6 widths × 80 seeds, every amplitude of each.
#[test]
fn random_clifford_amplitudes_match_the_dense_statevector() {
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    for n in 1..=6u32 {
        for seed in 0..80u64 {
            let mut rng = Lcg(0x5eed_0000 + seed * 977 + u64::from(n));
            let depth = 6 + 4 * n as usize;
            let c = random_clifford(n, depth, &mut rng);
            let want = dense(&c);
            let state = kernel(&c);
            state
                .check_invariants()
                .unwrap_or_else(|e| panic!("n={n} seed={seed}: {e}"));
            let got = state.to_statevector();
            assert_eq!(got.len(), want.len());
            for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                let d = (a - b).norm();
                if d > worst {
                    worst = d;
                    worst_at = format!("n={n} seed={seed} basis={i}");
                }
                assert!(
                    d < TOL,
                    "n={n} seed={seed} basis state {i}: kernel {a} vs dense {b}, \
                     |Δ| = {d:.3e}. Note this compares the complex number, so a \
                     phase-only disagreement reddens here."
                );
                compared += 1;
            }
        }
    }
    eprintln!("stabrank vs dense: {compared} amplitudes, worst |Δ| = {worst:.3e} at {worst_at}");
    assert!(
        compared >= 10_000,
        "only {compared} amplitudes compared — the sweep collapsed"
    );
    assert!(
        worst > 0.0,
        "every amplitude agreed bit-for-bit. That is not plausible across \
         {compared} values computed by two different algorithms, and it is what \
         a kernel that secretly called the dense backend would report."
    );
}

/// The sweep above would still pass if some gate never appeared in it. Each
/// generator is pinned alone, on a state where it is not the identity.
#[test]
fn every_gate_in_the_s0_table_is_exercised_against_the_dense_oracle() {
    // A Hadamard in front so no single-qubit gate acts on a Z eigenstate,
    // where S, Z and Sdg are all indistinguishable from the identity.
    for kind in &ONE_Q {
        let mut c = CircuitIR::new(1, CircuitType::GateBased);
        c.add_op(op(&GateKind::H, &[0]));
        c.add_op(op(kind, &[0]));
        let want = dense(&c);
        let got = kernel(&c).to_statevector();
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).norm() < TOL,
                "{kind:?} amplitude {i}: kernel {a} vs dense {b}"
            );
        }
    }
    for kind in &TWO_Q {
        // H on both wires: a product of |+⟩ is not an eigenstate of any of
        // CX, CY, CZ or SWAP, so each has somewhere to move.
        let mut c = CircuitIR::new(2, CircuitType::GateBased);
        c.add_op(op(&GateKind::H, &[0]));
        c.add_op(op(&GateKind::H, &[1]));
        c.add_op(op(&GateKind::S, &[1]));
        c.add_op(op(kind, &[0, 1]));
        let want = dense(&c);
        let got = kernel(&c).to_statevector();
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).norm() < TOL,
                "{kind:?} amplitude {i}: kernel {a} vs dense {b}"
            );
        }
    }
}

/// The kernel is a state, not a ray: the norm is 1 and `ω` is a unit
/// complex number after any Clifford sequence. A `desuperpose` branch that
/// dropped a `1/√2` would survive a ratio comparison and dies here.
#[test]
fn the_state_stays_normalised_and_the_global_phase_stays_on_the_unit_circle() {
    for n in 1..=5u32 {
        for seed in 0..25u64 {
            let mut rng = Lcg(0xbeef_0000 + seed * 131 + u64::from(n));
            let c = random_clifford(n, 30, &mut rng);
            let state = kernel(&c);
            let norm: f64 = state.to_statevector().iter().map(|a| a.norm_sqr()).sum();
            assert!(
                (norm - 1.0).abs() < 1e-12,
                "n={n} seed={seed}: ‖φ‖² = {norm}"
            );
            let w = state.global_phase().norm();
            assert!(
                (w - 1.0).abs() < 1e-12,
                "n={n} seed={seed}: |ω| = {w}, so a branch scaled the state"
            );
        }
    }
}

/// `execute()` refuses and says what to use instead — the majoranaprop
/// contract, asserted rather than assumed.
#[test]
fn execute_refuses_and_names_the_alternative() {
    let c = CircuitIR::new(1, CircuitType::GateBased);
    let err = StabRankBackend::new()
        .execute(&c, &ParameterBinding::new(), &ExecConfig::default())
        .expect_err("stabrank must not sample");
    let msg = err.to_string();
    assert!(
        msg.contains("expectation") && msg.contains("statevector"),
        "the refusal must name both the supported entry point and a backend \
         that can sample; got: {msg}"
    );
}

/// A `T` is refused by the **single-branch** door, by name, and the message
/// names the door that does execute it. It is never silently executed as
/// `S`, which is the Stim tag-dialect hazard (plan §1.3, K8 trap 2)
/// reappearing one layer in.
///
/// S0 asserted the message named S1 as the phase that *would* support `T`.
/// S1 has landed, so what the message has to name is the entry point, and
/// the claim this fixture makes is narrower than it was: `simulate` carries
/// one CH form and a `T` does not fit in one. The ban on executing `T` as
/// something else is unchanged and is now also held by
/// `clifford_t_vs_statevector.rs`, which would catch a `T` run as an `S`
/// against the dense oracle rather than against a string.
#[test]
fn a_t_gate_is_refused_by_the_single_branch_door_rather_than_approximated() {
    let mut c = CircuitIR::new(1, CircuitType::GateBased);
    c.add_op(op(&GateKind::H, &[0]));
    c.add_op(op(&GateKind::T, &[0]));
    let err = StabRankBackend::new()
        .simulate(&c, &ParameterBinding::new())
        .expect_err("T does not fit in one CH-form branch");
    let msg = err.to_string();
    assert!(
        msg.contains("T") && msg.contains("simulate_sum"),
        "the refusal must name the gate and the door that executes it; got: {msg}"
    );
    // And the sum door does execute it, on the same circuit.
    let sum = StabRankBackend::new()
        .simulate_sum(&c, &ParameterBinding::new())
        .expect("the sum door carries T");
    assert_eq!(sum.chi(), 2, "one T is one 2-term split");
}
