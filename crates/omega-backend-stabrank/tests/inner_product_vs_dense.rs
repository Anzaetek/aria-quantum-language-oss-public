// SPDX-License-Identifier: Apache-2.0
//! The pairwise quantities the whole lane is for: `⟨φ|φ′⟩` and
//! `⟨φ|P|φ′⟩` between **two different** stabilizer states, against the
//! dense statevector, complex, at 1e-12.
//!
//! S1's readout is `Σᵢⱼ c̄ᵢcⱼ⟨φᵢ|O|φⱼ⟩`. The diagonal terms are ordinary
//! expectations and a phase-blind tableau gets them right; the off-diagonal
//! ones are where the relative phase between two different branches lives,
//! and they are what this file checks. `amplitude_vs_statevector.rs` cannot
//! stand in for it: there both legs start from `|0…0⟩` and one state is
//! involved, so no *relative* phase between two states is ever formed.
//!
//! The dense leg builds both circuits with the statevector backend and
//! contracts `Σ_x conj(a_x)·b_x` in f64. For `⟨φ|P|φ′⟩` the Pauli is
//! appended to the ket's circuit as ordinary `x`/`y`/`z` gates, so the
//! oracle applies it with its own matrices rather than with the kernel's.
//!
//! # Which mutation this fixture catches, and which it does not
//!
//! Catches: everything in the `inner_product` path that the amplitude
//! sweep does not reach — the `U_C†` factorisation (the `S^γ` and `CZ`
//! phase polynomial and the CNOT network from `cx_network`), the
//! right-multiplication column rules, and any bra/ket conjugation mix-up.
//!
//! Does NOT catch: an error in the *shared* final step. Both the kernel
//! and this oracle finish by reading amplitudes, so a defect in
//! `ChForm::amplitude` is caught here only because `amplitude_vs_dense`
//! also exists; on its own this file would be comparing a quantity to
//! itself through two routes that share `amplitude`. It also cannot see a
//! constant global phase common to *both* states, which cancels in
//! `conj(a)·b` by construction — that is what `global_phase_pin.rs` is for.

use num_complex::Complex64;
use omega_backend_stabrank::{ChForm, StabRankBackend};
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, ExecConfig, Observable, PauliOp};
use omega_core::params::ParameterBinding;

const TOL: f64 = 1e-12;

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
        .expect("Clifford circuit")
}

fn contract(bra: &[Complex64], ket: &[Complex64]) -> Complex64 {
    bra.iter().zip(ket).map(|(a, b)| a.conj() * b).sum()
}

/// A rotating set of Pauli strings, one per (n, seed), so the sweep covers
/// the identity, single letters and multi-wire products.
fn pauli_for(n: u32, k: usize) -> Vec<(usize, PauliOp)> {
    let letters = [PauliOp::X, PauliOp::Y, PauliOp::Z];
    match k % 5 {
        0 => vec![],
        1 => vec![(0, letters[k % 3])],
        2 => vec![((n - 1) as usize, letters[(k + 1) % 3])],
        3 => (0..n as usize).map(|q| (q, letters[(q + k) % 3])).collect(),
        _ => (0..n as usize)
            .step_by(2)
            .map(|q| (q, letters[(q + k) % 3]))
            .collect(),
    }
}

fn pauli_gates(p: &[(usize, PauliOp)]) -> Vec<GateOp> {
    p.iter()
        .filter_map(|&(q, l)| {
            let kind = match l {
                PauliOp::I => return None,
                PauliOp::X => GateKind::X,
                PauliOp::Y => GateKind::Y,
                PauliOp::Z => GateKind::Z,
            };
            Some(op(&kind, &[q as u32]))
        })
        .collect()
}

/// The sweep. Two independent random Clifford circuits per cell, the plain
/// overlap and a Pauli matrix element, both parts of both.
#[test]
fn pairwise_overlaps_match_the_dense_contraction() {
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    let mut nonzero_imaginary = 0usize;
    for n in 1..=5u32 {
        for seed in 0..40u64 {
            let mut rng = Lcg(0xc0ffee00 + seed * 7919 + u64::from(n));
            let depth = 5 + 3 * n as usize;
            let ca = random_clifford(n, depth, &mut rng);
            let cb = random_clifford(n, depth, &mut rng);
            let (ka, kb) = (kernel(&ca), kernel(&cb));
            ka.check_invariants().unwrap();
            kb.check_invariants().unwrap();
            let (da, db) = (dense(&ca), dense(&cb));

            let p = pauli_for(n, seed as usize);
            let mut cbp = cb.clone();
            for g in pauli_gates(&p) {
                cbp.add_op(g);
            }

            for (what, got, want) in [
                ("⟨φ|φ′⟩", ka.inner_product(&kb), contract(&da, &db)),
                (
                    "⟨φ|P|φ′⟩",
                    ka.pauli_matrix_element(&p, &kb),
                    contract(&da, &dense(&cbp)),
                ),
            ] {
                let d = (got - want).norm();
                if d > worst {
                    worst = d;
                    worst_at = format!("n={n} seed={seed} {what}");
                }
                assert!(
                    d < TOL,
                    "n={n} seed={seed} P={p:?} {what}: kernel {got} vs dense {want}, \
                     |Δ| = {d:.3e}"
                );
                if want.im.abs() > 1e-3 {
                    nonzero_imaginary += 1;
                }
                compared += 1;
            }
        }
    }
    eprintln!(
        "stabrank pairwise: {compared} overlaps, {nonzero_imaginary} with a \
         nonzero imaginary part, worst |Δ| = {worst:.3e} at {worst_at}"
    );
    assert!(compared >= 400, "only {compared} overlaps compared");
    assert!(
        nonzero_imaginary >= 50,
        "only {nonzero_imaginary} of {compared} overlaps had an imaginary part \
         worth asserting on — this sweep would largely pass against a \
         real-valued kernel and is not evidence for the phase"
    );
}

/// The backend door, not just the kernel: `⟨P⟩` on a Clifford circuit
/// against the dense backend's own expectation path.
#[test]
fn backend_expectation_matches_the_dense_backend() {
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    for n in 1..=5u32 {
        for seed in 0..20u64 {
            let mut rng = Lcg(0xabcd_0000 + seed * 65537 + u64::from(n));
            let c = random_clifford(n, 6 + 3 * n as usize, &mut rng);
            for k in 0..5usize {
                let p = pauli_for(n, k);
                if p.is_empty() {
                    continue;
                }
                let obs = Observable {
                    terms: vec![(1.0, p.iter().map(|&(q, l)| (q as u32, l)).collect())],
                };
                let got = StabRankBackend::new()
                    .expectation(&c, &ParameterBinding::new(), &obs)
                    .expect("Clifford circuit, Pauli observable");
                let want = StatevectorBackend::new()
                    .expectation(&c, &ParameterBinding::new(), &obs)
                    .expect("dense oracle");
                let d = (got - want).abs();
                worst = worst.max(d);
                assert!(
                    d < TOL,
                    "n={n} seed={seed} P={p:?}: stabrank {got} vs dense {want}"
                );
                compared += 1;
            }
        }
    }
    eprintln!("stabrank expectation: {compared} values, worst |Δ| = {worst:.3e}");
    assert!(compared >= 300, "only {compared} expectations compared");
}

/// A Clifford expectation is an integer in {−1, 0, +1} on a stabilizer
/// state. Asserting that pins the kernel to exactness rather than to a
/// tolerance, and it is the same property that makes Stim's
/// `peek_observable_expectation` an exact oracle (see
/// `stim_clifford_oracle.rs`).
#[test]
fn clifford_expectations_land_on_exact_integers() {
    for n in 1..=5u32 {
        for seed in 0..20u64 {
            let mut rng = Lcg(0x1234_5678 + seed * 31 + u64::from(n));
            let c = random_clifford(n, 6 + 3 * n as usize, &mut rng);
            let state = kernel(&c);
            for k in 1..5usize {
                let p = pauli_for(n, k);
                let v = state.pauli_expectation(&p);
                assert!(v.im.abs() < 1e-12, "⟨P⟩ is not real: {v}");
                assert!(
                    (v.re - v.re.round()).abs() < 1e-12,
                    "n={n} seed={seed} P={p:?}: ⟨P⟩ = {} is not an integer",
                    v.re
                );
                assert!(v.re.abs() <= 1.0 + 1e-12, "⟨P⟩ = {} is out of range", v.re);
            }
        }
    }
}
