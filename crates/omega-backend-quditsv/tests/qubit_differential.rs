// SPDX-License-Identifier: Apache-2.0
//! **Leg (ii): on `d = 2`, index for index against the dense qubit engine.**
//! (PLAN-QUDIT.md Q2)
//!
//! Random circuits over every qubit gate this engine embeds, compared with
//! `omega-backend-statevector` at 1e-12 on the full statevector and on
//! expectation values. Because wire 0 is the least significant digit here
//! and qubit 0 the least significant bit there, the two vectors must agree
//! WITHOUT reindexing — which is what makes this a test of the mixed-radix
//! addressing (strides, gather/scatter, two- and three-wire kernels) at the
//! size where an error is cheapest to see.
//!
//! On its own this leg is the A10 trap the plan names: it passes under
//! secret delegation to the qubit engine. `anchors.rs` is what rules that
//! out. Together they say: the engine is exact where a reference exists,
//! and does something no reference could do.

use num_complex::Complex64;
use omega_backend_quditsv::QuditSvBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn op(gate: GateKind, wires: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: wires.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// Deterministic; no rand dependency.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn angle(&mut self) -> f64 {
        self.next_f64() * 2.0 * std::f64::consts::PI - std::f64::consts::PI
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next_f64() * n as f64) as u32 % n
    }
    fn distinct(&mut self, n: u32, k: usize) -> Vec<u32> {
        // A generator that can hang on a legal input is a bug in the
        // generator: fail loudly instead of rejection-sampling forever.
        assert!(
            k as u32 <= n,
            "distinct({n}, {k}): cannot draw {k} distinct wires from {n}"
        );
        let mut v = Vec::new();
        while v.len() < k {
            let q = self.below(n);
            if !v.contains(&q) {
                v.push(q);
            }
        }
        v
    }
}

fn random_circuit(n: u32, depth: usize, rng: &mut Lcg) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for _ in 0..depth {
        // `distinct(n, k)` cannot return for k > n: on one wire every draw
        // below 24 must stay single-wire, or the generator spins forever.
        let arms = if n >= 2 { 24 } else { 16 };
        let g = match rng.below(arms) {
            0 => op(GateKind::H, &[rng.below(n)], &[]),
            1 => op(GateKind::X, &[rng.below(n)], &[]),
            2 => op(GateKind::Y, &[rng.below(n)], &[]),
            3 => op(GateKind::Z, &[rng.below(n)], &[]),
            4 => op(GateKind::S, &[rng.below(n)], &[]),
            5 => op(GateKind::Sdg, &[rng.below(n)], &[]),
            6 => op(GateKind::T, &[rng.below(n)], &[]),
            7 => op(GateKind::Tdg, &[rng.below(n)], &[]),
            8 => op(GateKind::Sx, &[rng.below(n)], &[]),
            9 => op(GateKind::Sxdg, &[rng.below(n)], &[]),
            10 => op(GateKind::Rx, &[rng.below(n)], &[rng.angle()]),
            11 => op(GateKind::Ry, &[rng.below(n)], &[rng.angle()]),
            12 => op(GateKind::Rz, &[rng.below(n)], &[rng.angle()]),
            13 => op(GateKind::U1, &[rng.below(n)], &[rng.angle()]),
            14 => op(GateKind::U2, &[rng.below(n)], &[rng.angle(), rng.angle()]),
            15 => op(
                GateKind::U3,
                &[rng.below(n)],
                &[rng.angle(), rng.angle(), rng.angle()],
            ),
            16 => op(GateKind::CX, &rng.distinct(n, 2), &[]),
            17 => op(GateKind::CY, &rng.distinct(n, 2), &[]),
            18 => op(GateKind::CZ, &rng.distinct(n, 2), &[]),
            19 => op(GateKind::Swap, &rng.distinct(n, 2), &[]),
            20 => op(GateKind::CRz, &rng.distinct(n, 2), &[rng.angle()]),
            21 => op(
                GateKind::CU3,
                &rng.distinct(n, 2),
                &[rng.angle(), rng.angle(), rng.angle()],
            ),
            22 => op(GateKind::Rbs, &rng.distinct(n, 2), &[rng.angle()]),
            _ => {
                if n >= 3 && rng.below(2) == 0 {
                    op(GateKind::CCX, &rng.distinct(n, 3), &[])
                } else if n >= 3 {
                    op(GateKind::CSwap, &rng.distinct(n, 3), &[])
                } else {
                    op(GateKind::H, &[rng.below(n)], &[])
                }
            }
        };
        c.add_op(g);
    }
    c
}

fn statevector_of(b: &dyn Backend, c: &CircuitIR) -> Vec<Complex64> {
    let cfg = ExecConfig {
        shots: None,
        ..Default::default()
    };
    match b.execute(c, &ParameterBinding::new(), &cfg).unwrap() {
        ExecResult::Statevector(v) => v,
        other => panic!("expected a statevector, got {other:?}"),
    }
}

#[test]
fn random_qubit_circuits_agree_with_the_dense_qubit_engine_index_for_index() {
    let mut rng = Lcg(20260927); // seed = the date this test was written
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    for n in 1..=5u32 {
        for _ in 0..12 {
            let c = random_circuit(n, 6 + 4 * n as usize, &mut rng);
            let ours = statevector_of(&QuditSvBackend::new(), &c);
            let theirs = statevector_of(&StatevectorBackend::new(), &c);
            assert_eq!(ours.len(), theirs.len());
            for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
                let d = (a - b).norm();
                worst = worst.max(d);
                assert!(
                    d < 1e-12,
                    "n={n} index {i}: quditsv {a} vs statevector {b}\n{:?}",
                    c.ops
                );
            }
            compared += 1;
        }
    }
    assert_eq!(compared, 60);
    eprintln!("60 circuits, worst |Δ amplitude| = {worst:e}");
}

/// One wire is a legal circuit size and used to hang the generator
/// (`distinct(1, 2)` never returns). Pinned on its own so widening the
/// range above can never silently drop it again.
#[test]
fn a_single_wire_circuit_is_generated_and_agrees() {
    let mut rng = Lcg(1);
    for _ in 0..20 {
        let c = random_circuit(1, 10, &mut rng);
        assert!(c.ops.iter().all(|g| g.qubits.len() == 1), "{:?}", c.ops);
        let ours = statevector_of(&QuditSvBackend::new(), &c);
        let theirs = statevector_of(&StatevectorBackend::new(), &c);
        assert_eq!(ours.len(), 2);
        for (a, b) in ours.iter().zip(&theirs) {
            assert!((a - b).norm() < 1e-12, "{a} vs {b}\n{:?}", c.ops);
        }
    }
}

#[test]
fn expectation_values_agree_on_every_weight_le_2_pauli() {
    let mut rng = Lcg(7);
    let n = 3u32;
    let c = random_circuit(n, 20, &mut rng);
    let mut obs = Vec::new();
    let paulis = [PauliOp::X, PauliOp::Y, PauliOp::Z];
    for q in 0..n {
        for p in paulis {
            obs.push(Observable {
                terms: vec![(1.0, vec![(q, p)])],
            });
        }
    }
    for a in 0..n {
        for b in (a + 1)..n {
            for p in paulis {
                for r in paulis {
                    obs.push(Observable {
                        terms: vec![(0.7, vec![(a, p), (b, r)])],
                    });
                }
            }
        }
    }
    assert_eq!(obs.len(), 9 + 27);
    let ours = QuditSvBackend::new()
        .expectation_multi(&c, &ParameterBinding::new(), &obs)
        .unwrap();
    let theirs = StatevectorBackend::new()
        .expectation_multi(&c, &ParameterBinding::new(), &obs)
        .unwrap();
    for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert!((a - b).abs() < 1e-12, "observable {i}: {a} vs {b}");
    }
}

#[test]
fn sampling_on_qubits_reproduces_the_dense_distribution() {
    // Same seed does not mean the same shots (different sampler); the
    // distribution does. 20k shots on a 3-qubit Bell-ish circuit: every
    // outcome's frequency within 5σ of the exact probability.
    let c = {
        let mut c = CircuitIR::new(3, CircuitType::GateBased);
        c.add_op(op(GateKind::H, &[0], &[]));
        c.add_op(op(GateKind::CX, &[0, 1], &[]));
        c.add_op(op(GateKind::Ry, &[2], &[0.9]));
        c.add_op(op(GateKind::Measure, &[0], &[]));
        c
    };
    let exact: Vec<f64> = statevector_of(&StatevectorBackend::new(), &c)
        .iter()
        .map(|a| a.norm_sqr())
        .collect();
    let shots = 20_000u32;
    let cfg = ExecConfig {
        shots: Some(shots),
        seed: Some(11),
        ..Default::default()
    };
    let ExecResult::Counts(counts) = QuditSvBackend::new()
        .execute(&c, &ParameterBinding::new(), &cfg)
        .unwrap()
    else {
        panic!("expected counts");
    };
    let total: u32 = counts.values().sum();
    assert_eq!(total, shots);
    for (i, p) in exact.iter().enumerate() {
        let got = counts
            .iter()
            .find(|(k, _)| k.as_u64() == Some(i as u64))
            .map(|(_, v)| *v)
            .unwrap_or(0) as f64;
        let mean = p * shots as f64;
        let sigma = (p * (1.0 - p) * shots as f64).sqrt().max(1.0);
        assert!(
            (got - mean).abs() <= 5.0 * sigma,
            "outcome {i}: got {got}, expected {mean} ± {sigma}"
        );
    }
}
