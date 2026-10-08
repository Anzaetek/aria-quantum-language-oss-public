// SPDX-License-Identifier: Apache-2.0
//! The sector backend against the dense statevector backend on random
//! number-conserving circuits, every sector, every readout path.
//!
//! The two share gate MATRICES by construction (`omega_backend_statevector::
//! gates`), so what this checks is everything else: the restricted basis,
//! the ranking, the pair traversal in `apply_2q`, the Pauli-string readout
//! with its `i^{|Y|}` and Z-parity, sampling, and the dense re-expansion.
use num_complex::Complex64;
use omega_backend_sector::SectorBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode, Observable, PauliOp};
use omega_core::fermion::{cphase, givens, FermionicOp, Ladder};
use omega_core::params::ParameterBinding;
use smallvec::smallvec;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn angle(&mut self) -> f64 {
        (self.next() % 10_000) as f64 / 10_000.0 * 2.0 * std::f64::consts::PI - std::f64::consts::PI
    }
}

fn op(gate: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// A random circuit every gate of which conserves number, on top of a random
/// occupation. Mixes the fermionic constructors with the raw gates the
/// matrix test admits, on adjacent and non-adjacent qubits alike.
fn random_conserving(rng: &mut Rng, n: u32, depth: usize) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for q in 0..n {
        if rng.below(2) == 1 {
            c.add_op(op(GateKind::X, &[q], &[]));
        }
    }
    for _ in 0..depth {
        let p = rng.below(n as u64) as u32;
        let mut q = rng.below(n as u64) as u32;
        while q == p && n > 1 {
            q = rng.below(n as u64) as u32;
        }
        // On one qubit only the single-qubit arms (6..=10) are well-formed.
        let pick = if n > 1 {
            rng.below(12)
        } else {
            6 + rng.below(5)
        };
        let g = match pick {
            0 if p.abs_diff(q) == 1 => givens(p, q, rng.angle()).unwrap(),
            0 | 1 => op(GateKind::Rbs, &[p, q], &[rng.angle()]),
            2 => cphase(p, q, rng.angle()),
            3 => op(GateKind::CRz, &[p, q], &[rng.angle()]),
            4 => op(GateKind::Swap, &[p, q], &[]),
            5 => op(GateKind::CZ, &[p, q], &[]),
            6 => op(GateKind::Rz, &[p], &[rng.angle()]),
            7 => op(GateKind::U1, &[p], &[rng.angle()]),
            8 => op(GateKind::U3, &[p], &[0.0, rng.angle(), rng.angle()]),
            9 => op(GateKind::S, &[p], &[]),
            10 => op(GateKind::Tdg, &[p], &[]),
            _ => op(GateKind::CU3, &[p, q], &[0.0, rng.angle(), rng.angle()]),
        };
        c.add_op(g);
    }
    c
}

fn random_observables(rng: &mut Rng, n: u32) -> Vec<Observable> {
    let jw = |o: FermionicOp| o.jordan_wigner().unwrap();
    let mut v = vec![];
    for _ in 0..6 {
        let p = rng.below(n as u64) as u32;
        let mut q = rng.below(n as u64) as u32;
        while q == p && n > 1 {
            q = rng.below(n as u64) as u32;
        }
        v.push(jw(FermionicOp::number(p)));
        v.push(jw(FermionicOp::hopping(p, q, 0.7)));
        let t = FermionicOp::term(Complex64::i(), vec![Ladder::raise(p), Ladder::lower(q)]);
        let d = t.dagger();
        v.push(jw(t + d));
        v.push(jw(FermionicOp::interaction(p, q, -0.4)));
    }
    // Raw Pauli strings too, including ones that leave the sector (whose
    // expectation must be exactly 0 in both engines) and Y-bearing ones.
    for _ in 0..8 {
        let mut terms = vec![];
        for _ in 0..3 {
            let mut s = vec![];
            for q in 0..n {
                let p = match rng.below(4) {
                    0 => continue,
                    1 => PauliOp::X,
                    2 => PauliOp::Y,
                    _ => PauliOp::Z,
                };
                s.push((q, p));
            }
            terms.push((rng.angle(), s));
        }
        v.push(Observable { terms });
    }
    v
}

#[test]
fn expectations_agree_with_dense_in_every_sector() {
    let mut rng = Rng(0x5EC7_0A11_D00D_F00D);
    let sv = StatevectorBackend::new();
    let sb = SectorBackend::new();
    let mut cases = 0;
    for n in 1..=7u32 {
        for _ in 0..12 {
            let c = random_conserving(&mut rng, n, 3 * n as usize + 2);
            let obs = random_observables(&mut rng, n);
            for o in &obs {
                o.validate_qubits(n).unwrap();
            }
            let a = sv
                .expectation_multi(&c, &ParameterBinding::new(), &obs)
                .unwrap();
            let b = sb
                .expectation_multi(&c, &ParameterBinding::new(), &obs)
                .unwrap();
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (x - y).abs() < 1e-12,
                    "n={n} observable {i}: dense {x}, sector {y}, |Δ|={:e}\n{:?}",
                    (x - y).abs(),
                    c.ops
                );
            }
            // The single-observable path is the same engine.
            let y0 = sb
                .expectation(&c, &ParameterBinding::new(), &obs[0])
                .unwrap();
            assert!((y0 - b[0]).abs() < 1e-15);
            cases += 1;
        }
    }
    assert_eq!(cases, 84);
}

#[test]
fn dense_readout_is_the_statevector_with_zeros_outside_the_sector() {
    let mut rng = Rng(42);
    let sv = StatevectorBackend::new();
    let sb = SectorBackend::new();
    let cfg = ExecConfig {
        shots: None,
        seed: None,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    for n in 1..=6u32 {
        let c = random_conserving(&mut rng, n, 2 * n as usize + 3);
        let k: u32 = c.ops.iter().filter(|o| o.gate == GateKind::X).count() as u32;
        let a = sv.execute(&c, &ParameterBinding::new(), &cfg).unwrap();
        let b = sb.execute(&c, &ParameterBinding::new(), &cfg).unwrap();
        let (a, b) = match (a, b) {
            (ExecResult::Statevector(a), ExecResult::Statevector(b)) => (a, b),
            _ => panic!("both must return a statevector"),
        };
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!((x - y).norm() < 1e-12, "n={n} amplitude {i}: {x} vs {y}");
            if (i as u64).count_ones() != k {
                assert_eq!(*y, Complex64::new(0.0, 0.0));
            }
        }
    }
}

#[test]
fn sampling_stays_in_the_sector_and_matches_dense_probabilities() {
    let mut rng = Rng(7);
    let sv = StatevectorBackend::new();
    let sb = SectorBackend::new();
    let n = 6u32;
    let c = random_conserving(&mut rng, n, 20);
    let k: u32 = c.ops.iter().filter(|o| o.gate == GateKind::X).count() as u32;
    let exact = match sv
        .execute(
            &c,
            &ParameterBinding::new(),
            &ExecConfig {
                shots: None,
                seed: None,
                mid_circuit_mode: MidCircuitMode::Skip,
            },
        )
        .unwrap()
    {
        ExecResult::Statevector(v) => v,
        _ => unreachable!(),
    };
    let shots = 200_000u32;
    let cfg = ExecConfig {
        shots: Some(shots),
        seed: Some(99),
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let r = sb.execute(&c, &ParameterBinding::new(), &cfg).unwrap();
    let counts = r.counts();
    let mut total = 0u32;
    for (outcome, count) in counts {
        let bits = outcome.as_u64().expect("6 bits fit");
        assert_eq!(bits.count_ones(), k, "sampled {bits:#b} outside the sector");
        let p = exact[bits as usize].norm_sqr();
        let f = *count as f64 / shots as f64;
        // 5σ on a binomial with 200k shots.
        let sigma = (p * (1.0 - p) / shots as f64).sqrt().max(1e-4);
        assert!(
            (f - p).abs() < 5.0 * sigma,
            "{bits:#b}: freq {f}, exact {p}"
        );
        total += count;
    }
    assert_eq!(total, shots);
    // Seeded → reproducible.
    let r2 = sb.execute(&c, &ParameterBinding::new(), &cfg).unwrap();
    assert_eq!(r.counts(), r2.counts());
}

#[test]
fn symbolic_parameters_resolve_through_the_binding() {
    let sb = SectorBackend::new();
    let sv = StatevectorBackend::new();
    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    c.symbols.insert(0, "theta".into());
    c.symbols.insert(1, "phi".into());
    c.add_op(op(GateKind::X, &[0], &[]));
    c.add_op(op(GateKind::X, &[2], &[]));
    c.add_op(GateOp {
        gate: GateKind::Rbs,
        qubits: smallvec![Qubit(0), Qubit(1)],
        params: smallvec![ParamExpr::Symbol(0)],
        classical_bit: None,
        condition: None,
    });
    c.add_op(omega_core::fermion::cphase_expr(
        1,
        2,
        ParamExpr::Mul(
            Box::new(ParamExpr::Symbol(1)),
            Box::new(ParamExpr::Concrete(2.0)),
        ),
    ));
    let mut b = ParameterBinding::new();
    b.bind(0, 0.37);
    b.bind(1, -1.1);
    let o = FermionicOp::hopping(1, 2, 1.0).jordan_wigner().unwrap();
    let x = sv.expectation(&c, &b, &o).unwrap();
    let y = sb.expectation(&c, &b, &o).unwrap();
    assert!((x - y).abs() < 1e-12, "{x} vs {y}");
    // Unbound → the binding's error, not a panic.
    assert!(sb.expectation(&c, &ParameterBinding::new(), &o).is_err());
}
