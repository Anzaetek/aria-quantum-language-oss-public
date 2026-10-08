// SPDX-License-Identifier: Apache-2.0
//! The engine against two independent oracles: the statevector backend
//! (dense, exact) and exact pauliprop (same Heisenberg idea, different basis,
//! different code). Every gate in the table, a fermionic workload, the
//! headline length-preservation property, the dropped-mass bound, and the
//! refusals.

use omega_backend_majoranaprop::MajoranaPropBackend;
use omega_backend_pauliprop::PauliPropBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, Observable, PauliOp};
use omega_core::params::ParameterBinding;

fn g(kind: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

fn obs(paulis: &[(u32, PauliOp)]) -> Observable {
    Observable {
        terms: vec![(1.0, paulis.to_vec())],
    }
}

/// Deterministic angles; no rand dependency.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 * std::f64::consts::PI
            - std::f64::consts::PI
    }
}

fn sv(c: &CircuitIR, o: &Observable) -> f64 {
    StatevectorBackend::new()
        .expectation(c, &ParameterBinding::new(), o)
        .unwrap()
}
/// `None` where this build's pauliprop refuses the circuit. It lowers only
/// Clifford + Rx/Ry/Rz/U1/T/Tdg/CRz and REFUSES U2/U3/CU3/CRx/CRy/Rbs, so
/// it is a cross-check where it can run, never the binding oracle — that is
/// the statevector, which runs on every row.
fn pp(c: &CircuitIR, o: &Observable) -> Option<f64> {
    PauliPropBackend::new()
        .expectation(c, &ParameterBinding::new(), o)
        .ok()
}
fn mp(c: &CircuitIR, o: &Observable) -> f64 {
    MajoranaPropBackend::new()
        .expectation(c, &ParameterBinding::new(), o)
        .unwrap()
}

/// Random single-qubit dressing so no Pauli axis carries zero amplitude.
fn dress(c: &mut CircuitIR, n: u32, rng: &mut Lcg) {
    for q in 0..n {
        c.add_op(g(GateKind::Ry, &[q], &[rng.next()]));
        c.add_op(g(GateKind::Rz, &[q], &[rng.next()]));
    }
}

fn observables(n: u32) -> Vec<Observable> {
    use PauliOp::*;
    let mut v = vec![
        obs(&[(0, Z)]),
        obs(&[(1, X)]),
        obs(&[(n - 1, Y)]),
        obs(&[(0, Z), (1, Z)]),
        obs(&[(0, X), (1, Y)]),
        obs(&[(0, Y), (2, X)]),
        obs(&[(0, X), (1, Z), (2, Y)]),
    ];
    if n >= 4 {
        v.push(obs(&[(1, Y), (3, X)]));
    }
    // A multi-term observable with mixed signs.
    v.push(Observable {
        terms: vec![
            (0.7, vec![(0, Z)]),
            (-0.4, vec![(1, X), (2, X)]),
            (0.25, vec![]),
        ],
    });
    v
}

#[test]
fn every_gate_in_the_table_matches_statevector_and_pauliprop() {
    let mut rng = Lcg(7);
    let n = 4;
    let cases: Vec<(GateKind, Vec<u32>, usize)> = vec![
        (GateKind::X, vec![1], 0),
        (GateKind::Y, vec![2], 0),
        (GateKind::Z, vec![0], 0),
        (GateKind::S, vec![1], 0),
        (GateKind::Sdg, vec![1], 0),
        (GateKind::T, vec![2], 0),
        (GateKind::Tdg, vec![2], 0),
        (GateKind::Sx, vec![0], 0),
        (GateKind::Sxdg, vec![3], 0),
        (GateKind::H, vec![1], 0),
        (GateKind::Rx, vec![0], 1),
        (GateKind::Ry, vec![1], 1),
        (GateKind::Rz, vec![2], 1),
        (GateKind::U1, vec![2], 1),
        (GateKind::U2, vec![1], 2),
        (GateKind::U3, vec![0], 3),
        (GateKind::CZ, vec![0, 2], 0),
        (GateKind::CX, vec![1, 0], 0),
        (GateKind::CX, vec![0, 3], 0),
        (GateKind::CY, vec![2, 1], 0),
        (GateKind::Swap, vec![0, 2], 0),
        (GateKind::CRz, vec![1, 3], 1),
        (GateKind::Rbs, vec![1, 2], 1),
        (GateKind::Rbs, vec![0, 3], 1),
        (GateKind::CCX, vec![0, 1, 2], 0),
        (GateKind::CCX, vec![3, 1, 0], 0),
        (GateKind::CSwap, vec![1, 0, 3], 0),
    ];
    for (kind, qs, np) in cases {
        let mut c = CircuitIR::new(n, CircuitType::GateBased);
        dress(&mut c, n, &mut rng);
        let ps: Vec<f64> = (0..np).map(|_| rng.next()).collect();
        c.add_op(g(kind.clone(), &qs, &ps));
        dress(&mut c, n, &mut rng);
        for o in observables(n) {
            let want = sv(&c, &o);
            let got = mp(&c, &o);
            assert!(
                (got - want).abs() < 1e-9,
                "{kind:?} on {qs:?}: majoranaprop {got} vs statevector {want}"
            );
            if let Some(cross) = pp(&c, &o) {
                assert!(
                    (cross - want).abs() < 1e-9,
                    "pauliprop oracle itself off: {cross} vs {want}"
                );
            }
        }
    }
    // CU3 in its diagonal form; the general form is refused below.
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    dress(&mut c, n, &mut rng);
    c.add_op(g(GateKind::CU3, &[2, 0], &[0.0, 0.0, rng.next()]));
    dress(&mut c, n, &mut rng);
    for o in observables(n) {
        assert!((mp(&c, &o) - sv(&c, &o)).abs() < 1e-9, "CU3(0,0,λ)");
    }
}

/// A JW-encoded fermionic workload: particles prepared with X, adjacent
/// Givens layers, density–density interactions, on-site phases.
fn fermionic_circuit(n: u32, layers: usize, interact: bool, rng: &mut Lcg) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for q in (0..n).step_by(2) {
        c.add_op(g(GateKind::X, &[q], &[]));
    }
    for l in 0..layers {
        let start = (l % 2) as u32;
        let mut q = start;
        while q + 1 < n {
            c.add_op(g(GateKind::Rbs, &[q, q + 1], &[rng.next()]));
            if interact {
                c.add_op(g(GateKind::CU3, &[q, q + 1], &[0.0, 0.0, rng.next()]));
            }
            q += 2;
        }
        for q in 0..n {
            c.add_op(g(GateKind::Rz, &[q], &[rng.next() * 0.3]));
        }
    }
    c
}

#[test]
fn fermionic_workload_matches_both_oracles() {
    let mut rng = Lcg(11);
    let n = 6;
    for interact in [false, true] {
        let c = fermionic_circuit(n, 5, interact, &mut rng);
        for o in observables(n) {
            let want = sv(&c, &o);
            assert!((mp(&c, &o) - want).abs() < 1e-9, "interact={interact}");
            if let Some(p) = pp(&c, &o) {
                assert!((p - want).abs() < 1e-9);
            }
        }
    }
}

/// The reason this basis exists: under adjacent Givens rotations a length-2
/// observable stays length-2, so `max_length = 2` is EXACT with zero drop —
/// while the same circuit in the Pauli basis grows weight and pauliprop at
/// `max_weight = 2` must discard mass.
#[test]
fn givens_only_circuit_is_exact_at_the_observable_length() {
    let mut rng = Lcg(23);
    let n = 8;
    let c = fermionic_circuit(n, 6, false, &mut rng);
    let params = ParameterBinding::new();
    for o in [
        obs(&[(3, PauliOp::Z)]),
        obs(&[(2, PauliOp::Y), (3, PauliOp::X)]),
    ] {
        let want = sv(&c, &o);
        let (got, cert) = MajoranaPropBackend::with_truncation(0.0, Some(2))
            .expectation_with_certificate(&c, &params, &o)
            .unwrap();
        assert_eq!(
            cert.dropped_mass, 0.0,
            "length-2 cut must drop nothing on Givens-only"
        );
        assert!(cert.is_exact());
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        assert!(
            cert.peak_terms <= 2 * n as usize * (2 * n as usize),
            "quadratically many terms at most"
        );
        assert_eq!(
            cert.branching_gates, 0,
            "Givens generators are length-2: Thm 2(1) says they never branch"
        );

        // The contrast the plan states: the same circuit in the Pauli basis
        // grows weight — `Z` anticommutes with both `Rbs` generators and
        // spreads to `XX`/`YY`, which the next layer spreads further — so
        // weight-2 pauliprop MUST discard where length-2 majoranaprop is
        // exact. The dropped mass is a bound, so the error must sit inside it.
        // (Until 2026-09-18 pauliprop refused `Rbs` and this was a tripwire.)
        //
        // At the default ceiling pauliprop REFUSES this run: the mass it has
        // to discard at weight 2 exceeds the observable's own range, so the
        // bound excludes nothing. That refusal is the contrast in its
        // strongest form. The ceiling is lifted below only to read the
        // numbers the certificate would have carried.
        assert!(
            PauliPropBackend::with_truncation(0.0, Some(2))
                .expectation(&c, &params, &o)
                .is_err(),
            "weight-2 pauliprop should refuse as vacuous on this circuit"
        );
        let (pp_got, pp_cert) = PauliPropBackend::with_truncation(0.0, Some(2))
            .with_max_dropped_mass(Some(f64::INFINITY))
            .expectation_with_certificate(&c, &params, &o)
            .expect("pauliprop lowers Rbs");
        eprintln!(
            "weight-2 pauliprop: dropped_mass {:.4e}, |err| {:.3e}; length-2 majoranaprop exact",
            pp_cert.dropped_mass,
            (pp_got - want).abs()
        );
        assert!(
            pp_cert.dropped_mass > 1.0,
            "weight-2 pauliprop's bound should be vacuous here, got {}",
            pp_cert.dropped_mass
        );
        assert!(
            (pp_got - want).abs() <= pp_cert.dropped_mass + 1e-12,
            "pauliprop error {} exceeds its dropped_mass bound {}",
            (pp_got - want).abs(),
            pp_cert.dropped_mass
        );
    }
}

/// With interactions the length grows and the cut bites; the certificate
/// must then bound the error against the exact value, at every cut.
#[test]
fn dropped_mass_bounds_the_error_under_length_truncation() {
    let mut rng = Lcg(31);
    let n = 6;
    let c = fermionic_circuit(n, 4, true, &mut rng);
    let params = ParameterBinding::new();
    let mut saw_truncation = false;
    for o in observables(n) {
        let exact = sv(&c, &o);
        for max_len in [2usize, 4, 6] {
            let (got, cert) = MajoranaPropBackend::with_truncation(0.0, Some(max_len))
                .with_max_dropped_mass(Some(f64::INFINITY))
                .expectation_with_certificate(&c, &params, &o)
                .unwrap();
            assert!(
                (got - exact).abs() <= cert.dropped_mass + 1e-9,
                "max_length={max_len}: |{got} − {exact}| > dropped {}",
                cert.dropped_mass
            );
            if cert.dropped_mass > 0.0 {
                saw_truncation = true;
            }
        }
        // Coefficient floor too.
        let (got, cert) = MajoranaPropBackend::with_truncation(5e-2, None)
            .with_max_dropped_mass(Some(f64::INFINITY))
            .expectation_with_certificate(&c, &params, &o)
            .unwrap();
        assert!((got - exact).abs() <= cert.dropped_mass + 1e-9);
    }
    assert!(
        saw_truncation,
        "the test must actually exercise a lossy cut"
    );
}

#[test]
fn a_vacuous_bound_is_refused_not_returned() {
    let mut rng = Lcg(5);
    let c = fermionic_circuit(4, 2, true, &mut rng);
    let err = MajoranaPropBackend::with_truncation(1e9, None)
        .expectation(&c, &ParameterBinding::new(), &obs(&[(0, PauliOp::Z)]))
        .unwrap_err();
    assert!(err.to_string().contains("excludes nothing"), "{err}");
}

#[test]
fn refusals() {
    let params = ParameterBinding::new();
    let o = obs(&[(0, PauliOp::Z)]);
    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(g(GateKind::CU3, &[0, 1], &[0.3, 0.0, 0.1]));
    let e = MajoranaPropBackend::new()
        .expectation(&c, &params, &o)
        .unwrap_err();
    assert!(e.to_string().contains("diagonal form"), "{e}");

    let mut c = CircuitIR::new(2, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[0], &[]));
    c.add_op(g(GateKind::Reset, &[0], &[]));
    let e = MajoranaPropBackend::new()
        .expectation(&c, &params, &o)
        .unwrap_err();
    assert!(e.to_string().contains("reset"), "{e}");

    let c = CircuitIR::new(2, CircuitType::GateBased);
    let e = MajoranaPropBackend::new()
        .execute(&c, &params, &ExecConfig::default())
        .unwrap_err();
    assert!(e.to_string().contains("expectation-value backend"), "{e}");

    let e = MajoranaPropBackend::new()
        .expectation(&c, &params, &obs(&[(5, PauliOp::Z)]))
        .unwrap_err();
    assert!(e.to_string().contains("qubit 5"), "{e}");

    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    for _ in 0..4 {
        c.add_op(g(GateKind::CCX, &[0, 1, 2], &[]));
        c.add_op(g(GateKind::H, &[0], &[]));
        c.add_op(g(GateKind::H, &[1], &[]));
    }
    let e = MajoranaPropBackend::new()
        .with_max_terms(Some(2))
        .expectation(&c, &params, &obs(&[(2, PauliOp::Z)]))
        .unwrap_err();
    assert!(e.to_string().contains("above the ceiling of 2"), "{e}");
}

/// The a-priori field is present exactly when there is a length cut, and
/// interactions are what make gates count as branching.
#[test]
fn apriori_bound_rides_on_the_length_cut_and_counts_branching_gates() {
    let mut rng = Lcg(31);
    let n = 6;
    let c = fermionic_circuit(n, 3, true, &mut rng);
    let params = ParameterBinding::new();
    let o = obs(&[(2, PauliOp::Z)]);
    let (_, full) = MajoranaPropBackend::with_truncation(0.0, Some(2 * n as usize))
        .expectation_with_certificate(&c, &params, &o)
        .unwrap();
    assert!(full.is_exact(), "cut at 2N is no cut");
    assert!(full.branching_gates > 0, "interacting circuit must branch");
    let r = full.apriori_mse_ratio.expect("length cut ⇒ prior present");
    assert!(r > 0.0 && r.is_finite());
    assert_eq!(
        full.apriori_is_vacuous(),
        Some(r >= 1.0),
        "vacuity flag is the ratio ≥ 1"
    );

    let (_, none) = MajoranaPropBackend::new()
        .expectation_with_certificate(&c, &params, &o)
        .unwrap();
    assert_eq!(none.branching_gates, full.branching_gates);
    assert!(
        none.apriori_mse_ratio.is_none(),
        "no cut ⇒ no w₀ ⇒ no prior"
    );
    assert_eq!(none.apriori_is_vacuous(), None);
}
