// SPDX-License-Identifier: Apache-2.0
//! PLAN-FERMIONIC F3 — direct Majorana seeding against the via-JW door.
//!
//! (i)   The two doors reach the *same monomial seeds* — keys and coefficients
//!       bit-identical — across the full product sweep (every ladder product
//!       up to length three on three modes, symmetrised, with a complex
//!       coefficient so the phase bookkeeping is exercised), and the two
//!       runs agree on the certificate: counts exactly, sums to 1e-12.
//!       Why 1e-12 and not bit-identity on the sums: `MajoranaSum` is a
//!       `HashMap` with per-instance random hashing, so even the Pauli door
//!       is not bit-reproducible against itself once terms merge in a
//!       different order. The seed comparison is the bit-identical leg.
//! (ii)  The statevector oracle as tie-breaker, so (i) cannot pass by both
//!       doors sharing a mistake.
//! (iii) The non-triviality pin: on a non-adjacent hopping term the direct
//!       path builds strictly fewer intermediate objects than the Pauli
//!       path. A "direct" seed that secretly routed through JW would count
//!       equal, and fail.

use num_complex::Complex64;
use omega_backend_majoranaprop::engine::{seed_from_fermionic, seed_from_pauli};
use omega_backend_majoranaprop::majorana::MajoranaKey;
use omega_backend_majoranaprop::MajoranaPropBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::fermion::{FermionicOp, Ladder};
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

/// A circuit that moves every Majorana: single-qubit dressing (so no axis
/// carries zero amplitude), fermionic Givens rotations on each adjacent pair
/// (length-preserving), and one `CRz` (length-growing), so the sum both
/// rotates and branches.
fn workload(n: u32, seed: u64) -> CircuitIR {
    let mut rng = Lcg(seed);
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for q in 0..n {
        c.add_op(g(GateKind::Ry, &[q], &[rng.next()]));
        c.add_op(g(GateKind::Rz, &[q], &[rng.next()]));
    }
    for q in 0..n - 1 {
        c.add_op(g(GateKind::Rbs, &[q, q + 1], &[rng.next()]));
    }
    c.add_op(g(GateKind::CRz, &[0, n - 1], &[rng.next()]));
    for q in 0..n {
        c.add_op(g(GateKind::Rx, &[q], &[rng.next()]));
    }
    c
}

fn all_ladders(n: u32) -> Vec<Ladder> {
    (0..n)
        .flat_map(|m| [Ladder::raise(m), Ladder::lower(m)])
        .collect()
}

/// All products of exactly `len` ladders drawn (with repetition) from `n` modes.
fn all_products(n: u32, len: usize) -> Vec<Vec<Ladder>> {
    let ls = all_ladders(n);
    let mut out = vec![vec![]];
    for _ in 0..len {
        out = out
            .iter()
            .flat_map(|p| {
                ls.iter().map(move |l| {
                    let mut q = p.clone();
                    q.push(*l);
                    q
                })
            })
            .collect();
    }
    out
}

/// `c · prod + h.c.` — Hermitian by construction, with a complex `c` so the
/// direct seed's `i^e` bookkeeping is on the line, not just the signs.
fn symmetrised(prod: &[Ladder]) -> FermionicOp {
    let t = FermionicOp::term(Complex64::new(0.7, -0.3), prod.to_vec());
    let td = t.dagger();
    t + td
}

fn sorted_terms(sum: &omega_backend_majoranaprop::engine::MajoranaSum) -> Vec<(Vec<usize>, f64)> {
    let mut v: Vec<(Vec<usize>, f64)> = sum.terms.iter().map(|(k, c)| (k.indices(), *c)).collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

// ---------------------------------------------------------------------------
// (i) same seeds, same certificate
// ---------------------------------------------------------------------------

#[test]
fn direct_seed_matches_the_pauli_seed_bit_for_bit_across_the_product_sweep() {
    let n = 3u32;
    let mut checked = 0usize;
    let mut nonempty = 0usize;
    for len in 1..=3 {
        for prod in all_products(n, len) {
            let op = symmetrised(&prod);
            let via_jw = op
                .jordan_wigner()
                .expect("symmetrised operator is Hermitian");
            let pauli = seed_from_pauli(&via_jw, n as usize).unwrap();
            let (direct, _) = seed_from_fermionic(&op, n as usize).unwrap();
            let a = sorted_terms(&pauli);
            let b = sorted_terms(&direct);
            assert_eq!(a.len(), b.len(), "{op}: term count differs");
            for ((ka, ca), (kb, cb)) in a.iter().zip(b.iter()) {
                assert_eq!(ka, kb, "{op}: monomial sets differ");
                assert_eq!(
                    ca.to_bits(),
                    cb.to_bits(),
                    "{op}: coefficient on γ{ka:?} differs: pauli {ca:e}, direct {cb:e}"
                );
            }
            checked += 1;
            nonempty += usize::from(!a.is_empty());
        }
    }
    assert_eq!(checked, 6 + 36 + 216);
    // `a_p a_p` and its like are zero; most of the sweep is not.
    assert!(nonempty > checked / 2, "{nonempty} non-empty of {checked}");
}

#[test]
fn both_doors_report_the_same_certificate_on_the_sweep() {
    let n = 3u32;
    let circuit = workload(n, 7);
    let params = ParameterBinding::new();
    let exact = MajoranaPropBackend::new();
    // A cut that discards something on the longer products, so `dropped_mass`
    // is on the line, not just zero on both sides.
    let cut = MajoranaPropBackend::with_truncation(2e-2, Some(4));
    for backend in [&exact, &cut] {
        let mut dropped_seen = false;
        for len in 1..=3 {
            for prod in all_products(n, len) {
                let op = symmetrised(&prod);
                let via_jw = op.jordan_wigner().unwrap();
                let a = backend.expectation_with_certificate(&circuit, &params, &via_jw);
                let b = backend.expectation_fermionic_with_certificate(&circuit, &params, &op);
                match (a, b) {
                    (Ok((va, ca)), Ok((vb, cb))) => {
                        assert!((va - vb).abs() < 1e-12, "{op}: value {va} vs {vb}");
                        assert_eq!(ca.final_terms, cb.final_terms, "{op}: final_terms");
                        assert_eq!(ca.peak_terms, cb.peak_terms, "{op}: peak_terms");
                        assert_eq!(ca.branching_gates, cb.branching_gates, "{op}");
                        assert_eq!(
                            ca.observable_range.to_bits(),
                            cb.observable_range.to_bits(),
                            "{op}: observable_range {} vs {}",
                            ca.observable_range,
                            cb.observable_range
                        );
                        assert!(
                            (ca.dropped_mass - cb.dropped_mass).abs() < 1e-12,
                            "{op}: dropped_mass {} vs {}",
                            ca.dropped_mass,
                            cb.dropped_mass
                        );
                        dropped_seen |= ca.dropped_mass > 0.0;
                    }
                    // Both doors must refuse together (the vacuity gate).
                    (Err(ea), Err(eb)) => {
                        assert_eq!(ea.to_string(), eb.to_string(), "{op}: refusals differ");
                    }
                    (a, b) => panic!("{op}: doors disagree on refusal: {a:?} vs {b:?}"),
                }
            }
        }
        if backend.coeff_min() > 0.0 {
            assert!(
                dropped_seen,
                "the cut never discarded anything — vacuous comparison"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (ii) the statevector as tie-breaker
// ---------------------------------------------------------------------------

#[test]
fn direct_seed_agrees_with_the_statevector_oracle() {
    let n = 4u32;
    let circuit = workload(n, 11);
    let params = ParameterBinding::new();
    let ops: Vec<FermionicOp> = vec![
        FermionicOp::parse("[0^ 2] + [2^ 0]").unwrap(),
        FermionicOp::parse("[1^ 1]").unwrap(),
        FermionicOp::parse("0.5 [0^ 0 3^ 3]").unwrap(),
        FermionicOp::parse("[0^ 1^ 3 2] + [2^ 3^ 1 0]").unwrap(),
        FermionicOp::parse("(0.3+0.4j) [3^ 1] + (0.3-0.4j) [1^ 3] - 0.25 [2^ 2]").unwrap(),
    ];
    for op in &ops {
        let via_jw = op.jordan_wigner().unwrap();
        let oracle = StatevectorBackend::new()
            .expectation(&circuit, &params, &via_jw)
            .unwrap();
        let (direct, cert) = MajoranaPropBackend::new()
            .expectation_fermionic_with_certificate(&circuit, &params, op)
            .unwrap();
        assert_eq!(cert.dropped_mass, 0.0);
        assert!(
            (direct - oracle).abs() < 1e-9,
            "{op}: direct {direct} vs statevector {oracle}"
        );
    }
}

// ---------------------------------------------------------------------------
// (iii) the direct path really is direct
// ---------------------------------------------------------------------------

#[test]
fn direct_seed_builds_strictly_fewer_intermediates_than_the_pauli_detour() {
    let n = 3usize;
    let op = FermionicOp::parse("[0^ 2] + [2^ 0]").unwrap();

    let (direct_sum, direct_count) = seed_from_fermionic(&op, n).unwrap();

    // The Pauli detour, counted the same way: partial products in the
    // ladder → Pauli expansion, plus one monomial product per non-identity
    // site to re-condense each surviving string.
    let (strings, expand_count) = op.jordan_wigner_terms_counted();
    let mut condense_count = 0usize;
    let mut pauli_sum_terms = 0usize;
    for (_, s) in &strings {
        let mut x = vec![false; n];
        let mut z = vec![false; n];
        for (q, p) in s {
            match p {
                PauliOp::I => {}
                PauliOp::X => x[*q as usize] = true,
                PauliOp::Z => z[*q as usize] = true,
                PauliOp::Y => {
                    x[*q as usize] = true;
                    z[*q as usize] = true;
                }
            }
        }
        let (_, _, steps) = MajoranaKey::from_pauli_counted(&x, &z);
        condense_count += steps;
        pauli_sum_terms += 1;
    }
    let pauli_count = expand_count + condense_count;

    // Pinned arithmetic: two 2-ladder terms → 2·(2+4) partial products on
    // either side; the detour then re-condenses ½(X Z X + Y Z Y) — two
    // strings of three sites — for six more.
    assert_eq!(direct_count, 12, "direct partial products");
    assert_eq!(expand_count, 12, "JW partial products");
    assert_eq!(pauli_sum_terms, 2);
    assert_eq!(condense_count, 6, "from_pauli site products");
    assert!(
        direct_count < pauli_count,
        "direct {direct_count} is not below pauli {pauli_count}: the direct \
         path did not skip the re-condensation"
    );
    assert_eq!(direct_sum.terms.len(), 2);
    assert!((direct_sum.l1_norm() - 1.0).abs() < 1e-15);
}

// ---------------------------------------------------------------------------
// refusals, in this basis
// ---------------------------------------------------------------------------

#[test]
fn non_hermitian_operator_is_refused_naming_the_monomial() {
    let op = FermionicOp::parse("[0^ 1]").unwrap();
    let err = seed_from_fermionic(&op, 2).unwrap_err().to_string();
    assert!(err.contains("not Hermitian"), "{err}");
    assert!(err.contains("op.dagger()"), "{err}");
}

#[test]
fn mode_outside_the_circuit_is_refused_naming_the_mode() {
    let op = FermionicOp::parse("[3^ 3]").unwrap();
    let err = seed_from_fermionic(&op, 3).unwrap_err().to_string();
    assert!(err.contains("mode 3"), "{err}");
    assert!(err.contains("3 qubits"), "{err}");
}

#[test]
fn dephasing_drops_exactly_the_x_or_y_monomials() {
    // On qubit 1: a†₁a₁ = ½(1 − Z₁) has no X/Y on qubit 1 → kept whole;
    // a†₀a₁ + h.c. is X/Y on both 0 and 1 → dropped whole.
    let keep = FermionicOp::parse("[1^ 1]").unwrap();
    let (mut s, _) = seed_from_fermionic(&keep, 2).unwrap();
    let before = s.terms.len();
    s.dephase(&[1]);
    assert_eq!(s.terms.len(), before);

    let drop = FermionicOp::parse("[0^ 1] + [1^ 0]").unwrap();
    let (mut s, _) = seed_from_fermionic(&drop, 2).unwrap();
    assert_eq!(s.terms.len(), 2);
    s.dephase(&[1]);
    assert!(s.terms.is_empty());

    // Cross-check against the Pauli door's `Observable::dephase` on a mixed
    // operator: same surviving monomials.
    let mixed = FermionicOp::parse("[0^ 1] + [1^ 0] + 0.5 [1^ 1] - 0.25 [0^ 0 1^ 1]").unwrap();
    let via_jw: Observable = mixed.jordan_wigner().unwrap().dephase(&[1]);
    let pauli = seed_from_pauli(&via_jw, 2).unwrap();
    let (mut direct, _) = seed_from_fermionic(&mixed, 2).unwrap();
    direct.dephase(&[1]);
    assert_eq!(sorted_terms(&pauli), sorted_terms(&direct));
}
