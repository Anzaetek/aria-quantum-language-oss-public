// SPDX-License-Identifier: Apache-2.0
//! §3c.4 **leg 1** — the dense Fock-space oracle for the fermionic layer.
//!
//! Every convention in `omega_core::fermion` is checked here against a
//! `2^N × 2^N` matrix built from nothing but Kronecker products of `2 × 2`
//! blocks. Nothing in this file calls the Jordan–Wigner code to build its
//! reference; the ladder matrices are `Z ⊗ … ⊗ Z ⊗ σ ⊗ I ⊗ …` written out,
//! and the first test checks that *they* satisfy the anticommutation
//! relations before anything is compared to them.
//!
//! Qubit `q` is bit `q` of the basis index (`(i >> q) & 1`), the convention
//! the statevector backend uses, so a passing comparison here means the same
//! matrix the backend applies.
use ndarray::Array2;
use num_complex::Complex64;
use omega_core::circuit::{GateKind, ParamExpr};
use omega_core::executor::PauliOp;
use omega_core::fermion::{givens, FermionicOp, Ladder};

type M = Array2<Complex64>;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn eye(n: usize) -> M {
    Array2::eye(n)
}

fn kron(a: &M, b: &M) -> M {
    let (ar, ac) = a.dim();
    let (br, bc) = b.dim();
    let mut out = Array2::zeros((ar * br, ac * bc));
    for i in 0..ar {
        for j in 0..ac {
            for k in 0..br {
                for l in 0..bc {
                    out[[i * br + k, j * bc + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }
    out
}

fn single(op: PauliOp) -> M {
    let (z, o, i) = (c(0.0, 0.0), c(1.0, 0.0), c(0.0, 1.0));
    let rows = match op {
        PauliOp::I => [[o, z], [z, o]],
        PauliOp::X => [[z, o], [o, z]],
        PauliOp::Y => [[z, -i], [i, z]],
        PauliOp::Z => [[o, z], [z, -o]],
    };
    Array2::from_shape_fn((2, 2), |(r, col)| rows[r][col])
}

/// `|0⟩⟨1|` — annihilation on one mode.
fn sigma_minus() -> M {
    Array2::from_shape_fn((2, 2), |(r, col)| {
        if r == 0 && col == 1 {
            c(1.0, 0.0)
        } else {
            c(0.0, 0.0)
        }
    })
}

/// Embed per-qubit `2 × 2` blocks into `n` qubits, identity elsewhere, with
/// qubit 0 as the least significant index bit.
fn embed(n: u32, blocks: &[(u32, M)]) -> M {
    let mut acc = eye(1);
    for q in (0..n).rev() {
        let f = blocks
            .iter()
            .find(|(k, _)| *k == q)
            .map(|(_, m)| m.clone())
            .unwrap_or_else(|| eye(2));
        acc = kron(&acc, &f);
    }
    acc
}

fn string_mat(n: u32, s: &[(u32, PauliOp)]) -> M {
    let blocks: Vec<(u32, M)> = s.iter().map(|&(q, p)| (q, single(p))).collect();
    embed(n, &blocks)
}

fn pauli_sum_mat(n: u32, terms: &[(Complex64, Vec<(u32, PauliOp)>)]) -> M {
    let dim = 1usize << n;
    let mut acc: M = Array2::zeros((dim, dim));
    for (coeff, s) in terms {
        acc = acc + string_mat(n, s).mapv(|v| v * coeff);
    }
    acc
}

/// The oracle: `a_j = Z_0 ⋯ Z_{j−1} · |0⟩⟨1|_j`, `a†_j` its adjoint.
fn ladder_mat(n: u32, l: Ladder) -> M {
    let mut blocks: Vec<(u32, M)> = (0..l.mode).map(|k| (k, single(PauliOp::Z))).collect();
    let sm = sigma_minus();
    blocks.push((
        l.mode,
        if l.dagger {
            sm.t().mapv(|v| v.conj())
        } else {
            sm
        },
    ));
    embed(n, &blocks)
}

fn fermi_mat(n: u32, op: &FermionicOp) -> M {
    let dim = 1usize << n;
    let mut acc: M = Array2::zeros((dim, dim));
    for (coeff, prod) in &op.terms {
        let mut m = eye(dim);
        for l in prod {
            m = m.dot(&ladder_mat(n, *l));
        }
        acc = acc + m.mapv(|v| v * coeff);
    }
    acc
}

fn max_abs_diff(a: &M, b: &M) -> f64 {
    (a - b).iter().map(|v| v.norm()).fold(0.0, f64::max)
}

fn assert_close(a: &M, b: &M, what: &str) {
    let d = max_abs_diff(a, b);
    assert!(d < 1e-10, "{what}: max |Δ| = {d:.3e}");
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

// ---------------------------------------------------------------------------

#[test]
fn oracle_satisfies_the_canonical_anticommutation_relations() {
    // The reference is only worth anything if it is itself a fermion algebra.
    let n = 3;
    let dim = 1usize << n;
    for p in 0..n {
        for q in 0..n {
            let a_p = ladder_mat(n, Ladder::lower(p));
            let a_q = ladder_mat(n, Ladder::lower(q));
            let ad_q = ladder_mat(n, Ladder::raise(q));
            let anti = a_p.dot(&ad_q) + ad_q.dot(&a_p);
            let expect = if p == q {
                eye(dim)
            } else {
                Array2::zeros((dim, dim))
            };
            assert_close(&anti, &expect, &format!("{{a_{p}, a†_{q}}}"));
            let anti2 = a_p.dot(&a_q) + a_q.dot(&a_p);
            assert_close(
                &anti2,
                &Array2::zeros((dim, dim)),
                &format!("{{a_{p}, a_{q}}}"),
            );
        }
    }
}

#[test]
fn jordan_wigner_matches_the_oracle_for_every_product_up_to_length_three() {
    // 8 ladders at 4 modes: 8 + 64 + 512 products, each compared as a 16×16
    // matrix. Exhaustive rather than sampled because a Z-string or phase
    // error shows only on specific (non-adjacent, mixed-order) combinations.
    let n = 4;
    let mut checked = 0;
    for len in 1..=3 {
        for prod in all_products(n, len) {
            let op = FermionicOp::term(c(1.0, 0.0), prod.clone());
            let jw = pauli_sum_mat(n, &op.jordan_wigner_terms());
            let truth = fermi_mat(n, &op);
            assert_close(&jw, &truth, &format!("JW of {prod:?}"));
            checked += 1;
        }
    }
    assert_eq!(checked, 8 + 64 + 512);
}

/// PLAN-FERMIONIC F1 leg (i): the text spelling, through the parser
/// and `jordan_wigner_terms`, against the same dense oracle. The text is
/// built by hand from the enumerated product — not via `Display` — so the
/// parser shares nothing with the reference, and any sign, ordering or
/// Z-string error in it shows up as a matrix mismatch.
#[test]
fn parsed_text_matches_the_oracle_for_every_product_up_to_length_three() {
    let n = 4;
    let mut checked = 0;
    for len in 1..=3 {
        for prod in all_products(n, len) {
            let actions = prod
                .iter()
                .map(|l| format!("{}{}", l.mode, if l.dagger { "^" } else { "" }))
                .collect::<Vec<_>>()
                .join(" ");
            let text = format!("1.0 [{actions}]");
            let op = FermionicOp::parse(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
            let jw = pauli_sum_mat(n, &op.jordan_wigner_terms());
            let truth = fermi_mat(n, &FermionicOp::term(c(1.0, 0.0), prod.clone()));
            assert_close(&jw, &truth, &format!("JW of parsed {text:?}"));
            checked += 1;
        }
    }
    assert_eq!(checked, 8 + 64 + 512);
}

/// The deliberate wrong-string case from PLAN-FERMIONIC §1.2: for
/// `a†_1 a_3 + h.c.` the tempting hand Jordan–Wigner `0.5·X1X3 + 0.5·Y1Y3`
/// omits the `Z2` string. The parsed text must carry it, and the two must
/// differ by a measured amount on a reference state — here by exactly 2:
/// `⟨ψ|H|ψ⟩ = −1` with the string, `+1` without, for
/// `ψ = (|q1 q2⟩ + |q2 q3⟩)/√2`, where mode 2 is occupied in both branches.
#[test]
fn parsed_non_adjacent_hopping_carries_the_z_string_the_hand_spelling_forgets() {
    let n = 4;
    let parsed = FermionicOp::parse("1 [1^ 3] + 1 [3^ 1]").unwrap();
    let right = parsed.jordan_wigner().unwrap();
    let complexify = |o: &omega_core::executor::Observable| -> Vec<_> {
        o.terms
            .iter()
            .map(|(k, s)| (c(*k, 0.0), s.clone()))
            .collect()
    };
    // The parsed text is the oracle's operator…
    let right_m = pauli_sum_mat(n, &complexify(&right));
    assert_close(
        &right_m,
        &fermi_mat(n, &FermionicOp::hopping(1, 3, 1.0)),
        "parsed hopping vs oracle",
    );
    // …and the string a user would write by hand is not.
    let wrong = omega_core::executor::Observable::parse("0.5*X1X3+0.5*Y1Y3").unwrap();
    let wrong_m = pauli_sum_mat(n, &complexify(&wrong));
    let dim = 1usize << n;
    let mut psi = ndarray::Array1::<Complex64>::zeros(dim);
    psi[0b0110] = c(std::f64::consts::FRAC_1_SQRT_2, 0.0); // modes 1, 2 occupied
    psi[0b1100] = c(std::f64::consts::FRAC_1_SQRT_2, 0.0); // modes 2, 3 occupied
    let expect = |m: &M| psi.mapv(|v| v.conj()).dot(&m.dot(&psi)).re;
    let (ev_right, ev_wrong) = (expect(&right_m), expect(&wrong_m));
    assert!((ev_right + 1.0).abs() < 1e-10, "with Z string: {ev_right}");
    assert!(
        (ev_wrong - 1.0).abs() < 1e-10,
        "without Z string: {ev_wrong}"
    );
    assert!(
        (ev_right - ev_wrong).abs() > 1.0,
        "the Z string must be measurable: Δ = {}",
        ev_right - ev_wrong
    );
}

#[test]
fn number_operator_is_half_identity_minus_half_z() {
    let obs = FermionicOp::number(2).jordan_wigner().unwrap();
    let mut terms = obs.terms.clone();
    terms.sort_by_key(|a| a.1.len());
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0], (0.5, vec![]));
    assert_eq!(terms[1], (-0.5, vec![(2, PauliOp::Z)]));
}

#[test]
fn hermitian_operators_map_to_real_observables_and_non_hermitian_are_refused() {
    let n = 4;
    // A small Hubbard-like Hamiltonian: hopping over a non-adjacent pair
    // (Z string on mode 1 and 2), an on-site term and an interaction.
    let h = FermionicOp::hopping(0, 3, -0.7)
        + FermionicOp::number(1).scale(c(0.3, 0.0))
        + FermionicOp::interaction(1, 2, 1.1);
    let obs = h.jordan_wigner().expect("Hermitian operator must map");
    let as_complex: Vec<_> = obs
        .terms
        .iter()
        .map(|(coeff, s)| (c(*coeff, 0.0), s.clone()))
        .collect();
    assert_close(
        &pauli_sum_mat(n, &as_complex),
        &fermi_mat(n, &h),
        "Hubbard-like H",
    );
    // Every coefficient is real by construction; the matrix must be Hermitian.
    let m = fermi_mat(n, &h);
    assert_close(&m, &m.t().mapv(|v| v.conj()), "H = H†");

    // a†_0 alone is not Hermitian: refuse, and say why.
    let err = FermionicOp::raise(0)
        .jordan_wigner()
        .unwrap_err()
        .to_string();
    assert!(err.contains("not Hermitian"), "{err}");
    assert!(err.contains("op + op.dagger()"), "{err}");
    // …and the suggested fix works.
    let sym = FermionicOp::raise(0) + FermionicOp::raise(0).dagger();
    assert!(sym.jordan_wigner().is_ok());
}

#[test]
fn normal_ordering_preserves_the_operator_and_reaches_canonical_form() {
    // All products of length 3 and 4 at 3 modes: 216 + 1296. Each must keep
    // its 8×8 matrix, and land in normal order.
    let n = 3;
    let mut nonzero = 0;
    for len in 3..=4 {
        for prod in all_products(n, len) {
            let op = FermionicOp::term(c(1.0, 0.0), prod.clone());
            let no = op.normal_ordered();
            assert!(no.is_normal_ordered(), "{prod:?} → {:?}", no.terms);
            assert_close(
                &fermi_mat(n, &no),
                &fermi_mat(n, &op),
                &format!("NO of {prod:?}"),
            );
            if !no.terms.is_empty() {
                nonzero += 1;
            }
        }
    }
    assert!(nonzero > 0);

    // Canonical: two spellings of one operator compare equal only after it.
    let lhs = FermionicOp::lower(1).mul(&FermionicOp::raise(1)); // a_1 a†_1
    let rhs = FermionicOp::identity() + FermionicOp::number(1).scale(c(-1.0, 0.0)); // 1 − n_1
    assert_ne!(lhs, rhs);
    assert_eq!(lhs.normal_ordered(), rhs.normal_ordered());
}

#[test]
fn dagger_is_the_matrix_adjoint() {
    let n = 3;
    let op = FermionicOp::term(
        c(0.3, -0.8),
        vec![Ladder::raise(2), Ladder::lower(0), Ladder::raise(1)],
    );
    let expect = fermi_mat(n, &op).t().mapv(|v| v.conj());
    assert_close(&fermi_mat(n, &op.dagger()), &expect, "dagger");
}

#[test]
fn dropping_the_z_string_changes_the_answer() {
    // The §3b.2 shape: the wrong thing must be asserted as a DISAGREEMENT, so
    // a future "optimisation" that strips the string cannot pass by accident.
    let n = 3;
    let hop = FermionicOp::hopping(0, 2, 1.0);
    let with_z = hop.jordan_wigner().unwrap();
    assert!(
        with_z
            .terms
            .iter()
            .all(|(_, s)| s.iter().any(|(q, p)| *q == 1 && *p == PauliOp::Z)),
        "every term of a (0,2) hop must carry Z1: {:?}",
        with_z.terms
    );
    let stripped: Vec<_> = with_z
        .terms
        .iter()
        .map(|(coeff, s)| {
            (
                c(*coeff, 0.0),
                s.iter()
                    .copied()
                    .filter(|(q, _)| *q != 1)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let d = max_abs_diff(&pauli_sum_mat(n, &stripped), &fermi_mat(n, &hop));
    assert!(
        d > 0.5,
        "stripping Z1 must move the operator, moved {d:.3e}"
    );
}

#[test]
fn rbs_on_adjacent_modes_is_the_fermionic_givens_rotation_in_either_order() {
    // Rbs(θ) = exp(−iθ/2 (Y_p X_q − X_p Y_q)) per `GateKind::Rbs`'s doc.
    // Claim pinned by `givens`: it equals exp(θ (a†_p a_q − a†_q a_p)).
    // Both sides are exponentials of operators whose square is (minus) a
    // projector, so exp is the closed form I + sin·G + (1 − cos)·G².
    let n = 3;
    let theta: f64 = 0.37;
    let dim = 1usize << n;
    for (p, q) in [(0u32, 1u32), (1, 0), (1, 2), (2, 1)] {
        // Fermionic side, G real antisymmetric with G² = −P.
        let g_op = FermionicOp::raise(p).mul(&FermionicOp::lower(q))
            + FermionicOp::raise(q)
                .mul(&FermionicOp::lower(p))
                .scale(c(-1.0, 0.0));
        let g = fermi_mat(n, &g_op);
        let expected =
            eye(dim) + g.mapv(|v| v * theta.sin()) + g.dot(&g).mapv(|v| v * (1.0 - theta.cos()));

        // Gate side, K = (Y_p X_q − X_p Y_q)/2 Hermitian with K² = P.
        let k = (string_mat(n, &[(p, PauliOp::Y), (q, PauliOp::X)])
            - string_mat(n, &[(p, PauliOp::X), (q, PauliOp::Y)]))
        .mapv(|v| v * 0.5);
        let proj = k.dot(&k);
        let rbs = eye(dim)
            + proj.mapv(|v| v * (theta.cos() - 1.0))
            + k.mapv(|v| v * c(0.0, -theta.sin()));

        assert_close(&rbs, &expected, &format!("Rbs vs Givens on ({p},{q})"));

        let gate = givens(p, q, theta).unwrap();
        assert_eq!(gate.gate, GateKind::Rbs);
        assert_eq!(
            gate.qubits.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec![p, q]
        );
        assert!(matches!(gate.params[0], ParamExpr::Concrete(t) if t == theta));
    }
}

#[test]
fn givens_refuses_non_adjacent_modes_instead_of_emitting_a_wrong_gate() {
    let err = givens(0, 3, 0.1).unwrap_err().to_string();
    assert!(err.contains("Z string over modes 1..=2"), "{err}");
    assert!(givens(2, 3, 0.1).is_ok());
}
