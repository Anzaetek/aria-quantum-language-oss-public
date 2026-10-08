// SPDX-License-Identifier: Apache-2.0
//! Exhaustive dense-matrix oracle for the Majorana monomial algebra at
//! n = 2 and n = 3 modes (16 and 64 monomials). Every claim in
//! `majorana.rs`'s module doc is checked here against matrices built only
//! from Pauli matrices and Kronecker products — nothing from the crate under
//! test is used to build the reference.

use ndarray::Array2;
use num_complex::Complex64;
use omega_backend_majoranaprop::MajoranaKey;

type M = Array2<Complex64>;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}
fn eye(d: usize) -> M {
    Array2::eye(d)
}
fn pauli(ch: u8) -> M {
    let z = c(0.0, 0.0);
    let o = c(1.0, 0.0);
    match ch {
        b'I' => eye(2),
        b'X' => Array2::from_shape_vec((2, 2), vec![z, o, o, z]).unwrap(),
        b'Y' => Array2::from_shape_vec((2, 2), vec![z, c(0.0, -1.0), c(0.0, 1.0), z]).unwrap(),
        b'Z' => Array2::from_shape_vec((2, 2), vec![o, z, z, -o]).unwrap(),
        _ => unreachable!(),
    }
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
/// Pauli string on n qubits, qubit 0 = most significant (matches the
/// statevector backend's |q0 q1 …⟩ ordering).
fn string_mat(s: &[u8]) -> M {
    s.iter().fold(eye(1), |acc, &ch| kron(&acc, &pauli(ch)))
}
/// γ_k dense: γ_{2j} = Z^{⊗j} X_j, γ_{2j+1} = Z^{⊗j} Y_j.
fn gamma(n: usize, k: usize) -> M {
    let j = k / 2;
    let mut s = vec![b'I'; n];
    for ch in s.iter_mut().take(j) {
        *ch = b'Z';
    }
    s[j] = if k.is_multiple_of(2) { b'X' } else { b'Y' };
    string_mat(&s)
}
fn ipow(e: u32) -> Complex64 {
    match e % 4 {
        0 => c(1.0, 0.0),
        1 => c(0.0, 1.0),
        2 => c(-1.0, 0.0),
        _ => c(0.0, -1.0),
    }
}
/// Dense M_b from the crate's stated definition (phase i^{r}, ascending product).
fn dense(n: usize, key: &MajoranaKey) -> M {
    let idx = key.indices();
    let m = idx.len() as u32;
    let r = (m * m.wrapping_sub(1) / 2) % 4;
    let mut out = eye(1 << n) * ipow(r);
    for k in idx {
        out = out.dot(&gamma(n, k));
    }
    out
}
fn close(a: &M, b: &M) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).norm() < 1e-12)
}
fn all_keys(n: usize) -> Vec<MajoranaKey> {
    (0..(1u64 << (2 * n)))
        .map(|bits| MajoranaKey::from_words(&[bits], n))
        .collect()
}

#[test]
fn every_monomial_is_hermitian_and_squares_to_one() {
    for n in [1usize, 2, 3] {
        for k in all_keys(n) {
            let m = dense(n, &k);
            let dag = m.t().map(|v| v.conj());
            assert!(close(&m, &dag), "n={n} {:?} not Hermitian", k.indices());
            assert!(
                close(&m.dot(&m), &eye(1 << n)),
                "n={n} {:?} M²≠1",
                k.indices()
            );
            assert_eq!(k.length() as usize, k.indices().len());
        }
    }
}

#[test]
fn product_phase_matches_dense_for_all_pairs() {
    for n in [2usize, 3] {
        let keys = all_keys(n);
        for a in &keys {
            for b in &keys {
                let (k, e) = a.mul(b);
                let lhs = dense(n, a).dot(&dense(n, b));
                let rhs = dense(n, &k) * ipow(e);
                assert!(
                    close(&lhs, &rhs),
                    "n={n} {:?}·{:?}: claimed i^{e}·{:?}",
                    a.indices(),
                    b.indices(),
                    k.indices()
                );
            }
        }
    }
}

#[test]
fn commutation_matches_dense_for_all_pairs() {
    for n in [2usize, 3] {
        let keys = all_keys(n);
        for a in &keys {
            for b in &keys {
                let ab = dense(n, a).dot(&dense(n, b));
                let ba = dense(n, b).dot(&dense(n, a));
                let dense_commutes = close(&ab, &ba);
                assert_eq!(
                    a.commutes(b),
                    dense_commutes,
                    "n={n} {:?} vs {:?}",
                    a.indices(),
                    b.indices()
                );
            }
        }
    }
}

#[test]
fn fock_readout_matches_dense_for_every_state_and_monomial() {
    for n in [1usize, 2, 3] {
        for k in all_keys(n) {
            let m = dense(n, &k);
            for occ in 0..(1u64 << n) {
                // Fock state |n_0 n_1 …⟩ with qubit 0 = MSB: index bit reversal.
                let mut idx = 0usize;
                for j in 0..n {
                    if (occ >> j) & 1 == 1 {
                        idx |= 1 << (n - 1 - j);
                    }
                }
                let want = m[[idx, idx]];
                assert!(want.im.abs() < 1e-12);
                let got = k.fock_expectation(&[occ]);
                assert!(
                    (got - want.re).abs() < 1e-12,
                    "n={n} occ={occ:b} {:?}: got {got} want {}",
                    k.indices(),
                    want.re
                );
            }
        }
    }
    let k = MajoranaKey::from_indices(3, &[0, 1, 4, 5]);
    assert_eq!(k.vacuum_expectation(), 1.0);
    assert_eq!(
        MajoranaKey::from_indices(3, &[0, 1]).vacuum_expectation(),
        -1.0
    );
    assert_eq!(
        MajoranaKey::from_indices(3, &[0, 2]).vacuum_expectation(),
        0.0
    );
}

/// U = exp(−iφ M_c/2) = cos(φ/2) − i sin(φ/2) M_c since M_c² = 1.
fn rot(n: usize, ckey: &MajoranaKey, phi: f64) -> M {
    let mc = dense(n, ckey);
    eye(1 << n) * c((phi / 2.0).cos(), 0.0) + mc * c(0.0, -(phi / 2.0).sin())
}

#[test]
fn rotation_rule_matches_dense_conjugation() {
    let angles = [0.3, -1.1, 2.7];
    for n in [2usize, 3] {
        let keys = all_keys(n);
        for cc in &keys {
            if cc.is_identity() {
                continue;
            }
            for b in &keys {
                for &phi in &angles {
                    let u = rot(n, cc, phi);
                    let udag = u.t().map(|v| v.conj());
                    let got = udag.dot(&dense(n, b)).dot(&u);
                    let want = if cc.commutes(b) {
                        dense(n, b)
                    } else {
                        let (k, s) = b.rotate_by(cc);
                        assert_eq!(
                            k.length() as i64,
                            cc.length() as i64 + b.length() as i64
                                - 2 * cc.indices().iter().filter(|i| b.bit(**i)).count() as i64
                        );
                        dense(n, b) * c(phi.cos(), 0.0) + dense(n, &k) * c(s * phi.sin(), 0.0)
                    };
                    assert!(
                        close(&got, &want),
                        "n={n} c={:?} b={:?} φ={phi}",
                        cc.indices(),
                        b.indices()
                    );
                }
            }
        }
    }
}

#[test]
fn pauli_strings_map_to_majorana_monomials_exactly() {
    for n in [1usize, 2, 3] {
        for code in 0..(1u64 << (2 * n)) {
            let mut x = vec![false; n];
            let mut z = vec![false; n];
            let mut s = vec![b'I'; n];
            for j in 0..n {
                x[j] = (code >> (2 * j)) & 1 == 1;
                z[j] = (code >> (2 * j + 1)) & 1 == 1;
                s[j] = match (x[j], z[j]) {
                    (false, false) => b'I',
                    (true, false) => b'X',
                    (true, true) => b'Y',
                    (false, true) => b'Z',
                };
            }
            let (k, e) = MajoranaKey::from_pauli(&x, &z);
            let want = string_mat(&s);
            let got = dense(n, &k) * ipow(e);
            assert!(
                close(&got, &want),
                "n={n} {}: claimed i^{e}·{:?}",
                String::from_utf8_lossy(&s),
                k.indices()
            );
        }
    }
}

/// The two gate generators the engine relies on, against the statevector
/// backend's actual matrices — the same gate table the differential harness
/// pins to the JW oracle. `Rbs(θ) = exp(−iθ/2 (Y⊗X − X⊗Y))`,
/// `CU3(0,0,λ) = diag(1,1,1,e^{iλ}) = exp(iλ n_p n_q)`.
#[test]
fn gate_generators_match_statevector_matrices() {
    use omega_backend_statevector::gates::{cu3, rbs};
    let n = 2;
    let as_mat = |g: [Complex64; 16]| Array2::from_shape_vec((4, 4), g.to_vec()).unwrap();
    let theta = 0.83;
    // Rbs on (0,1): rotations exp(−iθ/2 Y0X1) · exp(+iθ/2 X0Y1); the two
    // Pauli terms commute so the split exponential is exact.
    let (yx, e1) = MajoranaKey::from_pauli(&[true, true], &[true, false]);
    let (xy, e2) = MajoranaKey::from_pauli(&[true, true], &[false, true]);
    let s1 = ipow(e1).re;
    let s2 = ipow(e2).re;
    assert!(s1.abs() == 1.0 && s2.abs() == 1.0);
    let u = rot(n, &yx, theta * s1).dot(&rot(n, &xy, -theta * s2));
    assert!(close(&u, &as_mat(rbs(theta))), "Rbs generator mismatch");
    assert!(
        yx.length() == 2 && xy.length() == 2,
        "adjacent Givens must be length-2"
    );

    // CU3(0,0,λ): e^{iλ/4} · exp(−i(λ/2)/2 Z0) · exp(−i(λ/2)/2 Z1) · exp(+i(λ/2)/2 Z0Z1)
    let lam = -1.37;
    // Each Pauli is ±M for its monomial (single Z_j = −M_{2j,2j+1}); the
    // sign rides along into the rotation angle.
    let (z0, e0) = MajoranaKey::from_pauli(&[false, false], &[true, false]);
    let (z1, e1) = MajoranaKey::from_pauli(&[false, false], &[false, true]);
    let (zz, ezz) = MajoranaKey::from_pauli(&[false, false], &[true, true]);
    let (s0, s1, szz) = (ipow(e0).re, ipow(e1).re, ipow(ezz).re);
    let u = rot(n, &z0, lam / 2.0 * s0)
        .dot(&rot(n, &z1, lam / 2.0 * s1))
        .dot(&rot(n, &zz, -lam / 2.0 * szz))
        * Complex64::from_polar(1.0, lam / 4.0);
    assert!(
        close(&u, &as_mat(cu3(0.0, 0.0, lam))),
        "CU3 generator mismatch"
    );
}
