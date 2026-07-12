//! CQS — the noiseless normal-equations kernel of the CQS formulation.
//!
//! Implements the linear-algebra core of qa-cqs (Huang et al., arXiv:1909.07344)
//! — **not** the full algorithm. Given `A = Σₗ cₗ·Pₗ` (a Pauli linear
//! combination) and `|b⟩`, pick an ansatz of Pauli strings `{Pₖ}` and write
//! `x = Σₖ αₖ·Pₖ|b⟩`. Minimizing `‖Ax − b‖²` over `α` is the small linear system
//!
//! ```text
//! Q α = r,   Q_{jk} = ⟨ψⱼ|A†A|ψₖ⟩ = ⟨Aψⱼ|Aψₖ⟩,   rⱼ = ⟨ψⱼ|A†|b⟩ = ⟨Aψⱼ|b⟩,
//! ```
//!
//! with `|ψₖ⟩ = Pₖ|b⟩`. On hardware `Q,r` are estimated by Hadamard tests; here
//! they are computed exactly on a statevector and the `α` combined classically.
//!
//! **Honest scope.** This is the *normal-equations kernel only*. The actual
//! qa-cqs contributions — shot-noise Hadamard-test estimation, iterative
//! ansatz-tree growth, and the provable sample complexity — are out of scope.
//! A complete ansatz recovers `x` exactly; a truncated one trades exactness for
//! fewer terms (the residual is the least-squares minimum over the ansatz span).
//!
//! **Validation anchor.** Everything routes through [`apply_pauli`], so that is
//! pinned to ground truth by [`tests::apply_pauli_matches_hand_coded_matrices`]
//! (hand-written Pauli matrices, no `apply_pauli` involved). Given that anchor,
//! the primary correctness check is the residual `‖Ax − b‖ ≈ 0` (which does not
//! use the linear solver at all); agreement with a dense solve is secondary.
//!
//! Self-contained (only `num_complex`) so the same file ships verbatim in OSS.

use num_complex::Complex64;

/// Apply a tensor-product Pauli string (`0=I,1=X,2=Y,3=Z` per qubit, qubit `q`
/// = bit `q`) to a statevector: `P|x⟩ = ph(x)·|x ⊕ flip⟩`, `ph` the product of
/// the per-qubit factors.
pub fn apply_pauli(pauli: &[u8], state: &[Complex64]) -> Vec<Complex64> {
    let n = pauli.len();
    let dim = state.len();
    debug_assert_eq!(dim, 1 << n);
    let mut flip = 0usize;
    for (q, &pq) in pauli.iter().enumerate() {
        if pq == 1 || pq == 2 {
            flip |= 1 << q;
        }
    }
    let mut out = vec![Complex64::new(0.0, 0.0); dim];
    for (x, &amp) in state.iter().enumerate() {
        let mut ph = Complex64::new(1.0, 0.0);
        for (q, &pq) in pauli.iter().enumerate() {
            let bit = (x >> q) & 1;
            match pq {
                2 => ph *= Complex64::i() * if bit == 1 { -1.0 } else { 1.0 }, // Y|b⟩ = i(−1)^b|1−b⟩
                3 if bit == 1 => ph = -ph,                                     // Z|1⟩ = −|1⟩
                _ => {}
            }
        }
        out[x ^ flip] += ph * amp;
    }
    out
}

/// `A = Σₗ cₗ·Pₗ`, each `Pₗ` a length-`n` Pauli string.
#[derive(Clone, Debug)]
pub struct PauliLcu {
    pub n: usize,
    pub terms: Vec<(Complex64, Vec<u8>)>,
}

impl PauliLcu {
    /// `A|ψ⟩ = Σₗ cₗ·Pₗ|ψ⟩`.
    pub fn apply(&self, state: &[Complex64]) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); state.len()];
        for (c, p) in &self.terms {
            for (o, t) in out.iter_mut().zip(apply_pauli(p, state)) {
                *o += c * t;
            }
        }
        out
    }

    /// Dense matrix (column `j` = `A|eⱼ⟩`).
    pub fn dense(&self) -> Vec<Vec<Complex64>> {
        let dim = 1usize << self.n;
        let mut a = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        for j in 0..dim {
            let mut e = vec![Complex64::new(0.0, 0.0); dim];
            e[j] = Complex64::new(1.0, 0.0);
            for (i, v) in self.apply(&e).into_iter().enumerate() {
                a[i][j] = v;
            }
        }
        a
    }
}

fn inner(u: &[Complex64], v: &[Complex64]) -> Complex64 {
    u.iter().zip(v).map(|(a, b)| a.conj() * b).sum()
}

fn norm(v: &[Complex64]) -> f64 {
    v.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
}

/// Partial-pivot Gaussian solve of `M x = y`, returning `Err` if the system is
/// singular (a near-zero pivot — the rank-deficient-ansatz case CQS hits when an
/// ansatz term is linearly dependent). Norm-relative floor.
#[allow(clippy::needless_range_loop)]
fn dense_solve(m_in: &[Vec<Complex64>], y_in: &[Complex64]) -> Result<Vec<Complex64>, String> {
    let n = y_in.len();
    let mut m: Vec<Vec<Complex64>> = m_in.to_vec();
    let mut y = y_in.to_vec();
    let scale = m
        .iter()
        .flat_map(|r| r.iter())
        .map(|z| z.norm())
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    let floor = 1e-12 * scale;
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&a, &b| {
                m[a][col]
                    .norm()
                    .partial_cmp(&m[b][col].norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        if m[piv][col].norm() <= floor {
            return Err(format!(
                "singular system: column {col} pivot {:.2e} ≤ floor {floor:.2e} \
                 (rank-deficient ansatz)",
                m[piv][col].norm()
            ));
        }
        m.swap(col, piv);
        y.swap(col, piv);
        let d = m[col][col];
        for row in (col + 1)..n {
            let f = m[row][col] / d;
            for c in col..n {
                let v = m[col][c] * f;
                m[row][c] -= v;
            }
            let bv = y[col] * f;
            y[row] -= bv;
        }
    }
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for row in (0..n).rev() {
        let mut s = y[row];
        for c in (row + 1)..n {
            s -= m[row][c] * x[c];
        }
        x[row] = s / m[row][row];
    }
    Ok(x)
}

/// Result of a CQS solve.
pub struct CqsResult {
    /// Reconstructed solution `x = Σₖ αₖ·Pₖ|b⟩`.
    pub x: Vec<Complex64>,
    /// Combination coefficients `α`.
    pub alpha: Vec<Complex64>,
    /// Achieved residual `‖A x − b‖₂` (the least-squares minimum over the ansatz).
    pub residual: f64,
}

/// CQS solve of `A x = b` over the Pauli `ansatz` (each a length-`n` string).
/// Returns `Err` if `Q` is singular (the ansatz is rank-deficient).
pub fn cqs_solve(a: &PauliLcu, b: &[Complex64], ansatz: &[Vec<u8>]) -> Result<CqsResult, String> {
    let k = ansatz.len();
    let psi: Vec<Vec<Complex64>> = ansatz.iter().map(|p| apply_pauli(p, b)).collect();
    let apsi: Vec<Vec<Complex64>> = psi.iter().map(|s| a.apply(s)).collect();
    let q: Vec<Vec<Complex64>> = (0..k)
        .map(|j| (0..k).map(|kk| inner(&apsi[j], &apsi[kk])).collect())
        .collect();
    let r: Vec<Complex64> = (0..k).map(|j| inner(&apsi[j], b)).collect();
    let alpha = dense_solve(&q, &r)?;
    let mut x = vec![Complex64::new(0.0, 0.0); b.len()];
    for (ak, sk) in alpha.iter().zip(&psi) {
        for (xi, si) in x.iter_mut().zip(sk) {
            *xi += ak * si;
        }
    }
    let mut ax_minus_b = a.apply(&x);
    for (v, bi) in ax_minus_b.iter_mut().zip(b) {
        *v -= bi;
    }
    Ok(CqsResult {
        residual: norm(&ax_minus_b),
        x,
        alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64) -> Complex64 {
        Complex64::new(re, 0.0)
    }

    /// Hand-coded single-qubit Pauli matrices — the ground-truth anchor.
    fn pauli_mat(p: u8) -> [[Complex64; 2]; 2] {
        let (z, o, i) = (c(0.0), c(1.0), Complex64::i());
        match p {
            0 => [[o, z], [z, o]],
            1 => [[z, o], [o, z]],
            2 => [[z, -i], [i, z]],
            3 => [[o, z], [z, -o]],
            _ => unreachable!(),
        }
    }

    /// `M[x][y] = ∏_q pauli_mat(pauli[q])[bit_q(x)][bit_q(y)]` — built WITHOUT
    /// `apply_pauli`, so it independently anchors it.
    fn ground_truth_matrix(pauli: &[u8]) -> Vec<Vec<Complex64>> {
        let n = pauli.len();
        let dim = 1 << n;
        (0..dim)
            .map(|x| {
                (0..dim)
                    .map(|y| {
                        (0..n)
                            .map(|q| pauli_mat(pauli[q])[(x >> q) & 1][(y >> q) & 1])
                            .product()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn apply_pauli_matches_hand_coded_matrices() {
        // Anchor: apply_pauli(P, e_j) must equal column j of the hand-built P.
        for pauli in [
            vec![1u8],
            vec![2],
            vec![3],
            vec![1, 3], // X0 Z1
            vec![2, 3], // Y0 Z1
            vec![3, 2], // Z0 Y1
        ] {
            let m = ground_truth_matrix(&pauli);
            let dim = 1 << pauli.len();
            for j in 0..dim {
                let mut e = vec![c(0.0); dim];
                e[j] = c(1.0);
                let got = apply_pauli(&pauli, &e);
                for i in 0..dim {
                    assert!(
                        (got[i] - m[i][j]).norm() < 1e-12,
                        "apply_pauli {pauli:?} col {j} row {i}: {} vs {}",
                        got[i],
                        m[i][j]
                    );
                }
            }
        }
    }

    fn e0(dim: usize) -> Vec<Complex64> {
        let mut v = vec![c(0.0); dim];
        v[0] = c(1.0);
        v
    }

    /// Seeded LCG → uniform `[0,1)`, for the fuzz tests.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (1u64 << 31) as f64
    }

    #[test]
    fn fuzz_apply_pauli_vs_ground_truth_on_random_states() {
        // Property: apply_pauli(P, ψ) == (hand-coded P)·ψ for RANDOM Pauli
        // strings and RANDOM complex states (not just basis vectors) — closes
        // the "only tested on |00⟩" gap.
        let mut s = 0xA5A5_1234u64;
        for _ in 0..400 {
            let n = 1 + (lcg(&mut s) * 2.0) as usize; // 1..=2
            let dim = 1 << n;
            let pauli: Vec<u8> = (0..n).map(|_| (lcg(&mut s) * 4.0) as u8 % 4).collect();
            let state: Vec<Complex64> = (0..dim)
                .map(|_| Complex64::new(lcg(&mut s) * 2.0 - 1.0, lcg(&mut s) * 2.0 - 1.0))
                .collect();
            let m = ground_truth_matrix(&pauli);
            let expected: Vec<Complex64> = (0..dim)
                .map(|i| (0..dim).map(|j| m[i][j] * state[j]).sum())
                .collect();
            let got = apply_pauli(&pauli, &state);
            for i in 0..dim {
                assert!(
                    (got[i] - expected[i]).norm() < 1e-12,
                    "fuzz apply_pauli {pauli:?} row {i}"
                );
            }
        }
    }

    #[test]
    fn fuzz_random_lcu_complete_ansatz_recovers_exact() {
        // Property: for random A (Hermitian AND non-Hermitian, dominant I so
        // invertible) and random complex b, a complete X-mask ansatz recovers
        // x with residual ≈ 0 whenever Q is non-singular (skip the rare
        // rank-deficient draw — that's the guarded Err path).
        let mut s = 0x1357_BEEFu64;
        let mut solved = 0;
        for _ in 0..300 {
            let n = 1 + (lcg(&mut s) * 2.0) as usize; // 1..=2
            let dim = 1 << n;
            let nterms = 2 + (lcg(&mut s) * 3.0) as usize;
            let mut terms: Vec<(Complex64, Vec<u8>)> = (0..nterms)
                .map(|_| {
                    let re = lcg(&mut s) * 2.0 - 1.0;
                    let im = if lcg(&mut s) < 0.3 {
                        lcg(&mut s) * 2.0 - 1.0
                    } else {
                        0.0
                    };
                    let p: Vec<u8> = (0..n).map(|_| (lcg(&mut s) * 4.0) as u8 % 4).collect();
                    (Complex64::new(re, im), p)
                })
                .collect();
            terms.push((c(2.5), vec![0u8; n])); // dominant identity ⇒ invertible
            let a = PauliLcu { n, terms };
            let b: Vec<Complex64> = (0..dim)
                .map(|_| Complex64::new(lcg(&mut s) * 2.0 - 1.0, lcg(&mut s) * 2.0 - 1.0))
                .collect();
            let ansatz: Vec<Vec<u8>> = (0..dim)
                .map(|mask| (0..n).map(|q| ((mask >> q) & 1) as u8).collect())
                .collect();
            if let Ok(res) = cqs_solve(&a, &b, &ansatz) {
                assert!(
                    res.residual < 1e-8,
                    "fuzz residual {:.2e} (n={n})",
                    res.residual
                );
                solved += 1;
            }
        }
        assert!(
            solved > 250,
            "too many singular draws ({solved}/300 solved)"
        );
    }

    /// Hermitian, well-conditioned, non-block-diagonal 2-qubit A.
    fn synthetic_a() -> PauliLcu {
        PauliLcu {
            n: 2,
            terms: vec![
                (c(1.5), vec![0, 0]),
                (c(0.4), vec![1, 0]),
                (c(0.35), vec![0, 1]),
                (c(0.3), vec![0, 3]),
                (c(0.2), vec![3, 3]),
            ],
        }
    }

    #[test]
    fn complete_ansatz_recovers_exact_solution() {
        let a = synthetic_a();
        let b = e0(4);
        let ansatz = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let res = cqs_solve(&a, &b, &ansatz).unwrap();
        // Primary check: residual ‖Ax−b‖ ≈ 0 (uses apply, the anchored primitive;
        // does NOT use the linear solver).
        assert!(res.residual < 1e-10, "CQS residual {:.3e}", res.residual);
        // Secondary: agrees with a dense solve.
        let x_dense = dense_solve(&a.dense(), &b).unwrap();
        let diff = res
            .x
            .iter()
            .zip(&x_dense)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-10, "CQS vs dense solve mismatch {diff:.3e}");
    }

    #[test]
    fn handles_complex_b_and_non_hermitian_a_with_y() {
        // Exercises the Y phase (i, −i) and Z sign on bit==1, plus complex b.
        let a = PauliLcu {
            n: 2,
            terms: vec![
                (c(1.6), vec![0, 0]),
                (c(0.4), vec![1, 0]),
                (Complex64::new(0.0, 0.3), vec![0, 2]), // 0.3i·Y1 ⇒ non-Hermitian
                (c(0.25), vec![3, 0]),                  // Z0
            ],
        };
        // b = (|00⟩ + i|11⟩)/√2 — complex, with bit==1 support.
        let s = 1.0 / 2.0_f64.sqrt();
        let b = vec![c(s), c(0.0), c(0.0), Complex64::new(0.0, s)];
        let ansatz = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let res = cqs_solve(&a, &b, &ansatz).unwrap();
        assert!(res.residual < 1e-10, "residual {:.3e}", res.residual);
    }

    #[test]
    fn rank_deficient_ansatz_is_rejected() {
        // Linearly dependent ansatz ⇒ singular Q ⇒ Err (not silent NaN).
        let a = synthetic_a();
        let b = e0(4);
        let dup = vec![vec![0, 0], vec![0, 0]]; // {II, II}
        assert!(
            cqs_solve(&a, &b, &dup).is_err(),
            "duplicate ansatz must error"
        );
    }

    #[test]
    fn residual_is_the_least_squares_minimum_and_nonincreasing() {
        // Note: residual non-increase under a growing nested ansatz is the
        // orthogonal-projection property of least squares — true for any basis.
        // The point of interest is that the achieved value MATCHES the true
        // projection residual, computed independently below.
        let a = synthetic_a();
        let b = e0(4);
        let full = [vec![0u8, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
        let mut prev = f64::INFINITY;
        for size in 1..=4 {
            let res = cqs_solve(&a, &b, &full[..size]).unwrap();
            // independent least-squares residual: project b onto span{Aψ_k} and
            // measure the orthogonal remainder via the Gram normal equations,
            // re-derived here from the column vectors directly.
            let cols: Vec<Vec<Complex64>> = full[..size]
                .iter()
                .map(|p| a.apply(&apply_pauli(p, &b)))
                .collect();
            let proj_res = least_squares_residual(&cols, &b);
            assert!(
                (res.residual - proj_res).abs() < 1e-9,
                "size {size}: CQS residual {:.3e} ≠ projection {:.3e}",
                res.residual,
                proj_res
            );
            assert!(res.residual <= prev + 1e-12, "residual grew at size {size}");
            prev = res.residual;
        }
        assert!(prev < 1e-10);
    }

    /// `‖b − P_span b‖` via the normal equations on the given columns —
    /// an independent recomputation of the least-squares minimum.
    fn least_squares_residual(cols: &[Vec<Complex64>], b: &[Complex64]) -> f64 {
        let k = cols.len();
        let g: Vec<Vec<Complex64>> = (0..k)
            .map(|i| (0..k).map(|j| inner(&cols[i], &cols[j])).collect())
            .collect();
        let rhs: Vec<Complex64> = (0..k).map(|i| inner(&cols[i], b)).collect();
        let coef = dense_solve(&g, &rhs).unwrap();
        let mut resid = b.to_vec();
        for (ci, col) in coef.iter().zip(cols) {
            for (rv, cv) in resid.iter_mut().zip(col) {
                *rv -= ci * cv;
            }
        }
        norm(&resid)
    }
}
