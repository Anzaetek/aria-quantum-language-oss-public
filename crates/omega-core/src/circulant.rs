//! Exact classical DFT solve of a banded-circulant system.
//!
//! This is the **diagonalization structure** the qa-cqs-circulant quantum
//! algorithm (Huang–Li–Koor–Rebentrost, *New J. Phys.* 2026) exploits — a
//! circulant `C = Σⱼ cⱼ Qᵖᵒʷ⁽ʲ⁾` (a linear combination of cyclic-shift
//! permutations) is diagonalized by the DFT, eigenvalues
//! `λ_k = Σⱼ cⱼ·ω^{−powⱼ·k}` (`ω = e^{2πi/N}`), eigenvectors the Fourier modes:
//!
//! ```text
//! C x = b   ⟹   x = IDFT( DFT(b) / λ ).
//! ```
//!
//! **This is NOT the CQS solver.** It reproduces *none* of the paper's
//! contribution — the classical-combination-of-quantum-states variational
//! ansatz, the truncation parameter `K`, or the `O(K·κ·log κ)` error bound.
//! It is the exact (here O(N²), classical) baseline those results approximate,
//! and the worked operator they study. On hardware the DFT is the QFT; here it
//! is a naive transform used only to produce the reference solution.
//!
//! The 1-D heat-transfer operator `C = (−2−ξ)I + Q + Q⁻¹` is the example:
//! `λ_k = (−2−ξ) + 2cos(2πk/N)`; **for even `N`** the condition number is
//! `κ = (4+ξ)/ξ`, so `ξ` tunes the conditioning (the knee the paper studies).
//!
//! **Validation.** The solve is checked against the *definitionally* assembled
//! dense matrix [`Circulant::dense`] two ways — an independent partial-pivot
//! Gaussian solve ([`solve_dense_gaussian`], a different algorithm that never
//! touches the eigenvalue formula) and the explicit residual `‖Cx − b‖`.
//! Separately, [`Circulant::eigenvalues`] (closed form) is cross-checked against
//! [`Circulant::eigenvalues_from_dense`] (DFT of the assembled first column),
//! so an error in either the formula or the assembly is caught. All synthetic.
//!
//! Self-contained (only `num_complex`) so the same file ships verbatim in the
//! OSS `omega-core`.

use num_complex::Complex64;
use std::f64::consts::PI;

/// Partial-pivot Gaussian elimination solve of a dense complex system `A x = b`
/// — the independent oracle the DFT diagonalization is validated against (a
/// different algorithm, sharing no code with the eigenvalue path).
#[allow(clippy::needless_range_loop)] // elimination indexes two distinct rows
pub fn solve_dense_gaussian(a_in: &[Vec<Complex64>], b_in: &[Complex64]) -> Vec<Complex64> {
    let n = b_in.len();
    let mut a: Vec<Vec<Complex64>> = a_in.to_vec();
    let mut b = b_in.to_vec();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&r1, &r2| {
                a[r1][col]
                    .norm()
                    .partial_cmp(&a[r2][col].norm())
                    .expect("finite")
            })
            .expect("non-empty");
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for row in (col + 1)..n {
            let f = a[row][col] / d;
            for c in col..n {
                let v = a[col][c] * f;
                a[row][c] -= v;
            }
            let bv = b[col] * f;
            b[row] -= bv;
        }
    }
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for c in (row + 1)..n {
            s -= a[row][c] * x[c];
        }
        x[row] = s / a[row][row];
    }
    x
}

/// A circulant matrix as a linear combination of cyclic-shift powers:
/// `C = Σ (coeff · Q^pow)`, where `Q` is the lower cyclic shift
/// (`Q[i][j] = 1` iff `i = (j+1) mod n`). `pow` may be negative (`Q⁻¹` is the
/// upper shift). This is the K-banded form the paper solves.
#[derive(Clone, Debug)]
pub struct Circulant {
    pub n: usize,
    /// `(pow, coeff)` terms; `pow` taken mod `n`.
    pub terms: Vec<(i64, f64)>,
}

impl Circulant {
    /// The 1-D heat-transfer circulant `C = (−2−ξ)I + Q + Q⁻¹`.
    pub fn heat_transfer(n: usize, xi: f64) -> Self {
        assert!(n >= 1, "n must be ≥ 1");
        Circulant {
            n,
            terms: vec![(0, -2.0 - xi), (1, 1.0), (-1, 1.0)],
        }
    }

    /// Analytic eigenvalues `λ_k = Σⱼ cⱼ · ω^{−powⱼ·k}`, `ω = e^{2πi/N}`.
    pub fn eigenvalues(&self) -> Vec<Complex64> {
        let n = self.n;
        (0..n)
            .map(|k| {
                self.terms
                    .iter()
                    .map(|&(pow, c)| {
                        let angle = -2.0 * PI * (pow as f64) * (k as f64) / (n as f64);
                        Complex64::from_polar(1.0, angle) * c
                    })
                    .sum()
            })
            .collect()
    }

    /// Eigenvalues derived from the **assembled** matrix (DFT of its first
    /// column), independent of the analytic [`Circulant::eigenvalues`] closed
    /// form — an orthogonal cross-check on both the formula and `dense()`.
    pub fn eigenvalues_from_dense(&self) -> Vec<Complex64> {
        let n = self.n;
        let a = self.dense();
        let col: Vec<Complex64> = (0..n).map(|i| a[i][0]).collect();
        (0..n)
            .map(|k| {
                (0..n)
                    .map(|m| {
                        col[m] * Complex64::from_polar(1.0, -2.0 * PI * (m * k) as f64 / n as f64)
                    })
                    .sum()
            })
            .collect()
    }

    /// Assemble the dense `n×n` matrix (the definitional spec the solve is
    /// validated against). Row-major `Vec<Vec<…>>`.
    #[allow(clippy::needless_range_loop)] // i is a shift of j, not a row iterator
    pub fn dense(&self) -> Vec<Vec<Complex64>> {
        let n = self.n;
        let mut a = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for &(pow, c) in &self.terms {
            // Q^pow: entry 1 at (i, j) with i = (j + pow) mod n.
            for j in 0..n {
                let i = ((j as i64 + pow).rem_euclid(n as i64)) as usize;
                a[i][j] += Complex64::new(c, 0.0);
            }
        }
        a
    }

    /// Spectral condition number `max|λ_k| / min|λ_k|`. Always exact (computed
    /// from the actual spectrum); the closed form `(4+ξ)/ξ` is the even-`N`
    /// special case for the heat-transfer operator only.
    pub fn condition_number(&self) -> f64 {
        let lam = self.eigenvalues();
        let mags: Vec<f64> = lam.iter().map(|z| z.norm()).collect();
        let max = mags.iter().cloned().fold(0.0_f64, f64::max);
        let min = mags.iter().cloned().fold(f64::INFINITY, f64::min);
        max / min
    }

    /// Solve `C x = b` by DFT diagonalization: `x = IDFT(DFT(b) / λ)`. The
    /// diagonalization the QFT performs on hardware, computed here as a naive
    /// O(N²) transform. Returns `Err` if the system is singular (some
    /// `|λ_k|` below a norm-relative floor) — e.g. the heat operator at `ξ = 0`.
    pub fn solve_via_dft(&self, b: &[Complex64]) -> Result<Vec<Complex64>, String> {
        let n = self.n;
        assert_eq!(b.len(), n, "rhs length must equal n");
        let lambda = self.eigenvalues();
        let max_mag = lambda.iter().map(|z| z.norm()).fold(0.0_f64, f64::max);
        let floor = 1e-12 * max_mag.max(1e-300);
        let w = |e: f64| Complex64::from_polar(1.0, e);
        // forward DFT, normalized: β_k = (1/N) Σ_m b_m ω^{-mk}
        let beta: Vec<Complex64> = (0..n)
            .map(|k| {
                let s: Complex64 = (0..n)
                    .map(|m| b[m] * w(-2.0 * PI * (m * k) as f64 / n as f64))
                    .sum();
                s / n as f64
            })
            .collect();
        // divide by eigenvalues
        let mut gamma = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..n {
            if lambda[k].norm() <= floor {
                return Err(format!(
                    "singular circulant: |λ_{k}| = {:.3e} ≤ floor {floor:.3e}",
                    lambda[k].norm()
                ));
            }
            gamma[k] = beta[k] / lambda[k];
        }
        // inverse DFT: x_m = Σ_k γ_k ω^{mk}
        Ok((0..n)
            .map(|m| {
                (0..n)
                    .map(|k| gamma[k] * w(2.0 * PI * (m * k) as f64 / n as f64))
                    .sum()
            })
            .collect())
    }

    /// Residual `‖C x − b‖₂` via the assembled dense matrix — a consistency
    /// cross-check on a given `x` (it reuses `dense()`, so it is not a *second*
    /// independent oracle beyond the LU comparison; it guards against an `x`
    /// that drifted from the system).
    pub fn residual(&self, x: &[Complex64], b: &[Complex64]) -> f64 {
        let a = self.dense();
        let n = self.n;
        (0..n)
            .map(|i| {
                let ax: Complex64 = (0..n).map(|j| a[i][j] * x[j]).sum();
                (ax - b[i]).norm_sqr()
            })
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic synthetic rhs in `[-1,1] + i[-1,1]`.
    fn synthetic_b(n: usize, seed: u64) -> Vec<Complex64> {
        let mut st = seed | 1;
        let mut nxt = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
        };
        (0..n).map(|_| Complex64::new(nxt(), nxt())).collect()
    }

    #[test]
    fn dft_solve_matches_independent_lu() {
        // Diagonalization solve vs partial-pivot LU on the dense matrix — two
        // different algorithms — to tight tolerance.
        let c = Circulant::heat_transfer(64, 0.2);
        let b = synthetic_b(64, 7);
        let x_dft = c.solve_via_dft(&b).unwrap();
        let x_lu = solve_dense_gaussian(&c.dense(), &b);
        let max_diff = x_dft
            .iter()
            .zip(x_lu.iter())
            .map(|(d, l)| (d - l).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_diff <= 1e-9, "DFT vs LU mismatch {max_diff:.3e}");
    }

    #[test]
    fn asymmetric_circulant_solve_matches_lu() {
        // The catch for exponent-sign bugs: an ASYMMETRIC circulant has complex
        // eigenvalues, so a flipped ω-sign would NOT cancel (unlike the
        // real-symmetric heat operator). Solve vs independent LU.
        let c = Circulant {
            n: 48,
            terms: vec![(0, 2.0), (1, 1.0), (2, 0.5)],
        };
        let b = synthetic_b(48, 13);
        let x_dft = c.solve_via_dft(&b).unwrap();
        let x_lu = solve_dense_gaussian(&c.dense(), &b);
        let max_diff = x_dft
            .iter()
            .zip(x_lu.iter())
            .map(|(d, l)| (d - l).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_diff <= 1e-9, "asymmetric DFT vs LU mismatch {max_diff:.3e}");
        // and its eigenvalues are genuinely complex (test is non-degenerate).
        assert!(
            c.eigenvalues().iter().any(|z| z.im.abs() > 0.1),
            "asymmetric circulant should have complex eigenvalues"
        );
    }

    #[test]
    fn analytic_eigenvalues_match_dft_of_assembled_column() {
        // Orthogonal oracle: closed-form λ vs DFT of the assembled first column.
        for c in [
            Circulant::heat_transfer(31, 0.2), // odd N too
            Circulant {
                n: 48,
                terms: vec![(0, 2.0), (1, 1.0), (2, 0.5)],
            },
        ] {
            let a = c.eigenvalues();
            let d = c.eigenvalues_from_dense();
            let worst = a
                .iter()
                .zip(&d)
                .map(|(x, y)| (x - y).norm())
                .fold(0.0_f64, f64::max);
            assert!(worst <= 1e-10, "eigenvalue formula vs assembly {worst:.3e}");
        }
    }

    #[test]
    fn n2_eigenvalues_match_proven_lean_corr1() {
        // The 2×2 circulant `!![a,b;b,a]` is diagonalized by the DFT to
        // `diag(a+b, a-b)` (the QFT conjugation `F·C·Fᴴ`, formally proven in the
        // upstream Lean proof tree). This test checks the `eigenvalues()` formula
        // reproduces exactly that spectrum `{a+b, a-b}`, then cross-checks the
        // DFT solve `x = F⁻¹ Λ⁻¹ F b` against an independent dense LU.
        for (a, b) in [(2.0_f64, 1.0_f64), (0.7, -0.3), (5.0, 5.0)] {
            let c = Circulant {
                n: 2,
                terms: vec![(0, a), (1, b)],
            };
            let lam = c.eigenvalues();
            assert!((lam[0].re - (a + b)).abs() <= 1e-12 && lam[0].im.abs() <= 1e-12);
            assert!((lam[1].re - (a - b)).abs() <= 1e-12 && lam[1].im.abs() <= 1e-12);
            if (a + b).abs() > 1e-9 && (a - b).abs() > 1e-9 {
                let bvec = synthetic_b(2, 3);
                let x_dft = c.solve_via_dft(&bvec).unwrap();
                let x_lu = solve_dense_gaussian(&c.dense(), &bvec);
                let worst = x_dft
                    .iter()
                    .zip(&x_lu)
                    .map(|(d, l)| (d - l).norm())
                    .fold(0.0_f64, f64::max);
                assert!(worst <= 1e-10, "n=2 DFT vs LU mismatch {worst:.3e}");
            }
        }
    }

    #[test]
    fn general_n_solve_operator_and_noise_match_proven_lean() {
        // Validates the GENERAL-n theorems `CirculantSolveGeneral.{circulant_solve_operator,
        // circulant_solve_noise_deviation}` numerically at N = 2^n (N=4, N=8) in the
        // Lean `+`-exponent convention: F_{jk} = (1/√N)·ω^{+jk}, ω = e^{2πi/N},
        // circEigen[k] = Σ_m v[m]·ω^{+km}. (The solve operator equals C⁻¹ regardless
        // of the ± convention since C⁻¹ is unique; we use `+` to mirror the proof.)
        let w = |e: f64| Complex64::from_polar(1.0, e);
        for &nbits in &[2usize, 3] {
            let n = 1usize << nbits; // N = 4, 8
            let v = synthetic_b(n, 0x51 + nbits as u64);
            // Dense circulant C[i][j] = v[(i - j) mod N]  (mathlib `circulant v i j = v(i-j)`).
            let c: Vec<Vec<Complex64>> = (0..n)
                .map(|i| (0..n).map(|j| v[(i + n - j) % n]).collect())
                .collect();
            // Eigenvalues, `+` convention: circEigen[k] = Σ_m v[m]·ω^{km}.
            let eig: Vec<Complex64> = (0..n)
                .map(|k| (0..n).map(|m| v[m] * w(2.0 * PI * (k * m) as f64 / n as f64)).sum())
                .collect();
            assert!(eig.iter().all(|z| z.norm() > 1e-6), "circulant must be nonsingular");
            // Conjugation operator `op(d) = Fᴴ·diag(d)·F`, entrywise:
            //   op(d)[a][b] = (1/N) Σ_k ω^{k(b−a)} · d[k].
            let op = |d: &[Complex64]| -> Vec<Vec<Complex64>> {
                (0..n)
                    .map(|a| {
                        (0..n)
                            .map(|b| {
                                let acc: Complex64 = (0..n)
                                    .map(|k| {
                                        w(2.0 * PI * ((k * b) as f64 - (k * a) as f64) / n as f64)
                                            * d[k]
                                    })
                                    .sum();
                                acc / n as f64
                            })
                            .collect()
                    })
                    .collect()
            };
            let inv_eig: Vec<Complex64> = eig.iter().map(|z| z.inv()).collect();
            // (a) circulant_solve_operator: S = Fᴴ·diag(1/eig)·F = C⁻¹, so S·C = I.
            let s = op(&inv_eig);
            for a in 0..n {
                for b in 0..n {
                    let sc: Complex64 = (0..n).map(|k| s[a][k] * c[k][b]).sum();
                    let exp = if a == b { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) };
                    assert!((sc - exp).norm() <= 1e-9, "S·C ≠ I at N={n}, ({a},{b})");
                }
            }
            // (b) circulant_solve_noise_deviation. With noisy reciprocals μ = 1/eig + ε,
            //     S(μ) − C⁻¹ = Fᴴ·diag(μ − 1/eig)·F = op(ε). To make this a REAL check of
            //     the theorem (not just linearity of `op`), the baseline C⁻¹ is computed
            //     INDEPENDENTLY by dense Gaussian solves against the identity columns — so
            //     the assertion forces op(1/eig) = C⁻¹ (the theorem's content), not the
            //     tautology op(μ) − op(1/eig) = op(ε).
            let mut c_inv = vec![vec![Complex64::new(0.0, 0.0); n]; n];
            for j in 0..n {
                let mut ej = vec![Complex64::new(0.0, 0.0); n];
                ej[j] = Complex64::new(1.0, 0.0);
                let col = solve_dense_gaussian(&c, &ej);
                for i in 0..n {
                    c_inv[i][j] = col[i];
                }
            }
            let eps: Vec<Complex64> =
                synthetic_b(n, 0x9e + nbits as u64).iter().map(|z| z * 0.1).collect();
            let mu: Vec<Complex64> = inv_eig.iter().zip(&eps).map(|(r, e)| r + e).collect();
            let s_mu = op(&mu);
            let dev_pred = op(&eps);
            for a in 0..n {
                for b in 0..n {
                    let actual = s_mu[a][b] - c_inv[a][b]; // op(μ) − (independent) C⁻¹
                    assert!(
                        (actual - dev_pred[a][b]).norm() <= 1e-9,
                        "noise deviation identity broke at N={n}, ({a},{b})"
                    );
                }
            }
        }
    }

    #[test]
    fn n2_noisy_solve_deviation_matches_proven_lean() {
        // CORR-1 noise model (formally proven in the upstream Lean proof tree,
        // `proofs/lean4/QuantumProofs/CirculantSolve.lean`): a NOISY solver that
        // applies imperfect eigenvalue reciprocals μ = 1/λ + ε (as from noisy
        // QPE) deviates from the exact solve by exactly the QFT-conjugated error
        // `Fᴴ·diag(ε)·F`. For N=2, F = (1/√2)·[[1,1],[1,-1]] (real, Fᴴ=F). This
        // checks the identity numerically + deviation linear in ε, → 0 as noise → 0.
        let r2 = std::f64::consts::FRAC_1_SQRT_2;
        // x = F · diag(μ0,μ1) · F · b  for the 2×2 real DFT.
        let dft2_solve = |mu0: Complex64, mu1: Complex64, b: &[Complex64; 2]| -> [Complex64; 2] {
            let f0 = (b[0] + b[1]) * r2; // (F b)_0
            let f1 = (b[0] - b[1]) * r2; // (F b)_1
            let g0 = mu0 * f0; // diag(μ)·(F b)
            let g1 = mu1 * f1;
            [(g0 + g1) * r2, (g0 - g1) * r2] // F · g
        };
        let a = 2.0;
        let b_coeff = 1.0;
        let lam0 = Complex64::new(a + b_coeff, 0.0); // a+b
        let lam1 = Complex64::new(a - b_coeff, 0.0); // a-b
        let bvec = [Complex64::new(0.7, -0.2), Complex64::new(-1.1, 0.5)];
        for &eps in &[0.3_f64, 0.15, 0.0] {
            let e0 = Complex64::new(eps, 0.0);
            let e1 = Complex64::new(-0.5 * eps, 0.0);
            let mu0 = lam0.inv() + e0; // noisy reciprocal
            let mu1 = lam1.inv() + e1;
            let x_exact = dft2_solve(lam0.inv(), lam1.inv(), &bvec);
            let x_noisy = dft2_solve(mu0, mu1, &bvec);
            // Lean identity: (x_noisy − x_exact) == Fᴴ·diag(ε)·F·b == dft2_solve(ε, b).
            let dev_predicted = dft2_solve(e0, e1, &bvec);
            for k in 0..2 {
                let actual = x_noisy[k] - x_exact[k];
                assert!(
                    (actual - dev_predicted[k]).norm() <= 1e-12,
                    "noise deviation identity broke at eps={eps}, k={k}"
                );
            }
            // Vanishing/linearity: ‖deviation‖ ≤ max|ε| · ‖b‖ (F is an isometry).
            let dev_norm = (x_noisy[0] - x_exact[0]).norm().hypot((x_noisy[1] - x_exact[1]).norm());
            let bnorm = bvec[0].norm().hypot(bvec[1].norm());
            assert!(dev_norm <= eps.abs() * bnorm + 1e-12, "deviation exceeds ‖ε‖·‖b‖");
            if eps == 0.0 {
                assert!(dev_norm <= 1e-12, "zero noise must give zero deviation");
            }
        }
    }

    #[test]
    fn dft_solve_residual_is_tiny() {
        let c = Circulant::heat_transfer(128, 0.3);
        let b = synthetic_b(128, 11);
        let x = c.solve_via_dft(&b).unwrap();
        assert!(c.residual(&x, &b) <= 1e-10, "residual too large");
    }

    #[test]
    fn condition_number_matches_analytic_even_n_and_grows_as_xi_shrinks() {
        // κ = (4+ξ)/ξ holds for EVEN N (k=N/2 hits cos π = −1 exactly).
        for &xi in &[0.5, 0.2, 0.05] {
            let c = Circulant::heat_transfer(64, xi);
            let analytic = (4.0 + xi) / xi;
            let rel = (c.condition_number() - analytic).abs() / analytic;
            assert!(rel <= 1e-6, "even-N κ off by {rel:.3e} at ξ={xi}");
        }
        assert!(
            Circulant::heat_transfer(64, 0.05).condition_number()
                > Circulant::heat_transfer(64, 0.5).condition_number(),
            "condition number must grow as ξ shrinks"
        );
    }

    #[test]
    fn odd_n_closed_form_is_known_to_differ() {
        // Documented boundary: the (4+ξ)/ξ closed form is even-N-only. At odd N
        // it does NOT hold (no exact cos = −1 mode) — assert we know this, so a
        // future caller is not misled. condition_number() itself stays exact.
        let c = Circulant::heat_transfer(7, 0.2);
        let analytic_even = (4.0 + 0.2) / 0.2;
        let rel = (c.condition_number() - analytic_even).abs() / analytic_even;
        assert!(rel > 0.01, "odd-N should deviate from the even-N closed form");
    }

    #[test]
    fn singular_system_returns_err() {
        // ξ = 0 ⇒ λ_0 = 0: the paper's knee endpoint. Must be a recoverable
        // Err, not a panic or silent garbage.
        let c = Circulant::heat_transfer(32, 0.0);
        let b = synthetic_b(32, 1);
        let res = c.solve_via_dft(&b);
        assert!(res.is_err(), "ξ=0 must report singular");
        assert!(c.condition_number().is_infinite(), "κ should be ∞ at ξ=0");
    }

    #[test]
    fn fuzz_random_circulant_dft_matches_gaussian_and_eigformula() {
        // Property over random banded circulants + random complex b: the DFT
        // solve == independent Gaussian solve, and the analytic eigenvalues ==
        // DFT-of-assembled-column. Dominant diagonal keeps most draws solvable.
        let mut st = 0xBEEF_0077u64;
        let mut lcg = |s: &mut u64| {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 33) as f64 / (1u64 << 31) as f64
        };
        let mut solved = 0;
        for _ in 0..150 {
            let n = 8 + (lcg(&mut st) * 8.0) as usize; // 8..=15
            let nterms = 2 + (lcg(&mut st) * 3.0) as usize;
            let mut terms: Vec<(i64, f64)> = (0..nterms)
                .map(|_| ((lcg(&mut st) * 7.0) as i64 - 3, lcg(&mut st) * 2.0 - 1.0))
                .collect();
            terms.push((0, 4.0)); // dominant diagonal ⇒ usually non-singular
            let circ = Circulant { n, terms };

            let worst_e = circ
                .eigenvalues()
                .iter()
                .zip(circ.eigenvalues_from_dense())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0_f64, f64::max);
            assert!(worst_e < 1e-9, "fuzz eig-formula vs assembly {worst_e:.2e}");

            let b: Vec<Complex64> = (0..n)
                .map(|_| Complex64::new(lcg(&mut st) * 2.0 - 1.0, lcg(&mut st) * 2.0 - 1.0))
                .collect();
            if let Ok(x) = circ.solve_via_dft(&b) {
                let xg = solve_dense_gaussian(&circ.dense(), &b);
                let d = x
                    .iter()
                    .zip(&xg)
                    .map(|(a, g)| (a - g).norm())
                    .fold(0.0_f64, f64::max);
                assert!(d < 1e-6, "fuzz DFT vs Gaussian {d:.2e} (n={n})");
                solved += 1;
            }
        }
        assert!(solved > 120, "too many singular draws ({solved}/150)");
    }

    #[test]
    fn eigenvalues_match_heat_transfer_formula() {
        let (n, xi) = (32usize, 0.2);
        let c = Circulant::heat_transfer(n, xi);
        for (k, lk) in c.eigenvalues().iter().enumerate() {
            let expect = (-2.0 - xi) + 2.0 * (2.0 * PI * k as f64 / n as f64).cos();
            assert!((lk.re - expect).abs() <= 1e-12, "λ_{k} re");
            assert!(lk.im.abs() <= 1e-12, "λ_{k} should be real");
        }
    }
}
