//! Quantum Oracle Sketching (QOS) — emulation of Zhao, Babbush, Huang et al.
//! 2026, "Exponential quantum advantage in processing massive classical data"
//! (arXiv:2604.07639; reference code github.com/haimengzhao/quantum-oracle-sketching).
//!
//! QOS instantiates the query oracle a quantum algorithm needs **from a stream
//! of random classical samples**, without ever loading the full dataset (no
//! QRAM). Each sample applies a tiny data-dependent phase rotation; the
//! rotations accumulate coherently, so after enough samples the quantum state
//! approximates the target oracle. The qubit count grows only
//! *polylogarithmically* with the dataset size (`n = ⌈log₂ dim⌉`).
//!
//! **Scope (read this before believing the variable names).** This module
//! reproduces the *sample-complexity convergence* of the phase-oracle sketch at
//! small `n`, two independent ways that must agree:
//!
//!   1. [`state_sketch`] — the **expected-unitary** closed form the authors use
//!      for benchmarking (a deterministic upper bound). Per diagonal entry with
//!      target value `t`:
//!      ```text
//!      sketch_N(t) = (1 + p·(e^{ i φ t /(p N) } − 1))^N  ──N→∞──▶  e^{ i φ t }.
//!      ```
//!   2. [`state_sketch_stochastic`] — a **Monte-Carlo simulation of the actual
//!      sampling process** (per-sample Bernoulli inclusion). It never uses the
//!      closed form, so `E[stochastic] = state_sketch` is a real cross-check,
//!      not an identity restated.
//!
//! The headline numeric certificate is that (1) the closed form is the mean of
//! (2), and (2)'s/(1)'s infidelity to the exact oracle falls as `~1/N²`.
//! **Out of scope** (the rest of the paper): the QSVT linear-algebra step, the
//! interferometric classical-shadow readout, and the real RNA-seq / sentiment
//! experiments. No circuit is executed on `n` qubits here — this is a numerical
//! reproduction of the convergence, not a fault-tolerant implementation.

use num_complex::Complex64;

/// Number of qubits needed to index a `dim`-dimensional vector: `⌈log₂ dim⌉`.
/// Correct for any `dim ≥ 1` (not just powers of two).
pub fn index_qubits(dim: usize) -> usize {
    assert!(dim >= 1, "dim must be ≥ 1");
    if dim <= 1 {
        0
    } else {
        (dim - 1).ilog2() as usize + 1
    }
}

/// The exact target state `|T⟩ = D_target |+⟩^n`, where `D_target =
/// diag(e^{i φ v_j})` is the phase oracle of the data vector `v`, applied to the
/// uniform superposition and normalized. `v.len()` must be a power of two.
pub fn target_state(v: &[f64], phi: f64) -> Vec<Complex64> {
    let dim = v.len();
    let amp = 1.0 / (dim as f64).sqrt();
    v.iter()
        .map(|&t| Complex64::from_polar(amp, phi * t))
        .collect()
}

/// The QOS **expected-unitary** sketch of the phase oracle of `v`, built from
/// `n_samples` streamed samples each touching the oracle with inclusion
/// probability `p ∈ (0,1]`. Closed form; as `n_samples → ∞` it converges to
/// [`target_state`]. This is the deterministic mean of [`state_sketch_stochastic`].
pub fn state_sketch(v: &[f64], phi: f64, p: f64, n_samples: u64) -> Vec<Complex64> {
    let dim = v.len();
    let amp = 1.0 / (dim as f64).sqrt();
    let n = n_samples as f64;
    let i = Complex64::i();
    v.iter()
        .map(|&t| {
            // per-sample angle chosen so the N→∞ limit is exactly e^{iφt}
            let per = i * (phi * t / (p * n));
            let base = Complex64::new(1.0, 0.0) + p * (per.exp() - 1.0);
            // `powf` is on the principal branch; valid because Re(base) > 0 in
            // the small-per-sample-angle regime QOS operates in.
            debug_assert!(base.re > 0.0, "powf branch requires Re(base) > 0");
            base.powf(n) * amp
        })
        .collect()
}

/// One realization of the **stochastic** sampling process whose mean the
/// expected unitary computes: each of `n_samples` independent draws includes
/// entry `j` with probability `p`, advancing its phase by `α_j = φ v_j/(p N)`
/// when included. The realized per-entry phase is `e^{i α_j K_j}` with
/// `K_j ~ Binomial(N, p)`, so `E[·] = state_sketch`. `rng` is advanced in place
/// (seeded LCG) — distinct seeds give independent realizations.
///
/// This deliberately does **not** call [`state_sketch`]; averaging many
/// realizations and comparing to the closed form is an *independent* validation.
pub fn state_sketch_stochastic(
    v: &[f64],
    phi: f64,
    p: f64,
    n_samples: u64,
    rng: &mut u64,
) -> Vec<Complex64> {
    let dim = v.len();
    let amp = 1.0 / (dim as f64).sqrt();
    let n = n_samples as f64;
    v.iter()
        .map(|&t| {
            let alpha = phi * t / (p * n);
            // K ~ Binomial(n_samples, p) by counting Bernoulli(p) inclusions.
            let mut k = 0u64;
            for _ in 0..n_samples {
                *rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (*rng >> 33) as f64 / (1u64 << 31) as f64;
                if u < p {
                    k += 1;
                }
            }
            Complex64::from_polar(amp, alpha * k as f64)
        })
        .collect()
}

/// Infidelity `1 − |⟨a|b⟩|² / (‖a‖²‖b‖²)`. Normalizing by the norms makes this
/// robust to the slight non-unitarity of the expected-unitary approximation (a
/// true quantum state is normalized), so it measures the *state* error.
pub fn infidelity(a: &[Complex64], b: &[Complex64]) -> f64 {
    let ip: Complex64 = a.iter().zip(b).map(|(x, y)| x.conj() * y).sum();
    let na: f64 = a.iter().map(|x| x.norm_sqr()).sum();
    let nb: f64 = b.iter().map(|x| x.norm_sqr()).sum();
    (1.0 - ip.norm_sqr() / (na * nb)).max(0.0)
}

/// Empirical convergence exponent: least-squares slope of `ln(infidelity)` vs
/// `ln(N)` over `sample_counts`. For the expected-unitary sketch the amplitude
/// error is `O(1/N)`, so the infidelity is `Var(φv)·O(1/N²)` and the slope is
/// `≈ -2` — the paper's sample-complexity scaling.
///
/// **Panics** if any infidelity is below `1e-12` (the asymptotic fit is only
/// meaningful above the float floor — choose smaller `N` or a higher-variance
/// instance; a zero-variance `v` also trips this, by design).
pub fn scaling_exponent(v: &[f64], phi: f64, p: f64, sample_counts: &[u64]) -> f64 {
    let target = target_state(v, phi);
    let infs: Vec<f64> = sample_counts
        .iter()
        .map(|&n| infidelity(&state_sketch(v, phi, p, n), &target))
        .collect();
    assert!(
        infs.iter().all(|&f| f > 1e-12),
        "scaling_exponent: infidelity at/below the float floor (min {:.2e}) — \
         fit only valid in the asymptotic regime (smaller N / higher-variance v)",
        infs.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    let xs: Vec<f64> = sample_counts.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = infs.iter().map(|&f| f.ln()).collect();
    let k = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / k;
    let my = ys.iter().sum::<f64>() / k;
    let num: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    num / den
}

/// Halmos dilation of a diagonal Hermitian operator with real entries `a` and
/// `‖a‖∞ ≤ 1`: the `2dim × 2dim` unitary
/// `U = [[A, √(I−A²)], [√(I−A²), −A]]` whose top-left block is `A`. A standalone
/// block-encoding primitive (the dilation the paper's QSVT step would consume) —
/// kept available but **not** part of the convergence path above.
pub fn halmos_dilation_diag(a: &[f64]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    let mut u = vec![vec![Complex64::new(0.0, 0.0); 2 * n]; 2 * n];
    for (j, &aj) in a.iter().enumerate() {
        debug_assert!(aj.abs() <= 1.0 + 1e-12, "‖A‖∞ ≤ 1 required");
        let s = (1.0 - aj * aj).max(0.0).sqrt();
        u[j][j] = Complex64::new(aj, 0.0); // A
        u[j][n + j] = Complex64::new(s, 0.0); // √(I−A²)
        u[n + j][j] = Complex64::new(s, 0.0); // √(I−A²)
        u[n + j][n + j] = Complex64::new(-aj, 0.0); // −A
    }
    u
}

/// Extract the `block_dim × block_dim` top-left block of a unitary (the
/// block-encoded operator).
pub fn get_block_encoded(u: &[Vec<Complex64>], block_dim: usize) -> Vec<Vec<Complex64>> {
    (0..block_dim).map(|r| u[r][..block_dim].to_vec()).collect()
}

/// Deterministic, seeded generators for **synthetic** QOS instances (eval-only;
/// no real datasets). Each models one of the paper's oracle kinds as a length-
/// `dim` target vector `v` consumed by [`target_state`] / [`state_sketch`].
/// `dim` should be a power of two so `n = ⌈log₂ dim⌉` qubits index it exactly.
pub mod synthetic {
    /// One step of a truncated LCG (PCG multiplier, no output permutation),
    /// returning a uniform value in `[0, 1)`. Adequate for generating bounded,
    /// nonzero-variance instances — the only property the convergence fit needs.
    fn unit(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// General data vector: entries uniform in `[-1, 1]`.
    pub fn random_vector(dim: usize, seed: u64) -> Vec<f64> {
        let mut st = seed | 1;
        (0..dim).map(|_| unit(&mut st) * 2.0 - 1.0).collect()
    }

    /// Boolean-function oracle: entries in `{0, 1}` — the phase oracle of a
    /// random `f : {0,1}^n → {0,1}`.
    pub fn boolean_oracle(dim: usize, seed: u64) -> Vec<f64> {
        let mut st = seed | 1;
        (0..dim)
            .map(|_| if unit(&mut st) < 0.5 { 0.0 } else { 1.0 })
            .collect()
    }

    /// Matrix-element oracle: entries of a random rank-`r` matrix reshaped to a
    /// length-`dim` vector and rescaled to `[-1, 1]` (the value an `A[i,j]`
    /// query returns). Low rank is what a real data matrix has; the rescale
    /// keeps it inside the phase-oracle domain.
    pub fn matrix_element_oracle(dim: usize, seed: u64) -> Vec<f64> {
        assert!(dim >= 1);
        let mut st = seed | 1;
        let rank = 3usize;
        let m = (dim as f64).sqrt().ceil() as usize; // row stride
        let cols = dim.div_ceil(m);
        let u: Vec<Vec<f64>> = (0..rank)
            .map(|_| (0..m).map(|_| unit(&mut st) * 2.0 - 1.0).collect())
            .collect();
        let w: Vec<Vec<f64>> = (0..rank)
            .map(|_| (0..cols).map(|_| unit(&mut st) * 2.0 - 1.0).collect())
            .collect();
        let raw: Vec<f64> = (0..dim)
            .map(|k| (0..rank).map(|r| u[r][k % m] * w[r][k / m]).sum())
            .collect();
        let max = raw.iter().fold(0.0f64, |a, &b| a.max(b.abs())).max(1e-12);
        raw.iter().map(|&x| x / max).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_unitary_is_the_mean_of_the_stochastic_process() {
        // The non-tautological certificate: the closed-form expected unitary is
        // the Monte-Carlo mean of the actual stochastic sampling process (which
        // never touches the closed form), within Monte-Carlo error.
        let v = synthetic::random_vector(64, 9);
        let (phi, p, n, trials) = (0.7, 0.5, 256u64, 400usize);
        let analytic = state_sketch(&v, phi, p, n);
        let dim = v.len();
        let mut avg = vec![Complex64::new(0.0, 0.0); dim];
        let mut st = 12345u64;
        for _ in 0..trials {
            let s = state_sketch_stochastic(&v, phi, p, n, &mut st);
            for (a, x) in avg.iter_mut().zip(&s) {
                *a += x;
            }
        }
        for a in avg.iter_mut() {
            *a /= trials as f64;
        }
        // MC amplitude error ~ 1/√trials ⇒ infidelity ~ 1/trials; 400 ⇒ ≲ 1e-2.
        let inf = infidelity(&avg, &analytic);
        assert!(
            inf <= 1e-2,
            "stochastic mean should match the closed form within MC error, got {inf}"
        );
    }

    #[test]
    fn stochastic_sketch_also_converges_to_the_exact_oracle() {
        // The stochastic process itself (independent of the closed form) gets
        // closer to the exact oracle as N grows — averaged over realizations.
        let v = synthetic::random_vector(32, 4);
        let (phi, p) = (0.6, 0.5);
        let target = target_state(&v, phi);
        let mean_inf = |n: u64, seed: u64| {
            let trials = 200usize;
            let dim = v.len();
            let mut avg = vec![Complex64::new(0.0, 0.0); dim];
            let mut st = seed;
            for _ in 0..trials {
                let s = state_sketch_stochastic(&v, phi, p, n, &mut st);
                for (a, x) in avg.iter_mut().zip(&s) {
                    *a += x;
                }
            }
            for a in avg.iter_mut() {
                *a /= trials as f64;
            }
            infidelity(&avg, &target)
        };
        assert!(
            mean_inf(1024, 1) < mean_inf(256, 1),
            "more samples ⇒ closer to the exact oracle"
        );
    }

    #[test]
    fn sketch_converges_to_target_with_samples() {
        // Closed-form sanity: infidelity to the exact oracle decreases
        // monotonically as the sample count grows, toward 0.
        let v = synthetic::random_vector(256, 7);
        let (phi, p) = (0.8, 0.5);
        let target = target_state(&v, phi);
        let mut prev = f64::INFINITY;
        for &n in &[64u64, 256, 1024, 4096, 16384] {
            let inf = infidelity(&state_sketch(&v, phi, p, n), &target);
            assert!(
                inf < prev,
                "infidelity should decrease: {inf} !< {prev} at N={n}"
            );
            prev = inf;
        }
        assert!(
            prev <= 1e-8,
            "infidelity at N=16384 should be tiny, got {prev}"
        );
    }

    #[test]
    fn scaling_exponent_is_minus_two_across_instance_types() {
        // QLSS Phase-1 certificate: the fitted log-log convergence exponent is
        // ≈ -2 (infidelity ~ 1/N²) for every QOS oracle kind. Tight tolerance
        // (±0.02): the real deviation is ~2e-4, so this pins the scaling, it is
        // not a loose smoke test.
        let dim = 256;
        let counts = [256u64, 512, 1024, 2048, 4096];
        for (label, v) in [
            ("vector", synthetic::random_vector(dim, 1)),
            ("boolean", synthetic::boolean_oracle(dim, 2)),
            ("matrix-element", synthetic::matrix_element_oracle(dim, 3)),
        ] {
            let slope = scaling_exponent(&v, 0.8, 0.5, &counts);
            assert!(
                (slope + 2.0).abs() <= 0.02,
                "{label}: fitted exponent {slope} not within ±0.02 of -2"
            );
        }
    }

    #[test]
    #[should_panic(expected = "float floor")]
    fn scaling_exponent_refuses_the_floor_regime() {
        // Boundary: pushing N until infidelity hits the float floor must be
        // *rejected*, not silently fit to a garbage slope. Proves the guard
        // works so a pass elsewhere means the fit was in the valid regime.
        let v = synthetic::random_vector(64, 5);
        let counts = [1u64 << 18, 1 << 20, 1 << 22, 1 << 24];
        let _ = scaling_exponent(&v, 0.8, 0.5, &counts);
    }

    #[test]
    fn sketch_error_scales_inverse_square() {
        // Doubling the samples cuts the infidelity ~4× (error ~O(1/N) in
        // amplitude ⇒ infidelity ~O(1/N²)). Tight band around 4.0.
        let v = synthetic::random_vector(64, 3);
        let (phi, p) = (1.0, 0.4);
        let target = target_state(&v, phi);
        let i_n = infidelity(&state_sketch(&v, phi, p, 1000), &target);
        let i_2n = infidelity(&state_sketch(&v, phi, p, 2000), &target);
        let ratio = i_n / i_2n;
        assert!(
            (3.9..4.1).contains(&ratio),
            "expected ~4× drop, got {ratio}"
        );
    }

    #[test]
    fn fuzz_random_v_scaling_exponent_is_minus_two() {
        // Property over random data vectors and random φ: the fitted convergence
        // exponent stays ≈ −2 (infidelity ~ 1/N²) — not just the 3 fixed kinds.
        let mut s = 0xF00D_2024u64;
        let mut lcg = |st: &mut u64| {
            *st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*st >> 33) as f64 / (1u64 << 31) as f64
        };
        let counts = [256u64, 512, 1024, 2048, 4096];
        for _ in 0..60 {
            let dim = 1usize << (6 + (lcg(&mut s) * 3.0) as usize); // 64..=512
            let v: Vec<f64> = (0..dim).map(|_| lcg(&mut s) * 2.0 - 1.0).collect();
            let phi = 0.3 + lcg(&mut s) * 0.7; // 0.3..=1.0
            let slope = scaling_exponent(&v, phi, 0.5, &counts);
            assert!(
                (slope + 2.0).abs() <= 0.05,
                "fuzz slope {slope} (phi={phi})"
            );
        }
    }

    #[test]
    fn synthetic_generators_are_in_range_and_deterministic() {
        assert_eq!(
            synthetic::random_vector(64, 5),
            synthetic::random_vector(64, 5),
            "seeded generator must be deterministic"
        );
        assert!(synthetic::random_vector(64, 5)
            .iter()
            .all(|x| (-1.0..=1.0).contains(x)));
        assert!(synthetic::boolean_oracle(64, 1)
            .iter()
            .all(|&x| x == 0.0 || x == 1.0));
        assert!(synthetic::matrix_element_oracle(64, 1)
            .iter()
            .all(|x| (-1.0..=1.0).contains(x)));
        // Non-power-of-two dims must not panic or go out of bounds.
        assert_eq!(synthetic::matrix_element_oracle(100, 1).len(), 100);
    }

    #[test]
    fn target_state_is_normalized() {
        let v = synthetic::random_vector(128, 11);
        let s = target_state(&v, 0.5);
        let norm: f64 = s.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() <= 1e-12, "norm {norm}");
    }

    #[test]
    fn halmos_dilation_is_unitary_and_round_trips() {
        let a = vec![0.3, -0.7, 0.95, 0.0];
        let u = halmos_dilation_diag(&a);
        let n2 = u.len();
        let mut worst = 0.0f64;
        for r in 0..n2 {
            for c in 0..n2 {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..n2 {
                    acc += u[r][k] * u[c][k].conj();
                }
                let expect = if r == c { 1.0 } else { 0.0 };
                worst = worst.max((acc - Complex64::new(expect, 0.0)).norm());
            }
        }
        assert!(worst <= 1e-12, "U not unitary, worst {worst}");
        let block = get_block_encoded(&u, a.len());
        let mut berr = 0.0f64;
        for (j, &aj) in a.iter().enumerate() {
            berr = berr.max((block[j][j] - Complex64::new(aj, 0.0)).norm());
        }
        assert!(berr <= 1e-12, "block-encoding error {berr}");
    }

    #[test]
    fn index_qubits_is_ceil_log2() {
        assert_eq!(index_qubits(1024), 10);
        assert_eq!(index_qubits(65536), 16);
        assert_eq!(index_qubits(1000), 10); // non-power-of-two: ⌈log₂⌉
        assert_eq!(index_qubits(1), 0);
    }
}
