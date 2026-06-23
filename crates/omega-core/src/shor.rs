//! Shor's algorithm demo: factor small integers via quantum period finding.
//!
//! This is a *demonstration*, not a scalable implementation. Period finding is
//! simulated faithfully: the combined (counting ⊗ work) register is built
//! explicitly, the inverse-QFT measurement marginal is computed, one counting
//! outcome is sampled, and continued fractions recover the period.
//!
//! Statevector size is `2^(t+n)` where `t = 2 * ceil(log2 N)` and
//! `n = ceil(log2 N)`. For N ≤ 63 the total register fits comfortably in
//! under 2^18 amplitudes.

use std::collections::HashMap;
use std::f64::consts::PI;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Outcome of a Shor factoring attempt.
#[derive(Clone, Debug)]
pub struct ShorResult {
    /// The composite input.
    pub n: u64,
    /// Non-trivial factor pair (p, q) with p*q == n, if found.
    pub factors: Option<(u64, u64)>,
    /// Recovered period r (if period-finding succeeded at least once).
    pub period: Option<u64>,
    /// Number of random-`a` attempts made.
    pub iterations: u32,
    /// The `a` that finally worked (0 if no success).
    pub chosen_a: u64,
}

/// Attempt to factor `n` via Shor's algorithm.
///
/// `max_attempts` caps the number of random-`a` trials. Returns the result
/// even on failure so callers can inspect `iterations`.
pub fn factor(n: u64, seed: Option<u64>, max_attempts: u32) -> ShorResult {
    // --- Trivial / classical short-circuits ---
    if n < 2 {
        return ShorResult {
            n,
            factors: None,
            period: None,
            iterations: 0,
            chosen_a: 0,
        };
    }
    if n.is_multiple_of(2) {
        return ShorResult {
            n,
            factors: Some((2, n / 2)),
            period: None,
            iterations: 0,
            chosen_a: 0,
        };
    }
    if let Some((p, q)) = integer_root_factor(n) {
        return ShorResult {
            n,
            factors: Some((p, q)),
            period: None,
            iterations: 0,
            chosen_a: 0,
        };
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => rand::make_rng::<StdRng>(),
    };

    let mut last_period = None;
    for attempt in 1..=max_attempts {
        let a = rng.random_range(2..n);
        let g = gcd(a, n);
        if g > 1 {
            // Lucky classical hit; still counts as a Shor iteration.
            return ShorResult {
                n,
                factors: Some((g, n / g)),
                period: last_period,
                iterations: attempt,
                chosen_a: a,
            };
        }

        // Per-attempt seed: derive from user seed so runs remain reproducible.
        let sub_seed = seed.map(|s| s.wrapping_add(attempt as u64));
        let r = quantum_period_finding(a, n, sub_seed);
        if let Some(r) = r {
            last_period = Some(r);
            if r % 2 == 0 {
                let x = mod_exp(a, r / 2, n);
                if x != n - 1 {
                    let p1 = gcd(x + 1, n);
                    // x ≥ 0 so x.wrapping_sub(1) is safe only if x >= 1.
                    let p2 = if x >= 1 { gcd(x - 1, n) } else { 1 };
                    for p in [p1, p2] {
                        if p > 1 && p < n {
                            return ShorResult {
                                n,
                                factors: Some((p, n / p)),
                                period: last_period,
                                iterations: attempt,
                                chosen_a: a,
                            };
                        }
                    }
                }
            }
        }
    }

    ShorResult {
        n,
        factors: None,
        period: last_period,
        iterations: max_attempts,
        chosen_a: 0,
    }
}

/// Simulate quantum period-finding for f(x) = a^x mod n.
///
/// Builds |ψ⟩ = (1/√T) Σ_x |x⟩ ⊗ |a^x mod n⟩ on (counting ⊗ work),
/// computes the marginal distribution on the counting register after an
/// inverse QFT, samples a counting outcome k ∈ [0, T), and runs continued
/// fractions on k/T to recover the period r (with r < n).
pub fn quantum_period_finding(a: u64, n: u64, seed: Option<u64>) -> Option<u64> {
    let n_qubits = qubits_needed(n);
    let t_qubits = 2 * n_qubits;
    let t_dim: u64 = 1 << t_qubits;

    // Tabulate a^x mod n for x ∈ [0, T).
    let pow_table: Vec<u64> = (0..t_dim).map(|x| mod_exp(a, x, n)).collect();

    // Group counting indices by their work-register value.
    let mut y_to_xs: HashMap<u64, Vec<u64>> = HashMap::new();
    for (x, &y) in pow_table.iter().enumerate() {
        y_to_xs.entry(y).or_default().push(x as u64);
    }

    // Marginal probability over counting register after inverse QFT and
    // tracing out the work register.
    //
    //   p(k) = (1/T²) Σ_y | Σ_{x ∈ orbit(y)} exp(-2πi x k / T) |²
    let mut p_k = vec![0.0f64; t_dim as usize];
    let t_f = t_dim as f64;
    let inv_t_sq = 1.0 / (t_f * t_f);

    for xs in y_to_xs.values() {
        for k in 0..t_dim {
            let mut re = 0.0;
            let mut im = 0.0;
            for &x in xs {
                let phase = -2.0 * PI * ((x * k) % t_dim) as f64 / t_f;
                re += phase.cos();
                im += phase.sin();
            }
            p_k[k as usize] += (re * re + im * im) * inv_t_sq;
        }
    }

    // Renormalise to guard against accumulated FP error.
    let total: f64 = p_k.iter().sum();
    if total <= 0.0 {
        return None;
    }
    for p in &mut p_k {
        *p /= total;
    }

    // Sample a counting outcome.
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => rand::make_rng::<StdRng>(),
    };
    let rv: f64 = rng.random();
    let mut cum = 0.0;
    let mut sampled_k = 0u64;
    for (k, p) in p_k.iter().enumerate() {
        cum += *p;
        if rv <= cum {
            sampled_k = k as u64;
            break;
        }
    }

    // Continued fractions: find denominator r < n approximating k / T.
    // Verify by checking a^r mod n == 1.
    let candidates = continued_fraction_candidates(sampled_k, t_dim, n);
    candidates
        .into_iter()
        .find(|&r| r > 0 && r < n && mod_exp(a, r, n) == 1)
}

/// Convergents of k/T with denominator < bound, as candidate periods.
fn continued_fraction_candidates(k: u64, t_dim: u64, bound: u64) -> Vec<u64> {
    // Simple Euclidean expansion; collect denominators of successive convergents.
    let mut a = k;
    let mut b = t_dim;
    let mut h_prev: u64 = 0;
    let mut h_curr: u64 = 1;
    let mut k_prev: u64 = 1;
    let mut k_curr: u64 = 0;
    let mut out = Vec::new();
    while b != 0 {
        let q = a / b;
        let h_next = q.saturating_mul(h_curr).saturating_add(h_prev);
        let k_next = q.saturating_mul(k_curr).saturating_add(k_prev);
        if k_next >= bound {
            break;
        }
        out.push(k_next);
        h_prev = h_curr;
        h_curr = h_next;
        k_prev = k_curr;
        k_curr = k_next;
        let rem = a - q * b;
        a = b;
        b = rem;
    }
    out
}

// --- Small number-theoretic helpers ---

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// `base^exp mod modulus`, for modulus small enough that (modulus-1)² fits in u128.
fn mod_exp(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u128 = 1;
    let mut b: u128 = (base % modulus) as u128;
    let m: u128 = modulus as u128;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = (result * b) % m;
        }
        e >>= 1;
        b = (b * b) % m;
    }
    result as u64
}

/// Qubits needed to hold the set of values {0, 1, ..., n-1}.
fn qubits_needed(n: u64) -> usize {
    let mut bits = 0usize;
    let mut v = n.saturating_sub(1).max(1);
    while v > 0 {
        bits += 1;
        v >>= 1;
    }
    bits.max(1)
}

/// If n is a perfect power p^k with p,k ≥ 2 or has a small square root,
/// return one such factorisation.
fn integer_root_factor(n: u64) -> Option<(u64, u64)> {
    // Check p^k for k = 2..log2(n)
    let log_n = 64 - n.leading_zeros() as u64;
    for k in 2..=log_n {
        let p = integer_kth_root(n, k);
        if p >= 2 {
            let mut pk: u128 = 1;
            for _ in 0..k {
                pk *= p as u128;
                if pk > n as u128 {
                    break;
                }
            }
            if pk == n as u128 {
                return Some((p, n / p));
            }
        }
    }
    None
}

fn integer_kth_root(n: u64, k: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let approx = (n as f64).powf(1.0 / k as f64);
    let guess = approx.round() as u64;
    for cand in [guess.saturating_sub(1), guess, guess + 1] {
        let mut pk: u128 = 1;
        let mut overflow = false;
        for _ in 0..k {
            pk *= cand as u128;
            if pk > n as u128 {
                overflow = true;
                break;
            }
        }
        if !overflow && pk == n as u128 {
            return cand;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_15() {
        let r = factor(15, Some(42), 16);
        assert!(r.factors.is_some(), "expected to factor 15: {:?}", r);
        let (p, q) = r.factors.unwrap();
        assert_eq!(p * q, 15);
        assert!(p > 1 && q > 1);
        assert!(
            (p == 3 && q == 5) || (p == 5 && q == 3),
            "unexpected factor pair {:?}",
            (p, q)
        );
    }

    #[test]
    fn factor_21() {
        let r = factor(21, Some(7), 32);
        assert!(r.factors.is_some(), "expected to factor 21: {:?}", r);
        let (p, q) = r.factors.unwrap();
        assert_eq!(p * q, 21);
        assert!(p > 1 && q > 1);
    }

    #[test]
    fn mod_exp_basics() {
        assert_eq!(mod_exp(7, 4, 15), 1);
        assert_eq!(mod_exp(2, 10, 1000), 24);
        assert_eq!(mod_exp(0, 0, 7), 1);
    }

    #[test]
    fn gcd_basics() {
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(17, 23), 1);
        assert_eq!(gcd(100, 0), 100);
    }

    #[test]
    fn period_of_7_mod_15() {
        // ord_15(7) = 4  (7^1=7, 7^2=49≡4, 7^3=28≡13, 7^4=91≡1)
        let r = quantum_period_finding(7, 15, Some(1)).unwrap();
        assert_eq!(r, 4, "expected period 4, got {}", r);
    }

    #[test]
    fn factor_even_immediate() {
        let r = factor(22, Some(0), 4);
        assert_eq!(r.factors, Some((2, 11)));
        assert_eq!(r.iterations, 0);
    }

    #[test]
    fn factor_prime_power() {
        let r = factor(9, Some(0), 4);
        assert_eq!(r.factors, Some((3, 3)));
    }
}
