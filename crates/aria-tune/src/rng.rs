// SPDX-License-Identifier: Apache-2.0
//! SplitMix64 — the one source of randomness in this crate.
//!
//! Every stochastic decision (sampler proposals, tie-breaks) draws from a
//! stream seeded explicitly by the caller, so a study is reproducible
//! bit-for-bit from its seed. No `rand`, no thread-local state, no global
//! seeding that another part of the process could disturb.

/// A SplitMix64 generator.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `0..n` (`0` when `n == 0`).
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// Sample an index from a normalised categorical distribution.
    pub fn categorical(&mut self, probs: &[f64]) -> usize {
        let u = self.unit();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if u <= acc {
                return i;
            }
        }
        probs.len().saturating_sub(1)
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_is_pinned() {
        // A change here silently moves every study; pin the first draw.
        let mut r = Rng::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn unit_is_in_range_and_reproducible() {
        let draws = |seed| {
            let mut r = Rng::new(seed);
            (0..100).map(|_| r.unit()).collect::<Vec<_>>()
        };
        let a = draws(42);
        assert_eq!(a, draws(42));
        assert_ne!(a, draws(43));
        assert!(a.iter().all(|v| (0.0..1.0).contains(v)));
    }

    #[test]
    fn below_covers_its_range() {
        let mut r = Rng::new(7);
        let mut seen = [false; 5];
        for _ in 0..200 {
            seen[r.below(5)] = true;
        }
        assert!(seen.iter().all(|s| *s));
        assert_eq!(r.below(0), 0);
    }

    #[test]
    fn categorical_respects_weights() {
        let mut r = Rng::new(3);
        let probs = [0.0, 1.0, 0.0];
        for _ in 0..50 {
            assert_eq!(r.categorical(&probs), 1);
        }
    }

    #[test]
    fn normal_has_roughly_zero_mean_unit_variance() {
        let mut r = Rng::new(11);
        let xs: Vec<f64> = (0..20_000).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
    }
}
