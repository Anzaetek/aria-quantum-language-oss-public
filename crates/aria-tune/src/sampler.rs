// SPDX-License-Identifier: Apache-2.0
//! Samplers: given the trials so far, propose the next point of the grid.

use crate::rng::Rng;
use crate::space::Space;

/// Whether higher scores are better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Maximize,
    Minimize,
}

impl Direction {
    /// Is `a` at least as good as `b`?
    pub fn at_least_as_good(self, a: f64, b: f64) -> bool {
        match self {
            Direction::Maximize => a >= b,
            Direction::Minimize => a <= b,
        }
    }
}

/// A completed observation: the grid point and its final score.
pub type Observation = (Vec<usize>, f64);

/// Proposes the next grid point.
pub trait Sampler {
    fn suggest(
        &mut self,
        space: &Space,
        history: &[Observation],
        direction: Direction,
    ) -> Vec<usize>;
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Random
// ---------------------------------------------------------------------------

/// Uniform over the grid — the baseline every other sampler must beat.
#[derive(Debug, Clone)]
pub struct RandomSampler {
    rng: Rng,
}

impl RandomSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed ^ 0x5EED_5EED_5EED_5EED),
        }
    }
}

impl Sampler for RandomSampler {
    fn suggest(&mut self, space: &Space, _h: &[Observation], _d: Direction) -> Vec<usize> {
        space
            .dims()
            .into_iter()
            .map(|k| self.rng.below(k))
            .collect()
    }
    fn name(&self) -> &'static str {
        "random"
    }
}

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

/// Exhaustive enumeration in row-major order, wrapping once exhausted.
#[derive(Debug, Clone, Default)]
pub struct GridSampler {
    next: usize,
}

impl GridSampler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode a linear index into a grid point (last dimension varies fastest).
    fn decode(dims: &[usize], mut n: usize) -> Vec<usize> {
        let mut out = vec![0usize; dims.len()];
        for d in (0..dims.len()).rev() {
            let k = dims[d].max(1);
            out[d] = n % k;
            n /= k;
        }
        out
    }
}

impl Sampler for GridSampler {
    fn suggest(&mut self, space: &Space, _h: &[Observation], _d: Direction) -> Vec<usize> {
        let dims = space.dims();
        let total = space.size().max(1);
        let combo = Self::decode(&dims, self.next % total);
        self.next = self.next.wrapping_add(1);
        combo
    }
    fn name(&self) -> &'static str {
        "grid"
    }
}

// ---------------------------------------------------------------------------
// TPE
// ---------------------------------------------------------------------------

/// Tree-structured Parzen Estimator, categorical form (Bergstra et al. 2011;
/// Optuna's categorical variant).
///
/// 1. Split the observations into "good" (best `gamma` fraction) and "bad".
/// 2. Per dimension, build Laplace-smoothed categorical densities `l` (good)
///    and `g` (bad) over the grid indices.
/// 3. Draw `n_candidates` points from `l` and keep the one maximising the
///    expected-improvement proxy `Σ_d [ln l_d(v) − ln g_d(v)]`.
/// 4. The first `n_startup` trials are uniform random.
///
/// Because [`Space`] discretises *every* dimension to an index grid, this one
/// categorical implementation also covers the int / float / log-float
/// dimensions — no per-type kernel-density code.
#[derive(Debug, Clone)]
pub struct TpeSampler {
    pub gamma: f64,
    pub n_candidates: usize,
    pub n_startup: usize,
    pub prior_weight: f64,
    rng: Rng,
}

impl TpeSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            gamma: 0.25,
            n_candidates: 24,
            n_startup: 5,
            prior_weight: 1.0,
            rng: Rng::new(seed ^ 0x7BE5_7BE5_7BE5_7BE5),
        }
    }

    pub fn with_params(mut self, gamma: f64, n_candidates: usize, n_startup: usize) -> Self {
        self.gamma = gamma;
        self.n_candidates = n_candidates;
        self.n_startup = n_startup;
        self
    }

    /// Laplace-smoothed density over dimension `d` for the given rows.
    fn density(&self, rows: &[&Vec<usize>], d: usize, k: usize) -> Vec<f64> {
        let mut counts = vec![self.prior_weight; k];
        for r in rows {
            if let Some(&i) = r.get(d) {
                counts[i.min(k - 1)] += 1.0;
            }
        }
        let total: f64 = counts.iter().sum();
        counts.iter().map(|c| c / total).collect()
    }
}

impl Sampler for TpeSampler {
    fn suggest(
        &mut self,
        space: &Space,
        history: &[Observation],
        direction: Direction,
    ) -> Vec<usize> {
        let dims = space.dims();
        let random = |rng: &mut Rng| -> Vec<usize> { dims.iter().map(|&k| rng.below(k)).collect() };
        // Need at least one point on each side of the split.
        if history.len() < self.n_startup.max(2) {
            return random(&mut self.rng);
        }

        let mut order: Vec<usize> = (0..history.len()).collect();
        order.sort_by(|&a, &b| {
            let (sa, sb) = (history[a].1, history[b].1);
            match direction {
                Direction::Maximize => sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal),
                Direction::Minimize => sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        let n_good = ((history.len() as f64) * self.gamma).ceil().max(1.0) as usize;
        let n_good = n_good.min(history.len() - 1).max(1);
        let good: Vec<&Vec<usize>> = order[..n_good].iter().map(|&i| &history[i].0).collect();
        let bad: Vec<&Vec<usize>> = order[n_good..].iter().map(|&i| &history[i].0).collect();

        let l: Vec<Vec<f64>> = dims
            .iter()
            .enumerate()
            .map(|(d, &k)| self.density(&good, d, k))
            .collect();
        let g: Vec<Vec<f64>> = dims
            .iter()
            .enumerate()
            .map(|(d, &k)| self.density(&bad, d, k))
            .collect();

        let mut best: Option<(f64, Vec<usize>)> = None;
        for _ in 0..self.n_candidates.max(1) {
            let cand: Vec<usize> = (0..dims.len())
                .map(|d| self.rng.categorical(&l[d]))
                .collect();
            let ei: f64 = (0..dims.len())
                .map(|d| l[d][cand[d]].max(1e-12).ln() - g[d][cand[d]].max(1e-12).ln())
                .sum();
            if best.as_ref().is_none_or(|(b, _)| ei > *b) {
                best = Some((ei, cand));
            }
        }
        best.map(|(_, c)| c)
            .unwrap_or_else(|| random(&mut self.rng))
    }

    fn name(&self) -> &'static str {
        "tpe"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space_1d(k: usize) -> Space {
        Space::new().int("x", 0, k as i64 - 1, 1)
    }

    /// Noisy quadratic bandit: score peaks at `opt` and decays quadratically.
    fn bandit(x: usize, opt: f64, rng: &mut Rng) -> f64 {
        let d = x as f64 - opt;
        -(d * d) + 0.5 * rng.normal()
    }

    /// Run `n_trials` of `sampler` on the bandit and return the best score.
    fn run(sampler: &mut dyn Sampler, k: usize, n_trials: usize, seed: u64) -> f64 {
        let space = space_1d(k);
        let mut noise = Rng::new(seed);
        let mut history: Vec<Observation> = Vec::new();
        let mut best = f64::MIN;
        for _ in 0..n_trials {
            let c = sampler.suggest(&space, &history, Direction::Maximize);
            let s = bandit(c[0], 14.0, &mut noise);
            best = best.max(s);
            history.push((c, s));
        }
        best
    }

    #[test]
    fn tpe_beats_random_on_a_noisy_quadratic_bandit() {
        // Accept: TPE ≥ Random after 40 trials. Averaged over several seeds so
        // the gate measures the sampler, not one lucky stream.
        let (mut tpe_total, mut rnd_total) = (0.0, 0.0);
        for seed in 0..8u64 {
            tpe_total += run(&mut TpeSampler::new(seed), 20, 40, 1000 + seed);
            rnd_total += run(&mut RandomSampler::new(seed), 20, 40, 1000 + seed);
        }
        let (tpe, rnd) = (tpe_total / 8.0, rnd_total / 8.0);
        assert!(tpe >= rnd, "TPE mean best {tpe} < random mean best {rnd}");
    }

    #[test]
    fn tpe_concentrates_near_the_optimum() {
        // Beyond "≥ random": TPE's later proposals should cluster on the peak.
        let space = space_1d(20);
        let mut sampler = TpeSampler::new(5).with_params(0.25, 32, 5);
        let mut noise = Rng::new(77);
        let mut history: Vec<Observation> = Vec::new();
        let mut late = Vec::new();
        for i in 0..60 {
            let c = sampler.suggest(&space, &history, Direction::Maximize);
            if i >= 40 {
                late.push(c[0] as f64);
            }
            let s = bandit(c[0], 14.0, &mut noise);
            history.push((c, s));
        }
        let mean = late.iter().sum::<f64>() / late.len() as f64;
        assert!(
            (mean - 14.0).abs() < 5.0,
            "late proposals centred at {mean}"
        );
    }

    #[test]
    fn tpe_is_deterministic_for_a_seed() {
        let seq = |seed| {
            let space = space_1d(8);
            let mut s = TpeSampler::new(seed);
            let mut h: Vec<Observation> = Vec::new();
            let mut out = Vec::new();
            for i in 0..20 {
                let c = s.suggest(&space, &h, Direction::Maximize);
                out.push(c.clone());
                h.push((c, i as f64 % 3.0));
            }
            out
        };
        assert_eq!(seq(1), seq(1));
        assert_ne!(seq(1), seq(2));
    }

    #[test]
    fn tpe_respects_direction() {
        // Minimising, the good set is the LOW scores; proposals should drift
        // toward the low-index region where we placed them.
        let space = space_1d(10);
        let mut s = TpeSampler::new(9).with_params(0.3, 32, 4);
        let mut h: Vec<Observation> = Vec::new();
        // Seed history: low index ⇒ low score.
        for i in 0..10usize {
            h.push((vec![i], i as f64));
        }
        let mut sum = 0.0;
        for _ in 0..40 {
            sum += s.suggest(&space, &h, Direction::Minimize)[0] as f64;
        }
        assert!(sum / 40.0 < 4.5, "minimising drifted high: {}", sum / 40.0);
    }

    #[test]
    fn grid_enumerates_every_point_then_wraps() {
        let space = Space::new().int("a", 0, 2, 1).categorical("b", &["x", "y"]);
        let mut g = GridSampler::new();
        let seen: Vec<Vec<usize>> = (0..6)
            .map(|_| g.suggest(&space, &[], Direction::Maximize))
            .collect();
        assert_eq!(
            seen,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![1, 0],
                vec![1, 1],
                vec![2, 0],
                vec![2, 1]
            ]
        );
        // Wraps back to the start.
        assert_eq!(g.suggest(&space, &[], Direction::Maximize), vec![0, 0]);
    }

    #[test]
    fn random_stays_inside_the_grid() {
        let space = Space::new().int("a", 0, 3, 1).float("b", 0.0, 1.0, 5);
        let mut r = RandomSampler::new(4);
        for _ in 0..200 {
            let c = r.suggest(&space, &[], Direction::Maximize);
            assert!(c[0] < 4 && c[1] < 5, "{c:?} escaped the grid");
        }
    }

    #[test]
    fn samplers_handle_an_empty_space() {
        let space = Space::new();
        assert!(RandomSampler::new(1)
            .suggest(&space, &[], Direction::Maximize)
            .is_empty());
        assert!(GridSampler::new()
            .suggest(&space, &[], Direction::Maximize)
            .is_empty());
        assert!(TpeSampler::new(1)
            .suggest(&space, &[], Direction::Maximize)
            .is_empty());
    }
}
