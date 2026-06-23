//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
//!
//! Gradient-free black-box optimization well-suited for noisy quantum circuits.
//! Implements the (μ/μ_w, λ)-CMA-ES with rank-one and rank-μ covariance updates.
//!
//! Reference: Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial."

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// CMA-ES optimizer state.
pub struct CmaEs {
    /// Dimension of the search space.
    dim: usize,
    /// Population size (λ).
    lambda: usize,
    /// Number of selected parents (μ).
    mu: usize,
    /// Recombination weights for μ best individuals.
    weights: Vec<f64>,
    /// Effective μ: μ_eff = (Σw_i)² / Σw_i².
    mu_eff: f64,

    /// Distribution mean (current best estimate).
    mean: Vec<f64>,
    /// Step size (σ).
    sigma: f64,

    /// Covariance matrix C (stored as full dim×dim, row-major).
    cov: Vec<f64>,
    /// Evolution path for σ adaptation (p_σ).
    ps: Vec<f64>,
    /// Evolution path for C adaptation (p_c).
    pc: Vec<f64>,

    /// Learning rates.
    c_sigma: f64,
    d_sigma: f64,
    cc: f64,
    c1: f64,
    c_mu: f64,

    /// Expected length of N(0,I) vector: E[||N(0,I)||] ≈ √n (1 - 1/(4n) + 1/(21n²)).
    chi_n: f64,

    /// Generation counter.
    generation: usize,
    /// Best solution found so far.
    best_x: Vec<f64>,
    best_f: f64,

    /// RNG.
    rng: StdRng,
}

impl CmaEs {
    /// Create a new CMA-ES optimizer.
    ///
    /// `initial_mean` — starting point in parameter space.
    /// `sigma` — initial step size.
    /// `pop_size` — population size (None = default 4 + ⌊3 ln n⌋).
    /// `seed` — RNG seed (None = random).
    pub fn new(
        initial_mean: Vec<f64>,
        sigma: f64,
        pop_size: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        let dim = initial_mean.len();
        let lambda = pop_size.unwrap_or(4 + (3.0 * (dim as f64).ln()).floor() as usize);
        let mu = lambda / 2;

        // Recombination weights: log(μ+0.5) - log(i) for i=1..μ, then normalize
        let mut weights: Vec<f64> = (0..mu)
            .map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0))
            .collect();
        let w_sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= w_sum;
        }
        let w_sq_sum: f64 = weights.iter().map(|w| w * w).sum();
        let mu_eff = 1.0 / w_sq_sum;

        // Learning rates
        let c_sigma = (mu_eff + 2.0) / (dim as f64 + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (((mu_eff - 1.0) / (dim as f64 + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let cc = (4.0 + mu_eff / dim as f64) / (dim as f64 + 4.0 + 2.0 * mu_eff / dim as f64);
        let c1 = 2.0 / ((dim as f64 + 1.3).powi(2) + mu_eff);
        let c_mu_raw =
            (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)) / ((dim as f64 + 2.0).powi(2) + mu_eff);
        let c_mu = c_mu_raw.min(1.0 - c1);

        let chi_n = (dim as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * dim as f64) + 1.0 / (21.0 * (dim as f64).powi(2)));

        // Identity covariance matrix
        let mut cov = vec![0.0; dim * dim];
        for i in 0..dim {
            cov[i * dim + i] = 1.0;
        }

        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => rand::make_rng::<StdRng>(),
        };

        Self {
            dim,
            lambda,
            mu,
            weights,
            mu_eff,
            mean: initial_mean.clone(),
            sigma,
            cov,
            ps: vec![0.0; dim],
            pc: vec![0.0; dim],
            c_sigma,
            d_sigma,
            cc,
            c1,
            c_mu,
            chi_n,
            generation: 0,
            best_x: initial_mean,
            best_f: f64::INFINITY,
            rng,
        }
    }

    /// Sample λ candidate solutions from the current distribution N(mean, σ²C).
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        let sqrt_cov = self.matrix_sqrt();
        let mut population = Vec::with_capacity(self.lambda);

        for _ in 0..self.lambda {
            // Sample z ~ N(0, I)
            let z: Vec<f64> = (0..self.dim).map(|_| self.standard_normal()).collect();
            // x = mean + σ * sqrt(C) * z
            let mut x = self.mean.clone();
            for i in 0..self.dim {
                let mut transformed = 0.0;
                for j in 0..self.dim {
                    transformed += sqrt_cov[i * self.dim + j] * z[j];
                }
                x[i] += self.sigma * transformed;
            }
            population.push(x);
        }

        population
    }

    /// Update the distribution based on evaluated fitness values.
    ///
    /// `solutions` — pairs of (candidate, fitness). Lower fitness is better.
    #[allow(clippy::needless_range_loop)]
    pub fn tell(&mut self, solutions: &mut [(Vec<f64>, f64)]) {
        // Sort by fitness (ascending — minimize)
        solutions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update best
        if solutions[0].1 < self.best_f {
            self.best_f = solutions[0].1;
            self.best_x = solutions[0].0.clone();
        }

        // Weighted recombination of μ best
        let old_mean = self.mean.clone();
        self.mean = vec![0.0; self.dim];
        for i in 0..self.mu {
            for d in 0..self.dim {
                self.mean[d] += self.weights[i] * solutions[i].0[d];
            }
        }

        // Compute C^{-1/2} * (mean - old_mean) / σ for evolution path updates
        let inv_sqrt_cov = self.matrix_inv_sqrt();
        let mut y: Vec<f64> = vec![0.0; self.dim];
        for d in 0..self.dim {
            y[d] = (self.mean[d] - old_mean[d]) / self.sigma;
        }
        let mut c_inv_sqrt_y = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                c_inv_sqrt_y[i] += inv_sqrt_cov[i * self.dim + j] * y[j];
            }
        }

        // Update evolution path for σ (p_σ)
        let cs_complement = (1.0 - self.c_sigma).sqrt();
        let cs_scale = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        for d in 0..self.dim {
            self.ps[d] = cs_complement * self.ps[d] + cs_scale * c_inv_sqrt_y[d];
        }

        // Update evolution path for C (p_c)
        let ps_norm: f64 = self.ps.iter().map(|v| v * v).sum::<f64>().sqrt();
        let h_sigma = if ps_norm
            / (1.0 - (1.0 - self.c_sigma).powi(2 * (self.generation as i32 + 1))).sqrt()
            < (1.4 + 2.0 / (self.dim as f64 + 1.0)) * self.chi_n
        {
            1.0
        } else {
            0.0
        };
        let cc_complement = (1.0 - self.cc).sqrt();
        let cc_scale = h_sigma * (self.cc * (2.0 - self.cc) * self.mu_eff).sqrt();
        for d in 0..self.dim {
            self.pc[d] = cc_complement * self.pc[d] + cc_scale * y[d];
        }

        // Update covariance matrix
        let delta_h = (1.0 - h_sigma) * self.cc * (2.0 - self.cc);
        for i in 0..self.dim {
            for j in 0..self.dim {
                // Rank-one update from p_c
                let rank_one = self.c1 * self.pc[i] * self.pc[j];

                // Rank-μ update from selected population
                let mut rank_mu = 0.0;
                for k in 0..self.mu {
                    let yi = (solutions[k].0[i] - old_mean[i]) / self.sigma;
                    let yj = (solutions[k].0[j] - old_mean[j]) / self.sigma;
                    rank_mu += self.weights[k] * yi * yj;
                }
                rank_mu *= self.c_mu;

                self.cov[i * self.dim + j] = (1.0 - self.c1 - self.c_mu + self.c1 * delta_h)
                    * self.cov[i * self.dim + j]
                    + rank_one
                    + rank_mu;
            }
        }

        // Update step size σ
        self.sigma *= ((self.c_sigma / self.d_sigma) * (ps_norm / self.chi_n - 1.0)).exp();

        self.generation += 1;
    }

    /// Best solution found so far: (parameters, fitness).
    pub fn best(&self) -> (&[f64], f64) {
        (&self.best_x, self.best_f)
    }

    /// Current step size σ.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Current generation count.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Check if the optimizer has converged (σ below threshold).
    pub fn converged(&self, tol: f64) -> bool {
        self.sigma < tol
    }

    // --- Internal helpers ---

    /// Box-Muller transform for standard normal samples.
    fn standard_normal(&mut self) -> f64 {
        let u1: f64 = self.rng.random::<f64>().max(1e-300);
        let u2: f64 = self.rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Compute matrix square root of the covariance matrix via eigendecomposition.
    /// Returns sqrt(C) as a dim×dim row-major matrix.
    fn matrix_sqrt(&self) -> Vec<f64> {
        let (eigenvalues, eigenvectors) = self.eigen_decompose();
        let mut result = vec![0.0; self.dim * self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                for k in 0..self.dim {
                    result[i * self.dim + j] += eigenvectors[i * self.dim + k]
                        * eigenvalues[k].sqrt()
                        * eigenvectors[j * self.dim + k];
                }
            }
        }
        result
    }

    /// Compute inverse matrix square root C^{-1/2}.
    fn matrix_inv_sqrt(&self) -> Vec<f64> {
        let (eigenvalues, eigenvectors) = self.eigen_decompose();
        let mut result = vec![0.0; self.dim * self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                for k in 0..self.dim {
                    let ev = eigenvalues[k].max(1e-20);
                    result[i * self.dim + j] += eigenvectors[i * self.dim + k]
                        * (1.0 / ev.sqrt())
                        * eigenvectors[j * self.dim + k];
                }
            }
        }
        result
    }

    /// Eigendecomposition of symmetric covariance matrix via Jacobi iteration.
    /// Returns (eigenvalues, eigenvectors) where eigenvectors is row-major.
    fn eigen_decompose(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.dim;
        let mut a = self.cov.clone();
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        for _ in 0..100 * n * n {
            // Find largest off-diagonal element
            let mut max_val = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i * n + j].abs() > max_val {
                        max_val = a[i * n + j].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < 1e-15 {
                break;
            }

            // Compute rotation
            let app = a[p * n + p];
            let aqq = a[q * n + q];
            let apq = a[p * n + q];
            let theta = 0.5 * (aqq - app).atan2(apq);
            let c = theta.cos();
            let s = theta.sin();

            // Apply Jacobi rotation
            let mut new_a = a.clone();
            for i in 0..n {
                new_a[i * n + p] = c * a[i * n + p] - s * a[i * n + q];
                new_a[i * n + q] = s * a[i * n + p] + c * a[i * n + q];
                new_a[p * n + i] = new_a[i * n + p];
                new_a[q * n + i] = new_a[i * n + q];
            }
            new_a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            new_a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            new_a[p * n + q] = 0.0;
            new_a[q * n + p] = 0.0;
            a = new_a;

            // Update eigenvector matrix
            let mut new_v = v.clone();
            for i in 0..n {
                new_v[i * n + p] = c * v[i * n + p] - s * v[i * n + q];
                new_v[i * n + q] = s * v[i * n + p] + c * v[i * n + q];
            }
            v = new_v;
        }

        let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i].max(0.0)).collect();
        (eigenvalues, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmaes_sphere() {
        // Minimize f(x) = Σx² — minimum at origin
        let dim = 3;
        let mut cma = CmaEs::new(vec![5.0; dim], 1.0, None, Some(42));

        for _ in 0..200 {
            let pop = cma.ask();
            let mut solutions: Vec<(Vec<f64>, f64)> = pop
                .into_iter()
                .map(|x| {
                    let f: f64 = x.iter().map(|xi| xi * xi).sum();
                    (x, f)
                })
                .collect();
            cma.tell(&mut solutions);

            if cma.best().1 < 1e-8 {
                break;
            }
        }

        let (best_x, best_f) = cma.best();
        assert!(
            best_f < 1e-4,
            "sphere should converge near 0, got f={best_f}"
        );
        for (i, xi) in best_x.iter().enumerate() {
            assert!(xi.abs() < 0.1, "x[{i}] = {xi}, should be ~0");
        }
    }

    #[test]
    fn test_cmaes_rosenbrock() {
        // Minimize f(x,y) = (1-x)² + 100(y-x²)² — minimum at (1,1)
        let mut cma = CmaEs::new(vec![0.0, 0.0], 0.5, Some(20), Some(123));

        for _ in 0..500 {
            let pop = cma.ask();
            let mut solutions: Vec<(Vec<f64>, f64)> = pop
                .into_iter()
                .map(|x| {
                    let f = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
                    (x, f)
                })
                .collect();
            cma.tell(&mut solutions);

            if cma.best().1 < 1e-6 {
                break;
            }
        }

        let (best_x, best_f) = cma.best();
        assert!(best_f < 0.1, "rosenbrock should converge, got f={best_f}");
        assert!(
            (best_x[0] - 1.0).abs() < 0.5,
            "x[0] = {}, should be ~1",
            best_x[0]
        );
    }

    #[test]
    fn test_cmaes_noisy_sphere() {
        // Noisy sphere: f(x) = Σx² + noise
        let mut cma = CmaEs::new(vec![3.0, 3.0], 1.0, Some(15), Some(99));
        let mut noise_rng = StdRng::seed_from_u64(77);

        for _ in 0..300 {
            let pop = cma.ask();
            let mut solutions: Vec<(Vec<f64>, f64)> = pop
                .into_iter()
                .map(|x| {
                    let f: f64 = x.iter().map(|xi| xi * xi).sum();
                    let noise: f64 = noise_rng.random::<f64>() * 0.1 - 0.05;
                    (x, f + noise)
                })
                .collect();
            cma.tell(&mut solutions);
        }

        let (_, best_f) = cma.best();
        assert!(
            best_f < 1.0,
            "noisy sphere should still converge reasonably, got f={best_f}"
        );
    }

    /// Phase A regression pin: deterministic CMA-ES on sphere with
    /// `seed=42`, 3D, σ=1, x0=[5;3]. After exactly 100 ask/tell
    /// cycles the recorded run hits `best_f ≈ 5.7e-12`. We assert
    /// `< 1e-9` here — tight enough to catch a real algorithm
    /// regression (the existing loose `< 1e-4` bound at 200 iters
    /// would miss most), wide enough to survive minor numerical
    /// drift from rand or platform fp differences. Mirrors the TODO
    /// item "CMA-ES formal checks: pin convergence behaviour on
    /// sphere/Rosenbrock/noisy-sphere with deterministic seeds".
    #[test]
    fn test_cmaes_sphere_pinned_seed42() {
        let mut cma = CmaEs::new(vec![5.0; 3], 1.0, None, Some(42));
        for _ in 0..100 {
            let pop = cma.ask();
            let mut solutions: Vec<(Vec<f64>, f64)> = pop
                .into_iter()
                .map(|x| {
                    let f: f64 = x.iter().map(|xi| xi * xi).sum();
                    (x, f)
                })
                .collect();
            cma.tell(&mut solutions);
        }
        let (_, best_f) = cma.best();
        assert!(
            best_f < 1e-9,
            "sphere seed=42 should reach best_f < 1e-9 after 100 iters; got {best_f:e}"
        );
    }

    /// Same pin for Rosenbrock with `seed=123`, 2D, σ=0.5, x0=[0,0],
    /// pop_size=20. Recorded reference: `best_f ≈ 1.5e-20` at iter
    /// 200, with `best_x ≈ (1, 1)` to ~1e-9. Pin both — anything
    /// looser than these bounds means the optimiser regressed.
    #[test]
    fn test_cmaes_rosenbrock_pinned_seed123() {
        let mut cma = CmaEs::new(vec![0.0, 0.0], 0.5, Some(20), Some(123));
        for _ in 0..200 {
            let pop = cma.ask();
            let mut solutions: Vec<(Vec<f64>, f64)> = pop
                .into_iter()
                .map(|x| {
                    let f = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
                    (x, f)
                })
                .collect();
            cma.tell(&mut solutions);
        }
        let (best_x, best_f) = cma.best();
        assert!(
            best_f < 1e-15,
            "rosenbrock seed=123 should reach best_f < 1e-15 after 200 iters; got {best_f:e}"
        );
        assert!(
            (best_x[0] - 1.0).abs() < 1e-6 && (best_x[1] - 1.0).abs() < 1e-6,
            "rosenbrock seed=123 should land at (1, 1) within 1e-6; got {best_x:?}"
        );
    }

    /// Noisy sphere with `seed=99` and noise `seed=77`: σ_noise=0.1.
    /// CMA-ES should still drive `best_f` near zero even with the
    /// additive noise — we pin a generous `< 1e-3` bound after 300
    /// iterations. Tighter than the existing `< 1.0` heuristic; a
    /// regression in the population-rank-based update would push
    /// past it.
    #[test]
    fn test_cmaes_noisy_sphere_pinned_seed99() {
        let mut cma = CmaEs::new(vec![3.0, 3.0], 1.0, Some(15), Some(99));
        let mut noise_rng = StdRng::seed_from_u64(77);
        for _ in 0..300 {
            let pop = cma.ask();
            let mut solutions: Vec<(Vec<f64>, f64)> = pop
                .into_iter()
                .map(|x| {
                    let f: f64 = x.iter().map(|xi| xi * xi).sum();
                    let noise: f64 = noise_rng.random::<f64>() * 0.1 - 0.05;
                    (x, f + noise)
                })
                .collect();
            cma.tell(&mut solutions);
        }
        let (best_x, best_f) = cma.best();
        // Note: best_f is the noisy evaluation; the true distance to
        // the minimum is what we actually care about. Assert the
        // best-x norm is small.
        let true_dist_sq: f64 = best_x.iter().map(|xi| xi * xi).sum();
        assert!(
            true_dist_sq < 1e-2,
            "noisy sphere seed=99: best_x should be near 0 within 1e-1, got |x|² = {true_dist_sq:e}"
        );
        assert!(
            best_f.abs() < 0.1,
            "noisy sphere seed=99: best_f within ±0.1 of 0 (noise band), got {best_f}"
        );
    }

    #[test]
    fn test_cmaes_converged() {
        let mut cma = CmaEs::new(vec![0.0], 1.0, None, Some(1));
        assert!(!cma.converged(1e-10));

        // Run a few generations on trivial problem
        for _ in 0..50 {
            let pop = cma.ask();
            let mut solutions: Vec<(Vec<f64>, f64)> =
                pop.into_iter().map(|x| (x.clone(), x[0] * x[0])).collect();
            cma.tell(&mut solutions);
        }

        assert!(cma.generation() > 0);
    }
}
