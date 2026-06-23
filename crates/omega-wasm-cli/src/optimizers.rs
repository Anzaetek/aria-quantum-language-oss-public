//! Native optimizers for the `--optimizer NAME` flag.
//!
//! These run inside the host process, not inside the WASM guest, so they
//! sidestep the guest's hard-coded `NUM_PARAMS = 2`. The optimization loop
//! calls a `cost` closure (and, for gradient-based methods, a `grad`
//! closure) that wraps `Backend::expectation` and `compute_gradient` from
//! omega-core, so any number of free parameters works as long as the
//! circuit and statevector simulator can handle it.

use omega_core::cmaes::CmaEs;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizerKind {
    /// Run the shipped vqe.wasm / qaoa.wasm guest in gradient-descent mode.
    WasmGd,
    /// Run the shipped vqe.wasm / qaoa.wasm guest in Adam mode (the guest
    /// picks the optimizer from its input JSON; this just sets the field).
    WasmAdam,
    /// Native plain gradient descent (fixed step).
    Gd,
    /// Native Adam (adaptive moments).
    Adam,
    /// Native CMA-ES (gradient-free).
    CmaEs,
}

impl OptimizerKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "wasm-gd" => Some(Self::WasmGd),
            "wasm-adam" => Some(Self::WasmAdam),
            "gd" => Some(Self::Gd),
            "adam" => Some(Self::Adam),
            "cmaes" | "cma-es" => Some(Self::CmaEs),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WasmGd => "wasm-gd",
            Self::WasmAdam => "wasm-adam",
            Self::Gd => "gd",
            Self::Adam => "adam",
            Self::CmaEs => "cmaes",
        }
    }

    /// True when the optimizer runs natively (no WASM guest).
    #[allow(dead_code)]
    pub fn is_native(&self) -> bool {
        !matches!(self, Self::WasmGd | Self::WasmAdam)
    }

    /// True when the optimizer is a WASM guest (any mode).
    pub fn is_wasm(&self) -> bool {
        matches!(self, Self::WasmGd | Self::WasmAdam)
    }

    /// In-guest optimizer name passed to the WASM guest via input JSON
    /// (`"gd"` or `"adam"`). For native variants this is unused.
    pub fn guest_optimizer_name(&self) -> &'static str {
        match self {
            Self::WasmGd => "gd",
            Self::WasmAdam => "adam",
            _ => "gd",
        }
    }

    /// True when the optimizer needs gradient evaluations.
    pub fn needs_gradient(&self) -> bool {
        matches!(self, Self::Gd | Self::Adam)
    }
}

#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub optimal_params: Vec<f64>,
    pub optimal_value: f64,
    pub iterations: usize,
    /// (iteration, value) — one entry per cost evaluation. For population
    /// methods like CMA-ES, multiple entries share an iteration index.
    pub progress: Vec<(u32, f64)>,
}

/// Plain gradient descent: x ← x - lr · ∇f(x).
///
/// Tracks the best-seen iterate, since fixed-step GD can overshoot.
pub fn run_gd<F, G>(
    initial: Vec<f64>,
    cost: F,
    grad: G,
    max_iters: usize,
    learning_rate: f64,
    convergence_tol: f64,
) -> OptimizationResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let mut x = initial;
    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;
    let mut prev_f = f64::INFINITY;
    let mut progress = Vec::with_capacity(max_iters);
    let mut iterations = 0;

    for t in 0..max_iters {
        let f = cost(&x);
        progress.push((t as u32, f));
        if f < best_f {
            best_f = f;
            best_x = x.clone();
        }
        iterations = t + 1;
        if t > 0 && (prev_f - f).abs() < convergence_tol {
            break;
        }
        prev_f = f;

        let g = grad(&x);
        for (i, xi) in x.iter_mut().enumerate() {
            *xi -= learning_rate * g.get(i).copied().unwrap_or(0.0);
        }
    }

    OptimizationResult {
        optimal_params: best_x,
        optimal_value: best_f,
        iterations,
        progress,
    }
}

/// Adam (Kingma & Ba, 2014). Defaults β1=0.9, β2=0.999, ε=1e-8 — the
/// values from the original paper, which work well across most problems.
pub fn run_adam<F, G>(
    initial: Vec<f64>,
    cost: F,
    grad: G,
    max_iters: usize,
    learning_rate: f64,
    convergence_tol: f64,
) -> OptimizationResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const EPS: f64 = 1e-8;

    let mut x = initial;
    let dim = x.len();
    let mut m = vec![0.0; dim];
    let mut v = vec![0.0; dim];
    let mut best_x = x.clone();
    let mut best_f = f64::INFINITY;
    let mut prev_f = f64::INFINITY;
    let mut progress = Vec::with_capacity(max_iters);
    let mut iterations = 0;

    for t in 1..=max_iters {
        let f = cost(&x);
        progress.push(((t - 1) as u32, f));
        if f < best_f {
            best_f = f;
            best_x = x.clone();
        }
        iterations = t;
        if t > 1 && (prev_f - f).abs() < convergence_tol {
            break;
        }
        prev_f = f;

        let g = grad(&x);
        let bc1 = 1.0 - BETA1.powi(t as i32);
        let bc2 = 1.0 - BETA2.powi(t as i32);
        for i in 0..dim {
            let gi = g.get(i).copied().unwrap_or(0.0);
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * gi;
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * gi * gi;
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            x[i] -= learning_rate * m_hat / (v_hat.sqrt() + EPS);
        }
    }

    OptimizationResult {
        optimal_params: best_x,
        optimal_value: best_f,
        iterations,
        progress,
    }
}

/// CMA-ES wrapper — gradient-free, population-based. Useful when the
/// landscape is noisy or the gradient is unavailable / too slow.
///
/// `max_iters` here counts CMA-ES *generations*; each generation evaluates
/// `pop_size` candidates, so the total cost-eval count is ≈ `max_iters *
/// pop_size`. Default pop_size from omega-core is `4 + ⌊3 ln n⌋`.
pub fn run_cmaes<F>(
    initial: Vec<f64>,
    cost: F,
    max_iters: usize,
    initial_sigma: f64,
    seed: Option<u64>,
) -> OptimizationResult
where
    F: Fn(&[f64]) -> f64,
{
    let mut cmaes = CmaEs::new(initial.clone(), initial_sigma, None, seed);
    let mut progress = Vec::new();
    let mut best_x = initial;
    let mut best_f = f64::INFINITY;
    let mut iterations = 0;

    for gen in 0..max_iters {
        let candidates = cmaes.ask();
        let mut scored: Vec<(Vec<f64>, f64)> = candidates
            .into_iter()
            .map(|x| {
                let f = cost(&x);
                (x, f)
            })
            .collect();

        for (x, f) in &scored {
            progress.push((gen as u32, *f));
            if *f < best_f {
                best_f = *f;
                best_x = x.clone();
            }
        }

        cmaes.tell(&mut scored);
        iterations = gen + 1;
        if cmaes.converged(1e-8) {
            break;
        }
    }

    OptimizationResult {
        optimal_params: best_x,
        optimal_value: best_f,
        iterations,
        progress,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convex paraboloid f(x,y) = (x-1)² + (y+2)² with global min at (1,-2),
    /// f* = 0. Used as a sanity check: any reasonable optimizer should hit
    /// near-zero in a few hundred iterations.
    fn paraboloid(p: &[f64]) -> f64 {
        let dx = p[0] - 1.0;
        let dy = p[1] + 2.0;
        dx * dx + dy * dy
    }

    fn paraboloid_grad(p: &[f64]) -> Vec<f64> {
        vec![2.0 * (p[0] - 1.0), 2.0 * (p[1] + 2.0)]
    }

    #[test]
    fn gd_converges_on_paraboloid() {
        let r = run_gd(vec![5.0, 5.0], paraboloid, paraboloid_grad, 500, 0.1, 1e-12);
        assert!(r.optimal_value < 1e-6, "got {}", r.optimal_value);
        assert!((r.optimal_params[0] - 1.0).abs() < 1e-3);
        assert!((r.optimal_params[1] + 2.0).abs() < 1e-3);
    }

    #[test]
    fn adam_converges_on_paraboloid() {
        let r = run_adam(vec![5.0, 5.0], paraboloid, paraboloid_grad, 500, 0.1, 1e-12);
        assert!(r.optimal_value < 1e-3, "got {}", r.optimal_value);
    }

    #[test]
    fn cmaes_converges_on_paraboloid() {
        let r = run_cmaes(vec![5.0, 5.0], paraboloid, 200, 1.0, Some(42));
        assert!(r.optimal_value < 1e-3, "got {}", r.optimal_value);
        assert!((r.optimal_params[0] - 1.0).abs() < 0.1);
        assert!((r.optimal_params[1] + 2.0).abs() < 0.1);
    }

    #[test]
    fn adam_handles_high_dim() {
        // 10-D paraboloid: f(x) = Σ (x_i - i)². Min at x_i = i, f* = 0.
        let target: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let cost = |p: &[f64]| -> f64 {
            p.iter()
                .zip(target.iter())
                .map(|(&pi, &ti)| (pi - ti).powi(2))
                .sum()
        };
        let target2 = target.clone();
        let grad = move |p: &[f64]| -> Vec<f64> {
            p.iter()
                .zip(target2.iter())
                .map(|(&pi, &ti)| 2.0 * (pi - ti))
                .collect()
        };
        let r = run_adam(vec![0.0; 10], cost, grad, 2000, 0.5, 1e-15);
        assert!(r.optimal_value < 1e-3, "got {}", r.optimal_value);
    }

    #[test]
    fn cmaes_handles_high_dim() {
        // CMA-ES on a 5-D sphere (well-posed for a black-box method).
        let cost = |p: &[f64]| -> f64 { p.iter().map(|&x| x * x).sum() };
        let r = run_cmaes(vec![3.0; 5], cost, 200, 1.0, Some(7));
        assert!(r.optimal_value < 1e-3, "got {}", r.optimal_value);
    }

    #[test]
    fn optimizer_kind_parse_roundtrip() {
        for k in [
            OptimizerKind::WasmGd,
            OptimizerKind::Gd,
            OptimizerKind::Adam,
            OptimizerKind::CmaEs,
        ] {
            assert_eq!(OptimizerKind::parse(k.as_str()), Some(k));
        }
        // alias
        assert_eq!(OptimizerKind::parse("cma-es"), Some(OptimizerKind::CmaEs));
        assert_eq!(OptimizerKind::parse("nope"), None);
    }

    #[test]
    fn optimizer_kind_classifies_native_and_gradient_use() {
        assert!(!OptimizerKind::WasmGd.is_native());
        assert!(!OptimizerKind::WasmAdam.is_native());
        assert!(OptimizerKind::Gd.is_native());
        assert!(OptimizerKind::Adam.is_native());
        assert!(OptimizerKind::CmaEs.is_native());

        assert!(OptimizerKind::WasmGd.is_wasm());
        assert!(OptimizerKind::WasmAdam.is_wasm());
        assert!(!OptimizerKind::Gd.is_wasm());

        assert!(OptimizerKind::Gd.needs_gradient());
        assert!(OptimizerKind::Adam.needs_gradient());
        assert!(!OptimizerKind::CmaEs.needs_gradient());
        assert!(!OptimizerKind::WasmGd.needs_gradient());
    }

    #[test]
    fn guest_optimizer_name_picks_correct_string() {
        assert_eq!(OptimizerKind::WasmGd.guest_optimizer_name(), "gd");
        assert_eq!(OptimizerKind::WasmAdam.guest_optimizer_name(), "adam");
    }

    #[test]
    fn gd_records_progress_per_iteration() {
        let r = run_gd(vec![5.0, 5.0], paraboloid, paraboloid_grad, 50, 0.1, 1e-15);
        assert_eq!(r.progress.len(), r.iterations);
        // Cost should be monotonically non-increasing in best-seen, but the
        // raw stream can overshoot once before the convergence check fires.
        let final_val = r.progress.last().unwrap().1;
        assert!(final_val <= r.progress[0].1, "no improvement at all");
    }

    #[test]
    fn run_cmaes_short_run_makes_progress() {
        // Only 5 generations — should still beat the initial point.
        let initial = vec![5.0, 5.0];
        let f0 = paraboloid(&initial);
        let r = run_cmaes(initial, paraboloid, 5, 1.0, Some(123));
        assert!(
            r.optimal_value < f0,
            "no improvement: {} >= {}",
            r.optimal_value,
            f0
        );
    }
}
