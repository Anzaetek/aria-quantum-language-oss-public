// SPDX-License-Identifier: Apache-2.0
//! Pure-Rust variational training for Aria circuits.
//!
//! Minimizes the expectation value `⟨ψ(θ)|O|ψ(θ)⟩` of a Pauli observable over
//! the circuit's free (trainable) symbols by gradient descent, using
//! omega-core's parameter-shift gradients — **no libtorch**. This is the
//! engine behind `aria train` and the proof that Aria's `symbolic[k]`
//! parameters train end-to-end on the pure-Rust backend. The optional `tch`
//! plugin (Phase 4) accelerates this same loop with batched/GPU autograd.
//!
//! Layer-wise training (arXiv:2606.03517): `TrainConfig::frozen` names
//! symbols excluded from both the gradient evaluations and the update
//! step, and `TrainConfig::init` pins their (or any symbol's) starting
//! values — so earlier-trained layers enter later stages as fixed
//! constants, exactly the staged butterfly protocol.

use std::collections::{HashMap, HashSet};

use aria_core::ast::Circuit;
use omega_core::executor::Observable;
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::params::ParameterBinding;

use crate::lower::lower;
use crate::BackendSel;

/// Optimizer for the training loop (shared with [`omega_core::qml`]).
pub use omega_core::qml::Optimizer;

/// Training hyper-parameters.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub lr: f64,
    pub seed: u64,
    /// Initial parameters are drawn uniformly from `[-init_scale, init_scale]`.
    pub init_scale: f64,
    /// Symbol names excluded from training (layer-wise freezing). Their
    /// values stay at the seeded init unless pinned via [`Self::init`].
    /// Unknown names are rejected up front — a typo must not silently
    /// train the parameter it meant to freeze.
    pub frozen: Vec<String>,
    /// Explicit initial values by symbol name, overriding the seeded
    /// random init — how earlier-trained layers enter a later stage.
    pub init: HashMap<String, f64>,
    /// Update rule. Defaults to plain gradient descent.
    pub optimizer: Optimizer,
    /// Gradient method. Defaults to the per-slot parameter-shift rules;
    /// `GradMethod::ParallelParameterShift` batches trailing-commuting-
    /// block gradients into one evaluation (arXiv:2606.03517).
    pub grad_method: GradMethod,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 200,
            lr: 0.15,
            seed: 1,
            init_scale: 1.0,
            frozen: Vec::new(),
            init: HashMap::new(),
            optimizer: Optimizer::Gd,
            grad_method: GradMethod::ParameterShift,
        }
    }
}

/// Outcome of a training run.
#[derive(Clone, Debug)]
pub struct TrainResult {
    /// Final trained parameter values, keyed by Aria symbol name.
    pub params: HashMap<String, f64>,
    /// Objective value `⟨O⟩` at the start of each step (length = `steps`).
    pub history: Vec<f64>,
    /// Objective value after the final update.
    pub final_value: f64,
}

/// Deterministic SplitMix64 → uniform f64 in `[-scale, scale]`.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);
        // 53-bit mantissa → [0, 1)
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Per-parameter Adam state.
#[derive(Default)]
struct AdamState {
    m: f64,
    v: f64,
}

/// Minimize `⟨observable⟩` over the circuit's free symbols by gradient descent
/// with parameter-shift gradients. Pure Rust; `sel` chooses the backend the
/// expectation/gradient run on.
pub fn train_expectation(
    circuit: &Circuit,
    observable: &str,
    cfg: &TrainConfig,
    sel: BackendSel,
) -> Result<TrainResult, String> {
    // Expectation + parameter-shift gradients run on the selected backend, so
    // `aria train --backend tch` trains on libtorch, `--backend gpu` on the GPU,
    // etc. (Parameter-shift only needs `Backend::expectation`.)
    let backend = crate::run::make_backend(sel)?;

    let low = lower(circuit)?;
    if low.symbol_ids.is_empty() {
        return Err("circuit has no trainable symbols (use `symbolic[..]`)".into());
    }
    let obs = Observable::parse(observable)?;

    // Validate freeze/init names before anything trains.
    for name in cfg.frozen.iter().chain(cfg.init.keys()) {
        if !low.symbol_ids.contains_key(name) {
            let mut known: Vec<&String> = low.symbol_ids.keys().collect();
            known.sort();
            return Err(format!(
                "unknown symbol '{name}' in frozen/init (circuit has: {})",
                known
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    let frozen: HashSet<&str> = cfg.frozen.iter().map(|s| s.as_str()).collect();
    let trainable_ids: HashSet<u32> = low
        .symbol_ids
        .iter()
        .filter(|(n, _)| !frozen.contains(n.as_str()))
        .map(|(_, &id)| id)
        .collect();
    if trainable_ids.is_empty() {
        return Err("every symbol is frozen — nothing to train".into());
    }

    // id → name, and seeded initial values per name (sorted for
    // deterministic draw order), overridden by cfg.init.
    let id_to_name: HashMap<u32, String> = low
        .symbol_ids
        .iter()
        .map(|(n, i)| (*i, n.clone()))
        .collect();
    let mut rng = SplitMix64(cfg.seed);
    let mut names: Vec<&String> = low.symbol_ids.keys().collect();
    names.sort();
    let mut theta: HashMap<String, f64> = names
        .iter()
        .map(|n| {
            let seeded = (rng.next_f64() * 2.0 - 1.0) * cfg.init_scale;
            let v = cfg.init.get(n.as_str()).copied().unwrap_or(seeded);
            ((*n).clone(), v)
        })
        .collect();

    let binding_from = |theta: &HashMap<String, f64>| -> ParameterBinding {
        let mut b = ParameterBinding::new();
        for (name, &id) in &low.symbol_ids {
            b.bind(id, theta[name]);
        }
        b
    };

    let expect = |binding: &ParameterBinding| -> Result<f64, String> {
        backend
            .expectation(&low.ir, binding, &obs)
            .map_err(|e| e.to_string())
    };

    let mut adam: HashMap<u32, AdamState> = HashMap::new();
    let mut history = Vec::with_capacity(cfg.steps);
    for step in 0..cfg.steps {
        let binding = binding_from(&theta);
        history.push(expect(&binding)?);

        let grads = compute_gradient_for(
            backend.as_ref(),
            &low.ir,
            &binding,
            &obs,
            &cfg.grad_method,
            Some(&trainable_ids),
        )
        .map_err(|e| e.to_string())?;

        for (id, g) in grads {
            if !trainable_ids.contains(&id) {
                continue;
            }
            let Some(name) = id_to_name.get(&id) else {
                continue;
            };
            let update = match cfg.optimizer {
                Optimizer::Gd => cfg.lr * g,
                Optimizer::Adam {
                    beta1,
                    beta2,
                    epsilon,
                } => {
                    let st = adam.entry(id).or_default();
                    st.m = beta1 * st.m + (1.0 - beta1) * g;
                    st.v = beta2 * st.v + (1.0 - beta2) * g * g;
                    let t = (step + 1) as f64;
                    let m_hat = st.m / (1.0 - beta1.powf(t));
                    let v_hat = st.v / (1.0 - beta2.powf(t));
                    cfg.lr * m_hat / (v_hat.sqrt() + epsilon)
                }
            };
            *theta.get_mut(name).unwrap() -= update;
        }
    }

    let final_value = expect(&binding_from(&theta))?;
    Ok(TrainResult {
        params: theta,
        history,
        final_value,
    })
}
