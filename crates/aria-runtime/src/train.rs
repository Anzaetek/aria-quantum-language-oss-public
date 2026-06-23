// SPDX-License-Identifier: Apache-2.0
//! Pure-Rust variational training for Aria circuits.
//!
//! Minimizes the expectation value `⟨ψ(θ)|O|ψ(θ)⟩` of a Pauli observable over
//! the circuit's free (trainable) symbols by gradient descent, using
//! omega-core's parameter-shift gradients — **no libtorch**. This is the
//! engine behind `aria train` and the proof that Aria's `symbolic[k]`
//! parameters train end-to-end on the pure-Rust backend. The optional `tch`
//! plugin (Phase 4) accelerates this same loop with batched/GPU autograd.

use std::collections::HashMap;

use aria_core::ast::Circuit;
use omega_core::executor::Observable;
use omega_core::gradient::{compute_gradient, GradMethod};
use omega_core::params::ParameterBinding;

use crate::lower::lower;
use crate::BackendSel;

/// Training hyper-parameters.
#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    pub steps: usize,
    pub lr: f64,
    pub seed: u64,
    /// Initial parameters are drawn uniformly from `[-init_scale, init_scale]`.
    pub init_scale: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 200,
            lr: 0.15,
            seed: 1,
            init_scale: 1.0,
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

    // id → name, and seeded initial values per name.
    let id_to_name: HashMap<u32, String> = low
        .symbol_ids
        .iter()
        .map(|(n, i)| (*i, n.clone()))
        .collect();
    let mut rng = SplitMix64(cfg.seed);
    let mut theta: HashMap<String, f64> = low
        .symbol_ids
        .keys()
        .map(|n| (n.clone(), (rng.next_f64() * 2.0 - 1.0) * cfg.init_scale))
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

    let mut history = Vec::with_capacity(cfg.steps);
    for _ in 0..cfg.steps {
        let binding = binding_from(&theta);
        history.push(expect(&binding)?);

        let grads = compute_gradient(
            backend.as_ref(),
            &low.ir,
            &binding,
            &obs,
            &GradMethod::ParameterShift,
        )
        .map_err(|e| e.to_string())?;

        for (id, g) in grads {
            if let Some(name) = id_to_name.get(&id) {
                *theta.get_mut(name).unwrap() -= cfg.lr * g;
            }
        }
    }

    let final_value = expect(&binding_from(&theta))?;
    Ok(TrainResult {
        params: theta,
        history,
        final_value,
    })
}
