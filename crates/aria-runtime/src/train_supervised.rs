// SPDX-License-Identifier: Apache-2.0
//! Supervised dataset training for Aria circuits — the entry point the CLI's
//! `aria train --data X.csv --labels y.csv` drives.
//!
//! Unlike [`crate::train::train_expectation`] (VQE-shaped: minimize ONE
//! observable expectation), this fits a labelled dataset. The model
//! convention is data re-uploading: the circuit declares
//!
//! ```text
//!   let x = symbolic[d]   -- d feature symbols (x_0 .. x_{d-1})
//!   let theta = symbolic[..]   -- weight symbols (any name that isn't x_*)
//! ```
//!
//! and the readout is a single Pauli observable `⟨O⟩`. Per sample the feature
//! symbols are bound to the row and the circuit is re-used (lower ONCE, bind
//! many). A cheap closed-form affine head `(s, b)` maps `⟨O⟩` to a prediction:
//! `ŷ = s·⟨O⟩ + b` for MSE, `p = σ(s·⟨O⟩ + b)` for BCE. The head is refit each
//! step, so the circuit weights only have to make `⟨O⟩` *separable* — the same
//! hybrid-sandwich pattern the QML example apps use.
//!
//! Gradients: `dL/dθ = Σ_row (∂L/∂⟨O⟩)·(∂⟨O⟩/∂θ)`, with `∂⟨O⟩/∂θ` from a single
//! adjoint pass per row (`GradMethod::Adjoint` by default) and `∂L/∂⟨O⟩` the
//! affine-head chain-rule factor: `2·(ŷ−y)·s` (MSE) or `(p−y)·s` (BCE).

use std::collections::{HashMap, HashSet};

use aria_core::ast::Circuit;
use omega_core::circuit::SymbolId;
use omega_core::executor::Observable;
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::params::ParameterBinding;

use crate::lower::lower;
use crate::train::{Optimizer, SplitMix64};
use crate::BackendSel;

/// Training objective.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Loss {
    /// Mean squared error against `ŷ = s·⟨O⟩ + b`. Labels are used as-is.
    #[default]
    Mse,
    /// Binary cross-entropy against `p = σ(s·⟨O⟩ + b)`. Labels are treated
    /// as `{0, 1}` (values are thresholded at the midpoint of their range).
    Bce,
}

impl Loss {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "mse" => Ok(Loss::Mse),
            "bce" => Ok(Loss::Bce),
            other => Err(format!("unknown --loss '{other}' (mse | bce)")),
        }
    }
}

/// Supervised training hyper-parameters. Mirrors [`crate::TrainConfig`] but
/// adds the dataset-model knobs (feature prefix, loss).
#[derive(Clone, Debug)]
pub struct SupervisedConfig {
    pub steps: usize,
    pub lr: f64,
    pub seed: u64,
    /// Initial weights drawn uniformly from `[-init_scale, init_scale]`.
    pub init_scale: f64,
    /// Symbols whose names start with `"{feature_prefix}_"` are the data
    /// features (bound per row); everything else is a trainable weight.
    pub feature_prefix: String,
    pub loss: Loss,
    pub optimizer: Optimizer,
    /// Weight symbols excluded from updates (frozen).
    pub frozen: Vec<String>,
    pub grad_method: GradMethod,
}

impl Default for SupervisedConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            lr: 0.1,
            seed: 1,
            init_scale: 0.3,
            feature_prefix: "x".to_string(),
            loss: Loss::Mse,
            optimizer: Optimizer::Gd,
            frozen: Vec::new(),
            grad_method: GradMethod::Adjoint,
        }
    }
}

/// Outcome of supervised training.
#[derive(Clone, Debug)]
pub struct SupervisedResult {
    /// Trained weight values by Aria symbol name.
    pub weights: HashMap<String, f64>,
    /// The final affine head `(s, b)` mapping `⟨O⟩` to the prediction.
    pub head: (f64, f64),
    /// Mean training loss at the start of each step (length = `steps`).
    pub loss_history: Vec<f64>,
    /// Mean training loss after the final update.
    pub final_loss: f64,
    /// Training-set ROC-AUC of the final model (rank statistic; 0.5 = chance).
    pub final_auc: f64,
}

/// Per-parameter Adam moments.
#[derive(Default, Clone, Copy)]
struct Moments {
    m: f64,
    v: f64,
}

/// Rank-based ROC-AUC of `scores` against binary `labels` (positive =
/// `label > threshold`); ties contribute ½. 0.5 is chance.
fn auc(scores: &[f64], labels: &[f64], threshold: f64) -> f64 {
    let (mut pos, mut neg, mut wins) = (0.0, 0.0, 0.0);
    for (i, &si) in scores.iter().enumerate() {
        if labels[i] <= threshold {
            continue;
        }
        pos += 1.0;
        for (j, &sj) in scores.iter().enumerate() {
            if labels[j] > threshold {
                continue;
            }
            wins += if si > sj {
                1.0
            } else if si == sj {
                0.5
            } else {
                0.0
            };
        }
    }
    for &l in labels {
        if l <= threshold {
            neg += 1.0;
        }
    }
    if pos == 0.0 || neg == 0.0 {
        return 0.5;
    }
    wins / (pos * neg)
}

/// Least-squares affine head `ŷ = s·z + b` (MSE): `s = cov(z,y)/var(z)`.
fn fit_head_mse(z: &[f64], y: &[f64]) -> (f64, f64) {
    let n = z.len() as f64;
    let mz = z.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (&zi, &yi) in z.iter().zip(y) {
        cov += (zi - mz) * (yi - my);
        var += (zi - mz) * (zi - mz);
    }
    if var < 1e-12 {
        return (0.0, my);
    }
    let s = cov / var;
    (s, my - s * mz)
}

/// 1-D logistic head `p = σ(s·z + b)` (BCE), fit by a few GD steps on the
/// 2 scalars with `z` fixed — deterministic and cheap.
fn fit_head_bce(z: &[f64], y01: &[f64]) -> (f64, f64) {
    let n = z.len() as f64;
    // Warm start from the least-squares direction (well-conditioned).
    let (mut s, mut b) = fit_head_mse(z, y01);
    for _ in 0..100 {
        let (mut gs, mut gb) = (0.0, 0.0);
        for (&zi, &yi) in z.iter().zip(y01) {
            let p = 1.0 / (1.0 + (-(s * zi + b)).exp());
            let d = p - yi;
            gs += d * zi / n;
            gb += d / n;
        }
        s -= 0.5 * gs;
        b -= 0.5 * gb;
    }
    (s, b)
}

/// Train `circuit` on `(train_x, train_y)` reading out `observable`, returning
/// the trained weights, affine head, loss history, and final AUC. Lower once,
/// bind many; gradients via `cfg.grad_method` (adjoint by default).
pub fn train_supervised(
    circuit: &Circuit,
    train_x: &[Vec<f64>],
    train_y: &[f64],
    observable: &str,
    cfg: &SupervisedConfig,
    sel: BackendSel,
) -> Result<SupervisedResult, String> {
    if train_x.is_empty() {
        return Err("empty training set".into());
    }
    if train_x.len() != train_y.len() {
        return Err(format!(
            "feature rows ({}) and labels ({}) length mismatch",
            train_x.len(),
            train_y.len()
        ));
    }
    let backend = crate::run::make_backend(sel)?;
    let low = lower(circuit)?;
    let obs = Observable::parse(observable)?;

    // Partition symbols into features (name = "{prefix}_{i}") and weights.
    let d = train_x[0].len();
    let feat_prefix = format!("{}_", cfg.feature_prefix);
    let mut feature_ids: Vec<SymbolId> = vec![0; d];
    let mut have_feature = vec![false; d];
    let mut weight_names: Vec<String> = Vec::new();
    for (name, &id) in &low.symbol_ids {
        if let Some(rest) = name.strip_prefix(&feat_prefix) {
            if let Ok(i) = rest.parse::<usize>() {
                if i >= d {
                    return Err(format!(
                        "feature symbol '{name}' index {i} ≥ feature count {d} \
                         (data rows have {d} columns)"
                    ));
                }
                feature_ids[i] = id;
                have_feature[i] = true;
                continue;
            }
        }
        weight_names.push(name.clone());
    }
    for (i, ok) in have_feature.iter().enumerate() {
        if !ok {
            return Err(format!(
                "circuit is missing feature symbol '{}{i}' — data has {d} columns, so the \
                 circuit must declare `let {} = symbolic[{d}]` (feature-prefix '{}')",
                feat_prefix, cfg.feature_prefix, cfg.feature_prefix
            ));
        }
    }
    if weight_names.is_empty() {
        return Err("circuit has no trainable weight symbols (only features)".into());
    }
    weight_names.sort();

    // Validate freeze names, build id maps.
    let known: HashSet<&str> = weight_names.iter().map(|s| s.as_str()).collect();
    for f in &cfg.frozen {
        if !known.contains(f.as_str()) {
            return Err(format!(
                "unknown frozen weight '{f}' (weights: {})",
                weight_names.join(", ")
            ));
        }
    }
    let frozen: HashSet<&str> = cfg.frozen.iter().map(|s| s.as_str()).collect();
    let weight_ids: Vec<SymbolId> = weight_names.iter().map(|n| low.symbol_ids[n]).collect();
    let trainable: HashSet<SymbolId> = weight_names
        .iter()
        .filter(|n| !frozen.contains(n.as_str()))
        .map(|n| low.symbol_ids[n])
        .collect();
    if trainable.is_empty() {
        return Err("every weight is frozen — nothing to train".into());
    }
    let weight_index: HashMap<SymbolId, usize> = weight_ids
        .iter()
        .enumerate()
        .map(|(k, &id)| (id, k))
        .collect();

    // Seeded init.
    let mut rng = SplitMix64(cfg.seed);
    let mut w: Vec<f64> = (0..weight_ids.len())
        .map(|_| (rng.next_f64() * 2.0 - 1.0) * cfg.init_scale)
        .collect();

    // BCE label canonicalisation: threshold at the midpoint of the range.
    let (lo, hi) = train_y
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &y| {
            (a.min(y), b.max(y))
        });
    let mid = 0.5 * (lo + hi);
    let y01: Vec<f64> = train_y
        .iter()
        .map(|&y| if y > mid { 1.0 } else { 0.0 })
        .collect();

    // Bind a row (features + current weights) into a reusable binding.
    let binding = |w: &[f64], x: &[f64]| -> ParameterBinding {
        let mut b = ParameterBinding::new();
        for (i, &id) in feature_ids.iter().enumerate() {
            b.bind(id, x[i]);
        }
        for (k, &id) in weight_ids.iter().enumerate() {
            b.bind(id, w[k]);
        }
        b
    };
    // Row-batched readout: build one binding per row and evaluate them in a
    // single (backend-parallel) call. Index-preserving, so results are
    // identical to a sequential loop.
    let readout = |w: &[f64]| -> Result<Vec<f64>, String> {
        let bnds: Vec<ParameterBinding> = train_x.iter().map(|x| binding(w, x)).collect();
        let refs: Vec<&ParameterBinding> = bnds.iter().collect();
        backend
            .expectation_batch(&low.ir, &refs, &obs)
            .map_err(|e| e.to_string())
    };

    let n = train_x.len() as f64;
    let mut adam: HashMap<SymbolId, Moments> = HashMap::new();
    let mut loss_history = Vec::with_capacity(cfg.steps);

    for step in 0..cfg.steps {
        // Build the per-row bindings ONCE and reuse them for both the readout
        // and the gradient pass (they only depend on the current weights).
        let bnds: Vec<ParameterBinding> = train_x.iter().map(|x| binding(&w, x)).collect();
        let refs: Vec<&ParameterBinding> = bnds.iter().collect();
        let z = backend
            .expectation_batch(&low.ir, &refs, &obs)
            .map_err(|e| e.to_string())?;
        // Refit the affine head to the current ⟨O⟩ profile.
        let head = match cfg.loss {
            Loss::Mse => fit_head_mse(&z, train_y),
            Loss::Bce => fit_head_bce(&z, &y01),
        };
        let (s, b) = head;

        // Loss + per-row dL/d⟨O⟩.
        let mut loss = 0.0;
        let mut dl_dz = vec![0.0; z.len()];
        for (i, &zi) in z.iter().enumerate() {
            match cfg.loss {
                Loss::Mse => {
                    let yhat = s * zi + b;
                    let r = yhat - train_y[i];
                    loss += r * r;
                    dl_dz[i] = 2.0 * r * s; // ∂/∂z of (s·z+b − y)²
                }
                Loss::Bce => {
                    let p = 1.0 / (1.0 + (-(s * zi + b)).exp());
                    let y = y01[i];
                    let eps = 1e-12;
                    loss += -(y * (p + eps).ln() + (1.0 - y) * (1.0 - p + eps).ln());
                    dl_dz[i] = (p - y) * s; // ∂/∂z of BCE(σ(s·z+b))
                }
            }
        }
        loss_history.push(loss / n);

        // Accumulate weight gradients: Σ_row dL/dz · ∂z/∂θ.
        let mut grad: HashMap<SymbolId, f64> = HashMap::new();
        if matches!(cfg.grad_method, GradMethod::Adjoint) {
            // Fast path: one backend-parallel batch of adjoint passes over all
            // rows (the loop's dominant cost), reusing the readout bindings,
            // then a deterministic reduction.
            let per_row = backend
                .adjoint_gradient_batch(&low.ir, &refs, &obs)
                .map_err(|e| e.to_string())?;
            for (i, row) in per_row.into_iter().enumerate() {
                if dl_dz[i] == 0.0 {
                    continue;
                }
                let row = row.ok_or("backend returned no adjoint gradient")?;
                for (sym, g) in row {
                    // adjoint returns every symbol; keep only trainable weights
                    // (this is where freezing is applied on the batch path).
                    if trainable.contains(&sym) {
                        *grad.entry(sym).or_insert(0.0) += dl_dz[i] * g / n;
                    }
                }
            }
        } else {
            // Parameter-shift / parallel-shift: per-row, with the `only` filter
            // doing the freezing.
            for (i, bnd) in bnds.iter().enumerate() {
                if dl_dz[i] == 0.0 {
                    continue;
                }
                let grads = compute_gradient_for(
                    backend.as_ref(),
                    &low.ir,
                    bnd,
                    &obs,
                    &cfg.grad_method,
                    Some(&trainable),
                )
                .map_err(|e| e.to_string())?;
                for (sym, g) in grads {
                    *grad.entry(sym).or_insert(0.0) += dl_dz[i] * g / n;
                }
            }
        }

        // Stall escape: a flat readout (degenerate affine head, s ≈ 0) makes
        // every dL/d⟨O⟩ zero, so there is no gradient. Rather than silently
        // freezing, nudge the trainable weights deterministically to break the
        // plateau and try the step again next iteration.
        let grad_norm: f64 = grad.values().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < 1e-12 && step + 1 < cfg.steps {
            let mut rng = SplitMix64(cfg.seed ^ 0x5715_2AE1 ^ (step as u64 + 1));
            for (k, &id) in weight_ids.iter().enumerate() {
                if trainable.contains(&id) {
                    w[k] += (rng.next_f64() * 2.0 - 1.0) * 1e-2;
                }
            }
            continue;
        }

        // Optimizer update over trainable weights.
        for (&sym, &g) in &grad {
            let Some(&k) = weight_index.get(&sym) else {
                continue;
            };
            let update = match cfg.optimizer {
                Optimizer::Gd => cfg.lr * g,
                Optimizer::Adam {
                    beta1,
                    beta2,
                    epsilon,
                } => {
                    let st = adam.entry(sym).or_default();
                    st.m = beta1 * st.m + (1.0 - beta1) * g;
                    st.v = beta2 * st.v + (1.0 - beta2) * g * g;
                    let t = (step + 1) as f64;
                    let m_hat = st.m / (1.0 - beta1.powf(t));
                    let v_hat = st.v / (1.0 - beta2.powf(t));
                    cfg.lr * m_hat / (v_hat.sqrt() + epsilon)
                }
            };
            w[k] -= update;
        }
    }

    // Final metrics.
    let z = readout(&w)?;
    let head = match cfg.loss {
        Loss::Mse => fit_head_mse(&z, train_y),
        Loss::Bce => fit_head_bce(&z, &y01),
    };
    let (s, b) = head;
    let final_loss = {
        let mut l = 0.0;
        for (i, &zi) in z.iter().enumerate() {
            match cfg.loss {
                Loss::Mse => {
                    let r = s * zi + b - train_y[i];
                    l += r * r;
                }
                Loss::Bce => {
                    let p = 1.0 / (1.0 + (-(s * zi + b)).exp());
                    let y = y01[i];
                    let eps = 1e-12;
                    l += -(y * (p + eps).ln() + (1.0 - y) * (1.0 - p + eps).ln());
                }
            }
        }
        l / n
    };
    // AUC on the head output (monotone in z, so equivalently on s·z).
    let scores: Vec<f64> = z.iter().map(|&zi| s * zi + b).collect();
    let final_auc = auc(&scores, train_y, mid);

    let weights: HashMap<String, f64> = weight_names
        .iter()
        .cloned()
        .zip(w.iter().copied())
        .collect();
    Ok(SupervisedResult {
        weights,
        head,
        loss_history,
        final_loss,
        final_auc,
    })
}
