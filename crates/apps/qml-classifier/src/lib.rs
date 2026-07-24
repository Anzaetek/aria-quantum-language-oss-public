// SPDX-License-Identifier: Apache-2.0
//! qml_classifier — data-reuploading binary classifier (confusion matrix).
//!
//! WHAT: classify a SYNTHETIC 1-D dataset (label y = sign(x)).
//! QUANTUM: the single-qubit data-reuploading model in qml_classifier.aria,
//!   prediction = sign(⟨Z⟩). Weights are trained host-side with **exact
//!   adjoint gradients** (`StatevectorBackend::adjoint_gradient` — one
//!   forward + one backward sweep per point, no finite-difference epsilon),
//!   then the trained model runs INFERENCE on a held-out test set THROUGH
//!   omega_app.wasm (expectations mode), and we build the confusion matrix
//!   from those wasm-computed predictions.
//! CLASSICAL: the ground-truth labels y = sign(x) of the same test set.
//! CHECK: classification accuracy ≥ 0.9.
//!
//! Synthetic data only (no external datasets) — matches the repo's eval rules.
//!
//! This is the canonical "train an Aria circuit from Rust" example: it
//! lowers ONCE, binds many, and drives the gradient loop through the public
//! `omega_core::executor::Backend` trait — the library surface an external
//! integration should copy (see `docs/LIBRARY.md`). MSE loss
//! `L = Σ (⟨Z⟩ − y)²` gives `dL/dθ = Σ 2·(⟨Z⟩ − y)·∂⟨Z⟩/∂θ`, and the
//! per-point `∂⟨Z⟩/∂θ` for every trainable weight comes from ONE
//! `compute_gradient_for(.., GradMethod::Adjoint, Some(&weights))` call.

use std::collections::{HashMap, HashSet};

use aria_verify_core::{banner, harness, resolve, Observable, Transport, Verdict};
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::SymbolId;
use omega_core::executor::Backend as _;
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::params::ParameterBinding;

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let l: i64 = 3;
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "qml_classifier",
        "binary classify synthetic y=sign(x); confusion matrix from the model",
        &transport.label(guest),
    );

    // Lower ONCE; every training/inference step binds into this same IR.
    let lowered = harness::load_lowered("qml_classifier.aria", "QMLClassifier", &[("L", l)])?;
    let n_sym = lowered.symbol_ids.len();
    let x_id: SymbolId = *lowered
        .symbol_ids
        .get("x_0")
        .ok_or("qml_classifier.aria: missing data symbol x_0")?;
    let weight_ids: Vec<SymbolId> = (0..3 * l)
        .map(|k| {
            lowered
                .symbol_ids
                .get(&format!("theta_{k}"))
                .copied()
                .ok_or(format!("missing weight theta_{k}"))
        })
        .collect::<Result<_, _>>()?;
    // Only these symbols train; `x_0` carries the data and is held fixed
    // per point. `compute_gradient_for(.., Some(&trainable))` skips the
    // rest, so the data symbol never receives a spurious gradient.
    let trainable: HashSet<SymbolId> = weight_ids.iter().copied().collect();
    let weight_index: HashMap<SymbolId, usize> = weight_ids
        .iter()
        .enumerate()
        .map(|(k, &id)| (id, k))
        .collect();

    // Synthetic data: label = sign(feature). Feature scaled into the rotation.
    let scale = 1.2_f64;
    let make = |xs: &[f64]| -> Vec<(f64, f64)> {
        xs.iter()
            .map(|&x| (x * scale, if x >= 0.0 { 1.0 } else { -1.0 }))
            .collect()
    };
    // Train on a grid, test on the interleaved off-grid points (held out).
    let train_x: Vec<f64> = (0..40)
        .map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / 40.0)
        .collect();
    let test_x: Vec<f64> = (0..20).map(|i| -0.95 + 2.0 * (i as f64) / 19.0).collect();
    let train = make(&train_x);
    let test = make(&test_x);

    // Assemble the flat symbol vector (data at x_id, weights at weight_ids)
    // for the wasm inference transport, which takes params ordered by
    // ascending SymbolId.
    let assemble = |feature: f64, w: &[f64]| -> Vec<f64> {
        let mut p = vec![0.0; n_sym];
        p[x_id as usize] = feature;
        for (k, &wid) in weight_ids.iter().enumerate() {
            p[wid as usize] = w[k];
        }
        p
    };
    // The same (feature, weights) as an omega `ParameterBinding` — the
    // native library surface used for training gradients.
    let binding = |feature: f64, w: &[f64]| -> ParameterBinding {
        let mut b = ParameterBinding::new();
        b.bind(x_id, feature);
        for (k, &wid) in weight_ids.iter().enumerate() {
            b.bind(wid, w[k]);
        }
        b
    };

    // ---- Train weights host-side with EXACT ADJOINT gradients. ----
    // No finite-difference epsilon: `adjoint_gradient` returns every
    // trainable weight's ∂⟨Z⟩/∂θ in one forward + one backward sweep,
    // ~4× faster than the two central-difference evaluations per weight
    // it replaces (and exact, not O(ε²)).
    let backend = StatevectorBackend::new();
    let z_obs = Observable::parse("1.0*Z0")?;
    let n_w = weight_ids.len();
    let mut w = vec![0.1_f64; n_w];
    let lr = 0.5;
    let expect = |w: &[f64], feature: f64| -> f64 {
        backend
            .expectation(&lowered.ir, &binding(feature, w), &z_obs)
            .unwrap_or(0.0)
    };
    let mse = |w: &[f64]| -> f64 {
        train
            .iter()
            .map(|&(f, y)| (expect(w, f) - y).powi(2))
            .sum::<f64>()
            / train.len() as f64
    };
    for _ in 0..250 {
        let mut grad = vec![0.0; n_w];
        for &(f, y) in &train {
            let b = binding(f, &w);
            let residual = expect(&w, f) - y;
            // dMSE/dθ = (2/N)·Σ residual · ∂⟨Z⟩/∂θ; the 1/N folds into lr
            // scaling below via the accumulated sum.
            let grads = compute_gradient_for(
                &backend,
                &lowered.ir,
                &b,
                &z_obs,
                &GradMethod::Adjoint,
                Some(&trainable),
            )
            .map_err(|e| e.to_string())?;
            for (sym, g) in grads {
                if let Some(&k) = weight_index.get(&sym) {
                    grad[k] += 2.0 * residual * g / train.len() as f64;
                }
            }
        }
        for i in 0..n_w {
            w[i] -= lr * grad[i];
        }
    }
    println!(
        "  trained {n_w} weights (exact adjoint gradients); final train MSE = {:.6}",
        mse(&w)
    );

    // ---- Inference on the held-out test set THROUGH wasm. ----
    let z_obs = Observable::parse("1.0*Z0")?;
    let (mut tp, mut fn_, mut fp, mut tn) = (0u32, 0u32, 0u32, 0u32);
    for &(feature, y) in &test {
        let params = assemble(feature, &w);
        let (_payload, z) = harness::execute_report(
            transport,
            lowered.ir.clone(),
            harness::AppMode::Expectations(vec![z_obs.clone()]),
            &params,
        )?;
        let pred = if z >= 0.0 { 1.0 } else { -1.0 };
        match (y > 0.0, pred > 0.0) {
            (true, true) => tp += 1,
            (true, false) => fn_ += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
        }
    }

    Ok(banner::report_confusion(
        "qml_classifier",
        "sign(⟨Z⟩), inference in wasm",
        tp,
        fn_,
        fp,
        tn,
        0.9,
    ))
}
