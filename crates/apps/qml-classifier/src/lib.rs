// SPDX-License-Identifier: Apache-2.0
//! qml_classifier — data-reuploading binary classifier (confusion matrix).
//!
//! WHAT: classify a SYNTHETIC 1-D dataset (label y = sign(x)).
//! QUANTUM: the single-qubit data-reuploading model in qml_classifier.aria,
//!   prediction = sign(⟨Z⟩). Weights are trained host-side; the trained model
//!   then runs INFERENCE on a held-out test set THROUGH omega_app.wasm
//!   (expectations mode), and we build the confusion matrix from those
//!   wasm-computed predictions.
//! CLASSICAL: the ground-truth labels y = sign(x) of the same test set.
//! CHECK: classification accuracy ≥ 0.9.
//!
//! Synthetic data only (no external datasets) — matches the repo's eval rules.

use aria_verify_core::{banner, harness, resolve, HostState, Observable, Transport, Verdict};

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let l: i64 = 3;
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "qml_classifier",
        "binary classify synthetic y=sign(x); confusion matrix from the model",
        &transport.label(guest),
    );

    let lowered = harness::load_lowered("qml_classifier.aria", "QMLClassifier", &[("L", l)])?;
    let n_sym = lowered.symbol_ids.len();
    let x_id = *lowered
        .symbol_ids
        .get("x_0")
        .ok_or("qml_classifier.aria: missing data symbol x_0")? as usize;
    let weight_ids: Vec<usize> = (0..3 * l)
        .map(|k| {
            lowered
                .symbol_ids
                .get(&format!("theta_{k}"))
                .map(|&i| i as usize)
                .ok_or(format!("missing weight theta_{k}"))
        })
        .collect::<Result<_, _>>()?;

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

    // Assemble the full symbol vector (data at x_id, weights at weight_ids).
    let assemble = |feature: f64, w: &[f64]| -> Vec<f64> {
        let mut p = vec![0.0; n_sym];
        p[x_id] = feature;
        for (k, &wid) in weight_ids.iter().enumerate() {
            p[wid] = w[k];
        }
        p
    };

    // ---- Train weights host-side (fast, deterministic GD on MSE). ----
    let mut host = HostState::new();
    let cid = host.register_circuit(lowered.ir.clone());
    let oid = host.register_observable(Observable::parse("1.0*Z0")?);
    let predict_z = |host: &HostState, feature: f64, w: &[f64]| -> f64 {
        host.execute_expectation(cid, &assemble(feature, w), oid)
            .unwrap_or(0.0)
    };
    let n_w = weight_ids.len();
    let mut w = vec![0.1_f64; n_w];
    let lr = 0.5;
    let eps = 1e-5;
    let loss = |host: &HostState, w: &[f64]| -> f64 {
        train
            .iter()
            .map(|&(f, y)| (predict_z(host, f, w) - y).powi(2))
            .sum::<f64>()
            / train.len() as f64
    };
    for _ in 0..250 {
        let mut grad = vec![0.0; n_w];
        for i in 0..n_w {
            let mut wp = w.clone();
            wp[i] += eps;
            let mut wm = w.clone();
            wm[i] -= eps;
            grad[i] = (loss(&host, &wp) - loss(&host, &wm)) / (2.0 * eps);
        }
        for i in 0..n_w {
            w[i] -= lr * grad[i];
        }
    }
    println!(
        "  trained {n_w} weights; final train MSE = {:.6}",
        loss(&host, &w)
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
