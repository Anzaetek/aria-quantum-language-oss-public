// SPDX-License-Identifier: Apache-2.0
//! Supervised dataset training: BCE/MSE convergence on a synthetic problem,
//! freeze-mask semantics, and a smoke run on the vendored open UCI heart set.

use aria_core::ast::{parse_aria, Circuit};
use aria_runtime::{train_supervised, BackendSel, Loss, Optimizer, SupervisedConfig};

/// A small data-reuploading 1-qubit classifier: feature `x_0` re-uploaded
/// across `L` layers, each with an RZ·RY·RZ trainable rotation. Readout Z0.
fn reuploader(l: usize) -> Circuit {
    let mut body = String::new();
    for layer in 0..l {
        body.push_str("  apply RY(x[0]) on q[0]\n");
        body.push_str(&format!("  apply RZ(theta[{}]) on q[0]\n", 3 * layer));
        body.push_str(&format!("  apply RY(theta[{}]) on q[0]\n", 3 * layer + 1));
        body.push_str(&format!("  apply RZ(theta[{}]) on q[0]\n", 3 * layer + 2));
    }
    let src = format!(
        "circuit C() {{\n  qreg q[1]\n  let theta = symbolic[{}]\n  let x = symbolic[1]\n{body}}}\n",
        3 * l
    );
    parse_aria(&src).unwrap().instantiate("C", &[]).unwrap()
}

/// Synthetic 1-D dataset: label = sign(feature), feature scaled into range.
fn synthetic() -> (Vec<Vec<f64>>, Vec<f64>) {
    let xs: Vec<f64> = (0..60)
        .map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / 60.0)
        .collect();
    let x = xs.iter().map(|&v| vec![v * 1.5]).collect();
    let y = xs
        .iter()
        .map(|&v| if v >= 0.0 { 1.0 } else { 0.0 })
        .collect();
    (x, y)
}

#[test]
fn bce_converges_on_synthetic_sign() {
    let c = reuploader(3);
    let (x, y) = synthetic();
    let cfg = SupervisedConfig {
        steps: 120,
        lr: 0.15,
        seed: 7,
        loss: Loss::Bce,
        optimizer: Optimizer::adam(),
        ..Default::default()
    };
    let r = train_supervised(&c, &x, &y, "Z0", &cfg, BackendSel::Sim).unwrap();
    assert!(
        r.final_auc >= 0.98,
        "BCE AUC {:.3} did not reach 0.98 (loss {:.4})",
        r.final_auc,
        r.final_loss
    );
    // Loss decreased substantially from the start.
    assert!(
        r.final_loss < r.loss_history[0] - 0.1,
        "loss did not decrease: {:.4} -> {:.4}",
        r.loss_history[0],
        r.final_loss
    );
}

#[test]
fn mse_converges_on_synthetic_sign() {
    let c = reuploader(3);
    let (x, mut y) = synthetic();
    // MSE against ±1 labels.
    for v in &mut y {
        *v = if *v > 0.5 { 1.0 } else { -1.0 };
    }
    let cfg = SupervisedConfig {
        steps: 120,
        lr: 0.15,
        seed: 7,
        loss: Loss::Mse,
        optimizer: Optimizer::adam(),
        ..Default::default()
    };
    let r = train_supervised(&c, &x, &y, "Z0", &cfg, BackendSel::Sim).unwrap();
    assert!(r.final_auc >= 0.98, "MSE AUC {:.3}", r.final_auc);
}

#[test]
fn frozen_weights_do_not_move() {
    let c = reuploader(2); // weights theta_0..theta_5
    let (x, y) = synthetic();
    let base = SupervisedConfig {
        steps: 30,
        lr: 0.2,
        seed: 3,
        loss: Loss::Bce,
        optimizer: Optimizer::adam(),
        ..Default::default()
    };

    // steps = 0 → weights are exactly the seeded init (same seed as below,
    // so the two runs share their starting weights).
    let init_run = SupervisedConfig {
        steps: 0,
        ..base.clone()
    };
    let init = train_supervised(&c, &x, &y, "Z0", &init_run, BackendSel::Sim).unwrap();

    // Freeze only theta_0 → it stays put while others move.
    let one_frozen = SupervisedConfig {
        frozen: vec!["theta_0".into()],
        ..base
    };
    let r = train_supervised(&c, &x, &y, "Z0", &one_frozen, BackendSel::Sim).unwrap();
    assert!(
        (r.weights["theta_0"] - init.weights["theta_0"]).abs() < 1e-12,
        "frozen theta_0 moved: {} -> {}",
        init.weights["theta_0"],
        r.weights["theta_0"]
    );
    let moved = (1..6).any(|k| {
        let n = format!("theta_{k}");
        (r.weights[&n] - init.weights[&n]).abs() > 1e-6
    });
    assert!(moved, "no unfrozen weight moved");
}

#[test]
fn unknown_frozen_weight_is_an_error() {
    let c = reuploader(1);
    let (x, y) = synthetic();
    let cfg = SupervisedConfig {
        frozen: vec!["nope".into()],
        ..Default::default()
    };
    let err = train_supervised(&c, &x, &y, "Z0", &cfg, BackendSel::Sim).unwrap_err();
    assert!(err.contains("unknown frozen weight 'nope'"), "err: {err}");
}

#[test]
fn feature_count_mismatch_is_reported() {
    let c = reuploader(1); // declares x[1]
                           // Two feature columns but the circuit only has x_0.
    let x = vec![vec![0.1, 0.2], vec![-0.3, 0.4]];
    let y = vec![1.0, 0.0];
    let err = train_supervised(
        &c,
        &x,
        &y,
        "Z0",
        &SupervisedConfig::default(),
        BackendSel::Sim,
    )
    .unwrap_err();
    assert!(err.contains("missing feature symbol 'x_1'"), "err: {err}");
}

/// A structurally-degenerate model: the feature `x_0` drives q[1] but the
/// readout is Z0 on q[0], which only ever sees trainable rotations. ⟨Z0⟩ is
/// therefore identical across every row, so the affine head is degenerate
/// (variance ≈ 0, slope ≈ 0) and dL/d⟨O⟩ is zero on every step. Training must
/// not panic or divide-by-zero — the stall escape keeps the loop advancing and
/// the run returns finite, honest metrics (AUC ≈ 0.5, no better than chance).
fn degenerate_model() -> Circuit {
    let src = "circuit Deg() {\n  qreg q[2]\n  let theta = symbolic[2]\n  let x = symbolic[1]\n  \
               apply RY(x[0]) on q[1]\n  apply RY(theta[0]) on q[0]\n  apply RZ(theta[1]) on q[0]\n}\n";
    parse_aria(src).unwrap().instantiate("Deg", &[]).unwrap()
}

#[test]
fn degenerate_head_does_not_panic_and_stays_finite() {
    let c = degenerate_model();
    let (x, y) = synthetic();
    let cfg = SupervisedConfig {
        steps: 20,
        lr: 0.2,
        seed: 11,
        loss: Loss::Bce,
        optimizer: Optimizer::adam(),
        ..Default::default()
    };
    let r = train_supervised(&c, &x, &y, "Z0", &cfg, BackendSel::Sim).unwrap();
    assert!(
        r.final_loss.is_finite(),
        "loss went non-finite: {}",
        r.final_loss
    );
    assert!(
        r.final_auc.is_finite() && (0.0..=1.0).contains(&r.final_auc),
        "AUC out of range: {}",
        r.final_auc
    );
    // A model that structurally cannot correlate the readout with the feature
    // should not appear to have learned anything.
    assert!(
        r.final_auc < 0.7,
        "degenerate model reported AUC {:.3} — suspiciously high",
        r.final_auc
    );
}

// ---- Smoke test on the vendored open UCI Cleveland heart dataset. ----

/// Minimal CSV reader for the vendored numeric file (no external crate; the
/// library takes `&[Vec<f64>]`, file I/O lives at the call site).
fn load_heart() -> Option<(Vec<Vec<f64>>, Vec<f64>)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/data/heart_cleveland.csv");
    let text = std::fs::read_to_string(path).ok()?;
    // 8 continuous-ish feature columns (as the butterfly app) + num target.
    let cols = [0usize, 1, 2, 3, 4, 6, 8, 9];
    let target = 13usize;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for line in text.lines().skip(1) {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() <= target {
            continue;
        }
        let feats: Option<Vec<f64>> = cols.iter().map(|&c| cells[c].trim().parse().ok()).collect();
        let t: Option<f64> = cells[target].trim().parse().ok();
        if let (Some(f), Some(t)) = (feats, t) {
            let mut r = f;
            r.push(t);
            rows.push(r);
        }
    }
    if rows.is_empty() {
        return None;
    }
    // Standardize the 8 features column-wise (z-score); label = num > 0.
    let d = 8;
    let n = rows.len() as f64;
    let mut mean = vec![0.0; d];
    let mut std = vec![1.0; d];
    for c in 0..d {
        let m = rows.iter().map(|r| r[c]).sum::<f64>() / n;
        let v = rows.iter().map(|r| (r[c] - m).powi(2)).sum::<f64>() / n;
        mean[c] = m;
        std[c] = if v.sqrt() > 1e-9 { v.sqrt() } else { 1.0 };
    }
    let x = rows
        .iter()
        .map(|r| (0..d).map(|c| ((r[c] - mean[c]) / std[c]).atan()).collect())
        .collect();
    let y = rows
        .iter()
        .map(|r| if r[d] > 0.0 { 1.0 } else { 0.0 })
        .collect();
    Some((x, y))
}

/// An 8-qubit RY-encoded classifier with a per-qubit RZ·RY weight layer and a
/// CX ring, reading out Z0. Features x_0..x_7.
fn heart_model() -> Circuit {
    let mut body = String::from("  repeat i from 0 to 7 {\n    apply RY(x[i]) on q[i]\n  }\n");
    for q in 0..8 {
        body.push_str(&format!("  apply RZ(theta[{}]) on q[{q}]\n", 2 * q));
        body.push_str(&format!("  apply RY(theta[{}]) on q[{q}]\n", 2 * q + 1));
    }
    for q in 0..8 {
        body.push_str(&format!("  apply CX on q[{q}], q[{}]\n", (q + 1) % 8));
    }
    let src = format!(
        "circuit Heart() {{\n  qreg q[8]\n  let theta = symbolic[16]\n  let x = symbolic[8]\n{body}}}\n"
    );
    parse_aria(&src).unwrap().instantiate("Heart", &[]).unwrap()
}

#[test]
fn heart_dataset_smoke_reaches_reasonable_auc() {
    let Some((x, y)) = load_heart() else {
        eprintln!("heart_cleveland.csv not found — skipping smoke test");
        return;
    };
    assert!(
        x.len() > 250,
        "expected the full heart set, got {}",
        x.len()
    );
    let cfg = SupervisedConfig {
        steps: 60,
        lr: 0.1,
        seed: 20260724,
        loss: Loss::Bce,
        optimizer: Optimizer::adam(),
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let r = train_supervised(&heart_model(), &x, &y, "Z0", &cfg, BackendSel::Sim).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    // Throughput = rows × steps adjoint passes over the wall-clock — the
    // row-parallel batch path is what keeps this in a few seconds.
    let row_passes = (x.len() * cfg.steps) as f64;
    println!(
        "  heart smoke: final BCE loss {:.4}, train AUC {:.3}, {:.0} row-gradient passes in \
         {secs:.2}s ({:.0}/s, row-batched)",
        r.final_loss,
        r.final_auc,
        row_passes,
        row_passes / secs,
    );
    assert!(
        r.final_auc >= 0.80,
        "heart train AUC {:.3} below 0.80",
        r.final_auc
    );
}

#[test]
fn warm_start_makes_chunked_training_continue_rather_than_restart() {
    // Without `init_weights` an outer tuner cannot stop a trial early and
    // resume it: every call re-inits from the seed. With it, N chunks of k
    // steps must land in the same place as one run of N·k steps.
    let c = reuploader(2);
    let (x, y) = synthetic();
    let base = SupervisedConfig {
        steps: 12,
        lr: 0.25,
        seed: 9,
        loss: Loss::Mse,
        optimizer: Optimizer::Gd,
        ..Default::default()
    };

    let one_shot = train_supervised(&c, &x, &y, "Z0", &base, BackendSel::Sim).unwrap();

    // Same budget, split into three chunks, each warm-started from the last.
    let mut weights = None;
    let mut chunked = None;
    for _ in 0..3 {
        let cfg = SupervisedConfig {
            steps: 4,
            init_weights: weights.clone(),
            ..base.clone()
        };
        let r = train_supervised(&c, &x, &y, "Z0", &cfg, BackendSel::Sim).unwrap();
        weights = Some(r.weights.clone());
        chunked = Some(r);
    }
    let chunked = chunked.unwrap();

    for (name, v) in &one_shot.weights {
        let got = chunked.weights[name];
        assert!(
            (got - v).abs() < 1e-9,
            "weight {name}: chunked {got} vs one-shot {v}"
        );
    }
    assert!((chunked.final_loss - one_shot.final_loss).abs() < 1e-9);
}

#[test]
fn warm_start_accepts_a_partial_map() {
    // Symbols absent from the map fall back to the seeded init, so a caller
    // may pin just the weights it cares about.
    let c = reuploader(2);
    let (x, y) = synthetic();
    let mut partial = std::collections::HashMap::new();
    partial.insert("theta_0".to_string(), 0.75);
    let cfg = SupervisedConfig {
        steps: 0,
        init_weights: Some(partial),
        ..Default::default()
    };
    let r = train_supervised(&c, &x, &y, "Z0", &cfg, BackendSel::Sim).unwrap();
    assert!(
        (r.weights["theta_0"] - 0.75).abs() < 1e-12,
        "pinned weight was overwritten: {}",
        r.weights["theta_0"]
    );
    assert_eq!(r.weights.len(), 6, "other weights should still exist");
}
