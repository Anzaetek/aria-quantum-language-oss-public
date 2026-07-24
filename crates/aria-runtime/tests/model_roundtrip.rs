// SPDX-License-Identifier: Apache-2.0
//! The trained-model interchange format round-trips: train → save → load →
//! identical scores, and JSON survives a full serialise/parse cycle.

use aria_core::ast::parse_aria;
use aria_runtime::{train_supervised, BackendSel, Loss, Optimizer, SupervisedConfig, TrainedModel};

const SRC: &str = "circuit C() {\n  qreg q[1]\n  let theta = symbolic[6]\n  let x = symbolic[1]\n\
  apply RY(x[0]) on q[0]\n  apply RZ(theta[0]) on q[0]\n  apply RY(theta[1]) on q[0]\n  \
  apply RZ(theta[2]) on q[0]\n  apply RY(x[0]) on q[0]\n  apply RZ(theta[3]) on q[0]\n  \
  apply RY(theta[4]) on q[0]\n  apply RZ(theta[5]) on q[0]\n}\n";

fn dataset() -> (Vec<Vec<f64>>, Vec<f64>) {
    let xs: Vec<f64> = (0..50)
        .map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / 50.0)
        .collect();
    let x = xs.iter().map(|&v| vec![v * 1.5]).collect();
    let y = xs
        .iter()
        .map(|&v| if v >= 0.0 { 1.0 } else { 0.0 })
        .collect();
    (x, y)
}

fn trained_model(loss: Loss) -> (TrainedModel, Vec<Vec<f64>>) {
    let circuit = parse_aria(SRC).unwrap().instantiate("C", &[]).unwrap();
    let (x, y) = dataset();
    let cfg = SupervisedConfig {
        steps: 60,
        lr: 0.15,
        seed: 11,
        loss,
        optimizer: Optimizer::adam(),
        ..Default::default()
    };
    let result = train_supervised(&circuit, &x, &y, "Z0", &cfg, BackendSel::Sim).unwrap();
    let model = TrainedModel::from_result(
        SRC.to_string(),
        "C".to_string(),
        vec![],
        "x".to_string(),
        "Z0".to_string(),
        loss,
        &result,
        cfg.seed,
        cfg.steps,
    );
    (model, x)
}

#[test]
fn save_load_predict_is_identical() {
    for loss in [Loss::Mse, Loss::Bce] {
        let (model, x) = trained_model(loss);
        let path = format!("{}/model_{loss:?}.json", env!("CARGO_TARGET_TMPDIR"));

        let before = model.predict(&x, BackendSel::Sim).unwrap();
        model.save(&path).unwrap();
        let reloaded = TrainedModel::load(&path).unwrap();
        let after = reloaded.predict(&x, BackendSel::Sim).unwrap();

        // Scores reproduce to full f64-JSON precision (serde_json keeps ~15-17
        // significant figures, not bit-exact).
        assert_eq!(before.len(), after.len());
        for (i, (a, b)) in before.iter().zip(&after).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "{loss:?} row {i}: {a} != {b} after save/load"
            );
        }
    }
}

#[test]
fn json_round_trips_all_fields() {
    let (model, _) = trained_model(Loss::Bce);
    let json = model.to_json();
    let parsed = TrainedModel::from_json(&json).unwrap();

    assert_eq!(parsed.circuit, model.circuit);
    assert_eq!(parsed.feature_prefix, model.feature_prefix);
    assert_eq!(parsed.readout, model.readout);
    assert_eq!(parsed.loss, model.loss);
    assert_eq!(parsed.symbol_order, model.symbol_order);
    assert_eq!(parsed.weights.len(), model.weights.len());
    for (a, b) in parsed.weights.iter().zip(&model.weights) {
        assert!((a - b).abs() < 1e-12, "weight drift: {a} vs {b}");
    }
    assert!((parsed.head.0 - model.head.0).abs() < 1e-12);
    assert!((parsed.head.1 - model.head.1).abs() < 1e-12);
    assert_eq!(parsed.metadata.seed, model.metadata.seed);
    assert_eq!(parsed.metadata.steps, model.metadata.steps);
    assert_eq!(parsed.aria_source, model.aria_source);
}

#[test]
fn from_json_rejects_foreign_and_malformed() {
    assert!(TrainedModel::from_json("not json").is_err());
    assert!(TrainedModel::from_json(r#"{"format":"something-else"}"#).is_err());
    // Right format tag but missing required fields.
    assert!(TrainedModel::from_json(r#"{"format":"aria-trained-model"}"#).is_err());
}

#[test]
fn predict_rejects_wrong_feature_count() {
    let (model, _) = trained_model(Loss::Mse);
    // Circuit declares x[1]; give 2-column rows.
    let x = vec![vec![0.1, 0.2]];
    let err = model.predict(&x, BackendSel::Sim).unwrap_err();
    assert!(err.contains("missing feature symbol"), "err: {err}");
}
