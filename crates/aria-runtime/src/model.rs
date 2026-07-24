// SPDX-License-Identifier: Apache-2.0
//! Trained-model interchange format for supervised Aria circuits.
//!
//! A [`TrainedModel`] is a self-contained, human-readable JSON artefact: it
//! embeds the `.aria` source, the circuit name and integer template params, the
//! feature-prefix / readout / loss convention, the weight symbol names paired
//! with their trained values, and the affine head. `predict` re-parses and
//! re-lowers the embedded source, validates that the saved weight names still
//! match the circuit's symbols, binds features + weights per row, and applies
//! the head — so a trained model round-trips to the same scores (to f64-JSON
//! precision, ~15-17 significant figures) and needs no external file at
//! inference time.
//!
//! Serialisation is hand-rolled over `serde_json::Value` (no serde-derive
//! dependency): the schema is small and stable, and every field is validated on
//! load with a clear error rather than a deserialize panic.

use serde_json::{json, Value};

use aria_core::ast::parse_aria;
use omega_core::executor::Observable;
use omega_core::params::ParameterBinding;

use crate::lower::lower;
use crate::train_supervised::{Loss, SupervisedResult};
use crate::BackendSel;

/// Provenance recorded alongside the weights.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelMetadata {
    pub seed: u64,
    pub steps: usize,
    /// aria-runtime version that trained the model.
    pub aria_version: String,
}

/// A trained supervised model, fully self-contained.
#[derive(Clone, Debug)]
pub struct TrainedModel {
    /// The complete `.aria` program text (embedded so the model is portable).
    pub aria_source: String,
    /// Circuit name to instantiate from `aria_source`.
    pub circuit: String,
    /// Integer template parameters passed to `instantiate`.
    pub int_params: Vec<(String, i64)>,
    /// Feature-symbol prefix (features are `{feature_prefix}_{i}`).
    pub feature_prefix: String,
    /// Readout observable string (e.g. `"Z0"`).
    pub readout: String,
    pub loss: Loss,
    /// Weight symbol names, sorted — the authoritative order for `weights`.
    pub symbol_order: Vec<String>,
    /// Trained weight values, index-aligned with `symbol_order`.
    pub weights: Vec<f64>,
    /// Affine head `(s, b)`: prediction is `s·⟨O⟩ + b` (MSE) or
    /// `σ(s·⟨O⟩ + b)` (BCE).
    pub head: (f64, f64),
    pub metadata: ModelMetadata,
}

impl TrainedModel {
    /// Assemble a model from a finished [`SupervisedResult`] plus the training
    /// context the caller (the CLI) already holds.
    #[allow(clippy::too_many_arguments)]
    pub fn from_result(
        aria_source: String,
        circuit: String,
        int_params: Vec<(String, i64)>,
        feature_prefix: String,
        readout: String,
        loss: Loss,
        result: &SupervisedResult,
        seed: u64,
        steps: usize,
    ) -> Self {
        let mut symbol_order: Vec<String> = result.weights.keys().cloned().collect();
        symbol_order.sort();
        let weights: Vec<f64> = symbol_order.iter().map(|n| result.weights[n]).collect();
        Self {
            aria_source,
            circuit,
            int_params,
            feature_prefix,
            readout,
            loss,
            symbol_order,
            weights,
            head: result.head,
            metadata: ModelMetadata {
                seed,
                steps,
                aria_version: env!("CARGO_PKG_VERSION").to_string(),
            },
        }
    }

    /// Serialise to pretty JSON.
    pub fn to_json(&self) -> String {
        let loss = match self.loss {
            Loss::Mse => "mse",
            Loss::Bce => "bce",
        };
        let int_params: Vec<Value> = self.int_params.iter().map(|(k, v)| json!([k, v])).collect();
        let v = json!({
            "format": "aria-trained-model",
            "version": 1,
            "circuit": self.circuit,
            "int_params": int_params,
            "feature_prefix": self.feature_prefix,
            "readout": self.readout,
            "loss": loss,
            "symbol_order": self.symbol_order,
            "weights": self.weights,
            "head": [self.head.0, self.head.1],
            "metadata": {
                "seed": self.metadata.seed,
                "steps": self.metadata.steps,
                "aria_version": self.metadata.aria_version,
            },
            "aria_source": self.aria_source,
        });
        serde_json::to_string_pretty(&v).expect("model serialises")
    }

    /// Parse from JSON, validating the schema.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(s).map_err(|e| format!("invalid model JSON: {e}"))?;
        let get = |k: &str| v.get(k).ok_or_else(|| format!("model missing field '{k}'"));
        let as_str = |val: &Value, k: &str| -> Result<String, String> {
            val.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("model field '{k}' is not a string"))
        };

        if v.get("format").and_then(Value::as_str) != Some("aria-trained-model") {
            return Err("not an aria-trained-model JSON".into());
        }
        let circuit = as_str(get("circuit")?, "circuit")?;
        let int_params = get("int_params")?
            .as_array()
            .ok_or("'int_params' is not an array")?
            .iter()
            .map(|e| {
                let a = e.as_array().ok_or("int_params entry not [name, value]")?;
                let name = a
                    .first()
                    .and_then(Value::as_str)
                    .ok_or("int_params name not a string")?;
                let val = a
                    .get(1)
                    .and_then(Value::as_i64)
                    .ok_or("int_params value not an integer")?;
                Ok((name.to_string(), val))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let feature_prefix = as_str(get("feature_prefix")?, "feature_prefix")?;
        let readout = as_str(get("readout")?, "readout")?;
        let loss = Loss::parse(get("loss")?.as_str().ok_or("'loss' not a string")?)?;
        let symbol_order = get("symbol_order")?
            .as_array()
            .ok_or("'symbol_order' not an array")?
            .iter()
            .map(|e| as_str(e, "symbol_order entry"))
            .collect::<Result<Vec<_>, String>>()?;
        let weights = get("weights")?
            .as_array()
            .ok_or("'weights' not an array")?
            .iter()
            .map(|e| e.as_f64().ok_or_else(|| "weight not a number".to_string()))
            .collect::<Result<Vec<_>, String>>()?;
        if weights.len() != symbol_order.len() {
            return Err(format!(
                "weights ({}) and symbol_order ({}) length mismatch",
                weights.len(),
                symbol_order.len()
            ));
        }
        let head_arr = get("head")?.as_array().ok_or("'head' not an array")?;
        let head = (
            head_arr
                .first()
                .and_then(Value::as_f64)
                .ok_or("head[0] not a number")?,
            head_arr
                .get(1)
                .and_then(Value::as_f64)
                .ok_or("head[1] not a number")?,
        );
        let md = get("metadata")?;
        let metadata = ModelMetadata {
            seed: md.get("seed").and_then(Value::as_u64).unwrap_or(0),
            steps: md.get("steps").and_then(Value::as_u64).unwrap_or(0) as usize,
            aria_version: md
                .get("aria_version")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
        };
        let aria_source = as_str(get("aria_source")?, "aria_source")?;

        Ok(Self {
            aria_source,
            circuit,
            int_params,
            feature_prefix,
            readout,
            loss,
            symbol_order,
            weights,
            head,
            metadata,
        })
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.to_json()).map_err(|e| format!("write {path}: {e}"))
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        Self::from_json(&s)
    }

    /// Predict on a feature matrix. Re-lowers the embedded circuit, validates
    /// the saved weight names against its symbols, binds features + weights per
    /// row, and applies the affine head. Returns the model output per row:
    /// `s·⟨O⟩ + b` (MSE) or the probability `σ(s·⟨O⟩ + b)` (BCE).
    pub fn predict(&self, x: &[Vec<f64>], sel: BackendSel) -> Result<Vec<f64>, String> {
        if x.is_empty() {
            return Ok(Vec::new());
        }
        let backend = crate::run::make_backend(sel)?;
        let int_refs: Vec<(&str, i64)> = self
            .int_params
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        let circuit = parse_aria(&self.aria_source)?.instantiate(&self.circuit, &int_refs)?;
        let low = lower(&circuit)?;
        let obs = Observable::parse(&self.readout)?;

        // Resolve feature ids (x_0..x_{d-1}) and weight ids (by saved name).
        let d = x[0].len();
        let feat_prefix = format!("{}_", self.feature_prefix);
        let mut feature_ids = vec![0u32; d];
        let mut have = vec![false; d];
        for (name, &id) in &low.symbol_ids {
            if let Some(rest) = name.strip_prefix(&feat_prefix) {
                if let Ok(i) = rest.parse::<usize>() {
                    if i < d {
                        feature_ids[i] = id;
                        have[i] = true;
                    }
                }
            }
        }
        if let Some(i) = have.iter().position(|ok| !ok) {
            return Err(format!(
                "model circuit missing feature symbol '{feat_prefix}{i}' for {d}-column data"
            ));
        }
        let weight_ids: Vec<u32> = self
            .symbol_order
            .iter()
            .map(|n| {
                low.symbol_ids.get(n).copied().ok_or_else(|| {
                    format!(
                        "model weight '{n}' is not a symbol of circuit '{}'",
                        self.circuit
                    )
                })
            })
            .collect::<Result<_, String>>()?;

        // Bind rows and batch-evaluate ⟨O⟩.
        let bindings: Vec<ParameterBinding> = x
            .iter()
            .map(|row| {
                let mut b = ParameterBinding::new();
                for (i, &id) in feature_ids.iter().enumerate() {
                    b.bind(id, row[i]);
                }
                for (&id, &w) in weight_ids.iter().zip(&self.weights) {
                    b.bind(id, w);
                }
                b
            })
            .collect();
        let refs: Vec<&ParameterBinding> = bindings.iter().collect();
        let z = backend
            .expectation_batch(&low.ir, &refs, &obs)
            .map_err(|e| e.to_string())?;

        let (s, b) = self.head;
        Ok(z.iter()
            .map(|&zi| match self.loss {
                Loss::Mse => s * zi + b,
                Loss::Bce => 1.0 / (1.0 + (-(s * zi + b)).exp()),
            })
            .collect())
    }
}
