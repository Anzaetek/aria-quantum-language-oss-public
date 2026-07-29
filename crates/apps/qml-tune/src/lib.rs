// SPDX-License-Identifier: Apache-2.0
//! qml_tune — meta-parameter tuning of a fixed Aria circuit with `aria-tune`.
//!
//! WHAT: search `{n, L, lr, optimizer}` for the angle-encoded
//!   StronglyEntangling classifier in `qml_tune.aria`, on a SYNTHETIC
//!   two-class dataset (no external data — repo eval rules).
//! QUANTUM: each trial instantiates the template at its own `(n, L)`, then
//!   trains it with `aria_runtime::train_supervised` (exact adjoint
//!   gradients) against `⟨Z0⟩`.
//! TUNING: an `aria_tune::Study` drives the loop — TPE proposes, the trial
//!   reports its validation AUC every chunk, and a `MedianPruner` stops the
//!   hopeless ones. Pruning is *real*, not retrospective: training runs in
//!   chunks warm-started from the previous chunk's weights
//!   (`SupervisedConfig::init_weights`), so a pruned trial genuinely never
//!   pays for its remaining epochs.
//! CLASSICAL: the same study re-run with a `RandomSampler` on identical
//!   seeds and budget — the baseline TPE has to beat.
//! CHECK: best accuracy ≥ 0.85, at least one trial pruned, and TPE ≥ random.
//!
//! The circuit TOPOLOGY is fixed. Only meta-parameters are tuned; searching
//! over ansatz families or entanglement patterns is deliberately out of scope
//! for this crate.

use std::collections::HashMap;

use aria_runtime::{train_supervised, BackendSel, Loss, Optimizer, SupervisedConfig, TrainedModel};
use aria_tune::{
    Direction, MedianPruner, RandomSampler, Sampler, Space, Study, TpeSampler, TrialState,
};
use aria_verify_core::{banner, harness, resolve, Transport, Verdict};

/// Widest circuit the search may propose — the synthetic set is generated at
/// this width and sliced down per trial.
const MAX_QUBITS: usize = 8;
const N_TRIALS: usize = 18;
/// Training chunks per trial; the pruner gets a say after each.
const CHUNKS: usize = 4;
const STEPS_PER_CHUNK: usize = 6;
const SEED: u64 = 20260729;

/// Deterministic SplitMix64 — the harness must not depend on a global RNG.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Synthetic two-class problem: the label is the sign of a fixed linear
/// projection of the features, so it is learnable but not trivially so, and
/// every feature carries some signal.
fn synthetic(n_rows: usize, width: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Rng(seed);
    let w: Vec<f64> = (0..width).map(|i| 1.0 - 0.15 * i as f64).collect();
    let mut xs = Vec::with_capacity(n_rows);
    let mut ys = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let row: Vec<f64> = (0..width).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let s: f64 = row.iter().zip(&w).map(|(a, b)| a * b).sum();
        ys.push(if s >= 0.0 { 1.0 } else { 0.0 });
        xs.push(row);
    }
    (xs, ys)
}

/// Take the first `width` columns of each row.
fn slice_width(rows: &[Vec<f64>], width: usize) -> Vec<Vec<f64>> {
    rows.iter().map(|r| r[..width].to_vec()).collect()
}

/// Accuracy of the trained affine head `(s, b)` on `⟨Z0⟩`.
fn accuracy(preds: &[f64], y: &[f64]) -> f64 {
    let hit = preds
        .iter()
        .zip(y)
        .filter(|(p, t)| (**p >= 0.5) == (**t >= 0.5))
        .count();
    hit as f64 / y.len().max(1) as f64
}

/// The search space. Meta-parameters only — no topology dimensions.
fn space() -> Space {
    Space::new()
        .categorical("n", &["4", "6", "8"])
        .int("L", 1, 3, 1)
        .log_float("lr", 1e-3, 3e-1, 6)
        .categorical("optimizer", &["gd", "adam"])
}

/// Outcome of one study.
struct StudyResult {
    best_acc: f64,
    pruned: usize,
    csv_rows: usize,
}

/// Run one full study with `sampler`, returning its best accuracy and how
/// many trials the pruner stopped.
fn run_study(
    sampler: Box<dyn Sampler>,
    src: &str,
    x_all: &[Vec<f64>],
    y: &[f64],
) -> Result<StudyResult, String> {
    let n_train = (x_all.len() * 3) / 4;
    let mut study = Study::new(space(), Direction::Maximize)
        .with_sampler(sampler)
        .with_pruner(Box::new(MedianPruner::new(3, 1)));

    for _ in 0..N_TRIALS {
        let t = study.ask();
        let n: usize = t.cat("n").unwrap_or("4").parse().unwrap_or(4);
        let l = t.int("L").unwrap_or(1);
        let lr = t.float("lr").unwrap_or(0.1);
        let opt = match t.cat("optimizer") {
            Some("adam") => Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            _ => Optimizer::Gd,
        };

        // Each trial instantiates the FIXED template at its own (n, L).
        // (`harness::load_circuit` is gated behind the `remote` feature, so
        // parse + instantiate directly.)
        let circuit = aria_core::ast::parse_aria(src)?
            .instantiate("QmlTune", &[("n", n as i64), ("L", l)])?;
        let src = src.to_string();
        let xs = slice_width(x_all, n);
        let (xtr, ytr) = (&xs[..n_train], &y[..n_train]);
        let (xva, yva) = (&xs[n_train..], &y[n_train..]);

        // Chunked training, warm-started — so an early prune really does skip
        // the remaining chunks' compute.
        let mut weights: Option<HashMap<String, f64>> = None;
        let mut acc = 0.0;
        for chunk in 0..CHUNKS {
            let cfg = SupervisedConfig {
                steps: STEPS_PER_CHUNK,
                lr,
                seed: SEED,
                loss: Loss::Bce,
                optimizer: opt,
                init_weights: weights.clone(),
                ..Default::default()
            };
            let r = train_supervised(&circuit, xtr, ytr, "Z0", &cfg, BackendSel::Sim)?;
            weights = Some(r.weights.clone());

            // Validation accuracy through the trained head. TrainedModel
            // re-instantiates from source, so it also proves the trial's
            // (n, L) round-trips rather than only living in memory.
            let model = TrainedModel::from_result(
                src.clone(),
                "QmlTune".to_string(),
                vec![("n".to_string(), n as i64), ("L".to_string(), l)],
                "x".to_string(),
                "Z0".to_string(),
                Loss::Bce,
                &r,
                SEED,
                STEPS_PER_CHUNK,
            );
            let preds = model.predict(xva, BackendSel::Sim)?;
            acc = accuracy(&preds, yva);
            // Gradient health proxy: how much the loss actually moved.
            let slope = r
                .loss_history
                .first()
                .map(|f| (f - r.final_loss).abs())
                .unwrap_or(0.0);
            study.report(t.id, chunk, acc, &[("grad_norm".to_string(), slope)]);
            if study.should_prune(t.id) {
                break;
            }
        }
        study.tell(t.id, acc);
    }

    let best_acc = study
        .best()
        .and_then(|t| t.state.score())
        .unwrap_or(f64::NEG_INFINITY);
    let csv_rows = study.to_csv().trim_end().split('\n').count() - 1;
    Ok(StudyResult {
        best_acc,
        pruned: study
            .trials()
            .iter()
            .filter(|t| matches!(t.state, TrialState::Pruned))
            .count(),
        csv_rows,
    })
}

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "qml_tune",
        "tune {n, L, lr, optimizer} of a FIXED circuit template with aria-tune; \
         TPE vs random on the same budget",
        &transport.label(guest),
    );

    let (x_all, y) = synthetic(96, MAX_QUBITS, SEED);

    let src = std::fs::read_to_string(harness::aria_path("qml_tune.aria"))
        .map_err(|e| format!("read qml_tune.aria: {e}"))?;
    let tpe = run_study(Box::new(TpeSampler::new(SEED)), &src, &x_all, &y)?;
    let rnd = run_study(Box::new(RandomSampler::new(SEED)), &src, &x_all, &y)?;

    println!("qml_tune_trials {N_TRIALS}");
    println!("qml_tune_best_acc {:.4}", tpe.best_acc);
    println!("qml_tune_random_best_acc {:.4}", rnd.best_acc);
    println!("qml_tune_pruned {}", tpe.pruned);
    println!("qml_tune_csv_rows {}", tpe.csv_rows);
    println!(
        "qml_tune_tpe_ge_random {}",
        u8::from(tpe.best_acc >= rnd.best_acc)
    );

    let pass = tpe.best_acc >= 0.85
        && tpe.pruned >= 1
        && tpe.best_acc >= rnd.best_acc
        && tpe.csv_rows == N_TRIALS;
    println!("  QUANTUM   (tpe best accuracy): {:+.10}", tpe.best_acc);
    println!("  CLASSICAL (random best accuracy): {:+.10}", rnd.best_acc);
    Ok(Verdict {
        name: "qml_tune".to_string(),
        pass,
        max_abs_diff: (0.85 - tpe.best_acc).max(0.0),
        tol: 0.85,
    })
}
