// SPDX-License-Identifier: Apache-2.0
//! butterfly_qnn — layer-wise butterfly QNN training (arXiv:2606.03517)
//! on the open UCI Cleveland heart dataset.
//!
//! WHAT: impute a masked clinical feature (`thalach`, max heart rate)
//!   from 8 other features under 30% MCAR missingness — the open-data
//!   stand-in for the paper's MIMIC-III clinical-imputation task
//!   (MIMIC-III needs credentialed access; Cleveland is CC BY 4.0 and
//!   vendored in `examples/data/`, see its README).
//! QUANTUM: the 8-qubit butterfly of RBS gates in butterfly_qnn.aria
//!   (12 = (n/2)·log₂n parameters, log₂n depth) inside the paper's
//!   hybrid sandwich — quantum features ⟨Z₀..Z₇⟩ feed a classical
//!   linear head ŷ = w·⟨Z⟩ + b (closed-form ridge refit each epoch).
//!   The head folds into a single gradient observable H_w = Σ wᵢ·Zᵢ,
//!   so QNN gradients stay one backend call. Staged protocol:
//!     stage A — the two 4-qubit sub-butterflies (θ₀..θ₇) train with
//!               adjoint gradients + Adam (the "simulator" stage);
//!     stage B — θ₀..θ₇ FROZEN, only the trailing coupling layer
//!               (θ₈..θ₁₁) trains, gradients from the PARALLELISED
//!               parameter-shift rule: one circuit execution per step
//!               yields all four gradients as commuting observables
//!               i·[G_k, H_w] (vs 16 executions for serial 4-term shifts).
//!   Inference on the masked rows then runs THROUGH the wasm guest.
//! CLASSICAL: (a) serial per-slot parameter-shift gradients — must equal
//!   the parallel ones to 1e-9 (the paper's Eq. 6 correctness claim);
//!   (b) mean-imputation and ridge-regression baselines for MSE context.
//! CHECK: max |parallel − serial| gradient ≤ 1e-9 AND quantum MSE ≤ mean-
//!   imputer MSE. Per the Fourier wall (arXiv:2607.15815) a classical
//!   ridge on an open tabular set may match or beat the QNN — the app
//!   prints that comparison honestly instead of claiming advantage; the
//!   verified content is the O(log n) training mechanics.

use std::collections::{HashMap, HashSet};

use aria_verify_core::data;
use aria_verify_core::{banner, harness, resolve, Observable, Transport, Verdict};
use omega_backend_statevector::StatevectorBackend;
use omega_core::executor::{Backend as _, PauliOp};
use omega_core::gradient::{compute_gradient_for, GradMethod};
use omega_core::parallel_shift::parallel_parameter_shift_gradient;
use omega_core::params::ParameterBinding;

/// Feature columns of heart_cleveland.csv used as inputs (by index):
/// age, sex, cp, trestbps, chol, restecg, exang, oldpeak.
const FEATURES: [usize; 8] = [0, 1, 2, 3, 4, 6, 8, 9];
/// Target column: thalach (max heart rate) — continuous, strongly
/// correlated with age/exang/oldpeak, the natural regression target.
const TARGET: usize = 7;
const MASK_RATE: f64 = 0.30;
const SEED: u64 = 20260722;
const N_QUBITS: usize = 8;

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "butterfly_qnn",
        "impute masked `thalach` (UCI heart, 30% MCAR) via layer-wise butterfly QNN; \
         parallel-shift gradients vs serial",
        &transport.label(guest),
    );

    // ---- Data: load, standardize, squash to angles, MCAR-mask. ----
    let rows = data::load_csv("heart_cleveland.csv")?;
    let (means, stds) = data::column_stats(&rows);
    let mut xs: Vec<Vec<f64>> = Vec::new(); // encoding angles per sample
    let mut zx: Vec<Vec<f64>> = Vec::new(); // standardized raw features (baselines)
    let mut ty: Vec<f64> = Vec::new(); // tanh-squashed standardized target
    for row in &rows {
        // The used columns have no missing cells in Cleveland ('?' only
        // in ca/thal, which we don't use) — skip defensively otherwise.
        let feats: Option<Vec<f64>> = FEATURES.iter().map(|&c| row[c]).collect();
        let (Some(feats), Some(target)) = (feats, row[TARGET]) else {
            continue;
        };
        let z: Vec<f64> = FEATURES
            .iter()
            .zip(&feats)
            .map(|(&c, &v)| (v - means[c]) / stds[c])
            .collect();
        // Angle encoding: arctan keeps angles in (−π/2, π/2) — bounded,
        // monotone, and outlier-tolerant.
        xs.push(z.iter().map(|&v| v.atan()).collect());
        zx.push(z);
        // Target squashed to (−1, 1); ALL methods (quantum + baselines)
        // fit and score in this space.
        ty.push(((target - means[TARGET]) / stds[TARGET]).tanh());
    }
    let n = xs.len();

    // MCAR mask at 30%: masked rows are the imputation TEST set.
    let mut rng = data::SplitMix64(SEED);
    let masked: Vec<bool> = (0..n).map(|_| rng.next_f64() < MASK_RATE).collect();
    let train_idx: Vec<usize> = (0..n).filter(|&i| !masked[i]).collect();
    let test_idx: Vec<usize> = (0..n).filter(|&i| masked[i]).collect();
    println!(
        "  data: {n} rows; {} observed (train), {} masked at {:.0}% MCAR (test); seed {SEED}",
        train_idx.len(),
        test_idx.len(),
        MASK_RATE * 100.0
    );

    // ---- Model: lower the shipped .aria, map symbols. ----
    let lowered = harness::load_lowered("butterfly_qnn.aria", "ButterflyQNN", &[])?;
    let sym = |name: &str| -> Result<u32, String> {
        lowered
            .symbol_ids
            .get(name)
            .copied()
            .ok_or(format!("butterfly_qnn.aria: missing symbol {name}"))
    };
    let x_ids: Vec<u32> = (0..N_QUBITS)
        .map(|i| sym(&format!("x_{i}")))
        .collect::<Result<_, _>>()?;
    let theta_ids: Vec<u32> = (0..12)
        .map(|k| sym(&format!("theta_{k}")))
        .collect::<Result<_, _>>()?;
    let sub_ids: HashSet<u32> = theta_ids[0..8].iter().copied().collect();
    let coupling_ids: HashSet<u32> = theta_ids[8..12].iter().copied().collect();

    let backend = StatevectorBackend::new();
    let z_obs: Vec<Observable> = (0..N_QUBITS as u32)
        .map(|q| Observable {
            terms: vec![(1.0, vec![(q, PauliOp::Z)])],
        })
        .collect();
    // The classical head folded into one gradient observable H_w = Σ wᵢZᵢ.
    let head_obs = |w: &[f64]| -> Observable {
        Observable {
            terms: (0..N_QUBITS as u32)
                .map(|q| (w[q as usize], vec![(q, PauliOp::Z)]))
                .collect(),
        }
    };

    let mut theta: HashMap<u32, f64> = HashMap::new();
    // Small deterministic init for the sub-butterflies; coupling starts
    // at exactly 0 (RBS(0) = identity) until stage B trains it.
    let mut init_rng = data::SplitMix64(SEED ^ 0xB77E_4F17);
    for &id in &theta_ids[0..8] {
        theta.insert(id, (init_rng.next_f64() * 2.0 - 1.0) * 0.3);
    }
    for &id in &theta_ids[8..12] {
        theta.insert(id, 0.0);
    }

    let bind = |theta: &HashMap<u32, f64>, x: &[f64]| -> ParameterBinding {
        let mut b = ParameterBinding::new();
        for (&id, &v) in theta {
            b.bind(id, v);
        }
        for (&id, &v) in x_ids.iter().zip(x.iter()) {
            b.bind(id, v);
        }
        b
    };
    // Quantum feature vector ⟨Z₀..Z₇⟩ — one forward sweep.
    let features = |theta: &HashMap<u32, f64>, x: &[f64]| -> Result<Vec<f64>, String> {
        backend
            .expectation_multi(&lowered.ir, &bind(theta, x), &z_obs)
            .map_err(|e| e.to_string())
    };
    // Closed-form ridge refit of the head on the train split.
    let fit_head = |theta: &HashMap<u32, f64>| -> Result<Vec<f64>, String> {
        let f: Vec<Vec<f64>> = train_idx
            .iter()
            .map(|&i| features(theta, &xs[i]))
            .collect::<Result<_, _>>()?;
        let t: Vec<f64> = train_idx.iter().map(|&i| ty[i]).collect();
        data::ridge_regression(&f, &t, 1e-4)
    };
    let head_mse =
        |theta: &HashMap<u32, f64>, head: &[f64], idx: &[usize]| -> Result<f64, String> {
            let mut s = 0.0;
            for &i in idx {
                let r = data::ridge_predict(head, &features(theta, &xs[i])?) - ty[i];
                s += r * r;
            }
            Ok(s / idx.len() as f64)
        };

    // Adam step shared by both stages.
    struct Adam {
        m: HashMap<u32, f64>,
        v: HashMap<u32, f64>,
        t: f64,
    }
    impl Adam {
        fn new() -> Self {
            Adam {
                m: HashMap::new(),
                v: HashMap::new(),
                t: 0.0,
            }
        }
        fn step(&mut self, theta: &mut HashMap<u32, f64>, grads: &HashMap<u32, f64>, lr: f64) {
            self.t += 1.0;
            let (b1, b2, eps) = (0.9, 0.999, 1e-8);
            for (&id, &g) in grads {
                let m = self.m.entry(id).or_insert(0.0);
                *m = b1 * *m + (1.0 - b1) * g;
                let v = self.v.entry(id).or_insert(0.0);
                *v = b2 * *v + (1.0 - b2) * g * g;
                let m_hat = *m / (1.0 - b1.powf(self.t));
                let v_hat = *v / (1.0 - b2.powf(self.t));
                *theta.get_mut(&id).unwrap() -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }

    // ---- Stage A: sub-butterflies (θ₀..θ₇), adjoint + Adam. ----
    // Each epoch: refit head (closed form), then one adjoint pass per
    // sample against H_w with the residual chain rule
    //   dMSE/dθ = (2/N)·Σ r_s · d⟨H_w⟩/dθ.
    let epochs_a = 30;
    let mut adam = Adam::new();
    let mut head = fit_head(&theta)?;
    for epoch in 0..epochs_a {
        if epoch > 0 {
            head = fit_head(&theta)?;
        }
        let hw = head_obs(&head);
        let mut acc: HashMap<u32, f64> = HashMap::new();
        for &i in &train_idx {
            let b = bind(&theta, &xs[i]);
            let r = data::ridge_predict(&head, &features(&theta, &xs[i])?) - ty[i];
            let grads = compute_gradient_for(
                &backend,
                &lowered.ir,
                &b,
                &hw,
                &GradMethod::Adjoint,
                Some(&sub_ids),
            )
            .map_err(|e| e.to_string())?;
            for (id, g) in grads {
                *acc.entry(id).or_insert(0.0) += 2.0 * r * g / train_idx.len() as f64;
            }
        }
        adam.step(&mut theta, &acc, 0.05);
    }
    head = fit_head(&theta)?;
    println!(
        "  stage A: trained θ0..θ7 (sub-butterflies, adjoint+Adam, {epochs_a} epochs); \
         train MSE = {:.5}",
        head_mse(&theta, &head, &train_idx)?
    );

    // ---- Gradient cross-check: parallel vs serial on the coupling layer. ----
    // (First training point, coupling layer slightly displaced so the
    // gradients are non-trivial.)
    let mut theta_probe = theta.clone();
    for (k, &id) in theta_ids[8..12].iter().enumerate() {
        theta_probe.insert(id, 0.2 + 0.1 * k as f64);
    }
    let hw = head_obs(&head);
    let probe_bind = bind(&theta_probe, &xs[train_idx[0]]);
    let (par, report) = parallel_parameter_shift_gradient(
        &backend,
        &lowered.ir,
        &probe_bind,
        &hw,
        Some(&coupling_ids),
    )
    .map_err(|e| e.to_string())?;
    let ser = compute_gradient_for(
        &backend,
        &lowered.ir,
        &probe_bind,
        &hw,
        &GradMethod::ParameterShift,
        Some(&coupling_ids),
    )
    .map_err(|e| e.to_string())?;
    let par_v: Vec<f64> = par.iter().map(|(_, g)| *g).collect();
    let ser_v: Vec<f64> = ser.iter().map(|(_, g)| *g).collect();
    println!(
        "  parallel shift: {} coupling gradients from {} circuit execution(s) \
         (block of {} RBS gates); serial 4-term rule needs {} executions",
        par_v.len(),
        report.circuit_executions,
        report.block_gates,
        4 * 4
    );
    let grad_verdict = banner::report_values(
        "butterfly_qnn/gradients",
        "parallel commuting-block rule, 1 execution",
        &par_v,
        "serial per-slot 4-term Givens shifts",
        &ser_v,
        1e-9,
    );

    // ---- Stage B: coupling layer only (θ₈..θ₁₁), parallel rule + Adam. ----
    let epochs_b = 30;
    let mut adam_b = Adam::new();
    let mut parallel_execs = 0usize;
    for _ in 0..epochs_b {
        head = fit_head(&theta)?;
        let hw = head_obs(&head);
        let mut acc: HashMap<u32, f64> = HashMap::new();
        for &i in &train_idx {
            let b = bind(&theta, &xs[i]);
            let r = data::ridge_predict(&head, &features(&theta, &xs[i])?) - ty[i];
            let (grads, rep) = parallel_parameter_shift_gradient(
                &backend,
                &lowered.ir,
                &b,
                &hw,
                Some(&coupling_ids),
            )
            .map_err(|e| e.to_string())?;
            parallel_execs += rep.circuit_executions;
            for (id, g) in grads {
                *acc.entry(id).or_insert(0.0) += 2.0 * r * g / train_idx.len() as f64;
            }
        }
        adam_b.step(&mut theta, &acc, 0.05);
    }
    head = fit_head(&theta)?;
    let serial_execs = epochs_b * train_idx.len() * 4 * 4;
    println!(
        "  stage B: trained θ8..θ11 (coupling, frozen sub-butterflies, {epochs_b} epochs); \
         train MSE = {:.5}",
        head_mse(&theta, &head, &train_idx)?
    );
    println!(
        "  execution count (stage B gradients): parallel = {parallel_execs}, \
         serial equivalent = {serial_execs}  ({}x fewer)",
        serial_execs / parallel_execs.max(1)
    );

    // ---- Inference on the masked rows THROUGH wasm. ----
    let mut q_sq_err = 0.0;
    for &i in &test_idx {
        let mut params = vec![0.0; lowered.symbol_ids.len()];
        for (&id, &v) in &theta {
            params[id as usize] = v;
        }
        for (&id, &v) in x_ids.iter().zip(xs[i].iter()) {
            params[id as usize] = v;
        }
        let mut f = Vec::with_capacity(N_QUBITS);
        for obs in &z_obs {
            let (_payload, z) = harness::execute_report(
                transport,
                lowered.ir.clone(),
                harness::AppMode::Expectations(vec![obs.clone()]),
                &params,
            )?;
            f.push(z);
        }
        let pred = data::ridge_predict(&head, &f);
        q_sq_err += (pred - ty[i]) * (pred - ty[i]);
    }
    let mse_q = q_sq_err / test_idx.len() as f64;

    // ---- Classical baselines on the same split/space. ----
    let train_z: Vec<Vec<f64>> = train_idx.iter().map(|&i| zx[i].clone()).collect();
    let train_t: Vec<f64> = train_idx.iter().map(|&i| ty[i]).collect();
    let mean_t = train_t.iter().sum::<f64>() / train_t.len() as f64;
    let mse_mean = test_idx
        .iter()
        .map(|&i| (mean_t - ty[i]) * (mean_t - ty[i]))
        .sum::<f64>()
        / test_idx.len() as f64;
    let w = data::ridge_regression(&train_z, &train_t, 1e-6)?;
    let mse_ridge = test_idx
        .iter()
        .map(|&i| {
            let p = data::ridge_predict(&w, &zx[i]);
            (p - ty[i]) * (p - ty[i])
        })
        .sum::<f64>()
        / test_idx.len() as f64;

    println!("  imputation MSE on the {} masked rows:", test_idx.len());
    println!("    quantum butterfly QNN : {mse_q:.5}");
    println!("    classical mean-imputer: {mse_mean:.5}");
    println!("    classical ridge       : {mse_ridge:.5}");
    println!(
        "  (Fourier wall, arXiv:2607.15815: open tabular data is expected to be \
         classically matchable — the verified claim here is the O(log n) \
         training mechanics, not benchmark advantage.)"
    );

    // Verdict: gradients must match AND the QNN must beat mean imputation.
    let beats_mean = mse_q <= mse_mean;
    if !beats_mean {
        println!("  FAIL: QNN MSE {mse_q:.5} did not beat mean imputation {mse_mean:.5}");
    }
    Ok(Verdict {
        name: "butterfly_qnn".into(),
        pass: grad_verdict.pass && beats_mean,
        max_abs_diff: grad_verdict.max_abs_diff,
        tol: grad_verdict.tol,
    })
}
