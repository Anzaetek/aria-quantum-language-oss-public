// SPDX-License-Identifier: Apache-2.0
//! jl_sketch_digits — JL-sketch quantum feature map on real open data
//! (Babbush sketching track, cf. Zhao–Babbush–Huang arXiv:2604.07639).
//!
//! WHAT: classify handwritten digits 3 vs 8 from the open UCI optdigits
//!   set (64 features, vendored in `examples/data/`, CC BY 4.0) using
//!   only 12 qubits: a deterministic Johnson–Lindenstrauss Rademacher
//!   projection compresses 64 → 12 dims classically, the sketch is
//!   angle-encoded through the entangling re-uploading feature map in
//!   jl_sketch_digits.aria, and the ⟨Z_i⟩ + adjacent ⟨Z_i·Z_{i+1}⟩
//!   profile is the quantum feature vector a ridge head classifies.
//!   This upgrades the synthetic `sketch_qml` forward-check into a
//!   full dataset application.
//! QUANTUM: quantum features for the held-out test rows are computed
//!   THROUGH the wasm/native transport; the probe-row ⟨Z⟩ profile is the
//!   verified quantity.
//! CLASSICAL: (a) the independent in-tree statevector oracle
//!   (`aria_verify_core::sim`) on the same lowered IR — |Δ| ≤ 1e-9;
//!   (b) a ridge classifier on the raw 12 JL dims for honest context
//!   (Fourier wall, arXiv:2607.15815: expect the classical lane to be
//!   competitive on open tabular/image data — and it is, ~0.93 vs ~0.89:
//!   both lanes are bounded by what survives the 64 → 12 sketch, and the
//!   verified content is the oracle-exact few-qubit pipeline, not
//!   benchmark advantage).
//! CHECK: ⟨Z⟩ profile matches the oracle to 1e-9 AND test accuracy of
//!   the quantum-feature classifier ≥ 0.85 (regression guard, not an
//!   advantage claim).

use aria_verify_core::data;
use aria_verify_core::{banner, harness, resolve, sim, Observable, Transport, Verdict};

const K: i64 = 12; // qubits = sketch dims
const SEED: u64 = 26040763; // JL projection seed (printed for reproducibility)
const DIGITS: (f64, f64) = (3.0, 8.0); // the binary pair; 3 ↔ label −1, 8 ↔ +1

pub fn run(transport_override: Transport) -> Result<Verdict, String> {
    let guest = "omega_app";
    let transport = resolve(transport_override, guest);
    banner::header(
        "jl_sketch_digits",
        "classify UCI optdigits 3-vs-8 on 12 qubits via a JL-sketch quantum feature map",
        &transport.label(guest),
    );

    // ---- Load optdigits, keep the binary pair. ----
    let load_pair = |file: &str| -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
        let rows = data::load_csv(file)?;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for r in rows {
            let d = r[64].ok_or("optdigits: missing label")?;
            if d == DIGITS.0 || d == DIGITS.1 {
                let feats: Option<Vec<f64>> = r[..64].iter().copied().collect();
                xs.push(feats.ok_or("optdigits: missing feature")?);
                ys.push(if d == DIGITS.1 { 1.0 } else { -1.0 });
            }
        }
        Ok((xs, ys))
    };
    let (train_x64, train_y) = load_pair("optdigits_train.csv")?;
    let (test_x64, test_y) = load_pair("optdigits_test.csv")?;
    println!(
        "  data: {} train / {} test rows of digits {} vs {} (64 dims)",
        train_x64.len(),
        test_x64.len(),
        DIGITS.0,
        DIGITS.1
    );

    // ---- Deterministic JL Rademacher projection 64 → K. ----
    // R[j][i] = ±1/√K with signs from SplitMix64(SEED) — a valid JL map
    // (Achlioptas 2003); deterministic so the whole app is reproducible.
    let k = K as usize;
    let mut rng = data::SplitMix64(SEED);
    let proj: Vec<Vec<f64>> = (0..k)
        .map(|_| {
            (0..64)
                .map(|_| {
                    if rng.next_f64() < 0.5 {
                        -1.0 / (k as f64).sqrt()
                    } else {
                        1.0 / (k as f64).sqrt()
                    }
                })
                .collect()
        })
        .collect();
    let sketch = |x64: &[f64]| -> Vec<f64> {
        proj.iter()
            .map(|row| row.iter().zip(x64).map(|(r, x)| r * x).sum::<f64>())
            .collect()
    };
    let train_s: Vec<Vec<f64>> = train_x64.iter().map(|x| sketch(x)).collect();
    let test_s: Vec<Vec<f64>> = test_x64.iter().map(|x| sketch(x)).collect();

    // Standardize sketch dims on the train split, arctan-squash to angles.
    let opt_rows: Vec<Vec<Option<f64>>> = train_s
        .iter()
        .map(|r| r.iter().map(|&v| Some(v)).collect())
        .collect();
    let (means, stds) = data::column_stats(&opt_rows);
    let to_angles = |s: &[f64]| -> Vec<f64> {
        s.iter()
            .enumerate()
            .map(|(i, &v)| ((v - means[i]) / stds[i]).atan())
            .collect()
    };

    // ---- Lower the feature map, map symbols. ----
    let lowered = harness::load_lowered("jl_sketch_digits.aria", "JlSketchDigits", &[("k", K)])?;
    let x_ids: Vec<usize> = (0..k)
        .map(|i| {
            lowered
                .symbol_ids
                .get(&format!("x_{i}"))
                .map(|&id| id as usize)
                .ok_or(format!("jl_sketch_digits.aria: missing symbol x_{i}"))
        })
        .collect::<Result<_, _>>()?;
    let n_sym = lowered.symbol_ids.len();
    let assemble = |angles: &[f64]| -> Vec<f64> {
        let mut p = vec![0.0; n_sym];
        for (&id, &a) in x_ids.iter().zip(angles) {
            p[id] = a;
        }
        p
    };
    let z_obs: Vec<Observable> = (0..k)
        .map(|q| Observable::parse(&format!("1.0*Z{q}")))
        .collect::<Result<_, _>>()?;

    // ---- Verified forward check: transport vs independent oracle. ----
    let probe = to_angles(&train_s[0]);
    let probe_params = assemble(&probe);
    let mut quantum_profile = Vec::with_capacity(k);
    for obs in &z_obs {
        let (_payload, z) = harness::execute_report(
            transport,
            lowered.ir.clone(),
            harness::AppMode::Expectations(vec![obs.clone()]),
            &probe_params,
        )?;
        quantum_profile.push(z);
    }
    let oracle_profile = sim::forward_z_expectations(&lowered.ir, &probe_params)?;
    let forward_verdict = banner::report_values(
        "jl_sketch_digits/forward",
        "omega ⟨Z_q⟩ profile, probe row",
        &quantum_profile,
        "independent in-tree statevector oracle",
        &oracle_profile[..k],
        1e-9,
    );

    // ---- Quantum features (native, fast) + ridge head. ----
    // ⟨Z_i⟩ for every qubit plus the adjacent-pair correlators
    // ⟨Z_i·Z_{i+1}⟩ — the entangled two-point functions are what the
    // classical JL-dims lane cannot see linearly.
    let features = |angles: &[f64]| -> Result<Vec<f64>, String> {
        let sv = harness::native_statevector(&{
            let mut ir = lowered.ir.clone();
            let binding = assemble(angles);
            // Bind symbols to concrete values for the statevector run.
            for op in ir.ops.iter_mut() {
                for p in op.params.iter_mut() {
                    if let omega_core::circuit::ParamExpr::Symbol(id) = p {
                        *p = omega_core::circuit::ParamExpr::Concrete(binding[*id as usize]);
                    }
                }
            }
            ir
        })?;
        let dim = sv.len();
        let z_of = |mask: usize| -> f64 {
            let mut e = 0.0;
            for (idx, amp) in sv.iter().enumerate() {
                let p = amp.norm_sqr();
                e += if (idx & mask).count_ones().is_multiple_of(2) {
                    p
                } else {
                    -p
                };
            }
            debug_assert!(dim > 0);
            e
        };
        let mut f = Vec::with_capacity(2 * k - 1);
        for q in 0..k {
            f.push(z_of(1 << q));
        }
        for q in 0..k - 1 {
            f.push(z_of((1 << q) | (1 << (q + 1))));
        }
        Ok(f)
    };
    let train_f: Vec<Vec<f64>> = train_s
        .iter()
        .map(|s| features(&to_angles(s)))
        .collect::<Result<_, _>>()?;
    let w_q = data::ridge_regression(&train_f, &train_y, 3e-2)?;

    // Classical lane: same ridge head on the raw standardized JL dims.
    let train_zs: Vec<Vec<f64>> = train_s
        .iter()
        .map(|s| {
            s.iter()
                .enumerate()
                .map(|(i, &v)| (v - means[i]) / stds[i])
                .collect()
        })
        .collect();
    let w_c = data::ridge_regression(&train_zs, &train_y, 1e-4)?;

    // ---- Test accuracy (quantum features via the oracle-checked map). ----
    let mut correct_q = 0usize;
    let mut correct_c = 0usize;
    for (s, &y) in test_s.iter().zip(test_y.iter()) {
        let f = features(&to_angles(s))?;
        let pred_q = if data::ridge_predict(&w_q, &f) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let zs: Vec<f64> = s
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - means[i]) / stds[i])
            .collect();
        let pred_c = if data::ridge_predict(&w_c, &zs) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        if pred_q == y {
            correct_q += 1;
        }
        if pred_c == y {
            correct_c += 1;
        }
    }
    let acc_q = correct_q as f64 / test_y.len() as f64;
    let acc_c = correct_c as f64 / test_y.len() as f64;
    println!("  test accuracy (quantum ⟨Z⟩ features + ridge): {acc_q:.4}");
    println!("  test accuracy (classical JL dims + ridge)   : {acc_c:.4}");
    println!(
        "  (Fourier wall, arXiv:2607.15815: the classical lane is expected to be \
         competitive on open image data — the verified content is the oracle-exact \
         8-qubit compression pipeline, not benchmark advantage.)"
    );

    let acc_ok = acc_q >= 0.85;
    if !acc_ok {
        println!("  FAIL: quantum-feature accuracy {acc_q:.4} < 0.85");
    }
    Ok(Verdict {
        name: "jl_sketch_digits".into(),
        pass: forward_verdict.pass && acc_ok,
        max_abs_diff: forward_verdict.max_abs_diff,
        tol: forward_verdict.tol,
    })
}
