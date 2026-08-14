//! `omega-run --qubo problem.json` — QAOA-backed QUBO solver.
//!
//! Pipeline: parse JSON → Qubo → Ising → qaoa_circuit(depth) → optimizer →
//! sample → rank bit-strings by QUBO value. Emits ranked results as text or
//! JSON depending on `--format`.

use std::collections::HashMap;
use std::fs;

use serde_json::{json, Value};

use omega_backend_statevector::StatevectorBackend;
use omega_core::cmaes::CmaEs;
use omega_core::executor::{Backend, ExecConfig, ExecResult, MidCircuitMode};
use omega_core::gradient::{compute_gradient, GradMethod};
use omega_core::params::ParameterBinding;
use omega_core::qaoa::qaoa_circuit;
use omega_core::qubo::{IsingModel, Qubo};

use crate::serialize::Format;

/// Options parsed out of argv for the `--qubo` mode.
pub struct QuboOptions {
    pub path: String,
    pub depth: usize,
    pub shots: u32,
    pub seed: Option<u64>,
    pub optimizer: Optimizer,
    pub top_k: usize,
    pub max_iters: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Optimizer {
    CmaEs,
    Gradient,
}

pub fn run_qubo(opts: QuboOptions, format: Format) {
    let info = |msg: String| {
        if format.is_machine() {
            eprintln!("{}", msg);
        } else {
            println!("{}", msg);
        }
    };

    // --- Parse input ---
    let text = fs::read_to_string(&opts.path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", opts.path, e);
        std::process::exit(1);
    });
    let qubo = parse_qubo_json(&text).unwrap_or_else(|e| {
        eprintln!("QUBO parse error: {}", e);
        std::process::exit(1);
    });

    info(format!(
        "QUBO: n={}, {} nonzero entries, depth={}, optimizer={:?}, shots={}",
        qubo.n,
        qubo.entries.len(),
        opts.depth,
        opts.optimizer,
        opts.shots
    ));

    // --- Build QAOA circuit ---
    let ising = qubo.to_ising();
    let circuit = qaoa_circuit(&ising, opts.depth);
    let observable = ising.to_observable();

    // Deterministic symbol ordering: the qaoa module inserts gamma_0, beta_0,
    // gamma_1, beta_1, ... with ascending SymbolIds. Sort by SymbolId.
    let mut symbol_ids: Vec<u32> = circuit.symbols.keys().copied().collect();
    symbol_ids.sort();
    let n_params = symbol_ids.len(); // == 2 * depth

    // --- Optimize ---
    let backend = StatevectorBackend::new();

    let mk_params = |values: &[f64]| -> ParameterBinding {
        let mut pb = ParameterBinding::new();
        for (i, &sid) in symbol_ids.iter().enumerate() {
            pb.bind(sid, values[i]);
        }
        pb
    };

    let eval_cost = |values: &[f64]| -> f64 {
        let pb = mk_params(values);
        backend
            .expectation(&circuit, &pb, &observable)
            .expect("statevector expectation failed")
    };

    let initial: Vec<f64> = vec![0.5; n_params];
    let (opt_params, iterations, final_cost) = match opts.optimizer {
        Optimizer::CmaEs => run_cmaes(&initial, &eval_cost, opts.max_iters, opts.seed),
        Optimizer::Gradient => run_gradient(
            &initial,
            &backend,
            &circuit,
            &observable,
            &symbol_ids,
            opts.max_iters,
        ),
    };

    info(format!(
        "Optimizer done: iterations={}, final ⟨H_C⟩={:.6}",
        iterations, final_cost
    ));

    // --- Sample ---
    let pb_opt = mk_params(&opt_params);
    let cfg = ExecConfig {
        shots: Some(opts.shots),
        seed: opts.seed,
        mid_circuit_mode: MidCircuitMode::Skip,
    };
    let result = backend
        .execute(&circuit, &pb_opt, &cfg)
        .unwrap_or_else(|e| {
            eprintln!("Sampling failed: {}", e);
            std::process::exit(1);
        });

    let counts = match &result {
        ExecResult::Counts(c) => c,
        _ => {
            eprintln!(
                "Expected counts from backend, got {:?}",
                variant_name(&result)
            );
            std::process::exit(1);
        }
    };

    // --- Rank ---
    let samples = rank_samples(&qubo, counts, opts.shots, opts.top_k);
    let best = samples.first().cloned();

    // --- Brute force cross-check (n <= 20) ---
    let brute = if qubo.n <= 20 {
        let (bits, val) = qubo.brute_force();
        let bit_str = bits_vec_to_string(&bits);
        let matched = best
            .as_ref()
            .map(|s| (s.value - val).abs() < 1e-9)
            .unwrap_or(false);
        Some((bit_str, val, matched))
    } else {
        None
    };

    // --- Emit ---
    let (gammas, betas) = split_gamma_beta(&symbol_ids, &circuit.symbols, &opt_params);

    if format.is_machine() {
        let samples_json: Vec<Value> = samples
            .iter()
            .map(|s| {
                json!({
                    "bits": s.bits,
                    "value": s.value,
                    "count": s.count,
                    "probability": s.probability,
                })
            })
            .collect();
        let best_json = best.as_ref().map(|s| {
            json!({
                "bits": s.bits,
                "value": s.value,
                "count": s.count,
            })
        });
        let mut doc = json!({
            "mode": "qubo",
            "n": qubo.n,
            "best": best_json,
            "samples": samples_json,
            "qaoa": {
                "depth": opts.depth,
                "gamma": gammas,
                "beta":  betas,
                "iterations": iterations,
                "final_expected_cost": final_cost,
            },
        });
        if let Some((bits, val, matched)) = brute {
            doc["brute_force_check"] = json!({
                "bits": bits,
                "value": val,
                "matched": matched,
            });
        }
        println!("{}", doc);
    } else {
        println!("\nQAOA parameters:");
        for (i, g) in gammas.iter().enumerate() {
            println!("  gamma_{} = {:.6}", i, g);
        }
        for (i, b) in betas.iter().enumerate() {
            println!("  beta_{}  = {:.6}", i, b);
        }
        println!("\nTop {} bit-strings:", samples.len());
        for s in &samples {
            println!(
                "  {} : value={:.6}  count={}  p={:.4}",
                s.bits, s.value, s.count, s.probability
            );
        }
        if let Some((bits, val, matched)) = brute {
            println!(
                "\nBrute-force optimum: {} value={:.6} (matched QAOA best: {})",
                bits, val, matched
            );
        }
    }
}

// --- JSON input parsing ---

fn parse_qubo_json(text: &str) -> Result<Qubo, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("invalid JSON: {}", e))?;
    let n = v
        .get("n")
        .and_then(|x| x.as_u64())
        .ok_or("missing or invalid field 'n'")? as usize;
    let q_arr = v
        .get("Q")
        .and_then(|x| x.as_array())
        .ok_or("missing or invalid field 'Q' (expected array)")?;

    let mut qubo = Qubo::new(n);
    for (idx, entry) in q_arr.iter().enumerate() {
        let arr = entry
            .as_array()
            .ok_or_else(|| format!("Q[{}] is not an array [i, j, coeff]", idx))?;
        if arr.len() != 3 {
            return Err(format!(
                "Q[{}] must have exactly 3 elements [i, j, coeff]",
                idx
            ));
        }
        let i = arr[0]
            .as_u64()
            .ok_or_else(|| format!("Q[{}][0] (i) must be a non-negative integer", idx))?
            as usize;
        let j = arr[1]
            .as_u64()
            .ok_or_else(|| format!("Q[{}][1] (j) must be a non-negative integer", idx))?
            as usize;
        let c = arr[2]
            .as_f64()
            .ok_or_else(|| format!("Q[{}][2] (coeff) must be a number", idx))?;
        if i >= n || j >= n {
            return Err(format!(
                "Q[{}]: index out of range (i={}, j={}, n={})",
                idx, i, j, n
            ));
        }
        if i > j {
            return Err(format!(
                "Q[{}]: lower-triangular entry (i={}, j={}) rejected — use i <= j",
                idx, i, j
            ));
        }
        qubo.set(i, j, c);
    }
    Ok(qubo)
}

// --- Sample ranking ---

#[derive(Clone, Debug)]
struct RankedSample {
    bits: String,
    value: f64,
    count: u32,
    probability: f64,
}

fn rank_samples(
    qubo: &Qubo,
    counts: &HashMap<omega_core::outcome::Outcome, u32>,
    shots: u32,
    top_k: usize,
) -> Vec<RankedSample> {
    let mut items: Vec<RankedSample> = counts
        .iter()
        .map(|(key, &count)| {
            let x: Vec<bool> = (0..qubo.n).map(|i| key.bit(i as u32) == 1).collect();
            RankedSample {
                bits: bits_vec_to_string(&x),
                value: qubo.evaluate(&x),
                count,
                probability: count as f64 / shots as f64,
            }
        })
        .collect();
    // Sort: lowest (best) value first; ties broken by higher count, then bits lex.
    items.sort_by(|a, b| {
        a.value
            .partial_cmp(&b.value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.count.cmp(&a.count))
            .then(a.bits.cmp(&b.bits))
    });
    items.truncate(top_k);
    items
}

/// Stringify a Vec<bool> as "x_{n-1} x_{n-2} ... x_0" (MSB = highest index).
fn bits_vec_to_string(x: &[bool]) -> String {
    let mut s = String::with_capacity(x.len());
    for b in x.iter().rev() {
        s.push(if *b { '1' } else { '0' });
    }
    s
}

// --- Optimizers ---

fn run_cmaes<F: Fn(&[f64]) -> f64>(
    initial: &[f64],
    eval: &F,
    max_iters: usize,
    seed: Option<u64>,
) -> (Vec<f64>, usize, f64) {
    let sigma = 0.5;
    let mut cma = CmaEs::new(initial.to_vec(), sigma, None, seed);
    for _ in 0..max_iters {
        let population = cma.ask();
        let mut graded: Vec<(Vec<f64>, f64)> = population
            .into_iter()
            .map(|x| (x.clone(), eval(&x)))
            .collect();
        cma.tell(&mut graded);
    }
    let (best_x, best_f) = cma.best();
    (best_x.to_vec(), cma.generation(), best_f)
}

fn run_gradient(
    initial: &[f64],
    backend: &StatevectorBackend,
    circuit: &omega_core::circuit::CircuitIR,
    observable: &omega_core::executor::Observable,
    symbol_ids: &[u32],
    max_iters: usize,
) -> (Vec<f64>, usize, f64) {
    let lr = 0.1;
    let mut values = initial.to_vec();

    let mut last_cost = f64::INFINITY;
    for iter in 1..=max_iters {
        let mut pb = ParameterBinding::new();
        for (i, &sid) in symbol_ids.iter().enumerate() {
            pb.bind(sid, values[i]);
        }

        let grads = compute_gradient(backend, circuit, &pb, observable, &GradMethod::Adjoint)
            .expect("adjoint gradient failed");
        let grad_map: HashMap<u32, f64> = grads.into_iter().collect();
        let grad_vec: Vec<f64> = symbol_ids
            .iter()
            .map(|sid| *grad_map.get(sid).unwrap_or(&0.0))
            .collect();

        for (v, g) in values.iter_mut().zip(grad_vec.iter()) {
            *v -= lr * g;
        }

        let cost = backend
            .expectation(circuit, &pb, observable)
            .unwrap_or(f64::INFINITY);
        if (last_cost - cost).abs() < 1e-8 && iter > 5 {
            return (values, iter, cost);
        }
        last_cost = cost;
    }
    (values, max_iters, last_cost)
}

fn split_gamma_beta(
    symbol_ids: &[u32],
    names: &HashMap<u32, String>,
    values: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut gammas = Vec::new();
    let mut betas = Vec::new();
    for (&sid, &v) in symbol_ids.iter().zip(values.iter()) {
        let default = format!("sym_{}", sid);
        let name = names.get(&sid).unwrap_or(&default);
        if name.starts_with("gamma") {
            gammas.push(v);
        } else if name.starts_with("beta") {
            betas.push(v);
        }
    }
    (gammas, betas)
}

fn variant_name(result: &ExecResult) -> &'static str {
    match result {
        ExecResult::Counts(_) => "Counts",
        ExecResult::Statevector(_) => "Statevector",
        ExecResult::Probabilities(_) => "Probabilities",
    }
}

// silence unused IsingModel import warning if reorg changes
#[allow(dead_code)]
fn _touch_ising(_: &IsingModel) {}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- parse_qubo_json ----

    #[test]
    fn parse_qubo_json_accepts_well_formed_input() {
        let text = r#"{"n":3,"Q":[[0,0,-1.0],[0,1,2.0],[1,2,0.5]]}"#;
        let q = parse_qubo_json(text).unwrap();
        assert_eq!(q.n, 3);
        assert_eq!(q.entries.len(), 3);
        assert_eq!(q.entries.get(&(0, 0)).copied(), Some(-1.0));
        assert_eq!(q.entries.get(&(0, 1)).copied(), Some(2.0));
        assert_eq!(q.entries.get(&(1, 2)).copied(), Some(0.5));
    }

    #[test]
    fn parse_qubo_json_rejects_missing_n() {
        let err = parse_qubo_json(r#"{"Q":[]}"#).expect_err("missing n");
        assert!(err.contains("'n'"), "msg: {err}");
    }

    #[test]
    fn parse_qubo_json_rejects_missing_q() {
        let err = parse_qubo_json(r#"{"n":2}"#).expect_err("missing Q");
        assert!(err.contains("'Q'"), "msg: {err}");
    }

    #[test]
    fn parse_qubo_json_rejects_wrong_arity_entries() {
        let err = parse_qubo_json(r#"{"n":2,"Q":[[0,1]]}"#).expect_err("arity");
        assert!(err.contains("3 elements"), "msg: {err}");
    }

    #[test]
    fn parse_qubo_json_rejects_lower_triangular() {
        // i must be ≤ j; (1, 0) is lower-triangular.
        let err = parse_qubo_json(r#"{"n":2,"Q":[[1,0,0.5]]}"#).expect_err("lower-triangular");
        assert!(err.contains("lower-triangular"), "msg: {err}");
    }

    #[test]
    fn parse_qubo_json_rejects_out_of_range_indices() {
        // n=2 → indices 0, 1 only; (2, 2) is out of range.
        let err = parse_qubo_json(r#"{"n":2,"Q":[[2,2,1.0]]}"#).expect_err("out of range");
        assert!(err.contains("out of range"), "msg: {err}");
    }

    #[test]
    fn parse_qubo_json_rejects_invalid_json() {
        let err = parse_qubo_json("not json").expect_err("garbage");
        assert!(err.contains("invalid JSON"), "msg: {err}");
    }

    // ---- bits_vec_to_string ----

    #[test]
    fn bits_vec_to_string_is_msb_first() {
        // [false, false, true] = x_0=0, x_1=0, x_2=1.
        // String reads x_2 x_1 x_0 = "100".
        assert_eq!(bits_vec_to_string(&[false, false, true]), "100");
        // [true, false, false] = x_0=1, x_1=0, x_2=0 → "001".
        assert_eq!(bits_vec_to_string(&[true, false, false]), "001");
        // Empty input → empty string.
        assert_eq!(bits_vec_to_string(&[]), "");
        // All bits set.
        assert_eq!(bits_vec_to_string(&[true, true, true, true]), "1111");
    }

    // ---- split_gamma_beta ----

    #[test]
    fn split_gamma_beta_groups_by_name_prefix() {
        // The qaoa module names symbols `gamma_k` and `beta_k`. The
        // split walks symbol_ids in order and pushes each value into
        // the matching bucket — so the order in each bucket follows
        // the symbol_ids order, not the value order.
        let symbol_ids = vec![0u32, 1, 2, 3];
        let mut names = HashMap::new();
        names.insert(0u32, "gamma_0".to_string());
        names.insert(1u32, "beta_0".to_string());
        names.insert(2u32, "gamma_1".to_string());
        names.insert(3u32, "beta_1".to_string());
        let values = vec![0.1, 0.2, 0.3, 0.4];
        let (gammas, betas) = split_gamma_beta(&symbol_ids, &names, &values);
        assert_eq!(gammas, vec![0.1, 0.3]);
        assert_eq!(betas, vec![0.2, 0.4]);
    }

    #[test]
    fn split_gamma_beta_skips_other_symbol_names() {
        // Foreign symbols (not gamma_/beta_) are dropped — the helper
        // is QAOA-specific.
        let symbol_ids = vec![0u32, 1, 2];
        let mut names = HashMap::new();
        names.insert(0u32, "gamma_0".to_string());
        names.insert(1u32, "theta".to_string()); // not a QAOA symbol
        names.insert(2u32, "beta_0".to_string());
        let values = vec![1.0, 2.0, 3.0];
        let (gammas, betas) = split_gamma_beta(&symbol_ids, &names, &values);
        assert_eq!(gammas, vec![1.0]);
        assert_eq!(betas, vec![3.0]);
    }

    // ---- variant_name ----

    #[test]
    fn variant_name_for_each_exec_result_kind() {
        use std::collections::HashMap;
        assert_eq!(variant_name(&ExecResult::Counts(HashMap::new())), "Counts");
        assert_eq!(
            variant_name(&ExecResult::Statevector(vec![])),
            "Statevector"
        );
        assert_eq!(
            variant_name(&ExecResult::Probabilities(vec![])),
            "Probabilities"
        );
    }

    // ---- rank_samples ----

    #[test]
    fn rank_samples_orders_by_value_then_count_then_bits() {
        // Build a tiny QUBO where evaluating each bit-string is
        // deterministic. f(x) = -x_0 - x_1 + 2 x_0 x_1.
        let mut q = Qubo::new(2);
        q.set(0, 0, -1.0);
        q.set(1, 1, -1.0);
        q.set(0, 1, 2.0);

        // Three sampled bit-strings:
        //   |00> = bits 0 → f = 0
        //   |10> = bits 1 → f = -1
        //   |01> = bits 2 → f = -1
        let mut counts: HashMap<omega_core::outcome::Outcome, u32> = HashMap::new();
        counts.insert(omega_core::outcome::Outcome::from_u64(0, 2), 100);
        counts.insert(omega_core::outcome::Outcome::from_u64(1, 2), 200);
        counts.insert(omega_core::outcome::Outcome::from_u64(2, 2), 200);

        let ranked = rank_samples(&q, &counts, 500, 5);
        // Lowest value first; ties broken by higher count, then bits lex.
        assert_eq!(ranked.len(), 3);
        // Both |10> and |01> have value -1; ties broken by descending
        // count (both 200), then ascending bits string lex. "01" < "10".
        assert_eq!(ranked[0].value, -1.0);
        assert_eq!(ranked[1].value, -1.0);
        assert_eq!(ranked[2].value, 0.0);
        assert_eq!(ranked[0].bits, "01");
        assert_eq!(ranked[1].bits, "10");
    }

    #[test]
    fn rank_samples_truncates_to_top_k() {
        let q = Qubo::new(2);
        let mut counts: HashMap<omega_core::outcome::Outcome, u32> = HashMap::new();
        for bits in 0..4 {
            counts.insert(omega_core::outcome::Outcome::from_u64(bits, 2), 25);
        }
        let ranked = rank_samples(&q, &counts, 100, 2);
        assert_eq!(ranked.len(), 2);
    }
}
