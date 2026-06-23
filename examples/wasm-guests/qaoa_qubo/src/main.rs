//! QAOA-on-QUBO with a fully WASM-side optimizer.
//!
//! Demonstrates the end-to-end "custom optimization lambda" pattern:
//!
//!   1. The host stages a QUBO matrix (and a few hyperparameters) as the
//!      input payload.
//!   2. This guest reads the payload, asks the host to build a QAOA
//!      circuit + observable + Ising offset from the QUBO at the chosen
//!      depth (`omega_qaoa_from_qubo`).
//!   3. The optimizer — SPSA, Spall 1992 — is implemented entirely in
//!      WASM. No `omega_gradient` calls; only `omega_execute` for cost
//!      evaluations. Two evals per iteration regardless of parameter
//!      count, so it scales to arbitrary depth.
//!   4. Result is reported back via `omega_report_result`.
//!
//! Why SPSA: gradient-free, very small footprint, robust on noisy
//! landscapes — the canonical hardware-friendly optimizer for QAOA. It is
//! NOT shipped natively elsewhere in the repo; this guest is the only
//! place SPSA exists, which is exactly the point of the exercise.
//!
//! Input payload:
//! {
//!   "qubo":      "<QUBO JSON>"  (required),
//!   "depth":     <int, default 2>,
//!   "max_iters": <int, default 200>,
//!   "a":         <f64, default 0.2>,    // SPSA learning-rate scale
//!   "c":         <f64, default 0.1>,    // SPSA perturbation scale
//!   "alpha":     <f64, default 0.602>,  // SPSA learning-rate decay
//!   "gamma":     <f64, default 0.101>,  // SPSA perturbation decay
//!   "A":         <f64, default 5.0>,    // SPSA stability constant
//!   "init":      [f64,...],             // optional, default 0.5 each
//!   "seed":      <int, default 0xC0FFEE>,
//!   "log_every": <int, default 25>
//! }

use serde::Deserialize;

extern "C" {
    fn omega_execute(circuit_id: i32, params_ptr: *const f64, num_params: i32, observable_id: i32) -> f64;
    fn omega_circuit_num_params(circuit_id: i32) -> i32;
    fn omega_qaoa_from_qubo(
        qubo_json_ptr: *const u8,
        qubo_json_len: i32,
        depth: i32,
        out_ids_ptr: *mut i32,
    ) -> f64;
    fn omega_input_len() -> i32;
    fn omega_input_read(out_ptr: *mut u8, max_len: i32) -> i32;
    fn omega_report_progress(iteration: i32, value_bits: i64);
    fn omega_report_result(params_ptr: *const f64, num_params: i32, value_bits: i64, iterations: i32);
}

#[derive(Default, Debug, Deserialize)]
#[serde(default)]
struct Config {
    qubo: Option<String>,
    depth: Option<i32>,
    max_iters: Option<i32>,
    a: Option<f64>,
    c: Option<f64>,
    alpha: Option<f64>,
    gamma: Option<f64>,
    #[serde(rename = "A")]
    big_a: Option<f64>,
    init: Option<Vec<f64>>,
    seed: Option<u64>,
    log_every: Option<i32>,
}

fn read_input() -> Vec<u8> {
    unsafe {
        let len = omega_input_len();
        if len <= 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; len as usize];
        let n = omega_input_read(buf.as_mut_ptr(), len);
        buf.truncate(n as usize);
        buf
    }
}

/// Tiny LCG so the SPSA Bernoulli draws are deterministic given `seed`.
/// (Numerical Recipes constants.)
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    /// ±1 with equal probability.
    fn rademacher(&mut self) -> f64 {
        if (self.next_u64() >> 63) & 1 == 0 { 1.0 } else { -1.0 }
    }
}

fn evaluate(circuit_id: i32, observable_id: i32, params: &[f64]) -> f64 {
    unsafe {
        omega_execute(
            circuit_id,
            params.as_ptr(),
            params.len() as i32,
            observable_id,
        )
    }
}

fn main() {
    let raw = read_input();
    if raw.is_empty() {
        println!("[qaoa_qubo.wasm] missing input — supply a QUBO JSON via host input bytes");
        return;
    }
    let cfg: Config = match serde_json::from_slice(&raw) {
        Ok(c) => c,
        Err(e) => {
            println!("[qaoa_qubo.wasm] failed to parse input JSON: {}", e);
            return;
        }
    };

    let qubo_json = match cfg.qubo.as_deref() {
        Some(s) => s,
        None => {
            println!("[qaoa_qubo.wasm] input is missing required field 'qubo'");
            return;
        }
    };
    let depth = cfg.depth.unwrap_or(2).max(1);
    let max_iters = cfg.max_iters.unwrap_or(200) as usize;
    let a = cfg.a.unwrap_or(0.2);
    let c = cfg.c.unwrap_or(0.1);
    let alpha = cfg.alpha.unwrap_or(0.602);
    let gamma = cfg.gamma.unwrap_or(0.101);
    let big_a = cfg.big_a.unwrap_or(5.0);
    let log_every = cfg.log_every.unwrap_or(25) as usize;
    let seed = cfg.seed.unwrap_or(0x00C0_FFEEu64);

    // Build QAOA(depth) from the supplied QUBO at runtime.
    let mut out_ids = [0i32; 2];
    let ising_offset = unsafe {
        omega_qaoa_from_qubo(
            qubo_json.as_ptr(),
            qubo_json.len() as i32,
            depth,
            out_ids.as_mut_ptr(),
        )
    };
    if !ising_offset.is_finite() || out_ids[0] <= 0 {
        println!("[qaoa_qubo.wasm] omega_qaoa_from_qubo failed");
        return;
    }
    let circuit_id = out_ids[0];
    let observable_id = out_ids[1];
    let num_params = unsafe { omega_circuit_num_params(circuit_id) };
    if num_params <= 0 {
        println!("[qaoa_qubo.wasm] runtime circuit has no free parameters?");
        return;
    }
    let n_params = num_params as usize;

    println!(
        "[qaoa_qubo.wasm] depth={} num_params={} ising_offset={} circuit={} observable={}",
        depth, n_params, ising_offset, circuit_id, observable_id
    );

    // Initial point.
    let mut x: Vec<f64> = match cfg.init {
        Some(ref v) if !v.is_empty() => {
            let mut p = v.clone();
            p.resize(n_params, 0.5);
            p
        }
        _ => vec![0.5; n_params],
    };
    let mut x_plus = vec![0.0_f64; n_params];
    let mut x_minus = vec![0.0_f64; n_params];
    let mut delta = vec![0.0_f64; n_params];

    let mut rng = Lcg(seed);
    let mut best_cost = evaluate(circuit_id, observable_id, &x);
    let mut best_params = x.clone();
    let mut best_iter: i32 = 0;
    unsafe { omega_report_progress(0, best_cost.to_bits() as i64); }
    println!("  iter    0: cost = {:.10}  (initial)", best_cost);

    for iter in 1..=max_iters {
        let ak = a / ((iter as f64) + big_a).powf(alpha);
        let ck = c / (iter as f64).powf(gamma);
        for i in 0..n_params {
            delta[i] = rng.rademacher();
            x_plus[i]  = x[i] + ck * delta[i];
            x_minus[i] = x[i] - ck * delta[i];
        }
        let f_plus  = evaluate(circuit_id, observable_id, &x_plus);
        let f_minus = evaluate(circuit_id, observable_id, &x_minus);
        let two_ck  = 2.0 * ck;
        for i in 0..n_params {
            let g_hat = (f_plus - f_minus) / (two_ck * delta[i]);
            x[i] -= ak * g_hat;
        }
        let cost = evaluate(circuit_id, observable_id, &x);
        unsafe { omega_report_progress(iter as i32, cost.to_bits() as i64); }
        if cost < best_cost {
            best_cost = cost;
            best_params = x.clone();
            best_iter = iter as i32;
        }
        if log_every > 0 && iter % log_every == 0 {
            println!("  iter {:>4}: cost = {:.10}  (best so far {:.10})",
                     iter, cost, best_cost);
        }
    }

    let qubo_value = best_cost + ising_offset;
    println!(
        "[qaoa_qubo.wasm] best cost = {:.10} (Ising) / {:.10} (QUBO) at iter {}",
        best_cost, qubo_value, best_iter
    );

    unsafe {
        omega_report_result(
            best_params.as_ptr(),
            n_params as i32,
            best_cost.to_bits() as i64,
            max_iters as i32,
        );
    }
}
