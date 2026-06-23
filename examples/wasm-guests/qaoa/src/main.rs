//! QAOA optimization loop — parametrized.
//!
//! Reads a JSON config from the host via `omega_input_*`. Falls back to a
//! depth-1 default that assumes the host has pre-registered a (circuit,
//! observable) pair at IDs 1/1 — keeping the legacy behaviour exactly so
//! existing CLI recipes still converge.
//!
//! New capabilities:
//!   - `qubo` field: a QUBO JSON payload — the guest then asks the host
//!     to build the QAOA circuit + observable from it via
//!     `omega_qaoa_from_qubo`. With `depth: P`, the guest gets a 2P-param
//!     circuit at runtime, no NUM_PARAMS=2 cap.
//!   - `qasm` / `observable` fields: register a custom (circuit, observable)
//!     pair at runtime, like the VQE guest.
//!   - `optimizer`: "gd" (default, legacy) or "adam".
//!
//! Input config (all fields optional):
//! {
//!   "circuit_id":     <int, default 1>,
//!   "observable_id":  <int, default 1>,
//!   "qasm":           "<QASM>",                // overrides circuit_id
//!   "observable":     "<Pauli sum>",           // overrides observable_id
//!   "qubo":           "<QUBO JSON>",           // builds QAOA at runtime
//!   "depth":          <int, default 1>,        // QAOA depth (with `qubo`)
//!   "num_params":     <int, default = #symbols on circuit>,
//!   "init":           [f64, ...],
//!   "max_iters":      <int, default 150>,
//!   "lr":             <f64, default 0.3>,
//!   "tol":            <f64, default 1e-6>,
//!   "optimizer":      "gd" | "adam",
//!   "log_every":      <int, default 20>
//! }

use serde::Deserialize;

extern "C" {
    fn omega_execute(circuit_id: i32, params_ptr: *const f64, num_params: i32, observable_id: i32) -> f64;
    fn omega_gradient(circuit_id: i32, params_ptr: *const f64, num_params: i32, observable_id: i32, grad_out_ptr: *mut f64);
    fn omega_register_qasm(src_ptr: *const u8, src_len: i32) -> i32;
    fn omega_register_observable_str(spec_ptr: *const u8, spec_len: i32) -> i32;
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
    circuit_id: Option<i32>,
    observable_id: Option<i32>,
    qasm: Option<String>,
    observable: Option<String>,
    qubo: Option<String>,
    depth: Option<i32>,
    num_params: Option<i32>,
    init: Option<Vec<f64>>,
    max_iters: Option<i32>,
    lr: Option<f64>,
    tol: Option<f64>,
    optimizer: Option<String>,
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

fn main() {
    let raw = read_input();
    let cfg: Config = if raw.is_empty() {
        Config::default()
    } else {
        match serde_json::from_slice(&raw) {
            Ok(c) => c,
            Err(e) => {
                println!("[qaoa.wasm] failed to parse input JSON: {} — using defaults", e);
                Config::default()
            }
        }
    };

    let depth = cfg.depth.unwrap_or(1).max(1);

    // Resolve circuit / observable. Priority order:
    //   1. `qubo` JSON  → host builds QAOA(depth) circuit + observable from it
    //   2. `qasm` + `observable` → host registers them as new IDs
    //   3. `circuit_id` + `observable_id` (default 1/1) → host pre-registered
    let (circuit_id, observable_id) = if let Some(qubo) = cfg.qubo.as_deref() {
        let mut out_ids = [0i32; 2];
        let offset = unsafe {
            omega_qaoa_from_qubo(
                qubo.as_ptr(),
                qubo.len() as i32,
                depth,
                out_ids.as_mut_ptr(),
            )
        };
        if !offset.is_finite() || out_ids[0] <= 0 {
            println!("[qaoa.wasm] omega_qaoa_from_qubo failed");
            return;
        }
        println!(
            "[qaoa.wasm] built QAOA from QUBO: cid={} oid={} ising_offset={}",
            out_ids[0], out_ids[1], offset
        );
        (out_ids[0], out_ids[1])
    } else if cfg.qasm.is_some() || cfg.observable.is_some() {
        let cid = match cfg.qasm.as_deref() {
            Some(qasm) => {
                let id = unsafe { omega_register_qasm(qasm.as_ptr(), qasm.len() as i32) };
                if id < 0 {
                    println!("[qaoa.wasm] omega_register_qasm failed");
                    return;
                }
                id
            }
            None => cfg.circuit_id.unwrap_or(1),
        };
        let oid = match cfg.observable.as_deref() {
            Some(obs) => {
                let id = unsafe {
                    omega_register_observable_str(obs.as_ptr(), obs.len() as i32)
                };
                if id < 0 {
                    println!("[qaoa.wasm] omega_register_observable_str failed");
                    return;
                }
                id
            }
            None => cfg.observable_id.unwrap_or(1),
        };
        (cid, oid)
    } else {
        (cfg.circuit_id.unwrap_or(1), cfg.observable_id.unwrap_or(1))
    };

    let num_params: usize = match cfg.num_params {
        Some(n) if n > 0 => n as usize,
        _ => {
            let queried = unsafe { omega_circuit_num_params(circuit_id) };
            if queried <= 0 {
                println!(
                    "[qaoa.wasm] circuit {} has no free parameters and num_params not specified",
                    circuit_id
                );
                return;
            }
            queried as usize
        }
    };

    let max_iters = cfg.max_iters.unwrap_or(150) as usize;
    let lr = cfg.lr.unwrap_or(0.3);
    let tol = cfg.tol.unwrap_or(1e-6);
    let log_every = cfg.log_every.unwrap_or(20) as usize;
    let optimizer = cfg.optimizer.as_deref().unwrap_or("gd").to_lowercase();

    let mut params: Vec<f64> = match cfg.init {
        Some(ref v) if !v.is_empty() => {
            let mut p = v.clone();
            p.resize(num_params, 0.5);
            p
        }
        // Legacy default for the depth=1 path: gamma=beta=0.5 (the original
        // hard-coded init). For higher depths we tile the same value.
        _ => vec![0.5; num_params],
    };

    println!(
        "[qaoa.wasm] circuit_id={} observable_id={} num_params={} depth={} optimizer={} lr={}",
        circuit_id, observable_id, num_params, depth, optimizer, lr
    );

    let mut grad = vec![0.0_f64; num_params];
    let mut best_cost = f64::INFINITY;
    let mut best_params = params.clone();
    let mut best_iter: i32 = 0;
    let mut prev_cost = f64::INFINITY;
    let mut iters_run: i32 = 0;

    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;
    let eps: f64 = 1e-8;
    let mut m = vec![0.0_f64; num_params];
    let mut v = vec![0.0_f64; num_params];

    for iter in 0..max_iters {
        let cost = unsafe {
            omega_execute(
                circuit_id,
                params.as_ptr(),
                num_params as i32,
                observable_id,
            )
        };
        unsafe {
            omega_report_progress(iter as i32, cost.to_bits() as i64);
        }
        if cost < best_cost {
            best_cost = cost;
            best_params = params.clone();
            best_iter = iter as i32;
        }
        if log_every > 0 && iter % log_every == 0 {
            println!("  iter {:>4}: cost = {:.10}", iter, cost);
        }
        if iter > 0 && (prev_cost - cost).abs() < tol {
            println!("Converged at iter {} with cost = {:.10}", iter, cost);
            iters_run = iter as i32 + 1;
            break;
        }
        prev_cost = cost;

        unsafe {
            omega_gradient(
                circuit_id,
                params.as_ptr(),
                num_params as i32,
                observable_id,
                grad.as_mut_ptr(),
            );
        }

        match optimizer.as_str() {
            "adam" => {
                let t = (iter + 1) as i32;
                let bc1 = 1.0 - beta1.powi(t);
                let bc2 = 1.0 - beta2.powi(t);
                for i in 0..num_params {
                    let gi = grad[i];
                    m[i] = beta1 * m[i] + (1.0 - beta1) * gi;
                    v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
                    let m_hat = m[i] / bc1;
                    let v_hat = v[i] / bc2;
                    params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
            _ => {
                for i in 0..num_params {
                    params[i] -= lr * grad[i];
                }
            }
        }
        iters_run = iter as i32 + 1;
    }

    println!(
        "[qaoa.wasm] best cost = {:.10} at iter {} ({} params)",
        best_cost, best_iter, num_params
    );

    unsafe {
        omega_report_result(
            best_params.as_ptr(),
            num_params as i32,
            best_cost.to_bits() as i64,
            iters_run,
        );
    }
}
