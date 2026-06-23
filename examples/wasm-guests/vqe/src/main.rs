//! VQE (Variational Quantum Eigensolver) optimization loop — parametrized.
//!
//! Reads a JSON config from the host via `omega_input_*`, falls back to
//! a hard-coded H2-style default when no input is staged. The optimizer
//! lives entirely in this WASM module — that is the point of the
//! exercise. The host only provides primitives:
//!   omega_execute(circuit_id, params_ptr, num_params, observable_id) -> f64
//!   omega_gradient(circuit_id, params_ptr, num_params, observable_id, grad_out_ptr)
//!   omega_register_qasm(src_ptr, src_len) -> i32
//!   omega_register_observable_str(spec_ptr, spec_len) -> i32
//!   omega_circuit_num_params(circuit_id) -> i32
//!   omega_input_len() -> i32
//!   omega_input_read(out_ptr, max_len) -> i32
//!   omega_report_progress(iteration, value_bits)
//!   omega_report_result(params_ptr, num_params, value_bits, iterations)
//!
//! Input config (all fields optional):
//! {
//!   "circuit_id":     <int, default 1>,        // pre-registered by host
//!   "observable_id":  <int, default 1>,        // pre-registered by host
//!   "qasm":           "<QASM source>",         // overrides circuit_id
//!   "observable":     "<Pauli sum>",           // overrides observable_id
//!   "num_params":     <int, default = #symbols on circuit>,
//!   "init":           [f64, ...],              // default zeros
//!   "max_iters":      <int, default 200>,
//!   "lr":             <f64, default 0.4>,
//!   "tol":            <f64, default 1e-6>,
//!   "optimizer":      "gd" | "adam",           // default "gd"
//!   "log_every":      <int, default 20>
//! }

use serde::Deserialize;

extern "C" {
    fn omega_execute(circuit_id: i32, params_ptr: *const f64, num_params: i32, observable_id: i32) -> f64;
    fn omega_gradient(circuit_id: i32, params_ptr: *const f64, num_params: i32, observable_id: i32, grad_out_ptr: *mut f64);
    fn omega_register_qasm(src_ptr: *const u8, src_len: i32) -> i32;
    fn omega_register_observable_str(spec_ptr: *const u8, spec_len: i32) -> i32;
    fn omega_circuit_num_params(circuit_id: i32) -> i32;
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

fn register_qasm(qasm: &str) -> i32 {
    unsafe { omega_register_qasm(qasm.as_ptr(), qasm.len() as i32) }
}

fn register_observable(spec: &str) -> i32 {
    unsafe { omega_register_observable_str(spec.as_ptr(), spec.len() as i32) }
}

fn main() {
    let raw = read_input();
    let cfg: Config = if raw.is_empty() {
        Config::default()
    } else {
        match serde_json::from_slice(&raw) {
            Ok(c) => c,
            Err(e) => {
                println!("[vqe.wasm] failed to parse input JSON: {} — using defaults", e);
                Config::default()
            }
        }
    };

    // Resolve circuit and observable IDs. If `qasm` / `observable` strings
    // are supplied, register them at runtime; otherwise use the provided
    // IDs (default 1, the host-pre-registered convention).
    let circuit_id = if let Some(qasm) = cfg.qasm.as_deref() {
        let id = register_qasm(qasm);
        if id < 0 {
            println!("[vqe.wasm] omega_register_qasm failed");
            return;
        }
        id
    } else {
        cfg.circuit_id.unwrap_or(1)
    };

    let observable_id = if let Some(obs) = cfg.observable.as_deref() {
        let id = register_observable(obs);
        if id < 0 {
            println!("[vqe.wasm] omega_register_observable_str failed");
            return;
        }
        id
    } else {
        cfg.observable_id.unwrap_or(1)
    };

    // Number of parameters: explicit override, else query the host.
    let num_params: usize = match cfg.num_params {
        Some(n) if n > 0 => n as usize,
        _ => {
            let queried = unsafe { omega_circuit_num_params(circuit_id) };
            if queried <= 0 {
                println!(
                    "[vqe.wasm] circuit {} has no free parameters and num_params not specified",
                    circuit_id
                );
                return;
            }
            queried as usize
        }
    };

    let max_iters = cfg.max_iters.unwrap_or(200) as usize;
    let lr = cfg.lr.unwrap_or(0.4);
    let tol = cfg.tol.unwrap_or(1e-6);
    let log_every = cfg.log_every.unwrap_or(20) as usize;
    let optimizer = cfg.optimizer.as_deref().unwrap_or("gd").to_lowercase();

    let mut params: Vec<f64> = match cfg.init {
        Some(ref v) if !v.is_empty() => {
            let mut p = v.clone();
            p.resize(num_params, 0.0);
            p
        }
        _ => vec![0.0; num_params],
    };

    println!(
        "[vqe.wasm] circuit_id={} observable_id={} num_params={} max_iters={} optimizer={} lr={}",
        circuit_id, observable_id, num_params, max_iters, optimizer, lr
    );

    let mut grad = vec![0.0_f64; num_params];
    let mut best_e = f64::INFINITY;
    let mut best_params = params.clone();
    let mut best_iter: i32 = 0;
    let mut prev_e = f64::INFINITY;
    let mut iters_run: i32 = 0;

    // Adam state.
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;
    let eps: f64 = 1e-8;
    let mut m = vec![0.0_f64; num_params];
    let mut v = vec![0.0_f64; num_params];

    for iter in 0..max_iters {
        let energy = unsafe {
            omega_execute(
                circuit_id,
                params.as_ptr(),
                num_params as i32,
                observable_id,
            )
        };
        unsafe {
            omega_report_progress(iter as i32, energy.to_bits() as i64);
        }
        if energy < best_e {
            best_e = energy;
            best_params = params.clone();
            best_iter = iter as i32;
        }
        if log_every > 0 && iter % log_every == 0 {
            println!("  iter {:>4}: E = {:.10}", iter, energy);
        }
        if iter > 0 && (prev_e - energy).abs() < tol {
            println!("Converged at iter {} with E = {:.10}", iter, energy);
            iters_run = iter as i32 + 1;
            break;
        }
        prev_e = energy;

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
        "[vqe.wasm] best E = {:.10} at iter {} ({} params)",
        best_e, best_iter, num_params
    );

    unsafe {
        omega_report_result(
            best_params.as_ptr(),
            num_params as i32,
            best_e.to_bits() as i64,
            iters_run,
        );
    }
}
