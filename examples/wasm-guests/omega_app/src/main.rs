//! Generic "execute and report" guest for the non-variational examples.
//!
//! Loaded into omega-wasm-runtime in-process (the no-socket all-in-one case),
//! it reads a small JSON job from the host, runs the requested measurement on a
//! host-registered circuit through the omega host ABI, and ships the result
//! back through the (re-purposed) `omega_report_result` payload channel:
//!
//!   mode = "statevector"     -> payload = interleaved (re, im) amplitudes,
//!                               value   = number of amplitudes.
//!   mode = "counts"          -> payload = [bits0, count0, bits1, count1, ...],
//!                               value   = number of distinct outcomes.
//!   mode = "expectations"    -> payload = [⟨O_1⟩, ⟨O_2⟩, ...] for observable_ids,
//!                               value   = payload[0].
//!   mode = "classical_bits"  -> payload = [bit0, bit1, ...] (LSB first),
//!                               value   = integer value of the register.
//!
//! The host (aria-verify) chose the mode, so it knows how to read the payload.

use serde::Deserialize;

extern "C" {
    fn omega_execute_shots(
        circuit_id: i32,
        params_ptr: *const f64,
        num_params: i32,
        shots: i32,
        counts_out_ptr: *mut f64,
        max_entries: i32,
    ) -> i32;
    fn omega_execute_multi(
        circuit_id: i32,
        params_ptr: *const f64,
        num_params: i32,
        obs_ids_ptr: *const f64,
        num_obs: i32,
        results_out_ptr: *mut f64,
    );
    fn omega_read_classical(
        circuit_id: i32,
        params_ptr: *const f64,
        num_params: i32,
        seed_bits: i64,
        out_ptr: *mut u8,
        max_len: i32,
    ) -> i32;
    fn omega_read_statevector(
        circuit_id: i32,
        params_ptr: *const f64,
        num_params: i32,
        out_ptr: *mut f64,
        max_pairs: i32,
    ) -> i32;
    fn omega_circuit_num_qubits(circuit_id: i32) -> i32;
    fn omega_input_len() -> i32;
    fn omega_input_read(out_ptr: *mut u8, max_len: i32) -> i32;
    fn omega_report_result(params_ptr: *const f64, num_params: i32, value_bits: i64, iterations: i32);
}

#[derive(Deserialize)]
struct Job {
    circuit_id: i32,
    mode: String,
    #[serde(default)]
    params: Vec<f64>,
    #[serde(default)]
    observable_ids: Vec<i32>,
    #[serde(default)]
    shots: Option<i32>,
    #[serde(default)]
    seed: Option<i64>,
}

fn read_input() -> Vec<u8> {
    unsafe {
        let len = omega_input_len();
        if len <= 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; len as usize];
        let n = omega_input_read(buf.as_mut_ptr(), len);
        buf.truncate(n.max(0) as usize);
        buf
    }
}

fn report(payload: &[f64], value: f64) {
    unsafe {
        omega_report_result(
            payload.as_ptr(),
            payload.len() as i32,
            value.to_bits() as i64,
            0,
        );
    }
}

fn main() {
    let raw = read_input();
    if raw.is_empty() {
        println!("[omega_app.wasm] missing input job");
        return;
    }
    let job: Job = match serde_json::from_slice(&raw) {
        Ok(j) => j,
        Err(e) => {
            println!("[omega_app.wasm] bad input json: {e}");
            return;
        }
    };
    let p = job.params.as_ptr();
    let np = job.params.len() as i32;

    match job.mode.as_str() {
        "statevector" => {
            let nq = unsafe { omega_circuit_num_qubits(job.circuit_id) };
            let dim = 1usize << nq.max(0);
            let mut out = vec![0.0_f64; dim * 2];
            let n = unsafe {
                omega_read_statevector(job.circuit_id, p, np, out.as_mut_ptr(), dim as i32)
            };
            if n < 0 {
                println!("[omega_app.wasm] omega_read_statevector failed");
                return;
            }
            out.truncate((n as usize) * 2);
            println!("[omega_app.wasm] statevector: {} amplitudes", n);
            report(&out, n as f64);
        }
        "counts" => {
            let shots = job.shots.unwrap_or(8192);
            let max_entries = 256;
            let mut out = vec![0.0_f64; max_entries * 2];
            let n = unsafe {
                omega_execute_shots(job.circuit_id, p, np, shots, out.as_mut_ptr(), max_entries as i32)
            };
            if n < 0 {
                println!("[omega_app.wasm] omega_execute_shots failed");
                return;
            }
            out.truncate((n as usize) * 2);
            println!("[omega_app.wasm] counts: {} outcomes over {} shots", n, shots);
            report(&out, n as f64);
        }
        "expectations" => {
            let obs_ids_f64: Vec<f64> = job.observable_ids.iter().map(|&x| x as f64).collect();
            let mut out = vec![0.0_f64; obs_ids_f64.len()];
            unsafe {
                omega_execute_multi(
                    job.circuit_id,
                    p,
                    np,
                    obs_ids_f64.as_ptr(),
                    obs_ids_f64.len() as i32,
                    out.as_mut_ptr(),
                );
            }
            let first = out.first().copied().unwrap_or(0.0);
            println!("[omega_app.wasm] expectations: {:?}", out);
            report(&out, first);
        }
        "classical_bits" => {
            let nq = unsafe { omega_circuit_num_qubits(job.circuit_id) };
            let max_len = (nq.max(0) as usize).max(1);
            let mut bits = vec![0u8; max_len];
            let seed = job.seed.unwrap_or(7);
            let n = unsafe {
                omega_read_classical(job.circuit_id, p, np, seed, bits.as_mut_ptr(), max_len as i32)
            };
            if n < 0 {
                println!("[omega_app.wasm] omega_read_classical failed");
                return;
            }
            bits.truncate(n as usize);
            let mut value = 0u64;
            for (i, &b) in bits.iter().enumerate() {
                if b != 0 {
                    value |= 1u64 << i;
                }
            }
            let payload: Vec<f64> = bits.iter().map(|&b| b as f64).collect();
            println!("[omega_app.wasm] classical_bits: {:?} (= {})", bits, value);
            report(&payload, value as f64);
        }
        other => {
            println!("[omega_app.wasm] unknown mode '{other}'");
        }
    }
}
