//! WASM runtime: wasmtime integration with quantum host functions.

use std::sync::{Arc, Mutex};

use wasmtime::*;
use wasmtime_wasi::cli::stderr as wasi_stderr;
use wasmtime_wasi::preview1::WasiP1Ctx;
use wasmtime_wasi::WasiCtxBuilder;

use crate::host::{HostState, WasmResult};
use omega_core::error::{OmegaError, Result};
use omega_core::executor::Observable;
use omega_parser::lower_to_ir;

/// Default memory cap for guest WASM linear memory, in bytes (256 MiB).
/// Adjustable via the `OMEGA_WASM_MEMORY_LIMIT` env var on the host.
pub const DEFAULT_GUEST_MEMORY_LIMIT: usize = 256 * 1024 * 1024;

/// Default cap on the number of WASM tables / memories per store. A
/// well-behaved guest needs exactly one of each; capping at small
/// constants stops a malicious module from instantiating thousands.
const DEFAULT_GUEST_INSTANCES: usize = 4;

fn host_memory_limit() -> usize {
    std::env::var("OMEGA_WASM_MEMORY_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_GUEST_MEMORY_LIMIT)
}

/// Combined store data: WASI context + our quantum host state +
/// resource limits enforced by `Store::limiter`.
pub struct StoreData {
    wasi: WasiP1Ctx,
    host: Arc<Mutex<HostState>>,
    limits: StoreLimits,
}

impl StoreData {
    fn host(&self) -> &Arc<Mutex<HostState>> {
        &self.host
    }
}

/// WASM runner for executing hybrid quantum-classical loops.
pub struct WasmRunner {
    engine: Engine,
    host_state: Arc<Mutex<HostState>>,
    /// If true, the guest's WASI stdout is wired to the host's stderr instead
    /// of the host's stdout. Use this when the host owns the stdout stream
    /// (e.g. emitting machine-parseable JSON) and the guest's `println!`
    /// noise would otherwise contaminate it.
    stdout_to_stderr: bool,
    /// Optional host-provided RNG seed for WASI's `secure_random` /
    /// `insecure_random_get` syscalls. When `None`, the guest sees OS
    /// entropy (the default and historic behaviour). Setting a seed
    /// makes guest output reproducible — same seed, same circuits,
    /// same gradient samples — which is what the QML training and
    /// QAOA runs need for deterministic regression testing.
    ///
    /// Why provide this: the guest can't seed its own WASI RNG, so
    /// without a host-side hook the random_get path is non-reproducible
    /// even when every other part of the run is. (Phase 14a #6 sub-bullet:
    /// "WASM: host-provided RNG seed (no guest-unseeded entropy)".)
    rng_seed: Option<u64>,
}

impl WasmRunner {
    pub fn new(host_state: HostState) -> Result<Self> {
        let mut config = Config::new();
        config.consume_fuel(true);

        let engine = Engine::new(&config).map_err(|e| OmegaError::Backend(e.to_string()))?;

        Ok(Self {
            engine,
            host_state: Arc::new(Mutex::new(host_state)),
            stdout_to_stderr: false,
            rng_seed: None,
        })
    }

    /// Get a reference to the host state for pre-registering circuits/observables.
    pub fn host_state(&self) -> &Arc<Mutex<HostState>> {
        &self.host_state
    }

    /// Route the guest's stdout to the host's stderr instead of the host's
    /// stdout. Used by callers that emit machine-parseable output on stdout
    /// (`omega-wasm --format json`) to keep the JSON stream clean.
    pub fn redirect_guest_stdout_to_stderr(&mut self, enable: bool) -> &mut Self {
        self.stdout_to_stderr = enable;
        self
    }

    /// Seed the guest's WASI random source so `random_get` /
    /// `insecure_random_get` are deterministic. Pass `None` to keep the
    /// default (OS entropy). With a seed set, two runs of the same
    /// guest with the same seed produce identical output — required for
    /// reproducible QML / QAOA training runs that drive a stochastic
    /// optimiser.
    pub fn rng_seed(&mut self, seed: Option<u64>) -> &mut Self {
        self.rng_seed = seed;
        self
    }

    /// Run a WASM module.
    ///
    /// The module should export a `_start` function (WASI convention) or `main`.
    /// It can call host-imported functions to interact with the quantum runtime.
    pub fn run(&self, wasm_bytes: &[u8], fuel: u64) -> Result<Option<WasmResult>> {
        let module = Module::new(&self.engine, wasm_bytes)
            .map_err(|e| OmegaError::Backend(format!("WASM compile error: {}", e)))?;

        // Set up WASI. When the caller owns stdout (e.g. JSON output), point
        // the guest's stdout at the host's stderr so guest println! noise
        // doesn't pollute the parseable stream.
        let mut builder = WasiCtxBuilder::new();
        if self.stdout_to_stderr {
            builder.stdout(wasi_stderr());
        } else {
            builder.inherit_stdout();
        }
        // If the caller asked for a deterministic seed, plug a seeded
        // StdRng into both `secure_random` and `insecure_random` (and
        // its u128 seed counterpart). The two RNGs are independent
        // streams from the same root seed so guests that mix calls
        // don't see correlations between the two surfaces.
        if let Some(seed) = self.rng_seed {
            use rand::rngs::StdRng;
            use rand::SeedableRng;
            builder.secure_random(StdRng::seed_from_u64(seed));
            builder.insecure_random(StdRng::seed_from_u64(seed.wrapping_add(0x9E3779B97F4A7C15)));
            builder.insecure_random_seed(seed as u128);
        }
        let wasi = builder.inherit_stderr().build_p1();

        let limits = StoreLimitsBuilder::new()
            .memory_size(host_memory_limit())
            .instances(DEFAULT_GUEST_INSTANCES)
            .tables(DEFAULT_GUEST_INSTANCES)
            .memories(DEFAULT_GUEST_INSTANCES)
            .build();
        let store_data = StoreData {
            wasi,
            host: Arc::clone(&self.host_state),
            limits,
        };

        let mut store = Store::new(&self.engine, store_data);
        store.limiter(|data: &mut StoreData| &mut data.limits);
        store.set_fuel(fuel).unwrap();

        let mut linker = Linker::new(&self.engine);

        // Add WASI imports
        wasmtime_wasi::preview1::add_to_linker_sync(&mut linker, |data: &mut StoreData| {
            &mut data.wasi
        })
        .map_err(|e| OmegaError::Backend(format!("WASI link error: {}", e)))?;

        // Add quantum host functions
        add_host_functions(&mut linker)?;

        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| OmegaError::Backend(format!("WASM instantiation error: {}", e)))?;

        // Try to call _start (WASI convention)
        let start = instance
            .get_typed_func::<(), ()>(&mut store, "_start")
            .map_err(|e| OmegaError::Backend(format!("no _start export: {}", e)))?;

        start
            .call(&mut store, ())
            .map_err(|e| OmegaError::Backend(format!("WASM execution error: {}", e)))?;

        // Return the final result from host state
        let result = self.host_state.lock().unwrap().final_result.clone();
        Ok(result)
    }
}

/// Register quantum host functions in the wasmtime linker.
fn add_host_functions(linker: &mut Linker<StoreData>) -> Result<()> {
    // omega_execute(circuit_id: i32, params_ptr: i32, num_params: i32, observable_id: i32) -> f64
    linker
        .func_wrap(
            "env",
            "omega_execute",
            |mut caller: Caller<'_, StoreData>,
             circuit_id: i32,
             params_ptr: i32,
             num_params: i32,
             observable_id: i32|
             -> f64 {
                let params = read_f64_array(&mut caller, params_ptr, num_params);
                let host = Arc::clone(caller.data().host());
                let state = host.lock().unwrap();
                match state.execute_expectation(circuit_id as u32, &params, observable_id as u32) {
                    Ok(val) => val,
                    Err(e) => {
                        eprintln!("omega_execute error: {}", e);
                        f64::NAN
                    }
                }
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_gradient(circuit_id: i32, params_ptr: i32, num_params: i32, observable_id: i32, grad_out_ptr: i32)
    linker
        .func_wrap(
            "env",
            "omega_gradient",
            |mut caller: Caller<'_, StoreData>,
             circuit_id: i32,
             params_ptr: i32,
             num_params: i32,
             observable_id: i32,
             grad_out_ptr: i32| {
                let params = read_f64_array(&mut caller, params_ptr, num_params);
                let host = Arc::clone(caller.data().host());
                let state = host.lock().unwrap();
                match state.compute_gradient(circuit_id as u32, &params, observable_id as u32) {
                    Ok(grad) => {
                        write_f64_array(&mut caller, grad_out_ptr, &grad);
                    }
                    Err(e) => {
                        eprintln!("omega_gradient error: {}", e);
                    }
                }
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_compute_functional_gradient(
    //     circuit_id: i32,
    //     params_ptr: i32, num_params: i32,
    //     spec_ptr: i32, spec_len: i32,
    //     method_byte: i32,         // 0 = diagonal (exact), 1 = score-fn (REINFORCE)
    //     score_fn_shots: i32,      // only used when method_byte == 1; 0 → default 1024
    //     grad_out_ptr: i32,
    // )
    //
    // Computes ∂ E[f(x; θ)] / ∂θ for a Functional spec passed as a
    // JSON string in guest memory. Two shapes accepted (same as
    // `omega-run --gradient-of-fn`):
    //   `{"qubo":{"n":N,"Q":[[i,j,c],...]}}`
    //   `{"table":[["bits",value],...],"num_qubits":N}`
    // Writes `num_params` f64 gradients to `grad_out_ptr`. On any
    // error, writes zeros and prints a host-side stderr message —
    // matches the behaviour of `omega_gradient`.
    linker
        .func_wrap(
            "env",
            "omega_compute_functional_gradient",
            |mut caller: Caller<'_, StoreData>,
             circuit_id: i32,
             params_ptr: i32,
             num_params: i32,
             spec_ptr: i32,
             spec_len: i32,
             method_byte: i32,
             score_fn_shots: i32,
             grad_out_ptr: i32| {
                let params = read_f64_array(&mut caller, params_ptr, num_params);
                let spec_bytes = read_bytes(&mut caller, spec_ptr, spec_len);
                let spec = match std::str::from_utf8(&spec_bytes) {
                    Ok(s) => s.to_string(),
                    Err(e) => {
                        eprintln!("omega_compute_functional_gradient: invalid UTF-8 in spec: {e}");
                        write_f64_array(&mut caller, grad_out_ptr, &vec![0.0; num_params as usize]);
                        return;
                    }
                };
                let host = Arc::clone(caller.data().host());
                let state = host.lock().unwrap();
                match state.compute_functional_gradient_from_spec(
                    circuit_id as u32,
                    &params,
                    &spec,
                    method_byte.max(0) as u8,
                    score_fn_shots.max(0) as u32,
                ) {
                    Ok(grad) => {
                        write_f64_array(&mut caller, grad_out_ptr, &grad);
                    }
                    Err(e) => {
                        eprintln!("omega_compute_functional_gradient error: {e}");
                        write_f64_array(&mut caller, grad_out_ptr, &vec![0.0; num_params as usize]);
                    }
                }
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_report_progress(iteration: i32, value_bits: i64)
    // (f64 passed as i64 bits to avoid WASM f64 limitations in some toolchains)
    linker
        .func_wrap(
            "env",
            "omega_report_progress",
            |caller: Caller<'_, StoreData>, iteration: i32, value_bits: i64| {
                let value = f64::from_bits(value_bits as u64);
                let host = Arc::clone(caller.data().host());
                let mut state = host.lock().unwrap();
                state.progress.push((iteration as u32, value));
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_report_result(params_ptr: i32, num_params: i32, value_bits: i64, iterations: i32)
    linker
        .func_wrap(
            "env",
            "omega_report_result",
            |mut caller: Caller<'_, StoreData>,
             params_ptr: i32,
             num_params: i32,
             value_bits: i64,
             iterations: i32| {
                let params = read_f64_array(&mut caller, params_ptr, num_params);
                let value = f64::from_bits(value_bits as u64);
                let host = Arc::clone(caller.data().host());
                let mut state = host.lock().unwrap();
                state.final_result = Some(WasmResult {
                    optimal_params: params,
                    optimal_value: value,
                    iterations: iterations as u32,
                });
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_execute_shots(circuit_id: i32, params_ptr: i32, num_params: i32, shots: i32,
    //                     counts_out_ptr: i32, max_entries: i32) -> i32
    // Returns number of unique outcomes written. Each entry: [bitstring_f64, count_f64].
    linker
        .func_wrap(
            "env",
            "omega_execute_shots",
            |mut caller: Caller<'_, StoreData>,
             circuit_id: i32,
             params_ptr: i32,
             num_params: i32,
             shots: i32,
             counts_out_ptr: i32,
             max_entries: i32|
             -> i32 {
                let params = read_f64_array(&mut caller, params_ptr, num_params);
                let host = Arc::clone(caller.data().host());
                let state = host.lock().unwrap();
                match state.execute_with_shots(circuit_id as u32, &params, shots as u32, Some(42)) {
                    Ok(counts) => {
                        let mut sorted: Vec<_> = counts.into_iter().collect();
                        sorted.sort_by_key(|&(k, _)| k);
                        let n = sorted.len().min(max_entries as usize);
                        let mut out = Vec::with_capacity(n * 2);
                        for &(bits, count) in sorted.iter().take(n) {
                            out.push(bits as f64);
                            out.push(count as f64);
                        }
                        write_f64_array(&mut caller, counts_out_ptr, &out);
                        n as i32
                    }
                    Err(e) => {
                        eprintln!("omega_execute_shots error: {}", e);
                        -1
                    }
                }
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_read_classical(circuit_id: i32, params_ptr: i32, num_params: i32,
    //                       seed_bits: i64, out_ptr: i32, max_len: i32) -> i32
    //
    // Run one Collapse-mode trajectory of `circuit_id` and write the
    // final classical-register bits (one u8 per cbit, LSB-first) at
    // `out_ptr`. Returns the number of bits written, or -1 on error.
    // `seed_bits < 0` selects a non-deterministic host RNG; otherwise
    // the value is reinterpreted as `u64` and seeds the per-trajectory
    // measurement RNG. Matches Qiskit's last-write-wins creg semantics.
    linker
        .func_wrap(
            "env",
            "omega_read_classical",
            |mut caller: Caller<'_, StoreData>,
             circuit_id: i32,
             params_ptr: i32,
             num_params: i32,
             seed_bits: i64,
             out_ptr: i32,
             max_len: i32|
             -> i32 {
                let params = read_f64_array(&mut caller, params_ptr, num_params);
                let seed = if seed_bits < 0 {
                    None
                } else {
                    Some(seed_bits as u64)
                };
                let host = Arc::clone(caller.data().host());
                let state = host.lock().unwrap();
                match state.execute_classical(circuit_id as u32, &params, seed) {
                    Ok(bits) => {
                        let n = (bits.len()).min(max_len.max(0) as usize);
                        write_bytes(&mut caller, out_ptr, &bits[..n]);
                        n as i32
                    }
                    Err(e) => {
                        eprintln!("omega_read_classical error: {}", e);
                        -1
                    }
                }
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_execute_multi(circuit_id: i32, params_ptr: i32, num_params: i32,
    //                     observable_ids_ptr: i32, num_observables: i32, results_out_ptr: i32)
    linker
        .func_wrap(
            "env",
            "omega_execute_multi",
            |mut caller: Caller<'_, StoreData>,
             circuit_id: i32,
             params_ptr: i32,
             num_params: i32,
             obs_ids_ptr: i32,
             num_obs: i32,
             results_out_ptr: i32| {
                let params = read_f64_array(&mut caller, params_ptr, num_params);
                let obs_ids_f64 = read_f64_array(&mut caller, obs_ids_ptr, num_obs);
                let obs_ids: Vec<u32> = obs_ids_f64.iter().map(|&x| x as u32).collect();
                let host = Arc::clone(caller.data().host());
                let state = host.lock().unwrap();
                match state.execute_multi(circuit_id as u32, &params, &obs_ids) {
                    Ok(results) => {
                        write_f64_array(&mut caller, results_out_ptr, &results);
                    }
                    Err(e) => {
                        eprintln!("omega_execute_multi error: {}", e);
                    }
                }
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_read_statevector(circuit_id: i32, params_ptr: i32, num_params: i32,
    //                        out_ptr: i32, max_pairs: i32) -> i32
    //
    // Write the exact statevector of `circuit_id` (measurement gates skipped)
    // as interleaved (re, im) f64 pairs at `out_ptr`: amplitude k occupies
    // out[2k], out[2k+1]. Returns the number of AMPLITUDES written (≤ max_pairs),
    // or -1 on error. Lets a guest cross-check amplitude-level quantities.
    linker
        .func_wrap(
            "env",
            "omega_read_statevector",
            |mut caller: Caller<'_, StoreData>,
             circuit_id: i32,
             params_ptr: i32,
             num_params: i32,
             out_ptr: i32,
             max_pairs: i32|
             -> i32 {
                let params = read_f64_array(&mut caller, params_ptr, num_params);
                let host = Arc::clone(caller.data().host());
                let state = host.lock().unwrap();
                match state.statevector_interleaved(circuit_id as u32, &params) {
                    Ok(interleaved) => {
                        let n_amps = interleaved.len() / 2;
                        let n = n_amps.min(max_pairs.max(0) as usize);
                        write_f64_array(&mut caller, out_ptr, &interleaved[..n * 2]);
                        n as i32
                    }
                    Err(e) => {
                        eprintln!("omega_read_statevector error: {}", e);
                        -1
                    }
                }
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // ---- Parametrizable-guest extensions ----
    //
    // These let a guest discover circuit shape, read a JSON config payload
    // staged by the host, and register new circuits / observables / QAOA
    // problems at runtime — so the guest is no longer pinned to a single
    // hard-coded `(NUM_PARAMS, circuit_id, observable_id)` triple.

    // omega_input_len() -> i32
    // Length of the input payload staged via `HostState::set_input`. Returns
    // 0 if no input was staged. Always non-negative.
    linker
        .func_wrap(
            "env",
            "omega_input_len",
            |caller: Caller<'_, StoreData>| -> i32 {
                let host = Arc::clone(caller.data().host());
                let state = host.lock().unwrap();
                state.input_bytes.len() as i32
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_input_read(out_ptr: i32, max_len: i32) -> i32
    // Copy up to `max_len` bytes from the staged input payload into the
    // guest's linear memory at `out_ptr`. Returns the number of bytes
    // actually written.
    linker
        .func_wrap(
            "env",
            "omega_input_read",
            |mut caller: Caller<'_, StoreData>, out_ptr: i32, max_len: i32| -> i32 {
                let bytes = {
                    let host = Arc::clone(caller.data().host());
                    let state = host.lock().unwrap();
                    let n = (max_len as usize).min(state.input_bytes.len());
                    state.input_bytes[..n].to_vec()
                };
                write_bytes(&mut caller, out_ptr, &bytes);
                bytes.len() as i32
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_register_qasm(src_ptr: i32, src_len: i32) -> i32
    // Parse a QASM 2.0 / OPTICQASM source string from guest memory, register
    // it as a new circuit, and return its ID (>= 1). Returns -1 on parse
    // failure (the host stderr carries the parser message).
    linker
        .func_wrap(
            "env",
            "omega_register_qasm",
            |mut caller: Caller<'_, StoreData>, src_ptr: i32, src_len: i32| -> i32 {
                let bytes = read_bytes(&mut caller, src_ptr, src_len);
                let src = match std::str::from_utf8(&bytes) {
                    Ok(s) => s.to_string(),
                    Err(e) => {
                        eprintln!("omega_register_qasm: invalid UTF-8: {}", e);
                        return -1;
                    }
                };
                let circuit = match lower_to_ir(&src) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("omega_register_qasm: parse error: {}", e);
                        return -1;
                    }
                };
                let host = Arc::clone(caller.data().host());
                let mut state = host.lock().unwrap();
                state.register_circuit(circuit) as i32
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_register_observable_str(spec_ptr: i32, spec_len: i32) -> i32
    // Parse a Pauli-sum spec like "0.5*Z0+0.3*X0X1", register, return ID.
    // Returns -1 on parse failure.
    linker
        .func_wrap(
            "env",
            "omega_register_observable_str",
            |mut caller: Caller<'_, StoreData>, spec_ptr: i32, spec_len: i32| -> i32 {
                let bytes = read_bytes(&mut caller, spec_ptr, spec_len);
                let spec = match std::str::from_utf8(&bytes) {
                    Ok(s) => s.to_string(),
                    Err(e) => {
                        eprintln!("omega_register_observable_str: invalid UTF-8: {}", e);
                        return -1;
                    }
                };
                let observable = match Observable::parse(&spec) {
                    Ok(o) => o,
                    Err(e) => {
                        eprintln!("omega_register_observable_str: parse error: {}", e);
                        return -1;
                    }
                };
                let host = Arc::clone(caller.data().host());
                let mut state = host.lock().unwrap();
                state.register_observable(observable) as i32
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_circuit_num_params(circuit_id: i32) -> i32
    // Number of free symbols (i.e. parameters) on a registered circuit.
    // Returns -1 if the circuit ID is unknown.
    linker
        .func_wrap(
            "env",
            "omega_circuit_num_params",
            |caller: Caller<'_, StoreData>, circuit_id: i32| -> i32 {
                let host = Arc::clone(caller.data().host());
                let state = host.lock().unwrap();
                match state.circuits.get(&(circuit_id as u32)) {
                    Some(c) => c.symbols.len() as i32,
                    None => -1,
                }
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_circuit_num_qubits(circuit_id: i32) -> i32
    // Qubit count of a registered circuit. -1 on unknown ID.
    linker
        .func_wrap(
            "env",
            "omega_circuit_num_qubits",
            |caller: Caller<'_, StoreData>, circuit_id: i32| -> i32 {
                let host = Arc::clone(caller.data().host());
                let state = host.lock().unwrap();
                match state.circuits.get(&(circuit_id as u32)) {
                    Some(c) => c.num_qubits as i32,
                    None => -1,
                }
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    // omega_qaoa_from_qubo(qubo_json_ptr: i32, qubo_json_len: i32, depth: i32, out_ids_ptr: i32) -> f64
    // Build a QAOA circuit + observable from a QUBO JSON payload
    // (`{"n": N, "Q": [[i, j, c], ...]}`), register both, and write
    // [circuit_id: i32, observable_id: i32] (8 bytes total) at `out_ids_ptr`.
    // Returns the Ising offset (a constant the QAOA cost must be shifted
    // by to recover the QUBO objective). Returns NaN on parse failure;
    // both IDs are written as 0 in that case.
    linker
        .func_wrap(
            "env",
            "omega_qaoa_from_qubo",
            |mut caller: Caller<'_, StoreData>,
             qubo_json_ptr: i32,
             qubo_json_len: i32,
             depth: i32,
             out_ids_ptr: i32|
             -> f64 {
                let bytes = read_bytes(&mut caller, qubo_json_ptr, qubo_json_len);
                let json = match std::str::from_utf8(&bytes) {
                    Ok(s) => s.to_string(),
                    Err(e) => {
                        eprintln!("omega_qaoa_from_qubo: invalid UTF-8: {}", e);
                        write_i32_pair(&mut caller, out_ids_ptr, 0, 0);
                        return f64::NAN;
                    }
                };
                let host = Arc::clone(caller.data().host());
                let mut state = host.lock().unwrap();
                match state.build_qaoa_from_qubo_json(&json, depth as usize) {
                    Ok((cid, oid, offset)) => {
                        write_i32_pair(&mut caller, out_ids_ptr, cid as i32, oid as i32);
                        offset
                    }
                    Err(e) => {
                        eprintln!("omega_qaoa_from_qubo: {}", e);
                        write_i32_pair(&mut caller, out_ids_ptr, 0, 0);
                        f64::NAN
                    }
                }
            },
        )
        .map_err(|e| OmegaError::Backend(e.to_string()))?;

    Ok(())
}

/// Read an f64 array from WASM linear memory.
fn read_f64_array(caller: &mut Caller<'_, StoreData>, ptr: i32, count: i32) -> Vec<f64> {
    let memory = caller.get_export("memory").and_then(|e| e.into_memory());
    match memory {
        Some(mem) => {
            let data = mem.data(&caller);
            let offset = ptr as usize;
            let byte_len = count as usize * 8;
            if offset + byte_len > data.len() {
                return vec![0.0; count as usize];
            }
            let mut result = Vec::with_capacity(count as usize);
            for i in 0..count as usize {
                let bytes: [u8; 8] = data[offset + i * 8..offset + i * 8 + 8].try_into().unwrap();
                result.push(f64::from_le_bytes(bytes));
            }
            result
        }
        None => vec![0.0; count as usize],
    }
}

/// Write an f64 array to WASM linear memory.
fn write_f64_array(caller: &mut Caller<'_, StoreData>, ptr: i32, values: &[f64]) {
    let memory = caller.get_export("memory").and_then(|e| e.into_memory());
    if let Some(mem) = memory {
        let data = mem.data_mut(caller);
        let offset = ptr as usize;
        for (i, &val) in values.iter().enumerate() {
            let bytes = val.to_le_bytes();
            let start = offset + i * 8;
            if start + 8 <= data.len() {
                data[start..start + 8].copy_from_slice(&bytes);
            }
        }
    }
}

/// Read raw bytes from WASM linear memory.
fn read_bytes(caller: &mut Caller<'_, StoreData>, ptr: i32, len: i32) -> Vec<u8> {
    let memory = caller.get_export("memory").and_then(|e| e.into_memory());
    match memory {
        Some(mem) => {
            let data = mem.data(&caller);
            let offset = ptr as usize;
            let n = len as usize;
            if offset + n > data.len() {
                return Vec::new();
            }
            data[offset..offset + n].to_vec()
        }
        None => Vec::new(),
    }
}

/// Write raw bytes to WASM linear memory.
fn write_bytes(caller: &mut Caller<'_, StoreData>, ptr: i32, bytes: &[u8]) {
    let memory = caller.get_export("memory").and_then(|e| e.into_memory());
    if let Some(mem) = memory {
        let data = mem.data_mut(caller);
        let offset = ptr as usize;
        let end = offset + bytes.len();
        if end <= data.len() {
            data[offset..end].copy_from_slice(bytes);
        }
    }
}

/// Write two consecutive i32s (little-endian, 8 bytes total) to WASM memory.
fn write_i32_pair(caller: &mut Caller<'_, StoreData>, ptr: i32, a: i32, b: i32) {
    let memory = caller.get_export("memory").and_then(|e| e.into_memory());
    if let Some(mem) = memory {
        let data = mem.data_mut(caller);
        let offset = ptr as usize;
        if offset + 8 <= data.len() {
            data[offset..offset + 4].copy_from_slice(&a.to_le_bytes());
            data[offset + 4..offset + 8].copy_from_slice(&b.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::HostState;

    #[test]
    fn test_wasm_runner_creation() {
        let state = HostState::new();
        let runner = WasmRunner::new(state);
        assert!(runner.is_ok());
    }

    #[test]
    fn rng_seed_builder_round_trip() {
        // Verify the seed setter round-trips. The actual guest-visible
        // determinism check requires a WASM module that calls
        // random_get and reports its bytes via omega_report_progress;
        // a guest of that shape doesn't exist in dist/wasm/ today, so
        // the runtime-level reproducibility test lives as a TODO.
        let state = HostState::new();
        let mut runner = WasmRunner::new(state).expect("runner");
        assert!(runner.rng_seed.is_none());
        runner.rng_seed(Some(42));
        assert_eq!(runner.rng_seed, Some(42));
        runner.rng_seed(None);
        assert!(runner.rng_seed.is_none());
    }

    // The OMEGA_WASM_MEMORY_LIMIT env var is process-wide; serialise
    // tests that mutate it, mirroring the CRL_ENV_LOCK / POLICY_ENV_LOCK
    // patterns elsewhere in the workspace.
    static MEM_LIMIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn host_memory_limit_default_when_env_unset() {
        let _g = MEM_LIMIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("OMEGA_WASM_MEMORY_LIMIT");
        assert_eq!(host_memory_limit(), DEFAULT_GUEST_MEMORY_LIMIT);
    }

    #[test]
    fn host_memory_limit_reads_env_var() {
        let _g = MEM_LIMIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var("OMEGA_WASM_MEMORY_LIMIT", "67108864"); // 64 MiB
        assert_eq!(host_memory_limit(), 67_108_864);
        std::env::remove_var("OMEGA_WASM_MEMORY_LIMIT");
    }

    #[test]
    fn host_memory_limit_falls_back_on_garbage_env() {
        // A non-numeric value must fall back to the default rather
        // than panicking — a misconfigured operator shouldn't crash
        // the boot.
        let _g = MEM_LIMIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var("OMEGA_WASM_MEMORY_LIMIT", "not-a-number");
        assert_eq!(host_memory_limit(), DEFAULT_GUEST_MEMORY_LIMIT);
        std::env::remove_var("OMEGA_WASM_MEMORY_LIMIT");
    }

    #[test]
    fn host_memory_limit_default_is_256_mib() {
        // Pin the documented default so a future bump is a deliberate
        // call rather than silent drift. SECURITY_CHECKLIST.md cites
        // this number as the WASM memory cap.
        assert_eq!(DEFAULT_GUEST_MEMORY_LIMIT, 256 * 1024 * 1024);
    }

    #[test]
    fn host_memory_limit_zero_env_is_honoured_literally() {
        // OMEGA_WASM_MEMORY_LIMIT=0 currently produces a zero-byte
        // cap (which would refuse any guest memory). Pin the literal
        // behaviour so anyone tempted to "treat 0 as default" reads
        // this test first.
        let _g = MEM_LIMIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var("OMEGA_WASM_MEMORY_LIMIT", "0");
        assert_eq!(host_memory_limit(), 0);
        std::env::remove_var("OMEGA_WASM_MEMORY_LIMIT");
    }

    #[test]
    fn default_guest_instances_is_a_small_constant() {
        // A small cap stops a malicious module from instantiating
        // thousands of WASM instances/tables/memories at once. Pin
        // the value so a future bump (e.g. for a multi-instance
        // pipeline) is a deliberate change.
        assert_eq!(DEFAULT_GUEST_INSTANCES, 4);
    }
}
