//! C FFI surface for the Omega Functions quantum circuit runtime.
//!
//! Provides a stable C ABI for creating runtimes, parsing circuits,
//! executing them, and reading results.
//!
//! # Safety
//!
//! All public `extern "C"` functions in this crate are `unsafe` because they
//! take raw pointers and lengths from C callers. Callers must ensure pointers
//! are valid (non-null, properly aligned, pointing to live allocations of the
//! advertised type and length) and that any returned handles/pointers are
//! freed only through this crate's free functions. See `include/omega.h`.

#![allow(clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::slice;

use omega_backend_photonics::PhotonicsBackend;
use omega_backend_statevector::StatevectorBackend;
use omega_core::circuit::CircuitType;
use omega_core::executor::{Backend, ExecConfig, ExecResult};
use omega_core::params::ParameterBinding;
use omega_parser::lower_to_ir;

/// Opaque runtime handle.
pub struct OmegaRuntime {
    circuits: HashMap<u32, omega_core::circuit::CircuitIR>,
    next_circuit_id: u32,
}

/// Opaque result handle.
pub struct OmegaResult {
    result: ExecResult,
    _num_qubits: u32,
    /// Cached counts for C access
    cached_bitstrings: Vec<u64>,
    cached_counts: Vec<u32>,
}

// ---- Lifecycle ----

#[no_mangle]
pub extern "C" fn omega_runtime_new() -> *mut OmegaRuntime {
    Box::into_raw(Box::new(OmegaRuntime {
        circuits: HashMap::new(),
        next_circuit_id: 1,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn omega_runtime_free(rt: *mut OmegaRuntime) {
    if !rt.is_null() {
        drop(Box::from_raw(rt));
    }
}

// ---- Circuit Registration ----

/// Parse a QASM or OPTICQASM source and register the circuit.
/// Returns a circuit ID > 0 on success, 0 on failure.
#[no_mangle]
pub unsafe extern "C" fn omega_circuit_from_source(
    rt: *mut OmegaRuntime,
    source: *const c_char,
    source_len: usize,
) -> u32 {
    if rt.is_null() || source.is_null() {
        return 0;
    }
    let rt = &mut *rt;
    let src = if source_len > 0 {
        let bytes = slice::from_raw_parts(source as *const u8, source_len);
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => return 0,
        }
    } else {
        match CStr::from_ptr(source).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return 0,
        }
    };

    match lower_to_ir(&src) {
        Ok(circuit) => {
            let id = rt.next_circuit_id;
            rt.next_circuit_id += 1;
            rt.circuits.insert(id, circuit);
            id
        }
        Err(_) => 0,
    }
}

/// Get the number of qubits/modes in a circuit.
#[no_mangle]
pub unsafe extern "C" fn omega_circuit_num_qubits(rt: *const OmegaRuntime, circuit_id: u32) -> u32 {
    if rt.is_null() {
        return 0;
    }
    let rt = &*rt;
    rt.circuits
        .get(&circuit_id)
        .map(|c| c.num_qubits)
        .unwrap_or(0)
}

/// Get the number of free parameters in a circuit.
#[no_mangle]
pub unsafe extern "C" fn omega_circuit_num_params(rt: *const OmegaRuntime, circuit_id: u32) -> u32 {
    if rt.is_null() {
        return 0;
    }
    let rt = &*rt;
    rt.circuits
        .get(&circuit_id)
        .map(|c| c.num_free_symbols() as u32)
        .unwrap_or(0)
}

/// Check if circuit is photonic (1) or gate-based (0).
#[no_mangle]
pub unsafe extern "C" fn omega_circuit_is_photonic(
    rt: *const OmegaRuntime,
    circuit_id: u32,
) -> i32 {
    if rt.is_null() {
        return -1;
    }
    let rt = &*rt;
    match rt.circuits.get(&circuit_id) {
        Some(c) => {
            if c.circuit_type == CircuitType::Photonic {
                1
            } else {
                0
            }
        }
        None => -1,
    }
}

// ---- Execution ----

/// Execute a circuit with the given parameters.
///
/// - `params`: array of f64 parameter values, one per free symbol (ordered by symbol ID).
/// - `num_params`: length of params array.
/// - `shots`: number of measurement shots. 0 = exact statevector/probabilities.
/// - `seed`: random seed. 0 = random.
///
/// Returns a result handle, or null on error.
#[no_mangle]
pub unsafe extern "C" fn omega_execute(
    rt: *const OmegaRuntime,
    circuit_id: u32,
    params: *const f64,
    num_params: u32,
    shots: u32,
    seed: u64,
) -> *mut OmegaResult {
    if rt.is_null() {
        return ptr::null_mut();
    }
    let rt = &*rt;

    let circuit = match rt.circuits.get(&circuit_id) {
        Some(c) => c,
        None => return ptr::null_mut(),
    };

    // Bind parameters
    let mut binding = ParameterBinding::new();
    let param_slice = if !params.is_null() && num_params > 0 {
        slice::from_raw_parts(params, num_params as usize)
    } else {
        &[]
    };

    let mut symbol_ids: Vec<u32> = circuit.symbols.keys().copied().collect();
    symbol_ids.sort();
    for (i, &sym_id) in symbol_ids.iter().enumerate() {
        let value = if i < param_slice.len() {
            param_slice[i]
        } else {
            0.0
        };
        binding.bind(sym_id, value);
    }

    let config = ExecConfig {
        shots: if shots == 0 { None } else { Some(shots) },
        seed: if seed == 0 { None } else { Some(seed) },
        ..Default::default()
    };

    // Execute with appropriate backend
    let result = match circuit.circuit_type {
        CircuitType::GateBased => {
            let backend = StatevectorBackend::new();
            backend.execute(circuit, &binding, &config)
        }
        CircuitType::Photonic => {
            let backend = PhotonicsBackend::new();
            backend.execute(circuit, &binding, &config)
        }
    };

    match result {
        Ok(exec_result) => {
            let mut omega_result = OmegaResult {
                _num_qubits: circuit.num_qubits,
                cached_bitstrings: Vec::new(),
                cached_counts: Vec::new(),
                result: exec_result,
            };

            // Pre-cache counts for C access.
            //
            // `omega_result_get_counts` writes outcomes into a `*mut u64` — a
            // PUBLISHED C header, so it cannot be widened here without breaking
            // every existing caller. An outcome over 64 qubits therefore cannot
            // be expressed through this entry point at all.
            //
            // It is dropped, not truncated, and the cache is left EMPTY so the
            // count reads as zero rather than as a plausible wrong histogram.
            // Truncating would be worse here than anywhere else in the
            // codebase: a C caller has no way to notice. A wide API is the
            // follow-up; silently lying is not.
            if let ExecResult::Counts(ref counts) = omega_result.result {
                if counts.keys().all(|o| o.as_u64().is_some()) {
                    let mut pairs: Vec<_> = counts.iter().collect();
                    pairs.sort_by(|a, b| b.1.cmp(a.1));
                    omega_result.cached_bitstrings = pairs
                        .iter()
                        .map(|(o, _)| o.as_u64().expect("checked above"))
                        .collect();
                    omega_result.cached_counts = pairs.iter().map(|(_, &ct)| ct).collect();
                } else {
                    eprintln!(
                        "[omega-ffi] counts span more than 64 qubits and cannot be \
                         returned through omega_result_get_counts (u64 outcomes); \
                         reporting zero outcomes rather than a truncated histogram"
                    );
                }
            }

            Box::into_raw(Box::new(omega_result))
        }
        Err(_) => ptr::null_mut(),
    }
}

// ---- Result Access ----

/// Get the number of unique measurement outcomes.
#[no_mangle]
pub unsafe extern "C" fn omega_result_num_counts(result: *const OmegaResult) -> u32 {
    if result.is_null() {
        return 0;
    }
    let r = &*result;
    r.cached_bitstrings.len() as u32
}

/// Get the bitstrings and counts arrays.
///
/// `bitstrings_out` and `counts_out` must each point to at least
/// `omega_result_num_counts(result)` elements. Callers that don't size
/// the buffers correctly invoke undefined behaviour. As a safety net,
/// passing a null `result`, `bitstrings_out`, or `counts_out` makes
/// this function a no-op rather than UB. (The bounded `_n` variant
/// below is the recommended call for new callers — it caps at a
/// caller-supplied length and returns the number of elements written.)
#[no_mangle]
pub unsafe extern "C" fn omega_result_get_counts(
    result: *const OmegaResult,
    bitstrings_out: *mut u64,
    counts_out: *mut u32,
) {
    if result.is_null() || bitstrings_out.is_null() || counts_out.is_null() {
        return;
    }
    let r = &*result;
    for (i, (&bs, &ct)) in r
        .cached_bitstrings
        .iter()
        .zip(r.cached_counts.iter())
        .enumerate()
    {
        *bitstrings_out.add(i) = bs;
        *counts_out.add(i) = ct;
    }
}

/// Bounded variant of `omega_result_get_counts`: writes at most
/// `max_len` (bitstring, count) pairs and returns the number actually
/// written. Recommended for new callers; the unbounded version is
/// retained for back-compat with the existing C header.
///
/// Null `result` / `bitstrings_out` / `counts_out` returns 0 without
/// touching memory. Returns 0 if `max_len` is 0.
#[no_mangle]
pub unsafe extern "C" fn omega_result_get_counts_n(
    result: *const OmegaResult,
    bitstrings_out: *mut u64,
    counts_out: *mut u32,
    max_len: u32,
) -> u32 {
    if result.is_null() || bitstrings_out.is_null() || counts_out.is_null() || max_len == 0 {
        return 0;
    }
    let r = &*result;
    let n = r.cached_bitstrings.len().min(max_len as usize);
    for i in 0..n {
        *bitstrings_out.add(i) = r.cached_bitstrings[i];
        *counts_out.add(i) = r.cached_counts[i];
    }
    n as u32
}

/// Get the statevector length (number of amplitudes).
#[no_mangle]
pub unsafe extern "C" fn omega_result_statevector_len(result: *const OmegaResult) -> u32 {
    if result.is_null() {
        return 0;
    }
    let r = &*result;
    match &r.result {
        ExecResult::Statevector(sv) => sv.len() as u32,
        _ => 0,
    }
}

/// Get the statevector as interleaved (re, im) doubles.
///
/// `out` must point to at least `2 * omega_result_statevector_len(result)`
/// f64s. Callers that don't size the buffer correctly invoke
/// undefined behaviour. Passing a null `result` or `out` makes this
/// function a no-op as a safety net. The bounded `_n` variant below
/// is the recommended call for new code.
#[no_mangle]
pub unsafe extern "C" fn omega_result_get_statevector(result: *const OmegaResult, out: *mut f64) {
    if result.is_null() || out.is_null() {
        return;
    }
    let r = &*result;
    if let ExecResult::Statevector(ref sv) = r.result {
        for (i, amp) in sv.iter().enumerate() {
            *out.add(i * 2) = amp.re;
            *out.add(i * 2 + 1) = amp.im;
        }
    }
}

/// Bounded variant of `omega_result_get_statevector`: writes at most
/// `max_pairs` (re, im) f64 pairs (i.e. `2 * max_pairs` f64s total)
/// and returns the number of complex amplitudes actually written.
///
/// Null `result` / `out` returns 0 without touching memory. Returns 0
/// if the result isn't a statevector or `max_pairs` is 0.
#[no_mangle]
pub unsafe extern "C" fn omega_result_get_statevector_n(
    result: *const OmegaResult,
    out: *mut f64,
    max_pairs: u32,
) -> u32 {
    if result.is_null() || out.is_null() || max_pairs == 0 {
        return 0;
    }
    let r = &*result;
    if let ExecResult::Statevector(ref sv) = r.result {
        let n = sv.len().min(max_pairs as usize);
        for (i, amp) in sv.iter().take(n).enumerate() {
            *out.add(i * 2) = amp.re;
            *out.add(i * 2 + 1) = amp.im;
        }
        n as u32
    } else {
        0
    }
}

/// Free a result handle.
#[no_mangle]
pub unsafe extern "C" fn omega_result_free(result: *mut OmegaResult) {
    if !result.is_null() {
        drop(Box::from_raw(result));
    }
}

/// Get the API version.
#[no_mangle]
pub extern "C" fn omega_api_version() -> u32 {
    1
}
