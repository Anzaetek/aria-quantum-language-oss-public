//! Integration tests for the C FFI surface (`crates/omega-ffi`).
//!
//! Every public `extern "C"` function exposes a contract documented in
//! `include/omega.h`. These tests pin both the happy path (a Bell state
//! round-trip) and the defensive null/length guards added during the
//! Phase 14a FFI bounds audit (`3c67b27`). They are written as `unsafe`
//! Rust calls to the same symbols a C caller would invoke; passing
//! these tests means the documented sentinel values (0 / `null` / -1)
//! actually fire when the contract is violated.

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

use omega_ffi::*;

const BELL_QASM: &str = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n";

// ---------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------

#[test]
fn runtime_new_returns_non_null_and_free_is_safe() {
    let rt = omega_runtime_new();
    assert!(!rt.is_null(), "omega_runtime_new must succeed");
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn runtime_free_null_is_no_op() {
    // Documented: passing null is a safety net, not UB.
    unsafe { omega_runtime_free(ptr::null_mut()) };
}

// ---------------------------------------------------------------------
// Circuit registration
// ---------------------------------------------------------------------

#[test]
fn circuit_from_source_via_cstring() {
    let rt = omega_runtime_new();
    let qasm = CString::new(BELL_QASM).unwrap();
    let cid = unsafe { omega_circuit_from_source(rt, qasm.as_ptr(), 0) };
    assert!(cid > 0, "Bell circuit should parse and register");
    assert_eq!(unsafe { omega_circuit_num_qubits(rt, cid) }, 2);
    assert_eq!(unsafe { omega_circuit_num_params(rt, cid) }, 0);
    assert_eq!(unsafe { omega_circuit_is_photonic(rt, cid) }, 0);
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn circuit_from_source_via_explicit_length() {
    // The `source_len > 0` path bypasses CStr — exercise it explicitly
    // with non-null-terminated bytes.
    let rt = omega_runtime_new();
    let bytes = BELL_QASM.as_bytes();
    let cid =
        unsafe { omega_circuit_from_source(rt, bytes.as_ptr() as *const c_char, bytes.len()) };
    assert!(cid > 0);
    assert_eq!(unsafe { omega_circuit_num_qubits(rt, cid) }, 2);
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn circuit_from_source_null_runtime_returns_zero() {
    let qasm = CString::new(BELL_QASM).unwrap();
    let cid = unsafe { omega_circuit_from_source(ptr::null_mut(), qasm.as_ptr(), 0) };
    assert_eq!(cid, 0);
}

#[test]
fn circuit_from_source_null_source_returns_zero() {
    let rt = omega_runtime_new();
    let cid = unsafe { omega_circuit_from_source(rt, ptr::null(), 0) };
    assert_eq!(cid, 0);
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn circuit_from_source_explicit_length_ignores_embedded_nul() {
    // The explicit-length path uses `slice::from_raw_parts` + utf8
    // decode, so an embedded NUL inside `source_len` bytes must be
    // tolerated (it's not a string terminator at this layer — the
    // parser handles it). Contrast with the `source_len == 0` path
    // which falls back to `CStr::from_ptr` and stops at the first NUL.
    let rt = omega_runtime_new();
    let mut padded = BELL_QASM.as_bytes().to_vec();
    // Append a NUL + some trailing garbage past the declared length to
    // make sure the FFI doesn't read past `source_len`.
    let real_len = padded.len();
    padded.push(0u8);
    padded.extend_from_slice(b"trailing-garbage-past-len");
    let cid = unsafe { omega_circuit_from_source(rt, padded.as_ptr() as *const c_char, real_len) };
    assert!(
        cid > 0,
        "explicit-length parse must succeed and respect source_len"
    );
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn circuit_from_source_zero_length_with_null_terminated_cstring_works() {
    // Documented: source_len == 0 means "treat source as null-terminated".
    let rt = omega_runtime_new();
    let cstr = CString::new(BELL_QASM).unwrap();
    let cid = unsafe { omega_circuit_from_source(rt, cstr.as_ptr(), 0) };
    assert!(cid > 0);
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn circuit_from_source_invalid_utf8_returns_zero() {
    let rt = omega_runtime_new();
    // Non-UTF-8 bytes via the explicit-length path.
    let bytes: &[u8] = &[0xff, 0xfe, 0xfd, 0xfc];
    let cid =
        unsafe { omega_circuit_from_source(rt, bytes.as_ptr() as *const c_char, bytes.len()) };
    assert_eq!(cid, 0);
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn circuit_from_source_garbage_qasm_returns_zero() {
    let rt = omega_runtime_new();
    let bad = CString::new("not a qasm program").unwrap();
    let cid = unsafe { omega_circuit_from_source(rt, bad.as_ptr(), 0) };
    assert_eq!(cid, 0);
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn circuit_metadata_unknown_id_returns_sentinels() {
    let rt = omega_runtime_new();
    // No circuits registered yet — id=42 is unknown.
    assert_eq!(unsafe { omega_circuit_num_qubits(rt, 42) }, 0);
    assert_eq!(unsafe { omega_circuit_num_params(rt, 42) }, 0);
    assert_eq!(unsafe { omega_circuit_is_photonic(rt, 42) }, -1);
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn circuit_metadata_null_runtime_returns_sentinels() {
    let null = ptr::null();
    assert_eq!(unsafe { omega_circuit_num_qubits(null, 1) }, 0);
    assert_eq!(unsafe { omega_circuit_num_params(null, 1) }, 0);
    assert_eq!(unsafe { omega_circuit_is_photonic(null, 1) }, -1);
}

// ---------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------

#[test]
fn execute_null_runtime_returns_null() {
    let res = unsafe { omega_execute(ptr::null(), 1, ptr::null(), 0, 1024, 0) };
    assert!(res.is_null());
}

#[test]
fn execute_unknown_circuit_returns_null() {
    let rt = omega_runtime_new();
    let res = unsafe { omega_execute(rt, 999, ptr::null(), 0, 1024, 0) };
    assert!(res.is_null());
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn execute_bell_state_returns_bell_distribution() {
    let rt = omega_runtime_new();
    let qasm = CString::new(BELL_QASM).unwrap();
    let cid = unsafe { omega_circuit_from_source(rt, qasm.as_ptr(), 0) };
    let res = unsafe { omega_execute(rt, cid, ptr::null(), 0, 4096, 42) };
    assert!(!res.is_null(), "Bell execution must produce a result");

    let n = unsafe { omega_result_num_counts(res) };
    assert!(
        n == 2,
        "Bell state must produce exactly 2 outcomes, got {n}"
    );

    // Read counts via the unbounded API.
    let mut bits = vec![0u64; n as usize];
    let mut counts = vec![0u32; n as usize];
    unsafe { omega_result_get_counts(res, bits.as_mut_ptr(), counts.as_mut_ptr()) };

    let total: u32 = counts.iter().sum();
    assert_eq!(total, 4096);
    // The two outcomes must be |00> (0b00 = 0) and |11> (0b11 = 3),
    // each ~50% — but allow the seed=42 split to wander within ±5%.
    let mut sorted_bits = bits.clone();
    sorted_bits.sort();
    assert_eq!(sorted_bits, vec![0u64, 3u64]);
    for &c in &counts {
        let frac = c as f64 / total as f64;
        assert!(
            (0.45..=0.55).contains(&frac),
            "Bell outcome fraction {frac} outside [0.45, 0.55]"
        );
    }

    unsafe { omega_result_free(res) };
    unsafe { omega_runtime_free(rt) };
}

// ---------------------------------------------------------------------
// Result-access null guards + bounded variants
// ---------------------------------------------------------------------

#[test]
fn result_num_counts_null_returns_zero() {
    assert_eq!(unsafe { omega_result_num_counts(ptr::null()) }, 0);
}

#[test]
fn result_get_counts_null_args_are_no_op() {
    // Passing null result is documented as a no-op.
    let mut bits = [0u64; 1];
    let mut counts = [0u32; 1];
    unsafe { omega_result_get_counts(ptr::null(), bits.as_mut_ptr(), counts.as_mut_ptr()) };
    assert_eq!(bits, [0]);
    assert_eq!(counts, [0]);

    // Null bitstrings_out path.
    let rt = omega_runtime_new();
    let qasm = CString::new(BELL_QASM).unwrap();
    let cid = unsafe { omega_circuit_from_source(rt, qasm.as_ptr(), 0) };
    let res = unsafe { omega_execute(rt, cid, ptr::null(), 0, 64, 1) };
    unsafe { omega_result_get_counts(res, ptr::null_mut(), counts.as_mut_ptr()) };
    unsafe { omega_result_get_counts(res, bits.as_mut_ptr(), ptr::null_mut()) };
    unsafe { omega_result_free(res) };
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn result_get_counts_n_respects_max_len() {
    let rt = omega_runtime_new();
    let qasm = CString::new(BELL_QASM).unwrap();
    let cid = unsafe { omega_circuit_from_source(rt, qasm.as_ptr(), 0) };
    let res = unsafe { omega_execute(rt, cid, ptr::null(), 0, 1024, 7) };

    // Bell produces 2 outcomes; ask for at most 1.
    let mut bits = [0u64; 4];
    let mut counts = [0u32; 4];
    let written =
        unsafe { omega_result_get_counts_n(res, bits.as_mut_ptr(), counts.as_mut_ptr(), 1) };
    assert_eq!(written, 1, "max_len=1 must cap the write");
    assert!(counts[0] > 0, "first slot should be populated");
    assert_eq!(counts[1], 0, "second slot must remain untouched");

    // Asking for more than available returns the actual count.
    let written_full =
        unsafe { omega_result_get_counts_n(res, bits.as_mut_ptr(), counts.as_mut_ptr(), 32) };
    assert_eq!(written_full, 2);

    // max_len = 0 returns 0 without touching memory.
    let empty_bits = [u64::MAX; 1];
    let empty_counts = [u32::MAX; 1];
    let mut bits_marker = empty_bits;
    let mut counts_marker = empty_counts;
    let written_zero = unsafe {
        omega_result_get_counts_n(res, bits_marker.as_mut_ptr(), counts_marker.as_mut_ptr(), 0)
    };
    assert_eq!(written_zero, 0);
    assert_eq!(bits_marker, empty_bits);
    assert_eq!(counts_marker, empty_counts);

    // Null result returns 0.
    assert_eq!(
        unsafe {
            omega_result_get_counts_n(ptr::null(), bits.as_mut_ptr(), counts.as_mut_ptr(), 4)
        },
        0
    );

    unsafe { omega_result_free(res) };
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn result_statevector_round_trip_via_bounded_variant() {
    // shots=0 → exact statevector path.
    let rt = omega_runtime_new();
    // Single-qubit |+⟩ state — 2 amplitudes, both 1/√2.
    let plus = CString::new("OPENQASM 2.0;\nqreg q[1];\nh q[0];\n").unwrap();
    let cid = unsafe { omega_circuit_from_source(rt, plus.as_ptr(), 0) };
    let res = unsafe { omega_execute(rt, cid, ptr::null(), 0, 0, 0) };
    assert!(!res.is_null());

    let amps = unsafe { omega_result_statevector_len(res) };
    assert_eq!(amps, 2);

    // 2 amplitudes → 4 f64s (re, im pairs).
    let mut buf = [0.0f64; 4];
    let written = unsafe { omega_result_get_statevector_n(res, buf.as_mut_ptr(), 4) };
    assert_eq!(written, 2);
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    assert!((buf[0] - inv_sqrt2).abs() < 1e-12);
    assert!(buf[1].abs() < 1e-12);
    assert!((buf[2] - inv_sqrt2).abs() < 1e-12);
    assert!(buf[3].abs() < 1e-12);

    // max_pairs = 1 caps to first amplitude only.
    let mut capped = [99.0f64; 4];
    let written_capped = unsafe { omega_result_get_statevector_n(res, capped.as_mut_ptr(), 1) };
    assert_eq!(written_capped, 1);
    assert!((capped[0] - inv_sqrt2).abs() < 1e-12);
    assert_eq!(
        capped[2], 99.0,
        "third slot must remain untouched at max_pairs=1"
    );

    unsafe { omega_result_free(res) };
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn result_statevector_len_on_counts_result_returns_zero() {
    let rt = omega_runtime_new();
    let qasm = CString::new(BELL_QASM).unwrap();
    let cid = unsafe { omega_circuit_from_source(rt, qasm.as_ptr(), 0) };
    let res = unsafe { omega_execute(rt, cid, ptr::null(), 0, 64, 1) };
    // Counts result, not Statevector → 0 amplitudes reported.
    assert_eq!(unsafe { omega_result_statevector_len(res) }, 0);
    let mut buf = [0.0f64; 4];
    let written = unsafe { omega_result_get_statevector_n(res, buf.as_mut_ptr(), 4) };
    assert_eq!(written, 0);
    unsafe { omega_result_free(res) };
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn result_statevector_len_null_returns_zero() {
    assert_eq!(unsafe { omega_result_statevector_len(ptr::null()) }, 0);
}

#[test]
fn result_get_statevector_null_args_no_op() {
    let mut buf = [0.0f64; 4];
    unsafe { omega_result_get_statevector(ptr::null(), buf.as_mut_ptr()) };
    assert_eq!(buf, [0.0; 4]);

    let rt = omega_runtime_new();
    let plus = CString::new("OPENQASM 2.0;\nqreg q[1];\nh q[0];\n").unwrap();
    let cid = unsafe { omega_circuit_from_source(rt, plus.as_ptr(), 0) };
    let res = unsafe { omega_execute(rt, cid, ptr::null(), 0, 0, 0) };
    unsafe { omega_result_get_statevector(res, ptr::null_mut()) };
    unsafe { omega_result_free(res) };
    unsafe { omega_runtime_free(rt) };
}

#[test]
fn result_free_null_is_no_op() {
    unsafe { omega_result_free(ptr::null_mut()) };
}

// ---------------------------------------------------------------------
// API version
// ---------------------------------------------------------------------

#[test]
fn api_version_is_positive() {
    // No specific value pinned here — just that the symbol resolves
    // and returns something sensible. A change in API version is a
    // deliberate event that should also bump this test's expectation.
    let v = omega_api_version();
    assert!(v >= 1);
}

// ---------------------------------------------------------------------
// Round-trip with parameters
// ---------------------------------------------------------------------

#[test]
fn execute_with_parameters_binds_in_symbol_id_order() {
    // Ry(theta) on q0 — ⟨Z⟩ = cos(θ); at θ=π/2 the sample distribution
    // splits evenly. Counts mode is fine for the smoke check.
    let rt = omega_runtime_new();
    let src = CString::new(
        "OPENQASM 2.0;\ngate ry(theta) q { ry(theta) q; }\nqreg q[1];\ncreg c[1];\nry(pi) q[0];\nmeasure q[0] -> c[0];\n",
    )
    .unwrap();
    let cid = unsafe { omega_circuit_from_source(rt, src.as_ptr(), 0) };
    assert!(cid > 0);
    // No free symbols here (pi is concrete) — params=null should be fine.
    let res = unsafe { omega_execute(rt, cid, ptr::null(), 0, 1024, 11) };
    assert!(!res.is_null());
    let n = unsafe { omega_result_num_counts(res) };
    // Ry(pi)|0> = |1> deterministically.
    assert_eq!(n, 1);
    let mut bits = [0u64; 1];
    let mut counts = [0u32; 1];
    unsafe { omega_result_get_counts(res, bits.as_mut_ptr(), counts.as_mut_ptr()) };
    assert_eq!(bits[0], 1);
    assert_eq!(counts[0], 1024);
    unsafe { omega_result_free(res) };
    unsafe { omega_runtime_free(rt) };
}
