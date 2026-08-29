/**
 * Omega Functions - Quantum Circuit Runtime
 * C API Header
 *
 * Usage:
 *   OmegaRuntime *rt = omega_runtime_new();
 *   uint32_t cid = omega_circuit_from_source(rt, qasm_source, 0);
 *   OmegaResult *res = omega_execute(rt, cid, NULL, 0, 1024, 0);
 *   // ... read results ...
 *   omega_result_free(res);
 *   omega_runtime_free(rt);
 */

#ifndef OMEGA_H
#define OMEGA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef struct OmegaRuntime OmegaRuntime;
typedef struct OmegaResult OmegaResult;

/* API version.
 *
 * A single integer, incremented when an EXISTING entry point changes behaviour
 * or signature. Adding a new function does not bump it, so a caller that does
 * not use the new function is unaffected.
 *
 * Check this BEFORE calling anything else and refuse a version you do not
 * recognise, rather than discovering the mismatch inside the library.
 *
 *   1  initial surface
 *   2  omega_execute REFUSES a parameter count that does not match the
 *      circuit's free-symbol count. Version 1 padded a short array with 0.0
 *      and ignored extras, which silently executed a DIFFERENT circuit.
 *      Pass one value per symbol; omega_circuit_num_params gives the count.
 */
uint32_t omega_api_version(void);

/* Lifecycle */
OmegaRuntime* omega_runtime_new(void);
void omega_runtime_free(OmegaRuntime* rt);

/* Circuit registration.
 * source: QASM 2.0 or OPTICQASM 1.0 source string.
 * source_len: length in bytes, or 0 for null-terminated.
 * Returns circuit ID > 0 on success, 0 on failure. */
uint32_t omega_circuit_from_source(OmegaRuntime* rt, const char* source, size_t source_len);

/* Circuit queries */
uint32_t omega_circuit_num_qubits(const OmegaRuntime* rt, uint32_t circuit_id);
uint32_t omega_circuit_num_params(const OmegaRuntime* rt, uint32_t circuit_id);
int32_t  omega_circuit_is_photonic(const OmegaRuntime* rt, uint32_t circuit_id);

/* Execution.
 * params: array of parameter values (one per free symbol, sorted by symbol ID).
 * num_params: length of params array.
 * shots: number of measurement shots. 0 = exact statevector.
 * seed: random seed. 0 = use system entropy.
 * Returns result handle, or NULL on error. */
OmegaResult* omega_execute(
    const OmegaRuntime* rt,
    uint32_t circuit_id,
    const double* params,
    uint32_t num_params,
    uint32_t shots,
    uint64_t seed
);

/* Expectation value <psi|O|psi>.
 *
 * observable: null-terminated Pauli string, e.g. "Z0 Z1" or "0.5*X0 + Z1".
 *             Spaces are ignored.
 * params/num_params: one value per free symbol, exactly (see version 2 above).
 * out: receives the value ONLY on success.
 *
 * Returns 0 on success, non-zero on failure.
 *
 * A scalar has no NULL to signal failure, and a NaN sentinel would be a trap —
 * NaN compares false against everything including itself, so a caller that
 * forgot to check would propagate it silently. Hence a status code and an
 * out-parameter that is LEFT UNTOUCHED on error: ignoring the status code reads
 * back your own initial value, not a stale or partial answer.
 *
 * Refused (non-zero, *out untouched): null rt/observable/out, an unknown
 * circuit id, an unparseable observable, a parameter-count mismatch, and a
 * photonic circuit (which has no Pauli observable). */
int32_t omega_expectation(
    const OmegaRuntime* rt,
    uint32_t circuit_id,
    const double* params,
    uint32_t num_params,
    const char* observable,
    double* out
);

/* Result access - counts */
uint32_t omega_result_num_counts(const OmegaResult* result);
void omega_result_get_counts(
    const OmegaResult* result,
    uint64_t* bitstrings_out,
    uint32_t* counts_out
);
/* Bounded variant — caps at max_len, returns elements written. Recommended
   for new callers; null result/bitstrings_out/counts_out returns 0. */
uint32_t omega_result_get_counts_n(
    const OmegaResult* result,
    uint64_t* bitstrings_out,
    uint32_t* counts_out,
    uint32_t max_len
);

/* Result access - statevector */
uint32_t omega_result_statevector_len(const OmegaResult* result);
void omega_result_get_statevector(const OmegaResult* result, double* out);
/* Bounded variant — caps at max_pairs (i.e. up to 2*max_pairs doubles
   total), returns the number of complex amplitudes written. Recommended
   for new callers; null result/out returns 0. */
uint32_t omega_result_get_statevector_n(
    const OmegaResult* result,
    double* out,
    uint32_t max_pairs
);

/* Free result */
void omega_result_free(OmegaResult* result);

#ifdef __cplusplus
}
#endif

#endif /* OMEGA_H */
