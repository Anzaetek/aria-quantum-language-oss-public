/* SPDX-License-Identifier: Apache-2.0
 *
 * An embedder, from OUTSIDE the Rust workspace.
 *
 * This is the acceptance test for the "embed as a native shared object" work,
 * and it is deliberately a C program rather than a Rust test. A Rust test in
 * this workspace proves the linker works; it does not exercise the CONTRACT an
 * embedder actually depends on — the header compiling under a C compiler, the
 * ownership rules, and the behaviour of every error path. There was no `.c`
 * file anywhere in the tree before this one, so none of that had ever been
 * tried from the outside.
 *
 * Build (from the repo root):
 *   cargo build -p omega-ffi --release
 *   cc -I include examples/c-embed/embed.c -o /tmp/embed \
 *      -L target/release -lomega_ffi
 *   DYLD_LIBRARY_PATH=target/release /tmp/embed     # macOS
 *   LD_LIBRARY_PATH=target/release  /tmp/embed      # Linux
 *
 * Exits 0 when every check passes, non-zero on the first failure, and prints
 * what it checked either way. No test framework: an embedder does not have one.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omega.h"

static int failures = 0;

static void check(int ok, const char *what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

/* A Bell pair with one free parameter, so binding is exercised rather than
 * skipped. `theta` is the only free symbol. */
static const char *BELL =
    "OPENQASM 2.0;\n"
    "include \"qelib1.inc\";\n"
    "qreg q[2];\n"
    "creg c[2];\n"
    "rx(theta) q[0];\n"
    "h q[0];\n"
    "cx q[0], q[1];\n";

int main(void) {
    printf("omega C embedder checks\n");

    /* --- version handshake ------------------------------------------------
     * An embedder must be able to refuse a library it does not understand
     * BEFORE calling into it, rather than crashing inside. */
    uint32_t v = omega_api_version();
    printf("  omega_api_version() = %u\n", v);
    check(v >= 1, "api version is sane");

    OmegaRuntime *rt = omega_runtime_new();
    check(rt != NULL, "omega_runtime_new returns a handle");
    if (!rt) {
        return 1;
    }

    /* --- circuit registration --------------------------------------------- */
    uint32_t cid = omega_circuit_from_source(rt, BELL, 0);
    check(cid > 0, "omega_circuit_from_source accepts QASM (0 means failure)");

    check(omega_circuit_num_qubits(rt, cid) == 2, "num_qubits == 2");
    check(omega_circuit_num_params(rt, cid) == 1, "num_params == 1 (theta)");
    check(omega_circuit_is_photonic(rt, cid) == 0, "circuit is not photonic");

    /* Malformed source must fail cleanly, not abort the process. A parser that
     * crashes on bad input cannot be pointed at anything untrusted, and a C
     * caller has no way to recover from a panic across the ABI. */
    check(omega_circuit_from_source(rt, "this is not qasm", 0) == 0,
          "malformed source returns 0 rather than crashing");
    check(omega_circuit_num_qubits(rt, 9999) == 0,
          "an unknown circuit id returns 0 rather than crashing");

    /* --- execution, exact -------------------------------------------------- */
    double params[1] = {0.0};
    OmegaResult *res = omega_execute(rt, cid, params, 1, 0, 0);
    check(res != NULL, "omega_execute (shots=0) returns a result");

    uint32_t len = omega_result_statevector_len(res);
    check(len == 4, "statevector has 4 amplitudes for 2 qubits");

    /* Bounded accessor: rx(0) is the identity, so this is a plain Bell state
     * and |00> should carry 1/sqrt(2). */
    double amps[8];
    uint32_t got = omega_result_get_statevector_n(res, amps, 4);
    check(got == 4, "bounded statevector accessor writes 4 amplitudes");
    check(fabs(amps[0] - 0.7071067811865476) < 1e-12,
          "amplitude |00> is 1/sqrt(2)");
    check(fabs(amps[6] - 0.7071067811865476) < 1e-12,
          "amplitude |11> is 1/sqrt(2)");

    /* The bound must actually bind: asking for fewer than exist must not write
     * past what was requested. */
    double small[2] = {0.0, 0.0};
    uint32_t capped = omega_result_get_statevector_n(res, small, 1);
    check(capped == 1, "a max_pairs of 1 writes exactly 1 amplitude");

    check(omega_result_get_statevector_n(NULL, amps, 4) == 0,
          "null result returns 0 from the bounded accessor");

    omega_result_free(res);

    /* --- parameter binding is CHECKED, not padded -------------------------
     * The header states "one per free symbol". The implementation used to bind
     * any missing values to 0.0 silently, so a caller passing the wrong count
     * got a DIFFERENT circuit with no error — the same silent-substitution
     * class as an exporter writing a symbolic angle as zero. */
    OmegaResult *bad = omega_execute(rt, cid, NULL, 0, 0, 0);
    check(bad == NULL, "a missing parameter array is refused, not zero-filled");
    if (bad) {
        omega_result_free(bad);
    }

    double two[2] = {0.1, 0.2};
    OmegaResult *over = omega_execute(rt, cid, two, 2, 0, 0);
    check(over == NULL, "too many parameters is refused, not truncated");
    if (over) {
        omega_result_free(over);
    }

    /* --- execution, sampled ----------------------------------------------- */
    OmegaResult *shot = omega_execute(rt, cid, params, 1, 1024, 42);
    check(shot != NULL, "omega_execute (shots=1024) returns a result");
    uint32_t n = omega_result_num_counts(shot);
    check(n == 2, "a Bell pair yields exactly 2 distinct outcomes");

    uint64_t keys[8];
    uint32_t vals[8];
    uint32_t wrote = omega_result_get_counts_n(shot, keys, vals, 8);
    check(wrote == n, "bounded counts accessor writes num_counts entries");
    uint32_t total = 0;
    for (uint32_t i = 0; i < wrote; i++) {
        total += vals[i];
        /* Only |00> and |11> may appear. */
        check(keys[i] == 0 || keys[i] == 3, "outcome is |00> or |11>");
    }
    check(total == 1024, "shot counts sum to the requested 1024");
    omega_result_free(shot);

    /* --- expectation ------------------------------------------------------
     * The entry point a variational embedder cannot work without: the whole
     * remote/QML path returns scalars and there was no way to ask for one. */
    double zz = 0.0;
    int rc = omega_expectation(rt, cid, params, 1, "Z0 Z1", &zz);
    check(rc == 0, "omega_expectation succeeds");
    check(fabs(zz - 1.0) < 1e-12, "<Z0 Z1> on a Bell pair is +1");

    double xx = 0.0;
    rc = omega_expectation(rt, cid, params, 1, "X0 X1", &xx);
    check(rc == 0 && fabs(xx - 1.0) < 1e-12, "<X0 X1> on a Bell pair is +1");

    /* Error paths return non-zero and must not touch the out-parameter, so a
     * caller that ignores the status code cannot silently read a stale value
     * as though it were an answer. */
    double sentinel = -12345.0;
    rc = omega_expectation(rt, cid, params, 1, "not an observable", &sentinel);
    check(rc != 0, "a malformed observable returns non-zero");
    check(sentinel == -12345.0, "a failed expectation leaves *out untouched");

    rc = omega_expectation(rt, cid, params, 1, "Z0", NULL);
    check(rc != 0, "a null out-parameter is refused rather than dereferenced");

    rc = omega_expectation(NULL, cid, params, 1, "Z0", &zz);
    check(rc != 0, "a null runtime is refused");

    omega_runtime_free(rt);

    printf(failures ? "\nFAILED: %d check(s)\n" : "\nAll checks passed.\n",
           failures);
    return failures ? 1 : 0;
}
