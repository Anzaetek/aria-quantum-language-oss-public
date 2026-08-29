// SPDX-License-Identifier: Apache-2.0
//! **The result must not depend on how many threads ran it — bit for bit.**
//!
//! The gate kernels are now parallel. Floating-point addition is not
//! associative, so "parallel" and "reproducible" are only compatible if every
//! output amplitude is computed by exactly one thread from a fixed set of
//! inputs. That is true here — the kernels partition the state into disjoint
//! slices and each output is a fixed 2- or 4-term expression — and this file
//! pins it, because the property is easy to lose:
//!
//! * A reduction (a norm, a probability scan) added to a kernel would break it
//!   immediately, since the summation order would follow the split.
//! * A future "optimisation" that accumulates across chunks would too.
//!
//! Exact equality is therefore the right assertion, not a tolerance. A
//! tolerance here would pass on an implementation that is quietly
//! non-deterministic, and non-determinism is what makes a numerical bug
//! impossible to bisect.
//!
//! # Why several thread counts, and these ones
//!
//! 1 proves the parallel path is taken at all (rayon with one thread still runs
//! the parallel code). 2 and 3 are the load-bearing cases: they split unevenly,
//! so a boundary error shows up. 12 matches the host and would, on its own,
//! hand out chunks in an order that can coincide with the serial one and prove
//! nothing.

use num_complex::Complex64;
use omega_backend_statevector::sim::StatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit};
use omega_core::executor::{Backend, ExecConfig, ExecResult};
use omega_core::params::ParameterBinding;

fn g(kind: GateKind, qubits: &[u32], params: &[f64]) -> GateOp {
    GateOp {
        gate: kind,
        qubits: qubits.iter().map(|&q| Qubit(q)).collect(),
        params: params.iter().map(|&p| ParamExpr::Concrete(p)).collect(),
        classical_bit: None,
        condition: None,
    }
}

/// Exercises both kernels at **every** qubit position, including `q == n-1`,
/// which is the case where the outer parallel axis collapses to a single chunk
/// and a naive implementation silently serialises.
fn circuit(n: u32) -> CircuitIR {
    let mut c = CircuitIR::new(n, CircuitType::GateBased);
    for q in 0..n {
        c.add_op(g(GateKind::H, &[q], &[]));
        c.add_op(g(GateKind::Rz, &[q], &[0.1 + q as f64 * 0.07]));
    }
    for layer in 0..3 {
        for q in (layer % 2)..n.saturating_sub(1) {
            c.add_op(g(GateKind::CX, &[q, q + 1], &[]));
        }
        // Deliberately include the top qubit as the high operand.
        if n >= 2 {
            c.add_op(g(GateKind::Ry, &[n - 1], &[0.31]));
            c.add_op(g(GateKind::CX, &[0, n - 1], &[]));
        }
        // CCX and CSwap — the two WIDEST kernels, and until 2026-08-18 the only
        // two still serial while every other gate kernel in `sim.rs` had been
        // parallelised. This circuit did not contain them, so parallelising
        // them would have landed with no invariance coverage at all.
        //
        // Non-adjacent and REVERSED triples, not just ascending ones: the
        // three-bit `scatter` reinserts operand bits in sorted order, and an
        // ascending adjacent triple cannot distinguish a correct expansion from
        // one that only works when the bits are contiguous. The same blind spot
        // was found in the Metal octet kernel earlier today.
        if n >= 4 {
            c.add_op(g(GateKind::CCX, &[0, 2, n - 1], &[]));
            c.add_op(g(GateKind::CSwap, &[n - 1, 1, 0], &[]));
            c.add_op(g(GateKind::CCX, &[n - 1, 0, 2], &[]));
        }
    }
    c
}

fn run_with_threads(threads: usize, n: u32) -> Vec<Complex64> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    pool.install(|| {
        let backend = StatevectorBackend::new();
        let config = ExecConfig {
            shots: None,
            ..Default::default()
        };
        match backend
            .execute(&circuit(n), &ParameterBinding::default(), &config)
            .expect("execute")
        {
            ExecResult::Statevector(sv) => sv,
            other => panic!("expected a statevector, got {other:?}"),
        }
    })
}

/// **Bit-for-bit identical across thread counts.**
#[test]
fn results_are_bit_identical_across_thread_counts() {
    // Wide enough to be over `PAR_MIN_DIM`, so the parallel path is genuinely
    // exercised, and small enough to stay trivial on memory (2^14 = 256 KiB).
    const N: u32 = 14;
    let reference = run_with_threads(1, N);
    // `12` was chosen on a 12-core box, so nothing here ever exercised a thread
    // count ABOVE the machine's core count. `40` on a 20-core host puts rayon
    // into oversubscription, where work-stealing changes which thread takes
    // which chunk — the one regime where a partition that secretly depends on
    // scheduling would show up. It must still be bit-identical, because the
    // chunk boundaries are functions of `dim` and the target qubit only.
    for threads in [2usize, 3, 12, 40] {
        let got = run_with_threads(threads, N);
        assert_eq!(
            got.len(),
            reference.len(),
            "{threads} threads: length changed"
        );
        for (i, (a, b)) in got.iter().zip(reference.iter()).enumerate() {
            assert!(
                a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits(),
                "{threads} threads: amplitude {i} differs bit-for-bit — {a:?} vs {b:?}. \
                 Parallelism must partition, never re-associate a sum."
            );
        }
    }
}

/// Both sides of the parallel threshold preserve the norm.
///
/// **Renamed from `the_serial_and_parallel_paths_agree`, which promised more
/// than it delivered.** The old doc said "the two must agree on a shared
/// sub-circuit's value"; the body never made that comparison. It asserts one
/// thing — norm == 1 — at one width below the threshold and one above.
///
/// That is worth keeping and worth being honest about, because a permutation
/// PRESERVES NORM EXACTLY. A `ccx` that swapped the wrong partner passes this
/// test, passes the bit-identity test above (every thread count runs the same
/// wrong code), and was in fact demonstrated to do so by mutation.
///
/// The test that compares the parallel branch against a reference is
/// `sim.rs::group_walk_equivalence::the_parallel_branch_matches_the_scan_too`,
/// which runs the verbatim pre-2026-08-15 scan at `n = 13` — above the
/// threshold — and compares `to_bits()`. Correctness lives there; this file is
/// the INVARIANCE contract, which is a different and weaker property.
#[test]
fn the_norm_survives_both_sides_of_the_threshold() {
    // n=8 is below PAR_MIN_DIM (2^12); n=14 is above.
    for n in [8u32, 14] {
        let sv = run_with_threads(4, n);
        let norm: f64 = sv.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "n={n}: norm {norm} — a partition bug shows up here first"
        );
    }
}

/// The high-qubit case in isolation. `q == n-1` gives the outer axis exactly
/// one chunk; if only that axis were parallelised this would be a silent 1x on
/// the most expensive gate in the circuit — correct, but pointlessly serial.
/// Correctness is what this asserts; the speed claim is measured separately.
#[test]
fn the_top_qubit_is_handled_correctly() {
    const N: u32 = 14;
    let mut c = CircuitIR::new(N, CircuitType::GateBased);
    c.add_op(g(GateKind::H, &[N - 1], &[]));
    let backend = StatevectorBackend::new();
    let config = ExecConfig {
        shots: None,
        ..Default::default()
    };
    let sv = match backend
        .execute(&c, &ParameterBinding::default(), &config)
        .expect("execute")
    {
        ExecResult::Statevector(sv) => sv,
        other => panic!("expected a statevector, got {other:?}"),
    };
    let half = 1usize << (N - 1);
    let amp = 1.0 / 2.0_f64.sqrt();
    assert!(
        (sv[0].re - amp).abs() < 1e-12 && (sv[half].re - amp).abs() < 1e-12,
        "H on the top qubit must put 1/sqrt(2) at |0> and |2^(n-1)>, got \
         {:?} and {:?}",
        sv[0],
        sv[half]
    );
}
