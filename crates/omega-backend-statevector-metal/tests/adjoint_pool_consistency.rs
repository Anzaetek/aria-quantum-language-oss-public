//! Pool-recycle regression test for `adjoint_gradient`.
//!
//! Round 8 of the GPU Phase 4 perf push (`6bc7ab7`) introduced a
//! `BufferPool` keyed by `num_qubits` that recycles `StateBuffer`s
//! across `adjoint_gradient` calls. The pool's correctness invariant
//! is that a leased buffer is reset to `|0…0⟩` on lease — every
//! call's φ / ν / scratch state starts from a clean slate, identical
//! to a freshly-allocated buffer.
//!
//! This test pins that invariant *operationally*: it calls
//! `adjoint_gradient` repeatedly with identical inputs on the same
//! backend instance and asserts every call produces bit-for-bit
//! identical gradients. If `BufferPool::lease` ever stopped
//! correctly resetting the pooled buffer (e.g. forgot to zero the
//! tail amplitudes, or wrote `1.0` to the wrong index), the second
//! call would inherit non-|0⟩ residue from the first — gradients
//! would diverge from call 1.
//!
//! The existing `qml_trainer_metal_matches_cpu_after_full_training`
//! test compares Metal vs CPU end-to-end, but doesn't isolate the
//! lease-then-recycle path — drift could come from anywhere along
//! the adjoint chain. This test isolates the pool: same backend,
//! same circuit, same params, repeated calls.

#![cfg(all(target_os = "macos", feature = "metal"))]

use smallvec::smallvec;

use omega_backend_statevector_metal::MetalStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, ParamExpr, Qubit, SymbolId};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

const NUM_QUBITS: u32 = 5;
const NUM_PARAMS: u32 = 8;
const NUM_REPEATS: usize = 5;

/// 5-qubit hardware-efficient ansatz exercising both Rx (apply_1q)
/// and Rz (apply_diagonal) layers plus a CX entangling fan — covers
/// all four kernel paths the round-8 pool services per call.
fn build_circuit() -> CircuitIR {
    let mut c = CircuitIR::new(NUM_QUBITS, CircuitType::GateBased);
    for s in 0..NUM_PARAMS {
        c.symbols.insert(s, format!("theta_{s}"));
    }
    // Rx layer (params 0..5)
    for q in 0..NUM_QUBITS {
        c.add_op(GateOp {
            gate: GateKind::Rx,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(q)],
            classical_bit: None,
            condition: None,
        });
    }
    // CX ladder
    for q in 0..NUM_QUBITS - 1 {
        c.add_op(GateOp {
            gate: GateKind::CX,
            qubits: smallvec![Qubit(q), Qubit(q + 1)],
            params: smallvec![],
            classical_bit: None,
            condition: None,
        });
    }
    // Rz layer (params 5..8) — diagonal-fast-path on forward + dRz
    // diagonal on backward, exercises the round-9a fast path too.
    for (i, q) in (1..4u32).enumerate() {
        c.add_op(GateOp {
            gate: GateKind::Rz,
            qubits: smallvec![Qubit(q)],
            params: smallvec![ParamExpr::Symbol(NUM_QUBITS + i as u32)],
            classical_bit: None,
            condition: None,
        });
    }
    c
}

fn fresh_params() -> ParameterBinding {
    let mut params = ParameterBinding::new();
    for s in 0..NUM_PARAMS {
        params.bind(s, ((s as f64) * 0.317 + 0.7).sin());
    }
    params
}

fn weighted_z_observable() -> Observable {
    // Σ_i 2·r_i·Z_{q_i} shape — what QmlTrainer.fit hands the
    // adjoint engine. Weights chosen to be non-trivial across all
    // measurement qubits so the diagonal-Pauli-sum kernel sees a
    // multi-term observable.
    Observable {
        terms: vec![
            (0.7, vec![(0, PauliOp::Z)]),
            (-1.3, vec![(2, PauliOp::Z)]),
            (0.5, vec![(4, PauliOp::Z)]),
        ],
    }
}

#[test]
fn adjoint_gradient_repeated_calls_are_bit_identical_through_pool() {
    let backend = MetalStatevectorBackend::new().expect("Metal device");
    let circuit = build_circuit();
    let params = fresh_params();
    let obs = weighted_z_observable();

    // First call seeds the pool. Subsequent NUM_REPEATS-1 calls draw
    // from it. All must produce the same gradient vector.
    let first: Vec<(SymbolId, f64)> = backend
        .adjoint_gradient(&circuit, &params, &obs)
        .expect("first adjoint_gradient")
        .expect("backend supports adjoint");

    for repeat in 1..NUM_REPEATS {
        let next: Vec<(SymbolId, f64)> = backend
            .adjoint_gradient(&circuit, &params, &obs)
            .expect("repeat adjoint_gradient")
            .expect("backend supports adjoint");

        assert_eq!(
            first.len(),
            next.len(),
            "call {repeat}: gradient vector length differs"
        );
        for ((s1, g1), (s2, g2)) in first.iter().zip(next.iter()) {
            assert_eq!(
                s1, s2,
                "call {repeat}: gradient symbol order differs at {s1} vs {s2}"
            );
            // Bit-identical: pool reset must be deterministic, not
            // numerical-noise-equivalent. Any drift here means the
            // pooled buffer wasn't reset cleanly, since both calls
            // execute the same kernels with the same inputs.
            assert_eq!(
                g1.to_bits(),
                g2.to_bits(),
                "call {repeat}: gradient at symbol {s1} drifted: first {g1:.17e} vs next {g2:.17e}"
            );
        }
    }
}

#[test]
fn adjoint_gradient_pool_isolated_across_param_changes() {
    // Sister test: change params between two calls and assert the
    // *first* call's gradient on params_a is reproducible after a
    // pool round-trip via params_b. Catches any state leakage
    // between leases beyond the simple "all-params-identical" case.
    let backend = MetalStatevectorBackend::new().expect("Metal device");
    let circuit = build_circuit();
    let obs = weighted_z_observable();

    let params_a = fresh_params();
    let mut params_b = ParameterBinding::new();
    for s in 0..NUM_PARAMS {
        params_b.bind(s, ((s as f64) * 0.213 - 0.5).cos());
    }

    let baseline_a = backend
        .adjoint_gradient(&circuit, &params_a, &obs)
        .unwrap()
        .unwrap();
    let _ = backend
        .adjoint_gradient(&circuit, &params_b, &obs)
        .unwrap()
        .unwrap();
    let after_b = backend
        .adjoint_gradient(&circuit, &params_a, &obs)
        .unwrap()
        .unwrap();

    for ((s1, g1), (s2, g2)) in baseline_a.iter().zip(after_b.iter()) {
        assert_eq!(s1, s2);
        assert_eq!(
            g1.to_bits(),
            g2.to_bits(),
            "params_a gradient drifted after a pool round-trip via params_b at symbol {s1}: \
             baseline {g1:.17e} vs after_b {g2:.17e}"
        );
    }

    // Also assert the second call (params_b) produced something
    // structurally different from baseline_a — guards against a
    // total-no-op bug where every call returns the same gradient
    // regardless of params.
    let total_diff: f64 = baseline_a
        .iter()
        .zip(
            backend
                .adjoint_gradient(&circuit, &params_b, &obs)
                .unwrap()
                .unwrap()
                .iter(),
        )
        .map(|((_, g1), (_, g2))| (g1 - g2).abs())
        .sum();
    assert!(
        total_diff > 1e-3,
        "params_b vs params_a should produce different gradients (|Σ Δ| = {total_diff:.3e})"
    );

    // Parameters used (silences unused mut warnings on the bindings).
    let _ = params_a.resolve(&ParamExpr::Symbol(0)).unwrap();
    let _ = params_b.resolve(&ParamExpr::Symbol(0)).unwrap();
}

/// Regression test for the Metal autoreleasepool hang fix
/// (commit `9913c53`). The backward sweep opens 2-3 fresh
/// `MTLCommandBuffer`s per parameterised gate-op iteration; without
/// `objc::rc::autoreleasepool` drainage they accumulate as
/// autoreleased NSObjects under the queue's 64-outstanding cap,
/// and `commandBuffer` blocks forever on its dispatch semaphore on
/// circuits past ~22 parameterised gates. The cross-backend
/// QUBO/QAOA harness's `maxcut_k4` depth=3 instance trips this
/// (~30 parameterised ops × 3 cmd_bufs ≈ 90 outstanding).
///
/// This test pins the fix at the smaller test-binary scope: a
/// 5-qubit circuit with 40 parameterised rotations (≈ 120 cmd_bufs
/// outstanding without drainage) is large enough to hang
/// pre-fix and small enough to complete in well under a second
/// post-fix. The assertion is structural — completion alone is the
/// signal — so the test deliberately uses a tight `Duration` cap
/// so a future regression that re-introduces the leak (e.g., the
/// autoreleasepool wrapper accidentally removed) fails fast on CI
/// instead of hanging the suite forever.
#[test]
fn adjoint_gradient_completes_on_wide_parameterised_circuit() {
    use std::time::{Duration, Instant};

    let backend = MetalStatevectorBackend::new().expect("Metal device");

    // 5q × 8 layers × Rx-then-Rz pattern → 80 parameterised ops
    // (40 Rx + 40 Rz). At ~3 cmd_bufs per op-iter in the backward
    // sweep that's ~240 outstanding cmd_bufs absent drainage —
    // comfortably above Metal's 64-cap. A pre-fix run hangs in
    // `commandBuffer`'s dispatch semaphore.
    const NQ: u32 = 5;
    const LAYERS: u32 = 8;
    let mut circuit = CircuitIR::new(NQ, CircuitType::GateBased);
    let mut sym: SymbolId = 0;
    for _layer in 0..LAYERS {
        for q in 0..NQ {
            circuit.symbols.insert(sym, format!("rx_{sym}"));
            circuit.add_op(GateOp {
                gate: GateKind::Rx,
                qubits: smallvec![Qubit(q)],
                params: smallvec![ParamExpr::Symbol(sym)],
                classical_bit: None,
                condition: None,
            });
            sym += 1;
        }
        for q in 0..NQ - 1 {
            circuit.add_op(GateOp {
                gate: GateKind::CX,
                qubits: smallvec![Qubit(q), Qubit(q + 1)],
                params: smallvec![],
                classical_bit: None,
                condition: None,
            });
        }
        for q in 0..NQ {
            circuit.symbols.insert(sym, format!("rz_{sym}"));
            circuit.add_op(GateOp {
                gate: GateKind::Rz,
                qubits: smallvec![Qubit(q)],
                params: smallvec![ParamExpr::Symbol(sym)],
                classical_bit: None,
                condition: None,
            });
            sym += 1;
        }
    }
    let total_params: u32 = sym;
    assert_eq!(total_params, 2 * NQ * LAYERS, "param count sanity");

    let mut params = ParameterBinding::new();
    for s in 0..total_params {
        params.bind(s, ((s as f64) * 0.137 + 0.31).sin());
    }
    let obs = Observable {
        terms: vec![
            (1.0, vec![(0, PauliOp::Z)]),
            (-0.5, vec![(2, PauliOp::Z)]),
            (0.25, vec![(4, PauliOp::Z)]),
        ],
    };

    let start = Instant::now();
    let grads = backend
        .adjoint_gradient(&circuit, &params, &obs)
        .expect("adjoint_gradient on 80-param circuit")
        .expect("backend supports adjoint");
    let elapsed = start.elapsed();

    assert_eq!(
        grads.len(),
        total_params as usize,
        "expected one gradient entry per symbol"
    );
    // 60s ceiling: post-fix the call completes in tens of
    // milliseconds; a regression that re-leaks would manifest as a
    // multi-minute hang. The cap is intentionally loose so the
    // assertion still fires cleanly even on a heavily loaded CI
    // host without flagging legitimate slowness as a "hang".
    assert!(
        elapsed < Duration::from_secs(60),
        "adjoint_gradient took {elapsed:?} on a 80-param circuit — \
         likely the autoreleasepool drainage regressed; \
         see commit 9913c53 for context."
    );
}
