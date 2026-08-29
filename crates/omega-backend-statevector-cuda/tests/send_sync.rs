// SPDX-License-Identifier: Apache-2.0
//! `CudaStatevectorBackend` is `Send + Sync`, so `aria-py` can release the GIL.
//!
//! # Why a test and not just the `unsafe impl`
//!
//! The `unsafe impl` in `lib.rs` is an ASSERTION — it makes the compiler stop
//! checking. That is the opposite of a guarantee, so the two things worth
//! pinning are pinned here:
//!
//! * the backend really is usable as `Box<dyn Backend + Send + Sync>`, which is
//!   what `Python::allow_threads` needs of the captured reference; and
//! * a `&backend` genuinely crosses a thread boundary at runtime, so the claim
//!   is exercised rather than merely declared.
//!
//! # What was verified on the CUDA host (2026-08-17, GB10)
//!
//! The impls were written blind on a machine that could not compile CUDA. On a
//! host that can:
//!
//! * they are **load-bearing** — the handoff thought they might be redundant
//!   (E0119, "conflicting implementations") and should then be deleted. They
//!   are not redundant; removing them fails to compile.
//! * the ONLY thing blocking the automatic impl is `*mut CUgraph_st` /
//!   `*mut CUgraphExec_st` inside `TrainStepGraph`, reached through the
//!   `Mutex`-guarded graph cache. Everything else qualifies on its own.
//! * the assertion is sound because every use of a cached graph — capture AND
//!   replay — happens under that guard, so one `CUgraphExec` is never launched
//!   concurrently with itself. Moving handles between threads is fine: they are
//!   scoped to a CUDA context, not a thread.
//!
//! **This test does not prove the Mutex discipline** — no test can, since the
//! `unsafe impl` silences the compiler. If someone replays a cached graph
//! outside the guard, this test still passes and the code becomes unsound. The
//! invariant is documented at the `unsafe impl`; this only guards the bound.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "cuda"))]

use omega_backend_statevector_cuda::CudaStatevectorBackend;
use omega_core::circuit::{CircuitIR, CircuitType, GateKind, GateOp, Qubit};
use omega_core::executor::{Backend, Observable, PauliOp};
use omega_core::params::ParameterBinding;

/// Compile-time: the bounds hold, and the trait-object form `aria-py` needs is
/// declarable. If the `unsafe impl`s are removed this file stops compiling.
#[test]
fn the_backend_satisfies_the_bounds_the_gil_release_needs() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaStatevectorBackend>();

    let Ok(b) = CudaStatevectorBackend::new() else {
        eprintln!("SKIP (NOT A PASS): no CUDA device");
        return;
    };
    // The exact shape `Python::allow_threads` captures.
    let _boxed: Box<dyn Backend + Send + Sync> = Box::new(b);
}

/// Runtime: a shared `&backend` actually crosses a thread boundary and both
/// threads run circuits on it.
///
/// Note what this does and does not show. All calls share ONE stream, so the
/// two threads SERIALISE on the GPU — releasing the GIL is necessary for
/// parallelism here and not sufficient. This asserts correctness under sharing,
/// not speed-up; claiming the latter would need per-thread streams, which is
/// separate work.
#[test]
fn two_threads_share_one_backend_and_agree_with_a_single_thread() {
    let Ok(backend) = CudaStatevectorBackend::new() else {
        eprintln!("SKIP (NOT A PASS): no CUDA device");
        return;
    };

    let mut c = CircuitIR::new(3, CircuitType::GateBased);
    for (g, qs) in [
        (GateKind::H, vec![Qubit(0)]),
        (GateKind::CX, vec![Qubit(0), Qubit(1)]),
        (GateKind::CX, vec![Qubit(1), Qubit(2)]),
    ] {
        c.ops.push(GateOp {
            gate: g,
            qubits: qs.into(),
            params: Default::default(),
            classical_bit: None,
            condition: None,
        });
    }
    let obs = Observable {
        terms: vec![(1.0, vec![(0, PauliOp::Z), (1, PauliOp::Z)])],
    };
    let pb = ParameterBinding::new();

    let want = backend
        .expectation(&c, &pb, &obs)
        .expect("single-threaded reference");

    let got = std::thread::scope(|s| {
        let hs: Vec<_> = (0..2)
            .map(|_| {
                let (b, c, obs, pb) = (&backend, &c, &obs, &pb);
                s.spawn(move || {
                    (0..4)
                        .map(|_| b.expectation(c, pb, obs).expect("threaded"))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter()
            .flat_map(|h| h.join().expect("thread panicked"))
            .collect::<Vec<_>>()
    });

    assert_eq!(
        got.len(),
        8,
        "both threads should have produced 4 values each"
    );
    for (i, v) in got.iter().enumerate() {
        assert!(
            (v - want).abs() < 1e-6,
            "value {i} from a shared backend ({v}) disagrees with the \
             single-threaded reference ({want}). Sharing the backend across \
             threads changed the answer."
        );
    }
}
