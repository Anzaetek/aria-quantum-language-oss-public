<!-- SPDX-License-Identifier: Apache-2.0 -->
# Plan: finish the CUDA concurrency work — per-thread streams

## What is already done

`147fecd` settled the soundness half. `CudaStatevectorBackend` is
`Send + Sync`, the `unsafe impl` was compiled for the first time on a GB10 and
verified rather than assumed:

* the impls are **load-bearing** (not the redundant E0119 case the handoff
  feared);
* the only blocker is `*mut CUgraph_st` / `*mut CUgraphExec_st` inside
  `TrainStepGraph`, reached through the `Mutex`-guarded cache;
* it is sound because capture AND replay both happen under that guard, so one
  `CUgraphExec` is never launched concurrently with itself.

So `aria-py` can now release the GIL around CUDA calls.

## What is NOT done, and why it matters

**Releasing the GIL buys nothing on CUDA today.** Every call goes through ONE
stream (`stream: Arc<CudaStream>`), so two host threads submit into the same
queue and the driver serialises them. Results stay correct — each launch owns
its own `StateBuffer` — but the work does not overlap.

That is the gap: the Send+Sync work removes the *language-level* obstacle to
threading and leaves the *hardware-level* one untouched. Anyone reading
"backend is now Send + Sync" and concluding "CUDA is now parallel" would be
wrong, which is why the caveat is written at the impl and in `CUDA_TODO.md`.

## MEASURED 2026-08-17 — DO NOT BUILD THIS

**Step 0 was run and the answer is no.** Two host threads sharing one backend,
same total work, best-of-3, idle box (94.5% idle), release:

```
  nq  layers      calls     1 thread    2 threads   speedup
  10       4          8      3.1ms      4.5ms     0.70x
  14       4          8      4.0ms      5.6ms     0.71x
  18       4          8      9.9ms     10.4ms     0.96x
  20       3          8     40.3ms     45.1ms     0.89x
  22       3          8    373.7ms    395.5ms     0.94x
  24       2          8       1.2s       1.2s     0.99x
```

Two threads are **never faster, at any width measured**, and the values agree
with the single-threaded reference to 1e-5 so this is not a correctness
question.

**The prediction below was wrong, and instructively so.** It expected a win at
small widths (kernels too small to fill 48 SMs) and nothing at large ones. Large
widths behave as predicted — 0.99x at 24 qubits, a clean wash, the device
already saturated. But small widths are the WORST rows, not the best: 0.70x at
10 qubits. There, per-call host overhead and contention dominate and there is no
GPU parallelism to win back, so adding a thread just adds cost.

That kills the case from both ends. Per-thread streams could only pay in a
regime where kernels underfill the device, and in exactly that regime host
overhead — not stream serialisation — is the binding constraint. Building them
would be substantial work (per-stream graph re-instantiation or a per-thread
cache that multiplies device memory, plus re-deriving the `unsafe impl`'s
safety argument) against a ceiling of roughly 1.0x.

**Decision: do not build per-thread streams.** Releasing the GIL remains correct
and worth having for the CPU backends; on CUDA it buys correctness under
sharing, not throughput. Revisit only if the workload shifts to many small
concurrent circuits AND the per-call host overhead is attacked first.

Pinned by `crates/omega-backend-statevector-cuda/tests/stream_concurrency.rs`
(`#[ignore]`d, opt-in like the other timing tests).

## (original) Measure before building — this may not be worth doing

**Step 0 is a measurement, and it can cancel the rest of the plan.**

The GB10 has 48 SMs. A statevector kernel at any interesting width already
saturates them: one 24-qubit launch is 16.7M amplitudes over 48 SMs. Two
concurrent streams then contend for the same SMs and win nothing — stream
concurrency pays when kernels are small enough to leave the device idle, which
is the *opposite* of this workload.

So measure first, with two host threads on one backend:

| circuit width | expectation |
|---|---|
| small (≤14q) | possible win — kernels too small to fill 48 SMs |
| medium (18–22q) | unclear — this is where the answer is worth having |
| large (≥24q) | expect **no** win; the device is already saturated |

Only if a real margin appears at a width anyone uses does the rest proceed.
Recording a negative result is a complete outcome here, exactly as it was for
`PAULIPROP_GPU_MIN` and the MPS `gesvdj` verdict.

## If it does proceed: what per-thread streams actually require

Not a one-line change. The stream is threaded through everything:

1. **Stream ownership.** `Arc<CudaStream>` becomes per-call or thread-local.
   Every kernel launch, every H2D/D2H copy, and every `synchronize()` must take
   the caller's stream rather than the backend's.
2. **The graph cache is the hard part.** A `CUgraphExec` is instantiated
   against a stream. Sharing one cached graph across threads with different
   streams needs either re-instantiation per stream, or a per-thread cache —
   which multiplies device memory by the thread count, and that memory is not
   currently priced by the governor.
3. **The `Mutex` argument changes.** The soundness note in `lib.rs` rests on
   "capture and replay both happen under the guard". A per-thread cache changes
   what is shared and therefore what must be proven. **Re-derive the safety
   argument; do not port it.**
4. **Buffer pools.** Any pooled device allocation reached from two streams
   needs its own synchronisation.

## Ordering

The measurement is cheap and the implementation is not, so:

1. measure two-thread throughput on one backend across widths (**do this
   first**);
2. if no margin: write the numbers into `BACKEND-CROSSOVER.md`, mark the
   caveat "measured, not worth fixing", stop;
3. if margin: per-call streams for the non-graph path only (simpler, and the
   graph path is a training-loop special case);
4. graph path last, with its safety argument re-derived from scratch.

## Gates

Correctness before speed, as everywhere else here: two threads sharing one
backend must agree with the single-threaded reference
(`tests/send_sync.rs` already asserts this), the CUDA-vs-CPU parity suite must
stay green, and any speed claim needs an idle-machine measurement with
interleaved arms — this box is shared, and three measurements this session were
invalidated by load before that discipline was adopted.

## Explicitly not in scope

Multi-GPU, CUDA graphs for the forward path, and the `aria-py` side of the GIL
release. This plan is only about whether one process can keep a single GPU busy
from two host threads.
