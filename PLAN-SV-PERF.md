<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — CPU statevector performance (CR B3), under a thread-count-invariance contract

**Status: S0 measured, S1 implemented (2026-08-15). S2–S5 open.** Every number
in §1.5 was measured on this box; targets elsewhere are labelled as targets.

Incoming request: `fixes/rexxxx-2/REQUEST.md` §2 — 24 qubits, 1000 shots, aria
statevector **1.36 s** vs qiskit-aer **0.20 s** (6.7×); `qft_28` **> 300 s
(timed out)** vs **22.8 s** (> 13×). The report attributes it to rayon being
applied only across parameter rows, with no gate fusion.

The report's diagnosis is half right, and the half it misses is the cheaper
half.

> **Superseded and confirmed by `fixes/xxx-2/REQUEST.md` (CR 2026-08-15),**
> which arrived after this plan was drafted and names the same three causes
> from the same line numbers, in the same order: the scan-and-reject kernels
> (its Cause 2 = S1), serial kernels with rayon only in the batch paths (Cause
> 1 = S2), and no gate fusion (Cause 3 = S5). Independent agreement on the
> diagnosis; the numbers below are ours, on our box.
>
> Three things it adds that this plan must answer:
>
> 1. **`apply_ccx` is still unfixed.** It does useful work on **one index in
>    eight** — a worse ratio than `apply_2q`'s one in four — and S1 changed only
>    `apply_2q`. Their `grover_generic_28q` (1697 two/three-qubit gates, depth
>    1971, **t/o > 900 s** against aer's 422.8 s) is the circuit that exposes
>    it. **This is a gap in S1, not a future step.**
> 2. **Their attached circuits are better fixtures than ours** — `xy_model_20q`
>    is 20 qubits, cache-resident, gate-dense, and *still* 10.9× slower than
>    aer, which isolates per-gate overhead from memory bandwidth in a way GHZ
>    and QFT cannot.
> 3. **They expect 4× from Cause 2 and sequenced it first for that reason. We
>    measured 1.30×** (§S1 below). That should be reported back: the ordering
>    argument was sound, the size estimate was not, and Cause 1 is therefore
>    worth more of the remaining budget than their ordering implies.

---

## 1. What the code actually does — read today, not inferred

| site | shape | cost |
|---|---|---|
| `sim.rs:806` `apply_1q` | strided butterfly over blocks of `2·step`, **one thread** | `dim/2` butterflies |
| `sim.rs:826` `apply_2q` | **scans all `dim` indices and `continue`s on 3 of every 4**, one thread | `dim` trips + a data-dependent branch, to do `dim/4` units of work |
| `sim.rs:879` `apply_ccx`, `apply_cswap` | same scan-and-reject shape | same 4×/8× overshoot |
| `sim.rs:911` `sample_counts` | allocates **`probs` AND `cumulative`**, each `2^n` f64 | at 28q that is **2 × 2.1 GB on top of the 4.3 GB state** |
| `sim.rs:250,265` | the crate's *only* `par_iter` — across parameter rows | untouched by this plan |

So there are **three independent costs**, and only one of them is threading:

1. **arithmetic** — `apply_2q` does four times the loop trips it needs, on
   circuits (`qft` is a `cp` ladder, `trotter_chain` is `rzz`) that are almost
   entirely two-qubit gates;
2. **threading** — one core out of twelve;
3. **memory** — the sampler triples peak footprint at exactly the widths where
   the state is already the largest object in the process.

Order matters: **a loop that is 4× too long, parallelised, is still 4× too
long.** (1) lands before (2), and (2) is measured against a (1) baseline, not
against today's.

---

## 1.5 S0 — the baseline, measured

12-core Apple silicon, 24 GB, `rustc 1.95.0`, `--release`, one thread, 1000
shots. Evolution is timed as `--expectation Z0` (no sampler allocation);
`+sampling` is the same circuit at `--shots 1000`. RSS is `/usr/bin/time -l`
maximum resident set.

| circuit | evolution | +sampling | sampler | RSS evolution | RSS sampling |
|---|---|---|---|---|---|
| ghz_20 | 0.040 s | 0.040 s | 0.000 s | 18 MiB | 34 MiB |
| ghz_22 | 0.180 s | 0.180 s | 0.000 s | 66 MiB | 130 MiB |
| ghz_24 | 0.810 s | 0.820 s | 0.010 s | 258 MiB | 514 MiB |
| ghz_26 | 3.510 s | 3.610 s | 0.100 s | 1026 MiB | 2050 MiB |
| ghz_28 | 15.640 s | 15.990 s | 0.350 s | 4098 MiB | 7607 MiB |
| qft_20 | 0.410 s | 0.410 s | 0.000 s | 18 MiB | 35 MiB |
| qft_22 | 2.000 s | 2.020 s | 0.020 s | 66 MiB | 131 MiB |
| qft_24 | 9.650 s | 9.640 s | ~0 s | 259 MiB | 515 MiB |
| qft_26 | 45.490 s | 45.990 s | 0.500 s | 1027 MiB | 2051 MiB |
| qft_28 | **213.790 s** | 215.320 s | 1.530 s | 4099 MiB | **8195 MiB** |

**Two things this measurement settled, one of them against this plan's own
first draft:**

1. **The sampler is not a time problem.** 0.35 s of 16 s on `ghz_28`, 1.5 s of
   215 s on `qft_28` — under 1%. §6 warned that the report's 1.36 s might be
   mostly `sample_counts`; it is not, and S0 existing is what settled that
   rather than an assumption either way.
2. **The sampler is exactly the memory problem.** `qft_28` RSS goes 4099 →
   8195 MiB: the sampler's two `2^n` f64 vectors are **precisely** the 4096 MiB
   added. Evolution's 4099 MiB is `2^28 × 16 B` to the megabyte, so the state is
   the only large allocation in the gate loop. S3 keeps its priority on memory
   grounds, not speed.

`ghz_24` at 0.81 s against the report's 1.36 s is the same shape on a faster
box; the gap to Aer's 0.20 s is real but smaller than filed. `qft_28`'s 213 s
is the reported "> 300 s timeout", now a number.

---

## 2. The invariance contract — this is the gate, not the speedup

The requirement is that multithreading be **enforceable at one thread**. That
means five things, and they constrain the design rather than the tests:

1. **Bit-identical across thread counts.** For a fixed circuit and seed, the
   final amplitudes must be equal *bit for bit* at `T ∈ {1, 2, 12}` — not
   within a tolerance. This is achievable only because every region parallelised
   here is a **map over disjoint index groups**: each amplitude is written by
   exactly one worker, from the same expression with the same operands, so no
   floating-point association changes. A tolerance-based test would pass under a
   genuine association bug and is therefore not acceptable here.
2. **No reduction is parallelised.** The norm (`sim.rs:794`), the cumulative
   scan, and `expectation_pauli` are reductions: parallel summation reassociates
   and changes the last bits. They stay sequential in this plan. If one is ever
   parallelised it needs a **fixed-arity reduction tree** whose shape depends on
   `dim` alone, plus its own bit-identity test.
3. **Chunk sizes are a function of `dim` and the target qubit, never of
   `rayon::current_num_threads()`.** If a boundary moved with the thread count,
   invariance would hold today by luck and break the first time a reduction
   moves inside a chunk.
4. **One thread must run the parallel code path.** No `if threads == 1 { …
   sequential … }` shortcut: that makes the single-thread test prove nothing
   about the code that actually ships. A rayon pool of size 1 executes the same
   `par_iter_mut` body.
5. **Enforceable under `RUST_TEST_THREADS=1`.** The invariance test builds its
   own `ThreadPoolBuilder` pools and runs the same circuit inside each, so it is
   independent of cargo's harness threading — which also makes it safe next to
   E7's CUDA serialisation ask.

**Safety:** no `unsafe`. Every split below is `split_at_mut` + rayon zips, which
gives disjointness by construction rather than by argument.

### 2.1 Two ways to be thread-count-invariant — and rule 4 applies to only one

QuEra's `ppvm` solves the same problem differently, and reading it sharpened
this section (`ppvm-stim/src/executor.rs:155-160`, local checkout `661fc66`):

> *"The shot index lets callers derive a deterministic per-shot seed (e.g.
> `seed + i`) so results are independent of evaluation order — the same factory
> then yields identical results from `sample_parallel`."*

So there are **two** mechanisms, and which one applies decides whether a
serial/parallel split is legitimate:

| | mechanism | invariance comes from | may it branch on thread count? |
|---|---|---|---|
| **(a)** amplitude updates — ours | a **map over disjoint groups**, no reduction | the code being *the same code*, executed in any order | **no** — rules 3 and 4 |
| **(b)** shot sampling — ppvm's | per-item **index-derived seed** | each item's result being a pure function of its index | **yes** — two paths provably agree |

`ppvm` dispatches `if n_threads <= 1 || num_shots < 4 * n_threads { serial }
else { parallel }` (`executor.rs:327-333`) — precisely the shortcut rule 4
forbids — and is right to, because it is in column (b): its two paths cannot
disagree, since shot `i` depends only on `seed + i`.

**Therefore:** rules 3 and 4 bind the amplitude path (S2). If S3 ever
parallelises sampling, it moves to column (b) and must derive per-shot
randomness from the shot index, at which point a threshold on thread count is
allowed. Mixing the two up — a thread-count-dependent split on the amplitude
path — is the failure this table exists to prevent.

### 2.2 What Quantum++ (softwareQ) does, and what not to copy

Fetched from `softwareQinc/qpp@main`,
`include/qpp/internal/kernels/qubit/apply.hpp`:

- Its 2-qubit kernel **also enumerates the `2^(n-2)` groups directly** rather
  than scanning and rejecting — independent confirmation that S1's shape is the
  standard one, arrived at here from the code rather than from qpp.
- But it rebuilds each group's base index with an **O(n) bit loop per group**
  (`for p in 0..n { if p == p_i || p == p_j { continue } … }`). Our slice walk
  gets the same index for free from the chunk structure — **O(1) per group**. We
  should not adopt qpp's indexing; ours is cheaper.
- Threading is a bare `#pragma omp parallel for` over the outer loop, **with no
  threshold and no schedule clause** — and for the 1-qubit kernel the outer loop
  is over blocks, so at target qubit `n-1` there is exactly **one** block and
  the pragma yields no parallelism at all. That is the same weakness §3.1
  identifies; qpp has it, and copying its structure would import it.
- **No diagonal or permutation special-casing anywhere.** S1b is not from qpp;
  nothing in qpp suggests it is unnecessary.

Read for approach only; nothing is copied, and the two designs differ where it
matters (indexing cost, high-qubit parallelism).

### 2.3 What qiskit-aer does — the reference we are actually measured against

From `Qiskit/qiskit-aer@main`, `src/simulators/statevector/qubitvector.hpp` and
`src/transpile/fusion.hpp`. Aer is the baseline in every row of §1's table, so
its choices are the ones worth reading closely.

**a. `CX` is a permutation, and we run it as a dense 4×4.** Aer specialises
permutation matrices to unrolled `std::swap` over index pairs, and
multi-controlled X/Y/phase to hardcoded kernels. We do the opposite:
`sim.rs:644` dispatches `GateKind::CX => apply_2q(…, &gates::cx())`, so the
most common two-qubit gate in every circuit runs **16 complex multiplies and 12
adds per group** to accomplish one swap. `apply_ccx` and `apply_cswap` already
*are* swap kernels — `apply_2q`'s callers are the outlier. On `ghz_28` (27 CX
and nothing else) this is essentially the entire cost.

**b. Diagonal gets its own kernel** — `apply_diagonal_matrix(qubits, diag)`.
That is now **three independent sources** for S1b: our own Metal/CUDA/OpenCL
backends, and Aer. It is not a speculative optimisation.

**c. The OpenMP threshold is 14 qubits**, not the 12 the CR suggests:
`omp_threads_managed()` returns >1 only when `num_qubits_ > omp_threshold_`,
default `omp_threshold_ = 14`. A shipped, tuned number to measure S2's own
crossover against rather than inventing one.

**d. Compile-time specialisation by gate arity** — `switch (N) { case 1 … case
6 }` with static-size `areg_t<N>` index arrays, so the index set for a group
never touches the allocator and the loop unrolls. We are partly there already
by having separate `apply_1q` / `apply_2q` / `apply_ccx`.

**e. No cache blocking, no qubit reordering** in `qubitvector.hpp` — worth
knowing so nobody spends a week on it expecting Aer's numbers to come from
there. Aer's wins are fusion, specialisation and OpenMP. (A separate AVX2
variant exists in `qubitvector_avx2.hpp`.)

**f. Fusion is cost-based DP, not greedy** (`fusion.hpp`): `max_qubit = 5`,
`threshold = 14` qubits, `cost_factor = 1.8`; per-gate cost 1.0 for 1–2 qubits,
1.1 for 3, 3.0 for 4, `1.8^(q-2)` beyond, and **a run of diagonal gates costs a
flat 1.0**. The DP keeps `costs[i]` and `fusion_to[i]` and picks minimum-cost
windows. If S5 ever happens, these are the parameters to start from — and note
that fusion's payoff is entangled with (b): diagonal runs are cheap to fuse
precisely because a diagonal kernel is cheap to apply.

---

## 3. The parallel decomposition, concretely

### 3.1 `apply_1q`, target qubit `q`, `step = 1 << q`

The state is a sequence of blocks of `2·step`; within a block the low half pairs
elementwise with the high half. Two axes of parallelism, and **which one is
available depends on `q`** — this is the detail that makes a naive
`par_chunks_mut` useless for high target qubits:

- **outer** — `state.par_chunks_mut(2 * step)`: `dim / (2·step)` chunks. Plenty
  for small `q`, but **exactly one chunk at `q = n-1`**, i.e. no parallelism at
  the widest gate.
- **inner** — inside a chunk, `let (lo, hi) = chunk.split_at_mut(step);` then
  `lo.par_iter_mut().zip(hi.par_iter_mut())`: `step` independent butterflies.
  Empty for `q = 0`, maximal for `q = n-1`.

Using **both** (`par_chunks_mut` whose body itself zips in parallel) covers
every `q` with one code path and no branch on thread count. Rayon nests
work-stealing without oversubscribing.

### 3.2 `apply_2q`, targets `qa > qb` after the existing swap-and-transpose

Replace the scan-and-reject with a nested split. Inside each outer chunk of
`2·step_a`:

```
let (a0, a1) = chunk.split_at_mut(step_a);      // qa = 0 | qa = 1
let (i00, i01) = a0.split_at_mut(step_b);       // qb = 0 | qb = 1  (within qa=0)
let (i10, i11) = a1.split_at_mut(step_b);       // within qa=1
```

The four slices are the four amplitudes of every group, in order, so a 4-way
`zip` walks groups with **no branch and no rejected index** — this alone is the
4× in §1(1). Writes are disjoint by construction: four non-overlapping slices.

`apply_ccx` / `apply_cswap` decompose the same way with one more split level.

### 3.3 Threshold

Below some `dim` the pool costs more than it saves. Pick the threshold **by
measurement** (§5 S2), state it as a named constant with the measured
crossover in a comment, and — because invariance rule 3 forbids thread-count
dependence — apply it to `dim` only. Below it, the same code runs on a single
chunk; no separate sequential implementation exists to drift.

### 3.4 The pool: one, built once, never per gate

The requirement is *proper MT with no runtime overhead beyond a possible
start-up cost*. That rules out three things the naive version does:

1. **No pool construction per call.** A `ThreadPoolBuilder::build()` per gate —
   or per circuit — spawns OS threads on the hot path, which is exactly the
   ~390 ms-per-call floor R2 was filed about, one layer down. The pool is a
   process-wide `static POOL: OnceLock<rayon::ThreadPool>`, built on first use
   and reused for the life of the process. Cost: one start-up, paid once.
2. **No use of rayon's implicit global pool.** It is configured by
   `RAYON_NUM_THREADS` and shared with anything else linked in — including the
   existing `par_iter` over parameter rows (`sim.rs:250,265`). An owned pool is
   what makes "this run used N threads" a statement we control and can test.
3. **No fork/join below the threshold.** Above it, entering an already-built
   pool costs a work-stealing split, not thread creation — sub-microsecond
   against a `dim ≥ 2^20` gate. Below it the same code runs on one chunk.

**Nesting.** The row-parallel `par_iter` at `sim.rs:250,265` and the new
amplitude parallelism can both be live in a batch. Rayon nests without
oversubscribing (a worker that blocks steals), so this is safe — but S2 must
**measure** a batch case, because "safe" and "not slower" are different claims
and a batch of narrow circuits is where nesting can lose.

### 3.5 Choosing the thread count, and forcing one thread

Resolution order, highest priority first:

| source | meaning |
|---|---|
| explicit API argument (`StatevectorBackend::with_threads(n)`) | for callers that manage their own pool budget — e.g. `omega-server` admission, which already prices device work |
| `ARIA_THREADS` | operator override, documented in `PREREQUISITES.md` |
| `std::thread::available_parallelism()` | default |

`ARIA_THREADS=1` (or `with_threads(1)`) means **one worker running the parallel
code path** — not a sequential branch (rule 4). `0` and unparseable values are a
loud error, not a silent fallback: a typo'd throttle that silently uses 12
threads is the kind of thing that gets discovered from a bill.

The resolved count is reported once per run (stderr, alongside the existing
remote-stats line) so a timing number can never be attributed to the wrong
thread count after the fact.

### 3.6 Tests S2 must ship with

1. **Bit-identity across pools.** Same circuit and seed at `T ∈ {1, 2, 3, 12}`;
   final amplitudes compared with `to_bits()`. `T = 2` and `3` are the load
   bearing ones — they force uneven splits; 12 on a 12-core box can hand out
   chunks in the same order as 1 and prove nothing.
2. **Both target-qubit regimes.** A low `q` (many outer chunks) and `q = n-1`
   (one outer chunk, all parallelism inner) in the same test — the case a naive
   `par_chunks_mut` silently serialises, and qpp's kernel does (§2.2).
3. **One thread runs the parallel path.** Assert `current_num_threads() == 1`
   inside the pool while the *same* `par_*` code executes, so rule 4 is checked
   rather than asserted in a comment.
4. **The pool is built once.** A construction counter incremented in the
   `OnceLock` initialiser, asserted to be 1 after many gate applications.
5. **Overhead budget at small `dim`.** A 20-qubit circuit must not regress
   against S1 by more than a stated percentage. This is the test that fails if
   someone later removes the threshold.
6. All of the above must pass under `RUST_TEST_THREADS=1` (§2 rule 5).

---

## 4. Sampling: a memory fix that is also a speed fix

`sample_counts` today allocates `probs` (2^n f64) and `cumulative` (2^n f64),
then does one `partition_point` per shot.

- **Step A** — build the cumulative scan directly from `state.iter().map(norm_sqr)`
  and drop `probs`. Halves the auxiliary allocation, changes nothing numerically
  (same additions, same order).
- **Step B** — draw all `shots` uniforms, **sort them**, and walk `state` once,
  advancing a running sum. Auxiliary allocation drops to `shots` f64 — at 28q
  and 1000 shots that is **8 KB instead of 4.3 GB**.

**Why B does not change results:** the map `r ↦ idx` is unchanged (the same
running sum, the same `<` comparisons), and counts are a **multiset** — sorting
changes only the order in which the same uniforms are consumed. Same seed ⇒ same
counts, and that equality is the test. Per-shot *order* is not observable
through `Counts`; the collapse path uses `shots = 1`, where sorting is identity.

Peak footprint at 28q, f64: **8.6 GB → 4.3 GB.** That is the difference between
running and not running on a 24 GB box with anything else open.

---

## 5. Order of work

| step | what | gate |
|---|---|---|
| **S0** | Baseline: `ghz_{20,22,24,26,28}`, `qft_{20,22,24,26,28}`, release, 1 thread, timing the **evolution and the sampler separately**. Stamp machine + toolchain. | numbers in this file |
| **S1** | `apply_2q` / `apply_ccx` / `apply_cswap` index arithmetic (§3.2), still single-threaded. | bit-identical final state vs pre-change, on a full-mantissa fixture |
| **S2** | `apply_1q` / `apply_2q` parallel decomposition (§3.1–3.3) + threshold. | the §2 invariance test at T ∈ {1,2,12}, bit-for-bit |
| **S3** | Sampler steps A then B (§4). | same seed ⇒ same counts; peak-RSS recorded |
| **S4** | Re-measure S0's table; write the "what the CPU statevector is good for" sentence CR §2 asks for, into `LIMITATIONS.md`. | the table |
| **S5** | *Only if a gap remains:* gate fusion. **Changes FP association**, so it is a separate decision, gated on the differential cross-checks, with **no tolerance widening** to make it green. | out of scope of this plan |

Targets (labelled as targets): S2 up to the core count minus memory-bandwidth
saturation — on a 12-core box a realistic 4–6× at widths where the state exceeds
L3, not 12×.

### S1 — done, and the prediction in this plan was wrong

**Measured 1.30×**, evolution time, one thread, same box and build flags as
§1.5:

| circuit | S0 | S1 | speedup |
|---|---|---|---|
| ghz_24 | 0.810 s | 0.640 s | 1.27× |
| ghz_26 | 3.510 s | 2.790 s | 1.26× |
| ghz_28 | 15.640 s | 12.040 s | 1.30× |
| qft_22 | 2.000 s | 1.580 s | 1.27× |
| qft_24 | 9.650 s | 7.540 s | 1.28× |
| qft_26 | 45.490 s | 35.300 s | 1.29× |
| qft_28 | **213.790 s** | **163.350 s** | **1.31×** |

Flat across widths and across both circuit shapes, which is what a
constant-factor loop-overhead removal should look like — if it had grown with
width, the story would have been cache behaviour rather than trip count, and the
next step would be different.

This plan's first draft said "expect ≈ 4×",
reasoning from the 4× loop-trip count. That reasoning was wrong in a way worth
recording: the rejected three-quarters were a **cheap predictable branch**,
while the surviving quarter does 16 complex multiplies. Removing 3/4 of the
cheap part is 28%, not 300%. Loop-trip counts are not work.

Bit-identity is not a claim here, it is a test: `sim.rs`'s
`mod group_walk_equivalence` keeps the scan-and-reject loop **verbatim** and
compares against it over `n ∈ 2..=7` × every ordered qubit pair, on
full-mantissa amplitudes, with `to_bits()` rather than `==` so `0.0 == -0.0`
cannot hide a slip. Both mutations tried against it (crossing the `qa`/`qb`
roles; halving one inner chunk stride) fail the test.

### S1b — the lever S0 exposed, and every GPU backend already pulled it

**The CPU is the outlier.** Reading the GPU kernels for this plan turned up that
the structure S1 just introduced, and the specialisation S1b proposes, both
already exist in this tree — on the accelerators:

| backend | 2q indexing | diagonal fast path |
|---|---|---|
| CPU (before S1) | **scan all `dim`, reject 3/4** | none |
| CPU (after S1) | `dim/4` groups, O(1) each | none — S1b |
| Metal (`shaders/apply_2q.metal`) | one thread per quad, bit-deposit | yes — fused-diagonal walker |
| CUDA (`kernels/apply_2q.cu`) | one thread per quad, bit-deposit | yes — fused-diagonal walker |
| OpenCL (`kernels/apply_2q.cl`) | one work-item per quad | yes — `apply_diagonal.cl`, `apply_diagonal_2q.cl` |

OpenCL's `execute.rs:120-123` states the reason in the same terms this plan
reached independently: *"diagonal-in-CB ones (Z, S, Sdg, T, Tdg, Rz, U1)
dispatch through the `apply_diagonal` fast path (half the per-amplitude memory
traffic + skips the 2x2 matvec on off-diagonal zeros)"*.

Two consequences:

1. **Dispatch by `GateKind`, not by a runtime matrix scan.** The GPU backends
   decide from the gate identity, which is statically known, costs nothing, and
   — unlike a "are all off-diagonal entries `0.0`?" test — cannot be fooled by a
   parameter that happens to zero an entry. S1b should copy that decision
   procedure, since it is already the house convention.
2. **CPU and the GPU backends already disagree in the signed-zero corner**, and
   have since the fast paths landed. S1b does not introduce that difference — it
   *removes* it, by making the CPU do what the other three already do. That
   reframes the §2 rule-1 caveat: the change is still not bit-identical to
   today's CPU output, but it moves the CPU **toward** the rest of the fleet
   rather than away, and the parity tests are what should say so.

### S1b covers TWO sparsity classes, not one

Aer specialises both, and so should this (§2.3a, §2.3b):

| class | gates | dense cost today | specialised cost |
|---|---|---|---|
| **permutation** | `CX`, `CY`, `CZ`, `Swap` | 16 complex mults + 12 adds per group | a swap (`CZ`: one negate) |
| **diagonal** | `CRz`, `CU3(0,0,λ)` i.e. `cp`, `CZ` | same | one multiply on one amplitude, and 3/4 of the memory untouched |

Both share **one** deviation from bit-identity and therefore one gate: a dense
row computes `1·a₀₀ + 0·a₀₁ + 0·a₁₀ + 0·a₁₁`, and `0.0 × (−x)` is `−0.0`, so
the dense path maps some **signed zeros** to the opposite sign where the
specialised path preserves them. Numerically identical; not bit-identical. One
test covering both classes should assert exactly that difference and nothing
larger — i.e. equal under `==` everywhere, and `to_bits()`-equal everywhere the
value is non-zero.

`CX` is the priority inside S1b: it is the most common two-qubit gate, it is a
pure permutation, and `ghz_28` (27 CX and nothing else, 12.0 s after S1) is
almost entirely it.

### S1b, mechanically: sparse gates run as dense 4×4

`qft`'s `cp` lowers to `GateKind::CU3`, whose matrix is `diag(1, 1, 1, e^{iλ})`.
`apply_2q` runs it as a **dense 4×4**: 16 complex multiplies, 4 loads and 4
stores per group, to do one multiply on one amplitude. A **runtime**
diagonality check — 12 comparisons per *gate application*, not per amplitude —
routes those to a path touching a quarter of the memory. On QFT-shaped circuits
that plausibly beats S1 and S2 combined, and it is why Aer wins here without
being cleverer about threads.

**It is not bit-identical, in exactly one corner:** dense `0.0 * a01 + …` maps a
`-0.0` amplitude to `+0.0`. Numerically identical, but it violates the letter of
§2 rule 1, so S1b belongs in the same bucket as fusion — its own decision, its
own gate — rather than being slipped in beside a change that *is* bit-exact.
A test asserting the signed-zero difference is the honest way to ship it.

---

## 6. What could make this pass for the wrong reason

- **Timing the sampler and calling it the gates.** A 24q GHZ is `n-1` CX on a
  state with two non-zero amplitudes; the 1.36 s in the report may be mostly
  `sample_counts`' two 2^24 allocations. S0 times the two stages separately for
  exactly this reason — otherwise S3 would "prove" S2 worked.
- **A speedup from the loop being optimised away.** Every timing fixture asserts
  on the final state in the same run.
- **Bit-identity checked on GHZ.** Its amplitudes are `0` and `1/√2`; every
  association gives the same bits, so the test cannot fail. The invariance
  fixture must be rotations by irrational angles, full mantissas, on ≥ 3 qubits,
  with **both a low and a high target qubit** — a high-`q` gate is the case the
  outer-chunk-only decomposition would silently serialise.
- **Testing invariance at T ∈ {1, 12} only on a box with 12 cores**, where a
  12-thread pool may hand every chunk to a different worker in the same order as
  1 thread. T = 2 and T = 3 are the interesting ones: they force uneven splits.
- **Measuring 12 threads after 1 thread on a thermally throttled machine.** Run
  both orders; report both.
- **Calling a 20q ratio a scaling law.** At 20q the state is 16 MB and lives in
  cache; the reported gap grows with width because the real one is bandwidth.
- **A threshold tuned until the benchmark is green.** The crossover is measured
  once and written down with its measurement, not adjusted per circuit.
- **Adding a second full-size buffer** to make the parallel version easier. At
  28q that is +4.3 GB and turns a speed fix into an OOM. In-place only.

---

## 6.5 GPU backends: what each step does and does not touch

Every step here is CPU-side, but "CPU-side" is not the same as "no GPU
consequence". Per step:

| step | Metal | OpenCL | CUDA f32 | CUDA f64 |
|---|---|---|---|---|
| **S1** group walk | none — bit-identical, and its kernel already had this shape | none | none | none |
| **S2** MT | none to kernels; **check the shared pool** does not collide with the GPU host threads | same | same, plus **E7**: the CUDA backend is neither `Send` nor `Sync` | same |
| **S3** sampler | CPU sampler only; CUDA has its own `sample_counts_on_device` (`imp.rs:444`) and OpenCL `shot_sample.cl` — **their seeds and key widths must be re-checked against the new CPU path** | as Metal | own device sampler | own device sampler |
| **S1b** diagonal | **already implemented there** — S1b closes a gap, it does not open one | already implemented | already implemented | needs checking: f64 path is newer (`f64_path.rs`) |

Concretely, what has to be verified after these land — and cannot all be
verified from this box:

1. **Metal ≡ CPU** up to f32 tolerance, on 2q-heavy circuits, both before and
   after S1b. Runnable here (`ARIA_METAL=1 ./ci.sh`).
2. **OpenCL** — its diagonal path and the CPU's must agree *including* which
   gates each considers diagonal. A `GateKind` that one specialises and the
   other does not is a silent f64-vs-f64 divergence, not a tolerance question.
   Note `c7e3348`: the OpenCL feature 413'd every statevector run, so this lane
   has recently been under-exercised.
3. **CUDA f32 and f64 on the Linux / RTX 6000 Pro box.** Neither compiles on
   macOS (`cfg`-gated), so S2's pool and S1b's dispatch are **unverified on CUDA
   until run there**. This is the same gap `LIMITATIONS.md` already records for
   CUDA's Reset criterion; S2 must not silently widen it. Specifically:
   - `RUST_TEST_THREADS=1` for the CUDA stage (**E7**) interacts directly with
     S2 — the backend is neither `Send` nor `Sync` by construction, so a shared
     rayon pool must never be handed a CUDA backend handle;
   - the f64 path (`f64_path.rs`, patches 0011–0013) is the newest arm and the
     least covered by the existing differential cross-checks.
4. **The five Reset acceptance policies** (ledger A6) are untouched by this plan
   and stay untouched — no step here should be the reason one changes.

**Rule for this plan:** a step ships CPU-first, but a step that changes *values*
(S1b) does not close until the parity runs above have been executed on the
hardware that can run them. Steps that are bit-exact (S1) and steps that only
change scheduling (S2) close on the CPU evidence plus a Metal run.

---

## 7. Deliberately not done

- **Gate fusion** (S5) — changes results in the last bits; needs its own gate.
- **The MPS 19q pathology (CR B4)** — reproduced, undiagnosed, unrelated: it is
  ~210× against our *own* statevector, so it is not a threading story.
- **GPU** — `omega-backend-statevector-cuda` already exists; this is the CPU lane.
- **The noise/trajectory path** — per-shot re-evolution is a different cost
  model (shots × evolution) and wants its own plan.
- **`expectation_pauli` / adjoint parallelism** — reductions, see §2 rule 2.
