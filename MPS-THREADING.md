<!-- SPDX-License-Identifier: Apache-2.0 -->
# MPS threading — from "does not scale at all" to even with Aer, in one day

The record of the 2026-08-20 MPS performance programme: the external finding
that motivated it, the three fixes, the measured attribution of each, the
two-platform verification with pre-registered predictions, and the lessons
that outlived the numbers. Written so a reader can multiply the parts and
land on the totals; every figure names its machine and none of them is a
baseline (`PLAN-PERF-BASELINES.md` still owns that word).

## 0. The finding that started it

External measurement (the MIMIQ benchmark harness), `hea_31q_L16` at
`mps:256`, 1000 shots, idle Apple M5 Max, rev `b9e06b9`:

```
RAYON_NUM_THREADS=1    128.60 s
RAYON_NUM_THREADS=18   127.86 s      (0.6% apart across an 18x thread range)
```

One pair of numbers, and it justifies the programme better than any speedup
ratio below: the crate did not scale AT ALL, because nothing existed to
scale — no rayon, no BLAS, a hand-rolled serial Jacobi SVD. Meanwhile
qiskit-aer took 7.2 s (18 threads) on the same circuit and machine, a 17.4x gap on cells
where aria's answers were *exact* (discarded weight ~1e-30, certificate
honest). Correct and ignored is the worst way to lose a benchmark; the same
sweep found aria winning at shallow depth, so the gap was purely the
chi^3 regime.

## 1. The ledger — official idle-M5 measurements, all at `hea_31q_L16`, `mps:256`, 1000 shots, seed 7

The three chain rows FACTORISE: each picks up exactly where the last left
off, and their product is the total, which makes the ledger checkable rather
than merely plausible.

| chain row | what changed | pair | factor |
|---|---|---|---|
| column-major layout (0163, single-thread) | Jacobi works on contiguous columns instead of strided walks | 128.60 → 50.45 s | 2.549x |
| block fattening (0165, single-thread) | pure locality — no barrier exists at 1 thread | 50.45 → 47.19 s | 1.069x |
| threading at 8781ce3 (1 → 18 threads) | tournament rounds in parallel, barrier count cut ~16x | 47.19 → 16.71 s | 2.824x |
| **product / programme total to 8781ce3** | | 128.60 → 16.71 s | **7.696x, exact** |

**Alternative slice — NOT a chain row, do not multiply it into the ledger:**
block-Jacobi (0165) vs pair-tournament (0163) with threads held at 18 is
45.70 → 16.71 s = **2.7x**. That is the number that justifies 0165
specifically; it overlaps the threading row (same endpoints cut along a
different axis). Multiplying all four "factors" produces a fictitious 21x —
the ledger is multiplicative only along the chain.

**Sampler extension (0167) — PAIRED measurements on both platforms, and the
controls behaved.** Method on both boxes: old and new binaries interleaved
in time at each thread count, three reps, ratio per rep, clean rev stamps
required (a first M5 attempt was discarded for a `-dirty` stamp; a first
GB10 window was discarded for load contamination — both refusals were
cheaper than the argument they'd have caused).

| box | 1-thread control (must be ~1.00x) | parallel gain | detail |
|---|---|---|---|
| GB10, 10 pinned uniform X925 | 0.998x | **1.625x** at 10 thr | registered projection 1.58x, confirmed on its own hardware; medians of 3: 1.57/1.63/1.62 |
| M5 Max, 16 threads | **1.000x** (0.999/1.000/1.002) | **2.03x** at 16 thr | paired ratios 2.043/2.028/2.021; old 14.42 → new 7.11 s median |

Both discarded runs had produced PLAUSIBLE numbers — a believable null on
the GB10 (load contamination flattening the very curve under test), a
believable 2.26x on the M5 (drift plus a dirty stamp) — and neither
announced itself. That, not procedure for its own sake, is why the discards
are recorded: the discard was the cheap move, and the argument each number
would eventually have caused was the expensive one.

An earlier unpaired M5 reading of 2.26x with a 5% single-thread "gain" is
RETIRED: the GB10 side challenged it on the grounds that a 1-thread row
cannot move (nothing exists to parallelise), the paired control came back at
exactly 1.000, and the 5% was inter-session drift inflating the ratio. The
"profile rebalancing" hypothesis floated to explain the over-delivery is
withdrawn with it.

Both boxes sit on **the same mechanism with a per-machine constant**: a
fixed serial sampler cost divided by the available thread count. The
constant is NOT shared, and stating it per-machine is the stronger claim
because it survives the cross-check a shared number fails:

| box | patch-ratio inversion `S = (old−new)·p/(p−1)` | direct (shots-split subtraction) |
|---|---|---|
| GB10 | 6.76 s | 6.97 s @ 1 thr, 7.16 s @ 10 thr |
| M5 | 7.80 s | not directly measured here |

On the GB10 the constant is confirmed TWICE by independent methods (~5%
apart): subtracting a shots=10 run from a shots=1000 run, and inverting the
patch ratio. Cross-predicting with one box's constant misses the other by
~9% in both directions (7.16 s predicts M5@16 at 7.71 vs actual 7.11;
7.80 s predicts GB10@10 at 9.11 vs actual 10.05) — the sampler is
O(shots·n·χ²) of real work executed at different per-machine rates, so the
miss is expected, and now explained here rather than discovered by the next
reader. One mechanism, one per-machine constant, and nothing else needed to
explain 1.625x at 10 threads and 2.03x at 16.

**Cumulative on the M5:** official 128.60 s serial → paired-new 7.05-7.14 s
at 16 threads ≈ **18x** — level with qiskit-aer on the same machine and
circuit (aer 7.2 s, but at 18 threads against our 16: each engine at a
DIFFERENT thread count, which the comparison must say rather than imply, and
which makes "level" the honest word rather than "faster"). (The cumulative crosses build/session boundaries the paired
design does not control, so it is a programme summary, not a measurement
row; the rows above are the measurements.)

## 2. The three mechanisms, separated because the data separates them

* **0163 — layout.** The single-thread 2.549x has nothing to do with
  parallelism: one-sided Jacobi's working set became two contiguous slices
  per rotation. The same commit's pair-level parallel tournament scaled
  1→4 and then went BACKWARDS (45.70 s at 18 threads on the official sweep,
  with an optimum sitting at the fast-core count) — a barrier convoy: 511
  barriers per sweep on a 512-column matrix, ~3M rayon joins per run, each
  fencing ~1 µs of work per task.
* **0165 — block-Jacobi.** Rounds run over 16-column blocks: ~16x fewer
  barriers, each join fencing ~0.5 ms. The 18-thread doubling-back
  disappeared (official curve monotonic to 18); the slow-cluster convoy
  collapsed along with it, which retired a fast-core-default proposal on its
  proposer's own numbers. Two effects, one commit, separable in the table:
  barrier count fixed the high-thread collapse, task locality moved the
  1-thread row 7%.
* **0167 — parallel sampler.** The per-shot sampling loop was a FLAT ~7.0 s
  at every thread count (DGX Arm A, pinned uniform cores): 16% of the
  1-thread run but 41% of the 10-thread run, because everything around it
  got faster and it did not. Shots now draw from per-shot RNGs (seed, shot
  index), aggregated order-independently — a **declared count-stream
  change** (same seed, different statistically-identical map than 8781ce3),
  with the new invariance pinned by a gate:
  `sampling_counts_do_not_depend_on_thread_count` asserts bitwise-equal
  count maps across in-process pools (never trusting an env var to have
  taken effect), and a mutant deriving seeds from the worker thread index
  fails it.

## 3. Two-platform verification, pre-registered

The GB10 arms were pre-registered (predictions filed before the window
opened) and run on a deliberately quieted box, campaign SIGSTOPped, binary
identity recorded per row, `omega-run 0.1.0 (8781ce3)` content-verified by
tree hash against this box (`git am` re-authors commits, so SHA equality is
the wrong gate across machines; tree-hash equality is the right one).

* **Q1 — barrier residue after block-Jacobi: NONE.** Ten pinned uniform
  X925 cores scale 1 → 2.43x monotonically (42.81 / 20.36 / 17.62 / 17.65 s
  at 1/4/8/10). An earlier "9 uniform threads worse than 4" anomaly was the
  co-resident campaign's memory bandwidth, exactly as its owner suspected
  and refused to publish.
* **Q2 — the sampler split.** The registered speedup-ratio metric came back
  1.41x — between its own confirm/refute thresholds — but the isolated
  fixed-cost row (6.97 s at 1 thread, 7.16 s at 10) answered the decision
  the metric could not. **Recorded for reuse: a fixed serial cost measured
  against a serial baseline understates itself by construction.** The
  hypothesis test was right for the hypothesis and wrong for the decision.
* **Heterogeneity, third branch:** on the GB10 (10x Cortex-X925 + 10x
  A725, interleaved), best-pinned-uniform (17.62 s, 8 threads) beat
  best-unpinned (21.61 s, 16 threads) by 1.23x using half the threads. The
  M5's monotonic-through-18 does NOT generalise to that box. Consequence is
  **advice, not a default**: on heterogeneous aarch64, pin to the fast
  cluster (`taskset` + group `cpuinfo_max_freq`, take the top group) and
  size `RAYON_NUM_THREADS` to it. `RAYON_NUM_THREADS` stays authoritative —
  a library that second-guesses it is a trap that surfaces as someone
  else's unreproducible benchmark.

## 4. Exactness, across everything above

`discarded_weight` was a SINGLE value across all 27 DGX rows (every pin set,
thread count, and shot count) and across the official M5 sweep's 1..18
threads; `fidelity_estimate_is_a_bound` stayed false; `max_bond_reached`
256 throughout. The rev-to-rev lineage of the certificate's last bits is
declared, not hidden: 2.05e-30 (`b9e06b9`) → 2.58e-30 (`0195818`) →
2.41e-30 (`8781ce3`) — same order, different FP association, thread-count
invariant within each rev — and `f0f5822` changes the count stream (not the
certificate) per §2. Machine-readable `device_used` / `threads_used` ride
in every JSON/JSONL document (`f3ed353`) so a harness gates on data, not
stderr prose; the GPU-hook wiring gap that field closed, and the Metal
findings (1.8x pessimisation; f64 impossible on that GPU), live in
`LIMITATIONS.md`.

## 5. What remains, named

The residual serial fraction after 0167 is dominated by
`right_environments` — O(n·chi^3), once per run, serial. The DGX Amdahl
split puts the next ceiling around ~4.7x scaling; nobody has attempted it
and this doc makes no promise about it.

## 6. Lessons that outlived the numbers

* **The unarmed prologue** (named on the GB10 side): the window between a
  program starting and its safety mechanism becoming active — where the
  most likely failures (wrong path, missing file, typo'd flag) actually
  live, and where three incidents in one week all sat (`lean_axioms` dying
  under `set -e` before its own diagnostic; an EXIT trap installed after
  the validations most likely to fire; a log redirect into a directory the
  script had not yet created). The test costs one question: *"if this fails
  on its very first line, does my safety mechanism run?"* — and the pattern
  is invisible to inspection, because the code is correct and only the
  order is wrong.
* **Baselines must be named or ratios compound into fiction.** Three
  baselines were in play here (fully-serial, pair-tournament-1t,
  pair-tournament-18t); quoted carelessly they manufacture a 21x. The
  factorising ledger in §1 is the antidote: parts a reader can multiply.
* **Pre-registration made every outcome publishable.** Both DGX questions
  were answered by predictions filed before the run — including one
  registered metric that missed and says so.
* **Ask what a number looks like to someone RECOMPUTING it.** Earned the
  hard way on 2026-08-21, in both directions. Reviewing another team's
  figures with no access to their data, two errors fell out of arithmetic
  alone: a headline ratio that silently compared two different ENGINES, and
  a table whose adjacent columns came from two different HOSTS (caught
  because 31 qubits in f64 is 32 GiB and their stated machine had 36 GB —
  the reader recomputes, and the recomputation is what fails). Turning the
  same question on THIS page found the milder instance in §1: our 16-thread
  result sat beside aer's 18-thread one without saying so. None of the three
  was visible by reading; all three were visible by recomputing. Two
  corollaries worth keeping: state precision whenever a table's subject is
  memory, because the reader will do the arithmetic; and when a defect is
  caught downstream, **fix the SOURCE table** — a summary row that drops the
  qualifiers its detail sections carry is the row that gets quoted, and
  every future consumer repeats the error independently.
* **A green suite must be distinguishable from the property being false.**
  The sampler audit found no test pinning the old count stream — good news
  (nothing breaks) and a coverage gap (invariance untested), the same fact
  read twice. The invariance gate converts the declaration into something
  the next edit cannot silently break.
