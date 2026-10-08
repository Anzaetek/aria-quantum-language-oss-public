# Emulator comparison — competitors, not oracles

**Status: E2, E3 and E6 measured on 2026-10-07, andromeda.** Stabilizer rows are in
`tools/emu_compare/stim/rows.jsonl` (git `dff1f52`). MPS rows are in
`crates/omega-backend-mps/examples/mps_quimb_compare/e3.jsonl`. Fermionic rows
are in `tools/emu_compare/fermionic/rows.jsonl` (git `8508ac8`). Dense,
pauliprop and CV are still empty. The rules below were pinned before any
measurement existed. Rows accrete into this document per lane as they are
measured.

Every external implementation in this tree — qiskit/qiskit-aer, Stim,
PyMatching, piquasso, MQT, FQE — is wired as a **correctness oracle**: "is our
number right?" None was wired as a **competitor**: "is our emulator worth
using?" This document is the second question. It closes STATUS §5 item 17.

---

## Read this first

### These are dated measurements on named hardware. They are not regression gates.

Every row names its box, its date, and the version of every piece of software
in it. It is a measurement taken on one machine on one day, not a standing
truth, and **nothing in CI gates on these numbers**. The `benches/`
interleaving criterion benches serve regressions; this document serves
judgement.

The in-house cautionary tale is `BACKEND-CROSSOVER.md`: a whole document of
crossover wall-clocks measured on a machine the estate no longer has. Its own
preamble disowns its numbers. Nothing here inherits a number from it.

The estate is two machines:

| box | hardware | role |
|---|---|---|
| **andromeda** | Apple M4, 16 GB unified, 10 cores, Metal | the headline box for the MPS, stabilizer, pauliprop, fermionic and CV lanes; where the venvs live |
| **akilles** | linux x86_64, RTX PRO 6000 Blackwell (sm_120), CUDA 12.9, 123 GB host / 97 GB device, 32 cores | the only GPU lane, and the 32-core replicate for lanes whose conclusions depend on core topology |

There is no GB10. Any inherited plan or row naming one is stale on arrival.

### A loss gets published, at the moment it is measured.

Owner ruling, 2026-09-27. Losses are not collected for a final editing pass —
each lane appends its rows as it finishes, so the order of this document is the
order the numbers arrived in. The closing section lists every published loss in
one place so the ruling is checkable at a glance.

Where a competitor is expected to win by a large factor, that is stated before
the number: the stabilizer lane expects to lose to Stim, and the row goes in
anyway.

### A row is either a speed row or a capability row. Never a speed row with an empty cell.

This distinction is the whole subject, and this project blurred it once
already: old item 15 read "the photonic backend has never been compared on
*speed*, only on correctness", and un-blurring it took its own piece of work
(`STATUS.md:1153`).

* A **speed row** is the same computation in both arms, both timed, with a
  ratio and a classification.
* A **capability row** is one arm being unable to express the workload. It is
  reported as a capability fact with one line of evidence, **in either
  direction**, and it never appears as a blank in a speed table.

The two are different types in the schema, not one type with optional fields
(`crates/omega-emu-compare/src/row.rs`), so a capability row has nowhere to put
a ratio and a speed row cannot omit its timings.

### The tie rule (§4.6 of the plan), pinned before any number exists

A row is a **win or a loss only if both** of these hold:

1. the **min-of-N ratio is at least 1.3×**, and
2. the two arms' **[min, median] intervals are disjoint**.

**Everything else is a tie, and is printed with its ratio.** The ratio is
always published; only the *classification* has a threshold.

1.3 comes from this house's own data, not from taste. Item 15 treated a
1.17–1.21× single-core kernel gap as "only 1.2×" and then landed within
1.2–1.4× overall after fixing parallelism (`STATUS.md:1328-1364`). Differences
in that band have repeatedly turned out to be configuration, not kernel.

The threshold is written into the schema as a constant and carried in each
row's `derivation` block, so a reader can re-derive the classification from the
row alone. The failure this guards against is a tie reported as a win because
the threshold moved after the numbers existed.

### Precision is matched, or the row is marked (§4.5)

| our arm | competitor | matched? |
|---|---|---|
| CPU statevector, f64 | qulacs, f64 | yes |
| CUDA statevector, f32 | cuStateVec, f32 | yes |
| CPU statevector, f64 | qsim, **f32** | **no — annotated** |

A row whose precisions do not match **carries its annotation or does not
serialize**. Such a row counts as evidence when **we lose** and is never quoted
as a win: beating a single-precision implementation with a double-precision one
is not a result.

No Metal arm appears anywhere here. Its f32 contraction is off the production
path precisely because of how it handles certificates
(`STATUS.md:1372-1375`), and comparing a path we refuse to ship would be a fake
row.

### Overhead-dominated rows are annotated, or absent (§4.7)

Every competitor arm has a measured per-call floor — its smallest possible
invocation. Any row whose competitor time is under **10×** that floor is
annotated overhead-dominated, because it is measuring their binding against our
arithmetic rather than one engine against another.

This is why the ≤ 8-qubit fermionic fixtures are capability rows and not speed
rows, and the receipt is a row this estate deliberately does not have: the 6×6
permanent against `exqalibur`, which carries ~440 µs of fixed per-call cost
(`STATUS.md:1181-1189`).

### Every row carries its own witnesses, or it is not a row (§4.1)

A row states, per arm, the evidence that the path on its label is the path that
ran — and each witness reaches an `assert` or a non-zero exit, never a
`println`. A row that cannot produce one is published as a **void row** saying
why, not as a number.

This is not hypothetical rigour. The receipts are all in-house:
`contract_bench.rs` printed "0 of N pairs on GPU" *beside* "speedup 1.16×";
`cuda_svd_flat` fell back to the CPU on 20–45% of calls silently; and a CLI
"GPU MPS" comparison measured the CPU twice
(`PLAN-OPEN-20260825.md:2556-2575`). Assume every harness here has the same bug
until a witness proves otherwise.

Three of those witnesses are enforced by the schema rather than by review:

* **the code the row ran.** Each arm embeds its git revision at compile time;
  the runner compares it against the worktree's at run time. A mismatch, or a
  dirty tree, **voids the row**. A stale binary measured against a moved tree
  is the oldest recurring defect in this estate.
* **the quiet box.** 1-minute load average before *and* after every row, voided
  above a threshold set by the host's core count. The value lives only in
  `crates/omega-emu-compare/src/load.rs` (`load_void_above`); every lane gates
  on it and records it as the row's `void_above`. It is per host because a
  load of 6 leaves 26 of akilles' 32 cores idle and saturates andromeda's 10.
  Item 15's first four numbers were taken while another session held 28 of 32
  cores and disagreed with themselves by 29%.
* **the quiet card.** GPU contention is invisible to the load average. A GPU
  row is void if any other process holds the card or if utilisation exceeds
  10%, checked before and after, with each holder recorded as
  `{pid, mib, name}`. This was added after the GPU lane's first end-to-end
  attempt found the card at 98%, held by an ollama `llama-server` at 70.4 GB
  and a training run at 3.6 GB, while the load average reported the CPU side
  only.

### A value gate admits every speed row (§4.3)

A fast wrong number is disqualified. Before a row is timed, both arms must
agree with each other and — at every shape ≤ 26q, where a dense oracle is
affordable — with the oracle: **1e-10 relative on f64 rows, 1e-6 on f32 rows**.
Truncating rows additionally print both arms' discarded weight or fidelity
estimate next to the dense gap.

A row that fails its gate is published as a **disagreement finding**, not
silently dropped. That is how this comparison also pays rent as correctness
work.

### Reconstruction is outside the timed region, on both sides (§4.2)

Receipt: `mps_chi_profile.rs` timed `execute(shots: None)`, which ends in a
dense 2ⁿ `to_statevector`. At 14q × depth 4 the *entire evolution was 0.60 ms
inside a 4.8–9.1 ms `execute`*, and the "flat across χ" reading that deferred
Metal work for months was reconstruction flatness
(`STATUS.md:1408-1419`).

So every expectation row drives the library `expectation` path, and the same
discipline is imposed on the competitors: quimb uses its MPS-contracted
expectation and never `to_dense`; qulacs uses `Observable.get_expectation_value`
with no state copy-out; qsim uses `simulate_expectation_values`; Aer times
`run()` on a pre-transpiled circuit with transpile recorded separately.
Sampling rows time sampling and nothing else.

---

## The per-lane yardsticks

Which competitor is the yardstick differs per lane, so it is stated per lane.
"Promoted from oracle" means the wiring already exists and the marginal cost
was a timed arm plus witnesses.

| lane | our arm | yardstick | second opinion | box |
|---|---|---|---|---|
| dense statevector, CPU | `omega-backend-statevector`, f64 | **qulacs** (CPU, f64) | qsimcirq (f32, annotated) | both |
| dense statevector, GPU | `omega-backend-statevector-cuda`, f32 | **cuQuantum** (cuStateVec, f32) | — | akilles |
| MPS | `omega-backend-mps`, χ-capped | **quimb** (`CircuitMPS`) | — | andromeda headline, akilles replicate |
| stabilizer | `omega-backend-pauli` | **Stim 1.16.0**, promoted | — | andromeda |
| pauliprop | `omega-backend-pauliprop` | **monoprop 0.9.0**, promoted | — | andromeda |
| fermionic | `sector` / `majoranaprop` / `sim` | **ffsim**, promoted | FQE | andromeda |
| photonic DV | `omega-backend-photonics` | **Perceval** — measured and closed by item 15 | — | done |
| CV | `omega-backend-cv` | **piquasso**, promoted (one row) | — | andromeda |

ITensor and PauliPropagation.jl were considered and rejected: both are a Julia
toolchain lift, and quimb and monoprop cover the same lanes with wiring that
already exists. The GPU lane is cuQuantum only. No PennyLane, pytket, QuEST,
ProjectQ or qrack arms: adding arms is how a handful of defensible rows becomes
a suite nobody trusts.

**The photonic lane is not redone here.** Item 15 closed it on 2026-09-25 with
per-stage numbers and losses published (`STATUS.md:1169-1310`). This document
links that entry rather than repeating it.

---

## The pinned workload set

Fixed before anything ran, and committed:
`tools/emu_compare/qasm/`, written by `tools/emu_compare/gen_workloads.py`,
hashed in `tools/emu_compare/MANIFEST.json`. Every row names an artifact id and
its sha256.

| family | shapes | notes |
|---|---|---|
| **HEA** brickwall | 14q × {4, 12, 24}, 20q × {12, 24}, 24/26/28q × 24 | the `entangling_circuit` of `crates/omega-backend-mps/examples/mps_stage_profile.rs:82-108` — H layer, then per depth CX·Rz(0.25)·CX bricks plus an Rx(0.15) ring — **exported once to QASM2**, so every arm ingests the identical file |
| **RANDOM-1** | 20/24/26/28q × depth 20, **seed 0** | brickwall CX with per-layer random Rz/Rx; the 20q file is the single shots=1000 sampling row |
| **Clifford** brickwall | 24q and 100q × depth 100 | H/S/CX only — the same skeleton as the HEA with the two rotations replaced by one S |
| **LUCJ-16** | 16 modes (norb 8, nalpha 4, reps 1, seed 0, jastrow scale 0.1) | via the existing `crates/omega-bridges/python/lucj_gen.py`; ffsim's home turf, so a loss here is a real loss |

The HEA export being *one file* rather than a per-arm reimplementation is the
contract, and it is enforced:
`crates/omega-emu-compare/tests/workload_artifacts.rs` lowers every committed
HEA file with this repo's own parser and asserts it is op-for-op the
`entangling_circuit` it claims to be.

Two standing annotations carry over from item 16 and apply wherever these
shapes appear: **depth-4 rows are sub-millisecond and directional only, never
headlines** (`STATUS.md:1555-1557`), and **14q at χ ≥ 128 sits outside MPS's
useful regime** — the backend itself warns — and is kept in the grid only
because it was pinned.

### One exception, stated rather than papered over

**LUCJ-16 is the one artifact where "every arm ingests the identical file" does
not hold.** It spells its hopping layers `rbs`, which is not in `qelib1.inc`:
no Qiskit reader accepts it bare, and ours needs the `lenient` dialect. The
obvious repair — ship a `gate rbs(θ) a,b { … }` definition so every reader
inlines it — breaks *our* arm, because the expansion contains `h`, which does
not conserve particle number, so a decomposed `rbs` is refused by the `sector`
backend, the one engine the lane is built on.

So for this row the pin is the generator invocation, not the file's bytes, and
the fermionic lane owes it a value gate at 16 modes — affordable — showing both
arms built the same operator. §5's matching asymmetry runs the other way:
ffsim executes LUCJ layers as single fused orbital rotations while we decompose
to Givens.

---

## Rows

### Dense statevector, CPU — qulacs, with qsim as the second opinion

Measured 2026-10-08 on akilles (AMD Ryzen 9 9950X3D, 16 cores / 32 threads,
AVX-512, 123 GB), branch `emu/e4-dense`, hostgate `--watch --host-bytes 6G`
(peak resident 4.3 GB at 26q). HEA rows at git `6bb39ea`. RANDOM-1 rows
at `7e842bf`, under §4.4a. The sampling row at `076c07f`, re-measured under §4.3c. Ours is `omega_backend_statevector::StatevectorBackend`
(f64) on the library `expectation` path. qulacs 0.6.14 (f64). qsimcirq 0.22.1
on cirq-core 1.7.0 (f32, annotated). Oracle: qiskit-aer 0.17.2 on qiskit 2.5.2,
CPU, `precision="double"`, never timed. Versions are pinned in
`tools/emu_compare/dense/requirements-compare-dense.txt`.

**Contention is measured as CPU used outside the lane, not as the load
average (§4.4a).** At 24 and 26 qubits our own 16–32-thread arms lift the
1-minute load past §4.4's 8.0 by themselves: 5.95 → 13.67 on the 24q qulacs
row, 7.28 → 16.79 at 26q. The first attempt at those rows voided itself for
running. Rows from `7e842bf` carry `external_cpu`: the CPU time of every
process outside the lane's tree (the lane and its arm processes), integrated
over first warm-up to last repeat, in cores. It must stay under 8.0, and the
box must still be quiet when the row starts (load1 before ≤ 8.0). On these
rows it was 1.1–1.3 cores. The lane tests prove the census fires: twelve
busy processes outside the tree read as more than 10 cores and void the row.
The mac half of the lane (§3 says both boxes) is owed by andromeda.
**The external-CPU census is Linux-only.** It reads `/proc/stat` and
`/proc/<pid>/stat`, and its proof tests are `cfg(target_os = "linux")`. A row
recorded on macOS has no `external_cpu` and falls back to the
`load1_before.max(load1_after) > void_above` rule that §4.4a corrected. On a
10-core M4 a 20q multi-threaded row will raise load1 past 2.0 by itself and
void, as these rows did at 24q here. The mac half therefore needs a
`host_processor_info`/`getrusage`-based equivalent, or an explicit exemption,
before its rows can exist.

**The pinned observable is `O = Σᵢ ZᵢZᵢ₊₁ + Σᵢ Xᵢ` (2n−1 terms), not `Z0`.**
On every committed HEA artifact `⟨Zᵢ⟩ = 0` for every qubit to rounding (max
|Zᵢ| ≤ 1.4e-16), so a `Z0` value gate passes any arm that returns 0,
including a broken one. `O` is non-degenerate on every artifact (+14.6 to
+21.1 on HEA, +0.105 and +0.046 on RANDOM-1). It touches every qubit, so no
gate sits outside the observable's backward light cone, and `Z0` alone misses
gates far from qubit 0 at these depths. All three arms take it natively
(`Observable::parse`, qulacs `Observable`, `cirq.PauliSum`). A cell whose
oracle value is below 1e-3 is refused before any timing.

**Every arm's final state is gated, untimed, against the Aer double state:**
two-sided `|1−F| ≤ tol` and phase-aligned `max|Δψ| ≤ tol` (1e-10 for f64
arms, 1e-6 for qsim), asserted as an executed-path witness. Our weakest
witness, a plain CPU backend with nothing to dispatch to, is this state
comparison, plus `backend.name() == "statevector"` and
`rayon::current_num_threads()` asserted inside the pinned pool. The HEA
brickwall is symmetric under bit reversal, so a mirrored state is the same
state on every HEA cell. The bit-order tests therefore run on RANDOM-1.

**Knobs, in the competitor's favour (§8).** Per circuit, every arm runs one
warm-up and then the min of 3 at 16 and at 32 threads, and qsim also at
`max_fused_gate_size` 2, 3 and 4. The fastest setting is kept, and the sweep
is printed into each row's notes. On every HEA cell all three arms were
fastest at 16 threads. 32 was slower for each of them on every cell.

**qsim is timed on its kernel, not its public API.**
`simulate_expectation_values` translates the cirq circuit inside the call.
The row times `qsim_simulate_expectation_values` on a circuit pre-translated
exactly as qsimcirq 0.22.1 does it, as §4.2 times Aer on a pre-transpiled
circuit. The kernel value must equal the API value. On `hea-14q-d4` the public
API took 56.9 ms for a kernel the row times at 0.355 ms. Timing the API would
have been a 160× strawman in our favour.

**qsim's SIMD path is asserted, and a crippled build is refused.** The arm
exits unless `qsim_decide.detect_instructions()` is 0, the module the
simulator bound is `qsimcirq.qsim_avx512`, and that is the best module
`/proc/cpuinfo` supports. So `qsim_basic`, and a silent downgrade to sse or
avx2 on an AVX-512 box, are both refused.

Ratio is the competitor's min over ours. Every row below is N = 5, interleaved,
one discarded warm-up per arm. The per-call floors are the min of 5, and
nothing is overhead-dominated: the competitor's min is 59× to 6,626× its own
floor. Max/min across repeats was at most 1.26× on any arm.

| row | O (oracle) | ours min / median (s) | competitor min / median (s) | ratio | competitor faster | load1 | class |
|---|---|---|---|---|---|---|---|
| `dense-hea-20q-d12-o-expectation-vs-qulacs` | +20.950631118952 | 0.163099 / 0.165882 | 0.093563 / 0.094151 | 0.5737 | 1.74× | 4.24 → 4.24 | Loss |
| `dense-hea-20q-d24-o-expectation-vs-qulacs` | +21.119711250654 | 0.258440 / 0.263699 | 0.106487 / 0.113748 | 0.4120 | 2.43× | 6.62 → 7.37 | Loss |
| `dense-hea-14q-d12-o-expectation-vs-qulacs` | +14.641578281274 | 0.010184 / 0.010293 | 0.002024 / 0.002210 | 0.1988 | 5.03× | 3.12 → 3.12 | Loss |
| `dense-hea-14q-d24-o-expectation-vs-qulacs` | +14.771013897291 | 0.019496 / 0.019646 | 0.003568 / 0.003706 | 0.1830 | 5.46× | 3.12 → 3.12 | Loss |
| `dense-hea-14q-d4-o-expectation-vs-qulacs` | +15.102966646795 | 0.004196 / 0.004300 | 0.000938 / 0.000943 | 0.2235 | 4.47× | 3.12 → 3.12 | Loss (depth 4: directional only) |
| `dense-hea-14q-d4-o-expectation-vs-qsim` | +15.102966646795 | 0.003860 / 0.003999 | 0.000355 / 0.000377 | 0.0919 | 10.9× | 3.12 → 3.12 | Loss (f32, depth 4: directional only) |

RANDOM-1 (24q and 26q, depth 20, seed 0) and the sampling row. Ratio is the
competitor's min over ours. The 26q rows are N = 3, because each call exceeds
30 s (§4.4).

| row | O (oracle) | ours min / median (s) | competitor min / median (s) | ratio | competitor faster | load1 · external cores (window) | class |
|---|---|---|---|---|---|---|---|
| `dense-random1-24q-d20-o-expectation-vs-qulacs` | +0.046283512825 | 9.3908 / 9.4209 | 8.0680 / 8.0920 | 0.8591 | 1.16× | 5.95 → 13.67 · 1.12 (105 s) | **Tie** |
| `dense-random1-26q-d20-o-expectation-vs-qulacs` | +0.063857794133 | 53.4117 / 53.4399 | 51.1944 / 51.2240 | 0.9585 | 1.04× | 7.28 → 16.79 · 1.19 (419 s) | **Tie** |
| `dense-random1-24q-d20-o-expectation-vs-qsim` | +0.046283512825 | 9.3199 / 9.3951 | 0.7153 / 0.7340 | 0.0767 | 13.0× | 7.13 → 12.71 · 1.31 (61 s) | Loss (f32) |
| `dense-random1-26q-d20-o-expectation-vs-qsim` | +0.063857794133 | 53.3490 / 53.4699 | 5.5470 / 5.5788 | 0.1040 | 9.6× | 6.41 → 15.07 · 1.32 (236 s) | Loss (f32) |
| `dense-random1-20q-d20-sampling-1000-vs-qulacs` | 1000 shots, seeds 0–4 | 0.002526 / 0.002539 | 0.000847 / 0.000884 | 0.3355 | 2.98× | 1.33 → 1.33 · 0.00 | Loss |

**Against qulacs at 24q and 26q it is a tie.** qulacs is 1.16× and 1.04×
faster, under the 1.3× line. The two f64 engines also compute the same
thing: on RANDOM-1 our final state and qulacs's are **bit-identical** (checked
directly at 20q, `max|diff| = 0`), both 9.4e-18 from Aer. That is why the two
state-gate lines read the same digits. On HEA they differ in the last bits.
Neither arm sped up at 32 threads: both were fastest at 16 on every cell.

**qsim is 13.0× faster at 24q and 9.6× at 26q, in f32, and that is a loss.**
Its RANDOM-1 states pass the f32 gates (`|1−F|` 8.2e-8 and 1.6e-7, and O within
1.8e-8 and 7.7e-11). qsim was fastest at `max_fused_gate_size` 4 on both. Its
public API took 0.737 s and 5.54 s, and the translation inside that call took
6 ms, so at these sizes the kernel-only timing changes qsim's number by 3% or
less, not 160×. §4.5 does not let an f32 arm's row count as a win, and it does not
excuse a loss. How much of qsim's lead is f32 and how much is its kernels
(fusion, AVX-512) is not measured here.

**Sampling: qulacs is 2.98× faster at drawing 1000 shots** from an evolved
20q state (0.85 ms vs 2.53 ms). Evolution is setup on both sides (ours 0.17 s,
qulacs 0.08 s) and is not timed. Both arms' per-qubit `<Z>` estimates sit
within 3.1 standard errors of Aer's exact marginals on every qubit and every
seed (bound 5). The row's gate pins seed 0's per-qubit estimates, a 20-number
vector per arm. It was re-measured at git `076c07f` because its first version
pinned a sentence ("max |z| … = 3.106"), which §4.3c refuses. That version
read 3.25× and was also a loss. The sampling row's census window is the
sampling calls themselves, about 20 ms, so its external-cores reading is a
short sample, not a long integral. The plan's "the sampler doubles peak memory" reason for keeping
dense rows expectation-only is stale. `sample_counts` now needs only `shots`
f64s beyond the state.

**On HEA every speed row is a loss, and the 20q rows carry the weight.** qulacs is
faster than ours on all five HEA cells. The two 20q rows are the ones to
read, because they are long enough for the ratio to be about arithmetic
rather than per-call cost, and their ratios are the smallest: 1.74× and
2.43×. At 14q qulacs is 4.5–5.5× faster. The largest number in the table, a
10.9× loss to qsim, is the weakest: depth-4 rows are sub-millisecond and
directional only (§3), and qsim is f32 (§4.5). Its f32 annotation does not
soften the loss. An f32 arm's row counts as evidence when we lose, and this
is a loss.

Value gates: ours is within 1.6e-14 relative of Aer on every cell, and qulacs
within 2.0e-14. State gates: ours `|1−F|` ≤ 7.6e-14 and `max|Δψ|` ≤ 5.8e-16,
and qulacs within the same order. On `hea-14q-d4` qsim is 9.1e-8 from Aer on
O, with `|1−F|` 1.4e-7 and `max|Δψ|` 1.5e-8, inside the f32 gate.

**Four qsim cells are disagreement findings, published as VoidRows, not speed
rows.** `hea-14q-d12` and `hea-14q-d24` miss the f32 value gate (O off by
3.5e-6 and 7.7e-6 relative, against 1e-6). `hea-20q-d12` and `hea-20q-d24` miss
the f32 state gate (`|1−F|` 4.3e-6 and 1.3e-5, against 1e-6). This is f32
error growing with depth, not a wrong circuit. At a fixed 27-term observable,
qsim's absolute error on O is 1.4e-6 at depth 4, 5.1e-5 at depth 12 and
1.3e-4 at depth 24 (fuse 2). It is accumulated per gate. A term-count f32 rule
was proposed, computed on every cell and rejected, because it admitted
exactly the cells 1e-6 admits (§4.3b). qsim's speed at those shapes is
therefore not published. Our own value and state passed on all four cells.

**f64 in qsim, settled by measurement.** `QSimOptions` has no precision knob.
Against a complex128 reference, qsimcirq 0.22.1's states sit at `|1−F|`
1.7e-8 to 6.3e-7 and `max|Δψ|` 9e-9 to 8e-8, where a double path would sit
near 1e-14. Its CPU path is complex64. §4.5's annotation rule stands on that
measurement.

A defect each row would not catch: O is one scalar, so an error that leaves O
unchanged passes the value gate. That is what the state gate exists for, and
it is untimed. So a defect that appears only in the timed calls and not in
the gated state call would pass both. The HEA rows cannot see a bit-order bug
in any arm, because HEA is reversal-symmetric. That is covered by the lane
tests on RANDOM-1, not by these rows.

### Dense statevector, GPU — cuQuantum

Measured 2026-10-08 on akilles (NVIDIA RTX PRO 6000 Blackwell Max-Q, sm_120,
driver 580.178.04, CUDA 12.9), git `30c4854`, hostgate `--watch --host-bytes 6G`
(peak resident 3.1 GB, at 26q: the oracle's f64 state). Ours is
`omega-backend-statevector-cuda` (f32) on the library `Backend::expectation`
path. The CLI cannot serve, because `omega-run --device cuda --expectation` is
refused: that mode has no CUDA dispatch. The competitor is cuStateVec 1.15.0
(cuquantum-python 26.9.0) driven directly, with a loop written from NVIDIA's own
samples and every ambiguous knob set in its favour, printed in each row's
`knobs`. qiskit-aer-gpu 0.15.1 with `cuStateVec_enable=True` is the sanity
floor. The oracle is Aer CPU at f64, never timed. Versions are pinned in
`tools/emu_compare/gpu/requirements-compare-gpu-*.txt`.

**Why cuStateVec directly, and not through Aer.** The newest CUDA-12
qiskit-aer-gpu on PyPI is 0.15.1. Its own kernels are SASS and PTX for sm_70
to sm_90 only, so on this card they run through a driver JIT of sm_90 PTX: the
first call took 3.4 s at 24q, which is why its untimed warm-up is load-bearing.
cuStateVec 1.15.0 ships a native sm_120 cubin, checked with `cuobjdump` on the
downloaded wheel, not taken from release notes. Aer is the floor: if the
hand-written loop is slower than Aer at the same fusion setting, the loop is the
finding.

**The competitor is cuStateVec's fastest variant that passes the gates.** Each
cell first runs every candidate alone on the card, untimed as a row, against
the oracle's value and final state: ours; cuStateVec unfused, fused, and fused
at f64; Aer unfused, fused, and fused at f64. Fusing gates into dense
two-qubit blocks makes cuStateVec about 5× faster, but at f32 each dense block
is unitary only to about 5e-8. On HEA that compounds into a state off by
|1−F| ≈ 1.4e-5 to 1.7e-5, and Aer's own fusion misses by 2.1e-6 to 2.4e-6.
Both fail the 1e-6 gate there, so the HEA competitor is unfused cuStateVec. On
RANDOM-1 both fused paths pass, so the competitor is fused cuStateVec. The
fused miss is a statement about HEA, not about the configuration. The f64
variants are controls: they sit at 1e-14 to 1e-15 and show the drift is
precision and fusion, not anyone's implementation. They are never admitted.
Every candidate that is not timed in the row is in its `context_arms`, with its
gap and gate-stage timings.

**One GPU process at a time.** Two CUDA processes coexisting on this card slow
each other unevenly, depending on start order. At HEA 24q ours ran 97 ms alone
and 146 ms when started after a cuStateVec process; cuStateVec ran 125 ms alone
and 168 ms when started after ours. Each arm alone is steady across fresh
processes: 96–98 ms and 126–130 ms over 8 each. So each repeat starts an arm,
discards one warm-up, times one call, takes that process's device census and
quits it before the next arm starts. The arms interleave per repeat.

**Device census, the receipt.** Every row records holders and utilisation
before the arms start and after they exit, and every repeat's process must hold
at least its state on the card (`device_census_pid`). The census is not
decoration. On 10-06 our arm read 411 ms on HEA 24q while an ollama model held
the card. On a quiet card it has read 97–122 ms since. Those builds differ, so
3.4× is indicative, not a controlled measurement. The controlled one is the
same binary above: one other CUDA process alive took ours from 97 to 146 ms.
The CPU load average sees none of this.

| row | competitor | ours min / median | competitor min / median | ratio | class |
|---|---|---|---|---|---|
| `gpu-hea-24q-d24-o-expectation-vs-custatevec` | cuStateVec unfused | 97.4 / 97.6 ms | 128.9 / 129.4 ms | 1.324 | **win** |
| `gpu-random1-24q-d20-o-expectation-vs-custatevec` | cuStateVec fused | 96.2 / 96.5 ms | 20.9 / 21.1 ms | 0.217 | **loss** |
| `gpu-hea-26q-d24-o-expectation-vs-custatevec` | cuStateVec unfused | 983.7 / 984.1 ms | 1168.2 / 1168.8 ms | 1.188 | tie |
| `gpu-random1-26q-d20-o-expectation-vs-custatevec` | cuStateVec fused | 909.2 / 909.6 ms | 196.4 / 196.6 ms | 0.216 | **loss** |
| `gpu-hea-28q-d24-o-expectation-vs-custatevec` | cuStateVec unfused | 4247.6 / 4249.1 ms | 5052.5 / 5055.6 ms | 1.190 | tie |
| `gpu-random1-28q-d20-o-expectation-vs-custatevec` | cuStateVec fused | 3853.8 / 3855.4 ms | 836.8 / 837.1 ms | 0.217 | **loss** |

**Read the HEA 24q win exactly as stated.** It is 1.324 against cuStateVec
unfused, and it is a win against every cuStateVec configuration that meets the
accuracy gate. Fused cuStateVec runs the same cell in 24.8 ms, 3.9× faster than
ours, and misses the gate by 14×. HEA 26q and 28q are ties at 1.19, under the
1.3 threshold.

**On RANDOM-1 we lose by 4.6×, at every size.** There, fused cuStateVec passes
both gates and is the competitor. Those losses are, if anything, understated. At
26q and 28q the floor check reads "the loop is the finding": Aer's own fusion,
up to 5 qubits, beats my two-qubit fusion (158.8 vs 196.4 ms; 538.4 vs 836.8
ms). A cuStateVec loop with wider fusion would be faster still.

Above 26 qubits the dense oracle is out of reach (§4.3). There the value gate is
arm-vs-arm: the reference is cuStateVec at f64, which must agree with Aer-GPU
at f64 to 1e-10, and there is no state gate. The fixed cost of each arm (process
start, CUDA context, first-call excess) is reported separately from steady state
in each row's `fixed_cost_s` and notes: 0.25 s for ours, 0.37–0.39 s for
cuStateVec and 0.34–0.35 s for Aer. HEA at depth 24 and 24/26/28 qubits is a shape added for
the GPU lane, not one §3 pinned.

### MPS — quimb

JSONL: `crates/omega-backend-mps/examples/mps_quimb_compare/e3.jsonl`, measured at git `b066f9386278e8ac06b00aebe5114af5da16144e` on `andromeda.local` under `omega-hostgate run --watch --host-bytes 2G`. The truncation sweep that selected the knob, run before any of these rows were timed, is `crates/omega-backend-mps/examples/mps_quimb_compare/e3-map.json`. The knob on the capability row is quimb `cutoff=0`, `cutoff_mode=rsum2` (rank cap). It was the only sweep entry whose reported bond equalled the pinned chi on every 14q/20q depth-12 cell while the absolute dense gaps still matched.

No speed row was admitted. ⟨Z₀⟩ on this HEA is ~1e-18, so the schema's relative 1e-10 gate does not pass when the absolute gap is ~1e-15 to ~1e-16. The times below live in void rows and are not a classification. Depth 4 is directional only. 14q at χ ≥ 128 is outside MPS's useful regime. The akilles 32-core replicate was not run. The two mins are closest on `mps-hea-20q-d24-chi128`, the SVD-heavy row, which is where a 32-wide BLAS host could invert this box.

| row | load before → after | bonds ours/quimb | discarded weight | quimb fidelity / error estimate | absolute dense gap ours/quimb | min s ours/quimb |
|---|---|---|---|---|---|---|
| `mps-hea-14q-d4-chi32` | 1.64 → 1.64 | 4/16 | 1.06e-33 | 1.0 / −9.3e-15 | 1.90e-16 / 1.11e-15 | 0.000676 / 0.0113 |
| `mps-hea-14q-d4-chi128` | 1.64 → 1.64 | 4/16 | 1.06e-33 | 1.0 / −9.3e-15 | 1.90e-16 / 1.11e-15 | 0.000656 / 0.0112 |
| `mps-hea-14q-d12-chi32` | 1.64 → 1.64 | 32/32 | 7.09e-13 | 1.0 / −2.0e-14 | 3.24e-16 / 8.85e-16 | 0.0143 / 0.0594 |
| `mps-hea-14q-d12-chi128` | 1.35 → 1.35 | 64/128 | 3.41e-13 | 1.0 / −1.1e-14 | 3.25e-16 / 4.41e-16 | refused, not timed |
| `mps-hea-14q-d24-chi32` | 1.85 → 1.85 | 32/32 | 8.00e-7 | 1.0 / −5.0e-14 | 9.06e-17 / 5.62e-16 | 0.0515 / 0.137 |
| `mps-hea-14q-d24-chi128` | 1.44 → 1.44 | 128/128 | 6.78e-13 | 1.0 / −3.3e-14 | 7.38e-16 / 2.23e-15 | 0.132 / 0.304 |
| `mps-hea-20q-d12-chi32` | 1.90 → 1.91 | 32/32 | 2.31e-12 | 1.0 / −2.8e-14 | 1.96e-17 / 6.64e-16 | 0.0302 / 0.112 |
| `mps-hea-20q-d12-chi128` | 1.52 → 1.52 | 64/128 | 1.04e-12 | 1.0 / −1.8e-14 | 8.08e-18 / 3.31e-16 | refused, not timed |
| `mps-hea-20q-d24-chi32` | 1.61 → 1.61 | 32/32 | 7.27e-6 | 1.0 / −4.3e-14 | 2.37e-15 / 2.66e-15 | 0.119 / 0.268 |
| `mps-hea-20q-d24-chi128` | 1.99 → 1.99 | 128/128 | 2.78e-11 | 1.0 / −5.2e-14 | 4.04e-15 / 7.75e-16 | 1.126 / 1.865 |

The two refused rows stopped because our bond was 64 against pinned chi 128. Quimb's cutoff-0 state reported bond 128, which keeps singular values below our 1e-14 floor. A first attempt at `mps-hea-20q-d24-chi128` saw 1-minute load 2.47 and was discarded; the published attempt is the re-run at 1.99.

quimb's error estimate is slightly negative on every row because `fidelity_estimate` lands a hair above 1. That is roundoff in a norm-based estimate, not a signed error bound.

### Stabilizer — Stim

Re-measured 2026-10-08 on andromeda (Apple M4, 10 cores, 16 GB), git
`1a06841`, hostgate `--watch --host-bytes 2G`. Stim 1.16.0, Python 3.14.7,
NumPy 2.5.3. Both arms are one thread and no BLAS. These rows replace the
2026-10-07 rows at `dff1f52`. **Every classification is the same as before:**
Stim wins every speed row. What changed is what the verdicts rest on, in two
ways.

**The gate can now fail.** The first rows gated the Clifford brickwall on
`<Z0>`, which is exactly 0 on both arms and on the dense oracle, and the d=5
rows on logical Z, which is exactly 1. A gate reading `0 == 0` admits any
backend that computes nothing (§4.3c, which `check()` now refuses). Each row now
gates on three observables: `P` (the row's own, still the timed one), the
Heisenberg image `U P U†`, and its negation. For |ψ⟩ = U|0⟩,
⟨ψ|U P U†|ψ⟩ = ⟨0|P|0⟩ = +1 for any Z-type P, by construction. So the
reference is [measured, +1, −1], and no constant passes it. The image is
derived on Stim's side (`Circuit.to_tableau()` conjugation), not handed over
by the backend under test, and has weight 8 at 24q, 25 at 100q and 5 at d=5.
The timed region still holds one observable. The two extra peeks are fixed
cost.

**The quiet-box test is CPU outside the lane, not the load average
(§4.4a).** These rows carry `external_cpu`, measured as a census before the
row and an integral over it, each over at least 2 s. On andromeda the load
average does not measure CPU contention. On `stab-surface-d5-memory-expectation`
load1 read **10.16** while work outside the lane held **1.204** cores during
the row and **1.060** in the census just before it, on a 10-core box. On
`stab-clifford-24q-d100-z0-sample-10k` load1 went 12.02 → 10.16 with 1.042
external cores over a 13.5 s window. Under the old rule (void above 2.0)
neither row, nor any other on this box, could have been published.

Per-call floors, min of 5 after one discarded warm-up: expectation, pauli
5.41e-7 s and Stim 4.083e-6 s; sampling, pauli 5.0e-7 s and Stim 1.875e-6 s.
Ratio is Stim's min over ours. "Stim faster" is its reciprocal.

| row | gate [P, UPU†, −UPU†] | N | ours min / median (s) | Stim min / median (s) | ratio | Stim faster | external cores before / during · load1 | overhead |
|---|---|---|---|---|---|---|---|---|
| `stab-clifford-24q-d100-z0-expectation` | [0, 1, −1], dense oracle | 5 | 3.695e-4 / 3.792e-4 | 4.908e-5 / 4.921e-5 | 0.132821 | 7.53× | 0.95 / 1.20 · 7.96 | no |
| `stab-clifford-24q-d100-z0-sample-10k` | [0, 1, −1], dense oracle | 5 | 2.2411 / 2.2473 | 3.642e-4 / 3.689e-4 | 0.000162494 | 6,154× | 1.22 / 1.04 · 12.02 | no |
| `stab-surface-d5-memory-expectation` | [1, 1, −1] | 5 | 3.500e-5 / 3.596e-5 | 8.750e-6 / 1.008e-5 | 0.250000 | 4.00× | 1.06 / 1.20 · 10.16 | **yes** |
| `stab-surface-d5-memory-sample-10k` | [1, 1, −1] | 5 | 1.1697 / 1.1721 | 2.764e-4 / 2.853e-4 | 0.000236286 | 4,232× | 1.01 / 1.05 · 9.51 | no |
| `stab-clifford-100q-d100-z0-expectation` | [0, 1, −1] | 5 | 3.215e-3 / 3.486e-3 | 1.510e-4 / 1.559e-4 | 0.0469475 | 21.3× | 1.45 / 1.07 · 4.19 | no |
| `stab-clifford-100q-d100-z0-sample-10k` | [0, 1, −1] | 3 | 48.332 / 48.356 | 1.422e-3 / 1.423e-3 | 2.94242e-5 | 33,986× | 1.23 / 0.73 · 3.18 | no |

The d=5 expectation is a loss and it is overhead-dominated: Stim's min is
2.14× its own floor, under §4.7's 10× line. Its 4.00× is not an engine
comparison. The sampling rows are. On the 100q brickwall at 10⁴ shots,
Stim's min is 1.42 ms and ours is 48.3 s, which is 33,986×. `compile_sampler`
(`skip_reference_sample=false`, so the bits are measurement results) is fixed
cost on Stim's side, 0.001079 s on that row, outside the repeats. Our `execute`
rebuilds the tableau every shot, and that rebuild stays inside the timed
region because that is the call a user makes.

The 24q rows are gated against `omega-backend-statevector` at 1e-10
relative: oracle [0.0, 1.0000000000000002, −1.0000000000000002]. The 100q
and d=5 rows are past 26 qubits, so the gate is exact integer equality of the
two tableaus on all three observables. Sampling witnesses assert that the 10⁴
shots requested were 10⁴ shots observed, on each arm.

The d=5 circuit is one round of `SurfaceCode::new(5).syndrome_circuit()`,
lowered by `to_omega_core_ir`: 49 qubits, logical Z `Z0Z5Z10Z15Z20`, 24
ancilla measures. `tools/qec_cross_check` has no circuit file, so the row's
sha256 is of `crates/aria-qec/src/ecc/codes.rs`, and the row says so.
Expectation strips the measures on both arms. Sampling keeps them, Collapse
on our side.

**Capability, ours only.** `stabrank-clifford-t-vs-stim-refusal`. The
committed 24q depth-100 brickwall plus one `T` on qubit 0, observable X0.
stabrank's certificate: value 0, `final_chi` 2, `peak_chi` 2,
`expectation_error_bound` 0, `state_dropped_mass` 0, exact and informative.
The same brickwall without the T also has X0 = 0 under the pauli backend;
`final_chi == 2` is what says the T split happened. Stim 1.16.0 raises
`Gate not found: 'T'` on that text. External cores during the row 1.093
(load1 8.50, context only). This is not a precision comparison.

**Capability, Stim only.** `stim-d25-rotated-memory-vs-pauli-ceiling`.
`Circuit.generated("surface_code:rotated_memory_z", distance=25, rounds=25)`
with all four noise probabilities at 0.001, the docstring example, none
zeroed. 1324 qubits, 16225 measurements, 15600 detectors, 1 observable.
`compile_detector_sampler` then 10⁴ shots, `separate_observables=true`,
`bit_packed=true`, seed 0: 10⁴ detector shots and 10⁴ observable shots in
0.206597 s, compile 8.0e-5 s. The pauli backend refuses
`DEPOLARIZE1`, `DEPOLARIZE2`, `X_ERROR`, `DETECTOR`, `OBSERVABLE_INCLUDE`.
An empty 1324-qubit register has `<Z0> = 1`, so the width is in range and
the refusal is the instruction set. The noiseless 60,000-CX skeleton was
not timed. It is a different computation. External cores during the row
1.066 (load1 7.82, context only). Both capability rows were voided in an
earlier run of this re-measurement, by a load-average check still left in the
capability path. Neither row is a timing claim. The capability path now uses
the same external-CPU rule as the speed rows.

A defect each speed row would not catch: the three-observable gate sees
an error that moves `P`, its Heisenberg image or the image's sign, but not one
that preserves all three. A shared lowering bug agrees with itself. The sampling rows check the shot count, not the bits, so a
sampler that returns 10⁴ copies of the wrong outcome passes. The d=5
expectation does not see a syndrome bug that preserves logical Z, and its
sampling row does not check the syndrome bits. The Clifford+T row would not
catch a certificate whose value is 0 for a reason other than the T, beyond
the χ=2 witness it already asserts. The d=25 row would not catch detector
samples that have the right count and the wrong bits.

### Pauliprop — monoprop

Measured 2026-10-08 on akilles, git `a19baba` (branch `emu/e7-close`),
hostgate `--watch --host-bytes 8G`. Ours is
`omega_backend_pauliprop::PauliPropBackend::expectation_with_certificate`,
single-threaded by design (no rayon in the engine). The competitor is monoprop
0.9.0 (Algorithmiq, C++ core), promoted from oracle (§2). Its timed call is
`PauliPropagator(obs, [], cutoff, lower_atol)` + `propagate` +
`expectation_value`. monoprop has no thread knob and runs on its CPU affinity,
32 here, in its favour. On the exact 14q HEA it took 6.1 ms on 32 CPUs against
14.7 ms pinned to one. The value-gate oracle is `omega-backend-statevector`
(dense f64), an algorithm independent of both arms.

**The knobs were established before anything was timed.** monoprop's `cutoff`
is a Pauli-weight cap, which maps to our `max_weight`. Its `lower_atol` drops
monomials with `|coeff| < lower_atol` during evolution
(`monoprop/pauli_propagator.py:67`), which maps to our `coeff_min`. On the
mapped grid below, the two engines reach the **same error at every point**.
The weight caps agree to the last digit since our ZZ peephole
(`crates/omega-backend-pauliprop/tests/monoprop_xcheck.rs`), and the
coefficient floors agree within a few ulps. So the HEA has both an exact row
and a matched-accuracy row. The truncated fixture cases are not timed: at
equal knobs on those 4-qubit circuits the engines do not do the same work
(the test documents where), and the HEA row is where truncation is compared.

**monoprop takes Pauli rotations only, so the HEA is transcribed and the
transcription is proven.** The leading H layer acts on |0⟩, where H =
Ry(π/2). Each `cx a,b; rz(θ) b; cx a,b` is exp(−iθ/2 ZₐZ_b). `rx` is native.
Any other op, and an H after the first layer, is refused (the lane tests show
each refusal). Exact monoprop on the transcription matches the dense oracle to
1.3e-14. Observable `O = Σ ZᵢZᵢ₊₁ + Σ Xᵢ`, as in E4.

| row | O (oracle) | ours min / median (s) | monoprop min / median (s) | ratio | class | note |
|---|---|---|---|---|---|---|
| `pp-hea-14q-d12-exact-vs-monoprop` | +14.641578281274 | 0.002707 / 0.002713 | 0.004938 / 0.004988 | 1.8242 | **Win** | both exact; monoprop's min is 19.1× its floor |
| `pp-hea-14q-d12-o-matched-1e-6-vs-monoprop` | +14.641578281274 | 0.002169 / 0.002174 | 0.002807 / 0.002834 | 1.2942 | **Tie** | both at weight cap 5; error 3.0e-7 relative on both |

**The exact HEA row is a win, 1.82×.** Both arms are exact (ours 1.3e-14, monoprop
1.3e-14 from the oracle), and monoprop's min is 19× its own per-call floor, so
this is arithmetic, not binding. Ours ends with 378 Pauli terms. monoprop's
`size()` reports 147,349, but that counts its own graph representation, which
is not comparable.

**The matched-accuracy row is a tie, and it sits on the threshold.** Each arm
took its fastest setting on one grid (weight cap n/10/8/6/5/4 × coefficient
floor 0/1e-10/1e-8/1e-6) within |rel err| ≤ 1e-6 of the oracle. That tolerance
is the existing f32 gate constant, reused so it could not be chosen to suit a
number. Both chose a weight cap of 5, and both land at 3.0e-7. At cap 4 both
miss (7.8e-5). The ratio was 1.29 here. An identical run an hour earlier,
discarded unpublished only because its row ids repeated the fixture's `_exact`
suffix, measured 1.38, a win. A row that flips across the 1.3× line between
two runs is a tie, and it is published as one. monoprop's 2.8 ms includes its
0.26 ms per-call floor (min/floor 10.6, just above §4.7's line).

**Certificate, not estimate.** On the matched row our `dropped_mass` is 0.41.
That is a valid bound on |Δ⟨O⟩| and far looser than the 4.4e-6 absolute error
actually incurred. monoprop reports no error quantity at all (see the
capability matrix).

The six exact fixture rows (1–4 qubits) are all **overhead-dominated** and are
not wins, whatever their classification reads. monoprop's min is 0.95× to
2.2× its own 0.26–0.31 ms per-call floor, so they time its Python-side
construction, not propagation. Ours is 0.4 µs to 0.3 ms. Every value matches
the oracle to ≤ 6.7e-16 on both arms.

| row | ratio | class (schema) | monoprop min / its floor |
|---|---|---|---|
| `pp-fixture-rx-z0-exact-vs-monoprop` | 766.6 | Win, overhead-dominated | 1.04× |
| `pp-fixture-ry-rxx-z0-exact-vs-monoprop` | 265.3 | Win, overhead-dominated | 1.05× |
| `pp-fixture-rz-rzz-x1-exact-vs-monoprop` | 274.3 | Win, overhead-dominated | 0.95× |
| `pp-fixture-hea3-l2-zz01-exact-vs-monoprop` | 8.19 | Win, overhead-dominated | 1.86× |
| `pp-fixture-hea4-l2-x2-exact-vs-monoprop` | 3.56 | Win, overhead-dominated | 2.16× |
| `pp-fixture-hea4-l2-mixed-exact-vs-monoprop` | 1.93 | Win, overhead-dominated | 2.17× |

A defect these rows would not catch: the matched row compares equal accuracy
on one circuit and one observable, so a setting that is accurate here and not
elsewhere is not tested. And the transcription is proven only on this HEA
artifact.

### Fermionic — ffsim, FQE second opinion

Measured 2026-10-07 on andromeda (Apple M4, 10 cores, 16 GB), git `8508ac8`,
hostgate `--watch --host-bytes 2G` (peak resident 221 MB). The release build
of the harness used 4G and peaked at 363 MB; that cap is a build, not a row.
ffsim 0.0.84, NumPy 2.5.3, Qiskit 2.5.2, SciPy 1.18.1, PySCF 2.14.0, Python
3.14.7. ffsim's BLAS is Accelerate, left at the default 10 threads, which is
the competitor's favor. The sector arm is one thread and no BLAS. Load1 was
1.59 before and after every row, under the void line of 2. Nothing was re-run.

Kitaev n2, Kitaev n8, Hubbard2 and H2 are capability rows. At that width a
wall-clock measures call overhead, in either direction, and the harness
refuses a speed classification on those four ids. Both arms returned the
energy. The schema's `direction` field is `ours_only` because the type has
no third variant; the refusal on these four rows is the refusal of a ratio.
ffsim did compute them. Kitaev n64 is the row where the competitor cannot.

The value gate is two-sided relative error at 1e-10 on f64: each arm against
the reference, and the arms against each other. A seeded sign flip is checked
against the unflipped ffsim value, or against the closed form where ffsim
returns no value, and it has to move the result by more than 1e-2.

| row | reference | ours | ffsim | rel. gaps ours/ffsim | sign-flip abs. delta |
|---|---|---|---|---|---|
| `ferm-kitaev-n2` | −1 | statevector −1, majoranaprop −1 exact, `dropped_mass` 0, 1 term, seed `ladder` | −1 | 4.4e-16 / 4.4e-16 | 1.00 |
| `ferm-kitaev-n8` | −7 | statevector −7, majoranaprop −7 exact, 7 terms, seed `ladder` | −7 | 1.3e-16 / 2.5e-16 | 7.00 |
| `ferm-hubbard2` | 2−2√2 | statevector −0.8284271247461905 | −0.8284271247461898 | 2.2e-16 / 4.4e-16 | 2.83 |
| `ferm-h2-ground` | E_FCI −1.137270174625328 | statevector −1.137270174625330 | −1.137270174625328 | 1.6e-15 / 3.9e-16 | 0.0813 |

Kitaev n2 and n8 go through the statevector (Jordan–Wigner, then expectation)
and through majoranaprop. ffsim's `linear_operator` refuses the pairing
operator ("does not conserve particle number"); the published ffsim number is
its own `jordan_wigner` on the alpha register with beta in `|0>`. Hubbard and
H2 are the statevector against ffsim's occupation-basis contraction in the
(1,1) sector. ffsim builds the Hubbard singlet and the H2
cos θ|HF>+sin θ|doubles> state from the operator file. It does not replay
the QASM.

`ferm-kitaev-n64` is majoranaprop only. Energy −63, `dropped_mass` 0,
`final_terms` 63, `peak_terms` 63, `observable_range` 63, seed `ladder`,
exact and informative. The statevector is not called. ffsim's
`linear_operator` at `norb=64` raises before allocating, and the response
records `allocated: false` against 2^64 amplitudes. FQE's sector restriction
does not help: the Kitaev pairing terms do not conserve particle number, so
no (n_α, n_β) block contains the state. That is why the row is a capability
result.

LUCJ-16 (`ferm-lucj16`) is the lane's one speed row. The pin is the generator
invocation `lucj_gen.py 8 4 --seed 0 --reps 1 --jastrow-scale 0.1`, not the
QASM bytes. The value gate is ⟨n₀⟩ = `1 [0^ 0]`: sector 0.9877312230298874,
a statevector of the same lowered circuit 0.9877312230298874, ffsim's fused
UCJ 0.9877312230298954, relative gap 8.0e-15. Sector dimension C(16,8) =
12870. ffsim's spin-balanced dimension C(8,4)² = 4900. Ours does not split
α from β, which favors them.

| | ours (sector) | ffsim (fused UCJ) |
|---|---|---|
| min / median (s), N=5 | 0.004281083 / 0.00429175 | 0.000518584 / 0.000526333 |
| per-call floor (s) | 1.875e-6 | 5.483e-5 |
| fixed cost (s), outside the repeats | 0.000760 | 0.000945 |
| threads / BLAS | 1 / none | 10 / Accelerate |

Ratio `competitor.min / ours.min` = 0.121134. The intervals are disjoint and
the ratio is past 1.3×, so the classification is a **loss**: ffsim is 8.26×
faster on the mins. max/min across the five repeats is 1.025 (ours) and 1.093
(ffsim), under the 5× flag. The row is also overhead-dominated: ffsim's min is
9.46× its own floor, under the 10× line. Gate-fusion asymmetry sits on top of
that: ffsim applies `UCJOpSpinBalanced` as one fused orbital rotation, and we
decompose to Givens (`rbs`) plus diagonal phases (`rz`, `cp`, `p`). The 8.26×
is that asymmetry together with the thread asymmetry (10 vs 1) and the sector
asymmetry (12870 vs 4900). It is not a fused-kernel-only number, and the
overhead annotation says the ffsim side is still close to a single call.

What each row would not catch. Kitaev n2 and n8: `|->`^N is also a ground
state at the same energy, so the other product state passes, as does a term
with zero expectation on `|+>`. Hubbard: the energy is stationary in the `rbs`
angle, so a small angle error inside 1e-10 is invisible, and ffsim never
executes the H/CX ladder. H2: the energy is stationary at θ, adding 2π to the
`rz` leaves it unchanged, and ffsim builds the occupation state rather than
the Pauli rotation. LUCJ: the gate is one occupation, ⟨n₀⟩. An error that
preserves it passes. Negating every `rbs` is one of those — it leaves ⟨n₀⟩
unchanged at 1e-14 — so the pin is one later `rbs(-0.9916747497747819) q[1],q[0]`,
whose flip moves ⟨n₀⟩ by 0.591 against the unflipped ffsim value. The file's
first `rbs` on that pair is the identity on two filled modes. Kitaev n64: the
certificate does not distinguish `|+>` from `|->`, and a different operator with
the same `<+|H|+>` and 63 monomials would pass. It does not check competitor
bits, because there are none.

FQE did not run. `make fqe-venv` failed immediately: `make: uv: No such file
or directory` (this box has neither `uv` nor CPython 3.11). Installing `uv`
and CPython 3.11, then `make -C crates/omega-bridges/python fqe-venv`, is what
it would take: FQE 0.3.0 is a Cython sdist the Makefile pins to 3.11 and
`numpy<2`, installed `--no-deps`. Hubbard, H2 and LUCJ conserve particle
number, so that venv could second-opinion their values. Kitaev pairing changes
particle number, so a sector wavefunction would not contain the state even
after the install. No row claims FQE agreed.

### CV — piquasso

`cv-tms-phased-on-photons-mean-n`, measured 2026-10-08 on akilles at git
`a6de04b`. One row, from the committed fixture
(`tools/cv_cross_check/piquasso_multimode_fixture.jsonl`, case
`tms_phased_on_photons`): two-mode squeezing r = 0.05 then a beamsplitter, on
|1,1⟩, cutoff 14, observable ⟨n₀⟩, ⟨n₁⟩. `omega-backend-cv` against piquasso
8.0.1 `PureFockSimulator`.

**A correctness row, not a speed row.** ⟨n₀⟩ = ⟨n₁⟩ = 1.007506252083676 on our
side. piquasso live and the fixture agree to 5.7e-14. The row refuses a speed
classification, and says why in its own numbers:
- piquasso's hot path is NumPy plus numba, so a ratio would measure our Rust
  against NumPy (§2).
- Its first call is a 4.8 s numba compile.
- Two modes at cutoff 14 says nothing about CV at scale.

It is **not** overhead-dominated by §4.7: piquasso's min of 66.5 ms is 482× its
floor. So the refusal does not rest on that rule. The ratio of mins, 39.2, is
in the row for completeness and is not a result.

Found on the way: piquasso 8.0.1's numba on-disk cache does not reload on
akilles (numba 0.68.0). The first process writes it, and every later process
aborts with `LLVM ERROR: Symbol not found: .numba.unresolved$…hafnian…`. The
lane gives each run a fresh `NUMBA_CACHE_DIR`, so piquasso compiles inside its
untimed warm-up. The row says so.

---

## Capability matrix

Reported in both directions, each with one line of evidence, and each claim
verified against what the competitor actually reports rather than asserted.

### They cannot, we can

*E3, `mps-capability-discarded-weight`.* Our MPS attaches an accumulated
discarded-weight certificate (the per-split relative discarded singular-value
weight the refusal ceiling is written in) next to a `fidelity_estimate`. quimb
`CircuitMPS` reports `fidelity_estimate` and `error_estimate`, which its
docstring calls estimates from the state norm, and the arm's JSON has no
discarded-weight field. This is a capability fact, not a speed result. Both
fidelity numbers are estimates. `discarded_weight` is not a proven bound on
|Δ⟨O⟩|: this MPS is non-canonical. On the row's own evidence (14q, depth 12,
χ=32) the certificate is 7.09e-13, essentially exact. The certificate leaves
zero on the depth-24 χ=32 voids: 8.00e-7 at 14q and 7.27e-6 at 20q, while
quimb's error estimate stays at roundoff. ⟨Z₀⟩'s dense gap does not see that
tail.

* **Clifford+T with a derived certificate** (`stabrank-clifford-t-vs-stim-refusal`,
  2026-10-07). stabrank on the committed 24q depth-100 brickwall plus one T,
  observable X0: exact, `final_chi` 2, error bound 0. Stim 1.16.0 raises
  `Gate not found: 'T'`.

* **A certified error bound under truncation** (`pp-hea-14q-d12-o-matched-1e-6-vs-monoprop`,
  2026-10-08). On the matched-accuracy row our pauliprop returns `dropped_mass`
  0.41, an L1 bound on |Δ⟨O⟩| (`crates/omega-backend-pauliprop/src/sim.rs`,
  `PauliPropCertificate::dropped_mass`: "a bound, not an estimate"). The true
  error is 4.4e-6. monoprop truncates with the same knobs at the same accuracy
  and **reports no error quantity**. Read, not assumed: `lower_atol` "discards
  monomials with |coeff| < lower_atol during evolution"
  (`monoprop/pauli_propagator.py:67`). No public method of `PauliPropagator`
  or its `_core` returns a dropped weight, an estimate or a bound, and
  `tools/emu_compare/pauliprop/tests/lane.rs`
  (`monoprop_reports_no_error_quantity`) fails if a later version adds one.
  The claim is made per competitor only where the reporting was read:
  - quimb reports `fidelity_estimate` and `error_estimate`, which its docstring
    calls estimates (E3, above).
  - Stim is exact and Clifford-only, so it has nothing to bound.
  - qulacs and qsim truncate nothing in these rows, so the distinction does
    not apply to them, and no claim is made against them.
  - Our MPS `discarded_weight` is **not** claimed here as a bound: that MPS is
    non-canonical (E3).

Still open, pinned in the plan and not yet a row: mid-circuit Reset with a stated **acceptance policy**
(worded as *policy* — the five-policies ledger is still open and the row must
not oversell); and qudits, where the honest competitor column for quimb reads
"quimb could; unmeasured here" rather than "impossible". Clifford+T left this
list on 2026-10-07 when E2 made it a row. The 64-mode Kitaev energy left it
the same day, as `ferm-kitaev-n64`.

* **64-mode Kitaev energy** (`ferm-kitaev-n64`, 2026-10-07). majoranaprop
  returns −63 with `dropped_mass` 0, 63 monomials, seed `ladder`, exact.
  Dense competitors face 2^64 amplitudes; ffsim's `linear_operator` refuses
  before allocating, because the operator does not conserve particle number.
  FQE's sector restriction does not help for the same reason: the pairing
  terms change particle number, so no (n_α, n_β) block contains the state.

### We cannot, they can

* **Stim's d=25 detector sampling** (`stim-d25-rotated-memory-vs-pauli-ceiling`,
  2026-10-07). Stim samples a distance-25, 25-round rotated memory at the
  docstring noise (all four probabilities 0.001): 1324 qubits, 15600
  detectors, 10⁴ shots in 0.203345 s. The pauli backend has no noise channel and
  no detector sampler. An empty register of that width is in range (`<Z0> = 1`);
  the ceiling is the instruction set.

Still open, pinned in the plan and not yet a row: cuStateVec multi-GPU and
distributed statevector (out of scope — one GPU in the estate — and said so
rather than tested at n=1 and called a comparison); quimb's general
tensor-network geometries where ours is a chain. Aer's noise-model throughput
at scale is a **known gap with no measured row behind it**: building the Aer
comparison for it was declined by the owner, so nothing here was tried and
lost. Our noise surface is narrower (the ppvm bridge refuses noise), and the
doc does not claim otherwise. ffsim's orbital-rotation-native gate set left this list
on 2026-10-07 as the LUCJ loss below.

* **ffsim's fused orbital rotation** (`ferm-lucj16`, 2026-10-07). ffsim applies
  the 16-mode LUCJ as one `UCJOpSpinBalanced`. We decompose it to Givens and
  diagonal phases. On ⟨n₀⟩ ffsim's min is 8.26× ours (ratio 0.121134), and
  the row is overhead-dominated: that min is 9.46× ffsim's own per-call floor.
  The comparison also gives them 10 Accelerate threads against our one, and
  C(8,4)² = 4900 amplitudes against our C(16,8) = 12870. The value gate
  passed (relative gap 8.0e-15), so this is a loss on a shared operator, not
  a disagreement.

---

## Every published loss, in one place

E2, andromeda, re-measured 2026-10-08, git `1a06841`, under the
three-observable gate and the external-CPU criterion. Stim is faster on every
speed row, as it was on 2026-10-07. Ratio is Stim's min over ours.

| row | ratio | Stim faster | note |
|---|---|---|---|
| `stab-clifford-24q-d100-z0-expectation` | 0.132821 | 7.53× | |
| `stab-clifford-24q-d100-z0-sample-10k` | 0.000162494 | 6,154× | |
| `stab-surface-d5-memory-expectation` | 0.250000 | 4.00× | overhead-dominated: Stim's min is 2.14× its floor |
| `stab-surface-d5-memory-sample-10k` | 0.000236286 | 4,232× | |
| `stab-clifford-100q-d100-z0-expectation` | 0.0469475 | 21.3× | |
| `stab-clifford-100q-d100-z0-sample-10k` | 2.94242e-5 | 33,986× | |

E6, andromeda, 2026-10-07, git `8508ac8`. One speed row. Ratio is ffsim's min
over ours. The four ≤8-qubit rows are not in this table: they are capability
rows, and a speed classification on one is refused.

| row | ratio | ffsim faster | note |
|---|---|---|---|
| `ferm-lucj16` | 0.121134 | 8.26× | overhead-dominated (ffsim's min is 9.46× its floor); fusion, 10 threads vs 1, and C(8,4)² vs C(16,8) |

E4, akilles, 2026-10-08, git `6bb39ea`. Ratio is the competitor's min over
ours. The 20q rows carry the weight. The 14q and depth-4 rows are shorter and
the depth-4 ones are directional only (§3).

| row | ratio | competitor faster | note |
|---|---|---|---|
| `dense-hea-20q-d12-o-expectation-vs-qulacs` | 0.5737 | 1.74× | |
| `dense-hea-20q-d24-o-expectation-vs-qulacs` | 0.4120 | 2.43× | |
| `dense-hea-14q-d12-o-expectation-vs-qulacs` | 0.1988 | 5.03× | |
| `dense-hea-14q-d24-o-expectation-vs-qulacs` | 0.1830 | 5.46× | |
| `dense-hea-14q-d4-o-expectation-vs-qulacs` | 0.2235 | 4.47× | depth 4, directional only |
| `dense-hea-14q-d4-o-expectation-vs-qsim` | 0.0919 | 10.9× | f32 arm (§4.5: evidence of a loss), depth 4, directional only |
| `dense-random1-24q-d20-o-expectation-vs-qsim` | 0.0767 | 13.0× | f32 arm, git `7e842bf` |
| `dense-random1-26q-d20-o-expectation-vs-qsim` | 0.1040 | 9.6× | f32 arm, git `7e842bf` |
| `dense-random1-20q-d20-sampling-1000-vs-qulacs` | 0.3355 | 2.98× | sampling 1000 shots, git `076c07f` |


The two RANDOM-1 rows against qulacs are ties (0.8591 at 24q, 0.9585 at 26q)
and are not in this table.

E5, akilles, 2026-10-08, git `30c4854`. Ratio is cuStateVec's min over ours.
The competitor on these rows is fused cuStateVec, which passes both gates on
RANDOM-1 (it fails them on HEA).

| row | ratio | cuStateVec faster | note |
|---|---|---|---|
| `gpu-random1-24q-d20-o-expectation-vs-custatevec` | 0.2174 | 4.60× | |
| `gpu-random1-26q-d20-o-expectation-vs-custatevec` | 0.2160 | 4.63× | understated: Aer's fusion beats our cuStateVec loop (floor) |
| `gpu-random1-28q-d20-o-expectation-vs-custatevec` | 0.2171 | 4.61× | understated: Aer's fusion beats our cuStateVec loop (floor) |

E7, akilles, 2026-10-08. **No loss.** The pauliprop lane has one win
(`pp-hea-14q-d12-exact-vs-monoprop`, 1.82×), one tie on the threshold
(`pp-hea-14q-d12-o-matched-1e-6-vs-monoprop`, 1.29, which measured 1.38 in a
discarded earlier run), and six overhead-dominated fixture rows that are not
wins. The CV row refuses a speed classification.

*E3 published no speed row, so it has no win and no loss.* The ratios in the
void rows above are not classifications. Ruling 3 is enforced by the ordering
of the work rather than by intention, and this section exists so the ruling is
checkable at a glance.

## Every lane in one line

- **Dense CPU (E4, akilles).** qulacs beats us on HEA, 1.74× and 2.43× at 20q
  and 4.5–5.5× at 14q. It ties us on RANDOM-1 at 24q and 26q. qsim, in f32, is
  9.6–13× ahead where it passed its gates and fails the f32 gates on four HEA
  cells. **Incomplete:** the mac half (§3, both boxes) is owed. The §4.4a
  census it needs is Linux-only.
- **Dense GPU (E5, akilles).** Against cuStateVec driven directly, the
  fastest variant that passes the gates: on HEA, where fused cuStateVec fails
  the f32 gate, we win at 24q (1.32) and tie at 26q and 28q (1.19). On
  RANDOM-1, where fused cuStateVec passes, we lose 4.6× at 24, 26 and 28
  qubits.
- **MPS (E3, andromeda).** No speed row: ⟨Z₀⟩ on the pinned HEA is ~1e-18, so
  no relative gate could pass. One capability row: our discarded-weight
  certificate next to quimb's estimates, not claimed as a proven bound.
- **Stabilizer (E2, andromeda).** Stim wins every speed row, from 4.0× to
  33,986× on sampling. Two capability rows, one each way: Clifford+T with a
  certificate (ours), and d=25 detector sampling (Stim's). The rows were
  re-measured under a gate that can fail, `[P, UPU†, −UPU†]`, after the first
  version gated on ⟨Z₀⟩ = 0 exactly. No classification moved.
- **Pauliprop (E7, akilles).** Exact 14q HEA: we are 1.82× faster than
  monoprop. Matched accuracy: a tie. Every fixture row is overhead-dominated.
  We certify a bound where monoprop reports nothing.
- **Fermionic (E6, andromeda).** ffsim is 8.26× faster on LUCJ-16, and that
  row is overhead-dominated with stated asymmetries. Five capability or
  correctness rows, including the 64-mode Kitaev energy no dense competitor can
  hold.
- **CV (E7, akilles).** One correctness row against piquasso, agreeing to
  5.7e-14. Speed classification refused: NumPy plus numba at two modes.

---

## How a row is stored

One JSONL file per lane beside the lane's script, written through
`crates/omega-emu-compare`'s `RowWriter`. The schema is one struct for every
lane, deliberately: the shape was about to be settled by whichever lane
produced a row first, and two shapes for one comparison means rows that cannot
go in a table together and prose citing fields only half the rows have.

A row carries `(circuit id, file + sha256, generator + seed, qubits, depth, χ,
shots, precision, box, date, git revision pair, per-arm versions,
BLAS/threads, knobs, executed-path witnesses, min/median/repeats, fixed startup
cost, value and oracle gap, load and device census, classification, ratio)` plus
a `derivation` block holding the inputs of every derived field.

**A row that cannot name its witnesses does not serialize.** That is a
type-level property, not a convention: the absent case has no JSON encoding, so
a lane that forgets to check still cannot publish such a row. Every prose
number in this document must resolve to a row in the JSONL with its witnesses —
checked by a script when the document closes, because prose without a row is
the dangling-citation defect one domain over.

## Scope, stated so it can be held to

One results document. Roughly 46 rows were budgeted — about 38 speed rows
and 8 capability rows. As published: 32 speed rows, 9 capability or
correctness rows and 14 void rows. That is fewer speed rows than budgeted,
because E3's were voided. Small enough that every row can be read and defended
individually in review. One script per lane. No runner framework, no plugin
abstraction, no YAML. If a row cannot be defended individually, it is deleted
rather than automated.
