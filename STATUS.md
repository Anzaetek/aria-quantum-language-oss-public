<!-- SPDX-License-Identifier: Apache-2.0 -->
# Status — 2026-08-16

Snapshot of what changed, what is verified, and what is knowingly left open.
Every number here was measured; estimates are labelled as estimates.

The previous edition of this file covered only the 2026-08-05 backend-correctness
work (§4 below, kept because it is still true). Twelve commits landed after it
without it being updated — that gap is what this edition closes.

## 1. Landed since 2026-08-05

| area | what | evidence |
|---|---|---|
| **Counts width** | `ExecResult::Counts` is keyed by `Outcome`, not `u64`; the 64-qubit cliff is gone | `omega-run ghz1024_full.qasm --backend mps`, all 1024 qubits measured (`89da782`) |
| **MPS fidelity** | per-run `fidelity_estimate` + `discarded_weight`, labelled an estimate | `50b7441` |
| **pauliprop** | `dropped_mass` reported by `omega-run` and tested **as a bound**, not an estimate | `crates/omega-cli/tests/dropped_mass_is_a_bound.rs` (`3725ba3`) |
| **pauliprop ↔ ppvm** | QuEra's `ppvm` as the *same-algorithm* anchor — the earlier "anchor" compared ppvm against Qiskit and never built our backend | 26 (circuit, observable) pairs, worst \|Δ\| = 0.000e0, 0 skipped (`d4d1229`) |
| **MPS ↔ Aer** | our wide MPS sampling against Qiskit Aer's `matrix-product-state` | 4 circuits; worst TVD 0.0327, worst per-qubit \|ΔP(1)\| 0.0158, worst \|Δ⟨ZᵢZⱼ⟩\| 0.0302 (`388cf31`) |
| **Governor** | five admission defects: batch pricing, the `Opaque` hole, the device memo | `PLAN-GOVERNOR-ADMISSION.md`, implemented and reviewed (`9ddf8e1`) |
| **OpenCL** | the feature 413'd *every* statevector run on the hardware it exists for | `c7e3348` |
| **QASM2 import** | `u` was the last import gap; `rxx`/`ryy`/`rzz` land too | `PLAN-CR-20260813.md` B1 |
| **CLI** | an unrecognised `--flag` was accepted and silently ignored | `d10c1cd` |

## 2. Landed 2026-08-15

**The `Outcome` migration had missed four feature-gated backends.** None of them
is built by `cargo test --workspace`, so each survived the migration by not
being compiled at all:

| crate | feature | defect |
|---|---|---|
| `aria-runtime` | `remote` | the wire key decoded through `u64::from_str_radix` — capped a remote run at 64 qubits and discarded the width below it |
| `omega-backend-statevector-metal` | `metal` | `ExecResult::Counts` built from a `u64` map |
| `omega-backend-statevector-opencl` | `opencl` | same |
| `omega-backend-statevector-cuda` | `cuda` | same, two sites |

`PLAN-WIDE-COUNTS.md` named this class of site in advance — *"`from_str_radix`
still funnels through `u64` … the conversion sites are where the defect will
survive if it survives anywhere."* It survived in exactly four of them.

The remote decoder's fix ships with three tests, each **mutation-checked**:
decoding back through `u64::from_str_radix` fails the 70-bit test and only that
one; pinning the width to 64 fails the width test and only that one.

**CI hygiene, same day:** stage 1 (rustfmt) and stage 2 (clippy `-D warnings`)
were both red on `main` — six files of fmt drift, and 16 lints in
`omega-parser/src/lower.rs` — from the three preceding commits.

**CPU statevector, PLAN-SV-PERF S1:** `apply_2q` scanned all `dim` indices and
rejected three of every four. It now walks the `dim/4` groups directly.
Measured **1.30×**, flat across widths and circuit shapes:

| circuit | before | after |
|---|---|---|
| ghz_28 | 15.640 s | 12.040 s |
| qft_26 | 45.490 s | 35.300 s |
| qft_28 | 213.790 s | 163.350 s |

Bit-for-bit identical, and tested that way: the replaced loop is kept verbatim
in `mod group_walk_equivalence` and compared against with `to_bits()` over
`n ∈ 2..=7` × every ordered qubit pair.

## 3. Measured baseline — CPU statevector

12-core Apple silicon, 24 GB, `rustc 1.95.0`, `--release`, one thread, 1000
shots. Full table in `PLAN-SV-PERF.md` §1.5.

- The **sampler is not a time cost** — 1.5 s of 215 s at 28 qubits.
- The **sampler doubles peak memory**: `qft_28` RSS 4099 MiB (evolution) → 8195
  MiB (with sampling). Evolution's 4099 MiB is `2^28 × 16 B` to the megabyte, so
  the state is the only large allocation in the gate loop; the other 4096 MiB is
  `sample_counts` holding two `2^n` f64 vectors. Fixing that is S3.

## 4. Verified (2026-08-05 work, unchanged)

- **`verification/Verification/Backend/PauliAlgebra.lean`** — 8 theorems,
  **proved**: no `sorry`, no Mathlib, no `native_decide`.
- **Qiskit differential cross-check** (`ARIA_QISKIT_XCHECK=1`): aria CPU vs
  Qiskit `4.441e-16`; stabilizer vs CPU `4.441e-16`; Metal vs CPU `1.857e-7`
  (f32).
- **Reset audited across all 11 backends** — no silent skips.
- Five defects fixed and evidenced: `stabilizer_expectation`'s pivot-less group
  test, `pauli_mult_phase`'s inverted `X·Z` (in two copies), the fast-reject on
  non-diagonal stabilizers, and two Metal `Reset` defects.

## 4b. Landed 2026-08-16 — the Linux/CUDA pass, on a real GB10

First execution of the CUDA stage on **Grace-Blackwell** (`aarch64`, sm_121,
48 SMs, 24 MiB L2, 121.7 GiB unified, CUDA 13.0.88, driver 580.159.03). Every
prior CUDA verification in this repo was x86_64 + RTX PRO 6000 + nvcc 12.9.
`ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1 ./ci.sh` → **exit 0**.

| area | what | evidence |
|---|---|---|
| **CUDA build** | two blind `Outcome` casts blocked the whole stage from compiling | `fc1a549`; the handoff doc listed 5 sites, there were 7 |
| **f64** | exact against an independent implementation on sm_121 | `f64_vs_qiskit` worst \|Δamplitude\| **1.388e-16**, ⟨Z⟩ **2.220e-16** |
| **Qiskit differential** | the mandatory gate, never run on this box | 60 agree, 0 disagree, worst \|Δp\| **4.441e-16** |
| **CUDA 13 / aarch64** | `cudarc 0.19.4` builds; NVRTC compiles 25 `.cu` × {f32,f64} on sm_121 | `precision_compile` |
| **SV 2q perf** | CX **1.73×**, SWAP **1.74×**, CZ **2.91×**, CRz **1.74×** | `31d53f5`, `4325a50`, `+0036` |
| **E7** | `RUST_TEST_THREADS=1`, stage-scoped | `33a40b4` |

**Two findings that are not fixes.**

- **The PauliProp GPU "accelerator" decelerates.** 0.11×–0.79× against the CPU
  branch across 6→100 qubits, **no crossover** — including past the 64-qubit
  symplectic word boundary where one might be expected. It had only ever been
  gated on *agreement*, never timed. Structural, not kernel: `BranchHook` takes
  a **host** `&mut PauliSum` and is called once per rotation gate, so each gate
  pays ~15 device allocations, 7 uploads, a full `synchronize()`, 7 downloads,
  and a host merge costing about what the CPU's whole branch loop costs. The
  `PAULIPROP_GPU_MIN = 256` default is **deliberately left alone** — it looks
  far too low, but no amd64 measurement exists.
- **A "speedup" that was a slowdown at low qubit indices, caught by review.**
  The 2q byte saving is a property of `min(qa, qb)`, not of the gate: DRAM
  granularity is a 32 B sector = 4 amplitudes, so below `min = 2` a two-slot
  gate (CX/SWAP/CRz) touches 2 of every 4 — the same sectors as dense, strided
  within them. Measured like-for-like at `qa ∈ {0,1}`: CX specialised 148.99 ms
  vs dense 135.41 ms, i.e. **1.10× slower**. CX ladders, GHZ chains and QFT all
  start at `(0,1)`. Those gates are now index-gated; CZ is not, because a
  one-slot diagonal touches 1 in 4 and wins at every index (122.28 vs 135.50 ms).
  A blanket threshold was written first and would have made CZ 12% slower.
- **Two claims corrected by measuring them.** "Register pressure" was a
  **local-memory spill** (`cuFuncGetAttribute`: 35 regs / 64 B local → 18 / 0
  after `#pragma unroll`, worth 16.7% on CRz), and CRz "bit-identity" was
  passing by luck of FMA contraction — it went red on this box when an unrelated
  unroll moved the schedule, exactly as it would have on nvcc 12.9 / H100.
- **The CUDA-graph path still builds dense 4×4s**, so the QML training hot path
  (`TrainStepGraph`, live production code) gets none of these speedups. The
  numbers above are the non-graph forward/adjoint path only.
- **The GB10 topology branch cannot fire on a GB10.** `nvidia-smi` reports
  `memory.total` as `[N/A]`, so the device row is dropped and the machine
  classifies `HostOnly`. Fails safe here by luck. The obvious fix was
  **withdrawn as dangerous** — `Unified` means no per-device ceiling
  (`worker.rs:762-765`), so it would over-commit a discrete H100. Needs an
  amd64 box. → `PLAN-LINUX-CUDA-GB10.md` §4

## 5. Known open — recorded, not fixed

> **Audited 2026-08-26.** Every entry re-checked against the code rather than
> carried forward. Four clauses had gone stale — items 6, 10 (both halves), the
> CUDA `CY` correction nested under 10, and 11 — and are struck through with the
> evidence inline. Two were closed by work that never came back to update this
> list; two were closed by someone else's work months earlier.
>
> Worth stating because of *how* they were found: item 10 is one sentence
> making two claims, and they went stale at different times. Auditing the entry
> as a unit would have kept whichever half was still true and preserved the
> other. Check each clause, not each bullet — a stale open item is worse than
> no list, since it sends the next reader to re-solve closed work and makes the
> live entries look equally doubtful.

1. ~~**CUDA is unverified for the 2026-08-15 fixes.**~~ **CLOSED 2026-08-16 on
   a DGX Spark GB10** — not the RTX 6000 Pro box this expected, but the harder
   target (`aarch64`, sm_121, CUDA 13.0.88). Two of the blind edits did not
   compile; `LINUX-CUDA-VERIFICATION.md` §1 tabulated five and there were
   seven. Fixed in `fc1a549`, then f32 **and** f64 both run:
   `f64_vs_qiskit` worst |Δamplitude| **1.388e-16**, and the mandatory Qiskit
   differential agrees on **60 circuits at 4.441e-16**.
   `ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1 ./ci.sh` → exit 0.
   **Not fully closed:** the `lib.rs:858` collapse-counts *width* question is
   semantic, so compiling it answers nothing, and CUDA still has no equivalent
   of `collapse_counts_use_the_creg_width.rs`.
2. **CUDA's Reset criterion diverges** (refuses on random *outcome*, not
   entanglement). A false rejection, not a wrong answer. → `LIMITATIONS.md`.
   Still open, but no longer *unfixable from here*: the `cfg`-gated arm now
   compiles and runs on a GB10, so the purity criterion the code comment
   prescribes (`reset_is_deterministic_within`) can finally be implemented and
   verified on hardware.
3. **Metal per-shot GPU trajectories block at ~64 shots** — in-flight
   command-buffer exhaustion. `execute` delegates to CPU meanwhile.
4. **`Reset.lean` / `StabilizerExpectation.lean`: 7 `sorry` targets.** They need
   a real ordered field; core Lean has none.
5. **Five distinct Reset acceptance policies** across backends. Only CPU↔Metal
   is conformance-tested. → ledger A6
6. ~~**`ci.sh` runs clippy without `--all-targets`.**~~ **CLOSED — verified
   2026-08-26.** Stage 2 is `Clippy -D warnings (WHOLE WORKSPACE, all targets)`
   and it passes: `grep -c 'clippy --workspace --all-targets' ci.sh` → 2, and a
   full run exits 0. The named regression
   (`omega-parser/tests/gate_arity_is_validated.rs:165`) is gone with it. The
   entry outlived the fix, which is the failure mode this list exists to avoid
   — a stale open item sends the next reader to re-solve closed work and makes
   the live entries look equally doubtful. → request E2
7. **`tools/qec_cross_check/run.sh` bootstraps with plain `pip`** —
   `pymatching` cannot source-build on aarch64. → request E6.
   **Confirmed on the hardware 2026-08-16**, and it is worse than "plain pip":
   there is **no aarch64 wheel on PyPI at all**
   (`--only-binary=:all:` → "no matching distribution"), *and* the cmake source
   build fails on a GB10. The MANDATORY QEC cross-check therefore cannot run on
   arm64 today by any route.
8. ~~**No `RUST_TEST_THREADS` in `ci.sh`**~~ **CLOSED 2026-08-16** — set
   stage-scoped on the CUDA stage. But the stated cause was **wrong**:
   `!Send`/`!Sync` is a compile-time property that *prevents* cross-thread
   sharing and cannot itself segfault a harness where each test builds its own
   backend, and `forward_graph.rs:167-172` already captures with
   `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`. `ForwardGraph::capture` is reachable
   only from `#[cfg(test)]`, so serialising is a fix for the symptom; **the
   cause is still unidentified** and `ci.sh` says so.
9. **`aria_py` bypasses `omega-server` admission entirely** — no way for a
   client to ask what a circuit would cost. → request R2b
10. ~~**CPU statevector is single-threaded per circuit** and 2q-heavy circuits
    run sparse gates as dense 4×4 — the CPU is the only backend without a
    diagonal fast path.~~ **CLOSED — both clauses, verified 2026-08-26.**
    Checked clause by clause rather than as one entry, which is what surfaced
    that they had gone stale at different times:

    * *single-threaded* — false. `apply_1q`/`apply_2q` carry 18 rayon parallel
      sites behind a `PAR_MIN_DIM` threshold; `PLAN-SV-PERF.md` S2 landed.
    * *sparse gates as dense 4×4* — closed 2026-08-25. `CZ`/`CRz` take a
      diagonal path, `SWAP` a permutation, `CY`/`CU3` a controlled-U path that
      never reads the control-zero half, and `Rbs` a {|01>,|10>} subspace
      kernel. Detected from the MATRIX, not routed per call site, so a diagonal
      gate added later is covered with no dispatch table to keep in sync.
      Measured 1.27–1.57× at n=22 (`98403d4`, `0ef773d`, `88ed02e`).

    → `PLAN-SV-PERF.md` S1b, S2; `PLAN-OPEN-20260825.md` §2.1

    **Correction, 2026-08-16: CUDA did not have that fast path either.** It has
    a diagonal kernel, but `apply_cx`/`apply_cy`/`apply_cz`/`apply_swap`/
    `apply_crz` all built a dense `[Complex64; 16]` and dispatched the generic
    `apply_2q` — 4 loads + 4 stores and 160 real flops per quad, to apply a
    permutation or a diagonal. `diagonal_factor` never caught them because it
    carries a **1-qubit** factor only. Fixed for CX/SWAP/CZ/CRz (`31d53f5`,
    `4325a50`); ~~**CY is still on the dense path.**~~ **CLOSED — verified
    2026-08-26**: `apply_cy` (`statevector-cuda/src/lib.rs:788`) routes through
    `apply_quad_swap_phase`, falling back to dense only below
    `QUAD_KERNEL_MIN_QUBIT_2SLOT`.
11. ~~**MPS is ~210× slower than our own statevector on one 19-qubit chain.**
    Reproduced, undiagnosed.~~ **CLOSED 2026-08-20, and the attribution here
    was wrong.** It was not an unexplained slowdown: the original invocation
    (`--backend mps`, then-default bond 64) was **silently truncating**,
    discarding 34% of the norm, and now REFUSES in 0.194 s with the dirty
    certificate in the message. The sanctioned `mps:auto` run is exact
    (certificate 3.47e-24, fidelity 1.0) and slow for a diagnosed reason —
    `max_bond_reached` 511 is full rank at the 19q middle cut, so the CR's
    "strictly low entanglement" label was itself wrong.
    → `PLAN-CR-20260813.md` B4
12. **The ppvm bridge refuses any noise request** — ppvm's tableau supports the
    channels including atom loss; the mapping is unwritten.
14. **`hhl.aria` and `proofs/lean4/QuantumProofs/HHL.lean` need cleaning up —
    and the link between them is an annotation, not a check.** The off-by-one
    fixed on 2026-08-15 is the symptom; the structure around it is the item.
    Three separate things to settle:
    - **The example does not implement what the proof proves.** `HHL.lean` is
      sorry-free and genuine — `hhl_solves_system` is a real matrix-vector
      identity `A · o = C · b` for `A = diagonal λ`, with an exact QPE and a
      controlled `RY(2·arcsin(C/λᵢ))`. The Aria example instead uses a
      *proxy* eigenvalue (`theta = 0.5 / (k + 1)`, commented "eigenvalue proxy
      λ = k + 1") and, at `hhl.aria` step 3, an "inverse QFT" that is **H on
      each qubit and nothing else** — a true inverse QFT needs the controlled
      phase ladder, so for `n ≥ 2` that step is incomplete. Whether the example
      should be brought up to the proof, or the header claim narrowed to what
      the example is (a template, not a certified HHL), is a decision, not a
      bug fix.
    - **`@prove "hhl_recovers_inverse"` is not verified against anything.** The
      annotation names a property; `ci.sh:656` separately checks the Lean
      theorems are sorry-free. Nothing connects the two — no check would notice
      if the circuit and the theorem drifted apart, and they have.
    - **The harness cannot see circuit-level defects at all.** It compares the
      WASM path against an in-process oracle **on the same lowered IR**, so it
      is a transport check. It reported `Δmax = 0.000e0 PASS` for as long as the
      off-by-one existed. **The other 48 examples run through the same
      structurally-blind check.** The *loop-bound* class specifically has now
      been audited — see below — but that is one class of circuit defect, and
      the harness is blind to all of them.

    **Loop-bound audit, 2026-08-16 — `hhl` was the only one.** All **115
    `repeat` loops across the 44 `examples/aria/*.aria`** were checked against
    their register declarations. Everything else is right, and several are
    right in a way that shows the inclusive semantics were understood:
    `bernstein_vazirani.aria` and `deutsch_jozsa.aria` deliberately use
    `to n` for the Hadamard sweep (the answer qubit at index `n` gets
    `X` then `H`, making |−⟩) and `to n - 1` for the query register, in the
    same file. `qft.aria` and `qec_qft.aria` implement the real controlled-phase
    ladder with `from i + 1 to n - 1` and reverse it with `step -1`.

    Two things that follow. First, `hhl.aria` was an outlier rather than a
    symptom of a house-wide misunderstanding. Second, **`qft.aria` already
    contains the correct inverse QFT that `hhl.aria` step 3 is missing** — so
    reconciling the example needs no new physics, only reuse.

    Scope of that audit, stated so nobody over-reads it: it covers loop bounds
    against register sizes. It does **not** cover gate choice, angle formulas,
    or eigenvalue proxies — the other three ways `hhl` is still not what
    `HHL.lean` proves.
15. **The photonic backend has never been compared on *speed*, only on
    agreement.** `bridge-perceval` exists and the DV/CV conventions are matched
    verbatim, so the correctness axis is covered — but Perceval's hot path is
    C++ (`quandelibc`) and piquasso's is NumPy/JAX, and neither has been timed
    against `omega-backend-photonics`. Deferred deliberately: the statevector
    lane (`PLAN-SV-PERF.md`) and the adjoint lane
    (`PLAN-ADJOINT-MEMORY.md`) are the measured bottlenecks today, and a
    photonic comparison wants the same discipline they got — a baseline before
    a claim, and per-stage timings rather than wall clock.
