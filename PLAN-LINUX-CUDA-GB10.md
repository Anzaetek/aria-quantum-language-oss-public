<!-- SPDX-License-Identifier: Apache-2.0 -->
# Plan — the Linux/CUDA pass, on a real GB10 (2026-08-16)

**Revision 2**, after an adversarial review that killed two items and corrected
four. Revision 1's mistakes are kept visible in §9 rather than quietly deleted,
because two of them are the exact failure mode this repo keeps recording.

The box: **DGX Spark GB10**, `aarch64`, sm_121, 48 SMs, 24 MiB L2, 121.7 GiB
unified, driver 580.159.03, **CUDA 13.0** (nvcc 13.0.88), rustc 1.95.0, 20
cores. Measured via `cudaGetDeviceProperties(0)`:

```
integrated = 1        totalGlobalMem = 130662936576 (121.69 GiB)
pageableMemoryAccessUsesHostPageTables = 1     sm_121   48 SMs   L2 = 24 MiB
cudaMemGetInfo: free = 9.25 GiB   total = 121.69 GiB
```

Every prior CUDA verification in this repo was x86_64 + RTX PRO 6000 + nvcc 12.9.

## 0. The standing constraint

**Everything must keep working on amd64/Xeon with an RTX 6000 Pro or an H100,
and on nvcc 12.x.** The GB10 exposes the gaps; it is not the box the fixes get
specialised to.

Revision 1 put a portability table here asserting what those machines do. The
review's response was correct and worth keeping as the rule: **an assertion
about hardware nobody in this plan runs on is the sin the table claims to
prevent.** So the table is now a list of *unknowns to be measured*, not
invariants:

| platform | status |
|---|---|
| aarch64 + GB10 | **measured**, this box |
| amd64 + H100 / RTX 6000 Pro | **UNVERIFIED** — no such box in this pass |
| nvcc 12.x | **UNVERIFIED** — only 13.0.88 here |
| aarch64 + GH200 | **UNKNOWN** — `integrated` is load-bearing and unmeasured |

Consequence, adopted as policy: **no change whose failure mode is invisible on
GB10 may land in this pass.** That is what disqualifies W3/W4 below.

---

## 1. Measured ground truth, before any change

**Green, first contact on Grace-Blackwell:**

| check | result |
|---|---|
| `cudarc 0.19.4` against CUDA 13.0 on aarch64 | builds |
| `precision_compile` — 24 `.cu` × {f32, f64} on sm_121 via NVRTC | pass |
| `cargo test --features cuda --lib` | **28 passed, 0 failed, 2 ignored** |
| `f64_agreement` | 3/3 |
| `f64_vs_qiskit` — worst \|Δamplitude\| **1.388e-16**, worst \|Δ⟨Z⟩\| **2.220e-16** | 2/2 |
| `graph_replay_smoke`, `oom_fallback`, `qml_device_hint`, `qml_no_host_syncs` | 8/8 |

**Red, and fixed (`fc1a549`, patch 0029):** two `Outcome as usize` casts blocked
the whole `ARIA_CUDA=1` stage from building.

**A false green, caught and closed.** `f64_vs_qiskit` reports `ok` when
`.venv-qiskit` is absent — a loud skip that still counts as a pass. The venv did
not exist on this box, so the f64-vs-Qiskit gate was "passing" by not running. It
has been built (qiskit 2.5.2, aer 0.17.2, numpy 2.5.2, all aarch64 wheels) and
the numbers above are from a real run.

**Version drift to watch:** `reset_channel_matches_aer_ground_truth` pins its
numbers to qiskit 2.4.1 / aer 0.17.2; this box has qiskit **2.5.2**. It passes,
but the pin is now stale by one minor version and should be re-pinned or
widened deliberately rather than by accident.

---

## 2. W1 — Unblock the CUDA test build ✅ DONE (`fc1a549`)

Two `Outcome as usize` casts at `lib.rs:1971,2043`, both test-side, both
*beyond* the five `LINUX-CUDA-VERIFICATION.md` §1 lists. Swept every
feature-gated crate by hand for the same pattern; no further sites.

**Still open, and W1 is not closed until it exists.** The review's sharpest
point: *"the plan declares correct what was only compiled."*
`LINUX-CUDA-VERIFICATION.md` §1 raises a **semantic** question a compiler cannot
answer — whether `lib.rs:858`'s collapse arm should key counts on
`circuit.num_qubits` or the creg width, noting *"if the CUDA collapse counts
come out at the wrong width, this line is why."* The CPU has a dedicated test
(`omega-cli/tests/collapse_counts_use_the_creg_width.rs`); **CUDA has none**, and
`ci.sh:359-388` never exercises `MidCircuitMode::Collapse` on CUDA.

→ **W1b: add a CUDA collapse-counts-width test.** §1 may not be closed in the
docs until it passes.

## 3. W2 — `RUST_TEST_THREADS`, with an honest diagnosis

Revision 1 repeated the docs' explanation: flaky *"because
`CudaStatevectorBackend` is neither `Send` nor `Sync` by construction."* **That
reasoning is wrong.** `!Send`/`!Sync` is a compile-time property that *prevents*
cross-thread sharing; it cannot itself cause a SIGSEGV when each `#[test]`
builds its own backend. And `forward_graph.rs:167-172` already uses
`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` precisely so other threads' default-stream
work is not captured — the named mechanism is both wrong and already mitigated.

What is true: `ForwardGraph::capture` is called **only** from `lib.rs:2372,
2475, 2581`, all inside `#[cfg(test)]`. It is not reachable from `execute()`. So
the capture/execute race is genuinely test-only and serialising tests is a
defensible **workaround** — but it is not a diagnosis, and the commit must say
so rather than inherit the wrong one.

Two corrections to revision 1:

- **Stage-scoped, not process-wide.** Revision 1 said "on the CUDA stage" in one
  place and `RUST_TEST_THREADS=1 ARIA_CUDA=1 ./ci.sh` in another. The latter
  serialises every CPU stage too, which is slow and would mask an unrelated
  CPU-side race. Stage-scoped is correct.
- **"8 consecutive runs" is not a gate.** At the documented 1-in-4 flake rate,
  `P(8 clean | nothing fixed) = 0.75⁸ = 0.100` — a 10% chance of declaring
  victory having changed nothing. **n ≥ 11** for 95%, **n ≥ 17** for 99%. Use 17
  or stop calling it evidence.

## 4. W3 / W4 — topology. **WITHDRAWN from this pass.**

Revision 1 proposed: a device whose capacity `nvidia-smi` will not report
(`[N/A]`, the GB10 case) should stop being dropped, and should fall through to
rule 7 → `Unified`, *"the safe direction by the file's own stated asymmetry."*

**That is wrong, and it is the most dangerous thing revision 1 contained.** The
asymmetry at `topology.rs:15-21` holds only if `Unified.total_bytes` is the
small number. It is not: `classify` builds `Unified { total_bytes:
host.max(device_total) }` (`topology.rs:245`), and then —

- `worker.rs:762-765` — `pool_for()` returns the Unified pool for **any**
  target, `ExecTarget::Device(i)` included;
- `worker.rs:781-784` — `has_pool_for(Device(i))` is **unconditionally true**
  when a Unified pool exists.

**Unified means there is no per-device ceiling at all.** So on an 8×H100 DGX
where one row reports `[N/A]`, today's behaviour drops the bad row and yields
`Discrete` with seven 80 GB pools — under-provisioned, safe. Revision 1's change
would have blocked rule 6 and produced **one 2 TB pool** admitting every GPU job:
a 25× device over-commit. And with all rows `[N/A]` on a discrete box (driver
hiccup, GPU off the bus), today refuses GPU work; revision 1 would have admitted
it against host RAM. It converts *"refuse work we cannot size"* into *"admit it
against the wrong pool."*

That failure is **invisible on GB10** — where `HostOnly` happens to be
numerically right — and fires on the amd64 box this pass never runs on. By §0's
policy it cannot land here.

**Two further reasons it is not ready**, both of which also sink W4:

- **W4 does not work as specified.** Rule 2 (`devices.is_empty()` → `HostOnly`,
  `topology.rs:169`) runs **before** rule 3 (`cuda_integrated`,
  `topology.rs:179`). Populating the flag alone never reaches rule 3, because on
  GB10 `probe.devices` is empty. W4 needs its own device enumeration
  (`cudaGetDeviceCount` + properties) replacing the `nvidia-smi` list, plus the
  `integrated` check moved above the `is_empty()` guard.
- **W4 would make W3 dead code** on exactly the builds that matter: once rule 3
  short-circuits, W3's path is reachable only on the non-CUDA build — the one
  nobody runs on a GPU box.

**A real pre-existing bug found in passing, and NOT introduced by this plan.**
Rule 5 fires when a single device is within `SAME_POOL_TOLERANCE = 0.25` of
host. An RTX 6000 Pro (96 GB) in a 128 GB workstation: `|96 − 128| = 32 ≤ 128 ×
0.25 = 32` → `DeviceMatchesHost` → **Unified today**, on a discrete box, with
no per-device ceiling. That is the hard constraint's own hardware,
misclassified by the shipped code, before anything here changes.

**Also recorded:** `classified.reason` is **discarded** at `worker.rs:749` and
`health_snapshot` (`worker.rs:915-950`) has no reason field — so
`topology.rs:72`'s claim that the reason is *"surfaced in `/health`"* is already
false. Any new `TopologyReason` variant changes nothing observable without a
`worker.rs` change revision 1 did not scope.

→ **All of this moves to a follow-up with an amd64 box in the loop**, and wants
fixture tests built from *captured real* `nvidia-smi` + `/proc/meminfo` output
from a discrete machine, not from round numbers chosen to make the assertion
pass. Recorded in `CUDA_TODO.md`; nothing lands here.

## 5. W5 — CUDA's Reset criterion diverges

Unchanged from revision 1 and still valid. CUDA refuses when the *outcome* is
random; CPU/Metal refuse when the qubit is *entangled*, so `H q0; Reset q0` is
accepted by CPU/Metal and rejected by CUDA (`lib.rs:1396-1422`,
`LIMITATIONS.md:181-227`). A false rejection, not a wrong answer. Adopt
`reset_is_deterministic_within`; needs a real dep on the CPU crate and a device
readback. Arch-neutral. Verify: `H q0; Reset q0` accepted and equal to CPU,
entangled cases still refused by both, both Reset gates still green.

## 6. W6 / W7 — sweep and run

- **W6** — `bindings/aria-py` is a separate workspace; its `cuda` feature
  (`Cargo.toml:41-44`, chaining all three CUDA backends) has never been compiled
  anywhere. `cargo check --features cuda --all-targets` inside it.
- **W7** — the stage, **f32 and f64 both**, each set run twice.
  `gpu_cuda_agrees_with_sim` is the one that matters: the CPU side moved this
  branch (PLAN-SV-PERF S1) claiming bit-identity, so **any** delta movement is a
  finding, not a tolerance to widen.
- **OpenCL is out of scope** by explicit instruction. (For the record, it cannot
  build here anyway: only `ocl-icd-libopencl1` is installed — `libOpenCL.so.1`
  exists but there is no `libOpenCL.so` dev symlink and no `CL/` headers.)

---

## 7. P1 — CUDA statevector performance

**Revision 1's candidate list was largely not grounded in the kernels.** Two of
four were wrong; see §9. What follows is read out of the `.cu` files.

### The finding: every 2q gate pays a dense 4×4 matvec

`lib.rs:528-584` — `apply_cx`, `apply_cy`, `apply_cz`, `apply_swap` each build a
full `[Complex64; 16]` and call `apply_2q`. `apply_2q.cu:61-79` then does, per
quad, unconditionally: **4 loads + 4 stores, 16 complex multiplies + 12 complex
adds** — to apply gates that are permutations or diagonals and need no
arithmetic at all.

`diagonal_factor` (`lib.rs:1308-1349`) returns only a **1-qubit** factor and
matches only Z/S/Sdg/T/Tdg/Rz/U1, so 2q diagonals (CZ, CP/CU1) are not caught
upstream either — a CZ ladder is N dense kernel launches.

### P1-a — CX / SWAP permutation kernels. **BIT-IDENTICAL.**

CX and SWAP permute two of the four amplitudes in a quad and leave the other two
untouched: 2 loads + 2 stores + **zero flops**, vs 4+4 and 160 flops.

> **CORRECTION 2026-08-27 — do not set the acceptance bar at bit-identity.**
> This section read "a permutation moves bits, it does not compute — so the
> acceptance bar is **bit-identity**, the strongest gate this repo has (cf. the
> CPU S1 work's `to_bits()` comparison)". The CPU side then implemented S1b
> against exactly that bar and shipped the claim three times before retracting
> it.
>
> A permutation computing nothing is true and is not the relevant fact. The
> difference comes from the two amplitudes the kernel LEAVES UNTOUCHED: the
> dense reference writes them as `1*a00 + 0*a01 + 0*a10 + 0*a11`, and `0.0 * a`
> is `-0.0` for negative `a`, so the dense result carries sign information the
> permutation kernel never reads. A `-0.0` survives the fast path and is
> canonicalised to `+0.0` by the dense one. `kernels/apply_quad_perm.cu`
> §SIGNED ZERO has said so all along.
>
> **The bar that works**: equal under `==` everywhere, and `to_bits()`-equal
> everywhere the value is non-zero. That is what
> `every_2q_gate_differs_from_the_dense_scan_only_in_the_sign_of_zero` asserts
> in `omega-backend-statevector`, and it is what `PLAN-SV-PERF.md` §S1b
> prescribed before any of this was written.
>
> A `to_bits()` bar here fails one of two ways, both bad: spuriously, if the
> fixture contains signed zeros; or VACUOUSLY, if it does not — which is how
> the CPU version passed while asserting something false. Use a fixture with
> zeros beside negative neighbours, or the bar proves nothing.

CX dominates every fixture here (ghz, qft ladders, HEA, RBS layers). Highest
value, lowest risk. **First.**

### P1-b — CZ / CP as 2q diagonals. Two tiers, different risk.

- **tier 1** — route CZ/CP to a 2q-diagonal kernel: touch only `i11`. The
  multiply is exact (`×(−1.0)` exactly; `×e^{iφ}` is the same single complex
  multiply either way), but **"touch only `i11`" carries the same signed-zero
  exposure as P1-a** — the three untouched slots are written by the dense
  reference and their `-0.0` canonicalised. Measured on the CPU equivalent:
  the diagonal differs in 7 of 16 sign combinations. Use the P1-a bar, not
  `to_bits()`.
- **tier 2** — extend the fusion chain to carry 2q diagonals into
  `flush_pending`. This **reorders floating-point multiplies** into a fused
  product, so it is *not* bit-identical and needs a stated tolerance. Only if
  tier 1's measurement says launches are the cost.

### Explicitly NOT doing

- **Block-size tuning.** Revision 1 proposed querying it from the device. Block
  size here is **not a free knob, it is a correctness precondition**:
  `imp.rs:404-407` says the in-kernel reductions assume power-of-two
  `blockDim.x`, and `imp.rs:362` sizes shared memory as `block_size * 4`.
  Changing it changes the **reduction tree shape** (so `pauli_expectation` /
  `inner_product` summation order becomes machine-dependent — GB10 and H100
  would differ in the last bits) and re-partitions the **CDF scan**
  (`imp.rs:340-400`), moving where seeded shots land. The banded seeded tests
  would not catch a small shift and would flake on a large one. A "no functional
  change" patch that makes GPU results machine-dependent is the one thing the
  cross-backend gate structure assumes cannot happen. **Dropped.** If launch
  geometry is ever touched, it must be grid-stride with a *fixed* block size and
  a device-queried *grid*, leaving reduction tree and CDF partition invariant.
- **Host-side 1q matrix fusion.** Revision 1 called it "arithmetic-free and
  exact". It is neither: sequential is `round(U₂·round(U₁·v))`, fused is
  `round((U₂U₁)·v)` — different rounding. Worse, diagonals commute with each
  other but **not** with a non-diagonal 1q gate on the same qubit, so any walker
  that fuses "adjacent 1q on the same qubit" without modelling the unflushed
  `pending` diagonals reorders gates and produces a wrong state. Deferred; if it
  ever lands it needs a stated tolerance and a same-qubit `H·Rz·H` regression.
- **Graph-capture coverage measurement.** `ForwardGraph::capture` is called only
  from tests, so it is 0% of `execute()` by construction. The honest item is
  "wire it into `execute` at all" — a feature, not a measurement.

### P1-0 — the baseline instrument must be fixed FIRST

`benches/cuda_bench.rs` has only `bell_chain` and `qaoa_layer`, sweeps
`[12,14,16,18,20]` qubits, and has **no f64 arm**. At 20 qubits the f32 state is
2²⁰ × 8 B = **8 MiB — inside this device's 24 MiB L2.** The entire sweep is
L2-resident and launch-latency-dominated: it cannot measure memory traffic and
will make "fewer launches" look like the whole game.

**A baseline taken on the current bench is worse than none, because it will be
cited.** So: extend to 24/26/28 qubits (28q f32 = 2 GiB, comfortable in 121.7
GiB), add a QFT and a CX-dense fixture, add an f64 arm. *Then* baseline, then
optimise.

## 8. P2 — PauliProp GPU

Revision 1 framed this as baseline-discovery. The structure is not actually open
— `lib.rs:68` calls `gpu::branch_on_gpu` **once per rotation gate**, and
`gpu.rs:185-330` does full `memcpy_htod` of the packed sum → launch →
`s.synchronize()` (`gpu.rs:313`) → six `clone_dtoh`. **Branches are marshalled
host↔device per gate, with an unconditional sync.** The optimisation is
therefore already identified: keep the sum resident on device across consecutive
rotations and sync once per layer.

**But first, a gate that does not exist.** Revision 1 claimed
`dropped_mass_is_a_bound.rs` protects this. It does not: it constructs only
`PauliPropBackend` (the CPU crate), `omega-cli` has no `cuda` feature, and
`ci.sh`'s CUDA stage never invokes it. **P2's only named safety net does not
test the code P2 modifies.** → add a CUDA-arm `dropped_mass` bound test *before*
touching the marshalling.

Also dropped: comparing the CUDA split against `omega-backend-pauliprop-metal`.
Metal's `gpu.rs:66` has the identical `branch_on_gpu` signature and structure —
there is no different split to compare against.

---

## 9. What revision 1 got wrong

Kept deliberately. Two of these are this repo's signature defect, committed by
the plan that warns about it.

1. **W3 "unknown capacity → Unified is the safe direction."** Wrong, and
   dangerous: `Unified` removes the per-device ceiling entirely. Would have
   OOM'd an H100 weeks later with `/health` reporting a healthy pool.
2. **P1 "check the GPU 2q path for the same waste the CPU just fixed."**
   `apply_2q.cu:4-5` already threads one thread per quad with bit-deposit
   addressing. Worse, `PLAN-SV-PERF.md:412` is *titled* "S1b — the lever S0
   exposed, and **every GPU backend already pulled it**". The plan cited that
   document three times and proposed the thing it records as done. Written from
   the CPU plan's shape rather than from the `.cu` files.
3. **P1 block-size tuning** — proposed as portability hygiene; would have made
   numerics machine-dependent.
4. **P1 1q fusion "exact"** — exempted itself from the tolerance rule stated in
   the same paragraph.
5. **W2's diagnosis** — inherited from the docs without checking; `!Send`/`!Sync`
   cannot cause the described SIGSEGV.
6. **W7's environment survey** — asserted venvs were absent; `.venv-qiskit` now
   exists and the Qiskit gates run.

The pattern in 1, 2 and 5: **reasoning from a document instead of from the
code.** `CUDA_TODO.md`'s own standing rule — *"do not close any item below from
reasoning alone — run it"* — applies to opening them too.

---

## 10. Sequencing, CI, patches, rollback

**Correctness first, then performance:**

W1 ✅ → W1b → W2 → W5 → W6 → W7 → **P1-0 (baseline) → P1-a → P1-b tier 1** →
P2-gate → P2 → W8 (docs).

W3/W4 withdrawn. P1-b tier 2 and 1q fusion deferred with tolerances unstated.

**Rollback** (absent from revision 1): every item is one commit and one patch
file, and each is independently revertable — no two items in this sequence touch
the same function. The revert boundary for the performance work is P1-0: it adds
a bench and changes no shipped code, so `git revert` of any P1-x leaves the
baseline in place to re-measure against. Nothing in this pass touches
`topology.rs`, so the governor cannot regress.

Per commit, without exception:

1. `./ci.sh` — plus `ARIA_CUDA=1 ./ci.sh` (stage-scoped `RUST_TEST_THREADS=1`)
   for anything touching CUDA.
2. `git commit` — **never `git push`**.
3. `git format-patch -1 -o fixes/patches/ --start-number NN`, continuing 0029.
4. Row appended to `fixes/patches/README.md`.

**Review gates:** brutal subagent review of this plan (done — revision 2 is its
output), again after P1-a and P1-b land, and again before W8 closes any doc
item.
