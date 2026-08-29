<!-- SPDX-License-Identifier: Apache-2.0 -->
# CUDA — work waiting for a Linux + NVIDIA box

**Two target platforms, and they exercise different code:**

| platform | hardware | why it matters here |
|---|---|---|
| **linux/amd64** | discrete NVIDIA (RTX, A100, H100, DGX A100/H100) | the **Discrete** topology path: host pool + one pool per GPU |
| **linux/arm64** | Grace-Blackwell **GB10 (DGX Spark)**, Grace-Hopper **GH200** | the **Unified** path — and the heuristic that decides it |

**arm64 is the higher-risk target**, and currently the least tested. The
topology classifier treats `aarch64` specially: a single device whose
`memory.total` is within 25% of host RAM is read as *one shared pool seen
twice*, which is the GB10 signature. That branch has **never run on real
hardware** — only against synthetic probe values in unit tests.

Get it wrong on a GB10 and the governor budgets host and device separately, i.e.
hands out roughly **twice the machine's memory**, and the OOM killer enforces
the difference. Get it wrong the other way on a GH200 (which is coherent but has
*distinct* LPDDR5X + HBM3 capacities, so it should classify **Discrete**) and
you merely under-use HBM. The asymmetry is why the code defaults to Unified when
unsure — but a default is not a substitute for measuring.

Everything here is
**unrunnable on the macOS dev machine** (`nvidia-smi` absent,
and the CUDA crates are `cfg`-gated to linux/windows). It is written down rather
than remembered, because this project has already been burned once by landing
GPU code that was never executed: `f11a9f5` shipped with 2/70 Metal tests
failing because the work was verified on a CUDA box and the Metal mirror was
"deferred".

Read that as the standing rule: **do not close any item below from reasoning
alone — run it.**

## 0. NEW CUDA SURFACES — check these, they are easy to miss

**Read this before assuming §1's stage covers everything.** The CUDA footprint
grew on 2026-08-17 and several of the new pieces are behind features or flags
that a default build never touches. This project's most repeated defect is
exactly that: code behind a `cfg` that no CI command compiles. It is how the
`Outcome` migration survived in four backends at once, and how the CUDA
statevector shipped with two casts that had never been through a compiler.

| surface | how to exercise it | gated by CI? |
|---|---|---|
| **`omega-server --features cuda`** — NEW. The daemon had no CUDA dependency at all; a job submitted over HTTP could not reach the GPU however the box was built. | `cargo test -p omega-server --features cuda` | **yes, added to the `ARIA_CUDA=1` stage** |
| **`--multi-control exact`** — exact CCX/CSwap octet permutation, 30.6x faster and more accurate, **opt-in** so the default is unchanged | `omega-run c.qasm --device cuda --multi-control exact` | yes (`tests/multi_control_modes.rs`) |
| **PauliProp over HTTP** — the server had NO `PauliProp` variant; the backend was unreachable by any spelling | server tests | yes |
| **New quad/octet kernels** — `apply_quad_swap`, `apply_quad_swap_phase`, `apply_quad_phase1`, `apply_quad_phase`, `apply_octet_swap` | `tests/quad_perm_bit_identity.rs`, `tests/kernel_resource_report.rs` | yes |
| **PauliProp vs Qiskit** — this backend had NO independent cross-check before | `ARIA_QISKIT_XCHECK=1 ./ci.sh` | yes |
| **`tch` on aarch64** | `ARIA_TCH_CUDA=1 tools/setup-libtorch.sh` | no — opt-in script |

**Two traps that cost time here, both aarch64-specific:**

1. **`tch` needs `CARGO_BUILD_TARGET`.** `RUSTFLAGS` with `--no-as-needed
   -ltorch_cuda` lands on host **proc-macro** `.so`s, which rustc then `dlopen`s
   — dragging the CUDA stack into the compiler and **SIGSEGV**ing it while
   building unrelated crates (`serde`, `zerocopy`). `setup-libtorch.sh` now
   exports the host triple; if you bypass the script, pass `--target` yourself.
2. **`--device cuda` does NOT reach the MPS SVD hook from `omega-run`.**
   `omega-cli` never wires `cuda_svd_flat`; only `aria-runtime` does
   (`run.rs`, `with_svd_fn`). A CPU-vs-"CUDA" MPS comparison through the CLI
   measures the CPU twice. The tell is a missing ~0.58 s CUDA startup cost in
   the "GPU" column — if a device measurement contains no device cost, it is not
   one.

**And two results that are counter-intuitive enough to re-check rather than
assume** (`BACKEND-CROSSOVER.md` has the numbers): PauliProp on GPU is **slower
than the CPU at every width measured**, and MPS via cuSOLVER `gesvdj` is
**0.34x** here — it is native f64 and GB10 runs f64 at 1/64 rate, which is
roughly a 32x handicap against a datacentre part at ~1/2. **No RTX PRO 6000 or
H100 measurement exists** — nobody has run this code on one — so treat the 3x
figure in `GPU_BACKEND_PLAN.md` as the target it always was, not as a result
from other hardware. On any amd64 discrete box, **measure before believing
either of these.**

## 1. Run the stage at all

```console
$ ARIA_CUDA=1 ./ci.sh
```

Gates the CUDA statevector, MPS (`gesvdj`), pauliprop, RBS forward + adjoint,
and the Reset-channel tests (`reset_matches_cpu`,
`reset_channel_matches_aer_ground_truth` with on-device Born sampling).

Also run the mandatory cross-checks there — they are not Mac-specific:

```console
$ ARIA_QISKIT_XCHECK=1 ARIA_QEC_XCHECK=1 ARIA_CUDA=1 ./ci.sh
```

## 2. Verify the resource governor against real discrete GPUs

This is the highest-value item, because **A7b's device pools have only ever met
the OpenCL path on unified-memory hardware.** On a discrete box the code takes
branches nothing has exercised.

- `topology.rs` must classify the machine **Discrete**, not Unified. Check
  `GET /health` reports `"unified": false` and one pool per GPU. The heuristics
  are unit-tested against synthetic probes; this is the first contact with real
  `nvidia-smi` output, so confirm `parse_nvidia_smi` handles the actual format
  (multi-GPU, ECC-reserved memory reducing `memory.total`, MIG partitions).
- A job larger than device memory must be **refused** even when host RAM is
  ample, quoting the *card's* capacity.
- A full GPU must **not** block host work — the pools are independent.
- `ExecTarget::Device` is only constructed under `--features opencl` today.
  Wiring CUDA means `exec_target_for` must mirror CUDA dispatch the way it
  mirrors OpenCL's. **If those two ever disagree the reservation is against the
  wrong pool** — noted in the code, worth re-reading before touching it.
- Device work is priced at **f32** (8 B/amplitude). Confirm the CUDA backend
  really is f32; if any path is f64, the pricing under-refuses by 2×.

## 3. Grace-Hopper / GB10 caution

If the box is a **DGX Spark (GB10)** or **GH200**, topology detection is the
first thing to check, and the two are *not* the same:

- **GB10** shares one physical pool between CPU and GPU. It must classify
  **Unified**. If it comes out Discrete, the governor budgets host + device
  separately and can hand out ~2× the machine's memory.
- **GH200** is NVLink-coherent but has *distinct* capacities (LPDDR5X + HBM3),
  so it should classify **Discrete** with the spill behaviour noted in
  `FIXES_PLAN.md` A7b.

Detection prefers `cudaDeviceProp.integrated` when a CUDA-linked build can ask
for it; this server does not link CUDA, so it currently falls back to the
`device_total ≈ host_total` heuristic. **Populating `cuda_integrated` on a
CUDA build is the clean fix** and would remove the guesswork entirely.

`OMEGA_MEM_TOPOLOGY=unified|discrete|host` overrides detection if it gets it
wrong — but please report the misclassification rather than just overriding it,
since the heuristic is meant to be right by default.

## 4. RESOLVED (verified 2026-08-17) — the CUDA Reset criterion divergence

**Confirmed fixed by test**, not by inspection:
`crates/omega-backend-statevector-cuda/tests/reset_criterion.rs` passes 3/3
under `--features cuda`, including
`unentangled_superposition_reset_is_accepted_and_matches_cpu` — i.e. the exact
case this section says was wrongly refused (`H q0; Reset q0`) is now accepted
and agrees with the CPU. The other two pin the criterion from the opposite
side, so it cannot be "fixed" by waving everything through.

Original write-up kept below.

### (original) Known-open CUDA defect to confirm

`STATUS.md` / `LIMITATIONS.md` record that **CUDA's Reset criterion diverges**:
it refuses on a random *outcome* rather than on entanglement, so it rejects
`H q0; Reset q0` — which the CPU accepts. That is a false *rejection*, not a
wrong answer, and it could not be fixed from the Mac because the arm is
`cfg`-gated. Confirm it still reproduces, then align it with the CPU's criterion.

## 5. Reporting back

**Run `./tools/cuda-report.sh` and paste the output** — it collects everything
below, so the handoff does not depend on remembering which numbers matter. Add
`--with-ci` to also run `ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1 ARIA_QEC_XCHECK=1
./ci.sh`.

It deliberately reports the **raw probe** (`nvidia-smi memory.total` plus
`/proc/meminfo MemTotal`) alongside the verdict: if the classifier gets the
topology wrong, those two numbers are what shows why, and they are what the 25%
tolerance is tuned against.

Record actual numbers, not "passed":

- `./ci.sh` exit code, with the CUDA stage's `OK:` lines
- `/health` output showing the detected topology and per-pool capacities
- GPU model(s), driver, CUDA version, and `nvidia-smi --query-gpu=index,name,memory.total`
- `uname -m` — **`x86_64` vs `aarch64` changes which classifier branch runs**,
  so always report it alongside the `/health` topology

`OPTIONAL_TESTS.md` has the table to update; `FIXES_PLAN.md` A7b has the design
rationale if a decision looks wrong on real hardware.

## linux/arm64 specifics (GB10 / GH200)

- **Report the raw probe, not just the verdict**: `nvidia-smi
  --query-gpu=index,name,memory.total --format=csv,noheader,nounits` plus
  `MemTotal` from `/proc/meminfo`. If the classifier is wrong, those two numbers
  are what shows why, and they are what the 25% tolerance is tuned against.
- **GB10 must classify `Unified`; GH200 must classify `Discrete`.** Check
  `GET /health` → `"unified"`. A wrong answer here is the one failure mode that
  can take the box down rather than merely waste capacity.
- **The clean fix is to stop guessing.** Detection prefers
  `cudaDeviceProp.integrated` when a CUDA-linked build can ask for it; this
  server does not link CUDA, so it falls back to the heuristic. Populating
  `cuda_integrated` on an arm64 CUDA build removes the guesswork entirely and is
  worth doing while you are on the hardware.
- Toolchain: expect to build for `aarch64-unknown-linux-gnu`. ~~libtorch's
  auto-fetch covers `Darwin/arm64` and `Linux/x86_64` only — `Linux/aarch64`
  has no URL~~ — **STALE as of 2026-08-16.** `tools/setup-libtorch.sh:78-99`
  has since grown a `Linux/aarch64` route: pytorch.org still publishes no C++
  dist for ARM Linux, so the script takes libtorch out of the pip wheel
  instead. The `cuXXX` wheel index and the `torch.libs`/`nvidia/*` runtime
  layout remain **unverified on a real GB10** — that part of the item stands.

## Platform notes for amd64 Linux

- **OpenCL needs the ICD loader dev symlink** `libOpenCL.so`, not just the
  runtime `libOpenCL.so.1` — `cl-sys` emits `-lOpenCL` and the linker resolves
  that only against the `.so`. Usually `apt install ocl-icd-opencl-dev`. A
  CUDA-only box has it at `$CUDA/targets/x86_64-linux/lib/libOpenCL.so`, off the
  default link path:
  `RUSTFLAGS="-L native=/usr/local/cuda/targets/x86_64-linux/lib" ARIA_OPENCL=1 ./ci.sh`
- **The mandatory cross-checks are not Mac-specific** — set up both venvs there
  too (`PREREQUISITES.md`). CI now prints a loud warning when the Qiskit
  cross-check did not run, so an absent venv will announce itself.
- **cgroups**: if the box runs the server in a container, confirm the governor
  budgets against the container limit and not host RAM. That path exists but has
  only been exercised by unit tests with injected probe values.

## `GateKind::Sx` / `Sxdg` — forward + adjoint DONE on CUDA (2026-08-13)

`√X` and `√X†` landed as first-class `GateKind` variants (see the doc comment on
`GateKind::Sx` for why they are not aliased to `U3`). CPU statevector, MPS,
Pauli, **Metal** and **OpenCL** were already done.

**RESOLVED on a Linux + NVIDIA box (RTX PRO 6000, nvcc 12.9):** the forward
dispatch (`apply_op`, `src/lib.rs`) and the adjoint (`apply_op_dagger`,
`src/adjoint.rs`) now apply the exact `gates::sx`/`sxdg` matrices via the generic
`apply_1q` — NOT `U3`. `diagonal_factor` correctly leaves them out. Commits
`e47fd15` (fix) + `9c8c01c` (tests: on-device forward phase-pin +
`adjoint_cuda_matches_cpu_12q_hea` exercising the dagger arms). `--features cuda`
compiles; 27/27 lib + integration green; clippy clean both feature states.

**Graph-capture path — DONE (`3f43d06`).** `forward_graph.rs` and
`backward_graph.rs` now list `Sx`/`Sxdg` (generic-1q plan +
`classify_op_kernel`/`forward_1q_matrix`, exact matrices; the dagger is derived
by conj-transpose, and non-parametric gates skip the derivative path).
`train_step_graph_matches_naive_backward` carries a √X/√X† layer, so the
graph-captured gradients are pinned against the naive adjoint. `√X` in a CUDA
*training* circuit now uses the graph fast-path instead of falling back.

Sites the compiler will point at:

| file | what it is | what to add |
|---|---|---|
| `src/lib.rs:~1247` | forward dispatch | `Sx`/`Sxdg` via the **generic 1q** path, exact matrices below |
| `src/lib.rs:~1301` | `diagonal_factor` classifier | **leave them out** — `sx` is NOT diagonal, so it must not be classified as a fusion factor |
| `src/adjoint.rs:~252` | adjoint (inverse) pass | `Sx → sxdg`, `Sxdg → sx` (`sx·sxdg = I`, verified 0.000e+00) |
| `src/forward_graph.rs:~313,~379` | graph-capture gate list + diagonal factors | add to the gate list; **not** to the diagonal factors |
| `src/backward_graph.rs:~1741,~1798` | same, backward | same, with the inverse mapping |

Exact matrices (identical to `omega_backend_statevector::gates::{sx,sxdg}`):

```
sx   = ½·[[1+i, 1−i], [1−i, 1+i]]
sxdg = ½·[[1−i, 1+i], [1+i, 1−i]]
```

**Do NOT route these through `apply_u3`.** `sx = e^{iπ/4}·U3(π/2, −π/2, π/2)`;
the global phase makes `|sx − U3| = 0.541` and `det(sx) = i` vs `det(U3) = 1`.
It is invisible in counts and expectations but wrong in any statevector
comparison — and a GPU statevector backend is exactly where `gpu_parity.rs`
would compare amplitudes. Metal and OpenCL both use their generic `apply_1q`
with the exact matrix for this reason.

Verification once implemented: `03_sqrt_x.qasm` in the N-way counts matrix
(`ARIA_NWAY=1`) exercises `sx`+`sxdg`, and `tests/sqrt_x_conventions.rs` pins
the matrices, the `sx·sx = X` identity and the Clifford tableau action.

## R8 — f64 CUDA statevector: landed, with follow-ups (2026-08-13)

The double-precision forward path landed (commits `9a04d9c` + `4aa11a2`):
precision-parametric kernels (all 24 `.cu` compile f32 **and** f64 — 48/48 on
sm_120 / nvcc 12.9), `src/f64_path.rs` (`StateF64`), validated amplitude-by-
amplitude vs Qiskit 2.5.1 at ≤1e-13. f32 stays the default and bit-identical.
Open, all low priority:

- **Adjoint / training / sampling / multi-observable stay f32.** By design (the
  f64 argument is forward *agreement*); revisit only when an f64 *training* loop
  is wanted. The kernels already compile in f64, so it is host buffers + graph
  capture that need the work.
- `Precision::bytes_per_amplitude` is unused pub API; the memory governor still
  prices device work at f32 (8 B/amp) — wire it in before f64 is bench-priced.
- A few `.cu` comments still say "f32" after the `real`/`real2` rewrite; and
  `StateF64::expectation_z` is a naive O(dim) sum (fine at tested n≈6, drifts
  past the 1e-13 bar near n≈20 — use pairwise/Kahan before claiming large n).

## libtorch / `tch` GPU backend on Linux (2026-08-13)

`--backend tch` now runs on the GPU (`TchBackend::cuda_or_cpu`, commits
`84bd5ba` + `012a8d0`); `ARIA_TCH_CUDA=1 tools/setup-libtorch.sh` fetches a CUDA
libtorch and the script force-retains `libtorch_cuda` in the link
(`--no-as-needed -ltorch_cuda -lc10_cuda`, else the linker drops it and
`is_available()` is false). Proven on RTX PRO 6000 Blackwell + in ubuntu 22/24/26
containers (see `INSTALL_LIBTORCH.md` §4).

Fixed this session: dist-swap re-provision on a CPU↔CUDA toggle + the aarch64
CUDA pip index (`457e1ab`); preset-`LIBTORCH` RUSTFLAGS retention + aarch64-aware
stale-dist advice (`93c8ad6`).

Open:

- **DGX Spark (aarch64)**: the `cuXXX` wheel index and the `torch.libs`/`nvidia/*`
  runtime-lib layout are unverified on a real GB10 — confirm there.
- **E6**: `tools/qec_cross_check/run.sh` bootstraps with plain `pip`, which cannot
  build `pymatching` on **aarch64** — needs the wheel index / build deps.

## Still open after the 2026-08-13 CUDA + Linux session

Everything the session set out to do (Sx/Sxdg on CUDA incl. graph paths, R8 f64
mode, GPU-enabled `tch`, the aarch64/GCC libtorch dance, `--all-targets` clippy)
is landed and verified. What remains, all lower priority:

1. **`Precision::bytes_per_amplitude` is unused** — the memory governor still
   prices device work at f32 (8 B/amp). Wire it in as part of §2 before f64 is
   bench-priced.
2. ~~**DGX Spark (aarch64+CUDA) hardware pass**~~ — **DONE 2026-08-16.** See
   `LINUX-CUDA-VERIFICATION.md`, which is now a results document rather than a
   checklist. `ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1 ./ci.sh` exit 0 on a real GB10
   (sm_121, CUDA 13.0.88, driver 580.159.03); f64 exact against Qiskit at
   1.388e-16; Qiskit differential 60/60 at 4.441e-16. **cudarc 0.19.4 builds
   against CUDA 13 on aarch64** — this file's "nvcc 12.9" assumption is not a
   requirement. Two blind `Outcome` casts had to be fixed first (`fc1a549`).
3. **E6** — aarch64 `pymatching` bootstrap in the QEC cross-check. **Still
   open, now measured rather than assumed:** there is **no aarch64 wheel** on
   PyPI (`pip install --only-binary=:all: pymatching` → "no matching
   distribution") and the cmake source build **fails** on a GB10. So the
   QEC cross-check cannot run on arm64 at all today.

   **Re-confirmed 2026-08-18** on this DGX: `tools/qec_cross_check/run.sh` still
   dies at `cmake --build . --target _cpp_pymatching`, exit 1. Nothing has
   changed upstream.

   **This is no longer treated as a gap in THIS box's coverage.** Decision of
   2026-08-18: the QEC cross-check is mandatory *on the machine that can run
   it* — macOS today — and platform-owned rather than expected everywhere. See
   the policy block in `OPTIONAL_TESTS.md` and `PLATFORM-OWNERSHIP.md`. E6 stays
   open as a portability item, but it no longer blocks anything here, and a DGX
   run skipping this stage is the expected outcome rather than a lapse.

### §3 topology on a real GB10 — the branch never fires

**`nvidia-smi --query-gpu=memory.total` returns `[N/A]` on a GB10.** So
`parse_nvidia_smi` drops the device row, `probe.devices` comes back empty, and
`classify` returns `HostOnly` — rule 5, the 25%-tolerance branch that IS the
GB10 signature and that this file says to check first, is **unreachable on the
hardware it was written for**. Its unit test feeds synthetic capacities this
machine never produces.

It fails safe (one ~128 GB pool is numerically right for a GB10), but by luck.
**The obvious fix is unsafe and was deliberately withdrawn:** `Unified` means
no per-device ceiling at all (`worker.rs:762-765`, `worker.rs:781-784`), so
routing unknown capacity there would turn a conservative under-count into an
unbounded device over-commit on discrete amd64 boxes — invisible on GB10,
firing on an H100. Needs an amd64 box in the loop; see
`PLAN-LINUX-CUDA-GB10.md` §4.

This also promotes *"populating `cuda_integrated` is the clean fix"* from
nice-to-have to the only reliable answer, since on GB10 the heuristic's input
does not exist. Measured here: `cudaDeviceProp.integrated = 1` and
`totalGlobalMem = 130662936576`, i.e. the driver reports both facts
`nvidia-smi` will not. **Note the ordering trap:** rule 2
(`devices.is_empty()` → `HostOnly`) runs *before* rule 3 (`cuda_integrated`),
so populating the flag alone changes nothing — the `cuda` build must enumerate
devices itself.

### §2/§3 follow-up — classify on BOX TYPE, not a capacity tolerance (TODO, no code)

The 25% `SAME_POOL_TOLERANCE` heuristic cannot work on the hardware it was
written for: **`nvidia-smi --query-gpu=memory.total` returns `[N/A]` on a
GB10**, so the device row is dropped and the machine reads `HostOnly`. The
GB10 signature branch is unreachable on a GB10.

**Do not "fix" it by routing unknown capacity to `Unified`.** That was designed
and withdrawn in round 1: `Unified` means there is **no per-device ceiling at
all** (`worker.rs:762-765` returns the unified pool for ANY target;
`worker.rs:781-784` makes `has_pool_for(Device(i))` unconditionally true). On an
8xH100 box where one row reports `[N/A]`, today's behaviour drops that row and
yields 7 real pools — under-provisioned but safe — while the "fix" yields ONE
2 TB pool admitting every GPU job. It converts a conservative under-count into
an unbounded device over-commit, invisible on a GB10 and fatal on amd64.

**The rule should classify on what the box IS, not on how its capacities
compare.** Signals available from a CUDA-linked build, in preference order:

1. `cudaDeviceProp.integrated` — canonical. **Measured on this GB10: `1`.**
2. `pageableMemoryAccessUsesHostPageTables` — **`1` here**; distinguishes a
   truly shared pool from merely coherent (GH200 is coherent with DISTINCT
   LPDDR5X + HBM3 capacities and must classify **Discrete**).
3. `totalGlobalMem` — **130662936576 (121.69 GiB) here**, i.e. the driver
   reports the capacity `nvidia-smi` will not. Enumerating devices via
   `cudaGetDeviceCount` + properties removes the `nvidia-smi` text parse
   entirely.

**Ordering trap, which sinks the obvious implementation:** rule 2
(`devices.is_empty()` -> `HostOnly`, `topology.rs:169`) runs BEFORE rule 3
(`cuda_integrated`, `topology.rs:179`). Populating the flag alone therefore
never reaches rule 3, because on a GB10 `probe.devices` is empty. The `cuda`
build must build `probe.devices` itself and the `integrated` check must move
above the `is_empty()` guard.

**A pre-existing misclassification to fix at the same time, found in passing and
NOT introduced by any of this:** an RTX 6000 Pro (96 GB) in a 128 GB host hits
rule 5 exactly — `|96 - 128| = 32 <= 128 x 0.25 = 32` — and classifies
**Unified today**, on a discrete box. That is the amd64 hardware the standing
constraint names.

Also open: `classified.reason` is **discarded** at `worker.rs:749` and
`health_snapshot` has no field for it, so `topology.rs:72`'s claim that the
reason is "surfaced in `/health`" is already false. Any new `TopologyReason`
variant changes nothing observable until `worker.rs` carries it.

**Needs an amd64 box with a discrete GPU to land.** Fixture tests should be
built from CAPTURED real `nvidia-smi` output and `/proc/meminfo` from such a
machine, not from round numbers chosen to make the assertion pass.

## PARKED BY DECISION — not blocked, not forgotten, not to be re-raised

**Owner decision, 2026-08-17.** These are understood, measured where
measurable, and deliberately NOT being worked. They are listed together here
because they were previously recorded in three different files under three
different labels, which is how parked work turns into forgotten work. Do not
re-open them as "caveats" on an unrelated run; if one is to be revived, that is
a fresh decision.

| item | where the reasoning lives | why parked |
|---|---|---|
| **PauliProp device residency** (the `BranchHook` contract) | `BranchHook` doc in `omega-backend-pauliprop/src/sim.rs`, plus the DEFERRED section below | The ONLY remaining fix for the GPU PauliProp verdict — the loss is structural (~14 device allocations + a full `synchronize()` per gate), not the kernel. Naive batching is WRONG: `freq = min` interacts with `max_freq`, so a merge must happen device-side. |
| **PauliSum container redesign** (sorted-vec merge vs `HashMap`) | `PLAN-PAULISUM-MAP.md` | The out-of-support skip REMOVED the work this would only MOVE. Costs a 24-site refactor of a `pub` field and changes coefficient summation order. Still viable in principle — the map is never queried by key anywhere in the workspace — but must be re-judged against the post-skip baseline, not the numbers that originally motivated it. |
| **Topology classification by box type** | `### §2/§3 follow-up` below | TODO, no code, by owner decision. Cannot be validated here anyway: the failure mode is invisible on a GB10 and fatal on a discrete card. |

Everything in this table is CI-green as it stands. Parking costs no
correctness — each is a performance or classification refinement, not a defect.

## Deferred — needs hardware this box is not

**Not open questions, not blockers: parked until the right machine exists.**
Recorded once here so they stop being re-raised as caveats on every run.

### Needs an amd64 / x86_64 box (discrete GPU preferred: RTX 6000 Pro, H100)

- **Topology by box type** (§ below). Cannot land here at all — the failure mode
  is invisible on a GB10 and fatal on a discrete card.
- **The RTX 6000 Pro rule-5 misclassification** — 96 GB in a 128 GB host hits
  the 25% tolerance exactly and reads Unified today.
- **Every perf number in `BACKEND-CROSSOVER.md`** is GB10-only. The statevector
  crossover, the `RAYON_NUM_THREADS=8` ceiling and the PauliProp verdict all
  assume unified memory; on a discrete card every host↔device hop costs real
  time, and the PauliProp verdict could plausibly **invert**.
- **The two-kernel phase split** (`apply_quad_phase1`) is measured redundant
  here and kept because register allocation is arch-specific. One measurement
  there retires it.
- **The CCX/CSwap exact-kernel A/B**, once the switch exists.

### Needs macOS (or any box that can build the QEC venv)

- **The QEC cross-check runs dark on aarch64 Linux and stays that way.**
  `pymatching` has no aarch64 wheel and its cmake source build fails on a GB10 —
  measured, not assumed. `ci.sh` still reports it as a stage that did not run,
  which is correct and should stay; it is simply not actionable here.
- **Metal crossover rows** for `BACKEND-CROSSOVER.md` — unified memory like the
  GB10 but no NVRTC step, so the fixed cost is likely lower and the crossover
  earlier.

### Needs an OpenCL dev package

- OpenCL cannot build on this box (runtime ICD only — no `libOpenCL.so`, no
  `CL/` headers). Its crossover row is unmeasured for that reason alone.

### DEFERRED — PauliProp device residency (the `BranchHook` contract)

Dropped deliberately, not forgotten. Full rationale lives on `BranchHook` in
`omega-backend-pauliprop/src/sim.rs`, where the next person will actually look.
Summary:

- The hook takes a **host** `&mut PauliSum` once per **rotation gate**, so every
  gate pays a full round trip. That is the entire reason the GPU arm is
  0.51x-0.79x of the CPU after the packed-key, buffer-reuse, pre-sizing and
  out-of-support work (re-measured 2026-08-17; this line previously read
  0.33x-0.81x).
- The cheap version — batch a gate's `k` rotations, merge once — is **WRONG**,
  not approximate: the merge keeps `freq = min`, and a later `max_freq` check
  reads it, so an unmerged copy carrying the larger `freq` gets truncated where
  the merged term survives. That changes `dropped_mass`, a certified bound.
- So it needs a **device-side merge** preserving `freq = min` and the bound
  exactly — real GPU engineering against the invariant this project most cares
  about.
- **Probably not worth it as things stand:** the merge is 86-89% of the branch
  step and the CPU does the same operation, so a device merge must beat a host
  `HashMap` at the dominant cost, for a path `BACKEND-CROSSOVER.md` already
  advises against on this hardware.

**Revisit when** an amd64 box with a discrete GPU exists (per-gate transfers
cost real time there and the verdict could invert), or a workload appears whose
sums are large enough that the kernel stops being noise.

**Better use of the same effort:** speed up the CPU merge, which is the path
that runs. Not the hash function — that was measured and rejected — but the map
(open addressing over the packed words, or a sorted-vec merge).

### DONE (2026-08-16, `cd5c403`) — exact CCX/CSwap kernel, behind an opt switch

**Shipped** as `--multi-control exact` (default stays `decompose`). Measured
**30.6x** faster than the 15-gate path and closer to the CPU f64 reference;
gated by `tests/multi_control_modes.rs` and by `ccx`/`cswap` in the Qiskit
corpus. Original write-up kept below.


CUDA (and Metal) apply `CCX` as a **15-gate Nielsen-Chuang decomposition**
(`lib.rs`), each gate rounding in f32, with `T`/`Tdg` carrying an irrational
`e^{±iπ/4}`. `CSwap` = `CX · CCX · CX` inherits all of it. But CCX is a
*permutation* — it swaps two amplitudes of an octet — so a dedicated subspace
kernel is both faster (one pass, not 15) and exact. The CPU backend already
does this (`2ed1eea`).

**Ship it as a switch, in the CLI *and* the job interface**, defaulting to the
decomposition:

- it changes the numbers (removing 14 gates of f32 rounding), so a caller with
  recorded results needs to be able to ask for the old path;
- the decomposition is identical to Metal's *on purpose*, so the two GPUs agree
  bit-for-bit; making CUDA exact broke that until Metal followed — **Metal has
  now followed** (`apply_octet_swap.metal` + `with_multi_control`), so the two
  agree at either setting provided both are set the same way;
- the speed claim is an expectation, not a measurement — the switch is what
  makes the A/B possible.

Correctness gate: **against Qiskit**, not against the decomposition it
replaces, which shares its conventions and would agree with a shared mistake.
See `GATE-EXACTNESS.md` §2.3.

### DONE (2026-08-16, `84f77b6`) — PauliProp has no Qiskit cross-check

**Shipped**: `omega-xcheck --bin pauliprop_xcheck` +
`tools/qiskit_xcheck/compare_pauliprop.py`, run by the `ARIA_QISKIT_XCHECK=1`
stage. This is what unblocked the packed-key, pre-sizing and out-of-support
work that followed. Original write-up kept below.


`omega-xcheck` drives `StatevectorBackend` only. PauliProp's correctness rests
on our own CPU statevector, the ppvm same-algorithm anchor, and GPU-vs-CPU
parity — all internal or same-family. Two implementations sharing a convention
agree on a shared mistake; this project has already shipped a defect that every
internal agreement gate missed.

Needed before any engine refactor (packed `PauliKey`, branch batching, device
residency), because those touch every term.

### Performance, measured 2026-08-16 (was never measured before)

- **Statevector 2q gates: CX 1.73×, SWAP 1.74×, CZ 2.91×, CRz 1.74×** — they
  were dispatching a dense 4×4 matvec to apply a permutation or a diagonal.
  `31d53f5`, `4325a50`, and the review fixes on top.
  - CX/SWAP/CZ are **bit-identical**; CRz is held to `4·f32::EPSILON·|input|`,
    because FMA contraction moves between the two paths and a bit-identity
    claim there was passing by luck of one toolkit and one arch.
  - **Index-gated.** Below `min(qa,qb) = 2` a two-slot gate is ~1.10× SLOWER
    than dense (sector granularity), so CX/SWAP/CRz fall back there. CZ is not
    gated — a one-slot diagonal wins everywhere.
  - **The CUDA-graph path is unchanged** and still dense, so the QML training
    loop sees none of this.
- **PauliProp GPU branch is SLOWER than the CPU at every width from 6 to 100
  qubits** (0.11×–0.79×, no crossover). It had only ever been gated on
  agreement, never timed. The cause is structural — `BranchHook` takes a host
  `&mut PauliSum` per rotation gate — not the kernel. `33a40b4`. The
  `PAULIPROP_GPU_MIN = 256` default is left alone deliberately: it looks far
  too low, but no amd64 measurement exists.
- `benches/cuda_bench.rs` swept 12–20 qubits, and 20q f32 is 8 MiB **inside**
  this device's 24 MiB L2 — it could not measure a memory-traffic change at
  all. A 26-qubit group was added.

Out of scope for that session (not CUDA/Linux, tracked in the handoff, not here):
aria-py CI stage (E1), missing wasm example circuits (E3), bloqade/perceval
Python (E5), aria-py resource bounding (R2b), the pest-grammar residual for
conditioned measure/reset.

---

## RESOLVED 2026-08-17 on a GB10 — `Send + Sync` audit (was BLIND)

**The three questions this section asked have been answered by compiling it.**

1. **Are the `unsafe impl`s redundant?** No. The note below predicted they might
   fail with `E0119` if `cudarc` already derived `Send + Sync`, and said to
   delete them in that case. They compile fine, and removing them fails to
   build: they are **load-bearing. Do not delete.**

2. **What actually blocks the automatic impl?** Exactly one thing, and it is the
   one the note flagged as the open question: `*mut CUgraph_st` and
   `*mut CUgraphExec_st`, raw handles inside `CudaGraph` → `TrainStepGraph` →
   the `Mutex`-guarded cache. `Arc<CudaContext>`, `Arc<CudaStream>` and
   `KernelLibrary` all qualify unaided, so the per-field reasoning about them
   was right but irrelevant.

3. **Is the assertion sound?** Yes, and it rests on one invariant. CUDA graph
   handles are scoped to a CONTEXT, not a thread, so moving them across threads
   is fine; what the driver forbids is launching one `CUgraphExec` concurrently
   with itself. Every use of a cached graph — the capture AND the
   `replay_with_y_labels` — runs while the guard is still held, so that cannot
   happen. **If a future change replays a cached graph outside the guard, this
   becomes UNSOUND and the compiler will not warn.**

Pinned by `crates/omega-backend-statevector-cuda/tests/send_sync.rs`: the bound
holds, `Box<dyn Backend + Send + Sync>` is declarable, and two threads sharing
one backend agree with the single-threaded reference. Note the test cannot
prove the Mutex discipline — an `unsafe impl` silences the compiler by
definition — so that invariant lives as a comment at the impl.

**The CONCURRENCY CAVEAT below is now MEASURED, and the answer is to leave it
alone.** All calls share one stream, and two host threads on one backend are
never faster at any width: 0.70x at 10 qubits, 0.99x at 24 (idle box,
best-of-3, values agreeing to 1e-5). Small widths are the worst rows, not the
best -- host per-call overhead dominates there, so per-thread streams could not
help even in the regime they were meant for. See `PLAN-CUDA-STREAMS.md` for the
table and the decision not to build it.

### (original, written blind on the Mac)

## For the Linux/CUDA agents — `Send + Sync` audit (added 2026-08-17, BLIND)

**One command decides this:**

```sh
cargo test -p omega-cli --features cuda --test backends_are_send_sync
```

### Why it exists

`aria-py` holds the GIL for the whole duration of every simulation —
`allow_threads` appears nowhere in `bindings/` or `crates/` — so two Python
threads calling `expectation` serialise completely. Releasing it needs the
closure passed to `Python::allow_threads` to be `Send`, which needs the captured
`&dyn Backend` to be `Sync`. `make_backend` in `aria-py` constructs `gpu:cuda`,
so **CUDA is implicated**.

Measured on the Mac: CPU (7 backends), Metal and OpenCL are all `Send + Sync`,
and `Box<dyn Backend + Send + Sync>` is declarable. CUDA could not be compiled
there, so it is the only unknown.

### What was written blind, and what to do with it

`unsafe impl Send`/`Sync for CudaStatevectorBackend` was added in
`crates/omega-backend-statevector-cuda/src/lib.rs`, with a per-field safety
argument in the comment above it. **It has never been compiled.** Three
outcomes:

1. **E0119 "conflicting implementations"** → `cudarc`'s `CudaContext` /
   `CudaStream` already derive `Send + Sync`, the struct qualifies
   automatically, and the impls are unnecessary. **Delete both** and the test
   should pass on its own.
2. **Compiles and the test passes** → keep them, but please *audit* rather than
   accept: an `unsafe impl` is a soundness assertion, and unlike a missing bound
   a wrong one is UB rather than a build error. The field needing the closest
   look is `train_step_graph_cache: Mutex<Option<(u64, TrainStepGraph)>>` —
   `Mutex<T>: Sync` requires `T: Send`, so `TrainStepGraph` (which holds
   `CudaSlice` pools and a `DeviceHandle`) is the real question.
3. **Compiles, test fails on another type** → that names the offending field.
   Prefer a narrow `unsafe impl` on *that* type with its own justification over
   a blanket one on the backend.

### The caveat that matters more than the compile result

**All calls share one stream.** Two host threads running circuits concurrently
serialise on the GPU even with the GIL released: each launch uses its own
`StateBuffer` so results stay correct, but the work does not overlap. So
releasing the GIL is necessary for the CPU backends and very likely **not
sufficient** for this one — a per-call or per-thread stream is the actual fix
and is separate work.

Please record the measured two-thread wall clock either way. If CUDA shows no
speedup, that is a finding, not a failure, and it belongs in the ledger.
