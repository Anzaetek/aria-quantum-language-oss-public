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

## 4. Known-open CUDA defect to confirm

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
- Toolchain: expect to build for `aarch64-unknown-linux-gnu`. libtorch's
  auto-fetch in `tools/setup-libtorch.sh` covers `Darwin/arm64` and
  `Linux/x86_64` only — **`Linux/aarch64` has no URL**, so the tch stage will
  print its SKIP there until one is added.

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

**Still open (graph-capture path):** `forward_graph.rs` / `backward_graph.rs` do
NOT list `Sx`/`Sxdg`, so a circuit with `√X` in the CUDA-graph *training* path
returns `Unsupported` and falls back to the fused `apply_ops`/adjoint (correct
result, no graph fast-path). Wire `Sx`/`Sxdg` into `classify_op_kernel`
(`backward_graph.rs`) + the forward-graph gate list to close it. Not a
correctness gap — a performance one — so it was not bundled into the build fix.

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
containers (see `INSTALL_LIBTORCH.md` §4). Open:

- **Dist-swap idempotency**: the download-path check compares only the version
  *base* (`${have_ver%%+*}`), so toggling `ARIA_TCH_CUDA=1` on a box that already
  has the `2.7.0+cpu` dist SKIPS the re-download and silently stays CPU. Key the
  idempotency on the full local tag (`+cpu` vs `+cuXXX`) so a CPU↔CUDA switch
  re-provisions.
- A user-exported `LIBTORCH` bypasses `tch-env.sh`, losing the CUDA/`torch.libs`
  `RUSTFLAGS`; and a stale wrong-version aarch64 `$LIBTORCH` lands on the
  "download from pytorch.org" branch that has no aarch64 C++ dist.
- **DGX Spark (aarch64)**: the `cuXXX` wheel index and the `torch.libs`/`nvidia/*`
  runtime-lib layout are unverified on a real GB10 — confirm there.
- **E6**: `tools/qec_cross_check/run.sh` bootstraps with plain `pip`, which cannot
  build `pymatching` on **aarch64** — needs the wheel index / build deps.
