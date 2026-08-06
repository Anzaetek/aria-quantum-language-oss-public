<!-- SPDX-License-Identifier: Apache-2.0 -->
# CUDA — work waiting for an amd64 Linux + NVIDIA box

Target platform: **x86_64 (amd64) Linux with an NVIDIA GPU.** Everything here is
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

Record actual numbers, not "passed":

- `./ci.sh` exit code, with the CUDA stage's `OK:` lines
- `/health` output showing the detected topology and per-pool capacities
- GPU model(s), driver, CUDA version, and `nvidia-smi --query-gpu=index,name,memory.total`
- `uname -m` (expected `x86_64`) — the topology heuristic treats aarch64
  differently, since that is where GB10/GH200 live

`OPTIONAL_TESTS.md` has the table to update; `FIXES_PLAN.md` A7b has the design
rationale if a decision looks wrong on real hardware.

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
