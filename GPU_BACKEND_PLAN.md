# GPU Backends + PauliPropagation.jl Parity — Work Plan

> **Status (2026-07-05): COMPLETE on this CUDA box.** Phases 0–5 landed and CI-green
> (default + `ARIA_CUDA=1`). StateVector, MPS (cuSOLVER `gesvdj`), and PauliProp
> (NVRTC branch kernel) GPU paths are **wired into `--backend` and numerically
> gated vs CPU**; PauliPropagation.jl `max_freq`/`U1`/CLI knobs added.
>
> **Status (2026-07-06): Metal arms COMPLETE on Apple M5 Max.** All three GPU
> paths are now wired and numerically gated vs CPU under `ARIA_METAL=1`:
> StateVector (f32, ≤1e-6), MPS **two-site θ-contraction** (f32; SVD stays on
> CPU — Apple has no native f64, so on-GPU Jacobi SVD is the one deferred piece),
> and PauliProp **branch** (new `omega-backend-pauliprop-metal` crate: integer
> symplectic work on the GPU, f64 coefficients on the CPU → **exact**, ≤1e-9).
> An `ARIA_METAL=1 ./ci.sh` stage mirrors `ARIA_CUDA=1`. Only M4 re-verification
> and the deferred on-GPU SVD remain (see below).

> Active dev repo: **`aria-quantum-language-oss-public`** (per the README banner,
> all future dev lands here). Rules from `CLAUDE.md` apply: local CI is the
> single source of truth (`./ci.sh`), **commit often, never push**, every
> acceptance is **numeric** against a golden ± tolerance.
>
> Build/test box for the GPU work: **this Linux machine** — NVIDIA RTX PRO 6000
> Blackwell (sm_120, 98 GB), CUDA 12.9. **Metal arms build but are test-deferred
> to the mac box** (no cross-vendor test here). CUDA is *optional*: every GPU
> path falls back to the pure-Rust CPU backend when the feature is off or the
> device is unavailable at runtime.

## Goal

1. Land the 5 uncommitted consumer-bug fixes (`fixes/` A1–A5), CI-verified.
2. Close the headline gaps vs **PauliPropagation.jl** (non-Clifford truncated
   Pauli propagation) — it is already an official Aria `--backend pauliprop`;
   make it *feature-comparable*.
3. Ship **GPU-accelerated, backend-selector-WIRED** versions of **StateVector,
   MPS, and PauliProp** for **CUDA (tested here) + Metal (deferred)**.
4. Docs updated, including **omega-functions** feature provenance.

## Current state (verified 2026-07-05)

| Backend | CPU | GPU crate present | GPU wired into `--backend`? |
|---|---|---|---|
| StateVector | `omega-backend-statevector` | `-statevector-{cuda,metal,opencl}` | **Yes** via `make_gpu()` (feature cascade) — needs RTX6000 verification + CI gate |
| MPS | `omega-backend-mps` | `-mps-{cuda,metal}` (cuSOLVER `gesvdj` SVD accel) | **No** — crate exists but nothing calls it |
| PauliProp | `omega-backend-pauliprop` | *(none)* | N/A — no GPU path exists yet |

- `BackendSel` (`crates/aria-runtime/src/run.rs:29`) = `Sim | Mps | Gpu | Tch |
  PauliProp`. `Gpu` dispatches to statevector-GPU only.
- MPS-GPU (`omega-backend-mps-cuda`) is a **truncated-SVD accelerator** for bond
  compression (cuSOLVER `cusolverDnZgesvdj`, native f64 → 1e-10 reachable), not a
  standalone Backend. It must be wired as an SVD provider *inside* the MPS
  backend, selected by a feature.
- PauliProp core lives canonically in **`../omega-functions/crates/omega-backend-pauliprop`**;
  the `-public` copy is byte-similar. Data layout: `PauliKey{ x,z: Vec<bool> }` +
  `HashMap<PauliKey, Complex64>`. `branch()` (`sim.rs:223`) is a generic
  `exp(-iθ/2·R)` primitive — new gates only need new `Gen` builders.
- Gaps vs PauliPropagation.jl: no `max_freq` truncation; limited gate set
  (Clifford + Rx/Ry/Rz/T/Tdg/CRz); `HashMap` iteration is nondeterministic
  (blocks stable GPU batching); no differentiable/parametric propagation.

## Phases

### Phase 0 — Verify & land the fixes WIP (A1–A5)  ✅ in progress
- Baseline `./ci.sh` green (fmt, clippy -D, builds, all tests, 32 harnesses,
  socket transport).
- Re-run each acceptance check from `fixes/FIXED.md`:
  - A1 `RY(1e-05)` → |1⟩ amp 5e-6 (±1e-6); `1e999` rejected.
  - A2 partial `measure→creg` → creg-width counts (sim + mps + remote).
  - A3 MPS `--shots` freq within sampling error of `--statevector` (≤4σ).
  - A4 `--backend pauliprop` Bell ⟨Z0 Z1⟩ = 1; sampling rejected with error.
  - A5 release-build guidance present in TESTING.md.
- Commit in logical chunks (Anzaetek Team). **Acceptance:** CI green + all
  five checks pass numerically.

### Phase 1 — PauliPropagation.jl parity (CPU, algorithmic)
Edit both the `-public` crate and the `../omega-functions` source (keep them
in sync; omega-functions is canonical).
1. **`max_freq` truncation** — cap the number of non-Clifford *splits* a term
   has survived (frequency). Track a per-term split counter; drop / don't-branch
   beyond `max_freq`; fold dropped coefficient L1 mass into `dropped_mass` so the
   error bound stays certified. Knob on `PauliPropBackend` + `expectation_with_budget`.
2. **Gate set** — add `Rzz/Rxx/Ryy`, `CY`, `Toffoli/CCX`, and generic single-
   qubit `U`/`Rphi` via new `Gen` builders routed through `branch()`; each new
   arm gets a numeric test vs the in-file statevector oracle.
3. **Determinism** — replace default-hashed `HashMap` iteration with a canonical
   ordering (sort symplectic keys, or `IndexMap`/`BTreeMap`) so results and any
   top-K truncation are reproducible and GPU-batchable. Assert bit-identical
   output across runs.
4. **CLI** — expose `--truncate C --max-weight W --max-freq F` on the Aria
   `run --backend pauliprop --expectation` path (currently only `quantum expect`
   has truncation knobs; Aria `pauliprop` is exact-only).
5. Docs: TESTING.md pauliprop section + note the omega-functions origin.
- **Acceptance:** new-gate expectations match statevector ≤1e-9; truncation
  error ≤ reported `dropped_mass` and monotone; deterministic across runs.
- *Deferred (out of scope unless time permits):* symbolic/differentiable
  Pauli-path propagation & surrogates — documented as a known gap.

### Phase 2 — StateVector GPU (CUDA verified here, Metal deferred)
1. Build `-statevector-cuda` for sm_120 (`--features cuda`); confirm NVRTC
   kernels compile on CUDA 12.9 / Blackwell; fix any arch issues.
2. Numeric differential CI gate: for a fixed gate list (H/RY/CX/T…), assert GPU
   statevector ≡ CPU statevector amplitudes ≤1e-6 (f32 kernels) across n∈{4,…,20}.
3. Confirm `aria run --backend gpu --features cuda` runs on the box; fall-back
   path preserved when device absent.
4. Add a CI stage (opt-in `ARIA_CUDA=1`, like the tch/lean opt-ins) so the
   default offline CI stays green.
- **Acceptance:** GPU≡CPU ≤1e-6; `--backend gpu` end-to-end numeric run passes.

### Phase 3 — MPS GPU, WIRED (cuSOLVER gesvdj)
1. Give `omega-backend-mps` a pluggable SVD provider (feature `cuda` →
   `omega-backend-mps-cuda::gesvdj`, else CPU Jacobi). Resolve the current
   dependency direction (mps-cuda → mps) — likely invert to a trait/hook the
   mps backend calls, or have aria-runtime construct an `MpsBackend` configured
   with the CUDA SVD.
2. **Wire into the selector**: `--backend gpu` with `--features cuda` (and a way
   to pick MPS-on-GPU explicitly, e.g. `--backend mps` honoring the cuda SVD
   feature, or a `mps-gpu` alias) so GPU MPS is reachable from the CLI.
3. Numeric parity: GPU-SVD bond compression ≡ CPU Jacobi forward results
   ≤1e-10 on the existing MPS regression circuits (incl. the new
   `sampling_matches_statevector_probabilities`).
- **Acceptance:** `--backend <mps-gpu path> --features cuda` reachable + numeric
  parity ≤1e-10; CPU fallback intact.

### Phase 4 — PauliProp GPU, WIRED
1. New crate `omega-backend-pauliprop-cuda` (feature `cuda`, cudarc/NVRTC),
   mirroring the mps-cuda pattern: SoA `u64`-word symplectic arrays (from the
   Phase-1 canonical ordering), a kernel that batches the O(terms×n)
   anticommute test + `branch`/`mul_raw` over all terms per gate, with
   coefficient-magnitude/weight/freq truncation on device. CPU fallback when
   feature off or device absent.
2. **Wire into the selector**: reachable via `--backend pauliprop --features
   cuda` (GPU path chosen at runtime when available; else CPU).
3. Numeric parity: GPU ≡ CPU pauliprop expectation ≤1e-9 on the full pauliprop
   test suite + the 40-qubit GHZ; dropped_mass identical.
- **Acceptance:** GPU pauliprop reachable from CLI + numeric parity ≤1e-9.

### Phase 5 — Docs + omega-functions feature docs + brutal review
1. TESTING.md: numeric, copy-pasteable GPU sections (CUDA opt-in), pauliprop
   truncation knobs, MPS-GPU parity check.
2. README/GRAMMAR: backend table incl. GPU + pauliprop; `--max-freq`.
3. **omega-functions**: document the features the integration copies pull in
   (pauliprop engine, any shared kernels); reference `../omega-functions` where
   the in-tree copy is thin.
4. Brutal subagent review of the entire diff (correctness, determinism, fallback
   safety, numeric gates); fix findings.
- **Acceptance:** CI green (incl. opt-in CUDA stage on this box); review clean.

## Metal / Mac — DONE on Apple M5 Max (2026-07-06)

All three Metal arms are wired into `--backend` and numerically gated vs CPU on
an Apple **M5 Max**, verified under `ARIA_METAL=1 ./ci.sh`:

- ✅ **StateVector Metal** — `aria run --backend gpu --features metal --statevector`
  ≡ `--backend sim` (`gpu_metal_agrees_with_sim_on_qft`, ≤1e-6, f32 kernels).
  Plus the full `omega-backend-statevector-metal` suite (adjoint/QML training
  parity), all green on this box.
- ✅ **MPS Metal θ-contraction** — the two-site contraction runs on the GPU
  (f32), the SVD stays on CPU. Wired via a new `Contract2qFn` hook on
  `omega-backend-mps` (mirrors the `SvdFlatFn` CUDA hook), engaged above the
  bond-dim threshold. `gpu_mps_metal_agrees_with_sim` ≡ exact CPU statevector on
  a 12q entangling brickwork (≤1e-3, f32). The **SVD-on-GPU half stays deferred**
  on Apple GPUs (no native f64 / ~100 µs dispatch — see
  `omega-backend-mps-cuda/src/lib.rs` header); re-open triggers unchanged.
- ✅ **PauliProp Metal branch** — new crate `omega-backend-pauliprop-metal`,
  sibling of `-cuda`. Because Apple has no native f64, it offloads only the
  **integer** symplectic work (anticommute parity + sign + child key `R·P`) to
  the GPU and keeps every coefficient on the CPU in f64 — so the result is
  **bit-for-bit the CPU branch**, not an f32 approximation. Wired via
  `with_branch_hook`; `gpu_pauliprop_metal_agrees_with_sim` + the crate's
  `gpu_branch_matches_cpu_{exact,with_max_freq}` gate value **and** dropped-mass
  ≤1e-9.
- ✅ **`ARIA_METAL=1` ci.sh stage** added, mirroring `ARIA_CUDA=1`.

Remaining (not blocking):
- **M4 re-verification** — this box is an M5 Max; the arms are target-gated and
  device-probed, so M4 should behave identically, but a second-device run is
  still open. Also confirm M4/M5 unified-memory sizing for wide runs.
- **On-GPU SVD for MPS** — deferred by Apple's f64/dispatch limits; re-open per
  the `omega-backend-mps-metal` crate header triggers (batched SVD via MLX /
  Accelerate, randomized SVD, or block Lanczos).

## Cross-cutting rules
- **Commit often**, small logical commits, author **Anzaetek Team
  <team@anzaetek.com>**, **never push**.
- Every GPU feature is **optional** and **falls back** — default `./ci.sh` must
  stay green with no GPU.
- Run a **brutal review subagent** after each major phase.
- Keep `-public` and `../omega-functions` pauliprop sources in sync.
