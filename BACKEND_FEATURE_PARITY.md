# GPU statevector backend feature parity

Which capabilities each GPU statevector backend supports. Four backends:
**Metal** (`omega-backend-statevector-metal`), **OpenCL**
(`omega-backend-statevector-opencl`), **CUDA-f32** (the default CUDA path in
`omega-backend-statevector-cuda`), and **CUDA-f64** (the R8 double-precision
path, `omega-backend-statevector-cuda/src/f64_path.rs`).

`SUPPORTED` / `PARTIAL` / `—` (unsupported). Line numbers drift; treat them as
starting points, not contracts.

## Capability matrix

| Capability | Metal | OpenCL | CUDA-f32 | CUDA-f64 |
|---|---|---|---|---|
| Forward execute | SUPPORTED | SUPPORTED | SUPPORTED | PARTIAL — raw 1q/2q only, no circuit driver |
| Shot sampling | SUPPORTED (on-device) | SUPPORTED (on-device) | SUPPORTED (on-device) | — |
| Expectation, single obs | SUPPORTED | SUPPORTED | SUPPORTED | PARTIAL — host-side, Z-only |
| Expectation, multi obs | SUPPORTED | SUPPORTED | SUPPORTED | — |
| Adjoint gradient / training | SUPPORTED | PARTIAL — `adjoint_gradient` only, no fused train-step | SUPPORTED (+ CUDA-graph capture/replay) | — |
| Reset channel | SUPPORTED | — | SUPPORTED (stochastic, per-shot trajectories) | — |
| Mid-circuit measure (Collapse) | PARTIAL | — | — | — |
| Precision | f32 | f32 | f32 | **f64** (≤1e-13 vs Qiskit) |

## Gate coverage

All full backends support: H, X, Y, Z, S, Sdg, T, Tdg, Rx, Ry, Rz, CX, CY, CZ,
Swap, Id, Barrier, U1, U2, U3.

| Gate | Metal | OpenCL | CUDA-f32 |
|---|---|---|---|
| Sx / Sxdg | SUPPORTED | SUPPORTED | SUPPORTED |
| CRz | SUPPORTED | SUPPORTED | SUPPORTED |
| CU3 | SUPPORTED | SUPPORTED | SUPPORTED |
| Rbs | SUPPORTED | **—** | SUPPORTED |
| CCX | SUPPORTED | **—** | SUPPORTED |
| CSwap | SUPPORTED | **—** | SUPPORTED |
| PhaseShifter / BeamSplitterRx / Custom | — | — | — |

CUDA-f64 does **no** `GateKind` dispatch: `StateF64` exposes only raw 1q/2q
unitaries (`apply_1q`, `apply_2q`), so a caller decomposes higher-level gates
itself.

## Notable gaps

1. **CUDA-f64 is forward-only** — no adjoint/training, no sampling, no
   multi-observable, host-side Z-expectation only. Deliberate (R8): the f64
   argument is about forward *agreement* with an independent reference; the
   adjoint and CUDA-graph training machinery stay f32 until an f64 *training*
   loop is actually wanted. It is the only path that reaches the project's
   1e-9 cross-check bar (f32 backends bottom out at ~1e-7).
2. **OpenCL is the narrowest full backend** — no Reset, no CCX/CSwap/Rbs, and no
   fused `expectation_multi_then_gradient` (plain `adjoint_gradient` only).
3. **Mid-circuit measurement (Collapse mode) is unsupported on every GPU
   backend** — OpenCL and CUDA explicitly refuse it and fall back to CPU; Metal
   has partial handling.
4. **Adjoint declines rather than errors on Reset circuits** (Metal, CUDA return
   `Ok(None)` so the caller can fall back); CUDA's *fused* train-step path
   hard-errors on Reset instead.
5. **All four full/forward backends compute in f32 except CUDA-f64.**

See also `GPU_BACKEND_PLAN.md`, `CUDA_TODO.md`, and `fixes/REQUEST-R8-cuda-f64.md`.
