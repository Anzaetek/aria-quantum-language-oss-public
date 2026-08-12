<!-- SPDX-License-Identifier: Apache-2.0 -->
# Platform ownership — who is working on what

**Updated 2026-08-12.** Two agents work this repository on different hardware.
This file says which work is claimed by which, so the two do not collide on the
same files and so neither waits on hardware it does not have.

The rule is simple: **if you cannot RUN it, do not land it.** This repository's
recurring defect is a check that passes without exercising the thing it names,
and an untested platform patch is that defect with a compiler in front of it.

---

## macOS agent (Apple Silicon) — ACTIVE, holds these

Everything below is being worked on here **first**. The Linux agent should not
start on these; they will arrive on `main` and are likely to conflict.

| area | state |
|---|---|
| Part K — N-way counts + expectation matrices | **done**, `ARIA_NWAY=1` |
| Part H — QASM2 export carries or refuses classical guards | **done** |
| K9 — `cargo test --workspace` green; CI off the typed crate list | **done** |
| `GateKind::Sx` / `Sxdg` — CPU, MPS, Pauli, **Metal**, **OpenCL** | **done** |
| Aria language: `CRZ`, `RESET` spellable | **done** |
| `omega-parser` accepts `cp` / `cu1` (round-trip gap) | **done** |
| Aria emitter carries classical guards | **done** |
| Shared `runner_io` stdout protocol guard (all 5 bridge runners) | **done** |
| ppvm + Stim same-algorithm expectation anchors | **done** |
| aria-py bindings (opt-in cuda feature, Pauli propagation, reusable backends, accelerator getter) | **done** |
| tch DYLD inline fix (macOS SIP strips exported `DYLD_*`) | **done** |
| **Part L** — remoting resilience / WS dispatch | **scoped, in progress** |
| **#22** — `to_qasm3` classical control flow | **claimed** |
| **#7** — job lifecycle, durable batches, async | **claimed** |
| OPTICQASM export/import integrity (`PLAN-OPTICQASM-INTEGRITY.md`) | **O1–O3, O5 done**; O4, O6 open |

Verified here every run: Qiskit + QEC cross-checks, bridge arms
(perceval/tsim/ppvm), CV↔piquasso, N-way matrices, **Metal**, **OpenCL**,
Lean 4.

---

## Linux agent — these are YOURS, and only you can validate them

Each of these is blocked on hardware or an OS this Mac does not have. They are
deliberately **not** applied here, because applying a patch you cannot execute
is how an untested change acquires the appearance of review.

### 1. CUDA — **DONE 2026-08-13**, no longer owned here

`GateKind::Sx` / `Sxdg` landed on CUDA across the forward, adjoint and
graph-capture paths, on a Linux + NVIDIA box (RTX PRO 6000, nvcc 12.9). The
deliberate compile error this section used to describe is gone.

Reviewed from the macOS side without hardware, which is all that is possible
here and is stated as such: the matrices in `forward_graph.rs` and
`backward_graph.rs` are `½[[1±i, 1∓i],[1∓i, 1±i]]`, so `sx·sx` has diagonal
`[(1+i)²+(1−i)²]/4 = 0` and off-diagonal `2(1−i²)/4 = 1` — exactly `X`. The
documented trap was avoided: `diagonal_factor` classifies only Z/S/Sdg/T and
does **not** include Sx, which is correct because `sx` is not diagonal.

**This is a code read, not a run.** `ARIA_CUDA=1` has still never executed on
this machine, and the Linux agent's verification is the one that counts.

Remaining CUDA items are tracked in `CUDA_TODO.md` §"Still open" and stay
Linux-owned: the unused `Precision::bytes_per_amplitude` in the memory governor,
a DGX Spark (aarch64+CUDA) hardware pass, and E6 aarch64 `pymatching`.

### 2. tch on Linux — **DONE 2026-08-13**, no longer owned here

GPU-enabled `tch` and the aarch64/GCC libtorch provisioning both landed and were
verified on Linux. The macOS half (SIP strips exported `DYLD_*`, so it is passed
inline on the cargo invocation) remains fixed here.

### 3. Anything else needing an NVIDIA GPU or a Linux-only toolchain

`ARIA_CUDA=1` is the one CI stage that has never run here. Every green summary
this Mac produces ends with:

```
All CI stages that ran passed — but 1 stage(s) did NOT run:
   - CUDA backends — set ARIA_CUDA=1 on a CUDA box
   A green run is only as strong as what actually ran.
```

That line is the handoff.

---

## Coordination

* **Do not** apply Mac-side work on Linux or Linux-side work on macOS — both
  would be unverifiable where they land. (The `fixes/0011`–`0013` CUDA patches
  this bullet used to name are landed and gone.)
* `fixes/` is gitignored, so both agents see the same patch series but neither
  can commit it. Record what was applied in the commit message, as the commits
  referencing 0001/0002/0004/0005/0006/0007/0008/0010 do.
* When CUDA work lands, **delete its section here** and update `CUDA_TODO.md` —
  a stale ownership file is worse than none, because it claims coverage that has
  moved.
