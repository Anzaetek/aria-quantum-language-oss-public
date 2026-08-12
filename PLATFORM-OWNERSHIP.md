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

Verified here every run: Qiskit + QEC cross-checks, bridge arms
(perceval/tsim/ppvm), CV↔piquasso, N-way matrices, **Metal**, **OpenCL**,
Lean 4.

---

## Linux agent — these are YOURS, and only you can validate them

Each of these is blocked on hardware or an OS this Mac does not have. They are
deliberately **not** applied here, because applying a patch you cannot execute
is how an untested change acquires the appearance of review.

### 1. CUDA — `fixes/0011`, `0012`, `0013`

`GateKind::Sx` / `Sxdg` landed on every backend **except CUDA**, on purpose.
`cargo build -p omega-backend-statevector-cuda --features cuda` will **fail**
with `non-exhaustive patterns: &GateKind::Sx and &GateKind::Sxdg not covered`.
That is the intended state — a compile error on the box that can test it beats
an implementation written blind.

`CUDA_TODO.md` names all six sites, the exact matrices, the inverse mapping,
and one trap: **`sx` is NOT diagonal**, so it must not join the
`diagonal_factor` fusion classifier alongside S/T.

The three queued patches are the f64 precision-parametric kernel work
(`REQUEST-R8-cuda-f64.md`). They have never been compiled here.

### 2. tch on Linux — `fixes/0003-tch-on-Linux-...`

Takes libtorch from the pip wheel and sets `LD_LIBRARY_PATH`. The macOS half of
this problem is fixed here (SIP strips exported `DYLD_*`, so it is passed
inline on the cargo invocation); the Linux half is untested on this machine.

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

* **Do not** apply the `fixes/0011`–`0013` CUDA patches on macOS, and do not
  apply the Mac-side work on Linux — both would be unverifiable where they land.
* `fixes/` is gitignored, so both agents see the same patch series but neither
  can commit it. Record what was applied in the commit message, as the commits
  referencing 0001/0002/0004/0005/0006/0007/0008/0010 do.
* When CUDA work lands, **delete its section here** and update `CUDA_TODO.md` —
  a stale ownership file is worse than none, because it claims coverage that has
  moved.
