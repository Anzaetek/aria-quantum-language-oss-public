<!-- SPDX-License-Identifier: Apache-2.0 -->
# The Linux / CUDA checklist — RUN 2026-08-16 on a DGX Spark GB10

**This file was a list of unverified work. It has now been executed.** The
original checklist is preserved below each item so the prediction can be read
against the result.

**The box was not the one this file expected.** It anticipated a Linux / RTX
6000 Pro machine — x86_64, nvcc 12.9. What ran it was a **DGX Spark GB10**:
`aarch64`, sm_121, 48 SMs, 24 MiB L2, 121.7 GiB unified, driver 580.159.03,
**CUDA 13.0.88**. That is the target `CUDA_TODO.md` §3 calls the higher-risk
one and had never seen this code.

```
integrated = 1        totalGlobalMem = 130662936576 (121.69 GiB)
pageableMemoryAccessUsesHostPageTables = 1     sm_121   48 SMs   L2 = 24 MiB
```

---

> ## AGENTS ARRIVING AFTER 2026-08-17: THE CUDA SURFACE HAS GROWN
>
> This file records what was verified **on 2026-08-16**. Work landed after it
> that is **not** covered by anything below, and part of it is behind a Cargo
> feature or an opt-in flag — the exact place this project keeps losing code.
> Before you conclude "CUDA is verified", read **`CUDA_TODO.md` §0**, which
> tabulates each new surface and how to exercise it. In short:
>
> * **`omega-server --features cuda`** — the HTTP daemon had **no CUDA
>   dependency at all**, and no `PauliProp` backend variant, so that backend was
>   unreachable over HTTP by any spelling. Both are now wired, and the
>   `ARIA_CUDA=1` stage builds and clippies the feature. `cargo test --workspace`
>   does **not** — it builds `omega-server` with default features.
> * **`--multi-control exact`** — an exact CCX/CSwap octet permutation, 30.6x
>   faster than the 15-gate decomposition and closer to the CPU f64 reference.
>   **Opt-in**, so the default path is unchanged and a default-config test will
>   not touch it.
> * **New quad/octet kernels** for CX/SWAP/CZ/CRz/CY/CCX/CSwap, gated by
>   bit-identity tests against the dense path.
> * **PauliProp now has an independent Qiskit cross-check.** It had none.
>
> Two live traps on this hardware, both aarch64: `tch` needs
> `CARGO_BUILD_TARGET` or it SIGSEGVs *rustc*, and `omega-run --device cuda`
> does **not** reach the MPS SVD hook (only `aria-runtime` wires it), so a
> CLI "GPU MPS" measurement is the CPU twice. Both are written up in
> `CUDA_TODO.md` §0.

---

## 1. The CUDA `Outcome` edits — RESOLVED, and there were more than five

**Predicted:** five blind edits, *"not one line has ever been through a
compiler"*, and *"three of the five are test-side, and test-side is where a
blind edit is most likely to be wrong."*

**Result: the failure class was called exactly; the count was wrong.** There
are **seven** `Outcome` sites in the file, and the two this document does *not*
list are the two that failed to compile:

```
error[E0605]: non-primitive cast: `Outcome` as `usize`
  src/lib.rs:1971   f[(k as usize) & 3] = v as f64 / SHOTS as f64;
  src/lib.rs:2043   out[(k as usize) & 3] += v;
```

The five documented edits compile clean, so they were right as written. Fixed
in `fc1a549`. Every feature-gated crate was then swept **by hand** for the same
pattern rather than waiting for a compiler `cfg` had switched off — `as
usize`/`as u64` on a key, `for &k in`, `{k}` in a format string,
`from_str_radix`. No further sites. (The count has gone 5 → 6 → 8 across three
passes, each time because the compiler was never asked.)

**STILL OPEN — this section is NOT fully closed.** The "width note that
deserves a second opinion" about `lib.rs:858` — whether the collapse arm should
key on `circuit.num_qubits` or the creg width — is a **semantic** question and
a compiler cannot answer it. The CPU has
`omega-cli/tests/collapse_counts_use_the_creg_width.rs`; **CUDA has no
equivalent**, and `ci.sh`'s CUDA stage never exercises `MidCircuitMode::Collapse`.
Until that test exists, this item is *compiled*, not *answered*.

## 2. CUDA f64 — RESOLVED, and it is exact

**Predicted:** `f64_path.rs` is *"the least exercised code in the crate"*.

**Result, on sm_121:**

| gate | result |
|---|---|
| `precision_compile` — every `.cu` × {f32, f64} via NVRTC | pass (25 kernels) |
| `f64_agreement` | 3/3 |
| `f64_vs_qiskit` — amplitude by amplitude, n=6, 54 gates | worst **1.388e-16** |
| `f64_vs_qiskit` — ⟨Z_q⟩ over 5 wires | worst **2.220e-16** |

**A false green was found and closed on the way.** `f64_vs_qiskit` reports `ok`
when `.venv-qiskit` is absent — a loud skip that still counts as a pass — and
the venv did not exist on this box, so the strongest f64 gate was "passing" by
not running. The venv was built (qiskit 2.5.2, aer 0.17.2, numpy 2.5.2, all
aarch64 wheels) and the numbers above are from a real run.

`gpu_cuda_agrees_with_sim_on_qft` passes. **No movement against S1**, as its
bit-identity claim requires.

## 3. `RUST_TEST_THREADS` — CLOSED, but the diagnosis in this file is wrong

**Predicted cause:** *"`CudaStatevectorBackend` is neither `Send` nor `Sync`
by construction, so this is a property of the type."*

**That reasoning does not survive contact with the code.** `!Send`/`!Sync` is a
compile-time property that *prevents* cross-thread sharing; it cannot itself
segfault a harness where each `#[test]` builds its own backend. And
`forward_graph.rs:167-172` already captures with
`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` precisely so other threads' default-stream
work is not swept up — the named mechanism is both wrong and already mitigated.

What *is* true: `ForwardGraph::capture` is reachable only from `#[cfg(test)]`
code, so the interaction is test-only and serialising is a legitimate fix for
the symptom. `ci.sh` now sets `RUST_TEST_THREADS=1` **stage-scoped** (exporting
it process-wide would serialise every CPU stage and could mask an unrelated
race). **The cause remains unidentified** and the comment in `ci.sh` says so.

**MEASURED on this box 2026-08-16 — the flake is real and the fix holds:**

| configuration | runs | failures |
|---|---|---|
| default parallelism | 12 | **1** |
| `RUST_TEST_THREADS=1` | **18** | **0** |

The failing run took down `train_step_graph_matches_naive_backward` and
`reset_channel_matches_aer_ground_truth` together — and the first of those is a
`ForwardGraph::capture` test, which is precisely the interaction this file
names. So the *symptom* is confirmed even though the *cause* is not.

18 serialised runs is the n ≥ 17 bar this file's own "8 consecutive runs"
proposal fell short of.

Also worth recording: this file proposed *"8 consecutive runs"* as the bar. At
the documented 1-in-4 flake rate, `P(8 clean | nothing fixed) = 0.75⁸ = 0.100` —
a 10% chance of declaring victory having changed nothing. **n ≥ 11** for 95%,
**n ≥ 17** for 99%.

## 4. The unswept feature sets

- **`cuda` in `bindings/aria-py`** — still unswept. Separate workspace, own
  lockfile.
- **`tch`** — `CUDA_TODO.md`'s claim that *"`Linux/aarch64` has no URL"* is
  **stale**: `tools/setup-libtorch.sh:78-99` has since grown a pip-wheel route.

## 5. The many-core x86 item — NOT closed, and cannot be from here

This box is 20 **arm64** cores. That is not the many-core x86 machine the item
asks for. Claiming it would be the exact sin this repository keeps recording.

---

## What ran, and what did not

`ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1 ./ci.sh` → **exit 0, all stages that ran
passed.**

- CUDA stage: statevector, MPS (`gesvdj`), pauliprop branch, RBS forward +
  gradient, Reset channel ≡ Aer — all green.
- **Qiskit differential cross-check: 60 circuits agree, 0 disagree, worst
  |Δp| = 4.441e-16.** Feedforward: 8 agree, worst TVD 0.0147. This mandatory
  gate had never run on this machine.

**Nine stages still did not run**, and one is mandatory:

- **QEC cross-check (MANDATORY) — blocked, now with evidence rather than a
  claim.** `pymatching` has **no aarch64 wheel** on PyPI
  (`--only-binary=:all:` finds no distribution) and its cmake source build
  **fails** on this box. This is request **E6**, confirmed on the hardware.
- Metal (not macOS), OpenCL (out of scope by instruction; also unbuildable here
  — only `ocl-icd-libopencl1` is installed, so there is `libOpenCL.so.1` but no
  `libOpenCL.so` dev symlink and no `CL/` headers), Lean, and the aria-py /
  bridge / N-way / CV venvs.

## Not on the original checklist, found while here

**The GB10 topology branch cannot fire on a GB10.** `nvidia-smi
--query-gpu=memory.total` returns **`[N/A]`** on this device, so
`parse_nvidia_smi` (`topology.rs:298`) drops the row, `probe.devices` is empty,
and `classify` exits at rule 2 → `HostOnly`. Rule 5 — the 25%-tolerance branch
that *is* the GB10 signature — is unreachable on the hardware it was written
for, and its unit test feeds synthetic capacities this machine never emits.

It **fails safe** here (one ~128 GB pool is the right answer for a GB10), which
is luck rather than design. **The obvious fix is not safe** and was withdrawn:
`Unified` means *no per-device ceiling at all* (`worker.rs:762-765`,
`worker.rs:781-784`), so routing unknown capacity to `Unified` would turn a
conservative under-count into an unbounded device over-commit on discrete
amd64 boxes. See `PLAN-LINUX-CUDA-GB10.md` §4.

Related, pre-existing, and **not introduced by any of this**: an RTX 6000 Pro
(96 GB) in a 128 GB host hits rule 5 exactly (`|96−128| = 32 ≤ 128×0.25`) and
classifies **Unified today** on a discrete box.
