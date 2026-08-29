<!-- SPDX-License-Identifier: Apache-2.0 -->
# When to move a workload from CPU to GPU

Recommended switch sizes per algorithm, **measured, not guessed**. Every number
below was taken on one machine and that machine is named; a row without a
measurement says so instead of carrying a plausible number.

**Measured on:** DGX Spark **GB10** — aarch64, sm_121, 48 SMs, 24 MiB L2,
121.7 GiB unified LPDDR5X, CUDA 13.0.88, driver 580.159.03, 20 CPU cores,
`--release`. CPU statevector is f64 and rayon-parallel; CUDA statevector is f32.

> **What was NOT controlled, stated because "measured, not guessed" oversells it
> otherwise.** Almost every wall-clock figure in this document was hand-timed on
> a SHARED box with **no recorded idle check, no load average, and no
> interleaving of the arms.** Three measurements earlier in the session that
> produced them were invalidated by background load before that discipline was
> adopted, and the sibling M5 machine nearly filed a 3.4x regression this week
> that turned out to be an unrelated browser process at 67% CPU.
>
> An environmental swing of that size would swamp most of the ratios here. Treat
> a figure below as an order-of-magnitude sizing aid — which is all a crossover
> table needs to be — and NOT as a number to detect a regression against. For
> that, use a bench that interleaves: `crates/omega-backend-pauliprop/benches/`
> and the seven other `benches/` directories.
>
> Counts are the exception and are worth more than they look: an insert count or
> a skip count is identical on an idle box and a loaded one. Where a claim here
> rests on one, it is pinned by a test rather than a table —
> `insert_count_is_flat_across_width.rs` is the model.

---

## The number that decides most cases: ~0.58 s of fixed GPU cost

```
4-qubit circuit, --device cuda :  0.585 s, 0.586 s, 0.570 s
4-qubit circuit, --device cpu  :  0.003 s
```

Backend construction NVRTC-compiles all 26 kernels, and that cost is paid once
per process regardless of circuit size. **Below the crossover you are not
measuring the GPU, you are measuring the compile.**

Two consequences worth stating before any table:

- A short-lived CLI invocation on a small circuit is *always* faster on CPU.
  The crossovers below are dominated by this constant, not by bandwidth.
- A long-lived process that reuses one backend across many circuits amortises
  it, and its crossover is much lower than the table says. If you are
  embedding the library rather than shelling out, measure your own.

## 1. Statevector — switch at **~24 qubits**, sooner for 2q-heavy circuits

`--device cpu` vs `--device cuda`, `--shots 1000`, wall clock including startup.

| circuit | 16q | 20q | 22q | 24q | 26q | 28q |
|---|---|---|---|---|---|---|
| **ghz** (n−1 CX, shallow) | 0.00× | 0.04× | 0.12× | 0.38× | **1.21×** | **1.50×** |
| **qft** (O(n²) CRz) | 0.02× | 0.20× | 0.60× | **1.56×** | **2.86×** | **3.67×** |
| **hea** (4 layers, 2q-heavy) | 0.02× | 0.18× | 0.49× | **1.28×** | **2.31×** | **3.02×** |

Ratio > 1 means the GPU wins.

**Recommendation:**

| circuit shape | switch at |
|---|---|
| gate-dense, O(n²) or layered (qft, hea, VQE ansätze) | **≥ 24 qubits** |
| shallow, O(n) gates (ghz, bell chains) | **≥ 26 qubits** |
| anything, in a long-lived process reusing the backend | lower — measure |

The split is not about the gates being different; it is that a shallow circuit
does too little work per amplitude to cover the 0.58 s, so it needs a bigger
state before it does.

## 2. PauliProp — **do not switch**, on this hardware, at any size measured

| qubits | 6 | 8 | 10 | 12 | 16 | 24 | 40 | 72 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| GPU vs CPU | 0.52× | 0.69× | 0.78× | 0.79× | 0.70× | 0.61× | 0.51× | 0.52× | 0.53× |
| CPU absolute | 9.6ms | 214ms | 533ms | 536ms | 86ms | 15.2ms | **2.6ms** | **3.0ms** | **3.0ms** |

**No crossover anywhere from 6 to 100 qubits**, including past the 64-qubit
symplectic word boundary where one might be expected. The GPU branch hook is
1.3×–1.9× *slower* than the CPU. (Re-measured 2026-08-17 after the
out-of-support skip; the previous edition of this row read 0.11× / 0.59× /
0.79× / 0.75× / 0.58× / 0.55× / 0.45× / 0.53× / 0.62×. The verdict did not
change, the numbers did — the skip removes a device ROUND-TRIP from the GPU arm
where it removes only a rebuild from the CPU arm, so it helps the GPU arm more.)

**Read the absolute row, not just the ratio — but read it carefully, because
this corpus varies depth and width TOGETHER.** The 40/72/100 columns are all 3
layers; the 10/12 columns are 6. So the fair comparison is *within* a fixed
depth:

> At 3 layers, the CPU cost is **flat across a 2.5× widening**: 2.6 / 3.0 /
> 3.0 ms at 40, 72 and 100 qubits, with the GPU branch count flat too (78 calls
> at all three widths).

That flatness is the point. Cost is bounded by the observable's LIGHT CONE, not
by the register, so widening a circuit whose observable is local adds no work.
The 10–12 qubit rows are expensive for a different reason — twice the depth, and
at that width the cone covers the whole register — so they are **not** evidence
that fewer qubits cost more. Depth is not held fixed between those rows and the
wide ones, and no conclusion should be drawn across them.

A consequence worth stating for anyone choosing a backend: **"how many qubits"
is the wrong question for PauliProp.** What costs is the non-Clifford depth
*inside the observable's light cone*. Width alone is close to free.

The cause is structural, not the kernel: `BranchHook` takes a **host**
`&mut PauliSum` and is called once per rotation gate, so each gate pays ~14
device allocations, 7 uploads, a full `synchronize()`, 7 downloads, and a host
merge that costs about what the CPU's whole branch loop costs.

`PAULIPROP_GPU_MIN` (default 256 terms) is the knob, and it is **deliberately
left at its default**: raising it would be right for this box and is unmeasured
on amd64, where the balance may differ. If you are on a GB10 and want the CPU
path unconditionally, set `PAULIPROP_GPU_MIN` very high.

## 3. MPS — **do not switch on a GB10**: 0.34x, and it is f64's fault

`examples/mps_chi128_bench.rs`, 14 qubits, depth 12, brickwall + Rx ring,
CPU Jacobi SVD vs cuSOLVER `gesvdj`:

| χ | CPU | CUDA | ratio |
|---|---|---|---|
| 32 | 108.8 ms | 188.2 ms | 0.58× |
| 64 | 113.4 ms | 338.1 ms | 0.34× |
| 128 | 113.2 ms | 331.9 ms | 0.34× |
| 256 | 113.3 ms | 336.3 ms | 0.34× |
| 512 | 113.2 ms | 333.8 ms | 0.34× |

Flat past χ = 128 because 14 qubits caps the bond dimension at 2⁷ — that is the
whole reachable range for this shape, not a plateau in the hardware.

**It misses its own acceptance gate.** The bench documents *"≥ 3× CPU on the
Linux+NVIDIA host (RTX PRO 6000 Blackwell)"*, and that gate was presumably met
there. Here it is 0.34×, i.e. **three times slower** — a ~9× swing between two
Blackwell GPUs.

**The mechanism is f64, and it is the sharpest example of the warning below.**
`gesvdj` is **native f64** end to end (`gesvdj.rs`: every device buffer is
`CudaSlice<f64>`), and this repository's own note records GB10 running f64 at
**1/64 rate** (`kernels.rs`, on `Precision`). A workstation RTX PRO 6000 does
not. So the same code, on two cards of the same architecture generation, lands
on opposite sides of the recommendation — which is exactly why the f32/f64 row
in §"To do on other hardware" says the ratio is hardware-specific by a factor
of 32 and must not be reasoned about by analogy.

**Earlier attempt, recorded because it was wrong in an instructive way.** Timing
`omega-run --backend mps:64 --device cpu` against `--device cuda` gave 0.007 s
vs 0.006 s at 20 qubits — an apparent clean 1.8× at 28. It was measuring
nothing: **`omega-cli` never wires `cuda_svd_flat` at all**, so `--device cuda`
on `--backend mps` runs the CPU SVD in both columns. The mps-cuda hook is
reachable only through `aria-runtime` (`run.rs`, `with_svd_fn`). The tell was
that the ~0.58 s CUDA startup appeared in neither column.

## 4. CPU thread scaling — saturates at ~2x, and more cores do not help

The CPU statevector's gate kernels are rayon-parallel (`2ed1eea`). How far that
scales had never been measured. On 20 arm64 cores, 26 qubits, `--statevector`
(gate evolution only — `--shots` would measure the serial sampler and report a
flat curve):

| threads | qft 26q | speedup | efficiency | hea 26q | speedup |
|---|---|---|---|---|---|
| 1 | 42.27 s | 1.00x | 100% | 29.46 s | 1.00x |
| 2 | 29.26 s | 1.44x | 72% | 22.66 s | 1.29x |
| 4 | 23.22 s | 1.82x | 45% | 20.04 s | 1.46x |
| 8 | 20.93 s | **2.02x** | 25% | 19.57 s | **1.50x** |
| 16 | 20.64 s | 2.04x | 12% | 19.70 s | 1.49x |
| 20 | 20.77 s | 2.03x | 10% | — | — |
| 40 | 21.82 s | 1.93x | 4% | — | — |

(The hea rows at 20 and 40 threads were cut off by a wall-clock limit on the
sweep, not omitted deliberately.)

**Scaling stops at ~8 threads.** Past that the curve is flat, and at 40 —
oversubscribed on 20 cores — it is slightly *worse*. Efficiency at the full core
count is 10%.

**This is memory bandwidth, not a defect in the parallelisation.** At 26 qubits
the f64 state is `2^26 x 16 B = 1 GiB`, and applying one gate streams the whole
thing in and out. GB10's LPDDR5X is shared between CPU and GPU, so the CPU side
saturates it at around 8 threads and further threads queue on the same bus.

Practical consequences:

- **`RAYON_NUM_THREADS=8` is the setting**, on this machine. Allocating 20 cores
  to one circuit wastes 12 of them; run more circuits concurrently instead.
- It is also part of why the GPU wins at 24+ qubits despite sharing the same
  physical memory: it computes in f32, so it moves **half the bytes**, and it
  has far more memory-level parallelism to hide latency with.
- **Bit-identity across thread counts holds**, including at oversubscription —
  `thread_count_invariance.rs` now covers `T = 40` on a 20-core host, which is
  where a partition that secretly depended on scheduling would have shown up. It
  does not: chunk boundaries are functions of `dim` and the target qubit only.

Two caveats, because this number will be quoted:

- This is an **arm64, unified-memory** curve. A many-core x86 box with dedicated
  DDR channels per socket should scale further, and that measurement is still
  open — 20 arm64 cores is not the many-core x86 machine `PLAN-SV-PERF.md` asks
  for, and this does not close that item.
- `PLAN-SV-PERF.md` §3.4 specifies a **dedicated** rayon pool rather than the
  global registry, and §3.5 an `ARIA_THREADS` knob. Neither is implemented, so
  `RAYON_NUM_THREADS` works here only because that design decision was skipped.
  If §3.4 lands, this sweep's mechanism changes.

---

## To do on other hardware — parked, not open

These are tracked in `CUDA_TODO.md` § "Deferred — needs hardware this box is
not". Listed here because a reader comparing backends needs to know which rows
are missing and why, not because any of it is actionable on a GB10.

None of the above transfers. Each of these needs the same measurement, and
until it has one the honest answer is "unknown", not "probably similar".

- **Metal** (Apple Silicon) — unified memory like the GB10 but a different
  kernel compiler and no NVRTC step, so the fixed cost is likely *lower* and the
  crossover *earlier*. Untested. Do the §1 sweep under `ARIA_METAL=1`.
- **OpenCL** — untested. Note it cannot currently build on this box (runtime ICD
  only, no `libOpenCL.so` or `CL/` headers), so this needs a machine with the
  dev package.
- **CUDA f32 vs f64, on hardware where the choice matters.** This is the row
  most likely to be got wrong by analogy, because the ratio is
  **hardware-specific by a factor of 32**:
  - GB10 (sm_121, consumer-class Blackwell): f64 runs at **1/64** rate. Choosing
    f64 here is close to giving up the GPU.
  - H100 (sm_90): f64 at **~1/2** rate. The same choice is nearly free.

  So a crossover measured in f32 on a GB10 says nothing about an f64 workload on
  an H100, in either direction. Both need their own row.

  **Blocked today:** `f64_path` has no `Backend` impl and no circuit walker, and
  `--device cuda` always runs f32 — there is no way to select f64 from the CLI,
  so the f64 crossover cannot be measured at all yet. See
  `BACKEND_FEATURE_PARITY.md`.
- **amd64 + discrete GPU (RTX 6000 Pro, H100).** Everything here is
  unified-memory. On a discrete card every host↔device transfer is a PCIe/NVLink
  hop that costs nothing measurable on a GB10, so the statevector crossover
  should move and the PauliProp verdict — which is dominated by per-gate
  transfers — could plausibly *invert*. This is the single most valuable
  follow-up in this file.

## Method, so a rerun is comparable

```sh
cargo build --release -p omega-cli --features cuda
# per (circuit, width): wall clock of the whole invocation, both devices
omega-run c.qasm --device cpu  --shots 1000
omega-run c.qasm --device cuda --shots 1000
```

Wall clock of the whole process is used deliberately: it is what a user
experiences, and it is the only way the 0.58 s fixed cost — the thing that
actually decides the crossover — shows up at all. Per-kernel timings are in
`benches/cuda_bench.rs` and answer a different question.
