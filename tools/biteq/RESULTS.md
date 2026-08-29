<!-- SPDX-License-Identifier: Apache-2.0 -->
# biteq verdict — 2026-08-20: the bit-for-bit claim is FALSE, and not because of the decomposition

One run, two machines, exit 2. Everything below is from `diff.py` over the
two artifact sets; both sides at rev **b9e06b9**, clean trees.

| side | device | host |
|---|---|---|
| A | metal-f32 (Apple M5 Max) | Darwin 25.5.0 (xnu-12377.121.10~1, RELEASE_ARM64_T6050) |
| B | cuda-f32 (NVIDIA GB10, sm_121) | Linux 6.17.0-1026-nvidia, Ubuntu |

## Gates

**Gate 1 — CPU f64 cross-machine identity: PASS, bit-identical on all nine
strict circuits.** macOS libm and glibc produced the same f64 bits for the
whole exact-constant corpus. Every downstream difference therefore belongs to
the GPU kernels, not host math. (The `p90_ry_probe` — host `cos`/`sin` —
ALSO came back bit-identical across the two hosts on this corpus, so even
the rotation-gate hazard did not fire here; it stays excluded from the
verdict on principle.)

**Gate 2 — the mode switch demonstrably switches on BOTH boxes** (Exact vs
Decompose, differing amplitude parts across the strict corpus): Metal 152,
CUDA 148. Per circuit:

| circuit | Metal | CUDA |
|---|---|---|
| c01_ccx_basis_fire | 4 | 2 |
| c02_ccx_basis_nofire | 4 | 4 |
| c03_ccx_super | 16 | 4 |
| c04_ccx_super_nofire | 6 | 0 |
| c05_cswap_super | 0 | 0 |
| c06_ccx_repeat | 42 | 16 |
| c07_cswap_ccx_mix_6q | 64 | 108 |
| c08_ccx_adjacent_vs_strided | 16 | 14 |
| c09_cz_only_control | 0 | 0 |

c09 is zero by construction (no multi-control gate — it is the negative
control, and it produced the load-bearing headline row below). c05 is zero on
both boxes: on THAT STATE the CSwap decomposition and the permutation agree
bit-wise on both platforms — a fact about that state as much as the kernels,
and no licence to infer the two paths agree in general. It carries headline
value while contributing nothing to gate 2; those are different jobs.

The CUDA side also confirmed `cd5c403`'s "more accurate" claim from an
independent angle: on `c02` (controls NOT set), Exact is bit-clean against
the f64 reference while Decompose carries ~1e-8 of accumulated f32 junk in
the untriggered amplitudes. The surprise is not that the untriggered case is
worst — it is not (c07 carries far more differing parts) — but that the
untriggered chain accumulates error at all instead of cancelling to
identity, which is why the corpus insisted on that circuit.

## Headline — Metal-Decompose vs CUDA-Decompose: differs on all 9 circuits

Every circuit differs (3 to 116 amplitude parts; no signed-zero-only cases
anywhere, so the H5 dense-vs-permutation CX mechanism was not the visible
failure mode). The decisive rows are not the CCX ones:

* **c09 (no multi-control at all): 16 parts differ.** The disagreement
  exists with the decomposition entirely out of the picture.
* **Exact-vs-Exact localiser:** bit-identical on the basis-state circuits
  (c01, c02, c06 chains of permutations) — the CCX/CSwap permutation kernels
  agree perfectly across platforms — but differs on every superposition
  circuit, same shape as the headline.

So the cause is the **elementary gate kernels' f32 arithmetic**: Metal
compiles its shaders at runtime with fast-math, CUDA fuses with `fmad`, and
identical sequences of complex multiplies contract differently. Typical
first-diff values are at the 1e-8 scale (f32 epsilon) on amplitudes of order
1 — e.g. `3f3504f2` vs `3f3504f0` around 1/√2. The huge ULP counts in the
raw report are zero-vs-1e-8 comparisons, not large absolute errors.

## Magnitudes — "differs" bounded (2026-08-20, same artifacts)

Computed on the CUDA side at the project owner's request and **independently
recomputed on the Mac from both artifact sets before landing here** — every
load-bearing figure agreed. Cross-platform, per mode, over the strict corpus
(ULP gated at |v| ≥ 1e-6; a 0.0-vs-1e-17 pair reads as ~8e8 meaningless ULP
otherwise):

| | Exact vs Exact (platform alone) | Decompose vs Decompose (platform + decomposition) |
|---|---|---|
| max infidelity 1−F | 2.0e-15 | 2.0e-14 |
| max relative L2 | 5.3e-08 | 1.7e-07 |
| max per-amplitude \|Δ\| | 3.3e-08 (0.28 × f32 ε) | 1.2e-07 (1.01 × f32 ε) |
| max gated ULP | **6** | **48** |
| differing parts (of 512) | 142, of which 118 near-zero | 305, of which 220 near-zero |

Readings, in the order they matter:

* **Every deviation sits at or below one f32 epsilon; corpus-wide
  infidelity ≤ 2e-14.** "Bit-for-bit is false" and "the difference is
  acceptable" are different claims; both are now established, the second
  with a bound at the representation's own rounding floor.
* **The decomposition contributes roughly an order of magnitude more than
  the platform difference** (6 → 48 max ULP) — fast-math-vs-fmad is the
  *cause* of bit-inequality, but once present the 15-gate chain is the
  larger *magnitude* contributor.
* ~72% of differing parts in Decompose mode are near-zero amplitudes
  (true zero vs denormal junk); the raw "305 of 512" overstates what a
  consumer of non-negligible amplitudes would see.
* The max-ULP outlier is `p90_ry_probe` (median 2.5, max 48 — one
  amplitude); c07, the deepest circuit, sits at median 2 / max 3. Medians
  are reported precisely so the tail cannot pose as typical.
* Fidelity alone would have hidden all of this: 1−F spans 1e-16..2e-14
  while bit-differences span 0..116 parts — max-|Δ| and gated ULP are the
  discriminating metrics, F is reported alongside, never as the headline.
* Independent corroboration of `cd5c403` on both platforms: Exact is closer
  to CPU f64 than Decompose on each (CUDA 3.1e-15 vs 2.1e-14; Metal
  4.0e-15 vs 7.8e-15, infidelity vs the f64 reference).
* Norm sanity: every artifact's ‖ψ‖² within n·ε of 1 (worst 5.2e-07);
  nothing renormalised.

**Distinguishability, in units someone can refuse to care about:** the two
backends become statistically distinguishable by computational-basis sampling
only after **~9.0e13 shots** (worst circuit, worse mode); platform-alone is
7.2e14 — eight times further out. Trace distance, TVD, shots-to-distinguish
and Hellinger are defined — each with the caveat that makes it usable, two
with their measured failure modes — in `analysis/METRICS.md`;
`analysis/compute_metrics.py` regenerates `analysis/metrics.json` from the
two artifact directories (computed CUDA-side, independently recomputed here;
the only scatter was 1-ULP python-libm differences at the 1e-16 floor, which
is the phenomenon under study saying hello from inside the measuring stick).

Left open, deliberately: nothing further for this claim. A future accuracy
statement about either GPU should cite the bound above, scoped to these
OS/compiler pairs, not re-derive it from inspection.

## What changes, what does not

* `omega-core/src/executor.rs` (`MultiControlMode::Decompose`) and
  `GATE-EXACTNESS.md` §2.3 no longer promise cross-GPU bit equality; they
  say what is true — the same decomposition, each validated against CPU f64
  (1e-6), with the platform divergence measured and scoped to the pairs
  above.
* **Nothing here is a correctness defect.** Both GPUs sit inside their 1e-6
  gates against CPU f64 on every circuit; the CPU references agree across
  hosts bit-for-bit.
* The verdict is scoped to the measured OS/compiler pairs. A macOS or CUDA
  toolchain update can move it in either direction, which is precisely why
  the artifacts stamp `os_version` and the build rev.

## Reproducing

Artifacts were produced by `tools/biteq/run_side.sh` on each box (Metal side
on the M5 Max, CUDA side on the GB10 at detached b9e06b9) and exchanged by
scp; `python3 tools/biteq/diff.py <mac-dir> <dgx-dir>` re-renders everything
above, exit 2.
