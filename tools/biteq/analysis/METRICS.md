<!-- SPDX-License-Identifier: Apache-2.0 -->
# Bit-equality magnitudes — what each metric is, and why more than one

Companion to `tools/biteq/RESULTS.md`. The verdict ("CUDA and Metal are not
bit-identical") is established there. This file answers the follow-up: **how far
apart, and what does that number mean.**

Rev `b9e06b9` on both sides. 10 circuits, 3–6 qubits. GB10 aarch64 / CUDA 13.0
and Apple M5 / Metal. f32 machine epsilon `ε = 2^-23 = 1.1920929e-07`.

## Headline

| | value | meaning |
|---|---|---|
| worst amplitude error | **1.01 × ε** | largest disagreement on any single amplitude, corpus-wide |
| worst infidelity | **2.0e-14** | 0 = same state, 1 = orthogonal |
| fewest shots to notice | **9.0e+13** | measurements needed before the difference beats sampling noise |

Every deviation sits at or below the rounding floor of the format. The backends
disagree in the last representable bit — the smallest disagreement f32 permits
other than none.

## The metrics

No single metric is honest alone here; two are actively misleading if read by
themselves. Each is reported with the caveat that makes it usable.

### max |a_i − b_i|  — max amplitude error
Largest gap between corresponding amplitudes, as a distance in the complex plane.
The most direct answer to "how wrong is any one number?"
**Small means:** below `1.192e-07`. At that scale the two values are adjacent
representable floats — nothing exists between them.

### ULP distance — units in the last place
How many representable f32 values lie between the two numbers. Counts steps of
the format rather than absolute magnitude, so it stays meaningful across scales.
**Caveat, load-bearing here:** ULP is nonsense near zero. A true `0.0` against a
denormal `1e-17` reads as ~8e8 ULP while being numerically identical for any
purpose. **All ULP figures are gated at |v| ≥ 1e-6**; excluded near-zero cases are
counted separately, never folded in.

### 1 − |⟨a|b⟩|²  — infidelity
Standard physics measure of state difference. Fidelity 1 = identical, 0 = orthogonal.
**Read with suspicion:** ignores global phase and is quadratically insensitive near 1.
Across this corpus it spans only `1e-16`..`2e-14` while differing bit counts span
0..116. Leading with it makes every row look identical. It is reported *because*
its insensitivity is itself a finding.

### T = √(1 − F)  — trace distance
Maximum probability, over **every** measurement anyone could perform, of telling
the two states apart in one shot. The strongest general bound available.
**Why include it:** it bounds any experiment, not just the one that was run.

### TVD = ½ Σ |p_i − q_i|  — total variation distance
The same idea restricted to what the code actually does: computational-basis
sampling. Compares the two outcome distributions directly.
**Relation to T:** TVD is what you see with the measurement performed; T is the
worst case over all measurements. TVD ≤ T always, here by orders of magnitude.

### ≈ 1 / TVD²  — shots to distinguish
Order-of-magnitude samples before the difference rises above sampling noise.
Translates a float comparison into an experimental one.
**Heuristic, not a theorem:** the constant depends on confidence level and
distribution shape. It conveys scale; the scale is "unobservable".

## Results

**Platform alone (Exact vs Exact)** — decomposition contributes nothing, so this
is the compilers' contraction choices in isolation:

| worst 1−F | worst T | worst TVD | worst max&nbsp;\|Δ\| | max ULP | shots |
|---|---|---|---|---|---|
| 2.22e-15 | 4.71e-08 | 3.74e-08 | 3.33e-08 (0.28 ε) | 6 | 7.2e+14 |

142 of 512 parts differ, 118 near-zero.

**Platform + decomposition (Decompose vs Decompose):**

| worst 1−F | worst T | worst TVD | worst max&nbsp;\|Δ\| | max ULP | shots |
|---|---|---|---|---|---|
| 1.91e-14 | 1.38e-07 | 1.05e-07 | 1.20e-07 (1.01 ε) | 48 | 9.0e+13 |

305 of 512 parts differ, 220 near-zero — so ~72% of the raw "differing bits"
count is amplitudes that are all effectively zero.

**What the decomposition costs:** infidelity 8.6×, max |Δ| 3.6×, ULP 8.0×.
Differing contraction is the *cause* of bit-inequality (it makes the Exact rows
non-zero at all), but the decomposition is the larger contributor to *magnitude*.

**The 48 is one amplitude, not a trend.** `p90_ry_probe` has median 2.5 ULP with a
single 48 outlier. Without the median column it would set the corpus headline and
read as typical. The deepest circuit, `c07` at 6 qubits, is median 2 / max 3.

**Accuracy vs the f64 reference** — Exact is closer to truth than Decompose on
*both* platforms (CUDA 2.89e-15 vs 2.07e-14; Metal 3.77e-15 vs 7.77e-15),
independently corroborating the accuracy claim made when the exact kernel landed.

## What these numbers do not say

- **Not a general bound.** 10 circuits at 3–6 qubits. Deeper circuits accumulate
  more; nothing extrapolates to 20 qubits or 200 gates.
- **Scoped to the measured pairs.** One GB10/CUDA 13.0 and one M5/Metal. A different
  compiler version can make different contraction choices — that is the mechanism.
- **Norms checked, not corrected.** Every ‖ψ‖² is within n·ε of 1 (worst 5.2e-07
  against 1.9e-06 tolerance). Nothing was renormalised to flatter a fidelity.
- **Gate 1 verified on amplitudes, not files.** The `.cpu.json` files differ
  byte-wise on host metadata; a naive `cmp` reports DIFFERS on all 10 and means
  nothing. On bit patterns the f64 references are identical across both hosts.

Script: `metrics.json` + generator alongside this file. Computed CUDA-side,
verified independently against both artifact sets Mac-side.
