<!-- SPDX-License-Identifier: Apache-2.0 -->
# The PauliSum merge: what was measured, and what was done instead — 2026-08-17

This started as a plan to replace the term `HashMap` with a sorted-vec merge.
The measurement taken to justify that redesign found something else, and the
redesign was **not** done. This file records both, because the numbers bound
what is left to try.

## What was already settled before this

* **A faster hasher (FxHash) is SLOWER** on these keys. Pauli keys are sparse
  and highly structured — mostly-zero words differing in a handful of bits —
  the worst input for a multiply-xor hash. Extra collisions cost more than the
  cheaper hash saves.
* **Pre-sizing the rebuilt map at the 2n worst case is a PESSIMISATION**
  (0.91–0.93×, i.e. worse than not pre-sizing). Sizing at `n` wins 1.19–1.39×
  and is committed (`f26c281`). An over-sized table is a sparser table.

## The measurement that redirected the work

Step 1 of the original plan was to count how often an insert actually MERGES
onto an existing key, to decide whether the `HashMap` was earning its keep.

The merge rate came back low (13–16% non-truncating, <1% truncating), which
would have cleared the bar for the redesign. But one number was wrong-looking:

```
 nq  layers        inserts         merges
 10       6        8962945        1470538
 12       6       10007789        1470538
 14       6       11052633        1470538
 20       6       14187165        1470538      <- identical across ALL widths
 10       5        2075415         305622      <- but depth changes it
 10       7       38263747        6829750
```

**Merges were width-independent to the digit; inserts were not.** The cause:
with a local observable and a nearest-neighbour CX chain, the observable's light
cone was fully contained well inside these registers, so the real dynamics were
identical at 10 and at 20 qubits. Every extra insert was a gate acting **outside
the support**, rebuilding the entire term map into a bit-identical copy —
`img.apply(false, false)` is `(false, false, ONE)`.

That is not a container problem. It is work that should not happen at all.

## What was done: skip out-of-support gates

A conservative support mask on `PauliSum` (OR of `x | z` over inserted keys)
lets `map_single`, `map_two` and `branch` return immediately when a gate cannot
touch any term. The mask may over-approximate (truncation does not clear bits),
which only makes the skip fire less often — never wrongly.

Insert counts became **flat across width** (8,737,669 at nq = 10, 14, 20 and 30),
and wall-clock followed:

```
 nq  lay  mf |  guards OFF   guards ON |  speedup
 10    6   - |     982.3ms     963.1ms |    1.02x
 14    6   - |    1170.0ms     960.1ms |    1.22x
 20    6   - |    1440.0ms     957.6ms |    1.50x
 30    6   - |    1910.0ms     961.2ms |    1.99x
 20    6   3 |      12.0ms       6.4ms |    1.86x
 30    6   3 |      17.4ms       6.9ms |    2.54x
```

**Conditions for the wall-clock columns: hand-timed, no recorded idle check, no
load average, arms not interleaved.** Read them as shape, not as measurements to
regress against. The insert counts either side of them are worth more, because a
count is identical on an idle box and a loaded one.

**And the count above was itself unreproducible when written** — there was no
insert counter in the tree; it came from an ad-hoc patch that was never
committed. Fixed: `pauli::inserts()` is permanent, and
`tests/insert_count_is_flat_across_width.rs` asserts the flatness on a smaller
corpus (14,893 inserts at nq = 10, 14, 20, 30, with skips rising 79 -> 319).
Timings live in `benches/pauliprop_bench.rs`, which interleaves.

The ratio is not the point — the **flat ON column** is. Cost now tracks the
observable's light cone instead of the register, which is the property
pauliprop's width-unboundedness claim rests on and which was silently not
holding. The speedup keeps growing with width, so at the 100+ qubit widths this
backend exists for it is larger than anything in this table.

## Why the container redesign was NOT done

With the skip in place the case is materially weaker:

* the skip **removed** work; sort-vs-hash only **moves** it;
* it costs a 24-site refactor of a `pub` field;
* it changes coefficient summation order, so results move in the last bits.

The door is not closed — the merge is still the largest remaining line in the
GPU phase profile — but it should be re-measured against the post-skip baseline,
not against the numbers that motivated it.

## Facts worth keeping

* `PauliSum.terms` is **never queried by key** anywhere in the workspace. Its
  entire operation surface is `len`, `drain`, `retain`, `iter`, `iter_mut`, plus
  `entry` inside `add_weighted`. The map is a de-duplicator, nothing more — so a
  sorted-vec merge remains viable in principle.
* Coefficient summation order is **already** nondeterministic run to run
  (`HashMap` iteration), so bit-identity across versions was never on offer.
  Tolerances, not `to_bits()`, are the correct gate here.

## Next candidate

Push the same light-cone insight into the **GPU** branch path: it still uploads
terms that provably commute with the generator. Measure before assuming — the
CUDA branch is already slower than the CPU on this hardware.
