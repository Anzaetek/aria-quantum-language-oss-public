<!-- SPDX-License-Identifier: Apache-2.0 -->
# Plan: make `PauliSum`'s support invariant structural, not conventional

> **Revised after an adversarial review that disproved the first draft's central
> choice by experiment.** The original put the guard once per gate; that
> placement is blind to the exact class of bypass the noise-adjoint code already
> performs. Both the wrong version and the evidence are kept below, because the
> reasoning that produced it is the reasoning that will produce it again.

## The hazard

`PauliSum.support` is a conservative superset of the qubits any term touches. The
out-of-support skip (`map_single`, `map_two`, `branch`) is correct **only**
because it over-approximates: an extra bit makes the skip fire less often, never
wrongly.

The mask is maintained in exactly one place — `PauliSum::add_weighted`
(`pauli.rs:328`), which calls the private `note_support` unconditionally, and
`add` delegates to it. But **`terms` is `pub`** (`pauli.rs:266`). Anything doing
`sum.terms.insert(..)` bypasses that and leaves the mask an UNDER-approximation,
at which point the skip drops gates it must not.

The failure mode is the bad one: **a silently wrong expectation value.** No
crash, no compile error, no failing test.

**Why the hazard is bounded, which the first draft never said.** `PauliSum`
derives only `Clone, Debug, Default` (`pauli.rs:264`) — there is no serde in the
crate's manifest, so no `Deserialize` path — and `support` is **private**
(`pauli.rs:288`). No external crate can build a `PauliSum` by struct literal. The
entire external surface is `.terms` and `.dropped_mass`. That single fact is what
makes closing `terms` a complete fix rather than a partial one.

Today the invariant holds. Both GPU hooks only READ `.terms`:

| site | use |
|---|---|
| `pauliprop-cuda/src/gpu.rs:324, 339` | `.len()`, iterate |
| `pauliprop-cuda/src/lib.rs:68` | `.len()` |
| `pauliprop-metal/src/gpu.rs:80, 96` | `.len()`, iterate |
| `pauliprop-metal/src/lib.rs:75` | `.len()` |

**Correction to the first draft:** it said the only other `terms` hits were "an
unrelated local in `omega-parser`". False. `truncate` calls `retain`
(`pauli.rs:388`) and the rebuild path calls `drain` (`sim.rs:757, 778, 848`).
All are removals, all safe — but they are the whole reason the superset direction
is required, so writing them out of the picture removed the justification for the
design.

## 1. The guard, and where the first draft put it wrong

### What the first draft said, and the experiment that killed it

It placed the check once per gate, at the top of `propagate`'s
`for op in circuit.ops.iter().rev()` loop, reasoning that per-consumer would be
too hot.

`map_single` (`sim.rs:755`), `map_two` (`776`) and `branch` (`846`) each build a
**fresh** `PauliSum::with_capacity(..)` — whose `support` starts empty
(`pauli.rs:313`) — fill it exclusively through `add_weighted`, then `*sum = out`.
**The mask is regenerated from scratch on every non-skipped gate.** A bypass
written into a live sum is washed away before the next gate's assert ever runs.

Measured, not argued: an injected `terms.insert` immediately after the skip check
in `map_single` produced three test failures — all **value** mismatches from the
injected term, and **zero** assert firings. The guard never saw it.

**And the blind spot is occupied.** `sim.rs:387` calls
`apply_gate_noise_adjoint`, which reaches `scale_by_local_pauli`
(`sim.rs:1286`, `terms.iter_mut()`) and `apply_amplitude_damping_adjoint`
(`sim.rs:1298`, `std::mem::take(&mut sum.terms)`). Those are in-place mutators of
a live sum, running *after* the assert, with the gate's rebuild following them.
The entire noise-adjoint family sat in the hole.

The draft covered its stated threat only by luck: the GPU hooks build `out` via
`add_weighted` and `branch` returns immediately after the hook (`sim.rs:828`),
with no rebuild to wash it out.

### Where it actually goes

**Immediately before each mask consumer.** There are exactly three, all in
`sim.rs`, confirmed by a workspace grep:

* `sim.rs:753` — `touches_qubit` in `map_single`
* `sim.rs:774` — two `touches_qubit` in `map_two`
* `sim.rs:819` — `overlaps` in `branch`

Same order of cost (at most 7 per gate, `CCX` being the worst) and it fires
*before* the decision the mask exists to make, which is the only moment that
matters. The draft rejected "per-query" — but that argument was against putting
the check *inside* `touches_qubit`/`overlaps`, a different and much hotter
placement. It knocked down the wrong option and adopted the weaker one.

### Shape

```rust
pub(crate) fn debug_assert_support_is_superset(&self) {
    debug_assert!( /* whole computation inside the macro */ );
}
```

**Not** `#[cfg(debug_assertions)]` on the method, as the draft had it. That
requires a matching `cfg` on every call site (the draft never mentioned this, and
`cargo build --release` fails without it) and makes the crate's public API
cfg-dependent. A body inside `debug_assert!` compiles out entirely in release.

### Direction: superset, and equality is provably wrong

Assert `recomputed & !stored == 0`. A stored bit with no live term is **legal** —
`truncate` (`pauli.rs:388`) removes terms via `retain` without clearing the mask.

Verified: flipping the assert to equality fires on *correct* code at the first
truncation, in `truncation_error_curve_is_certified_and_converges`
(`recomputed 0x1e, stored 0x1f`).

### The limitation, stated because a gate that reads stronger than it is, is worse than none

`recomputed & !stored` ORs over **all** live terms, so a bypassed insert is
invisible whenever its support is already covered by another surviving term. In a
light-cone-local sum — the regime this engine exists for — terms share support,
so **a bypass is more likely invisible than visible.**

This is a smoke detector, not a proof. The mutation test must therefore use a
bypass that introduces a support bit **no other term carries**, and the test must
say that is what it is doing and why a differently-shaped bypass would slip past.

### Cost

~+13% on a synthetic 24-qubit debug run (240.6 s -> 271.8 s). **One run each, no
idle check, no interleaving** — which is precisely the discipline §4 says is
missing everywhere else in this repo, so read it as "same order, roughly 10%",
not as 13%. Release is unaffected by construction.

## 2. `pub` -> `pub(crate)` on `terms`, with read accessors

`pub(crate)`, not fully private: `sim.rs` legitimately needs `drain()` on the hot
rebuild path (`757, 778, 848`), and it is a sibling module.

**State the limit: this blocks bypasses from OUTSIDE the crate only.** An
in-crate mistake is still possible, which is what the guard in step 1 is for, and
why step 1 lands first rather than being made redundant by step 2.

New public surface, matching what the six external sites use:

```rust
pub fn len(&self) -> usize
pub fn is_empty(&self) -> bool          // clippy demands it alongside len()
pub fn iter(&self) -> impl Iterator<Item = (&PauliKey, &Weighted)>
```

Verified by applying it: exactly **2** errors workspace-wide
(`metal/src/lib.rs:75`, `cuda/src/lib.rs:68`). The crate's own integration tests
are unaffected — every `.terms` in `tests/expectation.rs` is `Observable::terms`
from omega-core. `omega-bridges` never touches it. No doctests.

The external surface is **not** only those six reads: both GPU crates also call
`PauliSum::with_capacity` and `add_weighted`, and write `out.dropped_mass`. Those
stay public — they are the maintained path, which is the entire point.

## 3. No escape hatch

M5 proposed an `insert_unchecked`-style bypass for a future bulk path. Not adding
it: with no caller, its correct usage cannot be demonstrated, tested, or
reviewed, and an untested bypass-by-convention API is worse than none. When a
bulk path arrives, the hatch and its first caller land in the same commit.

**Struck from the draft: "the entire value of this change is that bypassing
becomes impossible."** It is not impossible — §2 concedes in-crate bypass
survives, and §1 shows the guard misses the in-place class. The untested-API
argument stands on its own and does not need the overstatement.

M5 reviewed the distinction from keeping `EmittedButUnreadable` earlier today and
agreed it is real rather than invented: that variant's absence pushes the next
author toward a lie and misuse is loud; a bypass API's absence pushes toward the
correct call and misuse is silent. Opposite gradients, not the same case decided
twice.

## 4. Make the load-bearing evidence reproducible — a COUNTER, not just a bench

`omega-backend-pauliprop` has no `benches/` directory; eight other crates do. So
the packed-key and light-cone speedups cannot be reproduced by anyone.

`BACKEND-CROSSOVER.md` and `PLAN-PAULISUM-MAP.md` record no idle check, no load
average, no interleaving for any figure. Those ratios are load-sensitive and
unlabelled, and three measurements earlier this session were invalidated by load.

The light-cone claim does not rest on a timing — it rests on the insert **count**
going flat across width (8,737,669 at nq = 10, 14, 20, 30;
`PLAN-PAULISUM-MAP.md:53`). A count is immune to machine load.

**But the draft's defence had a hole it could not see.** There is no insert
counter in the tree — the only instruments are `note_peak` and `note_skipped`
(`sim.rs:163, 177`). That number came from an ad-hoc unrecorded patch, so it is
load-immune but **not reproducible**, and the draft's own headline ("cannot be
reproduced by anyone, on any box") applies to it too. A bench measures *time* and
will never re-derive a *count*.

So ship both:

1. a permanent insert counter in the `GATES_SKIPPED` mould, plus a test asserting
   the count is **flat across width** — that is the actual claim;
2. a `benches/` directory for the timings, labelled with what was controlled.

## Gates

* `cargo test --workspace` green in **debug**, where the guard is live.
* `cargo clippy --workspace --all-targets -- -D warnings` exits 0.
* Guard mutation-tested with a bypass introducing a support bit no other term
  carries (per §1's limitation), and `truncate`'s legal over-approximation must
  NOT trip it.
* The crate's first `#[cfg(test)] mod tests` lands **in step 1's commit** — there
  is none today (`grep cfg(test) src/*.rs` -> no matches), and after step 2 no
  external test can perform the bypass at all. Without this the mutation gate is
  satisfiable at step 1 and rots at step 2.
* Insert-count flatness test passes at nq = 10, 14, 20, 30.
* **`ARIA_METAL=1 ./ci.sh` on the M5 box.** That already runs
  `cargo test --release -p omega-backend-pauliprop-metal --features metal` and a
  featured clippy — added by `6f4054a`, yesterday. The draft asked for "M5 reviews
  the gpu.rs diff", which is strictly weaker than a compiler that already exists.
  Keep the human review as a supplement, not as the gate.
* Full CI green here, CUDA + Qiskit included.

## Explicitly not in scope

The sort-vs-hash container redesign (parked by user decision), `dropped_mass`
(pub, no cross-field invariant — and **both** GPU hooks write it, not just Metal
as the draft said), and making `terms` fully private.
