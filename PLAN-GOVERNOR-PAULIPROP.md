<!-- SPDX-License-Identifier: Apache-2.0 -->
# Plan: price PauliProp by the circuit, not by the cap

## The defect

`cost_kind_for` (`quantum_bridge.rs:315`) takes only the backend selector. It
never sees the circuit. So **every** PauliProp job is priced at
`DEFAULT_MAX_TERMS` (2²¹ terms), identically:

```
n=   4  reserved = 285.2 MB
n=  12  reserved = 285.2 MB
n=  40  reserved = 285.2 MB
n= 100  reserved = 352.3 MB
```

A 4-qubit Bell-pair expectation reserves 285 MB. The consequence is
**over-refusal**: concurrent PauliProp jobs that would fit comfortably get
turned away, and the one backend with no qubit ceiling is throttled by a
constant that has nothing to do with the work it was asked to do.

This is the gap the out-of-support skip exposed. We now know cost is bounded by
the **observable's light cone and the non-Clifford depth inside it**, not by the
register — but the governor still cannot see that.

## The bound

An estimate is not acceptable here: under-pricing fails as an **OOM**, not as a
clean refusal. So this computes a genuine **upper bound**, and every unknown
resolves toward the cap.

```
bound = seed_terms · 2^branches      capped by  4^|cone|  and  cap · 2^7
```

`seed_terms` is **`observable.terms.len()`** — the number of Pauli strings in
the observable, not the number of qubits it touches. Duplicates merge on
insert, so it is a safe over-estimate. (Spelling this out because "distinct
qubits in the observable" would under-price a many-term Hamiltonian by orders
of magnitude.)

Three independent ceilings, take the minimum:

1. **`seed_terms · 2^branches`** — the sum starts as the observable's terms, and
   each `branch` call at most doubles it (a cos child that always survives, plus
   at most one sin child). Clifford gates map one Pauli to one Pauli, so they
   cannot grow it at all.
2. **`4^|cone|`** — there are only `4^k` distinct Paulis on `k` qubits, so the
   sum cannot exceed that however deep the circuit is. This is what makes a wide
   shallow circuit cheap.

   **Precondition, now enforced:** this counts distinct *operators*, so it bounds
   `terms.len()` only if the map is keyed injectively by operator. That needs the
   canonical-form invariant (bits above `n` are zero) AND observable indices
   inside the register — which nothing checked until `731f806`, where an
   out-of-range index panicked four backends. With unmasked padding the key space
   is `4^(64w)`, not `4^n`, and this ceiling would be fiction.
3. **`DEFAULT_MAX_TERMS · 2^max_branches_per_gate`** — NOT the cap alone.

   The engine checks `sum.terms.len() > cap` **once, after the whole gate
   dispatch has run** (`sim.rs:652`), and deliberately so: the comment explains
   it measures a multi-branch gate "once, at its real peak". The consequence it
   misses is that a sum can enter `CCX`/`CSwap` at `cap - 1` terms and take
   **seven consecutive doublings before any check fires**. Real in-flight peak
   is up to `2^7 · cap`, i.e. ~268M terms — about **36 GB** at n=40, against a
   285 MB reservation.

   Taking `min(..., cap)` as the first draft did would have *actively hidden*
   this: a correct larger `seed · 2^branches` gets clamped down to a number the
   engine can exceed by 128x. So the third ceiling carries the `2^7` headroom.

   This is a pre-existing engine defect, not one this change introduces. It is
   recorded here because pricing against a bound the engine does not honour is
   how a governor ends up certifying an OOM.

### Counting branches correctly

**Branch CALLS, not gates.** Several gates branch more than once, and counting
gates would under-count — the unsafe direction:

| gate | branch calls |
|---|---|
| `H X Y Z S Sdg Sx Sxdg CX CZ CY Swap Id Barrier Measure` | **0** |
| Rz, Rx, Ry, U1, T, Tdg | 1 |
| CRz | 2 |
| U3, CU3 | 3 |
| **CSwap** | **7** |
| CCX | 7 |

**The zero row is not padding — it is load-bearing.** An earlier draft said
"any gate not in the table ⇒ cap", and listed only the branching gates. Under
that rule a Bell pair (`H; CX`) has two unlisted gates and prices at the cap:
the change would have been completely inert, reproducing the exact defect it
set out to fix. The cap fallback is for genuinely unknown kinds only (`U2`,
`Rbs`, `PhaseShifter`, `Custom` — all of which the engine refuses anyway).

**CSwap was wrong in the first draft** (6, from a regex count of `self.branch(`
per match arm). It is **7**: `sim.rs:612-618`, the seven non-empty subsets of
`{c,a,b}`, exactly like CCX — and the source comment says "the same seven
branchings". A 2x under-count per CSwap, in the unsafe direction, in the table
whose own caption warns about under-counting. Regex-counting match arms is not
verification; the count is now pinned by a test.

### The light cone

A single reverse pass. A gate that does not touch the current support cannot
affect the observable — the same argument the out-of-support skip rests on:

```
support = observable's qubits
for gate in ops.iter().rev():
    if gate.qubits ∩ support ≠ ∅:
        support ∪= gate.qubits
        branches += branch_calls(gate.kind)
```

Over-approximating the cone is safe (it only raises the bound). Under-
approximating is not, which is why the union happens before the count.

## Where it must fall back to the cap

Each of these resolves to `DEFAULT_MAX_TERMS` rather than a guess:

* **No observable** — `/execute` has none. (It already refuses PauliProp, but
  the pricing path must not assume that stays true.)
* **A noise model with amplitude damping.** Non-unital damping is
  `Z → (1−γ)Z + γ·I`: it **spawns** an identity companion, so it grows the sum
  without going through `branch`. Rather than model it, price it at the cap.
* **Arithmetic overflow — and note that release builds do NOT panic here.**
  The workspace sets no `[profile]` overrides, so `overflow-checks = false` in
  release and every natural spelling fails *toward zero*:

  | expression | `branches`/`k` | release result |
  |---|---|---|
  | `1usize << branches` | 64 | shift amount is MASKED → `1 << 0` = **1** |
  | `2usize.pow(branches)` | 64 | wraps → **0** |
  | `4usize.pow(k)` | 32 | wraps → **0** |
  | `1usize << (2*k)` | 32 | masked → **1** |

  `min(0, ...)` reserves nothing. So: short-circuit to the cap when
  `branches >= 64` or `2*|cone| >= 64` **before** any shift, use
  `checked_shl`/`checked_pow`/`saturating_mul`, and **the overflow test must run
  in release** — a debug-only test asserts behaviour that is not what ships.
  Note `4^11` already exceeds the cap, so nobody testing small cones ever
  reaches the dangerous range by accident.
* **Any gate kind not in the table.** Unknown gate ⇒ cap, so adding a
  branching gate later cannot silently under-price.

## Where the estimator lives — and why it is NOT in the server

**Put `term_upper_bound(ir, observable) -> u64` in `omega-backend-pauliprop`,
beside the gate dispatch it mirrors.**

The branch-call table is a shadow of `sim.rs`'s `match` arms. If it lives in
`omega-server`, then adding a branching gate to the engine — or changing CCX
from 7 branch calls to 8 — silently UNDER-PRICES every job using it, from a
crate whose author has no reason to look at the server. That is the same
failure shape as the feature-gated code this repo keeps losing, except the
symptom is an OOM rather than a compile error.

Co-located, the two are one file apart, and a test in that crate can assert
**every `GateKind` the dispatch accepts appears in the bound table** — so
adding a gate without pricing it fails a test instead of over-committing
memory in production.

The server keeps only the plumbing: thread the observable from
`quantum_bridge.rs:1229` (where it is already parsed) through `admit_batch`
and `shape_for` into `cost_kind_for`, which today takes just the selector.
`/execute` passes `None` and therefore prices at the cap.

## The GPU hook conditions the whole bound

`quantum_bridge.rs` installs `omega_backend_pauliprop_cuda::cuda_branch` as the
branch hook under the `cuda` feature, and `branch` hands the sum to it and
returns immediately if it claims the work. Both ceilings then rest on a device
kernel rather than the CPU loop:

* `seed · 2^branches` needs the hook to honour `|out| <= 2·|in|`;
* `4^|cone|` needs it to MERGE duplicate keys — and `BranchHook`'s own doc warns
  that a batched device path needs a device-side merge (sort-by-key + segmented
  reduce, or a device hash table), without which terms fail to merge and the map
  holds MORE entries than there are distinct Paulis.

Today's hook satisfies both (it merges on the host). The bound is stated as
**conditional on that**, and the condition is written at the hook's definition
so a future device-side merge cannot quietly invalidate it.

## Batches

`/expectation` takes a list of circuits sharing one observable, and
`admit_batch` already prices every row. The bound is computed **per row** and
the reservation takes the max — rows differ in width and depth, and the
observable that is local in one may cover the register in another.

## Tests

* **Bound holds**: run circuits through the real engine, record the PEAK term
  count, assert `peak <= bound`. The only test that really matters — a bound
  that is not a bound is a memory defect, not a pricing one.

  This needs an engine hook that does not exist: `propagate` is private and
  `expectation_with_budget` returns only `(f64, f64)`. Add a thread-local peak
  cell next to `GATES_SKIPPED` (same rationale, same shape — thread-local
  because cargo runs tests in parallel and a global counter reports other
  tests' work, which already bit me once this session). It must sample the peak
  **inside** the `CCX`/`CSwap` arms, not only at `sim.rs:652`, or it will
  confirm a bound the engine violates mid-gate.
* **Bound is useful**: a small local-observable job must price **far** below the
  cap, else the change is decorative. Assert a concrete ratio.
* **Width is nearly free**: the same dynamics embedded in a wider register must
  not raise the price — the property the skip established.
* **Depth is not free**: adding non-Clifford depth inside the cone must raise it.
* **Fallbacks price at the cap**: no observable, damping, overflow.
* **Mutation**, expanded after the review caught a table error the original
  list would have missed:
  - changing `CSwap` from 7 to 6 must fail a test (this was a real defect);
  - replacing `checked_shl` with `<<` must fail a **release-mode** test;
  - dropping the `4^|cone|` ceiling must fail;
  - counting gates instead of branch calls must fail;
  - removing the amplitude-damping fallback must fail.

## Revision history

First draft reviewed adversarially before any code was written. It contained
**two under-counting errors, both in the unsafe direction**: `CSwap` at 6
branch calls instead of 7, and `min(..., DEFAULT_MAX_TERMS)` treating the cap
as a ceiling the engine does not actually honour (real peak up to `2^7 · cap`).
A third finding — "unlisted gate ⇒ cap" combined with a table listing only
branching gates — would have made the feature inert, pricing a Bell pair at the
cap. All three are fixed above. The review also confirmed the central claim:
only `branch` and amplitude damping can grow the sum.

The review additionally surfaced an unrelated live defect — an out-of-range
observable panicking four backends from an HTTP body — fixed separately in
`731f806` before this work continues.

## Out of scope

The engine's own `max_terms` refusal, `CostKind::PauliProp`'s per-term byte
arithmetic (already committed and unchanged), and the admission policy itself.
This changes only the **number of terms** handed to the existing pricing.
