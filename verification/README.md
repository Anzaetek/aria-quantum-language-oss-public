<!-- SPDX-License-Identifier: Apache-2.0 -->
# `verification/` — Lean 4 specification targets

**Read this first: almost nothing here is proved yet.** This tree states the
semantics the Rust backends are supposed to have, in a form a proof can later
be discharged against. It is deliberately specification-first:

* `axiom` — the assumed trust boundary (state vectors, projectors, purity…).
  These are *not* claims we derived; they are what the model takes as given.
* `theorem … := by sorry` — a **target**. Stated precisely, not proved.
* `theorem … := by decide` / a real proof — actually machine-checked.

Every file records the *measured* defect that motivated it, so a future prover
knows what the statement is defending against rather than reconstructing it.

## Status

| File | Proved | Targets (`sorry`) |
|---|---|---|
| `Verification/Backend/Reset.lean` | — | 4 (`reset_outcome_irrelevant`, `reset_yields_zero`, `fold_is_not_reset`, `entangled_reset_not_pure`) |
| `Verification/Backend/StabilizerExpectation.lean` | **`gPhase_correct`** | 3 (`expectation_trichotomy`, `zero_only_when_anticommuting`, `echelon_reduction_complete`) |

`gPhase_correct` is the one real proof: the shipped `pauli_mult_phase` table
equals the Aaronson–Gottesman closed form on all 16 input pairs, by exhaustive
case split. It is **falsifiable** — restoring the pre-fix `X·Z`/`Z·X` rows makes
`decide` reject it, which was checked rather than assumed.

## `set_option autoImplicit false` is load-bearing

Both files disable it, and must keep doing so. Core Lean has no `Real`; with
`autoImplicit` on (the default) the `Real` in these signatures was silently
auto-bound as an implicit `Sort` parameter of every axiom:

```
axiom p0.{u} : {Real : Sort u} → {n : Nat} → State n → Fin n → Real
```

Instantiating it at `Empty` then *proved* `State n` uninhabited, and every
target was dischargeable by `.elim` with no physics whatsoever. The files
typechecked and specified nothing. That is why the carriers are now declared
explicitly as inhabited abstract types (`R`, `R.one`, …) — an *empty* carrier
reintroduces exactly the same vacuity.

Regression check (should FAIL to elaborate):

```lean
theorem states_are_empty {n : Nat} (s : State n) (q : Fin n) : False :=
  (p0 (Real := Empty) s q).elim   -- error: invalid argument name `Real`
```

## Typecheck

```bash
lean verification/Verification/Backend/Reset.lean
lean verification/Verification/Backend/StabilizerExpectation.lean
```

Both exit 0; the `sorry` warnings are the targets above and are expected. Core
Lean only — no Mathlib dependency, which is why `gPhase_correct` uses an
explicit `cases` split rather than `Fintype`'s decidable-forall.

## Why this tree exists

`crates/omega-backend-statevector/src/sim.rs` cited
`verification/Verification/Adjoint/{Linearity,PauliExpectation}.lean` — files
that did not exist in this repository. Those citations are now marked as
**targets not yet written** at their call sites rather than reading as
references to extant proofs.
