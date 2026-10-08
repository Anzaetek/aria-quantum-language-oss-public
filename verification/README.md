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

**The Reset targets hold in a concrete model** — but that does not discharge
them here. `proofs/lean4/QuantumProofs/ResetModel.lean` (Mathlib) gives every
symbol of `Reset.lean` a meaning — a state is a unit vector of amplitudes over
basis bitstrings, `project` keeps the `q = b` amplitudes and renormalises,
`applyX` flips bit `q`, `reducedPurity` is `tr ρ_q²`, `EqUpToPhase` is equality
up to a unit scalar, `fold` is the superseded coherent fold — and proves all
four targets there, sorry-free, with a witness that each premise is inhabited
(`|+⟩` for T1/T2, a Bell state for T4). Two differences from this file, both
deliberate: in the model T2 needs only `Possible`, not `Deterministic`; and
T3 is provable there only because `fold` is defined, where here it is a free
axiom. The four `sorry`s above stay: their symbols are unrelated axioms, so no
target is provable or refutable in this file, and adding bridge axioms would
assume the theorems.

**The first stabilizer target holds in a concrete model**, under the same terms.
`proofs/lean4/QuantumProofs/StabilizerModel.lean` gives the symbols of
`StabilizerExpectation.lean` a meaning. A Pauli is a record
`i^k · X^x · Z^z` acting on amplitudes over basis bitstrings. Records multiply
by `(k, x, z)(k', x', z') = (k + k' + 2 z·x', x + x', z + z')`, and `act_mul`
proves that is the operator product. A stabilizer state is `n` Hermitian,
pairwise commuting generators and a unit vector each one fixes.
`Anticommutes` is the symplectic condition `ω(gᵢ, P) = 1`, and
`anticommutes_iff_operators` proves it equals operator anticommutation.
`InGroupPlus`/`InGroupMinus` are **combinatorial**: `P` equals, as a record, a
product of generators, or minus one. They are not the eigen-condition
`Pψ = ±ψ`, which would make the trichotomy restate its hypothesis.
`expectation_trichotomy` (T1) is proved there. The witness `witness_T1` is the
Bell state with generators `XX` and `ZZ`: `Z ⊗ I` anticommutes, `XX` is in the
group, and `YY = −(XX)(ZZ)` is in it with sign `−`. `bell_YY_direct` computes
`⟨Φ⁺|YY|Φ⁺⟩ = −1` from the amplitudes, not through T1.

T2 holds there too, as `zero_only_when_anticommuting`, with independence of
the generators' bits as an explicit hypothesis. That hypothesis does work.
`full_rank_needed` lists `Z ⊗ I` twice on `|0⟩ ⊗ |+⟩`, and there `I ⊗ Z` has
expectation `0` while anticommuting with nothing. The proof is linear algebra
over `𝔽₂`. Mathlib has the dimension theorem for orthogonal complements under
a nondegenerate form (`LinearMap.BilinForm.finrank_orthogonal`, from
`finrank_add_finrank_orthogonal'`). It has no Pauli operators, and nothing
connecting operator commutation to the symplectic form. Those are built here:
`act_mul`, `act_swap`, `anticommutes_iff_operators` and `omega_eq_symp`. A
defect in the model would sit there, which is why the mutations target those
definitions. `commuting_hermitian_in_group` is the stronger form: a Hermitian
Pauli that commutes with every generator is `±` a product of generators,
never `±i` times one. That is the fact the shipped elimination relies on
when it reaches a group element. T3 is not done yet, and the three `sorry`s
above stay for the same reason as Reset's.

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
cd proofs/lean4   # so elan picks the pinned toolchain
lean ../../verification/Verification/Backend/Reset.lean
lean ../../verification/Verification/Backend/StabilizerExpectation.lean
```

Both exit 0; the `sorry` warnings are the targets above and are expected. Core
Lean only — no Mathlib dependency, which is why `gPhase_correct` uses an
explicit `cases` split rather than `Fintype`'s decidable-forall.

`ci.sh` runs exactly this wherever `lean` is on PATH and **fails unless the
`sorry` count per file is exactly the one in the table above** (0 / 4 / 3),
after first checking that the gate rejects a file with one undeclared `sorry`.
Discharging or adding a target means updating the table, STATUS §5 #4 and the
count in `ci.sh` in the same commit.

## What a proof would need

Nothing here is dischargeable in place: every symbol a target mentions is a
free `axiom` with no axiom relating it to another, so no target is provable
and none is refutable. Adding relating axioms until a target goes through is
assuming the theorem. A proof means a concrete model — `State n := Fin (2^n)
→ ℂ` with `project` a normalised projector, a tableau for the stabilizer
file — under Mathlib in `proofs/lean4`, whose `CircuitSemantics.lean` has the
unitary matrix semantics but no measurement, Reset or tableau yet.

**Statement defect fixed 2026-09-30.** `Reset.lean` T1 and T2 were false in
that model: on `|0⟩` the outcome-`true` branch has probability 0 and there is
nothing to renormalise. Both now assume `Possible s q b` (the forced outcome
has nonzero probability), and T1 ranges over every pair of possible outcomes.

## Why this tree exists

`crates/omega-backend-statevector/src/sim.rs` cited
`verification/Verification/Adjoint/{Linearity,PauliExpectation}.lean` — files
that did not exist in this repository. They were first re-marked as **targets
not yet written**, which was still too generous: the private monorepo was
checked on 2026-09-04 and has no `Verification/` tree and no theorem by any of
the cited names, so nobody was tracking them as targets either. Those call
sites now state the mathematical obligation and name the numerical test that
actually exercises it. See the note at the top of
`crates/omega-backend-statevector/src/adjoint.rs` for the full retraction —
that module claimed ~25 Lean theorems and a CI gate (`lake build` over an
umbrella importing them) that never existed; `ci.sh`'s Lean stage runs in
`proofs/lean4/`, is opt-in, and skips when `lake` is absent.
