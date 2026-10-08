/-
  Reset semantics — specification targets for the omega statevector backends.

  Status: **specification, not yet proved.** `theorem`s are targets (`sorry`);
  `axiom`s are the assumed trust boundary.

  Motivation (measured, 2026-08-05): `omega-backend-statevector-metal`
  implemented Reset as a COHERENT FOLD (`new0 = old0 + old1`, renormalise)
  while the CPU backend had been corrected to a channel. The fold is wrong even
  on an unentangled qubit — on `|−⟩` it yields `old0 + old1 = 0`, and the
  `norm_sq > 0` guard then skipped renormalisation and left a zero register.

  **`autoImplicit` is disabled deliberately.** With it on (the default), the
  `Real` in these signatures was silently auto-bound as an implicit `Sort`
  parameter of each axiom, so `p0 (Real := Empty) s q` proved `State n` empty
  and every target below was dischargeable by `.elim` without any physics. The
  file typechecked and specified nothing. Core Lean has no `Real`, so an
  abstract inhabited carrier `R` is declared instead.
-/

set_option autoImplicit false

namespace Verification.Backend.Reset

/-- Abstract carrier for probabilities/amplitudes. **Inhabited on purpose**: an
    empty carrier would let the axioms below prove `State n` uninhabited, which
    is exactly the vacuity `autoImplicit` introduced. -/
axiom R : Type
axiom R.zero : R
axiom R.one : R
axiom R.inhabited : Inhabited R

/-- A pure state on `n` qubits. -/
axiom State : Nat → Type

/-- Probability that a computational-basis measurement of qubit `q` yields 0. -/
axiom p0 : {n : Nat} → State n → Fin n → R

/-- Projective measurement of `q` with a forced outcome, renormalised. -/
axiom project : {n : Nat} → State n → Fin n → Bool → State n

/-- Pauli `X` on qubit `q`. -/
axiom applyX : {n : Nat} → State n → Fin n → State n

/-- Purity of the reduced density matrix on `q`: `R.one` iff `q` is unentangled
    from the rest of the register. -/
axiom reducedPurity : {n : Nat} → State n → Fin n → R

/-- Equality **up to global phase** — the right notion for Reset. The channel
    reaches `|0⟩` via a measurement whose branch is RNG-chosen, and the branches
    differ by a global sign. Comparing amplitudes directly reports ~1.667 on
    `|−⟩`, a phase artifact rather than a discrepancy. -/
axiom EqUpToPhase : {n : Nat} → State n → State n → Prop

/-- Probability that a computational-basis measurement of qubit `q` yields 1. -/
axiom p1 : {n : Nat} → State n → Fin n → R

/-- The forced outcome `b` has nonzero probability on `s`, i.e. `project s q b`
    has something to renormalise. **Added 2026-09-30**: without it T1 and T2
    are false in any model where `project` is measure-then-renormalise. On
    `|0⟩` — unentangled, so `Deterministic` holds — the outcome-`true` branch
    has probability 0, `project s q true` is the zero vector, and no
    renormalised state has `p0 = R.one` there. The RNG-driven CPU backend never
    takes a zero-probability branch and the RNG-free backends take the
    `p0 ≥ 1/2` one, so this hypothesis is exactly what every implementation
    already relies on; it was missing from the statement, not from the code. -/
def Possible {n : Nat} (s : State n) (q : Fin n) (b : Bool) : Prop :=
  (if b then p1 s q else p0 s q) ≠ R.zero

/-- **Reset as a channel**: measure `q`, then `X` if the outcome was 1. -/
noncomputable def reset {n : Nat} (s : State n) (q : Fin n) (outcome : Bool) : State n :=
  if outcome then applyX (project s q true) q else project s q false

/-- `q` is unentangled from the rest of the register. -/
def Deterministic {n : Nat} (s : State n) (q : Fin n) : Prop :=
  reducedPurity s q = R.one

/-- **T1 — outcome-independent on an unentangled qubit.** Why an RNG-free
    backend may take the `p0 ≥ 1/2` branch and still agree with an RNG-driven
    CPU run, and why the CPU's purity criterion (not CUDA's `p0 ∈ {0,1}`) is
    the correct one. Quantified over every pair of POSSIBLE outcomes rather
    than the fixed pair `(true, false)`: on `|0⟩` the `true` branch does not
    exist, and the claim the backends rely on is "whichever branch you can
    take, the answer is the same". -/
theorem reset_outcome_irrelevant {n : Nat} (s : State n) (q : Fin n)
    (h : Deterministic s q) (b b' : Bool)
    (hb : Possible s q b) (hb' : Possible s q b') :
    EqUpToPhase (reset s q b) (reset s q b') := by
  sorry

/-- **T2 — Reset lands in `|0⟩`.** The coherent fold violates this on `|−⟩`,
    where it produces the zero vector. -/
theorem reset_yields_zero {n : Nat} (s : State n) (q : Fin n) (b : Bool)
    (h : Deterministic s q) (hb : Possible s q b) :
    p0 (reset s q b) q = R.one := by
  sorry

/-- The superseded coherent fold. -/
axiom fold : {n : Nat} → State n → Fin n → State n

/-- **T3 — the fold is NOT the reset channel.** `|−⟩` is the witness. -/
theorem fold_is_not_reset :
    ∃ (n : Nat) (s : State n) (q : Fin n) (b : Bool),
      ¬ EqUpToPhase (fold s q) (reset s q b) := by
  sorry

/-- **T4 — an entangled Reset has no pure-state result.** Why both backends
    refuse it in analytic mode rather than returning a plausible pure state.
    No `Possible` hypothesis here: an entangled `q` has both outcomes
    possible, and proving that is part of the target. -/
theorem entangled_reset_not_pure {n : Nat} (s : State n) (q : Fin n)
    (h : ¬ Deterministic s q) :
    ∀ b : Bool, ¬ EqUpToPhase (reset s q b) (reset s q (!b)) := by
  sorry

end Verification.Backend.Reset
