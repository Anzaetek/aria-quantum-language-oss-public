/-
  Reset semantics — specification targets for the omega statevector backends.

  Status: **specification, not yet proved.** Every `theorem` below is stated
  against the model and left as a target (`sorry`); the `axiom`s are the trust
  boundary this file assumes rather than derives. They exist so the Rust fix
  has a machine-checkable meaning to be discharged against later, not so we can
  claim a proof today.

  Motivation (measured, 2026-08-05): `omega-backend-statevector-metal`
  implemented Reset as a COHERENT FOLD (`new0 = old0 + old1`, renormalise)
  while the CPU backend had been corrected to a channel. The fold is wrong even
  on an unentangled qubit — on `|−⟩` it yields `old0 + old1 = 0`, and the
  `norm_sq > 0` guard then skipped renormalisation and left a zero register.
  Metal also lacked the CPU's entanglement guard, so it returned a pure state
  where the truth is mixed.
-/

namespace Verification.Backend.Reset

/-- A pure state on `n` qubits, as an amplitude vector indexed by `Fin (2^n)`. -/
axiom State : Nat → Type

/-- Probability that a computational-basis measurement of qubit `q` yields 0. -/
axiom p0 : {n : Nat} → State n → Fin n → Real

/-- Projective measurement of qubit `q` with a forced outcome `b`, renormalised.
    Undefined (junk) when the corresponding probability is 0. -/
axiom project : {n : Nat} → State n → Fin n → Bool → State n

/-- Pauli `X` on qubit `q`. -/
axiom applyX : {n : Nat} → State n → Fin n → State n

/-- Purity of the reduced density matrix on qubit `q`: `1` iff `q` is
    unentangled from the rest of the register. -/
axiom reducedPurity : {n : Nat} → State n → Fin n → Real

/-- Two states are equal **up to global phase**. This is the right notion of
    equality for Reset: the channel reaches `|0⟩` via a measurement whose branch
    is RNG-chosen, and the two branches differ by a global sign. Comparing
    amplitudes directly reports a difference of ~1.667 on `|−⟩`, which is a
    phase artifact and not a discrepancy. -/
axiom EqUpToPhase : {n : Nat} → State n → State n → Prop

/-- **Reset as a channel**: measure `q`, then apply `X` if the outcome was 1.
    This is the CPU backend's `apply_reset`, and now Metal's. -/
noncomputable def reset {n : Nat} (s : State n) (q : Fin n) (outcome : Bool) : State n :=
  if outcome then applyX (project s q true) q else project s q false

/-- The qubit is unentangled from the rest of the register. -/
def Deterministic {n : Nat} (s : State n) (q : Fin n) : Prop :=
  reducedPurity s q = 1

/-- **T1 — Reset is outcome-independent on an unentangled qubit.**

    Both measurement branches land on the same state up to global phase, which
    is why a backend with no RNG (Metal, whose fused path carries none) may pick
    the `p0 ≥ 1/2` branch and still agree with an RNG-driven CPU run. -/
theorem reset_outcome_irrelevant {n : Nat} (s : State n) (q : Fin n)
    (h : Deterministic s q) :
    EqUpToPhase (reset s q true) (reset s q false) := by
  sorry

/-- **T2 — Reset lands in `|0⟩` on the reset qubit.** The defining property;
    the coherent fold violates it on `|−⟩`, where it produces the zero vector. -/
theorem reset_yields_zero {n : Nat} (s : State n) (q : Fin n) (b : Bool)
    (h : Deterministic s q) :
    p0 (reset s q b) q = 1 := by
  sorry

/-- **T3 — the coherent fold is NOT the reset channel.** Stated as a
    non-theorem: there exists a state on which `fold` and `reset` disagree even
    up to phase. `|−⟩` is the witness — `old0 + old1 = 0`. -/
axiom fold : {n : Nat} → State n → Fin n → State n

theorem fold_is_not_reset :
    ∃ (n : Nat) (s : State n) (q : Fin n) (b : Bool),
      ¬ EqUpToPhase (fold s q) (reset s q b) := by
  sorry

/-- **T4 — an entangled Reset has no pure-state result.** The justification for
    both backends REFUSING it in analytic mode rather than returning a
    plausible pure state. -/
theorem entangled_reset_not_pure {n : Nat} (s : State n) (q : Fin n)
    (h : ¬ Deterministic s q) :
    ∀ b : Bool, ¬ EqUpToPhase (reset s q b) (reset s q (!b)) := by
  sorry

end Verification.Backend.Reset
