/-
  Stabilizer expectation values — specification targets for
  `omega-backend-pauli::stabilizer_expectation`.

  Status: **specification, not yet proved.** `theorem`s are targets (`sorry`);
  `axiom`s are the assumed trust boundary. The one exception is
  `gPhase_correct`, which is a finite case check and IS proved by `decide`.

  Motivation (measured, 2026-08-05): two defects, both returning a legal-looking
  number instead of failing.

  1. The group-membership test was a single greedy pass with no pivoting,
     described in its own comment as Gaussian elimination. When it failed to
     reduce a genuine group element it fell through to `0.0` under "in the
     normalizer but not the group" — a failed ALGORITHM reported as physics.
     `⟨P⟩ = 0` is a legal expectation, so nothing looked wrong.
  2. `pauli_mult_phase` had the `X·Z` and `Z·X` rows inverted against the
     Aaronson–Gottesman `g` function.

  Witness: a Steane-encoded 2-qubit logical Grover circuit gave
  `⟨Z̄(patch 0)⟩ = 0.000000` while the statevector and Pauli-propagation
  backends both gave `+1.000000`.
-/

namespace Verification.Backend.StabilizerExpectation

/-- A Pauli operator on one qubit in the `i^{xz} X^x Z^z` convention the
    tableau rows use: `(false,false) = I`, `(true,false) = X`,
    `(false,true) = Z`, `(true,true) = Y`. -/
abbrev P1 := Bool × Bool

/-- The Aaronson–Gottesman `g` function: the power of `i` picked up when the
    per-qubit Pauli `a` is multiplied by `b`, mod 4.

    Closed form for this convention:
    `g = x₁z₁ + x₂z₂ + 2·z₁x₂ − (x₁⊕x₂)(z₁⊕z₂)  (mod 4)`. -/
def gClosed (a b : P1) : Nat :=
  let (x1, z1) := a
  let (x2, z2) := b
  let n : Int :=
    (if x1 && z1 then 1 else 0) + (if x2 && z2 then 1 else 0)
      + 2 * (if z1 && x2 then 1 else 0)
      - (if (xor x1 x2) && (xor z1 z2) then 1 else 0)
  ((n % 4 + 4) % 4).toNat

/-- The table the Rust `pauli_mult_phase` implements, after the fix. -/
def gTable (a b : P1) : Nat :=
  match a, b with
  | (false, false), _ => 0
  | _, (false, false) => 0
  | (true, false), (false, true) => 3   -- X·Z  ⇒ −1   (was wrongly 1)
  | (false, true), (true, false) => 1   -- Z·X  ⇒ +1   (was wrongly 3)
  | (true, false), (true, true)  => 1
  | (true, true),  (true, false) => 3
  | (false, true), (true, true)  => 3
  | (true, true),  (false, true) => 1
  | _, _ => 0

/-- **PROVED (not a target).** The shipped table equals the closed form on all
    16 input pairs. This is the defect-2 fix, machine-checked: the pre-fix table
    had `X·Z = 1` and `Z·X = 3`, which `decide` would reject. -/
theorem gPhase_correct : ∀ a b : P1, gTable a b = gClosed a b := by
  intro ⟨x1, z1⟩ ⟨x2, z2⟩
  cases x1 <;> cases z1 <;> cases x2 <;> cases z2 <;> decide

/-- An `n`-qubit stabilizer state. -/
axiom StabState : Nat → Type
/-- An `n`-qubit Pauli operator (sign + per-qubit letters). -/
axiom Pauli : Nat → Type

/-- `P` anticommutes with some stabilizer generator of `s`. -/
axiom Anticommutes : {n : Nat} → StabState n → Pauli n → Prop
/-- `P` is a member of the stabilizer group of `s`, with `+` sign. -/
axiom InGroupPlus : {n : Nat} → StabState n → Pauli n → Prop
/-- `P` is a member with `−` sign. -/
axiom InGroupMinus : {n : Nat} → StabState n → Pauli n → Prop

/-- `⟨ψ|P|ψ⟩`. -/
axiom expectation : {n : Nat} → StabState n → Pauli n → Real

/-- **T1 — the trichotomy.** For a stabilizer state and a Pauli, the
    expectation is exactly one of `0`, `+1`, `−1`, determined by membership.
    This is what the backend must compute; the greedy reduction returned `0`
    for members of the group, i.e. it broke the second and third cases. -/
theorem expectation_trichotomy {n : Nat} (s : StabState n) (p : Pauli n) :
    (Anticommutes s p → expectation s p = 0)
    ∧ (InGroupPlus s p → expectation s p = 1)
    ∧ (InGroupMinus s p → expectation s p = -1) := by
  sorry

/-- **T2 — `0` is reserved for anticommutation.** The precise statement the
    greedy reduction violated: a commuting Pauli on a full-rank `n`-generator
    stabilizer state IS in the group up to sign, so `⟨P⟩ = 0` can only happen
    by anticommuting. Any other route to `0` is an implementation failure being
    reported as physics. -/
theorem zero_only_when_anticommuting {n : Nat} (s : StabState n) (p : Pauli n)
    (h : expectation s p = 0) :
    Anticommutes s p := by
  sorry

/-- **T3 — elimination is complete.** Reducing `P` against a row-echelon basis
    of the generators terminates at the identity whenever `P` is in the group.
    This is the property the single greedy pass lacked, and the reason a real
    pivoted elimination was required. -/
axiom reducesToIdentity : {n : Nat} → StabState n → Pauli n → Prop

theorem echelon_reduction_complete {n : Nat} (s : StabState n) (p : Pauli n)
    (h : ¬ Anticommutes s p) :
    reducesToIdentity s p := by
  sorry

end Verification.Backend.StabilizerExpectation
