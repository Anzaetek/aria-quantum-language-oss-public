/-
  The Pauli group algorithm the stabilizer backend runs, written in Lean and
  **verified** — not axiomatised.

  Every declaration below is executable and every theorem is proved (no `sorry`,
  no `native_decide`, core Lean only). This is the algebra `omega-backend-pauli`
  implements in `stabilizer.rs`: one-qubit Paulis as `(x, z)` bit pairs in the
  `i^{xz} X^x Z^z` convention, multiplication by XOR, and the
  Aaronson–Gottesman phase `g`.

  Why this one first: a wrong `g` is what put **1000/1000 shots on
  zero-probability bitstrings** through the public API (the `X·Z`/`Z·X` rows
  were inverted, in two copies, one of which survived the first fix). The
  properties proved here — associativity of the phase (the cocycle condition),
  `P² = I`, and symplectic anticommutation — are exactly what make the pivoted
  GF(2) elimination in `stabilizer_expectation` sound.
-/

set_option autoImplicit false

namespace Verification.Backend.PauliAlgebra

/-- A one-qubit Pauli as `(x, z)`: `I = (0,0)`, `X = (1,0)`, `Z = (0,1)`,
    `Y = (1,1)`, in the `i^{xz} X^x Z^z` convention. -/
abbrev P1 := Bool × Bool

/-- Multiplication on the `(x, z)` part: componentwise XOR. -/
def mul (a b : P1) : P1 := (xor a.1 b.1, xor a.2 b.2)

/-- The Aaronson–Gottesman phase: the power of `i` picked up by `a * b`, mod 4.

    `g = x₁z₁ + x₂z₂ + 2·z₁x₂ − (x₁⊕x₂)(z₁⊕z₂)`, reduced mod 4. -/
def g (a b : P1) : Nat :=
  let bit : Bool → Int := fun c => if c then 1 else 0
  let n : Int :=
    bit a.1 * bit a.2 + bit b.1 * bit b.2
      + 2 * (bit a.2 * bit b.1)
      - bit (xor a.1 b.1) * bit (xor a.2 b.2)
  ((n % 4 + 4) % 4).toNat

/-- The table `stabilizer.rs::pauli_mult_phase` ships, after the fix. -/
def gTable (a b : P1) : Nat :=
  match a, b with
  | (false, false), _ => 0
  | _, (false, false) => 0
  | (true, false), (false, true) => 3   -- X·Z
  | (false, true), (true, false) => 1   -- Z·X
  | (true, false), (true, true)  => 1
  | (true, true),  (true, false) => 3
  | (false, true), (true, true)  => 3
  | (true, true),  (false, true) => 1
  | _, _ => 0

/-- Symplectic inner product — `true` iff `a` and `b` anticommute. -/
def anticommutes (a b : P1) : Bool :=
  xor (a.1 && b.2) (a.2 && b.1)

section Proved

/-- **The shipped table is the closed form.** All 16 pairs; the pre-fix table
    (with `X·Z = 1`, `Z·X = 3`) is rejected by this. -/
theorem gTable_eq_g : ∀ a b : P1, gTable a b = g a b := by
  intro ⟨x1, z1⟩ ⟨x2, z2⟩
  cases x1 <;> cases z1 <;> cases x2 <;> cases z2 <;> decide

/-- **`P² = I`** — squaring is the identity on the bit part. -/
theorem mul_self : ∀ a : P1, mul a a = (false, false) := by
  intro ⟨x, z⟩; cases x <;> cases z <;> decide

/-- **`mul` is associative.** The elimination multiplies a running target by a
    sequence of generators; without this the order would matter. -/
theorem mul_assoc : ∀ a b c : P1, mul (mul a b) c = mul a (mul b c) := by
  intro ⟨x1,z1⟩ ⟨x2,z2⟩ ⟨x3,z3⟩
  cases x1 <;> cases z1 <;> cases x2 <;> cases z2 <;> cases x3 <;> cases z3 <;> decide

/-- **`mul` is commutative on the bit part** — Paulis commute up to the phase,
    which is what `g` accounts for. -/
theorem mul_comm : ∀ a b : P1, mul a b = mul b a := by
  intro ⟨x1,z1⟩ ⟨x2,z2⟩; cases x1 <;> cases z1 <;> cases x2 <;> cases z2 <;> decide

/-- **The COCYCLE CONDITION — the key soundness property.**

    `g a b + g (a·b) c ≡ g b c + g a (b·c)  (mod 4)`

    The elimination folds a product left-to-right, accumulating phases as it
    goes. This says the accumulated phase does not depend on the association —
    i.e. the running sign the algorithm carries is well defined. An inverted
    row in `g` breaks this, which is why the bug corrupted multi-generator
    products (≥3 rows) while leaving single products intact. -/
theorem g_cocycle :
    ∀ a b c : P1, (g a b + g (mul a b) c) % 4 = (g b c + g a (mul b c)) % 4 := by
  intro ⟨x1,z1⟩ ⟨x2,z2⟩ ⟨x3,z3⟩
  cases x1 <;> cases z1 <;> cases x2 <;> cases z2 <;> cases x3 <;> cases z3 <;> decide

/-- **Anticommutation is symmetric** — the `for k in 0..n { if anticommutes }`
    scan gives the same verdict whichever operand is the observable. -/
theorem anticommutes_symm : ∀ a b : P1, anticommutes a b = anticommutes b a := by
  intro ⟨x1,z1⟩ ⟨x2,z2⟩; cases x1 <;> cases z1 <;> cases x2 <;> cases z2 <;> decide

/-- **Commuting Paulis have an EVEN phase.** `mul_pauli_row` only ever flips the
    sign on `phase % 4 == 2`, silently ignoring an odd phase. That is sound
    precisely because its operands commute — this is the proof. -/
theorem g_even_of_commuting :
    ∀ a b : P1, anticommutes a b = false → g a b % 2 = 0 := by
  intro ⟨x1,z1⟩ ⟨x2,z2⟩; cases x1 <;> cases z1 <;> cases x2 <;> cases z2 <;> decide

/-- **Anticommuting Paulis have an ODD phase** — the converse, so the parity of
    `g` *characterises* commutation rather than merely following from it. -/
theorem g_odd_of_anticommuting :
    ∀ a b : P1, anticommutes a b = true → g a b % 2 = 1 := by
  intro ⟨x1,z1⟩ ⟨x2,z2⟩; cases x1 <;> cases z1 <;> cases x2 <;> cases z2 <;> decide

end Proved

end Verification.Backend.PauliAlgebra
