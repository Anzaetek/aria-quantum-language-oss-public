/-
  Basic definitions for quantum computing proofs.

  Defines the state space, unitary matrices, and measurement axioms
  that underpin the circuit semantics.
-/

import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Data.Complex.Basic

namespace QuantumProofs

/-- A quantum state on n qubits is a unit vector in ℂ^(2^n). -/
def QState (n : ℕ) := { v : Fin (2^n) → ℂ // ∑ i, ‖v i‖^2 = 1 }

/-- A unitary operator on n qubits. -/
def Unitary (n : ℕ) := Matrix (Fin (2^n)) (Fin (2^n)) ℂ

-- TODO: Add unitarity constraint (U† U = I)

/-- The computational basis state |k⟩ for k < 2^n. -/
def basis_state (n : ℕ) (k : Fin (2^n)) : Fin (2^n) → ℂ :=
  fun i => if i = k then 1 else 0

/-- The all-zeros state |0...0⟩. -/
def zero_state (n : ℕ) : Fin (2^n) → ℂ :=
  basis_state n ⟨0, Nat.two_pow_pos n⟩

/-- Tensor product of two state vectors. -/
def tensor_product {m n : ℕ}
    (ψ : Fin (2^m) → ℂ) (φ : Fin (2^n) → ℂ)
    : Fin (2^(m+n)) → ℂ :=
  fun i =>
    let a : Fin (2^m) := ⟨i.val / 2^n,
      by
        have hi : i.val < 2^m * 2^n := (pow_add 2 m n) ▸ i.isLt
        exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hi)⟩
    let b : Fin (2^n) := ⟨i.val % 2^n, Nat.mod_lt _ (Nat.two_pow_pos n)⟩
    ψ a * φ b

/-- A square complex matrix is *unitary* iff `U† · U = I`. Used by
    every spec-extraction `@assert unitary` obligation; lifted here
    out of `Generated/` so closed proofs share one definition. -/
def is_unitary {N : ℕ} (U : Matrix (Fin N) (Fin N) ℂ) : Prop :=
  U.conjTranspose * U = 1

/-- Apply a matrix `U` to a state vector `ψ`: `(Uψ)ᵢ = Σⱼ Uᵢⱼ · ψⱼ`.
    Mirrors `Grover.apply_unitary` but indexed by an arbitrary
    `Fin N`, so it also fits the `2^n` and the small-N (extracted-
    template) shapes uniformly. Spec-extraction obligations of the
    form `apply (denote circuit) (basis_state n k) = …` resolve
    against this definition. -/
noncomputable def apply {N : ℕ} (U : Matrix (Fin N) (Fin N) ℂ) (ψ : Fin N → ℂ) :
    Fin N → ℂ :=
  fun i => ∑ j, U i j * ψ j

end QuantumProofs
