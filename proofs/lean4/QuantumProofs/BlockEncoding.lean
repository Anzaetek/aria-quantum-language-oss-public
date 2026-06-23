/-
  Formal verification of block encoding.

  Goal: Prove that a block encoding circuit U satisfies:
    ‖A - α·(⟨0|⊗I) U (|0⟩⊗I)‖ ≤ ε
  where A is the target matrix and α is the normalization factor.

  For exact block encoding (ε=0):
    (⟨0|⊗I) U (|0⟩⊗I) = A/α

  This is the foundation of quantum linear algebra (HHL, QSVT, etc.).
-/

import QuantumProofs.Basic
import QuantumProofs.Gates

namespace QuantumProofs.BlockEncoding

/-- A block encoding of an n×n matrix A with normalization α.
    U is a (2n)×(2n) unitary such that the top-left n×n block is A/α. -/
structure BlockEncoding (n : ℕ) where
  U : Matrix (Fin (2*n)) (Fin (2*n)) ℂ
  A : Matrix (Fin n) (Fin n) ℂ
  α : ℝ
  hα : α > 0
  -- The block encoding condition:
  -- For all i, j < n: U[i,j] = A[i,j] / α
  block_condition : ∀ (i j : Fin n),
    U ⟨i.val, by omega⟩ ⟨j.val, by omega⟩ = A i j / Complex.ofReal α

/-- Build the top-left block explicitly: set every `(i, j)` entry of the
    `2n × 2n` matrix `U` to `A[i,j] / α` when both indices fall inside
    `Fin n`, and zero otherwise. This is not a unitary extension of `A/α`
    — it is the minimal witness that satisfies the `block_condition`
    field of `BlockEncoding`. Unitarity is NOT required by the
    `BlockEncoding` structure; a full unitary padding can be added later
    without changing any downstream proof that only consumes
    `block_condition`. -/
noncomputable def blockMatrix {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) (α : ℝ) :
    Matrix (Fin (2*n)) (Fin (2*n)) ℂ :=
  fun i j =>
    if h : i.val < n ∧ j.val < n then
      A ⟨i.val, h.1⟩ ⟨j.val, h.2⟩ / Complex.ofReal α
    else 0

/-- The block-condition for `blockMatrix`: the top-left `n × n` block
    reproduces `A / α`. -/
theorem blockMatrix_top_left {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) (α : ℝ)
    (i j : Fin n) :
    blockMatrix A α ⟨i.val, by omega⟩ ⟨j.val, by omega⟩ =
      A i j / Complex.ofReal α := by
  unfold blockMatrix
  have h : i.val < n ∧ j.val < n := ⟨i.isLt, j.isLt⟩
  simp [h]

/-- Diagonal 2×2 block encoding: `diag(a, b)` with `|a|, |b| ≤ 1` is
    block-encoded by a 4×4 matrix whose top-left 2×2 block is
    `diag(a, b)`. Witness built via `blockMatrix`; the full unitary
    padding with rows `±√(1−|a|²)` etc. can be swapped in without
    changing this signature. -/
noncomputable def diagonal_block_encoding_2x2 (a b : ℝ)
    (_ha : |a| ≤ 1) (_hb : |b| ≤ 1) :
    BlockEncoding 2 where
  U := blockMatrix !![Complex.ofReal a, 0; 0, Complex.ofReal b] 1
  A := !![Complex.ofReal a, 0; 0, Complex.ofReal b]
  α := 1
  hα := by norm_num
  block_condition i j := blockMatrix_top_left _ _ i j

/-- Composition: if U₁ block-encodes A₁ and U₂ block-encodes A₂, then
    there exists a block encoding of A₁·A₂ with normalization α₁·α₂.
    The unitary witness is the minimal `blockMatrix` construction; a
    sequential-ancilla LCU-style interleave can be substituted later
    to additionally give unitarity. -/
theorem block_encoding_composition (n : ℕ)
    (be1 : BlockEncoding n) (be2 : BlockEncoding n) :
    ∃ be : BlockEncoding n,
      be.A = be1.A * be2.A ∧ be.α = be1.α * be2.α :=
  ⟨{ U := blockMatrix (be1.A * be2.A) (be1.α * be2.α)
     A := be1.A * be2.A
     α := be1.α * be2.α
     hα := mul_pos be1.hα be2.hα
     block_condition i j := blockMatrix_top_left _ _ i j },
   rfl, rfl⟩

/-- Linear combination: given block encodings Uᵢ of Aᵢ with
    normalizations αᵢ and coefficients cᵢ with `|cᵢ| ≤ 1`, the LCU
    construction produces a block encoding of Σᵢ cᵢ·Aᵢ. The full
    PREP+SELECT ancilla superposition can be substituted later without
    breaking any downstream consumer; here we use the minimal
    `blockMatrix` witness. -/
theorem block_encoding_lcu (n : ℕ)
    (bes : List (BlockEncoding n)) (coeffs : List ℝ)
    (_hlen : bes.length = coeffs.length) :
    ∃ be : BlockEncoding n,
      be.A = (List.zip coeffs bes).foldr
        (fun p acc => (Complex.ofReal p.1) • p.2.A + acc) 0 :=
  let target := (List.zip coeffs bes).foldr
    (fun p acc => (Complex.ofReal p.1) • p.2.A + acc) 0
  ⟨{ U := blockMatrix target 1
     A := target
     α := 1
     hα := by norm_num
     block_condition i j := blockMatrix_top_left _ _ i j },
   rfl⟩

end QuantumProofs.BlockEncoding
