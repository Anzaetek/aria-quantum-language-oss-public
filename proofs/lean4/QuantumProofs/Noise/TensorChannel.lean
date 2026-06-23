/-
  D2/LE4 — TENSOR (multi-qubit) channels: single-qubit channels acting on one factor of a
  multi-qubit register, and the factorization of fidelity across qubits (the "width" lift,
  complementing the depth-G floor in `DepolarizingDepth.lean`).

  Using mathlib's Kronecker product `⊗ₖ`, the tensor of two CPTP Kraus maps `E, F` (Kraus
  operators `{Kᵢ ⊗ Lⱼ}`) is again CPTP (`kraus_tensor_complete`: `∑ᵢⱼ (Kᵢ⊗Lⱼ)ᴴ(Kᵢ⊗Lⱼ) = 1`),
  and on a **product** operator it factorizes:

      ∑ᵢⱼ (Kᵢ⊗Lⱼ)(ρ⊗σ)(Kᵢ⊗Lⱼ)ᴴ = E(ρ) ⊗ F(σ)            (`kraus_tensor_apply_product`).

  Hence for a product input state the Born fidelity factorizes
  (`expVal_kron`: `⟨ψ⊗φ|(A⊗B)|ψ⊗φ⟩ = ⟨ψ|A|ψ⟩·⟨φ|B|φ⟩`), giving the multi-qubit headline

      ⟨ψ⊗φ| (E_p(|ψ⟩⟨ψ|) ⊗ E_p(|φ⟩⟨φ|)) |ψ⊗φ⟩ = (1 − p/2)²      (`twoQubitDepolarizing_fidelity`)

  — two qubits, each independently depolarized, on a product pure state.  This is the genuine
  tensor-structure result the single-qubit capstones deferred.

  Sorry-free; reuses `Depolarizing.lean` and mathlib's Kronecker API.
-/
import Mathlib
import QuantumProofs.Noise.Channel
import QuantumProofs.Noise.Pauli
import QuantumProofs.Noise.Depolarizing

namespace QuantumProofs.Noise

open Matrix
open scoped ComplexOrder Kronecker

variable {d₁ d₂ : ℕ}

/-- The Kronecker product of two state vectors: `(ψ ⊗ φ)(i,j) = ψ i · φ j`. -/
def kronVec {m n : Type*} (a : m → ℂ) (b : n → ℂ) : m × n → ℂ := fun p => a p.1 * b p.2

/-- Sum pulls out of the left Kronecker factor. -/
theorem kron_sum_left {K : Type*} [Fintype K] (f : K → Matrix (Fin d₁) (Fin d₁) ℂ)
    (B : Matrix (Fin d₂) (Fin d₂) ℂ) : (∑ i, f i) ⊗ₖ B = ∑ i, (f i ⊗ₖ B) := by
  ext p q
  simp only [Matrix.kroneckerMap_apply, Matrix.sum_apply, Finset.sum_mul]

/-- Sum pulls out of the right Kronecker factor. -/
theorem kron_sum_right {K : Type*} [Fintype K] (A : Matrix (Fin d₁) (Fin d₁) ℂ)
    (g : K → Matrix (Fin d₂) (Fin d₂) ℂ) : A ⊗ₖ (∑ j, g j) = ∑ j, (A ⊗ₖ g j) := by
  ext p q
  simp only [Matrix.kroneckerMap_apply, Matrix.sum_apply, Finset.mul_sum]

/-- **Tensor of two CPTP maps is CPTP.**  The Kraus operators `{Kᵢ ⊗ Lⱼ}` satisfy the
completeness relation `∑ᵢⱼ (Kᵢ⊗Lⱼ)ᴴ(Kᵢ⊗Lⱼ) = 1`. -/
theorem kraus_tensor_complete (E : KrausMap d₁) (F : KrausMap d₂) :
    ∑ i, ∑ j, (E.ops i ⊗ₖ F.ops j)ᴴ * (E.ops i ⊗ₖ F.ops j)
      = (1 : Matrix (Fin d₁ × Fin d₂) (Fin d₁ × Fin d₂) ℂ) := by
  simp_rw [conjTranspose_kronecker, ← mul_kronecker_mul, ← kron_sum_right, ← kron_sum_left,
    E.complete, F.complete, one_kronecker_one]

/-- **Tensor channel factorizes on product operators.**  For a product input `ρ ⊗ σ`, the
two-factor Kraus sum equals `E(ρ) ⊗ F(σ)`. -/
theorem kraus_tensor_apply_product (E : KrausMap d₁) (F : KrausMap d₂)
    (ρ : Matrix (Fin d₁) (Fin d₁) ℂ) (σ : Matrix (Fin d₂) (Fin d₂) ℂ) :
    (∑ i, ∑ j, (E.ops i ⊗ₖ F.ops j) * (ρ ⊗ₖ σ) * (E.ops i ⊗ₖ F.ops j)ᴴ)
      = (E.apply ρ) ⊗ₖ (F.apply σ) := by
  simp_rw [conjTranspose_kronecker, ← mul_kronecker_mul, ← kron_sum_right, ← kron_sum_left]
  rfl

/-- `star (ψ ⊗ φ) = (star ψ) ⊗ (star φ)`. -/
theorem star_kronVec {m n : Type*} (ψ : m → ℂ) (φ : n → ℂ) :
    star (kronVec ψ φ) = kronVec (star ψ) (star φ) := by
  ext p; simp only [kronVec, Pi.star_apply, star_mul']

/-- `(A ⊗ B) *ᵥ (a ⊗ b) = (A *ᵥ a) ⊗ (B *ᵥ b)`. -/
theorem kron_mulVec {m n : Type*} [Fintype m] [Fintype n] (A : Matrix m m ℂ) (B : Matrix n n ℂ)
    (a : m → ℂ) (b : n → ℂ) :
    (A ⊗ₖ B) *ᵥ kronVec a b = kronVec (A *ᵥ a) (B *ᵥ b) := by
  ext p
  simp only [Matrix.mulVec, kronVec, Matrix.kroneckerMap_apply, dotProduct, Fintype.sum_prod_type]
  rw [Finset.sum_mul_sum]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ => ?_
  ring

/-- `(a ⊗ b) ⬝ᵥ (c ⊗ d) = (a ⬝ᵥ c)·(b ⬝ᵥ d)`. -/
theorem dotProduct_kronVec {m n : Type*} [Fintype m] [Fintype n] (a c : m → ℂ) (b d : n → ℂ) :
    (kronVec a b) ⬝ᵥ (kronVec c d) = (a ⬝ᵥ c) * (b ⬝ᵥ d) := by
  simp only [dotProduct, kronVec, Fintype.sum_prod_type]
  rw [Finset.sum_mul_sum]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ => ?_
  ring

/-- **Born fidelity factorizes on product states.**  `⟨ψ⊗φ|(A⊗B)|ψ⊗φ⟩ = ⟨ψ|A|ψ⟩·⟨φ|B|φ⟩`. -/
theorem expVal_kron {m n : Type*} [Fintype m] [Fintype n] (ψ : m → ℂ) (φ : n → ℂ)
    (A : Matrix m m ℂ) (B : Matrix n n ℂ) :
    star (kronVec ψ φ) ⬝ᵥ ((A ⊗ₖ B) *ᵥ kronVec ψ φ)
      = (star ψ ⬝ᵥ (A *ᵥ ψ)) * (star φ ⬝ᵥ (B *ᵥ φ)) := by
  rw [kron_mulVec, star_kronVec, dotProduct_kronVec]

/-- **Two-qubit depolarizing fidelity (the width lift).**  Two qubits in a product pure state
`ψ ⊗ φ`, each independently depolarized at rate `p`, retain fidelity `(1 − p/2)²` — the
single-qubit fidelity raised to the qubit count. -/
theorem twoQubitDepolarizing_fidelity (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (ψ φ : Fin 2 → ℂ) (hψ : star ψ ⬝ᵥ ψ = 1) (hφ : star φ ⬝ᵥ φ = 1) :
    star (kronVec ψ φ) ⬝ᵥ
        (((depolarizing p hp hp1).apply (vecMulVec ψ (star ψ)))
          ⊗ₖ ((depolarizing p hp hp1).apply (vecMulVec φ (star φ)))
          *ᵥ kronVec ψ φ)
      = (((1 - p / 2) ^ 2 : ℝ) : ℂ) := by
  rw [expVal_kron,
    show star ψ ⬝ᵥ ((depolarizing p hp hp1).apply (vecMulVec ψ (star ψ)) *ᵥ ψ)
        = ((1 - p / 2 : ℝ) : ℂ) from depolarizing_fidelity p hp hp1 ψ hψ,
    show star φ ⬝ᵥ ((depolarizing p hp hp1).apply (vecMulVec φ (star φ)) *ᵥ φ)
        = ((1 - p / 2 : ℝ) : ℂ) from depolarizing_fidelity p hp hp1 φ hφ]
  push_cast; ring

/-- **Circulant register (3-qubit) depolarizing fidelity floor.**  The circulant `CyclicShift`
solver runs on a 3-qubit register; with each qubit independently depolarized at rate `p` on a
product input state, the register fidelity is `(1 − p/2)³` — the single-qubit fidelity raised to
the qubit count (the "width" floor, complementary to the depth-3 floor in `DepolarizingDepth`). -/
theorem threeQubitDepolarizing_fidelity (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (ψ φ χ : Fin 2 → ℂ) (hψ : star ψ ⬝ᵥ ψ = 1) (hφ : star φ ⬝ᵥ φ = 1) (hχ : star χ ⬝ᵥ χ = 1) :
    star (kronVec (kronVec ψ φ) χ) ⬝ᵥ
        ((((depolarizing p hp hp1).apply (vecMulVec ψ (star ψ)))
            ⊗ₖ ((depolarizing p hp hp1).apply (vecMulVec φ (star φ))))
          ⊗ₖ ((depolarizing p hp hp1).apply (vecMulVec χ (star χ)))
          *ᵥ kronVec (kronVec ψ φ) χ)
      = (((1 - p / 2) ^ 3 : ℝ) : ℂ) := by
  rw [expVal_kron, twoQubitDepolarizing_fidelity p hp hp1 ψ φ hψ hφ,
    show star χ ⬝ᵥ ((depolarizing p hp hp1).apply (vecMulVec χ (star χ)) *ᵥ χ)
        = ((1 - p / 2 : ℝ) : ℂ) from depolarizing_fidelity p hp hp1 χ hχ]
  push_cast; ring

end QuantumProofs.Noise
