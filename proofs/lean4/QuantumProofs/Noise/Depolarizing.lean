import Mathlib
import QuantumProofs.Noise.Channel
import QuantumProofs.Noise.Pauli

namespace QuantumProofs.Noise

open Matrix
open scoped ComplexOrder

variable {d : ℕ}

/-- For a real scalar `c` and a Hermitian `P`, conjugation by `c • P` scales by `c²`:
`(c•P) ρ (c•P)ᴴ = c² • (P ρ P)`. -/
theorem smul_invol_apply (c : ℝ) {P : Matrix (Fin 2) (Fin 2) ℂ} (hP : Pᴴ = P)
    (ρ : Matrix (Fin 2) (Fin 2) ℂ) :
    ((c : ℂ) • P) * ρ * ((c : ℂ) • P)ᴴ = ((c ^ 2 : ℝ) : ℂ) • (P * ρ * P) := by
  have hstar : star ((c : ℝ) : ℂ) = ((c : ℝ) : ℂ) := by simp
  rw [conjTranspose_smul, hstar, hP, smul_mul_assoc, smul_mul_assoc, mul_smul_comm, smul_smul]
  congr 1
  push_cast; ring

/-- For a real scalar `c` and a Hermitian involution `P` (`Pᴴ=P`, `P*P=1`),
`(c•P)ᴴ (c•P) = c² • 1`. -/
theorem conjT_smul_invol (c : ℝ) {P : Matrix (Fin 2) (Fin 2) ℂ} (hP : Pᴴ = P)
    (hPP : P * P = 1) :
    ((c : ℂ) • P)ᴴ * ((c : ℂ) • P) = ((c ^ 2 : ℝ) : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  have hstar : star ((c : ℝ) : ℂ) = ((c : ℝ) : ℂ) := by simp
  rw [conjTranspose_smul, hstar, hP, smul_mul_assoc, mul_smul_comm, hPP, smul_smul]
  congr 1
  push_cast; ring

/-- The four Kraus operators of the single-qubit depolarizing channel at rate `p`:
`√(1-3p/4)·I, √(p/4)·X, √(p/4)·Y, √(p/4)·Z`. -/
noncomputable def depolarizingOps (p : ℝ) : Fin 4 → Matrix (Fin 2) (Fin 2) ℂ :=
  ![((Real.sqrt (1 - 3 * p / 4) : ℝ) : ℂ) • 1,
    ((Real.sqrt (p / 4) : ℝ) : ℂ) • σX,
    ((Real.sqrt (p / 4) : ℝ) : ℂ) • σY,
    ((Real.sqrt (p / 4) : ℝ) : ℂ) • σZ]

set_option linter.unusedSimpArgs false in
/-- The single-qubit **depolarizing channel** at rate `p ∈ [0,1]`, as a genuine CPTP
Kraus map (the completeness relation is discharged from `0 ≤ p ≤ 1`). -/
noncomputable def depolarizing (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) : KrausMap 2 where
  k := 4
  ops := depolarizingOps p
  complete := by
    have ha : Real.sqrt (1 - 3 * p / 4) ^ 2 = 1 - 3 * p / 4 := Real.sq_sqrt (by linarith)
    have hb : Real.sqrt (p / 4) ^ 2 = p / 4 := Real.sq_sqrt (by linarith)
    rw [Fin.sum_univ_four]
    simp only [depolarizingOps, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons, Matrix.cons_val_two, Matrix.cons_val_three, Matrix.tail_cons]
    rw [conjT_smul_invol _ conjTranspose_one (one_mul 1),
      conjT_smul_invol _ σX_herm σX_sq, conjT_smul_invol _ σY_herm σY_sq,
      conjT_smul_invol _ σZ_herm σZ_sq, ← add_smul, ← add_smul, ← add_smul]
    simp only [ha, hb]
    have hsum : ((1 - 3 * p / 4 : ℝ) : ℂ) + ((p / 4 : ℝ) : ℂ) + ((p / 4 : ℝ) : ℂ)
        + ((p / 4 : ℝ) : ℂ) = 1 := by push_cast; ring
    rw [hsum, one_smul]

set_option linter.unusedSimpArgs false in
/-- **Closed form of the depolarizing channel.**  `E_p(ρ) = (1-p)·ρ + (p/2)·(Tr ρ)·1`. -/
theorem depolarizing_apply (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (ρ : Matrix (Fin 2) (Fin 2) ℂ) :
    (depolarizing p hp hp1).apply ρ
      = ((1 - p : ℝ) : ℂ) • ρ + ((p / 2 : ℝ) : ℂ) • (ρ.trace • (1 : Matrix (Fin 2) (Fin 2) ℂ)) := by
  have ha : Real.sqrt (1 - 3 * p / 4) ^ 2 = 1 - 3 * p / 4 := Real.sq_sqrt (by linarith)
  have hb : Real.sqrt (p / 4) ^ 2 = p / 4 := Real.sq_sqrt (by linarith)
  have hS : σX * ρ * σX + σY * ρ * σY + σZ * ρ * σZ
      = (2 * ρ.trace) • (1 : Matrix (Fin 2) (Fin 2) ℂ) - ρ := by
    rw [eq_sub_iff_add_eq,
      show σX * ρ * σX + σY * ρ * σY + σZ * ρ * σZ + ρ
        = ρ + σX * ρ * σX + σY * ρ * σY + σZ * ρ * σZ from by abel]
    exact pauli_twirl ρ
  rw [show (depolarizing p hp hp1).apply ρ
        = ∑ i : Fin 4, depolarizingOps p i * ρ * (depolarizingOps p i)ᴴ from rfl, Fin.sum_univ_four]
  simp only [depolarizingOps, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.cons_val_two, Matrix.cons_val_three, Matrix.tail_cons]
  rw [smul_invol_apply _ conjTranspose_one, smul_invol_apply _ σX_herm,
    smul_invol_apply _ σY_herm, smul_invol_apply _ σZ_herm,
    Matrix.one_mul, Matrix.mul_one, ha, hb]
  -- now: a²•ρ + b²•(XρX) + b²•(YρY) + b²•(ZρZ) = RHS, with a²,b² literal reals
  rw [show ((1 - 3 * p / 4 : ℝ) : ℂ) • ρ + ((p / 4 : ℝ) : ℂ) • (σX * ρ * σX)
        + ((p / 4 : ℝ) : ℂ) • (σY * ρ * σY) + ((p / 4 : ℝ) : ℂ) • (σZ * ρ * σZ)
      = ((1 - 3 * p / 4 : ℝ) : ℂ) • ρ
          + ((p / 4 : ℝ) : ℂ) • (σX * ρ * σX + σY * ρ * σY + σZ * ρ * σZ) from by
    rw [smul_add, smul_add]; abel]
  rw [hS, smul_sub, smul_smul]
  match_scalars <;> ring

/-- `⟨ψ|ρ|ψ⟩`, the Born expectation / fidelity of operator `ρ` against pure state `ψ`. -/
noncomputable def expVal (ψ : Fin 2 → ℂ) (ρ : Matrix (Fin 2) (Fin 2) ℂ) : ℂ :=
  star ψ ⬝ᵥ (ρ *ᵥ ψ)

/-- Expectation of the pure density `|ψ⟩⟨ψ|` in state `ψ` equals `⟨ψ|ψ⟩²`. -/
theorem pure_exp (ψ : Fin 2 → ℂ) :
    star ψ ⬝ᵥ (vecMulVec ψ (star ψ) *ᵥ ψ) = (star ψ ⬝ᵥ ψ) * (star ψ ⬝ᵥ ψ) := by
  simp only [dotProduct, Matrix.mulVec, Matrix.vecMulVec_apply, Fin.sum_univ_two]
  ring

/-- Trace of the pure density `|ψ⟩⟨ψ|` equals `⟨ψ|ψ⟩`. -/
theorem trace_pure (ψ : Fin 2 → ℂ) :
    (vecMulVec ψ (star ψ)).trace = star ψ ⬝ᵥ ψ := by
  rw [trace_vecMulVec, dotProduct_comm]

/-- **Depolarizing fidelity law (the LE3 headline).**  For a normalized single-qubit
state `ψ` (`⟨ψ|ψ⟩ = 1`), the fidelity of the depolarized ideal output against the ideal
state is exactly `1 - p/2`:  `⟨ψ| E_p(|ψ⟩⟨ψ|) |ψ⟩ = 1 - p/2`. -/
theorem depolarizing_fidelity (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (ψ : Fin 2 → ℂ) (hψ : star ψ ⬝ᵥ ψ = 1) :
    expVal ψ ((depolarizing p hp hp1).apply (vecMulVec ψ (star ψ))) = ((1 - p / 2 : ℝ) : ℂ) := by
  unfold expVal
  rw [depolarizing_apply, add_mulVec, smul_mulVec, smul_mulVec, dotProduct_add,
    dotProduct_smul, dotProduct_smul, pure_exp, smul_mulVec, one_mulVec, dotProduct_smul,
    trace_pure, hψ]
  simp only [smul_eq_mul, mul_one]
  push_cast; ring

/-- The real-valued depolarizing fidelity `F(p) = 1 - p/2` (the modulus of
`depolarizing_fidelity`, which is real for a normalized state). -/
noncomputable def depoFidelity (p : ℝ) : ℝ := 1 - p / 2

/-- The proven channel fidelity is exactly the real fidelity `depoFidelity p`, as a complex
number — bridging `depolarizing_fidelity` to the real-valued threshold below. -/
theorem depolarizing_fidelity_eq (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (ψ : Fin 2 → ℂ) (hψ : star ψ ⬝ᵥ ψ = 1) :
    expVal ψ ((depolarizing p hp hp1).apply (vecMulVec ψ (star ψ)))
      = ((depoFidelity p : ℝ) : ℂ) := by
  rw [depolarizing_fidelity p hp hp1 ψ hψ]; rfl

/-- **Robustness threshold.**  The depolarized fidelity meets a target `τ` iff the noise
rate stays below `2(1-τ)` — a monotone, numerically-checkable knee (the LE3 "phase
transition" point), here at `F = τ ⟺ p = 2(1-τ)`. -/
theorem depolarizing_threshold (p τ : ℝ) :
    depoFidelity p ≥ τ ↔ p ≤ 2 * (1 - τ) := by
  unfold depoFidelity
  constructor <;> intro h <;> linarith

end QuantumProofs.Noise
