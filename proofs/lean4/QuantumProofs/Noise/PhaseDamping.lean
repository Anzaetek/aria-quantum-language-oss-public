/-
  D2 — single-qubit PHASE-DAMPING (pure dephasing, T₂) channel (LEAN_ERROR_PLAN.md LE3).

  Loss of quantum coherence without energy exchange.  In the Pauli-mixture representation the
  Kraus operators are `√(1−λ/2)·I` and `√(λ/2)·Z`, so

      E_λ(ρ) = (1−λ/2)·ρ + (λ/2)·ZρZ,

  a genuine CPTP map (`phaseDamping λ : KrausMap 2`).  Its action keeps the populations and
  scales the off-diagonal coherences by `(1−λ)`:

      E_λ(ρ) = [[a, (1−λ)·b],[(1−λ)·c, d]]   for ρ = [[a,b],[c,d]].

  Two headline laws:
    * **populations / `⟨Z⟩` preserved** — `⟨Z⟩(E_λ ρ) = ⟨Z⟩(ρ)` (`phaseDamping_expZ`),
      the defining feature of pure dephasing (no relaxation, unlike amplitude damping);
    * **coherence decay** — `(E_λ ρ)₀₁ = (1−λ)·ρ₀₁` (`phaseDamping_coherence`), the off-diagonal
      `(1−λ)` shrink that drives `T₂` decay.

  Sorry-free; reuses the Hermitian-involution conjugation lemmas from `Depolarizing.lean`.
-/
import Mathlib
import QuantumProofs.Noise.Channel
import QuantumProofs.Noise.Pauli
import QuantumProofs.Noise.Depolarizing

namespace QuantumProofs.Noise

open Matrix
open scoped ComplexOrder

set_option linter.unusedSimpArgs false

/-- The two Kraus operators of the phase-damping channel at rate `λ`:
`√(1−λ/2)·I, √(λ/2)·Z`. -/
noncomputable def phaseDampingOps (l : ℝ) : Fin 2 → Matrix (Fin 2) (Fin 2) ℂ :=
  ![((Real.sqrt (1 - l / 2) : ℝ) : ℂ) • 1, ((Real.sqrt (l / 2) : ℝ) : ℂ) • σZ]

/-- The single-qubit **phase-damping channel** at rate `λ ∈ [0,1]`, as a CPTP Kraus map. -/
noncomputable def phaseDamping (l : ℝ) (hl : 0 ≤ l) (hl1 : l ≤ 1) : KrausMap 2 where
  k := 2
  ops := phaseDampingOps l
  complete := by
    have ha : Real.sqrt (1 - l / 2) ^ 2 = 1 - l / 2 := Real.sq_sqrt (by linarith)
    have hb : Real.sqrt (l / 2) ^ 2 = l / 2 := Real.sq_sqrt (by linarith)
    rw [Fin.sum_univ_two]
    simp only [phaseDampingOps, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    rw [conjT_smul_invol _ conjTranspose_one (one_mul 1), conjT_smul_invol _ σZ_herm σZ_sq,
      ← add_smul]
    simp only [ha, hb]
    have hsum : ((1 - l / 2 : ℝ) : ℂ) + ((l / 2 : ℝ) : ℂ) = 1 := by push_cast; ring
    rw [hsum, one_smul]

/-- **Closed form of the phase-damping channel.**  `E_λ(ρ) = (1−λ/2)·ρ + (λ/2)·ZρZ`. -/
theorem phaseDamping_apply (l : ℝ) (hl : 0 ≤ l) (hl1 : l ≤ 1)
    (ρ : Matrix (Fin 2) (Fin 2) ℂ) :
    (phaseDamping l hl hl1).apply ρ
      = ((1 - l / 2 : ℝ) : ℂ) • ρ + ((l / 2 : ℝ) : ℂ) • (σZ * ρ * σZ) := by
  have ha : Real.sqrt (1 - l / 2) ^ 2 = 1 - l / 2 := Real.sq_sqrt (by linarith)
  have hb : Real.sqrt (l / 2) ^ 2 = l / 2 := Real.sq_sqrt (by linarith)
  rw [show (phaseDamping l hl hl1).apply ρ
        = ∑ i : Fin 2, phaseDampingOps l i * ρ * (phaseDampingOps l i)ᴴ from rfl, Fin.sum_univ_two]
  simp only [phaseDampingOps, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
  rw [smul_invol_apply _ conjTranspose_one, smul_invol_apply _ σZ_herm, Matrix.one_mul,
    Matrix.mul_one, ha, hb]

/-- **Phase damping preserves populations / `⟨Z⟩`.**  `⟨Z⟩(E_λ ρ) = ⟨Z⟩(ρ)` — pure dephasing
does not relax the diagonal (contrast amplitude damping, where `⟨Z⟩` shifts by `2γ·d`). -/
theorem phaseDamping_expZ (l : ℝ) (hl : 0 ≤ l) (hl1 : l ≤ 1)
    (ρ : Matrix (Fin 2) (Fin 2) ℂ) :
    expZ ((phaseDamping l hl hl1).apply ρ) = expZ ρ := by
  rw [phaseDamping_apply, expZ_eq, expZ_eq]
  simp only [σZ, Matrix.add_apply, Matrix.smul_apply, Matrix.mul_apply, Fin.sum_univ_two,
    Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.empty_val', Matrix.cons_val_fin_one, smul_eq_mul]
  push_cast; ring

/-- **Coherence decay (the T₂ law).**  The off-diagonal coherence shrinks by `(1−λ)`:
`(E_λ ρ)₀₁ = (1−λ)·ρ₀₁`. -/
theorem phaseDamping_coherence (l : ℝ) (hl : 0 ≤ l) (hl1 : l ≤ 1)
    (ρ : Matrix (Fin 2) (Fin 2) ℂ) :
    ((phaseDamping l hl hl1).apply ρ) 0 1 = ((1 - l : ℝ) : ℂ) * ρ 0 1 := by
  rw [phaseDamping_apply]
  simp only [σZ, Matrix.add_apply, Matrix.smul_apply, Matrix.mul_apply, Fin.sum_univ_two,
    Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.empty_val', Matrix.cons_val_fin_one, smul_eq_mul]
  push_cast; ring

end QuantumProofs.Noise
