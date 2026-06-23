/-
  D2 — single-qubit Pauli algebra (support for the depolarizing-noise fidelity law, LE3).

  The three Pauli matrices over `Matrix (Fin 2) (Fin 2) ℂ`, with the algebraic facts the
  depolarizing channel needs: they are Hermitian involutions (`pauli_herm`, `pauli_sq`), and
  the **Pauli twirl identity**

      ρ + XρX + YρY + ZρZ = 2 (Tr ρ) • 1        (`pauli_twirl`)

  which collapses the depolarizing operator-sum to the closed form `(1-p)ρ + (p/2)(Tr ρ)·1`.

  Sorry-free; pure 2×2 entrywise computation.
-/
import Mathlib

namespace QuantumProofs.Noise

open Matrix

/-- Pauli `X`. -/
def σX : Matrix (Fin 2) (Fin 2) ℂ := !![0, 1; 1, 0]
/-- Pauli `Y`. -/
def σY : Matrix (Fin 2) (Fin 2) ℂ := !![0, -Complex.I; Complex.I, 0]
/-- Pauli `Z`. -/
def σZ : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, -1]

theorem σX_herm : σXᴴ = σX := by ext i j; fin_cases i <;> fin_cases j <;> simp [σX]
theorem σY_herm : σYᴴ = σY := by ext i j; fin_cases i <;> fin_cases j <;> simp [σY]
theorem σZ_herm : σZᴴ = σZ := by ext i j; fin_cases i <;> fin_cases j <;> simp [σZ]

theorem σX_sq : σX * σX = 1 := by ext i j; fin_cases i <;> fin_cases j <;> simp [σX]
theorem σY_sq : σY * σY = 1 := by ext i j; fin_cases i <;> fin_cases j <;> simp [σY]
theorem σZ_sq : σZ * σZ = 1 := by ext i j; fin_cases i <;> fin_cases j <;> simp [σZ]

-- The 16-fold entrywise `mul_fin_two`/`ring_nf` expansion over ℂ needs more than the
-- default heartbeat budget; the proof is a finite computation, not a search.
set_option linter.style.maxHeartbeats false in
set_option maxHeartbeats 800000 in
set_option linter.unusedSimpArgs false in
/-- **Pauli twirl.** For any single-qubit operator, `ρ + XρX + YρY + ZρZ = 2 (Tr ρ) • 1`.
This is the identity that turns the depolarizing Kraus sum into a closed form. -/
theorem pauli_twirl (ρ : Matrix (Fin 2) (Fin 2) ℂ) :
    ρ + σX * ρ * σX + σY * ρ * σY + σZ * ρ * σZ
      = (2 * ρ.trace) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  rw [eta_fin_two ρ]
  simp only [σX, σY, σZ, mul_fin_two, one_fin_two, trace_fin_two_of]
  ext i j
  fin_cases i <;> fin_cases j <;>
    (simp only [Fin.mk_zero, Fin.mk_one, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
       Matrix.cons_val_one, Matrix.head_cons, Matrix.empty_val', Matrix.cons_val_fin_one,
       Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
     ring_nf
     rw [Complex.I_sq]
     ring)

/-- `⟨Z⟩(ρ) = Tr(Z·ρ)`, the Pauli-Z expectation of operator `ρ`. -/
noncomputable def expZ (ρ : Matrix (Fin 2) (Fin 2) ℂ) : ℂ := (σZ * ρ).trace

/-- `⟨Z⟩(ρ) = ρ₀₀ − ρ₁₁` (the population imbalance). -/
theorem expZ_eq (ρ : Matrix (Fin 2) (Fin 2) ℂ) : expZ ρ = ρ 0 0 - ρ 1 1 := by
  simp only [expZ, σZ, Matrix.trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two, Matrix.of_apply,
    Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
    Matrix.empty_val', Matrix.cons_val_fin_one]
  ring

end QuantumProofs.Noise
