import Mathlib

namespace DeferProbe
open Matrix
noncomputable section

abbrev M2 := Matrix (Fin 2) (Fin 2) ℂ

/-- The measurement channel on one qubit with the outcome DISCARDED:
    `ρ ↦ |0⟩⟨0| ρ |0⟩⟨0| + |1⟩⟨1| ρ |1⟩⟨1|`, i.e. kill the off-diagonals. -/
def deph (ρ : M2) : M2 := !![ρ 0 0, 0; 0, ρ 1 1]

def P0 : M2 := !![1, 0; 0, 0]
def P1 : M2 := !![0, 0; 0, 1]

/-- The closed form IS the Kraus sum. -/
theorem deph_eq_kraus (ρ : M2) : deph ρ = P0 * ρ * P0 + P1 * ρ * P1 := by
  -- Destructure ρ into literal 2x2 form so both sides are concrete.
  rw [Matrix.eta_fin_two ρ]
  simp [deph, P0, P1]

/-- Kraus completeness: `∑ Kᵢᴴ Kᵢ = 1`, so the channel is trace-preserving. -/
theorem kraus_complete : P0ᴴ * P0 + P1ᴴ * P1 = (1 : M2) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [P0, P1, Matrix.mul_apply, Fin.sum_univ_two,
          Matrix.conjTranspose_apply]

/-- **(★) THE CAPSTONE.** The channel is self-adjoint under the
    Hilbert–Schmidt pairing: measuring the STATE and dephasing the OBSERVABLE
    give the same expectation. This is what licenses deferring the measurement
    and dephasing the observable instead. -/
theorem trace_deph_left_eq_right (ρ O : M2) :
    Matrix.trace (deph ρ * O) = Matrix.trace (ρ * deph O) := by
  rw [Matrix.eta_fin_two ρ, Matrix.eta_fin_two O]
  simp [deph, Matrix.trace, Fin.sum_univ_two]

/-- (†) X is killed. -/
theorem deph_X : deph !![0, 1; 1, 0] = 0 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [deph]

/-- (†) Y is killed. -/
theorem deph_Y : deph !![0, -Complex.I; Complex.I, 0] = 0 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [deph]

/-- (†) Z survives. -/
theorem deph_Z : deph !![1, 0; 0, -1] = !![1, 0; 0, -1] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [deph]

/-- (†) I survives. -/
theorem deph_I : deph (1 : M2) = 1 := by
  ext i j; fin_cases i <;> fin_cases j <;>
    simp [deph]

end
end DeferProbe
