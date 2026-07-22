/-
  RBS (Reconfigurable Beam Splitter / Givens rotation) gate — the
  primitive of butterfly QNN circuits (Kerenidis et al., arXiv:2606.03517).

      RBS(θ) = exp(−i·θ/2·(Y⊗X − X⊗Y))
             = [[1, 0,      0,     0],
                [0, cos θ, −sin θ, 0],
                [0, sin θ,  cos θ, 0],
                [0, 0,      0,     1]]

  Cross-referenced by `crates/omega-backend-statevector/src/gates.rs::rbs`
  (and `::drbs`). Proved here, sorry-free:

  * `rbs_unitary`          — RBS(θ)ᴴ · RBS(θ) = 1 (the `@assert unitary`
                             obligation of `examples/aria/butterfly_qnn.aria`);
  * `rbs_add`              — RBS(θ₁)·RBS(θ₂) = RBS(θ₁+θ₂) (one-parameter
                             group; justifies rotation merging + the
                             `inv @` rule RBS(θ)⁻¹ = RBS(−θ));
  * `drbs_eq_neg_i_g_rbs`  — dRBS(θ) = (−i) · G · RBS(θ) with the
                             generator G = (Y⊗X − X⊗Y)/2 restricted to
                             the 4-dim two-qubit space — the identity the
                             adjoint-AD derivative matrix `drbs` and the
                             4-term Givens parameter-shift rule
                             (frequencies {1, 2}) rest on;
  * `g_spectrum_squares`   — G³ = G (spectrum ⊆ {0, ±1}), the algebraic
                             fact behind the {1, 2} shift-rule frequencies.
-/

import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.LinearAlgebra.Matrix.Notation

namespace QuantumProofs.Rbs

open Complex Matrix

/-- RBS(θ): identity on {|00⟩, |11⟩}, planar rotation on span{|01⟩, |10⟩}. -/
noncomputable def rbs (θ : ℝ) : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 0, 0, 0;
     0, (Real.cos θ : ℂ), (-(Real.sin θ) : ℂ), 0;
     0, (Real.sin θ : ℂ), (Real.cos θ : ℂ), 0;
     0, 0, 0, 1]

/-- Entrywise derivative dRBS/dθ (what `gates.rs::drbs` implements). -/
noncomputable def drbs (θ : ℝ) : Matrix (Fin 4) (Fin 4) ℂ :=
  !![0, 0, 0, 0;
     0, (-(Real.sin θ) : ℂ), (-(Real.cos θ) : ℂ), 0;
     0, (Real.cos θ : ℂ), (-(Real.sin θ) : ℂ), 0;
     0, 0, 0, 0]

/-- The generator G = (Y⊗X − X⊗Y)/2 on the two-qubit space:
    zero on {|00⟩, |11⟩}, the Pauli-Y pattern on span{|01⟩, |10⟩}. -/
def G : Matrix (Fin 4) (Fin 4) ℂ :=
  !![0, 0, 0, 0;
     0, 0, -Complex.I, 0;
     0, Complex.I, 0, 0;
     0, 0, 0, 0]

/-- RBS(θ) is unitary: RBS(θ)ᴴ · RBS(θ) = 1. -/
theorem rbs_unitary (θ : ℝ) : (rbs θ)ᴴ * rbs θ = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rbs, Matrix.mul_apply, Fin.sum_univ_four, Matrix.conjTranspose_apply,
      -Complex.ofReal_cos, -Complex.ofReal_sin, Complex.conj_ofReal] <;>
    norm_cast <;>
    nlinarith [Real.sin_sq_add_cos_sq θ]

/-- One-parameter group: RBS(θ₁) · RBS(θ₂) = RBS(θ₁ + θ₂). -/
theorem rbs_add (θ₁ θ₂ : ℝ) : rbs θ₁ * rbs θ₂ = rbs (θ₁ + θ₂) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rbs, Matrix.mul_apply, Fin.sum_univ_four, Real.cos_add, Real.sin_add] <;>
    ring

/-- Generator form of the derivative: dRBS(θ) = (−i) · G · RBS(θ).
    This is the matrix-level identity behind the adjoint-AD derivative
    and the 4-term Givens parameter-shift rule of arXiv:2606.03517. -/
theorem drbs_eq_neg_i_g_rbs (θ : ℝ) : drbs θ = (-Complex.I) • (G * rbs θ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [drbs, G, rbs, Matrix.smul_apply] <;>
    ring_nf <;>
    simp [Complex.I_sq]

/-- G³ = G — the generator's spectrum is contained in {0, ±1}, which is
    why RBS gradients carry the two frequencies {1, 2} and need the
    4-term shift rule (π/4, π/2 shifts), not the 2-term ±π/2 rule. -/
theorem g_spectrum_squares : G * G * G = G := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [G, Matrix.mul_apply, Fin.sum_univ_four]

end QuantumProofs.Rbs
