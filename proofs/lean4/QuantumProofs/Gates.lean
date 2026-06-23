/-
  Standard quantum gate definitions as matrices.

  Each gate is defined as a Matrix (Fin 2) (Fin 2) ℂ (or larger for multi-qubit gates).
  These definitions serve as the ground truth for circuit semantics proofs.
-/

import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

namespace QuantumProofs.Gates

open Complex Matrix

/-- Hadamard gate: H = (1/√2) [[1, 1], [1, -1]] -/
noncomputable def H : Matrix (Fin 2) (Fin 2) ℂ :=
  let s := (1 : ℂ) / Complex.ofReal (Real.sqrt 2)
  !![s, s; s, -s]

/-- Scaled Hadamard (no √2): HCore = [[1, 1], [1, -1]]
    We have H = (1/√2) * HCore, so H*H = (1/2) * HCore*HCore.
    If HCore*HCore = 2*I, then H*H = I. -/
def HCore : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1, 1; 1, -1]

/-- Pauli-X (NOT) gate: X = [[0, 1], [1, 0]] -/
def X : Matrix (Fin 2) (Fin 2) ℂ :=
  !![0, 1; 1, 0]

/-- Pauli-Y gate: Y = [[0, -i], [i, 0]] -/
def Y : Matrix (Fin 2) (Fin 2) ℂ :=
  !![0, -I; I, 0]

/-- Pauli-Z gate: Z = [[1, 0], [0, -1]] -/
def Z : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1, 0; 0, -1]

/-- Phase gate: S = [[1, 0], [0, i]] -/
def S : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1, 0; 0, I]

/-- T gate: T = [[1, 0], [0, e^{iπ/4}]] -/
noncomputable def T : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1, 0; 0, exp (I * Complex.ofReal (Real.pi / 4))]

/-- Rotation around Z axis: RZ(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]] -/
noncomputable def RZ (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![exp (-I * Complex.ofReal (θ/2)), 0;
     0, exp (I * Complex.ofReal (θ/2))]

/-- Rotation around Y axis: RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]] -/
noncomputable def RY (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![Complex.ofReal (Real.cos (θ/2)), -Complex.ofReal (Real.sin (θ/2));
     Complex.ofReal (Real.sin (θ/2)), Complex.ofReal (Real.cos (θ/2))]

/-- CNOT gate (4×4): [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]] -/
def CNOT : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 0, 0, 0;
     0, 1, 0, 0;
     0, 0, 0, 1;
     0, 0, 1, 0]

-- ===========================================================================
-- Gate properties: self-inverse proofs
-- ===========================================================================

/-- X is self-inverse: X * X = I -/
theorem X_self_inverse : X * X = (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [X, mul_apply, Fin.sum_univ_two] <;> ring

/-- Y is self-inverse: Y * Y = I -/
theorem Y_self_inverse : Y * Y = (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Y, mul_apply, Fin.sum_univ_two, Complex.I_sq] <;> ring

/-- Z is self-inverse: Z * Z = I -/
theorem Z_self_inverse : Z * Z = (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Z, mul_apply, Fin.sum_univ_two] <;> ring

-- ===========================================================================
-- Gate relations
-- ===========================================================================

/-- X * Z = -Z * X (anti-commutation) -/
theorem XZ_anti_commute : X * Z = -(Z * X) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [X, Z, mul_apply, Fin.sum_univ_two] <;> ring

/-- S * S = Z -/
theorem S_squared_eq_Z : S * S = Z := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [S, Z, mul_apply, Fin.sum_univ_two, Complex.I_sq] <;> ring

/-- CNOT is self-inverse -/
theorem CNOT_self_inverse : CNOT * CNOT = (1 : Matrix (Fin 4) (Fin 4) ℂ) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [CNOT, mul_apply, Fin.sum_univ_four] <;> ring

-- ===========================================================================
-- Hadamard proofs via scaled HCore
-- ===========================================================================

/-- Key lemma: HCore * HCore = 2 * I -/
theorem HCore_squared : HCore * HCore = (2 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [HCore, mul_apply, Fin.sum_univ_two, Matrix.smul_apply] <;> ring

/-- HCore * X * HCore = 2 * Z -/
theorem HCore_X_HCore : HCore * X * HCore = (2 : ℂ) • Z := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [HCore, X, Z, mul_apply, Fin.sum_univ_two, Matrix.smul_apply] <;> ring

/-- HCore * Z * HCore = 2 * X -/
theorem HCore_Z_HCore : HCore * Z * HCore = (2 : ℂ) • X := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [HCore, X, Z, mul_apply, Fin.sum_univ_two, Matrix.smul_apply] <;> ring

/-- (√2 : ℂ) squared is 2. Packaged as a lemma so the H proofs stay short. -/
private lemma sqrt2_sq_C :
    Complex.ofReal (Real.sqrt 2) * Complex.ofReal (Real.sqrt 2) = (2 : ℂ) := by
  rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by norm_num : (2:ℝ) ≥ 0)]
  norm_num

/-- (√2 : ℂ) is nonzero, needed for `field_simp`. -/
private lemma sqrt2_ne_zero_C :
    Complex.ofReal (Real.sqrt 2) ≠ 0 := by
  simp [Complex.ofReal_eq_zero, Real.sqrt_eq_zero']

/-- H expressed as the scaled HCore matrix. -/
private lemma H_eq_scaled :
    H = ((1 : ℂ) / Complex.ofReal (Real.sqrt 2)) • HCore := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [H, HCore, Matrix.smul_apply, neg_div]

/-- H * H = I (self-inverse). Derived from HCore² = 2·I and (1/√2)² = 1/2. -/
theorem H_self_inverse : H * H = (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  rw [H_eq_scaled, Matrix.smul_mul, Matrix.mul_smul, smul_smul, HCore_squared,
      smul_smul]
  have h2 : ((1 : ℂ) / Complex.ofReal (Real.sqrt 2)) *
            ((1 : ℂ) / Complex.ofReal (Real.sqrt 2)) * 2 = 1 := by
    have hsq : Complex.ofReal (Real.sqrt 2) ^ 2 = 2 := by
      rw [sq]; exact sqrt2_sq_C
    field_simp
    rw [hsq]
  rw [h2, one_smul]

/-- H * X * H = Z (basis change) via HCore * X * HCore = 2 * Z and scalar (1/2). -/
theorem HXH_eq_Z : H * X * H = Z := by
  rw [H_eq_scaled, Matrix.smul_mul, Matrix.mul_smul, Matrix.smul_mul, smul_smul,
      HCore_X_HCore, smul_smul]
  have h2 : ((1 : ℂ) / Complex.ofReal (Real.sqrt 2)) *
            ((1 : ℂ) / Complex.ofReal (Real.sqrt 2)) * 2 = 1 := by
    have hsq : Complex.ofReal (Real.sqrt 2) ^ 2 = 2 := by
      rw [sq]; exact sqrt2_sq_C
    field_simp
    rw [hsq]
  rw [h2, one_smul]

/-- H * Z * H = X (basis change) — dual of HXH. -/
theorem HZH_eq_X : H * Z * H = X := by
  rw [H_eq_scaled, Matrix.smul_mul, Matrix.mul_smul, Matrix.smul_mul, smul_smul,
      HCore_Z_HCore, smul_smul]
  have h2 : ((1 : ℂ) / Complex.ofReal (Real.sqrt 2)) *
            ((1 : ℂ) / Complex.ofReal (Real.sqrt 2)) * 2 = 1 := by
    have hsq : Complex.ofReal (Real.sqrt 2) ^ 2 = 2 := by
      rw [sq]; exact sqrt2_sq_C
    field_simp
    rw [hsq]
  rw [h2, one_smul]

end QuantumProofs.Gates
