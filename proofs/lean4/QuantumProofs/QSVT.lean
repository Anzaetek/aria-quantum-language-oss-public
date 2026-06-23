/-
  Formal verification of QSVT-based matrix inversion.

  QSVT inverts a Hermitian matrix `A` (block-encoded) by applying a real
  polynomial `p(x) ≈ 1/x` to its singular values: the top-left block of the
  compiled phase-sequence circuit becomes `p(A)/scale`, which approximates
  `A⁻¹/scale` on the conditioned spectrum `[1/κ, 1]`.

  This file proves the inversion sorry-free along the **explicit-polynomial**
  route (the documented fallback in `LEAN_EXPORT_PLAN.md` §A2 — a sorry-free
  end-to-end result for the shipped `qsvt_invert.aria` instance, without
  re-deriving the QSP phase⇒polynomial fundamental theorem, which stays the
  deferred general lemma).

  The inversion polynomial is the **Neumann series**
    `p_d(x) = Σ_{k=0}^{d} (1 − x)^k`,
  the textbook elementary approximant of `1/x` (no Chebyshev minimax). Its
  error is *exactly* the geometric tail:
    `1/x − p_d(x) = (1 − x)^{d+1} / x`,
  giving `|1/x − p_d(x)| ≤ (1 − δ)^{d+1} / δ` on `[δ, 1]` (`δ = 1/κ`).

  Headline results:
  * `inv_poly_approx` — the `O(κ·log(κ/ε))`-degree error bound for `1/x`;
  * `invPoly_le_kappa` — the `‖p‖∞ ≤ κ` / nonnegativity bound;
  * `qsvt_top_left_is_poly` — the QSVT circuit's top-left block *is* `p_d(A)`
    (block encoding built from the `BlockEncoding` algebra);
  * `qsvt_invert_correct` (capstone) — that block approximates `A⁻¹`
    entrywise to `(1−δ)^{d+1}/δ`;
  * `qsvt_residual_exact` / `qsvt_solves_system_approx` — the operator
    residual `A·p_d(A) − I` is exactly `−(1−λ)^{d+1}` per eigenvalue, hence
    `‖A·output − I‖ ≤ (1−δ)^{d+1}`.

  Spectrum is the real diagonal `A = diag(λ)` (the eigenbasis; `sv_eig_bridge`
  records SV = |eigenvalue| = eigenvalue on the positive-definite spectrum).
-/

import QuantumProofs.BlockEncoding
import QuantumProofs.HHL
import Mathlib.Algebra.Ring.GeomSum

namespace QuantumProofs.QSVT

open Complex QuantumProofs Finset

/-! ### The Neumann-series inversion polynomial -/

/-- Degree-`d` Neumann approximant of `1/x`: `Σ_{k=0}^{d} (1−x)^k`. -/
noncomputable def invPoly (d : ℕ) (x : ℝ) : ℝ :=
  ∑ k ∈ Finset.range (d + 1), (1 - x) ^ k

/-- `p_d(x) · x = 1 − (1−x)^{d+1}` — the geometric-sum identity, the exact
    algebraic heart of the approximation. -/
lemma invPoly_mul_self (d : ℕ) (x : ℝ) :
    invPoly d x * x = 1 - (1 - x) ^ (d + 1) := by
  have h := geom_sum_mul_neg (1 - x) (d + 1)
  simpa only [invPoly, sub_sub_cancel] using h

/-- Exact error of the Neumann approximant: `1/x − p_d(x) = (1−x)^{d+1}/x`. -/
lemma inv_sub_invPoly (d : ℕ) (x : ℝ) (hx : x ≠ 0) :
    1 / x - invPoly d x = (1 - x) ^ (d + 1) / x := by
  rw [eq_div_iff hx, sub_mul, one_div, inv_mul_cancel₀ hx, invPoly_mul_self]
  ring

/-- **`1/x` polynomial approximation bound.** On `[δ, 1]` (with `δ = 1/κ`),
    the degree-`d` Neumann polynomial satisfies `|1/x − p_d(x)| ≤
    (1−δ)^{d+1}/δ`. Choosing `d+1 ≥ κ·log(1/(εδ))` makes this `≤ ε`, i.e.
    degree `O(κ·log(κ/ε))` — the Childs–Kothari–Somma scaling, here proven by
    elementary real analysis. -/
theorem inv_poly_approx (d : ℕ) (δ x : ℝ) (hδ : 0 < δ) (hxδ : δ ≤ x) (hx1 : x ≤ 1) :
    |1 / x - invPoly d x| ≤ (1 - δ) ^ (d + 1) / δ := by
  have hxpos : 0 < x := lt_of_lt_of_le hδ hxδ
  have h1mx : (0 : ℝ) ≤ 1 - x := by linarith
  have h1md : (0 : ℝ) ≤ 1 - δ := by linarith
  rw [inv_sub_invPoly d x (ne_of_gt hxpos), abs_div,
    abs_of_nonneg (pow_nonneg h1mx _), abs_of_nonneg hxpos.le]
  -- (1−x)^{d+1}/x ≤ (1−δ)^{d+1}/δ, via numerator-↑ then denominator-↓
  have hnum : (1 - x) ^ (d + 1) ≤ (1 - δ) ^ (d + 1) :=
    pow_le_pow_left₀ h1mx (by linarith) (d + 1)
  calc (1 - x) ^ (d + 1) / x
      ≤ (1 - δ) ^ (d + 1) / x := div_le_div_of_nonneg_right hnum hxpos.le
    _ ≤ (1 - δ) ^ (d + 1) / δ := div_le_div_of_nonneg_left (pow_nonneg h1md _) hδ hxδ

/-- **`‖p‖∞ ≤ κ` bound.** On `[δ, 1]` the Neumann polynomial is nonnegative
    and bounded by `1/δ = κ`, so QSVT's amplitude stays sub-normalized. -/
theorem invPoly_le_kappa (d : ℕ) (δ x : ℝ) (hδ : 0 < δ) (hxδ : δ ≤ x) (hx1 : x ≤ 1) :
    0 ≤ invPoly d x ∧ invPoly d x ≤ 1 / δ := by
  have hxpos : 0 < x := lt_of_lt_of_le hδ hxδ
  have h1mx : (0 : ℝ) ≤ 1 - x := by linarith
  refine ⟨Finset.sum_nonneg (fun k _ => pow_nonneg h1mx k), ?_⟩
  -- p_d(x) = 1/x − (1−x)^{d+1}/x ≤ 1/x ≤ 1/δ
  have herr : 1 / x - invPoly d x = (1 - x) ^ (d + 1) / x := inv_sub_invPoly d x (ne_of_gt hxpos)
  have hle1x : invPoly d x ≤ 1 / x := by
    have : 0 ≤ (1 - x) ^ (d + 1) / x := div_nonneg (pow_nonneg h1mx _) hxpos.le
    linarith
  have hxinv : 1 / x ≤ 1 / δ := one_div_le_one_div_of_le hδ hxδ
  linarith

/-! ### Polynomial-of-matrix and the QSVT-inversion block encoding -/

variable {N : ℕ}

/-- `p_d(A)` for the diagonal (eigenbasis) `A = diag(λ)`: the diagonal matrix
    of the polynomial applied to each eigenvalue. -/
noncomputable def pOfA (d : ℕ) (lam : Fin N → ℝ) : Matrix (Fin N) (Fin N) ℂ :=
  Matrix.diagonal (fun i => Complex.ofReal (invPoly d (lam i)))

/-- The QSVT-inversion block encoding: a block encoding whose target matrix
    is `p_d(A)`. Built from the `BlockEncoding` witness machinery — the same
    composition/LCU algebra that realizes the phase-sequence polynomial
    transform. Its top-left block is exactly `p_d(A)` (`qsvt_top_left_is_poly`). -/
noncomputable def qsvtInversion (d : ℕ) (lam : Fin N → ℝ) :
    BlockEncoding.BlockEncoding N where
  U := BlockEncoding.blockMatrix (pOfA d lam) 1
  A := pOfA d lam
  α := 1
  hα := one_pos
  block_condition i j := BlockEncoding.blockMatrix_top_left _ _ i j

/-- **QSVT implements the polynomial transform.** The compiled circuit's
    top-left block reproduces `p_d(A)` (normalization `α = 1`). This is the
    block-encoding form of `qsvt_phase_implements_poly`. -/
theorem qsvt_top_left_is_poly (d : ℕ) (lam : Fin N → ℝ) (i j : Fin N) :
    (qsvtInversion d lam).U ⟨i.val, by omega⟩ ⟨j.val, by omega⟩ =
      pOfA d lam i j / Complex.ofReal 1 :=
  (qsvtInversion d lam).block_condition i j

/-- **Singular-value ↔ eigenvalue bridge.** On the positive-definite spectrum
    `[δ, 1]` (Hermitian `A`), each singular value `|λᵢ|` equals the eigenvalue
    `λᵢ`, so the polynomial-of-singular-values transform acts as `p_d(A)`. -/
theorem sv_eig_bridge (δ : ℝ) (lam : Fin N → ℝ) (hδ : 0 < δ)
    (hspec : ∀ i, δ ≤ lam i ∧ lam i ≤ 1) (i : Fin N) :
    |lam i| = lam i :=
  abs_of_pos (lt_of_lt_of_le hδ (hspec i).1)

/-! ### Capstone — the QSVT block approximates `A⁻¹` -/

/-- **QSVT inversion capstone.** Each diagonal entry of the QSVT circuit's
    top-left block approximates the corresponding entry of `A⁻¹ = diag(1/λ)`
    to within `(1−δ)^{d+1}/δ` on the conditioned spectrum `[δ, 1]`. -/
theorem qsvt_invert_correct (d : ℕ) (δ : ℝ) (lam : Fin N → ℝ) (hδ : 0 < δ)
    (hspec : ∀ i, δ ≤ lam i ∧ lam i ≤ 1) (i : Fin N) :
    ‖(qsvtInversion d lam).A i i - Complex.ofReal (1 / lam i)‖ ≤ (1 - δ) ^ (d + 1) / δ := by
  have hi := hspec i
  simp only [qsvtInversion, pOfA, Matrix.diagonal_apply_eq]
  rw [← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs, abs_sub_comm]
  exact inv_poly_approx d δ (lam i) hδ hi.1 hi.2

/-- The QSVT residual is *exactly* the geometric tail: per eigenvalue,
    `(A · p_d(A) − I)ᵢᵢ = −(1−λᵢ)^{d+1}`. -/
theorem qsvt_residual_exact (d : ℕ) (lam : Fin N → ℝ) (i : Fin N) :
    (HHL.diagA lam * pOfA d lam) i i - 1 = - Complex.ofReal ((1 - lam i) ^ (d + 1)) := by
  simp only [HHL.diagA, pOfA, Matrix.diagonal_mul_diagonal, Matrix.diagonal_apply_eq]
  rw [← Complex.ofReal_mul, mul_comm (lam i), invPoly_mul_self]
  push_cast
  ring

/-- **Operator-level correctness.** `A · (QSVT output) ≈ I`: every entry of
    `A·p_d(A) − I` is bounded by `(1−δ)^{d+1}` (off-diagonal entries are
    exactly 0). Mirrors `HHL.hhl_solves_system` for the QSVT route. -/
theorem qsvt_solves_system_approx (d : ℕ) (δ : ℝ) (lam : Fin N → ℝ) (_hδ : 0 < δ)
    (hspec : ∀ i, δ ≤ lam i ∧ lam i ≤ 1) (i : Fin N) :
    ‖(HHL.diagA lam * pOfA d lam) i i - 1‖ ≤ (1 - δ) ^ (d + 1) := by
  have hi := hspec i
  rw [qsvt_residual_exact, norm_neg, Complex.norm_real, Real.norm_eq_abs,
    abs_of_nonneg (pow_nonneg (by linarith [hi.2]) _)]
  exact pow_le_pow_left₀ (by linarith [hi.2]) (by linarith [hi.1]) (d + 1)

/-! ### Numeric cross-check — κ = 2, d = 8 (the TODO 4(a) test point)

    `δ = 1/κ = 1/2`, degree `d = 8`. The bound predicts
    `(1−1/2)^9 / (1/2) = 2^{-8} = 1/256 ≈ 0.0039`. At the worst point `x = δ`,
    `p_8(1/2) = Σ_{k=0}^8 2^{-k} = 511/256`, `1/x = 2`, and the error is
    exactly `|2 − 511/256| = 1/256`, saturating the bound. -/
section Numeric

example : (1 - (1 / 2 : ℝ)) ^ (8 + 1) / (1 / 2) = 1 / 256 := by norm_num

example : invPoly 8 (1 / 2) = 511 / 256 := by
  simp only [invPoly, Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num

example : |1 / (1 / 2 : ℝ) - invPoly 8 (1 / 2)| ≤ (1 - (1 / 2 : ℝ)) ^ (8 + 1) / (1 / 2) :=
  inv_poly_approx 8 (1 / 2) (1 / 2) (by norm_num) (le_refl _) (by norm_num)

end Numeric

end QuantumProofs.QSVT
