/-
  Formal verification of the HHL quantum linear-solver.

  HHL prepares (up to normalization) the solution `x ∝ A⁻¹ b` of a linear
  system `A x = b` for a Hermitian `A`. The algorithm works eigenvalue-wise:
  in the eigenbasis of `A = diag(λ₀, …, λ_{N-1})` with `b = Σ βᵢ |uᵢ⟩`,

    1. QPE loads each eigenphase `λᵢ` into the counting register (exact
       loading is `QPE.qpe_exact`);
    2. a controlled `RY(2·arcsin(C/λᵢ))` rotates the ancilla so that its
       `|1⟩`-amplitude becomes `C/λᵢ`;
    3. post-selecting the ancilla on `|1⟩` and uncomputing the QPE register
       leaves the (unnormalized) state `Σ βᵢ (C/λᵢ) |uᵢ⟩ = C · A⁻¹ b`.

  This file proves the three load-bearing facts sorry-free:

  * `controlled_inv_rotation` — the RY ancilla amplitude is exactly `C/λ`
    (the trig core of step 2);
  * `hhl_solves_system` — the post-selected output `o` satisfies
    `A · o = C · b`, i.e. `o = C · A⁻¹ b` (the capstone, step 3);
  * `hhl_success_prob_lower` — post-selection succeeds with probability
    `≥ (C/Λ)²` where `Λ` bounds the spectrum (the Born-rule bound).

  We also record `hhl_qpe_eigenphase`, the thin instantiation of
  `QPE.qpe_exact` that is step 1.

  Following the repo house style (cf. `QPE.lean`, `BlockEncoding.lean`) the
  algorithm state is tracked as explicit amplitude vectors over `Fin N`;
  `A` is the genuine diagonal matrix `Matrix.diagonal (λ ·)` and the
  capstone is a genuine matrix-vector identity, so "output ∝ A⁻¹ b" is a
  theorem about the actual operator, not a restatement.
-/

import QuantumProofs.QPE
import QuantumProofs.Gates
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

namespace QuantumProofs.HHL

open Complex QuantumProofs

/-- The eigenvalue-inversion matrix `A = diag(λ₀, …, λ_{N-1})` whose
    spectrum HHL inverts. Hermitian iff the `λᵢ` are real, which they are
    here by construction. -/
noncomputable def diagA {N : ℕ} (lam : Fin N → ℝ) : Matrix (Fin N) (Fin N) ℂ :=
  Matrix.diagonal (fun i => Complex.ofReal (lam i))

/-- The (unnormalized) post-selected HHL output amplitude on eigenvector
    `|uᵢ⟩`: the input component `βᵢ` scaled by the ancilla amplitude
    `C/λᵢ`. -/
noncomputable def hhl_output {N : ℕ} (C : ℝ) (lam : Fin N → ℝ) (beta : Fin N → ℂ) :
    Fin N → ℂ :=
  fun i => beta i * (Complex.ofReal C / Complex.ofReal (lam i))

/-! ### Step 2 — controlled inverse rotation -/

/-- **Controlled inverse rotation (trig core of HHL step 2).**
    Applying `RY(2·arcsin t)` to `|0⟩` puts amplitude exactly `t` on `|1⟩`
    and `√(1−t²)` on `|0⟩`, for any `|t| ≤ 1`. With `t = C/λ` this realizes
    the `C/λ` inverse-eigenvalue amplitude HHL needs on the ancilla. -/
theorem controlled_inv_rotation (t : ℝ) (ht : |t| ≤ 1) :
    Gates.RY (2 * Real.arcsin t) 1 0 = Complex.ofReal t ∧
    Gates.RY (2 * Real.arcsin t) 0 0 = Complex.ofReal (Real.sqrt (1 - t ^ 2)) := by
  obtain ⟨ht1, ht2⟩ := abs_le.mp ht
  have hhalf : 2 * Real.arcsin t / 2 = Real.arcsin t := by ring
  constructor
  · -- entry (1,0) of RY is sin(θ/2); sin(arcsin t) = t
    simp only [Gates.RY, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.of_apply, hhalf]
    rw [Real.sin_arcsin ht1 ht2]
  · -- entry (0,0) of RY is cos(θ/2); cos(arcsin t) = √(1 − t²)
    simp only [Gates.RY, Matrix.cons_val_zero,
      Matrix.of_apply, hhalf]
    rw [Real.cos_arcsin]

/-! ### Step 1 — QPE eigenphase loading (reuse of `QPE.qpe_exact`) -/

/-- **Eigenphase loading (HHL step 1).** When the (rescaled) eigenphase is
    dyadic, `θ = m/2ⁿ`, the inverse-QFT'd QPE counting register collapses
    onto `|m⟩` with probability 1. This is exactly `QPE.qpe_exact`; HHL
    consumes it to read each eigenvalue into the counting register before
    the controlled rotation. -/
theorem hhl_qpe_eigenphase (n : ℕ) (m : Fin (2 ^ n)) (θ : ℝ)
    (hθ : θ = ↑m.val / ↑(2 ^ n)) :
    QPE.measure_prob (QPE.apply_inverse_dft n (QPE.after_controlled_u n θ)) m = 1 :=
  QPE.qpe_exact n m θ hθ

/-! ### Step 3 — capstone: the post-selected output solves `A x = C·b` -/

/-- **HHL capstone.** In the eigenbasis, the post-selected HHL output `o`
    satisfies `A · o = C · b`. Equivalently `o = C · A⁻¹ b`, i.e. the
    output is proportional to `A⁻¹ b` with proportionality constant `C` —
    the defining correctness statement of HHL, here as a genuine
    matrix-vector identity over the real diagonal operator `A = diag(λ)`. -/
theorem hhl_solves_system {N : ℕ} (C : ℝ) (lam : Fin N → ℝ) (beta : Fin N → ℂ)
    (hlam : ∀ i, lam i ≠ 0) :
    apply (diagA lam) (hhl_output C lam beta) = fun i => Complex.ofReal C * beta i := by
  funext i
  unfold apply diagA hhl_output
  -- the diagonal collapses the sum to the i-th term
  rw [Finset.sum_eq_single i]
  · have hli : (Complex.ofReal (lam i)) ≠ 0 := by
      simp only [ne_eq, Complex.ofReal_eq_zero]; exact hlam i
    rw [Matrix.diagonal_apply_eq]
    field_simp
  · intro j _ hji
    rw [Matrix.diagonal_apply_ne _ (by simpa [eq_comm] using hji)]
    ring
  · intro h; exact absurd (Finset.mem_univ i) h

/-! ### Born-rule success probability of post-selection -/

/-- The (unnormalized) probability of post-selecting the ancilla on `|1⟩`,
    summed over the eigenbasis: `Σᵢ |βᵢ|² · (C/λᵢ)²`. -/
noncomputable def success_prob {N : ℕ} (C : ℝ) (lam : Fin N → ℝ) (beta : Fin N → ℂ) : ℝ :=
  ∑ i, Complex.normSq (hhl_output C lam beta i)

/-- **HHL post-selection success bound.** If every eigenvalue satisfies
    `|λᵢ| ≤ Λ` and the input `b` is normalized (`Σ |βᵢ|² = 1`), then the
    ancilla post-selects on `|1⟩` with probability at least `(C/Λ)²`. This
    is the Born-rule lower bound that makes HHL run in `O(κ)`-ish amplitude
    amplification rounds (`Λ ≈ λ_max`, `C ≈ λ_min`). -/
theorem hhl_success_prob_lower {N : ℕ} (C : ℝ) (lam : Fin N → ℝ) (beta : Fin N → ℂ)
    (Λ : ℝ) (hΛ : 0 < Λ) (hC : 0 ≤ C) (hlam : ∀ i, lam i ≠ 0)
    (hbound : ∀ i, |lam i| ≤ Λ) (hnorm : ∑ i, Complex.normSq (beta i) = 1) :
    (C / Λ) ^ 2 ≤ success_prob C lam beta := by
  unfold success_prob hhl_output
  -- per-term lower bound: normSq(βᵢ · C/λᵢ) ≥ normSq(βᵢ) · (C/Λ)²
  have hterm : ∀ i : Fin N,
      Complex.normSq (beta i) * (C / Λ) ^ 2 ≤
        Complex.normSq (beta i * (Complex.ofReal C / Complex.ofReal (lam i))) := by
    intro i
    have hli : lam i ≠ 0 := hlam i
    have hli2 : (lam i) ^ 2 ≤ Λ ^ 2 := by
      have := hbound i
      nlinarith [abs_nonneg (lam i), sq_abs (lam i), this, hΛ.le]
    rw [Complex.normSq_mul, Complex.normSq_div, Complex.normSq_ofReal, Complex.normSq_ofReal]
    -- goal: normSq β * (C/Λ)² ≤ normSq β * (C*C/(λ*λ))
    have hbeta_nonneg : 0 ≤ Complex.normSq (beta i) := Complex.normSq_nonneg _
    apply mul_le_mul_of_nonneg_left _ hbeta_nonneg
    -- (C/Λ)² ≤ C*C/(λ*λ)
    have hCsq : (C / Λ) ^ 2 = C * C / (Λ * Λ) := by ring
    rw [hCsq]
    have hlsq : 0 < lam i * lam i := by
      rcases lt_or_gt_of_ne hli with h | h
      · exact mul_pos_of_neg_of_neg h h
      · exact mul_pos h h
    have hCC : 0 ≤ C * C := mul_nonneg hC hC
    have hll : lam i * lam i ≤ Λ * Λ := by nlinarith [hli2]
    exact div_le_div_of_nonneg_left hCC hlsq hll
  calc (C / Λ) ^ 2
      = (C / Λ) ^ 2 * 1 := by ring
    _ = (C / Λ) ^ 2 * ∑ i, Complex.normSq (beta i) := by rw [hnorm]
    _ = ∑ i, Complex.normSq (beta i) * (C / Λ) ^ 2 := by
        rw [Finset.mul_sum]; congr 1; funext i; ring
    _ ≤ ∑ i, Complex.normSq (beta i * (Complex.ofReal C / Complex.ofReal (lam i))) :=
        Finset.sum_le_sum (fun i _ => hterm i)

/-! ### Numeric cross-check — concrete 2×2 SPD system

    `A = diag(1, 2)`, `b = (1, 1)`, `C = 1`. The solver output is
    `(1, 1/2)` and `A · output = (1, 1) = C · b`; success probability is
    `1·1 + 1·(1/2)² = 5/4`. (Unnormalized `b`; matches the analytic
    `A⁻¹b = (1, 1/2)`.) -/
section Numeric

noncomputable def lam2 : Fin 2 → ℝ := ![1, 2]
noncomputable def beta2 : Fin 2 → ℂ := ![1, 1]

example : apply (diagA lam2) (hhl_output 1 lam2 beta2) = fun i => (1 : ℂ) * beta2 i :=
  hhl_solves_system 1 lam2 beta2 (by intro i; fin_cases i <;> simp [lam2])

/-- The recovered solution direction is `(1, 1/2)` — i.e. `A⁻¹ b`. -/
example : hhl_output 1 lam2 beta2 0 = 1 ∧ hhl_output 1 lam2 beta2 1 = 1 / 2 := by
  constructor <;> simp [hhl_output, lam2, beta2]

end Numeric

end QuantumProofs.HHL
