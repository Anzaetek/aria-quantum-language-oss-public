/-
  Formal verification of Grover's search algorithm.

  Goal: Prove that Grover's algorithm finds a marked item in O(√N) iterations
  with probability ≥ 1 - 1/N.

  Theorem (Grover correctness):
    After k = ⌊π√N/4⌋ iterations of the Grover operator G = -H⊗ⁿ·Z₀·H⊗ⁿ·Zf,
    measuring the state yields the marked item with probability
    sin²((2k+1)·arcsin(1/√N)) ≥ 1 - 1/N.

  Proof strategy:
  1. Define the Grover operator as composition of oracle + diffusion
  2. Express the state in the 2D subspace {|w⟩, |s'⟩}
  3. Show G is a rotation by 2·arcsin(1/√N) in this subspace
  4. Count iterations to reach the marked state
-/

import QuantumProofs.CircuitSemantics
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

namespace QuantumProofs.Grover

/-- The Grover diffusion operator on n qubits.
    D = 2|s⟩⟨s| - I where |s⟩ = H⊗ⁿ|0⟩ is the uniform superposition.
    Entrywise: Dᵢⱼ = 2/N - δᵢⱼ, with N = 2^n. -/
noncomputable def diffusion_operator (n : ℕ) : Matrix (Fin (2^n)) (Fin (2^n)) ℂ :=
  fun i j =>
    let N : ℝ := (2^n : ℕ)
    if i = j then
      Complex.ofReal (2 / N - 1)
    else
      Complex.ofReal (2 / N)

/-- The oracle operator for a marked item w.
    O_w|x⟩ = -|x⟩ if x = w, |x⟩ otherwise. -/
noncomputable def oracle (n : ℕ) (w : Fin (2^n)) : Matrix (Fin (2^n)) (Fin (2^n)) ℂ :=
  fun i j =>
    if i = j then
      if i = w then -1 else 1
    else 0

/-- The Grover operator G = D · O_w -/
noncomputable def grover_operator (n : ℕ) (w : Fin (2^n)) : Matrix (Fin (2^n)) (Fin (2^n)) ℂ :=
  diffusion_operator n * oracle n w

/-- The angle of rotation per Grover iteration. -/
noncomputable def grover_angle (n : ℕ) : ℝ :=
  2 * Real.arcsin (1 / Real.sqrt (2^n : ℕ))

/-- Optimal number of Grover iterations: the nearest integer to
    `π/(4θ) − 1/2`, where `θ = arcsin(1/√N)` is the per-iteration rotation
    half-angle. This is the *exact* optimizer — `round(π/(4θ) − 1/2)` lands
    `(2k+1)θ` within `θ` of `π/2` by the rounding property (`abs_sub_round`),
    which is what `grover_optimal_success` needs. (The textbook approximation
    `⌊π√N/4⌋` agrees asymptotically but over-shoots for some `N`, since
    `√N·arcsin(1/√N) ≥ 1`, so it would make the bound *false* for those `N`.) -/
noncomputable def optimal_iterations (n : ℕ) : ℕ :=
  (round (Real.pi / (4 * Real.arcsin (1 / Real.sqrt (2^n : ℕ))) - 1/2 : ℝ)).toNat

/-- Uniform superposition state `|s⟩ = (1/√N)·∑ᵢ|i⟩` as a column
    vector (indexed by `Fin (2^n)`). -/
noncomputable def uniform_state (n : ℕ) : Fin (2^n) → ℂ :=
  fun _ => 1 / Complex.ofReal (Real.sqrt (2^n : ℕ))

/-- Amplitude of a basis state `|k⟩` in a state vector `ψ`. -/
@[inline] noncomputable def amplitude {n : ℕ}
    (ψ : Fin (2^n) → ℂ) (k : Fin (2^n)) : ℂ := ψ k

/-- Apply an n-qubit unitary to a state vector. -/
noncomputable def apply_unitary {n : ℕ}
    (U : Matrix (Fin (2^n)) (Fin (2^n)) ℂ) (ψ : Fin (2^n) → ℂ) :
    Fin (2^n) → ℂ :=
  fun i => ∑ j, U i j * ψ j

-- ===========================================================================
-- 2D subspace structure for Grover.
--
-- The Grover operator `G = D · O_w` preserves the 2-dimensional subspace
-- spanned by `|w⟩` and `|s'⟩`, where `|s'⟩` is the uniform superposition
-- over the unmarked elements. On this subspace it acts as a rotation by
-- `grover_angle = 2·arcsin(1/√N)`. Iterating `k` times rotates the initial
-- state (which starts at angle `arcsin(1/√N)` from `|s'⟩`) to angle
-- `(2k+1)·arcsin(1/√N)`, so the amplitude on `|w⟩` is
-- `sin((2k+1)·arcsin(1/√N))`.
--
-- The proof is decomposed below into named step obligations so each piece
-- has a meaningful statement; closing them requires the explicit matrix
-- computations plus standard trig identities from
-- `Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic`.
-- ===========================================================================

/-- The "bad subspace" state `|s'⟩ = (1/√(N-1)) · Σ_{x≠w} |x⟩`:
    the uniform superposition over all unmarked basis elements. -/
noncomputable def bad_state (n : ℕ) (w : Fin (2^n)) : Fin (2^n) → ℂ :=
  fun i =>
    if i = w then 0
    else 1 / Complex.ofReal (Real.sqrt ((2^n - 1 : ℕ) : ℝ))

/-- Single-amplitude shorthand used in the rotation argument:
    `θ := arcsin(1/√N)`. -/
noncomputable def grover_theta (n : ℕ) : ℝ :=
  Real.arcsin (1 / Real.sqrt (2^n : ℕ))

/-- `grover_angle = 2 · grover_theta`. Unfolding lemma. -/
theorem grover_angle_eq (n : ℕ) : grover_angle n = 2 * grover_theta n := rfl

/-- `apply_unitary` of the identity matrix is the identity map. -/
theorem apply_unitary_one {n : ℕ} (ψ : Fin (2^n) → ℂ) : apply_unitary 1 ψ = ψ := by
  funext i; unfold apply_unitary
  rw [Finset.sum_eq_single i]
  · rw [Matrix.one_apply_eq, one_mul]
  · intro j _ hj; rw [Matrix.one_apply_ne (Ne.symm hj), zero_mul]
  · intro h; exact absurd (Finset.mem_univ i) h

/-- `apply_unitary` is a homomorphism: `apply_unitary (A·B) = apply_unitary A ∘
    apply_unitary B`. Lets `Gᵏ⁺¹ = G · Gᵏ` peel one factor in the induction. -/
theorem apply_unitary_mul {n : ℕ} (A B : Matrix (Fin (2^n)) (Fin (2^n)) ℂ)
    (ψ : Fin (2^n) → ℂ) :
    apply_unitary (A * B) ψ = apply_unitary A (apply_unitary B ψ) := by
  funext i; unfold apply_unitary
  simp only [Matrix.mul_apply, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]; congr 1; funext j; congr 1; funext k; ring

/-- `apply_unitary` is linear in the state (the 2-term shape used in the
    rotation induction). -/
theorem apply_unitary_lin {n : ℕ} (U : Matrix (Fin (2^n)) (Fin (2^n)) ℂ)
    (c d : ℂ) (f g : Fin (2^n) → ℂ) :
    apply_unitary U (fun j => c * f j + d * g j)
      = fun i => c * apply_unitary U f i + d * apply_unitary U g i := by
  funext i; unfold apply_unitary
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  congr 1; funext j; ring

/-- **Step 1** (initial decomposition): the uniform superposition
    `|s⟩ = sin(θ)·|w⟩ + cos(θ)·|s'⟩` in the `{|w⟩, |s'⟩}` basis, with
    `θ = grover_theta = arcsin(1/√N)`. Pointwise: at `w` the amplitude is
    `sin θ = 1/√N`; at any unmarked `i` it is `cos θ · (1/√(N−1)) = 1/√N`
    (since `cos θ = √(N−1)/√N`). -/
theorem grover_uniform_decomp (n : ℕ) (w : Fin (2^n)) :
    uniform_state n =
      (fun i => Complex.ofReal (Real.sin (grover_theta n)) * (if i = w then 1 else 0)
        + Complex.ofReal (Real.cos (grover_theta n)) * bad_state n w i) := by
  funext i
  unfold uniform_state grover_theta bad_state
  have hN1 : (1:ℝ) ≤ ((2^n:ℕ):ℝ) := by exact_mod_cast Nat.one_le_two_pow
  have hNpos : (0:ℝ) < ((2^n:ℕ):ℝ) := by linarith
  have hsqrtpos : (0:ℝ) < Real.sqrt ((2^n:ℕ):ℝ) := Real.sqrt_pos.mpr hNpos
  have hxle : 1 / Real.sqrt ((2^n:ℕ):ℝ) ≤ 1 := by
    rw [div_le_one hsqrtpos]
    calc (1:ℝ) = Real.sqrt 1 := (Real.sqrt_one).symm
      _ ≤ Real.sqrt ((2^n:ℕ):ℝ) := Real.sqrt_le_sqrt hN1
  have hxge : (-1:ℝ) ≤ 1 / Real.sqrt ((2^n:ℕ):ℝ) :=
    le_trans (by norm_num : (-1:ℝ) ≤ 0) (by positivity)
  have hsin : Real.sin (Real.arcsin (1 / Real.sqrt ((2^n:ℕ):ℝ))) = 1 / Real.sqrt ((2^n:ℕ):ℝ) :=
    Real.sin_arcsin hxge hxle
  by_cases hiw : i = w
  · simp only [hiw, ↓reduceIte, mul_one, mul_zero, add_zero, hsin]
    rw [one_div, one_div, Complex.ofReal_inv]
  · have hcard : 1 < Fintype.card (Fin (2^n)) := Fintype.one_lt_card_iff.mpr ⟨i, w, hiw⟩
    rw [Fintype.card_fin] at hcard
    have hposR : (0:ℝ) < ((2^n:ℕ):ℝ) - 1 := by
      have : (1:ℝ) < ((2^n:ℕ):ℝ) := by exact_mod_cast hcard
      linarith
    have hcos : Real.cos (Real.arcsin (1 / Real.sqrt ((2^n:ℕ):ℝ)))
        = Real.sqrt (((2^n:ℕ):ℝ) - 1) / Real.sqrt ((2^n:ℕ):ℝ) := by
      rw [Real.cos_arcsin,
          show (1:ℝ) - (1/Real.sqrt ((2^n:ℕ):ℝ))^2 = (((2^n:ℕ):ℝ)-1)/((2^n:ℕ):ℝ) from by
            rw [div_pow, one_pow, Real.sq_sqrt (le_of_lt hNpos)]; field_simp,
          Real.sqrt_div (by linarith)]
    have hcast : ((2^n - 1 : ℕ):ℝ) = ((2^n:ℕ):ℝ) - 1 := by
      rw [Nat.cast_sub Nat.one_le_two_pow]; norm_num
    have h1 : (Complex.ofReal (Real.sqrt (((2^n:ℕ):ℝ)-1))) ≠ 0 := by
      rw [Complex.ofReal_ne_zero]; exact Real.sqrt_ne_zero'.mpr hposR
    have h2 : (Complex.ofReal (Real.sqrt ((2^n:ℕ):ℝ))) ≠ 0 := by
      rw [Complex.ofReal_ne_zero]; exact ne_of_gt hsqrtpos
    simp only [hiw, ↓reduceIte, mul_zero, zero_add, hcos]
    rw [hcast, Complex.ofReal_div]; field_simp

/-- The oracle is the diagonal matrix `diag(±1)` (`−1` at the marked
    index `w`, `+1` elsewhere). Lets `D · O_w` collapse via
    `Matrix.mul_diagonal`. -/
theorem oracle_diagonal (n : ℕ) (w : Fin (2^n)) :
    oracle n w = Matrix.diagonal (fun k => if k = w then (-1 : ℂ) else 1) := by
  ext i j; unfold oracle Matrix.diagonal; by_cases h : i = j <;> simp [h]

/-- Each row of the diffusion operator `D = 2|s⟩⟨s| − I` sums to `1`:
    one diagonal entry `2/N − 1` plus `N−1` off-diagonal entries `2/N`. -/
theorem diffusion_row_sum (n : ℕ) (i : Fin (2^n)) :
    ∑ j, diffusion_operator n i j = 1 := by
  have hcard : (Finset.univ : Finset (Fin (2^n))).card = 2^n := by simp
  show (∑ j, if i = j then Complex.ofReal (2 / ((2^n:ℕ):ℝ) - 1)
        else Complex.ofReal (2 / ((2^n:ℕ):ℝ))) = 1
  set A : ℂ := Complex.ofReal (2 / ((2^n:ℕ):ℝ) - 1) with hA
  set B : ℂ := Complex.ofReal (2 / ((2^n:ℕ):ℝ)) with hB
  have hsplit : ∀ j : Fin (2^n), (if i = j then A else B) = B + (if i = j then A - B else 0) := by
    intro j; by_cases h : i = j <;> simp [h]
  rw [Finset.sum_congr rfl (fun j _ => hsplit j), Finset.sum_add_distrib,
      Finset.sum_const, Finset.sum_ite_eq Finset.univ i, hcard]
  simp only [Finset.mem_univ, if_true, nsmul_eq_mul]
  rw [hA, hB]
  have hN : ((2^n : ℕ) : ℝ) ≠ 0 := by positivity
  have h2n : ((2^n : ℕ) : ℂ) = Complex.ofReal ((2^n:ℕ):ℝ) := by norm_cast
  rw [h2n, ← Complex.ofReal_mul, ← Complex.ofReal_sub, ← Complex.ofReal_add,
      show (1:ℂ) = Complex.ofReal 1 from by norm_num, Complex.ofReal_inj]
  set N := ((2^n:ℕ):ℝ) with hNdef
  field_simp; ring

/-- Trig identity: `cos(grover_angle) = 1 − 2/N`. (Via `cos(2θ) = 2cos²θ−1`
    and `cos(arcsin x) = √(1−x²)` with `x = 1/√N`.) -/
theorem grover_cos_angle (n : ℕ) : Real.cos (grover_angle n) = 1 - 2 / ((2^n:ℕ):ℝ) := by
  unfold grover_angle
  have hN1 : (1:ℝ) ≤ ((2^n:ℕ):ℝ) := by exact_mod_cast Nat.one_le_two_pow
  have hNpos : (0:ℝ) < ((2^n:ℕ):ℝ) := by linarith
  have hx2 : (1 / Real.sqrt ((2^n:ℕ):ℝ))^2 = 1 / ((2^n:ℕ):ℝ) := by
    rw [div_pow, one_pow, Real.sq_sqrt (le_of_lt hNpos)]
  rw [Real.cos_two_mul, Real.cos_arcsin,
      Real.sq_sqrt (by rw [hx2, sub_nonneg, div_le_one hNpos]; exact hN1), hx2]
  field_simp; ring

/-- Trig identity: `sin(grover_angle) = (2/N)·√(N−1)`. (Via `sin(2θ) =
    2 sinθ cosθ`, `sin(arcsin x) = x`, `cos(arcsin x) = √(1−x²)`.) -/
theorem grover_sin_angle (n : ℕ) :
    Real.sin (grover_angle n) = (2 / ((2^n:ℕ):ℝ)) * Real.sqrt ((2^n - 1 : ℕ):ℝ) := by
  unfold grover_angle
  have hN1 : (1:ℝ) ≤ ((2^n:ℕ):ℝ) := by exact_mod_cast Nat.one_le_two_pow
  have hNpos : (0:ℝ) < ((2^n:ℕ):ℝ) := by linarith
  have hsqrtpos : (0:ℝ) < Real.sqrt ((2^n:ℕ):ℝ) := Real.sqrt_pos.mpr hNpos
  have hxle : 1 / Real.sqrt ((2^n:ℕ):ℝ) ≤ 1 := by
    rw [div_le_one hsqrtpos]
    calc (1:ℝ) = Real.sqrt 1 := (Real.sqrt_one).symm
      _ ≤ Real.sqrt ((2^n:ℕ):ℝ) := Real.sqrt_le_sqrt hN1
  have hxge : (-1:ℝ) ≤ 1 / Real.sqrt ((2^n:ℕ):ℝ) :=
    le_trans (by norm_num : (-1:ℝ) ≤ 0) (by positivity)
  have hx2 : (1 / Real.sqrt ((2^n:ℕ):ℝ))^2 = 1 / ((2^n:ℕ):ℝ) := by
    rw [div_pow, one_pow, Real.sq_sqrt (le_of_lt hNpos)]
  have hNN : Real.sqrt ((2^n:ℕ):ℝ) * Real.sqrt ((2^n:ℕ):ℝ) = ((2^n:ℕ):ℝ) :=
    Real.mul_self_sqrt (le_of_lt hNpos)
  rw [Real.sin_two_mul, Real.sin_arcsin hxge hxle, Real.cos_arcsin, hx2]
  have hcast : ((2^n - 1 : ℕ):ℝ) = ((2^n:ℕ):ℝ) - 1 := by
    rw [Nat.cast_sub Nat.one_le_two_pow]; norm_num
  rw [hcast, show (1 : ℝ) - 1/((2^n:ℕ):ℝ) = (((2^n:ℕ):ℝ) - 1)/((2^n:ℕ):ℝ) from by field_simp,
     Real.sqrt_div (by linarith), mul_assoc,
     show (1/Real.sqrt ((2^n:ℕ):ℝ)) * (Real.sqrt (((2^n:ℕ):ℝ)-1)/Real.sqrt ((2^n:ℕ):ℝ))
        = Real.sqrt (((2^n:ℕ):ℝ)-1)/((2^n:ℕ):ℝ) from by rw [div_mul_div_comm, one_mul, hNN]]
  ring

/-- The `|w⟩`-column of `G = D·O_w`: `G·|w⟩ = (1−2/N)·|w⟩ − (2/N)√(N−1)·|s'⟩`.
    Computed entrywise — `O_w = diag(±1)` flips the sign at `w`, so the
    column reads off `−D·,w`, and `D` has diagonal `2/N−1`, off-diagonal
    `2/N`. -/
theorem grover_w_column (n : ℕ) (w : Fin (2^n)) :
    apply_unitary (grover_operator n w) (fun i => if i = w then 1 else 0) =
      (fun i => Complex.ofReal (1 - 2/((2^n:ℕ):ℝ)) * (if i = w then 1 else 0)
        + Complex.ofReal (-(2/((2^n:ℕ):ℝ)) * Real.sqrt ((2^n - 1 : ℕ) : ℝ)) * bad_state n w i) := by
  funext i
  unfold apply_unitary grover_operator
  rw [oracle_diagonal]
  simp only [Matrix.mul_diagonal]
  rw [Finset.sum_eq_single w]
  · unfold diffusion_operator bad_state
    by_cases hiw : i = w
    · simp [hiw]
    · have hpos : (0:ℝ) < ((2^n - 1 : ℕ) : ℝ) := by
        have hcard : 1 < Fintype.card (Fin (2^n)) := Fintype.one_lt_card_iff.mpr ⟨i, w, hiw⟩
        rw [Fintype.card_fin] at hcard
        have h1 : 1 ≤ 2^n - 1 := by omega
        exact_mod_cast Nat.lt_of_lt_of_le Nat.zero_lt_one h1
      have hs_ne : Real.sqrt ((2^n - 1 : ℕ) : ℝ) ≠ 0 := by positivity
      simp only [hiw, ↓reduceIte, mul_one, mul_neg, mul_zero, zero_add]
      rw [one_div, ← Complex.ofReal_inv, ← Complex.ofReal_mul, ← Complex.ofReal_neg,
          Complex.ofReal_inj]
      field_simp
  · intro j _ hj; simp [hj]
  · intro h; exact absurd (Finset.mem_univ w) h

/-- The `|s'⟩`-column of `G = D·O_w`:
    `G·|s'⟩ = (2−2/N)/√(N−1)·|w⟩ + (1−2/N)·|s'⟩`. The `w`-summand drops out
    (`s'_w = 0`), leaving `(1/√(N−1))·(1 − D i w)` via the row-sum
    `∑_j D i j = 1`. -/
theorem grover_sprime_column (n : ℕ) (w : Fin (2^n)) :
    apply_unitary (grover_operator n w) (bad_state n w) =
      (fun i => Complex.ofReal ((2 - 2/((2^n:ℕ):ℝ)) / Real.sqrt ((2^n - 1 : ℕ) : ℝ)) * (if i = w then 1 else 0)
        + Complex.ofReal (1 - 2/((2^n:ℕ):ℝ)) * bad_state n w i) := by
  funext i
  unfold apply_unitary grover_operator
  rw [oracle_diagonal]
  simp only [Matrix.mul_diagonal]
  have hterm : ∀ j, diffusion_operator n i j * (if j = w then (-1:ℂ) else 1) * bad_state n w j
              = diffusion_operator n i j * (if j = w then (0:ℂ)
                  else 1 / Complex.ofReal (Real.sqrt ((2^n - 1 : ℕ) : ℝ))) := by
    intro j; unfold bad_state; by_cases h : j = w <;> simp [h]
  rw [Finset.sum_congr rfl (fun j _ => hterm j)]
  set c : ℂ := 1 / Complex.ofReal (Real.sqrt ((2^n - 1 : ℕ) : ℝ)) with hc
  have hsub : ∀ j, diffusion_operator n i j * (if j = w then (0:ℂ) else c)
              = diffusion_operator n i j * c - (if j = w then diffusion_operator n i j * c else 0) := by
    intro j; by_cases h : j = w <;> simp [h]
  rw [Finset.sum_congr rfl (fun j _ => hsub j), Finset.sum_sub_distrib,
      Finset.sum_ite_eq' Finset.univ w, ← Finset.sum_mul, diffusion_row_sum]
  simp only [Finset.mem_univ, if_true, one_mul]
  unfold diffusion_operator bad_state
  by_cases hiw : i = w
  · simp only [hiw, ↓reduceIte, mul_zero, add_zero, mul_one]
    rw [hc]; push_cast; ring
  · simp only [hiw, ↓reduceIte, mul_zero, zero_add]
    rw [hc]; push_cast; ring

/-- **Step 2** (`G` preserves the subspace): `G · |w⟩` and `G · |s'⟩`
    both lie in `span{|w⟩, |s'⟩}`. Immediate from the concrete columns
    `grover_w_column` / `grover_sprime_column`. -/
theorem grover_preserves_subspace (n : ℕ) (w : Fin (2^n)) :
    ∃ (a₁ a₂ b₁ b₂ : ℂ),
      apply_unitary (grover_operator n w) (fun i => if i = w then 1 else 0) =
        (fun i => a₁ * (if i = w then 1 else 0) + a₂ * bad_state n w i) ∧
      apply_unitary (grover_operator n w) (bad_state n w) =
        (fun i => b₁ * (if i = w then 1 else 0) + b₂ * bad_state n w i) :=
  ⟨_, _, _, _, grover_w_column n w, grover_sprime_column n w⟩

/-- **Step 3** (rotation shape): any coefficients realizing the columns of
    `G` on `{|w⟩, |s'⟩}` are exactly the rotation-matrix entries for
    `grover_angle n`. The hypotheses pin `(a₁,a₂,b₁,b₂)` (uniqueness:
    `|w⟩` and `|s'⟩` are independent — needs `1 ≤ n` so some `i ≠ w`
    exists to read off the `|s'⟩` component), and the trig identities
    `grover_cos_angle`/`grover_sin_angle` convert the explicit values.

    NB: this is the rotation `[[cos, sin], [−sin, cos]]` — i.e.
    `a₂ = −sin`, `b₁ = +sin` (an earlier draft had these signs flipped,
    which is *false* against the actual `D·O_w` computation; the
    orientation here matches `grover_w_column`/`grover_sprime_column`).
    `grover_probability` only sees `normSq`, so the sign is immaterial
    downstream. -/
theorem grover_rotation_coefficients (n : ℕ) (w : Fin (2^n)) (hn : 1 ≤ n)
    (a₁ a₂ b₁ b₂ : ℂ)
    (ha : apply_unitary (grover_operator n w) (fun i => if i = w then 1 else 0) =
          (fun i => a₁ * (if i = w then 1 else 0) + a₂ * bad_state n w i))
    (hb : apply_unitary (grover_operator n w) (bad_state n w) =
          (fun i => b₁ * (if i = w then 1 else 0) + b₂ * bad_state n w i)) :
    a₁ = Complex.ofReal (Real.cos (grover_angle n)) ∧
    a₂ = -Complex.ofReal (Real.sin (grover_angle n)) ∧
    b₁ = Complex.ofReal (Real.sin (grover_angle n)) ∧
    b₂ = Complex.ofReal (Real.cos (grover_angle n)) := by
  have h2 : 2 ≤ 2^n := by
    calc 2 = 2^1 := (pow_one 2).symm
      _ ≤ 2^n := Nat.pow_le_pow_right (by norm_num) hn
  haveI : Nontrivial (Fin (2^n)) := Fin.nontrivial_iff_two_le.mpr h2
  obtain ⟨i₀, hi₀⟩ := exists_ne w
  have hpos : (0:ℝ) < ((2^n - 1 : ℕ) : ℝ) := by
    have : 1 ≤ 2^n - 1 := by omega
    exact_mod_cast Nat.lt_of_lt_of_le Nat.zero_lt_one this
  have hs_ne : Real.sqrt ((2^n - 1 : ℕ) : ℝ) ≠ 0 := by positivity
  have hbne : bad_state n w i₀ ≠ 0 := by
    simp only [bad_state, if_neg hi₀]
    rw [one_div, ne_eq, inv_eq_zero, Complex.ofReal_eq_zero]; exact hs_ne
  -- combine the hypotheses with the concrete columns
  have hEqA := ha.symm.trans (grover_w_column n w)
  have hEqB := hb.symm.trans (grover_sprime_column n w)
  -- read off a₁, b₁ at i = w (the |s'⟩ component vanishes there)
  have ea1 : a₁ = Complex.ofReal (1 - 2/((2^n:ℕ):ℝ)) := by
    have := congrFun hEqA w; simpa [bad_state] using this
  have eb1 : b₁ = Complex.ofReal ((2 - 2/((2^n:ℕ):ℝ)) / Real.sqrt ((2^n - 1 : ℕ) : ℝ)) := by
    have := congrFun hEqB w; simpa [bad_state] using this
  -- read off a₂, b₂ at i = i₀ ≠ w (cancel the nonzero |s'⟩ amplitude)
  have ea2 : a₂ = Complex.ofReal (-(2/((2^n:ℕ):ℝ)) * Real.sqrt ((2^n - 1 : ℕ) : ℝ)) := by
    have h := congrFun hEqA i₀
    simp only [if_neg hi₀, mul_zero, zero_add] at h
    exact mul_right_cancel₀ hbne h
  have eb2 : b₂ = Complex.ofReal (1 - 2/((2^n:ℕ):ℝ)) := by
    have h := congrFun hEqB i₀
    simp only [if_neg hi₀, mul_zero, zero_add] at h
    exact mul_right_cancel₀ hbne h
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [ea1, grover_cos_angle]
  · rw [ea2, grover_sin_angle, ← Complex.ofReal_neg, neg_mul]
  · rw [eb1, grover_sin_angle, Complex.ofReal_inj, div_eq_iff hs_ne]
    have hNN : Real.sqrt ((2^n - 1 : ℕ):ℝ) * Real.sqrt ((2^n - 1 : ℕ):ℝ) = ((2^n - 1 : ℕ):ℝ) :=
      Real.mul_self_sqrt (le_of_lt hpos)
    have hcast : ((2^n - 1 : ℕ):ℝ) = ((2^n:ℕ):ℝ) - 1 := by
      rw [Nat.cast_sub Nat.one_le_two_pow]; norm_num
    rw [mul_assoc, hNN, hcast]; field_simp
  · rw [eb2, grover_cos_angle]

/-- **Step 4** (iteration): `k` applications of a 2D rotation by
    `grover_angle` starting from state `(sin θ, cos θ)` yields
    `(sin((2k+1)θ), cos((2k+1)θ))`. Captured as the angular formula. -/
theorem grover_iterate_angle (n : ℕ) (k : ℕ) :
    (2 * k + 1 : ℕ) * grover_theta n =
      (2 * k : ℕ) * (grover_angle n / 2) + grover_theta n := by
  unfold grover_theta grover_angle
  push_cast; ring

/-- **The 2D rotation in closed form.** After `k` Grover iterations the
    state is exactly `sin((2k+1)θ)·|w⟩ + cos((2k+1)θ)·|s'⟩`. Induction on
    `k`: the base case is `grover_uniform_decomp`; each step applies `G`
    (linear, via `apply_unitary_lin`) to the columns
    `grover_w_column`/`grover_sprime_column`, whose `cos/sin(grover_angle)`
    coefficients (`grover_cos_angle`/`grover_sin_angle`) advance the angle
    by `grover_angle = 2θ` via `sin_add`/`cos_add`. Needs `1 ≤ n`. -/
theorem grover_iterate (n : ℕ) (w : Fin (2^n)) (hn : 1 ≤ n) (k : ℕ) :
    apply_unitary ((grover_operator n w) ^ k) (uniform_state n) =
      (fun i => Complex.ofReal (Real.sin ((2*k+1 : ℕ) * grover_theta n)) * (if i = w then 1 else 0)
        + Complex.ofReal (Real.cos ((2*k+1 : ℕ) * grover_theta n)) * bad_state n w i) := by
  induction k with
  | zero => rw [pow_zero, apply_unitary_one, grover_uniform_decomp n w]; simp
  | succ k ih =>
      rw [pow_succ', apply_unitary_mul, ih, apply_unitary_lin,
          grover_w_column, grover_sprime_column]
      funext i
      set ψ : ℝ := (2*k+1 : ℕ) * grover_theta n with hψ
      have hangle : ((2*(k+1)+1 : ℕ) : ℝ) * grover_theta n = ψ + grover_angle n := by
        rw [hψ, grover_angle_eq]; push_cast; ring
      have hc : Complex.ofReal (1 - 2/((2^n:ℕ):ℝ)) = Complex.ofReal (Real.cos (grover_angle n)) := by
        rw [grover_cos_angle]
      have hs : Complex.ofReal ((2 - 2/((2^n:ℕ):ℝ)) / Real.sqrt ((2^n - 1 : ℕ) : ℝ))
          = Complex.ofReal (Real.sin (grover_angle n)) := by
        rw [grover_sin_angle, Complex.ofReal_inj]
        have hpos : (0:ℝ) < ((2^n - 1 : ℕ) : ℝ) := by
          have hcard : 1 < 2^n := by
            calc 1 < 2^1 := by norm_num
              _ ≤ 2^n := Nat.pow_le_pow_right (by norm_num) hn
          have : 1 ≤ 2^n - 1 := by omega
          exact_mod_cast Nat.lt_of_lt_of_le Nat.zero_lt_one this
        have hs_ne : Real.sqrt ((2^n - 1 : ℕ) : ℝ) ≠ 0 := by positivity
        rw [div_eq_iff hs_ne]
        have hNN : Real.sqrt ((2^n - 1 : ℕ):ℝ) * Real.sqrt ((2^n - 1 : ℕ):ℝ) = ((2^n - 1 : ℕ):ℝ) :=
          Real.mul_self_sqrt (le_of_lt hpos)
        have hcast : ((2^n - 1 : ℕ):ℝ) = ((2^n:ℕ):ℝ) - 1 := by
          rw [Nat.cast_sub Nat.one_le_two_pow]; norm_num
        rw [mul_assoc, hNN, hcast]; field_simp
      have hsa : Complex.ofReal (-(2/((2^n:ℕ):ℝ)) * Real.sqrt ((2^n - 1 : ℕ) : ℝ))
          = -Complex.ofReal (Real.sin (grover_angle n)) := by
        rw [grover_sin_angle, ← Complex.ofReal_neg, neg_mul]
      rw [hangle, hc, hsa, hs, Real.sin_add, Real.cos_add]
      push_cast [Complex.ofReal_sin, Complex.ofReal_cos]
      ring

/-- After k iterations of the Grover operator starting from the uniform
    superposition, the probability of measuring the marked item `w` is
    `sin²((2k+1) · arcsin(1/√N))`. The amplitude at `w` is read off
    `grover_iterate` (the `|s'⟩` part vanishes at `w`); `normSq ∘ ofReal`
    squares it. Needs `1 ≤ n` (Grover needs at least one qubit). -/
theorem grover_probability (n : ℕ) (w : Fin (2^n)) (hn : 1 ≤ n) (k : ℕ) :
    Complex.normSq
        (amplitude (apply_unitary ((grover_operator n w) ^ k) (uniform_state n)) w)
      =
    Real.sin ((2 * k + 1 : ℕ) * Real.arcsin (1 / Real.sqrt (2^n : ℕ))) ^ 2 := by
  unfold amplitude
  rw [grover_iterate n w hn k]
  simp only [↓reduceIte, mul_one, bad_state, mul_zero, add_zero]
  rw [Complex.normSq_ofReal]
  unfold grover_theta
  ring

set_option maxHeartbeats 1200000 in
/-- At the optimal number of iterations, the success probability is
    `≥ 1 − 1/N` for `n ≥ 2`. From `grover_probability` the probability is
    `sin²((2k+1)θ)`; the rounding choice of `optimal_iterations` puts
    `(2k+1)θ` within `θ` of `π/2` (`abs_sub_round`), so
    `sin((2k+1)θ) = cos(π/2 − (2k+1)θ) ≥ cos θ ≥ 0` (cos antitone on
    `[0,π]`), hence `sin²((2k+1)θ) ≥ cos²θ = 1 − sin²θ = 1 − 1/N`. -/
theorem grover_optimal_success (n : ℕ) (w : Fin (2^n)) (hn : n ≥ 2) :
    Complex.normSq
        (amplitude
          (apply_unitary
            ((grover_operator n w) ^ (optimal_iterations n))
            (uniform_state n))
          w)
      ≥ 1 - 1 / (2^n : ℕ) := by
  have h1n : 1 ≤ n := by omega
  have hNR : (1:ℝ) < ((2^n:ℕ):ℝ) := by
    have h2 : (2:ℕ) ≤ 2^n := by
      calc 2 = 2^1 := (pow_one 2).symm
        _ ≤ 2^n := Nat.pow_le_pow_right (by norm_num) h1n
    have : (2:ℝ) ≤ ((2^n:ℕ):ℝ) := by exact_mod_cast h2
    linarith
  have hsqrt_pos : 0 < Real.sqrt ((2^n:ℕ):ℝ) := Real.sqrt_pos.mpr (by linarith)
  have hx_pos : 0 < 1 / Real.sqrt ((2^n:ℕ):ℝ) := by positivity
  have hx_le1 : 1 / Real.sqrt ((2^n:ℕ):ℝ) ≤ 1 := by
    rw [div_le_one hsqrt_pos]
    calc (1:ℝ) = Real.sqrt 1 := Real.sqrt_one.symm
      _ ≤ _ := Real.sqrt_le_sqrt (by linarith)
  set θ := Real.arcsin (1 / Real.sqrt ((2^n:ℕ):ℝ)) with hθdef
  have hθ_pos : 0 < θ := Real.arcsin_pos.mpr hx_pos
  have hθ_le : θ ≤ Real.pi/2 := Real.arcsin_le_pi_div_two _
  have hsinθ : Real.sin θ = 1 / Real.sqrt ((2^n:ℕ):ℝ) := Real.sin_arcsin (by linarith) hx_le1
  have hround_nn : 0 ≤ round (Real.pi / (4 * θ) - 1/2 : ℝ) := by
    rw [round_eq]; apply Int.floor_nonneg.mpr
    rw [show Real.pi/(4*θ) - 1/2 + 1/2 = Real.pi/(4*θ) from by ring]; positivity
  have hk_real : ((optimal_iterations n : ℕ):ℝ) = ((round (Real.pi/(4*θ) - 1/2 : ℝ) : ℤ):ℝ) := by
    rw [optimal_iterations, ← hθdef, ← Int.cast_natCast, Int.toNat_of_nonneg hround_nn]
  have hbound : |(2*((optimal_iterations n : ℕ):ℝ)+1)*θ - Real.pi/2| ≤ θ := by
    rw [hk_real]
    have hr : |((round (Real.pi/(4*θ) - 1/2 : ℝ):ℤ):ℝ) - (Real.pi/(4*θ) - 1/2)| ≤ 1/2 := by
      rw [abs_sub_comm]; exact abs_sub_round (Real.pi/(4*θ) - 1/2)
    rw [show (2*((round (Real.pi/(4*θ) - 1/2 : ℝ):ℤ):ℝ)+1)*θ - Real.pi/2
          = 2*θ*(((round (Real.pi/(4*θ) - 1/2 : ℝ):ℤ):ℝ) - (Real.pi/(4*θ) - 1/2)) from by
        field_simp; ring]
    rw [abs_mul, abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 2), abs_of_pos hθ_pos]
    nlinarith [hr, hθ_pos.le,
      abs_nonneg (((round (Real.pi/(4*θ) - 1/2 : ℝ):ℤ):ℝ) - (Real.pi/(4*θ) - 1/2))]
  rw [grover_probability n w h1n (optimal_iterations n), ← hθdef]
  have hcast : ((2*(optimal_iterations n)+1 : ℕ):ℝ) = 2*((optimal_iterations n:ℕ):ℝ)+1 := by
    push_cast; ring
  rw [hcast]
  set φ := (2*((optimal_iterations n : ℕ):ℝ)+1)*θ with hφdef
  have hcos_le : Real.cos θ ≤ Real.sin φ := by
    rw [hφdef, ← Real.cos_pi_div_two_sub, ← Real.cos_abs (Real.pi/2 - _)]
    apply Real.cos_le_cos_of_nonneg_of_le_pi (abs_nonneg _) (by linarith)
    rw [abs_sub_comm]; exact hbound
  have hcosθ_nn : 0 ≤ Real.cos θ := Real.cos_nonneg_of_mem_Icc ⟨by linarith [Real.pi_pos], hθ_le⟩
  have hcos_sq : Real.cos θ ^ 2 = 1 - 1/((2^n:ℕ):ℝ) := by
    have hpyth : Real.cos θ ^2 = 1 - Real.sin θ ^2 := by nlinarith [Real.sin_sq_add_cos_sq θ]
    rw [hpyth, hsinθ, div_pow, one_pow, Real.sq_sqrt (by linarith)]
  rw [ge_iff_le]
  calc 1 - 1/((2^n:ℕ):ℝ) = Real.cos θ ^ 2 := hcos_sq.symm
    _ ≤ Real.sin φ ^ 2 := by nlinarith [hcos_le, hcosθ_nn]

end QuantumProofs.Grover
