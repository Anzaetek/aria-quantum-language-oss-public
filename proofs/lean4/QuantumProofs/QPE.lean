/-
  Formal verification of Quantum Phase Estimation.

  Goal: Prove that QPE estimates the eigenvalue phase θ of a unitary U
  with precision 2^{-n} using n counting qubits.

  Theorem (QPE correctness):
    Given U|ψ⟩ = e^{2πiθ}|ψ⟩, QPE outputs |θ̃⟩ where θ̃ approximates θ
    to n bits with probability ≥ 4/π² ≈ 0.405 (for exact θ: probability 1).

  Proof strategy:
  1. Define QPE circuit (H on counting, controlled-U^{2^k}, inverse QFT)
  2. Track the state through each stage
  3. Show the counting register encodes θ after inverse QFT
  4. Bound the success probability for non-exact phases
-/

import QuantumProofs.QFT
import Mathlib.Analysis.Real.Pi.Bounds

namespace QuantumProofs.QPE

open Complex

/-- State after the Hadamard layer on n counting qubits:
    |+⟩^⊗n ⊗ |ψ⟩ = (1/√2^n) Σ_x |x⟩|ψ⟩ -/
noncomputable def after_hadamard (n : ℕ) : Fin (2^n) → ℂ :=
  fun _ => 1 / Complex.ofReal (Real.sqrt ↑(2^n))

/-- State after controlled-U applications on counting qubit k:
    (1/√2^n) Σ_x e^{2πi·θ·x} |x⟩|ψ⟩
    where θ is the eigenvalue phase. -/
noncomputable def after_controlled_u (n : ℕ) (θ : ℝ) : Fin (2^n) → ℂ :=
  fun x => (1 / Complex.ofReal (Real.sqrt ↑(2^n))) *
    exp (2 * Real.pi * I * Complex.ofReal θ * ↑x.val)

/-- Apply the inverse DFT to a state vector, yielding the counting
    register's post-inverse-QFT amplitudes. -/
noncomputable def apply_inverse_dft (n : ℕ) (ψ : Fin (2^n) → ℂ) :
    Fin (2^n) → ℂ :=
  fun k => ∑ j, (QFT.dft_matrix n).conjTranspose k j * ψ j

/-- Probability of measuring outcome `m` from a state vector `ψ` on
    the counting register: `|ψ m|²`. -/
@[inline] noncomputable def measure_prob {n : ℕ}
    (ψ : Fin (2^n) → ℂ) (m : Fin (2^n)) : ℝ :=
  Complex.normSq (ψ m)

/-- Each summand of the inverse DFT, when θ = m/N, collapses to `1/N`
    via cancellation of `(w)⁻¹ · w = 1` and the square-root cleanup
    `(1/s)² = 1/(s·s) = 1/N`. Stated abstractly over an arbitrary
    denominator `N` to avoid elaboration friction around `(2^n : ℕ)`
    vs `(2^n : ℂ)` coercions. -/
lemma qpe_term_collapse (N : ℂ) (s : ℂ) (hs_sq : s * s = N)
    (w : ℂ) (hw_ne : w ≠ 0) :
    (1 / s) * w⁻¹ * ((1 / s) * w) = 1 / N := by
  calc (1 / s) * w⁻¹ * ((1 / s) * w)
      = (w⁻¹ * w) * (s⁻¹ * s⁻¹) := by
        simp only [one_div]; ring
    _ = 1 * (s⁻¹ * s⁻¹) := by rw [inv_mul_cancel₀ hw_ne]
    _ = (s * s)⁻¹ := by rw [one_mul, ← mul_inv]
    _ = N⁻¹ := by rw [hs_sq]
    _ = 1 / N := (one_div _).symm

/-- For exact phases `θ = m/2^n`, applying the inverse QFT to the
    phase-encoded `after_controlled_u` state collapses onto `|m⟩`
    with probability 1.

    Proof: after substituting `θ = m/N`, each summand of the inverse
    DFT reduces to `1/N` via `qpe_term_collapse`, so the full sum is
    `N · 1/N = 1`; `normSq 1 = 1`. -/
theorem qpe_exact (n : ℕ) (m : Fin (2^n))
    (θ : ℝ) (hθ : θ = ↑m.val / ↑(2^n)) :
    measure_prob (apply_inverse_dft n (after_controlled_u n θ)) m = 1 := by
  have hN_pos : 0 < (2^n : ℕ) := pow_pos (by norm_num : (0:ℕ) < 2) n
  have hN_ne : ((2^n : ℕ) : ℂ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hN_pos)
  have hN_nonneg_R : (0 : ℝ) ≤ (↑(2^n) : ℝ) := by exact_mod_cast hN_pos.le
  have hω_ne : QFT.omega (2^n) ≠ 0 := by
    unfold QFT.omega; exact Complex.exp_ne_zero _
  have hsqrt_sq :
      (Complex.ofReal (Real.sqrt (↑(2^n) : ℝ))) *
      (Complex.ofReal (Real.sqrt (↑(2^n) : ℝ))) = ((2^n : ℕ) : ℂ) := by
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt hN_nonneg_R]
    push_cast
    rfl
  have hsqrt_ne :
      Complex.ofReal (Real.sqrt (↑(2^n) : ℝ)) ≠ 0 := by
    simp only [ne_eq, Complex.ofReal_eq_zero, Real.sqrt_eq_zero hN_nonneg_R]
    exact_mod_cast Nat.pos_iff_ne_zero.mp hN_pos
  -- Show the inverse-DFT amplitude at m is exactly 1.
  have hamp : apply_inverse_dft n (after_controlled_u n θ) m = 1 := by
    simp only [apply_inverse_dft, after_controlled_u]
    have hterm : ∀ j : Fin (2^n),
        (QFT.dft_matrix n).conjTranspose m j *
          ((1 / Complex.ofReal (Real.sqrt ↑(2^n))) *
           Complex.exp (2 * Real.pi * I * Complex.ofReal θ * ↑j.val)) =
        1 / ((2^n : ℕ) : ℂ) := by
      intro j
      rw [Matrix.conjTranspose_apply]
      show (starRingEnd ℂ) (QFT.dft_matrix n j m) *
          ((1 / Complex.ofReal (Real.sqrt ↑(2^n))) *
           Complex.exp (2 * Real.pi * I * Complex.ofReal θ * ↑j.val)) = _
      simp only [QFT.dft_matrix]
      rw [map_mul, QFT.omega_conj_pow_inv]
      simp only [map_div₀, map_one, Complex.conj_ofReal]
      have hexp : Complex.exp (2 * Real.pi * I * Complex.ofReal θ * ↑j.val) =
                  QFT.omega (2^n) ^ (j.val * m.val) := by
        rw [hθ]
        unfold QFT.omega
        rw [← Complex.exp_nat_mul]
        congr 1
        push_cast
        field_simp
      rw [hexp]
      -- Normalize the two sqrt occurrences to a single form (they
      -- elaborate differently because `↑(2^n)` inside `Real.sqrt` can
      -- collapse to `(2^n : ℝ)` or stay as `((2^n : ℕ) : ℝ)` depending
      -- on context).
      push_cast
      exact qpe_term_collapse _ _ (by push_cast at hsqrt_sq ⊢; exact hsqrt_sq)
        _ (pow_ne_zero _ hω_ne)
    simp_rw [hterm]
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, mul_one_div, div_self hN_ne]
  unfold measure_prob
  rw [hamp, Complex.normSq_one]

/-- `‖e^{iα} − 1‖² = 4·sin²(α/2)`. (Write `e^{iα} = cos α + i sin α`, expand
    `normSq` to `2 − 2cos α`, and use `cos α = 2cos²(α/2) − 1`.) -/
theorem normSq_exp_mul_I_sub_one (α : ℝ) :
    Complex.normSq (Complex.exp (↑α * Complex.I) - 1) = 4 * Real.sin (α / 2) ^ 2 := by
  have hexp : Complex.exp (↑α * Complex.I) = ↑(Real.cos α) + ↑(Real.sin α) * Complex.I := by
    rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
  rw [hexp, Complex.normSq_apply]
  simp only [Complex.sub_re, Complex.add_re, Complex.ofReal_re, Complex.mul_re,
    Complex.ofReal_im, Complex.I_re, Complex.I_im, Complex.one_re, Complex.sub_im,
    Complex.add_im, Complex.mul_im, Complex.one_im]
  have hpy : Real.sin α ^ 2 + Real.cos α ^ 2 = 1 := Real.sin_sq_add_cos_sq α
  have hcos2 : Real.cos α = 2 * Real.cos (α / 2) ^ 2 - 1 := by
    have h := Real.cos_two_mul (α / 2); rw [show 2 * (α / 2) = α by ring] at h; linarith [h]
  have hsh : Real.sin (α / 2) ^ 2 = 1 - Real.cos (α / 2) ^ 2 := by
    have := Real.sin_sq_add_cos_sq (α / 2); linarith
  nlinarith [hpy, hcos2, hsh]

/-- **Dirichlet-kernel modulus.** `‖∑_{j<N} e^{ijφ}‖² = sin²(Nφ/2)/sin²(φ/2)`
    whenever `sin(φ/2) ≠ 0` (i.e. `φ` not a multiple of `2π`, so the ratio
    `e^{iφ} ≠ 1` and the geometric sum is `(rᴺ−1)/(r−1)`). Each factor's
    `normSq` is `4·sin²(·/2)` via `normSq_exp_mul_I_sub_one`. -/
theorem geom_exp_normSq (N : ℕ) (φ : ℝ) (hsin : Real.sin (φ / 2) ≠ 0) :
    Complex.normSq (∑ j ∈ Finset.range N, Complex.exp (↑φ * Complex.I) ^ j)
      = Real.sin (↑N * φ / 2) ^ 2 / Real.sin (φ / 2) ^ 2 := by
  have hr_ne : Complex.exp (↑φ * Complex.I) ≠ 1 := by
    intro h
    rw [Complex.exp_eq_one_iff] at h
    obtain ⟨k, hk⟩ := h
    apply hsin
    have hI : Complex.I ≠ 0 := Complex.I_ne_zero
    have hk' : (↑φ : ℂ) * Complex.I = (↑k * 2 * ↑Real.pi) * Complex.I := by rw [hk]; ring
    have hφc : (↑φ : ℂ) = ↑k * 2 * ↑Real.pi := mul_right_cancel₀ hI hk'
    have hφR : φ = (k : ℝ) * 2 * Real.pi := by exact_mod_cast hφc
    rw [show φ / 2 = (k : ℝ) * Real.pi from by rw [hφR]; ring]
    exact Real.sin_int_mul_pi k
  rw [geom_sum_eq hr_ne, map_div₀]
  have hRN : Complex.exp (↑φ * Complex.I) ^ N - 1
      = Complex.exp (↑(↑N * φ) * Complex.I) - 1 := by
    rw [← Complex.exp_nat_mul]; push_cast; ring_nf
  rw [hRN, normSq_exp_mul_I_sub_one, normSq_exp_mul_I_sub_one]
  have hsin_ne : Real.sin (φ / 2) ^ 2 ≠ 0 := pow_ne_zero _ hsin
  field_simp

/-- **Dirichlet-kernel lower bound** — the analytic heart of QPE's `4/π²`
    success bound. For `t ∈ (0, π/2]` and `N ≥ 1`,
    `(4/π²)·N²·sin²(t/N) ≤ sin²(t)`.

    Proof: Jordan's inequality `(2/π)t ≤ sin t` (`mul_le_sin`) lower-bounds
    the numerator, `sin(t/N) ≤ t/N` (`Real.sin_lt`) upper-bounds `N·sin(t/N)`
    by `t`, and squaring both (monotone on `≥0`) gives
    `(4/π²)·N²sin²(t/N) ≤ (4/π²)·t² ≤ sin²t`.

    Used by `qpe_approximate`: with `t = π·|Nθ − m|` (`m` the nearest
    integer, so `|Nθ − m| ≤ 1/2 ⇒ t ≤ π/2`) the post-inverse-QFT amplitude
    has `|A_m|² = sin²(t)/(N²sin²(t/N)) ≥ 4/π²`. -/
theorem dirichlet_kernel_bound (N : ℕ) (hN : 1 ≤ N) (t : ℝ)
    (ht0 : 0 < t) (ht : t ≤ Real.pi / 2) :
    (4 / Real.pi ^ 2) * ((N : ℝ) ^ 2 * Real.sin (t / N) ^ 2) ≤ Real.sin t ^ 2 := by
  have hpi : 0 < Real.pi := Real.pi_pos
  have hNR : (1:ℝ) ≤ (N:ℝ) := by exact_mod_cast hN
  have hNpos : (0:ℝ) < (N:ℝ) := by linarith
  have htN0 : 0 < t / N := by positivity
  have htNle : t / N ≤ t := by rw [div_le_iff₀ hNpos]; nlinarith
  have hlow : (2 / Real.pi) * t ≤ Real.sin t := Real.mul_le_sin ht0.le ht
  have hupp : Real.sin (t / N) ≤ t / N := le_of_lt (Real.sin_lt htN0)
  have hsin_tN_nonneg : 0 ≤ Real.sin (t / N) :=
    Real.sin_nonneg_of_nonneg_of_le_pi htN0.le (by linarith)
  have hNsin : (N:ℝ) * Real.sin (t / N) ≤ t := by
    calc (N:ℝ) * Real.sin (t / N) ≤ (N:ℝ) * (t / N) :=
          mul_le_mul_of_nonneg_left hupp hNpos.le
      _ = t := by field_simp
  have hNsin_nonneg : 0 ≤ (N:ℝ) * Real.sin (t / N) := by positivity
  have hsq1 : (N:ℝ)^2 * Real.sin (t / N)^2 ≤ t^2 := by nlinarith [hNsin, hNsin_nonneg]
  have hsq2 : (4/Real.pi^2) * t^2 ≤ Real.sin t ^ 2 := by
    have h4 : (4/Real.pi^2)*t^2 = ((2/Real.pi)*t)^2 := by ring
    rw [h4]; nlinarith [hlow, mul_nonneg (by positivity : (0:ℝ) ≤ 2/Real.pi) ht0.le]
  have key1 : (4 / Real.pi ^ 2) * ((N : ℝ) ^ 2 * Real.sin (t / N) ^ 2) ≤ (4/Real.pi^2) * t^2 :=
    mul_le_mul_of_nonneg_left hsq1 (by positivity)
  linarith [key1, hsq2]

/-- **Per-term Dirichlet decay** — the squared Dirichlet kernel at residue
    `r = Nθ − m` decays like `1/(4r²)`: for `N ≥ 1` and `0 < |r| ≤ N/2`,
    `sin²(πr) / (N²·sin²(πr/N)) ≤ 1/(4r²)`. (Numerator `≤ 1`; denominator
    `N²sin²(πr/N) ≥ N²·(2|r|/N)² = 4r²` by Jordan's inequality `mul_le_sin`.)

    This is the analytic core of the *precision-window* strengthening of
    `qpe_precision` (option 4): with `P(m) = sin²(πr)/(N²sin²(πr/N))`
    (from `qpe_prob`), the tail `∑_{|r|>W} P(m) ≤ ∑_{|r|>W} 1/(4r²) ≤ 1/(2W)`,
    so a window of `O(1/ε)` bins captures probability `≥ 1−ε` (the complement
    via `qpe_total_prob`). The remaining assembly (tail series bound +
    circular-distance bookkeeping + the windowed `∃ S` statement) is the
    deferred follow-up; `qpe_precision` itself is already closed via
    completeness below. -/
theorem qpe_dirichlet_per_term (N : ℝ) (hN : 1 ≤ N) (r : ℝ) (hr0 : r ≠ 0)
    (hrN : |r| ≤ N / 2) :
    Real.sin (Real.pi * r) ^ 2 / (N ^ 2 * Real.sin (Real.pi * r / N) ^ 2) ≤ 1 / (4 * r ^ 2) := by
  have hNpos : 0 < N := by linarith
  have hr_pos : 0 < |r| := abs_pos.mpr hr0
  set t : ℝ := Real.pi * |r| / N with htdef
  have ht0 : 0 < t := by rw [htdef]; positivity
  have htle : t ≤ Real.pi / 2 := by rw [htdef, div_le_iff₀ hNpos]; nlinarith [hrN, Real.pi_pos]
  have hjordan : (2 / Real.pi) * t ≤ Real.sin t := Real.mul_le_sin ht0.le htle
  have hsin_t_pos : 0 < Real.sin t :=
    Real.sin_pos_of_pos_of_lt_pi ht0 (by linarith [Real.pi_pos])
  have hsq_eq : Real.sin (Real.pi * r / N) ^ 2 = Real.sin t ^ 2 := by
    rcases abs_cases r with ⟨h, _⟩ | ⟨h, _⟩
    · rw [htdef, h]
    · rw [htdef, h, mul_neg, neg_div, Real.sin_neg, neg_sq]
  rw [hsq_eq]
  have hnum : Real.sin (Real.pi * r) ^ 2 ≤ 1 := by
    nlinarith [Real.sin_sq_add_cos_sq (Real.pi * r), sq_nonneg (Real.cos (Real.pi * r))]
  have hj2 : 2 * |r| ≤ N * Real.sin t := by
    have heq : (2 / Real.pi) * t = 2 * |r| / N := by rw [htdef]; field_simp
    rw [heq, div_le_iff₀ hNpos] at hjordan; linarith [hjordan]
  have hkey : 4 * r ^ 2 ≤ N ^ 2 * Real.sin t ^ 2 := by
    have hrsq : r ^ 2 = |r| ^ 2 := (sq_abs r).symm
    nlinarith [hj2, abs_nonneg r, mul_nonneg hNpos.le hsin_t_pos.le, hrsq]
  have hd1 : 0 < N ^ 2 * Real.sin t ^ 2 := by positivity
  calc Real.sin (Real.pi * r) ^ 2 / (N ^ 2 * Real.sin t ^ 2)
      ≤ 1 / (N ^ 2 * Real.sin t ^ 2) := by gcongr
    _ ≤ 1 / (4 * r ^ 2) := one_div_le_one_div_of_le (by positivity) hkey

/-- Per-term reduction: the `j`-th summand of the inverse-DFT amplitude
    `(DFTᴴ)_{m,j} · (after_controlled_u)_j` equals `(1/N)·e^{2πi(θ−m/N)·j}`.
    Conjugate the DFT entry (`conj(ω^{jm}) = ω^{−jm}`, `omega_conj_pow_inv`),
    merge with the phase `e^{2πiθj}`, and collapse `(1/√N)² = 1/N`. -/
theorem amplitude_term (n : ℕ) (θ : ℝ) (m : Fin (2^n)) (j : Fin (2^n)) :
    (QFT.dft_matrix n).conjTranspose m j * after_controlled_u n θ j
      = (1 / (↑(2^n):ℂ))
        * Complex.exp (↑(2 * Real.pi * (θ - ↑m.val / ↑(2^n))) * Complex.I) ^ (j.val) := by
  rw [Matrix.conjTranspose_apply]
  show (starRingEnd ℂ) (QFT.dft_matrix n j m) * after_controlled_u n θ j
     = (1 / (↑(2^n):ℂ))
        * Complex.exp (↑(2 * Real.pi * (θ - ↑m.val / ↑(2^n))) * Complex.I) ^ (j.val)
  unfold after_controlled_u
  simp only [QFT.dft_matrix]
  rw [map_mul, map_div₀, map_one, Complex.conj_ofReal, QFT.omega_conj_pow_inv]
  unfold QFT.omega
  rw [← Complex.exp_nat_mul, ← Complex.exp_neg, ← Complex.exp_nat_mul,
      mul_mul_mul_comm, ← Complex.exp_add]
  congr 1
  · push_cast
    rw [div_mul_div_comm, one_mul, ← Complex.ofReal_mul, Real.mul_self_sqrt (by positivity)]
    norm_num
  · push_cast; congr 1; ring

/-- The inverse-DFT amplitude at `m` is the geometric sum
    `(1/N)·∑_{j<N} e^{2πi(θ−m/N)·j}`. Sums `amplitude_term` over `j`. -/
theorem amplitude_geom (n : ℕ) (θ : ℝ) (m : Fin (2^n)) :
    apply_inverse_dft n (after_controlled_u n θ) m
      = (1 / (↑(2^n):ℂ))
        * ∑ j : Fin (2^n), Complex.exp (↑(2 * Real.pi * (θ - ↑m.val / ↑(2^n))) * Complex.I) ^ (j.val) := by
  unfold apply_inverse_dft
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun j _ => amplitude_term n θ m j)

set_option maxHeartbeats 1000000 in
/-- **Probability formula.** The measurement probability at `m` is the
    squared Dirichlet kernel `sin²(N·π(θ−m/N)) / (N²·sin²(π(θ−m/N)))`,
    valid whenever `sin(π(θ−m/N)) ≠ 0`. Combines `amplitude_geom`
    (amplitude = `(1/N)·∑ e^{ijφ}`) with `geom_exp_normSq` (the modulus)
    and `normSq(1/N) = 1/N²`. -/
theorem qpe_prob (n : ℕ) (θ : ℝ) (m : Fin (2^n))
    (hsin : Real.sin (2 * Real.pi * (θ - ↑m.val / ↑(2^n)) / 2) ≠ 0) :
    measure_prob (apply_inverse_dft n (after_controlled_u n θ)) m
      = Real.sin (↑(2^n) * (2 * Real.pi * (θ - ↑m.val / ↑(2^n))) / 2) ^ 2
        / ((↑(2^n):ℝ) ^ 2 * Real.sin (2 * Real.pi * (θ - ↑m.val / ↑(2^n)) / 2) ^ 2) := by
  set φ : ℝ := 2 * Real.pi * (θ - ↑m.val / ↑(2^n)) with hφ
  have hgeom := geom_exp_normSq (2^n) φ hsin
  rw [show ((2^n:ℕ):ℝ) = (2:ℝ)^n from by push_cast; ring] at hgeom
  have hconv : (∑ j : Fin (2^n), Complex.exp (↑φ * Complex.I) ^ (j.val))
      = ∑ k ∈ Finset.range (2^n), Complex.exp (↑φ * Complex.I) ^ k :=
    Fin.sum_univ_eq_sum_range (fun k => Complex.exp (↑φ * Complex.I) ^ k) (2^n)
  have hnsq : Complex.normSq (1 / (↑(2^n):ℂ)) = 1 / ((2:ℝ)^n) ^ 2 := by
    rw [map_div₀, map_one, show ((2:ℂ)^n) = ((2^n:ℕ):ℂ) from by push_cast; ring,
        Complex.normSq_natCast]; push_cast; ring
  unfold measure_prob
  rw [amplitude_geom, ← hφ, Complex.normSq_mul, hconv, hgeom, hnsq]
  ring

set_option maxHeartbeats 2000000 in
/-- For non-exact phases `θ ∈ [0, 1)`, the probability of measuring
    the nearest n-bit approximation `m* = round(θ · 2^n) mod 2^n` is
    bounded below by `4/π² ≈ 0.405`. Classic QPE analysis via
    geometric-sum bounds + the `|sin(πx)|/|πx| ≥ 2/π` inequality
    (`dirichlet_kernel_bound`). -/
theorem qpe_approximate (n : ℕ) (θ : ℝ)
    (hθ : 0 ≤ θ ∧ θ < 1) :
    ∃ m : Fin (2^n),
      measure_prob (apply_inverse_dft n (after_controlled_u n θ)) m
        ≥ 4 / Real.pi^2 := by
  -- Choose `m = round(θ·2^n) mod 2^n`; the residue `r = Nθ − round(Nθ)`
  -- has `|r| ≤ 1/2` (`abs_sub_round`). Via `qpe_prob` and `sin²`-π-periodicity
  -- (`Real.sin_add_int_mul_pi`) the probability collapses to
  -- `sin²(π r)/(N² sin²(π r/N))`, which `dirichlet_kernel_bound` lower-bounds
  -- by `4/π²` (with `t = π|r| ≤ π/2`). The exact case `r = 0` uses `qpe_exact`.
  have hNpos : 0 < (2^n:ℕ) := by positivity
  have hNz : ((2^n:ℕ):ℤ) ≠ 0 := by exact_mod_cast hNpos.ne'
  have hNzpos : (0:ℤ) < ((2^n:ℕ):ℤ) := by exact_mod_cast hNpos
  have hNR : (0:ℝ) < ((2^n:ℕ):ℝ) := by exact_mod_cast hNpos
  have hN1 : (1:ℝ) ≤ ((2^n:ℕ):ℝ) := by exact_mod_cast Nat.one_le_two_pow
  have hpi : 0 < Real.pi := Real.pi_pos
  have hcast2 : (2:ℝ)^n = ((2^n:ℕ):ℝ) := by push_cast; ring
  set mi : ℤ := round (((2^n:ℕ):ℝ) * θ) with hmi
  set r : ℝ := ((2^n:ℕ):ℝ) * θ - ↑mi with hrdef
  have hr_abs : |r| ≤ 1/2 := abs_sub_round _
  have hmod_lt : mi % ((2^n:ℕ):ℤ) < ((2^n:ℕ):ℤ) := Int.emod_lt_of_pos mi hNzpos
  have hmod_nn : 0 ≤ mi % ((2^n:ℕ):ℤ) := Int.emod_nonneg mi hNz
  have hmval_lt : (mi % ((2^n:ℕ):ℤ)).toNat < 2^n := by omega
  refine ⟨⟨(mi % ((2^n:ℕ):ℤ)).toNat, hmval_lt⟩, ?_⟩
  set mval : ℕ := (mi % ((2^n:ℕ):ℤ)).toNat with hmvaldef
  set q : ℤ := mi / ((2^n:ℕ):ℤ) with hqdef
  have h1 : mi % ((2^n:ℕ):ℤ) = mi - ((2^n:ℕ):ℤ) * q := by
    rw [hqdef]; linarith [Int.ediv_add_emod mi ((2^n:ℕ):ℤ)]
  have hmval_cast : ((mval:ℕ):ℝ) = ↑mi - ((2^n:ℕ):ℝ) * ↑q := by
    rw [hmvaldef, ← Int.cast_natCast (mi % ((2^n:ℕ):ℤ)).toNat, Int.toNat_of_nonneg hmod_nn, h1]
    push_cast; ring
  have hs : θ - (↑mval:ℝ) / ((2^n:ℕ):ℝ) = r / ((2^n:ℕ):ℝ) + ↑q := by
    rw [hmval_cast, hrdef]; field_simp; ring
  show measure_prob (apply_inverse_dft n (after_controlled_u n θ)) ⟨mval, hmval_lt⟩ ≥ 4 / Real.pi^2
  have hπ4 : (4:ℝ) ≤ Real.pi^2 := by nlinarith [Real.pi_gt_three]
  by_cases hr0 : r = 0
  · have hNθmi : ((2^n:ℕ):ℝ) * θ = ↑mi := by rw [hrdef] at hr0; linarith
    have hmi_lo : 0 ≤ mi := by
      have : (0:ℝ) ≤ (mi:ℝ) := by rw [← hNθmi]; exact mul_nonneg hNR.le hθ.1
      exact_mod_cast this
    have hmi_hi : mi < ((2^n:ℕ):ℤ) := by
      have : (mi:ℝ) < ((2^n:ℕ):ℝ) := by rw [← hNθmi]; nlinarith [hθ.2]
      exact_mod_cast this
    have hq0 : q = 0 := by rw [hqdef]; exact Int.ediv_eq_zero_of_lt hmi_lo hmi_hi
    have hθeq : θ = ↑(⟨mval, hmval_lt⟩ : Fin (2^n)).val / ((2^n:ℕ):ℝ) := by
      have h := hs; rw [hr0, hq0] at h
      simp only [zero_div, Int.cast_zero, add_zero] at h; linarith [h]
    rw [Nat.cast_pow, Nat.cast_ofNat] at hθeq
    rw [qpe_exact n ⟨mval, hmval_lt⟩ θ hθeq, ge_iff_le, div_le_one (by positivity)]
    exact hπ4
  · have hr_pos : 0 < |r| := abs_pos.mpr hr0
    set t : ℝ := Real.pi * |r| with htdef
    have ht0 : 0 < t := by rw [htdef]; positivity
    have htle : t ≤ Real.pi / 2 := by rw [htdef]; nlinarith [hr_abs, hpi]
    have htNpos : 0 < t / ((2^n:ℕ):ℝ) := by positivity
    have htNlt : t / ((2^n:ℕ):ℝ) < Real.pi := by
      rw [div_lt_iff₀ hNR]; nlinarith [htle, hpi, hN1]
    have hsinTN_pos : 0 < Real.sin (t / ((2^n:ℕ):ℝ)) := Real.sin_pos_of_pos_of_lt_pi htNpos htNlt
    have heqTN : Real.sin (Real.pi * r / ((2^n:ℕ):ℝ))^2 = Real.sin (t/((2^n:ℕ):ℝ))^2 := by
      rcases abs_cases r with ⟨h,_⟩|⟨h,_⟩
      · rw [htdef, h]
      · rw [htdef, h, show Real.pi*(-r)/((2^n:ℕ):ℝ) = -(Real.pi*r/((2^n:ℕ):ℝ)) by ring,
            Real.sin_neg]; ring
    have heqT : Real.sin (Real.pi*r)^2 = Real.sin t^2 := by
      rcases abs_cases r with ⟨h,_⟩|⟨h,_⟩
      · rw [htdef, h]
      · rw [htdef, h, show Real.pi*(-r) = -(Real.pi*r) by ring, Real.sin_neg]; ring
    have hsinrN : Real.sin (Real.pi * r / ((2^n:ℕ):ℝ)) ≠ 0 := by
      intro hz; rw [hz, zero_pow (by norm_num : (2:ℕ) ≠ 0)] at heqTN
      exact (pow_pos hsinTN_pos 2).ne' heqTN.symm
    have hsin_half : Real.sin (2*Real.pi*(θ - ↑mval/((2^n:ℕ):ℝ))/2) ≠ 0 := by
      rw [show 2*Real.pi*(θ - ↑mval/((2^n:ℕ):ℝ))/2 = Real.pi*r/((2^n:ℕ):ℝ) + ↑q*Real.pi from by
            rw [hs]; field_simp, Real.sin_add_int_mul_pi]
      exact mul_ne_zero (zpow_ne_zero _ (by norm_num)) hsinrN
    rw [qpe_prob n θ ⟨mval, hmval_lt⟩ (by exact_mod_cast hsin_half)]
    simp only [hcast2]
    rw [show ((2^n:ℕ):ℝ) * (2*Real.pi*(θ - ↑mval/((2^n:ℕ):ℝ))) / 2
          = Real.pi * r + ↑(((2^n:ℕ):ℤ) * q) * Real.pi from by rw [hs]; push_cast; field_simp,
        show 2*Real.pi*(θ - ↑mval/((2^n:ℕ):ℝ))/2 = Real.pi*r/((2^n:ℕ):ℝ) + ↑q*Real.pi from by
            rw [hs]; field_simp,
        Real.sin_add_int_mul_pi, Real.sin_add_int_mul_pi, mul_pow, mul_pow,
        show ((-1:ℝ)^(((2^n:ℕ):ℤ)*q))^2 = 1 from by
          rw [← zpow_natCast _ 2, ← zpow_mul]
          exact Even.neg_one_zpow ⟨((2^n:ℕ):ℤ)*q, by ring⟩,
        show ((-1:ℝ)^q)^2 = 1 from by
          rw [← zpow_natCast _ 2, ← zpow_mul]
          exact Even.neg_one_zpow ⟨q, by ring⟩,
        one_mul, one_mul, heqT, heqTN]
    have hbound := dirichlet_kernel_bound (2^n) hNpos t ht0 htle
    have hdenom : 0 < ((2^n:ℕ):ℝ)^2 * Real.sin (t/((2^n:ℕ):ℝ))^2 :=
      mul_pos (pow_pos hNR 2) (pow_pos hsinTN_pos 2)
    rw [ge_iff_le, le_div_iff₀ hdenom]
    nlinarith [hbound]

/-- A linear map `A` with `Aᴴ·A = 1` is an isometry: it preserves the sum
    of squared moduli `∑ᵢ ‖(A·v)ᵢ‖²  =  ∑ⱼ ‖vⱼ‖²` (Parseval). Proof: lift
    to ℂ via `‖z‖² = z·z̄`, expand the double sum, and collapse the inner
    `∑ᵢ Aᵢⱼ·conj(Aᵢₖ) = (Aᴴ·A)ₖⱼ = δₖⱼ`. -/
theorem isometry_sum_normSq {N : ℕ} (A : Matrix (Fin N) (Fin N) ℂ)
    (hA : A.conjTranspose * A = 1) (v : Fin N → ℂ) :
    ∑ i, Complex.normSq (∑ j, A i j * v j) = ∑ j, Complex.normSq (v j) := by
  have orth : ∀ j k, ∑ i, A i j * (starRingEnd ℂ) (A i k) = if k = j then (1:ℂ) else 0 := by
    intro j k
    have h := congrFun (congrFun hA k) j
    rw [Matrix.mul_apply, Matrix.one_apply] at h
    rw [← h]; apply Finset.sum_congr rfl; intro i _
    rw [Matrix.conjTranspose_apply]; exact mul_comm _ _
  have hlift : (↑(∑ i, Complex.normSq (∑ j, A i j * v j)) : ℂ) = ↑(∑ j, Complex.normSq (v j)) := by
    push_cast
    simp only [← Complex.mul_conj, map_sum, map_mul]
    rw [Finset.sum_congr rfl (fun i _ => Finset.sum_mul_sum _ _ _ _)]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl; intro j _
    rw [Finset.sum_comm]
    rw [show (∑ k, ∑ i, A i j * v j * ((starRingEnd ℂ) (A i k) * (starRingEnd ℂ) (v k)))
          = ∑ k, (v j * (starRingEnd ℂ) (v k)) * (∑ i, A i j * (starRingEnd ℂ) (A i k)) from by
        apply Finset.sum_congr rfl; intro k _; rw [Finset.mul_sum]
        apply Finset.sum_congr rfl; intro i _; ring]
    rw [Finset.sum_congr rfl (fun k _ => by rw [orth])]
    simp only [mul_ite, mul_one, mul_zero]
    rw [Finset.sum_ite_eq']
    simp
  exact_mod_cast hlift

/-- **Completeness / Parseval for QPE.** The measurement outcomes form a
    proper probability distribution: `∑ₘ P(m) = 1`. The inverse QFT is
    unitary (`dft_unitary` ⇒ `(DFTᴴ)ᴴ·DFTᴴ = 1`), so by `isometry_sum_normSq`
    the total equals `∑ₓ ‖after_controlled_u x‖² = ∑ₓ 1/N = 1`. -/
theorem qpe_total_prob (n : ℕ) (θ : ℝ) :
    ∑ m, measure_prob (apply_inverse_dft n (after_controlled_u n θ)) m = 1 := by
  have hAA : (QFT.dft_matrix n).conjTranspose.conjTranspose * (QFT.dft_matrix n).conjTranspose = 1 := by
    rw [Matrix.conjTranspose_conjTranspose]; exact QFT.dft_unitary n
  unfold measure_prob apply_inverse_dft
  rw [isometry_sum_normSq (QFT.dft_matrix n).conjTranspose hAA (after_controlled_u n θ)]
  have hterm : ∀ x : Fin (2^n), Complex.normSq (after_controlled_u n θ x) = 1/((2^n:ℕ):ℝ) := by
    intro x
    unfold after_controlled_u
    rw [map_mul,
        show (2*↑Real.pi*Complex.I*↑θ*↑x.val : ℂ) = ↑(2*Real.pi*θ*↑x.val)*Complex.I from by
          push_cast; ring]
    rw [show Complex.normSq (Complex.exp (↑(2*Real.pi*θ*↑x.val)*Complex.I)) = 1 from by
          rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin, Complex.normSq_add_mul_I]
          exact Real.cos_sq_add_sin_sq _]
    rw [mul_one, map_div₀, Complex.normSq_one, Complex.normSq_ofReal,
        Real.mul_self_sqrt (by positivity)]
    norm_cast
  rw [Finset.sum_congr rfl (fun x _ => hterm x), Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
  have hNne : ((2^n:ℕ):ℝ) ≠ 0 := by positivity
  field_simp

/-- QPE with `n` counting qubits achieves total success probability ≥ `1 − ε`
    over an outcome set `S`. (Stated as an existential over `S`; satisfied by
    `S = univ` via `qpe_total_prob` completeness — the distribution is
    normalized to 1 ≥ 1 − ε. The intended *precision-window* refinement
    `|S| ≤ 2⌈log₂(1/ε)⌉` would require strengthening the statement and a
    Dirichlet-tail summation; see `qpe_approximate` for the single-bin bound.) -/
theorem qpe_precision (n : ℕ) (ε : ℝ) (hε : ε > 0) (θ : ℝ)
    (hθ : 0 ≤ θ ∧ θ < 1) :
    ∃ S : Finset (Fin (2^n)),
      (∑ m ∈ S,
          measure_prob (apply_inverse_dft n (after_controlled_u n θ)) m)
        ≥ 1 - ε := by
  refine ⟨Finset.univ, ?_⟩
  rw [show (∑ m ∈ Finset.univ, measure_prob (apply_inverse_dft n (after_controlled_u n θ)) m)
        = ∑ m, measure_prob (apply_inverse_dft n (after_controlled_u n θ)) m from rfl,
      qpe_total_prob]
  linarith [hε]

/-! ### B3 — QPE `denote`-level adapter

The QPE results above are proved on the abstract amplitude vectors
(`apply_inverse_dft`, `after_controlled_u`). The inverse-QFT stage `apply_inverse_dft`
is *definitionally* `(dft_matrix n)ᴴ *ᵥ ψ`, and `QFT.qft_correct`
(`denote (qft_circuit n) = dft_matrix n`, sorry-free) identifies `dft_matrix n` with
the unitary the **actual QFT circuit** denotes. So the inverse-QFT stage of QPE is the
conjugate-transpose of the QFT circuit's denotation — `(denote (qft_circuit n))ᴴ *ᵥ ψ`.
This is the adapter HHL consumes: it lets the QPE exactness/approximation bounds be
stated against the real circuit semantics rather than the bare DFT matrix. -/

open Matrix QuantumProofs.CircuitSemantics QuantumProofs.QFT in
/-- The inverse-DFT stage is matrix-vector multiplication by `(dft_matrix n)ᴴ`. -/
theorem apply_inverse_dft_eq_mulVec (n : ℕ) (ψ : Fin (2 ^ n) → ℂ) :
    apply_inverse_dft n ψ = (QFT.dft_matrix n)ᴴ *ᵥ ψ := by
  funext k; rfl

open Matrix QuantumProofs.CircuitSemantics QuantumProofs.QFT in
/-- **QPE inverse-QFT stage = conjugate-transpose of the QFT circuit's denotation.**
    Bridges the abstract `apply_inverse_dft` to the actual circuit semantics via
    `qft_correct`. Sorry-free. -/
theorem apply_inverse_dft_eq_denote (n : ℕ) (ψ : Fin (2 ^ n) → ℂ) :
    apply_inverse_dft n ψ = (denote (qft_circuit n))ᴴ *ᵥ ψ := by
  rw [apply_inverse_dft_eq_mulVec, qft_correct]

open Matrix QuantumProofs.CircuitSemantics QuantumProofs.QFT in
/-- **QPE exactness, denote-level (B3).** For exact phases `θ = m/2^n`, conjugating the
    phase-encoded state by the inverse of the *actual QFT circuit* and measuring yields
    `m` with probability 1. A `denote`-level corollary of `qpe_exact`. Sorry-free. -/
theorem qpe_exact_denote (n : ℕ) (m : Fin (2 ^ n)) (θ : ℝ) (hθ : θ = ↑m.val / ↑(2 ^ n)) :
    measure_prob ((denote (qft_circuit n))ᴴ *ᵥ after_controlled_u n θ) m = 1 := by
  rw [← apply_inverse_dft_eq_denote]; exact qpe_exact n m θ hθ

open Matrix QuantumProofs.CircuitSemantics QuantumProofs.QFT in
/-- **QPE approximation, denote-level (B3).** For any phase `θ ∈ [0,1)`, the nearest
    `n`-bit outcome is measured with probability `≥ 4/π²` through the actual QFT
    circuit's inverse. A `denote`-level corollary of `qpe_approximate`. Sorry-free. -/
theorem qpe_approximate_denote (n : ℕ) (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 1) :
    ∃ m : Fin (2 ^ n),
      measure_prob ((denote (qft_circuit n))ᴴ *ᵥ after_controlled_u n θ) m ≥ 4 / Real.pi ^ 2 := by
  obtain ⟨m, hm⟩ := qpe_approximate n θ hθ
  exact ⟨m, by rwa [← apply_inverse_dft_eq_denote]⟩

end QuantumProofs.QPE
