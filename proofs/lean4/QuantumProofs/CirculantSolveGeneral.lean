/-
  General-`n` circulant diagonalization (LEAN_EXPORT CORR-1.2).

  Lifts `CirculantSolve.qft_diagonalizes_circ2` (the `n = 1` case) to ALL `n`,
  for an arbitrary circulant `Matrix.circulant v` over `Fin (2ⁿ)`:

      dft_matrix n · circulant v · (dft_matrix n)ᴴ = diagonal (circEigen v)

  where `circEigen v k = Σ_m v(m)·ω^{k·m}` is the DFT of the first column (the
  eigenvalues). Combined with `qft_correct` (`denote (qft_circuit n) =
  dft_matrix n`) this is the general "the QFT diagonalizes a circulant"
  statement; the `n = 1` instance is `CirculantSolve.qft_diagonalizes_circ2`.

  ## Proof strategy (the slick route)

  Rather than expand the `F·C·Fᴴ` double sum and invoke orthogonality twice, we
  prove the **eigenvector property** in one step:

      dft_mul_circulant :  F · C = Λ · F           (Λ = diagonal (circEigen v))

  i.e. each DFT row is a left eigenvector. Then the diagonalization is immediate
  from the *already proved* `dft_unitary` (`F · Fᴴ = 1`):

      F · C · Fᴴ = (Λ · F) · Fᴴ = Λ · (F · Fᴴ) = Λ · 1 = Λ.

  The only real computation is the single reindex inside `dft_mul_circulant`:

      Σ_p ω^{j·p} · v(p − q)   ──[ p = m + q ]──►   ω^{j·q} · Σ_m v(m)·ω^{j·m}
                                                  = ω^{j·q} · circEigen v j

  using `Equiv.addRight q` for the bijection and `omega_pow_mod` to discharge the
  `mod N` that `Fin` addition introduces (`(m+q).val = (m.val+q.val) % N`), via
  `omega_periodic` (`ωᴺ = 1`).
-/

import QuantumProofs.QFT
import QuantumProofs.CirculantSolve
import Mathlib.LinearAlgebra.Matrix.Circulant
import Mathlib.Analysis.CStarAlgebra.Matrix

namespace QuantumProofs.CirculantSolveGeneral

open QuantumProofs.QFT Complex Matrix BigOperators Finset

/-- The eigenvalues of `circulant v`: the DFT of its first column,
    `circEigen v k = Σ_m v(m)·ω^{k·m}` (`ω = omega (2ⁿ)`). This is the spectrum of
    `circulant v` in *this* file's `+` exponent convention — the one that makes
    `dft_matrix·C·dft_matrixᴴ` diagonal. NOTE the Rust kernel
    (`circulant::eigenvalues`) uses the **opposite** sign `ω^{−pow·k}`; the two
    are complex conjugates and agree only when the spectrum is `k ↦ −k` symmetric
    (real-symmetric circulants, e.g. the heat operator, and every `N = 2` case
    where `k ≡ −k`). They are genuinely different scalars for an asymmetric
    circulant — so "matches the Rust eigenvalues" holds for the symmetric/`N=2`
    cases used here, not in general. -/
noncomputable def circEigen {n : ℕ} (v : Fin (2 ^ n) → ℂ) (k : Fin (2 ^ n)) : ℂ :=
  ∑ m : Fin (2 ^ n), v m * omega (2 ^ n) ^ (k.val * m.val)

/-- **Exponent periodicity:** `ω^{j·a} = ω^{j·(a mod N)}`. Kills the `mod N` that
    `Fin` addition introduces (`(m+q).val = (m.val+q.val) % N`). Proof: write
    `a = N·(a/N) + a%N`; the `N·(a/N)` part contributes `(ωᴺ)^… = 1`. -/
theorem omega_pow_mod (N j a : ℕ) (hN : 0 < N) :
    omega N ^ (j * a) = omega N ^ (j * (a % N)) := by
  conv_lhs => rw [← Nat.div_add_mod a N, Nat.mul_add, pow_add]
  rw [show j * (N * (a / N)) = N * (j * (a / N)) by ring, pow_mul, omega_periodic N hN,
      one_pow, one_mul]

/-- **Eigenvector property of the DFT for a circulant:** `F · C = Λ · F`, where
    `F = dft_matrix n`, `C = circulant v`, and `Λ = diagonal (circEigen v)`. Each
    row of `F` is a left eigenvector of `C` with eigenvalue `circEigen v j`. This
    is the whole content of the diagonalization; the rest is `dft_unitary`. -/
theorem dft_mul_circulant {n : ℕ} (v : Fin (2 ^ n) → ℂ) :
    dft_matrix n * Matrix.circulant v = Matrix.diagonal (circEigen v) * dft_matrix n := by
  have hN : 0 < 2 ^ n := pow_pos (by norm_num) n
  ext j q
  rw [Matrix.mul_apply, Matrix.diagonal_mul]
  -- The reindexed inner sum: `Σ_p ω^{j·p}·v(p−q) = ω^{j·q}·circEigen v j`.
  have key : (∑ p : Fin (2 ^ n), omega (2 ^ n) ^ (j.val * p.val) * v (p - q))
      = omega (2 ^ n) ^ (j.val * q.val) * circEigen v j := by
    -- reindex `p = m + q` (bijection `Equiv.addRight q`); then `(m+q) − q = m`.
    rw [← Equiv.sum_comp (Equiv.addRight q)
          (fun p => omega (2 ^ n) ^ (j.val * p.val) * v (p - q))]
    simp only [Equiv.coe_addRight, add_sub_cancel_right]
    rw [circEigen, Finset.mul_sum]
    refine Finset.sum_congr rfl (fun m _ => ?_)
    -- `ω^{j·(m+q).val}·v m = ω^{j·q}·(v m·ω^{j·m})`, via val_add + omega_pow_mod.
    rw [Fin.val_add, ← omega_pow_mod (2 ^ n) j.val (m.val + q.val) hN, Nat.mul_add, pow_add]
    ring
  -- Pull the `1/√N` out of the sum, apply `key`, then both sides agree by `ring`.
  simp only [dft_matrix, Matrix.circulant_apply, mul_assoc]
  rw [← Finset.mul_sum, key]
  ring

/-- **General circulant diagonalization (CORR-1.2).** The DFT unitary `F`
    conjugates any circulant `circulant v` (over `Fin (2ⁿ)`) to the diagonal of
    its eigenvalues `circEigen v` — the DFT of the first column. With
    `qft_correct` this says the `n`-qubit QFT diagonalizes any circulant; the
    `n = 1` case is `CirculantSolve.qft_diagonalizes_circ2`. Sorry-free. -/
theorem dft_diagonalizes_circulant {n : ℕ} (v : Fin (2 ^ n) → ℂ) :
    dft_matrix n * Matrix.circulant v * (dft_matrix n)ᴴ
      = Matrix.diagonal (circEigen v) := by
  rw [dft_mul_circulant, mul_assoc, dft_unitary, mul_one]

/-- **The `n`-qubit QFT circuit diagonalizes any circulant over `Fin (2ⁿ)`**
    (general `n`; power-of-two size — the QFT-addressable case).
    The headline CORR-1.2 statement: the unitary `denote (qft_circuit n)`
    conjugates `circulant v` to `diagonal (circEigen v)`. Immediate from
    `qft_correct` (`denote (qft_circuit n) = dft_matrix n`) +
    `dft_diagonalizes_circulant`. This is the all-`n` generalization of
    `CirculantSolve.qft_diagonalizes_circ2` (for `n = 1`, `circ2 a b =
    Matrix.circulant ![a,b]` and `circEigen ![a,b] = ![a+b, a-b]`). Sorry-free. -/
theorem qft_diagonalizes_circulant {n : ℕ} (v : Fin (2 ^ n) → ℂ) :
    CircuitSemantics.denote (qft_circuit n) * Matrix.circulant v
        * (CircuitSemantics.denote (qft_circuit n))ᴴ
      = Matrix.diagonal (circEigen v) := by
  rw [qft_correct]
  exact dft_diagonalizes_circulant v

/-- **General-`n` solve operator = `C⁻¹`.** When every eigenvalue is nonzero
    (`C = circulant v` nonsingular), the QFT-conjugation operator
    `S = Fᴴ · diag(1/circEigen v) · F` is a left inverse of `C` (`S · C = 1`),
    so `S = C⁻¹`. Read right-to-left, `S` is the quantum solver order: QFT, scale
    by `1/λ`, inverse-QFT — and `x = S·b` solves `C x = b`. Lifts
    `CirculantSolve.circ2_solve_operator` from `n = 1` to all `n`. Same proof:
    `F·C = Λ·F` (`dft_mul_circulant`) + `Fᴴ·F = 1` (`dft_unitary`). -/
theorem circulant_solve_operator {n : ℕ} (v : Fin (2 ^ n) → ℂ)
    (hv : ∀ k, circEigen v k ≠ 0) :
    (dft_matrix n)ᴴ * Matrix.diagonal (fun k => (circEigen v k)⁻¹) * dft_matrix n
        * Matrix.circulant v = 1 := by
  have hFtF : (dft_matrix n)ᴴ * dft_matrix n = 1 := mul_eq_one_comm.mp (dft_unitary n)
  have hΛ : Matrix.diagonal (fun k => (circEigen v k)⁻¹) * Matrix.diagonal (circEigen v)
      = (1 : Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ) := by
    rw [Matrix.diagonal_mul_diagonal]
    ext i j
    rcases eq_or_ne i j with h | h
    · subst h
      simp [Matrix.diagonal_apply_eq, Matrix.one_apply_eq, inv_mul_cancel₀ (hv i)]
    · simp [Matrix.diagonal_apply_ne _ h, Matrix.one_apply_ne h]
  calc
    (dft_matrix n)ᴴ * Matrix.diagonal (fun k => (circEigen v k)⁻¹) * dft_matrix n
          * Matrix.circulant v
        = (dft_matrix n)ᴴ * Matrix.diagonal (fun k => (circEigen v k)⁻¹)
            * (dft_matrix n * Matrix.circulant v) := by rw [mul_assoc]
      _ = (dft_matrix n)ᴴ * Matrix.diagonal (fun k => (circEigen v k)⁻¹)
            * (Matrix.diagonal (circEigen v) * dft_matrix n) := by rw [dft_mul_circulant]
      _ = (dft_matrix n)ᴴ * (Matrix.diagonal (fun k => (circEigen v k)⁻¹)
            * Matrix.diagonal (circEigen v)) * dft_matrix n := by
            simp only [mul_assoc]
      _ = (dft_matrix n)ᴴ * dft_matrix n := by rw [hΛ, mul_one]
      _ = 1 := hFtF

/-- **General-`n` noisy-solve deviation.** A noisy solver applying imperfect
    eigenvalue reciprocals `μ` (instead of the exact `1/circEigen v`) deviates
    from `C⁻¹` by exactly the QFT-conjugated error `Fᴴ · diag(μ − 1/λ) · F` —
    linear in the per-eigenvalue error, → 0 as `μ → 1/λ`. An exact operator
    identity (pure linearity of conjugation). Lifts
    `CirculantSolve.circ2_solve_noise_deviation` to all `n`. -/
theorem circulant_solve_noise_deviation {n : ℕ} (v : Fin (2 ^ n) → ℂ) (μ : Fin (2 ^ n) → ℂ) :
    (dft_matrix n)ᴴ * Matrix.diagonal μ * dft_matrix n
        - (dft_matrix n)ᴴ * Matrix.diagonal (fun k => (circEigen v k)⁻¹) * dft_matrix n
      = (dft_matrix n)ᴴ * Matrix.diagonal (fun k => μ k - (circEigen v k)⁻¹) * dft_matrix n := by
  have hdiag : Matrix.diagonal (fun k => μ k - (circEigen v k)⁻¹)
      = Matrix.diagonal μ - Matrix.diagonal (fun k => (circEigen v k)⁻¹) := by
    ext i j
    by_cases h : i = j
    · subst h; simp [Matrix.diagonal_apply_eq, Matrix.sub_apply]
    · simp [Matrix.diagonal_apply_ne _ h, Matrix.sub_apply]
  rw [hdiag, mul_sub, sub_mul]

/-- **Spectral structure of the approximate (HHL-style) solve error — CORR-2,
    operator level.** Write `S(d) = Fᴴ·diag(d)·F` for the QFT-conjugation solver
    using eigenvalue-reciprocals `d`. The QFT diagonalizes the difference of two
    such solvers, `S(μ) − S(1/circEigen v)`, to `diagonal (μ − 1/circEigen v)`.
    So that error operator's eigenvalues are *exactly* the per-mode reciprocal
    differences `εₖ = μₖ − (circEigen v k)⁻¹`. **When `C = circulant v` is
    nonsingular** (`∀k, circEigen v k ≠ 0`), `S(1/circEigen v) = C⁻¹`
    (`circulant_solve_operator`), so this is the deviation of the noisy solver
    from the true inverse — the `εₖ` are what finite-precision QPE + the
    truncated controlled rotation leave in HHL, and the error is controlled
    **mode-by-mode** by eigenvalue accuracy (the operator/spectral content of
    "`≈ within ε`"). NOTE: this theorem itself assumes NO nonsingularity — it is
    an identity in `μ` and the symbol `(circEigen v)⁻¹` (which is Lean's `0` on
    singular modes); the `C⁻¹` reading needs the nonsingularity hypothesis above.
    The scalar bound `‖S(μ) − C⁻¹‖₂ ≤ maxₖ|εₖ|` is the analysis follow-on (needs
    the L2 operator-norm + `F`-isometry, not developed here). Proof: the error is
    `Fᴴ·diag(ε)·F` (`circulant_solve_noise_deviation`); conjugating back by the
    unitary `F` (`dft_unitary`, twice) cancels to `diag(ε)`. Sorry-free. -/
theorem qft_diagonalizes_solve_error {n : ℕ} (v : Fin (2 ^ n) → ℂ) (μ : Fin (2 ^ n) → ℂ) :
    dft_matrix n
        * ((dft_matrix n)ᴴ * Matrix.diagonal μ * dft_matrix n
           - (dft_matrix n)ᴴ * Matrix.diagonal (fun k => (circEigen v k)⁻¹) * dft_matrix n)
        * (dft_matrix n)ᴴ
      = Matrix.diagonal (fun k => μ k - (circEigen v k)⁻¹) := by
  rw [circulant_solve_noise_deviation]
  have h1 : dft_matrix n * (dft_matrix n)ᴴ = 1 := dft_unitary n
  calc
    dft_matrix n
          * ((dft_matrix n)ᴴ * Matrix.diagonal (fun k => μ k - (circEigen v k)⁻¹) * dft_matrix n)
          * (dft_matrix n)ᴴ
        = (dft_matrix n * (dft_matrix n)ᴴ)
            * Matrix.diagonal (fun k => μ k - (circEigen v k)⁻¹)
            * (dft_matrix n * (dft_matrix n)ᴴ) := by noncomm_ring
      _ = Matrix.diagonal (fun k => μ k - (circEigen v k)⁻¹) := by
          simp only [h1, Matrix.one_mul, Matrix.mul_one]

/-- The eigenvalues for `N = 2`: `circEigen ![a,b] = ![a+b, a−b]` (`ω₂ = -1`).
    Formally backs the `n = 1` consistency claim. -/
theorem circEigen_two (a b : ℂ) :
    circEigen (n := 1) ![a, b] = ![a + b, a - b] := by
  ext k
  fin_cases k <;>
    simp [circEigen, Fin.sum_univ_two, QuantumProofs.CirculantSolve.omega2] <;> ring

/-- **Consistency:** `CirculantSolve.qft_diagonalizes_circ2` is exactly the
    `v = ![a,b]` instance of the general `qft_diagonalizes_circulant` — recovered
    via `circ2 = Matrix.circulant ![a,b]` (`circ2_eq_circulant`) and
    `circEigen ![a,b] = ![a+b,a-b]` (`circEigen_two`). So the general theorem
    genuinely subsumes the hand-proved `n = 1` case. -/
theorem qft_diagonalizes_circ2_from_general (a b : ℂ) :
    CircuitSemantics.denote (qft_circuit 1) * QuantumProofs.CirculantSolve.circ2 a b
        * (CircuitSemantics.denote (qft_circuit 1))ᴴ
      = Matrix.diagonal ![a + b, a - b] := by
  rw [QuantumProofs.CirculantSolve.circ2_eq_circulant, qft_diagonalizes_circulant,
      circEigen_two]

-- ===========================================================================
-- CORR-2, scalar level: the quantitative `‖·‖₂ ≤ ε` bound.
--
-- The operator-level `qft_diagonalizes_solve_error` says the noisy-solve error
-- `E = S(μ) − S(1/λ)` diagonalizes (via the QFT) to `diag(εₖ)` with per-mode
-- reciprocal errors `εₖ = μₖ − (circEigen v k)⁻¹`.  Here we turn that into the
-- quantitative statement the correspondence theorem wants: in the L2 operator
-- norm, `‖E‖₂ ≤ maxₖ |εₖ|`.  Since `E = Fᴴ·diag(ε)·F` with `F` unitary
-- (`‖F‖ = 1`), submultiplicativity of the operator norm gives
-- `‖E‖ ≤ ‖Fᴴ‖·‖diag(ε)‖·‖F‖ = ‖ε‖∞`, and `‖diag(ε)‖₂ = ‖ε‖∞` (the sup norm of
-- the diagonal) is exactly `maxₖ|εₖ|`.  So if QPE + the truncated controlled
-- rotation keep every eigenvalue-reciprocal accurate to `ε`, the HHL-style
-- solve operator is within `ε` of the ideal `C⁻¹` — the quantitative
-- "`≈ within ε`" of the quantum↔classical correspondence.
--
-- Uses mathlib's scoped L2 operator norm on matrices
-- (`Matrix.Norms.L2Operator`): the identification `Matrix n n ℂ ≃ CLM on
-- EuclideanSpace`, with `l2_opNorm_mul` (submultiplicative),
-- `l2_opNorm_conjTranspose` (`‖Aᴴ‖ = ‖A‖`) and `l2_opNorm_diagonal`
-- (`‖diagonal v‖ = ‖v‖∞`).
-- ===========================================================================

section OperatorNorm

open scoped Matrix.Norms.L2Operator

/-- The `Fin (2ⁿ)` index type is nonempty (size `2ⁿ ≥ 1`). -/
private instance instNonemptyFinPow (n : ℕ) : Nonempty (Fin (2 ^ n)) :=
  ⟨⟨0, pow_pos (by norm_num) n⟩⟩

/-- The identity matrix has L2 operator norm `1`: `1 = diagonal (fun _ ↦ 1)`, and
    the sup norm of the all-ones vector over the nonempty index `Fin (2ⁿ)` is `1`. -/
private lemma l2_opNorm_one_pow (n : ℕ) :
    ‖(1 : Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ)‖ = 1 := by
  rw [← Matrix.diagonal_one, Matrix.l2_opNorm_diagonal, pi_norm_const, norm_one]

/-- The DFT matrix is an L2 isometry: `‖dft_matrix n‖₂ = 1`.  From the C*-identity
    `‖Fᴴ·F‖ = ‖F‖²` and `Fᴴ·F = 1` (unitarity), `‖F‖² = 1`, so `‖F‖ = 1`
    (`‖F‖ ≥ 0` rules out `−1`). -/
private lemma l2_opNorm_dft (n : ℕ) : ‖dft_matrix n‖ = 1 := by
  have hFtF : (dft_matrix n)ᴴ * dft_matrix n = 1 := mul_eq_one_comm.mp (dft_unitary n)
  have hsq : ‖dft_matrix n‖ * ‖dft_matrix n‖ = 1 := by
    rw [← Matrix.l2_opNorm_conjTranspose_mul_self, hFtF, l2_opNorm_one_pow]
  rcases mul_self_eq_one_iff.mp hsq with h | h
  · exact h
  · exfalso; linarith [norm_nonneg (dft_matrix n)]

/-- **CORR-2 (scalar/operator-norm level).** In the L2 operator norm, the
    noisy-solve error operator `S(μ) − S(1/circEigen v) = Fᴴ·diag(ε)·F` is
    bounded by the sup norm of the per-mode reciprocal errors
    `εₖ = μₖ − (circEigen v k)⁻¹` — i.e. `‖E‖₂ ≤ maxₖ |εₖ|`.  Under nonsingularity
    (`∀k, circEigen v k ≠ 0`) `S(1/circEigen v) = C⁻¹` (`circulant_solve_operator`),
    so this is exactly `‖S(μ) − C⁻¹‖₂ ≤ maxₖ|εₖ|`.  Proof: rewrite the error to
    `Fᴴ·diag(ε)·F` (`circulant_solve_noise_deviation`), then submultiplicativity
    `‖Fᴴ·diag(ε)·F‖ ≤ ‖Fᴴ‖·‖diag(ε)‖·‖F‖` with `‖Fᴴ‖ = ‖F‖ = 1` (`l2_opNorm_dft`)
    and `‖diag(ε)‖ = ‖ε‖∞` (`l2_opNorm_diagonal`). -/
theorem solve_error_l2_opNorm_le {n : ℕ} (v μ : Fin (2 ^ n) → ℂ) :
    ‖(dft_matrix n)ᴴ * Matrix.diagonal μ * dft_matrix n
        - (dft_matrix n)ᴴ * Matrix.diagonal (fun k => (circEigen v k)⁻¹) * dft_matrix n‖
      ≤ ‖fun k => μ k - (circEigen v k)⁻¹‖ := by
  rw [circulant_solve_noise_deviation]
  set ε : Fin (2 ^ n) → ℂ := fun k => μ k - (circEigen v k)⁻¹ with hε
  calc ‖(dft_matrix n)ᴴ * Matrix.diagonal ε * dft_matrix n‖
      ≤ ‖(dft_matrix n)ᴴ * Matrix.diagonal ε‖ * ‖dft_matrix n‖ := Matrix.l2_opNorm_mul _ _
    _ ≤ ‖(dft_matrix n)ᴴ‖ * ‖Matrix.diagonal ε‖ * ‖dft_matrix n‖ := by
        gcongr; exact Matrix.l2_opNorm_mul _ _
    _ = ‖ε‖ := by
        rw [Matrix.l2_opNorm_conjTranspose, l2_opNorm_dft, Matrix.l2_opNorm_diagonal,
            one_mul, mul_one]

/-- **Quantitative correspondence (the `≈ within ε` form).** If finite-precision
    QPE + the truncated controlled rotation keep *every* eigenvalue-reciprocal
    accurate to `M` (`∀k, ‖μₖ − (circEigen v k)⁻¹‖ ≤ M`), then the HHL-style solve
    operator deviates from the ideal `S(1/circEigen v)` (`= C⁻¹` when nonsingular)
    by at most `M` in the L2 operator norm.  Immediate from
    `solve_error_l2_opNorm_le` + `pi_norm_le_iff_of_nonneg`. -/
theorem solve_error_l2_opNorm_le_of_modes {n : ℕ} (v μ : Fin (2 ^ n) → ℂ) (M : ℝ)
    (hM : ∀ k, ‖μ k - (circEigen v k)⁻¹‖ ≤ M) :
    ‖(dft_matrix n)ᴴ * Matrix.diagonal μ * dft_matrix n
        - (dft_matrix n)ᴴ * Matrix.diagonal (fun k => (circEigen v k)⁻¹) * dft_matrix n‖
      ≤ M := by
  have hM0 : 0 ≤ M := le_trans (norm_nonneg _) (hM (Classical.arbitrary _))
  exact le_trans (solve_error_l2_opNorm_le v μ) ((pi_norm_le_iff_of_nonneg hM0).mpr hM)

end OperatorNorm

end QuantumProofs.CirculantSolveGeneral
