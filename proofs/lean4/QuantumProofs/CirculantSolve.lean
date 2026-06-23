/-
  Circulant `quantum ≡ classical` correspondence (LEAN_EXPORT CORR-1).

  The classical circulant solver (`crates/omega-core/src/circulant.rs`,
  `solve_via_dft`) solves `C x = b` by **DFT-diagonalization**: `x = F⁻¹ Λ⁻¹ F b`,
  where `F` is the DFT and `Λ = diag(eigenvalues)`. The *quantum* solver performs
  the same diagonalization on a register: QFT, then scale by `1/λ`, then inverse
  QFT. This file proves the bridge the whole correspondence rests on:

  **the unitary the QFT circuit implements (`denote (qft_circuit n)` = the DFT,
  by `qft_correct`) diagonalizes a circulant matrix to `diag(eigenvalues)`** —
  exactly the diagonalization `solve_via_dft` relies on.

  SCOPE: this is the explicit, fully-computed `n = 1` (2×2) case — useful as a
  worked example and the entry point. It proves at that size: (1)
  `qft_diagonalizes_circ2`: `F · C · Fᴴ = Λ` (`C⁻¹` is not unitary, so the
  statement is this conjugation, NOT `denote circuit = C⁻¹`); (2)
  `circ2_solve_operator`: the full solve operator `Fᴴ · Λ⁻¹ · F = C⁻¹` (QFT →
  scale by `1/λ` → inverse-QFT), so `x = (Fᴴ Λ⁻¹ F) b` solves `C x = b`; and (3)
  `circ2_solve_noise_deviation`: the noisy-solve deviation.
  **All three are now proved for GENERAL `n`** in `CirculantSolveGeneral`
  (`dft_diagonalizes_circulant` / `circulant_solve_operator` /
  `circulant_solve_noise_deviation`), and `qft_diagonalizes_circ2_from_general`
  shows this `n = 1` case is exactly the `v = ![a,b]` instance. `C` is literally
  a classical matrix (exact encoding).

  First instance: the `n = 1` (2×2) circulant, where `dft_matrix 1 = H` and the
  eigenvalues are `{a+b, a-b}` — equal to the Rust `Circulant::eigenvalues`
  formula `λ_k = Σⱼ cⱼ ω^{−powⱼ·k}` for terms `[(0,a),(1,b)]` on `N = 2`. (For
  `N = 2` the `±` exponent sign is immaterial since `k ≡ −k mod 2`; in general
  this file's DFT uses the `+` convention, the conjugate of the Rust kernel's —
  see `CirculantSolveGeneral.circEigen`.) General-`n` is proved in
  `CirculantSolveGeneral` (CORR-1.2).

  ## Proof architecture (lemma dependency chain)

  The four headline theorems all rest on one reusable idea — **factor the DFT
  into a scalar times an integer matrix, so the hard algebra happens over `ℤ`
  (closed by `ring`/`simp`) and the only `√2` bookkeeping is a single scalar
  collapse.** The same trick proves `BellPrep.bell_unitary`.

  ```
  qft_correct            (QFT.lean)   denote (qft_circuit 1) = dft_matrix 1
  dft_unitary            (QFT.lean)   F · Fᴴ = 1
      │
  omega2 ───► dft1_eq                 dft_matrix 1 = (1/√2) • H2   (scalar • integer matrix)
  H2_conjTranspose                    H2ᴴ = H2                     (H2 is real-symmetric)
  H2_diag                             H2 · C · H2 = 2 • diag(a+b, a-b)   (integer arithmetic)
      │
      ├─► qft_diagonalizes_circ2      F · C · Fᴴ = diag(a+b, a-b)
      │        │
      │        ├─► circ2_solve_operator       Fᴴ · Λ⁻¹ · F · C = 1     (so FᴴΛ⁻¹F = C⁻¹)
      │        └─► circ2_solve_noise_deviation  noisy-solve error = Fᴴ · diag(ε) · F
      │
  circ2_eq_circulant                  circ2 a b = Matrix.circulant ![a,b]  (mathlib bridge)
  ```

  Why the scalar/integer split (`dft1_eq` + `H2_diag`): a direct entrywise proof
  of `F C Fᴴ = Λ` would carry `1/√2` factors through all 16 entries and stall
  `simp`/`ring` on `√2`. Instead `qft_diagonalizes_circ2` pulls the two `(1/√2)`
  scalars out of the triple product (via `Matrix.smul_mul`/`mul_smul`/`smul_smul`),
  leaving the *integer* identity `H2·C·H2 = 2•diag` (pure `ring`), and finishes
  with one scalar fact `(1/√2)·(1/√2)·2 = 1`.
-/

import QuantumProofs.QFT
import Mathlib.LinearAlgebra.Matrix.Circulant

namespace QuantumProofs.CirculantSolve

open QuantumProofs.CircuitSemantics QuantumProofs.QFT Complex Matrix

/-- The **unnormalized** real DFT-2 matrix (entries in `{±1}`). The genuine
    DFT-2 / Hadamard is `(1/√2)·H2`; keeping the normalization factor *out* of
    `H2` is the whole point — it lets the core identity `H2·C·H2 = 2•diag`
    (`H2_diag`) be proved by integer `ring`, with the `√2` handled once, later. -/
def H2 : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1, 1; 1, -1]

/-- A 2×2 circulant `C = !![a, b; b, a]` (first column `(a, b)`). The smallest
    nontrivial circulant; `circ2_eq_circulant` shows it is the `N = 2` case of
    mathlib's `Matrix.circulant`. -/
def circ2 (a b : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![a, b; b, a]

/-- `circ2 a b` is exactly the size-2 case of mathlib's general `Matrix.circulant`
    (first column `![a, b]`; recall `Matrix.circulant v i j = v (i - j)`). This
    confirms `circ2` is genuinely a circulant, and pins it as the `N = 2` instance
    of the general circulant theory a full CORR-1.2 would diagonalize. -/
theorem circ2_eq_circulant (a b : ℂ) : circ2 a b = Matrix.circulant ![a, b] := by
  -- Decide all 4 entries: `circulant ![a,b] i j = ![a,b] (i - j)`, and on `Fin 2`
  -- the subtraction wraps (`0 - 1 = 1`), so off-diagonals read `b`, diagonals `a`.
  ext i j
  fin_cases i <;> fin_cases j <;> simp [circ2, Matrix.circulant_apply] <;> rfl

/-- `ω₂ = e^{2πi/2} = e^{iπ} = -1`, the primitive 2nd root of unity. Needed to
    evaluate `dft_matrix 1`'s entries `(1/√2)·ω₂^{jk}` in `dft1_eq`. -/
theorem omega2 : omega 2 = -1 := by
  unfold omega
  -- `2·π·I / (2:ℕ) = π·I` (the nat-cast 2 is the scalar 2), then `e^{iπ} = -1`.
  rw [show (2 * (Real.pi : ℂ) * I / ((2 : ℕ) : ℂ)) = (Real.pi : ℂ) * I by push_cast; ring]
  exact Complex.exp_pi_mul_I

/-- **Role: the scalar/integer split.** `dft_matrix 1 = (1/√2) • H2`, i.e. the
    DFT-2 matrix is the normalized Hadamard. Writing it as `scalar • (integer
    matrix)` is what lets every downstream proof do its matrix algebra over `H2`
    (integers) and defer all `√2` reasoning to a single scalar step. Proof: the
    four entries are `(1/√2)·ω₂^{j·k} = (1/√2)·(±1)`, with `ω₂ = -1` by `omega2`. -/
theorem dft1_eq : dft_matrix 1 = (Complex.ofReal (Real.sqrt 2))⁻¹ • H2 := by
  ext j k
  fin_cases j <;> fin_cases k <;>
    simp [dft_matrix, H2, omega2, Matrix.smul_apply, smul_eq_mul, one_div]

/-- `H2ᴴ = H2`: `H2` has real entries and is symmetric, so conjugate-transpose
    is the identity on it. Used in `qft_diagonalizes_circ2` to turn `Fᴴ` (which
    `dft1_eq` expands to `(1/√2) • H2ᴴ`) back into `(1/√2) • H2`. -/
theorem H2_conjTranspose : H2ᴴ = H2 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [H2, Matrix.conjTranspose_apply]

/-- **Role: the integer core of the diagonalization.** `H2 · C · H2 = 2 • diag(a+b,
    a-b)`. Once the `(1/√2)` scalars are pulled out of `F·C·Fᴴ`, *this* is the
    matrix identity that remains — pure integer arithmetic (`Fin.sum_univ_two`
    expands the 2×2 products, `ring` closes each of the 4 entries). The factor
    `2` here cancels the `(1/√2)² = 1/2` outside, giving the clean `diag(a+b,a-b)`. -/
theorem H2_diag (a b : ℂ) :
    H2 * circ2 a b * H2 = (2 : ℂ) • Matrix.diagonal ![a + b, a - b] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [H2, circ2, Matrix.mul_apply, Fin.sum_univ_two, Matrix.smul_apply,
          Matrix.diagonal, smul_eq_mul] <;> ring

/-- **Circulant diagonalization correspondence (CORR-1, n = 1).** The unitary
    the QFT circuit denotes (`= dft_matrix 1`, by `qft_correct`) conjugates the
    2×2 circulant `C = !![a,b;b,a]` to `diag(a+b, a-b)` — its eigenvalues, the
    same `Λ` the classical `solve_via_dft` diagonalizes against. Sorry-free. -/
theorem qft_diagonalizes_circ2 (a b : ℂ) :
    denote (qft_circuit 1) * circ2 a b * (denote (qft_circuit 1))ᴴ
      = Matrix.diagonal ![a + b, a - b] := by
  -- `√2 · √2 = 2`, the one scalar fact the whole proof needs.
  have hsqrt : Complex.ofReal (Real.sqrt 2) * Complex.ofReal (Real.sqrt 2) = 2 := by
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by norm_num)]; norm_num
  -- `(1/√2)` is real, so it is its own conjugate (needed for the `Fᴴ` factor).
  have hstar : star ((Complex.ofReal (Real.sqrt 2))⁻¹) = (Complex.ofReal (Real.sqrt 2))⁻¹ := by
    simp
  -- Rewrite chain — read it as the scalar/integer split in action:
  --   qft_correct      : denote (qft_circuit 1) ↦ dft_matrix 1
  --   dft1_eq          : dft_matrix 1 ↦ (1/√2) • H2     (both the bare F and the Fᴴ)
  --   conjTranspose_smul + H2_conjTranspose + hstar : Fᴴ ↦ (1/√2) • H2
  --   smul_mul/mul_smul/smul_smul : pull both (1/√2) scalars out front of the product
  --   H2_diag          : the remaining integer core H2·C·H2 ↦ 2 • diag(a+b,a-b)
  --   smul_smul        : collapse the nested scalars into one  ((1/√2)·(1/√2)·2) • diag
  rw [qft_correct, dft1_eq, Matrix.conjTranspose_smul, H2_conjTranspose, hstar,
      Matrix.smul_mul, Matrix.smul_mul, Matrix.mul_smul, smul_smul, H2_diag, smul_smul]
  -- The accumulated scalar is `(1/√2)·(1/√2)·2 = 1`, so `Λ` is left exactly.
  rw [show (Complex.ofReal (Real.sqrt 2))⁻¹ * (Complex.ofReal (Real.sqrt 2))⁻¹ * 2 = 1 by
        field_simp; linear_combination -hsqrt]
  rw [one_smul]

/-- **Solve relation (CORR-1, n = 1).** The QFT-conjugation operator
    `S = Fᴴ · Λ⁻¹ · F` (where `F = denote (qft_circuit 1)` and
    `Λ = diag(a+b, a-b)`) is a left inverse of the circulant `C` — i.e.
    `S · C = 1`, so `S = C⁻¹`. Read right-to-left, `S` is exactly the quantum
    solver order on a state: **QFT (`F`), scale by `1/λ` (`Λ⁻¹`), inverse-QFT
    (`Fᴴ`)** — and `x = S · b` solves `C x = b`, the same `x` the classical
    `solve_via_dft` computes. This is the `Λ⁻¹`-scaling + recombination link
    (one size; general `n` is CORR-1.2). Requires the eigenvalues nonzero
    (`C` nonsingular). Sorry-free. -/
theorem circ2_solve_operator (a b : ℂ) (hp : a + b ≠ 0) (hm : a - b ≠ 0) :
    (denote (qft_circuit 1))ᴴ * Matrix.diagonal ![(a + b)⁻¹, (a - b)⁻¹]
        * denote (qft_circuit 1) * circ2 a b = 1 := by
  -- Abbreviate the QFT unitary as `F`; the proof is pure operator algebra:
  --   C = Fᴴ Λ F (eigendecomposition) ⇒ C⁻¹ = Fᴴ Λ⁻¹ F, shown as `(Fᴴ Λ⁻¹ F)·C = 1`.
  set F := denote (qft_circuit 1) with hF
  -- `F` is unitary: `F Fᴴ = 1` (dft_unitary), hence also `Fᴴ F = 1` (square matrices).
  have hFFt : F * Fᴴ = 1 := by rw [hF, qft_correct]; exact dft_unitary 1
  have hFtF : Fᴴ * F = 1 := mul_eq_one_comm.mp hFFt
  -- Intertwining `F · C = Λ · F`, from `F · C · Fᴴ = Λ` by right-multiplying by `F`
  -- and using `Fᴴ · F = 1`. (Kept left-associated so `h` matches verbatim.)
  have hFC : F * circ2 a b = Matrix.diagonal ![a + b, a - b] * F := by
    have h := qft_diagonalizes_circ2 a b
    calc
      F * circ2 a b = F * circ2 a b * (Fᴴ * F) := by rw [hFtF, Matrix.mul_one]
      _ = F * circ2 a b * Fᴴ * F := by rw [← Matrix.mul_assoc]
      _ = Matrix.diagonal ![a + b, a - b] * F := by rw [← hF] at h; rw [h]
  -- The diagonal inverse: `Λ⁻¹ · Λ = 1` (entrywise `(λₖ)⁻¹·λₖ = 1`, needs `λₖ ≠ 0`).
  have hΛ : Matrix.diagonal ![(a + b)⁻¹, (a - b)⁻¹] * Matrix.diagonal ![a + b, a - b]
      = (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
    rw [Matrix.diagonal_mul_diagonal]
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [Matrix.diagonal] <;> field_simp
  -- Assemble: Fᴴ Λ⁻¹ F C = Fᴴ Λ⁻¹ (F C) = Fᴴ Λ⁻¹ (Λ F) = Fᴴ (Λ⁻¹ Λ) F = Fᴴ F = 1.
  calc
    Fᴴ * Matrix.diagonal ![(a + b)⁻¹, (a - b)⁻¹] * F * circ2 a b
        = Fᴴ * Matrix.diagonal ![(a + b)⁻¹, (a - b)⁻¹] * (F * circ2 a b) := by
          rw [Matrix.mul_assoc]
    _ = Fᴴ * Matrix.diagonal ![(a + b)⁻¹, (a - b)⁻¹]
          * (Matrix.diagonal ![a + b, a - b] * F) := by rw [hFC]
    _ = Fᴴ * (Matrix.diagonal ![(a + b)⁻¹, (a - b)⁻¹] * Matrix.diagonal ![a + b, a - b]) * F := by
          rw [Matrix.mul_assoc, Matrix.mul_assoc, Matrix.mul_assoc]
    _ = Fᴴ * F := by rw [hΛ, Matrix.mul_one]
    _ = 1 := hFtF

/-- **Noisy-solve deviation (CORR-1, n = 1) — the noise model.** A noisy quantum
    solver estimates the eigenvalue reciprocals `1/λ` imperfectly (noisy QPE /
    noisy QFT phases): it applies `Λ̃⁻¹ = diag(m₀, m₁)` instead of the exact
    `diag(1/(a+b), 1/(a−b))`. This proves the resulting solve operator's
    **deviation from the exact `C⁻¹` is exactly the QFT-conjugated eigenvalue
    error** `Fᴴ · diag(m₀ − 1/(a+b), m₁ − 1/(a−b)) · F`. Hence the error is linear
    in the per-eigenvalue estimate error and vanishes as the noise → 0 (when
    `mₖ → 1/λₖ`, recovering `circ2_solve_operator`). No norm machinery — an exact
    operator identity. Sorry-free. -/
theorem circ2_solve_noise_deviation (a b m0 m1 : ℂ) :
    (denote (qft_circuit 1))ᴴ * Matrix.diagonal ![m0, m1] * denote (qft_circuit 1)
        - (denote (qft_circuit 1))ᴴ * Matrix.diagonal ![(a + b)⁻¹, (a - b)⁻¹]
            * denote (qft_circuit 1)
      = (denote (qft_circuit 1))ᴴ
          * Matrix.diagonal ![m0 - (a + b)⁻¹, m1 - (a - b)⁻¹] * denote (qft_circuit 1) := by
  -- The proof is one line of linearity: conjugation `X ↦ Fᴴ X F` is additive, and
  -- `diag(m) - diag(1/λ) = diag(m - 1/λ)`. So the deviation of the two solve
  -- operators is the conjugation of the diagonal eigenvalue-reciprocal error.
  -- First, split the error diagonal as a difference of the two diagonals:
  have hdiag : Matrix.diagonal ![m0 - (a + b)⁻¹, m1 - (a - b)⁻¹]
      = Matrix.diagonal ![m0, m1] - Matrix.diagonal ![(a + b)⁻¹, (a - b)⁻¹] := by
    ext i j
    fin_cases i <;> fin_cases j <;> simp [Matrix.diagonal, Matrix.sub_apply]
  -- Then distribute `Fᴴ · (X - Y) · F` over the subtraction (`mul_sub`, `sub_mul`),
  -- which is definitionally the LHS difference of the two solve operators.
  rw [hdiag, mul_sub, sub_mul]

end QuantumProofs.CirculantSolve
