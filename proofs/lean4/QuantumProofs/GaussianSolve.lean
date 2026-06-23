/-
  Dense Gaussian solver correctness (LE1-b, `solve_dense_gaussian`).

  Faithful model of
  `crates/omega-core/src/circulant.rs::solve_dense_gaussian` — the
  independent partial-pivot Gaussian oracle the DFT-diagonalization solve is
  validated against. The Rust routine has two phases on the augmented system
  `(A | b)`:

  1. **Forward elimination** with partial pivoting (rows swapped to the
     largest-magnitude pivot, then `row -= f·pivotRow`), turning `A` into an
     upper-triangular `U` and `b` into the eliminated RHS `c`.
  2. **Back-substitution** (`circulant.rs` lines 68-75):
     ```text
     for row in (0..n).rev():
         s = c[row] - Σ_{j>row} U[row][j]·x[j]
         x[row] = s / U[row][row]
     ```

  This file proves **layer 1 of the correctness chain: back-substitution
  exactly solves an upper-triangular system.** Given an upper-triangular `U`
  with nonzero diagonal and a RHS `b`, the vector produced by the back-sub
  recurrence satisfies `U *ᵥ x = b` — an *exact* identity over a field (the
  algorithm divides by each diagonal pivot, so "diagonal nonzero" is exactly
  the routine's runtime precondition). Worked over a general `Field K`; the
  Rust kernel runs on `Complex64 ≈ ℂ`, an instance.

  A later layer will close the forward pass (each pivot/elimination step is
  left-multiplication by an invertible elementary matrix, so the produced
  `(U, c)` satisfy `U = M·A`, `c = M·b` for invertible `M`, whence
  `U *ᵥ x = c` gives `A *ᵥ x = b`).

  ## Why `x i` needs no induction over the other entries

  The back-sub value `x i = (b i − Σ_{j>i} U i j · x j) / U i i` references only
  entries `x j` with `j > i`. The *pointwise* equation `(U *ᵥ x) i = b i` then
  follows from a single unfolding of `x i`: the row sum `Σ_j U i j · x j` splits
  as `Σ_{j<i}(=0, upper-triangular) + U i i · x i + Σ_{j>i}`, and
  `U i i · x i = b i − Σ_{j>i}` cancels the tail — the values `x j` (`j > i`)
  appear on both sides and never need to be evaluated.
-/
import Mathlib.LinearAlgebra.Matrix.Circulant
import Mathlib.LinearAlgebra.Matrix.Permutation
import Mathlib.Data.Matrix.PEquiv
import Mathlib.Tactic

namespace QuantumProofs.GaussianSolve

open Matrix BigOperators Finset

variable {K : Type*} [Field K] {n : ℕ}

/-- Back-substitution for an upper-triangular system, the exact-arithmetic model
    of `solve_dense_gaussian`'s back-sub loop (`circulant.rs` lines 68-75):
    `x i = (b i − Σ_{j>i} U i j · x j) / U i i`. Defined by well-founded
    recursion downward (`n − i` decreases, since every recursive call is on some
    `j > i`); `.attach` carries the `i < j` membership proof into the termination
    check. -/
def backSub (U : Matrix (Fin n) (Fin n) K) (b : Fin n → K) (i : Fin n) : K :=
  (b i - ∑ j ∈ (univ.filter (i < ·)).attach, U i j.1 * backSub U b j.1) / U i i
termination_by n - i.1
decreasing_by
  have hij : i < j.1 := (mem_filter.mp j.2).2
  omega

/-- **Back-substitution correctness (LE1-b, layer 1).** For an upper-triangular
    `U` (`U i j = 0` whenever `j < i`) with nonzero diagonal, the back-sub vector
    solves the system exactly: `U *ᵥ backSub U b = b`. Sorry-free. -/
theorem backSub_solves (U : Matrix (Fin n) (Fin n) K) (b : Fin n → K)
    (hU : ∀ i j, j < i → U i j = 0) (hd : ∀ i, U i i ≠ 0) :
    U *ᵥ backSub U b = b := by
  funext i
  set x := backSub U b with hx
  -- One unfolding of `x i`, with the `.attach` sum collapsed back to a plain sum.
  have hunfold : x i = (b i - ∑ j ∈ univ.filter (i < ·), U i j * x j) / U i i := by
    rw [hx, backSub, Finset.sum_attach (univ.filter (i < ·)) (fun j => U i j * backSub U b j)]
  -- Clear the division: `U i i · x i = b i − Σ_{j>i} U i j · x j`.
  have hmul : U i i * x i = b i - ∑ j ∈ univ.filter (i < ·), U i j * x j := by
    rw [hunfold, mul_comm, div_mul_cancel₀ _ (hd i)]
  -- The `j ≤ i` part of the row sum is just the diagonal term (sub-diagonal = 0).
  have hrest : ∑ j ∈ univ.filter (fun j => ¬ i < j), U i j * x j = U i i * x i := by
    refine Finset.sum_eq_single_of_mem i ?_ ?_
    · simp
    · intro j hj hji
      have hjle : j ≤ i := not_lt.mp (mem_filter.mp hj).2
      rw [hU i j (lt_of_le_of_ne hjle hji), zero_mul]
  -- Split the full row sum at `i` and cancel the tail.
  have key : ∑ j, U i j * x j = U i i * x i + ∑ j ∈ univ.filter (i < ·), U i j * x j := by
    rw [← Finset.sum_filter_add_sum_filter_not univ (i < ·) (fun j => U i j * x j), hrest,
        add_comm]
  show (U *ᵥ x) i = b i
  rw [mulVec, dotProduct]
  rw [key, hmul]
  ring

/-- **Forward-pass reduction interface (LE1-b, layer 2).** This is the exact
    contract the still-open forward-elimination layer will fulfil: it
    left-multiplies the augmented system `(A | b)` by an *invertible* matrix `M`
    (assembled from the pivot row-swaps and the `row -= f·pivotRow` elementary
    operations) producing an upper-triangular `U = M·A` with nonzero diagonal and
    eliminated RHS `c = M·b`. Given that, back-substitution on `(U, c)` solves the
    **original** system `A *ᵥ x = b` — because `M` is invertible and cancels:
    `(M·A) *ᵥ x = M *ᵥ b ⟹ A *ᵥ x = b`. Sorry-free. -/
theorem backSub_solves_of_reduction (A U : Matrix (Fin n) (Fin n) K) (b c : Fin n → K)
    (M : Matrix (Fin n) (Fin n) K) [Invertible M]
    (hU : U = M * A) (hc : c = M *ᵥ b)
    (hupper : ∀ i j, j < i → U i j = 0) (hdiag : ∀ i, U i i ≠ 0) :
    A *ᵥ backSub U c = b := by
  subst hU hc
  -- Back-sub solves the reduced (upper-triangular) system exactly.
  have hsolve : (M * A) *ᵥ backSub (M * A) (M *ᵥ b) = M *ᵥ b :=
    backSub_solves (M * A) (M *ᵥ b) hupper hdiag
  -- Apply `⅟M` on the left and collapse `⅟M·M = 1`, cancelling the reduction.
  have h2 := congrArg (fun v => ⅟M *ᵥ v) hsolve
  simpa only [Matrix.mulVec_mulVec, ← Matrix.mul_assoc, invOf_mul_self,
    Matrix.one_mul, Matrix.one_mulVec] using h2

/-! ### Forward elimination, layer 3a: the atomic Gauss transform

The forward pass clears one pivot column at a time. The Rust inner double-loop
(`circulant.rs` lines 58-66) subtracts `f·(pivot row)` from each row below, with
`f = a[row][col] / a[col][col]`. Algebraically that is left-multiplication by the
**atomic Gauss transform** `Lₖ = 1 − m ⊗ eₖ`, where `mᵢ = A i k / A k k` for
`i > k` (else `0`) and `eₖ` is the `k`-th basis covector. Its left action is
`(Lₖ · A) i j = A i j − mᵢ · A k j` — exactly the Rust update — and it is
lower-triangular with unit diagonal, hence invertible (`N := −m⊗eₖ` is nilpotent,
`N² = 0`, so `(1 + N)⁻¹ = 1 − N`). -/

/-- Elimination multipliers for pivot column `k`: `mᵢ = A i k / A k k` for rows
    `i > k`, else `0` (the Rust `f = a[row][col]/a[col][col]`, restricted to rows
    below the pivot). -/
def gaussMult (A : Matrix (Fin n) (Fin n) K) (k : Fin n) : Fin n → K :=
  fun i => if k < i then A i k / A k k else 0

@[simp] lemma gaussMult_self (A : Matrix (Fin n) (Fin n) K) (k : Fin n) :
    gaussMult A k k = 0 := by simp [gaussMult]

/-- The nilpotent part `N = −m ⊗ eₖ` of the Gauss transform. -/
def gaussN (A : Matrix (Fin n) (Fin n) K) (k : Fin n) : Matrix (Fin n) (Fin n) K :=
  Matrix.of (fun i j => - gaussMult A k i * (if j = k then 1 else 0))

/-- The atomic Gauss transform `Lₖ = 1 + N = 1 − m ⊗ eₖ`. -/
def gaussTransform (A : Matrix (Fin n) (Fin n) K) (k : Fin n) : Matrix (Fin n) (Fin n) K :=
  1 + gaussN A k

/-- Left action of the nilpotent part: `(N · B) i j = −mᵢ · B k j` (the `eₖ` row
    selector collapses the sum to row `k`). -/
lemma gaussN_mul_apply (A B : Matrix (Fin n) (Fin n) K) (k i j : Fin n) :
    (gaussN A k * B) i j = - gaussMult A k i * B k j := by
  simp only [Matrix.mul_apply, gaussN, Matrix.of_apply, mul_assoc, ← Finset.mul_sum]
  congr 1
  simp [ite_mul, Finset.sum_ite_eq']

/-- **The Gauss-transform action** `(Lₖ · B) i j = B i j − mᵢ · B k j` — exactly
    the Rust inner-loop update `a[row][c] -= a[col][c]·f`. -/
lemma gaussTransform_mul_apply (A B : Matrix (Fin n) (Fin n) K) (k i j : Fin n) :
    (gaussTransform A k * B) i j = B i j - gaussMult A k i * B k j := by
  simp only [gaussTransform, Matrix.add_mul, Matrix.one_mul, Matrix.add_apply, gaussN_mul_apply]
  ring

/-- Rows at or above the pivot (`¬ k < i`) are left unchanged — the multiplier is `0`. -/
lemma gaussTransform_preserves (A B : Matrix (Fin n) (Fin n) K) (k i : Fin n)
    (hik : ¬ k < i) (j : Fin n) : (gaussTransform A k * B) i j = B i j := by
  rw [gaussTransform_mul_apply]; simp [gaussMult, hik]

/-- The pivot column is cleared below the pivot: `(Lₖ · A) i k = 0` for `i > k`
    (needs the pivot `A k k ≠ 0`, the division precondition). -/
lemma gaussTransform_clears (A : Matrix (Fin n) (Fin n) K) (k i : Fin n)
    (hki : k < i) (hkk : A k k ≠ 0) : (gaussTransform A k * A) i k = 0 := by
  rw [gaussTransform_mul_apply]
  unfold gaussMult
  rw [if_pos hki, div_mul_cancel₀ _ hkk, sub_self]

/-- `N` is nilpotent: `N² = 0` (since the `eₖ` covector hits the zero multiplier
    `m k = 0`). -/
lemma gaussN_sq (A : Matrix (Fin n) (Fin n) K) (k : Fin n) :
    gaussN A k * gaussN A k = 0 := by
  ext i j
  rw [gaussN_mul_apply]
  simp [gaussN, gaussMult_self]

/-- The Gauss transform is invertible with inverse `1 − N` (unit lower-triangular). -/
instance gaussTransform_invertible (A : Matrix (Fin n) (Fin n) K) (k : Fin n) :
    Invertible (gaussTransform A k) where
  invOf := 1 - gaussN A k
  invOf_mul_self := by
    simp only [gaussTransform, sub_mul, mul_add, Matrix.one_mul, Matrix.mul_one, gaussN_sq]
    abel
  mul_invOf_self := by
    simp only [gaussTransform, mul_sub, add_mul, Matrix.one_mul, Matrix.mul_one, gaussN_sq]
    abel

/-! ### Forward elimination, layer 3b: the column recursion and end-to-end solve

Composing the atomic Gauss transforms column by column produces the accumulated
(invertible) transform `Mₖ` and the partially-reduced matrix `Uₖ = Mₖ · A`. After
`n` columns `Uₙ` is upper-triangular with nonzero diagonal, so back-substitution
on `(Uₙ, Mₙ·b)` solves the original `A *ᵥ x = b`.

**Scope.** This models the *swap-free* elimination. The Rust adds **partial
pivoting** (a max-magnitude row swap before each column): that is a left
multiplication by a permutation matrix — also invertible — chosen purely for
numerical stability and to dodge a zero pivot. Over exact arithmetic it does not
change the computed solution, so the correctness statement here is conditioned on
the *natural* pivots `Uₖ k k` being nonzero (exactly the routine's runtime success
condition: it divides by each pivot). Folding the permutations into `Mₖ` is a
mechanical follow-on. -/

/-- Accumulated forward-elimination transform after `k` columns: `M₀ = 1`,
    `Mₖ₊₁ = Lₖ · Mₖ` where `Lₖ = gaussTransform (Mₖ·A) k` clears column `k`.
    The reduced matrix is `Uₖ = elimM A k * A`. -/
def elimM (A : Matrix (Fin n) (Fin n) K) : ℕ → Matrix (Fin n) (Fin n) K
  | 0 => 1
  | (k + 1) =>
    if h : k < n then gaussTransform (elimM A k * A) ⟨k, h⟩ * elimM A k else elimM A k

/-- One-column unfolding of `elimM` when the column index is in range. -/
lemma elimM_succ (A : Matrix (Fin n) (Fin n) K) {k : ℕ} (hk : k < n) :
    elimM A (k + 1) = gaussTransform (elimM A k * A) ⟨k, hk⟩ * elimM A k := by
  rw [elimM, dif_pos hk]

/-- `elimM A k` is a unit (a product of invertible Gauss transforms). -/
lemma elimM_isUnit (A : Matrix (Fin n) (Fin n) K) (k : ℕ) : IsUnit (elimM A k) := by
  induction k with
  | zero => rw [elimM]; exact isUnit_one
  | succ k ih =>
    rw [elimM]
    split
    · exact (isUnit_of_invertible _).mul ih
    · exact ih

/-- **Forward-elimination invariant (LE1-b, layer 3b).** Provided the natural
    pivots `Uₖ k k` are all nonzero, after `k` columns `Uₖ = elimM A k * A` is
    upper-triangular in its first `k` columns (`Uₖ i j = 0` for `j < k`, `j < i`)
    and its first `k` diagonal entries are nonzero. By induction on `k`, using only
    the Gauss-transform clear/preserve lemmas. Sorry-free. -/
lemma elimM_invariant (A : Matrix (Fin n) (Fin n) K)
    (hpiv : ∀ (k : ℕ) (h : k < n), (elimM A k * A) ⟨k, h⟩ ⟨k, h⟩ ≠ 0) :
    ∀ k : ℕ, k ≤ n →
      (∀ i j : Fin n, (j : ℕ) < k → j < i → (elimM A k * A) i j = 0) ∧
      (∀ j : Fin n, (j : ℕ) < k → (elimM A k * A) j j ≠ 0) := by
  intro k
  induction k with
  | zero =>
    intro _
    exact ⟨fun _ _ hj _ => absurd hj (Nat.not_lt_zero _),
           fun _ hj => absurd hj (Nat.not_lt_zero _)⟩
  | succ k ih =>
    intro hk1
    have hk : k < n := Nat.lt_of_succ_le hk1
    obtain ⟨ihU, ihD⟩ := ih (le_of_lt hk)
    have hp : (elimM A k * A) ⟨k, hk⟩ ⟨k, hk⟩ ≠ 0 := hpiv k hk
    have hUk1 : elimM A (k + 1) * A
        = gaussTransform (elimM A k * A) ⟨k, hk⟩ * (elimM A k * A) := by
      rw [elimM_succ A hk, Matrix.mul_assoc]
    refine ⟨?_, ?_⟩
    · -- upper-triangular in first k+1 columns
      intro i j hj hji
      rw [hUk1]
      rcases (Nat.lt_succ_iff.mp hj).lt_or_eq with hjk | hjk
      · -- column j < k: entry unchanged (pivot row is zero there), zero by IH
        rw [gaussTransform_mul_apply]
        rw [ihU ⟨k, hk⟩ j hjk hjk, ihU i j hjk hji]
        ring
      · -- column j = k: cleared below the pivot
        have hjeq : j = ⟨k, hk⟩ := Fin.ext hjk
        subst hjeq
        exact gaussTransform_clears (elimM A k * A) ⟨k, hk⟩ i hji hp
    · -- nonzero diagonal in first k+1 columns
      intro j hj
      rw [hUk1]
      rcases (Nat.lt_succ_iff.mp hj).lt_or_eq with hjk | hjk
      · -- j < k: diagonal entry preserved, nonzero by IH
        have hik : ¬ (⟨k, hk⟩ : Fin n) < j := by simp only [Fin.lt_def]; omega
        rw [gaussTransform_preserves _ _ _ _ hik]
        exact ihD j hjk
      · -- j = k: the pivot itself, preserved
        have hjeq : j = ⟨k, hk⟩ := Fin.ext hjk
        subst hjeq
        rw [gaussTransform_preserves _ _ _ _ (lt_irrefl _)]
        exact hp

/-- The dense Gaussian solver: forward-eliminate to upper-triangular, then
    back-substitute. `gaussianSolve A b = backSub (Uₙ) (Mₙ·b)` with
    `Uₙ = elimM A n * A`, `Mₙ = elimM A n` — the exact-arithmetic model of
    `solve_dense_gaussian` (swap-free; see the layer-3b note on pivoting). -/
noncomputable def gaussianSolve (A : Matrix (Fin n) (Fin n) K) (b : Fin n → K) : Fin n → K :=
  backSub (elimM A n * A) (elimM A n *ᵥ b)

/-- **Dense Gaussian solver correctness (LE1-b).** If every natural pivot is
    nonzero (the routine's runtime success condition), the solver's output solves
    the original system exactly: `A *ᵥ gaussianSolve A b = b`. Sorry-free.
    Assembles `elimM_invariant` (upper-triangular `Uₙ`, nonzero diagonal) +
    `elimM_isUnit` (invertible `Mₙ`) into `backSub_solves_of_reduction`. -/
theorem gaussianSolve_solves (A : Matrix (Fin n) (Fin n) K) (b : Fin n → K)
    (hpiv : ∀ (k : ℕ) (h : k < n), (elimM A k * A) ⟨k, h⟩ ⟨k, h⟩ ≠ 0) :
    A *ᵥ gaussianSolve A b = b := by
  obtain ⟨hU, hD⟩ := elimM_invariant A hpiv n (le_refl n)
  letI : Invertible (elimM A n) := (elimM_isUnit A n).invertible
  exact backSub_solves_of_reduction A (elimM A n * A) b (elimM A n *ᵥ b) (elimM A n)
    rfl rfl (fun i j hji => hU i j j.isLt hji) (fun i => hD i i.isLt)

/-! ### Forward elimination, layer 3c: partial pivoting

The Rust routine swaps the largest-magnitude row into the pivot position before
each column (`circulant.rs` 47-56). Each swap is left-multiplication by a
permutation matrix `Pₖ` — also invertible — so the reduction becomes
`Uₖ₊₁ = Lₖ · Pₖ · Uₖ`. We parameterize by the per-column transpositions `σ k`
(supplied by the max-magnitude oracle, abstracted), required only to **fix
indices `< k`** and **keep indices `≥ k` in range `≥ k`** — which any swap of two
rows `≥ k` satisfies. Correctness then holds under the *pivoted* nonzero-pivot
condition `(Pₖ · Uₖ) k k ≠ 0` (the max-magnitude entry, nonzero whenever the
column has any nonzero entry below the pivot — the partial-pivoting guarantee). -/

/-- Entry of a permutation matrix: a single `1` in row `i` at column `σ i`. -/
lemma permMatrix_entry (σ : Equiv.Perm (Fin n)) (i j : Fin n) :
    (σ.permMatrix K) i j = if σ i = j then 1 else 0 := by
  rw [Equiv.Perm.permMatrix, PEquiv.toMatrix_apply, Equiv.toPEquiv_apply]
  simp [Option.mem_def]

/-- Left action of a permutation matrix: `(Pσ · M) i j = M (σ i) j` (row `i` of the
    product reads row `σ i` of `M` — i.e. the rows are permuted by `σ`). -/
lemma permMatrix_mul_apply (σ : Equiv.Perm (Fin n)) (M : Matrix (Fin n) (Fin n) K)
    (i j : Fin n) : (σ.permMatrix K * M) i j = M (σ i) j := by
  simp only [Matrix.mul_apply, permMatrix_entry, ite_mul, one_mul, zero_mul]
  rw [Finset.sum_ite_eq Finset.univ (σ i) (fun l => M l j)]
  simp

/-- `σ⁻¹.permMatrix` is a right inverse, so a permutation matrix is a unit. -/
lemma permMatrix_mul_inv (σ : Equiv.Perm (Fin n)) :
    σ.permMatrix K * σ⁻¹.permMatrix K = 1 := by
  ext i j
  rw [permMatrix_mul_apply, permMatrix_entry, Matrix.one_apply]
  have hinv : σ⁻¹ (σ i) = i := by simp
  rw [hinv]

lemma permMatrix_isUnit (σ : Equiv.Perm (Fin n)) : IsUnit (σ.permMatrix K) :=
  IsUnit.of_mul_eq_one (σ⁻¹.permMatrix K) (permMatrix_mul_inv σ)

/-- Pivoted forward-elimination transform: `M₀ = 1`,
    `Mₖ₊₁ = Lₖ · Pₖ · Mₖ` where `Pₖ = (σ k).permMatrix` swaps in the pivot row and
    `Lₖ = gaussTransform (Pₖ·Mₖ·A) k` clears column `k`. `Uₖ = elimMP A σ k * A`. -/
def elimMP (A : Matrix (Fin n) (Fin n) K) (σ : ℕ → Equiv.Perm (Fin n)) :
    ℕ → Matrix (Fin n) (Fin n) K
  | 0 => 1
  | (k + 1) =>
    if h : k < n then
      gaussTransform ((σ k).permMatrix K * elimMP A σ k * A) ⟨k, h⟩
        * (σ k).permMatrix K * elimMP A σ k
    else elimMP A σ k

lemma elimMP_succ (A : Matrix (Fin n) (Fin n) K) (σ : ℕ → Equiv.Perm (Fin n)) {k : ℕ}
    (hk : k < n) :
    elimMP A σ (k + 1)
      = gaussTransform ((σ k).permMatrix K * elimMP A σ k * A) ⟨k, hk⟩
          * (σ k).permMatrix K * elimMP A σ k := by
  rw [elimMP, dif_pos hk]

lemma elimMP_isUnit (A : Matrix (Fin n) (Fin n) K) (σ : ℕ → Equiv.Perm (Fin n)) (k : ℕ) :
    IsUnit (elimMP A σ k) := by
  induction k with
  | zero => rw [elimMP]; exact isUnit_one
  | succ k ih =>
    rw [elimMP]
    split
    · exact ((isUnit_of_invertible _).mul (permMatrix_isUnit _)).mul ih
    · exact ih

/-- **Pivoted forward-elimination invariant (LE1-b, layer 3c).** With per-column
    transpositions `σ` that fix indices `< k` (`hfix`) and keep indices `≥ k` in
    range (`hge`), and the pivoted pivots nonzero (`hpiv`), after `k` columns
    `Uₖ = elimMP A σ k * A` is upper-triangular in its first `k` columns with
    nonzero first-`k` diagonal. The permutation step preserves earlier zeros
    because a row index `< k` is fixed while one `≥ k` stays `≥ k > j`. Sorry-free. -/
lemma elimMP_invariant (A : Matrix (Fin n) (Fin n) K) (σ : ℕ → Equiv.Perm (Fin n))
    (hfix : ∀ (k : ℕ) (i : Fin n), (i : ℕ) < k → σ k i = i)
    (hge : ∀ (k : ℕ) (i : Fin n), k ≤ (i : ℕ) → k ≤ (σ k i : ℕ))
    (hpiv : ∀ (k : ℕ) (h : k < n),
        ((σ k).permMatrix K * (elimMP A σ k * A)) ⟨k, h⟩ ⟨k, h⟩ ≠ 0) :
    ∀ k : ℕ, k ≤ n →
      (∀ i j : Fin n, (j : ℕ) < k → j < i → (elimMP A σ k * A) i j = 0) ∧
      (∀ j : Fin n, (j : ℕ) < k → (elimMP A σ k * A) j j ≠ 0) := by
  intro k
  induction k with
  | zero =>
    intro _
    exact ⟨fun _ _ hj _ => absurd hj (Nat.not_lt_zero _),
           fun _ hj => absurd hj (Nat.not_lt_zero _)⟩
  | succ k ih =>
    intro hk1
    have hk : k < n := Nat.lt_of_succ_le hk1
    obtain ⟨ihU, ihD⟩ := ih (le_of_lt hk)
    have key : elimMP A σ (k + 1) * A
        = gaussTransform ((σ k).permMatrix K * (elimMP A σ k * A)) ⟨k, hk⟩
            * ((σ k).permMatrix K * (elimMP A σ k * A)) := by
      rw [elimMP_succ A σ hk]; simp only [Matrix.mul_assoc]
    have hVent : ∀ i j : Fin n, ((σ k).permMatrix K * (elimMP A σ k * A)) i j
        = (elimMP A σ k * A) (σ k i) j :=
      fun i j => permMatrix_mul_apply (σ k) (elimMP A σ k * A) i j
    have hp : ((σ k).permMatrix K * (elimMP A σ k * A)) ⟨k, hk⟩ ⟨k, hk⟩ ≠ 0 := hpiv k hk
    refine ⟨?_, ?_⟩
    · intro i j hj hji
      rw [key]
      rcases (Nat.lt_succ_iff.mp hj).lt_or_eq with hjk | hjk
      · rw [gaussTransform_mul_apply]
        have hpivrow : ((σ k).permMatrix K * (elimMP A σ k * A)) ⟨k, hk⟩ j = 0 := by
          rw [hVent]
          refine ihU _ j hjk ?_
          have hge' : k ≤ (σ k ⟨k, hk⟩ : ℕ) := hge k ⟨k, hk⟩ (le_refl k)
          simp only [Fin.lt_def]; omega
        have hrow : ((σ k).permMatrix K * (elimMP A σ k * A)) i j = 0 := by
          rw [hVent]
          refine ihU _ j hjk ?_
          rcases Nat.lt_or_ge (i : ℕ) k with hik | hik
          · rw [hfix k i hik]; exact hji
          · have hge' : k ≤ (σ k i : ℕ) := hge k i hik
            simp only [Fin.lt_def]; omega
        rw [hpivrow, hrow]; ring
      · have hjeq : j = ⟨k, hk⟩ := Fin.ext hjk
        subst hjeq
        exact gaussTransform_clears _ ⟨k, hk⟩ i hji hp
    · intro j hj
      rw [key]
      rcases (Nat.lt_succ_iff.mp hj).lt_or_eq with hjk | hjk
      · have hik : ¬ (⟨k, hk⟩ : Fin n) < j := by simp only [Fin.lt_def]; omega
        rw [gaussTransform_preserves _ _ _ _ hik, hVent, hfix k j hjk]
        exact ihD j hjk
      · have hjeq : j = ⟨k, hk⟩ := Fin.ext hjk
        subst hjeq
        rw [gaussTransform_preserves _ _ _ _ (lt_irrefl _)]
        exact hp

/-- The dense Gaussian solver **with partial pivoting**: forward-eliminate with
    the row swaps `σ`, then back-substitute. The exact-arithmetic model of
    `solve_dense_gaussian` including its pivoting. -/
noncomputable def gaussianSolvePivot (A : Matrix (Fin n) (Fin n) K)
    (σ : ℕ → Equiv.Perm (Fin n)) (b : Fin n → K) : Fin n → K :=
  backSub (elimMP A σ n * A) (elimMP A σ n *ᵥ b)

/-- **Dense Gaussian solver with partial pivoting — correctness (LE1-b).** Under
    the structural conditions on the pivot swaps (`hfix`/`hge`) and the pivoted
    nonzero-pivot guarantee (`hpiv`), the solver solves the original system
    exactly: `A *ᵥ gaussianSolvePivot A σ b = b`. Sorry-free. -/
theorem gaussianSolvePivot_solves (A : Matrix (Fin n) (Fin n) K)
    (σ : ℕ → Equiv.Perm (Fin n)) (b : Fin n → K)
    (hfix : ∀ (k : ℕ) (i : Fin n), (i : ℕ) < k → σ k i = i)
    (hge : ∀ (k : ℕ) (i : Fin n), k ≤ (i : ℕ) → k ≤ (σ k i : ℕ))
    (hpiv : ∀ (k : ℕ) (h : k < n),
        ((σ k).permMatrix K * (elimMP A σ k * A)) ⟨k, h⟩ ⟨k, h⟩ ≠ 0) :
    A *ᵥ gaussianSolvePivot A σ b = b := by
  obtain ⟨hU, hD⟩ := elimMP_invariant A σ hfix hge hpiv n (le_refl n)
  letI : Invertible (elimMP A σ n) := (elimMP_isUnit A σ n).invertible
  exact backSub_solves_of_reduction A (elimMP A σ n * A) b (elimMP A σ n *ᵥ b) (elimMP A σ n)
    rfl rfl (fun i j hji => hU i j j.isLt hji) (fun i => hD i i.isLt)

/-! ### Forward elimination, layer 3d: the partial-pivoting guarantee (nonsingular `A`)

`gaussianSolvePivot_solves` is conditioned on the pivoted pivots being nonzero
(`hpiv`). For an **invertible** `A` that condition is always satisfiable: at every
column a nonzero pivot candidate exists among the rows `≥ k`, so the
max-magnitude swap lands a nonzero pivot. This is the formal content of "partial
pivoting never hits a zero pivot on a nonsingular matrix." The mathematical core
is `pivot_exists` (a rank/linear-independence argument); `elimMP_mul_isUnit_det`
keeps the eliminated matrix invertible so the core applies at each step. -/

/-- **Pivot existence (the linear-algebra core).** An invertible matrix `U` that is
    upper-triangular in its first `k` columns (`U i j = 0` for `j < k`, `j < i`) has
    a nonzero entry in column `k` at some row `p ≥ k`. Proof: otherwise columns
    `0..k` would all vanish on rows `≥ k`, placing `k+1` columns in the
    `k`-dimensional span of the first `k` coordinates — linearly dependent, giving a
    nonzero kernel vector, so `det U = 0`, contradicting invertibility. Sorry-free. -/
theorem pivot_exists (U : Matrix (Fin n) (Fin n) K) (hU : IsUnit U.det) (k : Fin n)
    (hupper : ∀ i j : Fin n, (j : ℕ) < (k : ℕ) → j < i → U i j = 0) :
    ∃ p : Fin n, (k : ℕ) ≤ (p : ℕ) ∧ U p k ≠ 0 := by
  by_contra hcon
  have hcon' : ∀ p : Fin n, (k : ℕ) ≤ (p : ℕ) → U p k = 0 := by
    intro p hp; by_contra hne; exact hcon ⟨p, hp, hne⟩
  have hcol : ∀ i j : Fin n, (k : ℕ) ≤ (i : ℕ) → (j : ℕ) ≤ (k : ℕ) → U i j = 0 := by
    intro i j hi hj
    rcases lt_or_eq_of_le hj with hjk | hjk
    · exact hupper i j hjk (by simp only [Fin.lt_def]; omega)
    · have hje : j = k := Fin.ext hjk
      subst hje; exact hcon' i hi
  set K0 := (k : ℕ) with hK0
  have hkn : K0 < n := k.isLt
  let w : Fin (K0 + 1) → (Fin K0 → K) := fun j' i' => U ⟨i'.1, by omega⟩ ⟨j'.1, by omega⟩
  have hdep : ¬ LinearIndependent K w := by
    intro hli
    have hcard := hli.fintype_card_le_finrank
    simp only [Fintype.card_fin] at hcard
    have hfr : Module.finrank K (Fin K0 → K) = K0 := by simp
    rw [hfr] at hcard; omega
  rw [Fintype.not_linearIndependent_iff] at hdep
  obtain ⟨g, hsum, j0, hj0⟩ := hdep
  let v : Fin n → K := fun i => if h : (i : ℕ) < K0 + 1 then g ⟨i, h⟩ else 0
  have hvc : ∀ j' : Fin (K0 + 1), v ⟨j'.1, by omega⟩ = g j' := by
    intro j'
    simp only [v]
    rw [dif_pos (show ((⟨j'.1, by omega⟩ : Fin n) : ℕ) < K0 + 1 from j'.2)]
  have hv0 : v ≠ 0 := by
    intro hv; apply hj0
    have h1 := hvc j0; rw [hv] at h1; simpa using h1.symm
  have hUv : U *ᵥ v = 0 := by
    funext i
    simp only [mulVec, dotProduct, Pi.zero_apply]
    rcases lt_or_ge (i : ℕ) K0 with hik | hik
    · have hcomp := congrFun hsum ⟨(i : ℕ), hik⟩
      simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply, w] at hcomp
      rw [show (∑ j : Fin n, U i j * v j)
            = ∑ j' : Fin (K0 + 1), g j' * U ⟨(i : ℕ), by omega⟩ ⟨j'.1, by omega⟩ from ?_]
      · exact hcomp
      · rw [(Finset.sum_filter_of_ne (p := fun j : Fin n => (j : ℕ) < K0 + 1)
              (f := fun j => U i j * v j) ?_).symm]
        · refine Finset.sum_bij'
            (i := fun (j : Fin n) (_ : j ∈ _) => (⟨j.1, by aesop⟩ : Fin (K0 + 1)))
            (j := fun (j' : Fin (K0 + 1)) (_ : j' ∈ Finset.univ) => (⟨j'.1, by omega⟩ : Fin n))
            ?_ ?_ ?_ ?_ ?_
          · intro a _; exact Finset.mem_univ _
          · intro b _; simp only [Finset.mem_filter, Finset.mem_univ, true_and]; exact b.2
          · intro a _; apply Fin.ext; rfl
          · intro b _; apply Fin.ext; rfl
          · intro a ha
            have hjf : (a : ℕ) < K0 + 1 := (Finset.mem_filter.mp ha).2
            rw [mul_comm]
            congr 1
            simp only [v]; rw [dif_pos hjf]
        · intro j _ hne
          by_contra h
          exact hne (by simp only [v, dif_neg h, mul_zero])
    · apply Finset.sum_eq_zero
      intro j _
      by_cases h : (j : ℕ) < K0 + 1
      · rw [hcol i j (by omega) (by omega), zero_mul]
      · simp only [v, dif_neg h, mul_zero]
  exact hU.ne_zero (exists_mulVec_eq_zero_iff.mp ⟨v, hv0, hUv⟩)

/-- The partially-eliminated matrix `Uₖ = elimMP A σ k * A` stays invertible when `A`
    is (a product of units), so `pivot_exists` applies at every column. -/
lemma elimMP_mul_isUnit_det (A : Matrix (Fin n) (Fin n) K) (σ : ℕ → Equiv.Perm (Fin n))
    (k : ℕ) (hA : IsUnit A.det) : IsUnit ((elimMP A σ k * A).det) := by
  rw [Matrix.det_mul]
  exact (((Matrix.isUnit_iff_isUnit_det _).mp (elimMP_isUnit A σ k))).mul hA

/-- `elimMP A σ k` depends only on `σ` at indices `< k` (the recursion never reads
    `σ` at index `≥ k`). -/
lemma elimMP_congr (A : Matrix (Fin n) (Fin n) K) (σ σ' : ℕ → Equiv.Perm (Fin n)) :
    ∀ k : ℕ, (∀ j, j < k → σ j = σ' j) → elimMP A σ k = elimMP A σ' k := by
  intro k
  induction k with
  | zero => intro _; rfl
  | succ k ih =>
    intro hagree
    have hih : elimMP A σ k = elimMP A σ' k :=
      ih (fun j hj => hagree j (Nat.lt_succ_of_lt hj))
    by_cases hk : k < n
    · rw [elimMP_succ A σ hk, elimMP_succ A σ' hk, hih, hagree k (Nat.lt_succ_self k)]
    · have e1 : elimMP A σ (k + 1) = elimMP A σ k := by rw [elimMP]; exact dif_neg hk
      have e2 : elimMP A σ' (k + 1) = elimMP A σ' k := by rw [elimMP]; exact dif_neg hk
      rw [e1, e2, hih]

/-- Bounded `elimMP_invariant`: the invariant at column `K` needs the pivoted-pivot
    condition only for columns `< K` (used by the σ-construction, where the pivots are
    chosen one column at a time). Sorry-free. -/
lemma elimMP_invariant_lt (A : Matrix (Fin n) (Fin n) K) (σ : ℕ → Equiv.Perm (Fin n))
    (hfix : ∀ (k : ℕ) (i : Fin n), (i : ℕ) < k → σ k i = i)
    (hge : ∀ (k : ℕ) (i : Fin n), k ≤ (i : ℕ) → k ≤ (σ k i : ℕ)) :
    ∀ m : ℕ, m ≤ n →
      (∀ (j : ℕ) (h : j < n), j < m →
        ((σ j).permMatrix K * (elimMP A σ j * A)) ⟨j, h⟩ ⟨j, h⟩ ≠ 0) →
      (∀ i j : Fin n, (j : ℕ) < m → j < i → (elimMP A σ m * A) i j = 0) ∧
      (∀ j : Fin n, (j : ℕ) < m → (elimMP A σ m * A) j j ≠ 0) := by
  intro m
  induction m with
  | zero =>
    intro _ _
    exact ⟨fun _ _ hj _ => absurd hj (Nat.not_lt_zero _),
           fun _ hj => absurd hj (Nat.not_lt_zero _)⟩
  | succ k ih =>
    intro hk1 hpiv_lt
    have hk : k < n := Nat.lt_of_succ_le hk1
    obtain ⟨ihU, ihD⟩ := ih (le_of_lt hk) (fun j h hj => hpiv_lt j h (Nat.lt_succ_of_lt hj))
    have hp : ((σ k).permMatrix K * (elimMP A σ k * A)) ⟨k, hk⟩ ⟨k, hk⟩ ≠ 0 :=
      hpiv_lt k hk (Nat.lt_succ_self k)
    have key : elimMP A σ (k + 1) * A
        = gaussTransform ((σ k).permMatrix K * (elimMP A σ k * A)) ⟨k, hk⟩
            * ((σ k).permMatrix K * (elimMP A σ k * A)) := by
      rw [elimMP_succ A σ hk]; simp only [Matrix.mul_assoc]
    have hVent : ∀ i j : Fin n, ((σ k).permMatrix K * (elimMP A σ k * A)) i j
        = (elimMP A σ k * A) (σ k i) j :=
      fun i j => permMatrix_mul_apply (σ k) (elimMP A σ k * A) i j
    refine ⟨?_, ?_⟩
    · intro i j hj hji
      rw [key]
      rcases (Nat.lt_succ_iff.mp hj).lt_or_eq with hjk | hjk
      · rw [gaussTransform_mul_apply]
        have hpivrow : ((σ k).permMatrix K * (elimMP A σ k * A)) ⟨k, hk⟩ j = 0 := by
          rw [hVent]
          refine ihU _ j hjk ?_
          have hge' : k ≤ (σ k ⟨k, hk⟩ : ℕ) := hge k ⟨k, hk⟩ (le_refl k)
          simp only [Fin.lt_def]; omega
        have hrow : ((σ k).permMatrix K * (elimMP A σ k * A)) i j = 0 := by
          rw [hVent]
          refine ihU _ j hjk ?_
          rcases Nat.lt_or_ge (i : ℕ) k with hik | hik
          · rw [hfix k i hik]; exact hji
          · have hge' : k ≤ (σ k i : ℕ) := hge k i hik
            simp only [Fin.lt_def]; omega
        rw [hpivrow, hrow]; ring
      · have hjeq : j = ⟨k, hk⟩ := Fin.ext hjk
        subst hjeq
        exact gaussTransform_clears _ ⟨k, hk⟩ i hji hp
    · intro j hj
      rw [key]
      rcases (Nat.lt_succ_iff.mp hj).lt_or_eq with hjk | hjk
      · have hik : ¬ (⟨k, hk⟩ : Fin n) < j := by simp only [Fin.lt_def]; omega
        rw [gaussTransform_preserves _ _ _ _ hik, hVent, hfix k j hjk]
        exact ihD j hjk
      · have hjeq : j = ⟨k, hk⟩ := Fin.ext hjk
        subst hjeq
        rw [gaussTransform_preserves _ _ _ _ (lt_irrefl _)]
        exact hp

/-- **Pivot swaps exist (nonsingular `A`).** For invertible `A` there are per-column
    transpositions `σ` meeting every structural condition of `gaussianSolvePivot_solves`
    — built column by column: at `k`, `pivot_exists` on the invertible eliminated matrix
    `Uₖ` gives a nonzero pivot row `p ≥ k`, and `σ k := Equiv.swap ⟨k⟩ p`. Sorry-free. -/
theorem exists_pivot_perm (A : Matrix (Fin n) (Fin n) K) (hA : IsUnit A.det) :
    ∀ m : ℕ, m ≤ n → ∃ σ : ℕ → Equiv.Perm (Fin n),
      (∀ (k : ℕ) (i : Fin n), (i : ℕ) < k → σ k i = i) ∧
      (∀ (k : ℕ) (i : Fin n), k ≤ (i : ℕ) → k ≤ (σ k i : ℕ)) ∧
      (∀ (j : ℕ) (h : j < n), j < m →
        ((σ j).permMatrix K * (elimMP A σ j * A)) ⟨j, h⟩ ⟨j, h⟩ ≠ 0) := by
  intro m
  induction m with
  | zero =>
    intro _
    refine ⟨fun _ => 1, ?_, ?_, ?_⟩
    · intro _ _ _; rfl
    · intro _ i hi; simpa using hi
    · intro _ _ hj; exact absurd hj (Nat.not_lt_zero _)
  | succ m ih =>
    intro hm1
    have hm : m < n := Nat.lt_of_succ_le hm1
    obtain ⟨σ, hfix, hge, hpivIH⟩ := ih (le_of_lt hm)
    have hUinv : IsUnit ((elimMP A σ m * A).det) := elimMP_mul_isUnit_det A σ m hA
    have hupperK := (elimMP_invariant_lt A σ hfix hge m (le_of_lt hm) hpivIH).1
    obtain ⟨p, hp_ge, hp_ne⟩ := pivot_exists (elimMP A σ m * A) hUinv ⟨m, hm⟩
      (fun i j hj hji => hupperK i j hj hji)
    have hpge : m ≤ (p : ℕ) := hp_ge
    refine ⟨fun j => if j = m then Equiv.swap ⟨m, hm⟩ p else σ j, ?_, ?_, ?_⟩
    · -- hfix
      intro k i hi
      show (if k = m then Equiv.swap ⟨m, hm⟩ p else σ k) i = i
      by_cases hkK : k = m
      · have him : (i : ℕ) < m := hkK ▸ hi
        rw [if_pos hkK]
        apply Equiv.swap_apply_of_ne_of_ne
        · exact Fin.ne_of_val_ne (show (i : ℕ) ≠ m by omega)
        · exact Fin.ne_of_val_ne (show (i : ℕ) ≠ (p : ℕ) by omega)
      · rw [if_neg hkK]; exact hfix k i hi
    · -- hge
      intro k i hi
      show k ≤ ((if k = m then Equiv.swap ⟨m, hm⟩ p else σ k) i : ℕ)
      by_cases hkK : k = m
      · rw [if_pos hkK, hkK]
        rw [hkK] at hi
        by_cases e1 : i = ⟨m, hm⟩
        · rw [e1, Equiv.swap_apply_left]; exact hpge
        · by_cases e2 : i = p
          · rw [e2]; simp [Equiv.swap_apply_right]
          · rw [Equiv.swap_apply_of_ne_of_ne e1 e2]; exact hi
      · rw [if_neg hkK]; exact hge k i hi
    · -- hpiv up to m+1
      intro j h hjm1
      rcases (Nat.lt_succ_iff.mp hjm1).lt_or_eq with hjm | hjm
      · have hcj : elimMP A (fun l => if l = m then Equiv.swap ⟨m, hm⟩ p else σ l) j
            = elimMP A σ j :=
          elimMP_congr A _ σ j (fun l hl => if_neg (Nat.ne_of_lt (lt_trans hl hjm)))
        show ((if j = m then Equiv.swap ⟨m, hm⟩ p else σ j).permMatrix K
            * (elimMP A (fun l => if l = m then Equiv.swap ⟨m, hm⟩ p else σ l) j * A))
            ⟨j, h⟩ ⟨j, h⟩ ≠ 0
        rw [if_neg (Nat.ne_of_lt hjm), hcj]
        exact hpivIH j h hjm
      · subst hjm
        have hcK : elimMP A (fun l => if l = j then Equiv.swap ⟨j, hm⟩ p else σ l) j
            = elimMP A σ j :=
          elimMP_congr A _ σ j (fun l hl => if_neg (Nat.ne_of_lt hl))
        show ((if j = j then Equiv.swap ⟨j, hm⟩ p else σ j).permMatrix K
            * (elimMP A (fun l => if l = j then Equiv.swap ⟨j, hm⟩ p else σ l) j * A))
            ⟨j, h⟩ ⟨j, h⟩ ≠ 0
        rw [if_pos rfl, hcK, permMatrix_mul_apply,
            show (Equiv.swap ⟨j, hm⟩ p) ⟨j, h⟩ = p from Equiv.swap_apply_left _ _]
        exact hp_ne

/-- **Dense Gaussian solver correctness — unconditional for invertible `A` (LE1-b
    fully closed).** For an invertible `A`, partial-pivoting swaps `σ` exist for which
    the solver solves the system exactly, with **no pivot hypothesis** — the
    nonzero-pivot condition is discharged by nonsingularity. Sorry-free. -/
theorem exists_gaussianSolvePivot_solves (A : Matrix (Fin n) (Fin n) K) (b : Fin n → K)
    (hA : IsUnit A.det) :
    ∃ σ : ℕ → Equiv.Perm (Fin n), A *ᵥ gaussianSolvePivot A σ b = b := by
  obtain ⟨σ, hfix, hge, hpiv⟩ := exists_pivot_perm A hA n (le_refl n)
  exact ⟨σ, gaussianSolvePivot_solves A σ b hfix hge (fun k h => hpiv k h h)⟩

end QuantumProofs.GaussianSolve
