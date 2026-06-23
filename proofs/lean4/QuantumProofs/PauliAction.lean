/-
  Pauli-action kernel correctness (LEAN_EXPORT LE1-b, `cqs::apply_pauli`).

  The Rust kernel `cqs::apply_pauli(pauli, state)` applies a length-`n` Pauli
  string (byte-encoded `0=I, 1=X, 2=Y, 3=Z`, qubit `q` = bit `q`) to a state
  vector by a scatter-add loop:

      out[x ^ flip] += phase(x) · state[x]                      (over all x)

  where `flip` has bit `q` set iff `Pᵩ ∈ {X, Y}`, and per qubit the phase is
      X : 1            Y : i·(−1)^{xᵩ}            Z : (−1)^{xᵩ}            I : 1.

  This file pins down three things:

  1. The **single-qubit** phase/flip convention (`applyPauli1_eq_mulVec`),
     validated against the ground-truth gate matrices `Gates.X / Gates.Y /
     Gates.Z`.  The `Y` phase `i·(−1)^b` (the easy place to drop a sign or an
     `i`), the `Z` sign, and the `X`/`Y` flip are exactly the parts of
     `apply_pauli` a bug would live in; each is shown to reproduce the standard
     matrix action `state ↦ P·state` exactly.

  2. The **general-`n`** reading (`applyPauliN_eq_mulVec`): the full scatter-add
     loop over a length-`n` Pauli string equals the matrix–vector product by the
     Pauli tensor `⨂_q Pᵩ`.  Working on the bit-string basis `Fin n → Fin 2`
     (the natural tensor-product index), the sum over all `2ⁿ` basis indices
     collapses to the unique source `pauliSrcN p y` — because each `pauliGate`
     has a single nonzero entry per row — so the kernel's scatter-add reproduces
     the tensor matrix action exactly.

  3. The **flat-array** form (`applyPauliFlat_eq_mulVec`): the same statement on
     the *literal* flat `Fin (2^n)` state vector the kernel actually indexes,
     where the source is `Y ^^^ flip` (`Nat` XOR with the flip-mask).  The bridge
     `bitEquiv_pauliSrcN` proves — bitwise, via the little-endian encoder
     `bitEquiv = finFunctionFinEquiv` — that the pointwise-`Fin.rev` source of (2)
     is exactly this XOR, so `applyPauliFlat p state = (⨂_q Pᵩ) ·ᵥ state` on
     `Fin (2^n)`.

  Only remaining LE1-b gap: `circulant::solve_dense_gaussian` correctness (the
  returned `x` satisfies `A·x = b` for Gaussian elimination with partial
  pivoting) — a separate, larger imperative-algorithm formalization.
-/

import QuantumProofs.Gates
import Mathlib.Data.Nat.Bitwise

namespace QuantumProofs.PauliAction

open Complex Matrix QuantumProofs.Gates

/-- The single-qubit ground-truth matrix for a Pauli byte code
    (`1 = X, 2 = Y, 3 = Z`, anything else `I`). Mirrors the `match pq` arms of
    `cqs::apply_pauli` at the *matrix* level — the reference the kernel must
    reproduce. -/
noncomputable def pauliGate : ℕ → Matrix (Fin 2) (Fin 2) ℂ
  | 1 => Gates.X
  | 2 => Gates.Y
  | 3 => Gates.Z
  | _ => 1

/-- Per-qubit phase the kernel multiplies in, as a function of the *source*
    bit `x` (faithful to `apply_pauli`'s inner `match pq`): `Y ↦ i·(−1)ˣ`,
    `Z ↦ (−1)ˣ`, else `1`. -/
noncomputable def pauliPhase1 (k : ℕ) (x : Fin 2) : ℂ :=
  if k = 2 then Complex.I * (if x = 1 then -1 else 1)
  else if k = 3 then (if x = 1 then -1 else 1)
  else 1

/-- The source index the kernel reads for output index `y`: `apply_pauli` writes
    `out[x ^ flip]`, so the (unique) `x` feeding `out[y]` is `y` with the flip
    bit undone. For one qubit the flip is set exactly for `X`/`Y` (`k ∈ {1,2}`),
    and the bit-flip is `Fin.rev` (`0 ↔ 1`); for `I`/`Z` the source is `y`. -/
def pauliSrc (k : ℕ) (y : Fin 2) : Fin 2 := if k = 1 ∨ k = 2 then y.rev else y

/-- **Faithful single-qubit model of `cqs::apply_pauli`** (`n = 1`). The
    scatter-add `out[x ^ flip] += phase(x)·state[x]` has a unique source
    `x = pauliSrc k y` for each output `y`, so `out[y] = phase(x)·state[x]`. -/
noncomputable def applyPauli1 (k : ℕ) (state : Fin 2 → ℂ) (y : Fin 2) : ℂ :=
  pauliPhase1 k (pauliSrc k y) * state (pauliSrc k y)

/-- **Single-qubit `apply_pauli` correctness.** For every Pauli code `k`, the
    kernel's single-qubit action equals the matrix–vector product by the
    ground-truth gate `pauliGate k`: `applyPauli1 k state = (pauliGate k) ·ᵥ
    state`. Covers `I, X, Y, Z` uniformly — in particular it certifies the
    `Y` phase `i·(−1)ˣ` and the `Z` sign against `Gates.Y` / `Gates.Z`. -/
theorem applyPauli1_eq_mulVec (k : ℕ) (state : Fin 2 → ℂ) :
    applyPauli1 k state = (pauliGate k).mulVec state := by
  funext y
  -- Split on the four relevant codes (I/other, X, Y, Z); each is a 2×2 check.
  match k with
  | 1 =>
      fin_cases y <;>
        simp [applyPauli1, pauliPhase1, pauliSrc, pauliGate, Gates.X, Matrix.mulVec,
              dotProduct, Fin.sum_univ_two, Fin.rev]
  | 2 =>
      fin_cases y <;>
        simp [applyPauli1, pauliPhase1, pauliSrc, pauliGate, Gates.Y, Matrix.mulVec,
              dotProduct, Fin.sum_univ_two, Fin.rev]
  | 3 =>
      fin_cases y <;>
        simp [applyPauli1, pauliPhase1, pauliSrc, pauliGate, Gates.Z, Matrix.mulVec,
              dotProduct, Fin.sum_univ_two]
  | 0 =>
      fin_cases y <;>
        simp [applyPauli1, pauliPhase1, pauliSrc, pauliGate, Matrix.mulVec,
              dotProduct, Matrix.one_apply]
  | (m + 4) =>
      fin_cases y <;>
        simp [applyPauli1, pauliPhase1, pauliSrc, pauliGate, Matrix.mulVec,
              dotProduct, Matrix.one_apply]

/-- `X` is involutive on the kernel action (`apply_pauli [1]` twice = identity),
    a direct corollary: `(X·ᵥ)∘(X·ᵥ) = id` via `X_self_inverse`. A small sanity
    check that the model composes like the matrix. -/
theorem applyPauli1_X_involutive (state : Fin 2 → ℂ) :
    applyPauli1 1 (applyPauli1 1 state) = state := by
  rw [applyPauli1_eq_mulVec, applyPauli1_eq_mulVec, Matrix.mulVec_mulVec]
  show (Gates.X * Gates.X) *ᵥ state = state
  rw [Gates.X_self_inverse, Matrix.one_mulVec]

/-- **Matrix-entry form of a single Pauli gate.** Every `pauliGate k` has exactly
    one nonzero entry per row: column `b = pauliSrc k a` carries the phase
    `pauliPhase1 k b`, all others vanish. This is the bridge between the *matrix*
    picture (`pauliGate`) and the *kernel* picture (`pauliSrc` + `pauliPhase1`),
    and the seed for the general-`n` tensor reading below. -/
theorem pauliGate_apply_eq (k : ℕ) (a b : Fin 2) :
    pauliGate k a b = if b = pauliSrc k a then pauliPhase1 k b else 0 := by
  match k with
  | 1 => fin_cases a <;> fin_cases b <;>
           simp [pauliGate, pauliSrc, pauliPhase1, Gates.X, Fin.rev]
  | 2 => fin_cases a <;> fin_cases b <;>
           simp [pauliGate, pauliSrc, pauliPhase1, Gates.Y, Fin.rev]
  | 3 => fin_cases a <;> fin_cases b <;>
           simp [pauliGate, pauliSrc, pauliPhase1, Gates.Z]
  | 0 => fin_cases a <;> fin_cases b <;>
           simp [pauliGate, pauliSrc, pauliPhase1]
  | (m + 4) => fin_cases a <;> fin_cases b <;>
           simp [pauliGate, pauliSrc, pauliPhase1]

-- ===========================================================================
-- General-`n` Pauli string: scatter-add = matrix–vector product by ⨂_q Pᵩ
-- ===========================================================================

section GeneralN

variable {n : ℕ}

/-- Phase the kernel multiplies in for an `n`-qubit Pauli string `p : Fin n → ℕ`
    at the basis index `x : Fin n → Fin 2`: the product of the per-qubit phases
    (`apply_pauli` accumulates one factor per qubit as it walks the string). -/
noncomputable def pauliPhaseN (p : Fin n → ℕ) (x : Fin n → Fin 2) : ℂ :=
  ∏ q, pauliPhase1 (p q) (x q)

/-- The (unique) source index `apply_pauli` reads for output index `y`, qubitwise:
    flip bit `q` iff `Pᵩ ∈ {X, Y}`. This is the flat-array flip-mask `y ^ flip`
    expressed on the bit-string index `Fin n → Fin 2` (pointwise `Fin.rev`). -/
def pauliSrcN (p : Fin n → ℕ) (y : Fin n → Fin 2) : Fin n → Fin 2 :=
  fun q => pauliSrc (p q) (y q)

/-- **Faithful `n`-qubit model of `cqs::apply_pauli`.** The scatter-add
    `out[x ^ flip] += phase(x)·state[x]` has, for each output `y`, a unique source
    `x = pauliSrcN p y` (the flip is a bijection), contributing
    `pauliPhaseN p x · state x`. -/
noncomputable def applyPauliN (p : Fin n → ℕ) (state : (Fin n → Fin 2) → ℂ) :
    (Fin n → Fin 2) → ℂ :=
  fun y => pauliPhaseN p (pauliSrcN p y) * state (pauliSrcN p y)

/-- The `n`-qubit Pauli tensor `⨂_q Pᵩ` as a matrix in the computational
    (bit-string) basis `Fin n → Fin 2`: its `(y, x)` entry is the product of the
    single-qubit entries `∏_q (Pᵩ)_{yᵩ xᵩ}` — the standard Kronecker-product
    formula in the basis of bit strings. -/
noncomputable def pauliTensor (p : Fin n → ℕ) :
    Matrix (Fin n → Fin 2) (Fin n → Fin 2) ℂ :=
  fun y x => ∏ q, pauliGate (p q) (y q) (x q)

/-- **Single-nonzero-entry form of the Pauli tensor.** Because each `pauliGate`
    has one nonzero entry per row (`pauliGate_apply_eq`), the product over qubits
    is nonzero only at the single source index `x = pauliSrcN p y`, where it
    equals `pauliPhaseN p x`. -/
theorem pauliTensor_apply (p : Fin n → ℕ) (y x : Fin n → Fin 2) :
    pauliTensor p y x = if x = pauliSrcN p y then pauliPhaseN p x else 0 := by
  have hfac : ∀ q, pauliGate (p q) (y q) (x q)
      = if x q = pauliSrc (p q) (y q) then pauliPhase1 (p q) (x q) else 0 :=
    fun q => pauliGate_apply_eq _ _ _
  rw [pauliTensor]
  simp only [hfac]
  by_cases h : x = pauliSrcN p y
  · -- every qubit's guard holds → the product collapses to the phase product
    rw [if_pos h, pauliPhaseN]
    refine Finset.prod_congr rfl (fun q _ => ?_)
    have hq : x q = pauliSrc (p q) (y q) := by
      have := congrFun h q; rwa [pauliSrcN] at this
    rw [if_pos hq]
  · -- some qubit's guard fails → one factor is 0, so the product is 0
    rw [if_neg h]
    obtain ⟨q, hq⟩ := Function.ne_iff.mp h
    refine Finset.prod_eq_zero (Finset.mem_univ q) ?_
    rw [if_neg]
    simpa only [pauliSrcN] using hq

/-- **General-`n` `apply_pauli` correctness.** The scatter-add kernel applied to a
    full length-`n` Pauli string equals the matrix–vector product by the Pauli
    tensor `⨂_q Pᵩ`: `applyPauliN p state = pauliTensor p ·ᵥ state`. The sum over
    all `2ⁿ` basis indices collapses to the unique source `pauliSrcN p y`, so the
    matrix action reproduces the kernel's scatter-add exactly. Together with
    `applyPauli1_eq_mulVec` this closes the kernel↔matrix correspondence for
    `cqs::apply_pauli` over the bit-string basis. -/
theorem applyPauliN_eq_mulVec (p : Fin n → ℕ) (state : (Fin n → Fin 2) → ℂ) :
    applyPauliN p state = (pauliTensor p).mulVec state := by
  funext y
  have hsum : (pauliTensor p).mulVec state y
      = ∑ x, (if x = pauliSrcN p y then pauliPhaseN p x * state x else 0) := by
    simp only [Matrix.mulVec, dotProduct]
    exact Finset.sum_congr rfl
      (fun x _ => by rw [pauliTensor_apply, ite_mul, zero_mul])
  rw [hsum, Finset.sum_ite_eq' Finset.univ (pauliSrcN p y)
        (fun x => pauliPhaseN p x * state x)]
  simp only [Finset.mem_univ, if_true, applyPauliN]

end GeneralN

-- ===========================================================================
-- Flat-array bridge: pointwise `Fin.rev` flip ↔ literal `x ^ flip` XOR
-- ===========================================================================

section FlatBridge

variable {n : ℕ}

/-- The little-endian bit-string encoder `(Fin n → Fin 2) ≃ Fin (2^n)` of the
    flat state-vector layout: `bitEquiv f = ∑_q (f q)·2^q`. This is mathlib's
    `finFunctionFinEquiv` specialised to qubits, and the dictionary translating
    the bit-string-basis result above to the flat `Fin (2^n)` array the kernel
    actually indexes. -/
abbrev bitEquiv : (Fin n → Fin 2) ≃ Fin (2 ^ n) := finFunctionFinEquiv

/-- Digit extraction: the base-2 digit `j` of the flat index `bitEquiv f` is the
    qubit-`j` bit `f j`. Falls straight out of the equivalence's explicit inverse
    `a ↦ a / 2^j % 2`. -/
lemma bitEquiv_div_mod (f : Fin n → Fin 2) (j : Fin n) :
    (bitEquiv f : ℕ) / 2 ^ (j : ℕ) % 2 = (f j : ℕ) := by
  have hsymm : ((finFunctionFinEquiv.symm (finFunctionFinEquiv f)) j : ℕ)
      = (finFunctionFinEquiv f : ℕ) / 2 ^ (j : ℕ) % 2 := rfl
  rw [Equiv.symm_apply_apply] at hsymm
  exact hsymm.symm

/-- The `j`-th bit of the flat index `bitEquiv f` (for `j < n`) is set iff qubit
    `j` carries a 1 — the testBit form of `bitEquiv_div_mod`. -/
lemma testBit_bitEquiv (f : Fin n → Fin 2) {j : ℕ} (hj : j < n) :
    (bitEquiv f : ℕ).testBit j = decide ((f ⟨j, hj⟩ : ℕ) = 1) := by
  rw [Nat.testBit_eq_decide_div_mod_eq,
      show (bitEquiv f : ℕ) / 2 ^ j % 2 = (f ⟨j, hj⟩ : ℕ) from bitEquiv_div_mod f ⟨j, hj⟩]

/-- Indicator of the qubits `apply_pauli` flips: bit `q` is 1 iff `Pᵩ ∈ {X, Y}`. -/
def flipVec (p : Fin n → ℕ) : Fin n → Fin 2 :=
  fun q => if p q = 1 ∨ p q = 2 then 1 else 0

/-- The kernel's flat XOR flip-mask `flip`: the integer whose bit `q` is set iff
    `Pᵩ ∈ {X, Y}` (i.e. `bitEquiv` of `flipVec`). -/
def pauliFlipMask (p : Fin n → ℕ) : ℕ := (bitEquiv (flipVec p) : ℕ)

lemma pauliFlipMask_lt (p : Fin n → ℕ) : pauliFlipMask p < 2 ^ n :=
  (bitEquiv (flipVec p)).isLt

/-- Single-bit core of the flat bridge: at one qubit, the source bit
    `pauliSrc k b` is set iff `b`'s bit XOR the flip indicator (`Pₖ ∈ {X, Y}`) is
    set. The whole bit-string identity is this, qubit by qubit. -/
private lemma pauliSrc_testBit (k : ℕ) (b : Fin 2) :
    decide ((pauliSrc k b : ℕ) = 1)
      = (decide ((b : ℕ) = 1) ^^
          decide (((if k = 1 ∨ k = 2 then 1 else 0 : Fin 2) : ℕ) = 1)) := by
  unfold pauliSrc
  by_cases hc : k = 1 ∨ k = 2 <;> fin_cases b <;> simp [hc, Fin.rev]

/-- **Flat-index bridge.** Under the encoder `bitEquiv`, the pointwise-`Fin.rev`
    source `pauliSrcN p y` (the index `applyPauliN` reads) is *exactly* the
    kernel's literal XOR with the flip-mask:
    `bitEquiv (pauliSrcN p y) = bitEquiv y ^^^ pauliFlipMask p`. Proved bitwise:
    within the `n` qubit bits each side flips bit `q` iff `Pᵩ ∈ {X, Y}`, and
    above bit `n` both sides vanish (every flat index is `< 2^n`). This is the
    book-keeping that carries the bit-string-basis correspondence onto the flat
    `out[x ^ flip]` array layout. -/
theorem bitEquiv_pauliSrcN (p : Fin n → ℕ) (y : Fin n → Fin 2) :
    (bitEquiv (pauliSrcN p y) : ℕ) = (bitEquiv y : ℕ) ^^^ pauliFlipMask p := by
  apply Nat.eq_of_testBit_eq
  intro j
  rw [Nat.testBit_xor]
  by_cases hj : j < n
  · -- within the n bits: bit j flips iff Pⱼ ∈ {X, Y}
    rw [testBit_bitEquiv _ hj, testBit_bitEquiv _ hj, pauliFlipMask, testBit_bitEquiv _ hj]
    simp only [pauliSrcN, flipVec]
    exact pauliSrc_testBit (p ⟨j, hj⟩) (y ⟨j, hj⟩)
  · -- above bit n: every flat index is < 2^n, so all three bits are 0
    have hpow : (2 : ℕ) ^ n ≤ 2 ^ j := Nat.pow_le_pow_right (by norm_num) (Nat.le_of_not_lt hj)
    rw [Nat.testBit_eq_false_of_lt (lt_of_lt_of_le (bitEquiv _).isLt hpow),
        Nat.testBit_eq_false_of_lt (lt_of_lt_of_le (bitEquiv _).isLt hpow),
        Nat.testBit_eq_false_of_lt (lt_of_lt_of_le (pauliFlipMask_lt p) hpow)]
    rfl

/-- The flat source index `apply_pauli` gathers from for output `Y`: literally
    `Y ^^^ flip`, kept in `Fin (2^n)` via `Nat.xor_lt_two_pow`. -/
def flatSrc (p : Fin n → ℕ) (Y : Fin (2 ^ n)) : Fin (2 ^ n) :=
  ⟨(Y : ℕ) ^^^ pauliFlipMask p, Nat.xor_lt_two_pow Y.isLt (pauliFlipMask_lt p)⟩

/-- The literal XOR source equals the transported pointwise-flip source:
    `flatSrc p Y = bitEquiv (pauliSrcN p (bitEquiv.symm Y))`. Immediate from the
    bridge. -/
theorem flatSrc_eq (p : Fin n → ℕ) (Y : Fin (2 ^ n)) :
    flatSrc p Y = bitEquiv (pauliSrcN p (bitEquiv.symm Y)) := by
  apply Fin.ext
  rw [bitEquiv_pauliSrcN, Equiv.apply_symm_apply]
  rfl

/-- The `n`-qubit Pauli tensor on the flat state-vector layout `Fin (2^n)`: the
    bit-string tensor `pauliTensor p` reindexed along `bitEquiv`. -/
noncomputable def pauliTensorFlat (p : Fin n → ℕ) :
    Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ :=
  Matrix.reindex bitEquiv bitEquiv (pauliTensor p)

/-- **Faithful flat-array model of `cqs::apply_pauli`.** Output `Y` gathers from
    `flatSrc p Y = Y ^^^ flip` with phase `pauliPhaseN` read off the source bits
    — exactly the kernel's `out[x ^ flip] += phase(x)·state[x]` loop on the flat
    `Fin (2^n)` vector. -/
noncomputable def applyPauliFlat (p : Fin n → ℕ) (state : Fin (2 ^ n) → ℂ) :
    Fin (2 ^ n) → ℂ :=
  fun Y => pauliPhaseN p (bitEquiv.symm (flatSrc p Y)) * state (flatSrc p Y)

/-- **Flat `apply_pauli` correctness.** On the literal flat `Fin (2^n)` layout the
    kernel's `out[x ^ flip]` scatter-add equals the matrix–vector product by the
    Pauli tensor `⨂_q Pᵩ`: `applyPauliFlat p state = pauliTensorFlat p ·ᵥ state`.
    This is `applyPauliN_eq_mulVec` transported along `bitEquiv`, with the XOR
    flip-mask discharged by `flatSrc_eq` — closing the kernel↔matrix
    correspondence on the indexing the Rust kernel actually uses. -/
theorem applyPauliFlat_eq_mulVec (p : Fin n → ℕ) (state : Fin (2 ^ n) → ℂ) :
    applyPauliFlat p state = (pauliTensorFlat p).mulVec state := by
  funext Y
  rw [pauliTensorFlat, Matrix.reindex_apply, Matrix.submatrix_mulVec_equiv,
      Function.comp_apply, Equiv.symm_symm, ← applyPauliN_eq_mulVec]
  rw [applyPauliFlat, flatSrc_eq, Equiv.symm_apply_apply]
  rfl

end FlatBridge

end QuantumProofs.PauliAction
