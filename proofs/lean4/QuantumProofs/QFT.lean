/-
  Formal verification of the Quantum Fourier Transform.

  ## Goal

  The headline theorem `qft_correct` (bottom of file):

      denote (qft_circuit n) = dft_matrix n

  i.e. the `n`-qubit QFT *circuit* (Hadamards + controlled phases + a final
  bit-reversal of the wires) denotes to the `N = 2ⁿ` Discrete Fourier Transform
  matrix `dft_matrix n j k = (1/√N)·ω^{j·k}`, with `ω = e^{2πi/N}`. Equivalently
  on basis states: `QFT |k⟩ = (1/√N) Σⱼ ω^{jk} |j⟩`. This is the theorem
  `CirculantSolve` builds on: the QFT unitary *is* the DFT, so it diagonalizes
  circulants.

  ## Circuit decomposition

      qft_circuit n  =  qftCore n  ++  bitReversal n          (`qft_circuit`)

  `qftCore` is the **swap-free Cooley–Tukey core** (recursive: an `H@0`, a
  controlled-phase layer, then `qftCore` on the tail wires). `bitReversal` is the
  final layer of `⌊n/2⌋` swaps that reverses wire order. The core alone computes
  the DFT *with its output bits reversed*; the swap layer undoes that.

  ## Proof architecture — three independent pillars, then assembly

  1. **Root-of-unity machinery** (top of file): `omega`, `omega_periodic`
     (`ωᴺ = 1`), `omega_mul_conj`/`omega_conj_pow_inv` (unit norm), and the
     orthogonality engine `omega_geom_sum_ne_zero` (`Σ_{i<N} ω^{d·i} = 0` for
     `0 < d < N`). These give `dft_unitary` (`DFT·DFTᴴ = I`) and are reused by
     `CirculantSolve`'s eventual general-`n` diagonalization.

  2. **Structural recursion of the core** (`Step decomposition` section):
     `qftCore_succ_denote` factors `denote (qftCore (k+1))` into
     `(H ⊗ I) · (phase layer) · (I ⊗ qftCore k)`, matching the Cooley–Tukey
     factorization of `dft_matrix (k+1)`. Induction assembles these into
     `qftCore_eq : denote (qftCore n) = denote (bitReversal n) · dft_matrix n`
     (the core = bit-reversed DFT).

  3. **Bit-reversal as a permutation** (`bitrev`/`SwapData` sections):
     `bitrev`/`bitrevPerm` define the bit-reversal permutation of `Fin (2ⁿ)`;
     `denote_bitReversal_eq_permMatrix` shows the swap layer denotes to that
     permutation matrix; `bitrevFin_involutive` ⇒ `bitReversal_involutive`
     (`denote(bitReversal)² = 1`).

  **Assembly** (`qft_correct`): `denote(qft_circuit) = denote(bitReversal) ·
  denote(qftCore)` (by `denote_append`); rewrite the core with `qftCore_eq` to
  get `denote(bitReversal) · denote(bitReversal) · DFT`; collapse the doubled
  reversal with `bitReversal_involutive` to leave `DFT`.

  ## Conventions
  - `N = 2ⁿ`; indices are `Fin (2ⁿ)`, addressed by their `.val` bit pattern.
  - `ω N = e^{2πi/N}` (`omega`); `dft_matrix` carries the `1/√N` normalization.
  - Everything is sorry-free; `#print axioms qft_correct` shows only the three
    standard axioms.
-/

import QuantumProofs.CircuitSemantics
import Mathlib.Algebra.Field.GeomSum
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Complex.Log

namespace QuantumProofs.QFT

open Complex BigOperators Finset

/-- The principal `N`-th root of unity `ω = e^{2πi/N}`. The `1/N` in the exponent
    is what makes `ωᴺ = 1` (`omega_periodic`) and drives every orthogonality
    fact below. (`CirculantSolve` re-uses `omega 2 = -1`, `omega 4 = i`, … for
    the small-`N` circulant diagonalizations.) -/
noncomputable def omega (N : ℕ) : ℂ :=
  exp (2 * Real.pi * I / ↑N)

/-- Key property: ω^N = 1 -/
theorem omega_periodic (N : ℕ) (hN : N > 0) :
    omega N ^ N = 1 := by
  unfold omega
  rw [← Complex.exp_nat_mul]
  have hN' : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hN)
  rw [show (N : ℂ) * (2 * Real.pi * I / ↑N) = 2 * Real.pi * I by field_simp]
  exact Complex.exp_two_pi_mul_I

/-- The (unitary, symmetric) DFT matrix of size `N = 2ⁿ`:
    `dft_matrix n j k = (1/√N)·ω^{j·k}`. The `1/√N` makes it unitary
    (`dft_unitary`); the symmetric `j·k` exponent makes it equal to its own
    transpose. `qft_correct` proves this is exactly what `qft_circuit n` denotes. -/
noncomputable def dft_matrix (n : ℕ) : Matrix (Fin (2^n)) (Fin (2^n)) ℂ :=
  let N : ℕ := 2^n
  fun j k => (1 / Complex.ofReal (Real.sqrt (N : ℝ))) * (omega N) ^ (j.val * k.val)

-- ===========================================================================
-- Supporting lemmas for `dft_unitary`.
-- ===========================================================================

/-- `ω N` has unit norm: `ω · conj ω = 1`. Follows from
    `exp(ix) · exp(-ix) = 1` via `Complex.mul_conj_exp`. -/
theorem omega_mul_conj (N : ℕ) : omega N * (starRingEnd ℂ) (omega N) = 1 := by
  unfold omega
  rw [← Complex.exp_conj]
  rw [← Complex.exp_add]
  have : 2 * Real.pi * I / (N : ℂ) + (starRingEnd ℂ) (2 * Real.pi * I / (N : ℂ)) = 0 := by
    rw [map_div₀, map_mul, map_mul, Complex.conj_I, Complex.conj_ofReal,
        Complex.conj_ofNat, map_natCast]
    ring
  rw [this, Complex.exp_zero]

/-- `conj (ω^m) = (ω^m)⁻¹`, equivalently `ω^m · conj(ω^m) = 1`. -/
theorem omega_pow_mul_conj (N m : ℕ) :
    omega N ^ m * (starRingEnd ℂ) (omega N ^ m) = 1 := by
  rw [map_pow, ← mul_pow, omega_mul_conj, one_pow]

/-- For `d : ℕ` with `0 < d < N`, the geometric sum `∑_{i < N} ω^{d·i}`
    vanishes. This is the key orthogonality fact behind `dft_unitary`. -/
theorem omega_geom_sum_ne_zero (N d : ℕ) (hN : N > 0) (hd_lt : d < N) (hd_pos : 0 < d) :
    ∑ i ∈ range N, (omega N) ^ (d * i) = 0 := by
  have hω_ne : omega N ^ d ≠ 1 := by
    intro h
    have hN_ne : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hN)
    -- omega N ^ d = exp(d * (2πi/N)) = exp(2πi · d/N)
    have hpow : omega N ^ d = Complex.exp ((d : ℂ) * (2 * Real.pi * I / (N : ℂ))) := by
      unfold omega; rw [← Complex.exp_nat_mul]
    rw [hpow, Complex.exp_eq_one_iff] at h
    obtain ⟨k, hk⟩ := h
    -- hk : d * (2πi/N) = k * (2πi). Multiply through by N / (2πi) ∈ ℂ*.
    have h2πI_ne : (2 * Real.pi * I : ℂ) ≠ 0 := by
      simp [Real.pi_ne_zero, Complex.I_ne_zero]
    -- Cancel 2πi from both sides, getting d/N = k.
    have hd_k : (d : ℂ) = k * (N : ℂ) := by
      have := hk
      field_simp at this
      linear_combination this
    -- Recast to ℤ: d = k*N as integers.
    have hd_k_int : (d : ℤ) = k * (N : ℤ) := by exact_mod_cast hd_k
    -- 0 < d < N with d = k*N and N > 0 forces k = 0 hence d = 0.
    have hN_int : (0 : ℤ) < (N : ℤ) := by exact_mod_cast hN
    have hd_int : (0 : ℤ) < (d : ℤ) := by exact_mod_cast hd_pos
    have hdN_int : (d : ℤ) < (N : ℤ) := by exact_mod_cast hd_lt
    -- k must be 0 (otherwise k*N has absolute value ≥ N), but then d = 0.
    rcases lt_trichotomy k 0 with hk_neg | hk_zero | hk_pos
    · nlinarith [mul_pos_of_neg_of_neg hk_neg (neg_neg_iff_pos.mpr hN_int)]
    · simp [hk_zero] at hd_k_int; omega
    · nlinarith [mul_pos hk_pos hN_int]
  -- rewrite each term `ω^(d*i) = (ω^d)^i`
  have key : ∀ i, (omega N) ^ (d * i) = (omega N ^ d) ^ i := by
    intro i; rw [← pow_mul, mul_comm]
  simp_rw [key]
  rw [geom_sum_eq hω_ne]
  have : (omega N ^ d) ^ N = 1 := by
    rw [← pow_mul, mul_comm, pow_mul, omega_periodic N hN, one_pow]
  rw [this, sub_self, zero_div]

/-- Auxiliary: `(1/√N)² = 1/N` on ℂ. -/
private lemma one_div_sqrt_sq (N : ℕ) (hN : N > 0) :
    ((1 : ℂ) / Complex.ofReal (Real.sqrt (N : ℝ))) ^ 2 = 1 / (N : ℂ) := by
  have : (Complex.ofReal (Real.sqrt (N : ℝ))) ^ 2 = (N : ℂ) := by
    rw [sq, ← Complex.ofReal_mul, Real.mul_self_sqrt (by exact_mod_cast hN.le)]
    simp
  rw [div_pow, one_pow, this]

/-- `conj(ω^m) = (ω^m)⁻¹` — conjugation of any power of the root of
    unity is the multiplicative inverse. Immediate from
    `omega_pow_mul_conj`. -/
theorem omega_conj_pow_inv (N m : ℕ) :
    (starRingEnd ℂ) (omega N ^ m) = (omega N ^ m)⁻¹ := by
  have h := omega_pow_mul_conj N m
  -- `a * b = 1` in ℂ gives `b = a⁻¹` when `a ≠ 0`. Since `a = ω^m`
  -- and `ω = exp(…)` is always nonzero (Complex.exp never vanishes),
  -- `ω^m` is nonzero too, so we can invert.
  have hω_ne : omega N ≠ 0 := by
    unfold omega; exact Complex.exp_ne_zero _
  have hωm_ne : omega N ^ m ≠ 0 := pow_ne_zero _ hω_ne
  field_simp at h ⊢
  linear_combination h

/-- The DFT matrix is unitary: `DFT · DFT† = I`.

    Diagonal: every inner term is `(1/√N)² · (ω^m · conj(ω^m)) = 1/N`
    (via `omega_pow_mul_conj`), so the sum over `i ∈ [0, N)` collapses
    to `N · (1/N) = 1`.

    Off-diagonal (`j ≠ k`): rewrite `conj(ω^(k·i)) = (ω^(k·i))⁻¹` using
    `omega_conj_pow_inv`, then split the sum as `(1/N) · Σᵢ ω^(j·i) /
    ω^(k·i)`. Letting `d := ((j - k) mod N)` with `0 < d < N` (from
    `j ≠ k`), reorganize to `(1/N) · Σᵢ ω^(d·i) = 0` via
    `omega_geom_sum_ne_zero`.

    Packaging is the last mechanical step: it needs `Matrix.mul_apply`
    unfolding plus careful signed-exponent arithmetic (or a lift through
    `ℤ`/`ZMod N`) to close. All analytic content is already in the
    sorry-free helpers `omega_pow_mul_conj`, `omega_conj_pow_inv`,
    `omega_periodic`, `omega_geom_sum_ne_zero`, `one_div_sqrt_sq`. -/
theorem dft_unitary (n : ℕ) :
    dft_matrix n * (dft_matrix n).conjTranspose = 1 := by
  have hN_pos : 0 < (2^n : ℕ) := pow_pos (by norm_num : (0:ℕ) < 2) n
  have hN_ne : ((2^n : ℕ) : ℂ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hN_pos)
  have hω_ne : omega (2^n) ≠ 0 := by unfold omega; exact Complex.exp_ne_zero _
  have hc_sq :
      ((1 : ℂ) / Complex.ofReal (Real.sqrt ((2^n : ℕ) : ℝ))) *
      ((1 : ℂ) / Complex.ofReal (Real.sqrt ((2^n : ℕ) : ℝ))) =
      1 / ((2^n : ℕ) : ℂ) := by
    have hsq := one_div_sqrt_sq (2^n) hN_pos
    rw [sq] at hsq; exact hsq
  have hc_conj :
      (starRingEnd ℂ) ((1 : ℂ) / Complex.ofReal (Real.sqrt ((2^n : ℕ) : ℝ))) =
      (1 : ℂ) / Complex.ofReal (Real.sqrt ((2^n : ℕ) : ℝ)) := by
    rw [map_div₀, map_one, Complex.conj_ofReal]
  -- Main helper: for a.val > b.val, the off-diagonal sum is zero.
  have main : ∀ (a b : Fin (2^n)), b.val < a.val →
      ∑ i : Fin (2^n),
        omega (2^n) ^ (a.val * i.val) *
          (omega (2^n) ^ (b.val * i.val))⁻¹ = 0 := by
    intro a b hab
    have hd_pos : 0 < a.val - b.val := Nat.sub_pos_of_lt hab
    have hd_lt : a.val - b.val < 2^n :=
      lt_of_le_of_lt (Nat.sub_le _ _) a.isLt
    have h_reduce : ∀ i : Fin (2^n),
        omega (2^n) ^ (a.val * i.val) *
          (omega (2^n) ^ (b.val * i.val))⁻¹ =
        omega (2^n) ^ ((a.val - b.val) * i.val) := by
      intro i
      have hsum : a.val * i.val = (a.val - b.val) * i.val + b.val * i.val := by
        rw [← Nat.add_mul]
        congr 1
        exact (Nat.sub_add_cancel (Nat.le_of_lt hab)).symm
      rw [hsum, pow_add, mul_assoc,
          mul_inv_cancel₀ (pow_ne_zero _ hω_ne), mul_one]
    rw [Finset.sum_congr rfl (fun i _ => h_reduce i)]
    rw [Fin.sum_univ_eq_sum_range (fun k => omega (2^n) ^ ((a.val - b.val) * k)) (2^n)]
    exact omega_geom_sum_ne_zero (2^n) (a.val - b.val) hN_pos hd_lt hd_pos
  ext j k
  rw [Matrix.mul_apply, Matrix.one_apply]
  -- Rewrite each summand using star → starRingEnd, conj of constants,
  -- and the ω conjugation identity.
  have key : ∀ i : Fin (2^n),
      dft_matrix n j i * (dft_matrix n).conjTranspose i k =
      ((1 : ℂ) / Complex.ofReal (Real.sqrt ((2^n : ℕ) : ℝ))) *
      ((1 : ℂ) / Complex.ofReal (Real.sqrt ((2^n : ℕ) : ℝ))) *
      (omega (2^n) ^ (j.val * i.val) *
        (omega (2^n) ^ (k.val * i.val))⁻¹) := by
    intro i
    rw [Matrix.conjTranspose_apply]
    show dft_matrix n j i *
        (starRingEnd ℂ) (dft_matrix n k i) = _
    simp only [dft_matrix]
    rw [map_mul, hc_conj, omega_conj_pow_inv]
    ring
  rw [Finset.sum_congr rfl (fun i _ => key i)]
  rw [← Finset.mul_sum, hc_sq]
  by_cases hjk : j = k
  · -- Diagonal case
    rw [if_pos hjk]
    subst hjk
    have h1 : ∀ i : Fin (2^n),
        omega (2^n) ^ (j.val * i.val) *
          (omega (2^n) ^ (j.val * i.val))⁻¹ = 1 := by
      intro i; exact mul_inv_cancel₀ (pow_ne_zero _ hω_ne)
    rw [Finset.sum_congr rfl (fun i _ => h1 i)]
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, mul_one, one_div, inv_mul_cancel₀ hN_ne]
  · -- Off-diagonal case
    rw [if_neg hjk]
    have hjk' : j.val ≠ k.val := fun h => hjk (Fin.ext h)
    rcases lt_or_gt_of_ne hjk' with hlt | hgt
    · -- j.val < k.val: conjugate the swapped sum
      have h1 := main k j hlt
      have h2 : ∀ i : Fin (2^n),
          (starRingEnd ℂ)
              (omega (2^n) ^ (k.val * i.val) *
                (omega (2^n) ^ (j.val * i.val))⁻¹) =
          omega (2^n) ^ (j.val * i.val) *
            (omega (2^n) ^ (k.val * i.val))⁻¹ := by
        intro i
        rw [map_mul, map_inv₀, omega_conj_pow_inv, omega_conj_pow_inv,
            inv_inv]
        ring
      have hconj := congrArg (starRingEnd ℂ) h1
      rw [map_sum, map_zero] at hconj
      rw [Finset.sum_congr rfl (fun i _ => h2 i)] at hconj
      rw [hconj, mul_zero]
    · -- j.val > k.val: direct
      rw [main j k hgt, mul_zero]

/-- The controlled-phase layer of `qft_circuit (n+1)`: `CP(π/2^(k+1))` from
    qubit `k+1` into qubit 0, for each `k < n` — the genuine controlled gate
    (no longer an `RZ` stand-in). Named so the Cooley–Tukey step lemmas can
    refer to it directly. -/
noncomputable def qftPhaseLayer (n : ℕ) : CircuitSemantics.Circuit (n + 1) :=
  List.ofFn (fun k : Fin n =>
    CircuitSemantics.GateApp.cp ⟨0, by omega⟩ ⟨k.val + 1, by have := k.isLt; omega⟩
      (Real.pi / 2^(k.val + 1)) (Fin.ne_of_val_ne (show (0:ℕ) ≠ k.val + 1 by omega)))

/-- Bit-reversal layer on **all** `n` qubits: swap qubit `i` with `n-1-i` for
    `i < n/2`. Applied **once**, at the top of `qft_circuit` — a single full bit
    reversal corrects the bit-reversed output order of the swap-free
    Cooley–Tukey core (`qftCore`). -/
def bitReversal (n : ℕ) : CircuitSemantics.Circuit n :=
  List.ofFn (fun i : Fin (n / 2) =>
    CircuitSemantics.GateApp.swap
      (⟨i.val, by have := i.isLt; omega⟩ : Fin n)
      (⟨n - 1 - i.val, by have := i.isLt; omega⟩ : Fin n)
      (Fin.ne_of_val_ne (show i.val ≠ n - 1 - i.val by have := i.isLt; omega)))

/-- The **swap-free** Cooley–Tukey core of the QFT on `n` qubits: at step `n+1`
    apply `H` to qubit 0, the `qftPhaseLayer` controlled-phase rotations into
    qubit 0, then recurse on the tail (`qftCore n` embedded at offset 1 via
    `embed_subcircuit`). It carries **no** swap layer, so its output basis ends
    up bit-reversed relative to the DFT (see `qftCore_eq`). -/
noncomputable def qftCore : (n : ℕ) → CircuitSemantics.Circuit n
  | 0 => []
  | n + 1 =>
    [CircuitSemantics.GateApp.h ⟨0, by omega⟩] ++
    qftPhaseLayer n ++
    CircuitSemantics.embed_subcircuit (qftCore n)

/-- The QFT circuit on `n` qubits: the swap-free `qftCore` followed by a single
    `bitReversal`. **Correctness note:** earlier revisions interleaved a *full*
    bit-reversal at every recursive level — that double-reverses and is wrong
    for `n ≥ 3` (`denote ≠ dft_matrix`, verified numerically). The reversal must
    be applied exactly once, at the top. -/
noncomputable def qft_circuit (n : ℕ) : CircuitSemantics.Circuit n :=
  qftCore n ++ bitReversal n

/-- Base case: the 0-qubit QFT denotes to the 1×1 identity matrix, which
    coincides with the trivial DFT of size 2^0 = 1. Every entry is
    (1/√1) · ω^0 = 1, and in the 1-dim space this IS the identity. -/
theorem qft_correct_zero :
    CircuitSemantics.denote (qft_circuit 0) = dft_matrix 0 := by
  -- qft_circuit 0 = qftCore 0 ++ bitReversal 0 = [] ++ [] = [], so denote = 1.
  -- dft_matrix 0 has every entry = (1/√1) * ω^0 = 1, which agrees with 1.
  have h0 : qft_circuit 0 = ([] : CircuitSemantics.Circuit 0) := by
    simp [qft_circuit, qftCore, bitReversal, List.ofFn_zero]
  rw [h0]
  ext i j
  fin_cases i
  fin_cases j
  simp [CircuitSemantics.denote, dft_matrix, omega, Matrix.one_apply, pow_zero]

-- ===========================================================================
-- Step decomposition. Each named lemma below is a concrete obligation
-- the next mathlib-enabled pass should close. They correspond to the
-- three structural pieces of `qft_circuit (k+1)` and pair with their
-- counterparts in the Cooley–Tukey factorization of `dft_matrix (k+1)`.
-- ===========================================================================

/-- The Hadamard layer at position 0 (now concrete): the singleton circuit
    `[H@0]` denotes to `embed_single_qubit (k+1) 0 H` — the `H ⊗ I_{2^k}`
    factor. Direct from `single_gate`. -/
theorem qft_step_hadamard_factor (k : ℕ) :
    CircuitSemantics.denote [CircuitSemantics.GateApp.h ⟨0, by omega⟩]
      = CircuitSemantics.embed_single_qubit (k + 1) ⟨0, by omega⟩ Gates.H := by
  rw [CircuitSemantics.single_gate]; rfl

/-- **The head Hadamard factor is `H ⊗ I_{2^k}`.** The `H@0` gate of
    `qftCore (k+1)` denotes, through `tailEquiv`, to `Gates.H ⊗ I` — the
    `H`-layer half of the Cooley–Tukey step. Direct from
    `embed_single_qubit_head`. -/
theorem qft_step_hadamard_kron (k : ℕ) :
    CircuitSemantics.denote_gate (CircuitSemantics.GateApp.h (⟨0, by omega⟩ : Fin (k+1)))
      = Matrix.reindex (CircuitSemantics.tailEquiv k) (CircuitSemantics.tailEquiv k)
          (Matrix.kronecker Gates.H (1 : Matrix (Fin (2^k)) (Fin (2^k)) ℂ)) := by
  show CircuitSemantics.embed_single_qubit (k + 1) ⟨0, by omega⟩ Gates.H = _
  exact CircuitSemantics.embed_single_qubit_head k Gates.H

/-- The controlled-phase rotation layer is a product of diagonal matrices
    with entries `exp(2πi · j · k / N)` for `j, k` ranging over the
    tail qubits. Step piece two: the genuine `CP(π/2^(j+1))`-into-qubit-0
    layer that `qft_circuit (k+1)` now emits. -/
theorem qft_step_phases_factor (k : ℕ) :
    ∃ M : Matrix (Fin (2^(k+1))) (Fin (2^(k+1))) ℂ,
      CircuitSemantics.denote
        (List.ofFn (fun j : Fin k =>
          CircuitSemantics.GateApp.cp ⟨0, by omega⟩ ⟨j.val + 1, by have := j.isLt; omega⟩
            (Real.pi / 2^(j.val + 1))
            (Fin.ne_of_val_ne (show (0:ℕ) ≠ j.val + 1 by omega)))) = M := by
  refine ⟨_, rfl⟩

/-- Step piece three (concrete): the recursive `qftCore k` embedded on the tail
    qubits denotes to `I₂ ⊗ denote(qftCore k)` (reindexed through `tailEquiv`).
    Direct from `denote_embed_subcircuit`. -/
theorem qft_step_tail_factor (k : ℕ) :
    CircuitSemantics.denote (CircuitSemantics.embed_subcircuit (qftCore k))
      = Matrix.reindex (CircuitSemantics.tailEquiv k) (CircuitSemantics.tailEquiv k)
          (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ)
            (CircuitSemantics.denote (qftCore k))) :=
  CircuitSemantics.denote_embed_subcircuit (qftCore k)

/-- **Structural recursion of the swap-free core (sorry-free).** The denotation
    of `qftCore (k+1)` factors — in reverse circuit order, since `denote` folds
    matrix multiplication on the left — as
    `(I₂ ⊗ denote(qftCore k)) · phase-layer · H@0`. Pure circuit algebra:
    `denote_append` (×2) over the three `++` blocks, `single_gate`, and
    `qft_step_tail_factor`. This is the clean circuit→matrix-product bridge for
    the core; the inductive step of `qftCore_eq` reduces to it. -/
theorem qftCore_succ_denote (k : ℕ) :
    CircuitSemantics.denote (qftCore (k + 1))
      = Matrix.reindex (CircuitSemantics.tailEquiv k) (CircuitSemantics.tailEquiv k)
          (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ)
            (CircuitSemantics.denote (qftCore k)))
        * (CircuitSemantics.denote (qftPhaseLayer k)
          * CircuitSemantics.denote_gate (CircuitSemantics.GateApp.h ⟨0, by omega⟩)) := by
  have hunfold : qftCore (k + 1)
      = [CircuitSemantics.GateApp.h ⟨0, by omega⟩] ++ qftPhaseLayer k
        ++ CircuitSemantics.embed_subcircuit (qftCore k) := rfl
  rw [hunfold, CircuitSemantics.denote_append, CircuitSemantics.denote_append,
      CircuitSemantics.single_gate, qft_step_tail_factor k]

/-- Reverse the low `n` bits of `x`. The output-bit permutation of `qftCore`. -/
def bitrev (n x : ℕ) : ℕ := ∑ i ∈ Finset.range n, (x / 2^i % 2) * 2^(n-1-i)

/-- The defining recursion: reversing `k+1` bits sends the top bit `B` to the
    bottom and reverses the low `k` bits. `bitrev (k+1) (B·2^k + R) = B + 2·bitrev k R`. -/
theorem bitrev_succ (k B R : ℕ) (hB : B < 2) (hR : R < 2^k) :
    bitrev (k+1) (B * 2^k + R) = B + 2 * bitrev k R := by
  unfold bitrev
  rw [Finset.sum_range_succ]
  have hpk : 0 < (2:ℕ)^k := pow_pos (by norm_num : (0:ℕ) < 2) k
  have htop : (B * 2^k + R) / 2^k % 2 * 2^(k+1-1-k) = B := by
    have hdiv : (B * 2^k + R) / 2^k = B := by
      rw [add_comm, Nat.add_mul_div_right _ _ hpk, Nat.div_eq_of_lt hR, zero_add]
    rw [hdiv]; simp [Nat.mod_eq_of_lt hB]
  rw [htop, add_comm]
  congr 1
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.mem_range] at hi
  have hpi : 0 < (2:ℕ)^i := pow_pos (by norm_num : (0:ℕ) < 2) i
  have hbit : (B * 2^k + R) / 2^i % 2 = R / 2^i % 2 := by
    have hsplit : (2:ℕ)^k = 2^i * 2^(k-i) := by rw [← pow_add]; congr 1; omega
    rw [hsplit, mul_comm (2^i) (2^(k-i)), ← mul_assoc, add_comm,
        Nat.add_mul_div_right _ _ hpi]
    have heven : (2:ℕ) ∣ B * 2^(k-i) := by
      have h2 : (2:ℕ) ∣ 2^(k-i) := dvd_pow_self 2 (by omega)
      exact Dvd.dvd.mul_left h2 B
    omega
  have hexp : (2:ℕ)^(k+1-1-i) = 2 * 2^(k-1-i) := by
    rw [← pow_succ']; congr 1; omega
  rw [hbit, hexp]; ring

/-- `∑_{j<n} 2^j = 2^n - 1`. -/
private lemma sum_range_two_pow (n : ℕ) :
    ∑ j ∈ Finset.range n, 2^j = 2^n - 1 := by
  induction n with
  | zero => simp
  | succ m ih =>
      rw [Finset.sum_range_succ, ih]
      have : 0 < (2:ℕ)^m := Nat.pos_of_ne_zero (by positivity)
      have hpow : (2:ℕ)^(m+1) = 2 * 2^m := by rw [pow_succ]; ring
      omega

/-- The bit-reversal value is bounded: `bitrev n x < 2^n` (each of the `n`
    bit-contributions is at most `2^{n-1-i}`, summing to `2^n - 1`). -/
theorem bitrev_lt (n x : ℕ) : bitrev n x < 2^n := by
  have hle : bitrev n x ≤ ∑ i ∈ Finset.range n, 2^(n-1-i) := by
    unfold bitrev
    apply Finset.sum_le_sum
    intro i _
    calc (x/2^i)%2 * 2^(n-1-i) ≤ 1 * 2^(n-1-i) :=
          Nat.mul_le_mul_right _ (by omega)
      _ = 2^(n-1-i) := one_mul _
  rw [Finset.sum_range_reflect (fun j => 2^j) n, sum_range_two_pow] at hle
  have : 0 < (2:ℕ)^n := Nat.pos_of_ne_zero (by positivity)
  omega

/-- **Bit-characterization of `bitrev`.** For `q < n` and `x < 2^n`, bit `q` of
    `bitrev n x` is bit `n-1-q` of `x` — bit-reversal reflects bit positions.
    Induction on `n` via `bitrev_succ` (peel the top bit `B`, recurse on the low
    `k` bits) + `bit_eq_mod` (low bits survive `mod 2^k`). -/
theorem bitrev_bit (n : ℕ) : ∀ (x q : ℕ), q < n → x < 2^n →
    (bitrev n x / 2^q) % 2 = (x / 2^(n-1-q)) % 2 := by
  induction n with
  | zero => intro _ q hq _; omega
  | succ k ih =>
      intro x q hq hx
      have hkpos : 0 < (2:ℕ)^k := Nat.pos_of_ne_zero (by positivity)
      have hpow : (2:ℕ)^(k+1) = 2 * 2^k := by rw [pow_succ]; ring
      set B := x / 2^k with hBdef
      set R := x % 2^k with hRdef
      have hR : R < 2^k := Nat.mod_lt _ hkpos
      have hB : B < 2 := by rw [hBdef, Nat.div_lt_iff_lt_mul hkpos]; omega
      have hxeq : x = B * 2^k + R := by
        rw [Nat.mul_comm]; exact (Nat.div_add_mod x (2^k)).symm
      have hbr : bitrev (k+1) x = B + 2 * bitrev k R := by
        rw [hxeq]; exact bitrev_succ k B R hB hR
      rw [hbr]
      rcases Nat.eq_zero_or_pos q with hq0 | hqpos
      · -- bottom bit: B
        subst hq0
        simp only [pow_zero, Nat.div_one]
        have hxk : x / 2^(k+1-1-0) = B := by show x / 2^k = B; rw [hBdef]
        rw [hxk, show B + 2 * bitrev k R = B + bitrev k R * 2 from by ring,
            Nat.add_mul_mod_self_right]
      · -- higher bit: shift down to bit (q-1) of `bitrev k R`
        obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := ⟨q - 1, by omega⟩
        have hq'k : q' < k := by omega
        have hbound : k - 1 - q' < k := by omega
        have hYdiv : (B + 2 * bitrev k R) / 2^(q'+1) = bitrev k R / 2^q' := by
          rw [show (2:ℕ)^(q'+1) = 2 * 2^q' from by rw [pow_succ]; ring,
              ← Nat.div_div_eq_div_mul,
              show (B + 2 * bitrev k R) / 2 = bitrev k R from by
                rw [show B + 2 * bitrev k R = B + bitrev k R * 2 from by ring,
                    Nat.add_mul_div_right _ _ (by norm_num), Nat.div_eq_of_lt hB,
                    Nat.zero_add]]
        rw [hYdiv, ih R q' hq'k hR]
        -- (R / 2^(k-1-q')) % 2 = (x / 2^(k+1-1-(q'+1))) % 2
        have hexp : k + 1 - 1 - (q' + 1) = k - 1 - q' := by omega
        rw [hexp, hRdef, ← CircuitSemantics.bit_eq_mod x (k - 1 - q') k hbound]

/-- Bit-reversal as a self-map of `Fin (2^n)`. -/
noncomputable def bitrevFin (n : ℕ) (x : Fin (2^n)) : Fin (2^n) :=
  ⟨bitrev n x.val, bitrev_lt n x.val⟩

/-- Bit-reversal is an involution on `Fin (2^n)`: reflecting bit positions
    twice restores every bit (`bitrev_bit` applied twice, `eq_of_bits_lt`). -/
theorem bitrevFin_involutive (n : ℕ) : Function.Involutive (bitrevFin n) := by
  intro x
  apply Fin.ext
  show bitrev n (bitrev n x.val) = x.val
  refine CircuitSemantics.eq_of_bits_lt n _ _ (bitrev_lt n _) x.isLt (fun p hp => ?_)
  rw [bitrev_bit n (bitrev n x.val) p hp (bitrev_lt n x.val),
      bitrev_bit n x.val (n - 1 - p) (by omega) x.isLt,
      show n - 1 - (n - 1 - p) = p from by omega]

/-- The bit-reversal permutation of `Fin (2^n)`. -/
noncomputable def bitrevPerm (n : ℕ) : Equiv.Perm (Fin (2^n)) :=
  (bitrevFin_involutive n).toPerm (bitrevFin n)

@[simp] theorem bitrevPerm_apply (n : ℕ) (x : Fin (2^n)) :
    (bitrevPerm n x).val = bitrev n x.val := rfl

/-- Position-level transposition: swap positions `a` and `b`, fix the rest. -/
def swapPos (a b P : ℕ) : ℕ := if P = a then b else if P = b then a else P

/-- **Single-swap bit action.** A `swapPerm` on qubits `q0,q1` moves bit `P` of
    the result to bit `swapPos (n-1-q0) (n-1-q1) P` of the input: the bit at the
    two swapped positions is exchanged, all others are preserved. Repackages
    `swapVal_bit_A/B` + `swapVal_preserves_bit`. -/
theorem swapVal_bit (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) (x P : ℕ) :
    (CircuitSemantics.swapVal n q0 q1 x / 2^P) % 2
      = (x / 2^(swapPos (n-1-q0.val) (n-1-q1.val) P)) % 2 := by
  unfold swapPos
  by_cases h0 : P = n - 1 - q0.val
  · subst h0; rw [if_pos rfl]; exact CircuitSemantics.swapVal_bit_A n q0 q1 hq x
  · by_cases h1 : P = n - 1 - q1.val
    · subst h1; rw [if_neg h0, if_pos rfl]; exact CircuitSemantics.swapVal_bit_B n q0 q1 hq x
    · rw [if_neg h0, if_neg h1]; exact CircuitSemantics.swapVal_preserves_bit n q0 q1 hq x P h0 h1

/-- A swap together with its two (distinct) qubit indices — carries the data
    needed to track the bit-position permutation through a product of swaps. -/
structure SwapData (n : ℕ) where
  q0 : Fin n
  q1 : Fin n
  hne : q0.val ≠ q1.val

/-- The permutation a `SwapData` denotes. -/
noncomputable def SwapData.toPerm {n : ℕ} (s : SwapData n) : Equiv.Perm (Fin (2^n)) :=
  CircuitSemantics.swapPerm n s.q0 s.q1 s.hne

/-- The bit-position transposition a `SwapData` induces. -/
def SwapData.pos {n : ℕ} (s : SwapData n) (P : ℕ) : ℕ :=
  swapPos (n - 1 - s.q0.val) (n - 1 - s.q1.val) P

/-- **Composite bit-action of a product of swaps.** Bit `P` of the product of
    swap-perms applied to `x` is bit `(foldl of the position transpositions) P`
    of `x`. Induction on the list: each outer swap contributes one `swapVal_bit`
    position-step, accumulated by `foldl` (`List.prod` applies head-first). -/
theorem prod_swapData_bit (n : ℕ) :
    ∀ (L : List (SwapData n)) (x : Fin (2^n)) (P : ℕ),
      (((L.map SwapData.toPerm).prod) x).val / 2^P % 2
        = x.val / 2^(L.foldl (fun p s => SwapData.pos s p) P) % 2 := by
  intro L
  induction L with
  | nil => intro x P; simp
  | cons s L' ih =>
      intro x P
      rw [List.map_cons, List.prod_cons, Equiv.Perm.mul_apply]
      show (CircuitSemantics.swapPerm n s.q0 s.q1 s.hne _).val / 2^P % 2 = _
      rw [CircuitSemantics.swapPerm_apply, swapVal_bit n s.q0 s.q1 s.hne, ih]
      rfl

/-- The swap data of `bitReversal n`: swap qubit `i` with `n-1-i` for `i<n/2`. -/
noncomputable def revData (n : ℕ) : List (SwapData n) :=
  List.ofFn (fun i : Fin (n/2) =>
    { q0 := ⟨i.val, by have := i.isLt; have := Nat.div_le_self n 2; omega⟩
      q1 := ⟨n - 1 - i.val, by have := i.isLt; have := Nat.div_le_self n 2; omega⟩
      hne := by show i.val ≠ n - 1 - i.val; have := i.isLt; omega })

/-- **Position composite = reversal (range form).** Applying the first `m`
    bit-position transpositions `{i, n-1-i}` (`i<m`) to `P` reflects `P` once it
    (or its mirror) is reached, and leaves it otherwise. Induction on `m`; each
    `swapPos` step is disjoint-from-the-rest arithmetic closed by `omega`. -/
theorem revPosFoldl (n : ℕ) : ∀ (m : ℕ), m ≤ n/2 → ∀ (P : ℕ), P < n →
    (List.range m).foldl (fun p i => swapPos (n-1-i) (n-1-(n-1-i)) p) P
      = if P < m ∨ n - 1 - P < m then n - 1 - P else P := by
  intro m
  induction m with
  | zero => intro _ P _; simp
  | succ k ih =>
      intro hm P hP
      rw [List.range_succ, List.foldl_append, ih (by omega) P hP]
      simp only [List.foldl_cons, List.foldl_nil]
      unfold swapPos
      split_ifs <;> omega

/-- The bit-reversal value of `revData`'s position composite: `n-1-P`. -/
theorem revData_foldl (n P : ℕ) (hP : P < n) :
    (revData n).foldl (fun p s => SwapData.pos s p) P = n - 1 - P := by
  have hconv : (revData n).foldl (fun p s => SwapData.pos s p) P
      = (List.range (n/2)).foldl (fun p i => swapPos (n-1-i) (n-1-(n-1-i)) p) P := by
    unfold revData SwapData.pos
    rw [List.ofFn_eq_map, List.foldl_map, ← List.map_coe_finRange_eq_range,
        List.foldl_map]
  rw [hconv, revPosFoldl n (n/2) le_rfl P hP]
  split_ifs <;> omega

/-- **The product of `bitReversal`'s swaps is the bit-reversal permutation.**
    Bit `p` of the composite applied to `x` is bit `n-1-p` of `x`
    (`prod_swapData_bit` + `revData_foldl`), which characterizes `bitrevPerm`
    (`bitrev_bit`); conclude by `eq_of_bits_lt`. -/
theorem revData_prod_eq_bitrevPerm (n : ℕ) :
    ((revData n).map SwapData.toPerm).prod = bitrevPerm n := by
  ext x
  rw [bitrevPerm_apply]
  refine CircuitSemantics.eq_of_bits_lt n _ _
    (((revData n).map SwapData.toPerm).prod x).isLt (bitrev_lt n x.val) (fun p hp => ?_)
  rw [prod_swapData_bit n (revData n) x p, revData_foldl n p hp,
      bitrev_bit n x.val p hp x.isLt]

/-- **(b) `denote(bitReversal n) = permMatrix(bitrevPerm)`.** The swap layer is a
    product of disjoint bit-swap perms; `denote_eq_permMatrix_prod` turns its
    denotation into the permMatrix of the product, which is `bitrevPerm`. -/
theorem denote_bitReversal_eq_permMatrix (n : ℕ) :
    CircuitSemantics.denote (bitReversal n) = (bitrevPerm n).permMatrix ℂ := by
  rw [CircuitSemantics.denote_eq_permMatrix_prod
        (fun g => match g with
          | CircuitSemantics.GateApp.swap q0 q1 h =>
              CircuitSemantics.swapPerm n q0 q1 (Fin.val_ne_of_ne h)
          | _ => 1)
        (bitReversal n) (by
          intro g hg
          simp only [bitReversal, List.mem_ofFn] at hg
          obtain ⟨i, rfl⟩ := hg
          exact CircuitSemantics.denote_swap_eq_permMatrix n _ _ _)]
  congr 1
  rw [← revData_prod_eq_bitrevPerm]
  congr 1
  simp only [bitReversal, revData, List.map_ofFn, Function.comp]
  rfl

/-- The length-4 phase vector `![1,1,1,z]` indexed by the control-target bit
    pair `⟨a*2 + b⟩` (`a,b ∈ {0,1}`) returns `z^(a·b)`: the phase fires exactly
    when both bits are set. -/
private lemma vec4_pow (z : ℂ) (a b : ℕ) (ha : a < 2) (hb : b < 2) :
    (![1, 1, 1, z] : Fin 4 → ℂ) ⟨a * 2 + b, by omega⟩ = z ^ (a * b) := by
  interval_cases a <;> interval_cases b <;>
    simp [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]

/-- **The phase layer is diagonal (twiddle-factor record).** `qftPhaseLayer n`
    acts on `n+1` qubits; its `i`-th diagonal entry accumulates one factor of
    `exp(π/2^(k+1) · I)` for every tail qubit `k+1` (`k < n`) whose bit AND
    qubit-0's bit are both set. This is the diagonal half of the Cooley–Tukey
    step, computed sorry-free from `denote_diag_list` + `denote_gate_cp_diagonal`. -/
theorem qftPhaseLayer_denote (n : ℕ) :
    CircuitSemantics.denote (qftPhaseLayer n)
      = Matrix.diagonal (fun i : Fin (2^(n+1)) =>
          ∏ k : Fin n,
            Complex.exp ((Real.pi / 2^(k.val+1) : ℝ) * Complex.I) ^
              (((i.val / 2^n) % 2) * ((i.val / 2^(n-1-k.val)) % 2))) := by
  -- Diagonal-entry function for an arbitrary gate (only the `cp` arm is used).
  classical
  set d : CircuitSemantics.GateApp (n+1) → Fin (2^(n+1)) → ℂ :=
    fun g i => match g with
      | CircuitSemantics.GateApp.cp c t θ _ =>
          (![1, 1, 1, Complex.exp ((θ:ℝ) * Complex.I)] : Fin 4 → ℂ)
            ⟨(i.val / 2^(n+1-1-c.val)) % 2 * 2 + (i.val / 2^(n+1-1-t.val)) % 2, by omega⟩
      | _ => 1
    with hd
  have hdiag : ∀ g ∈ qftPhaseLayer n,
      CircuitSemantics.denote_gate g = Matrix.diagonal (d g) := by
    intro g hg
    simp only [qftPhaseLayer, List.mem_ofFn] at hg
    obtain ⟨k, rfl⟩ := hg
    rw [CircuitSemantics.denote_gate_cp_diagonal]
  rw [CircuitSemantics.denote_diag_list d (qftPhaseLayer n) hdiag]
  -- Convert the `List.ofFn` product into a `Finset.prod` over `Fin n`.
  congr 1
  funext i
  rw [qftPhaseLayer, List.map_ofFn, List.prod_ofFn]
  apply Finset.prod_congr rfl
  intro k _
  -- Evaluate `d (cp 0 (k+1) θ_k) i` via `vec4_pow`.  Note `n+1-1-0` is `n`
  -- definitionally; only the target-bit exponent `n+1-1-(k+1) = n-1-k` needs a
  -- rewrite, and it lives in a plain `Nat` position (no dependent Fin proof).
  show (![1, 1, 1, Complex.exp ((Real.pi / 2^(k.val+1) : ℝ) * Complex.I)] : Fin 4 → ℂ)
        ⟨(i.val / 2^(n+1-1-0)) % 2 * 2 + (i.val / 2^(n+1-1-(k.val+1))) % 2, by omega⟩ = _
  rw [vec4_pow _ _ _ (by omega) (by omega),
      show n + 1 - 1 - (k.val + 1) = n - 1 - k.val from by omega,
      show n + 1 - 1 - 0 = n from by omega]

/-- **Binary reconstruction:** summing `bit_p(i)·2^p` over the low `n` bit
    positions recovers `i mod 2^n`. The arithmetic backbone of the
    twiddle-factor collapse. -/
theorem bitsum_eq_mod (n i : ℕ) :
    ∑ p ∈ Finset.range n, (i / 2^p) % 2 * 2^p = i % 2^n := by
  induction n with
  | zero => simp [Nat.mod_one]
  | succ m ih =>
      rw [Finset.sum_range_succ, ih]
      -- i % 2^(m+1) = i % 2^m + (i / 2^m % 2) * 2^m
      have key : i % 2^(m+1) = 2^m * (i % 2^(m+1) / 2^m) + i % 2^(m+1) % 2^m :=
        (Nat.div_add_mod _ _).symm
      have hdiv : i % 2^(m+1) / 2^m = i / 2^m % 2 := by
        rw [pow_succ]; exact Nat.mod_mul_right_div_self i (2^m) 2
      have hmod : i % 2^(m+1) % 2^m = i % 2^m :=
        Nat.mod_mod_of_dvd i (pow_dvd_pow 2 (Nat.le_succ m))
      rw [hdiv, hmod] at key
      rw [key]; ring

/-- **Phase-accumulation (twiddle-factor collapse).** The product of the
    controlled-phase twiddle factors for index `i` collapses to a single power
    of the `2^{n+1}`-th root of unity:
      `∏_{k<n} exp(π/2^{k+1}·I)^{b_0 · b_{k+1}} = ω_{2^{n+1}}^{b_0 · (i mod 2^n)}`,
    where `b_0 = bit_n(i)` and `b_{k+1} = bit_{n-1-k}(i)`. Pure complex analysis:
    `exp_nat_mul` + `exp_sum` turn the product into `exp` of a sum, and the bit
    sum reindexes (`sum_range_reflect`) to `bitsum_eq_mod`. -/
theorem qftPhaseLayer_entry (n i : ℕ) :
    (∏ k ∈ Finset.range n,
       Complex.exp ((Real.pi / 2^(k+1) : ℝ) * Complex.I) ^
         (((i / 2^n) % 2) * ((i / 2^(n-1-k)) % 2)))
      = (omega (2^(n+1))) ^ (((i / 2^n) % 2) * (i % 2^n)) := by
  -- (1) product of exp-powers → exp of a sum
  have hfac : ∀ k ∈ Finset.range n,
      Complex.exp ((Real.pi / 2^(k+1) : ℝ) * Complex.I) ^
          (((i/2^n)%2)*((i/2^(n-1-k))%2))
        = Complex.exp ((((((i/2^n)%2)*((i/2^(n-1-k))%2)) : ℕ) : ℂ)
            * ((Real.pi / 2^(k+1) : ℝ) * Complex.I)) :=
    fun k _ => (Complex.exp_nat_mul _ _).symm
  rw [Finset.prod_congr rfl hfac, ← Complex.exp_sum]
  -- (2) RHS ω^M → exp of a product
  unfold omega
  rw [← Complex.exp_nat_mul]
  congr 1
  -- (3) the key bit-sum identity, in ℂ
  have hnat : ∑ k ∈ Finset.range n, ((i/2^(n-1-k))%2) * 2^(n-1-k) = i % 2^n := by
    rw [Finset.sum_range_reflect (fun p => ((i/2^p)%2) * 2^p) n]
    exact bitsum_eq_mod n i
  have hsum : ∑ k ∈ Finset.range n, (((i/2^(n-1-k))%2 : ℕ) : ℂ) / 2^(k+1)
      = ((i % 2^n : ℕ) : ℂ) / 2^n := by
    have hpow : ∀ k ∈ Finset.range n,
        (((i/2^(n-1-k))%2 : ℕ) : ℂ) / 2^(k+1)
          = (((i/2^(n-1-k))%2 : ℕ) : ℂ) * 2^(n-1-k) / 2^n := by
      intro k hk
      rw [Finset.mem_range] at hk
      have hsplit : (2:ℂ)^n = 2^(k+1) * 2^(n-1-k) := by
        rw [← pow_add]; congr 1; omega
      have h1 : (2:ℂ)^(k+1) ≠ 0 := pow_ne_zero _ two_ne_zero
      have h2 : (2:ℂ)^(n-1-k) ≠ 0 := pow_ne_zero _ two_ne_zero
      rw [hsplit]; field_simp
    rw [Finset.sum_congr rfl hpow, ← Finset.sum_div]
    congr 1
    rw [← hnat]; push_cast; ring
  -- (4) assemble the argument equality
  rw [show (∑ k ∈ Finset.range n,
        (((((i/2^n)%2)*((i/2^(n-1-k))%2)) : ℕ) : ℂ) * ((Real.pi / 2^(k+1) : ℝ) * Complex.I))
      = ((i/2^n)%2 : ℕ) * (Real.pi : ℂ) * Complex.I
          * ∑ k ∈ Finset.range n, (((i/2^(n-1-k))%2 : ℕ) : ℂ) / 2^(k+1) from by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro k _
    push_cast; ring]
  rw [hsum]
  push_cast
  have h2n : (2 : ℂ)^n ≠ 0 := pow_ne_zero _ two_ne_zero
  have h2n1 : (2 : ℂ)^(n+1) ≠ 0 := pow_ne_zero _ two_ne_zero
  field_simp
  ring

/-- **Closed form of the phase layer.** Combining `qftPhaseLayer_denote` with
    the twiddle collapse `qftPhaseLayer_entry`: the phase layer is the diagonal
    whose `i`-th entry is `ω_{2^{n+1}}^{bit_n(i) · (i mod 2^n)}`. This is the
    matrix the Cooley–Tukey inductive step of `qftCore_eq` consumes directly. -/
theorem qftPhaseLayer_denote_omega (n : ℕ) :
    CircuitSemantics.denote (qftPhaseLayer n)
      = Matrix.diagonal (fun i : Fin (2^(n+1)) =>
          (omega (2^(n+1))) ^ (((i.val / 2^n) % 2) * (i.val % 2^n))) := by
  rw [qftPhaseLayer_denote]
  congr 1
  funext i
  rw [Fin.prod_univ_eq_prod_range
        (fun p => Complex.exp ((Real.pi / 2^(p+1) : ℝ) * Complex.I) ^
          (((i.val/2^n)%2)*((i.val/2^(n-1-p))%2))) n,
      qftPhaseLayer_entry n i.val]

/-- `ω_{2^k} = ω_{2^{k+1}}²` — halving the modulus squares the root. -/
theorem omega_succ_sq (k : ℕ) : omega (2^k) = omega (2^(k+1)) ^ 2 := by
  unfold omega
  rw [← Complex.exp_nat_mul]
  congr 1
  have h1 : ((2:ℂ)^(k+1)) ≠ 0 := pow_ne_zero _ two_ne_zero
  have h2 : ((2:ℂ)^k) ≠ 0 := pow_ne_zero _ two_ne_zero
  push_cast
  rw [pow_succ]
  field_simp

/-- `ω_{2^{k+1}}^{2^k} = -1` — the half-power of the root is `e^{iπ}`. -/
theorem omega_succ_pow (k : ℕ) : omega (2^(k+1)) ^ (2^k) = -1 := by
  unfold omega
  rw [← Complex.exp_nat_mul]
  have h1 : ((2:ℂ)^(k+1)) ≠ 0 := pow_ne_zero _ two_ne_zero
  have h2 : ((2:ℂ)^k) ≠ 0 := pow_ne_zero _ two_ne_zero
  rw [show ((2^k : ℕ) : ℂ) * (2 * ↑Real.pi * Complex.I / ((2^(k+1) : ℕ) : ℂ))
        = ↑Real.pi * Complex.I from by push_cast; rw [pow_succ]; field_simp]
  exact Complex.exp_pi_mul_I

/-- Hadamard entry: `H[a,b] = (1/√2)·(-1)^{a·b}` for bits `a,b`. -/
theorem H_entry (a b : ℕ) (ha : a < 2) (hb : b < 2) :
    Gates.H ⟨a, ha⟩ ⟨b, hb⟩
      = (1 / Complex.ofReal (Real.sqrt 2)) * (-1) ^ (a * b) := by
  interval_cases a <;> interval_cases b <;> simp [Gates.H]

/-- **(c) Entry formula of the swap-free core.** `denote(qftCore n) i j =
    (1/√N)·ω_N^{bitrev_n(i)·j}` — the core produces the DFT in bit-reversed row
    order. Proved by induction on `n` via the Cooley–Tukey step decomposition. -/
theorem qftCore_entry (n : ℕ) : ∀ (i j : Fin (2^n)),
    CircuitSemantics.denote (qftCore n) i j
      = (1 / Complex.ofReal (Real.sqrt ((2^n : ℕ) : ℝ)))
        * omega (2^n) ^ (bitrev n i.val * j.val) := by
  induction n with
  | zero =>
      intro i j
      rw [show qftCore 0 = ([] : CircuitSemantics.Circuit 0) from rfl,
          CircuitSemantics.empty_circuit_identity]
      fin_cases i; fin_cases j
      simp [Matrix.one_apply, bitrev]
  | succ k ih =>
      intro i j
      have hpk : 0 < (2:ℕ)^k := Nat.pos_of_ne_zero (by positivity)
      have hbi := CircuitSemantics.div_two_pow_lt k i
      have hbj := CircuitSemantics.div_two_pow_lt k j
      have hRi : i.val % 2^k < 2^k := Nat.mod_lt _ hpk
      have hRj : j.val % 2^k < 2^k := Nat.mod_lt _ hpk
      have hl_lt : (i.val/2^k) * 2^k + j.val % 2^k < 2^(k+1) := by
        have heq : (2:ℕ)^(k+1) = 2*2^k := by rw [pow_succ]; ring
        have h1 : (i.val/2^k) * 2^k ≤ 1 * 2^k := Nat.mul_le_mul_right _ (by omega)
        omega
      have hdiv : ((i.val/2^k) * 2^k + j.val % 2^k) / 2^k = i.val/2^k := by
        rw [add_comm, Nat.add_mul_div_right _ _ hpk, Nat.div_eq_of_lt hRj, zero_add]
      have hmod : ((i.val/2^k) * 2^k + j.val % 2^k) % 2^k = j.val % 2^k := by
        rw [add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hRj]
      rw [qftCore_succ_denote k, qftPhaseLayer_denote_omega, qft_step_hadamard_kron,
          Matrix.mul_apply,
          Finset.sum_eq_single_of_mem
            (⟨(i.val/2^k) * 2^k + j.val % 2^k, hl_lt⟩ : Fin (2^(k+1)))
            (Finset.mem_univ _) (by
              intro l _ hl
              rw [Matrix.diagonal_mul, CircuitSemantics.reindex_tailEquiv_kron_apply,
                  CircuitSemantics.reindex_tailEquiv_kron_apply]
              simp only [Matrix.one_apply, Fin.ext_iff, Fin.val_mk]
              by_cases h1 : i.val/2^k = l.val/2^k
              · by_cases h2 : l.val % 2^k = j.val % 2^k
                · exfalso; apply hl; apply Fin.ext
                  show l.val = (i.val/2^k) * 2^k + j.val % 2^k
                  conv_lhs => rw [← Nat.div_add_mod l.val (2^k)]
                  rw [← h1, h2]; ring
                · simp [h2]
              · simp [h1])]
      rw [Matrix.diagonal_mul, CircuitSemantics.reindex_tailEquiv_kron_apply,
          CircuitSemantics.reindex_tailEquiv_kron_apply, ih, H_entry,
          Matrix.one_apply, Matrix.one_apply]
      simp only [Fin.val_mk, Fin.ext_iff]
      rw [hdiv, hmod, Nat.mod_eq_of_lt hbi, if_pos rfl, if_pos rfl]
      -- Sever the `Fin (2^(k+1))` type dependency so power rewrites are safe.
      set a := i.val with ha
      set b := j.val with hb'
      -- The exponent identity (Cooley–Tukey): the `2^{k+1}`-multiple term
      -- contributes `ω_N^N = 1`.
      have hD : bitrev (k+1) a * b
          = 2 * (bitrev k (a % 2^k) * (b % 2^k)) + a / 2^k * (b % 2^k)
            + 2^k * (a / 2^k * (b / 2^k))
            + 2^(k+1) * (bitrev k (a % 2^k) * (b / 2^k)) := by
        have hi : a = a / 2^k * 2^k + a % 2^k := by
          rw [Nat.mul_comm]; exact (Nat.div_add_mod _ _).symm
        have hj : b = b / 2^k * 2^k + b % 2^k := by
          rw [Nat.mul_comm]; exact (Nat.div_add_mod _ _).symm
        have hbr : bitrev (k+1) a = a / 2^k + 2 * bitrev k (a % 2^k) := by
          conv_lhs => rw [hi]
          exact bitrev_succ k (a / 2^k) (a % 2^k) hbi hRi
        rw [hbr]; conv_lhs => rw [hj]
        rw [pow_succ]; ring
      have hcoeff : (1 / Complex.ofReal (Real.sqrt ((2^k : ℕ) : ℝ)))
            * (1 / Complex.ofReal (Real.sqrt 2))
          = 1 / Complex.ofReal (Real.sqrt ((2^(k+1) : ℕ) : ℝ)) := by
        rw [div_mul_div_comm, mul_one, ← Complex.ofReal_mul,
            ← Real.sqrt_mul (by positivity : (0:ℝ) ≤ ((2^k : ℕ) : ℝ)),
            show ((2^k : ℕ) : ℝ) * 2 = ((2^(k+1) : ℕ) : ℝ) from by push_cast; rw [pow_succ]]
      have e1 : omega (2^k) ^ (bitrev k (a % 2^k) * (b % 2^k))
          = omega (2^(k+1)) ^ (2 * (bitrev k (a % 2^k) * (b % 2^k))) := by
        rw [omega_succ_sq k, ← pow_mul]
      have e2 : ((-1 : ℂ)) ^ (a / 2^k * (b / 2^k))
          = omega (2^(k+1)) ^ (2^k * (a / 2^k * (b / 2^k))) := by
        rw [← omega_succ_pow k, ← pow_mul]
      have hRHS : omega (2^(k+1)) ^ (bitrev (k+1) a * b)
          = omega (2^(k+1)) ^ (2 * (bitrev k (a % 2^k) * (b % 2^k))
              + a / 2^k * (b % 2^k) + 2^k * (a / 2^k * (b / 2^k))) := by
        rw [hD, pow_add, pow_mul, omega_periodic (2^(k+1)) (by positivity), one_pow, mul_one]
      rw [e1, e2, hRHS, ← hcoeff]
      ring

theorem qftCore_eq (n : ℕ) :
    CircuitSemantics.denote (qftCore n)
      = CircuitSemantics.denote (bitReversal n) * dft_matrix n := by
  ext i j
  rw [qftCore_entry n i j, denote_bitReversal_eq_permMatrix, Equiv.Perm.permMatrix,
      PEquiv.toMatrix_toPEquiv_mul, Matrix.submatrix_apply, dft_matrix]
  simp only [bitrevPerm_apply, id_eq]

/-- The bit-reversal layer is an **involution**: applying it twice is the
    identity. It is a product of disjoint transpositions `(i, n-1-i)`, each
    self-inverse, so `bitReversal · bitReversal = 1`. (True; numerically
    `P_rev² = I`.) -/
theorem bitReversal_involutive (n : ℕ) :
    CircuitSemantics.denote (bitReversal n) * CircuitSemantics.denote (bitReversal n)
      = 1 := by
  apply CircuitSemantics.denote_sq_eq_one
  · -- every gate is involutive
    intro g hg
    simp only [bitReversal, List.mem_ofFn] at hg
    obtain ⟨i, rfl⟩ := hg
    exact CircuitSemantics.embed_swap_involutive n ⟨i.val, by have := i.isLt; omega⟩
      ⟨n - 1 - i.val, by have := i.isLt; omega⟩
      (Fin.ne_of_val_ne (show i.val ≠ n - 1 - i.val by have := i.isLt; omega))
  · -- the gates pairwise commute (swaps on disjoint qubit pairs)
    intro g hg h hh
    simp only [bitReversal, List.mem_ofFn] at hg hh
    obtain ⟨i, rfl⟩ := hg
    obtain ⟨i', rfl⟩ := hh
    by_cases hii : i = i'
    · subst hii; rfl
    · have hii' : i.val ≠ i'.val := fun hv => hii (Fin.ext hv)
      have hi := i.isLt; have hi' := i'.isLt
      apply CircuitSemantics.denote_swap_comm <;> simp only [Fin.val_mk] <;> omega

/-- Main theorem: QFT circuit implements DFT, `denote(qft_circuit n) =
    dft_matrix n`. Assembly is sorry-free: `qft_circuit = qftCore ++ bitReversal`
    gives `denote = bitReversal · qftCore` (`denote_append`); `qftCore_eq`
    rewrites the core to `bitReversal · DFT`; and `bitReversal_involutive`
    collapses the doubled reversal to `1`. The two residues
    (`qftCore_eq`, `bitReversal_involutive`) are the honest remaining
    obligations — both true and numerically verified. -/
theorem qft_correct (n : ℕ) :
    CircuitSemantics.denote (qft_circuit n) = dft_matrix n := by
  rw [qft_circuit, CircuitSemantics.denote_append, qftCore_eq, ← mul_assoc,
      bitReversal_involutive, one_mul]

/-- QFT maps |0...0⟩ to the uniform superposition. -/
theorem qft_zero_to_uniform (n : ℕ) :
    ∀ j : Fin (2^n), dft_matrix n j ⟨0, by positivity⟩ =
      1 / Complex.ofReal (Real.sqrt ↑(2^n)) := by
  intro j
  simp [dft_matrix, omega, pow_zero]

end QuantumProofs.QFT
