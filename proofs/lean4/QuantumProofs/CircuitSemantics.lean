/-
  Circuit semantics: denotational mapping from circuit AST to unitary matrices.

  A circuit is interpreted as a sequence of gate applications,
  each of which is a unitary matrix acting on the full state space.

  The semantics maps: Circuit → Unitary(2^n, 2^n)
  by composing individual gate unitaries with appropriate tensor products
  to embed single/two-qubit gates into the full space.
-/

import QuantumProofs.Basic
import QuantumProofs.Gates
import Mathlib.LinearAlgebra.Matrix.Permutation
import Mathlib.Data.Matrix.PEquiv
import Mathlib.Tactic.IntervalCases

namespace QuantumProofs.CircuitSemantics

open Matrix QuantumProofs.Gates

/-- A gate application: gate type + target qubit indices. -/
inductive GateApp (n : ℕ) where
  | h (q : Fin n) : GateApp n
  | x (q : Fin n) : GateApp n
  | y (q : Fin n) : GateApp n
  | z (q : Fin n) : GateApp n
  | s (q : Fin n) : GateApp n
  | t (q : Fin n) : GateApp n
  | rz (q : Fin n) (θ : ℝ) : GateApp n
  | ry (q : Fin n) (θ : ℝ) : GateApp n
  | cx (ctrl tgt : Fin n) (h : ctrl ≠ tgt) : GateApp n
  | cz (q0 q1 : Fin n) (h : q0 ≠ q1) : GateApp n
  | cp (ctrl tgt : Fin n) (θ : ℝ) (h : ctrl ≠ tgt) : GateApp n
  | swap (q0 q1 : Fin n) (h : q0 ≠ q1) : GateApp n
  | ccz (q0 q1 q2 : Fin n) (h01 : q0 ≠ q1) (h02 : q0 ≠ q2) (h12 : q1 ≠ q2) : GateApp n

/-- A circuit is a list of gate applications on n qubits. -/
abbrev Circuit (n : ℕ) := List (GateApp n)

/-- The local unitary matrix for a single-qubit gate application (2×2). -/
noncomputable def local_unitary_1q {n : ℕ} : GateApp n → Matrix (Fin 2) (Fin 2) ℂ
  | .h _ => Gates.H
  | .x _ => Gates.X
  | .y _ => Gates.Y
  | .z _ => Gates.Z
  | .s _ => Gates.S
  | .t _ => Gates.T
  | .rz _ θ => Gates.RZ θ
  | .ry _ θ => Gates.RY θ
  | .cx _ _ _ => 1  -- unused; 2-qubit gates go through local_unitary_2q
  | .cz _ _ _ => 1  -- unused
  | .cp _ _ _ _ => 1  -- unused
  | .swap _ _ _ => 1  -- unused
  | .ccz _ _ _ _ _ _ => 1  -- unused; 3-qubit gate goes through embed_three_qubit

/-- CZ as a 4×4 matrix (ctrl first, target second). -/
def CZ4 : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 0, 0, 0;
     0, 1, 0, 0;
     0, 0, 1, 0;
     0, 0, 0, -1]

/-- Controlled-phase `CP(θ)` as a 4×4 matrix: applies the phase `e^{iθ}`
    to the `|11⟩` component, identity elsewhere. Diagonal (hence symmetric
    in its two qubits). The genuine controlled-phase the QFT needs (a plain
    single-qubit `RZ` does *not* implement this). -/
noncomputable def CPhase4 (θ : ℝ) : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 0, 0, 0;
     0, 1, 0, 0;
     0, 0, 1, 0;
     0, 0, 0, Complex.exp (θ * Complex.I)]

/-- SWAP as a 4×4 matrix: exchanges the two qubits. Used for the QFT's
    final bit-reversal layer. -/
def SWAP4 : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 0, 0, 0;
     0, 0, 1, 0;
     0, 1, 0, 0;
     0, 0, 0, 1]

/-- The local unitary matrix for a two-qubit gate application (4×4). -/
noncomputable def local_unitary_2q {n : ℕ} : GateApp n → Matrix (Fin 4) (Fin 4) ℂ
  | .cx _ _ _ => Gates.CNOT
  | .cz _ _ _ => CZ4
  | .cp _ _ θ _ => CPhase4 θ
  | .swap _ _ _ => SWAP4
  | _ => 1

/-- Embed a single-qubit gate on qubit q into the full n-qubit Hilbert space.
    U_full = I ⊗ ... ⊗ U ⊗ ... ⊗ I
    where U is at position q. -/
noncomputable def embed_single_qubit (n : ℕ) (q : Fin n) (U : Matrix (Fin 2) (Fin 2) ℂ)
    : Matrix (Fin (2^n)) (Fin (2^n)) ℂ :=
  fun i j =>
    -- i and j agree on all qubits except q
    -- The matrix element is U[i_q, j_q] if all other bits match, else 0
    let i_q := (i.val / 2^(n - 1 - q.val)) % 2
    let j_q := (j.val / 2^(n - 1 - q.val)) % 2
    let i_rest := i.val - i_q * 2^(n - 1 - q.val)
    let j_rest := j.val - j_q * 2^(n - 1 - q.val)
    if i_rest = j_rest then
      U ⟨i_q, by omega⟩ ⟨j_q, by omega⟩
    else
      0

/-- Embed a two-qubit gate on qubits (q0, q1) into the full n-qubit space.
    The matrix element at (i, j) is U[(i_q0, i_q1), (j_q0, j_q1)] when every
    "other" bit of i and j agrees, else 0. Follows the same bit-projection
    pattern as `embed_single_qubit`. -/
noncomputable def embed_two_qubit (n : ℕ) (q0 q1 : Fin n)
    (U : Matrix (Fin 4) (Fin 4) ℂ) : Matrix (Fin (2^n)) (Fin (2^n)) ℂ :=
  fun i j =>
    let i_q0 := (i.val / 2^(n - 1 - q0.val)) % 2
    let i_q1 := (i.val / 2^(n - 1 - q1.val)) % 2
    let j_q0 := (j.val / 2^(n - 1 - q0.val)) % 2
    let j_q1 := (j.val / 2^(n - 1 - q1.val)) % 2
    let mask0 := 2^(n - 1 - q0.val)
    let mask1 := 2^(n - 1 - q1.val)
    let i_rest := i.val - i_q0 * mask0 - i_q1 * mask1
    let j_rest := j.val - j_q0 * mask0 - j_q1 * mask1
    if i_rest = j_rest then
      U ⟨i_q0 * 2 + i_q1, by omega⟩ ⟨j_q0 * 2 + j_q1, by omega⟩
    else
      0

/-- `CCZ` as an 8×8 matrix: phases the `|111⟩` component by `-1`, identity
    elsewhere. Diagonal (hence symmetric in its three qubits). The genuine
    controlled-controlled-Z the Grover oracle/diffusion need. -/
def CCZ8 : Matrix (Fin 8) (Fin 8) ℂ :=
  Matrix.diagonal ![1, 1, 1, 1, 1, 1, 1, -1]

/-- Embed a three-qubit gate on qubits `(q0, q1, q2)` into the full `n`-qubit
    space. The `(i, j)` entry is `U[(i_q0,i_q1,i_q2), (j_q0,j_q1,j_q2)]` when every
    "other" bit of `i` and `j` agrees, else `0` — the three-qubit analog of
    `embed_two_qubit`, with the gate index packed as `q0·4 + q1·2 + q2`. -/
noncomputable def embed_three_qubit (n : ℕ) (q0 q1 q2 : Fin n)
    (U : Matrix (Fin 8) (Fin 8) ℂ) : Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ :=
  fun i j =>
    let i_q0 := (i.val / 2 ^ (n - 1 - q0.val)) % 2
    let i_q1 := (i.val / 2 ^ (n - 1 - q1.val)) % 2
    let i_q2 := (i.val / 2 ^ (n - 1 - q2.val)) % 2
    let j_q0 := (j.val / 2 ^ (n - 1 - q0.val)) % 2
    let j_q1 := (j.val / 2 ^ (n - 1 - q1.val)) % 2
    let j_q2 := (j.val / 2 ^ (n - 1 - q2.val)) % 2
    let mask0 := 2 ^ (n - 1 - q0.val)
    let mask1 := 2 ^ (n - 1 - q1.val)
    let mask2 := 2 ^ (n - 1 - q2.val)
    let i_rest := i.val - i_q0 * mask0 - i_q1 * mask1 - i_q2 * mask2
    let j_rest := j.val - j_q0 * mask0 - j_q1 * mask1 - j_q2 * mask2
    if i_rest = j_rest then
      U ⟨i_q0 * 4 + i_q1 * 2 + i_q2, by omega⟩ ⟨j_q0 * 4 + j_q1 * 2 + j_q2, by omega⟩
    else
      0

/-- Entrywise form of `embed_three_qubit`, named for rewriting. -/
theorem embed_three_qubit_apply (n : ℕ) (q0 q1 q2 : Fin n)
    (U : Matrix (Fin 8) (Fin 8) ℂ) (i j : Fin (2 ^ n)) :
    embed_three_qubit n q0 q1 q2 U i j =
      if i.val - (i.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val)
              - (i.val / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)
              - (i.val / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val)
         = j.val - (j.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val)
              - (j.val / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)
              - (j.val / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val)
      then U ⟨(i.val / 2 ^ (n - 1 - q0.val)) % 2 * 4
              + (i.val / 2 ^ (n - 1 - q1.val)) % 2 * 2
              + (i.val / 2 ^ (n - 1 - q2.val)) % 2, by omega⟩
             ⟨(j.val / 2 ^ (n - 1 - q0.val)) % 2 * 4
              + (j.val / 2 ^ (n - 1 - q1.val)) % 2 * 2
              + (j.val / 2 ^ (n - 1 - q2.val)) % 2, by omega⟩
      else 0 := rfl

/-- `CCZ8` is Hermitian (real diagonal). -/
theorem CCZ8_conjTranspose : (CCZ8)ᴴ = CCZ8 := by
  rw [CCZ8, Matrix.diagonal_conjTranspose]
  congr 1
  funext i
  fin_cases i <;> simp

/-- Each diagonal entry of `CCZ8` has unit modulus (`star v * v = 1`). The
    per-entry obligation `diagonal_mem_unitary` needs for the embedded `CCZ8`. -/
theorem CCZ8_diag_unit (m : Fin 8) :
    star ((![1, 1, 1, 1, 1, 1, 1, -1] : Fin 8 → ℂ) m)
      * (![1, 1, 1, 1, 1, 1, 1, -1] : Fin 8 → ℂ) m = 1 := by
  fin_cases m <;> norm_num

/-- Denotation of a single gate application as a full-space unitary. -/
noncomputable def denote_gate {n : ℕ} (g : GateApp n) : Matrix (Fin (2^n)) (Fin (2^n)) ℂ :=
  match g with
  | .h q | .x q | .y q | .z q | .s q | .t q =>
      embed_single_qubit n q (local_unitary_1q g)
  | .rz q _ | .ry q _ =>
      embed_single_qubit n q (local_unitary_1q g)
  | .cx c t _ => embed_two_qubit n c t Gates.CNOT
  | .cz q0 q1 _ => embed_two_qubit n q0 q1 CZ4
  | .cp c t θ _ => embed_two_qubit n c t (CPhase4 θ)
  | .swap q0 q1 _ => embed_two_qubit n q0 q1 SWAP4
  | .ccz q0 q1 q2 _ _ _ => embed_three_qubit n q0 q1 q2 CCZ8

/-- Denotation of a full circuit: compose all gate unitaries. -/
noncomputable def denote {n : ℕ} (c : Circuit n) : Matrix (Fin (2^n)) (Fin (2^n)) ℂ :=
  c.foldl (fun acc g => denote_gate g * acc) 1

-- ===========================================================================
-- Sub-circuit embedding — used by `QFT.lean` to state the recursive
-- step of the Cooley–Tukey factorization.
-- ===========================================================================

/-- Shift every qubit index of a gate application by +1, mapping an
    `n`-qubit circuit to an `(n+1)`-qubit circuit that acts on qubits
    `1 .. n` (leaving qubit 0 untouched). -/
def shift_gate {n : ℕ} (g : GateApp n) : GateApp (n + 1) :=
  match g with
  | .h q   => .h ⟨q.val + 1, by omega⟩
  | .x q   => .x ⟨q.val + 1, by omega⟩
  | .y q   => .y ⟨q.val + 1, by omega⟩
  | .z q   => .z ⟨q.val + 1, by omega⟩
  | .s q   => .s ⟨q.val + 1, by omega⟩
  | .t q   => .t ⟨q.val + 1, by omega⟩
  | .rz q θ => .rz ⟨q.val + 1, by omega⟩ θ
  | .ry q θ => .ry ⟨q.val + 1, by omega⟩ θ
  | .cx c t h => .cx ⟨c.val + 1, by omega⟩ ⟨t.val + 1, by omega⟩
      (Fin.ne_of_val_ne (show c.val + 1 ≠ t.val + 1 by have := Fin.val_ne_of_ne h; omega))
  | .cz a b h => .cz ⟨a.val + 1, by omega⟩ ⟨b.val + 1, by omega⟩
      (Fin.ne_of_val_ne (show a.val + 1 ≠ b.val + 1 by have := Fin.val_ne_of_ne h; omega))
  | .cp c t θ h => .cp ⟨c.val + 1, by omega⟩ ⟨t.val + 1, by omega⟩ θ
      (Fin.ne_of_val_ne (show c.val + 1 ≠ t.val + 1 by have := Fin.val_ne_of_ne h; omega))
  | .swap a b h => .swap ⟨a.val + 1, by omega⟩ ⟨b.val + 1, by omega⟩
      (Fin.ne_of_val_ne (show a.val + 1 ≠ b.val + 1 by have := Fin.val_ne_of_ne h; omega))
  | .ccz a b c h01 h02 h12 => .ccz ⟨a.val + 1, by omega⟩ ⟨b.val + 1, by omega⟩
      ⟨c.val + 1, by omega⟩
      (Fin.ne_of_val_ne (show a.val + 1 ≠ b.val + 1 by have := Fin.val_ne_of_ne h01; omega))
      (Fin.ne_of_val_ne (show a.val + 1 ≠ c.val + 1 by have := Fin.val_ne_of_ne h02; omega))
      (Fin.ne_of_val_ne (show b.val + 1 ≠ c.val + 1 by have := Fin.val_ne_of_ne h12; omega))

/-- Embed an `n`-qubit sub-circuit into an `(n+1)`-qubit circuit on the
    "tail" qubits `1 .. n`, leaving qubit 0 alone. Used by
    `qft_circuit (n+1)` to reuse `qft_circuit n` on the remaining wires.
-/
def embed_subcircuit {n : ℕ} (c : Circuit n) : Circuit (n + 1) :=
  c.map shift_gate

-- ===========================================================================
-- Key theorems (composition + folding)
-- ===========================================================================

/-- Empty circuit is the identity. -/
theorem empty_circuit_identity (n : ℕ) :
    denote ([] : Circuit n) = 1 := by
  simp [denote, List.foldl]

/-- Single gate circuit equals the gate's denotation. -/
theorem single_gate (n : ℕ) (g : GateApp n) :
    denote [g] = denote_gate g := by
  simp [denote, List.foldl, mul_one]

/-- Folding with a non-identity accumulator is the same as folding from 1 and
    then right-multiplying by the accumulator. This is the workhorse lemma
    behind `denote_append`: every gate is composed on the left, so a starting
    accumulator `A` slides out to the right untouched. -/
private lemma foldl_mul_acc {n : ℕ} (c : Circuit n)
    (A : Matrix (Fin (2^n)) (Fin (2^n)) ℂ) :
    c.foldl (fun acc g => denote_gate g * acc) A = denote c * A := by
  induction c generalizing A with
  | nil => simp [denote]
  | cons g rest ih =>
      simp only [List.foldl]
      rw [ih (denote_gate g * A)]
      show denote rest * (denote_gate g * A) = denote (g :: rest) * A
      rw [show denote (g :: rest) = denote rest * denote_gate g from by
        show List.foldl (fun acc g => denote_gate g * acc) 1 (g :: rest)
             = denote rest * denote_gate g
        rw [List.foldl_cons, ih (denote_gate g * 1), mul_one]]
      rw [mul_assoc]

/-- `denote (g :: c) = denote c * denote_gate g`. Concrete cons-form of
    `denote_append` — the head gate appears as a right factor because
    `denote` folds matrix multiplication on the left. -/
theorem denote_cons {n : ℕ} (g : GateApp n) (c : Circuit n) :
    denote (g :: c) = denote c * denote_gate g := by
  show List.foldl (fun acc g => denote_gate g * acc) 1 (g :: c)
       = denote c * denote_gate g
  rw [List.foldl_cons, foldl_mul_acc c (denote_gate g * 1), mul_one]

/-- Circuit composition is matrix multiplication. -/
theorem denote_append {n : ℕ} (c1 c2 : Circuit n) :
    denote (c1 ++ c2) = denote c2 * denote c1 := by
  show (c1 ++ c2).foldl _ 1 = _
  rw [List.foldl_append]
  exact foldl_mul_acc c2 (denote c1)

-- ===========================================================================
-- `embed_subcircuit` Kronecker structure
-- ===========================================================================

/-- The reindexing equivalence used to view `Fin 2 × Fin (2^n)` as
    `Fin (2^(n+1))`: pair `(b, r)` lands at `b · 2^n + r`. Bundled here
    so the Kronecker statements below stay readable. -/
noncomputable def tailEquiv (n : ℕ) :
    Fin 2 × Fin (2^n) ≃ Fin (2^(n+1)) :=
  finProdFinEquiv.trans (finCongr (by ring))

/-- Trivial cons-decomposition for `embed_subcircuit`. -/
theorem embed_subcircuit_nil {n : ℕ} :
    embed_subcircuit ([] : Circuit n) = ([] : Circuit (n+1)) := rfl

theorem embed_subcircuit_cons {n : ℕ} (g : GateApp n) (c : Circuit n) :
    embed_subcircuit (g :: c) = shift_gate g :: embed_subcircuit c := rfl

-- Bit-arithmetic helpers for the gate-shift Kronecker identity.

/-- `2^n = 2^(n - 1 - q) · 2^(q + 1)` for any `q < n`. The exponent
    reassembly that drives the shift identity. -/
private lemma pow_split (n : ℕ) (q : Fin n) :
    (2 : ℕ) ^ n = 2 ^ (n - 1 - q.val) * 2 ^ (q.val + 1) := by
  rw [← pow_add]
  congr 1
  have := q.isLt
  omega

/-- For any `i : Fin (2^(n+1))`, splitting `i.val = b · 2^n + r` with
    `b < 2`, `r < 2^n`, the bit at qubit-`q+1` of `i` (in the (n+1)-qubit
    layout) equals the bit at qubit-`q` of `r` (in the n-qubit layout):
    `(i.val / M) % 2 = (r / M) % 2` where `M = 2^(n-1-q)`. -/
private lemma shifted_qubit_bit (n : ℕ) (q : Fin n) (i : Fin (2 ^ (n + 1))) :
    (i.val / 2 ^ (n - 1 - q.val)) % 2 =
      ((i.val % 2 ^ n) / 2 ^ (n - 1 - q.val)) % 2 := by
  have hM_pos : 0 < 2 ^ (n - 1 - q.val) := Nat.pos_of_ne_zero (by positivity)
  have hN_split : (2 : ℕ) ^ n = 2 ^ (n - 1 - q.val) * 2 ^ (q.val + 1) := pow_split n q
  -- Step 1: replace 2^n by M · 2^(q+1) in i.val % 2^n.
  -- Use Nat.add_mul_div_left: (a + b·c)/c = a/c + b.
  have hdiv : i.val / 2 ^ (n - 1 - q.val) =
      (i.val / 2 ^ n) * 2 ^ (q.val + 1) + (i.val % 2 ^ n) / 2 ^ (n - 1 - q.val) := by
    -- i.val = (i.val/2^n)·2^n + i.val % 2^n  =  (i.val/2^n)·M·2^(q+1) + i.val%2^n
    --       = i.val%2^n + ((i.val/2^n)·2^(q+1))·M
    -- divide by M.
    have hi_dec : i.val =
        i.val % 2 ^ n + 2 ^ (n - 1 - q.val) * ((i.val / 2 ^ n) * 2 ^ (q.val + 1)) := by
      have hbase : i.val = 2 ^ n * (i.val / 2 ^ n) + i.val % 2 ^ n :=
        (Nat.div_add_mod i.val (2^n)).symm
      have hmul : 2 ^ n * (i.val / 2 ^ n) =
          2 ^ (n - 1 - q.val) * ((i.val / 2 ^ n) * 2 ^ (q.val + 1)) := by
        rw [hN_split]; ring
      omega
    conv_lhs => rw [hi_dec]
    rw [Nat.add_mul_div_left _ _ hM_pos]
    ring
  -- Mod 2: 2^(q+1) is even.
  rw [hdiv]
  have hpow : 2 ^ (q.val + 1) = 2 * 2 ^ q.val := by rw [pow_succ]; ring
  rw [hpow]
  rw [show (i.val / 2^n) * (2 * 2 ^ q.val) =
          ((i.val / 2^n) * 2 ^ q.val) * 2 from by ring]
  omega

/-- Companion to `shifted_qubit_bit`: the `i_rest` value also decomposes
    as `b · 2^n + r_rest` where `r_rest` is the n-qubit-space rest. -/
private lemma shifted_qubit_rest (n : ℕ) (q : Fin n) (i : Fin (2 ^ (n + 1))) :
    i.val - ((i.val / 2 ^ (n - 1 - q.val)) % 2) * 2 ^ (n - 1 - q.val) =
      (i.val / 2 ^ n) * 2 ^ n +
        (i.val % 2 ^ n -
          ((i.val % 2 ^ n) / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)) := by
  have hM_pos : 0 < 2 ^ (n - 1 - q.val) := Nat.pos_of_ne_zero (by positivity)
  have hbit := shifted_qubit_bit n q i
  -- i.val = (i.val / 2^n) * 2^n + i.val % 2^n
  have hi_dec : i.val = (i.val / 2^n) * 2^n + i.val % 2^n :=
    (Nat.div_add_mod' i.val (2^n)).symm
  -- Bound: ((i.val % 2^n) / M) % 2 * M ≤ i.val % 2^n
  have h_rest_le :
      ((i.val % 2^n) / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val) ≤ i.val % 2^n := by
    calc ((i.val % 2^n) / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
        ≤ ((i.val % 2^n) / 2 ^ (n - 1 - q.val)) * 2 ^ (n - 1 - q.val) :=
          Nat.mul_le_mul_right _ (Nat.mod_le _ _)
      _ ≤ i.val % 2^n := Nat.div_mul_le_self _ _
  rw [hbit]
  calc i.val - ((i.val % 2 ^ n) / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
      = (i.val / 2 ^ n) * 2 ^ n + i.val % 2 ^ n
          - ((i.val % 2 ^ n) / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val) := by
        rw [← hi_dec]
    _ = (i.val / 2 ^ n) * 2 ^ n
          + (i.val % 2 ^ n
              - ((i.val % 2 ^ n) / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)) :=
        Nat.add_sub_assoc h_rest_le _

-- `(tailEquiv n).symm` projections in terms of div/mod by `2^n`.

private lemma tailEquiv_symm_fst_val (n : ℕ) (i : Fin (2 ^ (n + 1))) :
    ((tailEquiv n).symm i).1.val = i.val / 2 ^ n := rfl

private lemma tailEquiv_symm_snd_val (n : ℕ) (i : Fin (2 ^ (n + 1))) :
    ((tailEquiv n).symm i).2.val = i.val % 2 ^ n := rfl

/-- Top-bit equality: two `(n+1)`-qubit indices share the same first
    `tailEquiv` component iff their high bits agree. This is the
    `1[·,·]` side of the Kronecker product. -/
private lemma tailEquiv_symm_fst_eq_iff (n : ℕ) (i j : Fin (2 ^ (n + 1))) :
    ((tailEquiv n).symm i).1 = ((tailEquiv n).symm j).1 ↔
      i.val / 2 ^ n = j.val / 2 ^ n := by
  constructor
  · intro h
    have := congrArg Fin.val h
    simpa [tailEquiv_symm_fst_val] using this
  · intro h
    apply Fin.ext
    simpa [tailEquiv_symm_fst_val] using h

/-- Embedding a single-qubit gate at qubit `q+1` on `(n+1)` qubits is
    `I₂ ⊗ (embed at qubit q on n qubits)`, modulo the `tailEquiv`
    reindexing. Direct entry-wise consequence of `shifted_qubit_bit`
    and `shifted_qubit_rest`. -/
private lemma embed_single_qubit_shift (n : ℕ) (q : Fin n)
    (U : Matrix (Fin 2) (Fin 2) ℂ) :
    embed_single_qubit (n + 1) ⟨q.val + 1, by have := q.isLt; omega⟩ U =
      Matrix.reindex (tailEquiv n) (tailEquiv n)
        (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ)
          (embed_single_qubit n q U)) := by
  ext i j
  have hN_pos : 0 < (2 : ℕ) ^ n := Nat.pos_of_ne_zero (by positivity)
  -- Shared M = 2^(n-1-q.val); the shifted exponent collapses to this.
  have hExp : (n + 1) - 1 - (q.val + 1) = n - 1 - q.val := by omega
  -- Bit/rest facts from the helpers.
  have hi_bit := shifted_qubit_bit n q i
  have hj_bit := shifted_qubit_bit n q j
  have hi_rest := shifted_qubit_rest n q i
  have hj_rest := shifted_qubit_rest n q j
  -- Unfold RHS: reindex → submatrix → kronecker entry.
  simp only [Matrix.reindex_apply, Matrix.submatrix_apply, Matrix.kronecker,
             Matrix.kroneckerMap_apply]
  -- Unfold embed_single_qubit on both sides; normalize the shifted exponent.
  unfold embed_single_qubit
  simp only [hExp, tailEquiv_symm_snd_val]
  -- Now rewrite LHS rests in terms of iR := i.val % 2^n, etc.
  rw [hi_rest, hj_rest]
  -- Two cases on top-bit equality.
  by_cases hTop : i.val / 2 ^ n = j.val / 2 ^ n
  · -- Same top bit ⇒ kronecker indicator = 1.
    have hFst : ((tailEquiv n).symm i).1 = ((tailEquiv n).symm j).1 :=
      (tailEquiv_symm_fst_eq_iff n i j).mpr hTop
    rw [hFst, Matrix.one_apply_eq, one_mul]
    -- Both if-conditions are equivalent; both if-branches are equal (via hi_bit/hj_bit).
    congr 1
    · -- Condition equality: iT*2^n + A = iT*2^n + B ↔ A = B.
      rw [hTop]
      apply propext
      exact ⟨fun h => Nat.add_left_cancel h, fun h => by rw [h]⟩
    · -- Then-branch (U entry) equality.
      apply congr_arg₂
      · apply Fin.ext; simpa using hi_bit
      · apply Fin.ext; simpa using hj_bit
  · -- Different top bits ⇒ kronecker indicator = 0.
    have hFst : ((tailEquiv n).symm i).1 ≠ ((tailEquiv n).symm j).1 := by
      intro h
      exact hTop ((tailEquiv_symm_fst_eq_iff n i j).mp h)
    rw [Matrix.one_apply_ne hFst, zero_mul]
    -- Show LHS = 0 by disproving the if-condition.
    have hiR_lt : i.val % 2 ^ n < 2 ^ n := Nat.mod_lt _ hN_pos
    have hjR_lt : j.val % 2 ^ n < 2 ^ n := Nat.mod_lt _ hN_pos
    set M := (2 : ℕ) ^ (n - 1 - q.val)
    have hBnd_i : (i.val % 2 ^ n / M) % 2 * M ≤ i.val % 2 ^ n := by
      calc (i.val % 2 ^ n / M) % 2 * M ≤ i.val % 2 ^ n / M * M :=
            Nat.mul_le_mul_right _ (Nat.mod_le _ _)
        _ ≤ i.val % 2 ^ n := Nat.div_mul_le_self _ _
    have hBnd_j : (j.val % 2 ^ n / M) % 2 * M ≤ j.val % 2 ^ n := by
      calc (j.val % 2 ^ n / M) % 2 * M ≤ j.val % 2 ^ n / M * M :=
            Nat.mul_le_mul_right _ (Nat.mod_le _ _)
        _ ≤ j.val % 2 ^ n := Nat.div_mul_le_self _ _
    have hNeq :
        ¬ i.val / 2 ^ n * 2 ^ n +
            (i.val % 2 ^ n - (i.val % 2 ^ n / M) % 2 * M) =
          j.val / 2 ^ n * 2 ^ n +
            (j.val % 2 ^ n - (j.val % 2 ^ n / M) % 2 * M) := by
      intro h
      have hiR' : i.val % 2 ^ n - (i.val % 2 ^ n / M) % 2 * M < 2 ^ n :=
        lt_of_le_of_lt (Nat.sub_le _ _) hiR_lt
      have hjR' : j.val % 2 ^ n - (j.val % 2 ^ n / M) % 2 * M < 2 ^ n :=
        lt_of_le_of_lt (Nat.sub_le _ _) hjR_lt
      have heq_mod :
          (i.val / 2 ^ n * 2 ^ n + (i.val % 2 ^ n - (i.val % 2 ^ n / M) % 2 * M)) % 2 ^ n =
          (j.val / 2 ^ n * 2 ^ n + (j.val % 2 ^ n - (j.val % 2 ^ n / M) % 2 * M)) % 2 ^ n := by
        rw [h]
      rw [Nat.mul_add_mod_of_lt hiR', Nat.mul_add_mod_of_lt hjR'] at heq_mod
      have hmul : i.val / 2 ^ n * 2 ^ n = j.val / 2 ^ n * 2 ^ n := by omega
      exact hTop (Nat.eq_of_mul_eq_mul_right hN_pos hmul)
    rw [if_neg hNeq]

/-- Embedding a single-qubit gate at the **head** qubit `0` on `(n+1)` qubits
    is `U ⊗ I_{2^n}`, modulo the `tailEquiv` reindexing. The head-qubit analog
    of `embed_single_qubit_shift`: here the gate's mask is exactly `2^n`, the
    `tailEquiv` split point, so the high bit is the first component and the low
    `n` bits are the second. -/
theorem embed_single_qubit_head (n : ℕ) (U : Matrix (Fin 2) (Fin 2) ℂ) :
    embed_single_qubit (n + 1) ⟨0, by omega⟩ U =
      Matrix.reindex (tailEquiv n) (tailEquiv n)
        (Matrix.kronecker U (1 : Matrix (Fin (2^n)) (Fin (2^n)) ℂ)) := by
  ext i j
  have hN_pos : 0 < (2 : ℕ) ^ n := Nat.pos_of_ne_zero (by positivity)
  have hpow : (2 : ℕ) ^ (n + 1) = 2 * 2 ^ n := by rw [pow_succ]; ring
  have hi : i.val < 2 * 2 ^ n := by have := i.isLt; omega
  have hj : j.val < 2 * 2 ^ n := by have := j.isLt; omega
  have hi2 : i.val / 2 ^ n < 2 := by rw [Nat.div_lt_iff_lt_mul hN_pos]; omega
  have hj2 : j.val / 2 ^ n < 2 := by rw [Nat.div_lt_iff_lt_mul hN_pos]; omega
  -- High bit is just the quotient (mod 2 is identity since quotient < 2).
  have hi_q : (i.val / 2 ^ (n + 1 - 1 - 0)) % 2 = i.val / 2 ^ n :=
    Nat.mod_eq_of_lt hi2
  have hj_q : (j.val / 2 ^ (n + 1 - 1 - 0)) % 2 = j.val / 2 ^ n :=
    Nat.mod_eq_of_lt hj2
  -- The rest is the low `n` bits.
  have hexp : n + 1 - 1 - 0 = n := rfl
  have hi_rest : i.val - (i.val / 2 ^ (n + 1 - 1 - 0)) % 2 * 2 ^ (n + 1 - 1 - 0)
      = i.val % 2 ^ n := by
    rw [hexp, Nat.mod_eq_of_lt hi2, mul_comm]
    have := Nat.div_add_mod i.val (2 ^ n); omega
  have hj_rest : j.val - (j.val / 2 ^ (n + 1 - 1 - 0)) % 2 * 2 ^ (n + 1 - 1 - 0)
      = j.val % 2 ^ n := by
    rw [hexp, Nat.mod_eq_of_lt hj2, mul_comm]
    have := Nat.div_add_mod j.val (2 ^ n); omega
  -- Unfold RHS: reindex → submatrix → kronecker entry.
  simp only [Matrix.reindex_apply, Matrix.submatrix_apply, Matrix.kronecker,
             Matrix.kroneckerMap_apply]
  unfold embed_single_qubit
  simp only [hi_rest, hj_rest, tailEquiv_symm_snd_val]
  by_cases hRest : i.val % 2 ^ n = j.val % 2 ^ n
  · -- Same low bits ⇒ identity indicator = 1.
    have hSnd : ((tailEquiv n).symm i).2 = ((tailEquiv n).symm j).2 := by
      apply Fin.ext; rw [tailEquiv_symm_snd_val, tailEquiv_symm_snd_val]; exact hRest
    rw [if_pos hRest, hSnd, Matrix.one_apply_eq, mul_one]
    -- U-entry equality: indices are the high bits = first tailEquiv component.
    apply congr_arg₂
    · apply Fin.ext; rw [tailEquiv_symm_fst_val]; exact hi_q
    · apply Fin.ext; rw [tailEquiv_symm_fst_val]; exact hj_q
  · -- Different low bits ⇒ identity indicator = 0.
    have hSnd : ((tailEquiv n).symm i).2 ≠ ((tailEquiv n).symm j).2 := by
      intro h
      apply hRest
      have := congrArg Fin.val h
      rwa [tailEquiv_symm_snd_val, tailEquiv_symm_snd_val] at this
    rw [if_neg hRest, Matrix.one_apply_ne hSnd, mul_zero]

/-- A single bit-contribution at the lower position `a` (with `a < b`) is
    bounded by the low `b` bits of `r`: the bit lives strictly below `2^b`,
    so it is absorbed by `r % 2^b`. The bit value at `a` is the same whether
    read off `r` or off `r % 2^b` (the higher factor `2^(b-a)` is even). -/
private lemma bit_le_mod (r a b : ℕ) (h : a < b) :
    (r / 2 ^ a) % 2 * 2 ^ a ≤ r % 2 ^ b := by
  have hA_pos : 0 < (2 : ℕ) ^ a := Nat.pos_of_ne_zero (by positivity)
  have hsplit : (2 : ℕ) ^ b = 2 ^ a * 2 ^ (b - a) := by rw [← pow_add]; congr 1; omega
  have hdiv : r / 2 ^ a = r % 2 ^ b / 2 ^ a + (r / 2 ^ b) * 2 ^ (b - a) := by
    have hdec : r = r % 2 ^ b + ((r / 2 ^ b) * 2 ^ (b - a)) * 2 ^ a := by
      have hbase : r = 2 ^ b * (r / 2 ^ b) + r % 2 ^ b :=
        (Nat.div_add_mod r (2 ^ b)).symm
      have hmul : 2 ^ b * (r / 2 ^ b) = ((r / 2 ^ b) * 2 ^ (b - a)) * 2 ^ a := by
        rw [hsplit]; ring
      omega
    conv_lhs => rw [hdec]
    rw [Nat.add_mul_div_right _ _ hA_pos]
  have heven : (r / 2 ^ b) * 2 ^ (b - a) % 2 = 0 := by
    have h2 : 2 ^ (b - a) = 2 * 2 ^ (b - a - 1) := by rw [← pow_succ']; congr 1; omega
    rw [h2, show (r / 2 ^ b) * (2 * 2 ^ (b - a - 1)) =
          ((r / 2 ^ b) * 2 ^ (b - a - 1)) * 2 from by ring]
    omega
  have hbiteq : (r / 2 ^ a) % 2 = (r % 2 ^ b / 2 ^ a) % 2 := by rw [hdiv]; omega
  calc (r / 2 ^ a) % 2 * 2 ^ a
      = (r % 2 ^ b / 2 ^ a) % 2 * 2 ^ a := by rw [hbiteq]
    _ ≤ (r % 2 ^ b / 2 ^ a) * 2 ^ a := Nat.mul_le_mul_right _ (Nat.mod_le _ _)
    _ ≤ r % 2 ^ b := Nat.div_mul_le_self _ _

/-- Bit `a` of `x` equals bit `a` of `x % 2^b`, for `a < b`. (Relocated ahead of
    the two- and three-qubit shift lemmas, which precede the bit-extraction
    algebra section below but need this fact for the three-bit bound.) -/
lemma bit_eq_mod (z a b : ℕ) (h : a < b) :
    (z / 2^a) % 2 = ((z % 2^b) / 2^a) % 2 := by
  have hsplit : 2^a * 2^(b-a) = 2^b := by rw [← pow_add]; congr 1; omega
  have h1 : (z % 2^b) / 2^a = z / 2^a % 2^(b-a) := by
    rw [← hsplit]; exact Nat.mod_mul_right_div_self z (2^a) (2^(b-a))
  have h2 : (2:ℕ) ∣ 2^(b-a) := dvd_pow_self 2 (by omega)
  rw [h1, Nat.mod_mod_of_dvd _ h2]

/-- The two bit-contributions at *distinct* positions `a ≠ b` sum to at most
    `r`. Splitting at the higher position `b`, the high bit is bounded by
    `(r/2^b)·2^b` and the low bit by `r % 2^b`; together they reassemble to
    `r`. (This is exactly where `q0 ≠ q1` is needed: a repeated position
    would double-count and could exceed `r`.) -/
private lemma two_bit_le (r a b : ℕ) (h : a < b) :
    (r / 2 ^ a) % 2 * 2 ^ a + (r / 2 ^ b) % 2 * 2 ^ b ≤ r := by
  have hb : r = (r / 2 ^ b) * 2 ^ b + r % 2 ^ b := (Nat.div_add_mod' r (2 ^ b)).symm
  have h_bit_b : (r / 2 ^ b) % 2 * 2 ^ b ≤ (r / 2 ^ b) * 2 ^ b :=
    Nat.mul_le_mul_right _ (Nat.mod_le _ _)
  have h_bit_a : (r / 2 ^ a) % 2 * 2 ^ a ≤ r % 2 ^ b := bit_le_mod r a b h
  omega

/-- Three bit-contributions at *sorted* positions `a < b < c` sum to at most `r`.
    Split at the top position `c`: the `c`-bit is bounded by `(r/2^c)·2^c` and the
    `a`,`b` bits jointly by `r % 2^c` (`two_bit_le` on the low part, via
    `bit_eq_mod`). -/
private lemma three_bit_le_sorted (r a b c : ℕ) (hab : a < b) (hbc : b < c) :
    (r / 2 ^ a) % 2 * 2 ^ a + (r / 2 ^ b) % 2 * 2 ^ b + (r / 2 ^ c) % 2 * 2 ^ c ≤ r := by
  have hc : r = (r / 2 ^ c) * 2 ^ c + r % 2 ^ c := (Nat.div_add_mod' r (2 ^ c)).symm
  have hcc : (r / 2 ^ c) % 2 * 2 ^ c ≤ (r / 2 ^ c) * 2 ^ c :=
    Nat.mul_le_mul_right _ (Nat.mod_le _ _)
  have hab2 : (r / 2 ^ a) % 2 * 2 ^ a + (r / 2 ^ b) % 2 * 2 ^ b ≤ r % 2 ^ c := by
    rw [bit_eq_mod r a c (by omega), bit_eq_mod r b c hbc]
    exact two_bit_le (r % 2 ^ c) a b hab
  omega

/-- Three bit-contributions at any three *distinct* positions sum to at most `r`.
    The sum is symmetric, so reduce to the sorted case (`three_bit_le_sorted`).
    The three-position analog of `two_bit_le`; backs `shifted_qubit_rest_three`. -/
private lemma three_bit_le (r a b c : ℕ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    (r / 2 ^ a) % 2 * 2 ^ a + (r / 2 ^ b) % 2 * 2 ^ b + (r / 2 ^ c) % 2 * 2 ^ c ≤ r := by
  rcases lt_trichotomy a b with h1 | h1 | h1
  · rcases lt_trichotomy b c with h2 | h2 | h2
    · exact three_bit_le_sorted r a b c h1 h2
    · omega
    · rcases lt_trichotomy a c with h3 | h3 | h3
      · have := three_bit_le_sorted r a c b h3 h2; omega
      · omega
      · have := three_bit_le_sorted r c a b h3 h1; omega
  · omega
  · rcases lt_trichotomy b c with h2 | h2 | h2
    · rcases lt_trichotomy a c with h3 | h3 | h3
      · have := three_bit_le_sorted r b a c h1 h3; omega
      · omega
      · have := three_bit_le_sorted r b c a h2 h3; omega
    · omega
    · have := three_bit_le_sorted r c b a h2 h1; omega

/-- **Three-bit injectivity:** indices with equal extracted bits at `q0,q1,q2`
    and equal `rest` are equal. The injectivity backbone of the three-qubit
    diagonal collapse (3-qubit analog of `two_bit_inj`). -/
private lemma three_bit_inj (n : ℕ) (q0 q1 q2 : Fin n)
    (h01 : q0.val ≠ q1.val) (h02 : q0.val ≠ q2.val) (h12 : q1.val ≠ q2.val)
    (i j : Fin (2 ^ n))
    (hb0 : (i.val / 2 ^ (n - 1 - q0.val)) % 2 = (j.val / 2 ^ (n - 1 - q0.val)) % 2)
    (hb1 : (i.val / 2 ^ (n - 1 - q1.val)) % 2 = (j.val / 2 ^ (n - 1 - q1.val)) % 2)
    (hb2 : (i.val / 2 ^ (n - 1 - q2.val)) % 2 = (j.val / 2 ^ (n - 1 - q2.val)) % 2)
    (hr : i.val - (i.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val)
                - (i.val / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)
                - (i.val / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val)
        = j.val - (j.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val)
                - (j.val / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)
                - (j.val / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val)) :
    i = j := by
  apply Fin.ext
  have d0 := q0.isLt; have d1 := q1.isLt; have d2 := q2.isLt
  have hT0 : (i.val / 2 ^ (n-1-q0.val)) % 2 * 2 ^ (n-1-q0.val)
           = (j.val / 2 ^ (n-1-q0.val)) % 2 * 2 ^ (n-1-q0.val) := by rw [hb0]
  have hT1 : (i.val / 2 ^ (n-1-q1.val)) % 2 * 2 ^ (n-1-q1.val)
           = (j.val / 2 ^ (n-1-q1.val)) % 2 * 2 ^ (n-1-q1.val) := by rw [hb1]
  have hT2 : (i.val / 2 ^ (n-1-q2.val)) % 2 * 2 ^ (n-1-q2.val)
           = (j.val / 2 ^ (n-1-q2.val)) % 2 * 2 ^ (n-1-q2.val) := by rw [hb2]
  have hbi := three_bit_le i.val (n-1-q0.val) (n-1-q1.val) (n-1-q2.val) (by omega) (by omega) (by omega)
  have hbj := three_bit_le j.val (n-1-q0.val) (n-1-q1.val) (n-1-q2.val) (by omega) (by omega) (by omega)
  omega

/-- **`embed_three_qubit` of a diagonal is diagonal:** the embedded matrix carries
    `Matrix.diagonal d` to the diagonal whose `i`-th entry is `d` at `i`'s extracted
    bit-triple `q0·4 + q1·2 + q2`. (Used for the Grover oracle/diffusion, whose
    `CCZ` core is diagonal.) The 3-qubit analog of `embed_two_qubit_diagonal`. -/
theorem embed_three_qubit_diagonal (n : ℕ) (q0 q1 q2 : Fin n)
    (h01 : q0 ≠ q1) (h02 : q0 ≠ q2) (h12 : q1 ≠ q2) (d : Fin 8 → ℂ) :
    embed_three_qubit n q0 q1 q2 (Matrix.diagonal d)
      = Matrix.diagonal (fun i : Fin (2 ^ n) =>
          d ⟨(i.val / 2 ^ (n-1-q0.val)) % 2 * 4 + (i.val / 2 ^ (n-1-q1.val)) % 2 * 2
              + (i.val / 2 ^ (n-1-q2.val)) % 2, by omega⟩) := by
  have h01' : q0.val ≠ q1.val := fun h => h01 (Fin.ext h)
  have h02' : q0.val ≠ q2.val := fun h => h02 (Fin.ext h)
  have h12' : q1.val ≠ q2.val := fun h => h12 (Fin.ext h)
  ext i j
  rw [Matrix.diagonal_apply]
  unfold embed_three_qubit
  simp only [Matrix.diagonal_apply]
  by_cases hij : i = j
  · subst hij; simp
  · rw [if_neg hij]
    split_ifs with hr hbits
    · exfalso; apply hij
      have hval : (i.val/2^(n-1-q0.val))%2*4 + (i.val/2^(n-1-q1.val))%2*2 + (i.val/2^(n-1-q2.val))%2
                = (j.val/2^(n-1-q0.val))%2*4 + (j.val/2^(n-1-q1.val))%2*2 + (j.val/2^(n-1-q2.val))%2 := by
        have := congrArg Fin.val hbits; simpa using this
      have hb0 : (i.val/2^(n-1-q0.val))%2 = (j.val/2^(n-1-q0.val))%2 := by omega
      have hb1 : (i.val/2^(n-1-q1.val))%2 = (j.val/2^(n-1-q1.val))%2 := by omega
      have hb2 : (i.val/2^(n-1-q2.val))%2 = (j.val/2^(n-1-q2.val))%2 := by omega
      exact three_bit_inj n q0 q1 q2 h01' h02' h12' i j hb0 hb1 hb2 hr
    · rfl
    · rfl

/-- **`embed_single_qubit` of a diagonal is diagonal:** the embedded matrix
    carries `Matrix.diagonal d` to the diagonal whose `i`-th entry is `d` at
    `i`'s extracted `q`-bit. The 1-qubit analog of `embed_two_qubit_diagonal`
    / `embed_three_qubit_diagonal`. Consumed here by `embed_single_qubit_one`;
    intended for the single-qubit diagonal gates (`Z`, `S`, `T`, `rz`) and the
    diagonal layers of the forthcoming Grover diffusion bridge. -/
theorem embed_single_qubit_diagonal (n : ℕ) (q : Fin n) (d : Fin 2 → ℂ) :
    embed_single_qubit n q (Matrix.diagonal d)
      = Matrix.diagonal (fun i : Fin (2 ^ n) =>
          d ⟨(i.val / 2 ^ (n - 1 - q.val)) % 2, by omega⟩) := by
  ext i j
  rw [Matrix.diagonal_apply]
  unfold embed_single_qubit
  simp only [Matrix.diagonal_apply]
  by_cases hij : i = j
  · subst hij; simp
  · rw [if_neg hij]
    split_ifs with hr hbit
    · exfalso
      apply hij
      apply Fin.ext
      have hb : (i.val / 2 ^ (n - 1 - q.val)) % 2
              = (j.val / 2 ^ (n - 1 - q.val)) % 2 := by
        have := congrArg Fin.val hbit; simpa using this
      have hm : (i.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
              = (j.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val) := by
        rw [hb]
      have hi : (i.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val) ≤ i.val :=
        le_trans (mul_le_mul_right' (Nat.mod_le _ _) _) (Nat.div_mul_le_self _ _)
      have hj : (j.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val) ≤ j.val :=
        le_trans (mul_le_mul_right' (Nat.mod_le _ _) _) (Nat.div_mul_le_self _ _)
      omega
    · rfl
    · rfl

/-- Joint `rest` decomposition for two-qubit embeddings. Generalizes
    `shifted_qubit_rest`: subtracting the `q0` and `q1` bit-contributions
    from `i.val` (in the `(n+1)`-qubit layout) equals `iT * 2^n` plus the
    n-qubit-layout residue. The target form matches the inner
    `embed_two_qubit n q0 q1 U` entry.

    The required inequality `(iR/M0)%2*M0 + (iR/M1)%2*M1 ≤ iR` holds because
    `q0 ≠ q1` makes the two bit positions `n-1-q0`, `n-1-q1` distinct, so
    their contributions sum to at most `iR` (`two_bit_le`). -/
private lemma shifted_qubit_rest_two (n : ℕ) (q0 q1 : Fin n) (hq : q0 ≠ q1)
    (i : Fin (2 ^ (n + 1))) :
    i.val - ((i.val / 2 ^ (n - 1 - q0.val)) % 2) * 2 ^ (n - 1 - q0.val) -
      ((i.val / 2 ^ (n - 1 - q1.val)) % 2) * 2 ^ (n - 1 - q1.val) =
      (i.val / 2 ^ n) * 2 ^ n +
        (i.val % 2 ^ n -
          ((i.val % 2 ^ n) / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
          ((i.val % 2 ^ n) / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)) := by
  have hi_dec : i.val = (i.val / 2 ^ n) * 2 ^ n + i.val % 2 ^ n :=
    (Nat.div_add_mod' i.val (2 ^ n)).symm
  rw [shifted_qubit_bit n q0 i, shifted_qubit_bit n q1 i]
  have hne : (n - 1 - q0.val) ≠ (n - 1 - q1.val) := by
    have h0 := q0.isLt; have h1 := q1.isLt
    have : q0.val ≠ q1.val := fun h => hq (Fin.ext h)
    omega
  set r := i.val % 2 ^ n with hr
  have hAB : (r / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) +
             (r / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) ≤ r := by
    rcases Nat.lt_or_ge (n - 1 - q0.val) (n - 1 - q1.val) with hlt | hge
    · exact two_bit_le r _ _ hlt
    · have hlt' : (n - 1 - q1.val) < (n - 1 - q0.val) := by omega
      have := two_bit_le r _ _ hlt'
      omega
  omega

/-- Embedding a two-qubit gate with both indices shifted by `+1` on
    `(n+1)` qubits is `I₂ ⊗ (embed at qubits q0, q1 on n qubits)`,
    modulo `tailEquiv`. Same structure as `embed_single_qubit_shift`
    but with two bit-subtractions (packaged via
    `shifted_qubit_rest_two`). -/
private lemma embed_two_qubit_shift (n : ℕ) (q0 q1 : Fin n) (hq : q0 ≠ q1)
    (U : Matrix (Fin 4) (Fin 4) ℂ) :
    embed_two_qubit (n + 1) ⟨q0.val + 1, by have := q0.isLt; omega⟩
        ⟨q1.val + 1, by have := q1.isLt; omega⟩ U =
      Matrix.reindex (tailEquiv n) (tailEquiv n)
        (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ)
          (embed_two_qubit n q0 q1 U)) := by
  ext i j
  have hN_pos : 0 < (2 : ℕ) ^ n := Nat.pos_of_ne_zero (by positivity)
  have hExp0 : (n + 1) - 1 - (q0.val + 1) = n - 1 - q0.val := by omega
  have hExp1 : (n + 1) - 1 - (q1.val + 1) = n - 1 - q1.val := by omega
  have hi_bit0 := shifted_qubit_bit n q0 i
  have hj_bit0 := shifted_qubit_bit n q0 j
  have hi_bit1 := shifted_qubit_bit n q1 i
  have hj_bit1 := shifted_qubit_bit n q1 j
  have hi_rest2 := shifted_qubit_rest_two n q0 q1 hq i
  have hj_rest2 := shifted_qubit_rest_two n q0 q1 hq j
  simp only [Matrix.reindex_apply, Matrix.submatrix_apply, Matrix.kronecker,
             Matrix.kroneckerMap_apply]
  unfold embed_two_qubit
  simp only [hExp0, hExp1, tailEquiv_symm_snd_val]
  rw [hi_rest2, hj_rest2]
  by_cases hTop : i.val / 2 ^ n = j.val / 2 ^ n
  · -- Same top bit ⇒ kronecker indicator = 1.
    have hFst : ((tailEquiv n).symm i).1 = ((tailEquiv n).symm j).1 :=
      (tailEquiv_symm_fst_eq_iff n i j).mpr hTop
    rw [hFst, Matrix.one_apply_eq, one_mul]
    congr 1
    · rw [hTop]
      apply propext
      exact ⟨fun h => Nat.add_left_cancel h, fun h => by rw [h]⟩
    · apply congr_arg₂
      · apply Fin.ext
        show _ * 2 + _ = _ * 2 + _
        rw [hi_bit0, hi_bit1]
      · apply Fin.ext
        show _ * 2 + _ = _ * 2 + _
        rw [hj_bit0, hj_bit1]
  · -- Different top bits ⇒ kronecker indicator = 0. LHS condition must fail.
    have hFst : ((tailEquiv n).symm i).1 ≠ ((tailEquiv n).symm j).1 := fun h =>
      hTop ((tailEquiv_symm_fst_eq_iff n i j).mp h)
    rw [Matrix.one_apply_ne hFst, zero_mul]
    have hiR_lt : i.val % 2 ^ n < 2 ^ n := Nat.mod_lt _ hN_pos
    have hjR_lt : j.val % 2 ^ n < 2 ^ n := Nat.mod_lt _ hN_pos
    have hNeq :
        ¬ i.val / 2 ^ n * 2 ^ n +
            (i.val % 2 ^ n -
              (i.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
              (i.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)) =
          j.val / 2 ^ n * 2 ^ n +
            (j.val % 2 ^ n -
              (j.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
              (j.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)) := by
      intro h
      have hiR' : i.val % 2 ^ n -
            (i.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
            (i.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) < 2 ^ n :=
        lt_of_le_of_lt (le_trans (Nat.sub_le _ _) (Nat.sub_le _ _)) hiR_lt
      have hjR' : j.val % 2 ^ n -
            (j.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
            (j.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) < 2 ^ n :=
        lt_of_le_of_lt (le_trans (Nat.sub_le _ _) (Nat.sub_le _ _)) hjR_lt
      have heq_mod :
          (i.val / 2 ^ n * 2 ^ n +
              (i.val % 2 ^ n -
                (i.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
                (i.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val))) % 2 ^ n =
          (j.val / 2 ^ n * 2 ^ n +
              (j.val % 2 ^ n -
                (j.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
                (j.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val))) % 2 ^ n := by
        rw [h]
      rw [Nat.mul_add_mod_of_lt hiR', Nat.mul_add_mod_of_lt hjR'] at heq_mod
      have hmul : i.val / 2 ^ n * 2 ^ n = j.val / 2 ^ n * 2 ^ n := by omega
      exact hTop (Nat.eq_of_mul_eq_mul_right hN_pos hmul)
    rw [if_neg hNeq]

/-- Joint `rest` decomposition for three-qubit embeddings. The three-position
    analog of `shifted_qubit_rest_two`: subtracting the `q0`, `q1`, `q2`
    bit-contributions from `i.val` (in the `(n+1)`-qubit layout) equals
    `iT * 2^n` plus the n-qubit-layout residue. The required inequality
    (sum of the three bit-contributions ≤ residue) holds because `q0,q1,q2`
    are pairwise distinct, so the three bit positions are distinct
    (`three_bit_le`). -/
private lemma shifted_qubit_rest_three (n : ℕ) (q0 q1 q2 : Fin n)
    (h01 : q0 ≠ q1) (h02 : q0 ≠ q2) (h12 : q1 ≠ q2)
    (i : Fin (2 ^ (n + 1))) :
    i.val - ((i.val / 2 ^ (n - 1 - q0.val)) % 2) * 2 ^ (n - 1 - q0.val) -
      ((i.val / 2 ^ (n - 1 - q1.val)) % 2) * 2 ^ (n - 1 - q1.val) -
      ((i.val / 2 ^ (n - 1 - q2.val)) % 2) * 2 ^ (n - 1 - q2.val) =
      (i.val / 2 ^ n) * 2 ^ n +
        (i.val % 2 ^ n -
          ((i.val % 2 ^ n) / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
          ((i.val % 2 ^ n) / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) -
          ((i.val % 2 ^ n) / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val)) := by
  have hi_dec : i.val = (i.val / 2 ^ n) * 2 ^ n + i.val % 2 ^ n :=
    (Nat.div_add_mod' i.val (2 ^ n)).symm
  rw [shifted_qubit_bit n q0 i, shifted_qubit_bit n q1 i, shifted_qubit_bit n q2 i]
  have hne01 : (n - 1 - q0.val) ≠ (n - 1 - q1.val) := by
    have h0 := q0.isLt; have h1 := q1.isLt
    have : q0.val ≠ q1.val := fun h => h01 (Fin.ext h)
    omega
  have hne02 : (n - 1 - q0.val) ≠ (n - 1 - q2.val) := by
    have h0 := q0.isLt; have h2 := q2.isLt
    have : q0.val ≠ q2.val := fun h => h02 (Fin.ext h)
    omega
  have hne12 : (n - 1 - q1.val) ≠ (n - 1 - q2.val) := by
    have h1 := q1.isLt; have h2 := q2.isLt
    have : q1.val ≠ q2.val := fun h => h12 (Fin.ext h)
    omega
  set r := i.val % 2 ^ n with hr
  have hABC : (r / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) +
             (r / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) +
             (r / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val) ≤ r :=
    three_bit_le r _ _ _ hne01 hne02 hne12
  omega

/-- Embedding a three-qubit gate with all three indices shifted by `+1` on
    `(n+1)` qubits is `I₂ ⊗ (embed at qubits q0, q1, q2 on n qubits)`,
    modulo `tailEquiv`. Same structure as `embed_two_qubit_shift` with three
    bit-subtractions (packaged via `shifted_qubit_rest_three`). The shift
    Kronecker identity the forthcoming `ccz` gate arm of
    `denote_gate_shift_kronecker` consumes. -/
private lemma embed_three_qubit_shift (n : ℕ) (q0 q1 q2 : Fin n)
    (h01 : q0 ≠ q1) (h02 : q0 ≠ q2) (h12 : q1 ≠ q2)
    (U : Matrix (Fin 8) (Fin 8) ℂ) :
    embed_three_qubit (n + 1) ⟨q0.val + 1, by have := q0.isLt; omega⟩
        ⟨q1.val + 1, by have := q1.isLt; omega⟩
        ⟨q2.val + 1, by have := q2.isLt; omega⟩ U =
      Matrix.reindex (tailEquiv n) (tailEquiv n)
        (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ)
          (embed_three_qubit n q0 q1 q2 U)) := by
  ext i j
  have hN_pos : 0 < (2 : ℕ) ^ n := Nat.pos_of_ne_zero (by positivity)
  have hExp0 : (n + 1) - 1 - (q0.val + 1) = n - 1 - q0.val := by omega
  have hExp1 : (n + 1) - 1 - (q1.val + 1) = n - 1 - q1.val := by omega
  have hExp2 : (n + 1) - 1 - (q2.val + 1) = n - 1 - q2.val := by omega
  have hi_bit0 := shifted_qubit_bit n q0 i
  have hj_bit0 := shifted_qubit_bit n q0 j
  have hi_bit1 := shifted_qubit_bit n q1 i
  have hj_bit1 := shifted_qubit_bit n q1 j
  have hi_bit2 := shifted_qubit_bit n q2 i
  have hj_bit2 := shifted_qubit_bit n q2 j
  have hi_rest3 := shifted_qubit_rest_three n q0 q1 q2 h01 h02 h12 i
  have hj_rest3 := shifted_qubit_rest_three n q0 q1 q2 h01 h02 h12 j
  simp only [Matrix.reindex_apply, Matrix.submatrix_apply, Matrix.kronecker,
             Matrix.kroneckerMap_apply]
  unfold embed_three_qubit
  simp only [hExp0, hExp1, hExp2, tailEquiv_symm_snd_val]
  rw [hi_rest3, hj_rest3]
  by_cases hTop : i.val / 2 ^ n = j.val / 2 ^ n
  · -- Same top bit ⇒ kronecker indicator = 1.
    have hFst : ((tailEquiv n).symm i).1 = ((tailEquiv n).symm j).1 :=
      (tailEquiv_symm_fst_eq_iff n i j).mpr hTop
    rw [hFst, Matrix.one_apply_eq, one_mul]
    congr 1
    · rw [hTop]
      apply propext
      exact ⟨fun h => Nat.add_left_cancel h, fun h => by rw [h]⟩
    · apply congr_arg₂
      · apply Fin.ext
        show _ * 4 + _ * 2 + _ = _ * 4 + _ * 2 + _
        rw [hi_bit0, hi_bit1, hi_bit2]
      · apply Fin.ext
        show _ * 4 + _ * 2 + _ = _ * 4 + _ * 2 + _
        rw [hj_bit0, hj_bit1, hj_bit2]
  · -- Different top bits ⇒ kronecker indicator = 0. LHS condition must fail.
    have hFst : ((tailEquiv n).symm i).1 ≠ ((tailEquiv n).symm j).1 := fun h =>
      hTop ((tailEquiv_symm_fst_eq_iff n i j).mp h)
    rw [Matrix.one_apply_ne hFst, zero_mul]
    have hiR_lt : i.val % 2 ^ n < 2 ^ n := Nat.mod_lt _ hN_pos
    have hjR_lt : j.val % 2 ^ n < 2 ^ n := Nat.mod_lt _ hN_pos
    have hNeq :
        ¬ i.val / 2 ^ n * 2 ^ n +
            (i.val % 2 ^ n -
              (i.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
              (i.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) -
              (i.val % 2 ^ n / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val)) =
          j.val / 2 ^ n * 2 ^ n +
            (j.val % 2 ^ n -
              (j.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
              (j.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) -
              (j.val % 2 ^ n / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val)) := by
      intro h
      have hiR' : i.val % 2 ^ n -
            (i.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
            (i.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) -
            (i.val % 2 ^ n / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val) < 2 ^ n :=
        lt_of_le_of_lt
          (le_trans (Nat.sub_le _ _) (le_trans (Nat.sub_le _ _) (Nat.sub_le _ _))) hiR_lt
      have hjR' : j.val % 2 ^ n -
            (j.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
            (j.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) -
            (j.val % 2 ^ n / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val) < 2 ^ n :=
        lt_of_le_of_lt
          (le_trans (Nat.sub_le _ _) (le_trans (Nat.sub_le _ _) (Nat.sub_le _ _))) hjR_lt
      have heq_mod :
          (i.val / 2 ^ n * 2 ^ n +
              (i.val % 2 ^ n -
                (i.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
                (i.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) -
                (i.val % 2 ^ n / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val))) % 2 ^ n =
          (j.val / 2 ^ n * 2 ^ n +
              (j.val % 2 ^ n -
                (j.val % 2 ^ n / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) -
                (j.val % 2 ^ n / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) -
                (j.val % 2 ^ n / 2 ^ (n - 1 - q2.val)) % 2 * 2 ^ (n - 1 - q2.val))) % 2 ^ n := by
        rw [h]
      rw [Nat.mul_add_mod_of_lt hiR', Nat.mul_add_mod_of_lt hjR'] at heq_mod
      have hmul : i.val / 2 ^ n * 2 ^ n = j.val / 2 ^ n * 2 ^ n := by omega
      exact hTop (Nat.eq_of_mul_eq_mul_right hN_pos hmul)
    rw [if_neg hNeq]

/-- Gate-level Kronecker identity: shifting a gate's qubit indices by
    `+1` and embedding into the `(n+1)`-qubit space yields exactly
    `I₂ ⊗ (denote_gate g)`, after reindexing through `tailEquiv`. Case
    analysis on `g` reduces to `embed_single_qubit_shift` (for
    1-qubit gates) or `embed_two_qubit_shift` (for `.cx` / `.cz`). -/
theorem denote_gate_shift_kronecker {n : ℕ} (g : GateApp n) :
    denote_gate (shift_gate g) =
      Matrix.reindex (tailEquiv n) (tailEquiv n)
        (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ) (denote_gate g)) := by
  cases g with
  | h q    => exact embed_single_qubit_shift n q Gates.H
  | x q    => exact embed_single_qubit_shift n q Gates.X
  | y q    => exact embed_single_qubit_shift n q Gates.Y
  | z q    => exact embed_single_qubit_shift n q Gates.Z
  | s q    => exact embed_single_qubit_shift n q Gates.S
  | t q    => exact embed_single_qubit_shift n q Gates.T
  | rz q θ => exact embed_single_qubit_shift n q (Gates.RZ θ)
  | ry q θ => exact embed_single_qubit_shift n q (Gates.RY θ)
  | cx c t h => exact embed_two_qubit_shift n c t h Gates.CNOT
  | cz q0 q1 h => exact embed_two_qubit_shift n q0 q1 h CZ4
  | cp c t θ h => exact embed_two_qubit_shift n c t h (CPhase4 θ)
  | swap q0 q1 h => exact embed_two_qubit_shift n q0 q1 h SWAP4
  | ccz q0 q1 q2 h01 h02 h12 => exact embed_three_qubit_shift n q0 q1 q2 h01 h02 h12 CCZ8

/-- `1 ⊗ 1 = 1` for the specific shapes used here, packaged behind
    `tailEquiv`. Pure mathlib repackaging — uses `one_kronecker_one`
    plus `submatrix_one_equiv` (via `reindex_apply`). -/
theorem reindex_one_kronecker_one (n : ℕ) :
    Matrix.reindex (tailEquiv n) (tailEquiv n)
        (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ)
                          (1 : Matrix (Fin (2^n)) (Fin (2^n)) ℂ)) =
      (1 : Matrix (Fin (2^(n+1))) (Fin (2^(n+1))) ℂ) := by
  rw [show (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ)
            (1 : Matrix (Fin (2^n)) (Fin (2^n)) ℂ)) =
        (1 : Matrix (Fin 2 × Fin (2^n)) (Fin 2 × Fin (2^n)) ℂ) from
    Matrix.one_kronecker_one]
  -- `reindex e e 1 = 1`
  rw [Matrix.reindex_apply, Matrix.submatrix_one_equiv]

/-- High bit of an `(n+1)`-qubit index is `< 2`. -/
theorem div_two_pow_lt (k : ℕ) (i : Fin (2^(k+1))) : i.val / 2^k < 2 := by
  have h2 : 0 < (2:ℕ)^k := Nat.pos_of_ne_zero (by positivity)
  have heq : (2:ℕ)^(k+1) = 2 * 2^k := by rw [pow_succ]; ring
  rw [Nat.div_lt_iff_lt_mul h2]
  exact lt_of_lt_of_eq i.isLt heq

/-- **Entry of a Kronecker product through `tailEquiv`.** The `(i,l)` entry of
    `reindex (tailEquiv k) (tailEquiv k) (A ⊗ B)` factors as
    `A[i/2^k, l/2^k] · B[i mod 2^k, l mod 2^k]` — the high bits index `A`, the
    low `k` bits index `B`. -/
theorem reindex_tailEquiv_kron_apply (k : ℕ) (A : Matrix (Fin 2) (Fin 2) ℂ)
    (B : Matrix (Fin (2^k)) (Fin (2^k)) ℂ) (i l : Fin (2^(k+1))) :
    Matrix.reindex (tailEquiv k) (tailEquiv k) (Matrix.kronecker A B) i l
      = A ⟨i.val/2^k, div_two_pow_lt k i⟩ ⟨l.val/2^k, div_two_pow_lt k l⟩
        * B ⟨i.val%2^k, Nat.mod_lt _ (Nat.pos_of_ne_zero (by positivity))⟩
            ⟨l.val%2^k, Nat.mod_lt _ (Nat.pos_of_ne_zero (by positivity))⟩ := by
  rw [Matrix.reindex_apply, Matrix.submatrix_apply, Matrix.kronecker,
      Matrix.kroneckerMap_apply]
  exact congr_arg₂ (· * ·)
    (congr_arg₂ A (Fin.ext (tailEquiv_symm_fst_val k i))
      (Fin.ext (tailEquiv_symm_fst_val k l)))
    (congr_arg₂ B (Fin.ext (tailEquiv_symm_snd_val k i))
      (Fin.ext (tailEquiv_symm_snd_val k l)))

/-- The `reindex (kron 1 _)` operation distributes over matrix
    multiplication, since `(I ⊗ A) · (I ⊗ B) = I ⊗ (A · B)` and
    `reindex` commutes with multiplication when both sides use the
    same equivalence. -/
private lemma reindex_kron_mul {n : ℕ}
    (A B : Matrix (Fin (2^n)) (Fin (2^n)) ℂ) :
    Matrix.reindex (tailEquiv n) (tailEquiv n)
        (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ) A) *
      Matrix.reindex (tailEquiv n) (tailEquiv n)
        (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ) B) =
    Matrix.reindex (tailEquiv n) (tailEquiv n)
      (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ) (A * B)) := by
  rw [Matrix.reindex_apply, Matrix.reindex_apply, Matrix.reindex_apply,
      Matrix.submatrix_mul_equiv]
  congr 1
  -- (1 ⊗ A) * (1 ⊗ B) = (1 * 1) ⊗ (A * B) = 1 ⊗ (A * B)
  show Matrix.kroneckerMap (· * ·) (1 : Matrix (Fin 2) (Fin 2) ℂ) A *
       Matrix.kroneckerMap (· * ·) (1 : Matrix (Fin 2) (Fin 2) ℂ) B =
       Matrix.kroneckerMap (· * ·) (1 : Matrix (Fin 2) (Fin 2) ℂ) (A * B)
  rw [← Matrix.mul_kronecker_mul, Matrix.one_mul]

/-- Denoting an embedded sub-circuit is the Kronecker product
    `I_2 ⊗ denote c`, reindexed through `tailEquiv n`. The structural
    induction over `c` reduces this to the gate-level Kronecker
    identity `denote_gate_shift_kronecker`. -/
theorem denote_embed_subcircuit {n : ℕ} (c : Circuit n) :
    denote (embed_subcircuit c) =
      Matrix.reindex (tailEquiv n) (tailEquiv n)
        (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ) (denote c)) := by
  induction c with
  | nil =>
      rw [embed_subcircuit_nil, empty_circuit_identity, empty_circuit_identity,
          reindex_one_kronecker_one]
  | cons g rest ih =>
      rw [embed_subcircuit_cons, denote_cons, denote_cons, ih,
          denote_gate_shift_kronecker, reindex_kron_mul]

-- ===========================================================================
-- `embed_two_qubit` composition algebra (same qubit pair). Foundations for
-- swap / bit-reversal involution in `QFT.lean`.
-- ===========================================================================

/-- The two extracted bit-contributions of an index never exceed it (so the
    `rest` subtraction in `embed_two_qubit` never underflows). -/
private lemma two_bit_bound (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) (x : ℕ) :
    (x / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val)
      + (x / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) ≤ x := by
  have h0 := q0.isLt; have h1 := q1.isLt
  rcases Nat.lt_or_ge (n - 1 - q0.val) (n - 1 - q1.val) with hlt | hge
  · exact two_bit_le x _ _ hlt
  · have hlt' : (n - 1 - q1.val) < (n - 1 - q0.val) := by omega
    have := two_bit_le x _ _ hlt'
    omega

/-- **Same-pair injectivity:** two indices with equal extracted bits (at `q0`,
    `q1`) and equal `rest` are equal. The injectivity backbone of the
    `embed_two_qubit` composition lemmas. -/
private lemma two_bit_inj (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val)
    (i j : Fin (2 ^ n))
    (hb0 : (i.val / 2 ^ (n - 1 - q0.val)) % 2 = (j.val / 2 ^ (n - 1 - q0.val)) % 2)
    (hb1 : (i.val / 2 ^ (n - 1 - q1.val)) % 2 = (j.val / 2 ^ (n - 1 - q1.val)) % 2)
    (hr : i.val - (i.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val)
                - (i.val / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)
        = j.val - (j.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val)
                - (j.val / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)) :
    i = j := by
  apply Fin.ext
  have hT0 : (i.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val)
           = (j.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val) := by rw [hb0]
  have hT1 : (i.val / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)
           = (j.val / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val) := by rw [hb1]
  have hbi := two_bit_bound n q0 q1 hq i.val
  have hbj := two_bit_bound n q0 q1 hq j.val
  omega

-- Bit-extraction algebra (over `ℕ`) — the surjection/round-trip facts that,
-- with `two_bit_inj`, give the `embed_two_qubit` multiplicativity law below.

/-- Clearing bit `a` (subtracting its contribution) makes bit `a` zero. -/
private lemma clear_bit_self (x a : ℕ) : ((x - (x / 2^a) % 2 * 2^a) / 2^a) % 2 = 0 := by
  have hpa : 0 < 2^a := Nat.pos_of_ne_zero (by positivity)
  set q := x / 2^a with hq
  have hdm : x = 2^a * q + x % 2^a := (Nat.div_add_mod x (2^a)).symm
  have h1 : 2^a * q = 2^a * (2 * (q / 2)) + q % 2 * 2^a := by
    conv_lhs => rw [show q = 2 * (q / 2) + q % 2 from (Nat.div_add_mod q 2).symm]
    ring
  have hmod_lt : x % 2^a < 2^a := Nat.mod_lt _ hpa
  have hclear : x - q % 2 * 2^a = 2^a * (2 * (q / 2)) + x % 2^a := by omega
  rw [hclear, Nat.mul_add_div hpa, Nat.div_eq_of_lt hmod_lt, add_zero, Nat.mul_mod_right]

/-- Clearing bit `b` preserves bit `a` (for `a ≠ b`). -/
private lemma clear_bit_other (x a b : ℕ) (hab : a ≠ b) :
    ((x - (x / 2^b) % 2 * 2^b) / 2^a) % 2 = (x / 2^a) % 2 := by
  have hbb_le : (x / 2^b) % 2 * 2^b ≤ x :=
    le_trans (Nat.mul_le_mul_right _ (Nat.mod_le _ _)) (Nat.div_mul_le_self _ _)
  rcases Nat.lt_or_ge a b with hlt | hge
  · have hy : (x - (x / 2^b) % 2 * 2^b) % 2^b = x % 2^b := by
      conv_rhs => rw [show x = (x - (x / 2^b) % 2 * 2^b) + (x / 2^b) % 2 * 2^b from by omega]
      rw [Nat.add_mul_mod_self_right]
    rw [bit_eq_mod _ a b hlt, bit_eq_mod x a b hlt, hy]
  · have hba : b < a := by omega
    have hpa : 0 < 2^a := by positivity
    have hble : (x / 2^b) % 2 * 2^b ≤ x % 2^a := by
      rcases Nat.eq_zero_or_pos ((x / 2^b) % 2) with h0 | h1
      · simp [h0]
      · have hbb1 : (x / 2^b) % 2 = 1 := by omega
        rw [hbb1, one_mul]
        have hbe : ((x % 2^a) / 2^b) % 2 = 1 := by rw [← bit_eq_mod x b a hba, hbb1]
        have hpos : 1 ≤ (x % 2^a) / 2^b := by
          rcases Nat.eq_zero_or_pos ((x % 2^a) / 2^b) with h | h
          · rw [h] at hbe; simp at hbe
          · exact h
        calc 2^b = 1 * 2^b := (one_mul _).symm
          _ ≤ (x % 2^a) / 2^b * 2^b := Nat.mul_le_mul_right _ hpos
          _ ≤ x % 2^a := Nat.div_mul_le_self _ _
    have hxd : x = 2^a * (x / 2^a) + x % 2^a := (Nat.div_add_mod x (2^a)).symm
    have hlt2 : x % 2^a - (x / 2^b) % 2 * 2^b < 2^a := by
      have := Nat.mod_lt x hpa; omega
    have hyform : x - (x / 2^b) % 2 * 2^b
        = 2^a * (x / 2^a) + (x % 2^a - (x / 2^b) % 2 * 2^b) := by omega
    rw [hyform, Nat.mul_add_div hpa, Nat.div_eq_of_lt hlt2, add_zero]

/-- The `rest` (both bits cleared) has zero bits at both positions. -/
private lemma rest_bits_zero (x A B : ℕ) (hAB : A ≠ B) :
    ((x - (x/2^A)%2*2^A - (x/2^B)%2*2^B) / 2^A) % 2 = 0
  ∧ ((x - (x/2^A)%2*2^A - (x/2^B)%2*2^B) / 2^B) % 2 = 0 := by
  constructor
  · have comm : x - (x/2^A)%2*2^A - (x/2^B)%2*2^B
              = (x - (x/2^B)%2*2^B) - (x/2^A)%2*2^A := by
      rw [Nat.sub_sub, Nat.sub_sub, Nat.add_comm]
    rw [comm]
    have hbit : (x/2^A)%2 = ((x - (x/2^B)%2*2^B)/2^A)%2 := (clear_bit_other x A B hAB).symm
    rw [hbit]
    exact clear_bit_self (x - (x/2^B)%2*2^B) A
  · have hbit : (x/2^B)%2 = ((x - (x/2^A)%2*2^A)/2^B)%2 :=
      (clear_bit_other x B A (Ne.symm hAB)).symm
    rw [hbit]
    exact clear_bit_self (x - (x/2^A)%2*2^A) B

/-- Adding `c` at a zero bit `A` sets bit `A` to `c` (`c < 2`). -/
private lemma add_bit_self (u c A : ℕ) (hc : c < 2) (hu0 : (u/2^A)%2 = 0) :
    ((u + c*2^A)/2^A)%2 = c := by
  have hpa : 0 < 2^A := by positivity
  rw [Nat.add_mul_div_right u c hpa]
  omega

/-- Adding `c` at a zero bit `B` preserves bit `A` (`A ≠ B`, `c < 2`). -/
private lemma add_bit_other (u c A B : ℕ) (hAB : A ≠ B) (hc : c < 2) (hu0 : (u/2^B)%2 = 0) :
    ((u + c*2^B)/2^A)%2 = (u/2^A)%2 := by
  rcases Nat.lt_or_ge A B with hlt | hge
  · have hmod : (u + c*2^B) % 2^B = u % 2^B := by rw [Nat.add_mul_mod_self_right]
    rw [bit_eq_mod _ A B hlt, bit_eq_mod u A B hlt, hmod]
  · have hba : B < A := by omega
    have hpa : 0 < 2^A := by positivity
    have hpb : 0 < 2^B := by positivity
    have hsplit : 2^B * 2^(A-B) = 2^A := by rw [← pow_add]; congr 1; omega
    have hSeven : (2:ℕ)^(A-B) % 2 = 0 := by
      have : (2:ℕ) ∣ 2^(A-B) := dvd_pow_self 2 (by omega); omega
    set w := u % 2^A with hw
    have hwlt : w < 2^A := Nat.mod_lt _ hpa
    have hwB : (w / 2^B) % 2 = 0 := by rw [hw, ← bit_eq_mod u B A hba]; exact hu0
    have hQR : w = 2^B * (w / 2^B) + w % 2^B := (Nat.div_add_mod w (2^B)).symm
    have hR2 : w % 2^B < 2^B := Nat.mod_lt _ hpb
    have hQlt : w / 2^B < 2^(A-B) := by
      rw [Nat.div_lt_iff_lt_mul hpb]
      calc w < 2^A := hwlt
        _ = 2^B * 2^(A-B) := hsplit.symm
        _ = 2^(A-B) * 2^B := by ring
    have hbound : w + c*2^B < 2^A := by
      calc w + c*2^B = 2^B * (w/2^B) + w%2^B + c*2^B := by omega
        _ = 2^B * (w/2^B + c) + w%2^B := by ring
        _ < 2^B * (w/2^B + c) + 2^B := by omega
        _ = 2^B * (w/2^B + c + 1) := by ring
        _ ≤ 2^B * 2^(A-B) := by gcongr; omega
        _ = 2^A := hsplit
    have hxd : u = 2^A * (u/2^A) + u % 2^A := (Nat.div_add_mod u (2^A)).symm
    have hform : u + c*2^B = 2^A * (u/2^A) + (w + c*2^B) := by rw [hw]; omega
    rw [hform, Nat.mul_add_div hpa, Nat.div_eq_of_lt hbound, add_zero]

/-- **Bit-set round-trip (surjection core):** reconstructing an index from a
    zero-bit `rest` `R` and target bits `(c0,c1)` re-extracts exactly those
    bits. Combined with `two_bit_inj`, this is the bijection `Fin 4 ≃ {fixed
    rest}` underlying the `embed_two_qubit` multiplicativity sum. -/
private lemma bit_of_set (R c0 c1 A B : ℕ) (hAB : A ≠ B) (hc0 : c0 < 2) (hc1 : c1 < 2)
    (hRA : (R/2^A)%2 = 0) (hRB : (R/2^B)%2 = 0) :
    ((R + c0*2^A + c1*2^B)/2^A)%2 = c0
  ∧ ((R + c0*2^A + c1*2^B)/2^B)%2 = c1 := by
  constructor
  · have hreorder : R + c0*2^A + c1*2^B = (R + c1*2^B) + c0*2^A := by ring
    rw [hreorder]
    apply add_bit_self _ c0 A hc0
    rw [add_bit_other R c1 A B hAB hc1 hRB]; exact hRA
  · apply add_bit_self _ c1 B hc1
    rw [add_bit_other R c0 B A (Ne.symm hAB) hc0 hRA]; exact hRB

/-- Entry form of `embed_two_qubit` (definitional). -/
private lemma embed_two_qubit_apply (n : ℕ) (q0 q1 : Fin n)
    (U : Matrix (Fin 4) (Fin 4) ℂ) (i j : Fin (2 ^ n)) :
    embed_two_qubit n q0 q1 U i j =
      if i.val - (i.val / 2 ^ (n-1-q0.val)) % 2 * 2 ^ (n-1-q0.val)
              - (i.val / 2 ^ (n-1-q1.val)) % 2 * 2 ^ (n-1-q1.val)
         = j.val - (j.val / 2 ^ (n-1-q0.val)) % 2 * 2 ^ (n-1-q0.val)
                 - (j.val / 2 ^ (n-1-q1.val)) % 2 * 2 ^ (n-1-q1.val)
      then U ⟨(i.val / 2 ^ (n-1-q0.val)) % 2 * 2 + (i.val / 2 ^ (n-1-q1.val)) % 2, by omega⟩
             ⟨(j.val / 2 ^ (n-1-q0.val)) % 2 * 2 + (j.val / 2 ^ (n-1-q1.val)) % 2, by omega⟩
      else 0 := rfl

/-- Setting bit `A` (`< n`) of a value `< 2^n` that has bit `A` zero keeps it
    `< 2^n`. Used to bound the reconstruction `recon c < 2^n`. -/
private lemma set_bit_lt (v c A n : ℕ) (hv : v < 2^n) (hA : A < n)
    (hbit : (v/2^A)%2 = 0) (hc : c < 2) : v + c*2^A < 2^n := by
  have hpA : 0 < 2^A := by positivity
  have hpA1 : 0 < 2^(A+1) := by positivity
  have hsplit : 2^(A+1) * 2^(n-(A+1)) = 2^n := by rw [← pow_add]; congr 1; omega
  have hbitmod : (v % 2^(A+1) / 2^A) % 2 = 0 := by
    rw [← bit_eq_mod v A (A+1) (by omega)]; exact hbit
  have hwd : v % 2^(A+1) / 2^A < 2 := by
    rw [Nat.div_lt_iff_lt_mul hpA]
    have h22 : (2:ℕ) * 2^A = 2^(A+1) := by rw [pow_succ]; ring
    rw [h22]; exact Nat.mod_lt _ hpA1
  have hd0 : v % 2^(A+1) / 2^A = 0 := by
    rcases Nat.eq_zero_or_pos (v % 2^(A+1) / 2^A) with h | h
    · exact h
    · have hW1 : v % 2^(A+1) / 2^A = 1 := by omega
      rw [hW1] at hbitmod; simp at hbitmod
  have hlt2A : v % 2^(A+1) < 2^A := by
    have hdm := Nat.div_add_mod (v % 2^(A+1)) (2^A)
    rw [hd0] at hdm
    have := Nat.mod_lt (v % 2^(A+1)) hpA
    omega
  have hcle : c*2^A ≤ 2^A := by
    have hc1 : c ≤ 1 := by omega
    calc c*2^A ≤ 1*2^A := by gcongr
      _ = 2^A := one_mul _
  have h2A : 2^A + 2^A = 2^(A+1) := by rw [pow_succ]; ring
  have hQR : v = 2^(A+1) * (v / 2^(A+1)) + v % 2^(A+1) := (Nat.div_add_mod v (2^(A+1))).symm
  have hQlt : v / 2^(A+1) < 2^(n-(A+1)) := by
    rw [Nat.div_lt_iff_lt_mul hpA1]
    calc v < 2^n := hv
      _ = 2^(A+1) * 2^(n-(A+1)) := hsplit.symm
      _ = 2^(n-(A+1)) * 2^(A+1) := by ring
  calc v + c*2^A = 2^(A+1) * (v / 2^(A+1)) + (v % 2^(A+1) + c*2^A) := by omega
    _ < 2^(A+1) * (v / 2^(A+1)) + 2^(A+1) := by omega
    _ = 2^(A+1) * (v / 2^(A+1) + 1) := by ring
    _ ≤ 2^(A+1) * 2^(n-(A+1)) := by gcongr; omega
    _ = 2^n := hsplit

/-- **`embed_two_qubit` preserves the identity:** `embed_two_qubit n q0 q1 1 = 1`.
    The first composition law for the two-qubit embedding; off-diagonal entries
    vanish because matching bits *and* `rest` forces index equality
    (`two_bit_inj`). -/
theorem embed_two_qubit_one (n : ℕ) (q0 q1 : Fin n) (hq : q0 ≠ q1) :
    embed_two_qubit n q0 q1 (1 : Matrix (Fin 4) (Fin 4) ℂ) = 1 := by
  have hq' : q0.val ≠ q1.val := fun h => hq (Fin.ext h)
  ext i j
  rw [Matrix.one_apply]
  unfold embed_two_qubit
  simp only [Matrix.one_apply]
  by_cases hij : i = j
  · subst hij; simp
  · rw [if_neg hij]
    split_ifs with hr hbits
    · exfalso
      apply hij
      have hval : (i.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 + (i.val / 2 ^ (n - 1 - q1.val)) % 2
                = (j.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 + (j.val / 2 ^ (n - 1 - q1.val)) % 2 := by
        have := congrArg Fin.val hbits
        simpa using this
      have hi1 : (i.val / 2 ^ (n - 1 - q1.val)) % 2 < 2 := by omega
      have hj1 : (j.val / 2 ^ (n - 1 - q1.val)) % 2 < 2 := by omega
      have hb0 : (i.val / 2 ^ (n - 1 - q0.val)) % 2 = (j.val / 2 ^ (n - 1 - q0.val)) % 2 := by omega
      have hb1 : (i.val / 2 ^ (n - 1 - q1.val)) % 2 = (j.val / 2 ^ (n - 1 - q1.val)) % 2 := by omega
      exact two_bit_inj n q0 q1 hq' i j hb0 hb1 hr
    · rfl
    · rfl

/-- **`embed_two_qubit` of a diagonal is diagonal:** the embedded matrix carries
    `Matrix.diagonal d` to the diagonal whose `i`-th entry is `d` at `i`'s
    extracted bit-pair. (Used for the controlled-phase layer, whose gates are
    diagonal.) -/
theorem embed_two_qubit_diagonal (n : ℕ) (q0 q1 : Fin n) (hq : q0 ≠ q1) (d : Fin 4 → ℂ) :
    embed_two_qubit n q0 q1 (Matrix.diagonal d)
      = Matrix.diagonal (fun i : Fin (2^n) =>
          d ⟨(i.val / 2^(n-1-q0.val)) % 2 * 2 + (i.val / 2^(n-1-q1.val)) % 2, by omega⟩) := by
  have hq' : q0.val ≠ q1.val := fun h => hq (Fin.ext h)
  ext i j
  rw [Matrix.diagonal_apply]
  unfold embed_two_qubit
  simp only [Matrix.diagonal_apply]
  by_cases hij : i = j
  · subst hij; simp
  · rw [if_neg hij]
    split_ifs with hr hbits
    · exfalso
      apply hij
      have hval : (i.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 + (i.val / 2 ^ (n - 1 - q1.val)) % 2
                = (j.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 + (j.val / 2 ^ (n - 1 - q1.val)) % 2 := by
        have := congrArg Fin.val hbits
        simpa using this
      have hi1 : (i.val / 2 ^ (n - 1 - q1.val)) % 2 < 2 := by omega
      have hj1 : (j.val / 2 ^ (n - 1 - q1.val)) % 2 < 2 := by omega
      have hb0 : (i.val / 2 ^ (n - 1 - q0.val)) % 2 = (j.val / 2 ^ (n - 1 - q0.val)) % 2 := by omega
      have hb1 : (i.val / 2 ^ (n - 1 - q1.val)) % 2 = (j.val / 2 ^ (n - 1 - q1.val)) % 2 := by omega
      exact two_bit_inj n q0 q1 hq' i j hb0 hb1 hr
    · rfl
    · rfl

/-- **Denotation of a circuit of diagonal gates** is the diagonal of the
    product of their entries. -/
theorem denote_diag_list {n : ℕ} (d : GateApp n → Fin (2^n) → ℂ) : ∀ (c : Circuit n),
    (∀ g ∈ c, denote_gate g = Matrix.diagonal (d g)) →
    denote c = Matrix.diagonal (fun i => (c.map (fun g => d g i)).prod) := by
  intro c
  induction c with
  | nil =>
      intro _
      rw [empty_circuit_identity]
      simp only [List.map_nil, List.prod_nil, Matrix.diagonal_one]
  | cons g rest ih =>
      intro hc
      rw [denote_cons, ih (fun g' hg' => hc g' (List.mem_cons_of_mem _ hg')),
          hc g List.mem_cons_self, Matrix.diagonal_mul_diagonal]
      congr 1
      funext i
      simp only [List.map_cons, List.prod_cons]
      ring

/-- `CPhase4 θ` is the diagonal matrix `diag(1,1,1,e^{iθ})`. -/
theorem CPhase4_eq_diagonal (θ : ℝ) :
    CPhase4 θ = Matrix.diagonal ![1, 1, 1, Complex.exp (θ * Complex.I)] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [CPhase4, Matrix.diagonal_apply]

/-- A controlled-phase gate denotes to the diagonal whose entry applies the
    phase `e^{iθ}` exactly when both controlled bits are set. -/
theorem denote_gate_cp_diagonal {n : ℕ} (c t : Fin n) (θ : ℝ) (h : c ≠ t) :
    denote_gate (GateApp.cp c t θ h)
      = Matrix.diagonal (fun i : Fin (2^n) =>
          ![1, 1, 1, Complex.exp (θ * Complex.I)]
            ⟨(i.val / 2^(n-1-c.val)) % 2 * 2 + (i.val / 2^(n-1-t.val)) % 2, by omega⟩) := by
  show embed_two_qubit n c t (CPhase4 θ) = _
  rw [CPhase4_eq_diagonal, embed_two_qubit_diagonal n c t h]

/-- **Same-pair multiplicativity:** `embed_two_qubit n q0 q1 U *
    embed_two_qubit n q0 q1 V = embed_two_qubit n q0 q1 (U*V)`. The product sum
    over `Fin (2^n)` collapses (off the fixed-rest fiber the terms vanish) and
    the fiber bijects with `Fin 4` via bit-extraction (`bit_of_set` round-trip,
    `set_bit_lt` bound, `two_bit_inj` injectivity). -/
theorem embed_two_qubit_mul_same (n : ℕ) (q0 q1 : Fin n) (hq : q0 ≠ q1)
    (U V : Matrix (Fin 4) (Fin 4) ℂ) :
    embed_two_qubit n q0 q1 U * embed_two_qubit n q0 q1 V
      = embed_two_qubit n q0 q1 (U * V) := by
  have hq' : q0.val ≠ q1.val := fun h => hq (Fin.ext h)
  have hA : n - 1 - q0.val < n := by have := q0.isLt; omega
  have hB : n - 1 - q1.val < n := by have := q1.isLt; omega
  have hAB : (n - 1 - q0.val) ≠ (n - 1 - q1.val) := by
    have h0 := q0.isLt; have h1 := q1.isLt; omega
  ext i j
  rw [Matrix.mul_apply]
  simp only [embed_two_qubit_apply]
  by_cases hR :
      i.val - (i.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (i.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)
    = j.val - (j.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (j.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)
  · rw [if_pos hR, Matrix.mul_apply]
    obtain ⟨hrA, hrB⟩ := rest_bits_zero i.val (n-1-q0.val) (n-1-q1.val) hAB
    have hrest_lt :
        i.val - (i.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (i.val/2^(n-1-q1.val))%2*2^(n-1-q1.val) < 2^n := by
      have := two_bit_bound n q0 q1 hq' i.val; omega
    set R := i.val - (i.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (i.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)
      with hRdef
    have hbnd : ∀ c : Fin 4,
        R + (c.val/2)*2^(n-1-q0.val) + (c.val%2)*2^(n-1-q1.val) < 2^n := by
      intro c
      have hc2 : c.val/2 < 2 := by omega
      have hc1 : c.val%2 < 2 := by omega
      have h1 := set_bit_lt R (c.val/2) (n-1-q0.val) n hrest_lt hA hrA hc2
      have hbB : ((R + (c.val/2)*2^(n-1-q0.val)) / 2^(n-1-q1.val)) % 2 = 0 := by
        rw [add_bit_other R (c.val/2) (n-1-q1.val) (n-1-q0.val) (Ne.symm hAB) hc2 hrA]
        exact hrB
      exact set_bit_lt _ (c.val%2) (n-1-q1.val) n h1 hB hbB hc1
    rw [← Finset.sum_subset (s₁ := Finset.univ.filter
          (fun k : Fin (2^n) => R = k.val - (k.val/2^(n-1-q0.val))%2*2^(n-1-q0.val)
                                      - (k.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)))
          (Finset.filter_subset _ _)
          (fun k _ hk => by
            simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
            rw [if_neg hk, zero_mul])]
    apply Finset.sum_nbij'
      (i := fun k => (⟨(k.val/2^(n-1-q0.val))%2*2 + (k.val/2^(n-1-q1.val))%2, by omega⟩ : Fin 4))
      (j := fun c => (⟨R + (c.val/2)*2^(n-1-q0.val) + (c.val%2)*2^(n-1-q1.val), hbnd c⟩ : Fin (2^n)))
    · intro k _; exact Finset.mem_univ _
    · intro c _
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      obtain ⟨hx0, hx1⟩ := bit_of_set R (c.val/2) (c.val%2) (n-1-q0.val) (n-1-q1.val)
        hAB (by omega) (by omega) hrA hrB
      simp only [hx0, hx1]; omega
    · intro k hk
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
      apply Fin.ext
      have hk0 : (k.val/2^(n-1-q0.val))%2 < 2 := by omega
      have hk1 : (k.val/2^(n-1-q1.val))%2 < 2 := by omega
      have hkrecon := two_bit_bound n q0 q1 hq' k.val
      show R + ((((k.val/2^(n-1-q0.val))%2*2 + (k.val/2^(n-1-q1.val))%2)/2)) * 2^(n-1-q0.val)
            + (((k.val/2^(n-1-q0.val))%2*2 + (k.val/2^(n-1-q1.val))%2) % 2) * 2^(n-1-q1.val) = k.val
      have he0 : ((k.val/2^(n-1-q0.val))%2*2 + (k.val/2^(n-1-q1.val))%2)/2 = (k.val/2^(n-1-q0.val))%2 := by omega
      have he1 : ((k.val/2^(n-1-q0.val))%2*2 + (k.val/2^(n-1-q1.val))%2)%2 = (k.val/2^(n-1-q1.val))%2 := by omega
      rw [he0, he1, hk]; omega
    · intro c _
      apply Fin.ext
      obtain ⟨hx0, hx1⟩ := bit_of_set R (c.val/2) (c.val%2) (n-1-q0.val) (n-1-q1.val)
        hAB (by omega) (by omega) hrA hrB
      show (((R + (c.val/2)*2^(n-1-q0.val) + (c.val%2)*2^(n-1-q1.val))/2^(n-1-q0.val))%2)*2
            + ((R + (c.val/2)*2^(n-1-q0.val) + (c.val%2)*2^(n-1-q1.val))/2^(n-1-q1.val))%2 = c.val
      rw [hx0, hx1]; omega
    · intro k hk
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
      have hjk : k.val - (k.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (k.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)
          = j.val - (j.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (j.val/2^(n-1-q1.val))%2*2^(n-1-q1.val) :=
        hk.symm.trans hR
      rw [if_pos hk, if_pos hjk]
  · rw [if_neg hR]
    apply Finset.sum_eq_zero
    intro k _
    by_cases h1 :
        i.val - (i.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (i.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)
      = k.val - (k.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (k.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)
    · rw [if_pos h1]
      by_cases h2 :
          k.val - (k.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (k.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)
        = j.val - (j.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (j.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)
      · exact absurd (h1.trans h2) hR
      · rw [if_neg h2, mul_zero]
    · rw [if_neg h1, zero_mul]

/-- Entry form of `embed_single_qubit` (definitional). -/
private lemma embed_single_qubit_apply (n : ℕ) (q : Fin n)
    (U : Matrix (Fin 2) (Fin 2) ℂ) (i j : Fin (2 ^ n)) :
    embed_single_qubit n q U i j =
      if i.val - (i.val / 2 ^ (n-1-q.val)) % 2 * 2 ^ (n-1-q.val)
         = j.val - (j.val / 2 ^ (n-1-q.val)) % 2 * 2 ^ (n-1-q.val)
      then U ⟨(i.val / 2 ^ (n-1-q.val)) % 2, by omega⟩
             ⟨(j.val / 2 ^ (n-1-q.val)) % 2, by omega⟩
      else 0 := rfl

/-- **Same-qubit multiplicativity:** `embed_single_qubit n q U *
    embed_single_qubit n q V = embed_single_qubit n q (U*V)`. The single-qubit
    analog of `embed_two_qubit_mul_same`: the product sum over `Fin (2^n)`
    collapses (off the fixed-rest fiber the terms vanish) and the fiber bijects
    with `Fin 2` via the `q`-bit (`add_bit_self` round-trip, `set_bit_lt` bound,
    `clear_bit_self` rest-bit-zero). Consumed here by
    `embed_single_qubit_x_involutive` (`X·X = I`), the self-inverse property the
    per-wire `X`-masks of the forthcoming general-`w` Grover oracle rely on. -/
theorem embed_single_qubit_mul_same (n : ℕ) (q : Fin n)
    (U V : Matrix (Fin 2) (Fin 2) ℂ) :
    embed_single_qubit n q U * embed_single_qubit n q V
      = embed_single_qubit n q (U * V) := by
  have hA : n - 1 - q.val < n := by have := q.isLt; omega
  ext i j
  rw [Matrix.mul_apply]
  simp only [embed_single_qubit_apply]
  by_cases hR :
      i.val - (i.val/2^(n-1-q.val))%2*2^(n-1-q.val)
    = j.val - (j.val/2^(n-1-q.val))%2*2^(n-1-q.val)
  · rw [if_pos hR, Matrix.mul_apply]
    have hrA : ((i.val - (i.val/2^(n-1-q.val))%2*2^(n-1-q.val)) / 2^(n-1-q.val)) % 2 = 0 :=
      clear_bit_self i.val (n-1-q.val)
    have hrest_lt : i.val - (i.val/2^(n-1-q.val))%2*2^(n-1-q.val) < 2^n := by
      have := i.isLt; omega
    set R := i.val - (i.val/2^(n-1-q.val))%2*2^(n-1-q.val) with hRdef
    have hbnd : ∀ c : Fin 2, R + c.val*2^(n-1-q.val) < 2^n := by
      intro c; exact set_bit_lt R c.val (n-1-q.val) n hrest_lt hA hrA c.isLt
    rw [← Finset.sum_subset (s₁ := Finset.univ.filter
          (fun k : Fin (2^n) => R = k.val - (k.val/2^(n-1-q.val))%2*2^(n-1-q.val)))
          (Finset.filter_subset _ _)
          (fun k _ hk => by
            simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
            rw [if_neg hk, zero_mul])]
    apply Finset.sum_nbij'
      (i := fun k => (⟨(k.val/2^(n-1-q.val))%2, by omega⟩ : Fin 2))
      (j := fun c => (⟨R + c.val*2^(n-1-q.val), hbnd c⟩ : Fin (2^n)))
    · intro k _; exact Finset.mem_univ _
    · intro c _
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      have hbit : ((R + c.val*2^(n-1-q.val))/2^(n-1-q.val))%2 = c.val :=
        add_bit_self R c.val (n-1-q.val) c.isLt hrA
      rw [hbit]; omega
    · intro k hk
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
      apply Fin.ext
      show R + ((k.val/2^(n-1-q.val))%2)*2^(n-1-q.val) = k.val
      have hle : (k.val/2^(n-1-q.val))%2*2^(n-1-q.val) ≤ k.val :=
        le_trans (mul_le_mul_right' (Nat.mod_le _ _) _) (Nat.div_mul_le_self _ _)
      omega
    · intro c _
      apply Fin.ext
      show ((R + c.val*2^(n-1-q.val))/2^(n-1-q.val))%2 = c.val
      exact add_bit_self R c.val (n-1-q.val) c.isLt hrA
    · intro k hk
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
      have hjk : k.val - (k.val/2^(n-1-q.val))%2*2^(n-1-q.val)
          = j.val - (j.val/2^(n-1-q.val))%2*2^(n-1-q.val) := hk.symm.trans hR
      rw [if_pos hk, if_pos hjk]
  · rw [if_neg hR]
    apply Finset.sum_eq_zero
    intro k _
    by_cases h1 :
        i.val - (i.val/2^(n-1-q.val))%2*2^(n-1-q.val)
      = k.val - (k.val/2^(n-1-q.val))%2*2^(n-1-q.val)
    · rw [if_pos h1]
      by_cases h2 :
          k.val - (k.val/2^(n-1-q.val))%2*2^(n-1-q.val)
        = j.val - (j.val/2^(n-1-q.val))%2*2^(n-1-q.val)
      · exact absurd (h1.trans h2) hR
      · rw [if_neg h2, mul_zero]
    · rw [if_neg h1, zero_mul]

/-- **Unit law:** embedding the identity on qubit `q` is the identity. Immediate
    from `embed_single_qubit_diagonal` (the identity is `diagonal 1`). The
    1-qubit analog of `embed_two_qubit_one`. -/
theorem embed_single_qubit_one (n : ℕ) (q : Fin n) :
    embed_single_qubit n q (1 : Matrix (Fin 2) (Fin 2) ℂ) = 1 := by
  rw [← Matrix.diagonal_one, embed_single_qubit_diagonal]
  exact Matrix.diagonal_one

/-- **Embedding a self-inverse gate is self-inverse on its wire.** If `U·U = 1`
    then `embed q U · embed q U = I`. Immediate from `embed_single_qubit_mul_same`
    and the unit law; the common core of the `X`/`H`/Pauli per-wire involutions. -/
theorem embed_single_qubit_involutive (n : ℕ) (q : Fin n)
    (U : Matrix (Fin 2) (Fin 2) ℂ) (hU : U * U = 1) :
    embed_single_qubit n q U * embed_single_qubit n q U = 1 := by
  rw [embed_single_qubit_mul_same, hU, embed_single_qubit_one]

/-- **The embedded `X` gate is an involution on every wire:**
    `embed q X · embed q X = I`. This is exactly the self-inverse property the
    per-wire `X`-masks of the general-`w` Grover oracle rely on (`X · CCZ · X`
    conjugation collapses the outer masks). -/
theorem embed_single_qubit_x_involutive (n : ℕ) (q : Fin n) :
    embed_single_qubit n q Gates.X * embed_single_qubit n q Gates.X = 1 :=
  embed_single_qubit_involutive n q Gates.X Gates.X_self_inverse

/-- **The embedded Hadamard is an involution on every wire** (`H·H = I`). The
    per-wire fact underlying `H^⊗n` involutivity, needed for the Grover
    *diffusion* bridge (`H^⊗3 · (I−2|0⟩⟨0|) · H^⊗3 = I−2|s⟩⟨s|`). -/
theorem embed_single_qubit_h_involutive (n : ℕ) (q : Fin n) :
    embed_single_qubit n q Gates.H * embed_single_qubit n q Gates.H = 1 :=
  embed_single_qubit_involutive n q Gates.H Gates.H_self_inverse

/-- **Disjoint-wire composition (keystone).** Two single-qubit embeddings on
    *distinct* wires `a ≠ b` compose to a single two-qubit embedding carrying the
    Kronecker pattern `U ⊗ V` (gate index packed as `a·2 + b`). Unlike
    `embed_single_qubit_mul_same`, the product sum collapses to a **unique**
    surviving index `k` (not a `Fin 2` fiber): the term at `(i, k)·(k, j)`
    survives only when `k` agrees with `i` off wire `a` *and* with `j` off wire
    `b`, forcing `k`'s `a`-bit `= j`'s, `b`-bit `= i`'s, rest `= i`'s rest (which
    must equal `j`'s rest). This is the brick that turns the `H`-layer
    `embed a H · embed b H · …` into a genuine tensor, unblocking the Grover
    *diffusion* bridge. -/
theorem embed_single_qubit_disjoint_mul (n : ℕ) (a b : Fin n) (hab : a ≠ b)
    (U V : Matrix (Fin 2) (Fin 2) ℂ) :
    embed_single_qubit n a U * embed_single_qubit n b V
      = embed_two_qubit n a b
          (fun r c => U ⟨r.val / 2, by omega⟩ ⟨c.val / 2, by omega⟩
                    * V ⟨r.val % 2, by omega⟩ ⟨c.val % 2, by omega⟩) := by
  have hab' : a.val ≠ b.val := fun h => hab (Fin.ext h)
  have hA : n - 1 - a.val < n := by have := a.isLt; omega
  have hB : n - 1 - b.val < n := by have := b.isLt; omega
  have hAB : (n - 1 - a.val) ≠ (n - 1 - b.val) := by
    have h0 := a.isLt; have h1 := b.isLt; omega
  ext i j
  rw [Matrix.mul_apply, embed_two_qubit_apply]
  simp only [embed_single_qubit_apply]
  set A := n - 1 - a.val with hAdef
  set B := n - 1 - b.val with hBdef
  have hia : (i.val / 2 ^ A) % 2 < 2 := by omega
  have hib : (i.val / 2 ^ B) % 2 < 2 := by omega
  have hja : (j.val / 2 ^ A) % 2 < 2 := by omega
  have hjb : (j.val / 2 ^ B) % 2 < 2 := by omega
  by_cases hR :
      i.val - (i.val / 2 ^ A) % 2 * 2 ^ A - (i.val / 2 ^ B) % 2 * 2 ^ B
    = j.val - (j.val / 2 ^ A) % 2 * 2 ^ A - (j.val / 2 ^ B) % 2 * 2 ^ B
  · rw [if_pos hR]
    obtain ⟨hriA, hriB⟩ := rest_bits_zero i.val A B hAB
    have hi_rest_lt :
        i.val - (i.val / 2 ^ A) % 2 * 2 ^ A - (i.val / 2 ^ B) % 2 * 2 ^ B < 2 ^ n := by
      have := two_bit_bound n a b hab' i.val; omega
    set R := i.val - (i.val / 2 ^ A) % 2 * 2 ^ A - (i.val / 2 ^ B) % 2 * 2 ^ B with hRdef
    have hk0lt : R + (j.val / 2 ^ A) % 2 * 2 ^ A + (i.val / 2 ^ B) % 2 * 2 ^ B < 2 ^ n := by
      have h1 := set_bit_lt R ((j.val / 2 ^ A) % 2) A n hi_rest_lt hA hriA hja
      have hbB : ((R + (j.val / 2 ^ A) % 2 * 2 ^ A) / 2 ^ B) % 2 = 0 := by
        rw [add_bit_other R ((j.val / 2 ^ A) % 2) B A (Ne.symm hAB) hja hriA]; exact hriB
      exact set_bit_lt _ ((i.val / 2 ^ B) % 2) B n h1 hB hbB hib
    set k0 : Fin (2 ^ n) :=
      ⟨R + (j.val / 2 ^ A) % 2 * 2 ^ A + (i.val / 2 ^ B) % 2 * 2 ^ B, hk0lt⟩ with hk0def
    have hk0a : (k0.val / 2 ^ A) % 2 = (j.val / 2 ^ A) % 2 :=
      (bit_of_set R ((j.val / 2 ^ A) % 2) ((i.val / 2 ^ B) % 2) A B hAB hja hib hriA hriB).1
    have hk0b : (k0.val / 2 ^ B) % 2 = (i.val / 2 ^ B) % 2 :=
      (bit_of_set R ((j.val / 2 ^ A) % 2) ((i.val / 2 ^ B) % 2) A B hAB hja hib hriA hriB).2
    rw [Finset.sum_eq_single k0]
    · have hk0v : k0.val
          = R + (j.val / 2 ^ A) % 2 * 2 ^ A + (i.val / 2 ^ B) % 2 * 2 ^ B := rfl
      have hcondA : i.val - (i.val / 2 ^ A) % 2 * 2 ^ A
          = k0.val - (k0.val / 2 ^ A) % 2 * 2 ^ A := by
        have hbi := two_bit_bound n a b hab' i.val
        rw [← hAdef, ← hBdef] at hbi
        rw [hk0a, hk0v]; omega
      have hcondB : k0.val - (k0.val / 2 ^ B) % 2 * 2 ^ B
          = j.val - (j.val / 2 ^ B) % 2 * 2 ^ B := by
        have hbj := two_bit_bound n a b hab' j.val
        rw [← hAdef, ← hBdef] at hbj
        rw [hk0b, hk0v]; omega
      rw [if_pos hcondA, if_pos hcondB]
      -- the matrix-entry indices retain the raw `n-1-_` exponents (`set` cannot
      -- fold inside the dependent `Fin.mk` proof positions), so unfold `A`/`B`.
      rw [hAdef] at hk0a; rw [hBdef] at hk0b
      simp only [hk0a, hk0b,
        show ((i.val / 2 ^ (n-1-a.val)) % 2 * 2 + (i.val / 2 ^ (n-1-b.val)) % 2) / 2
            = (i.val / 2 ^ (n-1-a.val)) % 2 from by omega,
        show ((i.val / 2 ^ (n-1-a.val)) % 2 * 2 + (i.val / 2 ^ (n-1-b.val)) % 2) % 2
            = (i.val / 2 ^ (n-1-b.val)) % 2 from by omega,
        show ((j.val / 2 ^ (n-1-a.val)) % 2 * 2 + (j.val / 2 ^ (n-1-b.val)) % 2) / 2
            = (j.val / 2 ^ (n-1-a.val)) % 2 from by omega,
        show ((j.val / 2 ^ (n-1-a.val)) % 2 * 2 + (j.val / 2 ^ (n-1-b.val)) % 2) % 2
            = (j.val / 2 ^ (n-1-b.val)) % 2 from by omega]
    · intro k _ hk
      by_cases hcA : i.val - (i.val / 2 ^ A) % 2 * 2 ^ A
          = k.val - (k.val / 2 ^ A) % 2 * 2 ^ A
      · by_cases hcB : k.val - (k.val / 2 ^ B) % 2 * 2 ^ B
            = j.val - (j.val / 2 ^ B) % 2 * 2 ^ B
        · exfalso; apply hk
          have hka : (k.val / 2 ^ A) % 2 = (j.val / 2 ^ A) % 2 := by
            have e1 := clear_bit_other k.val A B hAB
            have e2 := clear_bit_other j.val A B hAB
            rw [hcB] at e1; rw [e2] at e1; exact e1.symm
          have hkb : (k.val / 2 ^ B) % 2 = (i.val / 2 ^ B) % 2 := by
            have e1 := clear_bit_other k.val B A (Ne.symm hAB)
            have e2 := clear_bit_other i.val B A (Ne.symm hAB)
            rw [← hcA] at e1; rw [e2] at e1; exact e1.symm
          have hPa : (k.val / 2 ^ A) % 2 * 2 ^ A = (j.val / 2 ^ A) % 2 * 2 ^ A := by rw [hka]
          have hbk := two_bit_bound n a b hab' k.val
          have hbi := two_bit_bound n a b hab' i.val
          rw [← hAdef, ← hBdef] at hbk hbi
          apply Fin.ext
          show k.val = R + (j.val / 2 ^ A) % 2 * 2 ^ A + (i.val / 2 ^ B) % 2 * 2 ^ B
          omega
        · rw [if_neg hcB, mul_zero]
      · rw [if_neg hcA, zero_mul]
    · intro h; exact absurd (Finset.mem_univ k0) h
  · rw [if_neg hR]
    apply Finset.sum_eq_zero
    intro k _
    by_cases hcA : i.val - (i.val / 2 ^ A) % 2 * 2 ^ A
        = k.val - (k.val / 2 ^ A) % 2 * 2 ^ A
    · by_cases hcB : k.val - (k.val / 2 ^ B) % 2 * 2 ^ B
          = j.val - (j.val / 2 ^ B) % 2 * 2 ^ B
      · exfalso; apply hR
        have hka : (k.val / 2 ^ A) % 2 = (j.val / 2 ^ A) % 2 := by
          have e1 := clear_bit_other k.val A B hAB
          have e2 := clear_bit_other j.val A B hAB
          rw [hcB] at e1; rw [e2] at e1; exact e1.symm
        have hkb : (k.val / 2 ^ B) % 2 = (i.val / 2 ^ B) % 2 := by
          have e1 := clear_bit_other k.val B A (Ne.symm hAB)
          have e2 := clear_bit_other i.val B A (Ne.symm hAB)
          rw [← hcA] at e1; rw [e2] at e1; exact e1.symm
        have hPa : (k.val / 2 ^ A) % 2 * 2 ^ A = (j.val / 2 ^ A) % 2 * 2 ^ A := by rw [hka]
        have hPb : (k.val / 2 ^ B) % 2 * 2 ^ B = (i.val / 2 ^ B) % 2 * 2 ^ B := by rw [hkb]
        have hbk := two_bit_bound n a b hab' k.val
        have hbi := two_bit_bound n a b hab' i.val
        have hbj := two_bit_bound n a b hab' j.val
        rw [← hAdef, ← hBdef] at hbk hbi hbj
        omega
      · rw [if_neg hcB, mul_zero]
    · rw [if_neg hcA, zero_mul]

/-- **Disjoint-wire gates commute.** Single-qubit embeddings on distinct wires
    `a ≠ b` commute: `embed a U · embed b V = embed b V · embed a U`. Immediate
    from the disjoint-composition keystone — both products are the same two-qubit
    embedding (the rest-condition is symmetric in `a, b` since clearing two bits
    is order-independent, and the surviving entry `U·V` reorders to `V·U` by
    commutativity of `ℂ`). The rearrangement law the `H^⊗n` layer needs. -/
theorem embed_single_qubit_disjoint_comm (n : ℕ) (a b : Fin n) (hab : a ≠ b)
    (U V : Matrix (Fin 2) (Fin 2) ℂ) :
    embed_single_qubit n a U * embed_single_qubit n b V
      = embed_single_qubit n b V * embed_single_qubit n a U := by
  rw [embed_single_qubit_disjoint_mul n a b hab U V,
      embed_single_qubit_disjoint_mul n b a (Ne.symm hab) V U]
  ext i j
  simp only [embed_two_qubit_apply]
  have hi : i.val - (i.val / 2 ^ (n-1-b.val)) % 2 * 2 ^ (n-1-b.val)
              - (i.val / 2 ^ (n-1-a.val)) % 2 * 2 ^ (n-1-a.val)
          = i.val - (i.val / 2 ^ (n-1-a.val)) % 2 * 2 ^ (n-1-a.val)
              - (i.val / 2 ^ (n-1-b.val)) % 2 * 2 ^ (n-1-b.val) := by omega
  have hj : j.val - (j.val / 2 ^ (n-1-b.val)) % 2 * 2 ^ (n-1-b.val)
              - (j.val / 2 ^ (n-1-a.val)) % 2 * 2 ^ (n-1-a.val)
          = j.val - (j.val / 2 ^ (n-1-a.val)) % 2 * 2 ^ (n-1-a.val)
              - (j.val / 2 ^ (n-1-b.val)) % 2 * 2 ^ (n-1-b.val) := by omega
  rw [hi, hj]
  split_ifs with h
  · simp only [
      show ((i.val/2^(n-1-a.val))%2*2+(i.val/2^(n-1-b.val))%2)/2
          = (i.val/2^(n-1-a.val))%2 from by omega,
      show ((i.val/2^(n-1-a.val))%2*2+(i.val/2^(n-1-b.val))%2)%2
          = (i.val/2^(n-1-b.val))%2 from by omega,
      show ((j.val/2^(n-1-a.val))%2*2+(j.val/2^(n-1-b.val))%2)/2
          = (j.val/2^(n-1-a.val))%2 from by omega,
      show ((j.val/2^(n-1-a.val))%2*2+(j.val/2^(n-1-b.val))%2)%2
          = (j.val/2^(n-1-b.val))%2 from by omega,
      show ((i.val/2^(n-1-b.val))%2*2+(i.val/2^(n-1-a.val))%2)/2
          = (i.val/2^(n-1-b.val))%2 from by omega,
      show ((i.val/2^(n-1-b.val))%2*2+(i.val/2^(n-1-a.val))%2)%2
          = (i.val/2^(n-1-a.val))%2 from by omega,
      show ((j.val/2^(n-1-b.val))%2*2+(j.val/2^(n-1-a.val))%2)/2
          = (j.val/2^(n-1-b.val))%2 from by omega,
      show ((j.val/2^(n-1-b.val))%2*2+(j.val/2^(n-1-a.val))%2)%2
          = (j.val/2^(n-1-a.val))%2 from by omega]
    ring
  · rfl

/-! ### `X`-mask conjugation of a diagonal (general-`w` Grover oracle core)

An `X` gate on wire `q` is the permutation that flips the `q`-bit. Conjugating a
diagonal matrix by it permutes the diagonal entries along that flip. The Grover
oracle for an arbitrary marked item `w` is exactly `X-mask · CCZ · X-mask`, so
this conjugation (iterated over the masked wires) is the bridge from the
canonical `|1…1⟩` oracle to the general-`w` one. -/

/-- Flip the `q`-bit of a raw index (raw form of the `X`-gate permutation). -/
def flipBitVal (n : ℕ) (q : Fin n) (x : ℕ) : ℕ :=
  if (x / 2 ^ (n - 1 - q.val)) % 2 = 0 then x + 2 ^ (n - 1 - q.val)
  else x - 2 ^ (n - 1 - q.val)

private lemma flipBitVal_lt (n : ℕ) (q : Fin n) (x : Fin (2 ^ n)) :
    flipBitVal n q x.val < 2 ^ n := by
  have hA : n - 1 - q.val < n := by have := q.isLt; omega
  unfold flipBitVal
  split
  · rename_i hb
    have h := set_bit_lt x.val 1 (n - 1 - q.val) n x.isLt hA hb (by norm_num)
    omega
  · have := Nat.sub_le x.val (2 ^ (n - 1 - q.val)); have := x.isLt; omega

/-- Flipping the `q`-bit toggles it: `bit(flip x) = 1 − bit(x)`. -/
private lemma flipBitVal_bit (n : ℕ) (q : Fin n) (x : ℕ) :
    (flipBitVal n q x / 2 ^ (n - 1 - q.val)) % 2 = 1 - (x / 2 ^ (n - 1 - q.val)) % 2 := by
  have hxbit : (x / 2 ^ (n - 1 - q.val)) % 2 < 2 := by omega
  unfold flipBitVal
  by_cases hb : (x / 2 ^ (n - 1 - q.val)) % 2 = 0
  · rw [if_pos hb, hb]
    have := add_bit_self x 1 (n - 1 - q.val) (by norm_num) hb
    simpa using this
  · rw [if_neg hb]
    have hb1 : (x / 2 ^ (n - 1 - q.val)) % 2 = 1 := by omega
    have hc := clear_bit_self x (n - 1 - q.val)
    rw [hb1] at hc
    rw [hb1]
    simpa using hc

/-- Flipping the `q`-bit leaves the other bits (the `rest`) unchanged. -/
private lemma flipBitVal_rest (n : ℕ) (q : Fin n) (x : ℕ) :
    flipBitVal n q x - (flipBitVal n q x / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
      = x - (x / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val) := by
  have hbitf := flipBitVal_bit n q x
  have hxbit : (x / 2 ^ (n - 1 - q.val)) % 2 < 2 := by omega
  rw [hbitf]
  by_cases hb : (x / 2 ^ (n - 1 - q.val)) % 2 = 0
  · have hflip : flipBitVal n q x = x + 2 ^ (n - 1 - q.val) := by
      unfold flipBitVal; rw [if_pos hb]
    rw [hflip, hb]; omega
  · have hb1 : (x / 2 ^ (n - 1 - q.val)) % 2 = 1 := by omega
    have hflip : flipBitVal n q x = x - 2 ^ (n - 1 - q.val) := by
      unfold flipBitVal; rw [if_neg hb]
    have hge : 2 ^ (n - 1 - q.val) ≤ x := by
      have h : (x / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val) ≤ x :=
        le_trans (mul_le_mul_right' (Nat.mod_le _ 2) _) (Nat.div_mul_le_self _ _)
      rw [hb1] at h; simpa using h
    rw [hflip, hb1]; omega

/-- The `q`-bit-flip permutation on `Fin (2^n)` (the `X`-gate action). -/
def flipBit (n : ℕ) (q : Fin n) (x : Fin (2 ^ n)) : Fin (2 ^ n) :=
  ⟨flipBitVal n q x.val, flipBitVal_lt n q x⟩

/-- `Gates.X` entrywise: `X a b = 1` off the diagonal, `0` on it. -/
private lemma Gates_X_apply (a b : Fin 2) :
    Gates.X a b = if a = b then 0 else 1 := by
  fin_cases a <;> fin_cases b <;> simp [Gates.X]

/-- **The embedded `X` gate is the bit-flip permutation matrix:**
    `embed q X i j = 1` iff `j = flipBit i`, else `0`. -/
theorem embed_single_qubit_X_apply (n : ℕ) (q : Fin n) (i j : Fin (2 ^ n)) :
    embed_single_qubit n q Gates.X i j = if j = flipBit n q i then 1 else 0 := by
  rw [embed_single_qubit_apply, Gates_X_apply]
  by_cases hj : j = flipBit n q i
  · rw [if_pos hj]
    subst hj
    have hrest : i.val - (i.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
        = (flipBit n q i).val
          - ((flipBit n q i).val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val) :=
      (flipBitVal_rest n q i.val).symm
    have hbit : ¬ ((⟨(i.val / 2 ^ (n - 1 - q.val)) % 2, by omega⟩ : Fin 2)
        = ⟨((flipBit n q i).val / 2 ^ (n - 1 - q.val)) % 2, by omega⟩) := by
      have hfb : ((flipBit n q i).val / 2 ^ (n - 1 - q.val)) % 2
          = 1 - (i.val / 2 ^ (n - 1 - q.val)) % 2 := flipBitVal_bit n q i.val
      intro h
      rw [Fin.mk.injEq, hfb] at h
      omega
    rw [if_pos hrest, if_neg hbit]
  · rw [if_neg hj]
    by_cases hr : i.val - (i.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
        = j.val - (j.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
    · rw [if_pos hr]
      by_cases hbq : (⟨(i.val / 2 ^ (n - 1 - q.val)) % 2, by omega⟩ : Fin 2)
          = ⟨(j.val / 2 ^ (n - 1 - q.val)) % 2, by omega⟩
      · rw [if_pos hbq]
      · exfalso; apply hj; apply Fin.ext
        show j.val = flipBitVal n q i.val
        have hne : (i.val / 2 ^ (n - 1 - q.val)) % 2 ≠ (j.val / 2 ^ (n - 1 - q.val)) % 2 :=
          fun h => hbq (Fin.ext h)
        have hile : (i.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val) ≤ i.val :=
          le_trans (mul_le_mul_right' (Nat.mod_le _ _) _) (Nat.div_mul_le_self _ _)
        have hjle : (j.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val) ≤ j.val :=
          le_trans (mul_le_mul_right' (Nat.mod_le _ _) _) (Nat.div_mul_le_self _ _)
        have hiq2 : (i.val / 2 ^ (n - 1 - q.val)) % 2 = 0
            ∨ (i.val / 2 ^ (n - 1 - q.val)) % 2 = 1 := by omega
        have hjq2 : (j.val / 2 ^ (n - 1 - q.val)) % 2 = 0
            ∨ (j.val / 2 ^ (n - 1 - q.val)) % 2 = 1 := by omega
        unfold flipBitVal
        rcases hiq2 with hi0 | hi1
        · rw [if_pos hi0]
          rcases hjq2 with hj0 | hj1
          · exact absurd (hi0.trans hj0.symm) hne
          · rw [hi0] at hr hile; rw [hj1] at hr hjle; omega
        · rw [if_neg (by omega)]
          rcases hjq2 with hj0 | hj1
          · rw [hi1] at hr hile; rw [hj0] at hr hjle; omega
          · exact absurd (hi1.trans hj1.symm) hne
    · rw [if_neg hr]

/-- **Intertwining relation:** `embed q X · diag d = diag (d ∘ flipBit) · embed q X`.
    The `X`-permutation slides a diagonal across, relabelling it by the bit-flip. -/
theorem embed_single_qubit_X_intertwine (n : ℕ) (q : Fin n) (d : Fin (2 ^ n) → ℂ) :
    embed_single_qubit n q Gates.X * Matrix.diagonal d
      = Matrix.diagonal (fun i => d (flipBit n q i)) * embed_single_qubit n q Gates.X := by
  ext i j
  rw [Matrix.mul_diagonal, Matrix.diagonal_mul, embed_single_qubit_X_apply]
  by_cases hj : j = flipBit n q i
  · rw [if_pos hj, hj]; ring
  · rw [if_neg hj]; ring

/-- **`X`-mask conjugation of a diagonal:** `embed q X · diag d · embed q X
    = diag (d ∘ flipBit)`. Conjugating by the `X`-permutation relabels the
    diagonal along the `q`-bit flip. With `d` the `CCZ`/oracle diagonal this is
    the single-wire step of the general-`w` Grover oracle bridge. -/
theorem embed_single_qubit_X_conj_diagonal (n : ℕ) (q : Fin n) (d : Fin (2 ^ n) → ℂ) :
    embed_single_qubit n q Gates.X * Matrix.diagonal d * embed_single_qubit n q Gates.X
      = Matrix.diagonal (fun i => d (flipBit n q i)) := by
  rw [embed_single_qubit_X_intertwine, mul_assoc, embed_single_qubit_x_involutive, mul_one]

/-- **Two-wire `X`-mask conjugation (palindrome order).** Conjugating a diagonal
    by `X`-gates on wires `a, b` relabels it by both bit-flips. The palindrome
    `Xₐ·X_b·D·X_b·Xₐ` telescopes into two single-wire conjugations — no
    disjoint-wire commutation needed. -/
theorem embed_single_qubit_X_conj2_diagonal (n : ℕ) (a b : Fin n) (d : Fin (2 ^ n) → ℂ) :
    embed_single_qubit n a Gates.X * embed_single_qubit n b Gates.X * Matrix.diagonal d
        * embed_single_qubit n b Gates.X * embed_single_qubit n a Gates.X
      = Matrix.diagonal (fun i => d (flipBit n b (flipBit n a i))) := by
  have key : embed_single_qubit n a Gates.X * embed_single_qubit n b Gates.X * Matrix.diagonal d
        * embed_single_qubit n b Gates.X * embed_single_qubit n a Gates.X
      = embed_single_qubit n a Gates.X
        * (embed_single_qubit n b Gates.X * Matrix.diagonal d * embed_single_qubit n b Gates.X)
        * embed_single_qubit n a Gates.X := by
    simp only [mul_assoc]
  rw [key, embed_single_qubit_X_conj_diagonal, embed_single_qubit_X_conj_diagonal]

/-- **Three-wire `X`-mask conjugation (palindrome order).** The palindrome
    `Xₐ·X_b·X_c·D·X_c·X_b·Xₐ` telescopes into three single-wire conjugations. -/
theorem embed_single_qubit_X_conj3_diagonal (n : ℕ) (a b c : Fin n) (d : Fin (2 ^ n) → ℂ) :
    embed_single_qubit n a Gates.X * embed_single_qubit n b Gates.X
        * embed_single_qubit n c Gates.X * Matrix.diagonal d
        * embed_single_qubit n c Gates.X * embed_single_qubit n b Gates.X
        * embed_single_qubit n a Gates.X
      = Matrix.diagonal (fun i => d (flipBit n c (flipBit n b (flipBit n a i)))) := by
  have key : embed_single_qubit n a Gates.X * embed_single_qubit n b Gates.X
        * embed_single_qubit n c Gates.X * Matrix.diagonal d
        * embed_single_qubit n c Gates.X * embed_single_qubit n b Gates.X
        * embed_single_qubit n a Gates.X
      = embed_single_qubit n a Gates.X
        * (embed_single_qubit n b Gates.X
            * (embed_single_qubit n c Gates.X * Matrix.diagonal d * embed_single_qubit n c Gates.X)
            * embed_single_qubit n b Gates.X)
        * embed_single_qubit n a Gates.X := by
    simp only [mul_assoc]
  rw [key, embed_single_qubit_X_conj_diagonal, embed_single_qubit_X_conj_diagonal,
      embed_single_qubit_X_conj_diagonal]

/-- The index obtained by swapping the bits at qubit positions `q0, q1`. -/
def swapVal (n : ℕ) (q0 q1 : Fin n) (x : ℕ) : ℕ :=
  (x - (x/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (x/2^(n-1-q1.val))%2*2^(n-1-q1.val))
    + (x/2^(n-1-q1.val))%2*2^(n-1-q0.val) + (x/2^(n-1-q0.val))%2*2^(n-1-q1.val)

private lemma swapVal_lt (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val)
    (x : Fin (2^n)) : swapVal n q0 q1 x.val < 2^n := by
  have hA : n - 1 - q0.val < n := by have := q0.isLt; omega
  have hB : n - 1 - q1.val < n := by have := q1.isLt; omega
  have hAB : (n - 1 - q0.val) ≠ (n - 1 - q1.val) := by
    have h0 := q0.isLt; have h1 := q1.isLt; omega
  obtain ⟨hrA, hrB⟩ := rest_bits_zero x.val (n-1-q0.val) (n-1-q1.val) hAB
  have hrest_lt :
      x.val - (x.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (x.val/2^(n-1-q1.val))%2*2^(n-1-q1.val) < 2^n := by
    have := two_bit_bound n q0 q1 hq x.val; omega
  unfold swapVal
  have hc2 : (x.val/2^(n-1-q1.val))%2 < 2 := by omega
  have hc1 : (x.val/2^(n-1-q0.val))%2 < 2 := by omega
  have h1 := set_bit_lt _ ((x.val/2^(n-1-q1.val))%2) (n-1-q0.val) n hrest_lt hA hrA hc2
  have hbB : (((x.val - (x.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (x.val/2^(n-1-q1.val))%2*2^(n-1-q1.val))
      + (x.val/2^(n-1-q1.val))%2*2^(n-1-q0.val)) / 2^(n-1-q1.val)) % 2 = 0 := by
    rw [add_bit_other _ ((x.val/2^(n-1-q1.val))%2) (n-1-q1.val) (n-1-q0.val) (Ne.symm hAB) hc2 hrA]
    exact hrB
  exact set_bit_lt _ ((x.val/2^(n-1-q0.val))%2) (n-1-q1.val) n h1 hB hbB hc1

private lemma swapVal_involutive (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val)
    (x : Fin (2^n)) : swapVal n q0 q1 (swapVal n q0 q1 x.val) = x.val := by
  have hAB : (n - 1 - q0.val) ≠ (n - 1 - q1.val) := by
    have h0 := q0.isLt; have h1 := q1.isLt; omega
  obtain ⟨hrA, hrB⟩ := rest_bits_zero x.val (n-1-q0.val) (n-1-q1.val) hAB
  have hc1 : (x.val/2^(n-1-q0.val))%2 < 2 := by omega
  have hc2 : (x.val/2^(n-1-q1.val))%2 < 2 := by omega
  -- bits of `swapVal x`
  obtain ⟨hy0, hy1⟩ := bit_of_set
    (x.val - (x.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (x.val/2^(n-1-q1.val))%2*2^(n-1-q1.val))
    ((x.val/2^(n-1-q1.val))%2) ((x.val/2^(n-1-q0.val))%2)
    (n-1-q0.val) (n-1-q1.val) hAB hc2 hc1 hrA hrB
  have hbound := two_bit_bound n q0 q1 hq x.val
  unfold swapVal
  -- after unfolding both layers, rewrite the inner bits and reconstruct
  simp only [hy0, hy1]
  omega

/-- `SWAP4` is an involution. -/
private lemma SWAP4_mul_self : SWAP4 * SWAP4 = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SWAP4, Matrix.mul_apply, Fin.sum_univ_four, Matrix.one_apply]

/-- **A swap gate is involutive on the full space:**
    `embed_two_qubit n q0 q1 SWAP4` squared is the identity. Immediate from
    multiplicativity (`embed_two_qubit_mul_same`) + `SWAP4 * SWAP4 = 1` +
    `embed_two_qubit_one`. -/
theorem embed_swap_involutive (n : ℕ) (q0 q1 : Fin n) (hq : q0 ≠ q1) :
    embed_two_qubit n q0 q1 SWAP4 * embed_two_qubit n q0 q1 SWAP4 = 1 := by
  rw [embed_two_qubit_mul_same n q0 q1 hq SWAP4 SWAP4, SWAP4_mul_self,
      embed_two_qubit_one n q0 q1 hq]

/-- `SWAP4` entry at bit-pair indices: it swaps the two bits, so the `(a‖b,c‖d)`
    entry is `1` iff `(c,d) = (b,a)`. -/
private lemma SWAP4_apply (a b c d : ℕ) (ha : a < 2) (hb : b < 2) (hc : c < 2) (hd : d < 2) :
    SWAP4 ⟨a*2+b, by omega⟩ ⟨c*2+d, by omega⟩ = if c = b ∧ d = a then 1 else 0 := by
  interval_cases a <;> interval_cases b <;> interval_cases c <;> interval_cases d <;>
    simp [SWAP4]

/-- The bit-swap as a `Fin (2^n)` self-map. -/
private def swapFin (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) (x : Fin (2^n)) : Fin (2^n) :=
  ⟨swapVal n q0 q1 x.val, swapVal_lt n q0 q1 hq x⟩

private lemma swapFin_involutive (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) :
    Function.Involutive (swapFin n q0 q1 hq) :=
  fun x => Fin.ext (swapVal_involutive n q0 q1 hq x)

/-- The bit-swap as a permutation of `Fin (2^n)`. -/
noncomputable def swapPerm (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) :
    Equiv.Perm (Fin (2^n)) :=
  (swapFin_involutive n q0 q1 hq).toPerm (swapFin n q0 q1 hq)

lemma swapPerm_apply (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) (x : Fin (2^n)) :
    (swapPerm n q0 q1 hq x).val = swapVal n q0 q1 x.val := rfl

/-- **A swap gate denotes to the bit-swap permutation matrix:**
    `embed_two_qubit n q0 q1 SWAP4 = (swapPerm).permMatrix ℂ`. -/
theorem denote_swap_eq_permMatrix (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) :
    embed_two_qubit n q0 q1 SWAP4 = (swapPerm n q0 q1 hq).permMatrix ℂ := by
  have hAB : (n - 1 - q0.val) ≠ (n - 1 - q1.val) := by
    have h0 := q0.isLt; have h1 := q1.isLt; omega
  ext i j
  obtain ⟨hrA, hrB⟩ := rest_bits_zero i.val (n-1-q0.val) (n-1-q1.val) hAB
  have hiA : (i.val/2^(n-1-q0.val))%2 < 2 := by omega
  have hiB : (i.val/2^(n-1-q1.val))%2 < 2 := by omega
  have hjA : (j.val/2^(n-1-q0.val))%2 < 2 := by omega
  have hjB : (j.val/2^(n-1-q1.val))%2 < 2 := by omega
  have hjrec := two_bit_bound n q0 q1 hq j.val
  have hswval : swapVal n q0 q1 i.val
      = (i.val - (i.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (i.val/2^(n-1-q1.val))%2*2^(n-1-q1.val))
        + (i.val/2^(n-1-q1.val))%2*2^(n-1-q0.val) + (i.val/2^(n-1-q0.val))%2*2^(n-1-q1.val) := rfl
  obtain ⟨hsw0, hsw1⟩ := bit_of_set
    (i.val - (i.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (i.val/2^(n-1-q1.val))%2*2^(n-1-q1.val))
    ((i.val/2^(n-1-q1.val))%2) ((i.val/2^(n-1-q0.val))%2)
    (n-1-q0.val) (n-1-q1.val) hAB hiB hiA hrA hrB
  rw [← hswval] at hsw0 hsw1
  rw [embed_two_qubit_apply, SWAP4_apply _ _ _ _ hiA hiB hjA hjB,
      Equiv.Perm.permMatrix, PEquiv.toMatrix_apply, Equiv.toPEquiv_apply]
  have hmem : (j ∈ some (swapPerm n q0 q1 hq i)) ↔ j.val = swapVal n q0 q1 i.val := by
    rw [Option.mem_some_iff, eq_comm, Fin.ext_iff, swapPerm_apply]
  set X := i.val - (i.val/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (i.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)
    with hX
  by_cases hj : j.val = swapVal n q0 q1 i.val
  · rw [if_pos (hmem.mpr hj)]
    have e0 : (j.val/2^(n-1-q0.val))%2 = (i.val/2^(n-1-q1.val))%2 := by rw [hj]; exact hsw0
    have e1 : (j.val/2^(n-1-q1.val))%2 = (i.val/2^(n-1-q0.val))%2 := by rw [hj]; exact hsw1
    have hr : X = j.val - (j.val/2^(n-1-q0.val))%2*2^(n-1-q0.val)
                       - (j.val/2^(n-1-q1.val))%2*2^(n-1-q1.val) := by
      rw [e0, e1, hj, hswval]; omega
    rw [if_pos hr, if_pos ⟨e0, e1⟩]
  · rw [if_neg (fun h => hj (hmem.mp h))]
    by_cases hrest : X = j.val - (j.val/2^(n-1-q0.val))%2*2^(n-1-q0.val)
                           - (j.val/2^(n-1-q1.val))%2*2^(n-1-q1.val)
    · rw [if_pos hrest, if_neg]
      rintro ⟨hc, hd⟩
      apply hj
      rw [hswval, ← hc, ← hd]
      omega
    · rw [if_neg hrest]

/-- The bit-swap permutation is an involution. -/
private lemma swapPerm_involutive (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) :
    swapPerm n q0 q1 hq * swapPerm n q0 q1 hq = 1 := by
  ext x
  show swapVal n q0 q1 (swapVal n q0 q1 x.val) = x.val
  exact swapVal_involutive n q0 q1 hq x

/-- Bit extensionality: two values `< 2^N` agreeing on every bit `< N` are equal. -/
lemma eq_of_bits_lt (N : ℕ) : ∀ x y : ℕ, x < 2^N → y < 2^N →
    (∀ P, P < N → (x/2^P)%2 = (y/2^P)%2) → x = y := by
  induction N with
  | zero => intro x y hx hy _; simp only [pow_zero, Nat.lt_one_iff] at hx hy; omega
  | succ N ih =>
      intro x y hx hy h
      have hb0 : x % 2 = y % 2 := by have := h 0 (by omega); simpa using this
      have hpow : (2:ℕ)^(N+1) = 2^N * 2 := by rw [pow_succ]
      have hx2 : x / 2 < 2^N := by rw [Nat.div_lt_iff_lt_mul (by norm_num)]; omega
      have hy2 : y / 2 < 2^N := by rw [Nat.div_lt_iff_lt_mul (by norm_num)]; omega
      have hrec : x / 2 = y / 2 := by
        apply ih _ _ hx2 hy2
        intro P hP
        have key : ∀ z : ℕ, z / 2 / 2^P = z / 2^(P+1) := fun z => by
          rw [Nat.div_div_eq_div_mul]; congr 1; rw [pow_succ]; ring
        simp only [key]
        exact h (P+1) (by omega)
      omega

/-- Bit `q0` of `swapVal x` is bit `q1` of `x` (the swap). -/
lemma swapVal_bit_A (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) (x : ℕ) :
    (swapVal n q0 q1 x / 2^(n-1-q0.val)) % 2 = (x / 2^(n-1-q1.val)) % 2 := by
  have hAB : (n-1-q0.val) ≠ (n-1-q1.val) := by have h0:=q0.isLt; have h1:=q1.isLt; omega
  obtain ⟨hrA, hrB⟩ := rest_bits_zero x (n-1-q0.val) (n-1-q1.val) hAB
  exact (bit_of_set _ ((x/2^(n-1-q1.val))%2) ((x/2^(n-1-q0.val))%2)
    (n-1-q0.val) (n-1-q1.val) hAB (by omega) (by omega) hrA hrB).1

/-- Bit `q1` of `swapVal x` is bit `q0` of `x`. -/
lemma swapVal_bit_B (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) (x : ℕ) :
    (swapVal n q0 q1 x / 2^(n-1-q1.val)) % 2 = (x / 2^(n-1-q0.val)) % 2 := by
  have hAB : (n-1-q0.val) ≠ (n-1-q1.val) := by have h0:=q0.isLt; have h1:=q1.isLt; omega
  obtain ⟨hrA, hrB⟩ := rest_bits_zero x (n-1-q0.val) (n-1-q1.val) hAB
  exact (bit_of_set _ ((x/2^(n-1-q1.val))%2) ((x/2^(n-1-q0.val))%2)
    (n-1-q0.val) (n-1-q1.val) hAB (by omega) (by omega) hrA hrB).2

/-- `swapVal` leaves every bit other than `q0, q1` unchanged. -/
lemma swapVal_preserves_bit (n : ℕ) (q0 q1 : Fin n) (hq : q0.val ≠ q1.val) (x P : ℕ)
    (hPA : P ≠ n-1-q0.val) (hPB : P ≠ n-1-q1.val) :
    (swapVal n q0 q1 x / 2^P) % 2 = (x / 2^P) % 2 := by
  have hAB : (n-1-q0.val) ≠ (n-1-q1.val) := by have h0:=q0.isLt; have h1:=q1.isLt; omega
  obtain ⟨hrA, hrB⟩ := rest_bits_zero x (n-1-q0.val) (n-1-q1.val) hAB
  set rest := x - (x/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (x/2^(n-1-q1.val))%2*2^(n-1-q1.val) with hrest
  -- bit B of (rest + bitB·2^A) is 0
  have hbBu : ((rest + (x/2^(n-1-q1.val))%2*2^(n-1-q0.val)) / 2^(n-1-q1.val)) % 2 = 0 := by
    rw [add_bit_other rest ((x/2^(n-1-q1.val))%2) (n-1-q1.val) (n-1-q0.val)
        (Ne.symm hAB) (by omega) hrA]
    exact hrB
  -- peel the two added bits (at A then B don't touch position P)
  have hstep : (swapVal n q0 q1 x / 2^P) % 2 = (rest / 2^P) % 2 := by
    unfold swapVal
    rw [← hrest,
        add_bit_other (rest + (x/2^(n-1-q1.val))%2*2^(n-1-q0.val)) ((x/2^(n-1-q0.val))%2)
          P (n-1-q1.val) hPB (by omega) hbBu,
        add_bit_other rest ((x/2^(n-1-q1.val))%2) P (n-1-q0.val) hPA (by omega) hrA]
  rw [hstep, hrest]
  -- (rest / 2^P) % 2 = (x / 2^P) % 2: clearing bits A,B preserves P
  have hbB_u1 : (x/2^(n-1-q1.val))%2
      = ((x - (x/2^(n-1-q0.val))%2*2^(n-1-q0.val))/2^(n-1-q1.val))%2 :=
    (clear_bit_other x (n-1-q1.val) (n-1-q0.val) (Ne.symm hAB)).symm
  calc (x - (x/2^(n-1-q0.val))%2*2^(n-1-q0.val) - (x/2^(n-1-q1.val))%2*2^(n-1-q1.val)) / 2^P % 2
      = ((x - (x/2^(n-1-q0.val))%2*2^(n-1-q0.val))
          - ((x - (x/2^(n-1-q0.val))%2*2^(n-1-q0.val))/2^(n-1-q1.val))%2*2^(n-1-q1.val))
          / 2^P % 2 := by rw [hbB_u1]
    _ = ((x - (x/2^(n-1-q0.val))%2*2^(n-1-q0.val))/2^P)%2 :=
        clear_bit_other (x - (x/2^(n-1-q0.val))%2*2^(n-1-q0.val)) P (n-1-q1.val) hPB
    _ = (x/2^P)%2 := clear_bit_other x P (n-1-q0.val) hPA

/-- Two disjoint bit-swaps commute (on indices). -/
private lemma swapVal_comm (n : ℕ) (a b c d : Fin n) (hab : a.val ≠ b.val) (hcd : c.val ≠ d.val)
    (hAC : n-1-a.val ≠ n-1-c.val) (hAD : n-1-a.val ≠ n-1-d.val)
    (hBC : n-1-b.val ≠ n-1-c.val) (hBD : n-1-b.val ≠ n-1-d.val)
    (x : ℕ) (hx : x < 2^n) :
    swapVal n a b (swapVal n c d x) = swapVal n c d (swapVal n a b x) := by
  apply eq_of_bits_lt n
  · exact swapVal_lt n a b hab ⟨_, swapVal_lt n c d hcd ⟨x, hx⟩⟩
  · exact swapVal_lt n c d hcd ⟨_, swapVal_lt n a b hab ⟨x, hx⟩⟩
  intro P _
  by_cases hPa : P = n-1-a.val
  · subst hPa
    rw [swapVal_bit_A n a b hab (swapVal n c d x),
        swapVal_preserves_bit n c d hcd x (n-1-b.val) hBC hBD,
        swapVal_preserves_bit n c d hcd (swapVal n a b x) (n-1-a.val) hAC hAD,
        swapVal_bit_A n a b hab x]
  · by_cases hPb : P = n-1-b.val
    · subst hPb
      rw [swapVal_bit_B n a b hab (swapVal n c d x),
          swapVal_preserves_bit n c d hcd x (n-1-a.val) hAC hAD,
          swapVal_preserves_bit n c d hcd (swapVal n a b x) (n-1-b.val) hBC hBD,
          swapVal_bit_B n a b hab x]
    · by_cases hPc : P = n-1-c.val
      · subst hPc
        rw [swapVal_preserves_bit n a b hab (swapVal n c d x) (n-1-c.val)
              (Ne.symm hAC) (Ne.symm hBC),
            swapVal_bit_A n c d hcd x,
            swapVal_bit_A n c d hcd (swapVal n a b x),
            swapVal_preserves_bit n a b hab x (n-1-d.val) (Ne.symm hAD) (Ne.symm hBD)]
      · by_cases hPd : P = n-1-d.val
        · subst hPd
          rw [swapVal_preserves_bit n a b hab (swapVal n c d x) (n-1-d.val)
                (Ne.symm hAD) (Ne.symm hBD),
              swapVal_bit_B n c d hcd x,
              swapVal_bit_B n c d hcd (swapVal n a b x),
              swapVal_preserves_bit n a b hab x (n-1-c.val) (Ne.symm hAC) (Ne.symm hBC)]
        · rw [swapVal_preserves_bit n a b hab (swapVal n c d x) P hPa hPb,
              swapVal_preserves_bit n c d hcd x P hPc hPd,
              swapVal_preserves_bit n c d hcd (swapVal n a b x) P hPc hPd,
              swapVal_preserves_bit n a b hab x P hPa hPb]

private lemma swapPerm_comm (n : ℕ) (a b c d : Fin n) (hab : a.val ≠ b.val) (hcd : c.val ≠ d.val)
    (hAC : n-1-a.val ≠ n-1-c.val) (hAD : n-1-a.val ≠ n-1-d.val)
    (hBC : n-1-b.val ≠ n-1-c.val) (hBD : n-1-b.val ≠ n-1-d.val) :
    swapPerm n a b hab * swapPerm n c d hcd = swapPerm n c d hcd * swapPerm n a b hab := by
  ext x
  exact swapVal_comm n a b c d hab hcd hAC hAD hBC hBD x.val x.isLt

/-- **Disjoint swap gates commute.** -/
theorem denote_swap_comm (n : ℕ) (a b c d : Fin n) (hab : a ≠ b) (hcd : c ≠ d)
    (hAC : n-1-a.val ≠ n-1-c.val) (hAD : n-1-a.val ≠ n-1-d.val)
    (hBC : n-1-b.val ≠ n-1-c.val) (hBD : n-1-b.val ≠ n-1-d.val) :
    denote_gate (GateApp.swap a b hab) * denote_gate (GateApp.swap c d hcd)
      = denote_gate (GateApp.swap c d hcd) * denote_gate (GateApp.swap a b hab) := by
  have hab' : a.val ≠ b.val := fun h => hab (Fin.ext h)
  have hcd' : c.val ≠ d.val := fun h => hcd (Fin.ext h)
  show embed_two_qubit n a b SWAP4 * embed_two_qubit n c d SWAP4
     = embed_two_qubit n c d SWAP4 * embed_two_qubit n a b SWAP4
  rw [denote_swap_eq_permMatrix n a b hab', denote_swap_eq_permMatrix n c d hcd',
      ← Matrix.permMatrix_mul, ← Matrix.permMatrix_mul,
      swapPerm_comm n a b c d hab' hcd' hAC hAD hBC hBD]

/-- A gate that commutes with every gate of a circuit commutes with its denotation. -/
theorem gate_comm_denote {n : ℕ} (g : GateApp n) : ∀ (c : Circuit n),
    (∀ h ∈ c, denote_gate g * denote_gate h = denote_gate h * denote_gate g) →
    denote_gate g * denote c = denote c * denote_gate g := by
  intro c
  induction c with
  | nil => intro _; rw [empty_circuit_identity, mul_one, one_mul]
  | cons h rest ih =>
      intro hc
      rw [denote_cons, ← mul_assoc,
          ih (fun h' hh' => hc h' (List.mem_cons_of_mem _ hh')),
          mul_assoc, hc h List.mem_cons_self, ← mul_assoc]

/-- A circuit of pairwise-commuting involutive gates squares to the identity. -/
theorem denote_sq_eq_one {n : ℕ} : ∀ (c : Circuit n),
    (∀ g ∈ c, denote_gate g * denote_gate g = 1) →
    (∀ g ∈ c, ∀ h ∈ c, denote_gate g * denote_gate h = denote_gate h * denote_gate g) →
    denote c * denote c = 1 := by
  intro c
  induction c with
  | nil => intro _ _; rw [empty_circuit_identity, mul_one]
  | cons g rest ih =>
      intro hinv hcomm
      have hgr : denote_gate g * denote rest = denote rest * denote_gate g :=
        gate_comm_denote g rest
          (fun h hh => hcomm g List.mem_cons_self h (List.mem_cons_of_mem _ hh))
      rw [denote_cons]
      calc denote rest * denote_gate g * (denote rest * denote_gate g)
          = denote rest * (denote_gate g * denote rest) * denote_gate g := by
            rw [mul_assoc, mul_assoc, ← mul_assoc (denote_gate g)]
        _ = denote rest * (denote rest * denote_gate g) * denote_gate g := by rw [hgr]
        _ = (denote rest * denote rest) * (denote_gate g * denote_gate g) := by
            rw [mul_assoc, mul_assoc, mul_assoc]
        _ = 1 := by
            rw [ih (fun g' hg' => hinv g' (List.mem_cons_of_mem _ hg'))
                  (fun g' hg' h' hh' => hcomm g' (List.mem_cons_of_mem _ hg')
                    h' (List.mem_cons_of_mem _ hh')),
                hinv g List.mem_cons_self, mul_one]

/-- **Denotation of a circuit of permutation gates is the permutation matrix of
    the (list-ordered) product of the permutations.** If every gate `g` of `c`
    denotes to `(σ g).permMatrix`, then `denote c = ((c.map σ).prod).permMatrix`.
    Folds `permMatrix_mul` (the anti-homomorphism `(α·β).pm = β.pm·α.pm`) over the
    list; the left-multiplying `denote` fold lines up exactly with `List.prod`. -/
theorem denote_eq_permMatrix_prod {n : ℕ} (σ : GateApp n → Equiv.Perm (Fin (2^n))) :
    ∀ (c : Circuit n), (∀ g ∈ c, denote_gate g = (σ g).permMatrix ℂ) →
      denote c = ((c.map σ).prod).permMatrix ℂ := by
  intro c
  induction c with
  | nil =>
      intro _
      rw [empty_circuit_identity, List.map_nil, List.prod_nil, Matrix.permMatrix_one]
  | cons g rest ih =>
      intro h
      rw [denote_cons, ih (fun g' hg' => h g' (List.mem_cons_of_mem _ hg')),
          h g List.mem_cons_self, ← Matrix.permMatrix_mul, List.map_cons, List.prod_cons]

-- ===========================================================================
-- Compositional unitarity.
--
-- Goal: `denote c ∈ unitary _` for EVERY circuit `c`, with no per-circuit
-- explicit-matrix computation (which does not scale past 4×4 — see the
-- `ghz_unitary` deferral note in `GHZPrep.lean`). The structure is:
--
--   1. each local gate matrix (2×2 / 4×4) is unitary           [*_unitary]
--   2. `embed_single_qubit`/`embed_two_qubit` preserve unitarity
--        • single-qubit: induction on the qubit position, reusing the
--          head/shift Kronecker decompositions + "kron of unitaries is
--          unitary" + "reindex preserves unitary".
--        • two-qubit: algebraic — `(embed U)ᴴ = embed Uᴴ`
--          (`embed_two_qubit_conjTranspose`) together with the existing
--          same-pair multiplicativity (`embed_two_qubit_mul_same`) and unit
--          law (`embed_two_qubit_one`).  This works at ARBITRARY positions,
--          where no Kronecker factorization exists.
--   3. `denote_gate g` is unitary by case analysis on `g`        [denote_gate_unitary]
--   4. `denote c` is unitary because `unitary _` is a submonoid [denote_unitary]
--
-- Everything is phrased with mathlib's `unitary` submonoid, so `one_mem` /
-- `mul_mem` discharge the empty-circuit and composition steps for free.
-- ===========================================================================

open scoped ComplexConjugate

/-- A matrix is unitary as soon as one side of the inverse relation holds and
    the other is its conjugate transpose: package both into membership of the
    `unitary` submonoid (`star = ᴴ` for complex matrices). -/
theorem mem_unitary_of_conjTranspose {N : ℕ} (M : Matrix (Fin N) (Fin N) ℂ)
    (h1 : Mᴴ * M = 1) (h2 : M * Mᴴ = 1) :
    M ∈ unitary (Matrix (Fin N) (Fin N) ℂ) :=
  Unitary.mem_iff.mpr
    ⟨by rwa [Matrix.star_eq_conjTranspose], by rwa [Matrix.star_eq_conjTranspose]⟩

/-- **Kronecker product of unitaries is unitary.** `(A ⊗ B)ᴴ (A ⊗ B) =
    (Aᴴ A) ⊗ (Bᴴ B) = 1 ⊗ 1 = 1`. -/
theorem unitary_kron {ι κ : Type*} [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ]
    (A : Matrix ι ι ℂ) (B : Matrix κ κ ℂ)
    (hA : A ∈ unitary (Matrix ι ι ℂ)) (hB : B ∈ unitary (Matrix κ κ ℂ)) :
    Matrix.kronecker A B ∈ unitary (Matrix (ι × κ) (ι × κ) ℂ) := by
  obtain ⟨hA1, hA2⟩ := Unitary.mem_iff.mp hA
  obtain ⟨hB1, hB2⟩ := Unitary.mem_iff.mp hB
  rw [Matrix.star_eq_conjTranspose] at hA1 hA2 hB1 hB2
  refine Unitary.mem_iff.mpr ⟨?_, ?_⟩
  -- `⊗ₖ` is notation for `kroneckerMap (·*·)`; restate `Matrix.kronecker` in that
  -- form (defeq) so the kronecker rewrite lemmas fire.
  · show (Matrix.kroneckerMap (· * ·) A B)ᴴ * Matrix.kroneckerMap (· * ·) A B = 1
    rw [Matrix.conjTranspose_kronecker, ← Matrix.mul_kronecker_mul, hA1, hB1,
        Matrix.one_kronecker_one]
  · show Matrix.kroneckerMap (· * ·) A B * (Matrix.kroneckerMap (· * ·) A B)ᴴ = 1
    rw [Matrix.conjTranspose_kronecker, ← Matrix.mul_kronecker_mul, hA2, hB2,
        Matrix.one_kronecker_one]

/-- **`reindex e e` preserves unitarity.** `reindex` by the SAME equiv on rows
    and columns is conjugation by a permutation; it commutes with `ᴴ` and with
    matrix multiplication (`submatrix_mul_equiv`), so it carries `Mᴴ M = 1` to
    `(reindex M)ᴴ (reindex M) = reindex (Mᴴ M) = reindex 1 = 1`. -/
theorem unitary_reindex {ι l : Type*} [Fintype ι] [DecidableEq ι] [Fintype l] [DecidableEq l]
    (e : ι ≃ l) (M : Matrix ι ι ℂ) (hM : M ∈ unitary (Matrix ι ι ℂ)) :
    Matrix.reindex e e M ∈ unitary (Matrix l l ℂ) := by
  obtain ⟨h1, h2⟩ := Unitary.mem_iff.mp hM
  rw [Matrix.star_eq_conjTranspose] at h1 h2
  refine Unitary.mem_iff.mpr ⟨?_, ?_⟩
  · rw [Matrix.star_eq_conjTranspose, Matrix.conjTranspose_reindex,
        Matrix.reindex_apply, Matrix.reindex_apply, Matrix.submatrix_mul_equiv, h1,
        Matrix.submatrix_one_equiv]
  · rw [Matrix.star_eq_conjTranspose, Matrix.conjTranspose_reindex,
        Matrix.reindex_apply, Matrix.reindex_apply, Matrix.submatrix_mul_equiv, h2,
        Matrix.submatrix_one_equiv]

-- --------------------------------------------------------------------------
-- Local gate matrices are unitary.
-- --------------------------------------------------------------------------

/-- `exp w` with `w` purely imaginary (`w̄ = -w`) is a unit: `exp(w)ᴴ exp(w) =
    exp(-w) exp(w) = exp(0) = 1` (and symmetrically). -/
private lemma star_exp_purelyImag {w : ℂ} (hw : star w = -w) :
    star (Complex.exp w) * Complex.exp w = 1 ∧ Complex.exp w * star (Complex.exp w) = 1 := by
  have hstar : star (Complex.exp w) = Complex.exp (-w) := by
    rw [← starRingEnd_apply, ← Complex.exp_conj, starRingEnd_apply, hw]
  refine ⟨?_, ?_⟩
  · rw [hstar, ← Complex.exp_add, show -w + w = 0 by ring, Complex.exp_zero]
  · rw [hstar, ← Complex.exp_add, show w + -w = 0 by ring, Complex.exp_zero]

theorem H_unitary : Gates.H ∈ unitary (Matrix (Fin 2) (Fin 2) ℂ) := by
  have hHc : (Gates.H)ᴴ = Gates.H := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [Gates.H, Matrix.conjTranspose_apply, star_div₀, star_one, ← starRingEnd_apply,
            Complex.conj_ofReal]
  exact mem_unitary_of_conjTranspose _ (by rw [hHc]; exact Gates.H_self_inverse)
    (by rw [hHc]; exact Gates.H_self_inverse)

theorem X_unitary : Gates.X ∈ unitary (Matrix (Fin 2) (Fin 2) ℂ) := by
  apply mem_unitary_of_conjTranspose <;>
    · ext i j; fin_cases i <;> fin_cases j <;>
        simp [Gates.X, Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_two,
              Matrix.one_apply]

theorem Y_unitary : Gates.Y ∈ unitary (Matrix (Fin 2) (Fin 2) ℂ) := by
  apply mem_unitary_of_conjTranspose <;>
    · ext i j; fin_cases i <;> fin_cases j <;>
        simp [Gates.Y, Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_two,
              Matrix.one_apply, Complex.conj_I] <;> ring

theorem Z_unitary : Gates.Z ∈ unitary (Matrix (Fin 2) (Fin 2) ℂ) := by
  apply mem_unitary_of_conjTranspose <;>
    · ext i j; fin_cases i <;> fin_cases j <;>
        simp [Gates.Z, Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_two,
              Matrix.one_apply]

theorem S_unitary : Gates.S ∈ unitary (Matrix (Fin 2) (Fin 2) ℂ) := by
  apply mem_unitary_of_conjTranspose <;>
    · ext i j; fin_cases i <;> fin_cases j <;>
        simp [Gates.S, Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_two,
              Matrix.one_apply, Complex.conj_I] <;> ring

/-- A diagonal matrix whose entries have unit modulus (`d̄ k · d k = 1`) is
    unitary: `(diagonal d)ᴴ (diagonal d) = diagonal (d̄ · d) = diagonal 1 = 1`.
    The uniform tool for the (diagonal) phase gates `T`, `RZ`, `CZ`, `CP`, which
    avoids any per-entry `exp`/`star` simp normalization friction. -/
theorem diagonal_mem_unitary {N : ℕ} (d : Fin N → ℂ) (h : ∀ k, star (d k) * d k = 1) :
    Matrix.diagonal d ∈ unitary (Matrix (Fin N) (Fin N) ℂ) := by
  apply mem_unitary_of_conjTranspose
  · rw [Matrix.diagonal_conjTranspose, Matrix.diagonal_mul_diagonal]
    have hfun : (fun i => (star d) i * d i) = fun _ : Fin N => (1 : ℂ) := by
      funext k; simpa using h k
    rw [hfun, Matrix.diagonal_one]
  · rw [Matrix.diagonal_conjTranspose, Matrix.diagonal_mul_diagonal]
    have hfun : (fun i => d i * (star d) i) = fun _ : Fin N => (1 : ℂ) := by
      funext k; rw [Pi.star_apply, mul_comm]; exact h k
    rw [hfun, Matrix.diagonal_one]

theorem T_unitary : Gates.T ∈ unitary (Matrix (Fin 2) (Fin 2) ℂ) := by
  have hw : star (Complex.I * Complex.ofReal (Real.pi / 4))
      = -(Complex.I * Complex.ofReal (Real.pi / 4)) := by
    rw [star_mul', ← starRingEnd_apply, ← starRingEnd_apply, Complex.conj_I, Complex.conj_ofReal]
    ring
  rw [show Gates.T
        = Matrix.diagonal ![1, Complex.exp (Complex.I * Complex.ofReal (Real.pi / 4))] from by
      ext i j; fin_cases i <;> fin_cases j <;>
        simp [Gates.T, Matrix.diagonal_apply]]
  refine diagonal_mem_unitary _ (fun k => ?_)
  fin_cases k <;> first | exact (star_exp_purelyImag hw).1 | simp

theorem RZ_unitary (θ : ℝ) : Gates.RZ θ ∈ unitary (Matrix (Fin 2) (Fin 2) ℂ) := by
  have hwp : star (Complex.I * Complex.ofReal (θ / 2)) = -(Complex.I * Complex.ofReal (θ / 2)) := by
    rw [star_mul', ← starRingEnd_apply, ← starRingEnd_apply, Complex.conj_I, Complex.conj_ofReal]
    ring
  have hwn : star (-Complex.I * Complex.ofReal (θ / 2)) = -(-Complex.I * Complex.ofReal (θ / 2)) := by
    rw [star_mul', ← starRingEnd_apply, ← starRingEnd_apply, map_neg, Complex.conj_I,
        Complex.conj_ofReal]
    ring
  rw [show Gates.RZ θ
        = Matrix.diagonal ![Complex.exp (-Complex.I * Complex.ofReal (θ / 2)),
                            Complex.exp (Complex.I * Complex.ofReal (θ / 2))] from by
      ext i j; fin_cases i <;> fin_cases j <;>
        simp [Gates.RZ, Matrix.diagonal_apply]]
  refine diagonal_mem_unitary _ (fun k => ?_)
  fin_cases k <;>
    first
      | exact (star_exp_purelyImag hwn).1
      | exact (star_exp_purelyImag hwp).1
      | simp

theorem RY_unitary (θ : ℝ) : Gates.RY θ ∈ unitary (Matrix (Fin 2) (Fin 2) ℂ) := by
  have hcs : (↑(Real.cos (θ * (1 / 2))) : ℂ) ^ 2 + ↑(Real.sin (θ * (1 / 2))) ^ 2 = 1 := by
    rw [← Complex.ofReal_pow, ← Complex.ofReal_pow, ← Complex.ofReal_add]
    norm_cast
    rw [show θ * (1 / 2) = θ / 2 by ring]
    exact Real.cos_sq_add_sin_sq (θ / 2)
  apply mem_unitary_of_conjTranspose <;>
    · ext i j
      fin_cases i <;> fin_cases j <;>
        simp [Gates.RY, Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_two,
          Complex.conj_ofReal, -Complex.ofReal_cos, -Complex.ofReal_sin] <;>
        (try ring) <;>
        linear_combination hcs

theorem CNOT_unitary : Gates.CNOT ∈ unitary (Matrix (Fin 4) (Fin 4) ℂ) := by
  apply mem_unitary_of_conjTranspose <;>
    · ext i j; fin_cases i <;> fin_cases j <;>
        simp [Gates.CNOT, Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_four,
              Matrix.one_apply]

theorem CZ4_unitary : CZ4 ∈ unitary (Matrix (Fin 4) (Fin 4) ℂ) := by
  apply mem_unitary_of_conjTranspose <;>
    · ext i j; fin_cases i <;> fin_cases j <;>
        simp [CZ4, Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_four]

theorem SWAP4_unitary : SWAP4 ∈ unitary (Matrix (Fin 4) (Fin 4) ℂ) := by
  apply mem_unitary_of_conjTranspose <;>
    · ext i j; fin_cases i <;> fin_cases j <;>
        simp [SWAP4, Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_four]

theorem CPhase4_unitary (θ : ℝ) : CPhase4 θ ∈ unitary (Matrix (Fin 4) (Fin 4) ℂ) := by
  have hw : star ((θ : ℂ) * Complex.I) = -((θ : ℂ) * Complex.I) := by
    rw [star_mul', ← starRingEnd_apply, ← starRingEnd_apply, Complex.conj_ofReal, Complex.conj_I]
    ring
  rw [CPhase4_eq_diagonal]
  refine diagonal_mem_unitary _ (fun k => ?_)
  fin_cases k <;> first | exact (star_exp_purelyImag hw).1 | simp

-- --------------------------------------------------------------------------
-- Embeddings preserve unitarity.
-- --------------------------------------------------------------------------

/-- **`embed_single_qubit` preserves unitarity** (any qubit position).  Proof by
    induction on the position via the head/shift Kronecker factorizations: at
    qubit `0` the embed is `U ⊗ I` (`embed_single_qubit_head`); at qubit `q+1`
    it is `I ⊗ (embed at q)` (`embed_single_qubit_shift`).  Both factors are
    unitary (`U` by hypothesis, the identity trivially, the inner embed by the
    induction hypothesis), so the Kronecker product is (`unitary_kron`), and
    `reindex` keeps it unitary (`unitary_reindex`). -/
theorem embed_single_qubit_unitary (U : Matrix (Fin 2) (Fin 2) ℂ)
    (hU : U ∈ unitary (Matrix (Fin 2) (Fin 2) ℂ)) :
    ∀ (n : ℕ) (q : Fin n),
      embed_single_qubit n q U ∈ unitary (Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ) := by
  intro n
  induction n with
  | zero => intro q; exact q.elim0
  | succ n ih =>
      refine Fin.cases ?_ (fun i => ?_)
      · show embed_single_qubit (n + 1) (⟨0, by omega⟩ : Fin (n + 1)) U ∈ _
        rw [embed_single_qubit_head]
        exact unitary_reindex _ _ (unitary_kron _ _ hU (one_mem _))
      · show embed_single_qubit (n + 1) (⟨i.val + 1, by omega⟩ : Fin (n + 1)) U ∈ _
        rw [embed_single_qubit_shift]
        exact unitary_reindex _ _ (unitary_kron _ _ (one_mem _) (ih i))

/-- **Conjugate transpose commutes with `embed_two_qubit`:**
    `(embed_two_qubit n q0 q1 U)ᴴ = embed_two_qubit n q0 q1 Uᴴ`.  Entrywise: the
    `(i,j)` entry of the LHS is `conj` of the `(j,i)` entry of the embed of `U`,
    whose "rest" guard is symmetric and whose `U`-index pair is transposed —
    exactly the embed of `Uᴴ`. -/
theorem embed_two_qubit_conjTranspose (n : ℕ) (q0 q1 : Fin n) (U : Matrix (Fin 4) (Fin 4) ℂ) :
    (embed_two_qubit n q0 q1 U)ᴴ = embed_two_qubit n q0 q1 Uᴴ := by
  ext i j
  rw [Matrix.conjTranspose_apply, embed_two_qubit_apply, embed_two_qubit_apply]
  by_cases h :
      i.val - (i.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val)
            - (i.val / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)
       = j.val - (j.val / 2 ^ (n - 1 - q0.val)) % 2 * 2 ^ (n - 1 - q0.val)
               - (j.val / 2 ^ (n - 1 - q1.val)) % 2 * 2 ^ (n - 1 - q1.val)
  · rw [if_pos h.symm, if_pos h, Matrix.conjTranspose_apply]
  · rw [if_neg (fun hh => h hh.symm), if_neg h, star_zero]

/-- **`embed_two_qubit` preserves unitarity** (arbitrary distinct positions).
    No Kronecker factorization exists in general, so this is algebraic:
    `(embed U)ᴴ (embed U) = (embed Uᴴ)(embed U) = embed (Uᴴ U) = embed 1 = 1`,
    using `embed_two_qubit_conjTranspose`, the same-pair multiplicativity
    `embed_two_qubit_mul_same`, and the unit law `embed_two_qubit_one`. -/
theorem embed_two_qubit_unitary {n : ℕ} (q0 q1 : Fin n) (hq : q0 ≠ q1)
    (U : Matrix (Fin 4) (Fin 4) ℂ) (hU : U ∈ unitary (Matrix (Fin 4) (Fin 4) ℂ)) :
    embed_two_qubit n q0 q1 U ∈ unitary (Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ) := by
  obtain ⟨h1, h2⟩ := Unitary.mem_iff.mp hU
  rw [Matrix.star_eq_conjTranspose] at h1 h2
  apply mem_unitary_of_conjTranspose
  · rw [embed_two_qubit_conjTranspose, embed_two_qubit_mul_same n q0 q1 hq, h1,
        embed_two_qubit_one n q0 q1 hq]
  · rw [embed_two_qubit_conjTranspose, embed_two_qubit_mul_same n q0 q1 hq, h2,
        embed_two_qubit_one n q0 q1 hq]

-- --------------------------------------------------------------------------
-- Gates and circuits are unitary.
-- --------------------------------------------------------------------------

/-- **Every gate denotes to a unitary.** Case analysis on the gate: single-qubit
    gates go through `embed_single_qubit_unitary` with the corresponding local
    `*_unitary`; two-qubit gates through `embed_two_qubit_unitary` (the `q0 ≠ q1`
    obligation is carried in the gate constructor). -/
theorem denote_gate_unitary {n : ℕ} (g : GateApp n) :
    denote_gate g ∈ unitary (Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ) := by
  cases g with
  | h q => exact embed_single_qubit_unitary _ H_unitary n q
  | x q => exact embed_single_qubit_unitary _ X_unitary n q
  | y q => exact embed_single_qubit_unitary _ Y_unitary n q
  | z q => exact embed_single_qubit_unitary _ Z_unitary n q
  | s q => exact embed_single_qubit_unitary _ S_unitary n q
  | t q => exact embed_single_qubit_unitary _ T_unitary n q
  | rz q θ => exact embed_single_qubit_unitary _ (RZ_unitary θ) n q
  | ry q θ => exact embed_single_qubit_unitary _ (RY_unitary θ) n q
  | cx c t hct => exact embed_two_qubit_unitary c t hct _ CNOT_unitary
  | cz q0 q1 hq => exact embed_two_qubit_unitary q0 q1 hq _ CZ4_unitary
  | cp c t θ hct => exact embed_two_qubit_unitary c t hct _ (CPhase4_unitary θ)
  | swap q0 q1 hq => exact embed_two_qubit_unitary q0 q1 hq _ SWAP4_unitary
  | ccz q0 q1 q2 h01 h02 h12 =>
      show embed_three_qubit n q0 q1 q2 CCZ8 ∈ _
      rw [CCZ8, embed_three_qubit_diagonal n q0 q1 q2 h01 h02 h12]
      exact diagonal_mem_unitary _ (fun _ => CCZ8_diag_unit _)

/-- **Every circuit denotes to a unitary.**  `denote` is a left-folded product
    of gate unitaries; `unitary _` is a submonoid, so `one_mem` covers the empty
    circuit and `mul_mem` covers the cons step (`denote_cons`).  This is the
    compositional replacement for per-circuit explicit-matrix unitarity. -/
theorem denote_unitary {n : ℕ} (c : Circuit n) :
    denote c ∈ unitary (Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ) := by
  induction c with
  | nil => rw [empty_circuit_identity]; exact one_mem _
  | cons g rest ih => rw [denote_cons]; exact mul_mem ih (denote_gate_unitary g)

/-- Spec-extraction-facing corollary: `denote c` satisfies the `is_unitary`
    predicate (`(denote c)ᴴ · denote c = 1`) every `@assert unitary` obligation
    discharges.  This closes the `@assert unitary` for *any* circuit — including
    the GHZ circuit, whose explicit-matrix unitarity proof did not scale (see the
    former deferral in `GHZPrep.lean`). -/
theorem denote_is_unitary {n : ℕ} (c : Circuit n) : is_unitary (denote c) := by
  have h := (Unitary.mem_iff.mp (denote_unitary c)).1
  rwa [Matrix.star_eq_conjTranspose] at h

end QuantumProofs.CircuitSemantics
