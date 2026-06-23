/-
  `H^⊗n` Hadamard-layer fragment, lifted MBQC-free from `QuantumProofs.MBQC`.

  The OSS distribution carries no MBQC module, but the Grover gate-model proof
  (`QuantumProofs.GroverCircuit`) needs the abstract all-Hadamard operator and
  its column-0 (`|+⟩^n`) evaluation. This module re-homes exactly those
  declarations on top of `CircuitSemantics`, with no MBQC dependency.
-/
import QuantumProofs.CircuitSemantics
import QuantumProofs.Gates

namespace QuantumProofs.HadamardLayer

open QuantumProofs QuantumProofs.Gates
open QuantumProofs.CircuitSemantics (embed_single_qubit)
open Complex Matrix

/-- `H^⊗n`: a Hadamard on every wire, as the product of the single-qubit
    embeds (they commute — distinct targets — so order is immaterial). This is
    the `|+⟩^n` preparation operator (`H^⊗n · |0⟩^n = |+⟩^n`). -/
noncomputable def hadamardLayer (n : ℕ) : Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ :=
  (List.finRange n).foldl (fun acc q => embed_single_qubit n q Gates.H * acc) 1

/-- The empty Hadamard layer (0 qubits) is the trivial operator. -/
@[simp] theorem hadamardLayer_zero : hadamardLayer 0 = 1 := rfl

/-- `hadamardLayer 1 = H` on the one wire (the `|+⟩`-prep). -/
theorem hadamardLayer_one : hadamardLayer 1 = embed_single_qubit 1 0 Gates.H := by
  unfold hadamardLayer
  simp [List.finRange, Matrix.mul_one]

/-- `hadamardLayer n = denote` of the all-Hadamard circuit `[H@0,…,H@(n-1)]`.
    Same left-folded product of single-qubit `H` embeds. Lets the graph-state
    amplitude lemmas reuse the `CircuitSemantics` Kronecker machinery. -/
theorem hadamardLayer_eq_denote (n : ℕ) :
    hadamardLayer n
      = CircuitSemantics.denote (List.ofFn (fun q : Fin n => CircuitSemantics.GateApp.h q)) := by
  unfold hadamardLayer CircuitSemantics.denote
  rw [List.ofFn_eq_map, List.foldl_map]
  rfl

/-- **`H^⊗(n+1) = H ⊗ H^⊗n`** through the `tailEquiv` reindex: the head wire
    gets `H`, the tail is `hadamardLayer n`. -/
theorem hadamardLayer_succ (n : ℕ) :
    hadamardLayer (n+1)
      = Matrix.reindex (CircuitSemantics.tailEquiv n) (CircuitSemantics.tailEquiv n)
          (Matrix.kronecker Gates.H (hadamardLayer n)) := by
  rw [hadamardLayer_eq_denote (n+1), hadamardLayer_eq_denote n]
  have hlist : (List.ofFn (fun q : Fin (n+1) => CircuitSemantics.GateApp.h q))
      = CircuitSemantics.GateApp.h ⟨0, by omega⟩
        :: CircuitSemantics.embed_subcircuit
            (List.ofFn (fun q : Fin n => CircuitSemantics.GateApp.h q)) := by
    rw [List.ofFn_succ, CircuitSemantics.embed_subcircuit, List.map_ofFn]
    rfl
  rw [hlist, CircuitSemantics.denote_cons, CircuitSemantics.denote_embed_subcircuit]
  show Matrix.reindex (CircuitSemantics.tailEquiv n) (CircuitSemantics.tailEquiv n)
        (Matrix.kronecker 1 _)
      * CircuitSemantics.embed_single_qubit (n+1) ⟨0, by omega⟩ Gates.H = _
  rw [CircuitSemantics.embed_single_qubit_head, Matrix.reindex_apply, Matrix.reindex_apply,
      Matrix.reindex_apply, Matrix.submatrix_mul_equiv]
  congr 1
  simp only [Matrix.kronecker]
  rw [← Matrix.mul_kronecker_mul, Matrix.one_mul, Matrix.mul_one]

/-- The Hadamard column-`0` entries are all `1/√2` (any row, column `0`). -/
theorem H_col0 (a b : Fin 2) (hb : b.val = 0) :
    Gates.H a b = 1 / Complex.ofReal (Real.sqrt 2) := by
  have : b = 0 := Fin.ext hb
  subst this
  fin_cases a <;> simp [Gates.H]

/-- **`H^⊗n |0⟩ = |+⟩^n` has uniform amplitude `(1/√2)^n`** — column 0 of
    `hadamardLayer n` is the constant `(1/√2)^n`. Induction via `hadamardLayer_succ`
    + the Kronecker entry extractor. -/
theorem hadamardLayer_col0 (n : ℕ) (i : Fin (2^n)) :
    hadamardLayer n i 0 = (1 / Complex.ofReal (Real.sqrt 2)) ^ n := by
  induction n with
  | zero => fin_cases i; simp [hadamardLayer]
  | succ k ih =>
      rw [hadamardLayer_succ, CircuitSemantics.reindex_tailEquiv_kron_apply,
          show (1 / Complex.ofReal (Real.sqrt 2)) ^ (k+1)
            = (1 / Complex.ofReal (Real.sqrt 2)) * (1 / Complex.ofReal (Real.sqrt 2)) ^ k
          from pow_succ' _ _]
      congr 1
      · exact H_col0 _ _ (by simp)
      · convert ih ⟨i.val % 2 ^ k, Nat.mod_lt _ (Nat.pos_of_ne_zero (by positivity))⟩ using 2

end QuantumProofs.HadamardLayer
