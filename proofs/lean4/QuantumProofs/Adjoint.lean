/-
Circuit adjoints (LEAN_EXPORT Track B — QPE Level-B foundation).

Reusable infrastructure for "the adjoint of a circuit is its reverse with every
gate daggered". The key export is

    denote_daggerCircuit : denote (daggerCircuit c) = (denote c)ᴴ

for circuits built from the lowered-QFT gate set (Hadamard, SWAP, controlled
phase). It backs the explicit inverse-QFT *gate list* `qftInverseLowered` in
`QFTExport.lean` — letting the faithful QPE certificate express the inverse-QFT
stage as real gates (`denote (qftInverseLowered n) = (dft_matrix n)ᴴ`) rather
than an opaque matrix adjoint.

Scope: `daggerGate`/`denote_daggerGate` are proved correct on the three gates a
lowered QFT uses (`GateApp.QFTGate`). Extending the predicate to the Paulis,
CX/CZ, and RZ/RY (all genuinely (self-)adjoint within the gate set) is purely
mechanical and lands when Grover needs it.

Everything is sorry-free; `#print axioms denote_daggerCircuit` shows only the
three standard axioms.
-/
import QuantumProofs.CircuitSemantics

namespace QuantumProofs.CircuitSemantics

open Matrix

-- --------------------------------------------------------------------------
-- Conjugate transpose commutes with single-qubit embedding (the missing
-- sibling of `embed_two_qubit_conjTranspose`).
-- --------------------------------------------------------------------------

/-- Entrywise form of `embed_single_qubit`, named for rewriting. -/
theorem embed_single_qubit_apply (n : ℕ) (q : Fin n)
    (U : Matrix (Fin 2) (Fin 2) ℂ) (i j : Fin (2 ^ n)) :
    embed_single_qubit n q U i j =
      if i.val - (i.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
         = j.val - (j.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
      then U ⟨(i.val / 2 ^ (n - 1 - q.val)) % 2, by omega⟩
             ⟨(j.val / 2 ^ (n - 1 - q.val)) % 2, by omega⟩
      else 0 := rfl

/-- **Conjugate transpose commutes with `embed_single_qubit`:**
    `(embed_single_qubit n q U)ᴴ = embed_single_qubit n q Uᴴ`. The `(i,j)` entry
    of the LHS is `conj` of the `(j,i)` entry of the embed of `U`, whose "rest"
    guard is symmetric and whose `U`-index pair is transposed — exactly the
    embed of `Uᴴ`. (Mirror of `embed_two_qubit_conjTranspose`.) -/
theorem embed_single_qubit_conjTranspose (n : ℕ) (q : Fin n)
    (U : Matrix (Fin 2) (Fin 2) ℂ) :
    (embed_single_qubit n q U)ᴴ = embed_single_qubit n q Uᴴ := by
  ext i j
  rw [Matrix.conjTranspose_apply, embed_single_qubit_apply, embed_single_qubit_apply]
  by_cases h :
      i.val - (i.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
       = j.val - (j.val / 2 ^ (n - 1 - q.val)) % 2 * 2 ^ (n - 1 - q.val)
  · rw [if_pos h.symm, if_pos h, Matrix.conjTranspose_apply]
  · rw [if_neg (fun hh => h hh.symm), if_neg h, star_zero]

-- --------------------------------------------------------------------------
-- Hermitian / adjoint facts for the lowered-QFT gate matrices.
-- --------------------------------------------------------------------------

/-- The Hadamard is Hermitian: `Hᴴ = H`. -/
theorem H_conjTranspose : (Gates.H)ᴴ = Gates.H := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Gates.H, Matrix.conjTranspose_apply, ← starRingEnd_apply, Complex.conj_ofReal]

/-- `SWAP4` is Hermitian (real symmetric permutation). -/
theorem SWAP4_conjTranspose : (SWAP4)ᴴ = SWAP4 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [SWAP4, Matrix.conjTranspose_apply]

/-- `CPhase4 θ` adjoint negates the angle: `(CPhase4 θ)ᴴ = CPhase4 (-θ)`. It is
    diagonal `diag(1,1,1,e^{iθ})`, so the adjoint is `diag(1,1,1,e^{-iθ})`. -/
theorem CPhase4_conjTranspose (θ : ℝ) : (CPhase4 θ)ᴴ = CPhase4 (-θ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [CPhase4, Matrix.conjTranspose_apply, ← starRingEnd_apply, ← Complex.exp_conj,
          map_mul, Complex.conj_ofReal, Complex.conj_I, Complex.ofReal_neg, mul_neg, neg_mul]

-- --------------------------------------------------------------------------
-- Per-gate adjoint, then the reverse-and-dagger circuit adjoint.
-- --------------------------------------------------------------------------

/-- The gates appearing in lowered QFT / inverse-QFT circuits: Hadamard, SWAP,
    and controlled-phase. `daggerGate` is proved correct (`denote_daggerGate`)
    on exactly these; extending to the Paulis, CX/CZ, RZ/RY is mechanical. -/
def GateApp.QFTGate {n : ℕ} : GateApp n → Prop
  | .h _ => True
  | .swap _ _ _ => True
  | .cp _ _ _ _ => True
  | _ => False

/-- The adjoint gate, within the gate set. `cp` negates its angle; the
    self-adjoint `h`/`swap` map to themselves (as does everything else — the
    placeholder is harmless because `denote_daggerGate` is only invoked on
    `QFTGate` gates). -/
def daggerGate {n : ℕ} : GateApp n → GateApp n
  | .cp c t θ h => .cp c t (-θ) h
  | g => g

/-- **Each lowered-QFT gate's dagger denotes to the adjoint of its denotation.**
    `denote_gate (daggerGate g) = (denote_gate g)ᴴ` for `g : QFTGate`, via the
    `conjTranspose` facts for the embeddings and the local matrices. -/
theorem denote_daggerGate {n : ℕ} (g : GateApp n) (hg : g.QFTGate) :
    denote_gate (daggerGate g) = (denote_gate g)ᴴ := by
  cases g with
  | h q =>
      show embed_single_qubit n q Gates.H = (embed_single_qubit n q Gates.H)ᴴ
      rw [embed_single_qubit_conjTranspose, H_conjTranspose]
  | swap a b h =>
      show embed_two_qubit n a b SWAP4 = (embed_two_qubit n a b SWAP4)ᴴ
      rw [embed_two_qubit_conjTranspose, SWAP4_conjTranspose]
  | cp c t θ h =>
      show embed_two_qubit n c t (CPhase4 (-θ)) = (embed_two_qubit n c t (CPhase4 θ))ᴴ
      rw [embed_two_qubit_conjTranspose, CPhase4_conjTranspose]
  | _ => simp only [GateApp.QFTGate] at hg

/-- The adjoint of a circuit: reverse the gate order and dagger each gate. -/
def daggerCircuit {n : ℕ} (c : Circuit n) : Circuit n := (c.map daggerGate).reverse

/-- **The reverse-and-dagger circuit denotes to the adjoint of the circuit's
    denotation:** `denote (daggerCircuit c) = (denote c)ᴴ`, for circuits over the
    lowered-QFT gate set. By induction: the head gate's dagger lands at the tail
    of the reversed list, contributing `(denote_gate g)ᴴ` on the left exactly as
    `conjTranspose_mul` distributes over `denote_cons`. -/
theorem denote_daggerCircuit {n : ℕ} :
    ∀ (c : Circuit n), (∀ g ∈ c, g.QFTGate) → denote (daggerCircuit c) = (denote c)ᴴ
  | [], _ => by
      simp [daggerCircuit, empty_circuit_identity, Matrix.conjTranspose_one]
  | g :: rest, hc => by
      have hg : g.QFTGate := hc g (List.mem_cons.2 (Or.inl rfl))
      have hrest : ∀ x ∈ rest, x.QFTGate := fun x hx => hc x (List.mem_cons.2 (Or.inr hx))
      have ih := denote_daggerCircuit rest hrest
      have hstep : daggerCircuit (g :: rest) = daggerCircuit rest ++ [daggerGate g] := by
        simp [daggerCircuit, List.map_cons, List.reverse_cons]
      rw [hstep, denote_append, single_gate, denote_daggerGate g hg, ih, denote_cons,
          Matrix.conjTranspose_mul]

end QuantumProofs.CircuitSemantics
