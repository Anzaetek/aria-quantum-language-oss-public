/-
  Gate-model Bell state preparation.

  The MBQC tree proves `Bell_compiler_correct` (pattern ≡ circuit). This file
  adds the *gate-model* theorems the LEAN_EXPORT Track-B auto-exporter needs for
  `bell.aria`, which carries `@prove "bell_correct"` and `@assert unitary`:

    * `bell_state_prep_correct` — applying the Bell circuit (`H` on qubit 0, then
      `CX 0→1`) to `|00⟩` yields the Bell state `(|00⟩ + |11⟩)/√2`.
    * `bell_unitary` — the Bell circuit denotes to a unitary matrix.

  `aria export --gate-model` emits a file that closes both obligations by `exact`
  on these (see `Generated/GateModel/Bell_Spec.lean`), so the export is sorry-free.

  ## Conventions and proof strategy

  Basis indexing follows `CircuitSemantics.embed_*`: **qubit 0 is the high bit**,
  so basis index `= 2·b₀ + b₁`. Under that convention the Bell unitary
  `CNOT·(H⊗I)` is the real-orthogonal matrix `(1/√2)·bellM` with
  `bellM = !![1,0,1,0; 0,1,0,1; 0,1,0,-1; 1,0,-1,0]` (`denote_bell_apply`),
  whose column 0 is `(1/√2)(e₀+e₃) = (|00⟩+|11⟩)/√2`.

  Unitarity uses the **same scalar/integer split as `CirculantSolve`**: factor
  `denote bell_circuit = (1/√2) • bellM` (`denote_bell_eq`), reduce to the integer
  identity `bellMᴴ·bellM = 2•I` (`bellM_conjTranspose_mul`, real entries so
  conjugation is trivial and `ring`/`norm_num` close it), then collapse the one
  scalar `(1/√2)·(1/√2)·2 = 1`. This keeps `√2` out of the 16-entry matrix
  computation, where it would otherwise stall `simp`.

  Dependency chain:
  ```
  denote_bell ─► denote_bell_apply ─► denote_bell_eq ──┐
                          │                            ├─► bell_unitary
                          └─► bell_state_prep_correct   bellM_conjTranspose_mul ┘
  ```
-/

import QuantumProofs.CircuitSemantics
import QuantumProofs.Gates

namespace QuantumProofs.BellPrep

open QuantumProofs.CircuitSemantics QuantumProofs.Gates Complex Matrix

/-- Gate-model Bell circuit: `H` on qubit 0, then `CX(0 → 1)`. The two `Fin`
    bounds (`by omega`) and the `ctrl ≠ tgt` proof (`by decide`) are the gate
    well-formedness obligations `CircuitSemantics` requires. -/
def bell_circuit : Circuit 2 :=
  [.h ⟨0, by omega⟩, .cx ⟨0, by omega⟩ ⟨1, by omega⟩ (by decide)]

/-- The target amplitude vector: the Bell-state column `(|00⟩ + |11⟩)/√2`, i.e.
    amplitude `1/√2` at indices `0` (`|00⟩`) and `3` (`|11⟩`), `0` elsewhere. -/
noncomputable def bellAmp (i : Fin 4) : ℂ :=
  if i = 0 ∨ i = 3 then (1 : ℂ) / Complex.ofReal (Real.sqrt 2) else 0

/-- Unfold the circuit's denotation into a product of single-gate embeddings:
    `denote bell_circuit = (CNOT embedded) · (H embedded)`. `denote` folds the
    gate list with `denote_gate gᵢ * acc` from the identity, so the two-element
    list collapses to this product (`denote_cons` + `empty_circuit_identity`).
    Note the order: the *last* gate applied (`CX`) is the *leftmost* factor. -/
theorem denote_bell :
    denote bell_circuit
      = denote_gate (GateApp.cx ⟨0, by omega⟩ ⟨1, by omega⟩ (by decide))
        * denote_gate (GateApp.h ⟨0, by omega⟩) := by
  simp [bell_circuit, denote_cons, empty_circuit_identity]

/-- **Gate-model Bell state preparation** (the `@prove "bell_correct"` target).
    Column 0 of `denote bell_circuit` — its action on the input `|00⟩` (basis
    vector `e₀`) — is the Bell state `(|00⟩ + |11⟩)/√2`. Proof: unfold to the
    gate product, then evaluate column 0 of each of the 4 rows by expanding the
    embeddings and the 2-qubit matrix product (`Fin.sum_univ_four`). -/
theorem bell_state_prep_correct :
    ∀ i : Fin 4, denote bell_circuit i 0 = bellAmp i := by
  intro i
  rw [denote_bell]
  fin_cases i <;>
    simp [denote_gate, embed_single_qubit, embed_two_qubit, local_unitary_1q,
          Gates.H, Gates.CNOT, bellAmp, Matrix.mul_apply, Fin.sum_univ_four]

/-- The **unscaled** real Bell matrix `M` (entries in `{0, ±1}`): the Bell
    unitary is `(1/√2) · M`, i.e. `M = √2 · CNOT·(H⊗I)`. As in `CirculantSolve`,
    pulling the `1/√2` normalization out into a scalar lets the unitarity proof
    work over this integer matrix (`ring`/`norm_num`) instead of fighting `√2`. -/
def bellM : Matrix (Fin 4) (Fin 4) ℂ :=
  !![1, 0, 1, 0; 0, 1, 0, 1; 0, 1, 0, -1; 1, 0, -1, 0]

/-- Every entry: `denote bell_circuit i j = bellM i j / √2`. Establishes that the
    full circuit denotes to the real-orthogonal Bell unitary `(1/√2)·bellM`
    entrywise (the bridge from the abstract `denote` to the concrete `bellM`). -/
theorem denote_bell_apply (i j : Fin 4) :
    denote bell_circuit i j = bellM i j / Complex.ofReal (Real.sqrt 2) := by
  rw [denote_bell]
  fin_cases i <;> fin_cases j <;>
    simp [denote_gate, embed_single_qubit, embed_two_qubit, local_unitary_1q,
          Gates.H, Gates.CNOT, bellM, Matrix.mul_apply, Fin.sum_univ_four] <;>
    ring

/-- Matrix-level restatement of `denote_bell_apply`: `denote bell_circuit =
    (1/√2) • bellM`. The `scalar • integer-matrix` form the unitarity proof
    consumes (`a/√2 = (√2)⁻¹ · a` by `div_eq_mul_inv` + `mul_comm`). -/
theorem denote_bell_eq :
    denote bell_circuit = (Complex.ofReal (Real.sqrt 2))⁻¹ • bellM := by
  ext i j
  rw [Matrix.smul_apply, smul_eq_mul, denote_bell_apply, div_eq_mul_inv, mul_comm]

/-- **The integer core of unitarity:** `Mᴴ · M = 2·I`. The unscaled Bell matrix
    is orthogonal up to the `√2` normalization. Entries are real `{0,±1}` so the
    conjugate-transpose is just the transpose; `Fin.sum_univ_four` expands the
    4×4 products and `norm_num` closes the `1+1 = 2` on the diagonal. -/
theorem bellM_conjTranspose_mul : bellMᴴ * bellM = (2 : ℂ) • (1 : Matrix (Fin 4) (Fin 4) ℂ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [bellM, Matrix.mul_apply, Fin.sum_univ_four, Matrix.conjTranspose_apply,
          Matrix.smul_apply, smul_eq_mul] <;> norm_num

/-- **Gate-model Bell circuit is unitary** (`Uᴴ · U = 1`). Closes the
    `@assert unitary` obligation the spec-extractor emits for `bell.aria`.
    Same scalar/integer collapse as `CirculantSolve.qft_diagonalizes_circ2`:
    `((1/√2)•M)ᴴ · ((1/√2)•M) = ((1/√2)·(1/√2)) • (Mᴴ·M) = (1/2) • (2•I) = I`. -/
theorem bell_unitary : is_unitary (denote bell_circuit) := by
  -- The single scalar fact `√2 · √2 = 2`.
  have hsqrt : Complex.ofReal (Real.sqrt 2) * Complex.ofReal (Real.sqrt 2) = 2 := by
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by norm_num)]
    norm_num
  have h2 : Complex.ofReal (Real.sqrt 2) ≠ 0 := by
    have : Real.sqrt 2 ≠ 0 := by positivity
    exact Complex.ofReal_ne_zero.mpr this
  -- `(1/√2)` is real ⇒ self-conjugate, so `((1/√2)•M)ᴴ`'s scalar stays `1/√2`.
  have hstar : star ((Complex.ofReal (Real.sqrt 2))⁻¹) = (Complex.ofReal (Real.sqrt 2))⁻¹ := by
    simp
  unfold is_unitary
  -- Expand to `(1/√2)•M`, pull both scalars out (`smul_mul`/`mul_smul`/`smul_smul`),
  -- reduce the integer core to `2•I` (`bellM_conjTranspose_mul`), merge scalars.
  rw [denote_bell_eq, Matrix.conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul,
      smul_smul, bellM_conjTranspose_mul, smul_smul, hstar]
  -- The merged scalar is `(1/√2)·(1/√2)·2 = 1`, leaving `1 • I = I`.
  rw [show (Complex.ofReal (Real.sqrt 2))⁻¹ * (Complex.ofReal (Real.sqrt 2))⁻¹ * 2 = 1 by
        field_simp; linear_combination -hsqrt]
  rw [one_smul]

end QuantumProofs.BellPrep
