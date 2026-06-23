/-
  Gate-model GHZ state preparation (LEAN_EXPORT Track B — second recognized
  circuit after Bell).

  Adds the gate-model state-prep theorem the `--gate-model` exporter needs for a
  3-qubit GHZ circuit (`H@0; CX 0→1; CX 1→2`): applied to |000⟩ it yields the GHZ
  state (|000⟩ + |111⟩)/√2. Same technique as `BellPrep.bell_state_prep_correct`,
  one qubit larger (8×8, `Fin.sum_univ_eight`). Basis convention: qubit 0 is the
  high bit (so |000⟩ = e₀, |111⟩ = e₇), per `CircuitSemantics.embed_*`.
-/

import QuantumProofs.CircuitSemantics
import QuantumProofs.Gates

namespace QuantumProofs.GHZPrep

open QuantumProofs.CircuitSemantics QuantumProofs.Gates Complex Matrix

/-- Gate-model GHZ circuit: `H@0`, then `CX(0→1)`, then `CX(1→2)`. -/
def ghz_circuit : Circuit 3 :=
  [.h ⟨0, by omega⟩,
   .cx ⟨0, by omega⟩ ⟨1, by omega⟩ (by decide),
   .cx ⟨1, by omega⟩ ⟨2, by omega⟩ (by decide)]

/-- The GHZ-state column `(|000⟩ + |111⟩)/√2`: amplitude `1/√2` at indices 0, 7. -/
noncomputable def ghzAmp (i : Fin 8) : ℂ :=
  if i = 0 ∨ i = 7 then (1 : ℂ) / Complex.ofReal (Real.sqrt 2) else 0

/-- `denote ghz_circuit = CX(1,2)-embed · CX(0,1)-embed · H(0)-embed` (the last
    gate applied is the leftmost factor). -/
theorem denote_ghz :
    denote ghz_circuit
      = denote_gate (GateApp.cx ⟨1, by omega⟩ ⟨2, by omega⟩ (by decide))
        * denote_gate (GateApp.cx ⟨0, by omega⟩ ⟨1, by omega⟩ (by decide))
        * denote_gate (GateApp.h ⟨0, by omega⟩) := by
  simp [ghz_circuit, denote_cons, empty_circuit_identity, Matrix.mul_assoc]

/-- **Gate-model GHZ state preparation.** Column 0 of `denote ghz_circuit` (its
    action on |000⟩) is the GHZ state `(|000⟩ + |111⟩)/√2`. -/
theorem ghz_state_prep_correct :
    ∀ i : Fin 8, denote ghz_circuit i 0 = ghzAmp i := by
  intro i
  rw [denote_ghz]
  fin_cases i <;>
    simp [denote_gate, embed_single_qubit, embed_two_qubit, local_unitary_1q,
          Gates.H, Gates.CNOT, ghzAmp, Matrix.mul_apply, Fin.sum_univ_eight]

/-- **Gate-model GHZ unitarity.** `denote ghz_circuit` is unitary.

    Formerly DEFERRED: the `BellPrep.bell_unitary` technique (explicit unscaled
    matrix `M` with `Mᴴ·M = 2·I`) does NOT scale here — the 8×8 triple-gate
    product `CNOT(1,2)·CNOT(0,1)·(H⊗I⊗I)` blows the `whnf` heartbeat limit. The
    fix landed as *compositional* unitarity in `CircuitSemantics`: every gate's
    `denote_gate` is unitary, embeddings preserve unitarity
    (`embed_single_qubit_unitary` via the head/shift Kronecker bridge,
    `embed_two_qubit_unitary` via `conjTranspose` + same-pair multiplicativity),
    and `denote` is a submonoid product (`denote_is_unitary`).  GHZ unitarity is
    now a one-line instance — and the `--gate-model` exporter can close
    `@assert unitary` for *any* recognized circuit. -/
theorem ghz_unitary : is_unitary (denote ghz_circuit) :=
  denote_is_unitary ghz_circuit

end QuantumProofs.GHZPrep
