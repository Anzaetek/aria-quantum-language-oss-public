/-
QFT export bridge (PROOF_EXEC_PLAN item 4, n=2 seed).

The Aria exporter lowers `QFT(n)` to an explicit gate list (see
`examples/aria/qft.aria` and `CircuitBuilder::qft` in
`crates/aria-core/src/ast/builder.rs`):

    for i in 0..n:  H(q[i]); for j in i+1..n: CP(ctrl=q[j], tgt=q[i], π/2^(j-i))
    for i in 0..n/2: SWAP(q[i], q[n-1-i])

That lowered form is **not** definitionally equal to the library
`QFT.qft_circuit n`: the exporter emits `CP(ctrl=j, tgt=i)` (control on the
*higher* wire) whereas `qft_circuit` builds `cp i j` (control on the lower
wire). Because `CPhase4` only phases the |11⟩ component it is symmetric in its
two qubits, so the two circuits denote to the same unitary. This file proves
that bridge for the smallest non-trivial case `n = 2`, closing the
`@prove "qft_equals_dft" equiv { denote(QFT) = DFT_matrix(2^n) }` obligation
against the *exporter-emitted* circuit (not just the library one):

    denote (qft2_lowered) = dft_matrix 2

`denote_gate_cp_symm` is the reusable control/target-swap lemma; the general-n
bridge (induction over the lowered nest) is future work — this is the seed.

Everything is sorry-free; `#print axioms qft2_lowered_correct` shows only the
three standard axioms.
-/
import QuantumProofs.QFT
import QuantumProofs.Adjoint

namespace QuantumProofs.QFTExport

open Matrix CircuitSemantics QuantumProofs.QFT

/-- **Control/target symmetry of the controlled-phase gate at the denotation
    level.** `CPhase4 θ` is diagonal and phases only |11⟩, so swapping the two
    qubits leaves the unitary unchanged. This is the one structural gap between
    the `.aria`-lowered QFT (`CP(ctrl=j, tgt=i)`) and `qft_circuit` (`cp i j`). -/
theorem denote_gate_cp_symm {n : ℕ} (c t : Fin n) (θ : ℝ) (h : c ≠ t) :
    denote_gate (GateApp.cp c t θ h) = denote_gate (GateApp.cp t c θ (Ne.symm h)) := by
  rw [denote_gate_cp_diagonal, denote_gate_cp_diagonal]
  congr 1
  funext i
  -- The diagonal entry is `![1,1,1,e]` indexed by `cb*2+tb` (resp. `tb*2+cb`).
  -- That vector is `e` exactly at index 3 (= both bits set), `1` elsewhere, and
  -- `cb*2+tb = 3 ↔ tb*2+cb = 3`, so the two indexings agree.
  have key : ∀ (k : ℕ) (hk : k < 4),
      (![1, 1, 1, Complex.exp ((θ : ℂ) * Complex.I)] : Fin 4 → ℂ) ⟨k, hk⟩
        = if k = 3 then Complex.exp ((θ : ℂ) * Complex.I) else 1 := by
    intro k hk
    interval_cases k <;> rfl
  have hcb : (i.val / 2 ^ (n - 1 - c.val)) % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have htb : (i.val / 2 ^ (n - 1 - t.val)) % 2 < 2 := Nat.mod_lt _ (by norm_num)
  rw [key _ (by omega), key _ (by omega)]
  split_ifs <;> first | rfl | omega

/-- The exact gate list the Aria exporter lowers `QFT(n=2)` to:
    `H(0); CP(ctrl=1, tgt=0, π/2); H(1); SWAP(0,1)`. Note the controlled phase
    has its **control on the higher wire** (`cp 1 0`), the reverse of
    `qft_circuit 2`'s `cp 0 1`. The angle `π/2^(0+1)` is `π/2^(j-i)` at
    `j=1, i=0`, matching `qftPhaseLayer`'s `π/2^(k+1)` at `k=0`. -/
noncomputable def qft2_lowered : Circuit 2 :=
  [ GateApp.h 0,
    GateApp.cp 1 0 (Real.pi / 2 ^ (0 + 1)) (by decide),
    GateApp.h 1,
    GateApp.swap 0 1 (by decide) ]

/-- **Item-4 seed bridge.** The lowered `QFT(n=2)` circuit emitted by the Aria
    exporter denotes to the exact size-4 DFT matrix. The lowered circuit is not
    rfl-equal to `qft_circuit 2` (CP control/target reversed), so we bridge it
    to the library circuit with `denote_gate_cp_symm` and close with
    `qft_correct`. -/
theorem qft2_lowered_correct :
    denote qft2_lowered = dft_matrix 2 := by
  have hlib : qft_circuit 2
      = [ GateApp.h (0 : Fin 2),
          GateApp.cp (0 : Fin 2) 1 (Real.pi / 2 ^ (0 + 1)) (by decide),
          GateApp.h 1,
          GateApp.swap 0 1 (by decide) ] := by rfl
  have hbridge : denote qft2_lowered = denote (qft_circuit 2) := by
    rw [hlib, qft2_lowered]
    simp only [denote_cons, empty_circuit_identity, one_mul]
    rw [denote_gate_cp_symm (0 : Fin 2) 1 (Real.pi / 2 ^ (0 + 1)) (by decide)]
  rw [hbridge, qft_correct]

/- ========================================================================
   General-n bridge: the exporter-lowered QFT(n) denotes to `dft_matrix n`.

   `CircuitBuilder::qft` (crates/aria-core/src/ast/builder.rs) lowers QFT to
   the flat double loop

       for i in 0..n:  H(q[i]); for j in i+1..n: CP(ctrl=q[j], tgt=q[i], π/2^(j-i))
       for i in 0..n/2: SWAP(q[i], q[n-1-i])

   Mirrored below as `qftLowered`, structurally parallel to the library
   `qft_circuit` but with every controlled phase carrying its control on the
   *higher* wire (`cp (k+1) 0`, reverse of `qftPhaseLayer`'s `cp 0 (k+1)`). The
   bridge proves the two denote to the same unitary gate-by-gate via
   `denote_gate_cp_symm`, lifting through the `embed_subcircuit` recursion, then
   closes with `qft_correct`.
   ======================================================================== -/

/-- denote is a congruence under pointwise-equal `denote_gate`: two circuits
    whose gates agree denotation-by-denotation have equal denotations. -/
theorem denote_eq_of_forall2 {n : ℕ} :
    ∀ {c c' : Circuit n},
      List.Forall₂ (fun g g' => denote_gate g = denote_gate g') c c' →
      denote c = denote c'
  | _, _, .nil => rfl
  | _, _, .cons hg hrest => by
      rw [denote_cons, denote_cons, denote_eq_of_forall2 hrest, hg]

/-- A `Forall₂` relation between two `List.ofFn`s reduces to the pointwise one. -/
theorem forall2_ofFn {α β : Type*} {R : α → β → Prop} :
    ∀ {m : ℕ} (f : Fin m → α) (g : Fin m → β), (∀ i, R (f i) (g i)) →
      List.Forall₂ R (List.ofFn f) (List.ofFn g)
  | 0, _, _, _ => by simp [List.ofFn_zero]
  | k + 1, f, g, h => by
      rw [List.ofFn_succ, List.ofFn_succ]
      exact List.Forall₂.cons (h 0)
        (forall2_ofFn (fun i => f i.succ) (fun i => g i.succ) (fun i => h i.succ))

/-- The lowered controlled-phase layer of `qftLowered (n+1)`: `CP(ctrl=k+1,
    tgt=0, π/2^(k+1))` — the genuine controlled phase the exporter emits, with
    the control on the higher wire (reverse of `qftPhaseLayer`'s `cp 0 (k+1)`). -/
noncomputable def qftLoweredPhaseLayer (n : ℕ) : Circuit (n + 1) :=
  List.ofFn (fun k : Fin n =>
    GateApp.cp ⟨k.val + 1, by have := k.isLt; omega⟩ ⟨0, by omega⟩
      (Real.pi / 2 ^ (k.val + 1)) (Fin.ne_of_val_ne (show k.val + 1 ≠ 0 by omega)))

/-- The lowered swap-free core, structurally identical to `qftCore` but using
    `qftLoweredPhaseLayer` (control on the higher wire). Mirrors the forward
    pass of `CircuitBuilder::qft`. -/
noncomputable def qftLoweredCore : (n : ℕ) → Circuit n
  | 0 => []
  | n + 1 =>
    [GateApp.h ⟨0, by omega⟩] ++
    qftLoweredPhaseLayer n ++
    CircuitSemantics.embed_subcircuit (qftLoweredCore n)

/-- The full lowered `QFT(n)` the Aria exporter emits: the lowered core followed
    by the same bit-reversal layer as `qft_circuit`. -/
noncomputable def qftLowered (n : ℕ) : Circuit n :=
  qftLoweredCore n ++ bitReversal n

/-- Each lowered phase gate has the same denotation as the corresponding library
    gate (control/target swapped) — `denote_gate_cp_symm` applied pointwise. -/
theorem qftLoweredPhaseLayer_denote_eq (n : ℕ) :
    denote (qftLoweredPhaseLayer n) = denote (qftPhaseLayer n) := by
  apply denote_eq_of_forall2
  apply forall2_ofFn
  intro k
  exact denote_gate_cp_symm _ _ _ _

/-- **The core bridge.** The lowered swap-free core denotes to the library core,
    by induction over the `embed_subcircuit` recursion: the head `H(0)` matches,
    the phase layers agree by `qftLoweredPhaseLayer_denote_eq`, and the embedded
    tails agree by the induction hypothesis (lifted through `denote` via
    `denote_embed_subcircuit`). -/
theorem qftLoweredCore_denote_eq (n : ℕ) :
    denote (qftLoweredCore n) = denote (qftCore n) := by
  induction n with
  | zero => rfl
  | succ k ih =>
      have hl : qftLoweredCore (k + 1)
          = [GateApp.h ⟨0, by omega⟩] ++ qftLoweredPhaseLayer k
            ++ CircuitSemantics.embed_subcircuit (qftLoweredCore k) := rfl
      have hr : qftCore (k + 1)
          = [GateApp.h ⟨0, by omega⟩] ++ qftPhaseLayer k
            ++ CircuitSemantics.embed_subcircuit (qftCore k) := rfl
      rw [hl, hr, denote_append, denote_append, denote_append, denote_append,
          qftLoweredPhaseLayer_denote_eq,
          denote_embed_subcircuit (qftLoweredCore k),
          denote_embed_subcircuit (qftCore k), ih]

/-- **Item-4 general-n bridge.** For every `n`, the lowered `QFT(n)` circuit the
    Aria exporter emits denotes to the exact size-`2^n` DFT matrix — discharging
    the `@prove "qft_equals_dft" equiv { denote(QFT) = DFT_matrix(2^n) }`
    obligation against the *exporter-emitted* circuit (not just the library one).
    The lowered circuit is not rfl-equal to `qft_circuit n` (every CP has its
    control/target reversed); the gap is closed gate-by-gate by
    `denote_gate_cp_symm`. -/
theorem qftLowered_correct (n : ℕ) :
    denote (qftLowered n) = dft_matrix n := by
  rw [qftLowered, denote_append, qftLoweredCore_denote_eq,
      ← denote_append, ← qft_circuit, qft_correct]

/-- The n=2 seed is the `n = 2` instance of the general lowered circuit. -/
theorem qft2_lowered_eq_qftLowered : qft2_lowered = qftLowered 2 := rfl

/- ========================================================================
   Level-B: the explicit inverse-QFT GATE LIST.

   The faithful QPE certificate needs the inverse-QFT stage expressed as real
   gates, not as the matrix adjoint `(denote (qftLowered n))ᴴ`. `qftInverseLowered`
   mirrors `CircuitBuilder::inverse_qft` (crates/aria-core/src/ast/builder.rs):

       for i in 0..n/2: SWAP(q[i], q[n-1-i])             -- the bit-reversal layer
       for i in (0..n).rev(): for j in (i+1..n).rev():    -- reverse phase sweep
         CP(ctrl=q[j], tgt=q[i], -π/2^(j-i)); H(q[i])

   i.e. the bit-reversal SWAP layer (forward), followed by the gate-by-gate
   adjoint of the lowered forward core (`daggerCircuit (qftLoweredCore n)` —
   reversed order, each `H` unchanged, each `CP` angle negated). We prove it
   denotes to the exact inverse DFT `(dft_matrix n)ᴴ`, sorry-free.
   ======================================================================== -/

/-- Every gate of the lowered forward core is a lowered-QFT gate (`H`/`CP`), so
    `daggerCircuit`/`denote_daggerCircuit` apply to it. Induction over the
    `embed_subcircuit` recursion (`shift_gate` preserves `QFTGate`). -/
theorem qftLoweredCore_QFTGate :
    ∀ (n : ℕ), ∀ g ∈ qftLoweredCore n, g.QFTGate
  | 0 => by intro g hg; simp [qftLoweredCore] at hg
  | n + 1 => by
      intro g hg
      have hexp : qftLoweredCore (n + 1)
          = [GateApp.h ⟨0, by omega⟩] ++ qftLoweredPhaseLayer n
            ++ CircuitSemantics.embed_subcircuit (qftLoweredCore n) := rfl
      rw [hexp] at hg
      simp only [List.mem_append, List.mem_singleton] at hg
      rcases hg with (h1 | h2) | h3
      · subst h1; exact trivial
      · simp only [qftLoweredPhaseLayer, List.mem_ofFn] at h2
        obtain ⟨k, rfl⟩ := h2; exact trivial
      · simp only [CircuitSemantics.embed_subcircuit, List.mem_map] at h3
        obtain ⟨g', hg', rfl⟩ := h3
        have hg'q : g'.QFTGate := qftLoweredCore_QFTGate n g' hg'
        cases g' with
        | h q => exact trivial
        | swap a b h => exact trivial
        | cp c t θ h => exact trivial
        | _ => exact hg'q.elim

/-- `denote (bitReversal n)` is self-adjoint: it is unitary and involutive
    (`bitReversal_involutive`), and a unitary involution equals its own adjoint
    (`Uᴴ = Uᴴ·U·U = U`). -/
theorem denote_bitReversal_conjTranspose (n : ℕ) :
    (denote (bitReversal n))ᴴ = denote (bitReversal n) := by
  have hu : (denote (bitReversal n))ᴴ * denote (bitReversal n) = 1 :=
    denote_is_unitary (bitReversal n)
  have hinv := bitReversal_involutive n
  calc (denote (bitReversal n))ᴴ
      = (denote (bitReversal n))ᴴ * (denote (bitReversal n) * denote (bitReversal n)) := by
        rw [hinv, mul_one]
    _ = ((denote (bitReversal n))ᴴ * denote (bitReversal n)) * denote (bitReversal n) := by
        rw [mul_assoc]
    _ = denote (bitReversal n) := by rw [hu, one_mul]

/-- The explicit inverse-QFT gate list emitted by `CircuitBuilder::inverse_qft`:
    the bit-reversal SWAP layer, then the daggered lowered core (reverse sweep of
    `H` and negated `CP`s). -/
noncomputable def qftInverseLowered (n : ℕ) : Circuit n :=
  bitReversal n ++ daggerCircuit (qftLoweredCore n)

/-- **Level-B bridge.** The explicit inverse-QFT gate list denotes to the exact
    inverse DFT matrix `(dft_matrix n)ᴴ` — no opaque matrix adjoint, real gates.
    `denote (qftInverseLowered n) = (denote (qftLowered n))ᴴ = (dft_matrix n)ᴴ`,
    via `denote_daggerCircuit` on the core and the self-adjoint bit-reversal. -/
theorem qftInverseLowered_correct (n : ℕ) :
    denote (qftInverseLowered n) = (dft_matrix n)ᴴ := by
  have hcore : ∀ g ∈ qftLoweredCore n, g.QFTGate := qftLoweredCore_QFTGate n
  rw [qftInverseLowered, denote_append, denote_daggerCircuit (qftLoweredCore n) hcore,
      ← qftLowered_correct, qftLowered, denote_append, Matrix.conjTranspose_mul,
      denote_bitReversal_conjTranspose]

end QuantumProofs.QFTExport
