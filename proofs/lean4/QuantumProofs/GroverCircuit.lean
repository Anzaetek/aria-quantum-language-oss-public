/-
Grover gate-model circuit (n = 3 seed) — bridge to the abstract operator.

The abstract `QuantumProofs.Grover` development proves the Grover *operator*
`grover_operator n w = diffusion_operator n * oracle n w` rotates the uniform
superposition onto the marked item (`grover_iterate`, `grover_probability`,
`grover_optimal_success`) at the `Matrix`/`Fin → ℂ` level. This file starts the
**gate-model bridge**: concrete `GateApp` circuits whose `denote` reproduces the
abstract matrices, mirroring `grover3.aria`.

`grover3.aria` implements the multi-controlled `Z` as `H · CCX · H` on the
target; now that `CircuitSemantics` has a genuine `ccz` gate (denoting to the
diagonal `CCZ8`), the Lean model uses `ccz` directly.

The foundational identity here is the **oracle core**: for the all-ones marked
state `|111⟩`, the bare `ccz` gate denotes exactly to the abstract `oracle`. The
general-`w` oracle wraps `ccz` in `X`-masks, and the diffusion conjugates it by
the Hadamard layer — those bridges, the `k`-fold iteration, and the measurement
certificate are the remaining steps.

Everything here is sorry-free.
-/
import QuantumProofs.CircuitSemantics
import QuantumProofs.Grover
import QuantumProofs.HadamardLayer
import QuantumProofs.Adjoint

namespace QuantumProofs.GroverCircuit

open Matrix CircuitSemantics QuantumProofs.Grover

/-- The 3-qubit Grover **diffusion** gate list (`grover3.aria`):
    `H^⊗3 · X^⊗3 · CCZ · X^⊗3 · H^⊗3`. The two `X^⊗3` layers are written in
    palindromic order (`x0 x1 x2 … x2 x1 x0`) — faithful, since `X^⊗3` is
    order-independent (single-qubit `X`s on distinct wires commute) — so the
    circuit is *literally* `hLayerGates ++ diffusionDiagCoreGates ++ hLayerGates`
    (see `denote_diffusion_split`), exposing the `HL · (I−2|0⟩⟨0|) · HL`
    structure. This `def` only fixes the circuit; the `denote = −diffusion_operator`
    identity is the remaining Hadamard-layer step. The `ccz` replaces the
    `H·CCX·H`-on-the-target the `.aria` uses. -/
noncomputable def diffusionGates : Circuit 3 :=
  [GateApp.h 0, GateApp.h 1, GateApp.h 2,
   GateApp.x 0, GateApp.x 1, GateApp.x 2,
   GateApp.ccz 0 1 2 (by decide) (by decide) (by decide),
   GateApp.x 2, GateApp.x 1, GateApp.x 0,
   GateApp.h 0, GateApp.h 1, GateApp.h 2]

/-- The **oracle** gate list for the canonical all-ones marked state `|111⟩`:
    the bare `ccz` (no `X`-masks, since every marked bit is `1`). -/
noncomputable def oracleGatesTop : Circuit 3 :=
  [GateApp.ccz 0 1 2 (by decide) (by decide) (by decide)]

/-- `denote [ccz 0 1 2]` is the embedded `CCZ8`. -/
theorem denote_oracleGatesTop_eq_embed :
    denote oracleGatesTop = embed_three_qubit 3 0 1 2 CCZ8 := by
  rw [oracleGatesTop, denote_cons, empty_circuit_identity, one_mul]
  rfl

/-- **Oracle core identity (canonical marked state).** The `ccz` gate denotes
    exactly to the abstract Grover `oracle` for the all-ones marked item `|111⟩`
    (index `7`): both are the diagonal matrix with `−1` at index `7` and `1`
    elsewhere. This is the diagonal heart of the oracle-correctness bridge; the
    general-`w` case conjugates this by the `X`-mask permutation. -/
theorem denote_oracleGatesTop :
    denote oracleGatesTop = oracle 3 ⟨7, by norm_num⟩ := by
  rw [denote_oracleGatesTop_eq_embed, CCZ8,
      embed_three_qubit_diagonal 3 0 1 2 (by decide) (by decide) (by decide),
      oracle_diagonal]
  congr 1
  funext i
  fin_cases i <;> rfl

/-- The **oracle** gate list for the marked state `|011⟩` (index `3`): the bare
    `ccz` wrapped in an `X`-mask on qubit `0` (whose `w`-bit is `0`).
    `X₀ · CCZ · X₀`. -/
noncomputable def oracleGates011 : Circuit 3 :=
  [GateApp.x 0,
   GateApp.ccz 0 1 2 (by decide) (by decide) (by decide),
   GateApp.x 0]

/-- **General-`w` oracle bridge (single-mask instance).** With one `X`-mask the
    `ccz` oracle is conjugated onto a *different* marked item: `X₀ · CCZ · X₀`
    denotes exactly to the abstract Grover oracle for `|011⟩` (index `3`). The
    `X`-mask flips qubit `0`, moving the `−1` from index `7` to index `3` via
    `embed_single_qubit_X_conj_diagonal` (the bit-flip permutation conjugation).
    This is the general-`w` step beyond the canonical `denote_oracleGatesTop`;
    the full arbitrary-`w` oracle iterates this over every zero-bit wire. -/
theorem denote_oracleGates011 :
    denote oracleGates011 = oracle 3 ⟨3, by norm_num⟩ := by
  rw [oracleGates011, denote_cons, denote_cons, denote_cons,
      empty_circuit_identity, one_mul]
  show embed_single_qubit 3 0 Gates.X * embed_three_qubit 3 0 1 2 CCZ8
       * embed_single_qubit 3 0 Gates.X = oracle 3 ⟨3, by norm_num⟩
  rw [CCZ8, embed_three_qubit_diagonal 3 0 1 2 (by decide) (by decide) (by decide),
      embed_single_qubit_X_conj_diagonal, oracle_diagonal]
  congr 1
  funext i
  fin_cases i <;> rfl

/-- The **oracle** gate list for the marked state `|001⟩` (index `1`): the bare
    `ccz` wrapped in a two-wire palindrome `X`-mask on qubits `0, 1` (both have
    `w`-bit `0`). `X₀·X₁·CCZ·X₁·X₀`. -/
noncomputable def oracleGates001 : Circuit 3 :=
  [GateApp.x 0, GateApp.x 1,
   GateApp.ccz 0 1 2 (by decide) (by decide) (by decide),
   GateApp.x 1, GateApp.x 0]

/-- **General-`w` oracle bridge (two-mask instance).** `X₀·X₁·CCZ·X₁·X₀` denotes
    exactly to the Grover oracle for `|001⟩` (index `1`): the two `X`-masks flip
    qubits `0, 1`, moving the `−1` from index `7` to index `1` via
    `embed_single_qubit_X_conj2_diagonal`. -/
theorem denote_oracleGates001 :
    denote oracleGates001 = oracle 3 ⟨1, by norm_num⟩ := by
  simp only [oracleGates001, denote_cons, empty_circuit_identity, one_mul]
  show embed_single_qubit 3 0 Gates.X * embed_single_qubit 3 1 Gates.X
       * embed_three_qubit 3 0 1 2 CCZ8
       * embed_single_qubit 3 1 Gates.X * embed_single_qubit 3 0 Gates.X
       = oracle 3 ⟨1, by norm_num⟩
  rw [CCZ8, embed_three_qubit_diagonal 3 0 1 2 (by decide) (by decide) (by decide),
      embed_single_qubit_X_conj2_diagonal, oracle_diagonal]
  congr 1
  funext i
  fin_cases i <;> rfl

/-- The **diagonal core of the diffusion**: `X^⊗3 · CCZ · X^⊗3` (palindrome
    `X`-layers). Conjugating the `CCZ` diagonal by all three bit-flips moves the
    `−1` from `|111⟩` to `|000⟩`, so this equals the abstract `oracle 3 ⟨0⟩`
    `= I − 2|000⟩⟨000|`. The remaining diffusion step conjugates this by the
    Hadamard layer to obtain `I − 2|s⟩⟨s|` (the `−1`-sign Grover diffusion). -/
noncomputable def diffusionDiagCoreGates : Circuit 3 :=
  [GateApp.x 0, GateApp.x 1, GateApp.x 2,
   GateApp.ccz 0 1 2 (by decide) (by decide) (by decide),
   GateApp.x 2, GateApp.x 1, GateApp.x 0]

theorem denote_diffusionDiagCore :
    denote diffusionDiagCoreGates = oracle 3 ⟨0, by norm_num⟩ := by
  simp only [diffusionDiagCoreGates, denote_cons, empty_circuit_identity, one_mul]
  show embed_single_qubit 3 0 Gates.X * embed_single_qubit 3 1 Gates.X
       * embed_single_qubit 3 2 Gates.X * embed_three_qubit 3 0 1 2 CCZ8
       * embed_single_qubit 3 2 Gates.X * embed_single_qubit 3 1 Gates.X
       * embed_single_qubit 3 0 Gates.X
       = oracle 3 ⟨0, by norm_num⟩
  rw [CCZ8, embed_three_qubit_diagonal 3 0 1 2 (by decide) (by decide) (by decide),
      embed_single_qubit_X_conj3_diagonal, oracle_diagonal]
  congr 1
  funext i
  fin_cases i <;> rfl

/-- The 3-qubit **Hadamard layer** `H^⊗3` as a gate list (`[h0, h1, h2]`), the
    diffusion's conjugating layer in `grover3.aria`. -/
noncomputable def hLayerGates : Circuit 3 :=
  [GateApp.h 0, GateApp.h 1, GateApp.h 2]

/-- `denote [h0, h1, h2]` is the left-associated product of the three embedded
    Hadamards (qubit-`2` factor leftmost, per the `denote_cons` order). -/
theorem denote_hLayer :
    denote hLayerGates
      = embed_single_qubit 3 2 Gates.H
        * embed_single_qubit 3 1 Gates.H
        * embed_single_qubit 3 0 Gates.H := by
  simp only [hLayerGates, denote_cons, empty_circuit_identity, one_mul]
  rfl

/-- **The Hadamard layer is an involution:** `H^⊗3 · H^⊗3 = I`. The three
    embedded Hadamards pairwise commute (distinct wires,
    `embed_single_qubit_disjoint_comm`) so the squared product reorders to
    `(e2·e2)(e1·e1)(e0·e0)`, each collapsing via the per-wire involution
    `embed_single_qubit_h_involutive`. The first consumer of the disjoint-wire
    commutation lemma; the `H^⊗3 · H^⊗3 = I` half of the diffusion bridge. -/
theorem denote_hLayer_involutive :
    denote hLayerGates * denote hLayerGates = 1 := by
  rw [denote_hLayer]
  set e0 := embed_single_qubit 3 0 Gates.H with he0
  set e1 := embed_single_qubit 3 1 Gates.H with he1
  set e2 := embed_single_qubit 3 2 Gates.H with he2
  have hinv0 : e0 * e0 = 1 := embed_single_qubit_h_involutive 3 0
  have hinv1 : e1 * e1 = 1 := embed_single_qubit_h_involutive 3 1
  have hinv2 : e2 * e2 = 1 := embed_single_qubit_h_involutive 3 2
  have c02 : e0 * e2 = e2 * e0 :=
    embed_single_qubit_disjoint_comm 3 0 2 (by decide) Gates.H Gates.H
  have c01 : e0 * e1 = e1 * e0 :=
    embed_single_qubit_disjoint_comm 3 0 1 (by decide) Gates.H Gates.H
  have c12 : e1 * e2 = e2 * e1 :=
    embed_single_qubit_disjoint_comm 3 1 2 (by decide) Gates.H Gates.H
  have inner : e0 * (e2 * (e1 * e0)) = e2 * e1 := by
    rw [← mul_assoc e0 e2, c02, mul_assoc e2 e0, ← mul_assoc e0 e1, c01,
        mul_assoc e1 e0, hinv0, mul_one]
  have inner2 : e1 * (e2 * e1) = e2 := by
    rw [← mul_assoc e1 e2, c12, mul_assoc e2 e1, hinv1, mul_one]
  simp only [mul_assoc]
  rw [inner, inner2, hinv2]

/-- **Diffusion structural split.** Because the palindromic `X`-layers make
    `diffusionGates` literally `hLayerGates ++ diffusionDiagCoreGates ++
    hLayerGates`, its denotation factors as `HL · (I−2|000⟩⟨000|) · HL` — the
    Hadamard layer conjugating the diagonal core `oracle 3 ⟨0⟩`
    (`denote_diffusionDiagCore`). The remaining diffusion-bridge step is the
    rank-1 algebra `HL · (I−2|e0⟩⟨e0|) · HL = I−2|s⟩⟨s| = −diffusion_operator 3`
    (with `s = HL·e0 = uniform`, `HL·HL = I` from `denote_hLayer_involutive`). -/
theorem denote_diffusion_split :
    denote diffusionGates
      = denote hLayerGates * oracle 3 ⟨0, by norm_num⟩ * denote hLayerGates := by
  have hsplit : diffusionGates
      = hLayerGates ++ diffusionDiagCoreGates ++ hLayerGates := rfl
  rw [hsplit, denote_append, denote_append, denote_diffusionDiagCore, mul_assoc]

/-- The Hadamard-layer denotation is the abstract `H^⊗3` (`HadamardLayer.hadamardLayer 3`):
    both are the left-folded product of the three single-qubit `H` embeds. Bridges
    to `hadamardLayer_col0` (column 0 is the uniform amplitude). -/
theorem denote_hLayer_eq_hadamardLayer :
    denote hLayerGates = QuantumProofs.HadamardLayer.hadamardLayer 3 := by
  rw [QuantumProofs.HadamardLayer.hadamardLayer_eq_denote]
  rfl

/-- **The Hadamard layer is symmetric (Hermitian, real):** `(H^⊗3)ᴴ = H^⊗3`.
    `Hᴴ = H` per wire (`embed_single_qubit_conjTranspose`); conjugate-transposing
    reverses the product order, which the disjoint-wire commutation restores. With
    `hadamardLayer_col0` this gives row 0 of `HL` is also the uniform amplitude. -/
theorem denote_hLayer_symm :
    (denote hLayerGates)ᴴ = denote hLayerGates := by
  rw [denote_hLayer]
  have hH : (Gates.H)ᴴ = Gates.H := by
    ext a b; fin_cases a <;> fin_cases b <;>
      simp [Gates.H, Matrix.conjTranspose_apply, star_div₀, star_one, ← starRingEnd_apply,
            Complex.conj_ofReal]
  rw [Matrix.conjTranspose_mul, Matrix.conjTranspose_mul,
      embed_single_qubit_conjTranspose, embed_single_qubit_conjTranspose,
      embed_single_qubit_conjTranspose, hH]
  set e0 := embed_single_qubit 3 0 Gates.H
  set e1 := embed_single_qubit 3 1 Gates.H
  set e2 := embed_single_qubit 3 2 Gates.H
  have c01 : e0 * e1 = e1 * e0 :=
    embed_single_qubit_disjoint_comm 3 0 1 (by decide) Gates.H Gates.H
  have c02 : e0 * e2 = e2 * e0 :=
    embed_single_qubit_disjoint_comm 3 0 2 (by decide) Gates.H Gates.H
  have c12 : e1 * e2 = e2 * e1 :=
    embed_single_qubit_disjoint_comm 3 1 2 (by decide) Gates.H Gates.H
  rw [← mul_assoc, c01, mul_assoc, c02, ← mul_assoc, c12]

/-- **Grover diffusion gate-model bridge.** The `H^⊗3 · X^⊗3 · CCZ · X^⊗3 · H^⊗3`
    circuit denotes to `−diffusion_operator 3` (the abstract Grover diffusion, up
    to the harmless global `−1`). Proof: the structural split gives
    `HL · (I−2|000⟩⟨000|) · HL`; entrywise this is `(HL·HL)ᵢⱼ − 2·HLᵢ₀·HL₀ⱼ =
    δᵢⱼ − 2·(1/√2)⁶ = δᵢⱼ − 1/4`, using `HL·HL = I` (`denote_hLayer_involutive`),
    column 0 `HLᵢ₀ = (1/√2)³` (`hadamardLayer_col0`), and row 0 `HL₀ⱼ = (1/√2)³`
    (symmetry, `denote_hLayer_symm`). For `N = 8`, `δᵢⱼ − 1/4 = −(2/8 − [i=j]) =
    −diffusion_operator 3 ᵢⱼ`. The `−1` global phase is harmless under `normSq` in
    the measurement certificate. -/
theorem denote_diffusionGates :
    denote diffusionGates = -(diffusion_operator 3) := by
  rw [denote_diffusion_split, oracle_diagonal]
  set HL := denote hLayerGates with hHLdef
  have hinv : HL * HL = 1 := denote_hLayer_involutive
  have hcol : ∀ i : Fin (2 ^ 3), HL i 0 = (1 / Complex.ofReal (Real.sqrt 2)) ^ 3 := by
    intro i; rw [hHLdef, denote_hLayer_eq_hadamardLayer]
    exact QuantumProofs.HadamardLayer.hadamardLayer_col0 3 i
  have hsymm : HLᴴ = HL := by rw [hHLdef]; exact denote_hLayer_symm
  have hrow : ∀ j : Fin (2 ^ 3), HL 0 j = (1 / Complex.ofReal (Real.sqrt 2)) ^ 3 := by
    intro j
    have h1 : HLᴴ 0 j = star (HL j 0) := Matrix.conjTranspose_apply HL j 0
    rw [hsymm] at h1
    rw [h1, hcol j, star_pow, star_div₀, star_one, Complex.star_def, Complex.conj_ofReal]
  have hpow : ((1 / Complex.ofReal (Real.sqrt 2)) ^ 3)
      * ((1 / Complex.ofReal (Real.sqrt 2)) ^ 3) = 1 / 8 := by
    rw [← pow_add, div_pow, one_pow, ← Complex.ofReal_pow]
    have h6 : (Real.sqrt 2) ^ 6 = 8 := by
      have h2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
      calc Real.sqrt 2 ^ 6 = (Real.sqrt 2 ^ 2) ^ 3 := by ring
        _ = 2 ^ 3 := by rw [h2]
        _ = 8 := by norm_num
    rw [h6]; norm_num
  ext i j
  rw [Matrix.mul_apply]
  have key : ∀ l : Fin (2 ^ 3),
      (HL * Matrix.diagonal
          (fun k => if k = (⟨0, by norm_num⟩ : Fin (2 ^ 3)) then (-1 : ℂ) else 1)) i l * HL l j
        = HL i l * HL l j
          - 2 * (HL i l * (if l = (⟨0, by norm_num⟩ : Fin (2 ^ 3)) then (1 : ℂ) else 0) * HL l j) := by
    intro l
    rw [Matrix.mul_diagonal]
    by_cases hl : l = (⟨0, by norm_num⟩ : Fin (2 ^ 3))
    · rw [if_pos hl, if_pos hl]; ring
    · rw [if_neg hl, if_neg hl]; ring
  rw [Finset.sum_congr rfl (fun l _ => key l), Finset.sum_sub_distrib,
      ← Matrix.mul_apply, hinv, ← Finset.mul_sum,
      Finset.sum_eq_single (⟨0, by norm_num⟩ : Fin (2 ^ 3))]
  · -- main term
    have h00 : (⟨0, by norm_num⟩ : Fin (2 ^ 3)) = (0 : Fin (2 ^ 3)) := rfl
    rw [h00, if_pos rfl, mul_one, hcol i, hrow j, hpow, Matrix.one_apply, Matrix.neg_apply]
    simp only [diffusion_operator]
    by_cases hij : i = j
    · simp only [if_pos hij]; push_cast; norm_num
    · simp only [if_neg hij]; push_cast; norm_num
  · intro l _ hl; rw [if_neg hl, mul_zero, zero_mul]
  · intro h; exact absurd (Finset.mem_univ _) h

/-- One **Grover iteration** as a gate list for the canonical marked state
    `|111⟩` (index `7`): the oracle (bare `ccz`) followed by the diffusion. -/
noncomputable def groverIterTop : Circuit 3 := oracleGatesTop ++ diffusionGates

/-- **Grover iteration gate-model bridge (canonical marked state).** One full
    iteration `oracle · diffusion` denotes to `−grover_operator 3 ⟨7⟩` — the
    abstract Grover operator `G = D·O_w` (up to the harmless global `−1` carried
    by the diffusion, immaterial under `normSq`). Combines the oracle core
    identity (`denote_oracleGatesTop`) and the diffusion bridge
    (`denote_diffusionGates`): `denote (oracle ++ diffusion) = D · O =
    (−diffusion_operator)·oracle = −(diffusion_operator · oracle) =
    −grover_operator`. -/
theorem denote_groverIterTop :
    denote groverIterTop = -(grover_operator 3 ⟨7, by norm_num⟩) := by
  rw [groverIterTop, denote_append, denote_diffusionGates, denote_oracleGatesTop,
      grover_operator, neg_mul]

/-- `k`-fold repetition of a circuit denotes to the `k`-th power of its
    denotation (`denote` of `c` concatenated with itself `k` times). -/
theorem denote_flatten_replicate (c : Circuit 3) (k : ℕ) :
    denote ((List.replicate k c).flatten) = (denote c) ^ k := by
  induction k with
  | zero => rw [List.replicate_zero, List.flatten_nil, empty_circuit_identity, pow_zero]
  | succ k ih =>
      rw [List.replicate_succ, List.flatten_cons, denote_append, ih]
      exact (pow_succ _ _).symm

/-- The full **`k`-iteration Grover circuit** for the canonical marked `|111⟩`:
    `k` copies of `oracle · diffusion`. -/
noncomputable def groverIterTopPow (k : ℕ) : Circuit 3 :=
  (List.replicate k groverIterTop).flatten

/-- The `k`-iteration circuit denotes to `(−grover_operator 3 ⟨7⟩)^k`. -/
theorem denote_groverIterTopPow (k : ℕ) :
    denote (groverIterTopPow k) = (-(grover_operator 3 ⟨7, by norm_num⟩)) ^ k := by
  rw [groverIterTopPow, denote_flatten_replicate, denote_groverIterTop]

/-- **Grover gate-model measurement certificate (canonical marked state).** Running
    the gate circuit of `optimal_iterations 3` Grover iterations on the uniform
    state and measuring yields the marked item `|111⟩` with probability `≥ 1 − 1/8`.
    The gate denotation is `(−grover_operator 3 ⟨7⟩)^k = (−1)^k • (grover_operator)^k`;
    the global `(−1)^k` has unit `normSq` so it cancels in the measurement
    probability, which therefore equals the abstract one — discharged by
    `grover_optimal_success`. -/
theorem grover_gate_optimal_success :
    Complex.normSq
        (amplitude
          (apply_unitary (denote (groverIterTopPow (optimal_iterations 3))) (uniform_state 3))
          ⟨7, by norm_num⟩)
      ≥ 1 - 1 / (2 ^ 3 : ℕ) := by
  set w : Fin (2 ^ 3) := ⟨7, by norm_num⟩
  set k := optimal_iterations 3
  set G := grover_operator 3 w with hG
  rw [denote_groverIterTopPow]
  have hpow : (-(G)) ^ k = (-1 : ℂ) ^ k • G ^ k := by
    rw [show -(G) = (-1 : ℂ) • G from (neg_one_smul ℂ G).symm, smul_pow]
  have key : amplitude (apply_unitary ((-(G)) ^ k) (uniform_state 3)) w
      = (-1 : ℂ) ^ k * amplitude (apply_unitary (G ^ k) (uniform_state 3)) w := by
    rw [hpow]
    unfold amplitude apply_unitary
    simp only [Matrix.smul_apply, smul_eq_mul]
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _; ring
  rw [key, Complex.normSq_mul, map_pow, Complex.normSq_neg, Complex.normSq_one,
      one_pow, one_mul]
  exact grover_optimal_success 3 w (by norm_num)

end QuantumProofs.GroverCircuit
