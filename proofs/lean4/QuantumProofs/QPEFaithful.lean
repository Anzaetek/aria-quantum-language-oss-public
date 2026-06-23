/-
Faithful QPE certificate (LEAN_EXPORT Track B — QPE M2/M3 foundation).

The faithful certificate proves the *actual* (n+1)-qubit QPE circuit — eigenstate
prep (`X`), Hadamard layer, controlled-phase loading, and the explicit
inverse-QFT gate list (`QFTExport.qftInverseLowered`) — measures the counting
register to the phase with probability 1, with NO opaque matrix-adjoint and the
eigenstate qubit carried explicitly.

This file collects the building blocks. First up is the arithmetic heart of the
phase loading: controlled-phase `2^(n-1-k)` on counting qubit `k`, with the
eigenstate fixed at |1⟩, multiplies basis state |x⟩ by `exp(2πi·φ·2^(n-1-k))^{x_k}`;
the product over `k` is `exp(2πi·φ·Σ_k 2^(n-1-k) x_k) = exp(2πi·φ·x)` exactly
because the weighted bits reconstruct `x` (`sum_pow_bit`). That is the
`QPE.after_controlled_u` amplitude.

Sorry-free. The remaining circuit-level obligations (eigenstate tensor-factoring,
the low-wire embedding of `qftInverseLowered` onto the counting register, and the
final composition with `QPE.qpe_exact`) build on these.
-/
import QuantumProofs.QFTExport
import QuantumProofs.QPE

namespace QuantumProofs.QPEFaithful

open Finset

/-- **Binary place-value identity.** The bits of `x` weighted by their powers of
    two reconstruct `x mod 2^n`. Proven by induction on `n`: the `succ` step is
    exactly `Nat.mod_mul` (`x % (2^k·2) = x % 2^k + 2^k·(x/2^k % 2)`). -/
theorem sum_pow_bit (n x : ℕ) :
    ∑ j ∈ Finset.range n, 2 ^ j * ((x / 2 ^ j) % 2) = x % 2 ^ n := by
  induction n with
  | zero => rw [Finset.sum_range_zero, pow_zero, Nat.mod_one]
  | succ k ih => rw [Finset.sum_range_succ, ih, pow_succ, Nat.mod_mul]

/-- **QPE-indexed place-value.** Counting qubit `k` (`k = 0 … n-1`) carries the
    controlled-phase weight `2^(n-1-k)` and the bit `(x / 2^(n-1-k)) % 2`, so the
    loaded exponent is `Σ_k 2^(n-1-k)·x_k`. Reindexing `j = n-1-k`
    (`Finset.sum_range_reflect`) turns it into `sum_pow_bit`, giving `x` for
    `x < 2^n`. This is the integer in `exp(2πi·φ·x)` that
    `QPE.after_controlled_u` carries. -/
theorem sum_pow_bit_qpe (n x : ℕ) (hx : x < 2 ^ n) :
    ∑ k ∈ Finset.range n, 2 ^ (n - 1 - k) * ((x / 2 ^ (n - 1 - k)) % 2) = x := by
  rw [show (∑ k ∈ Finset.range n, 2 ^ (n - 1 - k) * ((x / 2 ^ (n - 1 - k)) % 2))
        = ∑ j ∈ Finset.range n, 2 ^ j * ((x / 2 ^ j) % 2)
      from Finset.sum_range_reflect (fun j => 2 ^ j * ((x / 2 ^ j) % 2)) n,
      sum_pow_bit, Nat.mod_eq_of_lt hx]

/-- **Phase loading accumulates to the eigenphase.** The controlled-phase layer
    of QPE multiplies basis state `|x⟩` by the phase `exp(2πi·θ·2^(n-1-k))` on
    counting qubit `k` exactly when its bit `x_k = (x / 2^(n-1-k)) % 2` is set
    (eigenstate fixed at `|1⟩`). The product over all `n` counting qubits is the
    single eigenphase `exp(2πi·θ·x)` — the `QPE.after_controlled_u` amplitude —
    because the weighted bits reconstruct `x` (`sum_pow_bit_qpe`).

    Proof: `Complex.exp_sum` turns the product into `exp` of a sum, `Finset.mul_sum`
    factors out the common `2πiθ`, and the remaining `ℕ`-sum is `x` by
    `sum_pow_bit_qpe`. Sorry-free. -/
theorem prod_controlled_phase (n x : ℕ) (hx : x < 2 ^ n) (θ : ℝ) :
    ∏ k ∈ Finset.range n,
        Complex.exp (2 * ↑Real.pi * Complex.I * ↑θ *
          ↑(2 ^ (n - 1 - k) * ((x / 2 ^ (n - 1 - k)) % 2)))
      = Complex.exp (2 * ↑Real.pi * Complex.I * ↑θ * ↑x) := by
  rw [← Complex.exp_sum]
  congr 1
  rw [← Finset.mul_sum, ← Nat.cast_sum, sum_pow_bit_qpe n x hx]

/-- **The loaded counting-register amplitude is `after_controlled_u`.** With the
    eigenstate fixed at `|1⟩`, the QPE state-prep layer (Hadamards giving the
    `1/√2^n` normalisation, then the controlled-phase cascade) places amplitude
    `(1/√2^n)·∏_k exp(2πi·θ·2^(n-1-k))^{x_k}` on `|x⟩`, which `prod_controlled_phase`
    collapses to exactly `QPE.after_controlled_u n θ x`. This identifies the
    *actual* loaded state with the abstract amplitude vector the QPE measurement
    theorems (`qpe_exact`, `qpe_approximate`) are stated against. Sorry-free. -/
theorem loaded_amp_eq_after_controlled_u (n : ℕ) (θ : ℝ) (x : Fin (2 ^ n)) :
    (1 / Complex.ofReal (Real.sqrt ↑(2 ^ n))) *
      ∏ k ∈ Finset.range n,
        Complex.exp (2 * ↑Real.pi * Complex.I * ↑θ *
          ↑(2 ^ (n - 1 - k) * ((x.val / 2 ^ (n - 1 - k)) % 2)))
      = QPE.after_controlled_u n θ x := by
  rw [prod_controlled_phase n x.val x.isLt θ, QPE.after_controlled_u]

/-! ### Eigenstate-at-top tensor factoring

The faithful QPE circuit carries the eigenstate qubit as **qubit 0** (the MSB of
the index), with the `n` counting qubits as `1 … n`. With that layout the
inverse-QFT stage acts only on the counting wires and is exactly
`CircuitSemantics.embed_subcircuit (QFTExport.qftInverseLowered n)` — reusing all
the `embed_subcircuit` Kronecker machinery (`denote_embed_subcircuit = I₂ ⊗ ·`).

`tensorTop e v` is the state with eigenstate amplitude `e : Fin 2 → ℂ` on qubit 0
and counting amplitude `v : Fin 2^n → ℂ` on the rest (`index b·2^n+r ↦ e b · v r`,
read off through `tailEquiv`). The key bridge `denote_embed_subcircuit_mulVec`
says an embedded counting-register circuit acts on `tensorTop e v` by acting only
on the counting factor `v`, leaving the eigenstate `e` untouched. That is what
lets the IQFT stage compose with `QPE.qpe_exact`, and what factors the final
measurement onto the counting register. -/

open Matrix in
/-- `(I₂ ⊗ A) *ᵥ (e ⊗ v) = e ⊗ (A *ᵥ v)` on the product index type: the top
    (eigenstate) factor is untouched, the bottom factor gets `A`. -/
theorem kron_one_mulVec_tensor {n : ℕ} (A : Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ)
    (e : Fin 2 → ℂ) (v : Fin (2 ^ n) → ℂ) (p : Fin 2 × Fin (2 ^ n)) :
    (Matrix.kronecker (1 : Matrix (Fin 2) (Fin 2) ℂ) A *ᵥ fun q => e q.1 * v q.2) p
      = e p.1 * (A *ᵥ v) p.2 := by
  simp only [Matrix.mulVec, dotProduct, Matrix.kronecker, Matrix.kroneckerMap_apply,
             Matrix.one_apply]
  rw [Fintype.sum_prod_type]
  simp only [ite_mul, one_mul, zero_mul]
  rw [Finset.sum_comm]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, if_true]
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun x2 _ => by ring)

open Matrix CircuitSemantics in
/-- Tensor an eigenstate-qubit amplitude `e : Fin 2 → ℂ` (qubit 0, MSB) with a
    counting-register amplitude `v : Fin 2^n → ℂ` into a full `Fin 2^(n+1)` state
    via `tailEquiv` (`index ↦ e (high bit) · v (low n bits)`). -/
noncomputable def tensorTop {n : ℕ} (e : Fin 2 → ℂ) (v : Fin (2 ^ n) → ℂ) :
    Fin (2 ^ (n + 1)) → ℂ :=
  fun i => e ((tailEquiv n).symm i).1 * v ((tailEquiv n).symm i).2

theorem tensorTop_tailEquiv {n : ℕ} (e : Fin 2 → ℂ) (v : Fin (2 ^ n) → ℂ)
    (q : Fin 2 × Fin (2 ^ n)) :
    tensorTop e v (CircuitSemantics.tailEquiv n q) = e q.1 * v q.2 := by
  simp [tensorTop, Equiv.symm_apply_apply]

open Matrix CircuitSemantics in
/-- **Embedded counting-register circuit acts only on the counting factor.**
    `denote (embed_subcircuit c) *ᵥ tensorTop e v = tensorTop e (denote c *ᵥ v)`:
    the eigenstate qubit (qubit 0) passes through untouched while the embedded
    `c` acts on the counting register. The IQFT-stage bridge for the faithful
    certificate — combines `denote_embed_subcircuit` (`= I₂ ⊗ denote c`) with
    `kron_one_mulVec_tensor` through `submatrix_mulVec_equiv`. Sorry-free. -/
theorem denote_embed_subcircuit_mulVec {n : ℕ} (c : Circuit n)
    (e : Fin 2 → ℂ) (v : Fin (2 ^ n) → ℂ) :
    denote (embed_subcircuit c) *ᵥ tensorTop e v = tensorTop e (denote c *ᵥ v) := by
  rw [denote_embed_subcircuit, Matrix.reindex_apply, Matrix.submatrix_mulVec_equiv,
      Equiv.symm_symm]
  funext i
  simp only [Function.comp_apply]
  show (Matrix.kronecker 1 (denote c) *ᵥ fun q => tensorTop e v (tailEquiv n q))
        ((tailEquiv n).symm i) = tensorTop e (denote c *ᵥ v) i
  rw [show (fun q => tensorTop e v (tailEquiv n q)) = (fun q => e q.1 * v q.2) from by
        funext q; rw [tensorTop_tailEquiv],
      kron_one_mulVec_tensor]
  rfl

open Matrix in
/-- `(U ⊗ I) *ᵥ (e ⊗ v) = (U *ᵥ e) ⊗ v` on the product index type: the top
    (eigenstate) factor gets `U`, the bottom (counting) factor is untouched.
    Mirror of `kron_one_mulVec_tensor`. -/
theorem kron_mulVec_one_tensor {n : ℕ} (U : Matrix (Fin 2) (Fin 2) ℂ)
    (e : Fin 2 → ℂ) (v : Fin (2 ^ n) → ℂ) (p : Fin 2 × Fin (2 ^ n)) :
    (Matrix.kronecker U (1 : Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ) *ᵥ fun q => e q.1 * v q.2) p
      = (U *ᵥ e) p.1 * v p.2 := by
  simp only [Matrix.mulVec, dotProduct, Matrix.kronecker, Matrix.kroneckerMap_apply,
             Matrix.one_apply]
  rw [Fintype.sum_prod_type]
  simp only [mul_ite, mul_one, mul_zero, ite_mul, zero_mul]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, if_true]
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl (fun x1 _ => by ring)

open Matrix CircuitSemantics in
/-- **Head-qubit single gate acts only on the eigenstate factor.**
    `embed_single_qubit (n+1) ⟨0,_⟩ U *ᵥ tensorTop e v = tensorTop (U *ᵥ e) v` —
    a gate on qubit 0 (the eigenstate) leaves the counting register untouched.
    Used for the `X` eigenstate prep and the head `H` of the Hadamard layer.
    Combines `embed_single_qubit_head` (`= U ⊗ I`) with `kron_mulVec_one_tensor`
    through `submatrix_mulVec_equiv`. Sorry-free. -/
theorem embed_single_qubit_head_mulVec {n : ℕ} (U : Matrix (Fin 2) (Fin 2) ℂ)
    (e : Fin 2 → ℂ) (v : Fin (2 ^ n) → ℂ) :
    embed_single_qubit (n + 1) ⟨0, by omega⟩ U *ᵥ tensorTop e v
      = tensorTop (U *ᵥ e) v := by
  rw [embed_single_qubit_head, Matrix.reindex_apply, Matrix.submatrix_mulVec_equiv,
      Equiv.symm_symm]
  funext i
  simp only [Function.comp_apply]
  show (Matrix.kronecker U 1 *ᵥ fun q => tensorTop e v (tailEquiv n q)) ((tailEquiv n).symm i)
        = tensorTop (U *ᵥ e) v i
  rw [show (fun q => tensorTop e v (tailEquiv n q)) = (fun q => e q.1 * v q.2) from by
        funext q; rw [tensorTop_tailEquiv],
      kron_mulVec_one_tensor]
  rfl

/-- The prepared eigenstate amplitude on qubit 0: the basis vector `|1⟩`. -/
noncomputable def eigenOne : Fin 2 → ℂ := fun b => if b = 1 then 1 else 0

/-! ### Hadamard layer → uniform superposition

The QPE state-prep opens with a Hadamard on every counting qubit. `hLayer n` is
that gate list (`H` on each of the `n` qubits, built via `embed_subcircuit`), and
`hLayer_correct` proves it sends the all-zero state `basis0 n` to the uniform
superposition `QPE.after_hadamard n = (1/√2^n)·𝟙`. The induction peels the head
`H` (qubit 0, `embed_single_qubit_head_mulVec`) off the embedded tail
(`denote_embed_subcircuit_mulVec`); the `√` bookkeeping is `tensorTop_hplus_after_hadamard`. -/

/-- The computational all-zero state `|0…0⟩` on `n` qubits. -/
noncomputable def basis0 (n : ℕ) : Fin (2 ^ n) → ℂ := fun i => if i = 0 then 1 else 0

/-- The single-qubit `|+⟩` amplitude `(1/√2, 1/√2)` — `H|0⟩`. -/
noncomputable def hplusVec : Fin 2 → ℂ := fun _ => (1 : ℂ) / Complex.ofReal (Real.sqrt 2)

/-- `tailEquiv` sends the corner `(0,0)` to index `0`. -/
theorem tailEquiv_zero (n : ℕ) : (CircuitSemantics.tailEquiv n) (0, 0) = 0 := by
  apply Fin.ext; simp [CircuitSemantics.tailEquiv, finProdFinEquiv]

open CircuitSemantics in
/-- `|0…0⟩` on `n+1` qubits is `|0⟩` (qubit 0) tensored with `|0…0⟩` on the rest. -/
theorem basis0_succ (n : ℕ) :
    basis0 (n + 1) = tensorTop (fun b => if b = 0 then 1 else 0) (basis0 n) := by
  funext i; simp only [tensorTop, basis0]
  by_cases h : i = 0
  · subst h
    rw [show (tailEquiv n).symm 0 = (0, 0) from by rw [Equiv.symm_apply_eq, tailEquiv_zero]]; simp
  · rw [if_neg h]
    have hne : (tailEquiv n).symm i ≠ (0, 0) := by
      rw [ne_eq, Equiv.symm_apply_eq, tailEquiv_zero]; exact h
    by_cases h1 : ((tailEquiv n).symm i).1 = 0
    · rw [if_pos h1, one_mul, if_neg (fun h2 => hne (Prod.ext h1 h2))]
    · rw [if_neg h1, zero_mul]

open Matrix in
/-- `H|0⟩ = |+⟩`: the Hadamard takes the basis-0 amplitude to the uniform pair. -/
theorem H_mulVec_e0 :
    Gates.H *ᵥ (fun b => if b = 0 then 1 else 0) = hplusVec := by
  funext i; simp only [Matrix.mulVec, dotProduct, Fin.sum_univ_two, hplusVec, Gates.H]
  fin_cases i <;> simp

/-- `|+⟩ ⊗ uniform_n = uniform_{n+1}`: the `√` bookkeeping `(1/√2)·(1/√2^n) = 1/√2^{n+1}`. -/
theorem tensorTop_hplus_after_hadamard (n : ℕ) :
    tensorTop hplusVec (QPE.after_hadamard n) = QPE.after_hadamard (n + 1) := by
  funext i; simp only [tensorTop, QPE.after_hadamard, hplusVec]
  rw [div_mul_div_comm, one_mul, ← Complex.ofReal_mul,
      ← Real.sqrt_mul (by norm_num : (0:ℝ) ≤ 2),
      show (2 : ℝ) * 2 ^ n = 2 ^ (n + 1) from by rw [pow_succ]; ring]

open CircuitSemantics in
/-- The Hadamard layer: `H` on every one of the `n` qubits (head `H` + embedded
    tail), mirroring the `H^⊗n` opening of the QPE state-prep. -/
noncomputable def hLayer : (n : ℕ) → Circuit n
  | 0 => []
  | n + 1 => GateApp.h ⟨0, by omega⟩ :: embed_subcircuit (hLayer n)

open Matrix CircuitSemantics in
/-- **Hadamard layer prepares the uniform superposition.**
    `denote (hLayer n) *ᵥ basis0 n = QPE.after_hadamard n` — running `H^⊗n` on
    `|0…0⟩` yields `(1/√2^n)·∑_x |x⟩`. Induction: the head `H` acts on qubit 0
    (`embed_single_qubit_head_mulVec` + `H_mulVec_e0`), the embedded tail by the IH
    (`denote_embed_subcircuit_mulVec`), recombined by `tensorTop_hplus_after_hadamard`.
    Sorry-free. -/
theorem hLayer_correct (n : ℕ) :
    denote (hLayer n) *ᵥ basis0 n = QPE.after_hadamard n := by
  induction n with
  | zero =>
      funext i; fin_cases i
      simp [hLayer, empty_circuit_identity, basis0, QPE.after_hadamard]
  | succ k ih =>
      have hcons : denote (hLayer (k + 1))
          = denote (embed_subcircuit (hLayer k))
            * denote_gate (GateApp.h (⟨0, by omega⟩ : Fin (k + 1))) := by
        show denote (GateApp.h _ :: embed_subcircuit (hLayer k)) = _
        rw [denote_cons]
      rw [hcons, ← Matrix.mulVec_mulVec, basis0_succ,
          show denote_gate (GateApp.h (⟨0, by omega⟩ : Fin (k + 1)))
             = embed_single_qubit (k + 1) ⟨0, by omega⟩ Gates.H from rfl,
          embed_single_qubit_head_mulVec, H_mulVec_e0,
          denote_embed_subcircuit_mulVec, ih, tensorTop_hplus_after_hadamard]

open Matrix in
/-- `X|0⟩ = |1⟩`: the eigenstate prep flips qubit 0 into the prepared `|1⟩`. -/
theorem X_mulVec_e0 :
    Gates.X *ᵥ (fun b => if b = 0 then 1 else 0) = eigenOne := by
  funext i; simp only [Matrix.mulVec, dotProduct, Fin.sum_univ_two, eigenOne, Gates.X]
  fin_cases i <;> simp

open Matrix CircuitSemantics in
/-- **State-prep up to phase loading.** Eigenstate prep (`X` on qubit 0) followed
    by the Hadamard layer on the counting register (`embed_subcircuit (hLayer n)`)
    takes `|0…0⟩` to `tensorTop eigenOne (after_hadamard n)` — the prepared `|1⟩`
    eigenstate tensored with the uniform counting-register superposition. Composes
    `X_mulVec_e0` + `embed_single_qubit_head_mulVec` (X prep) with `hLayer_correct`
    + `denote_embed_subcircuit_mulVec` (H layer). The only remaining front-half gap
    is the controlled-phase cascade `tensorTop eigenOne after_hadamard →
    tensorTop eigenOne after_controlled_u`. Sorry-free. -/
theorem prep_hadamard (n : ℕ) :
    denote (embed_subcircuit (hLayer n))
      *ᵥ (embed_single_qubit (n + 1) ⟨0, by omega⟩ Gates.X *ᵥ basis0 (n + 1))
      = tensorTop eigenOne (QPE.after_hadamard n) := by
  rw [basis0_succ, embed_single_qubit_head_mulVec, X_mulVec_e0,
      denote_embed_subcircuit_mulVec, hLayer_correct]

/-! ### Measurement factoring

The eigenstate basis amplitude `eigenOne` is the prepared `|1⟩` on qubit 0. The
faithful certificate measures only the *counting register*: `counting_marginal`
sums the squared moduli over the eigenstate qubit. For a product state
`tensorTop eigenOne ψ` that marginal is exactly `QPE.measure_prob ψ` on the
counting register, so the QPE measurement theorems transport verbatim. -/

/-- `∑_b ‖eigenOne b‖² = 1` — `|1⟩` is normalised. -/
theorem sum_normSq_eigenOne : ∑ b : Fin 2, Complex.normSq (eigenOne b) = 1 := by
  simp [eigenOne]

open CircuitSemantics in
/-- Probability of measuring counting-register outcome `m`, marginalising over
    the eigenstate qubit (qubit 0): `∑_b ‖ψ (b·2^n + m)‖²`. -/
noncomputable def counting_marginal {n : ℕ} (ψ : Fin (2 ^ (n + 1)) → ℂ)
    (m : Fin (2 ^ n)) : ℝ :=
  ∑ b : Fin 2, Complex.normSq (ψ (tailEquiv n (b, m)))

open CircuitSemantics in
/-- **Measurement factors onto the counting register.** For a product state with
    the eigenstate prepared at `|1⟩`, the counting-register marginal equals the
    single-register QPE probability `QPE.measure_prob ψ m = ‖ψ m‖²`. This is what
    lets `qpe_exact`/`qpe_approximate` (stated on the bare counting amplitude)
    certify the *actual* `(n+1)`-qubit measurement. Sorry-free. -/
theorem counting_marginal_tensorTop (n : ℕ) (ψ : Fin (2 ^ n) → ℂ) (m : Fin (2 ^ n)) :
    counting_marginal (tensorTop eigenOne ψ) m = QPE.measure_prob ψ m := by
  unfold counting_marginal QPE.measure_prob
  have h : ∀ b : Fin 2,
      Complex.normSq (tensorTop eigenOne ψ (tailEquiv n (b, m)))
        = Complex.normSq (eigenOne b) * Complex.normSq (ψ m) := by
    intro b; rw [tensorTop_tailEquiv, Complex.normSq_mul]
  rw [Finset.sum_congr rfl (fun b _ => h b), ← Finset.sum_mul, sum_normSq_eigenOne, one_mul]

/-! ### Back-half composition: IQFT stage + measurement

Composing the pieces above: starting from the loaded counting-register state
`after_controlled_u n θ` tensored with the prepared eigenstate `|1⟩` (qubit 0),
running the *real gate list* inverse-QFT on the counting register
(`embed_subcircuit (qftInverseLowered n)`, control-on-higher-wire SWAP + reverse
H/CP sweep) and measuring the counting register yields the phase `m` with
probability 1 — for exact phases `θ = m/2^n`.

This discharges the entire back half of the faithful certificate against actual
circuit denotation (no opaque matrix adjoint): the IQFT is the explicit
`qftInverseLowered` gate list, embedded on the counting wires, and the
measurement is the honest counting-register marginal. The remaining front-half
obligation is the state-prep layer (`X` eigenstate + `H^⊗n` + controlled-phase
cascade) producing `tensorTop eigenOne (after_controlled_u n θ)` — for which
`loaded_amp_eq_after_controlled_u` and `denote_embed_subcircuit_mulVec` (for the
`H` layer) are the building blocks. -/

open Matrix CircuitSemantics QFTExport QPE in
/-- **Faithful QPE back-half, exact case.** Running the explicit inverse-QFT gate
    list on the counting wires of the loaded `(n+1)`-qubit state and measuring the
    counting register gives outcome `m` with probability 1, for `θ = m/2^n`.
    All real-gate denotation; no matrix-adjoint caveat. Sorry-free. -/
theorem qpe_iqft_measure_exact (n : ℕ) (m : Fin (2 ^ n)) (θ : ℝ)
    (hθ : θ = ↑m.val / ↑(2 ^ n)) :
    counting_marginal
        (denote (embed_subcircuit (qftInverseLowered n))
          *ᵥ tensorTop eigenOne (after_controlled_u n θ)) m = 1 := by
  rw [denote_embed_subcircuit_mulVec, counting_marginal_tensorTop,
      qftInverseLowered_correct, ← apply_inverse_dft_eq_mulVec, qpe_exact n m θ hθ]

/-! ### The controlled-phase cascade and the full faithful certificate

The final front-half piece: the QPE controlled-phase loading layer `cpCascade n θ`
(for counting qubit `k+1`, a `CP` with the eigenstate qubit 0 carrying weight
`2^(n-1-k)`, mirroring `qpe.aria`). Every gate is diagonal (`denote_gate_cp_diagonal`),
so the whole cascade is one diagonal matrix; with the eigenstate prepared at `|1⟩`
its diagonal phase on `|x⟩` is `∏_k exp(2πiθ·2^(n-1-k))^{x_k} = exp(2πiθ·x)`
(`prod_controlled_phase`). Hence `cpCascade_mulVec` sends `tensorTop eigenOne
(after_hadamard n)` to `tensorTop eigenOne (after_controlled_u n θ)`.

`qpe_faithful` then composes the whole pipeline — `X` eigenstate prep + `H^⊗n` layer
(`prep_hadamard`) → controlled-phase cascade (`cpCascade_mulVec`) → inverse-QFT gate
list + measurement (`qpe_iqft_measure_exact`) — into the no-caveat certificate: running
the *actual* `(n+1)`-qubit QPE circuit on `|0…0⟩` and measuring the counting register
yields the phase `m` with probability 1, for exact `θ = m/2^n`. -/

section Cascade
open Matrix CircuitSemantics

theorem bit_lo (n m : ℕ) (hm : m < n) (B R : ℕ) :
    ((B * 2 ^ n + R) / 2 ^ m) % 2 = (R / 2 ^ m) % 2 := by
  have hsplit : (2:ℕ) ^ n = 2 ^ (n - m) * 2 ^ m := by rw [← pow_add]; congr 1; omega
  rw [hsplit, ← mul_assoc, Nat.add_comm, Nat.add_mul_div_right _ _ (by positivity)]
  rcases Nat.exists_eq_add_of_lt hm with ⟨d, hd⟩
  rw [show n - m = d + 1 from by omega, pow_succ, ← mul_assoc, Nat.add_mul_mod_self_right]
theorem bit_hi (n : ℕ) (B R : ℕ) (hB : B < 2) (hR : R < 2 ^ n) :
    ((B * 2 ^ n + R) / 2 ^ n) % 2 = B := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by positivity), Nat.div_eq_of_lt hR,
      Nat.zero_add, Nat.mod_eq_of_lt hB]
theorem tailEquiv_val (n : ℕ) (b : Fin 2) (r : Fin (2 ^ n)) :
    (tailEquiv n (b, r)).val = b.val * 2 ^ n + r.val := by
  simp [tailEquiv, finProdFinEquiv]; ring

noncomputable def cpCascade (n : ℕ) (θ : ℝ) : Circuit (n + 1) :=
  List.ofFn (fun k : Fin n =>
    GateApp.cp ⟨0, by omega⟩ ⟨k.val + 1, by have := k.isLt; omega⟩
      (2 * Real.pi * θ * 2 ^ (n - 1 - k.val))
      (Fin.ne_of_val_ne (show (0 : ℕ) ≠ k.val + 1 from by omega)))
noncomputable def cpD (n : ℕ) : GateApp (n + 1) → Fin (2 ^ (n + 1)) → ℂ :=
  fun g i => match g with
    | .cp c t θ _ => (![1, 1, 1, Complex.exp ((θ : ℝ) * Complex.I)] : Fin 4 → ℂ)
        ⟨(i.val / 2 ^ (n + 1 - 1 - c.val)) % 2 * 2 + (i.val / 2 ^ (n + 1 - 1 - t.val)) % 2, by omega⟩
    | _ => 1
theorem cpCascade_diag (n : ℕ) (θ : ℝ) :
    denote (cpCascade n θ)
      = Matrix.diagonal (fun i => ((cpCascade n θ).map (fun g => cpD n g i)).prod) := by
  apply denote_diag_list (cpD n)
  intro g hg
  simp only [cpCascade, List.mem_ofFn] at hg
  obtain ⟨k, rfl⟩ := hg
  rw [denote_gate_cp_diagonal]; rfl
theorem cpCascade_entry (n : ℕ) (θ : ℝ) (i : Fin (2 ^ (n + 1))) :
    ((cpCascade n θ).map (fun g => cpD n g i)).prod
      = ∏ k : Fin n, (![1, 1, 1, Complex.exp ((2*Real.pi*θ*2^(n-1-k.val):ℝ)*Complex.I)] : Fin 4 → ℂ)
          ⟨(i.val/2^n)%2*2 + (i.val/2^(n-1-k.val))%2, by omega⟩ := by
  rw [cpCascade, List.map_ofFn, List.prod_ofFn]
  apply Finset.prod_congr rfl
  intro k _
  have h2 : n - (k.val + 1) = n - 1 - k.val := by omega
  simp only [Function.comp_apply, cpD, Nat.add_sub_cancel, Nat.sub_zero, h2]
theorem cp_phase_term (θ : ℝ) (w c : ℕ) (hc2 : 2 + c < 4) (hc : c < 2) :
    (![1, 1, 1, Complex.exp ((2 * Real.pi * θ * 2 ^ w : ℝ) * Complex.I)] : Fin 4 → ℂ) ⟨2 + c, hc2⟩
      = Complex.exp (2 * ↑Real.pi * Complex.I * ↑θ * ↑(2 ^ w * c)) := by
  interval_cases c
  · have hv : (![1, 1, 1, Complex.exp ((2 * Real.pi * θ * 2 ^ w : ℝ) * Complex.I)] : Fin 4 → ℂ)
        ⟨2 + 0, hc2⟩ = 1 := rfl
    rw [hv, Nat.mul_zero, Nat.cast_zero, mul_zero, Complex.exp_zero]
  · have hv : (![1, 1, 1, Complex.exp ((2 * Real.pi * θ * 2 ^ w : ℝ) * Complex.I)] : Fin 4 → ℂ)
        ⟨2 + 1, hc2⟩ = Complex.exp ((2 * Real.pi * θ * 2 ^ w : ℝ) * Complex.I) := rfl
    rw [hv]; push_cast; ring_nf

theorem cpCascade_mulVec (n : ℕ) (θ : ℝ) :
    denote (cpCascade n θ) *ᵥ tensorTop eigenOne (QPE.after_hadamard n)
      = tensorTop eigenOne (QPE.after_controlled_u n θ) := by
  rw [cpCascade_diag]
  funext i
  rw [Matrix.mulVec_diagonal, cpCascade_entry]
  have hival : i.val = ((tailEquiv n).symm i).1.val * 2 ^ n + ((tailEquiv n).symm i).2.val := by
    have h := tailEquiv_val n ((tailEquiv n).symm i).1 ((tailEquiv n).symm i).2
    rwa [Prod.mk.eta, Equiv.apply_symm_apply] at h
  simp only [tensorTop]
  set b := ((tailEquiv n).symm i).1 with hbdef
  set r := ((tailEquiv n).symm i).2 with hrdef
  by_cases hb1 : b = 1
  · -- eigenstate bit = 1
    have hbv : b.val = 1 := by rw [hb1]; rfl
    have hbit_hi : (i.val / 2 ^ n) % 2 = 1 := by
      rw [hival, bit_hi n b.val r.val (by omega) r.isLt, hbv]
    have hprod : (∏ k : Fin n, (![1, 1, 1, Complex.exp ((2*Real.pi*θ*2^(n-1-k.val):ℝ)*Complex.I)] : Fin 4 → ℂ)
          ⟨(i.val/2^n)%2*2 + (i.val/2^(n-1-k.val))%2, by omega⟩)
        = Complex.exp (2 * ↑Real.pi * Complex.I * ↑θ * ↑r.val) := by
      have hterm : ∀ k : Fin n,
          (![1, 1, 1, Complex.exp ((2*Real.pi*θ*2^(n-1-k.val):ℝ)*Complex.I)] : Fin 4 → ℂ)
            ⟨(i.val/2^n)%2*2 + (i.val/2^(n-1-k.val))%2, by omega⟩
          = Complex.exp (2 * ↑Real.pi * Complex.I * ↑θ * ↑(2^(n-1-k.val) * ((r.val/2^(n-1-k.val))%2))) := by
        intro k
        have hlo : (i.val/2^(n-1-k.val))%2 = (r.val/2^(n-1-k.val))%2 := by
          rw [hival]; exact bit_lo n (n-1-k.val) (by have := k.isLt; omega) b.val r.val
        have hidx : (i.val/2^n)%2*2 + (i.val/2^(n-1-k.val))%2 = 2 + (r.val/2^(n-1-k.val))%2 := by
          rw [hbit_hi, hlo]
        rw [show (![1, 1, 1, Complex.exp ((2*Real.pi*θ*2^(n-1-k.val):ℝ)*Complex.I)] : Fin 4 → ℂ)
              ⟨(i.val/2^n)%2*2 + (i.val/2^(n-1-k.val))%2, by omega⟩
            = (![1, 1, 1, Complex.exp ((2*Real.pi*θ*2^(n-1-k.val):ℝ)*Complex.I)] : Fin 4 → ℂ)
              ⟨2 + (r.val/2^(n-1-k.val))%2, by omega⟩ from by congr 1; exact Fin.ext hidx]
        exact cp_phase_term θ (n-1-k.val) ((r.val/2^(n-1-k.val))%2) (by omega) (by omega)
      rw [Finset.prod_congr rfl (fun k _ => hterm k),
          Fin.prod_univ_eq_prod_range (fun j => Complex.exp (2 * ↑Real.pi * Complex.I * ↑θ * ↑(2^(n-1-j) * ((r.val/2^(n-1-j))%2)))) n]
      exact prod_controlled_phase n r.val r.isLt θ
    rw [hprod]
    simp only [eigenOne, hb1, if_true, one_mul]
    rw [QPE.after_hadamard, QPE.after_controlled_u]
    ring
  · -- eigenstate bit = 0
    have : eigenOne b = 0 := by simp only [eigenOne]; rw [if_neg hb1]
    rw [this]; ring

noncomputable def qpeCircuit (n : ℕ) (θ : ℝ) : Circuit (n + 1) :=
  (GateApp.x ⟨0, by omega⟩ :: embed_subcircuit (hLayer n))
    ++ cpCascade n θ
    ++ embed_subcircuit (QFTExport.qftInverseLowered n)

theorem qpe_faithful (n : ℕ) (m : Fin (2 ^ n)) (θ : ℝ) (hθ : θ = ↑m.val / ↑(2 ^ n)) :
    counting_marginal (denote (qpeCircuit n θ) *ᵥ basis0 (n + 1)) m = 1 := by
  have hstate : denote (qpeCircuit n θ) *ᵥ basis0 (n + 1)
      = denote (embed_subcircuit (QFTExport.qftInverseLowered n))
          *ᵥ tensorTop eigenOne (QPE.after_controlled_u n θ) := by
    rw [qpeCircuit, denote_append, denote_append, denote_cons,
        ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec,
        show denote_gate (GateApp.x (⟨0, by omega⟩ : Fin (n + 1)))
           = embed_single_qubit (n + 1) ⟨0, by omega⟩ Gates.X from rfl,
        prep_hadamard, cpCascade_mulVec]
  rw [hstate]
  exact qpe_iqft_measure_exact n m θ hθ

end Cascade

end QuantumProofs.QPEFaithful
