/-
  D2/LE4 — DEPTH-G depolarizing fidelity (the per-algorithm fidelity floor).

  The single-application result `depolarizing_fidelity` (`⟨ψ|E_p(|ψ⟩⟨ψ|)|ψ⟩ = 1 − p/2`) is lifted
  to a depth-`G` circuit under the **per-gate global-depolarizing** error model: a circuit of `G`
  gates is modelled as `G` sequential depolarizing channels of rate `p`.  The `G`-fold iterate has
  the closed form

      E_p^[G](ρ) = (1−p)^G · ρ + ((1−(1−p)^G)/2) · (Tr ρ) · 1        (`depolarizing_iterate_apply`)

  and hence, for a normalized state, the output fidelity is

      ⟨ψ| E_p^[G](|ψ⟩⟨ψ|) |ψ⟩ = (1 + (1−p)^G) / 2                    (`depolarizing_iterate_fidelity`)

  — a per-algorithm floor parameterised by the gate count `G` (at `G=1` it is `1 − p/2`).  The
  monotone knee `F_G ≥ τ ⟺ (1−p)^G ≥ 2τ−1` (`depolarizing_depth_threshold`) gives the admissible
  per-gate noise for a target fidelity at a given depth.  Instantiated at `G = 3` this is the
  circulant `CyclicShift` solver (CCX·CX·X, 3 gates on 3 qubits).

  Sorry-free; pure induction on `G` reusing `Depolarizing.lean`.
-/
import Mathlib
import QuantumProofs.Noise.Channel
import QuantumProofs.Noise.Pauli
import QuantumProofs.Noise.Depolarizing

namespace QuantumProofs.Noise

open Matrix
open scoped ComplexOrder

/-- **Closed form of the depth-`G` (iterated) depolarizing channel.**
`E_p^[G](ρ) = (1−p)^G·ρ + ((1−(1−p)^G)/2)·(Tr ρ)·1`. -/
theorem depolarizing_iterate_apply (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) (G : ℕ)
    (ρ : Matrix (Fin 2) (Fin 2) ℂ) :
    ((depolarizing p hp hp1).apply)^[G] ρ
      = (((1 - p) ^ G : ℝ) : ℂ) • ρ
        + (((1 - (1 - p) ^ G) / 2 : ℝ) : ℂ) • (ρ.trace • (1 : Matrix (Fin 2) (Fin 2) ℂ)) := by
  induction G with
  | zero => simp
  | succ G ih =>
    rw [Function.iterate_succ', Function.comp_apply, ih, depolarizing_apply]
    simp only [Matrix.trace_add, Matrix.trace_smul, Matrix.trace_one, Fintype.card_fin,
      smul_smul, smul_eq_mul, Nat.cast_ofNat]
    match_scalars <;> push_cast [pow_succ] <;> ring

/-- **Depth-`G` depolarizing fidelity (the per-algorithm floor).**  For a normalized single-qubit
state `ψ`, the fidelity after `G` per-gate depolarizing channels is `(1 + (1−p)^G)/2`. -/
theorem depolarizing_iterate_fidelity (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) (G : ℕ)
    (ψ : Fin 2 → ℂ) (hψ : star ψ ⬝ᵥ ψ = 1) :
    expVal ψ (((depolarizing p hp hp1).apply)^[G] (vecMulVec ψ (star ψ)))
      = (((1 + (1 - p) ^ G) / 2 : ℝ) : ℂ) := by
  unfold expVal
  rw [depolarizing_iterate_apply, add_mulVec, smul_mulVec, smul_mulVec, dotProduct_add,
    dotProduct_smul, dotProduct_smul, pure_exp, smul_mulVec, one_mulVec, dotProduct_smul,
    trace_pure, hψ]
  simp only [smul_eq_mul, mul_one]
  push_cast; ring

/-- The real-valued depth-`G` depolarizing fidelity `F_G(p) = (1 + (1−p)^G)/2`. -/
noncomputable def depoDepthFidelity (p : ℝ) (G : ℕ) : ℝ := (1 + (1 - p) ^ G) / 2

/-- **Depth-`G` robustness threshold.**  A depth-`G` circuit meets target fidelity `τ` iff the
surviving coherent weight `(1−p)^G` stays at or above `2τ−1` — the per-algorithm knee. -/
theorem depolarizing_depth_threshold (p : ℝ) (G : ℕ) (τ : ℝ) :
    depoDepthFidelity p G ≥ τ ↔ (1 - p) ^ G ≥ 2 * τ - 1 := by
  unfold depoDepthFidelity
  constructor <;> intro h <;> linarith

/-- **Circulant `CyclicShift` fidelity floor.**  The 3-gate circulant solver (CCX·CX·X) under
per-gate depolarizing noise has output fidelity `(1 + (1−p)^3)/2`. -/
theorem circulant_cyclicshift_fidelity (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1)
    (ψ : Fin 2 → ℂ) (hψ : star ψ ⬝ᵥ ψ = 1) :
    expVal ψ (((depolarizing p hp hp1).apply)^[3] (vecMulVec ψ (star ψ)))
      = (((1 + (1 - p) ^ 3) / 2 : ℝ) : ℂ) :=
  depolarizing_iterate_fidelity p hp hp1 3 ψ hψ

end QuantumProofs.Noise
