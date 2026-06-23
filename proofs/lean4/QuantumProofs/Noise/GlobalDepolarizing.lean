/-
  D2/LE4 — GLOBAL (register-wide) depolarizing on an entangled n-qubit output.

  The product-state width floor (`TensorChannel.lean`) assumes a product input; a real circuit
  output (e.g. the circulant `CyclicShift` register) can be entangled.  The **global** depolarizing
  channel on the whole `d`-dimensional register (`d = 2^n`),

      E_p(ρ) = (1 − p)·ρ + (p/d)·(Tr ρ)·1 ,

  is basis- and entanglement-INDEPENDENT (it shrinks toward the maximally-mixed state `1/d`), so its
  fidelity law holds for ANY pure state, entangled or not.  Iterating it over `G` gates gives the
  combined depth-`G` × width-`n` floor

      ⟨ψ| E_p^[G](|ψ⟩⟨ψ|) |ψ⟩ = (1 − p)^G + (1 − (1 − p)^G)/d        (`globalDepol_fidelity`)

  valid for every normalized `ψ : Fin d → ℂ`.  At `d = 2` it is the single-qubit `1 − p/2` law; at
  `d = 8` (`globalDepol_circulant_fidelity`) it is the circulant 3-qubit register under per-gate
  global depolarizing — the honest entangled-output model the per-qubit width floor cannot cover.

  The channel is trace-preserving (`globalDepol_trace`); it is the convex combination
  `(1−p)·id + p·(ρ ↦ (Tr ρ/d)·1)` of two CPTP maps, so complete-positivity is the standard
  Heisenberg–Weyl-twirl property (at `d = 2` it is the `KrausMap` `depolarizing`).  Here we prove
  the trace-preservation and the fidelity law — the numerically-meaningful floor.  Sorry-free.
-/
import Mathlib
import QuantumProofs.Noise.Channel
import QuantumProofs.Noise.Depolarizing

namespace QuantumProofs.Noise

open Matrix
open scoped ComplexOrder

variable {d : ℕ}

/-- The `d`-dimensional global depolarizing channel action `E_p(ρ) = (1−p)ρ + (p/d)(Tr ρ)·1`. -/
noncomputable def globalDepol (d : ℕ) (p : ℝ) (ρ : Matrix (Fin d) (Fin d) ℂ) :
    Matrix (Fin d) (Fin d) ℂ :=
  ((1 - p : ℝ) : ℂ) • ρ + ((p / d : ℝ) : ℂ) • (ρ.trace • (1 : Matrix (Fin d) (Fin d) ℂ))

/-- Trace of the pure density `|ψ⟩⟨ψ|` (general dimension) equals `⟨ψ|ψ⟩`. -/
theorem trace_pure_gen (ψ : Fin d → ℂ) : (vecMulVec ψ (star ψ)).trace = star ψ ⬝ᵥ ψ := by
  simp only [Matrix.trace, Matrix.diag, Matrix.vecMulVec_apply, dotProduct]
  exact Finset.sum_congr rfl fun i _ => mul_comm _ _

/-- `|ψ⟩⟨ψ| *ᵥ ψ = ⟨ψ|ψ⟩ • ψ`. -/
theorem vecMulVec_pure_mulVec (ψ : Fin d → ℂ) :
    vecMulVec ψ (star ψ) *ᵥ ψ = (star ψ ⬝ᵥ ψ) • ψ := by
  ext i
  simp only [Matrix.mulVec, dotProduct, Matrix.vecMulVec_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- Born expectation of the pure density `|ψ⟩⟨ψ|` in `ψ` (general dimension) equals `⟨ψ|ψ⟩²`. -/
theorem pure_exp_gen (ψ : Fin d → ℂ) :
    star ψ ⬝ᵥ (vecMulVec ψ (star ψ) *ᵥ ψ) = (star ψ ⬝ᵥ ψ) * (star ψ ⬝ᵥ ψ) := by
  rw [vecMulVec_pure_mulVec, dotProduct_smul, smul_eq_mul]

/-- **Trace preservation.**  The global depolarizing channel preserves trace. -/
theorem globalDepol_trace (hd : 0 < d) (p : ℝ) (ρ : Matrix (Fin d) (Fin d) ℂ) :
    (globalDepol d p ρ).trace = ρ.trace := by
  have hdc : (d : ℂ) ≠ 0 := by exact_mod_cast hd.ne'
  simp only [globalDepol, Matrix.trace_add, Matrix.trace_smul, Matrix.trace_one,
    Fintype.card_fin, smul_eq_mul]
  push_cast
  field_simp
  ring

/-- **Closed form of the depth-`G` global depolarizing channel.**
`E_p^[G](ρ) = (1−p)^G·ρ + ((1−(1−p)^G)/d)·(Tr ρ)·1`. -/
theorem globalDepol_iterate (hd : 0 < d) (p : ℝ) (G : ℕ) (ρ : Matrix (Fin d) (Fin d) ℂ) :
    (globalDepol d p)^[G] ρ
      = (((1 - p) ^ G : ℝ) : ℂ) • ρ
        + (((1 - (1 - p) ^ G) / d : ℝ) : ℂ) • (ρ.trace • (1 : Matrix (Fin d) (Fin d) ℂ)) := by
  have hdc : (d : ℂ) ≠ 0 := by exact_mod_cast hd.ne'
  induction G with
  | zero => simp
  | succ G ih =>
    rw [Function.iterate_succ', Function.comp_apply, ih]
    simp only [globalDepol, Matrix.trace_add, Matrix.trace_smul, Matrix.trace_one,
      Fintype.card_fin, smul_smul, smul_eq_mul, mul_one]
    match_scalars <;> push_cast [pow_succ] <;> field_simp <;> ring

/-- **Global depolarizing fidelity (the entangled-output floor).**  For ANY normalized state `ψ`
on the `d`-dimensional register (product or entangled), the fidelity after `G` global depolarizing
channels is `(1−p)^G + (1−(1−p)^G)/d`. -/
theorem globalDepol_fidelity (hd : 0 < d) (p : ℝ) (G : ℕ) (ψ : Fin d → ℂ)
    (hψ : star ψ ⬝ᵥ ψ = 1) :
    star ψ ⬝ᵥ ((globalDepol d p)^[G] (vecMulVec ψ (star ψ)) *ᵥ ψ)
      = (((1 - p) ^ G + (1 - (1 - p) ^ G) / d : ℝ) : ℂ) := by
  rw [globalDepol_iterate hd, add_mulVec, smul_mulVec, smul_mulVec, dotProduct_add,
    dotProduct_smul, dotProduct_smul, pure_exp_gen, smul_mulVec, one_mulVec, dotProduct_smul,
    trace_pure_gen, hψ]
  simp only [smul_eq_mul, mul_one]
  push_cast; ring

/-- **Circulant register (3-qubit, `d = 8`) global-depolarizing fidelity floor.**  The circulant
`CyclicShift` output lives in an 8-dimensional register and may be entangled; under `G` per-gate
global depolarizing channels its fidelity is `(1−p)^G + (1−(1−p)^G)/8` — the honest entangled-output
model.  (The product-state width floor `(1−p/2)^3` is the special case for a product output.) -/
theorem globalDepol_circulant_fidelity (p : ℝ) (G : ℕ) (ψ : Fin 8 → ℂ)
    (hψ : star ψ ⬝ᵥ ψ = 1) :
    star ψ ⬝ᵥ ((globalDepol 8 p)^[G] (vecMulVec ψ (star ψ)) *ᵥ ψ)
      = (((1 - p) ^ G + (1 - (1 - p) ^ G) / 8 : ℝ) : ℂ) :=
  globalDepol_fidelity (by norm_num) p G ψ hψ

end QuantumProofs.Noise
