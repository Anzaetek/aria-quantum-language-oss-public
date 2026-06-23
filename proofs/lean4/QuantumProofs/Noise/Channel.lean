/-
  D2 — quantum support library in Lean (LEAN_ERROR_PLAN.md LE3 / LEAN_EXPORT_PLAN.md Track D2).

  Foundation layer: density operators and CPTP maps in Kraus form over `Matrix (Fin d) (Fin d) ℂ`.

  A density operator is positive-semidefinite with unit trace (`IsDensity`). A quantum channel
  is given by a finite Kraus decomposition `{Kᵢ}` satisfying the completeness relation
  `∑ᵢ Kᵢᴴ Kᵢ = 1`, acting as `ρ ↦ ∑ᵢ Kᵢ ρ Kᵢᴴ` (`KrausMap`/`KrausMap.apply`).

  The headline well-definedness result is `KrausMap.apply_isDensity`: every Kraus map sends
  density operators to density operators (trace preservation + positivity), i.e. it is CPTP.
  This is the trust-boundary object the noise models (depolarizing, amplitude/phase damping)
  are instances of in the LE3 fidelity-vs-noise proofs.

  Sorry-free; reuses mathlib's `Matrix.PosSemidef` / `Matrix.trace` API.
-/
import Mathlib

namespace QuantumProofs.Noise

open Matrix
open scoped ComplexOrder

variable {d : ℕ}

/-- A density operator: positive semidefinite with unit trace.  (`PosSemidef` already
bundles Hermitian-ness, so this is the full standard definition.) -/
def IsDensity (ρ : Matrix (Fin d) (Fin d) ℂ) : Prop :=
  ρ.PosSemidef ∧ ρ.trace = 1

/-- A finite Kraus decomposition of a quantum channel: `k` operators satisfying the
completeness relation `∑ᵢ Kᵢᴴ Kᵢ = 1`.  This is exactly the CPTP (trace-preserving,
completely positive) data in operator-sum form. -/
structure KrausMap (d : ℕ) where
  /-- number of Kraus operators -/
  k : ℕ
  /-- the Kraus operators `K₀ … K_{k-1}` -/
  ops : Fin k → Matrix (Fin d) (Fin d) ℂ
  /-- completeness / trace-preservation relation `∑ Kᵢᴴ Kᵢ = 1` -/
  complete : ∑ i, (ops i)ᴴ * ops i = 1

/-- Action of a Kraus map on an operator: `ρ ↦ ∑ᵢ Kᵢ ρ Kᵢᴴ`. -/
noncomputable def KrausMap.apply (E : KrausMap d) (ρ : Matrix (Fin d) (Fin d) ℂ) :
    Matrix (Fin d) (Fin d) ℂ :=
  ∑ i, E.ops i * ρ * (E.ops i)ᴴ

/-- **Trace preservation.** A Kraus map preserves trace: `Tr(E ρ) = Tr ρ`.
Uses the cyclic property `Tr(Kᵢ ρ Kᵢᴴ) = Tr(Kᵢᴴ Kᵢ ρ)` and completeness. -/
theorem KrausMap.apply_trace (E : KrausMap d) (ρ : Matrix (Fin d) (Fin d) ℂ) :
    (E.apply ρ).trace = ρ.trace := by
  unfold KrausMap.apply
  rw [trace_sum]
  have h : ∀ i, (E.ops i * ρ * (E.ops i)ᴴ).trace = ((E.ops i)ᴴ * E.ops i * ρ).trace :=
    fun i => trace_mul_cycle (E.ops i) ρ (E.ops i)ᴴ
  simp_rw [h, ← trace_sum, ← Finset.sum_mul, E.complete, one_mul]

/-- **Positivity preservation.** A Kraus map sends positive-semidefinite operators to
positive-semidefinite operators (this is the "completely positive" half, at the
single-system level): each summand `Kᵢ ρ Kᵢᴴ` is PSD, and PSD is closed under sums. -/
theorem KrausMap.apply_posSemidef (E : KrausMap d) {ρ : Matrix (Fin d) (Fin d) ℂ}
    (hρ : ρ.PosSemidef) : (E.apply ρ).PosSemidef := by
  unfold KrausMap.apply
  exact posSemidef_sum _ (fun i _ => hρ.mul_mul_conjTranspose_same (E.ops i))

/-- **CPTP well-definedness.** A Kraus map sends density operators to density operators.
This is the core object the LE3 noise models specialise. -/
theorem KrausMap.apply_isDensity (E : KrausMap d) {ρ : Matrix (Fin d) (Fin d) ℂ}
    (hρ : IsDensity ρ) : IsDensity (E.apply ρ) :=
  ⟨E.apply_posSemidef hρ.1, by rw [E.apply_trace]; exact hρ.2⟩

end QuantumProofs.Noise
