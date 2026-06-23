/-
  D2 — single-qubit AMPLITUDE-DAMPING channel (LEAN_ERROR_PLAN.md LE3 / Track D2).

  Models energy relaxation (T₁): the excited population `|1⟩` decays into `|0⟩` at rate γ.
  Kraus operators

      K₀ = [[1, 0],[0, √(1−γ)]]        K₁ = [[0, √γ],[0, 0]]

  with `K₀ᴴK₀ + K₁ᴴK₁ = 1` (completeness), so `amplitudeDamping γ : KrausMap 2` is a genuine
  CPTP map.  The closed form is

      E_γ(ρ) = [[a + γd,  √(1−γ)·b],[√(1−γ)·c,  (1−γ)·d]]   for ρ = [[a,b],[c,d]],

  from which the headline **relaxation law** for the Pauli-Z expectation follows:

      ⟨Z⟩(E_γ ρ) = ⟨Z⟩(ρ) + 2γ·d          (`amplitudeDamping_expZ`)

  where `d = ρ 1 1` is the excited-state population.  This is exactly the channel-level
  statement that the OSS `noise` emulator certifies numerically as law (B),
  `⟨Z⟩ = ⟨Z⟩₀ + 2γ·sin²(θ/2)`, for the RY(θ)|0⟩ state (there `d = sin²(θ/2)`).
  The excited state fully relaxes: `⟨1|E_γ(|1⟩⟨1|)|1⟩ = 1 − γ` (`amplitudeDamping_relax`).

  Sorry-free; 2×2 entrywise computation reusing `Channel.lean`/`Pauli.lean`.
-/
import Mathlib
import QuantumProofs.Noise.Channel
import QuantumProofs.Noise.Pauli

namespace QuantumProofs.Noise

open Matrix
open scoped ComplexOrder

set_option linter.unusedSimpArgs false

/-- Kraus operator `K₀ = diag(1, √(1−γ))` of the amplitude-damping channel. -/
noncomputable def adK0 (γ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1, 0; 0, ((Real.sqrt (1 - γ) : ℝ) : ℂ)]

/-- Kraus operator `K₁ = [[0, √γ],[0, 0]]` (the `|1⟩→|0⟩` jump) of the amplitude-damping
channel. -/
noncomputable def adK1 (γ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![0, ((Real.sqrt γ : ℝ) : ℂ); 0, 0]

/-- `K₀` is real-diagonal, so `K₀ᴴ = K₀`. -/
theorem adK0_conjT (γ : ℝ) : (adK0 γ)ᴴ = adK0 γ := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [adK0, Matrix.conjTranspose_apply]

/-- `K₁ᴴ = [[0, 0],[√γ, 0]]`. -/
theorem adK1_conjT (γ : ℝ) : (adK1 γ)ᴴ = !![0, 0; ((Real.sqrt γ : ℝ) : ℂ), 0] := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [adK1, Matrix.conjTranspose_apply]

/-- Real-cast square facts (in `push_cast` normal form): `√(1−γ)·√(1−γ) = 1−γ` and
`√γ·√γ = γ` over `ℂ`. -/
theorem ad_sqrt_sq (γ : ℝ) (hγ : 0 ≤ γ) (hγ1 : γ ≤ 1) :
    ((Real.sqrt (1 - γ) : ℝ) : ℂ) * ((Real.sqrt (1 - γ) : ℝ) : ℂ) = 1 - (γ : ℂ)
    ∧ ((Real.sqrt γ : ℝ) : ℂ) * ((Real.sqrt γ : ℝ) : ℂ) = (γ : ℂ) :=
  ⟨by rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by linarith)]; push_cast; ring,
   by rw [← Complex.ofReal_mul, Real.mul_self_sqrt hγ]⟩

set_option maxHeartbeats 800000 in
/-- The single-qubit **amplitude-damping channel** at rate `γ ∈ [0,1]`, as a CPTP Kraus map. -/
noncomputable def amplitudeDamping (γ : ℝ) (hγ : 0 ≤ γ) (hγ1 : γ ≤ 1) : KrausMap 2 where
  k := 2
  ops := ![adK0 γ, adK1 γ]
  complete := by
    obtain ⟨h1, h2⟩ := ad_sqrt_sq γ hγ hγ1
    rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      adK0_conjT, adK1_conjT]
    simp only [adK0, adK1, Matrix.mul_fin_two, Matrix.one_fin_two]
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp only [Fin.mk_zero, Fin.mk_one, Matrix.add_apply, Matrix.of_apply, Matrix.cons_val',
        Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.empty_val',
        Matrix.cons_val_fin_one] <;>
      push_cast <;>
      (first
        | linear_combination h1 + h2
        | linear_combination h1
        | linear_combination h2
        | ring)

set_option maxHeartbeats 800000 in
/-- **Closed form of the amplitude-damping channel.**  For `ρ = [[a,b],[c,d]]`,
`E_γ(ρ) = [[a + γd, √(1−γ)·b],[√(1−γ)·c, (1−γ)·d]]`. -/
theorem amplitudeDamping_apply (γ : ℝ) (hγ : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (ρ : Matrix (Fin 2) (Fin 2) ℂ) :
    (amplitudeDamping γ hγ hγ1).apply ρ
      = !![ρ 0 0 + ((γ : ℝ) : ℂ) * ρ 1 1, ((Real.sqrt (1 - γ) : ℝ) : ℂ) * ρ 0 1;
           ((Real.sqrt (1 - γ) : ℝ) : ℂ) * ρ 1 0, ((1 - γ : ℝ) : ℂ) * ρ 1 1] := by
  obtain ⟨h1, h2⟩ := ad_sqrt_sq γ hγ hγ1
  have hsum : (amplitudeDamping γ hγ hγ1).apply ρ
      = adK0 γ * ρ * (adK0 γ)ᴴ + adK1 γ * ρ * (adK1 γ)ᴴ := by
    rw [show (amplitudeDamping γ hγ hγ1).apply ρ
          = ∑ i : Fin 2, ![adK0 γ, adK1 γ] i * ρ * (![adK0 γ, adK1 γ] i)ᴴ from rfl,
      Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
  rw [hsum, adK0_conjT, adK1_conjT, eta_fin_two ρ]
  simp only [adK0, adK1, Matrix.mul_fin_two]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp only [Fin.mk_zero, Fin.mk_one, Matrix.add_apply, Matrix.of_apply, Matrix.cons_val',
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.empty_val',
      Matrix.cons_val_fin_one] <;>
    push_cast <;>
    (first
      | linear_combination (ρ 1 1) * h2
      | linear_combination (ρ 1 1) * h1
      | ring)

/-- **Amplitude-damping relaxation law (the LE3 headline, OSS cross-check).**
`⟨Z⟩(E_γ ρ) = ⟨Z⟩(ρ) + 2γ·d`, where `d = ρ₁₁` is the excited-state population.  This is the
channel-level identity that the OSS `noise` emulator certifies numerically as law (B),
`⟨Z⟩ = ⟨Z⟩₀ + 2γ·sin²(θ/2)` (there `d = sin²(θ/2)` for the `RY(θ)|0⟩` input). -/
theorem amplitudeDamping_expZ (γ : ℝ) (hγ : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (ρ : Matrix (Fin 2) (Fin 2) ℂ) :
    expZ ((amplitudeDamping γ hγ hγ1).apply ρ) = expZ ρ + ((2 * γ : ℝ) : ℂ) * ρ 1 1 := by
  rw [expZ_eq, expZ_eq, amplitudeDamping_apply]
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.empty_val', Matrix.cons_val_fin_one]
  push_cast; ring

/-- **Full relaxation of the excited state.**  Starting from `|1⟩⟨1|`, the surviving
excited-state population after amplitude damping is `⟨1|E_γ(|1⟩⟨1|)|1⟩ = 1 − γ`. -/
theorem amplitudeDamping_relax (γ : ℝ) (hγ : 0 ≤ γ) (hγ1 : γ ≤ 1) :
    ((amplitudeDamping γ hγ hγ1).apply !![0, 0; 0, 1]) 1 1 = ((1 - γ : ℝ) : ℂ) := by
  rw [amplitudeDamping_apply]
  simp only [Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.empty_val', Matrix.cons_val_fin_one, mul_one]
