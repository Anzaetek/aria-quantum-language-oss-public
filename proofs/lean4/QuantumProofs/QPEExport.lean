/-
QPE export bridge (LEAN_EXPORT Track B — QPE).

The Aria/Rust exporter lowers the QFT to an explicit gate list `qftLowered n`
(see `QFTExport.lean`), proved sorry-free to denote to the exact DFT matrix
(`qftLowered_correct : denote (qftLowered n) = dft_matrix n`). Quantum Phase
Estimation runs that same circuit **backwards** on the counting register: the
inverse-QFT stage is the adjoint of the forward QFT, i.e. the operator
`(denote (qftLowered n))ᴴ`.

This file closes the `@prove "qpe_recovers_phase"` obligation against the
*exporter-emitted* QFT circuit (not just the library `qft_circuit`): for exact
phases `θ = m/2^n`, conjugating the controlled-U-loaded counting register
(`after_controlled_u n θ`, the model input that abstracts the state-prep +
controlled-U loading — exactly as `qpe_exact`/`qpe_exact_denote` take it) by the
inverse of the exporter's QFT and measuring yields `m` with probability 1.

`qpe_lowered_exact` is the QPE analogue of `QPE.qpe_exact_denote`, but stated
over the exporter-lowered `qftLowered n` rather than the library `qft_circuit`,
bridged by `qftLowered_correct`. Everything is sorry-free; `#print axioms
qpe_lowered_exact` shows only the three standard axioms.

NOTE (faithfulness boundary): this Level-A bridge expresses the inverse-QFT as
the *matrix adjoint* of the emitted forward-QFT circuit. The explicit
inverse-QFT **gate list** (`qftInverseLowered`, with conjugated controlled
phases) and the matching corrected `qpe.aria` are the Level-B follow-on.
-/
import QuantumProofs.QPE
import QuantumProofs.QFTExport

namespace QuantumProofs.QPEExport

open Matrix CircuitSemantics QuantumProofs.QFT QuantumProofs.QPE QuantumProofs.QFTExport

/-- **QPE exactness against the exporter-lowered QFT (Level-A bridge).** For
    exact phases `θ = m/2^n`, applying the inverse of the *exporter-emitted* QFT
    circuit (`(denote (qftLowered n))ᴴ`, i.e. running the lowered QFT backwards)
    to the controlled-U-loaded counting register `after_controlled_u n θ` and
    measuring yields `m` with probability 1.

    Closed by rewriting the lowered QFT to the exact DFT (`qftLowered_correct`),
    identifying `(dft_matrix n)ᴴ *ᵥ ·` with `apply_inverse_dft`
    (`apply_inverse_dft_eq_mulVec`), and invoking the proved `qpe_exact`. -/
theorem qpe_lowered_exact (n : ℕ) (m : Fin (2 ^ n)) (θ : ℝ)
    (hθ : θ = ↑m.val / ↑(2 ^ n)) :
    measure_prob ((denote (qftLowered n))ᴴ *ᵥ after_controlled_u n θ) m = 1 := by
  rw [qftLowered_correct, ← apply_inverse_dft_eq_mulVec]
  exact qpe_exact n m θ hθ

/-- **QPE approximation against the exporter-lowered QFT.** For any phase
    `θ ∈ [0,1)`, some `n`-bit outcome is measured with probability `≥ 4/π²`
    through the inverse of the exporter-emitted QFT. The lowered analogue of
    `QPE.qpe_approximate_denote`. -/
theorem qpe_lowered_approximate (n : ℕ) (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 1) :
    ∃ m : Fin (2 ^ n),
      measure_prob ((denote (qftLowered n))ᴴ *ᵥ after_controlled_u n θ) m
        ≥ 4 / Real.pi ^ 2 := by
  obtain ⟨m, hm⟩ := qpe_approximate n θ hθ
  refine ⟨m, ?_⟩
  rwa [qftLowered_correct, ← apply_inverse_dft_eq_mulVec]

end QuantumProofs.QPEExport
