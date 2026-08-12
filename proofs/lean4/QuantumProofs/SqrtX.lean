/-
  `√X` (`SX`) and `√X†` (`SXdg`) — matrix identities and the Clifford
  conjugation rules the stabilizer backend implements.

  ## Why this file exists

  `GateKind::Sx` / `Sxdg` were added as first-class variants rather than
  aliases for `U3(π/2, −π/2, π/2)`, and the justification has two halves that
  are checked in different places:

  * **The global phase** (`sx = e^{iπ/4}·U3(π/2,−π/2,π/2)`) is a numeric fact,
    pinned in Rust against Qiskit.
  * **The Clifford action** — `X → +X`, `Y → +Z`, `Z → −Y` for `sx`, and
    `X → +X`, `Y → −Z`, `Z → +Y` for `sxdg` — is what
    `StabilizerTableau::sx` / `sxdg` encode as a bit-level update
    (`x' = x XOR z`, `z' = z`, `sign ^= z AND NOT x` / `z AND x`).

  That second half is where a mistake would be *silent*: a wrong sign
  condition still produces a valid-looking stabilizer state, and a stabilizer
  backend agrees with itself perfectly whatever rule it uses. The rule was
  derived by hand and cross-checked against Stim, but "derived by hand and
  checked against one other implementation" is exactly the standard of evidence
  this project keeps finding insufficient. So it is proved here from the
  matrices.

  `sqrtX_conj_Z` is the theorem that would catch the realistic bug: it fixes
  the SIGN on the `Z → −Y` image, which is the single bit the tableau rule's
  `sign ^= z AND NOT x` clause exists to produce, and the one that differs
  between `sx` and `sxdg`.
-/

import QuantumProofs.Gates
import Mathlib.Data.Matrix.Basic
import Mathlib.Tactic

namespace QuantumProofs.SqrtX

open Complex Matrix QuantumProofs.Gates

/-- `√X = ½·[[1+i, 1−i], [1−i, 1+i]]` — Qiskit's `SXGate`, Stim's `SQRT_X`. -/
noncomputable def SX : Matrix (Fin 2) (Fin 2) ℂ :=
  !![(1 + I)/2, (1 - I)/2; (1 - I)/2, (1 + I)/2]

/-- `√X† = ½·[[1−i, 1+i], [1+i, 1−i]]` — Qiskit's `SXdgGate`, Stim's
    `SQRT_X_DAG`. -/
noncomputable def SXdg : Matrix (Fin 2) (Fin 2) ℂ :=
  !![(1 - I)/2, (1 + I)/2; (1 + I)/2, (1 - I)/2]

/-- **`√X` really is a square root of X.** This is the identity that makes the
    gate Clifford, and therefore the whole reason it is not lowered to `U3`:
    `PauliBackend` rejects `U3` outright, so aliasing would refuse an
    all-Clifford circuit. -/
theorem sqrtX_sq : SX * SX = X := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, X, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `√X · √X† = I`, i.e. `sxdg` is the inverse the adjoint pass uses. -/
theorem sqrtX_mul_sqrtXdg : SX * SXdg = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply] <;>
    ring_nf <;> simp [Complex.ext_iff] <;> ring

/-- `√X†` is the conjugate transpose of `√X`. -/
theorem sqrtXdg_eq_conjTranspose : SXdg = SXᴴ := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Matrix.conjTranspose_apply, Complex.ext_iff] <;> ring

/-! ### The Clifford conjugation rules

`StabilizerTableau::sx` implements `P ↦ SX · P · SX†` as a bit update. These
three theorems are the specification it must satisfy. -/

/-- `sx: X → +X`. The fixed axis — a rotation about X leaves X alone. -/
theorem sqrtX_conj_X : SX * X * SXdg = X := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, X, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `sx: Y → +Z`. -/
theorem sqrtX_conj_Y : SX * Y * SXdg = Z := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Y, Z, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `sx: Z → −Y`. **The sign here is the whole point of this file.**

    In the tableau this is the `sign ^= z AND NOT x` clause: the input `Z` has
    `(x,z) = (0,1)`, so `NOT x` holds and the sign flips. `sxdg` uses
    `z AND x` instead, which does *not* fire on `Z` — hence `Z → +Y` there.
    Getting the two conditions the wrong way round produces a perfectly
    well-formed stabilizer state that is silently wrong. -/
theorem sqrtX_conj_Z : SX * Z * SXdg = -Y := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Y, Z, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `sxdg: Z → +Y` — the opposite sign to [`sqrtX_conj_Z`], which is exactly
    why the two tableau rules cannot share a sign condition. -/
theorem sqrtXdg_conj_Z : SXdg * Z * SX = Y := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Y, Z, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `sxdg: Y → −Z`, likewise opposite to [`sqrtX_conj_Y`]. -/
theorem sqrtXdg_conj_Y : SXdg * Y * SX = -Z := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Y, Z, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

end QuantumProofs.SqrtX
