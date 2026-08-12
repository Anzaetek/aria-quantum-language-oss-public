/-
  `√X` (`SX`) and `√X†` (`SXdg`): matrix identities, and the Clifford rules
  the two stabilizer-style backends implement.

  ## THE TWO BACKENDS CONJUGATE IN OPPOSITE DIRECTIONS

  This is the point of the file, and the first version of it got the
  presentation wrong: it proved both directions but named the theorems after
  the *gates* (`sqrtX_conj_Z`, `sqrtXdg_conj_Z`) rather than after the *code
  they specify*. A reader mapping a theorem onto an implementation then has to
  redo the direction argument in their head, which is exactly where a sign
  goes missing.

  * `omega-backend-pauli` — `StabilizerTableau` rows are the **stabilizers of
    the state**. Applying `U` to `|ψ⟩` sends a stabilizer `S ↦ U S U†`. So the
    tableau uses the **FORWARD (Schrödinger/stabilizer) picture**.

  * `omega-backend-pauliprop` — `SingleImg` stores images of the **observable**
    under Heisenberg propagation, `O ↦ U† O U` (see the comment on
    `SingleImg`: "G† X G and G† Z G"). So it uses the **ADJOINT (Heisenberg)
    picture**.

  Both are correct, and they are not interchangeable. Measured, on `sx|0⟩`:

  ```
    stabilizer picture:  |0⟩ is stabilized by +Z;  SX·Z·SX† = −Y
                         ⇒ sx|0⟩ is stabilized by −Y, so ⟨Y⟩ = −1
    observable picture:  SX†·Y·SX = −Z
                         ⇒ ⟨0| SX† Y SX |0⟩ = ⟨0|−Z|0⟩ = −1
  ```

  Same answer, opposite conjugation. Swapping the two swaps `sx` with `sxdg`,
  which is a pure sign change that yields a perfectly valid Pauli sum or
  stabilizer state and is invisible to anything checking self-consistency.

  Every theorem below is therefore named for the implementation it specifies,
  and the `#### Rust` blocks quote the code. `sorry`-free; mutation-tested by
  flipping a sign in the statement, which makes Lean report `unsolved goals`.
-/

import QuantumProofs.Gates
import Mathlib.Data.Matrix.Basic
import Mathlib.Tactic

namespace QuantumProofs.SqrtX

open Complex Matrix QuantumProofs.Gates

/-- `√X = ½·[[1+i, 1−i], [1−i, 1+i]]` — Qiskit's `SXGate`, Stim's `SQRT_X`,
    ppvm's `sqrt_x`. Not `U3(π/2,−π/2,π/2)`, which differs by a global
    `e^{iπ/4}`. -/
noncomputable def SX : Matrix (Fin 2) (Fin 2) ℂ :=
  !![(1 + I)/2, (1 - I)/2; (1 - I)/2, (1 + I)/2]

/-- `√X† = ½·[[1−i, 1+i], [1+i, 1−i]]` — `SXdgGate` / `SQRT_X_DAG` /
    `sqrt_x_dag`. -/
noncomputable def SXdg : Matrix (Fin 2) (Fin 2) ℂ :=
  !![(1 - I)/2, (1 + I)/2; (1 + I)/2, (1 - I)/2]

/-! ## Gate-level identities (picture-independent) -/

/-- **`√X` is a square root of X**, hence Clifford. This is why
    `GateKind::Sx` exists as a first-class variant: `PauliBackend` rejects
    `U3` categorically as non-Clifford, so aliasing would make the stabilizer
    backend refuse an all-Clifford circuit. -/
theorem sqrtX_sq : SX * SX = X := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, X, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `√X · √X† = I` — justifies the adjoint pass substituting one for the
    other (`GateKind::Sx => …sxdg()` in every backend's adjoint dispatch). -/
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

/-! ## FORWARD picture — specifies `omega-backend-pauli`

`StabilizerTableau::sx` / `sxdg` update the state's stabilizers, `S ↦ U S U†`.

#### Rust
```rust
pub fn sx(&mut self, q: usize) {
    for row in &mut self.rows {
        row.sign ^= row.z[q] && !row.x[q];   // fires on Z (x=0,z=1)
        row.x[q] ^= row.z[q];
    }
}
```
Row encoding: `I=(0,0)`, `X=(1,0)`, `Y=(1,1)`, `Z=(0,1)`. -/

/-- `stabilizer sx: X ↦ +X`. The fixed axis; `x' = x XOR z` leaves `(1,0)`
    alone and the sign clause does not fire. -/
theorem stabilizer_sx_X : SX * X * SXdg = X := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, X, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `stabilizer sx: Y ↦ +Z`. Input `(1,1)` → `x' = 1 XOR 1 = 0`, giving
    `(0,1) = Z`; the sign clause `z ∧ ¬x` is false, so `+`. -/
theorem stabilizer_sx_Y : SX * Y * SXdg = Z := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Y, Z, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `stabilizer sx: Z ↦ −Y`. **The sign this whole file exists for.**

    Input `(0,1)` → `x' = 0 XOR 1 = 1`, giving `(1,1) = Y`; the clause
    `z ∧ ¬x` IS true here, so the sign flips. `sxdg` uses `z ∧ x`, which is
    false on `Z` — hence `+Y` there. Getting the two conditions the wrong way
    round produces a well-formed stabilizer state that is silently wrong. -/
theorem stabilizer_sx_Z : SX * Z * SXdg = -Y := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Y, Z, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `stabilizer sxdg: Z ↦ +Y` — the OPPOSITE sign to [`stabilizer_sx_Z`],
    which is why the two tableau rules cannot share a sign condition. -/
theorem stabilizer_sxdg_Z : SXdg * Z * SX = Y := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Y, Z, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `stabilizer sxdg: Y ↦ −Z`. -/
theorem stabilizer_sxdg_Y : SXdg * Y * SX = -Z := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Y, Z, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-! ## ADJOINT picture — specifies `omega-backend-pauliprop`

`SingleImg` stores `G† X G` and `G† Z G`, each as `factor · raw(x, z)` where
`raw(x,z) = X^x Z^z`, so `Y = i·raw(1,1)`.

#### Rust
```rust
const CLIFF_SX: SingleImg = SingleImg {
    ax: true, az: false, fx: ONE,               // sx† X sx = +1 · raw(1,0)
    bx: true, bz: true,  fz: Complex64::new(0.0,  1.0),  // sx† Z sx = +i·raw(1,1) = +Y
};
const CLIFF_SXDG: SingleImg = SingleImg {
    ax: true, az: false, fx: ONE,
    bx: true, bz: true,  fz: Complex64::new(0.0, -1.0),  // = −Y
};
```

**These are NOT the forward theorems above.** `CLIFF_SX`'s `Z` image is `+Y`
while `stabilizer_sx_Z` gives `−Y`. Both are right, in their own picture. -/

/-- `pauliprop CLIFF_SX: sx† X sx = +X` — matches `ax=true, az=false, fx=1`.

    Note this coincides with [`stabilizer_sx_X`]: `X` is the fixed axis in
    both pictures, so it is the one image that CANNOT distinguish them. Any
    test relying on the X image alone is blind to a direction error. -/
theorem pauliprop_cliffSx_X : SXdg * X * SX = X := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, X, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `pauliprop CLIFF_SX: sx† Z sx = +Y` — matches `fz = +i`, since
    `Y = i·raw(1,1)`.

    **Contrast [`stabilizer_sx_Z`], which gives `−Y`.** These two theorems
    being visibly adjacent, with opposite signs and names that say which code
    each governs, is the entire reason this file was restructured. -/
theorem pauliprop_cliffSx_Z : SXdg * Z * SX = Y := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Y, Z, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `pauliprop CLIFF_SXDG: sxdg† Z sxdg = −Y` — matches `fz = −i`. -/
theorem pauliprop_cliffSxdg_Z : SX * Z * SXdg = -Y := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [SX, SXdg, Y, Z, Matrix.mul_apply, Fin.sum_univ_two] <;> ring_nf <;>
    simp [Complex.ext_iff] <;> ring

/-- `SXᴴ = SXdg`, stated in the direction the adjoint picture needs. -/
theorem sqrtX_conjTranspose : SXᴴ = SXdg := (sqrtXdg_eq_conjTranspose).symm

/-- **The two pictures are related by exchanging the gate with its inverse.**

    Proved rather than asserted in a comment: conjugating `P` in the ADJOINT
    picture by `sx` is the same as conjugating it in the FORWARD picture by
    `sxdg`. That is exactly why misreading the direction swaps the two gates —
    and why `pauliprop_cliffSx_Z` (`+Y`) and `stabilizer_sx_Z` (`−Y`) disagree
    without either being wrong. -/
theorem adjoint_picture_is_forward_picture_of_the_inverse
    (P : Matrix (Fin 2) (Fin 2) ℂ) :
    SXdg * P * SX = SXdg * P * (SXdg)ᴴ := by
  have h : (SXdg)ᴴ = SX := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [SX, SXdg, Matrix.conjTranspose_apply, Complex.ext_iff] <;> ring
  rw [h]

end QuantumProofs.SqrtX
