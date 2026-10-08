/-
Stabilizer expectation values in a concrete operator model.

`verification/Verification/Backend/StabilizerExpectation.lean` states three
targets over free axioms (`StabState`, `Pauli`, `Anticommutes`, `InGroupPlus`,
`InGroupMinus`, `expectation`, `reducesToIdentity`) with nothing relating them,
so none of its targets can be proved — or refuted — where it stands. This file
gives those symbols a concrete meaning and proves the targets there, one at a
time: T1, the trichotomy, and T2, that `0` is reserved for anticommutation, are
proved below.
The abstract file's targets stay targets.

The trichotomy is also consistent and, on Hermitian Paulis, exhaustive:
`not_plus_and_minus` (`+P` and `−P` are never both group members) and
`exhaustive` (on a full-rank state, a Hermitian `P` anticommutes with a
generator or is `±` a product of generators). `exhaustive` is **not**
unconditional in `P`, and cannot be here: a record carries its own phase, and
`i · ZZ` on the Bell state is in none of the three branches
(`exhaustive_needs_hermitian`), because every signed group element is
Hermitian (`inGroup_hermitian`). `zeroState n` (`|0^n⟩`, stabilized by every
`Z_i`) is full rank at every `n` (`zeroState_independent`), with
`⟨Z_i⟩ = 1` and `⟨X_i⟩ = 0` (`zeroState_Z`, `zeroState_X`), so the premises are
not an `n = 2` accident. T3 — that the pivoted `𝔽₂` elimination algorithm is
complete — is still not proved here.

The model:

* a Pauli operator is `i^k · X^x · Z^z` with `k : ZMod 4` and bit vectors
  `x z : Fin n → ZMod 2`; it acts on amplitudes over basis bitstrings by
  `(P ψ)(y) = i^k (-1)^{z·(y+x)} ψ(y+x)`;
* multiplication is defined on the records, combinatorially:
  `(k, x, z)(k', x', z') = (k + k' + 2 z·x', x + x', z + z')`, and
  `act_mul` proves it is the operator product;
* a stabilizer state is `n` Hermitian, pairwise commuting generators together
  with a unit vector each of them fixes;
* `Anticommutes s p` is the symplectic condition `ω(gᵢ, p) = 1` for some
  generator, and `anticommutes_iff_operators` proves it is operator
  anticommutation;
* `InGroupPlus s p` is **combinatorial** membership: `p` equals, as a record,
  the product of some list of generators; `InGroupMinus` is the same with the
  sign `-1`. It is not the eigen-condition `P ψ = ± ψ`, which would make the
  trichotomy a restatement of its own hypothesis;
* `expectation s p = ⟨ψ, P ψ⟩`.
-/
import Mathlib

set_option autoImplicit false

namespace QuantumProofs.StabilizerModel

open Finset

variable {n : ℕ}

/-- Basis bitstrings, and the `x`/`z` coordinates of a Pauli, over `𝔽₂`. -/
abbrev Bits (n : ℕ) := Fin n → ZMod 2

/-- The Pauli operator `i^phase · X^x · Z^z` on `n` qubits. -/
@[ext] structure Pauli (n : ℕ) where
  phase : ZMod 4
  x : Bits n
  z : Bits n
  deriving DecidableEq

/-- `a ↦ 2a` from `𝔽₂` into `ℤ/4`: the phase that a sign `(-1)^a` contributes. -/
def two (a : ZMod 2) : ZMod 4 := 2 * (a.val : ZMod 4)

lemma two_add (a b : ZMod 2) : two (a + b) = two a + two b := by revert a b; decide

lemma neg_two (a : ZMod 2) : -two a = two a := by revert a; decide

lemma two_add_self (a : ZMod 2) : two a + two a = 0 := by revert a; decide

lemma zmod2_add_self (a : ZMod 2) : a + a = 0 := by revert a; decide

lemma bits_add_self (x : Bits n) : x + x = 0 := by
  funext i; exact zmod2_add_self (x i)

/-! ### The Pauli group, on records -/

/-- The identity. -/
def one : Pauli n := ⟨0, 0, 0⟩

/-- `(i^k X^x Z^z)(i^k' X^x' Z^z') = i^{k+k'} (-1)^{z·x'} X^{x+x'} Z^{z+z'}`. -/
def mul (p q : Pauli n) : Pauli n :=
  ⟨p.phase + q.phase + two (p.z ⬝ᵥ q.x), p.x + q.x, p.z + q.z⟩

/-- Multiply by the scalar `i^k`. -/
def scale (k : ZMod 4) (p : Pauli n) : Pauli n := ⟨k + p.phase, p.x, p.z⟩

/-- `-P`. -/
def neg (p : Pauli n) : Pauli n := scale 2 p

/-- The adjoint: `(i^k X^x Z^z)† = i^{-k} (-1)^{z·x} X^x Z^z`. -/
def adj (p : Pauli n) : Pauli n := ⟨-p.phase + two (p.z ⬝ᵥ p.x), p.x, p.z⟩

/-- The symplectic form. `ω(p, q) = 0` iff `p` and `q` commute. -/
def omega (p q : Pauli n) : ZMod 2 := p.x ⬝ᵥ q.z + p.z ⬝ᵥ q.x

/-- The ordered product of a list. -/
def prodList : List (Pauli n) → Pauli n
  | [] => one
  | p :: l => mul p (prodList l)

lemma omega_comm (p q : Pauli n) : omega p q = omega q p := by
  simp only [omega, dotProduct_comm p.x, dotProduct_comm p.z]; ring

/-- Swapping two factors costs the sign `(-1)^{ω(p,q)}`. -/
lemma mul_swap (p q : Pauli n) : mul p q = scale (two (omega p q)) (mul q p) := by
  ext
  · simp only [mul, scale, omega, two_add]
    rw [dotProduct_comm p.x q.z]
    linear_combination (-1 : ZMod 4) * two_add_self (q.z ⬝ᵥ p.x)
  · simp [mul, scale, add_comm]
  · simp [mul, scale, add_comm]

/-! ### Semantics: the operators -/

/-- Amplitudes over the `2^n` basis bitstrings. -/
abbrev Vec (n : ℕ) := Bits n → ℂ

/-- `(-1)^a`. -/
def sgn (a : ZMod 2) : ℂ := if a = 0 then 1 else -1

/-- `i^k`. -/
noncomputable def ipow (k : ZMod 4) : ℂ := Complex.I ^ k.val

/-- The operator `i^k X^x Z^z`: `(P ψ)(y) = i^k (-1)^{z·(y+x)} ψ(y+x)`. -/
noncomputable def act (p : Pauli n) (ψ : Vec n) : Vec n :=
  fun y => ipow p.phase * sgn (p.z ⬝ᵥ (y + p.x)) * ψ (y + p.x)

/-- `⟨φ, χ⟩`, antilinear in the first argument. -/
noncomputable def inner (φ χ : Vec n) : ℂ := ∑ y, star (φ y) * χ y

lemma zmod2_cases (a : ZMod 2) : a = 0 ∨ a = 1 := by revert a; decide

lemma zmod2_one_add_one : (1 : ZMod 2) + 1 = 0 := by decide

@[simp] lemma sgn_zero : sgn 0 = 1 := by simp [sgn]

@[simp] lemma sgn_one : sgn 1 = -1 := by simp [sgn]

lemma sgn_add (a b : ZMod 2) : sgn (a + b) = sgn a * sgn b := by
  rcases zmod2_cases a with rfl | rfl <;> rcases zmod2_cases b with rfl | rfl <;>
    simp [zmod2_one_add_one]

lemma sgn_mul_self (a : ZMod 2) : sgn a * sgn a = 1 := by
  rcases zmod2_cases a with rfl | rfl <;> simp

lemma star_sgn (a : ZMod 2) : star (sgn a) = sgn a := by
  rcases zmod2_cases a with rfl | rfl <;> simp

lemma sgn_ne_zero (a : ZMod 2) : sgn a ≠ 0 := by
  rcases zmod2_cases a with rfl | rfl <;> simp

lemma I_pow_mod_four (m : ℕ) : Complex.I ^ (m % 4) = Complex.I ^ m := by
  conv_rhs => rw [← Nat.mod_add_div m 4, pow_add, pow_mul, Complex.I_pow_four, one_pow, mul_one]

@[simp] lemma ipow_zero : ipow 0 = 1 := by simp [ipow]

lemma ipow_add (a b : ZMod 4) : ipow (a + b) = ipow a * ipow b := by
  simp only [ipow, ZMod.val_add, I_pow_mod_four, pow_add]

lemma ipow_two_eq : ipow 2 = -1 := by
  rw [ipow, show (2 : ZMod 4).val = 2 from rfl, Complex.I_sq]

lemma ipow_two (a : ZMod 2) : ipow (two a) = sgn a := by
  rcases zmod2_cases a with rfl | rfl
  · simp [two]
  · rw [show two 1 = 2 from rfl, ipow_two_eq, sgn_one]

lemma ipow_ne_zero (k : ZMod 4) : ipow k ≠ 0 := pow_ne_zero _ Complex.I_ne_zero

lemma star_ipow (k : ZMod 4) : star (ipow k) = ipow (-k) := by
  have h1 : star (ipow k) * ipow k = 1 := by
    rw [ipow, star_pow, ← mul_pow, Complex.star_def, Complex.conj_I]
    simp
  have h2 : ipow (-k) * ipow k = 1 := by rw [← ipow_add, neg_add_cancel, ipow_zero]
  exact mul_right_cancel₀ (ipow_ne_zero k) (h1.trans h2.symm)

/-! ### The record product is the operator product -/

theorem act_mul (p q : Pauli n) (ψ : Vec n) : act (mul p q) ψ = act p (act q ψ) := by
  funext y
  simp only [act, mul, ipow_add, ipow_two]
  have h : (p.z + q.z) ⬝ᵥ (y + (p.x + q.x))
      = p.z ⬝ᵥ (y + p.x) + p.z ⬝ᵥ q.x + q.z ⬝ᵥ (y + p.x + q.x) := by
    simp only [add_dotProduct, dotProduct_add]; ring
  rw [h, sgn_add, sgn_add, ← add_assoc y]
  linear_combination (ipow p.phase * ipow q.phase * sgn (p.z ⬝ᵥ (y + p.x))
    * sgn (q.z ⬝ᵥ (y + p.x + q.x)) * ψ (y + p.x + q.x)) * sgn_mul_self (p.z ⬝ᵥ q.x)

theorem act_scale (k : ZMod 4) (p : Pauli n) (ψ : Vec n) :
    act (scale k p) ψ = fun y => ipow k * act p ψ y := by
  funext y; simp only [act, scale, ipow_add]; ring

theorem act_one (ψ : Vec n) : act one ψ = ψ := by
  funext y; simp [act, one]

theorem act_neg (p : Pauli n) (ψ : Vec n) : act (neg p) ψ = fun y => -act p ψ y := by
  rw [neg, act_scale, ipow_two_eq]; funext y; ring

/-- **The commutation bridge, operator side.** `P Q = (-1)^{ω(P,Q)} Q P` as
    operators. This is where a sign error in the record product would show. -/
theorem act_swap (p q : Pauli n) (ψ : Vec n) :
    act p (act q ψ) = fun y => sgn (omega p q) * act q (act p ψ) y := by
  rw [← act_mul, mul_swap, act_scale, ipow_two, act_mul]

lemma act_ne_zero (p : Pauli n) {ψ : Vec n} (h : ∀ y, ψ y ≠ 0) (y : Bits n) :
    act p ψ y ≠ 0 := by
  simp only [act]
  exact mul_ne_zero (mul_ne_zero (ipow_ne_zero _) (sgn_ne_zero _)) (h _)

/-! ### The inner product and the adjoint -/

lemma inner_smul_right (φ χ : Vec n) (c : ℂ) :
    inner φ (fun y => c * χ y) = c * inner φ χ := by
  simp only [inner, mul_sum]; refine sum_congr rfl fun y _ => ?_; ring

lemma inner_neg_right (φ χ : Vec n) : inner φ (fun y => -χ y) = -inner φ χ := by
  simp only [inner, ← sum_neg_distrib]; refine sum_congr rfl fun y _ => ?_; ring

/-- `⟨P φ, χ⟩ = ⟨φ, P† χ⟩`. -/
theorem inner_act_left (p : Pauli n) (φ χ : Vec n) :
    inner (act p φ) χ = inner φ (act (adj p) χ) := by
  unfold inner
  rw [← Equiv.sum_comp (Equiv.addRight p.x)]
  refine sum_congr rfl fun y _ => ?_
  simp only [Equiv.coe_addRight, act, adj, add_assoc, bits_add_self, add_zero, star_mul',
    star_ipow, star_sgn, ipow_add, ipow_two, dotProduct_add, sgn_add]
  linear_combination (-(ipow (-p.phase) * sgn (p.z ⬝ᵥ y) * star (φ y) * χ (y + p.x)))
    * sgn_mul_self (p.z ⬝ᵥ p.x)

/-! ### Stabilizer states, and the three symbols the targets talk about -/

/-- An `n`-qubit stabilizer state: `n` Hermitian, pairwise commuting generators
    and a unit vector each of them fixes. Independence of the generators is
    **not** part of the structure; the targets that need it take it as a
    hypothesis, so a witness can show what fails without it. -/
structure StabState (n : ℕ) where
  gens : Fin n → Pauli n
  ψ : Vec n
  hermitian : ∀ i, adj (gens i) = gens i
  commuting : ∀ i j, omega (gens i) (gens j) = 0
  normalized : inner ψ ψ = 1
  stabilizes : ∀ i, act (gens i) ψ = ψ

/-- `p` anticommutes with some generator: the symplectic condition. -/
def Anticommutes (s : StabState n) (p : Pauli n) : Prop := ∃ i, omega (s.gens i) p = 1

/-- **Combinatorial** membership with sign `+`: `p` is, as a record, the product
    of a list of generators. -/
def InGroupPlus (s : StabState n) (p : Pauli n) : Prop :=
  ∃ l : List (Fin n), p = prodList (l.map s.gens)

/-- Combinatorial membership with sign `−`. -/
def InGroupMinus (s : StabState n) (p : Pauli n) : Prop :=
  ∃ l : List (Fin n), p = neg (prodList (l.map s.gens))

/-- `⟨ψ, P ψ⟩`. -/
noncomputable def expectation (s : StabState n) (p : Pauli n) : ℂ := inner s.ψ (act p s.ψ)

/-- **The commutation bridge.** The symplectic `Anticommutes` is operator
    anticommutation with some generator, on every vector. -/
theorem anticommutes_iff_operators (s : StabState n) (p : Pauli n) :
    Anticommutes s p ↔
      ∃ i, ∀ ψ : Vec n, act (s.gens i) (act p ψ) = fun y => -act p (act (s.gens i) ψ) y := by
  constructor
  · rintro ⟨i, hi⟩
    refine ⟨i, fun ψ => ?_⟩
    rw [act_swap, hi, sgn_one]; funext y; ring
  · rintro ⟨i, hi⟩
    refine ⟨i, ?_⟩
    rcases zmod2_cases (omega (s.gens i) p) with h0 | h1
    · exfalso
      have hc := hi (fun _ => 1)
      rw [act_swap, h0, sgn_zero] at hc
      have hy := congrFun hc 0
      simp only [one_mul] at hy
      have hne : act p (act (s.gens i) (fun _ => (1 : ℂ))) 0 ≠ 0 :=
        act_ne_zero _ (act_ne_zero _ (fun _ => one_ne_zero)) 0
      exact hne (by linear_combination hy / 2)
    · exact h1

lemma act_prodList_fixes (s : StabState n) (l : List (Fin n)) :
    act (prodList (l.map s.gens)) s.ψ = s.ψ := by
  induction l with
  | nil => exact act_one _
  | cons i l ih => rw [List.map_cons, prodList, act_mul, ih, s.stabilizes]

/-- **T1 — the trichotomy.** -/
theorem expectation_trichotomy (s : StabState n) (p : Pauli n) :
    (Anticommutes s p → expectation s p = 0)
    ∧ (InGroupPlus s p → expectation s p = 1)
    ∧ (InGroupMinus s p → expectation s p = -1) := by
  refine ⟨?_, ?_, ?_⟩
  · rintro ⟨i, hi⟩
    -- ⟨ψ, Pψ⟩ = ⟨gψ, Pψ⟩ = ⟨ψ, g P ψ⟩ = -⟨ψ, P g ψ⟩ = -⟨ψ, Pψ⟩
    have h : expectation s p = -expectation s p := by
      calc expectation s p = inner (act (s.gens i) s.ψ) (act p s.ψ) := by
              rw [s.stabilizes]; rfl
        _ = inner s.ψ (act (s.gens i) (act p s.ψ)) := by rw [inner_act_left, s.hermitian]
        _ = -expectation s p := by
              rw [act_swap, hi, sgn_one]
              simp only [neg_one_mul]
              rw [inner_neg_right, s.stabilizes]; rfl
    linear_combination h / 2
  · rintro ⟨l, rfl⟩
    rw [expectation, act_prodList_fixes, s.normalized]
  · rintro ⟨l, rfl⟩
    rw [expectation, act_neg, inner_neg_right, act_prodList_fixes, s.normalized]

/-! ### Witnesses: the Bell state `(|00⟩ + |11⟩)/√2`, stabilized by `XX` and `ZZ` -/

/-- `X ⊗ X`. -/
def pXX : Pauli 2 := ⟨0, ![1, 1], 0⟩
/-- `Z ⊗ Z`. -/
def pZZ : Pauli 2 := ⟨0, 0, ![1, 1]⟩
/-- `Y ⊗ Y = (iXZ) ⊗ (iXZ)`. -/
def pYY : Pauli 2 := ⟨2, ![1, 1], ![1, 1]⟩
/-- `Z ⊗ I`. -/
def pZI : Pauli 2 := ⟨0, 0, ![1, 0]⟩

lemma sum_zmod2 (f : ZMod 2 → ℂ) : ∑ a, f a = f 0 + f 1 := by
  rw [show (univ : Finset (ZMod 2)) = {0, 1} from rfl, sum_pair (by decide)]

lemma sum_bits2 (f : Bits 2 → ℂ) : ∑ y, f y = f ![0, 0] + f ![0, 1] + f ![1, 0] + f ![1, 1] := by
  rw [← (piFinTwoEquiv fun _ => ZMod 2).symm.sum_comp, Fintype.sum_prod_type]
  simp only [sum_zmod2]
  simp [piFinTwoEquiv, add_assoc]
  rfl

attribute [local simp] zmod2_one_add_one

/-- `1/√2`. -/
noncomputable def bellAmp : ℂ := ((Real.sqrt 2 : ℝ) : ℂ)⁻¹

lemma star_bellAmp : star bellAmp = bellAmp := by
  simp [bellAmp, Complex.conj_ofReal]

lemma bellAmp_mul_self : bellAmp * bellAmp = 1 / 2 := by
  have h : (Real.sqrt 2)⁻¹ * (Real.sqrt 2)⁻¹ = (1 / 2 : ℝ) := by
    rw [← mul_inv, Real.mul_self_sqrt (by norm_num)]; norm_num
  rw [bellAmp, ← Complex.ofReal_inv, ← Complex.ofReal_mul, h]; push_cast; ring

/-- `(|00⟩ + |11⟩)/√2`. -/
noncomputable def bellVec : Vec 2 := fun y => if y 0 = y 1 then bellAmp else 0

/-- The Bell state's generators, `XX` and `ZZ`. -/
def bellGens : Fin 2 → Pauli 2 := ![pXX, pZZ]

/-- The Bell state. -/
noncomputable def bell : StabState 2 where
  gens := bellGens
  ψ := bellVec
  hermitian := by decide
  commuting := by decide
  normalized := by
    rw [inner, sum_bits2]
    simp [bellVec, star_bellAmp, bellAmp_mul_self]
    norm_num
  stabilizes := by
    intro i; fin_cases i
    · funext y; simp [act, bellGens, pXX, bellVec]
    · funext y
      simp [act, bellGens, pZZ, bellVec, dotProduct, Fin.sum_univ_two]
      split_ifs with h
      · rw [h, zmod2_add_self, sgn_zero, one_mul]
      · rfl

/-- **Witness for T1.** Every premise of the trichotomy is inhabited on the Bell
    state: `Z ⊗ I` anticommutes with `XX`; `XX` is in the group with sign `+`;
    `YY = −(XX)(ZZ)` is in it with sign `−`, so the `−` case needs the record
    product's phase, not just its bits. -/
theorem witness_T1 :
    Anticommutes bell pZI ∧ InGroupPlus bell pXX ∧ InGroupMinus bell pYY :=
  ⟨⟨0, show omega (bellGens 0) pZI = 1 by decide⟩,
    ⟨[0], show pXX = prodList ([0].map bellGens) by rw [Pauli.ext_iff]; decide⟩,
    ⟨[0, 1], show pYY = neg (prodList ([0, 1].map bellGens)) by rw [Pauli.ext_iff]; decide⟩⟩

/-- The model's number agrees with the physics, computed directly from the
    amplitudes and not through T1: `⟨Φ⁺| Y ⊗ Y |Φ⁺⟩ = −1`. -/
theorem bell_YY_direct : expectation bell pYY = -1 := by
  rw [expectation, inner, sum_bits2]
  simp [bell, act, pYY, bellVec, dotProduct, Fin.sum_univ_two, ipow_two_eq, star_bellAmp,
    bellAmp_mul_self]
  norm_num

/-! ## T2 — `0` is reserved for anticommutation

The argument is linear algebra over `𝔽₂`. A Pauli's bits `(x, z)` live in
`𝔽₂^{2n}`, where the symplectic form `ω` is a nondegenerate bilinear form
(`symp_nondegenerate`). `n` independent, pairwise commuting generators span an
isotropic subspace `W` of dimension `n`. Mathlib's dimension theorem for
orthogonal complements (`LinearMap.BilinForm.finrank_orthogonal`, which needs
nondegeneracy) gives `dim W^⊥ = 2n − n = n`, so `W = W^⊥`. A Pauli commuting
with every generator has its bits in `W^⊥ = W`, so it is a phase times a product
of generators, and its expectation is that phase, which is never `0`. -/

/-- The bits of a Pauli, phase dropped. -/
def vec (p : Pauli n) : Bits n × Bits n := (p.x, p.z)

/-- The symplectic form on `𝔽₂^{2n}`, as a Mathlib bilinear form. -/
def symp : LinearMap.BilinForm (ZMod 2) (Bits n × Bits n) :=
  LinearMap.mk₂ (ZMod 2) (fun v w => v.1 ⬝ᵥ w.2 + v.2 ⬝ᵥ w.1)
    (fun _ _ _ => by simp only [Prod.fst_add, Prod.snd_add, add_dotProduct]; ring)
    (fun _ _ _ => by simp only [Prod.smul_fst, Prod.smul_snd, smul_dotProduct, smul_eq_mul]; ring)
    (fun _ _ _ => by simp only [Prod.fst_add, Prod.snd_add, dotProduct_add]; ring)
    (fun _ _ _ => by simp only [Prod.smul_fst, Prod.smul_snd, dotProduct_smul, smul_eq_mul]; ring)

@[simp] lemma symp_apply (v w : Bits n × Bits n) : symp v w = v.1 ⬝ᵥ w.2 + v.2 ⬝ᵥ w.1 := rfl

lemma omega_eq_symp (p q : Pauli n) : omega p q = symp (vec p) (vec q) := rfl

lemma symp_comm (v w : Bits n × Bits n) : symp v w = symp w v := by
  simp only [symp_apply, dotProduct_comm v.1, dotProduct_comm v.2]; ring

lemma symp_separating (v : Bits n × Bits n) (h : ∀ w, symp v w = 0) : v = 0 := by
  refine Prod.ext (funext fun i => ?_) (funext fun i => ?_)
  · simpa using h (0, Pi.single i 1)
  · simpa using h (Pi.single i 1, 0)

/-- `ω` is nondegenerate. -/
lemma symp_nondegenerate : (symp (n := n)).Nondegenerate :=
  ⟨symp_separating, fun w h => symp_separating w fun v => by rw [symp_comm]; exact h v⟩

/-- The generators' bits are linearly independent over `𝔽₂`: the state is full
    rank. -/
def Independent (s : StabState n) : Prop := LinearIndependent (ZMod 2) fun i => vec (s.gens i)

/-- The span of the generators' bits. -/
noncomputable def genSpan (s : StabState n) : Submodule (ZMod 2) (Bits n × Bits n) :=
  Submodule.span (ZMod 2) (Set.range fun i => vec (s.gens i))

lemma mem_orthogonal_of_gens (s : StabState n) (v : Bits n × Bits n)
    (h : ∀ i, symp (vec (s.gens i)) v = 0) : v ∈ symp.orthogonal (genSpan s) := by
  have hle : genSpan s ≤ LinearMap.ker (LinearMap.flip symp v) := by
    rw [genSpan, Submodule.span_le]
    rintro _ ⟨i, rfl⟩
    simpa using h i
  intro w hw
  simpa [LinearMap.BilinForm.IsOrtho] using hle hw

/-- The generators span an isotropic subspace. -/
lemma genSpan_isotropic (s : StabState n) : genSpan s ≤ symp.orthogonal (genSpan s) := by
  rw [genSpan, Submodule.span_le]
  rintro _ ⟨i, rfl⟩
  exact mem_orthogonal_of_gens s _ fun j => s.commuting j i

/-- **Lagrangian.** With `n` independent generators the isotropic span is its own
    symplectic complement. -/
lemma genSpan_orthogonal (s : StabState n) (hs : Independent s) :
    symp.orthogonal (genSpan s) = genSpan s := by
  have hW : Module.finrank (ZMod 2) (genSpan s) = n := by
    rw [genSpan, finrank_span_eq_card hs, Fintype.card_fin]
  have hV : Module.finrank (ZMod 2) (Bits n × Bits n) = n + n := by
    rw [Module.finrank_prod, Module.finrank_fintype_fun_eq_card, Fintype.card_fin]
  have hO : Module.finrank (ZMod 2) (symp.orthogonal (genSpan s)) = n := by
    rw [LinearMap.BilinForm.finrank_orthogonal symp_nondegenerate, hV, hW]; omega
  exact (Submodule.eq_of_le_of_finrank_eq (genSpan_isotropic s) (by rw [hW, hO])).symm

/-! ### Phase bookkeeping: bits of products, and the subset product -/

lemma vec_prodList (L : List (Pauli n)) : vec (prodList L) = (L.map vec).sum := by
  induction L with
  | nil => rfl
  | cons p L ih => rw [prodList, List.map_cons, List.sum_cons, ← ih]; rfl

lemma sum_map_filter {ι M : Type*} [AddCommMonoid M] (L : List ι) (P : ι → Prop) [DecidablePred P]
    (f : ι → M) : ((L.filter P).map f).sum = (L.map fun i => if P i then f i else 0).sum := by
  induction L with
  | nil => rfl
  | cons a L ih => by_cases h : P a <;> simp [h, ih]

/-- The product of the generators selected by `c`, in index order. -/
def subsetList (c : Fin n → ZMod 2) : List (Fin n) := (List.finRange n).filter fun i => c i = 1

lemma vec_subset_prod (s : StabState n) (c : Fin n → ZMod 2) :
    vec (prodList ((subsetList c).map s.gens)) = ∑ i, c i • vec (s.gens i) := by
  rw [vec_prodList, List.map_map, subsetList, sum_map_filter, Fin.sum_univ_def]
  congr 1
  refine List.map_congr_left fun i _ => ?_
  rcases zmod2_cases (c i) with h | h <;> simp [h]

/-- A Pauli commuting with every generator of a full-rank state is a phase times
    a product of generators. -/
theorem commuting_eq_scale_prod (s : StabState n) (hs : Independent s) (p : Pauli n)
    (hc : ∀ i, omega (s.gens i) p = 0) :
    ∃ (m : ZMod 4) (l : List (Fin n)), p = scale m (prodList (l.map s.gens)) := by
  have hmem : vec p ∈ genSpan s := by
    rw [← genSpan_orthogonal s hs]; exact mem_orthogonal_of_gens s _ hc
  obtain ⟨c, hcv⟩ := (Submodule.mem_span_range_iff_exists_fun (ZMod 2)).mp hmem
  have hv := (vec_subset_prod s c).trans hcv
  refine ⟨p.phase - (prodList ((subsetList c).map s.gens)).phase, subsetList c, ?_⟩
  exact Pauli.ext (by simp [scale]) (congrArg Prod.fst hv).symm (congrArg Prod.snd hv).symm

/-- **T2 — `0` is reserved for anticommutation.** On a full-rank stabilizer
    state, `⟨P⟩ = 0` only if `P` anticommutes with some generator. This is the
    property the greedy reduction violated: it returned `0` for group members. -/
theorem zero_only_when_anticommuting (s : StabState n) (hs : Independent s) (p : Pauli n)
    (h : expectation s p = 0) : Anticommutes s p := by
  by_contra hna
  have hc : ∀ i, omega (s.gens i) p = 0 := fun i =>
    (zmod2_cases _).resolve_right fun h1 => hna ⟨i, h1⟩
  obtain ⟨m, l, rfl⟩ := commuting_eq_scale_prod s hs p hc
  rw [expectation, act_scale, inner_smul_right, act_prodList_fixes, s.normalized, mul_one] at h
  exact ipow_ne_zero m h

/-! ### The centralizer is the group up to sign -/

lemma scale_zero (p : Pauli n) : scale 0 p = p := by ext <;> simp [scale]

lemma omega_one_left (q : Pauli n) : omega one q = 0 := by simp [omega, one]

lemma omega_mul_left (a b c : Pauli n) : omega (mul a b) c = omega a c + omega b c := by
  simp only [omega, mul, add_dotProduct]; ring

lemma adj_mul (a b : Pauli n) : adj (mul a b) = mul (adj b) (adj a) := by
  ext
  · simp only [adj, mul, add_dotProduct, dotProduct_add, two_add]
    rw [dotProduct_comm b.z a.x]
    ring
  · simp [adj, mul, add_comm]
  · simp [adj, mul, add_comm]

lemma omega_prodList_gens (s : StabState n) (l : List (Fin n)) (i : Fin n) :
    omega (prodList (l.map s.gens)) (s.gens i) = 0 := by
  induction l with
  | nil => exact omega_one_left _
  | cons j l ih => rw [List.map_cons, prodList, omega_mul_left, ih, s.commuting, add_zero]

/-- A product of commuting Hermitian generators is Hermitian. -/
lemma adj_prodList_gens (s : StabState n) (l : List (Fin n)) :
    adj (prodList (l.map s.gens)) = prodList (l.map s.gens) := by
  induction l with
  | nil => ext <;> simp [adj, prodList, one, two]
  | cons j l ih =>
    rw [List.map_cons, prodList, adj_mul, ih, s.hermitian, mul_swap, omega_prodList_gens]
    exact scale_zero _

lemma adj_scale (m : ZMod 4) (p : Pauli n) : adj (scale m p) = scale (-m) (adj p) := by
  ext <;> simp [adj, scale]; ring

/-- **The centralizer is the group up to sign.** On a full-rank state, a
    Hermitian Pauli that anticommutes with no generator is `±` a product of
    generators — never `±i` times one. With T1 this gives `⟨P⟩ = ±1`. -/
theorem commuting_hermitian_in_group (s : StabState n) (hs : Independent s) (p : Pauli n)
    (hp : adj p = p) (hna : ¬ Anticommutes s p) : InGroupPlus s p ∨ InGroupMinus s p := by
  have hc : ∀ i, omega (s.gens i) p = 0 := fun i =>
    (zmod2_cases _).resolve_right fun h1 => hna ⟨i, h1⟩
  obtain ⟨m, l, rfl⟩ := commuting_eq_scale_prod s hs p hc
  rw [adj_scale, adj_prodList_gens] at hp
  have h2 : m + m = 0 := by
    have := congrArg Pauli.phase hp
    simp only [scale] at this
    linear_combination -this
  have hm : m = 0 ∨ m = 2 := (by decide : ∀ m : ZMod 4, m + m = 0 → m = 0 ∨ m = 2) m h2
  rcases hm with rfl | rfl
  · exact Or.inl ⟨l, scale_zero _⟩
  · exact Or.inr ⟨l, rfl⟩

/-! ### Witnesses for T2 -/

lemma bell_independent : Independent bell := by
  have hv : (fun i => vec (bell.gens i)) = ![vec pXX, vec pZZ] := by
    funext i; fin_cases i <;> rfl
  rw [Independent, hv, LinearIndependent.pair_iff]
  intro a b h
  have h1 := congrArg (fun v : Bits 2 × Bits 2 => v.1 0) h
  have h2 := congrArg (fun v : Bits 2 × Bits 2 => v.2 0) h
  simp [vec, pXX, pZZ] at h1 h2
  exact ⟨h1, h2⟩

/-- **Witness for T2.** Its premises are inhabited: the Bell state is full rank,
    and `⟨Z ⊗ I⟩ = 0` there. -/
theorem witness_T2 : Independent bell ∧ expectation bell pZI = 0 :=
  ⟨bell_independent, (expectation_trichotomy bell pZI).1 witness_T1.1⟩

/-- `I ⊗ Z`. -/
def pIZ : Pauli 2 := ⟨0, 0, ![0, 1]⟩

/-- `|0⟩ ⊗ |+⟩`. -/
noncomputable def zeroPlusVec : Vec 2 := fun y => if y 0 = 0 then bellAmp else 0

/-- Dependent generators `Z ⊗ I`, `Z ⊗ I`. -/
def zzGens : Fin 2 → Pauli 2 := ![pZI, pZI]

/-- `|0⟩ ⊗ |+⟩` with the generator `Z ⊗ I` listed twice. It is a legal
    `StabState` in every field except that its generators are not
    independent. -/
noncomputable def zeroPlusDep : StabState 2 where
  gens := zzGens
  ψ := zeroPlusVec
  hermitian := by decide
  commuting := by decide
  normalized := by
    rw [inner, sum_bits2]
    simp [zeroPlusVec, star_bellAmp, bellAmp_mul_self]
    norm_num
  stabilizes := by
    intro i
    funext y
    fin_cases i <;>
    · simp [act, zzGens, pZI, zeroPlusVec, dotProduct, Fin.sum_univ_two]
      split_ifs with h
      · rw [h, sgn_zero, one_mul]
      · rfl

/-- **The full-rank hypothesis does work.** On `|0⟩ ⊗ |+⟩` with dependent
    generators, `I ⊗ Z` has expectation `0` and anticommutes with no
    generator: T2's conclusion fails once independence is dropped. -/
theorem full_rank_needed :
    ¬ Independent zeroPlusDep ∧ expectation zeroPlusDep pIZ = 0 ∧ ¬ Anticommutes zeroPlusDep pIZ := by
  refine ⟨fun h => ?_, ?_, ?_⟩
  · exact absurd (h.injective (show vec (zeroPlusDep.gens 0) = vec (zeroPlusDep.gens 1) from rfl))
      (by decide)
  · rw [expectation, inner, sum_bits2]
    simp [zeroPlusDep, act, pIZ, zeroPlusVec, dotProduct, Fin.sum_univ_two, star_bellAmp]
  · rintro ⟨i, hi⟩
    revert hi
    fin_cases i <;> decide


/-! ## Exhaustiveness and consistency of the trichotomy

In this record model a Pauli carries its own phase `i^k`, so the unconditional
statement "every `p` anticommutes, or `+p`, or `−p` is in the group" is
**false**: `i · ZZ` commutes with both Bell generators but is anti-Hermitian,
while every signed product of the (commuting, Hermitian) generators is
Hermitian (`inGroup_hermitian`). `exhaustive_needs_hermitian` is that
counterexample. The true statement, `exhaustive`, takes `adj p = p`, which is
exactly what is missing: off the anticommuting case, `p` is `±` a group
element iff it is Hermitian (`exhaustive` one way, `inGroup_hermitian` the
other). A model whose Paulis are phase-free bit pairs, each read as its
Hermitian operator, has this hypothesis built in. -/

/-- Every signed group element is Hermitian. -/
theorem inGroup_hermitian (s : StabState n) (p : Pauli n)
    (h : InGroupPlus s p ∨ InGroupMinus s p) : adj p = p := by
  rcases h with ⟨l, rfl⟩ | ⟨l, rfl⟩
  · exact adj_prodList_gens s l
  · rw [neg, adj_scale, adj_prodList_gens]; rfl

/-- **The trichotomy is exhaustive on Hermitian Paulis.** On a full-rank
    stabilizer state, a Hermitian Pauli anticommutes with a generator, or is
    `+` a product of generators, or `−` one. The Hermitian hypothesis cannot be
    dropped (`exhaustive_needs_hermitian`). `adj` fixes `x` and `z`, so
    `adj p = p` is exactly a condition on the phase: `2·k = 2·(z ⬝ᵥ x)` in
    `ZMod 4`, i.e. `k ≡ z·x (mod 2)` — `X`, `Z` and `Y = i·XZ` qualify, `i·X`
    and `XZ = −i·Y` do not. -/
theorem exhaustive (s : StabState n) (hs : Independent s) (p : Pauli n) (hp : adj p = p) :
    Anticommutes s p ∨ InGroupPlus s p ∨ InGroupMinus s p := by
  by_cases hna : Anticommutes s p
  · exact Or.inl hna
  · exact Or.inr (commuting_hermitian_in_group s hs p hp hna)

/-- **`i · ZZ` escapes all three branches** on the full-rank Bell state: it
    anticommutes with no generator and is neither `+` nor `−` a product of
    generators. So `exhaustive` without `adj p = p` is false in this model. -/
theorem exhaustive_needs_hermitian :
    Independent bell ∧
      ¬ (Anticommutes bell (scale 1 pZZ) ∨ InGroupPlus bell (scale 1 pZZ)
          ∨ InGroupMinus bell (scale 1 pZZ)) := by
  refine ⟨bell_independent, ?_⟩
  rintro (⟨i, hi⟩ | h)
  · revert hi; fin_cases i <;> decide
  · have := inGroup_hermitian bell _ h
    revert this; decide

/-- **The `+` and `−` branches never both fire.** Otherwise `P ψ = ψ` and
    `P ψ = −ψ`, so `1 = ⟨ψ, ψ⟩ = −⟨ψ, ψ⟩ = −1`. Proved from the definitions,
    not through T1, so a sign dropped from `InGroupMinus` fails here. -/
theorem not_plus_and_minus (s : StabState n) (p : Pauli n)
    (h1 : InGroupPlus s p) (h2 : InGroupMinus s p) : False := by
  obtain ⟨l, hl⟩ := h1
  obtain ⟨l', hl'⟩ := h2
  have e1 : act p s.ψ = s.ψ := by rw [hl, act_prodList_fixes]
  have e2 : act p s.ψ = fun y => -s.ψ y := by rw [hl', act_neg, act_prodList_fixes]
  have h := congrArg (inner s.ψ) (e1.symm.trans e2)
  rw [inner_neg_right, s.normalized] at h
  norm_num at h

/-! ### An inhabitant at every `n`: `|0^n⟩`, stabilized by every `Z_i` -/

/-- `Z_i`. -/
def zGen (n : ℕ) (i : Fin n) : Pauli n := ⟨0, 0, Pi.single i 1⟩

/-- `X_i`. -/
def xGen (n : ℕ) (i : Fin n) : Pauli n := ⟨0, Pi.single i 1, 0⟩

/-- `|0^n⟩`. -/
noncomputable def zeroVec (n : ℕ) : Vec n := fun y => if y = 0 then 1 else 0

/-- `|0^n⟩` with generators `Z_0, …, Z_{n-1}`. -/
noncomputable def zeroState (n : ℕ) : StabState n where
  gens := zGen n
  ψ := zeroVec n
  hermitian := fun i => Pauli.ext (by simp only [adj, zGen, dotProduct_zero]; decide) rfl rfl
  commuting := fun i j => by simp [omega, zGen]
  normalized := by
    rw [inner, Finset.sum_eq_single (0 : Bits n)]
    · simp [zeroVec]
    · intro y _ hy; simp [zeroVec, hy]
    · simp
  stabilizes := fun i => by
    funext y
    by_cases hy : y = 0
    · subst hy; simp [act, zGen, zeroVec]
    · simp [act, zGen, zeroVec, hy]

/-- `|0^n⟩`'s generators are independent: it is full rank at every `n`. -/
theorem zeroState_independent (n : ℕ) : Independent (zeroState n) := by
  have hv : (fun i => vec ((zeroState n).gens i))
      = LinearMap.inr (ZMod 2) (Bits n) (Bits n) ∘ (Pi.basisFun (ZMod 2) (Fin n)) := by
    funext i; simp [zeroState, vec, zGen]
  rw [Independent, hv]
  exact (Pi.basisFun (ZMod 2) (Fin n)).linearIndependent.map'
    _ (LinearMap.ker_eq_bot.mpr LinearMap.inr_injective)

/-- **`⟨0^n| Z_i |0^n⟩ = 1`**, through T1's `+` branch: `Z_i` is the one-element
    product `[i]`. -/
theorem zeroState_Z (n : ℕ) (i : Fin n) : expectation (zeroState n) (zGen n i) = 1 :=
  (expectation_trichotomy _ _).2.1
    ⟨[i], by ext <;> simp [zeroState, prodList, mul, one, zGen, two]⟩

/-- **`⟨0^n| X_i |0^n⟩ = 0`**, through T1's anticommuting branch: `X_i`
    anticommutes with `Z_i`. -/
theorem zeroState_X (n : ℕ) (i : Fin n) : expectation (zeroState n) (xGen n i) = 0 :=
  (expectation_trichotomy _ _).1 ⟨i, by simp [zeroState, omega, zGen, xGen]⟩

end QuantumProofs.StabilizerModel
