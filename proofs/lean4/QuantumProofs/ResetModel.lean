/-
Reset semantics in a concrete statevector model.

`verification/Verification/Backend/Reset.lean` states four targets about Reset
over free axioms (`State`, `p0`, `project`, `applyX`, `reducedPurity`,
`EqUpToPhase`, `fold`) with nothing relating them, so none of its targets can
be proved — or refuted — where it stands. This file gives every one of those
symbols a concrete meaning and proves the four targets there:

* a state on `n` qubits is a unit vector of amplitudes over basis bitstrings,
  `Vec n := EuclideanSpace ℂ (Fin n → Bool)`;
* `project s q b` keeps the `q = b` amplitudes and renormalises (the zero vector
  when there is nothing to renormalise, which is what the spec's `Possible`
  hypothesis exists to rule out);
* `applyX` flips bit `q`; `reset` is measure-then-`X`-on-1;
* `reducedPurity s q = tr ρ_q²` for the reduced density matrix of qubit `q`;
* `EqUpToPhase a b := ∃ c, ‖c‖ = 1 ∧ a = c • b`;
* `fold` is the superseded coherent fold `new0 = old0 + old1`, renormalised,
  which leaves the zero register when the sum vanishes.

So this is not a proof of the abstract file's targets — those stay targets —
but a model in which the specification is satisfiable and each target holds,
with an instantiation witness for every premise.

Writing `lo := part false q s` (the `q = 0` amplitudes) and
`hi := applyX (part true q s) q` (the `q = 1` amplitudes moved onto `q = 0`),
the reduced density matrix of `q` is `[[‖lo‖², ⟪hi,lo⟫], [⟪lo,hi⟫, ‖hi‖²]]`, so
`tr ρ² = ‖lo‖⁴ + ‖hi‖⁴ + 2‖⟪lo,hi⟫‖²` for a unit `s`; and since
`‖lo‖² + ‖hi‖² = 1`, purity is `1` exactly when Cauchy–Schwarz is an equality,
i.e. when `lo` and `hi` are parallel — the qubit factors out.
-/
import Mathlib.Analysis.InnerProductSpace.PiL2

namespace QuantumProofs.ResetModel

open scoped InnerProductSpace

/-- Amplitudes over the `2^n` basis bitstrings. -/
abbrev Vec (n : ℕ) := EuclideanSpace ℂ (Fin n → Bool)

/-- A pure state: a unit vector. -/
def State (n : ℕ) := {s : Vec n // ‖s‖ = 1}

variable {n : ℕ}

/-- Flip bit `q` of a basis bitstring. -/
def flip (q : Fin n) (x : Fin n → Bool) : Fin n → Bool := Function.update x q (!x q)

@[simp] lemma flip_apply_self (q : Fin n) (x : Fin n → Bool) : flip q x q = !x q := by
  simp [flip]

lemma flip_flip (q : Fin n) (x : Fin n → Bool) : flip q (flip q x) = x := by
  funext i
  by_cases h : i = q
  · subst h; simp [flip]
  · simp [flip, Function.update_of_ne h]

lemma flip_involutive (q : Fin n) : Function.Involutive (flip q) := flip_flip q

/-- The amplitudes with bit `q` equal to `b`; zero elsewhere. -/
def part (b : Bool) (q : Fin n) (s : Vec n) : Vec n :=
  WithLp.toLp 2 (fun x => if x q = b then s x else 0)

/-- Pauli `X` on qubit `q`. -/
def applyX (s : Vec n) (q : Fin n) : Vec n := WithLp.toLp 2 (fun x => s (flip q x))

/-- Divide by the norm (the zero vector stays zero). -/
noncomputable def normalize (v : Vec n) : Vec n := ((‖v‖ : ℂ))⁻¹ • v

/-- Projective measurement of `q` with forced outcome `b`, renormalised. -/
noncomputable def project (s : Vec n) (q : Fin n) (b : Bool) : Vec n := normalize (part b q s)

/-- Reset as a channel: measure `q`, then `X` if the outcome was 1. -/
noncomputable def reset (s : Vec n) (q : Fin n) (outcome : Bool) : Vec n :=
  if outcome then applyX (project s q true) q else project s q false

/-- The superseded coherent fold: `new0 = old0 + old1`, renormalised; the zero
    register when the sum vanishes. -/
noncomputable def fold (s : Vec n) (q : Fin n) : Vec n :=
  normalize (part false q s + applyX (part true q s) q)

/-- Probability of outcome 0 / 1 on qubit `q`. -/
noncomputable def p0 (s : Vec n) (q : Fin n) : ℝ := ‖part false q s‖ ^ 2 / ‖s‖ ^ 2
noncomputable def p1 (s : Vec n) (q : Fin n) : ℝ := ‖part true q s‖ ^ 2 / ‖s‖ ^ 2

/-- The forced outcome has nonzero probability. -/
def Possible (s : Vec n) (q : Fin n) (b : Bool) : Prop := (if b then p1 s q else p0 s q) ≠ 0

/-- `q = 0` amplitudes, and `q = 1` amplitudes moved onto `q = 0`. -/
def lo (s : Vec n) (q : Fin n) : Vec n := part false q s
def hi (s : Vec n) (q : Fin n) : Vec n := applyX (part true q s) q

/-- `tr ρ_q²` for the reduced density matrix of qubit `q` (see the header). -/
noncomputable def reducedPurity (s : Vec n) (q : Fin n) : ℝ :=
  (‖lo s q‖ ^ 4 + ‖hi s q‖ ^ 4 + 2 * ‖⟪lo s q, hi s q⟫_ℂ‖ ^ 2) / ‖s‖ ^ 4

/-- `q` is unentangled from the rest. -/
def Deterministic (s : Vec n) (q : Fin n) : Prop := reducedPurity s q = 1

/-- Equality up to global phase. -/
def EqUpToPhase (a b : Vec n) : Prop := ∃ c : ℂ, ‖c‖ = 1 ∧ a = c • b

/-! ## Basic facts -/

lemma EqUpToPhase.symm {a b : Vec n} (h : EqUpToPhase a b) : EqUpToPhase b a := by
  obtain ⟨c, hc, rfl⟩ := h
  have hc0 : c ≠ 0 := by intro h0; simp [h0] at hc
  refine ⟨c⁻¹, by simp [hc], ?_⟩
  rw [smul_smul, inv_mul_cancel₀ hc0, one_smul]

lemma EqUpToPhase.refl (a : Vec n) : EqUpToPhase a a := ⟨1, by simp, by simp⟩

lemma applyX_smul (c : ℂ) (v : Vec n) (q : Fin n) : applyX (c • v) q = c • applyX v q := by
  ext x; simp [applyX]

lemma norm_applyX (v : Vec n) (q : Fin n) : ‖applyX v q‖ = ‖v‖ := by
  have h : ‖applyX v q‖ ^ 2 = ‖v‖ ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq]
    simp only [applyX, PiLp.toLp_apply]
    exact Equiv.sum_comp (flip_involutive q).toPerm (fun x => ‖v.ofLp x‖ ^ 2)
  exact (pow_left_inj₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).mp h

lemma norm_hi (s : Vec n) (q : Fin n) : ‖hi s q‖ = ‖part true q s‖ := norm_applyX _ _

lemma applyX_normalize (v : Vec n) (q : Fin n) : applyX (normalize v) q = normalize (applyX v q) := by
  simp [normalize, applyX_smul, norm_applyX]

lemma norm_normalize {v : Vec n} (hv : v ≠ 0) : ‖normalize v‖ = 1 := norm_smul_inv_norm hv

lemma normalize_ne_zero {v : Vec n} (hv : v ≠ 0) : normalize v ≠ 0 := by
  intro h; have := norm_normalize hv; rw [h, norm_zero] at this; exact zero_ne_one this

/-- `‖s‖² = ‖lo‖² + ‖q=1 part‖²`. -/
lemma norm_sq_split (s : Vec n) (q : Fin n) :
    ‖s‖ ^ 2 = ‖part false q s‖ ^ 2 + ‖part true q s‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq,
    ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun x _ => ?_)
  simp only [part, PiLp.toLp_apply]
  cases x q <;> simp

lemma part_false_lo (s : Vec n) (q : Fin n) : part false q (lo s q) = lo s q := by
  ext x; simp only [part, lo, PiLp.toLp_apply]; split_ifs <;> simp_all

lemma part_false_hi (s : Vec n) (q : Fin n) : part false q (hi s q) = hi s q := by
  ext x
  simp only [part, hi, applyX, PiLp.toLp_apply, flip_apply_self]
  cases hx : x q <;> simp [hx]

lemma part_smul (b : Bool) (q : Fin n) (c : ℂ) (v : Vec n) :
    part b q (c • v) = c • part b q v := by
  ext x; simp only [part, PiLp.toLp_apply, PiLp.smul_apply, smul_eq_mul]; split_ifs <;> simp

/-! ## Reset lands on the `q = 0` subspace -/

lemma reset_false (s : Vec n) (q : Fin n) : reset s q false = normalize (lo s q) := rfl

lemma reset_true (s : Vec n) (q : Fin n) : reset s q true = normalize (hi s q) := by
  simp [reset, project, hi, applyX_normalize]

lemma reset_eq (s : Vec n) (q : Fin n) (b : Bool) :
    reset s q b = normalize (if b then hi s q else lo s q) := by
  cases b
  · exact reset_false s q
  · exact reset_true s q

/-! ## Possibility and purity, for unit states -/

lemma possible_iff (s : State n) (q : Fin n) (b : Bool) :
    Possible s.1 q b ↔ (if b then hi s.1 q else lo s.1 q) ≠ 0 := by
  cases b
  · simp [Possible, p0, s.2, lo, sq_eq_zero_iff, norm_eq_zero]
  · simp [Possible, p1, s.2, hi, sq_eq_zero_iff, norm_eq_zero, ← norm_applyX (part true q s.1) q]

/-- For a unit state, purity `1` is exactly Cauchy–Schwarz equality between the
    two halves. -/
lemma deterministic_iff (s : State n) (q : Fin n) :
    Deterministic s.1 q ↔ ‖⟪lo s.1 q, hi s.1 q⟫_ℂ‖ = ‖lo s.1 q‖ * ‖hi s.1 q‖ := by
  have hsplit : ‖lo s.1 q‖ ^ 2 + ‖hi s.1 q‖ ^ 2 = 1 := by
    rw [norm_hi, lo, ← norm_sq_split, s.2, one_pow]
  simp only [Deterministic, reducedPurity, s.2, one_pow, div_one]
  set a := ‖lo s.1 q‖; set b := ‖hi s.1 q‖; set c := ‖⟪lo s.1 q, hi s.1 q⟫_ℂ‖
  have ha : 0 ≤ a := norm_nonneg _
  have hb : 0 ≤ b := norm_nonneg _
  have hc : 0 ≤ c := norm_nonneg _
  constructor
  · intro h
    have hc2 : c ^ 2 = (a * b) ^ 2 := by nlinarith [hsplit]
    exact (pow_left_inj₀ hc (mul_nonneg ha hb) two_ne_zero).mp hc2
  · intro h
    have hc2 : c ^ 2 = (a * b) ^ 2 := by rw [h]
    nlinarith [hsplit, hc2]

/-- `normalize (r • w) = (r/‖r‖) • normalize w` for nonzero `r`, `w`. -/
lemma normalize_smul {r : ℂ} {w : Vec n} (hr : r ≠ 0) :
    normalize (r • w) = (r / (‖r‖ : ℂ)) • normalize w := by
  have hr' : (‖r‖ : ℂ) ≠ 0 := by exact_mod_cast norm_ne_zero_iff.mpr hr
  simp only [normalize, norm_smul, smul_smul]
  congr 1
  push_cast
  field_simp

lemma phase_of_parallel {u v : Vec n} {r : ℂ} (hr : r ≠ 0) (hv : v = r • u) :
    EqUpToPhase (normalize v) (normalize u) := by
  refine ⟨r / (‖r‖ : ℂ), ?_, by rw [hv, normalize_smul hr]⟩
  have : ‖r‖ ≠ 0 := norm_ne_zero_iff.mpr hr
  simp [norm_div, this]

/-! ## The four targets -/

/-- **T1 — outcome-independent on an unentangled qubit.** -/
theorem reset_outcome_irrelevant (s : State n) (q : Fin n)
    (h : Deterministic s.1 q) (b b' : Bool)
    (hb : Possible s.1 q b) (hb' : Possible s.1 q b') :
    EqUpToPhase (reset s.1 q b) (reset s.1 q b') := by
  have parallel : (lo s.1 q ≠ 0) → (hi s.1 q ≠ 0) →
      EqUpToPhase (normalize (hi s.1 q)) (normalize (lo s.1 q)) := by
    intro hl hh
    obtain ⟨r, hr, hv⟩ := (norm_inner_eq_norm_iff hl hh).mp ((deterministic_iff s q).mp h)
    exact phase_of_parallel hr hv
  rw [possible_iff] at hb hb'
  cases b <;> cases b' <;> simp only [reset_eq, Bool.false_eq_true, ↓reduceIte] at hb hb' ⊢
  · exact EqUpToPhase.refl _
  · exact (parallel hb hb').symm
  · exact parallel hb' hb
  · exact EqUpToPhase.refl _

/-- **T2 — Reset lands in `|0⟩`.** In this model it needs only that the forced
    branch is possible, not that the qubit is unentangled. -/
theorem reset_yields_zero (s : State n) (q : Fin n) (b : Bool)
    (_h : Deterministic s.1 q) (hb : Possible s.1 q b) :
    p0 (reset s.1 q b) q = 1 := by
  rw [possible_iff] at hb
  rw [reset_eq]
  set w := if b then hi s.1 q else lo s.1 q
  have hw : part false q w = w := by
    simp only [w]; split
    · exact part_false_hi _ _
    · exact part_false_lo _ _
  have hpn : part false q (normalize w) = normalize w := by
    simp only [normalize]; rw [part_smul, hw]
  simp only [p0, hpn, norm_normalize hb, one_pow, div_one]

/-- **T3 — the fold is not the reset channel**, witnessed by `|−⟩`, on which
    the fold produces the zero register. -/
theorem fold_is_not_reset :
    ∃ (n : ℕ) (s : State n) (q : Fin n) (b : Bool), ¬ EqUpToPhase (fold s.1 q) (reset s.1 q b) := by
  let v : Vec 1 := WithLp.toLp 2 (fun x => if x 0 then -1 else 1)
  have hv : v ≠ 0 := by
    intro h
    have := congrArg (fun w : Vec 1 => w (fun _ => false)) h
    simp [v] at this
  let s : State 1 := ⟨normalize v, norm_normalize hv⟩
  refine ⟨1, s, 0, false, ?_⟩
  have hfold : fold s.1 0 = 0 := by
    have hsum : part false 0 s.1 + applyX (part true 0 s.1) 0 = 0 := by
      ext x
      simp only [s, normalize, v, part, applyX, PiLp.add_apply, PiLp.toLp_apply,
        PiLp.smul_apply, flip_apply_self, smul_eq_mul, PiLp.zero_apply]
      cases hx : x 0 <;> simp [hx]
    simp [fold, hsum, normalize]
  have hlo : lo s.1 0 ≠ 0 := by
    intro h
    have := congrArg (fun w : Vec 1 => w (fun _ => false)) h
    simp only [lo, part, s, normalize, v, PiLp.toLp_apply, PiLp.smul_apply, smul_eq_mul,
      PiLp.zero_apply] at this
    simp [norm_ne_zero_iff.mpr hv] at this
    have := congrFun this (fun _ => false)
    simp at this
  rintro ⟨c, hc, h⟩
  rw [hfold, reset_false] at h
  have := congrArg norm h
  rw [norm_zero, norm_smul, hc, norm_normalize hlo, mul_one] at this
  exact zero_ne_one this

/-- **T4 — an entangled Reset has no pure-state result**: the two branches are
    not the same state up to phase. -/
theorem entangled_reset_not_pure (s : State n) (q : Fin n)
    (h : ¬ Deterministic s.1 q) :
    ∀ b : Bool, ¬ EqUpToPhase (reset s.1 q b) (reset s.1 q (!b)) := by
  have hl : lo s.1 q ≠ 0 := by
    intro h0; apply h; rw [deterministic_iff, h0]; simp
  have hh : hi s.1 q ≠ 0 := by
    intro h0; apply h; rw [deterministic_iff, h0]; simp
  -- if `normalize hi` is a phase times `normalize lo`, the halves are parallel
  have key : ¬ EqUpToPhase (normalize (hi s.1 q)) (normalize (lo s.1 q)) := by
    rintro ⟨c, hc, hcv⟩
    have hc0 : c ≠ 0 := by intro h0; simp [h0] at hc
    have hhl : hi s.1 q = (((‖hi s.1 q‖ : ℂ) * c * (‖lo s.1 q‖ : ℂ)⁻¹)) • lo s.1 q := by
      have e1 : hi s.1 q = (‖hi s.1 q‖ : ℂ) • normalize (hi s.1 q) := by
        have : (‖hi s.1 q‖ : ℂ) ≠ 0 := by exact_mod_cast norm_ne_zero_iff.mpr hh
        simp [normalize, smul_smul, mul_inv_cancel₀ this]
      conv_lhs => rw [e1]
      rw [hcv, normalize, smul_smul, smul_smul]
    have hr : (‖hi s.1 q‖ : ℂ) * c * (‖lo s.1 q‖ : ℂ)⁻¹ ≠ 0 := by
      have h1 : (‖hi s.1 q‖ : ℂ) ≠ 0 := by exact_mod_cast norm_ne_zero_iff.mpr hh
      have h2 : (‖lo s.1 q‖ : ℂ) ≠ 0 := by exact_mod_cast norm_ne_zero_iff.mpr hl
      exact mul_ne_zero (mul_ne_zero h1 hc0) (inv_ne_zero h2)
    exact h ((deterministic_iff s q).mpr ((norm_inner_eq_norm_iff hl hh).mpr ⟨_, hr, hhl⟩))
  intro b
  cases b <;> simp only [reset_eq, Bool.not_false, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  · exact fun e => key e.symm
  · exact key

/-! ## Instantiation witnesses: every premise above is inhabited -/

/-- `|+⟩` on one qubit. -/
noncomputable def plus : State 1 :=
  ⟨normalize (WithLp.toLp 2 (fun _ => 1)), norm_normalize (by
    intro h; have := congrArg (fun w : Vec 1 => w (fun _ => false)) h; simp at this)⟩

lemma lo_plus_eq_hi : lo plus.1 0 = hi plus.1 0 := by
  ext x; simp only [lo, hi, part, applyX, plus, normalize, PiLp.toLp_apply, PiLp.smul_apply,
    flip_apply_self, smul_eq_mul]
  cases hx : x 0 <;> simp [hx]

lemma lo_plus_ne_zero : lo plus.1 0 ≠ 0 := by
  intro h
  have := congrArg (fun w : Vec 1 => w (fun _ => false)) h
  simp only [lo, part, plus, normalize, PiLp.toLp_apply, PiLp.smul_apply, smul_eq_mul,
    PiLp.zero_apply] at this
  simp at this
  have := congrFun this (fun _ => false)
  simp at this

/-- T1's premises with **different** outcomes, both possible, on an unentangled
    qubit: `|+⟩`. -/
theorem witness_T1 :
    Deterministic plus.1 0 ∧ Possible plus.1 0 false ∧ Possible plus.1 0 true := by
  refine ⟨?_, ?_, ?_⟩
  · rw [deterministic_iff, ← lo_plus_eq_hi, inner_self_eq_norm_sq_to_K]
    simp [sq]
  · rw [possible_iff]; exact lo_plus_ne_zero
  · rw [possible_iff]; simpa [← lo_plus_eq_hi] using lo_plus_ne_zero

/-- T2's premises: `|+⟩`, outcome 1. -/
theorem witness_T2 : Deterministic plus.1 0 ∧ Possible plus.1 0 true :=
  ⟨witness_T1.1, witness_T1.2.2⟩

/-- The Bell state `(|00⟩ + |11⟩)/√2`. -/
noncomputable def bell : State 2 :=
  ⟨normalize (WithLp.toLp 2 (fun x => if x 0 = x 1 then 1 else 0)), norm_normalize (by
    intro h; have := congrArg (fun w : Vec 2 => w (fun _ => false)) h; simp at this)⟩

/-- T4's premise: qubit 0 of a Bell state is entangled. -/
theorem witness_T4 : ¬ Deterministic bell.1 0 := by
  have hl : lo bell.1 0 ≠ 0 := by
    intro h
    have := congrArg (fun w : Vec 2 => w (fun _ => false)) h
    simp only [lo, part, bell, normalize, PiLp.toLp_apply, PiLp.smul_apply, smul_eq_mul,
      PiLp.zero_apply] at this
    simp at this
    have := congrFun this (fun _ => false)
    simp at this
  have hh : hi bell.1 0 ≠ 0 := by
    intro h
    have := congrArg (fun w : Vec 2 => w (fun i => if i = 1 then true else false)) h
    simp only [hi, part, applyX, bell, normalize, PiLp.toLp_apply, PiLp.smul_apply, smul_eq_mul,
      PiLp.zero_apply, flip_apply_self] at this
    simp [flip] at this
    have := congrFun this (fun _ => false)
    simp at this
  have hinner : ⟪lo bell.1 0, hi bell.1 0⟫_ℂ = 0 := by
    rw [PiLp.inner_apply]
    refine Finset.sum_eq_zero (fun x _ => ?_)
    simp only [lo, hi, part, applyX, bell, normalize, PiLp.toLp_apply, PiLp.smul_apply,
      smul_eq_mul, flip_apply_self]
    cases h0 : x 0 <;> cases h1 : x 1 <;> simp [flip, h0, h1, Function.update]
  rw [deterministic_iff, hinner, norm_zero]
  exact fun h => (mul_ne_zero (norm_ne_zero_iff.mpr hl) (norm_ne_zero_iff.mpr hh)) h.symm

/-! ## The impossible branch is reachable, and T2 fails there

`project` returns the zero vector when the forced outcome has nothing to
renormalise. None of the four targets takes that branch (T1/T2 require
`Possible`; T4 proves both halves nonzero; T3 uses the possible branch of
`|−⟩`). This fixture shows the branch is real and that `Possible` is what
keeps T2 true: on `|0⟩` with forced outcome 1 the branch is impossible, reset
returns the zero vector, and `p0` of it is `0`, not `1`. -/

/-- `|0⟩` on one qubit. -/
noncomputable def ket0 : State 1 :=
  ⟨normalize (WithLp.toLp 2 (fun x => if x 0 then 0 else 1)), norm_normalize (by
    intro h; have := congrArg (fun w : Vec 1 => w (fun _ => false)) h; simp at this)⟩

lemma part_true_ket0 : part true 0 ket0.1 = 0 := by
  ext x
  simp only [part, ket0, normalize, PiLp.toLp_apply, PiLp.smul_apply, smul_eq_mul, PiLp.zero_apply]
  cases hx : x 0 <;> simp [hx]

theorem reset_yields_zero_needs_possible :
    ¬ Possible ket0.1 0 true ∧ reset ket0.1 0 true = 0 ∧ p0 (reset ket0.1 0 true) 0 ≠ 1 := by
  have hhi : hi ket0.1 0 = 0 := by
    simp only [hi, part_true_ket0]; ext x; simp [applyX]
  have hreset : reset ket0.1 0 true = 0 := by
    rw [reset_true, hhi]; simp [normalize]
  refine ⟨?_, hreset, ?_⟩
  · rw [possible_iff]; simp [hhi]
  · rw [hreset]; simp [p0]

end QuantumProofs.ResetModel
