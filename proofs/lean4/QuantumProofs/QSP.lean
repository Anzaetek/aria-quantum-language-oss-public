/-
  Quantum Signal Processing (QSP) — the forward fundamental theorem.

  A QSP sequence interleaves the signal/rotation operator
    W(x) = [[x, i·s], [i·s, x]]   (s = √(1−x²), an X-rotation)
  with single-qubit Z-phases  S(φ) = diag(e^{iφ}, e^{-iφ}):
    U_Φ(x) = S(φ₀) · ∏_{k} (W(x) · S(φ_k)).

  **Forward QSP theorem (this file, sorry-free):** for *any* phase list of
  length `d`, the resulting unitary has the polynomial form
    U_Φ(x) = [[A(x), s·B(x)], [s·C(x), D(x)]]
  with `A, B, C, D : ℂ[X]`, `deg A, deg D ≤ d`, `deg B, deg C < d`. In
  particular the `(0,0)` entry **is** a degree-`≤ d` polynomial in `x` — the
  phase sequence *implements a polynomial transform* (`qsp_implements_poly`).

  We use the conjugate-free 4-polynomial form `qspForm4` (diagonal = poly(x),
  off-diagonal = s·poly(x)), which is closed under multiplication by pure
  polynomial arithmetic plus the single substitution `s² = 1 − x²`
  (`qspForm4_mul`) — no QSP conjugation bookkeeping.

  **Scope.** This file proves the *forward* direction in full (polynomial form
  `qsp_is_qspForm4`, degree budget `qsp_is_qspForm4_degree`, parity `qsp_parity`)
  and, for the converse, both *ends* of its induction: the algebraic backbone —
  each QSP step is invertible (`phase_neg_mul_phase`, `signalInv_mul_signal`) so
  `qsp_peel` unwinds one `phase·signal` factor (the inductive step) — and the
  constructive base cases `qsp_converse_const` (`d = 0`), `qsp_converse_linear`
  (`d = 1`: every unit-modulus leading coefficient is realized by `[arg c]`) and
  `qsp_converse_deg2` (`d = 2`: every `(a, b)` with `‖a‖ = ‖b‖ = 1` is realized by
  a 2-phase list). The reduction *step* is also pinned down: `qspForm4_peel` gives
  the explicit peeled polynomials and `qspForm4_peel_top_coeff` computes the peeled
  diagonal's top coefficient `e^{-iφ}·lead(pA) + i·e^{iφ}·lead(pC)` — the exact
  quantity the inductive angle-choice cancels. The remaining converse content
  (proving that cancellation, with parity and the unitarity identity, forces the
  degree to drop and recurses — the completion/leading-coefficient induction of
  Gilyén–Su–Low–Wiebe) stays the deferred analytic residue.
-/

import QuantumProofs.Gates
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Algebra.Polynomial.Degree.Defs
import Mathlib.Algebra.Polynomial.Degree.Lemmas
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.Analysis.SpecialFunctions.Complex.Arg
import Mathlib.Tactic.ComputeDegree

namespace QuantumProofs.QSP

open Complex Polynomial Matrix

/-- The QSP signal (X-rotation) operator `W(x) = [[x, i·s],[i·s, x]]`, with
    `s` standing for `√(1−x²)`. -/
noncomputable def signal (x s : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![(x : ℂ), I * (s : ℂ); I * (s : ℂ), (x : ℂ)]

/-- The QSP phase operator `S(φ) = e^{iφZ} = diag(e^{iφ}, e^{-iφ})`. -/
noncomputable def phase (φ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![Complex.exp (I * (φ : ℂ)), 0; 0, Complex.exp (-(I * (φ : ℂ)))]

/-- The 4-polynomial QSP form: diagonal entries are `poly(x)`, off-diagonal
    entries are `s·poly(x)`. Conjugate-free and closed under multiplication. -/
noncomputable def qspForm4 (A B C D : Polynomial ℂ) (x s : ℝ) :
    Matrix (Fin 2) (Fin 2) ℂ :=
  !![A.eval (x : ℂ), (s : ℂ) * B.eval (x : ℂ);
     (s : ℂ) * C.eval (x : ℂ), D.eval (x : ℂ)]

/-- `s² = 1 − x²` lifted to `ℂ`. -/
private lemma hs_cast {x s : ℝ} (hs : s ^ 2 = 1 - x ^ 2) :
    (s : ℂ) ^ 2 = 1 - (x : ℂ) ^ 2 := by
  rw [← Complex.ofReal_pow, hs]; push_cast; ring

/-- **Closure of `qspForm4` under multiplication.** The product of two QSP
    forms is again a QSP form, with the new polynomials given by pure
    polynomial arithmetic (the `(1−X²)` factors come from `s² = 1−x²`). -/
lemma qspForm4_mul (A B C D A' B' C' D' : Polynomial ℂ) (x s : ℝ)
    (hs : s ^ 2 = 1 - x ^ 2) :
    qspForm4 A B C D x s * qspForm4 A' B' C' D' x s =
      qspForm4 (A * A' + (1 - X ^ 2) * B * C') (A * B' + B * D')
        (C * A' + D * C') ((1 - X ^ 2) * C * B' + D * D') x s := by
  have hsc : (s : ℂ) ^ 2 = 1 - (x : ℂ) ^ 2 := hs_cast hs
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [qspForm4, Matrix.mul_apply, Fin.sum_univ_two]
  · linear_combination (B.eval (x : ℂ) * C'.eval (x : ℂ)) * hsc
  · ring
  · ring
  · linear_combination (C.eval (x : ℂ) * B'.eval (x : ℂ)) * hsc

/-- `signal` as a QSP form. -/
lemma signal_eq (x s : ℝ) : signal x s = qspForm4 X (C I) (C I) X x s := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [signal, qspForm4, eval_X, eval_C] <;> ring

/-- `phase φ` as a QSP form (off-diagonal polynomials are 0). -/
lemma phase_eq (φ : ℝ) (x s : ℝ) :
    phase φ = qspForm4 (C (Complex.exp (I * (φ : ℂ)))) 0 0
      (C (Complex.exp (-(I * (φ : ℂ))))) x s := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [phase, qspForm4, eval_C]

/-- The identity is the trivial QSP form. -/
lemma one_eq (x s : ℝ) : (1 : Matrix (Fin 2) (Fin 2) ℂ) = qspForm4 (C 1) 0 0 (C 1) x s := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [qspForm4]

/-- The QSP unitary for a phase list `φs = [φ₀, …]`:
    `U = S(φ₀)·W·S(φ₁)·W·…` (each phase followed by a signal). -/
noncomputable def qsp (x s : ℝ) : List ℝ → Matrix (Fin 2) (Fin 2) ℂ
  | [] => 1
  | (φ :: rest) => phase φ * signal x s * qsp x s rest

/-- **Forward QSP — structural closure.** For *any* phase list, the QSP
    unitary has the 4-polynomial form `qspForm4 A B C D`: every entry is a
    polynomial in `x` (diagonal) or `s·(polynomial in x)` (off-diagonal),
    even though the signal `W(x)` itself has the non-polynomial entry
    `s = √(1−x²)`. Proven by induction via `qspForm4_mul`. -/
theorem qsp_is_qspForm4 (x s : ℝ) (hs : s ^ 2 = 1 - x ^ 2) (φs : List ℝ) :
    ∃ A B C D : Polynomial ℂ, qsp x s φs = qspForm4 A B C D x s := by
  induction φs with
  | nil => exact ⟨C 1, 0, 0, C 1, one_eq x s⟩
  | cons φ rest ih =>
    obtain ⟨A, B, C, D, h⟩ := ih
    rw [show qsp x s (φ :: rest) = phase φ * signal x s * qsp x s rest from rfl, h,
      phase_eq φ x s, signal_eq x s,
      qspForm4_mul _ _ _ _ _ _ _ _ x s hs, qspForm4_mul _ _ _ _ _ _ _ _ x s hs]
    exact ⟨_, _, _, _, rfl⟩

/-- **Forward QSP — the polynomial transform.** The `(0,0)` entry of the QSP
    unitary *is* the evaluation of a fixed polynomial `P ∈ ℂ[X]` at `x`: the
    phase sequence implements `x ↦ P(x)`. This is the forward direction of
    the QSP fundamental theorem. -/
theorem qsp_implements_poly (x s : ℝ) (hs : s ^ 2 = 1 - x ^ 2) (φs : List ℝ) :
    ∃ P : Polynomial ℂ, (qsp x s φs) 0 0 = P.eval (x : ℂ) := by
  obtain ⟨A, B, C, D, h⟩ := qsp_is_qspForm4 x s hs φs
  exact ⟨A, by rw [h]; simp [qspForm4]⟩

/-! ### Degree bound — the QSP polynomial has degree `≤ d`

The forward theorem above shows the `(0,0)` entry is *some* polynomial. We now
pin down its degree: a phase list of length `d` implements a polynomial of
degree `≤ d`, with the off-diagonal numerators of degree `< d`. This is the
quantitative half of the forward QSP fundamental theorem — the degree budget
that makes QSP a degree-`d` polynomial transform.

The bounds are tracked through the induction in the `+ 1 ≤ ↑d` form for the
off-diagonal polynomials `B, C` (equivalent to `degree < ↑d` over `WithBot ℕ`,
but it composes directly with `degree_mul_le`, and `degree 0 = ⊥` makes the
base case `⊥ + 1 = ⊥ ≤ 0` immediate). -/

/-- `degree (1 - X²) ≤ 2`. -/
private lemma degree_one_sub_X_sq_le : (1 - X ^ 2 : Polynomial ℂ).degree ≤ 2 := by
  compute_degree!

/-- `degree (C a * X) ≤ 1`. -/
private lemma degree_C_mul_X_le (a : ℂ) : (C a * X : Polynomial ℂ).degree ≤ 1 := by
  have h2 : (C a : Polynomial ℂ).degree + (X : Polynomial ℂ).degree ≤ 0 + 1 :=
    add_le_add degree_C_le degree_X_le
  simpa using (degree_mul_le _ _).trans h2

/-- `degree (C a * C b) ≤ 0`. -/
private lemma degree_C_mul_C_le (a b : ℂ) : (C a * C b : Polynomial ℂ).degree ≤ 0 := by
  have h2 : (C a : Polynomial ℂ).degree + (C b : Polynomial ℂ).degree ≤ 0 + 0 :=
    add_le_add degree_C_le degree_C_le
  simpa using (degree_mul_le _ _).trans h2

/-- `degree ((1 - X²) · (C a · C b)) ≤ 2`. -/
private lemma degree_sub_C_C_le (a b : ℂ) :
    ((1 - X ^ 2) * (C a * C b) : Polynomial ℂ).degree ≤ 2 :=
  (degree_mul_le _ _).trans <| by
    simpa using add_le_add degree_one_sub_X_sq_le (degree_C_mul_C_le a b)

/-- WithBot ℕ: from `a + 1 ≤ d` conclude `2 + a ≤ d + 1`. -/
private lemma wb_two_add {a d : WithBot ℕ} (h : a + 1 ≤ d) : (2 : WithBot ℕ) + a ≤ d + 1 := by
  have e : (2 : WithBot ℕ) + a = a + 1 + 1 := by
    have h2 : (2 : WithBot ℕ) = 1 + 1 := by decide
    rw [h2]; abel
  rw [e]; exact add_le_add h le_rfl

/-- WithBot ℕ: from `a + 1 ≤ d` conclude `1 + a ≤ d`. -/
private lemma wb_one_add_le {a d : WithBot ℕ} (h : a + 1 ≤ d) : (1 : WithBot ℕ) + a ≤ d :=
  (add_comm (1 : WithBot ℕ) a).le.trans h

/-- The product `phase φ · signal` is the QSP form of a length-1 phase list:
    diagonal degree `≤ 1`, off-diagonal degree `0`. Holds for *all* `x, s`
    (no `s² = 1 − x²` needed — the off-diagonal `(1−X²)` terms are killed by
    the zero entries of `phase`). -/
lemma phase_mul_signal (φ x s : ℝ) :
    phase φ * signal x s =
      qspForm4 (C (Complex.exp (I * (φ : ℂ))) * X) (C (Complex.exp (I * (φ : ℂ))) * C I)
        (C (Complex.exp (-(I * (φ : ℂ)))) * C I) (C (Complex.exp (-(I * (φ : ℂ)))) * X) x s := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [phase, signal, qspForm4, Matrix.mul_apply, Fin.sum_univ_two, eval_mul, eval_C,
      eval_X] <;> ring

/-- **Forward QSP with degree budget.** For a phase list of length `d`, the QSP
    unitary is `qspForm4 A B C D` with `deg A, deg D ≤ d` and
    `deg B + 1, deg C + 1 ≤ d` (i.e. `deg B, deg C < d`). Proven by induction:
    each `phase·signal` step (a length-1 form, [`phase_mul_signal`]) multiplies
    the running form, lifting the degree budget by exactly one via
    [`qspForm4_mul`] and `degree_mul_le`. -/
theorem qsp_is_qspForm4_degree (x s : ℝ) (hs : s ^ 2 = 1 - x ^ 2) (φs : List ℝ) :
    ∃ A B C D : Polynomial ℂ,
      qsp x s φs = qspForm4 A B C D x s ∧
      A.degree ≤ (φs.length : WithBot ℕ) ∧ D.degree ≤ (φs.length : WithBot ℕ) ∧
      B.degree + 1 ≤ (φs.length : WithBot ℕ) ∧ C.degree + 1 ≤ (φs.length : WithBot ℕ) := by
  induction φs with
  | nil =>
    refine ⟨C 1, 0, 0, C 1, one_eq x s, ?_, ?_, ?_, ?_⟩
    · simp
    · simp
    · simp
    · simp
  | cons φ rest ih =>
    obtain ⟨A₀, B₀, C₀, D₀, heq, hA, hD, hB, hC⟩ := ih
    have key : qsp x s (φ :: rest) =
        qspForm4
          (C (Complex.exp (I * (φ : ℂ))) * X * A₀
            + (1 - X ^ 2) * (C (Complex.exp (I * (φ : ℂ))) * C I) * C₀)
          (C (Complex.exp (I * (φ : ℂ))) * X * B₀
            + C (Complex.exp (I * (φ : ℂ))) * C I * D₀)
          (C (Complex.exp (-(I * (φ : ℂ)))) * C I * A₀
            + C (Complex.exp (-(I * (φ : ℂ)))) * X * C₀)
          ((1 - X ^ 2) * (C (Complex.exp (-(I * (φ : ℂ)))) * C I) * B₀
            + C (Complex.exp (-(I * (φ : ℂ)))) * X * D₀) x s := by
      rw [show qsp x s (φ :: rest) = phase φ * signal x s * qsp x s rest from rfl, heq,
        phase_mul_signal, qspForm4_mul _ _ _ _ _ _ _ _ x s hs]
    refine ⟨_, _, _, _, key, ?_, ?_, ?_, ?_⟩ <;>
      rw [List.length_cons, Nat.cast_add_one]
    -- A: diagonal, deg ≤ d + 1
    · refine (degree_add_le _ _).trans (max_le ?_ ?_)
      · exact (degree_mul_le _ _).trans
          ((add_le_add (degree_C_mul_X_le _) hA).trans (le_of_eq (add_comm _ _)))
      · exact (degree_mul_le _ _).trans
          ((add_le_add (degree_sub_C_C_le _ _) le_rfl).trans (wb_two_add hC))
    -- D: diagonal, deg ≤ d + 1
    · refine (degree_add_le _ _).trans (max_le ?_ ?_)
      · exact (degree_mul_le _ _).trans
          ((add_le_add (degree_sub_C_C_le _ _) le_rfl).trans (wb_two_add hB))
      · exact (degree_mul_le _ _).trans
          ((add_le_add (degree_C_mul_X_le _) hD).trans (le_of_eq (add_comm _ _)))
    -- B: off-diagonal, deg + 1 ≤ d + 1
    · refine add_le_add ((degree_add_le _ _).trans (max_le ?_ ?_)) le_rfl
      · exact (degree_mul_le _ _).trans
          ((add_le_add (degree_C_mul_X_le _) le_rfl).trans (wb_one_add_le hB))
      · exact (degree_mul_le _ _).trans
          ((add_le_add (degree_C_mul_C_le _ _) hD).trans (le_of_eq (zero_add _)))
    -- C: off-diagonal, deg + 1 ≤ d + 1
    · refine add_le_add ((degree_add_le _ _).trans (max_le ?_ ?_)) le_rfl
      · exact (degree_mul_le _ _).trans
          ((add_le_add (degree_C_mul_C_le _ _) hA).trans (le_of_eq (zero_add _)))
      · exact (degree_mul_le _ _).trans
          ((add_le_add (degree_C_mul_X_le _) le_rfl).trans (wb_one_add_le hC))

/-- **Forward QSP — degree-`d` polynomial transform.** The `(0,0)` entry of the
    QSP unitary for a length-`d` phase list is `P.eval x` for a fixed
    `P ∈ ℂ[X]` of degree `≤ d`. Strengthens [`qsp_implements_poly`] with the
    explicit degree budget: a length-`d` phase sequence implements a polynomial
    of degree at most `d`. -/
theorem qsp_implements_poly_degree (x s : ℝ) (hs : s ^ 2 = 1 - x ^ 2) (φs : List ℝ) :
    ∃ P : Polynomial ℂ, (qsp x s φs) 0 0 = P.eval (x : ℂ)
      ∧ P.degree ≤ (φs.length : WithBot ℕ) := by
  obtain ⟨A, B, C, D, h, hA, _, _, _⟩ := qsp_is_qspForm4_degree x s hs φs
  exact ⟨A, by rw [h]; simp [qspForm4], hA⟩

/-! ### Converse direction — QSP-step invertibility (peeling backbone)

The *converse* of the fundamental theorem (every admissible polynomial is
realized by some phase sequence) is proved in the literature by a downward
induction that **peels** one `S(φ)·W` factor off the front of the QSP unitary at
a time, reducing the degree by one. The engine of that induction is the fact
that each step is *invertible*: `phase φ` and `signal x s` are unitary, with
explicit inverses, so the running unitary can be unwound one factor at a time.

We make that backbone rigorous here, sorry-free: the phase inverse is `phase
(-φ)`, the signal inverse `signalInv x s = [[x, -i·s],[-i·s, x]]` (the adjugate;
`det = x² + s² = 1` when `s² = 1−x²`), and `qsp_peel` is the exact unwinding
identity `qsp rest = signalInv · phase(-φ) · qsp (φ :: rest)`. The remaining
content of the converse — choosing each `φ` from the leading coefficients of the
target polynomial so the peeled remainder stays an admissible lower-degree pair
(the Gilyén–Su–Low–Wiebe leading-coefficient / complementary-polynomial
induction) — stays the deferred analytic residue. -/

/-- The inverse signal `W(x)⁻¹ = [[x, -i·s],[-i·s, x]]` (the adjugate of
    `signal`; `det signal = x² + s² = 1` under `s² = 1−x²`). -/
noncomputable def signalInv (x s : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  !![(x : ℂ), -(I * (s : ℂ)); -(I * (s : ℂ)), (x : ℂ)]

/-- `signalInv` as a QSP form (so peeling stays inside the polynomial world):
    `signalInv = qspForm4 X (C (-I)) (C (-I)) X`. -/
lemma signalInv_eq (x s : ℝ) :
    signalInv x s = qspForm4 X (C (-I)) (C (-I)) X x s := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [signalInv, qspForm4, eval_X, eval_C] <;> ring

/-- `phase (-φ)` as a QSP form with the exponentials in `+φ` form:
    `phase (-φ) = qspForm4 (C e⁻¹) 0 0 (C e)` where `e = e^{iφ}`. -/
lemma phase_neg_eq (φ x s : ℝ) :
    phase (-φ) = qspForm4 (C (Complex.exp (-(I * (φ : ℂ))))) 0 0
      (C (Complex.exp (I * (φ : ℂ)))) x s := by
  rw [phase_eq (-φ) x s]
  simp only [Complex.ofReal_neg, mul_neg, neg_neg]

/-- `phase (-φ)` is the inverse of `phase φ`: `phase (-φ) · phase φ = 1`. -/
lemma phase_neg_mul_phase (φ : ℝ) : phase (-φ) * phase φ = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [phase, Matrix.mul_apply, Fin.sum_univ_two, Complex.ofReal_neg,
      mul_neg, ← Complex.exp_add]

/-- `signalInv x s` is a left inverse of `signal x s` (needs `s² = 1−x²` so the
    `x² + s²` diagonal collapses to `1`). -/
lemma signalInv_mul_signal (x s : ℝ) (hs : s ^ 2 = 1 - x ^ 2) :
    signalInv x s * signal x s = 1 := by
  have hsc : (s : ℂ) ^ 2 = 1 - (x : ℂ) ^ 2 := hs_cast hs
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [signal, signalInv, Matrix.mul_apply, Fin.sum_univ_two]
  · linear_combination hsc - (s : ℂ) ^ 2 * Complex.I_sq
  · ring
  · ring
  · linear_combination hsc - (s : ℂ) ^ 2 * Complex.I_sq

/-- **Peeling identity (converse backbone).** One `phase·signal` factor unwinds
    from the front of the QSP unitary: the length-`d` tail is recovered from the
    length-`(d+1)` unitary by the explicit inverses. This is the step that the
    GSLW converse induction iterates (choosing `φ` to keep the remainder
    admissible). -/
lemma qsp_peel (x s : ℝ) (hs : s ^ 2 = 1 - x ^ 2) (φ : ℝ) (rest : List ℝ) :
    qsp x s rest = signalInv x s * phase (-φ) * qsp x s (φ :: rest) := by
  rw [show qsp x s (φ :: rest) = phase φ * signal x s * qsp x s rest from rfl]
  rw [show signalInv x s * phase (-φ) * (phase φ * signal x s * qsp x s rest)
        = signalInv x s * (phase (-φ) * phase φ) * signal x s * qsp x s rest by
        simp only [Matrix.mul_assoc],
    phase_neg_mul_phase, Matrix.mul_one, signalInv_mul_signal x s hs, Matrix.one_mul]

/-! ### Unitarity — the QSP form is unitary (keystone for the converse)

The converse degree-reduction induction needs the QSP form to be *unitary*: that
is what forces the leading coefficients of the diagonal/off-diagonal polynomials
to have equal modulus (so the cancelling angle exists) and what is preserved when
a factor is peeled. Unitarity is clean here because the two building blocks are
unitary with the inverses already in hand: `(signal)ᴴ = signalInv` and
`(phase φ)ᴴ = phase (-φ)`, so `qsp` — a product of them — is unitary by induction,
each step collapsing via `phase(-φ)·phase φ = 1` and `signalInv·signal = 1`. -/

/-- `(signal x s)ᴴ = signalInv x s` (conjugate-transpose of the X-rotation is its
    adjugate, since `x, s` are real and `conj (i·s) = -i·s`). -/
lemma signal_conjTranspose (x s : ℝ) : (signal x s)ᴴ = signalInv x s := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [signal, signalInv, Matrix.conjTranspose_apply, Complex.conj_I]

/-- `(phase φ)ᴴ = phase (-φ)` (conjugate-transpose of the Z-phase negates the
    angle, since `conj e^{iφ} = e^{-iφ}`). -/
lemma phase_conjTranspose (φ : ℝ) : (phase φ)ᴴ = phase (-φ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [phase, Matrix.conjTranspose_apply, ← Complex.exp_conj, Complex.conj_I,
      Complex.ofReal_neg, mul_neg, neg_mul]

/-- **The QSP unitary is unitary** (`Uᴴ U = 1`). The keystone the converse
    degree-reduction consumes: a product of unitary `phase`/`signal` factors,
    proved by induction with each step collapsing through `phase_conjTranspose` +
    `phase_neg_mul_phase` and `signal_conjTranspose` + `signalInv_mul_signal`. -/
theorem qsp_unitary (x s : ℝ) (hs : s ^ 2 = 1 - x ^ 2) (φs : List ℝ) :
    (qsp x s φs)ᴴ * qsp x s φs = 1 := by
  induction φs with
  | nil => simp [show qsp x s [] = 1 from rfl]
  | cons φ rest ih =>
    rw [show qsp x s (φ :: rest) = phase φ * signal x s * qsp x s rest from rfl,
      Matrix.conjTranspose_mul, Matrix.conjTranspose_mul, signal_conjTranspose,
      phase_conjTranspose]
    rw [show (qsp x s rest)ᴴ * (signalInv x s * phase (-φ))
            * (phase φ * signal x s * qsp x s rest)
          = (qsp x s rest)ᴴ * signalInv x s * (phase (-φ) * phase φ) * signal x s
              * qsp x s rest by
          simp only [Matrix.mul_assoc],
      phase_neg_mul_phase, Matrix.mul_one,
      Matrix.mul_assoc (qsp x s rest)ᴴ (signalInv x s) (signal x s),
      signalInv_mul_signal x s hs, Matrix.mul_one, ih]

/-! ### Unitarity as a polynomial identity (`‖lead A‖ = ‖lead C‖` source)

`qsp_unitary` is a *matrix* identity at each real `(x, s)`. The converse
degree-reduction wants its `(0,0)` entry as a **polynomial identity**
`A·A⋆ + (1−X²)·C·C⋆ = 1` (where `P⋆ = P.map conj` is the conjugate-coefficient
polynomial). Comparing the top-degree coefficients of *that* identity is what
forces `‖lead A‖ = ‖lead C‖`, hence the existence of the cancelling angle. We get
it by hoisting the QSP polynomials out of the per-`(x, s)` existential
(`qsp_polys`) and then observing the `(0,0)` Gram relation holds for every
`x ∈ [-1,1]` (take `s = √(1−x²)`) — infinitely many points, so it holds as a
polynomial identity (`eq_zero_of_infinite_isRoot`). -/

/-- The QSP polynomials hoisted out of the per-`(x, s)` existential: one fixed
    `(A, B, C, D)` works for *all* `x, s` with `s² = 1 − x²`. -/
theorem qsp_polys (φs : List ℝ) :
    ∃ A B C D : Polynomial ℂ,
      ∀ x s : ℝ, s ^ 2 = 1 - x ^ 2 → qsp x s φs = qspForm4 A B C D x s := by
  induction φs with
  | nil => exact ⟨C 1, 0, 0, C 1, fun x s _ => one_eq x s⟩
  | cons φ rest ih =>
    obtain ⟨A, B, Cp, D, h⟩ := ih
    refine ⟨C (Complex.exp (I * (φ : ℂ))) * X * A
              + (1 - X ^ 2) * (C (Complex.exp (I * (φ : ℂ))) * C I) * Cp,
            C (Complex.exp (I * (φ : ℂ))) * X * B
              + C (Complex.exp (I * (φ : ℂ))) * C I * D,
            C (Complex.exp (-(I * (φ : ℂ)))) * C I * A
              + C (Complex.exp (-(I * (φ : ℂ)))) * X * Cp,
            (1 - X ^ 2) * (C (Complex.exp (-(I * (φ : ℂ)))) * C I) * B
              + C (Complex.exp (-(I * (φ : ℂ)))) * X * D,
            fun x s hs => ?_⟩
    rw [show qsp x s (φ :: rest) = phase φ * signal x s * qsp x s rest from rfl, h x s hs,
      phase_mul_signal φ x s, qspForm4_mul _ _ _ _ _ _ _ _ x s hs]

/-- **Unitarity as a polynomial identity (`(0,0)` Gram relation).** The QSP
    polynomials satisfy `A·A⋆ + (1−X²)·C·C⋆ = 1` in `ℂ[X]`, where `P⋆ = P.map conj`.
    This is the column-0 norm of the unitary `qsp`, promoted from "for each real
    `x ∈ [-1,1]`" to a polynomial identity via infinitely many roots. Comparing
    top coefficients here is what gives `‖lead A‖ = ‖lead C‖` — the modulus
    condition that makes the converse's cancelling angle exist. -/
theorem qsp_gram_diag (φs : List ℝ) :
    ∃ A B C D : Polynomial ℂ,
      (∀ x s : ℝ, s ^ 2 = 1 - x ^ 2 → qsp x s φs = qspForm4 A B C D x s) ∧
      A * A.map (starRingEnd ℂ) + (1 - X ^ 2) * (C * C.map (starRingEnd ℂ)) = 1 := by
  obtain ⟨A, B, Cp, D, h⟩ := qsp_polys φs
  refine ⟨A, B, Cp, D, h, ?_⟩
  set P : Polynomial ℂ :=
    A * A.map (starRingEnd ℂ) + (1 - X ^ 2) * (Cp * Cp.map (starRingEnd ℂ)) - 1 with hP
  have hroot : ∀ t : ℝ, t ∈ Set.Icc (-1 : ℝ) 1 → P.IsRoot (t : ℂ) := by
    intro t ht
    have hs : Real.sqrt (1 - t ^ 2) ^ 2 = 1 - t ^ 2 := by
      rw [Real.sq_sqrt]; nlinarith [ht.1, ht.2]
    have hsc : ((Real.sqrt (1 - t ^ 2) : ℝ) : ℂ) ^ 2 = 1 - (t : ℂ) ^ 2 := by
      rw [← Complex.ofReal_pow, hs]; push_cast; ring
    have hu := congrFun (congrFun (qsp_unitary t (Real.sqrt (1 - t ^ 2)) hs φs) 0) 0
    rw [h t _ hs] at hu
    simp only [Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_two, qspForm4,
      Matrix.cons_val_one, Matrix.of_apply, Matrix.cons_val', Matrix.empty_val',
      Matrix.cons_val_fin_one, Matrix.cons_val_zero, Matrix.one_apply_eq,
      ← starRingEnd_apply, map_mul, Complex.conj_ofReal] at hu
    have hAc : (A.map (starRingEnd ℂ)).eval (t : ℂ) = (starRingEnd ℂ) (A.eval (t : ℂ)) := by
      conv_lhs => rw [← Complex.conj_ofReal t]
      rw [eval_map_apply]
    have hCc : (Cp.map (starRingEnd ℂ)).eval (t : ℂ) = (starRingEnd ℂ) (Cp.eval (t : ℂ)) := by
      conv_lhs => rw [← Complex.conj_ofReal t]
      rw [eval_map_apply]
    simp only [Polynomial.IsRoot.def, hP, eval_sub, eval_add, eval_mul, eval_one, eval_pow,
      eval_X, hAc, hCc]
    linear_combination hu - (Cp.eval (t : ℂ) * (starRingEnd ℂ) (Cp.eval (t : ℂ))) * hsc
  have hsub : (Complex.ofReal '' Set.Icc (-1 : ℝ) 1) ⊆ {x | P.IsRoot x} := by
    rintro _ ⟨t, ht, rfl⟩; exact hroot t ht
  have hPzero : P = 0 :=
    eq_zero_of_infinite_isRoot _
      (((Set.Icc_infinite (by norm_num : (-1 : ℝ) < 1)).image
        Complex.ofReal_injective.injOn).mono hsub)
  rw [hP, sub_eq_zero] at hPzero
  exact hPzero

/-- **`‖lead A‖ = ‖lead C‖` from the Gram identity.** Comparing the top
    (`X²ᵏ⁺²`) coefficient of `A·A⋆ + (1−X²)·C·C⋆ = 1`: with `deg A ≤ k+1` and
    `deg C ≤ k`, the only surviving contributions are `‖A[k+1]‖²` (from `A·A⋆`) and
    `−‖C[k]‖²` (from the `−X²·C·C⋆` part), and they must sum to `0` (the RHS has no
    `X²ᵏ⁺²` term). Hence the candidate leading coefficients have equal modulus —
    exactly the condition that makes the converse's cancelling angle
    `e^{-iφ}·A[k+1] + i·e^{iφ}·C[k] = 0` solvable for a real `φ`. -/
theorem gram_lead_modulus (A Cp : Polynomial ℂ) (k : ℕ)
    (hA : A.natDegree ≤ k + 1) (hC : Cp.natDegree ≤ k)
    (hgram : A * A.map (starRingEnd ℂ) + (1 - X ^ 2) * (Cp * Cp.map (starRingEnd ℂ)) = 1) :
    ‖A.coeff (k + 1)‖ = ‖Cp.coeff k‖ := by
  have hAd : (A.map (starRingEnd ℂ)).natDegree ≤ k + 1 := natDegree_map_le.trans hA
  have hCd : (Cp.map (starRingEnd ℂ)).natDegree ≤ k := natDegree_map_le.trans hC
  have hQd : (Cp * Cp.map (starRingEnd ℂ)).natDegree ≤ k + k :=
    natDegree_mul_le.trans (add_le_add hC hCd)
  have hco := congrArg (fun p => Polynomial.coeff p (k + 1 + (k + 1))) hgram
  simp only [coeff_add, coeff_one, if_neg (by omega : ¬ k + 1 + (k + 1) = 0)] at hco
  rw [coeff_mul_add_eq_of_natDegree_le hA hAd, coeff_map,
    show (1 - X ^ 2) * (Cp * Cp.map (starRingEnd ℂ))
        = Cp * Cp.map (starRingEnd ℂ) - X ^ 2 * (Cp * Cp.map (starRingEnd ℂ)) from by ring,
    coeff_sub,
    coeff_eq_zero_of_natDegree_lt (hQd.trans_lt (by omega : k + k < k + 1 + (k + 1))),
    show k + 1 + (k + 1) = (k + k) + 2 from by ring, coeff_X_pow_mul,
    coeff_mul_add_eq_of_natDegree_le hC hCd, coeff_map] at hco
  have hkey : A.coeff (k + 1) * (starRingEnd ℂ) (A.coeff (k + 1))
            = Cp.coeff k * (starRingEnd ℂ) (Cp.coeff k) := by linear_combination hco
  rw [Complex.mul_conj, Complex.mul_conj] at hkey
  have hns : Complex.normSq (A.coeff (k + 1)) = Complex.normSq (Cp.coeff k) := by
    exact_mod_cast hkey
  rw [← Real.sqrt_sq (norm_nonneg (A.coeff (k + 1))),
    ← Real.sqrt_sq (norm_nonneg (Cp.coeff k)),
    ← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq, hns]

/-- Unit-modulus surjectivity of `φ ↦ e^{iφ}`: any `z` with `‖z‖ = 1` equals
    `e^{i·arg z}`. This is the angle that realizes a target leading coefficient. -/
private lemma expI_arg {z : ℂ} (hz : ‖z‖ = 1) :
    Complex.exp (I * (Complex.arg z : ℂ)) = z := by
  have h := Complex.norm_mul_exp_arg_mul_I z
  rw [hz, Complex.ofReal_one, one_mul] at h
  rw [mul_comm]; exact h

/-- **The cancelling angle exists.** Given equal-modulus candidate leading
    coefficients `‖a‖ = ‖c‖`, there is a real `φ` solving the cancellation
    equation `e^{-iφ}·a + i·e^{iφ}·c = 0` (the `Xᵏ⁺²` coefficient of the peeled
    diagonal, from `qspForm4_peel_top_coeff`, vanishes at this `φ`). The angle is
    `φ = arg(i·a/c)/2` (degenerate case `c = 0 ⇒ a = 0`, any `φ`). Combined with
    `gram_lead_modulus` (which supplies `‖a‖ = ‖c‖` from unitarity) this is the
    angle-selection rung of the converse induction. -/
theorem cancel_angle_exists (a c : ℂ) (h : ‖a‖ = ‖c‖) :
    ∃ φ : ℝ, Complex.exp (-(I * (φ : ℂ))) * a + I * Complex.exp (I * (φ : ℂ)) * c = 0 := by
  by_cases hc : c = 0
  · have ha : a = 0 := by rw [← norm_eq_zero, h, hc, norm_zero]
    exact ⟨0, by rw [ha, hc]; ring⟩
  · set z : ℂ := I * a / c with hz
    have hznorm : ‖z‖ = 1 := by
      rw [hz, norm_div, norm_mul, Complex.norm_I, one_mul, h, div_self (norm_ne_zero_iff.mpr hc)]
    refine ⟨Complex.arg z / 2, ?_⟩
    have he2 : Complex.exp (I * ((Complex.arg z / 2 : ℝ) : ℂ)) ^ 2 = z := by
      rw [← Complex.exp_nat_mul,
        show ((2 : ℕ) : ℂ) * (I * ((Complex.arg z / 2 : ℝ) : ℂ))
            = I * ((Complex.arg z : ℝ) : ℂ) from by push_cast; ring,
        expI_arg hznorm]
    have hene : Complex.exp (I * ((Complex.arg z / 2 : ℝ) : ℂ)) ≠ 0 := Complex.exp_ne_zero _
    rw [Complex.exp_neg,
      show (Complex.exp (I * ((Complex.arg z / 2 : ℝ) : ℂ)))⁻¹ * a
            + I * Complex.exp (I * ((Complex.arg z / 2 : ℝ) : ℂ)) * c
          = (Complex.exp (I * ((Complex.arg z / 2 : ℝ) : ℂ)))⁻¹
            * (a + I * Complex.exp (I * ((Complex.arg z / 2 : ℝ) : ℂ)) ^ 2 * c) from by
          field_simp,
      he2, hz]
    rw [mul_eq_zero]; right
    field_simp
    simp only [Complex.I_sq]; ring

/-! ### Converse — the explicit polynomial reduction (peel-and-reduce step)

The converse induction not only *peels* a factor (`qsp_peel`) but must track what
peeling does to the **polynomials**. `qspForm4_peel` makes that explicit: peeling
one `phase(-φ)·signalInv` off a QSP form `qspForm4 A B C D` yields another QSP
form whose entries are explicit polynomial combinations of `A, B, C, D` with the
`e^{±iφ}` and `1−X²` factors. From the closed form of the new diagonal entry,
`A' = X·(e^{-iφ}·A) + (1−X²)·(-i)·(e^{iφ}·C)`, the top-degree (`Xⁿ⁺¹`) coefficient
reads `e^{-iφ}·lead(A) + i·e^{iφ}·lead(C)`; the converse induction chooses `φ` to
*cancel* it, dropping the degree. Proving that cancellation forces `deg A' ≤ n−1`
(together with parity preservation and the unitarity identity giving
`‖lead A‖ = ‖lead C‖`) is the remaining deferred analytic content — but the
reduction's algebra is now pinned down exactly. -/

/-- **Explicit polynomial peel (reduction-step algebra).** Peeling one
    `phase(-φ)·signalInv` factor off a QSP form is again a QSP form, with the new
    polynomials given explicitly. This is the polynomial-level content of one step
    of the GSLW converse induction (cf. `qsp_peel` for the matrix-level unwind).
    The new diagonal `A'` has the `Xⁿ⁺¹` term `e^{-iφ}·lead(A) + i·e^{iφ}·lead(C)`
    that the induction cancels by its choice of `φ`. -/
lemma qspForm4_peel (φ x s : ℝ) (hs : s ^ 2 = 1 - x ^ 2) (pA pB pC pD : Polynomial ℂ) :
    signalInv x s * (phase (-φ) * qspForm4 pA pB pC pD x s) =
      qspForm4
        (X * (C (Complex.exp (-(I * (φ : ℂ)))) * pA)
          + (1 - X ^ 2) * C (-I) * (C (Complex.exp (I * (φ : ℂ))) * pC))
        (X * (C (Complex.exp (-(I * (φ : ℂ)))) * pB)
          + C (-I) * (C (Complex.exp (I * (φ : ℂ))) * pD))
        (C (-I) * (C (Complex.exp (-(I * (φ : ℂ)))) * pA)
          + X * (C (Complex.exp (I * (φ : ℂ))) * pC))
        ((1 - X ^ 2) * C (-I) * (C (Complex.exp (-(I * (φ : ℂ)))) * pB)
          + X * (C (Complex.exp (I * (φ : ℂ))) * pD))
        x s := by
  rw [phase_neg_eq φ x s, qspForm4_mul _ _ _ _ _ _ _ _ x s hs, signalInv_eq,
    qspForm4_mul _ _ _ _ _ _ _ _ x s hs]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [qspForm4, eval_add, eval_sub, eval_mul, eval_pow, eval_C, eval_X, eval_one]

/-- **Leading-coefficient of the peeled diagonal.** The top (`Xᵏ⁺²`) coefficient
    of the peeled diagonal polynomial `A' = X·(e⁻¹·pA) + (1−X²)·(-i)·(e·pC)` from
    [`qspForm4_peel`] is `e⁻¹·pA[k+1] + i·e·pC[k]`, provided `deg pC ≤ k`. For a
    length-`(k+1)` form (`deg pA ≤ k+1`, `deg pC ≤ k`) these are the leading
    coefficients of `pA, pC`; the GSLW converse induction chooses `φ` so that this
    sum vanishes, which (with parity killing the `Xᵏ⁺¹` coefficient) makes
    `deg A' ≤ k` — one rung of degree reduction. This pins the exact angle
    equation `e^{-iφ}·lead(pA) + i·e^{iφ}·lead(pC) = 0`. -/
lemma qspForm4_peel_top_coeff (φ : ℝ) (pA pC : Polynomial ℂ) (k : ℕ)
    (hC : pC.degree ≤ (k : WithBot ℕ)) :
    (X * (C (Complex.exp (-(I * (φ : ℂ)))) * pA)
        + (1 - X ^ 2) * C (-I) * (C (Complex.exp (I * (φ : ℂ))) * pC)).coeff (k + 2)
      = Complex.exp (-(I * (φ : ℂ))) * pA.coeff (k + 1)
        + I * Complex.exp (I * (φ : ℂ)) * pC.coeff k := by
  have hz : pC.coeff (k + 2) = 0 :=
    coeff_eq_zero_of_degree_lt
      (lt_of_le_of_lt hC (by exact_mod_cast Nat.lt_add_of_pos_right (by norm_num)))
  have h1 : (X * (C (Complex.exp (-(I * (φ : ℂ)))) * pA)).coeff (k + 2)
      = Complex.exp (-(I * (φ : ℂ))) * pA.coeff (k + 1) := by
    rw [show k + 2 = (k + 1) + 1 from rfl, coeff_X_mul, coeff_C_mul]
  have h2 : (X ^ 2 * (C (-I) * (C (Complex.exp (I * (φ : ℂ))) * pC))).coeff (k + 2)
      = -I * (Complex.exp (I * (φ : ℂ)) * pC.coeff k) := by
    rw [coeff_X_pow_mul, coeff_C_mul, coeff_C_mul]
  rw [show (X * (C (Complex.exp (-(I * (φ : ℂ)))) * pA)
        + (1 - X ^ 2) * C (-I) * (C (Complex.exp (I * (φ : ℂ))) * pC))
      = X * (C (Complex.exp (-(I * (φ : ℂ)))) * pA)
        + ((C (-I) * (C (Complex.exp (I * (φ : ℂ))) * pC))
          - X ^ 2 * (C (-I) * (C (Complex.exp (I * (φ : ℂ))) * pC))) from by ring,
    coeff_add, coeff_sub, h1, h2, coeff_C_mul, coeff_C_mul, hz]
  ring

/-! ### Converse — explicit low-degree realizations (induction base)

The GSLW converse induction (peel one factor via `qsp_peel`, reduce the degree,
recurse) bottoms out at degree `0`, `1`, `2`, which we discharge *constructively*
here — no completion argument needed at these rungs. Together with the peeling
step these anchor the converse induction; only the general inductive angle-choice
(leading-coefficient) lemma between them stays deferred. -/

/-- Closed form of the degree-1 QSP `(0,0)` entry: a single phase `φ` realizes
    the linear map `x ↦ e^{iφ}·x`. -/
lemma qsp_singleton_apply (φ x s : ℝ) :
    (qsp x s [φ]) 0 0 = Complex.exp (I * (φ : ℂ)) * (x : ℂ) := by
  rw [show qsp x s [φ] = phase φ * signal x s * qsp x s [] from rfl,
    show qsp x s [] = 1 from rfl, mul_one, phase_mul_signal]
  simp [qspForm4, eval_mul, eval_C, eval_X]

/-- Closed form of the degree-2 QSP `(0,0)` entry: two phases `φ₀, φ₁` realize
    `x ↦ e^{i(φ₀+φ₁)}·x² − (1−x²)·e^{i(φ₀−φ₁)}`. -/
lemma qsp_pair_apply (φ₀ φ₁ x s : ℝ) (hs : s ^ 2 = 1 - x ^ 2) :
    (qsp x s [φ₀, φ₁]) 0 0 =
      Complex.exp (I * (φ₀ : ℂ)) * Complex.exp (I * (φ₁ : ℂ)) * (x : ℂ) ^ 2
        - (1 - (x : ℂ) ^ 2)
          * (Complex.exp (I * (φ₀ : ℂ)) * Complex.exp (-(I * (φ₁ : ℂ)))) := by
  rw [show qsp x s [φ₀, φ₁]
        = phase φ₀ * signal x s * (phase φ₁ * signal x s * qsp x s []) from rfl,
    show qsp x s ([] : List ℝ) = 1 from rfl, mul_one, phase_mul_signal φ₀ x s,
    phase_mul_signal φ₁ x s, qspForm4_mul _ _ _ _ _ _ _ _ x s hs]
  simp only [qspForm4, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
    Matrix.empty_val', Matrix.cons_val_fin_one, eval_add, eval_sub, eval_mul, eval_C, eval_X,
    eval_pow, eval_one]
  linear_combination
    (1 - (x : ℂ) ^ 2) * (Complex.exp (I * (φ₀ : ℂ)) * Complex.exp (-(I * (φ₁ : ℂ))))
      * Complex.I_sq

/-- **Converse, constant case (`d = 0`).** The constant polynomial `P = 1` is the
    `(0,0)` entry of the empty phase list — the degree-0 base of the converse. -/
lemma qsp_converse_const (x s : ℝ) :
    (qsp x s []) 0 0 = (1 : Polynomial ℂ).eval (x : ℂ) := by
  simp [show qsp x s [] = 1 from rfl]

/-- **Converse, linear case (`d = 1`).** Every unit-modulus leading coefficient is
    achievable: for `c` with `‖c‖ = 1`, the polynomial `P = C c · X` (the map
    `x ↦ c·x`) is realized by the single-phase list `[arg c]`. This is the
    degree-1 base of the GSLW converse induction — given the admissible target,
    the angle is read off as `arg c` (here the leading coefficient *is* `c`). -/
lemma qsp_converse_linear (c : ℂ) (hc : ‖c‖ = 1) :
    ∃ φ : ℝ, ∀ x s : ℝ, (qsp x s [φ]) 0 0 = (C c * X).eval (x : ℂ) := by
  refine ⟨Complex.arg c, fun x s => ?_⟩
  rw [qsp_singleton_apply, expI_arg hc, eval_mul, eval_C, eval_X]

/-- **Converse, degree-2 case.** Every pair of unit-modulus coefficients
    `(a, b)`, `‖a‖ = ‖b‖ = 1`, is achievable: the degree-2 polynomial
    `P = C a · X² − (1 − X²) · C b` is the `(0,0)` entry of a 2-phase list, with
    the angles read off the target as `φ₁ = arg(a/b)/2`, `φ₀ = arg b + φ₁`
    (so `e^{i(φ₀+φ₁)} = a` and `e^{i(φ₀−φ₁)} = b`). The constructive degree-2
    rung of the converse induction. -/
lemma qsp_converse_deg2 (a b : ℂ) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
    ∃ φ₀ φ₁ : ℝ, ∀ x s : ℝ, s ^ 2 = 1 - x ^ 2 →
      (qsp x s [φ₀, φ₁]) 0 0 = (C a * X ^ 2 - (1 - X ^ 2) * C b).eval (x : ℂ) := by
  have hb0 : b ≠ 0 := by rw [← norm_ne_zero_iff, hb]; norm_num
  have hab : ‖a / b‖ = 1 := by rw [norm_div, ha, hb, div_one]
  refine ⟨Complex.arg b + Complex.arg (a / b) / 2, Complex.arg (a / b) / 2,
    fun x s hs => ?_⟩
  set φ₁ : ℝ := Complex.arg (a / b) / 2 with hφ₁
  set φ₀ : ℝ := Complex.arg b + φ₁ with hφ₀
  have key1 : Complex.exp (I * (φ₀ : ℂ)) * Complex.exp (-(I * (φ₁ : ℂ))) = b := by
    rw [← Complex.exp_add]
    have e : I * (φ₀ : ℂ) + -(I * (φ₁ : ℂ)) = I * (Complex.arg b : ℂ) := by
      rw [hφ₀]; push_cast; ring
    rw [e, expI_arg hb]
  have key0 : Complex.exp (I * (φ₀ : ℂ)) * Complex.exp (I * (φ₁ : ℂ)) = a := by
    rw [← Complex.exp_add]
    have e : I * (φ₀ : ℂ) + I * (φ₁ : ℂ)
        = I * (Complex.arg b : ℂ) + I * (Complex.arg (a / b) : ℂ) := by
      rw [hφ₀, hφ₁]; push_cast; ring
    rw [e, Complex.exp_add, expI_arg hb, expI_arg hab, mul_div_cancel₀ _ hb0]
  rw [qsp_pair_apply φ₀ φ₁ x s hs, key0, key1, eval_sub, eval_mul, eval_mul, eval_C,
    eval_pow, eval_X, eval_sub, eval_one, eval_pow, eval_X, eval_C]

/-! ### Parity — the QSP polynomials have definite parity

The third structural pillar of the forward QSP fundamental theorem: for a
length-`d` phase list the diagonal polynomials `A, D` have parity `d` (each is an
even function of `x` iff `d` is even) and the off-diagonal numerators `B, C` have
the *opposite* parity `d+1`. We encode "`P` has parity `p`" as the substitution
identity `P.comp(-X) = (-1)^p · P` (replacing `x ↦ -x` flips the sign by exactly
`(-1)^p`). This composes directly through the induction: `comp(-X)` distributes
over `+` and `*`, the signal factor `C e · X` flips sign, and the `(1−X²)` factor
is even — so every `phase·signal` step bumps the parity by one. -/

/-- **Forward QSP — parity budget.** For a phase list of length `d`, the QSP
    unitary is `qspForm4 A B C D` where the diagonal polynomials `A, D` have
    parity `d` and the off-diagonal numerators `B, C` have parity `d+1`, encoded
    as `P.comp(-X) = (-1)^p · P`. Proven by the same induction as the degree
    budget: each `phase·signal` step ([`phase_mul_signal`]) multiplies the
    running form ([`qspForm4_mul`]), and `comp(-X)` flips the diagonal/off-diagonal
    signs by exactly one. -/
theorem qsp_parity (x s : ℝ) (hs : s ^ 2 = 1 - x ^ 2) (φs : List ℝ) :
    ∃ A B C D : Polynomial ℂ,
      qsp x s φs = qspForm4 A B C D x s ∧
      A.comp (-X) = (-1) ^ φs.length * A ∧
      D.comp (-X) = (-1) ^ φs.length * D ∧
      B.comp (-X) = (-1) ^ (φs.length + 1) * B ∧
      C.comp (-X) = (-1) ^ (φs.length + 1) * C := by
  induction φs with
  | nil =>
    refine ⟨C 1, 0, 0, C 1, one_eq x s, ?_, ?_, ?_, ?_⟩ <;> simp
  | cons φ rest ih =>
    obtain ⟨A₀, B₀, C₀, D₀, heq, hA, hD, hB, hC⟩ := ih
    have key : qsp x s (φ :: rest) =
        qspForm4
          (C (Complex.exp (I * (φ : ℂ))) * X * A₀
            + (1 - X ^ 2) * (C (Complex.exp (I * (φ : ℂ))) * C I) * C₀)
          (C (Complex.exp (I * (φ : ℂ))) * X * B₀
            + C (Complex.exp (I * (φ : ℂ))) * C I * D₀)
          (C (Complex.exp (-(I * (φ : ℂ)))) * C I * A₀
            + C (Complex.exp (-(I * (φ : ℂ)))) * X * C₀)
          ((1 - X ^ 2) * (C (Complex.exp (-(I * (φ : ℂ)))) * C I) * B₀
            + C (Complex.exp (-(I * (φ : ℂ)))) * X * D₀) x s := by
      rw [show qsp x s (φ :: rest) = phase φ * signal x s * qsp x s rest from rfl, heq,
        phase_mul_signal, qspForm4_mul _ _ _ _ _ _ _ _ x s hs]
    refine ⟨_, _, _, _, key, ?_, ?_, ?_, ?_⟩ <;>
      simp only [List.length_cons, add_comp, sub_comp, mul_comp, pow_comp, X_comp, C_comp,
        one_comp, hA, hB, hC, hD] <;> ring

/-- Coefficient of the reflected polynomial: `(P(−X))[n] = (−1)ⁿ·P[n]`. -/
theorem coeff_comp_neg_X (P : Polynomial ℂ) (n : ℕ) :
    (P.comp (-X)).coeff n = (-1) ^ n * P.coeff n := by
  induction P using Polynomial.induction_on' with
  | add p q hp hq => simp only [add_comp, coeff_add, hp, hq, mul_add]
  | monomial m a =>
    rw [monomial_comp, neg_eq_neg_one_mul (X : Polynomial ℂ), mul_pow, ← C_1, ← C_neg, ← C_pow,
      coeff_C_mul, coeff_monomial, coeff_C_mul, coeff_X_pow]
    rcases eq_or_ne n m with rfl | h
    · simp; ring
    · simp [h, Ne.symm h]

/-- **Parity ⇒ coefficient vanishing.** A polynomial with parity `p` (in the
    `comp(-X) = (-1)^p·P` sense of [`qsp_parity`]) has *zero* coefficient at every
    index of the opposite parity (`n + p` odd). This is the bridge that lets the
    converse degree-drop kill the `Xᵏ⁺¹` coefficient of the peeled diagonal from
    the parity budget. -/
theorem parity_coeff_zero (P : Polynomial ℂ) (p n : ℕ)
    (hP : P.comp (-X) = (-1) ^ p * P) (hpar : Odd (n + p)) : P.coeff n = 0 := by
  have hCp : (C ((-1) ^ p) : Polynomial ℂ) = (-1) ^ p := by rw [map_pow, map_neg, map_one]
  have hcoe := congrArg (fun q => Polynomial.coeff q n) hP
  simp only [coeff_comp_neg_X] at hcoe
  rw [← hCp, coeff_C_mul] at hcoe
  have hfac : ((-1 : ℂ) ^ n - (-1) ^ p) * P.coeff n = 0 := by linear_combination hcoe
  have hne : (-1 : ℂ) ^ n - (-1) ^ p ≠ 0 := by
    intro hzero
    have heq : (-1 : ℂ) ^ n = (-1) ^ p := sub_eq_zero.mp hzero
    have hprod : (-1 : ℂ) ^ n * (-1) ^ p = -1 := by rw [← pow_add]; exact Odd.neg_one_pow hpar
    rw [heq, ← pow_add, ← two_mul, pow_mul, neg_one_sq, one_pow] at hprod
    norm_num at hprod
  exact (mul_eq_zero.mp hfac).resolve_left hne

/-! ### Converse — degree-drop rung assembly

With the four key lemmas in hand (Gram identity → modulus equality; cancelling
angle existence; parity → coefficient vanishing), the *assembly* of one rung of
the GSLW converse degree reduction is now finitary polynomial algebra. We bundle
it here for the A-component (`peelA`), the only component whose degree must drop
to continue the induction. -/

/-- A polynomial of degree `≤ k+2` whose two top coefficients (`Xᵏ⁺¹`, `Xᵏ⁺²`)
    both vanish has degree `≤ k`. The elementary degree-collapse behind one rung
    of the converse induction. -/
theorem natDegree_drop_two (p : Polynomial ℂ) (k : ℕ)
    (hdeg : p.natDegree ≤ k + 2) (h1 : p.coeff (k + 1) = 0) (h2 : p.coeff (k + 2) = 0) :
    p.natDegree ≤ k := by
  rw [natDegree_le_iff_coeff_eq_zero]
  intro m hm
  rcases lt_or_ge (k + 2) m with hlt | hle
  · exact coeff_eq_zero_of_natDegree_lt (lt_of_le_of_lt hdeg hlt)
  · have : m = k + 1 ∨ m = k + 2 := by omega
    rcases this with h | h <;> subst h <;> assumption

/-- **A-component of one GSLW peel step.** This is the first polynomial produced
    by [`qspForm4_peel`]; stripping a `phase φ · signal` factor (via their
    inverses) from an admissible form sends its diagonal `A` to `peelA φ A C`. -/
noncomputable def peelA (φ : ℝ) (A Cp : Polynomial ℂ) : Polynomial ℂ :=
  X * (C (Complex.exp (-(I * (φ : ℂ)))) * A)
    + (1 - X ^ 2) * C (-I) * (C (Complex.exp (I * (φ : ℂ))) * Cp)

/-- A-priori degree bound on the peeled diagonal: from `deg A ≤ k+2`,
    `deg Cp ≤ k+1` one gets `deg (peelA φ A Cp) ≤ k+3` (the extra `X` and
    `1 − X²` factors raise the budget by one and two respectively). -/
theorem peelA_natDegree_le (φ : ℝ) (A Cp : Polynomial ℂ) (k : ℕ)
    (hA : A.natDegree ≤ k + 2) (hC : Cp.natDegree ≤ k + 1) :
    (peelA φ A Cp).natDegree ≤ k + 3 := by
  unfold peelA; compute_degree <;> omega

/-- **Parity of the peeled diagonal.** If `A` carries parity `k+2` and `Cp`
    parity `k+1` (the diagonal/off-diagonal parities of a degree-`(k+2)`
    admissible form), then `peelA φ A Cp` carries parity `k+3`. The `X` factor
    flips `A`'s parity and the even `1 − X²` preserves `Cp`'s, and the two agree
    mod 2 — so the result has a definite parity. -/
theorem peelA_parity (φ : ℝ) (A Cp : Polynomial ℂ) (k : ℕ)
    (hpA : A.comp (-X) = (-1) ^ (k + 2) * A)
    (hpC : Cp.comp (-X) = (-1) ^ (k + 1) * Cp) :
    (peelA φ A Cp).comp (-X) = (-1) ^ (k + 3) * (peelA φ A Cp) := by
  unfold peelA
  simp only [add_comp, mul_comp, sub_comp, pow_comp, X_comp, C_comp, one_comp, hpA, hpC]
  ring

/-- The top (`Xᵏ⁺³`) coefficient of the peeled diagonal is the cancellation sum
    `e^{-iφ}·A[k+2] + i·e^{iφ}·Cp[k+1]` — a reindexing of
    [`qspForm4_peel_top_coeff`] to `peelA`. -/
theorem peelA_coeff_top (φ : ℝ) (A Cp : Polynomial ℂ) (k : ℕ)
    (hC : Cp.degree ≤ (k + 1 : WithBot ℕ)) :
    (peelA φ A Cp).coeff (k + 3)
      = Complex.exp (-(I * (φ : ℂ))) * A.coeff (k + 2)
        + I * Complex.exp (I * (φ : ℂ)) * Cp.coeff (k + 1) := by
  have := qspForm4_peel_top_coeff φ A Cp (k + 1) hC
  simpa only [peelA, show (k + 1) + 2 = k + 3 from rfl, show (k + 1) + 1 = k + 2 from rfl]
    using this

/-- **Degree-drop rung (A-component) — the assembly.** Given an admissible
    degree-`(k+2)` form — degree budget (`deg A ≤ k+2`, `deg Cp ≤ k+1`), matching
    parities, and the leading-modulus equality `‖A[k+2]‖ = ‖Cp[k+1]‖` that
    unitarity supplies (`gram_lead_modulus`) — there is a phase `φ` whose peel
    drops the A-component degree to `≤ k+1`. This is the row the converse
    induction had deferred: cancellation (`cancel_angle_exists`) kills the top
    `Xᵏ⁺³` coefficient, parity (`parity_coeff_zero`) kills the middle `Xᵏ⁺²`
    coefficient, and `natDegree_drop_two` collapses the degree by one rung. -/
theorem peel_degree_drop (A Cp : Polynomial ℂ) (k : ℕ)
    (hA : A.natDegree ≤ k + 2) (hC : Cp.natDegree ≤ k + 1)
    (hpA : A.comp (-X) = (-1) ^ (k + 2) * A)
    (hpC : Cp.comp (-X) = (-1) ^ (k + 1) * Cp)
    (hmod : ‖A.coeff (k + 2)‖ = ‖Cp.coeff (k + 1)‖) :
    ∃ φ : ℝ, (peelA φ A Cp).natDegree ≤ k + 1 := by
  obtain ⟨φ, hφ⟩ := cancel_angle_exists (A.coeff (k + 2)) (Cp.coeff (k + 1)) hmod
  refine ⟨φ, ?_⟩
  have hCdeg : Cp.degree ≤ (k + 1 : WithBot ℕ) := by
    refine le_trans degree_le_natDegree ?_
    exact_mod_cast hC
  have htop : (peelA φ A Cp).coeff (k + 3) = 0 := by
    rw [peelA_coeff_top φ A Cp k hCdeg]; exact hφ
  have hmid : (peelA φ A Cp).coeff (k + 2) = 0 :=
    parity_coeff_zero _ (k + 3) (k + 2) (peelA_parity φ A Cp k hpA hpC) ⟨k + 2, by ring⟩
  have hAdeg : (peelA φ A Cp).natDegree ≤ k + 3 := peelA_natDegree_le φ A Cp k hA hC
  have hfin := natDegree_drop_two (peelA φ A Cp) (k + 1)
    (by omega) (by simpa using hmid) (by simpa using htop)
  simpa using hfin

/-! ### Converse — first-column degree drop (one angle for both `A` and `C`)

The degree-drop rung above reduces the diagonal `A`. The decisive GSLW fact is
that the *same* phase `φ` — chosen once, from the first column's leading moduli —
also reduces the off-diagonal `C`. So one peel step sends the whole first column
`(A, C)` of a degree-`(k+2)` admissible form to `(A', C')` of degree
`(≤ k+1, ≤ k)`, the budget of a degree-`(k+1)` form. This is the column the
strong induction actually carries. -/

/-- **C-component (off-diagonal, first column) of one GSLW peel step.** This is
    the third polynomial produced by [`qspForm4_peel`]. -/
noncomputable def peelC (φ : ℝ) (A Cp : Polynomial ℂ) : Polynomial ℂ :=
  C (-I) * (C (Complex.exp (-(I * (φ : ℂ)))) * A)
    + X * (C (Complex.exp (I * (φ : ℂ))) * Cp)

/-- A-priori degree bound on the peeled off-diagonal: `deg (peelC φ A Cp) ≤ k+2`
    from `deg A ≤ k+2`, `deg Cp ≤ k+1`. -/
theorem peelC_natDegree_le (φ : ℝ) (A Cp : Polynomial ℂ) (k : ℕ)
    (hA : A.natDegree ≤ k + 2) (hC : Cp.natDegree ≤ k + 1) :
    (peelC φ A Cp).natDegree ≤ k + 2 := by
  unfold peelC; compute_degree <;> omega

/-- **Parity of the peeled off-diagonal.** From `A` parity `k+2`, `Cp` parity
    `k+1`, the peeled `peelC` carries parity `k+2` (the constant `-i` preserves
    `A`'s parity, the `X` factor flips `Cp`'s, and they agree mod 2). -/
theorem peelC_parity (φ : ℝ) (A Cp : Polynomial ℂ) (k : ℕ)
    (hpA : A.comp (-X) = (-1) ^ (k + 2) * A)
    (hpC : Cp.comp (-X) = (-1) ^ (k + 1) * Cp) :
    (peelC φ A Cp).comp (-X) = (-1) ^ (k + 2) * (peelC φ A Cp) := by
  unfold peelC
  simp only [add_comp, mul_comp, X_comp, C_comp, hpA, hpC]
  ring

/-- The top (`Xᵏ⁺²`) coefficient of the peeled off-diagonal is
    `-i·e^{-iφ}·A[k+2] + e^{iφ}·Cp[k+1]`. -/
theorem peelC_coeff_top (φ : ℝ) (A Cp : Polynomial ℂ) (k : ℕ) :
    (peelC φ A Cp).coeff (k + 2)
      = -I * (Complex.exp (-(I * (φ : ℂ))) * A.coeff (k + 2))
        + Complex.exp (I * (φ : ℂ)) * Cp.coeff (k + 1) := by
  unfold peelC
  rw [coeff_add, coeff_C_mul, coeff_C_mul, show k + 2 = (k + 1) + 1 from rfl,
    coeff_X_mul, coeff_C_mul]

/-- **First-column degree drop — one angle reduces both `A` and `C`.** The single
    cancelling phase `φ` (chosen from the first column's leading moduli) drops the
    diagonal `A` to degree `≤ k+1` *and* the off-diagonal `C` to degree `≤ k` at
    once. This is the GSLW key fact: the angle equation
    `e^{-iφ}·A[k+2] = -i·e^{iφ}·C[k+1]` makes `peelC`'s top coefficient
    `-i·e^{-iφ}·A[k+2] + e^{iφ}·C[k+1]` collapse to `0`, so the same `φ` that
    kills `A`'s leading term automatically kills `C`'s. One peel step thus carries
    the first column of a degree-`(k+2)` admissible form to the first column of a
    degree-`(k+1)` form. -/
theorem peel_first_column_drop (A Cp : Polynomial ℂ) (k : ℕ)
    (hA : A.natDegree ≤ k + 2) (hC : Cp.natDegree ≤ k + 1)
    (hpA : A.comp (-X) = (-1) ^ (k + 2) * A)
    (hpC : Cp.comp (-X) = (-1) ^ (k + 1) * Cp)
    (hmod : ‖A.coeff (k + 2)‖ = ‖Cp.coeff (k + 1)‖) :
    ∃ φ : ℝ, (peelA φ A Cp).natDegree ≤ k + 1 ∧ (peelC φ A Cp).natDegree ≤ k := by
  obtain ⟨φ, hφ⟩ := cancel_angle_exists (A.coeff (k + 2)) (Cp.coeff (k + 1)) hmod
  have hCdeg : Cp.degree ≤ (k + 1 : WithBot ℕ) :=
    le_trans degree_le_natDegree (by exact_mod_cast hC)
  refine ⟨φ, ?_, ?_⟩
  · have htop : (peelA φ A Cp).coeff (k + 3) = 0 := by
      rw [peelA_coeff_top φ A Cp k hCdeg]; exact hφ
    have hmid : (peelA φ A Cp).coeff (k + 2) = 0 :=
      parity_coeff_zero _ (k + 3) (k + 2) (peelA_parity φ A Cp k hpA hpC) ⟨k + 2, by ring⟩
    have hfin := natDegree_drop_two (peelA φ A Cp) (k + 1)
      (by have := peelA_natDegree_le φ A Cp k hA hC; omega)
      (by simpa using hmid) (by simpa using htop)
    simpa using hfin
  · have htop : (peelC φ A Cp).coeff (k + 2) = 0 := by
      rw [peelC_coeff_top φ A Cp k]
      linear_combination (-I) * hφ
        + (Complex.exp (I * (φ : ℂ)) * Cp.coeff (k + 1)) * Complex.I_sq
    have hmid : (peelC φ A Cp).coeff (k + 1) = 0 :=
      parity_coeff_zero _ (k + 2) (k + 1) (peelC_parity φ A Cp k hpA hpC) ⟨k + 1, by ring⟩
    exact natDegree_drop_two (peelC φ A Cp) k (peelC_natDegree_le φ A Cp k hA hC) hmid htop

/-! ### Converse — unitarity preserved under peel, and the realization recursion

Two more pillars the strong induction needs. First, the Gram (unitarity) relation
`A·A⋆ + (1−X²)·C·C⋆ = 1` is *preserved* by one peel step — so the leading-modulus
hypothesis of `peel_first_column_drop` stays available at every rung. Second, the
realization recursion: if a phase list realizes the first column of the peeled
pair, prepending the chosen phase realizes the first column of the original. -/

/-- **Unitarity preserved under peel.** The first-column Gram form
    `A·A⋆ + (1−X²)·C·C⋆` is invariant under one GSLW peel step:
    `peelA·peelA⋆ + (1−X²)·peelC·peelC⋆ = A·A⋆ + (1−X²)·C·C⋆`. This is an exact
    polynomial identity — the cross terms cancel and the diagonal terms recombine
    through `e^{iφ}·e^{−iφ} = 1` and `C(i)² = −1` (the scalar factor
    `(e·e⁻¹)·(X² − (1−X²)·C(i)²)` collapses to `1` since `X² + (1−X²) = 1`). Hence
    if `A,C` satisfy the Gram identity `= 1`, so do `peelA,peelC`: the peeled pair
    is again admissible, and `gram_lead_modulus` keeps applying down the induction. -/
theorem gram_peel_invariant (φ : ℝ) (A Cp : Polynomial ℂ) :
    peelA φ A Cp * (peelA φ A Cp).map (starRingEnd ℂ)
      + (1 - X ^ 2) * (peelC φ A Cp * (peelC φ A Cp).map (starRingEnd ℂ))
    = A * A.map (starRingEnd ℂ) + (1 - X ^ 2) * (Cp * Cp.map (starRingEnd ℂ)) := by
  have hab : (C (Complex.exp (I * (φ : ℂ))) * C (Complex.exp (-(I * (φ : ℂ)))) : ℂ[X]) = 1 := by
    rw [← C_mul, ← Complex.exp_add]; simp
  have hcI : (C I * C I : ℂ[X]) = -1 := by
    rw [← C_mul, Complex.I_mul_I, map_neg, map_one]
  calc peelA φ A Cp * (peelA φ A Cp).map (starRingEnd ℂ)
        + (1 - X ^ 2) * (peelC φ A Cp * (peelC φ A Cp).map (starRingEnd ℂ))
      = (C (Complex.exp (I * (φ : ℂ))) * C (Complex.exp (-(I * (φ : ℂ)))))
          * (X ^ 2 - (1 - X ^ 2) * (C I * C I))
          * (A * A.map (starRingEnd ℂ) + (1 - X ^ 2) * (Cp * Cp.map (starRingEnd ℂ))) := by
        simp only [peelA, peelC, Polynomial.map_add, Polynomial.map_mul, Polynomial.map_sub,
          Polynomial.map_one, Polynomial.map_pow, Polynomial.map_X, Polynomial.map_C,
          Polynomial.map_neg, ← Complex.exp_conj, map_mul, map_neg, Complex.conj_I,
          Complex.conj_ofReal]
        ring_nf
    _ = 1 * (X ^ 2 - (1 - X ^ 2) * (-1))
          * (A * A.map (starRingEnd ℂ) + (1 - X ^ 2) * (Cp * Cp.map (starRingEnd ℂ))) := by
        rw [hab, hcI]
    _ = A * A.map (starRingEnd ℂ) + (1 - X ^ 2) * (Cp * Cp.map (starRingEnd ℂ)) := by ring

/-- **First-column realization.** A phase list `φs` *realizes* the pair `(A, Cp)`
    when, for every admissible `(x, s)` (`s² = 1 − x²`), the first column of
    `qsp x s φs` is `(A(x), s·Cp(x))` — i.e. its `(0,0)` entry is `A(x)` and its
    `(1,0)` entry is `s·Cp(x)`. This is the data the converse induction carries. -/
def Realizes (φs : List ℝ) (A Cp : Polynomial ℂ) : Prop :=
  ∀ x s : ℝ, s ^ 2 = 1 - x ^ 2 →
    (qsp x s φs) 0 0 = A.eval (x : ℂ) ∧ (qsp x s φs) 1 0 = (s : ℂ) * Cp.eval (x : ℂ)

/-- **Realization recursion (peel back one phase).** If `φs` realizes the peeled
    first column `(peelA φ A Cp, peelC φ A Cp)`, then prepending `φ` realizes the
    original `(A, Cp)`. Proof: `qsp x s (φ::φs) = phase φ · signal x s · qsp x s φs`,
    so its first column is `phase φ · signal x s` applied to the realized column
    `(peelA(x), s·peelC(x))`; expanding and using `e^{iφ}e^{−iφ}=1`, `i·(−i)=1`,
    `s²=1−x²` collapses it to `(A(x), s·Cp(x))`. This is the inductive step that
    converts a degree-`(k+1)` realization into a degree-`(k+2)` one. -/
theorem realize_step (φ : ℝ) (φs : List ℝ) (A Cp : Polynomial ℂ)
    (h : Realizes φs (peelA φ A Cp) (peelC φ A Cp)) :
    Realizes (φ :: φs) A Cp := by
  intro x s hs
  obtain ⟨hA, hC⟩ := h x s hs
  have hsc : (s : ℂ) ^ 2 = 1 - (x : ℂ) ^ 2 := by exact_mod_cast hs
  have hexp : Complex.exp (I * (φ : ℂ)) * Complex.exp (-(I * (φ : ℂ))) = 1 := by
    rw [← Complex.exp_add]; simp
  have hqsp : qsp x s (φ :: φs) = phase φ * signal x s * qsp x s φs := rfl
  have e00 : (phase φ * signal x s * qsp x s φs) 0 0
      = (phase φ * signal x s) 0 0 * (qsp x s φs) 0 0
        + (phase φ * signal x s) 0 1 * (qsp x s φs) 1 0 := by
    rw [mul_apply]; simp [Fin.sum_univ_two]
  have e10 : (phase φ * signal x s * qsp x s φs) 1 0
      = (phase φ * signal x s) 1 0 * (qsp x s φs) 0 0
        + (phase φ * signal x s) 1 1 * (qsp x s φs) 1 0 := by
    rw [mul_apply]; simp [Fin.sum_univ_two]
  refine ⟨?_, ?_⟩
  · rw [hqsp, e00, hA, hC]
    simp only [phase, signal, mul_apply, Fin.sum_univ_two, of_apply, cons_val', cons_val_zero,
      cons_val_one, head_cons, empty_val', cons_val_fin_one, diagonal, Matrix.diagonal_apply,
      peelA, peelC, eval_add, eval_mul, eval_sub, eval_pow, eval_C, eval_X, eval_one]
    ring_nf
    linear_combination (A.eval (x : ℂ)) * hexp
      + (I * Complex.exp (I * (φ : ℂ)) ^ 2 * x * Cp.eval (x : ℂ)
         - I ^ 2 * Complex.exp (I * (φ : ℂ)) * Complex.exp (-(I * (φ : ℂ))) * A.eval (x : ℂ)) * hsc
      - (A.eval (x : ℂ)) * Complex.exp (I * (φ : ℂ)) * Complex.exp (-(I * (φ : ℂ)))
          * (1 - (x : ℂ) ^ 2) * Complex.I_sq
  · rw [hqsp, e10, hA, hC]
    simp only [phase, signal, mul_apply, Fin.sum_univ_two, of_apply, cons_val', cons_val_zero,
      cons_val_one, head_cons, empty_val', cons_val_fin_one, diagonal, Matrix.diagonal_apply,
      peelA, peelC, eval_add, eval_mul, eval_sub, eval_pow, eval_C, eval_X, eval_one]
    ring_nf
    linear_combination (s : ℂ) * Cp.eval (x : ℂ) * hexp
      - (s : ℂ) * Cp.eval (x : ℂ) * (1 - (x : ℂ) ^ 2)
          * Complex.exp (I * (φ : ℂ)) * Complex.exp (-(I * (φ : ℂ))) * Complex.I_sq

/-! ### Converse — global-phase scaling (the `SL₂` obstruction)

Both generators have determinant `1` (`det (phase φ) = 1`, `det (signal x s) = 1`
since `x² + s² = 1`), so every `qsp` product lies in `SL₂(ℂ)`. A bare global phase
`c₀·I` has determinant `c₀²`, hence is *not* realizable for general `c₀`: phase
gates scale the two components oppositely and cannot implement a global scalar.
The honest converse therefore realizes the admissible pair **up to a unit global
phase** `c₀` — equivalently, exactly for the determinant-normalized subfamily. The
following lemmas thread that `c₀` through the peel: `peelA`/`peelC` are linear, so
scaling `(A, Cp)` by `C c₀` scales the peel by `C c₀`, and (for `‖c₀‖ = 1`) the
Gram form is unchanged. -/

/-- `peelA` is linear: scaling the input pair by `C c₀` scales the output. -/
theorem peelA_smul (φ : ℝ) (c₀ : ℂ) (A Cp : Polynomial ℂ) :
    peelA φ (C c₀ * A) (C c₀ * Cp) = C c₀ * peelA φ A Cp := by
  unfold peelA; ring

/-- `peelC` is linear: scaling the input pair by `C c₀` scales the output. -/
theorem peelC_smul (φ : ℝ) (c₀ : ℂ) (A Cp : Polynomial ℂ) :
    peelC φ (C c₀ * A) (C c₀ * Cp) = C c₀ * peelC φ A Cp := by
  unfold peelC; ring

/-- Scaling an admissible pair by a unit global phase `C c₀` (`‖c₀‖ = 1`) leaves
    the Gram form `A·A⋆ + (1−X²)·C·C⋆` unchanged (`c₀·conj c₀ = ‖c₀‖² = 1`). -/
theorem gram_smul (c₀ : ℂ) (hc : ‖c₀‖ = 1) (A Cp : Polynomial ℂ) :
    (C c₀ * A) * (C c₀ * A).map (starRingEnd ℂ)
      + (1 - X ^ 2) * ((C c₀ * Cp) * (C c₀ * Cp).map (starRingEnd ℂ))
    = A * A.map (starRingEnd ℂ) + (1 - X ^ 2) * (Cp * Cp.map (starRingEnd ℂ)) := by
  have hcc : (C c₀ * C ((starRingEnd ℂ) c₀) : ℂ[X]) = 1 := by
    rw [← C_mul]
    have : c₀ * (starRingEnd ℂ) c₀ = 1 := by
      rw [Complex.mul_conj, Complex.normSq_eq_norm_sq, hc]; norm_num
    rw [this, map_one]
  simp only [Polynomial.map_mul, Polynomial.map_C]
  calc (C c₀ * A) * (C ((starRingEnd ℂ) c₀) * A.map (starRingEnd ℂ))
        + (1 - X ^ 2) * ((C c₀ * Cp) * (C ((starRingEnd ℂ) c₀) * Cp.map (starRingEnd ℂ)))
      = (C c₀ * C ((starRingEnd ℂ) c₀))
          * (A * A.map (starRingEnd ℂ) + (1 - X ^ 2) * (Cp * Cp.map (starRingEnd ℂ))) := by ring
    _ = _ := by rw [hcc, one_mul]

/-- **Both entries of the single-phase first column.** `qsp x s [φ]` has first
    column `(e^{iφ}·x, s·e^{-iφ}·i)`, i.e. it realizes the pair
    `(C(e^{iφ})·X, C(e^{-iφ}·i))`. This is the explicit degree-1 realization the
    converse induction's `d = 1` base reads off (the off-diagonal phase is rigidly
    tied to the diagonal one — the `SL₂` constraint at degree 1). -/
theorem qsp_singleton_realizes (φ : ℝ) :
    Realizes [φ] (C (Complex.exp (I * (φ : ℂ))) * X) (C (Complex.exp (-(I * (φ : ℂ))) * I)) := by
  intro x s hs
  have hq : qsp x s [φ] = phase φ * signal x s := by
    show phase φ * signal x s * qsp x s [] = phase φ * signal x s
    rw [show qsp x s [] = 1 from rfl, mul_one]
  refine ⟨?_, ?_⟩ <;>
  · rw [hq]
    simp only [phase, signal, mul_apply, Fin.sum_univ_two, of_apply, cons_val', cons_val_zero,
      cons_val_one, head_cons, empty_val', cons_val_fin_one, diagonal, Matrix.diagonal_apply,
      eval_mul, eval_C, eval_X]
    ring

/-! ### Converse — admissibility predicate and the degree-0 base

`Admissible d A Cp` bundles the converse invariant for a degree-`d` first column:
degree budget (`deg A ≤ d`, `deg Cp < d`), matching parities (`A` parity `d`,
`Cp` parity `d+1`), and the Gram (unitarity) identity. The converse — proven by
strong induction on `d` — produces a phase list and unit global phase `c₀`
realizing `(C c₀·A, C c₀·Cp)`. Here is the predicate and the `d = 0` base. -/

/-- **Admissible first-column pair of degree `d`.** Degree budget, matching
    parities (diagonal `d`, off-diagonal `d+1` — equal mod 2 to the forward
    theorem's `d-1`), and the Gram identity `A·A⋆ + (1−X²)·C·C⋆ = 1`. The
    `Cp.degree < d` form (rather than `natDegree ≤ d-1`) cleanly forces `Cp = 0`
    at `d = 0`. -/
def Admissible (d : ℕ) (A Cp : Polynomial ℂ) : Prop :=
  A.natDegree ≤ d ∧ Cp.degree < (d : WithBot ℕ) ∧
  A.comp (-X) = (-1) ^ d * A ∧ Cp.comp (-X) = (-1) ^ (d + 1) * Cp ∧
  A * A.map (starRingEnd ℂ) + (1 - X ^ 2) * (Cp * Cp.map (starRingEnd ℂ)) = 1

/-- **Converse base case `d = 0`.** An admissible degree-`0` pair has `Cp = 0` and
    `A = C a` a unit constant (`a·conj a = 1` from Gram). The empty phase list
    realizes it up to the global phase `c₀ = conj a`: `qsp [] = I` has `(0,0) = 1
    = c₀·a` and `(1,0) = 0`. -/
theorem converse_base0 (A Cp : Polynomial ℂ) (h : Admissible 0 A Cp) :
    ∃ (φs : List ℝ) (c₀ : ℂ), ‖c₀‖ = 1 ∧ φs.length = 0 ∧
      Realizes φs (C c₀ * A) (C c₀ * Cp) := by
  obtain ⟨hdA, hdC, _, _, hgram⟩ := h
  have hCp0 : Cp = 0 := by
    by_contra hne
    exact absurd (Polynomial.zero_le_degree_iff.mpr hne) (not_le.mpr (by exact_mod_cast hdC))
  have hAC : A = C (A.coeff 0) := Polynomial.eq_C_of_natDegree_le_zero hdA
  set a := A.coeff 0 with ha
  have haa : a * (starRingEnd ℂ) a = 1 := by
    have h2 := hgram
    rw [hCp0] at h2
    simp only [Polynomial.map_zero, mul_zero, zero_mul, add_zero] at h2
    rw [hAC, Polynomial.map_C, ← C_mul] at h2
    have := congrArg (fun p => Polynomial.coeff p 0) h2
    simpa using this
  have hns : Complex.normSq a = 1 := by
    have := haa; rw [Complex.mul_conj] at this; exact_mod_cast this
  have hna : ‖a‖ = 1 := by
    have h1 : ‖a‖ ^ 2 = 1 := by rw [← Complex.normSq_eq_norm_sq]; exact hns
    nlinarith [norm_nonneg a]
  have hc0a : (starRingEnd ℂ) a * a = 1 := by rw [mul_comm]; exact haa
  refine ⟨[], (starRingEnd ℂ) a, ?_, rfl, ?_⟩
  · rw [RCLike.norm_conj]; exact hna
  · intro x s hs
    rw [hCp0, hAC]
    refine ⟨?_, ?_⟩
    · show (qsp x s []) 0 0 = (C ((starRingEnd ℂ) a) * C a).eval (x : ℂ)
      rw [show qsp x s [] = 1 from rfl, ← C_mul, eval_C, hc0a, Matrix.one_apply_eq]
    · show (qsp x s []) 1 0 = (s : ℂ) * (C ((starRingEnd ℂ) a) * 0).eval (x : ℂ)
      rw [show qsp x s [] = 1 from rfl]
      simp [Matrix.one_apply]

/-- **Converse base case `d = 1`.** A degree-1 admissible pair is `A = C a₁·X`
    (the parity forces `A[0] = 0`) and `Cp = C b`, with `‖a₁‖ = ‖b‖ = 1` (Gram).
    The single-phase list realizes it up to a unit global phase: solve
    `e² = i·a₁·conj b` for `φ = arg(...)/2`, set `c₀ = e·conj a₁`; then
    `C c₀·A = C(e)·X` and `C c₀·Cp = C(e⁻¹·i)`, which `qsp_singleton_realizes`
    realizes. The rigid `e⁻¹·i` off-diagonal exhibits the `SL₂` phase tie. -/
theorem converse_base1 (A Cp : Polynomial ℂ) (h : Admissible 1 A Cp) :
    ∃ (φs : List ℝ) (c₀ : ℂ), ‖c₀‖ = 1 ∧ φs.length = 1 ∧
      Realizes φs (C c₀ * A) (C c₀ * Cp) := by
  obtain ⟨hdA, hdC, hpA, _, hgram⟩ := h
  have hCnd : Cp.natDegree ≤ 0 := by
    rcases eq_or_ne Cp 0 with h0 | h0
    · simp [h0]
    · exact Nat.le_of_lt_succ ((Polynomial.natDegree_lt_iff_degree_lt h0).mpr (by exact_mod_cast hdC))
  have hCb : Cp = C (Cp.coeff 0) := Polynomial.eq_C_of_natDegree_le_zero hCnd
  set b := Cp.coeff 0 with hb
  have hA0 : A.coeff 0 = 0 := parity_coeff_zero A 1 0 (by simpa using hpA) ⟨0, by ring⟩
  set a₁ := A.coeff 1 with ha1
  have hAX : A = C a₁ * X := by
    have h2 := Polynomial.eq_X_add_C_of_natDegree_le_one hdA
    rw [hA0, map_zero, add_zero] at h2
    exact h2
  have hgram2 : C (a₁ * (starRingEnd ℂ) a₁) * X ^ 2
      + (1 - X ^ 2) * C (b * (starRingEnd ℂ) b) = 1 := by
    have hg := hgram
    rw [hAX, hCb] at hg
    simp only [Polynomial.map_mul, Polynomial.map_C, Polynomial.map_X] at hg
    rw [C_mul, C_mul]; linear_combination hg
  have e0 := congrArg (fun p => Polynomial.coeff p 0) hgram2
  have e2 := congrArg (fun p => Polynomial.coeff p 2) hgram2
  simp only [coeff_add, coeff_C_mul, coeff_mul_C, coeff_X_pow, coeff_sub, coeff_one] at e0 e2
  norm_num at e0 e2
  have hpb : b * (starRingEnd ℂ) b = 1 := e0
  have hpa : a₁ * (starRingEnd ℂ) a₁ = 1 := by linear_combination e2 + e0
  -- norms
  have hna1 : ‖a₁‖ = 1 := by
    have hh : Complex.normSq a₁ = 1 := by
      have h := hpa; rw [Complex.mul_conj] at h; exact_mod_cast h
    nlinarith [norm_nonneg a₁, Complex.normSq_eq_norm_sq a₁, hh]
  have hnb : ‖b‖ = 1 := by
    have hh : Complex.normSq b = 1 := by
      have h := hpb; rw [Complex.mul_conj] at h; exact_mod_cast h
    nlinarith [norm_nonneg b, Complex.normSq_eq_norm_sq b, hh]
  -- angle
  set w : ℂ := I * a₁ * (starRingEnd ℂ) b with hw
  have hw1 : ‖w‖ = 1 := by
    rw [hw]; rw [norm_mul, norm_mul, Complex.norm_I, one_mul, RCLike.norm_conj, hna1, hnb, mul_one]
  set φ : ℝ := Complex.arg w / 2 with hφ
  set e : ℂ := Complex.exp (I * (φ : ℂ)) with he
  have he2 : e * e = w := by
    rw [he, ← Complex.exp_add]
    have hcast : I * (φ : ℂ) + I * (φ : ℂ) = I * (Complex.arg w : ℂ) := by
      rw [hφ]; push_cast; ring
    rw [hcast]; exact expI_arg hw1
  have hem : e * Complex.exp (-(I * (φ : ℂ))) = 1 := by
    rw [he, ← Complex.exp_add]; simp
  set c₀ : ℂ := e * (starRingEnd ℂ) a₁ with hc0
  have heq1 : c₀ * a₁ = e := by
    rw [hc0]; rw [mul_assoc]; rw [show (starRingEnd ℂ) a₁ * a₁ = 1 from by rw [mul_comm]; exact hpa]
    rw [mul_one]
  have heq2 : c₀ * b = Complex.exp (-(I * (φ : ℂ))) * I := by
    have he0 : e ≠ 0 := Complex.exp_ne_zero _
    apply mul_right_cancel₀ he0
    rw [hc0]
    calc e * (starRingEnd ℂ) a₁ * b * e
        = (e * e) * ((starRingEnd ℂ) a₁ * b) := by ring
      _ = w * ((starRingEnd ℂ) a₁ * b) := by rw [he2]
      _ = I * (a₁ * (starRingEnd ℂ) a₁) * (b * (starRingEnd ℂ) b) := by rw [hw]; ring
      _ = I := by rw [hpa, hpb]; ring
      _ = Complex.exp (-(I * (φ : ℂ))) * I * e := by
          have hbk : Complex.exp (-(I * (φ : ℂ))) * I * e = I := by
            rw [show Complex.exp (-(I * (φ:ℂ))) * I * e
                  = I * (Complex.exp (-(I*(φ:ℂ))) * e) from by ring,
              show Complex.exp (-(I*(φ:ℂ))) * e = 1 from by rw [mul_comm]; exact hem, mul_one]
          rw [hbk]
  refine ⟨[φ], c₀, ?_, rfl, ?_⟩
  · rw [hc0, norm_mul, RCLike.norm_conj, hna1, mul_one, he, Complex.norm_exp]
    simp
  · have hA' : C c₀ * A = C e * X := by
      rw [hAX, ← mul_assoc, ← C_mul, heq1]
    have hC' : C c₀ * Cp = C (Complex.exp (-(I * (φ : ℂ))) * I) := by
      rw [hCb, ← C_mul, heq2]
    rw [hA', hC']
    exact qsp_singleton_realizes φ

/-- **Converse inductive step (`d = k+2`).** Peel one phase via
    `peel_first_column_drop`: the chosen `φ` drops `(A, Cp)` to
    `(peelA φ A Cp, peelC φ A Cp)` of degree `(≤ k+1, ≤ k)`. That peeled pair is
    again `Admissible (k+1)` — degrees from the drop, parities from
    `peelA_parity`/`peelC_parity`, and the Gram identity from `gram_peel_invariant`
    (unitarity preserved). The induction hypothesis realizes the peeled pair up to
    some `c₀`; `realize_step` (with `peelA_smul`/`peelC_smul` threading the *same*
    `c₀`) prepends `φ` to realize the original pair. -/
theorem converse_step (k : ℕ) (A Cp : Polynomial ℂ) (h : Admissible (k + 2) A Cp)
    (IH : ∀ A' Cp' : Polynomial ℂ, Admissible (k + 1) A' Cp' →
      ∃ (φs : List ℝ) (c₀ : ℂ), ‖c₀‖ = 1 ∧ φs.length = k + 1 ∧
        Realizes φs (C c₀ * A') (C c₀ * Cp')) :
    ∃ (φs : List ℝ) (c₀ : ℂ), ‖c₀‖ = 1 ∧ φs.length = k + 2 ∧
      Realizes φs (C c₀ * A) (C c₀ * Cp) := by
  obtain ⟨hdA, hdC, hpA, hpC, hgram⟩ := h
  have hCnd : Cp.natDegree ≤ k + 1 := by
    rcases eq_or_ne Cp 0 with h0 | h0
    · simp [h0]
    · exact Nat.le_of_lt_succ ((Polynomial.natDegree_lt_iff_degree_lt h0).mpr (by exact_mod_cast hdC))
  have hpC' : Cp.comp (-X) = (-1) ^ (k + 1) * Cp := by
    rw [hpC, show ((-1 : ℂ[X])) ^ (k + 2 + 1) = (-1) ^ (k + 1) from by rw [pow_add]; ring]
  have hmod : ‖A.coeff (k + 2)‖ = ‖Cp.coeff (k + 1)‖ :=
    gram_lead_modulus A Cp (k + 1) hdA hCnd hgram
  obtain ⟨φ, hpeelA, hpeelC⟩ := peel_first_column_drop A Cp k hdA hCnd hpA hpC' hmod
  have hAdm' : Admissible (k + 1) (peelA φ A Cp) (peelC φ A Cp) := by
    refine ⟨hpeelA, ?_, ?_, ?_, ?_⟩
    · have h1 : (peelC φ A Cp).degree ≤ (k : WithBot ℕ) := by
        rcases eq_or_ne (peelC φ A Cp) 0 with h0 | h0
        · simp [h0]
        · rw [Polynomial.degree_eq_natDegree h0]; exact_mod_cast hpeelC
      exact lt_of_le_of_lt h1 (by exact_mod_cast Nat.lt_succ_self k)
    · rw [peelA_parity φ A Cp k hpA hpC',
        show ((-1 : ℂ[X])) ^ (k + 3) = (-1) ^ (k + 1) from by rw [pow_add]; ring]
    · exact peelC_parity φ A Cp k hpA hpC'
    · rw [gram_peel_invariant]; exact hgram
  obtain ⟨φs, c₀, hc0, hlen, hreal⟩ := IH (peelA φ A Cp) (peelC φ A Cp) hAdm'
  refine ⟨φ :: φs, c₀, hc0, by simp [hlen], ?_⟩
  apply realize_step
  rw [peelA_smul, peelC_smul]
  exact hreal

/-- **QSP fundamental theorem — converse (up to global phase).** Every admissible
    first-column pair `(A, Cp)` of degree `d` is realized — up to a single unit
    global phase `c₀` (forced by the `SL₂(ℂ)` determinant obstruction) — by a
    length-`d` phase list: there exist `φs` (`|φs| = d`) and `c₀` (`‖c₀‖ = 1`) with
    the first column of `qsp x s φs` equal to `(c₀·A(x), s·c₀·Cp(x))`. Proven by
    strong induction on `d`: bases `d = 0, 1` (`converse_base0`/`converse_base1`)
    and the peel step (`converse_step`). Together with the forward theorems
    (`qsp_implements_poly` + degree + parity + `qsp_gram_diag`), this closes the
    QSP characterization: a pair is QSP-realizable (up to global phase) **iff** it
    is admissible (degree budget + parity + the Gram unitarity identity). -/
theorem qsp_converse : ∀ (d : ℕ) (A Cp : Polynomial ℂ), Admissible d A Cp →
    ∃ (φs : List ℝ) (c₀ : ℂ), ‖c₀‖ = 1 ∧ φs.length = d ∧
      Realizes φs (C c₀ * A) (C c₀ * Cp) := by
  intro d
  induction d using Nat.strong_induction_on with
  | _ d IH =>
    intro A Cp h
    match d, h, IH with
    | 0, h, _ => exact converse_base0 A Cp h
    | 1, h, _ => exact converse_base1 A Cp h
    | (k + 2), h, IH =>
      exact converse_step k A Cp h (fun A' Cp' h' => IH (k + 1) (by omega) A' Cp' h')


end QuantumProofs.QSP
