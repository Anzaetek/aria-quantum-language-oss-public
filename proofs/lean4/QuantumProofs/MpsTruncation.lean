/-
  MPS truncation: the three facts the backend's *control flow* depends on.

  These are not decorative. Each one was argued informally during development,
  and in two of the three cases the informal argument was wrong for a while —
  which is the case for writing them down once, in a form that cannot drift
  from the code it specifies.

  ## What each theorem specifies

  * `certificate_monotone` — `Mps::apply_2q` accumulates
    `self.discarded_weight += rel_discarded` with `rel_discarded ≥ 0`
    (`mps.rs`). `MpsBackend::execute` relies on this to abort at the FIRST
    crossing of the ceiling instead of after the whole circuit: if the running
    value is already over, the final value is over too, so the set of refused
    runs is unchanged and only the time-to-refuse differs. That claim shipped
    as a comment; this is the proof of it.

  * `certificate_can_exceed_one` — the certificate is a SUM over splits of
    per-split *fractions*, not a probability, so a value like 6.586 is legal.
    This was mistaken for an accumulation bug ("you cannot discard 659% of a
    state"), and the error message was reworded because of it.

  * `schmidt_rank_le_half` — the bond a cut can require is at most
    `2^(n/2)`, so `χ ≥ 2^(n/2)` makes truncation impossible and an MPS run
    exact. This bound is what distinguishes "the bond is too small" from "the
    kernel is broken" — a distinction a wrong diagnosis turned on for a day.

  ## What is deliberately NOT claimed

  `DEFAULT_MAX_DISCARDED_WEIGHT`'s doc used to say the discarded weight *bounds
  the infidelity*. That theorem is for **canonical-gauge** truncation, and this
  MPS is explicitly non-canonical (`mps.rs` says so, and explains why). Nothing
  here proves a fidelity bound, and the Rust doc no longer claims one — it calls
  the number a proxy and cites a measured ~10× gauge inflation.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Group.Nat
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

namespace QuantumProofs.MpsTruncation

/-- A run's truncation certificate: the sum of the per-split relative discarded
weights, in split order. Mirrors `Mps::discarded_weight`, which is accumulated
with `+=` as the circuit is applied. -/
def certificate (w : List ℝ) : ℝ := w.foldl (· + ·) 0

/-- Every per-split term is non-negative: it is `Σσ²` over the dropped tail,
divided by a positive norm (`mps.rs`). -/
def Nonneg (w : List ℝ) : Prop := ∀ x ∈ w, 0 ≤ x

theorem certificate_nil : certificate [] = 0 := rfl

private theorem foldl_add_eq (w : List ℝ) (a : ℝ) :
    w.foldl (· + ·) a = a + w.foldl (· + ·) 0 := by
  induction w generalizing a with
  | nil => simp [List.foldl]
  | cons x xs ih =>
    simp only [List.foldl]
    rw [ih (a + x), ih (0 + x)]
    linarith

/-- Appending a split adds its (non-negative) term. -/
theorem certificate_cons (x : ℝ) (w : List ℝ) :
    certificate (x :: w) = x + certificate w := by
  simp only [certificate, List.foldl]
  rw [foldl_add_eq w (0 + x)]
  linarith
/-- **The certificate never decreases as splits are applied.**

    This is what licenses the early abort in `MpsBackend::execute` /
    `NoisyMpsBackend::execute`: a prefix that is already over the ceiling
    guarantees the whole run is over it, so stopping early refuses exactly the
    same runs. -/
theorem certificate_monotone (x : ℝ) (w : List ℝ) (hx : 0 ≤ x) :
    certificate w ≤ certificate (x :: w) := by
  rw [certificate_cons]
  linarith

/-- The same statement in prefix form: extending a run's split list can only
    raise the certificate. -/
theorem certificate_mono_append (w v : List ℝ) (hv : Nonneg v) :
    certificate w ≤ certificate (w ++ v) := by
  induction w with
  | nil =>
    simp only [certificate_nil, List.nil_append]
    induction v with
    | nil => simp [certificate_nil]
    | cons y ys ih =>
      rw [certificate_cons]
      have hy : 0 ≤ y := hv y (by simp)
      have hys : Nonneg ys := fun z hz => hv z (by simp [hz])
      linarith [ih hys]
  | cons x xs ih =>
    rw [List.cons_append, certificate_cons, certificate_cons]
    linarith [ih]

/-- **The certificate is not a probability and may exceed 1.**

    Two splits that each discard 60% of their local weight sum to 1.2. So
    `discarded_weight = 6.586` is not evidence of an accumulation bug — it is a
    sum over ~1555 splits. -/
theorem certificate_can_exceed_one :
    ∃ w : List ℝ, Nonneg w ∧ 1 < certificate w := by
  refine ⟨[(0.6 : ℝ), 0.6], ?_, ?_⟩
  · intro x hx
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with h | h <;> (subst h; norm_num)
  · rw [certificate_cons, certificate_cons, certificate_nil]
    norm_num

/-- **A cut of an `n`-qubit chain needs bond at most `2^(n/2)`.**

    The Schmidt rank across the cut after site `k` is at most
    `min (2^k) (2^(n-k))` — the dimension of the smaller side. Maximised over
    `k`, that is `2^(n/2)` (natural division, so `⌊n/2⌋`).

    Consequence, and the reason this is here: at `χ ≥ 2^(n/2)` no split can
    truncate, so any disagreement with an exact simulator at that bond is a
    KERNEL defect, not truncation loss. Measured at n = 8, 10, 12 with
    `χ = 2^(n/2)`: TVD 3.6e-15 / 5.8e-15 / 6.6e-15 against the dense
    statevector. -/
theorem schmidt_rank_le_half (n k : ℕ) (hk : k ≤ n) :
    min (2 ^ k) (2 ^ (n - k)) ≤ 2 ^ (n / 2) := by
  by_cases h : k ≤ n / 2
  · exact le_trans (min_le_left _ _) (Nat.pow_le_pow_right (by norm_num) h)
  · refine le_trans (min_le_right _ _) (Nat.pow_le_pow_right (by norm_num) ?_)
    omega

end QuantumProofs.MpsTruncation
