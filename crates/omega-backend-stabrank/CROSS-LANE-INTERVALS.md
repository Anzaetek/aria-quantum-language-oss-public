<!-- SPDX-License-Identifier: Apache-2.0 -->
# Two truncating engines, two certificates: what you may and may not do with them

`omega-run` has two engines that answer `⟨0|U†OU|0⟩` approximately and tell
you how wrong they might be: **majoranaprop** and **stabrank**. Both report a
discarded "mass". The two masses are **different quantities** and this
document exists so that nobody adds them.

The short version:

* **You may not combine the two dropped-mass fields.** No sum, no average, no
  feeding one engine's budget to the other. There is no arithmetic that does
  it, because the numbers live in different pictures.
* **You may intersect the two certified intervals.** That is sound, it is
  tighter than either, and it is the only supported way to combine the lanes.
* **An empty intersection is a bug in one of the two engines.** Never a
  property of the job, and never something to fix by widening a bound.

The rest of this document is why.

## 1. The two fields, and why they are not the same number

Both engines truncate. They truncate different things.

### majoranaprop: Heisenberg, observable-side, additive

majoranaprop works in the Heisenberg picture. It evolves the *observable*
backwards through the circuit as a sum of Majorana monomials
`O(t) = Σ_b c_b M_b` and reads it out against the initial state. Truncation
drops monomials, so what is discarded is part of the **observable**.

Every Majorana monomial is a product of Hermitian, unitary generators, so
`|⟨M_b⟩| ≤ 1` on any state at all. Therefore the error in the expectation is
at most the discarded L1 coefficient mass, directly:

```text
|Δ⟨O⟩| ≤ Σ_dropped |c_b| = dropped_mass
```

So majoranaprop's `dropped_mass` **is** the error bound. The certified
interval is `[value − dropped_mass, value + dropped_mass]`, and that is why
the JSON block carries `dropped_mass_is_a_bound: true`.

### stabrank: Schrödinger, state-side, bilinear

stabrank works in the Schrödinger picture. It writes the run's non-Clifford
content as a sum of χ stabilizer states `|ψ⟩ = Σᵢ cᵢ|φᵢ⟩` and reads the
observable out at the end. Truncation drops branches, so what is discarded is
part of the **state**.

Each `|φᵢ⟩` is a unit vector, so by the triangle inequality the discarded L1
coefficient mass `m = Σ_dropped |cᵢ|` bounds the error *in the state*:
`‖ψ − ψ′‖ ≤ m`. But the state enters the expectation **twice** — `⟨ψ|O|ψ⟩` is
bilinear in it — so that is not yet an error in `⟨O⟩`:

```text
|⟨ψ|O|ψ⟩ − ⟨ψ′|O|ψ′⟩|  ≤  ‖O‖ · ‖ψ − ψ′‖ · (‖ψ‖ + ‖ψ′‖)  ≤  R · m · (2 + m)
```

with `R = Σ|cᵢ|` the Pauli-L1 of the observable (`‖O‖ ≤ R`). The last step
uses `‖ψ‖ = 1` and `‖ψ′‖ ≤ ‖ψ‖ + ‖ψ − ψ′‖ ≤ 1 + m`, which is where the
`(2 + m)` comes from.

So stabrank's `state_dropped_mass` is **an input to** the bound, not the
bound. The bound is `expectation_error_bound`, it is `R·m·(2+m)`, and the JSON
block carries `state_dropped_mass_is_a_bound: false` to say so.

### The two differences, in one place

| | majoranaprop | stabrank |
|---|---|---|
| picture | Heisenberg | Schrödinger |
| truncates | the observable | the state |
| raw field | `dropped_mass` | `state_dropped_mass` |
| error bound | `dropped_mass` | `expectation_error_bound = R·m·(2+m)` |
| field *is* the bound | yes | **no** |
| enters the error | additively | quadratically, scaled by `R` |
| cheap when | the circuit is Givens-dominated | the circuit is Clifford+T-dominated |

The field names differ on purpose. Pouring a state-side mass into a field
whose documented meaning is "additive bound on `|Δ⟨O⟩|`" would make the
certificate lie, and a certificate in this project is a bound and not an
estimate.

On one measured job (four qubits, χ = 64, `R = 2.7`), the same two numbers:

```text
majoranaprop --max-length 6   dropped_mass          0.1414213562373095
                              bound                 0.1414213562373095   (= the field)
stabrank     --max-chi 63     state_dropped_mass    0.0031407832308854577
                              bound                 0.016986863648900685 (= 5.4085x the field)
```

A consumer that read `state_dropped_mass` as an error budget would claim an
interval **five times tighter** than the derivation supports. A consumer that
added `0.1414` to `0.0031` would be adding a number in the Heisenberg picture
to a number in the Schrödinger one, and the result would mean nothing at all.

## 2. What two runs of one job *do* give you

Each engine returns a value and a sound bound, hence a **certified interval**:

```text
majoranaprop:  [v_m − dropped_mass,            v_m + dropped_mass]
stabrank:      [v_s − expectation_error_bound, v_s + expectation_error_bound]
```

Both bounds are sound statements about the *same* exact number. So that number
lies in **both** intervals, hence in their intersection. The intersection is
therefore itself a sound certified interval, and it is no wider than either
input. Intersecting is the supported way to combine the lanes, and the only
one.

On the measured job above, with dense value `0.6664213562373094`:

```text
stabrank      [0.6500809614406684, 0.6840546887384698]   width 0.03397
majoranaprop  [0.3835786437626904, 0.6664213562373094]   width 0.28284
intersection  [0.6500809614406684, 0.6664213562373094]   width 0.01634
```

A genuine two-sided cut: the lower end comes from stabrank and the upper end
from majoranaprop, and the result is strictly narrower than either input.

Note what the intersection is **not**. It is not an averaged estimate, it is
not a confidence interval, and its width is not an error bar in the
statistical sense. It is a set the exact answer is guaranteed to lie in, given
that both engines' bounds hold.

Nothing in this project feeds one engine's truncated output into the other, so
the question of composing the bounds *sequentially* never arises. The two
runs are independent runs of the same job.

## 3. If the intersection is ever empty

**That is a bug in one of the two engines.** It is not a property of the job
and it is not a tolerance problem. Both engines claim their bound is a bound;
if the intervals are disjoint then the exact value is outside at least one of
them, so at least one claim is false.

The correct response is to find the unsound bound. The wrong response — and
the reason this paragraph is here — is to widen a bound until the intervals
meet, which converts the most valuable signal these two lanes can produce
into a green test.

`omega-cli/tests/stabrank_majoranaprop_intervals.rs` is the test that watches
for this, and its failure message prints both intervals and the dense value so
the guilty side can be identified from the output alone.

## 4. Reading the JSON

A machine consumer gets `majoranaprop_truncation` and/or
`stabrank_truncation` blocks beside the value. Four notes.

**`informative` is the key to branch on.** `false` means the bound excludes
nothing: every value the observable could take is consistent with the result.
stabrank refuses such a run rather than returning it, so `false` should not
reach you through that door, but the key is emitted either way so you need not
know that to be safe.

**The bound is re-derivable, and you should check it.** Exactly, in f64:

```text
expectation_error_bound == observable_range * state_dropped_mass * (2 + state_dropped_mass)
```

Both inputs are emitted beside the derived field precisely so this is
possible. One caveat found while testing it: an exact comparison needs a
**correctly-rounded float parser**. The emitted text is exact (shortest
round-trip), but some JSON parsers — including `serde_json` without its
`float_roundtrip` feature — can be one ULP off, which is enough to break
equality while being four orders of magnitude tighter than any wrong formula
would need. With a sloppy parser, compare to within one ULP.

**`truncated_norm_sqr` is "different from 1", not "at most 1".** It is
`⟨ψ′|ψ′⟩` of the state actually read out, and it is emitted so the
no-renormalisation rule is checkable rather than merely promised: the bound is
derived for the **raw** truncated state, and an engine that rescaled `ψ′` back
to unit norm would be satisfying a bound nobody derived — while usually
returning numbers that look *better*. The field is usually below 1 and it is
not always: the branches of a decomposition are not orthogonal, so
`‖ψ′‖² = 1 − ‖Δ‖² − 2Re⟨ψ′|Δ⟩` for the dropped part `Δ`, and a `Δ` that
overlaps the survivors negatively leaves the remainder *longer* than the state
it came from. Two of forty witness cells measured `1.2285` where the other
thirty-eight sat at `0.7285` or `0.9785`. Code that treats `> 1` as impossible
will reject correct runs.

**`seed_basis` differing between the two blocks is not a disagreement.**
majoranaprop reports `ladder` on `--expectation-fermionic`, because its seed
*is* the operator and it is seeded straight from the ladder terms. stabrank
reports `pauli` always, because the Schrödinger seed is the state `|0…0⟩` and
the observable arrives already mapped by Jordan–Wigner — exactly, so nothing
is lost, and there is no ladder-native seeding to choose on the state side
since the state is not an operator sum. What *must* agree across that
difference is `observable_range`: both engines compute `R` in the basis
actually read out, and a Pauli string and its Majorana monomial differ by a
sign only.

## 5. The knobs

The flags that change the answer, and the flags that do not:

| flag | engine | truncates? | effect |
|---|---|---|---|
| `--truncate C` | all three | yes | drop coefficients below `C` |
| `--max-length L` | majoranaprop | yes | drop Majorana monomials longer than `L` |
| `--max-chi X` | stabrank | yes | keep the `X` heaviest stabilizer branches |
| `--max-terms N` | pauliprop, majoranaprop | **no** | raise the Pauli-term ceiling |
| `--max-branches N` | stabrank | **no** | raise the branch ceiling |

The bottom two rows are not accuracy knobs. Reaching a ceiling **refuses** the
run and discards nothing, so two completed runs under different ceilings
return the same number; raising a ceiling buys a larger exact calculation and
costs memory and time, never accuracy. Reach for them first when a run is
refused.

For stabrank the distinction is worth repeating because the two flags both
sound like "a limit on χ": `--max-chi` cuts and owes a mass and a bound,
`--max-branches` refuses and owes nothing.

## 6. Which engine to put a job on

They are complementary cost regimes, not a fast one and a slow one.

* **majoranaprop** is cheap when the circuit is **Givens-dominated**: a
  generic-angle Givens rotation is free because it preserves Majorana length.
  Cost grows with branching gates — length-≥4 generators, and T.
* **stabrank** is cheap when the circuit is **Clifford+T-dominated**: any
  Clifford gate is free. Cost is exponential in the count of non-Clifford
  gates, which includes every generic-angle Givens rotation, since `Rbs(θ)` is
  Clifford only at multiples of π/2.

Their overlap — Clifford-angle Givens plus T — is where they can cross-check
each other, which is what the interval test uses. Neither engine replaces or
obsoletes the other.
