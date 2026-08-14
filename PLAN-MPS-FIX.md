<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — fixing the MPS backend

**Status: PLAN. Not implemented.** Written after the measurements below.

## The headline: there is no kernel bug, and I said there was

Yesterday's note (task #29) recorded "MPS is structurally wrong on
`wide_chain_19q` — TVD barely improves with χ". **That inference was wrong**,
and it is corrected here before anything is built on it.

The reasoning was: TVD vs the exact statevector went 0.594 → 0.539 → 0.510 as χ
went 16 → 32 → 64, and ordinary truncation error should collapse over a 4× bond
increase, so something structural must be broken.

The step I skipped was checking what χ this circuit actually needs. **On 19
qubits the maximum Schmidt rank is 2⁹ = 512.** χ=64 is one eighth of exact, on a
circuit with 135 two-qubit gates and depth 196 — deep enough to scramble. Slow
convergence at ⅛ of the required bond is not evidence of a defect.

### The decisive measurement

Same gate vocabulary as the failing circuit (`u`, `t`, `tdg`, `cx`), deep enough
to scramble, but on few enough qubits that **χ ≥ 2^(n/2) makes truncation
mathematically impossible**:

| n | layers | χ (exact) | TVD vs statevector | ‖ψ‖² | discarded |
|---|---|---|---|---|---|
| 8 | 24 | 16 | 3.573e-15 | 1.00000000 | 8.8e-33 |
| 10 | 24 | 32 | 5.762e-15 | 1.00000000 | 1.4e-32 |
| 12 | 20 | 64 | 6.578e-15 | 1.00000000 | 2.3e-32 |

Exact to machine precision, norm exactly 1, discarded weight at the rounding
floor. **The kernel is correct.** The one-sided Jacobi SVD that replaced the old
normal-equations kernel is doing its job.

And at 19 qubits the trend is consistent with genuine truncation, not corruption:

| χ | TVD | discarded | wall |
|---|---|---|---|
| 64 | 5.096e-1 | 6.586 | 51.7 s |
| 128 | 4.814e-1 | 4.801 | 341.4 s |

Improving, slowly, exactly as an under-provisioned bond should on a scrambling
circuit — and costing 6.6× the time for a 2× bond.

## So what is actually wrong

Three things, none of them the simulator's arithmetic.

### M1 — a silently approximate answer *(already fixed, recorded for completeness)*

The truncation certificate was computed, printed, and consulted by nothing, so a
run that discarded 6.586 of the state returned a distribution half of which was
wrong. Now gated by `DEFAULT_MAX_DISCARDED_WEIGHT`, with
`with_max_discarded_weight` to opt into an approximation deliberately.

### M2 — **the user cannot ask for a correct answer**

This is the real defect now, and it is a usability one with a correctness
consequence.

`omega-run --backend mps` hardcodes `MpsBackend::new(64)`. The `--help` text and
the backend documentation both advertise `mps:auto` and `mps:<chi>`; the
dispatcher matches the literal string `"mps"` and rejects everything else as an
unknown backend. So:

* a user whose circuit needs χ=512 has **no way to say so** through the CLI;
* after M1, that user now gets a refusal instead of a wrong answer — which is
  better, but leaves them with no route to a right one;
* the documentation describes a knob that does not exist, which is its own
  defect and was mistaken for a fix during this investigation.

`MpsBackend::with_adaptive` exists and is unreachable from the CLI.

**Fix:** parse `mps`, `mps:auto`, `mps:auto:<ceiling>`, `mps:<chi>` in the
backend selector, everywhere `"mps"` is currently matched — there are several
sites (execute, expectation, gradient, functional gradient), and fixing one is
how this becomes inconsistent. Then the `--help` text describes reality.

### M3 — fixed bond does full-χ work regardless of entanglement

The reported "~2000× slower than an external simulator" on `wide_chain_19q` is
partly this: with a fixed χ every split is contracted at full bond even when the
Schmidt spectrum is nearly rank-1. `qft_16` reaching `max_bond_reached=1` shows
the spectrum can be trivial while the cost is not.

Adaptive mode already implements the remedy. Once M2 exposes it, the question
becomes what the **default** should be — and that is a real decision, not an
implementation detail:

* `mps` meaning fixed χ=64 is predictable and is what every existing measurement
  in this repository was taken with;
* `mps` meaning adaptive is faster and more accurate on low-entanglement
  circuits and slower to explain when a result changes.

**Recommendation: leave the default alone, expose the alternatives.** Changing
what a bare `--backend mps` means would silently change every previously
recorded number, which is the class of thing this repository keeps getting
burned by.

## What must be true before any of this is called done

* **No performance change may alter a result.** M3 touches contraction order and
  truncation; every change gets the exact-bond equivalence test above (n = 8,
  10, 12 at χ = 2^(n/2)) plus the cross-backend comparison against Qiskit.
* **The exactness test belongs in the suite**, not in a scratch file. It is the
  thing that distinguishes "the bond is too small" from "the kernel is broken",
  and its absence is why a wrong diagnosis survived a day.
* **`wide_chain_19q` stays a fixture** — as a circuit that must be *refused* at
  χ=64 and *correct* at χ=512, not as a circuit that must be fast.

## What could make this pass for the wrong reason

* **Testing the kernel only where truncation happens.** Every disagreement is
  then attributable to truncation and the test cannot fail for a kernel reason.
  The exact-bond cases are the whole point.
* **Testing exactness only on shallow or product-like circuits.** `qft_16` on
  |0…0⟩ reaches bond 1 — it would pass with almost any kernel. The fixture has
  to actually saturate the bond, which is why the table above reports
  `max_bond_reached` equal to χ in every row.
* **Believing a fast result.** `mps:auto` returning in 0.00 s looked like a fix
  and was an unimplemented CLI flag producing no output at all.
* **Trusting the trend instead of the bound.** The error that produced the wrong
  diagnosis was reading a convergence *rate* without checking the χ the circuit
  requires. Any future claim about MPS accuracy states 2^(n/2) alongside it.
