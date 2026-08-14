<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — fixing the MPS backend (revision 2)

**Status: PLAN, revised after adversarial review.** Revision 1 reached a correct
conclusion by the same method that produced the wrong diagnosis before it —
asserting from a bound instead of measuring — and three of its four action items
were aimed at things that are not true. Both are recorded rather than quietly
replaced.

## The conclusion, and how it is actually established

There is **no MPS kernel bug**. `wide_chain_19q` is genuine truncation loss.

Revision 1 argued this from "19 qubits ⇒ max Schmidt rank 2⁹ = 512, so χ=64 is
⅛ of exact". That is a worst-case *bound*, not a property of this circuit, and
asserting from it is exactly the error being corrected. The measurement that
settles it, done in review by profiling the Schmidt spectrum along the actual
split schedule:

* the exact state reaches Schmidt rank **512** at cut 8 — it does saturate the
  bound, which revision 1 assumed rather than checked;
* **39% of Schmidt weight sits beyond rank 64** at the worst split;
* a canonical-gauge TEBD — the best *any* χ-capped MPS can do — still gives TVD
  **0.459 at χ=64** and **0.351 at χ=128**.

No gauge, no kernel and no implementation returns a good answer at χ ≤ 128 on
this circuit. The flat TVD trend is what genuine truncation looks like here.

Also corrected: the circuit is not "deep enough to scramble" as revision 1 called
it. It is a structured compute–uncompute ladder measuring **two qubits**.

### The user-visible output is worse than the TVD suggested

Revision 1 quoted TVD 0.51 over the full 2¹⁹ distribution — which the circuit
never measures. On the two measured bits the exact marginal is
`(0.876, 0.124, ~0, ~0)`; at χ=64 we produce `(0.456, 0.061, 0.425, 0.058)` —
**48% of the weight on an outcome of probability ~1e-30**. The refusal is
justified on what a user actually sees, which is a stronger statement than the
one that was made.

## What is actually wrong

### M1 — silently approximate answers *(landed, and it regressed the other CLI)*

The truncation certificate gated nothing, so a run that discarded 6.586 of the
state returned that distribution. Now gated.

**The gate broke `aria --strict-truncation`.** `make_mps` never lifted the
backend ceiling, so the backend refused at 1e-6 before the CLI's own check ran
and the documented "I accept this much approximation" knob could no longer
accept anything — measured: `--backend mps:2 --strict-truncation 0.5` errored at
1e-6 instead of accepting. Stats are recorded only on success, so the refusal
also discarded the certificate the CLI wanted to print.

Fixed by moving the policy to one place: the backend ceiling is lifted in
`aria-runtime`, and `report_mps_truncation` applies
`DEFAULT_MAX_DISCARDED_WEIGHT` when no flag is given and the user's value when
one is. Verified across all three paths — accept, refuse-at-user-bound,
refuse-at-default — with the certificate printed in each.

### M2 — the CLI knob *(premise corrected; smaller than revision 1 claimed)*

Revision 1 said "`--help` and the docs advertise `mps:auto`/`mps:<chi>`, and
`with_adaptive` is unreachable from the CLI". Both false, and they conflate two
binaries:

* `omega-run --help` lists only `statevector, mps, pauli, photonics, pauliprop`
  — it advertises nothing it rejects. The document that describes the grammar
  describes the **`aria`** CLI, where the knob **works today**.
* `with_adaptive` is reachable: `aria run --backend mps:auto`.

So the real gap is narrow: **`omega-run` alone cannot select χ**, while `aria`
can. And the fix is not to write a parser — `BackendSel::parse` already
implements the whole grammar with its defaults. Writing a second one at
"several sites", as revision 1 proposed, is the inconsistency mechanism it
warned about.

**Do:** have `omega-run` use the existing `BackendSel::parse`. Sites revision 1
missed: the noise-model gate `matches!(chosen, "statevector" | "sv" | "mps")`
(a `mps:512 --noise` run would be spuriously rejected), the `--list-backends`
names, and the `NoisyMpsBackend::with_model(64, …)` construction.

### M3 — **withdrawn: the mechanism does not exist**

Revision 1 claimed "with a fixed χ every split is contracted at full bond even
when the Schmidt spectrum is nearly rank-1". `svd.rs` already drops σ below
threshold, so bonds *do* collapse on trivial spectra — and revision 1's own
evidence contradicted it, since `qft_16` reaching `max_bond_reached=1` is that
mechanism working. Measured: `qft_16` on fixed `mps` runs in **0.005 s**.

Adaptive mode would not help the circuit in question either: its ranks go
2, 4, 8, … 512, saturating any ceiling immediately, so `mps:auto` would be
*slower*.

Where the time actually goes: 135 CX become **1555 splits** through the swap
network, because a distance-17 CX costs 33 splits. At saturated χ=64 each split
is ~250 Mflop of scalar Jacobi, giving ≈46 s — consistent with the measurement,
so there is no separate hidden performance defect.

**The lever nobody named is qubit ordering.** An interleaved mapping
(0, 8, 1, 9, …) cuts the required rank from 512 to **265**. That is the real
performance item, and it is a different piece of work from anything revision 1
proposed.

### M4 — the certificate is uncalibrated and its documentation overclaims *(new)*

`discarded_weight > 1` is legal — it is a sum over ~1555 per-split fractions,
not a probability — so 6.586 is not an accumulation bug. Two real problems:

* the doc comment says "the discarded weight bounds the infidelity". That
  theorem holds for **canonical-gauge** truncation, and this MPS is explicitly
  non-canonical. Measured on the same trajectory: canonical certificate 0.664
  vs ours 6.402 — a 10× gauge inflation. Conservative *here*; nothing proves it
  cannot understate elsewhere, which would let a bad run pass the 1e-6 gate.
* the message "discarded 6.586e0 of the state" is innumerate — you cannot
  discard 659% of a state. Reword to name it as an accumulated per-split sum.

### M5 — the refusal costs a full run — **FIXED 2026-08-14**

42.9 s -> 0.63 s on `wide_chain_19q`. The certificate is monotonically
non-decreasing, so stopping at the first crossing changes only how long a
refusal takes, not which runs are refused — verified against the actual set
rather than left as an argument.

Original description follows.



The check fires after evolution completes: 46 s of compute, then the error. On
the per-trajectory path it fires after `shots ×` full evolutions. The
certificate crosses 1e-6 within the first truncating splits, so an early abort
would refuse in milliseconds.

## Tests — the gap revision 1 lectured about and then left open

Revision 1's exactness tests run at χ = 2^(n/2), where ~1e-32 is dropped. The
rank-cap bookkeeping runs but **no meaningful σ is ever discarded**, so the
lossy path — the only path anything interesting happens on — stays untested. A
wrong-column repack or a gauge pathology would pass every test proposed.

**Required:**
1. Exact-bond equivalence (n = 8, 10, 12 at χ = 2^(n/2)), with
   `max_bond_reached == χ` asserted so the fixture provably saturates. Keep;
   move out of a scratch file into the suite.
2. **A test where truncation is active and meaningful** — matched-χ comparison
   against a canonical-compression reference on a mildly-truncating circuit.
   This is the missing one. Measured residual today: ours 0.505 vs canonical
   0.459 at χ=64, and 0.474 vs **0.351** at χ=128 — real, modest here,
   unbounded in principle.
3. `wide_chain_19q` as a **refusal** fixture. Not as a "correct at χ=512"
   fixture: χ³ scaling puts that at ~6 hours.

## What could make this pass for the wrong reason

* **Trusting a bound instead of measuring.** This is now twice. Any claim about
  what χ a circuit needs is measured from its Schmidt spectrum, not inferred
  from 2^(n/2).
* **Testing only where truncation cannot happen** — see the tests section.
* **Believing the external comparison.** The 0.027 s reference this all started
  from is almost certainly a wrong answer: no faithful MPS tracks rank 265–512
  in 27 ms, and canonical TEBD at χ=128 already has TVD 0.35. The report shows
  no counts. Before any performance target is set against that number, it must
  be checked for correctness — an unfaithful fast result is not a target.
* **Quoting a TVD over states the circuit never measures.** Report the marginal
  over the measured register.
