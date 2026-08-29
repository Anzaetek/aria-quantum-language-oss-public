<!-- SPDX-License-Identifier: Apache-2.0 -->
# Test stages beyond the default run

`./ci.sh` runs 9 core stages plus tch, Metal and OpenCL by default. Several
further stages exist but are **opt-in**, because they need tooling the default
run must not require (K13: headless, external tools skip cleanly).

This file is the durable record of what those stages are, what they need, and
what is set up on the development box — so a stage that has not run recently is
visible rather than quietly forgotten.

> ## POLICY (decided 2026-08-18): the Qiskit cross-check is RECOMMENDED, run PERIODICALLY
>
> This block used to say the check was MANDATORY and record the open question of
> whether `ci.sh` should FAIL rather than skip when the venv is absent. **That
> question is now answered: it should not.**
>
> The decision, so it is not re-litigated:
>
> * **Do not run it on every CI invocation.** It costs a Python venv and minutes
>   per run, and the edit/verify loop pays that for no signal on most changes.
> * **Do run it periodically during development** — before anything that touches
>   gate semantics, a parser or emitter lane, or a backend's numerics, and at
>   least once before any release or hand-off. It is the only INDEPENDENT
>   implementation checked against, and this project has already shipped a defect
>   that every *internal* agreement gate missed. Two Aria backends agreeing can
>   just mean they share a convention.
> * **A clean skip stays a clean skip.** That preserves K13 (headless machines
>   must be able to run CI) and it is why `ci.sh` still names every skipped stage
>   in its final summary — a green run that does not say what it did not do is
>   how coverage quietly erodes.
>
> The trade being accepted, stated plainly: a run without it is **weaker evidence
> than it looks**, and nothing in CI will stop you shipping on one. The cadence
> above is what makes that acceptable, and it is a human commitment rather than
> an enforced gate.

> ## POLICY (decided 2026-08-18): the QEC cross-check is MANDATORY **where the platform can run it**
>
> Deliberately a STRONGER rule than the Qiskit one above, and the asymmetry is
> the point rather than an oversight.
>
> **Why stronger.** Every other check here compares a computed *state* against a
> reference. This one compares a **decoder**. A decoder that is subtly wrong
> still decodes: it consumes syndromes, returns corrections, raises nothing, and
> the run is green. The error shows up only as a logical error rate that is off
> — a number nothing internal can produce a reference for. So the failure mode
> is silence, and the only thing that breaks the silence is an independent
> implementation of the same algorithm.
>
> **"Can run it" is measured, not assumed.** `ci.sh` makes the capability check
> itself — does `pymatching` import in the QEC venv — and reports three distinct
> outcomes:
>
> | outcome | meaning | CI |
> |---|---|---|
> | **RAN** | venv imports qiskit and pymatching | failure is fatal |
> | **INCAPABLE** | venv exists, `pymatching` does not import | clean skip, named as INCAPABLE |
> | **NOT RUN** | no venv, capability unverified | loud banner: a MANDATORY check did not run |
>
> Keeping those three apart IS the policy. Collapse them and the machine that
> can run the check produces the same output as the machine that cannot, and
> "mandatory" degrades into "optional" without anyone deciding to change it.
>
> INCAPABLE is a real state, not a courtesy: `pymatching`'s C++ extension does
> not build wheels on at least one aarch64 Linux host in this project. K13 says
> such a machine must still be able to run CI, so it skips cleanly — but it says
> INCAPABLE, which is a different sentence from "did not run".
>
> **Half-running counts as not running.** `tools/qec_cross_check/run.sh` used to
> drop the decoder half with a parenthetical `(skipping decoder cross-check —
> pymatching not installed)` and exit 0, so the quietest line in the output
> marked the loss of the only irreplaceable half. It is now a banner naming
> which half ran and which did not.
>
>     ARIA_QISKIT_XCHECK=1 ARIA_NWAY=1 ./ci.sh        # the recommended periodic run
>     ARIA_CUDA=1 ARIA_QISKIT_XCHECK=1 ARIA_NWAY=1 ./ci.sh   # ... on a CUDA box

Audited 2026-08-06 on Apple Silicon / macOS. `PREREQUISITES.md` covers install
details; this covers *what each stage buys you*.

| stage | enable with | needs | status here | last result |
|---|---|---|---|---|
| **Qiskit differential cross-check — RECOMMENDED, periodic** | `ARIA_QISKIT_XCHECK=1` | `.venv-qiskit` with `qiskit`, `qiskit-aer` | **INSTALLED** (qiskit 2.5.1 / aer 0.17.2) | **60 agree, 0 disagree, worst |Δp| = 4.441e-16** — re-measured 2026-08-06, `CI_EXIT=0` |
| **QEC cross-check — MANDATORY where capable** | `ARIA_QEC_XCHECK=1` | same venv + `pymatching` | **INSTALLED** (qiskit 2.2.3, PyMatching 2.4.0) | **re-measured 2026-08-18**, 32 assertions, 0 failed. Shot-for-shot logical-class agreement **100.00% (20000/20000)** in all four sectors — d=3 X and Z, d=5 X and Z. Logical rates within 3σ: d=3 X 0.0367/0.0367, d=3 Z 0.0336/0.0336, d=5 X 0.0251/0.0251, d=5 Z 0.0238/0.0238 (aria/PyMatching, p=0.05). All weight≤1 (d=3) and weight≤2 (d=5) errors correctable. Previous measurement 2026-08-06 quoted d=3 X only; the other three sectors were run then too but not recorded, which is why this row now names all four. |
| Lean 4 proof tree | `ARIA_LEAN=1` | `elan` + `lake`, warm mathlib cache | **READY** | 8281 jobs, exit 0 |
| TLA+ models | `tools/tla/check.sh` | JDK + `tla2tools.jar` | **READY** | safety holds (26 states); liveness violated — starvation, *expected* |
| CV ↔ piquasso fixture drift | `ARIA_CV_XCHECK=1` | `tools/cv_cross_check/.venv` with `piquasso` | **INSTALLED** (piquasso 8.0.1) | fixture matches live piquasso across 17 cases, worst 0.000e+00 — measured 2026-08-08. **This is a reproducibility number, not an agreement one**: it asks whether the recorded fixture still reproduces under live piquasso, so zero is expected. The Aria-vs-piquasso residual is **2.220e-16** (below) — quote that one for implementation agreement |
| N-way counts matrix | `ARIA_NWAY=1` | `.venv-qiskit` | **INSTALLED** | statevector / mps / noisy-mps(p=0) **14 of 14 fixtures compared, 0 disagree**; `pauli` 8 of 14 with 6 correct `cannot-express` refusals — re-measured 2026-08-11 after `sx`/`sxdg` landed natively. Found 3 live defects on its first run (see below) |
| N-way expectation matrix | `ARIA_NWAY=1` | `.venv-qiskit` | **INSTALLED** | statevector / mps **11 of 11 admitted, 0 disagree** at an analytic 1e-12 vs Qiskit `Statevector`; `pauliprop` **10 of 11**, `pauli` 5 of 11, rest correct refusals. 3 of **14** fixtures not admitted — re-measured 2026-08-17 |
| Bridge runner python tests | *(always, if a venv has pytest)* | any bridge venv + `pytest` | **INSTALLED** | 6 passed / 1 skipped from `.venv-qiskit`; 12 passed from `.venv-perceval`. Gates the stdout-protocol guard and the Perceval convention pins — neither had ever run in CI before 2026-08-11 |
| CUDA GPU backends | `ARIA_CUDA=1` | NVIDIA hardware | **N/A here — OWNED BY THE LINUX AGENT**, see `PLATFORM-OWNERSHIP.md` | untested on this box; `--features cuda` currently FAILS to compile by design (`GateKind::Sx`/`Sxdg` deliberately not added — see `CUDA_TODO.md`) |
| aria-py bindings | *(always; python tests need the venv)* | a CPython **&le; 3.13** (pyo3 0.23 cap) + `bindings/aria-py/.venv` with `maturin develop` for the python half | **INSTALLED** | cargo build clean; 8 python tests pass — 2026-08-12. A separate cargo project, so `cargo test --workspace` does NOT reach it. **Interpreter selection:** auto-detects a venv, then `python3.13`&hellip;`3.10`, then `python3` if &le; 3.13; override with `ARIA_PY_PYTHON=/path/to/python3.13`. No compatible interpreter &rarr; the stage SKIPS with a remedy; an interpreter you name explicitly but that cannot be used STOPS the run rather than silently auto-detecting past it. Full rationale in `ci.sh` above the stage |
| Deep harnesses | `ARIA_DEEP=1` | none (just slow) | available | `spectra_noise`, `spectra_scaling_noise` skipped in the default `all` |

## The Qiskit corpus has a hole — and it is the shape of the bugs

**Recorded 2026-08-08 after the cross-check passed 60/60 while three real
defects were live.**

`omega-xcheck` generates random circuits from **six Clifford gates**
(`H, S, Sdg, X, Z, CX`), and hardcodes two things on every case
(`crates/omega-xcheck/src/main.rs:8,20`):

```rust
classical_bit: None, condition: None
mid_circuit_mode: MidCircuitMode::Skip
```

So the sampled space contains **no measurement, no classical conditions, no
Reset, and no non-Clifford rotations**. All three defects found on 2026-08-08
live outside it:

| defect | why the corpus missed it |
|---|---|
| expectation silently skipped feedforward | no conditionals generated |
| MCM sampled once per run, not per shot | `Skip` mode; no measurement generated |
| QASM export dropped `if(c==V)` | no conditionals to export |

`4.441e-16` over 60 cases was **genuine agreement over a region where nothing
was wrong**. That is not a broken check — it is a check whose *corpus* does not
span the feature space, which reads identically to a thorough one from the
summary line.

**What the corpus needs**, in rough order of what it would have caught:

1. **Mid-circuit measurement with `MidCircuitMode::Collapse`**, compared on
   *distributions* over many shots rather than on an analytic vector — the
   defect was in the sampling, so an analytic comparison is structurally blind
   to it.
2. **Classically-conditioned gates** (`when c == v`), including the case where
   the condition is **sometimes false**. A condition that is always true hides a
   dropped guard: both engines agree because both lost the same thing.
3. ~~**`Reset`**~~ — **CLOSED 2026-08-18.** 8 circuits in the
   distribution-compared corpus (`|1⟩`-then-reset, superposition-then-reset,
   reset of half a Bell pair so PARTNER COLLAPSE is exercised, and
   reset-then-reuse). `16 agree, 0 disagree, worst TVD = 0.0147` alongside the
   feedforward corpus. Verified by mutation: dropping the resets gives
   `8 disagree, worst TVD = 1.0000`.

   Each circuit proves its own non-vacuity — the generator re-runs it with the
   `Reset` ops stripped and requires the distributions to differ by TVD ≥ 0.2,
   because `reset` on a qubit already in |0⟩ does nothing and a corpus of those
   passes whether or not Reset is implemented. That check rejected two of the
   first eight circuits at TVD 0.0103 and 0.0085: `H; CX; reset; H` and
   `H; CX; H` both end uniform over four outcomes.
4. ~~**Non-Clifford rotations**~~ — **CLOSED 2026-08-18.** `Rx/Ry/Rz/U3/CRz/CU3`
   at generic angles, one rotation circuit per Clifford circuit, `40 agree, 0
   disagree, worst |Δp| = 4.995e-13` at n=20+20. Angles are never a multiple of
   π/4, which is where a phase-convention error cancels, and the circuits open
   with `H` on every qubit — a rotation on |0…0⟩ is largely invisible in
   probabilities and `Rz` is entirely invisible.
5. ~~**Round-trip through QASM**~~ — **CLOSED 2026-08-18.**
   `aria-runtime --bin qasm_roundtrip_xcheck` + `compare_qasm_roundtrip.py`:
   `12 agree, 0 disagree, worst TVD = 0.0119`. Aria's NATIVE execution is the
   reference; Qiskit runs the exported text. Verified by mutation — deleting the
   `if (m == 1)` guards from the exported QASM gives `0 agree, 12 disagree, worst
   TVD = 0.5124`.

Item 5 is the one worth emphasising: a differential test conducted **through**
an export can agree precisely because the export dropped the same thing on both
sides. The 2026-08-08 QASM defect was found only by comparing Aria's native
execution against Qiskit running Aria's exported text.

## Why each one is worth running

**QEC cross-check** — differential-tests the exact MWPM decoder against
**PyMatching**, shot for shot rather than only in aggregate. Aggregate logical
rates can agree while individual decodings differ, so the 100% shot-for-shot
class agreement is the stronger claim.

It is the one check here whose absence cannot be inferred from the output of
anything else, because a wrong decoder is *silent* — it returns corrections and
the run goes green. That is why it is MANDATORY where the platform can run it,
while the Qiskit cross-check above is only RECOMMENDED.

**Qiskit cross-check** — the only *independent implementation* available. When
Aria and Qiskit agree to 1e-16, that is evidence; when two Aria backends agree,
it may only mean they share a convention. This project has already been bitten
by exactly that: the `Reset` defect survived every cross-backend "agreement"
gate because each pair happened to coincide in the basis being checked. This is
the highest-value optional stage, and the only one needing an install.

**Lean 4** — enforces sorry-free on the shipped circulant, noise-channel and
quantum-linear-algebra theorems (via `#print axioms` greps for `sorryAx`).
`lake build` alone does *not* error on `sorry`, so the axiom check is the real
gate. Known-open: `Reset.lean` and `StabilizerExpectation.lean` carry 7 `sorry`
targets — they need a real ordered field, which core Lean lacks. Those are
recorded, not regressions.

**CV ↔ piquasso** — the same independence argument as Qiskit, applied to
continuous-variable simulation. Split deliberately in two halves that fail for
different reasons:

* `cargo test -p omega-backend-cv` compares against a **committed** fixture, so
  it runs on every default CI with no Python. It catches regressions in our code.
* `ARIA_CV_XCHECK=1` reruns piquasso live and checks the fixture has not
  **drifted**. This is the half the Rust test structurally cannot do: if our
  conventions changed and the fixture were regenerated to match, the committed
  numbers would record our own opinion rather than an independent one, and the
  test would stay green. Verified to fire on a 1e-9 nudge — well below the Rust
  side's own tolerance, so the two halves genuinely overlap rather than duplicate.

**Amplitudes are compared, not just probabilities**, and that is load-bearing.
`kerr` and `phase_shift` are diagonal — `exp(i·χ·n²)` and `exp(i·φ·n)` — so they
move phases and nothing else. Against a probability vector, a no-op
implementation of either is *indistinguishable* from a correct one, and the
squeezing sign convention is equally invisible. Both were confirmed by mutation:
no-op'ing Kerr is caught at 4.9e-1 and flipping the squeezing sign at 1.4e-1,
both by the amplitude assertion, and neither would have moved a single
probability. Global phase is quotiented out against the largest-magnitude
component, since it is unobservable and the two libraries have no reason to agree
on it.

**Two holes, recorded rather than papered over:**

1. **Single mode only.** `FockState` is single-mode — there is no beamsplitter
   and no multimode state — so mode mixing, which is exactly where independent CV
   implementations diverge on convention, is untested because it is
   *unimplemented*.
2. **Squeezing is a constructor, not an operator.** ~~There is no
   `displace(&mut self, ..)`~~ — **that landed**, and this entry said otherwise
   for some time after. `displace` now has an operator form, so *squeeze then
   displace* is expressible and is cross-checked against piquasso
   (`displace_operator_matches_piquasso_on_a_squeezed_state`). The
   `NOT COMPARED` set the pin guards is now **empty**: every case in the corpus
   is expressible.

   What remains is `squeeze`, still available only as `squeezed_vacuum`
   building from vacuum. A corpus case applying squeezing to a state that
   already has structure would reappear in the pinned set — which is the
   mechanism working, not a hole.

   *Recorded because the stale version was quoted as current in a plan.* The pin
   (`preparations_are_still_constructors_only`) did its job and failed when the
   operator form landed; this prose did not, because nothing checks prose
   against the tree.

**A correction, recorded because the wrong version shipped.** This section used
to say the two implementations "genuinely differ" on squeezed vacuum — that Aria
takes closed-form amplitudes and cuts while piquasso applies an already-truncated
operator, "two defensible readings" — and that disagreements were held to the
backend's own `lost_norm` budget (4.7e-5 against a 6.3e-5 budget at `r=0.8`).

**That explanation was wrong in both halves.** The underlying amplitudes agree to
**1e-16**, so there was no difference of state; and piquasso does *not*
exponentiate a truncated generator (that disagrees by 1.8e-2 at `r=1.0`) — it
applies the truncation of the **exact** operator, via the Miatto–Quesada
recurrence.

The whole gap was a **normalisation convention**: piquasso returns raw truncated
probabilities (`sum = 0.999936664825` at `r=0.8`), Aria renormalises by the
represented mass. Renormalising piquasso's own vector reproduces **3.434e-08**
and **4.736e-05** — exactly the numbers the check reported as "reconciled by the
truncation budget". It passed, and its bound was even valid, but **for a
different reason than documented**, at a tolerance ~10¹⁰ too loose.

Both sides are now normalised once in `piquasso_ref.py`, and the tolerance is a
plain 1e-14. Measured worst residual across the corpus: **2.220e-16**. The
tightening is not cosmetic — it now catches a **1e-11 relative error in the Kerr
phase**, which the old budget-sized tolerance passed without complaint.

`lost_norm` and `lost_n_weight` still matter, but they bound the distance to
**truth** (`sinh²r`), not to piquasso. Their proper home is the analytic test in
`lib.rs`, not the differential one.

**TLA+** — properties that hold over *all* interleavings rather than one run.
Already earned its keep: `Governor.tla` produced a concrete lasso proving a
large job can be starved indefinitely under `try_acquire` with no queue, which
is the design input for the bounded queue. Note the liveness check is **expected
to fail** until that queue exists; see `proofs/tla/README.md`.

**CUDA** — the only way to exercise the CUDA statevector, MPS `gesvdj`,
pauliprop, RBS, and the on-device Born sampling for the Reset channel. Also the
only way to exercise A7b's device pools and f32 pricing against real discrete
hardware; so far they have only met the OpenCL path.

**Deep harnesses** — `ARIA_DEEP=1` adds the two spectra-noise apps the default
`all` skips for runtime. Worth running before a release.

## What `ARIA_NWAY=1` found on its first run

**Recorded 2026-08-09.** Three live defects, all invisible to every check that
existed before it. The details are in `FIXES_PLAN.md` K7; the point for this
document is *why* they were invisible.

1. **`MpsBackend` replayed one trajectory for every shot.** Its per-shot loop
   was guarded on `circuit_has_reset` alone, so a collapse-mode measurement took
   the single-evolution path. On `12_feedforward_sometimes_false.qasm` it
   returned `{0: 20000}` — certainty — where Aer gives ~50/50. This is the exact
   defect the statevector backend fixed in `11888a9`, which `NoisyMpsBackend`
   already carried and which never propagated here.
2. **`MpsBackend` never keyed by the creg** in collapse mode.
3. **`PauliBackend` evaluated the guard correctly, then re-measured every
   qubit** and keyed over the full register — key `3` for a 1-bit creg.

None of the three could be caught by an internal comparison. The broken MPS
backend agreed with itself perfectly on every rerun, and the *correct* MPS
backend was never placed beside it on a conditional circuit. That is the
concrete version of this file's standing caveat: agreement between our own
engines is weak evidence, and it stays weak no matter how many of them agree.

The matrix also caught one defect in **itself** — it rendered counts keys
MSB-first where the bridge wire format is LSB-first — and that one is worth
noting because of how it survived: the wrong order is *correct on every
palindromic outcome*, and `{00, 11}` (Bell, GHZ, partial-measure) is
palindromic. The first key-conversion test passed against the bug it was
written to catch. Anything asserting a bit order needs an asymmetric fixture.

## Standing caveat

Every one of these is a *differential* or *formal* check, and neither kind can
tell you an assumption is wrong. A model verifies a protocol against the weights
it is handed; a cross-check verifies agreement between two implementations of
the same idea. The governor defect that mattered most this year — pricing MPS by
its tensors while the backend contracted to a dense statevector — would have
passed both. That class stays closed by reading what the code actually
allocates. See `proofs/tla/README.md` for the longer version.
