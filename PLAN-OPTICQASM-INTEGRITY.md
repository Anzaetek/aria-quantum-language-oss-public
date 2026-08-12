<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — OPTICQASM export/import integrity

**Status: O1, O2, O3 and O5 are IMPLEMENTED (2026-08-13). O4 and O6 remain
open.** Written as a plan first, after the measurements below and before any
code; this header is the only part edited after the fact, so the reasoning
below is what was decided *before* the work, not a retrospective. Companion to `PLAN-EXPORT-INTEGRITY.md`,
which covers the same defect class on the qubit (QASM2/QASM3/Aria) side; this
document is P7 of that plan, expanded once it turned out to be seven defects
rather than two.

Governing principle, restated:

> **Output claiming to be language X must be readable by things that speak
> language X.** Otherwise emitting X has no point — it is a private format
> wearing a standard's name.

---

## 1. Who actually speaks OPTICQASM

This is the first question, and for OPTICQASM the answer is different from
QASM2's — which changes every conclusion, so it is settled first.

QASM2 has an external arbiter: Qiskit. That is what made `ryy` unfixable by
adding a reader (see `PLAN-EXPORT-INTEGRITY.md`) — teaching our parser to read
a name Qiskit rejects would make the round trip work *for us alone*.

OPTICQASM has **no external arbiter** — it is this project's own dialect. So
"readable by things that speak the language" reduces to two questions:

1. Can our own readers read it?
2. Do the **operations** it names exist in the emulators a photonic circuit
   would realistically be handed to?

Question 2 is the one that decides whether a gate belongs in the language at
all, and it was never asked. Measured against the two installed emulators
(`perceval 1.2.4` in `crates/omega-bridges/python/.venv-perceval`,
`piquasso 8.0.1` in `.venv-piquasso`):

| OPTICQASM spelling | Perceval (DV, linear-optical) | piquasso (CV) | our own CV backend |
|---|---|---|---|
| `ps` | `PS` ✓ | `Phaseshifter` ✓ | `.phase_shift()` |
| `bs_rx` | `BS` ✓ | `Beamsplitter` ✓ | — |
| `hwp` | `HWP` ✓ | n/a — polarization is DV | — |
| `pbs` | `PBS` ✓ | n/a | — |
| `squeeze` | n/a — not linear-optical | `Squeezing` ✓ | `squeezed_vacuum()` — **prep only**, see §O3 |
| `displace` | n/a | `Displacement` ✓ | `.displace()` |
| `kerr` | n/a | `Kerr` ✓ | `.kerr()` |

**Every gate OPTICQASM names has a direct counterpart in a major emulator.**
The operation set is genuinely common; nothing here is invented.

That reverses the conclusion the first measurement suggested. `squeeze`,
`displace` and `kerr` are **not** junk to be removed from the emitter — piquasso
implements all three, and so does this repository's own `omega-backend-cv`. The
only thing in the world that cannot read them is **our own OPTICQASM lowering**.
Deleting them from the emitter would destroy real interoperability to make a
test pass.

**Gates the emulators have and OPTICQASM does not** (recorded, deliberately out
of scope — this plan closes gaps, it does not grow the language):
Perceval `QWP`, `PR`, `PERM`, `TD`, `LC`, `WP`, `Unitary`;
piquasso `CrossKerr`, `CubicPhase`, `Squeezing2`, `MachZehnder`, `Interferometer`,
`Fourier`, `Loss`, `Attenuator`.

---

## 2. The seven measured defects

Every row was **executed**, not read. `D1` was found by a probe through
`omega_parser::lower_to_ir`; `D4`, `D5`, `D7` by a probe through
`aria_core::ast::opticqasm` (both temporary, removed after measuring).

| # | defect | measured behaviour | severity |
|---|---|---|---|
| **D1** | `omega-parser` rejects the CV gates `to_opticqasm` writes | `unknown photonic gate: squeeze` | export unreadable |
| **D2** | `hwp` / `pbs` readable but **never emitted** | 0 references in the emitter | one-way round trip |
| **D3** | the `pol` marker is never emitted | always `photon q[N];` | **silent mis-semantics** |
| **D4** | unsupported gates become comments | `H` → `// unsupported: H q[0];`, re-parses to **0 ops** | **silent drop** |
| **D5** | non-numeric parameters become `0.0` | `ps(θ)` → `ps(0) q[0];` | **silent wrong number** |
| **D6** | `bs` / `bs_ry` parse but do not lower | in the grammar, absent from `lower.rs` | dead spelling |
| **D7** | paren-less gates and `pol` decls silently skipped by the reader | `photon q[2] pol; pbs q[0],q[1];` → **`Ok`, 0 regs, 0 ops** | **worst: success reported for an empty circuit** |

### The three that are worse than D1

D1 is loud — the parser refuses and says why. Nothing downstream believes a
wrong number. The dangerous ones are the quiet ones:

**D7** is the worst defect in this file. A complete, grammatically valid
polarization circuit returns `Ok` with **zero registers and zero operations**.
Not an error, not a partial parse — a successful parse of nothing. Cause:
`from_opticqasm` matches lines with two regexes, `photon\s+(\w+)\[(\d+)\];`
(which the `pol` marker breaks) and `(\w+)\(([^)]*)\)\s+(.+);` (which requires
parentheses, so the parameter-less `pbs` never matches). A line matching
neither falls off the end of the loop with no `else`. The grammar explicitly
permits both forms — `pol_marker?` and `("(" ~ param_list ~ ")")?` — with
comments saying so.

**D5** is the only defect that yields a *wrong number* rather than a missing
one. `.try_as_f64().unwrap_or(0.0)` appears at 8 sites. A symbolic angle
becomes a hard zero, and the emitted file is well-formed, parses cleanly, and
simulates to confident garbage. Note `aria_emit.rs:109` has the identical
`unwrap_or(0.0)` — same defect, different emitter, out of scope here but
recorded.

**D4** is the RESET defect verbatim (`PLAN-EXPORT-INTEGRITY.md` P1a), one door
over. `to_opticqasm` returns `String`, not `Result` — exactly the shape
`to_qasm` had before Part H — so there is no channel to refuse on and the
fallback arm writes a comment. Unlike the Aria `--` bug the comment *does*
re-parse (`//` is the grammar's comment syntax), which is worse: the file
round-trips successfully into a **different circuit**.

**D3** is the subtlest. `photon q[N] pol;` means N *spatial* modes each carrying
H and V — 2N optical modes, indexed `2s+p`. `photon q[N];` means N optical
modes. The emitter always writes the second. So even after D2 is fixed, a
polarization circuit would emit `hwp`/`pbs` under a declaration asserting no
polarization, and every mode index would silently mean something else. No parse
error, no refusal, different physics.

---

## 3. Design decision: CV must NOT become an `omega-core` `GateKind`

The obvious fix for D1 — add `GateKind::Squeezing` / `Displacement` / `Kerr` to
`omega-core`, as `Sx`/`Sxdg` got — is wrong, and measurably so.

* **Cost:** 17 files match exhaustively on `GateKind`, including all three GPU
  backends. `omega-backend-statevector-cuda` is Linux-owned
  (`PLATFORM-OWNERSHIP.md`) and would go non-exhaustive again — re-creating on
  purpose the exact breakage that file exists to prevent.
* **Benefit: zero.** `omega-backend-cv` **does not depend on `omega-core`**
  (verified: no such dependency in its `Cargo.toml` or `lib.rs`). It is a
  standalone crate driven by a Rust API — `.phase_shift()`, `.displace()`,
  `.kerr()` — and has no text front end whatsoever. Adding CV variants to the
  DV IR would pay the whole backend surface and still not reach the executor
  that implements them.

The truthful model is that **OPTICQASM has two profiles**:

* a **DV profile** (`ps`, `bs_rx`, `hwp`, `pbs`) that lowers to `omega-core` IR
  and runs on `omega-backend-photonics` / Perceval;
* a **CV profile** (`squeeze`, `displace`, `kerr`) that belongs to
  `omega-backend-cv` / piquasso and **cannot** be expressed in the DV IR.

So `omega-parser`'s refusal of CV gates is **correct**. The defect is only that
it refuses with `unknown photonic gate: squeeze` — which is false. The gate is
perfectly well known; it is *not lowerable to this IR*. The message must be a
**route**, not a denial.

The repository already does exactly this one door over:
`aria-core/src/backends/omega.rs:294–310` refuses `Squeezing`/`Displacement`/
`Kerr` on the execution path with a message naming them as continuous-variable
gates, with a regression test at line 736. The parser simply does not use the
pattern its sibling already established.

---

## 4. Fix plan

**Landed so far:**

| step | state | evidence |
|---|---|---|
| O1 emitter refuses instead of corrupting | **done** | `opticqasm.rs` returns `Result`; D4/D5 tests, both mutation-verified |
| O2 reader refuses instead of skipping | **done, after a false start** | statement-based, not line-based — see below; `opticqasm_reader_agreement.rs` |
| O3 CV gates **import** | **done** | `omega_parser::lower_opticqasm_cv`; 8 unit tests + 5 cross-crate, 6 mutations verified |
| O4 emit `hwp`/`pbs` + `pol` | **open** | needs new `GateKind`s in `aria-core` — see below |
| O5 cross-backend agreement | **done (CV)** | **17/17 piquasso fixture cases agree**, amps 1e-13, probs 1e-14, **0 skipped**; DV/Perceval arm still open |
| O6 exhaustive-`match` guard | **open** | |

### What the adversarial review caught, after these were called done

Both are recorded because the plan's §5 predicted this class and the work fell
into it anyway.

**O2's first attempt re-committed the defect one level down.** It iterated over
`src.lines()`. The grammar's `WHITESPACE` includes `\n`, so a statement may
share a line with any other, and:

* `OPTICQASM 1.0; photon q[2]; ps(0.5) q[0];` on one line → **`Ok`, 0 registers,
  0 operations** — D7 verbatim, in the function whose doc comment claimed to
  have killed it, because `line.starts_with("OPTICQASM")` skipped the whole line
  rather than the header token;
* `ps(0.5) q[0]; ps(0.7) q[1];` on one line → **`Ok`, ONE gate**, parameter
  `0.5`, on modes `[0, 1]` — the two were *merged*, which is worse than a drop.

The reader now splits on `;` after stripping comments, which is what the grammar
does, and validates registers, mode ranges and both arities. Five inputs where
the two readers disagreed — always with `aria-core` silent — are pinned by
`opticqasm_reader_agreement.rs`.

**O5's acceptance test could not fail for the operation it most needed to
check.** It asserted `compared >= 10`. The fixture has 17 cases, 7 of which use
`squeeze`, and an executor error became a *skip* rather than a failure — so the
floor was set to exactly the number that survives when squeezing is entirely
broken. Measured: swapping `r` and `phi` in the CV import left the test **green**
at "10 cases agree, 7 skipped". It now asserts the skip list is empty and the
compared count equals the case count, and that same mutation fails it.

**One instruction in this plan was wrong.** §O3 said `bs_ry` "comes out of the
grammar". Removing it makes the diagnostic *worse*: `gate_name` is an ordered
choice over an atomic rule, so the earlier `"bs"` alternative then matches the
prefix and leaves `_ry` dangling — measured, a clear `unknown photonic gate:
bs_ry` degrades to pest's `--> 3:3`. The spelling stays tokenizable and the
lowering now says what is true: named by the grammar, implemented nowhere.

O4 turned out larger than this plan assumed: `aria-core`'s `GateKind` has **no
polarization variants at all** (no `HalfWavePlate`, no `PolarizingBeamSplitter`),
so emitting `hwp`/`pbs` means adding them to the AST first, plus the register
flag D3 needs. `from_opticqasm` therefore refuses both by name for now, saying
which crate does implement them, rather than reporting "unknown gate".


Ordered so that each step is verifiable on its own and none depends on a later
decision.

### O1 — `to_opticqasm` stops corrupting silently *(fixes D4, D5)*

Change the signature to `Result<String, String>`, mirroring `to_qasm` after
Part H. Then:

* the `_ =>` arm **refuses** instead of writing `// unsupported: {:?}`, naming
  the gate and saying OPTICQASM is a photonic dialect;
* every `try_as_f64().unwrap_or(0.0)` **refuses** on a non-numeric parameter
  instead of substituting zero. A symbolic angle is a real limitation of a
  format with no parameter syntax for it — but it must be *stated*, never
  silently set to 0.

Callers of `to_opticqasm` must be updated; `Barrier` keeps its comment (it has
no operational meaning to lose).

### O2 — `from_opticqasm` stops accepting what it cannot represent *(fixes D7)*

* Parse the optional `pol` marker and record it on the register.
* Accept parameter-less gate applications (`pbs q[0], q[1];`).
* **No line falls through.** Any non-empty, non-comment, non-header line that
  matches nothing is an error. This is the actual fix — the specific regexes are
  incidental, the missing `else` is the defect, and it will otherwise recur the
  next time the grammar grows.

### O3 — `squeeze` / `displace` / `kerr` become IMPORTABLE *(fixes D1)*

Not "refuse with a better message" — **import them**. They are real operations
with real executors (§1); a refusal, however politely worded, still leaves the
CV half of the language unreadable.

The import target cannot be `omega-core`'s `Circuit` (§3), so OPTICQASM gets a
second lowering:

```
omega_parser::lower_opticqasm_cv(src) -> Result<CvProgram, String>
```

with `CvProgram { modes, ops: Vec<CvOp> }` and
`CvOp::{PhaseShift, Squeeze, Displace, Kerr, BeamSplitter}`, defined **in
`omega-parser`** — not in `omega-backend-cv`, which deliberately carries a
single runtime dependency (`num-complex`) so it stays embeddable
(`docs/LIBRARY.md`). The dependency edge runs parser → nothing; the executor
adapter lives on the consumer side.

`lower_to_ir` keeps its DV behaviour, but its CV arm stops saying "unknown
photonic gate" — a false statement — and instead names the gate as
continuous-variable and points at `lower_opticqasm_cv`. Wording follows
`aria-core/src/backends/omega.rs:294–310` so the two agree.

**Measured limits of the executor, which the import must not paper over.**
`omega-backend-cv` is *single-mode*: `FockState` is one mode with a cutoff, and
it exposes `phase_shift`, `displace` and `kerr` as operations but offers
squeezing only as a **state constructor** (`squeezed_vacuum`), with no
beamsplitter at all. So:

* **import** accepts all five ops on any mode count — importing is reading, and
  a file must not become unreadable because our executor is narrow;
* **execution** refuses, explicitly, what this backend cannot do (multi-mode,
  or `squeeze` anywhere but as the first op on a mode). That refusal belongs in
  the adapter, and it must name piquasso, which has all of them.

Keeping those two separate is the whole point. Conflating them is what produced
D1: an executor limitation was written into the *reader* as if the gate did not
exist.

Also in this step:

* `bs` / `bs_ry` → either lower them or remove them from the grammar's
  `gate_name` (D6). `bs` is an accepted alias in `aria-core::from_opticqasm`, so
  **lower `bs` as `bs_rx`**; `bs_ry` has no reader anywhere and comes out of the
  grammar.
* anything else → the existing unknown-gate error.

### O4 — emit polarization *(fixes D2, D3)*

`to_opticqasm` learns `hwp` and `pbs`, **and** emits the `pol` marker on any
register carrying them. D3 makes these one change, not two: emitting the gates
without the marker is worse than emitting neither.

Requires a way to know a register is polarization-carrying — decide when
implementing; `aria-core`'s `Circuit` has no such flag today, and inventing one
is the only genuinely new design in this plan.

### O5 — hand-over tests against the real emulators

**The acceptance criterion for this entire plan: one OPTICQASM source, executed
on different backends, computes the same numbers.** Not "every backend accepts
it" — accepting is trivial and a wrong beamsplitter convention accepts
perfectly. The claim is numerical agreement, and if two backends disagree the
plan has failed regardless of how many tests are green.

That is what makes emitting a portable dialect worth anything: the file is a
description of a computation, and a description whose meaning changes with the
reader is not one. It is also the one property none of the seven defects above
could ever have been caught by a parse check — D3 and D5 both produce files
that every reader accepts and that compute something different.

The point of the whole exercise, and what the operation-set table above exists
to justify:

* **DV:** emit OPTICQASM → drive Perceval through the existing
  `perceval_runner.py` → compare against `omega-backend-photonics`. Conventions
  are already pinned by `tests/test_perceval_conventions.py`.
* **CV:** emit OPTICQASM → build the equivalent piquasso program → compare
  against `omega-backend-cv`. The comparison machinery exists
  (`tools/cv_cross_check/`, `piquasso_xcheck.rs`); what is missing is that
  **OPTICQASM text is not on that path today** — the CV cross-check is driven
  from Rust. Putting the text on the path is the test.

Both run opt-in, under the existing `ARIA_BRIDGE_XCHECK` / `ARIA_CV_XCHECK`
gates, and both must appear in `skipped()` when they do not run.

Agreement is asserted on **distributions and expectation values**, at a stated
tolerance with a stated reason for that tolerance — following the L2 gate
already used by the N-way matrices (`K·√(Σ p̂ₖ(1−p̂ₖ)(1/nₐ+1/n_b))`) for
sampled comparisons, and a tight absolute tolerance for analytic ones. A
tolerance chosen to make a test pass is not a tolerance.

### O6 — the guard that stops this class recurring

An **exhaustive `match` over `GateKind`** in a round-trip test — the same
mechanism as `PLAN-EXPORT-INTEGRITY.md` P3, and for the same reason: a new
variant then fails to compile until someone decides what OPTICQASM does with
it. Not a source-scraping guard; scraping cannot see D3, D5 or D7 at all.

---

## 5. What could make this pass for the wrong reason

This section is not boilerplate. Six of the defects above survived an existing
test suite, and the recurring failure in this repository is a check that passes
without exercising what it names.

* **`test_opticqasm_roundtrip` already exists** (`opticqasm.rs:212`) and asserts
  `contains("ps(0.5) q[0];")` and `contains("bs_rx(1.2, 0.3) q[0], q[1];")`. It
  passes today with all seven defects live, because its fixture uses only the
  two gates that work, only numeric parameters, and no polarization. **Any new
  fixture must include at least one gate from each profile, one symbolic
  parameter, and one polarization declaration**, or it will be this test again.
* **Substring assertions do not re-parse.** Every check here must round-trip
  through the actual reader — that is what `contains(...)` missed. Assert on the
  reconstructed circuit, not the text.
* **`ps` and `bs_rx` are the palindrome of this file.** They are the two gates
  that work; a fixture built from them is invariant under every defect above.
* **Mutation-test each fix.** Specifically: re-introduce `unwrap_or(0.0)`, the
  `// unsupported` arm, the missing `else` in the reader, and the omitted `pol`
  marker — each must turn exactly one test red. D3 is the one most likely to
  have no failing test, because its symptom is a *correct-looking* file.
* **Do not let O4 hide D3.** A test that emits `hwp` and asserts the text
  contains `hwp` passes whether or not the `pol` marker was written. The
  assertion has to be on the re-parsed mode count (2N vs N).
* **The emulator hand-over must compare numbers, not acceptance.** "Perceval
  loaded it" is not the claim; "Perceval computed the same distribution" is.
  A wrong beamsplitter convention loads perfectly.
