<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — export/parse integrity (the "one door over" defect class)

**Written 2026-08-12 after two adversarial reviews of that day's work. Nothing
here is implemented.**

## The governing principle

> **Output claiming to be language X must be readable by things that speak
> language X.** Otherwise emitting X has no point — the only reason to write a
> dialect is so something else can read it.

Two corollaries this repository keeps violating:

1. **"Our own parser reads it" is not sufficient.** `cp`, `rxx`, `rzz` are
   accepted by our parser (or now are) but **rejected by strict
   `qiskit.qasm2.loads`** — only Qiskit's deprecated legacy loader takes them.
   An export only we can read is a private format wearing a standard's name.
2. **"It parses" is not sufficient either.** It must carry the same MEANING.
   A guard silently dropped still parses; that is the whole Part H defect.

The test that follows from this is not "does the string contain X" but
**"re-parse it, with the consumer that is supposed to read it, and compare
semantics."**

## The dialects, and which have been checked

| dialect | emitter | consumer that must read it | checked? |
|---|---|---|---|
| QASM 2.0 | `to_qasm` | Qiskit (strict `qasm2.loads`), `omega-parser` | **partly** — guards fixed; `cp`/`rxx`/`rzz`/`ryy` open (P5) |
| QASM 3.0 | `to_qasm3` | any OQ3 consumer, `omega-parser` | **guards just fixed**; nothing else verified, and **no OQ3 parser exists in-tree to re-parse against** |
| Aria | `to_aria_source` | `parse_aria_circuit` | **just fixed** (RESET, guarded comments) |
| **OPTICQASM** | ? | `omega-parser::parse_opticqasm`, Perceval | **NOT CHECKED — see below** |
| **CV (photonic)** | ? | piquasso, our CV backend | **NOT CHECKED — see below** |

### P7 — OPTICQASM and the photonic dialects have never had this audit

Everything above concerns the qubit dialects. The photonic side has the same
structure and none of the same scrutiny:

* **Is there an OPTICQASM emitter at all?** If Aria can *read* `photon q[2]
  pol;`, `hwp(θ)`, `pbs`, `bs_rx(θ,φ)` but cannot *write* them, then any
  circuit that arrives as OPTICQASM cannot round-trip — and the polarization
  work (Part I) added `hwp`/`pbs` to the reader with no check that anything
  writes them.
* **Do the DV photonic exports reach Perceval?** The bridge sends OPTICQASM to
  `perceval_runner.py`, which re-implements a subset (`ps`, `bs_rx`, `bs_ry`)
  — so a construct we emit that the runner does not implement is exactly the
  `cp` defect in photonic clothing. `hwp` and `pbs` were added to the runner;
  **were they added to every path that emits?**
* **CV has no interchange dialect at all.** `omega-backend-cv` is reachable
  from no surface (task #8 / Part C3), and the piquasso cross-check drives it
  through a hand-written Python recipe rather than through anything Aria
  emits. So there is no "language X" to be unreadable — but that is itself the
  finding, and it should be stated rather than left as an absence.

### P7 — MEASURED 2026-08-12, and it is the same defect

Run against `omega_parser::lower_to_ir` on OPTICQASM source:

| construct | `aria-core::to_opticqasm` emits | `omega-parser` reads |
|---|---|---|
| `ps`, `bs_rx` | yes | **yes** — round trip OK |
| `hwp`, `pbs` (Part I polarization) | **NO** | yes |
| `squeeze`, `kerr`, `displace` (CV) | **YES** | **REJECTED** — "unknown photonic gate: squeeze" |

**Two distinct failures, both the class this document is about:**

1. **`to_opticqasm` emits CV gates nothing can read.** `squeeze`, `kerr` and
   `displace` are written into a file labelled `OPTICQASM 1.0;` that our own
   parser rejects outright. This is exactly the `cp` defect — an export in a
   named dialect that the dialect's reader refuses — except worse, because
   there is no second consumer (no Perceval equivalent for CV) that accepts it
   either. The file is readable by nothing.
2. **`hwp` / `pbs` can be read but never written.** Part I added polarization to
   the *reader* and to `perceval_runner.py`; the emitter has **zero** references
   to either. A polarization circuit that arrives as OPTICQASM cannot be
   re-emitted, so the round trip is one-way by construction.

Note the grammar hides both: `gate_name = @{ "ps" | "bs_rx" | "bs_ry" | "bs" |
ident }` — the `| ident` fallback means **any** identifier parses, so the
failure surfaces at lowering, not at parse. A syntax-level check would have
called all six of these fine.

**Fix, in order:**
* Decide what `to_opticqasm` should do with CV gates: emit them and teach the
  lowering (the CV backend exists and executes them), or refuse. Emitting a
  file nothing reads is the one option to remove.
* Emit `hwp` / `pbs`, or state that polarization is read-only and why.
* Then the enumeration: every construct the OPTICQASM lowering accepts must
  round-trip through the emitter, and vice versa.

**Concretely, do for each what P3 does for the qubit path:** enumerate the
constructs the OPTICQASM *grammar* accepts, and assert each one round-trips
through whatever claims to emit it, plus through Perceval where the bridge
claims coverage. `hwp`/`pbs` are the highest-risk pair — they are the newest,
they carry the conventions most easily got backwards (a global `i`, a swap
direction), and their pins live in tests that check matrices rather than
round trips.

## The pattern, stated first, because it is the actual problem

Three commits in a row fixed *"an emitter that silently drops X"*, and **each
one shipped with the same defect one door over**:

| commit | fixed | shipped broken beside it |
|---|---|---|
| `81ed534` | `to_qasm` dropped classical guards | `to_qasm3` still drops them |
| `c3c94bf` | `aria_emit` dropped classical guards | `aria_emit` still drops `RESET` |
| `b0745c0` | `cp` unreadable by our parser | `rxx`/`ryy`/`rzz` still unreadable |
| `b0745c0` | claimed the lowering mutations were caught | `aria-runtime/lower.rs` has **zero** coverage |

Fixing instances has a 100% failure rate at fixing the class. Every one of these
was found by a human or an adversarial reader diffing tables by hand — **no test
in this repository can see any of them.** So the first work item is not a defect
fix; it is the enumeration that makes the class visible.

---

## P0 — Actively harmful: our error message recommends silent data loss

`to_qasm` refuses a wide-register guard with *"Use `to_qasm3` (OpenQASM 3's
`if (c[i] == V)` addresses a single bit)"*. **`to_qasm3` contains zero
references to `inst.condition`** and emits every guarded instruction bare.

A user who follows our explicit advice gets exactly the circuit Part H measured
as `{"11": 2002, "10": 1998}` against a true `{"00": 1995, "11": 2005}`.

This is worse than the original bug: the original was silence, this is
misdirection.

**Fix.** Either:
1. `to_qasm3` emits `if (c[i] == V)` — OpenQASM 3 supports single-bit
   conditions, which is the entire premise of the refusal message; or
2. `to_qasm3` becomes fallible and refuses too, and the `to_qasm` message stops
   naming it.

**(1) is correct** — it closes the gap the message promises. Do (1).

**Verify:** round-trip a guarded circuit through `to_qasm3` and re-parse. The
existing `to_qasm3_accepts_what_qasm2_refuses` test asserts the **absence** of
`if` and must be inverted, not deleted — it is the canary that fires when this
lands.

---

## P1 — Silent data loss introduced on 2026-08-12

### P1a. `aria_emit` deletes `RESET`

`aria_emit.rs` emits `GateKind::Reset` as `-- reset q[0]`, a **comment**. Since
`b0745c0` made `apply RESET on q[0]` spellable, an Aria → Aria round trip now
deletes a channel that changes measurement statistics, with no diagnostic.

Introduced by the commit that made RESET spellable. `reset_is_spellable` never
touches emission, so nothing can see it.

**Fix:** emit `apply RESET on q[i]`, matching the spelling the parser now
accepts. Same for `barrier` if it has a spelling; if it does not, say so rather
than emitting a comment that reads like a faithful export.

### P1b. Guarded comment lines produce **unparseable** Aria

`aria_emit.rs` wraps every emitted line, including comments:

```
when m[0] == 1 { -- reset q[0] }
```

Aria's `--` comment runs to end-of-line, so the `}` is commented out and the
brace structure breaks. The code comment claiming *"the comment is what
round-trips"* is **false** — it does not parse at all.

That comment was reasoning, not measurement.

**Fix:** never wrap a comment line in a guard. Either emit the guard as part of
the comment (`-- when m[0] == 1: reset q[0]`) or refuse.

**Verify:** `the_aria_emitter_carries_the_guard_too` is a **substring check that
never re-parses** — the exact standard its sibling test states three lines
away. Make it re-parse. That change alone catches P1b.

---

## P2 — Correct a false claim in a commit message

`b0745c0` states: *"turning Reset into a Barrier [in `aria-runtime/src/lower.rs`]
left ALL FIVE TESTS PASSING… Re-ran both mutations: each is now caught by
exactly the corresponding test."*

**The second sentence is wrong.** The two tests added call
`aria_core::backends::omega::to_omega_ir` — a different crate, a different
function. `aria-runtime/src/lower.rs`'s `AKind::CRz` / `AKind::Reset` arms have
**no coverage at all**; the mutation described would pass the suite today.

**Fix:** add tests through `aria_runtime`'s own entry points (`run_counts`,
`expectation`), re-run both mutations there, and record the corrected result. A
follow-up commit must state that the earlier message was wrong — the claim is in
the permanent record and cannot be left standing.

---

## P3 — The structural fix: one enumeration, compiler-enforced

The scraping guard drafted earlier is **wrong twice over** and must not ship:

* It harvests `name_to_gate` arms only, so a gate handled by **decomposition in
  `lower_gate_app`** is invisible — meaning it would stay red forever on exactly
  the gates a decomposition fixes, or get "fixed" with a fake table entry.
* It reads Rust source with string matching, and its `> 15` threshold means the
  scrape can silently lose 11 entries and go **vacuously green** — the repo's
  named defect, inside the guard against it.

**Replace with a functional test.** `aria-core`'s dev-dependencies already
include `omega-parser`, and the existing tests already call `lower_to_ir` on
`to_qasm` output — the mechanism exists.

```
for every GateKind variant (exhaustive match — the COMPILER enforces coverage):
    build a minimal circuit using it
    to_qasm      -> lower_to_ir   must round-trip or refuse LOUDLY
    to_qasm3     -> (parser)      same
    to_aria_source -> parse_aria  same
```

An exhaustive `match` means **a new `GateKind` cannot be added without deciding
what each emitter does with it.** That is the property no table diff can give,
and it is what would have caught `cp`, `rxx`, `ryy`, `rzz`, `RESET`-as-comment
and the guarded-`measure` gap in one go.

Expect it to fail on ~6 known cases when first written. That is the point:
each failure is a decision that has been deferred silently.

---

## P4 — A latent panic — **FIXED 2026-08-13**

Confirmed by execution before fixing: all three panicked in
`omega-backend-statevector`, not merely lowered wrong. Both paths now go through
a shared `widen_cp_params` and a `check_gate_arity` whose `match` over `GateKind`
has no `_` arm, so a new kind must be given a signature or declared unchecked.
`crates/omega-parser/tests/gate_arity_is_validated.rs`; three mutations verified.

The original description follows.



`lower_gate_app`'s `cp`/`cu1` widening is **bypassed on the user-defined-gate
body path**. `gate mycp(t) a,b { cp(t) a,b; }` produces a `CU3` op with **one**
parameter; the statevector backend then indexes `resolved[2]` and **panics**.

Plain `cu3(0.7) a,b;` at top level has the same latent panic — there is no
arity validation anywhere between parse and the backend's index.

**Fix:** validate arity at lowering for every parametric gate, not just the two
that happen to be widened, and apply the widening on both paths. A wrong arity
must be a parse error, never an index panic in a backend.

---

## P5 — Round-trip gaps, in priority order

1. **Guarded `measure` / `reset` cannot re-parse.** Qiskit accepts both (measured),
   we emit both, and our grammar admits neither — `if_stmt` takes only
   `gate_app_stmt`, and `reset` is deliberately absent from `name_to_gate`.
2. **The emitter should write `cu1`, not `cp`.** Measured: strict
   `qiskit.qasm2.loads` **rejects** `cp`; only the deprecated legacy loader
   accepts it. `cu1` is accepted by both. My earlier claim that `cp` was "valid
   qelib1 which Qiskit accepts" was **loader-dependent and wrong as stated**.
3. **`rxx` / `rzz` / `ryy`.** Same correction applies: strict `loads` rejects all
   three. A **preamble `gate` definition** (`gate rzz(t) a,b { cx a,b; rz(t) b;
   cx a,b; }`) makes the file loadable by strict Qiskit **and** by our parser
   today with **zero parser changes**, via existing gate-def expansion — and it
   fixes `ryy` identically. This dissolves the whole "teach the parser"
   framing and must be evaluated before the decomposition route.
4. **`aria-runtime` refuses RXX/RYY/RZZ** even if the parser learns them, so the
   Aria front door stays shut while the QASM side door opens. Decide and state.

---

## P6 — Silent-failure philosophy is inconsistent between two files

`backends/omega.rs` **silently** drops a condition whose clbit is unmapped, and
**silently** retargets an unknown qubit to index 0. `aria-runtime/src/lower.rs`
returns errors for both. Same crate family, opposite philosophy, and the silent
one is used by callers that panic by design.

Also: `is_runtime_cond` decides runtime-ness by register **name** (`starts_with('m')`
or `== "c"`), so a creg named `flags` routes to compile-time evaluation and
fails obscurely — while the emitter happily writes `when flags[0] == 1`. And
`when m[0] == 2` is accepted and can never fire.

**Fix:** make the silent paths return errors, and refuse condition literals > 1
on a single bit.

---

## Order

P0 → P1 → P2 (record correction rides with P1) → **P3** → P4 → P5 → P6.

P3 is placed after the P0/P1 fixes deliberately: writing it first would produce
a long red list and tempt a wholesale allowlist. Writing it after the known
defects are fixed means every remaining failure is genuinely undecided.

## Standing rule this plan exists to enforce

**Fix the enumeration, not the instance.** If a fix does not come with a
mechanism that would have found it, the same bug is already sitting one door
over.
