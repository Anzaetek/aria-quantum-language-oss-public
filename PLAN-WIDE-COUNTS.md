<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — counts keys wider than 64 qubits

**Status: PLAN. Not implemented, and NOT fully reviewed** — the adversarial
review of this document was cut short by a session limit before it reported. Its
one parting note (the plugin protocol) turned out to be a real gap and is folded
in below; the rest of the plan has not had a second pair of eyes.

## Why now

`ExecResult::Counts(HashMap<u64, u32>)` keys a shot outcome by a `u64`, so above
64 qubits every high bit was silently dropped — a confident wrong answer, fixed
for now by *refusing* (`check_counts_width`). The refusal was always meant to be
step 1.

**The requirement that settles the design: MPS must support at least 1024
qubits**, on a 128 GB machine, comparable to what a commercial MPS simulator
advertises (hundreds to a few thousand; its statevector engine caps near 32).

The engine is already there — only the key is not:

| qubits | GHZ on MPS, `--expectation Z0` | peak RSS |
|---|---|---|
| 256 | 0.00 s | 3.0 MB |
| 1024 | 0.00 s | 3.9 MB |
| 2048 | 0.00 s | 6.2 MB |

Sampling those same circuits is refused. So the option `PLAN-CR-20260813.md`
listed as defensible — "keep `u64`, cap at 64, document it" — **is ruled out**:
it forbids exactly the regime MPS exists for.

## The blast radius, measured

* **49 files** reference `ExecResult::Counts` or the counts map — 32 under
  `src/`, 17 under `tests/`.
* **The JSON wire is already width-independent.** `aria-runtime/src/remote.rs`
  parses keys with `u64::from_str_radix(k, 2)` — the wire carries *bitstrings*
  (`{"0101": 42}`), and the `u64` is an internal bottleneck at the conversion,
  not a property of the protocol. This is much better than assumed and should be
  stated plainly: **no wire format change is required.**
* **TWO C ABIs are hard boundaries**, not one:
  * `omega-ffi` — `omega_result_get_counts` / `_n` write outcomes into
    `*mut u64`. A published C header.
  * **the plugin protocol** — `omega_core::ffi_types::FfiExecResult` also
    carries `pub bitstrings: *mut u64`, consumed in `plugin.rs` and produced by
    every backend plugin. This one was missed in the first draft of this plan
    and is arguably the harder of the two: third-party plugins compile against
    it, so a change is not ours alone to make.

## The decision the plan turns on: what replaces `u64`

Not chosen yet — this is the part to argue before writing code.

1. **`SmallVec<[u64; 1]>` (or `Box<[u64]>`) bitset.** Exact, ordered, hashable,
   and one inline word means ≤64 qubits stays allocation-free — the overwhelming
   majority of runs. Costs: `Hash`/`Ord`/`Debug` impls, and every construction
   site changes.
2. **`String` bitstring.** Simplest, matches the wire exactly, and removes the
   conversion entirely. Costs an allocation per shot in the sampling loop, which
   is the hot path — 1000 shots × a 1024-char string is not obviously fine and
   must be measured, not assumed.
3. **A newtype over either**, so the representation can change later without
   touching 49 files twice.

**Leaning to (1) behind a newtype `Outcome`.** But the sampling-loop cost of (2)
is a measurement, not a guess, and the measurement is cheap — do it first.

## The C ABI question

A `u64` out-parameter cannot carry 1024 bits. Options:

* **Keep the existing entry points for ≤64 qubits and add wide ones** that write
  packed words plus a word count. Back-compatible; two APIs to maintain.
* **Have the existing entry points refuse above 64 qubits** — they already
  would, via the backend gate, so this is mostly documenting reality — and add
  the wide API for new callers.

Either way the old functions must not silently truncate. That is the whole
defect being fixed, and repeating it at the C boundary would be worse, because
a C caller has no way to notice.

## A defect in the step-1 refusal, found while writing this

`check_counts_width(circuit.num_qubits)` gates on the **qubit count**, but the
outcome width depends on the mode:

* **collapse** — the mid-circuit `measure`s already ran, so the CREG is the
  result (`creg_to_u64`), and the width is the highest classical bit used;
* **skip** — the final qubit register is sampled, so `num_qubits` is right.

**FIXED 2026-08-14**, but the first statement of it was too broad and is
corrected here.

The claim was "a 1024-qubit circuit measuring 2 qubits is refused, and that is
wrong". Half right. It depends which mode the run is in, and the CLI picks the
mode from the circuit:

```
  1024q, mid-circuit measure + feed-forward -> COLLAPSE -> was refused, WRONG,
                                               now runs and prints |00>/|11>
  1024q, plain end-of-circuit measures      -> SKIP     -> still refused, and
                                               CORRECT: the backend samples all
                                               1024 qubits, so the key really is
                                               1024 bits
```

So the skip-mode refusal is not a guard bug. Making the skip path produce a
narrow key would mean projecting onto the measured qubits at sample time, which
is a real change to the sampling path.

**That change has since landed** (MPS and stabilizer), and step 0 is closed. A
measured circuit above the cliff is keyed on its creg in skip mode too —
`counts_keyed_on_creg` — so `1024q, plain end-of-circuit measures` now runs and
reports a 2-bit key rather than being refused. Only an *unmeasured* wide
circuit is still refused, which is correct: there is no narrower register to
report over.

The review of that work found seven defects, all fixed in `d4d9be8`. Three are
worth carrying forward as hazards for the steps below, because they are the
shape of thing this document's final section is about:

* **The same predicate must be used by the guard, the sampler and the
  renderer.** Two of the seven were exactly this — the MPS reset branch chose
  its outcome with `by_creg` while the guard admitting it used
  `by_creg || circuit_has_reset`, and the server rendered at `num_qubits` after
  the CLI renderers were fixed. `counts_outcome_width`'s second parameter is now
  named `by_creg` rather than `collapse` for this reason.
* **A projection applied twice is worse than one applied never.** `aria` called
  `project_counts_onto_creg` unconditionally, so above the cliff it re-read
  qubit positions out of a key that was already creg-packed: `|00>: 2000` where
  the truth is 0.79/0.21.
* **Every fixture mapped `q0 -> c0, q1 -> c1`**, under which "project onto the
  creg" and "take the low bits of the register" are the same function — so the
  whole suite stayed green with the projection destroyed. Fixtures for the
  steps below must use a **non-identity** qubit -> cbit map, and cross the bit
  order, or they test nothing.

The display had the same defect and it is the hazard listed at the end of this
document: the 2-bit outcome was rendered padded to 1024 characters, because the
renderer took `num_qubits`. Fixed at every render site (text, JSON, JSONL).

Note `project_counts_onto_creg` carries its own index-based 64 guard; the two
must stay consistent.

## Order

0. **Fix the mode-aware width first** (above). Independent of everything else
   and unblocks the common large-circuit case immediately.
1. **Measure option (2)'s sampling cost** — 1000 shots at 64, 256 and 1024
   qubits, string key vs current. Cheap, and it decides the representation.
2. Introduce the `Outcome` newtype over the chosen representation, with `Hash`,
   `Ord`, `Debug`, and explicit `from_bits`/`to_bitstring` conversions.
3. Convert `ExecResult::Counts`, then the backends, then the front ends. The
   compiler drives this; the risk is not compilation but the places that *format*
   an outcome and currently assume 64.
4. Replace `check_counts_width` with the real support. Keep a guard for anything
   that still cannot widen (the old C entry points), so nothing silently
   truncates anywhere.
5. Extend `counts_width_boundary.rs` past the old cliff: 63/64/65/66 stay, and
   **128, 256, 1024** are added on at least two backends.

## What could make this pass for the wrong reason

* **An identity qubit -> cbit map.** Measured: with `q0 -> c0, q1 -> c1` the
  entire projection can be replaced by `cbit_of[q] = q` and 145 tests still
  pass. Put the reported pair somewhere other than q0/q1 and cross the bit
  order.
* **Testing only GHZ.** GHZ yields two outcomes and is nearly the easiest
  possible case — it was also the fixture that hid the original bug, since a
  truncated GHZ still has exactly two keys. At least one fixture must produce
  outcomes that differ *above* bit 64, which GHZ's all-ones does only trivially.
* **Asserting on outcome counts rather than key contents.** Same trap as before.
* **Forgetting the formatting sites.** A key that is correct internally and
  printed through a `{:064b}` is still wrong to the user, and no type error
  catches it.
* **Believing the wire is fine because it is "already bitstrings".** It is —
  but `from_str_radix` still funnels through `u64`, and a 1024-bit key parsed
  there today would silently wrap or error depending on the path. The
  conversion sites are where the defect will survive if it survives anywhere.
* **Benchmarking the representation on a 4-qubit circuit.** The allocation cost
  of (2) only appears at width and shot count together.
