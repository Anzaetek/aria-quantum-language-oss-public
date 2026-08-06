<!-- SPDX-License-Identifier: Apache-2.0 -->
# TLA+ models — scheduling and resource management

Formal models of the parts of `omega-server` that are concurrent and
failure-prone, where the interesting properties hold over *all* interleavings
rather than over one run. Tests sample that space; a model checker exhausts it
over a small instance.

| model | covers | status |
|---|---|---|
| `Governor.tla` | admission control (`crates/omega-server/src/worker.rs`) | **checked** — safety holds, liveness fails as predicted |
| `DurableBatch.tla` | batch lifecycle across crash/reconnect (planned — `FIXES_PLAN.md` A9/A10) | not yet |

## What these models do *not* catch

Worth stating first, because the alternative is a spec that manufactures the
same false confidence it was meant to prevent.

A model verifies a **protocol against the assumptions it is handed**. It cannot
tell you an assumption is wrong. The most serious defect found in the governor
so far — MPS jobs priced by their tensor size while the backend actually
contracted to a dense `2^n` statevector, under-pricing the *default* path by
four orders of magnitude — would **not** have been caught here. The protocol was
correct; the weight fed into it was a lie.

That class of bug is closed by reading what the backends allocate and by
differential tests, not by TLA+. Use these models for what they are good at:
concurrency, ordering, and failure interleavings.

## Running them

```console
$ tools/tla/check.sh
```

The script finds a JDK (probing for a real one — macOS ships a `java` stub that
only offers to install one) and runs both configurations. It **skips cleanly**
when the JDK or the jar is missing, rather than failing.

TLC itself is project-local and **not committed** (2.2 MB binary, gitignored):

```console
$ curl -fsSL -o tools/tla/tla2tools.jar \
    https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
```

CI does **not** run this by default — `./ci.sh` must not acquire a Java
dependency (K13: headless, external tooling skips cleanly). Opt-in, in the same
spirit as the Qiskit cross-check.

`MCGovernor.tla` holds the concrete instance: TLC's `.cfg` format cannot express
a function literal, so the constants are defined in a module and substituted
with `<-`.

## `Governor.tla` — what it says

**Safety — checked, holds.** 26 distinct states, exhaustive:

```
Model checking completed. No error has been found.
67 states generated, 26 distinct states found, 0 states left on queue.
```

The admitted set never exceeds capacity, and a completed job returns exactly
what it took — the properties the byte-weighted semaphore exists to provide.

**Liveness — checked, FAILS. That is the finding.**

```
Error: Temporal properties were violated.
```

with this lasso (small jobs cycling, `big` never admitted):

```
State 1  admitted = {}            finished = {}
State 2  admitted = {s2}          finished = {}          <- Admit
State 3  admitted = {s1, s2}      finished = {}          <- Admit
State 4  admitted = {s1}          finished = {s2}        <- Complete
State 5  admitted = {}            finished = {s1, s2}    <- Complete
State 6  admitted = {}            finished = {s1}        <- Resubmit
Back to state 1                                          <- Resubmit
```

Note *why* weak fairness does not save `big`: `Admit(big)` is enabled at states
5, 6 and 1 (`Free = 4 >= 3`) but **not** at 2 and 3, so it is never
*continuously* enabled and `WF` never obliges it to fire. A trickle of small
work keeps re-closing the window. Meanwhile the client is receiving `429` with
`Retry-After` — a header promising a retry that will never succeed.

The instance is deliberately tiny: three jobs against a 4-unit budget, two small
(1 each) and one large (3). Starvation needs an interleaving, not scale.

This is recorded in `FIXES_PLAN.md` as A7 gap 3 ("no bounded queue; large jobs
are starvable"). The value of having it here is that the fix can be designed
against a concrete counterexample rather than an intuition, and the property
re-checked once a queue exists — at which point `EventuallyAdmitted` should
hold and this README should say so.

## Why this survives the move to a cluster

The model deliberately says nothing about **how work arrives**. `Admit` is
enabled by capacity alone, so push scheduling (a coordinator assigns) and pull
scheduling (a worker steals) produce the same state space here — they differ in
who *proposes* a job, not in what admission does with it.

That is what should keep A12 (the cluster manager) a moderate change rather than
a rewrite: adding a node dimension means indexing `admitted`/`Capacity` by node
and keeping the per-node invariant, while the safety property and the starvation
result carry over unchanged. The genuinely new properties are the lease
protocol's — a dead worker's row returns to the pending set, and no row is lost
or silently executed twice — which is exactly the part worth model-checking
*before* it is written.
