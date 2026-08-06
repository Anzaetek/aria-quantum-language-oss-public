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

**The question:** can a shared box be driven into memory exhaustion by the jobs
people actually submit?

**The scenario** (`MCGovernor.tla`): one 128 GB box, half its memory given to
simulation, so a 64 GB execution budget. Three jobs, sized by the real cost
model (`2^n * 16` bytes):

| job | qubits | needs |
|---|---|---|
| `qml1`, `qml2` | 28 | 4 GB each — inference rows, submitted often |
| `sweep` | 32 | 64 GB — one architecture-search trial |

Four extra qubits is **sixteen times** the memory. That ratio is why admission
control is not optional: an inference row fits trivially, and a search trial
needs the entire budget.

**Safety — checked, holds.** Exhaustive over 22 states:

```
Model checking completed. No error has been found.
```

Admitted jobs never exceed 64 GB, and a finished job gives its memory back. No
interleaving of submissions and completions exhausts the box.

**Liveness — checked, FAILS.**

```
Error: Temporal properties were violated.
```

One 4 GB inference row is enough to keep the 64 GB trial out. Admission uses a
non-blocking `try_acquire` with **no queue**, so a steady stream of small work
starves the search indefinitely while the client receives `429` with a
`Retry-After` promising a retry that never succeeds. That is A7 gap 3 as a
reproducible trace, and the design input for a bounded queue.

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
