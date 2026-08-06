<!-- SPDX-License-Identifier: Apache-2.0 -->
# TLA+ models — scheduling and resource management

Formal models of the parts of `omega-server` that are concurrent and
failure-prone, where the interesting properties hold over *all* interleavings
rather than over one run. Tests sample that space; a model checker exhausts it
over a small instance.

| model | covers | status |
|---|---|---|
| `Governor.tla` | admission control (`crates/omega-server/src/worker.rs`) | **written** |
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

Needs [TLC](https://lamport.azurewebsites.net/tla/tools.html) (a JVM tool):

```console
$ java -jar tla2tools.jar -config Governor.cfg Governor.tla
```

CI does **not** run this by default — `./ci.sh` must not acquire a Java
dependency (K13: headless, and external tooling skips cleanly). It is opt-in,
in the same spirit as the Qiskit cross-check.

## `Governor.tla` — what it says

**Safety (holds).** The admitted set never exceeds a pool's capacity, and a
completed job returns exactly what it took. These are the properties the
implementation's byte-weighted semaphore is there to provide.

**Liveness (fails — and that is the finding).** `EventuallyAdmitted` asserts
that a job which fits is eventually admitted. It does not hold, because
admission uses a non-blocking `try_acquire` with **no queue**: a steady trickle
of small jobs can be admitted ahead of a large one indefinitely, so `Free` never
simultaneously reaches the large job's weight. The client meanwhile receives
`429` with `Retry-After`, a header promising a retry that will never succeed.

The configured instance is deliberately tiny — three jobs against a 4-unit
budget, two small (1 each) and one large (3). Starvation needs an interleaving,
not scale.

This is recorded in `FIXES_PLAN.md` as A7 gap 3 ("no bounded queue; large jobs
are starvable"). The value of having it here is that the fix can be designed
against a concrete counterexample trace rather than an intuition, and the same
property can be re-checked once a queue exists — at which point
`EventuallyAdmitted` should hold and this README should say so.
