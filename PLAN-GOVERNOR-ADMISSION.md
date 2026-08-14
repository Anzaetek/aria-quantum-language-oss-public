<!-- SPDX-License-Identifier: Apache-2.0 -->
# PLAN — three admission defects in `omega-server`'s governor

**Status: PLAN, not implemented, not yet adversarially reviewed.**

Three defects were filed against the admission path as requests without patches
(G1, G2, G3 below; two of the three turn on where routing should live, which is
a design call rather than a bug with one obvious shape). All three were
**re-verified in this tree** before this plan was written — line numbers and
code below are from the current `main`, not from the commit they were found on.

The governor's whole job is that a reservation describes the allocation that
follows it. All three defects are cases where it does not, and all three fail in
the **under-admission** direction, which `worker.rs:64-66` already singles out
as worse than refusing.

## G1 — `/gradient` prices one row and executes all of them

**Live and unconditional: any build, no features, no GPU, no environment
variable.** The highest severity of the three, and the only one that needs no
special configuration.

`quantum_bridge.rs:758`:

```rust
let mut shape = shape_for(&circuits[0], true, circuits.len());
shape.gradient = true;
let _reservation = match admit_shape(&shape) { ... };
```

then `:768` loops over **every** row: `for (i, ir) in circuits.iter().enumerate()`.

Verified in this tree: `gradient_quantum_route` (`:725`) contains exactly one
reservation, at `:760`. There is no second one later in the handler.

This is the reintroduction of a hole the codebase documents as fixed three
functions above it. `admit_batch` (`:334`) exists precisely to close it, and its
own doc comment says so:

> Pricing only the widest row was a hole: rows carry independent `backend`
> selections, so a 40-qubit Clifford row (cheap, stabilizer) could be picked as
> "widest" while a 30-qubit Statevector row in the same request allocated 16 GiB
> against that reservation. Width does not imply cost.

`/expectation` uses `admit_batch` (`:937`). `/gradient` does not. The comment at
the defect explains only the `gradient = true` part, so the one-row pricing
reads as deliberate.

Note `shape_for(..., circuits.len())` passes the batch length, so the shape
*knows* it is a batch and prices a single row anyway.

**Impact.** `{"circuits": [{"num_qubits": 4}, {"num_qubits": 30}]}` is admitted
on the 4-qubit row — kilobytes — and executes a 30-qubit dense statevector with
a backward state resident alongside the forward one: order 32 GiB against a
reservation that accounted for none of it.

Ordering is not a mitigation. A caller has no reason to sort widest-first, and
`admit_batch`'s comment already establishes that width alone is the wrong key
anyway.

**Fix.** Price the worst row over the whole batch, as `/expectation` does.
Either reuse `admit_batch` with a gradient flag, or mirror its
`max_by_key(estimate_peak_bytes)` fold. Two details must survive:

* `unwrap_or(u64::MAX)` on an unpriceable row, so a row nobody can price is not
  silently out-ranked (`:342-344`);
* the non-empty precondition, which `/gradient` already checks at `:722-724`.

Reusing `admit_batch` is preferable to mirroring it — a second copy of the fold
is how the two drifted in the first place.

## G2 — `/expectation` and `/gradient` are priced as device work they never do

**Live on any `--features opencl` build with `OMEGA_DEVICE=opencl`.**

`exec_target_for` (`:295`) decides the **pricing** target and returns
`ExecTarget::Device(0)` for a Statevector/Auto selection when the resolved
device is OpenCL. Its doc comment claims it "Mirrors `exec_statevector`'s
routing exactly — if the two ever disagree, the reservation is against the wrong
pool."

It feeds three endpoints and only one can reach a device:

| endpoint | pricing | execution | reaches OpenCL? |
|---|---|---|---|
| `/execute` | `execute_shape` → `exec_target_for` | `exec_statevector:499` | **yes** |
| `/expectation` | `admit_batch:334` → `shape_for` → `exec_target_for` | `expectation_quantum_ir:469` | **no** |
| `/gradient` | `:758` → `shape_for` → `exec_target_for` | `:766` | **no** |

Verified in this tree: `expectation_quantum_ir:469` constructs
`omega_backend_statevector::StatevectorBackend::new()` unconditionally for a
Statevector selection, and the `/gradient` handler builds the same CPU backend
once before its loop. Neither consults `DeviceKind`. So "mirrors exactly" is
already false for two of the three callers.

**Impact.** `ExecTarget::Device(_)` prices at `BYTES_PER_AMPLITUDE / 2` = 8 B
per amplitude (`worker.rs:112-113`), on the stated grounds that device kernels
are f32. The CPU backend is `Complex64` — 16 B.

* **Unified memory**: the pool is accidentally right (there is only one) but the
  job is under-priced **2×**. A 30-qubit `/expectation` is admitted at 8 GiB and
  allocates 16 GiB.
* **Discrete device**: wrong pool *and* wrong width. The device pool is debited
  while host RAM does undebited work, so a full GPU budget can be consumed by
  jobs that never touch the GPU.

Compounds with G1: a `/gradient` batch is priced from one row *and* at half the
true width.

**Fix.** Make the pricing target a property of the **endpoint** rather than a
global re-derivation from `OMEGA_DEVICE`: `shape_for` takes the target from its
caller, `/execute` passes the device-capable decision, `/expectation` and
`/gradient` pass `Cpu`.

Explicitly **not** the larger refactor (a shared `statevector_route` enum
consulted by both pricing and dispatch): it would unify the one path that
already agrees and leave the two that actually diverge untouched.

## G3 — the OpenCL open-failure fallback makes pricing and execution disagree

**Live on `--features opencl` with `OMEGA_DEVICE=opencl`, whenever the device
fails to open.**

`exec_statevector:499` falls back on failure to open the device (`:504-517`),
printing to stderr and continuing on the CPU. As an *availability* decision that
is right, and the doc comment defends it as "never an error path for the
caller". But admission has already run: `/execute` reserves before any of this,
and `exec_target_for` returned `Device(0)` from the same `DeviceKind::resolve`.

1. Priced as `Device(0)` at 8 B/amplitude, debiting the **device** pool.
2. `OpenClStatevectorBackend::new()` fails — stale ICD, busy device, a driver
   that enumerates but will not create a context. Not exotic.
3. The job runs on the **CPU** at 16 B/amplitude in **host** RAM, against a
   reservation held on the device pool.

Pool and width are both wrong, and the failure is silent to the governor —
nothing re-prices.

**Why unifying the two lookups does not fix it.** Two functions consulting a
shared helper at different times are still two calls, and the fallback happens
*after* both. The invariant cannot be established by unifying the lookups,
because what breaks it is a runtime failure between them.

**Fix — a decision to make, not a mechanical change.** Two candidates:

* **(a) Decide once, at admission.** Probe or open the device at admission time,
  price against what actually opened, and hand execution the already-chosen
  backend (or a route value it must honour). Structurally sound; costs an
  open-and-hold on the admission path, and needs the handle to live as long as
  the reservation.
* **(b) Re-admit on fallback.** When the fallback fires, re-admit against the
  CPU shape and refuse if the host pool cannot take it. Cheaper to write, worse
  latency, and it can refuse *after* a caller was told it was admitted.

**Leaning to (a)**, because (b) turns an availability fallback into a possible
late refusal, which is the behaviour the fallback exists to avoid. But (a)'s
lifetime question is real and is the thing to settle before writing code.

## Order

1. **G1** — unconditional, highest severity, smallest change, no design question.
2. **G2** — endpoint-aware pricing. Independent of G1 and of G3.
3. **G3** — needs the (a)/(b) decision first.

G2 and G3 overlap in cause (pricing derives the target globally, execution
decides per call site) but neither fixes the other: G2 is the *static*
disagreement — the endpoint never had a device path — and G3 is the *dynamic*
one, where the device path existed and failed.

## What could make this pass for the wrong reason

* **Testing G1 with a batch whose first row is the widest.** That is the one
  ordering under which the defect is invisible. The fixture must put the
  expensive row **later**, and — because `admit_batch`'s comment is explicit
  that width does not imply cost — at least one case where the expensive row is
  *narrower* but carries a costlier `backend` selection.
* **Asserting that a request succeeds.** Every one of these defects *succeeds*;
  that is the problem. The assertion has to be on the **reserved amount**, which
  means the tests need to observe the governor rather than the response. If that
  is not reachable, the test is not testing the defect.
* **Testing G2/G3 without the feature.** Both are `#[cfg(feature = "opencl")]`
  paths. A test compiled without it passes vacuously, and this repository has
  shipped exactly that kind of green test before. Any G2/G3 test must be gated
  so that it *fails to compile* rather than silently disappears — or must drive
  `exec_target_for` directly, which is feature-gated internally and can be
  exercised by asserting the target for a known selection.
* **Trusting the doc comment.** `exec_target_for`'s comment asserts it mirrors
  execution. It does not, and has not for two of three callers. Comments in this
  area have been wrong before; check the call site.
* **Fixing G2 by making the execution paths device-capable instead.** That is a
  different (and larger) piece of work — wiring device execution into
  `/expectation` and `/gradient`. It would close the divergence, but by changing
  what the server does rather than what it charges, and it should not be
  smuggled in under an admission fix.
* **A second copy of the batch fold.** G1's fix must reuse `admit_batch`, not
  reimplement its `max_by_key`. Two copies drifting is the original defect.

## Not in scope

* The f32 pricing assumption itself is **correct today**: nothing outside
  `omega-backend-statevector-cuda` selects its `f64_path`, so the concern about
  an f64 device path is latent rather than live. It becomes live the moment such
  a path is dispatchable. A test pinning the assumed width would catch that, and
  is cheap; pricing plumbing for it is not needed yet.
* Wiring CUDA execution in. The `cuda-topology` feature's own doc comment is
  explicit that it does not route execution through CUDA.
* Whether `gradient = true` should be a factor of 2 or something else. Unchanged
  by any of this.
