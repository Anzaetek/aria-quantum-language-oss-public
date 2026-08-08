<!-- SPDX-License-Identifier: Apache-2.0 -->
# Plan — outstanding requests from `fixes/`

Scope: every request document in `fixes/` (gitignored, downstream-authored),
with the two new 2026-08-05 items — `REMOTING.md` and `PLAN-CV-BACKEND.md` —
planned in depth. Written 2026-08-05 against `25ba5b1`.

Method: every claim a request makes was re-checked against this tree before
being planned. Two of them were **wrong as stated**, and the corrections change
the work — they are called out in place rather than silently inherited.

---

## 0. Status of all `fixes/` requests

| # | Request | Verified status | Action |
|---|---|---|---|
| 1 | `REMOTING.md` §6 (5 gaps) | **OPEN** — confirmed | Part A |
| 2 | `PLAN-CV-BACKEND.md` (R0–R6) | **OPEN** — confirmed, with a scope correction | Part B |
| 3 | Photonic examples (DV + CV) | **OPEN** — zero photonic examples exist | Part C |
| 4 | `ARIA_BUG_mps_fixed_bond_no_truncation_error` (×2) | **CLOSED 2026-08-07** — verified by passing tests: `svd::discarded_weight_equals_dropped_singular_squares`, `svd::discarded_weight_is_zero_when_nothing_truncated`, `sim::under_provisioned_bond_reports_discarded_weight`, `sim::exact_bond_reports_negligible_discarded_weight` | none |
| 5 | `ARIA_BUG_mps_svd_instability_normal_equations` (×2) | **CLOSED 2026-08-07** — one-sided Hestenes Jacobi on `A` (never `A†A`); verified by `svd::test_svd_reconstruction`, `test_svd_identity`, `test_svd_rank1`, and `sim::deep_brickwork_is_unitary_at_exact_bond_dimension` | none |
| 6 | `ARIA_BUG_noise_ignored_non_statevector_backend` | **CLOSED 2026-08-07** — MPS carries trajectory noise, verified against *analytic* values by `sim::noise_mps_depolarizing_matches_analytic`, `noise_mps_amplitude_damping_matches_analytic`, `noise_mps_per_qubit_amplitude_damping`. Only stabilizer **sampling** noise remains unimplemented, and it is refused loudly — a false rejection, not a wrong answer | none |
| 7 | `ARIA_REQ_qasm3_interchange` | **PARTIAL** — QASM3 paths exist (`aria-core/src/ast/qasm.rs`, `omega-parser/src/lower.rs`, CLI) | audit vs the request's checklist |
| 8 | `ARIA_REQ_ibm_calibration_noise_import` | **OPEN (probable)** | Part D, after 6 |
| 9 | `CI-LIBTORCH-DYLD-FIX.md` + `0001-*.patch` | **SUPERSEDED** — CI now fetches libtorch and exports `DYLD_LIBRARY_PATH` via `tch-env.sh` | close as done |
| 10 | `PLAN-BACKEND-PLUGINS.md` | **LANDED** — plugin ABI + `omega-plugin-conformance` are CI stage 6b | close; CV may extend the corpus |
| 11 | `aria-supervised-qml-requirements` | **LANDED** (phases 0–4) | spot-check only |
| 12 | `aria-torch-requirements` | **LANDED** — tch stage on by default | none |
| 13 | `ARIA-KEEP-public-final.md` | **CONTRACT** — K1–K14, Q5/Q8/Q9 | binding on everything below |
| 14 | `PLAN-COMPLIANCE-2026-07-29`, `UPSTREAM_UPDATE_PLAN`, `aria-open-requirements` | historical/superseded | archive note |
| 15 | `TSIM-PPVM.md` **(new 2026-08-06)** | **OPEN** — two QuEra simulators via the existing bridge protocol | Part E |

Items 4, 5 and 6 were **verified and closed on 2026-08-07** by running the named
tests rather than reasoning from the code — three bug documents that had been
implying open defects for over a week. Item 7 (QASM3) still needs an audit
against its checklist; item 8 (IBM calibration import) is **not** blocked by 6,
since the noise model on statevector/MPS is now confirmed trustworthy.

Worth noting *why* these looked open for so long: a fixed bug leaves no trace
unless someone closes the report. The verification cost was three `cargo test`
invocations.

---

## Part A — Remoting (`REMOTING.md` §6)

The use case: laptop as caller, DGX Spark as executor. §1–§5 of that document
were spot-checked and are accurate. All five gaps are additive; no existing
route changes.

### A0. Scenarios covered, and the data-transfer budget

The wire is a laptop on the far side of an SSH tunnel. Design for it explicitly:
what crosses, how often, and what the payload scales with.

**Standing invariant (already true, keep it):** the statevector never crosses
the wire — `/execute` returns counts, `/expectation` returns scalars
(`remote.rs:10-12`). The one exception is `/v1/quantum/execute_pattern` (MBQC),
which returns a full output statevector, i.e. `O(2^n)` complex amplitudes. That
is fine for the small graph states it targets and unsuitable for anything large;
say so in its docs rather than discovering it at n=24.

| # | Scenario | Wire pattern | Status |
|---|---|---|---|
| S1 | One-shot run / counts | 1 circuit up, counts dict down | works today |
| S2 | Single score `⟨O⟩` | 1 circuit up, 1 scalar down | works today |
| S3 | **Batch scoring** (N data rows) | should be 1 request, N scalars | server takes `circuits`; **client sends N separate requests** (A2) |
| S4 | **Training step** (forward + gradient) | should be 1 request, 1 reduced gradient | **no gradient route at all** (A1) |
| S5 | Epoch-scale job | submit → poll/notify | inline in the request handler (A4) |
| S6 | Torch layer on the DGX | S3/S4 driven from `aria-py` | no HTTP path in pyo3 (A3) |

**The dominant cost is re-sent circuit topology, not results.** The wire IR's
`OmegaGateOp.params` is `Vec<f64>` — concrete only, no symbols
(`quantum_bridge.rs:76-82`; aria-core's mirror is the same). So a batch encodes
one row as one *fully baked* circuit, and `circuits: [...]` re-transmits the
entire gate list once per row even though every row shares the same ansatz and
differs only in a handful of feature parameters.

Worked example — HEA `n=10, L=6` (~190 ops, ~120 trainable params, 10 features),
batch of 256 rows, at roughly 90 bytes of JSON per gate op:

| encoding | up per forward pass |
|---|---|
| `circuits: [...]` (today) | 256 × ~17 KB ≈ **4.4 MB** |
| template + per-row bindings | ~17 KB once + 256 × 10 floats ≈ **48 KB** |

≈ **90× less**, and it grows with circuit depth rather than with `rows × depth`.
Over an epoch of 100 batches with forward *and* gradient, that is ~880 MB versus
~10 MB.

So the transfer-minimising design, to be settled **before** A1's route is coded
(it shapes both routes):

1. **Template + parameter matrix.** Accept one circuit plus
   `param_values: [[...], ...]` — one binding vector per row — instead of N baked
   circuits. Requires the wire IR to express symbolic parameters (today it
   cannot) *or* a positional override keyed by gate index. This is the single
   biggest win and it is currently blocked by the `Vec<f64>` params. Note
   `OmegaGradientRequest` already carries `circuit` + `param_values` for the
   single-row case — the batch form is the natural generalisation, not a new idea.
2. **Reduce the gradient server-side.** A trainer needs the batch-*aggregated*
   gradient, not per-row Jacobians. Returning `N × P` floats (256 × 120 ≈ 370 KB)
   makes the *response* dominant. Let the client send per-row loss weights
   `dL/d⟨O⟩_i` and have the server return `Σ_i w_i ∇_i` — one vector of `P`
   floats, ~1.5 KB. That is the same contract as the in-process
   `expectation_multi_then_gradient` + `GradientObservableFactory`, so it reuses
   a pattern the runtime already implements rather than inventing one.
   Keep per-row Jacobians available behind an explicit flag for diagnostics.
3. **Fuse forward + gradient into one round trip.** Predictions and gradient come
   from a single forward sweep in-process; splitting them across two HTTP calls
   both doubles round trips and re-sends the circuits. One call returns both.
4. **Round-trip count.** S3 done per-row is N requests: at 256 rows over a tunnel
   with ~20 ms RTT that is ~5 s of pure latency per forward pass versus ~20 ms
   batched — before any compute. This is what A2 buys even without item 1.
5. **Prefer scalars to counts for scoring.** `⟨O⟩` is one f64; a counts dict
   grows with the number of *distinct* observed bitstrings, which for a
   near-uniform 20-qubit output approaches `min(shots, 2^n)` entries. Use
   `/expectation` for scoring paths, not `/execute` + client-side reduction.
6. **Transport hygiene.** Reuse the connection (keep-alive) and enable gzip —
   the payload is highly repetitive JSON and compresses well. Cheap, and it
   partially mitigates item 1 until the template encoding lands.

Acceptance for the batch work should therefore be measured, not asserted: record
bytes-on-the-wire and request count for a 256-row forward+gradient step, and
pin them in `TESTING.md` alongside the numeric gates (Q9 — state the numbers).

### A1. `POST /v1/quantum/gradient` — the blocking gap

The wire type already exists on the language side (`OmegaGradientRequest`,
`aria-core/src/backends/omega.rs`) with no server route consuming it, so
`aria train` and the QML trainer are local-only.

- Route accepting `OmegaGradientRequest`, **batched like `/expectation`**
  (`circuits: [...]`), returning `d⟨O⟩/dθ` per circuit in input order.
> **BLOCKER discovered 2026-08-06, before writing the route.** A gradient route
> over the current wire IR **cannot compute a gradient at all**, because the
> wire cannot express what to differentiate *with respect to*.
>
> `OmegaGateOp.params` is `Vec<f64>`, and `translate_to_core_ir`
> (`quantum_bridge.rs:198`) maps every one to `ParamExpr::Concrete`. Zero
> `Symbol` occurrences survive translation. So the circuit reaching
> `Backend::adjoint_gradient` has **no free parameters**: there is no θ, and the
> adjoint has nothing to differentiate. `OmegaGradientRequest.param_values`
> carries `(String, f64)` symbol *names* that nothing on the wire binds to.
>
> Implementing the route without fixing this would produce an endpoint that
> returns empty or zero gradients very convincingly — the same shape of defect
> as the CV silent-drop and the MPS under-pricing: confident output, no physics.
>
> **Two ways forward, and the choice belongs to whoever owns the wire format:**
>
> 1. **Carry symbols on the wire.** Extend `OmegaGateOp.params` from `Vec<f64>`
>    to a `ParamExpr`-like sum (`Concrete(f64) | Symbol(name)`). Additive JSON,
>    but it changes a format downstream clients pin (K3), and it is the *same*
>    change A0 item 1 needs for the template + parameter-matrix encoding — so
>    doing it once unlocks both the ~90× payload win and gradients.
> 2. **Positional differentiation.** Keep params concrete and have the client
>    name what to differentiate by position (`gate_index`, `param_index`).
>    Requires no format change and is strictly less expressive: it cannot
>    express one symbol shared across several gates, which every real ansatz
>    does.
>
> **Recommendation: option 1**, because A0 needs it regardless and option 2's
> limitation (no shared parameters) rules out the ansätze this is for. A1 is
> therefore **blocked on a wire-format decision**, not on implementation effort.

- **Wire shape, specified.** `OmegaGradientRequest` (`omega.rs:371-376`) carries
  exactly **one** circuit, so batching needs a container. Mirror
  `QuantumExpectationReq` (`quantum_bridge.rs:534-546`) exactly, so the two
  routes stay learnable as a pair:

  ```jsonc
  // POST /v1/quantum/gradient
  { "circuit":  {…},        // XOR
    "circuits": [{…}, {…}], // one per data row
    "observable": "0.5*X0 + Z1 Z2",
    "param_values": [["theta", 0.5]],
    "method": "adjoint" }   // | "param-shift" | {"stochastic-param-shift": {"shots": 256}}
  // 200 → { "backend": "statevector",
  //         "gradients": [[["theta", 0.123]], …] }   // one row per circuit, in order
  ```
- Dispatch: adjoint where the backend offers it, parameter-shift otherwise —
  reusing the runtime's existing selection logic rather than a second copy.
- **Q5**: batch index preservation is a keep requirement; the response must hold
  row order under partial failure, not compact the successes. Note this is *new*
  behavior, not an existing convention — the current expectation batch returns
  400 on any bad row (`quantum_bridge.rs:589-591`).
- Gate: gradient over the wire ≡ in-process gradient, ≤ 1e-9 on a seeded
  circuit, plus an order-preservation test with a deliberately failing row.

### A2. Batched `expectation_remote`

`remote.rs:62-78` POSTs one `{circuit, observable}` per row although the server
already accepts `circuits`. One call per row-batch. Gate: N-row batch returns
the same vector as N single calls, one HTTP request.

### A3. Remote transport in `aria-py`

No HTTP path in the pyo3 layer, so the Torch layer cannot target a server.
Add `Remote`-backed batch/expectation/gradient entry points driven by
`OMEGA_SERVER`/`OMEGA_TOKEN`. Depends on A1 + A2. Doctrine: Python at the edge,
never in the loop.

### A4. Worker: stop holding the HTTP request open

`omega-server/src/worker.rs` is a stub; execution runs inline. `spawn_blocking`
+ a concurrency semaphore (the stub's own TODO), then a submit→poll/WS-notify
job surface for epoch-scale work.

### A5. Reach the GPU backends over the wire

**Partly done already** — `REMOTING.md` §6.5 overstates the absence. The server
*does* route `Statevector` **/execute** through OpenCL when built with
`--features opencl` and `OMEGA_DEVICE=opencl` (`quantum_bridge.rs:338-363`). What
is genuinely missing:

- `expectation_quantum_ir` uses the CPU `StatevectorBackend` unconditionally
  (`quantum_bridge.rs:313-314`) — so for the QML expectation (and future
  gradient) path, GPU is unreachable. This is the part that matters.
- CUDA and Metal are not dependencies of `omega-server` at all; only OpenCL is.
- `backend_name` echoes `"statevector"` even when OpenCL ran — only stderr knows.
  Echo the resolved device so a caller can *prove* the DGX used its GPU.

### A6. Async I/O — a job protocol over the existing WebSocket

Today every execution runs inline in the request handler (`worker.rs` is a
4-line stub whose own TODO says "spawn_blocking with a semaphore"). A training
epoch therefore holds one HTTP request open for its entire duration: no
progress, no cancellation, and an SSH tunnel or proxy idle-timeout kills the job
outright. The transport already exists — `/v1/ws`, PQC-encrypted
(`main.rs:28-29`, `routes.rs:45-46`) — but `omega-client` does not speak it
(`omega-client/src/main.rs:12-13`), so nothing uses it.

Job lifecycle, additive to the current routes:

- `submit` → `job_id` immediately; work runs on `spawn_blocking`.
- **Progress + partial results streamed as they complete.** For a batch this is
  the important part: row results go out as each finishes, so the client starts
  consuming immediately and the server never buffers `N` results. Pairs with
  §A0 — the batch stops being one big response.
- **Credit-based backpressure.** The client grants credits; the server sends at
  most that many un-acked frames. Without this a fast executor and a slow laptop
  over a tunnel just moves the memory problem into the socket buffer.
- **Heartbeat ping/pong.** Idle tunnels and reverse proxies drop silent
  connections; a long job must not die because it was quiet.
- **Reconnect and resume by `job_id`**, with an idempotency key on submit so a
  retried submit does not execute twice.
- **Cancellation that actually cancels.** A dropped client future does *not*
  stop a `spawn_blocking` closure — the job must poll a cancellation token
  between circuits (and between shot chunks) or a disconnected client leaves the
  box computing for nothing. This is the single most common way a job server
  gets wedged.
- **HTTP poll fallback** (`GET /v1/quantum/job/:id`) for clients that cannot do
  WS, so the WS path is an optimisation and not a requirement.
- Optional binary framing (CBOR/msgpack) — a further cut on top of the gzip in
  §A0, since the payload is repetitive numeric JSON.

### A7. Resource governor — do not kill a shared box

Target deployment is a **shared** DGX / CUDA host, so the failure mode is not
"my job is slow", it is "my job OOM-killed someone else's". Nothing in the
server currently bounds anything: no memory guard, no qubit ceiling, no
concurrency limit (only `auth/rate_limit.rs`'s per-IP request limiter, which
counts *requests*, not *cost* — 1 request can be 256 GB).

**Admission control comes first, because the cost is knowable before allocating.**
A statevector is `2^n × 16` bytes (complex128); the shape is declared in the
request, so the server can price a job exactly and refuse it in microseconds:

| qubits | dense statevector | + adjoint (2 copies) |
|---|---|---|
| 24 | 256 MB | 512 MB |
| 28 | 4 GB | 8 GB |
| 30 | 16 GB | 32 GB |
| 32 | 64 GB | 128 GB |
| 34 | 256 GB | 512 GB |

Two qubits is 4×. There is no graceful degradation — it either fits or the box
dies — which is exactly why this must be a *pre-flight rejection*, not an
allocation attempt with a fallback.

1. **Cost model + preflight.** Price each job from `(num_qubits, backend,
   batch_size, shots, gradient?)`: dense `2^n × 16` (×8 for f32 GPU), adjoint
   doubles it, MPS ≈ `n × χ² × d × 16`, batch multiplies by whatever runs
   concurrently. Reject over-budget jobs with **429 + `Retry-After`** (busy) or
   **413** (never satisfiable on this host), and put the computed requirement and
   the available headroom *in the message* — "needs 32 GB, 12 GB free" is
   actionable; "internal error" after an OOM-kill is not. **Fail closed:** if the
   model cannot price a request, refuse it rather than gamble.
2. **Weighted admission semaphore.** Permits denominated in **bytes, not job
   count** (`tokio::sync::Semaphore::acquire_many`), so N concurrent jobs can
   never sum past the memory budget. A plain "max 4 jobs" limit is useless when
   job cost spans six orders of magnitude.
3. **Per-device GPU governor.** One semaphore per device plus a free-VRAM query
   before admission; two jobs must never each assume they own the card. GPU jobs
   serialize per device by default — VRAM is smaller and less forgiving than host
   RAM, and CUDA OOM tends to take the process with it.
4. **CPU.** Bound the global rayon pool and cap concurrent CPU jobs at roughly
   `cores / threads_per_job`. Oversubscription on a shared box degrades *every*
   tenant, not just the offender.
5. **Bounded queue + wall-clock budget.** Fixed queue depth (429 when full,
   never unbounded growth), per-job timeout, and the cancellation token from A6
   honoured throughout.
6. **Per-token quotas.** Tokens are already scoped by rights; add per-token
   concurrency and memory ceilings so one tenant cannot starve the others. This
   is what makes "shared" actually work.
7. **Config + observability.** `OMEGA_MAX_MEM`, `OMEGA_MAX_QUBITS`,
   `OMEGA_MAX_CONCURRENCY`, `OMEGA_GPU_MEM_FRACTION`, with sane defaults derived
   from detected RAM/VRAM rather than hardcoded. `/health` reports headroom,
   queue depth and active jobs, so a caller can size work *before* submitting.
   Governor diagnostics go to tracing/stderr — never stdout (Q8).

Acceptance, measured: an `n=40` statevector submit returns a fast, explicit
refusal naming the requirement (not an OOM); K concurrent admitted jobs never
exceed the configured budget; a client disconnect mid-job stops the compute
(observed via the box going idle, not merely a closed socket).

#### A7 status (2026-08-06) — landed, with named gaps

Implemented in `omega-server/src/worker.rs` (the former stub): cost model,
byte-weighted admission, 413/429 + `Retry-After`, `/health` budget snapshot,
cgroup-aware capacity, and guards on `/v1/quantum/execute`, `/expectation`,
`/execute_pattern` and the registry circuit path.

An adversarial review caught that the **first** version was worse than nothing,
and the corrections are the substance of what shipped:

- **MPS was priced by its tensors but always densifies.** `MpsBackend::
  expectation` → `execute(shots: None)` → `to_statevector()`
  (`omega-backend-mps/src/mps.rs:418`), and `resolve_backend` sends every
  non-Clifford n ≥ 20 circuit to MPS under the *default* `Auto`. A 34-qubit job
  priced at ~4 MiB and allocated 256 GiB — i.e. the default path walked past the
  guard while `/health` advertised headroom. Now priced by what the backend
  allocates, not by its name.
- **Photonic cost is combinatorial in photons**, `C(m+p-1, p)` with
  `p = ceil(m/2)`, so a *mode* ceiling never bounded it: 26 modes ≈ 766 GB was
  admitted on a 1 MiB token. Now priced exactly.
- **The batch priced only the widest row**, but rows carry independent backends,
  so width ≠ cost. Now prices every row and reserves the worst.
- **`/execute_pattern` (MBQC) had no admission at all** — state doubles per
  vertex, and vertices are caller-supplied.
- **`densifies` vs `returns_statevector`** are now distinct: `/expectation`
  materialises `2^n` internally but returns scalars, so charging it the JSON
  encoding factor would have over-priced ordinary QML batches ~8×.

Still open, deliberately and in priority order:

1. **The WASM lambda path is unguarded** (`lambda.rs` → `omega-wasm-runtime`'s
   host `execute_expectation` / `execute_with_shots`). Wasmtime *fuel* bounds
   guest instructions, not the host-side statevector allocated outside the
   guest's linear memory. Closing it needs a hook in `omega-wasm-runtime`, so it
   is a cross-crate change rather than a local one. **Until then the governor's
   coverage is 4 of 5 execution entry points, and this is the hole.**
2. ~~GPU/VRAM priced against host RAM~~ — **CLOSED by A7b** (2026-08-06):
   per-pool budgets with unified/discrete detection, f32 device widths, and
   fail-closed on an unknown device.
3. **No bounded queue (A7.5).** `try_acquire` refuses immediately, so a steady
   trickle of small jobs can starve a large one indefinitely while `Retry-After`
   promises a retry that never succeeds.
4. **No per-token quotas (A7.6)** — one tenant can still occupy the whole budget.
5. **`/health` publishes capacity and live headroom unauthenticated**, which on a
   multi-tenant box is a side channel revealing neighbours' job sizes and timing.
6. **`JobShape::gradient` is never set true in production** — no gradient route
   exists yet (A1). The field and its ceiling arithmetic are ready for it.

### A7b. Memory topology — RAM, VRAM and unified, across DGX / CUDA / Mac

**PLAN ONLY — not implemented.** The governor today has exactly one budget,
derived from host RAM. That is correct on a CPU-only box and wrong everywhere
else, in opposite directions depending on the machine. A single number cannot
express these three machines, so the budget must become **per-pool**.

| platform | topology | pools | if we get it wrong |
|---|---|---|---|
| **DGX Spark** (GB10 Grace-Blackwell: CPU+GPU on one package, ~128 GB LPDDR5X) — **the target executor per `REMOTING.md`** | **unified**, on Linux/aarch64 **with CUDA present** | **one** shared pool | The dangerous case. Naive detection sees `nvidia-smi` and assumes discrete: host reports ~128 GB, device reports ~128 GB, governor budgets **256 GB on a 128 GB machine** and cheerfully OOMs it. Unified-ness here is *not* implied by the OS. |
| **Mac, Apple Silicon** (this box: 24 GB, arm64) | **unified** — CPU and GPU share one physical pool | **one**: system RAM | Charging a Metal job to a *separate* "VRAM budget" double-counts the same way: host + GPU each think they have 12 GB, together they exceed 24 GB. |
| **Discrete NVIDIA** (RTX, A100, H100) | separate host RAM + device HBM/GDDR | host + one per device | Pricing a GPU job against host RAM is wrong *both* ways: it refuses jobs that fit in VRAM, and admits jobs that exceed it and then CUDA-OOM (which usually takes the process, not just the job). |
| **DGX A100/H100** (8 discrete GPUs) | discrete, multi-GPU | host + **8** device pools | One global GPU budget lets two jobs land on the same card while another sits idle. Device identity matters, not just device memory. |
| **Grace-Hopper GH200** | NVLink-C2C coherent; **distinct** capacities (LPDDR5X + HBM3) | host + device, but a device allocation can spill over the link | Treating it as pure-unified over-admits on HBM; pure-discrete forbids legitimate spill. Price against HBM, and record that spill degrades rather than fails. Note this is *not* the same as GB10 — coherent ≠ shared-pool. |
| CPU-only server | host only | one | current behaviour, already correct |

**The GB10 case is why "unified" cannot be inferred from the OS.** macOS+arm64
⇒ unified is safe, but Linux+CUDA splits into *both* discrete (A100) and
unified (GB10) — and the two demand opposite budgets from identical-looking
probe output. So topology must be *detected*, not assumed, with a safe default:

> **When the topology is uncertain, assume unified.** Assuming discrete on a
> unified box double-counts and kills it. Assuming unified on a discrete box
> merely under-uses VRAM — wasteful, recoverable, and visible in `/health`.
> The asymmetry is the whole argument.

**Design.**

1. **`MemoryPool`**: `Host`, `Device(index)`, or `Unified`. The governor holds
   one weighted semaphore per pool instead of a single global one.
2. **`JobShape` gains an execution target.** Admission debits the pool the job
   will actually allocate from:
   - CPU backend → `Host`
   - GPU backend on a discrete machine → `Device(i)` (plus a small host staging
     charge for the readback buffer, which is real and currently unpriced)
   - Any backend on a unified machine → `Unified` — **the same pool**, which is
     the entire point of the distinction
3. **Detection at boot, no new dependencies:**
   - *Unified*: `target_os = "macos"` + arm64 (`sysctl hw.optional.arm64`).
     Capacity is `hw.memsize`; Metal additionally caps a single allocation at
     `recommendedMaxWorkingSetSize` (~75% of RAM), so the **GPU ceiling is
     lower than the pool** even though the pool is shared. Both bounds apply.
   - *NVIDIA*: `nvidia-smi --query-gpu=index,name,memory.total --format=csv`
     once at boot — same subprocess-at-boot precedent as the existing `sysctl`
     call. Absent `nvidia-smi` ⇒ no device pools, correct on this Mac.
     **`memory.total` alone cannot tell GB10 from A100**, so classify with:
     1. `cudaDeviceProp.integrated == 1` — the canonical answer, available only
        in a CUDA-linked build; use it when present.
     2. Otherwise the heuristic: `device_total ≈ host_total` (within a tolerance)
        on aarch64, or a name matching the Grace-Blackwell/GB10 family ⇒
        **unified**. Two ~128 GB pools on a 128 GB machine is not two pools.
     3. Otherwise ⇒ discrete, but only when host and device totals are clearly
        *disjoint* (e.g. 1 TB host, 80 GB device). Anything ambiguous falls to
        the safe default above.
   - *Host*: unchanged, cgroup limit preferred over `/proc/meminfo`.
   - **Every value overridable**: `OMEGA_MAX_MEM`, `OMEGA_MAX_VRAM`
     (per-device or `idx:bytes,...`), `OMEGA_MEM_TOPOLOGY=unified|discrete|host`.
     Detection is a convenience; an operator who knows better must win, and a
     container may make detection lie.
4. **f32 vs f64 on device.** The GPU statevector backends are f32 — `2^n × 8`,
   not `× 16`. Pricing device work at the CPU's 16 B/amplitude over-refuses by
   2×. `CostKind` must carry the element width, not assume it.
5. **Fail closed per pool.** If a job targets a device whose capacity could not
   be determined, refuse rather than fall back to the host budget — falling back
   is precisely how a 64 GB job reaches a 24 GB card.

**Acceptance** (must not require the hardware to run):
- Topology detection is a pure function over injected probe output, so DGX,
  discrete-CUDA, GH200, Apple-unified and CPU-only are all unit-testable on any
  box. Real hardware only validates the *probe*, not the policy.
- Unified: a Metal job and a CPU job together cannot exceed system RAM.
- Discrete: a job larger than device memory is refused even when host RAM is
  ample; a job that fits VRAM is admitted even when host RAM is busy.
- Multi-GPU: `K` jobs never oversubscribe any single device.
- `/health` reports each pool separately, so a caller can see *which* resource
  is scarce rather than one aggregate number.

### A7c. Operator throttles — use only *part* of the machine

**PLAN ONLY.** A7 answers "will this job kill the box"; A7c answers "please only
ever use a quarter of it". Different requirement: the box may be shared with
non-Aria work, or be someone's laptop. Today the only knobs are `OMEGA_MAX_MEM`
and `OMEGA_MAX_QUBITS`, and the 50% RAM share is **hardcoded**
(`DEFAULT_MEM_FRACTION`) — the single most obvious thing to want to change is
the one thing not exposed.

| knob | absolute | fraction | governs |
|---|---|---|---|
| Host memory | `OMEGA_MAX_MEM` *(exists)* | `OMEGA_MEM_FRACTION` | admission budget |
| Device memory | `OMEGA_MAX_VRAM` | `OMEGA_VRAM_FRACTION` | per-device pools (A7b) |
| CPU | `OMEGA_MAX_THREADS` | `OMEGA_CPU_FRACTION` | rayon pool + concurrent CPU jobs |
| Concurrency | `OMEGA_MAX_CONCURRENCY` | — | simultaneous jobs, *independent* of bytes |
| Width | `OMEGA_MAX_QUBITS` *(exists)* | — | cheap pre-pricing reject |

Rules that keep this predictable:

1. **Absolute beats fraction** when both are set, and the result is reported
   rather than silently resolved.
2. **Caps compose by `min`, never `max`.** Every applicable cap is a ceiling; a
   generous one must never widen a strict one. Easy to get backwards, and the
   failure is silent over-admission.
3. **On unified memory (A7b), `MEM_FRACTION` and `VRAM_FRACTION` address the
   same physical pool** and must not multiply. Taking 0.5 of RAM and 0.5 of
   "VRAM" on a DGX Spark must not yield a 128 GB budget on a 128 GB machine.
   Where they conflict, `min` wins (rule 2), and `/health` says which applied.
4. **Human units.** Accept `48G`, `0.5`, `75%` — an operator writing
   `OMEGA_MAX_MEM=48G` today gets a silent parse failure and the 4 GiB fallback,
   which is worse than an error.
5. **`OMEGA_RESOURCE_PROFILE=gentle|balanced|greedy`** as a one-knob preset over
   the fractions (gentle ≈ 0.25 / 2 threads), because most callers want "don't
   hog my laptop", not five env vars.
6. **CPU throttling is real work, not just a number**: size the global rayon
   pool (`ThreadPoolBuilder::num_threads`) *and* cap concurrent CPU jobs at
   about `threads / threads_per_job`. Setting one without the other just moves
   the oversubscription.
7. **Per-token quotas** (A7 gap 4) are the same mechanism scoped to a token, so
   build the cap type once and apply it globally or per tenant.

**`/health` must report effective caps and their provenance** — value plus
whether it came from env, detection, or default. "Why was my job refused?" has
to be answerable without reading the server's source; a refusal citing a limit
the operator cannot see is the same dead end as an OOM.

Acceptance: with `OMEGA_CPU_FRACTION=0.25` on a 20-core box the pool is 5
threads and concurrent CPU jobs cap accordingly; `OMEGA_MEM_FRACTION=0.1`
refuses a job that the default 0.5 admits; on a simulated unified topology the
two memory fractions do not compound; `48G`, `0.5` and `75%` all parse, and a
malformed value is a startup **error**, not a silent fallback.

### A8. Client-side resilience — survive a disconnect, and *say so*

**PLAN ONLY.** Today `remote.rs` does a blocking POST and propagates any error
straight up (`expectation_remote`, `run_counts_remote`). An SSH tunnel blip
mid-epoch therefore kills the whole training run, and the caller learns only
that *something* returned `Err`. Two requirements, and the second is the one
usually skipped:

1. a transient disconnect must not destroy the run, **and**
2. the client must always be able to answer "what is happening right now" —
   silent retrying is its own failure mode, because a dead server and a slow
   one look identical from the outside.

**Classify the failure before reacting.** Blind retry is wrong for most of
these, and A7's refusal codes are already designed to carry the distinction:

| condition | retry? | client action |
|---|---|---|
| connection refused / reset / DNS / tunnel down | yes, backoff | `Degraded`, keep the run alive |
| timeout with no response | yes, **only with an idempotency key** | the request may have been executed |
| `429` + `Retry-After` (governor: busy) | yes, honour the header | `Degraded`; the job *will* fit later |
| `413` (governor: never fits) | **no** | terminal; retrying cannot help, fix the circuit |
| `401` / `403` | **no** | terminal; token expired or lacks `EXECUTE` |
| `400` (bad observable, malformed IR) | **no** | terminal; it is a bug in the caller |
| `5xx` | yes, bounded | `Degraded`, then terminal after the budget |

**Ambiguous failures are the dangerous ones.** A timeout after the request was
sent may mean the job ran and the *response* was lost. Retrying then
double-executes. So submits carry an **idempotency key** and the server
de-duplicates — the same key A6 needs for resume, built once.

**Design:**

- **Explicit connection state**, not a boolean: `Connected` /
  `Degraded { attempt, of, next_retry_in, last_error }` / `Terminal { reason }`.
  Exposed as a field the caller can read *and* a callback it can subscribe to.
- **Bounded exponential backoff with jitter**, plus a **total budget**. Infinite
  retry is not resilience; it is a hang with extra steps. When the budget is
  spent, surface one clear terminal error naming what failed and for how long.
- **Circuit breaker**: after N consecutive failures stop hammering, report, and
  probe at a slow interval — a laptop reconnecting should not DoS the DGX.
- **Status must reach a human**: structured transitions on **stderr**, never
  stdout (Q8), and a progress line for `aria train` along the lines of
  `epoch 3/10 batch 12/100 — remote degraded, retry 2/5 in 4s`. "Know wtf the
  status is" means the line is printed *while degraded*, not after it resolves.
- **Resume beats restart.** With A6 job IDs, reconnect re-attaches to a job
  already running rather than resubmitting it. Without A6, at minimum keep local
  training state checkpointed so a terminal disconnect costs one batch, not the
  epoch.
- **Partial batch results.** If a 256-row batch dies at row 200, the client must
  learn *which* rows completed — the same row-index preservation Q5 requires of
  the gradient route. Losing 200 good rows because row 201 failed is the
  expensive version of this bug.
- **Heartbeats disambiguate slow from dead** (A6): without them a long job is
  indistinguishable from a hung connection, and every timeout policy is guesswork.
- **Local fallback is opt-in and loud.** Silently continuing on CPU when the
  remote vanishes changes performance by orders of magnitude and hides an
  outage; if offered, it must be requested explicitly and reported on every use.

Acceptance: kill the server mid-batch and the run reports `Degraded` and
survives its restart; kill it permanently and the run ends with one clear
terminal error, not a hang; a 413 is never retried; an interrupted submit
replayed with the same idempotency key executes **once**; the status is
observable *during* the outage, not only in hindsight.

### A8b. `--wait forever` — the commute mode

**PLAN ONLY.** Scenario: the client is a laptop that will *definitely* go away —
home → office, a train, a plane — while a mega-batch runs on the DGX. The wanted
behaviour is "never give up on the connection, but never leave me guessing".

This **refines** A8's "infinite retry is a hang with extra steps". That holds
for retry that is *silent and default*. Retry that is *explicitly requested and
continuously reported* is a different thing, and it is the right default for a
long batch followed from a laptop. So: opt in, and pay for it in loudness.

- `--wait forever` (also `--wait 2h`, and `OMEGA_WAIT`) on the follow/collect
  path. Default stays bounded; forever is a choice the operator makes.
- **Forever applies to connectivity, never to logic.** Waiting cannot fix a
  `413` (too large), `401`/`403` (token), `400` (bad request), or a `410 Gone`
  (retention expired). Those stay terminal even in this mode, with the reason
  stated. A mode that waits forever on a bad token is a hang.
- **Backoff caps and stays there** (~60 s). A laptop asleep for six hours must
  not wake into an hour-long backoff, and must not hammer the server either.
- **Wake detection.** Lid-close suspends the process; on resume, a large
  wall-clock jump means "probe now" rather than serving out the remaining
  backoff. Without this the commute case reconnects minutes late for no reason.

**Loud means continuously visible, not one line at the start:**

```
⚠ DISCONNECTED 00:14:32 — batch a1b2c3 · 41,337/100,000 done · retry in 47s
  last contact 14m ago · server-side work continues · results retained until 18:40
✓ RECONNECTED after 14m32s — resuming from row 41,339
```

- Status transitions and the live counter go to **stderr** (Q8), so piping
  results to a file still shows the state in the terminal.
- `--status-file FILE` writes the same state as JSON continuously, so a run can
  be checked from another shell — or after the fact, which is the actual need
  when you were on a plane.
- The **final report states the disruption**: total disconnected time, number of
  reconnects, and whether any row was lost. A run that survived a two-hour
  outage should say so rather than quietly looking like a clean run.
- Exit codes distinguish *completed*, *completed with failed rows*, and
  *terminated while disconnected*.

**Two interactions that decide whether this actually works:**

1. **Server retention must outlive the absence.** `OMEGA_BATCH_TTL` defaulting
   to hours silently defeats a transatlantic flight: the client returns to
   `410 Gone`. So the submit response must state the retention deadline up
   front, `--wait forever` should request an extended/pinned retention, and the
   client must warn *before* the deadline passes — not discover it afterwards.
   This is the single most likely way the feature disappoints in practice.
2. **Token TTL outlives nothing by default.** A long-lived follow can outlast
   its bearer token, turning a reconnect into a `401` that is (correctly)
   terminal. Either the follow path refreshes, or `--wait forever` refuses to
   start when the token expires before the batch plausibly finishes — warned at
   submit time, not discovered at hour six.

Acceptance: start a batch, sever the network for longer than several backoff
cycles, and the client keeps reporting elapsed-disconnected the whole time;
suspend/resume the machine and it probes promptly on wake; restore the network
and it resumes from the correct row with nothing recomputed; a token or
retention expiry ends the run with an explicit reason rather than an endless
wait; the final report names the outage.

### A9. Durable batches — disconnect, reconnect later, don't redo the work

**PLAN ONLY.** The driving scenario: submit a mega-batch of many jobs, the
server (or the tunnel, or the laptop lid) drops, and hours later a *new* client
process re-attaches, sees what finished, and collects the rest. Today that is
total loss — the batch lives only in the HTTP request, so the connection **is**
the job. A8's retry/backoff keeps a *live* run alive across a blip; it cannot
help a client that exited, and it cannot survive a server restart. This can.

**Consequence: job state must outlive the connection *and* the process.** The
server already carries `rusqlite` and an `OMEGA_DB_PATH` (the registry uses it),
so batch state belongs there rather than in a `HashMap` — which also makes
"reconnect later" work across a server restart, not just a client one.

**Model.**

- `POST /v1/quantum/batch` → `{batch_id}` **immediately**; work proceeds
  server-side, decoupled from the caller. Accepts a client **idempotency key**:
  resubmitting the same mega-batch after a crash returns the *same* `batch_id`
  instead of re-running 200 completed rows. Without this, the retry that A8
  performs is exactly what destroys the work.
- Rows are identified by **index** and completed **incrementally**, each result
  committed as it lands. A server restart then costs at most the in-flight
  chunk, not the batch. Chunk size is the checkpoint granularity — state it.
- `GET /v1/quantum/batch/{id}` → the status the client needs to reason:
  ```jsonc
  { "state": "running",        // queued | running | complete | failed | cancelled
    "total": 100000, "completed": 41337, "failed": 2,
    "next_pending": 41339,      // resume cursor
    "started_at": "...", "updated_at": "...",
    "errors": [{"row": 900, "error": "circuit[900]: bad observable"}] }
  ```
- `GET /v1/quantum/batch/{id}/results?from=&to=` → completed rows in **index
  order** (Q5), so a reconnecting client fetches only what it lacks.
- `DELETE /v1/quantum/batch/{id}` → cancel remaining work and release
  reservations. A reconnecting client must be able to *stop* a batch it no
  longer wants, or an abandoned mega-batch occupies the box forever.
- WS (A6) becomes an **optimisation over the same state**: push completions as
  they happen, with the polling endpoints as the fallback. The durable record is
  the source of truth; the socket is a delivery convenience. That ordering is
  what makes "reconnect later" work at all.

**Per-row failure must not fail the batch.** Row 900 having a bad observable
must not discard rows 0–899. Each row carries its own status, and the batch
completes with `failed: 2` rather than a 400 for the whole submission — a
deliberate departure from the current expectation route, which aborts everything
on the first bad row (`quantum_bridge.rs`). That change is the entire point of
the feature for a mega-batch, and it is why A1's partial-failure semantics
should be settled here rather than separately.

**Retention is a resource, and A7 governs it.** Stored results occupy space:
100k rows of `⟨O⟩` is trivial (~800 KB), but 100k × 120 per-row gradients is
~100 MB per batch, and abandoned batches accumulate.
- `OMEGA_BATCH_TTL` (default hours, not days) and `OMEGA_MAX_BATCH_RESULTS`.
- Results are droppable once **acknowledged** by the client (`?ack=true` on
  fetch), so a well-behaved client bounds the store without waiting for the TTL.
- Admission accounts for stored results, else a mega-batch evades the governor
  by parking its output — the same class of bypass A7's review found.
- Eviction is **loud**: a client asking for evicted rows gets an explicit "410
  Gone, retention expired", never silence or an empty array that reads as "no
  results".

**Acceptance** (the scenario, tested end to end): submit 1000 rows; kill the
server at row ~400; restart it; a **fresh client process** re-attaches by
`batch_id`, sees `completed ≈ 400` with a correct `next_pending`, and collects
all 1000 results with **no row computed twice and none missing**. Re-submitting
with the same idempotency key returns the same `batch_id` and does not restart
the work. Cancelling mid-run frees the governor reservation.

### A9b. QAS and QML batch shapes — where the naive batch model breaks

**PLAN ONLY.** A9 as written assumes "N rows of the same circuit with different
parameters". That is the QML *inference* shape. The two workloads actually
queued for the DGX — architecture search and training — violate it in ways that
would make the feature useless or actively harmful. Four corrections, the first
of which is a fix to A7 as already shipped.

> **Architecture search is an external client, and that is a design constraint,
> not a footnote.** QAS is implemented outside this repository — in a
> closed-source consumer, and equally in third-party setups that use Aria and
> know nothing about any particular one. So:
>
> - **No search logic in the server.** No TPE, no Hyperband, no pruning policy.
>   The server supplies *primitives* — heterogeneous rows, per-row cancel,
>   per-row progress, index-preserving results, durable opt-in — and the client
>   owns the strategy. Embedding a search policy here would fit one consumer and
>   obstruct every other.
> - **The batch API is a public contract** (K3 wire discipline): versioned,
>   additive, and documented for clients whose source this repo cannot see. It
>   cannot be quietly reshaped later to suit one caller.
> - **This repo cannot test the real QAS workload**, because it does not have it.
>   So CI must carry a *representative synthetic* heterogeneous sweep — varied
>   qubit counts, depths and backends, one deliberately oversized row, one pruned
>   mid-flight — and the in-repo `aria tune` (TPE study) serves as the exemplar
>   consumer proving the primitives suffice for a real search driver.
> - Corollary: **every requirement below must be justifiable from the API shape
>   alone**, never from knowledge of any particular client's internals.

**1. Reserve per chunk, not per batch. (Corrects A7.)** Admission currently
takes **one** reservation for a whole batch, priced at its worst row
(`admit_batch`). For a 20-row inference batch that is fine. For a 72-config QAS
sweep it is not: configs are *heterogeneous* — different qubit counts, depths,
and backends — so the batch is priced at its single largest config and holds
that reservation for the entire sweep, which may be hours. On a shared box that
starves every other tenant for the duration, and it over-reserves for all but
one config.
→ A durable batch must acquire and **release** its reservation per chunk as it
executes, not hold a worst-case reservation for its lifetime. The per-batch
reservation was correct only because batches were short and synchronous; A9
makes them neither.

**2. The template/parameter-matrix optimisation (A0 item 1) does not apply to
QAS.** It assumes every row shares one ansatz and differs only in bound
parameters. QAS rows are *different circuits by construction* — that is the
search. So QAS keeps paying full topology cost per config, and the transfer
budget must say so rather than promising a 90× reduction that only materialises
for QML inference. QML *training* does benefit, since every row there is the
same ansatz.

**3. Durability must be opt-in per submission, or retention explodes.** A QML
run is thousands of steps × many epochs, each a forward+gradient. Durably
storing every step's results is both pointless — the trainer consumes each
immediately — and a retention blow-up on the order of `steps × P` floats.
Meanwhile a QAS sweep is exactly what *should* be durable: few trials, long,
expensive, and painful to redo.
→ `durable: true` on submit. Training steps stay ephemeral and stream; search
trials persist. Defaulting everything to durable would turn a training run into
a disk-filling machine.

**4. Cancellation must be per-row, not just per-batch.** `aria tune` runs TPE
studies, and any serious search prunes: Hyperband/ASHA kill unpromising trials
early. A search that can only cancel the *whole* sweep cannot prune, which
removes most of the point of a search.
→ `DELETE /v1/quantum/batch/{id}/rows/{i}` (or a bulk form), releasing that
row's reservation and marking it `cancelled` distinctly from `failed` — a pruned
trial is not an error, and reports must not conflate the two.

**Two further QAS/QML-specific hazards:**

- **Head-of-line blocking.** One 30-qubit config in a sweep of 28-qubit configs
  can block the queue behind it while capacity exists for the smaller ones.
  Execution should be free to proceed out of order when a row does not fit *now*
  — with results still returned **in index order** (Q5), so out-of-order
  execution never becomes out-of-order reporting.
- **Long trials need intra-trial progress.** A QAS trial is itself a full
  training run of many steps. Batch-level `completed: 3/72` is useless for hours
  at a time; a trial needs a progress field of its own, or A8b's commute-mode
  display will show a frozen counter and look hung when it is working.

Acceptance: a heterogeneous 72-config sweep does not hold a worst-case
reservation for its duration and does not starve a concurrently-submitted small
job; pruning one trial frees its capacity immediately and reports `cancelled`,
not `failed`; a training run submitted non-durable leaves no stored results
behind; a sweep with one oversized config still completes the rest.

### A10. Formal models — TLA+ for the protocols, Lean 4 for the arithmetic

**PLAN ONLY.** The scheduling and admission logic is concurrent, failure-prone,
and mostly *untestable by example*: starvation, permit leaks after a crash, and
exactly-once execution across a reconnect are properties about all interleavings,
not about one run. Tests sample that space; a model checker exhausts it (over a
small instance). The repo already runs Lean 4 in CI with sorry-free axiom checks
(`ARIA_LEAN`), so this extends existing practice.

**Tool split by what each is actually good at:**
- **TLA+ / TLC** — temporal and concurrent behaviour: admission, scheduling,
  batch lifecycle, crash and reconnect.
- **Lean 4** — the pure arithmetic the governor depends on, where a proof is
  cheaper and permanent.

#### Honest scope first — what these will *not* catch

The A7 defect the adversarial review found (MPS priced by tensors while the
backend densifies) would **not** have been caught by any model here. The
protocol was fine; the *input* to it was wrong. A model checks that "no admitted
set exceeds capacity" given the weights it is handed — it cannot know that a
weight is a lie about what `omega-backend-mps` allocates. That gap stays closed
by reading the backends and by the differential tests, not by TLA+. Saying so up
front keeps the models from becoming the same false confidence the under-pricing
already produced once.

#### `proofs/tla/Governor.tla` — admission and scheduling

State: per-pool capacity, admitted jobs with weights, pending queue.
Actions: `Submit`, `Admit`, `Reject`, `Release`, `Cancel`, `Crash`.

| property | kind | why it matters |
|---|---|---|
| `∀ pool: Σ weight(admitted) ≤ capacity(pool)` | safety | the governor's entire reason to exist |
| no double `Release` | safety | a permit released twice silently inflates capacity — and crash/retry paths are exactly where that happens |
| unified pool aliasing: Host and Device charges hit the **same** counter when unified | safety | the A7b double-counting failure, stated as an invariant instead of a table |
| a job that fits is eventually admitted | **liveness** | this is the one that should fail today |
| pruned/cancelled rows release capacity | safety | A9b pruning |

The liveness check is the point. A7 uses `try_acquire` with no queue, so a steady
trickle of small jobs can starve a large one indefinitely while `Retry-After`
promises a retry that never succeeds (A7 gap 3). TLC under fairness assumptions
should produce that counterexample trace — turning "I think this can starve"
into a concrete interleaving, and justifying the bounded queue rather than
arguing for it from intuition.

#### `proofs/tla/DurableBatch.tla` — A9 lifecycle across failure

State: per-row status, chunk commit point, client connection, server liveness.
Actions: `SubmitBatch` (with idempotency key), `StartChunk`, `CommitChunk`,
`ServerCrash`, `ServerRestart`, `ClientDisconnect`, `ClientReconnect`, `Fetch`,
`Ack`, `Evict`, `CancelRow`.

| property | kind |
|---|---|
| **exactly-once**: no row computed twice, no row lost, across arbitrary crash/reconnect | safety — *the* A9 requirement |
| resubmit with the same idempotency key does not duplicate work | safety |
| results delivered in index order however execution was ordered (A9b out-of-order) | safety — Q5 |
| reservations are released on crash (no permit leak across restart) | safety |
| `next_pending` after restart is consistent with committed chunks | safety — validates the checkpoint-granularity claim |
| every row eventually reaches a terminal state | liveness |

Model at a small scale (≈5 rows, 2 chunks, ≥2 crashes) — exhaustive over that
instance, which is where these bugs live, not at scale.

#### Lean 4 — governor arithmetic (`proofs/lean4/QuantumProofs/Governor.lean`)

Small, permanent, and directly motivated: **I already shipped an off-by-one in
`default_qubit_ceiling`**, which advertised a ceiling one qubit wider than the
budget allows. That is exactly the class a proof closes for good.

- `to_mib(b) * MiB ≥ b` — permits round **up**. Rounding down would let the
  admitted set exceed the budget by up to 1 MiB per job; with many jobs that is
  unbounded.
- `default_qubit_ceiling c` = the greatest `n` with `cost(n) ≤ c` — the shipped
  bug, stated as a theorem.
- Cap composition is `min`, hence `effective ≤ every input cap` (A7c rule 2 —
  the rule easiest to write backwards, where the failure is silent
  over-admission).
- `estimate_peak_bytes` is monotone in `n` and never wraps (the `checked_*`
  chain is total, or returns `None`).
- `fock_dim m p = C(m+p-1, p)`, with the iterative form equal to the closed form
  — a wrong photonic count is a wrong budget on the one backend that cannot be
  bounded any other way.

#### CI integration (K13)

- Lean: add to the existing `ARIA_LEAN=1` stage, sorry-free-checked like the
  circulant and noise theorems already are.
- TLA+: new **opt-in** `ARIA_TLA=1` stage; TLC needs a JVM, so it must **skip
  cleanly** when absent, exactly like the Qiskit cross-check. The default
  `./ci.sh` must not acquire a Java dependency.
- Both record what was checked and at what bound, so "verified" never reads as
  broader than the instance actually explored.

Sequenced after A9/A9b are specified but **before** they are implemented — the
models are cheapest as a design instrument, and the starvation counterexample
should shape the queue design rather than post-rationalise it.

### A11. Timing breakdown — where did the wall clock actually go?

**PLAN ONLY.** With work split across a laptop and a remote box, the first
question about a slow sweep is *"is this the network, the queue, or the
simulation?"* — and nothing currently measures it. Without that split, tuning is
guesswork: the fix for transfer-bound work (A0's template encoding) and the fix
for compute-bound work (a bigger box) are completely different, and today there
is no way to tell which you have.

This is also the **measurement infrastructure A0 already assumes**: its
acceptance criterion is "record bytes-on-the-wire and request count for a 256-row
forward+gradient step". That cannot be done today. So A11 lands **with or before
A2**, or the batching win gets asserted rather than demonstrated.

**Phases to separate** (per request, aggregated per batch/epoch):

| phase | who | answers |
|---|---|---|
| bind + serialise | client | is JSON encoding the bottleneck? |
| connect / TLS | client | tunnel setup cost per request → keep-alive value |
| upload (+ bytes) | client | the A0 payload question, directly |
| **server queue / admission wait** | server | capacity contention vs real work |
| **execute** (per row) | server | actual simulation cost |
| serialise response (+ bytes) | server | the statevector-return blow-up |
| download | client | |
| deserialise | client | |
| **degraded**: backoff + disconnected | client | A8/A8b — time lost to the network being gone |

**Design.**

- **Server → `Server-Timing` header** (a standard, so existing tooling reads it):
  `admit;dur=0.4, exec;dur=812.5, serialize;dur=31.2`. Batch/row granularity in
  the JSON body, since a header cannot carry 256 rows.
- **Client accumulates a `Timing` struct** per call — phase durations plus
  `bytes_up` / `bytes_down` / `requests` — and aggregates per batch and per run.
- **Use a monotonic clock for durations**, wall clock only for timestamps. NTP
  steps otherwise produce negative or absurd intervals, and this feature exists
  to be trusted.
- **Distinguish "the machine was asleep" from "the server was slow."** In
  commute mode (A8b) a suspended laptop shows a huge wall-clock gap with no
  monotonic progress; counting that as server latency would make every commute
  look like an outage. Divergence between the two clocks *is* the signal.
- **Always-on summary, opt-in detail.** A handful of `Instant::now()` calls per
  request is free; per-row detail sits behind `--timing-detail` so a 100k-row
  batch does not accumulate 100k records by default.
- **Report to stderr** (Q8), with `--timing-report FILE` for JSON. The end-of-run
  summary should lead with the ratio that decides what to do next:

  ```
  wall 14m02s │ transfer 11m18s (80%) │ exec 2m31s (18%) │ queue 9s │ degraded 4s
              └─ 256 requests, 1.1 GB up, 2.4 MB down
  remoting overhead 11m31s (82%) — transfer-bound
  → batch these rows (1 request instead of 256)
  ```

  **The headline number is "is remoting a drag, or a detail?"**, so the report
  must split wall clock into *work you would pay anywhere* (`exec`) and
  *overhead you pay only because this is remote* (transfer + queue + degraded +
  connection setup). A percentage against a named verdict —
  `transfer-bound` / `compute-bound` / `contention-bound` — is the one line
  someone reads before deciding whether to fix the client, buy a bigger box, or
  stop worrying. Without that split, an 80%-overhead run and an 8%-overhead run
  look identical: both are just "slow".

- **Per-row execution time is what a search driver wants**, since it is the cost
  signal for its own scheduling — expose it in batch results (A9), not only in
  aggregate.

Acceptance: for a known workload the phases sum to wall clock within a small
tolerance (nothing unaccounted); a deliberately throttled link shows up as
transfer, not exec; a suspended-and-resumed client does not report the sleep as
server latency; and the A0 batching change is demonstrated by a before/after
byte count rather than a claim.

### A12. Optional cluster manager — same protocol, resources per node

**PLAN ONLY.** Scale the executor from one box to several without changing the
client at all. A coordinator speaks **exactly the protocol a single server
speaks**, so a caller cannot tell the difference and nothing in `remote.rs`,
`aria-py` or a third-party client needs to know a cluster exists. If the client
has to be cluster-aware, the design has failed.

**Pull, not push — because admission authority must stay on the node.** The
coordinator does not assign rows. Idle workers **steal** them from the batch's
pending set, and a worker takes a row only if **its own governor** admits it
(A7/A7b/A7c, unchanged). Push scheduling would require the coordinator to model
every node's live capacity, which is stale the moment it is read and wrong the
moment a job finishes early. Pull makes each node the authority on its own
memory, which is the only place that fact is ever correct.

**Nodes are heterogeneous, and that changes what a refusal means.** A cluster
may hold a DGX Spark (unified ~128 GB), a discrete-GPU box, and a spare laptop.
A row that fits one may fit none of the others. So:

- **413 only if *no* node could ever run it** — the ceiling is the max over
  nodes, not the min, and not the coordinator's own memory.
- **429 while some node could run it but none is free** — the existing
  distinction survives, which is what keeps client retry logic (A8) correct.
- The refusal should name the largest node, so "needs 64 GB; the largest node
  has 24 GB" is actionable rather than mysterious.

**Rows are the unit of work, so A9 is a prerequisite.** Durable batches already
give per-row identity, status, incremental commit and a resume cursor — exactly
the stealable work item. Building cluster support before A9 would mean inventing
that twice.

**Failure handling turns exactly-once into at-least-once + idempotency.** A
worker that dies mid-row must not swallow it:

- rows are **leased**, not assigned; a lease expires and the row returns to the
  pending set;
- so a row can execute **twice** (slow worker plus re-lease), which is fine for
  a pure circuit evaluation but must be stated, and the result write must be
  idempotent — last writer wins, same value;
- a worker heartbeats while running so a long row is not mistaken for a dead one
  (the same signal A6 needs, built once).

**Observability has to gain a node dimension or it becomes useless.**
`/health` aggregates per-node pools rather than summing them into one fictional
budget — a cluster with 4×24 GB cannot run a 96 GB job, and an aggregate number
would imply it can. A11's timing gains a `node` attribution, so "which node was
slow" is answerable; without it a cluster turns every performance question into
a guess.

**Explicitly out of scope:** cross-node parallelism *within* one circuit
(distributed statevector). That is a different and much harder problem —
communication-bound rather than embarrassingly parallel. This is a work
distributor for many independent circuits, which is exactly the shape of
architecture search and batched QML inference, and it should say so rather than
imply distributed simulation.

Sequenced after A9 (durable batches) and A10 (the models — `Governor.tla` grows
a node dimension, and the lease/steal protocol is precisely the kind of thing
worth model-checking before it is written).

## Part E — tsim & ppvm bridges (`fixes/TSIM-PPVM.md`, new 2026-08-06)

Two QuEra simulators, to be reached through the **existing `omega-bridges`
JSON-over-stdio subprocess protocol** — the request already surveyed the three
integration surfaces and chose this one. I agree with the choice, and the
reasoning is worth keeping: a C-ABI plugin's gate-only vtable has no expectation
entry point and its thread-safety contract is hostile to a Python GIL; HTTP
impersonation buys nothing because the wire IR is the same closed gate enum.

Verified against the tree: `bloqade.rs` is **34 lines**, the runner protocol
(`runner.rs`) already does `$OMEGA_BRIDGE_<SLUG>_CMD` → PATH → dev-fallback
discovery and maps a missing tool to `Unavailable` rather than an error, and
`python/` already carries three runners to mirror. So the estimate of ~150 lines
of Rust total is credible.

### E1. ppvm — a *validator*, not a new capability

The request is explicit and correct: **ppvm and `omega-backend-pauliprop` are
the same algorithm family** (Heisenberg Pauli-sum propagation with coefficient
truncation). So ppvm's value here is as an **independent numeric reference for
pauliprop** — precisely the kind of cross-check that has already earned its keep
twice this session (Qiskit at 4.441e-16, PyMatching at 100% shot-for-shot).

That framing should drive the work: the deliverable is a **differential test**,
not a backend users select. Ship it the way the Qiskit cross-check is shipped.

Note ppvm is **Rust**, so `ppvm-pauli-sum` could later be a direct git Cargo
dependency and validate in-process, skipping the subprocess entirely. Start with
the bridge (cheap, uniform), keep that door open.

### E2. tsim — genuinely new capability

Stabilizer-rank (ZX) decomposition: noisy Clifford+T sampling at scales the MPS
backend cannot reach. Unlike ppvm this is not duplicated in-tree.

**But the bridge speaks QASM2 + counts only**, and tsim's distinctive value is
*detector/observable* sampling for QEC. Through this surface it arrives as a
plain noisy sampler with its headline feature inexpressible. That is worth
saying out loud rather than discovering later: E2 buys scale, not QEC semantics.
A detector-aware extension is a separate piece of work, and only worth doing if
QEC sampling actually lands here.

### E0. No Python in a validation loop — this reshapes E1 and E2

**Directive 2026-08-06:** these emulators may also be used **for validation**,
and house doctrine is Rust core, Python at the edge only — no Python VM inside a
training or scoring loop. That is not a detail; it decides what each bridge can
ever be used for.

The bridge protocol spawns a **subprocess per call**. That is fine for a
one-shot differential check and unacceptable inside a loop: process spawn plus
interpreter start dominates, and it puts a Python VM on the critical path of
every iteration. So a single mechanism cannot serve both uses.

**Two tiers, and each emulator lands in a different one:**

| use | mechanism | acceptable? |
|---|---|---|
| One-shot / CI differential check | subprocess bridge | **yes** — it runs once, off the hot path |
| In-loop validation (per step, per row, per trial) | must be **in-process Rust** | subprocess is **not** acceptable |

- **ppvm is already Rust** (`ppvm-pauli-sum`, `ppvm-tableau`). So the in-process
  tier is available *without porting anything*: take it as a git Cargo
  dependency and call it directly.

  **Verified 2026-08-07** against the upstream repo rather than assumed — it is
  public and git-only (no crates.io), and documents exactly this usage:

  ```toml
  [dependencies]
  ppvm-pauli-sum = { git = "https://github.com/QuEraComputing/ppvm" }
  ```

  with `PauliSum` / `GeneralizedTableau` and an `overlap_with_zero()` entry
  point for the overlap that expectation values are built from. So the
  in-process tier needs **no Python, no subprocess, and no port** — which makes
  the bridge genuinely scaffolding rather than the destination.

  **Both open questions resolved 2026-08-07 by probing, not reading:**

  1. **Builds on `aarch64-darwin`** — a throwaway crate with the git dependency
     compiled clean in 13 s (`ppvm-traits`, `ppvm-pauli-word`,
     `ppvm-pauli-sum` at `661fc66f`). No system deps, no build-script surprises.
     Probed in a scratch crate so a failure could not break the workspace.
  2. **Truncation is a pluggable `Strategy`** (`sum/data.rs::truncate` →
     `ppvm_traits::traits::Strategy`), documented as covering
     "coefficient-magnitude, max-weight, combinations". Expectation comes from
     `PauliSum::overlap` (`sum/trace.rs:31`).

  **That lines up with pauliprop almost exactly.** In-tree,
  `PauliSum::truncate(coeff_min, max_weight, max_freq)`
  (`omega-backend-pauliprop/src/pauli.rs:123`) drops on coefficient magnitude,
  Pauli weight, and PauliPropagation.jl's `max_freq` split-frequency axis, and
  accumulates discarded L1 mass into `dropped_mass` as an error budget.

  So the differential test is configured by matching a ppvm `Strategy` to
  `coeff_min` + `max_weight`. Two caveats decide whether a disagreement *means*
  anything, and both must be settled before reporting numbers:

  * **ppvm has no `max_freq` equivalent — confirmed.** Its strategies are
    `NoStrategy`, `MaxPauliWeight`, `CoefficientThreshold`, `MaxLossWeight` and
    `CombinedStrategy` (`ppvm-pauli-sum/src/strategy.rs`); there is no
    split-frequency axis. So the comparison must **hold `max_freq` non-binding
    on the pauliprop side** (`None`, or high enough never to fire). Two
    implementations truncating on different axes disagree for reasons unrelated
    to correctness, which would make the cross-check actively misleading rather
    than merely weak.

    That gives the configuration mapping outright:

    | pauliprop | ppvm |
    |---|---|
    | `coeff_min` | `CoefficientThreshold` |
    | `max_weight` | `MaxPauliWeight` |
    | both together | `CombinedStrategy(CoefficientThreshold, MaxPauliWeight)` |
    | `max_freq` | **no equivalent — must not bind** |

    Which also bounds what the check can claim: it validates pauliprop's
    coefficient/weight truncation, **not** its `max_freq` axis. That axis stays
    covered only by in-tree tests, and the cross-check's write-up should say so
    rather than implying full coverage.
  * **Derive the tolerance from `dropped_mass`, do not pick one.** A difference
    inside pauliprop's own truncation error budget is not a defect, and
    reporting it as one trains people to ignore the check. The bridge remains the cheap way to get a
  first number, but **the destination for ppvm is a direct dependency**, not the
  subprocess. Re-sequence accordingly: bridge first only if it is genuinely
  faster to stand up, and treat it as scaffolding rather than the deliverable.
- **tsim is Python on JAX/XLA.** There is no in-process Rust path, so tsim
  **cannot** be used for in-loop validation as it stands. Using it that way
  requires porting the method (ZX stabilizer-rank decomposition) to Rust/C++ —
  a real project, not a wrapper. Until then tsim is restricted to one-shot CI
  cross-checks, and that restriction should be stated where someone would
  otherwise reach for it.

**Consequences to honour rather than discover:**

1. A validator invoked per training step must not spawn a process. If ppvm's
   direct dependency is not ready, in-loop validation waits — it does not
   silently fall back to the subprocess.
2. A tsim port is a **separate, scoped project** with its own justification. It
   should not be smuggled in as "part of the bridge work", and it is only worth
   it if in-loop stabilizer-rank validation is actually needed.
3. Both remain legitimate as **CI-time** cross-checks today. That is where the
   Qiskit and PyMatching checks live, and it is already proving its worth.

### E3. Porting tsim: use formal methods, not hope

E0 established that in-loop validation with tsim requires porting ZX
stabilizer-rank decomposition from Python/JAX to Rust. That port is the risky
part, and the risk is specific: **a hand port validated only against the
original inherits the oracle problem.** If the port and the original are wrong
the same way, every differential test agrees. This session has now produced four
separate instances of confident output with nothing behind it — a subtle
sampling algorithm is exactly where the fifth would hide.

Sibling tooling exists for precisely this shape of problem:

| tool | what it gives |
|---|---|
| `../leanlift` | *"Lift a function into a Lean 4 model and prove it's the same function — by bit-exact differential execution."* Plus `verify-creusot.sh` (deductive) and `verify-kani.sh` (bounded model checking) for Rust-level properties. |
| `../lean4-skills` | Lean 4 tooling/plugins; pulled 2026-08-06 (105 files changed, so it is moving). |
| `../le-harnais-v0` | Harness with its own `lean/`, `ci.sh` and skills; a working example of the pattern. |

This repo already runs Lean 4 in CI with sorry-free axiom checks
(`ARIA_LEAN=1`, the circulant / noise / HHL-QSVT theorems, plus
`verification/Verification/Backend/PauliAlgebra.lean`), so this extends
established practice rather than importing a new discipline.

**Approach — port kernel by kernel, verifying each before moving on:**

1. **Specify the core in Lean first**, from the method paper
   (arXiv:2604.01059), not from the Python source: the stabilizer decomposition
   of magic states, the sum over stabilizer terms, and the sampling step.
   Writing the spec from the paper is what keeps it an *independent* statement
   rather than a transcription of whatever the implementation happens to do.
2. **Port one kernel at a time to Rust**, and use `leanlift` to prove the Rust
   function agrees with its Lean model by bit-exact differential execution.
   Do not proceed to the next kernel until the current one is lifted.
3. **Creusot/Kani for the Rust-level obligations** a Lean equivalence proof does
   not cover: no overflow in the rank/index arithmetic, bounds on the term
   vector, termination of the decomposition loop.
4. **Keep the Python bridge as a second oracle** during the port — belt and
   braces — but the Lean model is the authority. If they disagree, the bug is
   not automatically in the Rust.

**The honest limit, same one recorded for the TLA+ models.** A proof that the
Rust matches the Lean model is only as strong as the model. A Lean spec that
misstates the algorithm yields a proof that is *valid and useless*. So the spec
must be checked independently: against the paper's stated properties, and
against tsim's own outputs on cases where the answer is known analytically.
Formal methods raise the floor here; they do not remove the need for a
cross-check.

**Do this only if in-loop tsim validation is actually wanted.** It is a
multi-week project. If tsim stays a CI-time cross-check (E2), the Python bridge
is sufficient and none of this is needed — and that should be the default
assumption until someone needs otherwise.

### Sequencing and acceptance

1. **E1 ppvm bridge + pauliprop differential test.** Highest value per line: it
   turns an in-tree backend from self-consistent into independently checked.
   Acceptance: agreement on the qualifying fixture subset, **with the number of
   qualifying fixtures reported** — a cross-check that silently tests 3 cases is
   worse than none.
2. **E2 tsim bridge** as a noisy sampler, documented as such.
3. Detector-aware tsim: **only** if QEC sampling materialises here.

Constraints carried from the request, all consistent with house rules:
- Out-of-subset gates **refuse loudly** (`kind: "<slug>-unsupported-gate"`),
  never silently skip — the same discipline as B0's CV refusal.
- A missing tool is `Unavailable`, never a hard failure, so the default `./ci.sh`
  stays green on a machine without JAX or a ppvm build.
- **A blocked integration ships as a findings note, not a fake-green bridge.**
- Install friction is real (tsim pulls JAX; ppvm is git-only) — each gets its own
  venv, as with the two Qiskit ones.

## Part F — Photonic performance and GPU acceleration

**PLAN ONLY.** Answers three questions asked 2026-08-07, with measured figures
rather than estimates.

### F0. Do we have GPU acceleration for photonics? **No.**

GPU crates exist for `statevector` (Metal / CUDA / OpenCL), `mps` (Metal /
CUDA) and `pauliprop` (Metal / CUDA). **Neither photonic backend has any** —
`omega-backend-photonics` (DV) and `omega-backend-cv` are CPU-only with no
feature flags for it. That is the honest answer, and it is the gap this part
plans.

### F1. Memory first, because it decides what a GPU could even help with

| model | dimension | figure |
|---|---|---|
| DV photonics (permanents / SLOS) | `C(m+p-1, p)` states, `p = ceil(m/2)` | 16 modes → 490 k states, 0.05 GB · **20 modes → 20.0 M, 2.4 GB** · 26 modes → 5.41 G, **766 GB** |
| CV truncated Fock | `cutoff^modes × 16 B` | 6 modes / cutoff 8 → 262 k amps, **4 MB** · 8/8 → 0.25 GB · **10/8 → 16 GB** |

Two conclusions follow immediately, and they point in opposite directions:

* **CV at the planned envelope is tiny.** `PLAN-CV-BACKEND`'s target
  (`n_modes ≤ 6`, `cutoff ≤ 8`) is **4 MB**. A GPU would be pure overhead —
  kernel launch and transfer dominate a 262 k-element state. **Do not
  GPU-accelerate CV for the search grid.** It becomes interesting only past
  ~9 modes, where the state crosses ~2 GB.
* **DV is the one with a cliff.** Between 20 and 26 modes it goes 2.4 GB →
  766 GB. GPU memory does not rescue that; only a better algorithm does. So the
  useful GPU target for DV is the **permanent evaluation** at moderate sizes,
  not the state.

*(Correction recorded: an earlier note put 26 modes at ~300 GB from
`C(38,13) ≈ 2.3e9`. The true value is `5,414,950,296` ≈ 766 GB — 2.4× worse.
Understating a memory cost inside a resource governor is the dangerous
direction, so the figure is now pinned by a test.)*

### F2. Benchmark against Perceval and piquasso — measure before optimising

Both references are already available: a Perceval bridge exists
(`bridge-perceval`), and piquasso 8.0.1 is installed in `.venv-piquasso` for the
CV cross-check. So the comparison costs setup, not integration.

What to measure, and against what:

| our backend | reference | shapes |
|---|---|---|
| `omega-backend-photonics` (DV) | **Perceval** (Quandela) | HOM, MZI, then 8/12/16/20 modes — the run that finds our cliff |
| `omega-backend-cv` | **piquasso** `PureFockSimulator` | the search grid: 3–6 modes × cutoff 4–8 |

Report **wall time and peak RSS together**. Time alone is misleading here: a
backend that is 2× faster while holding 4× the memory is worse on a shared box,
and A7's governor prices memory, not speed. Reuse A11's phase split so
"Perceval is slower" cannot be confused with "the Python bridge subprocess is
slower" — the bridge spawns a process per call, which would otherwise dominate
every small shape.

**Expected outcome worth stating in advance**, so a surprise is legible: for the
CV grid we should be *comparable* to piquasso, not dramatically faster — both
are dense truncated-Fock statevector evolution, and the arithmetic is the same.
A large win would more likely indicate a measurement error or a different
truncation than a real advantage. That is the honest prior, and the same
discipline the QML docs already apply to classical baselines.

### F3. If GPU work is justified, where it should go

Only after F2 says so. In priority order:

1. **DV permanents on GPU.** The inner loop is a sum over permutations /
   Ryser-style evaluation — regular, parallel, and the actual bottleneck at
   12–20 modes. Highest value if DV matters.
2. **CV beyond ~9 modes.** A dense `cutoff^modes` statevector is exactly the
   shape the existing `statevector` GPU kernels already handle; the work is
   qudit-generalising `apply_1q`/`apply_2q` to `d = cutoff` rather than new
   physics. Reuse before rewrite.
3. **Not the search grid.** Repeated for emphasis: 4 MB does not want a GPU.

**Platform coverage should mirror the existing fleet rather than invent one**:
Metal (Apple Silicon, unified), CUDA (discrete + GB10/GH200), OpenCL
(cross-vendor fallback). Each new GPU backend inherits obligations already
established elsewhere in this repo, and skipping them is how the Metal Reset
defect happened:

* numeric parity against the CPU backend as a CI gate, not a manual check;
* f32 device width priced correctly in the governor (A7b) — a photonic GPU
  backend must set `ExecTarget::Device` or its memory lands against the wrong
  pool;
* `Unsupported` refused loudly for gates the device path lacks, never silently
  skipped;
* **verified on the hardware it claims to support** — `f11a9f5` shipped 2/70
  failing Metal tests because the work was checked on a CUDA box.

## Part H — QASM export drops `if(c==V)` (scoped, not yet fixed)

**Third defect from the 2026-08-08 review cycle.** The other two are fixed
(`11888a9` expectation refusal, `ae6da5c` per-shot sampling); this one is
scoped but deliberately not started, because it needs a signature change and
the scoping is what was missing.

**The defect.** `to_qasm` emits a conditional gate **unconditionally** — the
guard is dropped with no diagnostic. Measured: Qiskit running Aria's own export
of `H q0; measure q0 -> c0; when c0 == 1 { X q1 }` returns
`{'11': 2002, '10': 1998}` where the true distribution is
`{'00': 1995, '11': 2005}`. The correlation is destroyed, and both the export
and the re-import succeed.

**Why it is not a one-line fix.** Aria's condition is a **single bit**
(`Instruction.condition: Option<(Clbit, u64)>`), while QASM 2.0's `if` compares
a **whole classical register**: `if (c == V) gate;`. So a condition on `c[i]` is
exactly expressible only when that register has size 1. On a wider register,
`if (c == 1)` means *the whole register equals 1*, which is a different
predicate — emitting it would trade a silent drop for a silent **change of
meaning**, which is worse.

**Design.** Make export fallible and refuse what QASM 2.0 cannot say:

* creg size 1 → emit `if (c == V) gate;`
* otherwise → **error**, naming the register and bit, and pointing at QASM 3
  (`to_qasm3`, whose `if (c[i] == V)` *can* address a single bit) or at
  restructuring the circuit.

Refusing matches every other decision in this repo — CV gates, unpriceable
jobs, leaking Fock states, feedforward expectations. A lossy export is
especially bad because it is *silent on both sides*: the file is valid QASM and
the importer has no way to know a guard was lost.

**Scope, measured rather than estimated:** `to_qasm` has **8 call sites, of
which 1 is production** (`aria-cli/src/main.rs:760`); the rest are tests. So
the signature change is small.

**Acceptance:** the round-trip test that currently cannot exist — export a
feedforward circuit, run the exported text through Qiskit, and require the
distribution to match Aria's own execution. That is the corpus item flagged in
`OPTIONAL_TESTS.md` as still missing, and it is the only check that would have
caught this: comparing *both engines on the same exported text* agrees
precisely because the export dropped the same thing on both sides.

## Part G — Embedding as a native library (`.dylib` / `.so` / `.dll`)

**PLAN. Reframes two request documents** (`PLAN-PYTHON-LIBS`,
`PLAN-PYTHON-FREE-THREADING`) around what is actually wanted: a library a
downstream can **embed as a native shared object with reasonable ease** — not a
Python package, and not a new binding layer.

### G0. Most of this already exists — verified 2026-08-07

`crates/omega-ffi` is already `crate-type = ["cdylib", "rlib"]` and builds today
to a **2.8 MB `.dylib`**, exporting 15 `extern "C"` functions:

```
omega_runtime_new/free · omega_circuit_from_source · omega_circuit_num_qubits
omega_circuit_num_params · omega_circuit_is_photonic · omega_execute
omega_result_num_counts · omega_result_get_counts[_n]
omega_result_statevector_len · omega_result_get_statevector[_n]
omega_result_free · omega_api_version
```

**The dependency situation is already what the request asks for.** Transitively,
38 external crates, all load-bearing infrastructure rather than hobby projects:

| purpose | crates |
|---|---|
| maths | `ndarray`, `num-complex`, `num-traits`, `matrixmultiply`, `rawpointer` |
| parallelism | `rayon`, `crossbeam-*`, `either` |
| parsing | `pest` family, `ucd-trie` |
| wire format | `serde`, `serde_json`, `itoa` |
| plumbing | `libc`, `libloading`, `cfg-if`, `smallvec`, `thiserror`, `rand`, `getrandom`, `memchr` |

**No `pyo3` in that tree.** The pyo3 path lives in `bindings/aria-py`, a
separate crate, so an embedder taking `omega-ffi` never pulls a Python
interpreter or its dependency weight. The C-ABI round trip is also already
proven end to end by the plugin ABI (`omega-backend-refplugin`, CI stage 6b),
including a version handshake (`omega_backend_abi_version`).

**So the question is not "build an embeddable library" — it is "does the surface
we already ship cover what an embedder needs".** That is a much smaller,
much cheaper question, and it should be answered before anything is built.

### G1. Free-threading is probably moot under this framing

`PLAN-PYTHON-FREE-THREADING` targets the GIL. **But the GIL only constrains work
that runs *inside* a Python interpreter.** If the deliverable is a `.dylib` the
caller loads, then:

* the simulation runs in Rust, on Rust threads (`rayon`), with no interpreter
  involved and no GIL to contend for;
* Python — if present at all — sits at the edge, calling in and getting numbers
  back, which is exactly the standing rule from Part E0.

A free-threaded CPython build (3.13+) is an **experimental** interpreter
configuration, with an ecosystem where many C extensions do not yet work. Taking
a dependency on it would violate the stated constraint ("no experimental APIs
unless critical") to solve a problem the dylib framing removes.

**Recommendation: defer free-threading, and state why.** Revisit only if a
concrete workload is shown to need Python *in* the loop — at which point the
right response is probably to move that work into Rust rather than to adopt an
experimental interpreter.

### G2. What actually needs doing

Small, and mostly about contract rather than capability:

1. **Audit the exported surface against a real embedder's needs.** The current
   15 functions cover: build a circuit from source, inspect it, execute, read
   counts or statevector, free. Likely gaps to check — **expectation values**
   (the whole remote/QML path returns scalars, and there is no
   `omega_expectation`), **gradients** (now that the route exists server-side),
   **parameter binding** (`num_params` is exposed but no setter is visible), and
   **noise configuration**. Answer this by asking an embedder, not by guessing.
2. **Version the ABI explicitly.** `omega_api_version` exists; give it the same
   documented compatibility contract the plugin ABI has, so an embedder can
   refuse a mismatched library rather than crash inside it.
3. **Ship a C header.** Hand-written or `cbindgen`; hand-written is fine at 15
   functions and adds no build dependency.
4. **Document the ownership rules.** Every `_free` pairing, which pointers the
   caller owns, what happens on error. A C ABI with unclear ownership is a
   memory-safety hazard for the *consumer*, and that is the part a Rust
   implementation cannot enforce for them.
5. **Keep the dependency budget as an explicit constraint**, not an accident.
   Record the current 38 and treat additions to the *embed path* as requiring
   justification. This is the property the request is really asking to preserve.

### G3. What to refuse

* **A second binding layer.** `omega-ffi` and `bindings/aria-py` already exist;
  a third surface would multiply the ABI-compatibility burden without adding
  reach.
* **Experimental interpreter builds**, per G1, unless a workload demonstrably
  requires them.
* **Vendoring the world into the embed path.** Any dependency added to
  `omega-ffi` is a dependency every embedder inherits, and they cannot opt out.

**Acceptance:** a third party loads the `.dylib`/`.so`, calls
`omega_api_version`, builds and runs a circuit, and reads results back — with a
header, documented ownership, and no Python anywhere in the process. Most of
that works today; the deliverable is proving it from outside the workspace and
closing whatever the audit in G2.1 finds.

### C3. Photonics from the Aria surface — **CONFIRMED scope** (2026-08-06)

Promoted from "only if separately prioritized" on explicit instruction. Today
`examples/aria/*.aria` cannot express photonics at all: `gate_from_name`
(`aria-core/src/ast/aria.rs:1603-1633`) has no photonic arm and returns
`Err("unknown gate")`, no `.aria` path reaches the photonics backend, and there
is no syntax for an input Fock state. Work items:

1. **Grammar**: photonic arms in `gate_from_name` — `beamsplitter` / `bs_rx`,
   `phaseshifter` / `ps`, and (once B1 lands) `squeeze`, `displace`, `kerr`.
   Follow K15: `displace(a_re, a_im)` is **Cartesian**.
2. **Input Fock state**: a way to declare `|1,1⟩`. HOM is meaningless without
   it, and the DV default for 2 modes is `|1,0⟩`
   (`omega-backend-photonics/src/sim.rs:88-96`).
3. **Dispatch**: `--backend photonics` in aria-cli, and a photonic lowering path
   in `aria-runtime` — today `lower.rs:176` refuses photonic gates while
   advising "use a photonic backend" the Aria path cannot select.
4. **Examples**: port `hom_dip` / `mzi` from OPTICQASM to `.aria`, keeping the
   OPTICQASM originals as the interchange-format examples.
5. **Ledger (K12)**: `.aria` examples join the verify corpus, so the counts in
   `TESTING.md` §12 (44 parse) and the 49/49 verify run **both move**. They must
   be regenerated from an actual run, never hand-edited.

Sequenced after B0/C1 (done) and the A-series; B1's CV gates are only needed for
item 1's CV half — the DV half (`bs_rx`, `ps`) can land immediately and makes
HOM/MZI expressible in Aria without waiting on the CV backend.

**Order:** A1 → A2 → A3 (unblocks remote QML training); then **A7** before A6 —
a shared host wants the guard rails before it gets an easier way to submit long
jobs. A4 is subsumed by A6/A7; A5 after.

---

## Part B — CV photonic backend (`PLAN-CV-BACKEND.md`)

### B0. The silent-drop trap — first, and narrower than the request states

Confirmed real, and it is the one item worth doing regardless of how the DGX
architecture search turns out:

- `map_gate_kind` (`aria-core/src/backends/omega.rs:146-177`) returns `None`
  for `Squeezing`/`Displacement`/`Kerr` via its `_ => None` arm.
- The lowering loop (`omega.rs:219`) is `if let Some(g) = map_gate_kind(...)`,
  so unmapped gates are **skipped without a word**.
- Sharper still: `is_photonic` (`omega.rs:207-216`) matches on exactly those
  three gates — the circuit is classified photonic *because* of gates the very
  next loop deletes.

**Correction to the request.** `PLAN-CV-BACKEND.md` §2 implies the trap is
general ("a program using `squeeze` type-checks … and then executes with its CV
gates deleted"). It is not. `aria-runtime/src/lower.rs:176` has a catch-all
`other =>` arm returning a clear error, so **local** execution already refuses
CV gates loudly. The silent drop is confined to `to_omega_ir`, whose callers are
the **remote** backend (`remote.rs:76,99`) and the QEC mirror
(`aria-qec/src/ecc/run.rs:136`).

That makes it a *remoting* bug as much as a photonics one — a CV circuit sent to
a server loses its CV gates and returns confident numbers — which is why B0
lands with Part A rather than waiting on the backend.

**Blast radius — larger than three call sites.** `to_omega_ir` returns
`OmegaCircuitIR`, not a `Result`. Beyond `remote.rs:76,99`, the QEC mirror call
at `aria-qec/src/ecc/run.rs:136` sits inside `pub fn to_omega_core_ir`, which
has 6 further non-test callers (`logical/run.rs:42`, `logical/algo.rs:59`,
`ecc/run.rs:162,346,607,639`), several inside public QEC entry points, plus ~10
test call sites in `omega.rs:428-556`.

Chosen strategy: add a fallible `try_to_omega_ir` and keep `to_omega_ir` as a
thin wrapper, so the QEC surface does not churn. `to_omega_core_ir` uses the
fallible form and `expect()`s internally, justified in a comment: QEC-generated
circuits are gate-model by construction and cannot contain CV gates. That
justification gets written down rather than assumed. `to_omega_ir` is not in the
Q2 pinned-signature list, so this is contract-legal; K11 ("CV gates fail loudly")
makes it a keep-contract fix, not a nicety.

While in there: OPTICQASM's own parser has the same defect — `opticqasm.rs:169`
`_ => continue` silently skips unknown gates. One line, same class, fix together.

Gate: a `squeeze` program returns a loud `Unsupported` through the remote path
and the QEC mirror; a regression test pins that it is never silently dropped.

### B1–B6. The backend itself

Envelope from the search grid: `n_modes ≤ 6`, `cutoff ≤ 8` (~2.6e5 amplitudes) —
laptop-sized, dense is fine.

| step | scope | acceptance |
|---|---|---|
| B1 | New crate `omega-backend-cv`: truncated-Fock qudit statevector, `dim = cutoff^n_modes`. Gates: `Squeezing`, `Displacement`, `Kerr` + existing `PhaseShifter`/`BeamSplitter`. CV variants in the omega `GateKind` + `map_gate_kind` entries. | analytic anchors: displaced vacuum `⟨n⟩ = |α|²`, squeezed vacuum `⟨n⟩ = sinh²r`, Kerr (diagonal, exact) |
| B2 | `⟨n_i⟩` readout for all modes in one call | matches piquasso at equal cutoff, ≤ 1e-6, on 3 grid shapes |
| B3 | Truncation policy (R6): renormalize **and report** lost norm, or refuse past a threshold | a leaking state is refused or reported, pinned by test — never a silent expectation from a leaking state |
| B4 | Gradients: product rule through the component chain, `dU/dθ` via Fréchet derivative of `expm` — exact on the truncated space | finite-difference cross-check ≤ 1e-6 |
| B5 | Batching inside the backend + pyo3 exposure | one epoch, no Python in the loop |
| B6 | Conformance shapes for `omega-plugin-conformance` | 3 analytic anchors + 1 piquasso transcript |

Honest constraints carried from R3, not designed around: truncated squeezing and
displacement are **not unitary** (truncation clips the ladder), so the qubit
adjoint trick does not port; and CV gates have no two-point parameter-shift rule.
Hence the Fréchet route.

Oracles: piquasso 8.0.1 (`_math/gate_matrices.py`, `_math/gradients.py`,
`_simulators/fock/pure/`), with The Walrus `fock_gradients` as the independent
derivation for the recurrences. Piquasso parity is a **test-time** dependency —
it must not enter the build, and per K13 the parity stage must be **optional
with a clean skip** when no venv is present (the `ARIA_QISKIT_XCHECK` pattern),
or CI green becomes network- and venv-dependent. The analytic anchors carry no
external dependency and stay in the default run.

Where it lives: new crate `omega-backend-cv` rather than a mode inside
`omega-backend-photonics`. That crate is DV-only (`FockKet` + permanents/SLOS);
mixing a dense CV qudit statevector into it would blur two different state
representations behind one name. (Maintainers' call — flagging the choice.)

**Timing.** B1–B6 are gated on the DGX search answering PR-AUC in CV's favour
(0.9751 AUC but 0.466 PR-AUC today). **B0 is not gated on anything.**

---

## Part C — Examples (DV and CV)

There are currently **no photonic examples at all** — `examples/aria/` has 44
programs, none using `beamsplitter`, `phaseshifter`, `squeeze`, `displace` or
`kerr`, even though the DV backend ships.

**Scope correction — this is not "just add example files."** The Aria surface
grammar does **not** parse any photonic gate: `gate_from_name`
(`aria-core/src/ast/aria.rs:1603-1633`) has no photonic arm and falls through to
`other => Err("unknown gate '{other}'")`. Only **OPTICQASM** parses them
(`opticqasm.rs:163-170`: `ps`, `bs_rx`/`bs`, `squeeze`, `displace`, `kerr`).
Nor can any `.aria` path reach the photonics backend: `aria-cli` offers
`sim|mps|gpu|tch|pauliprop|remote`, and `aria-runtime/src/lower.rs:176` refuses
photonic gates while advising "use a photonic backend" that the Aria path cannot
select. There is also no Aria syntax for an input Fock state, which HOM needs
(|1,1⟩; the DV default for 2 modes is |1,0⟩, `sim.rs:88-96`).

So Part C splits:

- **C1 — DV examples that work today, as OPTICQASM driven by `omega-cli`.**
  `examples/opticqasm/hom_dip.oqasm`: Hong–Ou–Mandel, two photons through a
  balanced beamsplitter, coincidence → 0 — exact, self-verifying, and runnable
  now via `omega-run --backend photonics --input 1,1`. Plus `mzi.oqasm`:
  Mach–Zehnder phase sweep with `⟨n⟩` tracing `cos²(φ/2)`. K14 pins OPTICQASM
  import by extension, so this needs no language change.
- **C2 — CV examples** (`cv_squeezed.oqasm`, `cv_displaced.oqasm`): the analytic
  anchors `⟨n⟩ = sinh²r` and `⟨n⟩ = |α|²`, doubling as B1/B2 acceptance. Until
  B1 lands they must **fail loudly** (B0) — worth pinning on its own.
  **Convention:** K15 pins `displace(a_re, a_im)` as **Cartesian**
  (`opticqasm.rs:81-99`). `PLAN-CV-BACKEND.md:13` mis-states it as polar
  `displace(r, phi)`; B1's gate construction and this example must follow K15 or
  they break a wire format downstream pins.
- **C3 — optional, separately scoped: photonics from the Aria surface.** Adding
  photonic arms to `gate_from_name`, an input-Fock-state syntax, and an
  aria-cli photonics dispatch. Only this makes `.aria` photonic examples
  possible. Not required by any request in `fixes/` — do not smuggle it in.

Ledger discipline (K12): the `aria-verify` corpus enumerates `examples/aria/*.aria`
(`harness.rs:55-57`), so OPTICQASM examples do **not** change the 49/49 verify
count or the ledger, and do not change `TESTING.md` §12's parse count of 44.
C3, if ever done, moves both — and those numbers must be regenerated, not edited.

---

## Part D — Older bug requests

- **D1 `noise` on non-statevector backends — verify and close, do not rebuild.**
  MPS carries trajectory-noise Monte-Carlo (`omega-backend-mps/src/sim.rs:295-335`)
  with analytic parity tests (:1206+); omega-cli refuses `--noise` loudly on
  backends that cannot represent it (`main.rs:937-944`, `:192-197`, `:543-548`);
  pauliprop applies noise on `--expectation` (`:827-829`). K14 lists `--noise` on
  statevector/MPS as a keep. The only genuinely open remainder is stabilizer
  **sampling** noise, which is refused loudly today — a false rejection, not a
  wrong answer. Confirm against the bug doc's reproduction and close it out.
- **D2 IBM calibration-noise import.** Not blocked by D1 — the noise model is
  already trustworthy on statevector/MPS. Can start whenever it is prioritized.
- **D3 QASM3 interchange.** Audit the existing implementation against the
  request's checklist; scope only what is genuinely missing.
- **D4 MPS truncation + SVD stability.** Both look already fixed. Confirm
  against each request's reproduction, then close them out explicitly rather
  than leaving stale bug docs implying open defects.

---

## Sequencing

1. ~~**B0**~~ — **DONE** (2026-08-05): silent CV drop closed, `./ci.sh` green
2. ~~**C1**~~ — **DONE**: DV OPTICQASM examples, HOM + MZI verified against output
3. ~~**A7**~~ — **DONE** (2026-08-06): resource governor, after an adversarial
   review found the first cut under-priced the default path. Gaps named above.
4. ~~**A7b**~~ — **DONE** (2026-08-06): per-pool memory topology. `topology.rs`
   splits impure probing from a pure `classify`, so DGX Spark (GB10), Apple
   Silicon, discrete CUDA, 8-GPU DGX and CPU-only are all unit-tested on any
   box. Safe default: uncertain ⇒ unified.
5. **A7c** — operator throttles (fractional CPU / RAM / VRAM caps). Follows A7b
   because a VRAM fraction is meaningless until pools exist, and because on
   unified hardware the two memory fractions must not compound.
6. **A11 → A1 → A2 → A3** — timing breakdown first (it is the instrument that
   makes A0/A2's payload claims measurable rather than asserted), then remote
   gradient, batching and pyo3, which together unblock remote training
7. **C3** — photonics from the Aria surface (**confirmed**). DV half needs
   nothing from B1 and can land as soon as A7b clears.
8. **A10 models → A9 → A6 + A8 + A8b** — model the scheduling and batch
   lifecycle *before* building them (the starvation counterexample should shape
   the queue, not post-rationalise it), then durable batches first (state that outlives the
   connection and the process), then the WS push and client resilience as
   layers over it. This order matters: build the socket first and "reconnect
   later" is unimplementable, because there is no durable state to reconnect
   *to*. A9 also settles the per-row partial-failure semantics A1 needs.
9. **D4, D1** — verify-and-close the three already-fixed bug docs
10. **D3** — QASM3 audit against the request's checklist
11. **A5** — expectation-path device routing + honest `backend_name`
12. **A12** — optional cluster manager (same protocol, per-node resources,
    work stealing). After A9/A10, since rows are the stealable unit and the
    lease protocol is worth model-checking first.
13. **B1–B6 + C2** — CV backend and its examples, gated on the DGX search

Constraints throughout: `./ci.sh` is the single source of truth and must stay
green (K13), with external oracles optional and cleanly skipped; no stdout from
libraries (Q8); tolerances stated numerically (Q9); batch index preservation
(Q5); assumption-ledger discipline (K12). Note the ledger counts do **not** move
for C1/C2 — the verify corpus enumerates `examples/aria/*.aria` only.

## Part I — Polarization in OPTICQASM (PBS / HWP), new 2026-08-08

Requested so the Quandela-style Grover can be expressed: add `pbs` (and the
`hwp` it needs) to OPTICQASM and cascade the change through parser, lowering,
backend, CLI, governor and the Perceval bridge.

### I0. Two measurements this plan rests on

Taken 2026-08-08, not assumed:

* **`HWP_perceval(θ) = i · BSrx(2θ, 0) · PS(π on the V sub-mode)`**, agreeing to
  **1.5e-16**. A half-wave plate needs **no new simulation primitive**: it is
  ops we already have, times a global phase.

  > **This line was WRONG in the first draft and the correction is the whole
  > point.** It originally read `HWP(θ) = BSrx(2θ,0)·PS(π)`, "agreeing to
  > 1e-16" — measured against the **i-less textbook** matrix while this same
  > section argued that Perceval's `i` is a correctness issue. Both cannot hold:
  > `det(BSrx) = +1` and `det(PS(π)) = −1`, so any such product has `det = −1`,
  > while Perceval's HWP has `det = +1`. Measured: the i-less formula differs
  > from Perceval by **1.0**, and placing that HWP mid-circuit in a 4-mode MZI
  > moves single-photon output probabilities by **0.413**.
  >
  > Two further details the first draft omitted, each of which silently breaks
  > the identity: `PS(π)` must target the **V** sub-mode (on H the error is
  > 1.65), and it must come **after** the beam splitter. The "1.41" quoted for
  > the reversed order is θ-dependent, not a constant.

* **omega's `bs_rx` is Perceval's `BS.Ry`, NOT `BS`.** Measured: `0.000e+00`
  against `BS.Ry(2θ)`, up to `1.0` against `BS(2θ)`. See I0b — this is a live
  defect in the bridge, not a planning note.
* **Perceval 1.2.4's conventions are not the textbook ones**, read directly off
  `compute_unitary()`:
  * `PBS` is 4×4 over **2 spatial modes**, ordering `[a_H, a_V, b_H, b_V]`,
    mapping `out_aH = in_bH`, `out_bH = in_aH`, `out_aV = in_aV`,
    `out_bV = in_bV`. It **swaps H and transmits V** — the opposite of the
    common "transmits H, reflects V" phrasing.
  * `HWP(θ)` returns `i · [[cos2θ, sin2θ], [sin2θ, −cos2θ]]`.

The `i` matters. It is unobservable on a lone element, but a polarization
element acts on a **subset** of the modes of a larger interferometer, where a
global factor on that block becomes a **relative** phase between paths. A Grover
mesh is precisely a device built out of such relative phases, so this is a
correctness issue, not bookkeeping. This project has already shipped one defect
of exactly this shape (`pauli_mult_phase`, in two copies).

* **PBS's swap block IS expressible**: `PS(π) · BSrx(π/2, ·)` gives the exact
  swap to 1e-16. `det = −1` works out because `PS(π)` supplies it. The first
  draft asserted "PBS is a permutation" with **no measurement at all** while
  claiming I0 "shows both new elements decompose" — it showed one, against the
  wrong reference.

### I0b. PREREQUISITE — a live bridge defect that must be fixed first

`perceval_runner.py:279-282` maps omega `bs_rx(θ,φ)` to Perceval `BS(theta=2θ)`,
the **Rx** convention. omega's matrix is Perceval's **Ry**. Measured error at
`θ=0.6`: **0.798**; at 50/50: **1.0**.

It has survived because the only photonic cross-backend fixture is the HOM dip,
which is **phase-insensitive** — an amplitude-only test standing over a phase
error. For plain BS meshes there is also a mode-local gauge (a phase `i` per
coupled pair) that makes Fock distributions agree anyway, which is why nothing
downstream complained either.

**Mode doubling destroys that gauge**: a spatial BS becomes a coupler on
distance-2 optical pairs while PBS demands an exact, ungauged swap on those same
pairs. Measured on `spatialBS → PBS → spatialBS` with one photon, the two
conventions give `[0,0,1,0]` versus `[1,0,0,0]` — **a distribution difference of
1.0**.

Consequence for sequencing: if this is not fixed first, I4's end-to-end Kwiat
check can fail *even when the new code is correct*, and the natural response
would be to "fix" correct code to match a wrong baseline. Fix is one token
(`BS` → `BS.Ry`) plus a **matrix-level** fixture — an amplitude-only fixture is
what let this through in the first place.

### I1. Mode model — the decision that shapes everything else

| option | what it is | cost |
|---|---|---|
| **A — mode doubling (recommended)** | polarization is an indexing convention: `m` spatial modes become `2m` optical modes, `(s, p) -> 2s + p`, `p=0` is H. Polarization elements lower to existing `PhaseShifter` / `BeamSplitterRx`. | parser + lowering only; **zero change to slos / permanent / sim** |
| B — first-class polarization DOF | new `PhotonicOp` variants, new indexing through SLOS and the permanent | rewrites the simulation core to express the *same* matrix |

**Take A.** Linear optics on `m` spatial modes × 2 polarizations *is* a `2m`-mode
linear unitary; B rewrites the core to produce a matrix A already produces. I0
shows both new elements decompose into existing ops, so A is not a shortcut —
it is the accurate model.

**Corrections to the first draft's cost analysis**, from review:

* **"zero change to slos / permanent / sim" was two-thirds right.** `slos.rs`
  and `permanent.rs` are genuinely layout-agnostic (`Vec<u32>` occupations,
  arbitrary index submatrices), and — contrary to the doc comment at
  `components.rs:36` claiming "two **adjacent** modes" —
  `apply_beam_splitter_rx` operates on arbitrary row pairs, so a PBS on optical
  modes `(0,2)` is fine. **That stale doc comment should be fixed**; taken at
  face value it says option A is impossible.
* **`sim.rs` is NOT zero-change, and hides a live bug.** `encode_fock`
  (`sim.rs:322-331`) nibble-packs Fock states into a `u64` and **silently
  `break`s past 16 modes** — beyond that, distinct states collide onto the same
  counts key with no error at all. Doubling halves the usable width to **8
  spatial modes** in shots mode. That silent collision is a defect today,
  independent of polarization.
* The first draft's own I1 table said "zero change to sim" while I3.4 demanded a
  `decode_fock_string` change, which lives in `sim.rs:347`. Contradictory.

**On the governor — the first draft aimed this at the wrong file.** Pricing is
`fock_dim(modes, photons)` = `C(m+p−1,p)` (`worker.rs:230-242`) with
`modes = shape.num_qubits` taken straight from the IR, and the photonic path is
**exempt from the qubit ceiling** (`worker.rs:215`), so pricing is the only
guard and the exhaustion risk is real. **But no `worker.rs` change is needed**:
if `photon q[N] pol;` lowers to an IR with `num_qubits == 2N`, the governor is
automatically correct. The real invariant to protect is therefore *double at
lowering*, pinned by a test asserting a `pol` declaration prices as `2N` — not,
as first written, "compute on 2m in the governor".

### I1b. Convention policy — MATCH THE REFERENCE IMPLEMENTATIONS (decided)

**Aria adopts Perceval's conventions for DV and piquasso's for CV, verbatim,
including the parts that differ from textbook statements.** Where they differ
from ours, ours moves.

Rationale: a third convention would mean every cross-check carries a fudge
factor, and a fudge factor is indistinguishable from a bug once someone forgets
why it is there. The bridge already has one such factor
(`theta_perceval = 2·theta_omega`) and it is precisely where the `BS`/`BS.Ry`
defect (I0b) hid. Matching outright means the cross-check compares raw matrices
with nothing in between, so a disagreement is a defect by construction.

The consequences, which must be **documented at the point of use**, not only
here — a convention recorded solely in a design doc is a convention nobody
applies:

| element | adopted (Perceval / piquasso) | differs from textbook how |
|---|---|---|
| `hwp(θ)` | `i · [[cos2θ, sin2θ],[sin2θ, −cos2θ]]` | carries a global `i`; `det = +1` not `−1` |
| `pbs` | swaps **H** between spatial modes, transmits **V** | usually stated the other way round |
| `bs_rx(θ,φ)` | Perceval `BS.Ry(2θ)` | the name says Rx; the matrix is Ry. **Rename or document loudly** — see I0b |
| CV `displace` | Aria stays **Cartesian** `(re, im)` per K15 | piquasso is polar `(r, φ)`; converted in exactly ONE place (`piquasso_ref.py`) |

The `bs_rx` row is the uncomfortable one: the gate is *named* for a convention
it does not implement. Renaming it is a breaking change to existing `.opticqasm`
files; leaving it means the name lies. Recommend keeping the name (files exist)
and stating the matrix explicitly in the grammar docs and every example header,
since the matrix is what the examples already document.

### I2. Grammar (`opticqasm.pest`)

* `photon_decl` gains an optional marker: `photon q[2] pol;` — declaring `2N`
  optical modes. Without it, behaviour is unchanged, so every existing file
  keeps working.
* New gate names `pbs`, `hwp` (and `qwp` if the wave-plate pair is worth
  completing). **Note `gate_name` already falls through to `ident`**, so these
  parse *today* and fail only at lowering — check that failure is loud and names
  the gate, per K11. A silently-ignored polarization element is the worst
  outcome available here.
* Polarization elements take **spatial** mode refs (`pbs q[0], q[1];`), matching
  Perceval, where `HWP().m == 1` and `PBS().m == 2` are spatial. Lowering
  expands to sub-modes; the surface stays readable.

### I3. The cascade

1. `crates/omega-parser/src/opticqasm.pest` — grammar
2. `crates/omega-parser/src/{opticqasm.rs, ast.rs, lower.rs}` — parse -> AST -> `PhotonicOp`
3. `crates/aria-core/src/ast/opticqasm.rs` — mirror
4. `crates/omega-backend-photonics` — no new sim code under option A, **but**
   `decode_fock_string` must label polarization (`|1H,0V,...>`) or every output
   becomes unreadable at doubled mode count
5. `crates/omega-cli` — `--input` takes `2N` occupations; the ordering must be
   documented at the point of use, not only in a design doc
6. **Governor pricing** — `2m` per I1. Highest-risk item.
7. `crates/omega-bridges/python/perceval_runner.py` — `_build_opticqasm_circuit`
   gains `pbs` / `hwp` with the I0 conventions pinned **in a test**, alongside
   the existing `theta_perceval = 2 * theta_omega` mapping
8. `crates/omega-bridges/tests/cross_backend.rs` — polarization fixtures

### I3b. Polarized INPUT — unsolved in the first draft, and it blocks the example

The first draft discussed gates at length and said nothing adequate about how a
polarized input is specified. It is not a documentation gap; the bridge contract
does not currently admit one:

* `perceval_runner.py:161` requires `input_fock` to be a flat int list with
  `len == num_modes`, and line 186 reads outputs as `sample[i] for i in
  range(num_modes)`. Perceval's PBS/HWP are **polarized components on spatial
  modes** (`HWP().m == 1`), so Perceval would see `N` modes where omega has
  `2N`: the length check rejects the input and polarized output `BasicState`s
  break the flat key loop.
* Two possible designs, and the plan must pick one rather than discover it
  mid-implementation:
  1. **Runner expands `pbs`/`hwp` itself** into plain `2N`-mode unitaries, so
     Perceval never sees a polarized component and the flat contract survives.
     Keeps the protocol unchanged. **Recommended.**
  2. Runner builds a genuinely polarized `N`-mode processor, and the protocol
     grows a polarization-aware input/output encoding.
* Design 1 also sidesteps a limit of design 2: `--input` occupation numbers
  (`omega-cli/src/main.rs:68`) can express **H/V basis** inputs, which is all
  the Kwiat example needs, but not diagonal or circular polarization.
* Separately: the **server** path builds `PhotonicsBackend::new()` with no input
  plumbing at all (`quantum_bridge.rs:438-440`), so polarized examples are
  **CLI-only** until that is addressed. Worth stating before someone tries one
  over the remote API.

### I4. Validation, and what would make it worthless

* Pin the conventions in tests that read the **matrix**, not in comments that
  drift. **The first draft's pin was one-sided** and would not have caught its
  own defect: pinning *Perceval's* HWP `i` passes happily against an omega
  implementation built from the (wrong) i-less formula. The pin that bites is:
  read **omega's lowered `2N`-mode unitary** for a lone `hwp` and assert
  equality with Perceval's polarized matrix, `i` included.
* End-to-end: Kwiat 4-element Grover, aria vs Perceval on the same source.
* **Mutation-test the check before trusting it**: swapping the H/V convention
  must fail it. If a swapped convention still passes, the fixture is agreeing
  for the wrong reason — the failure mode that has recurred throughout this
  project (vacuous TLA+ liveness, the phase-blind CV comparison, a Qiskit corpus
  that could not reach the defects it was meant to catch).
* A Grover mesh that returns the marked item is **not** evidence on its own:
  a wrong global phase on an HWP block can leave the marked-item probability
  intact while corrupting the others. Compare the **full** output distribution.
* **Sequencing is not optional.** Fix I0b first. Until then the end-to-end
  distribution check fails both when the implementation is wrong *and* when it
  is right, and a check with that property verifies nothing.

### I4b. Live defects found while scoping this — all independent of polarization

Recorded so they are fixed as defects rather than absorbed into a feature:

1. **`perceval_runner.py:281` uses the wrong Perceval component** (`BS` where
   omega means `BS.Ry`). Measured error up to 1.0. See I0b.
2. **`sim.rs:322` `encode_fock` silently truncates past 16 modes** — distinct
   Fock states collide onto one counts key with no error raised.
3. **`decompose.rs:126` `clements_decompose` delegates to Reck** while the
   module header advertises Clements' "better loss tolerance". The function's
   own doc says "For now, delegates to Reck", so it is a labelled delegation
   under a misleading header rather than a covert stub — but a QFT-as-multiport
   example built on it would inherit a false claim.
4. **Grammar/lowering asymmetry**: the grammar advertises `bs` and `bs_ry`
   (`opticqasm.pest:17`) that lowering rejects, while the Perceval runner
   supports a `bs_ry` omega can never emit. `bs_ry` therefore has **zero**
   cross-backend coverage.
5. **`cross_backend.rs:602` references `hom_effect.opticqasm`**; the file is
   `hom_dip.opticqasm`.

### I4c. What the review confirmed as SOUND

Not everything was wrong, and knowing which parts held matters for how much of
the plan to rework:

* Mode doubling needs **no** adjacency work — `apply_beam_splitter_rx` takes
  arbitrary mode pairs; `slos.rs` and `permanent.rs` are layout-agnostic.
* Unknown gates **do** fail loudly and by name (`"unknown photonic gate: pbs"`,
  `lower.rs:470`). Nothing is silently dropped, so `pbs`/`hwp` parsing today and
  failing at lowering is safe.
* Perceval's PBS permutation and HWP `i` readings (I0 b and c) are correct.
* The governor exhaustion risk is real; only the *location* of the fix was
  misidentified.

### I5. What this unlocks, with exact references

* **Grover, 4 elements, one query** — Kwiat, Mitchell, Schwindt & White,
  *"Grover's search algorithm: an optical approach"*, **J. Mod. Opt. 47(2–3),
  257–266 (2000)**. Quandela ships this as a Perceval notebook
  (`2-mode_Grover_algorithm`), so the reference implementation already exists
  and is runnable — this is the cheapest independent check available.
* Mesh-based examples (QFT as a multiport) need **no** polarization and are
  tracked separately: Reck, Zeilinger, Bernstein & Bertani, **PRL 73, 58
  (1994)**; Clements, Humphreys, Metcalf, Kolthammer & Walmsley, **Optica 3(12),
  1460–1465 (2016)**, arXiv:1603.08788.

### I6. Related defect found while scoping

`omega-backend-photonics/src/decompose.rs:126` — `clements_decompose` is
documented as "the Clements (symmetric) scheme" and its module header claims
"better loss tolerance", but the body is `reck_decompose(unitary)`. It is a stub
advertising a property it does not have. Either implement Clements or rename it;
a QFT-as-multiport example would otherwise be built on a claim that is false.

## Part J — Unblocking the two missing example families, new 2026-08-08

`docs/PHOTONICS_EXAMPLES.md` ships without a QFT example and without a CV QAOA
example. Neither is missing because nobody wrote it; each is blocked by a
specific defect, and this Part is how those get fixed.

A third item (J3) was found *while* planning this one, and it is a defect in
work already committed.

### J1. QFT as a multiport — blocked by a mislabelled decomposition

**Blocker.** `decompose.rs:126` — `clements_decompose` is documented as "the
Clements (symmetric) scheme" under a module header advertising "same count but
better loss tolerance", and its body is `reck_decompose(unitary)`. A
QFT-as-multiport example built on it would inherit a property the code does not
have. `reck_decompose` itself (338-line module) is real and tested.

**Options:**

| | action | consequence |
|---|---|---|
| **A** | implement Clements properly | ~120 lines; delivers the advertised half-depth |
| **B** | delete `clements_decompose`, keep Reck | honest immediately, loses nothing that works today |

**Recommend B first, then A as its own change.** The QFT example needs *a*
correct decomposition, not specifically Clements — Reck already provides that.
Shipping B unblocks the example this week and removes a false claim; A is a real
algorithm that deserves its own commit and its own verification rather than
riding along inside an examples task.

**References for A:**
* Reck, Zeilinger, Bernstein & Bertani, **Phys. Rev. Lett. 73, 58 (1994)** —
  triangular, `N(N−1)/2` splitters.
* Clements, Humphreys, Metcalf, Kolthammer & Walmsley, **Optica 3(12),
  1460–1465 (2016)**, arXiv:1603.08788 — symmetric mesh, **half the optical
  depth**, more loss-tolerant.

**Verification that would actually bite.** Not "it produces some ops":
1. Recompose — `build_unitary(m, decompose(U))` must equal `U` to ~1e-12, over
   random unitaries at several `m`, not one hand-picked matrix.
2. For the DFT, the **Fourier suppression law** — Tichy, Tiersch, de Melo,
   Mintert & Buchleitner, **Phys. Rev. Lett. 104, 220405 (2010)**: for a cyclic
   multi-photon input, every output with `Σⱼ j·nⱼ ≢ 0 (mod m)` is **strictly
   forbidden**. Exact zeros, and it exercises the **permanent**, which
   recomposition does not touch at all.

   > The first draft proposed "single-photon output must be uniform `1/m`" and
   > called it "a sharp, convention-free check". It is neither sharp nor
   > additive. Uniform single-photon output only constrains `|U_jk| = 1/√m`, so
   > **every complex Hadamard passes**: measured at m=4, the real Hadamard
   > `H⊗H/2` (‖U−DFT‖ = 1.0), the continuous family `F₄(a=0.7)` (0.34), and a
   > row-permuted DFT (1.0) all pass. It is also strictly **subsumed** by check
   > 1, which already fixes every modulus. Zero discriminating power added —
   > the exact shape of check this project keeps mistaking for coverage.

3. If A lands: assert Clements' depth is genuinely **lower** than Reck's for the
   same `U`. Without that assertion, a second delegation could reappear and
   nothing would notice — which is exactly how the current stub survived.

**Two defects in `reck_decompose` found while reviewing this** (it is otherwise
correct — verified externally against Haar unitaries m=2–8, DFT m=2–8 and
permutation matrices, all recomposing to ≤1.2e-15):

* `decompose.rs:15-17` early-returns empty for `m ≤ 1`, so a 1×1 unitary
  `[e^{iφ}]` silently recomposes to the identity — the phase is dropped.
* **The in-repo Reck tests were called circular — measured, and that is
  overstated.** `random_unitary` (`decompose.rs:137`) does build its inputs from
  `build_unitary` over the same primitives used to reconstruct. But two
  convention mutations were applied and **both suites caught both**:
  a phase-shifter sign flip (5 in-module failures) and a beam-splitter
  `e^{iφ}`/`e^{-iφ}` swap (5 in-module failures). The circularity is not total
  because `reck_decompose` derives its angles **analytically** from the target
  matrix rather than via `build_unitary`, so a convention change does not cleanly
  cancel through the decomposition step.
  Independent test inputs (closed-form DFT, Gram–Schmidt, permutations) are still
  worth adding — they are defence in depth and the only place the decomposition
  meets matrices it did not construct — but they fix no existing hole.

**Then the example**: `examples/circuits/qft4.opticqasm`, 4 modes, generated by
the decomposition and cross-checked against Perceval like every other DV
example. Note the emitted `bs_rx`/`ps` angles are concrete, so the file is
readable and diffable rather than an opaque matrix dump.

### J2. CV QAOA — blocked by preparations being constructors, not operators

**Blocker.** `omega-backend-cv` has `coherent(α)` and `squeezed_vacuum(r)` as
**constructors from vacuum**. There is no `displace(&mut self, α)` and no
`squeeze(&mut self, r)`. CV QAOA's mixer *is* a displacement applied to an
evolving state (Verdon et al., **arXiv:1902.00409**; photonic implementation
Enomoto et al., **Phys. Rev. Research 5, 043005 (2023)**, arXiv:2206.07214), so
the algorithm cannot be expressed at all.

**Do NOT implement these as a truncated matrix exponential.** Measured while
planning, against piquasso 8.0.1 at cutoff 20, aligning global phase:

| r | Aria's closed form vs piquasso | truncated `expm` vs piquasso |
|---|---|---|
| 0.3 | 1.1e-16 | 4.8e-07 |
| 0.5 | 2.2e-16 | 7.8e-05 |
| 0.8 | 2.8e-17 | 4.1e-03 |
| 1.0 | 5.6e-17 | **1.8e-02** |

`expm(truncated generator)` is **not** the truncation of the exact operator, and
the error grows fast. Whatever piquasso does, it is not that. So:

* **Operator forms must be built from the exact matrix elements**, i.e. the
  known closed-form `⟨m|D(α)|n⟩` and `⟨m|S(r)|n⟩`, evaluated on the truncated
  index range — not by exponentiating a truncated generator.
* **Keep the existing constructors.** `coherent`/`squeezed_vacuum` agree with
  piquasso to 1e-16 and know their own analytic tail, which is what feeds
  `lost_n_weight`. The operator forms are an addition, not a replacement.
* **The operator matrix is not unitary on the truncated space** — it is a
  compression of a unitary, so its columns have norm < 1 and the state's norm
  shrinks. That is absorbed by the existing renormalise-by-represented-mass
  plumbing (`probabilities()`, `expect_n`), so it is not a defect, but the code
  must say so rather than leave a reader wondering.

* **Truncation accounting — the error is `√ε`, not `ε`.** The first draft said
  "if a sound bound cannot be derived, refuse above a conservative threshold",
  which is hand-waving over a measured factor. For an operator applied to a
  state that has already lost mass `ε`, Cauchy–Schwarz with `‖D‖ ≤ 1` gives an
  amplitude-error bound of `√ε`. Measured (piquasso at cutoff 20 vs cutoff 160
  as truth):

  | circuit | intermediate lost mass ε | √ε | actual amplitude error |
  |---|---|---|---|
  | S(0.5) then D(0.63) | 3.9e-8 | 2.0e-4 | **6.5e-5** |
  | S(0.8) then D(1.0) | 6.3e-5 | 8.0e-3 | 1.7e-3 |
  | S(1.0) then D(1.0) | 1.1e-3 | 3.3e-2 | 6.6e-3 |

  At r=0.5 the true error is **~1700× the linear lost-mass budget**. So a
  "conservative threshold" is only sound at **tolerance²**: with the existing
  `1e-6` leak tolerance, a state that just passes the current gate can carry
  per-amplitude errors near 1e-3.

* **A sound bound largely CAN be derived**, so refusing is under-engineered
  relative to machinery the crate already has. For a tail-free input, the kept
  amplitudes of a truncated-**exact** operator are exact, so the new lost mass
  is exactly `‖in‖² − ‖out‖²`; and `lost_n_weight` can be bounded by applying
  the operator on a **padded internal range** — the same `+512` extension trick
  `coherent` and `squeezed_vacuum` already use (`lib.rs:293`, `lib.rs:337`).

**Verification — and what it CANNOT see.** Extend `piquasso_ref.py` with the
compositions this unlocks. But note the trap: **piquasso commits the identical
sequential-truncation error**, so a composition cross-check will pass tightly
whether or not either side is near the truth. It validates *convention match*,
not correctness. The first draft's "only after that is a CV QAOA example honest"
was therefore over-claimed.

Honesty additionally requires **analytic anchors**, which the first draft
omitted — e.g. displaced-squeezed `⟨n⟩ = |α|² + sinh²r`, checked at a cutoff
generous enough that the leak is negligible, so at least one point of the test
is tied to truth rather than to agreement.

**Sequencing matters, and the first draft got it backwards.** Only **one** case
is currently `NOT COMPARED` (`squeezed_r=0.5_phase_then_displace`,
`piquasso_xcheck.rs:427`); "displace then squeeze" is not in the corpus at all,
so "already emitted as NOT COMPARED" (plural) was wrong. And
`preparations_are_still_constructors_only` asserts an **exact set**: extending
the fixture first makes it fail because the set **grew**, whose canned message
says "a preparation regressed" — a false diagnosis. Land the operator forms
**first**, then extend the fixture.

### J3. A defect in already-committed work, found while planning J2

`crates/omega-backend-cv/tests/piquasso_xcheck.rs` and `OPTIONAL_TESTS.md` state
that Aria and piquasso "genuinely differ" on squeezed vacuum because "we
evaluate the closed-form Fock amplitudes and then cut; piquasso applies a
squeezing operator that is itself already truncated", calling them "two
defensible readings".

**Both halves are wrong.** Measured:

* The amplitudes agree to **1e-16** (table above), so there is no difference of
  state to defend.
* piquasso does not use `expm` of a truncated generator — that disagrees by
  1.8e-2 at r=1.0.

> **Careful with the second bullet — the first draft of THIS correction was
> also wrong.** It read "piquasso does not use a truncated operator", which is
> false and would have landed a fresh wrong explanation while fixing an old one.
> piquasso **does** apply a truncated operator matrix: it builds the
> `cutoff × cutoff` matrix of **exact** elements `⟨m|S|n⟩` via the
> Miatto–Quesada recurrence (`piquasso/_math/gate_matrices.py`,
> `create_single_mode_squeezing_matrix`) and applies it. The distinction that
> matters is *truncation of the exact operator* (what piquasso does, and what J2
> recommends) versus *exponential of the truncated generator* (what fails).
> Column 0 of the exact-element matrix **is** the closed form, which is exactly
> why the two agree on vacuum. Note this also means J2's recommendation is not
> an alternative to piquasso's method — it IS piquasso's method.

The real cause is a **normalisation convention**: Aria renormalises by the
represented mass, piquasso returns raw truncated probabilities. `sum(probs)` is
`0.999936664825` at r=0.8. Renormalising piquasso's own vector and diffing
reproduces **3.434e-08** at r=0.5 and **4.736e-05** at r=0.8 — *exactly* the two
numbers the cross-check reports as "reconciled by truncation budget".

So the check passes, and its bound is even valid (`p/(1−ε) − p ≈ pε`, maximal at
the dominant `p`, which is why the gap tracks the lost mass). But the
**explanation is wrong** and the tolerance is roughly **10¹⁰ looser than
necessary** — it is a leak-budget-sized tolerance standing in for a
floating-point-sized one. That is this project's signature failure wearing a
different hat: a check that passes for a reason other than the one documented.

**Fix:** compare on one explicit normalisation convention and tighten to ~1e-14,
dropping the leak-budget path from the probability comparison.

> **The first draft added "(keep it for `⟨n⟩`, where the weighted tail genuinely
> is the bound)" — which repeats the exact error J3 diagnoses, inside the
> correction for it.** The fixture's `mean_n` is computed as
> `(ks*probs).sum()/total` (`piquasso_ref.py:115`) — **already renormalised** —
> and Aria's `expect_n` divides by `norm_sqr()`. Same convention on both sides.
> Verified: the fixture's `mean_n` and a renormalised recomputation from its own
> `probs` differ by **0.000e+00** across all 16 cases, and the same-convention
> ⟨n⟩ diff against Aria is **2.2e-16**. Yet `piquasso_xcheck.rs:374` budgets it
> at `lost_n_weight().max(1e-9)` ≈ 1.4e-3 at r=0.8 — about **10¹² too loose**.
>
> `lost_n_weight` bounds `|⟨n⟩ − sinh²r|`, the distance to **truth**. It does
> not bound `|⟨n⟩ − piquasso|`, which is float noise. The weighted-tail budget
> is correct only where the reference is **analytic** — i.e. in
> `matches_piquasso_and_predicts_where_piquasso_goes_wrong` in `lib.rs`, not in
> the differential test. **Tighten the ⟨n⟩ lane to ~1e-14 as well.**

Expected to still pass with margin: renormalising both sides across all 16 cases
gives worst diffs 1.1e-16 (probs), 3.7e-16 (amps), 2.2e-16 (⟨n⟩). If it does
not, that is a real defect the loose tolerance was hiding.

**Also fix while there** — `piquasso_xcheck.rs:52` references `TRUNCATION_CASES`,
an identifier that exists nowhere in the repo. It was removed during review when
the per-case budget replaced the exception list, and the doc reference was left
dangling.

Correct the prose in both files rather than deleting it — the wrong explanation
should stay visible next to the measurement that overturned it.

## Part K — N-way correctness matrix (task #16), new 2026-08-08

> "a few correctness tests with ppvm / tsim / mps / pauli-prop / qiskit /
> qiskit-mps with the SAME QASM — correctness is key"

One corpus of QASM, every engine that can run it, one report. The value is not
in adding another green tick; it is that **a defect surviving one engine rarely
survives seven**. The risk is the opposite: a matrix is the easiest possible
place to hide a check that never ran.

### K1. Inventory — and the capability classes that make the matrix SPARSE

| engine | kind | runs qubit QASM? | output |
|---|---|---|---|
| `statevector` | in-tree, dense | yes | counts / probs / expectation |
| `noisy-statevector` | in-tree, dense + channels | yes | counts |
| `mps` / `noisy-mps` | in-tree, tensor | yes | counts / expectation |
| `pauli` | in-tree, **stabilizer** | **Clifford only** | counts / expectation |
| `pauliprop` | in-tree, Heisenberg | yes | **expectation ONLY — no sampling** |
| `photonics` | in-tree, Fock | **no** — photonic circuits, not qubit QASM | counts |
| `metal-` / `opencl-` / `cuda-statevector` | in-tree, GPU | yes, platform-gated | counts |
| Qiskit (Aer) | bridge | yes | counts |
| ppvm | bridge | **general qubit QASM** (no swap/ccx) | **counts — it is a SAMPLER** |
| tsim | bridge | general qubit QASM, incl. arbitrary rotations | counts |
| Perceval / Bloqade | bridge | **qubit QASM2** (already Qiskit-anchored) | counts |

> **The ppvm row was WRONG in the first draft** ("Pauli-sum, expectation-ish"),
> and it propagated into the lane structure below. `ppvm_runner.py:9-28` states
> the reason plainly: ppvm has two engines, and the bridge deliberately uses
> `GeneralizedTableau` via `sample_stim` because
> `PauliSum.overlap_with_zero()` "returns one number and could not honour
> `shots` without lying about what it measured". `src/ppvm.rs:34` returns
> `Counts`. **There is no expectation path in the bridge protocol at all**, and
> ppvm *already runs in a Qiskit-anchored counts comparison today* —
> `cross_backend.rs:576` (`ppvm_matches_qiskit_within_threshold`, L2 ≤ 0.0025
> at 4M shots).
>
> Consequence: putting ppvm in an "expectation lane" would have (a) removed it
> from the lane it already occupies and (b) silently required a whole new runner
> mode plus an observable wire format. If a pauliprop↔ppvm **expectation**
> cross-check is wanted — which is Part E's actual motivation — that is new
> protocol work and must be scoped as such, not assumed.
>
> Perceval and Bloqade were also mis-rowed as non-qubit; both already run qubit
> QASM2 in Qiskit-anchored arms.

**Three different reasons a cell is empty**, and they must not look alike:

1. **Cannot express** — `pauli` on a non-Clifford circuit, `photonics` on qubit
   QASM. A *correct* refusal.
2. **Not installed** — a bridge venv is absent. Environmental.
3. **Not implemented / crashed** — a real gap.

Collapsing these into "skipped" is how a matrix reports coverage it does not
have. Each cell records **which** of the three, and the report prints the counts
separately.

### K2. Common currency — the part that decides whether this is comparable at all

Engines return counts, probabilities, or expectation values. Comparing them
needs one currency, and the choice has teeth:

* **Counts → distribution over bitstrings.** Use **L2**, as
  `cross_backend.rs` already does, with the σ-derived per-arm thresholds it
  already carries (including a documented 2.5σ-flake → 5σ fix). L2's null scale
  is `√(Σ 2pₖ(1−pₖ)/n) ≤ √(2/n)` — **dimension-free**.

  > **The first draft proposed TVD at a `~1/√shots` threshold. That is
  > quantitatively wrong**, and wrong in the same genre as J2's unquantified
  > "conservative threshold". Null-hypothesis TVD grows like **√d**:
  > `E[TVD] ≈ Σₖ √(pₖ(1−pₖ)/(πn)) ≈ √(d/(πn))`. Measured, two-sample
  > multinomial at n=8192:
  >
  > | d | measured null TVD | `1/√n` | `√(d/πn)` |
  > |---|---|---|---|
  > | 2 | 0.0062 | 0.0110 | 0.0088 |
  > | 8 | 0.0162 | 0.0110 | 0.0176 |
  > | 64 | 0.0490 | 0.0110 | 0.0499 |
  > | 1024 | 0.1970 | 0.0110 | 0.1995 |
  >
  > A `c/√shots` threshold is **~2× too tight** on a 2-outcome Bell circuit and
  > **18× too loose** at d=1024 — vacuous exactly where a wide circuit could
  > hide a defect. So: keep L2. If TVD is wanted anyway, the threshold must be
  > computed per circuit from the anchor's empirical distribution as
  > `k·Σₖ√(p̂ₖ(1−p̂ₖ)/(πn))`, not from shots alone.

* **`pauliprop` is genuinely expectation-only** (`sim.rs:758` — `execute()`
  unconditionally returns `Unsupported`), so it needs an expectation lane. But
  see K3: that lane currently has **no independent anchor**.
* **Analytic vs stochastic must not share a tolerance.** `statevector` vs
  Qiskit `Statevector` is a 1e-15 comparison; `statevector` vs Aer at 8k shots
  is a 1e-2 comparison. One number for both is wrong twice.

### K3. Reference choice — anchor, do not compare all pairs

With `N` engines, all-pairs is `O(N²)` cells, most of them uninformative, and a
single bad engine reddens a whole row and column.

**Anchor on Qiskit** (the only fully independent implementation available) and
compare every engine to it, plus the analytic truth where a circuit has one.
Report pairwise only for engines Qiskit cannot reach.

> **The anchor does NOT currently reach the expectation lane, and the first
> draft did not say so.** `qiskit_runner.py:66-79` supports modes
> `execute` / `qpy_to_qasm2` / `qasm2_to_qpy` — no expectation, no statevector
> output — and `tools/qiskit_xcheck/compare.py` compares probabilities and
> counts, never expectations. With ppvm correctly moved back to the counts lane
> (K1), the expectation lane reduces to `pauliprop` vs `statevector` vs `mps`:
> **all in-tree**. That is exactly the weak-evidence configuration this section
> warns about — the one that let the `Reset` defect through — so the plan would
> have recommended anchoring while building the one lane that cannot anchor.
>
> Either add an expectation mode to the Qiskit runner (new protocol work, scope
> it) or state plainly that the expectation lane is internal-consistency only
> and carries less weight than the counts lane.

Rationale from this repo's own history: the `Reset` defect survived every
*internal* cross-backend gate because each pair coincided in the basis being
checked. Pairwise agreement among our own engines is weak evidence; agreement
with an outside implementation is strong.

### K4. The traps — each one has already bitten this project

1. **Do NOT route through Aria's QASM export.** Part H is still open: export
   drops `if(c==V)`. A differential test conducted *through* a lossy export
   agrees because both sides lost the same thing. Feed every engine the **same
   source text**; compare Aria's *native* execution against the others.
2. **Conditionals must sometimes be FALSE.** An always-true condition hides a
   dropped guard — both engines agree while one ignores the guard entirely.
3. **The corpus has holes today.** Measured over the 11 shared fixtures:
   `measure` in **10/11**, **`if(` in 0/11**, no `reset`. Gate coverage is otherwise
   good (h, cx, rz, ry, rx, u1/u2/u3, s/sdg, sx/sxdg, swap, cz, cy, crz, ccx,
   barrier, t, tdg, p). So the corpus **cannot currently reach** the exact defect
   class fixed in `11888a9`/`ae6da5c`. Extend it before trusting the matrix.

   > The first draft said `measure` in **11/11**. It is **10/11** —
   > `09_unitary_only.qasm` has no measure statement; the word appears only in
   > its comment ("No creg, no measure — the runner must synthesise a full-width
   > measurement"), and a bare `grep -l measure` matched the comment. A claim
   > stated as measured that was really a grep hitting prose: the exact failure
   > this document keeps cataloguing, committed while cataloguing it.
   >
   > The fixture is operationally relevant too, not just a counting slip:
   > no-measure and partial-measure circuits are precisely where **counts-keying
   > conventions diverge** — in-tree engines return `u64` keys over the full
   > register while bridges return creg-width LSB-first strings. That conversion
   > is unstated work in K5.

6. **The corpus identity is not stable.** `crosscheck_corpus()`
   (`cross_backend.rs:92-109`) *prefers* a private `../verify-qiskit` tree
   (369 files) when checked out, falling back to the 11 vendored fixtures. The
   hole analysis above describes the **fallback**, which is what is active on
   this machine. On another operator's box the matrix would silently run a
   different, unaudited corpus and report the same green. Pin which corpus ran,
   and print it.
4. **An all-Clifford corpus makes `pauli` agree trivially.** Non-Clifford
   rotations are what give a stabilizer backend a chance to be wrong.
5. **Report the QUALIFYING COUNT per engine**, not just failures. "17/17 agree"
   over a matrix where five engines ran two circuits each is worse than no
   matrix, because it reads as coverage.

### K5. Shape

**Build on `crates/omega-bridges/tests/cross_backend.rs`, not on
`omega-xcheck`.** The first draft called this "an extension rather than a new
harness" of omega-xcheck. That understates it badly, and it names the wrong base.

* `omega-xcheck/src/main.rs` generates random circuits from a **six-gate**
  vocabulary (h/s/sdg/x/z/cx) plus a hardcoded 8-circuit feedforward set, over a
  bespoke `C …`/`M …` line protocol that `compare.py`'s gate dictionaries mirror
  exactly. It has **no QASM ingestion**, no mps/pauliprop/noisy dispatch, and no
  bridge invocation. Reaching the corpus gate set (u2/u3/sx/ccx/crz/p…) means
  rewriting emitter and parser — a new harness in all but name.
* `cross_backend.rs` **already is** the skeleton described here: a corpus walker
  (`crosscheck_corpus`), a Qiskit-anchored loop, capability filtering via runner
  gate-set introspection (the `{"mode":"gates"}` query), a skip taxonomy, and
  the vacuous-pass guard. Adding in-tree engines there via `omega-parser` plus
  the `Backend` trait is the short path.

Unstated work either way: **counts-key conversion**. In-tree engines key counts
by `u64` over the full register; bridges return creg-width LSB-first strings.
These diverge exactly on the partial-measure and no-measure fixtures.

Output: one row per (circuit, engine) with `status ∈ {agree, disagree,
cannot-express, not-installed, error}`, the metric and its threshold, and a
summary that prints **per-engine qualifying counts** and fails on any
`disagree` or `error`.

> **`cannot-express` cannot be populated from the bridge taxonomy as it
> stands.** `runner.rs:234-247` maps only kinds ending `-not-installed` to
> `BridgeError::Unavailable`; every other structured failure — **including
> `ppvm-unsupported-gate` and `tsim-unsupported-gate`** — collapses into
> `BridgeError::Backend` with the kind as a string prefix in the message. So
> distinguishing "cannot express" from "error" means string-sniffing, extending
> `BridgeError`, or relying on the up-front gates introspection — which only
> tsim and ppvm implement (qiskit/perceval/bloqade runners have no `"gates"`
> mode). Perceval limits already surface as a skip-with-continue
> (`cross_backend.rs:331`), i.e. the code **already commits** the
> cannot-express-looks-like-not-installed collapse K1 warns against. Fixing the
> taxonomy is part of this work, not a precondition someone else met.
>
> In-tree engines map cleanly: `OmegaError::Unsupported` → `cannot-express`, and
> both `pauli` and `pauliprop` refuse with typed errors.

Gate it behind `ARIA_NWAY=1` alongside the existing cross-checks, and route the
skip through `skipped()` so an absent matrix announces itself in the final
summary line rather than passing silently.

### K6. Order

1. Extend the shared corpus with conditionals (including sometimes-false) and
   `reset` — **before** building the matrix, so it is not built against a corpus
   that cannot fail.
2. Fix the bridge error taxonomy so `cannot-express` is typed, not sniffed.
3. Counts lane: statevector / mps / noisy-mps / pauli / tsim / **ppvm** /
   perceval / bloqade / Qiskit — extending `cross_backend.rs`, which already
   anchors several of these.
4. Expectation lane: pauliprop / statevector / mps — **and label it
   internal-consistency only** until the Qiskit runner gains an expectation
   mode, per K3.
4. GPU rows behind their existing platform flags.
5. Only then wire the CI stage.
