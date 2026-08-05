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
| 4 | `ARIA_BUG_mps_fixed_bond_no_truncation_error` (×2) | **LIKELY FIXED** — `MpsState::discarded_weight` accumulates per-SVD (`mps.rs:80,183,298`) | verify, then close |
| 5 | `ARIA_BUG_mps_svd_instability_normal_equations` (×2) | **LIKELY FIXED** — bond SVD is Jacobi (`svd::truncated_svd_flat`), not normal equations (`mps.rs:44,117`) | verify, then close |
| 6 | `ARIA_BUG_noise_ignored_non_statevector_backend` | **LIKELY FIXED** — MPS has trajectory-noise Monte-Carlo (`omega-backend-mps/src/sim.rs:295-335`, analytic parity tests at :1206+); omega-cli refuses loudly on backends that can't carry it (`main.rs:937-944`); the bug doc itself is marked FIXED UPSTREAM ✅ | verify, then close |
| 7 | `ARIA_REQ_qasm3_interchange` | **PARTIAL** — QASM3 paths exist (`aria-core/src/ast/qasm.rs`, `omega-parser/src/lower.rs`, CLI) | audit vs the request's checklist |
| 8 | `ARIA_REQ_ibm_calibration_noise_import` | **OPEN (probable)** | Part D, after 6 |
| 9 | `CI-LIBTORCH-DYLD-FIX.md` + `0001-*.patch` | **SUPERSEDED** — CI now fetches libtorch and exports `DYLD_LIBRARY_PATH` via `tch-env.sh` | close as done |
| 10 | `PLAN-BACKEND-PLUGINS.md` | **LANDED** — plugin ABI + `omega-plugin-conformance` are CI stage 6b | close; CV may extend the corpus |
| 11 | `aria-supervised-qml-requirements` | **LANDED** (phases 0–4) | spot-check only |
| 12 | `aria-torch-requirements` | **LANDED** — tch stage on by default | none |
| 13 | `ARIA-KEEP-public-final.md` | **CONTRACT** — K1–K14, Q5/Q8/Q9 | binding on everything below |
| 14 | `PLAN-COMPLIANCE-2026-07-29`, `UPSTREAM_UPDATE_PLAN`, `aria-open-requirements` | historical/superseded | archive note |

Items 4, 5, 7 need verification rather than implementation; 6 and 8 need a
confirming reproduction before an estimate is honest.

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
3. **A7** — resource governor: admission control before anything makes it
   easier to submit big jobs to a shared box
4. **A1 → A2 → A3** — remote gradient, batching, pyo3 (unblocks remote training)
5. **A6** — async job protocol over the existing `/v1/ws`
6. **D4, D1** — verify-and-close the three already-fixed bug docs
7. **D3** — QASM3 audit against the request's checklist
8. **A5** — expectation-path device routing + honest `backend_name`
9. **B1–B6 + C2** — CV backend and its examples, gated on the DGX search
10. **C3** — photonics from the Aria surface: only if separately prioritized

Constraints throughout: `./ci.sh` is the single source of truth and must stay
green (K13), with external oracles optional and cleanly skipped; no stdout from
libraries (Q8); tolerances stated numerically (Q9); batch index preservation
(Q5); assumption-ledger discipline (K12). Note the ledger counts do **not** move
for C1/C2 — the verify corpus enumerates `examples/aria/*.aria` only.
