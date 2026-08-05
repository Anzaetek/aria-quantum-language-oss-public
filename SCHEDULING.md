<!-- SPDX-License-Identifier: Apache-2.0 -->
# Scheduling, resource limits, and large batch workloads

For anyone driving `omega-server` with work that is **large, long-running, or
heterogeneous** — architecture search, QML training loops, parameter sweeps.
It states what the server guarantees **today**, what it does not, and what a
client must therefore do itself.

Written 2026-08-06. Where behaviour is planned rather than present, it says so;
nothing here describes something that does not exist.

## 1. The contract: the server supplies primitives, the client owns policy

The server deliberately contains **no search strategy**. No TPE, no Hyperband,
no pruning heuristic, no epoch scheduler. It executes circuits and enforces
resource limits. Strategy belongs to the client, because the alternative — one
search policy compiled into the server — fits exactly one consumer and obstructs
every other.

Concretely, that means a client is expected to own: trial selection, early
stopping, retry policy, checkpointing, and the decision about what to do when
capacity is refused. The server's job is to be *predictable and honest* about
what it will accept.

## 2. Admission control (present)

Every circuit is **priced before it is allocated**. A dense statevector needs
`2^n × 16` bytes and `n` is declared in the request, so the cost is known in
microseconds — no allocation attempt, no OOM discovery:

| qubits | statevector | with an adjoint/gradient sweep |
|---|---|---|
| 28 | 4 GB | 8 GB |
| 30 | 16 GB | 32 GB |
| 32 | 64 GB | 128 GB |
| 34 | 256 GB | 512 GB |

Each extra qubit **doubles** the requirement, and there is no graceful
degradation — a job fits or the host dies. Hence pre-flight refusal.

Two refusal codes, and **they mean different things**:

| code | meaning | correct client response |
|---|---|---|
| **413** | Larger than this host's entire budget. Waiting cannot help. | Do **not** retry. Reduce qubits, use a compact backend, or target a bigger host. |
| **429** + `Retry-After` | Fits, but not alongside what is running now. | Retry on the header's schedule. |

Treating 413 as retryable turns a clear answer into a hammering loop. Both
responses carry the computed requirement and the host's budget, so the numbers
needed to react are in the error itself.

Admission is weighted in **bytes, not job count** — a "max N concurrent jobs"
limit is meaningless when job cost spans six orders of magnitude.

`GET /health` publishes `execution.capacity_bytes`, `available_bytes` and
`max_qubits`, so a client can size work *before* submitting rather than
discovering the ceiling by rejection.

### Cost is priced by what a backend allocates, not by its name

Worth knowing when predicting whether your job will be accepted: a compact
*representation* is not automatically a compact *allocation*. The MPS backend
contracts to a full statevector on the expectation path, so an MPS circuit is
priced densely there — and the `Auto` selector routes non-Clifford circuits of
n ≥ 20 to MPS. Photonic cost is combinatorial in photon number, not mode count.
Pricing follows the allocation in each case.

### Memory pools: unified vs discrete hosts

How many budgets the server keeps depends on the machine, and it detects this
rather than assuming:

| machine | pools |
|---|---|
| CPU-only server | one (host) |
| Apple Silicon, DGX Spark (GB10) and other **unified**-memory hosts | **one shared pool** — CPU and GPU work debit the same budget, because that memory exists once |
| Discrete NVIDIA (RTX / A100 / H100) | host, plus one per GPU |
| DGX A100/H100 (8 GPUs) | host, plus eight |

The asymmetry matters: on a unified box, budgeting host and device separately
would double-count and OOM the machine, while on a discrete box assuming unified
merely under-uses VRAM. **So when detection is uncertain, the server assumes
unified** and says so in `/health`.

GPU work is priced at f32 (8 bytes per amplitude), not the CPU's 16, and a job
targeting a device is sized against **that device's** memory — not host RAM. A
device the server has no budget for is refused rather than quietly charged to
the host.

`GET /health` reports every pool separately, so you can see *which* resource is
scarce instead of one aggregate hiding a full GPU behind an idle host.

### Operator limits

Defaults derive from detected memory — a cgroup limit when present, so a
containerised server budgets against its container and not the host.

| variable | meaning |
|---|---|
| `OMEGA_MAX_MEM` | Absolute budget. **Human units accepted**: `48G`, `8GiB`, `1.5T`, or a plain byte count. |
| `OMEGA_MEM_FRACTION` | Share of detected memory: `0.25` or `25%`. |
| `OMEGA_MAX_CONCURRENCY` | Cap on simultaneous jobs, independent of their size. |
| `OMEGA_CPU_FRACTION` | Share of cores, converted into a job cap. |
| `OMEGA_RESOURCE_PROFILE` | `gentle` (25%), `balanced` (50%), `greedy` (90%) — one knob instead of five. |
| `OMEGA_MAX_QUBITS` | Hard width ceiling. |
| `OMEGA_MEM_TOPOLOGY` | `unified` / `discrete` / `host` — overrides pool detection. |

Three rules govern how these combine:

1. **An absolute setting beats a fraction**, and the resolution is reported.
2. **Caps compose by `min`, never `max`** — a generous cap never widens a strict
   one, and no cap can exceed what the hardware actually has.
3. **On unified memory the host and device shares address the same pool** and do
   not compound.

**A malformed value is a startup error, not a silent fallback.** Previously
`OMEGA_MAX_MEM=48G` parsed as garbage and quietly left a 4 GiB budget — which
looks exactly like a throttle that worked.

Two limits are orthogonal and both apply: **bytes** bound total memory, while
**concurrency** bounds CPU contention. Many small jobs can saturate the cores
without approaching the memory budget, which a memory-only limit cannot express.

`GET /health` reports every effective limit *and where it came from*
(`env:OMEGA_MEM_FRACTION`, `profile`, `detected`, `default`), so "why was my job
refused?" is answerable without reading the server's source.

## 3. Known limits today — plan around these

These are **current** constraints. Each is tracked in `FIXES_PLAN.md`; listed
here because a client built without knowing them will be surprised.

1. **Execution is synchronous.** A request is held open for the whole run. A
   long job is therefore exposed to every idle timeout between client and
   server. *Keep individual requests short.*
2. **No durable jobs.** Job state lives in the request. If the connection drops,
   the work is lost — there is no id to reconnect with and no partial result to
   collect. *Checkpoint client-side and submit in chunks you can afford to redo.*
3. **A batch aborts on the first bad row.** `POST /v1/quantum/expectation` with
   `circuits: [...]` returns `400` for the whole submission if any row fails —
   completed rows are discarded. *For heterogeneous sweeps where one config may
   be invalid, submit in small chunks so one bad row costs little.*
4. **One reservation per batch, priced at its worst row, held for the batch.**
   A sweep mixing a 30-qubit config with 28-qubit ones reserves for the largest
   and holds it throughout. On a shared host that reduces what others can run.
   *Prefer several small batches over one large heterogeneous one.*
5. **No per-row cancellation.** A batch can only be abandoned wholesale, so
   pruning individual trials mid-flight is not possible server-side. *Drive
   pruning by submitting trials in small batches you can simply stop issuing.*
6. **No gradient endpoint.** Training loops are local-only; the wire carries
   scores, not derivatives.
7. **Every row carries its full circuit.** There is no template + parameter
   matrix, so a batch re-sends the whole gate list per row. For same-ansatz
   workloads this dominates the payload. *Over a tunnel, prefer fewer, larger
   requests and enable compression at the transport.*
8. **One execution path is ungoverned**: circuits executed through the WASM
   lambda route are not admission-controlled. Guest fuel limits guest
   instructions, not the host-side statevector. *Do not send untrusted or
   unbounded-width circuits through it.*
9. **No queue.** Refusal is immediate rather than queued, so a steady stream of
   small jobs can keep a large one waiting indefinitely. *Back off, and do not
   treat `Retry-After` as a promise of eventual admission.*

## 4. Consequences for architecture search and QML training

**Architecture search** — heterogeneous by construction: each trial is a
*different* circuit, not the same circuit with different parameters.

- Batch-level payload optimisations that assume a shared ansatz do not apply;
  every trial pays full topology cost.
- Because a batch reserves for its worst row (§3.4) and cannot cancel rows
  (§3.5), **submit trials individually or in small groups**. That preserves the
  ability to prune, keeps reservations proportional, and limits the blast radius
  of one invalid configuration (§3.3).
- A trial that is itself a long training run gets no intra-trial progress
  reporting from the server; track progress client-side.

**QML training** — homogeneous rows, but many steps.

- Rows share an ansatz, so batching helps most here; group a data batch into one
  request rather than one request per row.
- Gradients are not available remotely (§3.6): training runs locally, or scores
  remotely and differentiates locally.
- Do not rely on the server to retain anything between steps. It does not.

**Both**: assume the connection can drop and the work is lost. Until durable
batches land, the client's checkpoint is the only thing that survives.

## 5. What is planned

`FIXES_PLAN.md` carries the design work, sequenced: per-pool memory accounting
across unified and discrete GPU hosts; fractional operator throttles; durable
batches with reconnect-and-resume, per-row status and per-row cancellation; an
async job protocol with heartbeats; client-side resilience including a
long-wait mode for laptops that disconnect; and formal models (TLA+ for the
scheduling and batch-lifecycle protocols, Lean 4 for the admission arithmetic).

Those models are worth one caveat, because it applies to the guarantees above
too: **a model verifies a protocol against its assumptions, and cannot tell you
an assumption is wrong.** The admission logic was once provably consistent while
under-pricing a backend by four orders of magnitude — the protocol was fine, the
input was not. Checking that pricing matches what backends actually allocate is
the job of differential tests against the backends, not of any model. Treat
resource guarantees as only as good as the measurements behind them.
