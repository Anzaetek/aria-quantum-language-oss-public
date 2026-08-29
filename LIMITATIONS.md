<!-- SPDX-License-Identifier: Apache-2.0 -->
# Known limitations

Honest scope boundaries of this OSS release. None of these affect the numeric
guarantees of the verified examples ([VERIFICATION.md](VERIFICATION.md)); they
record where a shipped artefact is a *template/showcase* rather than a faithful
end-to-end implementation, and which deeper features are deferred.

## Examples

- **`shor_ecdlp.aria` — showcase (does not lower).** The program parses and
  instantiates, but its `oracle ec_step` subroutine is declared on a separate
  `qreg r[7]` that the lowering does not map into the main circuit's qubit
  space. Cross-register oracle-subroutine inlining is **not implemented**, so
  the circuit cannot be lowered or run. It is excluded from `aria-verify` and
  labelled showcase in the verification table.

- **`hhl.aria` and `qsvt_invert.aria` — structural templates.** These express
  the *shape* of HHL / QSVT (QPE cascade + controlled rotation; alternating
  signal-rotation / signal-processing blocks) with placeholder angles ("baked in
  by host"). Their forward `⟨Z_q⟩` profile is numerically cross-checked
  (differential oracle), but the `.aria` form is **not** a faithful solver. The
  faithful, *proven* versions live in Lean — `proofs/lean4/QuantumProofs/HHL.lean`
  and `QSVT.lean` (sorry-free, axiom-clean) — and in the pure-Rust kernels
  `omega_core::solver` and `omega_core::chebyshev`.

## Gate set / backends

- **`RBS` (Givens rotation) backend coverage.** The native `RBS` gate runs on
  the CPU statevector and MPS backends — the engines every trainer and
  verification harness uses — with analytic adjoint derivatives and the 4-term
  Givens parameter-shift rule. It also runs on the **CUDA and Metal statevector
  backends**: the Givens rotation on span{|01⟩, |10⟩} goes through the generic
  2-qubit apply (f32 kernels), and `dRBS/dθ` is wired into the GPU adjoint, so
  both forward runs and adjoint gradients of butterfly / unary QML circuits are
  numerically gated against the CPU f64 path (`gpu_cuda_agrees_with_sim_on_rbs`
  / `gpu_cuda_rbs_gradient_agrees_with_sim` and their Metal mirrors). The
  OpenCL statevector backend refuses it explicitly (in both `execute` and the
  adjoint dagger), and the Clifford `pauli` backend and the optional `tch`
  plugin dispatch it to their wildcard arm; all three surface a clean
  *"unsupported gate"* error at runtime, so the CLI falls back to the CPU
  statevector backend rather than producing a wrong result. QASM 2.0 export
  decomposes it exactly, so exported circuits run anywhere.

- **Mid-circuit `Reset` semantics.** `reset q` is the non-unitary *channel*
  `ρ → |0⟩⟨0|_q ⊗ Tr_q(ρ)`: the qubit is discarded and replaced by a fresh
  |0⟩, and any entanglement it had is **destroyed**, not transferred. The
  result is a mixed state, so a statevector / MPS / tableau backend implements
  it by **sampling** — measure `q`, project, apply X if the outcome was 1 —
  and the ensemble comes from running independent shots. Consequences:

  - **`shots` is required for a reset on an entangled qubit.** Analytic runs
    (`shots = None`: `--statevector`, `--expectation`) refuse it, because one
    state vector holds one trajectory and the answer would otherwise depend
    silently on an RNG draw. A reset on an *unentangled* qubit is
    deterministic and is still served analytically (the CPU statevector tests
    reduced purity exactly; the stabilizer and MPS backends use a coarser,
    conservative test and may refuse some resets that are in fact
    deterministic).
  - **`pauliprop`, the `tch` plugin, and `aria-verify`'s reference simulator
    refuse `Reset`.** They evolve a pure state, or conjugate observables
    unitarily, and cannot represent the channel. They previously *skipped* it
    silently, which meant answering a different circuit than the one
    submitted.
  - **The OpenCL statevector backend refuses `Reset`**, so the CLI falls back
    to the CPU statevector backend.
  - **The Metal statevector backend implements the channel.** The port landed;
    re-measured 2026-08-16 on Apple silicon, `reset_matches_cpu` passes and the
    full Metal suite is 81/81 green under `ARIA_METAL=1`.

    This entry previously warned that Metal carried "the old, incorrect
    implementation" and told readers to "treat Metal `Reset` results as wrong".
    That was written while the port was outstanding and was never revisited, so
    for some time it was directing users to distrust correct results — the
    inverse of the failure this file usually guards against, and worth the same
    attention: a stale warning spends trust that a real one later needs.

## CPU statevector: what it is good for, and where to stop

**This section was wrong in three ways until 2026-08-18**, and each error pushed
a reader the wrong way: it said the backend was single-threaded (so do not
bother with cores), that sampling doubled peak memory (so over-provision by 2x),
and that multi-controlled gates were the weak spot (so avoid Toffoli circuits).
All three had been fixed in the code and not here. They are corrected below with
what the code does today.

### What it does now

- **Gate kernels are PARALLEL.** `apply_1q`, `apply_2q`, `apply_cx`,
  `apply_diagonal_1q`, `apply_ccx` and `apply_cswap` all use rayon above a
  `dim`-based threshold (`PAR_MIN_DIM`, currently `2^12`). Below it they run
  serially, because the dispatch costs more than the work.

  Results are **bit-identical across thread counts** — not within a tolerance.
  Every parallel region is a map over disjoint index groups, so each amplitude
  is written once from the same expression, and no floating-point association
  changes. `tests/thread_count_invariance.rs` asserts that at `T ∈ {1,2,12}`.

  Reductions are deliberately NOT parallelised: the norm, the cumulative scan
  and `expectation_pauli` would reassociate and change the last bits.

- **Sampling no longer doubles peak memory.** `sample_counts` used to hold two
  further `2^n` f64 vectors beside the state — at 28 qubits, 2 × 2.1 GB on top
  of 4.3 GB. It now sorts the `shots` draws and walks the state once, so the
  auxiliary allocation is `shots` f64s: **kilobytes, not gigabytes**. Peak at 28
  qubits went from ~8.6 GB to ~4.3 GB, which is the difference between running
  and not running on a 24 GB box with anything else open.

- **Memory is still the wall, just a lower one.** The state is `2^n × 16` bytes —
  4 GiB at 28 qubits — and that is now essentially all of it. The backend
  refuses rather than swaps: the capacity guard names the required and available
  figures.

- **Above ~28 qubits, change backend rather than wait.** `mps` for
  low-entanglement circuits (with a truncation certificate on every run),
  `pauli` for Clifford ones (exact to 1024 qubits), `pauliprop` for expectation
  values. The hard ceiling is 64 qubits and it is arithmetic, not resources:
  `2^64` indices do not fit in a `usize`.

- **Multi-controlled gates are no longer the weak spot.** `apply_ccx` and
  `apply_cswap` used to scan the whole index space and act on one index in
  eight. They now enumerate the `dim/8` subspace directly and run in parallel.

### The timing table is STALE, and is not being replaced here

| circuit | width | time |
|---|---|---|
| ghz_28 | 28 | 12.0 s |
| qft_26 | 26 | 35.3 s |
| qft_28 | 28 | 163.4 s |

Measured on 12-core Apple silicon, release, **one thread**, 1000 shots, at the
**S1** stage of `PLAN-SV-PERF.md` — before the gate kernels were parallelised
and before the sampler rewrite. So these are an upper bound on today's
single-thread cost and say nothing about the parallel path.

They are left in place rather than refreshed **on purpose**. Re-measuring here
would produce Apple M5 Max numbers, and `PLAN-PERF-BASELINES.md` rules that
machine out as a baseline: it is a high-end part, so figures taken on it flatter
the code and would hide a regression that matters on ordinary hardware.
Authoritative numbers are wanted on an **M4** and an **RTX 6000 Pro**, and until
one of those is available the honest state is a labelled stale table rather than
a fresh misleading one.

## Classical linear-algebra stack (`omega_core`, `aria_runtime::linalg`)

- The QSVT phase angles use a **placeholder heuristic** (`chebyshev::
  qsvt_inversion_angles`), not the full Wang–Lin angle-finding reduction; the
  block-encoding circuit it drives is a resource-estimation skeleton.
- `block_encode_dense` returns `α = NaN` for **non-power-of-two** matrix
  dimensions (the Pauli decomposition requires `2^k × 2^k`). Callers must pad.
- The classical reference solvers (`solver::solve`, `solve_classical`) are exact
  and fuzz-tested; the *quantum-circuit* recipe they pair with is for resource
  estimation, consistent with the templates above.

## Formal proofs

- The shipped gate-model export (`aria export --gate-model`) is proven
  sorry-free for the **recognized** circuits (Bell, GHZ, QFT, QPE, Grover) via a
  structural recognizer. The **general arbitrary-circuit** correspondence (the
  BKMP `circuit_to_pattern` induction) is **deferred** — an `.aria` outside the
  recognizer whitelist exports a circuit definition but no closed correspondence
  theorem.

## Verification scope

- The **differential** run-gate oracle compares the diagonal `⟨Z_q⟩` profile of
  the unitary part of a circuit. It is invariant to global phase and does not
  assert off-diagonal observables or measurement-protocol post-processing; those
  examples that *have* a closed-form answer use a stronger **classical** oracle
  (see the table in [VERIFICATION.md](VERIFICATION.md)).

## Mid-circuit measurement, feedforward, and what the counts keys mean

Mid-circuit measurement **works** on the CPU backends: `measure` mid-circuit,
`reset`, and `if (c == V)` / Aria `when` feedforward all execute, and the
correlations they create are real rather than approximated. This section exists
because the *output format* changes underneath you when you use them, which is
surprising the first time and silent when it happens.

**A circuit with no mid-circuit measurement** reports counts keyed by the
**qubit basis state** at the end of the run — one character per qubit.

> **Interop caveat, and it is the one that surprises people comparing to
> Qiskit.** Qiskit *always* keys counts by the classical register, so for a
> circuit with only terminal measures its key width is the `creg` width while
> ours is the qubit count. On `qreg q[2]; creg c[4]` with two measures, Qiskit
> gives `'0011'` and we give `|11>`. Neither is wrong — they are answering
> different questions, and ours is the one a user asking "what state did the
> register end in" wants — but a naive diff of the two shows every key
> mismatching. Add a mid-circuit measurement or feed-forward and the run becomes
> creg-keyed, at which point the widths agree exactly.
>
> This is **not** the same thing as the declared-versus-written width defect
> fixed in `counts_outcome_width`: that one made the creg-keyed path itself
> disagree with Qiskit on a partially-written register, and it is closed. This
> caveat is about which register is keyed at all.

**A circuit that measures into a classical register** reports counts keyed by
the **contents of that register**, MSB-first, padded to the register's declared
width. That is the right answer — the classical record is what a shot of such a
circuit actually produces, and it is what any other toolchain reports — but it
means two things:

- **The key width follows the `creg`, not the `qreg`.** A 2-qubit circuit
  declaring `creg c[4]` prints `|1001>`, four characters wide, because four is
  what was declared. Padding is always to the full declared width, so keys stay
  comparable across shots.
- **A qubit measured twice reports its last value**, since the register holds a
  record, not a history.

The switch is pinned by `crates/omega-cli/tests/collapse_counts_use_the_creg_width.rs`:
`collapse_width_is_the_creg_not_the_qubit_count` (a 1024-qubit register with
`creg c[2]` reports width 2) and `skip_width_is_the_qubit_count` (no
mid-circuit measurement, so the whole qubit register is sampled and the width
is the qubit count). A third test asserts the two disagree, so a function that
ignored the distinction could not pass both.

The width is not capped at 64. Counts are keyed by an `Outcome` that carries as
many words as the register needs; `crates/omega-cli/tests/counts_width_boundary.rs`
pins 63, 64, 65, 128, 256 and 1024 qubits on both `mps` and `pauli`, asserting
key *contents*. The 64-qubit cliff a `u64` key used to impose is gone.

### What `⟨O⟩` means for a circuit that measures

Until 2026-08-20 every backend computed an expectation by **deleting** the
measurements and evaluating the remaining unitary. For
`h q0; measure q0 -> c0; if(c==1) x q1` that gave `⟨X₀X₁⟩ = +1`, the pure-Bell
value, where the truth for that circuit is **0** — a plausible number, silently
wrong, on the path QML uses.

It was not a convention borrowed from anywhere: Qiskit's `Statevector` and
`DensityMatrix` **refuse** a circuit with classical bits outright. Nothing
external ever contradicted the rule because nothing external was ever asked.

The contract now distinguishes three cases, and the distinction is between
*consequential* and *inert*, not between mid-circuit and terminal:

| the measurement | treatment | why |
|---|---|---|
| nothing reads its bit, nothing touches the qubit again | **elided** | it changes no observable of the prepared state, and this is what `remove_final_measurements()` does on the Qiskit side |
| a later gate is guarded on the bit it wrote | **deferred** into a quantum control, and the observable **dephased** on that qubit | the principle of deferred measurement, plus the dephasing that recovers the mixture |
| the qubit is used coherently afterwards | **refused**, naming the qubit | there is no deferred form; the measurement cannot move past work that depends on it having happened |

Inert measurements are elided rather than dephased deliberately. Dephasing them
would contradict the external oracle on 13 of the 14 crosscheck fixtures, and it
would mean that adding a readout line to a VQE ansatz silently changed its
energy.

The third case is the QEC ancilla-recycling pattern, so **refusal is the common
case for QEC circuits, not a corner**. `aria-qec` is unaffected because its
expectation path builds a measurement-free data-state circuit; the ancilla and
reset work goes through the counts path.

Verified against `AerSimulator(method="density_matrix")` — `⟨ZZ⟩ = 1.0`,
`⟨XX⟩ = 0.0`, `⟨YY⟩ = 0.0` against the pure-Bell `1.0 / 1.0 / −1.0` — and the
dephasing identity `Tr[M(ρ)O] = Tr[ρ M(O)]` is proved axiom-clean in
`proofs/lean4/QuantumProofs/DeferProbe.lean`.

**Every backend is on this contract:** CPU statevector, MPS, stabilizer,
pauliprop, Metal, OpenCL, CUDA and the `tch` plugin.
`crates/omega-cli/tests/deferred_expectation.rs` gates them across engines;
`tch` carries its own copy of those assertions
(`crates/aria-backend-tch/tests/deferred_expectation.rs`) because it links
libtorch and is excluded from the default workspace, so it cannot join that lane
without dragging libtorch into `omega-cli`'s build. Both call the same
`prepare_for_expectation`, so what is duplicated is the check, not the rule.

**Backend coverage.** CPU statevector, MPS and the stabilizer backend support
mid-circuit measurement. `pauliprop` cannot represent it in the sampling sense —
it propagates observables rather than states — and refuses rather than skipping;
for *expectation* it now answers deferrable feedforward circuits exactly, via the
shared deferral pass. For the GPU
statevector backends see `BACKEND_FEATURE_PARITY.md`, whose "Mid-circuit
measure" row is about those backends specifically and not about the CPU path
described here.

## Where gates are approximated rather than applied exactly

**`GATE-EXACTNESS.md` is the ledger.** Every gate whose implementation
introduces error — CUDA's `CRz` (FMA contraction), `CCX`/`CSwap` (a 15-gate
decomposition), the f32 default on every GPU backend, and the truncating
backends — is listed there with its bound, its reference, and the test that
enforces it. A gate that stops being exact must gain a row in the same commit.

## Reset: analytic mode costs a full host readback on CUDA

**The CUDA/CPU criterion divergence recorded here is RESOLVED (2026-08-17, on a
GB10).** CUDA used to refuse an analytic reset whenever `p0 ∈ (0,1)` — i.e.
whenever the *outcome* was random — while CPU and Metal refuse whenever the
qubit is *entangled*. `H q0; Reset q0` was therefore accepted by CPU/Metal and
rejected by CUDA: a false rejection. All three backends now use the same purity
criterion.

`Reset` in **analytic** mode (`shots = None`) has a well-defined pure-state
result exactly when the target qubit is **unentangled** — both measurement
branches land on `|0⟩ ⊗ rest`, so the outcome being random does not make the
*result* random.

**The tolerance is 1e-6, not the 1e-4 the old fix note proposed.** This trades a
criterion that was over-strict but never *wrong* for one that can be wrong, so
the bound matters: at 1e-4 a purity of 0.9999 is admitted — Schmidt weight
≈ 5e-5, amplitude error ≈ 7e-3 — three to four orders outside this project's own
gates (5e-7 f32, 1e-9 cross-check). Measured f32 purity noise on this device is
~1e-6. `tests/reset_criterion.rs` pins the boundary and is mutation-checked:
loosening 1e-6 → 1e-2 fails the near-boundary case and only that one.

**The remaining limitation is cost, and it is not small.** The criterion needs
the state on the host, so each analytic `Reset` op does a full `read_state()` —
16 B/amp pulled from an 8 B/amp device buffer, so **peak host memory doubles** —
followed by `reduced_purity`, a **serial** host loop over `dim`. This runs *per
Reset op*, not once per circuit. At n = 28 that is ~4 GiB and ~2.7e8 serial
iterations each time. It is confined to analytic mode (the shot path samples on
device and never pays it), but a wide analytic circuit with several resets will
feel it. Specified as `Reset.lean` T1.

### Reset support, audited across every backend (2026-08-05)

The audit behind the entry above. "Refuses" means an explicit
`OmegaError::Unsupported`, never a silent skip or a plausible-looking wrong
number — the failure mode this project keeps finding.

| backend | Reset | verdict |
|---|---|---|
| `statevector` (CPU) | channel (measure → conditional `X`), refuses entangled in analytic mode | **reference** |
| `statevector-metal` | same, via the *same* exported predicate (f32 tolerance) | fixed `a863c82` |
| `statevector-cuda` | channel; refuses on entanglement via the *same* exported predicate as CPU/Metal (1e-6) | aligned 2026-08-17 |
| `statevector-opencl` | refuses: *"Reset is non-unitary; not yet implemented"* | honest gap |
| `pauli` (stabilizer) | tableau reset (measure + conditional `X`); always well-defined | ok |
| `mps` | channel with guard | ok |
| `pauliprop` | refuses: *"cannot be represented by observable conjugation"* | honest gap |
| `mps-cuda`, `mps-metal`, `pauliprop-cuda`, `pauliprop-metal` | no Reset path — these are contraction/branch **hooks**; dispatch stays on the CPU crate | n/a |

All three statevector backends now share one criterion; the CUDA divergence
noted in earlier editions is gone. No backend silently skips Reset or returns a
value for it.

### MPS on Apple GPUs: f64 is impossible there, not merely slow

Two facts worth keeping next to each other (measured / verified 2026-08-20):
the Metal two-site θ-contraction hook, once actually reachable from
`omega-run` (`f3ed353` — it had never been wired), measured **~1.8× slower
than CPU** on the deep χ=256 shape; and Apple's Metal has **no double-precision
type at all**, so an exact-f64 MPS lane on that GPU is not a missing
optimisation but a hardware impossibility — any Metal MPS path is f32 by
construction, which is why the hook is opt-in behind an explicit
`--device metal` and the CPU path stays exact-f64. The performance answer for
deep MPS on Apple hardware is the parallel CPU path (`8781ce3`), not the GPU.

### Metal: shots-mode `Reset` delegates to the CPU backend

Metal's shot path evolves the state **once** and samples the final
distribution — valid for unitary circuits, invalid with `Reset`, which is a
channel whose true result is a mixture over trajectories.

So `MetalStatevectorBackend::execute` delegates to the CPU statevector backend
whenever `shots` is set and the circuit contains a `Reset`. Verified: Bell +
`Reset q0` at 512 shots now returns counts identical to the CPU
(`{0: 262, 2: 250}`) in 0.47 s.

**This is a fallback, not a fix — but the root cause is now identified.**

Per-shot GPU trajectories were implemented first (lease → evolve → sample, reset
branch drawn from an RNG) and are *correct*: verified at 16 shots against the
CPU, same support, no impossible outcomes. They **block at 0% CPU** from a few
hundred shots onward.

Measured 2026-08-05 with a loop of `lease → apply_h → begin_batch → drop`:

| variant | stalls after |
|---|---|
| batch left open | ~32–64 cycles |
| `end_batch_if_open()` before drop | ~32–64 cycles — **no better** |

So it is **not** the open batch, which was the obvious suspect and the first fix
attempted. Both variants stall at the same point, which matches Metal's limit on
**in-flight command buffers** (~64 by default): each iteration leases a state and
issues GPU work *without waiting for completion*, so buffers accumulate until
`commandBuffer()` blocks. `end_batch_if_open` ends the encoder; it does not wait
for the GPU to retire the work.

**The fix**, when this is re-opened: reuse ONE leased state across shots
(re-initialising rather than re-leasing) and force completion each iteration —
any call that reads back, e.g. `pauli_expectation`, waits — instead of leasing a
fresh state per shot. Delegation stands until that is implemented and measured.
