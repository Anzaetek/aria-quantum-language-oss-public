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
  - ⚠️ **The Metal statevector backend still carries the old, incorrect
    implementation.** CPU, MPS, stabilizer and CUDA implement the channel; the
    Metal mirror has not been ported because it cannot be compiled or run off
    a Mac. Its `reset_matches_cpu` gate will fail under `ARIA_METAL=1` until
    the port lands on the Mac box — treat Metal `Reset` results as wrong until
    then. See `GPU_BACKEND_PLAN.md`.

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

## Reset: the CUDA backend is stricter than CPU/Metal

`Reset` in **analytic** mode (`shots = None`) has a well-defined pure-state
result exactly when the target qubit is **unentangled** — both measurement
branches then land on `|0⟩ ⊗ rest`, so the outcome being random does not make
the *result* random.

The CPU and Metal backends use that criterion (reduced purity = 1). The CUDA
backend instead refuses whenever `p0 ∈ (0, 1)`, i.e. whenever the *outcome* is
random. The two agree on entangled qubits (all refuse) and on Z-eigenstates
(all allow), but differ on an unentangled superposition:

| state of `q` | purity | `p0` | CPU / Metal | CUDA |
|---|---|---|---|---|
| `\|0⟩`, `\|1⟩` | 1 | 1 or 0 | allow | allow |
| `\|+⟩`, `\|−⟩` unentangled | 1 | 0.5 | **allow** | **refuse** |
| Bell | 0.5 | 0.5 | refuse | refuse |

So `H q0; Reset q0` is accepted by CPU and Metal and rejected by CUDA. This is a
**false rejection** — a valid circuit errors out — not a wrong answer, so it is
recorded rather than hot-fixed.

**Not fixed here, deliberately.** The CUDA arm is `cfg`-gated to
`linux/windows + cuda` and cannot be compiled or executed on the macOS dev box,
so any change would be unverifiable. The fix is to adopt the purity criterion
(`omega_backend_statevector::sim::reset_is_deterministic_within(..., 1e-4)`),
which needs a dependency on the CPU crate plus a device readback, and must be
validated on a CUDA box. Specified as `Reset.lean` T1.

### Reset support, audited across every backend (2026-08-05)

The audit behind the entry above. "Refuses" means an explicit
`OmegaError::Unsupported`, never a silent skip or a plausible-looking wrong
number — the failure mode this project keeps finding.

| backend | Reset | verdict |
|---|---|---|
| `statevector` (CPU) | channel (measure → conditional `X`), refuses entangled in analytic mode | **reference** |
| `statevector-metal` | same, via the *same* exported predicate (f32 tolerance) | fixed `a863c82` |
| `statevector-cuda` | channel, but refuses on `p0 ∈ (0,1)` rather than on entanglement | stricter — see above |
| `statevector-opencl` | refuses: *"Reset is non-unitary; not yet implemented"* | honest gap |
| `pauli` (stabilizer) | tableau reset (measure + conditional `X`); always well-defined | ok |
| `mps` | channel with guard | ok |
| `pauliprop` | refuses: *"cannot be represented by observable conjugation"* | honest gap |
| `mps-cuda`, `mps-metal`, `pauliprop-cuda`, `pauliprop-metal` | no Reset path — these are contraction/branch **hooks**; dispatch stays on the CPU crate | n/a |

Only the CUDA criterion diverges, and it errs toward refusing. No backend
silently skips Reset or returns a value for it.

### Metal: shots-mode `Reset` delegates to the CPU backend

Metal's shot path evolves the state **once** and samples the final
distribution — valid for unitary circuits, invalid with `Reset`, which is a
channel whose true result is a mixture over trajectories.

So `MetalStatevectorBackend::execute` delegates to the CPU statevector backend
whenever `shots` is set and the circuit contains a `Reset`. Verified: Bell +
`Reset q0` at 512 shots now returns counts identical to the CPU
(`{0: 262, 2: 250}`) in 0.47 s.

**This is a fallback, not a fix.** Per-shot GPU trajectories were implemented
first (lease → evolve → sample, reset branch drawn from an RNG) and are
*correct* — verified at 16 shots against the CPU, same support, no impossible
outcomes — but they **block at 0% CPU** from a few hundred shots onward, and
draining the batch `apply_ops_fused` leaves open after a Reset did not resolve
it. The root cause is not yet identified, and shipping a hang would be worse
than the bug it replaces. Re-open when the pool/command-buffer interaction is
understood.
