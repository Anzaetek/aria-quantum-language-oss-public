<!-- SPDX-License-Identifier: Apache-2.0 -->
# Optional test stages — the full matrix

`./ci.sh` runs 9 core stages plus tch, Metal and OpenCL by default. Several
further stages exist but are **opt-in**, because they need tooling the default
run must not require (K13: headless, external tools skip cleanly).

This file is the durable record of what those stages are, what they need, and
what is set up on the development box — so a stage that has not run recently is
visible rather than quietly forgotten.

Audited 2026-08-06 on Apple Silicon / macOS. `PREREQUISITES.md` covers install
details; this covers *what each stage buys you*.

| stage | enable with | needs | status here | last result |
|---|---|---|---|---|
| Qiskit differential cross-check | `ARIA_QISKIT_XCHECK=1` | `.venv-qiskit` with `qiskit`, `qiskit-aer` | **NOT SET UP** | aria vs Qiskit `4.441e-16`; stabilizer vs CPU `4.441e-16`; Metal vs CPU `1.857e-7` (f32) — recorded in `STATUS.md`, not re-run this session |
| QEC encoded-demo cross-check | `ARIA_QEC_XCHECK=1` | same venv | **NOT SET UP** | — |
| Lean 4 proof tree | `ARIA_LEAN=1` | `elan` + `lake`, warm mathlib cache | **READY** | 8281 jobs, exit 0 |
| TLA+ models | `tools/tla/check.sh` | JDK + `tla2tools.jar` | **READY** | safety holds (26 states); liveness violated — starvation, *expected* |
| CUDA GPU backends | `ARIA_CUDA=1` | NVIDIA hardware | **N/A here** | untested on this box |
| Deep harnesses | `ARIA_DEEP=1` | none (just slow) | available | `spectra_noise`, `spectra_scaling_noise` skipped in the default `all` |

## Why each one is worth running

**Qiskit cross-check** — the only *independent implementation* available. When
Aria and Qiskit agree to 1e-16, that is evidence; when two Aria backends agree,
it may only mean they share a convention. This project has already been bitten
by exactly that: the `Reset` defect survived every cross-backend "agreement"
gate because each pair happened to coincide in the basis being checked. This is
the highest-value optional stage, and the only one needing an install.

**Lean 4** — enforces sorry-free on the shipped circulant, noise-channel and
quantum-linear-algebra theorems (via `#print axioms` greps for `sorryAx`).
`lake build` alone does *not* error on `sorry`, so the axiom check is the real
gate. Known-open: `Reset.lean` and `StabilizerExpectation.lean` carry 7 `sorry`
targets — they need a real ordered field, which core Lean lacks. Those are
recorded, not regressions.

**TLA+** — properties that hold over *all* interleavings rather than one run.
Already earned its keep: `Governor.tla` produced a concrete lasso proving a
large job can be starved indefinitely under `try_acquire` with no queue, which
is the design input for the bounded queue. Note the liveness check is **expected
to fail** until that queue exists; see `proofs/tla/README.md`.

**CUDA** — the only way to exercise the CUDA statevector, MPS `gesvdj`,
pauliprop, RBS, and the on-device Born sampling for the Reset channel. Also the
only way to exercise A7b's device pools and f32 pricing against real discrete
hardware; so far they have only met the OpenCL path.

**Deep harnesses** — `ARIA_DEEP=1` adds the two spectra-noise apps the default
`all` skips for runtime. Worth running before a release.

## Standing caveat

Every one of these is a *differential* or *formal* check, and neither kind can
tell you an assumption is wrong. A model verifies a protocol against the weights
it is handed; a cross-check verifies agreement between two implementations of
the same idea. The governor defect that mattered most this year — pricing MPS by
its tensors while the backend contracted to a dense statevector — would have
passed both. That class stays closed by reading what the code actually
allocates. See `proofs/tla/README.md` for the longer version.
