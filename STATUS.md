<!-- SPDX-License-Identifier: Apache-2.0 -->
# Status — backend correctness work, 2026-08-05

Snapshot of what changed, what is verified, and what is knowingly left open.
Every number here was measured; nothing is an estimate.

## Fixed

| # | Defect | Evidence |
|---|---|---|
| 1 | `stabilizer_expectation`: group-membership test was a **greedy pass with no pivoting** that returned `0.0` on failure — a failed algorithm reported as physics, invisible because `⟨P⟩ = 0` is legal | 800 random Clifford circuits, 4 backends, 0 disagreements |
| 2 | `pauli_mult_phase`: `X·Z`/`Z·X` inverted vs Aaronson–Gottesman — **in two copies**, one of which survived the first fix and drove the *measurement* path | 1000/1000 shots on zero-probability bitstrings → 0/1000 |
| 3 | `stabilizer_probability`: fast-reject applied to non-diagonal stabilizers, where its eigenvalue formula is invalid | 1576/3000 circuits unnormalised → 0/3000 |
| 4 | Metal `Reset`: superseded coherent-fold semantics, no entanglement guard, f64 tolerance on an f32 backend | Metal ≡ CPU up to global phase; both refuse entangled analytic |
| 5 | Metal shots-mode `Reset` refused input the CPU accepts (a regression from 4) | Metal ≡ CPU counts at 512 shots, 0.47 s |

## Verified

- **`verification/Verification/Backend/PauliAlgebra.lean`** — 8 theorems, **proved**: no `sorry`, no Mathlib, no `native_decide`. `g_cocycle` explains defect 2's signature (≥3-generator products corrupted, single products intact).
- **Qiskit differential cross-check** (`ARIA_QISKIT_XCHECK=1`): aria CPU vs Qiskit `4.441e-16`; stabilizer vs CPU `4.441e-16`; Metal vs CPU `1.857e-7` (f32).
- **Reset audited across all 11 backends** — no silent skips; every gap is an explicit `Unsupported`.
- `ARIA_METAL=1 ARIA_QISKIT_XCHECK=1 ./ci.sh` green.

## Known open — recorded, not fixed

1. **CUDA's Reset criterion diverges** (refuses on random *outcome*, not entanglement, so it rejects `H q0; Reset q0`). Not fixed: that arm is `cfg`-gated to linux/windows+cuda and cannot be compiled or run on the macOS dev box. A false *rejection*, not a wrong answer. → `LIMITATIONS.md`
2. **Metal per-shot GPU trajectories block at ~64 shots.** Root cause identified: in-flight command-buffer exhaustion, not the open batch. `execute` delegates to CPU meanwhile. → `LIMITATIONS.md`
3. **`Reset.lean` / `StabilizerExpectation.lean`: 7 `sorry` targets.** They need a real ordered field; core Lean has none.
4. **Five distinct Reset acceptance policies** across backends (CPU / Metal / CUDA / mps / pauli). Only the CPU↔Metal pair is conformance-tested. → ledger A6
