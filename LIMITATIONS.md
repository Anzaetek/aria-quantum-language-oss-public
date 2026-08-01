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
