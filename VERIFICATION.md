<!-- SPDX-License-Identifier: Apache-2.0 -->
# Numeric verification status of the shipped examples

Every `.aria` program under [`examples/aria/`](examples/aria) is held to a
numeric standard, not "looks right". Three gates apply:

1. **Parse + lower gate** — `cargo test -p aria-core --test aria_examples`
   parses and instantiates *every* file (a one-row table keeps it honest).
2. **Run gate** — `aria-verify all` runs each verified example through the omega
   runtime (wasm guest in-process, or native fallback) and compares a computed
   quantity to a **pure-Rust classical oracle**, printing `Δmax` vs a tolerance
   and a PASS/FAIL the CI asserts on.
3. **Unitarity fuzz** — `cargo test -p aria-runtime --test fuzz_backends`
   cross-checks the statevector vs MPS engines and the norm on random circuits.

There are two kinds of run-gate oracle:

- **classical** — a closed-form / independent classical *algorithm* computes the
  ground truth (DFT, SVD, brute-force max-cut, hidden-string recovery, phase,
  decoded bits, …). The quantum result must match the *answer*.
- **differential** — for parametrized circuits with no single closed-form answer
  (variational ansätze, feature maps, algorithm templates), an **independent
  pure-Rust statevector simulator** ([`aria_verify_core::sim`]) reproduces the
  full `⟨Z_q⟩` profile from the same lowered IR; the runtime must match it
  (Δ ≤ 2.2e-16 in practice). This catches lowering / execution regressions.

## Status — 32 / 33 numerically verified, 1 showcase

| Example | Gate | Oracle | What is checked |
|---|---|---|---|
| bell | run | classical | exact statevector probs == (½,0,0,½) |
| ghz | run | classical | exact statevector probs == (½ at \|000⟩/\|111⟩) |
| superdense | run | classical | decoded 2 bits == Alice's input (bijection), exact |
| simon | run | classical | every sampled y ⊥ hidden period s (fraction 1.0) |
| qpe | run | classical | recovered eigenphase φ̂ == 1/8 |
| qsp | run | classical | zero-phase ⟨Z₀⟩ == cos(d·θ) = T_d (Chebyshev); full phase⇒poly in QSP.lean |
| trotter | run | classical | circuit ⟨Z_q⟩ after first-order Trotter (t=1) == exact exp(-iHt); error→0 as steps→∞ |
| qft | run | classical | output amplitudes == DFT image |
| qsvd | run | classical | recovered singular values == Jacobi SVD |
| grover3 | run | classical | marked-state probability after amplification |
| bernstein_vazirani | run | classical | recovered hidden string == planted a |
| deutsch_jozsa | run | classical | constant-vs-balanced verdict |
| swap_test | run | classical | P(ancilla=0) == ½+½·\|⟨ψ\|φ⟩\|² |
| teleport | run | classical | Bob's state == Alice's input |
| qaoa_maxcut | run | classical | cut value vs brute-force max-cut |
| qml_classifier | run | classical | accuracy vs ground-truth labels |
| qos | run | classical | sketch error scales as O(1/N²) |
| circulant | run | classical | DFT solve == independent Gaussian solve |
| cqs | run | classical | Hadamard-test ⟨Z⟩ vs Pauli expectation |
| noise | run | classical | channel fidelity floor vs analytic CPTP value |
| iqp_born | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| quantum_kernel | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| qcnn | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| qcbm_strongly_entangling | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| qgan | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| qclassifier_rich | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| qssl | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| sketch_qml | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| strongly_entangling | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| qasm_gpu | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| hhl | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| qsvt_invert | run | differential | forward ⟨Z_q⟩ profile vs independent statevector |
| **shor_ecdlp** | **parse only** | **showcase** | parses + instantiates; does **not** lower — see [LIMITATIONS.md](LIMITATIONS.md) |

The faithful correctness of HHL and QSVT inversion is additionally established
in Lean (`proofs/lean4/QuantumProofs/{HHL,QSVT}.lean`) and in the pure-Rust
solvers (`omega_core::{solver, chebyshev}`); the differential run-gate above is
the example-level integration check on top of those proofs.
