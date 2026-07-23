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

## Status — 48 harnesses numerically verified, 1 parse-only showcase

One row per `aria-verify` harness (`cargo run -p aria-verify -- all` asserts
every PASS); the four `arch_*`/`spectra_scaling` rows are search/scaling
harnesses that reuse `spectra_heisenberg.aria` rather than shipping their own
`.aria` file. Of the 43 `.aria` example files, 42 are run-verified and
`shor_ecdlp` is the parse-only showcase.

| Example | Gate | Oracle | What is checked |
|---|---|---|---|
| bell | run | classical | exact statevector probs == (½,0,0,½) |
| ghz | run | classical | exact statevector probs == (½ at \|000⟩/\|111⟩) |
| superdense | run | classical | decoded 2 bits == Alice's input (bijection), exact |
| simon | run | classical | every sampled y ⊥ hidden period s (fraction 1.0) |
| qpe | run | classical | recovered eigenphase φ̂ == 1/8 |
| qsp | run | classical | zero-phase ⟨Z₀⟩ == cos(d·θ) = T_d (Chebyshev); full phase⇒poly in QSP.lean |
| trotter | run | classical | circuit ⟨Z_q⟩ after first-order Trotter (t=1) == exact exp(-iHt); error→0 as steps→∞ |
| qdrift | run | classical | circuit ⟨Z_q⟩ after QDrift (uniform λτ, freq ∝ |c_j|) == exact exp(-iHt); error→0 as N→∞ |
| taylor_lcu | run | classical | ancilla=|0⟩ block of the LCU == exp(-iXt)|0⟩ (closed form); success prob == 1/λ² |
| shor | run | classical | counting-register peaks {0,4} ⇒ period r=2 ⇒ gcd(11^{r/2}±1,15) = {3,5} factors N=15 |
| schrodingerize | run | classical | one warped-phase transport step: recovery Σ_{p≥0} w == e^{-a·Δp} (exact) |
| qft | run | classical | output amplitudes == DFT image |
| qsvd | run | classical | recovered singular values == Jacobi SVD |
| vqe_ansatz | run | classical | hardware-efficient ansatz minimizes ⟨H₂⟩ to the exact ground energy E₀ |
| grover3 | run | classical | marked-state probability after amplification |
| bernstein_vazirani | run | classical | recovered hidden string == planted a |
| deutsch_jozsa | run | classical | constant-vs-balanced verdict |
| swap_test | run | classical | P(ancilla=0) == ½+½·\|⟨ψ\|φ⟩\|² |
| teleport | run | classical | Bob's state == Alice's input |
| qaoa_maxcut | run | classical | cut value vs brute-force max-cut |
| qml_classifier | run | classical | accuracy vs ground-truth labels |
| butterfly_qnn | run | classical | parallel commuting-block gradients (arXiv:2606.03517) == serial 4-term Givens shifts, \|Δ\| ≤ 1e-9; imputation MSE ≤ mean-imputer on UCI heart (open data, 30% MCAR) |
| jl_sketch_digits | run | differential | forward ⟨Z_q⟩ profile vs independent statevector; optdigits 3-vs-8 accuracy ≥ 0.85 on quantum features (open data) |
| spectra | run | classical | SPECTRA certificate (arXiv:2607.15815): heart + planted-term pocket REFUSED (order-matched classical panel wins), quantum-generated Heisenberg substrate CERTIFIED (bootstrap CI_lo > 0 + ablation gate); probe ⟨Z_q⟩ vs independent oracle ≤ 1e-9 |
| qos | run | classical | sketch error scales as O(1/N²) |
| circulant | run | classical | DFT solve == independent Gaussian solve |
| cqs | run | classical | Hadamard-test ⟨Z⟩ vs Pauli expectation |
| noise | run | classical | channel fidelity floor vs analytic CPTP value |
| qec_grover | run | classical | encoded 2-qubit Grover on Steane [[7,1,3]] finds the marked state exactly |
| qec_qft | run | classical | logical-channel QFT satisfies the defining amplitude identities |
| qec_qpe | run | classical | logical-channel QPE recovers the planted eigenphase exactly |
| qec_memory | run | classical | surface-code memory Monte-Carlo matches seeded code-capacity statistics |
| arch_search | run | classical | coupling-graph search selects the generator's chain; builder ≡ lowered spectra_heisenberg.aria; learned J within 0.2 of the true draw |
| arch_evolve | run | classical | evolved RBS-mask genome ≤ hand-built butterfly MSE + 0.02, ≥ no-entanglement ablation; bit-exact re-evaluation |
| arch_priors | run | classical | classical periodogram prior within 0.25 of the planted frequencies; prior-init QNN ≥ flat-init + 0.03 AUC |
| spectra_scaling | run | classical | \|+⟩^⊗n invariant ≤ 1e-9 at n = 7…13; DMQ-vs-classical AUC gap ≥ 0.15 at every gap-checked size |
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
