# Lean 4 proof tree (`QuantumProofs`)

**License.** All files in this directory are licensed under **Apache-2.0**, the
same as the rest of this repository (see the top-level [`LICENSE`](../../LICENSE)
and [`NOTICE`](../../NOTICE)). They carry no per-file SPDX header to stay
byte-identical with the first-party upstream proof sources they are synced from;
the repository `LICENSE` is the authoritative grant.

**Provenance.** This is the minimal dependency closure (ported first-party from
the upstream toolkit) that makes `aria export <file> --circuit NAME --lean`
self-contained: the emitted Lean imports `QuantumProofs.CircuitSemantics` /
`QuantumProofs.Gates`, which build against this tree. The `QuantumProofs`
namespace is retained deliberately so exported files resolve unchanged.

**Contents.**

| file | what it proves |
|---|---|
| `QuantumProofs/Basic.lean`, `Gates.lean`, `CircuitSemantics.lean` | state space, gate matrices, circuit denotational semantics |
| `QuantumProofs/QFT.lean` | `qft_correct : denote (qft_circuit n) = dft_matrix n` |
| `QuantumProofs/BellPrep.lean` | gate-model Bell state-prep + unitarity |
| `QuantumProofs/GHZPrep.lean` | gate-model GHZ (3-qubit) state-prep (2nd `--gate-model`-recognized circuit) |
| `QuantumProofs/CirculantSolve.lean` | circulant `quantum ≡ classical` diagonalization (`n=1`), solve operator `Fᴴ·Λ⁻¹·F = C⁻¹`, and the **noisy-solve deviation** theorem |
| `QuantumProofs/CirculantSolveGeneral.lean` | **general `n`:** the QFT diagonalizes any `Matrix.circulant v` to `diagonal (circEigen v)` (the DFT of its first column) — `dft_diagonalizes_circulant` / `qft_diagonalizes_circulant`, with the `n=1` case subsumed |
| `QuantumProofs/Generated/GateModel/Bell_Spec.lean` | the `aria export --gate-model` artefact, closed sorry-free |

**Build (opt-in — needs a mathlib cache).**

```sh
cd proofs/lean4
lake exe cache get          # fetch the pinned mathlib oleans (multi-GB)
lake build                  # build the whole tree
```

Or, from the repo root, `ARIA_LEAN=1 ./ci.sh` builds the tree **and** runs the
`#print axioms` sorry-free gate. The default `./ci.sh` does *not* build Lean
(it would require the mathlib cache); it runs only the Rust-side string/emitter
unit tests. The "sorry-free / axiom-clean" guarantee is machine-enforced under
the opt-in `ARIA_LEAN=1` path, not by the default Rust-only CI run.
