# noise_qiskit_check

Cross-checks Aria's `--noise` implementation against **Qiskit Aer** as ground
truth. Aer's `density_matrix` method (exact) provides the reference
distribution / expectation; the same case is run through the Aria CLIs
(`aria`, `omega-run`) and the two are asserted equal.

## Run

```sh
tools/noise_qiskit_check/run.sh
```

`run.sh` builds the Aria CLIs and runs the checker under a Python that has
`qiskit` + `qiskit-aer`. It defaults to a sibling venv that already has a
compiled Aer; override with `AER_PYTHON=/path/to/venv/bin/python` (any venv
where `pip install qiskit qiskit-aer` has run).

## What it covers (the bugs this validates)

| Case | Pins |
|------|------|
| `readout_flip` on `X;measure` | `--noise` is applied, not silently noiseless |
| `amplitude_damping` on `\|1⟩` | trajectory sampler, per-shot (not one replayed branch) |
| `depolarizing` | the `p ↔ 4p/3` eigenvalue convention vs Aer |
| **depol + amplitude_damping** | the pauliprop amp-damp **adjoint-ordering** bug — `⟨Z0⟩` must equal Aer's `0.2` |
| per-qubit `amplitude_damping [0, 0.8]` | heterogeneous per-qubit rates |
| asymmetric `{p10, p01}` readout | per-qubit asymmetric readout confusion matrix |

## Semantic conversions (Aria ↔ Aer)

Two conventions must be translated for the two engines to agree exactly:

- **Depolarizing.** Aria's `depolarizing: p` means "with probability `p` apply a
  uniformly chosen X/Y/Z" (Pauli eigenvalue `1 − 4p/3`). Aer's
  `depolarizing_error(λ)` has eigenvalue `1 − λ`, so the equivalent is
  `depolarizing_error(4p/3)`.
- **Readout.** Aria's `readout_flip: p` / `{p10, p01}` maps to Aer's confusion
  matrix `[[1−p10, p10], [p01, 1−p01]]` (row = true bit, column = reported bit).

The combined-channel case additionally self-checks the Aer reference against the
hand-derived analytic value (`⟨Z⟩ = γ + (1−γ)·(−(1−4p/3)) = 0.2`) before
comparing Aria to it, so a mistake in the reference construction fails loudly
rather than masking an Aria bug.
