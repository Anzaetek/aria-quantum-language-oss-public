# qec_cross_check

Cross-checks the **aria-qec encoded-demo algorithms** (`crates/apps/qec-*`)
against **Qiskit** as an independent reference. The `qec-*` harnesses run key
algorithms on transversally QEC-encoded logical qubits and assert the encoded
result equals the ideal logical distribution; this tool closes the loop from the
outside.

For each demo it takes the *exact same* logical circuit aria emits
(`aria export --qasm` on the `examples/aria/qec_*.aria` twins), runs it through an
independent SDK (Qiskit `Statevector`, exact; plus a **stim** stabilizer tableau
for the Clifford case), and asserts three-way agreement: **aria == Qiskit ==
analytic golden**, within `1e-9`. A shared bug would have to fool two independent
simulators and the closed-form answer simultaneously.

## Run

```sh
tools/qec_cross_check/run.sh
```

Zero-config: if `QEC_PYTHON` is unset and no venv exists, `run.sh` creates
`tools/qec_cross_check/.venv` (git-ignored) and `pip install`s `qiskit` (and,
best-effort, `stim`/`qsimcirq`). Override with `QEC_PYTHON=/path/to/venv/bin/python`
(any venv where `pip install qiskit` has run). Per repo policy Python only ever
runs in a venv, never system Python.

Opt-in in local CI: `ARIA_QEC_XCHECK=1 ./ci.sh` runs this as an optional stage
(default CI stays green without qiskit, like the GPU/Lean stages).

## What it covers

| Demo | Independent check | Golden |
|------|-------------------|--------|
| `qec-grover` | Qiskit `Statevector` + **stim** stabilizer tableau, marked ∈ {0,1,2,3} | argmax == marked, `P(marked) == 1` (pure Clifford, exact) |
| `qec-qft` | Qiskit `Statevector` — aria's exported `QFT` and `IQFT` circuits | `QFT\|0000⟩` uniform; `QFT∘QFT⁻¹\|x⟩ = \|x⟩` for x ∈ {0,5,11,15} |
| `qec-qpe` | Qiskit `Statevector` — aria's exported `QPEDemo(m=3)`, φ = 3/8 | counting register → `\|011⟩ = 3` with `P = 1` |

The exported QASM is parsed by a tiny in-file OpenQASM-2 reader over the demo
gate set `{h, x, cz, cp(θ), swap}` (transparent and dependency-free, so the
reference construction can't silently diverge from aria's circuit). aria's own
`--statevector` output is parsed and compared against Qiskit in the same run, so
the tool pins **aria == Qiskit** directly, not just Qiskit == golden.

`stim` and `qsimcirq` are optional: their checks are skipped cleanly if the
package is absent (Qiskit `Statevector` alone is sufficient and exact).

## Not covered here (and why)

`qec-memory` (surface-code logical memory under neutral-atom noise) is a
**code-capacity Monte-Carlo with MWPM decoding**. Its logical-error rate is
validated internally by the distance-suppression signature `pL(d=5) < pL(d=3)`
and the crate's unit tests; an exact statevector cross-check would require
re-implementing the decoder, so it is out of scope here. A future
`PauliPropagation.jl` path (Julia) can cross-check the biased-Pauli logical
channel if/when it grows the matching per-qubit noise model.
