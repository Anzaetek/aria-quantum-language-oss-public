# qec_cross_check

Cross-checks aria-qec against independent references on two fronts:

1. **Encoded-demo algorithms** (`crates/apps/qec-*`) vs **Qiskit** (`check_qec.py`).
2. **The surface-code decoder** (`ecc/mwpm.rs`) vs **PyMatching**, the
   field-standard MWPM decoder (`check_decoder.py`).

## 1. Encoded algorithms vs Qiskit (`check_qec.py`)

The `qec-*` harnesses run key algorithms on transversally QEC-encoded logical
qubits and assert the encoded result equals the ideal logical distribution; this
tool closes the loop from the outside.

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

## 2. Surface-code decoder vs PyMatching (`check_decoder.py`)

`qec-memory` is a code-capacity Monte-Carlo whose engine is aria-qec's own exact
minimum-weight decoder (`ecc/mwpm.rs::decode_mwpm_correction`, bounded
enumeration). That decoder is unit-tested internally (every single-qubit error
corrected); this cross-checks it against **PyMatching** — the MWPM decoder the
QEC community pairs with stim — on the *identical* error samples.

The Rust `dump_surface_code` example emits the rotated surface code (check
matrices + logical operators) and a seeded batch of decoded error trials; the
script re-decodes the same samples with PyMatching and, for d ∈ {3, 5} and both
CSS sectors, asserts:

- every error of weight ≤ ⌊(d−1)/2⌋ is correctable (guaranteed-distance);
- **shot-for-shot logical-class agreement ≥ 99%** (both are minimum-weight, so
  they can differ only on exact weight-ties) — in practice **100.00%** at p=0.05;
- logical-error-rate agreement `aria ≈ PyMatching`, |Δ| ≤ 3σ (matches to ~1e-4).

Both decoders independently reproduce the distance suppression `pL(d=5) <
pL(d=3)` that the `qec-memory` demo asserts.

## Deferred

A `PauliPropagation.jl` (Julia) path could additionally cross-check the
biased-Pauli *logical channel* of `qec-memory` if/when it grows the matching
per-qubit neutral-atom noise model. Not required: the decoder itself is now
validated against PyMatching above.
