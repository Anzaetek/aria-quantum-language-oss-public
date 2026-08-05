<!-- SPDX-License-Identifier: Apache-2.0 -->
# Qiskit cross-check

Runs the **same** random Clifford circuits through the omega backends and
through Qiskit, and compares exact probabilities. An independent implementation
is the point: agreement between two of our own backends can be a shared
convention, agreement with Qiskit cannot.

Opt-in (needs a venv — never system Python, per the repo rules):

```bash
python3 -m venv .venv-qiskit && ./.venv-qiskit/bin/pip -q install qiskit
ARIA_QISKIT_XCHECK=1 ./ci.sh
```

Measured 2026-08-05, 60 circuits, 2–4 qubits, depth 4–13 over
`{H, S, S†, X, Z, CX}`:

| comparison | worst `\|Δp\|` |
|---|---|
| aria CPU statevector vs **Qiskit** | `4.441e-16` |
| aria stabilizer vs aria CPU | `4.441e-16` (0 unnormalised) |
| aria **Metal** vs aria CPU | `1.857e-7` (f32) |

The stabilizer row is the one that matters historically: before `b2a07be` it
was wrong on ~half of all Clifford circuits in exact-probabilities mode, and
its sampling put shots on zero-probability bitstrings.
