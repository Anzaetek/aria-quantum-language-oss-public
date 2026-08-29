# SPDX-License-Identifier: Apache-2.0
"""Measure the wire-payload reduction of the template expectation shape.

`PLAN-SIX-PROGRAMMES.md` P5 requires "a measured payload reduction on a realistic
ansatz" — so this is a script that produces the number rather than a number
written into a document, which decays silently.

The batch route (`circuits: [...]`) re-sends the ENTIRE gate list per row. For a
QML sweep that is the same ansatz N times with different angles baked in, so the
payload grows with `N x gates` while the information content grows with
`N x params`. On a realistic ansatz `params << gates`.

Run: `python3 tools/template_payload/measure.py`
"""
import json


def ansatz_ops(n, layers, *, symbolic, row=0):
    """Ry layer + CX ring, `layers` deep — the shape a hardware-efficient QML
    ansatz actually has."""
    ops = []
    for l in range(layers):
        for q in range(n):
            # `OmegaParam` is `#[serde(untagged)]`: a bare number, or
            # `{"symbol": name}`. The first draft of this script used the
            # externally-tagged spelling and so measured a payload the server
            # would have REFUSED with a 422 — the ratios were close but the
            # absolute bytes were of a wire format that does not exist.
            if symbolic:
                param = {"symbol": f"t{l}_{q}"}
            else:
                param = 0.1234567 + q * 0.01 + l * 0.02 + row * 0.001
            ops.append({"gate": "Ry", "qubits": [q], "params": [param],
                        "classical_bit": None, "condition": None})
        for q in range(n):
            ops.append({"gate": "CX", "qubits": [q, (q + 1) % n], "params": [],
                        "classical_bit": None, "condition": None})
    return ops


def circuit(n, ops):
    return {"num_qubits": n, "num_classical_bits": 0, "is_photonic": False,
            "mid_circuit_mode": "Skip", "backend": "Statevector", "ops": ops}


def batch_bytes(n, layers, rows):
    cs = [circuit(n, ansatz_ops(n, layers, symbolic=False, row=r)) for r in range(rows)]
    return len(json.dumps({"circuits": cs, "observable": "Z0"}))


def template_bytes(n, layers, rows):
    P = n * layers
    return len(json.dumps({
        "circuit": circuit(n, ansatz_ops(n, layers, symbolic=True)),
        "observable": "Z0",
        "params": [f"t{l}_{q}" for l in range(layers) for q in range(n)],
        "rows": [[0.1234567 + j * 0.001 + i * 0.01 for j in range(P)]
                 for i in range(rows)],
    }))


def main():
    print(f"{'qubits':>7} {'layers':>7} {'rows':>6} {'batch B':>12} "
          f"{'template B':>12} {'reduction':>10}")
    for n, layers, rows in [(8, 3, 1), (8, 3, 32), (8, 3, 128), (12, 4, 64), (16, 5, 256)]:
        b, t = batch_bytes(n, layers, rows), template_bytes(n, layers, rows)
        print(f"{n:>7} {layers:>7} {rows:>6} {b:>12,} {t:>12,} {b / t:>9.1f}x")
    print()
    print("Note the rows=1 line. At N=1 the template SAVES NOTHING — it is a")
    print("slightly different encoding of the same information, and can even be")
    print("larger because symbol names cost more than the floats they replace.")
    print("The shape is a batch optimisation, and any measurement of it taken at")
    print("N=1 would report that it does not work.")


if __name__ == "__main__":
    main()
