#!/usr/bin/env python3
"""Emit monoprop reference expectation values for `omega-backend-pauliprop`.

`Algorithmiq/monoprop` (Apache-2.0, C++ core, Python front-end) is an
INDEPENDENT Pauli-propagation implementation — the same algorithm family as
our backend, written by people who have never seen our code. Two of our own
backends agreeing may only mean they share a convention; agreement with
monoprop is evidence about the *algorithm*, in the same sense as the piquasso
cross-check for CV and the Qiskit cross-check for statevector.

## Shape of the output

One JSON object per line. A leading meta line stamps the versions, so a
regenerated fixture is attributable to what produced it:

    {"meta": {"monoprop": "0.9.0", "numpy": "2.x"}}
    {"case": ..., "n_qubits": 2, "ops": [...], "observable": [...],
     "cutoff": 2, "lower_atol": null, "expectation": f}

`ops` is the circuit as **structured data**, not QASM text and not prose:

    {"op": "rx",  "q": [0],    "theta": 0.3}
    {"op": "rzz", "q": [0, 1], "theta": 0.7}

Each side interprets `ops` for itself. Here it becomes monoprop `ExpGate`s;
in `crates/omega-backend-pauliprop/tests/monoprop_xcheck.rs` it is rendered to
OpenQASM 2 and goes through `omega-parser`'s lowering. Neither side reads the
other's rendering, so a convention shared by accident would have to be shared
by *both* renderers independently — the failure this file exists to prevent.

`observable` is a list of `[coeff, "PAULI_LETTERS", [qubits]]` terms.

## The sign convention, verified rather than assumed

monoprop's `ExpGate` applies e^{+iθH}. OpenQASM's `rx(θ)` (Qiskit's, and
ours — pinned by the N-way and Qiskit cross-checks) is e^{-iθX/2}. So every
`r<P>(θ)` becomes `ExpGate(P)` driven at **−θ/2**. Checked live before this
file was written: `rx(0.3)` → ⟨Z₀⟩ = 0.955336489125606 = cos 0.3, and
`ry(0.4) q0; rxx(0.9) q0,q1` → ⟨Z₀⟩ = 0.57254069525748 = cos 0.9 · cos 0.4.

## Truncation knobs, mapped

* monoprop `cutoff` (int) = maximum Pauli weight retained  ↔ our `max_weight`.
  `cutoff == n_qubits` means no weight truncation at all.
* monoprop `lower_atol` = discard |coeff| < atol during evolution ↔ our
  `coeff_min`. `null` means none.

Only `rx ry rz rxx rzz` appear. `ryy` is deliberately absent: our default
QASM2 dialect (Qiskit's legacy `from_qasm_str` table) rejects a bare `ryy`,
and a case that cannot be lowered on our side proves nothing.

Exit 0 on success. On failure emits one `{"error": ...}` line and exits 1, so
a broken oracle is never mistaken for an empty fixture.
"""
import json
import sys

import numpy as np

try:
    import monoprop as mp
except ImportError as exc:  # pragma: no cover
    print(json.dumps({"error": f"monoprop not importable: {exc}"}))
    sys.exit(1)

LETTERS = {"rx": "X", "ry": "Y", "rz": "Z", "rxx": "XX", "rzz": "ZZ"}


def hea_layer(n, angles, offset):
    """One hardware-efficient layer: single-qubit rx/ry/rz on every qubit,
    then a ring of rzz and one long-range rxx. `angles` is consumed in order
    so the recipe is fully determined by the list literal below."""
    ops = []
    k = offset
    for q in range(n):
        for name in ("rx", "ry", "rz"):
            ops.append({"op": name, "q": [q], "theta": angles[k]})
            k += 1
    for q in range(n - 1):
        ops.append({"op": "rzz", "q": [q, q + 1], "theta": angles[k]})
        k += 1
    ops.append({"op": "rxx", "q": [0, n - 1], "theta": angles[k]})
    k += 1
    return ops, k


# Fixed, hand-written angles. Not random: the recipe must be reproducible
# from the file alone, and a seed would make the *generator* the source of
# truth instead of the committed fixture.
ANGLES = [
    0.31, 1.07, -0.58, 0.92, -1.31, 0.44, 0.77, -0.19, 1.53, -0.66, 0.25,
    1.12, -0.83, 0.61, 0.38, -1.02, 0.49, 1.27, -0.35, 0.88, -0.71, 1.44,
    0.16, -0.94, 0.57, 1.19, -0.42, 0.73, -1.15, 0.29, 0.96, -0.53, 1.38,
    0.11, -0.87, 0.64, 1.01, -0.26, 0.82, -1.23, 0.47, 1.31, -0.09, 0.69,
]


def hea(n, layers):
    ops, k = [], 0
    for _ in range(layers):
        layer, k = hea_layer(n, ANGLES, k)
        ops += layer
    return ops


CASES = [
    # name, n, ops, observable, cutoff, lower_atol
    ("rx_z0", 1, [{"op": "rx", "q": [0], "theta": 0.3}], [[1.0, "Z", [0]]], 1, None),
    ("ry_rxx_z0", 2,
     [{"op": "ry", "q": [0], "theta": 0.4}, {"op": "rxx", "q": [0, 1], "theta": 0.9}],
     [[1.0, "Z", [0]]], 2, None),
    ("rz_rzz_x1", 2,
     [{"op": "rx", "q": [1], "theta": 1.1}, {"op": "rz", "q": [1], "theta": 0.6},
      {"op": "rzz", "q": [0, 1], "theta": 0.7}, {"op": "ry", "q": [0], "theta": -0.5}],
     [[1.0, "X", [1]]], 2, None),
    ("hea3_l2_zz01_exact", 3, hea(3, 2), [[1.0, "ZZ", [0, 1]]], 3, None),
    ("hea4_l2_x2_exact", 4, hea(4, 2), [[1.0, "X", [2]]], 4, None),
    ("hea4_l2_mixed_exact", 4, hea(4, 2),
     [[0.5, "Z", [0]], [-0.25, "XX", [1, 3]], [1.0, "YZ", [2, 3]]], 4, None),
    # Truncated cases. Chosen from a knob sweep on OUR side so that the
    # dropped-mass certificate is informative (well under the observable's L1
    # norm, where the backend refuses as vacuous). Aggressive knobs on this
    # circuit — weight cap 2, or atol 0.05 — push our dropped mass past 1.0 and
    # would prove nothing; see the Rust test header for why the two engines
    # truncate differently at equal knobs.
    ("hea4_l2_x2_atol005", 4, hea(4, 2), [[1.0, "X", [2]]], 4, 0.005),
    ("hea4_l2_zz12_atol01", 4, hea(4, 2), [[1.0, "ZZ", [1, 2]]], 4, 0.01),
    ("hea4_l2_zz12_w3", 4, hea(4, 2), [[1.0, "ZZ", [1, 2]]], 3, None),
    ("hea4_l2_zz12_w3_atol01", 4, hea(4, 2), [[1.0, "ZZ", [1, 2]]], 3, 0.01),
    ("hea4_l2_z0_atol005", 4, hea(4, 2), [[1.0, "Z", [0]]], 4, 0.005),
]


def run_case(n, ops, observable, cutoff, lower_atol):
    gates, params = [], []
    for i, op in enumerate(ops):
        gen = mp.PauliOperator({mp.Pauli(LETTERS[op["op"]], op["q"]): 1.0}, n)
        gates.append(mp.ExpGate(gen, i))
        params.append(-op["theta"] / 2.0)  # e^{+iθH} vs e^{-iθP/2}: see docstring
    circuit = mp.Circuit(gates, n, parameters=params)
    obs = mp.PauliOperator({mp.Pauli(s, q): c for c, s, q in observable}, n)
    kwargs = {"cutoff": cutoff}
    if lower_atol is not None:
        kwargs["lower_atol"] = lower_atol
    pp = mp.PauliPropagator(obs, [], **kwargs)  # [] = |0…0⟩ reference
    pp.propagate(circuit)
    return float(pp.expectation_value())


def main():
    try:
        print(json.dumps({"meta": {"monoprop": mp.__version__, "numpy": np.__version__}}))
        for name, n, ops, observable, cutoff, lower_atol in CASES:
            value = run_case(n, ops, observable, cutoff, lower_atol)
            print(json.dumps({
                "case": name, "n_qubits": n, "ops": ops, "observable": observable,
                "cutoff": cutoff, "lower_atol": lower_atol, "expectation": value,
            }))
    except Exception as exc:  # noqa: BLE001 — surfaced as data, on purpose
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
