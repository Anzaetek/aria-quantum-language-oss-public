# SPDX-License-Identifier: Apache-2.0
"""Measured benchmark: aria_py vs PennyLane on the same circuit.

Compares batched forward (⟨Z0⟩ over N parameter rows) and batched gradient
(∂⟨Z0⟩/∂θ over N rows) wall-clock. The point is a *measurement*, not a slogan —
run it and read the numbers on your machine.

    python bench/bench_pennylane.py [--qubits 6] [--layers 3] [--rows 256]

Requires the `bench` extra: pip install '.[bench]'  (pennylane).
"""
import argparse
import time

import numpy as np


def build_aria(nq, layers):
    import aria_py

    lines = [f"circuit B() {{ qreg q[{nq}]",
             f"  let theta = symbolic[{nq * layers}]",
             f"  let x = symbolic[{nq}]"]
    k = 0
    for _ in range(layers):
        for i in range(nq):
            lines.append(f"  apply RY(x[{i}]) on q[{i}]")
        for i in range(nq):
            lines.append(f"  apply RZ(theta[{k}]) on q[{i}]"); k += 1
        for i in range(nq - 1):
            lines.append(f"  apply CX on q[{i}], q[{i + 1}]")
    lines.append("}")
    return aria_py.load_source("\n".join(lines), "B")


def bench_aria(m, rows, reps):
    obs = "Z0"
    t = time.perf_counter()
    for _ in range(reps):
        m.expectation_batch(rows, obs)
    fwd = (time.perf_counter() - t) / reps
    t = time.perf_counter()
    for _ in range(reps):
        m.gradient_batch(rows, obs)
    grad = (time.perf_counter() - t) / reps
    return fwd, grad


def build_pennylane(nq, layers, device="default.qubit", diff_method="backprop"):
    import pennylane as qml

    dev = qml.device(device, wires=nq)

    @qml.qnode(dev, diff_method=diff_method, interface="autograd")
    def circuit(x, theta):
        k = 0
        for _ in range(layers):
            for i in range(nq):
                qml.RY(x[i], wires=i)
            for i in range(nq):
                qml.RZ(theta[k], wires=i); k += 1
            for i in range(nq - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    return circuit


def bench_pennylane(circuit, rows, nq, layers, reps):
    import pennylane as qml
    from pennylane import numpy as pnp

    n_theta = nq * layers
    theta = pnp.array(rows[0][:n_theta], requires_grad=True)  # dummy; per-row below

    def one_row(row):
        x = pnp.array(row[n_theta:], requires_grad=False)
        th = pnp.array(row[:n_theta], requires_grad=True)
        return x, th

    t = time.perf_counter()
    for _ in range(reps):
        for row in rows:
            x, th = one_row(row)
            circuit(x, th)
    fwd = (time.perf_counter() - t) / reps
    t = time.perf_counter()
    for _ in range(reps):
        for row in rows:
            x, th = one_row(row)
            qml.grad(circuit, argnums=1)(x, th)
    grad = (time.perf_counter() - t) / reps
    return fwd, grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    nq, layers, N = args.qubits, args.layers, args.rows
    n_sym = nq * layers + nq
    rng = np.random.default_rng(0)
    # Aria param rows: [theta.. , x..] must match ascending SymbolId order.
    m = build_aria(nq, layers)
    # ascending order is theta_0.. then x_0.. (theta declared first in source)
    rows = rng.standard_normal((N, n_sym)).tolist()

    print(f"circuit: {nq} qubits, {layers} layers, {n_sym} params; N={N} rows, reps={args.reps}")
    a_fwd, a_grad = bench_aria(m, rows, args.reps)
    print(f"  aria_py                 forward {a_fwd*1e3:8.2f} ms   gradient {a_grad*1e3:8.2f} ms   (batched, adjoint, row-parallel)")

    # PennyLane processes one row at a time (no native param-row batching), so
    # much of the gap is Python/QNode per-call overhead, not simulator speed —
    # hence both the reference (default.qubit) and the fast C++ (lightning.qubit,
    # adjoint) devices are shown. Read the numbers, not a single ratio.
    for device, diff in [("default.qubit", "backprop"), ("lightning.qubit", "adjoint")]:
        try:
            circ = build_pennylane(nq, layers, device, diff)
            p_fwd, p_grad = bench_pennylane(circ, rows, nq, layers, args.reps)
            print(f"  pennylane {device:<16} forward {p_fwd*1e3:8.2f} ms   gradient {p_grad*1e3:8.2f} ms   "
                  f"({diff}; {p_fwd/a_fwd:.0f}x / {p_grad/a_grad:.0f}x slower)")
        except Exception as e:  # noqa: BLE001
            print(f"  pennylane {device:<16} skipped ({e})")


if __name__ == "__main__":
    main()
