# SPDX-License-Identifier: Apache-2.0
"""Check the cuStateVec arm's host-side fusion (`fusion.fuse`) against
gate-by-gate application, at f64, with no GPU. Exit status 1 on any mismatch.

Fusion folds a gate into an earlier block and permutes 2-qubit matrices when a
CX arrives in the opposite orientation to the block it joins; an error there
would only show up indirectly, as the value or state gate refusing the fused
variant. This makes it a direct failure.
"""

from __future__ import annotations

import random
import sys

import numpy as np

from fusion import fuse, gate


def apply(state: np.ndarray, n: int, targets: list[int], m: np.ndarray) -> np.ndarray:
    """Apply `m` (index bit i <-> targets[i]) by tensor contraction; written
    independently of `fusion` so the two cannot share a convention error."""
    k = len(targets)
    st = state.reshape([2] * n)  # tensor axis a <-> qubit n-1-a
    axes = [n - 1 - t for t in reversed(targets)]  # matrix row bits, most significant first
    out = np.tensordot(m.reshape([2] * (2 * k)), st, axes=(list(range(k, 2 * k)), axes))
    return np.moveaxis(out, list(range(k)), axes).reshape(-1)


def run(n: int, ops: list) -> float:
    s0 = np.zeros(1 << n, dtype=complex)
    s0[0] = 1
    a = s0.copy()
    for name, ps, qs in ops:
        a = apply(a, n, qs, gate(name, ps))
    b = s0.copy()
    for targets, m in fuse(ops):
        b = apply(b, n, targets, m)
    return float(np.max(np.abs(a - b)))


def main() -> int:
    # The CX convention itself: control q0 = 1, target q1 = 0 -> both 1.
    s = np.zeros(4, dtype=complex)
    s[0b01] = 1
    if np.argmax(np.abs(apply(s, 2, [0, 1], gate("cx", [])))) != 0b11:
        print("FAIL: cx q[0],q[1] does not map |q1=0,q0=1> to |11>")
        return 1
    cases = {
        # a CX joining a block of the opposite orientation (the [0,2,1,3] path)
        "reversed pair": (2, [("h", [], [0]), ("cx", [], [0, 1]), ("rx", [0.3], [1]), ("cx", [], [1, 0]), ("rz", [0.7], [0])]),
        # single-qubit gates folding into a 2-qubit block from both sides
        "fold both sides": (3, [("rx", [0.2], [2]), ("cx", [], [1, 2]), ("rz", [1.1], [1]), ("rz", [0.4], [2]), ("h", [], [0]), ("cx", [], [0, 1])]),
    }
    rng = random.Random(0)
    for trial in range(40):
        n = rng.randint(2, 6)
        ops = []
        for _ in range(rng.randint(5, 60)):
            r = rng.random()
            if r < 0.35:
                c, t = rng.sample(range(n), 2)
                ops.append(("cx", [], [c, t]))
            else:
                g = rng.choice(["h", "rx", "rz"])
                ops.append((g, [] if g == "h" else [rng.uniform(0, 6.3)], [rng.randrange(n)]))
        cases[f"random {trial} ({n}q, {len(ops)} gates)"] = (n, ops)
    worst = 0.0
    for name, (n, ops) in cases.items():
        d = run(n, ops)
        worst = max(worst, d)
        if d > 1e-12:
            print(f"FAIL: {name}: fused and gate-by-gate states differ by {d:.3e}")
            return 1
    print(f"ok: {len(cases)} circuits, fused == gate-by-gate to {worst:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
