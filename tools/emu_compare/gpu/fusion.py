# SPDX-License-Identifier: Apache-2.0
"""Gate matrices and the host-side fusion of the cuStateVec arm, in pure numpy
so they can be checked without a GPU (`check_fusion.py`).

Convention (cuStateVec's): a block's matrix index bit i corresponds to its
targets[i]; the CX matrix takes targets [control, target].
"""

from __future__ import annotations

import math

import numpy as np


def gate(name: str, ps: list[float]):
    if name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    if name == "rx":
        c, s = math.cos(ps[0] / 2), math.sin(ps[0] / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "rz":
        return np.diag([np.exp(-0.5j * ps[0]), np.exp(0.5j * ps[0])])
    if name == "cx":  # targets [control, target]; matrix bit i <-> targets[i]
        return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
    raise ValueError(name)


def fuse(ops):
    """Greedy fusion into 1- and 2-qubit dense blocks. A gate folds into the
    last block touching its qubits when that block is also the last on every
    qubit it acts on, which is exactly when moving the gate earlier is valid.
    Checked exact against gate-by-gate application at f64 (|diff| ~1e-15)."""
    blocks, last = [], {}

    def embed(m1, q, targets):
        if len(targets) == 1:
            return m1
        return np.kron(np.eye(2), m1) if targets[0] == q else np.kron(m1, np.eye(2))

    for name, ps, qs in ops:
        m = gate(name, ps)
        if len(qs) == 1:
            q = qs[0]
            if q in last:
                b = blocks[last[q]]
                b[1] = embed(m, q, b[0]) @ b[1]
            else:
                blocks.append([[q], m]); last[q] = len(blocks) - 1
            continue
        c, t = qs
        targets = [c, t]
        bc, bt = last.get(c), last.get(t)
        if bc is not None and bc == bt:
            b = blocks[bc]
            m2 = m if b[0] == targets else m[[0, 2, 1, 3]][:, [0, 2, 1, 3]]
            b[1] = m2 @ b[1]
            continue
        acc = np.eye(4, dtype=complex)
        for q, bi in ((c, bc), (t, bt)):
            if bi is not None and len(blocks[bi][0]) == 1:
                acc = embed(blocks[bi][1], q, targets) @ acc
                blocks[bi] = None
        blocks.append([targets, m @ acc]); last[c] = last[t] = len(blocks) - 1
    return [(b[0], b[1]) for b in blocks if b is not None]
