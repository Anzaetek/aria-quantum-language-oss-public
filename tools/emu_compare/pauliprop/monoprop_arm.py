# SPDX-License-Identifier: Apache-2.0
"""monoprop arm of the pauliprop lane (E7). monoprop 0.9.0, Algorithmiq, C++ core.

One long-lived process; imports and circuit construction are outside every
timed region. The timed call is exactly what a monoprop user runs for one
expectation value: construct `PauliPropagator(observable, [], cutoff=..,
lower_atol=..)`, `propagate(circuit)`, `expectation_value()`.

Threads: monoprop sizes its worker pool to the process's CPU affinity and
exposes no thread knob. Measured on akilles, exact 14q HEA: 6.1 ms on all 32
CPUs against 14.7 ms pinned to one. The lane leaves it on all of them (§8: an
ambiguous knob is resolved in the competitor's favour) and records the
affinity count.

Circuits. monoprop takes Pauli-rotation `ExpGate`s only (no H, no CX). The
fixture cases are structured rotations already (`tools/pp_cross_check`). The
§3 HEA artifact is TRANSCRIBED, op for op, and refused if any op does not
match: the leading H layer acts on |0>, where H = Ry(pi/2); every
`cx a,b; rz(t) b; cx a,b` is exp(-i t/2 Z_a Z_b); `rx` is native. The
transcription is proven by the exact value gate (monoprop exact vs the dense
oracle), not assumed.

Sign convention (as tools/pp_cross_check/monoprop_ref.py, verified there):
`ExpGate` applies e^{+i theta H}; qelib `r<P>(t)` is e^{-i t P/2}; so the
parameter is -t/2.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import monoprop as mp
import numpy

PINNED_VERSION = "0.9.0"
_CIRCUITS: dict[str, tuple[int, object]] = {}


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _rotations_circuit(n: int, rots: list[tuple[str, list[int], float]]):
    gates, params = [], []
    for k, (letters, qs, theta) in enumerate(rots):
        gates.append(mp.ExpGate(mp.PauliOperator({mp.Pauli(letters, qs): 1.0}, n), k))
        params.append(-theta / 2.0)
    return mp.Circuit(gates, n, parameters=params)


LETTERS = {"rx": "X", "ry": "Y", "rz": "Z", "rxx": "XX", "rzz": "ZZ"}


def transcribe_qasm(path: str) -> tuple[int, list[tuple[str, list[int], float]]]:
    """The HEA artifact as rotations. Refuses anything it cannot map exactly."""
    import re
    n, ops = None, []
    pat = re.compile(r"^([a-z0-9]+)(?:\(([^)]*)\))?\s+(.+);$")
    for raw in open(path, encoding="utf-8"):
        line = raw.split("//")[0].strip()
        if not line or line.startswith(("OPENQASM", "include")):
            continue
        if line.startswith("qreg"):
            n = int(re.match(r"qreg q\[(\d+)\];", line).group(1))
            continue
        m = pat.match(line)
        if not m:
            raise ValueError(f"unreadable line {raw!r}")
        name, ps, args = m.groups()
        ops.append((name, [float(p) for p in ps.split(",")] if ps else [],
                    [int(x) for x in re.findall(r"q\[(\d+)\]", args)]))
    rots, i, past_h = [], 0, False
    while i < len(ops):
        name, ps, qs = ops[i]
        if name == "h":
            if past_h:
                raise ValueError("H after the first layer: not on |0>, cannot transcribe as Ry(pi/2)")
            rots.append(("Y", [qs[0]], math.pi / 2))
            i += 1
            continue
        past_h = True
        if (name == "cx" and i + 2 < len(ops) and ops[i + 1][0] == "rz"
                and ops[i + 1][2] == [qs[1]] and ops[i + 2][0] == "cx" and ops[i + 2][2] == qs):
            rots.append(("ZZ", qs, ops[i + 1][1][0]))
            i += 3
            continue
        if name in ("rx", "rz"):
            rots.append((LETTERS[name], [qs[0]], ps[0]))
            i += 1
            continue
        raise ValueError(f"op {i} {name} {qs} has no exact rotation form here")
    return n, rots


def _observable(n: int, terms: list):
    return mp.PauliOperator({mp.Pauli("".join(p for _, p in ps), [int(q) for q, _ in ps]): float(c)
                             for c, ps in terms}, n)


def hello(_req: dict) -> dict:
    if mp.__version__ != PINNED_VERSION:
        raise RuntimeError(f"monoprop {mp.__version__}, lane pins {PINNED_VERSION}")
    cpus = len(os.sched_getaffinity(0))
    # One-qubit sanity on the sign convention: rx(0.3) -> <Z0> = cos 0.3.
    c = _rotations_circuit(1, [("X", [0], 0.3)])
    pp = mp.PauliPropagator(_observable(1, [[1.0, [[0, "Z"]]]]), [], cutoff=1)
    pp.propagate(c)
    v = float(pp.expectation_value())
    if abs(v - math.cos(0.3)) > 1e-12:
        raise RuntimeError(f"sign-convention smoke: rx(0.3) gives <Z0> {v}, want cos 0.3")
    return {
        "versions": {"monoprop": mp.__version__, "numpy": numpy.__version__},
        "threads": cpus,
        "witnesses": [
            {"name": "monoprop_version", "observed": mp.__version__, "asserted": True},
            {"name": "propagator_class", "observed": type(pp).__name__, "asserted": True},
            {"name": "sign_convention_rx0.3_z0", "observed": f"{v:.15f}", "asserted": True},
            {"name": "cpu_affinity", "observed": str(cpus), "asserted": True},
        ],
    }


def load_ops(req: dict) -> dict:
    n = int(req["n"])
    rots = [(LETTERS[o["op"]], o["q"], float(o["theta"])) for o in req["ops"]]
    _CIRCUITS[req["name"]] = (n, _rotations_circuit(n, rots))
    return {"rotations": len(rots)}


def load_qasm(req: dict) -> dict:
    n, rots = transcribe_qasm(req["path"])
    _CIRCUITS[req["name"]] = (n, _rotations_circuit(n, rots))
    return {"rotations": len(rots), "qubits": n}


def expectation(req: dict) -> dict:
    n, circ = _CIRCUITS[req["name"]]
    obs = _observable(n, req["terms"])
    kw = {"cutoff": int(req["cutoff"])}
    if req.get("lower_atol") is not None:
        kw["lower_atol"] = float(req["lower_atol"])
    t = time.perf_counter()
    pp = mp.PauliPropagator(obs, [], **kw)
    pp.propagate(circ)
    v = float(pp.expectation_value())
    dt = time.perf_counter() - t
    return {"value": v, "seconds": dt, "terms": int(pp.size())}


def floor(_req: dict) -> dict:
    c = _rotations_circuit(1, [("X", [0], 0.0)])
    obs = _observable(1, [[1.0, [[0, "Z"]]]])
    t = time.perf_counter()
    pp = mp.PauliPropagator(obs, [], cutoff=1)
    pp.propagate(c)
    v = float(pp.expectation_value())
    dt = time.perf_counter() - t
    if abs(v - 1.0) > 1e-12:
        raise RuntimeError(f"floor <Z0> is {v}")
    return {"seconds": dt}


HANDLERS = {"hello": hello, "load_ops": load_ops, "load_qasm": load_qasm,
            "expectation": expectation, "floor": floor}

if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "quit":
            emit({"ok": True})
            break
        try:
            emit({"ok": True, **HANDLERS[req["op"]](req)})
        except Exception as e:
            emit({"error": f"{type(e).__name__}: {e}"})
