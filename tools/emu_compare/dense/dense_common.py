# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the dense lane's Python arms (qulacs, qsim, the Aer oracle).

Every arm is one long-lived process speaking JSON lines on stdin/stdout, so
interpreter start, imports and circuit construction are outside every timed
region (§4.2, §4.4). The Rust lane binary decides what is timed and when; a
response's `seconds` is `perf_counter` around exactly the call the row names.

The QASM reader accepts only the vocabulary the §3 HEA and RANDOM-1 artifacts
use (`h`, `cx`, `rx`, `rz`) and refuses anything else, so an arm can never be
handed a gate it silently maps wrong.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy

VOCABULARY = ("h", "cx", "rx", "rz")
_LINE = re.compile(r"^([a-z0-9]+)(?:\(([^)]*)\))?\s+(.+);$")


def parse_qasm(path: str) -> tuple[int, list[tuple[str, list[float], list[int]]]]:
    ops: list[tuple[str, list[float], list[int]]] = []
    n = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("//")[0].strip()
            if not line or line.startswith(("OPENQASM", "include")):
                continue
            if line.startswith("qreg"):
                m = re.match(r"qreg q\[(\d+)\];", line)
                if not m:
                    raise ValueError(f"unreadable qreg: {raw!r}")
                n = int(m.group(1))
                continue
            m = _LINE.match(line)
            if not m:
                raise ValueError(f"unreadable line: {raw!r}")
            name, params, args = m.groups()
            if name not in VOCABULARY:
                raise ValueError(f"gate {name!r} is outside the dense lane's vocabulary {VOCABULARY}")
            ps = [float(p) for p in params.split(",")] if params else []
            qs = [int(x) for x in re.findall(r"q\[(\d+)\]", args)]
            ops.append((name, ps, qs))
    if n is None:
        raise ValueError(f"{path}: no qreg")
    return n, ops


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def save_state(path: str, vec: numpy.ndarray) -> None:
    """Write a little-endian (qubit 0 = LOW bit) state as complex128 .npy.

    Precision of the arm is NOT changed by this cast: an f32 state stays an
    f32-computed state, it is only widened for the comparison.
    """
    numpy.save(path, numpy.ascontiguousarray(vec, dtype=numpy.complex128))


def little_endian_from_big(vec: numpy.ndarray, n: int) -> numpy.ndarray:
    """cirq/qsim index: qubit_order[0] is the MOST significant bit."""
    return vec.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)


def z_marginals_from_samples(bits: numpy.ndarray) -> list[float]:
    """Per-qubit <Z> estimate from a (shots, n) 0/1 array, qubit-indexed."""
    return [float(1.0 - 2.0 * bits[:, q].mean()) for q in range(bits.shape[1])]


def threads_from_env() -> int:
    raw = os.environ.get("EMU_DENSE_THREADS", "")
    if not raw.isdigit() or int(raw) < 1:
        raise SystemExit("EMU_DENSE_THREADS must be a positive integer: an arm "
                         "whose thread count is not pinned is not a row (E4)")
    return int(raw)


def serve(handlers: dict) -> None:
    """Request loop. An exception becomes an error response, never a value."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        op = req.get("op")
        if op == "quit":
            emit({"ok": True})
            return
        fn = handlers.get(op)
        if fn is None:
            emit({"error": f"unknown op {op!r}"})
            continue
        try:
            emit({"ok": True, **fn(req)})
        except Exception as e:  # reported to Rust, which refuses the row
            emit({"error": f"{type(e).__name__}: {e}"})
