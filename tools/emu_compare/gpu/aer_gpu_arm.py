# SPDX-License-Identifier: Apache-2.0
"""Aer-GPU arm of the GPU lane (E5): qiskit-aer-gpu 0.15.1 with cuStateVec.

The SANITY FLOOR, never the primary competitor. If the hand-written
cuStateVec loop (`custatevec_arm.py`) is slower than this stale wrapper at the
same fusion setting, the loop is the finding, not the hardware.

qiskit-aer-gpu 0.15.1 is the newest CUDA-12 build on PyPI. Its own kernels are
SASS+PTX for sm_70..sm_90; on this sm_120 card they run through a driver JIT
of sm_90 PTX, paid on the first call (measured 3.4 s at 24q, cached in
~/.nv/ComputeCache afterwards). The lane's untimed warm-up is therefore
load-bearing for this arm, not hygiene. It runs on qiskit 1.x (its own venv).

Precision and fusion are pinned for the process's life (EMU_GPU_VARIANT):
  unfused    precision=single, fusion_enable=False
  fused      precision=single, fusion_enable=True (Aer's default fusion)
  fused-f64  precision=double, fusion on: an f64 CONTROL, never a row
Every run's metadata is asserted (device GPU, cuStateVec_enable True): the
returned array is complex128 whatever Aer computed in, so the metadata and the
gap to the oracle, not the dtype, are the evidence (§4.3a, §4.5).
"""

from __future__ import annotations

import json
import os
import time

import numpy
import qiskit
import qiskit_aer
from qiskit import qasm2, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from dense_common import parse_qasm, save_state, serve, threads_from_env

THREADS = threads_from_env()
VARIANT = os.environ.get("EMU_GPU_VARIANT", "")
if VARIANT not in ("unfused", "fused", "fused-f64"):
    raise SystemExit(f"EMU_GPU_VARIANT must be unfused|fused|fused-f64, not {VARIANT!r}")
PRECISION = "double" if VARIANT == "fused-f64" else "single"
PINNED = {"qiskit": "1.4.6", "qiskit-aer-gpu": "0.15.1"}
_CIRCUITS: dict[str, tuple[int, qiskit.QuantumCircuit]] = {}
_PREPARED: dict[tuple[str, str], qiskit.QuantumCircuit] = {}


def _sim() -> AerSimulator:
    return AerSimulator(method="statevector", device="GPU", precision=PRECISION,
                        cuStateVec_enable=True, fusion_enable=(VARIANT != "unfused"),
                        max_parallel_threads=THREADS)


def _check(md: dict) -> None:
    if md.get("device") != "GPU":
        raise RuntimeError(f"Aer ran on {md.get('device')!r}, not the GPU")
    if md.get("cuStateVec_enable") is not True:
        raise RuntimeError("Aer ran without cuStateVec")


def hello(_req: dict) -> dict:
    got = {"qiskit": qiskit.__version__, "qiskit-aer-gpu": qiskit_aer.__version__}
    if got != PINNED:
        raise RuntimeError(f"versions {got}, lane pins {PINNED}")
    probe = qiskit.QuantumCircuit(1)
    probe.h(0)
    probe.save_statevector()
    res = _sim().run(probe, shots=1).result()
    md = res.results[0].metadata
    _check(md)
    return {"versions": {**got, "numpy": numpy.__version__}, "threads": THREADS,
            "witnesses": [
                {"name": "aer_device", "observed": str(md.get("device")), "asserted": True},
                {"name": "aer_cuStateVec_enable", "observed": str(md.get("cuStateVec_enable")),
                 "asserted": True},
                {"name": "aer_precision_option", "observed": _sim().options.precision, "asserted": True},
                {"name": "aer_fusion_enable", "observed": str(_sim().options.fusion_enable), "asserted": True},
                {"name": "variant", "observed": VARIANT, "asserted": True},
            ]}


def load(req: dict) -> dict:
    t = time.perf_counter()
    n, _ = parse_qasm(req["path"])  # the same vocabulary refusal as every arm
    qc = qasm2.load(req["path"], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    if qc.num_qubits != n:
        raise RuntimeError(f"qiskit reads {qc.num_qubits} qubits, parser {n}")
    _CIRCUITS[req["name"]] = (n, qc)
    return {"build_s": time.perf_counter() - t}


def _sparse(n: int, terms: list) -> SparsePauliOp:
    labels = []
    for coeff, paulis in terms:
        lab = ["I"] * n
        for q, p in paulis:
            lab[n - 1 - int(q)] = p  # qiskit label: rightmost char is qubit 0
        labels.append(("".join(lab), float(coeff)))
    return SparsePauliOp.from_list(labels)


def _prepared(name: str, terms: list) -> qiskit.QuantumCircuit:
    """Observable attached and transpiled ONCE, outside every timed call (§4.2)."""
    key = (name, json.dumps(terms))
    if key not in _PREPARED:
        n, qc = _CIRCUITS[name]
        c = qc.copy()
        c.save_expectation_value(_sparse(n, terms), list(range(n)), label="ev")
        _PREPARED[key] = transpile(c, _sim(), optimization_level=0)
    return _PREPARED[key]


def expectation(req: dict) -> dict:
    c = _prepared(req["name"], req["terms"])
    sim = _sim()
    t = time.perf_counter()
    res = sim.run(c, shots=1).result()
    dt = time.perf_counter() - t
    if not res.success:
        raise RuntimeError(f"Aer run failed: {res.status}")
    _check(res.results[0].metadata)
    return {"value": float(numpy.real(res.data(0)["ev"])), "seconds": dt}


def state(req: dict) -> dict:
    n, qc = _CIRCUITS[req["name"]]
    c = qc.copy()
    c.save_statevector(label="sv")
    res = _sim().run(transpile(c, _sim(), optimization_level=0), shots=1).result()
    _check(res.results[0].metadata)
    save_state(req["out"], numpy.asarray(res.data(0)["sv"]))  # qubit 0 = LOW bit
    return {}


def floor(_req: dict) -> dict:
    c = qiskit.QuantumCircuit(1)
    c.save_expectation_value(SparsePauliOp.from_list([("Z", 1.0)]), [0], label="ev")
    c = transpile(c, _sim(), optimization_level=0)
    sim = _sim()
    t = time.perf_counter()
    res = sim.run(c, shots=1).result()
    dt = time.perf_counter() - t
    _check(res.results[0].metadata)
    v = float(numpy.real(res.data(0)["ev"]))
    if abs(v - 1.0) > 1e-6:
        raise RuntimeError(f"floor <Z0> on |0> is {v}")
    return {"seconds": dt}


if __name__ == "__main__":
    serve({"hello": hello, "load": load, "expectation": expectation, "state": state, "floor": floor})
