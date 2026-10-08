# SPDX-License-Identifier: Apache-2.0
"""qulacs arm of the dense CPU lane (E4). complex128 throughout, like ours.

Thread count: qulacs parallelises with OpenMP and reads OMP_NUM_THREADS once,
at import. So the launcher pins EMU_DENSE_THREADS, this process copies it into
OMP_NUM_THREADS BEFORE importing qulacs, and asserts the two agree; a sweep
over thread counts is one process per count.

Timed regions (§4.2), `perf_counter` around exactly:
* expectation: allocate |0..0>, `update_quantum_state`, and
  `Observable.get_expectation_value` on the evolved state. No copy-out.
  The Observable object is built before the timer, from the row's terms.
* sample: `QuantumState.sampling(shots, seed)` on an already-evolved state;
  the evolution is this arm's setup for that row and is reported separately.
"""

from __future__ import annotations

import os
import time

from dense_common import emit, parse_qasm, save_state, serve, threads_from_env

THREADS = threads_from_env()
os.environ["OMP_NUM_THREADS"] = str(THREADS)

import numpy  # noqa: E402
import qulacs  # noqa: E402
from qulacs import Observable, QuantumCircuit, QuantumState  # noqa: E402

PINNED_VERSION = "0.6.14"
_CIRCUITS: dict[str, tuple[int, QuantumCircuit]] = {}
_EVOLVED: dict[str, QuantumState] = {}


def _build(path: str) -> tuple[int, QuantumCircuit]:
    n, ops = parse_qasm(path)
    c = QuantumCircuit(n)
    for name, ps, qs in ops:
        if name == "h":
            c.add_H_gate(qs[0])
        elif name == "cx":
            c.add_CNOT_gate(qs[0], qs[1])
        elif name == "rx":  # qulacs RotX = exp(-i th X/2) = qelib rx
            c.add_RotX_gate(qs[0], ps[0])
        elif name == "rz":  # qulacs RotZ = exp(-i th Z/2) = qelib rz
            c.add_RotZ_gate(qs[0], ps[0])
        else:  # parse_qasm already refused; this is a second line of defence
            raise ValueError(f"gate {name} not mapped")
    return n, c


def _observable(n: int, terms: list) -> Observable:
    """`terms` is [[coeff, [[qubit, "X"|"Y"|"Z"], ...]], ...] from the lane binary."""
    obs = Observable(n)
    for coeff, paulis in terms:
        obs.add_operator(float(coeff), " ".join(f"{p} {int(q)}" for q, p in paulis))
    return obs


def hello(_req: dict) -> dict:
    if qulacs.__version__ != PINNED_VERSION:
        raise RuntimeError(f"qulacs {qulacs.__version__}, lane pins {PINNED_VERSION}")
    st = QuantumState(1)
    device = st.get_device_name()
    if device != "cpu":
        raise RuntimeError(f"QuantumState reports device {device!r}, this is the CPU lane")
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError("OMP_NUM_THREADS does not match the pinned thread count")
    return {
        "versions": {"qulacs": qulacs.__version__, "numpy": numpy.__version__},
        "threads": THREADS,
        "witnesses": [
            {"name": "qulacs_version", "observed": qulacs.__version__, "asserted": True},
            {"name": "state_class_device", "observed": f"{type(st).__name__}/{device}", "asserted": True},
            {"name": "omp_num_threads", "observed": os.environ["OMP_NUM_THREADS"], "asserted": True},
        ],
    }


def load(req: dict) -> dict:
    t = time.perf_counter()
    _CIRCUITS[req["name"]] = _build(req["path"])
    return {"build_s": time.perf_counter() - t, "qubits": _CIRCUITS[req["name"]][0]}


def expectation(req: dict) -> dict:
    n, c = _CIRCUITS[req["name"]]
    obs = _observable(n, req["terms"])
    t = time.perf_counter()
    st = QuantumState(n)
    c.update_quantum_state(st)
    v = obs.get_expectation_value(st)
    dt = time.perf_counter() - t
    return {"value": float(numpy.real(v)), "seconds": dt}


def state(req: dict) -> dict:
    n, c = _CIRCUITS[req["name"]]
    st = QuantumState(n)
    c.update_quantum_state(st)
    save_state(req["out"], st.get_vector())  # qulacs: qubit 0 is the LOW bit, as ours
    return {"dtype": str(st.get_vector().dtype)}


def evolve(req: dict) -> dict:
    n, c = _CIRCUITS[req["name"]]
    t = time.perf_counter()
    st = QuantumState(n)
    c.update_quantum_state(st)
    dt = time.perf_counter() - t
    _EVOLVED[req["name"]] = st
    return {"seconds": dt}


def sample(req: dict) -> dict:
    st = _EVOLVED[req["name"]]
    n = st.get_qubit_count()
    shots = int(req["shots"])
    t = time.perf_counter()
    out = st.sampling(shots, int(req["seed"]))
    dt = time.perf_counter() - t
    if len(out) != shots:
        raise RuntimeError(f"asked {shots} shots, got {len(out)}")
    arr = numpy.asarray(out, dtype=numpy.uint64)
    bits = ((arr[:, None] >> numpy.arange(n, dtype=numpy.uint64)) & 1).astype(numpy.float64)
    marg = [float(1.0 - 2.0 * bits[:, q].mean()) for q in range(n)]
    return {"seconds": dt, "shots": len(out), "z_marginals": marg}


def floor(_req: dict) -> dict:
    # Smallest possible invocation of the timed expectation path: one qubit,
    # no gates, <Z0>.
    c = QuantumCircuit(1)
    obs = _observable(1, [[1.0, [[0, "Z"]]]])
    t = time.perf_counter()
    st = QuantumState(1)
    c.update_quantum_state(st)
    v = obs.get_expectation_value(st)
    dt = time.perf_counter() - t
    if abs(numpy.real(v) - 1.0) > 1e-12:
        raise RuntimeError(f"floor <Z0> on |0> is {v}")
    return {"seconds": dt}


def floor_sample(_req: dict) -> dict:
    st = QuantumState(1)
    t = time.perf_counter()
    out = st.sampling(1, 0)
    dt = time.perf_counter() - t
    if list(out) != [0]:
        raise RuntimeError(f"floor sample of |0> is {out}")
    return {"seconds": dt}


if __name__ == "__main__":
    serve({"hello": hello, "load": load, "expectation": expectation, "state": state,
           "evolve": evolve, "sample": sample, "floor": floor, "floor_sample": floor_sample})
