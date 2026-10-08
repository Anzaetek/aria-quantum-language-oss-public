# SPDX-License-Identifier: Apache-2.0
"""Aer CPU oracle for the dense lane's value gate (§7 E4: 1e-10).

Never timed, never an arm. It supplies, per circuit, the reference value of
the row's observable and the reference final state (complex128, qubit 0 = LOW
bit) that every arm's own state is checked against (§4.3a: two-sided
|1-F| <= tol AND phase-aligned max|dpsi|).

Witnessed at `hello` and after every run: method statevector, device CPU,
precision double, and the metadata Aer reports for the run itself -- Aer can
fall back silently, and its returned dtype says nothing about the precision it
computed in (§4.3a), so the metadata, not the array, is the evidence.
"""

from __future__ import annotations

import time

import numpy
import qiskit
import qiskit_aer
from qiskit import qasm2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from dense_common import parse_qasm, save_state, serve, threads_from_env

THREADS = threads_from_env()
PINNED = {"qiskit": "2.5.2", "qiskit-aer": "0.17.2"}
_CIRCUITS: dict[str, tuple[int, qiskit.QuantumCircuit]] = {}


def _sim() -> AerSimulator:
    return AerSimulator(method="statevector", device="CPU", precision="double",
                        max_parallel_threads=THREADS)


def _check(md: dict) -> None:
    if md.get("method") != "statevector" or md.get("device") != "CPU":
        raise RuntimeError(f"Aer ran as method={md.get('method')} device={md.get('device')}")


def hello(_req: dict) -> dict:
    got = {"qiskit": qiskit.__version__, "qiskit-aer": qiskit_aer.__version__}
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
                {"name": "aer_method_device", "observed": f"{md.get('method')}/{md.get('device')}",
                 "asserted": True},
                {"name": "aer_precision_option", "observed": _sim().options.precision,
                 "asserted": True},
            ]}


def load(req: dict) -> dict:
    # parse_qasm first: the same vocabulary refusal as the arms.
    n, _ = parse_qasm(req["path"])
    qc = qasm2.load(req["path"], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    if qc.num_qubits != n:
        raise RuntimeError(f"qiskit reads {qc.num_qubits} qubits, parser {n}")
    _CIRCUITS[req["name"]] = (n, qc)
    return {"qubits": n}


def _sparse(n: int, terms: list) -> SparsePauliOp:
    labels = []
    for coeff, paulis in terms:
        lab = ["I"] * n
        for q, p in paulis:
            lab[n - 1 - int(q)] = p  # qiskit label: rightmost char is qubit 0
        labels.append(("".join(lab), float(coeff)))
    return SparsePauliOp.from_list(labels)


def reference(req: dict) -> dict:
    """Reference value of the observable, and the state written to `out`."""
    n, qc = _CIRCUITS[req["name"]]
    c = qc.copy()
    c.save_expectation_value(_sparse(n, req["terms"]), list(range(n)), label="ev")
    c.save_statevector(label="sv")
    t = time.perf_counter()
    res = _sim().run(c, shots=1).result()
    dt = time.perf_counter() - t
    if not res.success:
        raise RuntimeError(f"Aer run failed: {res.status}")
    _check(res.results[0].metadata)
    data = res.data(0)
    sv = numpy.asarray(data["sv"])
    if sv.dtype != numpy.complex128:
        raise RuntimeError(f"Aer returned {sv.dtype} at precision=double")
    if req.get("out"):
        save_state(req["out"], sv)  # qiskit: qubit 0 is the LOW bit, as ours
    probs = numpy.abs(sv) ** 2
    idx = numpy.arange(len(probs), dtype=numpy.uint64)
    zmarg = [float(numpy.sum(probs * (1.0 - 2.0 * ((idx >> numpy.uint64(q)) & numpy.uint64(1))))) for q in range(n)]
    return {"value": float(numpy.real(data["ev"])), "seconds": dt, "z_marginals": zmarg,
            "norm": float(numpy.sum(probs))}


def compare(req: dict) -> dict:
    """Untimed state gate (§4.3a): `ref` and `other` are complex128 .npy files,
    qubit 0 = LOW bit. Returns the two-sided |1 - F| with F = |<ref|psi>|^2
    over UNNORMALISED inner products (so norm drift shows, as qsim's 1+9.6e-8
    did), and the max amplitude error after aligning the global phase.
    Read memory-mapped in chunks so a 26q pair stays far below the lane cap."""
    a = numpy.load(req["ref"], mmap_mode="r")
    b = numpy.load(req["other"], mmap_mode="r")
    if a.shape != b.shape or a.dtype != numpy.complex128 or b.dtype != numpy.complex128:
        raise RuntimeError(f"shapes/dtypes {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
    step = 1 << 22
    ip = 0j
    na = nb = 0.0
    for i in range(0, a.shape[0], step):
        x = numpy.asarray(a[i:i + step]); y = numpy.asarray(b[i:i + step])
        ip += numpy.vdot(x, y)
        na += float(numpy.vdot(x, x).real); nb += float(numpy.vdot(y, y).real)
    if abs(ip) == 0.0:
        raise RuntimeError("states are orthogonal")
    phase = ip / abs(ip)
    maxdiff = 0.0
    for i in range(0, a.shape[0], step):
        x = numpy.asarray(a[i:i + step]); y = numpy.asarray(b[i:i + step])
        maxdiff = max(maxdiff, float(numpy.max(numpy.abs(y * numpy.conj(phase) - x))))
    return {"one_minus_f_abs": abs(1.0 - abs(ip) ** 2), "max_abs_diff": maxdiff,
            "norm_ref": na, "norm_other": nb}


if __name__ == "__main__":
    serve({"hello": hello, "load": load, "reference": reference, "compare": compare})
