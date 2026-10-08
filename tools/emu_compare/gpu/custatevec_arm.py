# SPDX-License-Identifier: Apache-2.0
"""cuStateVec arm of the GPU lane (E5), driven directly through cuquantum-python.

Why direct and not through Aer: on PyPI the CUDA-12 qiskit-aer-gpu stops at
0.15.1, whose own kernels are SASS+PTX for sm_70..sm_90 only, so on an sm_120
card they run through a driver JIT of sm_90 PTX (3.4 s first call, measured);
cuStateVec 1.15.0 ships a native sm_120 cubin. Aer-GPU stays in the lane as a
sanity floor (`aer_gpu_arm.py`).

Written from NVIDIA's own samples (cuQuantum python/samples/bindings/custatevec:
gate_application.py, expectation_pauli.py, memory_handler.py), with every
ambiguous knob set in cuStateVec's favour (§8) and printed in the row:
  - matrices device-resident, uploaded before any clock;
  - workspace from the memory_handler.py pattern (default pool, release
    threshold raised, mallocAsync/freeAsync on a dedicated stream);
  - state vector allocated at `load`, outside the timed region (ours allocates
    inside its timed call);
  - the timed call is initialize_state_vector + every apply_matrix +
    compute_expectations_on_pauli_basis, synchronised.

The fusion variant is pinned for the process's life (EMU_GPU_VARIANT):
  unfused    one apply_matrix per gate, gate_application.py's shape;
  fused      host-side, untimed greedy fusion into <=2-qubit dense blocks
             (6x fewer applications; at f32 each dense block is unitary only
             to ~5e-8, which accumulates -- the lane's gates decide);
  fused-f64  the same blocks on a CUDA_C_64F state: an f64 CONTROL, never a row.
"""

from __future__ import annotations

import math
import os
import time

import cupy as cp
import cuquantum
import numpy as np
from cuquantum.bindings import custatevec as cusv

from dense_common import parse_qasm, save_state, serve, threads_from_env

THREADS = threads_from_env()
VARIANT = os.environ.get("EMU_GPU_VARIANT", "")
if VARIANT not in ("unfused", "fused", "fused-f64"):
    raise SystemExit(f"EMU_GPU_VARIANT must be unfused|fused|fused-f64, not {VARIANT!r}")
F64 = VARIANT == "fused-f64"
PINNED = {"cuquantum-python": "26.9.0", "custatevec": 11500}
DT = cuquantum.cudaDataType.CUDA_C_64F if F64 else cuquantum.cudaDataType.CUDA_C_32F
CT = cuquantum.ComputeType.COMPUTE_64F if F64 else cuquantum.ComputeType.COMPUTE_32F
NP_C = np.complex128 if F64 else np.complex64
CP_C = cp.complex128 if F64 else cp.complex64

_HANDLE = None
_STREAM = None
_DEV = cp.cuda.Device()
_LOADED: dict[str, tuple[int, list, object]] = {}


def _handle():
    global _HANDLE, _STREAM
    if _HANDLE is None:
        _HANDLE = cusv.create()
        pool = cp.cuda.runtime.deviceGetDefaultMemPool(_DEV.id)
        cp.cuda.runtime.memPoolSetAttribute(
            pool, cp.cuda.runtime.cudaMemPoolAttrReleaseThreshold, 0xFFFFFFFFFFFFFFFF)
        _STREAM = cp.cuda.Stream()
        cusv.set_stream(_HANDLE, _STREAM.ptr)
        cusv.set_device_mem_handler(_HANDLE, (
            lambda size, s: cp.cuda.runtime.mallocAsync(size, s),
            lambda ptr, size, s: cp.cuda.runtime.freeAsync(ptr, s),
            "e5 memory_handler"))
    return _HANDLE


def _gate(name: str, ps: list[float]):
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


def _fuse(ops):
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
        m = _gate(name, ps)
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


def hello(_req: dict) -> dict:
    got = {"cuquantum-python": cuquantum.__version__, "custatevec": cusv.get_version()}
    if got != PINNED:
        raise RuntimeError(f"versions {got}, lane pins {PINNED}")
    h = _handle()
    probe = cp.zeros(2, dtype=CP_C)
    attrs = cp.cuda.runtime.pointerGetAttributes(probe.data.ptr)
    if attrs.type != cp.cuda.runtime.memoryTypeDevice:
        raise RuntimeError("the state buffer is not device memory")
    cusv.initialize_state_vector(h, probe.data.ptr, DT, 1, cusv.StateVectorType.ZERO)
    _DEV.synchronize()
    if abs(complex(probe[0].get()) - 1) > 1e-6:
        raise RuntimeError("initialize_state_vector did not prepare |0>")
    props = cp.cuda.runtime.getDeviceProperties(_DEV.id)
    return {"versions": {k: str(v) for k, v in got.items()} | {"cupy": cp.__version__, "numpy": np.__version__},
            "threads": THREADS,
            "witnesses": [
                {"name": "device", "observed": f"{props['name'].decode()} sm_{props['major']}{props['minor']}",
                 "asserted": True},
                {"name": "state_buffer_memory_type", "observed": "cudaMemoryTypeDevice", "asserted": True},
                {"name": "precision", "observed": "CUDA_C_64F/COMPUTE_64F" if F64 else "CUDA_C_32F/COMPUTE_32F",
                 "asserted": True},
                {"name": "variant", "observed": VARIANT, "asserted": True},
            ]}


def load(req: dict) -> dict:
    t = time.perf_counter()
    n, ops = parse_qasm(req["path"])
    blocks = _fuse(ops) if VARIANT != "unfused" else [(qs, _gate(nm, ps)) for nm, ps, qs in ops]
    dev = [(np.asarray(tg, dtype=np.int32), cp.asarray(m.astype(NP_C))) for tg, m in blocks]
    _LOADED.clear()  # one state vector on the card at a time
    sv = cp.empty(1 << n, dtype=CP_C)
    _DEV.synchronize()
    _LOADED[req["name"]] = (n, dev, sv)
    return {"build_s": time.perf_counter() - t, "blocks": len(dev), "gates": len(ops)}


def _evolve(name: str):
    n, dev, sv = _LOADED[name]
    h = _handle()
    cusv.initialize_state_vector(h, sv.data.ptr, DT, n, cusv.StateVectorType.ZERO)
    for tg, dm in dev:
        cusv.apply_matrix(h, sv.data.ptr, DT, n, dm.data.ptr, DT, cusv.MatrixLayout.ROW, 0,
                          tg.ctypes.data, len(tg), 0, 0, 0, CT, 0, 0)
    return n, sv


def expectation(req: dict) -> dict:
    terms = req["terms"]
    paulis = [[getattr(cusv.Pauli, p) for _, p in t[1]] for t in terms]
    bases = [[int(q) for q, _ in t[1]] for t in terms]
    nb = [len(b) for b in bases]
    coeffs = np.array([float(t[0]) for t in terms])
    ev = np.empty(len(terms), dtype=np.float64)
    t0 = time.perf_counter()
    n, sv = _evolve(req["name"])
    cusv.compute_expectations_on_pauli_basis(_handle(), sv.data.ptr, DT, n, ev.ctypes.data,
                                            paulis, len(paulis), bases, nb)
    _DEV.synchronize()
    dt = time.perf_counter() - t0
    return {"value": float(coeffs @ ev), "seconds": dt}


def state(req: dict) -> dict:
    _, sv = _evolve(req["name"])
    _DEV.synchronize()
    save_state(req["out"], cp.asnumpy(sv))  # little-endian: qubit 0 = LOW bit
    return {}


def floor(_req: dict) -> dict:
    h = _handle()
    sv = cp.empty(2, dtype=CP_C)
    ev = np.empty(1, dtype=np.float64)
    t0 = time.perf_counter()
    cusv.initialize_state_vector(h, sv.data.ptr, DT, 1, cusv.StateVectorType.ZERO)
    cusv.compute_expectations_on_pauli_basis(h, sv.data.ptr, DT, 1, ev.ctypes.data,
                                            [[cusv.Pauli.Z]], 1, [[0]], [1])
    _DEV.synchronize()
    dt = time.perf_counter() - t0
    if abs(ev[0] - 1.0) > 1e-6:
        raise RuntimeError(f"floor <Z0> on |0> is {ev[0]}")
    return {"seconds": dt}


if __name__ == "__main__":
    serve({"hello": hello, "load": load, "expectation": expectation, "state": state, "floor": floor})
