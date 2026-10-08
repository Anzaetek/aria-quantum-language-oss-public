# SPDX-License-Identifier: Apache-2.0
"""qsim arm of the dense CPU lane (E4), on akilles only (§6).

qsimcirq 0.22.1 computes in complex64: measured on akilles against a
complex128 reference, |1-F| 1.7e-8..6.3e-7 and phase-aligned max|dpsi|
9e-9..8e-8, where a double path would sit near 1e-14. So every row this arm
feeds is precision-UNMATCHED and annotated (§4.5): evidence when we lose,
never quoted as a win.

THE REFUSAL (§7 E4, §8). A generic-path build is a strawman in our favour.
At `hello` this process asserts, and exits non-zero otherwise:
  1. `qsim_decide.detect_instructions()` -- what the build detects;
  2. the module `qsim_simulator` actually bound (`qsimcirq.qsim_avx512`, ...);
  3. that (2) is the best module this CPU supports per /proc/cpuinfo,
     so a silent downgrade to sse/avx2 on an AVX-512 box is refused too,
     and `qsim_basic` (the generic path) is refused on any x86.

Knobs, fixed per process by the launcher and printed into the row:
EMU_DENSE_THREADS -> QSimOptions.cpu_threads, EMU_DENSE_QSIM_FUSE ->
max_fused_gate_size. The lane sweeps both and keeps the fastest (§8: an
ambiguous knob is resolved in the competitor's favour).

Timed region (§4.2). `QSimSimulator.simulate_expectation_values` -- the API
the plan names -- translates the cirq circuit to qsim's form INSIDE the call.
On hea_14q_d12 that translation was 2.8 ms of a 3.7 ms call. Our arm lowers
its circuit before the clock starts, and §4.2 times Aer's `run()` on a
pre-transpiled circuit, so timing qsim's translation would be a strawman in
our favour. The timed call is therefore the kernel the public API ends in,
`<qsim module>.qsim_simulate_expectation_values(options, opsums, 0)`, with
`options["c"]` translated beforehand exactly as qsimcirq 0.22.1 does it (its
`_translate_circuit` and opsum construction, reproduced below and pinned to
that version). The public API is timed too (`expectation_api`) and reported
as context, and its value must equal the kernel's -- the evidence that the
reproduced setup is qsimcirq's own.
"""

from __future__ import annotations

import os
import platform
import time

from dense_common import little_endian_from_big, parse_qasm, save_state, serve, threads_from_env

THREADS = threads_from_env()
_fuse_raw = os.environ.get("EMU_DENSE_QSIM_FUSE", "")
if not _fuse_raw.isdigit() or int(_fuse_raw) < 1:
    raise SystemExit("EMU_DENSE_QSIM_FUSE must be a positive integer")
FUSE = int(_fuse_raw)

import cirq  # noqa: E402
import numpy  # noqa: E402
import qsimcirq  # noqa: E402
import qsimcirq.qsim_simulator as _qs  # noqa: E402
from qsimcirq import qsim_decide  # noqa: E402

PINNED_VERSION = "0.22.1"
PINNED_CIRQ = "1.7.0"
# qsim_decide.detect_instructions() codes, from qsimcirq/__init__.py
DETECT_MODULE = {0: "qsimcirq.qsim_avx512", 1: "qsimcirq.qsim_avx2",
                 2: "qsimcirq.qsim_sse"}
GENERIC = "qsimcirq.qsim_basic"

_CIRCUITS: dict[str, tuple[int, cirq.Circuit, list]] = {}
_PREPARED: dict[tuple, tuple[dict, list]] = {}


def cpu_best_module() -> str:
    """The best qsim CPU module this CPU can run, from /proc/cpuinfo flags."""
    with open("/proc/cpuinfo", encoding="utf-8") as f:
        flags = set()
        for line in f:
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                break
    if not flags:
        raise RuntimeError("no flags line in /proc/cpuinfo")
    if "avx512f" in flags:
        return "qsimcirq.qsim_avx512"
    if "avx2" in flags:
        return "qsimcirq.qsim_avx2"
    if "sse4_1" in flags:
        return "qsimcirq.qsim_sse"
    return GENERIC


def simd_witness() -> list[dict]:
    """Raises on a generic or downgraded build; returns the asserted fields."""
    if platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError(f"qsim rows are x86-only by §6; machine is {platform.machine()}")
    code = qsim_decide.detect_instructions()
    bound = _qs.qsim.__name__
    best = cpu_best_module()
    expected = DETECT_MODULE.get(code, GENERIC)
    if bound == GENERIC:
        raise RuntimeError(f"qsim bound the GENERIC path {bound} (detect code {code}): "
                           "a crippled competitor is a strawman (§8); refusing")
    if bound != expected:
        raise RuntimeError(f"qsim detected code {code} -> {expected} but bound {bound}")
    if bound != best:
        raise RuntimeError(f"qsim bound {bound} but this CPU supports {best}: "
                           "a downgraded competitor is a strawman (§8); refusing")
    return [
        {"name": "qsim_detect_instructions", "observed": str(code), "asserted": True},
        {"name": "qsim_bound_module", "observed": bound, "asserted": True},
        {"name": "cpu_best_qsim_module", "observed": best, "asserted": True},
    ]


def _sim() -> qsimcirq.QSimSimulator:
    return qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=THREADS,
                                                       max_fused_gate_size=FUSE))


def _build(path: str) -> tuple[int, cirq.Circuit, list]:
    n, ops = parse_qasm(path)
    q = cirq.LineQubit.range(n)
    gates = []
    for name, ps, qs in ops:
        t = [q[i] for i in qs]
        if name == "h":
            gates.append(cirq.H(*t))
        elif name == "cx":
            gates.append(cirq.CNOT(*t))
        elif name == "rx":
            gates.append(cirq.rx(ps[0]).on(*t))
        elif name == "rz":
            gates.append(cirq.rz(ps[0]).on(*t))
        else:
            raise ValueError(f"gate {name} not mapped")
    return n, cirq.Circuit(gates), q


def _psum(q: list, terms: list) -> cirq.PauliSum:
    m = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
    out = cirq.PauliSum()
    for coeff, paulis in terms:
        out += cirq.PauliString({q[int(i)]: m[p] for i, p in paulis}, coefficient=float(coeff))
    return out


def hello(_req: dict) -> dict:
    if qsimcirq.__version__ != PINNED_VERSION or cirq.__version__ != PINNED_CIRQ:
        raise RuntimeError(f"qsimcirq {qsimcirq.__version__} / cirq {cirq.__version__}, "
                           f"lane pins {PINNED_VERSION} / {PINNED_CIRQ}")
    witnesses = simd_witness()
    sim = _sim()
    witnesses.append({"name": "simulator_class", "observed": type(sim).__name__, "asserted": True})
    if type(sim).__name__ != "QSimSimulator":
        raise RuntimeError(f"simulator class {type(sim).__name__}")
    return {
        "versions": {"qsimcirq": qsimcirq.__version__, "cirq-core": cirq.__version__,
                     "numpy": numpy.__version__},
        "threads": THREADS,
        "fuse": FUSE,
        "witnesses": witnesses,
    }


def load(req: dict) -> dict:
    t = time.perf_counter()
    _CIRCUITS[req["name"]] = _build(req["path"])
    return {"build_s": time.perf_counter() - t, "qubits": _CIRCUITS[req["name"]][0]}


def _prepare(sim: qsimcirq.QSimSimulator, c: cirq.Circuit, q: list, psum: cirq.PauliSum):
    """qsimcirq 0.22.1's own setup for simulate_expectation_values, done
    before the clock: translated circuit into options["c"], seed into
    options["s"], and the opsum list with qsim's reversed qubit map."""
    from qsimcirq import qsim_circuit as qsimc
    cirq_order = cirq.QubitOrder.as_qubit_order(q).order_for(c.all_qubits())
    qsim_order = list(reversed(cirq_order))
    qubit_map = {qubit: index for index, qubit in enumerate(qsim_order)}
    opsum, opsum_qubits = [], set()
    for pstr in psum:
        opstring = _qs.qsim.OpString()
        opstring.weight = pstr.coefficient
        for qq, pauli in pstr.items():
            opsum_qubits.add(qq)
            qsimc.add_op_to_opstring(pauli.on(qq), qubit_map, opstring)
        opsum.append(opstring)
    options = dict(sim.qsim_options)
    options["c"], _ = sim._translate_circuit(qsimc.QSimCircuit(c), "translate_cirq_to_qsim",
                                             cirq_order)
    options["s"] = sim.get_seed()
    return options, [(opsum, len(opsum_qubits))]


def expectation(req: dict) -> dict:
    """Timed: the qsim kernel on a pre-translated circuit (see module doc)."""
    n, c, q = _CIRCUITS[req["name"]]
    key = (req["name"], repr(req["terms"]))
    sim = _sim()
    if key not in _PREPARED:
        _PREPARED[key] = _prepare(sim, c, q, _psum(q, req["terms"]))
    options, opsums = _PREPARED[key]
    kernel = sim._sim_module.qsim_simulate_expectation_values
    t = time.perf_counter()
    v = kernel(options, opsums, 0)
    dt = time.perf_counter() - t
    return {"value": float(numpy.real(v[0])), "seconds": dt}


def expectation_api(req: dict) -> dict:
    """The public API, translation included. Context only, never the row's time."""
    n, c, q = _CIRCUITS[req["name"]]
    obs = _psum(q, req["terms"])
    sim = _sim()
    t = time.perf_counter()
    v = sim.simulate_expectation_values(c, observables=obs, qubit_order=q)
    dt = time.perf_counter() - t
    return {"value": float(numpy.real(v[0])), "seconds": dt}


def translate(req: dict) -> dict:
    """The cirq->qsim translation alone, which simulate_expectation_values
    performs inside its timed call. Reported, not subtracted."""
    n, c, q = _CIRCUITS[req["name"]]
    t = time.perf_counter()
    qc = qsimcirq.QSimCircuit(c)
    qc.translate_cirq_to_qsim(cirq.QubitOrder.as_qubit_order(q))
    return {"seconds": time.perf_counter() - t}


def state(req: dict) -> dict:
    n, c, q = _CIRCUITS[req["name"]]
    v = _sim().simulate(c, qubit_order=q).final_state_vector
    save_state(req["out"], little_endian_from_big(v, n))
    return {"dtype": str(v.dtype)}


def floor(_req: dict) -> dict:
    q = cirq.LineQubit.range(1)
    c = cirq.Circuit(cirq.I(q[0]))
    obs = cirq.PauliSum.wrap(cirq.Z(q[0]))
    sim = _sim()
    options, opsums = _prepare(sim, c, q, obs)
    kernel = sim._sim_module.qsim_simulate_expectation_values
    t = time.perf_counter()
    v = kernel(options, opsums, 0)
    dt = time.perf_counter() - t
    if abs(numpy.real(v[0]) - 1.0) > 1e-6:
        raise RuntimeError(f"floor <Z0> on |0> is {v}")
    return {"seconds": dt}


if __name__ == "__main__":
    serve({"hello": hello, "load": load, "expectation": expectation,
           "expectation_api": expectation_api, "translate": translate,
           "state": state, "floor": floor})
