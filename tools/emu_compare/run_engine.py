# SPDX-License-Identifier: Apache-2.0
"""Run ONE engine on ONE circuit, in this process, and report one JSON line.

    run_engine.py ENGINE QASM OUT.npy [--reps 5]

Protocol, identical for every engine: build the engine's circuit object from
the QASM file (untimed), then run the full-state evolution 1 + REPS times in
this process. Run 0 is the warm-up; the reported time is the median of runs
1..REPS. Each timed run covers: allocate |0...0>, apply every gate, and hand
back the full final state as a host array through the engine's ordinary API.
Not timed: interpreter start, imports, QASM parsing, circuit construction,
bit-order conversion, writing OUT.npy. No sampling anywhere.

The state of the LAST run is written to OUT.npy as complex128 in the
little-endian convention (qubit q[0] is the LOW bit of the index), whatever
the engine's own convention is -- that conversion is per engine below, and
`bitorder_test.py` is the evidence it is right.

Exit status: 0 = ran, 7 = registered skip (the JSON line carries
"skip": reason), anything else = error.
"""

import argparse
import hashlib
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SKIP = 7  # not 3: omega-hostgate exits 3 when it refuses a run

# precision is a property of the engine as run here, stated, not measured.
ENGINES = {
    "omega-cpu": {"precision": "complex128", "kind": "omega", "device": "cpu"},
    # omega-run's default CUDA statevector arm is f32 (`cuda-f32`).
    "omega-cuda": {"precision": "complex64", "kind": "omega", "device": "cuda"},
    # `--precision f64` reaches the double arm (`cuda-f64`): 1q/2q unitary
    # circuits only, one launch per gate, no fusion.
    "omega-cuda-f64": {"precision": "complex128", "kind": "omega", "device": "cuda",
                       "omega_precision": "f64"},
    "qulacs-cpu": {"precision": "complex128", "kind": "qulacs"},
    # qsimcirq's state is float32 throughout; it has no double option.
    "qsim-cpu": {"precision": "complex64", "kind": "qsim"},
    "aer-cpu": {"precision": "complex128", "kind": "aer", "device": "CPU", "custatevec": False},
    "aer-gpu": {"precision": "complex128", "kind": "aer", "device": "GPU", "custatevec": False},
    "aer-custatevec": {"precision": "complex128", "kind": "aer", "device": "GPU", "custatevec": True},
    # The reference: qiskit.quantum_info.Statevector, plain numpy, shares no
    # simulation code with any engine above.
    "ref-numpy": {"precision": "complex128", "kind": "ref"},
}


# Which interpreter runs which engine. The GPU Aer wheel (qiskit-aer-gpu)
# pins an older qiskit, so it lives in its own venv.
VENVS = {
    "cpu": os.environ.get("EMU_CMP_PY_CPU", os.path.expanduser("~/work/agents/emu-compare-venv/bin/python")),
    "gpu": os.environ.get("EMU_CMP_PY_GPU", os.path.expanduser("~/work/agents/emu-compare-gpu-venv/bin/python")),
}


def python_for(engine):
    return VENVS["gpu" if engine in ("aer-gpu", "aer-custatevec") else "cpu"]


def thread_count():
    """One thread count for every engine (EMU_CMP_THREADS, default all logical
    CPUs), applied through each engine's own knob: OMP_NUM_THREADS (qulacs),
    RAYON_NUM_THREADS (omega-run), cpu_threads (qsim), max_parallel_threads
    (Aer). Set before any engine is imported."""
    t = int(os.environ.get("EMU_CMP_THREADS", os.cpu_count()))
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["RAYON_NUM_THREADS"] = str(t)
    return t


class Skip(Exception):
    pass


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def parse_qasm(path):
    """The comparison's own minimal reader, for engines with no usable QASM
    importer here (qulacs' converter lacks cu1/rzz; cirq's importer needs
    `ply`, which the venv does not have). Reads exactly the vocabulary
    workloads.py emits and refuses anything else."""
    ops, n = [], None
    pat = re.compile(r"^([a-z0-9]+)(?:\(([^)]*)\))?\s+(.+);$")
    with open(path) as f:
        for raw in f:
            line = raw.split("//")[0].strip()
            if not line or line.startswith(("OPENQASM", "include")):
                continue
            if line.startswith("qreg"):
                n = int(re.match(r"qreg q\[(\d+)\];", line).group(1))
                continue
            m = pat.match(line)
            if not m:
                raise ValueError(f"unreadable line: {raw!r}")
            name, params, args = m.groups()
            ps = [float(p) for p in params.split(",")] if params else []
            qs = [int(x) for x in re.findall(r"q\[(\d+)\]", args)]
            ops.append((name, ps, qs))
    return n, ops


def u3_matrix(th, ph, la):
    import numpy as np
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.array([[c, -np.exp(1j * la) * s],
                     [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]])


def little_endian_from_big(vec, n):
    """cirq/qsim index: qubit_order[0] is the MOST significant bit."""
    return vec.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)


# ---------------------------------------------------------------- engines


def prep_qulacs(qasm, spec):
    try:
        import qulacs
        from qulacs import QuantumCircuit, QuantumState
        from qulacs.gate import DenseMatrix, PauliRotation
    except ImportError as e:
        raise Skip(f"qulacs not importable: {e}")
    import numpy as np
    n, ops = parse_qasm(qasm)
    c = QuantumCircuit(n)
    for name, ps, qs in ops:
        if name == "x":
            c.add_X_gate(qs[0])
        elif name == "h":
            c.add_H_gate(qs[0])
        elif name == "cx":
            c.add_CNOT_gate(qs[0], qs[1])
        elif name == "cz":
            c.add_CZ_gate(qs[0], qs[1])
        elif name == "swap":
            c.add_SWAP_gate(qs[0], qs[1])
        elif name == "u3":
            c.add_U3_gate(qs[0], *ps)
        elif name == "u1":
            c.add_U1_gate(qs[0], ps[0])
        elif name == "rx":  # qulacs RotX = exp(-i th X/2) = qiskit rx
            c.add_RotX_gate(qs[0], ps[0])
        elif name == "rz":
            c.add_RotZ_gate(qs[0], ps[0])
        elif name == "rzz":  # qulacs PauliRotation = exp(+i th P/2)
            c.add_gate(PauliRotation(qs, [3, 3], -ps[0]))
        elif name == "cu1":
            g = DenseMatrix(qs[1], np.diag([1.0, np.exp(1j * ps[0])]))
            g.add_control_qubit(qs[0], 1)
            c.add_gate(g)
        else:
            raise ValueError(f"qulacs adapter: gate {name} not mapped")

    def run():
        st = QuantumState(n)
        c.update_quantum_state(st)
        return st.get_vector()

    meta = {"version": qulacs.__version__, "threads": int(os.environ["OMP_NUM_THREADS"])}
    # qulacs: qubit 0 is the least significant bit -- the same as ours.
    return n, run, (lambda v: v), meta


def prep_qsim(qasm, spec):
    try:
        import cirq
        import qsimcirq
    except ImportError as e:
        raise Skip(f"qsimcirq not importable: {e}")
    import numpy as np
    n, ops = parse_qasm(qasm)
    q = cirq.LineQubit.range(n)
    pi = np.pi
    gates = []
    for name, ps, qs in ops:
        t = [q[i] for i in qs]
        if name == "x":
            gates.append(cirq.X(*t))
        elif name == "h":
            gates.append(cirq.H(*t))
        elif name == "cx":
            gates.append(cirq.CNOT(*t))
        elif name == "cz":
            gates.append(cirq.CZ(*t))
        elif name == "swap":
            gates.append(cirq.SWAP(*t))
        elif name == "u3":
            gates.append(cirq.MatrixGate(u3_matrix(*ps)).on(*t))
        elif name == "u1":
            gates.append(cirq.ZPowGate(exponent=ps[0] / pi).on(*t))
        elif name == "rx":
            gates.append(cirq.rx(ps[0]).on(*t))
        elif name == "rz":
            gates.append(cirq.rz(ps[0]).on(*t))
        elif name == "rzz":
            gates.append(cirq.ZZPowGate(exponent=ps[0] / pi, global_shift=-0.5).on(*t))
        elif name == "cu1":
            gates.append(cirq.CZPowGate(exponent=ps[0] / pi).on(*t))
        else:
            raise ValueError(f"qsim adapter: gate {name} not mapped")
    circuit = cirq.Circuit(gates)
    threads = thread_count()
    sim = qsimcirq.QSimSimulator(qsimcirq.QSimOptions(cpu_threads=threads))

    def run():
        return sim.simulate(circuit, qubit_order=q).final_state_vector

    meta = {"version": qsimcirq.__version__, "cirq": cirq.__version__, "threads": threads,
            "options": "QSimOptions(cpu_threads=EMU_CMP_THREADS), other options default"}
    return n, run, (lambda v: little_endian_from_big(v, n)), meta


def _load_qiskit(qasm):
    from qiskit import qasm2
    return qasm2.load(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def prep_aer(qasm, spec):
    try:
        import qiskit
        import qiskit_aer
        from qiskit_aer import AerSimulator
    except ImportError as e:
        raise Skip(f"qiskit-aer not importable: {e}")
    import numpy as np
    sim = AerSimulator(method="statevector", device=spec["device"], precision="double",
                       cuStateVec_enable=spec["custatevec"],
                       max_parallel_threads=thread_count())
    # available_devices() lists GPU even in a CPU-only wheel; the only honest
    # probe is a one-qubit run on the device.
    probe = qiskit.QuantumCircuit(1)
    probe.h(0)
    probe.save_statevector()
    try:
        sim.run(probe, shots=1).result()
    except Exception as e:  # Aer raises plain RuntimeError
        if "not supported on this system" in str(e) or "cuStateVec" in str(e):
            raise Skip(f"Aer device {spec['device']} (cuStateVec={spec['custatevec']}) "
                       f"unavailable in this install: {e}")
        raise
    qc = _load_qiskit(qasm)
    n = qc.num_qubits
    qc.save_statevector()
    seen = {}

    def run():
        res = sim.run(qc, shots=1).result()
        if not res.success:
            raise RuntimeError(f"Aer run failed: {res.status}")
        md = res.results[0].metadata
        seen.update(device=md.get("device"), custatevec=md.get("cuStateVec_enable"),
                    method=md.get("method"), parallel_state_update=md.get("parallel_state_update"),
                    fusion=md.get("fusion", {}).get("applied"))
        return np.asarray(res.get_statevector(qc))

    def post(v):
        # The row must have run where it claims: Aer can fall back silently.
        if seen.get("device") != spec["device"] or bool(seen.get("custatevec")) != spec["custatevec"]:
            raise RuntimeError(f"Aer ran as {seen}, not as {spec}")
        if v.dtype != np.complex128:
            raise RuntimeError(f"Aer returned {v.dtype} with precision=double")
        return v  # qiskit: qubit 0 is the least significant bit

    meta = {"version": qiskit_aer.__version__, "qiskit": qiskit.__version__,
            "threads": thread_count(), "aer_metadata": seen}
    return n, run, post, meta


def prep_ref(qasm, spec):
    try:
        import qiskit
        from qiskit.quantum_info import Statevector
    except ImportError as e:
        raise Skip(f"qiskit not importable: {e}")
    import numpy as np
    qc = _load_qiskit(qasm)

    def run():
        return np.asarray(Statevector(qc).data)

    return qc.num_qubits, run, (lambda v: v), {"version": qiskit.__version__,
                                               "what": "qiskit.quantum_info.Statevector (numpy)"}


PYTHON_PREP = {"qulacs": prep_qulacs, "qsim": prep_qsim, "aer": prep_aer, "ref": prep_ref}


def run_python(engine, spec, qasm, out, reps):
    import numpy as np
    n, run, post, meta = PYTHON_PREP[spec["kind"]](qasm, spec)
    rss_before = rss_mib()
    times = []
    v = None
    for _ in range(1 + reps):
        v = None  # drop the previous state before the next allocation
        t = time.perf_counter()
        v = run()
        times.append(time.perf_counter() - t)
    peak = rss_mib()
    v = np.asarray(post(v), dtype=np.complex128)
    np.save(out, v)
    return n, times, {"peak_rss_mib": peak, "rss_before_mib": rss_before, **meta}


def omega_binary():
    return os.environ.get("OMEGA_RUN", os.path.join(REPO, "target", "release", "omega-run"))


def run_omega(engine, spec, qasm, out, reps):
    exe = omega_binary()
    if not os.path.exists(exe):
        raise Skip(f"omega-run not built at {exe}")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    cmd = [exe, qasm, "--statevector", "--backend", "statevector", "--device", spec["device"],
           "--dump-state-npy", out]
    if "omega_precision" in spec:
        cmd[-2:-2] = ["--precision", spec["omega_precision"]]
    # omega-run refuses `--timing-reps 0` (it would report the warm-up as the
    # median), so a state-only run (`--reps 0`, the bit-order check) omits the
    # flag. The explicit `--device` still guards the arm: `--dump-state-npy`
    # refuses an implicit device, and a device not compiled in is an error.
    if reps > 0:
        cmd[-2:-2] = ["--timing-reps", str(reps)]
    with tempfile.TemporaryFile("w+") as err:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        err.seek(0)
        stderr = err.read()
    if p.returncode != 0:
        if "feature not compiled in" in stderr:
            raise Skip(f"{exe} has no --features {spec['device']}: {stderr.strip()[:200]}")
        raise RuntimeError(f"omega-run exited {p.returncode}: {stderr.strip()[-800:]}")
    if reps == 0:
        with open(qasm) as f:
            n = int(re.search(r"qreg q\[(\d+)\];", f.read()).group(1))
        return n, [], {"peak_rss_mib": ru.ru_maxrss / 1024.0, "version": version,
                       "arm": None, "threads": None, "cmd": " ".join(cmd)}
    lines = [l for l in stderr.splitlines() if l.startswith("omega-timing: ")]
    if len(lines) != 1:
        raise RuntimeError(f"expected one omega-timing line, got {len(lines)}: {stderr[-800:]}")
    t = json.loads(lines[0][len("omega-timing: "):])
    want_arm = {"cpu": "cpu-f64", "cuda": "cuda-f32"}[spec["device"]]
    if t["threads"] != int(os.environ["RAYON_NUM_THREADS"]):
        raise RuntimeError(f"omega-run used {t['threads']} rayon threads, "
                           f"asked for {os.environ['RAYON_NUM_THREADS']}")
    if t["arm"] != want_arm:
        raise RuntimeError(f"omega-run ran arm {t['arm']}, row claims {want_arm}")
    times = [t["warmup_s"]] + t["reps_s"]
    with open(qasm) as f:
        n = int(re.search(r"qreg q\[(\d+)\];", f.read()).group(1))
    return n, times, {"peak_rss_mib": ru.ru_maxrss / 1024.0, "version": version,
                      "arm": t["arm"], "threads": t["threads"], "cmd": " ".join(cmd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=sorted(ENGINES))
    ap.add_argument("qasm")
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    thread_count()
    spec = ENGINES[a.engine]
    with open(a.qasm, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    row = {"engine": a.engine, "qasm": os.path.basename(a.qasm), "qasm_sha256": sha,
           "workload": os.path.basename(a.qasm).rsplit("_", 1)[0],
           "precision": spec["precision"]}
    try:
        fn = run_omega if spec["kind"] == "omega" else run_python
        n, times, meta = fn(a.engine, spec, os.path.abspath(a.qasm), os.path.abspath(a.out), a.reps)
    except Skip as e:
        row["skip"] = str(e)
        print(json.dumps(row))
        sys.exit(SKIP)
    row.update(n=n, warmup_s=times[0] if times else None, reps_s=times[1:],
               median_s=statistics.median(times[1:]) if len(times) > 1 else None, **meta)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
