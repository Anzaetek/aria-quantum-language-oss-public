#!/usr/bin/env python3
"""quimb CircuitMPS arm for the MPS comparison lane.

One JSON object per stdin line, one JSON object per stdout line. Imports and
the venv are paid at process start, before the first request, so they are
never inside a timed region.

The timed region is gate application on ``CircuitMPS`` plus
``local_expectation``. That call builds the one-site reduced density matrix
(2×2) through ``local_expectation_canonical``. It does not call
``CircuitMPS.to_dense``, which would contract the full 2^n wavefunction.
"""

from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np
import quimb as qu
import quimb.tensor as qtn


def blas_name() -> str:
    """Which BLAS this numpy links. Recorded into every row."""
    try:
        info = np.show_config(mode="dicts")
    except TypeError:
        return "unknown"
    if not isinstance(info, dict):
        return "unknown"
    build = info.get("Build Dependencies") or info.get("build_dependencies") or {}
    if isinstance(build, dict):
        blas = build.get("blas") or {}
        if isinstance(blas, dict):
            name = blas.get("name")
            if name:
                return str(name)
    # numpy 2 also nests the detection under "BLAS".
    blas = info.get("BLAS") or info.get("blas")
    if isinstance(blas, dict):
        name = blas.get("name") or blas.get("libraries")
        if name:
            return str(name)
    return "unknown"


def blas_build() -> dict:
    """The BLAS build numpy links, beyond its name.

    The akilles replicate runs scipy-openblas against andromeda's Accelerate,
    so a ranking that inverts there carries this difference as well as the
    core count. Recording the version and OpenBLAS's own configuration line
    (threading cap, target arch) lets a reader see that confound instead of
    attributing the inversion to topology alone. Absent fields are "unknown",
    never omitted.
    """
    out = {"blas_version": "unknown", "blas_config": "unknown"}
    try:
        info = np.show_config(mode="dicts")
    except TypeError:
        return out
    if not isinstance(info, dict):
        return out
    build = info.get("Build Dependencies") or info.get("build_dependencies") or {}
    blas = build.get("blas") if isinstance(build, dict) else None
    if isinstance(blas, dict):
        if blas.get("version"):
            out["blas_version"] = str(blas["version"])
        if blas.get("openblas configuration"):
            out["blas_config"] = str(blas["openblas configuration"])
    return out


def versions() -> dict:
    from importlib.metadata import version

    import autoray
    import cotengra
    import numba
    import scipy

    return {
        "quimb": version("quimb"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "autoray": autoray.__version__,
        "cotengra": cotengra.__version__,
        "python": platform.python_version(),
        "blas": blas_name(),
        **blas_build(),
    }


def apply_gates(circ: qtn.CircuitMPS, gates: list) -> None:
    for g in gates:
        name = str(g["name"]).upper()
        qubits = [int(q) for q in g["qubits"]]
        params = [float(p) for p in g.get("params", [])]
        if params:
            circ.apply_gate(name, params=params, qubits=qubits)
        else:
            circ.apply_gate(name, qubits=qubits)


def run_job(job: dict) -> dict:
    n = int(job["n"])
    chi = int(job["chi"])
    cutoff = float(job["cutoff"])
    cutoff_mode = str(job["cutoff_mode"])
    gates = job["gates"]
    # renorm stays off: quimb's own default, and the same choice our split
    # makes (singular values are not rescaled; the readout divides by the norm).
    circ = qtn.CircuitMPS(
        N=n,
        max_bond=chi,
        cutoff=cutoff,
        gate_opts={"cutoff_mode": cutoff_mode, "renorm": False},
        dtype="complex128",
    )
    z = qu.pauli("Z")
    t0 = time.perf_counter()
    apply_gates(circ, gates)
    # normalized=True matches our expectation, which divides by <ψ|ψ>.
    # The default on CircuitMPS.local_expectation is False; leaving it there
    # would scale a truncated state down and fail the value gate for a
    # reason that is not the truncation.
    value = circ.local_expectation(z, 0, normalized=True)
    seconds = time.perf_counter() - t0
    bond = circ.psi.max_bond()
    if bond is None:
        raise RuntimeError("quimb reported no bond (max_bond() is None)")
    return {
        "expectation": float(np.real(value)),
        "max_bond": int(bond),
        "fidelity_estimate": float(circ.fidelity_estimate()),
        "error_estimate": float(circ.error_estimate()),
        "seconds": seconds,
        "timed_path": "CircuitMPS.apply_gate+local_expectation",
        "to_dense_called": False,
        "dtype": "complex128",
        "normalized": True,
        "renorm": False,
        "chi": chi,
        "cutoff": cutoff,
        "cutoff_mode": cutoff_mode,
    }


def hello() -> dict:
    # A one-qubit-pair product state: <Z_0> on |00> is 1. This pays numba's
    # first compilation and proves the expectation path before any row.
    smoke = run_job(
        {
            "n": 2,
            "chi": 4,
            "cutoff": 0.0,
            "cutoff_mode": "rsum2",
            "gates": [],
        }
    )
    import numba

    return {
        "hello": True,
        "versions": versions(),
        "blas": blas_name(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        # What this process will actually use. The four env vars stay unset,
        # so this is the library default, not a pin we chose after looking
        # at a number.
        "threads": int(numba.get_num_threads()),
        "self_check_z0": smoke["expectation"],
        "self_check_max_bond": smoke["max_bond"],
        "thread_env_left_unset": [
            "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ],
    }


def main() -> None:
    sys.stdout.write(json.dumps(hello()) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        ident = job.get("id")
        try:
            if job.get("op") == "quit":
                sys.stdout.write(json.dumps({"id": ident, "quit": True}) + "\n")
                sys.stdout.flush()
                return
            resp = run_job(job)
            resp["id"] = ident
        except Exception as exc:  # noqa: BLE001 — the harness must see the failure
            resp = {"id": ident, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
