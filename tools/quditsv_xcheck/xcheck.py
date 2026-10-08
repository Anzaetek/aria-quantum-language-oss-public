#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""quditsv against two external qudit simulators, on circuits neither we nor
they wrote by hand. PLAN-QUDIT.md Q2 leg (iii).

Oracles, chosen for licence (Q0) and for what they can say:

  * mqt.qudits (MIT)        — mixed dimensions, and the non-Clifford `rxy`.
                              Its `QuantumCircuit.simulate()` (tensor-network
                              backend), built in Python, never through its
                              `from_qasm` (which drops rxy params — Q0's D2).
  * QuickQudits (Apache-2.0) — one global dimension, Clifford gates only:
                              H (Fourier), X (shift), Z (clock), CNOT
                              (= csum). Its `execute()` statevector backend.

Both oracles index the state BIG-endian (qudit 0 most significant); omega
indexes wire 0 LEAST significant, as every qubit engine here does. The
harness reindexes — and `--pin-order` proves it has to, by running one
asymmetric circuit both ways and requiring the un-reindexed comparison to
FAIL. A harness whose reindexing was accidentally the identity would pass
every symmetric circuit and prove nothing; that check is what makes this
leg falsifiable (§3f A10).

Usage (from the repo root):
  <python-with-mqt-or-quickqudits> tools/quditsv_xcheck/xcheck.py [--pin-order]
  OMEGA_RUN=path/to/omega-run   (default: target/release then target/debug)

Exit 0 only if every comparison the available oracle can make passes; a
missing oracle is reported per row, never silently skipped inside the file
(ci.sh registers the skip when NO oracle is importable).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOL = 1e-9


def omega_run() -> pathlib.Path:
    if p := os.environ.get("OMEGA_RUN"):
        return pathlib.Path(p)
    for prof in ("release", "debug"):
        p = ROOT / "target" / prof / "omega-run"
        if p.exists():
            return p
    sys.exit("no omega-run binary: build one or set OMEGA_RUN")


# --------------------------------------------------------------------------
# Circuits: a gate list in one neutral form, rendered to each side.
# (name, wires, params). Names are DITQASM's.
# --------------------------------------------------------------------------

def circuits():
    """Returns (label, dims, gates, needs_mixed_or_rxy)."""
    return [
        ("qutrit_fourier_shift", [3, 3], [("h", [0], []), ("x", [1], []), ("csum", [0, 1], []), ("z", [1], []), ("h", [1], [])], False),
        # examples/qudit/qutrit_ghz.ditqasm — analytic peaks at 0, 13, 26.
        # Phase alignment here does not replace that assertion.
        ("qutrit_ghz", [3, 3, 3], [("h", [0], []), ("csum", [0, 1], []), ("csum", [0, 2], [])], False),
        ("ququint_clifford", [5, 5], [("x", [0], []), ("x", [0], []), ("h", [1], []), ("csum", [1, 0], []), ("z", [0], []), ("x", [1], []), ("csum", [0, 1], [])], False),
        ("three_qutrits", [3, 3, 3], [("h", [0], []), ("csum", [0, 1], []), ("csum", [1, 2], []), ("x", [2], []), ("z", [0], []), ("h", [2], []), ("csum", [2, 0], [])], False),
        ("rxy_only", [3], [("rxy", [0], [0, 1, 0.7, 0.3]), ("rxy", [0], [1, 2, 1.1, -0.4]), ("rxy", [0], [0, 2, 0.25, 2.0])], True),
        ("mixed_325", [3, 2, 5], [("h", [0], []), ("x", [1], []), ("rxy", [2], [0, 3, 0.9, 0.1]), ("csum", [0, 2], []), ("csum", [1, 0], []), ("z", [2], []), ("h", [1], []), ("rxy", [0], [1, 2, 0.4, 1.2])], True),
        ("asymmetric_pin", [3, 2], [("x", [0], [])], True),  # |1⟩ on wire 0 only
    ]


def ditqasm(dims, gates) -> str:
    out = [f"DITQASM 2.0;", f"qreg q [{len(dims)}][{','.join(map(str, dims))}];"]
    for name, wires, params in gates:
        ps = f" ({', '.join(str(p) for p in params)})" if params else ""
        qs = ", ".join(f"q[{w}]" for w in wires)
        out.append(f"{name}{ps} {qs};")
    return "\n".join(out) + "\n"


def run_omega(dims, gates) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w", suffix=".qasm", delete=False) as f:
        f.write(ditqasm(dims, gates))
        path = f.name
    try:
        out = subprocess.run(
            [str(omega_run()), path, "--backend", "quditsv", "--statevector", "--format", "json"],
            capture_output=True, text=True, cwd=ROOT,
        )
    finally:
        os.unlink(path)
    if out.returncode != 0:
        raise RuntimeError(f"omega-run failed:\n{out.stderr}")
    doc = json.loads(out.stdout)
    assert doc["mode"] == "statevector", doc
    assert doc.get("quditsv", {}).get("exact") is True, "certificate must say exact"
    return np.array([complex(re, im) for re, im in doc["amplitudes"]])


def big_to_little(psi: np.ndarray, dims) -> np.ndarray:
    """Oracle order (qudit 0 most significant) → omega order (wire 0 least)."""
    return psi.reshape(dims).transpose(list(range(len(dims)))[::-1]).reshape(-1)


# --------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------

def mqt_state(dims, gates):
    from mqt.qudits.quantum_circuit import QuantumCircuit
    c = QuantumCircuit(len(dims), list(dims), 0)
    for name, wires, params in gates:
        if name == "h":
            c.h(wires[0])
        elif name == "x":
            c.x(wires[0])
        elif name == "z":
            c.z(wires[0])
        elif name == "csum":
            c.csum(list(wires))
        elif name == "rxy":
            c.r(wires[0], [int(params[0]), int(params[1]), float(params[2]), float(params[3])])
        else:
            raise ValueError(name)
    return np.asarray(c.simulate()).reshape(-1)


def quickqudits_state(dims, gates):
    import quickqudits as qq
    if len(set(dims)) != 1:
        return None  # one global dimension only
    c = qq.QuantumCircuit(len(dims), dims[0])
    for name, wires, params in gates:
        if name == "h":
            c.H(wires[0])
        elif name == "x":
            c.X(wires[0])
        elif name == "z":
            c.Z(wires[0])
        elif name == "csum":
            c.CNOT(wires[0], wires[1])
        else:
            return None  # rxy is not Clifford
    psi, _ = c.execute()
    return np.asarray(psi).reshape(-1)


def importable(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001 — absence is the finding
        return False


def phase_aligned_diff(a: np.ndarray, b: np.ndarray) -> float:
    """max |a − e^{iφ} b| over the global phase, which no oracle promises."""
    k = int(np.argmax(np.abs(b)))
    if abs(b[k]) < 1e-12:
        return float(np.max(np.abs(a - b)))
    ph = a[k] / b[k]
    ph = ph / abs(ph) if abs(ph) > 1e-12 else 1.0
    return float(np.max(np.abs(a - ph * b)))


def main() -> int:
    pin = "--pin-order" in sys.argv
    have_mqt, have_qq = importable("mqt.qudits"), importable("quickqudits")
    print(f"oracles: mqt.qudits={'yes' if have_mqt else 'no'} quickqudits={'yes' if have_qq else 'no'}")
    if not (have_mqt or have_qq):
        print("no oracle importable — nothing compared", file=sys.stderr)
        return 2
    failures, compared = [], 0
    for label, dims, gates, needs_mqt in circuits():
        ours = run_omega(dims, gates)
        assert ours.size == int(np.prod(dims)), (label, ours.size)
        rows = []
        if have_mqt:
            rows.append(("mqt.qudits", mqt_state(dims, gates)))
        if have_qq:
            s = quickqudits_state(dims, gates)
            if s is not None:
                rows.append(("quickqudits", s))
        if not rows:
            print(f"  ---- {label}: no oracle can express it ({'needs mqt' if needs_mqt else 'unknown'})")
            continue
        for oracle, theirs in rows:
            d = phase_aligned_diff(ours, big_to_little(theirs, dims))
            ok = d < TOL
            compared += 1
            print(f"  {'ok  ' if ok else 'FAIL'} {label:22s} vs {oracle:12s} max|Δ| = {d:.2e}")
            if not ok:
                failures.append(f"{label} vs {oracle}")
            if pin and label == "asymmetric_pin":
                raw = phase_aligned_diff(ours, theirs)
                print(f"  {'ok  ' if raw > 0.5 else 'FAIL'} {label:22s} UN-reindexed differs by {raw:.2f} (must be ~1: the orders really differ)")
                if raw <= 0.5:
                    failures.append("order pin: reindexing was a no-op")
    if failures:
        print(f"\nFAILED {len(failures)}: " + "; ".join(failures))
        return 1
    print(f"\nOK: quditsv agrees with the external oracles on {compared} comparisons at {TOL:g} (global phase aligned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
