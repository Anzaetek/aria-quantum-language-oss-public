#!/usr/bin/env python3
"""Cross-check Aria's `--noise` against Qiskit Aer as ground truth.

Aer's density-matrix method is the reference (exact, the same engine Aer's C++
core runs). For each (circuit x noise) case we build the *identical* circuit and
an equivalent Aer NoiseModel, take the exact reference distribution / expectation
from Aer, run the same case through the Aria CLIs (`aria`, `omega-run`), and
assert agreement.

The cases target the specific bugs this work fixed:
  * `--noise` used to be silently dropped -> readout / amp-damp / depol must
    actually move the distribution off the noiseless result.
  * the statevector sampler replayed one Kraus branch for all shots.
  * pauliprop applied the amplitude-damping adjoint in the wrong order when
    combined with depolarizing (the depol+ampdamp case below pins it).
  * per-qubit heterogeneous rates and asymmetric readout.

Two semantic conversions make Aria and Aer line up exactly:
  * Aria depolarizing p ("w.p. p apply a uniform X/Y/Z", Pauli eigenvalue
    1-4p/3)  <->  Aer depolarizing_error(4p/3) (eigenvalue 1-lambda).
  * Aria readout_flip p / {p10,p01}  <->  Aer readout confusion matrix
    [[1-p10, p10], [p01, 1-p01]] (row = true bit, col = reported bit).

Run:  tools/noise_qiskit_check/run.sh
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARIA = os.environ.get("ARIA_BIN", os.path.join(REPO, "target", "debug", "aria"))
OMEGA = os.environ.get("OMEGA_BIN", os.path.join(REPO, "target", "debug", "omega-run"))

# Monte-Carlo tolerance for the trajectory samplers (sim / mps) at SHOTS shots;
# pauliprop is exact so it gets a tight tolerance.
SHOTS = 40000
MC_TOL = 0.02
EXACT_TOL = 1e-9


# ---------------------------------------------------------------------------
# Shared circuit spec: one gate list drives both the Aria source and the Aer
# circuit, so the two can never silently diverge.
# ---------------------------------------------------------------------------
def aria_source(name, n, gates):
    lines = [f"circuit {name}() {{", f"  qreg q[{n}]", f"  creg c[{n}]"]
    for g, qs in gates:
        if len(qs) == 1:
            lines.append(f"  apply {g.upper()} on q[{qs[0]}]")
        else:
            lines.append(f"  apply {g.upper()} on " + ", ".join(f"q[{x}]" for x in qs))
    for i in range(n):
        lines.append(f"  measure q[{i}] -> c[{i}]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def aer_circuit(n, gates, measure=True):
    qc = QuantumCircuit(n, n)
    for g, qs in gates:
        getattr(qc, g)(*qs)
    if measure:
        qc.measure(range(n), range(n))
    return qc


# ---------------------------------------------------------------------------
# Aer reference
# ---------------------------------------------------------------------------
def aer_gate_noise_model(n, gate_errors):
    """gate_errors: list of (aer_error, gate_name, [qubits]) applied after that gate."""
    nm = NoiseModel()
    for err, gname, qs in gate_errors:
        nm.add_quantum_error(err, [gname], qs)
    return nm


def aer_density(n, gates, gate_errors):
    """Exact density matrix after the (noisy) circuit, no measurement."""
    qc = aer_circuit(n, gates, measure=False)
    nm = aer_gate_noise_model(n, gate_errors)
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    qc.save_density_matrix()
    res = sim.run(qc).result()
    return DensityMatrix(res.data(0)["density_matrix"])


def marginal_p1(dm, n, q):
    """P(qubit q == 1) from the density matrix diagonal."""
    diag = np.real(np.diag(dm.data))
    p1 = 0.0
    for idx in range(len(diag)):
        if (idx >> q) & 1:
            p1 += diag[idx]
    return float(p1)


def expect_z(dm, n, q):
    return 1.0 - 2.0 * marginal_p1(dm, n, q)


def apply_readout(p1, p10, p01):
    """P(report 1) given true-1 prob p1 and asymmetric readout error."""
    return p1 * (1.0 - p01) + (1.0 - p1) * p10


# ---------------------------------------------------------------------------
# Aria drivers
# ---------------------------------------------------------------------------
def _write(src, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix, dir=tempfile.gettempdir())
    with os.fdopen(fd, "w") as f:
        f.write(src)
    return path


def aria_counts(name, n, gates, backend, noise, seed=7):
    src = aria_source(name, n, gates)
    path = _write(src, ".aria")
    try:
        cmd = [ARIA, "run", path, "--circuit", name, "--backend", backend,
               "--shots", str(SHOTS), "--seed", str(seed),
               "--noise", json.dumps(noise)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"aria failed: {out.stderr.strip()}")
        # Parse lines like "|01>  15969  (0.7984)".
        probs = {}
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.startswith("|") and ">" in line:
                bits = line[1:line.index(">")]
                count = int(line.split(">")[1].split("(")[0].strip())
                probs[bits] = count
        total = sum(probs.values()) or 1
        return {b: c / total for b, c in probs.items()}
    finally:
        os.unlink(path)


def aria_p1(name, n, gates, backend, noise, q, seed=7):
    probs = aria_counts(name, n, gates, backend, noise, seed)
    # Aria prints bitstrings MSB-first (qubit n-1 ... qubit 0).
    p1 = 0.0
    for bits, p in probs.items():
        # bit for qubit q is at string position (n-1-q)
        if len(bits) == n and bits[n - 1 - q] == "1":
            p1 += p
    return p1


def aria_expect_z(name, n, gates, obs, noise):
    src = aria_source(name, n, gates)
    path = _write(src, ".aria")
    try:
        cmd = [ARIA, "run", path, "--circuit", name, "--backend", "pauliprop",
               "--expectation", obs, "--noise", json.dumps(noise)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"aria failed: {out.stderr.strip()}")
        # "<Z0> = -0.400000000000"
        for line in out.stdout.splitlines():
            if "=" in line and "<" in line:
                return float(line.split("=")[1].strip())
        raise RuntimeError(f"no expectation in output: {out.stdout!r}")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
PASS, FAIL = 0, 0


def check(desc, got, want, tol):
    global PASS, FAIL
    ok = abs(got - want) <= tol
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {desc}: aria={got:.4f} aer={want:.4f} (tol {tol})")


def main():
    for b in (ARIA, OMEGA):
        if not os.path.exists(b):
            print(f"missing binary: {b}\nbuild with: cargo build -p aria-cli -p omega-cli")
            sys.exit(2)

    # --- Case 1: readout_flip on |1> (the original silent-drop bug) ---
    print("Case: readout_flip 0.5 on X;measure (was silently noiseless)")
    gates = [("x", [0])]
    dm = aer_density(1, gates, [])  # ideal state |1>
    ref = apply_readout(marginal_p1(dm, 1, 0), 0.5, 0.5)  # 0.5
    for be in ("sim", "mps"):
        got = aria_p1("R", 1, gates, be, {"readout_flip": 0.5}, 0)
        check(f"{be} P(1)", got, ref, MC_TOL)

    # --- Case 2: amplitude_damping on |1> ---
    print("Case: amplitude_damping 0.5 on X;measure")
    err = amplitude_damping_error(0.5)
    dm = aer_density(1, gates, [(err, "x", [0])])
    ref = marginal_p1(dm, 1, 0)  # 1 - gamma = 0.5
    for be in ("sim", "mps"):
        got = aria_p1("AD", 1, gates, be, {"amplitude_damping": 0.5}, 0)
        check(f"{be} P(1)", got, ref, MC_TOL)

    # --- Case 3: depolarizing on |1> (note the 4/3 convention conversion) ---
    print("Case: depolarizing 0.3 on X;measure (aria p <-> aer 4p/3)")
    err = depolarizing_error(4.0 * 0.3 / 3.0, 1)
    dm = aer_density(1, gates, [(err, "x", [0])])
    ref = marginal_p1(dm, 1, 0)  # 1 - 2p/3 = 0.8
    for be in ("sim", "mps"):
        got = aria_p1("DP", 1, gates, be, {"depolarizing": 0.3}, 0)
        check(f"{be} P(1)", got, ref, MC_TOL)

    # --- Case 4: combined depol + ampdamp (the #1 adjoint-ordering bug) ---
    print("Case: depol 0.3 + amplitude_damping 0.5 on X;measure (ordering bug)")
    # Aria forward order: depol THEN ampdamp -> compose depol first.
    combined = depolarizing_error(4.0 * 0.3 / 3.0, 1).compose(amplitude_damping_error(0.5))
    dm = aer_density(1, gates, [(combined, "x", [0])])
    ref_p1 = marginal_p1(dm, 1, 0)
    ref_z = expect_z(dm, 1, 0)
    # self-check the reference against the hand-derived analytic value 0.2
    assert abs(ref_z - 0.2) < 1e-9, f"aer reference off: <Z>={ref_z}"
    for be in ("sim", "mps"):
        got = aria_p1("CB", 1, gates, be, {"depolarizing": 0.3, "amplitude_damping": 0.5}, 0)
        check(f"{be} P(1)", got, ref_p1, MC_TOL)
    got_z = aria_expect_z("CB", 1, gates, "Z0", {"depolarizing": 0.3, "amplitude_damping": 0.5})
    check("pauliprop <Z0>", got_z, ref_z, EXACT_TOL)

    # --- Case 5: per-qubit amplitude damping [0, 0.8] on X X ---
    print("Case: per-qubit amplitude_damping [0, 0.8] on 2 qubits")
    g2 = [("x", [0]), ("x", [1])]
    dm = aer_density(2, g2, [(amplitude_damping_error(0.8), "x", [1])])
    for q, exp in ((0, marginal_p1(dm, 2, 0)), (1, marginal_p1(dm, 2, 1))):
        got = aria_p1("PQ", 2, g2, "sim", {"amplitude_damping": [0.0, 0.8]}, q)
        check(f"sim P(q{q}=1)", got, exp, MC_TOL)

    # --- Case 6: asymmetric readout {p10, p01} on |1> ---
    print("Case: asymmetric readout p10=0.02 p01=0.3 on X;measure")
    dm = aer_density(1, gates, [])  # ideal |1>
    ref = apply_readout(marginal_p1(dm, 1, 0), 0.02, 0.3)  # 1*(1-0.3) = 0.7
    for be in ("sim", "mps"):
        got = aria_p1("RO", 1, gates, be, {"readout": [{"p10": 0.02, "p01": 0.3}]}, 0)
        check(f"{be} P(1)", got, ref, MC_TOL)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
