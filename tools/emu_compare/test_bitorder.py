# SPDX-License-Identifier: Apache-2.0
"""Bit-order evidence: every engine, through the SAME run_engine.py path the
comparison uses, on a circuit whose output is not invariant under reversing
the qubit order, checked against a state written down by hand.

    x q[0]; h q[1]; cx q[1],q[n-1]; cx q[1],q[0]; u1(0.5) q[n-1];

Little-endian (q[0] = LOW bit of the index), the state is
    (|idx 1> + e^{0.5i} |idx 2 + 2^(n-1)>) / sqrt(2),
built below from integers, not from any simulator. Reversing the order moves
the support to {2^(n-1), 2^(n-2) + 1}, which overlaps it in nothing (n >= 3),
so a wrong convention reads F = 0, not F = 0.99. (main() asserts that.)

Engines disagree on the convention — qiskit/Aer, qulacs and omega put qubit 0
in the LOW bit of the index; cirq/qsim put qubit_order[0] in the HIGH bit — so
a comparison campaign that skips this check can report clean agreement while
comparing mirrored bitstrings. GHZ cannot catch it: it is invariant under
reversal. This is a TEST, not a report: any FAIL or ERROR fails it, and so
does a run in which every engine skipped (nothing was checked).

    pytest tools/emu_compare/test_bitorder.py
    <venv python> tools/emu_compare/test_bitorder.py [ENGINE ...]
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_engine  # noqa: E402

NS = (3, 16)


def circuit(n):
    return ('OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            f"qreg q[{n}];\nx q[0];\nh q[1];\ncx q[1],q[{n - 1}];\ncx q[1],q[0];\nu1(0.5) q[{n - 1}];\n")


def expected(n, little_endian=True):
    v = np.zeros(2 ** n, dtype=np.complex128)
    bits_a = {0: 1}                      # q0 = 1, q1 = 0, q[n-1] = 0
    bits_b = {1: 1, n - 1: 1}            # q0 = 0, q1 = 1, q[n-1] = 1

    def index(bits):
        if little_endian:
            return sum(1 << q for q, b in bits.items() if b)
        return sum(1 << (n - 1 - q) for q, b in bits.items() if b)
    v[index(bits_a)] = 1 / np.sqrt(2)
    v[index(bits_b)] = np.exp(0.5j) / np.sqrt(2)
    return v


def check_engine(engine, hostgate=(), env_hostgate=False):
    env = dict(os.environ)
    if env_hostgate:
        env.setdefault("OMEGA_HOSTGATE", "/tmp/akilles-hostgate.ledger")
        env["OMEGA_HOSTGATE_MODE"] = "enforce"
    worst = []
    with tempfile.TemporaryDirectory() as d:
        for n in NS:
            q = os.path.join(d, f"bitorder_{n}.qasm")
            with open(q, "w") as f:
                f.write(circuit(n))
            out = os.path.join(d, f"{engine}_{n}.npy")
            cmd = list(hostgate) + [run_engine.python_for(engine),
                                    os.path.join(HERE, "run_engine.py"), engine, q, out,
                                    "--reps", "0"]
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if p.returncode == run_engine.SKIP:
                return "SKIP: " + json.loads(p.stdout.strip().splitlines()[-1])["skip"][:100]
            if p.returncode != 0:
                return f"ERROR: exit {p.returncode}: {p.stderr.strip()[-300:]}"
            psi = np.load(out)
            f_le = abs(np.vdot(expected(n, True), psi)) ** 2
            f_be = abs(np.vdot(expected(n, False), psi)) ** 2
            tol = 1e-5 if run_engine.ENGINES[engine]["precision"] == "complex64" else 1e-10
            if abs(1 - f_le) > tol or f_be > tol:
                return f"FAIL: n={n} F(little-endian)={f_le:.6f} F(big-endian)={f_be:.6f}"
            worst.append((abs(1 - f_le), f_be))
    return (f"PASS (max |1-F(LE)| {max(w[0] for w in worst):.1e}, "
            f"max F(reversed) {max(w[1] for w in worst):.1e})")



def _run(engines):
    results = {e: check_engine(e) for e in engines}
    for e, r in results.items():
        print(f"{e:16s} {r}")
    return results


def _verdict(results):
    bad = {e: r for e, r in results.items()
           if not (r.startswith("PASS") or r.startswith("SKIP"))}
    passed = [e for e, r in results.items() if r.startswith("PASS")]
    return bad, passed


def test_the_two_conventions_really_differ():
    # The test of the test: if the hand-built states overlapped, a reversed
    # engine would score near 1 and slip through.
    for n in NS:
        assert abs(np.vdot(expected(n, True), expected(n, False))) ** 2 < 1e-12


def test_every_engine_reads_qubit_0_as_the_low_bit():
    bad, passed = _verdict(_run(list(run_engine.ENGINES)))
    assert not bad, f"bit-order FAIL/ERROR: {bad}"
    assert passed, "every engine skipped: nothing was checked, which is not a pass"


def main():
    test_the_two_conventions_really_differ()
    bad, passed = _verdict(_run(sys.argv[1:] or list(run_engine.ENGINES)))
    if bad:
        sys.exit(1)
    if not passed:
        print("every engine skipped: nothing was checked", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
