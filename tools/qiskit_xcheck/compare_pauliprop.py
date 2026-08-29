# SPDX-License-Identifier: Apache-2.0
"""PauliProp expectations against Qiskit's Statevector.

Reads the emitter's output (`omega-xcheck --bin pauliprop_xcheck`), rebuilds the
same circuits in Qiskit, computes <psi|O|psi> exactly, and compares.

WHY THIS IS THE GATE THAT MATTERED. PauliProp had no independent check: it was
verified against our own CPU statevector, against `ppvm` (a same-algorithm
implementation), and GPU-vs-CPU. All of those share this project's conventions,
and two implementations sharing a convention agree on a shared mistake. This
repository has already shipped exactly that failure — the `Reset` channel was
wrong in three backends in three different bases, and every cross-backend
agreement gate passed because each pair happened to coincide in whatever basis
was being checked.

VACUOUS-PASS GUARD. A run that compares zero cases exits non-zero. A green
"0 compared" is the failure mode this file exists to prevent, not a pass.
"""
import sys

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

TOL = 1e-9  # exact engine, no truncation — this is a bug bar, not a tolerance


def build(n, toks):
    qc = QuantumCircuit(n)
    for tok in toks:
        parts = tok.split(":")
        g = parts[0]
        if g == "cx":
            a, b = parts[1].split(",")
            qc.cx(int(a), int(b))
        elif g == "h":
            qc.h(int(parts[1]))
        elif g == "s":
            qc.s(int(parts[1]))
        elif g == "t":
            qc.t(int(parts[1]))
        elif g in ("rz", "rx", "ry"):
            q, th = int(parts[1]), float(parts[2])
            getattr(qc, g)(th, q)
        else:
            raise SystemExit(f"unknown gate token in the corpus: {tok!r}")
    return qc


def main(path):
    worst = 0.0
    n_ok = n_bad = 0
    seen_end = False

    for line in open(path):
        line = line.strip()
        if not line or line == "#BEGIN":
            continue
        if line.startswith("#END"):
            seen_end = True
            continue

        circ, obs_s, val_s = line.split(" | ")
        parts = circ.split()
        n = int(parts[0])
        qc = build(n, parts[1:])

        # The emitter already reversed each label into Qiskit's little-endian
        # order, so these strings are consumed as-is. Getting that wrong is the
        # single most likely way for both sides to agree on a misreading, which
        # is why the reversal lives on one side only and is commented there.
        terms = []
        for t in obs_s.split():
            c, lab = t.split("*")
            terms.append((lab, float(c)))
        op = SparsePauliOp.from_list(terms)

        want = float(Statevector(qc).expectation_value(op).real)
        got = float(val_s)
        d = abs(want - got)
        worst = max(worst, d)
        if d <= TOL:
            n_ok += 1
        else:
            n_bad += 1
            if n_bad <= 5:
                print(f"  MISMATCH |d|={d:.3e}  pauliprop={got:.12f} qiskit={want:.12f}")
                print(f"    circuit: {circ}")
                print(f"    obs    : {obs_s}")

    total = n_ok + n_bad
    if not seen_end:
        print("pauliprop xcheck: emitter output truncated (no #END) — refusing to pass")
        return 1
    if total == 0:
        print("pauliprop xcheck: 0 cases compared — a vacuous pass is not a pass")
        return 1

    print(f"PauliProp vs Qiskit: {n_ok} agree, {n_bad} disagree, worst |d| = {worst:.3e}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
