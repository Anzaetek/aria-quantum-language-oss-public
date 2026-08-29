# SPDX-License-Identifier: Apache-2.0
"""Check that our OpenQASM **3** export is real QASM3 and means what we think.

Companion to ``qasm2_dialect.py``, and it exists because the 2.0 check cannot
stand in for it. Two distinct claims are tested here, and only the second one
needs Qiskit's gate library:

1. **The file is valid OpenQASM 3.** Parsed with ``qiskit.qasm3.loads`` (which
   needs ``qiskit-qasm3-import``), not with a QASM2 grammar. That distinction
   is the whole point: this workspace's QASM2 reader accepts names that are not
   in ``stdgates.inc`` at all — ``cu1`` among them — so "our own parser reads it
   back" would certify a file a strict QASM3 consumer rejects. Validating an
   emitter against a reader that shares its blind spots proves nothing.

2. **The operator is the one we named.** ``Operator(parsed)`` is compared
   against Qiskit's own native gate. Loading proves only that the text is
   well-formed; emitting ``crz`` where ``cp`` was meant loads perfectly and is a
   different circuit.

``stdgates.inc`` defines **no** ``rxx``/``ryy``/``rzz`` — a property of the
OpenQASM 3 standard library, verified here rather than assumed. Our emitter
carries a ``gate`` definition for those three rather than emitting a bare name
a strict consumer would reject.

# Why the corpus carries a CLASS

This harness used to call ``Operator(circuit)`` on every record, which silently
constrained the corpus to **unitary circuits only** — three files, one gate
each. ``Operator()`` raises on ``measure`` ("Cannot apply operation with
classical bits") and on ``reset`` (no ``to_matrix``, no ``definition``), so the
emitter's riskiest output could never be in the corpus at all.

That is not a hypothetical. The single-bit guard — the feature ``to_qasm``'s
refusal message directs users to — shipped as ``if (c[0] == 1)``, which Qiskit
refuses outright:

    conditions must be 'bit == const bool' or 'bitarray == const int',
    not 'bit == const int'

Two Rust tests pinned that invalid string as correct. Nothing caught it because
nothing in this corpus had a conditional in it.

So each record is ``name<TAB>class<TAB>theta<TAB>path`` and the oracle is chosen
per class:

* ``unitary``  — must load, and (where Qiskit has a native equivalent) must
  build the same operator.
* ``measured`` — must load, and its **structure** is asserted against a
  checked-in expectation. There is no operator to compare; asserting nothing
  beyond "it loaded" would let a dropped guard through, which is the exact
  defect this class was added to catch.

Usage: ``python qasm3_dialect.py <corpus-file>``.
"""

import sys

import numpy as np

# Structure each `measured` case must show once Qiskit has loaded it.
#
# `ops` is the exact ordered list of top-level instruction names. A guard that
# was dropped shows up as `x` where `if_else` belongs, and a measurement that
# vanished shortens the list — neither is visible to a bare "it parsed" check.
MEASURED_EXPECT = {
    "conditional": {
        "ops": ["h", "measure", "if_else"],
        "why": "the single-bit guard must survive as an if_else block, not a bare x",
    },
    "measured_ghz": {
        "ops": ["h", "cx", "cx", "measure", "measure", "measure"],
        "why": "three measurements on a crossed qubit->cbit map",
    },
    "reset_barrier": {
        "ops": ["h", "reset", "barrier", "cx", "measure"],
        "why": "reset and barrier must both reach the file",
    },
}


def check_measured(name, circuit, failures):
    """Assert the loaded circuit has the structure we expected to emit."""
    expect = MEASURED_EXPECT.get(name)
    if expect is None:
        failures.append(
            f"{name}: class 'measured' but no structural expectation is "
            f"recorded. Add one — 'it loaded' alone would pass a dropped guard."
        )
        return False
    got = [inst.operation.name for inst in circuit.data]
    if got != expect["ops"]:
        failures.append(
            f"{name}: structure differs — {expect['why']}.\n"
            f"      expected {expect['ops']}\n"
            f"      got      {got}"
        )
        return False
    return True


def main(corpus_path: str) -> int:
    try:
        from qiskit import qasm3
        from qiskit.quantum_info import Operator
        from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
    except ImportError as exc:  # pragma: no cover - environment guard
        print(f"  skipping QASM3 dialect check — {exc}")
        return 0
    try:
        import qiskit_qasm3_import  # noqa: F401
    except ImportError:
        print(
            "  skipping QASM3 dialect check — qiskit-qasm3-import is not "
            "installed (pip install qiskit-qasm3-import)"
        )
        return 0

    native = {"rxx": RXXGate, "ryy": RYYGate, "rzz": RZZGate}
    tol = 1e-12
    worst = 0.0
    unitary = measured = 0
    failures = []

    with open(corpus_path) as fh:
        records = [ln.split("\t") for ln in fh.read().splitlines() if ln.strip()]

    for rec in records:
        if len(rec) != 4:
            failures.append(
                f"malformed corpus record (expected name/class/theta/path): {rec}"
            )
            continue
        name, cls, theta_s, path = rec
        src = open(path).read()
        try:
            circuit = qasm3.loads(src)
        except Exception as exc:
            failures.append(f"{name}: does not parse as OpenQASM 3 — {exc}")
            continue

        if cls == "measured":
            if check_measured(name, circuit, failures):
                measured += 1
            continue

        if cls != "unitary":
            failures.append(f"{name}: unknown corpus class {cls!r}")
            continue

        # Unitary: `Operator` is safe here and only here.
        got = Operator(circuit).data
        if name in native:
            ref = Operator(native[name](float(theta_s))).data
            delta = float(np.abs(got - ref).max())
            worst = max(worst, delta)
            if delta > tol:
                failures.append(
                    f"{name}: parses, but the operator differs from qiskit's "
                    f"native gate by {delta:.3e} (tol {tol:.0e})"
                )
                continue
        unitary += 1

    if failures:
        print("  QASM3 DIALECT CHECK FAILED")
        for f in failures:
            print(f"    {f}")
        return 1

    print(
        f"  qasm3 dialect OK: {unitary} unitary circuits match qiskit's own "
        f"operators (worst |delta| = {worst:.3e}, tol {tol:.0e}); "
        f"{measured} measured/conditional circuits load and keep their structure"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
