#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is `aria-core`'s QASM2 the same dialect Qiskit speaks?

Reads a file of `====<name>` separated QASM2 blocks emitted by
`crates/aria-core/tests/qasm2_qiskit_dialect.rs` and checks two things per gate:

1. **Qiskit loads it**, against `LEGACY_CUSTOM_INSTRUCTIONS`.

2. **It means the same thing.** The loaded circuit's `Operator` is compared
   against the operator of Qiskit's own native gate. Loading proves nothing
   about semantics — a wrong parameter order, a swapped control, or a `crz`
   emitted where `cp` was meant all load perfectly. This is the check that
   matters; the others only localise a failure.

3. **The STRICT `qasm2.loads` also accepts it**, except where noted.

Check 3 was originally argued against on the grounds that Qiskit's own
`qasm2.dumps` output does not survive its strict loader (measured: `cp`, `rxx`,
`rzz`, `ryy`, `sx`, `p`, `cswap`, `swap` all fail when Qiskit itself wrote them),
so demanding it would demand something Qiskit does not do.

That reasoning was wrong about the goal. Matching Qiskit byte-for-byte would
inherit Qiskit's own round-trip defect. Emitting qelib1 spellings (`p` -> `u1`,
`cp` -> `cu1`) and preamble `gate` definitions (`swap`, `cswap`, `rxx`, `rzz`,
`ryy`) makes the file readable by any spec-conformant QASM2 parser — a superset
of Qiskit's legacy mode — while still loading in Qiskit and producing the
identical operator. 24 of 25 pass; `sx` is exempt for a stated reason.

Exit status is 0 only if every gate passes both.
"""
import sys

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator

# Native constructions to compare against. The parameter values are distinct and
# generic (no 0, no π/2, no repeats) so that a swapped or dropped argument
# cannot coincide with the right answer.
NATIVE = {
    "ryy": (2, lambda qc: qc.ryy(0.7, 0, 1)),
    "rxx": (2, lambda qc: qc.rxx(0.7, 0, 1)),
    "rzz": (2, lambda qc: qc.rzz(0.7, 0, 1)),
    "cp": (2, lambda qc: qc.cp(0.7, 0, 1)),
    "crz": (2, lambda qc: qc.crz(0.7, 0, 1)),
    "sx": (1, lambda qc: qc.sx(0)),
    "u": (1, lambda qc: qc.u(0.3, 0.4, 0.5, 0)),
    "p": (1, lambda qc: qc.p(0.7, 0)),
    "ccx": (3, lambda qc: qc.ccx(0, 1, 2)),
    "cswap": (3, lambda qc: qc.cswap(0, 1, 2)),
    "cx": (2, lambda qc: qc.cx(0, 1)),
    "cy": (2, lambda qc: qc.cy(0, 1)),
    "cz": (2, lambda qc: qc.cz(0, 1)),
    "swap": (2, lambda qc: qc.swap(0, 1)),
    "rx": (1, lambda qc: qc.rx(0.7, 0)),
    "ry": (1, lambda qc: qc.ry(0.7, 0)),
    "rz": (1, lambda qc: qc.rz(0.7, 0)),
    "h": (1, lambda qc: qc.h(0)),
    "s": (1, lambda qc: qc.s(0)),
    "sdg": (1, lambda qc: qc.sdg(0)),
    "t": (1, lambda qc: qc.t(0)),
    "tdg": (1, lambda qc: qc.tdg(0)),
    "x": (1, lambda qc: qc.x(0)),
    "y": (1, lambda qc: qc.y(0)),
    "z": (1, lambda qc: qc.z(0)),
}

# Both sides evaluate the same analytic expressions in f64. Anything looser
# would be a tolerance picked to make the check pass.
TOL = 1e-12


def main():
    if len(sys.argv) != 2:
        print("usage: qasm2_dialect.py <emitted.txt>", file=sys.stderr)
        return 2
    blocks = open(sys.argv[1]).read().split("====")[1:]
    if not blocks:
        print("FAIL: no gate blocks in the input — the emitter produced nothing",
              file=sys.stderr)
        return 1

    # Spellings that cannot be written in strict qelib1, with the reason.
    # `sx` is not in qelib1 and cannot be DEFINED in it: the natural body
    # `sdg; h; sdg` is u3(pi/2, -pi/2, pi/2), which differs from `sx` by a global
    # phase e^{i*pi/4} (measured: max|delta| = 5.412e-01 vs qiskit's native sx),
    # and QASM 2.0 has no syntax for a global phase. Emitting the u3 form would
    # silently substitute a different operator.
    STRICT_EXEMPT = {"sx"}

    failures, checked, strict_ok = [], 0, 0
    for block in blocks:
        name, src = block.split("\n", 1)
        name = name.strip()
        if name not in NATIVE:
            failures.append(f"{name}: emitted but this harness has no native "
                            f"reference for it — add one rather than skipping")
            continue
        nq, build = NATIVE[name]

        # Load for the OPERATOR comparison with the STRICT parser where possible.
        #
        # This is not a detail. The legacy parser's BUILTINS SHADOW a file's own
        # `gate` definitions, so a wrong definition body is invisible to it —
        # measured with a deliberately broken `gate swap q0,q1 { cx; cx; cx; }`:
        #
        #   strict loader: max|delta| vs native swap = 1.000e+00  (honours ours)
        #   legacy loader: max|delta| vs native swap = 0.000e+00  (uses builtin)
        #
        # So comparing in legacy mode was comparing qiskit's builtin against
        # qiskit's builtin — guaranteed to agree, and blind to exactly the
        # definitions this harness now checks. A wrong `swap`, `cswap`, `rxx`,
        # `rzz` or `ryy` body passed it.
        try:
            loaded = qasm2.loads(src)
        except Exception:
            try:
                loaded = qasm2.loads(
                    src, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            except Exception as e:
                failures.append(f"{name}: qiskit CANNOT LOAD our QASM2 — "
                                f"{type(e).__name__}: {str(e).splitlines()[0]}\n{src}")
                continue

        ref = QuantumCircuit(nq)
        build(ref)
        try:
            delta = np.abs(Operator(loaded).data - Operator(ref).data).max()
        except Exception as e:
            failures.append(f"{name}: operators not comparable ({e}) — "
                            f"probably a qubit-count mismatch:\n{src}")
            continue

        # Strict-loadability is a SEPARATE, stronger property than "qiskit reads
        # it". qiskit's own `qasm2.dumps` output does not survive `qasm2.loads`;
        # ours is emitted in qelib1-only forms so that it does, which makes the
        # file readable by any spec-conformant QASM2 parser rather than only by
        # qiskit's legacy mode.
        try:
            qasm2.loads(src)
            strict_ok += 1
            if name in STRICT_EXEMPT:
                failures.append(
                    f"{name}: now loads in the STRICT parser — remove it from "
                    f"STRICT_EXEMPT, the exemption has outlived its reason")
        except Exception as e:
            if name not in STRICT_EXEMPT:
                failures.append(
                    f"{name}: does not load in the strict qelib1 parser — "
                    f"{str(e).splitlines()[0]}\nEmit a qelib1 spelling or a "
                    f"preamble `gate` definition.\n{src}")

        if delta > TOL:
            failures.append(
                f"{name}: qiskit LOADS our QASM2 but it means something else — "
                f"max|delta| = {delta:.3e} vs qiskit's own {name} gate.\n{src}")
        else:
            checked += 1

    # Report the qualifying count. A run that compared nothing — because the
    # emitter changed shape, or every name fell through — must not look like a
    # pass.
    if checked < len(NATIVE):
        failures.append(
            f"only {checked} of {len(NATIVE)} known gates were compared; the "
            f"emitted set shrank and this check no longer covers the table")

    if failures:
        print("QASM2 DIALECT MISMATCH vs qiskit:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"  qasm2 dialect OK: {checked} gates load in qiskit AND match its "
          f"own gate operators (tol {TOL:.0e}); "
          f"{strict_ok}/{checked} also load in the STRICT qelib1 parser "
          f"(exempt: {sorted(STRICT_EXEMPT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
