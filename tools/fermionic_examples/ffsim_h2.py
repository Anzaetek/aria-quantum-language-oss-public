#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""⟨H⟩ for the H2 example, in ffsim, on the same fermion operator.

The state is the one the circuit prepares, written in the occupation basis
rather than replayed gate by gate: Hartree–Fock is both electrons in spatial
orbital 0, and the ground state is ``cos θ |HF⟩ + sin θ |doubles⟩`` with θ
from ``h2_reference.txt``. Mode order matches the operator file: even modes
are spin up, odd modes are spin down, spatial orbital ``mode // 2``.

Exits 1 if either energy disagrees with the committed OpenFermion digits by
more than 1e-9. This is a second implementation of the contraction, not a
second copy of our Jordan–Wigner matrix.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import ffsim

ROOT = pathlib.Path(__file__).resolve().parents[2]
FERM = ROOT / "examples" / "fermionic"
TOL = 1e-9


def reference() -> tuple[float, float, float]:
    hf = fci = theta = None
    for line in (FERM / "h2_reference.txt").read_text().splitlines():
        if line.startswith("E_HF "):
            hf = float(line.split()[1])
        elif line.startswith("E_FCI "):
            fci = float(line.split()[1])
        elif line.startswith("theta "):
            theta = float(line.split()[1])
    if hf is None or fci is None or theta is None:
        sys.exit("h2_reference.txt is missing E_HF, E_FCI or theta")
    return hf, fci, theta


def parse_operator(text: str) -> tuple[float, ffsim.FermionOperator]:
    body = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )
    rest = body.strip()
    const = 0.0
    terms: dict = {}
    while rest:
        open_b = rest.find("[")
        if open_b < 0:
            break
        close_b = rest.find("]", open_b)
        raw = "".join(rest[:open_b].split())
        if raw.startswith("+"):
            raw = raw[1:]
        coeff = 1.0 if raw == "" else float(raw)
        ops = []
        for tok in rest[open_b + 1 : close_b].split():
            create = tok.endswith("^")
            mode = int(tok[:-1] if create else tok)
            spatial, spin = divmod(mode, 2)
            alpha = spin == 0
            if create and alpha:
                act = ffsim.cre_a(spatial)
            elif create:
                act = ffsim.cre_b(spatial)
            elif alpha:
                act = ffsim.des_a(spatial)
            else:
                act = ffsim.des_b(spatial)
            ops.append(act)
        key = tuple(ops)
        if not key:
            const += coeff
        else:
            terms[key] = terms.get(key, 0.0) + coeff
        rest = rest[close_b + 1 :]
    return const, ffsim.FermionOperator(terms)


def main() -> int:
    script_hf, script_fci, theta = reference()
    const, op = parse_operator((FERM / "h2_op.txt").read_text())
    norb, nelec = 2, (1, 1)
    lin = ffsim.protocols.linear_operator(op, norb=norb, nelec=nelec)
    hf = ffsim.hartree_fock_state(norb, nelec)
    doubles = ffsim.slater_determinant(norb, occupied_orbitals=([1], [1]))

    def energy(vec: np.ndarray) -> float:
        return float(np.real(np.vdot(vec, lin @ vec) + const))

    e_hf = energy(hf)
    psi = np.cos(theta) * hf + np.sin(theta) * doubles
    e_fci = energy(psi)
    print(f"ffsim E_HF {e_hf:.17g}  file {script_hf:.17g}")
    print(f"ffsim E(theta) {e_fci:.17g}  file {script_fci:.17g}")
    bad = False
    if abs(e_hf - script_hf) > TOL:
        print(f"HF disagrees by {abs(e_hf - script_hf):.3e}", file=sys.stderr)
        bad = True
    if abs(e_fci - script_fci) > TOL:
        print(f"FCI disagrees by {abs(e_fci - script_fci):.3e}", file=sys.stderr)
        bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
