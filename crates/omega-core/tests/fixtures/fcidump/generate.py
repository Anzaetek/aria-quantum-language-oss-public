#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the FCIDUMP fixtures with PySCF, and fail if the committed bytes drift.

The files in this directory are bytes PySCF's own `tools.fcidump` wrote
(`from_integrals`, `from_scf`). A fixture written by the Rust reader would
be that reader agreeing with itself. Do not hand-edit them. Regenerate by
running this script with the ffsim venv, which has PySCF 2.14:

    crates/omega-bridges/python/.venv-ffsim/bin/python \\
        crates/omega-core/tests/fixtures/fcidump/generate.py

    ... generate.py --check
    ... generate.py --expand path/to/file.fcidump

`--check` rewrites nothing. It regenerates in memory, diffs against the
committed files, then flips one core-energy digit and checks that the
comparison notices. `--expand` prints the chemist Hamiltonian of one file
as OpenFermion ladder lines, using `fcidump.read` and `ao2mo.restore` — not
the Rust reader.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
from pyscf import ao2mo, gto, scf
from pyscf.tools.fcidump import from_integrals, from_scf, read

HERE = pathlib.Path(__file__).resolve().parent
NAMES = ("hubbard.fcidump", "h2_sto3g.fcidump", "random3.fcidump")


def write_all(directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _hubbard(directory / "hubbard.fcidump")
    _h2(directory / "h2_sto3g.fcidump")
    _random(directory / "random3.fcidump")


def _hubbard(path: pathlib.Path) -> None:
    """Two-site Hubbard, spin-blocked modes: 0,1 spin-up and 2,3 spin-down.

    t = 1, U = 4, core energy 0.5. The two-electron tensor is the chemist
    integrals (00|22) = (22|00) = U and (11|33) = (33|11) = U, which the
    Hamiltonian formula (1/2) Σ (pq|rs) a†_p a†_r a_s a_q turns into
    U n_0 n_2 + U n_1 n_3. The one-body piece is -t on each spin's bond.
    """
    n = 4
    h1 = np.zeros((n, n))
    h1[0, 1] = h1[1, 0] = -1.0
    h1[2, 3] = h1[3, 2] = -1.0
    h2 = np.zeros((n, n, n, n))
    u = 4.0
    h2[0, 0, 2, 2] = u
    h2[2, 2, 0, 0] = u
    h2[1, 1, 3, 3] = u
    h2[3, 3, 1, 1] = u
    from_integrals(path, h1, h2, n, nelec=4, nuc=0.5, ms=0)


def _h2(path: pathlib.Path) -> None:
    """H2 / STO-3G RHF. Spatial orbitals, as `from_scf` writes them."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    from_scf(mf, path)


def _random(path: pathlib.Path) -> None:
    """Three orbitals, seed 20261001. `from_integrals` packs the 8-fold form."""
    rng = np.random.default_rng(20261001)
    n = 3
    raw = rng.normal(size=(n, n))
    h1 = (raw + raw.T) / 2
    h2 = rng.normal(size=(n, n, n, n))
    from_integrals(path, h1, h2, n, nelec=4, nuc=1.25, ms=0)


def expand(path: pathlib.Path) -> None:
    """Print one OpenFermion term per line: `coeff [p^ q ...]`."""
    ctx = read(str(path), verbose=False)
    n = int(ctx["NORB"])
    h1 = ctx["H1"]
    h2 = ao2mo.restore(1, ctx["H2"], n)
    ecore = float(ctx.get("ECORE", 0.0))
    if ecore != 0.0:
        _emit(ecore, [])
    for p in range(n):
        for q in range(n):
            v = float(h1[p, q])
            if v != 0.0:
                _emit(v, [(p, True), (q, False)])
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    v = 0.5 * float(h2[p, q, r, s])
                    if v != 0.0:
                        # a†_p a†_r a_s a_q, chemist (pq|rs).
                        _emit(v, [(p, True), (r, True), (s, False), (q, False)])


def _emit(coeff: float, ops: list[tuple[int, bool]]) -> None:
    body = " ".join(f"{mode}^" if dagger else f"{mode}" for mode, dagger in ops)
    print(f"{coeff!r} [{body}]")


def _tooth(live: dict[str, str], committed: dict[str, str]) -> None:
    """A comparison that ignores a changed core energy is not a comparison."""
    original = live["hubbard.fcidump"]
    if "0.5" not in original:
        sys.exit("A10: hubbard.fcidump has no core energy 0.5 to perturb")
    flipped = original.replace("0.5", "9.5", 1)
    if flipped == original or flipped == committed["hubbard.fcidump"]:
        sys.exit("A10: changing the core energy left the fixture comparison equal")


def check() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        write_all(directory)
        live = {name: (directory / name).read_text() for name in NAMES}
    committed = {}
    for name in NAMES:
        path = HERE / name
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
            return 1
        committed[name] = path.read_text()
        if committed[name] != live[name]:
            print(
                f"{name} disagrees with live PySCF. "
                "Regenerate deliberately; do not hand-edit the fixture.",
                file=sys.stderr,
            )
            return 1
    _tooth(live, committed)
    print(f"OK: {', '.join(NAMES)} match PySCF; a changed core energy is detected")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args == []:
        write_all(HERE)
        for name in NAMES:
            print(f"wrote {name}")
        return 0
    if args == ["--check"]:
        return check()
    if len(args) == 2 and args[0] == "--expand":
        expand(pathlib.Path(args[1]))
        return 0
    print(
        "usage: generate.py [--check | --expand FILE]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
