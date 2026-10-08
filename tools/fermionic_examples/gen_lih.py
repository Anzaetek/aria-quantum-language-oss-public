#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LiH / STO-3G FCIDUMP, as bytes PySCF wrote, and the PySCF energies.

The FCIDUMP is ``pyscf.tools.fcidump.from_integrals`` on the spin-orbital
tensors of an RHF calculation. This script does not format those bytes
itself. A second run with ``--check`` regenerates them in a temporary
directory and diffs. It also rebuilds the reference text from a fresh
``scf.RHF`` and ``fci.FCI`` and diffs that.

Spin-orbital order is blocked, matching ``mode m[6] spin``: spatial
orbitals 0..5 are spin-up on modes 0..5 and spin-down on modes 6..11.
PySCF's MO columns are in energy order. The RHF determinant occupies the
two lowest spatial orbitals, both spins (modes 0, 1, 6, 7).

Geometry: Li at the origin, H at 1.6 Å on z. Basis STO-3G. The nuclear
repulsion is the FCIDUMP core energy.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
from pyscf import ao2mo, fci, gto, scf
from pyscf.tools.fcidump import from_integrals

ROOT = pathlib.Path(__file__).resolve().parents[2]
FERM = ROOT / "examples" / "fermionic"
DUMP_NAME = "lih_sto3g.fcidump"
REF_NAME = "lih_reference.txt"
R_ANGSTROM = 1.6


def molecule():
    mol = gto.M(
        atom=f"Li 0 0 0; H 0 0 {R_ANGSTROM}",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        sys.exit("RHF did not converge")
    occ = np.asarray(mf.mo_occ)
    if list(occ) != [2.0, 2.0, 0.0, 0.0, 0.0, 0.0]:
        sys.exit(f"expected two doubly occupied MOs in energy order, got {occ}")
    return mol, mf


def spin_orbital_tensors(mol, mf):
    """Blocked alpha-then-beta chemist integrals. One index, one mode."""
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    nso = 2 * norb
    h1s = np.zeros((nso, nso))
    h2s = np.zeros((nso, nso, nso, nso))
    for spin in (0, 1):
        a = spin * norb
        h1s[a : a + norb, a : a + norb] = h1
    for sp in (0, 1):
        for sr in (0, 1):
            for p, q, r, s in np.ndindex(norb, norb, norb, norb):
                v = float(eri[p, q, r, s])
                if v == 0.0:
                    continue
                h2s[p + sp * norb, q + sp * norb, r + sr * norb, s + sr * norb] = v
    return h1s, h2s


def hf_contraction(h1s, h2s, nuc, occupied):
    """Slater-determinant energy of the chemist Hamiltonian this file stores."""
    e = float(nuc)
    for i in occupied:
        e += float(h1s[i, i])
    for i in occupied:
        for j in occupied:
            e += 0.5 * (float(h2s[i, i, j, j]) - float(h2s[i, j, j, i]))
    return e


def energies(mol, mf):
    e_hf = float(mf.e_tot)
    e_fci = float(fci.FCI(mf).kernel()[0])
    if not e_fci < e_hf:
        sys.exit(f"FCI {e_fci} is not below RHF {e_hf}")
    return e_hf, e_fci


def reference_text(e_hf, e_fci) -> str:
    # .16g is stable for these magnitudes and is what --check diffs.
    return (
        "# LiH / STO-3G, R = 1.6 Angstrom.\n"
        "# PySCF 2.14.0: scf.RHF, then fci.FCI on that mean field.\n"
        "# Spin-orbital FCIDUMP from pyscf.tools.fcidump.from_integrals,\n"
        "# blocked alpha then beta. The circuit is the RHF determinant\n"
        "# (modes 0, 1, 6, 7), so it reproduces E_HF and not E_FCI.\n"
        f"E_HF {e_hf:.16g}\n"
        f"E_FCI {e_fci:.16g}\n"
        f"R_Angstrom {R_ANGSTROM}\n"
        "n_spatial 6\n"
        "n_modes 12\n"
        "nelec 4\n"
    )


def write_dump(path: pathlib.Path, mol, mf) -> None:
    h1s, h2s = spin_orbital_tensors(mol, mf)
    norb = mf.mo_coeff.shape[1]
    occupied = [0, 1, norb, norb + 1]
    contracted = hf_contraction(h1s, h2s, mol.energy_nuc(), occupied)
    if abs(contracted - float(mf.e_tot)) > 1e-10:
        sys.exit(
            f"spin-orbital contraction {contracted} disagrees with RHF {mf.e_tot}"
        )
    nso = h1s.shape[0]
    from_integrals(
        path,
        h1s,
        h2s,
        nso,
        nelec=int(mol.nelectron),
        nuc=float(mol.energy_nuc()),
        ms=0,
    )


def build(directory: pathlib.Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    mol, mf = molecule()
    write_dump(directory / DUMP_NAME, mol, mf)
    e_hf, e_fci = energies(mol, mf)
    text = reference_text(e_hf, e_fci)
    (directory / REF_NAME).write_text(text)
    return text


def check() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        live_dir = pathlib.Path(tmp)
        build(live_dir)
        for name in (DUMP_NAME, REF_NAME):
            live = (live_dir / name).read_text()
            path = FERM / name
            if not path.is_file():
                print(f"missing {path}", file=sys.stderr)
                return 1
            committed = path.read_text()
            if live != committed:
                print(
                    f"{name} disagrees with a fresh PySCF run. "
                    "Regenerate deliberately; do not hand-edit the fixture.",
                    file=sys.stderr,
                )
                return 1
    dump = (FERM / DUMP_NAME).read_text()
    core = [line for line in dump.splitlines() if line.rstrip().endswith("0  0  0  0")]
    if len(core) != 1:
        sys.exit(f"A10: expected one core-energy line, found {len(core)}")
    flipped = dump.replace(core[0], core[0].replace("0.", "1.", 1), 1)
    if flipped == dump:
        sys.exit("A10: changing the core energy left the FCIDUMP equal")
    print(f"OK: {DUMP_NAME} and {REF_NAME} match PySCF")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args == []:
        text = build(FERM)
        print(f"wrote {DUMP_NAME}")
        print(text, end="")
        return 0
    if args == ["--check"]:
        return check()
    print("usage: gen_lih.py [--check]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
