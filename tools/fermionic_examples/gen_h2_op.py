#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fermionic examples whose operators come from OpenFermion.

H2/STO-3G at R = 0.7414 Å, from the MolecularData file OpenFermion ships.

No PySCF. The integrals, the Hartree–Fock energy and the FCI energy are read
out of ``openfermion/testing/data/H2_sto-3g_singlet_0.7414.hdf5``. This script
writes

* ``examples/fermionic/h2_op.txt`` — the fermion operator, OpenFermion's own
  ``str()``, under a ``#`` header;
* ``examples/fermionic/h2_hf.qasm`` — ``x`` on the two occupied spin-orbitals;
* ``examples/fermionic/h2_ground.qasm`` — that prep plus ``exp(-i θ X₀X₁X₂Y₃)``;
* ``examples/fermionic/h2_reference.txt`` — θ, E_HF, E_FCI as printed.

``--check`` regenerates into a temporary directory and diffs. A hand-edited
coefficient fails here.

The ansatz angle is the θ of ``|ψ(θ)⟩ = cos θ |1100⟩ + sin θ |0011⟩`` that
minimises ⟨H⟩. For this Hamiltonian that state is the FCI ground state, which
is why one parameter is exact. The circuit below is that unitary: basis change
of ``X`` to ``Z`` is ``h``, of ``Y`` to ``Z`` is ``sdg`` then ``h``, the
Z-string is a CX ladder into ``rz(2θ)``, and the same gates run backwards.
The script simulates that circuit and refuses to write it if the energy is
not the FCI value.

The same run writes the 2-site Fermi-Hubbard operator at t = 1, U = 4
(``fermi_hubbard``, open boundaries) and its circuit. The number checked
there is the two-electron ground energy, 2 - 2*sqrt(2). With chemical
potential zero the lowest eigenvalue of the whole Fock space is -t = -1,
a single electron in the bonding orbital, and the script refuses to treat
that as the half-filled answer.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import tempfile

import numpy as np
from openfermion.chem import MolecularData
from openfermion.hamiltonians import fermi_hubbard
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import get_fermion_operator

import openfermion

ROOT = pathlib.Path(__file__).resolve().parents[2]
FERM = ROOT / "examples" / "fermionic"
DATA = (
    pathlib.Path(openfermion.__file__).parent
    / "testing"
    / "data"
    / "H2_sto-3g_singlet_0.7414"
)

# Literature print named for this geometry. The file's own energies are
# longer; the 1e-4 check against these digits lives in the CLI test.
LIT_HF = -1.116685
LIT_FCI = -1.137270

# Half-filled 2-site Hubbard, t = 1, U = 4: (U - sqrt(U^2 + 16 t^2)) / 2.
HUBBARD_E0 = 2.0 - 2.0 * math.sqrt(2.0)


def load():
    hdf5 = pathlib.Path(str(DATA) + ".hdf5")
    if not hdf5.is_file():
        sys.exit(f"OpenFermion data file missing: {hdf5}")
    mol = MolecularData(filename=str(DATA))
    fermion = get_fermion_operator(mol.get_molecular_hamiltonian())
    return mol, fermion


def ansatz_angle(fermion) -> tuple[float, float, float]:
    """θ for exp(-i θ X0 X1 X2 Y3) on |1100⟩, plus the two energies.

    OpenFermion's sparse index has qubit 0 as the most significant bit.
    Mode occupation (1,1,0,0) is index 12; (0,0,1,1) is index 3.
    """
    h = get_sparse_operator(fermion, n_qubits=4).toarray()
    hf, doub = 12, 3
    a = float(np.real(h[hf, hf]))
    b = float(np.real(h[doub, doub]))
    g = float(np.real(h[hf, doub]))
    # d/dθ (a cos² + b sin² + 2 g sin cos) = 0 ⇒ tan 2θ = -2g / (b - a).
    two = math.atan2(-2.0 * g, b - a)
    best = None
    for k in range(4):
        theta = (two + k * math.pi) / 2.0
        # Fold into (-π, π].
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
        c, s = math.cos(theta), math.sin(theta)
        energy = a * c * c + b * s * s + 2.0 * g * s * c
        if best is None or energy < best[0]:
            best = (energy, theta)
    assert best is not None
    return best[1], a, best[0]


def fmt(x: float) -> str:
    return format(float(x), ".17g")


def apply1(psi: np.ndarray, q: int, u: np.ndarray) -> np.ndarray:
    out = np.zeros_like(psi)
    bit = 1 << q
    for i, amp in enumerate(psi):
        b = (i >> q) & 1
        base = i & ~bit
        for bp in (0, 1):
            out[base | (bp << q)] += u[bp, b] * amp
    return out


def apply_cx(psi: np.ndarray, c: int, t: int) -> np.ndarray:
    out = np.zeros_like(psi)
    for i, amp in enumerate(psi):
        j = i ^ ((((i >> c) & 1) << t))
        out[j] += amp
    return out


def rz(lam: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * lam), np.exp(0.5j * lam)])


H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S = np.diag([1, 1j])
SDG = np.diag([1, -1j])
X = np.array([[0, 1], [1, 0]], dtype=complex)


def circuit_state(angle: float) -> np.ndarray:
    """Little-endian statevector of h2_ground.qasm. Qubit 0 is bit 0."""
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1
    psi = apply1(psi, 0, X)
    psi = apply1(psi, 1, X)
    psi = apply1(psi, 3, SDG)
    for q in range(4):
        psi = apply1(psi, q, H)
    psi = apply_cx(psi, 0, 1)
    psi = apply_cx(psi, 1, 2)
    psi = apply_cx(psi, 2, 3)
    psi = apply1(psi, 3, rz(angle))
    psi = apply_cx(psi, 2, 3)
    psi = apply_cx(psi, 1, 2)
    psi = apply_cx(psi, 0, 1)
    for q in range(4):
        psi = apply1(psi, q, H)
    psi = apply1(psi, 3, S)
    return psi


def simulate_energy(fermion, angle: float) -> float:
    h = get_sparse_operator(fermion, n_qubits=4).toarray()
    # Our statevector is qubit-0-LSB. OpenFermion's matrix is qubit-0-MSB.
    psi = circuit_state(angle)
    rev = np.zeros_like(psi)
    for i, amp in enumerate(psi):
        bits = f"{i:04b}"[::-1]
        rev[int(bits, 2)] = amp
    return float(np.real(np.vdot(rev, h @ rev)))


def apply_rbs(psi: np.ndarray, q0: int, q1: int, theta: float) -> np.ndarray:
    """Qubit RBS. First qubit is ``q0``, matching ``gates::rbs``.

    ``|01⟩ → cos|01⟩ + sin|10⟩`` and ``|10⟩ → −sin|01⟩ + cos|10⟩``.
    ``|00⟩`` and ``|11⟩`` stay put.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    out = np.zeros_like(psi)
    b0 = 1 << q0
    b1 = 1 << q1
    for i, amp in enumerate(psi):
        on0 = (i & b0) != 0
        on1 = (i & b1) != 0
        if on0 == on1:
            out[i] += amp
            continue
        flipped = i ^ b0 ^ b1
        out[i] += c * amp
        # |01> (q0 empty, q1 occupied) picks up +sin on |10>.
        # |10> picks up -sin on |01>.
        out[flipped] += (s if (not on0 and on1) else -s) * amp
    return out


def hubbard_state() -> np.ndarray:
    """Little-endian state of hubbard2.qasm. Qubit 0 is bit 0."""
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0
    psi = apply1(psi, 0, X)
    psi = apply1(psi, 3, X)
    for q in range(4):
        psi = apply1(psi, q, H)
    psi = apply_cx(psi, 0, 1)
    psi = apply_cx(psi, 1, 2)
    psi = apply_cx(psi, 2, 3)
    psi = apply1(psi, 3, rz(math.pi / 2.0))
    psi = apply_cx(psi, 2, 3)
    psi = apply_cx(psi, 1, 2)
    psi = apply_cx(psi, 0, 1)
    for q in range(4):
        psi = apply1(psi, q, H)
    psi = apply1(psi, 0, S)
    psi = apply_rbs(psi, 1, 3, math.pi / 8.0)
    return psi


def hubbard_operator():
    """Open boundaries. The default ``periodic=True`` would double the bond."""
    return fermi_hubbard(2, 1, tunneling=1.0, coulomb=4.0, periodic=False)


def two_electron_ground(fermion) -> float:
    h = np.real(get_sparse_operator(fermion, n_qubits=4).toarray())
    idx = [i for i in range(16) if bin(i).count("1") == 2]
    return float(np.linalg.eigvalsh(h[np.ix_(idx, idx)])[0])


def fock_ground(fermion) -> float:
    h = np.real(get_sparse_operator(fermion, n_qubits=4).toarray())
    return float(np.linalg.eigvalsh(h)[0])


def hubbard_circuit_energy(fermion) -> float:
    h = get_sparse_operator(fermion, n_qubits=4).toarray()
    psi = hubbard_state()
    rev = np.zeros_like(psi)
    for i, amp in enumerate(psi):
        rev[int(f"{i:04b}"[::-1], 2)] = amp
    return float(np.real(np.vdot(rev, h @ rev)))


def hubbard_files(fermion) -> dict[str, str]:
    op = str(fermion).rstrip() + "\n"
    header = (
        "# 2-site Fermi-Hubbard, t = 1, U = 4, chemical potential 0.\n"
        "# Mode order is site-major, spin-minor:\n"
        "# 0 = site 0 up, 1 = site 0 down, 2 = site 1 up, 3 = site 1 down.\n"
        "# Hopping is written in both orders. The identity is absent (no\n"
        "# constant shift). Open boundaries: fermi_hubbard(..., periodic=False).\n"
        "# The half-filled ground energy is 2 - 2*sqrt(2). The lowest\n"
        "# eigenvalue of this operator on the whole Fock space is -1.\n"
    )
    rbs = fmt(math.pi / 8.0)
    rz_angle = fmt(math.pi / 2.0)
    qasm = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "// 2-site Fermi-Hubbard, t = 1, U = 4, two electrons.\n"
        "// Neel determinant |1001>, then exp(-i (pi/4) X0 X1 X2 X3) and s on\n"
        "// q[0], which prepares the covalent singlet, then rbs(pi/8) on the\n"
        "// spin-down modes q[1], q[3]. Read with --qasm-dialect lenient.\n"
        "// The rbs angle is the closed form for this t and U, not a fit.\n"
        "qreg q[4];\n"
        "x q[0];\n"
        "x q[3];\n"
        "h q[0];\n"
        "h q[1];\n"
        "h q[2];\n"
        "h q[3];\n"
        "cx q[0],q[1];\n"
        "cx q[1],q[2];\n"
        "cx q[2],q[3];\n"
        f"rz({rz_angle}) q[3];\n"
        "cx q[2],q[3];\n"
        "cx q[1],q[2];\n"
        "cx q[0],q[1];\n"
        "h q[0];\n"
        "h q[1];\n"
        "h q[2];\n"
        "h q[3];\n"
        "s q[0];\n"
        f"rbs({rbs}) q[1],q[3];\n"
    )
    return {"hubbard2_op.txt": header + op, "hubbard2.qasm": qasm}


def files(mol, fermion, theta: float) -> dict[str, str]:
    nuc = float(mol.nuclear_repulsion)
    e_hf = float(mol.hf_energy)
    e_fci = float(mol.fci_energy)
    angle = float(fmt(2.0 * theta))
    op = str(fermion).rstrip() + "\n"
    # The same digits OpenFermion printed on `[]`, not a second rounding.
    ident = op.split("[", 1)[0].strip()
    header = (
        "# H2 / STO-3G at R = 0.7414 Angstrom.\n"
        "# Spin-orbital order: 0 = 1 sigma_g up, 1 = 1 sigma_g down,\n"
        "# 2 = 1 sigma_u up, 3 = 1 sigma_u down.\n"
        f"# The identity [] is the nuclear repulsion, {ident} Ha.\n"
        "# Integrals: OpenFermion's shipped MolecularData file\n"
        "# H2_sto-3g_singlet_0.7414 (no PySCF).\n"
    )
    hf_qasm = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "// Hartree-Fock for H2/STO-3G: both electrons in 1 sigma_g.\n"
        "// |1⟩ on modes 0 and 1, |0⟩ on modes 2 and 3.\n"
        "qreg q[4];\n"
        "x q[0];\n"
        "x q[1];\n"
    )
    ground_qasm = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "// Hartree-Fock plus exp(-i theta X0 X1 X2 Y3).\n"
        f"// theta = {fmt(theta)}. One parameter is the FCI state for this molecule.\n"
        "// X to Z is h; Y to Z is sdg then h; the Z string is a CX ladder and rz(2*theta).\n"
        "qreg q[4];\n"
        "x q[0];\n"
        "x q[1];\n"
        "sdg q[3];\n"
        "h q[0];\n"
        "h q[1];\n"
        "h q[2];\n"
        "h q[3];\n"
        "cx q[0],q[1];\n"
        "cx q[1],q[2];\n"
        "cx q[2],q[3];\n"
        f"rz({fmt(angle)}) q[3];\n"
        "cx q[2],q[3];\n"
        "cx q[1],q[2];\n"
        "cx q[0],q[1];\n"
        "h q[0];\n"
        "h q[1];\n"
        "h q[2];\n"
        "h q[3];\n"
        "s q[3];\n"
    )
    reference = (
        "# Printed by tools/fermionic_examples/gen_h2_op.py\n"
        "# from OpenFermion MolecularData H2_sto-3g_singlet_0.7414.\n"
        f"E_HF {fmt(e_hf)}\n"
        f"E_FCI {fmt(e_fci)}\n"
        f"theta {fmt(theta)}\n"
        f"nuclear_repulsion {fmt(nuc)}\n"
    )
    return {
        "h2_op.txt": header + op,
        "h2_hf.qasm": hf_qasm,
        "h2_ground.qasm": ground_qasm,
        "h2_reference.txt": reference,
    }


def write_all(dest: pathlib.Path, blob: dict[str, str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in blob.items():
        (dest / name).write_text(text)


def check(blob: dict[str, str]) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        got = pathlib.Path(tmp)
        write_all(got, blob)
        bad = False
        for name in blob:
            a = (FERM / name).read_bytes()
            b = (got / name).read_bytes()
            if a != b:
                print(f"DIFF {name}", file=sys.stderr)
                bad = True
        if bad:
            print(
                "committed fermionic example files differ from a fresh generation",
                file=sys.stderr,
            )
            return 1
    print("H2 files match the OpenFermion data file")
    print("Hubbard files match fermi_hubbard and the closed-form circuit")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mol, fermion = load()
    theta, e_sub_hf, e_sub_fci = ansatz_angle(fermion)
    e_hf = float(mol.hf_energy)
    e_fci = float(mol.fci_energy)
    if abs(e_sub_hf - e_hf) > 1e-10:
        sys.exit(f"subspace HF {e_sub_hf} != file {e_hf}")
    if abs(e_sub_fci - e_fci) > 1e-10:
        sys.exit(f"ansatz minimum {e_sub_fci} != file FCI {e_fci}")
    if abs(e_hf - LIT_HF) > 1e-4 or abs(e_fci - LIT_FCI) > 1e-4:
        sys.exit(
            f"file energies {e_hf} {e_fci} are outside 1e-4 of the literature "
            f"print {LIT_HF} {LIT_FCI}"
        )
    angle = float(fmt(2.0 * theta))
    simulated = simulate_energy(fermion, angle)
    if abs(simulated - e_fci) > 1e-10:
        sys.exit(f"circuit energy {simulated} != FCI {e_fci}")
    hop = hubbard_operator()
    sector = two_electron_ground(hop)
    if abs(sector - HUBBARD_E0) > 1e-12:
        sys.exit(
            f"two-electron Hubbard ground {sector} != 2 - 2*sqrt(2) = {HUBBARD_E0}"
        )
    unrestricted = fock_ground(hop)
    if abs(unrestricted - (-1.0)) > 1e-12:
        sys.exit(
            f"unrestricted Hubbard ground {unrestricted} is not -t = -1; "
            "refusing to guess which eigenvalue is half filling"
        )
    hop_circuit = hubbard_circuit_energy(hop)
    if abs(hop_circuit - HUBBARD_E0) > 1e-12:
        sys.exit(f"Hubbard circuit energy {hop_circuit} != {HUBBARD_E0}")
    blob = files(mol, fermion, theta)
    blob.update(hubbard_files(hop))
    if args.check:
        return check(blob)
    write_all(FERM, blob)
    print(f"E_HF {fmt(e_hf)}")
    print(f"E_FCI {fmt(e_fci)}")
    print(f"theta {fmt(theta)}")
    print(f"circuit {simulated}")
    print(f"hubbard_two_electron {fmt(sector)}")
    print(f"hubbard_unrestricted {fmt(unrestricted)}")
    print(f"hubbard_circuit {fmt(hop_circuit)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
