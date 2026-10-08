#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate conventions.json from ffsim, and fail if the committed file drifts.

PLAN-FERMIONICQASM F4.0. There is no FermionicQASM implementation to round-trip
through, so a fixture written by hand would be the writer-validated-by-its-own-
reader trap: F4.2 would lower with whatever sign this file claimed, and F4.3
would "confirm" it. Every amplitude below is a value ffsim returned.
qiskit-cold-atom is not executed (deprecated; may not install). Its spin-block
order is quoted from a named commit, read as source.

Run from anywhere with the ffsim venv:

    crates/omega-bridges/python/.venv-ffsim/bin/python \\
        crates/omega-parser/tests/fixtures/fermionicqasm/gen_conventions.py

    ... gen_conventions.py --check

`--check` is what ci.sh runs. It regenerates in memory, diffs against the
committed file, and then perturbs one matrix entry in memory to prove the
comparison notices. Re-running the writer on a clean tree must be a no-op.
A diff means ffsim changed: bump the recorded version deliberately and
regenerate. Never hand-edit conventions.json.
"""

from __future__ import annotations

import copy
import difflib
import importlib.metadata
import json
import pathlib
import sys

import ffsim
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[5]
OUT = pathlib.Path(__file__).resolve().parent / "conventions.json"
GENERATOR = "crates/omega-parser/tests/fixtures/fermionicqasm/gen_conventions.py"

# Stimulus passed into ffsim. Not a convention; the matrices are.
THETA = 0.7

# qiskit-cold-atom @ ad8893f7b0c2d167f9295df10873a23ccfa55295 (2024-02-14),
# https://github.com/qiskit-community/qiskit-cold-atom — read, not installed.
COLD_ATOM_COMMIT = "ad8893f7b0c2d167f9295df10873a23ccfa55295"

REQUIRED = (
    "apply_givens_rotation",
    "apply_num_interaction",
    "apply_num_num_interaction",
    "apply_num_op_prod_interaction",
    "apply_on_site_interaction",
    "apply_tunneling_interaction",
    "addresses_to_strings",
    "dim",
    "number_operator",
)


def qnum(x: float) -> float:
    """Round an ffsim amplitude for a stable JSON encoding.

    12 decimal places sits well above the 1e-9 F4.3 tolerance and flushes
    the 1e-16 leftovers of a unitary expm. The sign of zero is flushed so
    `-0.0` and `0.0` cannot make two runs differ.
    """
    y = round(float(x), 12)
    if y == 0.0:
        return 0.0
    return y


def cpair(z: complex) -> list[float]:
    return [qnum(np.real(z)), qnum(np.imag(z))]


def require_symbols() -> None:
    missing = [name for name in REQUIRED if not hasattr(ffsim, name)]
    if not hasattr(ffsim.qiskit, "ffsim_vec_to_qiskit_vec"):
        missing.append("qiskit.ffsim_vec_to_qiskit_vec")
    if not hasattr(ffsim.qiskit, "jordan_wigner"):
        missing.append("qiskit.jordan_wigner")
    if missing:
        raise SystemExit(
            "ffsim is missing "
            + ", ".join(missing)
            + "; refusing to invent the convention those calls would have pinned"
        )


def particle_sectors(norb: int, spinful: bool) -> list[int | tuple[int, int]]:
    if not spinful:
        return list(range(norb + 1))
    return [(na, nb) for na in range(norb + 1) for nb in range(norb + 1)]


def nelec_json(nelec: int | tuple[int, int]) -> int | list[int]:
    if isinstance(nelec, int):
        return nelec
    return [int(nelec[0]), int(nelec[1])]


def nelec_arg(stored: int | list[int]) -> int | tuple[int, int]:
    if isinstance(stored, int):
        return stored
    return (stored[0], stored[1])


def basis_of(norb: int, spinful: bool) -> list[dict]:
    """Pair each Fock basis vector with the index ffsim's Qiskit embedding uses."""
    n_qubits = norb if not spinful else 2 * norb
    dim_full = 1 << n_qubits
    rows: list[dict] = []
    for nelec in particle_sectors(norb, spinful):
        d = int(ffsim.dim(norb, nelec))
        labels = ffsim.addresses_to_strings(
            range(d),
            norb,
            nelec,
            bitstring_type=ffsim.BitstringType.STRING,
        )
        for addr in range(d):
            vec = np.zeros(d, dtype=complex)
            vec[addr] = 1.0
            full = ffsim.qiskit.ffsim_vec_to_qiskit_vec(vec, norb=norb, nelec=nelec)
            nz = np.flatnonzero(np.abs(full) > 0.5)
            if len(nz) != 1 or abs(full[int(nz[0])] - 1.0) > 1e-12:
                raise RuntimeError(
                    f"ffsim_vec_to_qiskit_vec did not place address {addr} "
                    f"of nelec={nelec} as a one-hot"
                )
            rows.append(
                {
                    "qiskit_index": int(nz[0]),
                    "nelec": nelec_json(nelec),
                    "ffsim_address": addr,
                    "bitstring": str(labels[addr]),
                }
            )
    rows.sort(key=lambda r: r["qiskit_index"])
    got = [r["qiskit_index"] for r in rows]
    if got != list(range(dim_full)):
        raise RuntimeError(
            f"ffsim's index map for norb={norb} spinful={spinful} "
            f"does not cover 0..{dim_full - 1}"
        )
    return rows


def matrix_of(norb: int, basis: list[dict], apply) -> list[list[list[float]]]:
    """Column j is the statevector ffsim returned for basis state j."""
    dim_full = len(basis)
    columns: list[np.ndarray | None] = [None] * dim_full
    for rec in basis:
        nelec = nelec_arg(rec["nelec"])
        d = int(ffsim.dim(norb, nelec))
        vec = np.zeros(d, dtype=complex)
        vec[rec["ffsim_address"]] = 1.0
        out = apply(vec, nelec)
        if getattr(out, "shape", None) != (d,):
            raise RuntimeError(
                f"ffsim returned shape {getattr(out, 'shape', None)} "
                f"for a sector of dimension {d}"
            )
        full = ffsim.qiskit.ffsim_vec_to_qiskit_vec(out, norb=norb, nelec=nelec)
        columns[rec["qiskit_index"]] = full
    if any(col is None for col in columns):
        raise RuntimeError("a Fock basis vector was never passed to ffsim")
    return [
        [cpair(columns[c][r]) for c in range(dim_full)]  # type: ignore[index]
        for r in range(dim_full)
    ]


def spin_kw(spin: ffsim.Spin | None) -> dict:
    if spin is None:
        return {}
    return {"spin": spin}


def spin_txt(spin: ffsim.Spin | None) -> str:
    if spin is None:
        return ""
    return f", spin={spin_name(spin)}"


def spin_name(spin: ffsim.Spin) -> str:
    return {
        ffsim.Spin.ALPHA: "ffsim.Spin.ALPHA",
        ffsim.Spin.BETA: "ffsim.Spin.BETA",
        ffsim.Spin.ALPHA_AND_BETA: "ffsim.Spin.ALPHA_AND_BETA",
    }[spin]


def gate_num(norb: int, orb: int, spin: ffsim.Spin | None, key: str, note: str) -> dict:
    call = (
        f"ffsim.apply_num_interaction(vec, {THETA}, {orb}, "
        f"norb={norb}, nelec=<sector>{spin_txt(spin)})"
    )

    def apply(vec, nelec):
        return ffsim.apply_num_interaction(
            vec, THETA, orb, norb=norb, nelec=nelec, **spin_kw(spin)
        )

    return {"key": key, "call": call, "note": note, "apply": apply}


def gate_numnum(norb: int, orbs: tuple[int, int], spin: ffsim.Spin | None, key: str, note: str) -> dict:
    p, q = orbs
    call = (
        f"ffsim.apply_num_num_interaction(vec, {THETA}, ({p}, {q}), "
        f"norb={norb}, nelec=<sector>{spin_txt(spin)})"
    )

    def apply(vec, nelec):
        return ffsim.apply_num_num_interaction(
            vec, THETA, (p, q), norb=norb, nelec=nelec, **spin_kw(spin)
        )

    return {"key": key, "call": call, "note": note, "apply": apply}


def gate_givens(norb: int, orbs: tuple[int, int], spin: ffsim.Spin | None, key: str, note: str) -> dict:
    p, q = orbs
    call = (
        f"ffsim.apply_givens_rotation(vec, {THETA}, ({p}, {q}), "
        f"norb={norb}, nelec=<sector>{spin_txt(spin)}, phi=0.0)"
    )

    def apply(vec, nelec):
        return ffsim.apply_givens_rotation(
            vec, THETA, (p, q), norb=norb, nelec=nelec, phi=0.0, **spin_kw(spin)
        )

    return {"key": key, "call": call, "note": note, "apply": apply}


def gate_tunnel(norb: int, orbs: tuple[int, int], spin: ffsim.Spin | None, key: str, note: str) -> dict:
    p, q = orbs
    call = (
        f"ffsim.apply_tunneling_interaction(vec, {THETA}, ({p}, {q}), "
        f"norb={norb}, nelec=<sector>{spin_txt(spin)})"
    )

    def apply(vec, nelec):
        return ffsim.apply_tunneling_interaction(
            vec, THETA, (p, q), norb=norb, nelec=nelec, **spin_kw(spin)
        )

    return {"key": key, "call": call, "note": note, "apply": apply}


def gate_onsite(norb: int, orb: int, key: str, note: str) -> dict:
    call = (
        f"ffsim.apply_on_site_interaction(vec, {THETA}, {orb}, "
        f"norb={norb}, nelec=<sector>)"
    )

    def apply(vec, nelec):
        return ffsim.apply_on_site_interaction(vec, THETA, orb, norb=norb, nelec=nelec)

    return {"key": key, "call": call, "note": note, "apply": apply}


def gate_numprod(norb: int, alpha: list[int], beta: list[int], key: str, note: str) -> dict:
    call = (
        f"ffsim.apply_num_op_prod_interaction(vec, {THETA}, "
        f"target_orbs=({alpha}, {beta}), norb={norb}, nelec=<sector>)"
    )

    def apply(vec, nelec):
        return ffsim.apply_num_op_prod_interaction(
            vec, THETA, target_orbs=(alpha, beta), norb=norb, nelec=nelec
        )

    return {"key": key, "call": call, "note": note, "apply": apply}


def spinless_gates(norb: int) -> list[dict]:
    gates = [
        gate_num(
            norb, 0, None, f"spinless_{norb}/num_orb0",
            "FermionicQASM num on mode 0. Spinless: nelec is an int, and the spin argument is not passed.",
        ),
        gate_num(
            norb, norb - 1, None, f"spinless_{norb}/num_orb{norb - 1}",
            "FermionicQASM num on the last mode. Pins which orbital takes the phase.",
        ),
        gate_numnum(
            norb, (0, 1), None, f"spinless_{norb}/numnum_0_1",
            "FermionicQASM numnum on modes 0, 1.",
        ),
        gate_givens(
            norb, (0, 1), None, f"spinless_{norb}/givens_0_1",
            "FermionicQASM givens on modes 0, 1. phi is passed as 0.0 (the one-parameter gate).",
        ),
        gate_tunnel(
            norb, (0, 1), None, f"spinless_{norb}/tunnel_0_1",
            "FermionicQASM tunnel on modes 0, 1. The sign is whatever this call returns.",
        ),
    ]
    if norb == 2:
        gates.extend(
            [
                gate_numnum(
                    norb, (1, 0), None, f"spinless_{norb}/numnum_1_0",
                    "Same call with the orbital pair reversed, so pair order is pinned rather than assumed.",
                ),
                gate_givens(
                    norb, (1, 0), None, f"spinless_{norb}/givens_1_0",
                    "Givens with the orbital pair reversed. (0, 1) and (1, 0) are different calls.",
                ),
                gate_tunnel(
                    norb, (1, 0), None, f"spinless_{norb}/tunnel_1_0",
                    "Tunnel with the orbital pair reversed.",
                ),
            ]
        )
    if norb >= 3:
        gates.extend(
            [
                gate_numnum(
                    norb, (0, 2), None, f"spinless_{norb}/numnum_0_2",
                    "numnum on non-adjacent modes 0, 2. ffsim accepts it; number operators carry no hop.",
                ),
                gate_givens(
                    norb, (0, 2), None, f"spinless_{norb}/givens_0_2",
                    "givens on non-adjacent orbitals 0, 2. ffsim applies it. "
                    "F4.2 still refuses that statement through fermion::givens_expr; "
                    "this matrix is the semantic pin for when a routing lands, not a lowering to ship.",
                ),
                gate_tunnel(
                    norb, (0, 2), None, f"spinless_{norb}/tunnel_0_2",
                    "tunnel on non-adjacent orbitals 0, 2, same status as givens_0_2.",
                ),
            ]
        )
    return gates


def spinful_gates(norb: int) -> list[dict]:
    """2 spatial orbitals → 4 wires. Spin selects a species; it does not mix them."""
    a = ffsim.Spin.ALPHA
    b = ffsim.Spin.BETA
    return [
        gate_num(
            norb, 0, a, f"spinful_{norb}/num_orb0_alpha",
            "num on alpha orbital 0 only. One wire in the first block, not both species.",
        ),
        gate_num(
            norb, 0, b, f"spinful_{norb}/num_orb0_beta",
            "num on beta orbital 0 only. One wire in the second block.",
        ),
        gate_num(
            norb, 1, a, f"spinful_{norb}/num_orb1_alpha",
            "num on alpha orbital 1. Within the alpha block, orbital index 1, not a reversed index.",
        ),
        gate_numnum(
            norb, (0, 1), a, f"spinful_{norb}/numnum_0_1_alpha",
            "numnum inside the alpha species, spatial orbitals 0 and 1. Not the opposite-spin term.",
        ),
        gate_givens(
            norb, (0, 1), a, f"spinful_{norb}/givens_0_1_alpha",
            "givens inside alpha. Spin.ALPHA does not also rotate beta.",
        ),
        gate_givens(
            norb, (0, 1), b, f"spinful_{norb}/givens_0_1_beta",
            "givens inside beta.",
        ),
        gate_tunnel(
            norb, (0, 1), a, f"spinful_{norb}/tunnel_0_1_alpha",
            "tunnel inside alpha.",
        ),
        gate_tunnel(
            norb, (0, 1), b, f"spinful_{norb}/tunnel_0_1_beta",
            "tunnel inside beta.",
        ),
        gate_onsite(
            norb, 0, f"spinful_{norb}/onsite_orb0",
            "Opposite spins, same spatial orbital. This is the ffsim call for numnum "
            "between the two wires of one site, not apply_num_num_interaction.",
        ),
        gate_numprod(
            norb, [0], [1], f"spinful_{norb}/numprod_alpha0_beta1",
            "Opposite spins, different spatial orbitals: alpha 0 with beta 1. "
            "apply_num_op_prod_interaction, not apply_num_num_interaction.",
        ),
    ]


def jordan_wigner_entries() -> dict:
    out: dict = {}

    def add(key: str, call: str, op, norb: int) -> None:
        jw = ffsim.qiskit.jordan_wigner(op, norb=norb)
        terms = []
        for pauli, coeff in zip(jw.paulis, jw.coeffs):
            z_qubits = [i for i, flag in enumerate(pauli.z) if bool(flag)]
            terms.append(
                {
                    "label": pauli.to_label(),
                    "z_qubits": z_qubits,
                    "coeff": cpair(coeff),
                }
            )
        terms.sort(key=lambda t: (t["label"], t["coeff"][0], t["coeff"][1]))
        out[key] = {"call": call, "norb": norb, "terms": terms}

    add(
        "n0_default_spin_norb2",
        "ffsim.qiskit.jordan_wigner(ffsim.number_operator(0), norb=2)",
        ffsim.number_operator(0),
        2,
    )
    for norb in (2, 4):
        add(
            f"n0_alpha_norb{norb}",
            f"ffsim.qiskit.jordan_wigner(ffsim.number_operator(0, spin=ffsim.Spin.ALPHA), norb={norb})",
            ffsim.number_operator(0, spin=ffsim.Spin.ALPHA),
            norb,
        )
        add(
            f"n0_beta_norb{norb}",
            f"ffsim.qiskit.jordan_wigner(ffsim.number_operator(0, spin=ffsim.Spin.BETA), norb={norb})",
            ffsim.number_operator(0, spin=ffsim.Spin.BETA),
            norb,
        )
        add(
            f"n1_alpha_norb{norb}",
            f"ffsim.qiskit.jordan_wigner(ffsim.number_operator(1, spin=ffsim.Spin.ALPHA), norb={norb})",
            ffsim.number_operator(1, spin=ffsim.Spin.ALPHA),
            norb,
        )
    return out


def spin_ordering() -> dict:
    return {
        "status": "pinned",
        "source": {
            "project": "qiskit-cold-atom",
            "url": "https://github.com/qiskit-community/qiskit-cold-atom",
            "commit": COLD_ATOM_COMMIT,
            "commit_date": "2024-02-14",
            "obtained_by": "source read at that commit; the package was not installed and not executed",
        },
        "wire_is_mode": {
            "file": "qiskit_cold_atom/fermions/__init__.py",
            "lines": "19-20",
            "quote": (
                "In this setting, each wire in a quantum circuit describes a fermionic mode. "
                "Upon measurement, each mode can be found to be occupied by a single particle (1) "
                "or be empty (0)."
            ),
        },
        "spin_blocks": {
            "file": "qiskit_cold_atom/fermions/__init__.py",
            "lines": "43-47",
            "quote": (
                "The fermions might also come in several distinguishable species, as is the case when "
                "they carry a spin degree of freedom. In this case, each spatial mode of an experiment "
                "can be occupied by a particle of each spin state. In the circuit description, each "
                "individual mode is assigned its own wire. For example, a system of spin-1/2 fermions "
                "in four spatial modes is described by a circuit with eight wires where the first four "
                "wires denote the spin-up and the last four wires denote the spin-down modes."
            ),
        },
        "wire_index_formula": {
            "file": "qiskit_cold_atom/fermions/base_fermion_backend.py",
            "lines": "80-92",
            "quote": (
                "if initial_state.num_species > 1:\n"
                "    registers = []\n"
                "    for i in range(initial_state.num_species):\n"
                "        registers.append(QuantumRegister(initial_state.sites, f\"spin_{i}\"))\n"
                "    circuit = QuantumCircuit(*registers)\n"
                "...\n"
                "        if occ:\n"
                "            circuit.append(LoadFermions(), qargs=[i * initial_state.sites + j])"
            ),
            "reading": (
                "Species i, site j is wire i * sites + j. The first register is spin_0, "
                "so species 0 occupies the first block of wires."
            ),
        },
        "sequential_not_interleaved": {
            "file": "qiskit_cold_atom/circuit_tools.py",
            "lines": "23-32 and 276-281",
            "quote": (
                "For example, a sequential register [0, 1, 2, 3, 4, 5] with two species implies that "
                "wires 0, 1, 2 are of the same type while an interleaved ordering implies that wires "
                "0, 2, and 4 are of the same type.\n"
                "Possible conventions are \"sequential\", where the first num_sites wires denote the "
                "first species, the second num_sites wires denote the second species etc., and "
                "\"interleaved\", where the first num_species wires denote the first site, the second "
                "num_species wires denote the second site etc."
            ),
        },
        "first_block_is_alpha": {
            "file": "qiskit_cold_atom/fermions/ffsim_backend.py",
            "lines": "365-380 and 631-637",
            "quote": (
                "occ_a, occ_b = set(), set()\n"
                "occupations = [occ_a, occ_b]\n"
                "...\n"
                "spin, orb = divmod(circuit.qubits.index(q), norb)\n"
                "...\n"
                "spin_a, orb_a = divmod(a, norb)\n"
                "spin_b, orb_b = divmod(b, norb)\n"
                "spins = (ffsim.Spin.ALPHA, ffsim.Spin.BETA)"
            ),
            "reading": (
                "Wire index // norb is the species. Species 0 is stored as alpha "
                "(ffsim.Spin.ALPHA) and species 1 as beta. Together with the module text "
                "that the first wires are spin-up, the first block is spin-up / alpha."
            ),
        },
        "adapter_reverses_orbital_index_before_calling_ffsim": {
            "file": "qiskit_cold_atom/fermions/ffsim_backend.py",
            "lines": "378-380 and 392-393",
            "quote": (
                "spin, orb = divmod(circuit.qubits.index(q), norb)\n"
                "# reverse index due to qiskit convention\n"
                "occupations[spin].add(norb - 1 - orb)\n"
                "...\n"
                "# reverse orbitals due to qiskit convention\n"
                "alpha_orbs = [norb - 1 - orb for orb in alpha_orbs]"
            ),
            "applied_to_these_matrices": False,
            "reading": (
                "That reversal lives in qiskit-cold-atom's adapter, on the way into ffsim. "
                "The gate matrices in this file pass orbital indices through unchanged. "
                "ffsim.qiskit.jordan_wigner places number_operator(0, Spin.ALPHA) on z_qubits [0] "
                "and number_operator(0, Spin.BETA) on z_qubits [norb] (see jordan_wigner). "
                "That is the block quote — site j of the first block on wire j — and not the "
                "adapter's norb-1-orb flip. A later phase that copies the flip will disagree "
                "with these matrices."
            ),
        },
    }


def unpinned() -> list[dict]:
    apply_names = sorted(name for name in ffsim.__all__ if name.startswith("apply_"))
    return [
        {
            "id": "cross_spin_givens_and_tunnel",
            "pinned": False,
            "blocks": ["F4.3"],
            "does_not_block": [
                "F4.1",
                "F4.2 same-spin givens, tunnel, num, and numnum",
            ],
            "ffsim_apply_exports": apply_names,
            "reason": (
                "ffsim.apply_givens_rotation and ffsim.apply_tunneling_interaction take a Spin "
                "flag and act inside that species. Neither call mixes an alpha orbital with a "
                "beta orbital. This generator does not synthesize a cross-spin matrix from some "
                "other operator. A FermionicQASM givens or tunnel whose wires sit in different "
                "spin blocks — including the adjacent pair across the block boundary, m[N-1], m[N] "
                "— has no ffsim matrix in this file. F4.3 must not treat same-spin agreement as "
                "covering that case. F4.2 can still lower it through fermion::givens_expr; the "
                "QASM2+rbs lane is the oracle for that lowering, and it is not an ffsim pin."
            ),
        }
    ]


def build() -> dict:
    require_symbols()
    if not (ROOT / GENERATOR).is_file():
        raise SystemExit(f"generator path {GENERATOR} is not under {ROOT}")

    systems: dict = {}
    gates: dict = {}
    catalog = [
        (2, False, spinless_gates(2)),
        (3, False, spinless_gates(3)),
        (4, False, spinless_gates(4)),
        (2, True, spinful_gates(2)),
    ]
    for norb, spinful, specs in catalog:
        name = f"{'spinful' if spinful else 'spinless'}_{norb}"
        basis = basis_of(norb, spinful)
        systems[name] = {
            "norb": norb,
            "spinful": spinful,
            "n_qubits": norb if not spinful else 2 * norb,
            "basis": basis,
        }
        for spec in specs:
            gates[spec["key"]] = {
                "system": name,
                "call": spec["call"],
                "note": spec["note"],
                "matrix": matrix_of(norb, basis, spec["apply"]),
            }

    return {
        "generator": GENERATOR,
        "library": "ffsim",
        "version": importlib.metadata.version("ffsim"),
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": importlib.metadata.version("numpy"),
        "qiskit": importlib.metadata.version("qiskit"),
        "stimulus": {
            "theta": THETA,
            "givens_phi": 0.0,
            "note": (
                "theta and phi are the inputs this generator passes to ffsim. "
                "They are not a convention. The convention is the matrix ffsim returns."
            ),
        },
        "serialization": (
            "Each amplitude is [re, im]. Each part is round(x, 12), and a part that "
            "rounds to zero is written as 0.0 so the sign of zero cannot drift."
        ),
        "assembly": (
            "Column j of a gate matrix is the statevector of computational-basis vector j. "
            "The script builds the one-hot ffsim sector vector at the address whose "
            "ffsim.qiskit.ffsim_vec_to_qiskit_vec image is j, calls the recorded function, "
            "and embeds the result with the same function. matrix[row][col] is the amplitude "
            "of basis state row in that image. Basis bitstrings are ffsim.addresses_to_strings; "
            "qiskit_index is where ffsim_vec_to_qiskit_vec placed the one-hot. "
            "No amplitude under gates or jordan_wigner is computed by this script. "
            "jordan_wigner terms are sorted by Pauli label for a stable encoding; "
            "z_qubits is the indices where that term's qiskit Pauli.z is true."
        ),
        "symbols_checked": [f"ffsim.{name}" for name in REQUIRED]
        + [
            "ffsim.qiskit.ffsim_vec_to_qiskit_vec",
            "ffsim.qiskit.jordan_wigner",
        ],
        "spin_ordering": spin_ordering(),
        "jordan_wigner": jordan_wigner_entries(),
        "systems": systems,
        "gates": gates,
        "not_a_pin": [
            {
                "id": "apply_givens_rotation_docstring_matrix",
                "reason": (
                    "The docstring of ffsim.apply_givens_rotation prints a 4x4 Jordan-Wigner "
                    "matrix whose off-diagonal signs are swapped relative to "
                    "gates['spinless_2/givens_0_1'] in this file (phi=0, basis order "
                    "00, 01, 10, 11 as ffsim.addresses_to_strings and "
                    "ffsim_vec_to_qiskit_vec recorded them). The matrix is the pin. "
                    "Editing the fixture to match the docstring would put the wrong sign "
                    "into F4.2."
                ),
            }
        ],
        "unpinned": unpinned(),
    }


def dumps(doc: dict) -> str:
    return json.dumps(doc, indent=1, allow_nan=False) + "\n"


def tooth(doc: dict) -> None:
    """A hand-edited amplitude must not compare equal. This is the A10 tooth."""
    mutant = copy.deepcopy(doc)
    key = next(iter(mutant["gates"]))
    mutant["gates"][key]["matrix"][0][0][0] += 1.0
    if dumps(mutant) == dumps(doc):
        raise SystemExit(
            "A10: adding 1 to one matrix entry left the fixture unchanged — "
            "the comparison has no teeth"
        )


def main() -> int:
    args = sys.argv[1:]
    if args not in ([], ["--check"]):
        print("usage: gen_conventions.py [--check]", file=sys.stderr)
        return 2
    doc = build()
    text = dumps(doc)
    tooth(doc)
    if args == ["--check"]:
        if not OUT.is_file():
            print(f"missing {OUT}", file=sys.stderr)
            return 1
        have = OUT.read_text()
        if have != text:
            diff = "".join(
                difflib.unified_diff(
                    have.splitlines(keepends=True),
                    text.splitlines(keepends=True),
                    fromfile="conventions.json",
                    tofile="live ffsim",
                    n=1,
                )
            )
            print(diff[:4000], file=sys.stderr)
            print(
                "conventions.json disagrees with live ffsim. "
                "Regenerate deliberately; do not hand-edit the fixture.",
                file=sys.stderr,
            )
            return 1
        print(
            f"OK: conventions.json matches ffsim {doc['version']} "
            f"({len(doc['gates'])} gates); a one-entry perturbation is detected"
        )
        return 0
    OUT.write_text(text)
    print(
        f"wrote {OUT.relative_to(ROOT)} "
        f"({len(doc['gates'])} gates, ffsim {doc['version']}, python {doc['python']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
