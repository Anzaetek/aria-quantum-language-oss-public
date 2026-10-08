#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ffsim arm for the fermionic comparison.

One JSON object per line on stdin, one JSON object per line on stdout.
stderr is free. The timed region is the contraction named by the op, not
the import and not the parse of the request.

Kitaev pairing does not conserve particle number, so
``FermionOperator._linear_operator_`` raises. The n=2 and n=8 values are
ffsim's own ``jordan_wigner`` on that operator, contracted on the spinless
product state (alpha qubits in |+>, beta qubits in |0>, which is where an
alpha-only operator lives). n=64 calls ``linear_operator`` and returns the
refusal. It does not allocate 2^64.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

ROOT_MARKERS = ("examples/fermionic",)


def _emit(obj: dict) -> None:
    sys.stdout.write(__import__("json").dumps(obj) + "\n")
    sys.stdout.flush()


def _versions() -> dict:
    import importlib.metadata as md
    import platform

    def ver(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "missing"

    info = np.show_config(mode="dicts")
    blas = info.get("Build Dependencies", {}).get("blas", {})
    return {
        "ffsim": ver("ffsim"),
        "numpy": ver("numpy"),
        "qiskit": ver("qiskit"),
        "scipy": ver("scipy"),
        "pyscf": ver("pyscf"),
        "python": platform.python_version(),
        "blas": str(blas.get("name", "unknown")),
        "blas_version": str(blas.get("version", "unknown")),
        "cpu_count": int(__import__("os").cpu_count() or 1),
    }


def _repo() -> "pathlib.Path":
    import pathlib

    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if all((parent / m).exists() for m in ROOT_MARKERS):
            return parent
    raise RuntimeError(f"repo root not found above {here}")


def _parse_operator(text: str):
    """OpenFermion ladder spelling, spin-orbital order even=alpha, odd=beta.

    Same reading as ``tools/fermionic_examples/ffsim_h2.py``.
    """
    import ffsim
    from ffsim.operators.fermion_action import cre_a, cre_b, des_a, des_b

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
            if create and spin == 0:
                act = cre_a(spatial)
            elif create:
                act = cre_b(spatial)
            elif spin == 0:
                act = des_a(spatial)
            else:
                act = des_b(spatial)
            ops.append(act)
        key = tuple(ops)
        if not key:
            const += coeff
        else:
            terms[key] = terms.get(key, 0.0) + coeff
        rest = rest[close_b + 1 :]
    return const, ffsim.FermionOperator(terms)


def _energy(vec: np.ndarray, lin, const: float) -> float:
    val = np.vdot(vec, lin @ vec) + const
    if abs(val.imag) > 1e-8:
        raise RuntimeError(f"non-real energy {val}")
    return float(val.real)


def h2() -> dict:
    import ffsim

    ferm = _repo() / "examples" / "fermionic"
    const, op = _parse_operator((ferm / "h2_op.txt").read_text())
    theta = None
    fci = None
    for line in (ferm / "h2_reference.txt").read_text().splitlines():
        if line.startswith("theta "):
            theta = float(line.split()[1])
        elif line.startswith("E_FCI "):
            fci = float(line.split()[1])
    if theta is None or fci is None:
        raise RuntimeError("h2_reference.txt is missing theta or E_FCI")
    norb, nelec = 2, (1, 1)
    t0 = time.perf_counter()
    lin = ffsim.protocols.linear_operator(op, norb=norb, nelec=nelec)
    hf = ffsim.hartree_fock_state(norb, nelec)
    doubles = ffsim.slater_determinant(norb, occupied_orbitals=([1], [1]))
    psi = np.cos(theta) * hf + np.sin(theta) * doubles
    value = _energy(psi, lin, const)
    seconds = time.perf_counter() - t0
    return {
        "ok": True,
        "value": value,
        "seconds": seconds,
        "reference": fci,
        "sector": "nelec=(1,1)",
        "path": "occupation-basis contraction of the operator file, not a replay of the QASM",
    }


def hubbard() -> dict:
    import ffsim

    text = (_repo() / "examples" / "fermionic" / "hubbard2_op.txt").read_text()
    const, op = _parse_operator(text)
    if not op.conserves_particle_number():
        raise RuntimeError("hubbard operator does not conserve particle number")
    norb, nelec = 2, (1, 1)
    t0 = time.perf_counter()
    lin = ffsim.protocols.linear_operator(op, norb=norb, nelec=nelec)

    def det(alpha, beta):
        return ffsim.slater_determinant(norb, occupied_orbitals=(alpha, beta))

    # Symmetric covalent singlet. The antisymmetric combination is the
    # energy-0 state of this matrix; the qubit circuit's relative minus is
    # a Jordan-Wigner phase, and this orbital-basis sign is the one whose
    # energy is 2 - 2*sqrt(2).
    symmetric = (det([0], [1]) + det([1], [0])) / np.sqrt(2)
    ionic = (det([0], [0]) + det([1], [1])) / np.sqrt(2)
    psi = np.cos(np.pi / 8) * symmetric + np.sin(np.pi / 8) * ionic
    value = _energy(psi, lin, const)
    seconds = time.perf_counter() - t0
    return {
        "ok": True,
        "value": value,
        "seconds": seconds,
        "reference": float(2 - 2 * np.sqrt(2)),
        "sector": "nelec=(1,1)",
        "path": "occupation-basis contraction of hubbard2_op.txt",
    }


def _kitaev_operator(n: int):
    import ffsim
    from ffsim.operators.fermion_action import cre, des

    coeffs: dict = {}

    def add(coeff: float, prod):
        key = tuple(cre(False, m) if dagger else des(False, m) for dagger, m in prod)
        coeffs[key] = coeffs.get(key, 0.0) + coeff

    for j in range(n - 1):
        add(-1.0, [(True, j), (False, j + 1)])
        add(-1.0, [(True, j + 1), (False, j)])
        add(1.0, [(False, j), (False, j + 1)])
        add(1.0, [(True, j + 1), (True, j)])
    return ffsim.FermionOperator(coeffs)


def _sector_refusal(op, n: int) -> str:
    import ffsim

    try:
        ffsim.protocols.linear_operator(op, norb=n, nelec=1)
    except ValueError as exc:
        return str(exc)
    raise RuntimeError(
        "ffsim.linear_operator accepted a Kitaev operator; the pairing terms "
        "were supposed to be refused for not conserving particle number"
    )


def kitaev(n: int) -> dict:
    import ffsim
    from qiskit.quantum_info import SparsePauliOp, Statevector

    if n not in (2, 8):
        raise RuntimeError(f"kitaev value is only for n=2 and n=8, got {n}")
    op = _kitaev_operator(n)
    refused = _sector_refusal(op, n)
    t0 = time.perf_counter()
    jw = ffsim.qiskit.jordan_wigner(op, norb=n)
    # SparsePauliOp is little-endian: the rightmost character is qubit 0.
    # Alpha orbitals are qubits 0..n-1, beta are n..2n-1. An alpha-only
    # operator is identity on beta, and |0> on beta with |+> on alpha is
    # the spinless product state.
    beta_hits = 0
    for pauli, _coeff in zip(jw.paulis, jw.coeffs, strict=True):
        label = pauli.to_label()
        # label[0] is the highest qubit.
        beta = label[:n]
        if any(ch != "I" for ch in beta):
            beta_hits += 1
    if beta_hits:
        raise RuntimeError(f"{beta_hits} Pauli strings act on the beta register")
    label = ("0" * n) + ("+" * n)
    state = Statevector.from_label(label)
    val = state.expectation_value(jw)
    seconds = time.perf_counter() - t0
    if abs(val.imag) > 1e-8:
        raise RuntimeError(f"non-real Kitaev energy {val}")
    return {
        "ok": True,
        "value": float(val.real),
        "seconds": seconds,
        "reference": float(-(n - 1)),
        "sector_refused": refused,
        "beta_identity": True,
        "jw_qubits": int(jw.num_qubits),
        "path": "ffsim.qiskit.jordan_wigner, alpha |+>, beta |0>; linear_operator refused",
    }


def kitaev64() -> dict:
    op = _kitaev_operator(64)
    refused = _sector_refusal(op, 64)
    return {
        "ok": False,
        "kind": "ffsim-sector-refused",
        "error": refused,
        "dense_amplitudes": str(2**64),
        "allocated": False,
        "fqe": (
            "FQE's wavefunction is one (n_alpha, n_beta) sector. Kitaev pairing "
            "a_j a_{j+1} changes the particle number, so no sector contains the state."
        ),
    }


_LUCJ = None


def _lucj():
    global _LUCJ
    if _LUCJ is not None:
        return _LUCJ
    import ffsim

    norb, nalpha = 8, 4
    nelec = (nalpha, nalpha)
    pairs = ([(p, p + 1) for p in range(norb - 1)], [(p, p) for p in range(norb)])
    t0 = time.perf_counter()
    op = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=1, interaction_pairs=pairs, seed=0
    )
    op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=op.diag_coulomb_mats * 0.1,
        orbital_rotations=op.orbital_rotations,
        final_orbital_rotation=op.final_orbital_rotation,
    )
    built = time.perf_counter() - t0
    from ffsim.operators.fermion_action import cre_a, des_a

    number = ffsim.FermionOperator({(cre_a(0), des_a(0)): 1.0})
    lin = ffsim.protocols.linear_operator(number, norb=norb, nelec=nelec)
    _LUCJ = (op, norb, nelec, lin, built)
    return _LUCJ


def lucj_prepare() -> dict:
    import ffsim

    _op, norb, nelec, _lin, built = _lucj()
    return {
        "ok": True,
        "fixed_cost_s": built,
        "dim": int(ffsim.dim(norb, nelec)),
        "norb": norb,
        "nelec": [nelec[0], nelec[1]],
        "path": "UCJOpSpinBalanced applied as one fused unitary, QASM not read",
    }


def lucj_expect() -> dict:
    import ffsim

    op, norb, nelec, lin, _built = _lucj()
    t0 = time.perf_counter()
    hf = ffsim.hartree_fock_state(norb, nelec)
    vec = ffsim.apply_unitary(hf, op, norb=norb, nelec=nelec)
    val = np.vdot(vec, lin @ vec)
    seconds = time.perf_counter() - t0
    if abs(val.imag) > 1e-8:
        raise RuntimeError(f"non-real LUCJ expectation {val}")
    return {
        "ok": True,
        "value": float(val.real),
        "seconds": seconds,
        "dim": int(vec.shape[0]),
        "norm": float(np.linalg.norm(vec)),
    }


def floor() -> dict:
    import ffsim
    from ffsim.operators.fermion_action import cre_a, des_a

    t0 = time.perf_counter()
    norb, nelec = 1, (1, 0)
    hf = ffsim.hartree_fock_state(norb, nelec)
    number = ffsim.FermionOperator({(cre_a(0), des_a(0)): 1.0})
    lin = ffsim.protocols.linear_operator(number, norb=norb, nelec=nelec)
    val = np.vdot(hf, lin @ hf)
    seconds = time.perf_counter() - t0
    if abs(float(val.real) - 1.0) > 1e-12:
        raise RuntimeError(f"floor expectation {val}, want 1")
    return {"ok": True, "value": float(val.real), "seconds": seconds}


def main() -> int:
    import json

    versions = _versions()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "hello":
                k2 = kitaev(2)
                _emit({
                    "ok": True,
                    **versions,
                    "self_check_kitaev_n2": k2["value"],
                    "self_check_sector_refused": "particle number" in k2["sector_refused"],
                })
            elif op == "h2":
                _emit(h2())
            elif op == "hubbard":
                _emit(hubbard())
            elif op == "kitaev":
                _emit(kitaev(int(req["n"])))
            elif op == "kitaev64":
                _emit(kitaev64())
            elif op == "lucj_prepare":
                _emit(lucj_prepare())
            elif op == "lucj":
                _emit(lucj_expect())
            elif op == "floor":
                _emit(floor())
            elif op == "quit":
                _emit({"ok": True})
                return 0
            else:
                _emit({"ok": False, "error": f"unknown op {op!r}"})
        except Exception as exc:  # noqa: BLE001
            _emit({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=4),
            })
    return 0


if __name__ == "__main__":
    sys.exit(main())
