# SPDX-License-Identifier: Apache-2.0
"""Pure preprocessing for the ffsim runner — everything that can be checked
WITHOUT ffsim installed.

Split out of `ffsim_runner.py` for two reasons. First, `ffsim_runner` must
import `runner_io` before anything else, and `runner_io` re-points file
descriptor 1 at import time — so importing the runner under pytest would
hijack pytest's own stdout. Second, ffsim was not installable on the host
that wrote this (PyPI unreachable through the local proxy's self-signed CA,
see PLAN-OPEN-20260825.md §3c.0e item 2), so the parts that only need Qiskit
had to be testable on their own or they would have shipped untested.

Only Qiskit is imported here, and lazily.

## The circuit shape ffsim can evolve

A Qiskit circuit (decoded from the QPY blob the Rust writer produced) is
accepted when it is

1. an optional LEADING layer of `x` gates — the occupation. "Leading" is
   per wire: an `x` counts iff no earlier gate acted on that qubit, whatever
   its position in `circuit.data` (a DAG round-trip such as
   `remove_final_measurements` may float it past commuting gates). `x` on a
   qubit toggles its occupancy, so two `x` on the same wire un-occupy it.
   This becomes ffsim's `PrepareSlaterDeterminantSpinlessJW`; then
2. only Hamming-weight-preserving gates ffsim recognises as such:
   `xx_plus_yy`, `cp`, `swap`, `p` / `u1`, `rz`, `rzz`, and `cu3` with
   θ = φ = 0, which IS `cp(λ)` (qiskit 2.5.2 `Operator` delta 0.0) and is
   how omega spells CPhase. `barrier` is dropped.

Anything else — an `x` after the occupation layer, `h`, `cx`, `ry`, a `cu3`
with a non-zero θ or φ — is REFUSED (`Refusal`, kind `ffsim-unsupported-gate`)
rather than decomposed. Decomposing would produce a circuit ffsim either
rejects or routes to a general simulator, and the whole point of this anchor
is that it is not one.

Unbound parameters are a caller error (`bad-request`): the Rust side is
expected to bind before export.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

# Qiskit instruction names (``Instruction.name``) ffsim evolves natively, and
# the class each is rebuilt with after the prep layer is peeled off.
HWP_GATES = ("xx_plus_yy", "cp", "swap", "p", "u1", "rz", "rzz")


class Refusal(Exception):
    """A typed refusal. `kind` follows the docs/BRIDGES.md suffix contract."""

    def __init__(self, msg: str, kind: str) -> None:
        super().__init__(msg)
        self.kind = kind


Body = List[Tuple[str, Tuple[int, ...], Tuple[float, ...]]]


def _concrete(name: str, params) -> Tuple[float, ...]:
    out = []
    for p in params:
        if getattr(p, "parameters", None):
            raise Refusal(
                f"gate {name!r} has an unbound parameter {p}; bind before export",
                kind="bad-request",
            )
        try:
            out.append(float(p))
        except (TypeError, ValueError) as e:
            raise Refusal(f"gate {name!r} parameter {p!r} is not a real number: {e}", kind="bad-request")
    return tuple(out)


def split_prep_layer(circuit) -> Tuple[List[int], Body]:
    """Peel the leading `x` layer off `circuit`.

    Returns `(occupied, body)`: the sorted list of occupied qubit indices and
    the remaining gates as `(name, qubit_indices, params)` with every gate
    already validated against `HWP_GATES` and `cu3(0,0,λ)` rewritten to
    `cp(λ)`. Raises `Refusal` on anything ffsim cannot evolve.
    """
    occupied: set = set()
    body: Body = []
    # "Leading" is judged PER WIRE, not by position in `circuit.data`: an `x`
    # is occupation iff nothing before it acted on that qubit. Positional
    # peeling is wrong because Qiskit's `remove_final_measurements` (and any
    # DAG round-trip) may topologically reorder commuting gates, floating an
    # `x` on an untouched wire past a body gate on other wires. Physically
    # the two orders are the same state, so they must be accepted alike.
    touched: set = set()
    for instr in circuit.data:
        op = instr.operation
        name = op.name
        qubits = tuple(circuit.find_bit(q).index for q in instr.qubits)
        if name == "barrier":
            continue
        if name == "x":
            if qubits[0] not in touched:
                occupied ^= {qubits[0]}
                continue
            raise Refusal(
                f"x on qubit {qubits[0]} after the leading occupation layer breaks "
                "particle-number conservation; ffsim cannot evolve it",
                kind="ffsim-unsupported-gate",
            )
        touched.update(qubits)
        params = _concrete(name, op.params)
        if name == "cu3":
            theta, phi, lam = params
            if theta == 0.0 and phi == 0.0:
                body.append(("cp", qubits, (lam,)))
                continue
            raise Refusal(
                f"cu3({theta}, {phi}, {lam}) is not Hamming-weight preserving unless "
                "theta = phi = 0 (which is cp(lambda)); ffsim cannot evolve it",
                kind="ffsim-unsupported-gate",
            )
        if name in HWP_GATES:
            body.append((name, qubits, params))
            continue
        raise Refusal(
            f"gate {name!r} is not one ffsim evolves natively ({', '.join(HWP_GATES)}, "
            "cu3 with theta = phi = 0); it is either not Hamming-weight preserving or "
            "not recognisable as such",
            kind="ffsim-unsupported-gate",
        )
    return sorted(occupied), body


def build_body_circuit(num_qubits: int, body: Body):
    """Rebuild the post-prep gates as a fresh `QuantumCircuit` (no prep gate).

    The caller prepends the ffsim state-preparation gate; keeping that out of
    here is what lets this function run where ffsim is absent.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import (
        CPhaseGate,
        PhaseGate,
        RZGate,
        RZZGate,
        SwapGate,
        XXPlusYYGate,
    )

    classes = {
        "xx_plus_yy": XXPlusYYGate,
        "cp": CPhaseGate,
        "swap": SwapGate,
        "p": PhaseGate,
        "u1": PhaseGate,
        "rz": RZGate,
        "rzz": RZZGate,
    }
    qc = QuantumCircuit(num_qubits)
    for name, qubits, params in body:
        qc.append(classes[name](*params), list(qubits))
    return qc


def observables_to_ops(obs_in: Sequence, num_qubits: int) -> list:
    """Wire observables → `SparsePauliOp`s.

    Same wire, same reversal, same pin as `qiskit_runner._expectation`: each
    observable is a list of `[pauli_string, coefficient]` terms, the string is
    DENSE and LSB-FIRST (leftmost char = qubit 0), and Qiskit's
    `SparsePauliOp` is MSB-first, so every string is reversed. Duplicated
    rather than imported because `qiskit_runner` inlines it — and because
    the two runners sharing this code is exactly what makes their agreement
    say something about the EVOLUTION and nothing else.
    """
    from qiskit.quantum_info import SparsePauliOp

    ops = []
    for obs in obs_in:
        terms = []
        for term in obs:
            try:
                pauli, coeff = term[0], float(term[1])
            except Exception as e:  # noqa: BLE001
                raise Refusal(f"malformed observable term {term!r}: {e}", kind="bad-request")
            if not isinstance(pauli, str) or len(pauli) != num_qubits or set(pauli) - set("IXYZ"):
                raise Refusal(
                    f"pauli {pauli!r} must be {num_qubits} chars over IXYZ (dense, LSB-first)",
                    kind="bad-request",
                )
            terms.append((pauli[::-1], coeff))
        ops.append(SparsePauliOp.from_list(terms))
    return ops


def _popcount(a):
    import numpy as np

    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(a).astype(np.int64)
    bytes_ = np.ascontiguousarray(a, dtype=np.uint64).view(np.uint8).reshape(a.size, 8)
    return np.unpackbits(bytes_, axis=1).sum(axis=1).astype(np.int64)


def sector_expectation(vec, strings, op) -> complex:
    """`<vec|op|vec>` read out inside a fixed-particle-number basis — no 2^n
    expansion, so the runner's ceiling is the sector's dimension, not the
    wire count.

    `vec[a]` is the amplitude at basis address `a`; `strings[a]` is that
    address's occupation bitmask with bit `i` = wire `i` (ffsim's
    `addresses_to_strings` order, which is also Qiskit's little-endian
    computational basis, so no reordering happens on the way in). `op` is a
    `SparsePauliOp`; `paulis.x[k][i]` / `paulis.z[k][i]` refer to wire `i`.

    A Pauli string is a bit-flip on its X/Y wires followed by a phase from its
    Z/Y wires: `P|s> = i^{#Y} (-1)^{|s & zmask|} |s ^ xmask>`. The image has
    the same weight iff the flip clears as many bits as it sets, so a term
    whose X/Y wires do not straddle the occupation evenly contributes exactly
    0 — that is the Jordan–Wigner reading of "changes the particle number",
    and it is what the dense path spent 2^n amplitudes computing. Terms that
    stay are `Σ_s conj(vec[addr(s ^ xmask)]) · phase(s) · vec[s]`.
    """
    import numpy as np

    vec = np.asarray(vec, dtype=np.complex128).reshape(-1)
    strings = np.asarray(strings, dtype=np.uint64).reshape(-1)
    if vec.shape != strings.shape:
        raise Refusal(
            f"sector readout: {vec.size} amplitudes for {strings.size} basis strings",
            kind="ffsim-execute",
        )
    order = np.argsort(strings, kind="stable")
    sorted_strings = strings[order]
    total = 0j
    for pauli, coeff in zip(op.paulis, op.coeffs):
        if int(pauli.phase) != 0:
            # SparsePauliOp folds phases into `coeffs`; a bare phase here would
            # mean a convention this reader does not model. Refuse, don't guess.
            raise Refusal(f"sector readout: Pauli {pauli} carries a phase", kind="ffsim-execute")
        x = np.asarray(pauli.x, dtype=bool)
        z = np.asarray(pauli.z, dtype=bool)
        wires = np.arange(x.size, dtype=np.uint64)
        one = np.uint64(1)
        xmask = np.bitwise_or.reduce(np.left_shift(one, wires[x]), initial=np.uint64(0))
        zmask = np.bitwise_or.reduce(np.left_shift(one, wires[z]), initial=np.uint64(0))
        n_x = int(np.count_nonzero(x))
        if n_x % 2:
            continue  # an odd number of flips never preserves the weight
        keep = _popcount(strings & xmask) * 2 == n_x
        if not keep.any():
            continue
        src = strings[keep]
        targets = src ^ xmask
        pos = np.searchsorted(sorted_strings, targets)
        pos_clamped = np.minimum(pos, sorted_strings.size - 1)
        if not np.array_equal(sorted_strings[pos_clamped], targets):
            raise Refusal(
                "sector readout: a weight-preserving flip landed outside the basis",
                kind="ffsim-execute",
            )
        addr = order[pos_clamped]
        sign = 1.0 - 2.0 * (_popcount(src & zmask) & 1)
        phase = 1j ** int(np.count_nonzero(x & z))
        total += complex(coeff) * phase * np.sum(np.conj(vec[addr]) * sign * vec[keep])
    return complex(total)
