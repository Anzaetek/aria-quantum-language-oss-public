# SPDX-License-Identifier: Apache-2.0
"""omega-bridge runner for OpenFermion-FQE — the JW-free fermionic oracle.

Test-only. There is no product backend behind this runner and there must not
be one: FQE is a *reference*, and its value is that it never sees a qubit.

Request (stdin, one JSON object)::

    {"mode": "fermionic-expectation",
     "num_modes": 4,
     "occupied": [0, 2],
     "gates": [{"kind": "givens", "p": 0, "q": 1, "theta": 0.3},
               {"kind": "cphase", "p": 1, "q": 2, "phi": 0.7}],
     "observables": [[[re, im, [[mode, dagger], ...]], ...], ...]}

Response (stdout, ONE JSON line)::

    {"ok": true, "values": [...]}
    {"ok": false, "error": "...", "kind": "..."}

Semantics, pinned to `omega_core::fermion`:

* `givens(p, q, θ)` is `exp(θ (a†_p a_q − a†_q a_p))`. FQE's `time_evolve(t, H)`
  is `exp(−i t H)`, so `H = i a†_p a_q − i a†_q a_p` (Hermitian) and `t = θ`.
  Adjacency is NOT required here — FQE evolves the fermionic operator, so the
  Z string omega's `givens` refuses to emit is implicit. The Rust harness only
  sends adjacent pairs because that is all `Rbs` means; a non-adjacent request
  is answered, not refused, and would disagree with a bare `Rbs` — by design.
* `cphase(p, q, φ)` is `exp(iφ n_p n_q)`: `H = n_p n_q`, `t = −φ`.
* Observables are ladder products read left to right, mode `m` ↦ OpenFermion
  spin-orbital `2m` (alpha). Terms go through `normal_ordered` — an algebraic
  identity, not a qubit mapping.

Spinless register ≡ all-alpha FQE sector: `Wavefunction([[k, k, N]])` with
`k` electrons, `sz = k`, `N` orbitals. FCI strings index the coefficient
vector, bit `p` set ⇔ orbital `p` occupied (`fci_graph.index_alpha`).

`kind` values: `bad-request`, `fqe-not-installed`, `fqe-unsupported-gate`,
`fqe-execute`, `execute`.
"""
from __future__ import annotations

# Protocol guard FIRST: fd 1 is the wire. See runner_io.py.
import runner_io  # noqa: F401  (import has the side effect we want)
from runner_io import emit as _emit, err as _err

import json
import sys


def _fermion_operator(terms, FermionOperator):
    op = FermionOperator()
    for term in terms:
        re, im, product = term
        prod = tuple((2 * int(m), 1 if d else 0) for m, d in product)
        op += FermionOperator(prod, complex(re, im))
    return op


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        _err(f"invalid JSON request: {e}", kind="bad-request")
        return 0
    if not isinstance(req, dict) or req.get("mode") != "fermionic-expectation":
        _err("mode must be 'fermionic-expectation'", kind="bad-request")
        return 0

    try:
        import numpy as np
        import fqe
        from fqe.wavefunction import Wavefunction
        from openfermion import FermionOperator, normal_ordered
    except ImportError as e:  # pragma: no cover - environment
        _err(f"fqe/openfermion not importable: {e}", kind="fqe-not-installed")
        return 0

    try:
        n = int(req["num_modes"])
        occupied = [int(m) for m in req.get("occupied", [])]
        gates = req.get("gates", [])
        observables = req.get("observables", [])
    except (KeyError, TypeError, ValueError) as e:
        _err(f"malformed request: {e}", kind="bad-request")
        return 0
    if n < 1 or len(set(occupied)) != len(occupied) or any(not 0 <= m < n for m in occupied):
        _err(f"bad occupation {occupied} for {n} modes", kind="bad-request")
        return 0

    try:
        k = len(occupied)
        wfn = Wavefunction([[k, k, n]])
        sector = wfn.sector((k, k))
        graph = sector.get_fcigraph()
        vec = np.zeros((graph.lena(), graph.lenb()), dtype=complex)
        bits = 0
        for m in occupied:
            bits |= 1 << m
        vec[graph.index_alpha(bits), 0] = 1.0
        wfn.set_wfn(strategy="from_data", raw_data={(k, k): vec})

        for g in gates:
            kind = g.get("kind")
            p, q = int(g["p"]), int(g["q"])
            if p == q or not (0 <= p < n and 0 <= q < n):
                _err(f"bad gate wires {g}", kind="bad-request")
                return 0
            if kind == "givens":
                theta = float(g["theta"])
                h = FermionOperator(((2 * p, 1), (2 * q, 0)), 1j) + FermionOperator(
                    ((2 * q, 1), (2 * p, 0)), -1j
                )
                wfn = wfn.time_evolve(theta, fqe.get_sparse_hamiltonian(h))
            elif kind == "cphase":
                phi = float(g["phi"])
                h = FermionOperator(((2 * p, 1), (2 * p, 0), (2 * q, 1), (2 * q, 0)), 1.0)
                wfn = wfn.time_evolve(-phi, fqe.get_sparse_hamiltonian(h))
            else:
                _err(f"unsupported gate kind {kind!r}", kind="fqe-unsupported-gate")
                return 0

        values = []
        for terms in observables:
            op = normal_ordered(_fermion_operator(terms, FermionOperator))
            if not op.terms:
                values.append(0.0)
                continue
            v = wfn.expectationValue(fqe.get_sparse_hamiltonian(op))
            v = complex(v)
            if abs(v.imag) > 1e-9:
                _err(f"non-real expectation {v}", kind="fqe-execute")
                return 0
            values.append(float(v.real))
    except Exception as e:  # noqa: BLE001
        _err(f"fqe: {type(e).__name__}: {e}", kind="fqe-execute")
        return 0

    _emit({"ok": True, "values": values})
    return 0


if __name__ == "__main__":
    sys.exit(main())
