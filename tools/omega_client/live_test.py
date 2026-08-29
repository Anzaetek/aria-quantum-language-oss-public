# SPDX-License-Identifier: Apache-2.0
"""End-to-end check of the template routes against a LIVE omega-server.

Run standalone:  python3 tools/omega_client/live_test.py --url URL --token TOK
CI wraps it in `ci.sh` stage 9's existing server, so no new process management.

# Why the assertions are closed forms

`<Z0>` on `Ry(t)|0>` is `cos(t)` and `d/dt` is `-sin(t)`. Every value here has a
reference from trigonometry rather than from a previous run of this code, so a
server that returned self-consistent nonsense would fail.

# The traps this deliberately avoids

`PLAN-SIX-PROGRAMMES.md` P5 names two:

* **N = 1** — at one row the template is indistinguishable from the existing
  route, so a test there says nothing about the shape. Five rows here.
* **parameters that do not vary per row** — a server binding every row to row 0
  returns plausible IDENTICAL numbers. So the angles vary and the results are
  asserted DISTINCT, which is the only assertion that catches it.

One more, found while writing the gradient route: the gradient columns are
checked with `params` declared in an order that does NOT match the server's
symbol numbering. A request whose column order happens to match the server's
cannot distinguish real alignment from "whatever order the backend returned".
"""
from __future__ import annotations

import argparse
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import omega_client as oc  # noqa: E402


def single_ry_circuit() -> dict:
    """`Ry(theta) q0` alone — one free symbol, closed-form expectation."""
    return {
        "num_qubits": 1, "num_classical_bits": 0, "is_photonic": False,
        "mid_circuit_mode": "Skip", "backend": "Statevector",
        "ops": [{"gate": "Ry", "qubits": [0],
                 "params": [{"symbol": "theta"}],
                 "classical_bit": None, "condition": None}],
    }


def two_ry_circuit() -> dict:
    """`Ry(a) q0; Ry(b) q1` — `<Z0>` depends on `a` only, so the `b` column of
    the gradient must be exactly 0. That zero is what proves the columns are
    aligned to the request rather than to the backend's ordering."""
    return {
        "num_qubits": 2, "num_classical_bits": 0, "is_photonic": False,
        "mid_circuit_mode": "Skip", "backend": "Statevector",
        "ops": [
            {"gate": "Ry", "qubits": [0], "params": [{"symbol": "a"}],
             "classical_bit": None, "condition": None},
            {"gate": "Ry", "qubits": [1], "params": [{"symbol": "b"}],
             "classical_bit": None, "condition": None},
        ],
    }


def check_expectation(client: oc.OmegaClient) -> None:
    thetas = [0.0, 0.4, 1.1, 2.0, math.pi]
    rows = [[t] for t in thetas]
    values = client.expectation_template(single_ry_circuit(), "Z0", ["theta"], rows)
    assert len(values) == len(rows), f"{len(values)} values for {len(rows)} rows"
    for t, v in zip(thetas, values):
        assert abs(v - math.cos(t)) < 1e-9, f"theta={t}: got {v}, want cos={math.cos(t)}"
    # Distinctness: a server binding every row to row 0 returns identical values.
    distinct = {round(v, 12) for v in values}
    assert len(distinct) == len(values), (
        f"only {len(distinct)} distinct values from {len(values)} rows — a bug "
        f"binding every row to the first would look exactly like this"
    )
    print(f"  expectation_template: {len(values)} rows, all match cos(theta), all distinct")


def check_gradient(client: oc.OmegaClient) -> None:
    # params REVERSED relative to declaration order in the circuit: 'b' first.
    # <Z0> does not depend on b, so column 0 must be 0 and column 1 = -sin(a).
    params = ["b", "a"]
    rows = [[0.3, 0.5], [1.2, 0.9], [2.5, 1.7], [0.7, 2.2]]
    grads = client.gradient_template(two_ry_circuit(), "Z0", params, rows)
    assert len(grads) == len(rows)
    for row, g in zip(rows, grads):
        assert len(g) == 2, f"expected 2 columns, got {len(g)}"
        b_val, a_val = row
        assert abs(g[0]) < 1e-9, (
            f"column 0 is d<Z0>/db and must be 0, got {g[0]} — a non-zero value "
            f"means the columns follow the server's symbol order, not `params`"
        )
        assert abs(g[1] + math.sin(a_val)) < 1e-9, (
            f"column 1 is d<Z0>/da = -sin(a); got {g[1]}, want {-math.sin(a_val)}"
        )
    firsts = {round(g[1], 12) for g in grads}
    assert len(firsts) == len(grads), "gradient rows are not distinct"
    print(f"  gradient_template:  {len(grads)} rows, columns aligned to `params`, "
          f"d/db == 0 exactly, all distinct")


def check_refusals(client: oc.OmegaClient) -> None:
    """A circuit with no free symbols must be refused, not silently evaluated."""
    bound = {
        "num_qubits": 1, "num_classical_bits": 0, "is_photonic": False,
        "mid_circuit_mode": "Skip", "backend": "Statevector",
        "ops": [{"gate": "Ry", "qubits": [0], "params": [0.5],
                 "classical_bit": None, "condition": None}],
    }
    try:
        client.expectation_template(bound, "Z0", ["theta"], [[0.1]])
    except oc.OmegaError as e:
        print(f"  refusal:            a bound circuit is refused ({e.status})")
        return
    raise AssertionError("a template with no free symbols must be refused")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", required=True)
    args = ap.parse_args()
    client = oc.OmegaClient(args.url, args.token)
    check_expectation(client)
    check_gradient(client)
    check_refusals(client)
    print("OK: template routes verified end to end against a live server")
    return 0


if __name__ == "__main__":
    sys.exit(main())
