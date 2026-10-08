# SPDX-License-Identifier: Apache-2.0
"""fqe_runner.py against closed forms. Skips without fqe (see `make fqe-venv`).

The Rust side (`tests/fqe_vs_statevector.rs`) compares FQE to OUR backend;
this file pins that the runner's OWN reading of `givens`/`cphase` is the one
`omega_core::fermion` documents, independent of anything in this repo.
"""
import json
import math
import os
import subprocess
import sys

import pytest

fqe = pytest.importorskip("fqe")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "..", "fqe_runner.py")


def run(req):
    p = subprocess.run(
        [sys.executable, RUNNER],
        input=json.dumps(req).encode(),
        capture_output=True,
        check=False,
    )
    lines = [line for line in p.stdout.decode().splitlines() if line.strip()]
    assert lines, p.stderr.decode()
    return json.loads(lines[-1])


def n(m):
    return [[1.0, 0.0, [[m, True], [m, False]]]]


def hop(p, q):
    return [[1.0, 0.0, [[p, True], [q, False]]], [1.0, 0.0, [[q, True], [p, False]]]]


def test_givens_rotates_occupation_with_the_documented_sign():
    theta = 0.3
    # |1 0> --givens(0,1,θ)--> cos θ |1 0> − sin θ |0 1>:
    # exp(θ(a†0 a1 − a†1 a0)) with a†0 a1 |10> = 0 and a†1 a0 |10> = |01>.
    res = run(
        {
            "mode": "fermionic-expectation",
            "num_modes": 2,
            "occupied": [0],
            "gates": [{"kind": "givens", "p": 0, "q": 1, "theta": theta}],
            "observables": [n(0), n(1), hop(0, 1)],
        }
    )
    assert res["ok"], res
    n0, n1, h = res["values"]
    assert n0 == pytest.approx(math.cos(theta) ** 2, abs=1e-12)
    assert n1 == pytest.approx(math.sin(theta) ** 2, abs=1e-12)
    # <a†0 a1 + h.c.> on cos|10> − sin|01> = −2 cos sin = −sin 2θ. The sign is
    # the whole point of this test; `fqe_vs_statevector.rs` pins the same
    # closed form on OUR side.
    assert h == pytest.approx(-math.sin(2 * theta), abs=1e-12)


def test_cphase_is_a_phase_only_on_double_occupation():
    phi = 0.7
    # |1 1 0> --givens(1,2,θ)--> mode 1 partly moves to 2; cphase(0,1,φ) then
    # phases the |1 1 0> component. Read out through the current
    # i(a†1 a2 − a†2 a1), which sees the relative phase.
    theta = 0.4
    base = {
        "mode": "fermionic-expectation",
        "num_modes": 3,
        "occupied": [0, 1],
        "observables": [
            [[0.0, 1.0, [[1, True], [2, False]]], [0.0, -1.0, [[2, True], [1, False]]]],
        ],
    }
    without = run({**base, "gates": [{"kind": "givens", "p": 1, "q": 2, "theta": theta}]})
    with_cp = run(
        {
            **base,
            "gates": [
                {"kind": "givens", "p": 1, "q": 2, "theta": theta},
                {"kind": "cphase", "p": 0, "q": 1, "phi": phi},
            ],
        }
    )
    assert without["ok"] and with_cp["ok"]
    # state after givens: cos θ |110> + sin θ |101>; cphase multiplies |110> by e^{iφ}.
    # current = i(a†1 a2 − h.c.) → <ψ|J|ψ> = 2 Im(conj(c_110) c_101 ...) up to the
    # JW-free sign: closed form is −sin 2θ · sin φ after the phase, 0 before.
    assert without["values"][0] == pytest.approx(0.0, abs=1e-12)
    assert abs(with_cp["values"][0]) == pytest.approx(abs(math.sin(2 * theta) * math.sin(phi)), abs=1e-12)


def test_empty_sector_and_refusals():
    res = run(
        {
            "mode": "fermionic-expectation",
            "num_modes": 3,
            "occupied": [],
            "gates": [{"kind": "givens", "p": 0, "q": 1, "theta": 0.3}],
            "observables": [n(0), hop(0, 1)],
        }
    )
    assert res["ok"] and res["values"] == [0.0, 0.0]
    bad = run({"mode": "fermionic-expectation", "num_modes": 2, "occupied": [0], "gates": [{"kind": "swap", "p": 0, "q": 1}], "observables": []})
    assert not bad["ok"] and bad["kind"] == "fqe-unsupported-gate"
