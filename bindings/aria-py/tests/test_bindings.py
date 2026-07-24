# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the pure (non-torch) aria_py bindings."""
import math

import aria_py

SRC1 = "circuit C() { qreg q[1]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0] }"


def test_load_and_symbols():
    m = aria_py.load_source(SRC1, "C")
    assert m.symbols == ["t_0"]
    assert m.num_qubits == 1
    assert m.num_symbols == 1


def test_expectation_and_gradient_match_closed_form():
    m = aria_py.load_source(SRC1, "C")
    for theta in (0.0, 0.7, 1.9, math.pi):
        assert abs(m.expectation([theta], "Z0") - math.cos(theta)) < 1e-12
        assert abs(m.gradient([theta], "Z0")[0] + math.sin(theta)) < 1e-9


def test_batches_match_per_row():
    m = aria_py.load_source(SRC1, "C")
    rows = [[0.0], [1.0], [math.pi], [-0.5]]
    zb = m.expectation_batch(rows, "Z0")
    gb = m.gradient_batch(rows, "Z0")
    for i, r in enumerate(rows):
        assert abs(zb[i] - m.expectation(r, "Z0")) < 1e-12
        assert abs(gb[i][0] - m.gradient(r, "Z0")[0]) < 1e-12


def test_mps_backend_agrees_with_statevector():
    # 2-qubit entangled circuit; MPS at chi=4 (=2^(n/2)) is exact.
    src = ("circuit E() { qreg q[2]\n  let t = symbolic[2]\n"
           "  apply RY(t[0]) on q[0]\n  apply RY(t[1]) on q[1]\n  apply CX on q[0], q[1] }")
    m = aria_py.load_source(src, "E")
    p = [0.6, -1.1]
    sv = m.expectation(p, "Z0 Z1", backend="sv")
    mps = m.expectation(p, "Z0 Z1", backend="mps:4")
    assert abs(sv - mps) < 1e-12


def test_int_params_and_errors():
    src = "circuit P(n: int) { qreg q[n]\n  let t = symbolic[1]\n  apply RY(t[0]) on q[0] }"
    m = aria_py.load_source(src, "P", [("n", 3)])
    assert m.num_qubits == 3
    # wrong param count
    try:
        m.expectation([0.1, 0.2], "Z0")
        assert False, "should reject wrong param count"
    except ValueError as e:
        assert "expected 1 params" in str(e)
    # bad backend
    try:
        m.expectation([0.1], "Z0", backend="nope")
        assert False
    except ValueError as e:
        assert "unknown backend" in str(e)
