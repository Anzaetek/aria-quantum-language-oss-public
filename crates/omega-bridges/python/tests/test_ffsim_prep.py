# SPDX-License-Identifier: Apache-2.0
"""The ffsim runner's preprocessing, checked without ffsim.

Run:  crates/omega-bridges/python/.venv-qiskit/bin/python -m pytest \\
        crates/omega-bridges/python/tests/test_ffsim_prep.py

Two layers. The pure functions in `ffsim_prep.py` are exercised directly
(only Qiskit needed). Then the runner itself is driven as a subprocess over
the real protocol: where ffsim is absent that pins the `ffsim-not-installed`
kind — the contract `runner.rs::classify_failure` maps to `Unavailable` — and
where it is present it pins the numbers against `Statevector.from_instruction`
on the same Qiskit circuit, so the test gets stronger the moment the venv
exists rather than needing to be rewritten.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import sys

import pytest

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed in this venv")

HERE = os.path.dirname(os.path.abspath(__file__))
PYDIR = os.path.dirname(HERE)
sys.path.insert(0, PYDIR)

import ffsim_prep as prep  # noqa: E402

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import CU3Gate, XXPlusYYGate  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402


def test_leading_x_layer_is_the_occupation_and_the_rest_is_the_body():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(2)
    qc.append(XXPlusYYGate(-0.6, math.pi / 2), [0, 1])
    qc.cp(0.4, 1, 2)
    qc.swap(0, 2)
    occupied, body = prep.split_prep_layer(qc)
    assert occupied == [0, 2]
    assert [b[0] for b in body] == ["xx_plus_yy", "cp", "swap"]
    assert body[0][1] == (0, 1) and body[0][2] == (-0.6, math.pi / 2)


def test_two_x_on_one_wire_unoccupy_it():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)
    qc.x(0)
    occupied, body = prep.split_prep_layer(qc)
    assert occupied == [1] and body == []


def test_x_reordered_past_body_on_other_wires_is_still_occupation():
    # Regression: remove_final_measurements() DAG round-trips and can float
    # an `x` on an untouched wire past a body gate on other wires. That is
    # the same state, so the peel must be per-wire, not positional.
    qc = QuantumCircuit(4)
    qc.append(XXPlusYYGate(0.7, 0.2), [0, 1])
    qc.x(2)
    qc.x(3)
    occupied, body = prep.split_prep_layer(qc)
    assert occupied == [2, 3]
    assert [name for name, _, _ in body] == ["xx_plus_yy"]


def test_x_after_the_occupation_layer_is_refused_as_unsupported_gate():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.append(XXPlusYYGate(-0.6, math.pi / 2), [0, 1])
    qc.x(1)
    with pytest.raises(prep.Refusal) as e:
        prep.split_prep_layer(qc)
    assert e.value.kind == "ffsim-unsupported-gate"


def test_cu3_with_zero_theta_phi_is_cphase_and_otherwise_refused():
    qc = QuantumCircuit(2)
    qc.append(CU3Gate(0.0, 0.0, 0.7), [0, 1])
    _, body = prep.split_prep_layer(qc)
    assert body == [("cp", (0, 1), (0.7,))]

    bad = QuantumCircuit(2)
    bad.append(CU3Gate(0.1, 0.0, 0.7), [0, 1])
    with pytest.raises(prep.Refusal) as e:
        prep.split_prep_layer(bad)
    assert e.value.kind == "ffsim-unsupported-gate"


@pytest.mark.parametrize("build", [lambda qc: qc.h(0), lambda qc: qc.cx(0, 1), lambda qc: qc.ry(0.3, 1)])
def test_non_hamming_weight_preserving_gates_are_refused(build):
    qc = QuantumCircuit(2)
    qc.x(0)
    build(qc)
    with pytest.raises(prep.Refusal) as e:
        prep.split_prep_layer(qc)
    assert e.value.kind == "ffsim-unsupported-gate"


def test_unbound_parameter_is_a_caller_error():
    from qiskit.circuit import Parameter

    qc = QuantumCircuit(2)
    qc.cp(Parameter("t"), 0, 1)
    with pytest.raises(prep.Refusal) as e:
        prep.split_prep_layer(qc)
    assert e.value.kind == "bad-request"


def test_rebuilt_body_is_the_same_unitary():
    qc = QuantumCircuit(3)
    qc.append(XXPlusYYGate(-0.6, math.pi / 2), [1, 2])
    qc.append(CU3Gate(0.0, 0.0, 0.7), [0, 1])
    qc.swap(0, 2)
    _, body = prep.split_prep_layer(qc)
    rebuilt = prep.build_body_circuit(3, body)
    from qiskit.quantum_info import Operator

    assert Operator(rebuilt).equiv(Operator(qc))


def test_observables_are_lsb_first_on_the_wire():
    # The asymmetric pin: x on q[0] of two qubits. "ZI" (leftmost = qubit 0)
    # must read -1; "IZ" must read +1. Backwards, both flip.
    qc = QuantumCircuit(2)
    qc.x(0)
    state = Statevector.from_instruction(qc)
    z0, z1 = prep.observables_to_ops([[["ZI", 1.0]], [["IZ", 1.0]]], 2)
    assert state.expectation_value(z0).real == pytest.approx(-1.0)
    assert state.expectation_value(z1).real == pytest.approx(+1.0)


def test_malformed_observable_is_bad_request():
    with pytest.raises(prep.Refusal) as e:
        prep.observables_to_ops([[["ZIZ", 1.0]]], 2)
    assert e.value.kind == "bad-request"


def _run_runner(request: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, os.path.join(PYDIR, "ffsim_runner.py")],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout must carry exactly one JSON line, got {proc.stdout!r}"
    return json.loads(lines[0])


def _qpy_b64(qc: QuantumCircuit) -> str:
    from qiskit import qpy

    buf = io.BytesIO()
    qpy.dump(qc, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_runner_protocol_end_to_end():
    """Through the real protocol. Which branch runs depends on the venv, and
    both branches assert something that can fail."""
    theta = 0.3
    qc = QuantumCircuit(2)
    qc.x(0)
    # Rust spells Rbs(theta) as XXPlusYYGate(-2 theta, pi/2) on (q0, q1).
    qc.append(XXPlusYYGate(-2 * theta, math.pi / 2), [0, 1])
    req = {
        "mode": "expectation",
        "qpy_b64": _qpy_b64(qc),
        "observables": [[["ZI", 1.0]], [["IZ", 1.0]], [["XX", 1.0]]],
    }
    resp = _run_runner(req)
    try:
        import ffsim  # noqa: F401
    except ImportError:
        assert resp["ok"] is False, resp
        assert resp["kind"] == "ffsim-not-installed", resp
        return
    assert resp["ok"] is True, resp
    # Closed form from omega's RBS matrix on |q0=1, q1=0>: -sin|01> + cos|10>.
    expect = [-math.cos(2 * theta), math.cos(2 * theta), -math.sin(2 * theta)]
    for got, want in zip(resp["values"], expect):
        assert got == pytest.approx(want, abs=1e-12)
    # And against Qiskit's own evolution of the very same circuit.
    ref = Statevector.from_instruction(qc)
    for got, op in zip(resp["values"], prep.observables_to_ops(req["observables"], 2)):
        assert got == pytest.approx(ref.expectation_value(op).real, abs=1e-12)


def test_runner_refuses_a_hadamard_over_the_protocol():
    qc = QuantumCircuit(2)
    qc.h(0)
    resp = _run_runner({"mode": "expectation", "qpy_b64": _qpy_b64(qc), "observables": [[["ZI", 1.0]]]})
    assert resp["ok"] is False
    # Without ffsim the import refusal comes first; with it, the gate refusal.
    assert resp["kind"] in ("ffsim-not-installed", "ffsim-unsupported-gate"), resp


def test_runner_rejects_counts_mode_as_a_capability_gap():
    resp = _run_runner({"mode": "counts", "qasm": "OPENQASM 2.0;", "shots": 10})
    assert resp["ok"] is False and resp["kind"] == "ffsim-not-supported", resp
