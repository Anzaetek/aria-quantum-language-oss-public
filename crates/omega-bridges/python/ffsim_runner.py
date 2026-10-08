# SPDX-License-Identifier: Apache-2.0
"""omega-bridge runner for ffsim — the fermionic differential anchor.

Request (stdin, one JSON object)::

    {"mode": "expectation",
     "qpy_b64": "<base64 QPY blob, one circuit>",
     "observables": [[["ZI", 0.5], ["IZ", -0.5]], ...]}

Response (stdout, ONE JSON line)::

    {"ok": true, "values": [...]}
    {"ok": false, "error": "...", "kind": "..."}

Expectation-only. There is no counts mode: ffsim is a reference for the
fermionic surface (`Rbs`, `CPhase`, Jordan-Wigner observables), not a sampler.

Why QPY and not QASM2: `Rbs` has no QASM2 spelling, and decomposing it into
CX + rotations gives ffsim a circuit it either rejects or routes to a general
simulator. The Rust writer spells `Rbs(θ)` as `XXPlusYYGate(-2θ, π/2)` (the
receipt is on `qpy/write.rs::qiskit_params`); this runner hands that to ffsim
unchanged.

What happens to the circuit — see `ffsim_prep.py`, which is the testable half:
the leading `x` layer becomes a `PrepareSlaterDeterminantSpinlessJW`, `cu3(0,0,λ)`
becomes `cp(λ)`, anything not Hamming-weight preserving is REFUSED with
`ffsim-unsupported-gate`, and the observables go through the same LSB-first
reversal and `SparsePauliOp` path as `qiskit_runner.py`. Only the evolution
differs between the two anchors, which is what makes agreement evidence.

Spinless throughout: omega qubits are modes with no spin structure. ffsim's
spinful API orders its basis differently and is not this lane.

`kind` values emitted here (harvested into `runner.rs::classify_tests`):
`bad-request`, `qpy-parse`, `qpy-multi-circuit`, `ffsim-not-installed`,
`ffsim-unsupported-gate`, `ffsim-not-supported`, `ffsim-execute`, `execute`.
"""

from __future__ import annotations

# Protocol guard FIRST: fd 1 is the wire. See runner_io.py.
import runner_io  # noqa: F401  (import has the side effect we want)
from runner_io import emit as _emit, err as _err

import base64
import io
import json
import sys

import ffsim_prep as prep


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        _err(f"invalid JSON request: {e}", kind="bad-request")
        return 0
    if not isinstance(req, dict):
        _err("request must be a JSON object", kind="bad-request")
        return 0
    mode = req.get("mode", "expectation")
    if mode != "expectation":
        _err(
            f"mode {mode!r}: the ffsim runner is expectation-only (it is a reference, "
            "not a sampler)",
            kind="ffsim-not-supported",
        )
        return 0
    return _expectation(req)


def _expectation(req: dict) -> int:
    qpy_b64 = req.get("qpy_b64")
    obs_in = req.get("observables")
    if not isinstance(qpy_b64, str) or not qpy_b64:
        _err("`qpy_b64` must be a non-empty base64 string", kind="bad-request")
        return 0
    if not isinstance(obs_in, list) or not obs_in:
        _err("`observables` must be a non-empty list", kind="bad-request")
        return 0

    try:
        from qiskit import QuantumCircuit, qpy
        from qiskit.quantum_info import Statevector
    except ImportError as e:
        # Qiskit is an ffsim dependency; the venv is one unit, so one kind.
        _err(f"qiskit import failed (is the ffsim venv built?): {e}", kind="ffsim-not-installed")
        return 0
    try:
        import ffsim
    except ImportError as e:
        _err(f"ffsim import failed: {e}", kind="ffsim-not-installed")
        return 0

    try:
        blob = base64.b64decode(qpy_b64, validate=True)
    except Exception as e:  # noqa: BLE001
        _err(f"`qpy_b64` is not valid base64: {e}", kind="bad-request")
        return 0
    try:
        circuits = qpy.load(io.BytesIO(blob))
    except Exception as e:  # noqa: BLE001
        _err(f"qpy.load: {e}", kind="qpy-parse")
        return 0
    if len(circuits) != 1:
        _err(f"QPY blob holds {len(circuits)} circuits; exactly one is required", kind="qpy-multi-circuit")
        return 0
    circuit = circuits[0]

    # Same refusals, same order, as qiskit_runner._expectation: a reset or a
    # classical condition makes "the expectation of this circuit" ill-defined.
    for instr in circuit.data:
        if instr.operation.name == "reset":
            _err("expectation is undefined for a circuit containing `reset`", kind="ffsim-not-supported")
            return 0
        if getattr(instr.operation, "condition", None) is not None:
            _err("expectation is undefined for a classically-conditioned gate", kind="ffsim-not-supported")
            return 0
    circuit.remove_final_measurements(inplace=True)
    for instr in circuit.data:
        if instr.operation.name == "measure":
            _err("mid-circuit measurement remains after removing terminal measurements", kind="ffsim-not-supported")
            return 0

    n = circuit.num_qubits
    try:
        occupied, body = prep.split_prep_layer(circuit)
        ops = prep.observables_to_ops(obs_in, n)
    except prep.Refusal as e:
        _err(str(e), kind=e.kind)
        return 0

    if not occupied:
        # Vacuum: every Hamming-weight-preserving gate acts on |0…0> as at
        # most a global phase, so the state is |0…0> and ffsim has nothing to
        # do. Answered directly rather than asking ffsim to prepare a
        # zero-electron determinant.
        state = Statevector.from_label("0" * n)
    else:
        try:
            fcirc = QuantumCircuit(n)
            fcirc.append(ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(n, occupied), range(n))
            fcirc.compose(prep.build_body_circuit(n, body), inplace=True)
        except Exception as e:  # noqa: BLE001
            _err(f"building the ffsim circuit: {e}", kind="ffsim-execute")
            return 0
        try:
            result = ffsim.qiskit.final_state_vector(fcirc)
        except (ValueError, NotImplementedError, TypeError, KeyError) as e:
            # ffsim's simulator raises on a gate it does not evolve. That is a
            # refusal, and reported as one — not a defect, and not a green.
            _err(f"ffsim.qiskit.final_state_vector refused the circuit: {e}", kind="ffsim-unsupported-gate")
            return 0
        except Exception as e:  # noqa: BLE001
            _err(f"ffsim.qiskit.final_state_vector: {e}", kind="ffsim-execute")
            return 0
        # ffsim returns its own StateVector (vec in the number-conserving
        # basis, plus norb / nelec); expand it to the 2^n Qiskit-ordered
        # vector so the observable path is byte-for-byte the qiskit runner's.
        vec = getattr(result, "vec", result)
        norb = getattr(result, "norb", n)
        nelec = getattr(result, "nelec", len(occupied))
        try:
            full = ffsim.qiskit.ffsim_vec_to_qiskit_vec(vec, norb=norb, nelec=nelec)
            state = Statevector(full)
        except Exception as e:  # noqa: BLE001
            _err(f"ffsim_vec_to_qiskit_vec: {e}", kind="ffsim-execute")
            return 0

    values = []
    for op in ops:
        try:
            values.append(float(state.expectation_value(op).real))
        except Exception as e:  # noqa: BLE001
            _err(f"expectation_value: {e}", kind="execute")
            return 0
    _emit({"ok": True, "values": values})
    return 0


if __name__ == "__main__":
    sys.exit(main())
