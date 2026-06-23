#!/usr/bin/env python3
"""omega-bridge Qiskit runner.

Reads a single JSON request from stdin and writes a single JSON
response to stdout. Designed to be invoked as a subprocess by the
Rust crate `omega-bridges`; the wrapper script
`omega-bridge-qiskit-runner` activates the venv and execs this file.

Request shape:
  {
    "qasm":  "OPENQASM 2.0; ...",   # required, QASM2 source
    "shots": 1024,                   # required, positive int
    "seed":  42,                     # optional, RNG seed for reproducibility
    "noise": {                       # optional, opaque dict; today only
                                     # {"depolarizing": p} is honoured.
      "depolarizing": 0.001
    }
  }

Response shape (success):
  {
    "ok": true,
    "counts": {"00": 512, "11": 512, ...}    # LSB-first bitstrings
  }

Response shape (error):
  {
    "ok": false,
    "error": "<message>",
    "kind": "qiskit-not-installed" | "qasm-parse" | "execute" | "<other>"
  }

The runner exits 0 in both cases — it's the Rust side's job to inspect
`ok`. A non-zero exit code means the runner itself crashed and the
output may be incomplete.
"""

from __future__ import annotations

import json
import sys
import traceback


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _err(msg: str, kind: str = "execute") -> None:
    _emit({"ok": False, "error": msg, "kind": kind})


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        _err("empty stdin", kind="bad-request")
        return 0
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        _err(f"bad JSON request: {e}", kind="bad-request")
        return 0

    # Mode dispatch. Default: QASM2 → counts. Optional:
    # "qpy_to_qasm2" — decode a base64-encoded QPY blob and emit
    # the equivalent QASM2 source so omega-parser can take over.
    # "qasm2_to_qpy" — symmetric: parse QASM2 with `qasm2.loads`
    # then emit the QPY blob via `qpy.dump` for round-trip /
    # interop with downstream Qiskit-only tooling.
    mode = req.get("mode") or "execute"
    if mode == "qpy_to_qasm2":
        return _qpy_to_qasm2(req)
    if mode == "qasm2_to_qpy":
        return _qasm2_to_qpy(req)
    if mode != "execute":
        _err(f"unknown mode {mode!r}", kind="bad-request")
        return 0

    qasm = req.get("qasm")
    shots = req.get("shots")
    if not isinstance(qasm, str) or not qasm.strip():
        _err("`qasm` must be a non-empty string", kind="bad-request")
        return 0
    if not isinstance(shots, int) or shots <= 0:
        _err("`shots` must be a positive integer", kind="bad-request")
        return 0
    seed = req.get("seed")
    noise = req.get("noise") or {}

    try:
        from qiskit import qasm2
        from qiskit_aer import AerSimulator
    except ImportError as e:
        _err(f"qiskit / qiskit-aer not importable: {e}", kind="qiskit-not-installed")
        return 0

    try:
        # Qiskit 2.x's qasm2 loader does not auto-expose the qelib1.inc
        # gate set — `swap`, `crz`, etc. are only resolved if we pass
        # the LEGACY_* config. The verify-qiskit corpus uses the full
        # qelib1 surface, so we wire the legacy hooks here.
        circuit = qasm2.loads(
            qasm,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )
    except Exception as e:  # noqa: BLE001 — surface Qiskit's own error string
        _err(f"qasm2.loads: {e}", kind="qasm-parse")
        return 0

    # If the QASM has no measurements, append `measure_all()` so the
    # caller still gets a Counts back. The cross-backend fidelity
    # corpus (verify-qiskit/fixtures/) is mostly unitary-only — the
    # caller wants per-basis-state probabilities expressed as counts.
    has_meas = any(
        instr.operation.name == "measure"
        for instr in circuit.data
    )
    if not has_meas:
        circuit.measure_all(inplace=True)

    try:
        sim_kwargs = {}
        # Minimal noise wiring: depolarizing channel after each gate.
        # Operators that need the full Aer noise model can build it
        # themselves and pass via a richer noise spec in a follow-up.
        depol = noise.get("depolarizing")
        if depol:
            try:
                from qiskit_aer.noise import NoiseModel, depolarizing_error
                err1 = depolarizing_error(depol, 1)
                err2 = depolarizing_error(depol, 2)
                model = NoiseModel()
                model.add_all_qubit_quantum_error(err1, ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "y", "z", "s", "t"])
                model.add_all_qubit_quantum_error(err2, ["cx", "cz", "swap"])
                sim_kwargs["noise_model"] = model
            except Exception as e:  # noqa: BLE001
                _err(f"noise model build: {e}", kind="noise-model")
                return 0

        sim = AerSimulator(**sim_kwargs)
        run_kwargs = {"shots": shots}
        if seed is not None:
            run_kwargs["seed_simulator"] = int(seed)
        result = sim.run(circuit, **run_kwargs).result()
        raw_counts = result.get_counts()
    except Exception as e:  # noqa: BLE001
        _err(f"AerSimulator run: {e}\n{traceback.format_exc()}", kind="execute")
        return 0

    # Qiskit returns counts keyed by space-separated creg fields,
    # MSB first. Normalise to a single LSB-first bitstring so the Rust
    # side can compare against omega's own counts directly.
    counts: dict[str, int] = {}
    for key, val in raw_counts.items():
        # Drop spaces between creg fields, reverse → LSB-first.
        flat = key.replace(" ", "")
        bits = flat[::-1]
        counts[bits] = counts.get(bits, 0) + int(val)

    _emit({"ok": True, "counts": counts})
    return 0


def _qpy_to_qasm2(req: dict) -> int:
    """Decode a base64 QPY blob and emit the equivalent QASM2 source.

    Lets omega-parser take a `.qpy` file by routing the actual
    deserialisation through Qiskit's own `qiskit.qpy.load`. omega
    only sees the QASM2 string that comes back, so the version-
    compatibility surface stays Qiskit's responsibility — whatever
    QPY versions the operator's installed Qiskit reads, omega
    reads. This avoids a multi-thousand-line pure-Rust QPY parser
    for the bring-up cost.
    """
    import base64

    qpy_b64 = req.get("qpy_b64")
    if not isinstance(qpy_b64, str) or not qpy_b64:
        _err("`qpy_b64` must be a non-empty string", kind="bad-request")
        return 0
    try:
        qpy_bytes = base64.b64decode(qpy_b64, validate=True)
    except Exception as e:  # noqa: BLE001
        _err(f"qpy_b64 decode: {e}", kind="bad-request")
        return 0
    if len(qpy_bytes) < 6 or qpy_bytes[:6] != b"QISKIT":
        _err("blob is not a QPY file (magic bytes missing)", kind="qpy-parse")
        return 0

    try:
        from qiskit import qasm2, qpy
    except ImportError as e:
        _err(
            f"qiskit not importable: {e}",
            kind="qiskit-not-installed",
        )
        return 0

    import io

    try:
        circuits = qpy.load(io.BytesIO(qpy_bytes))
    except Exception as e:  # noqa: BLE001
        _err(
            f"qpy.load: {e}\n{traceback.format_exc()}",
            kind="qpy-parse",
        )
        return 0
    if not circuits:
        _err("QPY blob contained zero circuits", kind="qpy-parse")
        return 0
    if len(circuits) > 1:
        # First-cut policy: QPY can hold multiple circuits but
        # omega's pipeline takes one. Surface this rather than
        # silently truncating.
        _err(
            f"QPY blob contains {len(circuits)} circuits; omega "
            "expects exactly one. Split the file with qiskit.qpy "
            "before sending.",
            kind="qpy-multi-circuit",
        )
        return 0

    try:
        qasm2_str = qasm2.dumps(circuits[0])
    except Exception as e:  # noqa: BLE001
        _err(
            f"qasm2.dumps: {e}\n{traceback.format_exc()}",
            kind="qasm-emit",
        )
        return 0

    _emit({"ok": True, "qasm2": qasm2_str})
    return 0


def _qasm2_to_qpy(req: dict) -> int:
    """Encode a QASM2 source as a base64 QPY blob.

    Symmetric to `_qpy_to_qasm2`: `qasm2.loads` lifts the source into
    a `QuantumCircuit`, then `qpy.dump` writes it to an in-memory
    `BytesIO` and we hand the base64 of those bytes back to the Rust
    side. Lets omega-cli / omega-server hand `.qpy` blobs to
    downstream Qiskit-only tooling without operators having to reach
    for a venv themselves.
    """
    import base64
    import io

    qasm = req.get("qasm")
    if not isinstance(qasm, str) or not qasm.strip():
        _err("`qasm` must be a non-empty string", kind="bad-request")
        return 0

    try:
        from qiskit import qasm2, qpy
    except ImportError as e:
        _err(
            f"qiskit not importable: {e}",
            kind="qiskit-not-installed",
        )
        return 0

    try:
        circuit = qasm2.loads(
            qasm,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )
    except Exception as e:  # noqa: BLE001 — surface Qiskit's own error string
        _err(
            f"qasm2.loads: {e}\n{traceback.format_exc()}",
            kind="qasm-parse",
        )
        return 0

    try:
        buf = io.BytesIO()
        qpy.dump(circuit, buf)
        qpy_bytes = buf.getvalue()
    except Exception as e:  # noqa: BLE001
        _err(
            f"qpy.dump: {e}\n{traceback.format_exc()}",
            kind="qpy-emit",
        )
        return 0

    qpy_b64 = base64.b64encode(qpy_bytes).decode("ascii")
    _emit({"ok": True, "qpy_b64": qpy_b64})
    return 0


if __name__ == "__main__":
    sys.exit(main())
