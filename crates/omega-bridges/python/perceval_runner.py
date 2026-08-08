#!/usr/bin/env python3
"""omega-bridge Perceval runner.

Reads a JSON request from stdin and writes a JSON response to stdout.
See `qiskit_runner.py` for the full I/O contract.

Two source formats are supported, auto-detected from the header:

1. **QASM2** (default) — gate-based circuit lifted to Perceval via
   `perceval.converters.QiskitConverter` (dual-rail encoding). Output
   counts are LSB-first qubit bit-strings, like the Qiskit runner.

2. **OPTICQASM 1.0** (`OPENQASM` is QASM2; `OPTICQASM` is photonic) —
   native photonic source: `ps(phi)`, `bs_rx(theta, phi_tr)`,
   `bs_ry(theta, phi_tr)` over `photon q[N];`. Output counts are
   comma-separated Fock-state occupation strings (e.g. `"1,0,1,0"`).
   The request must include `"input_fock": [n0, n1, ..., nM]`.

Reference for the OPTICQASM ↔ Perceval mapping lives in the parent
project's `naquada_parser_perceval.py`; we only re-implement the
subset omega-parser actually emits today (`ps`, `bs_rx`, `bs_ry`).
"""

from __future__ import annotations

import json
import re
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

    qasm = req.get("qasm")
    shots = req.get("shots")
    if not isinstance(qasm, str) or not qasm.strip():
        _err("`qasm` must be a non-empty string", kind="bad-request")
        return 0
    if not isinstance(shots, int) or shots <= 0:
        _err("`shots` must be a positive integer", kind="bad-request")
        return 0

    # Header detection: OPTICQASM uses its own header line. Anything
    # starting with `OPENQASM` (or no header at all) goes through the
    # QASM2 path.
    header_line = next(
        (ln.strip() for ln in qasm.splitlines() if ln.strip() and not ln.lstrip().startswith("//")),
        "",
    )
    is_opticqasm = header_line.upper().startswith("OPTICQASM")

    if is_opticqasm:
        return _run_opticqasm(qasm, shots, req.get("input_fock"))
    else:
        return _run_qasm2(qasm, shots)


def _run_qasm2(qasm: str, shots: int) -> int:
    """Gate-based QASM2 path — lift via Perceval's QiskitConverter."""
    try:
        import perceval as pcvl
        from qiskit import qasm2
    except ImportError as e:
        _err(
            f"perceval-quandela / qiskit not importable: {e}",
            kind="perceval-not-installed",
        )
        return 0

    try:
        from perceval.converters import QiskitConverter
    except ImportError as e:
        _err(
            f"QiskitConverter not importable from perceval.converters: {e}. "
            "Install the matching converter add-on package; see "
            "crates/omega-bridges/python/requirements-perceval.txt.",
            kind="perceval-converter-not-installed",
        )
        return 0

    try:
        qiskit_circuit = qasm2.loads(qasm)
    except Exception as e:  # noqa: BLE001
        _err(f"qasm2.loads: {e}", kind="qasm-parse")
        return 0

    try:
        processor = QiskitConverter(pcvl.catalog).convert(
            qiskit_circuit, use_postselection=True
        )
        sampler = pcvl.algorithm.Sampler(processor)
        sample_result = sampler.samples(shots)
    except Exception as e:  # noqa: BLE001
        _err(
            f"Perceval execute: {e}\n{traceback.format_exc()}",
            kind="execute",
        )
        return 0

    counts: dict[str, int] = {}
    n_qubits = qiskit_circuit.num_qubits
    for sample in sample_result.get("results", []):
        bits = _basis_state_to_bits(sample, n_qubits)
        counts[bits] = counts.get(bits, 0) + 1

    _emit({"ok": True, "counts": counts})
    return 0


def _run_opticqasm(source: str, shots: int, input_fock) -> int:
    """Native OPTICQASM path — build a Perceval circuit directly."""
    if not isinstance(input_fock, list) or not all(isinstance(n, int) and n >= 0 for n in input_fock):
        _err(
            "OPTICQASM requires `input_fock` as a list of non-negative integers "
            "(occupation per mode)",
            kind="bad-request",
        )
        return 0

    try:
        import perceval as pcvl
        from perceval.components import PS, BS
    except ImportError as e:
        _err(
            f"perceval-quandela not importable: {e}",
            kind="perceval-not-installed",
        )
        return 0

    try:
        circuit, num_modes = _build_opticqasm_circuit(source, pcvl, PS, BS)
    except _OpticQasmParseError as e:
        _err(f"opticqasm parse: {e}", kind="opticqasm-parse")
        return 0
    except Exception as e:  # noqa: BLE001
        _err(
            f"opticqasm build: {e}\n{traceback.format_exc()}",
            kind="opticqasm-build",
        )
        return 0

    if len(input_fock) != num_modes:
        _err(
            f"input_fock length {len(input_fock)} does not match the "
            f"declared mode count {num_modes}",
            kind="bad-request",
        )
        return 0

    try:
        processor = pcvl.Processor("SLOS", circuit)
        processor.with_input(pcvl.BasicState(list(input_fock)))
        processor.min_detected_photons_filter(0)
        sampler = pcvl.algorithm.Sampler(processor)
        sample_result = sampler.sample_count(shots)
    except Exception as e:  # noqa: BLE001
        _err(
            f"Perceval OPTICQASM execute: {e}\n{traceback.format_exc()}",
            kind="execute",
        )
        return 0

    # `sample_result["results"]` is a BSCount mapping BasisState → count.
    raw_counts = sample_result.get("results", {})
    counts: dict[str, int] = {}
    for sample, n in raw_counts.items():
        key = ",".join(str(sample[i]) for i in range(num_modes))
        counts[key] = counts.get(key, 0) + int(n)

    _emit({"ok": True, "counts": counts})
    return 0


class _OpticQasmParseError(Exception):
    pass


_PHOTON_DECL = re.compile(r"^photon\s+\w+\s*\[\s*(\d+)\s*\]\s*;\s*$")
_GATE_APP = re.compile(
    r"^(?P<name>ps|bs_rx|bs_ry)\s*\((?P<params>[^)]*)\)\s*"
    r"(?P<modes>(?:\w+\s*\[\s*\d+\s*\]\s*,?\s*)+);\s*$"
)
_MODE_REF = re.compile(r"\w+\s*\[\s*(\d+)\s*\]")


def _build_opticqasm_circuit(source: str, pcvl, PS, BS):
    """Parse a minimal OPTICQASM source into a `pcvl.Circuit`.

    Supports the subset omega-parser emits today: `ps`, `bs_rx`,
    `bs_ry` with concrete numeric parameters. Symbolic parameters
    (`$theta0`) are rejected — the bridge protocol doesn't carry a
    binding map, so the operator must resolve symbols upstream.
    """
    circuit = None
    num_modes = 0

    for raw_line in source.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        if line.upper().startswith("OPTICQASM"):
            continue

        m = _PHOTON_DECL.match(line)
        if m:
            num_modes = int(m.group(1))
            circuit = pcvl.Circuit(num_modes)
            continue

        m = _GATE_APP.match(line)
        if not m:
            raise _OpticQasmParseError(f"unsupported statement: {line!r}")
        if circuit is None:
            raise _OpticQasmParseError(
                f"gate before `photon q[N]`: {line!r}"
            )

        name = m.group("name")
        param_strs = [p.strip() for p in m.group("params").split(",") if p.strip()]
        params = []
        for p in param_strs:
            if p.startswith("$"):
                raise _OpticQasmParseError(
                    f"symbolic parameter {p!r} not supported by the bridge — "
                    f"resolve before invoking"
                )
            try:
                params.append(float(p))
            except ValueError as e:
                raise _OpticQasmParseError(
                    f"non-numeric parameter {p!r} in {line!r}: {e}"
                )

        modes = [int(idx) for idx in _MODE_REF.findall(m.group("modes"))]
        if any(idx >= num_modes for idx in modes):
            raise _OpticQasmParseError(
                f"mode index out of range (declared {num_modes} modes): {line!r}"
            )

        if name == "ps":
            if len(params) != 1 or len(modes) != 1:
                raise _OpticQasmParseError(
                    f"ps takes (phi) on 1 mode; got {len(params)} params, "
                    f"{len(modes)} modes"
                )
            circuit.add((modes[0],), PS(phi=params[0]))
        elif name == "bs_rx":
            if len(params) != 2 or len(modes) != 2:
                raise _OpticQasmParseError(
                    f"bs_rx takes (theta, phi_tr) on 2 modes; got "
                    f"{len(params)} params, {len(modes)} modes"
                )
            # omega's `bs_rx(θ, φ)` matrix is `[[cos θ, -e^{iφ} sin θ],
            # [e^{-iφ} sin θ, cos θ]]` (see
            # `omega-backend-photonics::components::apply_beam_splitter_rx`).
            #
            # That is Perceval's **Ry** form, NOT its default `BS`, despite the
            # gate's name. This mapping previously used `BS(theta=2θ,
            # phi_tr=φ)`, justified by "the same transmission/reflection split"
            # and "confirmed by HOM smoke test" — i.e. checked on MAGNITUDES
            # only, by a test that is phase-insensitive. Measured error of that
            # mapping against omega's matrix: 0.798 at θ=0.6, and 1.0 at 50/50.
            #
            # The transverse phase is a CONJUGATION, not a leg phase:
            #     bs_rx(θ,φ) = PS(+φ)·BS.Ry(2θ)·PS(−φ)   on the first mode
            # Verified to 5.6e-16 across θ ∈ {0.35, 0.6, π/4, 1.2} and
            # φ ∈ {0, ±0.4, ±0.7, 1.1, −2.3}. Passing φ as `phi_tr` — with
            # either sign — is exact only at φ=0 and diverges to ~1.0 otherwise,
            # which is why this is spelled out as three components rather than
            # one parameterised call.
            #
            # Pinned at the MATRIX level by tests/test_perceval_conventions.py.
            # An amplitude-only fixture is what let the original error through.
            theta, phi_tr = params[0], params[1]
            if phi_tr != 0.0:
                circuit.add(modes[0], PS(phi=-phi_tr))
            circuit.add((modes[0], modes[1]), BS.Ry(theta=2.0 * theta))
            if phi_tr != 0.0:
                circuit.add(modes[0], PS(phi=phi_tr))
        elif name == "bs_ry":
            if len(params) != 2 or len(modes) != 2:
                raise _OpticQasmParseError(
                    f"bs_ry takes (theta, phi_tr) on 2 modes; got "
                    f"{len(params)} params, {len(modes)} modes"
                )
            circuit.add(
                (modes[0], modes[1]),
                BS.Ry(theta=2.0 * params[0], phi_tr=params[1]),
            )
        else:
            raise _OpticQasmParseError(f"unknown gate {name!r}")

    if circuit is None:
        raise _OpticQasmParseError("no `photon q[N];` declaration found")

    return circuit, num_modes


def _basis_state_to_bits(state, n_qubits: int) -> str:
    """Map a Perceval `BasisState` (dual-rail Fock pattern) to a qubit
    bitstring. Qubit i is encoded on modes (2i, 2i+1) — photon in mode
    2i+1 means qubit |1⟩. LSB-first to match omega's own counts.
    """
    pattern = list(state)
    bits = []
    for q in range(n_qubits):
        m0 = pattern[2 * q] if 2 * q < len(pattern) else 0
        m1 = pattern[2 * q + 1] if 2 * q + 1 < len(pattern) else 0
        bits.append("1" if m1 > m0 else "0")
    return "".join(bits)


if __name__ == "__main__":
    sys.exit(main())
