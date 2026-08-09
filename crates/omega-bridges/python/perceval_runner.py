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
import math
import os
import re
import sys
import traceback


# STDOUT IS THE PROTOCOL. Only `_emit` may write to it.
#
# Perceval logs a DeprecationWarning when the converter builds a Processor
# ("Getting Processor from perceval.components is deprecated"). It landed in
# front of the JSON and the Rust side failed with `invalid JSON from runner:
# expected value at line 1 column 2`. The bridge looked broken; it was being
# talked over.
#
# Reassigning `sys.stdout` is NOT enough — measured: the warning still appeared.
# Perceval's logger writes to FILE DESCRIPTOR 1, which a Python-level rebind
# cannot intercept. So duplicate fd 1 to a private fd, point fd 1 at stderr,
# and hand the private one to `_emit`. Anything any library prints — Python or
# native — becomes operator-visible diagnostics on stderr instead of protocol
# corruption.
#
# This is a general robustness fix, not a Perceval one: every bridge shares
# this wire, and any dependency that printed would have broken any of them the
# same way. `qiskit_runner.py` has simply been lucky.
_PROTOCOL_FD = os.dup(1)
os.dup2(2, 1)
_PROTOCOL_STDOUT = os.fdopen(_PROTOCOL_FD, "w")
sys.stdout = sys.stderr


def _emit(payload: dict) -> None:
    _PROTOCOL_STDOUT.write(json.dumps(payload))
    _PROTOCOL_STDOUT.write("\n")
    _PROTOCOL_STDOUT.flush()


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

    # Perceval MOVED the converters out of the core package at 1.0: what was
    # `perceval.converters` is now the separate `perceval-interop`
    # distribution, importable as `perceval_interop`. Measured on
    # perceval-quandela 1.2.4 — `perceval.converters` raises ModuleNotFoundError
    # and the shipped submodules are algorithm/backends/components/
    # error_mitigation/providers/rendering/runtime/serialization/simulators/utils.
    #
    # Try both, new location first. Pinning to only the old path made the whole
    # Perceval arm report `Unavailable` on any current install, which the
    # cross-backend test then honestly reported as "compared nothing" — green,
    # and validating exactly zero circuits.
    QiskitConverter = None
    import_errors = []
    for module_name in ("perceval_interop", "perceval.converters"):
        try:
            QiskitConverter = __import__(
                module_name, fromlist=["QiskitConverter"]
            ).QiskitConverter
            break
        except (ImportError, AttributeError) as e:  # noqa: PERF203
            import_errors.append(f"{module_name}: {e}")
    if QiskitConverter is None:
        _err(
            "QiskitConverter not importable ("
            + "; ".join(import_errors)
            + "). Perceval >= 1.0 ships it in the separate `perceval-interop` "
            "package; see crates/omega-bridges/python/requirements-perceval.txt.",
            kind="perceval-converter-not-installed",
        )
        return 0

    try:
        # Qiskit 2.x's loader does not auto-expose the qelib1.inc gate set:
        # bare `qasm2.loads` rejects `p`, `swap`, `crz`, `sx`, ... with
        # "'p' is not defined in this scope". `qiskit_runner.py` already
        # passes the LEGACY_* hooks for exactly this reason; this runner did
        # not, so every fixture beyond the basic Clifford surface came back
        # as `qasm-parse` and never reached Perceval at all.
        qiskit_circuit = qasm2.loads(
            qasm,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )
    except Exception as e:  # noqa: BLE001
        _err(f"qasm2.loads: {e}", kind="qasm-parse")
        return 0

    # Perceval's converter accepts GATES ONLY — it asserts every instruction is
    # a `qiskit.circuit.gate.Gate` and dies with "Cannot convert instruction(s):
    # <class '_SingletonMeasure'>" otherwise. That is correct for its model: the
    # dual-rail encoding measures photons at the output, so a QASM `measure` is
    # implicit rather than an operation to translate.
    #
    # Terminal measurements are therefore removed. A MID-CIRCUIT measurement is
    # a different thing entirely — it collapses the state and cannot be pushed
    # to the end — so it is refused rather than silently dropped, which would
    # answer a different circuit.
    n_before = len(qiskit_circuit.data)
    qiskit_circuit.remove_final_measurements(inplace=True)
    leftover = [
        i.operation.name
        for i in qiskit_circuit.data
        if i.operation.name in ("measure", "reset")
        or getattr(i.operation, "condition", None) is not None
    ]
    if leftover:
        _err(
            f"circuit has mid-circuit {sorted(set(leftover))} after removing terminal "
            "measurements; Perceval's dual-rail conversion measures only at the output "
            "and cannot express it",
            kind="perceval-not-supported",
        )
        return 0
    _ = n_before

    try:
        # Perceval 1.x: `QiskitConverter(backend_name: str = "SLOS",
        # noise_model=None)`. Passing `pcvl.catalog` (the 0.x form) reaches
        # `Processor.__init__`, which asserts `isinstance(backend, ABackend)`
        # and dies with "'backend' must be an ABackend (got ...Catalog)".
        # SLOS is Perceval's default strong-simulation backend.
        processor = QiskitConverter().convert(
            qiskit_circuit, use_postselection=True
        )
        sampler = pcvl.algorithm.Sampler(processor)
        sample_result = sampler.samples(shots)
    except Exception as e:  # noqa: BLE001
        # UPSTREAM LIMITATION, not a defect on either side: perceval-interop
        # 1.2.4 collides on internal parameter names when a circuit contains
        # several parameterised gates, raising "The experiment already owns a
        # parameter named theta". Bisected on
        # `02_single_qubit_rotations.qasm`: ry, rz, rx, u2, u1, h convert
        # fine; adding the 7th op (u3) trips it.
        #
        # Report it as a REFUSAL so the cross-backend arm records
        # `cannot-express` and skips the fixture with a reason, rather than
        # counting a Perceval bug as a disagreement between our engines.
        # Matched on the exact message — a blanket "execute errors are
        # refusals" rule is the silent direction docs/BRIDGES.md warns about.
        if "already owns a parameter named" in str(e):
            _err(
                f"perceval-interop cannot convert this circuit: {e}. Upstream "
                "parameter-name collision across multiple parameterised gates "
                "(reproduced on perceval-quandela 1.2.4 / perceval-interop 1.1).",
                kind="perceval-not-supported",
            )
            return 0
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


_PHOTON_DECL = re.compile(r"^photon\s+\w+\s*\[\s*(\d+)\s*\]\s*(pol)?\s*;\s*$")
# `pbs` takes no parameters, so the parameter list is optional — matching the
# grammar in omega-parser/src/opticqasm.pest.
_GATE_APP = re.compile(
    r"^(?P<name>ps|bs_rx|bs_ry|hwp|pbs)\s*(?:\((?P<params>[^)]*)\))?\s*"
    r"(?P<modes>(?:\w+\s*\[\s*\d+\s*\]\s*,?\s*)+);\s*$"
)
_MODE_REF = re.compile(r"\w+\s*\[\s*(\d+)\s*\]")


def _add_two_mode(circuit, pcvl, comp, i, j):
    """Add a 2-mode component across modes `i` and `j`, adjacent or not.

    Perceval's `Circuit.add` asserts "Range must be a consecutive set of port
    indexes", but polarization expansion produces non-adjacent pairs by
    construction: a PBS acts on the H sub-modes `2a` and `2b`, which are two
    apart. So the pair is brought together with a PERM, the component applied,
    and the PERM undone.

    Using PERM rather than pre-computing the full mode unitary in numpy is
    deliberate. Handing Perceval a finished matrix would leave it doing only the
    SLOS/permanent step, and the cross-check would no longer test our gate
    conventions at all — it would test our matrix against our matrix. This way
    Perceval still assembles the circuit from its OWN components, so a
    convention error on our side still shows up as a disagreement.
    """
    if j < i:
        i, j = j, i
    if j == i + 1:
        circuit.add((i, j), comp)
        return

    # Swap mode j down to i+1, act, swap back. The PERM spans [i+1 .. j].
    span = j - i
    perm = list(range(span))
    perm[0], perm[span - 1] = perm[span - 1], perm[0]

    circuit.add(i + 1, pcvl.components.PERM(perm))
    circuit.add((i, i + 1), comp)
    circuit.add(i + 1, pcvl.components.PERM(perm))


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
            declared = int(m.group(1))
            # `photon q[N] pol;` -> N SPATIAL modes = 2N optical modes,
            # interleaved (s, p) -> 2s + p with p=0 meaning H.
            #
            # We expand polarization into PLAIN optical modes here rather than
            # building a polarized Perceval processor. That keeps the bridge
            # protocol unchanged: `input_fock` stays a flat list of length
            # num_modes and outputs stay flat integer tuples. A polarized
            # processor would need a polarization-aware input encoding (Perceval
            # writes |{P:H},0>), which the flat contract cannot express, and
            # would break the output key loop below.
            polarized = m.group(2) is not None
            num_modes = declared * 2 if polarized else declared
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
        raw_params = m.group("params") or ""
        param_strs = [p.strip() for p in raw_params.split(",") if p.strip()]
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
            _add_two_mode(circuit, pcvl, BS.Ry(theta=2.0 * theta), modes[0], modes[1])
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
        elif name == "hwp":
            # Half-wave plate on ONE spatial mode's (H, V) pair.
            #
            #   HWP(θ) = i · BSrx(2θ, 0) · PS(π on V)
            #
            # The global `i` is applied as PS(π/2) on BOTH sub-modes. It is not
            # droppable: the plate acts on a subset of the interferometer's
            # modes, so a global factor on that 2×2 block is a RELATIVE phase
            # between interfering paths. Aria adopts Perceval's convention
            # verbatim (FIXES_PLAN.md I1b) so this comparison needs no fudge
            # factor. Mirrors `lower_half_wave_plate` in omega-parser.
            if len(params) != 1 or len(modes) != 1:
                raise _OpticQasmParseError(
                    f"hwp takes (theta) on 1 spatial mode; got {len(params)} "
                    f"params, {len(modes)} modes"
                )
            h, v = 2 * modes[0], 2 * modes[0] + 1
            if v >= num_modes:
                raise _OpticQasmParseError(
                    f"hwp on spatial mode {modes[0]} needs a `pol` declaration"
                )
            circuit.add(v, PS(phi=math.pi))
            _add_two_mode(circuit, pcvl, BS.Ry(theta=2.0 * (2.0 * params[0])), h, v)
            circuit.add(h, PS(phi=math.pi / 2))
            circuit.add(v, PS(phi=math.pi / 2))
        elif name == "pbs":
            # Swaps H between the two spatial modes, transmits V — Perceval's
            # convention, which is the OPPOSITE of the usual textbook phrasing.
            # Swap = PS(π) · BSrx(π/2, π); the phase shifter supplies det = −1.
            if params or len(modes) != 2:
                raise _OpticQasmParseError(
                    f"pbs takes no params on 2 spatial modes; got "
                    f"{len(params)} params, {len(modes)} modes"
                )
            a_h, b_h = 2 * modes[0], 2 * modes[1]
            if max(a_h, b_h) + 1 >= num_modes:
                raise _OpticQasmParseError(
                    "pbs needs a `pol` declaration covering both spatial modes"
                )
            circuit.add(a_h, PS(phi=-math.pi))
            _add_two_mode(circuit, pcvl, BS.Ry(theta=2.0 * (math.pi / 2)), a_h, b_h)
            circuit.add(a_h, PS(phi=math.pi))
            circuit.add(b_h, PS(phi=math.pi))
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
