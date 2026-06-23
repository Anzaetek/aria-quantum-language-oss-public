#!/usr/bin/env python3
"""omega-bridge Bloqade (QuEra Aquila) runner.

Reads a single JSON request from stdin and writes a single JSON
response to stdout. Designed to be invoked as a subprocess by the
Rust crate `omega-bridges`; the wrapper script
`omega-bridge-bloqade-runner` activates the venv and execs this file.

**Gate-mode (`mode == "execute"`)** routes the QASM2 source through
`bloqade.qasm2.loads(..., returns="<creg>")` to lift it into the
kirin IR, then runs `bloqade.pyqrack.StackMemorySimulator.task(...)
.batch_run(shots=N)` for sampling. Bloqade returns
`dict[tuple[MeasurementResultValue, ...], probability]`; the
runner converts that to LSB-first bit-string counts so the Rust
side can compare against omega's other backends directly. Single
classical register supported today; multi-creg programs surface as
`bloqade-multi-creg-not-supported`.

**Analog AHS (`mode == "ahs"`)** is still scaffolded and surfaces
`bloqade-ahs-not-implemented` — the gate-mode path is the
priority for omega's QASM2-driven flows; AHS unlocks Rydberg-array
optimisation later.

Outcomes:

- Venv missing or `bloqade` not importable → `bloqade-not-installed`
  → Rust surfaces `BridgeError::Unavailable`.
- QASM2 lowering fails (parse / unsupported gate) → `bloqade-lower`.
- Simulator construction or `batch_run` fails → `bloqade-execute`.

Request shape (gate mode):

  {
    "mode":  "execute",         # optional, defaults to "execute"
    "qasm":  "OPENQASM 2.0; ...",
    "shots": 1024,
    "seed":  42,                # optional
    "creg":  "c"                # optional, defaults to "c" (or the
                                # first creg the QASM2 source declares)
  }

Response shape (success):

  {"ok": true, "counts": {"00": 512, "11": 512, ...}}    # LSB-first

Response shape (error):

  {"ok": false, "error": "<message>", "kind": "<kind>"}

The runner exits 0 in both success and structured-failure cases.
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings


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

    mode = req.get("mode") or "execute"
    if mode == "ahs":
        _err(
            "AHS analog mode not yet implemented in the omega-bloqade runner",
            kind="bloqade-ahs-not-implemented",
        )
        return 0
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
    # Cross-backend fidelity fixtures (verify-qiskit/fixtures/) are
    # mostly unitary-only — no `creg` declaration, no `measure`. The
    # Qiskit and Perceval runners both transparently add measurements
    # in that case so the caller still gets a Counts back. Match that
    # behaviour here: if the QASM2 source has no creg, inject one of
    # the qreg's size and a `measure q -> c;` line; if it has a creg
    # but no measure, append `measure q -> c;`. Bloqade's gate-mode
    # entry point requires both, so without this synthesis the runner
    # would reject every unmeasured fixture with `bloqade-lower`.
    qasm, synthesised_creg = _ensure_measurement(qasm)
    creg_name = req.get("creg") or synthesised_creg or _infer_creg_name(qasm) or "c"

    # Imports are guarded so the operator gets a clear "venv not built
    # / package missing" message rather than a generic ImportError
    # traceback if the runner is invoked outside the venv. The
    # wrapper script (`omega-bridge-bloqade-runner`) activates the
    # venv before exec-ing this file.
    try:
        # Suppress the harmless PyQrack-deprecation warning that
        # bloqade emits the first time `pyqrack` imports.
        warnings.filterwarnings("ignore", category=UserWarning, module="bloqade")
        from bloqade import qasm2
        from bloqade.pyqrack import StackMemorySimulator
    except ImportError as e:
        _err(
            f"bloqade not importable: {e}",
            kind="bloqade-not-installed",
        )
        return 0

    seed = req.get("seed")
    rng_kwargs: dict = {}
    if seed is not None:
        try:
            import numpy as np

            rng_kwargs["rng_state"] = np.random.default_rng(int(seed))
        except Exception as e:  # noqa: BLE001
            _err(
                f"failed to build seeded RNG for bloqade: {e}",
                kind="bad-request",
            )
            return 0

    try:
        program = qasm2.loads(qasm, returns=creg_name)
    except Exception as e:  # noqa: BLE001
        _err(
            f"qasm2.loads failed (creg={creg_name!r}): {e}",
            kind="bloqade-lower",
        )
        return 0

    try:
        sim = StackMemorySimulator(**rng_kwargs)
        task = sim.task(program)
        prob_dist = task.batch_run(shots=shots)
    except Exception as e:  # noqa: BLE001
        _err(
            f"bloqade simulator: {e}\n{traceback.format_exc()}",
            kind="bloqade-execute",
        )
        return 0

    counts = _prob_dist_to_counts(prob_dist, shots)
    if counts is None:
        _err(
            "bloqade batch_run returned a non-tuple key — check that "
            "the QASM2 program declares a single classical register "
            "and that `creg` matches its name",
            kind="bloqade-multi-creg-not-supported",
        )
        return 0

    _emit({"ok": True, "counts": counts})
    return 0


_CREG_PREFIX = "creg "
_QREG_PREFIX = "qreg "


def _qreg_size(qasm: str) -> tuple[str, int] | None:
    """Return ``(name, size)`` of the first ``qreg <name>[size];`` line, or
    None if no qreg is found. Single-qreg programs are the only shape
    supported here — the cross-backend fidelity fixtures all use a
    single ``qreg q[N]``.
    """
    for line in qasm.splitlines():
        stripped = line.strip()
        if stripped.startswith(_QREG_PREFIX):
            tail = stripped[len(_QREG_PREFIX) :].strip().rstrip(";")
            lb = tail.find("[")
            rb = tail.find("]")
            if lb > 0 and rb > lb:
                name = tail[:lb].strip()
                try:
                    n = int(tail[lb + 1 : rb])
                except ValueError:
                    return None
                return (name, n)
    return None


def _ensure_measurement(qasm: str) -> tuple[str, str | None]:
    """Make sure the QASM2 source has a creg + a `measure` instruction so
    the Bloqade runner's `qasm2.loads(..., returns="<creg>")` sees the
    measurement results. If the source already has a creg, returns
    ``(qasm, None)`` unchanged (just appending ``measure q -> c;`` when
    no measure line is present). If neither is present, synthesises a
    creg matching the qreg's size and appends both lines, returning
    the synthesised creg name. Returns ``(qasm, None)`` unchanged when
    the source's qreg shape can't be parsed.
    """
    has_creg = any(line.strip().startswith(_CREG_PREFIX) for line in qasm.splitlines())
    has_measure = any(
        line.strip().startswith("measure ") for line in qasm.splitlines()
    )
    if has_creg and has_measure:
        return qasm, None
    qreg = _qreg_size(qasm)
    if qreg is None:
        # Without a recognisable qreg we can't inject a sized creg; let
        # the existing error path fire and surface the parse failure.
        return qasm, None
    qreg_name, n = qreg
    extra: list[str] = []
    creg_name: str | None = None
    if not has_creg:
        creg_name = "c"
        extra.append(f"creg {creg_name}[{n}];")
    if not has_measure:
        target = creg_name or _infer_creg_name(qasm) or "c"
        extra.append(f"measure {qreg_name} -> {target};")
    if not extra:
        return qasm, None
    body = qasm if qasm.endswith("\n") else qasm + "\n"
    return body + "\n".join(extra) + "\n", creg_name


def _infer_creg_name(qasm: str) -> str | None:
    """Best-effort extraction of the first ``creg <name>[size];`` from
    a QASM2 source. Returns the name if found, else None — the caller
    falls back to the conventional default ``"c"``.
    """
    for line in qasm.splitlines():
        stripped = line.strip()
        if stripped.startswith(_CREG_PREFIX):
            tail = stripped[len(_CREG_PREFIX):].strip()
            bracket = tail.find("[")
            if bracket > 0:
                return tail[:bracket].strip()
            return tail.split()[0].rstrip(";")
    return None


def _prob_dist_to_counts(prob_dist: dict, shots: int) -> dict | None:
    """Convert bloqade's ``dict[tuple[MeasurementResultValue, ...],
    probability]`` to a flat LSB-first ``{bitstring: count}`` dict.

    The tuple comes back in classical-register declaration order
    (``(c[0], c[1], ...)``), which is already the LSB-first ordering
    the rest of the omega bridges use. Probabilities are exact
    rationals (count / shots), so multiplying by `shots` and
    rounding gives the integer counts back.

    Returns ``None`` when any key isn't a tuple — the caller signals
    ``bloqade-multi-creg-not-supported``.
    """
    counts: dict[str, int] = {}
    for key, prob in prob_dist.items():
        if not isinstance(key, tuple):
            return None
        bits = "".join(str(int(v)) for v in key)
        counts[bits] = counts.get(bits, 0) + int(round(float(prob) * shots))
    return counts


if __name__ == "__main__":
    sys.exit(main())
