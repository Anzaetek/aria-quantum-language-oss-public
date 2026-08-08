#!/usr/bin/env python3
"""omega-bridge tsim (QuEra) runner.

Reads a single JSON request from stdin and writes a single JSON
response to stdout. Designed to be invoked as a subprocess by the Rust
crate `omega-bridges`; the wrapper script `omega-bridge-tsim-runner`
resolves the venv and execs this file.

**What tsim is.** `bloqade-tsim` (PyPI, Apache-2.0) is QuEra's
ZX-calculus stabilizer-rank sampler — "feels just like Stim but
supports non-Clifford gates". It consumes Stim program text plus a
small tag dialect for T / R_X,R_Y,R_Z / U3 / CCZ / CCX, compiles it to
a ZX diagram via `pyzx`, and samples measurement outcomes through JAX.
Runtime scales with the circuit's T-count, not its qubit count.

**Ingestion choice.** tsim has no QASM front end, so the runner lowers
QASM2 itself via the shared `qasm2_stim` module and hands the resulting
Stim text to `tsim.Circuit(...)`. That converter is deliberately total
over a small subset: any gate outside it comes back as
`tsim-unsupported-gate` rather than being approximated. See
`qasm2_stim.py` for the subset and the bit-ordering contract.

**Seeding.** `Circuit.compile_sampler(seed=...)` is honoured, so a
request carrying `"seed"` is reproducible. Note tsim's own caveat: the
`batch_size` it picks from available memory also affects the sample
stream, so reproducibility holds per-machine, not across machines.

Outcomes:

- Venv missing or `tsim` not importable → `tsim-not-installed`
  → Rust surfaces `BridgeError::Unavailable`.
- Gate outside the supported subset → `tsim-unsupported-gate`.
- QASM2 the converter cannot parse → `tsim-lower`.
- Sampler construction / sampling failure → `tsim-execute`.

Request shape:

  {
    "qasm":  "OPENQASM 2.0; ...",   # required
    "shots": 1024,                   # required, positive int
    "seed":  42,                     # optional
    "strategy": "cat5"               # optional, tsim decomposition strategy
  }

Response shape (success):

  {"ok": true, "counts": {"00": 512, "11": 512, ...}}    # LSB-first

Response shape (error):

  {"ok": false, "error": "<message>", "kind": "<kind>"}

The runner exits 0 in both success and structured-failure cases.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qasm2_stim import (  # noqa: E402
    GATE_SETS,
    ConversionError,
    UnsupportedGate,
    bits_to_counts,
    convert,
)


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
    if mode == "gates":
        # Introspection mode: report the QASM2 gate names this bridge
        # can lower. `tests/cross_backend.rs` asks the runner for this
        # list rather than hard-coding a copy, so the fixture filter
        # can never drift from the converter.
        _emit({"ok": True, "gates": sorted(GATE_SETS["tsim"])})
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

    # tsim consumes no noise model through this path. Refuse loudly
    # rather than silently returning a noiseless distribution the
    # caller would read as noisy.
    noise = req.get("noise")
    if noise:
        _err(
            f"the tsim bridge does not implement a noise model yet "
            f"(request carried noise={noise!r}); rerun without `noise` or "
            "use the qiskit bridge",
            kind="tsim-noise-not-supported",
        )
        return 0

    try:
        stim_text, n_qubits, clbit_of_measurement, n_clbits = convert(
            qasm, GATE_SETS["tsim"]
        )
    except UnsupportedGate as e:
        _err(str(e), kind="tsim-unsupported-gate")
        return 0
    except ConversionError as e:
        _err(f"QASM2 → stim lowering failed: {e}", kind="tsim-lower")
        return 0

    try:
        import tsim
    except ImportError as e:
        _err(f"tsim not importable: {e}", kind="tsim-not-installed")
        return 0

    seed = req.get("seed")
    strategy = req.get("strategy") or "cat5"

    try:
        circuit = tsim.Circuit(stim_text)
    except Exception as e:  # noqa: BLE001
        _err(
            f"tsim.Circuit rejected the lowered program: {e}\n"
            f"--- lowered stim ---\n{stim_text}",
            kind="tsim-lower",
        )
        return 0

    try:
        sampler = circuit.compile_sampler(
            strategy=strategy,
            **({"seed": int(seed)} if seed is not None else {}),
        )
        samples = sampler.sample(shots)
    except Exception as e:  # noqa: BLE001
        _err(
            f"tsim sampler: {e}\n{traceback.format_exc()}",
            kind="tsim-execute",
        )
        return 0

    if len(samples) != shots:
        _err(
            f"tsim returned {len(samples)} rows for {shots} shots",
            kind="tsim-execute",
        )
        return 0
    n_records = len(clbit_of_measurement)
    if samples.shape[1] != n_records:
        _err(
            f"tsim returned {samples.shape[1]} measurement records, "
            f"expected {n_records} (qubits={n_qubits})",
            kind="tsim-execute",
        )
        return 0

    try:
        counts = bits_to_counts(samples, clbit_of_measurement, n_clbits)
    except ConversionError as e:
        _err(str(e), kind="tsim-execute")
        return 0

    _emit({"ok": True, "counts": counts})
    return 0


if __name__ == "__main__":
    # A crash inside the runner would exit non-zero, which the Rust side
    # reads as a *transport* failure and reports as `Unavailable` — i.e.
    # "tsim isn't installed here". That is the wrong diagnosis for a bug
    # in this file. Convert it into a structured `tsim-internal` response
    # so the operator sees the traceback and a Backend error instead.
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        _err(f"internal runner error:\n{traceback.format_exc()}", kind="tsim-internal")
        sys.exit(0)
