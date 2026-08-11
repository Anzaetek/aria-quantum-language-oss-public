#!/usr/bin/env python3
"""omega-bridge ppvm (QuEra Pauli Propagation VM) runner.

Reads a single JSON request from stdin and writes a single JSON
response to stdout. Designed to be invoked as a subprocess by the Rust
crate `omega-bridges`; the wrapper script `omega-bridge-ppvm-runner`
resolves the venv and execs this file.

**What ppvm is.** `ppvm` (github.com/QuEraComputing/ppvm, Apache-2.0)
is a Rust workspace with Python bindings offering two engines:

  1. `PauliSum` — Heisenberg-picture Pauli propagation, which computes
     *expectation values*, not shot distributions.
  2. `GeneralizedTableau` — a stabilizer tableau extended with
     non-Clifford gates (T, R_X/R_Y/R_Z, U3) and measurements, which
     *does* sample forward, including under noise and atom loss.

**Ingestion choice: `sample_stim` over `ppvm-cli`.** The counts
protocol wants a forward shot distribution, so engine (2) is the only
fit — `PauliSum.overlap_with_zero()` returns one number and could not
honour `shots` without lying about what it measured. Within engine (2),
`ppvm.sample_stim(StimProgram.parse(text), n_qubits=…, num_shots=…,
seed=…)` is a single in-process call that samples all shots in
parallel with the GIL released; the `ppvm-cli` binary would add a
second subprocess hop, a temp file, and a text-output parse for no
capability we need. The Python bindings are a first-class wheel
(`maturin` mixed Rust/Python), so there is no build-from-source cost
beyond the one-time `pip install git+…`.

ppvm's `stim-parser` accepts the same tag dialect tsim does
(`S[T]`, `I[R_Z(theta=<c>*pi)]`, `I[U3(...)]`), so both bridges share
one QASM2 lowering — see `qasm2_stim.py`. The gate sets differ:
ppvm's executor explicitly rejects SWAP/ISWAP/SQRT_XX/XC*/YC* and has
no CCX/CCZ sugar, so those QASM2 gates come back as
`ppvm-unsupported-gate` here while tsim accepts them.

**Seeding.** `sample_stim(..., seed=N)` is honoured and reproducible.

Outcomes:

- Venv missing or `ppvm` not importable → `ppvm-not-installed`
  → Rust surfaces `BridgeError::Unavailable`.
- Gate outside the supported subset → `ppvm-unsupported-gate`.
- QASM2 the converter cannot parse → `ppvm-lower`.
- Sampling failure → `ppvm-execute`.

Request shape:

  {
    "qasm":  "OPENQASM 2.0; ...",   # required
    "shots": 1024,                   # required, positive int
    "seed":  42                      # optional
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


# STDOUT IS THE WIRE — see runner_io. Imported FIRST, before any heavy
# third-party import, so nothing can print to the real stdout before the swap.

from runner_io import emit as _emit, err as _err  # noqa: E402


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
        _emit({"ok": True, "gates": sorted(GATE_SETS["ppvm"])})
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

    # ppvm's tableau *does* model depolarising / Pauli / loss channels,
    # but this bridge has no mapping from omega's opaque noise dict to
    # those instructions yet. Refuse loudly rather than return a
    # noiseless distribution to a caller who asked for noise.
    noise = req.get("noise")
    if noise:
        _err(
            f"the ppvm bridge does not map omega's noise config onto ppvm's "
            f"noise instructions yet (request carried noise={noise!r}); rerun "
            "without `noise` or use the qiskit bridge",
            kind="ppvm-noise-not-supported",
        )
        return 0

    try:
        stim_text, n_qubits, clbit_of_measurement, n_clbits = convert(
            qasm, GATE_SETS["ppvm"]
        )
    except UnsupportedGate as e:
        _err(str(e), kind="ppvm-unsupported-gate")
        return 0
    except ConversionError as e:
        _err(f"QASM2 → stim lowering failed: {e}", kind="ppvm-lower")
        return 0

    try:
        from ppvm import StimProgram, sample_stim
    except ImportError as e:
        _err(f"ppvm not importable: {e}", kind="ppvm-not-installed")
        return 0

    seed = req.get("seed")

    try:
        program = StimProgram.parse(stim_text)
    except Exception as e:  # noqa: BLE001
        _err(
            f"ppvm StimProgram.parse rejected the lowered program: {e}\n"
            f"--- lowered stim ---\n{stim_text}",
            kind="ppvm-lower",
        )
        return 0

    try:
        rows = sample_stim(
            program,
            n_qubits=n_qubits,
            num_shots=shots,
            **({"seed": int(seed)} if seed is not None else {}),
        )
    except Exception as e:  # noqa: BLE001
        _err(
            f"ppvm sample_stim: {e}\n{traceback.format_exc()}",
            kind="ppvm-execute",
        )
        return 0

    if len(rows) != shots:
        _err(
            f"ppvm returned {len(rows)} rows for {shots} shots",
            kind="ppvm-execute",
        )
        return 0
    n_records = len(clbit_of_measurement)
    if rows and len(rows[0]) != n_records:
        _err(
            f"ppvm returned {len(rows[0])} measurement records, "
            f"expected {n_records} (qubits={n_qubits})",
            kind="ppvm-execute",
        )
        return 0

    try:
        counts = bits_to_counts(rows, clbit_of_measurement, n_clbits)
    except ConversionError as e:
        _err(str(e), kind="ppvm-execute")
        return 0

    _emit({"ok": True, "counts": counts})
    return 0


if __name__ == "__main__":
    # A crash inside the runner would exit non-zero, which the Rust side
    # reads as a *transport* failure and reports as `Unavailable` — i.e.
    # "ppvm isn't installed here". That is the wrong diagnosis for a bug
    # in this file. Convert it into a structured `ppvm-internal` response
    # so the operator sees the traceback and a Backend error instead.
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        _err(f"internal runner error:\n{traceback.format_exc()}", kind="ppvm-internal")
        sys.exit(0)
