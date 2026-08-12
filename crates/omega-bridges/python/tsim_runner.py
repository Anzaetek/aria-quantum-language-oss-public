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

import qasm2_stim  # noqa: E402
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
    if mode == "expectation":
        return _expectation(req)
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


def _expectation(req: dict) -> int:
    """Exact Clifford expectation values via **plain Stim's tableau**.

    Request:  {"mode":"expectation","qasm":...,"observables":[[[pauli,coeff],...],...]}
    Response: {"ok":true,"values":[...]}

    This is the same-algorithm anchor for the in-tree `omega-backend-pauli`
    stabilizer backend: Stim is *the* reference stabilizer simulator, so a
    disagreement is a defect by construction rather than a modelling choice.

    ## It uses Stim, NOT tsim

    tsim's ZX stabilizer-rank engine handles non-Clifford circuits; that is its
    whole point, and it is irrelevant here. What this mode wants is an EXACT
    Clifford reference, and `TableauSimulator.peek_observable_expectation`
    returns exact integers (+1 / -1 / 0) with no float error at all. It lives
    in this runner only because `bloqade-tsim` already pulls Stim into the
    venv.

    ## Two traps, both measured, both of which would give a WRONG anchor

    1. **`TableauSimulator.state_vector()` is `complex64`.** `1/sqrt(2)` comes
       back as 0.70710677, error 1.21e-08. An earlier plan proposed comparing
       state vectors at 1e-15, which would have failed every Clifford fixture
       and invited a 1e-6 fudge factor covering a misunderstanding. Expectation
       values avoid it entirely -- they are integers.

    2. **Plain Stim SILENTLY MIS-EXECUTES our tag dialect.** `qasm2_stim`
       emits `S[T] 0` and `I[R_Z(theta=0.25*pi)] 0` for non-Clifford gates.
       Stim parses the bracket as an annotation on the base gate and accepts
       both without complaint -- applying **S instead of T**, and **identity
       instead of a rotation**. It does not refuse; it returns a confident
       wrong answer.

       So this mode REFUSES any lowering that emits a `[` tag, rather than
       trusting the upstream gate-set filter. Anchoring a non-Clifford circuit
       on plain Stim would manufacture a disagreement and blame our backend.

    ## Conventions

    Pauli strings are dense and **LSB-first** (leftmost char = qubit 0),
    matching `stim.PauliString`, ppvm and our wire format -- and OPPOSITE to
    Qiskit's `SparsePauliOp`, whose runner reverses on its own side. Verified
    on `x q[0]`: stim `"ZI"` = -1, qiskit `"IZ"` = -1, both naming qubit 0.
    """
    qasm = req.get("qasm")
    obs_in = req.get("observables")
    if not isinstance(qasm, str) or not qasm.strip():
        _err("`qasm` must be a non-empty string", kind="bad-request")
        return 0
    if not isinstance(obs_in, list) or not obs_in:
        _err("`observables` must be a non-empty list", kind="bad-request")
        return 0
    try:
        import stim
    except ImportError as e:
        _err(f"stim import failed: {e}", kind="tsim-not-installed")
        return 0

    try:
        text, n_qubits, _, _ = qasm2_stim.convert(qasm, GATE_SETS["tsim"])
    except qasm2_stim.UnsupportedGate as e:
        _err(f"qasm2 -> stim: {e}", kind="tsim-unsupported-gate")
        return 0
    except Exception as e:  # noqa: BLE001
        _err(f"qasm2 -> stim: {e}", kind="qasm-parse")
        return 0

    # THE CLIFFORD GUARD. A `[` in the emitted text means a tag, which means a
    # non-Clifford gate that Stim would silently mis-execute. Checked on the
    # TEXT, not on the input gate list, because the text is what Stim consumes.
    tagged = [ln for ln in text.splitlines() if "[" in ln]
    if tagged:
        _err(
            "circuit is not Clifford: the Stim lowering emitted tagged "
            f"instruction(s) {tagged[:3]}, which plain Stim parses as "
            "annotations and executes as the BASE gate (S[T] applies S, not T). "
            "Refusing rather than returning a confidently wrong reference.",
            kind="tsim-not-supported",
        )
        return 0

    body = "\n".join(
        ln for ln in text.splitlines()
        if not ln.strip().startswith(("M ", "MZ ", "MX ", "MY ", "R ", "RZ ", "TICK", "DETECTOR"))
    )
    try:
        sim = stim.TableauSimulator()
        sim.do(stim.Circuit(body))
    except Exception as e:  # noqa: BLE001
        _err(f"stim tableau: {e}", kind="execute")
        return 0

    values = []
    for obs in obs_in:
        total = 0.0
        for term in obs:
            pauli, coeff = term[0], float(term[1])
            if (not isinstance(pauli, str) or len(pauli) != n_qubits
                    or set(pauli) - set("IXYZ")):
                _err(f"pauli {pauli!r} must be {n_qubits} chars over IXYZ "
                     "(dense, LSB-first)", kind="bad-request")
                return 0
            try:
                v = sim.peek_observable_expectation(stim.PauliString(pauli))
            except Exception as e:  # noqa: BLE001
                _err(f"peek_observable_expectation({pauli}): {e}", kind="execute")
                return 0
            total += coeff * float(v)
        values.append(total)

    _emit({"ok": True, "values": values})
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
