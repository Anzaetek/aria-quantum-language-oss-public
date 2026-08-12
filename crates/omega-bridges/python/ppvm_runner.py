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
    if mode == "expectation":
        return _expectation(req)
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


# ppvm PauliSum method per QASM2 gate name; angle passed as `theta=`.
#
# `t` / `tdg` map to `rz(+-pi/4)`: PauliSum has no T, and T differs from
# RZ(pi/4) only by a global phase, which an expectation value cannot see. That
# substitution would be WRONG for a statevector and is exactly right here.
#
# ppvm has NATIVE `sqrt_x` / `sqrt_x_dag` — independent confirmation that
# treating sx/sxdg as first-class Clifford gates, rather than U3 aliases, is
# the right shape.
_PS_GATES = {
    "h": ("h", 0), "x": ("x", 0), "y": ("y", 0), "z": ("z", 0),
    "s": ("s", 0), "sdg": ("s_dag", 0),
    "sx": ("sqrt_x", 0), "sxdg": ("sqrt_x_dag", 0),
    "t": ("rz", "+pi4"), "tdg": ("rz", "-pi4"),
    "rx": ("rx", 1), "ry": ("ry", 1), "rz": ("rz", 1),
    "p": ("rz", 1), "u1": ("rz", 1),
    "cx": ("cx", 0), "cz": ("cz", 0), "cy": ("cy", 0),
    "id": (None, 0), "barrier": (None, 0),
}


def _expectation(req: dict) -> int:
    """Exact/truncated expectation values via ppvm's `PauliSum`.

    Request:  {"mode":"expectation","qasm":...,"observables":[[[pauli,coeff],...],...],
               "min_abs_coeff":float?, "max_pauli_weight":int?}
    Response: {"ok":true,"values":[...]}

    ppvm's `PauliSum` is Heisenberg propagation with truncation — **the same
    algorithm family as the in-tree `omega-backend-pauliprop`**, independently
    implemented. That is the stated reason ppvm is vendored at all
    (`docs/BRIDGES.md`), and nothing could use it that way until this mode
    existed, because the counts protocol only reaches ppvm's *other* engine.

    Two conventions that fail SILENTLY if reversed:

    1. **Gates apply in REVERSE circuit order** — correct for Heisenberg
       conjugation, and wrong with no error if forwards. Measured on
       `h q0; rz(0.9) q0` with observable X: reverse gives +0.6216099683,
       matching Qiskit exactly; forward gives +1.0000000000.
    2. **Pauli strings are LSB-first** (leftmost char = qubit 0), matching
       `PauliSum.new`, Stim and our wire format — the OPPOSITE of Qiskit's
       `SparsePauliOp`. The Qiskit runner reverses on its side; this one must
       not. Verified on `x q[0]`: ppvm `"ZI"` = -1, Qiskit `"IZ"` = -1.

    `min_abs_coeff` / `max_pauli_weight` mirror
    `PauliPropBackend::with_truncation`, so the truncation behaviour is
    comparable and not merely the exact result.

    Qiskit is used ONLY to parse QASM2, never to compute. Sharing the parser
    with the anchor removes "the two sides read the circuit differently" as an
    explanation for a disagreement.
    """
    import math

    qasm = req.get("qasm")
    obs_in = req.get("observables")
    if not isinstance(qasm, str) or not qasm.strip():
        _err("`qasm` must be a non-empty string", kind="bad-request")
        return 0
    if not isinstance(obs_in, list) or not obs_in:
        _err("`observables` must be a non-empty list", kind="bad-request")
        return 0
    try:
        from ppvm import PauliSum
    except ImportError as e:
        _err(f"ppvm import failed: {e}", kind="ppvm-not-installed")
        return 0
    try:
        from qiskit import qasm2 as qk
    except ImportError as e:
        _err(f"qiskit needed to parse QASM2 for this mode: {e}",
             kind="ppvm-not-installed")
        return 0
    try:
        circ = qk.loads(qasm, include_path=qk.LEGACY_INCLUDE_PATH,
                        custom_instructions=qk.LEGACY_CUSTOM_INSTRUCTIONS,
                        custom_classical=qk.LEGACY_CUSTOM_CLASSICAL, strict=False)
    except Exception as e:  # noqa: BLE001
        _err(f"qasm2.loads: {e}", kind="qasm-parse")
        return 0

    circ.remove_final_measurements(inplace=True)
    n = circ.num_qubits
    ops = []
    for instr in circ.data:
        name = instr.operation.name
        if getattr(instr.operation, "condition", None) is not None:
            _err("expectation is undefined for a classically-conditioned gate: "
                 "the circuit is a mixture over outcomes, not one unitary",
                 kind="ppvm-not-supported")
            return 0
        if name in ("measure", "reset"):
            _err(f"mid-circuit `{name}` cannot be represented by conjugation",
                 kind="ppvm-not-supported")
            return 0
        if name not in _PS_GATES:
            _err(f"ppvm PauliSum has no mapping for gate `{name}`",
                 kind="ppvm-unsupported-gate")
            return 0
        method, arity = _PS_GATES[name]
        if method is None:
            continue
        qubits = [circ.find_bit(q).index for q in instr.qubits]
        if arity == 1:
            theta = float(instr.operation.params[0])
        elif arity == "+pi4":
            theta = math.pi / 4
        elif arity == "-pi4":
            theta = -math.pi / 4
        else:
            theta = None
        ops.append((method, qubits, theta))

    min_abs = req.get("min_abs_coeff")
    max_w = req.get("max_pauli_weight")
    values = []
    for obs in obs_in:
        try:
            total = 0.0
            for term in obs:
                pauli, coeff = term[0], float(term[1])
                if (not isinstance(pauli, str) or len(pauli) != n
                        or set(pauli) - set("IXYZ")):
                    _err(f"pauli {pauli!r} must be {n} chars over IXYZ "
                         "(dense, LSB-first)", kind="bad-request")
                    return 0
                ps = PauliSum.new(n, [pauli])
                if min_abs is not None:
                    ps.min_abs_coeff = float(min_abs)
                if max_w is not None:
                    ps.max_pauli_weight = int(max_w)
                for method, qubits, theta in reversed(ops):   # REVERSED
                    f = getattr(ps, method)
                    if theta is None:
                        f(*qubits)
                    else:
                        f(*qubits, theta=theta)
                total += coeff * float(ps.overlap_with_zero())
            values.append(total)
        except Exception as e:  # noqa: BLE001
            _err(f"ppvm PauliSum: {e}", kind="execute")
            return 0

    _emit({"ok": True, "values": values})
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
