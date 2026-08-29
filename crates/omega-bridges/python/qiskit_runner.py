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
    "method": "matrix_product_state",# optional, Aer simulation method
    "mps_bond_dimension": 256,       # optional, only with the MPS method
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
import os
import sys
import traceback


# STDOUT IS THE WIRE — see runner_io. Imported FIRST, before any heavy
# third-party import, so nothing can print to the real stdout before the swap.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

    # Mode dispatch. Default: QASM2 → counts. Optional:
    # "qpy_to_qasm2" — decode a base64-encoded QPY blob and emit
    # the equivalent QASM2 source so omega-parser can take over.
    # "qasm2_to_qpy" — symmetric: parse QASM2 with `qasm2.loads`
    # then emit the QPY blob via `qpy.dump` for round-trip /
    # interop with downstream Qiskit-only tooling.
    mode = req.get("mode") or "execute"
    if mode == "capabilities":
        # Capability handshake — see the note in tsim_runner.py. Answering
        # before a run is what lets a caller avoid discovering a mismatch as a
        # mid-run error, or, for noise, not discovering it at all.
        _emit(
            {
                "ok": True,
                "capabilities": {
                    "backend": "qiskit",
                    "modes": [
                        "execute",
                        "expectation",
                        "expectation-mixture",
                        "qpy_to_qasm2",
                        "qasm2_to_qpy",
                    ],
                    "noise_keys": ["depolarizing"],
                    "notes": "scalar depolarizing only; per-qubit and per-pair forms are refused",
                },
            }
        )
        return 0
    if mode == "qpy_to_qasm2":
        return _qpy_to_qasm2(req)
    if mode == "qasm2_to_qpy":
        return _qasm2_to_qpy(req)
    if mode == "expectation":
        return _expectation(req)
    if mode == "expectation-mixture":
        return _expectation_mixture(req)
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
        # A noise key this bridge cannot honour must be REFUSED, not dropped.
        # Only `depolarizing` is wired below; `amplitude_damping`,
        # `phase_damping`, `pauli` and `readout` were accepted and silently
        # ignored, so a caller asking for amplitude damping got a distribution
        # with none and no way to tell. The per-PAIR `2q` form is refused for
        # the same reason: the scalar path below applies one rate everywhere,
        # which is a DIFFERENT noise model from the one requested.
        supported = {"depolarizing"}
        unsupported = sorted(set(noise) - supported)
        if unsupported:
            _err(
                "the qiskit bridge implements only `depolarizing`; it cannot "
                f"honour {unsupported} and will not silently drop them. Remove "
                "the unsupported keys, or use an in-process backend "
                "(statevector / mps), which implements the full model.",
                kind="qiskit-noise-key-not-supported",
            )
            return 0
        depol = noise.get("depolarizing")
        if isinstance(depol, dict):
            _err(
                "the qiskit bridge implements only a SCALAR `depolarizing` "
                "rate; the per-qubit / per-pair form would be applied as one "
                "uniform rate here, which is a different noise model than the "
                "one requested. Use an in-process backend for per-pair rates.",
                kind="qiskit-noise-key-not-supported",
            )
            return 0
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

        # Simulation METHOD. Default is Aer's automatic choice (statevector for
        # small registers); `matrix_product_state` is what makes a 128-qubit
        # comparison possible at all, and is the point of the wide cross-check:
        # our MPS against theirs, rather than ours against a dense oracle that
        # cannot reach the width.
        method = req.get("method")
        if method:
            sim_kwargs["method"] = str(method)
        bond = req.get("mps_bond_dimension")
        if bond is not None:
            sim_kwargs["matrix_product_state_max_bond_dimension"] = int(bond)

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


def _expectation(req: dict) -> int:
    """`{"mode":"expectation","qasm":...,"observables":[[["ZI",0.5],...],...]}`
    -> `{"ok":true,"values":[...]}`.

    EXACT: `Statevector.from_instruction` with no shots, so the value carries
    no sampling error and the in-tree comparison is analytic (see K2 —
    "analytic vs stochastic must not share a tolerance").

    ## The observable wire format, and the trap in it

    Each observable is a list of `[pauli_string, coefficient]` terms. The
    string is **DENSE and LSB-FIRST**: `len == num_qubits`, and the LEFTMOST
    character is qubit 0.

    That is deliberately NOT Qiskit's own convention. `SparsePauliOp` is
    MSB-first — measured on `x q[0]`, `SparsePauliOp("IZ")` gives -1 and
    `("ZI")` gives +1 — so this function REVERSES every string before handing
    it to Qiskit. LSB-first was chosen because it matches the two other
    references in this comparison (Stim's `PauliString` and ppvm's
    `PauliSum` are both LSB-first, both verified on the same asymmetric
    fixture) and because `omega_bridges::Counts` is already documented
    LSB-first, so one rule covers the whole wire.

    Get this backwards and the error is silent on any palindromic observable
    — `ZZ`, `XX`, `II` — which is most of the obvious test cases. The pin is
    an asymmetric one-qubit term.

    ## Measurements

    Terminal measurements are stripped (`remove_final_measurements`);
    expectation is a property of the unitary evolution. Verified equal to a
    textual strip of the `measure`/`creg` lines across all 11 applicable
    fixtures, worst |delta| 3.253e-19.

    A circuit with a *mid-circuit* measurement, a `reset`, or a classical
    condition is REFUSED rather than answered. Qiskit's own behaviour here is
    not safe to rely on: `from_instruction` raises on a leftover measure, but
    on an entangled `reset` it silently returns ONE stochastic trajectory —
    a different answer per invocation. Refusing keeps a nondeterministic
    anchor out of the matrix.
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
        from qiskit import qasm2
        from qiskit.quantum_info import SparsePauliOp, Statevector
    except ImportError as e:
        _err(f"qiskit import failed: {e}", kind="qiskit-not-installed")
        return 0

    try:
        circuit = qasm2.loads(
            qasm,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )
    except Exception as e:  # noqa: BLE001
        _err(f"qasm2.loads: {e}", kind="qasm-parse")
        return 0

    # Refuse constructs that make "the expectation of this circuit"
    # ill-defined, BEFORE stripping anything.
    for instr in circuit.data:
        name = instr.operation.name
        if name == "reset":
            _err(
                "expectation is undefined for a circuit containing `reset`: it is a "
                "non-unitary channel, and Statevector.from_instruction silently returns "
                "one stochastic trajectory rather than refusing",
                kind="qiskit-not-supported",
            )
            return 0
        if getattr(instr.operation, "condition", None) is not None:
            _err(
                "expectation is undefined for a classically-conditioned gate: the "
                "circuit is a mixture over measurement outcomes, not one unitary",
                kind="qiskit-not-supported",
            )
            return 0

    circuit.remove_final_measurements(inplace=True)
    for instr in circuit.data:
        if instr.operation.name == "measure":
            _err(
                "mid-circuit measurement remains after removing terminal measurements; "
                "expectation is not defined for this circuit",
                kind="qiskit-not-supported",
            )
            return 0

    n = circuit.num_qubits
    try:
        state = Statevector.from_instruction(circuit)
    except Exception as e:  # noqa: BLE001
        _err(f"Statevector.from_instruction: {e}", kind="execute")
        return 0

    values = []
    for obs in obs_in:
        terms = []
        for term in obs:
            try:
                pauli, coeff = term[0], float(term[1])
            except Exception:  # noqa: BLE001
                _err(f"malformed observable term {term!r}", kind="bad-request")
                return 0
            if not isinstance(pauli, str) or len(pauli) != n or set(pauli) - set("IXYZ"):
                _err(
                    f"pauli {pauli!r} must be {n} chars over IXYZ (dense, LSB-first)",
                    kind="bad-request",
                )
                return 0
            # LSB-first on the wire -> MSB-first for Qiskit.
            terms.append((pauli[::-1], coeff))
        try:
            op = SparsePauliOp.from_list(terms)
            values.append(float(state.expectation_value(op).real))
        except Exception as e:  # noqa: BLE001
            _err(f"expectation_value: {e}", kind="execute")
            return 0

    _emit({"ok": True, "values": values})
    return 0


def _expectation_mixture(req: dict) -> int:
    """`{"mode":"expectation-mixture","qasm":...,"observables":[...]}` ->
    `{"ok":true,"values":[...]}`.

    The oracle `_expectation` refuses to be: EXACT expectation of a circuit
    that is a MIXTURE over measurement outcomes — mid-circuit measurement
    plus classically-conditioned gates. Same wire format as `_expectation`
    (dense LSB-first Pauli strings, reversed before Qiskit sees them).

    ## How, and why it is still analytic

    Branch-exact evolution over Qiskit's own linear algebra: the state is a
    list of (unnormalised `Statevector`, classical-bit values). A measurement
    projects into both outcome branches (weights carried in the norms, so
    `expectation_value` — a raw quadratic form, verified — needs no explicit
    reweighting); a conditioned gate applies only on branches whose register
    matches. `⟨O⟩ = Σ_branches ⟨ψ_b|O|ψ_b⟩`. No shots anywhere, so this lane
    keeps its flat analytic tolerance rather than needing a √N split.

    Qiskit 2.x lowers `if (c==k) g q;` to an `if_else` block instruction; the
    legacy `.condition` attribute on a plain gate is handled too. `reset` and
    a conditioned/in-block measure are refused, and the branch count is
    capped: it grows as 2^measurements and a runaway corpus should fail
    loudly rather than swap.

    ## An INERT final measurement is ELIDED, not branch-split

    Matching `omega_core::defer_measure` and `remove_final_measurements`: a
    measure whose bit no later condition reads and whose qubit nothing later
    touches does not enter the branching — expectation is of the state
    BEFORE it. Splitting instead would silently dephase every later X/Y
    observable on that qubit, which is a different (post-measurement)
    question than the one every other engine in the lane answers. Qiskit's
    own `remove_final_measurements` is NOT used here: on these circuits the
    guard-read register is shared with the final measure, and what that
    method does to a half-used register is exactly the kind of behaviour
    this runner exists to not depend on.
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
        import numpy as np
        from qiskit import qasm2
        from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
    except ImportError as e:
        _err(f"qiskit import failed: {e}", kind="qiskit-not-installed")
        return 0

    try:
        circuit = qasm2.loads(
            qasm,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )
    except Exception as e:  # noqa: BLE001
        _err(f"qasm2.loads: {e}", kind="qasm-parse")
        return 0

    n = circuit.num_qubits
    MAX_BRANCHES = 64

    def cond_of(op):
        return getattr(op, "condition", None)

    def _reg_members(reg):
        # A condition target is a ClassicalRegister (index by iteration order)
        # or a single Clbit. Registers in current Qiskit are not `__iter__`
        # iterable, only indexable, so probe by list().
        try:
            return list(reg)
        except TypeError:
            return [reg]

    # ---- inert-measure elision (see docstring) ----------------------------
    data = list(circuit.data)
    read_later_clbits = set()  # clbit indices read by a condition AFTER position i
    used_later_qubits = set()
    keep = [True] * len(data)
    for i in range(len(data) - 1, -1, -1):
        instr = data[i]
        op = instr.operation
        # A CONDITIONED measure (legacy .condition, unreachable from qasm2
        # today) must never be elided here — it would silently drop the
        # conditional collapse the forward pass exists to refuse.
        if op.name == "measure" and cond_of(op) is None:
            cidx = circuit.find_bit(instr.clbits[0]).index
            qidx = circuit.find_bit(instr.qubits[0]).index
            if cidx not in read_later_clbits and qidx not in used_later_qubits:
                keep[i] = False
                continue
        cond = cond_of(op)
        if cond is not None:
            for b in _reg_members(cond[0]):
                read_later_clbits.add(circuit.find_bit(b).index)
        if op.name != "barrier":
            for q in instr.qubits:
                used_later_qubits.add(circuit.find_bit(q).index)

    # ---- branch evolution -------------------------------------------------
    def creg_value(bits, reg):
        return sum(
            bits.get(circuit.find_bit(b).index, 0) << k
            for k, b in enumerate(_reg_members(reg))
        )

    branches = [(Statevector.from_int(0, 2**n), {})]
    for i, instr in enumerate(data):
        if not keep[i]:
            continue
        op = instr.operation
        name = op.name
        qargs = [circuit.find_bit(q).index for q in instr.qubits]
        if name == "barrier":
            continue
        if name == "reset":
            _err(
                "expectation is undefined for a circuit containing `reset` "
                "(non-unitary channel outside this mode's mixture model)",
                kind="qiskit-not-supported",
            )
            return 0
        if name == "measure":
            if cond_of(op) is not None:
                _err("a conditioned measure is refused", kind="qiskit-not-supported")
                return 0
            cidx = circuit.find_bit(instr.clbits[0]).index
            axis = n - 1 - qargs[0]
            new = []
            for sv, bits in branches:
                arr = np.asarray(sv.data).reshape([2] * n)
                for outcome in (0, 1):
                    sel = np.zeros_like(arr)
                    idx = [slice(None)] * n
                    idx[axis] = outcome
                    sel[tuple(idx)] = arr[tuple(idx)]
                    if float(np.vdot(sel, sel).real) > 1e-30:
                        new.append(
                            (Statevector(sel.reshape(-1)), {**bits, cidx: outcome})
                        )
            branches = new
            if len(branches) > MAX_BRANCHES:
                _err(
                    f"more than {MAX_BRANCHES} measurement branches",
                    kind="qiskit-not-supported",
                )
                return 0
            continue

        cond = cond_of(op)
        if name == "if_else":
            reg, val = cond
            body = op.blocks[0]
            if len(op.blocks) > 1 and op.blocks[1] is not None and op.blocks[1].data:
                _err("an else-branch is refused", kind="qiskit-not-supported")
                return 0
            for binstr in body.data:
                if binstr.operation.name in ("measure", "reset"):
                    _err(
                        "a measure/reset inside a conditioned block is refused",
                        kind="qiskit-not-supported",
                    )
                    return 0
            new = []
            for sv, bits in branches:
                if creg_value(bits, reg) == val:
                    for binstr in body.data:
                        bq = [qargs[body.find_bit(q).index] for q in binstr.qubits]
                        try:
                            sv = sv.evolve(Operator(binstr.operation), bq)
                        except Exception as e:  # noqa: BLE001
                            _err(
                                f"cannot express {binstr.operation.name}: {e}",
                                kind="qiskit-not-supported",
                            )
                            return 0
                new.append((sv, bits))
            branches = new
            continue

        try:
            u = Operator(op)
        except Exception as e:  # noqa: BLE001
            _err(f"cannot express {name}: {e}", kind="qiskit-not-supported")
            return 0
        if cond is not None:
            reg, val = cond
            branches = [
                (sv.evolve(u, qargs) if creg_value(bits, reg) == val else sv, bits)
                for sv, bits in branches
            ]
        else:
            branches = [(sv.evolve(u, qargs), bits) for sv, bits in branches]

    total = sum(float(np.vdot(sv.data, sv.data).real) for sv, _ in branches)
    if abs(total - 1.0) > 1e-9:
        _err(
            f"mixture branches carry total probability {total!r}, not 1 — the "
            "evolution itself is wrong, refusing to answer",
            kind="execute",
        )
        return 0

    values = []
    for obs in obs_in:
        terms = []
        for term in obs:
            try:
                pauli, coeff = term[0], float(term[1])
            except Exception:  # noqa: BLE001
                _err(f"malformed observable term {term!r}", kind="bad-request")
                return 0
            if not isinstance(pauli, str) or len(pauli) != n or set(pauli) - set("IXYZ"):
                _err(
                    f"pauli {pauli!r} must be {n} chars over IXYZ (dense, LSB-first)",
                    kind="bad-request",
                )
                return 0
            # LSB-first on the wire -> MSB-first for Qiskit.
            terms.append((pauli[::-1], coeff))
        try:
            op = SparsePauliOp.from_list(terms)
            values.append(
                sum(float(sv.expectation_value(op).real) for sv, _ in branches)
            )
        except Exception as e:  # noqa: BLE001
            _err(f"expectation_value: {e}", kind="execute")
            return 0

    _emit({"ok": True, "values": values})
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
